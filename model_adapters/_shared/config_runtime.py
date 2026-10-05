"""Configuration and local model discovery used by each model directory."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch


VARIANTS = ["metaphor", "literal", "related", "unrelated"]
VARIANT_KEYS = {
    "metaphor": "metaphor",
    "literal": "literal",
    "related": "related_source_metaphor",
    "unrelated": "unrelated_source_metaphor",
}


@dataclass(frozen=True)
class ModelSpec:
    key: str
    name: str
    model_ids: tuple[str, ...]
    local_names: tuple[str, ...]
    env_var: str
    layer_paths: tuple[str, ...]
    mlp_names: tuple[str, ...]
    activation_names: tuple[str, ...]
    projection_names: tuple[str, ...]
    fallback_activation: str
    download_repo: Optional[str] = None


@dataclass
class Config:
    model_key: str
    model_name: str
    lang: str
    model_path: Path
    data_file: Path
    results_dir: Path
    device: str
    dtype_name: str
    n_samples: Optional[int]
    max_seq_len: int
    model_spec: ModelSpec

    @property
    def torch_dtype(self):
        return {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[self.dtype_name]


def is_gemma(model_key: str) -> bool:
    return model_key.lower().startswith("gemma")


def _valid_model_dir(path: Path) -> Optional[Path]:
    path = path.expanduser()
    if path.is_dir() and (path / "config.json").is_file():
        return path.resolve()
    return None


def _snapshot_dirs(cache_dir: Path):
    snapshots = cache_dir / "snapshots"
    if not snapshots.is_dir():
        return []
    return sorted(
        (p for p in snapshots.iterdir() if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def _repo_parts(model_id: str) -> tuple[str, str]:
    org, _, name = model_id.partition("/")
    return org, name or org


def _candidate_paths(spec: ModelSpec, models_root: Optional[str | Path]):
    roots: list[Path] = []
    if models_root:
        roots.append(Path(models_root).expanduser())
    env_root = os.environ.get("MODEL_ROOT", "").strip()
    if env_root:
        roots.append(Path(env_root).expanduser())

    # This is the model directory used by the Linux project environment. It is
    # harmless on other machines and makes the same scripts portable there.
    roots.append(Path(__file__).resolve().parents[2] / "models")

    for root in roots:
        for local_name in spec.local_names:
            yield root / local_name

    cache_homes = [Path.home()]
    # When this project is copied between Windows and Linux, MODEL_CACHE_HOME
    # can point at the machine's cache without changing the source code.
    cache_home = os.environ.get("MODEL_CACHE_HOME", "").strip()
    if cache_home:
        cache_homes.insert(0, Path(cache_home).expanduser())
    for cache_root in cache_homes:
        modelscope = cache_root / ".cache" / "modelscope" / "hub"
        huggingface = cache_root / ".cache" / "huggingface" / "hub"
        for model_id in spec.model_ids:
            org, name = _repo_parts(model_id)
            for candidate in (
                modelscope / "models" / org / name,
                modelscope / org / name,
                modelscope / "hub" / "models" / org / name,
            ):
                yield candidate
                # ModelScope nests snapshots under a revision subdirectory
                # (e.g. "master"), so surface those directly as well.
                if candidate.is_dir():
                    for child in sorted(
                        candidate.iterdir(),
                        key=lambda p: p.stat().st_mtime,
                        reverse=True,
                    ):
                        if child.is_dir():
                            yield child

            hf_cache = huggingface / f"models--{org}--{name}"
            for snapshot in _snapshot_dirs(hf_cache):
                yield snapshot


def resolve_model_path(
    spec: ModelSpec,
    explicit: Optional[str | Path] = None,
    models_root: Optional[str | Path] = None,
) -> Path:
    explicit_value = str(explicit).strip() if explicit is not None else ""
    if explicit_value:
        path = _valid_model_dir(Path(explicit_value))
        if path is None:
            raise FileNotFoundError(
                f"Invalid {spec.key} model path: {explicit_value!r}. "
                "The directory must contain config.json."
            )
        return path

    env_value = os.environ.get(spec.env_var, "").strip()
    if env_value:
        path = _valid_model_dir(Path(env_value))
        if path is None:
            raise FileNotFoundError(
                f"{spec.env_var} is set but does not contain a model directory "
                f"with config.json: {env_value}"
            )
        return path

    seen: set[str] = set()
    for candidate in _candidate_paths(spec, models_root):
        key = os.path.normcase(os.path.abspath(os.fspath(candidate.expanduser())))
        if key in seen:
            continue
        seen.add(key)
        path = _valid_model_dir(candidate)
        if path is not None:
            return path

    ids = ", ".join(spec.model_ids)
    if spec.download_repo:
        try:
            from modelscope import snapshot_download
        except ImportError as exc:
            raise RuntimeError(
                f"{spec.name} is not cached locally and modelscope is required "
                f"to download {spec.download_repo}. Install with: pip install modelscope"
            ) from exc
        print(
            f"  {spec.name} not found locally; downloading {spec.download_repo} "
            f"from ModelScope ...",
            flush=True,
        )
        downloaded_value = snapshot_download(spec.download_repo)
        if downloaded_value is None:
            raise FileNotFoundError(
                f"ModelScope returned no local path for {spec.download_repo}"
            )
        downloaded = Path(downloaded_value)
        found = _valid_model_dir(downloaded)
        if found is None and downloaded.is_dir():
            for child in sorted(
                downloaded.iterdir(),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            ):
                found = _valid_model_dir(child)
                if found is not None:
                    break
        if found is None:
            raise FileNotFoundError(
                f"Downloaded {spec.download_repo} but could not locate a model "
                f"directory with config.json under {downloaded}"
            )
        print(f"  Downloaded to {found}")
        return found
    raise FileNotFoundError(
        f"Cannot find {spec.name}. Checked ModelScope/Hugging Face caches and "
        f"known model roots. Set {spec.env_var}=<model_dir> or pass "
        f"--model-path <model_dir>. Repository IDs: {ids}"
    )


def _config_value(model_config, names: tuple[str, ...], default=None):
    for config in (model_config, getattr(model_config, "text_config", None)):
        if config is None:
            continue
        for name in names:
            value = config.get(name) if isinstance(config, dict) else getattr(config, name, None)
            if value is not None:
                return value
    return default


def read_architecture(model_path: Path) -> dict:
    with (model_path / "config.json").open("r", encoding="utf-8-sig") as handle:
        raw = json.load(handle)
    text = raw.get("text_config") if isinstance(raw.get("text_config"), dict) else {}
    merged = {**raw, **text}
    return {
        "model_type": merged.get("model_type", "unknown"),
        "num_layers": merged.get("num_hidden_layers", merged.get("num_layers")),
        "hidden_size": merged.get("hidden_size"),
        "intermediate_size": merged.get(
            "intermediate_size", merged.get("intermediate_dim", merged.get("moe_intermediate_size"))
        ),
    }


def make_get_config(spec: ModelSpec, module_file: str):
    project_root = Path(module_file).resolve().parents[1]
    data_dir = project_root.parent / "data"
    default_result_root = project_root.parent / "outputs"

    def get_config(
        lang: str = "cn",
        n_samples: Optional[int] = None,
        model_path: Optional[str | Path] = None,
        models_root: Optional[str | Path] = None,
        result_root: Optional[str | Path] = None,
        device: Optional[str] = None,
        dtype: Optional[str] = None,
        max_seq_len: int = 256,
    ) -> Config:
        if lang not in {"cn", "en"}:
            raise ValueError(f"Unknown language {lang!r}; choose cn or en")
        if n_samples is not None and n_samples < 1:
            raise ValueError("n_samples must be positive when provided")
        if max_seq_len < 1:
            raise ValueError("max_seq_len must be positive")

        requested_device = (device or "auto").lower()
        if requested_device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")
        if requested_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
        actual_device = "cuda" if requested_device == "cuda" or (
            requested_device == "auto" and torch.cuda.is_available()
        ) else "cpu"

        dtype_name = dtype or ("bfloat16" if actual_device == "cuda" else "float32")
        if dtype_name not in {"bfloat16", "float16", "float32"}:
            raise ValueError("dtype must be bfloat16, float16, or float32")

        resolved_model = resolve_model_path(spec, model_path, models_root)
        output_root = Path(result_root).expanduser() if result_root else default_result_root
        results_dir = output_root / f"{spec.key}_{lang}"
        results_dir.mkdir(parents=True, exist_ok=True)
        data_file = data_dir / ("chinese_samples.json" if lang == "cn" else "english_samples.json")
        if not data_file.is_file():
            raise FileNotFoundError(f"Missing data file: {data_file}")

        return Config(
            model_key=spec.key,
            model_name=spec.name,
            lang=lang,
            model_path=resolved_model,
            data_file=data_file,
            results_dir=results_dir,
            device=actual_device,
            dtype_name=dtype_name,
            n_samples=n_samples,
            max_seq_len=max_seq_len,
            model_spec=spec,
        )

    return get_config


def model_dimensions(model, layers) -> tuple[int, int, int]:
    config = model.config
    hidden = _config_value(config, ("hidden_size", "d_model"))
    intermediate = _config_value(
        config,
        ("intermediate_size", "intermediate_dim", "moe_intermediate_size"),
    )
    num_layers = _config_value(config, ("num_hidden_layers", "num_layers"))
    if hidden is None:
        hidden = getattr(layers[0], "hidden_size", None)
    if intermediate is None:
        mlp = next(
            (
                getattr(layers[0], name)
                for name in ("mlp", "feed_forward", "ffn")
                if hasattr(layers[0], name)
            ),
            None,
        )
        intermediate = getattr(mlp, "intermediate_size", None)
    if hidden is None or intermediate is None:
        raise RuntimeError(
            "Unable to infer hidden/intermediate dimensions from the loaded model"
        )
    return int(num_layers or len(layers)), int(hidden), int(intermediate)
