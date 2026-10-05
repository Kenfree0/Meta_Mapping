"""Run extraction, activation analysis and geometry for one model/language."""
from pathlib import Path
import argparse
import importlib
import importlib.util
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
MODELS = ['Gemma12b', 'qwen2.5-7b', 'llama-3.1-8b', 'deepseek_base', 'glm4-9b-0414']

def load_script(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--lang', choices=['cn', 'en'], default='cn')
    parser.add_argument('--stage', choices=['extract', 'activation', 'geometry', 'all'], default='all')
    parser.add_argument('--k', type=int, default=3, help='PCA dimension; 3 is appropriate for 10-row samples')
    args = parser.parse_args()
    if args.k < 1:
        parser.error('--k must be positive')
    sys.path.insert(0, str(ROOT / "causal"))
    import interventions as core
    if args.stage in ['extract', 'all']:
        adapter = core.SPECS[args.model][2]
        if adapter:
            sys.path.insert(0, str(ROOT / 'model_adapters'))
            sys.path.insert(0, str(ROOT / 'model_adapters' / adapter))
            config = importlib.import_module('config')
            model = importlib.import_module('model')
            from _shared.extract_runner import extract
            cfg = config.get_config(lang=args.lang, models_root=ROOT / 'models')
            extract(cfg, model)
        elif args.model == 'glm4-9b-0414':
            sys.path.insert(0, str(ROOT / 'model_adapters/glm'))
            config = importlib.import_module('config')
            api = importlib.import_module('utils')
            cfg = config.get_config(args.model, args.lang)
            api.extract_acts(cfg)
            api.extract_layer_out(cfg)
        else:
            sys.path.insert(0, str(ROOT / 'extraction/qwen_llama'))
            config = importlib.import_module('config')
            cfg = config.get_config(args.model, args.lang)
            api = importlib.import_module('extract_all')
            api.extract_all(cfg)
    folder, _, _ = core.locate(ROOT, args.model, args.lang)
    cfg = SimpleNamespace(model_key=args.model, lang=args.lang, results_dir=folder)
    if args.stage in ['activation', 'all']:
        load_script('activation_analysis', ROOT / 'activation/analyze_mapping_neurons.py').run(cfg)
    if args.stage in ['geometry', 'all']:
        load_script('geometry_analysis', ROOT / 'geometry/analyze_subspaces.py').run(cfg, k=args.k)

if __name__ == '__main__':
    main()
