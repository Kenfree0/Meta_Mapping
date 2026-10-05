from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "causal"))
"""4.2 protocol regression tests, without real model weights."""
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import analyze_causality as e


class Tokenizer:
    bos_token_id = 1

    def encode(self, text, add_special_tokens=False):
        return [ord(c) + 2 for c in text]


def record(text="Life is a river; a river flows.", source="river", wrong="chair"):
    return dict(metaphor=text, source_domain=source, unrelated_source_domain=wrong)


class ProtocolTests(unittest.TestCase):
    def test_old_diff_data_ignores_whitespace_group(self):
        item = self.diff_record('An arichpart is fire.', [(3, 'arichpart', 'a rich part'), (16, 'fire', 'desk')])
        item['source_groups'][0]['parts'][0]['related_source'] = 'a rich part'
        result = e.mask_record(item)
        self.assertEqual(result['masked_metaphor'], 'An arichpart is [BLANK_1].')
        self.assertEqual(len(result['ignored_formatting_differences']), 1)

    def test_candidate_collision_reports_text_and_ids(self):
        class CollidingTokenizer:
            def encode(self, text, add_special_tokens=False):
                return [7]
        with self.assertRaisesRegex(ValueError, "correct='river'.*unrelated='chair'.*correct_ids=\\[7\\]"):
            e.prepare_masked(CollidingTokenizer(), e.mask_record(record()))

    def diff_record(self, text, parts):
        return dict(metaphor=text, source_extraction={"method": "M_R_U_aligned_token_diff", "status": "ok"},
                    source_groups=[{"parts": [{"text": surface, "unrelated_source": negative,
                        "source_span": {"start": start, "end": start+len(surface), "text": surface}}]}
                        for start, surface, negative in parts])

    def test_diff_never_reads_legacy_labels(self):
        item = self.diff_record("Life is a river.", [(10, "river", "desk")])
        first = e.mask_record(item)
        item.update(source_domain=None, unrelated_source_domain=None,
                    source_span={"start": -1, "text": "obsolete", "source_domain": "wrong"})
        second = e.mask_record(item)
        self.assertEqual(first["prediction_groups"], second["prediction_groups"])
        self.assertEqual(first["masked_metaphor"], "Life is a [BLANK_1].")

    def test_diff_repeats_are_one_nonempty_group(self):
        item = self.diff_record("A river and river.", [(2, "river", "desk"), (12, "river", "desk")])
        result = e.mask_record(item)
        self.assertEqual(len(result["prediction_groups"]), 1)
        self.assertEqual(len(result["prediction_groups"][0]["occurrences"]), 2)

    def test_diff_nested_surfaces_are_already_hidden(self):
        item = self.diff_record("They fight fire with fire.", [(5, "fight fire", "bake cake"), (21, "fire", "cake")])
        result = e.mask_record(item)
        self.assertEqual(result["masked_metaphor"], "They [BLANK_1] with [BLANK_2].")
        self.assertEqual(len(result["prediction_groups"]), 2)

    def test_preview_diff_dataset_can_prepare_and_score(self):
        for lang, expected in (("cn", 10), ("en", 10)):
            path = Path(__file__).resolve().parents[1] / "data" / ("chinese_samples.json" if lang == "cn" else "english_samples.json")
            if not path.exists():
                self.skipTest("Local dataset unavailable")
            data = json.loads(path.read_text(encoding="utf-8-sig"))
            self.assertEqual(len(data), expected)
            with patch.object(e, "RAW_SCORE", return_value={"S": 1., "logP_source": -1., "logP_unrelated_source": -2.}):
                for row, entry in enumerate(data):
                    with self.subTest(lang=lang, row=row):
                        prepared = e.mask_record(entry)
                        self.assertTrue(all(g["occurrences"] for g in prepared["prediction_groups"]))
                        contexts, correct, wrong, prompts = e.prepare_masked(Tokenizer(), prepared, lang=lang)
                        result = e.score_sentence(None, contexts, correct, wrong)
                        self.assertAlmostEqual(result["q"], e.relative_probability(-1., -2.))

    def test_all_repetitions_and_word_boundaries(self):
        result = e.mask_record(record("Art is art, not a cart.", "art"))
        self.assertEqual(result["masked_metaphor"], "[BLANK_1] is [BLANK_2], not a cart.")
        self.assertEqual(len(result["prediction_groups"][0]["occurrences"]), 2)

    def test_legacy_span_expands_repetitions(self):
        item = record()
        item["source_span"] = dict(start=10, end=15, text="river")
        self.assertNotIn("river", e.mask_record(item)["masked_metaphor"])

    def test_ambiguous_multiple_parts_require_mapping(self):
        item = record("Life is fire and ice.", "fire/ice")
        item["source_span"] = {"spans": [dict(start=8, end=12, text="fire"),
                                           dict(start=17, end=20, text="ice")]}
        with self.assertRaisesRegex(ValueError, "parallel_label_or_sentence_missing"):
            e.mask_record(item)
        annotation = {"source_groups": [{"parts": [
            {"text": "fire", "unrelated_source": "chair"},
            {"text": "ice", "unrelated_source": "desk"}]}]}
        result = e.mask_record(item, annotation)
        self.assertEqual(result["masked_metaphor"], "Life is [BLANK_1] and [BLANK_2].")
        annotation["source_groups"][0]["parts"].pop()
        with self.assertRaisesRegex(ValueError, "omit_annotated"):
            e.mask_record(item, annotation)

    def test_independent_prompts_keep_every_blank(self):
        item = e.mask_record(record())
        contexts, corr, wrong, prompts = e.prepare_masked(Tokenizer(), item)
        self.assertEqual(len(prompts[0]), 2)
        for prompt in prompts[0]:
            self.assertIn(item["masked_metaphor"], prompt)
            self.assertNotIn("river", prompt)
        self.assertNotEqual(prompts[0][0], prompts[0][1])
        self.assertEqual(corr[0][0], corr[0][1])

    def test_parallel_conjunctions_and_audit(self):
        item = record("狂风如刀，以大地为砧板。", "刀/砧板", "书和茶杯")
        item.update(related_source_domain="锤与铁砧",
                    related_source_metaphor="狂风如锤，以大地为铁砧。",
                    unrelated_source_metaphor="狂风如书，以大地为茶杯。")
        result = e.mask_record(item)
        self.assertEqual(result["alignment_method"], "grounded_parallel_label_split")
        self.assertEqual([g["occurrences"][0]["unrelated_source"]
                          for g in result["prediction_groups"]], ["书", "茶杯"])
        item["unrelated_source_domain"] = "书和砧板"
        item["unrelated_source_metaphor"] = "狂风如书，以大地为砧板。"
        with self.assertRaisesRegex(ValueError, "negative_unchanged"):
            e.mask_record(item)

    def test_parallel_grounding_and_order(self):
        item = record("Ice follows fire.", "fire/ice", "desk and chair")
        item.update(related_source_domain="sun and snow",
                    related_source_metaphor="Snow follows sun.",
                    unrelated_source_metaphor="Chair follows desk.")
        self.assertEqual(len(e.mask_record(item)["prediction_groups"]), 2)
        item["unrelated_source_metaphor"] = "Desk follows chair."
        with self.assertRaisesRegex(ValueError, "order_mismatch"):
            e.mask_record(item)
        item["unrelated_source_metaphor"] = "Chair follows pencil."
        with self.assertRaisesRegex(ValueError, "not_grounded"):
            e.mask_record(item)

    def test_conjunction_inside_word_and_single_slash_expression(self):
        self.assertEqual(e.split_parallel_label("和平/茶杯", "和平与茶杯", 2), ["和平", "茶杯"])
        item = record("It is copy/paste.", "copy/paste", "rain")
        self.assertEqual(e.mask_record(item)["alignment_method"], "single_surface")
        item = record("Life is cold.", "cold thing/object", "chair")
        item["source_span"] = {"start": 8, "end": 12, "text": "cold"}
        self.assertEqual(e.mask_record(item)["alignment_method"], "single_surface")

    def test_probability_and_hierarchical_averaging(self):
        # Group 1: part 0 occurs twice (q=.2,.4), part 1 once (.9).
        # Group 2: one part (.8). Result = ((.3+.9)/2+.8)/2 = .7.
        contexts = [[{"ids": [1], "part_index": 0}, {"ids": [2], "part_index": 0},
                     {"ids": [3], "part_index": 1}], [{"ids": [4], "part_index": 0}]]
        tokens = [[[1], [1], [1]], [[1]]]
        probabilities = [.2, .4, .9, .8]
        def fake_score(model, context, correct, wrong):
            q = probabilities[context[0]-1]
            lp, ln = math.log(q), math.log(1-q)
            return dict(S=lp-ln, logP_source=lp, logP_unrelated_source=ln)
        with patch.object(e, "RAW_SCORE", fake_score):
            result = e.score_sentence(None, contexts, tokens, tokens)
        self.assertAlmostEqual(result["q"], .7)
        self.assertAlmostEqual(result["q_length_normalized"], .7)
        self.assertEqual(e.relative_probability(-10000, 0), 0)
        self.assertEqual(e.relative_probability(0, -10000), 1)

    def test_overlap_stale_annotation_and_bad_candidates(self):
        with self.assertRaisesRegex(ValueError, "different_metaphor"):
            e.mask_record(record(), {"metaphor": "stale"})
        with self.assertRaisesRegex(ValueError, "identical_candidates"):
            e.mask_record(record(wrong="river"))
        duplicate = e.mask_record(record(), {"source_groups": [{"parts": [
            {"text": "river", "unrelated_source": "chair"},
            {"text": "river", "unrelated_source": "desk"}]}]})
        self.assertEqual(sum(len(g["occurrences"]) for g in duplicate["prediction_groups"]), 2)

    def test_data_preparation_blocks_partial_and_uses_new_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            data.mkdir(parents=True)
            (data / "english_samples.json").write_text(json.dumps([record(), None]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Full-data"):
                e.prepare_data(root, root / "out", "en", {}, False)
            _, prepared, count = e.prepare_data(root, root / "out", "en", {}, True)
            self.assertEqual((len(prepared), count), (1, 2))
            pending = json.loads((root / "out/prepared/en_pending.json").read_text(encoding="utf-8"))
            self.assertEqual(pending["pending"][0]["reason"], "record_not_object")

    def test_worker_output_and_restoration(self):
        item = e.mask_record(record())
        item["original_row"] = 9
        original_score = e.core.score
        original_prepare = e.core.prepare_sample
        def fake_run(args, root, key, lang):
            self.assertIsNot(e.core.score, original_score)
            baseline = {"q": .8, "q_length_normalized": .7}
            selected = {"q": .6, "q_length_normalized": .5}
            random = {"q": .75, "q_length_normalized": .65}
            row = dict(row=0, baseline=baseline, selected=selected,
                       random_trials=[random]*3, dS_selected=-.2,
                       dS_random=-.05, dS_selected_minus_random=-.15)
            e.core.dump(args.out / key / lang / "intervention_results.json",
                        dict(n=1, samples=[row], summary={}))
            self.assertIn("dS_selected", row)  # saving must not mutate core rows
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(out=root, percents=[10], model_paths=None,
                                   device="cpu", dtype=None)
            with patch.object(e.core, "run_one", fake_run):
                e.worker(args, root, "dummy", "en", root / "en.json", [item], 2)
            result = json.loads((root / "top_0.1/dummy/en/causal_results.json").read_text(encoding="utf-8"))
            self.assertEqual(result["evaluation_mode"], e.PROTOCOL)
            self.assertEqual(result["original_dataset_coverage"], .5)
            self.assertFalse(result["is_full_original_dataset"])
            self.assertEqual(result["samples"][0]["original_row"], 9)
            self.assertAlmostEqual(result["summary"]["dq_selected"]["mean"], -.2)
            self.assertAlmostEqual(result["summary"]["baseline_q"]["mean"], .8)
        self.assertIs(e.core.score, original_score)
        self.assertIs(e.core.prepare_sample, original_prepare)

    def test_progress_reports_positions_without_changing_scores(self):
        item = e.mask_record(record())
        contexts, correct, wrong, _ = e.prepare_masked(Tokenizer(), item)
        updates = []
        with patch.object(e, "RAW_SCORE", return_value={"S": 1., "logP_source": -1., "logP_unrelated_source": -2.}):
            result = e.score_sentence(None, contexts, correct, wrong,
                                      progress=lambda n, total: updates.append((n, total)))
        self.assertEqual(updates, [(1, 2), (2, 2)])
        self.assertAlmostEqual(result["q"], e.relative_probability(-1., -2.))

    def test_worker_progress_conditions_and_cleanup(self):
        item = e.mask_record(record())
        item["original_row"] = 0
        original_score = e.core.score
        def fake_run(args, root, key, lang):
            context, corr, wrong, _ = e.core.prepare_sample(Tokenizer(), item, lang=lang)
            for _ in range(5):
                e.core.score(None, context, corr, wrong)
            raise RuntimeError("simulated forward error")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(out=root, percents=[10], model_paths=None, device="cpu", dtype=None)
            with patch.object(e, "EvaluationProgress") as progress, patch.object(e.core, "run_one", fake_run), patch.object(
                    e, "RAW_SCORE", return_value={"S": 1., "logP_source": -1., "logP_unrelated_source": -2.}):
                progress.return_value.done = 0
                with self.assertRaisesRegex(RuntimeError, "simulated"):
                    e.worker(args, root, "dummy", "en", root / "en.json", [item], 1)
                statuses = [call.args[0] for call in progress.return_value.stage.call_args_list]
                for expected in ("tokenizing", "baseline", "selected ablation", "random seed=42", "random seed=43", "random seed=44"):
                    self.assertTrue(any(expected in status for status in statuses), expected)
                progress.return_value.close.assert_called_once()
        self.assertIs(e.core.score, original_score)

    def completed_fixture(self, root, pct=10):
        item = e.mask_record(record())
        item["original_row"] = 0
        row = dict(row=0, original_row=0, masked_metaphor=item["masked_metaphor"],
                   prediction_groups=item["prediction_groups"], baseline={"q": .8}, selected={"q": .7},
                   random_trials=[{"q": .8} for _ in range(3)])
        result = dict(complete=True, evaluation_mode=e.PROTOCOL, model="dummy", lang="en",
                      top_fraction=pct/100, selection_version=e.core.SELECTION_VERSION,
                      random_seeds=[42, 43, 44], seed=42, n_requested=0,
                      n=1, dataset_n=1, original_dataset_n=1, samples=[row],
                      summary={name: {"mean": 0, "ci95": [0, 0]} for name in
                               ("dq_selected", "dq_random", "dq_selected_minus_random")})
        path = root / f"top_{pct/100:g}/dummy/en/causal_results.json"
        e.core.dump(path, result)
        return item, path, result

    def test_resume_existing_complete_result_without_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            item, _, _ = self.completed_fixture(root)
            args = SimpleNamespace(out=root, percents=[10], model_paths=None, device="cpu", dtype=None)
            with patch.object(e.core, "run_one") as run, patch.object(e.core, "load_runtime") as load:
                e.worker(args, root, "dummy", "en", root / "data.json", [item], 1)
                run.assert_not_called()
                load.assert_not_called()

    def test_resume_only_missing_percentages(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            item, _, _ = self.completed_fixture(root, pct=2)
            args = SimpleNamespace(out=root, percents=[1, 2, 5], model_paths=None, device="cpu", dtype=None)
            with patch.object(e.core, "run_one") as run, patch.object(e, "EvaluationProgress"):
                e.worker(args, root, "dummy", "en", root / "data.json", [item], 1)
            self.assertEqual([call.args[0].top_fraction for call in run.call_args_list], [.01, .05])

    def test_resume_rejects_incomplete_corrupt_and_changed_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            item, path, result = self.completed_fixture(root)
            check = lambda: e.completed_result(path, "dummy", "en", 10, [item], 1)
            result["complete"] = False
            e.core.dump(path, result)
            self.assertFalse(check())
            path.write_text('{"complete":', encoding="utf-8")
            self.assertFalse(check())
            result["complete"] = True
            result["samples"] = []
            e.core.dump(path, result)
            self.assertFalse(check())
            item, path, result = self.completed_fixture(root)
            result["samples"][0]["masked_metaphor"] = "changed"
            e.core.dump(path, result)
            with self.assertRaisesRegex(ValueError, "different data"):
                check()


if __name__ == "__main__":
    unittest.main()
