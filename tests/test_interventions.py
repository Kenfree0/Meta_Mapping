from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "causal"))
"""Regression checks for the new 4.1 experiment; never load real model weights."""
from pathlib import Path
import tempfile
import unittest
import subprocess
import sys
import json

import numpy as np
import interventions as e

try:
    import torch
except ImportError:
    torch = None


class CharTokenizer:
    bos_token_id = 1

    def encode(self, text, add_special_tokens=False):
        return [ord(c) + 2 for c in text]


class SelectionTests(unittest.TestCase):
    def test_full_prompt_covers_all_local_dataset_entries(self):
        for lang, expected in (("cn", 10), ("en", 10)):
            path = Path(__file__).resolve().parents[1] / 'data' / ('chinese_samples.json' if lang == 'cn' else 'english_samples.json')
            if not path.is_file():
                self.skipTest('Local datasets unavailable')
            data = json.loads(path.read_text(encoding='utf-8-sig'))
            self.assertEqual(len(data), expected)
            for row in data:
                context, correct, wrong, prompt = e.prepare_sample(CharTokenizer(), row, 'full_prompt', lang)
                self.assertIn(row['metaphor'].strip(), prompt)
                self.assertTrue(context and correct and wrong)

    def test_paper_conditions_without_M(self):
        # Index 0: R>U but U<L. Index 1: R>U but R<L.
        # Index 2: R=U. Index 3: U=L. Only indices 4/5 satisfy all conditions.
        r = np.array([[10., -1., 3., 4., 5., 6.]])
        u = np.array([[-1., -2., 3., 0., 2., 1.]])
        l = np.zeros_like(r)
        mapping, counts = e.select_ru(r, u, l, 1.0)
        self.assertEqual(counts, [2])
        self.assertEqual(mapping[0].tolist(), [5, 4])
        self.assertEqual(e.select_ru(r, u, l)[0][0].tolist(), [5])

    def test_npy_preferred_over_corrupt_npz(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            array = np.ones((2, 3, 4), dtype=np.float32)
            np.save(folder / 'unrelated_mlp_act.npy', array)
            (folder / 'unrelated_mlp_act.npz').write_bytes(b'broken zip')
            path = e.feature_file(folder, 'unrelated')
            self.assertEqual(path.suffix, '.npy')
            means, shape = e.layer_means(path)
            self.assertEqual(shape, array.shape)
            np.testing.assert_array_equal(means, array.mean(1))

    def test_shared_result_directory_and_ambiguity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            for key in ('Gemma12b', 'deepseek_base'):
                for lang in ('cn', 'en'):
                    folder = root / 'outputs' / f'{key}_{lang}'
                    folder.mkdir(parents=True)
                    for variant in ('related', 'unrelated', 'literal'):
                        np.save(folder / f'{variant}_mlp_act.npy', np.ones((1, 2, 3)))
                    self.assertEqual(e.locate(root, key, lang)[0], folder)
            duplicate = root / 'outputs/Gemma12b-duplicate_cn'
            duplicate.mkdir(parents=True)
            for variant in ('related', 'unrelated', 'literal'):
                np.save(duplicate / f'{variant}_mlp_act.npy', np.ones((1, 2, 3)))
            with self.assertRaisesRegex(FileNotFoundError, 'Ambiguous complete folders'):
                e.locate(root, 'Gemma12b', 'cn')

    def test_selection_and_ties(self):
        r = np.array([[0, 4, 4, 2, -1], [0, 0, 0, 0, 0]], dtype=float)
        u = np.zeros_like(r)
        mapping, counts = e.select_ru(r, u, np.full_like(r, -200), 1.0)
        self.assertEqual(counts, [3, 0])
        self.assertEqual(mapping[0].tolist(), [1, 2, 3])
        self.assertNotIn(1, mapping)
        self.assertEqual(e.select_ru(r, u, np.full_like(r, -200), .1)[0][0].tolist(), [1])
        self.assertEqual(e.select_ru(r, u, np.full_like(r, -200), .7)[0][0].tolist(), [1, 2])

    def test_default_top_ten_percent_r_activation_not_difference(self):
        r = np.arange(1, 26, dtype=float)[None, :]
        u = r - 0.5
        u[0, 0] = -100  # Largest R-U, but smallest R: must not be selected.
        mapping, counts = e.select_ru(r, u, np.full_like(r, -200))
        self.assertEqual(counts, [25])
        self.assertEqual(mapping[0].tolist(), [24, 23])  # floor(25 * .1)

    def test_ru_only_discovery_and_streaming(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = np.arange(60, dtype=np.float32).reshape(2, 6, 5)
            for key, (_, pattern, _) in e.SPECS.items():
                for lang in ("cn", "en"):
                    folder = root / pattern.format(lang=lang).replace('*', '-test-')
                    folder.mkdir(parents=True)
                    np.savez_compressed(folder / 'related_mlp_act.npz', data=a)
                    np.save(folder / 'unrelated_test_mlp_act.npy', a)
                    np.save(folder / 'literal_mlp_act.npy', a - 1)
                    found, rf, uf = e.locate(root, key, lang)
                    self.assertEqual(found, folder)
                    for path in (rf, uf):
                        means, shape = e.layer_means(path)
                        np.testing.assert_array_equal(means, a.mean(1))
                        self.assertEqual(shape, a.shape)
                    with self.assertRaises(ValueError):
                        e.feature_file(folder, 'metaphor')
            self.assertEqual(e.project_root(root / 'outputs'), root.resolve())

    def test_random_layer_counts(self):
        mapping = {0: np.array([0, 1, 2, 3]), 2: np.array([1])}
        a, b = e.random_map(mapping, 5, 42), e.random_map(mapping, 5, 42)
        for layer in mapping:
            np.testing.assert_array_equal(a[layer], b[layer])
            self.assertEqual(len(np.unique(a[layer])), len(mapping[layer]))

    def test_context_never_contains_source(self):
        entry = dict(metaphor='Life is a journey.', source_domain='journey', unrelated_source_domain='table')
        context, corr, wrong, prefix = e.prepare_sample(CharTokenizer(), entry)
        self.assertEqual(prefix, 'Life is a ')
        self.assertEqual(context, [1] + CharTokenizer().encode(prefix))
        self.assertEqual(corr, CharTokenizer().encode('journey'))
        self.assertEqual(wrong, CharTokenizer().encode('table'))
        for text in ('journey is life', 'journey and journey', 'Life is something'):
            with self.assertRaises(ValueError):
                e.prepare_sample(CharTokenizer(), dict(entry, metaphor=text))

    def test_bad_values_rejected(self):
        for fraction in (0, 1.1):
            with self.assertRaises(ValueError):
                e.select_ru(np.ones((1, 2)), np.zeros((1, 2)), np.full((1, 2), -1), fraction)
        with self.assertRaises(ValueError):
            e.select_ru(np.array([[np.nan]]), np.zeros((1, 1)), np.full((1, 1), -1))

    def test_selection_needs_no_M_or_torch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / e.SPECS['qwen2.5-7b'][1].format(lang='cn').replace('*', '_')
            folder.mkdir(parents=True)
            np.save(folder / 'related_mlp_act.npy', np.ones((2, 3, 4)))
            np.save(folder / 'unrelated_mlp_act.npy', np.zeros((2, 3, 4)))
            np.save(folder / 'literal_mlp_act.npy', np.full((2, 3, 4), -1.0))
            # Neither M activations nor any evaluation dataset exists.
            from types import SimpleNamespace
            args = SimpleNamespace(out=root / 'intervention_outputs/top_0.1',
                top_fraction=0.1, random_seeds=[42, 43, 44], select_only=True,
                evaluation_mode='full_prompt')
            e.run_one(args, root, 'qwen2.5-7b', 'cn')
            result = json.loads((args.out / 'qwen2.5-7b/cn/selection.json').read_text())
            self.assertEqual(result['selected_count'], 2)
            self.assertEqual(result['top_fraction'], 0.1)
            self.assertEqual(result['ranking'], 'mu_R descending; neuron index ascending for ties')
            self.assertEqual(set(result['activation_inputs']), {'R', 'U', 'L'})


@unittest.skipIf(torch is None, 'PyTorch is not installed')
class InterventionTests(unittest.TestCase):
    def test_hook_zeroes_and_cleans_up_on_error(self):
        projection = torch.nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            projection.weight.copy_(torch.eye(4))
        value = torch.ones(1, 3, 4)
        with self.assertRaisesRegex(RuntimeError, 'intentional'):
            with e.ablate([projection], {0: np.array([1, 3])}):
                result = projection(value)
                self.assertTrue(torch.equal(result[..., [1, 3]], torch.zeros(1, 3, 2)))
                self.assertTrue(torch.equal(result[..., [0, 2]], torch.ones(1, 3, 2)))
                raise RuntimeError('intentional')
        self.assertTrue(torch.equal(projection(value), value))
        self.assertEqual(len(projection._forward_pre_hooks), 0)

    def test_sequence_scoring_alignment(self):
        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(5, 5)

            def get_input_embeddings(self):
                return self.embedding

            def forward(self, input_ids, use_cache=False):
                logits = torch.nn.functional.one_hot((input_ids + 1) % 5, 5).float() * 3
                return type('Output', (), {'logits': logits})()

        model = Toy()
        actual = e.score(model, [0, 1], [2, 3], [4, 0])
        # Both correct next tokens get logit 3; first wrong token gets 0.
        self.assertAlmostEqual(actual['S'], 3.0, places=5)


if __name__ == '__main__':
    unittest.main()
