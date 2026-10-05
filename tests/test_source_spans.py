from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "causal"))
import unittest
from prepare_source_spans import extract_details


class DiffTests(unittest.TestCase):
    def test_whitespace_difference_is_not_a_source(self):
        self.check('An arichpart is ramalama and dingdong.',
                   'An a rich part is peanut butter and jelly.',
                   'An a rich part is a stapler and moonlight.',
                   [('ramalama', 'peanut butter', 'a stapler'),
                    ('dingdong', 'jelly', 'moonlight')])

    def test_whitespace_only_sentence_has_no_source(self):
        groups, status, _ = extract_details(dict(metaphor='An arichpart.',
            related_source_metaphor='An a rich part.', unrelated_source_metaphor='An a rich part.'))
        self.assertIsNone(groups)
        self.assertEqual(status, 'no_scorable_replacement')

    def check(self, m, r, u, expected):
        entry = dict(metaphor=m, related_source_metaphor=r, unrelated_source_metaphor=u)
        groups, status, ignored = extract_details(entry)
        self.assertEqual(status, "ok")
        parts = [p for g in groups for p in g["parts"]]
        self.assertEqual([(p["text"], p["related_source"], p["unrelated_source"]) for p in parts], expected)
        for p in parts:
            span = p["source_span"]
            self.assertEqual(m[span["start"]:span["end"]], p["text"])
        entry.update(source_domain="intentionally wrong", unrelated_source_domain="also wrong")
        self.assertEqual(extract_details(entry), (groups, status, ignored))

    def test_quote_does_not_discard_valid_replacement(self):
        self.check('"Microfinance defeats the monster of poverty',
                   'Microfinance defeats the disease of poverty',
                   'Microfinance defeats the green sofa of poverty',
                   [('monster', 'disease', 'green sofa')])

    def test_interior_anchor_collision(self):
        self.check('阿喀琉斯是一头狮子。', '阿喀琉斯是一头猛虎。', '阿喀琉斯是一块石头。',
                   [('头狮子', '头猛虎', '块石头')])

    def test_unchanged_negative_is_local(self):
        self.check('青春像太阳从正中变成夕阳。', '青春像花朵从盛开变成凋零。',
                   '青春像书本从正中变成夕阳。', [('太阳', '花朵', '书本')])

    def test_user_example(self):
        self.check('没有人是一座孤岛。', '没有人是一颗孤星。', '没有人是一块圆面包。',
                   [('座孤岛', '颗孤星', '块圆面包')])


if __name__ == '__main__':
    unittest.main()
