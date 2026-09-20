import math
import random
import unittest

from s2r_adaptation.agreement import (disagreement, levenshtein, levenshtein_python,
                                    normalize_for_scoring, reliability_weight)


class AgreementTests(unittest.TestCase):
    def test_nfc_categories_and_combining_marks(self):
        self.assertEqual(normalize_for_scoring(" A e\u0301\tကိ၊\u200b1+$"), "Aéကိ1")
        self.assertEqual(disagreement("e\u0301", "é"), 0)
        self.assertGreater(disagreement("A1", "a၁"), 0)

    def test_primary_denominator_and_unclipped_disagreement(self):
        self.assertEqual(disagreement("ab", "abcccc"), 2)
        self.assertEqual(disagreement("abcd", "abc"), .25)
        self.assertEqual(disagreement("abc", "abcd"), 1/3)

    def test_empty_secondary_and_primary(self):
        self.assertEqual(disagreement("ကခ", ""), 1)
        self.assertEqual(disagreement("ကခ", "!?"), 1)
        with self.assertRaises(ValueError): disagreement("!? \t", "abc")
        with self.assertRaises(TypeError): disagreement("abc", None)

    def test_cubic_floor_after_power(self):
        self.assertEqual(reliability_weight(0), 1)
        self.assertAlmostEqual(reliability_weight(.2), .512)
        self.assertAlmostEqual(reliability_weight(.5), .125)
        self.assertEqual(reliability_weight(.6), .1)
        self.assertEqual(reliability_weight(3), .1)
        self.assertEqual(reliability_weight(.5, gamma=1), .5)
        self.assertEqual(reliability_weight(1, w_min=0), 0)

    def test_invalid_numeric_inputs(self):
        for d in [-1, float("inf"), float("nan")]:
            with self.assertRaises(ValueError): reliability_weight(d)
        for gamma in [0, -1, float("inf"), float("nan")]:
            with self.assertRaises(ValueError): reliability_weight(.5, gamma=gamma)
        for floor in [-.1, 1.1, float("nan")]:
            with self.assertRaises(ValueError): reliability_weight(.5, w_min=floor)

    def test_edit_distance_known_and_accelerated_equivalence(self):
        self.assertEqual(levenshtein_python("kitten", "sitting"), 3)
        self.assertEqual(levenshtein_python("", "abc"), 3)
        rng = random.Random(42)
        for _ in range(100):
            a = "".join(rng.choices("abကိé", k=rng.randrange(20)))
            b = "".join(rng.choices("abကိé", k=rng.randrange(20)))
            self.assertEqual(levenshtein_python(a, b), levenshtein(a, b))


if __name__ == "__main__": unittest.main()
