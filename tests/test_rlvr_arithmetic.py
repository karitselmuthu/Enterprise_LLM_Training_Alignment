import unittest

from shared.training.rlvr_arithmetic import cases, verify


class RLVRTests(unittest.TestCase):
    def test_verifier_accepts_only_exact_sum(self):
        self.assertEqual(verify(2, 3, 5), 1)
        self.assertEqual(verify(2, 3, 4), 0)
        self.assertEqual(verify(2, 3, 6), 0)

    def test_split_does_not_overlap(self):
        split = cases()
        self.assertFalse(set(split["train"]) & set(split["test"]))
        self.assertEqual(len(split["train"]) + len(split["test"]), 25)

    def test_boolean_is_not_a_numeric_answer(self):
        with self.assertRaises(ValueError):
            verify(0, 1, True)


if __name__ == "__main__":
    unittest.main()
