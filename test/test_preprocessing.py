import unittest

from src.preprocessing import _remove_nonvoice_segments


class TestNonvoiceSegments(unittest.TestCase):
    def test_remove_nonvoice_segments_1(self):
        sig = [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0]
        sig_short = _remove_nonvoice_segments(
            sig, min_length=3, min_value_rel=0.5
        )
        self.assertEqual(len(sig_short), 7)

    def test_remove_nonvoice_segments_2(self):
        sig = [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0]
        sig_short = _remove_nonvoice_segments(
            sig, min_length=2, min_value_rel=0.5
        )
        self.assertEqual(len(sig_short), 5)

    def test_remove_nonvoice_segments_3(self):
        sig = [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0]
        sig_short = _remove_nonvoice_segments(
            sig, min_length=4, min_value_rel=0.5
        )
        self.assertEqual(len(sig_short), 10)


if __name__ == "__main__":
    unittest.main()
