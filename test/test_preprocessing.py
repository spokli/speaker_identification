import unittest

from src.preprocessing import _remove_nonvoice_segments


class TestNonvoiceSegments(unittest.TestCase):
    def test_list_int(self):
        """ """
        sig = [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0]
        sig_short = _remove_nonvoice_segments(sig, min_length=3, min_value=0.5)
        self.assertEqual(len(sig_short), 7)


if __name__ == "__main__":
    unittest.main()
