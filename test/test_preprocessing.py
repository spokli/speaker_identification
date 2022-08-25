import unittest

import pandas as pd

from src.preprocessing import (
    _remove_duplicate_labels,
    _remove_nonvoice_segments,
)


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


class TestRemoveDuplicateLabels(unittest.TestCase):
    def test_remove_duplicate_labels_1(self):
        df = pd.DataFrame(
            [
                ["file1", "20220101", "102030", "1"],
                ["file1", "20220303", "102030", "3"],
                ["file1", "20220202", "102030", "2"],
            ],
            columns=["filename", "label_date", "label_time", "label"],
        )
        df_removed_duplicates = _remove_duplicate_labels(df)
        self.assertListEqual(
            df_removed_duplicates.values.tolist(),
            [["file1", "20220303", "102030", "3"]],
        )
        self.assertEqual(len(df_removed_duplicates), 1)

    def test_remove_duplicate_labels_2(self):
        df = pd.DataFrame(
            [
                ["file1", "20220101", "102030", "1"],
                ["file2", "20220303", "102030", "3"],
                ["file1", "20220303", "102030", "3"],
                ["file1", "20220202", "102030", "2"],
                ["file2", "20220101", "102030", "1"],
                ["file2", "20220202", "102030", "2"],
                ["file3", "20220101", "102030", "1"],
            ],
            columns=["filename", "label_date", "label_time", "label"],
        )
        df_removed_duplicates = _remove_duplicate_labels(df)
        self.assertListEqual(
            df_removed_duplicates.values.tolist(),
            [
                ["file1", "20220303", "102030", "3"],
                ["file2", "20220303", "102030", "3"],
                ["file3", "20220101", "102030", "1"],
            ],
        )
        self.assertEqual(len(df_removed_duplicates), 3)


if __name__ == "__main__":
    unittest.main()
