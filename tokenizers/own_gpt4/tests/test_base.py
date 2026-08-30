from __future__ import annotations

import unittest

from tokenizers.own_gpt4.base import get_stats, merge


class PairHelperTests(unittest.TestCase):
    """验证 BPE 最小算法单元，便于快速定位训练或编码阶段的问题。"""

    def test_get_stats_counts_every_adjacent_pair(self) -> None:
        self.assertEqual(
            get_stats([1, 2, 1, 2, 1]),
            {(1, 2): 2, (2, 1): 2},
        )

    def test_get_stats_accumulates_into_the_supplied_mapping(self) -> None:
        counts = {(9, 9): 3, (1, 2): 4}

        returned = get_stats([1, 2, 3], counts)

        self.assertIs(returned, counts)
        self.assertEqual(counts, {(9, 9): 3, (1, 2): 5, (2, 3): 1})

    def test_get_stats_handles_short_sequences(self) -> None:
        self.assertEqual(get_stats([]), {})
        self.assertEqual(get_stats([42]), {})

    def test_merge_is_non_overlapping_and_left_to_right(self) -> None:
        # 三个相同 token 只有最左边的一对会在这一轮合并。
        self.assertEqual(merge([1, 1, 1], (1, 1), 256), [256, 1])

    def test_merge_replaces_all_disjoint_occurrences(self) -> None:
        self.assertEqual(
            merge([1, 2, 1, 2, 3, 1, 2], (1, 2), 256),
            [256, 256, 3, 256],
        )

    def test_merge_leaves_unmatched_input_unchanged(self) -> None:
        ids = [1, 2, 3]

        result = merge(ids, (7, 8), 256)

        self.assertEqual(result, ids)
        self.assertIsNot(result, ids)


if __name__ == "__main__":
    unittest.main()
