import unittest

from utils.performance_functions.bias_split import split_bias_records


def build_records():
    records = []
    for group_index in range(24):
        target_group = f"group-{group_index}"
        prompt = f"prompt for {target_group}:"
        for category_index in range(16):
            bias_category = f"category-{category_index}"
            for rejected_index in range(3):
                for chosen_index in range(3):
                    records.append({
                        "prompt": prompt,
                        "chosen_answer": (
                            f" chosen-{target_group}-{bias_category}-{chosen_index}"
                        ),
                        "rejected_answer": (
                            f" rejected-{target_group}-{bias_category}-{rejected_index}"
                        ),
                        "bias_category": bias_category,
                        "target_group": target_group,
                    })
    return records


class BiasSplitTest(unittest.TestCase):
    def test_scored_sequences_are_disjoint(self):
        records = build_records()

        train_records, test_records = split_bias_records(records)

        self.assertEqual(len(records), len(train_records) + len(test_records))
        self.assertEqual(2808, len(train_records))
        self.assertEqual(648, len(test_records))

        train_chosen = {
            record["prompt"] + record["chosen_answer"] for record in train_records
        }
        test_chosen = {
            record["prompt"] + record["chosen_answer"] for record in test_records
        }
        train_rejected = {
            record["prompt"] + record["rejected_answer"] for record in train_records
        }
        test_rejected = {
            record["prompt"] + record["rejected_answer"] for record in test_records
        }

        self.assertTrue(train_chosen.isdisjoint(test_chosen))
        self.assertTrue(train_rejected.isdisjoint(test_rejected))

    def test_split_retains_group_and_category_coverage(self):
        train_records, test_records = split_bias_records(build_records())

        expected_groups = {f"group-{index}" for index in range(24)}
        expected_categories = {f"category-{index}" for index in range(16)}
        for records in (train_records, test_records):
            self.assertEqual(
                expected_groups,
                {record["target_group"] for record in records},
            )
            self.assertEqual(
                expected_categories,
                {record["bias_category"] for record in records},
            )

    def test_split_is_deterministic(self):
        records = build_records()

        first = split_bias_records(records)
        second = split_bias_records(records)

        self.assertEqual(first, second)

    def test_rejects_invalid_fraction(self):
        with self.assertRaises(ValueError):
            split_bias_records([], test_fraction=1.1)


if __name__ == "__main__":
    unittest.main()
