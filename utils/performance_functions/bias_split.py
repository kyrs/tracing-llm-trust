import collections
import random
from typing import Dict, List, Tuple


BiasRecord = Dict[str, str]
BiasBlockKey = Tuple[str, str]


def split_bias_records(
    records: List[BiasRecord],
    test_fraction: float = 0.2,
    seed: int = 42,
) -> Tuple[List[BiasRecord], List[BiasRecord]]:
    """Split records without separating reused chosen or rejected answers.

    Each target-group/category block contains the full Cartesian product of its
    chosen and rejected templates. Keeping the block intact prevents either
    scored sequence from appearing in both splits.
    """
    if not 0 <= test_fraction <= 1:
        raise ValueError("`test_fraction` must be between 0 and 1")

    block_keys_by_group = collections.defaultdict(list)
    for record in records:
        block_key = (record["target_group"], record["bias_category"])
        if block_key not in block_keys_by_group[record["target_group"]]:
            block_keys_by_group[record["target_group"]].append(block_key)

    rng = random.Random(seed)
    test_block_keys = set()
    for target_group in sorted(block_keys_by_group):
        block_keys = sorted(block_keys_by_group[target_group])
        rng.shuffle(block_keys)
        n_test = int(len(block_keys) * test_fraction)
        test_block_keys.update(block_keys[:n_test])

    train_records = []
    test_records = []
    for record in records:
        block_key = (record["target_group"], record["bias_category"])
        destination = test_records if block_key in test_block_keys else train_records
        destination.append(record)

    return train_records, test_records
