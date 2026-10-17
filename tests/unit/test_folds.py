from collections import Counter

from cv.folds import make_folds

ITEMS = [(f"{stratum}/{i:03d}.png", stratum)
         for stratum, n in [("good", 20), ("broken_large", 20), ("broken_small", 22), ("contamination", 21)]
         for i in range(n)]


def test_every_item_gets_exactly_one_fold():
    folds = make_folds(ITEMS, k=5, seed=0)
    assert set(folds) == {item_id for item_id, _ in ITEMS}
    assert set(folds.values()) == {0, 1, 2, 3, 4}


def test_deterministic_for_same_seed_and_independent_of_input_order():
    a = make_folds(ITEMS, k=5, seed=0)
    b = make_folds(list(reversed(ITEMS)), k=5, seed=0)
    assert a == b


def test_different_seed_changes_assignment():
    assert make_folds(ITEMS, k=5, seed=0) != make_folds(ITEMS, k=5, seed=1)


def test_each_stratum_is_spread_evenly_across_folds():
    folds = make_folds(ITEMS, k=5, seed=0)
    for stratum in {s for _, s in ITEMS}:
        per_fold = Counter(folds[i] for i, s in ITEMS if s == stratum)
        assert max(per_fold.values()) - min(per_fold.values()) <= 1


def test_fold_sizes_are_balanced_overall():
    sizes = Counter(make_folds(ITEMS, k=5, seed=0).values())
    assert max(sizes.values()) - min(sizes.values()) <= 1
