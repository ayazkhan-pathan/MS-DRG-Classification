import pandas as pd

from drg.synthetic import LABELS, MARKERS, make_notes, write_splits


def test_same_seed_same_notes():
    assert make_notes(20, seed=3).equals(make_notes(20, seed=3))
    assert not make_notes(20, seed=3).equals(make_notes(20, seed=4))


def test_notes_are_plainly_fake_and_balanced():
    notes = make_notes(40, seed=0, first_record=12)
    assert notes["text"][0].startswith("synthetic record 12: part a:")
    assert notes["label"].value_counts().to_dict() == {label: 10 for label in LABELS}


def test_each_note_carries_markers_of_its_label():
    notes = make_notes(40, seed=1)
    for text, label in zip(notes["text"], notes["label"]):
        words = set(text.replace(".", " ").replace(":", " ").split())
        assert len(words & set(MARKERS[label])) >= 1


def test_splits_have_distinct_records(tmp_path):
    paths = write_splits(tmp_path, n_train=10, n_val=4, n_test=4)
    frames = {name: pd.read_csv(path) for name, path in paths.items()}
    assert {name: len(f) for name, f in frames.items()} == {"train": 10, "val": 4, "test": 4}
    ids = pd.concat(frames.values())["note_id"]
    assert ids.is_unique
