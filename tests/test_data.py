import pandas as pd
import pytest
from transformers import AutoTokenizer

from drg.data import LabelMap, NoteDataset, TextCollator, read_notes, subsample
from drg.synthetic import MARKERS, make_notes


@pytest.fixture
def notes_csv(tmp_path):
    path = tmp_path / "notes.csv"
    pd.DataFrame(
        {
            "hadm": ["a", "b", "c"],
            "note": ["first note\nwith a second line", "second note", "third note"],
            "drg_code": ["003", " 871", "003"],
        }
    ).to_csv(path, index=False)
    return path


def test_read_notes_renames_columns_and_keeps_codes_as_strings(notes_csv):
    notes = read_notes(notes_csv, text_column="note", label_column="drg_code", id_column="hadm")
    assert list(notes.columns) == ["text", "label", "id"]
    assert notes["label"].tolist() == ["003", "871", "003"]
    assert notes["text"][0] == "first note\nwith a second line"


def test_read_notes_without_labels(notes_csv):
    notes = read_notes(notes_csv, text_column="note", label_column=None)
    assert list(notes.columns) == ["text"]


def test_missing_column_names_the_columns_present(notes_csv):
    with pytest.raises(ValueError, match="missing column.*label.*the file has hadm, note"):
        read_notes(notes_csv, text_column="note")


def test_empty_text_is_an_error(tmp_path):
    path = tmp_path / "empty.csv"
    pd.DataFrame({"text": ["ok", "  "], "label": ["1", "2"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="empty value"):
        read_notes(path)


def test_label_map_is_sorted_and_round_trips(tmp_path):
    label_map = LabelMap.from_labels(["871", "003", "S02", "003"])
    assert label_map.labels == ("003", "871", "S02")
    assert label_map.id2label == {0: "003", 1: "871", 2: "S02"}
    label_map.save(tmp_path)
    assert LabelMap.load(tmp_path) == label_map


def test_label_map_encode_and_decode():
    label_map = LabelMap.from_labels(["S01", "S02"])
    assert label_map.encode(["S02", "S01"]) == [1, 0]
    assert label_map.decode([0, 1]) == ["S01", "S02"]
    assert "S02" in label_map and "S03" not in label_map
    with pytest.raises(KeyError, match="S03"):
        label_map.encode(["S03"])
    with pytest.raises(ValueError):
        LabelMap(())


def test_subsample_fraction_count_and_none():
    frame = pd.DataFrame({"x": range(100)})
    assert len(subsample(frame, 0.1)) == 10
    assert len(subsample(frame, 25)) == 25
    assert len(subsample(frame, 1000)) == 100
    assert subsample(frame, None) is frame
    picked = subsample(frame, 30, seed=1)["x"]
    assert picked.is_monotonic_increasing
    assert picked.tolist() == subsample(frame, 30, seed=1)["x"].tolist()
    for bad in (0, -1, 2.5):
        with pytest.raises(ValueError):
            subsample(frame, bad)


def test_collator_truncates_pads_and_adds_labels(tiny_model):
    tokenizer = AutoTokenizer.from_pretrained(tiny_model)
    collate = TextCollator(tokenizer, max_length=16)
    data = NoteDataset(["short", "a much longer note " * 20], [3, 1])
    batch = collate([data[0], data[1]])
    assert batch["input_ids"].shape == (2, 16)
    assert batch["attention_mask"][0].sum() < 16
    assert batch["labels"].tolist() == [3, 1]
    assert "labels" not in collate([NoteDataset(["no label"])[0]])


def label_from_text(text: str) -> str:
    """The label whose marker words occur most often in a synthetic note."""
    words = text.replace(".", " ").replace(":", " ").split()
    return max(MARKERS, key=lambda label: sum(word in MARKERS[label] for word in words))


def test_each_note_keeps_its_label_through_reading_subsampling_and_batching(tmp_path):
    path = tmp_path / "notes.csv"
    make_notes(40, seed=2).to_csv(path, index=False)
    notes = subsample(read_notes(path), 25, seed=3)
    label_map = LabelMap.from_labels(notes["label"])
    data = NoteDataset(notes["text"], label_map.encode(notes["label"]))
    for i in range(len(data)):
        item = data[i]
        assert label_map.decode([item["labels"]]) == [label_from_text(item["text"])]
