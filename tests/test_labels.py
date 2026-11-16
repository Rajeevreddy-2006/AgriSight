from pathlib import Path

from agrisight.labels import load_label_map, normalize

LABEL_MAP = Path(__file__).resolve().parents[1] / "configs" / "label_map.yaml"


def test_normalize_ignores_case_and_punctuation():
    assert normalize("Corn_(maize)___Common_rust_") == normalize("corn (maize) common rust")


def test_every_class_has_both_domains():
    lm = load_label_map(LABEL_MAP)
    assert len(lm.classes) == 28
    for source in ("plantvillage", "plantdoc"):
        mapped = set(lm.lookup[source].values())
        assert mapped == set(lm.classes), f"{source} is missing {set(lm.classes) - mapped}"


def test_lookup_and_display():
    lm = load_label_map(LABEL_MAP)
    assert lm.canonical("plantvillage", "Tomato___Late_blight") == "tomato_late_blight"
    assert lm.canonical("plantdoc", "Tomato leaf late blight") == "tomato_late_blight"
    assert lm.canonical("plantdoc", "Not a class") is None
    assert lm.display("tomato_late_blight") == "Tomato - Late blight"
    assert lm.info["tomato_healthy"]["healthy"] is True
