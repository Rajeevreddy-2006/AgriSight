"""Label mapping between PlantVillage (lab) and PlantDoc (field) class folders."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

SOURCES = ("plantvillage", "plantdoc")


def normalize(name: str) -> str:
    """Lowercase and keep only letters/digits so folder-name variants still match."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


@dataclass
class LabelMap:
    classes: list[str]
    info: dict[str, dict]
    lookup: dict[str, dict[str, str]] = field(default_factory=dict)

    def canonical(self, source: str, folder: str) -> str | None:
        return self.lookup.get(source, {}).get(normalize(folder))

    def display(self, label: str) -> str:
        i = self.info[label]
        return f"{i['crop']} - {i['disease']}"


def load_label_map(path: str | Path) -> LabelMap:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    info: dict[str, dict] = {}
    lookup: dict[str, dict[str, str]] = {s: {} for s in SOURCES}
    for name, spec in data["classes"].items():
        info[name] = {
            "crop": spec["crop"],
            "disease": spec["disease"],
            "healthy": spec["disease"].strip().lower() == "healthy",
        }
        for source in SOURCES:
            for folder in spec.get(source, []) or []:
                key = normalize(folder)
                previous = lookup[source].get(key)
                if previous and previous != name:
                    raise ValueError(f"{source} folder '{folder}' mapped to both {previous} and {name}")
                lookup[source][key] = name
    return LabelMap(classes=sorted(info), info=info, lookup=lookup)
