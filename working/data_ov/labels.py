"""Shared label logic for the bone-lesion distribution scripts.

Rules:
  - A ``palabra_manual`` value with a comma (multiple diagnoses) -> "uncertain by reports".
  - A single diagnosis is translated Spanish -> English via TRANSLATION.
  - Blank / missing -> "(missing)".
"""

UNCERTAIN_LABEL = "uncertain by reports"
MISSING_LABEL = "(missing)"

# Spanish -> English translation for single (unambiguous) diagnoses.
TRANSLATION = {
    "encondroma": "enchondroma",
    "osteocondroma": "osteochondroma",
    "quiste óseo": "bone cyst",
    "metástasis ósea": "bone metastasis",
    "plasmocitoma": "plasmacytoma",
    "paget": "Paget's disease",
    "lipoma intraóseo": "intraosseous lipoma",
    "condrosarcoma": "chondrosarcoma",
    "infarto óseo": "bone infarct",
    "osteosarcoma": "osteosarcoma",
    "tumor cartilaginoso atípico": "atypical cartilaginous tumor",
    "displasia fibrosa": "fibrous dysplasia",
    "fibroma no osificante": "non-ossifying fibroma",
    "quiste óseo aneurismatico": "aneurysmal bone cyst",
    "osteomielitis crónica": "chronic osteomyelitis",
    "osteomielitis": "osteomyelitis",
    "tumor de células gigantes": "giant cell tumor",
    "cordoma": "chordoma",
    "nora": "Nora's lesion",
    "hemangioma": "hemangioma",
    "linfoma oseo": "bone lymphoma",
    "fibroma desmoplásico": "desmoplastic fibroma",
    "osteoblastoma": "osteoblastoma",
    "osteoma osteoide": "osteoid osteoma",
    "mastocitosis": "mastocytosis",
    "osteomielolipoma": "osteomyelolipoma",
    "tumor pardo": "brown tumor",
    "ewing": "Ewing sarcoma",
    "angiosarcoma": "angiosarcoma",
    "hemangioendotelioma": "hemangioendothelioma",
    "fibroma condromixoide": "chondromyxoid fibroma",
    "adamantimoma": "adamantinoma",
    "ninguna": "none",
}


def to_label(value) -> str:
    """Map a raw palabra_manual value to its plotting label."""
    value = str(value).strip()
    if value in ("", MISSING_LABEL, "nan"):
        return MISSING_LABEL
    # Multiple entries (comma-separated) -> uncertain.
    if "," in value:
        return UNCERTAIN_LABEL
    english = TRANSLATION.get(value.lower())
    if english is None:
        print(f"WARNING: no translation for single entry {value!r} -> keeping as-is")
        return value
    return english
