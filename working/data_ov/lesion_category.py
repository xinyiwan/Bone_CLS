"""Maps diagnosis subtypes (labels.py's English labels) to a coarse category:
benign, malignant, mimicker (non-neoplastic tumor mimicker), none (no lesion),
or uncertain (ambiguous / missing report).
"""

from labels import MISSING_LABEL, UNCERTAIN_LABEL

NONE_LABEL = "none"

LESION_CATEGORY = {
    # --- benign bone tumors / tumor-like lesions ---
    "enchondroma": "benign",
    "osteochondroma": "benign",
    "bone cyst": "benign",
    "intraosseous lipoma": "benign",
    "fibrous dysplasia": "benign",
    "non-ossifying fibroma": "benign",
    "atypical cartilaginous tumor": "intermediate",   # locally aggressive, but biologically benign (ACT - long bones/CS1 - axial skeletion)
    "aneurysmal bone cyst": "benign",
    "giant cell tumor": "intermediate",               # locally aggressive, rarely metastasizes
    "Nora's lesion": "benign",
    "osteoblastoma": "intermediate",                  # per reference table
    "desmoplastic fibroma": "intermediate",           # per reference table
    "hemangioma": "benign",
    "osteoid osteoma": "benign",
    "chondromyxoid fibroma": "benign",
    "osteomyelolipoma": "benign",

    # --- malignant bone tumors ---
    "bone metastasis": "malignant",
    "chondrosarcoma": "malignant",
    "osteosarcoma": "malignant",
    "plasmacytoma": "malignant",
    "chordoma": "malignant",
    "adamantinoma": "malignant",
    "primary non-Hodgkin lymphoma of bone": "malignant", # same as "bone lymphoma" 
    "bone lymphoma": "malignant",
    "hemangioendothelioma": "malignant",        # epithelioid hemangioendothelioma -- borderline/malignant
    "Ewing sarcoma": "malignant",
    "angiosarcoma": "malignant",

    # --- bone tumor mimickers (non-neoplastic) ---
    "Paget's disease": "mimicker",
    "bone infarct": "mimicker",
    "chronic osteomyelitis": "mimicker",
    "osteomyelitis": "mimicker",
    "mastocytosis": "mimicker",                 # systemic disease with bone involvement
    "brown tumor": "mimicker",                    # hyperparathyroidism-related, not a true neoplasm


    # --- no lesion / uncertain ---
    "none": NONE_LABEL,
    UNCERTAIN_LABEL: "uncertain",
    MISSING_LABEL: "uncertain",
}


def to_category(label: str) -> str:
    """Map a labels.py diagnosis label to its coarse category."""
    category = LESION_CATEGORY.get(label)
    if category is None:
        print(f"WARNING: no category for label {label!r} -> 'uncertain'")
        return "uncertain"
    return category
