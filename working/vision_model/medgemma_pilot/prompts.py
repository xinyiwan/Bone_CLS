"""
Prompt construction for the MedGemma pilot, isolated from inference so the
WORDING can evolve without touching the orchestration / CSV / eval code in
run_medgemma.py.

MedGemma (4b-it / 27b-it) is a generative vision-language model: chat messages
go in, the model decodes one word. We split the prompt into a constant task and
per-image context:

  - The CONSTANT part of the task (role + feature definition + label meanings +
    task + strict answer format) goes in ONE system message.
  - Each user turn is just an image + its per-image context (modality, plane,
    location, ...).
  - Few-shot examples are prior (user: example image -> assistant: label) turns,
    so the task is stated once and every turn is "here is an image, classify it".

Message format (consumed by run_single via the HF processor's chat template,
which accepts the image as a PIL object directly):

    {"role": "system" | "user" | "assistant",
     "content": [ {"type": "text",  "text":  <str>}
                | {"type": "image", "image": <PIL.Image>} ]}
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional

# ---------------------------------------------------------------------------
# Shared: per-image clinical/imaging context (runtime values, not feature-specific)
# ---------------------------------------------------------------------------
def build_context(
    modality: str,
    plane: str,
    location: Optional[str] = None,
    other_planes: Optional[List[str]] = None,
    has_contour: bool = False,
) -> str:
    """The per-image context sentence(s): what this particular image shows. Kept
    separate from the feature task because in a few-shot prompt the task is
    stated ONCE (system message) while this context differs per image/turn.

    Args mirror the old build_prompt() context block:
        modality:      e.g. "T1", "T2FS", "T1FSC"  (per THIS image)
        plane:         orientation of THIS image (axial/coronal/sagittal)
        location:      lesion location, e.g. "distal femur, metaphysis" (optional)
        other_planes:  other orientations of the SAME lesion assessed separately
        has_contour:   True when a radiologist red-contour overlay is being fed
    """
    modality = (modality or "").strip()
    plane = (plane or "").strip() or "unknown-plane"

    parts: List[str] = []
    loc = f", from a bone lesion located in the {location}" if location else ", from a bone lesion"
    parts.append(
        f"This is a {modality} MRI image in the {plane} plane{loc}. The image is cropped to a "
        f"bounding box around the lesion (with a small margin), so the lesion fills most of the "
        f"frame. It is the {plane} slice with the LARGEST cross-sectional area of the lesion; make "
        f"your assessment based on this slice."
    )
    if other_planes:
        others = ", ".join(other_planes)
        parts.append(
            f"Its largest-area slice in other orientations ({others}) is assessed separately in "
            f"other images; judge only the image shown here."
        )
    if has_contour:
        parts.append(
            "A thin RED contour drawn on the image marks the lesion boundary segmented by a "
            "radiologist. Use it to locate the lesion; assess the region it encloses."
        )
    return " ".join(parts)


# ---------------------------------------------------------------------------
# medgemma (generative) — system text + chat messages
# ---------------------------------------------------------------------------
SYSTEM_ROLE = ["You are an expert musculoskeletal radiologist assessing MRI images of benign, malignant bone tumors and bone tumor micmickers.",
               "Maintain strict compliance with these rules:",
               "1. Use EXACT provided label options for your answer, with no extra words or punctuation.",
               "2. If a feature is not clear from the image, use the specified default for that field.",
               "3. NEVER add commentary, explanations, or deviate from the output structure"
]
SYSTEM_ROLE = "\n".join(SYSTEM_ROLE)

def build_system_text(feature_cfg: dict) -> str:
    """The CONSTANT task, assembled from the feature config (YAML). This is the
    system message: role + what the feature IS + what each label MEANS + what to
    DECIDE + the strict one-word answer format. It does not mention any specific
    image, so it can be shared across few-shot example turns and the query turn.

    Structure (all feature wording comes from feature_cfg):
        role                <- SYSTEM_ROLE
        + description       <- feature_cfg["description"]
        + label definitions <- feature_cfg["label_definitions"] (optional)
        + task              <- feature_cfg["task"]
        + answer format     <- from feature_cfg["label_options"]
    """
    opts = ", ".join(feature_cfg["label_options"])
    parts: List[str] = [SYSTEM_ROLE]

    description = (feature_cfg.get("description") or feature_cfg.get("prompt_description") or "").strip()
    if description:
        parts.append(description)
    defs = feature_cfg.get("label_definitions")
    if defs:
        parts.append("Label meanings -- " + "; ".join(f"{k}: {v}" for k, v in defs.items()) + ".")
    task = (feature_cfg.get("task") or "").strip()
    if task:
        parts.append(task)

    parts.append(f"Respond with exactly one word from label options: {opts}. Output only that word, nothing else.")
    return " ".join(parts)


def _user_turn(image, context: str) -> dict:
    """A single user turn: the image plus its per-image context."""
    return {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": context},
        ],
    }


def build_medgemma_messages(
    feature_cfg: dict,
    query_image,
    query_context: str,
    few_shot: Optional[List[dict]] = None,
) -> List[dict]:
    """Assemble the full chat message list for a generative MedGemma call.

    system: the constant task (build_system_text)
    then, for each few-shot example (optional): a user turn (example image +
        its context) followed by an assistant turn (the gold label word)
    finally: the query user turn (image to classify + its context)

    few_shot: list of {"image": PIL, "context": str, "label": str} produced by
    resolve_few_shot(). None/empty -> zero-shot (system + single query turn).
    """
    messages: List[dict] = [
        {"role": "system", "content": [{"type": "text", "text": build_system_text(feature_cfg)}]}
    ]
    for ex in few_shot or []:
        messages.append(_user_turn(ex["image"], ex["context"]))
        messages.append({"role": "assistant", "content": [{"type": "text", "text": ex["label"]}]})
    messages.append(_user_turn(query_image, query_context))
    return messages


def resolve_few_shot(
    feature_cfg: dict,
    base_dir: Path,
    load_image: Callable[[Path], object],
    limit: Optional[int] = None,
) -> List[dict]:
    """Load the per-feature few-shot examples declared in the YAML into ready-to-use
    turns. Each YAML example is:

        examples:
          - image_path: examples/shape_oval_1.jpg   # relative to the config dir (or absolute)
            label: oval                             # must be one of label_options
            modality: T2                            # optional context for the example image
            plane: axial                            # optional
            location: "distal femur, metaphysis"    # optional

    Returns a list of {"image": PIL, "context": str, "label": str}. Raises on a
    missing image or a label not in label_options (fail loud -- a bad exemplar
    silently teaches the model the wrong thing).

    NOTE (leakage): example images MUST be held out of the eval set. run_medgemma
    collects their paths and skips them during inference; still, curate them from
    cases you are not scoring.
    """
    raw = feature_cfg.get("examples") or []
    if limit is not None:
        raw = raw[:limit]
    options_lower = {o.lower(): o for o in feature_cfg["label_options"]}

    resolved: List[dict] = []
    for i, ex in enumerate(raw):
        img_field = ex.get("image_path") or ex.get("image")
        if not img_field:
            raise ValueError(f"few-shot example #{i} for a feature is missing 'image_path'")
        img_path = Path(img_field)
        if not img_path.is_absolute():
            img_path = base_dir / img_path

        label = str(ex.get("label", "")).strip()
        if label.lower() not in options_lower:
            raise ValueError(
                f"few-shot example {img_path} has label {label!r} not in label_options "
                f"{feature_cfg['label_options']}"
            )
        label = options_lower[label.lower()]  # canonical casing

        image = load_image(img_path)  # raises if missing/unreadable -- intentional
        context = build_context(
            ex.get("modality", ""),
            ex.get("plane", ""),
            location=ex.get("location"),
            other_planes=None,
            has_contour=bool(ex.get("has_contour", False)),
        )
        resolved.append({"image": image, "context": context, "label": label, "image_path": str(img_path)})
    return resolved


def few_shot_image_paths(feature_cfg: dict, base_dir: Path) -> List[str]:
    """The resolved image paths used as few-shot exemplars for a feature, so the
    caller can exclude them from the eval/inference set (leakage guard). Does not
    load anything."""
    paths: List[str] = []
    for ex in feature_cfg.get("examples") or []:
        img_field = ex.get("image_path") or ex.get("image")
        if not img_field:
            continue
        p = Path(img_field)
        paths.append(str(p if p.is_absolute() else base_dir / p))
    return paths
