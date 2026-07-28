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

import json
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional

log = logging.getLogger("medgemma.prompts")

# ---------------------------------------------------------------------------
# Shared: per-image clinical/imaging context (runtime values, not feature-specific)
# ---------------------------------------------------------------------------
def build_context(
    modality: str,
    plane: str,
    location: Optional[str] = None,
    other_planes: Optional[List[str]] = None,
    has_contour: bool = False,
    if_example: bool = False,
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
    modality = (modality or "").strip() or "MRI"
    plane = (plane or "").strip() or "unknown-plane"

    loc = f", of a bone lesion in the {location}" if location else ", of a bone lesion"
    parts: List[str] = [f"This is a {modality} MRI in the {plane} plane{loc}."]

    # The imaging explanation (cropped to the lesion, largest-area slice, sibling
    # orientations) is only needed for the QUERY -- stating it on every example
    # turn just repeats it. Examples keep a short context (modality/plane/location).
    if not if_example:
        parts.append(
            "The image is cropped to a bounding box around the lesion, so the lesion fills most "
            f"of the frame; it is the {plane} slice with the LARGEST cross-sectional area of the lesion. "
            "Make your assessment based on this slice."
        )
        if other_planes:
            parts.append(
                f"Its largest-area slice in other orientations ({', '.join(other_planes)}) is "
                "assessed separately in other images; judge only the image shown here."
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
# Role + the one non-format rule. The output-structure rules that used to live
# here are dropped -- the "# OUTPUT FORMAT" section states them once, in full.
SYSTEM_ROLE = (
    "You are an expert musculoskeletal radiologist. For each case, you are shown "
    "a single MRI slice of a bone lesion and asked to assess ONE specific imaging "
    "feature, choosing from a fixed set of labels defined below. Base your "
    "judgment only on what is visible in the image provided, not on other slices, "
    "planes, or assumptions about the case. If the feature is not clearly "
    "assessable from the image, choose the closest label and note the "
    "uncertainty in the reason."
)

# Announced once at the end of the system message (only in few-shot), BEFORE any
# example turns -- so the examples aren't glued onto the first image's context.
EXAMPLES_NOTE = ("The next turns are WORKED EXAMPLES: each shows an image and the correct answer, "
                 "to illustrate the task. Study them, then classify the final (query) image.")


def build_system_text(feature_cfg: dict, has_examples: bool = False) -> str:
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
    sections: List[str] = [SYSTEM_ROLE]

    description = (feature_cfg.get("description") or feature_cfg.get("prompt_description") or "").strip()
    if description:
        sections.append("# FEATURE\n" + description)
    defs = feature_cfg.get("label_definitions")
    if defs:
        sections.append("# LABEL DEFINITIONS\n" + "\n".join(f"- {k}: {v}" for k, v in defs.items()))
    task = (feature_cfg.get("task") or "").strip()
    if task:
        sections.append("# TASK\n" + task)

    # Strict structured-output constraint (constant across features; only the
    # allowed values vary). Stating it explicitly keeps output consistent and
    # forces the reason field.
    sections.append(
        "# OUTPUT FORMAT\n"
        'Reply with ONLY this JSON, nothing else (no markdown/fences):\n'
        '{"prediction": "<LABEL>", "reason": "<one short sentence>"}\n'
        f"<LABEL> must be exactly one of: {opts}."
    )
    # Announce few-shot examples here (once, before any example turn) rather than
    # gluing the announcement onto the first example's image context.
    if has_examples:
        sections.append("# EXAMPLES\n" + EXAMPLES_NOTE)
    # Blank line between sections so the model (and you, in the logged input_text)
    # can tell role / feature / definitions / task / format apart.
    return "\n\n".join(sections)


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
    few = few_shot or []
    messages: List[dict] = [
        {"role": "system",
         "content": [{"type": "text", "text": build_system_text(feature_cfg, has_examples=bool(few))}]}
    ]
    for ex in few:
        messages.append(_user_turn(ex["image"], ex["context"]))
        # The exemplar answer must be the SAME JSON structure we ask the model to
        # produce, or the examples teach a different format than the system text.
        answer = json.dumps({"prediction": ex["label"], "reason": ex.get("reason", "")})
        messages.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    messages.append(_user_turn(query_image, query_context))
    return messages


def messages_to_text(messages: List[dict]) -> str:
    """Render a chat message list as readable text -- the exact input fed to the
    model, with each image shown as an '<image>' placeholder. Used to log what
    was paired with each query image (system task + few-shot turns + context)."""
    lines: List[str] = []
    for m in messages:
        parts = []
        for item in m.get("content", []):
            if item.get("type") == "image":
                parts.append("<image>")
            else:
                parts.append(item.get("text", ""))
        lines.append(f"[{m.get('role', '?')}] " + " ".join(p for p in parts if p))
    return "\n".join(lines)


def overlay_variant(path: Path) -> Path:
    """The `_overlay` (red-contour) sibling of a crop, as written by the
    preprocessing pipeline (matches run_medgemma._overlay_variant)."""
    return path.with_name(path.stem + "_overlay" + path.suffix)


def resolve_few_shot(
    feature_cfg: dict,
    base_dir: Path,
    load_image: Callable[[Path], object],
    limit: Optional[int] = None,
    use_contour: bool = False,
) -> List[dict]:
    """Load the per-feature few-shot examples declared in the YAML into ready-to-use
    turns. Each YAML example is:

        examples:
          - image_path: examples/shape_oval_1.jpg   # relative to the config dir (or absolute)
            label: oval                             # must be one of label_options
            reason: "smooth egg-like outline"       # shown in the exemplar JSON answer
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

        # Match the query-image style: when contour is on (globally via use_contour
        # or per-example has_contour), feed the red-contour overlay instead of the
        # plain crop -- but only if the overlay actually exists, else fall back so
        # we never claim a contour the model can't see.
        want_contour = use_contour or bool(ex.get("has_contour", False))
        has_contour = False
        load_path = img_path
        if want_contour:
            ov = overlay_variant(img_path)
            if ov.exists():
                load_path, has_contour = ov, True
            else:
                log.warning("few-shot: no overlay for %s -- using plain crop (no contour)", img_path)

        image = load_image(load_path)  # raises if missing/unreadable -- intentional
        context = build_context(
            ex.get("modality", ""),
            ex.get("plane", ""),
            location=ex.get("location"),
            other_planes=None,
            has_contour=has_contour,
            if_example=True,  # frame this as a teaching example, not a query to assess
        )
        resolved.append({
            "image": image,
            "context": context,
            "label": label,
            "reason": str(ex.get("reason", "")).strip(),
            "image_path": str(img_path),  # plain path (leakage guard/subject keys off this)
        })
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
