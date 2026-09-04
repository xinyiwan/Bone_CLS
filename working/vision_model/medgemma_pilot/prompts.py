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
# The dataset stores modality as a terse code (T1W, T1W_C, ...). Spelled out for
# the model: a code like "T1W_C" carries no meaning to a language model, while
# "T1-weighted with contrast enhancement" does. Base sequence first, then the
# suffix flags, in the order they appear in the code.
_MODALITY_BASE = {
    "T1W": "T1-weighted",
    "T2W": "T2-weighted",
    "PDW": "proton-density-weighted",
    "STIR": "STIR (short-tau inversion recovery)",
    "DWI": "diffusion-weighted",
    "ADC": "ADC (apparent diffusion coefficient) map",
}
_MODALITY_FLAG = {
    "C": "contrast enhancement",
    "FS": "fat suppression",
}


def describe_modality(modality: str, plural: bool = False):
    """Expand a modality code into (noun phrase, acquisition sentence), e.g.
    "T1W_C" -> ("T1-weighted MRI image (T1W_C)",
                "It was acquired with contrast enhancement.")

    The flags are a separate sentence rather than a clause so the modality can
    stay adjacent to the plane/location in the opening sentence. Unknown
    codes/flags are passed through unchanged so an unexpected value degrades to
    the old behaviour instead of being dropped.
    """
    code = (modality or "").strip()
    noun = "MRI images" if plural else "MRI image"
    if not code:
        return noun, ""
    tokens = code.replace("-", "_").upper().split("_")
    base = _MODALITY_BASE.get(tokens[0])
    if base is None:
        return f"{code} {noun}", ""
    flags = [_MODALITY_FLAG.get(t, t.lower()) for t in tokens[1:] if t]
    acq = ""
    if flags:
        subject = "They were" if plural else "It was"
        acq = f"{subject} acquired with " + " and ".join(flags) + "."
    return f"{base} {noun} ({code})", acq


def build_context(
    modality: str,
    plane: str,
    location: Optional[str] = None,
    other_planes: Optional[List[str]] = None,
    has_contour: bool = False,
    if_example: bool = False,
    n_slices: int = 0,
) -> str:
    """The per-image context sentence(s): what this particular image shows. Kept
    separate from the feature task because in a few-shot prompt the task is
    stated ONCE (system message) while this context differs per image/turn.

    Args mirror the old build_prompt() context block:
        modality:      code, e.g. "T1W", "T1W_C", "T2W_FS" (per THIS image);
                       expanded to full wording by describe_modality()
        plane:         orientation of THIS image (axial/coronal/sagittal)
        location:      lesion location, e.g. "distal femur, metaphysis" (optional)
        other_planes:  other orientations of the SAME lesion assessed separately
        has_contour:   True when a radiologist red-contour overlay is being fed
        n_slices:      >0 selects the STACK arm: this turn carries n contiguous
                       slices of ONE lesion instead of a single slice. Mutually
                       exclusive with the largest-area/other-planes wording,
                       which is why it returns early rather than adding a clause.
    """
    modality_desc, modality_acq = describe_modality(modality, plural=bool(n_slices))
    plane = (plane or "").strip() or "unknown-plane"

    loc = f", of a bone lesion in the {location}" if location else ", of a bone lesion"
    lead = "These are" if n_slices else "This is a"
    parts: List[str] = [f"{lead} {modality_desc} in the {plane} plane{loc}."]
    if modality_acq:
        parts.append(modality_acq)

    # The STACK arm replaces the single-slice framing entirely: what the images
    # are, how many, and -- the part that carries the 3D information -- what
    # order they are in. A stack of images with no stated ordering is a bag of
    # slices, so `n_slices` and the through-plane direction are not optional.
    if n_slices:
        return " ".join(_stack_context_parts(parts, modality, plane, n_slices,
                                             has_contour, if_example))

    # The imaging explanation (cropped to the lesion, largest-area slice, sibling
    # orientations) is only needed for the QUERY -- stating it on every example
    # turn just repeats it. Examples keep a short context (modality/plane/location).

    if has_contour:
        parts.append(
            "A thin RED contour drawn on the image marks the lesion boundary segmented by a "
            "radiologist. Use it to locate the lesion; assess the region it encloses."
        )
    return " ".join(parts)


def _stack_context_parts(head: List[str], modality: str, plane: str, n_slices: int,
                         has_contour: bool, if_example: bool) -> List[str]:
    """Context clauses for one multi-slice turn (see build_context's n_slices)."""
    parts = list(head)
    parts.append(
        f"The following {n_slices} images are CONSECUTIVE {plane} slices through that ONE "
        "lesion, in anatomical order from the first slice on which it appears to the last. "
        "They are not separate cases: each image is a different level through the same mass."
    )
    if not if_example:
        parts.append(
            "Every slice shares the same crop box, so the lesion sits at the CENTRE of "
            "each frame -- the central mass, not the tissue at the edges. It fills most "
            "of the middle slices and is smaller, possibly off-centre, on the first and "
            "last few. Read the slices as one volume: judge how the lesion CHANGES "
            "across them, and count a finding as present if any slice shows it."
        )
    if has_contour:
        parts.append(
            "A thin RED contour on each image marks the lesion boundary segmented by a "
            "radiologist on that slice. Use it to locate the lesion; assess the region it encloses."
        )
    return parts


# ---------------------------------------------------------------------------
# medgemma (generative) — system text + chat messages
# ---------------------------------------------------------------------------
# Role + the one non-format rule. The output-structure rules that used to live
# here are dropped -- the "# OUTPUT FORMAT" section states them once, in full.
SYSTEM_ROLE = (
    "You are an expert musculoskeletal radiologist. You are shown a single MRI "
    "slice of a bone lesion and must assess ONE imaging feature, choosing from "
    "the fixed labels defined below. Judge only what is visible in this image. "
    "If the feature is not clearly assessable, pick the closest label and note "
    "the uncertainty in your reason."
)

# The free-text counterparts. SYSTEM_ROLE and SYSTEM_ROLE_STACK both promise a
# fixed label set and offer the hedging escape; under free text no vocabulary is
# given, so those clauses are a false premise and an invitation to a non-answer
# respectively. Kept as separate literals rather than assembled from fragments so
# the label-mode strings stay byte-identical to earlier runs.
SYSTEM_ROLE_FREE = (
    "You are an expert musculoskeletal radiologist. You are shown a single MRI "
    "slice of a bone lesion and must describe ONE imaging feature in your own "
    "words. Judge only what is visible in this image. Say whether you found the "
    "lesion and what you actually see; if the feature is genuinely not "
    "assessable, say so and say why."
)

SYSTEM_ROLE_STACK_FREE = (
    "You are an expert musculoskeletal radiologist. You are shown a STACK of "
    "consecutive MRI slices through one bone lesion and must describe ONE "
    "imaging feature in your own words. Judge only what is visible in these "
    "images, not other planes or assumptions about the case. Reason across the "
    "whole stack: the slices show one lesion at different levels, so a feature "
    "may be evident on only some of them. Say what you actually see; if the "
    "feature is genuinely not assessable, say so and say why."
)

# The STACK counterpart. Every clause that restricts the model to one image is
# inverted -- notably "not on other slices", which under a multi-slice prompt
# tells the model to ignore exactly the information the arm exists to test, and
# would make a null 3D result uninterpretable. The hedging escape ("choose the
# closest label and note the uncertainty") is dropped too: it belongs to the
# forced-choice format, and under free text it just licenses a non-answer.
SYSTEM_ROLE_STACK = (
    "You are an expert musculoskeletal radiologist. You are shown a STACK of "
    "consecutive MRI slices through one bone lesion and must assess ONE imaging "
    "feature. Judge only what is visible in these images, not other planes or "
    "assumptions about the case. Reason across the whole stack: the slices show "
    "one lesion at different levels, so a feature may be evident on only some of "
    "them."
)

# Free-text output. No vocabulary, no length cap -- the point of the arm is the
# reasoning, and a one-sentence cap is what produced unusable hedges before. The
# headings are fixed only so a 40-case manual read stays skimmable; they impose
# no answer set. The last line asks the model to ground each claim in a slice,
# which is what makes a wrong answer diagnosable rather than merely wrong.
def free_text_format(input_mode: str = "slice", rank_options: Optional[List[str]] = None) -> str:
    """The OUTPUT FORMAT block for the prose arms.

    rank_options turns ASSESSMENT from open prose into a RANKING over exactly
    those labels, written `best > next > worst`. That chain is the whole point:
    the free-text runs kept concluding "irregular and somewhat lobulated", three
    terms with no ordering, which is unscoreable and indistinguishable from a
    hedge. A total order is parseable by split('>'), and yields top-1 accuracy
    and mean reciprocal rank without anyone having to interpret prose. Every
    label must appear exactly once -- a partial list would silently conflate
    "ranked last" with "forgot to mention".
    """
    stack = input_mode == "stack"
    noun = "the images" if stack else "the image"
    # Only the stack arm can be asked to cite slices; saying it to a single-image
    # prompt invites the model to invent slice numbers it was never shown.
    cite = " Where a claim rests on particular slices, say which ones." if stack else ""
    # REASONING comes FIRST, and before ASSESSMENT specifically. Asked for after
    # the conclusion it becomes post-hoc justification for an answer already
    # committed to, which is the opposite of the deliberation this arm exists to
    # read. It is in-band prose, not the model's native thinking block -- see
    # run_batch on why that block may not be recoverable.
    # LESION is a GROUNDING check and comes before everything else: every later
    # heading describes a feature OF the lesion, so if the model never located
    # it, those descriptions are about nothing and read exactly like real ones.
    # Asked first, it cannot be reverse-engineered from a shape the model has
    # already committed to. The explicit permission to say "I cannot identify
    # it" is the load-bearing part -- without it the model will always name
    # something, which is precisely the hallucination this is meant to catch.
    where = ("which slices it appears on and where it sits within the frame"
             if stack else "where it sits within the frame")
    return (
        "[OUTPUT FORMAT]\n"
        "Reply in plain prose under exactly these four headings, in this order, "
        "nothing else (no JSON, no markdown, no bullet list of options).\n"
        f"LESION: can you identify the lesion at all? If so, say {where} and what "
        "distinguishes it from surrounding tissue. If you cannot confidently tell "
        "which structure is the lesion, say so plainly -- that is a valid answer, "
        "not a failure.\n"
        f"OBSERVATIONS: what you actually see in {noun} that bears on this feature.\n"
        "REASONING: weigh those observations against each other -- the "
        "alternatives, why you favour one, what leaves you uncertain. Think it "
        "through here, before you conclude.\n"
        + (
            "ASSESSMENT: your conclusion, in your own words. Do not pick from a "
            "list of terms -- none is given. Describe what is there."
            if not rank_options else
            "ASSESSMENT: rank ALL the descriptors below, best fit first, separated "
            "by '>', nothing else on that line.\n"
            f"  descriptors, written exactly as given: {', '.join(rank_options)}\n"
            f"  required form: ASSESSMENT: {' > '.join(rank_options)}\n"
            "Every descriptor exactly once, even ones that fit badly. Rank even a "
            "close call -- closer one first, explain the closeness in REASONING. "
            "Do not add your own words to this line; put a better word in "
            "REASONING instead."
        )
        + cite
    )


def free_text_task(feature_cfg: dict) -> str:
    """The TASK line for the free-text arm.

    The YAML `task` fields tell the model to CHOOSE a label, which contradicts
    this arm, and `description` is worded "as seen on this MRI slice". Both can
    be overridden per feature (`free_text_task`, `description_stack`); the
    fallback is derived from the feature name so a feature works untouched.
    """
    override = (feature_cfg.get("free_text_task") or "").strip()
    if override:
        return override
    name = (feature_cfg.get("display_name") or feature_cfg.get("name") or "this feature").strip()
    return (f"Describe {name} for the lesion shown, and explain what in the images leads you "
            "to that description. Do not assess any other feature.")


# Announced once at the end of the system message (only in few-shot), BEFORE any
# example turns -- so the examples aren't glued onto the first image's context.
EXAMPLES_NOTE = ("The next turns are WORKED EXAMPLES: each shows an image and the correct answer, "
                 "to illustrate the task. Study them, then classify the final (query) image.")


def build_system_text(feature_cfg: dict, has_examples: bool = False,
                      input_mode: str = "slice", output_mode: str = "label") -> str:
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
    if input_mode not in ("slice", "stack"):
        raise ValueError(f"input_mode must be 'slice' or 'stack', got {input_mode!r}")
    if output_mode not in ("label", "free_text", "ranked"):
        raise ValueError(f"output_mode must be 'label', 'free_text' or 'ranked', "
                         f"got {output_mode!r}")
    # 'ranked' shares free text's PROSE sections (it reasons the same way) but,
    # unlike free text, it IS given the vocabulary -- it just orders it instead
    # of picking one. So it is "free" for the role/task, not for the definitions.
    prose = output_mode in ("free_text", "ranked")

    role = {
        ("slice", "label"): SYSTEM_ROLE,
        ("slice", "free_text"): SYSTEM_ROLE_FREE,
        ("slice", "ranked"): SYSTEM_ROLE_FREE,
        ("stack", "label"): SYSTEM_ROLE_STACK,
        ("stack", "free_text"): SYSTEM_ROLE_STACK_FREE,
        ("stack", "ranked"): SYSTEM_ROLE_STACK_FREE,
    }[(input_mode, output_mode)]
    sections: List[str] = [role]

    # `description` is written "as seen on this MRI slice"; the stack arm can
    # override it per feature rather than feeding the model a false premise.
    description = (feature_cfg.get("description") or feature_cfg.get("prompt_description") or "").strip()
    if input_mode == "stack":
        description = (feature_cfg.get("description_stack") or "").strip() or description
    if description:
        sections.append("[FEATURE TO ASSESS]\n" + description)

    # LABEL DEFINITIONS is the vocabulary. Emitting it under free text would hand
    # the model the answer set the arm is designed to withhold, so it is dropped
    # there -- along with the label-valued OUTPUT FORMAT.
    defs = feature_cfg.get("label_definitions")
    if defs and output_mode != "free_text":
        sections.append("[DEFINITIONS]\n" + "\n".join(f"- {k}: {v}" for k, v in defs.items()))

    task = free_text_task(feature_cfg) if prose else (feature_cfg.get("task") or "").strip()
    if task:
        sections.append("[TASK]\n" + task)

    if output_mode == "ranked":
        sections.append(free_text_format(input_mode, feature_cfg["label_options"]))
    elif output_mode == "free_text":
        sections.append(free_text_format(input_mode))
    else:
        # Strict structured-output constraint (constant across features; only the
        # allowed values vary). Stating it explicitly keeps output consistent and
        # forces the reason field.
        opts = ", ".join(feature_cfg["label_options"])
        sections.append(
            "[OUTPUT FORMAT]\n"
            "Reply with ONLY this JSON, nothing else (no markdown, no fences, no headings):\n"
            '{"prediction": "<LABEL>", "reason": "<one short sentence>"}\n'
            f"<LABEL> must be exactly one of: {opts}."
        )
    # Announce few-shot examples here (once, before any example turn) rather than
    # gluing the announcement onto the first example's image context.
    if has_examples:
        sections.append("[EXAMPLES]\n" + EXAMPLES_NOTE)
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


def _user_turn_stack(images, context: str) -> dict:
    """One user turn carrying a whole slice stack: the context sentence FIRST,
    then every image in order. Context leads here (unlike _user_turn) because it
    is what tells the model the images that follow are one ordered volume rather
    than unrelated cases -- stated after them, it arrives too late to frame them."""
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": context},
            *({"type": "image", "image": im} for im in images),
        ],
    }


def build_medgemma_messages(
    feature_cfg: dict,
    query_image,
    query_context: str,
    few_shot: Optional[List[dict]] = None,
    output_mode: str = "label",
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
    # Few-shot exemplars answer with a label in JSON, which teaches the terse
    # forced-choice format the free-text arm exists to avoid -- so the two are
    # refused together rather than silently producing a contaminated prompt.
    if few and output_mode == "free_text":
        raise ValueError(
            "few-shot exemplars answer with a label, which contradicts output_mode='free_text'; "
            "run the free-text arm zero-shot, or write prose exemplar answers first"
        )
    messages: List[dict] = [
        {"role": "system",
         "content": [{"type": "text", "text": build_system_text(
             feature_cfg, has_examples=bool(few), output_mode=output_mode)}]}
    ]
    for ex in few:
        messages.append(_user_turn(ex["image"], ex["context"]))
        # The exemplar answer must be the SAME JSON structure we ask the model to
        # produce, or the examples teach a different format than the system text.
        answer = json.dumps({"prediction": ex["label"], "reason": ex.get("reason", "")})
        messages.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    messages.append(_user_turn(query_image, query_context))
    return messages


def build_stack_messages(
    feature_cfg: dict,
    query_images: List,
    query_context: str,
    output_mode: str = "free_text",
) -> List[dict]:
    """Zero-shot messages for the STACK arm: system + one multi-image user turn.

    No few_shot parameter, deliberately. Under free text the exemplar answers
    would have to be prose someone wrote by hand, which puts that person's
    vocabulary into the model's mouth and leaves you grading text you partly
    authored; under a stack each exemplar also multiplies the image budget.
    Few-shot on this arm is a separate decision, not a default.
    """
    if not query_images:
        raise ValueError("build_stack_messages needs at least one image")
    system = build_system_text(feature_cfg, has_examples=False,
                               input_mode="stack", output_mode=output_mode)
    return [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        _user_turn_stack(query_images, query_context),
    ]


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


# ---------------------------------------------------------------------------
# CLI: render a prompt with no model, no images, no metadata -- for eyeballing
# wording changes. `run_medgemma.py --mode quick` needs a loaded model and a
# real image just to show you the text; this needs neither.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import sys

    import yaml

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", type=Path, default=Path("feature_prompts.yaml"))
    ap.add_argument("--feature", default="shape")
    ap.add_argument("--input-mode", choices=["slice", "stack"], default="slice")
    ap.add_argument("--output-mode", choices=["label", "free_text", "ranked"], default="label")
    ap.add_argument("--n-slices", type=int, default=23,
                    help="only shown/used for --input-mode stack, as the per-image context")
    ap.add_argument("--few-shot", action="store_true",
                    help="also print the [EXAMPLES] announcement, as it would appear "
                         "with --num-few-shot > 0 (system text only; doesn't load example images)")
    args = ap.parse_args()

    with open(args.config) as fh:
        features = yaml.safe_load(fh)["features"]
    if args.feature not in features:
        raise SystemExit(f"no feature {args.feature!r} in {args.config} "
                         f"(have: {', '.join(features)})")
    cfg = features[args.feature]

    system = build_system_text(cfg, has_examples=args.few_shot,
                               input_mode=args.input_mode, output_mode=args.output_mode)
    context = build_context("T2W_FS", "axial", location="distal femur, metaphysis",
                            n_slices=args.n_slices if args.input_mode == "stack" else 0)

    print(f"===== [system] ({args.input_mode}/{args.output_mode}, "
          f"{len(system)} chars) =====", file=sys.stderr)
    print(system)
    print(f"\n===== [user context] ({len(context)} chars) =====", file=sys.stderr)
    print(context)
