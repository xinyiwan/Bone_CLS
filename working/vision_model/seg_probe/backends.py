"""
Box-promptable 2D segmenters, behind one interface.

    segment(image_rgb_uint8, box_xyxy) -> bool mask, same HxW as the image

Registry so a new model is one function, and so the scoring/reporting code never
knows which model produced a mask.

TWO OFFLINE BACKENDS, and they are not filler
    `box_fill` and `smooth_oracle` need no weights and no GPU. They exist because
    a metric you have not calibrated is a metric you cannot interpret:

    box_fill      fills the (jittered) prompt box. The FLOOR. On a compact lesion
                  in a tight crop this scores a surprisingly high Dice -- often
                  0.6-0.7. Any real segmenter must clear it by a wide margin, and
                  if it doesn't, "Dice 0.75" was never evidence of anything.

    smooth_oracle takes the ground-truth mask and morphologically smooths it.
                  Dice stays very high (~0.95) BY CONSTRUCTION while the boundary
                  detail is destroyed. It is the null model for the whole
                  question in this directory: if your roughness metrics do not
                  flag smooth_oracle, they will not flag SAM either. Run it first
                  and confirm the report catches it.

    Note smooth_oracle sees the ground truth, so its Dice is meaningless as
    performance. It is a metric test, not a model.

THE PROMPT-PROVENANCE CAVEAT
    Every model here is PROMPTABLE, not automatic. Prompting with a box derived
    from the radiologist's mask means a human is still in the loop -- jitter
    makes the number honest but does not make the pipeline autonomous. For that
    you need a detector (or nnU-Net) generating the box. Say so explicitly rather
    than letting "foundation model" imply "no human".
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, Tuple

import cv2
import numpy as np

log = logging.getLogger("seg_probe.backends")

Box = Tuple[int, int, int, int]
Segmenter = Callable[[np.ndarray, Box], np.ndarray]

_REGISTRY: Dict[str, Callable[..., Segmenter]] = {}


def register(name: str):
    def deco(fn):
        _REGISTRY[name] = fn
        return fn
    return deco


def available() -> list:
    return sorted(_REGISTRY)


def make_segmenter(name: str, **kwargs) -> Segmenter:
    if name not in _REGISTRY:
        raise SystemExit(f"unknown --backend {name!r}; available: {available()}")
    return _REGISTRY[name](**kwargs)


# --------------------------------------------------------------------------
# offline baselines
# --------------------------------------------------------------------------

@register("box_fill")
def _box_fill(**_) -> Segmenter:
    """Everything inside the prompt box. The floor for Dice."""
    def seg(image: np.ndarray, box: Box) -> np.ndarray:
        m = np.zeros(image.shape[:2], dtype=bool)
        x0, y0, x1, y1 = box
        m[y0:y1 + 1, x0:x1 + 1] = True
        return m
    return seg


@register("smooth_oracle")
def _smooth_oracle(smooth_px: int = 5, **_) -> Segmenter:
    """Ground truth with the boundary detail morphologically removed -- a
    synthetic stand-in for a segmenter with a strong smoothness prior.

    The GT mask is passed via the `gt` attribute set by the caller, which is a
    small hack, but keeping it out of the Segmenter signature means no real
    backend can accidentally reach the answer."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (smooth_px, smooth_px))

    def seg(image: np.ndarray, box: Box) -> np.ndarray:
        gt = getattr(seg, "gt", None)
        if gt is None:
            raise RuntimeError("smooth_oracle requires the ground-truth mask")
        m = (np.asarray(gt) > 0).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
        return m.astype(bool)
    return seg


# --------------------------------------------------------------------------
# SAM family (weights required)
# --------------------------------------------------------------------------

@register("medsam")
def _medsam(model_id: str = "wanglab/medsam-vit-base", device: str = "auto", **_) -> Segmenter:
    """MedSAM: SAM ViT-B fine-tuned on a large medical image/mask corpus, via the
    HF `SamModel` interface (MedSAM keeps SAM's architecture, so the standard
    processor applies).

    Bone-tumour MRI is OUT OF DISTRIBUTION for it -- the training mix is heavily
    CT / pathology / endoscopy, MRI a minority, bone lesions rarer still. Treat a
    weak result as a distribution statement, not a verdict on the approach, and
    always run the `sam2` and nnU-Net comparisons before concluding."""
    return _hf_sam(model_id, device)


@register("sam")
def _sam(model_id: str = "facebook/sam-vit-base", device: str = "auto", **_) -> Segmenter:
    """Vanilla SAM. Worth running next to MedSAM: on OOD anatomy the medical
    fine-tune does not reliably win, and knowing that saves a lot of time."""
    return _hf_sam(model_id, device)


def _hf_sam(model_id: str, device: str) -> Segmenter:
    import torch  # imported lazily so the offline backends need no torch
    from transformers import SamModel, SamProcessor

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("loading %s on %s", model_id, device)
    processor = SamProcessor.from_pretrained(model_id)
    model = SamModel.from_pretrained(model_id).to(device).eval()

    def seg(image: np.ndarray, box: Box) -> np.ndarray:
        x0, y0, x1, y1 = box
        inputs = processor(image, input_boxes=[[[float(x0), float(y0), float(x1), float(y1)]]],
                           return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs, multimask_output=False)
        masks = processor.image_processor.post_process_masks(
            out.pred_masks.cpu(), inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu())
        return masks[0][0][0].numpy().astype(bool)
    return seg


@register("sam2")
def _sam2(model_id: str = "facebook/sam2-hiera-large", device: str = "auto", **_) -> Segmenter:
    """SAM 2. Not medical, but frequently competitive on OOD anatomy precisely
    because everything is OOD, and its cross-frame propagation is the natural way
    to go back to 3D later (prompt one slice, propagate through the volume).

    The SAM 2 Python API has moved between releases; this uses the
    `sam2.sam2_image_predictor` entry point. Verify against your installed
    version before trusting a zero score -- an API mismatch and a genuinely bad
    segmentation look identical in the results CSV."""
    import torch
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("loading %s on %s", model_id, device)
    predictor = SAM2ImagePredictor.from_pretrained(model_id, device=device)

    def seg(image: np.ndarray, box: Box) -> np.ndarray:
        with torch.inference_mode():
            predictor.set_image(image)
            masks, scores, _ = predictor.predict(
                box=np.array(box, dtype=np.float32)[None, :], multimask_output=False)
        return np.asarray(masks[0]).astype(bool)
    return seg
