"""
Read the shape probe's answer as a DISTRIBUTION instead of a parsed string.

Two modes, both single-forward-pass (no autoregressive decoding, so both are
much faster per image than `run_shape_probe.py --mode infer`):

    score   forced-choice log-probabilities of each label, for calibration
    embed   pooled hidden states of the image tokens, for a linear probe

Why this exists
---------------
`run_shape_probe.py` generates text and parses one winner out of it. That
discards the margin: an image where the model put 0.34/0.33/0.33 on the three
classes is recorded identically to one where it put 0.98/0.01/0.01. The 0-shot
and few-shot confusion matrices showed the model emitting `lobulated` for ~70%
of images regardless of ground truth while its `irregular` PRECISION stayed well
above chance -- the signature of a usable signal behind a badly placed decision
threshold. You cannot see that, let alone correct it, from hard labels.

`score` mode fixes the answer prefix ('{"prediction": "') and asks the model for
the likelihood of each label string continuing from there. That yields one row of
log-probs per image, which `calibrate.py` then re-thresholds. `embed` mode asks a
different question -- is the signal in the representation at all, independent of
the model's ability to say so -- by dumping features for an external classifier.

Both modes reuse `run_shape_probe.build_messages`, so the prompt, definitions and
few-shot turns are byte-identical to the generative run by construction.

The thinking caveat
-------------------
MedGemma 1.5 thinks before answering, and forced-choice scoring straight after
the generation prompt skips that, so `score` is NOT the same inference path as
`--mode infer`. Pass `--thinking-from <results.csv>` to replay the `thinking`
column of an existing generative run as the scored prefix; the comparison between
the two tells you how much the chain of thought is actually worth here.

    # forced-choice log-probs, zero-shot, no thinking
    python score_logprobs.py --mode score --model-id /models/medgemma-1.5-4b-it \
        --metadata /results/shape_probe/clinical/shape_metadata.csv \
        --out /results/shape_probe/clinical/logprobs_0shot.csv

    # same, but scored after the thinking block a previous run produced
    python score_logprobs.py --mode score --model-id /models/medgemma-1.5-4b-it \
        --metadata /results/shape_probe/clinical/shape_metadata.csv \
        --thinking-from /results/shape_probe/clinical/probe_results.csv \
        --out /results/shape_probe/clinical/logprobs_0shot_cot.csv

    # features for the linear probe (zero-shot only -- see --mode embed below)
    python score_logprobs.py --mode embed --model-id /models/medgemma-1.5-4b-it \
        --metadata /results/shape_probe/clinical/shape_metadata.csv \
        --out /results/shape_probe/clinical/embeddings.npz

Then score both on CPU with `calibrate.py` (root uv environment -- it needs
scikit-learn, which medgemma_pilot deliberately does not depend on).

Sharding mirrors run_shape_probe.py (--num-shards / --shard-index, strided
split, one process per GPU). `score` output CSVs concatenate; `embed` output
NPZs are combined by passing all of them to calibrate.py.
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "medgemma_pilot"))
import run_medgemma as mg  # noqa: E402

import run_shape_probe as rsp  # noqa: E402

log = logging.getLogger("score_logprobs")

# The answer prefix we force before scoring the label. It is the literal opening
# of the JSON object the prompt asks for, so the model is being scored at exactly
# the position where the generative run would have emitted its label -- not on a
# bare-word continuation it was never asked for.
ANSWER_PREFIX = '{"prediction": "'


# --------------------------------------------------------------------------
# tokenisation of the scored continuations
# --------------------------------------------------------------------------
def continuation_ids(tokenizer, prefix: str, labels: List[str],
                     verbose: bool = True) -> Dict[str, Tuple[List[int], int]]:
    """label -> (token ids for prefix+label, index where the LABEL's tokens start).

    Tokenising `prefix` and `label` separately and concatenating is wrong: a
    SentencePiece-style tokenizer can merge across the boundary (the prefix ends
    in `"` and every label starts with a letter), so the ids you score would not
    be the ids the model would actually produce. Instead tokenise the joined
    string and locate the label by the longest shared prefix with the tokenised
    prefix alone. Any residual boundary effect is a per-class constant and is
    absorbed by the per-class bias `calibrate.py` fits.
    """
    pre = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    out: Dict[str, Tuple[List[int], int]] = {}
    for label in labels:
        full = tokenizer(prefix + label, add_special_tokens=False)["input_ids"]
        shared = 0
        while shared < min(len(pre), len(full)) and pre[shared] == full[shared]:
            shared += 1
        if shared == len(full):
            raise SystemExit(f"label {label!r} contributes no tokens after the prefix; "
                             "check ANSWER_PREFIX")
        out[label] = (full, shared)
        # Silenced with --thinking-from: the prefix is then per-image, so this is
        # called once per image and would emit thousands of identical-looking
        # lines. The shared-prefix case above is logged once at startup instead.
        if verbose:
            log.info("scoring %-11s as %d token(s) from position %d: %r",
                     label, len(full) - shared, shared,
                     tokenizer.convert_ids_to_tokens(full[shared:]))
    return out


# --------------------------------------------------------------------------
# score mode
# --------------------------------------------------------------------------
def score_batch(model, processor, batch_messages: List[List[dict]], labels: List[str],
                conts: Dict[str, Tuple[List[int], int]],
                prefixes: Optional[List[str]] = None) -> np.ndarray:
    """(B prompts) -> (B, len(labels)) summed log-probability of each label.

    One forward pass over a batch expanded to B*len(labels) rows. All rows of a
    prompt share the image and the prompt tokens, and differ only in the
    continuation appended after them.

    Padding: the processor left-pads the prompts (mandatory for this model, see
    mg.load_model), so every row's prompt ends at the same index P and the
    continuation always starts at P. The continuations then differ in length and
    are right-padded -- harmless under causal attention, since a trailing pad
    cannot influence any earlier position and we never read its logits.
    """
    import torch

    if prefixes is not None and len(prefixes) != len(batch_messages):
        raise ValueError("one prefix per prompt required")

    inputs = processor.apply_chat_template(
        batch_messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    )
    prompt_ids = inputs["input_ids"]
    B, P = prompt_ids.shape
    L = len(labels)

    # Per-prompt scored continuation. With --thinking-from, the model's own
    # thinking block is replayed before ANSWER_PREFIX so the label is scored in
    # the same context the generative run had; the thinking tokens themselves are
    # context, never scored.
    per_prompt: List[Dict[str, Tuple[List[int], int]]] = []
    if prefixes is None:
        per_prompt = [conts] * B
    else:
        for pre in prefixes:
            per_prompt.append(
                continuation_ids(processor.tokenizer, pre + ANSWER_PREFIX, labels, verbose=False)
                if pre else conts)

    max_cont = max(len(per_prompt[i][l][0]) for i in range(B) for l in labels)
    pad_id = processor.tokenizer.pad_token_id or 0

    ids = torch.full((B * L, P + max_cont), pad_id, dtype=prompt_ids.dtype)
    mask = torch.zeros((B * L, P + max_cont), dtype=inputs["attention_mask"].dtype)
    ids[:, :P] = prompt_ids.repeat_interleave(L, dim=0)
    mask[:, :P] = inputs["attention_mask"].repeat_interleave(L, dim=0)

    # Gemma-3 marks image positions with token_type_ids == 1; the appended text
    # tokens must be marked 0 or the model mis-routes them through the vision path.
    tti = inputs.get("token_type_ids")
    if tti is not None:
        tti_full = torch.zeros((B * L, P + max_cont), dtype=tti.dtype)
        tti_full[:, :P] = tti.repeat_interleave(L, dim=0)

    spans: List[Tuple[int, int, List[int]]] = []  # (row, start offset in cont, scored ids)
    for i in range(B):
        for j, label in enumerate(labels):
            cont, start = per_prompt[i][label]
            row = i * L + j
            ids[row, P:P + len(cont)] = torch.tensor(cont, dtype=ids.dtype)
            mask[row, P:P + len(cont)] = 1
            spans.append((row, start, cont[start:]))

    kwargs = {"input_ids": ids.to(model.device), "attention_mask": mask.to(model.device)}
    if tti is not None:
        kwargs["token_type_ids"] = tti_full.to(model.device)
    for key in ("pixel_values", "pixel_attention_mask"):
        if key in inputs:
            val = inputs[key]
            # One image entry per prompt row -> repeat along the leading axis so
            # each of a prompt's L rows sees the same image(s).
            kwargs[key] = val.repeat_interleave(L, dim=0).to(model.device, dtype=model.dtype)

    with torch.inference_mode():
        logits = model(**kwargs).logits.float()
    logprobs = torch.log_softmax(logits, dim=-1)

    out = np.zeros((B, L), dtype=np.float64)
    for row, start, scored in spans:
        total = 0.0
        for k, tok in enumerate(scored):
            # logits at index t predict the token AT index t+1.
            total += logprobs[row, P + start + k - 1, tok].item()
        out[row // L, row % L] = total
    return out


def _prefix_map(path: Optional[Path]) -> Dict[str, str]:
    """image_path -> thinking text, from a generative results CSV."""
    if path is None:
        return {}
    df = pd.read_csv(path, dtype=str).fillna("")
    if "thinking" not in df.columns:
        raise SystemExit(f"{path} has no `thinking` column")
    m = {str(r["image_path"]): str(r["thinking"]) for _, r in df.iterrows()}
    log.info("replaying thinking prefixes for %d image(s) from %s", len(m), path)
    return m


# --------------------------------------------------------------------------
# embed mode
# --------------------------------------------------------------------------
def resolve_layers(specs: List[str], n_layers: int) -> List[int]:
    """Layer specs -> indices into `hidden_states`, which has n_layers+1 entries
    (the embedding output first). 'mid' is resolved against the model actually
    loaded rather than hardcoded, since 4B and 27B have different depths.

    Negative indices are Python-style from the end, so -1 is the final layer.
    """
    out: List[int] = []
    for spec in specs:
        out.append(n_layers // 2 if spec == "mid" else int(spec))
    log.info("pooling hidden states at layer index %s (model has %d layers)", out, n_layers)
    return out


def embed_batch(model, processor, batch_messages: List[List[dict]],
                layers: List[int]) -> Dict[str, np.ndarray]:
    """(B prompts) -> pooled hidden states, keyed "<pool>_L<layer>" -> (B, d).

    Two poolings per requested layer:
      image_mean  mean over the query image's token positions -- the most direct
                  read of "what did the vision pathway encode about this image"
      last        the final prompt position, i.e. what the LM head is about to
                  decode from. Comparing the two separates "the features exist"
                  from "the features reached the decision point".
    """
    import torch

    inputs = processor.apply_chat_template(
        batch_messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    )
    kwargs = {k: v.to(model.device, dtype=model.dtype) if v.dtype.is_floating_point
              else v.to(model.device) for k, v in inputs.items()}
    with torch.inference_mode():
        hs = model(**kwargs, output_hidden_states=True).hidden_states

    # Image positions: token_type_ids == 1 on Gemma-3, else the image token id.
    if "token_type_ids" in inputs:
        img_mask = inputs["token_type_ids"].to(model.device) == 1
    else:
        tok_id = getattr(model.config, "image_token_index", None)
        if tok_id is None:
            raise SystemExit("cannot locate image token positions: no token_type_ids and no "
                             "config.image_token_index -- pass --pool last only")
        img_mask = inputs["input_ids"].to(model.device) == tok_id
    if not bool(img_mask.any()):
        raise SystemExit("no image token positions found in the prompt")

    out: Dict[str, np.ndarray] = {}
    for li in layers:
        h = hs[li].float()
        w = img_mask.unsqueeze(-1).to(h.dtype)
        out[f"image_mean_L{li}"] = ((h * w).sum(1) / w.sum(1).clamp(min=1)).cpu().numpy()
        out[f"last_L{li}"] = h[:, -1, :].cpu().numpy()
    return out


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------
def run(mode: str, metadata: Path, out_path: Path, model_id: str, shape_set: str,
        batch_size: int, shard_index: int, num_shards: int, limit: Optional[int],
        num_few_shot: int, few_shot_metadata: Optional[Path], seed: int,
        thinking_from: Optional[Path], layer_specs: List[str]) -> None:
    df = pd.read_csv(metadata, dtype=str).fillna("")
    if limit:
        df = df.head(limit)
    shape_set = rsp.resolve_shape_set(shape_set, df)
    labels = rsp.active_labels(shape_set, df)
    log.info("shape set %r -> %d label(s) %s", shape_set, len(labels), labels)

    if mode == "embed" and num_few_shot > 0:
        # Few-shot puts several images in context, so "the image tokens" is
        # ambiguous and the pooled vector would mix exemplars with the query.
        # The probe measures the representation, not the prompt, so zero-shot is
        # the right condition anyway.
        raise SystemExit("--mode embed is zero-shot only (few-shot pools exemplar "
                         "image tokens into the query vector); drop --num-few-shot")

    few_shot: List[tuple] = []
    if num_few_shot > 0:
        ex_df = pd.read_csv(few_shot_metadata, dtype=str).fillna("") if few_shot_metadata else df
        ex_rows = rsp.select_examples(ex_df, labels, num_few_shot, random.Random(seed))
        few_shot = [(mg.to_jpeg_rgb(r["image_path"]), str(r["shape"])) for r in ex_rows]
        if few_shot_metadata is None:
            ex_paths = {str(r["image_path"]) for r in ex_rows}
            df = df[~df["image_path"].astype(str).isin(ex_paths)]
        log.info("%d-shot per class (%d example turns)", num_few_shot, len(few_shot))

    prefixes_by_path = _prefix_map(thinking_from)

    if num_shards > 1:
        df = df.iloc[shard_index::num_shards]
        log.info("shard %d/%d -> %d row(s)", shard_index, num_shards, len(df))

    model, processor = mg.load_model(model_id)
    conts = continuation_ids(processor.tokenizer, ANSWER_PREFIX, labels)
    layers: List[int] = []
    if mode == "embed":
        text_cfg = getattr(model.config, "text_config", model.config)
        layers = resolve_layers(layer_specs, int(text_cfg.num_hidden_layers))

    rows = list(df.iterrows())
    n_batches = (len(rows) + batch_size - 1) // max(batch_size, 1)
    meta_keep = ["case_id", "image_path", "shape", "background", "difficulty",
                 "shape_params", "modality", "plane", "radius_px"]

    csv_fh = writer = None
    if mode == "score":
        out_path.parent.mkdir(parents=True, exist_ok=True)
        csv_fh = open(out_path, "w", newline="")
        writer = csv.DictWriter(csv_fh, fieldnames=[
            *meta_keep, "shape_set", "num_few_shot", "model_id", "scored_after_thinking",
            *(f"logp_{l}" for l in labels), "argmax_label",
        ])
        writer.writeheader()
    embeds: Dict[str, List[np.ndarray]] = {}
    embed_meta: List[dict] = []

    t0 = time.perf_counter()
    for bi in range(n_batches):
        chunk = [r for _, r in rows[bi * batch_size:(bi + 1) * batch_size]]
        batch_messages, batch_rows = [], []
        for row in chunk:
            try:
                image = mg.to_jpeg_rgb(row["image_path"])
            except Exception as e:  # noqa: BLE001
                log.warning("skip image %s: %s", row["image_path"], e)
                continue
            batch_messages.append(rsp.build_messages(
                image, row.get("modality", ""), row.get("plane", ""),
                shape_set=shape_set, few_shot=few_shot, labels=labels))
            batch_rows.append(row)
        if not batch_messages:
            continue

        if mode == "score":
            prefixes = None
            if prefixes_by_path:
                prefixes = [prefixes_by_path.get(str(r["image_path"]), "") for r in batch_rows]
            lp = score_batch(model, processor, batch_messages, labels, conts, prefixes)
            for row, scores in zip(batch_rows, lp):
                rec = {k: row.get(k, "") for k in meta_keep}
                rec.update(shape_set=shape_set, num_few_shot=num_few_shot, model_id=model_id,
                           scored_after_thinking=int(bool(prefixes_by_path)),
                           argmax_label=labels[int(np.argmax(scores))])
                rec.update({f"logp_{l}": f"{s:.6f}" for l, s in zip(labels, scores)})
                writer.writerow(rec)
            csv_fh.flush()
        else:
            pooled = embed_batch(model, processor, batch_messages, layers)
            for key, arr in pooled.items():
                embeds.setdefault(key, []).append(arr)
            embed_meta.extend({k: str(r.get(k, "")) for k in meta_keep} for r in batch_rows)

        rate = (bi + 1) * batch_size / max(time.perf_counter() - t0, 1e-9)
        log.info("batch %d/%d  %.2f img/s  ~%.1f min left",
                 bi + 1, n_batches, rate, (len(rows) - (bi + 1) * batch_size) / rate / 60)

    if csv_fh:
        csv_fh.close()
    if mode == "embed":
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, **{k: np.concatenate(v) for k, v in embeds.items()},
                            labels=np.array(labels))
        pd.DataFrame(embed_meta).to_csv(out_path.with_suffix(".meta.csv"), index=False)
        log.info("done -> %s (+ .meta.csv)", out_path)
    else:
        log.info("done -> %s", out_path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["score", "embed"], required=True)
    p.add_argument("--metadata", type=Path, required=True, help="shape_metadata.csv from build_shapes.py")
    p.add_argument("--out", type=Path, required=True, help="CSV for score mode, NPZ for embed mode")
    p.add_argument("--model-id", default="/models/medgemma-1.5-4b-it")
    p.add_argument("--shape-set", default="auto")
    p.add_argument("--batch-size", type=int, default=8,
                   help="prompts per forward pass. score mode expands this by the number of "
                        "labels internally, so peak memory is batch_size * n_labels rows")
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--num-few-shot", type=int, default=0, help="score mode only")
    p.add_argument("--few-shot-metadata", type=Path, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--thinking-from", type=Path, default=None,
                   help="generative results CSV whose `thinking` column is replayed as the "
                        "scored context, so the label is scored on the same path --mode infer used")
    p.add_argument("--layers", default="-1,mid",
                   help="embed mode: comma-separated hidden-state layers; 'mid' = n_layers//2")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    out = args.out
    if args.num_shards > 1:
        out = out.with_suffix(f".shard{args.shard_index}{out.suffix}")

    run(args.mode, args.metadata, out, args.model_id, args.shape_set, args.batch_size,
        args.shard_index, args.num_shards, args.limit, args.num_few_shot,
        args.few_shot_metadata, args.seed, args.thinking_from,
        [t.strip() for t in str(args.layers).split(",") if t.strip()])


if __name__ == "__main__":
    main()
