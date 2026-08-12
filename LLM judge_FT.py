"""
LLM-as-a-Judge evaluation for bridge defect repair recommendations.

Complements the automated cosine-similarity metric with a binary
technical-equivalence judgement from three independent open-source models
(Qwen2.5-7B-Instruct, Mistral-7B-Instruct-v0.3, Gemma-2-9B-it), none of
which is used anywhere in the pipeline being evaluated (fine-tuning bases:
LLaMA-3-8B / Phi-3; RAG generator: LLaMA-3-8B). Three families rather than
one guards against a single model's idiosyncrasies (a specific verb it
under-weighs, a template quirk) being mistaken for a property of the task;
where the three disagree, that disagreement is itself reported rather than
resolved by picking a favourite.

The judge decides only whether the generated text describes the SAME repair
action as the inspector-approved reference. It does not assess whether the
reference repair is itself appropriate: the reference is the standard of
correctness by definition. This keeps the task to semantic equivalence,
which is what the cosine metric fails at, rather than engineering adequacy.

Input : *_scores.csv produced by rag_generatedResponseEvaluation.py
        columns: cluster_id, user_input, reference_response,
                 generated_response, cosine_similarity, above_threshold
Output: for each judge model (tag in JUDGE_MODELS), one *_judged.csv per
                 input file under --out/<model_tag>/, with every input
                 column plus judge_verdict, judge_reason, reference_action,
                 proposed_action, reference_component, proposed_component,
                 judge_source, cosine_accept, judge_equivalent, agreement.
                 --out/<model_tag>/judge_summary.csv holds that model's
                 per-file agreement statistics against cosine.
        --out/combined/*_combined.csv: the three models' verdicts side by
                 side per row, plus majority_verdict (2-of-3, or "TIE" /
                 "PARSE_ERROR" when it cannot be formed) and agreement
                 with cosine using the majority.
        --out/combined/judge_summary_combined.csv: per-file majority-vs-
                 cosine statistics, pairwise Cohen's kappa between every
                 pair of models, and Fleiss' kappa across all three - the
                 inter-judge reliability figures for the paper.
Every row of every input file is judged; there is no row cap.

Each (model, input file) pair gets its own output file, written row by row
as it is judged (not held in memory until the file finishes) and only
given its final name once the whole file is done, so an interrupted run
can be resumed (rerun the same command; finished (model, file) pairs are
skipped unless --overwrite is passed). The combine step always re-scans
--out for whichever (model, file) outputs currently exist on disk, so
models can be run in separate invocations (even --models qwen2.5-7b today,
--models gemma-2-9b next week) and still combine correctly once complete.

Notes on validity
-----------------
* The judge never sees which pipeline (RAG / fine-tuning), which embedding
  model, or which seed produced a response: only the defect, the reference
  repair, and the candidate repair.
* Greedy decoding (do_sample=False) so judgements are reproducible.
* Order is fixed (reference first, candidate second) and stated explicitly
  in the prompt, so there is no position ambiguity to exploit.
* The prompt is a single user turn (no separate system role) so the exact
  same text is given verbatim to all three models: Gemma-2's chat template
  has no system role and enforces strict user/model alternation, and older
  Mistral templates are picky about a system turn too. Folding the system
  instructions into the one user turn sidesteps both and keeps the
  comparison fair (same words, different model).
* The cosine decision is taken from the above_threshold column written by
  the evaluation script, not recomputed here, so the two metrics cannot
  drift apart through a duplicated threshold constant.
* Items whose reference and candidate are the same string after whitespace,
  case and trailing-punctuation normalisation are recorded as EQUIVALENT
  without a model call: identity is not a judgement. The judge_source
  column records how each verdict was obtained so this is auditable.
* Equivalence requires an exact match of the repair ACTION (verb) and the
  COMPONENT (noun); no added, changed or omitted detail is tolerated. The
  judge is asked to name both terms for reference and proposed separately
  before deciding, and they are written to reference_action, proposed_action,
  reference_component, proposed_component so the exact-match rule can be
  audited rather than taken on trust.
* Model weights are cached under ./.cache (next to this script, on the same
  drive as the rest of the project), not the default user-profile HF cache,
  because three 7-9B models do not fit on a small system drive. Do not move
  an already-downloaded cache folder across drives with a plain file move/
  robocopy /MOVE: the HF cache uses NTFS hardlinks between snapshots/ and
  blobs/, and a cross-volume move silently breaks them into 0-byte files
  while reporting success. Deleting and re-downloading is the safe fix.
"""

import argparse
import csv
import gc
import json
import os
import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
JUDGE_MODELS = {
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "gemma-2-9b": "google/gemma-2-9b-it",
}
# Gemma-2's attention logit softcapping isn't implemented for sdpa in some
# transformers versions and silently degrades output; eager is the model
# card's documented workaround.
MODEL_LOAD_KWARGS = {
    "gemma-2-9b": {"attn_implementation": "eager"},
}
CACHE_DIR = str(Path(__file__).resolve().parent / ".cache")

MAX_NEW_TOKENS = 160         # verdict + reason + 4 extracted term fields
COSINE_THRESHOLD = 0.7       # fallback only, for files with no above_threshold column

REQUIRED_COLUMNS = ("user_input", "reference_response", "generated_response")

SYSTEM_PROMPT = (
    "You are a bridge maintenance engineer comparing two descriptions of a "
    "repair. You decide whether they specify the same repair action on the "
    "same component. Wording and ordering may differ, but the action verb, "
    "the component noun, and every requirement the reference states must "
    "match exactly: nothing may be added, changed or left out. You do not "
    "judge whether the reference repair is itself appropriate: it is the "
    "approved standard."
)

USER_TEMPLATE = """Decide whether the PROPOSED repair describes the same repair action as the REFERENCE repair.

DEFECT:
{defect}

REFERENCE REPAIR (inspector-approved, treat as the standard):
{reference}

PROPOSED REPAIR (to be compared):
{candidate}

Rules:
1. Break REFERENCE and PROPOSED into their individual repair steps (most descriptions are one step; some are two or more, usually joined by "and"). For each step identify the ACTION (the verb describing what is physically done, e.g. cut out, grind back, weld, replace, drill, install) and the COMPONENT (the noun naming the part being repaired, e.g. corroded section, flange, angle, plate, stiffener). If either text has more than one step, list all of them in the action/component fields, separated by "; ".
2. Mark EQUIVALENT only if REFERENCE and PROPOSED have the same number of steps and every step's ACTION and COMPONENT match exactly between them. A different verb (for example "grind back" where the reference says "cut out", or "install" where it says "weld") is NOT_EQUIVALENT even if the overall repair sounds similar. A differently named component is NOT_EQUIVALENT even if it is nearby or related.
3. Mark NOT_EQUIVALENT if PROPOSED has a step that REFERENCE does not have (an added step, action, material or detail), even if it does not contradict the reference. Extra completeness is not rewarded: the proposal must say neither more nor less than the reference.
4. Mark NOT_EQUIVALENT if PROPOSED is missing a step that REFERENCE states explicitly, or drops a requirement of a step (such as a dimension or material).
5. Ignore text that carries no repair meaning: a leading label such as "Finding 12:", trailing punctuation, and capitalisation are not part of the comparison.
6. Mark NOT_EQUIVALENT if PROPOSED is empty, a placeholder, or otherwise does not contain an identifiable repair step (for example a bare number or punctuation). Do not invent an action or component that is not actually written in the text.
7. Do not assess whether the reference repair is correct or appropriate. It is the standard by definition.
8. Base your decision only on the text given.

Respond with strict JSON and nothing else:
{{"reference_action": "<verb phrase>", "proposed_action": "<verb phrase>", "reference_component": "<noun phrase>", "proposed_component": "<noun phrase>", "verdict": "EQUIVALENT" or "NOT_EQUIVALENT", "reason": "<one short sentence>"}}"""


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------
def load_judge(model_name: str):
    print(f"Loading judge model: {model_name}")
    extra = MODEL_LOAD_KWARGS.get(
        next((tag for tag, name in JUDGE_MODELS.items() if name == model_name), ""), {}
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=CACHE_DIR,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
        **extra,
    )
    model.eval()
    # Gemma-2's shipped generation_config defaults to a hybrid/static cache,
    # which transformers tries to torch.compile for speed; that needs Triton,
    # which isn't reliably available on Windows. Forcing the plain dynamic
    # cache skips that path entirely (slower, but correct everywhere).
    model.generation_config.cache_implementation = None
    return tokenizer, model


def unload_judge(tokenizer, model):
    """Free GPU memory before loading the next judge; three 7-9B models
    do not fit at once, so models are run one at a time, not interleaved."""
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


TERM_FIELDS = ("reference_action", "proposed_action", "reference_component", "proposed_component")
EMPTY_TERMS = {k: "" for k in TERM_FIELDS}


def build_prompt(defect: str, reference: str, candidate: str, retry: bool = False) -> str:
    text = SYSTEM_PROMPT + "\n\n" + USER_TEMPLATE.format(
        defect=defect.strip(), reference=reference.strip(), candidate=candidate.strip(),
    )
    if retry:
        text += "\n\nReply with the JSON object only, nothing else."
    return text


@torch.inference_mode()
def judge_one(tokenizer, model, defect: str, reference: str, candidate: str):
    """Return (verdict, reason, terms). Greedy decoding for reproducibility.

    Always exactly one user turn, no system role: Gemma-2's chat template
    has no system role and enforces strict user/model alternation, and it
    is simplest and fairest to give all three judge models the identical
    single block of text rather than special-case each model's template.
    """
    verdict, reason, terms = "PARSE_ERROR", "", EMPTY_TERMS
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    # One retry: a stricter reminder recovers the occasional prose reply
    # without changing the decision rule.
    for attempt in range(2):
        messages = [{"role": "user", "content": build_prompt(defect, reference, candidate, retry=bool(attempt))}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer([text], return_tensors="pt").to(model.device)

        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,                      # deterministic
            # Some chat models ship sampling defaults in generation_config;
            # clear them so nothing but greedy decoding is in play.
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=pad_id,
        )
        reply = tokenizer.decode(
            out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        ).strip()

        verdict, reason, terms = parse_verdict(reply)
        if verdict != "PARSE_ERROR":
            break
    return verdict, reason, terms


def normalise(text: str) -> str:
    """Casing, whitespace and trailing punctuation carry no repair meaning."""
    return re.sub(r"\s+", " ", str(text or "")).strip().lower().rstrip(".;,")


def is_degenerate(text: str) -> bool:
    """True for empty or content-free text (e.g. a bare '1.' from a failed
    generation). Checked deterministically rather than left to the model:
    on this corpus the model has hallucinated a matching action/component
    for such text instead of reporting NOT_EQUIVALENT."""
    return len(re.sub(r"[^a-zA-Z]", "", str(text or ""))) < 3


def judge_cached(tokenizer, model, defect, reference, candidate, cache):
    """Judge a triple, reusing verdicts for text seen before.

    Returns (verdict, reason, terms, source) where source is exact / cache / model.
    The sweep repeats the same reference/candidate pairs across seeds,
    embedding models and datasets, so this removes most of the model calls.
    """
    ref_n, cand_n = normalise(reference), normalise(candidate)
    if ref_n and ref_n == cand_n:
        return "EQUIVALENT", "Identical text after normalisation.", EMPTY_TERMS, "exact"
    if is_degenerate(candidate) or is_degenerate(reference):
        return "NOT_EQUIVALENT", "Generated or reference text has no identifiable repair content.", EMPTY_TERMS, "degenerate"

    key = (normalise(defect), ref_n, cand_n)
    if key in cache:
        verdict, reason, terms = cache[key]
        return verdict, reason, terms, "cache"

    verdict, reason, terms = judge_one(tokenizer, model, defect, reference, candidate)
    if verdict != "PARSE_ERROR":            # never cache a failure
        cache[key] = (verdict, reason, terms)
    return verdict, reason, terms, "model"


def parse_verdict(reply: str):
    """Extract verdict/reason/terms, tolerating minor format drift."""
    m = re.search(r"\{.*\}", reply, re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            v = str(obj.get("verdict", "")).strip().upper().replace(" ", "_").replace("-", "_")
            if v in ("EQUIVALENT", "NOT_EQUIVALENT"):
                terms = {k: str(obj.get(k, "")).strip() for k in TERM_FIELDS}
                return v, str(obj.get("reason", "")).strip(), terms
        except json.JSONDecodeError:
            pass
    # fallback: look for the bare label (check negative form first)
    up = reply.upper().replace(" ", "_").replace("-", "_")
    if "NOT_EQUIVALENT" in up:
        return "NOT_EQUIVALENT", reply[:160], EMPTY_TERMS
    if "EQUIVALENT" in up:
        return "EQUIVALENT", reply[:160], EMPTY_TERMS
    return "PARSE_ERROR", reply[:160], EMPTY_TERMS


# ----------------------------------------------------------------------
# File processing
# ----------------------------------------------------------------------
def cosine_decision(row: dict, threshold: float):
    """The accept/reject already recorded by the evaluation script.

    Prefers the above_threshold column so the comparison uses the exact
    decision the pipeline made; falls back to the cosine score for older
    files that predate the column. Returns None if neither is usable.
    """
    raw = str(row.get("above_threshold", "")).strip().lower()
    if raw in ("true", "1", "yes"):
        return True
    if raw in ("false", "0", "no"):
        return False
    try:
        return float(row["cosine_similarity"]) >= threshold
    except (KeyError, TypeError, ValueError):
        return None


def cohens_kappa(a: int, b: int, c: int, d: int):
    """Chance-corrected agreement for the 2x2 table (a=both accept ... d=both reject).

    Raw agreement flatters both metrics when one verdict dominates, so the
    paper needs kappa alongside it.
    """
    n = a + b + c + d
    if not n:
        return 0.0
    po = (a + d) / n
    pe = ((a + b) * (a + c) + (c + d) * (b + d)) / (n * n)
    return round((po - pe) / (1 - pe), 4) if pe != 1 else 1.0


ADDED_FIELDS = ("judge_verdict", "judge_reason", *TERM_FIELDS,
                "judge_source", "cosine_accept", "judge_equivalent", "agreement")


def process_file(path: Path, tokenizer, model, out_path: Path, label: str,
                 threshold: float, cache: dict):
    """Judge every row of one *_scores.csv and write its own *_judged.csv.

    Rows are written to disk as soon as each is judged (not buffered until
    the file finishes), so a crash partway through a long file does not
    discard the rows already judged. The result is built under a .part
    name and only renamed to out_path once every row has been written, so
    an interrupted file is never mistaken for a finished one by --resume.
    """
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
    if not rows:
        print(f"  (empty) {path.name}")
        return None

    missing = [c for c in REQUIRED_COLUMNS if c not in rows[0]]
    if missing:
        print(f"  (skipped) {label}: missing column(s) {', '.join(missing)}")
        return None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = out_path.with_suffix(out_path.suffix + ".part")
    fieldnames = list(rows[0].keys()) + list(ADDED_FIELDS)

    judged = []
    with part_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in tqdm(rows, desc=f"  {label}", leave=False):
            verdict, reason, terms, source = judge_cached(
                tokenizer, model,
                r["user_input"], r["reference_response"], r["generated_response"],
                cache,
            )
            cosine_ok = cosine_decision(r, threshold)
            judge_ok = (verdict == "EQUIVALENT")

            r = dict(r)
            r["judge_verdict"] = verdict
            r["judge_reason"] = reason
            for k in TERM_FIELDS:
                r[k] = terms[k]
            r["judge_source"] = source
            r["cosine_accept"] = cosine_ok
            r["judge_equivalent"] = judge_ok
            r["agreement"] = (cosine_ok == judge_ok) if cosine_ok is not None else ""
            judged.append(r)
            w.writerow(r)
            f.flush()

    part_path.replace(out_path)  # atomic on the same filesystem
    return summarise(judged, label)


def summarise(judged: list, label: str):
    """Per-file counts. Rows the judge or the cosine metric could not score
    are excluded from the comparison but still reported."""
    n = len(judged)
    valid = [r for r in judged
             if r["judge_verdict"] != "PARSE_ERROR" and r["cosine_accept"] is not None]
    n_valid = len(valid)

    # 2x2 table: cosine decision against judge verdict
    both = sum(1 for r in valid if r["cosine_accept"] and r["judge_equivalent"])
    cos_only = sum(1 for r in valid if r["cosine_accept"] and not r["judge_equivalent"])
    jud_only = sum(1 for r in valid if not r["cosine_accept"] and r["judge_equivalent"])
    neither = sum(1 for r in valid if not r["cosine_accept"] and not r["judge_equivalent"])

    pct = lambda k: round(100 * k / n_valid, 2) if n_valid else 0.0
    stats = {
        "file": label,
        "n": n,
        "n_compared": n_valid,
        "n_parse_error": sum(1 for r in judged if r["judge_verdict"] == "PARSE_ERROR"),
        "n_model_calls": sum(1 for r in judged if r["judge_source"] == "model"),
        "cosine_accept_rate": pct(both + cos_only),
        "judge_equivalent_rate": pct(both + jud_only),
        "agreement_rate": pct(both + neither),
        "kappa": cohens_kappa(both, cos_only, jud_only, neither),
        "both_accept": both,
        "both_reject": neither,
        "cosine_too_strict": jud_only,    # judge says equivalent, cosine rejects
        "cosine_too_lenient": cos_only,   # cosine accepts, judge says not equivalent
    }
    print(f"  {label}: cosine {stats['cosine_accept_rate']}% | "
          f"judge {stats['judge_equivalent_rate']}% | "
          f"agreement {stats['agreement_rate']}% (kappa {stats['kappa']}) "
          f"(strict {jud_only}, lenient {cos_only})")
    return stats


def collect_files(inputs):
    """Directories are searched recursively: the sweep writes its *_scores.csv
    into per-experiment subfolders, so a top-level glob finds nothing.
    Explicit file arguments are taken as given."""
    files = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            files.extend(sorted(p.rglob("*_scores.csv")))
        elif p.is_file():
            files.append(p)
        else:
            print(f"  (not found) {item}")
    seen, unique = set(), []
    for f in files:
        key = f.resolve()
        if key not in seen:
            seen.add(key)
            unique.append(f)
    return unique


def output_label(path: Path, roots):
    """Stems repeat across experiment folders (52 distinct stems over 154
    files), so include the parent directories to keep outputs distinct."""
    for root in roots:
        try:
            rel = path.resolve().relative_to(root)
        except ValueError:
            continue
        return "__".join(rel.with_suffix("").parts)
    return "__".join(path.resolve().parts[-2:]).removesuffix(".csv")


def print_pooled(all_stats: list, header: str):
    """Aggregate a list of per-file 2x2-table stats dicts (each having
    both_accept/both_reject/cosine_too_strict/cosine_too_lenient) from raw
    counts, not by averaging already-rounded per-file percentages."""
    both = sum(s["both_accept"] for s in all_stats)
    neither = sum(s["both_reject"] for s in all_stats)
    strict = sum(s["cosine_too_strict"] for s in all_stats)
    lenient = sum(s["cosine_too_lenient"] for s in all_stats)
    tot = both + neither + strict + lenient
    pct = lambda k: 100 * k / tot if tot else 0.0

    print("\n" + "=" * 62)
    print(header)
    print("=" * 62)
    print(f"  Items compared          : {tot}")
    print(f"  Cosine acceptance rate  : {pct(both + lenient):.2f}%")
    print(f"  Judge equivalence rate  : {pct(both + strict):.2f}%")
    print(f"  Agreement rate          : {pct(both + neither):.2f}%")
    print(f"  Cohen's kappa           : {cohens_kappa(both, lenient, strict, neither)}")
    print(f"  Cosine too strict       : {strict}  (judge: equivalent, cosine: reject)")
    print(f"  Cosine too lenient      : {lenient}  (cosine: accept, judge: not equivalent)")


def run_model_sweep(tag: str, repo_id: str, files: list, roots: list,
                    out_dir: Path, threshold: float, overwrite: bool):
    """Run one judge model over every file not yet done for it, writing to
    out_dir/<tag>/. Models are loaded one at a time (never two at once):
    a 7-9B model in bf16 plus KV cache is 15-20GB, and three at once would
    not reliably fit even a 48GB GPU."""
    model_out_dir = out_dir / tag
    todo = []
    for f in files:
        label = output_label(f, roots)
        out_path = model_out_dir / f"{label}_judged.csv"
        stale_part = out_path.with_suffix(out_path.suffix + ".part")
        if stale_part.exists():
            stale_part.unlink()   # leftover from an interrupted previous run
        if out_path.exists() and not overwrite:
            print(f"  [{tag}] (done already) {label}")
            continue
        todo.append((f, label, out_path))
    if not todo:
        print(f"[{tag}] nothing to do, every output already exists.")
        return

    tokenizer, model = load_judge(repo_id)

    model_out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = model_out_dir / "judge_summary.csv"
    summary_exists = summary_path.exists() and not overwrite

    cache, all_stats = {}, []
    for f, label, out_path in todo:
        s = process_file(f, tokenizer, model, out_path, label, threshold, cache)
        if s:
            all_stats.append(s)
            # Written after every file, not just at the end, so the summary
            # for files already finished survives a crash on a later one.
            with summary_path.open(
                "a" if summary_exists else "w", newline="", encoding="utf-8"
            ) as fh:
                w = csv.DictWriter(fh, fieldnames=list(s.keys()))
                if not summary_exists:
                    w.writeheader()
                    summary_exists = True
                w.writerow(s)

    unload_judge(tokenizer, model)

    if all_stats:
        errors = sum(s["n_parse_error"] for s in all_stats)
        calls = sum(s["n_model_calls"] for s in all_stats)
        print_pooled(all_stats, f"POOLED RESULTS: {tag}")
        print(f"  Unparsed judgements     : {errors}")
        print(f"  Model calls made        : {calls}")
        print(f"\n  Summary written to: {summary_path}")


def fleiss_kappa(vote_rows: list):
    """Chance-corrected agreement across N>=2 raters and any number of
    categories (Fleiss 1971). vote_rows is a list of same-length tuples,
    one tuple of category labels per item, one label per rater. Cohen's
    kappa only handles two raters; this is the three-judge equivalent
    reported for the paper's inter-judge reliability figure."""
    if not vote_rows:
        return 0.0
    n_raters = len(vote_rows[0])
    categories = sorted(set(v for row in vote_rows for v in row))
    if len(categories) < 2:
        return 1.0
    idx = {c: i for i, c in enumerate(categories)}
    n_items = len(vote_rows)
    col_totals = [0] * len(categories)
    item_agreement = []
    for row in vote_rows:
        counts = [0] * len(categories)
        for v in row:
            counts[idx[v]] += 1
        for j, c in enumerate(counts):
            col_totals[j] += c
        item_agreement.append((sum(c * c for c in counts) - n_raters) / (n_raters * (n_raters - 1)))
    p_bar = sum(item_agreement) / n_items
    p_j = [t / (n_items * n_raters) for t in col_totals]
    p_e = sum(p * p for p in p_j)
    return round((p_bar - p_e) / (1 - p_e), 4) if p_e < 1 else 1.0


def combine_results(out_dir: Path, files: list, roots: list, threshold: float):
    """Re-scan out_dir for every judge model's output (whichever models
    have actually finished, regardless of which --models were requested
    in this invocation) and merge them per source file: one *_combined.csv
    with all three verdicts side by side plus a majority verdict, and
    pooled inter-judge reliability stats (pairwise Cohen's kappa, Fleiss'
    kappa across all three).
    """
    tags = list(JUDGE_MODELS.keys())
    combined_dir = out_dir / "combined"
    pairs = [(tags[i], tags[j]) for i in range(len(tags)) for j in range(i + 1, len(tags))]

    file_stats = []
    pooled_maj = [0, 0, 0, 0]         # both_accept, cosine_only, judge_only, neither
    pooled_pairs = {p: [0, 0, 0, 0] for p in pairs}
    pooled_fleiss_votes = []

    for f in files:
        label = output_label(f, roots)
        paths = {tag: out_dir / tag / f"{label}_judged.csv" for tag in tags}
        missing = [tag for tag, p in paths.items() if not p.exists()]
        if missing:
            print(f"  (combine skipped) {label}: waiting on {', '.join(missing)}")
            continue

        data = {tag: list(csv.DictReader(p.open(encoding="utf-8"))) for tag, p in paths.items()}
        lengths = {tag: len(rows) for tag, rows in data.items()}
        if len(set(lengths.values())) != 1:
            print(f"  (combine skipped) {label}: row count mismatch {lengths}")
            continue
        n = lengths[tags[0]]
        if n == 0:
            continue

        base_cols = [c for c in data[tags[0]][0].keys() if c not in ADDED_FIELDS]
        pair_counts = {p: [0, 0, 0, 0] for p in pairs}
        fleiss_votes = []
        maj_counts = [0, 0, 0, 0]
        combined_rows = []

        for i in range(n):
            base_row = {c: data[tags[0]][i][c] for c in base_cols}
            verdicts = {tag: data[tag][i]["judge_verdict"] for tag in tags}
            merged = dict(base_row)
            for tag in tags:
                r = data[tag][i]
                merged[f"{tag}_verdict"] = r["judge_verdict"]
                merged[f"{tag}_reason"] = r["judge_reason"]
                merged[f"{tag}_source"] = r["judge_source"]

            valid = [v for v in verdicts.values() if v != "PARSE_ERROR"]
            n_equiv = sum(1 for v in valid if v == "EQUIVALENT")
            if not valid:
                majority = "PARSE_ERROR"
            elif n_equiv * 2 > len(valid):
                majority = "EQUIVALENT"
            elif n_equiv * 2 < len(valid):
                majority = "NOT_EQUIVALENT"
            else:
                majority = "TIE"   # only possible when one judge PARSE_ERRORed, leaving 2 valid votes 1-1

            cosine_ok = cosine_decision(base_row, threshold)
            agree = ""
            if majority in ("EQUIVALENT", "NOT_EQUIVALENT") and cosine_ok is not None:
                judge_ok = (majority == "EQUIVALENT")
                agree = (cosine_ok == judge_ok)
                if cosine_ok and judge_ok: maj_counts[0] += 1
                elif cosine_ok and not judge_ok: maj_counts[1] += 1
                elif not cosine_ok and judge_ok: maj_counts[2] += 1
                else: maj_counts[3] += 1

            merged["n_valid_judges"] = len(valid)
            merged["n_equivalent"] = n_equiv
            merged["majority_verdict"] = majority
            merged["cosine_accept"] = cosine_ok
            merged["agreement"] = agree
            combined_rows.append(merged)

            for (t1, t2) in pairs:
                v1, v2 = verdicts[t1], verdicts[t2]
                if v1 == "PARSE_ERROR" or v2 == "PARSE_ERROR":
                    continue
                c = pair_counts[(t1, t2)]
                if v1 == "EQUIVALENT" and v2 == "EQUIVALENT": c[0] += 1
                elif v1 == "EQUIVALENT" and v2 == "NOT_EQUIVALENT": c[1] += 1
                elif v1 == "NOT_EQUIVALENT" and v2 == "EQUIVALENT": c[2] += 1
                else: c[3] += 1

            if len(valid) == len(tags):
                fleiss_votes.append(tuple(verdicts[tag] for tag in tags))

        combined_dir.mkdir(parents=True, exist_ok=True)
        with (combined_dir / f"{label}_combined.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(combined_rows[0].keys()))
            w.writeheader()
            w.writerows(combined_rows)

        both, cos_only, jud_only, neither = maj_counts
        n_valid_maj = sum(maj_counts)
        pct = lambda k, tot: round(100 * k / tot, 2) if tot else 0.0
        stats = {
            "file": label,
            "n": n,
            "n_compared": n_valid_maj,
            "cosine_accept_rate": pct(both + cos_only, n_valid_maj),
            "majority_equivalent_rate": pct(both + jud_only, n_valid_maj),
            "agreement_rate": pct(both + neither, n_valid_maj),
            "majority_kappa": cohens_kappa(both, cos_only, jud_only, neither),
            "fleiss_kappa": fleiss_kappa(fleiss_votes),
        }
        for (t1, t2) in pairs:
            a, b, c, d = pair_counts[(t1, t2)]
            stats[f"agree_{t1}_vs_{t2}"] = pct(a + d, a + b + c + d)
            stats[f"kappa_{t1}_vs_{t2}"] = cohens_kappa(a, b, c, d)
        file_stats.append(stats)
        print(f"  {label}: majority {stats['agreement_rate']}% agreement with cosine "
              f"(kappa {stats['majority_kappa']}), Fleiss kappa {stats['fleiss_kappa']}")

        for i in range(4):
            pooled_maj[i] += maj_counts[i]
        for p in pairs:
            for i in range(4):
                pooled_pairs[p][i] += pair_counts[p][i]
        pooled_fleiss_votes.extend(fleiss_votes)

    if not file_stats:
        return None, None

    both, cos_only, jud_only, neither = pooled_maj
    tot = sum(pooled_maj)
    pct = lambda k: 100 * k / tot if tot else 0.0
    pooled = {
        "cosine_accept_rate": round(pct(both + cos_only), 2),
        "majority_equivalent_rate": round(pct(both + jud_only), 2),
        "agreement_rate": round(pct(both + neither), 2),
        "majority_kappa": cohens_kappa(both, cos_only, jud_only, neither),
        "fleiss_kappa": fleiss_kappa(pooled_fleiss_votes),
        "n_compared": tot,
        "pairs": {
            p: {"agree": round(100 * (c[0] + c[3]) / sum(c), 2) if sum(c) else 0.0,
                "kappa": cohens_kappa(*c)}
            for p, c in pooled_pairs.items()
        },
    }
    return file_stats, pooled


def main():
    ap = argparse.ArgumentParser(description="LLM-as-a-judge evaluation with three independent judge models")
    # Optional so the file can be run straight from the IDE with no arguments:
    # the default walks the project and picks up every *_scores.csv.
    ap.add_argument("inputs", nargs="*", default=["."],
                    help="one or more *_scores.csv files, or directories "
                         "containing them (default: the current directory)")
    ap.add_argument("--out", default="judge_results", help="output directory")
    ap.add_argument("--models", default="all",
                    help=f"comma-separated judge tags to run this invocation, or 'all' "
                         f"(default). Available: {', '.join(JUDGE_MODELS)}. The combine "
                         f"step always uses whichever of the three have finished, even "
                         f"from earlier invocations with a different --models.")
    ap.add_argument("--threshold", type=float, default=COSINE_THRESHOLD,
                    help="fallback cosine threshold if above_threshold is absent")
    ap.add_argument("--overwrite", action="store_true",
                    help="re-judge files that already have a _judged.csv")
    ap.add_argument("--skip-combine", action="store_true",
                    help="only run the judge model(s), skip the cross-model combine step")
    args = ap.parse_args()

    if args.models == "all":
        selected = list(JUDGE_MODELS)
    else:
        selected = [t.strip() for t in args.models.split(",") if t.strip()]
        unknown = [t for t in selected if t not in JUDGE_MODELS]
        if unknown:
            sys.exit(f"Unknown model tag(s): {', '.join(unknown)}. "
                     f"Available: {', '.join(JUDGE_MODELS)}")

    inputs = args.inputs or ["."]
    files = collect_files(inputs)
    if not files:
        sys.exit("No *_scores.csv files found.")

    roots = [Path(i).resolve() for i in inputs if Path(i).is_dir()]
    out_dir = Path(args.out)
    # abspath, not resolve(): on Windows resolve() leaves a not-yet-created
    # directory as a bare relative name.
    print(f"Searching: {', '.join(os.path.abspath(i) for i in inputs)}")
    print(f"Found {len(files)} file(s). Every row of every file is judged, "
          f"no row cap. Judge model(s) this run: {', '.join(selected)}. "
          f"Output: {os.path.abspath(out_dir)}")

    for tag in selected:
        print(f"\n--- {tag} ({JUDGE_MODELS[tag]}) ---")
        run_model_sweep(tag, JUDGE_MODELS[tag], files, roots, out_dir,
                        args.threshold, args.overwrite)

    if args.skip_combine:
        return

    print("\n--- combining judge models ---")
    file_stats, pooled = combine_results(out_dir, files, roots, args.threshold)
    if not file_stats:
        print("  Nothing to combine yet (not every judge model has finished these files).")
        return

    combined_dir = out_dir / "combined"
    summary_path = combined_dir / "judge_summary_combined.csv"
    flat_rows = [{k: v for k, v in s.items()} for s in file_stats]
    with summary_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(flat_rows[0].keys()))
        w.writeheader()
        w.writerows(flat_rows)

    print("\n" + "=" * 62)
    print("POOLED RESULTS: majority vote (2-of-3) vs cosine")
    print("=" * 62)
    print(f"  Items compared          : {pooled['n_compared']}")
    print(f"  Cosine acceptance rate  : {pooled['cosine_accept_rate']}%")
    print(f"  Majority equivalence rate: {pooled['majority_equivalent_rate']}%")
    print(f"  Agreement rate          : {pooled['agreement_rate']}%")
    print(f"  Cohen's kappa           : {pooled['majority_kappa']}")
    print(f"\n  Inter-judge reliability (pairwise Cohen's kappa):")
    for (t1, t2), s in pooled["pairs"].items():
        print(f"    {t1} vs {t2:<12}: agreement {s['agree']}%  kappa {s['kappa']}")
    print(f"  Fleiss' kappa (all three): {pooled['fleiss_kappa']}")
    print(f"\n  Summary written to: {summary_path}")


if __name__ == "__main__":
    main()