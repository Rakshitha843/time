"""Evaluate the NLP layer on a labelled instruction set.

    python eval/evaluate.py            # prints intent accuracy, entity P/R/F1, constraint accuracy

Each line of eval_set.jsonl holds the sentence and the expected intent, entities
(days / subjects / teachers) and, where relevant, the hard/soft flag and constraint type.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models import sample_store            # noqa: E402
from nlp.parser import Parser, ParseContext  # noqa: E402


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 1.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def evaluate(path=ROOT / "eval" / "eval_set.jsonl", verbose=True):
    store = sample_store()
    parser = Parser(store)
    ctx = ParseContext(store, store.config)
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    intent_ok = 0
    ent = {"days": [0, 0, 0], "subjects": [0, 0, 0], "teachers": [0, 0, 0]}
    hard_ok = hard_n = type_ok = type_n = 0
    failures = []
    for r in rows:
        item = parser.parse(r["text"], ctx).items[0]
        ok_intent = item.intent == r["intent"]
        intent_ok += ok_intent
        if not ok_intent:
            failures.append((r["text"], "intent", r["intent"], item.intent))
        for k in ent:
            pred, gold = set(getattr(item.entities, k)), set(r[k])
            ent[k][0] += len(pred & gold)
            ent[k][1] += len(pred - gold)
            ent[k][2] += len(gold - pred)
            if pred != gold:
                failures.append((r["text"], k, sorted(gold), sorted(pred)))
        if r["ctype"]:
            type_n += 1
            c = item.constraints[0] if item.constraints else None
            if c and c.type == r["ctype"]:
                type_ok += 1
            else:
                failures.append((r["text"], "constraint type", r["ctype"], c.type if c else None))
            if r["hard"] is not None:
                hard_n += 1
                if c and c.hard == r["hard"]:
                    hard_ok += 1
                else:
                    failures.append((r["text"], "hard/soft", r["hard"], c.hard if c else None))
    n = len(rows)
    out = {"examples": n, "intent_accuracy": intent_ok / n,
           "constraint_type_accuracy": type_ok / max(type_n, 1), "hard_soft_accuracy": hard_ok / max(hard_n, 1)}
    for k, (tp, fp, fn) in ent.items():
        p, r_, f = prf(tp, fp, fn)
        out[f"{k}_f1"] = f
    if verbose:
        for k, v in out.items():
            print(f"{k:28s} {v:.3f}" if isinstance(v, float) else f"{k:28s} {v}")
        if failures:
            print("\nFailures:")
            for f in failures:
                print("  ", f)
    return out, failures




# --------------------------------------------------------------------------
# Intent evaluation: rules vs trained model vs hybrid, on hand-written held-out sets
# --------------------------------------------------------------------------
INTENTS = ["create", "modify", "move", "view", "regenerate", "reset", "unknown"]


def evaluate_intents(files=("intent_heldout.jsonl", "intent_heldout2.jsonl"), verbose=True):
    store = sample_store()
    ctx = ParseContext(store, store.config)
    rows = []
    for f in files:
        rows += [json.loads(l) for l in (ROOT / "eval" / f).read_text().splitlines() if l.strip()]
    results = {}
    for mode in ("rules", "model", "hybrid"):
        parser = Parser(store, mode)
        pred = [parser.parse_clause(r["text"], ctx).intent for r in rows]
        gold = [r["intent"] for r in rows]
        acc = sum(p == g for p, g in zip(pred, gold)) / len(rows)
        results[mode] = {"accuracy": acc, "pred": pred, "gold": gold}
    if verbose:
        print(f"\nIntent accuracy on {len(rows)} unseen sentences")
        for mode, r in results.items():
            print(f"  {mode:8s} {r['accuracy']:.1%}")
        pred, gold = results["hybrid"]["pred"], results["hybrid"]["gold"]
        print("\nPer-class (hybrid):   precision  recall  f1  support")
        for c in INTENTS:
            tp = sum(p == c and g == c for p, g in zip(pred, gold))
            fp = sum(p == c and g != c for p, g in zip(pred, gold))
            fn = sum(p != c and g == c for p, g in zip(pred, gold))
            if tp + fn == 0:
                continue
            p_, r_, f_ = prf(tp, fp, fn)
            print(f"  {c:11s} {p_:9.2f} {r_:7.2f} {f_:5.2f} {tp + fn:8d}")
        print("\nConfusion matrix (rows = true, columns = predicted)")
        print("            " + " ".join(f"{c[:6]:>6s}" for c in INTENTS))
        for g in INTENTS:
            if g not in gold:
                continue
            print(f"  {g:9s} " + " ".join(f"{sum(a == g and b == p for a, b in zip(gold, pred)):6d}" for p in INTENTS))
    return results


if __name__ == "__main__":
    evaluate()
    evaluate_intents()
