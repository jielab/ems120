#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_phone.py  — Human-in-the-loop training for Chinese cellphone "good number" scoring

This script has two subcommands:
  1) gen  — generate random candidate numbers and initial rule-based scores for HUMAN review
  2) fit  — train a simple linear regression (PyTorch) using human-adjusted scores

IMPORTANT:
- We ONLY score the 8-digit subscriber part (the "phone number"), after removing the 3-digit vendor prefix.
- All features and scores operate on those 8 digits.
- The CSV produced in 'gen' has columns: number, vendor, phone, init_score, human_score
  - 'number': full printable number (vendor + '-' + phone)
  - 'vendor': 3-digit prefix (ignored for scoring)
  - 'phone' : the 8-digit phone number actually being scored
  - 'init_score': rule-based score (0–100)
  - 'human_score': reviewer-adjusted score (optional; if blank, 'init_score' is used during training)

Examples:
  # 1) Generate 200 candidates
  python train_phone.py gen --n 200 --out candidates.csv --vendors 136,152,139,138,137,188,199,131,155,186

  # (Open candidates.csv, edit 'human_score' where you disagree with init_score)

  # 2) Fit the model
  python train_phone.py fit --in candidates.csv --out phone_scorer.pth
"""
import argparse, csv, json, math, random, re, os
from collections import Counter
from itertools import groupby

import torch
import torch.nn as nn

# ------------------------------- Utilities -------------------------------

def only_digits(s: str) -> str:
    return "".join(re.findall(r"\d", s))

# Luck weights per digit
LUCK_PER_DIGIT = {"8": 2.0, "6": 1.0, "9": 0.5, "4": -2.0}

def luck_norm_0to1(phone: str) -> float:
    # phone must be 8 digits
    if len(phone) != 8:
        return 0.5
    total = sum(LUCK_PER_DIGIT.get(d, 0.0) for d in phone)
    avg = total / 8.0  # in [-2, 2]
    return max(0.0, min(1.0, (avg + 2.0) / 4.0))

def longest_run(s: str) -> int:
    if not s:
        return 0
    return max((sum(1 for _ in g) for _, g in groupby(s)), default=0)

def repetition_score(s: str) -> float:
    if not s: return 0.0
    # normalize to [0,1] over 8 digits
    return (longest_run(s) - 1) / 7.0

def shannon_entropy(s: str) -> float:
    if not s: return 0.0
    c = Counter(s); n = len(s)
    ps = [v/n for v in c.values()]
    return -sum(p * math.log(p, 2) for p in ps)

def simplicity_score(s: str) -> float:
    # mix "few unique digits" and low entropy
    if not s: return 0.0
    n = len(s)  # should be 8
    uniq = len(set(s))
    uniq_part = 1 - (uniq - 1) / 7.0  # [0,1]
    H = shannon_entropy(s)
    Hmax = math.log(8, 2)  # = 3
    ent_part = 1 - (H / Hmax if Hmax > 0 else 1.0)
    # Clamp parts to [0,1] just in case
    uniq_part = max(0.0, min(1.0, uniq_part))
    ent_part  = max(0.0, min(1.0, ent_part))
    return 0.5 * (uniq_part + ent_part)

def patterns_anywhere_flags(s: str):
    """
    Scan all length-4 windows across the 8-digit phone string to detect:
      - AAAA, AABB, ABAB, ABBA
      - 1234 (step +1), 4321 (step -1)
    Return boolean flags (one per pattern type), where each pattern is awarded at most once.
    """
    AAAA = AABB = ABAB = ABBA = S1234 = S4321 = False
    for i in range(0, len(s) - 3):
        w = s[i:i+4]
        a,b,c,d = w[0], w[1], w[2], w[3]
        if not AAAA and len(set(w)) == 1:
            AAAA = True
        if not AABB and a == b and c == d and a != c:
            AABB = True
        if not ABAB and a == c and b == d and a != b:
            ABAB = True
        if not ABBA and a == d and b == c and a != b:
            ABBA = True
        # sequences
        try:
            diffs = [int(w[j+1]) - int(w[j]) for j in range(3)]
            if not S1234 and all(d == 1 for d in diffs):
                S1234 = True
            if not S4321 and all(d == -1 for d in diffs):
                S4321 = True
        except Exception:
            pass
    return AAAA, AABB, ABAB, ABBA, S1234, S4321

def whole_number_patterns(s: str):
    """
    Whole-8-digit patterns:
      - repeat_halves: s[:4] == s[4:]  (ABCDABCD)
      - AABBCCDD: pairs across the 8 digits are identical
    """
    repeat_halves = (s[:4] == s[4:])
    pairs = [s[i:i+2] for i in range(0, 8, 2)]
    AABBCCDD = all(len(p) == 2 and p[0] == p[1] for p in pairs)
    return repeat_halves, AABBCCDD

def pattern_score_anywhere_0to1(phone: str) -> float:
    AAAA, AABB, ABAB, ABBA, S1234, S4321 = patterns_anywhere_flags(phone)
    repeat_halves, AABBCCDD = whole_number_patterns(phone)
    S = 0.0
    if AAAA:        S += 1.0
    if AABB:        S += 0.8
    if ABAB:        S += 0.7
    if ABBA:        S += 0.6
    if S1234:       S += 0.9
    if S4321:       S += 0.9
    if repeat_halves: S += 0.8
    if AABBCCDD:     S += 0.7
    return min(1.0, S / 2.0)

def sequence_score(s: str) -> float:
    # longest monotonic step ±1 over the full 8 digits
    if len(s) < 2:
        return 0.0
    best = curr = 1
    curr_step = None
    for i in range(1, len(s)):
        diff = int(s[i]) - int(s[i-1])
        if diff in (1, -1) and (curr_step is None or diff == curr_step):
            curr += 1; curr_step = diff
        else:
            best = max(best, curr); curr = 1; curr_step = diff if diff in (1, -1) else None
    best = max(best, curr)
    return max(0.0, (best - 2) / 6.0)

def initial_rule_score_0to100(phone: str) -> float:
    # Weighted blend; tweakable
    W = {"luck": 0.35, "pat": 0.25, "rep": 0.15, "seq": 0.15, "simp": 0.10}
    subs = {
        "luck": luck_norm_0to1(phone),
        "pat":  pattern_score_anywhere_0to1(phone),
        "rep":  repetition_score(phone),
        "seq":  sequence_score(phone),
        "simp": simplicity_score(phone),
    }
    sc = sum(W[k] * subs[k] for k in W) * 100.0
    return round(sc, 3)

# ---- Feature extraction for ML (from 8-digit phone only) ----
def features_from_phone(phone: str):
    s = phone
    # counts 0..9
    counts = [s.count(str(d)) for d in range(10)]
    # normalized run/uniqueness/sequence
    lr = longest_run(s)
    longest_run_norm = (lr - 1) / 7.0
    uniq_norm = 1 - (len(set(s)) - 1) / 7.0
    seq_norm = sequence_score(s)

    # 4-digit subpatterns anywhere (presence flags)
    AAAA, AABB, ABAB, ABBA, S1234, S4321 = patterns_anywhere_flags(s)

    # whole-8 patterns
    repeat_halves, AABBCCDD = whole_number_patterns(s)

    # base sub-scores
    luck01 = luck_norm_0to1(s)
    pat01  = pattern_score_anywhere_0to1(s)
    rep01  = repetition_score(s)
    simp01 = simplicity_score(s)

    vec = counts + [
        longest_run_norm, uniq_norm, seq_norm,
        float(AAAA), float(AABB), float(ABAB), float(ABBA),
        float(S1234), float(S4321),
        float(repeat_halves), float(AABBCCDD),
        luck01, pat01, rep01, simp01
    ]
    names = [f"cnt_{d}" for d in range(10)] + [
        "longest_run_norm","uniq_norm","seq_norm",
        "pat_AAAA","pat_AABB","pat_ABAB","pat_ABBA",
        "pat_1234","pat_4321",
        "pat_repeat_halves","pat_AABBCCDD",
        "base_luck","base_pat","base_rep","base_simp"
    ]
    assert len(vec) == len(names) == 25
    return vec, names

# ----------------------- Model (simple linear reg) -----------------------

class LinearScorer(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, 1)
    def forward(self, x):
        return self.lin(x)

# ------------------------------- GEN step -------------------------------

def gen_candidates(n=100, vendors=None, seed=42, out_csv="candidates.csv"):
    random.seed(seed)
    if not vendors:
        vendors = ["136","152","139","138","137","188","199","131","155","186"]
    rows = []
    for _ in range(n):
        v = random.choice(vendors)
        phone = "".join(str(random.randint(0,9)) for _ in range(8))
        full = f"{v}-{phone}"
        init = initial_rule_score_0to100(phone)
        rows.append({
            "number": full,
            "vendor": v,
            "phone": phone,
            "init_score": init,
            "human_score": ""  # leave blank; user can edit
        })
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["number","vendor","phone","init_score","human_score"])
        w.writeheader()
        for r in rows: w.writerow(r)
    print(f"[gen] Wrote {len(rows)} candidates to {out_csv}")

# ------------------------------- FIT step -------------------------------

def fit_from_csv(in_csv, out_pth="phone_scorer.pth", epochs=1000, lr=0.05, weight_decay=0.0, verbose=True):
    # Load data
    rows = []
    with open(in_csv, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            number = r.get("number","")
            vendor = r.get("vendor","").strip() or only_digits(number)[:3]
            phone  = r.get("phone","").strip()
            if len(phone) != 8:
                # attempt to derive from number
                phone = only_digits(number)[-8:]
            if len(phone) != 8:
                continue
            init_score = float(r.get("init_score","0") or 0)
            hs = r.get("human_score","").strip()
            target = float(hs) if hs != "" else init_score
            rows.append((number, vendor, phone, init_score, target))

    if not rows:
        raise SystemExit("[fit] No valid rows found. Ensure CSV has 8-digit 'phone' or a parsable 'number'.")

    # Build features / targets
    X = []; y = []; feature_names = None
    import numpy as np
    for _, _, phone, init, target in rows:
        vec, names = features_from_phone(phone)
        if feature_names is None: feature_names = names
        X.append(vec); y.append(target)

    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float).reshape(-1,1)

    # Standardize features
    mean = X.mean(axis=0); std = X.std(axis=0); std[std==0] = 1.0
    Xn = (X - mean)/std

    # Torch tensors
    Xt = torch.tensor(Xn, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)

    # Model / loss / opt
    model = LinearScorer(in_dim=Xt.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    for ep in range(1, epochs+1):
        opt.zero_grad()
        pred = model(Xt)
        loss = loss_fn(pred, yt)
        loss.backward()
        opt.step()
        if verbose and (ep % max(epochs//10,1) == 0):
            mae = (pred.detach().numpy()-y).abs().mean()
            print(f"[fit] epoch {ep:4d}/{epochs}  MSE={loss.item():.4f}  MAE={mae:.3f}")

    # Save bundle
    bundle = {
        "state_dict": model.state_dict(),
        "feature_names": feature_names,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "metadata": {
            "model": "LinearScorer",
            "epochs": epochs,
            "lr": lr,
            "weight_decay": weight_decay,
            "target_info": "Targets are human_score if provided, else init_score. Scores are on 0-100 scale.",
            "scoring_note": "Scoring uses ONLY the 8-digit phone number; vendor prefix is ignored."
        }
    }
    torch.save(bundle, out_pth)
    print(f"[fit] Saved model to {out_pth}")
    return out_pth

# ---------------------------------- CLI ----------------------------------

def main():
    ap = argparse.ArgumentParser(description="Train a scorer for Chinese mobile numbers (8-digit phone number only).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_gen = sub.add_parser("gen", help="Generate candidate numbers with initial rule-based scores (for human review).")
    ap_gen.add_argument("--n", type=int, default=100, help="Number of candidates to generate.")
    ap_gen.add_argument("--vendors", type=str, default="136,152,139,138,137,188,199,131,155,186", help="Comma-separated vendor codes (first 3 digits).")
    ap_gen.add_argument("--seed", type=int, default=42, help="Random seed.")
    ap_gen.add_argument("--out", type=str, default="candidates.csv", help="Output CSV path.")

    ap_fit = sub.add_parser("fit", help="Fit linear model from CSV with init_score and optional human_score.")
    ap_fit.add_argument("--in", dest="in_csv", required=True, help="Input CSV (from 'gen'; after human edits).")
    ap_fit.add_argument("--out", dest="out_pth", default="phone_scorer.pth", help="Output .pth parameter file.")
    ap_fit.add_argument("--epochs", type=int, default=1000, help="Training epochs.")
    ap_fit.add_argument("--lr", type=float, default=0.05, help="Learning rate.")
    ap_fit.add_argument("--wd", type=float, default=0.0, help="Weight decay (L2).")
    ap_fit.add_argument("--quiet", action="store_true", help="Less logging.")

    args = ap.parse_args()
    if args.cmd == "gen":
        vendors = [v.strip() for v in args.vendors.split(",") if v.strip()]
        gen_candidates(n=args.n, vendors=vendors, seed=args.seed, out_csv=args.out)
    elif args.cmd == "fit":
        fit_from_csv(args.in_csv, out_pth=args.out_pth, epochs=args.epochs, lr=args.lr, weight_decay=args.wd, verbose=not args.quiet)

if __name__ == "__main__":
    main()
