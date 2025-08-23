#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
score_phone.py  — Score new numbers using a trained .pth (8-digit phone number only)

Inputs can be:
  - CSV with a 'number' column (full string). 'phone' column is optional.
  - TXT with one number per line.
  - Inline via --numbers "136-11228899, 13812345678"

Outputs CSV with columns:
  number, vendor, phone, ml_score, rule_score
Only the 8-digit 'phone' is scored; vendor is ignored except for display.
"""
import argparse, csv, math, re, sys, os
from collections import Counter
from itertools import groupby

import torch
import torch.nn as nn

# ------------------------------- Utilities -------------------------------

def only_digits(s: str) -> str:
    return "".join(re.findall(r"\\d", s))

LUCK_PER_DIGIT = {"8": 2.0, "6": 1.0, "9": 0.5, "4": -2.0}

def luck_norm_0to1(phone: str) -> float:
    if len(phone) != 8: return 0.5
    total = sum(LUCK_PER_DIGIT.get(d, 0.0) for d in phone)
    avg = total / 8.0
    return max(0.0, min(1.0, (avg + 2.0) / 4.0))

def longest_run(s: str) -> int:
    if not s: return 0
    return max((sum(1 for _ in g) for _, g in groupby(s)), default=0)

def repetition_score(s: str) -> float:
    if not s: return 0.0
    return (longest_run(s) - 1) / 7.0

def shannon_entropy(s: str) -> float:
    if not s: return 0.0
    c = Counter(s); n = len(s)
    ps = [v/n for v in c.values()]
    return -sum(p * math.log(p, 2) for p in ps)

def simplicity_score(s: str) -> float:
    if not s: return 0.0
    uniq_part = 1 - (len(set(s)) - 1) / 7.0
    H = shannon_entropy(s); Hmax = math.log(8, 2)
    ent_part = 1 - (H / Hmax if Hmax > 0 else 1.0)
    uniq_part = max(0.0, min(1.0, uniq_part))
    ent_part  = max(0.0, min(1.0, ent_part))
    return 0.5 * (uniq_part + ent_part)

def patterns_anywhere_flags(s: str):
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
    if len(s) < 2: return 0.0
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
    W = {"luck": 0.35, "pat": 0.25, "rep": 0.15, "seq": 0.15, "simp": 0.10}
    subs = {
        "luck": luck_norm_0to1(phone),
        "pat":  pattern_score_anywhere_0to1(phone),
        "rep":  repetition_score(phone),
        "seq":  sequence_score(phone),
        "simp": simplicity_score(phone),
    }
    return round(sum(W[k]*subs[k] for k in W) * 100.0, 3)

def features_from_phone(phone: str):
    s = phone
    counts = [s.count(str(d)) for d in range(10)]
    lr = longest_run(s); longest_run_norm = (lr - 1) / 7.0
    uniq_norm = 1 - (len(set(s)) - 1) / 7.0
    seq_norm = sequence_score(s)

    AAAA, AABB, ABAB, ABBA, S1234, S4321 = patterns_anywhere_flags(s)
    repeat_halves, AABBCCDD = whole_number_patterns(s)

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
    return vec, names

class LinearScorer(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, 1)
    def forward(self, x):
        return self.lin(x)

def load_model_bundle(pth_path):
    bundle = torch.load(pth_path, map_location="cpu")
    return bundle

def standardize(X, mean, std):
    import numpy as np
    std = std.copy()
    std[std==0] = 1.0
    return (X - mean)/std

def read_numbers_from_file(path):
    # If CSV with 'number' column -> read that.
    # Else treat as plain text, one per line.
    numbers = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            head = f.readline()
            if "number" in head.lower():
                f.seek(0)
                for r in csv.DictReader(f):
                    n = r.get("number","").strip()
                    if n: numbers.append(n)
            else:
                if head.strip(): numbers.append(head.strip())
                for line in f:
                    s = line.strip()
                    if s: numbers.append(s)
    except UnicodeDecodeError:
        # Try GBK as fallback
        with open(path, "r", encoding="gbk") as f:
            for line in f:
                s = line.strip()
                if s: numbers.append(s)
    return numbers

def main():
    ap = argparse.ArgumentParser(description="Score new numbers with a trained .pth (8-digit phone only).")
    ap.add_argument("--model", required=True, help="Path to phone_scorer.pth from train_phone.py fit step.")
    ap.add_argument("--in", dest="in_path", help="Input file (CSV with 'number' OR txt with one number per line).")
    ap.add_argument("--numbers", help="Comma-separated numbers inline.")
    ap.add_argument("--out", dest="out_csv", default="scored.csv", help="Output CSV path.")
    args = ap.parse_args()

    if not args.in_path and not args.numbers:
        raise SystemExit("Please provide --in or --numbers.")

    nums = []
    if args.in_path:
        nums.extend(read_numbers_from_file(args.in_path))
    if args.numbers:
        nums.extend([s.strip() for s in args.numbers.split(",") if s.strip()])

    # Load model bundle
    bundle = load_model_bundle(args.model)
    feat_names = bundle["feature_names"]
    import numpy as np
    mean = np.array(bundle["mean"], dtype=float)
    std  = np.array(bundle["std"], dtype=float)

    # Prepare features
    X = []; recs = []
    for n in nums:
        digits = only_digits(n)
        phone = digits[-8:]
        vendor = digits[:3] if len(digits) >= 11 else ""
        if len(phone) != 8:
            continue
        vec, names = features_from_phone(phone)
        if names != feat_names:
            raise SystemExit("Feature names mismatch. Ensure score_phone.py matches train_phone.py.")
        X.append(vec)
        recs.append((n, vendor, phone))

    if not X:
        raise SystemExit("No valid numbers to score (need at least 8 digits).")

    X = np.array(X, dtype=float)
    Xn = standardize(X, mean, std)
    Xt = torch.tensor(Xn, dtype=torch.float32)

    # Rebuild model and predict
    model = LinearScorer(in_dim=Xt.shape[1])
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    with torch.no_grad():
        yp = model(Xt).numpy().reshape(-1)

    # Also compute rule-based initial score for reference
    rule_scores = [initial_rule_score_0to100(p) for _,_,p in recs]

    # Prepare output rows
    rows = []
    for (full,vendor,phone), ml, rule in zip(recs, yp, rule_scores):
        ml_clamped = float(max(0.0, min(100.0, ml)))
        rows.append({
            "number": full,
            "vendor": vendor,
            "phone": phone,
            "ml_score": round(ml_clamped, 3),
            "rule_score": round(rule, 3)
        })

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["number","vendor","phone","ml_score","rule_score"])
        w.writeheader()
        for r in rows: w.writerow(r)

    print(f"Wrote {len(rows)} scored numbers to {args.out_csv}")

if __name__ == "__main__":
    main()
