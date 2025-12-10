#%% Shared variables (edit here once)
DIR0 = "/mnt/d"
INFILE  = f"{DIR0}/data/ems120/clean/2019.xlsx"
OUTFILE = f"{DIR0}/analysis/ems120/out/bert.hfl.xlsx"
ONLY_PREDICT = True # False
MODEL_NAME   = "bert" 
MODEL_PARAM  = f"{DIR0}/data/ai/bert/hfl" 
DIRTRAIN     = f"{DIR0}/analysis/ems120/bert/hfl"
BAIDU_AK     = "<你的百度AK>"
GEO_CITY     = "深圳市" 

def common_argv():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument('--infile', type=str, required=True)
    ap.add_argument('--outfile', type=str, required=True)
    ap.add_argument('--model_name', type=str, required=True)
    ap.add_argument('--model_param', type=str, required=True)
    ap.add_argument("--n_train", type=int, help="最前面行数")
    ap.add_argument('--dirtrain', type=str, required=True)
    return ap

def common_argv_pass():
    return ["prog", "--infile", INFILE, "--outfile", OUTFILE, "--model_name", MODEL_NAME, "--model_param", MODEL_PARAM, "--n_train", "10000", "--dirtrain", DIRTRAIN]

    
#%% Common libraries
import os, sys, json, torch, argparse
import math, random, requests, re, unicodedata
import pandas as pd
import numpy as np
import torch.nn as nn
from collections import Counter
from itertools import groupby
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM


#%% Common functions
d = os.path.dirname(OUTFILE)
if d: os.makedirs(d, exist_ok=True)
if DIRTRAIN: os.makedirs(DIRTRAIN, exist_ok=True)
class argv_ctx:
    def __init__(self, argv): self.argv, self._old = list(argv), None
    def __enter__(self): self._old = sys.argv; sys.argv = self.argv
    def __exit__(self, exc_type, exc, tb): sys.argv = self._old
def resolve_outfile_path(dirtrain: str, requested_outfile: str, model_name: str) -> str:
    if requested_outfile and str(requested_outfile).strip():
        outpath = str(requested_outfile).strip()
        parent = os.path.dirname(outpath)
        if parent: os.makedirs(parent, exist_ok=True)
        return outpath
    os.makedirs(dirtrain, exist_ok=True)
    return os.path.join(dirtrain, f"output.{model_name}.xlsx")


#%% 🏮📱Phone功能 
def to_halfwidth(s: str) -> str:
    try: return unicodedata.normalize("NFKC", str(s))
    except Exception: return str(s)
def only_digits(s: str) -> str: return "".join(re.findall(r"\d", to_halfwidth(s)))
def normalize_mobile(raw: str) -> str:
    s = only_digits(raw)
    s = s.lstrip('0')
    if len(s) == 11 and s[0] == '1': return s
    return ""
def extract_phone8(raw) -> str:
    mob = normalize_mobile(raw)
    return mob[-8:] if mob else ""
def simple_rule_new_raw(phone8: str) -> float:
    if len(phone8) != 8: return np.nan
    if "4" in phone8: return 0.0
    c = Counter(phone8)
    return 2.0 * c.get("8", 0) + 1.0 * c.get("9", 0) + 1.0 * c.get("6", 0)
def simple_rule_score_1to10(phone8: str) -> float:
    raw = simple_rule_new_raw(phone8)
    if not isinstance(raw, (int, float)) or (isinstance(raw, float) and np.isnan(raw)): return np.nan
    score = 1.0 + 9.0 * (raw / 16.0)
    return float(np.clip(score, 1.0, 10.0))
def simple_rule_explain(phone8: str, score: float, style: str = "brief") -> str:
    if len(phone8) != 8: return "号码不足8位，无法按简单规则评分"
    c = Counter(phone8)
    has4 = ("4" in phone8)
    raw = 0.0 if has4 else (2.0 * c.get("8", 0) + 1.0 * c.get("9", 0) + 1.0 * c.get("6", 0))
    if has4: rule_txt = "包含‘4’，raw=0"
    else: rule_txt = f"不含‘4’，raw=2*#8+1*#9+1*#6={2*c.get('8',0)}+{c.get('9',0)}+{c.get('6',0)}={raw:.1f}"
    if style == "detailed":
        per_digit = " ".join(f"{d}:{('+2' if d=='8' else '+1' if d in '69' else '0')}" for d in phone8)
        return f"{rule_txt}；norm=raw/16；最终={float(score):.1f}；位次贡献[{per_digit}]"
    return f"{rule_txt}；norm=raw/16；最终={float(score):.1f}"

# ------------------------------ 模式/特征工具 ------------------------------
def longest_run(s: str) -> int:
    if not s: return 0
    return max((sum(1 for _ in g) for _, g in groupby(s)), default=0)
def repetition_score01(s: str) -> float:
    if not s: return 0.0
    return (longest_run(s) - 1) / 7.0
def shannon_entropy(s: str) -> float:
    if not s: return 0.0
    c = Counter(s); n = len(s)
    ps = [v / n for v in c.values()]
    return -sum(p * np.log2(p) for p in ps)
def simplicity_score01(s: str) -> float:
    if not s: return 0.0
    uniq = len(set(s))
    uniq_part = 1 - (uniq - 1) / 7.0
    H = shannon_entropy(s); Hmax = np.log2(8)
    ent_part = 1 - (H / Hmax if Hmax > 0 else 1.0)
    return float(np.clip(0.5 * (np.clip(uniq_part,0,1) + np.clip(ent_part,0,1)), 0.0, 1.0))
def patterns_anywhere_flags(s: str):
    AAAA = AABB = ABAB = ABBA = S1234 = S4321 = False
    for i in range(0, len(s) - 3):
        w = s[i:i+4]; a,b,c,d = w[0],w[1],w[2],w[3]
        if not AAAA and len(set(w)) == 1: AAAA = True
        if not AABB and a==b and c==d and a!=c: AABB = True
        if not ABAB and a==c and b==d and a!=b: ABAB = True
        if not ABBA and a==d and b==c and a!=b: ABBA = True
        try:
            diffs = [int(w[j+1])-int(w[j]) for j in range(3)]
            if not S1234 and all(d==1 for d in diffs): S1234 = True
            if not S4321 and all(d==-1 for d in diffs): S4321 = True
        except Exception: pass
    return AAAA, AABB, ABAB, ABBA, S1234, S4321
def whole_number_patterns(s: str):
    repeat_halves = (s[:4] == s[4:])
    pairs = [s[i:i+2] for i in range(0,8,2)]
    AABBCCDD = all(len(p)==2 and p[0]==p[1] for p in pairs)
    return repeat_halves, AABBCCDD

# ------------------------------ 特征工程 ------------------------------
def base_feature_vector_full(phone8: str):
    s = phone8
    counts = [s.count(str(d)) for d in range(10)]
    lr = longest_run(s); longest_run_norm = (lr - 1) / 7.0
    uniq_norm = 1 - (len(set(s)) - 1) / 7.0
    AAAA, AABB, ABAB, ABBA, S1234, S4321 = patterns_anywhere_flags(s)
    repeat_halves, AABBCCDD = whole_number_patterns(s)
    total_rule = simple_rule_new_raw(s) if "4" not in s and len(s) == 8 else 0.0
    base_luck01 = (total_rule / 16.0) if len(s) == 8 else 0.0
    base_pat01 = float(AAAA)*0.5 + float(AABB)*0.4 + float(ABAB)*0.45 + float(ABBA)*0.3 + float(S1234)*0.45 + float(S4321)*0.45 + float(repeat_halves)*0.5 + float(AABBCCDD)*0.35
    base_pat01 = float(np.clip(base_pat01, 0.0, 1.0))
    base_rep01 = repetition_score01(s)
    base_simp01 = simplicity_score01(s)
    vec = counts + [
        float(longest_run_norm), float(uniq_norm), float(max(S1234, S4321)),
        float(AAAA), float(AABB), float(ABAB), float(ABBA),
        float(S1234), float(S4321),
        float(repeat_halves), float(AABBCCDD),
        float(base_luck01), float(base_pat01), float(base_rep01), float(base_simp01)
    ]
    names = [f"cnt_{d}" for d in range(10)] + [
        "longest_run_norm", "uniq_norm", "seq_any1234or4321", "pat_AAAA", "pat_AABB", "pat_ABAB", "pat_ABBA",
        "pat_1234","pat_4321", "pat_repeat_halves","pat_AABBCCDD", "base_luck01", "base_pat01", "base_rep01", "base_simp01"
    ]
    return np.array(vec, dtype=float), names  # 25 维
def base_feature_vector_minimal(phone8: str):
    s = phone8
    cnt_8, cnt_6, cnt_9, cnt_4 = s.count("8"), s.count("6"), s.count("9"), s.count("4")
    AAAA, AABB, ABAB, ABBA, S1234, S4321 = patterns_anywhere_flags(s)
    repeat_halves, AABBCCDD = whole_number_patterns(s)
    seq_any = float(max(S1234, S4321))
    total_rule = simple_rule_new_raw(s) if "4" not in s and len(s) == 8 else 0.0
    base_luck01 = (total_rule / 16.0) if len(s) == 8 else 0.0
    base_pat01 = float(AAAA)*0.5 + float(AABB)*0.4 + float(ABAB)*0.45 + float(ABBA)*0.3 + float(S1234)*0.45 + float(S4321)*0.45 + float(repeat_halves)*0.5 + float(AABBCCDD)*0.35
    base_pat01 = float(np.clip(base_pat01, 0.0, 1.0))
    base_rep01 = repetition_score01(s)
    base_simp01 = simplicity_score01(s)
    vec = [float(cnt_8),float(cnt_6),float(cnt_9),float(cnt_4), float(seq_any), float(base_pat01),float(base_rep01),float(base_simp01),float(base_luck01)]
    names = ["cnt_8","cnt_6","cnt_9","cnt_4","seq_any1234or4321","base_pat01","base_rep01","base_simp01","base_luck01"]
    return np.array(vec, dtype=float), names  # 9 维
def base_feature_vector(phone8: str, feature_set: str):
    if feature_set == "minimal": return base_feature_vector_minimal(phone8)
    return base_feature_vector_full(phone8)

# ------------------------------ 简单AI评分（模式感知） ------------------------------
def simple_ai_score_and_reason(phone8: str, feature_set: str = "full"):
    if len(phone8) != 8:
        return (np.nan, "号码不足8位，无法按简单AI评分")
    base_vec, names = base_feature_vector(phone8, feature_set=feature_set)
    idx = {n:i for i,n in enumerate(names)}
    def g(n): return float(base_vec[idx[n]]) if n in idx else 0.0
    pat, rep, simp = g("base_pat01"), g("base_rep01"), g("base_simp01")
    seq, luck = g("seq_any1234or4321"), g("base_luck01")
    w_pat, w_rep, w_simp, w_seq, w_luck = 0.45, 0.20, 0.15, 0.10, 0.10
    score01 = w_pat*pat + w_rep*rep + w_simp*simp + w_seq*seq + w_luck*luck
    score = float(np.clip(1.0 + 9.0*score01, 1.0, 10.0))
    AAAA, AABB, ABAB, ABBA, S1234, S4321 = patterns_anywhere_flags(phone8)
    rh, aabbccdd = whole_number_patterns(phone8)
    flags = []
    if AAAA: flags.append("AAAA")
    if AABB: flags.append("AABB")
    if ABAB: flags.append("ABAB")
    if ABBA: flags.append("ABBA")
    if S1234: flags.append("顺子")
    if S4321: flags.append("倒顺")
    if rh: flags.append("前后重复")
    if aabbccdd: flags.append("AABBCCDD")
    flag_txt = "、".join(flags) if flags else "无显著4位模式"
    reason = (f"模式={pat:.2f}×{w_pat}+重复={rep:.2f}×{w_rep}+简洁={simp:.2f}×{w_simp}+顺序={seq:.2f}×{w_seq}+吉利={luck:.2f}×{w_luck}；触发: {flag_txt}；最终={score:.1f}")
    return (score, reason)

# ------------------------------ 偏好解析（来自专家修改评分理由） ------------------------------
PREF_SCHEMA = {"digit_weights_delta": {"8": 0.0, "6": 0.0, "9": 0.0, "4": 0.0},
    "pattern_weights": {"AAAA": 0.0, "AABB": 0.0, "ABAB": 0.0, "ABBA": 0.0, "seq_up": 0.0, "seq_down": 0.0, "repeat_halves": 0.0, "AABBCCDD": 0.0},
    "simplicity_pref": 0.0, "repetition_pref": 0.0}
def clamp(v, lo, hi): return float(max(lo, min(hi, v)))
def heuristic_parse_reason(text: str):
    t = (text or "").strip()
    pref = json.loads(json.dumps(PREF_SCHEMA))
    if not t: return pref
    t_low = t.lower()
    if any(k in t for k in ["喜欢8","偏爱8","发发","888"]): pref["digit_weights_delta"]["8"] += 0.6
    if any(k in t for k in ["喜欢6","顺","666"]): pref["digit_weights_delta"]["6"] += 0.4
    if any(k in t for k in ["喜欢9","久久","999"]): pref["digit_weights_delta"]["9"] += 0.2
    if any(k in t for k in ["讨厌4","不喜欢4","忌4","避4","四不吉"]): pref["digit_weights_delta"]["4"] -= 0.7
    if "abab" in t_low or "交替" in t or "间隔" in t: pref["pattern_weights"]["ABAB"] += 0.7
    if any(k in t for k in ["顺子","递增","连升"]): pref["pattern_weights"]["seq_up"] += 0.6
    if any(k in t for k in ["倒顺","递减"]): pref["pattern_weights"]["seq_down"] += 0.5
    if any(k in t for k in ["豹子","四连","四个一样","aaaa"]): pref["pattern_weights"]["AAAA"] += 0.5
    if "aabb" in t_low: pref["pattern_weights"]["AABB"] += 0.4
    if any(k in t for k in ["镜像","对称","回文","abba"]): pref["pattern_weights"]["ABBA"] += 0.3
    if "aabbccdd" in t_low: pref["pattern_weights"]["AABBCCDD"] += 0.35
    if any(k in t for k in ["重复前后","前后相同"]): pref["pattern_weights"]["repeat_halves"] += 0.4
    if any(k in t for k in ["简洁","好记","数字少","简单"]): pref["simplicity_pref"] += 0.5
    if any(k in t for k in ["重复","连号","豹子"]): pref["repetition_pref"] += 0.4
    for d in pref["digit_weights_delta"]: pref["digit_weights_delta"][d] = clamp(pref["digit_weights_delta"][d], -1, 1)
    for p in pref["pattern_weights"]:   pref["pattern_weights"][p]   = clamp(pref["pattern_weights"][p],   -1, 1)
    pref["simplicity_pref"]  = clamp(pref["simplicity_pref"], 0, 1)
    pref["repetition_pref"]  = clamp(pref["repetition_pref"], 0, 1)
    return pref

# ------------------------------ 本地 LLM 解析器（可选） ------------------------------
class LocalReasonParser:
    def __init__(self, kind: str, model_path: str):
        self.kind = None; self.model = None; self.tok = None; self.device = None
        self.max_new_tokens = 256
        if kind in ("bert","qwen") and model_path and model_path.upper() != "NA":
            self.kind = kind
            self._load_model(model_path)
    def _load_model(self, path):
        try:
            self.tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                path, torch_dtype="auto", device_map="auto", trust_remote_code=True
            )
            self.device = next(self.model.parameters()).device
        except Exception as e:
            print(f"[warn] 加载本地模型失败：{e}。回退到启发式解析。")
            self.kind = None; self.model=None; self.tok=None; self.device=None
    def _build_prompt(self, text: str) -> str:
        return ("仅输出 JSON：digit_weights_delta{8,6,9,4}，pattern_weights{AAAA,AABB,ABAB,ABBA,seq_up,seq_down,repeat_halves,AABBCCDD},"
                "simplicity_pref,repetition_pref。解析：<<<" + (text or "") + ">>>")
    def _extract_json(self, s: str):
        try:
            start, end = s.find("{"), s.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(s[start:end+1])
        except Exception:
            pass
        return None
    def parse_batch(self, texts):
        if not self.model or not self.tok:
            return [heuristic_parse_reason(t) for t in texts]
        unique, order = {}, []
        for t in texts:
            key = (t or "").strip()
            if key not in unique: unique[key] = None
            order.append(key)
        for key in unique.keys():
            try:
                inputs = self.tok(self._build_prompt(key), return_tensors="pt").to(self.device)
                with torch.no_grad():
                    out = self.model.generate(
                        **inputs, max_new_tokens=self.max_new_tokens, do_sample=False, temperature=0.0,
                        pad_token_id=self.tok.eos_token_id, eos_token_id=self.tok.eos_token_id
                    )
                txt = self.tok.decode(out[0], skip_special_tokens=True)
                j = self._extract_json(txt) or {}
                pref = json.loads(json.dumps(PREF_SCHEMA))
                dw, pw = j.get("digit_weights_delta", {}), j.get("pattern_weights", {})
                pref["digit_weights_delta"]["8"] = clamp(float(dw.get("8",0)),-1,1)
                pref["digit_weights_delta"]["6"] = clamp(float(dw.get("6",0)),-1,1)
                pref["digit_weights_delta"]["9"] = clamp(float(dw.get("9",0)),-1,1)
                pref["digit_weights_delta"]["4"] = clamp(float(dw.get("4",0)),-1,1)
                for k in pref["pattern_weights"]: pref["pattern_weights"][k] = clamp(float(pw.get(k,0)),-1,1)
                pref["simplicity_pref"] = clamp(float(j.get("simplicity_pref",0)),0,1)
                pref["repetition_pref"] = clamp(float(j.get("repetition_pref",0)),0,1)
            except Exception as e:
                print(f"[warn] LLM 解析失败：{e}；回退启发式")
                pref = heuristic_parse_reason(key)
            unique[key] = pref
        return [unique[k] for k in order]

# ------------------------------ 偏好聚合与增强特征 ------------------------------
def aggregate_prefs(pref_list):
    if not pref_list: return json.loads(json.dumps(PREF_SCHEMA))
    agg = json.loads(json.dumps(PREF_SCHEMA))
    for p in pref_list:
        for d in agg["digit_weights_delta"]:
            agg["digit_weights_delta"][d] += float(p["digit_weights_delta"].get(d,0))
        for k in agg["pattern_weights"]:
            agg["pattern_weights"][k] += float(p["pattern_weights"].get(k,0))
        agg["simplicity_pref"] += float(p.get("simplicity_pref",0))
        agg["repetition_pref"] += float(p.get("repetition_pref",0))
    n = float(len(pref_list))
    for d in agg["digit_weights_delta"]: agg["digit_weights_delta"][d] = clamp(agg["digit_weights_delta"][d]/n, -1, 1)
    for k in agg["pattern_weights"]: agg["pattern_weights"][k] = clamp(agg["pattern_weights"][k]/n, -1, 1)
    agg["simplicity_pref"] = clamp(agg["simplicity_pref"]/n, 0, 1)
    agg["repetition_pref"] = clamp(agg["repetition_pref"]/n, 0, 1)
    return agg
def pref_augmented_features(base_vec: np.ndarray, base_names, phone8: str, global_pref: dict):
    idx = {n:i for i,n in enumerate(base_names)}
    def get(name): return base_vec[idx[name]] if name in idx else 0.0
    cnt8, cnt6, cnt9, cnt4 = get("cnt_8"), get("cnt_6"), get("cnt_9"), get("cnt_4")
    AAAA, AABB, ABAB, ABBA = get("pat_AAAA"), get("pat_AABB"), get("pat_ABAB"), get("pat_ABBA")
    S1234, S4321 = get("pat_1234"), get("pat_4321")
    repeat_halves, AABBCCDD = get("pat_repeat_halves"), get("pat_AABBCCDD")
    rep01, simp01, pat01, seq_any = get("base_rep01"), get("base_simp01"), get("base_pat01"), get("seq_any1234or4321")
    dw, pw = global_pref["digit_weights_delta"], global_pref["pattern_weights"]
    sp, rp = global_pref["simplicity_pref"], global_pref["repetition_pref"]
    AAAA_term = AAAA if AAAA!=0 else pat01 * (pw["AAAA"] if pat01!=0 else 0)
    AABB_term = AABB if AABB!=0 else pat01 * (pw["AABB"] if pat01!=0 else 0)
    ABAB_term = ABAB if ABAB!=0 else pat01 * (pw["ABAB"] if pat01!=0 else 0)
    ABBA_term = ABBA if ABBA!=0 else pat01 * (pw["ABBA"] if pat01!=0 else 0)
    RH_term = repeat_halves if repeat_halves!=0 else pat01 * (pw["repeat_halves"] if pat01!=0 else 0)
    AABBCCDD_term = AABBCCDD if AABBCCDD!=0 else pat01 * (pw["AABBCCDD"] if pat01!=0 else 0)
    seq_up_term = S1234 if S1234!=0 else seq_any
    seq_down_term = S4321 if S4321!=0 else seq_any
    aug = np.array([
        cnt8*dw["8"], cnt6*dw["6"], cnt9*dw["9"], cnt4*dw["4"], AAAA_term*pw["AAAA"], AABB_term*pw["AABB"], ABAB_term*pw["ABAB"], ABBA_term*pw["ABBA"],
        seq_up_term*pw["seq_up"], seq_down_term*pw["seq_down"], RH_term*pw["repeat_halves"], AABBCCDD_term*pw["AABBCCDD"], rep01*rp, simp01*sp
    ], dtype=float)
    names = list(base_names) + ["aug_cnt8","aug_cnt6","aug_cnt9","aug_cnt4", "aug_AAAA","aug_AABB","aug_ABAB","aug_ABBA", "aug_seq_up","aug_seq_down","aug_repeat_halves","aug_AABBCCDD", "aug_rep","aug_simp"]
    return np.concatenate([base_vec, aug], axis=0), names

# ------------------------------ 线性回归 ------------------------------
class LinearRegressor:
    def __init__(self, ridge: float = 0.0):
        self.w=None; self.b=None; self.mean=None; self.std=None
        self.feature_names=None; self.global_pref=None; self.meta={}; self.ridge=float(max(0.0,ridge))
    def fit(self, X: np.ndarray, y: np.ndarray, feature_names, global_pref: dict):
        assert X.ndim==2 and y.ndim==1
        self.feature_names=list(feature_names); self.global_pref=global_pref
        self.mean=X.mean(axis=0); self.std=X.std(axis=0); self.std[self.std==0]=1.0
        Z=(X-self.mean)/self.std; Z1=np.hstack([Z, np.ones((Z.shape[0],1))])
        lam=self.ridge
        if lam>0:
            nfeat=Z.shape[1]; A=Z1.T @ Z1
            for i in range(nfeat): A[i,i]+=lam
            theta=np.linalg.solve(A, Z1.T @ y)
        else: theta, *_=np.linalg.lstsq(Z1, y, rcond=None)
        self.w=theta[:-1]; self.b=theta[-1]
    def predict(self, X: np.ndarray) -> np.ndarray:
        Z=(X-self.mean)/self.std
        return Z @ self.w + self.b
    def save(self, path: str, meta: dict=None):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload={"w":self.w,"b":self.b,"mean":self.mean,"std":self.std, "feature_names":np.array(self.feature_names, dtype=object), "global_pref":json.dumps(self.global_pref, ensure_ascii=False),
                 "meta":json.dumps(meta or {}, ensure_ascii=False), "ridge":np.array([self.ridge], dtype=float)}
        np.savez(path, **payload)
    @staticmethod
    def load(path: str):
        data=np.load(path, allow_pickle=True)
        m=LinearRegressor(ridge=float(data.get("ridge", np.array([0.0]))[0]))
        m.w=data["w"]; m.b=data["b"]; m.mean=data["mean"]; m.std=data["std"]
        m.feature_names=list(data["feature_names"]); m.global_pref=json.loads(str(data["global_pref"]))
        try: m.meta=json.loads(str(data["meta"]))
        except Exception: m.meta={}
        return m
def kfold_eval(X, y, k=5, ridge: float = 0.0):
    if k<=1 or len(y)<k: return None
    n=len(y); fold_sizes=[(n+i)//k for i in range(k)]; idx=np.arange(n); offset=0
    maes, mses, rs = [], [], []
    for fs in range(len(fold_sizes)):
        size=fold_sizes[fs]; val_idx=idx[offset:offset+size]; tr_idx=np.setdiff1d(idx, val_idx); offset+=size
        model=LinearRegressor(ridge=ridge)
        model.fit(X[tr_idx], y[tr_idx], feature_names=[f"f{i}" for i in range(X.shape[1])], global_pref=PREF_SCHEMA)
        yp=model.predict(X[val_idx])
        mae=float(np.mean(np.abs(yp-y[val_idx]))); mse=float(np.mean((yp-y[val_idx])**2))
        r=float(np.corrcoef(yp, y[val_idx])[0,1]) if (np.std(yp)>1e-8 and np.std(y[val_idx])>1e-8) else float("nan")
        maes.append(mae); mses.append(mse); rs.append(r)
    return {"MAE":float(np.nanmean(maes)), "MSE":float(np.nanmean(mses)), "PearsonR":float(np.nanmean(rs))}


#%% Phone📱
def train_with_expert(df: pd.DataFrame, dirtrain: str, kfold: int, model_name: str, model_param: str, feature_set: str, ridge: float):
    os.makedirs(dirtrain, exist_ok=True)
    model_path = os.path.join(dirtrain, "phone_linear_pref.npz")
    # 仅使用“专家修改评分”作为监督
    train_mask = df["专家修改评分"].notna() & (df["_phone8"].str.len() == 8)
    df_train = df.loc[train_mask].copy()
    if df_train.empty:
        print("[info] 训练集为空（无‘专家修改评分’）。跳过训练。")
        return None
    reason_col = "专家修改评分理由"
    global_pref = json.loads(json.dumps(PREF_SCHEMA))
    ps = None
    if model_name in ("bert", "qwen") and model_param and model_param.upper() != "NA":
        ps = LocalReasonParser(model_name, model_param)
    if ps and ps.kind and reason_col in df.columns:
        mask = df_train[reason_col].notna()
        texts = [str(t) for t in df_train.loc[mask, reason_col].tolist() if str(t).strip() not in ("","nan","NaN")]
        if texts:
            print(f"[reason] 使用 {ps.kind} 解析 {len(texts)} 条专家修改理由。")
            prefs = ps.parse_batch(texts)
            global_pref = aggregate_prefs(prefs)
        else: print("[reason] 理由列为空，使用默认偏好。")
    else: print("[reason] 未启用 LLM 或无理由列，使用默认偏好。")
    # 组装训练特征
    X_list = []; base_names = None
    for p8 in df_train["_phone8"].tolist():
        base_vec, names = base_feature_vector(p8, feature_set=feature_set)
        if base_names is None: base_names = names
        vec, all_names = pref_augmented_features(base_vec, base_names, p8, global_pref)
        X_list.append(vec)
    X = np.vstack(X_list); y = df_train["专家修改评分"].astype(float).to_numpy()
    # 可选交叉验证
    if kfold and kfold > 1 and len(y) >= kfold:
        print(f"[cv] {kfold}-折交叉验证（ridge={ridge}）...")
        cv = kfold_eval(X, y, k=kfold, ridge=ridge)
        if cv:
            print(f"[cv] MAE={cv['MAE']:.4f}  MSE={cv['MSE']:.4f}  PearsonR={cv['PearsonR']:.4f}")
            with open(os.path.join(dirtrain, "cv_metrics.json"), "w", encoding="utf-8") as f: json.dump(cv, f, ensure_ascii=False, indent=2)
    # 训练模型
    model = LinearRegressor(ridge=ridge)
    model.fit(X, y, feature_names=all_names, global_pref=global_pref)
    meta = {"mode": ps.kind if ps and ps.kind else "NA", "n_train_rows": int(len(df_train)), "created_at": datetime.now().isoformat(timespec="seconds"), "feature_set": feature_set, "ridge": ridge, "trained_on": "专家修改评分"}
    model.meta = meta
    model.save(model_path, meta=meta)
    print(f"[train] 模型已训练并保存：{model_path}")
    return model

# ------------------------------ 预测 ------------------------------
def load_model(dirtrain: str):
    path = os.path.join(dirtrain, "phone_linear_pref.npz")
    if not os.path.exists(path): raise FileNotFoundError(f"未找到模型文件：{path}")
    return LinearRegressor.load(path)
def predict_with_model(df: pd.DataFrame, model: 'LinearRegressor', feature_set: str, out_col: str):
    feature_set = (model.meta or {}).get("feature_set", feature_set)
    base_names = base_feature_vector("00000000", feature_set=feature_set)[1]
    global_pref = model.global_pref
    preds = []
    for p8 in df["_phone8"].tolist():
        if len(p8) == 8:
            base_vec, _ = base_feature_vector(p8, feature_set=feature_set)
            vec, _ = pref_augmented_features(base_vec, base_names, p8, global_pref)
            yhat = float(model.predict(vec.reshape(1, -1))[0])
            preds.append(float(np.clip(np.round(yhat, 1), 1.0, 10.0)))
        else: preds.append(np.nan)
    df[out_col] = preds
def add_simple_columns(df: pd.DataFrame, feature_set: str, reason_style: str="brief"):
    rule_scores, rule_reasons, ai_scores, ai_reasons = [], [], [], []
    for p8 in df["_phone8"].tolist():
        sc_rule = simple_rule_score_1to10(p8) if len(p8)==8 else np.nan
        rule_scores.append(sc_rule)
        rule_reasons.append(simple_rule_explain(p8, sc_rule, style=reason_style))
        sc_ai, rs_ai = simple_ai_score_and_reason(p8, feature_set=feature_set) if len(p8)==8 else (np.nan, "号码不足8位")
        ai_scores.append(sc_ai); ai_reasons.append(rs_ai)
    df["简单规则评分"] = pd.to_numeric(rule_scores, errors="coerce").round(1)
    df["简单规则评分理由"] = rule_reasons
    df["简单AI评分"] = pd.to_numeric(ai_scores, errors="coerce").round(1)
    df["简单AI评分理由"] = ai_reasons
DESIRED_ORDER = ["简单规则评分", "简单规则评分理由", "简单AI评分", "简单AI评分理由", "专家修改评分", "专家修改评分理由"]
def reorder_columns(df: pd.DataFrame):
    for col in DESIRED_ORDER: 
        if col not in df.columns: df[col] = "" if "理由" in col else np.nan
    others = [c for c in df.columns if c not in DESIRED_ORDER]
    return df[others + DESIRED_ORDER]

#%% 🏮👉📱
def main_phone():
    ps = argparse.ArgumentParser(parents=[common_argv()])
    ps.add_argument("--kfold", type=int, default=0, help="K 折交叉验证（>1 启用；NA 模式无效）")
    ps.add_argument("--only_predict", action="store_true", help="仅预测：跳过训练。若存在模型则加载预测")
    ps.add_argument("--feature_set", choices=["full", "minimal"], default="full", help="特征集：full/minimal")
    ps.add_argument("--ridge", type=float, default=0.0, help="岭回归强度（默认 0.0）")
    ps.add_argument("--reason_style", choices=["brief", "detailed"], default="brief", help="简单规则评分理由风格")
    args = ps.parse_args()   
    df = pd.read_excel(args.infile)
    if "联系电话" not in df.columns: raise SystemExit("找不到列：‘联系电话’")
    df["_phone8"] = df["联系电话"].apply(extract_phone8)
    n_total = len(df)
    n_mobile = int((df["_phone8"].str.len() == 8).sum())
    print(f"[stats] 总行数={n_total}，可识别手机号(后8位可用)={n_mobile}")
    add_simple_columns(df, feature_set=args.feature_set, reason_style=args.reason_style)
    if "专家修改评分" in df.columns: df["专家修改评分"] = pd.to_numeric(df["专家修改评分"], errors="coerce")
    else: df["专家修改评分"] = np.nan
    if "专家修改评分理由" not in df.columns: df["专家修改评分理由"] = ""
    outfile = resolve_outfile_path(args.dirtrain, args.outfile, args.model_name)
    print(f"[path] 输出Excel将写入：{outfile}")
    # --- NA 模式：永远不训练 ---
    if args.model_name == "NA":
        df_out = reorder_columns(df)
        df_out.to_excel(outfile, index=False)
        print(f"[done] NA 模式：已写入（不训练）：{outfile}")
        return
    # --- only_predict：加载已有模型并预测（若存在） ---
    if args.only_predict:
        try:
            model = load_model(args.dirtrain)
            out_col = f"{args.model_name}模型评分" if args.model_name == "bert" else "XX模型评分"
            predict_with_model(df, model, args.feature_set, out_col)
        except Exception as e: print(f"[warn] only_predict：加载模型失败（{e}），仅输出简单规则/AI评分。")
        df_out = reorder_columns(df)
        df_out.to_excel(outfile, index=False)
        print(f"[done] only_predict：已写入：{outfile}")
        return
    # --- 训练 ---
    model = train_with_expert(df, args.dirtrain, args.kfold, args.model_name, args.model_param, args.feature_set, args.ridge)
    if model is not None:
        out_col = f"{args.model_name}模型评分" if args.model_name == "bert" else "XX模型评分"
        predict_with_model(df, model, args.feature_set, out_col)
    else: print("[info] 未训练模型：本次将只包含简单列。")
    df_out = reorder_columns(df)
    df_out.to_excel(outfile, index=False)
    print(f"[done] 已写入：{outfile}")
def build_argv_phone(): 
    return common_argv_pass() + [ "--feature_set", "full", "--ridge", "0.0", "--reason_style", "brief",
        *(["--only_predict"] if ONLY_PREDICT else [])]
with argv_ctx(build_argv_phone()): main_phone()

## 🏮🏮🏮🏮🏮🏮🏮🏮🏮🏮



#%% 🏮🏥 Dx 诊断
def set_config(labels, text_cols):
    global LABELS, LABEL2IDX, IDX2LABEL, TEXT_COLS
    LABELS = list(labels)
    LABEL2IDX = {n:i for i,n in enumerate(LABELS)}
    IDX2LABEL = {i:n for i,n in enumerate(LABELS)}
    TEXT_COLS = list(text_cols)
def set_seed(seed: int = 42): random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
def build_text_framewise(df: pd.DataFrame) -> pd.Series: tmp = df.reindex(columns=TEXT_COLS, fill_value='').fillna('').astype(str); return tmp.agg(' | '.join, axis=1)
class EMRDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int, has_label: bool):
        self.df = df.reset_index(drop=True); self.tokenizer = tokenizer; self.max_len = max_len; self.has_label = has_label
    def __len__(self): return len(self.df)
    def __getitem__(self, idx: int):
        text = str(self.df.loc[idx, 'text'])
        enc = self.tokenizer( text, padding='max_length', truncation=True, max_length=self.max_len, return_tensors='pt')
        input_ids = enc['input_ids'].squeeze(0)
        attention_mask = enc['attention_mask'].squeeze(0)
        if self.has_label: label = int(self.df.loc[idx, 'label2']); return input_ids, attention_mask, label
        else: return input_ids, attention_mask
def make_loader(df: pd.DataFrame, tokenizer, max_len: int, bs: int, has_label: bool, shuffle: bool, num_workers: int = 0) -> DataLoader:
    ds = EMRDataset(df, tokenizer, max_len, has_label); return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=num_workers)
def train_one_epoch(model, loader, optimizer, criterion, device, grad_accum=1):
    model.train(); running = 0.0; optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(loader, start=1):
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        loss = criterion(logits, labels) / grad_accum
        loss.backward()
        if step % grad_accum == 0: optimizer.step(); optimizer.zero_grad(set_to_none=True)
        running += loss.item() * grad_accum
    return running / max(1, len(loader))
def evaluate(model, loader, criterion, device):
    model.eval()
    running = 0.0; correct = 0; total = 0
    for batch in loader:
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        loss = criterion(logits, labels)
        running += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    acc = correct / total if total > 0 else 0.0
    return running / max(1, len(loader)), acc
def read_excel_robust(path: str) -> pd.DataFrame:
    try: return pd.read_excel(path)
    except Exception: return pd.read_excel(path, skiprows=4)
def prepare_training_df(df_raw: pd.DataFrame, n_train: int) -> pd.DataFrame:
    if '专家疾病分类' not in df_raw.columns: raise ValueError('输入数据缺少列：专家疾病分类')
    df_labeled = df_raw.dropna(subset=['专家疾病分类']).copy()
    df_labeled = df_labeled.head(n_train)
    df_labeled['text'] = build_text_framewise(df_labeled)
    unknown = []
    def to_label_idx(x): 
        if x in LABELS: return LABEL2IDX[x]; unknown.append(x); return None
    df_labeled['label2'] = df_labeled['专家疾病分类'].map(to_label_idx)
    df_labeled = df_labeled.dropna(subset=['label2']).copy()
    df_labeled['label2'] = df_labeled['label2'].astype(int)
    if unknown: uniq = sorted(set(unknown)); print(f'[警告] 以下专家标签不在 {len(LABELS)} 类列表中，已从训练集中剔除：', uniq)
    if len(df_labeled) < 10: raise ValueError(f'可用于训练的样本过少：{len(df_labeled)} 条。请检查“专家疾病分类”取值是否与 {len(LABELS)} 类一致。')
    return df_labeled[['text', 'label2']]
def stratified_kfold_indices(y: np.ndarray, k: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    labels = np.unique(y)
    per_label_chunks = {}
    for lab in labels: idx = np.where(y == lab)[0]; rng.shuffle(idx); chunks = np.array_split(idx, k); per_label_chunks[lab] = chunks
    folds = []
    for i in range(k):
        val_idx_parts = [per_label_chunks[lab][i] for lab in labels if len(per_label_chunks[lab][i]) > 0]
        val_idx = np.concatenate(val_idx_parts) if len(val_idx_parts) else np.array([], dtype=int)
        all_idx = np.arange(len(y))
        mask = np.ones(len(y), dtype=bool)
        if len(val_idx) > 0: mask[val_idx] = False
        train_idx = all_idx[mask]
        folds.append((train_idx, val_idx))
    return folds
def compute_class_weights(y: np.ndarray, num_labels: int):
    counts = np.bincount(y, minlength=num_labels)
    total = counts.sum()
    denom = np.maximum(counts, 1)
    weights = total / (num_labels * denom.astype(float))
    weights = np.maximum(weights, 1e-6).astype(np.float32)
    return weights
def predict_full(df_raw: pd.DataFrame, tokenizer, model, max_len: int, batch_size: int, device, colname: str, num_workers: int = 0) -> pd.DataFrame:
    df = df_raw.copy()
    df['text'] = build_text_framewise(df)
    ds = EMRDataset(df[['text']], tokenizer, max_len=max_len, has_label=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    preds = []
    model.eval()
    for batch in loader:
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        pred_idx = outputs.logits.argmax(dim=1).detach().cpu().numpy().tolist()
        preds.extend(pred_idx)
    df[colname] = [IDX2LABEL[i] for i in preds]
    return df
def train_and_select(model_name: str, pretrained_path: str, df_train_all: pd.DataFrame, tokenizer, args, device):
    os.makedirs(args.dirtrain, exist_ok=True)
    weights_path = os.path.join(args.dirtrain, f'trained_{model_name}_dx.pth')
    meta_json = os.path.join(args.dirtrain, f'meta_{model_name}.json')
    y_all = df_train_all['label2'].values
    if args.kfold <= 1:
        print(f'[{model_name}] kfold={args.kfold}：不启用交叉验证，直接训练。')
        train_loader = make_loader(df_train_all, tokenizer, args.max_len, args.train_bs, has_label=True, shuffle=True, num_workers=args.num_workers)
        model = AutoModelForSequenceClassification.from_pretrained(pretrained_path, num_labels=len(LABELS)).to(device)
        if args.class_weight:
            w = compute_class_weights(df_train_all['label2'].values, num_labels=len(LABELS))
            w_t = torch.tensor(w, dtype=torch.float32, device=device)
            criterion = nn.CrossEntropyLoss(weight=w_t)
            print(f'[{model_name}] 使用类别加权：', np.round(w, 3).tolist())
        else: criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, grad_accum=args.grad_accum)
            print(f'[{model_name}] Epoch {epoch:02d} | TrainLoss {train_loss:.4f}')
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, weights_path)
        meta = dict(model=model_name, pretrained=pretrained_path, max_len=args.max_len, labels=LABELS)
        with open(meta_json, 'w', encoding='utf-8') as f: json.dump(meta, f, ensure_ascii=False, indent=2)
        del model, optimizer, train_loader
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return {'weights_path': weights_path, 'meta_json': meta_json}
    else:
        k = args.kfold
        print(f'[{model_name}] 启用 {k}-折分层交叉验证')
        folds = stratified_kfold_indices(y_all, k=k, seed=args.seed)
        best_val_loss = float('inf')
        best_state = None
        best_fold_id = -1
        for fi, (tr_idx, va_idx) in enumerate(folds, start=1):
            print(f'\n[{model_name}] ===== Fold {fi}/{k} =====')
            df_tr = df_train_all.iloc[tr_idx].copy()
            df_va = df_train_all.iloc[va_idx].copy()
            train_loader = make_loader(df_tr, tokenizer, args.max_len, args.train_bs, has_label=True, shuffle=True, num_workers=args.num_workers)
            valid_loader = make_loader(df_va, tokenizer, args.max_len, args.valid_bs, has_label=True, shuffle=False, num_workers=args.num_workers)
            model = AutoModelForSequenceClassification.from_pretrained(pretrained_path, num_labels=len(LABELS)).to(device)
            if args.class_weight:
                w = compute_class_weights(df_tr['label2'].values, num_labels=len(LABELS))
                w_t = torch.tensor(w, dtype=torch.float32, device=device)
                criterion = nn.CrossEntropyLoss(weight=w_t)
                print(f'[{model_name}] 使用类别加权：', np.round(w, 3).tolist())
            else: criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            best_val_loss_fold = float('inf')
            best_state_fold = None
            for epoch in range(1, args.epochs + 1):
                train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, grad_accum=args.grad_accum)
                val_loss, val_acc = evaluate(model, valid_loader, criterion, device)
                print(f'[{model_name}] Epoch {epoch:02d} | TrainLoss {train_loss:.4f} | ValLoss {val_loss:.4f} | ValAcc {val_acc:.4f}')
                if val_loss < best_val_loss_fold: best_val_loss_fold = val_loss; best_state_fold = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            if args.save_folds and best_state_fold is not None:
                fold_path = os.path.join(args.dirtrain, f'trained_{model_name}_dx_fold{fi}.pth')
                torch.save(best_state_fold, fold_path)
                print(f'[{model_name}][Fold {fi}] 保存最佳权重：{fold_path}')
            if best_state_fold is not None and best_val_loss_fold < best_val_loss:
                best_val_loss = best_val_loss_fold
                best_state = best_state_fold
                best_fold_id = fi
            del model, optimizer, train_loader, valid_loader
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        if best_state is None: raise RuntimeError(f'[{model_name}] K 折训练未得到有效的最佳模型，请检查数据与参数。')
        torch.save(best_state, weights_path)
        meta = dict(model=model_name, pretrained=pretrained_path, max_len=args.max_len, labels=LABELS, best_fold=best_fold_id)
        with open(meta_json, 'w', encoding='utf-8') as f: json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f'\n[{model_name}][CV] 最佳折 = Fold {best_fold_id}，ValLoss={best_val_loss:.4f}，权重已保存：{weights_path}')
        return {'weights_path': weights_path, 'meta_json': meta_json}

#%%
def main_dx():
    ps = argparse.ArgumentParser(parents=[common_argv()])
    ps.add_argument('--max_len', type=int, default=300)
    ps.add_argument('--epochs', type=int, default=5)
    ps.add_argument('--train_bs', type=int, default=16)
    ps.add_argument('--valid_bs', type=int, default=32)
    ps.add_argument('--pred_bs', type=int, default=64)
    ps.add_argument('--lr', type=float, default=2e-5)
    ps.add_argument('--weight_decay', type=float, default=1e-6)
    ps.add_argument('--seed', type=int, default=42)
    ps.add_argument('--grad_accum', type=int, default=1, help='梯度累积步数')
    ps.add_argument('--num_workers', type=int, default=0, help='DataLoader 线程数')
    ps.add_argument('--kfold', type=int, default=0, help='K 折交叉验证的 K（0/1 表示不启用；例如 5 表示 5 折）')
    ps.add_argument('--class_weight', action='store_true', help='启用类别加权训练（balanced 近似）')
    ps.add_argument('--save_folds', action='store_true', help='在 K 折模式下保存每折最佳权重')
    ps.add_argument('--only_predict', action='store_true', help='只做推理，不训练')
    args = ps.parse_args()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    set_config(DX_LABELS, DX_TEXT_COLS) # 🏮
    df_raw = read_excel_robust(args.infile)
    pred_col = f"{args.model_name}疾病分类"
    pretrained = args.model_param
    print(f'\n===== 模型：{args.model_name} | 权重：{pretrained} =====')
    tokenizer = AutoTokenizer.from_pretrained(pretrained, use_fast=True)
    os.makedirs(args.dirtrain, exist_ok=True)
    weights_path = os.path.join(args.dirtrain, f'trained_{args.model_name}_dx.pth')
    if args.only_predict:
        if not os.path.exists(weights_path): raise FileNotFoundError(f'only_predict 模式下未找到权重：{weights_path}')
        model = AutoModelForSequenceClassification.from_pretrained(pretrained, num_labels=len(LABELS)).to(device)
        state = torch.load(weights_path, map_location='cpu')
        model.load_state_dict(state)
        df_out = predict_full(df_raw=df_raw, tokenizer=tokenizer, model=model, max_len=args.max_len, batch_size=args.pred_bs, device=device, colname=pred_col, num_workers=args.num_workers)
    else:
        df_train_all = prepare_training_df(df_raw, n_train=args.n_train)
        _ = train_and_select(model_name=args.model_name, pretrained_path=pretrained, df_train_all=df_train_all, tokenizer=tokenizer, args=args, device=device)
        model = AutoModelForSequenceClassification.from_pretrained(pretrained, num_labels=len(LABELS)).to(device)
        state = torch.load(weights_path, map_location='cpu')
        model.load_state_dict(state)
        df_out = predict_full(df_raw=df_raw, tokenizer=tokenizer, model=model, max_len=args.max_len, batch_size=args.pred_bs, device=device, colname=pred_col, num_workers=args.num_workers)
    out_dir = os.path.dirname(args.outfile)
    if out_dir: os.makedirs(out_dir, exist_ok=True)
    df_out.to_excel(args.outfile, index=False)
    print(f'\n完成！已写出：{args.outfile}')

DX_LABELS = [
    '儿科','其他-休克','其他-其他症状','其他-意识不清','其他-昏迷','其他-死亡','其他-胸闷',
    '内分泌系统疾病','创伤-交通事故','创伤-其他原因','创伤-暴力事件','创伤-跌倒','创伤-高处坠落',
    '呼吸系统疾病','妇产科','心脏骤停','心血管系统疾病-其他疾病','心血管系统疾病-胸痛','感染性疾病',
    '泌尿系统疾病','消化系统疾病','理化中毒','神经系统疾病-其他疾病','神经系统疾病-脑卒中','精神病'
]
DX_TEXT_COLS = ['呼叫原因','呼救原因','性别','病人性别','年龄','病人年龄','主诉','病情(主诉)','患者症状','主要体征','病史','现病史','初步诊断','补充诊断']
def build_argv_dx(): 
    return common_argv_pass() + ["--max_len", "300", "--epochs", "5", "--train_bs", "16", "--valid_bs", "32", "--pred_bs", "64",
        "--lr", "2e-5", "--weight_decay", "1e-6", "--seed", "42", "--grad_accum", "1", "--num_workers", "0",
        # "--kfold", "5", "--class_weight", "--save_folds",
    ]
with argv_ctx(build_argv_dx()): main_dx()

## 🏮🏮🏮🏮🏮🏮🏮🏮🏮🏮




#%% 🏮🗺 geo 地理 
# 坐标系转换
x_pi = 3.14159265358979324 * 3000.0 / 180.0
pi = 3.1415926535897932384626  # π
a = 6378245.0  # 长半轴
ee = 0.00669342162296594323  # 扁率
def encoding(address, key):
    # 接口地址
    url = "https://api.map.baidu.com/geocoding/v3"
    # 此处填写你在控制台-应用管理-创建应用后获取的AK
    params = {"address": address, "output": "json", "ak": key, "city": GEO_CITY}
    response = requests.get(url=url, params=params)
    if response: return response.json()
    else: print('地址', address, '  地理编码失败')
def gcj02towgs84(lng, lat):
    dlat = transformlat(lng - 105.0, lat - 35.0)
    dlng = transformlng(lng - 105.0, lat - 35.0)
    radlat = lat / 180.0 * pi
    magic = math.sin(radlat)
    magic = 1 - ee * magic * magic
    sqrtmagic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((a * (1 - ee)) / (magic * sqrtmagic) * pi)
    dlng = (dlng * 180.0) / (a / sqrtmagic * math.cos(radlat) * pi)
    mglat = lat + dlat
    mglng = lng + dlng
    return [lng * 2 - mglng, lat * 2 - mglat]
def transformlat(lng, lat):
    ret = -100.0 + 2.0 * lng + 3.0 * lat + 0.2 * lat * lat +  0.1 * lng * lat + 0.2 * math.sqrt(math.fabs(lng))
    ret += (20.0 * math.sin(6.0 * lng * pi) + 20.0 * math.sin(2.0 * lng * pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(lat * pi) + 40.0 * math.sin(lat / 3.0 * pi)) * 2.0 / 3.0
    ret += (160.0 * math.sin(lat / 12.0 * pi) + 320 * math.sin(lat * pi / 30.0)) * 2.0 / 3.0
    return ret
def transformlng(lng, lat):
    ret = 300.0 + lng + 2.0 * lat + 0.1 * lng * lng + 0.1 * lng * lat + 0.1 * math.sqrt(math.fabs(lng))
    ret += (20.0 * math.sin(6.0 * lng * pi) + 20.0 * math.sin(2.0 * lng * pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(lng * pi) + 40.0 * math.sin(lng / 3.0 * pi)) * 2.0 / 3.0
    ret += (150.0 * math.sin(lng / 12.0 * pi) + 300.0 * math.sin(lng / 30.0 * pi)) * 2.0 / 3.0
    return ret
def add_xy(df, key):
    print('还有' + str(len(df[df['现场地址纬度'] == 0])) + '条记录待添加经纬度')
    for i in df.index:
        if df.loc[i, '现场地址纬度'] == 0:
            xy_info = encoding(df.loc[i, '现场地址'], key)
            if xy_info['status'] == 0:
                df.loc[i, '现场地址纬度_原始'] = xy_info['result']['location']['lat']
                df.loc[i, '现场地址经度_原始'] = xy_info['result']['location']['lng']
                lng, lat = gcj02towgs84(xy_info['result']['location']['lng'], xy_info['result']['location']['lat'])
                df.loc[i, '现场地址纬度'] = lat
                df.loc[i, '现场地址经度'] = lng
                df.loc[i, '地址类型'] = xy_info['result']['level']
            elif xy_info['status'] == 2 or xy_info['status'] == 1:
                df.loc[i, '现场地址纬度_原始'] = -1
                df.loc[i, '现场地址经度_原始'] = -1
                df.loc[i, '现场地址纬度'] = -1
                df.loc[i, '现场地址经度'] = -1
                continue
            else:
                print('密钥错误或已达到限额，请切换密钥或明日继续')
                break
    if len(df[df['现场地址经度'] == 0]) == 0: print('地理编码全部完成')
    return df

#%% 🏃‍
def main_geo():
    ps = argparse.ArgumentParser(parents=[common_argv()])
    ps.add_argument('--baidu_ak', type=str, required=True, help='百度地理编码 AK')
    ps.add_argument('--geo_city', type=str, default='深圳市', help='城市名，仅用于日志')
    args = ps.parse_args()
    df = pd.read_excel(args.infile)
    for c in ["现场地址","现场地址纬度","现场地址经度","现场地址纬度_原始","现场地址经度_原始","地址类型"]: 
        if c not in df.columns: df[c] = 0.0 if ("度" in c or "经" in c) else ""
    df = add_xy(df, args.baidu_ak)
    df.to_excel(args.outfile, index=False)
    print(f"[geo] city={args.geo_city} -> {args.outfile}")
def build_argv_geo(): return common_argv_pass() + ["--baidu_ak", BAIDU_AK, "--geo_city", GEO_CITY]
with argv_ctx(build_argv_geo()): main_geo()
