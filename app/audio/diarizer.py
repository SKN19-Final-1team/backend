# stream_batch_diarize_fulltext_allinone.py
import os
import re
import json
import time
import argparse
import random
from typing import List, Dict, Any, Tuple, Optional
from difflib import SequenceMatcher
from collections import defaultdict, Counter

from dotenv import load_dotenv
from openai import OpenAI

# =========================
# Config / Prompts
# =========================
DEFAULT_SYSTEM_PROMPT = (
    "You are a transcript diarization assistant.\n"
    "Given a Raw Stream (ASR-like text), output a JSON object.\n"
    "The object must be: {\"items\": [ ... ]} where each element is {\"speaker\": \"agent\"|\"customer\", \"message\": \"...\"}.\n"
    "Rules:\n"
    "- Keep the original wording as much as possible.\n"
    "- Do not add explanations.\n"
    "- Output ONLY valid JSON (no markdown).\n"
)

SPEAKERS = ("agent", "customer")

DIAR_JSON_SCHEMA = {
    "name": "hana_diarization_object",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "speaker": {"type": "string", "enum": ["agent", "customer"]},
                        "message": {"type": "string"}
                    },
                    "required": ["speaker", "message"],
                    "additionalProperties": False
                }
            }
        },
        "required": ["items"],
        "additionalProperties": False
    }
}

# =========================
# Utilities
# =========================
def read_jsonl_line(path: str, index: int) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == index:
                return json.loads(line)
    raise IndexError(f"index={index} out of range for {path}")


def extract_raw_and_gt(problem: Dict[str, Any]) -> Tuple[str, List[Dict[str, str]], str]:
    """
    Returns:
      raw_stream (str),
      gt_list (list of {speaker,message}),
      system_prompt (str)
    """
    msgs = problem.get("messages", [])
    sys_prompt = None
    raw = None
    gt = None

    for m in msgs:
        if m["role"] == "system":
            sys_prompt = m["content"]
        elif m["role"] == "user":
            raw = m["content"]
        elif m["role"] == "assistant":
            gt = m["content"]

    if raw is None or gt is None:
        raise ValueError("Invalid jsonl format: missing user/assistant messages")

    # raw may start with "Raw Stream:\n"
    raw = raw.replace("\r\n", "\n")
    if raw.strip().lower().startswith("raw stream"):
        # keep after first newline if exists
        parts = raw.split("\n", 1)
        raw = parts[1] if len(parts) > 1 else raw

    gt_list = json.loads(gt)
    if not isinstance(gt_list, list):
        raise ValueError("assistant content is not a JSON array")

    if sys_prompt is None:
        sys_prompt = DEFAULT_SYSTEM_PROMPT

    return raw.strip(), gt_list, sys_prompt


def clean_text_basic(s: str) -> str:
    s = s.replace("\r", " ").replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def remove_new_start_tokens(s: str) -> str:
    # handle variations
    s = re.sub(r"<\s*NEW_START\s*>", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\bNEW_START\b", "", s, flags=re.IGNORECASE)
    return s


def normalize_for_compare(s: str) -> str:
    # remove spaces only (keeps chars order)
    return re.sub(r"\s+", "", s)


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()





def is_spam_repetition(msg: str,
                       min_len_nospace: int = 10,
                       max_unique_chars: int = 2) -> bool:
    """
    Detect low-entropy repetition spam like:
    - "아아아아아아아아..."
    - "ㅋㅋㅋㅋㅋㅋㅋㅋ..."
    - "000000000000..."
    기준(주인님 요청 반영):
    - 공백 제거 후 길이 >= min_len_nospace
    - 고유 문자 수 <= max_unique_chars
    """
    if not msg:
        return False
    s = normalize_for_compare(msg)  # removes spaces
    if len(s) < min_len_nospace:
        return False
    uniq = set(s)
    if len(uniq) <= max_unique_chars:
        return True
    return False


def filter_spam_items(items: List[Dict[str, str]],
                      min_len_nospace: int = 10,
                      max_unique_chars: int = 2) -> List[Dict[str, str]]:
    if not items:
        return []
    out = []
    for it in items:
        msg = clean_text_basic(it.get("message", ""))
        if not msg:
            continue
        if is_spam_repetition(msg, min_len_nospace=min_len_nospace, max_unique_chars=max_unique_chars):
            continue
        out.append({"speaker": it["speaker"], "message": msg})
    return out


def _norm_overlap(s: str) -> str:
    # more aggressive normalization for overlap/prefix detection
    s = normalize_for_compare(s)
    s = re.sub(r"[\"'`]", "", s)
    s = re.sub(r"[·•…]", "", s)
    s = re.sub(r"[,\.\!\?\:\;\(\)\[\]\{\}]", "", s)
    return s


def find_best_fuzzy_overlap_suffix_prefix(a: str,
                                         b: str,
                                         min_chars: int = 12,
                                         min_sim: float = 0.92) -> int:
    """
    Find best overlap length L where suffix(a, L) ≈ prefix(b, L).
    Returns L (0 if none). Works on already-normalized strings.
    """
    if not a or not b:
        return 0
    Lmax = min(len(a), len(b))
    for L in range(Lmax, min_chars - 1, -1):
        sa = a[-L:]
        sb = b[:L]
        if sa == sb:
            return L
        # allow small diffs
        if similarity(sa, sb) >= min_sim:
            return L
    return 0


def drop_fuzzy_overlap_by_utterances(global_items: List[Dict[str, str]],
                                     new_items: List[Dict[str, str]],
                                     max_k: int = 8,
                                     sim_th: float = 0.93) -> List[Dict[str, str]]:
    """
    Like drop_exact_overlap_by_utterances, but allows near-duplicate messages via similarity.
    Speakers must match for each position.
    """
    if not global_items or not new_items:
        return new_items
    g = global_items
    n = new_items
    max_k = min(max_k, len(g), len(n))
    for k in range(max_k, 0, -1):
        ok = True
        for i in range(k):
            if g[-k + i]["speaker"] != n[i]["speaker"]:
                ok = False
                break
            a = _norm_overlap(g[-k + i]["message"])
            b = _norm_overlap(n[i]["message"])
            if not a or not b:
                ok = False
                break
            if similarity(a, b) < sim_th:
                ok = False
                break
        if ok:
            return n[k:]
    return n


def _is_short_filler(msg: str) -> bool:
    s = _norm_overlap(msg)
    if not s:
        return True
    # very short acknowledgements / fillers
    return len(s) <= 3


def drop_boundary_prefix(global_items: List[Dict[str, str]],
                         new_items: List[Dict[str, str]],
                         lookback_utts: int = 3,
                         sim_th: float = 0.93) -> List[Dict[str, str]]:
    """
    Prefix 제거:
    - new prefix가 global tail과 거의 같은 발화면(동일 speaker) prefix drop
    - cross-speaker 보조: new[0]이 짧은 filler이고, new[1]이 global tail과 겹치면 new[1] 쪽을 trim/drop
    """
    if not global_items or not new_items:
        return new_items

    tail = global_items[-min(lookback_utts, len(global_items)):]
    # 1) same-speaker near-dup on first item → drop it
    first = new_items[0]
    for t in reversed(tail):
        if t["speaker"] != first["speaker"]:
            continue
        a = _norm_overlap(t["message"])
        b = _norm_overlap(first["message"])
        if a and b and similarity(a, b) >= sim_th:
            return new_items[1:]
        # if new is a prefix of tail (or vice versa) with good sim on that part
        L = find_best_fuzzy_overlap_suffix_prefix(a, b, min_chars=min(12, len(a), len(b)), min_sim=0.94)
        if L and L >= 12 and L == len(b):
            return new_items[1:]

    # 2) cross-speaker helper: short filler then duplicated next utterance
    if len(new_items) >= 2 and _is_short_filler(new_items[0]["message"]):
        second = new_items[1]
        for t in reversed(tail):
            if t["speaker"] != second["speaker"]:
                continue
            a = _norm_overlap(t["message"])
            b = _norm_overlap(second["message"])
            if a and b and similarity(a, b) >= sim_th:
                # drop duplicated second utterance, keep filler
                return [new_items[0]] + new_items[2:]
    return new_items


def trim_partial_overlap_last_first(global_items: List[Dict[str, str]],
                                   new_items: List[Dict[str, str]],
                                   min_chars: int = 12,
                                   min_sim: float = 0.92) -> List[Dict[str, str]]:
    """
    유사도 기반 overlap 트리밍 (same speaker boundary).
    - global last message suffix ≈ new first message prefix → new first에서 겹치는 prefix 제거
    - cross-speaker 보조: new[0]이 짧은 filler이고 new[1]이 same speaker면 new[1] 쪽으로도 시도
    """
    if not global_items or not new_items:
        return new_items

    def _trim_one(raw_msg: str, overlap_len: int) -> str:
        # trim first overlap_len non-space chars from original raw string
        cut = 0
        removed = 0
        for idx, ch in enumerate(raw_msg):
            if ch.isspace():
                continue
            removed += 1
            if removed >= overlap_len:
                cut = idx + 1
                break
        return clean_text_basic(raw_msg[cut:])

    last = global_items[-1]

    # case A: same speaker on first
    if last["speaker"] == new_items[0]["speaker"]:
        g = _norm_overlap(last["message"])
        n0_raw = new_items[0]["message"]
        n0 = _norm_overlap(n0_raw)
        L = find_best_fuzzy_overlap_suffix_prefix(g, n0, min_chars=min_chars, min_sim=min_sim)
        if L:
            trimmed = _trim_one(n0_raw, L)
            if trimmed:
                new2 = new_items.copy()
                new2[0] = {"speaker": new_items[0]["speaker"], "message": trimmed}
                return new2
            else:
                return new_items[1:]

    # case B: cross-speaker helper
    if len(new_items) >= 2 and _is_short_filler(new_items[0]["message"]) and last["speaker"] == new_items[1]["speaker"]:
        g = _norm_overlap(last["message"])
        n1_raw = new_items[1]["message"]
        n1 = _norm_overlap(n1_raw)
        L = find_best_fuzzy_overlap_suffix_prefix(g, n1, min_chars=min_chars, min_sim=min_sim)
        if L:
            trimmed = _trim_one(n1_raw, L)
            if trimmed:
                new2 = new_items.copy()
                new2[1] = {"speaker": new_items[1]["speaker"], "message": trimmed}
                return new2
            else:
                # if fully overlapped, drop second
                return [new_items[0]] + new_items[2:]

    return new_items

def _strip_code_fences(text: str) -> str:
    t = text.strip()
    if "```" in t:
        # remove leading fence
        t = re.sub(r"^\s*```[a-zA-Z0-9_-]*\s*", "", t)
        # remove trailing fence
        t = re.sub(r"\s*```\s*$", "", t)
        t = t.strip()
    return t

def _extract_first_json_candidate(text: str) -> Optional[str]:
    """
    Extract the first balanced JSON object/array substring from text using bracket matching.
    Returns None if not found.
    """
    t = text
    # find first '{' or '['
    starts = [(t.find("{"), "{"), (t.find("["), "[")]
    starts = [(i, ch) for i, ch in starts if i != -1]
    if not starts:
        return None
    i0, ch0 = min(starts, key=lambda x: x[0])
    stack = []
    in_str = False
    esc = False
    for i in range(i0, len(t)):
        c = t[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        else:
            if c == '"':
                in_str = True
                continue
            if c in "{[":
                stack.append(c)
            elif c in "}]":
                if not stack:
                    break
                top = stack[-1]
                if (top == "{" and c == "}") or (top == "[" and c == "]"):
                    stack.pop()
                    if not stack:
                        return t[i0:i+1]
                else:
                    # mismatched closing - give up
                    break
    return None

def _salvage_truncated_items(text: str) -> list[dict] | None:
    """
    Best-effort salvage when output looks like truncated JSON.
    Strategy: find an items array (or raw array), cut to last complete object '}' and close brackets.
    This is useful when max_output_tokens truncates output.
    """
    t = text
    # Prefer object wrapper
    key_pos = t.find('"items"')
    if key_pos != -1:
        arr_start = t.find("[", key_pos)
        if arr_start != -1:
            last_obj = t.rfind("}", arr_start)
            if last_obj != -1 and last_obj > arr_start:
                cand = t[:last_obj+1]
                # ensure we start from '{'
                obj_start = cand.find("{")
                if obj_start == -1:
                    return None
                cand = cand[obj_start:]
                # close array/object
                if cand.count("[") > cand.count("]"):
                    cand += "]"
                if cand.count("{") > cand.count("}"):
                    cand += "}"
                try:
                    obj = json.loads(cand)
                    if isinstance(obj, dict) and isinstance(obj.get("items"), list):
                        return obj["items"]
                except Exception:
                    return None
    # Fallback: raw array
    arr_start = t.find("[")
    if arr_start != -1:
        last_obj = t.rfind("}", arr_start)
        if last_obj != -1 and last_obj > arr_start:
            cand = t[arr_start:last_obj+1] + "]"
            try:
                obj = json.loads(cand)
                if isinstance(obj, list):
                    return obj
            except Exception:
                return None
    return None

def parse_json_array_loose(text: str):
    """
    Parse model output into a list of dict items.
    Accepts:
      - JSON array
      - JSON object {"items": [...]}
      - text containing JSON (with/without code fences)
    Includes recovery for truncated JSON outputs.
    """
    t = _strip_code_fences(text)
    if not t:
        raise ValueError("Empty model output")

    # 1) direct parse
    try:
        obj = json.loads(t)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict) and isinstance(obj.get("items"), list):
            return obj["items"]
    except Exception:
        pass

    # 2) extract first balanced JSON candidate
    cand = _extract_first_json_candidate(t)
    if cand:
        try:
            obj = json.loads(cand)
            if isinstance(obj, list):
                return obj
            if isinstance(obj, dict) and isinstance(obj.get("items"), list):
                return obj["items"]
        except Exception:
            pass

    # 3) salvage truncated
    salv = _salvage_truncated_items(t)
    if salv is not None:
        return salv

    # 4) last resort regex array
    m = re.search(r"\[.*\]", t, flags=re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass

    raise ValueError("Model output does not contain a JSON array")

def ensure_schema(items: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    out = []
    for it in items:
        sp = it.get("speaker")
        msg = it.get("message")
        if sp not in SPEAKERS:
            # try map common variants
            if sp in ("agent", "counselor", "상담사", "상담원"):
                sp = "agent"
            elif sp in ("customer", "client", "손님", "고객"):
                sp = "customer"
        if sp not in SPEAKERS:
            continue
        if not isinstance(msg, str):
            continue
        msg = remove_new_start_tokens(msg)
        msg = clean_text_basic(msg)
        if msg:
            out.append({"speaker": sp, "message": msg})
    return out


def merge_same_speaker(items: List[Dict[str, str]]) -> List[Dict[str, str]]:
    if not items:
        return []
    merged = [items[0].copy()]
    for it in items[1:]:
        if it["speaker"] == merged[-1]["speaker"]:
            merged[-1]["message"] = clean_text_basic(merged[-1]["message"] + " " + it["message"])
        else:
            merged.append(it.copy())
    return merged


def split_sentences_ko(msg: str, max_len: int = 120) -> List[str]:
    """
    Light sentence splitting: punctuation first, then fallback split by length.
    """
    msg = clean_text_basic(msg)
    if len(msg) <= max_len:
        return [msg]

    # split by punctuation boundaries
    parts = re.split(r"([\.!\?。！？])", msg)
    sents = []
    buf = ""
    for p in parts:
        if not p:
            continue
        buf += p
        if re.fullmatch(r"[\.!\?。！？]", p):
            sents.append(clean_text_basic(buf))
            buf = ""
    if buf.strip():
        sents.append(clean_text_basic(buf))

    # if still too long, chunk by length
    out = []
    for s in sents:
        if len(s) <= max_len:
            out.append(s)
        else:
            cur = s
            while len(cur) > max_len:
                out.append(clean_text_basic(cur[:max_len]))
                cur = cur[max_len:]
            if cur.strip():
                out.append(clean_text_basic(cur))
    return [x for x in out if x]


def apply_sentence_split(items: List[Dict[str, str]], max_len: int = 120) -> List[Dict[str, str]]:
    out = []
    for it in items:
        sents = split_sentences_ko(it["message"], max_len=max_len)
        for s in sents:
            if s:
                out.append({"speaker": it["speaker"], "message": s})
    return out


def dedupe_near_duplicates(items: List[Dict[str, str]], ratio: float = 0.95) -> List[Dict[str, str]]:
    """
    Remove duplicated/near-duplicated consecutive utterances (often caused by overlap or model echo).
    """
    if not items:
        return []
    out = [items[0].copy()]
    for cur in items[1:]:
        prev = out[-1]
        a = normalize_for_compare(prev["message"])
        b = normalize_for_compare(cur["message"])
        if not a or not b:
            out.append(cur.copy())
            continue

        sim = similarity(a, b)
        if sim >= ratio:
            # keep the longer one; if equal, keep prev
            if len(b) > len(a):
                out[-1] = cur.copy()
            # else drop cur
            continue

        out.append(cur.copy())
    return out


def drop_exact_overlap_by_utterances(global_items: List[Dict[str, str]],
                                    new_items: List[Dict[str, str]],
                                    max_k: int = 8) -> List[Dict[str, str]]:
    """
    If last k utterances of global == first k utterances of new (speaker+normalized msg), drop that prefix.
    """
    if not global_items or not new_items:
        return new_items
    g = global_items
    n = new_items
    max_k = min(max_k, len(g), len(n))
    for k in range(max_k, 0, -1):
        ok = True
        for i in range(k):
            if g[-k + i]["speaker"] != n[i]["speaker"]:
                ok = False
                break
            if normalize_for_compare(g[-k + i]["message"]) != normalize_for_compare(n[i]["message"]):
                ok = False
                break
        if ok:
            return n[k:]
    return n


def trim_partial_overlap_last_first(global_items: List[Dict[str, str]],
                                   new_items: List[Dict[str, str]],
                                   min_chars: int = 12,
                                   min_sim: float = 0.92) -> List[Dict[str, str]]:
    """
    유사도 기반 overlap 트리밍 (same speaker boundary).
    - global last message suffix ≈ new first message prefix → new first에서 겹치는 prefix 제거
    - cross-speaker 보조: new[0]이 짧은 filler이고 new[1]이 same speaker면 new[1] 쪽으로도 시도
    """
    if not global_items or not new_items:
        return new_items

    def _trim_one(raw_msg: str, overlap_len: int) -> str:
        # trim first overlap_len non-space chars from original raw string
        cut = 0
        removed = 0
        for idx, ch in enumerate(raw_msg):
            if ch.isspace():
                continue
            removed += 1
            if removed >= overlap_len:
                cut = idx + 1
                break
        return clean_text_basic(raw_msg[cut:])

    last = global_items[-1]

    # case A: same speaker on first
    if last["speaker"] == new_items[0]["speaker"]:
        g = _norm_overlap(last["message"])
        n0_raw = new_items[0]["message"]
        n0 = _norm_overlap(n0_raw)
        L = find_best_fuzzy_overlap_suffix_prefix(g, n0, min_chars=min_chars, min_sim=min_sim)
        if L:
            trimmed = _trim_one(n0_raw, L)
            if trimmed:
                new2 = new_items.copy()
                new2[0] = {"speaker": new_items[0]["speaker"], "message": trimmed}
                return new2
            else:
                return new_items[1:]

    # case B: cross-speaker helper
    if len(new_items) >= 2 and _is_short_filler(new_items[0]["message"]) and last["speaker"] == new_items[1]["speaker"]:
        g = _norm_overlap(last["message"])
        n1_raw = new_items[1]["message"]
        n1 = _norm_overlap(n1_raw)
        L = find_best_fuzzy_overlap_suffix_prefix(g, n1, min_chars=min_chars, min_sim=min_sim)
        if L:
            trimmed = _trim_one(n1_raw, L)
            if trimmed:
                new2 = new_items.copy()
                new2[1] = {"speaker": new_items[1]["speaker"], "message": trimmed}
                return new2
            else:
                # if fully overlapped, drop second
                return [new_items[0]] + new_items[2:]

    return new_items

def merge_batches(global_items: List[Dict[str, str]],
                  batch_items: List[Dict[str, str]],
                  max_overlap_utts: int = 8,
                  min_partial_overlap_chars: int = 12) -> List[Dict[str, str]]:
    """
    global + batch merge with:
    - exact overlap drop (speaker+exact msg)
    - fuzzy overlap drop (speaker+near-dup)
    - prefix drop helper (same speaker + cross-speaker filler assist)
    - fuzzy partial overlap trimming
    - same-speaker merge
    - spam repetition removal
    - near-duplicate dedupe
    """
    if not batch_items:
        return global_items

    # 0) boundary prefix helper (handles "filler + duplicated next utterance" cases)
    batch_items = drop_boundary_prefix(global_items, batch_items, lookback_utts=3, sim_th=0.93)

    # 1) drop exact overlap by utterances
    batch_items = drop_exact_overlap_by_utterances(global_items, batch_items, max_k=max_overlap_utts)

    # 2) drop fuzzy overlap by utterances
    batch_items = drop_fuzzy_overlap_by_utterances(global_items, batch_items, max_k=max_overlap_utts, sim_th=0.93)

    # 3) partial overlap trim on boundary (fuzzy)
    batch_items = trim_partial_overlap_last_first(global_items, batch_items, min_chars=min_partial_overlap_chars, min_sim=0.92)

    merged = global_items + batch_items

    # 4) merge same speaker
    merged = merge_same_speaker(merged)

    # 5) spam repetition removal (aggressive)
    merged = filter_spam_items(merged, min_len_nospace=10, max_unique_chars=2)

    # 6) remove near duplicates
    merged = dedupe_near_duplicates(merged, ratio=0.95)

    return merged

def simulate_stt_fragments(raw_stream: str,
                           mean_chars: int = 24,
                           std_chars: int = 10,
                           min_chars: int = 10,
                           max_chars: int = 60,
                           seed: int = 42) -> List[str]:
    """
    Simulate ASR partial chunks.
    """
    rng = random.Random(seed)
    s = clean_text_basic(raw_stream)
    frags = []
    i = 0
    while i < len(s):
        L = int(rng.gauss(mean_chars, std_chars))
        L = max(min_chars, min(max_chars, L))
        frag = s[i:i+L]
        i += L
        frags.append(frag)
    return frags


# =========================
# Model call
# =========================

async def call_diarizer_fulltext(
    client,
    model: str,
    system_prompt: str,
    raw_stream_batch: str,
    temperature: float = 0.0,
    max_output_tokens: int = 2500,
    retries: int = 2,
):
    """
    Calls the fine-tuned diarizer and returns items=[{speaker,message},...].

    Reliability strategy:
    - Attempt 1: normal diarization output (merged utterances)
    - Attempt 2: stricter JSON-only + higher max tokens
    - Attempt 3: per-fragment fallback (small output, very hard to break)
    - Also includes truncated-output salvage in parse_json_array_loose()
    """
    schema = {
        "name": "hana_diarization_object",
        "schema": {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "speaker": {"type": "string", "enum": ["agent", "customer"]},
                            "message": {"type": "string"},
                        },
                        "required": ["speaker", "message"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["items"],
            "additionalProperties": False,
        },
        "strict": True,
    }

    # Prebuild a stable "data" block so retries don't accidentally change inputs.
    data_block = (
        "### STT fragments (time order)\n"
        f"{raw_stream_batch}\n\n"
        "### Task\n"
        "Return diarized utterances in JSON ONLY.\n"
    )

    # build prompt variants
    def make_user_prompt(mode: str) -> str:
        if mode == "normal":
            return (
                data_block
                + "You may merge adjacent fragments if they are clearly the same speaker.\n"
                  "Keep the number of utterances reasonable.\n"
                  "Output MUST be a JSON object: {\"items\": [...]} only.\n"
            )
        if mode == "strict":
            return (
                data_block
                + "IMPORTANT:\n"
                  "- Output MUST be valid JSON object exactly matching {\"items\": [...]}.\n"
                  "- No markdown, no code fences, no commentary.\n"
                  "- Be concise. Prefer splitting long messages over making one huge message.\n"
                  "Output MUST be a JSON object only.\n"
            )
        # per-fragment fallback
        return (
            data_block
            + "FALLBACK MODE (robust):\n"
              "- Output EXACTLY one item per fragment, in the same order.\n"
              "- message MUST equal the fragment text exactly (do not rewrite).\n"
              "- Only decide speaker per fragment.\n"
              "Output MUST be a JSON object: {\"items\": [...]} only.\n"
        )

    attempt_modes = ["normal", "strict", "per_fragment"]
    max_attempts = 1 + max(0, retries)
    last_err = None

    for attempt in range(max_attempts):
        mode = attempt_modes[min(attempt, len(attempt_modes) - 1)]
        user_prompt = make_user_prompt(mode)

        # Expand output budget on retry #1 because truncation is a common JSON-break cause.
        # Keep it bounded to avoid runaway latency.
        if attempt == 0:
            out_budget = max_output_tokens
        elif attempt == 1:
            out_budget = max(max_output_tokens, 4000)
        else:
            out_budget = max(max_output_tokens, 2500)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            t0 = time.time()
            resp = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=out_budget,
                response_format={"type": "json_schema", "json_schema": schema},
            )
            api_sec = time.time() - t0
            text = resp.choices[0].message.content or ""
            items = parse_json_array_loose(text)

            # normalize basic schema safety
            cleaned = []
            for it in items:
                if not isinstance(it, dict):
                    continue
                sp = it.get("speaker")
                msg = it.get("message")
                if sp not in ("agent", "customer"):
                    continue
                if not isinstance(msg, str):
                    continue
                cleaned.append({"speaker": sp, "message": msg})
            if not cleaned:
                raise ValueError("Parsed items is empty after cleaning")

            usage = getattr(resp, "usage", None)
            usage_dict = None
            if usage is not None:
                usage_dict = {
                    "prompt_tokens": getattr(usage, "prompt_tokens", None),
                    "completion_tokens": getattr(usage, "completion_tokens", None),
                    "total_tokens": getattr(usage, "total_tokens", None),
                }
            finish_reason = None
            try:
                finish_reason = resp.choices[0].finish_reason
            except Exception:
                pass
            resp_id = getattr(resp, "id", None)
            resp_model = getattr(resp, "model", None)
            resp_created = getattr(resp, "created", None)

            prompt_chars = 0
            try:
                prompt_chars = sum(len(m.get("content", "") or "") for m in messages)
            except Exception:
                prompt_chars = None

            meta = {
                "mode": mode,
                "out_budget": out_budget,
                "attempt_used": attempt + 1,
                "prompt_chars": prompt_chars,
                "output_chars": len(text or ""),
                "usage": usage_dict,
                "finish_reason": finish_reason,
                "response_id": resp_id,
                "response_model": resp_model,
                "response_created": resp_created,
                "last_err": None,
            }

            return cleaned, api_sec, meta

        except Exception as e:
            last_err = e
            # If final attempt, raise
            if attempt == max_attempts - 1:
                raise ValueError(f"Failed to parse JSON array from model output. last_err={last_err}") from None
            # otherwise retry
            continue

    raise ValueError(f"Failed to parse JSON array from model output. last_err={last_err}")

def build_char_stream(items: List[Dict[str, str]]) -> Tuple[str, List[str], List[int]]:
    """
    Returns:
      text_chars_joined (no-space),
      speaker_label_per_char,
      utterance_index_per_char
    """
    chars = []
    labels = []
    utt_idx = []
    for i, it in enumerate(items):
        t = normalize_for_compare(it["message"])
        chars.extend(list(t))
        labels.extend([it["speaker"]] * len(t))
        utt_idx.extend([i] * len(t))
    return "".join(chars), labels, utt_idx


def eval_char_aligned(gt_items: List[Dict[str, str]],
                      pred_items: List[Dict[str, str]],
                      min_err_span_chars: int = 20,
                      context_chars: int = 40,
                      top_k_spans: int = 20) -> Dict[str, Any]:
    """
    Alignment-based diarization accuracy using equal-character matches only.
    Produces mismatch spans + 주변 snippet.
    """
    gt_text, gt_lab, gt_u = build_char_stream(gt_items)
    pr_text, pr_lab, pr_u = build_char_stream(pred_items)

    sm = SequenceMatcher(None, gt_text, pr_text)
    equal_ops = [(i1, i2, j1, j2) for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag == "equal"]

    conf = defaultdict(int)
    total = 0
    correct = 0
    mismatches = []

    for i1, i2, j1, j2 in equal_ops:
        L = i2 - i1
        for k in range(L):
            gl = gt_lab[i1 + k]
            pl = pr_lab[j1 + k]
            conf[(gl, pl)] += 1
            total += 1
            if gl == pl:
                correct += 1
            else:
                mismatches.append((i1 + k, j1 + k))

    coverage = (total / len(gt_text)) if gt_text else 0.0
    acc = (correct / total) if total else 0.0

    # group mismatch spans (by gt index continuity)
    spans = []
    if mismatches:
        a0, b0 = mismatches[0]
        pa, pb = a0, b0
        for a, b in mismatches[1:]:
            if a == pa + 1 and b == pb + 1:
                pa, pb = a, b
            else:
                spans.append((a0, pa, b0, pb))
                a0, b0 = a, b
                pa, pb = a, b
        spans.append((a0, pa, b0, pb))

    # build detailed span info
    span_details = []
    for (ga, gb, pa, pb) in spans:
        span_len = gb - ga + 1
        if span_len < min_err_span_chars:
            continue

        gs = max(0, ga - context_chars)
        ge = min(len(gt_text), gb + context_chars + 1)
        ps = max(0, pa - context_chars)
        pe = min(len(pr_text), pb + context_chars + 1)

        gt_snip = gt_text[gs:ge]
        pr_snip = pr_text[ps:pe]

        gt_span_labels = gt_lab[ga:gb+1]
        pr_span_labels = pr_lab[pa:pb+1]
        gt_major = Counter(gt_span_labels).most_common(1)[0][0] if gt_span_labels else None
        pr_major = Counter(pr_span_labels).most_common(1)[0][0] if pr_span_labels else None

        gt_utt_range = (gt_u[ga], gt_u[gb]) if gt_u else (None, None)
        pr_utt_range = (pr_u[pa], pr_u[pb]) if pr_u else (None, None)

        span_details.append({
            "span_len_chars": span_len,
            "gt_char_range": [ga, gb],
            "pred_char_range": [pa, pb],
            "gt_major_speaker": gt_major,
            "pred_major_speaker": pr_major,
            "gt_utt_range": list(gt_utt_range),
            "pred_utt_range": list(pr_utt_range),
            "gt_snippet": gt_snip,
            "pred_snippet": pr_snip,
        })

    # sort spans by length desc
    span_details.sort(key=lambda x: x["span_len_chars"], reverse=True)
    span_details = span_details[:top_k_spans]

    # confusion matrix friendly
    cm = {
        "labels": ["agent", "customer"],
        "matrix": [
            [conf[("agent", "agent")], conf[("agent", "customer")]],
            [conf[("customer", "agent")], conf[("customer", "customer")]],
        ]
    }

    return {
        "acc_aligned_chars": acc,
        "coverage_aligned_chars": coverage,
        "total_aligned_chars": total,
        "gt_chars": len(gt_text),
        "pred_chars": len(pr_text),
        "confusion": cm,
        "error_spans": span_details,
    }


# =========================
# Main pipeline
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_jsonl", required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--model", required=True)

    parser.add_argument("--simulate_stt", action="store_true")
    parser.add_argument("--frag_interval_sec", type=float, default=1.5)
    parser.add_argument("--batch_window_sec", type=float, default=30.0)
    parser.add_argument("--overlap_sec", type=float, default=3.0)

    parser.add_argument("--frag_mean_chars", type=int, default=24)
    parser.add_argument("--frag_std_chars", type=int, default=10)
    parser.add_argument("--frag_min_chars", type=int, default=10)
    parser.add_argument("--frag_max_chars", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--max_output_tokens", type=int, default=2000)
    parser.add_argument("--out_prefix", default="callTranscript_pred_fulltext_allinone")

    
    parser.add_argument("--run_tag", type=str, default="", help="Output filename tag, e.g. index321_1 (appended to all outputs).")
    parser.add_argument("--sentence_split", action="store_true")
    parser.add_argument("--sentence_split_max_len", type=int, default=120)

    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--min_err_span_chars", type=int, default=20)

    # Optional: evaluate an already-produced prediction JSON (skip API)
    parser.add_argument("--pred_json", default=None)

    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not found. Put it in .env or environment variables.")

    problem = read_jsonl_line(args.test_jsonl, args.index)
    raw, gt_list, system_prompt = extract_raw_and_gt(problem)

    # If pred_json is provided, skip API
    if args.pred_json:
        with open(args.pred_json, "r", encoding="utf-8") as f:
            pred_items = json.load(f)
        api_logs = []
        tail_latency_parallel = None
        tail_latency_single = None
        assumed_call_duration = None
        total_wall = None
    else:
        client = OpenAI(api_key=api_key)

        # Build fragments
        if args.simulate_stt:
            frags = simulate_stt_fragments(
                raw,
                mean_chars=args.frag_mean_chars,
                std_chars=args.frag_std_chars,
                min_chars=args.frag_min_chars,
                max_chars=args.frag_max_chars,
                seed=args.seed,
            )
        else:
            # fallback: treat as single fragment
            frags = [clean_text_basic(raw)]

        frag_interval = args.frag_interval_sec
        batch_window = args.batch_window_sec
        overlap = args.overlap_sec

        batch_size = max(1, int(round(batch_window / frag_interval)))
        overlap_size = max(0, int(round(overlap / frag_interval)))
        step = batch_size

        assumed_call_duration = len(frags) * frag_interval

        api_logs = []
        global_items: List[Dict[str, str]] = []

        # timing simulation
        prev_finish_single = 0.0
        t_wall0 = time.perf_counter()

        batch_id = 0
        for start in range(0, len(frags), step):
            end = min(start + batch_size, len(frags))
            inc_start = max(0, start - overlap_size)
            inc_end = end

            batch_text = " ".join(frags[inc_start:inc_end]).strip()

            scheduled_time = end * frag_interval  # batch is available at this time
            # single-worker: cannot start before previous finishes
            start_single = max(scheduled_time, prev_finish_single)

            # call API (offline it's sequential, but we log ideal vs single-worker)
            items, api_sec, meta = call_diarizer_fulltext(
                client=client,
                model=args.model,
                system_prompt=system_prompt,
                raw_stream_batch=batch_text,
                temperature=0.0,
                max_output_tokens=args.max_output_tokens,
                retries=2,
            )

            # cleanup per batch
            items = merge_same_speaker(items)
            items = dedupe_near_duplicates(items, ratio=0.95)

            # merge into global
            global_items = merge_batches(
                global_items,
                items,
                max_overlap_utts=8,
                min_partial_overlap_chars=12,
            )

            # optional sentence split at the end (or per batch)
            if args.sentence_split:
                global_items = apply_sentence_split(global_items, max_len=args.sentence_split_max_len)
                global_items = merge_same_speaker(global_items)
                global_items = dedupe_near_duplicates(global_items, ratio=0.95)

            finish_parallel = scheduled_time + api_sec
            finish_single = start_single + api_sec
            prev_finish_single = finish_single

            api_logs.append({
                "batch_id": batch_id,
                "frag_range_included": [inc_start, inc_end - 1],
                "frag_range_main": [start, end - 1],
                "scheduled_time_sec": scheduled_time,
                "api_sec": api_sec,
                "mode": meta.get("mode") if isinstance(meta, dict) else None,
                "out_budget": meta.get("out_budget") if isinstance(meta, dict) else None,
                "attempt_used": meta.get("attempt_used") if isinstance(meta, dict) else None,
                "prompt_chars": meta.get("prompt_chars") if isinstance(meta, dict) else None,
                "output_chars": meta.get("output_chars") if isinstance(meta, dict) else None,
                "prompt_tokens": (meta.get("usage") or {}).get("prompt_tokens") if isinstance(meta, dict) else None,
                "completion_tokens": (meta.get("usage") or {}).get("completion_tokens") if isinstance(meta, dict) else None,
                "total_tokens": (meta.get("usage") or {}).get("total_tokens") if isinstance(meta, dict) else None,
                "finish_reason": meta.get("finish_reason") if isinstance(meta, dict) else None,
                "response_id": meta.get("response_id") if isinstance(meta, dict) else None,
                "response_model": meta.get("response_model") if isinstance(meta, dict) else None,
                "response_created": meta.get("response_created") if isinstance(meta, dict) else None,
"finish_time_parallel_sec": finish_parallel,
                "start_time_single_sec": start_single,
                "finish_time_single_sec": finish_single,
                "batch_text_chars": len(batch_text),
                "batch_items": len(items),
                "global_items_now": len(global_items),
            })

            batch_id += 1

        total_wall = time.perf_counter() - t_wall0
        pred_items = global_items

        # tail latency estimates
        if api_logs:
            call_end = assumed_call_duration
            tail_latency_parallel = max(0.0, api_logs[-1]["finish_time_parallel_sec"] - call_end)
            tail_latency_single = max(0.0, api_logs[-1]["finish_time_single_sec"] - call_end)
        else:
            tail_latency_parallel = 0.0
            tail_latency_single = 0.0

    # Save outputs
    suffix = f"_{args.run_tag}" if getattr(args, "run_tag", "") else ""
    out_json = f"{args.out_prefix}{suffix}.json"
    out_js = f"{args.out_prefix}{suffix}.js"
    out_api_logs = f"{args.out_prefix}_api_logs{suffix}.json"
    out_logs = out_api_logs

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(pred_items, f, ensure_ascii=False, indent=2)

    # JS format for frontend quickly
    js_str = "export const callTranscript = " + json.dumps(pred_items, ensure_ascii=False, indent=2) + ";\n"
    with open(out_js, "w", encoding="utf-8") as f:
        f.write(js_str)

    with open(out_logs, "w", encoding="utf-8") as f:
        json.dump(api_logs, f, ensure_ascii=False, indent=2)

    # Evaluation
    if args.eval:
        # Make sure GT schema is {speaker,message}
        gt_items = []
        for x in gt_list:
            sp = x.get("speaker")
            msg = x.get("message")
            if sp in SPEAKERS and isinstance(msg, str):
                gt_items.append({"speaker": sp, "message": clean_text_basic(msg)})

        metrics = eval_char_aligned(
            gt_items=gt_items,
            pred_items=pred_items,
            min_err_span_chars=args.min_err_span_chars,
            context_chars=40,
            top_k_spans=20,
        )

        eval_json = f"{args.out_prefix}_eval{suffix}.json"
        with open(eval_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        # also save a compact csv summary
        eval_csv = f"{args.out_prefix}_eval{suffix}.csv"
        with open(eval_csv, "w", encoding="utf-8", newline="") as f:
            import csv
            w = csv.writer(f)
            w.writerow(["index", "acc_aligned_chars", "coverage_aligned_chars", "total_aligned_chars", "gt_chars", "pred_chars"])
            w.writerow([
                args.index,
                f"{metrics['acc_aligned_chars']:.6f}",
                f"{metrics['coverage_aligned_chars']:.6f}",
                metrics["total_aligned_chars"],
                metrics["gt_chars"],
                metrics["pred_chars"],
            ])

if __name__ == "__main__":
    main()
