#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_subquestions.py
---------------------
Convert a dataset where each original record contains one multi-hop question and its two sub-questions
into a 3x expanded JSONL (one line per question).

Input assumptions (robust to common variants):
- Each input record is JSON with at least these logical fields (various key names supported):
    * group id: one of ["id", "example_id", "qid", "uid"]
    * multi-hop question: one of ["multi_hop_question", "multi_hop_q", "question"]
    * multi-hop answer: one of ["answer", "final_answer", "multi_hop_answer"]
    * sub-question 1: one of ["sub_question1", "sub_q1", "sub1"]
    * answer 1: one of ["answer1", "sub_answer1", "a1"]
    * sub-question 2: one of ["sub_question2", "sub_q2", "sub2"]
    * answer 2: one of ["answer2", "sub_answer2", "a2"]

Output (JSONL):
- One line per question (3 per original record). Each line has:
    {
        "group_id": "<original id or auto index>",
        "orig_id": "<copied original id if present>",
        "qtype": "multi" | "sub1" | "sub2",
        "id": "<group_id>::<qtype>",         # unique id per question
        "question": "<text>",
        "answer": "<gold answer text>"
    }

추가:
- 각 줄에 원본의 document_a, document_b를 같이 저장
- qtype에 따라 context도 달리 저장
  * multi → [document_a, document_b]
  * sub1  → document_a
  * sub2  → document_b

Usage:
python build_subquestions_documents.py \
    --input /home/dilab/Desktop/Jiwon/HopWeaver/output/comparison_qa_numeric_subqa.jsonl \
    --output /home/dilab/Desktop/Jiwon/HopWeaver/output/comparison_qa_numeric_subqa_expanded_v1.jsonl
"""

import argparse, json

def load_concatenated_json(path):
    s = open(path, "r", encoding="utf-8-sig").read()
    objs, in_string, esc, depth, start = [], False, False, 0, None
    i, n = 0, len(s)
    while i < n:
        ch = s[i]
        if in_string:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_string = False
        else:
            if ch.isspace():
                i += 1
                continue
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start is not None:
                    seg = s[start:i+1]
                    try:
                        objs.append(json.loads(seg))
                    except Exception:
                        pass
                    start = None
        i += 1

    if not objs:
        with open(path, "r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    objs.append(json.loads(line))
                except Exception:
                    continue
    return objs

def pick(dic, keys, default=None):
    for k in keys:
        if k in dic and dic[k] is not None:
            return dic[k]
    return default

def normalize_text(x):
    if x is None:
        return ""
    if isinstance(x, (int, float)):
        return str(x)
    return str(x).strip()

def extract_multi(rec):
    mh = rec.get("multi_hop_question")
    if isinstance(mh, dict):
        q = mh.get("multi_hop_question") or mh.get("question")
        a = mh.get("answer") or mh.get("final_answer") or mh.get("multi_hop_answer")
        if q:
            return normalize_text(q), normalize_text(a)
    q = pick(rec, ["multi_hop_question", "multi_hop_q", "question"])
    a = pick(rec, ["answer", "final_answer", "multi_hop_answer"])
    return normalize_text(q), normalize_text(a)

def extract_subs(rec):
    subs_container = rec.get("sub_questions")
    if isinstance(subs_container, dict):
        inner = subs_container.get("sub_questions")
        if isinstance(inner, list) and len(inner) >= 2:
            s1q = normalize_text(inner[0].get("question"))
            s1a = normalize_text(inner[0].get("answer"))
            s2q = normalize_text(inner[1].get("question"))
            s2a = normalize_text(inner[1].get("answer"))
            return (s1q, s1a), (s2q, s2a)
    s1q  = pick(rec, ["sub_question1", "sub_q1", "sub1"])
    s1a  = pick(rec, ["answer1", "sub_answer1", "a1"])
    s2q  = pick(rec, ["sub_question2", "sub_q2", "sub2"])
    s2a  = pick(rec, ["answer2", "sub_answer2", "a2"])
    return (normalize_text(s1q), normalize_text(s1a)), (normalize_text(s2q), normalize_text(s2a))

# def slim_doc(doc):
#     if not isinstance(doc, dict):
#         return {"title": "", "contents": ""}
#     return {
#         "title": normalize_text(doc.get("title", "")),
#         "contents": normalize_text(doc.get("contents", "")),
#     }

def slim_doc(doc):
    if not isinstance(doc, dict):
        return {"title": "", "contents": ""}

    title = doc.get("title", "")

    # 우리 포맷: contents
    contents = doc.get("contents", "")
    # 위 레코드처럼 content만 있을 때 fallback
    if not contents:
        contents = doc.get("content", "")

    return {
        "title": normalize_text(title),
        "contents": normalize_text(contents),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    data = load_concatenated_json(args.input)
    out = []

    seq = 301          # <-- id는 301부터 시작
    group_id_base = 100   # <-- group_id는 100에서 시작

    for idx, rec in enumerate(data):
        group_id = str(group_id_base + idx)   # <-- 100, 101, 102 … 자동 증가

        mh_q, mh_a = extract_multi(rec)
        (s1_q, s1_a), (s2_q, s2_a) = extract_subs(rec)

        raw_doc_a = rec.get("document_a") or rec.get("doc_a") or rec.get("documentA") or rec.get("source_doc")
        raw_doc_b = rec.get("document_b") or rec.get("doc_b") or rec.get("documentB") or rec.get("target_doc")
        doc_a = slim_doc(raw_doc_a)
        doc_b = slim_doc(raw_doc_b)

        triples = [
            ("multi", mh_q, mh_a),
            ("sub1",  s1_q, s1_a),
            ("sub2",  s2_q, s2_a),
        ]

        for qtype, qtext, ans in triples:
            if not qtext:
                continue

            item = {
                "id": seq,
                "qid": f"{group_id}::{qtype}",   # 100::multi / 100::sub1 / ...
                "group_id": group_id,
                "qtype": qtype,
                "question": normalize_text(qtext),
                "golden_answers": normalize_text(ans),
                "document_a": doc_a,
                "document_b": doc_b,
                "context": {
                    "use": ["document_a", "document_b"]
                }
            }

            out.append(item)
            seq += 1   # <-- id 증가 (301,302,303,...)
    
    with open(args.output, "w", encoding="utf-8") as f:
        for it in out:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    print(f"[build_subquestions no-dup] Read {len(data)} original records.")
    print(f"[build_subquestions no-dup] Wrote {len(out)} expanded questions.")
    if len(out) == 0:
        print("[WARN] No questions extracted; check input keys.]")

if __name__ == "__main__":
    main()