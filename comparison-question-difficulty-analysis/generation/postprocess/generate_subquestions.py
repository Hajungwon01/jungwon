# -*- coding: utf-8 -*-
"""
Comparison형 multi-hop 질문을 2개의 단일-홉(sub-questions)으로 분해.
주의: 각 record의 status가 PASS가 아니면 refined_* 필드를 사용.
- 모델: gpt-4o (OpenAI Responses API; OPENAI_API_KEY 필요)
- 입력: JSONL / JSON 배열 / concatenated JSON 모두 지원
- 출력: 각 레코드에 sub_questions(sub_questions: [{question, answer, entity, evidence_span, source}]) 추가
"""

import os, json, argparse
from typing import Dict, Any, List
from openai import OpenAI  # pip install --upgrade openai

# 0) 로더: JSONL / 배열 / concatenated JSON 모두 허용
def load_records(path: str) -> List[Dict[str, Any]]:
    s = open(path, "r", encoding="utf-8-sig").read().strip()
    if not s:
        return []
    if s.startswith("[") and s.endswith("]"):
        return json.loads(s)
    # try JSONL
    recs, ok = [], True
    for ln, line in enumerate(s.splitlines(), 1):
        t = line.strip()
        if not t:
            continue
        try:
            recs.append(json.loads(t))
        except Exception:
            ok = False
            break
    if ok:
        return recs
    # concatenated JSON
    recs = []
    in_str, esc, depth, start = False, False, 0, None
    for i, ch in enumerate(s):
        if in_str:
            if esc: esc = False
            elif ch == "\\": esc = True
            elif ch == '"': in_str = False
        else:
            if ch == '"': in_str = True
            elif ch == "{":
                if depth == 0: start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start is not None:
                    recs.append(json.loads(s[start:i+1])); start = None
    if depth != 0:
        raise ValueError("JSON brace mismatch.")
    return recs

# 1) id 생성 (document_a/document_b 또는 source_doc/target_doc 케이스 호환)
def make_pair_id(rec: Dict[str, Any]) -> str:
    try:
        a = rec.get("document_a", {}).get("document_id") or rec.get("document_a", {}).get("id")
        b = rec.get("document_b", {}).get("id") or rec.get("document_b", {}).get("document_id")
        if a and b: return f"{a}_{b}"
    except Exception:
        pass
    try:
        a = rec["source_doc"]["id"]; b = rec["target_doc"]["id"]; return f"{a}_{b}"
    except Exception:
        pass
    return str(abs(hash(json.dumps(rec, ensure_ascii=False))) % (10**12))

# 2) PASS vs refined 선택 로직
def resolve_fields(rec: Dict[str, Any]) -> Dict[str, Any]:
    """
    반환:
      {
        'question': 사용될 질문 문자열,
        'fact_a':   사용될 fact_entity_a 문자열,
        'fact_b':   사용될 fact_entity_b 문자열,
        'entity_a': (가능하면) 이름,
        'entity_b': (가능하면) 이름,
        'attribute': (가능하면) 속성명,
        'answer_final': (있으면) 최종 정답 힌트
      }
    """
    pol = rec.get("polished_result") or rec.get("polish_result") or {}
    status = (pol.get("status") or "PASS").upper()

    base_q = rec.get("multi_hop_question")
    if isinstance(base_q, dict):
        base_q = base_q.get("multi_hop_question")
    ref_q = pol.get("refined_question")

    base_fa = rec.get("fact_entity_a")
    base_fb = rec.get("fact_entity_b")
    ref_fa  = pol.get("refined_fact_a")
    ref_fb  = pol.get("refined_fact_b")

    if status == "PASS":
        question = base_q or ref_q or ""
        fact_a   = base_fa or ref_fa or ""
        fact_b   = base_fb or ref_fb or ""
        answer_final = rec.get("answer")
    else:
        question = ref_q or base_q or ""
        fact_a   = ref_fa or base_fa or ""
        fact_b   = ref_fb or base_fb or ""
        answer_final = pol.get("refined_answer") or rec.get("answer")

    return {
        "question": question or "",
        "fact_a":   fact_a or "",
        "fact_b":   fact_b or "",
        "entity_a": rec.get("entity_a"),
        "entity_b": rec.get("entity_b"),
        "attribute": rec.get("attribute_compared"),
        "answer_final": answer_final,
    }

# -------------------------------
# 3) 비교형 하위질문 프롬프트 (EN) 주입
# -------------------------------

def _guess_attribute_type(attr: str) -> str:
    """간단 휴리스틱: attribute 타입 추정"""
    if not attr:
        return "set/list-derived"
    al = attr.strip().lower()
    if any(k in al for k in ["year", "date", "age", "population", "length", "height", "weight", "duration", "runtime", "count", "number", "no."]):
        return "numeric"
    if any(k in al for k in ["rank", "order", "position"]):
        return "ordinal"
    if any(k in al for k in ["role", "theme", "genre", "type", "category"]):
        return "set/list-derived"
    if any(k in al for k in ["is ", "has ", "whether", "boolean"]):
        return "boolean"
    return "set/list-derived"

def _rubric_preset(attr: str) -> str:
    """Role/Theme 등 비수치 속성용 루브릭 프리셋"""
    if not attr:
        return ""
    al = attr.strip().lower()
    if "role" in al:
        return (
            "Rubric for non-numeric (if applicable): "
            "Extract distinct role/occupation labels explicitly stated in each document; "
            "compare by the count of unique roles (|distinct roles|)."
        )
    if "theme" in al:
        return (
            "Rubric for non-numeric (if applicable): "
            "Count explicit dark motif tokens (e.g., revenge, murder/kill, immurement, deceit, psychological horror) "
            "mentioned in each document; compare by motif count (ties can be noted but not resolved here)."
        )
    return ""  # 기타 속성은 일반 템플릿의 비수치 안내만 사용

def build_instruction1(fields: Dict[str, Any]) -> str:
    attr = fields.get("attribute") or ""
    attr_type = _guess_attribute_type(attr)
    # numeric 전용이므로 rubric 프리셋은 사용하지 않음
    rubric_line = ""  # _rubric_preset(attr)  # <- 비수치용 루브릭은 비활성화

    core = f"""---Goal---
Given two documents about two entities (Entity A, Entity B) and a target comparison question, generate two sequential sub-questions that each elicit a **numeric** value for the SAME comparison attribute. The two numeric answers must be sufficient to deterministically answer the target comparison (e.g., greater/less, earlier/later).

---Key Update: Numeric-only answers---
• Design each sub-question so that the answer is a **number only** (digits), derivable from a single document:
  - Year/date → return **YYYY only** (4 digits).
  - Counts → return **an integer** (e.g., number of items/awards/roles explicitly listed).
  - Measurements (length, duration, runtime, population, etc.) → return **a number only**; omit units unless both documents share the **same unit**. If units differ, specify a normalization rule and return the normalized numeric value (no unit).
  - Ranges (e.g., 1871–1872): if the target question implies "first/earliest" or "latest", return **min/max** of the range accordingly; otherwise return the **single value explicitly requested** (prefer earliest if ambiguous).
• Provide an `evidence_span` verbatim from the relevant document that justifies the numeric answer.

---Reuse multi-hop question phrasing (but numeric)---
• You MAY reuse exact phrases from the target question (entities, attribute terms) to keep sub-questions specific.
• You MUST NOT introduce qualifiers/scopes that are NOT in the documents or the target question (no new time windows, purposes, or domains).

---Hard Constraints---
• Use ONLY (i) wording from the target question and/or (ii) facts from the corresponding single document.
• Do NOT add extra domain/time/purpose qualifiers beyond what appears in the target question.
• If an attribute is provided, use ONLY the attribute phrase as given: "{attr}".
• If no attribute is provided, infer the **minimal neutral numeric attribute phrase** from the comparative wording (e.g., "earlier publication" → year; "more awards" → count of awards).
• Each sub-question must be answerable using a single document (A for Q1, B for Q2).

---Answer Style (strict)---
• Return **numbers only** in answers:
  - Year: `YYYY`
  - Integers/floats: digits only (e.g., `12`, `95.3`)
  - No sentences, no words like "years", "minutes", "people", etc.
• Evidence: include a verbatim `evidence_span` with the numeric source text.
• If normalization is necessary (different units/scales), specify it under "Normalization" and still return **a single numeric** result (normalized) for each entity.

---Decomposition Rules---
1) Choose and validate ONE numeric comparison attribute (use "{attr}" if provided; otherwise infer a numeric attribute minimally from the multi question).
2) Attribute Type (forced numeric path): treat the attribute as **numeric** even if phrased loosely; define how to map text → number (e.g., count tokens, extract year).
3) Define a **numeric** Comparison Rule (e.g., greater/less; earlier/later; higher/lower). Ensure comparability (same definition/scope/time-frame/unit). State normalization if needed.
4) Generate two sub-questions:
   - Q1 (Document A only): ask for Entity A’s numeric value for the SAME attribute. **Do NOT mention Entity B.**
   - Q2 (Document B only): ask for Entity B’s numeric value for the SAME attribute. **Do NOT reference A’s answer.**
5) Do NOT perform the final comparison; add one line: "How to compute final answer" (e.g., compare Answer 1 vs Answer 2 by the chosen rule).

---Numeric Question Templates (pick the simplest; DO NOT add extra qualifiers)---
• Year/date (earlier/later): "In what year was {{Entity}} {('' if not attr else attr)}?"
  - If the target question implies "first/earliest/latest", use that wording: "In what year was {{Entity}} first/earliest/latest {('' if not attr else attr)}?"
• Counts: "How many {attr if attr else 'X'} did {{Entity}} have/receive/publish/win/list?"
  - Examples: "How many awards did {{Entity}} receive?", "How many roles are listed for {{Entity}}?"
• Numeric property: "What is the {attr if attr else 'numeric value'} of {{Entity}}?"
  - Examples: "What is the runtime of {{Entity}}?", "What is the population of {{Entity}}?"

---Mapping text → number (guidance)---
• Year: extract 4-digit year from evidence (choose min/max if earliest/latest is implied).
• Counts: count explicit, distinct items in the evidence (e.g., awards listed, roles enumerated).
• Measurements: extract number; if units differ, specify normalization and return normalized numeric only.

---Output constraints---
Return JSON only (schema provided separately). Keep answers **numeric-only** as specified above.
Normalization (if any): clearly state conversions (e.g., miles→km, hours→minutes) but **answers remain numbers only**.

---Presets (numeric view)---
• Earlier/later publication/serialization → year (YYYY).
• More awards/publications/roles listed → count (integer).
• Longer/shorter runtime/length/duration → normalized numeric magnitude.
"""
    if rubric_line:
        core += "\n" + rubric_line + "\n"
    return core

def build_instruction(fields: Dict[str, Any]) -> str:
    attr = fields.get("attribute") or ""
    attr_type = _guess_attribute_type(attr)
    rubric_line = _rubric_preset(attr)  # Role/Theme 등 비수치 프리셋 가이드 유지

    core = f"""---Goal---
Given two documents about two entities (Entity A, Entity B) and a target comparison question, generate two sequential sub-questions that each elicit a **concise, non-numeric** value for the SAME comparison attribute. Design each sub-question to extract a **decisive signal on the comparison axis** (e.g., change status, role/association strength token, scope/breadth), so the two answers are sufficient to deterministically answer the comparison.

---Key Update: Target the comparison axis directly---
• Do NOT ask only for a current/static label that fails to decide the comparison (e.g., just “Political Party” when the multi-hop asks “who changed parties”).
• Instead, craft each sub-question to retrieve the **signal that resolves the comparison**:
  - Change/Transition: ask whether a change occurred (Yes/No) and the from→to token if available.
  - Association Strength/Role Hierarchy: ask for the official title/epithet that indicates higher linkage (e.g., “chief prosecutor”, “father of …”).
  - Breadth/Scope (set/list-derived): ask for the distinct token set (roles/themes/recognitions) if diversity/variety is compared.
• Keep the answer short (single token or short token list); put explanatory context in `evidence_span`.

---MUST reuse multi-hop question phrasing---
• Reuse the exact attribute/domain descriptors from the target question and record-level attribute_compared (if present) to keep sub-questions specific.
• Do NOT introduce any qualifier/scope not present in the target question (no new time windows, purposes, or domains).

---Hard Constraints---
• Use ONLY (i) wording from the target question and/or (ii) facts from the corresponding single document.
• Do NOT add domain/time/purpose qualifiers beyond what appears in the target question.
• If an attribute is provided, use ONLY the attribute phrase exactly as given: "{attr}".
• If no attribute is provided, infer the **minimal neutral attribute phrase** directly from the comparison wording and reuse that phrasing.
• Each sub-question must be answerable using a single document (A for Q1, B for Q2).

---Answer Style (concise, non-numeric)---
• Answers must be SHORT and FACTUAL, supported only by the corresponding document:
  - Return a **single token** or a **comma-separated token list** (roles, themes, recognitions, sports), not a sentence.
  - Tokens must appear **verbatim** in the evidence whenever possible; light normalization (plural→singular) is okay.
  - For change/transition, `Yes`/`No` is allowed; if the document states from→to tokens, you MAY answer with a concise change token like "opposition → PDG".
• Provide an `evidence_span` verbatim from the doc that justifies the answer.

---Decomposition Rules---
1) Choose ONE comparison attribute (use "{attr}" if provided; else infer minimally from the question and reuse that phrase).
2) Attribute Type: one of {{categorical, set/list-derived, ordinal, boolean}}. (guessed: {attr_type})
3) Define a Comparison Rule appropriate to the type and ensure comparability:
   - Change detection → entity with `Yes` wins.
   - Role/association strength → compare title/epithet salience (e.g., “chief prosecutor” outranks “lawyer”).
   - Diversity/breadth → compare |distinct tokens|.
4) Generate two sub-questions:
   - Q1 (Document A only): elicit A’s **decisive signal** for the SAME attribute. **Reuse the question/attribute phrasing; do NOT mention B.**
   - Q2 (Document B only): elicit B’s **decisive signal** for the SAME attribute. **Reuse the same phrasing; do NOT reference A.**
5) Do NOT perform the final comparison; add one line: "How to compute final answer" that states the chosen rule (e.g., `Yes` > `No`, higher title > lower title, larger |set| wins).

---Templates (pick the one that targets the comparison axis; DO NOT add extra qualifiers)---
• Change/Transition: "Did {{Entity}} change {('' if not attr else attr)}?"  (answer: Yes/No; if available, concise change token is allowed)
• Association Strength / Role Hierarchy: "What {('' if not attr else attr)} title links {{Entity}} to {{domain in question}}?"  (answer: short title token)
• Breadth/Scope (diversity): "What {('' if not attr else attr)} did {{Entity}} have?" (answer: comma-separated distinct tokens)
• Fallback (if none of the above apply): "What is {{Entity}}’s {('attribute' if not attr else attr)}?"

---Mapping text → decisive signal (guidance)---
• Change detection: extract an explicit Yes/No from the doc (or a concise from→to token if available).
• Role/association strength: choose the most specific official title/epithet present in the doc.
• Breadth/Scope: extract distinct tokens explicitly stated; return as a short list.

---Tie-break & ambiguity guard---
• If both entities would yield **identical surface tokens** for a naive sub-question, refine each sub-question to retrieve the **decisive signal** (e.g., change status; from→to; higher-rank title; set cardinality) so the comparison becomes resolvable.

---Output constraints---
Return JSON only (schema provided separately). Keep answers **short, token-level**. Include `evidence_span` verbatim for each.

---Presets (non-numeric view)---
• Party change → compare Yes/No (change occurred).
• Guantánamo linkage → compare official role title salience.
• Tartan Noir linkage → compare epithet/title strength (e.g., “father of …” outranks generic affiliation).
• USCT vs a specific regiment → compare scope/breadth (organization vs single regiment; |distinct units| if stated).
• Role/Recognition/Theme diversity → compare |distinct tokens|.
"""
    if rubric_line:
        core += "\n" + rubric_line + "\n"
    return core

# 시스템 메시지(엄격 JSON 응답 요구)
SYSTEM_MSG = (
    "You are a precise QA decomposition assistant. "
    "Always respond ONLY with a valid JSON object that matches the requested schema. "
    "Do NOT include any extra text, code fences, or explanations—JSON only."
)

def build_user_payload(rid: str, fields: Dict[str, Any]) -> str:
    # 비교형 프롬프트를 payload의 'instruction'에 실어 모델이 반드시 읽게 함
    comparison_instr = build_instruction(fields)

    payload = {
        "instruction": comparison_instr,
        "record_id": rid,
        "question": fields.get("question"),
        "entity_a": fields.get("entity_a"),
        "entity_b": fields.get("entity_b"),
        "attribute": fields.get("attribute"),
        "fact_entity_a": fields.get("fact_a"),
        "fact_entity_b": fields.get("fact_b"),
        # 힌트는 강제 정답 아님(모델 안정화용)
        "hint_final_answer": fields.get("answer_final"),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)

STRUCTURED_SCHEMA = {
    "name": "SubQOutput",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "id": {"type": "string"},
            "sub_questions": {
                "type": "array",
                "minItems": 2, "maxItems": 2,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "question": {"type": "string"},
                        "answer":   {"type": "string"},
                        "entity":   {"type": "string", "enum": ["entity_a", "entity_b"]},
                        "evidence_span": {"type": "string"}
                    },
                    "required": ["question", "answer", "entity", "evidence_span"]
                }
            }
        },
        "required": ["id", "sub_questions"]
    },
}

def _extract_json(text: str):
    """출력이 잡문+JSON 섞여올 때를 대비해 첫 번째 JSON 오브젝트만 안전 추출"""
    if not isinstance(text, str):
        return {}
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start is not None:
                segment = text[start:i+1]
                try:
                    return json.loads(segment)
                except Exception:
                    break
    try:
        return json.loads(text)
    except Exception:
        return {}

def _coerce_to_schema(obj, rid):
    """
    모델 출력이 약간 달라도 우리가 원하는 스키마로 보정
    target:
      {"id": rid, "sub_questions": [
          {"question": str, "answer": str, "entity": "entity_a|entity_b", "evidence_span": str},
          {"question": str, "answer": str, "entity": "entity_a|entity_b", "evidence_span": str}
      ]}
    """
    if not isinstance(obj, dict):
        obj = {}
    sqs = obj.get("sub_questions") or obj.get("subqs") or obj.get("subs") or []
    if isinstance(sqs, dict):
        sqs = [sqs]

    norm = []
    for i, it in enumerate(sqs[:2]):
        it = it or {}
        q  = it.get("question") or it.get("q") or ""
        a  = it.get("answer")   or it.get("ans") or ""
        e  = it.get("entity")   or ("entity_a" if i == 0 else "entity_b")
        ev = it.get("evidence_span") or it.get("evidence") or it.get("span") or ""
        if e not in ("entity_a", "entity_b"):
            e = "entity_a" if i == 0 else "entity_b"
        norm.append({"question": q, "answer": a, "entity": e, "evidence_span": ev})

    while len(norm) < 2:
        e = "entity_a" if len(norm) == 0 else "entity_b"
        norm.append({"question": "", "answer": "", "entity": e, "evidence_span": ""})

    return {"id": rid, "sub_questions": norm[:2]}

def call_gpt_4o(client: OpenAI, rid: str, user_payload: str) -> dict:
    """
    Chat Completions + response_format={"type":"json_object"} 경로
    """
    messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {
            "role": "user",
            "content": (
                "Return a JSON object with this schema:\n"
                "{\n"
                '  "id": string,\n'
                '  "sub_questions": [\n'
                '    {"question": string, "answer": string, "entity": "entity_a", "evidence_span": string},\n'
                '    {"question": string, "answer": string, "entity": "entity_b", "evidence_span": string}\n'
                "  ]\n"
                "}\n"
                "The response MUST be valid JSON.\n\n"
                "INPUT (JSON):\n" + user_payload
            ),
        },
    ]

    try:
        chat = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0,
            response_format={"type": "json_object"},
        )
        text = chat.choices[0].message.content
    except Exception:
        chat = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0,
        )
        text = chat.choices[0].message.content

    out = _extract_json(text)
    return _coerce_to_schema(out, rid)

# 4) 병합 저장 (EACL 서브질문 블록 형태)
def merge_subqs(rec: Dict[str, Any], subq_obj: Dict[str, Any]) -> Dict[str, Any]:
    sqs_out = []
    for it in subq_obj["sub_questions"]:
        sqs_out.append({
            "question": it["question"],
            "answer":   it["answer"],
            "entity":   it["entity"],
            "source":   "Document A" if it["entity"]=="entity_a" else "Document B",
            "evidence_span": it["evidence_span"],
        })
    rec_out = dict(rec)
    rec_out["sub_questions"] = {"sub_questions": sqs_out}
    return rec_out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--api_key", default=os.getenv("OPENAI_API_KEY"))
    ap.add_argument("--max_records", type=int, default=None)
    args = ap.parse_args()

    if not args.api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    client = OpenAI(api_key=args.api_key)
    records = load_records(args.input)
    if args.max_records is not None:
        records = records[:args.max_records]

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fout:
        for rec in records:
            rid = make_pair_id(rec)
            fields = resolve_fields(rec)
            payload = build_user_payload(rid, fields)
            subq_obj = call_gpt_4o(client, rid, payload)

            # 안전장치: entity_a 먼저 정렬
            subqs = subq_obj.get("sub_questions", [])
            subqs_sorted = sorted(subqs, key=lambda x: 0 if x["entity"]=="entity_a" else 1)
            subq_obj["sub_questions"] = subqs_sorted

            merged = merge_subqs(rec, subq_obj)
            fout.write(json.dumps(merged, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
