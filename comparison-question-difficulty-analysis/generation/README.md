# Question Generation Pipeline

본 디렉토리는 Comparison 질문 난이도 분석을 위해 사용된
질문 생성 및 후처리 파이프라인을 포함합니다.

본 프로젝트에서는 기존 HopWeaver 프레임워크를 기반으로 질문을 생성하되,
원 논문의 기본 설정을 그대로 사용하는 것이 아니라,
실험 목적에 맞게 질문 생성 범위와 구조를 확장하여 사용하였습니다.

---

## Corpus Source

질문 생성을 위해 사용된 문서 코퍼스는
Wikimedia Foundation에서 제공하는 영어 위키백과 공식 덤프입니다.

- Source: https://dumps.wikimedia.org/enwiki/latest/
- File: `enwiki-latest-pages-articles.xml.bz2`

본 프로젝트에서는 위 파일을 기반으로 문서를 파싱하여
질문 생성에 필요한 텍스트 코퍼스를 구축하였습니다.

해당 코퍼스는 공개 라이선스(CC BY-SA)를 따르며,
데이터 사용 및 라이선스 제약을 고려하여
원본 문서 텍스트는 본 레포지토리에 포함하지 않습니다.

---

## Directory Structure

### `hopweaver_base/`
HopWeaver 논문의 공식 구현을 포함한 디렉토리입니다.

- Paper: *Cross-document synthesis of high-quality and authentic multi-hop questions*
- Authors: Zhiyu Shen et al.
- Venue: ACL 2025
- Official GitHub: https://github.com/Zh1yuShen/HopWeaver

본 프로젝트에서는 해당 구현을 **2-hop 질문 생성을 위한 기반 코드로 사용**하였으며,
이를 출발점으로 질문 생성 파이프라인을 확장하였습니다.

3-hop 질문 생성은 원본 구현을 그대로 사용하는 방식이 아니라,
HopWeaver의 전체 생성 흐름을 분석한 뒤,
별도의 확장 구현을 통해 수행됩니다.

본 디렉토리는
- 2-hop 질문 생성의 기준 구현 제공
- 원본 코드에 대한 출처 명시
- 확장 구현과의 비교 기준
역할을 합니다.

---

### `extensions/`
HopWeaver 기반 질문 생성 파이프라인을
**3-hop 구조까지 지원하도록 수정·확장한 전체 생성 코드**를 포함합니다.

본 프로젝트에서는 HopWeaver를 단순 호출하는 방식이 아니라,
질문 생성 흐름 전반을 재구성하여 다음과 같은 변경을 적용하였습니다.

- 기본 2-hop 질문 생성 구조를 3-hop 질문 생성이 가능하도록 확장
- Comparison 및 Bridge 질문 모두에서
  hop 수가 명시적으로 유지되도록 생성 로직 수정
- 중간 엔티티 연결(entity chaining)이 유지되도록
  질문 생성 단계 전반 수정
- 생성 결과가 후속 실험에 바로 사용 가능하도록
  출력 포맷 및 메타데이터 구조 재설계

즉, `extensions/` 디렉토리는
HopWeaver를 기반으로 하되,
**3-hop 질문 생성을 직접 지원하도록
전체 생성 파이프라인을 수정·확장한 핵심 구현**을 포함합니다.

---

### `postprocess/`
질문 생성 이후의 결과를
실험에 사용 가능한 **표준화된 데이터셋(JSONL)** 형태로
변환하는 후처리 단계입니다.

이 단계에서는 HopWeaver 및 확장 파이프라인에서 생성된
multi-hop 질문과 sub-question을
**질문 단위 기준으로 재구성(expand & normalize)** 합니다.

- `build_subquestions_documents.py`  
  하나의 레코드에 multi-question과 sub-question들이 함께 포함된
  원본 생성 결과를 입력으로 받아,

  - multi / sub1 / sub2 질문을 각각 독립적인 JSONL 레코드로 확장
  - `group_id` 기반으로 질문 간 관계 유지
  - question type(`multi`, `sub1`, `sub2`) 명시
  - 각 질문에 대응되는 document context(document_a / document_b) 정규화
  - 실험에서 바로 사용할 수 있는 공통 스키마로 변환

이를 통해, 하나의 multi-hop 질문 묶음은
**최종적으로 3개의 질문 레코드(multi + sub1 + sub2)** 로 확장되며,
이후 평가 단계에서
중간 추론 경로 단위 분석이 가능하도록 구성됩니다.

---

## Summary

본 질문 생성 파이프라인은 다음 흐름으로 구성됩니다.

1. Wikipedia 공식 덤프 기반 문서 코퍼스 구축
2. HopWeaver 원본 구현을 사용한 2-hop 질문 생성
3. 전체 생성 로직을 확장하여 3-hop 질문 생성 지원
4. 생성 결과를 질문 단위(JSONL)로 재구성
5. multi-question과 sub-question 간 관계를 유지한 데이터셋 구축

이를 통해, 단순 정답 예측 성능 비교를 넘어
중간 추론 단계(sub-question)의 붕괴 여부까지 분석 가능한
Comparison QA 데이터셋을 구축하였습니다.
