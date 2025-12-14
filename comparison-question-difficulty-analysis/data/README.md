# 데이터셋 개요

본 디렉토리는 Multi-hop Question Answering 환경에서
Comparison 질문의 난이도를 분석하기 위해 사용된 데이터셋을 포함합니다.

데이터셋은 HopWeaver 프레임워크를 기반으로 질문을 생성한 후,
기존 파이프라인을 확장하여 3-hop Comparison 및 Bridge 질문을
추가로 생성하였습니다. 생성된 질문들은 본 연구의 실험 설계에 맞게
선별, 정제 및 구조화 과정을 거쳐 최종 데이터셋으로 구성되었습니다.

데이터 사용 및 라이선스 제약으로 인해,
전체 데이터셋은 공개하지 않으며,
데이터 구조를 설명하기 위한 스키마와
소수의 익명화된 샘플만 공개합니다.

---

## 디렉토리 구조

### schema.json
각 데이터 항목의 구조를 정의한 파일로, 다음 정보를 포함합니다.

- multi-question과 sub-question 간의 관계
- 문서 컨텍스트 사용 정보
- hop 수 및 질문 유형 메타데이터

데이터셋 설계를 문서화하기 위한 파일로,
공개에 문제가 없습니다.

---

### sample.jsonl
2-hop 및 3-hop Comparison 질문의 데이터 형식을
확인할 수 있도록 구성된 소량의 샘플 데이터입니다.

실제 실험에 사용된 데이터는 포함하지 않으며,
데이터 구조를 이해하기 위한 최소한의 예시만 제공합니다.

---

### raw/ (비공개)
질문 생성 과정에서 발생한 중간 산출물을 저장하는 디렉토리입니다.

- HopWeaver 기반 2-hop 질문 생성 결과
- 확장된 3-hop 질문 생성 결과
- 생성 과정에 대한 로그 및 통계 정보

해당 디렉토리는 버전 관리에서 제외됩니다.

---

### processed/ (비공개)
정제 및 검증을 거쳐 실험에 실제로 사용된
최종 데이터셋을 포함합니다.

processed/
├─ 2hop/
│  ├─ bridge.jsonl
│  └─ comparison.jsonl
├─ 3hop/
│  ├─ bridge.jsonl
│  └─ comparison.jsonl
└─ metadata.json

- 2hop/, 3hop/ : hop 수 기준으로 분류된 최종 데이터셋
- bridge.jsonl : Bridge 유형의 multi-hop 질문
- comparison.jsonl : Comparison 유형의 multi-hop 질문
- metadata.json : 질문 수 및 분포에 대한 통계 정보

해당 디렉토리는 버전 관리에서 제외됩니다.

---

## 참고 사항

본 데이터셋은 다음 조건을 기준으로
통제된 비교 실험이 가능하도록 설계되었습니다.

- hop 수 (2-hop / 3-hop)
- 질문 유형 (Comparison / Bridge)
- multi-question과 sub-question 간 추론 경로

데이터 구조 설명에 필요한 최소한의 정보만 공개하며,
전체 데이터셋은 비공개로 유지합니다.
