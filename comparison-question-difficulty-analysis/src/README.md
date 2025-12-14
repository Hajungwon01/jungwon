# Experimental Pipeline (src)

본 디렉토리는 Comparison 질문 난이도 분석을 위한
모델 실행, 평가, 분석 전 과정을 담당하는 실험 파이프라인입니다.

본 프로젝트의 목표는 단순 정답 정확도(EM/F1)를 넘어,
멀티홉 추론 과정이 실제로 어떻게 붕괴되는지를 구조적으로 분석하는 것입니다.

이를 위해 데이터 로딩부터 모델 실행, 추론 경로 평가,
난이도 요인 분석까지의 전 과정을 모듈화하여 구성하였습니다.

---

## Overall Pipeline

dataset
  -> models
  -> evaluation
  -> analysis

---

## Directory Structure

src/
- dataset/
- models/
- evaluation/
- analysis/
- utils/

---

## 1. dataset/
실험 단위 기준 데이터 로딩

- JSONL 형식의 multi-hop QA 데이터셋 로드
- group_id 기준으로 multi-question과 sub-question을 하나의 추론 단위로 유지
- hop 수(2-hop / 3-hop), 질문 유형(comparison / bridge) 기준 필터링 지원

주요 파일
- loader.py: 실험 및 분석 단계에서 사용되는 표준 데이터 로더

---

## 2. models/
다양한 LLM을 동일 조건에서 실행하기 위한 공통 인터페이스

모든 모델은 다음 3단계를 동일하게 수행합니다.
format_prompt -> generate -> parse_response

포함 모델
- gpt.py        : OpenAI GPT 계열
- openhermes.py : OpenHermes (LLaMA 계열)
- qwen.py       : Qwen Instruct 모델
- mistral.py    : Mistral Instruct 모델
- deepseek.py   : DeepSeek Reasoning 모델

---

## 3. evaluation/
정답률 + 추론 경로 기반 평가

- metrics.py
  SQuAD-style EM / token-level F1 계산

- path_labeler.py
  multi / sub-question 단위로 정오답을 C/W로 라벨링하여
  C/C/C, C/W/W 등의 추론 경로 정의

- scorer.py
  hop / 질문 유형 / 모델 / 역할 기준 성능 집계

---

## 4. analysis/
Comparison 질문 난이도의 원인 분석

- path_distribution.py
  C/W 추론 경로 분포 분석

- mask_effect.py
  문서 내 숫자·날짜 정보를 마스킹하여
  증거 제거가 추론 경로에 미치는 영향 분석

- framing_robustness.py
  질문 표현 변화에 따른 성능 및 경로 안정성 분석

---

## 5. utils/
공통 유틸리티

- JSON / JSONL 입출력
- 실험 결과 저장 및 로깅

---

## Summary

본 실험 파이프라인은 단순 정답률 비교를 넘어
멀티홉 추론 과정의 성공과 실패를 구조적으로 분석하기 위해 설계되었습니다.

이를 통해 Comparison 질문이 왜 더 어려운지를
데이터, 모델, 증거 관점에서 분해하여 설명합니다.
