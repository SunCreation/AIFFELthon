# Biomni Benchmark Experiment Log

## Experiment #1: Baseline (biomni_a1)
- **Date**: 2026-02-27
- **Agent**: `biomni_a1` (원본 A1 에이전트, 수정 없음)
- **Config**: 8 workers, 433 tasks, gpt-oss-20b, timeout=600s
- **Result dir**: `logs/experiments/20260227_133815_biomni_biomni_a1/`
- **Log file**: (없음, 당시 미보존)

### Results
| Task Type | Correct | Total | Accuracy |
|---|---|---|---|
| gwas_causal_gene_opentargets | 39 | 50 | 78.0% |
| gwas_causal_gene_pharmaprojects | 33 | 50 | 66.0% |
| gwas_causal_gene_gwas_catalog | 25 | 50 | 50.0% |
| screen_gene_retrieval | 22 | 50 | 44.0% |
| lab_bench_seqqa | 18 | 50 | 36.0% |
| lab_bench_dbqa | 17 | 50 | 34.0% |
| gwas_variant_prioritization | 6 | 43 | 14.0% |
| crispr_delivery | 0 | 10 | 0.0% |
| patient_gene_detection | 0 | 50 | 0.0% |
| rare_disease_diagnosis | 0 | 30 | 0.0% |
| **OVERALL** | **160** | **433** | **37.0%** |

### Observations
- 평균 실행 시간: 85.2s/task
- 30초 초과 비율: 86.8% (대부분의 태스크에서 도구 호출 발생 추정)
- tool retriever 기반 자동 도구 선택 사용
- know_how_docs 2개 포함 (프롬프트 ~50K chars)

---

## Experiment #2: Multi-Agent v1 (biomni_a1_multi)
- **Date**: 2026-03-03
- **Agent**: `biomni_a1_multi` (멀티에이전트 v1)
- **Config**: 8 workers, 433 tasks, gpt-oss-20b, timeout=600s
- **Result dir**: `logs/experiments/20260303_112816_biomni_biomni_a1_multi/`
- **Log file**: `/tmp/benchmark_multi_run.log`

### Changes from Baseline (동시에 변경 — 원인 분리 불가)
1. **시스템 프롬프트 주입** — 태스크별 curated instruction 추가
2. **도구 큐레이션** — task_name별 관련 도구만 필터링 (tool retriever 비활성화)
3. **postprocess_answer()** — 태스크별 응답 후처리 (정규식 기반 추출)
4. **know_how_docs 제거** — 프롬프트 크기 50K → 28K chars
5. **MemorySaver 리셋** — 매 태스크마다 대화 히스토리 초기화
6. **classify_prompt()** — 프롬프트 패턴으로 task_name 자동 분류

### Results
| Task Type | Correct | Total | Accuracy | vs Baseline |
|---|---|---|---|---|
| gwas_causal_gene_opentargets | 26 | 50 | 52.0% | **-26.0pp** |
| gwas_causal_gene_pharmaprojects | 25 | 50 | 50.0% | **-16.0pp** |
| gwas_causal_gene_gwas_catalog | 13 | 49 | 26.5% | **-23.5pp** |
| screen_gene_retrieval | 18 | 50 | 36.0% | -8.0pp |
| lab_bench_seqqa | 14 | 50 | 28.0% | -8.0pp |
| lab_bench_dbqa | 16 | 50 | 32.0% | -2.0pp |
| gwas_variant_prioritization | 1 | 43 | 2.3% | **-11.6pp** |
| crispr_delivery | 3 | 10 | 30.0% | **+30.0pp** |
| patient_gene_detection | 5 | 50 | 10.0% | **+10.0pp** |
| rare_disease_diagnosis | 0 | 30 | 0.0% | +0.0pp |
| **OVERALL** | **121** | **432** | **28.0%** | **-9.0pp** |

### Tool Calling Statistics
| Task Type | Tool Call Rate | Baseline avg time | Multi avg time |
|---|---|---|---|
| gwas_causal_gene_opentargets | 50.0% | 63.6s | 43.1s |
| gwas_causal_gene_pharmaprojects | 56.0% | 75.1s | 85.6s |
| gwas_causal_gene_gwas_catalog | 44.9% | 84.1s | 29.3s |
| screen_gene_retrieval | 38.0% | 83.0s | 39.1s |
| lab_bench_seqqa | 0.0% | 51.0s | 45.2s |
| lab_bench_dbqa | 0.0% | 40.3s | 35.1s |
| gwas_variant_prioritization | 46.5% | 189.6s | 43.8s |
| crispr_delivery | 0.0% | 52.8s | 27.6s |
| patient_gene_detection | 18.4% | 128.8s | 63.7s |
| rare_disease_diagnosis | 56.7% | 63.8s | 76.9s |
| **OVERALL** | **32.5%** | **85.2s** | **49.7s** |

### Failure Pattern Analysis
- **Normal wrong answer**: 158/246 (64.2%) — 도구 호출했으나 잘못된 결과 or 도구 미호출
- **Placeholder in answer**: 65/246 (26.4%) — `{top_gene}`, `{best_variant}` 등 미치환 변수
- **Backtick/tag leak**: 23/246 (9.3%) — LLM이 `<solution>` 태그를 올바르게 사용하지 못함

### Key Observations

#### 1. 도구 호출이 오히려 감소함
- Baseline은 평균 85.2s (86.8%가 30s 초과) → 대부분 도구를 호출했을 것
- Multi-agent는 평균 49.7s (35.3%가 30s 초과) → **도구 호출 빈도가 현저히 낮음**
- 특히 `gwas_causal_gene_gwas_catalog` (84.1s → 29.3s)은 도구 없이 바로 답변하는 비율이 높아짐

#### 2. 프롬프트 변경이 LLM 동작을 방해
- `"V2G"`, `"{top_gene}"` 같은 코드 변수명이 답변으로 나오는 것은 LLM이 curated instruction의 예시를 답변으로 혼동한 것
- `"` tags."` 류 답변은 태그 사용법 설명을 답변으로 출력한 것
- baseline에서는 이런 패턴이 없었음

#### 3. 긍정적 변화도 존재
- `crispr_delivery`: 0% → 30% (curated instruction이 효과적)
- `patient_gene_detection`: 0% → 10% (도구 호출이 새로 활성화)

### Root Cause Hypotheses (검증 필요)
1. **know_how_docs 제거가 가장 큰 원인일 가능성** — baseline의 know_how_docs가 도구 사용법을 상세히 설명하여 도구 호출률이 높았을 수 있음
2. **tool retriever 비활성화** — curated tools만 주입하면서 LLM이 이전에 자동으로 찾던 도구를 못 쓰게 됨
3. **curated instruction이 방해** — 태스크별 지시문이 LLM의 기존 행동을 오염시킴

### Next Steps
심층 1:1 비교 분석 + Oracle 전략 의뢰 기반으로 Exp#3 설계 (아래 참조)

---

## Baseline 재실행 (tools= 필드 포함)
- **Date**: 2026-03-03
- **Agent**: `biomni_a1` (원본, Exp#1과 동일)
- **Purpose**: Exp#2와 동일 조건 로그 형식으로 1:1 비교 가능하게 재실행
- **Config**: 8 workers, 433 tasks, gpt-oss-20b, timeout=600s
- **Result dir**: `logs/experiments/20260303_140948_biomni_biomni_a1/`
- **Log file**: `/tmp/benchmark_baseline_rerun.log`

### Results
| Task Type | Correct | Total | Accuracy | vs Exp#1 |
|---|---|---|---|---|
| gwas_causal_gene_opentargets | 37 | 50 | 74.0% | -4.0pp |
| gwas_causal_gene_pharmaprojects | 34 | 50 | 68.0% | +2.0pp |
| gwas_causal_gene_gwas_catalog | 29 | 50 | 58.0% | +8.0pp |
| screen_gene_retrieval | 22 | 50 | 44.0% | 0.0pp |
| lab_bench_seqqa | 14 | 50 | 28.0% | -8.0pp |
| lab_bench_dbqa | 15 | 50 | 30.0% | -4.0pp |
| gwas_variant_prioritization | 3 | 43 | 7.0% | -7.0pp |
| crispr_delivery | 3 | 10 | 30.0% | +30.0pp |
| patient_gene_detection | 1 | 50 | 2.0% | +2.0pp |
| rare_disease_diagnosis | 0 | 30 | 0.0% | 0.0pp |
| **OVERALL** | **158** | **433** | **36.5%** | **-0.5pp** |

### Tool Usage (Baseline)
- Overall tool call rate: 20.4% (88/432)
- Top tools: query_gwas_catalog(41), query_ensembl(28), query_monarch(20), query_pubmed(11)

### Notes
- Exp#1 (37.0%) 대비 -0.5pp 차이는 LLM 비결정론으로 예상 범위 내
- 이 재실행 결과가 Exp#2와 동일 로그 형식 → 정확한 1:1 비교 가능

---

## 심층 비교 분석: Baseline 재실행 vs Exp#2 Multi-v1

### 1:1 Task 비교 (433 tasks)
| 결과 | 건수 | 비율 |
|---|---|---|
| Both correct | 93 | 21.5% |
| Both wrong | 247 | 57.0% |
| BL correct → ML wrong (DEGRADED) | 65 | 15.0% |
| BL wrong → ML correct (IMPROVED) | 28 | 6.5% |
| Net loss: 37 tasks → 9pp 하락 설명 | | |

### ML Wrong Answer 분류 (312건 전체)
| 유형 | 건수 | 비율 |
|---|---|---|
| Tag leak ("tags.", "` tags." 등) | 71 | 22.8% |
| Placeholder/Ensembl ID | 67 | 21.5% |
| V2G text leak | 6 | 1.9% |
| Normal wrong | 168 | 53.8% |

### Degradation 패턴 (65건 상세)
| 패턴 | 건수 | 의미 |
|---|---|---|
| no_tool→no_tool | 40 | 프롬프트 텍스트 자체가 해로움 |
| no_tool→used_tool | 16 | 불필요한 도구 호출이 오답 유발 |
| used_tool→no_tool | 6 | 도구를 써야 했는데 안 씀 |
| tool→diff_tool | 3 | 다른 도구 사용 |

### 핵심 발견: TASK_PROMPTS 텍스트 주입이 20B 모델에 유해
- 65 degraded 중 40건이 **no_tool→no_tool** — 도구와 무관, 프롬프트 텍스트 자체가 원인
- 40건 중: tag leak 17, V2G leak 2, placeholder 3, 순수 wrong 18
- 20B 모델은 prompt-sensitive — 기존 28K 시스템 프롬프트에 추가 텍스트 주입이 성능 저하 유발
- **결론: TASK_TOOLS 필터링은 유지, TASK_PROMPTS 주입만 제거가 최적 전략**

---

## Experiment #3: Tool-Only Injection (TASK_PROMPTS 제거)
- **Date**: 2026-03-03
- **Agent**: `biomni_a1_multi` (Exp#3 수정)
- **Config**: 8 workers, 433 tasks, gpt-oss-20b, timeout=600s
- **Log file**: `/tmp/benchmark_exp3_run2.log`

### Changes from Exp#2 (단일 변수 변경)
1. `TASK_PROMPTS = {}` — 모든 태스크별 프롬프트 주입 비활성화
2. `_inject_curated_system_prompt()`에서 프롬프트 prepend 코드 주석 처리
3. **TASK_TOOLS 필터링은 그대로 유지** — 도구 가시성 제한만 적용
4. **postprocess_answer(), classify_prompt() 등 나머지 로직 유지**

### Hypothesis
- TASK_PROMPTS 제거로 65 degraded 중 40건(no_tool→no_tool) 회복 기대
- Tag leak 71건, V2G leak 6건 대부분 해소 기대
- 예상 정확도: ~38-42% (baseline 수준 회복 + TASK_TOOLS 효과)

### Results
| Task Type | Correct | Total | Accuracy | vs Baseline |
|---|---|---|---|---|
| gwas_causal_gene_opentargets | 35 | 50 | 70.0% | -4.0pp |
| gwas_causal_gene_pharmaprojects | 34 | 50 | 68.0% | 0.0pp |
| gwas_causal_gene_gwas_catalog | 28 | 50 | 56.0% | -2.0pp |
| screen_gene_retrieval | 21 | 50 | 42.0% | -2.0pp |
| lab_bench_seqqa | 15 | 50 | 30.0% | +2.0pp |
| lab_bench_dbqa | 15 | 50 | 30.0% | 0.0pp |
| gwas_variant_prioritization | 3 | 43 | 7.0% | 0.0pp |
| crispr_delivery | 1 | 10 | 10.0% | -20.0pp |
| patient_gene_detection | 1 | 50 | 2.0% | 0.0pp |
| rare_disease_diagnosis | 0 | 30 | 0.0% | 0.0pp |
| **OVERALL** | **153** | **433** | **35.3%** | **-1.2pp** |

### Result dir
`logs/experiments/20260303_150602_biomni_biomni_a1_multi/`

### Key Observations
- TASK_PROMPTS 제거만으로는 baseline 회복 실패 (-1.2pp vs baseline 재실행)
- tag leak / placeholder 문제는 해소되었으나, 근본적인 정확도 개선은 없음
- gwas_causal_gene 계열은 -3 정도 하락 (비결정론 범위 내)
- **결론: 단순 프롬프트 조정으로는 60% 목표 달성 불가 → 아키텍처 변경 필요**
- 유저와 논의 후 "진짜 멀티에이전트" 아키텍처로 결정 (Exp#4)

---

## Experiment #4: True Multi-Agent Architecture (IN PROGRESS)
- **Date**: 2026-03-03
- **Agent**: `biomni_a1_multi` (진짜 멀티에이전트 구조)
- **Config**: 8 workers, 433 tasks, gpt-oss-20b, timeout=600s

### Architecture
```
[Orchestrator] — 태스크 분류 + 파라미터 추출 + 라우팅
    ├─ [GWAS Catalog 전문가] — rsID 조회, association 파싱, p-value 계산
    ├─ [Monarch 전문가] — HPO→Disease, Gene→Phenotype 매핑
    ├─ [OpenTargets 전문가] — GraphQL, V2G, causal gene
    ├─ [Ensembl 전문가] — Gene lookup, ENSG mapping
    └─ [Synthesizer] → 전문가 findings 종합 → 최종 답변
```

### Changes from Exp#3 (아키텍처 변경)
1. **TOOL_KNOWLEDGE** — 각 도구의 API 반환 구조, 파싱 패턴, 코드 예시를 전문가 프롬프트에 주입
2. **TASK_PIPELINES** — 태스크별 전문가 에이전트 호출 순서 정의
3. **_run_specialist()** — 전문가 에이전트: 작은 컨텍스트, 해당 도구만, API 지식 포함
4. **_run_synthesis()** — 합성 에이전트: 도구 없이 LLM만으로 최종 답변 생성
5. **predict() 라우팅** — TASK_PIPELINES에 매칭되면 멀티에이전트, 아니면 기존 단일 에이전트 fallback

### Hypothesis
- 각 전문가 에이전트가 작은 컨텍스트에서 자기 도구만 깊게 다뤄 도구 사용 정확도 향상
- 특히 gwas_variant_prioritization (3/43), patient_gene_detection (1/50), rare_disease_diagnosis (0/30) 대폭 개선 기대
- 예상 정확도: ~45-55% (전문가 에이전트 효과)

### Results
| Task Type | Correct | Total | Accuracy | vs Baseline |
|---|---|---|---|---|
| gwas_causal_gene_opentargets | 39 | 50 | 78.0% | +4.0pp |
| gwas_causal_gene_pharmaprojects | 33 | 50 | 66.0% | -2.0pp |
| gwas_causal_gene_gwas_catalog | 24 | 50 | 48.0% | **-10.0pp** |
| screen_gene_retrieval | 22 | 50 | 44.0% | 0.0pp |
| lab_bench_seqqa | 21 | 50 | 42.0% | **+14.0pp** |
| lab_bench_dbqa | 14 | 50 | 28.0% | -2.0pp |
| gwas_variant_prioritization | 1 | 43 | 2.3% | -4.7pp |
| crispr_delivery | 1 | 10 | 10.0% | -20.0pp |
| patient_gene_detection | 6 | 50 | 12.0% | **+10.0pp** |
| rare_disease_diagnosis | 0 | 30 | 0.0% | 0.0pp |
| **OVERALL** | **161** | **433** | **37.2%** | **+0.7pp** |

### Key Observations
- **프롬프트 축소 실패**: `_generate_system_prompt()`가 도구 1개만 넣어도 ~27K 생성 → TOOL_KNOWLEDGE 추가로 오히려 ~29K
- **patient_gene_detection**: 1→6 (+5건, +10pp) — Monarch specialist의 TOOL_KNOWLEDGE가 효과적
- **lab_bench_seqqa**: 14→21 (+7건, +14pp) — 가장 큰 개선. 단일 도구 집중이 효과적인 것으로 보임
- **gwas_causal_gene_gwas_catalog**: 29→24 (-5건, -10pp) — specialist 프롬프트가 기존 패턴을 방해
- **crispr_delivery**: 3→1 (-2건, -20pp) — 도구 없는 MCQ에서 specialist 라우팅이 오히려 해로움
- **결론**: `_generate_system_prompt()` 함수 자체가 프롬프트 최적화의 병목. Exp#5에서 이를 완전히 우회하여 ~2-3.6K 직접 작성 프롬프트 사용

---

## Experiment #5: Lightweight Specialist Prompts (PENDING)
- **Date**: 2026-03-03
- **Agent**: `biomni_a1_exp5` (경량 specialist 프롬프트)
- **Config**: 8 workers, 433 tasks, gpt-oss-20b, timeout=600s
- **File**: `agent/biomni_a1_exp5.py`

### Architecture (Exp#4와 동일한 멀티에이전트 구조, 프롬프트만 변경)
```
[Orchestrator] — 태스크 분류 + 파라미터 추출 + 라우팅
    ├─ [GWAS Catalog 전문가] — 2,326 chars (vs Exp#4: ~29K)
    ├─ [Monarch 전문가] — 3,633 chars
    ├─ [OpenTargets 전문가] — 1,783 chars
    ├─ [Ensembl 전문가] — 1,592 chars
    └─ [Synthesizer] → 전문가 findings 종합 → 최종 답변
```

### Key Changes from Exp#4 (단일 변수 변경: 프롬프트 크기)
1. **`_generate_system_prompt()` 완전 우회** — a1.system_prompt에 직접 할당
2. **`SPECIALIST_SYSTEM_PROMPTS`** — 각 도구별 ~2-3.6K chars의 직접 작성 프롬프트
3. **`_A1_CORE_INSTRUCTIONS`** — A1 루프 파싱 태그만 935 chars로 추출
4. **나머지 로직 동일** — 라우팅, 합성, postprocess 등 Exp#4와 동일

### Hypothesis
- 프롬프트 27K→2~3.6K로 축소 → 20B 모델의 instruction following 대폭 개선
- 도구별 API 지식 + 코드 예시가 집중 주입되어 도구 사용 정확도 향상
- 예상 정확도: ~40-50% (프롬프트 축소 효과)

### Results
_실행 대기 중_
