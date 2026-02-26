# Bio-Agent-Benchmark 병렬 실행 & 모니터링 가이드

## 목차
1. [무엇을 바꿨나?](#1-무엇을-바꿨나)
2. [실행 방법](#2-실행-방법)
3. [웹 모니터링 대시보드](#3-웹-모니터링-대시보드)
4. [변경된 파일 상세](#4-변경된-파일-상세)
5. [벤치마크 결과 (gpt-oss-20b)](#5-벤치마크-결과-gpt-oss-20b)
6. [알려진 이슈 & 해결 방법](#6-알려진-이슈--해결-방법)

---

## 1. 무엇을 바꿨나?

기존 코드는 **순차 실행 + OpenAI API 직접 호출**만 지원했습니다. 다음 4가지를 추가했습니다:

| 기능 | 설명 |
|---|---|
| **커스텀 LLM 엔드포인트** | `OPENAI_BASE_URL` 환경변수로 lmdeploy, vLLM, Ollama 등 아무 서버나 사용 가능 |
| **병렬 실행** | `--parallel N` 옵션으로 N개 워커가 동시에 태스크 처리 |
| **스트리밍 + Stall 감지** | 토큰을 실시간으로 받으면서, 서버가 멈추면 자동 타임아웃 |
| **웹 모니터링 대시보드** | 실시간 진행률, 정확도, latency 분포를 웹 UI로 확인 |

**브랜치**: `feat/streaming-monitor` (main 대비 커밋 4개, 파일 6개 수정)

---

## 2. 실행 방법

### 2-1. 사전 준비

```bash
# 1) 레포 클론 & 브랜치 전환
git clone https://github.com/New-Drug-Intoxication/AIFFELthon.git
cd AIFFELthon
git checkout feat/streaming-monitor

# 2) 패키지 설치
cd Bio-Agent-Benchmark
pip install -r requirements.txt
pip install fastapi uvicorn   # 웹 모니터용 (requirements.txt에 없음)
```

### 2-2. 환경변수 설정

```bash
# 필수: LLM 서버 주소 (lmdeploy, vLLM 등)
export OPENAI_BASE_URL=http://서버주소:포트/v1
export OPENAI_API_KEY=dummy               # lmdeploy는 아무 값이나 OK
export OPENAI_MODEL=openai/gpt-oss-20b    # 서버에 배포된 모델명

# 선택: WandB 로깅 끄기
export WANDB_MODE=disabled
```

> **Ollama 사용 시**: `OPENAI_BASE_URL=http://localhost:11434/v1`, `OPENAI_MODEL=llama3.1`

### 2-3. 벤치마크 실행

```bash
# 기본 (순차 실행)
python run.py run --benchmark=biomni --agent=llm

# 병렬 4개
python run.py run --benchmark=biomni --agent=llm --parallel=4

# 병렬 8개
python run.py run --benchmark=biomni --agent=llm --parallel=8
```

로그는 자동으로 표준출력에 찍히지만, **웹 모니터**를 사용하려면 로그를 파일로 저장해야 합니다:

```bash
# 로그를 파일로 저장하면서 실행
python run.py run --benchmark=biomni --agent=llm --parallel=8 \
  > /tmp/benchmark_run.log 2>&1 &

echo "PID: $!"
```

### 2-4. 웹 모니터링 실행 (별도 터미널)

```bash
# 벤치마크 로그 파일을 지정해서 웹 모니터 실행
python web_monitor.py /tmp/benchmark_run.log --port 8080
```

브라우저에서 **http://localhost:8080** 접속하면 실시간 대시보드를 볼 수 있습니다.

### 2-5. tmux 사용 (권장 — 터미널 닫아도 계속 실행)

```bash
# tmux 세션 생성
tmux new-session -d -s bench

# 벤치마크 실행
tmux send-keys -t bench "cd /path/to/Bio-Agent-Benchmark && \
  OPENAI_BASE_URL=http://서버:포트/v1 \
  OPENAI_API_KEY=dummy \
  OPENAI_MODEL=모델명 \
  WANDB_MODE=disabled \
  python run.py run --benchmark=biomni --agent=llm --parallel=8 \
  > /tmp/benchmark_run.log 2>&1 &" Enter

# 웹 모니터 실행
tmux send-keys -t bench "python web_monitor.py /tmp/benchmark_run.log --port 8080" Enter

# tmux 나가기 (세션은 백그라운드에서 계속 실행)
# Ctrl+B, D

# 다시 접속
tmux attach -t bench
```

---

## 3. 웹 모니터링 대시보드

http://localhost:8080 에서 다음 정보를 **2초마다 자동 갱신**으로 볼 수 있습니다:

| 영역 | 내용 |
|---|---|
| **Progress Bar** | 전체 진행률 (완료/전체) |
| **KPI 카드** | 정확도(%), 평균 latency, 처리량(tasks/min), 에러 수 |
| **Active Workers** | 현재 각 워커가 처리 중인 태스크, 토큰 수, reasoning 토큰 수 |
| **Latency 분포** | 0-5s, 5-10s, 10-30s, 30-60s, 1-5m, 5-10m, 10m+ 버킷별 분포 |
| **태스크별 통계** | 태스크 타입별 완료 수, 정확도, 평균 latency |

**API 엔드포인트**: `GET http://localhost:8080/api/stats` — JSON으로 모든 통계를 반환합니다.

---

## 4. 변경된 파일 상세

### `run.py` — CLI 진입점
```diff
+ --parallel 옵션 추가 (기본값: 1 = 순차 실행)
+ parallel 값을 runner에 전달
```

### `agent/llm.py` — LLM 호출 로직
```diff
+ OPENAI_BASE_URL 환경변수 지원 (기존에는 OpenAI API만 가능)
+ 스트리밍 모드: 토큰을 실시간으로 받으면서 진행 상황 로깅
+ Stall Detection: 60초간 토큰이 안 오면 자동 타임아웃
+ reasoning 토큰 카운팅: [STREAM_PROGRESS] 로그에 reasoning=N 포함
+ TTFT(Time To First Token) 측정
+ 구조화된 로그: [REQ], [STREAM_START], [STREAM_PROGRESS], [STREAM_END], [STREAM_STALL]
```

### `experiments/runner.py` — 벤치마크 실행 엔진
```diff
+ ThreadPoolExecutor 기반 병렬 실행
+ 인라인 스코어 계산: 각 태스크 완료 즉시 정답 여부 판정
+ [SCORE] 로그 이벤트: 웹 모니터에서 실시간 정확도 추적용
```

### `monitor.py` — CLI 로그 파서 (새 파일)
```diff
+ 로그 파일을 실시간 파싱하여 통계 집계
+ [REQ], [STREAM_*], [SCORE], [ERR] 이벤트 추적
+ 정확도, latency, 에러 수, 태스크별 통계 계산
```

### `web_monitor.py` — 웹 대시보드 (새 파일)
```diff
+ FastAPI 서버 + 내장 HTML/CSS/JS 대시보드
+ /api/stats 엔드포인트: JSON 통계 반환
+ 다크 테마, 2초마다 자동 갱신
+ Active Worker 카드에 reasoning 토큰 실시간 표시
+ Latency 분포 차트 (0-5s ~ 10m+)
+ 태스크 타입별 정확도 테이블
```

### `benchmarks/biomni.py` — 벤치마크 데이터 로더
```diff
+ task_id를 agent에 전달 (모니터링용 식별자)
```

---

## 5. 벤치마크 결과 (gpt-oss-20b)

`openai/gpt-oss-20b` (lmdeploy 서버) 기준, parallel=4로 실행한 결과:

| 항목 | 값 |
|---|---|
| 전체 정확도 | **130/433 (30.0%)** |
| 완료율 | 426/433 (98.4%) |
| Stall 에러 | 7건 (1.6%) |
| 소요 시간 | 2시간 41분 |
| 처리량 | 2.6 tasks/min |

### 태스크별 정확도

| 태스크 타입 | 정확도 | 평균 응답시간 | 비고 |
|---|---|---|---|
| gwas_causal_gene_opentargets | **76.0%** (38/50) | 1.6s | 최고 성적 |
| gwas_causal_gene_pharmaprojects | **70.0%** (35/50) | 4.2s | |
| gwas_causal_gene_gwas_catalog | **56.0%** (28/50) | 3.7s | |
| screen_gene_retrieval | **46.0%** (23/50) | 4.9s | |
| gwas_variant_prioritization | 14.0% (6/43) | 83.4s | |
| crispr_delivery | 0.0% (0/10) | 2.0s | 태스크 10개뿐 |
| lab_bench_dbqa | 0.0% (0/50) | 54.9s | |
| lab_bench_seqqa | 0.0% (0/50) | 268.0s | stall 6건 |
| patient_gene_detection | 0.0% (0/50) | 138.4s | |
| rare_disease_diagnosis | 0.0% (0/30) | 230.0s | |

### Latency 분포

```
   0-5s:  231 (54.2%)  ████████████████████████████
  5-10s:   66 (15.5%)  ████████
 10-30s:   56 (13.1%)  ███████
 30-60s:   19 ( 4.5%)  ██
   1-5m:   30 ( 7.0%)  ████
  5-10m:    8 ( 1.9%)  █
   10m+:   16 ( 3.8%)  ██

P50: 4.4s | P90: 87.3s | P95: 370.8s | Max: 1561.8s
```

---

## 6. 알려진 이슈 & 해결 방법

### Stall Timeout (서버 응답 중단)

**증상**: `[STREAM_STALL] task=xxx | idle=60.5s` 로그가 찍히며 해당 태스크 실패

**원인**: 모델이 reasoning 토큰을 수만~수십만 개 생성하다가 서버의 `session_len` 또는 GPU 메모리 한도에 도달하면, 서버가 토큰 전송을 멈춤 (연결은 유지된 채로)

**완화 방법**:
- `--parallel` 값을 줄이기 (4 → 2) — GPU 메모리 압박 감소
- 서버의 `--session-len` 값 확인 및 증가
- 서버에서 `nvidia-smi`로 GPU 메모리 모니터링

### 추가 패키지 필요

`web_monitor.py`는 `fastapi`와 `uvicorn`이 필요합니다:
```bash
pip install fastapi uvicorn
```

### 환경변수가 안 먹을 때

`.env` 파일을 사용하는 경우, `Bio-Agent-Benchmark/.env`에 작성:
```
OPENAI_BASE_URL=http://서버:포트/v1
OPENAI_API_KEY=dummy
OPENAI_MODEL=모델명
WANDB_MODE=disabled
```

---

## Quick Start (복붙용)

```bash
# 1. 브랜치 전환
git checkout feat/streaming-monitor

# 2. 패키지 설치
pip install -r requirements.txt
pip install fastapi uvicorn

# 3. 환경변수 설정
export OPENAI_BASE_URL=http://서버:포트/v1
export OPENAI_API_KEY=dummy
export OPENAI_MODEL=모델명
export WANDB_MODE=disabled

# 4. 벤치마크 실행 (백그라운드)
python run.py run --benchmark=biomni --agent=llm --parallel=8 \
  > /tmp/benchmark_run.log 2>&1 &

# 5. 웹 모니터 실행
python web_monitor.py /tmp/benchmark_run.log --port 8080

# 6. 브라우저에서 http://localhost:8080 접속
```
