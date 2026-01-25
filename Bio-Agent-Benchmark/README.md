# Bio-Agent-Benchmark (Phase 1)

Biomedical Tool Use Agent를 위한 통합 벤치마크 평가 파이프라인입니다.
다양한 생물의학 벤치마크(Biomni Eval1, Lab-bench)를 단일 인터페이스로 통합하여, LLM 에이전트의 성능을 객관적으로 측정하고 분석합니다.

## 🎯 Phase 1 목표 및 달성 기능
- **통합 벤치마크**: `Biomni Eval1`과 `Lab-bench` (8개 서브셋)를 통합하여 단일 명령어로 실행 가능.
- **모듈형 아키텍처**: 벤치마크(`benchmarks/`)와 에이전트(`agent/`)가 분리되어 있어 손쉽게 확장 가능.
- **실험 관리**: `ExperimentRunner`를 통해 실험 실행, 결과 저장, 로깅을 자동화.
- **로깅 및 시각화**: 
  - **W&B (Weights & Biases)** 연동을 통한 실시간 대시보드 제공.
  - 로컬 파일 시스템(`logs/experiments/`)에 상세 결과(JSONL) 및 실패 케이스 자동 저장.
- **자동 분석**: 실험 종료 후 카테고리별 성능 및 실패 원인을 요약한 리포트 출력.

## 🛠️ 설치 (Installation)

### 1. 환경 설정
프로젝트 루트에서 제공되는 설정 스크립트를 실행하여 가상환경을 만들고 의존성을 설치합니다.

```bash
chmod +x setup.sh
./setup.sh
source .venv/bin/activate
```

### 2. API 키 설정
`.env` 파일을 생성하고 필요한 키를 입력합니다.

```ini
# .env 파일 생성
OPENAI_API_KEY=sk-proj-...  # 실제 LLM 에이전트 사용 시 필수
WANDB_API_KEY=...           # W&B 로깅 사용 시 선택 사항 (또는 `wandb login` 명령어 사용)
OPENAI_MODEL=gpt-4o-mini    # 사용할 모델 (기본값)
```

## 🚀 사용법 (Usage)

모든 실행은 `run.py` 스크립트를 통해 이루어집니다.

### 1. 기본 실행 (Mock Agent 테스트)
파이프라인이 정상 동작하는지 확인하기 위해 가짜(Mock) 에이전트로 테스트합니다.

```bash
# Lab-bench 전체 서브셋 실행 (Mock Agent)
python Bio-Agent-Benchmark/run.py run --benchmark=labbench --agent=mock --limit=5
```

### 2. LLM 에이전트로 실제 평가
OpenAI GPT 모델을 사용하여 실제로 문제를 풉니다.

```bash
# Biomni 벤치마크 실행
python Bio-Agent-Benchmark/run.py run --benchmark=biomni --agent=llm --limit=10

# Lab-bench 실행 (W&B 로깅 포함)
python Bio-Agent-Benchmark/run.py run --benchmark=labbench --agent=llm --use_wandb=true
```

### 3. 주요 옵션 설명

| 옵션 | 설명 | 기본값 | 예시 |
| :--- | :--- | :--- | :--- |
| `--benchmark` | 실행할 벤치마크 이름 | `biomni` | `biomni`, `labbench` |
| `--agent` | 사용할 에이전트 유형 | `mock` | `mock`, `llm` |
| `--subset` | Lab-bench 실행 시 특정 서브셋 지정 | `all` | `LitQA2`, `DbQA`, `all` |
| `--limit` | 테스트할 문제 수 제한 (디버깅용) | `None` (전체) | `10`, `100` |
| `--use_wandb` | W&B 실시간 로깅 사용 여부 | `True` | `true`, `false` |

## 📂 프로젝트 구조

```
Bio-Agent-Benchmark/
├── benchmarks/           # 벤치마크 구현체 (Base, Biomni, Lab-bench)
├── agent/                # 에이전트 구현체 (Mock, LLM)
├── experiments/          # 실험 실행기 (Runner)
├── evaluation/           # 결과 분석기 (Analyzer, Metrics)
├── storage/              # 결과 저장소 (Saver, Schema)
├── scripts/              # 유틸리티 스크립트
├── logs/                 # 실험 결과 저장 경로
├── config/               # 설정 파일
└── run.py                # 메인 실행 진입점 (CLI)
```

## 📊 결과 확인

실험이 완료되면 다음 두 곳에서 결과를 확인할 수 있습니다.

1. **터미널 출력**: 실행 직후 분석 리포트가 출력됩니다. (정확도, 실패 샘플 등)
2. **로컬 파일**: `logs/experiments/YYYYMMDD_HHMMSS_benchmark_agent/` 폴더에 다음 파일들이 저장됩니다.
   - `summary.json`: 전체 메트릭 요약
   - `results.jsonl`: 모든 문제에 대한 상세 로그 (질문, 모델 답변, 정답)
   - `failures.json`: 틀린 문제만 모은 파일 (디버깅용)
3. **W&B 대시보드**: `--use_wandb=true` 옵션 사용 시 웹에서 시각화된 결과를 볼 수 있습니다.

---
**Note**: 현재 Phase 1 단계에서는 LLM 에이전트가 도구(Tool)를 사용하지 않으므로, 전문 지식이 필요한 문제에서 낮은 점수가 나올 수 있습니다. 이는 정상적인 결과이며, Phase 2에서 Tool Use 기능이 추가될 예정입니다.
