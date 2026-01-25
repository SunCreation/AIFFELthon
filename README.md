# Biomedical AI Agent Project

의생명 분야 동료과학자(Co-Scientist) Agent 개발 및 고도화 프로젝트입니다.  
연구자의 자연어 질문을 이해하고, 적절한 Bioinformatics 도구를 사용하여 분석을 수행한 뒤, 결과 리포트를 생성하는 시스템을 고도화하는 것을 목표로 합니다.

## 📂 Project Structure

### [Bio-Agent-Benchmark](./Bio-Agent-Benchmark/) (Phase 1)
- **역할**: AI Agent의 성능을 평가하기 위한 통합 벤치마크 파이프라인
- **주요 기능**:
  - `Biomni Eval1` 및 `Lab-bench` 데이터셋 통합
  - OpenAI LLM Agent 실행 및 평가
  - W&B (Weights & Biases) 로그 연동
  - 결과 자동 분석 및 리포팅
- **실행 방법**:
  ```bash
  cd Bio-Agent-Benchmark
  # 가상환경 설정
  ./setup.sh
  source .venv/bin/activate
  
  # 벤치마크 실행
  python run.py run --benchmark=biomni --agent=llm --limit=5
  ```

## 🚀 Roadmap

- **Phase 1 (Complete)**: 통합 벤치마크 평가 파이프라인 구축
- **Phase 2 (Planned)**: Tool Use Agent 고도화 (Planner-Executor 구조, RAG 도입)
- **Phase 3 (Planned)**: 자동화된 시각화 리포트 생성 (Volcano plot, Survival curve 등)

## 🛠️ Environment Setup

프로젝트 루트의 `Bio-Agent-Benchmark` 폴더 내 `setup.sh`를 사용하여 독립적인 가상환경을 구축하는 것을 권장합니다.

```bash
cd Bio-Agent-Benchmark
./setup.sh
```
