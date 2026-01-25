#!/bin/bash
set -e  # 에러 발생 시 즉시 중단

echo "🚀 Setting up Bio-Agent-Benchmark environment..."

# 1. 가상환경 생성 (프로젝트 루트의 .venv 폴더)
echo "📦 Creating virtual environment (.venv)..."
python3 -m venv .venv

# 2. 가상환경 활성화 및 pip 업그레이드
source .venv/bin/activate
echo "🔄 Upgrading pip..."
pip install --upgrade pip

# 3. 의존성 설치
echo "📥 Installing dependencies from requirements.txt..."
pip install -r Bio-Agent-Benchmark/requirements.txt

echo "✅ Setup complete!"
echo ""
echo "👉 To activate the environment, run:"
echo "    source .venv/bin/activate"
