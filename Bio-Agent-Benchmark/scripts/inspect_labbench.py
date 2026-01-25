from datasets import load_dataset
import pandas as pd

# Lab-Bench의 알려진 서브셋 중 하나인 'LitQA'와 'DbQA' 등을 확인해봅니다.
# 8개 서브셋: 'LitQA', 'DbQA', 'SeqQA' 등 추정.
# 일단 에러 메시지 등을 통해 서브셋 목록을 확인하거나, 기본 설정을 시도합니다.

try:
    print("Trying to load 'LitQA' subset...")
    ds = load_dataset(
        "futurehouse/lab-bench", "LitQA", split="train[:2]", trust_remote_code=True
    )
    print("\n[LitQA Sample]")
    print(ds[0])
    print("Columns:", ds.column_names)
except Exception as e:
    print(f"Failed to load LitQA: {e}")

try:
    print("\nTrying to load 'DbQA' subset...")
    ds = load_dataset(
        "futurehouse/lab-bench", "DbQA", split="train[:2]", trust_remote_code=True
    )
    print("\n[DbQA Sample]")
    print(ds[0])
except Exception as e:
    print(f"Failed to load DbQA: {e}")
