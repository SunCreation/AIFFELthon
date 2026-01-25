from datasets import load_dataset
import pandas as pd

# 'LitQA2' 서브셋을 trust_remote_code 없이 로드 시도
try:
    print("Trying to load 'LitQA2' subset...")
    ds = load_dataset("futurehouse/lab-bench", "LitQA2", split="train[:2]")
    print("\n[LitQA2 Sample]")
    print(ds[0])
    print("Columns:", ds.column_names)
except Exception as e:
    print(f"Failed to load LitQA2: {e}")
