import pandas as pd
from datasets import load_dataset

dataset = pd.read_csv("data.csv")

df = load_dataset("csv", data_files="data.csv")

print(f"Loaded {len(dataset)} rows with pandas")
print(f"Loaded dataset with HuggingFace datasets")