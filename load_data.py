import pandas as pd

df = pd.read_csv("data/sample_chat.csv", encoding="utf-8")
print(df.head())
print("클래스 분포:\n", df['label'].value_counts())