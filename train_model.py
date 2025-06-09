import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
import os

# 1) 데이터 불러오기
df = pd.read_csv("data/sample_chat.csv", encoding="utf-8")

# 2) 라벨 인코딩
le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['label'])

print(df.head())
print("라벨 클래스:", list(le.classes_))
print("인코딩 결과 예시:\n", df[['label', 'label_encoded']].head())

# 3) 학습/테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(
    df['sentence'], 
    df['label_encoded'], 
    test_size=0.2, 
    random_state=100
)

print(f"학습 데이터: {len(X_train)}개, 테스트 데이터: {len(X_test)}개")

# 4) TF-IDF 벡터라이저 생성 및 벡터화
tfidf = TfidfVectorizer(max_features=500)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# 5) 로지스틱 회귀 모델 생성 및 학습
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# 6) 테스트 데이터 예측 및 평가
y_pred = model.predict(X_test_tfidf)
print(classification_report(y_test, y_pred, target_names=le.classes_))

# 7) 모델과 벡터라이저 저장
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/logistic_model.pkl")
joblib.dump(tfidf, "model/tfidf_vectorizer.pkl")
joblib.dump(le, "model/label_encoder.pkl")  # 라벨 인코더도 저장
print("모델과 벡터라이저 저장 완료!")
