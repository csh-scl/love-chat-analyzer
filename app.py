import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# 모델과 벡터라이저 불러오기
@st.cache_resource
def load_model():
    model = joblib.load("model/logistic_model.pkl")
    vectorizer = joblib.load("model/tfidf_vectorizer.pkl")
    label_encoder = joblib.load("model/label_encoder.pkl")
    return model, vectorizer, label_encoder

model, vectorizer, label_encoder = load_model()

st.title("💌 Love Chat Emotion Analyzer")

user_input = st.text_area("연인과 주고 받은 문장을 입력하세요:", height=100)

if st.button("감정 분석하기"):
    if user_input.strip() == "":
        st.warning("문장을 입력해 주세요!")
    else:
        # 벡터화 후 예측
        X = vectorizer.transform([user_input])
        pred = model.predict(X)[0]
        
        # 예측 결과를 원래 라벨로 변환
        emotion = label_encoder.inverse_transform([pred])[0]
        
        # 감정에 따라 다른 색상으로 표시
        if emotion == "긍정":
            st.success(f"예측 감정: {emotion}")
            st.write("😊 긍정적인 감정이 담긴 메시지네요!")
        elif emotion == "부정":
            st.error(f"예측 감정: {emotion}")
            st.write("😢 부정적인 감정이 느껴지는 메시지입니다.")
        else:  # 중립
            st.info(f"예측 감정: {emotion}")
            st.write("😐 중립적인 톤의 메시지입니다.")
