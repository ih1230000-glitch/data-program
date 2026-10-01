import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="오픈소스 데이터 회귀 분석기", layout="wide")
st.title("📊 오픈소스 데이터 기반 선형 & 비선형 회귀 계산기")
st.write("실제 CSV 데이터 파일을 업로드하여 독립/종속 변수를 선택하고, 경사하강법으로 회귀 수식을 도출하세요.")

# 사이드바: 데이터 파일 업로드
st.sidebar.header("1. 데이터 파일 업로드")
uploaded_file = st.sidebar.file_uploader("CSV 파일을 업로드하세요", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📋 업로드된 데이터 미리보기 (상위 5행)")
    st.dataframe(df.head())

    columns = df.columns.tolist()

    # 사이드바: 변수 선택
    st.sidebar.header("2. 분석 변수 선택")
    x_col = st.sidebar.selectbox("X축 독립 변수 선택", columns)
    y_col = st.sidebar.selectbox("Y축 종속 변수 선택", columns)

    # 사이드바: 모델 및 하이퍼파라미터 설정
    st.sidebar.header("3. 모델 & 하이퍼파라미터")
    model_type = st.sidebar.radio("모델 선택", ["선형 회귀 (y = ax + b)", "2차 비선형 회귀 (y = w2·x² + w1·x + b)"])
    learning_rate = st.sidebar.number_input("학습률 (Learning Rate)", value=0.0001, format="%.5f")
    epochs = st.sidebar.slider("학습 횟수 (Epochs)", 100, 10000, 2000, step=100)

    # 데이터 정제 (선택한 컬럼의 결측치 제거)
    data = df[[x_col, y_col]].dropna()
    X = data[x_col].values
    y_true = data[y_col].values
    N = len(X)

    if N > 1:
        # 1) 선형 회귀 학습
        if model_type == "선형 회귀 (y = ax + b)":
            w, b = 0.0, 0.0
            for _ in range(epochs):
                y_pred = w * X + b
                dw = (-2 / N) * np.sum(X * (y_true - y_pred))
                db = (-2 / N) * np.sum(y_true - y_pred)
                w -= learning_rate * dw
                b -= learning_rate * db

            x_dense = np.linspace(min(X), max(X), 100)
            y_dense = w * x_dense + b
            formula_str = f"y = {w:.4f}x + {b:.4f}"

        # 2) 2차 비선형 회귀 학습
        else:
            w2, w1, b = 0.0, 0.0, 0.0
            for _ in range(epochs):
                y_pred = w2 * (X**2) + w1 * X + b
                error = y_true - y_pred
                dw2 = (-2 / N) * np.sum((X**2) * error)
                dw1 = (-2 / N) * np.sum(X * error)
                db = (-2 / N) * np.sum(error)
                w2 -= learning_rate * dw2
                w1 -= learning_rate * dw1
                b -= learning_rate * db

            x_dense = np.linspace(min(X), max(X), 100)
            y_dense = w2 * (x_dense**2) + w1 * x_dense + b
            formula_str = f"y = {w2:.4f}x² + {w1:.4f}x + {b:.4f}"

        # 화면 출력 구성
        col1, col2 = st.columns([2, 1])

        with col1:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(X, y_true, color='crimson', alpha=0.7, label='오픈소스 데이터 포인트', zorder=5)
            ax.plot(x_dense, y_dense, color='navy', linewidth=2, label='도출된 회귀선')
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.5)
            st.pyplot(fig)

        with col2:
            st.subheader("📌 학습 결과 요약")
            st.success(f"**도출된 최종 수식:**\n\n${formula_str}$")
            st.metric(label="분석에 사용된 데이터 수", value=f"{N}개 행")
    else:
        st.warning("⚠️ 선택한 컬럼에 유효한 데이터가 2개 이상 필요합니다.")
else:
    st.info("👈 왼쪽 사이드바에서 분석할 **CSV 형식의 오픈소스 데이터 파일**을 업로드해 주세요!")

    # 안내용 가짜 데이터 예시 보여주기
    st.markdown("### 💡 CSV 파일 형식 예시")
    sample_df = pd.DataFrame({
        "시간(X)": [1, 2, 3, 4, 5],
        "값(Y)": [2.1, 4.2, 8.8, 16.1, 25.0]
    })
    st.dataframe(sample_df)
    st.markdown("위와 같이 열 이름과 숫자로 이루어진 CSV 파일을 준비하시면 됩니다.")
