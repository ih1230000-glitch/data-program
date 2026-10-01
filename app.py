import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="오픈소스 데이터 회귀 분석기", layout="wide")
st.title("📊 오픈소스 데이터 기반 선형 & 비선형 회귀 계산기")
st.write("실제 CSV 데이터 파일을 업로드하여 독립/종속 변수를 선택하고, 경사하강법으로 회귀 수식을 도출하세요.")

st.sidebar.header("1. 데이터 파일 업로드")
uploaded_file = st.sidebar.file_uploader("CSV 파일을 업로드하세요", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("📋 업로드된 데이터 미리보기 (상위 5행)")
        st.dataframe(df.head())
        
        columns = df.columns.tolist()
        
        st.sidebar.header("2. 변수 선택")
        x_col = st.sidebar.selectbox("X축 독립 변수 선택", columns, key="x_col")
        y_col = st.sidebar.selectbox("Y축 종속 변수 선택", columns, key="y_col")
        
        st.sidebar.header("3. 모델 & 하이퍼파라미터")
        model_type = st.sidebar.radio("모델 선택", ["선형 회귀 (y = ax + b)", "2차 비선형 회귀 (y = w2·x² + w1·x + b)"])
        learning_rate = st.sidebar.number_input("학습률 (Learning Rate)", value=0.0001, format="%.5f")
        epochs = st.sidebar.slider("학습 횟수 (Epochs)", 100, 10000, 2000, step=100)
        
        # 1차원 배열로 안전하게 숫자를 추출하도록 개선된 로직
        temp_df = df[[x_col, y_col]].copy()
        X = pd.to_numeric(temp_df[x_col], errors='coerce').values
        y_true = pd.to_numeric(temp_df[y_col], errors='coerce').values
        
        # NaN 값 제거 및 길이 동기화
        valid_mask = ~np.isnan(X) & ~np.isnan(y_true)
        X = X[valid_mask]
        y_true = y_true[valid_mask]
        N = len(X)
        
        if N > 1:
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

            col1, col2 = st.columns([2, 1])

            with col1:
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.scatter(X, y_true, color='crimson', alpha=0.7, label='데이터 포인트', zorder=5)
                ax.plot(x_dense, y_dense, color='navy', linewidth=2, label='회귀선')
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
                ax.legend()
                ax.grid(True, linestyle='--', alpha=0.5)
                st.pyplot(fig)

            with col2:
                st.subheader("📌 학습 결과 요약")
                st.success(f"**도출된 최종 수식:**\n\n${formula_str}$")
                st.metric(label="분석 데이터 수", value=f"{N}개")
        else:
            st.warning("⚠️ 선택한 컬럼에 유효한 숫자 데이터가 2개 이상 필요합니다. 올바른 숫자 컬럼을 선택해 주세요.")
    except Exception as e:
        st.error(f"⚠️ 데이터를 처리하는 중 오류가 발생했습니다: {e}")
else:
    st.info("👈 왼쪽 사이드바에서 분석할 **CSV 형식의 데이터 파일**을 업로드해 주세요!")
