import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# 페이지 설정
st.set_page_config(page_title="사용자 입력 회귀 분석기", layout="wide")
st.title("📊 직접 숫자를 입력하는 선형 & 비선형 회귀 프로그램")
st.write("X값과 Y값을 직접 입력하여 오차를 최소화하는 회귀 수식과 그래프를 확인하세요.")

# 사이드바: 데이터 직접 입력
st.sidebar.header("1. 숫자 데이터 입력")
x_input = st.sidebar.text_input("X 값 입력 (쉼표로 구분)", "1, 2, 3, 4, 5")
y_input = st.sidebar.text_input("Y 값 입력 (쉼표로 구분)", "2.1, 3.9, 9.2, 15.8, 25.1")

# 사이드바: 모델 및 파라미터 설정
st.sidebar.header("2. 모델 & 하이퍼파라미터")
model_type = st.sidebar.radio("모델 선택", ["선형 회귀 (y = ax + b)", "2차 비선형 회귀 (y = w2·x² + w1·x + b)"])
learning_rate = st.sidebar.slider("학습률 (Learning Rate)", 0.0001, 0.05, 0.002, step=0.0005, format="%.4f")
epochs = st.sidebar.slider("학습 횟수 (Epochs)", 100, 10000, 3000, step=100)

# 데이터 변환 및 검증
try:
    X = np.array([float(i.strip()) for i in x_input.split(",") if i.strip() != ""])
    y_true = np.array([float(i.strip()) for i in y_input.split(",") if i.strip() != ""])

    if len(X) != len(y_true):
        st.error("⚠️ X와 Y의 데이터 개수가 일치하지 않습니다. 개수를 맞춰주세요!")
    elif len(X) < 2:
        st.error("⚠️ 최소 2개 이상의 데이터를 입력해야 회귀 분석이 가능합니다.")
    else:
        N = len(X)

        # 1) 선형 회귀 경사하강법
        if model_type == "선형 회귀 (y = ax + b)":
            w, b = 0.0, 0.0
            for _ in range(epochs):
                y_pred = w * X + b
                dw = (-2 / N) * np.sum(X * (y_true - y_pred))
                db = (-2 / N) * np.sum(y_true - y_pred)
                w -= learning_rate * dw
                b -= learning_rate * db

            # 부드러운 그래프를 그리기 위한 연속 입력값 생성
            x_dense = np.linspace(min(X) - 1, max(X) + 1, 100)
            y_dense = w * x_dense + b
            formula_str = f"y = {w:.4f}x + {b:.4f}"

        # 2) 2차 비선형 회귀 경사하강법
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

            x_dense = np.linspace(min(X) - 1, max(X) + 1, 100)
            y_dense = w2 * (x_dense**2) + w1 * x_dense + b
            formula_str = f"y = {w2:.4f}x² + {w1:.4f}x + {b:.4f}"

        # 결과 화면 레이아웃 (시각화 및 요약)
        col1, col2 = st.columns([2, 1])

        with col1:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(X, y_true, color='red', s=90, label='입력한 데이터 점', zorder=5)
            ax.plot(x_dense, y_dense, color='royalblue', linewidth=2, label=f'도출된 회귀선')
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.6)
            st.pyplot(fig)

        with col2:
            st.subheader("📌 학습 결과 요약")
            st.success(f"**도출된 최종 수식:**\n\n${formula_str}$")
            st.write("**입력된 좌표 목록:**")
            for xi, yi in zip(X, y_true):
                st.write(f"- ($x$: {xi}, $y$: {yi})")

except ValueError:
    st.error("⚠️ 숫자와 쉼표(,) 형식으로만 입력해 주세요! (예: 1, 2, 3, 4, 5)")
