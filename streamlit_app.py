import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import tempfile

st.set_page_config(page_title="音訊視覺化", layout="wide")
st.title("🎵 音訊視覺化工具")

uploaded_file = st.file_uploader("請上傳音訊檔（支援 mp3 / wav）", type=["mp3", "wav"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name[-4:]) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    with st.spinner("正在處理音訊..."):
        try:
            y, sr = librosa.load(tmp_path, sr=None)
            
            fig, ax = plt.subplots(figsize=(12, 4))
            librosa.display.waveshow(y, sr=sr, ax=ax)
            ax.set_title("音訊波形圖", fontsize=14)
            ax.set_xlabel("時間 (秒)")
            ax.set_ylabel("振幅")

            st.pyplot(fig)

        except Exception as e:
            st.error(f"音訊處理錯誤：{e}")
