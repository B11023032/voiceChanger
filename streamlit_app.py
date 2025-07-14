import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# --- 頁面設定 ---
st.set_page_config(
    page_title="音訊視覺化工具",
    page_icon="🎵",
    layout="wide"
)

# --- 主標題 ---
st.title("音訊檔案視覺化工具")
st.write("上傳您的 .wav 或 .mp3 檔案，即可查看其音訊波形圖與頻譜圖。")

# --- 檔案上傳元件 ---
uploaded_file = st.file_uploader(
    "請在這裡選擇或拖曳您的音訊檔：",
    type=['wav', 'mp3']  # 限制使用者只能上傳這兩種格式
)

# --- 當使用者上傳檔案後執行的程式碼 ---
if uploaded_file is not None:
    try:
        # 顯示成功訊息和播放器
        st.success(f"檔案上傳成功：**{uploaded_file.name}**")
        
        # 直接使用 st.audio 播放上傳的檔案
        st.audio(uploaded_file, format='audio/wav')

        # 顯示處理中的提示
        with st.spinner('正在分析音訊並繪製圖表中，請稍候...'):
            # 使用 librosa 載入音訊
            # st.file_uploader 回傳的物件是 file-like 的，可以直接給 librosa.load 使用
            y, sr = librosa.load(uploaded_file, sr=None)

            # --- 繪製音訊波形圖 ---
            st.write("### 歌曲音訊波形圖")
            fig_wave, ax_wave = plt.subplots(figsize=(12, 4))
            librosa.display.waveshow(y, sr=sr, ax=ax_wave, color='royalblue')
            ax_wave.set_title('音訊波形 (Waveform)')
            ax_wave.set_xlabel('時間 (秒)')
            ax_wave.set_ylabel('振幅')
            st.pyplot(fig_wave)

            # --- 繪製音訊頻譜圖 (Spectrogram) ---
            st.write("### 歌曲音訊頻譜圖")
            # 進行短時距傅立葉變換 (STFT) 並轉換為分貝單位
            D = librosa.stft(y)
            S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
            
            fig_spec, ax_spec = plt.subplots(figsize=(12, 5))
            img = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log', ax=ax_spec)
            fig_spec.colorbar(img, ax=ax_spec, format='%+2.0f dB', label='分貝 (dB)')
            ax_spec.set_title('音訊頻譜圖 (Spectrogram)')
            ax_spec.set_xlabel('時間 (秒)')
            ax_spec.set_ylabel('頻率 (赫茲)')
            st.pyplot(fig_spec)

    except Exception as e:
        st.error(f"分析檔案時發生錯誤：{e}")
        st.warning("請確認您上傳的檔案未損壞且為支援的 .wav 或 .mp3 格式。")

else:
    st.info("請上傳一個音訊檔案以開始分析。")