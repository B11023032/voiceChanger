import streamlit as st
from pytube import YouTube
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import io

# 設定頁面標題
st.title("YouTube 音訊視覺化工具 🎵")

# 輸入 YouTube 網址
youtube_url = st.text_input("請在這裡輸入 YouTube 影片網址：")

if youtube_url:
    try:
        # 顯示影片標題
        yt = YouTube(youtube_url)
        st.write(f"**影片標題:** {yt.title}")
        st.image(yt.thumbnail_url, width=300)

        # 下載音訊
        with st.spinner('正在下載並處理音訊中，請稍候...'):
            audio_stream = yt.streams.filter(only_audio=True).first()
            
            # 將音訊下載到記憶體中
            audio_buffer = io.BytesIO()
            audio_stream.stream_to_buffer(audio_buffer)
            audio_buffer.seek(0)

            # 使用 st.audio 播放音訊
            st.audio(audio_buffer, format='audio/mp4')

            # 使用 librosa 載入音訊
            # librosa 需要一個檔案路徑或是一個支援 seek/read 的檔案物件
            y, sr = librosa.load(audio_buffer, sr=None)

            # 繪製音訊波形圖
            st.write("### 歌曲音訊波形圖")
            fig, ax = plt.subplots(figsize=(10, 4))
            librosa.display.waveshow(y, sr=sr, ax=ax)
            ax.set_title('音訊波形')
            ax.set_xlabel('時間 (秒)')
            ax.set_ylabel('振幅')
            st.pyplot(fig)

            # 繪製音訊頻譜圖 (Spectrogram)
            st.write("### 歌曲音訊頻譜圖")
            D = librosa.stft(y)
            S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
            fig_spec, ax_spec = plt.subplots(figsize=(10, 4))
            img = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log', ax=ax_spec)
            fig_spec.colorbar(img, ax=ax_spec, format='%+2.0f dB', label='分貝')
            ax_spec.set_title('音訊頻譜圖')
            ax_spec.set_xlabel('時間 (秒)')
            ax_spec.set_ylabel('頻率 (赫茲)')
            st.pyplot(fig_spec)

    except Exception as e:
        st.error(f"發生錯誤：{e}")
        st.warning("請確認您輸入的是有效的 YouTube 影片網址。有些影片可能因為版權或隱私設定而無法下載。")