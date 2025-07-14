import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import yt_dlp
import os
import io

# 設定頁面標題
st.title("YouTube 音訊視覺化工具 🎵 (yt-dlp 版)")

# 輸入 YouTube 網址
youtube_url = st.text_input("請在這裡輸入 YouTube 影片網址：")

if youtube_url:
    try:
        # --- yt-dlp 相關設定 ---
        # 設定下載選項：選擇最佳音訊，格式為 m4a，不要播放列表
        ydl_opts = {
            'format': 'm4a/bestaudio/best',
            'noplaylist': True,
            # 'outtmpl' 讓我們可以控制輸出檔案的暫存位置和名稱
            # '%(id)s.%(ext)s' 會使用影片ID和副檔名作為檔案名稱
            'outtmpl': '%(id)s.%(ext)s',
        }

        with st.spinner('正在獲取影片資訊...'):
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # 只獲取資訊，不下載
                info = ydl.extract_info(youtube_url, download=False)
                title = info.get('title', '無標題')
                thumbnail = info.get('thumbnail', None)
                video_id = info.get('id')
                ext = info.get('ext')
                # 組合暫存檔案的路徑
                temp_audio_path = f"{video_id}.{ext}"

        st.write(f"**影片標題:** {title}")
        if thumbnail:
            st.image(thumbnail, width=300)

        # 下載音訊
        with st.spinner('正在下載並處理音訊中，請稍候...'):
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # 執行下載
                ydl.download([youtube_url])

            # 檢查檔案是否存在
            if os.path.exists(temp_audio_path):
                # 使用 st.audio 播放音訊
                with open(temp_audio_path, 'rb') as audio_file:
                    audio_bytes = audio_file.read()
                    st.audio(audio_bytes)

                # 使用 librosa 載入音訊
                y, sr = librosa.load(temp_audio_path, sr=None)

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

                # 刪除暫存檔案
                os.remove(temp_audio_path)
            else:
                st.error("下載失敗，找不到音訊檔案。")

    except Exception as e:
        st.error(f"發生錯誤：{e}")
        st.warning("請確認您輸入的是有效的 YouTube 影片網址。有些影片可能因為版權或隱私設定而無法下載。")
        # 如果暫存檔案在出錯前已建立，也嘗試刪除它
        if 'temp_audio_path' in locals() and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)