import streamlit as st
from streamlit_webrtc import VideoTransformer, AudioTransformer, webrtc_streamer, RTCConfiguration

class VideoProcessor(VideoTransformer):
    def __init__(self):
        self.frame_count = 0

    def transform(self, frame):
        self.frame_count += 1
        # ここでOpenCVなどのライブラリを使用してフレームを処理することができます。
        # この例では、単純にフレームのカウントを表示しています。
        st.write(f"Processed frame {self.frame_count}")
        return frame

class AudioProcessor(AudioTransformer):
    def __init__(self):
        self.audio_count = 0

    def transform(self, audio_chunk):
        self.audio_count += 1
        # ここでPyAudioなどのライブラリを使用してオーディオを処理することができます。
        # この例では、単純にオーディオのカウントを表示しています。
        st.write(f"Processed audio chunk {self.audio_count}")
        return audio_chunk

def main():
    st.title("WebRTCを使用したストリームリットアプリ")

    rtc_configuration = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    video_transformer = VideoProcessor()
    audio_transformer = AudioProcessor()

    webrtc_streamer(
        key="example",
        video_transformer_factory=video_transformer,
        audio_transformer_factory=audio_transformer,
        rtc_configuration=rtc_configuration,
        async_processing=True,
        enable_video=True,
        enable_audio=True,
        client_settings={"width": 640, "height": 480},
    )

if __name__ == "__main__":
    main()
