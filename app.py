import streamlit as st
import streamlit_webrtc as webrtc

def main():
    st.title("WebRTCを使用したストリームリットアプリ")

    # カメラを起動する
    video_stream = webrtc.VideoTransformer(
        input_video=True,
        preferred_output_format=webrtc.OutputFormat.MJPEG,
        on_error=st.error,
    )

    # マイクを起動する
    audio_stream = webrtc.AudioTransformer(
        input_audio=True,
        on_error=st.error,
    )

    # WebRTCのメディアストリームを表示する
    webrtc_streamer(
        key="example",
        video_transformer_factory=video_stream,
        audio_transformer_factory=audio_stream,
    )

if __name__ == "__main__":
    main()
