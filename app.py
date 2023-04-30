import io

import av
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_webrtc import webrtc_streamer

from models.detect import detect_kinotake


class VideoProcessor:
    def __init__(self) -> None:
        self.state = False
        self.threshold = 0.6
        self.disp_score = False
        self.disp_counter = True
        self.weights = ""

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        if self.state:
            img = detect_kinotake(
                img,
                weights_file=self.weights,
                confidence_threshold=self.threshold,
                disp_score=self.disp_score,
                disp_counter=self.disp_counter,
            )
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            pass

        img = av.VideoFrame.from_ndarray(img, format="bgr24")
        return img


def main():
    st.title("Kinoko Takenoko Detection")
    st.caption("「きのこの山」と「たけのこの里」を検出します")

    with st.sidebar:
        weights = st.selectbox(
            "Select Weights:",
            [
                "kinotake_ssd_v1.pth",
                "kinotake_ssd_v2.pth",
                "kinotake_ssd_v3.pth",
            ],
            index=2,
        )
        disp_score = st.checkbox("Score", value=False)
        disp_counter = st.checkbox("Counter", value=True)
        threshold = st.slider(
            "Threshold", min_value=0.0, max_value=1.0, step=0.05, value=0.6
        )

    tab1, tab2, tab3 = st.tabs(["Real-time From Camera", "From Image", "From Video"])

    with tab1:
        ctx = webrtc_streamer(
            key="chocorooms",
            video_processor_factory=VideoProcessor,
            rtc_configuration={
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            },
        )
        if ctx.video_processor:
            ctx.video_processor.state = st.checkbox("DETECTION")
            ctx.video_processor.weights = weights
            ctx.video_processor.disp_score = disp_score
            ctx.video_processor.disp_counter = disp_counter
            ctx.video_processor.threshold = threshold

    with tab2:
        image_file = st.file_uploader("Image", type=["jpg", "jpeg", "png"])
        if image_file:
            with st.spinner("Detecting ..."):
                img = np.array(Image.open(image_file))
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                img = detect_kinotake(
                    img,
                    weights_file=weights,
                    confidence_threshold=threshold,
                    disp_score=disp_score,
                    disp_counter=disp_counter,
                )
                st.image(
                    img,
                    caption="Uploaded Image with Detection",
                    width=None,
                    use_column_width="auto",
                )

    with tab3:
        video_file = st.file_uploader("Video", type=["mov", "mp4", "avi"])
        if video_file is not None:
            with open("temp.mp4", "wb") as f:
                f.write(video_file.read())

            cap = cv2.VideoCapture("temp.mp4")
            if not cap.isOpened():
                st.error("Sorry. Could not open the video file")
                return

            image_container = st.empty()

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = detect_kinotake(
                    frame,
                    weights_file=weights,
                    confidence_threshold=threshold,
                    disp_score=disp_score,
                    disp_counter=disp_counter,
                )
                image_container.image(frame, channels="RGB")

            cap.release()

    with st.expander("開発ストーリー"):
        st.markdown("**データ収集**")
        st.markdown("- iPhone + ミニ三脚で撮影")
        st.markdown("- 画像サイズ: 1200 x 1200")
        st.markdown("- きのこの山6個、たけのこの里6個を1セットとし、1セットあたり10枚撮影")
        st.markdown("- きのこたけのこそれぞれ1箱分のサンプルを使用し、合計100枚のデータを用意")
        st.markdown("")
        st.image("./docs/taking_photo.jpg", width=300)
        st.image("./docs/kinoko-takenoko.jpg", width=300)
        st.markdown("")
        st.markdown("**前処理（アノテーション）**")
        st.markdown("- labelImgを使用")
        st.markdown("- すべてのオブジェクトをボックスで囲む")
        st.markdown("")
        st.image("./docs/annotation.png", width=480)
        st.markdown("")
        st.markdown("**モデル作成**")
        st.markdown("- SSD (Single Shot MultiBox Detector) を使用")
        st.markdown("- 参考情報[2]のコードを転用")
        st.markdown("")
        st.markdown("**学習**")
        st.markdown("- データ拡張（ランダムに拡大・切り出し・反転）")
        st.markdown("- 500エポックを学習。Colab（GPU使用）で2時間程度")
        st.markdown("")
        st.markdown("**Streamlit**")
        st.markdown("- Webカメラの映像をリアルタイム表示")
        st.markdown(
            "- 使用ライブラリ: [streamlit-webrtc](https://github.com/whitphx/streamlit-webrtc)"
        )
        st.markdown("")
        st.markdown("**参考にした情報**")
        st.markdown("- [1] キカガク 画像処理特化コース")
        st.markdown(
            "- [2] チーム・カルポ「[物体検出とGAN、オートエンコーダー、画像処理入門](https://www.amazon.co.jp/gp/product/B09MHLC3F8/)」"
        )
        st.markdown("")
        st.image(
            "https://m.media-amazon.com/images/I/51jOT49zsAL._SX386_BO1,204,203,200_.jpg",
            width=128,
        )
        st.markdown("")
        st.markdown("**リポジトリ**")
        st.markdown(
            "- [GitHub - rockyhg/chocorooms](https://github.com/rockyhg/chocorooms)"
        )


if __name__ == "__main__":
    main()
