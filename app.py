import streamlit as st
import cv2

def main():
    st.title("Webカメラを使用したストリームリットアプリ")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("カメラを開けませんでした")
        return

    while True:
        ret, frame = cap.read()

        if not ret:
            st.warning("フレームを取得できませんでした")
            break

        # OpenCVでフレームを処理する
        # ...

        # ストリームリットアプリでフレームを表示する
        st.image(frame, channels="BGR")

    cap.release()

if __name__ == "__main__":
    main()
