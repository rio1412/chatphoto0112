import cv2
import streamlit as st
import numpy as np

# 背景差分法を使用するための設定
background_subtractor = cv2.createBackgroundSubtractorMOG2()

# streamlitの設定
st.set_page_config(page_title="Background Subtraction", page_icon=":camera:", layout="wide")
st.title("Background Subtraction")

# カメラ起動ボタンを作成する
start_camera = st.button('カメラを起動する')

# カメラ画像を表示するエリアを作成する
image_placeholder = st.empty()
image_placeholder2 = st.empty()

# カメラが起動しているかどうかのフラグ
camera_running = False

if start_camera:
    # カメラを起動する
    capture = cv2.VideoCapture(0)
    camera_running = True

while camera_running:
    # フレームを読み込む
    ret, frame = capture.read()

    # 背景差分法を適用する
    foreground_mask = background_subtractor.apply(frame)

    # ノイズを取り除く
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, kernel)

    # 二値化する
    threshold_value = 128
    ret, foreground_mask = cv2.threshold(foreground_mask, threshold_value, 255, cv2.THRESH_BINARY)

    # 輪郭を抽出する
    contours, hierarchy = cv2.findContours(foreground_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 輪郭の周囲長を計算する
    for contour in contours:
      perimeter = cv2.arcLength(contour, True)

      # 輪郭の周囲長が一定以上であれば、人が動いているとみなす
      if perimeter > 300:
          cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)

    # 結果を表示する
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    foreground_mask = cv2.cvtColor(foreground_mask, cv2.COLOR_GRAY2RGB)
    image_placeholder.image(frame, channels="RGB", use_column_width=True)
    image_placeholder2.image(foreground_mask, channels="RGB", use_column_width=True)

    # "q"キーで終了する
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# カメラを解放する
if camera_running:
    capture.release()

# ウィンドウをすべて閉じる
cv2.destroyAllWindows()
