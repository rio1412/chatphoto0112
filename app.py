import streamlit as st
import requests
import base64
from io import BytesIO
import numpy as np
import cv2
import tensorflow as tf
import tensorflow_hub as hub

st.title("ESRGAN: Super-Resolution")

# モデルの読み込み
hub_url = 'https://tfhub.dev/captain-pool/esrgan-tf2/1'
sr_model = hub.load(hub_url)

# 画像のアップロード用ウィジェット
uploader = st.file_uploader(label='Upload an image')

# スライダーの作成
contrast_slider = st.slider(
    label='Contrast:',
    min_value=0.0,
    max_value=2.0,
    step=0.01,
    value=1.0
)

hue_slider = st.slider(
    label='Hue:',
    min_value=-0.5,
    max_value=0.5,
    step=0.01,
    value=0.0
)

brightness_slider = st.slider(
    label='Brightness:',
    min_value=-0.5,
    max_value=0.5,
    step=0.01,
    value=0.0
)

white_balance_slider = st.slider(
    label='White Balance:',
    min_value=0.0,
    max_value=2.0,
    step=0.01,
    value=1.0
)

# ボタンのクリック時の処理
def on_button_click():
    if uploader is None:
        st.warning('Please select an image to enhance.')
        return
    input_image = cv2.imdecode(np.fromstring(uploader.read(), np.uint8), cv2.IMREAD_COLOR)
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

    # コントラストの調整
    contrast = contrast_slider
    output_image = tf.image.adjust_contrast(input_image, contrast)

    # 色味の調整
    hue = hue_slider
    output_image = tf.image.adjust_hue(output_image, hue)

    output_image = tf.image.convert_image_dtype(output_image, tf.float32)
    output_image = tf.expand_dims(output_image, axis=0)
    output_image = sr_model(output_image)[0]
    output_image = tf.squeeze(output_image)
    output_image = tf.clip_by_value(output_image, 0, 1)
    output_image = tf.image.convert_image_dtype(output_image, dtype=tf.uint8)
    output_image = np.array(output_image)
    output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
    output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)

    # 入力画像を表示
    st.image(input_image, channels='RGB', caption='Input Image')

    # 出力画像を表示
    st.image(output_image, channels='RGB', caption='Enhanced Image')

    # 画像の保存
    file_extension = '.jpg'
    with open('enhanced_image' + file_extension, 'wb') as f:
        f.write(cv2.imencode(file_extension, output_image)[1])

# ボタンの作成
if uploader is not None:
    st.button(label='Enhance image', on_click=on_button_click)

# ウィジェットの表示
st.write('Adjust the image:')
st.write('Contrast:')
st.write(contrast_slider)
st.write('Hue:')
st.write(hue_slider)
st.write('Brightness:')
st.write(brightness_slider)
st.write('White_balance:')
st.write(white_balance_slider)
