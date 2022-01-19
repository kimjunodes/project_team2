import cv2
import streamlit as st
from screen import main

st.title("Webcam Test")
run = st.checkbox('Run')
FRAME_WINDOW = st.image([])
cam = cv2.VideoCapture(0)
vol = st.slider("volume", min_value=0, max_value=100)
while run:
    ret, frame = cam.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(main)

else:
    st.write('Stopped')

st.write("volume", vol)
