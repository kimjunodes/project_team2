import cv2
from torch import rand
import streamlitTest2 as st
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import random

devices = AudioUtilities.GetSpeakers()  # 오디오 받아오기
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))


st.title("Webcam Test")

rad = st.sidebar.radio("Solution",["MAIN"])
FRAME_WINDOW = st.image([])
ss = st.sidebar.slider("ABC",0,100)
print(volume)

if rad == "MAIN":
    bt1 = st.button("ButtonA")
    bt2 = st.button("ButtonB")
    bt3 = st.button("ButtonC")

    if bt1:
        st.write("A")
        rad = st.sidebar.radio("Solution",["A"])

    
    if bt2:
        st.write("B")
        rad = st.sidebar.radio("Solution",["B"])

   
    if bt3 == True:
        st.write("C")
        rad = st.sidebar.radio("Solution",["C"])    
