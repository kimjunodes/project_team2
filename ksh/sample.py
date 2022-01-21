import cv2
import streamlit as st

st.title("Webcam Test")

rad = st.sidebar.radio("Solution",["A", "B", "C"])
ckb1 = st.sidebar.checkbox("A")
ckb2 = st.sidebar.checkbox("B")
ckb3 = st.sidebar.checkbox("C")
FRAME_WINDOW = st.image([])

if rad == "A":
    st.write("A")
    if st.button("Button") == True:
        rad = "B"
    
if rad == "B":
    st.write("B")

if rad == "C":
    st.write("C")


if ckb1 == True:
    st.write("A")

if ckb2 == "B":
    st.write("B")

if ckb3 == "C": 
    st.write("C")
