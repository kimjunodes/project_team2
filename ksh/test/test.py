import streamlit as st
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

cVol = int(volume.GetMasterVolumeLevelScalar() * 100)
sVol = volume.SetMasterVolumeLevel()
volshow = st.slider('Volume', 0, 100, sVol)
st.write("Volume : ", volshow)