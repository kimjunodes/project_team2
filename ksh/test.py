import cv2
from torch import rand
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

from vol import vol

devices = AudioUtilities.GetSpeakers()  # 오디오 받아오기
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

print(AudioUtilities.GetSpeakers())