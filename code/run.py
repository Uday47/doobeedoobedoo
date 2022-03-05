from cleanrun import *
import streamlit as st
import tempfile


SEQUENCE_LENGTH = 16
VIDEO_PATH = r"C:/sih/videos/Abuse007_x264_SparkVideo.mp4"
FRAME_DES = r"C:/sih/videos/"
MODEL_PATH = r"C:/sih/models/c3d_11_0.01.h5"

# StreamLit App

st.title('Crime Recognition')
st.header("Upload video")

uploaded_file = st.file_uploader("Choose a video for analysis", type = ['mp4'])
if uploaded_file is not None:
	video_file = uploaded_file.read()
	st.video(video_file)
	tfile = tempfile.NamedTemporaryFile(delete=False) 
	tfile.write(video_file)

	SEQ_IMAGE = preprocessvideo(SEQUENCE_LENGTH = SEQUENCE_LENGTH, VIDEO_PATH = tfile.name, FRAME_DES = FRAME_DES)
	CLASS = infercrime(SEQ_IMAGE = SEQ_IMAGE, MODEL_PATH = MODEL_PATH)
	PREDICTION = "Crime Event: " + CLASS.upper()
	st.subheader(PREDICTION)
	# st.subheader(CLASS)