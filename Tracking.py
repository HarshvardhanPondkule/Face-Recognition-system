import streamlit as st
import cv2
import face_recognition as frg
import yaml 
from utils import recognize, build_dataset
# Path: code\app.py
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, WebRtcMode
import av

st.set_page_config(page_title="Face Recognition", page_icon="🧠", layout="wide")
#Config
cfg = yaml.load(open('config.yaml','r'),Loader=yaml.FullLoader)
PICTURE_PROMPT = cfg['INFO']['PICTURE_PROMPT']
WEBCAM_PROMPT = cfg['INFO']['WEBCAM_PROMPT']



st.markdown(
    """
    <style>
      .identity-box{background:#f7f7f9;border:1px solid #e6e8eb;padding:12px 14px;border-radius:10px;margin-bottom:8px}
      .identity-title{font-weight:600;color:#0f172a;margin-bottom:6px}
      .identity-value{color:#065f46;background:#ecfdf5;border:1px solid #d1fae5;display:inline-block;padding:4px 8px;border-radius:6px}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("## 🧠 Face Recognition")
st.caption("Upload an image or use your webcam. The app detects faces and displays the recognized Moodle ID_Name.")
st.divider()

st.sidebar.title("Settings")



#Create a menu bar
menu = ["Picture","Webcam"]
choice = st.sidebar.selectbox("Input type",menu)
#Put slide to adjust tolerance
TOLERANCE = st.sidebar.slider("Tolerance",0.0,1.0,0.5,0.01)
st.sidebar.info("Tolerance is the threshold for face recognition. The lower the tolerance, the more strict the face recognition. The higher the tolerance, the more loose the face recognition.")

#Infomation section 
st.sidebar.title("Student Information")
combined_container = st.sidebar.empty()
combined_container.markdown(
    """
    <div class="identity-box">
      <div class="identity-title">Moodle ID_Name</div>
      <div class="identity-value">Unknown</div>
    </div>
    """,
    unsafe_allow_html=True,
)
if choice == "Picture":
    st.title("Face Recognition App")
    st.write(PICTURE_PROMPT)
    uploaded_images = st.file_uploader("Upload",type=['jpg','png','jpeg'],accept_multiple_files=True)
    if len(uploaded_images) != 0:
        #Read uploaded image with face_recognition
        for image in uploaded_images:
            image = frg.load_image_file(image)
            image, name, id = recognize(image,TOLERANCE) 
            combined_label = 'Unknown' if id == 'Unknown' else f"{id}_{name.replace(' ', '_')}"
            combined_container.markdown(
                f"""
                <div class=\"identity-box\">
                  <div class=\"identity-title\">Moodle ID_Name</div>
                  <div class=\"identity-value\">{combined_label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.image(image)
    else: 
        st.info("Please upload an image")
    
elif choice == "Webcam":
    st.title("Face Recognition App")
    st.write(WEBCAM_PROMPT)
    # WebRTC-based webcam (works in cloud over HTTPS)
    class VideoProcessor(VideoTransformerBase):
        def __init__(self):
            self.tolerance = 0.5
            self.last_label = "Unknown"

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            img_out, name, id = recognize(img, self.tolerance)
            label = 'Unknown' if id == 'Unknown' else f"{id}_{name.replace(' ', '_')}"
            self.last_label = label
            # recognize() already draws boxes/labels. Ensure BGR for output frame
            return av.VideoFrame.from_ndarray(img_out, format="bgr24")

    rtc_config = RTCConfiguration({
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    })

    ctx = webrtc_streamer(
        key="face-recognition-webrtc",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration=rtc_config,
    )

    # Keep tolerance in sync with processor and display label in sidebar
    if ctx.video_processor:
        ctx.video_processor.tolerance = TOLERANCE
        combined_label = getattr(ctx.video_processor, "last_label", "Unknown")
        combined_container.markdown(
            f"""
            <div class=\"identity-box\">
              <div class=\"identity-title\">Moodle ID_Name</div>
              <div class=\"identity-value\">{combined_label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

with st.sidebar.form(key='my_form'):
    st.title("Developer Section")
    submit_button = st.form_submit_button(label='REBUILD DATASET')
    if submit_button:
        with st.spinner("Rebuilding dataset..."):
            build_dataset()
        st.success("Dataset has been reset")