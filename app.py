#Modified by Augmented Startups 2021
#Face Landmark User Interface with StreamLit
#Watch Computer Vision Tutorials at www.augmentedstartups.info/YouTube
import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tempfile
import time
from PIL import Image

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic

st.title('Pose Detection Menggunakan MediaPipe')

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 350px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 350px;
        margin-left: -350px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title('Pose Detection Menggunakan MediaPipe')
#st.sidebar.subheader('Parameters')

@st.cache()
def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized

app_mode = st.sidebar.selectbox('Pilih Mode Aplikasi',
['Info','Run on Image','Run on Video']
)

if app_mode =='Info':
    st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 400px;
        margin-left: -400px;
    }
    </style>
    """,
    unsafe_allow_html=True,
    )

elif app_mode =='Run on Video':

    st.set_option('deprecation.showfileUploaderEncoding', False)

    use_webcam = st.sidebar.button('Use Webcam')
    record = st.sidebar.checkbox("Record Video")
    if record:
        st.checkbox("Recording", value=True)

    st.sidebar.markdown('---')
    st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 400px;
        margin-left: -400px;
    }
    </style>
    """,
    unsafe_allow_html=True,
        )

    detection_confidence = st.sidebar.slider('Min Detection Confidence', min_value =0.0,max_value = 1.0,value = 0.5)
    tracking_confidence = st.sidebar.slider('Min Tracking Confidence', min_value = 0.0,max_value = 1.0,value = 0.5)

    st.sidebar.markdown('---')

    st.markdown(' ## Output')

    stframe = st.empty()
    video_file_buffer = st.sidebar.file_uploader("Upload a video", type=[ "mp4", "mov",'avi','asf', 'm4v' ])
    tfflie = tempfile.NamedTemporaryFile(delete=False)


    if not video_file_buffer:
        if use_webcam:
            vid = cv2.VideoCapture(0)
        else:
            st.markdown('Masukkan Video')

    
    else:
        tfflie.write(video_file_buffer.read())
        vid = cv2.VideoCapture(tfflie.name)

        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_input = int(vid.get(cv2.CAP_PROP_FPS))

        #codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        codec = cv2.VideoWriter_fourcc('v','p','0','9')
        #codec = cv2.VideoWriter_fourcc('M','P','G','4') #MP4V-ES
        out = cv2.VideoWriter('output1.mp4', codec, fps_input, (width, height))

        st.sidebar.text('Input Video')
        st.sidebar.video(tfflie.name)
        fps = 0
        i = 0
        drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=2)

        kpi1, kpi2, kpi3 = st.columns(3)

        with kpi1:
            st.markdown("**FrameRate**")
            kpi1_text = st.markdown("0")

        with kpi2:
            st.markdown("**Detected Faces**")
            kpi2_text = st.markdown("0")

        with kpi3:
            st.markdown("**Image Width**")
            kpi3_text = st.markdown("0")

        st.markdown("<hr/>", unsafe_allow_html=True)

        with mp_pose.Pose(
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence) as pose:
            prevTime = 0

            while vid.isOpened():
                i +=1
                ret, frame = vid.read()
                if not ret:
                    continue

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame.flags.writeable = False
                results = pose.process(frame)

                frame.flags.writeable = True
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                currTime = time.time()
                fps = 1 / (currTime - prevTime)
                prevTime = currTime
                if record:
                    #st.checkbox("Recording", value=True)
                    out.write(frame)
                #Dashboard
                kpi1_text.write(f"<h1 style='text-align: center; color: red;'>{int(fps)}</h1>", unsafe_allow_html=True)
                #kpi2_text.write(f"<h1 style='text-align: center; color: red;'>{face_count}</h1>", unsafe_allow_html=True)
                kpi3_text.write(f"<h1 style='text-align: center; color: red;'>{width}</h1>", unsafe_allow_html=True)

                frame = cv2.resize(frame,(0,0),fx = 0.8 , fy = 0.8)
                frame = image_resize(image = frame, width = 720)
                stframe.image(frame,channels = 'BGR',use_column_width=True)

        st.text('Video Processed')

        output_video = open('output.mp4','rb')
        out_bytes = output_video.read()
        st.video(out_bytes)

        vid.release()
        out. release()

elif app_mode =='Run on Image':

    drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=1)

    st.sidebar.markdown('---')

    st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 400px;
        margin-left: -400px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
    st.markdown("Masukkan Gambar . .")
    st.markdown('---')

    detection_confidence = st.sidebar.slider('Detection Confidence Value', min_value =0.0,max_value = 1.0,value = 0.5)
    st.sidebar.markdown('---')

    img_file_buffer = st.sidebar.file_uploader("Upload an image", type=[ "jpg", "jpeg",'png'])

    if img_file_buffer is not None:
        image = np.array(Image.open(img_file_buffer))
        st.sidebar.text('Original Image')
        st.sidebar.image(image)


        # Dashboard
        with mp_pose.Pose(
                static_image_mode=True,
                model_complexity=2,
                min_detection_confidence=detection_confidence) as pose:

            results = pose.process(image)
            out_image = image.copy()
            #annotated_image = image.copy()
            mp_drawing.draw_landmarks(out_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.imwrite(r'output.png', out_image)

            st.subheader('Output Image')
            st.image(out_image, use_column_width=True)

    else:
        st.sidebar.text('Original Image')
        #st.sidebar.image(image)

        #demo_image = DEMO_IMAGE
        #image = np.array(Image.open(demo_image))



# Watch Tutorial at www.augmentedstartups.info/YouTube