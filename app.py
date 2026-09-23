import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import streamlit as st
import cv2
import numpy as np
import urllib.request
import tempfile
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ai_coach import FitnessCoachLLM

# --- Custom Streamlit UI Styling ---
st.set_page_config(page_title="AI Fitness Coach", page_icon="🏋️", layout="wide")
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .stTabs [data-baseweb="tab-list"] { gap: 24px; }
        .stTabs [data-baseweb="tab"] {
            height: 50px; white-space: pre-wrap; background-color: transparent;
            border-radius: 4px 4px 0px 0px; gap: 1px; padding-top: 10px; padding-bottom: 10px;
        }
        [data-testid="stImage"] {
            display: flex;
            justify-content: center;
        }
    </style>
""", unsafe_allow_html=True)

MODEL_PATH = "pose_landmarker_lite.task"
if not os.path.exists(MODEL_PATH):
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task", 
        MODEL_PATH
    )

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    return 360 - angle if angle > 180.0 else angle

def calculate_distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def draw_full_body_mesh(image, landmarks):
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10), 
        (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19), 
        (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20), (11, 23), 
        (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28), (27, 29), 
        (28, 30), (29, 31), (30, 32), (27, 31), (28, 32)
    ]
    h, w, _ = image.shape
    for connection in connections:
        p1, p2 = landmarks[connection[0]], landmarks[connection[1]]
        if p1.visibility > 0.4 and p2.visibility > 0.4:
            cv2.line(image, (int(p1.x * w), int(p1.y * h)), (int(p2.x * w), int(p2.y * h)), (255, 255, 0), 2, cv2.LINE_AA)
    for landmark in landmarks:
        if landmark.visibility > 0.4:
            cv2.circle(image, (int(landmark.x * w), int(landmark.y * h)), 5, (255, 0, 255), -1, cv2.LINE_AA)

def draw_transparent_overlay(image, x, y, w, h, alpha=0.6):
    overlay = image.copy()
    cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

with st.sidebar:
    st.title("⚙️ System Control")
    st.markdown("---")
    exercise = st.selectbox("🎯 Target Exercise", ["Bicep Curl", "Squat", "Push-up", "Shoulder Press"])
    
    input_source = st.radio("📹 Input Source", ["Live Webcam", "Video Upload"])
    uploaded_file = None
    if input_source == "Video Upload":
        uploaded_file = st.file_uploader("Upload a workout video", type=["mp4", "mov", "avi"])
        
    run_tracking = st.toggle("🚀 Start Tracking", value=False)
    st.markdown("---")
    st.caption("AI powered by Google Gemini & MediaPipe Tasks Vision API.")

st.title("⚡ AI Elite Fitness Coach")

tab1, tab2 = st.tabs(["🏋️‍♂️ Live Tracking HUD", "🧠 Gemini AI Coach"])

with tab2:
    st.markdown("### 🤖 Ask your personal AI trainer")
    st.write("Leverage Google Gemini to optimize your routine, understand biomechanics, or troubleshoot plateauing.")
    
    if run_tracking:
        st.warning("⚠️ **Tracking is currently active.** Please toggle off 'Start Tracking' in the sidebar to free up system resources.")
    
    api_key = st.text_input("🔑 Enter Google Gemini API Key", type="password")
    
    if api_key:
        coach = FitnessCoachLLM(api_key)
        user_query = st.text_area("What would you like to know?", placeholder="e.g., How can I improve my ankle mobility for deeper squats?")
        
        if st.button("Generate Advice", type="primary", disabled=run_tracking):
            with st.spinner("Analyzing..."):
                response = coach.get_advice(user_query)
                st.success(response)
    else:
        st.info("Please provide a valid Gemini API key to unlock the chat interface.")

with tab1:
    frame_placeholder = st.empty()
    
    if not run_tracking:
        st.info("👈 Please select an input source and initialize tracking from the sidebar.")
    
    # Strictly Whole Numbers Now
    counter = 0      
    stage_l = None
    stage_r = None
    stage_main = None
    feedback = "Awaiting Setup..."
    feedback_color = (0, 255, 255)

    if run_tracking:
        cap = None
        if input_source == "Live Webcam":
            cap = cv2.VideoCapture(0)
        elif input_source == "Video Upload" and uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            cap = cv2.VideoCapture(tfile.name)
        else:
            st.warning("Please upload a video file first to start tracking.")
            
        if cap is not None:
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0 or np.isnan(fps):
                fps = 30.0
            frame_duration = 1.0 / fps
            frame_duration_ms = int(frame_duration * 1000)
            
            base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO,
                min_pose_detection_confidence=0.5,
                min_tracking_confidence=0.5)
                
            detector = vision.PoseLandmarker.create_from_options(options)
            
            timestamp_ms = 0
            last_rep_ms_l = -5000
            last_rep_ms_r = -5000
            
            while cap.isOpened() and run_tracking:
                loop_start = time.time()
                
                ret, frame = cap.read()
                if not ret: 
                    st.success("Video processing complete.")
                    break
                
                h, w, _ = frame.shape
                max_height = 650
                max_width = 1000
                scale = min(max_height / h, max_width / w)
                
                if scale < 1.0: 
                    frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
                    h, w, _ = frame.shape 
                
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                timestamp_ms += frame_duration_ms
                results = detector.detect_for_video(mp_image, timestamp_ms)
                
                percentage = 0
                
                if results.pose_landmarks:
                    lm = results.pose_landmarks[0]
                    try:
                        get_px = lambda i: [lm[i].x * w, lm[i].y * h]
                        sh_l, hp_l = get_px(11), get_px(23)
                        torso_length = calculate_distance(sh_l, hp_l)
                        
                        if exercise in ["Bicep Curl", "Shoulder Press"]:
                            vis_l = lm[11].visibility > 0.6 and lm[13].visibility > 0.6 and lm[15].visibility > 0.6
                            vis_r = lm[12].visibility > 0.6 and lm[14].visibility > 0.6 and lm[16].visibility > 0.6
                            
                            perc_l, perc_r = 0, 0
                            fb_l, fb_r = "Perfect Form", "Perfect Form"
                            
                            # --- LEFT ARM PROCESSING ---
                            if vis_l:
                                s_l, e_l, w_l = get_px(11), get_px(13), get_px(15)
                                angle_l = calculate_angle(s_l, e_l, w_l)
                                
                                if exercise == "Bicep Curl":
                                    if abs(e_l[0] - s_l[0]) > (0.20 * torso_length):
                                        fb_l = "ERROR: Keep Elbows Fixed!"
                                    perc_l = np.interp(angle_l, (55, 150), (100, 0))
                                    if angle_l > 140: stage_l = "down"
                                    if angle_l < 65 and stage_l == "down":
                                        stage_l = "up"
                                        if fb_l == "Perfect Form":
                                            # Adds a full +1 unless the right arm just counted a fraction of a second ago
                                            if (timestamp_ms - last_rep_ms_r) > 500:
                                                counter += 1
                                            last_rep_ms_l = timestamp_ms
                                        
                                elif exercise == "Shoulder Press":
                                    perc_l = np.interp(angle_l, (90, 150), (0, 100))
                                    if angle_l < 95: stage_l = "down"
                                    if angle_l > 140 and stage_l == "down":
                                        stage_l = "up"
                                        if fb_l == "Perfect Form":
                                            if (timestamp_ms - last_rep_ms_r) > 500:
                                                counter += 1
                                            last_rep_ms_l = timestamp_ms

                            # --- RIGHT ARM PROCESSING ---
                            if vis_r:
                                s_r, e_r, w_r = get_px(12), get_px(14), get_px(16)
                                angle_r = calculate_angle(s_r, e_r, w_r)
                                
                                if exercise == "Bicep Curl":
                                    if abs(e_r[0] - s_r[0]) > (0.20 * torso_length):
                                        fb_r = "ERROR: Keep Elbows Fixed!"
                                    perc_r = np.interp(angle_r, (55, 150), (100, 0))
                                    if angle_r > 140: stage_r = "down"
                                    if angle_r < 65 and stage_r == "down":
                                        stage_r = "up"
                                        if fb_r == "Perfect Form":
                                            # Adds a full +1 unless the left arm just counted a fraction of a second ago
                                            if (timestamp_ms - last_rep_ms_l) > 500:
                                                counter += 1
                                            last_rep_ms_r = timestamp_ms
                                        
                                elif exercise == "Shoulder Press":
                                    perc_r = np.interp(angle_r, (90, 150), (0, 100))
                                    if angle_r < 95: stage_r = "down"
                                    if angle_r > 140 and stage_r == "down":
                                        stage_r = "up"
                                        if fb_r == "Perfect Form":
                                            if (timestamp_ms - last_rep_ms_l) > 500:
                                                counter += 1
                                            last_rep_ms_r = timestamp_ms

                            if not vis_l and not vis_r:
                                feedback, feedback_color = "Adjust Camera: Arms Not Visible", (0, 165, 255)
                            elif fb_l != "Perfect Form" or fb_r != "Perfect Form":
                                feedback, feedback_color = "ERROR: Keep Elbows Fixed!", (0, 0, 255)
                            else:
                                feedback, feedback_color = "Perfect Form", (0, 255, 0)
                                
                            percentage = max(perc_l, perc_r)

                        # --- FULL BODY TRACKING (Squat & Push-up) ---
                        elif exercise == "Squat":
                            if lm[11].visibility > 0.6 and lm[23].visibility > 0.6 and lm[25].visibility > 0.6 and lm[27].visibility > 0.6:
                                knee, ankle = get_px(25), get_px(27)
                                angle = calculate_angle(hp_l, knee, ankle)
                                
                                if abs(sh_l[0] - knee[0]) > (0.35 * torso_length):
                                    feedback, feedback_color = "ERROR: Leaning Too Far Forward!", (0, 0, 255)
                                else:
                                    feedback, feedback_color = "Perfect Form", (0, 255, 0)
                                    
                                percentage = np.interp(angle, (105, 160), (100, 0))
                                
                                if angle > 150: stage_main = "up"
                                if angle < 115 and stage_main == "up":
                                    stage_main = "down"
                                    if feedback == "Perfect Form": counter += 1
                            else:
                                feedback, feedback_color = "Adjust Camera: Full Body Not Visible", (0, 165, 255)
                                    
                        elif exercise == "Push-up":
                            if lm[11].visibility > 0.6 and lm[13].visibility > 0.6 and lm[15].visibility > 0.6 and lm[23].visibility > 0.6:
                                elbow, wrist = get_px(13), get_px(15)
                                angle = calculate_angle(sh_l, elbow, wrist)
                                
                                if hp_l[1] > sh_l[1] + (0.25 * torso_length):
                                    feedback, feedback_color = "ERROR: Keep Back Straight!", (0, 0, 255)
                                else:
                                    feedback, feedback_color = "Perfect Form", (0, 255, 0)
                                    
                                percentage = np.interp(angle, (95, 150), (100, 0))
                                
                                if angle > 140: stage_main = "up"
                                if angle < 100 and stage_main == "up":
                                    stage_main = "down"
                                    if feedback == "Perfect Form": counter += 1
                            else:
                                feedback, feedback_color = "Adjust Camera: Joints Not Visible", (0, 165, 255)
                    except Exception as e:
                        pass
                    
                    draw_full_body_mesh(frame, lm)
                    
                    # Top Dashboard
                    draw_transparent_overlay(frame, 0, 0, w, 80)
                    cv2.putText(frame, f"MODE: {exercise.upper()}", (20, 30), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(frame, "REPS", (20, 60), cv2.FONT_HERSHEY_DUPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA)
                    # Forcing integer conversion to guarantee no decimals are displayed
                    cv2.putText(frame, str(int(counter)), (90, 65), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 0), 2, cv2.LINE_AA)
                    
                    # Bottom Dashboard
                    draw_transparent_overlay(frame, 0, h-60, w, 60)
                    cv2.putText(frame, f"STATUS: {feedback}", (20, h-20), cv2.FONT_HERSHEY_DUPLEX, 0.8, feedback_color, 2, cv2.LINE_AA)
                    
                    bar_height = int(h * 0.4)               
                    y_top = (h - bar_height) // 2           
                    y_bottom = y_top + bar_height
                    x_left = 20                             
                    x_right = 45
                    
                    bar_val = int(np.interp(percentage, (0, 100), (y_bottom, y_top)))
                    
                    cv2.rectangle(frame, (x_left, y_top), (x_right, y_bottom), (50, 50, 50), -1) 
                    cv2.rectangle(frame, (x_left, bar_val), (x_right, y_bottom), (0, 255, 0) if feedback == "Perfect Form" else (0, 0, 255), -1)
                    cv2.rectangle(frame, (x_left, y_top), (x_right, y_bottom), (255, 255, 255), 2)
                    cv2.putText(frame, f'{int(percentage)}%', (x_left - 5, y_top - 15), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    
                frame_placeholder.image(frame, channels="BGR", use_container_width=False)
                
                # FPS Pacer
                elapsed_time = time.time() - loop_start
                time_to_wait = frame_duration - elapsed_time
                if time_to_wait > 0:
                    time.sleep(time_to_wait)
                
            cap.release()