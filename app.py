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

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    return 360 - angle if angle > 180.0 else angle

def calculate_distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

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
    st.caption("AI powered by Google Gemini & MediaPipe.")

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
            
            timestamp_ms = 0
            last_rep_ms_l = -5000
            last_rep_ms_r = -5000
            
            with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
                frame_count = 0
                while cap.isOpened() and run_tracking:
                    loop_start = time.time()
                    
                    ret, frame = cap.read()
                    if not ret: 
                        break
                    
                    frame_count += 1
                    h, w, _ = frame.shape
                    
                    # OPTIMIZED: Clamped to 480p for lightning-fast WebSocket streaming
                    max_height = 480
                    max_width = 800
                    scale = min(max_height / h, max_width / w)
                    
                    if scale < 1.0: 
                        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
                        h, w, _ = frame.shape 
                    
                    timestamp_ms += int(frame_duration * 1000)
                    
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    rgb_frame.flags.writeable = False
                    results = pose.process(rgb_frame)
                    rgb_frame.flags.writeable = True
                    
                    percentage = 0
                    
                    if results.pose_landmarks:
                        lm = results.pose_landmarks.landmark
                        try:
                            get_px = lambda i: [lm[i].x * w, lm[i].y * h]
                            sh_l, hp_l = get_px(11), get_px(23)
                            torso_length = calculate_distance(sh_l, hp_l)
                            
                            if exercise in ["Bicep Curl", "Shoulder Press"]:
                                vis_l = lm[11].visibility > 0.6 and lm[13].visibility > 0.6 and lm[15].visibility > 0.6
                                vis_r = lm[12].visibility > 0.6 and lm[14].visibility > 0.6 and lm[16].visibility > 0.6
                                
                                perc_l, perc_r = 0, 0
                                fb_l, fb_r = "Perfect Form", "Perfect Form"
                                
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
                        
                        mp_drawing.draw_landmarks(
                            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=2),
                            mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2, circle_radius=2)
                        )
                        
                        draw_transparent_overlay(frame, 0, 0, w, 80)
                        cv2.putText(frame, f"MODE: {exercise.upper()}", (20, 30), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                        cv2.putText(frame, "REPS", (20, 55), cv2.FONT_HERSHEY_DUPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
                        cv2.putText(frame, str(int(counter)), (80, 58), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 0), 2, cv2.LINE_AA)
                        
                        draw_transparent_overlay(frame, 0, h-45, w, 45)
                        cv2.putText(frame, f"STATUS: {feedback}", (15, h-15), cv2.FONT_HERSHEY_DUPLEX, 0.6, feedback_color, 2, cv2.LINE_AA)
                        
                        bar_height = int(h * 0.4)               
                        y_top = (h - bar_height) // 2           
                        y_bottom = y_top + bar_height
                        x_left = 15                             
                        x_right = 35
                        
                        bar_val = int(np.interp(percentage, (0, 100), (y_bottom, y_top)))
                        
                        cv2.rectangle(frame, (x_left, y_top), (x_right, y_bottom), (50, 50, 50), -1) 
                        cv2.rectangle(frame, (x_left, bar_val), (x_right, y_bottom), (0, 255, 0) if feedback == "Perfect Form" else (0, 0, 255), -1)
                        cv2.rectangle(frame, (x_left, y_top), (x_right, y_bottom), (255, 255, 255), 2)
                        cv2.putText(frame, f'{int(percentage)}%', (x_left - 5, y_top - 10), cv2.FONT_HERSHEY_DUPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                        
                    frame_placeholder.image(frame, channels="BGR", use_container_width=False)
                    
                    elapsed_time = time.time() - loop_start
                    time_to_wait = frame_duration - elapsed_time
                    if time_to_wait > 0:
                        time.sleep(time_to_wait)
                    
                cap.release()
