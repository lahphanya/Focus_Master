import streamlit as st
import cv2
import time
import numpy as np
import math
import mediapipe as mp 

def run_calibration():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if not ret:
        st.error("카메라가 인식되지 않습니다. 카메라 연결을 확인해주세요.")
        cap.release()
        return 

    status_text = st.empty()
    countdown_text = st.empty()
    frame_placeholder = st.empty()
    progress_bar = st.empty()

    for i in range(3, 0, -1):
        countdown_text.markdown(f"<h1 style='text-align: center; color: #FF4B4B; font-size: 80px;'>{i}</h1>", unsafe_allow_html=True)
        status_text.info("영점 조절을 위해 허리를 펴고 모니터 정면을 응시하는 바른 자세를 취해주세요.")
        
        ret, frame = cap.read()
        if ret:
            frame = cv2.flip(frame, 1) 
            frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
        time.sleep(1)

    countdown_text.empty()
    status_text.success("데이터 수집 시작. 자세를 5초간 유지해주세요.")

    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(min_detection_confidence=0.5)
    mp_hands = mp.solutions.hands.Hands(min_detection_confidence=0.5)

    start_time = time.time()
    duration = 5.0 

    head_pitch_list = []
    hand_movements = []
    prev_x, prev_y = None, None

    my_bar = progress_bar.progress(0.0)

    while True:
        elapsed_time = time.time() - start_time
        if elapsed_time > duration:
            break
            
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1) 
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        face_results = mp_face_mesh.process(rgb_frame)
        hand_results = mp_hands.process(rgb_frame)

        if face_results.multi_face_landmarks:
            for face in face_results.multi_face_landmarks:
                forehead_y = face.landmark[10].y
                chin_y = face.landmark[152].y
                nose_y = face.landmark[1].y
                
                pitch = (nose_y - forehead_y) / (chin_y - forehead_y + 1e-6)
                head_pitch_list.append(pitch)

        if hand_results.multi_hand_landmarks:
            for hand in hand_results.multi_hand_landmarks:
                curr_x = hand.landmark[8].x
                curr_y = hand.landmark[8].y

                if prev_x is not None and prev_y is not None:
                    dist = math.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
                    hand_movements.append(dist)

                prev_x, prev_y = curr_x, curr_y

        frame_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)
        my_bar.progress(min(elapsed_time / duration, 1.0))

    cap.release()
    mp_face_mesh.close()
    mp_hands.close()
    
    frame_placeholder.empty()
    progress_bar.empty()

    success = False
    msg = []
    
    if len(head_pitch_list) > 10:
        st.session_state.head_normal = np.mean(head_pitch_list)
        msg.append(f"정면 기준점({st.session_state.head_normal:.3f})")
        success = True
        
    if len(hand_movements) > 10:
        st.session_state.hand_threshold = np.mean(hand_movements) * 0.5
        msg.append(f"손 임계점({st.session_state.hand_threshold:.3f})")
        success = True

    if success:
        status_text.success(f"영점 조절 완료. [" + ", ".join(msg) + "]")
        st.balloons()
    else:
        status_text.error("얼굴이나 손이 인식되지 않았습니다. 다시 시도해주세요.")