import streamlit as st
import cv2
import time
import numpy as np
import base64
import pandas as pd
import os
from datetime import datetime

import vision_analyzer as va
import database_manager as db

db.init_db()

def get_base64_audio(file_path):
    try:
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except:
        return ""

siren_b64 = get_base64_audio("siren.mp3")
stop_audio_script = '<script>var audio = parent.document.getElementById("siren") || document.getElementById("siren"); if(audio) { audio.pause(); audio.currentTime = 0; }</script>'

def init_session_state():
    defaults = {
        'is_running': False, 'current_mode': "대기중", 'pure_study_time': 0.0,
        'head_down_thresh': 0.75, 'head_up_thresh': 0.45, 'yaw_thresh': 0.40, 'hand_threshold': 0.008, 
        'allowed_device': "없음", 'current_session_id': None, 'prev_status': "집중중", 
        'last_frame_time': None, 'study_min': 50, 'break_min': 10, 'total_cycles': 3,
        'current_cycle': 1, 'phase_start_time': None, 'is_meal_time': False,
        'realtime_events': [], 'yaw_normal': 0.5, 
        'status_lock_until': 0.0, 'locked_status': "집중중",
        'is_calibrating': False, 'cal_start_time': 0.0, 'cal_yaw_data': [], 'cal_hand_data': [],
        # 💡 연속 졸음 감지를 위한 전용 타이머 추가!
        'sleep_start_time': 0.0 
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session_state()
st.set_page_config(page_title="집중력 마스터", layout="wide")

@st.cache_resource
def get_analyzer():
    return va.FocusAnalyzer()

@st.cache_resource
def get_camera():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap

def render_schedule_graph(elapsed_sec=0):
    html = '<div style="display: flex; width: 100%; height: 30px; border-radius: 5px; overflow: hidden; margin-bottom: 5px;">'
    if st.session_state.is_meal_time:
        percent = min(100, int((elapsed_sec / 3600) * 100))
        bg = f"linear-gradient(to right, #007BFF {percent}%, #FFD700 {percent}%)"
        html += f'<div style="flex: 1; background: {bg}; color: black; text-align: center; font-size: 14px; font-weight: bold; line-height: 30px;">식사 (1시간) - {percent}% 진행</div>'
    else:
        for i in range(st.session_state.current_cycle, st.session_state.total_cycles + 1):
            if st.session_state.current_mode == "공부중" and i == st.session_state.current_cycle:
                percent = min(100, int((elapsed_sec / (st.session_state.study_min * 60)) * 100))
                bg = f"linear-gradient(to right, #007BFF {percent}%, #FF4B4B {percent}%)"
                html += f'<div style="flex: {st.session_state.study_min}; background: {bg}; color: white; text-align: center; font-size: 12px; font-weight: bold; line-height: 30px; border-right: 1px solid white;">공부(진행중)</div>'
            else:
                html += f'<div style="flex: {st.session_state.study_min}; background-color: #ff9999; color: white; text-align: center; font-size: 12px; line-height: 30px; border-right: 1px solid white;">공부</div>'
            if i < st.session_state.total_cycles:
                if st.session_state.current_mode == "쉬는시간" and i == st.session_state.current_cycle:
                    percent = min(100, int((elapsed_sec / (st.session_state.break_min * 60)) * 100))
                    bg = f"linear-gradient(to right, #007BFF {percent}%, #28a745 {percent}%)"
                    html += f'<div style="flex: {st.session_state.break_min}; background: {bg}; color: white; text-align: center; font-size: 12px; font-weight: bold; line-height: 30px; border-right: 1px solid white;">휴식(진행중)</div>'
                else:
                    html += f'<div style="flex: {st.session_state.break_min}; background-color: #85e09b; color: white; text-align: center; font-size: 12px; line-height: 30px; border-right: 1px solid white;">휴식</div>'
    html += '</div>'
    return html

with st.sidebar:
    st.title("시스템 설정")
    menu = st.radio("메뉴 이동", ["실시간 모니터링", "통계 및 분석"])
    st.divider()
    
    if menu == "실시간 모니터링":
        st.subheader("학습 및 스케줄 설정")
        st.session_state.allowed_device = st.selectbox("허용 전자기기", ["없음", "태블릿", "스마트폰"])
        st.session_state.study_min = st.number_input("공부 시간 (분)", min_value=1, value=50)
        st.session_state.break_min = st.number_input("쉬는 시간 (분)", min_value=1, value=10)
        st.session_state.total_cycles = st.number_input("반복 횟수 (텀)", min_value=1, value=3)
        
        with st.expander("자세 판별 정밀 감도 조절"):
            st.caption("상하 각도는 절대값 기준이며, 좌우 각도는 정면 대비 흔들림 기준입니다.")
            st.session_state.head_down_thresh = st.slider("고개 숙임 기준선", 0.50, 1.00, st.session_state.head_down_thresh, 0.01)
            st.session_state.head_up_thresh = st.slider("고개 들림 기준선", 0.20, 0.50, st.session_state.head_up_thresh, 0.01)
            st.session_state.yaw_thresh = st.slider("↔고개 돌림 감도", 0.10, 0.80, st.session_state.yaw_thresh, 0.01)

        st.divider()
        col1, col2 = st.columns(2)
        if col1.button("▶️ 시작", type="primary", use_container_width=True):
            st.session_state.is_running = True
            st.session_state.is_calibrating = False 
            st.session_state.current_mode = "공부중"
            st.session_state.current_cycle = 1
            st.session_state.is_meal_time = False
            st.session_state.phase_start_time = time.time()
            st.session_state.last_frame_time = time.time()
            st.session_state.prev_status = "집중중"
            st.session_state.realtime_events = [] 
            st.session_state.status_lock_until = 0.0 
            st.session_state.current_session_id = db.start_session()
            
        if col2.button("⏹️ 종료", use_container_width=True):
            st.session_state.is_running = False
            st.session_state.is_calibrating = False
            st.session_state.current_mode = "대기중"
            if st.session_state.current_session_id:
                db.end_session(st.session_state.current_session_id, int(st.session_state.pure_study_time))
                st.session_state.current_session_id = None
                
        if st.button("식사 (1시간 휴식)", use_container_width=True):
            if st.session_state.is_running:
                st.session_state.is_meal_time = True
                st.session_state.current_mode = "식사시간"
                st.session_state.phase_start_time = time.time()

if menu == "실시간 모니터링":
    st.title("실시간 집중력 모니터링")
    
    with st.expander("자세 영점 조절"):
        st.info("시작 전 카메라의 '정면'을 응시하며 기준점을 세팅하세요.")
        if st.button("영점 조절 시작 (5초)", use_container_width=True):
            st.session_state.is_running = False 
            st.session_state.is_calibrating = True
            st.session_state.cal_start_time = time.time()
            st.session_state.cal_yaw_data = []
            st.session_state.cal_hand_data = []
    
    st.caption(f"현재 스케줄 (총 {st.session_state.total_cycles}텀 중 {st.session_state.current_cycle}텀 진행중)")
    schedule_placeholder = st.empty()
    phase_time_placeholder = st.empty() 
    st.divider()
    
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1: m1_placeholder = st.empty()
    with col_m2: m2_placeholder = st.empty()
    
    show_debug_info = st.checkbox("실시간 인식 객체 및 수치 텍스트 표시", value=True)
    st.divider()
    
    col_cam, col_data = st.columns([7, 3])
    with col_cam:
        frame_placeholder = st.empty()
        warning_placeholder = st.empty() 

    with col_data:
        st.subheader("실시간 수치 및 알림")
        alarm_volume = st.slider("사이렌 볼륨", 0.0, 1.0, 0.5) 
        status_box = st.empty()
        head_box = st.empty()
        yaw_box = st.empty()
        hand_box = st.empty()
        audio_placeholder = st.empty() 
        
        st.divider()
        st.subheader("실시간 딴짓 분석 (5분 단위)")
        realtime_table_placeholder = st.empty()

    cap = get_camera()
    analyzer = get_analyzer() 
    
    last_ui_update = 0.0
    UI_UPDATE_INTERVAL = 0.2

    while True:
        ret, frame = cap.read()
        
        if not ret:
            if os.path.exists("no_cam.jpg"):
                frame_placeholder.image(cv2.cvtColor(cv2.imread("no_cam.jpg"), cv2.COLOR_BGR2RGB), use_container_width=True)
            else:
                no_cam_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(no_cam_rgb, "NO CAMERA", (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
                frame_placeholder.image(no_cam_rgb, channels="RGB", use_container_width=True)
            warning_placeholder.error("캠 연결이 끊어졌습니다. 카메라 상태를 확인해주세요.")
            break
            
        frame = cv2.flip(frame, 1) 
        warning_placeholder.empty()
        current_time = time.time()

        if st.session_state.is_calibrating:
            elapsed = current_time - st.session_state.cal_start_time
            display_frame = frame.copy()
            
            if elapsed < 3.0:
                countdown = int(4 - elapsed)
                cv2.putText(display_frame, f"Ready: {countdown}", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)
            elif elapsed < 8.0:
                cv2.putText(display_frame, "Calibrating Yaw & Hands...", (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                result = analyzer.analyze_frame(frame, draw_debug=True)
                
                if result.get('face_detected', False):
                    st.session_state.cal_yaw_data.append(result['curr_yaw'])
                if result['curr_hand'] > 0:
                    st.session_state.cal_hand_data.append(result['curr_hand'])
                display_frame = result['annotated_frame']
            else:
                msg = []
                if len(st.session_state.cal_yaw_data) > 5:
                    st.session_state.yaw_normal = np.mean(st.session_state.cal_yaw_data)
                    msg.append(f"좌우영점({st.session_state.yaw_normal:.3f})")
                if len(st.session_state.cal_hand_data) > 5:
                    st.session_state.hand_threshold = np.mean(st.session_state.cal_hand_data) * 0.5
                    msg.append(f"손({st.session_state.hand_threshold:.3f})")
                
                st.session_state.is_calibrating = False
                if msg:
                    st.toast(f"영점 조절 완료. " + ", ".join(msg))
                else:
                    st.toast("인식 실패. 카메라에 얼굴이 보이게 다시 시도해주세요.")
                
                time.sleep(2) 
                st.rerun() 

            frame_placeholder.image(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            time.sleep(0.03)
            continue 

        if not st.session_state.is_running:
            if current_time - last_ui_update > UI_UPDATE_INTERVAL:
                schedule_placeholder.markdown(render_schedule_graph(0), unsafe_allow_html=True)
                phase_time_placeholder.info("시작 버튼을 누르면 스케줄이 진행됩니다. 화면을 보고 자세를 세팅하세요.")
                m1_placeholder.metric("현재 상태", "대기중")
                m, s = divmod(int(st.session_state.pure_study_time), 60)
                m2_placeholder.metric("순공부시간", f"{m//60:02d}:{m%60:02d}:{s:02d}")
                last_ui_update = current_time
            
            display_frame = frame.copy()
            cv2.putText(display_frame, "STANDBY", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 3)
            frame_placeholder.image(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            time.sleep(0.05) 
            continue 

        elapsed_phase_sec = current_time - st.session_state.phase_start_time
        
        if st.session_state.is_meal_time:
            if elapsed_phase_sec >= 3600:
                st.session_state.is_meal_time = False
                st.session_state.current_mode = "공부중"
                st.session_state.phase_start_time = current_time
        elif st.session_state.current_mode == "공부중":
            if elapsed_phase_sec >= (st.session_state.study_min * 60):
                if st.session_state.current_cycle < st.session_state.total_cycles:
                    st.session_state.current_mode = "쉬는시간"
                    st.session_state.phase_start_time = current_time
                else:
                    st.session_state.current_mode = "완료"
                    st.session_state.is_running = False
                    st.success("모든 학습 스케줄이 완료되었습니다.")
                    break
        elif st.session_state.current_mode == "쉬는시간":
            if elapsed_phase_sec >= (st.session_state.break_min * 60):
                st.session_state.current_mode = "공부중"
                st.session_state.current_cycle += 1
                st.session_state.phase_start_time = current_time

        if st.session_state.current_mode == "공부중":
            result = analyzer.analyze_frame(
                frame=frame, allowed_device=st.session_state.allowed_device,
                head_down_thresh=st.session_state.head_down_thresh, 
                head_up_thresh=st.session_state.head_up_thresh,
                yaw_thresh=st.session_state.yaw_thresh,
                hand_threshold=st.session_state.hand_threshold,
                yaw_normal=st.session_state.yaw_normal, draw_debug=show_debug_info 
            )
            
            if result['status'] == "졸음":
                if st.session_state.sleep_start_time == 0.0:
                    st.session_state.sleep_start_time = current_time
                    result['status'] = "필기중" 
                else:
                    if current_time - st.session_state.sleep_start_time < 5.0:
                        result['status'] = "필기중"
            else:
                st.session_state.sleep_start_time = 0.0
            
            if current_time < st.session_state.status_lock_until:
                result['status'] = st.session_state.locked_status
            else:
                if result['status'] not in ["집중중", "필기중"]:
                    st.session_state.locked_status = result['status']
                    st.session_state.status_lock_until = current_time + 3.0
            
            display_frame = result["annotated_frame"]
            
            if result['status'] not in ["집중중", "필기중"]:
                red_overlay = np.full(display_frame.shape, (0, 0, 255), dtype=np.uint8)
                display_frame = cv2.addWeighted(display_frame, 0.7, red_overlay, 0.3, 0)
                if st.session_state.prev_status in ["집중중", "필기중"] and siren_b64:
                    audio_html = f'<audio id="siren" autoplay loop><source src="data:audio/mp3;base64,{siren_b64}" type="audio/mp3"></audio><script>document.getElementById("siren").volume = {alarm_volume};</script>'
                    audio_placeholder.markdown(audio_html, unsafe_allow_html=True)
            else:
                if st.session_state.prev_status not in ["집중중", "필기중"]:
                    audio_placeholder.markdown(stop_audio_script, unsafe_allow_html=True)

            dt = current_time - st.session_state.last_frame_time 
            if result['status'] in ["필기중", "집중중"]:
                st.session_state.pure_study_time += dt 
            else:
                if st.session_state.prev_status in ["필기중", "집중중"]:
                    elapsed_study_min = int(elapsed_phase_sec // 60)
                    db.log_event(st.session_state.current_session_id, result['status'], elapsed_study_min)
                    bucket_start = (elapsed_study_min // 5) * 5
                    st.session_state.realtime_events.append({"시간대": f"{bucket_start} ~ {bucket_start + 5}분", "사유": result['status'], "카운트": 1})
            
            st.session_state.prev_status = result['status']
            
            if current_time - last_ui_update > UI_UPDATE_INTERVAL:
                schedule_placeholder.markdown(render_schedule_graph(elapsed_phase_sec), unsafe_allow_html=True)
                
                rem_sec = max(0, (st.session_state.study_min * 60) - elapsed_phase_sec)
                phase_time_placeholder.info(f"**이번 텀 공부 종료까지:** {int(rem_sec)//60:02d}분 {int(rem_sec)%60:02d}초 남았습니다.")
                
                status_box.info(f"**현재 판별:** {result['status']}")
                
                head_box.metric("상하(Pitch) - 절대값", f"{result['curr_pitch']:.3f}", delta=f"정상범위: {st.session_state.head_up_thresh} ~ {st.session_state.head_down_thresh}", delta_color="off")
                yaw_box.metric("좌우(Yaw) - 회전", f"{result['curr_yaw']:.3f}", delta=f"허용오차: ±{st.session_state.yaw_thresh:.2f}", delta_color="off")
                hand_box.metric("손 움직임", f"{result['curr_hand']:.3f}", delta=f"임계점: {st.session_state.hand_threshold:.3f}", delta_color="off")
                
                status_text = f"공부중 ({int(rem_sec)//60:02d}:{int(rem_sec)%60:02d})"
                m1_placeholder.metric("현재 상태", status_text)
                
                m, s = divmod(int(st.session_state.pure_study_time), 60)
                m2_placeholder.metric("순공부시간", f"{m//60:02d}:{m%60:02d}:{s:02d}")
                
                if st.session_state.realtime_events:
                    df_realtime = pd.DataFrame(st.session_state.realtime_events)
                    df_grouped = df_realtime.groupby(['시간대', '사유'])['카운트'].sum().reset_index()
                    realtime_table_placeholder.dataframe(df_grouped, hide_index=True, use_container_width=True)
                else:
                    realtime_table_placeholder.info("이번 텀에 아직 딴짓 기록이 없습니다.")
                    
                last_ui_update = current_time

        else:
            display_frame = frame
            
            if st.session_state.prev_status not in ["집중중", "필기중"]:
                audio_placeholder.markdown(stop_audio_script, unsafe_allow_html=True)
                st.session_state.prev_status = "집중중" 
            
            if current_time - last_ui_update > UI_UPDATE_INTERVAL:
                schedule_placeholder.markdown(render_schedule_graph(elapsed_phase_sec), unsafe_allow_html=True)
                
                status_text = st.session_state.current_mode
                if st.session_state.current_mode == "쉬는시간":
                    rem_sec = max(0, (st.session_state.break_min * 60) - elapsed_phase_sec)
                    phase_time_placeholder.success(f"**쉬는 시간 종료까지:** {int(rem_sec)//60:02d}분 {int(rem_sec)%60:02d}초 남았습니다.")
                    status_text = f"쉬는시간 ({int(rem_sec)//60:02d}:{int(rem_sec)%60:02d})"
                elif st.session_state.current_mode == "식사시간":
                    rem_sec = max(0, 3600 - elapsed_phase_sec)
                    phase_time_placeholder.warning(f"**식사 종료까지:** {int(rem_sec)//60:02d}분 {int(rem_sec)%60:02d}초 남았습니다.")
                    status_text = f"식사시간 ({int(rem_sec)//60:02d}:{int(rem_sec)%60:02d})"
                
                status_box.success(f"현재는 {st.session_state.current_mode}입니다.")
                m1_placeholder.metric("현재 상태", status_text)
                
                m, s = divmod(int(st.session_state.pure_study_time), 60)
                m2_placeholder.metric("순공부시간", f"{m//60:02d}:{m%60:02d}:{s:02d}")
                last_ui_update = current_time

        st.session_state.last_frame_time = current_time
        
        frame_placeholder.image(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
        time.sleep(0.03) 
        
elif menu == "통계 및 분석":
    db.render_dashboard()