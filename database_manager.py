import sqlite3
import pandas as pd
import streamlit as st
import altair as alt
from datetime import datetime

DB_NAME = "focus_master.db"

def init_db(): 
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS study_sessions (id INTEGER PRIMARY KEY AUTOINCREMENT, date TEXT, start_time TEXT, end_time TEXT, pure_study_sec INTEGER)''')
    
    cursor.execute('''CREATE TABLE IF NOT EXISTS distraction_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT, 
        session_id INTEGER, 
        timestamp TEXT, 
        event_type TEXT, 
        elapsed_min INTEGER, 
        FOREIGN KEY(session_id) REFERENCES study_sessions(id))''')
    conn.commit()
    conn.close()

def start_session(): 
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    date_str = datetime.now().strftime("%Y-%m-%d")
    time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('INSERT INTO study_sessions (date, start_time, pure_study_sec) VALUES (?, ?, 0)', (date_str, time_str))
    session_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return session_id

def end_session(session_id, pure_study_sec): 
    if session_id is None: return
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    end_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('UPDATE study_sessions SET end_time = ?, pure_study_sec = ? WHERE id = ?', (end_time_str, pure_study_sec, session_id))
    conn.commit()
    conn.close()

def log_event(session_id, event_type, elapsed_min): 
    if session_id is None: return
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('INSERT INTO distraction_events (session_id, timestamp, event_type, elapsed_min) VALUES (?, ?, ?, ?)', (session_id, timestamp, event_type, elapsed_min))
    conn.commit()
    conn.close()

def render_dashboard():
    st.header("📊 학습 패턴 세부 분석")
    conn = sqlite3.connect(DB_NAME)
    try:
        df_sessions = pd.read_sql_query("SELECT * FROM study_sessions", conn)
        df_events = pd.read_sql_query("SELECT * FROM distraction_events", conn)
    except:
        st.error("데이터베이스 오류. 기존 DB 파일을 삭제하고 다시 실행해주세요.")
        return
    finally:
        conn.close()

    if df_sessions.empty:
        st.info("아직 저장된 데이터가 없습니다.")
        return

    df_sessions['date'] = pd.to_datetime(df_sessions['date'])
    df_sessions['year'] = df_sessions['date'].dt.year
    df_sessions['month'] = df_sessions['date'].dt.month
    df_sessions['day'] = df_sessions['date'].dt.day

    view_type = st.radio("보기 기준", ["연도별", "월별", "날짜별"], horizontal=True)
    
    col1, col2, col3 = st.columns(3)
    years = df_sessions['year'].unique().tolist()
    selected_year = col1.selectbox("조회 연도", years, index=len(years)-1)
    
    target_sessions = df_sessions[df_sessions['year'] == selected_year]
    selected_month, selected_day = None, None

    if view_type in ["월별", "날짜별"]:
        months = target_sessions['month'].unique().tolist()
        months.sort()
        selected_month = col2.selectbox("조회 월", months, index=len(months)-1 if months else 0)
        target_sessions = target_sessions[target_sessions['month'] == selected_month]

    if view_type == "날짜별":
        days = target_sessions['day'].unique().tolist()
        days.sort()
        selected_day = col3.selectbox("일", days, index=len(days)-1 if days else 0)
        target_sessions = target_sessions[target_sessions['day'] == selected_day]

    st.divider()
    m1, m2 = st.columns(2)
    
    days_count = target_sessions['date'].nunique() if not target_sessions.empty else 1
    if days_count == 0: days_count = 1

    if not target_sessions.empty:
        total_sec = target_sessions['pure_study_sec'].sum()
        avg_sec = total_sec / days_count
        m1.metric("해당 기간 일일 평균 순공부", format_time(avg_sec))
        m2.metric("해당 기간 총 순공부시간", format_time(total_sec))
    else:
        st.warning("선택한 기간의 데이터가 없습니다.")
        return

    st.divider()

    merged_df = pd.merge(df_events, df_sessions[['id', 'year', 'month', 'day', 'start_time', 'end_time']], left_on='session_id', right_on='id')
    target_events = merged_df[merged_df['year'] == selected_year]
    if selected_month: target_events = target_events[target_events['month'] == selected_month]
    if selected_day: target_events = target_events[target_events['day'] == selected_day]

    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.subheader("주요 방해 요인 (총 횟수)")
        if not target_events.empty:
            event_counts = target_events['event_type'].value_counts().reset_index()
            event_counts.columns = ['방해요인', '횟수']
            
            # 1. 총합 막대 그래프 (빨간색)
            chart_total = alt.Chart(event_counts).mark_bar(color='#FF4B4B').encode(
                x=alt.X('방해요인', axis=alt.Axis(labelAngle=0)), 
                y='횟수', 
                tooltip=['방해요인', '횟수']
            ).properties(height=250)
            st.altair_chart(chart_total, use_container_width=True)
            
            # 2. 일평균 막대 그래프 추가 (연도별/월별 조회 시에만 표시)
            if view_type in ["연도별", "월별"]:
                st.subheader("주요 방해 요인 (일평균 발생 횟수)")
                event_counts['일평균'] = (event_counts['횟수'] / days_count).round(2)
                
                chart_avg = alt.Chart(event_counts).mark_bar(color='#4B8BFF').encode(
                    x=alt.X('방해요인', axis=alt.Axis(labelAngle=0)), 
                    y='일평균', 
                    tooltip=['방해요인', '일평균']
                ).properties(height=250)
                st.altair_chart(chart_avg, use_container_width=True)
                
        else:
            st.write("해당 기간에는 딴짓 기록이 없습니다")

    with col_chart2:
        st.subheader("취약 시간대 분석")
        if not target_sessions.empty and view_type == "날짜별":
            first_start = target_sessions.iloc[0]['start_time'][11:16]
            last_end = target_sessions.iloc[-1]['end_time']
            last_end_str = last_end[11:16] if pd.notna(last_end) else "진행중"
            st.caption(f"**이날의 학습 세션:** {first_start} ~ {last_end_str}")

        if not target_events.empty:
            target_events['hour'] = pd.to_datetime(target_events['timestamp']).dt.hour
            hour_counts = target_events['hour'].value_counts().sort_index().reindex(range(24), fill_value=0)
            st.line_chart(hour_counts)
            
    st.divider()

    st.subheader("텀 시작 후 경과 시간별 딴짓 빈도 (5분 단위)")
    if not target_events.empty and 'elapsed_min' in target_events.columns:
        target_events['bucket_start'] = (target_events['elapsed_min'] // 5) * 5
        target_events['시간대'] = target_events['bucket_start'].apply(lambda x: f"{int(x)} ~ {int(x)+5}분")
        
        interval_df = target_events.groupby(['시간대', 'event_type']).size().reset_index(name='카운트')
        interval_df.rename(columns={'event_type': '사유'}, inplace=True)
        
        interval_df['sort_key'] = interval_df['시간대'].apply(lambda x: int(x.split(' ')[0]))
        interval_df = interval_df.sort_values(by=['sort_key', '사유']).drop('sort_key', axis=1)
        
        st.dataframe(interval_df, hide_index=True, use_container_width=True)
    else:
        st.info("분석할 딴짓 데이터가 부족합니다.")

def format_time(seconds):
    if pd.isna(seconds): return "00:00:00"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"