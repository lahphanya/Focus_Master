import cv2
import math
import numpy as np
from ultralytics import YOLO
import mediapipe as mp 

class FocusAnalyzer:
    def __init__(self):
        self.yolo_model = YOLO('yolov8n.pt') 
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        self.hands = mp.solutions.hands.Hands(
            max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        self.prev_hand_x = None
        self.prev_hand_y = None
        
        self.frame_count = 0
        self.cached_classes = []
        self.cached_plot = None

    def analyze_frame(self, frame, allowed_device="없음", head_down_thresh=0.70, head_up_thresh=0.45, yaw_thresh=0.40, hand_threshold=0.02, yaw_normal=0.5, draw_debug=False):
        self.frame_count += 1
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        result = {
            "status": "집중중",
            "curr_pitch": 0.5,
            "curr_yaw": 0.5,
            "curr_hand": 0.0,
            "face_detected": False,
            "objects": [],
            "annotated_frame": frame.copy() 
        }

        if self.frame_count % 3 == 1 or draw_debug:
            yolo_results = self.yolo_model(frame, stream=True, verbose=False) 
            detected_classes = []
            
            for r in yolo_results:
                if draw_debug:
                    result["annotated_frame"] = r.plot() 
                    self.cached_plot = result["annotated_frame"].copy()
                    
                for box in r.boxes:
                    cls_name = self.yolo_model.names[int(box.cls[0])]
                    conf = float(box.conf[0])
                    
                    if cls_name == 'cell phone' and conf < 0.5:
                        continue
                    detected_classes.append(cls_name)
                    
            self.cached_classes = detected_classes
        else:
            detected_classes = self.cached_classes
            if draw_debug and self.cached_plot is not None:
                result["annotated_frame"] = self.cached_plot.copy()

        result["objects"] = list(set(detected_classes))
        
        is_phone = 'cell phone' in detected_classes
        is_tablet = 'laptop' in detected_classes or 'tv' in detected_classes

        if allowed_device == "없음" and (is_phone or is_tablet):
            result["status"] = "전자기기"
            return result
        elif allowed_device == "스마트폰" and is_tablet and not is_phone:
            result["status"] = "태블릿"
            return result
        elif allowed_device == "태블릿" and is_phone:
            result["status"] = "스마트폰"
            return result

        face_results = self.face_mesh.process(rgb_frame)
        hand_results = self.hands.process(rgb_frame)
        
        is_leaning_back = False
        is_face_down = False
        is_looking_away = False
        is_writing = False

        if not face_results.multi_face_landmarks:
            result["status"] = "자리비움"
            return result

        if face_results.multi_face_landmarks:
            result["face_detected"] = True 
            for face_landmarks in face_results.multi_face_landmarks:
                
                forehead_y = face_landmarks.landmark[10].y
                chin_y = face_landmarks.landmark[152].y
                nose_y = face_landmarks.landmark[1].y
                
                pitch = (nose_y - forehead_y) / (chin_y - forehead_y + 1e-6)
                result["curr_pitch"] = pitch
                
                if pitch <= head_up_thresh:
                    is_leaning_back = True
                elif pitch >= head_down_thresh:
                    is_face_down = True
                    
                left_x = face_landmarks.landmark[234].x
                right_x = face_landmarks.landmark[454].x
                nose_x = face_landmarks.landmark[1].x
                
                yaw = (nose_x - left_x) / (right_x - left_x + 1e-6)
                result["curr_yaw"] = yaw
                
                if abs(yaw - yaw_normal) > yaw_thresh:
                    is_looking_away = True
                
                if draw_debug:
                    mp.solutions.drawing_utils.draw_landmarks(
                        image=result["annotated_frame"],
                        landmark_list=face_landmarks,
                        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style()
                    )

        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                curr_x = hand_landmarks.landmark[8].x
                curr_y = hand_landmarks.landmark[8].y
                
                if self.prev_hand_x is not None and self.prev_hand_y is not None:
                    distance = math.sqrt((curr_x - self.prev_hand_x)**2 + (curr_y - self.prev_hand_y)**2)
                    result["curr_hand"] = distance
                    if distance > hand_threshold: 
                        is_writing = True
                
                self.prev_hand_x = curr_x
                self.prev_hand_y = curr_y

                if draw_debug:
                    mp.solutions.drawing_utils.draw_landmarks(
                        result["annotated_frame"], hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
        else:
            result["curr_hand"] = 0.0
            self.prev_hand_x, self.prev_hand_y = None, None

        if is_looking_away:
            result["status"] = "고개 돌림"
        elif is_leaning_back:
            result["status"] = "누움/뒤척임"
        elif is_face_down:
            if is_writing: 
                result["status"] = "필기중"
            else: 
                result["status"] = "졸음"
        else:
            result["status"] = "집중중"

        return result