import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tempfile
import os
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PitcherFormAnalyzer:
    def __init__(self):
        """투수 폼 분석기 초기화"""
        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils
        
        # Mediapipe Pose 모델 설정
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,  # 정확도 향상을 위해 1로 변경
            smooth_landmarks=True,  # 랜드마크 스무딩 활성화
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # 관절 각도 임계값 설정
        self.angle_thresholds = {
            'shoulder': {'min': 80, 'max': 100},
            'elbow': {'min': 85, 'max': 95},
            'hip': {'min': 170, 'max': 180},
            'knee': {'min': 170, 'max': 180}
        }

    def calculate_angle(self, a: list, b: list, c: list) -> float:
        """
        세 점 사이의 각도를 계산
        
        Args:
            a, b, c: 각각 시작점, 중심점, 끝점의 좌표
        
        Returns:
            float: 계산된 각도
        """
        try:
            a = np.array(a)
            b = np.array(b)
            c = np.array(c)
            
            ba = a - b
            bc = c - b
            
            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
            
            return np.degrees(angle)
        except Exception as e:
            logger.error(f"각도 계산 중 오류 발생: {e}")
            return 0.0

    def analyze_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[str], Dict[str, float]]:
        """
        프레임에서 자세를 분석하고 피드백을 생성
        
        Args:
            frame: 분석할 비디오 프레임
            
        Returns:
            Tuple[np.ndarray, List[str], Dict[str, float]]: 
            처리된 이미지, 피드백 목록, 관절 각도 딕셔너리
        """
        try:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(image)
            
            feedback = []
            angles = {}
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # 주요 관절 좌표 추출
                joints = {
                    'shoulder': (
                        [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                         landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y],
                        [landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                         landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y],
                        [landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                         landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                    ),
                    'elbow': (
                        [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                         landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y],
                        [landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                         landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y],
                        [landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                         landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                    ),
                    'hip': (
                        [landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                         landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y],
                        [landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                         landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].y],
                        [landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                         landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                    )
                }
                
                # 각도 계산 및 피드백 생성
                for joint, (a, b, c) in joints.items():
                    angle = self.calculate_angle(a, b, c)
                    angles[joint] = angle
                    
                    if angle < self.angle_thresholds[joint]['min']:
                        feedback.append(f"{joint.capitalize()} 각도가 너무 작습니다 ({angle:.1f}°)")
                    elif angle > self.angle_thresholds[joint]['max']:
                        feedback.append(f"{joint.capitalize()} 각도가 너무 큽니다 ({angle:.1f}°)")
                
                # 랜드마크 시각화
                self.mp_draw.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_draw.DrawingSpec(
                        color=(245,117,66),
                        thickness=2,
                        circle_radius=2
                    ),
                    connection_drawing_spec=self.mp_draw.DrawingSpec(
                        color=(245,66,230),
                        thickness=2,
                        circle_radius=2
                    )
                )
            
            return image, feedback, angles
            
        except Exception as e:
            logger.error(f"프레임 분석 중 오류 발생: {e}")
            return frame, ["분석 중 오류가 발생했습니다."], {}

def main():
    st.set_page_config(
        page_title="⚾ 야구 투수 자세 분석기",
        page_icon="⚾",
        layout="wide"
    )
    
    st.title("⚾ 야구 투수 자세 분석기")
    st.write("영상을 업로드하여 투구 자세를 분석해보세요.")
    
    uploaded_file = st.file_uploader(
        "동영상 파일을 선택하세요",
        type=['mp4', 'avi', 'mov'],
        help="지원 형식: MP4, AVI, MOV"
    )
    
    if uploaded_file is not None:
        temp_file = None
        cap = None
        
        try:
            # 임시 파일 생성 및 저장
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            temp_file.write(uploaded_file.read())
            temp_file.close()
            
            # 비디오 캡처 객체 생성
            cap = cv2.VideoCapture(temp_file.name)
            if not cap.isOpened():
                raise ValueError("비디오 파일을 열 수 없습니다.")
            
            analyzer = PitcherFormAnalyzer()
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # 프로그레스 바 및 상태 표시
            progress_col, status_col = st.columns([3, 1])
            with progress_col:
                progress_bar = st.progress(0)
            with status_col:
                status_text = st.empty()
            
            # 분석 결과 저장용 컨테이너
            all_feedback = []
            all_angles = []
            
            # 프레임별 분석
            for frame_idx in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 프레임 분석
                image, feedback, angles = analyzer.analyze_frame(frame)
                
                # 결과 저장
                if feedback:
                    all_feedback.extend(feedback)
                if angles:
                    all_angles.append(angles)
                
                # 진행 상황 업데이트
                progress = int((frame_idx + 1) / total_frames * 100)
                progress_bar.progress(progress)
                status_text.text(f"분석 중... {progress}%")
            
            # 분석 결과 표시
            st.success("✅ 분석이 완료되었습니다!")
            
            if all_angles:
                # 각도 데이터 분석
                angle_df = pd.DataFrame(all_angles)
                
                # 탭으로 결과 구분
                tab1, tab2 = st.tabs(["📊 각도 분석", "💡 자세 피드백"])
                
                with tab1:
                    st.subheader("각도 분석 결과")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # 평균 각도 표시
                        st.write("평균 각도:")
                        mean_angles = angle_df.mean()
                        for joint, angle in mean_angles.items():
                            st.metric(
                                label=f"{joint.capitalize()} 각도",
                                value=f"{angle:.1f}°"
                            )
                    
                    with col2:
                        # 각도 변화 그래프
                        st.write("각도 변화 그래프:")
                        st.line_chart(angle_df)
                
                with tab2:
                    st.subheader("자세 피드백")
                    if all_feedback:
                        unique_feedback = list(set(all_feedback))
                        for idx, fb in enumerate(unique_feedback, 1):
                            st.warning(f"{idx}. {fb}")
                    else:
                        st.success("모든 관절 각도가 적정 범위 내에 있습니다! 👍")
            
        except Exception as e:
            st.error(f"😢 분석 중 오류가 발생했습니다: {str(e)}")
            logger.error(f"분석 중 오류 발생: {e}", exc_info=True)
            
        finally:
            # 리소스 정리
            if cap is not None:
                cap.release()
            
            if temp_file is not None and os.path.exists(temp_file.name):
                try:
                    os.unlink(temp_file.name)
                except Exception as e:
                    logger.error(f"임시 파일 삭제 중 오류 발생: {e}")

if __name__ == '__main__':
    main()