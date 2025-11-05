"""
피드백 수집 시스템
사용자 피드백을 수집하고 학습 데이터에 반영하는 시스템
"""

import logging
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path

from core.logging import get_logger

logger = get_logger(__name__)


class FeedbackType(Enum):
    """피드백 유형"""
    POSITIVE = "positive"  # 긍정적 (SQL이 올바름)
    NEGATIVE = "negative"  # 부정적 (SQL이 틀림)
    CORRECTION = "correction"  # 수정 (SQL 수정 제안)
    CLARIFICATION = "clarification"  # 명확화 (쿼리 이해 요청)


@dataclass
class UserFeedback:
    """사용자 피드백"""
    feedback_id: str
    session_id: str
    user_query: str
    generated_sql: str
    feedback_type: FeedbackType
    corrected_sql: Optional[str] = None  # 수정된 SQL (CORRECTION 타입일 때)
    feedback_text: Optional[str] = None  # 사용자 피드백 텍스트
    accuracy_score: Optional[float] = None  # 정확도 점수
    timestamp: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class FeedbackCollector:
    """
    피드백 수집기
    사용자 피드백을 수집, 저장, 분석
    """
    
    def __init__(self, feedback_file: Optional[str] = None):
        """
        FeedbackCollector 초기화
        
        Args:
            feedback_file: 피드백 저장 파일 경로
        """
        if feedback_file is None:
            feedback_file = "feedback_data.json"
        
        self.feedback_file = Path(feedback_file)
        self.feedback_history: List[UserFeedback] = []
        self._load_feedback_history()
        logger.info(f"FeedbackCollector initialized (storing to {self.feedback_file})")
    
    def collect_feedback(
        self,
        session_id: str,
        user_query: str,
        generated_sql: str,
        feedback_type: FeedbackType,
        corrected_sql: Optional[str] = None,
        feedback_text: Optional[str] = None,
        accuracy_score: Optional[float] = None
    ) -> UserFeedback:
        """
        피드백 수집
        
        Args:
            session_id: 세션 ID
            user_query: 사용자 쿼리
            generated_sql: 생성된 SQL
            feedback_type: 피드백 유형
            corrected_sql: 수정된 SQL (선택사항)
            feedback_text: 피드백 텍스트 (선택사항)
            accuracy_score: 정확도 점수 (선택사항)
            
        Returns:
            UserFeedback: 수집된 피드백 객체
        """
        import uuid
        feedback_id = str(uuid.uuid4())
        
        feedback = UserFeedback(
            feedback_id=feedback_id,
            session_id=session_id,
            user_query=user_query,
            generated_sql=generated_sql,
            feedback_type=feedback_type,
            corrected_sql=corrected_sql,
            feedback_text=feedback_text,
            accuracy_score=accuracy_score
        )
        
        # 피드백 저장
        self.feedback_history.append(feedback)
        self._save_feedback_history()
        
        logger.info(f"Feedback collected: {feedback_type.value} for session {session_id}")
        return feedback
    
    def get_feedback_for_learning(self) -> List[Dict[str, Any]]:
        """
        학습에 사용할 피드백 데이터 반환
        
        Returns:
            학습 데이터 형식의 피드백 리스트
        """
        learning_data = []
        
        for feedback in self.feedback_history:
            # 부정적 피드백이나 수정 피드백만 학습 데이터로 사용
            if feedback.feedback_type in [FeedbackType.NEGATIVE, FeedbackType.CORRECTION]:
                learning_item = {
                    "user_query": feedback.user_query,
                    "generated_sql": feedback.generated_sql,
                    "corrected_sql": feedback.corrected_sql or feedback.generated_sql,
                    "feedback_type": feedback.feedback_type.value,
                    "accuracy_score": feedback.accuracy_score,
                    "timestamp": feedback.timestamp
                }
                learning_data.append(learning_item)
        
        return learning_data
    
    def analyze_feedback_patterns(self) -> Dict[str, Any]:
        """
        피드백 패턴 분석
        
        Returns:
            피드백 패턴 분석 결과
        """
        if not self.feedback_history:
            return {
                "total_feedback": 0,
                "feedback_distribution": {},
                "common_issues": [],
                "improvement_areas": []
            }
        
        # 피드백 유형별 분포
        feedback_distribution = {}
        for feedback_type in FeedbackType:
            count = sum(1 for f in self.feedback_history if f.feedback_type == feedback_type)
            feedback_distribution[feedback_type.value] = count
        
        # 일반적인 문제점 추출
        negative_feedbacks = [f for f in self.feedback_history if f.feedback_type == FeedbackType.NEGATIVE]
        correction_feedbacks = [f for f in self.feedback_history if f.feedback_type == FeedbackType.CORRECTION]
        
        common_issues = []
        if negative_feedbacks:
            # 피드백 텍스트에서 공통 패턴 추출
            issue_texts = [f.feedback_text for f in negative_feedbacks if f.feedback_text]
            if issue_texts:
                common_issues = list(set(issue_texts))[:5]  # 상위 5개
        
        # 개선 영역 식별
        improvement_areas = []
        low_accuracy_feedbacks = [f for f in self.feedback_history if f.accuracy_score is not None and f.accuracy_score < 0.5]
        if low_accuracy_feedbacks:
            improvement_areas.append("낮은 정확도 쿼리 처리")
        
        if correction_feedbacks:
            improvement_areas.append("SQL 수정 요청 처리")
        
        return {
            "total_feedback": len(self.feedback_history),
            "feedback_distribution": feedback_distribution,
            "common_issues": common_issues,
            "improvement_areas": improvement_areas,
            "negative_feedback_count": len(negative_feedbacks),
            "correction_feedback_count": len(correction_feedbacks),
            "average_accuracy": sum(f.accuracy_score for f in self.feedback_history if f.accuracy_score is not None) / max(1, sum(1 for f in self.feedback_history if f.accuracy_score is not None))
        }
    
    def _load_feedback_history(self) -> None:
        """피드백 이력 로드"""
        try:
            if self.feedback_file.exists():
                with open(self.feedback_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.feedback_history = []
                    for item in data:
                        if isinstance(item, dict):
                            # feedback_type 문자열을 Enum으로 변환
                            if 'feedback_type' in item and isinstance(item['feedback_type'], str):
                                try:
                                    item['feedback_type'] = FeedbackType(item['feedback_type'])
                                except ValueError:
                                    logger.warning(f"Invalid feedback_type: {item['feedback_type']}, skipping")
                                    continue
                            try:
                                self.feedback_history.append(UserFeedback(**item))
                            except Exception as e:
                                logger.warning(f"Failed to load feedback record: {e}, skipping")
                logger.info(f"Loaded {len(self.feedback_history)} feedback records")
        except Exception as e:
            logger.warning(f"Failed to load feedback history: {e}")
            self.feedback_history = []
    
    def _save_feedback_history(self) -> None:
        """피드백 이력 저장"""
        try:
            # 디렉토리 생성
            self.feedback_file.parent.mkdir(parents=True, exist_ok=True)
            
            # 데이터 직렬화 (Enum을 문자열로 변환)
            data = []
            for feedback in self.feedback_history:
                feedback_dict = asdict(feedback)
                # FeedbackType Enum을 문자열로 변환
                if isinstance(feedback_dict.get('feedback_type'), FeedbackType):
                    feedback_dict['feedback_type'] = feedback_dict['feedback_type'].value
                data.append(feedback_dict)
            
            with open(self.feedback_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save feedback history: {e}")


# 싱글톤 인스턴스
_feedback_collector_instance: Optional[FeedbackCollector] = None


def get_feedback_collector(feedback_file: Optional[str] = None) -> FeedbackCollector:
    """피드백 수집기 싱글톤 인스턴스 반환"""
    global _feedback_collector_instance
    if _feedback_collector_instance is None:
        _feedback_collector_instance = FeedbackCollector(feedback_file)
    return _feedback_collector_instance

