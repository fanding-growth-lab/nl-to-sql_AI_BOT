"""
학습 데이터 자동 통합 시스템
정확도 평가 결과와 피드백을 학습 데이터로 자동 추가
"""

import logging
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path

from .sql_accuracy_evaluator import SQLAccuracyMetrics, AccuracyLevel, get_accuracy_evaluator
from .feedback_collector import FeedbackCollector, FeedbackType, get_feedback_collector
from .auto_learning_system import AutoLearningSystem
from core.logging import get_logger

logger = get_logger(__name__)


class LearningDataIntegrator:
    """
    학습 데이터 통합기
    정확도 평가 결과와 피드백을 학습 데이터로 자동 통합
    """
    
    def __init__(
        self,
        learning_data_path: Optional[str] = None,
        feedback_file: Optional[str] = None
    ):
        """
        LearningDataIntegrator 초기화
        
        Args:
            learning_data_path: 학습 데이터 파일 경로
            feedback_file: 피드백 파일 경로
        """
        self.auto_learning_system = AutoLearningSystem(learning_data_path)
        self.feedback_collector = get_feedback_collector(feedback_file)
        self.accuracy_evaluator = get_accuracy_evaluator()
        
        # 자동 학습 활성화 설정
        self.auto_learn_enabled = True
        self.min_accuracy_for_learning = 0.5  # 학습 데이터로 추가할 최소 정확도
        
        logger.info("LearningDataIntegrator initialized")
    
    def integrate_accuracy_evaluation(
        self,
        user_query: str,
        generated_sql: str,
        accuracy_metrics: SQLAccuracyMetrics,
        session_id: Optional[str] = None
    ) -> bool:
        """
        정확도 평가 결과를 학습 데이터로 통합
        
        Args:
            user_query: 사용자 쿼리
            generated_sql: 생성된 SQL
            accuracy_metrics: 정확도 평가 결과
            session_id: 세션 ID
            
        Returns:
            bool: 통합 성공 여부
        """
        try:
            # 낮은 정확도 쿼리는 학습 데이터로 추가 (개선 필요)
            if accuracy_metrics.accuracy_level in [AccuracyLevel.LOW, AccuracyLevel.FAILED]:
                logger.info(f"Low accuracy query detected ({accuracy_metrics.accuracy_score:.2%}), adding to learning data")
                
                # AutoLearningSystem에 실패 패턴 추가
                # record_query_interaction 메서드 사용 (실제 메서드명)
                self.auto_learning_system.record_query_interaction(
                    user_id=session_id or "system",
                    query=user_query,
                    mapping_result={"sql_query": generated_sql},
                    confidence=accuracy_metrics.accuracy_score,
                    success=False
                )
                
                # 피드백 수집 (자동 부정 피드백)
                if session_id:
                    self.feedback_collector.collect_feedback(
                        session_id=session_id,
                        user_query=user_query,
                        generated_sql=generated_sql,
                        feedback_type=FeedbackType.NEGATIVE,
                        accuracy_score=accuracy_metrics.accuracy_score,
                        feedback_text=f"Low accuracy detected: {accuracy_metrics.accuracy_level.value}"
                    )
                
                return True
            
            # 높은 정확도 쿼리는 성공 패턴으로 기록
            elif accuracy_metrics.accuracy_level in [AccuracyLevel.PERFECT, AccuracyLevel.HIGH]:
                logger.debug(f"High accuracy query detected ({accuracy_metrics.accuracy_score:.2%}), recording success pattern")
                
                self.auto_learning_system.record_query_interaction(
                    user_id=session_id or "system",
                    query=user_query,
                    mapping_result={"sql_query": generated_sql},
                    confidence=accuracy_metrics.accuracy_score,
                    success=True
                )
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to integrate accuracy evaluation: {e}")
            return False
    
    def integrate_feedback(
        self,
        session_id: str,
        user_query: str,
        generated_sql: str,
        feedback_type: FeedbackType,
        corrected_sql: Optional[str] = None,
        feedback_text: Optional[str] = None
    ) -> bool:
        """
        사용자 피드백을 학습 데이터로 통합
        
        Args:
            session_id: 세션 ID
            user_query: 사용자 쿼리
            generated_sql: 생성된 SQL
            feedback_type: 피드백 유형
            corrected_sql: 수정된 SQL (선택사항)
            feedback_text: 피드백 텍스트 (선택사항)
            
        Returns:
            bool: 통합 성공 여부
        """
        try:
            # 피드백 수집
            feedback = self.feedback_collector.collect_feedback(
                session_id=session_id,
                user_query=user_query,
                generated_sql=generated_sql,
                feedback_type=feedback_type,
                corrected_sql=corrected_sql,
                feedback_text=feedback_text
            )
            
            # 학습 데이터에 반영
            if feedback_type == FeedbackType.CORRECTION and corrected_sql:
                # 수정된 SQL을 성공 패턴으로 기록
                logger.info(f"Adding corrected SQL to learning data")
                self.auto_learning_system.record_query_interaction(
                    user_id=session_id or "system",
                    query=user_query,
                    mapping_result={"sql_query": corrected_sql},
                    confidence=0.9,  # 수정된 SQL은 높은 신뢰도
                    success=True
                )
                
                # 원본 실패 쿼리도 기록
                self.auto_learning_system.record_query_interaction(
                    user_id=session_id or "system",
                    query=user_query,
                    mapping_result={"sql_query": generated_sql},
                    confidence=0.3,  # 낮은 신뢰도
                    success=False
                )
                
            elif feedback_type == FeedbackType.NEGATIVE:
                # 부정 피드백을 실패 패턴으로 기록
                logger.info(f"Adding negative feedback to learning data")
                self.auto_learning_system.record_query_interaction(
                    user_id=session_id or "system",
                    query=user_query,
                    mapping_result={"sql_query": generated_sql},
                    confidence=0.2,
                    success=False
                )
                
            elif feedback_type == FeedbackType.POSITIVE:
                # 긍정 피드백을 성공 패턴으로 기록
                logger.info(f"Adding positive feedback to learning data")
                self.auto_learning_system.record_query_interaction(
                    user_id=session_id or "system",
                    query=user_query,
                    mapping_result={"sql_query": generated_sql},
                    confidence=0.9,
                    success=True
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to integrate feedback: {e}")
            return False
    
    def analyze_and_improve(self) -> Dict[str, Any]:
        """
        학습 데이터 분석 및 개선 제안
        
        Returns:
            분석 결과 및 개선 제안
        """
        try:
            # 피드백 패턴 분석
            feedback_analysis = self.feedback_collector.analyze_feedback_patterns()
            
            # 정확도 통계
            accuracy_stats = self.accuracy_evaluator.get_accuracy_statistics()
            
            # 학습 메트릭 (실제 메서드명 사용)
            learning_metrics = self.auto_learning_system.get_performance_metrics()
            
            # 개선 제안 생성
            improvement_suggestions = []
            
            # 낮은 정확도 쿼리 분석
            if accuracy_stats["average_accuracy"] < 0.7:
                improvement_suggestions.append(
                    f"평균 정확도가 낮습니다 ({accuracy_stats['average_accuracy']:.2%}). "
                    "프롬프트 템플릿 개선이 필요합니다."
                )
            
            # 부정 피드백 분석
            if feedback_analysis.get("negative_feedback_count", 0) > 0:
                negative_ratio = feedback_analysis["negative_feedback_count"] / max(1, feedback_analysis["total_feedback"])
                if negative_ratio > 0.3:
                    improvement_suggestions.append(
                        f"부정 피드백 비율이 높습니다 ({negative_ratio:.2%}). "
                        "SQL 생성 로직 개선이 필요합니다."
                    )
            
            # 일반적인 문제점
            common_issues = feedback_analysis.get("common_issues", [])
            if common_issues:
                improvement_suggestions.append(
                    f"일반적인 문제점: {', '.join(common_issues[:3])}"
                )
            
            return {
                "feedback_analysis": feedback_analysis,
                "accuracy_statistics": accuracy_stats,
                "learning_metrics": learning_metrics,
                "improvement_suggestions": improvement_suggestions,
                "overall_health": self._calculate_overall_health(
                    accuracy_stats, feedback_analysis, learning_metrics
                )
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze and improve: {e}")
            return {
                "error": str(e),
                "improvement_suggestions": []
            }
    
    def _calculate_overall_health(
        self,
        accuracy_stats: Dict[str, Any],
        feedback_analysis: Dict[str, Any],
        learning_metrics: Any
    ) -> str:
        """전체 시스템 건강도 계산"""
        try:
            avg_accuracy = accuracy_stats.get("average_accuracy", 0.0)
            success_rate = accuracy_stats.get("success_rate", 0.0)
            
            total_feedback = feedback_analysis.get("total_feedback", 0)
            negative_count = feedback_analysis.get("negative_feedback_count", 0)
            
            negative_ratio = negative_count / max(1, total_feedback)
            
            # 건강도 점수 계산
            health_score = (avg_accuracy * 0.5 + success_rate * 0.3 + (1 - negative_ratio) * 0.2)
            
            if health_score >= 0.8:
                return "excellent"
            elif health_score >= 0.6:
                return "good"
            elif health_score >= 0.4:
                return "fair"
            else:
                return "poor"
                
        except Exception as e:
            logger.warning(f"Failed to calculate overall health: {e}")
            return "unknown"
    
    def export_learning_data_for_prompts(self) -> List[Dict[str, Any]]:
        """
        프롬프트 템플릿에 사용할 학습 데이터 내보내기
        
        Returns:
            프롬프트용 학습 데이터 리스트
        """
        try:
            # 피드백에서 학습 데이터 추출
            feedback_learning_data = self.feedback_collector.get_feedback_for_learning()
            
            # AutoLearningSystem에서 성공 패턴 추출
            learning_examples = []
            for pattern_key, pattern in self.auto_learning_system.query_patterns.items():
                if pattern.success_rate >= 0.7 and pattern.avg_confidence >= 0.7:
                    # 성공률이 높은 패턴만 학습 데이터로 사용
                    learning_examples.append({
                        "query_pattern": pattern.pattern,
                        "success_rate": pattern.success_rate,
                        "avg_confidence": pattern.avg_confidence,
                        "source": "auto_learning_system"
                    })
            
            # 피드백 데이터와 결합
            combined_data = []
            combined_data.extend(feedback_learning_data)
            combined_data.extend(learning_examples)
            
            return combined_data
            
        except Exception as e:
            logger.error(f"Failed to export learning data: {e}")
            return []


# 싱글톤 인스턴스
_integrator_instance: Optional[LearningDataIntegrator] = None


def get_learning_integrator(
    learning_data_path: Optional[str] = None,
    feedback_file: Optional[str] = None
) -> LearningDataIntegrator:
    """학습 데이터 통합기 싱글톤 인스턴스 반환"""
    global _integrator_instance
    if _integrator_instance is None:
        _integrator_instance = LearningDataIntegrator(learning_data_path, feedback_file)
    return _integrator_instance

