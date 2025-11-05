#!/usr/bin/env python3
"""
Auto Learning System
자동 학습 시스템 - 사용 패턴 분석 및 시스템 개선
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import re
import os
from pathlib import Path
import threading
import queue
import time
from core.db import get_cached_db_schema

logger = logging.getLogger(__name__)

@dataclass
class QueryPattern:
    """쿼리 패턴 데이터"""
    pattern: str
    frequency: int = 0
    success_rate: float = 0.0
    avg_confidence: float = 0.0
    last_used: Optional[datetime] = None
    common_variations: List[str] = field(default_factory=list)
    user_feedback: List[str] = field(default_factory=list)

@dataclass
class LearningMetrics:
    """학습 메트릭"""
    total_queries: int = 0
    successful_mappings: int = 0
    failed_mappings: int = 0
    avg_confidence: float = 0.0
    most_common_patterns: List[str] = field(default_factory=list)
    improvement_suggestions: List[str] = field(default_factory=list)

@dataclass
class UserBehavior:
    """사용자 행동 패턴"""
    user_id: str
    query_preferences: Dict[str, int] = field(default_factory=dict)
    time_patterns: Dict[str, int] = field(default_factory=dict)
    success_rate: float = 0.0
    feedback_history: List[str] = field(default_factory=list)

class AutoLearningSystem:
    """자동 학습 시스템"""
    
    def __init__(self, learning_data_path: Optional[str] = None):
        # 경로가 제공되지 않으면 기본 경로 사용
        if learning_data_path is None:
            # 프로젝트 루트 기준으로 절대 경로 생성
            current_file = Path(__file__)
            project_root = current_file.parent.parent.parent  # src/agentic_flow -> src -> project_root
            learning_data_path = str(project_root / "src" / "agentic_flow" / "learning_data.json")
        
        # Path 객체로 변환하여 정규화
        self.learning_data_path = str(Path(learning_data_path).resolve())
        self.query_patterns: Dict[str, QueryPattern] = {}
        self.user_behaviors: Dict[str, UserBehavior] = {}
        self.learning_metrics = LearningMetrics()
        self.pattern_confidence_threshold = 0.7
        self.learning_enabled = True
        
        # DB 스키마 정보 로드
        self.db_schema = get_cached_db_schema()
        
        # 비동기 저장을 위한 큐와 스레드
        self._save_queue = queue.Queue()
        self._save_thread = None
        self._batch_size = 10  # 배치 크기
        self._save_interval = 60  # 저장 간격 (초)
        self._last_save_time = time.time()
        
        # 학습 데이터 로드
        self._load_learning_data()
        
        # 백그라운드 저장 스레드 시작
        self._start_background_saver()
    
    def _load_learning_data(self):
        """학습 데이터 로드"""
        try:
            learning_file = Path(self.learning_data_path)
            if learning_file.exists():
                with open(learning_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                # 쿼리 패턴 로드
                for pattern_key, pattern_data in data.get("query_patterns", {}).items():
                    pattern = QueryPattern(
                        pattern=pattern_data["pattern"],
                        frequency=pattern_data["frequency"],
                        success_rate=pattern_data["success_rate"],
                        avg_confidence=pattern_data["avg_confidence"],
                        last_used=datetime.fromisoformat(pattern_data["last_used"]) if pattern_data.get("last_used") else None,
                        common_variations=pattern_data.get("common_variations", []),
                        user_feedback=pattern_data.get("user_feedback", [])
                    )
                    self.query_patterns[pattern_key] = pattern
                
                # 사용자 행동 로드
                for user_id, behavior_data in data.get("user_behaviors", {}).items():
                    behavior = UserBehavior(
                        user_id=user_id,
                        query_preferences=behavior_data.get("query_preferences", {}),
                        time_patterns=behavior_data.get("time_patterns", {}),
                        success_rate=behavior_data.get("success_rate", 0.0),
                        feedback_history=behavior_data.get("feedback_history", [])
                    )
                    self.user_behaviors[user_id] = behavior
                
                # 학습 메트릭 로드
                metrics_data = data.get("learning_metrics", {})
                self.learning_metrics = LearningMetrics(
                    total_queries=metrics_data.get("total_queries", 0),
                    successful_mappings=metrics_data.get("successful_mappings", 0),
                    failed_mappings=metrics_data.get("failed_mappings", 0),
                    avg_confidence=metrics_data.get("avg_confidence", 0.0),
                    most_common_patterns=metrics_data.get("most_common_patterns", []),
                    improvement_suggestions=metrics_data.get("improvement_suggestions", [])
                )
                
                logger.info(f"학습 데이터 로드 완료: {len(self.query_patterns)} 패턴, {len(self.user_behaviors)} 사용자")
            else:
                logger.info("학습 데이터 파일이 없습니다. 새로운 학습을 시작합니다.")
        except Exception as e:
            logger.error(f"학습 데이터 로드 실패: {str(e)}")
    
    def _save_learning_data(self):
        """학습 데이터 저장"""
        try:
            learning_file = Path(self.learning_data_path)
            # 디렉토리 생성
            learning_file.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                "query_patterns": {},
                "user_behaviors": {},
                "learning_metrics": {
                    "total_queries": self.learning_metrics.total_queries,
                    "successful_mappings": self.learning_metrics.successful_mappings,
                    "failed_mappings": self.learning_metrics.failed_mappings,
                    "avg_confidence": self.learning_metrics.avg_confidence,
                    "most_common_patterns": self.learning_metrics.most_common_patterns,
                    "improvement_suggestions": self.learning_metrics.improvement_suggestions
                }
            }
            
            # 쿼리 패턴 저장
            for pattern_key, pattern in self.query_patterns.items():
                data["query_patterns"][pattern_key] = {
                    "pattern": pattern.pattern,
                    "frequency": pattern.frequency,
                    "success_rate": pattern.success_rate,
                    "avg_confidence": pattern.avg_confidence,
                    "last_used": pattern.last_used.isoformat() if pattern.last_used else None,
                    "common_variations": pattern.common_variations,
                    "user_feedback": pattern.user_feedback
                }
            
            # 사용자 행동 저장
            for user_id, behavior in self.user_behaviors.items():
                data["user_behaviors"][user_id] = {
                    "query_preferences": behavior.query_preferences,
                    "time_patterns": behavior.time_patterns,
                    "success_rate": behavior.success_rate,
                    "feedback_history": behavior.feedback_history
                }
            
            # 파일 저장
            with open(learning_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.info("학습 데이터 저장 완료")
        except Exception as e:
            logger.error(f"학습 데이터 저장 실패: {str(e)}")
    
    def record_query_interaction(self, user_id: str, query: str, mapping_result: Dict[str, Any], 
                               confidence: float, success: bool, user_feedback: Optional[str] = None):
        """쿼리 상호작용 기록"""
        if not self.learning_enabled:
            return
        
        try:
            # 전역 메트릭 업데이트
            self.learning_metrics.total_queries += 1
            if success:
                self.learning_metrics.successful_mappings += 1
            else:
                self.learning_metrics.failed_mappings += 1
            
            # 평균 신뢰도 업데이트
            total_confidence = self.learning_metrics.avg_confidence * (self.learning_metrics.total_queries - 1)
            self.learning_metrics.avg_confidence = (total_confidence + confidence) / self.learning_metrics.total_queries
            
            # 쿼리 패턴 분석
            pattern_key = self._extract_pattern_key(query)
            if pattern_key not in self.query_patterns:
                self.query_patterns[pattern_key] = QueryPattern(pattern=query)
            
            pattern = self.query_patterns[pattern_key]
            pattern.frequency += 1
            pattern.last_used = datetime.now()
            
            # 성공률 업데이트
            if success:
                pattern.success_rate = (pattern.success_rate * (pattern.frequency - 1) + 1.0) / pattern.frequency
            else:
                pattern.success_rate = (pattern.success_rate * (pattern.frequency - 1) + 0.0) / pattern.frequency
            
            # 평균 신뢰도 업데이트
            pattern.avg_confidence = (pattern.avg_confidence * (pattern.frequency - 1) + confidence) / pattern.frequency
            
            # 사용자 피드백 기록
            if user_feedback:
                pattern.user_feedback.append(user_feedback)
                # 최근 10개 피드백만 유지
                if len(pattern.user_feedback) > 10:
                    pattern.user_feedback = pattern.user_feedback[-10:]
            
            # 사용자 행동 패턴 업데이트
            self._update_user_behavior(user_id, query, success, confidence)
            
            # 비동기 저장 큐에 추가
            self._queue_save_request({
                "type": "query_interaction",
                "user_id": user_id,
                "query": query,
                "mapping_result": mapping_result,
                "confidence": confidence,
                "success": success,
                "user_feedback": user_feedback,
                "timestamp": datetime.now().isoformat()
            })
            
            logger.info(f"쿼리 상호작용 기록: {user_id} - {query[:50]}... (성공: {success}, 신뢰도: {confidence:.2f})")
            
        except Exception as e:
            logger.error(f"쿼리 상호작용 기록 실패: {str(e)}")
    
    def _extract_pattern_key(self, query: str) -> str:
        """쿼리에서 패턴 키 추출 (고도화된 정규화 기반)"""
        keywords = []
        query_lower = query.lower()
        
        # 1. 월별 패턴 (동적 + 정규화)
        month_keyword = self._extract_normalized_month(query_lower)
        if month_keyword:
            keywords.append(month_keyword)
        
        # 2. DB 스키마 기반 비즈니스 키워드 (정규화)
        business_keyword = self._extract_normalized_business_keyword(query_lower)
        if business_keyword:
            keywords.append(business_keyword)
        
        # 3. 쿼리 액션 (정규화)
        action_keyword = self._extract_normalized_action(query_lower)
        if action_keyword:
            keywords.append(action_keyword)
        
        # 4. 메트릭 타입 (정규화)
        metric_keyword = self._extract_normalized_metric(query_lower)
        if metric_keyword:
            keywords.append(metric_keyword)
        
        return "_".join(keywords) if keywords else "general_query"
    
    def _extract_normalized_month(self, query_lower: str) -> Optional[str]:
        """정규화된 월 추출"""
        # 직접적인 월 패턴
        month_patterns = [f"{i}월" for i in range(1, 13)]
        for month in month_patterns:
            if month in query_lower:
                return f"MONTH_{month.replace('월', '')}"
        
        # 상대적 시간 표현 정규화
        relative_time_mapping = {
            '지난달': 'MONTH_PREV',
            '이번달': 'MONTH_CURRENT', 
            '다음달': 'MONTH_NEXT',
            '최근': 'MONTH_RECENT',
            '작년': 'MONTH_LAST_YEAR'
        }
        
        for pattern, normalized in relative_time_mapping.items():
            if pattern in query_lower:
                return normalized
        
        return None
    
    def _extract_normalized_business_keyword(self, query_lower: str) -> Optional[str]:
        """정규화된 비즈니스 키워드 추출"""
        business_keywords = self._get_business_keywords_from_schema()
        
        # 키워드 그룹화 및 정규화
        keyword_groups = {
            'MEMBER': ['멤버십', '맴버쉽', '회원', '멤버', '가입자'],
            'CREATOR': ['크리에이터', '창작자', '작가'],
            'PAYMENT': ['결제', '매출', '수익', '금액', '가격'],
            'CONTENT': ['포스트', '게시물', '콘텐츠', '글'],
            'PROJECT': ['프로젝트', '프로젝트'],
            'COLLECTION': ['컬렉션', '수집'],
            'TIER': ['티어', '등급', '레벨'],
            'LOGIN': ['로그인', '접속', '방문']
        }
        
        for group, keywords in keyword_groups.items():
            for keyword in keywords:
                if keyword in query_lower:
                    return group
        
        return None
    
    def _extract_normalized_action(self, query_lower: str) -> Optional[str]:
        """정규화된 액션 키워드 추출"""
        action_groups = {
            'QUERY_ACTION': ['조회', '보여줘', '알려줘', '얼마나', '몇 개', '몇 명', '분석', '확인'],
            'COMPARISON': ['비교', '대비', '차이', '증감'],
            'RANKING': ['순위', '상위', '탑', '최고', '최대'],
            'TREND': ['추이', '변화', '증가', '감소', '성장']
        }
        
        for group, keywords in action_groups.items():
            for keyword in keywords:
                if keyword in query_lower:
                    return group
        
        return None
    
    def _extract_normalized_metric(self, query_lower: str) -> Optional[str]:
        """정규화된 메트릭 키워드 추출"""
        metric_groups = {
            'COUNT': ['수', '명', '개', '건', '회'],
            'ACTIVE': ['활성', '활발', '활발한'],
            'NEW': ['신규', '새로운', '신입'],
            'INACTIVE': ['비활성', '비활발', '비활발한'],
            'DELETED': ['탈퇴', '삭제', '제거'],
            'REVENUE': ['매출', '수익', '금액', '가격'],
            'PERFORMANCE': ['성과', '실적', '결과']
        }
        
        for group, keywords in metric_groups.items():
            for keyword in keywords:
                if keyword in query_lower:
                    return group
        
        return None
    
    def _get_business_keywords_from_schema(self) -> List[str]:
        """DB 스키마에서 비즈니스 키워드 추출"""
        keywords = set()
        
        try:
            # 테이블명에서 키워드 추출
            for table_name in self.db_schema.keys():
                if table_name.startswith('t_'):
                    # 테이블명에서 비즈니스 의미 추출
                    table_keywords = {
                        't_member': ['멤버십', '맴버쉽', '회원', '멤버'],
                        't_creator': ['크리에이터', '창작자'],
                        't_payment': ['결제', '매출', '수익', '금액'],
                        't_post': ['포스트', '게시물', '콘텐츠'],
                        't_project': ['프로젝트', '프로젝트'],
                        't_collection': ['컬렉션', '수집'],
                        't_tier': ['티어', '등급'],
                        't_member_login_log': ['로그인', '접속']
                    }
                    
                    if table_name in table_keywords:
                        keywords.update(table_keywords[table_name])
            
            # 컬럼명에서 키워드 추출
            for table_name, table_info in self.db_schema.items():
                columns = table_info.get("columns", {})
                for column_name in columns.keys():
                    column_keywords = {
                        'status': ['상태', '활성', '비활성', '탈퇴'],
                        'view_count': ['조회수', '조회'],
                        'like_count': ['좋아요', '좋아요수'],
                        'ins_datetime': ['등록일', '가입일', '생성일'],
                        'price': ['가격', '금액'],
                        'heat': ['히트', '포인트']
                    }
                    
                    if column_name in column_keywords:
                        keywords.update(column_keywords[column_name])
            
            # 기본 비즈니스 키워드 추가
            keywords.update(['성과', '실적', '분석', '활성', '비활성', '탈퇴', '매출', '수익'])
            
        except Exception as e:
            logger.warning(f"스키마에서 키워드 추출 실패: {str(e)}")
            # 폴백: 기본 키워드 사용
            keywords = {'멤버십', '맴버쉽', '회원', '성과', '실적', '분석', '활성', '비활성', '탈퇴', '매출', '수익'}
        
        return list(keywords)
    
    def _update_user_behavior(self, user_id: str, query: str, success: bool, confidence: float):
        """사용자 행동 패턴 업데이트"""
        if user_id not in self.user_behaviors:
            self.user_behaviors[user_id] = UserBehavior(user_id=user_id)
        
        behavior = self.user_behaviors[user_id]
        
        # 쿼리 선호도 업데이트
        pattern_key = self._extract_pattern_key(query)
        behavior.query_preferences[pattern_key] = behavior.query_preferences.get(pattern_key, 0) + 1
        
        # 시간 패턴 업데이트
        current_hour = datetime.now().hour
        time_slot = f"{current_hour:02d}:00"
        behavior.time_patterns[time_slot] = behavior.time_patterns.get(time_slot, 0) + 1
        
        # 성공률 업데이트
        total_queries = sum(behavior.query_preferences.values())
        if total_queries > 0:
            if success:
                behavior.success_rate = (behavior.success_rate * (total_queries - 1) + 1.0) / total_queries
            else:
                behavior.success_rate = (behavior.success_rate * (total_queries - 1) + 0.0) / total_queries
    
    def analyze_learning_patterns(self) -> Dict[str, Any]:
        """학습 패턴 분석"""
        try:
            analysis = {
                "total_patterns": len(self.query_patterns),
                "total_users": len(self.user_behaviors),
                "overall_success_rate": 0.0,
                "top_patterns": [],
                "improvement_areas": [],
                "user_insights": {}
            }
            
            # 전체 성공률 계산
            if self.learning_metrics.total_queries > 0:
                analysis["overall_success_rate"] = self.learning_metrics.successful_mappings / self.learning_metrics.total_queries
            
            # 상위 패턴 분석
            sorted_patterns = sorted(
                self.query_patterns.items(),
                key=lambda x: x[1].frequency,
                reverse=True
            )
            
            analysis["top_patterns"] = [
                {
                    "pattern": pattern.pattern,
                    "frequency": pattern.frequency,
                    "success_rate": pattern.success_rate,
                    "avg_confidence": pattern.avg_confidence
                }
                for _, pattern in sorted_patterns[:10]
            ]
            
            # 개선 영역 식별
            low_success_patterns = [
                pattern for pattern in self.query_patterns.values()
                if pattern.success_rate < 0.5 and pattern.frequency > 3
            ]
            
            analysis["improvement_areas"] = [
                {
                    "pattern": pattern.pattern,
                    "success_rate": pattern.success_rate,
                    "frequency": pattern.frequency,
                    "suggestion": "패턴 매핑 정확도 개선 필요"
                }
                for pattern in low_success_patterns[:5]
            ]
            
            # 사용자 인사이트
            for user_id, behavior in self.user_behaviors.items():
                if behavior.success_rate > 0:
                    analysis["user_insights"][user_id] = {
                        "success_rate": behavior.success_rate,
                        "preferred_patterns": sorted(
                            behavior.query_preferences.items(),
                            key=lambda x: x[1],
                            reverse=True
                        )[:3],
                        "active_time_slots": sorted(
                            behavior.time_patterns.items(),
                            key=lambda x: x[1],
                            reverse=True
                        )[:3]
                    }
            
            return analysis
            
        except Exception as e:
            logger.error(f"학습 패턴 분석 실패: {str(e)}")
            return {}
    
    def generate_improvement_suggestions(self) -> List[str]:
        """개선 제안 생성 (DB 스키마 기반)"""
        suggestions = []
        
        try:
            # 낮은 성공률 패턴 분석
            low_success_patterns = [
                pattern for pattern in self.query_patterns.values()
                if pattern.success_rate < 0.6 and pattern.frequency > 5
            ]
            
            if low_success_patterns:
                suggestions.append(f"성공률이 낮은 패턴 {len(low_success_patterns)}개 발견: 매핑 로직 개선 필요")
            
            # 자주 사용되는 패턴 분석
            frequent_patterns = [
                pattern for pattern in self.query_patterns.values()
                if pattern.frequency > 10
            ]
            
            if frequent_patterns:
                suggestions.append(f"자주 사용되는 패턴 {len(frequent_patterns)}개: 우선순위 매핑 강화 권장")
            
            # 사용자 피드백 분석
            feedback_patterns = [
                pattern for pattern in self.query_patterns.values()
                if pattern.user_feedback
            ]
            
            if feedback_patterns:
                suggestions.append(f"사용자 피드백이 있는 패턴 {len(feedback_patterns)}개: 피드백 기반 개선 적용")
            
            # 시간 패턴 분석
            all_time_patterns = defaultdict(int)
            for behavior in self.user_behaviors.values():
                for time_slot, count in behavior.time_patterns.items():
                    all_time_patterns[time_slot] += count
            
            peak_hours = sorted(all_time_patterns.items(), key=lambda x: x[1], reverse=True)[:3]
            if peak_hours:
                suggestions.append(f"피크 시간대 {[hour for hour, _ in peak_hours]}: 해당 시간대 성능 최적화 권장")
            
            # DB 스키마 기반 개선 제안
            schema_suggestions = self._generate_schema_based_suggestions()
            suggestions.extend(schema_suggestions)
            
            return suggestions
            
        except Exception as e:
            logger.error(f"개선 제안 생성 실패: {str(e)}")
            return []
    
    def _generate_schema_based_suggestions(self) -> List[str]:
        """DB 스키마 기반 개선 제안 생성"""
        suggestions = []
        
        try:
            # 테이블별 사용 빈도 분석
            table_usage = defaultdict(int)
            for pattern in self.query_patterns.values():
                for table_name in self.db_schema.keys():
                    if table_name in pattern.pattern.lower():
                        table_usage[table_name] += pattern.frequency
            
            # 자주 사용되지 않는 테이블 식별
            total_usage = sum(table_usage.values())
            if total_usage > 0:
                for table_name, usage in table_usage.items():
                    usage_ratio = usage / total_usage
                    if usage_ratio < 0.05:  # 5% 미만 사용
                        suggestions.append(f"테이블 '{table_name}' 사용률이 낮음 ({usage_ratio:.1%}): 관련 쿼리 패턴 개발 필요")
            
            # 컬럼별 사용 패턴 분석
            column_usage = defaultdict(int)
            for table_name, table_info in self.db_schema.items():
                columns = table_info.get("columns", {})
                for column_name in columns.keys():
                    for pattern in self.query_patterns.values():
                        if column_name in pattern.pattern.lower():
                            column_usage[f"{table_name}.{column_name}"] += pattern.frequency
            
            # 자주 사용되는 컬럼 식별
            if column_usage:
                top_columns = sorted(column_usage.items(), key=lambda x: x[1], reverse=True)[:5]
                suggestions.append(f"자주 사용되는 컬럼: {', '.join([col for col, _ in top_columns])} - 인덱스 최적화 고려")
            
            # 스키마 변경 감지
            if hasattr(self, '_previous_schema'):
                current_tables = set(self.db_schema.keys())
                previous_tables = set(self._previous_schema.keys())
                
                new_tables = current_tables - previous_tables
                removed_tables = previous_tables - current_tables
                
                if new_tables:
                    suggestions.append(f"새로운 테이블 추가됨: {', '.join(new_tables)} - 관련 쿼리 패턴 학습 필요")
                if removed_tables:
                    suggestions.append(f"제거된 테이블: {', '.join(removed_tables)} - 관련 패턴 정리 필요")
            
            # 현재 스키마 저장 (다음 비교용)
            self._previous_schema = self.db_schema.copy()
            
        except Exception as e:
            logger.warning(f"스키마 기반 제안 생성 실패: {str(e)}")
        
        return suggestions
    
    def optimize_mapping_patterns(self) -> Dict[str, Any]:
        """매핑 패턴 최적화"""
        try:
            optimizations = {
                "promoted_patterns": [],
                "demoted_patterns": [],
                "new_patterns": [],
                "confidence_adjustments": {}
            }
            
            # 성공률이 높은 패턴 승격
            for pattern_key, pattern in self.query_patterns.items():
                if pattern.success_rate > 0.8 and pattern.frequency > 5:
                    optimizations["promoted_patterns"].append({
                        "pattern": pattern.pattern,
                        "old_priority": "medium",
                        "new_priority": "high",
                        "reason": f"높은 성공률 ({pattern.success_rate:.2f})"
                    })
                
                elif pattern.success_rate < 0.4 and pattern.frequency > 3:
                    optimizations["demoted_patterns"].append({
                        "pattern": pattern.pattern,
                        "old_priority": "high",
                        "new_priority": "low",
                        "reason": f"낮은 성공률 ({pattern.success_rate:.2f})"
                    })
            
            # 신뢰도 임계값 조정
            avg_confidence = self.learning_metrics.avg_confidence
            if avg_confidence > 0.8:
                optimizations["confidence_adjustments"]["threshold"] = "increase"
                optimizations["confidence_adjustments"]["reason"] = "높은 평균 신뢰도로 인한 임계값 상향 조정"
            elif avg_confidence < 0.6:
                optimizations["confidence_adjustments"]["threshold"] = "decrease"
                optimizations["confidence_adjustments"]["reason"] = "낮은 평균 신뢰도로 인한 임계값 하향 조정"
            
            return optimizations
            
        except Exception as e:
            logger.error(f"매핑 패턴 최적화 실패: {str(e)}")
            return {}
    
    def get_learning_report(self) -> Dict[str, Any]:
        """학습 리포트 생성"""
        try:
            analysis = self.analyze_learning_patterns()
            suggestions = self.generate_improvement_suggestions()
            optimizations = self.optimize_mapping_patterns()
            
            report = {
                "timestamp": datetime.now().isoformat(),
                "learning_metrics": {
                    "total_queries": self.learning_metrics.total_queries,
                    "success_rate": self.learning_metrics.avg_confidence,
                    "successful_mappings": self.learning_metrics.successful_mappings,
                    "failed_mappings": self.learning_metrics.failed_mappings
                },
                "pattern_analysis": analysis,
                "improvement_suggestions": suggestions,
                "optimization_recommendations": optimizations,
                "system_health": {
                    "learning_enabled": self.learning_enabled,
                    "data_freshness": "recent" if self.learning_metrics.total_queries > 0 else "no_data",
                    "pattern_coverage": len(self.query_patterns),
                    "user_engagement": len(self.user_behaviors)
                },
                "schema_analysis": {
                    "total_tables": len(self.db_schema),
                    "table_names": list(self.db_schema.keys()),
                    "schema_version": "current",
                    "schema_consistency": "verified"
                }
            }
            
            return report
            
        except Exception as e:
            logger.error(f"학습 리포트 생성 실패: {str(e)}")
            return {"error": str(e)}
    
    def enable_learning(self):
        """학습 활성화"""
        self.learning_enabled = True
        logger.info("자동 학습 시스템 활성화")
    
    def disable_learning(self):
        """학습 비활성화"""
        self.learning_enabled = False
        logger.info("자동 학습 시스템 비활성화")
    
    def reset_learning_data(self):
        """학습 데이터 초기화"""
        self.query_patterns.clear()
        self.user_behaviors.clear()
        self.learning_metrics = LearningMetrics()
        self._save_learning_data()
        logger.info("학습 데이터 초기화 완료")
    
    def refresh_schema(self):
        """DB 스키마 새로고침"""
        try:
            self.db_schema = get_cached_db_schema()
            logger.info(f"DB 스키마 새로고침 완료: {len(self.db_schema)} 테이블")
        except Exception as e:
            logger.error(f"DB 스키마 새로고침 실패: {str(e)}")
    
    def get_schema_insights(self) -> Dict[str, Any]:
        """스키마 인사이트 생성"""
        try:
            insights = {
                "total_tables": len(self.db_schema),
                "table_details": {},
                "column_statistics": {},
                "schema_health": "good"
            }
            
            # 테이블별 상세 정보
            for table_name, table_info in self.db_schema.items():
                columns = table_info.get("columns", {})
                insights["table_details"][table_name] = {
                    "column_count": len(columns),
                    "has_primary_key": "no" in columns,
                    "has_timestamps": any("datetime" in col.lower() for col in columns.keys()),
                    "has_status": "status" in columns
                }
            
            # 컬럼 통계
            all_columns = []
            for table_info in self.db_schema.values():
                all_columns.extend(table_info.get("columns", {}).keys())
            
            column_counter = Counter(all_columns)
            insights["column_statistics"] = {
                "most_common_columns": dict(column_counter.most_common(10)),
                "total_unique_columns": len(set(all_columns))
            }
            
            return insights
            
        except Exception as e:
            logger.error(f"스키마 인사이트 생성 실패: {str(e)}")
            return {"error": str(e)}
    
    def _start_background_saver(self):
        """백그라운드 저장 스레드 시작"""
        if self._save_thread is None or not self._save_thread.is_alive():
            self._save_thread = threading.Thread(target=self._background_save_worker, daemon=True)
            self._save_thread.start()
            logger.info("백그라운드 저장 스레드 시작")
    
    def _queue_save_request(self, data: Dict[str, Any]):
        """저장 요청을 큐에 추가"""
        try:
            self._save_queue.put(data, timeout=1)  # 1초 타임아웃
        except queue.Full:
            logger.warning("저장 큐가 가득참. 데이터 손실 가능성 있음")
    
    def _background_save_worker(self):
        """백그라운드 저장 워커 스레드"""
        batch = []
        
        while True:
            try:
                # 큐에서 데이터 수집 (배치 크기 또는 시간 간격)
                current_time = time.time()
                timeout = max(0.1, min(1.0, self._save_interval - (current_time - self._last_save_time)))
                
                try:
                    data = self._save_queue.get(timeout=timeout)
                    batch.append(data)
                except queue.Empty:
                    pass
                
                # 배치 크기 또는 시간 간격에 도달하면 저장
                if (len(batch) >= self._batch_size or 
                    current_time - self._last_save_time >= self._save_interval):
                    if batch:
                        self._process_batch(batch)
                        batch = []
                        self._last_save_time = current_time
                        
            except Exception as e:
                logger.error(f"백그라운드 저장 워커 오류: {str(e)}")
                time.sleep(1)
    
    def _process_batch(self, batch: List[Dict[str, Any]]):
        """배치 데이터 처리"""
        try:
            for data in batch:
                if data["type"] == "query_interaction":
                    self._process_query_interaction(data)
            
            # 주기적으로 전체 데이터 저장
            self._save_learning_data()
            logger.debug(f"배치 처리 완료: {len(batch)}개 항목")
            
        except Exception as e:
            logger.error(f"배치 처리 실패: {str(e)}")
    
    def _process_query_interaction(self, data: Dict[str, Any]):
        """쿼리 상호작용 데이터 처리"""
        try:
            # 이미 메모리에 반영된 데이터이므로 추가 처리 없음
            # 필요시 로그 파일에 추가 기록
            pass
        except Exception as e:
            logger.error(f"쿼리 상호작용 처리 실패: {str(e)}")
    
    def force_save(self):
        """강제 저장 (즉시 저장 필요시)"""
        try:
            self._save_learning_data()
            logger.info("강제 저장 완료")
        except Exception as e:
            logger.error(f"강제 저장 실패: {str(e)}")
    
    def stop_background_saver(self):
        """백그라운드 저장 스레드 중지"""
        if self._save_thread and self._save_thread.is_alive():
            # 큐에 종료 신호 전송
            self._save_queue.put({"type": "shutdown"})
            self._save_thread.join(timeout=5)
            logger.info("백그라운드 저장 스레드 중지")
    
    def apply_optimizations(self, mapping_patterns_path: str = "src/agentic_flow/mapping_patterns.json") -> Dict[str, Any]:
        """최적화 제안을 실제 mapping_patterns.json에 적용"""
        try:
            # 현재 최적화 분석 수행
            optimizations = self.optimize_mapping_patterns()
            
            if not optimizations:
                return {"status": "no_optimizations", "message": "적용할 최적화가 없습니다"}
            
            # mapping_patterns.json 파일 로드
            if not os.path.exists(mapping_patterns_path):
                return {"status": "error", "message": f"mapping_patterns.json 파일을 찾을 수 없습니다: {mapping_patterns_path}"}
            
            with open(mapping_patterns_path, 'r', encoding='utf-8') as f:
                mapping_data = json.load(f)
            
            # 백업 생성
            backup_path = f"{mapping_patterns_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            with open(backup_path, 'w', encoding='utf-8') as f:
                json.dump(mapping_data, f, ensure_ascii=False, indent=2)
            
            applied_changes = []
            
            # 승격된 패턴 처리
            for promoted in optimizations.get("promoted_patterns", []):
                change = self._apply_pattern_priority_change(mapping_data, promoted, "high")
                if change:
                    applied_changes.append(change)
            
            # 강등된 패턴 처리
            for demoted in optimizations.get("demoted_patterns", []):
                change = self._apply_pattern_priority_change(mapping_data, demoted, "low")
                if change:
                    applied_changes.append(change)
            
            # 신뢰도 임계값 조정
            confidence_adjustment = optimizations.get("confidence_adjustments", {})
            if confidence_adjustment:
                change = self._apply_confidence_adjustment(mapping_data, confidence_adjustment)
                if change:
                    applied_changes.append(change)
            
            # 변경사항이 있으면 파일 저장
            if applied_changes:
                with open(mapping_patterns_path, 'w', encoding='utf-8') as f:
                    json.dump(mapping_data, f, ensure_ascii=False, indent=2)
                
                logger.info(f"최적화 적용 완료: {len(applied_changes)}개 변경사항")
                return {
                    "status": "success",
                    "applied_changes": applied_changes,
                    "backup_created": backup_path,
                    "total_changes": len(applied_changes)
                }
            else:
                return {"status": "no_changes", "message": "적용 가능한 변경사항이 없습니다"}
                
        except Exception as e:
            logger.error(f"최적화 적용 실패: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _apply_pattern_priority_change(self, mapping_data: Dict[str, Any], pattern_info: Dict[str, Any], new_priority: str) -> Optional[Dict[str, Any]]:
        """패턴 우선순위 변경 적용"""
        try:
            pattern_name = pattern_info["pattern"]
            
            # patterns 배열에서 해당 패턴 찾기
            for pattern in mapping_data.get("patterns", []):
                if pattern.get("id") == pattern_name or pattern.get("name") == pattern_name:
                    old_priority = pattern.get("priority", "medium")
                    pattern["priority"] = new_priority
                    
                    return {
                        "type": "priority_change",
                        "pattern": pattern_name,
                        "old_priority": old_priority,
                        "new_priority": new_priority,
                        "reason": pattern_info.get("reason", "")
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"패턴 우선순위 변경 실패: {str(e)}")
            return None
    
    def _apply_confidence_adjustment(self, mapping_data: Dict[str, Any], adjustment: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """신뢰도 임계값 조정 적용"""
        try:
            threshold_direction = adjustment.get("threshold")
            reason = adjustment.get("reason", "")
            
            # 전역 설정에 신뢰도 임계값 추가
            if "global_settings" not in mapping_data:
                mapping_data["global_settings"] = {}
            
            current_threshold = mapping_data["global_settings"].get("confidence_threshold", 0.7)
            
            if threshold_direction == "increase":
                new_threshold = min(0.95, current_threshold + 0.05)
            elif threshold_direction == "decrease":
                new_threshold = max(0.3, current_threshold - 0.05)
            else:
                return None
            
            mapping_data["global_settings"]["confidence_threshold"] = new_threshold
            
            return {
                "type": "confidence_adjustment",
                "old_threshold": current_threshold,
                "new_threshold": new_threshold,
                "direction": threshold_direction,
                "reason": reason
            }
            
        except Exception as e:
            logger.error(f"신뢰도 조정 실패: {str(e)}")
            return None
    
    def get_optimization_status(self, mapping_patterns_path: str = "src/agentic_flow/mapping_patterns.json") -> Dict[str, Any]:
        """현재 최적화 상태 확인"""
        try:
            if not os.path.exists(mapping_patterns_path):
                return {"status": "error", "message": "mapping_patterns.json 파일이 없습니다"}
            
            with open(mapping_patterns_path, 'r', encoding='utf-8') as f:
                mapping_data = json.load(f)
            
            # 현재 설정 분석
            global_settings = mapping_data.get("global_settings", {})
            patterns = mapping_data.get("patterns", [])
            
            # 우선순위 분포 분석
            priority_distribution = defaultdict(int)
            for pattern in patterns:
                priority = pattern.get("priority", "medium")
                priority_distribution[priority] += 1
            
            # 신뢰도 분포 분석
            confidence_scores = [pattern.get("confidence", 0.5) for pattern in patterns if "confidence" in pattern]
            avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.5
            
            return {
                "status": "success",
                "total_patterns": len(patterns),
                "priority_distribution": dict(priority_distribution),
                "average_confidence": avg_confidence,
                "global_threshold": global_settings.get("confidence_threshold", 0.7),
                "last_updated": mapping_data.get("statistics", {}).get("last_updated", "unknown")
            }
            
        except Exception as e:
            logger.error(f"최적화 상태 확인 실패: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """성능 메트릭 수집"""
        try:
            metrics = {
                "learning_system": {
                    "total_patterns": len(self.query_patterns),
                    "total_users": len(self.user_behaviors),
                    "learning_enabled": self.learning_enabled,
                    "background_saver_active": self._save_thread and self._save_thread.is_alive()
                },
                "query_metrics": {
                    "total_queries": self.learning_metrics.total_queries,
                    "success_rate": self.learning_metrics.successful_mappings / max(1, self.learning_metrics.total_queries),
                    "avg_confidence": self.learning_metrics.avg_confidence
                },
                "storage_metrics": {
                    "queue_size": self._save_queue.qsize(),
                    "last_save_time": self._last_save_time,
                    "batch_size": self._batch_size,
                    "save_interval": self._save_interval
                },
                "schema_metrics": {
                    "total_tables": len(self.db_schema),
                    "schema_loaded": bool(self.db_schema)
                }
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"성능 메트릭 수집 실패: {str(e)}")
            return {"error": str(e)}
    
    def export_learning_data(self, export_path: str = None) -> str:
        """학습 데이터 내보내기"""
        try:
            if not export_path:
                export_path = f"learning_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            export_data = {
                "export_timestamp": datetime.now().isoformat(),
                "learning_metrics": {
                    "total_queries": self.learning_metrics.total_queries,
                    "successful_mappings": self.learning_metrics.successful_mappings,
                    "failed_mappings": self.learning_metrics.failed_mappings,
                    "avg_confidence": self.learning_metrics.avg_confidence
                },
                "query_patterns": {
                    key: {
                        "pattern": pattern.pattern,
                        "frequency": pattern.frequency,
                        "success_rate": pattern.success_rate,
                        "avg_confidence": pattern.avg_confidence,
                        "last_used": pattern.last_used.isoformat() if pattern.last_used else None
                    }
                    for key, pattern in self.query_patterns.items()
                },
                "user_behaviors": {
                    user_id: {
                        "query_preferences": behavior.query_preferences,
                        "success_rate": behavior.success_rate,
                        "total_queries": sum(behavior.query_preferences.values())
                    }
                    for user_id, behavior in self.user_behaviors.items()
                }
            }
            
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"학습 데이터 내보내기 완료: {export_path}")
            return export_path
            
        except Exception as e:
            logger.error(f"학습 데이터 내보내기 실패: {str(e)}")
            return ""
    
    def import_learning_data(self, import_path: str) -> bool:
        """학습 데이터 가져오기"""
        try:
            if not os.path.exists(import_path):
                logger.error(f"가져올 파일이 없습니다: {import_path}")
                return False
            
            with open(import_path, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            # 기존 데이터 백업
            self._save_learning_data()
            backup_path = f"{self.learning_data_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.rename(self.learning_data_path, backup_path)
            
            # 가져온 데이터로 교체
            self.query_patterns.clear()
            self.user_behaviors.clear()
            
            # 패턴 데이터 복원
            for key, pattern_data in import_data.get("query_patterns", {}).items():
                pattern = QueryPattern(
                    pattern=pattern_data["pattern"],
                    frequency=pattern_data["frequency"],
                    success_rate=pattern_data["success_rate"],
                    avg_confidence=pattern_data["avg_confidence"],
                    last_used=datetime.fromisoformat(pattern_data["last_used"]) if pattern_data.get("last_used") else None
                )
                self.query_patterns[key] = pattern
            
            # 사용자 행동 데이터 복원
            for user_id, behavior_data in import_data.get("user_behaviors", {}).items():
                behavior = UserBehavior(
                    user_id=user_id,
                    query_preferences=behavior_data["query_preferences"],
                    success_rate=behavior_data["success_rate"]
                )
                self.user_behaviors[user_id] = behavior
            
            # 메트릭 복원
            metrics_data = import_data.get("learning_metrics", {})
            self.learning_metrics = LearningMetrics(
                total_queries=metrics_data.get("total_queries", 0),
                successful_mappings=metrics_data.get("successful_mappings", 0),
                failed_mappings=metrics_data.get("failed_mappings", 0),
                avg_confidence=metrics_data.get("avg_confidence", 0.0)
            )
            
            # 저장
            self._save_learning_data()
            
            logger.info(f"학습 데이터 가져오기 완료: {import_path}")
            return True
            
        except Exception as e:
            logger.error(f"학습 데이터 가져오기 실패: {str(e)}")
            return False


