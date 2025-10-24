#!/usr/bin/env python3
"""
사용자 검토 및 승인 노드
중간 검증 단계에서 사용자의 승인을 받는 노드
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime

from .validation_node import ValidationResult, ValidationStatus

logger = logging.getLogger(__name__)


class ReviewStatus(Enum):
    """검토 상태"""
    PENDING = "pending"           # 검토 대기 중
    APPROVED = "approved"         # 승인됨
    REJECTED = "rejected"         # 거부됨
    MODIFIED = "modified"         # 수정됨
    AUTO_APPROVED = "auto_approved"  # 자동 승인됨


@dataclass
class UserReviewRequest:
    """사용자 검토 요청"""
    query_id: str
    user_query: str
    sql_query: str
    validation_result: ValidationResult
    confidence: float
    estimated_execution_time: float
    estimated_row_count: int
    data_preview: Optional[List[Dict]] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class UserReviewResponse:
    """사용자 검토 응답"""
    query_id: str
    status: ReviewStatus
    user_feedback: Optional[str] = None
    modified_sql: Optional[str] = None
    approved_at: datetime = None
    
    def __post_init__(self):
        if self.approved_at is None and self.status in [ReviewStatus.APPROVED, ReviewStatus.AUTO_APPROVED]:
            self.approved_at = datetime.now()


class UserReviewNode:
    """
    사용자 검토 및 승인을 담당하는 LangGraph 노드
    """
    
    def __init__(self):
        """UserReviewNode 초기화"""
        self.pending_reviews = {}  # query_id -> UserReviewRequest
        self.review_history = []   # UserReviewResponse 리스트
        self.auto_approval_threshold = 0.85  # 자동 승인 임계값
        
        logger.info("UserReviewNode initialized")
    
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        사용자 검토 처리
        
        Args:
            state: LangGraph 상태 딕셔너리
            
        Returns:
            Dict[str, Any]: 업데이트된 상태
        """
        try:
            logger.info("Processing UserReviewNode")
            
            user_query = state.get("user_query", "")
            sql_query = state.get("sql_query", "")
            validation_result = state.get("validation_result")
            processing_decision = state.get("processing_decision", {})
            needs_user_review = processing_decision.get("needs_user_review", False)
            
            logger.info(f"UserReviewNode - needs_user_review: {needs_user_review}")
            
            if not needs_user_review:
                # 자동 승인 처리
                auto_approval = self._handle_auto_approval(state, validation_result)
                state["review_result"] = auto_approval
                state["review_status"] = ReviewStatus.AUTO_APPROVED
                logger.info("Query auto-approved")
                return state
            
            # 사용자 검토 필요
            review_request = self._create_review_request(state, validation_result)
            state["review_request"] = review_request
            state["review_status"] = ReviewStatus.PENDING
            
            # 검토 요청을 대기 목록에 추가
            self.pending_reviews[review_request.query_id] = review_request
            
            logger.info(f"Review request created: {review_request.query_id}")
            logger.info(f"Pending reviews count: {len(self.pending_reviews)}")
            
            return state
            
        except Exception as e:
            logger.error(f"UserReviewNode processing failed: {str(e)}", exc_info=True)
            
            # 오류 시 안전한 폴백
            state["review_status"] = ReviewStatus.PENDING
            state["review_error"] = str(e)
            
            return state
    
    def _handle_auto_approval(self, state: Dict[str, Any], validation_result: ValidationResult) -> UserReviewResponse:
        """자동 승인 처리"""
        try:
            query_id = state.get("query_id", "auto_" + str(datetime.now().timestamp()))
            
            auto_approval = UserReviewResponse(
                query_id=query_id,
                status=ReviewStatus.AUTO_APPROVED,
                user_feedback="자동 승인됨 - 높은 신뢰도"
            )
            
            # 히스토리에 추가
            self.review_history.append(auto_approval)
            
            logger.info(f"Auto-approved query: {query_id}")
            return auto_approval
            
        except Exception as e:
            logger.error(f"Auto-approval failed: {str(e)}")
            return UserReviewResponse(
                query_id=state.get("query_id", "unknown"),
                status=ReviewStatus.PENDING,
                user_feedback=f"자동 승인 실패: {str(e)}"
            )
    
    def _create_review_request(self, state: Dict[str, Any], validation_result: ValidationResult) -> UserReviewRequest:
        """검토 요청 생성"""
        try:
            query_id = state.get("query_id", f"review_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            user_query = state.get("user_query", "")
            sql_query = state.get("sql_query", "")
            
            # 데이터 미리보기 생성 (실제로는 DB에서 샘플 데이터 조회)
            data_preview = self._generate_data_preview(sql_query)
            
            review_request = UserReviewRequest(
                query_id=query_id,
                user_query=user_query,
                sql_query=sql_query,
                validation_result=validation_result,
                confidence=validation_result.confidence,
                estimated_execution_time=validation_result.execution_time_estimate or 1.0,
                estimated_row_count=validation_result.estimated_row_count or 100,
                data_preview=data_preview
            )
            
            return review_request
            
        except Exception as e:
            logger.error(f"Review request creation failed: {str(e)}")
            # 기본 검토 요청 생성
            return UserReviewRequest(
                query_id=state.get("query_id", "error"),
                user_query=state.get("user_query", ""),
                sql_query=state.get("sql_query", ""),
                validation_result=validation_result,
                confidence=0.0,
                estimated_execution_time=1.0,
                estimated_row_count=100
            )
    
    def _generate_data_preview(self, sql_query: str) -> List[Dict]:
        """데이터 미리보기 생성"""
        try:
            # 실제로는 데이터베이스에서 LIMIT 5로 샘플 데이터 조회
            # 여기서는 모의 데이터 생성
            preview_data = []
            
            # SQL 쿼리 타입에 따른 샘플 데이터
            if "COUNT" in sql_query.upper():
                preview_data = [{"count": 1234}]
            elif "SUM" in sql_query.upper():
                preview_data = [{"total_amount": 567890}]
            elif "SELECT" in sql_query.upper():
                # 일반적인 SELECT 쿼리에 대한 샘플
                preview_data = [
                    {"id": 1, "name": "샘플 데이터 1", "value": 100},
                    {"id": 2, "name": "샘플 데이터 2", "value": 200},
                    {"id": 3, "name": "샘플 데이터 3", "value": 300}
                ]
            
            return preview_data
            
        except Exception as e:
            logger.error(f"Data preview generation failed: {str(e)}")
            return []
    
    def submit_user_review(self, query_id: str, status: ReviewStatus, 
                          feedback: Optional[str] = None, modified_sql: Optional[str] = None) -> bool:
        """
        사용자 검토 결과 제출
        
        Args:
            query_id: 쿼리 ID
            status: 검토 상태
            feedback: 사용자 피드백
            modified_sql: 수정된 SQL (있는 경우)
            
        Returns:
            bool: 제출 성공 여부
        """
        try:
            if query_id not in self.pending_reviews:
                logger.warning(f"Review request not found: {query_id}")
                return False
            
            # 검토 응답 생성
            review_response = UserReviewResponse(
                query_id=query_id,
                status=status,
                user_feedback=feedback,
                modified_sql=modified_sql
            )
            
            # 히스토리에 추가
            self.review_history.append(review_response)
            
            # 대기 목록에서 제거
            del self.pending_reviews[query_id]
            
            logger.info(f"User review submitted: {query_id} - {status.value}")
            return True
            
        except Exception as e:
            logger.error(f"User review submission failed: {str(e)}")
            return False
    
    def get_review_request(self, query_id: str) -> Optional[UserReviewRequest]:
        """검토 요청 조회"""
        return self.pending_reviews.get(query_id)
    
    def get_pending_reviews(self) -> Dict[str, UserReviewRequest]:
        """대기 중인 검토 요청 목록"""
        return self.pending_reviews.copy()
    
    def get_review_history(self, limit: int = 50) -> List[UserReviewResponse]:
        """검토 히스토리 조회"""
        return self.review_history[-limit:] if limit > 0 else self.review_history.copy()
    
    def format_review_request_for_user(self, review_request: UserReviewRequest) -> str:
        """사용자용 검토 요청 포맷팅"""
        try:
            formatted_text = f"""
🔍 **쿼리 검토 요청**

**원본 질문**: {review_request.user_query}

**생성된 SQL**:
```sql
{review_request.sql_query}
```

**검증 결과**:
- 신뢰도: {review_request.confidence:.1%}
- 예상 실행 시간: {review_request.estimated_execution_time:.1f}초
- 예상 결과 행 수: {review_request.estimated_row_count:,}행

**발견된 이슈**:
"""
            
            if review_request.validation_result.issues:
                for issue in review_request.validation_result.issues:
                    formatted_text += f"- ❌ {issue}\n"
            else:
                formatted_text += "- ✅ 이슈 없음\n"
            
            formatted_text += "\n**경고사항**:\n"
            if review_request.validation_result.warnings:
                for warning in review_request.validation_result.warnings:
                    formatted_text += f"- ⚠️ {warning}\n"
            else:
                formatted_text += "- ✅ 경고 없음\n"
            
            formatted_text += "\n**제안사항**:\n"
            if review_request.validation_result.suggestions:
                for suggestion in review_request.validation_result.suggestions:
                    formatted_text += f"- 💡 {suggestion}\n"
            else:
                formatted_text += "- ✅ 제안 없음\n"
            
            if review_request.data_preview:
                formatted_text += "\n**데이터 미리보기**:\n"
                formatted_text += "```\n"
                for i, row in enumerate(review_request.data_preview[:3]):
                    formatted_text += f"{i+1}. {row}\n"
                formatted_text += "```\n"
            
            formatted_text += f"\n**쿼리 ID**: {review_request.query_id}"
            formatted_text += f"\n**생성 시간**: {review_request.created_at.strftime('%Y-%m-%d %H:%M:%S')}"
            
            return formatted_text
            
        except Exception as e:
            logger.error(f"Review request formatting failed: {str(e)}")
            return f"검토 요청 포맷팅 실패: {str(e)}"
    
    def clear_old_reviews(self, max_age_hours: int = 24):
        """오래된 검토 요청 정리"""
        try:
            from datetime import datetime, timedelta
            
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            old_query_ids = []
            
            for query_id, request in self.pending_reviews.items():
                if request.created_at < cutoff_time:
                    old_query_ids.append(query_id)
            
            for query_id in old_query_ids:
                del self.pending_reviews[query_id]
            
            if old_query_ids:
                logger.info(f"Cleared {len(old_query_ids)} old review requests")
            
        except Exception as e:
            logger.error(f"Old review cleanup failed: {str(e)}")
    
    def get_review_statistics(self) -> Dict[str, Any]:
        """검토 통계 반환"""
        try:
            total_reviews = len(self.review_history)
            pending_reviews = len(self.pending_reviews)
            
            if total_reviews == 0:
                return {
                    "total_reviews": 0,
                    "pending_reviews": pending_reviews,
                    "approval_rate": 0.0,
                    "auto_approval_rate": 0.0
                }
            
            approved_count = sum(1 for r in self.review_history if r.status == ReviewStatus.APPROVED)
            auto_approved_count = sum(1 for r in self.review_history if r.status == ReviewStatus.AUTO_APPROVED)
            rejected_count = sum(1 for r in self.review_history if r.status == ReviewStatus.REJECTED)
            
            return {
                "total_reviews": total_reviews,
                "pending_reviews": pending_reviews,
                "approved_count": approved_count,
                "auto_approved_count": auto_approved_count,
                "rejected_count": rejected_count,
                "approval_rate": (approved_count + auto_approved_count) / total_reviews,
                "auto_approval_rate": auto_approved_count / total_reviews,
                "rejection_rate": rejected_count / total_reviews
            }
            
        except Exception as e:
            logger.error(f"Review statistics failed: {str(e)}")
            return {
                "total_reviews": 0,
                "pending_reviews": 0,
                "approval_rate": 0.0,
                "auto_approval_rate": 0.0,
                "error": str(e)
            }


