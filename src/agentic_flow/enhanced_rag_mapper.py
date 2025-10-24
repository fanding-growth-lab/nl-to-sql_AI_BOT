#!/usr/bin/env python3
"""
Enhanced RAG Mapper
향상된 RAG 기반 스키마 매핑 시스템
"""

import json
import logging
import re
import time
import os
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
from .sql_schema_analyzer import SQLSchemaAnalyzer
from .sql_schema_mapper import SQLSchemaMapper
from .auto_learning_system import AutoLearningSystem
from .performance_monitor import PerformanceMonitor
from core.db import get_cached_db_schema


class MappingSource(Enum):
    """매핑 소스"""
    RAG_MAPPING = "rag_mapping"
    LLM_GENERATION = "llm_generation"
    HYBRID_APPROACH = "hybrid_approach"
    FALLBACK_MAPPING = "fallback_mapping"


@dataclass
class EnhancedMappingResult:
    """향상된 매핑 결과"""
    matched_pattern: Optional[Dict[str, Any]]
    confidence: float
    source: MappingSource
    sql_template: str
    reasoning: str
    metadata: Dict[str, Any]


class EnhancedRAGMapper:
    """향상된 RAG 기반 스키마 매핑 시스템"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.mapping_patterns = self._load_mapping_patterns()
        self.fallback_mapper = SQLSchemaMapper(config)
        self.schema_analyzer = SQLSchemaAnalyzer(config)
        self.usage_stats = {}
        self.learning_system = AutoLearningSystem()
        self.performance_monitor = PerformanceMonitor(config)
        # 중앙화된 스키마 정보 로드
        self.db_schema = get_cached_db_schema()
    
    def _load_mapping_patterns(self) -> Dict[str, Any]:
        """매핑 패턴 로드"""
        try:
            patterns_file = os.path.join(os.path.dirname(__file__), 'mapping_patterns.json')
            with open(patterns_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading mapping patterns: {str(e)}")
            return {"patterns": [], "statistics": {}}
    
    def map_query_to_schema(self, user_query: str, generated_sql: str = "", context: Dict[str, Any] = None) -> EnhancedMappingResult:
        """
        사용자 쿼리를 스키마에 매핑 (Early Exit 패턴 적용)
        
        Args:
            user_query: 사용자 쿼리
            generated_sql: 생성된 SQL (선택사항)
            context: 추가 컨텍스트 정보
            
        Returns:
            EnhancedMappingResult: 매핑 결과
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"Enhanced RAG mapping query: {user_query}")
            
            # 1. RAG 매핑 시도 (우선순위 기반) - Early Exit
            rag_result = self._try_priority_rag_mapping(user_query, context)
            if rag_result and rag_result.confidence > 0.8:
                # 동적 연도 처리 적용
                rag_result = self._apply_dynamic_year_processing(user_query, rag_result)
                self._update_usage_stats(rag_result.matched_pattern)
                self._record_successful_mapping(user_query, rag_result, context)
                return self._finalize_result(start_time, rag_result, True, rag_result.confidence, context)

            # 2. 하이브리드 접근법 (RAG + LLM) - Early Exit
            hybrid_result = self._try_enhanced_hybrid_mapping(user_query, generated_sql, context)
            if hybrid_result and hybrid_result.confidence > 0.6:
                self._update_usage_stats(hybrid_result.matched_pattern)
                self._record_successful_mapping(user_query, hybrid_result, context)
                return self._finalize_result(start_time, hybrid_result, True, hybrid_result.confidence, context)

            # 3. LLM 폴백 (최종 단계)
            llm_result = self._try_llm_mapping(user_query, generated_sql, context)
            self._update_usage_stats(None)
            self._record_mapping_result(user_query, llm_result, context)
            success = llm_result.confidence > 0.5
            return self._finalize_result(start_time, llm_result, success, llm_result.confidence, context)
            
        except Exception as e:
            self.logger.error(f"Error in enhanced RAG mapping: {str(e)}")
            self._record_mapping_failure(user_query, str(e), context)
            
            error_result = EnhancedMappingResult(
                matched_pattern=None,
                confidence=0.0,
                source=MappingSource.FALLBACK_MAPPING,
                sql_template="",
                reasoning=f"Enhanced RAG mapping failed: {str(e)}",
                metadata={}
            )
            return self._finalize_result(start_time, error_result, False, 0.0, context)
    
    def _apply_dynamic_year_processing(self, user_query: str, rag_result: EnhancedMappingResult) -> EnhancedMappingResult:
        """동적 연도 처리 적용"""
        try:
            from .date_utils import DateUtils
            
            # 쿼리에서 연도 추출
            extracted_year = DateUtils.extract_year_from_query(user_query)
            if not extracted_year:
                return rag_result
            
            # SQL에서 연도 부분을 동적으로 교체
            sql_template = rag_result.sql_template
            
            # 현재 연도를 추출된 연도로 교체
            if "CONCAT(YEAR(NOW()), " in sql_template:
                # CONCAT(YEAR(NOW()), '-09') 형태를 CONCAT('2024', '-09') 형태로 교체
                sql_template = sql_template.replace("CONCAT(YEAR(NOW()), ", f"CONCAT('{extracted_year}', ")
            elif "YEAR(NOW())" in sql_template:
                # YEAR(NOW())를 '2024'로 교체
                sql_template = sql_template.replace("YEAR(NOW())", f"'{extracted_year}'")
            
            # 결과 업데이트
            return EnhancedMappingResult(
                matched_pattern=rag_result.matched_pattern,
                confidence=rag_result.confidence,
                source=rag_result.source,
                sql_template=sql_template,
                reasoning=f"{rag_result.reasoning} (동적 연도 처리: {extracted_year}년 적용)",
                metadata={
                    **rag_result.metadata,
                    "dynamic_year": extracted_year,
                    "original_sql": rag_result.sql_template
                }
            )
        except Exception as e:
            self.logger.warning(f"동적 연도 처리 실패: {str(e)}")
            return rag_result
    
    def _try_priority_rag_mapping(self, user_query: str, context: Dict[str, Any] = None) -> Optional[EnhancedMappingResult]:
        """우선순위 기반 RAG 매핑 시도"""
        user_lower = user_query.lower()
        
        # 우선순위별로 패턴 검색
        priority_order = ["very_high", "high", "medium", "low"]
        
        for priority in priority_order:
            for pattern in self.mapping_patterns.get("patterns", []):
                if pattern.get("priority") == priority:
                    # 정규식 매칭
                    for query_pattern in pattern.get("query_patterns", []):
                        if re.search(query_pattern, user_lower):
                            # SQL 템플릿 생성
                            sql_template = self._generate_enhanced_sql_from_pattern(pattern, user_query, context)
                            
                            return EnhancedMappingResult(
                                matched_pattern=pattern,
                                confidence=pattern.get("confidence", 0.8),
                                source=MappingSource.RAG_MAPPING,
                                sql_template=sql_template,
                                reasoning=f"RAG 매핑 성공: {pattern.get('category', 'unknown')} (우선순위: {priority})",
                                metadata=pattern.get("metadata", {})
                            )
        
        return None
    
    def _try_enhanced_hybrid_mapping(self, user_query: str, generated_sql: str, context: Dict[str, Any] = None) -> Optional[EnhancedMappingResult]:
        """향상된 하이브리드 매핑 시도"""
        # 1. RAG로 기본 구조 파악
        rag_result = self._try_priority_rag_mapping(user_query, context)
        if not rag_result:
            return None
        
        # 2. LLM 결과와 비교하여 보완
        if generated_sql:
            similarity = self._calculate_enhanced_sql_similarity(rag_result.sql_template, generated_sql)
            
            # 유사도 기반 하이브리드 전략
            if similarity > 0.8:
                # 높은 유사도: RAG 우선, LLM으로 세부 보완
                enhanced_sql = self._enhance_sql_with_llm_advanced(rag_result.sql_template, generated_sql, context)
                confidence_boost = 0.05  # 높은 유사도 보너스
                
            elif similarity > 0.6:
                # 중간 유사도: RAG와 LLM 융합
                enhanced_sql = self._fuse_rag_and_llm_sql(rag_result.sql_template, generated_sql, context)
                confidence_boost = 0.0  # 중간 유사도는 보정 없음
                
            elif similarity > 0.4:
                # 낮은 유사도: LLM 우선, RAG로 구조 보완
                enhanced_sql = self._enhance_llm_with_rag_structure(generated_sql, rag_result.sql_template, context)
                confidence_boost = -0.1  # 낮은 유사도 페널티
                
            else:
                # 매우 낮은 유사도: RAG 결과 유지
                enhanced_sql = rag_result.sql_template
                confidence_boost = -0.2  # 매우 낮은 유사도 페널티
            
            return EnhancedMappingResult(
                matched_pattern=rag_result.matched_pattern,
                confidence=max(0.0, min(1.0, rag_result.confidence + confidence_boost)),
                source=MappingSource.HYBRID_APPROACH,
                sql_template=enhanced_sql,
                reasoning=f"하이브리드 매핑: RAG + LLM 융합 (유사도: {similarity:.2f}, 보정: {confidence_boost:+.2f})",
                metadata={
                    **rag_result.metadata,
                    "similarity": similarity,
                    "confidence_boost": confidence_boost,
                    "hybrid_strategy": self._determine_hybrid_strategy(similarity)
                }
            )
        
        return rag_result
    
    def _try_llm_mapping(self, user_query: str, generated_sql: str, context: Dict[str, Any] = None) -> EnhancedMappingResult:
        """LLM 폴백 매핑"""
        # 기존 SQLSchemaMapper 사용
        db_schema = self._get_enhanced_schema()
        mapping_result = self.fallback_mapper.map_sql_to_schema(generated_sql, db_schema, user_query)
        
        return EnhancedMappingResult(
            matched_pattern=None,
            confidence=mapping_result.confidence,
            source=MappingSource.LLM_GENERATION,
            sql_template=mapping_result.mapped_sql,
            reasoning="LLM 폴백 매핑",
            metadata={"fallback": True}
        )
    
    def _generate_enhanced_sql_from_pattern(self, pattern: Dict[str, Any], user_query: str, context: Dict[str, Any] = None) -> str:
        """패턴에서 향상된 SQL 생성"""
        sql_template = pattern.get("sql_template", "")
        
        # 월 추출 (9월, 8월 등)
        month_match = re.search(r'(\d+)월', user_query)
        month = month_match.group(1) if month_match else "9"
        
        # 연도 추출 (2024년, 2025년 등)
        year_match = re.search(r'(\d{4})년', user_query)
        year = year_match.group(1) if year_match else "2025"
        
        # TOP N 추출 (상위 5, TOP 10 등)
        top_n_match = re.search(r'(?:상위|top|탑)\s*(\d+)', user_query.lower())
        top_n = top_n_match.group(1) if top_n_match else "5"
        
        # SQL 템플릿 변수 치환 (안전한 치환)
        try:
            sql_template = sql_template.replace("{month}", month.zfill(2))
            sql_template = sql_template.replace("{year}", year)
            sql_template = sql_template.replace("{top_n}", top_n)
            
            # 추가 파라미터들도 처리 (향후 확장성)
            if "{days}" in sql_template:
                days_match = re.search(r'(\d+)일', user_query)
                days = days_match.group(1) if days_match else "30"
                sql_template = sql_template.replace("{days}", days)
                
            if "{months}" in sql_template:
                months_match = re.search(r'(\d+)개?월', user_query)
                months = months_match.group(1) if months_match else "12"
                sql_template = sql_template.replace("{months}", months)
                
        except Exception as e:
            self.logger.warning(f"SQL 템플릿 파라미터 치환 실패: {str(e)}")
            # 기본값으로 폴백
            sql_template = sql_template.replace("{month}", "09")
            sql_template = sql_template.replace("{year}", "2024")
            sql_template = sql_template.replace("{top_n}", "5")
        
        # 컨텍스트 기반 추가 조정
        if context:
            # 사용자 선호도 반영
            if context.get("prefer_detailed", False):
                sql_template = self._add_detailed_columns(sql_template)
            
            # 성능 최적화
            if context.get("optimize_performance", False):
                sql_template = self._add_performance_optimization(sql_template)
        
        return sql_template
    
    def _calculate_enhanced_sql_similarity(self, sql1: str, sql2: str) -> float:
        """향상된 SQL 유사도 계산"""
        # 키워드 기반 유사도
        keywords1 = set(re.findall(r'\b\w+\b', sql1.upper()))
        keywords2 = set(re.findall(r'\b\w+\b', sql2.upper()))
        
        intersection = len(keywords1.intersection(keywords2))
        union = len(keywords1.union(keywords2))
        
        base_similarity = intersection / union if union > 0 else 0.0
        
        # 구조적 유사도 추가
        structure1 = self._extract_sql_structure(sql1)
        structure2 = self._extract_sql_structure(sql2)
        
        structure_similarity = len(structure1.intersection(structure2)) / len(structure1.union(structure2)) if structure1.union(structure2) else 0.0
        
        # 가중 평균
        return (base_similarity * 0.7) + (structure_similarity * 0.3)
    
    def _extract_sql_structure(self, sql: str) -> set:
        """SQL 구조 추출"""
        structure = set()
        sql_upper = sql.upper()
        
        if 'SELECT' in sql_upper:
            structure.add('SELECT')
        if 'FROM' in sql_upper:
            structure.add('FROM')
        if 'WHERE' in sql_upper:
            structure.add('WHERE')
        if 'GROUP BY' in sql_upper:
            structure.add('GROUP_BY')
        if 'ORDER BY' in sql_upper:
            structure.add('ORDER_BY')
        if 'COUNT' in sql_upper:
            structure.add('COUNT')
        if 'CASE' in sql_upper:
            structure.add('CASE')
        
        return structure
    
    def _enhance_sql_with_llm_advanced(self, rag_sql: str, llm_sql: str, context: Dict[str, Any] = None) -> str:
        """RAG SQL을 LLM 결과로 고급 보완 (높은 유사도)"""
        # RAG SQL을 기본으로 하고, LLM의 세부 사항만 선택적으로 통합
        enhanced_sql = rag_sql
        
        # LLM에서 더 나은 컬럼명이나 조건 추출
        llm_columns = self._extract_columns_from_sql(llm_sql)
        rag_columns = self._extract_columns_from_sql(rag_sql)
        
        # LLM의 더 정확한 컬럼명이 있으면 적용
        for llm_col in llm_columns:
            if llm_col not in rag_columns and self._is_valid_column(llm_col):
                enhanced_sql = self._add_column_to_sql(enhanced_sql, llm_col)
        
        return enhanced_sql
    
    def _fuse_rag_and_llm_sql(self, rag_sql: str, llm_sql: str, context: Dict[str, Any] = None) -> str:
        """RAG와 LLM SQL 융합 (중간 유사도)"""
        # 두 SQL의 장점을 결합
        rag_structure = self._extract_sql_structure(rag_sql)
        llm_structure = self._extract_sql_structure(llm_sql)
        
        # RAG의 안정적인 구조 + LLM의 동적 요소
        if 'SELECT' in rag_structure and 'FROM' in rag_structure:
            # RAG 구조 유지
            base_sql = rag_sql
        else:
            # LLM 구조 사용
            base_sql = llm_sql
        
        # 컨텍스트 기반 최적화
        if context and context.get("prefer_detailed", False):
            base_sql = self._add_detailed_columns(base_sql)
        
        return base_sql
    
    def _enhance_llm_with_rag_structure(self, llm_sql: str, rag_sql: str, context: Dict[str, Any] = None) -> str:
        """LLM SQL을 RAG 구조로 보완 (낮은 유사도)"""
        # LLM SQL을 기본으로 하고, RAG의 안정적인 구조 요소만 적용
        enhanced_sql = llm_sql
        
        # RAG의 안정적인 테이블명이나 기본 구조 적용
        rag_tables = self._extract_tables_from_sql(rag_sql)
        llm_tables = self._extract_tables_from_sql(llm_sql)
        
        # RAG의 검증된 테이블명이 있으면 적용
        for rag_table in rag_tables:
            if rag_table not in llm_tables and self._is_valid_table(rag_table):
                enhanced_sql = self._replace_table_in_sql(enhanced_sql, llm_tables[0] if llm_tables else "t_member", rag_table)
        
        return enhanced_sql
    
    def _determine_hybrid_strategy(self, similarity: float) -> str:
        """유사도 기반 하이브리드 전략 결정"""
        if similarity > 0.8:
            return "rag_priority_with_llm_enhancement"
        elif similarity > 0.6:
            return "balanced_fusion"
        elif similarity > 0.4:
            return "llm_priority_with_rag_structure"
        else:
            return "rag_fallback"
    
    def _extract_columns_from_sql(self, sql: str) -> List[str]:
        """SQL에서 컬럼명 추출"""
        import re
        # SELECT 절에서 컬럼명 추출
        select_match = re.search(r'SELECT\s+(.*?)\s+FROM', sql, re.IGNORECASE | re.DOTALL)
        if select_match:
            columns_text = select_match.group(1)
            # 쉼표로 분리하고 별칭 제거
            columns = []
            for col in columns_text.split(','):
                col = col.strip()
                if ' as ' in col.lower():
                    col = col.split(' as ')[0].strip()
                if col and col != '*':
                    columns.append(col)
            return columns
        return []
    
    def _extract_tables_from_sql(self, sql: str) -> List[str]:
        """SQL에서 테이블명 추출"""
        import re
        # FROM 절에서 테이블명 추출
        from_match = re.search(r'FROM\s+(\w+)', sql, re.IGNORECASE)
        if from_match:
            return [from_match.group(1)]
        return []
    
    def _is_valid_column(self, column: str) -> bool:
        """컬럼명 유효성 검사 (동적 스키마 정보 활용)"""
        try:
            # 모든 테이블의 컬럼 정보를 수집
            all_columns = set()
            for table_name, table_info in self.db_schema.items():
                columns = table_info.get("columns", {})
                all_columns.update(columns.keys())
            
            return column.lower() in [col.lower() for col in all_columns]
        except Exception as e:
            self.logger.warning(f"컬럼 유효성 검사 실패: {str(e)}")
            # 폴백: 실제 존재하는 주요 컬럼명들로 검사
            fallback_columns = ["no", "member_no", "creator_no", "status", "c_email", "nickname", "price", "heat", "view_count", "like_count", "ins_datetime"]
            return column.lower() in fallback_columns
    
    def _is_valid_table(self, table: str) -> bool:
        """테이블명 유효성 검사 (동적 스키마 정보 활용)"""
        try:
            # 실제 DB에 존재하는 테이블명들로 검사
            actual_tables = set(self.db_schema.keys())
            return table.lower() in [t.lower() for t in actual_tables]
        except Exception as e:
            self.logger.warning(f"테이블 유효성 검사 실패: {str(e)}")
            # 폴백: 실제 존재하는 테이블명들로 검사
            fallback_tables = ["t_member", "t_creator", "t_payment", "t_post", "t_member_login_log", "t_collection", "t_tier", "t_project"]
            return table.lower() in fallback_tables
    
    def _add_column_to_sql(self, sql: str, column: str) -> str:
        """SQL에 컬럼 추가"""
        # 간단한 구현: SELECT 절에 컬럼 추가
        if 'SELECT' in sql.upper():
            sql = sql.replace('SELECT', f'SELECT {column},')
        return sql
    
    def _replace_table_in_sql(self, sql: str, old_table: str, new_table: str) -> str:
        """SQL에서 테이블명 교체"""
        return sql.replace(old_table, new_table)
    
    def _record_successful_mapping(self, user_query: str, result: EnhancedMappingResult, context: Dict[str, Any] = None):
        """성공적인 매핑 기록"""
        try:
            user_id = context.get("user_id", "anonymous") if context else "anonymous"
            self.learning_system.record_query_interaction(
                user_id=user_id,
                query=user_query,
                mapping_result={
                    "source": result.source.value,
                    "confidence": result.confidence,
                    "sql_template": result.sql_template,
                    "reasoning": result.reasoning
                },
                confidence=result.confidence,
                success=True,
                user_feedback=context.get("user_feedback") if context else None
            )
        except Exception as e:
            self.logger.warning(f"Failed to record successful mapping: {str(e)}")
    
    def _record_mapping_result(self, user_query: str, result: EnhancedMappingResult, context: Dict[str, Any] = None):
        """매핑 결과 기록"""
        try:
            user_id = context.get("user_id", "anonymous") if context else "anonymous"
            success = result.confidence > 0.5  # 신뢰도 기반 성공 판단
            self.learning_system.record_query_interaction(
                user_id=user_id,
                query=user_query,
                mapping_result={
                    "source": result.source.value,
                    "confidence": result.confidence,
                    "sql_template": result.sql_template,
                    "reasoning": result.reasoning
                },
                confidence=result.confidence,
                success=success,
                user_feedback=context.get("user_feedback") if context else None
            )
        except Exception as e:
            self.logger.warning(f"Failed to record mapping result: {str(e)}")
    
    def _record_mapping_failure(self, user_query: str, error: str, context: Dict[str, Any] = None):
        """매핑 실패 기록"""
        try:
            user_id = context.get("user_id", "anonymous") if context else "anonymous"
            self.learning_system.record_query_interaction(
                user_id=user_id,
                query=user_query,
                mapping_result={"error": error},
                confidence=0.0,
                success=False,
                user_feedback=context.get("user_feedback") if context else None
            )
        except Exception as e:
            self.logger.warning(f"Failed to record mapping failure: {str(e)}")
    
    def get_learning_insights(self) -> Dict[str, Any]:
        """학습 인사이트 조회"""
        try:
            return self.learning_system.get_learning_report()
        except Exception as e:
            self.logger.error(f"Failed to get learning insights: {str(e)}")
            return {"error": str(e)}
    
    def optimize_based_on_learning(self) -> Dict[str, Any]:
        """학습 기반 최적화"""
        try:
            # 학습 패턴 분석
            analysis = self.learning_system.analyze_learning_patterns()
            
            # 개선 제안 생성
            suggestions = self.learning_system.generate_improvement_suggestions()
            
            # 매핑 패턴 최적화
            optimizations = self.learning_system.optimize_mapping_patterns()
            
            return {
                "analysis": analysis,
                "suggestions": suggestions,
                "optimizations": optimizations,
                "recommendations": self._generate_optimization_recommendations(analysis, suggestions)
            }
        except Exception as e:
            self.logger.error(f"Failed to optimize based on learning: {str(e)}")
            return {"error": str(e)}
    
    def _generate_optimization_recommendations(self, analysis: Dict[str, Any], suggestions: List[str]) -> List[str]:
        """최적화 권장사항 생성"""
        recommendations = []
        
        try:
            # 성공률 기반 권장사항
            if analysis.get("overall_success_rate", 0) < 0.7:
                recommendations.append("전체 성공률이 낮습니다. 매핑 로직 개선이 필요합니다.")
            
            # 패턴 사용량 기반 권장사항
            top_patterns = analysis.get("top_patterns", [])
            if len(top_patterns) > 0:
                most_used = top_patterns[0]
                if most_used.get("frequency", 0) > 50:
                    recommendations.append(f"패턴 '{most_used['pattern']}'이 자주 사용됩니다. 우선순위를 높이는 것을 고려하세요.")
            
            # 개선 영역 기반 권장사항
            improvement_areas = analysis.get("improvement_areas", [])
            if improvement_areas:
                recommendations.append(f"{len(improvement_areas)}개의 패턴이 개선이 필요합니다. 매핑 정확도를 높이세요.")
            
            # 사용자 인사이트 기반 권장사항
            user_insights = analysis.get("user_insights", {})
            if user_insights:
                high_success_users = [uid for uid, data in user_insights.items() if data.get("success_rate", 0) > 0.8]
                if high_success_users:
                    recommendations.append(f"성공률이 높은 사용자 {len(high_success_users)}명의 패턴을 분석하여 다른 사용자에게 적용하세요.")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Failed to generate optimization recommendations: {str(e)}")
            return ["최적화 권장사항 생성 중 오류가 발생했습니다."]
    
    def _record_quality_metrics(self, user_query: str, result: EnhancedMappingResult, context: Dict[str, Any] = None):
        """품질 메트릭 기록"""
        try:
            # 기본 품질 메트릭 계산
            accuracy = result.confidence  # 신뢰도를 정확도로 사용
            precision = result.confidence  # 신뢰도를 정밀도로 사용
            recall = result.confidence  # 신뢰도를 재현율로 사용
            f1_score = result.confidence  # 신뢰도를 F1 점수로 사용
            
            # 사용자 만족도 (컨텍스트에서 추출)
            user_satisfaction = context.get("user_satisfaction") if context else None
            
            # 성능 모니터에 품질 메트릭 기록
            self.performance_monitor.record_quality_metric(
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1_score,
                confidence=result.confidence,
                user_satisfaction=user_satisfaction
            )
            
        except Exception as e:
            self.logger.warning(f"품질 메트릭 기록 실패: {str(e)}")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """성능 리포트 조회"""
        try:
            return self.performance_monitor.generate_performance_report()
        except Exception as e:
            self.logger.error(f"성능 리포트 조회 실패: {str(e)}")
            return {"error": str(e)}
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """성능 요약 조회"""
        try:
            return self.performance_monitor.get_performance_summary()
        except Exception as e:
            self.logger.error(f"성능 요약 조회 실패: {str(e)}")
            return {"error": str(e)}
    
    def get_quality_summary(self) -> Dict[str, Any]:
        """품질 요약 조회"""
        try:
            return self.performance_monitor.get_quality_summary()
        except Exception as e:
            self.logger.error(f"품질 요약 조회 실패: {str(e)}")
            return {"error": str(e)}
    
    def get_system_health(self) -> str:
        """시스템 상태 조회"""
        try:
            return self.performance_monitor.get_system_health()
        except Exception as e:
            self.logger.error(f"시스템 상태 조회 실패: {str(e)}")
            return "error"
    
    def get_recent_alerts(self, hours: int = 24) -> List[Dict[str, Any]]:
        """최근 알림 조회"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_alerts = [
                {
                    "timestamp": alert.timestamp.isoformat(),
                    "level": alert.level.value,
                    "message": alert.message,
                    "metric_type": alert.metric_type.value,
                    "current_value": alert.current_value,
                    "threshold": alert.threshold
                }
                for alert in self.performance_monitor.alerts
                if alert.timestamp >= cutoff_time
            ]
            return recent_alerts
        except Exception as e:
            self.logger.error(f"최근 알림 조회 실패: {str(e)}")
            return []
    
    def export_performance_data(self, filepath: str):
        """성능 데이터 내보내기"""
        try:
            self.performance_monitor.export_metrics(filepath)
            self.logger.info(f"성능 데이터 내보내기 완료: {filepath}")
        except Exception as e:
            self.logger.error(f"성능 데이터 내보내기 실패: {str(e)}")
    
    def clear_old_performance_data(self, days: int = 7):
        """오래된 성능 데이터 정리"""
        try:
            self.performance_monitor.clear_old_data(days)
            self.logger.info(f"{days}일 이상 된 성능 데이터 정리 완료")
        except Exception as e:
            self.logger.error(f"성능 데이터 정리 실패: {str(e)}")
    
    def stop_monitoring(self):
        """모니터링 중지"""
        try:
            self.performance_monitor.stop_monitoring()
            self.logger.info("성능 모니터링 중지")
        except Exception as e:
            self.logger.error(f"모니터링 중지 실패: {str(e)}")
    
    def _add_detailed_columns(self, sql: str) -> str:
        """상세 컬럼 추가"""
        # 기본 SQL에 추가 분석 컬럼 포함
        if "COUNT(*)" in sql and "CASE WHEN" not in sql:
            # 이미 as 키워드가 있는지 확인
            if "COUNT(*) as " in sql:
                # 이미 COUNT(*)에 as가 있으면 건드리지 않음
                pass
            else:
                # COUNT(*)만 있고 as가 없으면 추가
                sql = sql.replace("COUNT(*)", "COUNT(*) as total_count")
        return sql
    
    def _add_performance_optimization(self, sql: str) -> str:
        """성능 최적화 추가"""
        # LIMIT 추가, 인덱스 힌트 등
        if "FROM t_member" in sql and "LIMIT" not in sql:
            sql += " LIMIT 1000"
        return sql
    
    def _get_enhanced_schema(self) -> Dict[str, Any]:
        """향상된 스키마 정보 (동적 스키마 사용)"""
        try:
            # 중앙화된 스키마 정보 사용
            return self.db_schema
        except Exception as e:
            self.logger.warning(f"동적 스키마 로드 실패, 폴백 스키마 사용: {str(e)}")
            # 폴백: 기본 스키마 정보
            return {
                "t_member": {
                    "description": "회원 테이블",
                    "columns": {
                        "no": {"type": "INTEGER", "description": "회원번호", "nullable": False},
                        "c_email": {"type": "VARCHAR(190)", "description": "이메일", "nullable": True},
                        "status": {"type": "CHAR(1)", "description": "상태 (A:가입완료, J:인증대기, D:회원탈퇴)", "nullable": False},
                        "nickname": {"type": "VARCHAR(50)", "description": "닉네임", "nullable": True}
                    }
                },
                "t_creator": {
                    "description": "크리에이터 테이블",
                    "columns": {
                        "no": {"type": "INTEGER", "description": "크리에이터 번호", "nullable": False},
                        "member_no": {"type": "INTEGER", "description": "회원 번호", "nullable": False}
                    }
                }
            }
    
    def _update_usage_stats(self, matched_pattern: Optional[Dict[str, Any]]):
        """사용 통계 업데이트"""
        if matched_pattern:
            pattern_id = matched_pattern.get("id", "unknown")
            if pattern_id not in self.usage_stats:
                self.usage_stats[pattern_id] = 0
            self.usage_stats[pattern_id] += 1
    
    def get_enhanced_mapping_stats(self) -> Dict[str, Any]:
        """향상된 매핑 통계"""
        patterns = self.mapping_patterns.get("patterns", [])
        categories = {}
        priorities = {}
        
        for pattern in patterns:
            category = pattern.get("category", "unknown")
            priority = pattern.get("priority", "unknown")
            
            if category not in categories:
                categories[category] = 0
            categories[category] += 1
            
            if priority not in priorities:
                priorities[priority] = 0
            priorities[priority] += 1
        
        return {
            "total_patterns": len(patterns),
            "categories": categories,
            "priorities": priorities,
            "usage_stats": self.usage_stats,
            "avg_confidence": sum(pattern.get("confidence", 0) for pattern in patterns) / len(patterns) if patterns else 0,
            "coverage_estimate": self.mapping_patterns.get("statistics", {}).get("coverage_estimate", "unknown")
        }
    
    def add_custom_pattern(self, pattern: Dict[str, Any]):
        """사용자 정의 패턴 추가"""
        if "patterns" not in self.mapping_patterns:
            self.mapping_patterns["patterns"] = []
        
        self.mapping_patterns["patterns"].append(pattern)
        self.logger.info(f"Added custom pattern: {pattern.get('id', 'unknown')}")
    
    def optimize_patterns(self):
        """패턴 최적화"""
        # 사용 통계 기반으로 패턴 우선순위 조정
        for pattern in self.mapping_patterns.get("patterns", []):
            pattern_id = pattern.get("id")
            if pattern_id in self.usage_stats:
                usage_count = self.usage_stats[pattern_id]
                if usage_count > 10:  # 자주 사용되는 패턴
                    pattern["priority"] = "very_high"
                elif usage_count > 5:
                    pattern["priority"] = "high"
        
        self.logger.info("Patterns optimized based on usage statistics")
    
    def _finalize_result(self, start_time: float, result: EnhancedMappingResult, success: bool, confidence: float, context: Dict[str, Any] = None) -> EnhancedMappingResult:
        """결과 최종화 및 성능 모니터링 기록"""
        try:
            end_time = time.time()
            
            # 성능 모니터링에 요청 기록
            self.performance_monitor.record_request(
                start_time=start_time,
                end_time=end_time,
                success=success,
                confidence=confidence,
                context=context
            )
            
            # 품질 메트릭 기록 (성공한 경우)
            if success and confidence > 0:
                self._record_quality_metrics("", result, context)
            
            return result
            
        except Exception as e:
            self.logger.error(f"결과 최종화 실패: {str(e)}")
            return result
