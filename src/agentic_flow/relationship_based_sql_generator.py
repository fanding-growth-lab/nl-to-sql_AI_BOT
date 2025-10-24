#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
관계 기반 SQL 생성 템플릿 시스템
외래키 관계, 조인 경로, 엔티티 추출을 활용한 고급 SQL 생성 시스템
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
from agentic_flow.foreign_key_mapper import ForeignKeyMapper
from agentic_flow.join_path_optimizer import JoinPathOptimizer, OptimizationStrategy
from agentic_flow.entity_relationship_extractor import EntityRelationshipExtractor, QueryIntent
from core.db import get_db_session, execute_query


class SQLTemplateType(Enum):
    """SQL 템플릿 유형"""
    SIMPLE_SELECT = "simple_select"
    COMPLEX_JOIN = "complex_join"
    AGGREGATION = "aggregation"
    STATISTICS = "statistics"
    TREND_ANALYSIS = "trend_analysis"
    RANKING = "ranking"
    COMPARISON = "comparison"


@dataclass
class SQLTemplate:
    """SQL 템플릿"""
    template_type: SQLTemplateType
    name: str
    description: str
    sql_template: str
    required_entities: List[str]
    required_relationships: List[str]
    parameters: Dict[str, Any] = field(default_factory=dict)
    complexity_score: float = 0.0
    confidence: float = 1.0


@dataclass
class GeneratedSQL:
    """생성된 SQL"""
    sql: str
    template_used: str
    confidence: float
    estimated_rows: int
    estimated_time: float
    complexity_score: float
    optimization_notes: List[str]
    parameters: Dict[str, Any] = field(default_factory=dict)


class RelationshipBasedSQLGenerator:
    """관계 기반 SQL 생성기"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 하위 시스템 초기화
        self.fk_mapper = ForeignKeyMapper()
        self.join_optimizer = JoinPathOptimizer(self.fk_mapper)
        self.entity_extractor = EntityRelationshipExtractor()
        
        # SQL 템플릿 초기화
        self.sql_templates = self._initialize_sql_templates()
        
        # 외래키 관계 탐지
        self.fk_mapper.discover_foreign_keys()
        
    def _initialize_sql_templates(self) -> Dict[str, SQLTemplate]:
        """SQL 템플릿 초기화"""
        templates = {}
        
        # 1. 기본 조회 템플릿
        templates["simple_member_query"] = SQLTemplate(
            template_type=SQLTemplateType.SIMPLE_SELECT,
            name="회원 기본 조회",
            description="회원 정보를 기본적으로 조회하는 템플릿",
            sql_template="""
            SELECT 
                m.no as member_id,
                m.nickname,
                m.email,
                m.status,
                m.created_at
            FROM t_member m
            {where_clause}
            {order_clause}
            {limit_clause}
            """,
            required_entities=["회원"],
            required_relationships=[],
            parameters={
                "where_clause": "",
                "order_clause": "ORDER BY m.created_at DESC",
                "limit_clause": "LIMIT 100"
            }
        )
        
        # 2. 회원-크리에이터 조인 템플릿
        templates["member_creator_join"] = SQLTemplate(
            template_type=SQLTemplateType.COMPLEX_JOIN,
            name="회원-크리에이터 조인",
            description="회원과 크리에이터 정보를 조인하여 조회",
            sql_template="""
            SELECT 
                m.no as member_id,
                m.nickname as member_name,
                c.no as creator_id,
                c.creator_name,
                c.is_active as creator_active,
                m.created_at as member_created,
                c.created_at as creator_created
            FROM t_member m
            LEFT JOIN t_creator c ON m.no = c.member_no
            {where_clause}
            {order_clause}
            {limit_clause}
            """,
            required_entities=["회원", "크리에이터"],
            required_relationships=["회원-크리에이터"],
            parameters={
                "where_clause": "",
                "order_clause": "ORDER BY m.created_at DESC",
                "limit_clause": "LIMIT 100"
            }
        )
        
        # 3. 포스트 통계 템플릿
        templates["post_statistics"] = SQLTemplate(
            template_type=SQLTemplateType.AGGREGATION,
            name="포스트 통계",
            description="포스트 관련 통계 정보를 조회",
            sql_template="""
            SELECT 
                COUNT(*) as total_posts,
                COUNT(CASE WHEN p.status = 'A' THEN 1 END) as active_posts,
                COUNT(CASE WHEN p.status = 'D' THEN 1 END) as deleted_posts,
                AVG(p.like_count) as avg_likes,
                MAX(p.like_count) as max_likes,
                MIN(p.created_at) as earliest_post,
                MAX(p.created_at) as latest_post
            FROM t_post p
            {where_clause}
            """,
            required_entities=["포스트"],
            required_relationships=[],
            parameters={
                "where_clause": ""
            }
        )
        
        # 4. 회원 활동 분석 템플릿
        templates["member_activity_analysis"] = SQLTemplate(
            template_type=SQLTemplateType.STATISTICS,
            name="회원 활동 분석",
            description="회원의 활동 패턴을 분석하는 템플릿",
            sql_template="""
            SELECT 
                m.no as member_id,
                m.nickname,
                COUNT(DISTINCT p.no) as post_count,
                COUNT(DISTINCT pr.no) as reply_count,
                COUNT(DISTINCT pl.member_no) as like_count,
                MAX(p.created_at) as last_post_date,
                MAX(pr.created_at) as last_reply_date
            FROM t_member m
            LEFT JOIN t_post p ON m.no = p.member_no
            LEFT JOIN t_post_reply pr ON m.no = pr.member_no
            LEFT JOIN t_post_like_log pl ON m.no = pl.member_no
            {where_clause}
            GROUP BY m.no, m.nickname
            {order_clause}
            {limit_clause}
            """,
            required_entities=["회원", "포스트"],
            required_relationships=["회원-포스트"],
            parameters={
                "where_clause": "",
                "order_clause": "ORDER BY post_count DESC",
                "limit_clause": "LIMIT 50"
            }
        )
        
        # 5. 시간별 트렌드 분석 템플릿
        templates["trend_analysis"] = SQLTemplate(
            template_type=SQLTemplateType.TREND_ANALYSIS,
            name="시간별 트렌드 분석",
            description="시간별 데이터 트렌드를 분석하는 템플릿",
            sql_template="""
            SELECT 
                DATE_FORMAT({date_column}, '%Y-%m') as period,
                COUNT(*) as count,
                {aggregation_columns}
            FROM {main_table}
            {join_clauses}
            WHERE {date_column} >= DATE_SUB(NOW(), INTERVAL {period_days} DAY)
            {where_clause}
            GROUP BY DATE_FORMAT({date_column}, '%Y-%m')
            ORDER BY period DESC
            """,
            required_entities=["시간"],
            required_relationships=[],
            parameters={
                "date_column": "created_at",
                "main_table": "t_member",
                "join_clauses": "",
                "period_days": "30",
                "aggregation_columns": "COUNT(*) as total_count",
                "where_clause": ""
            }
        )
        
        # 6. 랭킹 분석 템플릿
        templates["ranking_analysis"] = SQLTemplate(
            template_type=SQLTemplateType.RANKING,
            name="랭킹 분석",
            description="데이터의 랭킹을 분석하는 템플릿",
            sql_template="""
            SELECT 
                {ranking_columns},
                ROW_NUMBER() OVER (ORDER BY {ranking_criteria} DESC) as ranking,
                {ranking_criteria} as score
            FROM {main_table}
            {join_clauses}
            {where_clause}
            ORDER BY {ranking_criteria} DESC
            LIMIT {top_count}
            """,
            required_entities=["랭킹"],
            required_relationships=[],
            parameters={
                "ranking_columns": "name, id",
                "ranking_criteria": "like_count",
                "main_table": "t_post",
                "join_clauses": "",
                "where_clause": "",
                "top_count": "10"
            }
        )
        
        return templates
    
    def generate_sql_from_query(self, query: str) -> GeneratedSQL:
        """자연어 쿼리로부터 SQL 생성"""
        self.logger.info(f"SQL 생성 시작: {query}")
        
        try:
            # 1. 엔티티와 관계 추출
            query_intent = self.entity_extractor.extract_entities_and_relationships(query)
            
            # 2. 적합한 템플릿 선택
            selected_template = self._select_best_template(query_intent)
            
            if not selected_template:
                return self._generate_fallback_sql(query, query_intent)
            
            # 3. 템플릿 매개변수 설정
            parameters = self._configure_template_parameters(selected_template, query_intent)
            
            # 4. SQL 생성
            generated_sql = self._generate_sql_from_template(selected_template, parameters)
            
            # 5. 조인 경로 최적화
            optimized_sql = self._optimize_join_paths(generated_sql, query_intent)
            
            # 6. 성능 메트릭 계산
            performance_metrics = self._calculate_performance_metrics(optimized_sql)
            
            result = GeneratedSQL(
                sql=optimized_sql,
                template_used=selected_template.name,
                confidence=query_intent.confidence,
                estimated_rows=performance_metrics.get("estimated_rows", 1000),
                estimated_time=performance_metrics.get("estimated_time", 1.0),
                complexity_score=performance_metrics.get("complexity_score", 0.5),
                optimization_notes=performance_metrics.get("notes", []),
                parameters=parameters
            )
            
            self.logger.info(f"SQL 생성 완료: {selected_template.name}")
            return result
            
        except Exception as e:
            self.logger.error(f"SQL 생성 실패: {str(e)}")
            return self._generate_error_sql(str(e))
    
    def _select_best_template(self, query_intent: QueryIntent) -> Optional[SQLTemplate]:
        """최적의 템플릿 선택"""
        if not query_intent.entities:
            return None
        
        # 엔티티 기반 템플릿 매칭
        entity_names = [e.name for e in query_intent.entities]
        
        # 의도별 템플릿 선택
        if query_intent.intent_type == "통계":
            if "회원" in entity_names and "포스트" in entity_names:
                return self.sql_templates["member_activity_analysis"]
            elif "포스트" in entity_names:
                return self.sql_templates["post_statistics"]
        
        elif query_intent.intent_type == "랭킹":
            return self.sql_templates["ranking_analysis"]
        
        elif query_intent.intent_type == "트렌드":
            return self.sql_templates["trend_analysis"]
        
        elif query_intent.intent_type == "조회":
            if "회원" in entity_names and "크리에이터" in entity_names:
                return self.sql_templates["member_creator_join"]
            elif "회원" in entity_names:
                return self.sql_templates["simple_member_query"]
        
        # 기본 템플릿
        return self.sql_templates["simple_member_query"]
    
    def _configure_template_parameters(self, template: SQLTemplate, query_intent: QueryIntent) -> Dict[str, Any]:
        """템플릿 매개변수 설정"""
        parameters = template.parameters.copy()
        
        # WHERE 절 설정
        where_conditions = []
        
        # 시간 필터 적용
        for time_filter in query_intent.time_filters:
            if time_filter["type"] == "month":
                where_conditions.append(f"MONTH(created_at) = {time_filter['value']}")
            elif time_filter["type"] == "recent_days":
                where_conditions.append(f"created_at >= DATE_SUB(NOW(), INTERVAL {time_filter['value']} DAY)")
        
        # 조건 적용
        where_conditions.extend(query_intent.conditions)
        
        if where_conditions:
            parameters["where_clause"] = f"WHERE {' AND '.join(where_conditions)}"
        
        # ORDER BY 절 설정
        if query_intent.intent_type == "랭킹":
            parameters["order_clause"] = "ORDER BY score DESC"
        
        # LIMIT 절 설정
        if query_intent.intent_type == "랭킹":
            parameters["limit_clause"] = "LIMIT 10"
        elif query_intent.intent_type == "조회":
            parameters["limit_clause"] = "LIMIT 100"
        
        return parameters
    
    def _generate_sql_from_template(self, template: SQLTemplate, parameters: Dict[str, Any]) -> str:
        """템플릿으로부터 SQL 생성"""
        sql = template.sql_template
        
        # 매개변수 치환
        for key, value in parameters.items():
            placeholder = f"{{{key}}}"
            sql = sql.replace(placeholder, str(value))
        
        return sql.strip()
    
    def _optimize_join_paths(self, sql: str, query_intent: QueryIntent) -> str:
        """조인 경로 최적화"""
        # SQL에서 테이블 추출
        tables = self._extract_tables_from_sql(sql)
        
        if len(tables) < 2:
            return sql
        
        # 최적 조인 경로 찾기
        try:
            result = self.join_optimizer.find_optimal_join_path(
                [tables[0]], 
                tables[1:], 
                OptimizationStrategy.COST_BASED
            )
            
            if result.best_path:
                # 최적화된 SQL 생성
                optimized_sql = self.join_optimizer.generate_sql_from_path(
                    result.best_path,
                    select_columns=["*"],
                    limit=100
                )
                return optimized_sql
                
        except Exception as e:
            self.logger.warning(f"조인 경로 최적화 실패: {str(e)}")
        
        return sql
    
    def _extract_tables_from_sql(self, sql: str) -> List[str]:
        """SQL에서 테이블명 추출"""
        import re
        
        # FROM, JOIN 절에서 테이블명 추출
        table_pattern = r'(?:FROM|JOIN)\s+(\w+)'
        tables = re.findall(table_pattern, sql.upper())
        
        return list(set(tables))  # 중복 제거
    
    def _calculate_performance_metrics(self, sql: str) -> Dict[str, Any]:
        """성능 메트릭 계산"""
        try:
            # 쿼리 복잡도 분석
            analysis = self.join_optimizer.analyze_query_complexity(sql)
            
            return {
                "estimated_rows": 1000,  # 기본값
                "estimated_time": 1.0,   # 기본값
                "complexity_score": analysis.get("complexity_score", 0.5),
                "notes": analysis.get("recommendations", [])
            }
            
        except Exception as e:
            self.logger.warning(f"성능 메트릭 계산 실패: {str(e)}")
            return {
                "estimated_rows": 1000,
                "estimated_time": 1.0,
                "complexity_score": 0.5,
                "notes": []
            }
    
    def _generate_fallback_sql(self, query: str, query_intent: QueryIntent) -> GeneratedSQL:
        """폴백 SQL 생성"""
        # 기본 SELECT 쿼리 생성
        fallback_sql = "SELECT * FROM t_member LIMIT 100"
        
        return GeneratedSQL(
            sql=fallback_sql,
            template_used="fallback",
            confidence=0.3,
            estimated_rows=100,
            estimated_time=0.5,
            complexity_score=0.2,
            optimization_notes=["기본 폴백 쿼리 사용"]
        )
    
    def _generate_error_sql(self, error_message: str) -> GeneratedSQL:
        """오류 SQL 생성"""
        return GeneratedSQL(
            sql="SELECT 'ERROR' as message",
            template_used="error",
            confidence=0.0,
            estimated_rows=1,
            estimated_time=0.1,
            complexity_score=1.0,
            optimization_notes=[f"오류: {error_message}"]
        )
    
    def get_available_templates(self) -> List[Dict[str, Any]]:
        """사용 가능한 템플릿 목록 반환"""
        templates = []
        
        for key, template in self.sql_templates.items():
            templates.append({
                "key": key,
                "name": template.name,
                "description": template.description,
                "type": template.template_type.value,
                "required_entities": template.required_entities,
                "required_relationships": template.required_relationships,
                "complexity_score": template.complexity_score,
                "confidence": template.confidence
            })
        
        return templates
    
    def add_custom_template(self, template: SQLTemplate) -> bool:
        """사용자 정의 템플릿 추가"""
        try:
            template_key = f"custom_{template.name.lower().replace(' ', '_')}"
            self.sql_templates[template_key] = template
            self.logger.info(f"사용자 정의 템플릿 추가: {template.name}")
            return True
        except Exception as e:
            self.logger.error(f"사용자 정의 템플릿 추가 실패: {str(e)}")
            return False
    
    def validate_sql(self, sql: str) -> Dict[str, Any]:
        """SQL 유효성 검증"""
        try:
            # 기본 SQL 문법 검사
            if not sql.strip().upper().startswith('SELECT'):
                return {
                    "is_valid": False,
                    "error": "SELECT 문이 아닙니다.",
                    "suggestions": ["SELECT 문으로 시작하세요."]
                }
            
            # 위험한 키워드 검사
            dangerous_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE']
            for keyword in dangerous_keywords:
                if keyword in sql.upper():
                    return {
                        "is_valid": False,
                        "error": f"위험한 키워드 '{keyword}'가 포함되어 있습니다.",
                        "suggestions": ["읽기 전용 쿼리만 허용됩니다."]
                    }
            
            return {
                "is_valid": True,
                "error": None,
                "suggestions": []
            }
            
        except Exception as e:
            return {
                "is_valid": False,
                "error": f"SQL 검증 실패: {str(e)}",
                "suggestions": ["SQL 문법을 확인하세요."]
            }
    
    def suggest_improvements(self, sql: str) -> List[str]:
        """SQL 개선 제안"""
        suggestions = []
        
        # LIMIT 절 확인
        if 'LIMIT' not in sql.upper():
            suggestions.append("성능을 위해 LIMIT 절을 추가하세요.")
        
        # ORDER BY 절 확인
        if 'ORDER BY' not in sql.upper():
            suggestions.append("결과의 일관성을 위해 ORDER BY 절을 추가하세요.")
        
        # 인덱스 힌트 확인
        if 'USE INDEX' not in sql.upper():
            suggestions.append("성능 최적화를 위해 적절한 인덱스를 사용하세요.")
        
        # 조인 최적화 확인
        join_count = sql.upper().count('JOIN')
        if join_count > 3:
            suggestions.append("복잡한 조인이 있습니다. 서브쿼리로 분리하는 것을 고려하세요.")
        
        return suggestions


