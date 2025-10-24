#!/usr/bin/env python3
"""
Schema RAG Mapper
RAG 기반 스키마 매핑 시스템 (하이브리드 접근법)
"""

import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import re
from .sql_schema_analyzer import SQLSchemaAnalyzer
from .sql_schema_mapper import SQLSchemaMapper


class MappingSource(Enum):
    """매핑 소스"""
    RAG_MAPPING = "rag_mapping"
    LLM_GENERATION = "llm_generation"
    HYBRID_APPROACH = "hybrid_approach"


@dataclass
class SchemaMappingDoc:
    """스키마 매핑 문서"""
    query_pattern: str
    table: str
    columns: List[str]
    conditions: str
    aggregation: str
    confidence: float
    examples: List[str]
    metadata: Dict[str, Any]


@dataclass
class RAGMappingResult:
    """RAG 매핑 결과"""
    matched_doc: Optional[SchemaMappingDoc]
    confidence: float
    source: MappingSource
    sql_template: str
    reasoning: str


class SchemaRAGMapper:
    """RAG 기반 스키마 매핑 시스템"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.mapping_docs = self._load_mapping_documents()
        self.fallback_mapper = SQLSchemaMapper(config)
        self.schema_analyzer = SQLSchemaAnalyzer(config)
    
    def _load_mapping_documents(self) -> List[SchemaMappingDoc]:
        """매핑 문서 로드"""
        return [
            # 멤버십 성과 분석
            SchemaMappingDoc(
                query_pattern=".*(9월|8월|7월|6월|5월|4월|3월|2월|1월|10월|11월|12월).*(멤버십|맴버쉽).*(성과|실적|분석).*",
                table="t_member",
                columns=["status", "no"],
                conditions="status = 'A'",
                aggregation="COUNT(*)",
                confidence=0.95,
                examples=["9월 멤버십 성과 분석", "8월 맴버쉽 실적", "7월 멤버십 성과"],
                metadata={"category": "membership_performance", "frequency": "high"}
            ),
            
            # 회원 수 조회
            SchemaMappingDoc(
                query_pattern=".*(전체|총|모든).*(회원|맴버).*(수|명|명수).*",
                table="t_member",
                columns=["no"],
                conditions="1=1",
                aggregation="COUNT(*)",
                confidence=0.98,
                examples=["전체 회원 수", "총 맴버 수", "모든 회원 명수"],
                metadata={"category": "member_count", "frequency": "very_high"}
            ),
            
            # 활성 회원 수
            SchemaMappingDoc(
                query_pattern=".*(활성|active).*(회원|맴버).*(수|명).*",
                table="t_member",
                columns=["status"],
                conditions="status = 'A'",
                aggregation="COUNT(*)",
                confidence=0.97,
                examples=["활성 회원 수", "active 맴버 수"],
                metadata={"category": "active_members", "frequency": "high"}
            ),
            
            # 비활성 회원 수
            SchemaMappingDoc(
                query_pattern=".*(비활성|inactive|비활성화).*(회원|맴버).*(수|명).*",
                table="t_member",
                columns=["status"],
                conditions="status = 'I'",
                aggregation="COUNT(*)",
                confidence=0.95,
                examples=["비활성 회원 수", "inactive 맴버 수"],
                metadata={"category": "inactive_members", "frequency": "medium"}
            ),
            
            # 삭제된 회원 수
            SchemaMappingDoc(
                query_pattern=".*(삭제|deleted|제거).*(회원|맴버).*(수|명).*",
                table="t_member",
                columns=["status"],
                conditions="status = 'D'",
                aggregation="COUNT(*)",
                confidence=0.95,
                examples=["삭제된 회원 수", "deleted 맴버 수"],
                metadata={"category": "deleted_members", "frequency": "low"}
            ),
            
            # 월별 회원 추이
            SchemaMappingDoc(
                query_pattern=".*(월별|월간).*(회원|맴버).*(추이|변화|증감).*",
                table="t_member",
                columns=["status"],
                conditions="status = 'A'",
                aggregation="COUNT(*)",
                confidence=0.90,
                examples=["월별 회원 추이", "월간 맴버 변화"],
                metadata={"category": "monthly_trend", "frequency": "medium"}
            )
        ]
    
    def map_query_to_schema(self, user_query: str, generated_sql: str = "") -> RAGMappingResult:
        """
        사용자 쿼리를 스키마에 매핑
        
        Args:
            user_query: 사용자 쿼리
            generated_sql: 생성된 SQL (선택사항)
            
        Returns:
            RAGMappingResult: 매핑 결과
        """
        try:
            self.logger.info(f"RAG mapping query: {user_query}")
            
            # 1. RAG 매핑 시도
            rag_result = self._try_rag_mapping(user_query)
            if rag_result and rag_result.confidence > 0.8:
                return rag_result
            
            # 2. 하이브리드 접근법 (RAG + LLM)
            hybrid_result = self._try_hybrid_mapping(user_query, generated_sql)
            if hybrid_result and hybrid_result.confidence > 0.6:
                return hybrid_result
            
            # 3. LLM 폴백
            llm_result = self._try_llm_mapping(user_query, generated_sql)
            return llm_result
            
        except Exception as e:
            self.logger.error(f"Error in RAG mapping: {str(e)}")
            return RAGMappingResult(
                matched_doc=None,
                confidence=0.0,
                source=MappingSource.LLM_GENERATION,
                sql_template="",
                reasoning=f"RAG mapping failed: {str(e)}"
            )
    
    def _try_rag_mapping(self, user_query: str) -> Optional[RAGMappingResult]:
        """RAG 매핑 시도"""
        user_lower = user_query.lower()
        
        for doc in self.mapping_docs:
            # 정규식 매칭
            if re.search(doc.query_pattern, user_lower):
                # SQL 템플릿 생성
                sql_template = self._generate_sql_from_doc(doc, user_query)
                
                return RAGMappingResult(
                    matched_doc=doc,
                    confidence=doc.confidence,
                    source=MappingSource.RAG_MAPPING,
                    sql_template=sql_template,
                    reasoning=f"RAG 매핑 성공: {doc.metadata.get('category', 'unknown')}"
                )
        
        return None
    
    def _try_hybrid_mapping(self, user_query: str, generated_sql: str) -> Optional[RAGMappingResult]:
        """하이브리드 매핑 시도 (RAG + LLM)"""
        # 1. RAG로 기본 구조 파악
        rag_result = self._try_rag_mapping(user_query)
        if not rag_result:
            return None
        
        # 2. LLM으로 세부 조정
        if generated_sql:
            # 생성된 SQL과 RAG 결과 비교
            similarity = self._calculate_sql_similarity(rag_result.sql_template, generated_sql)
            if similarity > 0.7:
                # RAG 결과를 LLM 결과로 보완
                enhanced_sql = self._enhance_sql_with_llm(rag_result.sql_template, generated_sql)
                
                return RAGMappingResult(
                    matched_doc=rag_result.matched_doc,
                    confidence=rag_result.confidence * 0.9,  # 하이브리드 보정
                    source=MappingSource.HYBRID_APPROACH,
                    sql_template=enhanced_sql,
                    reasoning=f"하이브리드 매핑: RAG + LLM 보완"
                )
        
        return rag_result
    
    def _try_llm_mapping(self, user_query: str, generated_sql: str) -> RAGMappingResult:
        """LLM 폴백 매핑"""
        # 기존 SQLSchemaMapper 사용
        db_schema = self._get_basic_schema()
        mapping_result = self.fallback_mapper.map_sql_to_schema(generated_sql, db_schema, user_query)
        
        return RAGMappingResult(
            matched_doc=None,
            confidence=mapping_result.confidence,
            source=MappingSource.LLM_GENERATION,
            sql_template=mapping_result.mapped_sql,
            reasoning="LLM 폴백 매핑"
        )
    
    def _generate_sql_from_doc(self, doc: SchemaMappingDoc, user_query: str) -> str:
        """문서에서 SQL 생성"""
        # 월 추출 (9월, 8월 등)
        month_match = re.search(r'(\d+)월', user_query)
        month = month_match.group(1) if month_match else "9"
        
        # 기본 SQL 템플릿
        if doc.metadata.get("category") == "membership_performance":
            return f"""
            SELECT 
                '2025-{month.zfill(2)}' as analysis_month,
                COUNT(*) as total_members,
                COUNT(CASE WHEN status = 'A' THEN 1 END) as active_members,
                COUNT(CASE WHEN status = 'I' THEN 1 END) as inactive_members,
                COUNT(CASE WHEN status = 'D' THEN 1 END) as deleted_members,
                ROUND(COUNT(CASE WHEN status = 'A' THEN 1 END) * 100.0 / COUNT(*), 2) as active_rate_percent,
                ROUND(COUNT(CASE WHEN status = 'I' THEN 1 END) * 100.0 / COUNT(*), 2) as inactive_rate_percent,
                ROUND(COUNT(CASE WHEN status = 'D' THEN 1 END) * 100.0 / COUNT(*), 2) as deletion_rate_percent
            FROM {doc.table}
            """
        elif doc.metadata.get("category") == "member_count":
            return f"SELECT COUNT(*) as total_members FROM {doc.table}"
        elif doc.metadata.get("category") == "active_members":
            return f"SELECT COUNT(*) as active_members FROM {doc.table} WHERE {doc.conditions}"
        else:
            return f"SELECT {doc.aggregation} FROM {doc.table} WHERE {doc.conditions}"
    
    def _calculate_sql_similarity(self, sql1: str, sql2: str) -> float:
        """SQL 유사도 계산"""
        # 간단한 키워드 기반 유사도
        keywords1 = set(re.findall(r'\b\w+\b', sql1.upper()))
        keywords2 = set(re.findall(r'\b\w+\b', sql2.upper()))
        
        intersection = len(keywords1.intersection(keywords2))
        union = len(keywords1.union(keywords2))
        
        return intersection / union if union > 0 else 0.0
    
    def _enhance_sql_with_llm(self, rag_sql: str, llm_sql: str) -> str:
        """RAG SQL을 LLM 결과로 보완"""
        # RAG SQL을 기본으로 하고, LLM의 좋은 부분만 가져오기
        # 여기서는 간단히 RAG SQL 반환 (실제로는 더 정교한 로직 필요)
        return rag_sql
    
    def _get_basic_schema(self) -> Dict[str, Any]:
        """기본 스키마 정보"""
        return {
            "t_member": {
                "comment": "회원 테이블",
                "columns": [
                    {"name": "no", "type": "int", "nullable": False, "default": None, "comment": "회원번호"},
                    {"name": "name", "type": "varchar", "nullable": True, "default": None, "comment": "이름"},
                    {"name": "status", "type": "char", "nullable": False, "default": "A", "comment": "상태"},
                    {"name": "email", "type": "varchar", "nullable": True, "default": None, "comment": "이메일"}
                ]
            }
        }
    
    def add_mapping_document(self, doc: SchemaMappingDoc):
        """새로운 매핑 문서 추가"""
        self.mapping_docs.append(doc)
        self.logger.info(f"Added mapping document: {doc.metadata.get('category', 'unknown')}")
    
    def get_mapping_stats(self) -> Dict[str, Any]:
        """매핑 통계"""
        categories = {}
        for doc in self.mapping_docs:
            category = doc.metadata.get("category", "unknown")
            if category not in categories:
                categories[category] = 0
            categories[category] += 1
        
        return {
            "total_documents": len(self.mapping_docs),
            "categories": categories,
            "avg_confidence": sum(doc.confidence for doc in self.mapping_docs) / len(self.mapping_docs)
        }


