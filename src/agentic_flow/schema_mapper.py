#!/usr/bin/env python3
"""
스키마 매핑 시스템
자연어 쿼리를 데이터베이스 스키마에 매핑하는 핵심 모듈
"""

import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import re

from core.db import get_db_connection
from .schema_cache import SchemaCache, SchemaCacheManager

logger = logging.getLogger(__name__)


@dataclass
class TableInfo:
    """테이블 정보를 담는 데이터 클래스"""
    name: str
    columns: List[Dict[str, Any]]
    primary_keys: List[str]
    foreign_keys: List[Dict[str, Any]]
    indexes: List[Dict[str, Any]]
    row_count: Optional[int] = None


@dataclass
class ColumnInfo:
    """컬럼 정보를 담는 데이터 클래스"""
    name: str
    table: str
    data_type: str
    is_nullable: bool
    is_primary_key: bool
    is_foreign_key: bool
    referenced_table: Optional[str] = None
    referenced_column: Optional[str] = None


@dataclass
class MappingResult:
    """매핑 결과를 담는 데이터 클래스"""
    entity: str
    type: str  # 'table' or 'column'
    name: str
    table: Optional[str] = None
    score: float = 0.0
    confidence: str = "low"  # 'high', 'medium', 'low'
    reason: Optional[str] = None


class SchemaMapper:
    """
    데이터베이스 스키마 매핑 시스템
    자연어 엔티티를 데이터베이스 스키마 요소에 매핑
    """
    
    def __init__(self):
        """SchemaMapper 초기화"""
        self.db_connection = None
        self.schema_cache = {}
        self.last_schema_update = None
        
        # 고급 캐싱 시스템 초기화
        self.cache_manager = SchemaCacheManager()
        self.schema_cache_instance = self.cache_manager.create_cache(
            "schema_metadata",
            max_size=500,
            ttl_seconds=3600,
            max_memory_mb=50
        )
        self.mapping_cache_instance = self.cache_manager.create_cache(
            "entity_mappings",
            max_size=1000,
            ttl_seconds=1800,
            max_memory_mb=30
        )
        
        # 백그라운드 정리 시작
        self.cache_manager.start_all_background_cleanup()
        
        logger.info("SchemaMapper initialized with advanced caching system")
    
    def _get_db_connection(self):
        """데이터베이스 연결 획득"""
        if not self.db_connection:
            self.db_connection = get_db_connection()
        return self.db_connection
    
    def refresh_schema_metadata(self) -> bool:
        """
        데이터베이스로부터 최신 스키마 정보를 추출하여 캐싱
        
        Returns:
            bool: 성공 여부
        """
        try:
            logger.info("Refreshing schema metadata...")
            start_time = time.time()
            
            db_conn = self._get_db_connection()
            
            # 1. 테이블 정보 추출
            tables = self._extract_tables(db_conn)
            
            # 2. 컬럼 정보 추출
            columns = self._extract_columns(db_conn)
            
            # 3. 관계 정보 추출
            relationships = self._extract_relationships(db_conn)
            
            # 4. 캐시 업데이트
            self.schema_cache = {
                "tables": tables,
                "columns": columns,
                "relationships": relationships,
                "last_updated": datetime.now(),
                "extraction_time": time.time() - start_time
            }
            
            self.last_schema_update = datetime.now()
            
            logger.info(f"Schema metadata refreshed successfully in {time.time() - start_time:.2f}s")
            logger.info(f"Found {len(tables)} tables, {len(columns)} columns, {len(relationships)} relationships")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to refresh schema metadata: {str(e)}", exc_info=True)
            return False
    
    def _extract_tables(self, db_conn) -> List[Dict[str, Any]]:
        """테이블 정보 추출"""
        try:
            # INFORMATION_SCHEMA.TABLES에서 테이블 정보 조회
            query = """
            SELECT 
                TABLE_NAME,
                TABLE_TYPE,
                ENGINE,
                TABLE_ROWS,
                DATA_LENGTH,
                INDEX_LENGTH
            FROM INFORMATION_SCHEMA.TABLES 
            WHERE TABLE_SCHEMA = DATABASE()
            AND TABLE_TYPE = 'BASE TABLE'
            ORDER BY TABLE_NAME
            """
            
            result = db_conn.execute_query(query)
            tables = []
            
            for row in result:
                tables.append({
                    "name": row["TABLE_NAME"],
                    "type": row["TABLE_TYPE"],
                    "engine": row["ENGINE"],
                    "row_count": row["TABLE_ROWS"],
                    "data_size": row["DATA_LENGTH"],
                    "index_size": row["INDEX_LENGTH"]
                })
            
            logger.info(f"Extracted {len(tables)} tables")
            return tables
            
        except Exception as e:
            logger.error(f"Failed to extract tables: {str(e)}")
            return []
    
    def _extract_columns(self, db_conn) -> List[Dict[str, Any]]:
        """컬럼 정보 추출"""
        try:
            # INFORMATION_SCHEMA.COLUMNS에서 컬럼 정보 조회
            query = """
            SELECT 
                TABLE_NAME,
                COLUMN_NAME,
                DATA_TYPE,
                IS_NULLABLE,
                COLUMN_KEY,
                COLUMN_DEFAULT,
                EXTRA,
                COLUMN_COMMENT
            FROM INFORMATION_SCHEMA.COLUMNS 
            WHERE TABLE_SCHEMA = DATABASE()
            ORDER BY TABLE_NAME, ORDINAL_POSITION
            """
            
            result = db_conn.execute_query(query)
            columns = []
            
            for row in result:
                columns.append({
                    "table": row["TABLE_NAME"],
                    "name": row["COLUMN_NAME"],
                    "data_type": row["DATA_TYPE"],
                    "is_nullable": row["IS_NULLABLE"] == "YES",
                    "is_primary_key": row["COLUMN_KEY"] == "PRI",
                    "is_unique": row["COLUMN_KEY"] == "UNI",
                    "is_indexed": row["COLUMN_KEY"] in ["PRI", "UNI", "MUL"],
                    "default_value": row["COLUMN_DEFAULT"],
                    "extra": row["EXTRA"],
                    "comment": row["COLUMN_COMMENT"]
                })
            
            logger.info(f"Extracted {len(columns)} columns")
            return columns
            
        except Exception as e:
            logger.error(f"Failed to extract columns: {str(e)}")
            return []
    
    def _extract_relationships(self, db_conn) -> List[Dict[str, Any]]:
        """테이블 간 관계 정보 추출"""
        try:
            # INFORMATION_SCHEMA.KEY_COLUMN_USAGE에서 외래키 관계 조회
            query = """
            SELECT 
                TABLE_NAME,
                COLUMN_NAME,
                REFERENCED_TABLE_NAME,
                REFERENCED_COLUMN_NAME,
                CONSTRAINT_NAME
            FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE 
            WHERE TABLE_SCHEMA = DATABASE()
            AND REFERENCED_TABLE_NAME IS NOT NULL
            ORDER BY TABLE_NAME, COLUMN_NAME
            """
            
            result = db_conn.execute_query(query)
            relationships = []
            
            for row in result:
                relationships.append({
                    "source_table": row["TABLE_NAME"],
                    "source_column": row["COLUMN_NAME"],
                    "target_table": row["REFERENCED_TABLE_NAME"],
                    "target_column": row["REFERENCED_COLUMN_NAME"],
                    "constraint_name": row["CONSTRAINT_NAME"]
                })
            
            logger.info(f"Extracted {len(relationships)} relationships")
            return relationships
            
        except Exception as e:
            logger.error(f"Failed to extract relationships: {str(e)}")
            return []
    
    def get_schema_info(self) -> Dict[str, Any]:
        """현재 스키마 정보 반환"""
        # 고급 캐시에서 조회 시도
        cached_schema = self.schema_cache_instance.get("schema_metadata")
        if cached_schema:
            logger.debug("Schema metadata retrieved from cache")
            return cached_schema
        
        # 캐시에 없으면 새로고침
        if self.refresh_schema_metadata():
            # 새로고침된 데이터를 캐시에 저장
            self.schema_cache_instance.set("schema_metadata", self.schema_cache)
            return self.schema_cache
        
        return {}
    
    def _is_cache_expired(self) -> bool:
        """캐시 만료 여부 확인"""
        if not self.last_schema_update:
            return True
        
        elapsed = (datetime.now() - self.last_schema_update).total_seconds()
        return elapsed > self.cache_ttl
    
    def map_entity_to_schema(self, entity: str, threshold: float = 0.7) -> Optional[MappingResult]:
        """
        자연어 엔티티를 스키마 요소에 매핑
        
        Args:
            entity: 매핑할 엔티티명
            threshold: 최소 신뢰도 임계값
            
        Returns:
            MappingResult: 매핑 결과
        """
        try:
            # 고급 캐시에서 조회 시도
            cache_key = f"mapping_{entity}_{threshold}"
            cached_result = self.mapping_cache_instance.get(cache_key)
            if cached_result:
                logger.debug(f"Entity mapping retrieved from cache: {entity}")
                return cached_result
            
            # 스키마 정보 가져오기
            schema_info = self.get_schema_info()
            if not schema_info:
                return None
            
            best_match = None
            best_score = 0.0
            
            # 1. 테이블 매핑 시도
            for table in schema_info.get("tables", []):
                score = self._calculate_similarity(entity, table["name"])
                if score > threshold and score > best_score:
                    best_score = score
                    best_match = MappingResult(
                        entity=entity,
                        type="table",
                        name=table["name"],
                        score=score,
                        confidence=self._get_confidence_level(score)
                    )
            
            # 2. 컬럼 매핑 시도
            for column in schema_info.get("columns", []):
                score = self._calculate_similarity(entity, column["name"])
                if score > threshold and score > best_score:
                    best_score = score
                    best_match = MappingResult(
                        entity=entity,
                        type="column",
                        name=column["name"],
                        table=column["table"],
                        score=score,
                        confidence=self._get_confidence_level(score)
                    )
            
            # 결과를 고급 캐시에 저장
            if best_match:
                self.mapping_cache_instance.set(cache_key, best_match)
                logger.debug(f"Entity mapping cached: {entity} -> {best_match.name}")
            
            return best_match
            
        except Exception as e:
            logger.error(f"Failed to map entity '{entity}': {str(e)}")
            return None
    
    def _calculate_similarity(self, entity: str, schema_name: str) -> float:
        """
        엔티티와 스키마 이름 간의 유사도 계산
        
        Args:
            entity: 자연어 엔티티
            schema_name: 스키마 이름
            
        Returns:
            float: 유사도 점수 (0.0 ~ 1.0)
        """
        try:
            # 정규화
            entity_norm = self._normalize_name(entity)
            schema_norm = self._normalize_name(schema_name)
            
            # 1. 정확한 매칭
            if entity_norm == schema_norm:
                return 1.0
            
            # 2. 포함 관계 매칭
            if entity_norm in schema_norm or schema_norm in entity_norm:
                return 0.8
            
            # 3. 부분 매칭 (단어 단위)
            entity_words = set(entity_norm.split('_'))
            schema_words = set(schema_norm.split('_'))
            
            if entity_words & schema_words:  # 교집합이 있는 경우
                intersection = len(entity_words & schema_words)
                union = len(entity_words | schema_words)
                return intersection / union
            
            # 4. 편집 거리 기반 유사도
            return self._calculate_edit_distance_similarity(entity_norm, schema_norm)
            
        except Exception as e:
            logger.error(f"Failed to calculate similarity: {str(e)}")
            return 0.0
    
    def _normalize_name(self, name: str) -> str:
        """이름 정규화"""
        if not name:
            return ""
        
        # 소문자 변환
        normalized = name.lower()
        
        # 특수 문자를 언더스코어로 변환
        normalized = re.sub(r'[^a-z0-9]', '_', normalized)
        
        # 연속된 언더스코어 제거
        normalized = re.sub(r'_+', '_', normalized)
        
        # 앞뒤 언더스코어 제거
        normalized = normalized.strip('_')
        
        return normalized
    
    def _calculate_edit_distance_similarity(self, str1: str, str2: str) -> float:
        """편집 거리 기반 유사도 계산"""
        if not str1 or not str2:
            return 0.0
        
        # 간단한 편집 거리 계산 (Levenshtein distance)
        m, n = len(str1), len(str2)
        if m == 0:
            return 0.0 if n > 0 else 1.0
        if n == 0:
            return 0.0
        
        # DP 테이블 초기화
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        # 편집 거리 계산
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if str1[i-1] == str2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        
        # 유사도 점수 계산 (0.0 ~ 1.0)
        max_len = max(m, n)
        if max_len == 0:
            return 1.0
        
        distance = dp[m][n]
        similarity = 1.0 - (distance / max_len)
        
        return max(0.0, similarity)
    
    def _get_confidence_level(self, score: float) -> str:
        """신뢰도 점수에 따른 레벨 반환"""
        if score >= 0.9:
            return "high"
        elif score >= 0.7:
            return "medium"
        else:
            return "low"
    
    def _cleanup_cache(self):
        """캐시 정리 (LRU 정책)"""
        if len(self.mapping_cache) <= self.max_cache_size:
            return
        
        # 타임스탬프 기준으로 정렬하여 오래된 항목 제거
        sorted_items = sorted(
            self.mapping_cache.items(),
            key=lambda x: x[1]["timestamp"]
        )
        
        # 오래된 항목들을 제거
        items_to_remove = len(self.mapping_cache) - self.max_cache_size + 10
        for key, _ in sorted_items[:items_to_remove]:
            del self.mapping_cache[key]
        
        logger.info(f"Cache cleaned up, removed {items_to_remove} items")
    
    def get_table_info(self, table_name: str) -> Optional[TableInfo]:
        """특정 테이블의 상세 정보 반환"""
        try:
            schema_info = self.get_schema_info()
            if not schema_info:
                return None
            
            # 테이블 찾기
            table_data = None
            for table in schema_info.get("tables", []):
                if table["name"] == table_name:
                    table_data = table
                    break
            
            if not table_data:
                return None
            
            # 컬럼 정보 수집
            columns = []
            for column in schema_info.get("columns", []):
                if column["table"] == table_name:
                    columns.append(column)
            
            # 기본키 찾기
            primary_keys = [col["name"] for col in columns if col["is_primary_key"]]
            
            # 외래키 찾기
            foreign_keys = []
            for rel in schema_info.get("relationships", []):
                if rel["source_table"] == table_name:
                    foreign_keys.append(rel)
            
            return TableInfo(
                name=table_name,
                columns=columns,
                primary_keys=primary_keys,
                foreign_keys=foreign_keys,
                indexes=[],  # TODO: 인덱스 정보 추출
                row_count=table_data.get("row_count")
            )
            
        except Exception as e:
            logger.error(f"Failed to get table info for '{table_name}': {str(e)}")
            return None
    
    def get_related_tables(self, table_name: str) -> List[str]:
        """특정 테이블과 관련된 테이블 목록 반환"""
        try:
            schema_info = self.get_schema_info()
            if not schema_info:
                return []
            
            related_tables = set()
            
            # 외래키 관계 확인
            for rel in schema_info.get("relationships", []):
                if rel["source_table"] == table_name:
                    related_tables.add(rel["target_table"])
                elif rel["target_table"] == table_name:
                    related_tables.add(rel["source_table"])
            
            return list(related_tables)
            
        except Exception as e:
            logger.error(f"Failed to get related tables for '{table_name}': {str(e)}")
            return []
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """캐시 통계 정보 반환"""
        try:
            return self.cache_manager.get_all_stats()
        except Exception as e:
            logger.error(f"Failed to get cache stats: {str(e)}")
            return {}
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """성능 메트릭 반환"""
        try:
            stats = self.get_cache_stats()
            
            # 성능 메트릭 계산
            total_hits = sum(cache_stats.get("hits", 0) for cache_stats in stats.values())
            total_misses = sum(cache_stats.get("misses", 0) for cache_stats in stats.values())
            total_requests = total_hits + total_misses
            
            overall_hit_rate = total_hits / total_requests if total_requests > 0 else 0.0
            
            return {
                "overall_hit_rate": overall_hit_rate,
                "total_requests": total_requests,
                "total_hits": total_hits,
                "total_misses": total_misses,
                "cache_details": stats,
                "memory_usage": {
                    "total_size_bytes": sum(cache_stats.get("total_size_bytes", 0) for cache_stats in stats.values()),
                    "max_memory_bytes": sum(cache_stats.get("max_memory_bytes", 0) for cache_stats in stats.values())
                }
            }
        except Exception as e:
            logger.error(f"Failed to get performance metrics: {str(e)}")
            return {}
    
    def cleanup_caches(self):
        """모든 캐시 정리"""
        try:
            self.cache_manager.cleanup_all()
            logger.info("All caches cleaned up")
        except Exception as e:
            logger.error(f"Failed to cleanup caches: {str(e)}")
    
    def context_aware_mapping(self, query_entities: List[str], query_context: Optional[Dict[str, Any]] = None, history: Optional[List[Dict[str, Any]]] = None) -> Dict[str, MappingResult]:
        """
        컨텍스트를 고려한 향상된 스키마 매핑
        
        Args:
            query_entities: 매핑할 엔티티 목록
            query_context: 쿼리 컨텍스트 정보
            history: 이전 쿼리 히스토리
            
        Returns:
            Dict[str, MappingResult]: 엔티티별 매핑 결과
        """
        try:
            logger.info(f"Context-aware mapping for {len(query_entities)} entities")
            
            mappings = {}
            
            # 1. 쿼리 컨텍스트 분석
            context_tables = self._extract_context_tables(query_context) if query_context else []
            
            # 2. 히스토리 기반 가중치 계산
            history_weights = self._calculate_history_weights(history) if history else {}
            
            # 3. 각 엔티티에 대해 매핑 수행
            for entity in query_entities:
                # 기본 매핑 수행
                base_mapping = self.map_entity_to_schema(entity)
                
                if base_mapping:
                    # 컨텍스트 및 히스토리 기반 가중치 적용
                    enhanced_mapping = self._apply_context_weights(
                        base_mapping, context_tables, history_weights
                    )
                    mappings[entity] = enhanced_mapping
                else:
                    # 매핑 실패 시 대안 제안
                    alternative = self._suggest_alternatives(entity, context_tables)
                    if alternative:
                        mappings[entity] = alternative
            
            logger.info(f"Context-aware mapping completed: {len(mappings)} successful mappings")
            return mappings
            
        except Exception as e:
            logger.error(f"Failed to perform context-aware mapping: {str(e)}")
            return {}
    
    def _extract_context_tables(self, query_context: Dict[str, Any]) -> List[str]:
        """쿼리 컨텍스트에서 관련 테이블 추출"""
        try:
            context_tables = []
            
            # 이미 언급된 테이블들
            if "mentioned_tables" in query_context:
                context_tables.extend(query_context["mentioned_tables"])
            
            # 쿼리 유형에 따른 기본 테이블 추론
            if "query_type" in query_context:
                query_type = query_context["query_type"]
                if query_type == "user_analysis":
                    context_tables.extend(["t_member", "t_member_login_log"])
                elif query_type == "product_analysis":
                    context_tables.extend(["t_product", "t_order", "t_creator"])
                elif query_type == "content_analysis":
                    context_tables.extend(["t_post", "t_community"])
            
            return list(set(context_tables))  # 중복 제거
            
        except Exception as e:
            logger.error(f"Failed to extract context tables: {str(e)}")
            return []
    
    def _calculate_history_weights(self, history: List[Dict[str, Any]]) -> Dict[str, float]:
        """히스토리 기반 가중치 계산"""
        try:
            weights = {}
            
            # 최근 쿼리에서 자주 사용된 매핑에 가중치 부여
            for query in history[-10:]:  # 최근 10개 쿼리만 고려
                if "mappings" in query:
                    for entity, mapping in query["mappings"].items():
                        if isinstance(mapping, dict) and "name" in mapping:
                            key = f"{entity}_{mapping['name']}"
                            weights[key] = weights.get(key, 0) + 0.1
            
            return weights
            
        except Exception as e:
            logger.error(f"Failed to calculate history weights: {str(e)}")
            return {}
    
    def _apply_context_weights(self, base_mapping: MappingResult, context_tables: List[str], history_weights: Dict[str, float]) -> MappingResult:
        """컨텍스트 및 히스토리 가중치 적용"""
        try:
            # 기본 점수
            adjusted_score = base_mapping.score
            
            # 컨텍스트 테이블과의 연관성 확인
            if base_mapping.type == "table" and base_mapping.name in context_tables:
                adjusted_score = min(1.0, adjusted_score + 0.2)
            elif base_mapping.type == "column" and base_mapping.table in context_tables:
                adjusted_score = min(1.0, adjusted_score + 0.1)
            
            # 히스토리 가중치 적용
            history_key = f"{base_mapping.entity}_{base_mapping.name}"
            if history_key in history_weights:
                adjusted_score = min(1.0, adjusted_score + history_weights[history_key])
            
            # 조정된 매핑 결과 반환
            return MappingResult(
                entity=base_mapping.entity,
                type=base_mapping.type,
                name=base_mapping.name,
                table=base_mapping.table,
                score=adjusted_score,
                confidence=self._get_confidence_level(adjusted_score),
                reason=f"Context-enhanced: {base_mapping.reason or 'Original mapping'}"
            )
            
        except Exception as e:
            logger.error(f"Failed to apply context weights: {str(e)}")
            return base_mapping
    
    def _suggest_alternatives(self, entity: str, context_tables: List[str]) -> Optional[MappingResult]:
        """매핑 실패 시 대안 제안"""
        try:
            # 컨텍스트 테이블에서 유사한 컬럼 찾기
            schema_info = self.get_schema_info()
            if not schema_info:
                return None
            
            best_alternative = None
            best_score = 0.0
            
            for table in context_tables:
                for column in schema_info.get("columns", []):
                    if column["table"] == table:
                        score = self._calculate_similarity(entity, column["name"])
                        if score > 0.3 and score > best_score:  # 낮은 임계값으로 대안 제안
                            best_score = score
                            best_alternative = MappingResult(
                                entity=entity,
                                type="column",
                                name=column["name"],
                                table=column["table"],
                                score=score,
                                confidence="low",
                                reason=f"Alternative suggestion from context table {table}"
                            )
            
            return best_alternative
            
        except Exception as e:
            logger.error(f"Failed to suggest alternatives: {str(e)}")
            return None

    def monitor_schema_changes(self, interval: int = 3600):
        """
        주기적으로 스키마 변경 감지 및 갱신
        
        Args:
            interval: 감지 간격 (초)
        """
        try:
            logger.info(f"Starting schema change monitoring with {interval}s interval")
            
            while True:
                try:
                    current_schema_hash = self._calculate_schema_hash()
                    
                    if current_schema_hash != self._last_schema_hash:
                        logger.info("Schema change detected, refreshing metadata...")
                        if self.refresh_schema_metadata():
                            self._last_schema_hash = current_schema_hash
                            logger.info("Schema metadata updated successfully")
                        else:
                            logger.error("Failed to update schema metadata")
                    
                    time.sleep(interval)
                    
                except Exception as e:
                    logger.error(f"Error in schema monitoring: {str(e)}")
                    time.sleep(interval)
                    
        except KeyboardInterrupt:
            logger.info("Schema monitoring stopped by user")
        except Exception as e:
            logger.error(f"Schema monitoring failed: {str(e)}")
    
    def _calculate_schema_hash(self) -> str:
        """스키마 해시 계산"""
        try:
            import hashlib
            
            # 현재 스키마 정보 가져오기
            schema_info = self.get_schema_info()
            if not schema_info:
                return ""
            
            # 스키마 정보를 문자열로 변환하여 해시 계산
            schema_string = str(sorted(schema_info.items()))
            return hashlib.md5(schema_string.encode()).hexdigest()
            
        except Exception as e:
            logger.error(f"Failed to calculate schema hash: {str(e)}")
            return ""

    def __del__(self):
        """소멸자 - 백그라운드 스레드 정리"""
        try:
            self.cache_manager.stop_all_background_cleanup()
        except:
            pass
