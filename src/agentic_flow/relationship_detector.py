#!/usr/bin/env python3
"""
테이블 관계 감지 시스템
데이터베이스 테이블 간의 관계를 자동으로 감지하고 조인 경로를 생성하는 시스템
"""

import logging
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import networkx as nx
from collections import defaultdict

logger = logging.getLogger(__name__)


class RelationshipType(Enum):
    """관계 타입 열거형"""
    ONE_TO_ONE = "one_to_one"
    ONE_TO_MANY = "one_to_many"
    MANY_TO_MANY = "many_to_many"
    SELF_REFERENCE = "self_reference"


@dataclass
class TableRelationship:
    """테이블 관계 정보"""
    source_table: str
    target_table: str
    source_column: str
    target_column: str
    relationship_type: RelationshipType
    confidence: float
    constraint_name: Optional[str] = None
    join_condition: Optional[str] = None


@dataclass
class JoinPath:
    """조인 경로 정보"""
    source_table: str
    target_table: str
    path: List[TableRelationship]
    total_cost: float
    confidence: float


class RelationshipDetector:
    """
    테이블 간 관계를 자동으로 감지하고 조인 경로를 생성하는 시스템
    """
    
    def __init__(self, schema_mapper):
        """
        RelationshipDetector 초기화
        
        Args:
            schema_mapper: SchemaMapper 인스턴스
        """
        self.schema_mapper = schema_mapper
        self.relationships: List[TableRelationship] = []
        self.relationship_graph: nx.DiGraph = nx.DiGraph()
        self.name_patterns = self._build_name_patterns()
        
        logger.info("RelationshipDetector initialized")
    
    def _build_name_patterns(self) -> Dict[str, List[str]]:
        """이름 패턴 기반 관계 감지를 위한 패턴 구축"""
        return {
            "foreign_key_patterns": [
                r"(\w+)_id$",
                r"(\w+)_no$",
                r"(\w+)_num$",
                r"(\w+)_number$",
                r"(\w+)_key$",
                r"(\w+)_ref$",
                r"(\w+)_ref_id$"
            ],
            "junction_table_patterns": [
                r"(\w+)_(\w+)_mapping$",
                r"(\w+)_(\w+)_relation$",
                r"(\w+)_(\w+)_junction$",
                r"(\w+)_(\w+)_link$",
                r"(\w+)_(\w+)_bridge$"
            ],
            "self_reference_patterns": [
                r"parent_\w+$",
                r"child_\w+$",
                r"super_\w+$",
                r"sub_\w+$",
                r"upper_\w+$",
                r"lower_\w+$"
            ]
        }
    
    def detect_relationships(self) -> List[TableRelationship]:
        """
        데이터베이스에서 테이블 간 관계를 자동으로 감지
        
        Returns:
            List[TableRelationship]: 감지된 관계 목록
        """
        try:
            logger.info("Starting relationship detection...")
            self.relationships = []
            
            # 스키마 정보 가져오기
            schema_info = self.schema_mapper.get_schema_info()
            if not schema_info:
                logger.error("No schema information available")
                return []
            
            # 1. 외래키 기반 관계 감지
            fk_relationships = self._detect_foreign_key_relationships(schema_info)
            self.relationships.extend(fk_relationships)
            
            # 2. 이름 패턴 기반 관계 감지
            pattern_relationships = self._detect_name_pattern_relationships(schema_info)
            self.relationships.extend(pattern_relationships)
            
            # 3. 데이터 기반 관계 감지
            data_relationships = self._detect_data_based_relationships(schema_info)
            self.relationships.extend(data_relationships)
            
            # 4. 관계 그래프 구축
            self._build_relationship_graph()
            
            logger.info(f"Detected {len(self.relationships)} relationships")
            return self.relationships
            
        except Exception as e:
            logger.error(f"Failed to detect relationships: {str(e)}", exc_info=True)
            return []
    
    def _detect_foreign_key_relationships(self, schema_info: Dict[str, Any]) -> List[TableRelationship]:
        """외래키 기반 관계 감지"""
        relationships = []
        seen_relationships = set()  # 중복 방지를 위한 집합
        
        try:
            # 외래키 관계 정보 추출
            for rel_info in schema_info.get("relationships", []):
                source_table = rel_info["source_table"]
                target_table = rel_info["target_table"]
                source_column = rel_info["source_column"]
                target_column = rel_info["target_column"]
                constraint_name = rel_info["constraint_name"]
                
                # 중복 관계 확인
                relationship_key = (source_table, target_table, source_column, target_column)
                if relationship_key not in seen_relationships:
                    seen_relationships.add(relationship_key)
                    
                    # 관계 타입 결정
                    relationship_type = self._determine_relationship_type(
                        source_table, target_table, source_column, target_column
                    )
                    
                    # 조인 조건 생성
                    join_condition = f"{source_table}.{source_column} = {target_table}.{target_column}"
                    
                    relationship = TableRelationship(
                        source_table=source_table,
                        target_table=target_table,
                        source_column=source_column,
                        target_column=target_column,
                        relationship_type=relationship_type,
                        confidence=0.95,  # 외래키는 높은 신뢰도
                        constraint_name=constraint_name,
                        join_condition=join_condition
                    )
                    
                    relationships.append(relationship)
                
        except Exception as e:
            logger.error(f"Failed to detect foreign key relationships: {str(e)}")
        
        return relationships
    
    def _detect_name_pattern_relationships(self, schema_info: Dict[str, Any]) -> List[TableRelationship]:
        """이름 패턴 기반 관계 감지"""
        relationships = []
        seen_relationships = set()  # 중복 방지를 위한 집합
        
        try:
            tables = schema_info.get("tables", [])
            columns = schema_info.get("columns", [])
            
            # 테이블 이름으로 매핑
            table_names = {table["name"]: table for table in tables}
            
            # 컬럼별로 패턴 매칭 수행
            for column in columns:
                column_name = column["name"]
                table_name = column["table"]
                
                # 외래키 패턴 확인
                for pattern in self.name_patterns["foreign_key_patterns"]:
                    match = self._match_pattern(pattern, column_name)
                    if match:
                        referenced_table = match.group(1)
                        
                        # 참조 테이블이 존재하는지 확인
                        if referenced_table in table_names:
                            # 기본키 컬럼 찾기
                            pk_column = self._find_primary_key_column(referenced_table, columns)
                            if pk_column:
                                # 중복 관계 확인
                                relationship_key = (table_name, referenced_table, column_name, pk_column)
                                if relationship_key not in seen_relationships:
                                    seen_relationships.add(relationship_key)
                                    
                                    relationship = TableRelationship(
                                        source_table=table_name,
                                        target_table=referenced_table,
                                        source_column=column_name,
                                        target_column=pk_column,
                                        relationship_type=RelationshipType.ONE_TO_MANY,
                                        confidence=0.8,  # 패턴 기반은 중간 신뢰도
                                        join_condition=f"{table_name}.{column_name} = {referenced_table}.{pk_column}"
                                    )
                                    relationships.append(relationship)
                
                # 자체 참조 패턴 확인 (순환 참조 방지 로직 추가)
                for pattern in self.name_patterns["self_reference_patterns"]:
                    if self._match_pattern(pattern, column_name):
                        pk_column = self._find_primary_key_column(table_name, columns)
                        if pk_column and column_name != pk_column:  # 자기 자신을 참조하지 않는 경우만
                            # 중복 관계 확인
                            relationship_key = (table_name, table_name, column_name, pk_column)
                            if relationship_key not in seen_relationships:
                                seen_relationships.add(relationship_key)
                                
                                # 자체 참조인 경우 특별 처리
                                relationship = TableRelationship(
                                    source_table=table_name,
                                    target_table=table_name,
                                    source_column=column_name,
                                    target_column=pk_column,
                                    relationship_type=RelationshipType.SELF_REFERENCE,
                                    confidence=0.7,
                                    join_condition=f"{table_name}.{column_name} = {table_name}.{pk_column}"
                                )
                                relationships.append(relationship)
            
        except Exception as e:
            logger.error(f"Failed to detect name pattern relationships: {str(e)}")
        
        return relationships
    
    def _detect_data_based_relationships(self, schema_info: Dict[str, Any]) -> List[TableRelationship]:
        """데이터 기반 관계 감지 (데이터 분포 분석)"""
        relationships = []
        
        try:
            # 실제 데이터 분포를 분석하여 관계를 추론
            # 이는 복잡한 분석이므로 기본적인 구현만 제공
            
            # 예: 동일한 값 분포를 가진 컬럼들 간의 관계 추론
            # 실제 구현에서는 데이터 샘플링 및 통계 분석 필요
            
            logger.info("Data-based relationship detection not fully implemented")
            
        except Exception as e:
            logger.error(f"Failed to detect data-based relationships: {str(e)}")
        
        return relationships
    
    def _match_pattern(self, pattern: str, text: str) -> Optional[Any]:
        """정규표현식 패턴 매칭"""
        import re
        return re.search(pattern, text, re.IGNORECASE)
    
    def _determine_relationship_type(self, source_table: str, target_table: str, 
                                   source_column: str, target_column: str) -> RelationshipType:
        """관계 타입 결정"""
        if source_table == target_table:
            return RelationshipType.SELF_REFERENCE
        
        # 기본적으로 ONE_TO_MANY로 가정
        # 실제로는 카디널리티 분석이 필요
        return RelationshipType.ONE_TO_MANY
    
    def _find_primary_key_column(self, table_name: str, columns: List[Dict[str, Any]]) -> Optional[str]:
        """테이블의 기본키 컬럼 찾기"""
        # 정확한 매칭 먼저 시도
        for column in columns:
            if column["table"] == table_name and column.get("column_key") == "PRI":
                return column["name"]
        
        # 정확한 매칭이 실패하면 부분 매칭 시도
        table_variations = self._get_table_name_variations(table_name)
        for variation in table_variations:
            for column in columns:
                if column["table"] == variation and column.get("column_key") == "PRI":
                    logger.debug(f"Found primary key for table '{table_name}' using variation '{variation}': {column['name']}")
                    return column["name"]
        
        return None
    
    def _get_table_name_variations(self, table_name: str) -> List[str]:
        """테이블명의 다양한 변형 생성"""
        variations = [table_name]
        
        # 접두사/접미사 변형
        if table_name.startswith("t_"):
            variations.append(table_name[2:])  # t_ 제거
        else:
            variations.append(f"t_{table_name}")  # t_ 추가
        
        # 언더스코어 변형
        if "_" in table_name:
            variations.append(table_name.replace("_", ""))  # 언더스코어 제거
            variations.extend(table_name.split("_"))  # 언더스코어로 분리
        
        # 복수형 변형
        if table_name.endswith("s"):
            variations.append(table_name[:-1])  # 복수형 s 제거
        else:
            variations.append(f"{table_name}s")  # 복수형 s 추가
        
        return list(set(variations))  # 중복 제거
    
    def _build_relationship_graph(self):
        """관계 그래프 구축"""
        try:
            self.relationship_graph.clear()
            
            # 노드 추가 (테이블)
            for relationship in self.relationships:
                if not self.relationship_graph.has_node(relationship.source_table):
                    self.relationship_graph.add_node(relationship.source_table)
                if not self.relationship_graph.has_node(relationship.target_table):
                    self.relationship_graph.add_node(relationship.target_table)
                
                # 엣지 추가 (관계)
                self.relationship_graph.add_edge(
                    relationship.source_table,
                    relationship.target_table,
                    relationship=relationship,
                    weight=1.0 / relationship.confidence  # 신뢰도가 높을수록 가중치 낮음
                )
            
            logger.info(f"Built relationship graph with {self.relationship_graph.number_of_nodes()} nodes and {self.relationship_graph.number_of_edges()} edges")
            
        except Exception as e:
            logger.error(f"Failed to build relationship graph: {str(e)}")
    
    def find_join_path(self, source_table: str, target_table: str, 
                      max_depth: int = 3) -> Optional[JoinPath]:
        """
        두 테이블 간의 최적 조인 경로 찾기
        
        Args:
            source_table: 시작 테이블
            target_table: 목표 테이블
            max_depth: 최대 탐색 깊이
            
        Returns:
            JoinPath: 조인 경로 정보
        """
        try:
            # 그래프가 비어있으면 관계 감지 수행
            if self.relationship_graph.number_of_nodes() == 0:
                self.detect_relationships()
            
            # 테이블 존재 확인
            if source_table not in self.relationship_graph.nodes():
                logger.warning(f"Source table '{source_table}' not found in relationship graph")
                return None
            
            if target_table not in self.relationship_graph.nodes():
                logger.warning(f"Target table '{target_table}' not found in relationship graph")
                return None
            
            # 최단 경로 찾기
            try:
                path_nodes = nx.shortest_path(
                    self.relationship_graph, 
                    source_table, 
                    target_table
                )
            except nx.NetworkXNoPath:
                logger.warning(f"No path found between '{source_table}' and '{target_table}'")
                return None
            
            # 경로가 너무 깊으면 제한
            if len(path_nodes) - 1 > max_depth:
                logger.warning(f"Path too deep: {len(path_nodes) - 1} > {max_depth}")
                return None
            
            # 관계 정보 추출
            relationships = []
            total_cost = 0.0
            
            for i in range(len(path_nodes) - 1):
                current_table = path_nodes[i]
                next_table = path_nodes[i + 1]
                
                edge_data = self.relationship_graph.get_edge_data(current_table, next_table)
                if edge_data:
                    relationship = edge_data.get("relationship")
                    if relationship:
                        relationships.append(relationship)
                        total_cost += edge_data.get("weight", 1.0)
            
            # 평균 신뢰도 계산
            avg_confidence = sum(rel.confidence for rel in relationships) / len(relationships) if relationships else 0.0
            
            join_path = JoinPath(
                source_table=source_table,
                target_table=target_table,
                path=relationships,
                total_cost=total_cost,
                confidence=avg_confidence
            )
            
            logger.debug(f"Found join path: {source_table} -> {target_table} (cost: {total_cost:.2f}, confidence: {avg_confidence:.2f})")
            return join_path
            
        except Exception as e:
            logger.error(f"Failed to find join path from '{source_table}' to '{target_table}': {str(e)}")
            return None
    
    def get_related_tables(self, table_name: str, max_degree: int = 2) -> List[str]:
        """
        특정 테이블과 관련된 테이블 목록 반환
        
        Args:
            table_name: 기준 테이블명
            max_degree: 최대 관계 차수
            
        Returns:
            List[str]: 관련 테이블 목록
        """
        try:
            if table_name not in self.relationship_graph.nodes():
                return []
            
            related_tables = set()
            
            # 직접 관련된 테이블들
            neighbors = list(self.relationship_graph.neighbors(table_name))
            related_tables.update(neighbors)
            
            # 간접 관련된 테이블들 (차수 제한)
            if max_degree > 1:
                for neighbor in neighbors:
                    second_degree_neighbors = list(self.relationship_graph.neighbors(neighbor))
                    related_tables.update(second_degree_neighbors)
            
            # 자기 자신 제거
            related_tables.discard(table_name)
            
            return list(related_tables)
            
        except Exception as e:
            logger.error(f"Failed to get related tables for '{table_name}': {str(e)}")
            return []
    
    def get_relationship_stats(self) -> Dict[str, Any]:
        """관계 통계 정보 반환"""
        try:
            stats = {
                "total_relationships": len(self.relationships),
                "total_tables": self.relationship_graph.number_of_nodes(),
                "total_edges": self.relationship_graph.number_of_edges(),
                "relationship_types": defaultdict(int),
                "average_confidence": 0.0,
                "most_connected_tables": []
            }
            
            if self.relationships:
                # 관계 타입별 통계
                for rel in self.relationships:
                    stats["relationship_types"][rel.relationship_type.value] += 1
                
                # 평균 신뢰도
                stats["average_confidence"] = sum(rel.confidence for rel in self.relationships) / len(self.relationships)
                
                # 가장 많이 연결된 테이블들
                degree_centrality = nx.degree_centrality(self.relationship_graph)
                sorted_tables = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)
                stats["most_connected_tables"] = [(table, centrality) for table, centrality in sorted_tables[:5]]
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get relationship stats: {str(e)}")
            return {}
    
    def validate_relationships(self) -> List[str]:
        """관계 유효성 검증"""
        issues = []
        
        try:
            # 1. 중복 관계 확인 (우선 처리)
            relationship_keys = set()
            duplicate_count = 0
            for rel in self.relationships:
                key = (rel.source_table, rel.target_table, rel.source_column, rel.target_column)
                if key in relationship_keys:
                    duplicate_count += 1
                else:
                    relationship_keys.add(key)
            
            if duplicate_count > 0:
                issues.append(f"Duplicate relationships detected: {duplicate_count} duplicates found")
            
            # 2. 순환 참조 확인 (정상적인 양방향 관계 제외)
            cycles = []
            try:
                # 자체 참조를 제외한 그래프에서 순환 확인
                filtered_graph = self.relationship_graph.copy()
                for edge in list(filtered_graph.edges()):
                    if edge[0] == edge[1]:  # 자체 참조 엣지 제거
                        filtered_graph.remove_edge(edge[0], edge[1])
                
                if not nx.is_directed_acyclic_graph(filtered_graph):
                    cycles = list(nx.simple_cycles(filtered_graph))
                    for cycle in cycles:
                        if len(cycle) > 1:  # 진짜 순환만 (자체 참조 제외)
                            # 정상적인 양방향 관계인지 확인
                            if not self._is_valid_bidirectional_relationship(cycle):
                                issues.append(f"Circular reference detected: {' -> '.join(cycle)}")
                            else:
                                logger.info(f"Valid bidirectional relationship detected: {' -> '.join(cycle)}")
            except Exception as e:
                logger.warning(f"Could not check for cycles: {str(e)}")
            
            # 3. 고아 테이블 확인
            isolated_nodes = list(nx.isolates(self.relationship_graph))
            if isolated_nodes:
                issues.append(f"Isolated tables detected: {isolated_nodes}")
            
            # 4. 낮은 신뢰도 관계 확인
            low_confidence_rels = [rel for rel in self.relationships if rel.confidence < 0.5]
            if low_confidence_rels:
                issues.append(f"Low confidence relationships detected: {len(low_confidence_rels)}")
            
            # 5. 자체 참조 관계 검증
            self_ref_issues = self._validate_self_references()
            issues.extend(self_ref_issues)
            
        except Exception as e:
            logger.error(f"Failed to validate relationships: {str(e)}")
            issues.append(f"Validation error: {str(e)}")
        
        return issues
    
    def _is_valid_bidirectional_relationship(self, cycle: List[str]) -> bool:
        """
        순환 참조가 정상적인 양방향 관계인지 확인
        
        Args:
            cycle: 순환 경로의 테이블 리스트
            
        Returns:
            bool: 정상적인 양방향 관계인지 여부
        """
        try:
            if len(cycle) != 2:
                return False  # 2개 테이블 간의 양방향 관계만 허용
            
            table1, table2 = cycle[0], cycle[1]
            
            # 특정 패턴의 양방향 관계 확인
            valid_patterns = [
                # 상태-로그 관계 패턴
                (('fanding', 'fanding_log'), ('current', 'log')),
                (('payment', 'payment_log'), ('current', 'log')),
                (('member', 'member_log'), ('current', 'log')),
                
                # 메인-서브 관계 패턴
                (('order', 'order_item'), ('main', 'detail')),
                (('invoice', 'invoice_detail'), ('main', 'detail')),
                
                # 계층 구조 관계 패턴
                (('parent', 'child'), ('parent', 'child')),
                (('category', 'subcategory'), ('parent', 'child')),
            ]
            
            # 테이블명 패턴 매칭
            for pattern_pair, relationship_type in valid_patterns:
                pattern1, pattern2 = pattern_pair
                
                # 테이블명에 패턴이 포함되어 있는지 확인
                if (pattern1 in table1.lower() and pattern2 in table2.lower()) or \
                   (pattern2 in table1.lower() and pattern1 in table2.lower()):
                    logger.debug(f"Valid bidirectional pattern detected: {table1} <-> {table2} ({relationship_type})")
                    return True
            
            # 관계 컬럼명으로 판단
            bidirectional_relationships = []
            for rel in self.relationships:
                if (rel.source_table == table1 and rel.target_table == table2) or \
                   (rel.source_table == table2 and rel.target_table == table1):
                    bidirectional_relationships.append(rel)
            
            if len(bidirectional_relationships) >= 2:
                # 양방향 관계가 존재하는 경우, 컬럼명 패턴으로 판단
                for rel in bidirectional_relationships:
                    # 현재 상태를 나타내는 컬럼명 패턴
                    current_patterns = ['current_', 'active_', 'latest_', 'now_']
                    log_patterns = ['_log', '_history', '_record']
                    
                    source_col = rel.source_column.lower()
                    target_col = rel.target_column.lower()
                    
                    # 현재-로그 관계 패턴 확인
                    if any(pattern in source_col for pattern in current_patterns) and \
                       any(pattern in target_col for pattern in log_patterns):
                        logger.debug(f"Valid current-log relationship detected: {rel.source_table}.{rel.source_column} <-> {rel.target_table}.{rel.target_column}")
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to validate bidirectional relationship: {str(e)}")
            return False
    
    def _validate_self_references(self) -> List[str]:
        """자체 참조 관계 검증"""
        issues = []
        
        try:
            self_ref_relationships = [rel for rel in self.relationships if rel.relationship_type == RelationshipType.SELF_REFERENCE]
            
            for rel in self_ref_relationships:
                # 자기 자신을 참조하는 경우 문제
                if rel.source_column == rel.target_column:
                    issues.append(f"Invalid self-reference: {rel.source_table}.{rel.source_column} references itself")
                
                # NULL 값 허용 여부 확인 (실제 구현에서는 컬럼 정보 필요)
                # 여기서는 기본적인 검증만 수행
                
        except Exception as e:
            logger.error(f"Failed to validate self-references: {str(e)}")
        
        return issues
