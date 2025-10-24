#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
데이터베이스 스키마 관계 분석 시스템
관계형 데이터베이스의 테이블 간 관계를 분석하고 매핑하는 시스템
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import networkx as nx
from core.db import get_db_session, execute_query


class RelationshipType(Enum):
    """관계 유형"""
    ONE_TO_ONE = "one_to_one"
    ONE_TO_MANY = "one_to_many"
    MANY_TO_ONE = "many_to_one"
    MANY_TO_MANY = "many_to_many"
    SELF_REFERENCE = "self_reference"


@dataclass
class TableColumn:
    """테이블 컬럼 정보"""
    name: str
    data_type: str
    is_nullable: bool
    is_primary_key: bool = False
    is_foreign_key: bool = False
    referenced_table: Optional[str] = None
    referenced_column: Optional[str] = None
    default_value: Optional[str] = None


@dataclass
class TableInfo:
    """테이블 정보"""
    name: str
    columns: Dict[str, TableColumn] = field(default_factory=dict)
    primary_keys: List[str] = field(default_factory=list)
    foreign_keys: List[str] = field(default_factory=list)
    indexes: List[str] = field(default_factory=list)
    row_count: int = 0


@dataclass
class Relationship:
    """테이블 간 관계 정보"""
    from_table: str
    to_table: str
    from_column: str
    to_column: str
    relationship_type: RelationshipType
    confidence: float
    is_optional: bool = False
    description: str = ""


@dataclass
class JoinPath:
    """조인 경로 정보"""
    tables: List[str]
    joins: List[Tuple[str, str, str, str]]  # (table1, column1, table2, column2)
    cost: float
    confidence: float
    description: str = ""


class DatabaseRelationshipAnalyzer:
    """데이터베이스 관계 분석기"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.tables: Dict[str, TableInfo] = {}
        self.relationships: List[Relationship] = []
        self.relationship_graph: nx.DiGraph = nx.DiGraph()
        self._schema_cache: Dict[str, Any] = {}
        
    def analyze_database_schema(self) -> Dict[str, Any]:
        """데이터베이스 스키마 전체 분석"""
        self.logger.info("데이터베이스 스키마 관계 분석 시작")
        
        try:
            # 1. 모든 테이블 정보 수집
            self._collect_table_information()
            
            # 2. 외래키 관계 분석
            self._analyze_foreign_key_relationships()
            
            # 3. 이름 기반 관계 추론
            self._infer_relationships_by_naming()
            
            # 4. 데이터 기반 관계 추론
            self._infer_relationships_by_data()
            
            # 5. 관계 그래프 구성
            self._build_relationship_graph()
            
            # 6. 관계 최적화
            self._optimize_relationships()
            
            analysis_result = {
                "tables": {name: self._table_to_dict(table) for name, table in self.tables.items()},
                "relationships": [self._relationship_to_dict(rel) for rel in self.relationships],
                "graph_stats": self._get_graph_statistics(),
                "join_paths": self._find_common_join_paths(),
                "analysis_summary": self._generate_analysis_summary()
            }
            
            self.logger.info(f"스키마 분석 완료: {len(self.tables)}개 테이블, {len(self.relationships)}개 관계")
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"스키마 분석 실패: {str(e)}")
            return {"error": str(e)}
    
    def _collect_table_information(self):
        """테이블 정보 수집"""
        self.logger.info("테이블 정보 수집 중...")
        
        try:
            with get_db_session() as session:
                # 모든 테이블 조회
                tables_query = """
                SELECT TABLE_NAME, TABLE_ROWS
                FROM INFORMATION_SCHEMA.TABLES 
                WHERE TABLE_SCHEMA = DATABASE()
                ORDER BY TABLE_NAME
                """
                
                tables_result = execute_query(tables_query)
                
                for table_row in tables_result:
                    table_name = table_row['TABLE_NAME']
                    row_count = table_row.get('TABLE_ROWS', 0)
                    
                    # 테이블 정보 생성
                    table_info = TableInfo(
                        name=table_name,
                        row_count=row_count
                    )
                    
                    # 컬럼 정보 조회
                    columns_query = f"""
                    SELECT 
                        COLUMN_NAME, 
                        DATA_TYPE, 
                        IS_NULLABLE, 
                        COLUMN_KEY,
                        COLUMN_DEFAULT,
                        EXTRA
                    FROM INFORMATION_SCHEMA.COLUMNS 
                    WHERE TABLE_SCHEMA = DATABASE() 
                    AND TABLE_NAME = '{table_name}'
                    ORDER BY ORDINAL_POSITION
                    """
                    
                    columns_result = execute_query(columns_query)
                    
                    for col in columns_result:
                        column = TableColumn(
                            name=col['COLUMN_NAME'],
                            data_type=col['DATA_TYPE'],
                            is_nullable=col['IS_NULLABLE'] == 'YES',
                            is_primary_key=col['COLUMN_KEY'] == 'PRI',
                            default_value=col['COLUMN_DEFAULT']
                        )
                        
                        table_info.columns[column.name] = column
                        
                        if column.is_primary_key:
                            table_info.primary_keys.append(column.name)
                    
                    self.tables[table_name] = table_info
                
                self.logger.info(f"테이블 정보 수집 완료: {len(self.tables)}개 테이블")
                
        except Exception as e:
            self.logger.error(f"테이블 정보 수집 실패: {str(e)}")
            raise
    
    def _analyze_foreign_key_relationships(self):
        """외래키 관계 분석"""
        self.logger.info("외래키 관계 분석 중...")
        
        try:
            with get_db_session() as session:
                # 외래키 관계 조회
                fk_query = """
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
                
                fk_result = execute_query(fk_query)
                
                for fk in fk_result:
                    from_table = fk['TABLE_NAME']
                    from_column = fk['COLUMN_NAME']
                    to_table = fk['REFERENCED_TABLE_NAME']
                    to_column = fk['REFERENCED_COLUMN_NAME']
                    
                    # 외래키 관계 생성
                    relationship = Relationship(
                        from_table=from_table,
                        to_table=to_table,
                        from_column=from_column,
                        to_column=to_column,
                        relationship_type=RelationshipType.MANY_TO_ONE,
                        confidence=1.0,
                        is_optional=True,
                        description=f"FK: {from_table}.{from_column} -> {to_table}.{to_column}"
                    )
                    
                    self.relationships.append(relationship)
                    
                    # 테이블 정보에 외래키 표시
                    if from_table in self.tables and from_column in self.tables[from_table].columns:
                        self.tables[from_table].columns[from_column].is_foreign_key = True
                        self.tables[from_table].columns[from_column].referenced_table = to_table
                        self.tables[from_table].columns[from_column].referenced_column = to_column
                        self.tables[from_table].foreign_keys.append(from_column)
                
                self.logger.info(f"외래키 관계 분석 완료: {len(fk_result)}개 관계")
                
        except Exception as e:
            self.logger.error(f"외래키 관계 분석 실패: {str(e)}")
            raise
    
    def _infer_relationships_by_naming(self):
        """이름 기반 관계 추론"""
        self.logger.info("이름 기반 관계 추론 중...")
        
        # 공통 이름 패턴으로 관계 추론
        for table_name, table_info in self.tables.items():
            for column_name, column in table_info.columns.items():
                # 다른 테이블과의 이름 매칭 시도
                for other_table_name, other_table_info in self.tables.items():
                    if table_name == other_table_name:
                        continue
                    
                    # 컬럼명이 다른 테이블명과 유사한 경우
                    if self._is_name_similar(column_name, other_table_name):
                        # 해당 테이블의 기본키와 매칭 시도
                        for pk in other_table_info.primary_keys:
                            if self._is_name_similar(column_name, pk):
                                relationship = Relationship(
                                    from_table=table_name,
                                    to_table=other_table_name,
                                    from_column=column_name,
                                    to_column=pk,
                                    relationship_type=RelationshipType.MANY_TO_ONE,
                                    confidence=0.7,
                                    is_optional=True,
                                    description=f"이름 기반 추론: {table_name}.{column_name} -> {other_table_name}.{pk}"
                                )
                                
                                # 중복 관계 확인
                                if not self._relationship_exists(relationship):
                                    self.relationships.append(relationship)
    
    def _infer_relationships_by_data(self):
        """데이터 기반 관계 추론"""
        self.logger.info("데이터 기반 관계 추론 중...")
        
        # 샘플 데이터를 기반으로 관계 추론
        for table_name, table_info in self.tables.items():
            if table_info.row_count == 0:
                continue
                
            try:
                # 각 테이블의 샘플 데이터 조회
                sample_query = f"SELECT * FROM {table_name} LIMIT 10"
                sample_data = execute_query(sample_query)
                
                if sample_data:
                    # 다른 테이블과의 데이터 매칭 시도
                    for other_table_name, other_table_info in self.tables.items():
                        if table_name == other_table_name or other_table_info.row_count == 0:
                            continue
                        
                        # 데이터 기반 관계 추론 로직
                        inferred_relationships = self._analyze_data_patterns(
                            table_name, table_info, sample_data,
                            other_table_name, other_table_info
                        )
                        
                        for rel in inferred_relationships:
                            if not self._relationship_exists(rel):
                                self.relationships.append(rel)
                                
            except Exception as e:
                self.logger.warning(f"테이블 {table_name} 데이터 분석 실패: {str(e)}")
                continue
    
    def _analyze_data_patterns(self, table1: str, table1_info: TableInfo, 
                             sample_data: List[Dict], table2: str, table2_info: TableInfo) -> List[Relationship]:
        """데이터 패턴 분석으로 관계 추론"""
        relationships = []
        
        try:
            # 테이블2의 샘플 데이터도 조회
            sample2_query = f"SELECT * FROM {table2} LIMIT 10"
            sample2_data = execute_query(sample2_query)
            
            if not sample2_data:
                return relationships
            
            # 컬럼 간 데이터 매칭 분석
            for col1_name, col1_info in table1_info.columns.items():
                for col2_name, col2_info in table2_info.columns.items():
                    # 데이터 타입이 같은 경우
                    if col1_info.data_type == col2_info.data_type:
                        # 샘플 데이터에서 값 매칭 확인
                        matching_ratio = self._calculate_data_matching_ratio(
                            sample_data, col1_name, sample2_data, col2_name
                        )
                        
                        if matching_ratio > 0.3:  # 30% 이상 매칭
                            relationship = Relationship(
                                from_table=table1,
                                to_table=table2,
                                from_column=col1_name,
                                to_column=col2_name,
                                relationship_type=RelationshipType.MANY_TO_ONE,
                                confidence=matching_ratio,
                                is_optional=True,
                                description=f"데이터 기반 추론: {table1}.{col1_name} -> {table2}.{col2_name} (매칭율: {matching_ratio:.2f})"
                            )
                            relationships.append(relationship)
        
        except Exception as e:
            self.logger.warning(f"데이터 패턴 분석 실패 ({table1} -> {table2}): {str(e)}")
        
        return relationships
    
    def _calculate_data_matching_ratio(self, data1: List[Dict], col1: str, 
                                      data2: List[Dict], col2: str) -> float:
        """데이터 매칭 비율 계산"""
        if not data1 or not data2:
            return 0.0
        
        values1 = set(str(row.get(col1, '')) for row in data1 if col1 in row)
        values2 = set(str(row.get(col2, '')) for row in data2 if col2 in row)
        
        if not values1 or not values2:
            return 0.0
        
        intersection = values1.intersection(values2)
        union = values1.union(values2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _build_relationship_graph(self):
        """관계 그래프 구성"""
        self.logger.info("관계 그래프 구성 중...")
        
        # 그래프 초기화
        self.relationship_graph = nx.DiGraph()
        
        # 노드 추가 (테이블)
        for table_name in self.tables.keys():
            self.relationship_graph.add_node(table_name)
        
        # 엣지 추가 (관계)
        for rel in self.relationships:
            self.relationship_graph.add_edge(
                rel.from_table, 
                rel.to_table,
                relationship=rel,
                weight=1.0 / rel.confidence  # 신뢰도가 높을수록 가중치 낮음
            )
        
        self.logger.info(f"관계 그래프 구성 완료: {self.relationship_graph.number_of_nodes()}개 노드, {self.relationship_graph.number_of_edges()}개 엣지")
    
    def _optimize_relationships(self):
        """관계 최적화"""
        self.logger.info("관계 최적화 중...")
        
        # 중복 관계 제거
        unique_relationships = []
        seen_relationships = set()
        
        for rel in self.relationships:
            rel_key = (rel.from_table, rel.to_table, rel.from_column, rel.to_column)
            if rel_key not in seen_relationships:
                unique_relationships.append(rel)
                seen_relationships.add(rel_key)
            else:
                # 중복 관계 중 신뢰도가 높은 것 선택
                for i, existing_rel in enumerate(unique_relationships):
                    if (existing_rel.from_table == rel.from_table and 
                        existing_rel.to_table == rel.to_table and
                        existing_rel.from_column == rel.from_column and
                        existing_rel.to_column == rel.to_column):
                        if rel.confidence > existing_rel.confidence:
                            unique_relationships[i] = rel
                        break
        
        self.relationships = unique_relationships
        self.logger.info(f"관계 최적화 완료: {len(self.relationships)}개 관계")
    
    def find_join_path(self, from_table: str, to_table: str) -> Optional[JoinPath]:
        """두 테이블 간 조인 경로 찾기"""
        try:
            if from_table not in self.relationship_graph or to_table not in self.relationship_graph:
                return None
            
            # 최단 경로 찾기
            try:
                path = nx.shortest_path(self.relationship_graph, from_table, to_table)
            except nx.NetworkXNoPath:
                return None
            
            # 조인 경로 구성
            joins = []
            for i in range(len(path) - 1):
                current_table = path[i]
                next_table = path[i + 1]
                
                # 두 테이블 간의 관계 찾기
                relationship = self._find_relationship_between_tables(current_table, next_table)
                if relationship:
                    joins.append((
                        current_table, relationship.from_column,
                        next_table, relationship.to_column
                    ))
            
            # 경로 비용 계산
            cost = len(path) - 1
            confidence = min(rel.confidence for rel in self._get_relationships_in_path(path))
            
            return JoinPath(
                tables=path,
                joins=joins,
                cost=cost,
                confidence=confidence,
                description=f"{from_table} -> {to_table} 경로"
            )
            
        except Exception as e:
            self.logger.error(f"조인 경로 찾기 실패 ({from_table} -> {to_table}): {str(e)}")
            return None
    
    def _find_relationship_between_tables(self, table1: str, table2: str) -> Optional[Relationship]:
        """두 테이블 간의 관계 찾기"""
        for rel in self.relationships:
            if rel.from_table == table1 and rel.to_table == table2:
                return rel
        return None
    
    def _get_relationships_in_path(self, path: List[str]) -> List[Relationship]:
        """경로 내의 모든 관계 반환"""
        relationships = []
        for i in range(len(path) - 1):
            rel = self._find_relationship_between_tables(path[i], path[i + 1])
            if rel:
                relationships.append(rel)
        return relationships
    
    def _find_common_join_paths(self) -> List[JoinPath]:
        """자주 사용되는 조인 경로 찾기"""
        common_paths = []
        
        # 모든 테이블 쌍에 대해 경로 찾기
        table_names = list(self.tables.keys())
        
        for i, table1 in enumerate(table_names):
            for table2 in table_names[i+1:]:
                path = self.find_join_path(table1, table2)
                if path and path.confidence > 0.5:
                    common_paths.append(path)
        
        # 신뢰도와 비용을 기준으로 정렬
        common_paths.sort(key=lambda x: (x.confidence, -x.cost), reverse=True)
        
        return common_paths[:10]  # 상위 10개 경로 반환
    
    def _get_graph_statistics(self) -> Dict[str, Any]:
        """그래프 통계 정보"""
        if not self.relationship_graph:
            return {}
        
        return {
            "nodes": self.relationship_graph.number_of_nodes(),
            "edges": self.relationship_graph.number_of_edges(),
            "density": nx.density(self.relationship_graph),
            "is_connected": nx.is_weakly_connected(self.relationship_graph),
            "strongly_connected_components": len(list(nx.strongly_connected_components(self.relationship_graph))),
            "weakly_connected_components": len(list(nx.weakly_connected_components(self.relationship_graph)))
        }
    
    def _generate_analysis_summary(self) -> Dict[str, Any]:
        """분석 요약 정보"""
        return {
            "total_tables": len(self.tables),
            "total_relationships": len(self.relationships),
            "foreign_key_relationships": len([r for r in self.relationships if r.confidence == 1.0]),
            "inferred_relationships": len([r for r in self.relationships if r.confidence < 1.0]),
            "high_confidence_relationships": len([r for r in self.relationships if r.confidence > 0.8]),
            "tables_with_relationships": len(set(r.from_table for r in self.relationships) | set(r.to_table for r in self.relationships))
        }
    
    # 유틸리티 메서드들
    def _is_name_similar(self, name1: str, name2: str) -> bool:
        """이름 유사도 확인"""
        name1_lower = name1.lower()
        name2_lower = name2.lower()
        
        # 정확한 매칭
        if name1_lower == name2_lower:
            return True
        
        # 부분 매칭
        if name1_lower in name2_lower or name2_lower in name1_lower:
            return True
        
        # 공통 접미사/접두사
        common_suffixes = ['_id', '_no', '_key', '_pk']
        for suffix in common_suffixes:
            if name1_lower.endswith(suffix) and name2_lower.endswith(suffix):
                base1 = name1_lower[:-len(suffix)]
                base2 = name2_lower[:-len(suffix)]
                if base1 == base2:
                    return True
        
        return False
    
    def _relationship_exists(self, relationship: Relationship) -> bool:
        """관계 중복 확인"""
        for existing_rel in self.relationships:
            if (existing_rel.from_table == relationship.from_table and
                existing_rel.to_table == relationship.to_table and
                existing_rel.from_column == relationship.from_column and
                existing_rel.to_column == relationship.to_column):
                return True
        return False
    
    def _table_to_dict(self, table: TableInfo) -> Dict[str, Any]:
        """테이블 정보를 딕셔너리로 변환"""
        return {
            "name": table.name,
            "columns": {name: {
                "name": col.name,
                "data_type": col.data_type,
                "is_nullable": col.is_nullable,
                "is_primary_key": col.is_primary_key,
                "is_foreign_key": col.is_foreign_key,
                "referenced_table": col.referenced_table,
                "referenced_column": col.referenced_column,
                "default_value": col.default_value
            } for name, col in table.columns.items()},
            "primary_keys": table.primary_keys,
            "foreign_keys": table.foreign_keys,
            "row_count": table.row_count
        }
    
    def _relationship_to_dict(self, relationship: Relationship) -> Dict[str, Any]:
        """관계 정보를 딕셔너리로 변환"""
        return {
            "from_table": relationship.from_table,
            "to_table": relationship.to_table,
            "from_column": relationship.from_column,
            "to_column": relationship.to_column,
            "relationship_type": relationship.relationship_type.value,
            "confidence": relationship.confidence,
            "is_optional": relationship.is_optional,
            "description": relationship.description
        }


