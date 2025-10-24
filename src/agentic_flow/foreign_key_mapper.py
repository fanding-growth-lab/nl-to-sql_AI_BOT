#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
외래키 관계 자동 탐지 및 매핑 시스템
데이터베이스의 외래키 관계를 자동으로 탐지하고 매핑하는 고급 시스템
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import re
from core.db import get_db_session, execute_query


class ForeignKeyType(Enum):
    """외래키 유형"""
    EXPLICIT = "explicit"  # 명시적 외래키 제약조건
    IMPLICIT = "implicit"  # 암시적 외래키 (이름 패턴 기반)
    DATA_BASED = "data_based"  # 데이터 기반 추론
    SEMANTIC = "semantic"  # 의미적 관계


@dataclass
class ForeignKeyMapping:
    """외래키 매핑 정보"""
    source_table: str
    source_column: str
    target_table: str
    target_column: str
    fk_type: ForeignKeyType
    confidence: float
    constraint_name: Optional[str] = None
    is_cascade_delete: bool = False
    is_cascade_update: bool = False
    description: str = ""
    usage_frequency: int = 0
    data_integrity_score: float = 0.0


@dataclass
class TableRelationship:
    """테이블 관계 정보"""
    table1: str
    table2: str
    relationships: List[ForeignKeyMapping] = field(default_factory=list)
    relationship_strength: float = 0.0
    common_columns: List[str] = field(default_factory=list)
    semantic_similarity: float = 0.0


class ForeignKeyMapper:
    """외래키 매핑 시스템"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.foreign_key_mappings: List[ForeignKeyMapping] = []
        self.table_relationships: Dict[Tuple[str, str], TableRelationship] = {}
        self._naming_patterns = self._initialize_naming_patterns()
        self._semantic_patterns = self._initialize_semantic_patterns()
        
    def _initialize_naming_patterns(self) -> Dict[str, List[str]]:
        """이름 패턴 초기화"""
        return {
            "id_patterns": [
                r"(\w+)_id$",
                r"(\w+)_no$", 
                r"(\w+)_key$",
                r"(\w+)_pk$",
                r"(\w+)id$",
                r"(\w+)no$"
            ],
            "reference_patterns": [
                r"ref_(\w+)$",
                r"(\w+)_ref$",
                r"(\w+)_reference$"
            ],
            "foreign_patterns": [
                r"fk_(\w+)$",
                r"(\w+)_fk$",
                r"foreign_(\w+)$"
            ]
        }
    
    def _initialize_semantic_patterns(self) -> Dict[str, List[str]]:
        """의미적 패턴 초기화"""
        return {
            "user_related": ["user", "member", "customer", "client", "person"],
            "content_related": ["post", "article", "content", "document", "file"],
            "category_related": ["category", "type", "class", "group", "tag"],
            "time_related": ["date", "time", "created", "updated", "modified"],
            "status_related": ["status", "state", "condition", "flag"],
            "location_related": ["address", "location", "place", "region", "country"]
        }
    
    def discover_foreign_keys(self) -> Dict[str, Any]:
        """외래키 관계 자동 탐지"""
        self.logger.info("외래키 관계 자동 탐지 시작")
        
        try:
            # 1. 명시적 외래키 탐지
            explicit_fks = self._discover_explicit_foreign_keys()
            
            # 2. 암시적 외래키 탐지 (이름 패턴 기반)
            implicit_fks = self._discover_implicit_foreign_keys()
            
            # 3. 데이터 기반 외래키 추론
            data_based_fks = self._discover_data_based_foreign_keys()
            
            # 4. 의미적 관계 탐지
            semantic_fks = self._discover_semantic_relationships()
            
            # 5. 모든 외래키 통합 및 중복 제거
            all_fks = explicit_fks + implicit_fks + data_based_fks + semantic_fks
            self.foreign_key_mappings = self._deduplicate_foreign_keys(all_fks)
            
            # 6. 테이블 관계 분석
            self._analyze_table_relationships()
            
            # 7. 사용 빈도 및 데이터 무결성 분석
            self._analyze_usage_patterns()
            
            discovery_result = {
                "total_foreign_keys": len(self.foreign_key_mappings),
                "explicit_foreign_keys": len(explicit_fks),
                "implicit_foreign_keys": len(implicit_fks),
                "data_based_foreign_keys": len(data_based_fks),
                "semantic_foreign_keys": len(semantic_fks),
                "table_relationships": len(self.table_relationships),
                "high_confidence_mappings": len([fk for fk in self.foreign_key_mappings if fk.confidence > 0.8]),
                "mappings": [self._fk_mapping_to_dict(fk) for fk in self.foreign_key_mappings],
                "relationships": [self._table_relationship_to_dict(rel) for rel in self.table_relationships.values()]
            }
            
            self.logger.info(f"외래키 탐지 완료: {len(self.foreign_key_mappings)}개 매핑, {len(self.table_relationships)}개 관계")
            return discovery_result
            
        except Exception as e:
            self.logger.error(f"외래키 탐지 실패: {str(e)}")
            return {"error": str(e)}
    
    def _discover_explicit_foreign_keys(self) -> List[ForeignKeyMapping]:
        """명시적 외래키 탐지"""
        self.logger.info("명시적 외래키 탐지 중...")
        
        explicit_fks = []
        
        try:
            with get_db_session() as session:
                # 외래키 제약조건 조회
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
                    mapping = ForeignKeyMapping(
                        source_table=fk['TABLE_NAME'],
                        source_column=fk['COLUMN_NAME'],
                        target_table=fk['REFERENCED_TABLE_NAME'],
                        target_column=fk['REFERENCED_COLUMN_NAME'],
                        fk_type=ForeignKeyType.EXPLICIT,
                        confidence=1.0,
                        constraint_name=fk['CONSTRAINT_NAME'],
                        is_cascade_delete=False,  # 기본값으로 설정
                        is_cascade_update=False,  # 기본값으로 설정
                        description=f"명시적 외래키: {fk['CONSTRAINT_NAME']}"
                    )
                    
                    explicit_fks.append(mapping)
                
                self.logger.info(f"명시적 외래키 탐지 완료: {len(explicit_fks)}개")
                return explicit_fks
                
        except Exception as e:
            self.logger.error(f"명시적 외래키 탐지 실패: {str(e)}")
            return []
    
    def _discover_implicit_foreign_keys(self) -> List[ForeignKeyMapping]:
        """암시적 외래키 탐지 (이름 패턴 기반)"""
        self.logger.info("암시적 외래키 탐지 중...")
        
        implicit_fks = []
        
        try:
            with get_db_session() as session:
                # 모든 테이블과 컬럼 정보 조회
                tables_query = """
                SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE
                FROM INFORMATION_SCHEMA.COLUMNS 
                WHERE TABLE_SCHEMA = DATABASE()
                ORDER BY TABLE_NAME, ORDINAL_POSITION
                """
                
                tables_result = execute_query(tables_query)
                
                # 테이블별 컬럼 그룹화
                table_columns = defaultdict(list)
                for row in tables_result:
                    table_columns[row['TABLE_NAME']].append({
                        'name': row['COLUMN_NAME'],
                        'type': row['DATA_TYPE']
                    })
                
                # 각 테이블의 컬럼들을 다른 테이블과 매칭
                for source_table, source_columns in table_columns.items():
                    for source_col in source_columns:
                        # 이름 패턴 분석
                        potential_targets = self._analyze_column_naming_patterns(
                            source_col['name'], source_col['type']
                        )
                        
                        for target_table, target_columns in table_columns.items():
                            if source_table == target_table:
                                continue
                                
                            for target_col in target_columns:
                                # 매칭 가능성 계산
                                match_score = self._calculate_naming_match_score(
                                    source_col, target_col, source_table, target_table
                                )
                                
                                if match_score > 0.6:  # 60% 이상 매칭
                                    mapping = ForeignKeyMapping(
                                        source_table=source_table,
                                        source_column=source_col['name'],
                                        target_table=target_table,
                                        target_column=target_col['name'],
                                        fk_type=ForeignKeyType.IMPLICIT,
                                        confidence=match_score,
                                        description=f"이름 패턴 기반 추론: {source_col['name']} -> {target_col['name']}"
                                    )
                                    
                                    implicit_fks.append(mapping)
                
                self.logger.info(f"암시적 외래키 탐지 완료: {len(implicit_fks)}개")
                return implicit_fks
                
        except Exception as e:
            self.logger.error(f"암시적 외래키 탐지 실패: {str(e)}")
            return []
    
    def _discover_data_based_foreign_keys(self) -> List[ForeignKeyMapping]:
        """데이터 기반 외래키 추론"""
        self.logger.info("데이터 기반 외래키 추론 중...")
        
        data_based_fks = []
        
        try:
            with get_db_session() as session:
                # 모든 테이블의 샘플 데이터 수집
                table_samples = self._collect_table_samples()
                
                # 테이블 간 데이터 매칭 분석
                for table1, sample1 in table_samples.items():
                    for table2, sample2 in table_samples.items():
                        if table1 == table2:
                            continue
                        
                        # 컬럼 간 데이터 매칭 분석
                        matches = self._analyze_data_matches(table1, sample1, table2, sample2)
                        
                        for match in matches:
                            if match['confidence'] > 0.5:
                                mapping = ForeignKeyMapping(
                                    source_table=match['source_table'],
                                    source_column=match['source_column'],
                                    target_table=match['target_table'],
                                    target_column=match['target_column'],
                                    fk_type=ForeignKeyType.DATA_BASED,
                                    confidence=match['confidence'],
                                    data_integrity_score=match['integrity_score'],
                                    description=f"데이터 기반 추론: {match['source_column']} -> {match['target_column']} (매칭율: {match['confidence']:.2f})"
                                )
                                
                                data_based_fks.append(mapping)
                
                self.logger.info(f"데이터 기반 외래키 추론 완료: {len(data_based_fks)}개")
                return data_based_fks
                
        except Exception as e:
            self.logger.error(f"데이터 기반 외래키 추론 실패: {str(e)}")
            return []
    
    def _discover_semantic_relationships(self) -> List[ForeignKeyMapping]:
        """의미적 관계 탐지"""
        self.logger.info("의미적 관계 탐지 중...")
        
        semantic_fks = []
        
        try:
            with get_db_session() as session:
                # 테이블과 컬럼의 의미적 분석
                tables_query = """
                SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE
                FROM INFORMATION_SCHEMA.COLUMNS 
                WHERE TABLE_SCHEMA = DATABASE()
                ORDER BY TABLE_NAME, ORDINAL_POSITION
                """
                
                tables_result = execute_query(tables_query)
                
                # 의미적 유사도 분석
                for row1 in tables_result:
                    for row2 in tables_result:
                        if row1['TABLE_NAME'] == row2['TABLE_NAME']:
                            continue
                        
                        # 의미적 유사도 계산
                        semantic_score = self._calculate_semantic_similarity(
                            row1['TABLE_NAME'], row1['COLUMN_NAME'],
                            row2['TABLE_NAME'], row2['COLUMN_NAME']
                        )
                        
                        if semantic_score > 0.7:
                            mapping = ForeignKeyMapping(
                                source_table=row1['TABLE_NAME'],
                                source_column=row1['COLUMN_NAME'],
                                target_table=row2['TABLE_NAME'],
                                target_column=row2['COLUMN_NAME'],
                                fk_type=ForeignKeyType.SEMANTIC,
                                confidence=semantic_score,
                                description=f"의미적 관계: {row1['COLUMN_NAME']} -> {row2['COLUMN_NAME']} (유사도: {semantic_score:.2f})"
                            )
                            
                            semantic_fks.append(mapping)
                
                self.logger.info(f"의미적 관계 탐지 완료: {len(semantic_fks)}개")
                return semantic_fks
                
        except Exception as e:
            self.logger.error(f"의미적 관계 탐지 실패: {str(e)}")
            return []
    
    def _analyze_column_naming_patterns(self, column_name: str, data_type: str) -> List[str]:
        """컬럼 이름 패턴 분석"""
        patterns = []
        
        # ID 패턴 분석
        for pattern in self._naming_patterns["id_patterns"]:
            match = re.match(pattern, column_name.lower())
            if match:
                base_name = match.group(1)
                patterns.append(base_name)
        
        return patterns
    
    def _calculate_naming_match_score(self, source_col: Dict, target_col: Dict, 
                                    source_table: str, target_table: str) -> float:
        """이름 매칭 점수 계산"""
        score = 0.0
        
        # 1. 컬럼명 유사도
        col_similarity = self._calculate_string_similarity(
            source_col['name'].lower(), target_col['name'].lower()
        )
        score += col_similarity * 0.4
        
        # 2. 테이블명 유사도
        table_similarity = self._calculate_string_similarity(
            source_table.lower(), target_table.lower()
        )
        score += table_similarity * 0.3
        
        # 3. 데이터 타입 일치
        if source_col['type'] == target_col['type']:
            score += 0.3
        
        return min(score, 1.0)
    
    def _calculate_string_similarity(self, str1: str, str2: str) -> float:
        """문자열 유사도 계산 (Jaccard 유사도)"""
        set1 = set(str1)
        set2 = set(str2)
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def _collect_table_samples(self, sample_size: int = 100) -> Dict[str, List[Dict]]:
        """테이블 샘플 데이터 수집"""
        samples = {}
        
        try:
            with get_db_session() as session:
                # 모든 테이블 조회
                tables_query = """
                SELECT TABLE_NAME 
                FROM INFORMATION_SCHEMA.TABLES 
                WHERE TABLE_SCHEMA = DATABASE()
                ORDER BY TABLE_NAME
                """
                
                tables_result = execute_query(tables_query)
                
                for table_row in tables_result:
                    table_name = table_row['TABLE_NAME']
                    
                    try:
                        # 샘플 데이터 조회
                        sample_query = f"SELECT * FROM {table_name} LIMIT {sample_size}"
                        sample_data = execute_query(sample_query)
                        
                        if sample_data:
                            samples[table_name] = sample_data
                            
                    except Exception as e:
                        self.logger.warning(f"테이블 {table_name} 샘플 데이터 수집 실패: {str(e)}")
                        continue
                
                self.logger.info(f"테이블 샘플 데이터 수집 완료: {len(samples)}개 테이블")
                return samples
                
        except Exception as e:
            self.logger.error(f"테이블 샘플 데이터 수집 실패: {str(e)}")
            return {}
    
    def _analyze_data_matches(self, table1: str, sample1: List[Dict], 
                            table2: str, sample2: List[Dict]) -> List[Dict]:
        """데이터 매칭 분석"""
        matches = []
        
        if not sample1 or not sample2:
            return matches
        
        # 각 컬럼 쌍에 대해 데이터 매칭 분석
        for col1_name in sample1[0].keys():
            for col2_name in sample2[0].keys():
                # 데이터 타입 확인
                if not self._are_compatible_types(sample1, col1_name, sample2, col2_name):
                    continue
                
                # 데이터 매칭 비율 계산
                match_ratio = self._calculate_data_match_ratio(sample1, col1_name, sample2, col2_name)
                integrity_score = self._calculate_data_integrity_score(sample1, col1_name, sample2, col2_name)
                
                if match_ratio > 0.3:  # 30% 이상 매칭
                    matches.append({
                        'source_table': table1,
                        'source_column': col1_name,
                        'target_table': table2,
                        'target_column': col2_name,
                        'confidence': match_ratio,
                        'integrity_score': integrity_score
                    })
        
        return matches
    
    def _are_compatible_types(self, sample1: List[Dict], col1: str, 
                            sample2: List[Dict], col2: str) -> bool:
        """데이터 타입 호환성 확인"""
        # 간단한 타입 호환성 체크
        try:
            val1 = sample1[0].get(col1)
            val2 = sample2[0].get(col2)
            
            # 둘 다 숫자인지 확인
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                return True
            
            # 둘 다 문자열인지 확인
            if isinstance(val1, str) and isinstance(val2, str):
                return True
            
            return False
            
        except:
            return False
    
    def _calculate_data_match_ratio(self, sample1: List[Dict], col1: str, 
                                  sample2: List[Dict], col2: str) -> float:
        """데이터 매칭 비율 계산"""
        values1 = set(str(row.get(col1, '')) for row in sample1 if col1 in row)
        values2 = set(str(row.get(col2, '')) for row in sample2 if col2 in row)
        
        if not values1 or not values2:
            return 0.0
        
        intersection = values1.intersection(values2)
        union = values1.union(values2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _calculate_data_integrity_score(self, sample1: List[Dict], col1: str, 
                                      sample2: List[Dict], col2: str) -> float:
        """데이터 무결성 점수 계산"""
        # NULL 값 비율 계산
        null_count1 = sum(1 for row in sample1 if col1 in row and row[col1] is None)
        null_count2 = sum(1 for row in sample2 if col2 in row and row[col2] is None)
        
        null_ratio1 = null_count1 / len(sample1) if sample1 else 0
        null_ratio2 = null_count2 / len(sample2) if sample2 else 0
        
        # NULL 비율이 낮을수록 높은 무결성
        integrity1 = 1.0 - null_ratio1
        integrity2 = 1.0 - null_ratio2
        
        return (integrity1 + integrity2) / 2.0
    
    def _calculate_semantic_similarity(self, table1: str, col1: str, 
                                     table2: str, col2: str) -> float:
        """의미적 유사도 계산"""
        score = 0.0
        
        # 테이블명 의미적 유사도
        table_similarity = self._calculate_semantic_table_similarity(table1, table2)
        score += table_similarity * 0.5
        
        # 컬럼명 의미적 유사도
        col_similarity = self._calculate_semantic_column_similarity(col1, col2)
        score += col_similarity * 0.5
        
        return min(score, 1.0)
    
    def _calculate_semantic_table_similarity(self, table1: str, table2: str) -> float:
        """테이블 의미적 유사도 계산"""
        table1_lower = table1.lower()
        table2_lower = table2.lower()
        
        # 의미적 패턴 매칭
        for category, keywords in self._semantic_patterns.items():
            match1 = any(keyword in table1_lower for keyword in keywords)
            match2 = any(keyword in table2_lower for keyword in keywords)
            
            if match1 and match2:
                return 0.8  # 같은 카테고리에 속함
        
        return 0.0
    
    def _calculate_semantic_column_similarity(self, col1: str, col2: str) -> float:
        """컬럼 의미적 유사도 계산"""
        col1_lower = col1.lower()
        col2_lower = col2.lower()
        
        # 공통 키워드 매칭
        for category, keywords in self._semantic_patterns.items():
            match1 = any(keyword in col1_lower for keyword in keywords)
            match2 = any(keyword in col2_lower for keyword in keywords)
            
            if match1 and match2:
                return 0.9  # 같은 의미적 카테고리
        
        return 0.0
    
    def _deduplicate_foreign_keys(self, all_fks: List[ForeignKeyMapping]) -> List[ForeignKeyMapping]:
        """외래키 중복 제거"""
        unique_fks = []
        seen_mappings = set()
        
        # 신뢰도 순으로 정렬
        all_fks.sort(key=lambda x: x.confidence, reverse=True)
        
        for fk in all_fks:
            mapping_key = (fk.source_table, fk.source_column, fk.target_table, fk.target_column)
            
            if mapping_key not in seen_mappings:
                unique_fks.append(fk)
                seen_mappings.add(mapping_key)
        
        return unique_fks
    
    def _analyze_table_relationships(self):
        """테이블 관계 분석"""
        self.logger.info("테이블 관계 분석 중...")
        
        # 관계별로 그룹화
        relationship_groups = defaultdict(list)
        
        for fk in self.foreign_key_mappings:
            # 양방향 관계 키 생성
            key1 = (fk.source_table, fk.target_table)
            key2 = (fk.target_table, fk.source_table)
            
            relationship_groups[key1].append(fk)
            relationship_groups[key2].append(fk)
        
        # 각 관계 그룹 분석
        for (table1, table2), fks in relationship_groups.items():
            if len(fks) == 0:
                continue
            
            # 관계 강도 계산
            relationship_strength = sum(fk.confidence for fk in fks) / len(fks)
            
            # 공통 컬럼 찾기
            common_columns = self._find_common_columns(table1, table2)
            
            # 의미적 유사도 계산
            semantic_similarity = self._calculate_semantic_table_similarity(table1, table2)
            
            relationship = TableRelationship(
                table1=table1,
                table2=table2,
                relationships=fks,
                relationship_strength=relationship_strength,
                common_columns=common_columns,
                semantic_similarity=semantic_similarity
            )
            
            self.table_relationships[(table1, table2)] = relationship
        
        self.logger.info(f"테이블 관계 분석 완료: {len(self.table_relationships)}개 관계")
    
    def _find_common_columns(self, table1: str, table2: str) -> List[str]:
        """공통 컬럼 찾기"""
        # 실제 구현에서는 테이블 스키마를 조회해야 함
        # 여기서는 간단한 예시만 제공
        return []
    
    def _analyze_usage_patterns(self):
        """사용 패턴 분석"""
        self.logger.info("사용 패턴 분석 중...")
        
        # 각 외래키의 사용 빈도 및 데이터 무결성 분석
        for fk in self.foreign_key_mappings:
            try:
                # 사용 빈도 계산 (간단한 예시)
                fk.usage_frequency = self._calculate_usage_frequency(fk)
                
                # 데이터 무결성 점수 계산
                fk.data_integrity_score = self._calculate_data_integrity_score_for_fk(fk)
                
            except Exception as e:
                self.logger.warning(f"외래키 {fk.source_table}.{fk.source_column} 사용 패턴 분석 실패: {str(e)}")
                continue
        
        self.logger.info("사용 패턴 분석 완료")
    
    def _calculate_usage_frequency(self, fk: ForeignKeyMapping) -> int:
        """사용 빈도 계산"""
        # 실제 구현에서는 쿼리 로그나 통계를 분석
        # 여기서는 간단한 예시
        return 1
    
    def _calculate_data_integrity_score_for_fk(self, fk: ForeignKeyMapping) -> float:
        """외래키 데이터 무결성 점수 계산"""
        # 실제 구현에서는 데이터 무결성 검사를 수행
        # 여기서는 간단한 예시
        return fk.confidence
    
    def get_foreign_keys_for_table(self, table_name: str) -> List[ForeignKeyMapping]:
        """특정 테이블의 외래키 목록 반환"""
        return [fk for fk in self.foreign_key_mappings 
                if fk.source_table == table_name or fk.target_table == table_name]
    
    def get_join_path(self, from_table: str, to_table: str) -> Optional[List[ForeignKeyMapping]]:
        """두 테이블 간의 조인 경로 반환"""
        # BFS를 사용한 최단 경로 찾기
        from collections import deque
        
        if from_table == to_table:
            return []
        
        # 그래프 구성
        graph = defaultdict(list)
        for fk in self.foreign_key_mappings:
            graph[fk.source_table].append((fk.target_table, fk))
            graph[fk.target_table].append((fk.source_table, fk))
        
        # BFS로 경로 찾기
        queue = deque([(from_table, [])])
        visited = {from_table}
        
        while queue:
            current_table, path = queue.popleft()
            
            if current_table == to_table:
                return path
            
            for next_table, fk in graph[current_table]:
                if next_table not in visited:
                    visited.add(next_table)
                    new_path = path + [fk]
                    queue.append((next_table, new_path))
        
        return None
    
    def _fk_mapping_to_dict(self, fk: ForeignKeyMapping) -> Dict[str, Any]:
        """외래키 매핑을 딕셔너리로 변환"""
        return {
            "source_table": fk.source_table,
            "source_column": fk.source_column,
            "target_table": fk.target_table,
            "target_column": fk.target_column,
            "fk_type": fk.fk_type.value,
            "confidence": fk.confidence,
            "constraint_name": fk.constraint_name,
            "is_cascade_delete": fk.is_cascade_delete,
            "is_cascade_update": fk.is_cascade_update,
            "description": fk.description,
            "usage_frequency": fk.usage_frequency,
            "data_integrity_score": fk.data_integrity_score
        }
    
    def _table_relationship_to_dict(self, rel: TableRelationship) -> Dict[str, Any]:
        """테이블 관계를 딕셔너리로 변환"""
        return {
            "table1": rel.table1,
            "table2": rel.table2,
            "relationships": [self._fk_mapping_to_dict(fk) for fk in rel.relationships],
            "relationship_strength": rel.relationship_strength,
            "common_columns": rel.common_columns,
            "semantic_similarity": rel.semantic_similarity
        }

