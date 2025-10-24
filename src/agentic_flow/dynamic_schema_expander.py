"""
동적 스키마 확장 시스템
- 사용자 쿼리 패턴 분석
- 새로운 테이블/컬럼 조합 자동 학습
- RAG 패턴 동적 생성
"""

import json
import logging
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import difflib

logger = logging.getLogger(__name__)

@dataclass
class SchemaPattern:
    """스키마 패턴 데이터 클래스"""
    table: str
    columns: List[str]
    query_patterns: List[str]
    sql_template: str
    confidence: float
    usage_count: int = 0
    last_used: Optional[datetime] = None

class DynamicSchemaExpander:
    """동적 스키마 확장 시스템"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 확장된 스키마 정보
        self.extended_schema = self._load_extended_schema()
        
        # 학습된 패턴들
        self.learned_patterns: Dict[str, SchemaPattern] = {}
        
        # 사용자 쿼리 히스토리
        self.query_history: List[Dict[str, Any]] = []
        
    def _load_extended_schema(self) -> Dict[str, Dict[str, Any]]:
        """확장된 스키마 정보 로드"""
        return {
            "t_member": {
                "description": "회원 정보",
                "columns": {
                    "no": {"type": "INT", "description": "회원 ID (Primary Key)"},
                    "name": {"type": "VARCHAR", "description": "회원명"},
                    "email": {"type": "VARCHAR", "description": "이메일"},
                    "phone": {"type": "VARCHAR", "description": "전화번호"},
                    "status": {"type": "CHAR", "description": "상태 (A: 활성, I: 비활성, D: 삭제)"},
                    "ins_datetime": {"type": "DATETIME", "description": "가입일시"},
                    "mod_datetime": {"type": "DATETIME", "description": "수정일시"},
                    "del_datetime": {"type": "DATETIME", "description": "삭제일시"},
                    "birth_date": {"type": "DATE", "description": "생년월일"},
                    "gender": {"type": "CHAR", "description": "성별"},
                    "address": {"type": "TEXT", "description": "주소"}
                },
                "relationships": ["t_payment", "t_member_login_log", "t_post"]
            },
            "t_payment": {
                "description": "결제 정보",
                "columns": {
                    "no": {"type": "INT", "description": "결제 ID (Primary Key)"},
                    "member_no": {"type": "INT", "description": "회원 ID (FK)"},
                    "price": {"type": "DECIMAL", "description": "결제 금액"},
                    "status": {"type": "VARCHAR", "description": "결제 상태"},
                    "payment_method": {"type": "VARCHAR", "description": "결제 방법"},
                    "ins_datetime": {"type": "DATETIME", "description": "결제일시"},
                    "mod_datetime": {"type": "DATETIME", "description": "수정일시"},
                    "refund_amount": {"type": "DECIMAL", "description": "환불 금액"},
                    "refund_datetime": {"type": "DATETIME", "description": "환불일시"}
                },
                "relationships": ["t_member"]
            },
            "t_post": {
                "description": "게시글 정보",
                "columns": {
                    "no": {"type": "INT", "description": "게시글 ID (Primary Key)"},
                    "title": {"type": "VARCHAR", "description": "제목"},
                    "content": {"type": "TEXT", "description": "내용"},
                    "view_count": {"type": "INT", "description": "조회수"},
                    "like_count": {"type": "INT", "description": "좋아요 수"},
                    "ins_datetime": {"type": "DATETIME", "description": "작성일시"},
                    "mod_datetime": {"type": "DATETIME", "description": "수정일시"},
                    "status": {"type": "CHAR", "description": "상태"},
                    "category": {"type": "VARCHAR", "description": "카테고리"},
                    "tags": {"type": "TEXT", "description": "태그"}
                },
                "relationships": ["t_post_view_log", "t_post_like_log", "t_post_reply"]
            },
            "t_post_view_log": {
                "description": "게시글 조회 로그",
                "columns": {
                    "no": {"type": "INT", "description": "로그 ID (Primary Key)"},
                    "post_no": {"type": "INT", "description": "게시글 ID (FK)"},
                    "member_no": {"type": "INT", "description": "회원 ID (FK)"},
                    "ins_datetime": {"type": "DATETIME", "description": "조회일시"},
                    "ip_address": {"type": "VARCHAR", "description": "IP 주소"},
                    "user_agent": {"type": "TEXT", "description": "사용자 에이전트"}
                },
                "relationships": ["t_post", "t_member"]
            },
            "t_member_login_log": {
                "description": "회원 로그인 로그",
                "columns": {
                    "no": {"type": "INT", "description": "로그 ID (Primary Key)"},
                    "member_no": {"type": "INT", "description": "회원 ID (FK)"},
                    "ins_datetime": {"type": "DATETIME", "description": "로그인일시"},
                    "ip_address": {"type": "VARCHAR", "description": "IP 주소"},
                    "user_agent": {"type": "TEXT", "description": "사용자 에이전트"},
                    "login_type": {"type": "VARCHAR", "description": "로그인 타입"}
                },
                "relationships": ["t_member"]
            },
            "t_fanding": {
                "description": "펀딩 정보",
                "columns": {
                    "no": {"type": "INT", "description": "펀딩 ID (Primary Key)"},
                    "title": {"type": "VARCHAR", "description": "펀딩 제목"},
                    "description": {"type": "TEXT", "description": "펀딩 설명"},
                    "target_amount": {"type": "DECIMAL", "description": "목표 금액"},
                    "current_amount": {"type": "DECIMAL", "description": "현재 금액"},
                    "status": {"type": "VARCHAR", "description": "펀딩 상태"},
                    "ins_datetime": {"type": "DATETIME", "description": "생성일시"},
                    "end_datetime": {"type": "DATETIME", "description": "종료일시"},
                    "creator_no": {"type": "INT", "description": "크리에이터 ID (FK)"}
                },
                "relationships": ["t_creator", "t_payment"]
            },
            "t_creator": {
                "description": "크리에이터 정보",
                "columns": {
                    "no": {"type": "INT", "description": "크리에이터 ID (Primary Key)"},
                    "name": {"type": "VARCHAR", "description": "크리에이터명"},
                    "email": {"type": "VARCHAR", "description": "이메일"},
                    "phone": {"type": "VARCHAR", "description": "전화번호"},
                    "status": {"type": "CHAR", "description": "상태"},
                    "ins_datetime": {"type": "DATETIME", "description": "등록일시"},
                    "mod_datetime": {"type": "DATETIME", "description": "수정일시"},
                    "bio": {"type": "TEXT", "description": "소개"},
                    "specialty": {"type": "VARCHAR", "description": "전문 분야"}
                },
                "relationships": ["t_fanding"]
            }
        }
    
    def analyze_query_pattern(self, user_query: str, sql_result: Optional[List[Dict]] = None) -> Optional[SchemaPattern]:
        """사용자 쿼리 패턴 분석하여 새로운 스키마 패턴 생성"""
        try:
            # 쿼리 정규화
            normalized_query = self._normalize_query(user_query)
            
            # 기존 패턴과 매칭 시도
            existing_pattern = self._find_existing_pattern(normalized_query)
            if existing_pattern:
                return existing_pattern
            
            # 새로운 패턴 생성
            new_pattern = self._generate_new_pattern(normalized_query, sql_result)
            if new_pattern:
                self._save_pattern(new_pattern)
                return new_pattern
                
        except Exception as e:
            self.logger.error(f"Error analyzing query pattern: {e}")
            
        return None
    
    def _normalize_query(self, query: str) -> str:
        """쿼리 정규화"""
        # 소문자 변환
        normalized = query.lower().strip()
        
        # 특수문자 제거
        normalized = re.sub(r'[^\w\s가-힣]', ' ', normalized)
        
        # 연속 공백 제거
        normalized = re.sub(r'\s+', ' ', normalized)
        
        return normalized
    
    def _find_existing_pattern(self, query: str) -> Optional[SchemaPattern]:
        """기존 패턴과 매칭 시도"""
        for pattern_id, pattern in self.learned_patterns.items():
            for query_pattern in pattern.query_patterns:
                if re.search(query_pattern, query):
                    # 사용 통계 업데이트
                    pattern.usage_count += 1
                    pattern.last_used = datetime.now()
                    return pattern
        return None
    
    def _generate_new_pattern(self, query: str, sql_result: Optional[List[Dict]]) -> Optional[SchemaPattern]:
        """새로운 패턴 생성"""
        try:
            # 테이블 추출
            tables = self._extract_tables_from_query(query)
            if not tables:
                return None
            
            # 컬럼 추출
            columns = self._extract_columns_from_query(query, tables[0])
            if not columns:
                return None
            
            # SQL 템플릿 생성
            sql_template = self._generate_sql_template(tables[0], columns, query)
            if not sql_template:
                return None
            
            # 패턴 ID 생성
            pattern_id = f"dynamic_{hash(query) % 10000}"
            
            # 신뢰도 계산
            confidence = self._calculate_pattern_confidence(query, sql_template, sql_result)
            
            # 패턴 생성
            pattern = SchemaPattern(
                table=tables[0],
                columns=columns,
                query_patterns=[self._create_regex_pattern(query)],
                sql_template=sql_template,
                confidence=confidence,
                usage_count=1,
                last_used=datetime.now()
            )
            
            return pattern
            
        except Exception as e:
            self.logger.error(f"Error generating new pattern: {e}")
            return None
    
    def _extract_tables_from_query(self, query: str) -> List[str]:
        """쿼리에서 테이블 추출"""
        tables = []
        
        # 키워드 기반 테이블 매핑
        table_keywords = {
            '회원': 't_member',
            '맴버': 't_member',
            '결제': 't_payment',
            '게시글': 't_post',
            '포스트': 't_post',
            '펀딩': 't_fanding',
            '크리에이터': 't_creator',
            '조회': 't_post_view_log',
            '로그인': 't_member_login_log'
        }
        
        for keyword, table in table_keywords.items():
            if keyword in query:
                tables.append(table)
        
        return tables
    
    def _extract_columns_from_query(self, query: str, table: str) -> List[str]:
        """쿼리에서 컬럼 추출"""
        columns = []
        
        # 테이블 스키마 정보 가져오기
        table_schema = self.extended_schema.get(table, {})
        table_columns = table_schema.get('columns', {})
        
        # 키워드 기반 컬럼 매핑
        column_keywords = {
            '수': ['no'],
            '명': ['no'],
            '금액': ['price'],
            '조회': ['view_count'],
            '좋아요': ['like_count'],
            '제목': ['title'],
            '내용': ['content'],
            '상태': ['status'],
            '날짜': ['ins_datetime'],
            '시간': ['ins_datetime']
        }
        
        for keyword, possible_columns in column_keywords.items():
            if keyword in query:
                for col in possible_columns:
                    if col in table_columns:
                        columns.append(col)
        
        # 기본 컬럼 추가
        if not columns:
            columns = ['no']  # 기본적으로 ID 컬럼
        
        return columns
    
    def _generate_sql_template(self, table: str, columns: List[str], query: str) -> str:
        """SQL 템플릿 생성"""
        try:
            # 기본 SELECT 쿼리
            if '수' in query or '명' in query:
                # 카운트 쿼리
                sql = f"SELECT COUNT(*) as total_count FROM {table}"
            elif '금액' in query or '매출' in query:
                # 금액 관련 쿼리
                if 'price' in columns:
                    sql = f"SELECT SUM(price) as total_amount, AVG(price) as avg_amount FROM {table}"
                else:
                    sql = f"SELECT COUNT(*) as total_count FROM {table}"
            else:
                # 일반 조회 쿼리
                column_list = ', '.join(columns)
                sql = f"SELECT {column_list} FROM {table}"
            
            # 조건 추가
            if '활성' in query:
                sql += " WHERE status = 'A'"
            elif '비활성' in query:
                sql += " WHERE status = 'I'"
            elif '삭제' in query:
                sql += " WHERE status = 'D'"
            
            # 정렬 추가
            if '최신' in query or '최근' in query:
                sql += " ORDER BY ins_datetime DESC"
            elif '인기' in query or '많은' in query:
                if 'view_count' in columns:
                    sql += " ORDER BY view_count DESC"
                elif 'like_count' in columns:
                    sql += " ORDER BY like_count DESC"
            
            # 제한 추가
            if 'top' in query.lower() or '상위' in query:
                sql += " LIMIT 5"
            elif '최대' in query:
                sql += " LIMIT 10"
            
            return sql
            
        except Exception as e:
            self.logger.error(f"Error generating SQL template: {e}")
            return None
    
    def _create_regex_pattern(self, query: str) -> str:
        """쿼리에서 정규식 패턴 생성"""
        # 특정 단어들을 일반화
        pattern = query.lower()
        
        # 숫자 일반화 (이스케이프 처리)
        pattern = re.sub(r'\d+', r'\\d+', pattern)
        
        # 월 일반화
        pattern = re.sub(r'(1월|2월|3월|4월|5월|6월|7월|8월|9월|10월|11월|12월)', r'(\d+월)', pattern)
        
        # 일반적인 단어들 일반화
        replacements = {
            '얼마나': '.*',
            '몇': '.*',
            '어떤': '.*',
            '어느': '.*',
            '알려줘': '.*',
            '보여줘': '.*',
            '조회': '.*',
            '분석': '.*'
        }
        
        for old, new in replacements.items():
            pattern = pattern.replace(old, new)
        
        return f".*{pattern}.*"
    
    def _calculate_pattern_confidence(self, query: str, sql_template: str, sql_result: Optional[List[Dict]]) -> float:
        """패턴 신뢰도 계산"""
        confidence = 0.5  # 기본 신뢰도
        
        # 쿼리 복잡도에 따른 조정
        if len(query.split()) > 5:
            confidence += 0.1
        
        # SQL 템플릿 품질에 따른 조정
        if 'WHERE' in sql_template:
            confidence += 0.1
        if 'ORDER BY' in sql_template:
            confidence += 0.1
        if 'LIMIT' in sql_template:
            confidence += 0.1
        
        # 결과 데이터에 따른 조정
        if sql_result and len(sql_result) > 0:
            confidence += 0.2
        
        return min(confidence, 0.95)  # 최대 0.95
    
    def _save_pattern(self, pattern: SchemaPattern):
        """패턴 저장"""
        pattern_id = f"dynamic_{hash(pattern.sql_template) % 10000}"
        self.learned_patterns[pattern_id] = pattern
        
        # 쿼리 히스토리에 추가
        self.query_history.append({
            "query": pattern.query_patterns[0],
            "sql_template": pattern.sql_template,
            "confidence": pattern.confidence,
            "timestamp": datetime.now().isoformat()
        })
        
        self.logger.info(f"New pattern saved: {pattern_id}")
    
    def get_extended_schema_info(self, table_name: Optional[str] = None) -> Dict[str, Any]:
        """확장된 스키마 정보 반환"""
        if table_name:
            return self.extended_schema.get(table_name, {})
        return self.extended_schema
    
    def get_learned_patterns(self) -> Dict[str, SchemaPattern]:
        """학습된 패턴들 반환"""
        return self.learned_patterns
    
    def get_query_suggestions(self, partial_query: str) -> List[str]:
        """부분 쿼리에 대한 제안 반환"""
        suggestions = []
        
        # 기존 패턴에서 제안
        for pattern in self.learned_patterns.values():
            for query_pattern in pattern.query_patterns:
                if partial_query.lower() in query_pattern:
                    suggestions.append(query_pattern)
        
        # 스키마 기반 제안
        for table_name, table_info in self.extended_schema.items():
            if any(keyword in partial_query.lower() for keyword in ['회원', '맴버', '결제', '게시글', '펀딩']):
                suggestions.append(f"{table_name} 테이블 정보 조회")
        
        return suggestions[:5]  # 최대 5개 제안
    
    def export_patterns(self, filepath: str):
        """패턴들을 JSON 파일로 내보내기"""
        try:
            export_data = {
                "version": "1.0",
                "exported_at": datetime.now().isoformat(),
                "patterns": []
            }
            
            for pattern_id, pattern in self.learned_patterns.items():
                pattern_data = {
                    "id": pattern_id,
                    "table": pattern.table,
                    "columns": pattern.columns,
                    "query_patterns": pattern.query_patterns,
                    "sql_template": pattern.sql_template,
                    "confidence": pattern.confidence,
                    "usage_count": pattern.usage_count,
                    "last_used": pattern.last_used.isoformat() if pattern.last_used else None
                }
                export_data["patterns"].append(pattern_data)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Patterns exported to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error exporting patterns: {e}")
    
    def import_patterns(self, filepath: str):
        """JSON 파일에서 패턴들 가져오기"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            for pattern_data in import_data.get("patterns", []):
                pattern = SchemaPattern(
                    table=pattern_data["table"],
                    columns=pattern_data["columns"],
                    query_patterns=pattern_data["query_patterns"],
                    sql_template=pattern_data["sql_template"],
                    confidence=pattern_data["confidence"],
                    usage_count=pattern_data.get("usage_count", 0),
                    last_used=datetime.fromisoformat(pattern_data["last_used"]) if pattern_data.get("last_used") else None
                )
                
                self.learned_patterns[pattern_data["id"]] = pattern
            
            self.logger.info(f"Patterns imported from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error importing patterns: {e}")
