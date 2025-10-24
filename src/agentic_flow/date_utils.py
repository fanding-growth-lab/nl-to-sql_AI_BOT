#!/usr/bin/env python3
"""
Date Utilities
날짜 관련 유틸리티 함수들
"""

from datetime import datetime, date
from typing import Optional, Dict, Any
import re


class DateUtils:
    """날짜 관련 유틸리티 클래스"""
    
    @staticmethod
    def get_current_year() -> int:
        """현재 연도 반환"""
        return datetime.now().year
    
    @staticmethod
    def get_current_month() -> int:
        """현재 월 반환"""
        return datetime.now().month
    
    @staticmethod
    def get_current_date() -> str:
        """현재 날짜를 YYYY-MM-DD 형식으로 반환"""
        return datetime.now().strftime('%Y-%m-%d')
    
    @staticmethod
    def extract_month_from_query(query: str, default_year: Optional[int] = None) -> Optional[str]:
        """쿼리에서 월 추출 (동적 연도 처리)"""
        if default_year is None:
            default_year = DateUtils.get_current_year()
        
        # 월별 패턴 매칭
        month_patterns = {
            '1월': f'{default_year}-01', '2월': f'{default_year}-02', '3월': f'{default_year}-03', '4월': f'{default_year}-04',
            '5월': f'{default_year}-05', '6월': f'{default_year}-06', '7월': f'{default_year}-07', '8월': f'{default_year}-08',
            '9월': f'{default_year}-09', '10월': f'{default_year}-10', '11월': f'{default_year}-11', '12월': f'{default_year}-12'
        }
        
        query_lower = query.lower()
        for month_kr, month_code in month_patterns.items():
            if month_kr in query_lower:
                return month_code
        
        return None
    
    @staticmethod
    def extract_year_from_query(query: str) -> Optional[int]:
        """쿼리에서 연도 추출"""
        # 연도 패턴 찾기 (4자리 숫자, "년" 포함 또는 단독)
        year_patterns = [
            r'\b(20\d{2})년\b',  # "2024년" 형태
            r'\b(20\d{2})\b'      # "2024" 형태
        ]
        
        for pattern in year_patterns:
            matches = re.findall(pattern, query)
            if matches:
                return int(matches[0])
        
        return None
    
    @staticmethod
    def get_analysis_month(query: str) -> str:
        """쿼리에서 분석할 월 추출 (연도 포함)"""
        # 1. 쿼리에서 연도 추출 시도
        extracted_year = DateUtils.extract_year_from_query(query)
        
        # 2. 연도가 있으면 해당 연도, 없으면 현재 연도 사용
        target_year = extracted_year if extracted_year else DateUtils.get_current_year()
        
        # 3. 월 추출
        month = DateUtils.extract_month_from_query(query, target_year)
        
        if month:
            return month
        
        # 4. 월이 없으면 현재 월 반환
        current_year = DateUtils.get_current_year()
        current_month = DateUtils.get_current_month()
        return f'{current_year}-{current_month:02d}'
    
    @staticmethod
    def is_current_year(query: str) -> bool:
        """쿼리가 현재 연도를 언급하는지 확인"""
        current_year = DateUtils.get_current_year()
        extracted_year = DateUtils.extract_year_from_query(query)
        
        if extracted_year:
            return extracted_year == current_year
        
        # 연도가 명시되지 않으면 현재 연도로 간주
        return True
    
    @staticmethod
    def get_relative_month(months_ago: int) -> str:
        """몇 개월 전의 월 반환"""
        from dateutil.relativedelta import relativedelta
        
        target_date = datetime.now() - relativedelta(months=months_ago)
        return target_date.strftime('%Y-%m')
    
    @staticmethod
    def parse_relative_time(query: str) -> Optional[str]:
        """상대적 시간 표현 파싱"""
        query_lower = query.lower()
        
        # "이번 달", "이번달"
        if any(phrase in query_lower for phrase in ['이번 달', '이번달', '이번 월', '이번월']):
            return DateUtils.get_current_date()[:7]  # YYYY-MM
        
        # "지난 달", "지난달"
        if any(phrase in query_lower for phrase in ['지난 달', '지난달', '지난 월', '지난월']):
            return DateUtils.get_relative_month(1)
        
        # "2개월 전", "3개월 전" 등
        for i in range(1, 13):
            if f'{i}개월 전' in query_lower or f'{i}달 전' in query_lower:
                return DateUtils.get_relative_month(i)
        
        return None
