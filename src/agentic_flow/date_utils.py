#!/usr/bin/env python3
"""
Date Utilities for Fanding Data Report System

This module provides date parsing and extraction utilities for natural language queries.
"""

import re
from datetime import datetime, timedelta
from typing import Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


class DateUtils:
    """날짜 관련 유틸리티 클래스"""
    
    @staticmethod
    def extract_month_from_query(query: str) -> Optional[str]:
        """
        쿼리에서 월 정보 추출 (기존 호환성 유지)
        
        Args:
            query: 사용자 쿼리
            
        Returns:
            월 정보 문자열 (예: "09") 또는 None
        """
        query_lower = query.lower()
        
        # 직접적인 월 패턴 (1월, 2월, ...)
        month_patterns = {
            r'(\d+)월': r'\1',
            r'(\d+)월달': r'\1',
            r'(\d+)월분': r'\1'
        }
        
        for pattern, replacement in month_patterns.items():
            match = re.search(pattern, query_lower)
            if match:
                month = match.group(1)
                if 1 <= int(month) <= 12:
                    return month.zfill(2)  # 01, 02, etc.
        
        # 상대적 시간 표현
        relative_month_mapping = {
            '지난달': '12',  # 기본값 (실제로는 현재 월 - 1)
            '이번달': '01',  # 기본값 (실제로는 현재 월)
            '다음달': '02',  # 기본값 (실제로는 현재 월 + 1)
            '최근': '01',    # 기본값 (실제로는 현재 월)
        }
        
        for pattern, month in relative_month_mapping.items():
            if pattern in query_lower:
                return month
        
        return None
    
    @staticmethod
    def extract_month_with_year_from_query(query: str) -> Optional[Tuple[int, int]]:
        """
        쿼리에서 연도와 월 정보를 함께 추출 (개선된 버전)
        
        Args:
            query: 사용자 쿼리
            
        Returns:
            (연도, 월) 튜플 또는 None
        """
        query_lower = query.lower()
        current_year = datetime.now().year
        current_month = datetime.now().month
        
        # 직접적인 월 패턴 (1월, 2월, ...)
        month_patterns = [
            r'(\d+)월',
            r'(\d+)월달',
            r'(\d+)월분'
        ]
        
        for pattern in month_patterns:
            match = re.search(pattern, query_lower)
            if match:
                month = int(match.group(1))
                if 1 <= month <= 12:
                    # 연도 정보가 있는지 확인
                    year = DateUtils._extract_year_from_query(query_lower)
                    if year:
                        return (year, month)
                    else:
                        # 연도 정보가 없으면 현재 연도 사용
                        return (current_year, month)
        
        # 상대적 시간 표현 처리
        relative_result = DateUtils._handle_relative_time_expressions(query_lower, current_year, current_month)
        if relative_result:
            return relative_result
        
        return None
    
    @staticmethod
    def extract_date_from_query(query: str) -> Optional[Tuple[int, int, Optional[int]]]:
        """
        쿼리에서 연도, 월, 일 정보를 함께 추출
        
        Args:
            query: 사용자 쿼리
            
        Returns:
            (연도, 월, 일) 튜플 (일이 없으면 None) 또는 None
        """
        query_lower = query.lower()
        current_year = datetime.now().year
        current_month = datetime.now().month
        
        # "11월 1일" 패턴: (\d+)월\s*(\d+)일
        full_date_pattern = r'(\d+)월\s*(\d+)일'
        full_date_match = re.search(full_date_pattern, query_lower)
        if full_date_match:
            month = int(full_date_match.group(1))
            day = int(full_date_match.group(2))
            if 1 <= month <= 12 and 1 <= day <= 31:
                # 연도 정보가 있는지 확인
                year = DateUtils._extract_year_from_query(query_lower)
                if not year:
                    year = current_year
                return (year, month, day)
        
        # "11월" 패턴: 월만 있는 경우
        month_info = DateUtils.extract_month_with_year_from_query(query)
        if month_info:
            year, month = month_info
            return (year, month, None)
        
        return None
    
    @staticmethod
    def _extract_year_from_query(query_lower: str) -> Optional[int]:
        """
        쿼리에서 연도 정보 추출
        
        Args:
            query_lower: 소문자로 변환된 쿼리
            
        Returns:
            연도 또는 None
        """
        # 직접적인 연도 패턴 (2024년, 2024, 작년, 올해 등)
        year_patterns = [
            r'(\d{4})년',
            r'(\d{4})',
            r'올해',
            r'작년',
            r'지난해',
            r'내년',
            r'다음해'
        ]
        
        current_year = datetime.now().year
        
        for pattern in year_patterns:
            if pattern in ['올해']:
                return current_year
            elif pattern in ['작년', '지난해']:
                return current_year - 1
            elif pattern in ['내년', '다음해']:
                return current_year + 1
            else:
                match = re.search(pattern, query_lower)
                if match:
                    year = int(match.group(1))
                    if 2000 <= year <= 2100:  # 합리적인 연도 범위
                        return year
        
        return None
    
    @staticmethod
    def _handle_relative_time_expressions(query_lower: str, current_year: int, current_month: int) -> Optional[Tuple[int, int]]:
        """
        상대적 시간 표현 처리
        
        Args:
            query_lower: 소문자로 변환된 쿼리
            current_year: 현재 연도
            current_month: 현재 월
            
        Returns:
            (연도, 월) 튜플 또는 None
        """
        relative_expressions = {
            '지난달': (current_year if current_month > 1 else current_year - 1, 
                      current_month - 1 if current_month > 1 else 12),
            '이번달': (current_year, current_month),
            '다음달': (current_year if current_month < 12 else current_year + 1,
                      current_month + 1 if current_month < 12 else 1),
            '최근': (current_year, current_month),
            '이번 달': (current_year, current_month),
            '이번달': (current_year, current_month),
            '이번 월': (current_year, current_month)
        }
        
        for expression, (year, month) in relative_expressions.items():
            if expression in query_lower:
                return (year, month)
        
        return None
    
    @staticmethod
    def get_analysis_month(query: str) -> Optional[str]:
        """
        분석용 월 정보 반환 (기존 호환성 유지)
        
        Args:
            query: 사용자 쿼리
            
        Returns:
            YYYY-MM 형식의 월 정보 또는 None
        """
        month_info = DateUtils.extract_month_with_year_from_query(query)
        if month_info:
            year, month = month_info
            return f"{year}-{month:02d}"
        return None
    
    @staticmethod
    def extract_year_from_query(query: str) -> Optional[int]:
        """
        쿼리에서 연도 정보 추출 (기존 호환성 유지)
        
        Args:
            query: 사용자 쿼리
            
        Returns:
            연도 또는 None
        """
        return DateUtils._extract_year_from_query(query.lower())
    
    @staticmethod
    def format_date_for_sql(year: int, month: int) -> str:
        """
        SQL 쿼리용 날짜 형식 생성
        
        Args:
            year: 연도
            month: 월
            
        Returns:
            YYYY-MM 형식의 날짜 문자열
        """
        return f"{year}-{month:02d}"
    
    @staticmethod
    def get_date_range_for_period(period: str) -> Optional[Tuple[datetime, datetime]]:
        """
        기간 표현에서 날짜 범위 추출
        
        Args:
            period: 기간 표현 (예: "최근 7일", "지난 30일")
            
        Returns:
            (시작일, 종료일) 튜플 또는 None
        """
        period_lower = period.lower()
        now = datetime.now()
        
        # 일 단위 패턴
        day_patterns = [
            (r'최근\s*(\d+)\s*일', lambda days: (now - timedelta(days=int(days)), now)),
            (r'지난\s*(\d+)\s*일', lambda days: (now - timedelta(days=int(days)), now)),
            (r'(\d+)\s*일간', lambda days: (now - timedelta(days=int(days)), now)),
            (r'(\d+)\s*일\s*동안', lambda days: (now - timedelta(days=int(days)), now))
        ]
        
        for pattern, date_func in day_patterns:
            match = re.search(pattern, period_lower)
            if match:
                days = int(match.group(1))
                return date_func(days)
        
        # 월 단위 패턴
        month_patterns = [
            (r'최근\s*(\d+)\s*개?월', lambda months: (now - timedelta(days=30*int(months)), now)),
            (r'지난\s*(\d+)\s*개?월', lambda months: (now - timedelta(days=30*int(months)), now)),
            (r'(\d+)\s*개?월간', lambda months: (now - timedelta(days=30*int(months)), now)),
            (r'(\d+)\s*개?월\s*동안', lambda months: (now - timedelta(days=30*int(months)), now))
        ]
        
        for pattern, date_func in month_patterns:
            match = re.search(pattern, period_lower)
            if match:
                months = int(match.group(1))
                return date_func(months)
        
        return None
