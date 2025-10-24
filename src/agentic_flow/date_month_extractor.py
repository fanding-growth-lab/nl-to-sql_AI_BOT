"""
날짜/월 추출 및 매칭 시스템

이 모듈은 자연어 쿼리에서 날짜와 월 정보를 정확하게 추출하고 매칭하는 기능을 제공합니다.
"""

import re
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class DateInfo:
    """추출된 날짜 정보"""
    year: Optional[int] = None
    month: Optional[int] = None
    day: Optional[int] = None
    is_relative: bool = False
    relative_type: Optional[str] = None  # 'yesterday', 'last_week', 'last_month', etc.
    confidence: float = 0.0


class DateMonthExtractor:
    """날짜/월 추출 및 매칭 클래스"""
    
    def __init__(self):
        self.month_mapping = self._create_month_mapping()
        self.relative_patterns = self._create_relative_patterns()
        self.logger = logging.getLogger(__name__)
    
    def _create_month_mapping(self) -> Dict[str, int]:
        """한국어/영어 월 이름 매핑 테이블 생성"""
        return {
            # 한국어 월 표현
            "1월": 1, "일월": 1, "1": 1, "첫째달": 1,
            "2월": 2, "이월": 2, "2": 2, "둘째달": 2,
            "3월": 3, "삼월": 3, "3": 3, "셋째달": 3,
            "4월": 4, "사월": 4, "4": 4, "넷째달": 4,
            "5월": 5, "오월": 5, "5": 5, "다섯째달": 5,
            "6월": 6, "육월": 6, "6": 6, "여섯째달": 6,
            "7월": 7, "칠월": 7, "7": 7, "일곱째달": 7,
            "8월": 8, "팔월": 8, "8": 8, "여덟째달": 8,
            "9월": 9, "구월": 9, "9": 9, "아홉째달": 9,
            "10월": 10, "십월": 10, "10": 10, "열째달": 10,
            "11월": 11, "십일월": 11, "11": 11, "열한째달": 11,
            "12월": 12, "십이월": 12, "12": 12, "열두째달": 12,
            
            # 영어 월 표현 (전체 이름)
            "january": 1, "february": 2, "march": 3, "april": 4,
            "may": 5, "june": 6, "july": 7, "august": 8,
            "september": 9, "october": 10, "november": 11, "december": 12,
            
            # 영어 월 표현 (축약형)
            "jan": 1, "feb": 2, "mar": 3, "apr": 4,
            "jun": 6, "jul": 7, "aug": 8, "sep": 9,
            "oct": 10, "nov": 11, "dec": 12,
            
            # 숫자만 있는 표현 (1-12)
            "01": 1, "02": 2, "03": 3, "04": 4, "05": 5, "06": 6,
            "07": 7, "08": 8, "09": 9, "10": 10, "11": 11, "12": 12
        }
    
    def _create_relative_patterns(self) -> Dict[str, Dict[str, Any]]:
        """상대적 날짜 표현 패턴 생성"""
        return {
            "yesterday": {
                "patterns": [r"어제", r"yesterday"],
                "days_offset": -1
            },
            "today": {
                "patterns": [r"오늘", r"today", r"현재"],
                "days_offset": 0
            },
            "tomorrow": {
                "patterns": [r"내일", r"tomorrow"],
                "days_offset": 1
            },
            "last_week": {
                "patterns": [r"지난주", r"last week", r"지난 주"],
                "days_offset": -7
            },
            "this_week": {
                "patterns": [r"이번주", r"this week", r"이번 주"],
                "days_offset": 0
            },
            "next_week": {
                "patterns": [r"다음주", r"next week", r"다음 주"],
                "days_offset": 7
            },
            "last_month": {
                "patterns": [r"지난달", r"last month", r"지난 달"],
                "months_offset": -1
            },
            "this_month": {
                "patterns": [r"이번달", r"this month", r"이번 달"],
                "months_offset": 0
            },
            "next_month": {
                "patterns": [r"다음달", r"next month", r"다음 달"],
                "months_offset": 1
            },
            "last_year": {
                "patterns": [r"작년", r"last year", r"지난해", r"지난 해"],
                "years_offset": -1
            },
            "this_year": {
                "patterns": [r"올해", r"this year", r"금년"],
                "years_offset": 0
            },
            "next_year": {
                "patterns": [r"내년", r"next year", r"다음해", r"다음 해"],
                "years_offset": 1
            }
        }
    
    def extract_date_info(self, query: str) -> DateInfo:
        """쿼리에서 날짜 정보 추출"""
        query_lower = query.lower().strip()
        
        # 1. 상대적 날짜 표현 확인
        relative_info = self._extract_relative_date(query_lower)
        if relative_info:
            return relative_info
        
        # 2. 절대적 날짜 표현 확인
        absolute_info = self._extract_absolute_date(query_lower)
        if absolute_info:
            return absolute_info
        
        # 3. 월만 있는 표현 확인
        month_info = self._extract_month_only(query_lower)
        if month_info:
            return month_info
        
        # 4. 년도만 있는 표현 확인
        year_info = self._extract_year_only(query_lower)
        if year_info:
            return year_info
        
        return DateInfo(confidence=0.0)
    
    def _extract_relative_date(self, query: str) -> Optional[DateInfo]:
        """상대적 날짜 표현 추출"""
        for relative_type, pattern_info in self.relative_patterns.items():
            for pattern in pattern_info["patterns"]:
                if re.search(pattern, query, re.IGNORECASE):
                    current_date = datetime.now()
                    
                    if "days_offset" in pattern_info:
                        target_date = current_date + timedelta(days=pattern_info["days_offset"])
                    elif "months_offset" in pattern_info:
                        # 월 단위 계산 (월말일 처리)
                        target_date = self._add_months(current_date, pattern_info["months_offset"])
                    elif "years_offset" in pattern_info:
                        target_date = self._add_years(current_date, pattern_info["years_offset"])
                    else:
                        continue
                    
                    return DateInfo(
                        year=target_date.year,
                        month=target_date.month,
                        day=target_date.day,
                        is_relative=True,
                        relative_type=relative_type,
                        confidence=0.9
                    )
        
        return None
    
    def _extract_absolute_date(self, query: str) -> Optional[DateInfo]:
        """절대적 날짜 표현 추출"""
        # YYYY-MM-DD 형식
        date_pattern = r"(\d{4})[-\/](\d{1,2})[-\/](\d{1,2})"
        match = re.search(date_pattern, query)
        if match:
            year, month, day = int(match.group(1)), int(match.group(2)), int(match.group(3))
            if self._is_valid_date(year, month, day):
                return DateInfo(
                    year=year,
                    month=month,
                    day=day,
                    is_relative=False,
                    confidence=0.95
                )
        
        # YYYY-MM 형식
        month_pattern = r"(\d{4})[-\/](\d{1,2})"
        match = re.search(month_pattern, query)
        if match:
            year, month = int(match.group(1)), int(match.group(2))
            if 1 <= month <= 12:
                return DateInfo(
                    year=year,
                    month=month,
                    is_relative=False,
                    confidence=0.9
                )
        
        return None
    
    def _extract_month_only(self, query: str) -> Optional[DateInfo]:
        """월만 있는 표현 추출"""
        # 년도가 함께 있는지 먼저 확인
        year_match = re.search(r"(\d{4})년", query)
        year = int(year_match.group(1)) if year_match else None
        
        # 월 이름 매핑 확인
        for month_name, month_num in self.month_mapping.items():
            if month_name in query:
                current_date = datetime.now()
                target_year = year if year else current_date.year
                return DateInfo(
                    year=target_year,
                    month=month_num,
                    is_relative=False,
                    confidence=0.8
                )
        
        # 숫자 월 패턴 (1-12)
        month_pattern = r"\b([1-9]|1[0-2])\b"
        match = re.search(month_pattern, query)
        if match:
            month = int(match.group(1))
            current_date = datetime.now()
            target_year = year if year else current_date.year
            return DateInfo(
                year=target_year,
                month=month,
                is_relative=False,
                confidence=0.7
            )
        
        return None
    
    def _extract_year_only(self, query: str) -> Optional[DateInfo]:
        """년도만 있는 표현 추출"""
        year_pattern = r"\b(19|20)\d{2}\b"
        match = re.search(year_pattern, query)
        if match:
            year = int(match.group(0))
            current_date = datetime.now()
            return DateInfo(
                year=year,
                month=current_date.month,
                is_relative=False,
                confidence=0.6
            )
        
        return None
    
    def _add_months(self, date: datetime, months: int) -> datetime:
        """월 추가 (월말일 처리)"""
        year = date.year
        month = date.month + months
        
        # 년도 조정
        while month > 12:
            month -= 12
            year += 1
        while month < 1:
            month += 12
            year -= 1
        
        # 월말일 처리
        try:
            return datetime(year, month, date.day)
        except ValueError:
            # 해당 월의 마지막 날로 조정
            if month == 2:
                last_day = 29 if self._is_leap_year(year) else 28
            elif month in [4, 6, 9, 11]:
                last_day = 30
            else:
                last_day = 31
            
            return datetime(year, month, min(date.day, last_day))
    
    def _add_years(self, date: datetime, years: int) -> datetime:
        """년도 추가 (윤년 처리)"""
        year = date.year + years
        
        # 윤년 처리
        if date.month == 2 and date.day == 29:
            if not self._is_leap_year(year):
                return datetime(year, 2, 28)
        
        return datetime(year, date.month, date.day)
    
    def _is_leap_year(self, year: int) -> bool:
        """윤년 판단"""
        return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
    
    def _is_valid_date(self, year: int, month: int, day: int) -> bool:
        """유효한 날짜인지 확인"""
        try:
            datetime(year, month, day)
            return True
        except ValueError:
            return False
    
    def get_sql_date_condition(self, date_info: DateInfo) -> str:
        """날짜 정보를 SQL 조건으로 변환"""
        if not date_info.month:
            return "1=1"
        
        conditions = []
        
        if date_info.year:
            conditions.append(f"EXTRACT(YEAR FROM ins_datetime) = {date_info.year}")
        
        if date_info.month:
            conditions.append(f"EXTRACT(MONTH FROM ins_datetime) = {date_info.month}")
        
        if date_info.day:
            conditions.append(f"EXTRACT(DAY FROM ins_datetime) = {date_info.day}")
        
        return " AND ".join(conditions) if conditions else "1=1"
    
    def get_human_readable_date(self, date_info: DateInfo) -> str:
        """날짜 정보를 사람이 읽기 쉬운 형태로 변환"""
        if date_info.is_relative:
            return f"{date_info.relative_type} ({date_info.year}년 {date_info.month}월)"
        
        parts = []
        if date_info.year:
            parts.append(f"{date_info.year}년")
        if date_info.month:
            parts.append(f"{date_info.month}월")
        if date_info.day:
            parts.append(f"{date_info.day}일")
        
        return " ".join(parts) if parts else "날짜 정보 없음"


# 테스트 함수들
def test_date_extraction():
    """날짜 추출 테스트"""
    extractor = DateMonthExtractor()
    
    test_cases = [
        "8월 신규 회원수",
        "August new members",
        "지난달 활성 사용자",
        "2024년 9월 매출",
        "2024-08-15 데이터",
        "이번달 성과",
        "작년 12월 통계"
    ]
    
    print("=== 날짜 추출 테스트 ===")
    for query in test_cases:
        date_info = extractor.extract_date_info(query)
        print(f"쿼리: {query}")
        print(f"  → {date_info}")
        print(f"  → SQL 조건: {extractor.get_sql_date_condition(date_info)}")
        print(f"  → 사람이 읽기 쉬운 형태: {extractor.get_human_readable_date(date_info)}")
        print()


if __name__ == "__main__":
    test_date_extraction()

