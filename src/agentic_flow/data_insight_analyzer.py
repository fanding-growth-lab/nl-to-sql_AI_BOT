#!/usr/bin/env python3
"""
데이터 인사이트 분석기
SQL 실행 결과를 분석하여 비즈니스 인사이트와 액션 추천을 제공합니다.
"""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import re

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.output_parsers import JsonOutputParser
    from langchain_core.prompts import ChatPromptTemplate
except ImportError:
    ChatGoogleGenerativeAI = None
    JsonOutputParser = None
    ChatPromptTemplate = None

from .prompt_validator import PromptValidator, PromptDebugger, PromptExecutor

logger = logging.getLogger(__name__)

@dataclass
class BusinessInsight:
    """비즈니스 인사이트 데이터 클래스"""
    insight_type: str  # 'trend', 'anomaly', 'opportunity', 'risk', 'performance'
    title: str
    description: str
    confidence: float  # 0.0 ~ 1.0
    impact_level: str  # 'high', 'medium', 'low'
    data_points: List[Dict[str, Any]]
    recommendations: List[str]
    next_actions: List[str]

@dataclass
class InsightReport:
    """인사이트 리포트 데이터 클래스"""
    query: str
    sql_result: List[Dict[str, Any]]
    insights: List[BusinessInsight]
    summary: str
    key_metrics: Dict[str, Any]
    generated_at: datetime

class DataInsightAnalyzer:
    """데이터 인사이트 분석기"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.llm = self._initialize_llm()
        
        # 프롬프트 검증 시스템 초기화
        self.prompt_validator = PromptValidator()
        self.prompt_debugger = PromptDebugger()
        self.prompt_executor = PromptExecutor(self.llm, self.prompt_debugger)
        
    def _initialize_llm(self) -> Optional[ChatGoogleGenerativeAI]:
        """LLM 초기화"""
        try:
            if ChatGoogleGenerativeAI is None:
                self.logger.warning("LangChain Google GenerativeAI not available")
                return None
                
            llm_config = self.config.get('llm', {})
            api_key = llm_config.get('api_key', '')
            model = llm_config.get('model', 'gemini-2.5-pro')
            
            if not api_key:
                self.logger.warning("Google API key not found")
                return None
                
            return ChatGoogleGenerativeAI(
                model=model,
                google_api_key=api_key,
                temperature=0.3,
                max_output_tokens=2048,
                timeout=30,
                convert_system_message_to_human=True  # 최신 버전 호환성
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM: {e}")
            return None
    
    def analyze_data(self, query: str, sql_result: List[Dict[str, Any]], 
                    sql_query: str = "") -> InsightReport:
        """
        SQL 결과를 분석하여 비즈니스 인사이트를 생성합니다.
        
        Args:
            query: 사용자 쿼리
            sql_result: SQL 실행 결과
            sql_query: 실행된 SQL 쿼리
            
        Returns:
            InsightReport: 분석 결과 리포트
        """
        try:
            self.logger.info(f"Analyzing data for query: {query[:50]}...")
            
            # 1. 기본 메트릭 추출
            key_metrics = self._extract_key_metrics(sql_result)
            
            # 2. LLM 기반 인사이트 생성
            insights = []
            if self.llm and sql_result:
                insights = self._generate_insights_with_llm(query, sql_result, sql_query)
            
            # 3. 규칙 기반 인사이트 생성 (LLM 실패 시 백업)
            if not insights:
                insights = self._generate_rule_based_insights(query, sql_result, key_metrics)
            
            # 4. 요약 생성
            summary = self._generate_summary(query, insights, key_metrics)
            
            return InsightReport(
                query=query,
                sql_result=sql_result,
                insights=insights,
                summary=summary,
                key_metrics=key_metrics,
                generated_at=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing data: {e}")
            return self._create_error_report(query, sql_result, str(e))
    
    def _extract_key_metrics(self, sql_result: List[Dict[str, Any]]) -> Dict[str, Any]:
        """핵심 메트릭 추출"""
        if not sql_result or not isinstance(sql_result, list):
            return {}
        
        metrics = {}
        
        # 숫자형 컬럼들의 통계 계산
        numeric_columns = {}
        for row in sql_result:
            if isinstance(row, dict):
                for key, value in row.items():
                    if isinstance(value, (int, float)) and value is not None:
                        if key not in numeric_columns:
                            numeric_columns[key] = []
                        numeric_columns[key].append(value)
        
        # 각 숫자형 컬럼의 통계 계산
        for column, values in numeric_columns.items():
            if values:
                metrics[column] = {
                    'sum': sum(values),
                    'avg': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'count': len(values)
                }
        
        # 전체 결과 수
        metrics['total_records'] = len(sql_result)
        
        return metrics
    
    def _generate_insights_with_llm(self, query: str, sql_result: List[Dict[str, Any]], 
                                  sql_query: str) -> List[BusinessInsight]:
        """LLM을 사용한 인사이트 생성"""
        try:
            # 데이터를 JSON 문자열로 변환 (처음 10개 행만)
            sample_data = sql_result[:10] if len(sql_result) > 10 else sql_result
            data_json = json.dumps(sample_data, default=str, ensure_ascii=False)
            
            # 프롬프트 생성
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", """당신은 데이터 분석 전문가입니다. 
                SQL 쿼리 결과를 분석하여 비즈니스 인사이트를 제공해야 합니다.
                
                다음 형식으로 JSON 응답을 제공하세요:
                {
                    "insights": [
                        {
                            "insight_type": "trend|anomaly|opportunity|risk|performance",
                            "title": "인사이트 제목",
                            "description": "상세 설명",
                            "confidence": 0.0-1.0,
                            "impact_level": "high|medium|low",
                            "data_points": [{"metric": "값", "value": 숫자}],
                            "recommendations": ["추천사항1", "추천사항2"],
                            "next_actions": ["다음 액션1", "다음 액션2"]
                        }
                    ]
                }
                
                인사이트는 다음 기준으로 생성하세요:
                - 데이터의 패턴과 트렌드 분석
                - 비정상적인 값이나 이상치 탐지
                - 비즈니스 기회나 위험 요소 식별
                - 성과 지표의 의미 해석
                """),
                ("human", f"""
                사용자 쿼리: {query}
                SQL 쿼리: {sql_query}
                데이터 결과: {data_json}
                
                위 데이터를 분석하여 비즈니스 인사이트를 생성해주세요.
                """)
            ])
            
            # 프롬프트 검증 및 실행
            variables = {
                "query": query,
                "sql_query": sql_query,
                "data_json": data_json
            }
            
            # 프롬프트 검증
            validation_result = self.prompt_validator.validate_prompt_variables(
                f"사용자 쿼리: {query}\nSQL 쿼리: {sql_query}\n데이터 결과: {data_json}", 
                variables
            )
            
            if not validation_result.is_valid:
                self.logger.warning(f"프롬프트 변수 오류: {validation_result.missing_vars}")
                # 자동 보정 시도
                for missing_var in validation_result.missing_vars:
                    if missing_var == "query":
                        variables[missing_var] = "사용자 쿼리"
                    elif missing_var == "sql_query":
                        variables[missing_var] = "SELECT * FROM table"
                    elif missing_var == "data_json":
                        variables[missing_var] = "[]"
                    else:
                        variables[missing_var] = f"<{missing_var}_placeholder>"
            
            # LLM 호출
            chain = prompt_template | self.llm | JsonOutputParser()
            result = chain.invoke(variables)
            
            # 결과 파싱
            insights = []
            for insight_data in result.get('insights', []):
                insight = BusinessInsight(
                    insight_type=insight_data.get('insight_type', 'performance'),
                    title=insight_data.get('title', ''),
                    description=insight_data.get('description', ''),
                    confidence=float(insight_data.get('confidence', 0.5)),
                    impact_level=insight_data.get('impact_level', 'medium'),
                    data_points=insight_data.get('data_points', []),
                    recommendations=insight_data.get('recommendations', []),
                    next_actions=insight_data.get('next_actions', [])
                )
                insights.append(insight)
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Error generating LLM insights: {e}")
            return []
    
    def _generate_rule_based_insights(self, query: str, sql_result: List[Dict[str, Any]], 
                                    key_metrics: Dict[str, Any]) -> List[BusinessInsight]:
        """규칙 기반 인사이트 생성 (LLM 실패 시 백업)"""
        insights = []
        
        if not sql_result or not isinstance(sql_result, list):
            return insights
        
        # 1. 데이터 크기 기반 인사이트
        total_records = len(sql_result)
        if total_records > 1000:
            insights.append(BusinessInsight(
                insight_type='performance',
                title='대용량 데이터셋',
                description=f'총 {total_records:,}건의 데이터가 분석되었습니다.',
                confidence=1.0,
                impact_level='medium',
                data_points=[{'metric': '총 레코드 수', 'value': total_records}],
                recommendations=['데이터 샘플링을 고려해보세요', '인덱스 최적화를 검토하세요'],
                next_actions=['성능 모니터링', '쿼리 최적화 검토']
            ))
        
        # 2. 숫자형 메트릭 기반 인사이트
        for metric_name, metric_data in key_metrics.items():
            if isinstance(metric_data, dict) and 'sum' in metric_data:
                sum_value = metric_data['sum']
                avg_value = metric_data['avg']
                
                # 높은 합계값 인사이트
                if sum_value > 1000000:
                    insights.append(BusinessInsight(
                        insight_type='performance',
                        title=f'{metric_name} 높은 수치',
                        description=f'{metric_name}의 총합이 {sum_value:,}로 매우 높습니다.',
                        confidence=0.8,
                        impact_level='high',
                        data_points=[{'metric': metric_name, 'value': sum_value}],
                        recommendations=['이 수치의 의미를 분석하세요', '트렌드 변화를 모니터링하세요'],
                        next_actions=['상세 분석 수행', '비교 분석 실행']
                    ))
        
        return insights
    
    def _generate_summary(self, query: str, insights: List[BusinessInsight], 
                         key_metrics: Dict[str, Any]) -> str:
        """요약 생성"""
        if not insights:
            total_records = key_metrics.get('total_records', 0)
            return f"'{query}' 쿼리에 대한 데이터 분석이 완료되었습니다. 총 {total_records}건의 데이터가 처리되었습니다."
        
        high_impact_insights = [i for i in insights if i.impact_level == 'high']
        medium_impact_insights = [i for i in insights if i.impact_level == 'medium']
        
        summary_parts = [
            f"'{query}' 쿼리 분석 결과:",
            f"- 총 {len(insights)}개의 인사이트 발견",
            f"- 고위험/고영향: {len(high_impact_insights)}개",
            f"- 중간 영향: {len(medium_impact_insights)}개"
        ]
        
        if high_impact_insights:
            summary_parts.append(f"- 주요 발견: {high_impact_insights[0].title}")
        
        return " ".join(summary_parts)
    
    def _create_error_report(self, query: str, sql_result: List[Dict[str, Any]], 
                           error_message: str) -> InsightReport:
        """오류 리포트 생성"""
        # sql_result가 리스트인지 확인
        total_records = len(sql_result) if isinstance(sql_result, list) else 0
        
        return InsightReport(
            query=query,
            sql_result=sql_result if isinstance(sql_result, list) else [],
            insights=[],
            summary=f"데이터 분석 중 오류가 발생했습니다: {error_message}",
            key_metrics={'total_records': total_records},
            generated_at=datetime.now()
        )
    
    def format_insight_report(self, report: InsightReport) -> str:
        """인사이트 리포트를 사용자 친화적 형식으로 포맷팅"""
        output = []
        
        # 헤더
        output.append("=" * 50)
        output.append("데이터 분석 리포트")
        output.append("=" * 50)
        output.append(f"쿼리: {report.query}")
        output.append(f"생성 시간: {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}")
        output.append("")
        
        # 요약
        output.append("요약")
        output.append("-" * 20)
        output.append(report.summary)
        output.append("")
        
        # 핵심 메트릭
        if report.key_metrics:
            output.append("핵심 메트릭")
            output.append("-" * 20)
            for metric, value in report.key_metrics.items():
                if isinstance(value, dict):
                    output.append(f"{metric}:")
                    for sub_metric, sub_value in value.items():
                        if isinstance(sub_value, (int, float)):
                            output.append(f"  - {sub_metric}: {sub_value:,.2f}")
                        else:
                            output.append(f"  - {sub_metric}: {sub_value}")
                else:
                    output.append(f"{metric}: {value}")
            output.append("")
        
        # 인사이트
        if report.insights:
            output.append("비즈니스 인사이트")
            output.append("-" * 20)
            
            for i, insight in enumerate(report.insights, 1):
                output.append(f"{i}. {insight.title}")
                output.append(f"   유형: {insight.insight_type}")
                output.append(f"   영향도: {insight.impact_level}")
                output.append(f"   신뢰도: {insight.confidence:.1%}")
                output.append(f"   설명: {insight.description}")
                
                if insight.recommendations:
                    output.append("   추천사항:")
                    for rec in insight.recommendations:
                        output.append(f"     • {rec}")
                
                if insight.next_actions:
                    output.append("   다음 액션:")
                    for action in insight.next_actions:
                        output.append(f"     → {action}")
                
                output.append("")
        else:
            output.append("비즈니스 인사이트")
            output.append("-" * 20)
            output.append("분석 가능한 인사이트가 없습니다.")
            output.append("")
        
        return "\n".join(output)
