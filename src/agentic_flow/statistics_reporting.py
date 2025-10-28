#!/usr/bin/env python3
"""
Intent Classification Statistics Reporting and Visualization System

This module implements comprehensive reporting and visualization capabilities
for intent classification statistics, including real-time dashboards, trend analysis,
performance metrics, and data export functionality.
"""

import json
import time
import logging
import statistics
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import math

from .intent_classification_stats import IntentClassifierStats, ClassificationMetrics
from .statistics_persistence import StatisticsPersistenceManager


@dataclass
class ReportConfig:
    """보고서 생성 설정"""
    include_trends: bool = True
    include_performance_metrics: bool = True
    include_error_analysis: bool = True
    include_confidence_distribution: bool = True
    include_response_time_analysis: bool = True
    time_range_hours: int = 24
    max_data_points: int = 1000


@dataclass
class TrendAnalysis:
    """추세 분석 결과"""
    period: str
    total_classifications: int
    average_confidence: float
    error_rate: float
    average_response_time_ms: float
    intent_distribution: Dict[str, int]
    confidence_trend: List[float]
    response_time_trend: List[float]
    error_trend: List[float]


@dataclass
class PerformanceInsights:
    """성능 인사이트"""
    overall_performance_score: float
    confidence_stability: float
    response_time_efficiency: float
    error_rate_stability: float
    recommendations: List[str]
    alerts: List[str]


@dataclass
class VisualizationData:
    """시각화를 위한 데이터 구조"""
    chart_data: Dict[str, Any]
    metrics_summary: Dict[str, Any]
    trends: List[TrendAnalysis]
    insights: PerformanceInsights


class StatisticsReporter:
    """통계 보고 및 분석 시스템"""
    
    def __init__(self, persistence_manager: Optional[StatisticsPersistenceManager] = None):
        self.logger = logging.getLogger(__name__)
        self.persistence_manager = persistence_manager or StatisticsPersistenceManager()
        
        # 캐시된 데이터
        self._cached_stats: Optional[IntentClassifierStats] = None
        self._cache_timestamp: float = 0
        self._cache_ttl: float = 60.0  # 60초 캐시
        
    def generate_comprehensive_report(self, config: Optional[ReportConfig] = None) -> Dict[str, Any]:
        """종합 보고서 생성"""
        if config is None:
            config = ReportConfig()
            
        self.logger.info("Generating comprehensive statistics report")
        
        try:
            # 최신 통계 데이터 로드
            stats = self._get_latest_stats()
            
            # 기본 메트릭 계산
            basic_metrics = self._calculate_basic_metrics(stats)
            
            # 추세 분석
            trends = []
            if config.include_trends:
                trends = self._analyze_trends(stats, config.time_range_hours)
            
            # 성능 인사이트
            insights = None
            if config.include_performance_metrics:
                insights = self._generate_performance_insights(stats, trends)
            
            # 오류 분석
            error_analysis = None
            if config.include_error_analysis:
                error_analysis = self._analyze_errors(stats)
            
            # 신뢰도 분포
            confidence_distribution = None
            if config.include_confidence_distribution:
                confidence_distribution = self._analyze_confidence_distribution(stats)
            
            # 응답 시간 분석
            response_time_analysis = None
            if config.include_response_time_analysis:
                response_time_analysis = self._analyze_response_times(stats)
            
            report = {
                "report_metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "time_range_hours": config.time_range_hours,
                    "total_classifications": stats.total_classifications,
                    "report_version": "1.0"
                },
                "basic_metrics": basic_metrics,
                "trends": [asdict(trend) for trend in trends] if trends else [],
                "performance_insights": asdict(insights) if insights else None,
                "error_analysis": error_analysis,
                "confidence_distribution": confidence_distribution,
                "response_time_analysis": response_time_analysis
            }
            
            self.logger.info(f"Generated comprehensive report with {len(trends)} trend periods")
            return report
            
        except Exception as e:
            self.logger.error(f"Failed to generate comprehensive report: {e}")
            return self._generate_error_report(str(e))
    
    def generate_real_time_dashboard_data(self) -> Dict[str, Any]:
        """실시간 대시보드 데이터 생성"""
        try:
            stats = self._get_latest_stats()
            
            # 실시간 메트릭 계산
            current_time = time.time()
            recent_window = 300  # 최근 5분
            
            recent_classifications = [
                m for m in stats.classification_history 
                if current_time - m.timestamp <= recent_window
            ]
            
            dashboard_data = {
                "timestamp": current_time,
                "real_time_metrics": {
                    "classifications_per_minute": len(recent_classifications) / (recent_window / 60),
                    "current_error_rate": self._calculate_error_rate(recent_classifications),
                    "average_confidence": self._calculate_average_confidence(recent_classifications),
                    "average_response_time_ms": self._calculate_average_response_time(recent_classifications)
                },
                "system_status": {
                    "total_classifications": stats.total_classifications,
                    "uptime_hours": (current_time - stats.start_time) / 3600,
                    "last_classification": max([m.timestamp for m in stats.classification_history]) if stats.classification_history else None
                },
                "intent_distribution": dict(Counter([m.intent for m in recent_classifications])),
                "alerts": self._check_alerts(stats, recent_classifications)
            }
            
            return dashboard_data
            
        except Exception as e:
            self.logger.error(f"Failed to generate real-time dashboard data: {e}")
            return {"error": str(e), "timestamp": time.time()}
    
    def export_statistics_data(self, format: str = "json", time_range_hours: int = 24) -> Dict[str, Any]:
        """통계 데이터 내보내기"""
        try:
            stats = self._get_latest_stats()
            
            # 시간 범위 필터링
            cutoff_time = time.time() - (time_range_hours * 3600)
            filtered_history = [
                m for m in stats.classification_history 
                if m.timestamp >= cutoff_time
            ]
            
            export_data = {
                "export_metadata": {
                    "exported_at": datetime.now().isoformat(),
                    "format": format,
                    "time_range_hours": time_range_hours,
                    "total_records": len(filtered_history)
                },
                "classification_history": [asdict(m) for m in filtered_history],
                "summary_statistics": {
                    "total_classifications": len(filtered_history),
                    "average_confidence": self._calculate_average_confidence(filtered_history),
                    "error_rate": self._calculate_error_rate(filtered_history),
                    "average_response_time_ms": self._calculate_average_response_time(filtered_history),
                    "intent_distribution": dict(Counter([m.intent for m in filtered_history]))
                }
            }
            
            if format == "csv":
                # CSV 형태로 변환
                csv_data = self._convert_to_csv(filtered_history)
                export_data["csv_data"] = csv_data
            
            self.logger.info(f"Exported {len(filtered_history)} records in {format} format")
            return export_data
            
        except Exception as e:
            self.logger.error(f"Failed to export statistics data: {e}")
            return {"error": str(e)}
    
    def _get_latest_stats(self) -> IntentClassifierStats:
        """최신 통계 데이터 가져오기 (캐시 활용)"""
        current_time = time.time()
        
        if (self._cached_stats is None or 
            current_time - self._cache_timestamp > self._cache_ttl):
            
            # 캐시에서 로드 시도
            try:
                self._cached_stats = self.persistence_manager.load_statistics()
                self._cache_timestamp = current_time
                self.logger.debug("Loaded statistics from persistence cache")
            except Exception as e:
                self.logger.warning(f"Failed to load from cache: {e}")
                # 기본 통계 생성
                self._cached_stats = IntentClassifierStats()
        
        return self._cached_stats
    
    def _calculate_basic_metrics(self, stats: IntentClassifierStats) -> Dict[str, Any]:
        """기본 메트릭 계산"""
        if not stats.classification_history:
            return {
                "total_classifications": 0,
                "average_confidence": 0.0,
                "error_rate": 0.0,
                "average_response_time_ms": 0.0,
                "intent_distribution": {},
                "confidence_stats": {
                    "min": 0.0,
                    "max": 0.0,
                    "median": 0.0,
                    "std_dev": 0.0
                }
            }
        
        confidences = [m.confidence for m in stats.classification_history]
        response_times = [m.response_time_ms for m in stats.classification_history]
        
        return {
            "total_classifications": stats.total_classifications,
            "average_confidence": statistics.mean(confidences),
            "error_rate": self._calculate_error_rate(stats.classification_history),
            "average_response_time_ms": statistics.mean(response_times),
            "intent_distribution": dict(Counter([m.intent for m in stats.classification_history])),
            "confidence_stats": {
                "min": min(confidences),
                "max": max(confidences),
                "median": statistics.median(confidences),
                "std_dev": statistics.stdev(confidences) if len(confidences) > 1 else 0.0
            }
        }
    
    def _analyze_trends(self, stats: IntentClassifierStats, time_range_hours: int) -> List[TrendAnalysis]:
        """추세 분석"""
        trends = []
        
        if not stats.classification_history:
            return trends
        
        # 시간 윈도우 설정
        window_size_hours = max(1, time_range_hours // 10)  # 최대 10개 구간
        current_time = time.time()
        start_time = current_time - (time_range_hours * 3600)
        
        # 시간 구간별로 분석
        for i in range(0, time_range_hours, window_size_hours):
            period_start = start_time + (i * 3600)
            period_end = period_start + (window_size_hours * 3600)
            
            period_data = [
                m for m in stats.classification_history
                if period_start <= m.timestamp <= period_end
            ]
            
            if period_data:
                trend = TrendAnalysis(
                    period=f"{i}-{i+window_size_hours}h",
                    total_classifications=len(period_data),
                    average_confidence=self._calculate_average_confidence(period_data),
                    error_rate=self._calculate_error_rate(period_data),
                    average_response_time_ms=self._calculate_average_response_time(period_data),
                    intent_distribution=dict(Counter([m.intent for m in period_data])),
                    confidence_trend=[m.confidence for m in period_data],
                    response_time_trend=[m.response_time_ms for m in period_data],
                    error_trend=[1 if m.is_error else 0 for m in period_data]
                )
                trends.append(trend)
        
        return trends
    
    def _generate_performance_insights(self, stats: IntentClassifierStats, trends: List[TrendAnalysis]) -> PerformanceInsights:
        """성능 인사이트 생성"""
        if not stats.classification_history:
            return PerformanceInsights(
                overall_performance_score=0.0,
                confidence_stability=0.0,
                response_time_efficiency=0.0,
                error_rate_stability=0.0,
                recommendations=[],
                alerts=[]
            )
        
        # 기본 메트릭 계산
        confidences = [m.confidence for m in stats.classification_history]
        response_times = [m.response_time_ms for m in stats.classification_history]
        error_rate = self._calculate_error_rate(stats.classification_history)
        
        # 성능 점수 계산 (0-100 스케일)
        confidence_score = min(100, statistics.mean(confidences) * 100)
        response_time_score = max(0, 100 - (statistics.mean(response_times) / 10))  # 1초당 10점 감점
        error_rate_score = max(0, 100 - (error_rate * 1000))  # 0.1%당 100점 감점
        
        overall_score = (confidence_score + response_time_score + error_rate_score) / 3
        
        # 안정성 계산
        confidence_stability = 100 - (statistics.stdev(confidences) * 100) if len(confidences) > 1 else 100
        response_time_stability = 100 - (statistics.stdev(response_times) / 100) if len(response_times) > 1 else 100
        
        # 추천사항 생성
        recommendations = []
        if statistics.mean(confidences) < 0.7:
            recommendations.append("평균 신뢰도가 낮습니다. 프롬프트 개선을 고려하세요.")
        if statistics.mean(response_times) > 2000:
            recommendations.append("응답 시간이 느립니다. 모델 최적화를 고려하세요.")
        if error_rate > 0.05:
            recommendations.append("오류율이 높습니다. 오류 처리 로직을 검토하세요.")
        
        # 알림 생성
        alerts = []
        if error_rate > 0.1:
            alerts.append("HIGH_ERROR_RATE")
        if statistics.mean(response_times) > 5000:
            alerts.append("SLOW_RESPONSE_TIME")
        if confidence_stability < 50:
            alerts.append("UNSTABLE_CONFIDENCE")
        
        return PerformanceInsights(
            overall_performance_score=overall_score,
            confidence_stability=confidence_stability,
            response_time_efficiency=response_time_score,
            error_rate_stability=error_rate_score,
            recommendations=recommendations,
            alerts=alerts
        )
    
    def _analyze_errors(self, stats: IntentClassifierStats) -> Dict[str, Any]:
        """오류 분석"""
        error_data = [m for m in stats.classification_history if m.is_error]
        
        if not error_data:
            return {
                "total_errors": 0,
                "error_rate": 0.0,
                "error_patterns": {},
                "recent_errors": []
            }
        
        # 최근 오류들
        recent_errors = sorted(error_data, key=lambda x: x.timestamp, reverse=True)[:10]
        
        return {
            "total_errors": len(error_data),
            "error_rate": len(error_data) / len(stats.classification_history),
            "error_patterns": dict(Counter([m.intent for m in error_data])),
            "recent_errors": [asdict(m) for m in recent_errors]
        }
    
    def _analyze_confidence_distribution(self, stats: IntentClassifierStats) -> Dict[str, Any]:
        """신뢰도 분포 분석"""
        if not stats.classification_history:
            return {"distribution": {}, "percentiles": {}}
        
        confidences = [m.confidence for m in stats.classification_history]
        
        # 구간별 분포
        bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        distribution = {}
        for i in range(len(bins) - 1):
            count = sum(1 for c in confidences if bins[i] <= c < bins[i+1])
            distribution[f"{bins[i]:.1f}-{bins[i+1]:.1f}"] = count
        
        # 백분위수
        confidences_sorted = sorted(confidences)
        percentiles = {}
        for p in [25, 50, 75, 90, 95, 99]:
            idx = int(len(confidences_sorted) * p / 100)
            percentiles[f"p{p}"] = confidences_sorted[min(idx, len(confidences_sorted) - 1)]
        
        return {
            "distribution": distribution,
            "percentiles": percentiles,
            "mean": statistics.mean(confidences),
            "std_dev": statistics.stdev(confidences) if len(confidences) > 1 else 0.0
        }
    
    def _analyze_response_times(self, stats: IntentClassifierStats) -> Dict[str, Any]:
        """응답 시간 분석"""
        if not stats.classification_history:
            return {"distribution": {}, "percentiles": {}}
        
        response_times = [m.response_time_ms for m in stats.classification_history]
        
        # 구간별 분포
        bins = [0, 500, 1000, 2000, 5000, float('inf')]
        bin_labels = ["<0.5s", "0.5-1s", "1-2s", "2-5s", ">5s"]
        distribution = {}
        for i in range(len(bins) - 1):
            count = sum(1 for rt in response_times if bins[i] <= rt < bins[i+1])
            distribution[bin_labels[i]] = count
        
        # 백분위수
        response_times_sorted = sorted(response_times)
        percentiles = {}
        for p in [25, 50, 75, 90, 95, 99]:
            idx = int(len(response_times_sorted) * p / 100)
            percentiles[f"p{p}"] = response_times_sorted[min(idx, len(response_times_sorted) - 1)]
        
        return {
            "distribution": distribution,
            "percentiles": percentiles,
            "mean": statistics.mean(response_times),
            "std_dev": statistics.stdev(response_times) if len(response_times) > 1 else 0.0
        }
    
    def _calculate_error_rate(self, metrics: List[ClassificationMetrics]) -> float:
        """오류율 계산"""
        if not metrics:
            return 0.0
        return sum(1 for m in metrics if m.is_error) / len(metrics)
    
    def _calculate_average_confidence(self, metrics: List[ClassificationMetrics]) -> float:
        """평균 신뢰도 계산"""
        if not metrics:
            return 0.0
        return statistics.mean([m.confidence for m in metrics])
    
    def _calculate_average_response_time(self, metrics: List[ClassificationMetrics]) -> float:
        """평균 응답 시간 계산"""
        if not metrics:
            return 0.0
        return statistics.mean([m.response_time_ms for m in metrics])
    
    def _check_alerts(self, stats: IntentClassifierStats, recent_metrics: List[ClassificationMetrics]) -> List[str]:
        """알림 확인"""
        alerts = []
        
        if not recent_metrics:
            return alerts
        
        # 최근 오류율 확인
        recent_error_rate = self._calculate_error_rate(recent_metrics)
        if recent_error_rate > 0.1:
            alerts.append("HIGH_RECENT_ERROR_RATE")
        
        # 응답 시간 확인
        avg_response_time = self._calculate_average_response_time(recent_metrics)
        if avg_response_time > 5000:
            alerts.append("SLOW_RECENT_RESPONSE_TIME")
        
        # 신뢰도 확인
        avg_confidence = self._calculate_average_confidence(recent_metrics)
        if avg_confidence < 0.5:
            alerts.append("LOW_RECENT_CONFIDENCE")
        
        return alerts
    
    def _convert_to_csv(self, metrics: List[ClassificationMetrics]) -> str:
        """CSV 형태로 변환"""
        if not metrics:
            return "timestamp,intent,confidence,response_time_ms,is_error\n"
        
        csv_lines = ["timestamp,intent,confidence,response_time_ms,is_error"]
        for m in metrics:
            csv_lines.append(f"{m.timestamp},{m.intent},{m.confidence},{m.response_time_ms},{m.is_error}")
        
        return "\n".join(csv_lines)
    
    def _generate_error_report(self, error_message: str) -> Dict[str, Any]:
        """오류 보고서 생성"""
        return {
            "error": True,
            "error_message": error_message,
            "generated_at": datetime.now().isoformat(),
            "report_version": "1.0"
        }


class VisualizationGenerator:
    """시각화 데이터 생성기"""
    
    def __init__(self, reporter: StatisticsReporter):
        self.reporter = reporter
        self.logger = logging.getLogger(__name__)
    
    def generate_chart_data(self, chart_type: str, time_range_hours: int = 24) -> Dict[str, Any]:
        """차트 데이터 생성"""
        try:
            stats = self.reporter._get_latest_stats()
            
            if chart_type == "confidence_trend":
                return self._generate_confidence_trend_chart(stats, time_range_hours)
            elif chart_type == "response_time_trend":
                return self._generate_response_time_trend_chart(stats, time_range_hours)
            elif chart_type == "intent_distribution":
                return self._generate_intent_distribution_chart(stats)
            elif chart_type == "error_rate_trend":
                return self._generate_error_rate_trend_chart(stats, time_range_hours)
            else:
                return {"error": f"Unknown chart type: {chart_type}"}
                
        except Exception as e:
            self.logger.error(f"Failed to generate chart data for {chart_type}: {e}")
            return {"error": str(e)}
    
    def _generate_confidence_trend_chart(self, stats: IntentClassifierStats, time_range_hours: int) -> Dict[str, Any]:
        """신뢰도 추세 차트 데이터"""
        if not stats.classification_history:
            return {"labels": [], "datasets": []}
        
        # 시간 윈도우 설정
        window_size_minutes = 30  # 30분 윈도우
        current_time = time.time()
        start_time = current_time - (time_range_hours * 3600)
        
        labels = []
        confidence_data = []
        
        for i in range(0, time_range_hours * 2):  # 30분 간격
            window_start = start_time + (i * 30 * 60)
            window_end = window_start + (30 * 60)
            
            window_data = [
                m for m in stats.classification_history
                if window_start <= m.timestamp <= window_end
            ]
            
            if window_data:
                avg_confidence = self.reporter._calculate_average_confidence(window_data)
                labels.append(datetime.fromtimestamp(window_start).strftime("%H:%M"))
                confidence_data.append(avg_confidence)
        
        return {
            "labels": labels,
            "datasets": [{
                "label": "Average Confidence",
                "data": confidence_data,
                "borderColor": "rgb(75, 192, 192)",
                "backgroundColor": "rgba(75, 192, 192, 0.2)"
            }]
        }
    
    def _generate_response_time_trend_chart(self, stats: IntentClassifierStats, time_range_hours: int) -> Dict[str, Any]:
        """응답 시간 추세 차트 데이터"""
        if not stats.classification_history:
            return {"labels": [], "datasets": []}
        
        # 시간 윈도우 설정
        window_size_minutes = 30
        current_time = time.time()
        start_time = current_time - (time_range_hours * 3600)
        
        labels = []
        response_time_data = []
        
        for i in range(0, time_range_hours * 2):
            window_start = start_time + (i * 30 * 60)
            window_end = window_start + (30 * 60)
            
            window_data = [
                m for m in stats.classification_history
                if window_start <= m.timestamp <= window_end
            ]
            
            if window_data:
                avg_response_time = self.reporter._calculate_average_response_time(window_data)
                labels.append(datetime.fromtimestamp(window_start).strftime("%H:%M"))
                response_time_data.append(avg_response_time)
        
        return {
            "labels": labels,
            "datasets": [{
                "label": "Average Response Time (ms)",
                "data": response_time_data,
                "borderColor": "rgb(255, 99, 132)",
                "backgroundColor": "rgba(255, 99, 132, 0.2)"
            }]
        }
    
    def _generate_intent_distribution_chart(self, stats: IntentClassifierStats) -> Dict[str, Any]:
        """의도 분포 차트 데이터"""
        if not stats.classification_history:
            return {"labels": [], "datasets": []}
        
        intent_counts = Counter([m.intent for m in stats.classification_history])
        
        return {
            "labels": list(intent_counts.keys()),
            "datasets": [{
                "label": "Classification Count",
                "data": list(intent_counts.values()),
                "backgroundColor": [
                    "rgba(255, 99, 132, 0.8)",
                    "rgba(54, 162, 235, 0.8)",
                    "rgba(255, 205, 86, 0.8)",
                    "rgba(75, 192, 192, 0.8)",
                    "rgba(153, 102, 255, 0.8)",
                    "rgba(255, 159, 64, 0.8)"
                ]
            }]
        }
    
    def _generate_error_rate_trend_chart(self, stats: IntentClassifierStats, time_range_hours: int) -> Dict[str, Any]:
        """오류율 추세 차트 데이터"""
        if not stats.classification_history:
            return {"labels": [], "datasets": []}
        
        # 시간 윈도우 설정
        window_size_minutes = 30
        current_time = time.time()
        start_time = current_time - (time_range_hours * 3600)
        
        labels = []
        error_rate_data = []
        
        for i in range(0, time_range_hours * 2):
            window_start = start_time + (i * 30 * 60)
            window_end = window_start + (30 * 60)
            
            window_data = [
                m for m in stats.classification_history
                if window_start <= m.timestamp <= window_end
            ]
            
            if window_data:
                error_rate = self.reporter._calculate_error_rate(window_data)
                labels.append(datetime.fromtimestamp(window_start).strftime("%H:%M"))
                error_rate_data.append(error_rate * 100)  # 퍼센트로 변환
        
        return {
            "labels": labels,
            "datasets": [{
                "label": "Error Rate (%)",
                "data": error_rate_data,
                "borderColor": "rgb(255, 0, 0)",
                "backgroundColor": "rgba(255, 0, 0, 0.2)"
            }]
        }


# 전역 인스턴스 생성
_reporter_instance: Optional[StatisticsReporter] = None
_visualization_instance: Optional[VisualizationGenerator] = None


def get_reporter() -> StatisticsReporter:
    """StatisticsReporter 싱글톤 인스턴스 반환"""
    global _reporter_instance
    if _reporter_instance is None:
        _reporter_instance = StatisticsReporter()
    return _reporter_instance


def get_visualization_generator() -> VisualizationGenerator:
    """VisualizationGenerator 싱글톤 인스턴스 반환"""
    global _visualization_instance
    if _visualization_instance is None:
        _visualization_instance = VisualizationGenerator(get_reporter())
    return _visualization_instance
