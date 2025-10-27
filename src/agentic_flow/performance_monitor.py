#!/usr/bin/env python3
"""
Performance Monitor
성능 모니터링 및 품질 평가 시스템
"""

import time
import psutil
import threading
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
import os
from enum import Enum

logger = logging.getLogger(__name__)

class MetricType(Enum):
    """메트릭 타입"""
    RESPONSE_TIME = "response_time"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    SUCCESS_RATE = "success_rate"
    ERROR_RATE = "error_rate"
    THROUGHPUT = "throughput"
    CONFIDENCE_SCORE = "confidence_score"

class AlertLevel(Enum):
    """알림 레벨"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

@dataclass
class PerformanceMetric:
    """성능 메트릭"""
    timestamp: datetime
    metric_type: MetricType
    value: float
    unit: str
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class QualityMetric:
    """품질 메트릭"""
    timestamp: datetime
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confidence: float
    user_satisfaction: Optional[float] = None

@dataclass
class SystemAlert:
    """시스템 알림"""
    timestamp: datetime
    level: AlertLevel
    message: str
    metric_type: MetricType
    current_value: float
    threshold: float
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PerformanceReport:
    """성능 리포트"""
    timestamp: datetime
    period: str
    metrics_summary: Dict[str, Any]
    quality_summary: Dict[str, Any]
    alerts: List[SystemAlert]
    recommendations: List[str]
    system_health: str

class PerformanceMonitor:
    """성능 모니터링 시스템"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.metrics_history: deque = deque(maxlen=1000)  # 최근 1000개 메트릭만 유지
        self.quality_history: deque = deque(maxlen=500)  # 최근 500개 품질 메트릭만 유지
        self.alerts: List[SystemAlert] = []
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # 임계값 설정 (메모리 사용량 임계값 상향 조정)
        self.thresholds = {
            MetricType.RESPONSE_TIME: 2.0,  # 2초
            MetricType.MEMORY_USAGE: 90.0,   # 80% -> 90%로 상향
            MetricType.CPU_USAGE: 90.0,     # 90%
            MetricType.SUCCESS_RATE: 70.0,  # 70%
            MetricType.ERROR_RATE: 10.0,    # 10%
            MetricType.CONFIDENCE_SCORE: 0.6 # 60%
        }
        
        # 통계 데이터
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "avg_response_time": 0.0,
            "avg_confidence": 0.0,
            "peak_memory_usage": 0.0,
            "peak_cpu_usage": 0.0
        }
        
        # 모니터링 시작
        self.start_monitoring()
    
    def start_monitoring(self):
        """모니터링 시작"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            logger.info("성능 모니터링 시작")
    
    def stop_monitoring(self):
        """모니터링 중지"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        logger.info("성능 모니터링 중지")
    
    def _monitor_loop(self):
        """모니터링 루프"""
        while self.monitoring_active:
            try:
                # 시스템 리소스 모니터링
                self._collect_system_metrics()
                time.sleep(5)  # 5초마다 수집
            except Exception as e:
                logger.error(f"모니터링 루프 오류: {str(e)}")
                time.sleep(10)
    
    def _collect_system_metrics(self):
        """시스템 메트릭 수집"""
        try:
            # 메모리 사용량
            memory_percent = psutil.virtual_memory().percent
            self._record_metric(MetricType.MEMORY_USAGE, memory_percent, "%")
            
            # CPU 사용량
            cpu_percent = psutil.cpu_percent(interval=1)
            self._record_metric(MetricType.CPU_USAGE, cpu_percent, "%")
            
            # 통계 업데이트
            self.stats["peak_memory_usage"] = max(self.stats["peak_memory_usage"], memory_percent)
            self.stats["peak_cpu_usage"] = max(self.stats["peak_cpu_usage"], cpu_percent)
            
        except Exception as e:
            logger.error(f"시스템 메트릭 수집 오류: {str(e)}")
    
    def record_request(self, start_time: float, end_time: float, success: bool, 
                      confidence: float = 0.0, context: Dict[str, Any] = None):
        """요청 기록"""
        try:
            response_time = end_time - start_time
            
            # 응답 시간 메트릭
            self._record_metric(MetricType.RESPONSE_TIME, response_time, "seconds", context)
            
            # 성공률 메트릭
            success_rate = 1.0 if success else 0.0
            self._record_metric(MetricType.SUCCESS_RATE, success_rate, "%", context)
            
            # 신뢰도 메트릭
            if confidence > 0:
                self._record_metric(MetricType.CONFIDENCE_SCORE, confidence, "score", context)
            
            # 통계 업데이트
            self.stats["total_requests"] += 1
            if success:
                self.stats["successful_requests"] += 1
            else:
                self.stats["failed_requests"] += 1
            
            # 평균 응답 시간 업데이트
            total_requests = self.stats["total_requests"]
            current_avg = self.stats["avg_response_time"]
            self.stats["avg_response_time"] = (current_avg * (total_requests - 1) + response_time) / total_requests
            
            # 평균 신뢰도 업데이트
            if confidence > 0:
                current_avg_conf = self.stats["avg_confidence"]
                self.stats["avg_confidence"] = (current_avg_conf * (total_requests - 1) + confidence) / total_requests
            
            # 임계값 체크 및 알림 생성
            self._check_thresholds()
            
        except Exception as e:
            logger.error(f"요청 기록 오류: {str(e)}")
    
    def record_quality_metric(self, accuracy: float, precision: float, recall: float, 
                            f1_score: float, confidence: float, user_satisfaction: Optional[float] = None):
        """품질 메트릭 기록"""
        try:
            quality_metric = QualityMetric(
                timestamp=datetime.now(),
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1_score,
                confidence=confidence,
                user_satisfaction=user_satisfaction
            )
            
            self.quality_history.append(quality_metric)
            logger.info(f"품질 메트릭 기록: F1={f1_score:.3f}, 정확도={accuracy:.3f}")
            
        except Exception as e:
            logger.error(f"품질 메트릭 기록 오류: {str(e)}")
    
    def _record_metric(self, metric_type: MetricType, value: float, unit: str, context: Dict[str, Any] = None):
        """메트릭 기록"""
        try:
            metric = PerformanceMetric(
                timestamp=datetime.now(),
                metric_type=metric_type,
                value=value,
                unit=unit,
                context=context or {}
            )
            
            self.metrics_history.append(metric)
            
        except Exception as e:
            logger.error(f"메트릭 기록 오류: {str(e)}")
    
    def _check_thresholds(self):
        """임계값 체크 및 알림 생성"""
        try:
            # 최근 메트릭 분석
            recent_metrics = list(self.metrics_history)[-10:]  # 최근 10개
            
            for metric in recent_metrics:
                threshold = self.thresholds.get(metric.metric_type)
                if threshold is None:
                    continue
                
                # 임계값 초과 체크
                if metric.value > threshold:
                    alert_level = self._determine_alert_level(metric.metric_type, metric.value, threshold)
                    
                    alert = SystemAlert(
                        timestamp=datetime.now(),
                        level=alert_level,
                        message=f"{metric.metric_type.value} 임계값 초과: {metric.value:.2f} > {threshold}",
                        metric_type=metric.metric_type,
                        current_value=metric.value,
                        threshold=threshold,
                        context=metric.context
                    )
                    
                    self.alerts.append(alert)
                    logger.warning(f"알림 생성: {alert.message}")
            
            # 최근 알림만 유지 (최대 100개)
            if len(self.alerts) > 100:
                self.alerts = self.alerts[-100:]
                
        except Exception as e:
            logger.error(f"임계값 체크 오류: {str(e)}")
    
    def _determine_alert_level(self, metric_type: MetricType, value: float, threshold: float) -> AlertLevel:
        """알림 레벨 결정"""
        ratio = value / threshold
        
        if ratio >= 2.0:  # 임계값의 2배 이상
            return AlertLevel.CRITICAL
        elif ratio >= 1.5:  # 임계값의 1.5배 이상
            return AlertLevel.WARNING
        else:
            return AlertLevel.INFO
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """성능 요약 조회"""
        try:
            # 최근 1시간 메트릭 분석
            one_hour_ago = datetime.now() - timedelta(hours=1)
            recent_metrics = [m for m in self.metrics_history if m.timestamp >= one_hour_ago]
            
            # 메트릭별 통계
            metrics_by_type = defaultdict(list)
            for metric in recent_metrics:
                metrics_by_type[metric.metric_type].append(metric.value)
            
            summary = {
                "timestamp": datetime.now().isoformat(),
                "period": "last_hour",
                "total_requests": self.stats["total_requests"],
                "successful_requests": self.stats["successful_requests"],
                "failed_requests": self.stats["failed_requests"],
                "success_rate": (self.stats["successful_requests"] / max(1, self.stats["total_requests"])) * 100,
                "avg_response_time": self.stats["avg_response_time"],
                "avg_confidence": self.stats["avg_confidence"],
                "peak_memory_usage": self.stats["peak_memory_usage"],
                "peak_cpu_usage": self.stats["peak_cpu_usage"],
                "recent_metrics": {}
            }
            
            # 최근 메트릭 통계
            for metric_type, values in metrics_by_type.items():
                if values:
                    summary["recent_metrics"][metric_type.value] = {
                        "count": len(values),
                        "avg": sum(values) / len(values),
                        "min": min(values),
                        "max": max(values),
                        "latest": values[-1] if values else 0
                    }
            
            return summary
            
        except Exception as e:
            logger.error(f"성능 요약 조회 오류: {str(e)}")
            return {"error": str(e)}
    
    def get_quality_summary(self) -> Dict[str, Any]:
        """품질 요약 조회"""
        try:
            if not self.quality_history:
                return {"message": "품질 메트릭 데이터가 없습니다."}
            
            # 최근 품질 메트릭 분석
            recent_quality = list(self.quality_history)[-10:]  # 최근 10개
            
            summary = {
                "timestamp": datetime.now().isoformat(),
                "total_quality_metrics": len(self.quality_history),
                "recent_metrics": len(recent_quality),
                "avg_accuracy": sum(q.accuracy for q in recent_quality) / len(recent_quality),
                "avg_precision": sum(q.precision for q in recent_quality) / len(recent_quality),
                "avg_recall": sum(q.recall for q in recent_quality) / len(recent_quality),
                "avg_f1_score": sum(q.f1_score for q in recent_quality) / len(recent_quality),
                "avg_confidence": sum(q.confidence for q in recent_quality) / len(recent_quality),
                "avg_user_satisfaction": None
            }
            
            # 사용자 만족도 평균 (있는 경우)
            satisfaction_values = [q.user_satisfaction for q in recent_quality if q.user_satisfaction is not None]
            if satisfaction_values:
                summary["avg_user_satisfaction"] = sum(satisfaction_values) / len(satisfaction_values)
            
            return summary
            
        except Exception as e:
            logger.error(f"품질 요약 조회 오류: {str(e)}")
            return {"error": str(e)}
    
    def get_system_health(self) -> str:
        """시스템 상태 평가"""
        try:
            # 최근 알림 분석
            recent_alerts = [a for a in self.alerts if a.timestamp >= datetime.now() - timedelta(hours=1)]
            
            critical_alerts = [a for a in recent_alerts if a.level == AlertLevel.CRITICAL]
            warning_alerts = [a for a in recent_alerts if a.level == AlertLevel.WARNING]
            
            if critical_alerts:
                return "critical"
            elif warning_alerts:
                return "warning"
            elif self.stats["total_requests"] == 0:
                return "unknown"
            else:
                return "healthy"
                
        except Exception as e:
            logger.error(f"시스템 상태 평가 오류: {str(e)}")
            return "error"
    
    def generate_recommendations(self) -> List[str]:
        """개선 권장사항 생성"""
        recommendations = []
        
        try:
            # 성공률 기반 권장사항
            success_rate = (self.stats["successful_requests"] / max(1, self.stats["total_requests"])) * 100
            if success_rate < 70:
                recommendations.append(f"성공률이 낮습니다 ({success_rate:.1f}%). 매핑 로직 개선이 필요합니다.")
            
            # 응답 시간 기반 권장사항
            if self.stats["avg_response_time"] > 2.0:
                recommendations.append(f"평균 응답 시간이 느립니다 ({self.stats['avg_response_time']:.2f}초). 성능 최적화가 필요합니다.")
            
            # 메모리 사용량 기반 권장사항 (임계값 상향 조정)
            if self.stats["peak_memory_usage"] > 90:  # 80% -> 90%로 상향
                recommendations.append(f"메모리 사용량이 높습니다 ({self.stats['peak_memory_usage']:.1f}%). 메모리 최적화가 필요합니다.")
            
            # CPU 사용량 기반 권장사항
            if self.stats["peak_cpu_usage"] > 90:
                recommendations.append(f"CPU 사용량이 높습니다 ({self.stats['peak_cpu_usage']:.1f}%). CPU 최적화가 필요합니다.")
            
            # 신뢰도 기반 권장사항
            if self.stats["avg_confidence"] < 0.6:
                recommendations.append(f"평균 신뢰도가 낮습니다 ({self.stats['avg_confidence']:.2f}). 매핑 정확도 개선이 필요합니다.")
            
            # 알림 기반 권장사항
            recent_alerts = [a for a in self.alerts if a.timestamp >= datetime.now() - timedelta(hours=1)]
            if recent_alerts:
                critical_count = len([a for a in recent_alerts if a.level == AlertLevel.CRITICAL])
                if critical_count > 0:
                    recommendations.append(f"최근 {critical_count}개의 심각한 알림이 발생했습니다. 즉시 조치가 필요합니다.")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"권장사항 생성 오류: {str(e)}")
            return ["권장사항 생성 중 오류가 발생했습니다."]
    
    def generate_performance_report(self) -> PerformanceReport:
        """성능 리포트 생성"""
        try:
            performance_summary = self.get_performance_summary()
            quality_summary = self.get_quality_summary()
            system_health = self.get_system_health()
            recommendations = self.generate_recommendations()
            
            # 최근 알림
            recent_alerts = [a for a in self.alerts if a.timestamp >= datetime.now() - timedelta(hours=24)]
            
            report = PerformanceReport(
                timestamp=datetime.now(),
                period="last_24_hours",
                metrics_summary=performance_summary,
                quality_summary=quality_summary,
                alerts=recent_alerts,
                recommendations=recommendations,
                system_health=system_health
            )
            
            return report
            
        except Exception as e:
            logger.error(f"성능 리포트 생성 오류: {str(e)}")
            return PerformanceReport(
                timestamp=datetime.now(),
                period="error",
                metrics_summary={"error": str(e)},
                quality_summary={"error": str(e)},
                alerts=[],
                recommendations=["리포트 생성 중 오류가 발생했습니다."],
                system_health="error"
            )
    
    def export_metrics(self, filepath: str):
        """메트릭 데이터 내보내기"""
        try:
            data = {
                "export_timestamp": datetime.now().isoformat(),
                "metrics": [
                    {
                        "timestamp": m.timestamp.isoformat(),
                        "metric_type": m.metric_type.value,
                        "value": m.value,
                        "unit": m.unit,
                        "context": m.context
                    }
                    for m in self.metrics_history
                ],
                "quality_metrics": [
                    {
                        "timestamp": q.timestamp.isoformat(),
                        "accuracy": q.accuracy,
                        "precision": q.precision,
                        "recall": q.recall,
                        "f1_score": q.f1_score,
                        "confidence": q.confidence,
                        "user_satisfaction": q.user_satisfaction
                    }
                    for q in self.quality_history
                ],
                "alerts": [
                    {
                        "timestamp": a.timestamp.isoformat(),
                        "level": a.level.value,
                        "message": a.message,
                        "metric_type": a.metric_type.value,
                        "current_value": a.current_value,
                        "threshold": a.threshold,
                        "context": a.context
                    }
                    for a in self.alerts
                ],
                "statistics": self.stats
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"메트릭 데이터 내보내기 완료: {filepath}")
            
        except Exception as e:
            logger.error(f"메트릭 데이터 내보내기 오류: {str(e)}")
    
    def clear_old_data(self, days: int = 7):
        """오래된 데이터 정리"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            # 오래된 메트릭 제거
            self.metrics_history = deque(
                [m for m in self.metrics_history if m.timestamp >= cutoff_date],
                maxlen=1000
            )
            
            # 오래된 품질 메트릭 제거
            self.quality_history = deque(
                [q for q in self.quality_history if q.timestamp >= cutoff_date],
                maxlen=500
            )
            
            # 오래된 알림 제거
            self.alerts = [a for a in self.alerts if a.timestamp >= cutoff_date]
            
            logger.info(f"{days}일 이상 된 데이터 정리 완료")
            
        except Exception as e:
            logger.error(f"데이터 정리 오류: {str(e)}")


