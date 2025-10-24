#!/usr/bin/env python3
"""
시스템 건강 상태 검사 스크립트
정기적으로 시스템의 건강 상태를 점검하고 리포트를 생성합니다.
"""

import sys
import os
import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import argparse

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.core.db import test_database_connection, execute_query
from src.core.logging import get_logger
from src.monitoring.error_monitor import get_error_monitor
from src.monitoring.alert_system import get_alert_manager

logger = get_logger(__name__)


class HealthChecker:
    """시스템 건강 상태 검사기"""
    
    def __init__(self):
        """건강 상태 검사기 초기화"""
        self.results = {}
        self.start_time = time.time()
        self.error_monitor = get_error_monitor()
        self.alert_manager = get_alert_manager()
        
        logger.info("시스템 건강 상태 검사 시작")
    
    async def run_health_checks(self) -> Dict[str, Any]:
        """전체 건강 상태 검사 실행"""
        logger.info("건강 상태 검사 실행 중...")
        
        # 각 컴포넌트별 검사 실행
        checks = [
            ("database", self._check_database_health),
            ("slack", self._check_slack_health),
            ("sql_generation", self._check_sql_generation_health),
            ("monitoring", self._check_monitoring_health),
            ("performance", self._check_performance_health),
            ("error_rates", self._check_error_rates),
        ]
        
        for check_name, check_func in checks:
            try:
                logger.info(f"{check_name} 건강 상태 검사 중...")
                self.results[check_name] = await check_func()
                logger.info(f"{check_name} 검사 완료: {self.results[check_name]['status']}")
            except Exception as e:
                logger.error(f"{check_name} 검사 실패: {str(e)}")
                self.results[check_name] = {
                    "status": "error",
                    "message": str(e),
                    "timestamp": datetime.now().isoformat()
                }
        
        # 전체 상태 결정
        overall_status = self._determine_overall_status()
        
        # 검사 완료 시간 기록
        execution_time = time.time() - self.start_time
        
        return {
            "overall_status": overall_status,
            "execution_time_seconds": execution_time,
            "timestamp": datetime.now().isoformat(),
            "components": self.results,
            "summary": self._generate_summary()
        }
    
    async def _check_database_health(self) -> Dict[str, Any]:
        """데이터베이스 건강 상태 검사"""
        try:
            # 연결 테스트
            connection_ok = test_database_connection()
            if not connection_ok:
                return {
                    "status": "critical",
                    "message": "데이터베이스 연결 실패",
                    "timestamp": datetime.now().isoformat()
                }
            
            # 기본 쿼리 테스트
            start_time = time.time()
            result = execute_query("SELECT 1 as health_check", readonly=True)
            query_time = time.time() - start_time
            
            if not result or len(result) == 0:
                return {
                    "status": "critical",
                    "message": "기본 쿼리 실행 실패",
                    "timestamp": datetime.now().isoformat()
                }
            
            # 테이블 존재 확인
            table_check = execute_query(
                "SELECT COUNT(*) as table_count FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA = 'fanding'",
                readonly=True
            )
            
            table_count = table_check[0]['table_count'] if table_check else 0
            
            # 성능 기준 확인
            status = "healthy"
            if query_time > 2.0:
                status = "warning"
            elif query_time > 5.0:
                status = "critical"
            
            return {
                "status": status,
                "message": f"데이터베이스 정상 작동 (쿼리 시간: {query_time:.2f}초)",
                "timestamp": datetime.now().isoformat(),
                "metrics": {
                    "connection_ok": connection_ok,
                    "query_time_seconds": query_time,
                    "table_count": table_count
                }
            }
            
        except Exception as e:
            return {
                "status": "critical",
                "message": f"데이터베이스 검사 중 오류: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    async def _check_slack_health(self) -> Dict[str, Any]:
        """Slack 건강 상태 검사"""
        try:
            # Slack 설정 확인
            from src.core.config import get_settings
            settings = get_settings()
            
            slack_config = settings.slack
            has_bot_token = bool(slack_config.bot_token and slack_config.bot_token != "xoxb-your-bot-token-here")
            has_app_token = bool(slack_config.app_token and slack_config.app_token != "xapp-your-app-token-here")
            has_signing_secret = bool(slack_config.signing_secret and slack_config.signing_secret != "your-signing-secret-here")
            
            if not (has_bot_token and has_app_token and has_signing_secret):
                return {
                    "status": "warning",
                    "message": "Slack 토큰이 설정되지 않음 (서비스 제한적 작동)",
                    "timestamp": datetime.now().isoformat(),
                    "metrics": {
                        "bot_token_configured": has_bot_token,
                        "app_token_configured": has_app_token,
                        "signing_secret_configured": has_signing_secret
                    }
                }
            
            # Slack 핸들러 초기화 테스트
            try:
                from src.slack.handlers.message_handler import MessageHandler
                handler = MessageHandler()
                
                return {
                    "status": "healthy",
                    "message": "Slack 구성 요소 정상 초기화",
                    "timestamp": datetime.now().isoformat(),
                    "metrics": {
                        "bot_token_configured": has_bot_token,
                        "app_token_configured": has_app_token,
                        "signing_secret_configured": has_signing_secret,
                        "handler_initialized": True
                    }
                }
            except Exception as e:
                return {
                    "status": "warning",
                    "message": f"Slack 핸들러 초기화 실패: {str(e)}",
                    "timestamp": datetime.now().isoformat(),
                    "metrics": {
                        "bot_token_configured": has_bot_token,
                        "app_token_configured": has_app_token,
                        "signing_secret_configured": has_signing_secret,
                        "handler_initialized": False
                    }
                }
            
        except Exception as e:
            return {
                "status": "critical",
                "message": f"Slack 검사 중 오류: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    async def _check_sql_generation_health(self) -> Dict[str, Any]:
        """SQL 생성 건강 상태 검사"""
        try:
            from src.agentic_flow.sql_generator import SQLGenerator
            
            # SQL 생성기 초기화 테스트
            config = {
                'db_schema': {
                    't_member': {
                        'columns': {
                            'id': {'type': 'INT', 'nullable': False},
                            'email': {'type': 'VARCHAR', 'nullable': False}
                        }
                    }
                }
            }
            
            sql_generator = SQLGenerator(config)
            
            # MariaDB 구문 수정 테스트
            problematic_sql = "SELECT COUNT(*) as exists FROM t_member"
            fixed_sql = sql_generator._fix_mariadb_syntax(problematic_sql)
            
            syntax_fix_working = "table_exists" in fixed_sql and "as exists" not in fixed_sql
            
            # LLM 초기화 상태 확인
            llm_available = sql_generator.llm is not None
            
            status = "healthy"
            if not syntax_fix_working:
                status = "warning"
            if not llm_available:
                status = "warning" if status == "healthy" else status
            
            return {
                "status": status,
                "message": "SQL 생성기 정상 작동" if status == "healthy" else "SQL 생성기 부분적 작동",
                "timestamp": datetime.now().isoformat(),
                "metrics": {
                    "syntax_fix_working": syntax_fix_working,
                    "llm_available": llm_available,
                    "schema_configured": bool(config.get('db_schema'))
                }
            }
            
        except Exception as e:
            return {
                "status": "critical",
                "message": f"SQL 생성기 검사 중 오류: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    async def _check_monitoring_health(self) -> Dict[str, Any]:
        """모니터링 시스템 건강 상태 검사"""
        try:
            # 오류 모니터 상태 확인
            health_status = self.error_monitor.get_health_status()
            error_stats = self.error_monitor.get_error_statistics(hours=1)
            
            # 알림 시스템 상태 확인
            alert_history = self.alert_manager.get_alert_history(hours=24)
            
            # 모니터링 시스템 자체 상태
            monitoring_status = "healthy"
            if health_status['status'] == 'critical':
                monitoring_status = "critical"
            elif health_status['status'] == 'warning':
                monitoring_status = "warning"
            
            return {
                "status": monitoring_status,
                "message": f"모니터링 시스템 {health_status['message']}",
                "timestamp": datetime.now().isoformat(),
                "metrics": {
                    "system_health": health_status['status'],
                    "recent_events_count": health_status['recent_events_count'],
                    "critical_events_24h": health_status['critical_events_24h'],
                    "alert_count_24h": len(alert_history),
                    "components_status": health_status['components_status']
                }
            }
            
        except Exception as e:
            return {
                "status": "critical",
                "message": f"모니터링 시스템 검사 중 오류: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    async def _check_performance_health(self) -> Dict[str, Any]:
        """성능 건강 상태 검사"""
        try:
            # 데이터베이스 성능 테스트
            db_performance = await self._test_database_performance()
            
            # 메모리 사용량 확인
            import psutil
            memory_info = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # 성능 기준 확인
            status = "healthy"
            if db_performance['avg_query_time'] > 2.0:
                status = "warning"
            if memory_info.percent > 90:
                status = "critical" if status == "healthy" else status
            if cpu_percent > 80:
                status = "warning" if status == "healthy" else status
            
            return {
                "status": status,
                "message": f"성능 상태 {'정상' if status == 'healthy' else '주의 필요'}",
                "timestamp": datetime.now().isoformat(),
                "metrics": {
                    "database_performance": db_performance,
                    "memory_usage_percent": memory_info.percent,
                    "cpu_usage_percent": cpu_percent,
                    "memory_available_gb": memory_info.available / (1024**3)
                }
            }
            
        except Exception as e:
            return {
                "status": "warning",
                "message": f"성능 검사 중 오류: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    async def _test_database_performance(self) -> Dict[str, Any]:
        """데이터베이스 성능 테스트"""
        queries = [
            "SELECT 1 as test1",
            "SELECT COUNT(*) as count FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA = 'fanding'",
            "SELECT COUNT(*) as count FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA = 'fanding' AND TABLE_NAME LIKE 't_%'"
        ]
        
        query_times = []
        for query in queries:
            start_time = time.time()
            try:
                execute_query(query, readonly=True)
                query_time = time.time() - start_time
                query_times.append(query_time)
            except Exception as e:
                query_times.append(float('inf'))  # 실패한 쿼리는 무한대 시간
        
        return {
            "avg_query_time": sum(query_times) / len(query_times),
            "max_query_time": max(query_times),
            "min_query_time": min(query_times),
            "failed_queries": len([t for t in query_times if t == float('inf')])
        }
    
    async def _check_error_rates(self) -> Dict[str, Any]:
        """오류율 검사"""
        try:
            # 최근 1시간 오류 통계
            error_stats = self.error_monitor.get_error_statistics(hours=1)
            
            # 오류율 계산 (대략적)
            total_events = error_stats['total_events']
            critical_events = error_stats['severity_distribution'].get('critical', 0)
            high_events = error_stats['severity_distribution'].get('high', 0)
            
            # 상태 결정
            status = "healthy"
            if critical_events > 0:
                status = "critical"
            elif high_events >= 5:
                status = "warning"
            elif total_events >= 20:
                status = "warning"
            
            return {
                "status": status,
                "message": f"오류율 {'정상' if status == 'healthy' else '높음'}",
                "timestamp": datetime.now().isoformat(),
                "metrics": {
                    "total_events_1h": total_events,
                    "critical_events_1h": critical_events,
                    "high_events_1h": high_events,
                    "severity_distribution": error_stats['severity_distribution'],
                    "component_distribution": error_stats['component_distribution']
                }
            }
            
        except Exception as e:
            return {
                "status": "warning",
                "message": f"오류율 검사 중 오류: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    def _determine_overall_status(self) -> str:
        """전체 상태 결정"""
        statuses = [result['status'] for result in self.results.values()]
        
        if 'critical' in statuses:
            return 'critical'
        elif 'warning' in statuses:
            return 'warning'
        elif 'error' in statuses:
            return 'error'
        else:
            return 'healthy'
    
    def _generate_summary(self) -> Dict[str, Any]:
        """검사 결과 요약 생성"""
        total_checks = len(self.results)
        healthy_checks = len([r for r in self.results.values() if r['status'] == 'healthy'])
        warning_checks = len([r for r in self.results.values() if r['status'] == 'warning'])
        critical_checks = len([r for r in self.results.values() if r['status'] == 'critical'])
        error_checks = len([r for r in self.results.values() if r['status'] == 'error'])
        
        return {
            "total_checks": total_checks,
            "healthy_checks": healthy_checks,
            "warning_checks": warning_checks,
            "critical_checks": critical_checks,
            "error_checks": error_checks,
            "health_percentage": (healthy_checks / total_checks) * 100 if total_checks > 0 else 0
        }


async def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='시스템 건강 상태 검사')
    parser.add_argument('--output', '-o', help='결과 출력 파일 경로')
    parser.add_argument('--format', '-f', choices=['json', 'text'], default='json', help='출력 형식')
    parser.add_argument('--verbose', '-v', action='store_true', help='상세 출력')
    
    args = parser.parse_args()
    
    # 건강 상태 검사 실행
    checker = HealthChecker()
    results = await checker.run_health_checks()
    
    # 결과 출력
    if args.format == 'json':
        output = json.dumps(results, indent=2, ensure_ascii=False)
    else:
        output = format_text_report(results)
    
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(output)
        logger.info(f"건강 상태 리포트 저장: {args.output}")
    else:
        print(output)
    
    # 종료 코드 설정
    if results['overall_status'] in ['critical', 'error']:
        sys.exit(1)
    elif results['overall_status'] == 'warning':
        sys.exit(2)
    else:
        sys.exit(0)


def format_text_report(results: Dict[str, Any]) -> str:
    """텍스트 형식 리포트 생성"""
    report = []
    report.append("=" * 60)
    report.append("DataTalk Bot 시스템 건강 상태 리포트")
    report.append("=" * 60)
    report.append(f"검사 시간: {results['timestamp']}")
    report.append(f"실행 시간: {results['execution_time_seconds']:.2f}초")
    report.append(f"전체 상태: {results['overall_status'].upper()}")
    report.append("")
    
    # 요약 정보
    summary = results['summary']
    report.append("검사 요약:")
    report.append(f"  총 검사 항목: {summary['total_checks']}")
    report.append(f"  정상: {summary['healthy_checks']}")
    report.append(f"  경고: {summary['warning_checks']}")
    report.append(f"  치명적: {summary['critical_checks']}")
    report.append(f"  오류: {summary['error_checks']}")
    report.append(f"  건강도: {summary['health_percentage']:.1f}%")
    report.append("")
    
    # 컴포넌트별 상세 정보
    report.append("컴포넌트별 상태:")
    report.append("-" * 40)
    
    for component, result in results['components'].items():
        status_emoji = {
            'healthy': '✅',
            'warning': '⚠️',
            'critical': '🚨',
            'error': '❌'
        }.get(result['status'], '❓')
        
        report.append(f"{status_emoji} {component.upper()}: {result['status']}")
        report.append(f"    메시지: {result['message']}")
        
        if 'metrics' in result:
            report.append("    메트릭:")
            for metric, value in result['metrics'].items():
                if isinstance(value, (int, float)):
                    report.append(f"      {metric}: {value}")
                elif isinstance(value, dict):
                    report.append(f"      {metric}: {len(value)}개 항목")
        
        report.append("")
    
    return "\n".join(report)


if __name__ == "__main__":
    asyncio.run(main())




