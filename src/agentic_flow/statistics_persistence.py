#!/usr/bin/env python3
"""
Statistics Persistence Layer

This module implements comprehensive data persistence for intent classification statistics,
including file-based storage, data retention policies, and backup mechanisms.
"""

import json
import os
import time
import threading
import shutil
import gzip
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import logging
import sqlite3
from contextlib import contextmanager


@dataclass
class PersistenceConfig:
    """데이터 지속성 설정"""
    storage_directory: str = "statistics_data"
    backup_directory: str = "statistics_backups"
    retention_days: int = 30
    compression_enabled: bool = True
    backup_interval_hours: int = 24
    max_file_size_mb: int = 100
    enable_database: bool = False
    database_path: str = "statistics.db"


@dataclass
class DataRetentionPolicy:
    """데이터 보존 정책"""
    daily_data_days: int = 7
    hourly_data_days: int = 30
    raw_data_days: int = 3
    compressed_data_days: int = 365
    archive_data_days: int = 1095  # 3 years


class StatisticsPersistenceManager:
    """통계 데이터 지속성 관리자"""
    
    def __init__(self, config: Optional[PersistenceConfig] = None):
        self.config = config or PersistenceConfig()
        self.logger = logging.getLogger(__name__)
        self._lock = threading.RLock()
        
        # 디렉토리 생성
        self._ensure_directories()
        
        # 데이터베이스 초기화
        if self.config.enable_database:
            self._init_database()
        
        # 백업 스케줄러
        self._backup_timer: Optional[threading.Timer] = None
        self._schedule_backup()
    
    def _ensure_directories(self):
        """필요한 디렉토리 생성"""
        directories = [
            self.config.storage_directory,
            self.config.backup_directory,
            os.path.join(self.config.storage_directory, "daily"),
            os.path.join(self.config.storage_directory, "hourly"),
            os.path.join(self.config.storage_directory, "raw"),
            os.path.join(self.config.storage_directory, "compressed"),
            os.path.join(self.config.storage_directory, "archive")
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def _init_database(self):
        """SQLite 데이터베이스 초기화"""
        try:
            with sqlite3.connect(self.config.database_path) as conn:
                cursor = conn.cursor()
                
                # 통계 테이블 생성
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS classification_stats (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp REAL NOT NULL,
                        total_classifications INTEGER NOT NULL,
                        total_errors INTEGER NOT NULL,
                        average_confidence REAL NOT NULL,
                        error_rate REAL NOT NULL,
                        intent_distribution TEXT NOT NULL,
                        confidence_distribution TEXT NOT NULL,
                        response_times TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # 개별 메트릭 테이블 생성
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS classification_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp REAL NOT NULL,
                        intent TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        response_time_ms REAL NOT NULL,
                        is_error BOOLEAN NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # 인덱스 생성
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_stats_timestamp ON classification_stats(timestamp)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON classification_metrics(timestamp)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_metrics_intent ON classification_metrics(intent)")
                
                conn.commit()
                self.logger.info("Database initialized successfully")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {e}")
            self.config.enable_database = False
    
    @contextmanager
    def _get_db_connection(self):
        """데이터베이스 연결 컨텍스트 매니저"""
        conn = sqlite3.connect(self.config.database_path)
        try:
            yield conn
        finally:
            conn.close()
    
    def save_statistics(self, stats: Any, 
                        metrics_batch: Optional[List[Any]] = None) -> bool:
        """통계 데이터 저장"""
        try:
            with self._lock:
                timestamp = time.time()
                
                # 파일 기반 저장
                self._save_to_files(stats, metrics_batch, timestamp)
                
                # 데이터베이스 저장
                if self.config.enable_database:
                    self._save_to_database(stats, metrics_batch, timestamp)
                
                # 데이터 압축 및 아카이빙
                self._compress_old_data()
                
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to save statistics: {e}")
            return False
    
    def _save_to_files(self, stats: Any, 
                       metrics_batch: Optional[List[Any]], 
                       timestamp: float):
        """파일 기반 저장"""
        date_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d")
        hour_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d_%H")
        
        # 일별 통계 저장
        daily_file = os.path.join(self.config.storage_directory, "daily", f"{date_str}.json")
        self._append_to_file(daily_file, {
            "timestamp": timestamp,
            "stats": stats.to_dict(),
            "type": "daily_summary"
        })
        
        # 시간별 통계 저장
        hourly_file = os.path.join(self.config.storage_directory, "hourly", f"{hour_str}.json")
        self._append_to_file(hourly_file, {
            "timestamp": timestamp,
            "stats": stats.to_dict(),
            "type": "hourly_summary"
        })
        
        # 원시 메트릭 저장
        if metrics_batch:
            raw_file = os.path.join(self.config.storage_directory, "raw", f"{date_str}_metrics.json")
            for metrics in metrics_batch:
                self._append_to_file(raw_file, {
                    "timestamp": metrics.timestamp,
                    "metrics": asdict(metrics),
                    "type": "raw_metrics"
                })
    
    def _save_to_database(self, stats: Any, 
                          metrics_batch: Optional[List[Any]], 
                          timestamp: float):
        """데이터베이스 저장"""
        try:
            with sqlite3.connect(self.config.database_path) as conn:
                cursor = conn.cursor()
                
                # 통계 저장
                cursor.execute("""
                    INSERT INTO classification_stats 
                    (timestamp, total_classifications, total_errors, average_confidence, 
                     error_rate, intent_distribution, confidence_distribution, response_times)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    timestamp,
                    stats.total_classifications,
                    stats.total_errors,
                    stats.average_confidence,
                    stats.error_rate,
                    json.dumps(dict(stats.intent_distribution)),
                    json.dumps(dict(stats.confidence_distribution)),
                    json.dumps(stats.response_times)
                ))
                
                # 개별 메트릭 저장
                if metrics_batch:
                    metrics_data = [
                        (metrics.timestamp, metrics.intent, metrics.confidence, 
                         metrics.response_time_ms, metrics.is_error)
                        for metrics in metrics_batch
                    ]
                    cursor.executemany("""
                        INSERT INTO classification_metrics 
                        (timestamp, intent, confidence, response_time_ms, is_error)
                        VALUES (?, ?, ?, ?, ?)
                    """, metrics_data)
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to save to database: {e}")
    
    def _append_to_file(self, file_path: str, data: Dict[str, Any]):
        """파일에 데이터 추가"""
        try:
            # 파일 크기 확인
            if os.path.exists(file_path):
                file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
                if file_size_mb > self.config.max_file_size_mb:
                    self._rotate_file(file_path)
            
            # 데이터 추가
            with open(file_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(data, ensure_ascii=False) + '\n')
                
        except Exception as e:
            self.logger.error(f"Failed to append to file {file_path}: {e}")
    
    def _rotate_file(self, file_path: str):
        """파일 로테이션"""
        try:
            timestamp = int(time.time())
            rotated_path = f"{file_path}.{timestamp}"
            shutil.move(file_path, rotated_path)
            
            # 압축
            if self.config.compression_enabled:
                self._compress_file(rotated_path)
                
        except Exception as e:
            self.logger.error(f"Failed to rotate file {file_path}: {e}")
    
    def _compress_file(self, file_path: str):
        """파일 압축"""
        try:
            compressed_path = f"{file_path}.gz"
            with open(file_path, 'rb') as f_in:
                with gzip.open(compressed_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            os.remove(file_path)
            self.logger.debug(f"Compressed file: {file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to compress file {file_path}: {e}")
    
    def _compress_old_data(self):
        """오래된 데이터 압축"""
        try:
            cutoff_time = time.time() - (self.config.retention_days * 24 * 3600)
            
            for subdir in ["daily", "hourly", "raw"]:
                dir_path = os.path.join(self.config.storage_directory, subdir)
                if not os.path.exists(dir_path):
                    continue
                
                for filename in os.listdir(dir_path):
                    if filename.endswith('.json'):
                        file_path = os.path.join(dir_path, filename)
                        if os.path.getmtime(file_path) < cutoff_time:
                            self._compress_file(file_path)
                            
        except Exception as e:
            self.logger.error(f"Failed to compress old data: {e}")
    
    def load_statistics(self, start_time: Optional[float] = None, 
                       end_time: Optional[float] = None) -> List[Dict[str, Any]]:
        """통계 데이터 로드"""
        try:
            with self._lock:
                if self.config.enable_database:
                    return self._load_from_database(start_time, end_time)
                else:
                    return self._load_from_files(start_time, end_time)
                    
        except Exception as e:
            self.logger.error(f"Failed to load statistics: {e}")
            return []
    
    def _load_from_database(self, start_time: Optional[float], 
                           end_time: Optional[float]) -> List[Dict[str, Any]]:
        """데이터베이스에서 로드"""
        try:
            with sqlite3.connect(self.config.database_path) as conn:
                cursor = conn.cursor()
                
                query = "SELECT * FROM classification_stats WHERE 1=1"
                params = []
                
                if start_time:
                    query += " AND timestamp >= ?"
                    params.append(start_time)
                
                if end_time:
                    query += " AND timestamp <= ?"
                    params.append(end_time)
                
                query += " ORDER BY timestamp DESC"
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                # 컬럼 이름 가져오기
                columns = [description[0] for description in cursor.description]
                
                # 결과 변환
                results = []
                for row in rows:
                    row_dict = dict(zip(columns, row))
                    
                    # JSON 필드 파싱
                    row_dict['intent_distribution'] = json.loads(row_dict['intent_distribution'])
                    row_dict['confidence_distribution'] = json.loads(row_dict['confidence_distribution'])
                    row_dict['response_times'] = json.loads(row_dict['response_times'])
                    
                    results.append(row_dict)
                
                return results
                
        except Exception as e:
            self.logger.error(f"Failed to load from database: {e}")
            return []
    
    def _load_from_files(self, start_time: Optional[float], 
                        end_time: Optional[float]) -> List[Dict[str, Any]]:
        """파일에서 로드"""
        results = []
        
        try:
            # 일별 데이터 로드
            daily_dir = os.path.join(self.config.storage_directory, "daily")
            if os.path.exists(daily_dir):
                for filename in os.listdir(daily_dir):
                    if filename.endswith('.json') or filename.endswith('.json.gz'):
                        file_path = os.path.join(daily_dir, filename)
                        file_results = self._load_from_file(file_path, start_time, end_time)
                        results.extend(file_results)
            
            # 시간순 정렬
            results.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to load from files: {e}")
            return []
    
    def _load_from_file(self, file_path: str, start_time: Optional[float], 
                       end_time: Optional[float]) -> List[Dict[str, Any]]:
        """단일 파일에서 로드"""
        results = []
        
        try:
            # 압축 파일 처리
            if file_path.endswith('.gz'):
                with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                    lines = f.readlines()
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
            
            for line in lines:
                try:
                    data = json.loads(line.strip())
                    
                    # 시간 필터링
                    timestamp = data.get('timestamp', 0)
                    if start_time and timestamp < start_time:
                        continue
                    if end_time and timestamp > end_time:
                        continue
                    
                    results.append(data)
                    
                except json.JSONDecodeError:
                    continue
                    
        except Exception as e:
            self.logger.error(f"Failed to load from file {file_path}: {e}")
        
        return results
    
    def cleanup_old_data(self, retention_policy: Optional[DataRetentionPolicy] = None):
        """오래된 데이터 정리"""
        if retention_policy is None:
            retention_policy = DataRetentionPolicy()
        
        try:
            with self._lock:
                current_time = time.time()
                
                # 각 디렉토리별 정리
                cleanup_rules = [
                    ("raw", retention_policy.raw_data_days),
                    ("hourly", retention_policy.hourly_data_days),
                    ("daily", retention_policy.daily_data_days),
                    ("compressed", retention_policy.compressed_data_days),
                    ("archive", retention_policy.archive_data_days)
                ]
                
                for subdir, retention_days in cleanup_rules:
                    self._cleanup_directory(subdir, current_time, retention_days)
                
                # 데이터베이스 정리
                if self.config.enable_database:
                    self._cleanup_database(current_time, retention_policy)
                    
        except Exception as e:
            self.logger.error(f"Failed to cleanup old data: {e}")
    
    def _cleanup_directory(self, subdir: str, current_time: float, retention_days: int):
        """디렉토리 정리"""
        try:
            dir_path = os.path.join(self.config.storage_directory, subdir)
            if not os.path.exists(dir_path):
                return
            
            cutoff_time = current_time - (retention_days * 24 * 3600)
            
            for filename in os.listdir(dir_path):
                file_path = os.path.join(dir_path, filename)
                
                if os.path.isfile(file_path) and os.path.getmtime(file_path) < cutoff_time:
                    os.remove(file_path)
                    self.logger.debug(f"Removed old file: {file_path}")
                    
        except Exception as e:
            self.logger.error(f"Failed to cleanup directory {subdir}: {e}")
    
    def _cleanup_database(self, current_time: float, retention_policy: DataRetentionPolicy):
        """데이터베이스 정리"""
        try:
            with sqlite3.connect(self.config.database_path) as conn:
                cursor = conn.cursor()
                
                # 개별 메트릭 정리 (3일)
                cutoff_time = current_time - (retention_policy.raw_data_days * 24 * 3600)
                cursor.execute("DELETE FROM classification_metrics WHERE timestamp < ?", (cutoff_time,))
                
                # 통계 데이터 정리 (30일)
                cutoff_time = current_time - (retention_policy.hourly_data_days * 24 * 3600)
                cursor.execute("DELETE FROM classification_stats WHERE timestamp < ?", (cutoff_time,))
                
                # VACUUM 실행
                cursor.execute("VACUUM")
                
                conn.commit()
                self.logger.info("Database cleanup completed")
                
        except Exception as e:
            self.logger.error(f"Failed to cleanup database: {e}")
    
    def create_backup(self) -> bool:
        """백업 생성"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = os.path.join(self.config.backup_directory, f"backup_{timestamp}")
            
            # 디렉토리 백업
            shutil.copytree(self.config.storage_directory, backup_path)
            
            # 데이터베이스 백업
            if self.config.enable_database and os.path.exists(self.config.database_path):
                db_backup_path = os.path.join(backup_path, "statistics.db")
                shutil.copy2(self.config.database_path, db_backup_path)
            
            # 압축
            if self.config.compression_enabled:
                compressed_backup = f"{backup_path}.tar.gz"
                shutil.make_archive(backup_path, 'gztar', backup_path)
                shutil.rmtree(backup_path)
                backup_path = compressed_backup
            
            self.logger.info(f"Backup created: {backup_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create backup: {e}")
            return False
    
    def _schedule_backup(self):
        """백업 스케줄링"""
        if self._backup_timer:
            self._backup_timer.cancel()
        
        self._backup_timer = threading.Timer(
            self.config.backup_interval_hours * 3600,
            self._perform_scheduled_backup
        )
        self._backup_timer.start()
    
    def _perform_scheduled_backup(self):
        """스케줄된 백업 실행"""
        try:
            self.create_backup()
            self._schedule_backup()  # 다음 백업 스케줄링
            
        except Exception as e:
            self.logger.error(f"Scheduled backup failed: {e}")
    
    def get_storage_info(self) -> Dict[str, Any]:
        """저장소 정보 반환"""
        try:
            info = {
                "storage_directory": self.config.storage_directory,
                "backup_directory": self.config.backup_directory,
                "database_enabled": self.config.enable_database,
                "database_path": self.config.database_path if self.config.enable_database else None,
                "retention_days": self.config.retention_days,
                "compression_enabled": self.config.compression_enabled,
                "backup_interval_hours": self.config.backup_interval_hours,
                "max_file_size_mb": self.config.max_file_size_mb
            }
            
            # 디렉토리 크기 계산
            total_size = 0
            for root, dirs, files in os.walk(self.config.storage_directory):
                for file in files:
                    file_path = os.path.join(root, file)
                    if os.path.exists(file_path):
                        total_size += os.path.getsize(file_path)
            
            info["total_storage_size_mb"] = total_size / (1024 * 1024)
            
            return info
            
        except Exception as e:
            self.logger.error(f"Failed to get storage info: {e}")
            return {}
    
    def shutdown(self):
        """지속성 관리자 종료"""
        try:
            if self._backup_timer:
                self._backup_timer.cancel()
            
            # 마지막 백업 생성
            self.create_backup()
            
            self.logger.info("Persistence manager shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")


# 전역 지속성 관리자 인스턴스
_persistence_manager: Optional[StatisticsPersistenceManager] = None


def get_persistence_manager() -> StatisticsPersistenceManager:
    """전역 지속성 관리자 인스턴스 반환"""
    global _persistence_manager
    if _persistence_manager is None:
        _persistence_manager = StatisticsPersistenceManager()
    return _persistence_manager


def shutdown_persistence_manager():
    """전역 지속성 관리자 종료"""
    global _persistence_manager
    if _persistence_manager:
        _persistence_manager.shutdown()
        _persistence_manager = None
