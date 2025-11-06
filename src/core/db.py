"""
Database connection module for DataTalk bot.
Handles MariaDB connection using SQLAlchemy with read-only and read-write access.
"""

import logging
import re
from typing import Optional, Dict, Any, List, Union, Generator
from contextlib import contextmanager
from datetime import datetime
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker, Session

from .config import get_settings, get_database_url
from functools import lru_cache
from sqlalchemy import inspect
from sqlalchemy.exc import OperationalError, ProgrammingError
import sqlparse
from sqlparse.sql import IdentifierList, Identifier, TokenList
from sqlparse.tokens import Keyword, DML, Name
try:
    from src.monitoring.error_monitor import record_database_error
except ImportError:
    # 모니터링 모듈이 없는 경우 무시
    def record_database_error(message, severity):
        pass


logger = logging.getLogger(__name__)


class DatabaseConnectionError(Exception):
    """Custom exception for database connection errors."""
    pass


class DatabaseQueryError(Exception):
    """Custom exception for database query errors."""
    pass


class BaseDatabaseConnection:
    """
    Base database connection class with common functionality.
    Provides shared methods for both read-only and read-write connections.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize base database connection.
        
        Args:
            config: Database configuration dictionary
        """
        self.settings = get_settings()
        self.config = config or self._get_db_config()
        self.sqlalchemy_engine: Optional[Engine] = None
        self._connection_attempts = 0
        self._max_retries = 3
        
    def _get_db_config(self) -> Dict[str, Any]:
        """Get database configuration from settings."""
        db_config = self.settings.database
        return {
            'host': db_config.host,
            'port': db_config.port,
            'user': db_config.username,
            'password': db_config.password,
            'database': db_config.database,
            'charset': db_config.charset,
            'autocommit': db_config.autocommit,
            'use_unicode': True,
            'sql_mode': 'TRADITIONAL',
        }
    
    def create_sqlalchemy_engine(self, pool_size: int = 20, max_overflow: int = 20) -> Engine:
        """
        Create SQLAlchemy engine for database operations.
        
        Args:
            pool_size: Number of connections in the pool
            max_overflow: Maximum overflow connections
            
        Returns:
            Engine: SQLAlchemy engine instance
            
        Raises:
            DatabaseConnectionError: If engine creation fails
        """
        try:
            database_url = get_database_url()
            
            self.sqlalchemy_engine = create_engine(
                database_url,
                poolclass=QueuePool,
                pool_size=pool_size,
                max_overflow=max_overflow,
                pool_pre_ping=True,
                pool_recycle=1800,  # 30 minutes
                pool_timeout=30,
                echo=self.settings.debug,
                # Ensure explicit transaction control
                isolation_level="AUTOCOMMIT" if self.settings.database.autocommit else "READ_COMMITTED"
            )
            
            logger.info(f"SQLAlchemy engine created successfully (pool_size={pool_size})")
            return self.sqlalchemy_engine
            
        except Exception as e:
            logger.error(f"Failed to create SQLAlchemy engine: {e}")
            raise DatabaseConnectionError(f"Failed to create SQLAlchemy engine: {e}")
    
    @contextmanager
    def get_connection(self):
        """
        Get a database connection from SQLAlchemy engine.
        
        Yields:
            SQLAlchemy connection
            
        Raises:
            DatabaseConnectionError: If connection fails
        """
        connection = None
        try:
            if not self.sqlalchemy_engine:
                self.create_sqlalchemy_engine()
            
            connection = self.sqlalchemy_engine.connect()
            yield connection
            
        except SQLAlchemyError as e:
            logger.error(f"Database connection error: {e}")
            raise DatabaseConnectionError(f"Database connection failed: {e}")
        finally:
            if connection:
                connection.close()
    
    def test_connection(self) -> bool:
        """
        Test database connection.
        
        Returns:
            bool: True if connection is successful
        """
        try:
            with self.get_connection() as connection:
                result = connection.execute(text("SELECT 1"))
                return result.fetchone()[0] == 1
                
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False
    
    def close_connections(self):
        """Close all database connections."""
        try:
            if self.sqlalchemy_engine:
                self.sqlalchemy_engine.dispose()
                self.sqlalchemy_engine = None
                
            logger.info("Database connections closed")
            
        except Exception as e:
            logger.error(f"Error closing database connections: {e}")
    
    def _handle_database_error(self, error: Exception, query: str = "") -> None:
        """
        Handle database errors with consistent logging and monitoring.
        
        Args:
            error: The database error that occurred
            query: The query that caused the error (optional)
        """
        error_msg = str(error)
        logger.error(f"Database error: {error_msg}")
        
        # Record error in monitoring system
        try:
            severity = "high" if "timeout" in error_msg.lower() or "connection" in error_msg.lower() else "medium"
            record_database_error(f"Database error: {error_msg}", severity)
        except Exception as monitoring_error:
            logger.error(f"Failed to record database error in monitoring: {str(monitoring_error)}")


class DatabaseManager:
    """
    Database manager for MariaDB connections.
    Provides unified API for both read-only and read-write operations.
    """
    
    def __init__(self):
        """Initialize database manager with connections."""
        self._read_only_connection: Optional['ReadOnlyDatabaseConnection'] = None
        self._read_write_connection: Optional['ReadWriteDatabaseConnection'] = None
        self._initialize_connections()
    
    def _initialize_connections(self):
        """Initialize database connections."""
        try:
            self._read_only_connection = ReadOnlyDatabaseConnection()
            self._read_write_connection = ReadWriteDatabaseConnection()
            
            # SQLAlchemy 엔진 사전 생성 (성능 최적화)
            self._preload_engines()
            
            logger.info("DatabaseManager initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize DatabaseManager: {e}")
            raise DatabaseConnectionError(f"DatabaseManager initialization failed: {e}")
    
    def _preload_engines(self):
        """사전에 SQLAlchemy 엔진을 생성하여 첫 번째 요청 시 지연을 방지"""
        try:
            # 읽기 전용 엔진 사전 생성 (최적화: 20개 연결)
            if self._read_only_connection:
                self._read_only_connection.create_sqlalchemy_engine(pool_size=20, max_overflow=20)
                logger.info("Read-only SQLAlchemy engine preloaded with 20 connections")
            
            # 읽기/쓰기 엔진 사전 생성 (최적화: 10개 연결)
            if self._read_write_connection:
                self._read_write_connection.create_sqlalchemy_engine(pool_size=10, max_overflow=10)
                logger.info("Read-write SQLAlchemy engine preloaded with 10 connections")
                
        except Exception as e:
            logger.warning(f"Failed to preload SQLAlchemy engines: {e}")
            # 엔진 사전 생성 실패는 치명적이지 않으므로 경고만 출력
    
    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None, readonly: bool = True) -> Union[int, List[Dict[str, Any]]]:
        """
        Execute a database query using the appropriate connection type.
        
        Args:
            query: SQL query string
            params: Query parameters
            readonly: If True, uses read-only connection; if False, uses read-write connection
            
        Returns:
            Union[int, List[Dict[str, Any]]]: For SELECT queries returns results, for others returns affected rows
            
        Raises:
            DatabaseQueryError: If query execution fails
        """
        if readonly:
            if not self._read_only_connection:
                self._read_only_connection = ReadOnlyDatabaseConnection()
            return self._read_only_connection.execute_query(query, params)
        else:
            if not self._read_write_connection:
                self._read_write_connection = ReadWriteDatabaseConnection()
            return self._read_write_connection.execute_query(query, params)
    
    @contextmanager
    def get_connection(self, readonly: bool = True):
        """
        Get a database connection.
        
        Args:
            readonly: If True, returns read-only connection; if False, returns read-write connection
            
        Yields:
            Database connection instance
        """
        if readonly:
            if not self._read_only_connection:
                self._read_only_connection = ReadOnlyDatabaseConnection()
            yield self._read_only_connection.get_connection()
        else:
            if not self._read_write_connection:
                self._read_write_connection = ReadWriteDatabaseConnection()
            yield self._read_write_connection.get_connection()
    
    def test_connections(self) -> Dict[str, bool]:
        """
        Test both read-only and read-write connections.
        
        Returns:
            Dict with connection test results
        """
        results = {}
        
        try:
            if self._read_only_connection:
                results['read_only'] = self._read_only_connection.test_connection()
            else:
                results['read_only'] = False
        except Exception as e:
            logger.error(f"Read-only connection test failed: {e}")
            results['read_only'] = False
        
        try:
            if self._read_write_connection:
                results['read_write'] = self._read_write_connection.test_connection()
            else:
                results['read_write'] = False
        except Exception as e:
            logger.error(f"Read-write connection test failed: {e}")
            results['read_write'] = False
        
        return results
    
    def close_all_connections(self):
        """Close all database connections."""
        if self._read_only_connection:
            self._read_only_connection.close_connections()
        if self._read_write_connection:
            self._read_write_connection.close_connections()
        logger.info("All database connections closed")


class ReadWriteDatabaseConnection(BaseDatabaseConnection):
    """
    Read-write database connection manager for MariaDB.
    Used for database initialization, migrations, and schema operations.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize read-write database connection.
        
        Args:
            config: Database configuration dictionary
        """
        super().__init__(config)
    
    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> Union[int, List[Dict[str, Any]]]:
        """
        Execute a read-write SQL query.
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            Union[int, List[Dict[str, Any]]]: For SELECT queries returns results, for others returns affected rows
            
        Raises:
            DatabaseQueryError: If query execution fails
            DatabaseConnectionError: If connection fails
        """
        try:
            with self.get_connection() as connection:
                result = connection.execute(text(query), params or {})
                
                if query.strip().upper().startswith('SELECT'):
                    return [dict(row._mapping) for row in result]
                else:
                    # For non-SELECT queries, return affected rows
                    return result.rowcount
                
        except SQLAlchemyError as e:
            self._handle_database_error(e, query)
            raise DatabaseQueryError(f"Read-write query execution failed: {e}")


class ReadOnlyDatabaseConnection(BaseDatabaseConnection):
    """
    Read-only database connection manager for MariaDB.
    Ensures all operations are read-only for security.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize database connection.
        
        Args:
            config: Database configuration dictionary
        """
        super().__init__(config)
    
    
    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Execute a read-only SQL query.
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            List[Dict[str, Any]]: Query results as list of dictionaries
            
        Raises:
            DatabaseQueryError: If query execution fails
            DatabaseConnectionError: If connection fails
        """
        # Validate query is read-only
        if not self._is_read_only_query(query):
            raise DatabaseQueryError("Only SELECT queries are allowed")
        
        try:
            with self.get_connection() as connection:
                result = connection.execute(text(query), params or {})
                return [dict(row._mapping) for row in result]
                
        except SQLAlchemyError as e:
            self._handle_database_error(e, query)
            raise DatabaseQueryError(f"Query execution failed: {e}")
    
    def _is_read_only_query(self, query: str) -> bool:
        """
        Check if query is read-only.
        
        Args:
            query: SQL query string
            
        Returns:
            bool: True if query is read-only
        """
        query_upper = query.strip().upper()
        
        # 1. SELECT로 시작하는지 확인 (주석이나 공백 제거 후)
        query_clean = query_upper.lstrip()
        if not query_clean.startswith('SELECT'):
            return False
        
        # 2. 위험한 키워드가 포함되어 있는지 확인 (단어 경계 사용)
        dangerous_keywords = [
            'INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER',
            'TRUNCATE', 'REPLACE', 'GRANT', 'REVOKE', 'LOCK', 'UNLOCK'
        ]
        
        # 단어 경계를 사용하여 정확한 키워드만 검사
        for keyword in dangerous_keywords:
            pattern = r'\b' + keyword + r'\b'
            if re.search(pattern, query_upper):
                return False
        
        return True
    
    def get_table_schema(self, table_name: str) -> List[Dict[str, Any]]:
        """
        Get table schema information.
        
        Args:
            table_name: Name of the table
            
        Returns:
            List[Dict[str, Any]]: Table schema information
            
        Raises:
            DatabaseQueryError: If schema query fails
        """
        query = """
        SELECT 
            COLUMN_NAME,
            DATA_TYPE,
            IS_NULLABLE,
            COLUMN_DEFAULT,
            COLUMN_KEY,
            EXTRA,
            COLUMN_COMMENT
        FROM INFORMATION_SCHEMA.COLUMNS 
        WHERE TABLE_SCHEMA = :database AND TABLE_NAME = :table_name
        ORDER BY ORDINAL_POSITION
        """
        
        return self.execute_query(query, {'database': self.config['database'], 'table_name': table_name})
    
    def get_database_tables(self) -> List[str]:
        """
        Get list of tables in the database.
        
        Returns:
            List[str]: List of table names
            
        Raises:
            DatabaseQueryError: If table list query fails
        """
        query = """
        SELECT TABLE_NAME 
        FROM INFORMATION_SCHEMA.TABLES 
        WHERE TABLE_SCHEMA = :database
        ORDER BY TABLE_NAME
        """
        
        results = self.execute_query(query, {'database': self.config['database']})
        return [row['TABLE_NAME'] for row in results]


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


def get_db_manager() -> DatabaseManager:
    """
    Get the global database manager instance.
    Creates a new instance if one doesn't exist.
    
    Returns:
        DatabaseManager: Database manager instance
    """
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


def close_db_manager():
    """Close the global database manager."""
    global _db_manager
    if _db_manager:
        _db_manager.close_all_connections()
        _db_manager = None


def test_database_connection() -> bool:
    """
    Test the database connection.
    
    Returns:
        bool: True if connection is successful
    """
    try:
        db_manager = get_db_manager()
        results = db_manager.test_connections()
        return results.get('read_only', False)
    except Exception as e:
        logger.error(f"Database connection test failed: {e}")
        return False


def execute_query(query: str, params: Optional[Dict[str, Any]] = None, readonly: bool = True) -> Union[int, List[Dict[str, Any]]]:
    """
    Execute a database query using the appropriate connection type.
    
    Args:
        query: SQL query string
        params: Query parameters
        readonly: If True, uses read-only connection; if False, uses read-write connection
        
    Returns:
        Union[int, List[Dict[str, Any]]]: For SELECT queries returns results, for others returns affected rows
        
    Raises:
        DatabaseQueryError: If query execution fails
    """
    db_manager = get_db_manager()
    return db_manager.execute_query(query, params, readonly)


def initialize_database() -> bool:
    """
    Initialize database with required tables and indexes.
    Uses transaction to ensure atomicity - all operations succeed or all fail.
    
    Returns:
        bool: True if initialization was successful
    """
    try:
        logger.info("Starting database initialization...")
        
        # Check if we have read-write access
        db_manager = get_db_manager()
        connection_test = db_manager.test_connections()
        
        if not connection_test.get('read_write', False):
            logger.warning("No read-write database access available. Skipping initialization.")
            return False
        
        # Create query history table
        create_query_history_table = """
        CREATE TABLE IF NOT EXISTS query_history (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id VARCHAR(50) NOT NULL,
            channel_id VARCHAR(50),
            natural_query TEXT NOT NULL,
            sql_query TEXT NOT NULL,
            success BOOLEAN DEFAULT TRUE,
            error_message TEXT,
            processing_time_ms INT,
            confidence_score DECIMAL(3,2),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_user_id (user_id),
            INDEX idx_channel_id (channel_id),
            INDEX idx_created_at (created_at),
            INDEX idx_success (success)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        """
        
        # Create pipeline metrics table
        create_metrics_table = """
        CREATE TABLE IF NOT EXISTS pipeline_metrics (
            id INT AUTO_INCREMENT PRIMARY KEY,
            session_id VARCHAR(100) NOT NULL,
            user_id VARCHAR(50),
            channel_id VARCHAR(50),
            component_name VARCHAR(50) NOT NULL,
            processing_time_ms INT NOT NULL,
            success BOOLEAN DEFAULT TRUE,
            confidence_score DECIMAL(3,2),
            error_message TEXT,
            metadata JSON,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_session_id (session_id),
            INDEX idx_user_id (user_id),
            INDEX idx_component_name (component_name),
            INDEX idx_created_at (created_at)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        """
        
        # Create database version tracking table
        create_version_table = """
        CREATE TABLE IF NOT EXISTS database_version (
            id INT AUTO_INCREMENT PRIMARY KEY,
            version VARCHAR(20) NOT NULL UNIQUE,
            description TEXT,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_version (version)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        """
        
        # Use transaction to ensure atomicity
        with db_manager.get_connection(readonly=False) as connection:
            # Start transaction
            trans = connection.begin()
            try:
                # Execute all table creation queries in single transaction
                connection.execute(text(create_query_history_table))
                connection.execute(text(create_metrics_table))
                connection.execute(text(create_version_table))
                
                # Check if version record exists
                check_version_query = "SELECT COUNT(*) as count FROM database_version WHERE version = :version"
                result = connection.execute(text(check_version_query), {'version': '1.0.0'})
                version_count = result.scalar()
                
                # Insert version record if it doesn't exist
                if version_count == 0:
                    insert_version_query = """
                    INSERT INTO database_version (version, description) 
                    VALUES (:version, :description)
                    """
                    connection.execute(text(insert_version_query), {
                        'version': '1.0.0', 
                        'description': 'Initial database schema with query_history, pipeline_metrics, and database_version tables'
                    })
                
                # Commit transaction
                trans.commit()
                logger.info("Database initialization completed successfully")
                return True
                
            except Exception as e:
                # Rollback transaction on error
                trans.rollback()
                logger.error(f"Database initialization failed, transaction rolled back: {e}")
                raise
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        return False


def get_database_schema_info() -> Dict[str, Any]:
    """
    Get comprehensive database schema information.
    
    Returns:
        Dict containing schema information
    """
    try:
        db_manager = get_db_manager()
        read_only_connection = db_manager._read_only_connection
        
        if not read_only_connection:
            raise DatabaseConnectionError("No read-only connection available")
        
        # Get all tables
        tables = read_only_connection.get_database_tables()
        
        schema_info = {
            'database_name': read_only_connection.config['database'],
            'tables': {},
            'table_count': len(tables),
            'timestamp': datetime.now().isoformat()
        }
        
        # Get detailed info for each table
        for table_name in tables:
            try:
                table_schema = read_only_connection.get_table_schema(table_name)
                schema_info['tables'][table_name] = {
                    'columns': table_schema,
                    'column_count': len(table_schema)
                }
            except Exception as e:
                logger.warning(f"Failed to get schema for table {table_name}: {e}")
                schema_info['tables'][table_name] = {
                    'error': str(e),
                    'column_count': 0
                }
        
        return schema_info
        
    except Exception as e:
        logger.error(f"Failed to get database schema info: {e}")
        return {'error': str(e)}


def cleanup_old_records(days_to_keep: int = 30) -> Dict[str, int]:
    """
    Clean up old records from database tables.
    Uses transaction to ensure atomicity.
    
    Args:
        days_to_keep: Number of days of records to keep
        
    Returns:
        Dict with cleanup statistics
    """
    try:
        db_manager = get_db_manager()
        
        if not db_manager._read_write_connection:
            logger.warning("No read-write connection available for cleanup")
            return {'error': 'No read-write access'}
        
        # Use transaction to ensure atomicity
        with db_manager.get_connection(readonly=False) as connection:
            trans = connection.begin()
            try:
                cleanup_stats = {}
                
                # Clean up old query history
                query_history_cleanup = """
                DELETE FROM query_history 
                WHERE created_at < DATE_SUB(NOW(), INTERVAL :days_to_keep DAY)
                """
                result1 = connection.execute(text(query_history_cleanup), {'days_to_keep': days_to_keep})
                cleanup_stats['deleted_query_records'] = result1.rowcount
                
                # Clean up old metrics
                metrics_cleanup = """
                DELETE FROM pipeline_metrics 
                WHERE created_at < DATE_SUB(NOW(), INTERVAL :days_to_keep DAY)
                """
                result2 = connection.execute(text(metrics_cleanup), {'days_to_keep': days_to_keep})
                cleanup_stats['deleted_metric_records'] = result2.rowcount
                
                # Commit transaction
                trans.commit()
                logger.info(f"Database cleanup completed: {cleanup_stats}")
                return cleanup_stats
                
            except Exception as e:
                # Rollback transaction on error
                trans.rollback()
                logger.error(f"Database cleanup failed, transaction rolled back: {e}")
                raise
        
    except Exception as e:
        logger.error(f"Database cleanup failed: {e}")
        return {'error': str(e)}


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """Get a database session for SQLAlchemy operations."""
    db_manager = get_db_manager()
    read_only_connection = db_manager._read_only_connection
    
    if not read_only_connection.sqlalchemy_engine:
        read_only_connection.create_sqlalchemy_engine()
    
    SessionLocal = sessionmaker(bind=read_only_connection.sqlalchemy_engine)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


# Database Schema Caching Functions
@lru_cache(maxsize=1)
def get_cached_db_schema() -> Dict[str, Any]:
    """
    Load database schema and cache the result.
    
    This function uses LRU cache to ensure the schema is loaded only once
    and shared across all nodes that need it.
    
    Returns:
        Dict[str, Any]: Database schema with table and column information
    """
    try:
        settings = get_settings()
        database_url = get_database_url()
        engine = create_engine(database_url)
        inspector = inspect(engine)
        
        db_schema = {}
        
        # 테이블 로드 ('t_'로 시작하는 테이블만)
        table_names = [t for t in inspector.get_table_names() if t.startswith("t_")]
        
        # 뷰 로드 ('v_'로 시작하는 뷰만)
        try:
            view_names = [v for v in inspector.get_view_names() if v.startswith("v_")]
        except Exception as e:
            # get_view_names()가 지원되지 않는 경우 빈 리스트
            logger.debug(f"get_view_names() not supported or failed: {e}")
            view_names = []
        
        # 테이블과 뷰 모두 처리
        all_names = table_names + view_names
        
        for object_name in all_names:
            is_view = object_name.startswith("v_")
            
            # 테이블/뷰 코멘트 가져오기 (제약조건 정보 필터링)
            try:
                if is_view:
                    # 뷰는 코멘트가 없을 수 있으므로 기본 설명 사용
                    object_description = f"{object_name} view"
                else:
                    table_comment = inspector.get_table_comment(object_name)
                    if table_comment and table_comment.get("text"):
                        comment_text = table_comment.get("text")
                        # 제약조건 정보가 포함된 경우 순수한 설명만 추출
                        if "CONSTRAINT" in comment_text or "FOREIGN KEY" in comment_text:
                            # 제약조건 정보 제거하고 순수한 설명만 사용
                            lines = comment_text.split('\n')
                            clean_lines = []
                            for line in lines:
                                line = line.strip()
                                if not (line.startswith('CONSTRAINT') or 
                                       line.startswith('FOREIGN KEY') or 
                                       line.startswith('REFERENCES') or
                                       line.startswith('PRIMARY KEY')):
                                    clean_lines.append(line)
                            object_description = ' '.join(clean_lines).strip() or f"{object_name} table"
                        else:
                            object_description = comment_text
                    else:
                        object_description = f"{object_name} table"
            except Exception as e:
                # 스키마 파싱 오류 시 기본값 사용
                logger.debug(f"{'뷰' if is_view else '테이블'} 코멘트 파싱 실패 ({object_name}): {e}")
                object_description = f"{object_name} {'view' if is_view else 'table'}"
            
            # 컬럼 정보 수집
            columns = {}
            try:
                for column in inspector.get_columns(object_name):
                    columns[column['name']] = {
                        "type": str(column['type']),
                        "description": column.get('comment', ''),
                        "nullable": column.get('nullable', True),
                        "default": column.get('default')
                    }
            except Exception as e:
                logger.warning(f"Failed to get columns for {object_name}: {e}")
                columns = {}
            
            db_schema[object_name] = {
                "description": object_description,
                "columns": columns,
                "type": "view" if is_view else "table"
            }
        
        table_count = len(table_names)
        view_count = len(view_names)
        logger.info(f"Database schema loaded and cached: {table_count} tables, {view_count} views (total: {len(db_schema)} objects)")
        return db_schema
        
    except OperationalError as e:
        logger.error(f"Database connection error during schema loading: {e}")
        return _get_fallback_schema()
    except ProgrammingError as e:
        logger.error(f"Database permission error during schema loading: {e}")
        return _get_fallback_schema()
    except SQLAlchemyError as e:
        logger.error(f"Database query error during schema loading: {e}")
        return _get_fallback_schema()
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading schema: {e}")
        return _get_fallback_schema()


def _get_fallback_schema() -> Dict[str, Any]:
    """
    Fallback schema when dynamic loading fails.
    
    Returns:
        Dict[str, Any]: Basic fallback schema
    """
    return {
        "t_member": {
            "description": "회원 정보 테이블 (fallback)",
            "columns": {
                "id": {"type": "int", "description": "회원 ID"},
                "email": {"type": "varchar", "description": "이메일 주소"},
                "nickname": {"type": "varchar", "description": "닉네임"},
                "status": {"type": "varchar", "description": "회원 상태"},
                "created_at": {"type": "timestamp", "description": "가입일"}
            }
        },
        "t_creator": {
            "description": "크리에이터 정보 테이블 (fallback)",
            "columns": {
                "id": {"type": "int", "description": "크리에이터 ID"},
                "nickname": {"type": "varchar", "description": "크리에이터 닉네임"},
                "description": {"type": "text", "description": "크리에이터 소개"},
                "category": {"type": "varchar", "description": "카테고리"}
            }
        },
        "t_funding": {
            "description": "펀딩 프로젝트 테이블 (fallback)",
            "columns": {
                "id": {"type": "int", "description": "프로젝트 ID"},
                "title": {"type": "varchar", "description": "프로젝트 제목"},
                "goal_amount": {"type": "int", "description": "목표 금액"},
                "current_amount": {"type": "int", "description": "현재 모금액"},
                "status": {"type": "varchar", "description": "프로젝트 상태"},
                "created_at": {"type": "timestamp", "description": "생성일"}
            }
        }
    }


def clear_schema_cache():
    """
    Clear the cached database schema.
    
    This function should be called when the database schema changes
    and the cache needs to be refreshed.
    """
    get_cached_db_schema.cache_clear()
    logger.info("Database schema cache cleared")


def get_schema_info() -> Dict[str, Any]:
    """
    Get schema information and statistics.
    
    Returns:
        Dict[str, Any]: Schema information including table count, column count, etc.
    """
    schema = get_cached_db_schema()
    
    total_columns = sum(len(table.get('columns', {})) for table in schema.values())
    
    return {
        "table_count": len(schema),
        "total_columns": total_columns,
        "table_names": list(schema.keys()),
        "schema_source": "cached" if get_cached_db_schema.cache_info().hits > 0 else "fresh"
    }


# SQL Parsing Functions
class SQLParser:
    """Enhanced SQL parser using sqlparse library for robust SQL analysis."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def extract_table_names(self, sql_query: str) -> List[str]:
        """
        Extract table names from SQL query using sqlparse.
        
        This method is more robust than regex-based approaches and can handle:
        - Subqueries
        - CTE (Common Table Expressions)
        - Complex aliases
        - Nested queries
        
        Args:
            sql_query: SQL query string
            
        Returns:
            List[str]: List of table names found in the query
        """
        try:
            parsed = sqlparse.parse(sql_query)
            if not parsed:
                return []
            
            tables = set()
            
            for statement in parsed:
                tables.update(self._extract_tables_from_statement(statement))
            
            return list(tables)
            
        except Exception as e:
            self.logger.error(f"Error parsing SQL for table names: {e}")
            # Fallback to regex-based approach
            return self._extract_table_names_fallback(sql_query)
    
    def _extract_tables_from_statement(self, statement) -> set:
        """Extract table names from a single SQL statement."""
        tables = set()
        
        # Handle different types of tokens
        for token in statement.flatten():
            if token.ttype is Keyword and token.value.upper() in ['FROM', 'JOIN', 'UPDATE', 'INSERT']:
                # Get the next token which should be the table name
                next_token = self._get_next_meaningful_token(statement, token)
                if next_token:
                    table_name = self._extract_table_from_token(next_token)
                    if table_name:
                        tables.add(table_name)
        
        return tables
    
    def _get_next_meaningful_token(self, statement, current_token):
        """Get the next meaningful token after the current one."""
        try:
            # Find the position of current token
            for i, token in enumerate(statement.tokens):
                if token == current_token and i + 1 < len(statement.tokens):
                    return statement.tokens[i + 1]
        except Exception:
            pass
        return None
    
    def _extract_table_from_token(self, token):
        """Extract table name from a token."""
        if isinstance(token, Identifier):
            return token.get_real_name()
        elif isinstance(token, IdentifierList):
            # Handle multiple tables in JOIN
            tables = []
            for identifier in token.get_identifiers():
                table_name = identifier.get_real_name()
                if table_name:
                    tables.append(table_name)
            return tables[0] if tables else None
        elif hasattr(token, 'value') and isinstance(token.value, str):
            # Handle simple string tokens
            return token.value.strip()
        
        return None
    
    def _extract_table_names_fallback(self, sql_query: str) -> List[str]:
        """Fallback regex-based table name extraction."""
        table_names = []
        
        # FROM 절에서 테이블명 추출
        from_pattern = r'FROM\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        matches = re.findall(from_pattern, sql_query, re.IGNORECASE)
        table_names.extend(matches)
        
        # JOIN 절에서 테이블명 추출
        join_pattern = r'JOIN\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        matches = re.findall(join_pattern, sql_query, re.IGNORECASE)
        table_names.extend(matches)
        
        return list(set(table_names))
    
    def extract_column_names(self, sql_query: str) -> List[str]:
        """
        Extract column names from SQL query.
        
        Args:
            sql_query: SQL query string
            
        Returns:
            List[str]: List of column names found in the query
        """
        try:
            parsed = sqlparse.parse(sql_query)
            if not parsed:
                return []
            
            columns = set()
            
            for statement in parsed:
                columns.update(self._extract_columns_from_statement(statement))
            
            return list(columns)
            
        except Exception as e:
            self.logger.error(f"Error parsing SQL for column names: {e}")
            return self._extract_column_names_fallback(sql_query)
    
    def _extract_columns_from_statement(self, statement) -> set:
        """Extract column names from a single SQL statement."""
        columns = set()
        
        for token in statement.flatten():
            if token.ttype is Name:
                # This is likely a column name
                column_name = token.value.strip()
                if column_name and not column_name.upper() in ['SELECT', 'FROM', 'WHERE', 'GROUP', 'ORDER', 'HAVING']:
                    columns.add(column_name)
        
        return columns
    
    def _extract_column_names_fallback(self, sql_query: str) -> List[str]:
        """Fallback regex-based column name extraction."""
        column_names = []
        
        # SELECT 절에서 컬럼명 추출
        select_pattern = r'SELECT\s+(.*?)\s+FROM'
        select_match = re.search(select_pattern, sql_query, re.IGNORECASE | re.DOTALL)
        
        if select_match:
            select_clause = select_match.group(1)
            # 간단한 컬럼명 추출 (복잡한 쿼리는 고려하지 않음)
            column_pattern = r'([a-zA-Z_][a-zA-Z0-9_]*)'
            matches = re.findall(column_pattern, select_clause)
            column_names.extend(matches)
        
        return list(set(column_names))
    
    def validate_sql_syntax(self, sql_query: str) -> Dict[str, Any]:
        """
        Validate SQL syntax using sqlparse.
        
        Args:
            sql_query: SQL query string
            
        Returns:
            Dict[str, Any]: Validation result with success status and details
        """
        try:
            parsed = sqlparse.parse(sql_query)
            
            if not parsed:
                return {
                    "is_valid": False,
                    "error": "Empty query",
                    "details": "No SQL statement found"
                }
            
            # Check if it's a SELECT statement
            first_statement = parsed[0]
            if not self._is_select_statement(first_statement):
                return {
                    "is_valid": False,
                    "error": "Invalid SQL statement",
                    "details": "Query must be a SELECT statement"
                }
            
            return {
                "is_valid": True,
                "details": "Syntax validation passed"
            }
            
        except Exception as e:
            return {
                "is_valid": False,
                "error": "Syntax error",
                "details": str(e)
            }
    
    def _is_select_statement(self, statement) -> bool:
        """Check if the statement is a SELECT statement."""
        for token in statement.tokens:
            if token.ttype is DML and token.value.upper() == 'SELECT':
                return True
        return False


# Global instance for easy access
sql_parser = SQLParser()


def extract_table_names(sql_query: str) -> List[str]:
    """Convenience function to extract table names from SQL."""
    return sql_parser.extract_table_names(sql_query)


def extract_column_names(sql_query: str) -> List[str]:
    """Convenience function to extract column names from SQL."""
    return sql_parser.extract_column_names(sql_query)


def validate_sql_syntax(sql_query: str) -> Dict[str, Any]:
    """Convenience function to validate SQL syntax."""
    return sql_parser.validate_sql_syntax(sql_query)