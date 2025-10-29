"""
Core configuration module for DataTalk bot.
Handles environment variables, API keys, and application settings using Pydantic.
"""

from typing import Optional, Any, List
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from enum import Enum
import re
import os
from functools import lru_cache
from dotenv import load_dotenv


def get_project_root() -> str:
    """Get the project root directory."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level from src/core to src, then up to project root
    project_root = os.path.dirname(os.path.dirname(current_dir))
    return project_root


def get_env_file_path() -> str:
    """Get the .env file path relative to project root."""
    return os.path.join(get_project_root(), ".env")


def load_env_variables():
    """Load environment variables from .env file."""
    env_file_path = get_env_file_path()
    if os.path.exists(env_file_path):
        load_dotenv(env_file_path)
        return True
    return False


class Environment(str, Enum):
    """Application environment types."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"


class DatabaseConfig(BaseSettings):
    """Database configuration settings."""
    model_config = SettingsConfigDict(
        env_prefix="DB_",
        case_sensitive=False,
        env_file=get_env_file_path(),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    host: str = Field(default="epic-readonly.ceind7azkfjy.ap-northeast-2.rds.amazonaws.com")
    port: int = Field(default=3306)
    username: str = Field(default="readonly_user_business_data")
    password: str = Field(...)  # 필수 필드로 변경
    database: str = Field(default="fanding")
    charset: str = Field(default="utf8mb4")
    autocommit: bool = Field(default=True)
    
    @field_validator('port')
    @classmethod
    def validate_port(cls, v):
        """Validate database port."""
        if not (1 <= v <= 65535):
            raise ValueError("포트 번호는 1-65535 범위여야 합니다")
        return v
    
    @field_validator('host')
    @classmethod
    def validate_host(cls, v):
        """Validate database host."""
        if not v or len(v.strip()) == 0:
            raise ValueError("데이터베이스 호스트가 비어있습니다")
        return v.strip()
    
    @field_validator('database')
    @classmethod
    def validate_database_name(cls, v):
        """Validate database name."""
        if not re.match(r'^[a-zA-Z0-9_]+$', v):
            raise ValueError("데이터베이스 이름은 영문자, 숫자, 언더스코어만 허용됩니다")
        return v
    



class SlackConfig(BaseSettings):
    """Slack application configuration."""
    bot_token: str = Field(...)  # 필수 필드로 변경
    app_token: str = Field(...)  # 필수 필드로 변경
    signing_secret: str = Field(...)  # 필수 필드로 변경
    
    model_config = SettingsConfigDict(
        env_prefix="SLACK_",
        case_sensitive=False,
        env_file=get_env_file_path(),
        env_file_encoding="utf-8",
        extra="ignore"
    )


class LLMConfig(BaseSettings):
    """LLM (Large Language Model) configuration."""
    model_config = SettingsConfigDict(
        env_prefix="LLM_",
        case_sensitive=False,
        env_file=get_env_file_path(),
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    provider: str = Field(default="google")
    model: str = Field(default="gemini-2.5-pro")
    api_key: str = Field(default="")
    temperature: float = Field(default=0.1)
    max_tokens: int = Field(default=2048)
    
    # Intent classification specific settings
    intent_model: str = Field(default="gemini-2.5-flash")
    intent_temperature: float = Field(default=0.1)
    intent_max_tokens: int = Field(default=256)
    
    def __init__(self, **kwargs):
        # Load environment variables before initialization
        load_env_variables()
        # Get GOOGLE_API_KEY from environment
        if 'api_key' not in kwargs:
            kwargs['api_key'] = os.getenv('GOOGLE_API_KEY', '')
        super().__init__(**kwargs)
    
    @field_validator('provider')
    @classmethod
    def validate_provider(cls, v):
        """Validate LLM provider."""
        valid_providers = ['google', 'openai', 'anthropic', 'azure']
        if v.lower() not in valid_providers:
            raise ValueError(f"지원되는 LLM 제공업체: {', '.join(valid_providers)}")
        return v.lower()
    
    @field_validator('temperature')
    @classmethod
    def validate_temperature(cls, v):
        """Validate temperature value."""
        if not (0.0 <= v <= 2.0):
            raise ValueError("Temperature는 0.0-2.0 범위여야 합니다")
        return v
    
    @field_validator('max_tokens')
    @classmethod
    def validate_max_tokens(cls, v):
        """Validate max tokens value."""
        if not (1 <= v <= 100000):
            raise ValueError("Max tokens는 1-100000 범위여야 합니다")
        return v
    
    @field_validator('api_key')
    @classmethod
    def validate_api_key(cls, v):
        """Validate API key format."""
        if v and len(v.strip()) < 10:
            raise ValueError("API 키가 너무 짧습니다. 최소 10자 이상이어야 합니다")
        return v.strip() if v else v


class LoggingConfig(BaseSettings):
    """Logging configuration."""
    level: str = Field(default="INFO")
    format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_path: Optional[str] = Field(default=None)
    
    model_config = SettingsConfigDict(env_prefix="LOG_")


class PipelineConfig(BaseSettings):
    """Pipeline configuration settings."""
    max_retries: int = Field(default=3)
    confidence_threshold: float = Field(default=0.7)
    enable_debug: bool = Field(default=False)
    enable_monitoring: bool = Field(default=True)
    max_history: int = Field(default=1000)
    dangerous_sql_keywords: List[str] = Field(default=[
        'INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 
        'ALTER', 'TRUNCATE', 'EXEC', 'EXECUTE', 'UNION',
        'SCRIPT', 'GRANT', 'REVOKE', 'COMMIT', 'ROLLBACK'
    ])
    
    @field_validator('dangerous_sql_keywords')
    @classmethod
    def validate_dangerous_keywords(cls, v):
        """Validate dangerous SQL keywords list."""
        if not isinstance(v, list):
            raise ValueError("dangerous_sql_keywords must be a list")
        if not all(isinstance(keyword, str) for keyword in v):
            raise ValueError("All dangerous SQL keywords must be strings")
        return [keyword.upper() for keyword in v]  # 대문자로 정규화
    
    model_config = SettingsConfigDict(env_prefix="PIPELINE_")


class Settings(BaseSettings):
    """Main application settings."""
    
    # Environment
    environment: Environment = Field(default=Environment.DEVELOPMENT)
    debug: bool = Field(default=False)
    
    # Application
    app_name: str = Field(default="DataTalk Bot")
    version: str = Field(default="1.0.0")
    
    # Server
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    allowed_origins: list[str] = Field(default=["http://localhost:3000", "http://localhost:8000"])
    allowed_hosts: list[str] = Field(default=["localhost", "127.0.0.1"])
    
    # Configuration sections
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    slack: SlackConfig = Field(default_factory=SlackConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    
    @field_validator('environment', mode='before')
    @classmethod
    def validate_environment(cls, v):
        """Validate environment value."""
        if isinstance(v, str):
            return Environment(v.lower())
        return v
    
    @field_validator('debug')
    @classmethod
    def validate_debug(cls, v, info):
        """Set debug based on environment."""
        if info.data and 'environment' in info.data:
            return info.data['environment'] == Environment.DEVELOPMENT
        return v
    
    def get_masked_settings(self) -> dict[str, Any]:
        """
        Get settings with sensitive information masked for logging/debugging.
        
        Returns:
            dict[str, Any]: Settings dictionary with sensitive values masked
        """
        settings_dict = self.model_dump()
        
        # List of sensitive fields to mask
        sensitive_fields = [
            "password", "api_key", "bot_token", "app_token", "signing_secret"
        ]
        
        def mask_value(value: str, visible_chars: int = 4) -> str:
            """Mask sensitive values showing only first and last few characters."""
            if not value or len(value) <= visible_chars * 2:
                return "***"
            return f"{value[:visible_chars]}...{value[-visible_chars:]}"
        
        def mask_dict_recursive(obj: Any) -> Any:
            """Recursively mask sensitive values in nested dictionaries."""
            if isinstance(obj, dict):
                return {
                    key: mask_value(value, 4) if key in sensitive_fields and isinstance(value, str)
                    else mask_dict_recursive(value)
                    for key, value in obj.items()
                }
            elif isinstance(obj, list):
                return [mask_dict_recursive(item) for item in obj]
            else:
                return obj
        
        return mask_dict_recursive(settings_dict)
    
    def get_config_summary(self) -> dict[str, Any]:
        """
        Get a summary of the configuration.
        
        Returns:
            dict[str, Any]: Configuration summary
        """
        return {
            "environment": self.environment.value,
            "debug": self.debug,
            "app_name": self.app_name,
            "version": self.version,
            "host": self.host,
            "port": self.port,
            "database": {
                "host": self.database.host,
                "port": self.database.port,
                "database": self.database.database,
                "charset": self.database.charset,
                "autocommit": self.database.autocommit
            },
            "llm": {
                "provider": self.llm.provider,
                "model": self.llm.model,
                "temperature": self.llm.temperature,
                "max_tokens": self.llm.max_tokens,
                "api_key_configured": bool(self.llm.api_key)
            },
            "logging": {
                "level": self.logging.level,
                "file_path": self.logging.file_path
            },
            "pipeline": {
                "max_retries": self.pipeline.max_retries,
                "confidence_threshold": self.pipeline.confidence_threshold,
                "enable_debug": self.pipeline.enable_debug,
                "enable_monitoring": self.pipeline.enable_monitoring,
                "max_history": self.pipeline.max_history
            }
        }
    
    model_config = SettingsConfigDict(
        env_file=get_env_file_path(),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )


# Global settings instance
_settings: Optional[Settings] = None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Get the global settings instance.
    Creates a new instance if one doesn't exist.
    Uses LRU cache for better performance.
    
    Returns:
        Settings: The global settings instance
    """
    # Load environment variables from .env file
    load_env_variables()
    return Settings()


def reload_settings() -> Settings:
    """
    Reload the global settings instance.
    Useful for testing or when environment variables change.
    Clears the cache and creates a new instance.
    
    Returns:
        Settings: The reloaded settings instance
    """
    get_settings.cache_clear()  # Clear the LRU cache
    return get_settings()


def get_database_url() -> str:
    """
    Get the database connection URL.
    
    Returns:
        str: Database connection URL
    """
    settings = get_settings()
    db = settings.database
    return f"mysql+pymysql://{db.username}:{db.password}@{db.host}:{db.port}/{db.database}?charset={db.charset}"


def is_development() -> bool:
    """Check if running in development environment."""
    return get_settings().environment == Environment.DEVELOPMENT


def is_production() -> bool:
    """Check if running in production environment."""
    return get_settings().environment == Environment.PRODUCTION


def is_testing() -> bool:
    """Check if running in testing environment."""
    return get_settings().environment == Environment.TESTING


def get_environment() -> str:
    """Get current environment name."""
    return get_settings().environment.value

