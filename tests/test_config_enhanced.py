"""
Enhanced Configuration Module Tests

This module tests the improved configuration system with validation,
sensitive data protection, and caching mechanisms.
"""

import pytest
import os
import tempfile
from unittest.mock import patch, mock_open
from pathlib import Path

from src.core.config import (
    Settings, DatabaseConfig, LLMConfig, SlackConfig, LoggingConfig, PipelineConfig,
    get_settings, get_masked_settings, reload_settings, validate_required_settings,
    get_config_summary, get_database_url, is_development, is_production, is_testing,
    get_environment, Environment
)


class TestDatabaseConfig:
    """Test database configuration validation."""
    
    def test_valid_database_config(self):
        """Test valid database configuration."""
        config = DatabaseConfig(
            host="localhost",
            port=3306,
            username="test_user",
            password="test_pass",
            database="test_db"
        )
        assert config.host == "localhost"
        assert config.port == 3306
        assert config.database == "test_db"
    
    def test_invalid_port(self):
        """Test invalid port validation."""
        with pytest.raises(ValueError, match="포트 번호는 1-65535 범위여야 합니다"):
            DatabaseConfig(port=70000)
        
        with pytest.raises(ValueError, match="포트 번호는 1-65535 범위여야 합니다"):
            DatabaseConfig(port=0)
    
    def test_empty_host(self):
        """Test empty host validation."""
        with pytest.raises(ValueError, match="데이터베이스 호스트가 비어있습니다"):
            DatabaseConfig(host="")
        
        with pytest.raises(ValueError, match="데이터베이스 호스트가 비어있습니다"):
            DatabaseConfig(host="   ")
    
    def test_invalid_database_name(self):
        """Test invalid database name validation."""
        with pytest.raises(ValueError, match="데이터베이스 이름은 영문자, 숫자, 언더스코어만 허용됩니다"):
            DatabaseConfig(database="test-db")
        
        with pytest.raises(ValueError, match="데이터베이스 이름은 영문자, 숫자, 언더스코어만 허용됩니다"):
            DatabaseConfig(database="test.db")


class TestLLMConfig:
    """Test LLM configuration validation."""
    
    def test_valid_llm_config(self):
        """Test valid LLM configuration."""
        with patch.dict(os.environ, {
            "LLM_PROVIDER": "google",
            "LLM_MODEL": "gemini-2.5-pro",
            "GOOGLE_API_KEY": "sk-1234567890abcdef",
            "LLM_TEMPERATURE": "0.5",
            "LLM_MAX_TOKENS": "1024"
        }):
            config = LLMConfig()
            assert config.provider == "google"
            assert config.temperature == 0.5
            assert config.max_tokens == 1024
    
    def test_invalid_provider(self):
        """Test invalid provider validation."""
        with patch.dict(os.environ, {
            "LLM_PROVIDER": "invalid_provider"
        }):
            with pytest.raises(ValueError, match="지원되는 LLM 제공업체"):
                LLMConfig()
    
    def test_invalid_temperature(self):
        """Test invalid temperature validation."""
        with patch.dict(os.environ, {
            "LLM_TEMPERATURE": "3.0"
        }):
            with pytest.raises(ValueError, match="Temperature는 0.0-2.0 범위여야 합니다"):
                LLMConfig()
        
        with patch.dict(os.environ, {
            "LLM_TEMPERATURE": "-0.1"
        }):
            with pytest.raises(ValueError, match="Temperature는 0.0-2.0 범위여야 합니다"):
                LLMConfig()
    
    def test_invalid_max_tokens(self):
        """Test invalid max_tokens validation."""
        with patch.dict(os.environ, {
            "LLM_MAX_TOKENS": "0"
        }):
            with pytest.raises(ValueError, match="Max tokens는 1-100000 범위여야 합니다"):
                LLMConfig()
        
        with patch.dict(os.environ, {
            "LLM_MAX_TOKENS": "200000"
        }):
            with pytest.raises(ValueError, match="Max tokens는 1-100000 범위여야 합니다"):
                LLMConfig()
    
    def test_short_api_key(self):
        """Test short API key validation."""
        with patch.dict(os.environ, {
            "GOOGLE_API_KEY": "short"
        }):
            with pytest.raises(ValueError, match="API 키가 너무 짧습니다"):
                LLMConfig()


class TestSettingsValidation:
    """Test main Settings class validation."""
    
    def test_environment_validation(self):
        """Test environment validation."""
        with patch.dict(os.environ, {
            "ENVIRONMENT": "production",
            "DB_HOST": "localhost",
            "DB_DATABASE": "test_db"
        }):
            settings = Settings()
            assert settings.environment == Environment.PRODUCTION
    
    def test_debug_auto_setting(self):
        """Test debug auto-setting based on environment."""
        with patch.dict(os.environ, {
            "ENVIRONMENT": "development",
            "DB_HOST": "localhost",
            "DB_DATABASE": "test_db"
        }):
            settings = Settings()
            assert settings.debug is True


class TestSensitiveDataProtection:
    """Test sensitive data masking functionality."""
    
    def test_mask_sensitive_data(self):
        """Test masking of sensitive data."""
        with patch.dict(os.environ, {
            "DB_HOST": "localhost",
            "DB_DATABASE": "test_db",
            "DB_PASSWORD": "secret_password_123",
            "GOOGLE_API_KEY": "test-api-key-masked",
            "SLACK_BOT_TOKEN": "test-bot-token-masked"
        }):
            masked_settings = get_masked_settings()
            
            # Check that sensitive data is masked
            assert masked_settings["database"]["password"] == "secr...d123"
            assert masked_settings["llm"]["api_key"] == "sk-1...cdef"
            
            # Check that non-sensitive data is preserved
            assert masked_settings["database"]["host"] == "localhost"
            assert masked_settings["database"]["database"] == "test_db"
    
    def test_mask_short_values(self):
        """Test masking of very short sensitive values."""
        with patch.dict(os.environ, {
            "DB_HOST": "localhost",
            "DB_DATABASE": "test_db",
            "DB_PASSWORD": "short"
        }):
            masked_settings = get_masked_settings()
            assert masked_settings["database"]["password"] == "***"


class TestCachingMechanism:
    """Test caching functionality."""
    
    def test_settings_caching(self):
        """Test that settings are properly cached."""
        # Clear any existing cache
        get_settings.cache_clear()
        
        with patch.dict(os.environ, {
            "DB_HOST": "localhost",
            "DB_DATABASE": "test_db"
        }):
            settings1 = get_settings()
            settings2 = get_settings()
            
            # Should return the same instance (cached)
            assert settings1 is settings2
    
    def test_reload_settings(self):
        """Test reloading settings clears cache."""
        with patch.dict(os.environ, {
            "DB_HOST": "localhost",
            "DB_DATABASE": "test_db"
        }):
            settings1 = get_settings()
            settings2 = reload_settings()
            
            # Should return different instances
            assert settings1 is not settings2


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_environment_checkers(self):
        """Test environment checker functions."""
        with patch.dict(os.environ, {
            "ENVIRONMENT": "development",
            "DB_HOST": "localhost",
            "DB_DATABASE": "test_db"
        }):
            assert is_development() is True
            assert is_production() is False
            assert is_testing() is False
            assert get_environment() == "development"
    
    def test_database_url_generation(self):
        """Test database URL generation."""
        with patch.dict(os.environ, {
            "DB_HOST": "localhost",
            "DB_PORT": "3306",
            "DB_USERNAME": "user",
            "DB_PASSWORD": "pass",
            "DB_DATABASE": "db",
            "DB_CHARSET": "utf8mb4"
        }):
            url = get_database_url()
            expected = "mysql+pymysql://user:pass@localhost:3306/db?charset=utf8mb4"
            assert url == expected
    
    def test_validate_required_settings(self):
        """Test validation of required settings."""
        with patch.dict(os.environ, {
            "DB_HOST": "",
            "DB_DATABASE": "test_db"
        }):
            errors = validate_required_settings()
            assert "DB_HOST is required" in errors
    
    def test_config_summary(self):
        """Test configuration summary generation."""
        with patch.dict(os.environ, {
            "DB_HOST": "localhost",
            "DB_DATABASE": "test_db",
            "LLM_MODEL": "gemini-2.5-pro"
        }):
            summary = get_config_summary()
            
            assert summary["environment"] == "development"
            assert summary["database"]["host"] == "localhost"
            assert summary["llm"]["api_key_configured"] is False


class TestEnvironmentFileLoading:
    """Test loading configuration from environment files."""
    
    def test_env_file_loading(self):
        """Test loading configuration from .env file."""
        env_content = """
        ENVIRONMENT=testing
        DEBUG=true
        DB_HOST=test-host
        DB_DATABASE=test_db
        LLM_MODEL=test-model
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write(env_content)
            env_file_path = f.name
        
        try:
            with patch.dict(os.environ, {}):
                # Create settings with custom env file
                settings = Settings(_env_file=env_file_path)
                assert settings.environment == Environment.TESTING
                assert settings.debug is True
                assert settings.database.host == "test-host"
        finally:
            os.unlink(env_file_path)


class TestErrorHandling:
    """Test error handling scenarios."""
    
    def test_missing_required_environment_variables(self):
        """Test handling of missing required environment variables."""
        with patch.dict(os.environ, {}, clear=True):
            # Should not raise an error due to default values
            settings = Settings()
            assert settings.database.host is not None
    
    def test_invalid_environment_value(self):
        """Test handling of invalid environment values."""
        with patch.dict(os.environ, {
            "ENVIRONMENT": "invalid_env",
            "DB_HOST": "localhost",
            "DB_DATABASE": "test_db"
        }):
            with pytest.raises(ValueError):
                Settings()


if __name__ == "__main__":
    pytest.main([__file__])
