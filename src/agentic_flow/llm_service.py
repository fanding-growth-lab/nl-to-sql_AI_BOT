"""
LLM Service Module
LLM 인스턴스를 생성하고 관리하는 서비스 모듈
"""

import logging
from typing import Optional
from functools import lru_cache
from pydantic import SecretStr

from langchain_google_genai import ChatGoogleGenerativeAI
from core.config import get_settings

logger = logging.getLogger(__name__)


class LLMService:
    """LLM 서비스 클래스 - 여러 종류의 LLM 인스턴스를 관리"""
    
    def __init__(self):
        """LLM 서비스 초기화"""
        self._intent_llm: Optional[ChatGoogleGenerativeAI] = None
        self._sql_llm: Optional[ChatGoogleGenerativeAI] = None
        self._settings = get_settings()
        self._initialize_llms()
    
    def _initialize_llms(self):
        """모든 LLM 인스턴스 초기화"""
        try:
            # 인텐트 분류용 LLM 초기화 (빠른 모델)
            self._intent_llm = self._create_intent_llm()
            logger.info(f"Intent LLM initialized: {self._settings.llm.intent_model}")
        except Exception as e:
            logger.error(f"Failed to initialize intent LLM: {e}")
            self._intent_llm = None
        
        try:
            # SQL 생성용 LLM 초기화 (고성능 모델)
            self._sql_llm = self._create_sql_llm()
            logger.info(f"SQL LLM initialized: {self._settings.llm.model}")
        except Exception as e:
            logger.error(f"Failed to initialize SQL LLM: {e}")
            self._sql_llm = None
    
    def _create_intent_llm(self) -> Optional[ChatGoogleGenerativeAI]:
        """인텐트 분류용 LLM 생성"""
        try:
            settings = self._settings.llm
            
            if not settings.api_key:
                logger.warning("Google API key not found. Intent LLM will not be available.")
                return None
            
            return ChatGoogleGenerativeAI(
                model=settings.intent_model,
                api_key=SecretStr(settings.api_key) if settings.api_key else None,
                temperature=settings.intent_temperature,
                max_tokens=settings.intent_max_tokens,
                timeout=10.0
            )
        except Exception as e:
            logger.error(f"Error creating intent LLM: {e}")
            return None
    
    def _create_sql_llm(self) -> Optional[ChatGoogleGenerativeAI]:
        """SQL 생성용 LLM 생성"""
        try:
            settings = self._settings.llm
            
            if not settings.api_key:
                logger.warning("Google API key not found. SQL LLM will not be available.")
                return None
            
            return ChatGoogleGenerativeAI(
                model=settings.model,
                api_key=SecretStr(settings.api_key) if settings.api_key else None,
                temperature=settings.temperature,
                max_tokens=settings.max_tokens,
                timeout=15.0
            )
        except Exception as e:
            logger.error(f"Error creating SQL LLM: {e}")
            return None
    
    def get_intent_llm(self) -> Optional[ChatGoogleGenerativeAI]:
        """
        인텐트 분류용 LLM 인스턴스 반환
        
        Returns:
            Optional[ChatGoogleGenerativeAI]: 인텐트 분류용 LLM 인스턴스
        """
        if self._intent_llm is None:
            logger.warning("Intent LLM not available. Attempting to reinitialize...")
            self._intent_llm = self._create_intent_llm()
        
        return self._intent_llm
    
    def get_sql_llm(self) -> Optional[ChatGoogleGenerativeAI]:
        """
        SQL 생성용 LLM 인스턴스 반환
        
        Returns:
            Optional[ChatGoogleGenerativeAI]: SQL 생성용 LLM 인스턴스
        """
        if self._sql_llm is None:
            logger.warning("SQL LLM not available. Attempting to reinitialize...")
            self._sql_llm = self._create_sql_llm()
        
        return self._sql_llm
    
    def reload(self):
        """설정 변경 후 LLM 인스턴스 재로드"""
        logger.info("Reloading LLM instances...")
        self._settings = get_settings()
        self._initialize_llms()


# 싱글톤 인스턴스
_llm_service_instance: Optional[LLMService] = None


@lru_cache(maxsize=1)
def get_llm_service() -> LLMService:
    """
    LLM 서비스 싱글톤 인스턴스 반환
    
    Returns:
        LLMService: LLM 서비스 인스턴스
    """
    global _llm_service_instance
    
    if _llm_service_instance is None:
        _llm_service_instance = LLMService()
    
    return _llm_service_instance


def reload_llm_service():
    """LLM 서비스 재로드 (설정 변경 시 사용)"""
    global _llm_service_instance
    get_llm_service.cache_clear()
    _llm_service_instance = LLMService()
    return _llm_service_instance

