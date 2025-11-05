"""
LangChain Output Parser 유틸리티 모듈

LLM 응답을 파싱하기 위한 표준화된 유틸리티 함수를 제공합니다.
LangChain의 Output Parser를 활용하여 일관된 파싱 방식을 제공합니다.
"""

import json
import re
from typing import Dict, Any, Optional, List
from langchain.output_parsers.json import SimpleJsonOutputParser
from langchain_core.messages import BaseMessage
from core.logging import get_logger

logger = get_logger(__name__)


def parse_json_response(
    response: BaseMessage,
    parser: Optional[SimpleJsonOutputParser] = None,
    fallback_extract: bool = True
) -> Optional[Dict[str, Any]]:
    """
    LLM 응답에서 JSON을 파싱 (LangChain 표준 방식)
    
    Args:
        response: LLM 응답 메시지 (BaseMessage)
        parser: SimpleJsonOutputParser 인스턴스 (None이면 자동 생성)
        fallback_extract: 파싱 실패 시 텍스트에서 추출 시도 여부
        
    Returns:
        파싱된 JSON 딕셔너리 또는 None
    """
    try:
        # LangChain 표준 파서 사용
        if parser is None:
            parser = SimpleJsonOutputParser()
        
        # SimpleJsonOutputParser는 메시지 객체를 직접 받을 수 있음
        # 하지만 일부 LLM은 content 속성만 반환할 수 있으므로 두 가지 방식 지원
        try:
            # 방법 1: LangChain 파서 직접 사용 (권장)
            parsed = parser.parse(response.content if hasattr(response, 'content') else str(response))
            if isinstance(parsed, dict):
                logger.debug("Successfully parsed JSON using SimpleJsonOutputParser")
                return parsed
        except Exception as parser_err:
            logger.debug(f"SimpleJsonOutputParser failed: {parser_err}, trying fallback")
        
        # 방법 2: content 추출 후 직접 파싱
        response_text = _extract_content_from_response(response)
        if not response_text:
            logger.warning("Empty response content")
            return None
        
        # 마크다운 코드 블록 제거 및 정리
        cleaned_text = _clean_markdown_code_blocks(response_text)
        
        # JSON 파싱 시도
        try:
            parsed = json.loads(cleaned_text)
            if isinstance(parsed, dict):
                logger.debug("Successfully parsed JSON from cleaned text")
                return parsed
        except json.JSONDecodeError:
            logger.debug("Direct JSON parsing failed, trying extraction")
        
        # 방법 3: fallback - 텍스트에서 JSON 추출
        if fallback_extract:
            extracted = _extract_json_from_text(cleaned_text)
            if extracted:
                logger.debug("Successfully extracted JSON using fallback method")
                return extracted
        
        logger.warning(f"Failed to parse JSON from response. Content: {response_text[:200]}")
        return None
        
    except Exception as e:
        logger.error(f"Error parsing JSON response: {str(e)}", exc_info=True)
        return None


def parse_text_response(response: BaseMessage) -> str:
    """
    LLM 응답에서 텍스트를 추출 (LangChain 표준 방식)
    
    Args:
        response: LLM 응답 메시지 (BaseMessage)
        
    Returns:
        추출된 텍스트 문자열
    """
    try:
        content = _extract_content_from_response(response)
        if not content:
            return ""
        
        # 마크다운 코드 블록에서 텍스트만 추출 (필요한 경우)
        # SQL 코드 블록 등은 제거하지 않음 (사용자가 결정)
        return content.strip()
        
    except Exception as e:
        logger.error(f"Error parsing text response: {str(e)}")
        return ""


def _extract_content_from_response(response: BaseMessage) -> str:
    """
    응답 메시지에서 텍스트 내용 추출
    
    Args:
        response: LLM 응답 메시지
        
    Returns:
        추출된 텍스트 문자열
    """
    if hasattr(response, 'content'):
        content = response.content
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            # 리스트인 경우 모든 항목을 문자열로 조합
            return " ".join(str(item) for item in content if item)
        else:
            return str(content) if content else ""
    else:
        return str(response) if response else ""


def _clean_markdown_code_blocks(text: str) -> str:
    """
    마크다운 코드 블록 제거 및 정리
    
    Args:
        text: 원본 텍스트
        
    Returns:
        정리된 텍스트
    """
    if not text or not isinstance(text, str):
        return ""
    
    # 이스케이프 문자 처리
    cleaned = text.replace('\\n', '\n').replace('\\t', '\t').replace('\\r', '\r')
    
    # 마크다운 코드 블록 제거 (```json ... ``` 또는 ``` ... ```)
    code_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    code_block_match = re.search(code_block_pattern, cleaned, re.DOTALL | re.IGNORECASE)
    if code_block_match:
        cleaned = code_block_match.group(1).strip()
    
    # 앞뒤 공백 제거
    cleaned = cleaned.strip()
    
    # JSON 객체 시작 부분 찾기
    start_idx = cleaned.find('{')
    if start_idx != -1:
        cleaned = cleaned[start_idx:]
    
    return cleaned


def _extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    텍스트에서 JSON 객체 추출 (중괄호 균형 맞춤)
    
    Args:
        text: JSON이 포함된 텍스트
        
    Returns:
        추출된 JSON 딕셔너리 또는 None
    """
    if not text or not isinstance(text, str):
        return None
    
    try:
        # 중괄호로 시작하는 부분 찾기
        start_idx = text.find('{')
        if start_idx == -1:
            return None
        
        # 중괄호 균형을 맞춰서 올바른 JSON 객체 찾기
        brace_count = 0
        end_idx = start_idx
        for i in range(start_idx, len(text)):
            if text[i] == '{':
                brace_count += 1
            elif text[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i
                    break
        
        if brace_count == 0 and end_idx > start_idx:
            json_str = text[start_idx:end_idx+1]
            try:
                parsed = json.loads(json_str)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass
        
        return None
        
    except Exception as e:
        logger.debug(f"Failed to extract JSON from text: {str(e)}")
        return None


def create_json_parser() -> SimpleJsonOutputParser:
    """
    SimpleJsonOutputParser 인스턴스 생성
    
    Returns:
        SimpleJsonOutputParser 인스턴스
    """
    return SimpleJsonOutputParser()

