"""
FastAPI Application with Slack Bolt Integration

Main entry point for the NL-to-SQL Slack Bot application.
Integrates FastAPI web framework with Slack Bolt for event handling.
"""

import os
import sys
import argparse
import logging
import time
import uuid
from typing import Dict, Any, Optional

from core.config import get_settings
from core.logging import get_logger
from agentic_flow import AgentGraphRunner, AgentState, ExecutionMode

import warnings
from sqlalchemy import exc as sa_exc

# ignore warning messages
warnings.filterwarnings("ignore", category=sa_exc.SAWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Initialize logger
logger = get_logger(__name__)


def initialize_pipeline() -> Optional[AgentGraphRunner]:
    """Source: SlackBot._initialize_pipeline()"""
    try:
        # Get settings for pipeline configuration
        settings = get_settings()
        
        # Load actual database schema
        from core.db import get_cached_db_schema
        db_schema = get_cached_db_schema()
        logger.info(f"Loaded database schema: {len(db_schema)} tables")
        
        # Create pipeline configuration
        pipeline_config = {
            "llm_config": {
                "provider": settings.llm.provider,
                "model": settings.llm.model,
                "api_key": settings.llm.api_key,
                "temperature": settings.llm.temperature,
                "max_tokens": settings.llm.max_tokens
            },
            "database_config": {
                "host": settings.database.host,
                "port": settings.database.port,
                "username": settings.database.username,
                "password": settings.database.password,
                "database": settings.database.database,
                "charset": settings.database.charset
            },
            "pipeline_config": {
                "max_retries": settings.pipeline.max_retries,
                "confidence_threshold": settings.pipeline.confidence_threshold,
                "enable_debug": settings.pipeline.enable_debug,
                "enable_monitoring": settings.pipeline.enable_monitoring
            },
            "llm": {
                "provider": settings.llm.provider,
                "model": settings.llm.model,
                "api_key": settings.llm.api_key,
                "temperature": settings.llm.temperature,
                "max_tokens": settings.llm.max_tokens
            }
        }
        
        # Initialize pipeline runner with actual DB schema
        pipeline_runner = AgentGraphRunner(
            db_schema=db_schema,  # 실제 DB 스키마 전달
            config=pipeline_config
        )
        
        logger.info("NL-to-SQL pipeline initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize NL-to-SQL pipeline: {e}")
        # Continue without pipeline for basic bot functionality
        pipeline_runner = None
    
    return pipeline_runner


def format_result_as_table(result_data: list) -> str:
    """Source: MessageHandler._format_result_as_table()"""
    if not result_data:
        return "결과가 없습니다."
    
    # Get all unique keys from all rows
    all_keys = set()
    for row in result_data:
        all_keys.update(row.keys())
    
    # Sort keys for consistent display
    sorted_keys = sorted(all_keys)
    
    # Create table header
    header = " | ".join(f"{key:<15}" for key in sorted_keys)
    separator = " | ".join("-" * 15 for _ in sorted_keys)
    
    # Create table rows
    rows = []
    for row in result_data:
        row_values = []
        for key in sorted_keys:
            value = row.get(key, "")
            # Truncate long values
            if isinstance(value, str) and len(value) > 15:
                value = value[:12] + "..."
            row_values.append(f"{str(value):<15}")
        rows.append(" | ".join(row_values))
    
    # Combine header, separator, and rows
    table_parts = [header, separator] + rows
    return "\n".join(table_parts)


def format_agent_response(result_dict: Dict[str, Any]) -> str:
    """Source: MessageHandler._format_agent_response()"""
    # 일반 대화 응답이 있는 경우 우선 처리
    if result_dict.get("conversation_response"):
        logger.info("💬 Using conversation response for display")
        return result_dict["conversation_response"]
    
    formatted_parts = []
    
    # Add SQL query if available
    if result_dict.get("sql_query") or result_dict.get("final_sql"):
        sql_query = result_dict.get("sql_query") or result_dict.get("final_sql")
        if sql_query and isinstance(sql_query, str):
            formatted_parts.append("*📝 생성된 SQL 쿼리:*")
            formatted_parts.append(f"```sql\n{sql_query.replace('                       ', '')}\n```")
    
    # Add query result if available
    if result_dict.get("query_result") or result_dict.get("data_summary"):
        query_result = result_dict.get("query_result")
        data_summary = result_dict.get("data_summary")
        
        if query_result and isinstance(query_result, list) and len(query_result) > 0:
            formatted_parts.append("*📊 쿼리 결과:*")
            
            # Format as table if small result set
            if len(query_result) <= 10:
                table_text = format_result_as_table(query_result)
                formatted_parts.append(f"```\n{table_text}\n```")
            else:
                formatted_parts.append(f"총 {len(query_result)}개 행이 반환되었습니다.")
        
        # Add summary if available
        if data_summary:
            formatted_parts.append("*📋 요약:*")
            formatted_parts.append(data_summary)
        
        # Add business insights if available
        insight_report_formatted = result_dict.get("insight_report_formatted")
        if insight_report_formatted:
            formatted_parts.append("*💡 비즈니스 인사이트:*")
            # Slack 메시지 길이 제한을 고려하여 인사이트를 간소화
            insight_lines = insight_report_formatted.split('\n')
            simplified_insights = []
            for line in insight_lines:
                if line.strip() and not line.startswith('=') and not line.startswith('-'):
                    simplified_insights.append(line)
                    if len(simplified_insights) >= 10:  # 최대 10줄로 제한
                        break
            
            if simplified_insights:
                formatted_parts.append('\n'.join(simplified_insights))
        
        # Add business insights summary if available
        business_insights = result_dict.get("business_insights", [])
        if business_insights:
            high_impact_count = len([i for i in business_insights if i.get('impact_level') == 'high'])
            if high_impact_count > 0:
                formatted_parts.append(f"🚨 *{high_impact_count}개의 고위험/고영향 인사이트 발견*")
    
    # Add confidence score if available
    confidence_scores = result_dict.get("confidence_scores", {})
    if confidence_scores:
        sql_confidence = confidence_scores.get("sql_generation", 0)
        if sql_confidence > 0:
            confidence_emoji = "🟢" if sql_confidence >= 0.8 else "🟡" if sql_confidence >= 0.6 else "🔴"
            formatted_parts.append(f"{confidence_emoji} 신뢰도: {sql_confidence:.1%}")
    
    # Add error message if present
    if result_dict.get("error_message"):
        formatted_parts.append(f"⚠️ *경고:* {result_dict['error_message']}")
    
    # If no specific results, show general success message
    if not formatted_parts:
        if result_dict.get("success", False):
            formatted_parts.append("✅ 쿼리가 성공적으로 처리되었습니다.")
        else:
            formatted_parts.append("❌ 쿼리 처리에 실패했습니다.")
    
    # Add feedback request for successful queries
    if result_dict.get("success", False):
        formatted_parts.append("📝 **피드백이 있으시면 언제든 말씀해주세요!**")
    
    return "\n\n".join(formatted_parts)


def main():
    """Main entry point for running the application."""
    parser = argparse.ArgumentParser(description="PF_bearbot API Server")
    parser.add_argument("--log-level", default="error", help="Log level (debug, info, warning, error)")

    args = parser.parse_args()

    # Set log level (하위 로거 포함 전체 적용)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True
    )

    # Determine if reload should be enabled@
    pipeline = initialize_pipeline()
    is_review_response = False

    try:
        while True:
            if is_review_response:
                # TBD
                continue

            print("system: exit 입력 시 종료됩니다.")
            user_query = input("user: ")
            if user_query == 'exit': break 

            if pipeline is None:
                print("❌ 파이프라인이 초기화되지 않았습니다.")
                continue

            session_id = str(uuid.uuid4())
            
            # state_machine에서 구현한 워크플로우에서 사용할 인자 넣어주기
            result = pipeline.process_query(
                user_query=user_query,
                session_id=session_id,
                skip_sql_generation=False,
                conversation_response=None,
                intent="DATA_QUERY",
                # ...
            )

            result_dict: Dict[str, Any]
            if hasattr(result, 'to_dict'):
                result_dict = result.to_dict()
            elif hasattr(result, '__dict__'):
                result_dict = result.__dict__
            elif isinstance(result, dict):
                result_dict = result
            else:
                # Convert to dict if possible
                result_dict = {"result": result, "success": False}
            
            # If it's a conversational response, don't send processing message
            if isinstance(result_dict, dict) and result_dict.get("conversation_response"):
                formatted_response = format_agent_response(result_dict)
                print(formatted_response, "\n\n")
                continue
            
            # For data queries, send processing message
            print("🔍 **쿼리를 분석하고 있습니다...**\n\n⏳ 자연어 처리 → SQL 생성 → 데이터 조회 순서로 진행됩니다.")
            
            # Format and send response
            if isinstance(result_dict, dict):
                formatted_response = format_agent_response(result_dict)
                print(formatted_response, "\n\n")
            else:
                print(f"❌ 예상치 못한 결과 형식: {type(result_dict)}")
                print(f"결과: {result_dict}\n\n")
    except KeyboardInterrupt:
        print("\n🛑 사용자가 실행을 중단했습니다.")
    except Exception as e:
        # Send processing message for errors too
        print("🔍 **쿼리를 분석하고 있습니다...**\n\n⏳ 자연어 처리 → SQL 생성 → 데이터 조회 순서로 진행됩니다.")

        logger.error(f"Agent pipeline error: {str(e)}", exc_info=True)
    finally:
        print("Forcing process exit...")
        os._exit(0)
            

if __name__ == "__main__":
    main()
