"""
Message Handler for Slack Events

This module handles message events including direct messages and app mentions.
"""

import asyncio
from typing import Dict, Any, Optional
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from core.logging import get_logger
from .base_handler import BaseSlackHandler
try:
    from src.monitoring.error_monitor import record_slack_error, record_sql_generation_error, record_nl_processing_error
except ImportError:
    # 모니터링 모듈이 없는 경우 무시
    def record_slack_error(message, severity, user_id=None, channel_id=None):
        pass
    def record_sql_generation_error(message, severity, user_id=None, channel_id=None):
        pass
    def record_nl_processing_error(message, severity, user_id=None, channel_id=None):
        pass

logger = get_logger(__name__)


class MessageHandler(BaseSlackHandler):
    """Handler for Slack message events."""
    
    def _register_handlers(self):
        """Register message event handlers."""
        # Register message handler for direct messages
        self.app.message()(self.handle_message)
        
        # Register app mention handler for channel mentions
        self.app.event("app_mention")(self.handle_app_mention)
        
        logger.info("Message handlers registered successfully")
    
    def handle_message(self, message: Dict[str, Any], say: callable):
        """
        Handle direct message events.
        
        Args:
            message: Slack message event
            say: Function to send response message
        """
        try:
            # Skip bot messages to avoid infinite loops
            if self._is_bot_message(message):
                return
            
            # Only respond to direct messages
            channel_type = message.get("channel_type", "")
            if not self._is_direct_message(channel_type):
                return
            
            # Extract and validate query
            query = message.get("text", "").strip()
            if not query:
                help_message = self._get_help_message()
                say(help_message)
                return
            
            # Log the event
            self._log_event("direct_message", message)
            
            # Get thread timestamp for responses
            thread_ts = self._get_thread_timestamp(message)
            
            # Process the query first to check if it's a conversational response
            if self.agent_runner:
                try:
                    # Run the agent pipeline
                    result = self._run_agent_pipeline(query)
                    
                    # Check if it's a conversational response
                    if hasattr(result, 'to_dict'):
                        result_dict = result.to_dict()
                    elif hasattr(result, '__dict__'):
                        result_dict = result.__dict__
                    else:
                        result_dict = result
                    
                    # If it's a conversational response, don't send processing message
                    if result_dict.get("conversation_response"):
                        formatted_response = self._format_agent_response(result)
                        say(formatted_response, thread_ts=thread_ts)
                        return
                    
                    # For data queries, send processing message
                    say("🔍 **쿼리를 분석하고 있습니다...**\n\n"
                         "⏳ 자연어 처리 → SQL 생성 → 데이터 조회 순서로 진행됩니다.", 
                         thread_ts=thread_ts)
                    
                    # Format and send response
                    formatted_response = self._format_agent_response(result)
                    say(formatted_response, thread_ts=thread_ts)
                    
                except Exception as e:
                    # Send processing message for errors too
                    say("🔍 **쿼리를 분석하고 있습니다...**\n\n"
                         "⏳ 자연어 처리 → SQL 생성 → 데이터 조회 순서로 진행됩니다.", 
                         thread_ts=thread_ts)
                    
                    error_msg = self._format_error_message(e)
                    say(error_msg, thread_ts=thread_ts)
                    
                    # Add alternative suggestions based on error type
                    suggestions = self._get_alternative_suggestions(e, query)
                    if suggestions:
                        say(suggestions, thread_ts=thread_ts)
                    
                    logger.error(f"Agent pipeline error: {str(e)}", exc_info=True)
            else:
                say("❌ 에이전트가 초기화되지 않았습니다.", thread_ts=thread_ts)
                
        except Exception as e:
            logger.error(f"Message handler error: {str(e)}", exc_info=True)
            say("❌ 메시지 처리 중 오류가 발생했습니다.")
    
    def handle_app_mention(self, event: Dict[str, Any], say: callable):
        """
        Handle app mention events in channels.
        
        Args:
            event: Slack app mention event
            say: Function to send response message
        """
        try:
            # Extract query from mention
            query = self._extract_query_from_mention(event.get("text", ""))
            if not query:
                help_message = self._get_help_message(is_mention=True)
                say(help_message)
                return
            
            # Log the event
            self._log_event("app_mention", event)
            
            # Get thread timestamp for responses
            thread_ts = self._get_thread_timestamp(event)
            
            # Process the query first to check if it's a conversational response
            if self.agent_runner:
                try:
                    # Run the agent pipeline
                    result = self._run_agent_pipeline(query)
                    
                    # Check if it's a conversational response
                    if hasattr(result, 'to_dict'):
                        result_dict = result.to_dict()
                    elif hasattr(result, '__dict__'):
                        result_dict = result.__dict__
                    else:
                        result_dict = result
                    
                    # If it's a conversational response, don't send processing message
                    if result_dict.get("conversation_response"):
                        formatted_response = self._format_agent_response(result)
                        say(formatted_response, thread_ts=thread_ts)
                        return
                    
                    # For data queries, send processing message
                    say("🔍 **쿼리를 분석하고 있습니다...**\n\n"
                         "⏳ 자연어 처리 → SQL 생성 → 데이터 조회 순서로 진행됩니다.", 
                         thread_ts=thread_ts)
                    
                    # Format and send response
                    formatted_response = self._format_agent_response(result)
                    say(formatted_response, thread_ts=thread_ts)
                    
                except Exception as e:
                    # Send processing message for errors too
                    say("🔍 **쿼리를 분석하고 있습니다...**\n\n"
                         "⏳ 자연어 처리 → SQL 생성 → 데이터 조회 순서로 진행됩니다.", 
                         thread_ts=thread_ts)
                    
                    error_msg = self._format_error_message(e)
                    say(error_msg, thread_ts=thread_ts)
                    
                    # Add alternative suggestions based on error type
                    suggestions = self._get_alternative_suggestions(e, query)
                    if suggestions:
                        say(suggestions, thread_ts=thread_ts)
                    
                    logger.error(f"Agent pipeline error: {str(e)}", exc_info=True)
            else:
                # No agent runner available
                say("❌ 에이전트가 초기화되지 않았습니다.", thread_ts=thread_ts)
                
        except Exception as e:
            logger.error(f"App mention handler error: {str(e)}", exc_info=True)
            say("❌ 멘션 처리 중 오류가 발생했습니다.")
    
    def _run_agent_pipeline(self, query: str) -> Dict[str, Any]:
        """
        Run the agent pipeline to process the query.
        
        Args:
            query: Natural language query
            
        Returns:
            Agent pipeline result
        """
        if hasattr(self.agent_runner, 'process_query_async'):
            # Use async method if available
            import asyncio
            return asyncio.run(self.agent_runner.process_query_async(user_query=query))
        else:
            # Use sync method directly
            return self.agent_runner.process_query(user_query=query)
    
    def _format_agent_response(self, result) -> str:
        """
        Format agent result for Slack message display.
        
        Args:
            result: Agent pipeline result
            
        Returns:
            Formatted message text
        """
        if not result:
            return "❌ 결과를 처리하는 중 오류가 발생했습니다."
        
        # Handle GraphExecutionResult object
        if hasattr(result, 'to_dict'):
            result_dict = result.to_dict()
        elif hasattr(result, '__dict__'):
            result_dict = result.__dict__
        else:
            result_dict = result
        
        # 일반 대화 응답이 있는 경우 우선 처리
        if result_dict.get("conversation_response"):
            logger.info("💬 Using conversation response for display")
            return result_dict["conversation_response"]
        
        formatted_parts = []
        
        # Add SQL query if available
        if result_dict.get("sql_query") or result_dict.get("final_sql"):
            sql_query = result_dict.get("sql_query") or result_dict.get("final_sql")
            formatted_parts.append("*📝 생성된 SQL 쿼리:*")
            formatted_parts.append(f"```sql\n{sql_query}\n```")
        
        # Add query result if available
        if result_dict.get("query_result") or result_dict.get("data_summary"):
            query_result = result_dict.get("query_result")
            data_summary = result_dict.get("data_summary")
            
            if query_result and isinstance(query_result, list) and len(query_result) > 0:
                formatted_parts.append("*📊 쿼리 결과:*")
                
                # Format as table if small result set
                if len(query_result) <= 10:
                    table_text = self._format_result_as_table(query_result)
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
    
    def _format_result_as_table(self, result_data: list) -> str:
        """
        Format query result as a simple table.
        
        Args:
            result_data: List of result dictionaries
            
        Returns:
            Formatted table string
        """
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
    
    def _format_error_message(self, error: Exception, show_details: bool = False) -> str:
        """
        Format error message for user display.
        
        Args:
            error: Exception that occurred
            show_details: Whether to show detailed error information
            
        Returns:
            Formatted error message
        """
        error_type = type(error).__name__
        error_message = str(error).lower()
        
        # 사용자 친화적 오류 메시지 매핑
        if "timeout" in error_message or isinstance(error, TimeoutError):
            return ("⏰ **요청 처리 시간이 초과되었습니다**\n\n"
                   "더 간단한 쿼리로 다시 시도해주세요:\n"
                   "• '사용자 수를 알려줘'\n"
                   "• '최근 펀딩 10개 보여줘'")
        
        elif "connection" in error_message or "database" in error_message:
            return ("🗄️ **데이터베이스 연결에 문제가 있습니다**\n\n"
                   "잠시 후 다시 시도해주세요. 문제가 지속되면 관리자에게 문의하세요.")
        
        elif "validation" in error_message or isinstance(error, ValueError):
            return ("❓ **질문을 이해하지 못했습니다**\n\n"
                   "다음과 같이 다시 질문해주세요:\n"
                   "• '모든 사용자 목록을 보여줘'\n"
                   "• '활성 사용자 수를 알려줘'\n"
                   "• '최근 펀딩 프로젝트 5개 보여줘'")
        
        elif "permission" in error_message or isinstance(error, PermissionError):
            return ("🔒 **권한이 없습니다**\n\n"
                   "이 작업을 수행할 권한이 없습니다. 관리자에게 문의하세요.")
        
        elif "api" in error_message or "slack" in error_message:
            return ("📱 **Slack 연동에 문제가 있습니다**\n\n"
                   "잠시 후 다시 시도해주세요. 문제가 지속되면 관리자에게 문의하세요.")
        
        elif "NoneType" in error_message or "'NoneType' object has no attribute 'get'" in error_message:
            return ("⚠️ **데이터 처리 중 오류 발생**\n\n"
                   "예상치 못한 오류가 발생했습니다.\n\n"
                   "💡 **해결 방법:**\n"
                   "• 질문을 다시 정리해서 말씀해주세요\n"
                   "• 더 구체적인 조건을 포함해주세요\n"
                   "• 문제가 지속되면 관리자에게 문의해주세요")
        
        else:
            # 일반적인 오류에 대한 친화적 메시지
            if show_details:
                return f"❌ **오류가 발생했습니다**\n\n{str(error)}\n\n나중에 다시 시도해주세요."
            else:
                return ("❌ **처리 중 문제가 발생했습니다**\n\n"
                       "다른 방식으로 질문해보시거나, 잠시 후 다시 시도해주세요.\n\n"
                       "도움이 필요하시면 관리자에게 문의하세요.")
    
    def _get_alternative_suggestions(self, error: Exception, original_query: str) -> str:
        """
        Get alternative suggestions based on error type and original query.
        
        Args:
            error: Exception that occurred
            original_query: Original user query
            
        Returns:
            Formatted suggestions string or empty string if no suggestions
        """
        error_message = str(error).lower()
        
        # Timeout errors - suggest simpler queries
        if "timeout" in error_message or isinstance(error, TimeoutError):
            return ("💡 **더 간단한 쿼리를 시도해보세요:**\n\n"
                   "• '사용자 수를 알려줘'\n"
                   "• '최근 펀딩 5개 보여줘'\n"
                   "• '활성 사용자 목록 보여줘'\n"
                   "• '오늘 가입한 사용자 수 알려줘'")
        
        # Validation errors - suggest better query formats
        elif "validation" in error_message or isinstance(error, ValueError):
            return ("💡 **다음과 같이 질문해보세요:**\n\n"
                   "• '모든 사용자 목록을 보여줘'\n"
                   "• '펀딩 프로젝트 중 성공한 것들 보여줘'\n"
                   "• '크리에이터 목록을 알려줘'\n"
                   "• '최근 7일간 가입한 사용자 수 알려줘'")
        
        # Connection errors - suggest basic queries
        elif "connection" in error_message or "database" in error_message:
            return ("💡 **기본적인 쿼리로 시도해보세요:**\n\n"
                   "• '사용자 수를 세어줘'\n"
                   "• '테이블 목록을 보여줘'\n"
                   "• '데이터베이스 상태 확인해줘'")
        
        # Query-related errors - suggest alternative approaches
        elif "sql" in error_message or "query" in error_message:
            return ("💡 **다른 방식으로 질문해보세요:**\n\n"
                   "• 더 구체적으로: '활성 상태인 사용자 목록'\n"
                   "• 더 간단하게: '사용자 수'\n"
                   "• 다른 테이블: '펀딩 프로젝트 목록'\n"
                   "• 날짜 조건: '이번 달 가입한 사용자'")
        
        # No suggestions for other errors
        return ""
    
    def _record_error_for_monitoring(self, error: Exception, query: str, user_id: Optional[str] = None, channel_id: Optional[str] = None):
        """오류를 모니터링 시스템에 기록"""
        try:
            error_type = type(error).__name__
            error_message = str(error)
            
            # Determine severity based on error type
            severity = "high" if isinstance(error, (ConnectionError, TimeoutError)) else "medium"
            
            # Determine component based on error message
            if "slack" in error_message.lower() or "api" in error_message.lower():
                record_slack_error(error_message, severity, user_id, channel_id)
            elif "sql" in error_message.lower() or "database" in error_message.lower():
                record_sql_generation_error(error_message, severity, user_id, channel_id)
            else:
                record_nl_processing_error(error_message, severity, user_id, channel_id)
                
        except Exception as e:
            logger.error(f"모니터링 시스템에 오류 기록 실패: {str(e)}")
    
    def _get_help_message(self, is_mention: bool = False) -> str:
        """
        Get comprehensive help message for users.
        
        Args:
            is_mention: Whether this is for app mention or direct message
            
        Returns:
            Formatted help message
        """
        mention_prefix = "@DataTalk Bot " if is_mention else ""
        
        return (f"🤖 **DataTalk Bot 도움말**\n\n"
               f"안녕하세요! 자연어로 데이터베이스 쿼리를 요청할 수 있습니다.\n\n"
               f"**📝 사용 방법:**\n"
               f"• `{mention_prefix}모든 사용자 목록을 보여줘`\n"
               f"• `{mention_prefix}활성 사용자 수를 알려줘`\n"
               f"• `{mention_prefix}최근 펀딩 프로젝트 5개 보여줘`\n"
               f"• `{mention_prefix}크리에이터 목록을 알려줘`\n\n"
               f"**💡 자주 묻는 질문:**\n"
               f"• 사용자 관련: '사용자 수', '가입한 사용자', '활성 사용자'\n"
               f"• 펀딩 관련: '펀딩 목록', '성공한 프로젝트', '진행 중인 프로젝트'\n"
               f"• 크리에이터 관련: '크리에이터 목록', '크리에이터 수'\n\n"
               f"**❓ 문제가 있으시면:**\n"
               f"• 더 간단한 질문으로 다시 시도해보세요\n"
               f"• 구체적인 조건을 추가해보세요 (예: '최근 7일간')\n"
               f"• 관리자에게 문의하세요\n\n"
               f"**🔧 예시 질문:**\n"
               f"• `{mention_prefix}이번 달에 가입한 사용자 수 알려줘`\n"
               f"• `{mention_prefix}성공한 펀딩 프로젝트 10개 보여줘`\n"
               f"• `{mention_prefix}크리에이터 중 가장 많은 팔로워를 가진 사람`")
