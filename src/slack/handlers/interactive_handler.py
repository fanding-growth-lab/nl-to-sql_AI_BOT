"""
Interactive Handler for Slack Events

This module handles interactive components like buttons, modals, and dropdowns.
"""

import asyncio
from typing import Dict, Any, Optional
from slack_bolt import App
from core.logging import get_logger
from .base_handler import BaseSlackHandler

logger = get_logger(__name__)


class InteractiveHandler(BaseSlackHandler):
    """Handler for Slack interactive components."""
    
    def _register_handlers(self):
        """Register interactive component handlers."""
        # Register button click handler
        self.app.action("regenerate_sql")(self.handle_regenerate_sql)
        self.app.action("explain_sql")(self.handle_explain_sql)
        self.app.action("show_schema")(self.handle_show_schema)
        
        # Register modal submission handler
        self.app.view("query_modal")(self.handle_query_modal)
        
        # Register shortcut handler
        self.app.shortcut("nl-to-sql-query")(self.handle_query_shortcut)
        
        logger.info("Interactive handlers registered successfully")
    
    async def handle_regenerate_sql(self, ack: callable, body: Dict[str, Any], client):
        """
        Handle SQL regeneration button click.
        
        Args:
            ack: Function to acknowledge the action
            body: Interactive component payload
            client: Slack WebClient
        """
        try:
            # Acknowledge the action
            ack()
            
            # Extract context from the action
            action = body["actions"][0]
            channel_id = body["channel"]["id"]
            thread_ts = body["message"]["thread_ts"]
            user_id = body["user"]["id"]
            
            # Get the original query from the message text or stored context
            original_query = self._extract_query_from_context(body)
            
            if not original_query:
                await client.chat_postMessage(
                    channel=channel_id,
                    thread_ts=thread_ts,
                    text="❌ 원본 쿼리를 찾을 수 없습니다."
                )
                return
            
            # Send processing message
            await client.chat_postMessage(
                channel=channel_id,
                thread_ts=thread_ts,
                text="🔄 SQL을 재생성 중입니다..."
            )
            
            # Process the query with regeneration flag
            if self.agent_runner:
                try:
                    # Add regeneration context
                    context = {"regenerate": True, "user_id": user_id}
                    
                    # Run the agent pipeline
                    result = await self._run_agent_pipeline(original_query, context=context)
                    
                    # Format and send response
                    formatted_response = self._format_agent_response(result)
                    await client.chat_postMessage(
                        channel=channel_id,
                        thread_ts=thread_ts,
                        text=formatted_response
                    )
                    
                except Exception as e:
                    error_msg = self._format_error_message(e)
                    await client.chat_postMessage(
                        channel=channel_id,
                        thread_ts=thread_ts,
                        text=error_msg
                    )
                    logger.error(f"SQL regeneration error: {str(e)}")
            else:
                await client.chat_postMessage(
                    channel=channel_id,
                    thread_ts=thread_ts,
                    text="❌ 에이전트가 초기화되지 않았습니다."
                )
                
        except Exception as e:
            logger.error(f"Regenerate SQL handler error: {str(e)}")
    
    async def handle_explain_sql(self, ack: callable, body: Dict[str, Any], client):
        """
        Handle SQL explanation button click.
        
        Args:
            ack: Function to acknowledge the action
            body: Interactive component payload
            client: Slack WebClient
        """
        try:
            # Acknowledge the action
            ack()
            
            # Extract context
            channel_id = body["channel"]["id"]
            thread_ts = body["message"]["thread_ts"]
            
            # Get SQL query from the message
            sql_query = self._extract_sql_from_message(body)
            
            if not sql_query:
                await client.chat_postMessage(
                    channel=channel_id,
                    thread_ts=thread_ts,
                    text="❌ SQL 쿼리를 찾을 수 없습니다."
                )
                return
            
            # Send processing message
            await client.chat_postMessage(
                channel=channel_id,
                thread_ts=thread_ts,
                text="📖 SQL 쿼리를 분석 중입니다..."
            )
            
            # Generate explanation
            explanation = await self._generate_sql_explanation(sql_query)
            
            # Format and send explanation
            formatted_explanation = self._format_sql_explanation(sql_query, explanation)
            await client.chat_postMessage(
                channel=channel_id,
                thread_ts=thread_ts,
                text=formatted_explanation
            )
            
        except Exception as e:
            logger.error(f"Explain SQL handler error: {str(e)}")
    
    async def handle_show_schema(self, ack: callable, body: Dict[str, Any], client):
        """
        Handle show schema button click.
        
        Args:
            ack: Function to acknowledge the action
            body: Interactive component payload
            client: Slack WebClient
        """
        try:
            # Acknowledge the action
            ack()
            
            # Extract context
            channel_id = body["channel"]["id"]
            thread_ts = body["message"]["thread_ts"]
            
            # Send processing message
            await client.chat_postMessage(
                channel=channel_id,
                thread_ts=thread_ts,
                text="📋 데이터베이스 스키마를 조회 중입니다..."
            )
            
            # Get database schema
            schema_info = await self._get_database_schema()
            
            # Format and send schema information
            formatted_schema = self._format_schema_info(schema_info)
            await client.chat_postMessage(
                channel=channel_id,
                thread_ts=thread_ts,
                text=formatted_schema
            )
            
        except Exception as e:
            logger.error(f"Show schema handler error: {str(e)}")
    
    async def handle_query_modal(self, ack: callable, body: Dict[str, Any], client):
        """
        Handle query modal submission.
        
        Args:
            ack: Function to acknowledge the modal submission
            body: Modal submission payload
            client: Slack WebClient
        """
        try:
            # Acknowledge the modal submission
            ack()
            
            # Extract form values
            values = body["view"]["state"]["values"]
            query_text = values["query_input"]["query_value"]["value"]
            user_id = body["user"]["id"]
            
            if not query_text:
                return
            
            # Send to a DM or channel based on context
            # This is a simplified implementation
            await client.chat_postMessage(
                channel=f"@{user_id}",
                text=f"📝 모달에서 받은 쿼리: {query_text}\n\n🔍 처리 중입니다..."
            )
            
            # Process the query
            if self.agent_runner:
                try:
                    result = await self._run_agent_pipeline(query_text)
                    formatted_response = self._format_agent_response(result)
                    
                    await client.chat_postMessage(
                        channel=f"@{user_id}",
                        text=formatted_response
                    )
                    
                except Exception as e:
                    error_msg = self._format_error_message(e)
                    await client.chat_postMessage(
                        channel=f"@{user_id}",
                        text=error_msg
                    )
                    
        except Exception as e:
            logger.error(f"Query modal handler error: {str(e)}")
    
    async def handle_query_shortcut(self, ack: callable, body: Dict[str, Any], client):
        """
        Handle slash command or shortcut for query.
        
        Args:
            ack: Function to acknowledge the shortcut
            body: Shortcut payload
            client: Slack WebClient
        """
        try:
            # Acknowledge the shortcut
            ack()
            
            # Open a modal for query input
            modal_view = self._create_query_modal()
            
            await client.views_open(
                trigger_id=body["trigger_id"],
                view=modal_view
            )
            
        except Exception as e:
            logger.error(f"Query shortcut handler error: {str(e)}")
    
    def _extract_query_from_context(self, body: Dict[str, Any]) -> Optional[str]:
        """Extract original query from interactive component context."""
        try:
            # Try to get from message text
            message_text = body.get("message", {}).get("text", "")
            
            # Extract query from message text (this is a simplified approach)
            lines = message_text.split("\n")
            for line in lines:
                if "생성된 SQL 쿼리" in line or "📝" in line:
                    continue
                elif line.strip() and not line.startswith("*") and not line.startswith("```"):
                    return line.strip()
            
            return None
        except Exception:
            return None
    
    def _extract_sql_from_message(self, body: Dict[str, Any]) -> Optional[str]:
        """Extract SQL query from message text."""
        try:
            message_text = body.get("message", {}).get("text", "")
            
            # Look for SQL code blocks
            import re
            sql_match = re.search(r'```sql\n(.*?)\n```', message_text, re.DOTALL)
            if sql_match:
                return sql_match.group(1).strip()
            
            return None
        except Exception:
            return None
    
    async def _run_agent_pipeline(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run the agent pipeline with optional context."""
        if hasattr(self.agent_runner, 'run_async'):
            return await self.agent_runner.run_async(query=query, context=context)
        else:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, 
                lambda: self.agent_runner.run(query=query, context=context)
            )
    
    async def _generate_sql_explanation(self, sql_query: str) -> str:
        """Generate explanation for SQL query."""
        # This is a placeholder - in a real implementation, you would use an LLM
        # to generate explanations based on the SQL query and database schema
        
        explanation_parts = []
        
        # Basic SQL analysis
        sql_lower = sql_query.lower()
        
        if "select" in sql_lower:
            explanation_parts.append("이 쿼리는 데이터를 조회합니다.")
        if "from" in sql_lower:
            explanation_parts.append("FROM 절에서 테이블을 지정합니다.")
        if "where" in sql_lower:
            explanation_parts.append("WHERE 절에서 조건을 필터링합니다.")
        if "join" in sql_lower:
            explanation_parts.append("JOIN을 사용하여 여러 테이블을 연결합니다.")
        if "group by" in sql_lower:
            explanation_parts.append("GROUP BY를 사용하여 데이터를 그룹화합니다.")
        if "order by" in sql_lower:
            explanation_parts.append("ORDER BY를 사용하여 결과를 정렬합니다.")
        
        if not explanation_parts:
            explanation_parts.append("SQL 쿼리의 구조를 분석할 수 없습니다.")
        
        return " ".join(explanation_parts)
    
    async def _get_database_schema(self) -> Dict[str, Any]:
        """Get database schema information."""
        # This is a placeholder - in a real implementation, you would query
        # the database to get schema information
        
        return {
            "tables": [
                {"name": "users", "columns": ["id", "name", "email", "created_at"]},
                {"name": "orders", "columns": ["id", "user_id", "amount", "order_date"]},
                {"name": "products", "columns": ["id", "name", "price", "category"]}
            ]
        }
    
    def _format_sql_explanation(self, sql_query: str, explanation: str) -> str:
        """Format SQL explanation for display."""
        return f"""*📖 SQL 쿼리 설명*

```sql
{sql_query}
```

*분석:*
{explanation}"""
    
    def _format_schema_info(self, schema_info: Dict[str, Any]) -> str:
        """Format database schema information for display."""
        tables = schema_info.get("tables", [])
        
        if not tables:
            return "❌ 데이터베이스 스키마 정보를 가져올 수 없습니다."
        
        formatted_parts = ["*📋 데이터베이스 스키마*"]
        
        for table in tables:
            table_name = table.get("name", "unknown")
            columns = table.get("columns", [])
            
            formatted_parts.append(f"\n*{table_name}* 테이블:")
            for column in columns:
                formatted_parts.append(f"  • {column}")
        
        return "\n".join(formatted_parts)
    
    def _create_query_modal(self) -> Dict[str, Any]:
        """Create a modal view for query input."""
        return {
            "type": "modal",
            "callback_id": "query_modal",
            "title": {
                "type": "plain_text",
                "text": "NL-to-SQL 쿼리"
            },
            "submit": {
                "type": "plain_text",
                "text": "실행"
            },
            "close": {
                "type": "plain_text",
                "text": "취소"
            },
            "blocks": [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "자연어로 데이터베이스 쿼리를 입력해주세요."
                    }
                },
                {
                    "type": "input",
                    "block_id": "query_input",
                    "element": {
                        "type": "plain_text_input",
                        "action_id": "query_value",
                        "multiline": True,
                        "placeholder": {
                            "type": "plain_text",
                            "text": "예: 모든 사용자 목록을 보여줘"
                        }
                    },
                    "label": {
                        "type": "plain_text",
                        "text": "쿼리"
                    }
                }
            ]
        }








