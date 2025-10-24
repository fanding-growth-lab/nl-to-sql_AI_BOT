"""
Slack Bot Implementation

Main Slack bot class that integrates with the natural language to SQL pipeline.
"""

import os
import asyncio
import logging
from typing import Dict, Any, Optional, Callable
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_sdk.errors import SlackApiError

from slack.config import SlackConfig
from .handlers import SlackEventHandler
from core.config import get_settings
from agentic_flow import AgentGraphRunner, AgentState, ExecutionMode

logger = logging.getLogger(__name__)


class SlackBot:
    """
    Main Slack bot class that handles Slack events and integrates with the NL-to-SQL pipeline.
    """
    
    def __init__(self, config: Optional[SlackConfig] = None):
        """
        Initialize the Slack bot.
        
        Args:
            config: Slack configuration. If None, loads from environment.
        """
        self.config = config or SlackConfig.from_env()
        self.app = None
        self.pipeline_runner = None
        self._initialize_bot()
        self._initialize_pipeline()
        self._setup_handlers()
    
    def _initialize_bot(self):
        """Initialize the Slack app."""
        try:
            # Validate tokens before creating the app
            if not self.config.validate_tokens():
                logger.warning("Slack tokens validation failed. Some features may not work.")
                logger.info(f"Config: {self.config.get_masked_config()}")
            
            # Create Slack app
            self.app = App(
                token=self.config.bot_token,
                signing_secret=self.config.signing_secret
            )
            
            logger.info("Slack bot initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Slack bot: {e}")
            raise
    
    def _initialize_pipeline(self):
        """Initialize the NL-to-SQL pipeline."""
        try:
            # Get settings for pipeline configuration
            settings = get_settings()
            
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
            
            # Initialize pipeline runner with full config
            self.pipeline_runner = AgentGraphRunner(pipeline_config)
            
            logger.info("NL-to-SQL pipeline initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize NL-to-SQL pipeline: {e}")
            # Continue without pipeline for basic bot functionality
            self.pipeline_runner = None
    
    def _setup_handlers(self):
        """Set up event handlers for the Slack bot."""
        try:
            # Initialize the main event handler with all sub-handlers
            self.event_handler = SlackEventHandler(
                app=self.app,
                agent_runner=self.pipeline_runner,
                show_error_details=self.config.show_error_details
            )
            
            logger.info("Slack event handlers setup completed successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup Slack handlers: {e}")
            # Fallback to basic handlers
            self._setup_fallback_handlers()
    
    def _setup_fallback_handlers(self):
        """Set up basic fallback handlers if main handlers fail."""
        try:
            @self.app.command("/help")
            def handle_help_command(ack, respond):
                """Handle help command."""
                ack()
                
                help_text = f"""
{self.config.bot_emoji} *{self.config.bot_name} Help*

*Available Commands:*
• `/help` - Show this help message
• `/status` - Check bot status
• `/ping` - Test bot connectivity

*Natural Language Queries:*
Just type your question in natural language! For example:
• "Show me sales data from last month"
• "What are the top 10 customers?"
• "How many orders were placed yesterday?"

*Examples:*
• Ask about data: "What's the total revenue this quarter?"
• Get insights: "Which products are selling best?"
• Query specific data: "Show me all orders from customer ID 12345"

*Learning Commands:*
• `/learning-insights` - Get AI learning system insights
• `/learning-optimize` - Apply learning-based optimizations
• `/learning-status` - Check learning system status

*Note:* Make sure to ask questions about data that exists in the database.
                """
                
                respond(help_text)
            
            @self.app.command("/status")
            def handle_status_command(ack, respond):
                """Handle status command."""
                ack()
                
                status_info = f"""
{self.config.bot_emoji} *{self.config.bot_name} Status*

• *Bot Status:* ✅ Online
• *Pipeline Status:* {'✅ Ready' if self.pipeline_runner else '❌ Not Available'}
• *Database:* {'✅ Connected' if self._check_database_connection() else '❌ Disconnected'}
• *Mode:* {'Socket Mode' if self.config.socket_mode else 'HTTP Mode'}
                """
                
                respond(status_info)
            
            @self.app.command("/ping")
            def handle_ping_command(ack, respond):
                """Handle ping command."""
                ack()
                respond(f"{self.config.bot_emoji} Pong! Bot is responsive.")
            
            @self.app.command("/learning-insights")
            def handle_learning_insights(ack, respond):
                """Handle /learning-insights command."""
                ack()
                try:
                    from agentic_flow.auto_learning_system import AutoLearningSystem
                    learning_system = AutoLearningSystem()
                    insights = learning_system.get_learning_report()
                    
                    # Format insights for Slack
                    total_queries = insights.get("learning_metrics", {}).get("total_queries", 0)
                    success_rate = insights.get("learning_metrics", {}).get("success_rate", 0)
                    total_patterns = insights.get("pattern_analysis", {}).get("total_patterns", 0)
                    
                    response = f"""🤖 *AI Learning System Insights*
                    
📊 *Performance Metrics:*
• Total Queries: {total_queries}
• Success Rate: {success_rate:.1%}
• Learned Patterns: {total_patterns}

🔍 *System Health:*
• Learning Enabled: {insights.get("system_health", {}).get("learning_enabled", "Unknown")}
• Data Freshness: {insights.get("system_health", {}).get("data_freshness", "Unknown")}
• Pattern Coverage: {insights.get("system_health", {}).get("pattern_coverage", 0)}

💡 *Recent Suggestions:*
{chr(10).join([f"• {suggestion}" for suggestion in insights.get("improvement_suggestions", [])[:3]])}
                    """
                    
                    respond(response)
                except Exception as e:
                    respond(f"❌ Failed to get learning insights: {str(e)}")
            
            @self.app.command("/learning-optimize")
            def handle_learning_optimize(ack, respond):
                """Handle /learning-optimize command."""
                ack()
                try:
                    from agentic_flow.auto_learning_system import AutoLearningSystem
                    learning_system = AutoLearningSystem()
                    result = learning_system.apply_optimizations()
                    
                    if result.get("status") == "success":
                        changes = result.get("total_changes", 0)
                        backup = result.get("backup_created", "")
                        response = f"""✅ *Learning Optimization Applied*
                        
🔄 *Changes Applied:* {changes}
💾 *Backup Created:* {backup.split('/')[-1] if backup else "N/A"}

📈 *Applied Changes:*
{chr(10).join([f"• {change.get('type', 'Unknown')}: {change.get('pattern', 'N/A')}" for change in result.get("applied_changes", [])[:5]])}
                        """
                    else:
                        response = f"ℹ️ *Optimization Result:* {result.get('message', 'No changes needed')}"
                    
                    respond(response)
                except Exception as e:
                    respond(f"❌ Failed to apply optimizations: {str(e)}")
            
            @self.app.command("/learning-status")
            def handle_learning_status(ack, respond):
                """Handle /learning-status command."""
                ack()
                try:
                    from agentic_flow.auto_learning_system import AutoLearningSystem
                    learning_system = AutoLearningSystem()
                    status = learning_system.get_optimization_status()
                    metrics = learning_system.get_performance_metrics()
                    
                    if status.get("status") == "success":
                        total_patterns = status.get("total_patterns", 0)
                        priority_dist = status.get("priority_distribution", {})
                        avg_confidence = status.get("average_confidence", 0)
                        
                        response = f"""📊 *Learning System Status*
                        
🎯 *Pattern Statistics:*
• Total Patterns: {total_patterns}
• Average Confidence: {avg_confidence:.2f}
• High Priority: {priority_dist.get('high', 0)}
• Medium Priority: {priority_dist.get('medium', 0)}
• Low Priority: {priority_dist.get('low', 0)}

⚡ *Performance Metrics:*
• Queue Size: {metrics.get("storage_metrics", {}).get("queue_size", 0)}
• Background Saver: {'🟢 Active' if metrics.get("learning_system", {}).get("background_saver_active") else '🔴 Inactive'}
• Learning Enabled: {'🟢 Yes' if metrics.get("learning_system", {}).get("learning_enabled") else '🔴 No'}
                        """
                    else:
                        response = f"❌ *Status Error:* {status.get('message', 'Unknown error')}"
                    
                    respond(response)
                except Exception as e:
                    respond(f"❌ Failed to get learning status: {str(e)}")
            
            logger.info("Fallback handlers setup completed")
            
        except Exception as e:
            logger.error(f"Failed to setup fallback handlers: {e}")
    
    def _process_message(self, text: str, channel: str, user: str, say: Callable, client):
        """
        Process a natural language message and generate SQL response.
        
        Args:
            text: Message text
            channel: Slack channel ID
            user: Slack user ID
            say: Slack say function
            client: Slack client
        """
        try:
            # Show typing indicator
            client.conversations_setTyping(channel=channel)
            
            # Check if pipeline is available
            if not self.pipeline_runner:
                say("❌ Sorry, the data processing pipeline is not available right now.")
                return
            
            # Check if message looks like a data query
            if not self._is_data_query(text):
                say("💡 Tip: Ask me about your data! Try: 'Show me sales data from last month'")
                return
            
            # Process with pipeline
            response = self._run_pipeline(text, user, channel)
            
            # Send response
            self._send_response(response, channel, say)
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            if self.config.show_error_details:
                say(f"❌ Error: {str(e)}")
            else:
                say("❌ Sorry, I couldn't process your request. Please try again.")
    
    def _is_data_query(self, text: str) -> bool:
        """
        Check if the text looks like a data query.
        
        Args:
            text: Input text
            
        Returns:
            bool: True if it looks like a data query
        """
        # Simple heuristics to detect data queries
        data_keywords = [
            "show", "get", "find", "list", "count", "total", "sum", "average",
            "data", "sales", "orders", "customers", "products", "revenue",
            "last month", "this week", "yesterday", "today", "quarter",
            "top", "best", "worst", "highest", "lowest"
        ]
        
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in data_keywords)
    
    def _run_pipeline(self, query: str, user: str, channel: str) -> Dict[str, Any]:
        """
        Run the NL-to-SQL pipeline.
        
        Args:
            query: Natural language query
            user: Slack user ID
            channel: Slack channel ID
            
        Returns:
            Dict containing pipeline results
        """
        try:
            # Create initial state
            initial_state = {
                "user_query": query,
                "user_id": user,
                "channel_id": channel,
                "session_id": f"{user}_{channel}_{int(asyncio.get_event_loop().time())}",
                "context": {
                    "platform": "slack",
                    "bot_name": self.config.bot_name
                }
            }
            
            # Run pipeline
            result = self.pipeline_runner.run_sync(
                initial_state=initial_state,
                mode=ExecutionMode.STANDARD
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Pipeline execution error: {e}")
            return {
                "success": False,
                "error": str(e),
                "final_response": "❌ Sorry, I couldn't process your query."
            }
    
    def _send_response(self, result: Dict[str, Any], channel: str, say: Callable):
        """
        Send the pipeline result to Slack.
        
        Args:
            result: Pipeline execution result
            channel: Slack channel ID
            say: Slack say function
        """
        try:
            if result.get("success", False):
                response_text = result.get("final_response", "✅ Query processed successfully!")
                
                # Add metadata if available
                if "debug_info" in result:
                    debug_info = result["debug_info"]
                    if "sql_query" in debug_info:
                        response_text += f"\n\n```sql\n{debug_info['sql_query']}\n```"
            else:
                error_msg = result.get("error", "Unknown error occurred")
                response_text = f"❌ Error: {error_msg}"
            
            # Truncate if too long
            if len(response_text) > self.config.max_message_length:
                response_text = response_text[:self.config.max_message_length - 3] + "..."
            
            say(response_text)
            
        except Exception as e:
            logger.error(f"Error sending response: {e}")
            say("❌ Sorry, I couldn't send the response.")
    
    def _check_database_connection(self) -> bool:
        """Check if database connection is available."""
        try:
            # This would check the actual database connection
            # For now, return True if pipeline is available
            return self.pipeline_runner is not None
        except:
            return False
    
    def start(self):
        """Start the Slack bot."""
        try:
            if self.config.socket_mode:
                # Use Socket Mode
                handler = SocketModeHandler(
                    app=self.app,
                    app_token=self.config.app_token
                )
                logger.info("Starting Slack bot in Socket Mode...")
                handler.start()
            else:
                # Use HTTP mode (for production)
                logger.info("Starting Slack bot in HTTP Mode...")
                self.app.start(port=int(os.getenv("PORT", 3000)))
                
        except Exception as e:
            logger.error(f"Failed to start Slack bot: {e}")
            raise
    
    def stop(self):
        """Stop the Slack bot."""
        logger.info("Stopping Slack bot...")
        # Add cleanup logic here if needed


def create_slack_bot() -> SlackBot:
    """
    Factory function to create a configured Slack bot.
    
    Returns:
        SlackBot: Configured Slack bot instance
    """
    try:
        config = SlackConfig.from_env()
        bot = SlackBot(config)
        logger.info("Slack bot created successfully")
        return bot
    except Exception as e:
        logger.error(f"Failed to create Slack bot: {e}")
        raise


def get_bot_status(bot: SlackBot) -> Dict[str, Any]:
    """
    Get comprehensive status of the Slack bot.
    
    Args:
        bot: SlackBot instance
        
    Returns:
        Dictionary containing bot status information
    """
    status = {
        "bot_initialized": bot is not None,
        "config_valid": bot.config.validate_tokens() if bot else False,
        "pipeline_available": bot.pipeline_runner is not None if bot else False,
        "handlers_available": hasattr(bot, 'event_handler') and bot.event_handler is not None,
    }
    
    if bot and hasattr(bot, 'event_handler') and bot.event_handler:
        status["handler_details"] = bot.event_handler.get_handler_status()
        status["error_statistics"] = bot.event_handler.get_error_statistics()
    
    return status


def setup_slack_app() -> SlackBot:
    """
    Setup and configure Slack app for FastAPI integration.
    
    Returns:
        Configured SlackBot instance
    """
    try:
        config = SlackConfig.from_env()
        bot = SlackBot(config)
        logger.info("Slack app setup completed successfully")
        return bot
    except Exception as e:
        logger.error(f"Failed to setup Slack app: {e}")
        raise
