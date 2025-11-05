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
            self.pipeline_runner = AgentGraphRunner(
                db_schema=db_schema,  # 실제 DB 스키마 전달
                config=pipeline_config
            )
            
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
                # Use connect() instead of start() to avoid signal handler issues in threads
                # connect() runs in a loop, so we need to keep it running
                # Add retry logic for connection failures
                max_retries = 5
                retry_delay = 5
                
                for attempt in range(max_retries):
                    try:
                        logger.info(f"Attempting Socket Mode connection (attempt {attempt + 1}/{max_retries})...")
                        handler.connect()
                        # If connect() returns, connection was successful
                        logger.info("Socket Mode connection established successfully")
                        break
                    except Exception as connect_error:
                        logger.error(f"Socket Mode connection error (attempt {attempt + 1}/{max_retries}): {connect_error}", exc_info=True)
                        if attempt < max_retries - 1:
                            logger.info(f"Retrying in {retry_delay} seconds...")
                            import time
                            time.sleep(retry_delay)
                        else:
                            logger.error("Failed to establish Socket Mode connection after all retries")
                            raise
            else:
                # Use HTTP mode (for production)
                logger.info("Starting Slack bot in HTTP Mode...")
                self.app.start(port=int(os.getenv("PORT", 3000)))
                
        except Exception as e:
            logger.error(f"Failed to start Slack bot: {e}", exc_info=True)
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
