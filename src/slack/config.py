"""
Slack Bot Configuration

This module handles Slack-specific configuration settings.
"""

from typing import Optional
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class SlackConfig(BaseModel):
    """Slack bot configuration settings."""
    
    # Slack API tokens
    bot_token: str = Field(..., description="Slack Bot User OAuth Token (xoxb-...)")
    app_token: str = Field(..., description="Slack App-Level Token (xapp-...)")
    signing_secret: str = Field(..., description="Slack Signing Secret for request verification")
    
    # Socket Mode settings
    socket_mode: bool = Field(default=True, description="Enable Socket Mode for development")
    
    # Bot behavior settings
    bot_name: str = Field(default="pfbearbot", description="Display name for the bot")
    bot_emoji: str = Field(default=":robot_face:", description="Emoji for bot messages")
    
    # Response settings
    typing_delay: float = Field(default=0.5, description="Delay before sending typing indicator")
    max_message_length: int = Field(default=4000, description="Maximum message length")
    
    # Error handling
    show_error_details: bool = Field(default=False, description="Show detailed error messages to users")
    
    @classmethod
    def from_env(cls) -> "SlackConfig":
        """Create SlackConfig from environment variables."""
        return cls(
            bot_token=os.getenv("SLACK_BOT_TOKEN", ""),
            app_token=os.getenv("SLACK_APP_TOKEN", ""),
            signing_secret=os.getenv("SLACK_SIGNING_SECRET", "")
        )
    
    def validate_tokens(self) -> bool:
        """Validate that all required tokens are present and properly formatted."""
        if not self.bot_token or not self.bot_token.startswith("xoxb-"):
            return False
        if not self.app_token or not self.app_token.startswith("xapp-"):
            return False
        if not self.signing_secret:
            return False
        return True
    
    def get_masked_config(self) -> dict:
        """Get configuration with masked sensitive values."""
        return {
            "bot_token": f"{self.bot_token[:10]}..." if self.bot_token else "Not set",
            "app_token": f"{self.app_token[:10]}..." if self.app_token else "Not set",
            "signing_secret": f"{self.signing_secret[:10]}..." if self.signing_secret else "Not set",
            "socket_mode": self.socket_mode,
            "bot_name": self.bot_name,
            "bot_emoji": self.bot_emoji,
            "typing_delay": self.typing_delay,
            "max_message_length": self.max_message_length,
            "show_error_details": self.show_error_details
        }

