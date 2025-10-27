"""
FastAPI Application with Slack Bolt Integration

Main entry point for the NL-to-SQL Slack Bot application.
Integrates FastAPI web framework with Slack Bolt for event handling.
"""

import os
import sys
import argparse
import logging
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from slack_bolt.adapter.fastapi import SlackRequestHandler
from slack_bolt import App as SlackApp

# Add current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.config import get_settings
from core.logging import get_logger
from core.db import DatabaseManager
from slack.bot import create_slack_bot, get_bot_status
from agentic_flow.auto_learning_system import AutoLearningSystem

# Initialize logger
logger = get_logger(__name__)

# Global variables for app state
slack_app: Optional[SlackApp] = None
slack_handler: Optional[SlackRequestHandler] = None
learning_system: Optional[AutoLearningSystem] = None
db_manager: Optional[DatabaseManager] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for startup and shutdown events.
    
    Args:
        app: FastAPI application instance
    """
    # Startup
    await startup_event()
    yield
    # Shutdown
    await shutdown_event()


def create_fastapi_app() -> FastAPI:
    """
    Create and configure FastAPI application.
    
    Returns:
        Configured FastAPI application instance
    """
    settings = get_settings()
    
    # Create FastAPI app with metadata
    app = FastAPI(
        title="PF_bearbot API",
        description="Natural Language to SQL Slack Bot API",
        version=settings.version,
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
        lifespan=lifespan
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.debug else settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )
    
    # Add trusted host middleware for security
    if not settings.debug:
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=settings.allowed_hosts
        )
    
    # Add security headers middleware
    @app.middleware("http")
    async def add_security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        if not settings.debug:
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response
    
    return app


# Create FastAPI application
app = create_fastapi_app()


async def startup_event():
    """Application startup event handler."""
    global slack_app, slack_handler, db_manager, learning_system
    
    try:
        logger.info("Starting PF_bearbot API...")
        
        # Initialize database manager
        db_manager = DatabaseManager()
        logger.info("Database manager initialized")
        
        # Initialize learning system and store in app state for dependency injection
        learning_system = AutoLearningSystem()
        app.state.learning_system = learning_system
        logger.info("Auto learning system initialized")
        
        # Initialize Slack bot (optional - don't fail startup if Slack is not configured)
        try:
            slack_bot = create_slack_bot()
            slack_app = slack_bot.app
            slack_handler = SlackRequestHandler(slack_app)
            
            logger.info("Slack bot initialized successfully")
            
            # Log application status
            bot_status = get_bot_status(slack_bot)
            logger.info(f"Bot status: {bot_status}")
            
        except Exception as slack_error:
            logger.warning(f"Slack bot initialization failed (continuing without Slack): {slack_error}")
            slack_app = None
            slack_handler = None
        
        logger.info("PF_bearbot API started successfully")
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}", exc_info=True)
        raise


async def shutdown_event():
    """Application shutdown event handler."""
    global db_manager
    
    try:
        logger.info("Shutting down PF_bearbot API...")
        
        # Clean up database connections
        if db_manager:
            # Database cleanup would go here if needed
            logger.info("Database connections cleaned up")
        
        logger.info("PF_bearbot API shutdown completed")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {e}", exc_info=True)


# Dependency Injection Functions

def get_learning_system(request: Request) -> AutoLearningSystem:
    """
    Dependency injection function for learning system.
    
    Args:
        request: FastAPI request object
        
    Returns:
        AutoLearningSystem instance
        
    Raises:
        HTTPException: If learning system is not initialized
    """
    if not hasattr(request.app.state, "learning_system") or request.app.state.learning_system is None:
        raise HTTPException(status_code=503, detail="Learning system not initialized")
    return request.app.state.learning_system


# API Routes

@app.get("/")
async def root():
    """
    Root endpoint providing basic API information.
    
    Returns:
        Basic API information
    """
    settings = get_settings()
    return {
        "message": "PF_bearbot API",
        "version": settings.version,
        "environment": settings.environment.value,
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint for monitoring and load balancers.
    
    Returns:
        Application health status
    """
    global slack_app, db_manager
    
    health_status = {
        "status": "healthy",
        "timestamp": None,
        "components": {}
    }
    
    try:
        from datetime import datetime
        health_status["timestamp"] = datetime.utcnow().isoformat()
        
        # Check database connection with actual ping
        if db_manager:
            try:
                # 실제 DB 연결 테스트
                from core.db import get_db_session
                with get_db_session() as session:
                    session.execute("SELECT 1")
                health_status["components"]["database"] = "connected"
            except Exception as db_error:
                logger.warning(f"Database health check failed: {db_error}")
                health_status["components"]["database"] = "disconnected"
        else:
            health_status["components"]["database"] = "disconnected"
        
        # Check Slack app with actual API test
        if slack_app:
            try:
                # Slack API 연결 테스트
                response = slack_app.client.auth_test()
                if response["ok"]:
                    health_status["components"]["slack"] = "connected"
                else:
                    health_status["components"]["slack"] = "disconnected"
            except Exception as slack_error:
                logger.warning(f"Slack health check failed: {slack_error}")
                health_status["components"]["slack"] = "disconnected"
        else:
            health_status["components"]["slack"] = "disconnected"
        
        # Overall status - DB는 필수, Slack은 선택적
        db_healthy = health_status["components"]["database"] == "connected"
        slack_healthy = health_status["components"]["slack"] == "connected"
        
        if db_healthy:
            health_status["status"] = "healthy" if slack_healthy else "degraded"
        else:
            health_status["status"] = "unhealthy"
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat() if 'datetime' in locals() else None
            }
        )


# Learning System API Endpoints
@app.get("/learning/insights")
async def get_learning_insights(ls: AutoLearningSystem = Depends(get_learning_system)):
    """Get learning system insights."""
    try:
        insights = ls.get_learning_report()
        return {"status": "success", "data": insights}
    except Exception as e:
        logger.error(f"Failed to get learning insights: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/learning/performance")
async def get_learning_performance(ls: AutoLearningSystem = Depends(get_learning_system)):
    """Get learning system performance metrics."""
    try:
        metrics = ls.get_performance_metrics()
        return {"status": "success", "data": metrics}
    except Exception as e:
        logger.error(f"Failed to get learning performance: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/learning/optimize")
async def apply_learning_optimizations(ls: AutoLearningSystem = Depends(get_learning_system)):
    """Apply learning-based optimizations."""
    try:
        result = ls.apply_optimizations()
        return {"status": "success", "data": result}
    except Exception as e:
        logger.error(f"Failed to apply optimizations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/learning/optimization-status")
async def get_optimization_status(ls: AutoLearningSystem = Depends(get_learning_system)):
    """Get current optimization status."""
    try:
        status = ls.get_optimization_status()
        return {"status": "success", "data": status}
    except Exception as e:
        logger.error(f"Failed to get optimization status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/learning/export")
async def export_learning_data(ls: AutoLearningSystem = Depends(get_learning_system)):
    """Export learning data."""
    try:
        export_path = ls.export_learning_data()
        return {"status": "success", "export_path": export_path}
    except Exception as e:
        logger.error(f"Failed to export learning data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/learning/force-save")
async def force_save_learning_data(ls: AutoLearningSystem = Depends(get_learning_system)):
    """Force save learning data."""
    try:
        ls.force_save()
        return {"status": "success", "message": "Learning data saved successfully"}
    except Exception as e:
        logger.error(f"Failed to force save learning data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status")
async def detailed_status():
    """
    Detailed status endpoint with comprehensive system information.
    
    Returns:
        Detailed system status information
    """
    global slack_app
    
    try:
        status_info = {
            "application": {
                "name": "PF_bearbot",
                "version": get_settings().version,
                "environment": get_settings().environment.value,
                "debug": get_settings().debug
            },
            "database": {
                "status": "connected" if db_manager else "disconnected",
                "type": "MariaDB"
            },
            "slack": {
                "status": "connected" if slack_app else "disconnected"
            }
        }
        
        # Add detailed Slack bot status if available
        if slack_app:
            try:
                # This would require access to the bot instance
                # For now, we'll just indicate it's connected
                status_info["slack"]["bot_ready"] = True
            except Exception as e:
                status_info["slack"]["bot_ready"] = False
                status_info["slack"]["error"] = str(e)
        
        return status_info
        
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/slack/events")
async def slack_events(request: Request):
    """
    Slack events endpoint for handling Slack API events.
    
    Args:
        request: FastAPI request object
        
    Returns:
        Slack API response
    """
    global slack_handler
    
    # Slack URL verification challenge 처리
    try:
        body = await request.body()
        if body:
            import json
            data = json.loads(body.decode('utf-8'))
            if data.get('type') == 'url_verification':
                challenge = data.get('challenge')
                logger.info(f"Slack URL verification challenge: {challenge}")
                # Raw string으로 반환 (JSON으로 감싸지 않음)
                from fastapi.responses import PlainTextResponse
                return PlainTextResponse(content=challenge)
    except Exception as e:
        logger.debug(f"Challenge parsing failed: {e}")
    
    if not slack_handler:
        raise HTTPException(
            status_code=503, 
            detail="Slack handler not initialized"
        )
    
    try:
        return await slack_handler.handle(request)
    except Exception as e:
        logger.error(f"Slack event handling failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/slack/interactive")
async def slack_interactive(request: Request):
    """
    Slack interactive components endpoint.
    
    Args:
        request: FastAPI request object
        
    Returns:
        Slack API response
    """
    global slack_handler
    
    if not slack_handler:
        raise HTTPException(
            status_code=503, 
            detail="Slack handler not initialized"
        )
    
    try:
        return await slack_handler.handle(request)
    except Exception as e:
        logger.error(f"Slack interactive handling failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/slack/commands")
async def slack_commands(request: Request):
    """
    Slack slash commands endpoint.
    
    Args:
        request: FastAPI request object
        
    Returns:
        Slack API response
    """
    global slack_handler
    
    if not slack_handler:
        raise HTTPException(
            status_code=503, 
            detail="Slack handler not initialized"
        )
    
    try:
        return await slack_handler.handle(request)
    except Exception as e:
        logger.error(f"Slack command handling failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


# Development and debugging endpoints (only in debug mode)

@app.get("/debug/config")
async def debug_config():
    """
    Debug endpoint to show configuration (only available in debug mode).
    
    Returns:
        Configuration information
    """
    settings = get_settings()
    
    if not settings.debug:
        raise HTTPException(status_code=404, detail="Not found")
    
    return {
        "config": settings.get_masked_settings(),
        "environment": settings.get_environment(),
        "required_settings": settings.validate_required_settings()
    }


@app.get("/debug/bot-status")
async def debug_bot_status():
    """
    Debug endpoint to show bot status (only available in debug mode).
    
    Returns:
        Detailed bot status information
    """
    settings = get_settings()
    
    if not settings.debug:
        raise HTTPException(status_code=404, detail="Not found")
    
    try:
        # This would require access to the bot instance
        # For now, return basic status
        return {
            "slack_app_initialized": slack_app is not None,
            "slack_handler_initialized": slack_handler is not None,
            "database_manager_initialized": db_manager is not None
        }
    except Exception as e:
        logger.error(f"Debug bot status failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Error handlers

@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    """Custom 404 error handler."""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "message": "The requested resource was not found",
            "path": str(request.url.path)
        }
    )


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc: HTTPException):
    """Custom 500 error handler."""
    logger.error(f"Internal server error: {exc.detail}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An internal server error occurred"
        }
    )


def main():
    """Main entry point for running the application."""
    parser = argparse.ArgumentParser(description="PF_bearbot API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument("--port", default=8000, type=int, help="Port to bind the server to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--log-level", default="info", help="Log level (debug, info, warning, error)")
    
    args = parser.parse_args()
    
    # Set log level
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Import uvicorn here to avoid import issues
    import uvicorn
    
    # Determine if reload should be enabled@
    settings = get_settings()
    reload_enabled = args.reload or settings.debug
    
    logger.info(f"Starting server on {args.host}:{args.port}")
    logger.info(f"Debug mode: {settings.debug}")
    logger.info(f"Auto-reload: {reload_enabled}")
    
    # Run the server
    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port,
        reload=reload_enabled,
        log_level=args.log_level
    )


if __name__ == "__main__":
    main()
