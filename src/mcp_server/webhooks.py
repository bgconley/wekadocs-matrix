"""
Phase 5, Task 5.1 - Webhook endpoints for external connectors.
Handles incoming webhooks from GitHub, Notion, Confluence, etc.
"""

import logging
from typing import Dict, Optional

from fastapi import APIRouter, Header, HTTPException, Request

logger = logging.getLogger(__name__)

# Create router for webhook endpoints
router = APIRouter(prefix="/webhooks", tags=["webhooks"])


@router.post("/github")
async def github_webhook(
    request: Request,
    x_hub_signature_256: Optional[str] = Header(None),
    x_github_event: Optional[str] = Header(None),
):
    """
    GitHub webhook endpoint.
    Processes push events for documentation changes.

    Headers:
        X-Hub-Signature-256: Webhook signature for verification
        X-GitHub-Event: Event type (push, pull_request, etc.)

    Returns:
        {"status": "success|error", ...}
    """
    try:
        # Get connector manager from app state
        connector_manager = request.app.state.connector_manager
        if not connector_manager:
            raise HTTPException(
                status_code=503,
                detail="Connector manager not initialized",
            )

        # Get GitHub connector
        github_connector = connector_manager.get_connector("github")
        if not github_connector:
            raise HTTPException(
                status_code=404, detail="GitHub connector not configured"
            )

        # Read raw body for signature verification
        body = await request.body()

        # Verify signature
        if x_hub_signature_256:
            is_valid = await github_connector.verify_webhook_signature(
                body, x_hub_signature_256
            )
            if not is_valid:
                logger.warning("Invalid GitHub webhook signature")
                raise HTTPException(status_code=401, detail="Invalid signature")

        # Parse JSON payload
        payload = await request.json()

        # Process webhook
        result = await github_connector.process_webhook(payload, x_hub_signature_256)

        logger.info(
            f"GitHub webhook processed: {result.get('status')}",
            event_type=x_github_event,
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing GitHub webhook: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/notion")
async def notion_webhook(request: Request):
    """
    Notion webhook endpoint.
    Future implementation for Notion page updates.
    """
    return {"status": "not_implemented"}


@router.post("/confluence")
async def confluence_webhook(request: Request):
    """
    Confluence webhook endpoint.
    Future implementation for Confluence page updates.
    """
    return {"status": "not_implemented"}


@router.get("/health")
async def webhook_health(request: Request) -> Dict:
    """
    Health check for webhook endpoints.
    Returns status of configured connectors.
    """
    try:
        connector_manager = request.app.state.connector_manager
        if not connector_manager:
            return {"status": "unavailable", "connectors": []}

        connectors = connector_manager.get_all_stats()
        return {"status": "healthy", "connectors": connectors}

    except Exception as e:
        logger.error(f"Error in webhook health check: {e}")
        return {"status": "error", "error": str(e)}
