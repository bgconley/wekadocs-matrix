"""
Phase 5, Task 5.1 - Webhook endpoints for external connectors.
Handles incoming webhooks from GitHub, Notion, Confluence, etc.
"""

import json
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

        # Read raw body once for signature verification and payload parsing
        body = await request.body()

        # Signature is required whenever a webhook secret is configured
        if github_connector.config.webhook_secret and not x_hub_signature_256:
            logger.warning("Missing GitHub webhook signature header")
            raise HTTPException(status_code=401, detail="Missing signature header")

        try:
            payload = json.loads(body.decode("utf-8"))
        except ValueError as exc:
            logger.warning("Invalid GitHub webhook payload: %s", exc)
            raise HTTPException(status_code=400, detail="Invalid JSON payload") from exc

        # Process webhook
        result = await github_connector.process_webhook(
            payload, signature=x_hub_signature_256, raw_body=body
        )

        if result.get("status") == "error":
            error_code = result.get("error") or "webhook_error"
            if error_code in {"missing_signature", "invalid_signature"}:
                raise HTTPException(status_code=401, detail="Invalid signature")
            if error_code == "raw_body_required":
                raise HTTPException(
                    status_code=500,
                    detail="Webhook handler misconfigured: raw body missing",
                )
            raise HTTPException(status_code=500, detail=error_code)

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
