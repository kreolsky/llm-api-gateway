"""Embedding creation service proxying requests to configured providers."""
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse

from ..core.context import AuthContext
from ..core.logging import logger
from .base import BaseService


class EmbeddingService(BaseService):

    async def create_embeddings(self, request: Request, auth_context: AuthContext) -> Any:
        """Validate, dispatch, and return an embedding creation request."""
        prepared = await self._prepare_dispatch(
            request, auth_context,
            component="embedding_service", log_title="Embedding Request JSON",
            api="openai-completions",
        )

        async with self._guard_service_errors(prepared.error_ctx):
            response_data = await prepared.provider.embeddings(
                prepared.request_body, prepared.provider_model_name, prepared.model_config,
                request_id=prepared.request_id, extra_headers=prepared.identity_headers
            )

            logger.debug_data(
                title="Embedding Response JSON",
                data=response_data,
                request_id=prepared.request_id,
                component="embedding_service",
                data_flow="from_provider"
            )

            logger.info(
                f"Response: Embedding Creation | model={prepared.requested_model}",
                request_id=prepared.request_id,
                user_id=prepared.user_id,
                model_id=prepared.requested_model,
                token_usage={
                    "prompt_tokens": response_data.get("usage", {}).get("prompt_tokens", 0),
                    "total_tokens": response_data.get("usage", {}).get("total_tokens", 0)
                }
            )

            usage = response_data.get("usage", {})
            if usage:
                prepared.stats.set_usage(usage)

            return JSONResponse(content=response_data)
