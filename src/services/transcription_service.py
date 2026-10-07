"""Audio transcription service with default model fallback."""
from typing import Any

from fastapi import Request, UploadFile

from ..core.context import AuthContext, request_context
from ..core.error_handling import ErrorType, create_error
from ..core.logging import logger
from ..utils.mask import mask_headers
from .base import BaseService


def _select_upload(audio_file: UploadFile | None, file: UploadFile | None,
                   request_id: str, user_id: str) -> UploadFile:
    """The uploaded audio, from whichever form field the client used."""
    # WHY: some clients send 'audio_file', others 'file' — accept both.
    # The selection lives HERE, after the header log, so the refusal
    # (the request where the headers were worth having) is still logged.
    if audio_file:
        return audio_file
    if file:
        return file
    raise create_error(
        ErrorType.MISSING_REQUIRED_FIELD,
        field_name="audio_file or file",
        request_id=request_id,
        user_id=user_id,
    )


def _provider_request_body(uploaded_file: UploadFile, audio_data: bytes,
                           params: dict[str, Any]) -> dict[str, Any]:
    """The body shape Provider.transcriptions reads."""
    return {
        "audio": {
            "filename": uploaded_file.filename,
            "content_type": uploaded_file.content_type,
            "data": audio_data,
        },
        "params": params,
    }


class TranscriptionService(BaseService):
    """
    Transcription service that handles audio transcription requests.

    Supports both explicit model selection and default model fallback.
    Uses BaseService for validation, provider lookup, and dispatch.
    """

    async def create_transcription(
        self,
        request: Request,
        audio_file: UploadFile | None,
        auth_context: AuthContext,
        model_id: str | None = None,
        response_format: str = "json",
        temperature: float = 0.0,
        language: str | None = None,
        return_timestamps: bool = False,
        file: UploadFile | None = None,
    ) -> Any:
        """Create a transcription from an audio file using the specified or default model."""
        ctx = request_context(request)
        request_id, user_id = ctx.request_id, ctx.user_id
        logger.debug_data(title="Transcription Request Headers", data=mask_headers(dict(request.headers)),
                          request_id=request_id, component="transcription_service", data_flow="incoming")

        uploaded_file = _select_upload(audio_file, file, request_id, user_id)
        audio_data = await uploaded_file.read()

        params = {"language": language, "temperature": temperature,
                  "response_format": response_format, "return_timestamps": return_timestamps}
        logger.debug_data(
            title="Transcription Request Parameters",
            data={"model_id": model_id, **params, "filename": uploaded_file.filename,
                  "content_type": uploaded_file.content_type,
                  "file_size": len(audio_data) if audio_data else 0},
            request_id=request_id, component="transcription_service", data_flow="incoming",
        )
        model_id = model_id or self._default_model(user_id)

        # Transcriptions carry no usage block: tokens stay 0 and has_usage
        # stays False — the row still makes transcriptions visible in stats.
        async with self._guard_service_errors(
            {"request_id": request_id, "user_id": user_id, "model_id": model_id}
        ):
            # ARCH: multipart cannot enter the JSON wrapper, so transcription
            # rides the body-agnostic resolver — the same funnel
            # _prepare_dispatch delegates to after parsing its JSON body.
            target = await self._resolve_target(request, auth_context, model_id,
                                                api="openai-completions")
            response = await target.provider.transcriptions(
                _provider_request_body(uploaded_file, audio_data, params),
                target.provider_model_name, target.model_config,
                request_id=request_id, extra_headers=target.identity_headers,
            )
            logger.debug_data(title="Transcription Response JSON", data=response, request_id=request_id,
                              component="transcription_service", data_flow="from_provider")
            return response

    def _default_model(self, user_id: str) -> str:
        """DEFAULT_STT_MODEL, used when the request names no model."""
        model_id = self.config_manager.settings.default_stt_model
        logger.info(f"Using default transcription model: {model_id}",
            user_id=user_id,
            default_model=model_id
        )
        return model_id
