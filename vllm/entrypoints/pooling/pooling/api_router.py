# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from http import HTTPStatus

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse
from typing_extensions import assert_never

from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.openai.utils import validate_json_request
from vllm.entrypoints.pooling.pooling.protocol import (
    IOProcessorResponse,
    PoolingBytesResponse,
    PoolingRequest,
    PoolingResponse,
    PoolingResponseData,
)
from vllm.entrypoints.pooling.pooling.serving import OpenAIServingPooling
from vllm.entrypoints.utils import load_aware_call, with_cancellation

router = APIRouter()


def pooling(request: Request) -> OpenAIServingPooling | None:
    return request.app.state.openai_serving_pooling


@router.post(
    "/pooling",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
@with_cancellation
@load_aware_call
async def create_pooling(request: PoolingRequest, raw_request: Request):
    handler = pooling(raw_request)
    if handler is None:
        base_server = raw_request.app.state.openai_serving_tokenization
        return base_server.create_error_response(
            message="The model does not support Pooling API"
        )

    # Handle embed_with_sparse: call both embed and token_classify, merge results
    if request.task == "embed_with_sparse":
        try:
            # Get tokenizer for tokenizing input
            tokenizer = handler.renderer.get_tokenizer()

            # Create requests for embed and token_classify
            embed_request = request.model_copy(deep=True)
            embed_request.task = "embed"
            sparse_request = request.model_copy(deep=True)
            sparse_request.task = "token_classify"

            # Call both handlers and get results
            embed_result = await handler.create_pooling(embed_request, raw_request)
            sparse_result = await handler.create_pooling(sparse_request, raw_request)

            # Merge results - create custom JSON response
            if isinstance(embed_result, PoolingResponse) and isinstance(sparse_result, PoolingResponse):
                # Get input texts from request
                input_texts = request.input if isinstance(request.input, list) else [request.input]
                merged_data = []
                for idx, (embed_data, sparse_data, text) in enumerate(zip(embed_result.data, sparse_result.data, input_texts)):
                    # Tokenize input to get token_ids
                    tokens = tokenizer.encode(text, add_special_tokens=False)
                    # Map token_ids to sparse weights
                    sparse_weights = sparse_data.data
                    if isinstance(sparse_weights, list):
                        sparse_dict = {str(token_id): w for token_id, w in zip(tokens, sparse_weights) if w > 0}
                    else:
                        sparse_dict = {}
                    merged_output = {
                        "index": idx,
                        "object": "pooling",
                        "data": {
                            "dense": embed_data.data,
                            "sparse": sparse_dict
                        }
                    }
                    merged_data.append(merged_output)
                response_body = {
                    "id": embed_result.id,
                    "object": "list",
                    "created": embed_result.created,
                    "model": embed_result.model,
                    "data": merged_data,
                    "usage": embed_result.usage.model_dump() if hasattr(embed_result.usage, 'model_dump') else embed_result.usage
                }
                return JSONResponse(content=response_body)
        except Exception as e:
            return JSONResponse(
                content={"error": {"message": str(e), "type": "InternalServerError", "code": 500}},
                status_code=500
            )

    try:
        generator = await handler.create_pooling(request, raw_request)
    except Exception as e:
        generator = handler.create_error_response(e)

    if isinstance(generator, ErrorResponse):
        return JSONResponse(
            content=generator.model_dump(), status_code=generator.error.code
        )
    elif isinstance(generator, (PoolingResponse, IOProcessorResponse)):
        return JSONResponse(content=generator.model_dump())
    elif isinstance(generator, PoolingBytesResponse):
        return StreamingResponse(
            content=generator.content,
            headers=generator.headers,
            media_type=generator.media_type,
        )

    assert_never(generator)
