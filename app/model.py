from typing import AsyncGenerator, Dict, List, Callable
from ray import serve
from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.routing import APIRoute
import logging
import os
import asyncio
from queue import Empty
import torch
from transformers import pipeline, AutoTokenizer, TextIteratorStreamer
from starlette.responses import StreamingResponse
import json
import time
import uuid
import gc

DEFAULT_MAX_MODEL_LENGTH = 8096
DEFAULT_MAX_TOKENS = 100
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9
DEFAULT_TOP_K = 50
DEFAULT_REPETITION_PENALTY = 1.1
DEFAULT_DEVICE_FALLBACK = "cpu"

num_gpus = torch.cuda.device_count()


class InternalErrorLoggingModule(APIRoute):
    def get_route_handler(self) -> Callable:
        original_route_handler = super().get_route_handler()

        async def custom_route_handler(request: Request) -> Response:
            try:
                return await original_route_handler(request)
            except Exception as e:
                logger.exception(e)
                detail = {"errors": [f"Internal server error: {e}"]}
                raise HTTPException(status_code=500, detail=detail)

        return custom_route_handler


logger = logging.getLogger("ray.serve")
app = FastAPI()
app.router.route_class = InternalErrorLoggingModule


async def check_model_liveness():
    status = serve.status()
    is_running = status.applications["model"].status == "RUNNING"
    if not is_running:
        raise HTTPException(status_code=503, detail="Model is not currently available.")
    logger.info("Model is running and available.")
    return True


class Model:
    def __init__(
        self, model_path: str, max_model_length: int = DEFAULT_MAX_MODEL_LENGTH
    ) -> None:
        self.pipeline = None
        self.tokenizer = None

        self.__model_path = model_path
        self.__loop = asyncio.get_running_loop()
        self.__device = "cuda" if torch.cuda.is_available() else DEFAULT_DEVICE_FALLBACK
        self.__dtype = torch.bfloat16 if self.__device == "cuda" else torch.float16
        self.__max_model_length = max_model_length

    def load_model(self) -> None:
        logger.info("Loading model...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.__model_path)

        assert self.tokenizer is not None

        self.pipeline = pipeline(
            task="text-generation",
            model=self.__model_path,
            tokenizer=self.tokenizer,
            device_map=self.__device,
            torch_dtype=self.__dtype,
        )
        logger.info("Model loaded successfully with optimizations.")

    def get_prompt(self, messages: List[Dict], max_tokens: int) -> str:
        assert self.tokenizer is not None

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # calculate the numebr of tokens in the prompt
        num_tokens = self.tokenizer.encode(prompt)
        if len(num_tokens) + max_tokens > self.__max_model_length:
            raise HTTPException(
                status_code=400,
                detail=f"Max model length is {self.__max_model_length} tokens. {len(num_tokens)} provided in the prompt and {max_tokens} max tokens requested.",
            )
        return prompt

    def __get_inference_args(self, body: Dict) -> Dict:
        assert self.tokenizer is not None
        max_tokens = body.get("max_tokens", DEFAULT_MAX_TOKENS)
        temperature = body.get("temperature", DEFAULT_TEMPERATURE)
        top_p = body.get("top_p", DEFAULT_TOP_P)
        top_k = body.get("top_k", DEFAULT_TOP_K)
        repetition_penalty = body.get("repetition_penalty", DEFAULT_REPETITION_PENALTY)

        do_sample = temperature is not None and temperature > 0
        return {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "return_full_text": False,
            "eos_token_id": self.tokenizer.eos_token_id,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "repetition_penalty": repetition_penalty,
        }

    def _clear_memory(self):
        """Clear GPU memory and run garbage collection"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    async def consume_streamer(
        self, streamer: TextIteratorStreamer
    ) -> AsyncGenerator[str, None]:
        """
        Consume tokens from the streamer asynchronously.
        """
        assert self.__model_path is not None
        try:
            while True:
                try:
                    for token in streamer:
                        if token:
                            formatted_token = json.dumps(
                                {
                                    "id": str(uuid.uuid4()),
                                    "object": "chat.completion.chunk",
                                    "created": int(time.time()),
                                    "model": self.__model_path,
                                    "choices": [
                                        {
                                            "index": 0,
                                            "delta": {
                                                "role": "assistant",
                                                "content": token,
                                            },
                                            "logprobs": None,
                                            "finish_reason": None,
                                        }
                                    ],
                                }
                            )
                            yield "data: " + formatted_token + "\n\n"
                    break
                except Empty:
                    await asyncio.sleep(0.001)
        finally:
            self._clear_memory()

    def _generate_text_for_stream(self, prompt: str, streamer, inference_args):
        if not self.pipeline:
            raise ValueError("Pipeline is not loaded")
        logger.info("Generating text for stream.")

        try:
            self.pipeline(
                prompt,
                streamer=streamer,
                **inference_args,
            )
        finally:
            self._clear_memory()

    def generate(self, prompt: str, body: Dict):
        assert self.pipeline is not None
        assert self.tokenizer is not None

        prompt = self.get_prompt(body.get("messages", []), body.get("max_tokens", 100))

        inference_args = self.__get_inference_args(body)

        try:
            with torch.no_grad():
                response = self.pipeline(
                    prompt,
                    **inference_args,
                )
            return response[0]["generated_text"]
        finally:
            self._clear_memory()

    def generate_stream(self, body: Dict):
        assert self.pipeline is not None
        assert self.tokenizer is not None

        prompt = self.get_prompt(body.get("messages", []), body.get("max_tokens", 100))

        inference_args = self.__get_inference_args(body)

        with torch.no_grad():
            streamer = TextIteratorStreamer(
                self.tokenizer, timeout=0, skip_prompt=True, skip_special_tokens=True
            )
            self.__loop.run_in_executor(
                None, self._generate_text_for_stream, prompt, streamer, inference_args
            )
            return StreamingResponse(
                self.consume_streamer(streamer),
                media_type="text/event-stream",
            )


@serve.deployment(
    num_replicas=1,
    ray_actor_options={
        "num_gpus": num_gpus,
        "num_cpus": 1,
    },
)
@serve.ingress(app)
class App:
    def __init__(self):
        self.__model_path = os.environ.get(
            "MODEL_PATH", "/pvc-home/app/download/base_model"
        )
        self.__model_id = os.environ.get("MODEL_ID", "tiiuae/Falcon-H1-500M-Instruct")
        self.__max_model_length = int(
            os.environ.get("MAX_MODEL_LENGTH", DEFAULT_MAX_MODEL_LENGTH)
        )

        self.model = None
        self.__load_model()

    def __load_model(self):
        assert self.model is None

        self.model = Model(self.__model_path, self.__max_model_length)
        self.model.load_model()

    @app.post("/v1/chat/completions")
    async def generate(self, request: Request):
        logger.info("calling /v1/chat/completions")

        try:
            body = await request.json()
            stream = body.get("stream", False)
            if stream:
                return self.model.generate_stream(body)
            else:
                return self.model.generate(body)
        except Exception as e:
            logger.error(f"Error generating chat completion: {e}")
            return {"error": "Failed to generate chat completion"}, 500

    @app.get("/v1/models")
    async def get_models(self):
        logger.info("calling /v1/models")
        await check_model_liveness()
        return {
            "object": "list",
            "data": [
                {
                    "id": self.__model_id,
                    "object": "model",
                    "owned_by": "openinnovationai",
                    "root": self.__model_id,
                }
            ],
        }
