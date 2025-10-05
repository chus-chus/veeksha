import asyncio
import functools
from multiprocessing import Process
from multiprocessing import Queue as MPQueue
from queue import Empty
from typing import Any, Callable, List, Optional
import time

import aiohttp

from veeksha.config.client import ClientConfig
from veeksha.core.llm_clients import construct_client
from veeksha.core.llm_clients.base_llm_client import BaseLLMClient
from veeksha.logger import init_logger
from veeksha.metrics.request_metrics import RequestMetrics

logger = init_logger(__name__)


class RequestsLauncher:
    """Launch requests from LLMClients to their respective LLM APIs."""

    def __init__(
        self,
        client_config: ClientConfig,
        input_queue: MPQueue,
        output_queue: MPQueue,
        ready_queue: Optional[MPQueue] = None,
    ):
        self.client_config = client_config
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.ready_queue = ready_queue
        self.clients: List[Process] = []

    def start(self) -> None:
        """Start the clients."""
        for client_id in range(self.client_config.num_clients):
            client = Process(
                target=self.run_client,
                args=(client_id, self.ready_queue),
            )
            self.clients.append(client)
            client.start()

    def run_client(self, client_id: int, ready_queue: Optional[MPQueue] = None) -> None:
        """Run a client process that sends requests to the LLM API."""
        asyncio.run(self.run_async_worker(client_id, ready_queue))

    async def run_async_worker(self, client_id: int, ready_queue: Optional[MPQueue]) -> None:
        """Run the async worker that processes requests."""
        logger.debug("Starting async worker %s", client_id)

        # Create LLM client for this worker process
        llm_client = construct_client(
            model_name=self.client_config.model,
            tokenizer_name=self.client_config.tokenizer or self.client_config.model,
            llm_api=self.client_config.llm_api,
        )

        # Attach tracing to capture precise client-side request timings (e.g. write completion)
        trace_config = aiohttp.TraceConfig()

        @trace_config.on_request_start.append
        async def _on_request_start(session: aiohttp.ClientSession, context: Any, params: Any) -> None:
            try:
                if isinstance(context, dict):
                    context["request_start_at"] = time.monotonic()
            except Exception:
                pass

        @trace_config.on_request_chunk_sent.append
        async def _on_request_chunk_sent(session: aiohttp.ClientSession, context: Any, params: Any) -> None:
            try:
                if isinstance(context, dict):
                    context["last_chunk_sent_at"] = time.monotonic()
            except Exception:
                pass

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.client_config.request_timeout),
            trace_configs=[trace_config],
        ) as session:
            tasks = [
                asyncio.create_task(
                    self.process_requests_async(client_id, i, llm_client, session)
                )
                for i in range(self.client_config.num_concurrent_requests_per_client)
            ]
            # Signal readiness for each task created (available concurrency unit)
            if ready_queue is not None:
                for _ in range(self.client_config.num_concurrent_requests_per_client):
                    try:
                        ready_queue.put("READY")
                    except Exception:
                        pass
            await asyncio.gather(*tasks)

        logger.debug("Async worker %s finished", client_id)

    async def process_requests_async(
        self,
        client_id: int,
        task_id: int,
        llm_client: BaseLLMClient,
        session: aiohttp.ClientSession,
    ) -> None:
        """A task that processes requests from the input queue."""
        loop = asyncio.get_running_loop()
        get_from_queue = functools.partial(self.input_queue.get, timeout=1.0)
        put_to_queue = functools.partial(self.output_queue.put)

        while True:
            request_config = None
            try:
                try:
                    request_config = await loop.run_in_executor(None, get_from_queue)
                except Empty:
                    continue  # Poll the queue again

                if request_config is None:
                    break

                try:
                    result = await llm_client.send_llm_request(request_config, session)
                    await loop.run_in_executor(None, put_to_queue, result)
                except asyncio.CancelledError:
                    logger.debug("Worker %s task %s was cancelled", client_id, task_id)
                    break
                except Exception as e:
                    logger.exception(
                        "send_llm_request failed for client_id=%s, task_id=%s",
                        client_id,
                        task_id,
                    )
                    await self._emit_error_result(
                        loop=loop,
                        put_to_queue=put_to_queue,
                        e=e,
                        request_config=request_config,
                        client_id=client_id,
                        task_id=task_id,
                    )
                    continue

            except asyncio.CancelledError:
                logger.warning(
                    "Worker %s task %s cancelled during queue operation",
                    client_id,
                    task_id,
                )
                break
            except Exception as e:
                logger.exception(
                    "Unexpected error in worker %s task %s", client_id, task_id
                )
                await self._emit_error_result(
                    loop=loop,
                    put_to_queue=put_to_queue,
                    e=e,
                    request_config=request_config,
                    client_id=client_id,
                    task_id=task_id,
                )
                continue

    async def _emit_error_result(
        self,
        loop: asyncio.AbstractEventLoop,
        put_to_queue: Callable[[Any], None],
        e: Exception,
        request_config: Optional[Any],
        client_id: int,
        task_id: int,
    ) -> None:
        """Emit an error RequestMetrics tuple to the output queue.

        Mirrors the standard error path to keep counters consistent across
        all failure scenarios.

        Args:
            loop: The current asyncio event loop.
            put_to_queue: Callable to put items onto the output queue.
            e: The exception that occurred.
            request_config: The request configuration associated with the failure, if any.
            client_id: ID of the worker client.
            task_id: ID of the worker task within the client.
        """
        try:
            try:
                prompt_len = (
                    request_config.prompt[1]
                    if request_config and getattr(request_config, "prompt", None)
                    else 0
                )
            except Exception:
                prompt_len = 0

            error_code = (
                e.status if isinstance(e, aiohttp.ClientResponseError) else None
            )
            metrics = RequestMetrics(
                request_dispatched_at=0.0,
                inter_token_times=[],
                num_prompt_tokens=prompt_len,
                num_output_tokens=0,
                error_msg=str(e),
                error_code=error_code,
            )
            await loop.run_in_executor(None, put_to_queue, (metrics, None))
        except Exception:
            logger.exception(
                "Failed to enqueue error result for worker %s task %s",
                client_id,
                task_id,
            )

    def complete_tasks(self) -> None:
        """Signal worker processes to complete their tasks and exit."""
        for _ in range(
            self.client_config.num_clients
            * self.client_config.num_concurrent_requests_per_client
        ):
            self.input_queue.put(None)

    def wait_for_clients(self) -> None:
        """Wait for all clients to complete their tasks and exit."""
        self.complete_tasks()
        for client in self.clients:
            client.join()

    def wait_until_ready(self) -> None:
        """Block until all worker concurrency units report readiness."""
        if self.ready_queue is None:
            return
        expected = (
            self.client_config.num_clients
            * self.client_config.num_concurrent_requests_per_client
        )
        ready_count = 0
        while ready_count < expected:
            try:
                msg = self.ready_queue.get(timeout=5.0)
            except Empty:
                continue
            if msg == "READY":
                ready_count += 1
