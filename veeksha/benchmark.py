import multiprocessing
import os
import platform
import random
import threading
import time
from multiprocessing import Queue
from queue import Empty
from threading import Thread
from typing import List, Optional, Tuple, cast

import requests
from tqdm import tqdm  # type: ignore

from veeksha.benchmark_data_utils import (
    load_corpus,
    store_generated_texts,
    store_lmeval_results,
)
from veeksha.config.benchmark import BenchmarkConfig
from veeksha.config.client import ClientConfig
from veeksha.config.utils import prepare_benchmark_output_dir
from veeksha.core.hf_utils import get_tokenizer
from veeksha.core.requests_launcher import RequestsLauncher
from veeksha.core.response import Response
from veeksha.core.seeding import (
    SeedManager,
)
from veeksha.generators.request_generator.base_generator import BaseRequestGenerator
from veeksha.core.request_config import RequestConfig
from veeksha.generators.request_generator.generator_registry import (
    RequestGeneratorRegistry,
)
from veeksha.logger import init_logger
from veeksha.metrics.service_metrics import ServiceMetrics
from veeksha.types import RequestGeneratorType

logger = init_logger(__name__)


def setup_api_environment(
    api_key=None,
    api_url=None,
):
    """Set up environment variables for OpenAI API"""
    assert api_key is not None, "API key is required"
    assert api_url is not None, "API URL is required"
    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["OPENAI_API_BASE"] = api_url


def _probe_min_tokens_param_support(client_config: ClientConfig) -> bool:
    """Probe if server accepts the configured min token parameter."""
    import os

    import requests  # type: ignore

    min_param: Optional[str] = client_config.min_tokens_param
    if not min_param:
        return False

    base_url = os.environ.get("OPENAI_API_BASE")
    if not base_url:
        logger.warning("OPENAI_API_BASE not set; cannot probe min token parameter.")
        return False
    if not base_url.endswith("/"):
        base_url = base_url + "/"

    url = base_url + (client_config.address_append_value or "chat/completions")
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', '')}",
        "Content-Type": "application/json",
    }

    body = {
        "model": client_config.model,
        "stream": False,
        "max_tokens": 1,
    }
    if client_config.llm_api == "openai_completions":
        body["prompt"] = "Hello"
    else:
        body["messages"] = [{"role": "user", "content": "Hello"}]

    def send_probe_request(param_value):
        test_body = body.copy()
        test_body[min_param] = param_value
        try:
            resp = requests.post(url, headers=headers, json=test_body, timeout=10)
            return 200 <= resp.status_code < 300
        except Exception:
            return False

    if not send_probe_request(1):
        logger.warning(
            f"Server rejected parameter '{min_param}'; falling back to prompt control."
        )
        return False

    if not send_probe_request({"invalid": "type"}):
        return True

    return False


def _initialize_min_tokens_support(benchmark_config: BenchmarkConfig) -> None:
    """Initialize min tokens parameter support by probing the server.

    This function probes the server to determine if it supports the configured
    min_tokens_param. If not supported, it disables the parameter and logs
    a warning about falling back to prompt-based control.
    """
    if benchmark_config.client_config.min_tokens_param:
        is_supported = _probe_min_tokens_param_support(benchmark_config.client_config)
        min_tokens_param = benchmark_config.client_config.min_tokens_param
        if not is_supported:
            object.__setattr__(benchmark_config.client_config, "min_tokens_param", None)
            logger.warning(
                f"min_tokens_param '{min_tokens_param}' not supported by server; switching to prompt-based minimum token control. This will include, in each request, an instruction to generate at least the requested number of tokens. Might lead to inaccurate lengths being generated."
            )
        else:
            logger.info(
                f"min_tokens_param '{min_tokens_param}' supported in request body."
            )


def should_send_new_request(
    service_metrics: ServiceMetrics, num_errored_requests_handled: int
) -> bool:
    """Check if a request should be sent based on the current state of the service."""
    return (service_metrics.num_requests < service_metrics.max_requests) or (
        service_metrics.num_requests >= service_metrics.max_requests
        and num_errored_requests_handled < service_metrics.num_errored_requests
    )


def dispatch_requests(
    input_queue: Queue,
    service_metrics: ServiceMetrics,
    request_generator: BaseRequestGenerator,
    stop_event: threading.Event,
) -> None:
    """Thread function to generate and dispatch requests."""
    num_errored_requests_handled = 0
    # Empirical cumulative schedule in seconds since this thread started handling the first request
    cumulative_empirical_timestamp = 0.0

    while not stop_event.is_set():
        if should_send_new_request(service_metrics, num_errored_requests_handled):
            request_start_time = time.monotonic()

            # check if we should handle error request
            if service_metrics.num_requests >= service_metrics.max_requests:
                num_errored_requests_handled += 1

            # get next request and its dispatch time
            try:
                req_tuple = request_generator.get_request()
                request_config, input_length, output_length, next_timestamp = cast(
                    Tuple["RequestConfig", int, int, float], req_tuple
                )
            except StopIteration as e:
                service_metrics.notify_error(e)
                stop_event.set()
                break
            request_dispatch_delay = request_config.dispatch_delay

            if request_dispatch_delay == -1:
                logger.info(
                    "Benchmark ending early due to stop policy (generator sentinel received)."
                )
                service_metrics.request_stop()
                stop_event.set()
                break
            elif request_dispatch_delay < 0:
                raise ValueError(
                    f"Invalid request dispatch delay '{request_dispatch_delay}' from request metadata."
                )

            # wait for dispatch time
            while not stop_event.is_set():
                elapsed_time = time.monotonic() - request_start_time
                if elapsed_time >= request_dispatch_delay:
                    break
                # remaining sleep time to avoid drift
                remaining_time = request_dispatch_delay - elapsed_time
                if remaining_time > 0:
                    # capped sleep at 100ms
                    sleep_duration = min(remaining_time, 0.1)
                    time.sleep(sleep_duration)

            # if another thread has set the stop event we don't send the request
            if stop_event.is_set():
                continue

            # dispatch
            service_metrics.register_launched_request()
            # record enqueue time (absolute) and empirical scheduled time (cumulative elapsed)
            now_mono = time.monotonic()
            empirical_delay = now_mono - request_start_time
            request_config.enqueued_at = now_mono
            request_config.scheduled_timestamp = cumulative_empirical_timestamp + empirical_delay
            cumulative_empirical_timestamp = request_config.scheduled_timestamp
            input_queue.put(request_config)
            logger.info(
                f"Dispatched request {request_config.id}, in {input_length}, out {output_length}, enqueued at {request_config.enqueued_at}, scheduled at {request_config.scheduled_timestamp}, (mono - start_time = {empirical_delay}), theoretical dispatch delay {request_dispatch_delay}"
            )
        else:
            time.sleep(0.01)


def process_results(
    output_queue: Queue,
    service_metrics: ServiceMetrics,
    generated_responses: List[Response],
    pbar: tqdm,
    stop_event: threading.Event,
) -> None:
    """Thread function to process results from the output queue."""
    # On stop, attempt to drain for a short grace period, then exit
    POLL_TIMEOUT_S = 0.1
    DRAIN_MAX_EMPTY_POLLS = 50  # ~5s
    consecutive_empty_polls_after_stop = 0
    while not stop_event.is_set() or (
        service_metrics.error is None
        and service_metrics.num_completed_requests < service_metrics.num_requests
    ):
        try:
            result = output_queue.get(timeout=POLL_TIMEOUT_S)
            consecutive_empty_polls_after_stop = 0
        except Empty:
            if stop_event.is_set():
                consecutive_empty_polls_after_stop += 1
                if consecutive_empty_polls_after_stop >= DRAIN_MAX_EMPTY_POLLS:
                    logger.info(
                        "Result processor drained for ~%.1fs after stop; exiting.",
                        DRAIN_MAX_EMPTY_POLLS * POLL_TIMEOUT_S,
                    )
                    break
            continue

        if result is None:  # Sentinel check
            break

        request_metrics, generated_response = result
        service_metrics.add_request_metrics(request_metrics)
        if generated_response is not None:
            generated_responses.append(generated_response)

        pbar.update(service_metrics.num_completed_requests - pbar.n)


def run_main_loop(
    benchmark_config: BenchmarkConfig,
    request_generator: BaseRequestGenerator,
    service_metrics: ServiceMetrics,
    generated_responses: List[Response],
    pbar: tqdm,
):
    """Run the main loop for the benchmark."""

    logger.info("Starting the main loop.")

    # Create queues for communication
    input_queue = Queue()
    output_queue = Queue()
    stop_event = threading.Event()

    # Initialize request launcher
    req_launcher = RequestsLauncher(
        client_config=benchmark_config.client_config,
        input_queue=input_queue,
        output_queue=output_queue,
    )

    # Establish a common monotonic epoch for relative timing across processes
    os.environ["VEEKSHA_BENCHMARK_EPOCH_MONO"] = str(time.monotonic())

    # Start the request launcher processes
    req_launcher.start()

    # Create and start producer-consumer threads
    dispatcher_thread = Thread(
        target=dispatch_requests,
        args=(
            input_queue,
            service_metrics,
            request_generator,
            stop_event,
        ),
    )

    processor_thread = Thread(
        target=process_results,
        args=(
            output_queue,
            service_metrics,
            generated_responses,
            pbar,
            stop_event,
        ),
    )

    dispatcher_thread.start()
    processor_thread.start()

    # Monitor and wait for completion
    with service_metrics:
        while not service_metrics.should_stop():
            time.sleep(0.1)
        logger.info("Stopping the main loop.")
        if benchmark_config.request_generator_config.get_type() == RequestGeneratorType.SYNTHETIC:
            request_generator.save_trace(os.path.join(service_metrics.output_dir, "trace.jsonl"))
        if service_metrics.stop_requested and service_metrics.error is None:
            logger.info(
                "Main loop exited due to stop policy; partial metrics will be saved."
            )

    # Signal threads to stop and wait for completion
    stop_event.set()
    dispatcher_thread.join()

    # Wait for all client processes to terminate
    req_launcher.wait_for_clients()

    # Signal the results processor to finish after draining and join it
    output_queue.put(None)
    processor_thread.join()

    pbar.close()
    
    if benchmark_config.vajra_api_url and benchmark_config.vajra_plots_output_dir:
        # save vajra plots
        plot_url = f"{benchmark_config.vajra_api_url}/v1/debug/plot_metrics"
        payload = {"output_dir": benchmark_config.vajra_plots_output_dir}
        headers = {"Content-Type": "application/json"}
        response = requests.post(plot_url, json=payload, headers=headers)
        response.raise_for_status()
        logger.info("Vajra plots saved to %s.", benchmark_config.vajra_plots_output_dir)
        
    if service_metrics.error is None:
        logger.info("Main loop completed.")
    else:
        raise service_metrics.error


def run_benchmark(
    benchmark_config: BenchmarkConfig,
):
    """Run the benchmark and return the in-memory metrics object.

    Args:
        benchmark_config: The benchmark configuration.

    Returns:
        ServiceMetrics containing the collected metrics (including the `MetricStore`).
    """

    prepare_benchmark_output_dir(benchmark_config)
    logger.info(
        f"Benchmark output directory: {benchmark_config.metrics_config.output_dir}"
    )

    setup_api_environment(
        api_key=benchmark_config.api_key,
        api_url=benchmark_config.api_url,
    )

    _initialize_min_tokens_support(benchmark_config)

    generated_responses: List[Response] = []

    assert (
        benchmark_config.client_config.tokenizer is not None
    ), "Tokenizer is required."

    tokenizer = get_tokenizer(
        tokenizer_name=benchmark_config.client_config.tokenizer,
        trust_remote_code=True,
    )

    request_generator_params = {}
    request_generator_config_type = benchmark_config.request_generator_config.get_type()

    if (
        request_generator_config_type == RequestGeneratorType.SYNTHETIC
        or request_generator_config_type == RequestGeneratorType.TRACE
    ):
        request_generator_params = {
            "corpus_lines": load_corpus(),
        }

    seed_manager = SeedManager(benchmark_config.seed)

    request_generator = RequestGeneratorRegistry.get(
        benchmark_config.request_generator_config.get_type(),
        config=benchmark_config.request_generator_config,
        tokenizer=tokenizer,
        client_config=benchmark_config.client_config,
        seed_manager=seed_manager,
        **request_generator_params,
    )

    max_requests = (
        request_generator.num_requests
        if benchmark_config.request_generator_config.get_type()
        == RequestGeneratorType.LMEVAL
        else benchmark_config.max_completed_requests
    )
    pbar = tqdm(total=max_requests)

    service_metrics = ServiceMetrics(
        max_requests=max_requests,
        timeout=benchmark_config.timeout,
        metrics_config=benchmark_config.metrics_config,
    )

    run_main_loop(
        benchmark_config=benchmark_config,
        request_generator=request_generator,
        service_metrics=service_metrics,
        generated_responses=generated_responses,
        pbar=pbar,
    )

    service_metrics.store_output()
    logger.info(f"Metrics stored to {service_metrics.output_dir}")

    store_generated_texts(service_metrics.output_dir, generated_responses)
    
    service_metrics.print_terminal_table()

    # lm-eval specific
    if (
        benchmark_config.request_generator_config.get_type()
        == RequestGeneratorType.LMEVAL
    ):
        request_generator.get_responses(generated_responses)
        lmeval_results = request_generator.evaluate()

        store_lmeval_results(service_metrics.output_dir, lmeval_results)

    return service_metrics


if __name__ == "__main__":
    if platform.system() == "Darwin":
        multiprocessing.set_start_method("fork", force=True)

    benchmark_configs = BenchmarkConfig.create_from_cli_args()

    if len(benchmark_configs) > 1:
        logger.info(
            f"Running {len(benchmark_configs)} benchmark configurations sequentially."
        )

    for i, benchmark_config in enumerate(benchmark_configs):
        print(f"Running benchmark with config: {benchmark_config}")
        if len(benchmark_configs) > 1:
            logger.info(f"Starting benchmark {i+1}/{len(benchmark_configs)}")

        random.seed(benchmark_config.seed)
        service_metrics = run_benchmark(benchmark_config=benchmark_config)

        if len(benchmark_configs) > 1:
            logger.info(f"Completed benchmark {i+1}/{len(benchmark_configs)}")

    logger.info("All benchmarks completed.")
