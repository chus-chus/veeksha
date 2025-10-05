import json
import os
from collections import defaultdict
from itertools import accumulate
from typing import DefaultDict, Dict, Optional

import pandas as pd
import rekha as rk
import wandb

from veeksha.config.metrics import MetricsConfig
from veeksha.logger import init_logger
from veeksha.metrics.cdf_sketch import CDFSketch
from veeksha.metrics.metric_utils import (
    find_min_tbt_deadline_to_meet,
    get_deadline_miss_rate_for_target_tbt_values,
    get_request_level_deadline_miss_rate,
    get_throughput_metrics,
)
from veeksha.metrics.request_level_metrics import RequestLevelMetrics
from veeksha.metrics.request_metrics import RequestMetrics

logger = init_logger(__name__)


TARGET_TBT_RANGE = [i * 0.001 for i in range(1, 101)]
QUANTILE_FOR_DEADLINE_MISS_RATE = 0.99


class MetricStore:
    def __init__(
        self,
        timeout: float,
        max_requests: int,
        metrics_config: MetricsConfig,
    ) -> None:
        self.timeout = timeout
        self.max_requests = max_requests

        self.num_requests: int = 0
        self.num_errored_requests: int = 0
        self.num_completed_requests: int = 0
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.error_code_freq: DefaultDict[int, int] = defaultdict(int)
        self.ttft_deadline: float = metrics_config.deadline_report.ttft_deadline
        self.tbt_deadline: float = metrics_config.deadline_report.tbt_deadline
        self.target_deadline_miss_rate: float = (
            metrics_config.deadline_report.target_deadline_miss_rate
        )
        self.service_level_missed_deadlines: int = 0
        self.service_level_total_deadlines: int = 0
        self.should_write_metrics_to_wandb: bool = (
            metrics_config.should_write_metrics_to_wandb
        )
        self.wandb_project: Optional[str] = metrics_config.wandb_project
        self.wandb_group: Optional[str] = metrics_config.wandb_group
        self.wandb_run_name: Optional[str] = metrics_config.wandb_run_name

        self.request_level_metrics = RequestLevelMetrics(
            deadline_config=metrics_config.deadline_report,
        )

        self.summaries: Dict[str, CDFSketch] = {
            "num_prompt_tokens": CDFSketch(
                "Number of Prompt Tokens", self.should_write_metrics_to_wandb
            ),
            "num_output_tokens": CDFSketch(
                "Number of Output Tokens", self.should_write_metrics_to_wandb
            ),
            "num_total_tokens": CDFSketch(
                "Number of Total Tokens", self.should_write_metrics_to_wandb
            ),
            "tpot": CDFSketch(
                "Time per Output Token",
                self.should_write_metrics_to_wandb,
                display_unit_scale=1e3,
                display_unit_suffix=" (ms)",
            ),
            "ttft": CDFSketch(
                "Time to First Token",
                self.should_write_metrics_to_wandb,
                display_unit_scale=1e3,
                display_unit_suffix=" (ms)",
            ),
            "tbt": CDFSketch(
                "Time Between Tokens",
                self.should_write_metrics_to_wandb,
                display_unit_scale=1e3,
                display_unit_suffix=" (ms)",
            ),
            "end_to_end_latency": CDFSketch(
                "End to End Latency",
                self.should_write_metrics_to_wandb,
                display_unit_scale=1e3,
                display_unit_suffix=" (ms)",
            ),
            "normalized_end_to_end_latency": CDFSketch(
                "Normalized End to End Latency",
                self.should_write_metrics_to_wandb,
                display_unit_scale=1e3,
                display_unit_suffix=" (ms)",
            ),
            "output_throughput": CDFSketch(
                "Output Throughput", self.should_write_metrics_to_wandb
            ),
            "deadline_miss_rate": CDFSketch(
                f"Deadline Miss Rate with {self.tbt_deadline}s TBT Deadline, {self.ttft_deadline}s TTFT Deadline",
                self.should_write_metrics_to_wandb,
            ),
            "min_tbt_deadline_to_meet": CDFSketch(
                f"Min Deadline to Meet Target Deadline Miss Rate of {self.target_deadline_miss_rate * 100}%",
                self.should_write_metrics_to_wandb,
                display_unit_scale=1e3,
                display_unit_suffix=" (ms)",
            ),
        }

        self._init_wandb()

    def _init_wandb(self):
        if not self.should_write_metrics_to_wandb:
            logger.info("wandb disabled; not initialized")
            return

        wandb.init(
            project=self.wandb_project,
            group=self.wandb_group,
            name=self.wandb_run_name,
            config={
                "timeout": self.timeout,
                "max_requests": self.max_requests,
                "ttft_deadline": self.ttft_deadline,
                "tbt_deadline": self.tbt_deadline,
                "target_deadline_miss_rate": self.target_deadline_miss_rate,
            },
        )
        logger.info("wandb enabled")

    @property
    def error_rate(self):
        return (
            self.num_errored_requests / self.num_requests
            if self.num_requests > 0
            else 0.0
        )

    def register_launched_request(self):
        self.num_requests += 1

    def add_request_metrics(self, request_metrics: RequestMetrics):
        if request_metrics.error_code:
            # do not add errored requests to metric sketches
            self.error_code_freq[request_metrics.error_code] += 1
            self.num_errored_requests += 1
            return
        else:
            self.num_completed_requests += 1

        for metric_name, cdf_sketch in self.summaries.items():
            if metric_name == "tbt":
                cdf_sketch.extend(request_metrics.inter_token_times[1:])
            elif metric_name == "deadline_miss_rate":
                (
                    deadline_miss_rate,
                    missed_deadlines,
                    total_deadlines,
                ) = get_request_level_deadline_miss_rate(
                    inter_token_times=request_metrics.inter_token_times,
                    ttft_deadline=self.ttft_deadline,
                    tbt_deadline=self.tbt_deadline,
                )
                cdf_sketch.put(deadline_miss_rate)
                self.service_level_missed_deadlines += missed_deadlines
                self.service_level_total_deadlines += total_deadlines
            elif metric_name == "min_tbt_deadline_to_meet":
                cdf_sketch.put(
                    find_min_tbt_deadline_to_meet(
                        inter_token_times=request_metrics.inter_token_times,
                        ttft_deadline=self.ttft_deadline,
                        target_deadline_miss_rate=self.target_deadline_miss_rate,
                    )
                )
            else:
                cdf_sketch.put(getattr(request_metrics, metric_name))

        self.request_level_metrics.put(request_metrics)

    def get_aggregated_summary(self) -> Dict[str, float]:
        return {
            "Number of Requests": self.num_requests,
            "Number of Errored Requests": self.num_errored_requests,
            "Number of Completed Requests": self.num_completed_requests,
            "Error Rate": self.error_rate,
            "Deadline Miss Rate": (
                self.service_level_missed_deadlines / self.service_level_total_deadlines
                if self.service_level_total_deadlines > 0
                else 0.0
            ),
        }

    def get_summary(self) -> Dict[str, float]:
        perf_summary = {}

        for cdf_sketch in self.summaries.values():
            perf_summary.update(cdf_sketch.get_summary())

        return {
            **self.get_aggregated_summary(),
            **perf_summary,
        }

    def get_terminal_table(self, only: Optional[list] = None) -> str:
        """Return a formatted ASCII table of metric summaries for the terminal.

        Columns: Metric, Min, Max, Mean, Median, P90, P99.

        Notes:
            - Time-based metrics are displayed in milliseconds via the
              CDFSketch display scaling (values remain stored in seconds).
            - Non-time metrics retain their natural units (e.g., tokens, tok/s).
        """
        headers = ["Metric", "Min", "Max", "Mean", "Median", "P90", "P99"]

        # Desired default ordering and labels (time metrics reported in ms)
        default_keys_in_order = [
            "end_to_end_latency",
            "ttft",
            "tpot",
            "tbt",
            "num_prompt_tokens",
            "num_output_tokens",
            "num_total_tokens",
        ]
        key_to_label = {
            "end_to_end_latency": "e2e latency (ms)",
            "ttft": "time to first token (ms)",
            "tpot": "time per output token (ms)",
            "tbt": "time between tokens (ms)",
            "num_prompt_tokens": "num prefill tokens (count)",
            "num_output_tokens": "num decode tokens (count)",
            "num_total_tokens": "num total tokens (count)",
        }

        def fmt(value: float) -> str:
            return f"{value:,.3f}"

        rows = []
        keys = default_keys_in_order if only is None else [k for k in only]
        for key in keys:
            sketch = self.summaries.get(key)
            if sketch is None:
                continue
            # skip empty sketches
            if len(sketch) == 0:
                continue

            # derive display name and scale
            display_name = key_to_label.get(
                key, f"{sketch.metric_name}{sketch.display_unit_suffix}"
            )
            scale = getattr(sketch, "display_unit_scale", 1.0)

            # pull stats from DDSketch; coerce None -> 0.0
            min_v = sketch.sketch._min if sketch.sketch._min is not None else 0.0
            max_v = sketch.sketch._max if sketch.sketch._max is not None else 0.0
            mean_v = sketch.sketch.avg if sketch.sketch.avg is not None else 0.0
            p50_v = sketch.sketch.get_quantile_value(0.5) or 0.0
            p90_v = sketch.sketch.get_quantile_value(0.9) or 0.0
            p99_v = sketch.sketch.get_quantile_value(0.99) or 0.0

            # apply display scaling (e.g., seconds -> ms)
            min_v *= scale
            max_v *= scale
            mean_v *= scale
            p50_v *= scale
            p90_v *= scale
            p99_v *= scale

            rows.append(
                [
                    display_name,
                    fmt(min_v),
                    fmt(max_v),
                    fmt(mean_v),
                    fmt(p50_v),
                    fmt(p90_v),
                    fmt(p99_v),
                ]
            )

        # compute column widths
        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(str(cell)))

        # build table string
        def pad(cell: str, width: int) -> str:
            return cell + " " * (width - len(cell))

        line_sep = "+" + "+".join(["-" * (w + 2) for w in col_widths]) + "+"
        out_lines = [line_sep]
        header_line = "| " + " | ".join(
            [pad(h, col_widths[i]) for i, h in enumerate(headers)]
        ) + " |"
        out_lines.append(header_line)
        out_lines.append(line_sep)
        for row in rows:
            out_lines.append(
                "| "
                + " | ".join([pad(str(c), col_widths[i]) for i, c in enumerate(row)])
                + " |"
            )
        out_lines.append(line_sep)

        return "\n".join(out_lines)

    def print_terminal_table(self, only: Optional[list] = None) -> None:
        """Print the terminal table of metric summaries."""
        print(self.get_terminal_table(only=only))

    def store_output(self, output_dir: str):
        perf_csv_path = os.path.join(output_dir, "perf_metrics.csv")
        summary_stats_path = os.path.join(output_dir, "summary_stats.json")

        # store request level metrics
        self.request_level_metrics.save(output_dir)

        # store metric objects
        for metric_name, metric_summary in self.summaries.items():
            metric_summary._save_df(metric_summary._to_df(), output_dir, metric_name)
            metric_summary.plot_cdf(output_dir, metric_name, metric_name)

        # store service level deadline stats
        with open(os.path.join(output_dir, "service_level_metrics.json"), "w") as f:
            json.dump(
                {
                    "service_level_missed_deadlines": self.service_level_missed_deadlines,
                    "service_level_total_deadlines": self.service_level_total_deadlines,
                    "service_level_deadline_miss_rate": (
                        self.service_level_missed_deadlines
                        / self.service_level_total_deadlines
                        if self.service_level_total_deadlines > 0
                        else 0.0
                    ),
                },
                f,
            )

        # store performance metrics
        perf_header = self.summaries["num_prompt_tokens"].get_csv_header()
        perf_rows = [perf_header]
        for cdf_sketch in self.summaries.values():
            perf_rows.append(cdf_sketch.to_csv_row())

        with open(perf_csv_path, "w") as f:
            f.write("\n".join(perf_rows))

        # store summary stats
        with open(summary_stats_path, "w") as f:
            json.dump(
                {**self.get_summary(), "error_code_freq": dict(self.error_code_freq)}, f
            )

        # store additional outputs
        self.store_additional_outputs(output_dir)

    def store_additional_outputs(self, output_dir: str):
        self.store_deadline_miss_rate_for_target_tbt(output_dir)
        self.store_throughput_metrics(output_dir)
        self.store_ttft_violin_plots(output_dir)
        self.store_generation_stalls(output_dir)

    def store_deadline_miss_rate_for_target_tbt(self, output_dir: str):
        # plot deadline miss rate for target TBT values
        deadline_miss_rate_for_target_tbt_values = (
            get_deadline_miss_rate_for_target_tbt_values(
                tbt_times=self.request_level_metrics.tbt,
                target_tbt_deadline_array=TARGET_TBT_RANGE,
                quantile=QUANTILE_FOR_DEADLINE_MISS_RATE,
            )
        )

        percentile_value = int(QUANTILE_FOR_DEADLINE_MISS_RATE * 100)
        x_axis_label = "Target TBT (ms)"
        y_axis_label = f"Miss Rate P({percentile_value})"

        data = {
            x_axis_label: [int(i * 1e3) for i in TARGET_TBT_RANGE],
            y_axis_label: deadline_miss_rate_for_target_tbt_values,
        }
        df = pd.DataFrame(data)

        with open(
            os.path.join(
                output_dir,
                f"p{percentile_value}_deadline_miss_rate_for_target_tbt_values.json",
            ),
            "w",
        ) as f:
            json.dump(data, f)

        if self.should_write_metrics_to_wandb and wandb.run:
            # plot deadline miss rate for target TBT values
            wandb.log(
                {
                    f"p{percentile_value}_deadline_miss_rate": wandb.plot.line(
                        table=wandb.Table(dataframe=df),
                        x=x_axis_label,
                        y=y_axis_label,
                        title="Deadline Miss Rate for Target TBT Values",
                    )
                },
                step=0,
            )

    def store_throughput_metrics(self, output_dir: str):
        (
            tpot_based_throughput,
            tbt_based_throughput,
            deadline_based_throughput,
        ) = get_throughput_metrics(
            self.request_level_metrics.tpot, self.request_level_metrics.tbt
        )

        throughput_metrics = {
            "tpot_based_throughput": tpot_based_throughput,
            "tbt_based_throughput": tbt_based_throughput,
            "deadline_based_throughput": deadline_based_throughput,
        }

        with open(os.path.join(output_dir, "throughput_metrics.json"), "w") as f:
            json.dump(throughput_metrics, f)

        # log plot of throughput metrics to wandb
        data = {
            "Metric Type": ["TPOT Based", "TBT Based", "Deadline Based"],
            "Throughput (tok/s)": [
                tpot_based_throughput,
                tbt_based_throughput,
                deadline_based_throughput,
            ],
        }
        df = pd.DataFrame(data)

        if self.should_write_metrics_to_wandb and wandb.run:
            wandb.log(
                {
                    "throughput_metrics": wandb.plot.bar(
                        table=wandb.Table(dataframe=df),
                        label="Metric Type",
                        value="Throughput (tok/s)",
                        title="Token Throughput",
                    )
                },
                step=0,
            )

    def store_ttft_violin_plots(self, output_dir: str):
        data = {}
        for i, ttft in enumerate(self.request_level_metrics.ttft):
            if str(self.request_level_metrics.num_prompt_tokens[i]) not in data:
                data[str(self.request_level_metrics.num_prompt_tokens[i])] = []
            # Convert to ms for reporting
            data[str(self.request_level_metrics.num_prompt_tokens[i])].append(ttft * 1e3)
        df = pd.DataFrame(
            {
                "ttft": [ttft for ttfts in data.values() for ttft in ttfts],
                "prompt_length": [
                    prompt_length
                    for prompt_length in data.keys()
                    for _ in data[prompt_length]
                ],
            }
        )
        df = df.sort_values("prompt_length", key=lambda x: x.astype(int))
        fig = rk.box(
            df,
            x="prompt_length",
            y="ttft",
            labels={"prompt_length": "Number of Prompt Tokens", "ttft": "TTFT (ms)"},
        )
        fig.save(os.path.join(output_dir, "ttft_violin_plot.png"))
        if self.should_write_metrics_to_wandb and wandb.run:
            wandb.log(
                {
                    "ttft_violin_plot": wandb.Image(
                        os.path.join(output_dir, "ttft_violin_plot.png")
                    )
                }
            )
            wandb.log({"ttft_violin_data": wandb.Table(dataframe=df)})

    def store_generation_stalls(self, output_dir: str, request_idx: int = 0):
        # just generate for 1 request for now
        if request_idx >= len(self.request_level_metrics.ttft):
            return
        token_generated_times = [
            self.request_level_metrics.ttft[request_idx]
        ] + self.request_level_metrics.tbt[request_idx]
        token_generated_times = list(accumulate(token_generated_times))
        tokens_generated = list(range(1, len(token_generated_times) + 1))
        data = {
            "Time (s)": token_generated_times,
            "Tokens Generated": tokens_generated,
        }
        fig = rk.line(
            pd.DataFrame(data),
            x="Time (s)",
            y="Tokens Generated",
            title="Tokens Generated vs Time",
        )
        fig.save(os.path.join(output_dir, "tokens_generated_vs_time.png"))
        if self.should_write_metrics_to_wandb and wandb.run:
            wandb.log(
                {
                    "tokens_generated_vs_time": wandb.Image(
                        os.path.join(output_dir, "tokens_generated_vs_time.png")
                    )
                }
            )
