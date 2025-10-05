from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import rekha as rk
import wandb
from ddsketch import DDSketch

from veeksha.logger import init_logger

logger = init_logger(__name__)


SUMMARY_PERCENTILES = [0.5, 0.9, 0.99]


class CDFSketch:
    def __init__(
        self,
        metric_name: str,
        should_write_to_wandb: bool = True,
        display_unit_scale: float = 1.0,
        display_unit_suffix: str = "",
    ) -> None:
        # metrics are a data series of two-dimensional (x, y) datapoints
        self.sketch = DDSketch(relative_accuracy=0.001)
        # column name
        self.metric_name = metric_name

        # most recently collected y datapoint for incremental updates
        # to aid incremental updates to y datapoints
        self.last_data = 0

        self.should_write_to_wandb = should_write_to_wandb
        # Display-only scaling. Internal values remain unscaled (seconds, tokens, etc.).
        self.display_unit_scale = display_unit_scale
        self.display_unit_suffix = display_unit_suffix

    def __len__(self):
        return int(self.sketch.count)

    # add a new x, y datapoint
    def put(self, data: float) -> None:
        self.last_data = data
        self.sketch.add(data)

    def extend(self, values: List[float]):
        for value in values:
            self.put(value)

    # add a new datapoint as an incremental (delta) update to
    # recently collected datapoint
    def put_delta(self, delta: float) -> None:
        data = self.last_data + delta
        self.put(data)

    def print_distribution_stats(self, plot_name: str) -> None:
        if self.sketch._count == 0:
            return

        logger.info(
            f"{plot_name}: {self.metric_name} stats:"
            f" min: {self.sketch._min},"
            f" max: {self.sketch._max},"
            f" mean: {self.sketch.avg},"
            f" 25th percentile: {self.sketch.get_quantile_value(0.25)},"
            f" median: {self.sketch.get_quantile_value(0.5)},"
            f" 75th percentile: {self.sketch.get_quantile_value(0.75)},"
            f" 95th percentile: {self.sketch.get_quantile_value(0.95)},"
            f" 99th percentile: {self.sketch.get_quantile_value(0.99)}"
            f" 99.9th percentile: {self.sketch.get_quantile_value(0.999)}"
        )
        if wandb.run and self.should_write_to_wandb:
            wandb.log(
                {
                    f"{plot_name}_min": self.sketch._min,
                    f"{plot_name}_max": self.sketch._max,
                    f"{plot_name}_mean": self.sketch.avg,
                    f"{plot_name}_25th_percentile": self.sketch.get_quantile_value(
                        0.25
                    ),
                    f"{plot_name}_median": self.sketch.get_quantile_value(0.5),
                    f"{plot_name}_75th_percentile": self.sketch.get_quantile_value(
                        0.75
                    ),
                    f"{plot_name}_95th_percentile": self.sketch.get_quantile_value(
                        0.95
                    ),
                    f"{plot_name}_99th_percentile": self.sketch.get_quantile_value(
                        0.99
                    ),
                    f"{plot_name}_99.9th_percentile": self.sketch.get_quantile_value(
                        0.999
                    ),
                },
                step=0,
            )

    def _to_df(self) -> pd.DataFrame:
        # get quantiles at 1% intervals
        quantiles = np.linspace(0, 1, 101)
        # get quantile values and apply display scaling
        quantile_values = []
        for q in quantiles:
            v = self.sketch.get_quantile_value(q)
            v = 0.0 if v is None else v
            quantile_values.append(v * self.display_unit_scale)
        # create dataframe with display name
        display_name = f"{self.metric_name}{self.display_unit_suffix}"
        df = pd.DataFrame({"cdf": quantiles, display_name: quantile_values})

        return df

    @property
    def sum(self) -> float:
        return self.sketch.sum

    def _save_df(self, df: pd.DataFrame, path: str, plot_name: str) -> None:
        df.to_csv(f"{path}/{plot_name}.csv")

        if wandb.run and self.should_write_to_wandb:
            wand_table = wandb.Table(dataframe=df)
            wandb.log({f"{plot_name}_table": wand_table}, step=0)

    def plot_cdf(
        self, path: str, plot_name: str, x_axis_label: Optional[str] = None
    ) -> None:
        if self.sketch._count == 0:
            return

        if x_axis_label is None:
            x_axis_label = f"{self.metric_name}{self.display_unit_suffix}"

        df = self._to_df()
        display_name = f"{self.metric_name}{self.display_unit_suffix}"

        fig = rk.line(
            df,
            x=display_name,
            y="cdf",
            markers=True,
            labels={display_name: x_axis_label, "cdf": "CDF"},
        )

        if wandb.run and self.should_write_to_wandb:
            wandb_df = df.copy()
            # rename the display column to x_axis_label for wandb plot readability
            wandb_df = wandb_df.rename(columns={display_name: x_axis_label})

            wandb.log(
                {
                    f"{plot_name}_cdf": wandb.plot.line(
                        table=wandb.Table(dataframe=wandb_df),
                        x=x_axis_label,
                        y="cdf",
                        title=plot_name,
                    )
                },
                step=0,
            )

        fig.save(f"{path}/{plot_name}.png")
        self._save_df(df, path, plot_name)

    def get_summary(self) -> Dict[str, float]:
        display_name = f"{self.metric_name}{self.display_unit_suffix}"
        if self.sketch.count > 0:
            avg = self.sketch.avg
            avg = 0.0 if avg is None else avg
            summary = {f"{display_name} (Mean)": avg * self.display_unit_scale}
            for p in SUMMARY_PERCENTILES:
                val = self.sketch.get_quantile_value(p)
                val = 0.0 if val is None else val
                summary[f"{display_name} (P{int(p * 100)})"] = (
                    val * self.display_unit_scale
                )
            return summary
        return {
            f"{display_name} (Mean)": 0,
            **{f"{display_name} (P{int(p * 100)})": 0 for p in SUMMARY_PERCENTILES},
        }

    def to_csv_row(self) -> str:
        return ",".join([f"{v:.5f}" for v in self.get_summary().values()])

    def get_csv_header(self) -> str:
        return ",".join([f"{k}" for k in self.get_summary().keys()])

    def __str__(self) -> str:
        summary_str = ", ".join(
            [f"{k}: {v:.5f}" for k, v in self.get_summary().items()]
        )
        # remove the repeated metric name
        summary_str = summary_str.replace(self.metric_name, "")
        summary_str = summary_str.replace("(", "").replace(")", "").strip()
        # remove double spaces
        summary_str = " ".join(summary_str.split())
        return f"{self.metric_name} - {summary_str}"

    def __repr__(self) -> str:
        return self.__str__()
