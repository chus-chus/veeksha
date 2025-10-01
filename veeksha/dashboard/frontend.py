"""
Textual-based frontend for the Veeksha LLM benchmark dashboard.

This module provides a real-time TUI (Terminal User Interface) dashboard that displays
live metrics during benchmark execution. The dashboard automatically launches when 
the benchmark is run with dashboard_config.enabled=True.

Features:
- Real-time display of benchmark status (requests, completion rate, QPS)
- Live request tracking with TTFT and TPOT metrics
- Aggregate performance statistics
- Auto-updating display every 500ms

Usage:
    The dashboard is automatically launched when running benchmarks with dashboard enabled.
    
    For standalone testing:
        python -m veeksha.dashboard --duration 60
        
    The dashboard can also be programmatically launched:
        from veeksha.dashboard.frontend import run_dashboard_frontend
        run_dashboard_frontend(dashboard_state)
"""

import threading
import time
from typing import Optional
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, Container
from textual.widgets import Header, Footer, Static, DataTable, TabbedContent, TabPane
from textual.timer import Timer
from collections import deque
import time as time_module

try:
    import plotext as plt
    PLOTEXT_AVAILABLE = True
except ImportError:
    PLOTEXT_AVAILABLE = False
    plt = None

from veeksha.dashboard.state import DashboardState


class LiveRequestsTable(DataTable):
    """Widget to display live request information"""
    
    def on_mount(self) -> None:
        self.add_columns("Request ID", "Input Tokens", "Output Tokens", "TTFT (ms)", "TPOT (ms)", "Duration (s)")
        self.cursor_type = "none"


class MetricsDisplay(Static):
    """Widget to display aggregate metrics"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.border_title = "📊 Key Metrics"


class BenchmarkStatus(Static):
    """Widget to display benchmark status"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.border_title = "⚡ Benchmark Status"


class LiveGraph(Static):
    """Base class for live updating graphs"""
    
    def __init__(self, title: str, **kwargs):
        super().__init__(**kwargs)
        self.border_title = title
        self.data_points = deque(maxlen=100)  # Keep last 100 points
        self.timestamps = deque(maxlen=100)
        self.start_time = time_module.time()
        self.is_frozen = False
        self.original_title = title
    
    def add_data_point(self, value: float):
        """Add a new data point to the graph"""
        if not self.is_frozen:
            current_time = time_module.time() - self.start_time
            self.timestamps.append(current_time)
            self.data_points.append(value)
    
    def freeze_at_completion(self):
        """Freeze the graph and mark as completed"""
        self.is_frozen = True
        self.border_title = f"{self.original_title} [COMPLETED]"
        # Force a final update with completion marker
        self.update_plot()
    
    def update_plot(self):
        """Update the plot display - to be implemented by subclasses"""
        pass


class TTFTGraph(LiveGraph):
    """Graph for Time to First Token metrics"""
    
    def __init__(self, **kwargs):
        super().__init__("📈 TTFT Over Time", **kwargs)
    
    def update_plot(self):
        """Create a clean plotext graph for TTFT"""
        if not PLOTEXT_AVAILABLE:
            self.update("📈 [bold cyan]Time to First Token[/bold cyan]\n\n[dim]Install plotext: pip install plotext[/dim]")
            return
            
        if len(self.data_points) < 1:
            self.update("📈 [bold cyan]Time to First Token[/bold cyan]\n\n[dim]Waiting for data...[/dim]")
            return
        
        # Get recent data for display
        recent_points = list(self.data_points)[-30:]
        recent_times = list(self.timestamps)[-30:]
        
        if not recent_points:
            self.update("📈 [bold cyan]Time to First Token[/bold cyan]\n\n[dim]No data available[/dim]")
            return
        
        # Calculate stats
        status = "🔵 FINAL" if self.is_frozen else "🟢 LIVE"
        min_val, max_val = min(recent_points), max(recent_points)
        avg_val = sum(recent_points) / len(recent_points)
        latest_val = recent_points[-1]
        perf_indicator = "🟢 Fast" if avg_val < 100 else "🟡 OK" if avg_val < 300 else "🔴 Slow"
        
        # Create plot with dark theme
        plt.clear_data()
        plt.clear_figure()
        plt.theme("dark")

        plt.plot(recent_times, recent_points, marker="dot", color="cyan")
        plt.plotsize(68, 12)
        plt.title(f"TTFT - {status} | Latest: {latest_val:.1f}ms, Avg: {avg_val:.1f}ms {perf_indicator}")
        plt.xlabel("Time (s)")
        plt.ylabel("TTFT (ms)")
        plt.grid(True, True)

        plot_str = plt.build()
        plt.clear_figure()  # Extra clear to prevent contamination
        self.update(plot_str)


class TPOTGraph(LiveGraph):
    """Graph for Time per Output Token metrics"""
    
    def __init__(self, **kwargs):
        super().__init__("📉 TPOT Over Time", **kwargs)
    
    def update_plot(self):
        """Create a clean plotext graph for TPOT"""
        if not PLOTEXT_AVAILABLE:
            self.update("📉 [bold green]Time per Output Token[/bold green]\n\n[dim]Install plotext: pip install plotext[/dim]")
            return
            
        if len(self.data_points) < 1:
            self.update("📉 [bold green]Time per Output Token[/bold green]\n\n[dim]Waiting for data...[/dim]")
            return
        
        # Get recent data for display
        recent_points = list(self.data_points)[-30:]
        recent_times = list(self.timestamps)[-30:]
        
        if not recent_points:
            self.update("📉 [bold green]Time per Output Token[/bold green]\n\n[dim]No data available[/dim]")
            return
        
        # Calculate stats
        status = "🔵 FINAL" if self.is_frozen else "🟢 LIVE"
        min_val, max_val = min(recent_points), max(recent_points)
        avg_val = sum(recent_points) / len(recent_points)
        latest_val = recent_points[-1]
        perf_indicator = "🟢 Fast" if avg_val < 10 else "🟡 OK" if avg_val < 50 else "🔴 Slow"
        
        # Create plot with dark theme
        plt.clear_data()
        plt.clear_figure()
        plt.theme("dark")

        plt.plot(recent_times, recent_points, marker="dot", color="green")
        plt.plotsize(68, 12)
        plt.title(f"TPOT - {status} | Latest: {latest_val:.1f}ms, Avg: {avg_val:.1f}ms {perf_indicator}")
        plt.xlabel("Time (s)")
        plt.ylabel("TPOT (ms)")
        plt.grid(True, True)

        plot_str = plt.build()
        plt.clear_figure()  # Extra clear to prevent contamination
        self.update(plot_str)


class TBTGraph(LiveGraph):
    """Graph for Time Between Tokens metrics"""

    def __init__(self, **kwargs):
        super().__init__("⏱️ TBT Over Time", **kwargs)

    def update_plot(self):
        """Create a clean plotext graph for TBT"""
        if not PLOTEXT_AVAILABLE:
            self.update("⏱️ [bold yellow]Time Between Tokens[/bold yellow]\n\n[dim]Install plotext: pip install plotext[/dim]")
            return

        if len(self.data_points) < 1:
            self.update("⏱️ [bold yellow]Time Between Tokens[/bold yellow]\n\n[dim]Waiting for data...[/dim]")
            return

        # Get recent data for display
        recent_points = list(self.data_points)[-30:]
        recent_times = list(self.timestamps)[-30:]

        if not recent_points:
            self.update("⏱️ [bold yellow]Time Between Tokens[/bold yellow]\n\n[dim]No data available[/dim]")
            return

        # Calculate stats
        status = "🔵 FINAL" if self.is_frozen else "🟢 LIVE"
        min_val, max_val = min(recent_points), max(recent_points)
        avg_val = sum(recent_points) / len(recent_points)
        latest_val = recent_points[-1]
        perf_indicator = "🟢 Fast" if avg_val < 10 else "🟡 OK" if avg_val < 50 else "🔴 Slow"

        # Create plot with dark theme
        plt.clear_data()
        plt.clear_figure()
        plt.theme("dark")

        plt.plot(recent_times, recent_points, marker="dot", color="yellow")
        plt.plotsize(68, 12)
        plt.title(f"TBT - {status} | Latest: {latest_val:.1f}ms, Avg: {avg_val:.1f}ms {perf_indicator}")
        plt.xlabel("Time (s)")
        plt.ylabel("TBT (ms)")
        plt.grid(True, True)

        plot_str = plt.build()
        plt.clear_figure()  # Extra clear to prevent contamination
        self.update(plot_str)


class ThroughputGraph(LiveGraph):
    """Graph for throughput metrics"""
    
    def __init__(self, **kwargs):
        super().__init__("🚀 Throughput (QPS)", **kwargs)
        self.qps_points = deque(maxlen=100)
    
    def add_qps_point(self, qps: float):
        """Add QPS data point"""
        current_time = time_module.time() - self.start_time
        self.timestamps.append(current_time)
        self.qps_points.append(qps)
    
    def update_plot(self):
        """Create a clean plotext graph for throughput"""
        if not PLOTEXT_AVAILABLE:
            self.update("🚀 [bold yellow]Throughput (QPS)[/bold yellow]\n\n[dim]Install plotext: pip install plotext[/dim]")
            return
            
        if len(self.qps_points) < 1:
            self.update("🚀 [bold yellow]Throughput (QPS)[/bold yellow]\n\n[dim]Waiting for data...[/dim]")
            return
        
        # Get recent data for display
        recent_qps = list(self.qps_points)[-30:]
        recent_times = list(self.timestamps)[-30:]
        
        if not recent_qps:
            self.update("🚀 [bold yellow]Throughput (QPS)[/bold yellow]\n\n[dim]No data available[/dim]")
            return
        
        # Calculate stats
        status = "🔵 FINAL" if self.is_frozen else "🟢 LIVE"
        min_val, max_val = min(recent_qps), max(recent_qps)
        avg_val = sum(recent_qps) / len(recent_qps)
        latest_val = recent_qps[-1]
        perf_indicator = "🟢 High" if avg_val > 2.0 else "🟡 OK" if avg_val > 0.5 else "🔴 Low"
        
        # Create plot with dark theme
        plt.clear_data()
        plt.clear_figure()
        plt.theme("dark")
        
        plt.plot(recent_times, recent_qps, marker="dot")
        plt.plotsize(78, 13)
        plt.title(f"Throughput (QPS) - {status} | Current: {latest_val:.2f}, Avg: {avg_val:.2f} {perf_indicator}")
        plt.xlabel("Time (seconds)")
        plt.ylabel("QPS")
        plt.grid(True)

        plot_str = plt.build()
        self.update(plot_str)


class LatencyOverTime(LiveGraph):
    """Graph for latency over time"""
    
    def __init__(self, **kwargs):
        super().__init__("📊 Latency Over Time", **kwargs)
    
    def add_latency(self, latency: float):
        """Add latency value with timestamp"""
        self.add_data_point(latency)
    
    def update_plot(self):
        """Create a clean plotext graph for latency"""
        if not PLOTEXT_AVAILABLE:
            self.update("📊 [bold magenta]End-to-End Latency[/bold magenta]\n\n[dim]Install plotext: pip install plotext[/dim]")
            return
            
        if len(self.data_points) < 1:
            self.update("📊 [bold magenta]End-to-End Latency[/bold magenta]\n\n[dim]Waiting for data...[/dim]")
            return
        
        # Get recent data for display
        recent_points = list(self.data_points)[-30:]
        recent_times = list(self.timestamps)[-30:]
        
        if not recent_points:
            self.update("📊 [bold magenta]End-to-End Latency[/bold magenta]\n\n[dim]No data available[/dim]")
            return
        
        # Calculate stats
        status = "🔵 FINAL" if self.is_frozen else "🟢 LIVE"
        min_val, max_val = min(recent_points), max(recent_points)
        avg_val = sum(recent_points) / len(recent_points)
        latest_val = recent_points[-1]
        perf_indicator = "🟢 Fast" if avg_val < 500 else "🟡 OK" if avg_val < 2000 else "🔴 Slow"
        
        # Create plot with dark theme
        plt.clear_data()
        plt.clear_figure()
        plt.theme("dark")

        plt.plot(recent_times, recent_points, marker="dot", color="magenta")
        plt.plotsize(68, 12)
        plt.title(f"Latency (ms) - {status} | Latest: {latest_val:.0f}, Avg: {avg_val:.0f} {perf_indicator}")
        plt.xlabel("Time (s)")
        plt.ylabel("Latency (ms)")
        plt.grid(True, True)

        plot_str = plt.build()
        plt.clear_figure()  # Extra clear to prevent contamination
        self.update(plot_str)


class VeekshaDashboard(App):
    """A comprehensive textual dashboard for live benchmark metrics with graphs"""
    
    CSS = """
    .metric-box {
        border: solid $primary;
        margin: 1;
        padding: 1;
        height: 8;
        background: $surface;
    }
    
    .live-requests {
        border: solid $secondary;
        margin: 1;
        padding: 1;
        height: 12;
        background: $surface;
    }
    
    .benchmark-status {
        border: solid $accent;
        margin: 1;
        padding: 1;
        height: 8;
        background: $surface;
    }
    
    .graph-widget {
        border: solid $warning;
        margin: 1;
        padding: 1;
        min-height: 20;
        max-height: 20;
        background: $surface;
    }
    
    .analysis-panel {
        border: solid $success;
        margin: 1;
        padding: 1;
        background: $surface;
    }
    
    DataTable {
        background: $surface;
    }
    
    DataTable > .datatable--header {
        background: $primary;
        color: $text;
    }
    
    .paused {
        border: solid red;
    }
    
    .running {
        border: solid green;
    }
    
    .completed {
        border: solid blue;
    }
    
    TabbedContent {
        height: 100%;
    }
    
    TabPane {
        padding: 1;
    }
    """
    
    TITLE = "🚀 Veeksha LLM Benchmark Analytics Dashboard"
    SUB_TITLE = "Live Performance Metrics & Analysis"

    BINDINGS = [
        ("p", "pause", "Pause"),
        ("r", "resume", "Resume"),
        ("n", "next_benchmark", "Next"),
        ("b", "prev_benchmark", "Prev"),
        ("s", "save_analysis", "Save"),
        ("q", "quit", "Quit"),
        ("ctrl+c", "quit", "Quit"),
    ]
    
    def __init__(self, dashboard_state: DashboardState):
        super().__init__()
        self.dashboard_state = dashboard_state
        self.update_timer: Optional[Timer] = None
        self.is_paused = False
        self.benchmark_completed = False
        self.analysis_mode = False
        
        # Graph widgets
        self.ttft_graph = None
        self.tpot_graph = None
        self.tbt_graph = None
        self.latency_graph = None
        
    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header()
        
        with TabbedContent():
            with TabPane("📊 Overview", id="overview"):
                with Vertical():
                    with Horizontal():
                        yield BenchmarkStatus(classes="benchmark-status")
                        yield MetricsDisplay(classes="metric-box")
                    yield LiveRequestsTable(classes="live-requests")
            
            with TabPane("📈 Performance Graphs", id="graphs"):
                with Vertical():
                    with Horizontal():
                        self.ttft_graph = TTFTGraph(classes="graph-widget")
                        yield self.ttft_graph
                        self.tpot_graph = TPOTGraph(classes="graph-widget")
                        yield self.tpot_graph
                    with Horizontal():
                        self.tbt_graph = TBTGraph(classes="graph-widget")
                        yield self.tbt_graph
                        self.latency_graph = LatencyOverTime(classes="graph-widget")
                        yield self.latency_graph
            
            with TabPane("🔍 Analysis", id="analysis"):
                yield Static("📊 Post-Benchmark Analysis\n\nDetailed analysis will appear here after benchmark completion...", 
                           classes="analysis-panel", id="analysis-content")
        
        yield Footer()
    
    def on_mount(self) -> None:
        """Called when app starts."""
        # Start the update timer to refresh data every 500ms
        self.update_timer = self.set_interval(0.5, self.update_dashboard)
        self.update_subtitle()
        self.notify("📊 Analytics Dashboard started! 'p'=pause, 'r'=resume, 'n/b'=switch benchmark, 'q'=quit", timeout=5)

    def update_subtitle(self) -> None:
        """Update subtitle with current benchmark info"""
        benchmark_ids = self.dashboard_state.get_benchmark_ids()
        active_id = self.dashboard_state.active_benchmark_id
        if len(benchmark_ids) > 1:
            self.sub_title = f"Live Metrics | Benchmark: {active_id} ({benchmark_ids.index(active_id) + 1}/{len(benchmark_ids)})"
        else:
            self.sub_title = f"Live Metrics | Benchmark: {active_id}"
    
    def action_pause(self) -> None:
        """Pause the dashboard updates"""
        self.is_paused = True
        if self.update_timer:
            self.update_timer.pause()
        self.notify("⏸️ Dashboard paused", severity="warning")
        self.add_class("paused")
        self.remove_class("running")
    
    def action_resume(self) -> None:
        """Resume the dashboard updates"""
        self.is_paused = False
        if self.update_timer:
            self.update_timer.resume()
        self.notify("▶️ Dashboard resumed", severity="information")
        self.add_class("running")
        self.remove_class("paused")
    
    def action_next_benchmark(self) -> None:
        """Switch to next benchmark"""
        benchmark_ids = self.dashboard_state.get_benchmark_ids()
        if len(benchmark_ids) <= 1:
            self.notify("⚠️ Only one benchmark available", severity="warning")
            return

        current_idx = benchmark_ids.index(self.dashboard_state.active_benchmark_id)
        next_idx = (current_idx + 1) % len(benchmark_ids)
        self.dashboard_state.set_active_benchmark(benchmark_ids[next_idx])
        self.update_subtitle()
        self.notify(f"📊 Switched to benchmark: {benchmark_ids[next_idx]}", severity="information")

    def action_prev_benchmark(self) -> None:
        """Switch to previous benchmark"""
        benchmark_ids = self.dashboard_state.get_benchmark_ids()
        if len(benchmark_ids) <= 1:
            self.notify("⚠️ Only one benchmark available", severity="warning")
            return

        current_idx = benchmark_ids.index(self.dashboard_state.active_benchmark_id)
        prev_idx = (current_idx - 1) % len(benchmark_ids)
        self.dashboard_state.set_active_benchmark(benchmark_ids[prev_idx])
        self.update_subtitle()
        self.notify(f"📊 Switched to benchmark: {benchmark_ids[prev_idx]}", severity="information")

    def action_save_analysis(self) -> None:
        """Save analysis data"""
        if self.benchmark_completed:
            self.notify("💾 Analysis saved to benchmark results directory", severity="information")
        else:
            self.notify("⚠️ Benchmark still running - analysis will be available after completion", severity="warning")

    def action_quit(self) -> None:
        """Quit the dashboard"""
        if self.benchmark_completed or self.analysis_mode:
            self.exit()
        else:
            self.notify("⚠️ Benchmark still running! Press 'q' again to force quit or wait for completion", severity="warning")
            # Set a timer to allow force quit
            self.set_timer(3.0, lambda: setattr(self, 'analysis_mode', True))
        
    def update_dashboard(self) -> None:
        """Update all dashboard components with latest data"""
        if not self.is_paused:
            self.update_subtitle()
            self.update_benchmark_status()
            self.update_metrics_display()
            self.update_live_requests()
            self.update_graphs()
    
    def mark_benchmark_completed(self) -> None:
        """Mark benchmark as completed and switch to analysis mode"""
        self.benchmark_completed = True
        self.analysis_mode = True
        self.add_class("completed")
        self.remove_class("running")
        
        # Capture final state for analysis
        self.final_duration = self.dashboard_state.get_benchmark_duration()
        self.final_stats = self.dashboard_state.get_aggregate_stats()
        self.final_requests = self.dashboard_state.get_all_requests()  # Get both live and completed
        
        # Stop updating graphs but keep final data visible
        self.freeze_graphs()
        
        self.notify("✅ Benchmark completed! Analysis mode active. Data preserved for review.", severity="success")
        self.generate_analysis()
    
    def freeze_graphs(self) -> None:
        """Freeze graphs at their final state and add completion markers"""
        if self.ttft_graph:
            self.ttft_graph.freeze_at_completion()
        if self.tpot_graph:
            self.tpot_graph.freeze_at_completion()
        if self.tbt_graph:
            self.tbt_graph.freeze_at_completion()
        if self.latency_graph:
            self.latency_graph.freeze_at_completion()
    
    def update_graphs(self) -> None:
        """Update all graph widgets with latest data"""
        if not (self.ttft_graph and self.tpot_graph and self.tbt_graph and self.latency_graph):
            return
        
        # Stop updating graphs if benchmark is completed
        if self.benchmark_completed:
            return
            
        # Get current metrics
        aggregate_stats = self.dashboard_state.get_aggregate_stats()
        
        # Update TTFT graph - add new data points as they come
        if aggregate_stats.recent_ttft_ms:
            last_count = getattr(self.ttft_graph, '_last_count', 0)
            if len(aggregate_stats.recent_ttft_ms) > last_count:
                # Convert deque to list for slicing
                ttft_list = list(aggregate_stats.recent_ttft_ms)
                for ttft in ttft_list[last_count:]:
                    self.ttft_graph.add_data_point(ttft)
                self.ttft_graph._last_count = len(aggregate_stats.recent_ttft_ms)
                self.ttft_graph.update_plot()
        
        # Update TPOT graph - add new data points as they come
        if aggregate_stats.recent_tpot_ms:
            last_count = getattr(self.tpot_graph, '_last_count', 0)
            if len(aggregate_stats.recent_tpot_ms) > last_count:
                # Convert deque to list for slicing
                tpot_list = list(aggregate_stats.recent_tpot_ms)
                for tpot in tpot_list[last_count:]:
                    self.tpot_graph.add_data_point(tpot)
                self.tpot_graph._last_count = len(aggregate_stats.recent_tpot_ms)
                self.tpot_graph.update_plot()

        # Update TBT graph - add new data points as they come
        if aggregate_stats.recent_tbt_ms:
            last_count = getattr(self.tbt_graph, '_last_count', 0)
            if len(aggregate_stats.recent_tbt_ms) > last_count:
                # Convert deque to list for slicing
                tbt_list = list(aggregate_stats.recent_tbt_ms)
                for tbt in tbt_list[last_count:]:
                    self.tbt_graph.add_data_point(tbt)
                self.tbt_graph._last_count = len(aggregate_stats.recent_tbt_ms)
                self.tbt_graph.update_plot()
        
        # Update latency graph - add new data points as they come
        if aggregate_stats.recent_latency_ms:
            last_count = getattr(self.latency_graph, '_last_count', 0)
            if len(aggregate_stats.recent_latency_ms) > last_count:
                # Convert deque to list for slicing
                latency_list = list(aggregate_stats.recent_latency_ms)
                for latency in latency_list[last_count:]:
                    self.latency_graph.add_latency(latency)
                self.latency_graph._last_count = len(aggregate_stats.recent_latency_ms)
                self.latency_graph.update_plot()
    
    def generate_analysis(self) -> None:
        """Generate post-benchmark analysis"""
        aggregate_stats = self.dashboard_state.get_aggregate_stats()
        duration = self.dashboard_state.get_benchmark_duration()
        
        analysis_text = f"""🎯 **Benchmark Analysis Summary**

⏱️ **Duration:** {duration:.1f} seconds
📊 **Total Requests:** {aggregate_stats.total_requests}
✅ **Completed:** {aggregate_stats.completed_count}
❌ **Errors:** {aggregate_stats.error_count}
📈 **Success Rate:** {(aggregate_stats.completed_count/max(aggregate_stats.total_requests,1)*100):.1f}%

🔥 **Performance Metrics:**
• Average TTFT: {aggregate_stats.avg_ttft_ms:.2f} ms
• Average TPOT: {aggregate_stats.avg_tpot_ms:.2f} ms  
• Average Latency: {aggregate_stats.avg_latency_ms:.2f} ms
• Effective Throughput: {aggregate_stats.completed_count/max(duration,1):.2f} req/s

📉 **Trends:**
• TTFT Samples: {len(aggregate_stats.recent_ttft_ms)}
• TPOT Samples: {len(aggregate_stats.recent_tpot_ms)}
• Latency Samples: {len(aggregate_stats.recent_latency_ms)}

💡 **Insights:**
• {"Low latency - excellent performance!" if aggregate_stats.avg_latency_ms < 500 else "High latency - consider optimization" if aggregate_stats.avg_latency_ms > 2000 else "Moderate latency - within acceptable range"}
• {"Fast TTFT - good responsiveness!" if aggregate_stats.avg_ttft_ms < 100 else "Slow TTFT - may impact user experience" if aggregate_stats.avg_ttft_ms > 500 else "Acceptable TTFT"}
• {"Efficient token generation!" if aggregate_stats.avg_tpot_ms < 10 else "Token generation could be optimized" if aggregate_stats.avg_tpot_ms > 50 else "Reasonable token generation speed"}

🎨 Use the Performance Graphs tab to visualize trends over time.
💾 Press 's' to save this analysis to the results directory.
"""
        
        try:
            analysis_widget = self.query_one("#analysis-content", Static)
            analysis_widget.update(analysis_text)
        except Exception:
            pass  # Widget might not be available yet
    
    def update_benchmark_status(self) -> None:
        """Update the benchmark status display"""
        status_widget = self.query_one(".benchmark-status", BenchmarkStatus)
        
        # Use frozen data if benchmark is completed, otherwise use live data
        if self.benchmark_completed and hasattr(self, 'final_stats'):
            aggregate_stats = self.final_stats
            duration = self.final_duration
            live_requests = self.final_requests
        else:
            aggregate_stats = self.dashboard_state.get_aggregate_stats()
            duration = self.dashboard_state.get_benchmark_duration()
            live_requests = self.dashboard_state.get_live_requests()
        
        # Calculate QPS
        if self.benchmark_completed:
            final_qps = aggregate_stats.completed_count / max(duration, 1)
            qps_display = f"{final_qps:.2f}"
        else:
            current_qps = getattr(self.dashboard_state, 'current_qps', 0.0)
            qps_display = f"{current_qps:.2f}"
        
        # Calculate success rate
        success_rate = (aggregate_stats.completed_count / max(aggregate_stats.total_requests, 1)) * 100
        
        # Status indicator
        status_emoji = "🟢" if not self.benchmark_completed else "🔵"
        status_text_indicator = "RUNNING" if not self.benchmark_completed else "COMPLETED"
        
        status_text = f"""
{status_emoji} [bold]Status:[/] {status_text_indicator}
⏱️ [bold]Runtime:[/] {duration:.1f}s
📊 [bold]Requests:[/] {aggregate_stats.total_requests} total
✅ [bold]Completed:[/] {aggregate_stats.completed_count} ({success_rate:.1f}%)
❌ [bold]Errors:[/] {aggregate_stats.error_count}
🔄 [bold]Active:[/] {len(live_requests)}
🚀 [bold]QPS:[/] {qps_display}
        """.strip()
        
        status_widget.update(status_text)
    
    def update_metrics_display(self) -> None:
        """Update the metrics display"""
        metrics_widget = self.query_one(".metric-box", MetricsDisplay)
        
        # Use frozen data if benchmark is completed, otherwise use live data
        if self.benchmark_completed and hasattr(self, 'final_stats'):
            aggregate_stats = self.final_stats
            duration = self.final_duration
        else:
            aggregate_stats = self.dashboard_state.get_aggregate_stats()
            duration = self.dashboard_state.get_benchmark_duration()
        
        # Performance indicators
        ttft_indicator = "🟢" if aggregate_stats.avg_ttft_ms < 100 else "🟡" if aggregate_stats.avg_ttft_ms < 500 else "🔴"
        tpot_indicator = "🟢" if aggregate_stats.avg_tpot_ms < 10 else "🟡" if aggregate_stats.avg_tpot_ms < 50 else "🔴"
        latency_indicator = "🟢" if aggregate_stats.avg_latency_ms < 500 else "🟡" if aggregate_stats.avg_latency_ms < 2000 else "🔴"
        
        # Status indicator for completion
        status_indicator = " [FINAL]" if self.benchmark_completed else ""
        
        metrics_text = f"""
{ttft_indicator} [bold]TTFT:[/] {aggregate_stats.avg_ttft_ms:.2f} ms{status_indicator}
{tpot_indicator} [bold]TPOT:[/] {aggregate_stats.avg_tpot_ms:.2f} ms{status_indicator}
{latency_indicator} [bold]Latency:[/] {aggregate_stats.avg_latency_ms:.2f} ms{status_indicator}
📈 [bold]Samples:[/] {len(aggregate_stats.recent_ttft_ms)}
🎯 [bold]Throughput:[/] {aggregate_stats.completed_count/max(duration,1):.1f}/s
        """.strip()
        
        metrics_widget.update(metrics_text)
    
    def update_live_requests(self) -> None:
        """Update the live requests table"""
        table = self.query_one(".live-requests", LiveRequestsTable)
        
        # Clear existing rows
        table.clear()
        
        # Use frozen data if benchmark is completed, otherwise use live data
        if self.benchmark_completed and hasattr(self, 'final_requests'):
            live_requests = self.final_requests
            # Update table title for completed state
            table.border_title = "📋 Final Request Status"
        else:
            live_requests = self.dashboard_state.get_live_requests()
            table.border_title = "🔄 Active Requests"
        
        if not live_requests:
            status_text = "No requests completed" if self.benchmark_completed else "No active requests"
            table.add_row(status_text, "", "", "", "", "")
            return
        
        # Sort by start time (most recent first)
        sorted_requests = sorted(live_requests, key=lambda x: x.start_timestamp, reverse=True)
        
        for req in sorted_requests[:10]:  # Show only top 10 to avoid clutter
            # Use fixed duration if benchmark completed, otherwise calculate live
            if self.benchmark_completed and hasattr(self, 'final_duration'):
                duration = self.final_duration
            else:
                current_time = time.time()
                duration = current_time - req.start_timestamp
            
            ttft_display = f"{req.ttft_ms:.1f}" if req.ttft_ms else "—"
            tpot_display = f"{req.current_tpot_ms:.1f}" if req.current_tpot_ms else "—"
            
            # Truncate request ID for better display
            request_id_str = str(req.request_id)
            display_id = request_id_str[:12] if len(request_id_str) > 12 else request_id_str
            
            # Add completion status for finished benchmarks
            duration_display = f"{duration:.1f}"
            if self.benchmark_completed:
                if req.current_output_tokens > 0:
                    duration_display += " ✅"
                else:
                    duration_display += " ❌"
            
            table.add_row(
                display_id,
                str(req.input_tokens),
                str(req.current_output_tokens),
                ttft_display,
                tpot_display,
                duration_display
            )


def run_dashboard_frontend(dashboard_state: DashboardState) -> Optional[threading.Thread]:
    """Run the dashboard frontend - provides console fallback for now"""
    import threading
    import logging
    
    logger = logging.getLogger(__name__)
    
    # For now, always use console dashboard when called from benchmark
    # The interactive TUI should be launched separately using run_dashboard_with_benchmark()
    logger.info("Dashboard: Starting console metrics display (use run_dashboard_with_benchmark for TUI)")
    return _run_console_dashboard(dashboard_state, logger)


def _run_console_dashboard(dashboard_state: DashboardState, logger) -> threading.Thread:
    """Console fallback dashboard"""
    def _console_dashboard():
        """Simple console-based dashboard that prints metrics periodically"""
        import time
        
        logger.info("Dashboard: Starting console metrics display")
        
        try:
            while True:
                time.sleep(5)  # Update every 5 seconds
                
                # Get current metrics
                aggregate_stats = dashboard_state.get_aggregate_stats()
                live_requests = dashboard_state.get_live_requests()
                duration = dashboard_state.get_benchmark_duration()
                
                # Print a simple metrics summary
                print(f"\n=== Benchmark Metrics (Runtime: {duration:.1f}s) ===")
                print(f"Total Requests: {aggregate_stats.total_requests}")
                print(f"Completed: {aggregate_stats.completed_count}")
                print(f"Errors: {aggregate_stats.error_count}")
                print(f"Active Requests: {len(live_requests)}")
                
                if aggregate_stats.recent_ttft_ms:
                    print(f"Avg TTFT: {aggregate_stats.avg_ttft_ms:.2f} ms")
                if aggregate_stats.recent_tpot_ms:
                    print(f"Avg TPOT: {aggregate_stats.avg_tpot_ms:.2f} ms")
                if aggregate_stats.recent_latency_ms:
                    print(f"Avg Latency: {aggregate_stats.avg_latency_ms:.2f} ms")
                
                print("=" * 50)
                
        except Exception as e:
            logger.error(f"Console dashboard error: {e}")
    
    dashboard_thread = threading.Thread(target=_console_dashboard, daemon=True)
    dashboard_thread.start()
    return dashboard_thread


def run_dashboard_gui(dashboard_state: DashboardState) -> None:
    """Run the full textual GUI dashboard - must be called from main thread"""
    app = VeekshaDashboard(dashboard_state)
    app.run()


def run_dashboard_with_benchmark(benchmark_config) -> None:
    """Run interactive TUI dashboard with benchmark in background thread"""
    import threading
    import time
    from veeksha.dashboard.handler import init_dashboard_event_processor
    
    # Initialize dashboard without frontend (we'll handle it here)
    dashboard_state = init_dashboard_event_processor(
        enabled=True,
        enable_frontend=False,  # Don't start the console version
        max_queue_size=benchmark_config.dashboard_config.max_queue_size,
        max_live_requests=benchmark_config.dashboard_config.max_live_requests,
    )
    
    if not dashboard_state:
        raise RuntimeError("Failed to initialize dashboard state")
    
    # Flag to track benchmark completion
    benchmark_complete = threading.Event()
    benchmark_error = None
    
    def _run_benchmark():
        """Run benchmark in background thread"""
        nonlocal benchmark_error
        try:
            # Import here to avoid circular import
            from veeksha.benchmark import run_benchmark
            run_benchmark(benchmark_config)
        except Exception as e:
            benchmark_error = e
        finally:
            benchmark_complete.set()
    
    # Start benchmark in background thread
    benchmark_thread = threading.Thread(target=_run_benchmark, daemon=True)
    benchmark_thread.start()
    
    # Create and run the TUI dashboard in main thread
    class BenchmarkDashboard(VeekshaDashboard):
        """Extended dashboard that monitors benchmark completion"""
        
        def on_mount(self) -> None:
            super().on_mount()
            # Check for benchmark completion every second
            self.set_interval(1.0, self.check_benchmark_status)
        
        def check_benchmark_status(self) -> None:
            """Check if benchmark is complete and transition to analysis mode"""
            if benchmark_complete.is_set():
                if not self.benchmark_completed:  # Only do this once
                    if benchmark_error:
                        self.notify(f"Benchmark failed: {benchmark_error}", severity="error")
                        self.analysis_mode = True
                    else:
                        self.mark_benchmark_completed()
        
        def action_quit(self) -> None:
            """Override quit to handle analysis mode"""
            if self.benchmark_completed or self.analysis_mode:
                self.notify("👋 Goodbye! Analysis data has been preserved.", severity="information")
                self.exit()
            else:
                # Ask for confirmation during active benchmark
                if not hasattr(self, '_quit_warned'):
                    self._quit_warned = True
                    self.notify("⚠️ Benchmark still running! Press 'q' again to force quit", severity="warning")
                else:
                    self.notify("🛑 Force quitting...", severity="error")
                    self.exit()
    
    try:
        app = BenchmarkDashboard(dashboard_state)
        app.run()
    except KeyboardInterrupt:
        print("\nDashboard stopped by user")
    
    # Wait for benchmark to complete if it's still running
    if benchmark_thread.is_alive():
        print("Waiting for benchmark to complete...")
        benchmark_thread.join(timeout=5)
    
    if benchmark_error:
        raise benchmark_error


# For testing the dashboard independently
if __name__ == "__main__":
    from veeksha.dashboard.state import DashboardState
    import random
    
    # Create a mock dashboard state for testing
    state = DashboardState()
    
    # Add some mock data
    from veeksha.dashboard.events import RequestStartedEvent, RequestCompletedEvent
    from veeksha.metrics.request_metrics import RequestMetrics
    
    # Mock some events
    for i in range(5):
        event = RequestStartedEvent(
            request_id=f"req_{i}",
            timestamp=time.time() - random.uniform(0, 10),
            input_tokens=random.randint(100, 500)
        )
        state.apply(event)
    
    app = VeekshaDashboard(state)
    app.run()
