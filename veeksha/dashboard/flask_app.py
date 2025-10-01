"""Flask-based web dashboard for Veeksha benchmarks.

This replaces the Textual terminal dashboard with a web interface accessible via browser.
For remote servers, use SSH port forwarding:
    ssh -L 5000:localhost:5000 user@remote-server
Then open http://localhost:5000 in your local browser.
"""

import json
import time
from flask import Flask, render_template, Response, jsonify
from flask_cors import CORS
from typing import Optional
import threading
import os

from veeksha.dashboard.state import DashboardState
from markupsafe import Markup


def render_chart(data, color='#4fc3f7', label='Value'):
    """Render a simple SVG line chart."""
    if not data or len(data) == 0:
        return Markup('<div style="text-align: center; opacity: 0.5; padding: 40px;">Waiting for data...</div>')

    # SVG dimensions
    width = 800
    height = 200
    padding = 40

    # Calculate scales
    max_val = max(data) if data else 1
    min_val = min(data) if data else 0
    val_range = max_val - min_val if max_val != min_val else 1

    # Create SVG path
    points = []
    for i, value in enumerate(data):
        x = padding + (i / max(len(data) - 1, 1)) * (width - 2 * padding)
        y = height - padding - ((value - min_val) / val_range) * (height - 2 * padding)
        points.append(f"{x},{y}")

    path_data = "M " + " L ".join(points)

    svg = f'''
    <svg width="100%" height="200" viewBox="0 0 {width} {height}" style="background: rgba(255,255,255,0.02); border-radius: 8px;">
        <!-- Grid lines -->
        <line x1="{padding}" y1="{padding}" x2="{padding}" y2="{height-padding}" stroke="rgba(255,255,255,0.1)" stroke-width="1"/>
        <line x1="{padding}" y1="{height-padding}" x2="{width-padding}" y2="{height-padding}" stroke="rgba(255,255,255,0.1)" stroke-width="1"/>

        <!-- Data line -->
        <path d="{path_data}" fill="none" stroke="{color}" stroke-width="2"/>

        <!-- Data points -->
        {" ".join([f'<circle cx="{x.split(",")[0]}" cy="{x.split(",")[1]}" r="3" fill="{color}"/>' for x in points[-10:]])}

        <!-- Labels -->
        <text x="{padding}" y="20" fill="rgba(255,255,255,0.6)" font-size="12">{label}</text>
        <text x="{padding}" y="{padding-5}" fill="rgba(255,255,255,0.8)" font-size="14">Max: {max_val:.1f}</text>
        <text x="{padding}" y="{height-padding+20}" fill="rgba(255,255,255,0.8)" font-size="14">Min: {min_val:.1f}</text>
        <text x="{width-padding-100}" y="{height-padding+20}" fill="rgba(255,255,255,0.8)" font-size="14">Samples: {len(data)}</text>
    </svg>
    '''

    return Markup(svg)


def create_flask_app(dashboard_state: DashboardState) -> Flask:
    """Create and configure the Flask application."""

    # Get the directory where this file is located for templates
    template_dir = os.path.join(os.path.dirname(__file__), 'templates')
    static_dir = os.path.join(os.path.dirname(__file__), 'static')

    app = Flask(__name__,
                template_folder=template_dir,
                static_folder=static_dir)
    CORS(app)  # Enable CORS for development

    app.config['dashboard_state'] = dashboard_state

    # Add chart rendering function to Jinja context
    app.jinja_env.globals['render_chart'] = render_chart

    @app.route('/')
    def index():
        """Render the main dashboard page."""
        state = app.config['dashboard_state']

        # Get active benchmark data
        benchmark_ids = state.get_benchmark_ids()
        active_id = state.active_benchmark_id

        # If no benchmarks yet, show loading state
        if not benchmark_ids:
            return render_template('dashboard.html',
                                 benchmarks=[],
                                 active_benchmark='',
                                 stats={},
                                 graph_data={},
                                 live_requests=[])

        aggregate_stats = state.get_aggregate_stats(active_id)
        duration = state.get_benchmark_duration(active_id)
        live_requests = state.get_live_requests(active_id)[:10]

        # Prepare stats
        stats = {
            'total_requests': aggregate_stats.total_requests,
            'completed_count': aggregate_stats.completed_count,
            'error_count': aggregate_stats.error_count,
            'duration': duration,
            'avg_ttft_ms': aggregate_stats.avg_ttft_ms,
            'avg_tpot_ms': aggregate_stats.avg_tpot_ms,
            'avg_tbt_ms': aggregate_stats.avg_tbt_ms,
            'avg_latency_ms': aggregate_stats.avg_latency_ms
        }

        # Prepare graph data (last 50 points)
        graph_data = {
            'ttft': list(aggregate_stats.recent_ttft_ms)[-50:],
            'tpot': list(aggregate_stats.recent_tpot_ms)[-50:],
            'tbt': list(aggregate_stats.recent_tbt_ms)[-50:],
            'latency': list(aggregate_stats.recent_latency_ms)[-50:]
        }

        # Prepare live requests
        live_requests_data = []
        for req in live_requests:
            live_requests_data.append({
                'request_id': req.request_id or 'N/A',
                'input_tokens': req.input_tokens,
                'output_tokens': req.current_output_tokens,
                'ttft_ms': req.ttft_ms,
                'tpot_ms': req.current_tpot_ms,
                'progress_pct': req.progress_pct
            })

        return render_template('dashboard.html',
                             benchmarks=benchmark_ids,
                             active_benchmark=active_id,
                             stats=stats,
                             graph_data=graph_data,
                             live_requests=live_requests_data)

    @app.route('/api/benchmarks')
    def get_benchmarks():
        """Get list of all benchmark IDs."""
        state = app.config['dashboard_state']
        return jsonify({
            'benchmarks': state.get_benchmark_ids(),
            'active': state.active_benchmark_id
        })

    @app.route('/api/benchmark/<benchmark_id>/stats')
    def get_benchmark_stats(benchmark_id: str):
        """Get aggregate stats for a specific benchmark."""
        state = app.config['dashboard_state']
        aggregate_stats = state.get_aggregate_stats(benchmark_id)

        return jsonify({
            'total_requests': aggregate_stats.total_requests,
            'completed_count': aggregate_stats.completed_count,
            'error_count': aggregate_stats.error_count,
            'avg_ttft_ms': aggregate_stats.avg_ttft_ms,
            'avg_tpot_ms': aggregate_stats.avg_tpot_ms,
            'avg_tbt_ms': aggregate_stats.avg_tbt_ms,
            'avg_latency_ms': aggregate_stats.avg_latency_ms,
            'duration': state.get_benchmark_duration(benchmark_id)
        })

    @app.route('/api/benchmark/<benchmark_id>/graph_data')
    def get_graph_data(benchmark_id: str):
        """Get time-series data for graphs."""
        state = app.config['dashboard_state']
        aggregate_stats = state.get_aggregate_stats(benchmark_id)

        # Convert deques to lists for JSON serialization
        return jsonify({
            'ttft': list(aggregate_stats.recent_ttft_ms),
            'tpot': list(aggregate_stats.recent_tpot_ms),
            'tbt': list(aggregate_stats.recent_tbt_ms),
            'latency': list(aggregate_stats.recent_latency_ms)
        })

    @app.route('/api/benchmark/<benchmark_id>/live_requests')
    def get_live_requests(benchmark_id: str):
        """Get currently active requests."""
        state = app.config['dashboard_state']
        live_requests = state.get_live_requests(benchmark_id)

        # Convert to serializable format
        requests_data = []
        for req in live_requests[:10]:  # Top 10
            requests_data.append({
                'request_id': req.request_id,
                'input_tokens': req.input_tokens,
                'output_tokens': req.current_output_tokens,
                'ttft_ms': req.ttft_ms,
                'tpot_ms': req.current_tpot_ms,
                'progress_pct': req.progress_pct
            })

        return jsonify({'requests': requests_data})

    @app.route('/api/stream')
    def stream():
        """Server-Sent Events stream for real-time updates."""
        def event_stream():
            state = app.config['dashboard_state']
            last_update = 0

            while True:
                # Send updates every 500ms
                current_time = time.time()
                if current_time - last_update >= 0.5:
                    active_id = state.active_benchmark_id

                    # Get current stats
                    aggregate_stats = state.get_aggregate_stats(active_id)

                    data = {
                        'benchmark_id': active_id,
                        'timestamp': current_time,
                        'stats': {
                            'total_requests': aggregate_stats.total_requests,
                            'completed_count': aggregate_stats.completed_count,
                            'error_count': aggregate_stats.error_count,
                            'avg_ttft_ms': aggregate_stats.avg_ttft_ms,
                            'avg_tpot_ms': aggregate_stats.avg_tpot_ms,
                            'avg_tbt_ms': aggregate_stats.avg_tbt_ms,
                            'avg_latency_ms': aggregate_stats.avg_latency_ms
                        }
                    }

                    yield f"data: {json.dumps(data)}\n\n"
                    last_update = current_time

                time.sleep(0.1)

        return Response(event_stream(), mimetype='text/event-stream')

    @app.route('/api/set_active/<benchmark_id>')
    def set_active_benchmark(benchmark_id: str):
        """Set the active benchmark."""
        state = app.config['dashboard_state']
        state.set_active_benchmark(benchmark_id)
        return jsonify({'success': True, 'active': benchmark_id})

    @app.route('/benchmark/<benchmark_id>')
    def switch_benchmark(benchmark_id: str):
        """Switch to a different benchmark and reload page."""
        state = app.config['dashboard_state']
        state.set_active_benchmark(benchmark_id)
        return index()  # Render with new benchmark

    return app


def run_dashboard_flask(
    dashboard_state: DashboardState,
    host: str = 'localhost',
    port: int = 5000,
    debug: bool = False
) -> threading.Thread:
    """Run the Flask dashboard in a background thread.

    Args:
        dashboard_state: The shared dashboard state object
        host: Host to bind to (default: localhost for security)
        port: Port to bind to (default: 5000)
        debug: Enable Flask debug mode (default: False)

    Returns:
        Thread object running the Flask server

    Usage:
        For remote access via SSH, use port forwarding:
            ssh -L 5000:localhost:5000 user@remote-server
        Then open http://localhost:5000 in your local browser.
    """

    app = create_flask_app(dashboard_state)

    # Print startup message with instructions
    is_remote = bool(os.environ.get('SSH_CONNECTION') or os.environ.get('SSH_CLIENT'))

    print("\n" + "="*70)
    print("🚀 Veeksha Dashboard Starting")
    print("="*70)
    print(f"  Dashboard URL: http://{host}:{port}")
    print()

    if is_remote or host == 'localhost':
        print("  📡 Remote Access via SSH Port Forwarding:")
        print(f"     ssh -L {port}:localhost:{port} <user>@<your-server>")
        print(f"     Then open: http://localhost:{port}")
        print()

    if host == '0.0.0.0':
        print("  ⚠️  WARNING: Dashboard exposed on all network interfaces!")
        print("     This may be a security risk. Use SSH forwarding instead.")
        print()

    print("  Press Ctrl+C in the benchmark terminal to stop")
    print("="*70 + "\n")

    def run_app():
        app.run(host=host, port=port, debug=debug, threaded=True, use_reloader=False)

    thread = threading.Thread(target=run_app, daemon=True)
    thread.start()

    return thread
