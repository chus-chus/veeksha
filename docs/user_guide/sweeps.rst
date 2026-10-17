Configuration Sweeps
====================

Veeksha supports running multiple benchmarks with different parameter combinations
using the ``!expand`` YAML tag. This creates a Cartesian product of configurations,
enabling systematic exploration of the parameter space.


The !expand tag
---------------

Use ``!expand`` to expand a list into multiple configurations:

.. code-block:: yaml

    traffic_scheduler:
      type: rate
      interval_generator:
        type: poisson
        arrival_rate: !expand [5, 10, 20]  # Creates 3 runs

This creates three separate benchmark runs with rates 5, 10, and 20.

.. hint::
    `!expand` can only be specified for fields that were not originally typed as lists. For example, `arrival_rate` is a float, so it can be swept.


Cartesian product expansion
---------------------------

Multiple ``!expand`` tags create a Cartesian product:

.. code-block:: yaml

    traffic_scheduler:
      type: concurrent
      target_concurrent_sessions: !expand [4, 8, 16]  # 3 values

    session_generator:
      type: synthetic
      channels:
        - type: text
          body_length_generator:
            type: fixed
            value: !expand [256, 512]  # 2 values

This creates **6 runs** (3 × 2):

1. concurrency=4, prompt=256
2. concurrency=4, prompt=512
3. concurrency=8, prompt=256
4. concurrency=8, prompt=512
5. concurrency=16, prompt=256
6. concurrency=16, prompt=512


Basic example
-------------

Create a file ``sweep.veeksha.yml``:

.. code-block:: yaml

    seed: 42
    output_dir: sweep_results

    traffic_scheduler:
      type: rate
      interval_generator:
        type: poisson
        arrival_rate: !expand [5, 10, 20, 30]

    session_generator:
      type: synthetic
      session_graph:
        type: linear
        inherit_history: true
      channels:
        - type: text
          body_length_generator:
            type: uniform
            min: 100
            max: 500

    client:
      type: openai_chat_completions
      api_base: http://localhost:8000/v1
      model: meta-llama/Llama-3-8B-Instruct

    runtime:
      benchmark_timeout: 60

    evaluators:
      - type: performance
        target_channels: ["text"]

Run it:

.. code-block:: bash

    uvx -p 3.14t veeksha benchmark --config sweep.veeksha.yml

Veeksha automatically runs 4 benchmarks with rates 5, 10, 20, and 30.


Output structure
----------------

Sweeps create a parent directory containing all run subdirectories and summary files:

.. code-block:: text

    benchmark_output/
    └── sweep_09:01:2026-16:38:22/           # Sweep parent directory
        ├── sweep_manifest.json              # Sweep configuration
        ├── sweep_summary.json               # Aggregated results
        ├── sweep_summary.csv                # Results in CSV format
        ├── 09:01:2026-16:38:22-960db960/    # First run (rate=10)
        │   ├── config.yml
        │   ├── metrics/
        │   └── ...
        └── 09:01:2026-16:39:00-3f3db8a5/    # Second run (rate=11)
            └── ...

Each run's full metrics are preserved in its subdirectory, enabling detailed
cross-run analysis (e.g., comparing TTFC distributions, throughput, or SLO compliance).


Sweep summary
-------------

Veeksha automatically generates summary files at the end of each sweep:

**sweep_summary.json** aggregates key metrics across all runs (abbreviated below):

.. code-block:: text

    {
      "base_output_dir": "benchmark_output/sweep_09:01:2026-16:38:22",
      "num_runs": 2,
      "runs": [
        {
          "run_index": 0,
          "run_dir": "...",
          "traffic": {"arrival_rate": 10},
          "summary_stats": {...},
          "throughput_metrics": {...},
          "all_slos_met": true
        },
        ...
      ]
    }

**sweep_summary.csv** provides the same data in a tabular format for easy
spreadsheet analysis.

Cross-file expansion
--------------------

When using :ref:`!include <configuration-splitting>` to split config across files,
``!expand`` tags work across file boundaries. Veeksha collects **all** ``!expand``
markers from all included files and computes the Cartesian product.

**Example: Sweep across client endpoints and traffic rates**

Create ``client.yml`` with multiple endpoints:

.. code-block:: yaml

    # client.yml - sweep across 2 servers
    type: openai_chat_completions
    api_base: !expand [http://server-a:8000/v1, http://server-b:8000/v1]
    model: meta-llama/Llama-3-8B-Instruct

Create ``traffic.yml`` with multiple arrival rates:

.. code-block:: yaml

    # traffic.yml - sweep across 3 rates
    type: rate
    interval_generator:
      type: poisson
      arrival_rate: !expand [5, 10, 20]

Create a main config using ``!include``:

.. code-block:: yaml

    # main.yml
    client: !include client.yml
    traffic_scheduler: !include traffic.yml

    session_generator:
      type: synthetic
      channels:
        - type: text
          body_length_generator:
            type: uniform
            min: 100
            max: 500

    runtime:
      benchmark_timeout: 60

Run with a single ``--config``:

.. code-block:: bash

    uvx -p 3.14t veeksha benchmark --config main.yml

This creates **6 runs** (2 servers × 3 rates):

1. api_base=server-a, rate=5
2. api_base=server-a, rate=10
3. api_base=server-a, rate=20
4. api_base=server-b, rate=5
5. api_base=server-b, rate=10
6. api_base=server-b, rate=20

**Multi-file example**

You can split across as many included files as needed. If ``client.yml`` has 2
``!expand`` values, ``traffic.yml`` has 3, and the main config has
``!expand [256, 512]`` for prompt length, this creates **12 runs** (2 × 3 × 2).

.. hint::
    Cross-file expansion is particularly useful for:

    - Testing multiple server deployments with the same workload
    - Comparing different models without duplicating config files
    - Running the same sweep against staging and production endpoints


Common sweep patterns
---------------------

**Rate Sweep** - Find latency vs load relationship

.. code-block:: yaml

    traffic_scheduler:
      type: rate
      interval_generator:
        type: poisson
        arrival_rate: !expand [1, 2, 5, 10, 20, 50]

**Concurrency Sweep** - Throughput scaling

.. code-block:: yaml

    traffic_scheduler:
      type: concurrent
      target_concurrent_sessions: !expand [1, 2, 4, 8, 16, 32]
      rampup_seconds: 10

**Prompt Length Sweep** - Prefill scaling

.. code-block:: yaml

    session_generator:
      channels:
        - type: text
          body_length_generator:
            type: fixed
            value: !expand [128, 256, 512, 1024, 2048]
      output_spec:
        text:
          output_length_generator:
            type: fixed
            value: 128

**Output Length Sweep** - Decode scaling

.. code-block:: yaml

    session_generator:
      channels:
        - type: text
          body_length_generator:
            type: fixed
            value: 256
      output_spec:
        text:
          output_length_generator:
            type: fixed
            value: !expand [64, 128, 256, 512, 1024]

**Multi-Dimensional Sweep**

.. code-block:: yaml

    traffic_scheduler:
      type: concurrent
      target_concurrent_sessions: !expand [4, 8]

    session_generator:
      channels:
        - type: text
          body_length_generator:
            type: fixed
            value: !expand [256, 512]
      output_spec:
        text:
          output_length_generator:
            type: fixed
            value: !expand [128, 256]

Creates 2 × 2 × 2 = **8 runs**.

