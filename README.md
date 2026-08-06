<div align="center">
  <h1>
    <img src="./docs/guard.svg" width="100" alt="Guard Logo"><br>
    Guard Local Detector
  </h1>
  <p><em>A local detection engine for the Guard Python client, integrating visual safety filters into your applications</em></p>
  <p>
    <a href="https://www.gnu.org/licenses/agpl-3.0"><img src="https://img.shields.io/badge/License-AGPL%20v3-blue.svg" alt="License: AGPL v3"></a>
    <a href="https://github.com/elhio/guard-local-python/fork"><img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg" alt="PRs Welcome"></a>  
  </p>
</div>

## Features

**🛡️ Multi-Layered Content Moderation:** Automatically detects AI-generated, violent, and explicit content in images and 
videos.

**⚡ Fast Local Inference:** Runs a lightweight computer vision model entirely on-device using ONNX Runtime. It operates 
with zero network latency and completely avoids pulling in massive deep-learning dependencies like PyTorch.

**🏢 Seamless Production Testing:** Designed to perfectly mirror the cloud API's data structures. This allows
enterprise teams to easily build and test their integration logic locally before routing production traffic to the cloud 
API.

**🧑‍💻 Instant Open-Source Baseline:** Provides open-source developers and hobbyists with a free, immediately usable 
baseline classifier. Get a foundational media safety layer up and running in minutes, not days.

## Installation

### Recommended Way

We highly recommend using this package as an optional extension of the main Guard client. This provides a single, 
unified API for both cloud and local detection.

```bash
pip install guard-client[local]
```

### Standalone Way

```bash
pip install guard-local-detector
```

## Quick Start

### Using the Unified Client (Recommended)

If you installed via `guard-client[local]`, you do not need to import this package directly. Ask the client for the 
local engine and it routes every call here — no API key, and no network access.

```python
from guard_client import GuardClient

with GuardClient(engine="local") as client:
    result = client.analyze("/local/paths/to/video.mp4")

    for item in result.results:
        print(f"{item.label}: {item.score}")  # AI-Generated: 71
```

The results carry the same labels, task ids, and 0-100 scores a cloud run returns, so the same code works against 
either engine. A cloud-configured client can also send a single call locally with `client.analyze(source, 
engine="local")`.

### Using the Standalone Engine

The engine works on bytes and a MIME type, never a path — reading the source is the caller's job.

```python
import guard_local

engine = guard_local.LocalDetectorEngine()

with open("/local/paths/to/image.png", "rb") as handle:
    results = engine.analyze(handle.read(), "image/png")
    
for item in results:
    print(f"{item['label']}: {item['score']:.2f}")

# AI-Generated: 0.90
# Violence: 0.02
# Explicit: 0.01
```

## Development

This project uses [uv](https://docs.astral.sh/uv/) for lightning-fast Python package and environment management.

### Prerequisites

* [uv](https://docs.astral.sh/uv/) (already installed on your system)

### Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/elhio/guard-local-python.git
    cd guard-local-python
    ```

2. Sync the environment:
    ```bash
    uv sync
    ```
    *This command automatically creates a `.venv` virtual environment, reads the `uv.lock` file, and installs all core* 
    *and development dependencies exactly as they were locked.*

3. Run tests:
    ```bash
    uv run pytest
    ```

4. Formatting, linting, and type checking:
    ```bash
    uv run ruff format
    uv run ruff check
    uv run mypy
    ```

5. Build for production:
    ```bash
    uv build
    ```

## Contributing

We welcome contributions! Please note that all contributors must sign our automated CLA. Read more in our 
[Contributing Guide](CONTRIBUTING.md).

## License

This repository and its corresponding PyPI package are licensed under the GNU Affero General Public License v3.0 
(AGPL-3.0) - see the [LICENSE](LICENSE) file for details.
