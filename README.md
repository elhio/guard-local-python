<div align="center">
  <h1>
    <img src="./docs/assets/guard.svg" width="100" alt="Guard Logo"><br>
    Guard Local Detector
  </h1>
  <p><em>A seamless local detection engine for the Guard Python client, integrating visual safety filters into your applications</em></p>
  <p>
    <a href="https://www.gnu.org/licenses/agpl-3.0"><img src="https://img.shields.io/badge/License-AGPL%20v3-blue.svg" alt="License: AGPL v3"></a>
    <a href="https://github.com/elhio/guard-browser-extension/fork"><img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg" alt="PRs Welcome"></a>  
  </p>
</div>

## Features

**🛡️ Multi-Layered Content Moderation:** Automatically detects AI-generated, violent, and explicit content within 
images.

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

If you installed via `guard-client[local]`, you do not need to import this package directly. The main client will 
automatically detect its presence and route local file checks to this engine.

```python
from guard_client import GuardClient

client = GuardClient(api_key="your_api_key_here")

# URL provided: automatically routed to the local engine
cloud_result = client.check_media(url="[https://example.com/video.mp4](https://example.com/video.mp4)")

# File path provided: Automatically routed to the local AGPL engine
local_result = client.check_media(file_path="/local/paths/to/video.mp4")
```

### Using the Standalone Engine

```python
import guard_local

# Run the media detection locally on your hardware
result = guard_local.analyze_file("/local/paths/to/video.mp4")

print(f"Detection Results: {result}")
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

4. Formatting and linting:
    ```bash
    uv run ruff format
    uv run ruff check
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
