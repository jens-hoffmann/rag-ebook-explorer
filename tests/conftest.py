"""Test configuration and fixtures."""

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Podman configuration for testcontainers
os.environ.setdefault("TESTCONTAINERS_RYUK_DISABLED", "true")

# Try to detect Podman socket location
if "DOCKER_HOST" not in os.environ:
    # Default for Linux with standard Podman setup
    linux_socket = Path("/run/user/1000/podman/podman.sock")
    if linux_socket.exists():
        os.environ["DOCKER_HOST"] = f"unix://{linux_socket}"
    else:
        # Fallback - will need to be configured manually
        os.environ["DOCKER_HOST"] = "unix:///run/user/1000/podman/podman.sock"
