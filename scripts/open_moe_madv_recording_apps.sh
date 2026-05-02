#!/usr/bin/env bash
set -euo pipefail

open -a "System Information"
open -a "Activity Monitor"

cat <<'EOF'
Opened recording helpers:

1. System Information
   Keep the Hardware Overview visible so the video shows Apple M1 Max and 64 GB.

2. Activity Monitor
   Suggested tabs/windows:
   - Memory tab for memory pressure.
   - Window > GPU History for GPU activity charts.

For the terminal sidecar monitor, run:

  scripts/run_moe_madv_resource_monitor.sh

For live generation, run:

  scripts/run_moe_madv_live_generation_demo.sh
EOF
