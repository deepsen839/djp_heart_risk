#!/usr/bin/env bash
set -euo pipefail
djp-audit
djp-train --seed 42
djp-evaluate
djp-explain
pytest -q
