"""Two-stage recommender for the PoliMi Recommender Systems 2025 Kaggle challenge."""

import os
import sys

# Base models come from the course framework (RecSys_Course_AT_PoliMi), which is not
# a pip package. scripts/setup_framework.sh clones and compiles it into external/.
_DEFAULT_FRAMEWORK = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "external", "RecSys_Course_AT_PoliMi",
)
FRAMEWORK_PATH = os.environ.get("RECSYS_FRAMEWORK_PATH", _DEFAULT_FRAMEWORK)

if not os.path.isdir(FRAMEWORK_PATH):
    raise ImportError(
        f"Course framework not found at {FRAMEWORK_PATH}. "
        "Run scripts/setup_framework.sh or set RECSYS_FRAMEWORK_PATH."
    )
if FRAMEWORK_PATH not in sys.path:
    sys.path.insert(0, FRAMEWORK_PATH)
