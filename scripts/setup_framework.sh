#!/usr/bin/env bash
# Clone and compile the course framework (RecSys_Course_AT_PoliMi, AGPL-3.0) into external/.
set -euo pipefail

COMMIT=77608ac3939a4c3c18a77cd51f999a79728aa6a8   # version used for this repo
DEST="$(cd "$(dirname "$0")/.." && pwd)/external/RecSys_Course_AT_PoliMi"

if [ ! -d "$DEST" ]; then
  git clone https://github.com/MaurizioFD/RecSys_Course_AT_PoliMi.git "$DEST"
fi
cd "$DEST"
git checkout -q "$COMMIT"

# Cython 3 rejects the framework's qsort comparators unless the legacy noexcept rule is on.
perl -pi -e "s/\{'language_level': '3'\}/{'language_level': '3', 'legacy_implicit_noexcept': True}/" \
  CythonCompiler/compile_script.py

# NumPy >= 1.24 removed the np.int / np.float / np.bool aliases the framework still uses.
grep -rlE "np\.(int|float|bool)\b" --include=*.py --include=*.pyx Recommenders Evaluation Utils Data_manager \
  | xargs perl -pi -e 's/\bnp\.(int|float|bool)\b/$1/g'

CFLAGS="-Wno-unused-function -Wno-uninitialized -Wno-cpp" "${PYTHON:-python3}" run_compile_all_cython.py
