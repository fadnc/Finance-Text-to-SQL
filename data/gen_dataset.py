"""Deprecated entry point.

The hand-written synthetic finance generator has been replaced by the
Gretel Text-to-SQL dataset loader.  This shim keeps the old command working
while routing it to the new preparation script.
"""

import subprocess
import sys

print(
    "data/gen_dataset.py is deprecated. "
    "Preparing the gretelai/synthetic_text_to_sql snapshot instead..."
)
sys.exit(subprocess.call([sys.executable, "data/prepare_dataset.py"]))
