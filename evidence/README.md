# Exported evidence

`python scripts/collect_evidence.py` writes fresh, sanitized observations to the ignored `evidence/live/` directory. Five conversations retain tool status, request IDs, duration and original structured evidence. Pass `--trigger` only when intentionally authorizing the approved training DAG. Failures remain failures in the export.

The reviewed submission snapshot is stored in this directory as JSON exports. Each snapshot identifies the observation time, environment and whether results come from actual services or controlled test doubles. Never interpret test-double results as proof that infrastructure was running.

Historical reproduction uses a separate temporary Git clone and DVC checkout, preserving the working tree. See the technical report for observed hashes, real model runs, workflow task states and deployment conditions.
