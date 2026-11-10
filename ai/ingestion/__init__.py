"""Ingestion pipeline.

PIPELINE_VERSION is bumped whenever a change alters what ends up in the
database for a given file: chunk boundaries, extraction behaviour, or a new
per-chunk field. Documents ingested under an older version are not treated as
duplicates of a fresh upload, so re-uploading actually reprocesses them.

Without this, uploading the same PDF after a pipeline change silently returned
the old record -- the file looked ingested, the UI said "ready", and the new
fields stayed empty with no error anywhere.
"""

PIPELINE_VERSION = 12
