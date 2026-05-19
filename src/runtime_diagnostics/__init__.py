"""Runtime diagnostics for the live production pipeline.

This package is OFF by default. It only collects and reports timing
data when the environment variable VOLLEYHUB_LIVE_PROBE=1 is set.
Production behavior is unchanged when the probe is inactive.
"""
