"""Shared pytest configuration."""

import matplotlib


# Tests render plots to files and must not depend on a desktop GUI being available.
matplotlib.use('Agg', force=True)
