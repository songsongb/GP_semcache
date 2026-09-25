#!/usr/bin/env python3
"""Offline C0/C1 harness; see docs/cachegen_storage.md for environment separation."""
from _common import ROOT  # Establish the existing repository import convention.
from semcache.experiments.cachegen.harness import main

if __name__ == '__main__':
    main()
