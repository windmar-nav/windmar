"""
Weather pipeline package for WINDMAR API.

Replaces the monolithic ``api.routers.weather`` module with focused modules:

- ``grid_processor`` — subsample, NaN sanitize, shape guarantees
- ``ocean_mask``     — CMEMS NaN masking (no global_land_mask except ice)
- ``frame_builder``  — build cache envelopes from DB or providers
- ``prefetch``       — background prefetch, wind GRIB handler
- ``formatters``     — single-frame response formatting per component type
- ``router``         — thin FastAPI endpoints, no processing logic
"""
