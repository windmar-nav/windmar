# WINDMAR Installation Guide

## Quick Start (Basic Functionality)

For route optimization without real-time weather data:

```bash
# Install basic dependencies
pip install numpy scipy pandas matplotlib openpyxl requests

# Run simple demo (no GRIB files)
python examples/demo_simple.py
```

This will:
- ✅ Calculate fuel consumption
- ✅ Optimize routes using great circle
- ✅ Create basic visualizations
- ✅ Show fuel scenarios
- ❌ No real-time weather data

**Output:** Creates `data/windmar_demo.png` with route map and fuel comparison chart

---

## Standard Installation (Recommended)

For most features without GRIB visualization:

```bash
# Install from requirements.txt
pip install -r requirements.txt
```

This enables:
- ✅ All route optimization features
- ✅ Noon report parsing
- ✅ Model calibration
- ✅ Basic visualizations
- ⚠️ GRIB download (but not parsing)

---

## Full Installation (GRIB Support)

For complete weather integration with real-time NOAA data:

### Step 1: Install ECCODES Library

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install libeccodes-dev
```

**macOS (Homebrew):**
```bash
brew install eccodes
```

**CentOS/RHEL:**
```bash
sudo yum install eccodes-devel
```

**Windows:**
Download from: https://confluence.ecmwf.int/display/ECC/ecCodes+Home

### Step 2: Install Python Packages

```bash
# Install all dependencies
pip install -r requirements.txt

# Install pygrib (requires ECCODES from Step 1)
pip install pygrib

# Optional: Install cartopy for advanced mapping
# (requires GEOS, PROJ libraries)
pip install cartopy
```

### Step 3: Test Installation

```bash
# Test GRIB parsing
python -c "import pygrib; print('✓ pygrib working')"

# Run full example with weather data
python examples/example_ara_med.py
```

This enables:
- ✅ Real-time NOAA weather downloads
- ✅ GRIB file parsing
- ✅ Weather-optimized routing
- ✅ Wind/wave field visualization
- ✅ Route overlays on weather maps

**Output:** Creates weather maps with routes in `data/ara_med_route_wind.png`

---

## Visualization Examples

### What You Get at Each Level:

**Basic (no dependencies):**
- Text-based route output
- Fuel calculations

**Standard (matplotlib):**
```python
python examples/demo_simple.py
```
- Route maps (lat/lon plot)
- Fuel comparison bar charts
- Performance scenarios

![Demo Output](data/windmar_demo.png)

**Full (pygrib + cartopy):**
```python
python examples/example_ara_med.py
```
- Weather field maps with wind vectors
- Wave height contours
- Routes overlaid on real weather
- Animated forecast evolution

---

## Troubleshooting

### pygrib won't install

**Error:** `fatal error: eccodes.h: No such file or directory`

**Solution:** Install ECCODES library first (see Step 1 above)

### cartopy issues

**Error:** `GEOS library not found`

**Solution:**
```bash
# Ubuntu
sudo apt-get install libgeos-dev libproj-dev

# macOS
brew install geos proj
```

Or skip cartopy - basic matplotlib works without it.

### ImportError on numpy/scipy

**Solution:**
```bash
pip install --upgrade pip
pip install numpy scipy
```

---

## Testing Installation

Run test suite to verify everything works:

```bash
# Basic tests (no GRIB required)
pytest tests/unit/test_vessel_model.py -v
pytest tests/unit/test_router.py -v

# All tests
pytest -v
```

---

## Docker Alternative (Future)

For easiest installation with all dependencies:

```bash
# Build image
docker build -t windmar .

# Run example
docker run -v $(pwd)/data:/app/data windmar python examples/example_ara_med.py
```

*(Dockerfile not yet created - let us know if you need this)*

---

## Summary

| Feature | Basic | Standard | Full |
|---------|-------|----------|------|
| Route optimization | ✅ | ✅ | ✅ |
| Fuel calculations | ✅ | ✅ | ✅ |
| Basic visualization | ❌ | ✅ | ✅ |
| Noon report parsing | ❌ | ✅ | ✅ |
| Model calibration | ❌ | ✅ | ✅ |
| GRIB download | ❌ | ✅* | ✅ |
| GRIB parsing | ❌ | ❌ | ✅ |
| Weather routing | ❌ | ❌ | ✅ |
| Weather maps | ❌ | ❌ | ✅ |

*Downloads but can't parse without pygrib

---

## Need Help?

1. Check examples in `examples/` directory
2. Review test files in `tests/unit/` for usage patterns
3. See API documentation in source code docstrings
4. Report issues with full error output
