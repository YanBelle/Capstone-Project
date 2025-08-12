#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple import test for cash forecasting dependencies
"""

print("Testing Python import dependencies...")

try:
    import flask
    print("✓ Flask available")
except ImportError:
    print("✗ Flask not available")

try:
    import pandas as pd
    print("✓ Pandas available")
except ImportError:
    print("✗ Pandas not available")

try:
    import numpy as np
    print("✓ NumPy available")
except ImportError:
    print("✗ NumPy not available")

try:
    from sklearn.linear_model import LinearRegression
    print("✓ Scikit-learn available")
except ImportError:
    print("✗ Scikit-learn not available")

try:
    from prophet import Prophet
    print("✓ Prophet available")
except ImportError:
    print("✗ Prophet not available (using fallback)")

try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    print("✓ Statsmodels available")
except ImportError:
    print("✗ Statsmodels not available (using fallback)")

try:
    from sqlalchemy import create_engine
    print("✓ SQLAlchemy available")
except ImportError:
    print("✗ SQLAlchemy not available")

print("\nDependency check complete!")
