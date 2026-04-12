#!/bin/bash
# Railway start script for HawkEye Backend

# Install dependencies if needed
pip install -r requirements.txt

# Create uploads directory
mkdir -p uploads

# Start the Flask app with gunicorn
exec gunicorn app:app --bind 0.0.0.0:${PORT:-5000} --workers 2 --timeout 300
