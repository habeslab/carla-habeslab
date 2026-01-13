#!/bin/bash

# Activate your existing Python environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run the first script
echo "Running CAD_FL.py..."
python CAD_FL.py

# Run the second script
echo "Running 3.fedFL.py..."
python 3.fedFL.py

echo "Both scripts finished!"
