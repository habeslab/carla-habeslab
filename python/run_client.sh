#!/bin/bash

# Activate your existing Python environment
source ~/Documents/PythonAPI/aiman/bin/activate

# Run the first script
echo "Running CAD_FL.py..."
python CAD_FL.py

# Run the second script
echo "Running 3.fedFL.py..."
python 3.fedFL.py

echo "Both scripts finished!"
