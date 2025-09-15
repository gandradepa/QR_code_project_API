#!/bin/bash
# This script activates the virtual environment and then runs a Python script.
# It ensures that the script runs with all the necessary dependencies and
# environment variables, regardless of how it was called.

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Path to the virtual environment's activate script, relative to the project structure
VENV_ACTIVATE="/home/developer/Dashboard/venv/bin/activate"

# Check if the activate script exists
if [ ! -f "$VENV_ACTIVATE" ]; then
    echo "Error: Virtual environment activate script not found at $VENV_ACTIVATE" >&2
    exit 1
fi

# Activate the virtual environment
source "$VENV_ACTIVATE"

# The first argument to this script is the Python script to run.
# The 'exec' command replaces the shell script process with the Python process.
# "$@" passes all other arguments to the Python script.
exec python "$@"
