#!/bin/bash

# Run raytracer script with multiple configurations
# Usage: ./run.sh <executable> [commands...]
# Example: ./run.sh raytracer-cuda
#          ./run.sh raytracer-cpu

# Check if executable is provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <executable>"
    echo "Example: $0 raytracer-cuda"
    echo "         $0 raytracer-cpu"
    exit 1
fi

EXECUTABLE_PATH="$1"
# Get the executable name (extract just the filename from path)
EXECUTABLE_NAME=$(basename "$1")

# Check if executable exists
if [ ! -f "$EXECUTABLE_PATH" ]; then
    echo "Error: Executable '$EXECUTABLE_PATH' not found!"
    exit 1
fi

# Shared log file
LOG_FILE="logs/$EXECUTABLE_NAME.log"

# Create logs directory if it doesn't exist
mkdir -p "$(dirname "$LOG_FILE")"

mkdir -p "outputs/$EXECUTABLE_NAME"

# Clear previous log file
> "$LOG_FILE"

# Define multiple raytracer commands
# Format: "width samples max_bounce output_name"
declare -a COMMANDS=(
    "800 10 16"
    "800 100 16"
    "800 1000 16"
    "800 10000 16"
    "800 100 4"
    "800 100 8"
    "800 100 32"
    "800 100 64"
)

# Function to run a single raytracer command
run_raytracer() {
    local width=$1
    local samples=$2
    local max_bounce=$3

    local output_ppm="outputs/$EXECUTABLE_NAME/w${width}_s${samples}_b${max_bounce}.ppm"
    local output_png="outputs/$EXECUTABLE_NAME/w${width}_s${samples}_b${max_bounce}.png"

    echo "========================================" | tee -a "$LOG_FILE"
    echo "Running $EXECUTABLE_NAME:" | tee -a "$LOG_FILE"
    echo "  Width: $width" | tee -a "$LOG_FILE"
    echo "  Samples: $samples" | tee -a "$LOG_FILE"
    echo "  Max bounce: $max_bounce" | tee -a "$LOG_FILE"
    echo "  Output: $output_png" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"

    # Run raytracer executable
    # stdout -> output.ppm
    # stderr -> appended to LOG_FILE
    "$EXECUTABLE_PATH" "$width" "$samples" "$max_bounce" > "$output_ppm" 2>> "$LOG_FILE"

    # Check if raytracer succeeded
    if [ $? -ne 0 ]; then
        echo "Error: Raytracer failed for $output_name! Check $LOG_FILE for details." | tee -a "$LOG_FILE"
        return 1
    fi

    # Check if PPM file was created
    if [ ! -f "$output_ppm" ]; then
        echo "Error: $output_ppm was not created!" | tee -a "$LOG_FILE"
        return 1
    fi

    # Convert PPM to PNG
    echo "Converting $output_ppm to $output_png..." | tee -a "$LOG_FILE"
    convert "$output_ppm" "$output_png" 2>> "$LOG_FILE"

    if [ $? -ne 0 ]; then
        echo "Error: Failed to convert PPM to PNG!" | tee -a "$LOG_FILE"
        return 1
    fi

    echo "Success! Image saved to $output_png" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
}

# Main execution
echo "Starting batch raytracer execution..."
echo "Executable: $EXECUTABLE_NAME"
echo "All logs will be saved to: $LOG_FILE"
echo ""

# Run all commands
for cmd in "${COMMANDS[@]}"; do
    read -r width samples max_bounce <<< "$cmd"
    run_raytracer "$width" "$samples" "$max_bounce"

    if [ $? -ne 0 ]; then
        echo "Stopping execution due to error."
        exit 1
    fi
done

echo "========================================"
echo "All raytracer runs completed!"
echo "Logs saved to: $LOG_FILE"
echo "========================================"
