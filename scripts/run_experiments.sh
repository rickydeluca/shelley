#!/bin/bash

# Directory containing YAML configuration files
CONFIG_DIR="experiments"

# Function to handle interrupt and skip to the next file
handle_interrupt() {
    echo "Keyboard interrupt detected. Skipping to the next config file."
}

# Trap SIGINT (Ctrl+C) and call the function
trap handle_interrupt SIGINT

# Loop through all .yaml files in the directory
for config_file in "$CONFIG_DIR"/*.yaml; do
    # Extract just the file name without the directory path
    config_name=$(basename "$config_file")
    
    # Run the python command with the current config file
    echo "Running: python train_eval.py -e $config_file"
    
    # Run the python command and handle any interruption by skipping to the next file
    python train_eval.py -e "$config_file"
    
    if [ $? -ne 0 ]; then
        echo "An error occurred while running: $config_file. Continuing with the next file."
    fi
done

# Reset the trap back to default once the script finishes
trap - SIGINT
