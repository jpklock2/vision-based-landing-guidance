#!/bin/bash

# Replace 'LoRAT' and 'LoRAT_replacements' with the actual paths to your folders
source_dir="LoRAT_replacements"
target_dir="LoRAT"

# Recursively copy files from source to target, overwriting existing files
cp -rf "$source_dir"/* "$target_dir"