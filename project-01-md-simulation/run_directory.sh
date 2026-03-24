#!/bin/bash

# Find all directories containing "run_cluster.sh"
find_directories_with_script() {
  find . -type f -name "run_cluster.sh" -exec dirname {} \;
}

# Run "run_cluster.sh" in each found directory
run_scripts() {
  local directories=("$@")

  for dir in "${directories[@]}"; do
    echo "Entering directory: $dir"
    (cd "$dir" && bash run_cluster.sh) || echo "Failed to execute run_cluster.sh in $dir"
  done
}

# Main script execution
main() {
  echo "Finding all directories containing run_cluster.sh..."
  mapfile -t directories < <(find_directories_with_script)

  if [[ ${#directories[@]} -eq 0 ]]; then
    echo "No run_cluster.sh scripts found!"
    exit 1
  fi

  echo "Found directories: ${directories[*]}"
  echo "Starting simulations..."
  run_scripts "${directories[@]}"
  echo "All simulations started."
}

# Call the main function
main