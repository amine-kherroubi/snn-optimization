#!/bin/bash

# Test different network architectures on all datasets
RESULTS_FILE="network_test_results.csv"

echo "Implementation,Layers,Dataset,Time_Seconds" > $RESULTS_FILE

# Files to test
FILES=(
  "nn_cuda_reference.cu:reference:1"
  "nn_cuda_combined.cu:combined:1"
  "nn_cuda_reference_two_layers.cu:reference:2"
  "nn_cuda_combined_two_layers.cu:combined:2"
  "nn_cuda_reference_three_layers.cu:reference:3"
  "nn_cuda_combined_three_layers.cu:combined:3"
)

for FILE_INFO in "${FILES[@]}"; do
  IFS=':' read -r FILE NAME LAYERS <<< "$FILE_INFO"
  echo "Testing $FILE ($LAYERS hidden layers)..."
  
  # Compile
  nvcc -O3 -Xcompiler -fopenmp "$FILE" -o "test_network" 2>/dev/null || continue
  
  for DATASET in small medium large; do
    echo "  $DATASET..."
    OUTPUT=$(./test_network ../reference/data/synthetic_convex_${DATASET}.csv 2>&1)
    TIME=$(echo "$OUTPUT" | grep "Average training time" | awk '{print $6}')
    echo "$NAME,$LAYERS,$DATASET,$TIME" >> $RESULTS_FILE
  done
  
  rm -f test_network
done

echo "Done! Results in $RESULTS_FILE"
cat $RESULTS_FILE
