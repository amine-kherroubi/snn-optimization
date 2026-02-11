#!/bin/bash

# Test CUDA implementations with different hidden layer sizes
HIDDEN_SIZES=(128 256 1024)
RESULTS_FILE="neurones_test_results.csv"

echo "Implementation,Hidden_Size,Dataset,Time_Seconds" > $RESULTS_FILE

# Test each implementation
for FILE in nn_cuda_reference.cu nn_cuda_combined.cu; do
    NAME="${FILE%.cu}"
    echo "Testing $FILE..."
    
    cp "$FILE" "${FILE}.backup"
    
    for HIDDEN in "${HIDDEN_SIZES[@]}"; do
        echo "  Hidden size: $HIDDEN"
        
        sed -i "s/#define HIDDEN_SIZE [0-9]\+/#define HIDDEN_SIZE $HIDDEN/" "$FILE"
        nvcc -O3 -Xcompiler -fopenmp "$FILE" -o "test_${HIDDEN}" 2>/dev/null || continue
        
        for DATASET in small medium large; do
            echo "    $DATASET..."
            OUTPUT=$(./test_${HIDDEN} ../reference/data/synthetic_convex_${DATASET}.csv 2>&1)
            TIME=$(echo "$OUTPUT" | grep "Average training time")
            echo "$NAME,$HIDDEN,$DATASET,$TIME" >> $RESULTS_FILE
        done
        
        rm -f "test_${HIDDEN}"
    done
    
    mv "${FILE}.backup" "$FILE"
done

echo "Done! Results in $RESULTS_FILE"
cat $RESULTS_FILE
