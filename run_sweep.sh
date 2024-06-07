#!/bin/bash

# Set the Python script name as a variable
python_script='Simplified_HW_GECNET.py'
epochs=150
batch_size=16384
lr=2e-4
scaling=4
#assumes 9-bit ADC, 7-bit DAC
#gpus=(2 3 5 7 8)
gpus=(1 4 6 9) #1080s

echo "Running sweep for the Python script: $python_script"

# Function to check if a GPU is in use
is_gpu_in_use() {
    local gpu=$1
    echo $1 >&2
    nvidia-smi -i $gpu | grep 'No running processes found' -vqz
    local status=$?
    echo "Status: $status" >&2
    return $status
}

# Function to wait until a GPU is free
wait_for_free_gpu() {
    local gpu
    while true; do
        for gpu in "${gpus[@]}"; do
            if ! is_gpu_in_use $gpu; then
                echo $gpu
                return
            fi
        done
        echo "All GPUs in use, waiting..." >&2
        sleep 100
    done
}
##########SKIPPING 8 FOR NOW
# Iterate over the specified values for the slices
for i in 6 10 12 14 14
do
    # Wait for a free GPU
    if [ $i -eq 32 ]; then
        batch_size=$((batch_size / scaling))
        lr=$(awk "BEGIN {print $lr / $scaling}")
    fi
    echo "Trying to access GPU"
    gpu=$(wait_for_free_gpu)
    echo "PCM - Running with slices=$i and GPU=$gpu"
    # Run the Python script with different arguments
    python "$python_script" --slices=$i --epochs=$epochs --pcm --gpu=$gpu --batch_size=$batch_size --learning_rate=$lr &
    # Wait for the Python script to start using the GPU
    if [ $i -eq 32 ]; then
        batch_size=$((batch_size * scaling))
        lr=$(awk "BEGIN {print $lr * $scaling}")
    fi
    sleep 100
done
# Iterate over the specified values for the slices
for i in 6 10 12 14 14 #32 #2 4 8 16 #skip 1 because we already did it
do
    # Wait for a free GPU
    if [ $i -eq 32 ]; then
        batch_size=$((batch_size / scaling))
        lr=$(awk "BEGIN {print $lr / $scaling}")
    fi
    gpu=$(wait_for_free_gpu)
    echo "RRAM - Running with slices=$i and GPU=$gpu"
    # Run the Python script with different arguments
    python "$python_script" --slices=$i --epochs=$epochs --rram --gpu=$gpu --batch_size=$batch_size --learning_rate=$lr &
    #Wait for the Python script to start using the GPU
    if [ $i -eq 32 ]; then
        batch_size=$((batch_size * scaling))
        lr=$(awk "BEGIN {print $lr * $scaling}")
    fi
    sleep 100
done

# # Set the slices value as a variable
# slices=8

# # Iterate over the specified values for the drift
# for drift in 0.0 1.0 86400.0 172800.0
# do
#     # Wait for a free GPU
#     gpu=$(wait_for_free_gpu)
#     # Run the Python script with different arguments
#     python "$python_script" --slices=$slices --epochs=$epochs --pcm --drift=$drift --gpu=$gpu --batch_size=$batch_size --learning_rate=$lr &
# done

# # Iterate over the specified values for the drift
# for drift in 0.0 1.0 86400.0 172800.0
# do
#     # Wait for a free GPU
#     gpu=$(wait_for_free_gpu)
#     # Run the Python script with different arguments
#     python "$python_script" --slices=$slices --epochs=$epochs --rram --drift=$drift --gpu=$gpu --batch_size=$batch_size --learning_rate=$lr &
# done