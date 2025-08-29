# Check if GPU IDs are provided as arguments
if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <gpu_ids>"
  echo "Example: $0 0,1,2"
  exit 1
fi

# Get GPU IDs from arguments
GPU_IDS=$1

# Compute number of processes per node
IFS=',' read -ra GPU_ARRAY <<< "$GPU_IDS"
NUM_GPUS=${#GPU_ARRAY[@]}

# Print selected GPUs and number of processes
echo "Using GPUs: $GPU_IDS"
echo "Number of processes per node: $NUM_GPUS"

# Function to find an available port, starting from 29500
find_available_port() {
  local port=29500
  while netstat -tulnp 2>/dev/null | grep -q ":$port "; do
    echo "Port $port is in use, trying $((port+1))..."
    ((port++))
  done
  echo $port  # Only print the port number
}

# Get available port and store it correctly
MASTER_PORT=$(find_available_port | tail -n 1)
echo "Using master port: $MASTER_PORT"

# Run the training script
CUDA_VISIBLE_DEVICES=$GPU_IDS torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT train_flow_model.py