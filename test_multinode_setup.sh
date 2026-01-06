#!/bin/bash

# Test script to verify multi-node Ray setup
# Run this on each node to verify connectivity

echo "==================================="
echo "Multi-Node Setup Test"
echo "==================================="

# Check Python and packages
echo ""
echo "1. Checking Python and dependencies..."
python -c "import ray; print(f'Ray version: {ray.__version__}')" || echo "ERROR: Ray not installed"
python -c "import torch; print(f'PyTorch version: {torch.__version__}')" || echo "ERROR: PyTorch not installed"
python -c "import vllm; print(f'vLLM version: {vllm.__version__}')" || echo "ERROR: vLLM not installed"

# Check GPU visibility
echo ""
echo "2. Checking GPU visibility..."
nvidia-smi --query-gpu=index,name,memory.total --format=csv || echo "ERROR: No GPUs detected"
python -c "import torch; print(f'PyTorch sees {torch.cuda.device_count()} GPUs')"

# Check network interfaces
echo ""
echo "3. Checking network interfaces..."
ip addr show | grep "inet " | grep -v "127.0.0.1"

# Get hostname and IP
echo ""
echo "4. Node information..."
echo "Hostname: $(hostname)"
echo "IP Address: $(hostname --ip-address)"

# Check /dev/shm space
echo ""
echo "5. Checking /dev/shm space..."
df -h /dev/shm

# Test Ray basic functionality
echo ""
echo "6. Testing Ray basic functionality..."
python << 'PYTHON_END'
import ray
import sys

try:
    # Initialize Ray locally
    ray.init(address="local", include_dashboard=False, ignore_reinit_error=True)

    @ray.remote
    def test_func():
        return "Ray is working!"

    result = ray.get(test_func.remote())
    print(f"✓ Ray test successful: {result}")

    # Shutdown
    ray.shutdown()
    sys.exit(0)
except Exception as e:
    print(f"✗ Ray test failed: {e}")
    sys.exit(1)
PYTHON_END

echo ""
echo "==================================="
echo "Test Complete"
echo "==================================="
