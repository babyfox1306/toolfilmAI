import torch
import os
import subprocess

def main():
    print("\n===== KIỂM TRA GPU CHI TIẾT =====")
    
    # Kiểm tra PyTorch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        device_count = torch.cuda.device_count()
        print(f"GPU Count: {device_count}")
        for i in range(device_count):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"  Total Memory: {props.total_memory / 1e9:.2f} GB")
            print(f"  Compute Capability: {props.major}.{props.minor}")
    
    # Test tensor creation on GPU
    print("\nTesting GPU tensor operations:")
    try:
        x = torch.tensor([1.0, 2.0], device="cuda" if torch.cuda.is_available() else "cpu")
        y = x + x
        print(f"  Tensor device: {x.device}")
        print(f"  2+2={y[0].item()}, 4+4={y[1].item()}")
        print("✅ GPU tensor operations work correctly!")
    except Exception as e:
        print(f"❌ Error creating GPU tensor: {e}")
    
    # Check CUDA_HOME
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    print(f"\nCUDA_HOME: {cuda_home or 'Not set'}")
    
    # Check NVIDIA-SMI
    try:
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            print("\nNVIDIA-SMI:")
            for i, line in enumerate(lines[:6]):
                print(f"  {line}")
        else:
            print("\n❌ nvidia-smi failed to run")
    except:
        print("\n❌ nvidia-smi not found")
    
    # Check environment variables
    print("\nEnvironment variables:")
    print(f"  FORCE_CUDA: {os.environ.get('FORCE_CUDA', 'Not set')}")
    print(f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    print(f"  PYTORCH_CUDA_ALLOC_CONF: {os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'Not set')}")

    print("\n===== KIỂM TRA HOÀN TẤT =====")

if __name__ == "__main__":
    main()
