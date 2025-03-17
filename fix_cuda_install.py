import os
import sys
import subprocess
import platform
import torch
from pathlib import Path

def check_cuda_installation():
    """Check CUDA installation and PyTorch configuration"""
    print("===== CUDA & PYTORCH INSTALLATION DIAGNOSIS =====")
    
    # Check PyTorch version
    print(f"PyTorch version: {torch.__version__}")
    print(f"PyTorch installation location: {torch.__file__}")
    print(f"CUDA available (PyTorch): {torch.cuda.is_available()}")
    
    # Check if PyTorch version has CPU or CUDA suffix
    if "+cpu" in torch.__version__:
        print("⚠️ PyTorch CPU version detected - this will not work with CUDA")
        needs_reinstall = True
    else:
        needs_reinstall = False
        if "+cu" in torch.__version__:
            cuda_version = torch.__version__.split("+cu")[1].split()[0]
            print(f"✅ PyTorch CUDA version detected: cu{cuda_version}")
        else:
            print("⚠️ PyTorch version format doesn't indicate CUDA/CPU version")
            needs_reinstall = True
    
    # Check for CUDA installation
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    if cuda_home and os.path.exists(cuda_home):
        print(f"✅ CUDA installation found at: {cuda_home}")
        
        # Check CUDA version
        nvcc_path = os.path.join(cuda_home, "bin", "nvcc")
        if os.path.exists(nvcc_path):
            try:
                result = subprocess.run([nvcc_path, "--version"], capture_output=True, text=True)
                version_str = result.stdout
                cuda_version = version_str.strip().split("release ")[-1].split(",")[0]
                print(f"CUDA version: {cuda_version}")
            except Exception as e:
                print(f"⚠️ Could not determine CUDA version: {e}")
        else:
            print("⚠️ NVCC not found in CUDA installation")
    else:
        print("⚠️ CUDA_HOME environment variable not set or invalid")
        
        # Try to find CUDA installation
        if platform.system() == "Windows":
            cuda_paths = Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA")
            if cuda_paths.exists():
                versions = [d for d in cuda_paths.iterdir() if d.is_dir() and d.name.startswith("v")]
                if versions:
                    latest = sorted(versions, key=lambda x: x.name)[-1]
                    print(f"Found CUDA installation at: {latest}")
                    print("Please set CUDA_HOME to this path")
                    cuda_home = str(latest)
                else:
                    print("⚠️ CUDA installation found but no version directories")
            else:
                print("⚠️ CUDA installation not found in default location")
        elif platform.system() == "Linux":
            # Check common Linux locations
            for path in ["/usr/local/cuda", "/usr/lib/cuda"]:
                if os.path.exists(path):
                    print(f"Found CUDA installation at: {path}")
                    cuda_home = path
                    break
    
    # Check NVIDIA driver
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            driver_info = result.stdout.split('\n')[2]
            print(f"NVIDIA driver: {driver_info.strip()}")
        else:
            print("⚠️ NVIDIA driver not found or not working properly")
    except Exception as e:
        print(f"⚠️ Could not run nvidia-smi: {e}")
    
    return needs_reinstall, cuda_home

def fix_pytorch_installation(cuda_home=None):
    """
    Fix PyTorch installation to use CUDA
    """
    print("\n===== FIXING PYTORCH INSTALLATION =====")
    
    # Determine CUDA version from installation
    cuda_version = None
    if cuda_home:
        try:
            nvcc_path = os.path.join(cuda_home, "bin", "nvcc")
            result = subprocess.run([nvcc_path, "--version"], capture_output=True, text=True)
            version_str = result.stdout
            # Extract version like 11.7 or 12.1
            cuda_version = version_str.strip().split("release ")[-1].split(",")[0]
            major_minor = ''.join(cuda_version.split('.')[:2])
            print(f"Detected CUDA version: {cuda_version} (will use cu{major_minor})")
            cuda_version = major_minor
        except Exception as e:
            print(f"Could not determine CUDA version: {e}")
    
    # If we couldn't detect CUDA version, use a safe default
    if not cuda_version:
        print("Could not determine CUDA version, using default (cu118)")
        cuda_version = "118"
    
    # Map CUDA versions to compatible ones for PyTorch
    cuda_version_map = {
        "121": "118",  # 12.1 -> use 11.8 for compatibility
        "120": "118",  # 12.0 -> use 11.8
        "118": "118",  # 11.8 is directly supported
        "117": "117",  # 11.7 is directly supported
        "116": "116",  # 11.6 is directly supported
        "115": "115",  # 11.5 is directly supported
        "113": "113",  # 11.3 is directly supported
        "111": "111",  # 11.1 is directly supported
    }
    
    # Use compatible version
    torch_cuda = cuda_version_map.get(cuda_version, "118")
    print(f"Using PyTorch CUDA version: {torch_cuda}")
    
    # Create pip command
    pip_cmd = [
        sys.executable, "-m", "pip", "install", "--force-reinstall", 
        f"torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu{torch_cuda}"
    ]
    
    print("\nRunning the following command to reinstall PyTorch with CUDA support:")
    print(" ".join(pip_cmd))
    
    try:
        result = subprocess.run(pip_cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("\n✅ PyTorch reinstalled successfully!")
            # Verify the installation
            try:
                # Reload PyTorch to reflect new installation
                import importlib
                importlib.reload(torch)
                print(f"New PyTorch version: {torch.__version__}")
                print(f"CUDA available: {torch.cuda.is_available()}")
                if torch.cuda.is_available():
                    print(f"CUDA version: {torch.version.cuda}")
                    print(f"GPU device: {torch.cuda.get_device_name(0)}")
                else:
                    print("⚠️ CUDA still not available after reinstall")
            except Exception as e:
                print(f"Error reloading PyTorch: {e}")
                print("Please restart your Python environment to use the new PyTorch installation")
        else:
            print("\n⚠️ PyTorch reinstallation failed:")
            print(result.stderr)
            print("\nPlease try to reinstall PyTorch manually with the appropriate CUDA version")
    except Exception as e:
        print(f"\n⚠️ Error during PyTorch reinstallation: {e}")
        print("\nPlease try to reinstall PyTorch manually with the appropriate CUDA version")

def fix_ultralytics_installation():
    """Fix Ultralytics (YOLOv8) installation for CUDA support"""
    print("\n===== FIXING YOLOV8 INSTALLATION =====")
    
    # First try to import ultralytics to check if it's installed
    try:
        import ultralytics
        print(f"Ultralytics version: {ultralytics.__version__}")
    except ImportError:
        print("Ultralytics not installed, installing...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "ultralytics"], check=True)
            print("✅ Ultralytics installed successfully")
        except Exception as e:
            print(f"⚠️ Error installing ultralytics: {e}")
            return
    
    # Reinstall ultralytics to make sure it's compatible with the new PyTorch
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "--force-reinstall", "ultralytics"], check=True)
        print("✅ Ultralytics reinstalled successfully")
    except Exception as e:
        print(f"⚠️ Error reinstalling ultralytics: {e}")

def fix_yolov5_installation():
    """Fix YOLOv5 installation"""
    print("\n===== FIXING YOLOV5 INSTALLATION =====")
    
    # Try to install yolov5 from pip
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "yolov5"], check=True)
        print("✅ YOLOv5 installed successfully from pip")
    except Exception as e:
        print(f"⚠️ Error installing YOLOv5 from pip: {e}")
        
        # Alternative: Clone and install from GitHub
        try:
            if not os.path.exists("yolov5"):
                print("Cloning YOLOv5 repository...")
                subprocess.run(["git", "clone", "https://github.com/ultralytics/yolov5.git"], check=True)
            
            # Install dependencies
            requirements_file = os.path.join("yolov5", "requirements.txt")
            if os.path.exists(requirements_file):
                print("Installing YOLOv5 dependencies...")
                subprocess.run([sys.executable, "-m", "pip", "install", "-r", requirements_file], check=True)
                print("✅ YOLOv5 dependencies installed")
            else:
                print("⚠️ YOLOv5 requirements.txt not found")
        except Exception as e2:
            print(f"⚠️ Error setting up YOLOv5 from GitHub: {e2}")

def main():
    """Main function to fix CUDA and PyTorch installation"""
    needs_reinstall, cuda_home = check_cuda_installation()
    
    if needs_reinstall:
        user_input = input("\nDo you want to fix PyTorch installation to use CUDA? (y/n): ").strip().lower()
        if user_input == 'y':
            fix_pytorch_installation(cuda_home)
            fix_ultralytics_installation()
            fix_yolov5_installation()
            
            print("\n===== INSTALLATION FIXES COMPLETE =====")
            print("Please restart your Python environment and run the application again.")
        else:
            print("Skipping PyTorch reinstallation.")
    else:
        print("\nPyTorch installation appears to be correctly configured for CUDA.")
        print("If you're still having issues, you can force a reinstall.")
        user_input = input("Force reinstall PyTorch with CUDA support? (y/n): ").strip().lower()
        if user_input == 'y':
            fix_pytorch_installation(cuda_home)
            fix_ultralytics_installation()
            fix_yolov5_installation()

if __name__ == "__main__":
    main()
