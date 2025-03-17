import os
import torch
import numpy as np
import subprocess
import sys
import platform

def get_gpu_info():
    """Get GPU information"""
    # Check forced CUDA first
    force_cuda = os.environ.get('FORCE_CUDA', '0') == '1'
    has_cuda = torch.cuda.is_available()
    
    if not has_cuda and not force_cuda:
        return {
            'available': False,
            'devices': [],
            'force_cuda': force_cuda
        }
    
    # We have CUDA or FORCE_CUDA is enabled
    device_count = torch.cuda.device_count() if has_cuda else 1
    devices = []
    
    for i in range(device_count) if has_cuda else [0]:
        if has_cuda:
            device_name = torch.cuda.get_device_name(i)
            props = torch.cuda.get_device_properties(i)
            total_memory = props.total_memory / 1e9  # Convert to GB
            devices.append({
                'index': i,
                'name': device_name,
                'total_memory_gb': total_memory,
                'compute_capability': f"{props.major}.{props.minor}"
            })
        else:
            devices.append({
                'index': 0,
                'name': 'Forced CUDA Device',
                'total_memory_gb': 0.0,
                'compute_capability': 'unknown'
            })
    
    return {
        'available': True,
        'device_count': device_count,
        'devices': devices,
        'force_cuda': force_cuda
    }

def optimize_for_video_processing(device_id=0):
    """Apply optimal settings for video processing on GPU"""
    if not torch.cuda.is_available():
        return False
    
    # Force PyTorch to use specified GPU
    torch.cuda.set_device(device_id)
    
    # Apply performance optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    # Enable TF32 if available (on Ampere or newer GPUs)
    if hasattr(torch.backends.cuda, 'matmul') and hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
        torch.backends.cuda.matmul.allow_tf32 = True
    
    # Clear GPU memory
    torch.cuda.empty_cache()
    
    return True

def force_gpu_usage():
    """Buộc sử dụng GPU ngay cả khi torch.cuda.is_available() trả về False"""
    force_cuda = os.environ.get('FORCE_CUDA', '0')
    if force_cuda == '1':
        # Thử khởi tạo CUDA trực tiếp
        try:
            # Tìm thư viện CUDA
            cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
            if not cuda_home:
                if os.path.exists('C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA'):
                    for cuda_dir in os.listdir('C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA'):
                        if cuda_dir.startswith('v'):
                            cuda_home = f'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/{cuda_dir}'
                            break
            
            if cuda_home:
                os.environ['CUDA_HOME'] = cuda_home
                print(f"✅ Đã thiết lập CUDA_HOME = {cuda_home}")
                
                # Kiểm tra và đặt biến môi trường CUDA
                if not os.environ.get('CUDA_VISIBLE_DEVICES'):
                    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
                
                return True
        except Exception as e:
            print(f"⚠️ Không thể buộc dùng GPU: {e}")
    
    return False

def check_cpp_dependencies():
    """Check for necessary C++ dependencies and provide installation instructions if missing"""
    print("Checking C++ dependencies...")
    
    # Check for CMake
    cmake_found = False
    try:
        cmake_version = subprocess.check_output(["cmake", "--version"], 
                                              stderr=subprocess.STDOUT,
                                              text=True)
        cmake_found = True
        print(f"✅ CMake found: {cmake_version.split()[2]}")
    except (subprocess.SubprocessError, FileNotFoundError):
        print("❌ CMake not found")
    
    # Check for compiler
    compiler_found = False
    compiler_name = ""
    
    if platform.system() == "Windows":
        # Check for MSVC
        try:
            msvc_output = subprocess.check_output(["cl"], 
                                                stderr=subprocess.STDOUT,
                                                text=True)
            compiler_found = True
            compiler_name = "MSVC"
            print("✅ Microsoft Visual C++ compiler found")
        except (subprocess.SubprocessError, FileNotFoundError):
            print("❌ Microsoft Visual C++ compiler not found")
    else:
        # Check for GCC/g++
        try:
            gcc_version = subprocess.check_output(["g++", "--version"], 
                                                stderr=subprocess.STDOUT,
                                                text=True)
            compiler_found = True
            compiler_name = "GCC"
            print(f"✅ GCC/G++ found: {gcc_version.split()[3]}")
        except (subprocess.SubprocessError, FileNotFoundError):
            print("❌ GCC/G++ not found")
    
    # Check OpenCV
    try:
        import cv2
        opencv_found = True
        print(f"✅ OpenCV found: {cv2.__version__}")
    except ImportError:
        opencv_found = False
        print("❌ OpenCV not found")
    
    # Check CUDA for C++ compilation
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    if cuda_home and os.path.exists(cuda_home):
        print(f"✅ CUDA found at: {cuda_home}")
        
        # Check nvcc compiler
        try:
            nvcc_version = subprocess.check_output([os.path.join(cuda_home, "bin", "nvcc"), "--version"], 
                                                stderr=subprocess.STDOUT,
                                                text=True)
            print(f"✅ NVCC compiler found: {nvcc_version.split('release')[-1].strip()}")
        except (subprocess.SubprocessError, FileNotFoundError):
            print("❌ NVCC compiler not found")
    else:
        print("❌ CUDA not found for C++ compilation")
    
    # Print advice if dependencies are missing
    if not (cmake_found and compiler_found and opencv_found):
        print("\n⚠️ Some C++ dependencies are missing. The C++ extensions will not be built.")
        
        missing_deps = []
        if not cmake_found:
            missing_deps.append("CMake (https://cmake.org/download/)")
        if not compiler_found:
            if platform.system() == "Windows":
                missing_deps.append("Visual Studio with C++ workload (https://visualstudio.microsoft.com/)")
            else:
                missing_deps.append("GCC/G++ (apt install g++ or equivalent)")
        if not opencv_found:
            missing_deps.append("OpenCV (pip install opencv-python)")
        
        print("\nMissing dependencies:")
        for dep in missing_deps:
            print(f"  - {dep}")
        
        print("\nAfter installing these dependencies, run build_cpp_modules.bat (Windows) or build_cpp_modules.sh (Linux/Mac)")
        return False
    
    return True

def build_cpp_extensions():
    """Biên dịch các C++ extensions nếu chưa có"""
    # Check dependencies first
    if not check_cpp_dependencies():
        print("⚠️ Skipping C++ extension build due to missing dependencies")
        return False

    try:
        from torch.utils.cpp_extension import load

        # Kiểm tra xem extension đã được biên dịch chưa
        import os
        if not os.path.exists("optical_flow_cpp.so") and not os.path.exists("optical_flow_cpp.pyd"):
            print("🔄 Đang biên dịch C++ extensions lần đầu tiên...")
            
            # Tạo thư mục cpp nếu chưa có
            if not os.path.exists("cpp"):
                os.makedirs("cpp", exist_ok=True)
            
            # Copy code từ template nếu chưa có
            for file_name in ["optical_flow.cpp", "video_extract.cpp", "video_concat.cpp"]:
                cpp_file = os.path.join("cpp", file_name)
                if not os.path.exists(cpp_file):
                    with open(cpp_file, "w") as f:
                        f.write("// Template for " + file_name)
            
            # Biên dịch
            try:
                optical_flow_cpp = load(
                    name="optical_flow_cpp",
                    sources=["cpp/optical_flow.cpp"],
                    verbose=True
                )
                
                video_extract_cpp = load(
                    name="video_extract_cpp",
                    sources=["cpp/video_extract.cpp"],
                    extra_cflags=["-O3"],
                    verbose=True
                )
                
                video_concat_cpp = load(
                    name="video_concat_cpp",
                    sources=["cpp/video_concat.cpp"],
                    verbose=True
                )
                
                print("✅ Biên dịch C++ extensions thành công!")
                return True
            except Exception as e:
                print(f"⚠️ Lỗi khi biên dịch C++ extensions: {e}")
        else:
            print("✅ C++ extensions đã được biên dịch trước đó")
            return True
            
    except ImportError as e:
        print(f"⚠️ Không thể import torch.utils.cpp_extension: {e}")
    
    return False

def try_load_cpp_extensions():
    """Try to load C++ extensions if available"""
    extensions = {}
    try:
        from torch.utils.cpp_extension import load
        
        # Create cpp directory if it doesn't exist
        if not os.path.exists("cpp"):
            print("Creating cpp directory for C++ extensions...")
            os.makedirs("cpp", exist_ok=True)
            return {}
        
        # Check for optical flow extension
        if os.path.exists("cpp/optical_flow.cpp"):
            print("Loading optical flow C++ extension...")
            try:
                sources = ["cpp/optical_flow.cpp"]
                # Add CUDA source if available
                if os.path.exists("cpp/optical_flow_cuda.cu"):
                    sources.append("cpp/optical_flow_cuda.cu")
                    print("CUDA source found for optical flow")
                
                optical_flow_cpp = load(
                    name="optical_flow_cpp",
                    sources=sources,
                    extra_cflags=["-O3"],
                    verbose=True
                )
                print("✅ Optical flow C++ extension loaded successfully")
                extensions["optical_flow"] = optical_flow_cpp
            except Exception as e:
                print(f"⚠️ Failed to load optical flow extension: {e}")
        
        # Check for video extraction extension
        if os.path.exists("cpp/video_extract.cpp"):
            print("Loading video extraction C++ extension...")
            try:
                video_extract_cpp = load(
                    name="video_extract_cpp",
                    sources=["cpp/video_extract.cpp"],
                    extra_cflags=["-O3"],
                    verbose=True
                )
                print("✅ Video extraction C++ extension loaded successfully")
                extensions["video_extract"] = video_extract_cpp
            except Exception as e:
                print(f"⚠️ Failed to load video extraction extension: {e}")
        
        # Check for video concatenation extension
        if os.path.exists("cpp/video_concat.cpp"):
            print("Loading video concatenation C++ extension...")
            try:
                video_concat_cpp = load(
                    name="video_concat_cpp",
                    sources=["cpp/video_concat.cpp"],
                    extra_cflags=["-O3", "-I/usr/include/ffmpeg"],
                    extra_ldflags=["-lavformat", "-lavcodec", "-lavutil", "-lswscale"],
                    verbose=True
                )
                print("✅ Video concatenation C++ extension loaded successfully")
                extensions["video_concat"] = video_concat_cpp
            except Exception as e:
                print(f"⚠️ Failed to load video concatenation extension: {e}")
        
        if not extensions:
            print("⚠️ No C++ extensions were loaded")
    except Exception as e:
        print(f"⚠️ Could not load C++ extensions: {e}")
    
    return extensions

def debug_gpu_detection():
    """Chi tiết về phát hiện GPU và cài đặt CUDA"""
    print("\n===== DEBUG THÔNG TIN GPU =====")
    
    # Kiểm tra PyTorch
    print(f"PyTorch version: {torch.__version__}")
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"PyTorch CUDA version: {torch.version.cuda}")
        print(f"PyTorch cuDNN version: {torch.backends.cudnn.version()}")
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Kiểm tra CUDA Environment
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    print(f"CUDA_HOME: {cuda_home or 'Not set'}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    print(f"FORCE_CUDA: {os.environ.get('FORCE_CUDA', 'Not set')}")
    
    # Kiểm tra NVIDIA driver
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            first_lines = result.stdout.split('\n')[:3]
            print("\nNVIDIA-SMI:")
            for line in first_lines:
                print(f"  {line}")
        else:
            print("nvidia-smi không khả dụng")
    except:
        print("Không thể chạy nvidia-smi")
    
    print("===============================\n")
    
    # Nếu FORCE_CUDA = 1 nhưng CUDA không khả dụng, hiện thông báo
    if os.environ.get('FORCE_CUDA', '0') == '1' and not torch.cuda.is_available():
        print("⚠️ FORCE_CUDA đã được cài đặt nhưng PyTorch vẫn không phát hiện GPU.")
        print("   Nguyên nhân có thể là:")
        print("   1. PyTorch không được cài đặt với hỗ trợ CUDA")
        print("   2. Trình điều khiển NVIDIA không tương thích")
        print("   3. Phiên bản CUDA toolkit không phù hợp với PyTorch")
        print("\nGợi ý: Cài đặt lại PyTorch với CUDA support từ https://pytorch.org/get-started/locally/")
