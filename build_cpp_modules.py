import os
import sys
import torch
from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
import subprocess

def main():
    print("===== BIÊN DỊCH MODULE C++ CHO TOOLFILMAI =====")
    
    # Kiểm tra CUDA
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    # Tạo thư mục cpp nếu chưa có
    os.makedirs("cpp", exist_ok=True)
    
    # Kiểm tra các file C++ đã tồn tại chưa
    files_exist = (
        os.path.exists("cpp/optical_flow.cpp") and 
        os.path.exists("cpp/video_extract.cpp") and
        os.path.exists("cpp/video_concat.cpp")
    )
    
    if not files_exist:
        print("⚠️ Các file C++ cần thiết không đầy đủ")
        return
    
    # Kiểm tra CUDA file nếu CUDA khả dụng
    if cuda_available and not os.path.exists("cpp/optical_flow_cuda.cu"):
        print("⚠️ Thiếu file optical_flow_cuda.cu cần thiết cho GPU acceleration")
        return
    
    # Kiểm tra CMake và Compiler
    try:
        subprocess.run(["cmake", "--version"], check=True, stdout=subprocess.PIPE)
        print("✅ Đã tìm thấy CMake")
    except:
        print("❌ Không tìm thấy CMake. Vui lòng cài đặt CMake trước khi tiếp tục")
        return
    
    # Xây dựng extensions
    try:
        print("\nĐang chuẩn bị biên dịch...")
        
        extensions = []
        
        # Optical Flow Extension
        if cuda_available:
            print("Cấu hình optical_flow_cpp với hỗ trợ CUDA")
            extensions.append(
                CUDAExtension(
                    name="optical_flow_cpp",
                    sources=["cpp/optical_flow.cpp", "cpp/optical_flow_cuda.cu"],
                    extra_compile_args={'cxx': ['-DWITH_CUDA'], 'nvcc': ['-DWITH_CUDA']}
                )
            )
        else:
            print("Cấu hình optical_flow_cpp không có CUDA")
            extensions.append(
                CppExtension(
                    name="optical_flow_cpp",
                    sources=["cpp/optical_flow.cpp"]
                )
            )
        
        # Video Extract Extension
        extensions.append(
            CppExtension(
                name="video_extract_cpp",
                sources=["cpp/video_extract.cpp"]
            )
        )
        
        # Video Concat Extension
        extensions.append(
            CppExtension(
                name="video_concat_cpp",
                sources=["cpp/video_concat.cpp"]
            )
        )
        
        # Build all extensions
        print("\nĐang biên dịch modules...")
        setup(
            name="toolfilmai_cpp_modules",
            ext_modules=extensions,
            cmdclass={
                'build_ext': BuildExtension
            },
            script_args=['build_ext', '--inplace']
        )
        
        print("\n✅ Biên dịch hoàn tất. Các modules đã sẵn sàng để sử dụng!")
        
    except Exception as e:
        print(f"\n❌ Lỗi khi biên dịch modules: {e}")
        print("\nVui lòng kiểm tra lỗi và thử lại. Có thể bạn cần cài đặt thêm các thư viện phát triển.")

if __name__ == "__main__":
    main()
