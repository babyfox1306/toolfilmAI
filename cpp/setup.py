from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import os
import glob
import setuptools

__version__ = '0.1.0'

# Detect if CUDA is available
cuda_available = False
cuda_home = os.environ.get('CUDA_HOME', os.environ.get('CUDA_PATH'))
if cuda_home and os.path.exists(cuda_home):
    cuda_available = True
    print(f"CUDA found at: {cuda_home}")
else:
    print("CUDA not found, building without CUDA support")

# Extension for optical flow with CUDA support if available
optical_flow_sources = ['optical_flow.cpp']
optical_flow_defines = []
optical_flow_libraries = []
optical_flow_include_dirs = []
optical_flow_extra_objects = []
optical_flow_extra_compile_args = []

if cuda_available:
    optical_flow_sources.append('optical_flow_cuda.cu')
    optical_flow_defines.append(('WITH_CUDA', None))
    optical_flow_include_dirs.append(os.path.join(cuda_home, 'include'))
    
    # Add CUDA libraries
    if sys.platform == 'win32':
        optical_flow_libraries.extend(['cudart', 'cuda'])
        optical_flow_extra_compile_args.append('/DWITH_CUDA')
    else:
        optical_flow_libraries.extend(['cudart', 'cuda'])
        optical_flow_extra_compile_args.append('-DWITH_CUDA')

# Find OpenCV include and lib paths
def find_opencv():
    # Try common installation locations
    opencv_paths = [
        "C:/opencv/build",  # Windows common path
        "/usr/local",       # Linux/macOS common path
        "/opt/homebrew",    # macOS Homebrew
    ]
    
    for base_path in opencv_paths:
        include_dir = os.path.join(base_path, "include")
        if os.path.exists(os.path.join(include_dir, "opencv2", "opencv.hpp")):
            # Found OpenCV include directory
            lib_paths = glob.glob(os.path.join(base_path, "lib", "opencv_*.lib")) + \
                       glob.glob(os.path.join(base_path, "lib", "libopencv_*.so")) + \
                       glob.glob(os.path.join(base_path, "lib", "libopencv_*.dylib"))
            
            if lib_paths:
                library_dirs = [os.path.dirname(lib_paths[0])]
                libraries = [os.path.splitext(os.path.basename(lib))[0].replace('lib', '') 
                            for lib in lib_paths]
                return include_dir, library_dirs, libraries
    
    return None, [], []

opencv_include_dir, opencv_library_dirs, opencv_libraries = find_opencv()
if opencv_include_dir:
    optical_flow_include_dirs.append(opencv_include_dir)
    print(f"Found OpenCV at: {opencv_include_dir}")
else:
    print("WARNING: OpenCV not found. Please install OpenCV or provide its location.")

# Define extensions
ext_modules = [
    Extension(
        'optical_flow_cpp',
        sources=optical_flow_sources,
        define_macros=optical_flow_defines,
        include_dirs=optical_flow_include_dirs,
        libraries=optical_flow_libraries,
        library_dirs=opencv_library_dirs,
        extra_compile_args=optical_flow_extra_compile_args,
    ),
    Extension(
        'video_extract_cpp',
        sources=['video_extract.cpp'],
        include_dirs=[opencv_include_dir] if opencv_include_dir else [],
        library_dirs=opencv_library_dirs,
        libraries=opencv_libraries,
    ),
    Extension(
        'video_concat_cpp',
        sources=['video_concat.cpp'],
        # FFmpeg configuration would go here
    ),
]

# Avoid rebuilding on pybind11 install
class BuildExt(build_ext):
    def has_flag(self, flagname):
        """Return a boolean indicating whether a flag name is supported on the specified compiler."""
        import tempfile
        import os
        with tempfile.NamedTemporaryFile('w', suffix='.cpp') as f:
            f.write('int main (int argc, char **argv) { return 0; }')
            try:
                self.compiler.compile([f.name], extra_postargs=[flagname])
            except setuptools.distutils.errors.CompileError:
                return False
        return True

    def build_extensions(self):
        ct = self.compiler.compiler_type
        
        # Add compiler-specific options
        opts = []
        if ct == 'unix':
            if self.has_flag('-std=c++14'):
                opts.append('-std=c++14')
        elif ct == 'msvc':
            opts.append('/std:c++14')
        
        # Add opts to all extensions
        for ext in self.extensions:
            ext.extra_compile_args += opts
        
        build_ext.build_extensions(self)

setup(
    name='toolfilmai_cpp',
    version=__version__,
    author='ToolfilmAI',
    author_email='user@example.com',
    description='C++ extensions for ToolfilmAI',
    long_description='',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExt},
    zip_safe=False,
)
