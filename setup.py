import os
import sys
import shutil
import subprocess
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
    class bdist_wheel(_bdist_wheel):
        def get_tag(self):
            python, abi, plat = super().get_tag()
            if plat.startswith('linux'):
                # Force PyPI compliant tag for binary wheels built on linux
                plat = 'manylinux2014_x86_64'
            return python, abi, plat
except ImportError:
    bdist_wheel = None

class MakeBuildExt(build_ext):
    def build_extensions(self):
        project_root = os.path.dirname(os.path.abspath(__file__))
        
        # 1. Run 'make all' via Makefile
        print("Building C/CUDA extensions via Makefile...")
        try:
            subprocess.check_call(['make', 'all'], cwd=project_root)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Failed to compile C/CUDA extensions via Makefile. "
                f"Ensure gcc and make are installed. Original error: {e}"
            ) from e
            
        # 2. Setup paths
        build_dir = os.path.join(project_root, 'build')
        pkg_dir = os.path.join(self.build_lib, 'addernet')
        os.makedirs(pkg_dir, exist_ok=True)
        
        # 3. Dynamically check for built shared libraries
        so_files_to_include = ['libaddernet.so', 'libaddernet_hdc.so']
        
        # Dynamically check if the optional CUDA backend was compiled
        if os.path.exists(os.path.join(build_dir, 'libaddernet_cuda.so')):
            print("CUDA extension found. Including libaddernet_cuda.so in package.")
            so_files_to_include.append('libaddernet_cuda.so')
        else:
            print("Optional CUDA extension (libaddernet_cuda.so) not found. Skipping.")

        # Update package_data inside the distribution object so setuptools bundles them
        if 'addernet' not in self.distribution.package_data:
            self.distribution.package_data['addernet'] = []
        for so_file in so_files_to_include:
            if so_file not in self.distribution.package_data['addernet']:
                self.distribution.package_data['addernet'].append(so_file)

        # 4. Copy the compiled shared libraries to the wheel build directory
        so_files_to_copy = ['libaddernet.so', 'libaddernet_hdc.so']
        if os.path.exists(os.path.join(build_dir, 'libaddernet_cuda.so')):
            so_files_to_copy.append('libaddernet_cuda.so')

        for lib in so_files_to_copy:
            src_lib = os.path.join(build_dir, lib)
            if os.path.exists(src_lib):
                dest_lib = os.path.join(pkg_dir, lib)
                shutil.copy(src_lib, dest_lib)
                print(f"Successfully copied {lib} to {pkg_dir}")
            else:
                print(f"Warning: Expected library {lib} not found!")

        # 5. Copy C/CUDA source files into the package for runtime auto-build.
        # This enables Colab to compile CUDA at runtime if nvcc is installed
        # after pip install (common scenario on Colab).
        src_dir = os.path.join(project_root, 'src')
        if os.path.isdir(src_dir):
            pkg_src = os.path.join(pkg_dir, 'src')
            if os.path.exists(pkg_src):
                shutil.rmtree(pkg_src)
            shutil.copytree(src_dir, pkg_src,
                           ignore=shutil.ignore_patterns('*.o', '*.so'))
            print(f"Copied C/CUDA sources to {pkg_src} for runtime build support")

# We define a dummy Extension to ensure setuptools marks the wheel as platform-specific
ext_modules = [
    Extension(name="addernet._dummy", sources=[])
]

cmdclass_dict = {'build_ext': MakeBuildExt}
if bdist_wheel is not None:
    cmdclass_dict['bdist_wheel'] = bdist_wheel

setup(
    version="1.3.8",
    ext_modules=ext_modules,
    cmdclass=cmdclass_dict,
    package_data={'addernet': ['src/*.c', 'src/*.h', 'src/*.cu']}
)
