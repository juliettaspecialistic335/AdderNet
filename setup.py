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
            print(f"Error compiling C/CUDA extensions: {e}", file=sys.stderr)
            sys.exit(1)
            
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
        for lib in so_files_to_include:
            src_lib = os.path.join(build_dir, lib)
            if os.path.exists(src_lib):
                dest_lib = os.path.join(pkg_dir, lib)
                shutil.copy(src_lib, dest_lib)
                print(f"Successfully copied {lib} to {pkg_dir}")
            else:
                print(f"Warning: Expected library {lib} not found!")

# We define a dummy Extension to ensure setuptools marks the wheel as platform-specific
ext_modules = [
    Extension(name="addernet._dummy", sources=[])
]

cmdclass_dict = {'build_ext': MakeBuildExt}
if bdist_wheel is not None:
    cmdclass_dict['bdist_wheel'] = bdist_wheel

setup(
    version="1.2.0",
    ext_modules=ext_modules,
    cmdclass=cmdclass_dict,
    package_data={'addernet': ['src/*.c', 'src/*.h']}
)
