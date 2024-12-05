import os
import sys
from copy import deepcopy
from setuptools import setup

import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

if torch.cuda.is_available():   
    # Some containers set this to contain old architectures that won't compile. We only need the one installed in the machine.
    os.environ["TORCH_CUDA_ARCH_LIST"] = ""

    common_library_dirs = []
    if '--fix-lcuda' in sys.argv:
        sys.argv.remove('--fix-lcuda')
        common_library_dirs.append(os.path.join(os.environ.get('CUDA_HOME'), 'lib64', 'stubs'))


    major, minor = torch.cuda.get_device_capability()
    compute_capability = major * 10 + minor

def get_ext():
    nvcc_flags = [
        "-std=c++14",
        "--extended-lambda",
        "--expt-relaxed-constexpr",
        # The following definitions must be undefined
        # since half-precision operation is required.
        '-U__CUDA_NO_HALF_OPERATORS__', 
        '-U__CUDA_NO_HALF2_OPERATORS__',
        '-U__CUDA_NO_HALF_CONVERSIONS__', 
        '-U__CUDA_NO_BFLOAT16_CONVERSIONS__',
        f"-gencode=arch=compute_{compute_capability},code=compute_{compute_capability}",
        f"-gencode=arch=compute_{compute_capability},code=sm_{compute_capability}",
    ]
    if os.name == "posix":
        c_flags = ["-std=c++14"]
        nvcc_flags += [
            "-Xcompiler=-mf16c",
            "-Xcompiler=-Wno-float-conversion",
            "-Xcompiler=-fno-strict-aliasing",
			"-Xcudafe=--diag_suppress=unrecognized_gcc_pragma",
        ]
    elif os.name == "nt":
        c_flags = ["/std:c++14"]

    print(f"Targeting compute capability {compute_capability}")

    definitions = [
    ]
    nvcc_flags += definitions
    c_flags += definitions

    # List of sources.
    source_files = [
        os.path.join(SCRIPT_DIR, "src/lotd_impl_2d.cu"),
        os.path.join(SCRIPT_DIR, "src/lotd_impl_3d.cu"),
        os.path.join(SCRIPT_DIR, "src/lotd_impl_4d.cu"),
        os.path.join(SCRIPT_DIR, "src/lotd_torch_api.cu"),
        os.path.join(SCRIPT_DIR, "src/lotd.cpp"),
    ]

    libraries = []
    library_dirs = deepcopy(common_library_dirs)
    extra_objects = []

    ext = CUDAExtension(
        name="B_spline_hash_encoder",
        sources=source_files,
        include_dirs=[
                      os.path.join(SCRIPT_DIR, "include")
                ],
        extra_compile_args={"cxx": c_flags, "nvcc": nvcc_flags},
        libraries=libraries,
        library_dirs=library_dirs,
        extra_objects=extra_objects
    )
    return ext


def get_extensions():
    ext_modules = []
    ext_modules.append(get_ext())   
    return ext_modules

setup(
    name="B_spline_hash_encoder",
    ext_modules=get_extensions(),
    cmdclass={"build_ext": BuildExtension}
)
