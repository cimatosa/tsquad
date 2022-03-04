# from . import tanhsinh_py
# from .ts_exceptions import *
#
# try:
#     from .ts_for_py import quad_ts
#     from .ts_for_py import quad_osc_ts
#     from .ts_for_py import quad_fourier_sin_ts
#     from .ts_for_py import quad_fourier_cos_ts
#     from .ts_for_py import quad_fourier_ts
#     kind = "cython"
# except ImportError as e:
#     import traceback
#     traceback.print_exc()
#     import warnings
#     warnings.warn("cython acceleration not available for tanhsinh package")
#     from .tanhsinh_py import quad_ts
#     from .tanhsinh_py import quad_osc_ts
#     kind = "python"

from .tsquad_py import QuadTS
