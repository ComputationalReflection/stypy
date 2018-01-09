
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: SciPy: A scientific computing package for Python
3: ================================================
4: 
5: Documentation is available in the docstrings and
6: online at https://docs.scipy.org.
7: 
8: Contents
9: --------
10: SciPy imports all the functions from the NumPy namespace, and in
11: addition provides:
12: 
13: Subpackages
14: -----------
15: Using any of these subpackages requires an explicit import.  For example,
16: ``import scipy.cluster``.
17: 
18: ::
19: 
20:  cluster                      --- Vector Quantization / Kmeans
21:  fftpack                      --- Discrete Fourier Transform algorithms
22:  integrate                    --- Integration routines
23:  interpolate                  --- Interpolation Tools
24:  io                           --- Data input and output
25:  linalg                       --- Linear algebra routines
26:  linalg.blas                  --- Wrappers to BLAS library
27:  linalg.lapack                --- Wrappers to LAPACK library
28:  misc                         --- Various utilities that don't have
29:                                   another home.
30:  ndimage                      --- n-dimensional image package
31:  odr                          --- Orthogonal Distance Regression
32:  optimize                     --- Optimization Tools
33:  signal                       --- Signal Processing Tools
34:  sparse                       --- Sparse Matrices
35:  sparse.linalg                --- Sparse Linear Algebra
36:  sparse.linalg.dsolve         --- Linear Solvers
37:  sparse.linalg.dsolve.umfpack --- :Interface to the UMFPACK library:
38:                                   Conjugate Gradient Method (LOBPCG)
39:  sparse.linalg.eigen          --- Sparse Eigenvalue Solvers
40:  sparse.linalg.eigen.lobpcg   --- Locally Optimal Block Preconditioned
41:                                   Conjugate Gradient Method (LOBPCG)
42:  spatial                      --- Spatial data structures and algorithms
43:  special                      --- Special functions
44:  stats                        --- Statistical Functions
45: 
46: Utility tools
47: -------------
48: ::
49: 
50:  test              --- Run scipy unittests
51:  show_config       --- Show scipy build configuration
52:  show_numpy_config --- Show numpy build configuration
53:  __version__       --- Scipy version string
54:  __numpy_version__ --- Numpy version string
55: 
56: '''
57: from __future__ import division, print_function, absolute_import
58: 
59: __all__ = ['test']
60: 
61: from numpy import show_config as show_numpy_config
62: if show_numpy_config is None:
63:     raise ImportError(
64:         "Cannot import scipy when running from numpy source directory.")
65: from numpy import __version__ as __numpy_version__
66: 
67: # Import numpy symbols to scipy name space
68: import numpy as _num
69: linalg = None
70: from numpy import *
71: from numpy.random import rand, randn
72: from numpy.fft import fft, ifft
73: from numpy.lib.scimath import *
74: 
75: # Allow distributors to run custom init code
76: from . import _distributor_init
77: 
78: __all__ += _num.__all__
79: __all__ += ['randn', 'rand', 'fft', 'ifft']
80: 
81: del _num
82: # Remove the linalg imported from numpy so that the scipy.linalg package can be
83: # imported.
84: del linalg
85: __all__.remove('linalg')
86: 
87: # We first need to detect if we're being called as part of the scipy
88: # setup procedure itself in a reliable manner.
89: try:
90:     __SCIPY_SETUP__
91: except NameError:
92:     __SCIPY_SETUP__ = False
93: 
94: 
95: if __SCIPY_SETUP__:
96:     import sys as _sys
97:     _sys.stderr.write('Running from scipy source directory.\n')
98:     del _sys
99: else:
100:     try:
101:         from scipy.__config__ import show as show_config
102:     except ImportError:
103:         msg = '''Error importing scipy: you cannot import scipy while
104:         being in scipy source directory; please exit the scipy source
105:         tree first, and relaunch your python intepreter.'''
106:         raise ImportError(msg)
107: 
108:     from scipy.version import version as __version__
109:     from scipy._lib._version import NumpyVersion as _NumpyVersion
110:     if _NumpyVersion(__numpy_version__) < '1.8.2':
111:         import warnings
112:         warnings.warn("Numpy 1.8.2 or above is recommended for this version of "
113:                       "scipy (detected version %s)" % __numpy_version__,
114:                       UserWarning)
115: 
116:     del _NumpyVersion
117: 
118:     from scipy._lib._ccallback import LowLevelCallable
119: 
120:     from scipy._lib._testutils import PytestTester
121:     test = PytestTester(__name__)
122:     del PytestTester
123: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, (-1)), 'str', "\nSciPy: A scientific computing package for Python\n================================================\n\nDocumentation is available in the docstrings and\nonline at https://docs.scipy.org.\n\nContents\n--------\nSciPy imports all the functions from the NumPy namespace, and in\naddition provides:\n\nSubpackages\n-----------\nUsing any of these subpackages requires an explicit import.  For example,\n``import scipy.cluster``.\n\n::\n\n cluster                      --- Vector Quantization / Kmeans\n fftpack                      --- Discrete Fourier Transform algorithms\n integrate                    --- Integration routines\n interpolate                  --- Interpolation Tools\n io                           --- Data input and output\n linalg                       --- Linear algebra routines\n linalg.blas                  --- Wrappers to BLAS library\n linalg.lapack                --- Wrappers to LAPACK library\n misc                         --- Various utilities that don't have\n                                  another home.\n ndimage                      --- n-dimensional image package\n odr                          --- Orthogonal Distance Regression\n optimize                     --- Optimization Tools\n signal                       --- Signal Processing Tools\n sparse                       --- Sparse Matrices\n sparse.linalg                --- Sparse Linear Algebra\n sparse.linalg.dsolve         --- Linear Solvers\n sparse.linalg.dsolve.umfpack --- :Interface to the UMFPACK library:\n                                  Conjugate Gradient Method (LOBPCG)\n sparse.linalg.eigen          --- Sparse Eigenvalue Solvers\n sparse.linalg.eigen.lobpcg   --- Locally Optimal Block Preconditioned\n                                  Conjugate Gradient Method (LOBPCG)\n spatial                      --- Spatial data structures and algorithms\n special                      --- Special functions\n stats                        --- Statistical Functions\n\nUtility tools\n-------------\n::\n\n test              --- Run scipy unittests\n show_config       --- Show scipy build configuration\n show_numpy_config --- Show numpy build configuration\n __version__       --- Scipy version string\n __numpy_version__ --- Numpy version string\n\n")

# Assigning a List to a Name (line 59):
__all__ = ['test']
module_type_store.set_exportable_members(['test'])

# Obtaining an instance of the builtin type 'list' (line 59)
list_381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 59)
# Adding element type (line 59)
str_382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 11), 'str', 'test')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 10), list_381, str_382)

# Assigning a type to the variable '__all__' (line 59)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 0), '__all__', list_381)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 61, 0))

# 'from numpy import show_numpy_config' statement (line 61)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/')
import_383 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 61, 0), 'numpy')

if (type(import_383) is not StypyTypeError):

    if (import_383 != 'pyd_module'):
        __import__(import_383)
        sys_modules_384 = sys.modules[import_383]
        import_from_module(stypy.reporting.localization.Localization(__file__, 61, 0), 'numpy', sys_modules_384.module_type_store, module_type_store, ['show_config'])
        nest_module(stypy.reporting.localization.Localization(__file__, 61, 0), __file__, sys_modules_384, sys_modules_384.module_type_store, module_type_store)
    else:
        from numpy import show_config as show_numpy_config

        import_from_module(stypy.reporting.localization.Localization(__file__, 61, 0), 'numpy', None, module_type_store, ['show_config'], [show_numpy_config])

else:
    # Assigning a type to the variable 'numpy' (line 61)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 0), 'numpy', import_383)

# Adding an alias
module_type_store.add_alias('show_numpy_config', 'show_config')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/')


# Type idiom detected: calculating its left and rigth part (line 62)
# Getting the type of 'show_numpy_config' (line 62)
show_numpy_config_385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 3), 'show_numpy_config')
# Getting the type of 'None' (line 62)
None_386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 24), 'None')

(may_be_387, more_types_in_union_388) = may_be_none(show_numpy_config_385, None_386)

if may_be_387:

    if more_types_in_union_388:
        # Runtime conditional SSA (line 62)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
    else:
        module_type_store = module_type_store

    
    # Call to ImportError(...): (line 63)
    # Processing the call arguments (line 63)
    str_390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 8), 'str', 'Cannot import scipy when running from numpy source directory.')
    # Processing the call keyword arguments (line 63)
    kwargs_391 = {}
    # Getting the type of 'ImportError' (line 63)
    ImportError_389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 10), 'ImportError', False)
    # Calling ImportError(args, kwargs) (line 63)
    ImportError_call_result_392 = invoke(stypy.reporting.localization.Localization(__file__, 63, 10), ImportError_389, *[str_390], **kwargs_391)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 63, 4), ImportError_call_result_392, 'raise parameter', BaseException)

    if more_types_in_union_388:
        # SSA join for if statement (line 62)
        module_type_store = module_type_store.join_ssa_context()



stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 65, 0))

# 'from numpy import __numpy_version__' statement (line 65)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/')
import_393 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 65, 0), 'numpy')

if (type(import_393) is not StypyTypeError):

    if (import_393 != 'pyd_module'):
        __import__(import_393)
        sys_modules_394 = sys.modules[import_393]
        import_from_module(stypy.reporting.localization.Localization(__file__, 65, 0), 'numpy', sys_modules_394.module_type_store, module_type_store, ['__version__'])
        nest_module(stypy.reporting.localization.Localization(__file__, 65, 0), __file__, sys_modules_394, sys_modules_394.module_type_store, module_type_store)
    else:
        from numpy import __version__ as __numpy_version__

        import_from_module(stypy.reporting.localization.Localization(__file__, 65, 0), 'numpy', None, module_type_store, ['__version__'], [__numpy_version__])

else:
    # Assigning a type to the variable 'numpy' (line 65)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 0), 'numpy', import_393)

# Adding an alias
module_type_store.add_alias('__numpy_version__', '__version__')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 68, 0))

# 'import numpy' statement (line 68)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/')
import_395 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 68, 0), 'numpy')

if (type(import_395) is not StypyTypeError):

    if (import_395 != 'pyd_module'):
        __import__(import_395)
        sys_modules_396 = sys.modules[import_395]
        import_module(stypy.reporting.localization.Localization(__file__, 68, 0), '_num', sys_modules_396.module_type_store, module_type_store)
    else:
        import numpy as _num

        import_module(stypy.reporting.localization.Localization(__file__, 68, 0), '_num', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 0), 'numpy', import_395)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/')


# Assigning a Name to a Name (line 69):
# Getting the type of 'None' (line 69)
None_397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 9), 'None')
# Assigning a type to the variable 'linalg' (line 69)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 0), 'linalg', None_397)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 70, 0))

# 'from numpy import ' statement (line 70)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/')
import_398 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 70, 0), 'numpy')

if (type(import_398) is not StypyTypeError):

    if (import_398 != 'pyd_module'):
        __import__(import_398)
        sys_modules_399 = sys.modules[import_398]
        import_from_module(stypy.reporting.localization.Localization(__file__, 70, 0), 'numpy', sys_modules_399.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 70, 0), __file__, sys_modules_399, sys_modules_399.module_type_store, module_type_store)
    else:
        from numpy import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 70, 0), 'numpy', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'numpy' (line 70)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 0), 'numpy', import_398)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 71, 0))

# 'from numpy.random import rand, randn' statement (line 71)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/')
import_400 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 71, 0), 'numpy.random')

if (type(import_400) is not StypyTypeError):

    if (import_400 != 'pyd_module'):
        __import__(import_400)
        sys_modules_401 = sys.modules[import_400]
        import_from_module(stypy.reporting.localization.Localization(__file__, 71, 0), 'numpy.random', sys_modules_401.module_type_store, module_type_store, ['rand', 'randn'])
        nest_module(stypy.reporting.localization.Localization(__file__, 71, 0), __file__, sys_modules_401, sys_modules_401.module_type_store, module_type_store)
    else:
        from numpy.random import rand, randn

        import_from_module(stypy.reporting.localization.Localization(__file__, 71, 0), 'numpy.random', None, module_type_store, ['rand', 'randn'], [rand, randn])

else:
    # Assigning a type to the variable 'numpy.random' (line 71)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 0), 'numpy.random', import_400)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 72, 0))

# 'from numpy.fft import fft, ifft' statement (line 72)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/')
import_402 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 72, 0), 'numpy.fft')

if (type(import_402) is not StypyTypeError):

    if (import_402 != 'pyd_module'):
        __import__(import_402)
        sys_modules_403 = sys.modules[import_402]
        import_from_module(stypy.reporting.localization.Localization(__file__, 72, 0), 'numpy.fft', sys_modules_403.module_type_store, module_type_store, ['fft', 'ifft'])
        nest_module(stypy.reporting.localization.Localization(__file__, 72, 0), __file__, sys_modules_403, sys_modules_403.module_type_store, module_type_store)
    else:
        from numpy.fft import fft, ifft

        import_from_module(stypy.reporting.localization.Localization(__file__, 72, 0), 'numpy.fft', None, module_type_store, ['fft', 'ifft'], [fft, ifft])

else:
    # Assigning a type to the variable 'numpy.fft' (line 72)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 0), 'numpy.fft', import_402)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 73, 0))

# 'from numpy.lib.scimath import ' statement (line 73)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/')
import_404 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 73, 0), 'numpy.lib.scimath')

if (type(import_404) is not StypyTypeError):

    if (import_404 != 'pyd_module'):
        __import__(import_404)
        sys_modules_405 = sys.modules[import_404]
        import_from_module(stypy.reporting.localization.Localization(__file__, 73, 0), 'numpy.lib.scimath', sys_modules_405.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 73, 0), __file__, sys_modules_405, sys_modules_405.module_type_store, module_type_store)
    else:
        from numpy.lib.scimath import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 73, 0), 'numpy.lib.scimath', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'numpy.lib.scimath' (line 73)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 0), 'numpy.lib.scimath', import_404)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 76, 0))

# 'from scipy import _distributor_init' statement (line 76)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/')
import_406 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 76, 0), 'scipy')

if (type(import_406) is not StypyTypeError):

    if (import_406 != 'pyd_module'):
        __import__(import_406)
        sys_modules_407 = sys.modules[import_406]
        import_from_module(stypy.reporting.localization.Localization(__file__, 76, 0), 'scipy', sys_modules_407.module_type_store, module_type_store, ['_distributor_init'])
        nest_module(stypy.reporting.localization.Localization(__file__, 76, 0), __file__, sys_modules_407, sys_modules_407.module_type_store, module_type_store)
    else:
        from scipy import _distributor_init

        import_from_module(stypy.reporting.localization.Localization(__file__, 76, 0), 'scipy', None, module_type_store, ['_distributor_init'], [_distributor_init])

else:
    # Assigning a type to the variable 'scipy' (line 76)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 0), 'scipy', import_406)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/')


# Getting the type of '__all__' (line 78)
all___408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 0), '__all__')
# Getting the type of '_num' (line 78)
_num_409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 11), '_num')
# Obtaining the member '__all__' of a type (line 78)
all___410 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 11), _num_409, '__all__')
# Applying the binary operator '+=' (line 78)
result_iadd_411 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 0), '+=', all___408, all___410)
# Assigning a type to the variable '__all__' (line 78)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 0), '__all__', result_iadd_411)


# Getting the type of '__all__' (line 79)
all___412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 0), '__all__')

# Obtaining an instance of the builtin type 'list' (line 79)
list_413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 11), 'list')
# Adding type elements to the builtin type 'list' instance (line 79)
# Adding element type (line 79)
str_414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 12), 'str', 'randn')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 11), list_413, str_414)
# Adding element type (line 79)
str_415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 21), 'str', 'rand')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 11), list_413, str_415)
# Adding element type (line 79)
str_416 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 29), 'str', 'fft')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 11), list_413, str_416)
# Adding element type (line 79)
str_417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 36), 'str', 'ifft')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 11), list_413, str_417)

# Applying the binary operator '+=' (line 79)
result_iadd_418 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 0), '+=', all___412, list_413)
# Assigning a type to the variable '__all__' (line 79)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 0), '__all__', result_iadd_418)

# Deleting a member
module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 81, 0), module_type_store, '_num')
# Deleting a member
module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 84, 0), module_type_store, 'linalg')

# Call to remove(...): (line 85)
# Processing the call arguments (line 85)
str_421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 15), 'str', 'linalg')
# Processing the call keyword arguments (line 85)
kwargs_422 = {}
# Getting the type of '__all__' (line 85)
all___419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 0), '__all__', False)
# Obtaining the member 'remove' of a type (line 85)
remove_420 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 0), all___419, 'remove')
# Calling remove(args, kwargs) (line 85)
remove_call_result_423 = invoke(stypy.reporting.localization.Localization(__file__, 85, 0), remove_420, *[str_421], **kwargs_422)



# SSA begins for try-except statement (line 89)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
# Getting the type of '__SCIPY_SETUP__' (line 90)
SCIPY_SETUP___424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 4), '__SCIPY_SETUP__')
# SSA branch for the except part of a try statement (line 89)
# SSA branch for the except 'NameError' branch of a try statement (line 89)
module_type_store.open_ssa_branch('except')

# Assigning a Name to a Name (line 92):
# Getting the type of 'False' (line 92)
False_425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 22), 'False')
# Assigning a type to the variable '__SCIPY_SETUP__' (line 92)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), '__SCIPY_SETUP__', False_425)
# SSA join for try-except statement (line 89)
module_type_store = module_type_store.join_ssa_context()


# Getting the type of '__SCIPY_SETUP__' (line 95)
SCIPY_SETUP___426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 3), '__SCIPY_SETUP__')
# Testing the type of an if condition (line 95)
if_condition_427 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 95, 0), SCIPY_SETUP___426)
# Assigning a type to the variable 'if_condition_427' (line 95)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 0), 'if_condition_427', if_condition_427)
# SSA begins for if statement (line 95)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 96, 4))

# 'import sys' statement (line 96)
import sys as _sys

import_module(stypy.reporting.localization.Localization(__file__, 96, 4), '_sys', _sys, module_type_store)


# Call to write(...): (line 97)
# Processing the call arguments (line 97)
str_431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 22), 'str', 'Running from scipy source directory.\n')
# Processing the call keyword arguments (line 97)
kwargs_432 = {}
# Getting the type of '_sys' (line 97)
_sys_428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 4), '_sys', False)
# Obtaining the member 'stderr' of a type (line 97)
stderr_429 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 4), _sys_428, 'stderr')
# Obtaining the member 'write' of a type (line 97)
write_430 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 4), stderr_429, 'write')
# Calling write(args, kwargs) (line 97)
write_call_result_433 = invoke(stypy.reporting.localization.Localization(__file__, 97, 4), write_430, *[str_431], **kwargs_432)

# Deleting a member
module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 98, 4), module_type_store, '_sys')
# SSA branch for the else part of an if statement (line 95)
module_type_store.open_ssa_branch('else')


# SSA begins for try-except statement (line 100)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 101, 8))

# 'from scipy.__config__ import show_config' statement (line 101)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/')
import_434 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 101, 8), 'scipy.__config__')

if (type(import_434) is not StypyTypeError):

    if (import_434 != 'pyd_module'):
        __import__(import_434)
        sys_modules_435 = sys.modules[import_434]
        import_from_module(stypy.reporting.localization.Localization(__file__, 101, 8), 'scipy.__config__', sys_modules_435.module_type_store, module_type_store, ['show'])
        nest_module(stypy.reporting.localization.Localization(__file__, 101, 8), __file__, sys_modules_435, sys_modules_435.module_type_store, module_type_store)
    else:
        from scipy.__config__ import show as show_config

        import_from_module(stypy.reporting.localization.Localization(__file__, 101, 8), 'scipy.__config__', None, module_type_store, ['show'], [show_config])

else:
    # Assigning a type to the variable 'scipy.__config__' (line 101)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'scipy.__config__', import_434)

# Adding an alias
module_type_store.add_alias('show_config', 'show')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/')

# SSA branch for the except part of a try statement (line 100)
# SSA branch for the except 'ImportError' branch of a try statement (line 100)
module_type_store.open_ssa_branch('except')

# Assigning a Str to a Name (line 103):
str_436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, (-1)), 'str', 'Error importing scipy: you cannot import scipy while\n        being in scipy source directory; please exit the scipy source\n        tree first, and relaunch your python intepreter.')
# Assigning a type to the variable 'msg' (line 103)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'msg', str_436)

# Call to ImportError(...): (line 106)
# Processing the call arguments (line 106)
# Getting the type of 'msg' (line 106)
msg_438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 26), 'msg', False)
# Processing the call keyword arguments (line 106)
kwargs_439 = {}
# Getting the type of 'ImportError' (line 106)
ImportError_437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 14), 'ImportError', False)
# Calling ImportError(args, kwargs) (line 106)
ImportError_call_result_440 = invoke(stypy.reporting.localization.Localization(__file__, 106, 14), ImportError_437, *[msg_438], **kwargs_439)

ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 106, 8), ImportError_call_result_440, 'raise parameter', BaseException)
# SSA join for try-except statement (line 100)
module_type_store = module_type_store.join_ssa_context()

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 108, 4))

# 'from scipy.version import __version__' statement (line 108)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/')
import_441 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 108, 4), 'scipy.version')

if (type(import_441) is not StypyTypeError):

    if (import_441 != 'pyd_module'):
        __import__(import_441)
        sys_modules_442 = sys.modules[import_441]
        import_from_module(stypy.reporting.localization.Localization(__file__, 108, 4), 'scipy.version', sys_modules_442.module_type_store, module_type_store, ['version'])
        nest_module(stypy.reporting.localization.Localization(__file__, 108, 4), __file__, sys_modules_442, sys_modules_442.module_type_store, module_type_store)
    else:
        from scipy.version import version as __version__

        import_from_module(stypy.reporting.localization.Localization(__file__, 108, 4), 'scipy.version', None, module_type_store, ['version'], [__version__])

else:
    # Assigning a type to the variable 'scipy.version' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'scipy.version', import_441)

# Adding an alias
module_type_store.add_alias('__version__', 'version')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 109, 4))

# 'from scipy._lib._version import _NumpyVersion' statement (line 109)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/')
import_443 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 109, 4), 'scipy._lib._version')

if (type(import_443) is not StypyTypeError):

    if (import_443 != 'pyd_module'):
        __import__(import_443)
        sys_modules_444 = sys.modules[import_443]
        import_from_module(stypy.reporting.localization.Localization(__file__, 109, 4), 'scipy._lib._version', sys_modules_444.module_type_store, module_type_store, ['NumpyVersion'])
        nest_module(stypy.reporting.localization.Localization(__file__, 109, 4), __file__, sys_modules_444, sys_modules_444.module_type_store, module_type_store)
    else:
        from scipy._lib._version import NumpyVersion as _NumpyVersion

        import_from_module(stypy.reporting.localization.Localization(__file__, 109, 4), 'scipy._lib._version', None, module_type_store, ['NumpyVersion'], [_NumpyVersion])

else:
    # Assigning a type to the variable 'scipy._lib._version' (line 109)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 4), 'scipy._lib._version', import_443)

# Adding an alias
module_type_store.add_alias('_NumpyVersion', 'NumpyVersion')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/')




# Call to _NumpyVersion(...): (line 110)
# Processing the call arguments (line 110)
# Getting the type of '__numpy_version__' (line 110)
numpy_version___446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 21), '__numpy_version__', False)
# Processing the call keyword arguments (line 110)
kwargs_447 = {}
# Getting the type of '_NumpyVersion' (line 110)
_NumpyVersion_445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 7), '_NumpyVersion', False)
# Calling _NumpyVersion(args, kwargs) (line 110)
_NumpyVersion_call_result_448 = invoke(stypy.reporting.localization.Localization(__file__, 110, 7), _NumpyVersion_445, *[numpy_version___446], **kwargs_447)

str_449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 42), 'str', '1.8.2')
# Applying the binary operator '<' (line 110)
result_lt_450 = python_operator(stypy.reporting.localization.Localization(__file__, 110, 7), '<', _NumpyVersion_call_result_448, str_449)

# Testing the type of an if condition (line 110)
if_condition_451 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 110, 4), result_lt_450)
# Assigning a type to the variable 'if_condition_451' (line 110)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 4), 'if_condition_451', if_condition_451)
# SSA begins for if statement (line 110)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 111, 8))

# 'import warnings' statement (line 111)
import warnings

import_module(stypy.reporting.localization.Localization(__file__, 111, 8), 'warnings', warnings, module_type_store)


# Call to warn(...): (line 112)
# Processing the call arguments (line 112)
str_454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 22), 'str', 'Numpy 1.8.2 or above is recommended for this version of scipy (detected version %s)')
# Getting the type of '__numpy_version__' (line 113)
numpy_version___455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 54), '__numpy_version__', False)
# Applying the binary operator '%' (line 112)
result_mod_456 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 22), '%', str_454, numpy_version___455)

# Getting the type of 'UserWarning' (line 114)
UserWarning_457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 22), 'UserWarning', False)
# Processing the call keyword arguments (line 112)
kwargs_458 = {}
# Getting the type of 'warnings' (line 112)
warnings_452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'warnings', False)
# Obtaining the member 'warn' of a type (line 112)
warn_453 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 8), warnings_452, 'warn')
# Calling warn(args, kwargs) (line 112)
warn_call_result_459 = invoke(stypy.reporting.localization.Localization(__file__, 112, 8), warn_453, *[result_mod_456, UserWarning_457], **kwargs_458)

# SSA join for if statement (line 110)
module_type_store = module_type_store.join_ssa_context()

# Deleting a member
module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 116, 4), module_type_store, '_NumpyVersion')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 118, 4))

# 'from scipy._lib._ccallback import LowLevelCallable' statement (line 118)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/')
import_460 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 118, 4), 'scipy._lib._ccallback')

if (type(import_460) is not StypyTypeError):

    if (import_460 != 'pyd_module'):
        __import__(import_460)
        sys_modules_461 = sys.modules[import_460]
        import_from_module(stypy.reporting.localization.Localization(__file__, 118, 4), 'scipy._lib._ccallback', sys_modules_461.module_type_store, module_type_store, ['LowLevelCallable'])
        nest_module(stypy.reporting.localization.Localization(__file__, 118, 4), __file__, sys_modules_461, sys_modules_461.module_type_store, module_type_store)
    else:
        from scipy._lib._ccallback import LowLevelCallable

        import_from_module(stypy.reporting.localization.Localization(__file__, 118, 4), 'scipy._lib._ccallback', None, module_type_store, ['LowLevelCallable'], [LowLevelCallable])

else:
    # Assigning a type to the variable 'scipy._lib._ccallback' (line 118)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 4), 'scipy._lib._ccallback', import_460)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 120, 4))

# 'from scipy._lib._testutils import PytestTester' statement (line 120)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/')
import_462 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 120, 4), 'scipy._lib._testutils')

if (type(import_462) is not StypyTypeError):

    if (import_462 != 'pyd_module'):
        __import__(import_462)
        sys_modules_463 = sys.modules[import_462]
        import_from_module(stypy.reporting.localization.Localization(__file__, 120, 4), 'scipy._lib._testutils', sys_modules_463.module_type_store, module_type_store, ['PytestTester'])
        nest_module(stypy.reporting.localization.Localization(__file__, 120, 4), __file__, sys_modules_463, sys_modules_463.module_type_store, module_type_store)
    else:
        from scipy._lib._testutils import PytestTester

        import_from_module(stypy.reporting.localization.Localization(__file__, 120, 4), 'scipy._lib._testutils', None, module_type_store, ['PytestTester'], [PytestTester])

else:
    # Assigning a type to the variable 'scipy._lib._testutils' (line 120)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 4), 'scipy._lib._testutils', import_462)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/')


# Assigning a Call to a Name (line 121):

# Call to PytestTester(...): (line 121)
# Processing the call arguments (line 121)
# Getting the type of '__name__' (line 121)
name___465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 24), '__name__', False)
# Processing the call keyword arguments (line 121)
kwargs_466 = {}
# Getting the type of 'PytestTester' (line 121)
PytestTester_464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 11), 'PytestTester', False)
# Calling PytestTester(args, kwargs) (line 121)
PytestTester_call_result_467 = invoke(stypy.reporting.localization.Localization(__file__, 121, 11), PytestTester_464, *[name___465], **kwargs_466)

# Assigning a type to the variable 'test' (line 121)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 4), 'test', PytestTester_call_result_467)
# Deleting a member
module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 122, 4), module_type_store, 'PytestTester')
# SSA join for if statement (line 95)
module_type_store = module_type_store.join_ssa_context()


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
