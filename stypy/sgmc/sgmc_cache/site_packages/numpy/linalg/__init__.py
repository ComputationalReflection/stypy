
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Core Linear Algebra Tools
3: =========================
4: 
5: =============== ==========================================================
6: Linear algebra basics
7: ==========================================================================
8: norm            Vector or matrix norm
9: inv             Inverse of a square matrix
10: solve           Solve a linear system of equations
11: det             Determinant of a square matrix
12: slogdet         Logarithm of the determinant of a square matrix
13: lstsq           Solve linear least-squares problem
14: pinv            Pseudo-inverse (Moore-Penrose) calculated using a singular
15:                 value decomposition
16: matrix_power    Integer power of a square matrix
17: matrix_rank     Calculate matrix rank using an SVD-based method
18: =============== ==========================================================
19: 
20: =============== ==========================================================
21: Eigenvalues and decompositions
22: ==========================================================================
23: eig             Eigenvalues and vectors of a square matrix
24: eigh            Eigenvalues and eigenvectors of a Hermitian matrix
25: eigvals         Eigenvalues of a square matrix
26: eigvalsh        Eigenvalues of a Hermitian matrix
27: qr              QR decomposition of a matrix
28: svd             Singular value decomposition of a matrix
29: cholesky        Cholesky decomposition of a matrix
30: =============== ==========================================================
31: 
32: =============== ==========================================================
33: Tensor operations
34: ==========================================================================
35: tensorsolve     Solve a linear tensor equation
36: tensorinv       Calculate an inverse of a tensor
37: =============== ==========================================================
38: 
39: =============== ==========================================================
40: Exceptions
41: ==========================================================================
42: LinAlgError     Indicates a failed linear algebra operation
43: =============== ==========================================================
44: 
45: '''
46: from __future__ import division, absolute_import, print_function
47: 
48: # To get sub-modules
49: from .info import __doc__
50: 
51: from .linalg import *
52: 
53: from numpy.testing.nosetester import _numpy_tester
54: test = _numpy_tester().test
55: bench = _numpy_tester().bench
56: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_138529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, (-1)), 'str', '\nCore Linear Algebra Tools\n=========================\n\n=============== ==========================================================\nLinear algebra basics\n==========================================================================\nnorm            Vector or matrix norm\ninv             Inverse of a square matrix\nsolve           Solve a linear system of equations\ndet             Determinant of a square matrix\nslogdet         Logarithm of the determinant of a square matrix\nlstsq           Solve linear least-squares problem\npinv            Pseudo-inverse (Moore-Penrose) calculated using a singular\n                value decomposition\nmatrix_power    Integer power of a square matrix\nmatrix_rank     Calculate matrix rank using an SVD-based method\n=============== ==========================================================\n\n=============== ==========================================================\nEigenvalues and decompositions\n==========================================================================\neig             Eigenvalues and vectors of a square matrix\neigh            Eigenvalues and eigenvectors of a Hermitian matrix\neigvals         Eigenvalues of a square matrix\neigvalsh        Eigenvalues of a Hermitian matrix\nqr              QR decomposition of a matrix\nsvd             Singular value decomposition of a matrix\ncholesky        Cholesky decomposition of a matrix\n=============== ==========================================================\n\n=============== ==========================================================\nTensor operations\n==========================================================================\ntensorsolve     Solve a linear tensor equation\ntensorinv       Calculate an inverse of a tensor\n=============== ==========================================================\n\n=============== ==========================================================\nExceptions\n==========================================================================\nLinAlgError     Indicates a failed linear algebra operation\n=============== ==========================================================\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 49, 0))

# 'from numpy.linalg.info import __doc__' statement (line 49)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/linalg/')
import_138530 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 49, 0), 'numpy.linalg.info')

if (type(import_138530) is not StypyTypeError):

    if (import_138530 != 'pyd_module'):
        __import__(import_138530)
        sys_modules_138531 = sys.modules[import_138530]
        import_from_module(stypy.reporting.localization.Localization(__file__, 49, 0), 'numpy.linalg.info', sys_modules_138531.module_type_store, module_type_store, ['__doc__'])
        nest_module(stypy.reporting.localization.Localization(__file__, 49, 0), __file__, sys_modules_138531, sys_modules_138531.module_type_store, module_type_store)
    else:
        from numpy.linalg.info import __doc__

        import_from_module(stypy.reporting.localization.Localization(__file__, 49, 0), 'numpy.linalg.info', None, module_type_store, ['__doc__'], [__doc__])

else:
    # Assigning a type to the variable 'numpy.linalg.info' (line 49)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 0), 'numpy.linalg.info', import_138530)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 51, 0))

# 'from numpy.linalg.linalg import ' statement (line 51)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/linalg/')
import_138532 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 51, 0), 'numpy.linalg.linalg')

if (type(import_138532) is not StypyTypeError):

    if (import_138532 != 'pyd_module'):
        __import__(import_138532)
        sys_modules_138533 = sys.modules[import_138532]
        import_from_module(stypy.reporting.localization.Localization(__file__, 51, 0), 'numpy.linalg.linalg', sys_modules_138533.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 51, 0), __file__, sys_modules_138533, sys_modules_138533.module_type_store, module_type_store)
    else:
        from numpy.linalg.linalg import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 51, 0), 'numpy.linalg.linalg', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'numpy.linalg.linalg' (line 51)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 0), 'numpy.linalg.linalg', import_138532)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 53, 0))

# 'from numpy.testing.nosetester import _numpy_tester' statement (line 53)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/linalg/')
import_138534 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 53, 0), 'numpy.testing.nosetester')

if (type(import_138534) is not StypyTypeError):

    if (import_138534 != 'pyd_module'):
        __import__(import_138534)
        sys_modules_138535 = sys.modules[import_138534]
        import_from_module(stypy.reporting.localization.Localization(__file__, 53, 0), 'numpy.testing.nosetester', sys_modules_138535.module_type_store, module_type_store, ['_numpy_tester'])
        nest_module(stypy.reporting.localization.Localization(__file__, 53, 0), __file__, sys_modules_138535, sys_modules_138535.module_type_store, module_type_store)
    else:
        from numpy.testing.nosetester import _numpy_tester

        import_from_module(stypy.reporting.localization.Localization(__file__, 53, 0), 'numpy.testing.nosetester', None, module_type_store, ['_numpy_tester'], [_numpy_tester])

else:
    # Assigning a type to the variable 'numpy.testing.nosetester' (line 53)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 0), 'numpy.testing.nosetester', import_138534)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/linalg/')


# Assigning a Attribute to a Name (line 54):

# Call to _numpy_tester(...): (line 54)
# Processing the call keyword arguments (line 54)
kwargs_138537 = {}
# Getting the type of '_numpy_tester' (line 54)
_numpy_tester_138536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 7), '_numpy_tester', False)
# Calling _numpy_tester(args, kwargs) (line 54)
_numpy_tester_call_result_138538 = invoke(stypy.reporting.localization.Localization(__file__, 54, 7), _numpy_tester_138536, *[], **kwargs_138537)

# Obtaining the member 'test' of a type (line 54)
test_138539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 7), _numpy_tester_call_result_138538, 'test')
# Assigning a type to the variable 'test' (line 54)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 0), 'test', test_138539)

# Assigning a Attribute to a Name (line 55):

# Call to _numpy_tester(...): (line 55)
# Processing the call keyword arguments (line 55)
kwargs_138541 = {}
# Getting the type of '_numpy_tester' (line 55)
_numpy_tester_138540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), '_numpy_tester', False)
# Calling _numpy_tester(args, kwargs) (line 55)
_numpy_tester_call_result_138542 = invoke(stypy.reporting.localization.Localization(__file__, 55, 8), _numpy_tester_138540, *[], **kwargs_138541)

# Obtaining the member 'bench' of a type (line 55)
bench_138543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 8), _numpy_tester_call_result_138542, 'bench')
# Assigning a type to the variable 'bench' (line 55)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 0), 'bench', bench_138543)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
