
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: ==================================================
3: Sparse linear algebra (:mod:`scipy.sparse.linalg`)
4: ==================================================
5: 
6: .. currentmodule:: scipy.sparse.linalg
7: 
8: Abstract linear operators
9: -------------------------
10: 
11: .. autosummary::
12:    :toctree: generated/
13: 
14:    LinearOperator -- abstract representation of a linear operator
15:    aslinearoperator -- convert an object to an abstract linear operator
16: 
17: Matrix Operations
18: -----------------
19: 
20: .. autosummary::
21:    :toctree: generated/
22: 
23:    inv -- compute the sparse matrix inverse
24:    expm -- compute the sparse matrix exponential
25:    expm_multiply -- compute the product of a matrix exponential and a matrix
26: 
27: Matrix norms
28: ------------
29: 
30: .. autosummary::
31:    :toctree: generated/
32: 
33:    norm -- Norm of a sparse matrix
34:    onenormest -- Estimate the 1-norm of a sparse matrix
35: 
36: Solving linear problems
37: -----------------------
38: 
39: Direct methods for linear equation systems:
40: 
41: .. autosummary::
42:    :toctree: generated/
43: 
44:    spsolve -- Solve the sparse linear system Ax=b
45:    spsolve_triangular -- Solve the sparse linear system Ax=b for a triangular matrix
46:    factorized -- Pre-factorize matrix to a function solving a linear system
47:    MatrixRankWarning -- Warning on exactly singular matrices
48:    use_solver -- Select direct solver to use
49: 
50: Iterative methods for linear equation systems:
51: 
52: .. autosummary::
53:    :toctree: generated/
54: 
55:    bicg -- Use BIConjugate Gradient iteration to solve A x = b
56:    bicgstab -- Use BIConjugate Gradient STABilized iteration to solve A x = b
57:    cg -- Use Conjugate Gradient iteration to solve A x = b
58:    cgs -- Use Conjugate Gradient Squared iteration to solve A x = b
59:    gmres -- Use Generalized Minimal RESidual iteration to solve A x = b
60:    lgmres -- Solve a matrix equation using the LGMRES algorithm
61:    minres -- Use MINimum RESidual iteration to solve Ax = b
62:    qmr -- Use Quasi-Minimal Residual iteration to solve A x = b
63:    gcrotmk -- Solve a matrix equation using the GCROT(m,k) algorithm
64: 
65: Iterative methods for least-squares problems:
66: 
67: .. autosummary::
68:    :toctree: generated/
69: 
70:    lsqr -- Find the least-squares solution to a sparse linear equation system
71:    lsmr -- Find the least-squares solution to a sparse linear equation system
72: 
73: Matrix factorizations
74: ---------------------
75: 
76: Eigenvalue problems:
77: 
78: .. autosummary::
79:    :toctree: generated/
80: 
81:    eigs -- Find k eigenvalues and eigenvectors of the square matrix A
82:    eigsh -- Find k eigenvalues and eigenvectors of a symmetric matrix
83:    lobpcg -- Solve symmetric partial eigenproblems with optional preconditioning
84: 
85: Singular values problems:
86: 
87: .. autosummary::
88:    :toctree: generated/
89: 
90:    svds -- Compute k singular values/vectors for a sparse matrix
91: 
92: Complete or incomplete LU factorizations
93: 
94: .. autosummary::
95:    :toctree: generated/
96: 
97:    splu -- Compute a LU decomposition for a sparse matrix
98:    spilu -- Compute an incomplete LU decomposition for a sparse matrix
99:    SuperLU -- Object representing an LU factorization
100: 
101: Exceptions
102: ----------
103: 
104: .. autosummary::
105:    :toctree: generated/
106: 
107:    ArpackNoConvergence
108:    ArpackError
109: 
110: '''
111: 
112: from __future__ import division, print_function, absolute_import
113: 
114: from .isolve import *
115: from .dsolve import *
116: from .interface import *
117: from .eigen import *
118: from .matfuncs import *
119: from ._onenormest import *
120: from ._norm import *
121: from ._expm_multiply import *
122: 
123: __all__ = [s for s in dir() if not s.startswith('_')]
124: 
125: from scipy._lib._testutils import PytestTester
126: test = PytestTester(__name__)
127: del PytestTester
128: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_391623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, (-1)), 'str', '\n==================================================\nSparse linear algebra (:mod:`scipy.sparse.linalg`)\n==================================================\n\n.. currentmodule:: scipy.sparse.linalg\n\nAbstract linear operators\n-------------------------\n\n.. autosummary::\n   :toctree: generated/\n\n   LinearOperator -- abstract representation of a linear operator\n   aslinearoperator -- convert an object to an abstract linear operator\n\nMatrix Operations\n-----------------\n\n.. autosummary::\n   :toctree: generated/\n\n   inv -- compute the sparse matrix inverse\n   expm -- compute the sparse matrix exponential\n   expm_multiply -- compute the product of a matrix exponential and a matrix\n\nMatrix norms\n------------\n\n.. autosummary::\n   :toctree: generated/\n\n   norm -- Norm of a sparse matrix\n   onenormest -- Estimate the 1-norm of a sparse matrix\n\nSolving linear problems\n-----------------------\n\nDirect methods for linear equation systems:\n\n.. autosummary::\n   :toctree: generated/\n\n   spsolve -- Solve the sparse linear system Ax=b\n   spsolve_triangular -- Solve the sparse linear system Ax=b for a triangular matrix\n   factorized -- Pre-factorize matrix to a function solving a linear system\n   MatrixRankWarning -- Warning on exactly singular matrices\n   use_solver -- Select direct solver to use\n\nIterative methods for linear equation systems:\n\n.. autosummary::\n   :toctree: generated/\n\n   bicg -- Use BIConjugate Gradient iteration to solve A x = b\n   bicgstab -- Use BIConjugate Gradient STABilized iteration to solve A x = b\n   cg -- Use Conjugate Gradient iteration to solve A x = b\n   cgs -- Use Conjugate Gradient Squared iteration to solve A x = b\n   gmres -- Use Generalized Minimal RESidual iteration to solve A x = b\n   lgmres -- Solve a matrix equation using the LGMRES algorithm\n   minres -- Use MINimum RESidual iteration to solve Ax = b\n   qmr -- Use Quasi-Minimal Residual iteration to solve A x = b\n   gcrotmk -- Solve a matrix equation using the GCROT(m,k) algorithm\n\nIterative methods for least-squares problems:\n\n.. autosummary::\n   :toctree: generated/\n\n   lsqr -- Find the least-squares solution to a sparse linear equation system\n   lsmr -- Find the least-squares solution to a sparse linear equation system\n\nMatrix factorizations\n---------------------\n\nEigenvalue problems:\n\n.. autosummary::\n   :toctree: generated/\n\n   eigs -- Find k eigenvalues and eigenvectors of the square matrix A\n   eigsh -- Find k eigenvalues and eigenvectors of a symmetric matrix\n   lobpcg -- Solve symmetric partial eigenproblems with optional preconditioning\n\nSingular values problems:\n\n.. autosummary::\n   :toctree: generated/\n\n   svds -- Compute k singular values/vectors for a sparse matrix\n\nComplete or incomplete LU factorizations\n\n.. autosummary::\n   :toctree: generated/\n\n   splu -- Compute a LU decomposition for a sparse matrix\n   spilu -- Compute an incomplete LU decomposition for a sparse matrix\n   SuperLU -- Object representing an LU factorization\n\nExceptions\n----------\n\n.. autosummary::\n   :toctree: generated/\n\n   ArpackNoConvergence\n   ArpackError\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 114, 0))

# 'from scipy.sparse.linalg.isolve import ' statement (line 114)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/')
import_391624 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 114, 0), 'scipy.sparse.linalg.isolve')

if (type(import_391624) is not StypyTypeError):

    if (import_391624 != 'pyd_module'):
        __import__(import_391624)
        sys_modules_391625 = sys.modules[import_391624]
        import_from_module(stypy.reporting.localization.Localization(__file__, 114, 0), 'scipy.sparse.linalg.isolve', sys_modules_391625.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 114, 0), __file__, sys_modules_391625, sys_modules_391625.module_type_store, module_type_store)
    else:
        from scipy.sparse.linalg.isolve import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 114, 0), 'scipy.sparse.linalg.isolve', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.sparse.linalg.isolve' (line 114)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 0), 'scipy.sparse.linalg.isolve', import_391624)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 115, 0))

# 'from scipy.sparse.linalg.dsolve import ' statement (line 115)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/')
import_391626 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 115, 0), 'scipy.sparse.linalg.dsolve')

if (type(import_391626) is not StypyTypeError):

    if (import_391626 != 'pyd_module'):
        __import__(import_391626)
        sys_modules_391627 = sys.modules[import_391626]
        import_from_module(stypy.reporting.localization.Localization(__file__, 115, 0), 'scipy.sparse.linalg.dsolve', sys_modules_391627.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 115, 0), __file__, sys_modules_391627, sys_modules_391627.module_type_store, module_type_store)
    else:
        from scipy.sparse.linalg.dsolve import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 115, 0), 'scipy.sparse.linalg.dsolve', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.sparse.linalg.dsolve' (line 115)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 0), 'scipy.sparse.linalg.dsolve', import_391626)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 116, 0))

# 'from scipy.sparse.linalg.interface import ' statement (line 116)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/')
import_391628 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 116, 0), 'scipy.sparse.linalg.interface')

if (type(import_391628) is not StypyTypeError):

    if (import_391628 != 'pyd_module'):
        __import__(import_391628)
        sys_modules_391629 = sys.modules[import_391628]
        import_from_module(stypy.reporting.localization.Localization(__file__, 116, 0), 'scipy.sparse.linalg.interface', sys_modules_391629.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 116, 0), __file__, sys_modules_391629, sys_modules_391629.module_type_store, module_type_store)
    else:
        from scipy.sparse.linalg.interface import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 116, 0), 'scipy.sparse.linalg.interface', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.sparse.linalg.interface' (line 116)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 0), 'scipy.sparse.linalg.interface', import_391628)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 117, 0))

# 'from scipy.sparse.linalg.eigen import ' statement (line 117)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/')
import_391630 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 117, 0), 'scipy.sparse.linalg.eigen')

if (type(import_391630) is not StypyTypeError):

    if (import_391630 != 'pyd_module'):
        __import__(import_391630)
        sys_modules_391631 = sys.modules[import_391630]
        import_from_module(stypy.reporting.localization.Localization(__file__, 117, 0), 'scipy.sparse.linalg.eigen', sys_modules_391631.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 117, 0), __file__, sys_modules_391631, sys_modules_391631.module_type_store, module_type_store)
    else:
        from scipy.sparse.linalg.eigen import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 117, 0), 'scipy.sparse.linalg.eigen', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.sparse.linalg.eigen' (line 117)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 0), 'scipy.sparse.linalg.eigen', import_391630)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 118, 0))

# 'from scipy.sparse.linalg.matfuncs import ' statement (line 118)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/')
import_391632 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 118, 0), 'scipy.sparse.linalg.matfuncs')

if (type(import_391632) is not StypyTypeError):

    if (import_391632 != 'pyd_module'):
        __import__(import_391632)
        sys_modules_391633 = sys.modules[import_391632]
        import_from_module(stypy.reporting.localization.Localization(__file__, 118, 0), 'scipy.sparse.linalg.matfuncs', sys_modules_391633.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 118, 0), __file__, sys_modules_391633, sys_modules_391633.module_type_store, module_type_store)
    else:
        from scipy.sparse.linalg.matfuncs import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 118, 0), 'scipy.sparse.linalg.matfuncs', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.sparse.linalg.matfuncs' (line 118)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 0), 'scipy.sparse.linalg.matfuncs', import_391632)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 119, 0))

# 'from scipy.sparse.linalg._onenormest import ' statement (line 119)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/')
import_391634 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 119, 0), 'scipy.sparse.linalg._onenormest')

if (type(import_391634) is not StypyTypeError):

    if (import_391634 != 'pyd_module'):
        __import__(import_391634)
        sys_modules_391635 = sys.modules[import_391634]
        import_from_module(stypy.reporting.localization.Localization(__file__, 119, 0), 'scipy.sparse.linalg._onenormest', sys_modules_391635.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 119, 0), __file__, sys_modules_391635, sys_modules_391635.module_type_store, module_type_store)
    else:
        from scipy.sparse.linalg._onenormest import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 119, 0), 'scipy.sparse.linalg._onenormest', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.sparse.linalg._onenormest' (line 119)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 0), 'scipy.sparse.linalg._onenormest', import_391634)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 120, 0))

# 'from scipy.sparse.linalg._norm import ' statement (line 120)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/')
import_391636 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 120, 0), 'scipy.sparse.linalg._norm')

if (type(import_391636) is not StypyTypeError):

    if (import_391636 != 'pyd_module'):
        __import__(import_391636)
        sys_modules_391637 = sys.modules[import_391636]
        import_from_module(stypy.reporting.localization.Localization(__file__, 120, 0), 'scipy.sparse.linalg._norm', sys_modules_391637.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 120, 0), __file__, sys_modules_391637, sys_modules_391637.module_type_store, module_type_store)
    else:
        from scipy.sparse.linalg._norm import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 120, 0), 'scipy.sparse.linalg._norm', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.sparse.linalg._norm' (line 120)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 0), 'scipy.sparse.linalg._norm', import_391636)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 121, 0))

# 'from scipy.sparse.linalg._expm_multiply import ' statement (line 121)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/')
import_391638 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 121, 0), 'scipy.sparse.linalg._expm_multiply')

if (type(import_391638) is not StypyTypeError):

    if (import_391638 != 'pyd_module'):
        __import__(import_391638)
        sys_modules_391639 = sys.modules[import_391638]
        import_from_module(stypy.reporting.localization.Localization(__file__, 121, 0), 'scipy.sparse.linalg._expm_multiply', sys_modules_391639.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 121, 0), __file__, sys_modules_391639, sys_modules_391639.module_type_store, module_type_store)
    else:
        from scipy.sparse.linalg._expm_multiply import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 121, 0), 'scipy.sparse.linalg._expm_multiply', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.sparse.linalg._expm_multiply' (line 121)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 0), 'scipy.sparse.linalg._expm_multiply', import_391638)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/')


# Assigning a ListComp to a Name (line 123):
# Calculating list comprehension
# Calculating comprehension expression

# Call to dir(...): (line 123)
# Processing the call keyword arguments (line 123)
kwargs_391648 = {}
# Getting the type of 'dir' (line 123)
dir_391647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 22), 'dir', False)
# Calling dir(args, kwargs) (line 123)
dir_call_result_391649 = invoke(stypy.reporting.localization.Localization(__file__, 123, 22), dir_391647, *[], **kwargs_391648)

comprehension_391650 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 11), dir_call_result_391649)
# Assigning a type to the variable 's' (line 123)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 11), 's', comprehension_391650)


# Call to startswith(...): (line 123)
# Processing the call arguments (line 123)
str_391643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 48), 'str', '_')
# Processing the call keyword arguments (line 123)
kwargs_391644 = {}
# Getting the type of 's' (line 123)
s_391641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 35), 's', False)
# Obtaining the member 'startswith' of a type (line 123)
startswith_391642 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 35), s_391641, 'startswith')
# Calling startswith(args, kwargs) (line 123)
startswith_call_result_391645 = invoke(stypy.reporting.localization.Localization(__file__, 123, 35), startswith_391642, *[str_391643], **kwargs_391644)

# Applying the 'not' unary operator (line 123)
result_not__391646 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 31), 'not', startswith_call_result_391645)

# Getting the type of 's' (line 123)
s_391640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 11), 's')
list_391651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 11), 'list')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 11), list_391651, s_391640)
# Assigning a type to the variable '__all__' (line 123)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 0), '__all__', list_391651)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 125, 0))

# 'from scipy._lib._testutils import PytestTester' statement (line 125)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/')
import_391652 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 125, 0), 'scipy._lib._testutils')

if (type(import_391652) is not StypyTypeError):

    if (import_391652 != 'pyd_module'):
        __import__(import_391652)
        sys_modules_391653 = sys.modules[import_391652]
        import_from_module(stypy.reporting.localization.Localization(__file__, 125, 0), 'scipy._lib._testutils', sys_modules_391653.module_type_store, module_type_store, ['PytestTester'])
        nest_module(stypy.reporting.localization.Localization(__file__, 125, 0), __file__, sys_modules_391653, sys_modules_391653.module_type_store, module_type_store)
    else:
        from scipy._lib._testutils import PytestTester

        import_from_module(stypy.reporting.localization.Localization(__file__, 125, 0), 'scipy._lib._testutils', None, module_type_store, ['PytestTester'], [PytestTester])

else:
    # Assigning a type to the variable 'scipy._lib._testutils' (line 125)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 0), 'scipy._lib._testutils', import_391652)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/')


# Assigning a Call to a Name (line 126):

# Call to PytestTester(...): (line 126)
# Processing the call arguments (line 126)
# Getting the type of '__name__' (line 126)
name___391655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 20), '__name__', False)
# Processing the call keyword arguments (line 126)
kwargs_391656 = {}
# Getting the type of 'PytestTester' (line 126)
PytestTester_391654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 7), 'PytestTester', False)
# Calling PytestTester(args, kwargs) (line 126)
PytestTester_call_result_391657 = invoke(stypy.reporting.localization.Localization(__file__, 126, 7), PytestTester_391654, *[name___391655], **kwargs_391656)

# Assigning a type to the variable 'test' (line 126)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 0), 'test', PytestTester_call_result_391657)
# Deleting a member
module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 127, 0), module_type_store, 'PytestTester')

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
