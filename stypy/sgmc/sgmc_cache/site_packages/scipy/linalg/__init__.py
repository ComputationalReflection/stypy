
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: ====================================
3: Linear algebra (:mod:`scipy.linalg`)
4: ====================================
5: 
6: .. currentmodule:: scipy.linalg
7: 
8: Linear algebra functions.
9: 
10: .. seealso::
11: 
12:    `numpy.linalg` for more linear algebra functions.  Note that
13:    although `scipy.linalg` imports most of them, identically named
14:    functions from `scipy.linalg` may offer more or slightly differing
15:    functionality.
16: 
17: 
18: Basics
19: ======
20: 
21: .. autosummary::
22:    :toctree: generated/
23: 
24:    inv - Find the inverse of a square matrix
25:    solve - Solve a linear system of equations
26:    solve_banded - Solve a banded linear system
27:    solveh_banded - Solve a Hermitian or symmetric banded system
28:    solve_circulant - Solve a circulant system
29:    solve_triangular - Solve a triangular matrix
30:    solve_toeplitz - Solve a toeplitz matrix
31:    det - Find the determinant of a square matrix
32:    norm - Matrix and vector norm
33:    lstsq - Solve a linear least-squares problem
34:    pinv - Pseudo-inverse (Moore-Penrose) using lstsq
35:    pinv2 - Pseudo-inverse using svd
36:    pinvh - Pseudo-inverse of hermitian matrix
37:    kron - Kronecker product of two arrays
38:    tril - Construct a lower-triangular matrix from a given matrix
39:    triu - Construct an upper-triangular matrix from a given matrix
40:    orthogonal_procrustes - Solve an orthogonal Procrustes problem
41:    matrix_balance - Balance matrix entries with a similarity transformation
42:    subspace_angles - Compute the subspace angles between two matrices
43:    LinAlgError
44: 
45: Eigenvalue Problems
46: ===================
47: 
48: .. autosummary::
49:    :toctree: generated/
50: 
51:    eig - Find the eigenvalues and eigenvectors of a square matrix
52:    eigvals - Find just the eigenvalues of a square matrix
53:    eigh - Find the e-vals and e-vectors of a Hermitian or symmetric matrix
54:    eigvalsh - Find just the eigenvalues of a Hermitian or symmetric matrix
55:    eig_banded - Find the eigenvalues and eigenvectors of a banded matrix
56:    eigvals_banded - Find just the eigenvalues of a banded matrix
57:    eigh_tridiagonal - Find the eigenvalues and eigenvectors of a tridiagonal matrix
58:    eigvalsh_tridiagonal - Find just the eigenvalues of a tridiagonal matrix
59: 
60: Decompositions
61: ==============
62: 
63: .. autosummary::
64:    :toctree: generated/
65: 
66:    lu - LU decomposition of a matrix
67:    lu_factor - LU decomposition returning unordered matrix and pivots
68:    lu_solve - Solve Ax=b using back substitution with output of lu_factor
69:    svd - Singular value decomposition of a matrix
70:    svdvals - Singular values of a matrix
71:    diagsvd - Construct matrix of singular values from output of svd
72:    orth - Construct orthonormal basis for the range of A using svd
73:    cholesky - Cholesky decomposition of a matrix
74:    cholesky_banded - Cholesky decomp. of a sym. or Hermitian banded matrix
75:    cho_factor - Cholesky decomposition for use in solving a linear system
76:    cho_solve - Solve previously factored linear system
77:    cho_solve_banded - Solve previously factored banded linear system
78:    polar - Compute the polar decomposition.
79:    qr - QR decomposition of a matrix
80:    qr_multiply - QR decomposition and multiplication by Q
81:    qr_update - Rank k QR update
82:    qr_delete - QR downdate on row or column deletion
83:    qr_insert - QR update on row or column insertion
84:    rq - RQ decomposition of a matrix
85:    qz - QZ decomposition of a pair of matrices
86:    ordqz - QZ decomposition of a pair of matrices with reordering
87:    schur - Schur decomposition of a matrix
88:    rsf2csf - Real to complex Schur form
89:    hessenberg - Hessenberg form of a matrix
90: 
91: .. seealso::
92: 
93:    `scipy.linalg.interpolative` -- Interpolative matrix decompositions
94: 
95: 
96: Matrix Functions
97: ================
98: 
99: .. autosummary::
100:    :toctree: generated/
101: 
102:    expm - Matrix exponential
103:    logm - Matrix logarithm
104:    cosm - Matrix cosine
105:    sinm - Matrix sine
106:    tanm - Matrix tangent
107:    coshm - Matrix hyperbolic cosine
108:    sinhm - Matrix hyperbolic sine
109:    tanhm - Matrix hyperbolic tangent
110:    signm - Matrix sign
111:    sqrtm - Matrix square root
112:    funm - Evaluating an arbitrary matrix function
113:    expm_frechet - Frechet derivative of the matrix exponential
114:    expm_cond - Relative condition number of expm in the Frobenius norm
115:    fractional_matrix_power - Fractional matrix power
116: 
117: 
118: Matrix Equation Solvers
119: =======================
120: 
121: .. autosummary::
122:    :toctree: generated/
123: 
124:    solve_sylvester - Solve the Sylvester matrix equation
125:    solve_continuous_are - Solve the continuous-time algebraic Riccati equation
126:    solve_discrete_are - Solve the discrete-time algebraic Riccati equation
127:    solve_continuous_lyapunov - Solve the continous-time Lyapunov equation
128:    solve_discrete_lyapunov - Solve the discrete-time Lyapunov equation
129: 
130: 
131: Sketches and Random Projections
132: ===============================
133: 
134: .. autosummary::
135:    :toctree: generated/
136: 
137:    clarkson_woodruff_transform - Applies the Clarkson Woodruff Sketch (a.k.a CountMin Sketch)
138: 
139: Special Matrices
140: ================
141: 
142: .. autosummary::
143:    :toctree: generated/
144: 
145:    block_diag - Construct a block diagonal matrix from submatrices
146:    circulant - Circulant matrix
147:    companion - Companion matrix
148:    dft - Discrete Fourier transform matrix
149:    hadamard - Hadamard matrix of order 2**n
150:    hankel - Hankel matrix
151:    helmert - Helmert matrix
152:    hilbert - Hilbert matrix
153:    invhilbert - Inverse Hilbert matrix
154:    leslie - Leslie matrix
155:    pascal - Pascal matrix
156:    invpascal - Inverse Pascal matrix
157:    toeplitz - Toeplitz matrix
158:    tri - Construct a matrix filled with ones at and below a given diagonal
159: 
160: Low-level routines
161: ==================
162: 
163: .. autosummary::
164:    :toctree: generated/
165: 
166:    get_blas_funcs
167:    get_lapack_funcs
168:    find_best_blas_type
169: 
170: .. seealso::
171: 
172:    `scipy.linalg.blas` -- Low-level BLAS functions
173: 
174:    `scipy.linalg.lapack` -- Low-level LAPACK functions
175: 
176:    `scipy.linalg.cython_blas` -- Low-level BLAS functions for Cython
177: 
178:    `scipy.linalg.cython_lapack` -- Low-level LAPACK functions for Cython
179: 
180: '''  # noqa: E501
181: 
182: from __future__ import division, print_function, absolute_import
183: 
184: from .linalg_version import linalg_version as __version__
185: 
186: from .misc import *
187: from .basic import *
188: from .decomp import *
189: from .decomp_lu import *
190: from .decomp_cholesky import *
191: from .decomp_qr import *
192: from ._decomp_qz import *
193: from .decomp_svd import *
194: from .decomp_schur import *
195: from ._decomp_polar import *
196: from .matfuncs import *
197: from .blas import *
198: from .lapack import *
199: from .special_matrices import *
200: from ._solvers import *
201: from ._procrustes import *
202: from ._decomp_update import *
203: from ._sketches import *
204: 
205: __all__ = [s for s in dir() if not s.startswith('_')]
206: 
207: from numpy.dual import register_func
208: for k in ['norm', 'inv', 'svd', 'solve', 'det', 'eig', 'eigh', 'eigvals',
209:           'eigvalsh', 'lstsq', 'cholesky']:
210:     try:
211:         register_func(k, eval(k))
212:     except ValueError:
213:         pass
214: 
215: try:
216:     register_func('pinv', pinv2)
217: except ValueError:
218:     pass
219: 
220: del k, register_func
221: 
222: from scipy._lib._testutils import PytestTester
223: test = PytestTester(__name__)
224: del PytestTester
225: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_38722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, (-1)), 'str', '\n====================================\nLinear algebra (:mod:`scipy.linalg`)\n====================================\n\n.. currentmodule:: scipy.linalg\n\nLinear algebra functions.\n\n.. seealso::\n\n   `numpy.linalg` for more linear algebra functions.  Note that\n   although `scipy.linalg` imports most of them, identically named\n   functions from `scipy.linalg` may offer more or slightly differing\n   functionality.\n\n\nBasics\n======\n\n.. autosummary::\n   :toctree: generated/\n\n   inv - Find the inverse of a square matrix\n   solve - Solve a linear system of equations\n   solve_banded - Solve a banded linear system\n   solveh_banded - Solve a Hermitian or symmetric banded system\n   solve_circulant - Solve a circulant system\n   solve_triangular - Solve a triangular matrix\n   solve_toeplitz - Solve a toeplitz matrix\n   det - Find the determinant of a square matrix\n   norm - Matrix and vector norm\n   lstsq - Solve a linear least-squares problem\n   pinv - Pseudo-inverse (Moore-Penrose) using lstsq\n   pinv2 - Pseudo-inverse using svd\n   pinvh - Pseudo-inverse of hermitian matrix\n   kron - Kronecker product of two arrays\n   tril - Construct a lower-triangular matrix from a given matrix\n   triu - Construct an upper-triangular matrix from a given matrix\n   orthogonal_procrustes - Solve an orthogonal Procrustes problem\n   matrix_balance - Balance matrix entries with a similarity transformation\n   subspace_angles - Compute the subspace angles between two matrices\n   LinAlgError\n\nEigenvalue Problems\n===================\n\n.. autosummary::\n   :toctree: generated/\n\n   eig - Find the eigenvalues and eigenvectors of a square matrix\n   eigvals - Find just the eigenvalues of a square matrix\n   eigh - Find the e-vals and e-vectors of a Hermitian or symmetric matrix\n   eigvalsh - Find just the eigenvalues of a Hermitian or symmetric matrix\n   eig_banded - Find the eigenvalues and eigenvectors of a banded matrix\n   eigvals_banded - Find just the eigenvalues of a banded matrix\n   eigh_tridiagonal - Find the eigenvalues and eigenvectors of a tridiagonal matrix\n   eigvalsh_tridiagonal - Find just the eigenvalues of a tridiagonal matrix\n\nDecompositions\n==============\n\n.. autosummary::\n   :toctree: generated/\n\n   lu - LU decomposition of a matrix\n   lu_factor - LU decomposition returning unordered matrix and pivots\n   lu_solve - Solve Ax=b using back substitution with output of lu_factor\n   svd - Singular value decomposition of a matrix\n   svdvals - Singular values of a matrix\n   diagsvd - Construct matrix of singular values from output of svd\n   orth - Construct orthonormal basis for the range of A using svd\n   cholesky - Cholesky decomposition of a matrix\n   cholesky_banded - Cholesky decomp. of a sym. or Hermitian banded matrix\n   cho_factor - Cholesky decomposition for use in solving a linear system\n   cho_solve - Solve previously factored linear system\n   cho_solve_banded - Solve previously factored banded linear system\n   polar - Compute the polar decomposition.\n   qr - QR decomposition of a matrix\n   qr_multiply - QR decomposition and multiplication by Q\n   qr_update - Rank k QR update\n   qr_delete - QR downdate on row or column deletion\n   qr_insert - QR update on row or column insertion\n   rq - RQ decomposition of a matrix\n   qz - QZ decomposition of a pair of matrices\n   ordqz - QZ decomposition of a pair of matrices with reordering\n   schur - Schur decomposition of a matrix\n   rsf2csf - Real to complex Schur form\n   hessenberg - Hessenberg form of a matrix\n\n.. seealso::\n\n   `scipy.linalg.interpolative` -- Interpolative matrix decompositions\n\n\nMatrix Functions\n================\n\n.. autosummary::\n   :toctree: generated/\n\n   expm - Matrix exponential\n   logm - Matrix logarithm\n   cosm - Matrix cosine\n   sinm - Matrix sine\n   tanm - Matrix tangent\n   coshm - Matrix hyperbolic cosine\n   sinhm - Matrix hyperbolic sine\n   tanhm - Matrix hyperbolic tangent\n   signm - Matrix sign\n   sqrtm - Matrix square root\n   funm - Evaluating an arbitrary matrix function\n   expm_frechet - Frechet derivative of the matrix exponential\n   expm_cond - Relative condition number of expm in the Frobenius norm\n   fractional_matrix_power - Fractional matrix power\n\n\nMatrix Equation Solvers\n=======================\n\n.. autosummary::\n   :toctree: generated/\n\n   solve_sylvester - Solve the Sylvester matrix equation\n   solve_continuous_are - Solve the continuous-time algebraic Riccati equation\n   solve_discrete_are - Solve the discrete-time algebraic Riccati equation\n   solve_continuous_lyapunov - Solve the continous-time Lyapunov equation\n   solve_discrete_lyapunov - Solve the discrete-time Lyapunov equation\n\n\nSketches and Random Projections\n===============================\n\n.. autosummary::\n   :toctree: generated/\n\n   clarkson_woodruff_transform - Applies the Clarkson Woodruff Sketch (a.k.a CountMin Sketch)\n\nSpecial Matrices\n================\n\n.. autosummary::\n   :toctree: generated/\n\n   block_diag - Construct a block diagonal matrix from submatrices\n   circulant - Circulant matrix\n   companion - Companion matrix\n   dft - Discrete Fourier transform matrix\n   hadamard - Hadamard matrix of order 2**n\n   hankel - Hankel matrix\n   helmert - Helmert matrix\n   hilbert - Hilbert matrix\n   invhilbert - Inverse Hilbert matrix\n   leslie - Leslie matrix\n   pascal - Pascal matrix\n   invpascal - Inverse Pascal matrix\n   toeplitz - Toeplitz matrix\n   tri - Construct a matrix filled with ones at and below a given diagonal\n\nLow-level routines\n==================\n\n.. autosummary::\n   :toctree: generated/\n\n   get_blas_funcs\n   get_lapack_funcs\n   find_best_blas_type\n\n.. seealso::\n\n   `scipy.linalg.blas` -- Low-level BLAS functions\n\n   `scipy.linalg.lapack` -- Low-level LAPACK functions\n\n   `scipy.linalg.cython_blas` -- Low-level BLAS functions for Cython\n\n   `scipy.linalg.cython_lapack` -- Low-level LAPACK functions for Cython\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 184, 0))

# 'from scipy.linalg.linalg_version import __version__' statement (line 184)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_38723 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 184, 0), 'scipy.linalg.linalg_version')

if (type(import_38723) is not StypyTypeError):

    if (import_38723 != 'pyd_module'):
        __import__(import_38723)
        sys_modules_38724 = sys.modules[import_38723]
        import_from_module(stypy.reporting.localization.Localization(__file__, 184, 0), 'scipy.linalg.linalg_version', sys_modules_38724.module_type_store, module_type_store, ['linalg_version'])
        nest_module(stypy.reporting.localization.Localization(__file__, 184, 0), __file__, sys_modules_38724, sys_modules_38724.module_type_store, module_type_store)
    else:
        from scipy.linalg.linalg_version import linalg_version as __version__

        import_from_module(stypy.reporting.localization.Localization(__file__, 184, 0), 'scipy.linalg.linalg_version', None, module_type_store, ['linalg_version'], [__version__])

else:
    # Assigning a type to the variable 'scipy.linalg.linalg_version' (line 184)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 0), 'scipy.linalg.linalg_version', import_38723)

# Adding an alias
module_type_store.add_alias('__version__', 'linalg_version')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 186, 0))

# 'from scipy.linalg.misc import ' statement (line 186)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_38725 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 186, 0), 'scipy.linalg.misc')

if (type(import_38725) is not StypyTypeError):

    if (import_38725 != 'pyd_module'):
        __import__(import_38725)
        sys_modules_38726 = sys.modules[import_38725]
        import_from_module(stypy.reporting.localization.Localization(__file__, 186, 0), 'scipy.linalg.misc', sys_modules_38726.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 186, 0), __file__, sys_modules_38726, sys_modules_38726.module_type_store, module_type_store)
    else:
        from scipy.linalg.misc import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 186, 0), 'scipy.linalg.misc', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.linalg.misc' (line 186)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 0), 'scipy.linalg.misc', import_38725)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 187, 0))

# 'from scipy.linalg.basic import ' statement (line 187)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_38727 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 187, 0), 'scipy.linalg.basic')

if (type(import_38727) is not StypyTypeError):

    if (import_38727 != 'pyd_module'):
        __import__(import_38727)
        sys_modules_38728 = sys.modules[import_38727]
        import_from_module(stypy.reporting.localization.Localization(__file__, 187, 0), 'scipy.linalg.basic', sys_modules_38728.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 187, 0), __file__, sys_modules_38728, sys_modules_38728.module_type_store, module_type_store)
    else:
        from scipy.linalg.basic import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 187, 0), 'scipy.linalg.basic', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.linalg.basic' (line 187)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 0), 'scipy.linalg.basic', import_38727)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 188, 0))

# 'from scipy.linalg.decomp import ' statement (line 188)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_38729 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 188, 0), 'scipy.linalg.decomp')

if (type(import_38729) is not StypyTypeError):

    if (import_38729 != 'pyd_module'):
        __import__(import_38729)
        sys_modules_38730 = sys.modules[import_38729]
        import_from_module(stypy.reporting.localization.Localization(__file__, 188, 0), 'scipy.linalg.decomp', sys_modules_38730.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 188, 0), __file__, sys_modules_38730, sys_modules_38730.module_type_store, module_type_store)
    else:
        from scipy.linalg.decomp import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 188, 0), 'scipy.linalg.decomp', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.linalg.decomp' (line 188)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 0), 'scipy.linalg.decomp', import_38729)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 189, 0))

# 'from scipy.linalg.decomp_lu import ' statement (line 189)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_38731 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 189, 0), 'scipy.linalg.decomp_lu')

if (type(import_38731) is not StypyTypeError):

    if (import_38731 != 'pyd_module'):
        __import__(import_38731)
        sys_modules_38732 = sys.modules[import_38731]
        import_from_module(stypy.reporting.localization.Localization(__file__, 189, 0), 'scipy.linalg.decomp_lu', sys_modules_38732.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 189, 0), __file__, sys_modules_38732, sys_modules_38732.module_type_store, module_type_store)
    else:
        from scipy.linalg.decomp_lu import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 189, 0), 'scipy.linalg.decomp_lu', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.linalg.decomp_lu' (line 189)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 0), 'scipy.linalg.decomp_lu', import_38731)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 190, 0))

# 'from scipy.linalg.decomp_cholesky import ' statement (line 190)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_38733 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 190, 0), 'scipy.linalg.decomp_cholesky')

if (type(import_38733) is not StypyTypeError):

    if (import_38733 != 'pyd_module'):
        __import__(import_38733)
        sys_modules_38734 = sys.modules[import_38733]
        import_from_module(stypy.reporting.localization.Localization(__file__, 190, 0), 'scipy.linalg.decomp_cholesky', sys_modules_38734.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 190, 0), __file__, sys_modules_38734, sys_modules_38734.module_type_store, module_type_store)
    else:
        from scipy.linalg.decomp_cholesky import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 190, 0), 'scipy.linalg.decomp_cholesky', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.linalg.decomp_cholesky' (line 190)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 0), 'scipy.linalg.decomp_cholesky', import_38733)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 191, 0))

# 'from scipy.linalg.decomp_qr import ' statement (line 191)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_38735 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 191, 0), 'scipy.linalg.decomp_qr')

if (type(import_38735) is not StypyTypeError):

    if (import_38735 != 'pyd_module'):
        __import__(import_38735)
        sys_modules_38736 = sys.modules[import_38735]
        import_from_module(stypy.reporting.localization.Localization(__file__, 191, 0), 'scipy.linalg.decomp_qr', sys_modules_38736.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 191, 0), __file__, sys_modules_38736, sys_modules_38736.module_type_store, module_type_store)
    else:
        from scipy.linalg.decomp_qr import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 191, 0), 'scipy.linalg.decomp_qr', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.linalg.decomp_qr' (line 191)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 0), 'scipy.linalg.decomp_qr', import_38735)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 192, 0))

# 'from scipy.linalg._decomp_qz import ' statement (line 192)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_38737 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 192, 0), 'scipy.linalg._decomp_qz')

if (type(import_38737) is not StypyTypeError):

    if (import_38737 != 'pyd_module'):
        __import__(import_38737)
        sys_modules_38738 = sys.modules[import_38737]
        import_from_module(stypy.reporting.localization.Localization(__file__, 192, 0), 'scipy.linalg._decomp_qz', sys_modules_38738.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 192, 0), __file__, sys_modules_38738, sys_modules_38738.module_type_store, module_type_store)
    else:
        from scipy.linalg._decomp_qz import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 192, 0), 'scipy.linalg._decomp_qz', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.linalg._decomp_qz' (line 192)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 0), 'scipy.linalg._decomp_qz', import_38737)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 193, 0))

# 'from scipy.linalg.decomp_svd import ' statement (line 193)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_38739 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 193, 0), 'scipy.linalg.decomp_svd')

if (type(import_38739) is not StypyTypeError):

    if (import_38739 != 'pyd_module'):
        __import__(import_38739)
        sys_modules_38740 = sys.modules[import_38739]
        import_from_module(stypy.reporting.localization.Localization(__file__, 193, 0), 'scipy.linalg.decomp_svd', sys_modules_38740.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 193, 0), __file__, sys_modules_38740, sys_modules_38740.module_type_store, module_type_store)
    else:
        from scipy.linalg.decomp_svd import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 193, 0), 'scipy.linalg.decomp_svd', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.linalg.decomp_svd' (line 193)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 0), 'scipy.linalg.decomp_svd', import_38739)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 194, 0))

# 'from scipy.linalg.decomp_schur import ' statement (line 194)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_38741 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 194, 0), 'scipy.linalg.decomp_schur')

if (type(import_38741) is not StypyTypeError):

    if (import_38741 != 'pyd_module'):
        __import__(import_38741)
        sys_modules_38742 = sys.modules[import_38741]
        import_from_module(stypy.reporting.localization.Localization(__file__, 194, 0), 'scipy.linalg.decomp_schur', sys_modules_38742.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 194, 0), __file__, sys_modules_38742, sys_modules_38742.module_type_store, module_type_store)
    else:
        from scipy.linalg.decomp_schur import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 194, 0), 'scipy.linalg.decomp_schur', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.linalg.decomp_schur' (line 194)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 0), 'scipy.linalg.decomp_schur', import_38741)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 195, 0))

# 'from scipy.linalg._decomp_polar import ' statement (line 195)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_38743 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 195, 0), 'scipy.linalg._decomp_polar')

if (type(import_38743) is not StypyTypeError):

    if (import_38743 != 'pyd_module'):
        __import__(import_38743)
        sys_modules_38744 = sys.modules[import_38743]
        import_from_module(stypy.reporting.localization.Localization(__file__, 195, 0), 'scipy.linalg._decomp_polar', sys_modules_38744.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 195, 0), __file__, sys_modules_38744, sys_modules_38744.module_type_store, module_type_store)
    else:
        from scipy.linalg._decomp_polar import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 195, 0), 'scipy.linalg._decomp_polar', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.linalg._decomp_polar' (line 195)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 0), 'scipy.linalg._decomp_polar', import_38743)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 196, 0))

# 'from scipy.linalg.matfuncs import ' statement (line 196)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_38745 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 196, 0), 'scipy.linalg.matfuncs')

if (type(import_38745) is not StypyTypeError):

    if (import_38745 != 'pyd_module'):
        __import__(import_38745)
        sys_modules_38746 = sys.modules[import_38745]
        import_from_module(stypy.reporting.localization.Localization(__file__, 196, 0), 'scipy.linalg.matfuncs', sys_modules_38746.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 196, 0), __file__, sys_modules_38746, sys_modules_38746.module_type_store, module_type_store)
    else:
        from scipy.linalg.matfuncs import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 196, 0), 'scipy.linalg.matfuncs', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.linalg.matfuncs' (line 196)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 0), 'scipy.linalg.matfuncs', import_38745)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 197, 0))

# 'from scipy.linalg.blas import ' statement (line 197)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_38747 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 197, 0), 'scipy.linalg.blas')

if (type(import_38747) is not StypyTypeError):

    if (import_38747 != 'pyd_module'):
        __import__(import_38747)
        sys_modules_38748 = sys.modules[import_38747]
        import_from_module(stypy.reporting.localization.Localization(__file__, 197, 0), 'scipy.linalg.blas', sys_modules_38748.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 197, 0), __file__, sys_modules_38748, sys_modules_38748.module_type_store, module_type_store)
    else:
        from scipy.linalg.blas import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 197, 0), 'scipy.linalg.blas', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.linalg.blas' (line 197)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 0), 'scipy.linalg.blas', import_38747)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 198, 0))

# 'from scipy.linalg.lapack import ' statement (line 198)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_38749 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 198, 0), 'scipy.linalg.lapack')

if (type(import_38749) is not StypyTypeError):

    if (import_38749 != 'pyd_module'):
        __import__(import_38749)
        sys_modules_38750 = sys.modules[import_38749]
        import_from_module(stypy.reporting.localization.Localization(__file__, 198, 0), 'scipy.linalg.lapack', sys_modules_38750.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 198, 0), __file__, sys_modules_38750, sys_modules_38750.module_type_store, module_type_store)
    else:
        from scipy.linalg.lapack import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 198, 0), 'scipy.linalg.lapack', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.linalg.lapack' (line 198)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 0), 'scipy.linalg.lapack', import_38749)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 199, 0))

# 'from scipy.linalg.special_matrices import ' statement (line 199)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_38751 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 199, 0), 'scipy.linalg.special_matrices')

if (type(import_38751) is not StypyTypeError):

    if (import_38751 != 'pyd_module'):
        __import__(import_38751)
        sys_modules_38752 = sys.modules[import_38751]
        import_from_module(stypy.reporting.localization.Localization(__file__, 199, 0), 'scipy.linalg.special_matrices', sys_modules_38752.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 199, 0), __file__, sys_modules_38752, sys_modules_38752.module_type_store, module_type_store)
    else:
        from scipy.linalg.special_matrices import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 199, 0), 'scipy.linalg.special_matrices', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.linalg.special_matrices' (line 199)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 0), 'scipy.linalg.special_matrices', import_38751)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 200, 0))

# 'from scipy.linalg._solvers import ' statement (line 200)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_38753 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 200, 0), 'scipy.linalg._solvers')

if (type(import_38753) is not StypyTypeError):

    if (import_38753 != 'pyd_module'):
        __import__(import_38753)
        sys_modules_38754 = sys.modules[import_38753]
        import_from_module(stypy.reporting.localization.Localization(__file__, 200, 0), 'scipy.linalg._solvers', sys_modules_38754.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 200, 0), __file__, sys_modules_38754, sys_modules_38754.module_type_store, module_type_store)
    else:
        from scipy.linalg._solvers import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 200, 0), 'scipy.linalg._solvers', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.linalg._solvers' (line 200)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 0), 'scipy.linalg._solvers', import_38753)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 201, 0))

# 'from scipy.linalg._procrustes import ' statement (line 201)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_38755 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 201, 0), 'scipy.linalg._procrustes')

if (type(import_38755) is not StypyTypeError):

    if (import_38755 != 'pyd_module'):
        __import__(import_38755)
        sys_modules_38756 = sys.modules[import_38755]
        import_from_module(stypy.reporting.localization.Localization(__file__, 201, 0), 'scipy.linalg._procrustes', sys_modules_38756.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 201, 0), __file__, sys_modules_38756, sys_modules_38756.module_type_store, module_type_store)
    else:
        from scipy.linalg._procrustes import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 201, 0), 'scipy.linalg._procrustes', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.linalg._procrustes' (line 201)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 0), 'scipy.linalg._procrustes', import_38755)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 202, 0))

# 'from scipy.linalg._decomp_update import ' statement (line 202)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_38757 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 202, 0), 'scipy.linalg._decomp_update')

if (type(import_38757) is not StypyTypeError):

    if (import_38757 != 'pyd_module'):
        __import__(import_38757)
        sys_modules_38758 = sys.modules[import_38757]
        import_from_module(stypy.reporting.localization.Localization(__file__, 202, 0), 'scipy.linalg._decomp_update', sys_modules_38758.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 202, 0), __file__, sys_modules_38758, sys_modules_38758.module_type_store, module_type_store)
    else:
        from scipy.linalg._decomp_update import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 202, 0), 'scipy.linalg._decomp_update', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.linalg._decomp_update' (line 202)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 0), 'scipy.linalg._decomp_update', import_38757)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 203, 0))

# 'from scipy.linalg._sketches import ' statement (line 203)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_38759 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 203, 0), 'scipy.linalg._sketches')

if (type(import_38759) is not StypyTypeError):

    if (import_38759 != 'pyd_module'):
        __import__(import_38759)
        sys_modules_38760 = sys.modules[import_38759]
        import_from_module(stypy.reporting.localization.Localization(__file__, 203, 0), 'scipy.linalg._sketches', sys_modules_38760.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 203, 0), __file__, sys_modules_38760, sys_modules_38760.module_type_store, module_type_store)
    else:
        from scipy.linalg._sketches import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 203, 0), 'scipy.linalg._sketches', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.linalg._sketches' (line 203)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 0), 'scipy.linalg._sketches', import_38759)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')


# Assigning a ListComp to a Name (line 205):
# Calculating list comprehension
# Calculating comprehension expression

# Call to dir(...): (line 205)
# Processing the call keyword arguments (line 205)
kwargs_38769 = {}
# Getting the type of 'dir' (line 205)
dir_38768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 22), 'dir', False)
# Calling dir(args, kwargs) (line 205)
dir_call_result_38770 = invoke(stypy.reporting.localization.Localization(__file__, 205, 22), dir_38768, *[], **kwargs_38769)

comprehension_38771 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 205, 11), dir_call_result_38770)
# Assigning a type to the variable 's' (line 205)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 11), 's', comprehension_38771)


# Call to startswith(...): (line 205)
# Processing the call arguments (line 205)
str_38764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 48), 'str', '_')
# Processing the call keyword arguments (line 205)
kwargs_38765 = {}
# Getting the type of 's' (line 205)
s_38762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 35), 's', False)
# Obtaining the member 'startswith' of a type (line 205)
startswith_38763 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 35), s_38762, 'startswith')
# Calling startswith(args, kwargs) (line 205)
startswith_call_result_38766 = invoke(stypy.reporting.localization.Localization(__file__, 205, 35), startswith_38763, *[str_38764], **kwargs_38765)

# Applying the 'not' unary operator (line 205)
result_not__38767 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 31), 'not', startswith_call_result_38766)

# Getting the type of 's' (line 205)
s_38761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 11), 's')
list_38772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 11), 'list')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 205, 11), list_38772, s_38761)
# Assigning a type to the variable '__all__' (line 205)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 0), '__all__', list_38772)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 207, 0))

# 'from numpy.dual import register_func' statement (line 207)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_38773 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 207, 0), 'numpy.dual')

if (type(import_38773) is not StypyTypeError):

    if (import_38773 != 'pyd_module'):
        __import__(import_38773)
        sys_modules_38774 = sys.modules[import_38773]
        import_from_module(stypy.reporting.localization.Localization(__file__, 207, 0), 'numpy.dual', sys_modules_38774.module_type_store, module_type_store, ['register_func'])
        nest_module(stypy.reporting.localization.Localization(__file__, 207, 0), __file__, sys_modules_38774, sys_modules_38774.module_type_store, module_type_store)
    else:
        from numpy.dual import register_func

        import_from_module(stypy.reporting.localization.Localization(__file__, 207, 0), 'numpy.dual', None, module_type_store, ['register_func'], [register_func])

else:
    # Assigning a type to the variable 'numpy.dual' (line 207)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 0), 'numpy.dual', import_38773)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')



# Obtaining an instance of the builtin type 'list' (line 208)
list_38775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 9), 'list')
# Adding type elements to the builtin type 'list' instance (line 208)
# Adding element type (line 208)
str_38776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 10), 'str', 'norm')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 9), list_38775, str_38776)
# Adding element type (line 208)
str_38777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 18), 'str', 'inv')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 9), list_38775, str_38777)
# Adding element type (line 208)
str_38778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 25), 'str', 'svd')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 9), list_38775, str_38778)
# Adding element type (line 208)
str_38779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 32), 'str', 'solve')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 9), list_38775, str_38779)
# Adding element type (line 208)
str_38780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 41), 'str', 'det')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 9), list_38775, str_38780)
# Adding element type (line 208)
str_38781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 48), 'str', 'eig')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 9), list_38775, str_38781)
# Adding element type (line 208)
str_38782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 55), 'str', 'eigh')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 9), list_38775, str_38782)
# Adding element type (line 208)
str_38783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 63), 'str', 'eigvals')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 9), list_38775, str_38783)
# Adding element type (line 208)
str_38784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 10), 'str', 'eigvalsh')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 9), list_38775, str_38784)
# Adding element type (line 208)
str_38785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 22), 'str', 'lstsq')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 9), list_38775, str_38785)
# Adding element type (line 208)
str_38786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 31), 'str', 'cholesky')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 9), list_38775, str_38786)

# Testing the type of a for loop iterable (line 208)
is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 208, 0), list_38775)
# Getting the type of the for loop variable (line 208)
for_loop_var_38787 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 208, 0), list_38775)
# Assigning a type to the variable 'k' (line 208)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 0), 'k', for_loop_var_38787)
# SSA begins for a for statement (line 208)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')


# SSA begins for try-except statement (line 210)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')

# Call to register_func(...): (line 211)
# Processing the call arguments (line 211)
# Getting the type of 'k' (line 211)
k_38789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 22), 'k', False)

# Call to eval(...): (line 211)
# Processing the call arguments (line 211)
# Getting the type of 'k' (line 211)
k_38791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 30), 'k', False)
# Processing the call keyword arguments (line 211)
kwargs_38792 = {}
# Getting the type of 'eval' (line 211)
eval_38790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 25), 'eval', False)
# Calling eval(args, kwargs) (line 211)
eval_call_result_38793 = invoke(stypy.reporting.localization.Localization(__file__, 211, 25), eval_38790, *[k_38791], **kwargs_38792)

# Processing the call keyword arguments (line 211)
kwargs_38794 = {}
# Getting the type of 'register_func' (line 211)
register_func_38788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 8), 'register_func', False)
# Calling register_func(args, kwargs) (line 211)
register_func_call_result_38795 = invoke(stypy.reporting.localization.Localization(__file__, 211, 8), register_func_38788, *[k_38789, eval_call_result_38793], **kwargs_38794)

# SSA branch for the except part of a try statement (line 210)
# SSA branch for the except 'ValueError' branch of a try statement (line 210)
module_type_store.open_ssa_branch('except')
pass
# SSA join for try-except statement (line 210)
module_type_store = module_type_store.join_ssa_context()

# SSA join for a for statement
module_type_store = module_type_store.join_ssa_context()



# SSA begins for try-except statement (line 215)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')

# Call to register_func(...): (line 216)
# Processing the call arguments (line 216)
str_38797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 18), 'str', 'pinv')
# Getting the type of 'pinv2' (line 216)
pinv2_38798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 26), 'pinv2', False)
# Processing the call keyword arguments (line 216)
kwargs_38799 = {}
# Getting the type of 'register_func' (line 216)
register_func_38796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 4), 'register_func', False)
# Calling register_func(args, kwargs) (line 216)
register_func_call_result_38800 = invoke(stypy.reporting.localization.Localization(__file__, 216, 4), register_func_38796, *[str_38797, pinv2_38798], **kwargs_38799)

# SSA branch for the except part of a try statement (line 215)
# SSA branch for the except 'ValueError' branch of a try statement (line 215)
module_type_store.open_ssa_branch('except')
pass
# SSA join for try-except statement (line 215)
module_type_store = module_type_store.join_ssa_context()

# Deleting a member
module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 220, 0), module_type_store, 'k')
# Deleting a member
module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 220, 0), module_type_store, 'register_func')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 222, 0))

# 'from scipy._lib._testutils import PytestTester' statement (line 222)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_38801 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 222, 0), 'scipy._lib._testutils')

if (type(import_38801) is not StypyTypeError):

    if (import_38801 != 'pyd_module'):
        __import__(import_38801)
        sys_modules_38802 = sys.modules[import_38801]
        import_from_module(stypy.reporting.localization.Localization(__file__, 222, 0), 'scipy._lib._testutils', sys_modules_38802.module_type_store, module_type_store, ['PytestTester'])
        nest_module(stypy.reporting.localization.Localization(__file__, 222, 0), __file__, sys_modules_38802, sys_modules_38802.module_type_store, module_type_store)
    else:
        from scipy._lib._testutils import PytestTester

        import_from_module(stypy.reporting.localization.Localization(__file__, 222, 0), 'scipy._lib._testutils', None, module_type_store, ['PytestTester'], [PytestTester])

else:
    # Assigning a type to the variable 'scipy._lib._testutils' (line 222)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 0), 'scipy._lib._testutils', import_38801)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')


# Assigning a Call to a Name (line 223):

# Call to PytestTester(...): (line 223)
# Processing the call arguments (line 223)
# Getting the type of '__name__' (line 223)
name___38804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 20), '__name__', False)
# Processing the call keyword arguments (line 223)
kwargs_38805 = {}
# Getting the type of 'PytestTester' (line 223)
PytestTester_38803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 7), 'PytestTester', False)
# Calling PytestTester(args, kwargs) (line 223)
PytestTester_call_result_38806 = invoke(stypy.reporting.localization.Localization(__file__, 223, 7), PytestTester_38803, *[name___38804], **kwargs_38805)

# Assigning a type to the variable 'test' (line 223)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 0), 'test', PytestTester_call_result_38806)
# Deleting a member
module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 224, 0), module_type_store, 'PytestTester')

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
