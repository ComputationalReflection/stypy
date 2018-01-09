
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # Copyright (C) 2009, Pauli Virtanen <pav@iki.fi>
2: # Distributed under the same license as Scipy.
3: 
4: from __future__ import division, print_function, absolute_import
5: 
6: import numpy as np
7: from numpy.linalg import LinAlgError
8: from scipy._lib.six import xrange
9: from scipy.linalg import get_blas_funcs, get_lapack_funcs
10: from .utils import make_system
11: 
12: from ._gcrotmk import _fgmres
13: 
14: __all__ = ['lgmres']
15: 
16: 
17: def lgmres(A, b, x0=None, tol=1e-5, maxiter=1000, M=None, callback=None,
18:            inner_m=30, outer_k=3, outer_v=None, store_outer_Av=True,
19:            prepend_outer_v=False):
20:     '''
21:     Solve a matrix equation using the LGMRES algorithm.
22: 
23:     The LGMRES algorithm [1]_ [2]_ is designed to avoid some problems
24:     in the convergence in restarted GMRES, and often converges in fewer
25:     iterations.
26: 
27:     Parameters
28:     ----------
29:     A : {sparse matrix, dense matrix, LinearOperator}
30:         The real or complex N-by-N matrix of the linear system.
31:     b : {array, matrix}
32:         Right hand side of the linear system. Has shape (N,) or (N,1).
33:     x0  : {array, matrix}
34:         Starting guess for the solution.
35:     tol : float, optional
36:         Tolerance to achieve. The algorithm terminates when either the relative
37:         or the absolute residual is below `tol`.
38:     maxiter : int, optional
39:         Maximum number of iterations.  Iteration will stop after maxiter
40:         steps even if the specified tolerance has not been achieved.
41:     M : {sparse matrix, dense matrix, LinearOperator}, optional
42:         Preconditioner for A.  The preconditioner should approximate the
43:         inverse of A.  Effective preconditioning dramatically improves the
44:         rate of convergence, which implies that fewer iterations are needed
45:         to reach a given error tolerance.
46:     callback : function, optional
47:         User-supplied function to call after each iteration.  It is called
48:         as callback(xk), where xk is the current solution vector.
49:     inner_m : int, optional
50:         Number of inner GMRES iterations per each outer iteration.
51:     outer_k : int, optional
52:         Number of vectors to carry between inner GMRES iterations.
53:         According to [1]_, good values are in the range of 1...3.
54:         However, note that if you want to use the additional vectors to
55:         accelerate solving multiple similar problems, larger values may
56:         be beneficial.
57:     outer_v : list of tuples, optional
58:         List containing tuples ``(v, Av)`` of vectors and corresponding
59:         matrix-vector products, used to augment the Krylov subspace, and
60:         carried between inner GMRES iterations. The element ``Av`` can
61:         be `None` if the matrix-vector product should be re-evaluated.
62:         This parameter is modified in-place by `lgmres`, and can be used
63:         to pass "guess" vectors in and out of the algorithm when solving
64:         similar problems.
65:     store_outer_Av : bool, optional
66:         Whether LGMRES should store also A*v in addition to vectors `v`
67:         in the `outer_v` list. Default is True.
68:     prepend_outer_v : bool, optional 
69:         Whether to put outer_v augmentation vectors before Krylov iterates.
70:         In standard LGMRES, prepend_outer_v=False.
71: 
72:     Returns
73:     -------
74:     x : array or matrix
75:         The converged solution.
76:     info : int
77:         Provides convergence information:
78: 
79:             - 0  : successful exit
80:             - >0 : convergence to tolerance not achieved, number of iterations
81:             - <0 : illegal input or breakdown
82: 
83:     Notes
84:     -----
85:     The LGMRES algorithm [1]_ [2]_ is designed to avoid the
86:     slowing of convergence in restarted GMRES, due to alternating
87:     residual vectors. Typically, it often outperforms GMRES(m) of
88:     comparable memory requirements by some measure, or at least is not
89:     much worse.
90: 
91:     Another advantage in this algorithm is that you can supply it with
92:     'guess' vectors in the `outer_v` argument that augment the Krylov
93:     subspace. If the solution lies close to the span of these vectors,
94:     the algorithm converges faster. This can be useful if several very
95:     similar matrices need to be inverted one after another, such as in
96:     Newton-Krylov iteration where the Jacobian matrix often changes
97:     little in the nonlinear steps.
98: 
99:     References
100:     ----------
101:     .. [1] A.H. Baker and E.R. Jessup and T. Manteuffel, "A Technique for
102:              Accelerating the Convergence of Restarted GMRES", SIAM J. Matrix
103:              Anal. Appl. 26, 962 (2005).
104:     .. [2] A.H. Baker, "On Improving the Performance of the Linear Solver
105:              restarted GMRES", PhD thesis, University of Colorado (2003).
106: 
107:     Examples
108:     --------
109:     >>> from scipy.sparse import csc_matrix
110:     >>> from scipy.sparse.linalg import lgmres
111:     >>> A = csc_matrix([[3, 2, 0], [1, -1, 0], [0, 5, 1]], dtype=float)
112:     >>> b = np.array([2, 4, -1], dtype=float)
113:     >>> x, exitCode = lgmres(A, b)
114:     >>> print(exitCode)            # 0 indicates successful convergence
115:     0
116:     >>> np.allclose(A.dot(x), b)
117:     True
118:     '''
119:     A,M,x,b,postprocess = make_system(A,M,x0,b)
120: 
121:     if not np.isfinite(b).all():
122:         raise ValueError("RHS must contain only finite numbers")
123: 
124:     matvec = A.matvec
125:     psolve = M.matvec
126: 
127:     if outer_v is None:
128:         outer_v = []
129: 
130:     axpy, dot, scal = None, None, None
131:     nrm2 = get_blas_funcs('nrm2', [b])
132: 
133:     b_norm = nrm2(b)
134:     if b_norm == 0:
135:         b_norm = 1
136: 
137:     for k_outer in xrange(maxiter):
138:         r_outer = matvec(x) - b
139: 
140:         # -- callback
141:         if callback is not None:
142:             callback(x)
143: 
144:         # -- determine input type routines
145:         if axpy is None:
146:             if np.iscomplexobj(r_outer) and not np.iscomplexobj(x):
147:                 x = x.astype(r_outer.dtype)
148:             axpy, dot, scal, nrm2 = get_blas_funcs(['axpy', 'dot', 'scal', 'nrm2'],
149:                                                    (x, r_outer))
150:             trtrs = get_lapack_funcs('trtrs', (x, r_outer))
151: 
152:         # -- check stopping condition
153:         r_norm = nrm2(r_outer)
154:         if r_norm <= tol * b_norm or r_norm <= tol:
155:             break
156: 
157:         # -- inner LGMRES iteration
158:         v0 = -psolve(r_outer)
159:         inner_res_0 = nrm2(v0)
160: 
161:         if inner_res_0 == 0:
162:             rnorm = nrm2(r_outer)
163:             raise RuntimeError("Preconditioner returned a zero vector; "
164:                                "|v| ~ %.1g, |M v| = 0" % rnorm)
165: 
166:         v0 = scal(1.0/inner_res_0, v0)
167: 
168:         try:
169:             Q, R, B, vs, zs, y = _fgmres(matvec,
170:                                          v0,
171:                                          inner_m,
172:                                          lpsolve=psolve,
173:                                          atol=tol*b_norm/r_norm,
174:                                          outer_v=outer_v,
175:                                          prepend_outer_v=prepend_outer_v)
176:             y *= inner_res_0
177:             if not np.isfinite(y).all():
178:                 # Overflow etc. in computation. There's no way to
179:                 # recover from this, so we have to bail out.
180:                 raise LinAlgError()
181:         except LinAlgError:
182:             # Floating point over/underflow, non-finite result from
183:             # matmul etc. -- report failure.
184:             return postprocess(x), k_outer + 1
185: 
186:         # -- GMRES terminated: eval solution
187:         dx = zs[0]*y[0]
188:         for w, yc in zip(zs[1:], y[1:]):
189:             dx = axpy(w, dx, dx.shape[0], yc)  # dx += w*yc
190: 
191:         # -- Store LGMRES augmentation vectors
192:         nx = nrm2(dx)
193:         if nx > 0:
194:             if store_outer_Av:
195:                 q = Q.dot(R.dot(y))
196:                 ax = vs[0]*q[0]
197:                 for v, qc in zip(vs[1:], q[1:]):
198:                     ax = axpy(v, ax, ax.shape[0], qc)
199:                 outer_v.append((dx/nx, ax/nx))
200:             else:
201:                 outer_v.append((dx/nx, None))
202: 
203:         # -- Retain only a finite number of augmentation vectors
204:         while len(outer_v) > outer_k:
205:             del outer_v[0]
206: 
207:         # -- Apply step
208:         x += dx
209:     else:
210:         # didn't converge ...
211:         return postprocess(x), maxiter
212: 
213:     return postprocess(x), 0
214: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import numpy' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/')
import_411036 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy')

if (type(import_411036) is not StypyTypeError):

    if (import_411036 != 'pyd_module'):
        __import__(import_411036)
        sys_modules_411037 = sys.modules[import_411036]
        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'np', sys_modules_411037.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy', import_411036)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from numpy.linalg import LinAlgError' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/')
import_411038 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.linalg')

if (type(import_411038) is not StypyTypeError):

    if (import_411038 != 'pyd_module'):
        __import__(import_411038)
        sys_modules_411039 = sys.modules[import_411038]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.linalg', sys_modules_411039.module_type_store, module_type_store, ['LinAlgError'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_411039, sys_modules_411039.module_type_store, module_type_store)
    else:
        from numpy.linalg import LinAlgError

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.linalg', None, module_type_store, ['LinAlgError'], [LinAlgError])

else:
    # Assigning a type to the variable 'numpy.linalg' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.linalg', import_411038)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from scipy._lib.six import xrange' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/')
import_411040 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy._lib.six')

if (type(import_411040) is not StypyTypeError):

    if (import_411040 != 'pyd_module'):
        __import__(import_411040)
        sys_modules_411041 = sys.modules[import_411040]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy._lib.six', sys_modules_411041.module_type_store, module_type_store, ['xrange'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_411041, sys_modules_411041.module_type_store, module_type_store)
    else:
        from scipy._lib.six import xrange

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy._lib.six', None, module_type_store, ['xrange'], [xrange])

else:
    # Assigning a type to the variable 'scipy._lib.six' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy._lib.six', import_411040)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from scipy.linalg import get_blas_funcs, get_lapack_funcs' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/')
import_411042 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.linalg')

if (type(import_411042) is not StypyTypeError):

    if (import_411042 != 'pyd_module'):
        __import__(import_411042)
        sys_modules_411043 = sys.modules[import_411042]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.linalg', sys_modules_411043.module_type_store, module_type_store, ['get_blas_funcs', 'get_lapack_funcs'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_411043, sys_modules_411043.module_type_store, module_type_store)
    else:
        from scipy.linalg import get_blas_funcs, get_lapack_funcs

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.linalg', None, module_type_store, ['get_blas_funcs', 'get_lapack_funcs'], [get_blas_funcs, get_lapack_funcs])

else:
    # Assigning a type to the variable 'scipy.linalg' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.linalg', import_411042)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from scipy.sparse.linalg.isolve.utils import make_system' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/')
import_411044 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.sparse.linalg.isolve.utils')

if (type(import_411044) is not StypyTypeError):

    if (import_411044 != 'pyd_module'):
        __import__(import_411044)
        sys_modules_411045 = sys.modules[import_411044]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.sparse.linalg.isolve.utils', sys_modules_411045.module_type_store, module_type_store, ['make_system'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_411045, sys_modules_411045.module_type_store, module_type_store)
    else:
        from scipy.sparse.linalg.isolve.utils import make_system

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.sparse.linalg.isolve.utils', None, module_type_store, ['make_system'], [make_system])

else:
    # Assigning a type to the variable 'scipy.sparse.linalg.isolve.utils' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.sparse.linalg.isolve.utils', import_411044)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from scipy.sparse.linalg.isolve._gcrotmk import _fgmres' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/')
import_411046 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.sparse.linalg.isolve._gcrotmk')

if (type(import_411046) is not StypyTypeError):

    if (import_411046 != 'pyd_module'):
        __import__(import_411046)
        sys_modules_411047 = sys.modules[import_411046]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.sparse.linalg.isolve._gcrotmk', sys_modules_411047.module_type_store, module_type_store, ['_fgmres'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_411047, sys_modules_411047.module_type_store, module_type_store)
    else:
        from scipy.sparse.linalg.isolve._gcrotmk import _fgmres

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.sparse.linalg.isolve._gcrotmk', None, module_type_store, ['_fgmres'], [_fgmres])

else:
    # Assigning a type to the variable 'scipy.sparse.linalg.isolve._gcrotmk' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.sparse.linalg.isolve._gcrotmk', import_411046)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/')


# Assigning a List to a Name (line 14):

# Assigning a List to a Name (line 14):
__all__ = ['lgmres']
module_type_store.set_exportable_members(['lgmres'])

# Obtaining an instance of the builtin type 'list' (line 14)
list_411048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 14)
# Adding element type (line 14)
str_411049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 11), 'str', 'lgmres')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 10), list_411048, str_411049)

# Assigning a type to the variable '__all__' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), '__all__', list_411048)

@norecursion
def lgmres(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 17)
    None_411050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 20), 'None')
    float_411051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 30), 'float')
    int_411052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 44), 'int')
    # Getting the type of 'None' (line 17)
    None_411053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 52), 'None')
    # Getting the type of 'None' (line 17)
    None_411054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 67), 'None')
    int_411055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 19), 'int')
    int_411056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 31), 'int')
    # Getting the type of 'None' (line 18)
    None_411057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 42), 'None')
    # Getting the type of 'True' (line 18)
    True_411058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 63), 'True')
    # Getting the type of 'False' (line 19)
    False_411059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 27), 'False')
    defaults = [None_411050, float_411051, int_411052, None_411053, None_411054, int_411055, int_411056, None_411057, True_411058, False_411059]
    # Create a new context for function 'lgmres'
    module_type_store = module_type_store.open_function_context('lgmres', 17, 0, False)
    
    # Passed parameters checking function
    lgmres.stypy_localization = localization
    lgmres.stypy_type_of_self = None
    lgmres.stypy_type_store = module_type_store
    lgmres.stypy_function_name = 'lgmres'
    lgmres.stypy_param_names_list = ['A', 'b', 'x0', 'tol', 'maxiter', 'M', 'callback', 'inner_m', 'outer_k', 'outer_v', 'store_outer_Av', 'prepend_outer_v']
    lgmres.stypy_varargs_param_name = None
    lgmres.stypy_kwargs_param_name = None
    lgmres.stypy_call_defaults = defaults
    lgmres.stypy_call_varargs = varargs
    lgmres.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'lgmres', ['A', 'b', 'x0', 'tol', 'maxiter', 'M', 'callback', 'inner_m', 'outer_k', 'outer_v', 'store_outer_Av', 'prepend_outer_v'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'lgmres', localization, ['A', 'b', 'x0', 'tol', 'maxiter', 'M', 'callback', 'inner_m', 'outer_k', 'outer_v', 'store_outer_Av', 'prepend_outer_v'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'lgmres(...)' code ##################

    str_411060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, (-1)), 'str', '\n    Solve a matrix equation using the LGMRES algorithm.\n\n    The LGMRES algorithm [1]_ [2]_ is designed to avoid some problems\n    in the convergence in restarted GMRES, and often converges in fewer\n    iterations.\n\n    Parameters\n    ----------\n    A : {sparse matrix, dense matrix, LinearOperator}\n        The real or complex N-by-N matrix of the linear system.\n    b : {array, matrix}\n        Right hand side of the linear system. Has shape (N,) or (N,1).\n    x0  : {array, matrix}\n        Starting guess for the solution.\n    tol : float, optional\n        Tolerance to achieve. The algorithm terminates when either the relative\n        or the absolute residual is below `tol`.\n    maxiter : int, optional\n        Maximum number of iterations.  Iteration will stop after maxiter\n        steps even if the specified tolerance has not been achieved.\n    M : {sparse matrix, dense matrix, LinearOperator}, optional\n        Preconditioner for A.  The preconditioner should approximate the\n        inverse of A.  Effective preconditioning dramatically improves the\n        rate of convergence, which implies that fewer iterations are needed\n        to reach a given error tolerance.\n    callback : function, optional\n        User-supplied function to call after each iteration.  It is called\n        as callback(xk), where xk is the current solution vector.\n    inner_m : int, optional\n        Number of inner GMRES iterations per each outer iteration.\n    outer_k : int, optional\n        Number of vectors to carry between inner GMRES iterations.\n        According to [1]_, good values are in the range of 1...3.\n        However, note that if you want to use the additional vectors to\n        accelerate solving multiple similar problems, larger values may\n        be beneficial.\n    outer_v : list of tuples, optional\n        List containing tuples ``(v, Av)`` of vectors and corresponding\n        matrix-vector products, used to augment the Krylov subspace, and\n        carried between inner GMRES iterations. The element ``Av`` can\n        be `None` if the matrix-vector product should be re-evaluated.\n        This parameter is modified in-place by `lgmres`, and can be used\n        to pass "guess" vectors in and out of the algorithm when solving\n        similar problems.\n    store_outer_Av : bool, optional\n        Whether LGMRES should store also A*v in addition to vectors `v`\n        in the `outer_v` list. Default is True.\n    prepend_outer_v : bool, optional \n        Whether to put outer_v augmentation vectors before Krylov iterates.\n        In standard LGMRES, prepend_outer_v=False.\n\n    Returns\n    -------\n    x : array or matrix\n        The converged solution.\n    info : int\n        Provides convergence information:\n\n            - 0  : successful exit\n            - >0 : convergence to tolerance not achieved, number of iterations\n            - <0 : illegal input or breakdown\n\n    Notes\n    -----\n    The LGMRES algorithm [1]_ [2]_ is designed to avoid the\n    slowing of convergence in restarted GMRES, due to alternating\n    residual vectors. Typically, it often outperforms GMRES(m) of\n    comparable memory requirements by some measure, or at least is not\n    much worse.\n\n    Another advantage in this algorithm is that you can supply it with\n    \'guess\' vectors in the `outer_v` argument that augment the Krylov\n    subspace. If the solution lies close to the span of these vectors,\n    the algorithm converges faster. This can be useful if several very\n    similar matrices need to be inverted one after another, such as in\n    Newton-Krylov iteration where the Jacobian matrix often changes\n    little in the nonlinear steps.\n\n    References\n    ----------\n    .. [1] A.H. Baker and E.R. Jessup and T. Manteuffel, "A Technique for\n             Accelerating the Convergence of Restarted GMRES", SIAM J. Matrix\n             Anal. Appl. 26, 962 (2005).\n    .. [2] A.H. Baker, "On Improving the Performance of the Linear Solver\n             restarted GMRES", PhD thesis, University of Colorado (2003).\n\n    Examples\n    --------\n    >>> from scipy.sparse import csc_matrix\n    >>> from scipy.sparse.linalg import lgmres\n    >>> A = csc_matrix([[3, 2, 0], [1, -1, 0], [0, 5, 1]], dtype=float)\n    >>> b = np.array([2, 4, -1], dtype=float)\n    >>> x, exitCode = lgmres(A, b)\n    >>> print(exitCode)            # 0 indicates successful convergence\n    0\n    >>> np.allclose(A.dot(x), b)\n    True\n    ')
    
    # Assigning a Call to a Tuple (line 119):
    
    # Assigning a Subscript to a Name (line 119):
    
    # Obtaining the type of the subscript
    int_411061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 4), 'int')
    
    # Call to make_system(...): (line 119)
    # Processing the call arguments (line 119)
    # Getting the type of 'A' (line 119)
    A_411063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 38), 'A', False)
    # Getting the type of 'M' (line 119)
    M_411064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 40), 'M', False)
    # Getting the type of 'x0' (line 119)
    x0_411065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 42), 'x0', False)
    # Getting the type of 'b' (line 119)
    b_411066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 45), 'b', False)
    # Processing the call keyword arguments (line 119)
    kwargs_411067 = {}
    # Getting the type of 'make_system' (line 119)
    make_system_411062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 26), 'make_system', False)
    # Calling make_system(args, kwargs) (line 119)
    make_system_call_result_411068 = invoke(stypy.reporting.localization.Localization(__file__, 119, 26), make_system_411062, *[A_411063, M_411064, x0_411065, b_411066], **kwargs_411067)
    
    # Obtaining the member '__getitem__' of a type (line 119)
    getitem___411069 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 4), make_system_call_result_411068, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 119)
    subscript_call_result_411070 = invoke(stypy.reporting.localization.Localization(__file__, 119, 4), getitem___411069, int_411061)
    
    # Assigning a type to the variable 'tuple_var_assignment_411018' (line 119)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 4), 'tuple_var_assignment_411018', subscript_call_result_411070)
    
    # Assigning a Subscript to a Name (line 119):
    
    # Obtaining the type of the subscript
    int_411071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 4), 'int')
    
    # Call to make_system(...): (line 119)
    # Processing the call arguments (line 119)
    # Getting the type of 'A' (line 119)
    A_411073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 38), 'A', False)
    # Getting the type of 'M' (line 119)
    M_411074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 40), 'M', False)
    # Getting the type of 'x0' (line 119)
    x0_411075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 42), 'x0', False)
    # Getting the type of 'b' (line 119)
    b_411076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 45), 'b', False)
    # Processing the call keyword arguments (line 119)
    kwargs_411077 = {}
    # Getting the type of 'make_system' (line 119)
    make_system_411072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 26), 'make_system', False)
    # Calling make_system(args, kwargs) (line 119)
    make_system_call_result_411078 = invoke(stypy.reporting.localization.Localization(__file__, 119, 26), make_system_411072, *[A_411073, M_411074, x0_411075, b_411076], **kwargs_411077)
    
    # Obtaining the member '__getitem__' of a type (line 119)
    getitem___411079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 4), make_system_call_result_411078, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 119)
    subscript_call_result_411080 = invoke(stypy.reporting.localization.Localization(__file__, 119, 4), getitem___411079, int_411071)
    
    # Assigning a type to the variable 'tuple_var_assignment_411019' (line 119)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 4), 'tuple_var_assignment_411019', subscript_call_result_411080)
    
    # Assigning a Subscript to a Name (line 119):
    
    # Obtaining the type of the subscript
    int_411081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 4), 'int')
    
    # Call to make_system(...): (line 119)
    # Processing the call arguments (line 119)
    # Getting the type of 'A' (line 119)
    A_411083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 38), 'A', False)
    # Getting the type of 'M' (line 119)
    M_411084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 40), 'M', False)
    # Getting the type of 'x0' (line 119)
    x0_411085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 42), 'x0', False)
    # Getting the type of 'b' (line 119)
    b_411086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 45), 'b', False)
    # Processing the call keyword arguments (line 119)
    kwargs_411087 = {}
    # Getting the type of 'make_system' (line 119)
    make_system_411082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 26), 'make_system', False)
    # Calling make_system(args, kwargs) (line 119)
    make_system_call_result_411088 = invoke(stypy.reporting.localization.Localization(__file__, 119, 26), make_system_411082, *[A_411083, M_411084, x0_411085, b_411086], **kwargs_411087)
    
    # Obtaining the member '__getitem__' of a type (line 119)
    getitem___411089 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 4), make_system_call_result_411088, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 119)
    subscript_call_result_411090 = invoke(stypy.reporting.localization.Localization(__file__, 119, 4), getitem___411089, int_411081)
    
    # Assigning a type to the variable 'tuple_var_assignment_411020' (line 119)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 4), 'tuple_var_assignment_411020', subscript_call_result_411090)
    
    # Assigning a Subscript to a Name (line 119):
    
    # Obtaining the type of the subscript
    int_411091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 4), 'int')
    
    # Call to make_system(...): (line 119)
    # Processing the call arguments (line 119)
    # Getting the type of 'A' (line 119)
    A_411093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 38), 'A', False)
    # Getting the type of 'M' (line 119)
    M_411094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 40), 'M', False)
    # Getting the type of 'x0' (line 119)
    x0_411095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 42), 'x0', False)
    # Getting the type of 'b' (line 119)
    b_411096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 45), 'b', False)
    # Processing the call keyword arguments (line 119)
    kwargs_411097 = {}
    # Getting the type of 'make_system' (line 119)
    make_system_411092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 26), 'make_system', False)
    # Calling make_system(args, kwargs) (line 119)
    make_system_call_result_411098 = invoke(stypy.reporting.localization.Localization(__file__, 119, 26), make_system_411092, *[A_411093, M_411094, x0_411095, b_411096], **kwargs_411097)
    
    # Obtaining the member '__getitem__' of a type (line 119)
    getitem___411099 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 4), make_system_call_result_411098, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 119)
    subscript_call_result_411100 = invoke(stypy.reporting.localization.Localization(__file__, 119, 4), getitem___411099, int_411091)
    
    # Assigning a type to the variable 'tuple_var_assignment_411021' (line 119)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 4), 'tuple_var_assignment_411021', subscript_call_result_411100)
    
    # Assigning a Subscript to a Name (line 119):
    
    # Obtaining the type of the subscript
    int_411101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 4), 'int')
    
    # Call to make_system(...): (line 119)
    # Processing the call arguments (line 119)
    # Getting the type of 'A' (line 119)
    A_411103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 38), 'A', False)
    # Getting the type of 'M' (line 119)
    M_411104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 40), 'M', False)
    # Getting the type of 'x0' (line 119)
    x0_411105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 42), 'x0', False)
    # Getting the type of 'b' (line 119)
    b_411106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 45), 'b', False)
    # Processing the call keyword arguments (line 119)
    kwargs_411107 = {}
    # Getting the type of 'make_system' (line 119)
    make_system_411102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 26), 'make_system', False)
    # Calling make_system(args, kwargs) (line 119)
    make_system_call_result_411108 = invoke(stypy.reporting.localization.Localization(__file__, 119, 26), make_system_411102, *[A_411103, M_411104, x0_411105, b_411106], **kwargs_411107)
    
    # Obtaining the member '__getitem__' of a type (line 119)
    getitem___411109 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 4), make_system_call_result_411108, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 119)
    subscript_call_result_411110 = invoke(stypy.reporting.localization.Localization(__file__, 119, 4), getitem___411109, int_411101)
    
    # Assigning a type to the variable 'tuple_var_assignment_411022' (line 119)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 4), 'tuple_var_assignment_411022', subscript_call_result_411110)
    
    # Assigning a Name to a Name (line 119):
    # Getting the type of 'tuple_var_assignment_411018' (line 119)
    tuple_var_assignment_411018_411111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 4), 'tuple_var_assignment_411018')
    # Assigning a type to the variable 'A' (line 119)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 4), 'A', tuple_var_assignment_411018_411111)
    
    # Assigning a Name to a Name (line 119):
    # Getting the type of 'tuple_var_assignment_411019' (line 119)
    tuple_var_assignment_411019_411112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 4), 'tuple_var_assignment_411019')
    # Assigning a type to the variable 'M' (line 119)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 6), 'M', tuple_var_assignment_411019_411112)
    
    # Assigning a Name to a Name (line 119):
    # Getting the type of 'tuple_var_assignment_411020' (line 119)
    tuple_var_assignment_411020_411113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 4), 'tuple_var_assignment_411020')
    # Assigning a type to the variable 'x' (line 119)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'x', tuple_var_assignment_411020_411113)
    
    # Assigning a Name to a Name (line 119):
    # Getting the type of 'tuple_var_assignment_411021' (line 119)
    tuple_var_assignment_411021_411114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 4), 'tuple_var_assignment_411021')
    # Assigning a type to the variable 'b' (line 119)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 10), 'b', tuple_var_assignment_411021_411114)
    
    # Assigning a Name to a Name (line 119):
    # Getting the type of 'tuple_var_assignment_411022' (line 119)
    tuple_var_assignment_411022_411115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 4), 'tuple_var_assignment_411022')
    # Assigning a type to the variable 'postprocess' (line 119)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 12), 'postprocess', tuple_var_assignment_411022_411115)
    
    
    
    # Call to all(...): (line 121)
    # Processing the call keyword arguments (line 121)
    kwargs_411122 = {}
    
    # Call to isfinite(...): (line 121)
    # Processing the call arguments (line 121)
    # Getting the type of 'b' (line 121)
    b_411118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 23), 'b', False)
    # Processing the call keyword arguments (line 121)
    kwargs_411119 = {}
    # Getting the type of 'np' (line 121)
    np_411116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 11), 'np', False)
    # Obtaining the member 'isfinite' of a type (line 121)
    isfinite_411117 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 11), np_411116, 'isfinite')
    # Calling isfinite(args, kwargs) (line 121)
    isfinite_call_result_411120 = invoke(stypy.reporting.localization.Localization(__file__, 121, 11), isfinite_411117, *[b_411118], **kwargs_411119)
    
    # Obtaining the member 'all' of a type (line 121)
    all_411121 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 11), isfinite_call_result_411120, 'all')
    # Calling all(args, kwargs) (line 121)
    all_call_result_411123 = invoke(stypy.reporting.localization.Localization(__file__, 121, 11), all_411121, *[], **kwargs_411122)
    
    # Applying the 'not' unary operator (line 121)
    result_not__411124 = python_operator(stypy.reporting.localization.Localization(__file__, 121, 7), 'not', all_call_result_411123)
    
    # Testing the type of an if condition (line 121)
    if_condition_411125 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 121, 4), result_not__411124)
    # Assigning a type to the variable 'if_condition_411125' (line 121)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 4), 'if_condition_411125', if_condition_411125)
    # SSA begins for if statement (line 121)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 122)
    # Processing the call arguments (line 122)
    str_411127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 25), 'str', 'RHS must contain only finite numbers')
    # Processing the call keyword arguments (line 122)
    kwargs_411128 = {}
    # Getting the type of 'ValueError' (line 122)
    ValueError_411126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 122)
    ValueError_call_result_411129 = invoke(stypy.reporting.localization.Localization(__file__, 122, 14), ValueError_411126, *[str_411127], **kwargs_411128)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 122, 8), ValueError_call_result_411129, 'raise parameter', BaseException)
    # SSA join for if statement (line 121)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Attribute to a Name (line 124):
    
    # Assigning a Attribute to a Name (line 124):
    # Getting the type of 'A' (line 124)
    A_411130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 13), 'A')
    # Obtaining the member 'matvec' of a type (line 124)
    matvec_411131 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 13), A_411130, 'matvec')
    # Assigning a type to the variable 'matvec' (line 124)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 4), 'matvec', matvec_411131)
    
    # Assigning a Attribute to a Name (line 125):
    
    # Assigning a Attribute to a Name (line 125):
    # Getting the type of 'M' (line 125)
    M_411132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 13), 'M')
    # Obtaining the member 'matvec' of a type (line 125)
    matvec_411133 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 13), M_411132, 'matvec')
    # Assigning a type to the variable 'psolve' (line 125)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 4), 'psolve', matvec_411133)
    
    # Type idiom detected: calculating its left and rigth part (line 127)
    # Getting the type of 'outer_v' (line 127)
    outer_v_411134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 7), 'outer_v')
    # Getting the type of 'None' (line 127)
    None_411135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 18), 'None')
    
    (may_be_411136, more_types_in_union_411137) = may_be_none(outer_v_411134, None_411135)

    if may_be_411136:

        if more_types_in_union_411137:
            # Runtime conditional SSA (line 127)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a List to a Name (line 128):
        
        # Assigning a List to a Name (line 128):
        
        # Obtaining an instance of the builtin type 'list' (line 128)
        list_411138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 128)
        
        # Assigning a type to the variable 'outer_v' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'outer_v', list_411138)

        if more_types_in_union_411137:
            # SSA join for if statement (line 127)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Tuple to a Tuple (line 130):
    
    # Assigning a Name to a Name (line 130):
    # Getting the type of 'None' (line 130)
    None_411139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 22), 'None')
    # Assigning a type to the variable 'tuple_assignment_411023' (line 130)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 4), 'tuple_assignment_411023', None_411139)
    
    # Assigning a Name to a Name (line 130):
    # Getting the type of 'None' (line 130)
    None_411140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 28), 'None')
    # Assigning a type to the variable 'tuple_assignment_411024' (line 130)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 4), 'tuple_assignment_411024', None_411140)
    
    # Assigning a Name to a Name (line 130):
    # Getting the type of 'None' (line 130)
    None_411141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 34), 'None')
    # Assigning a type to the variable 'tuple_assignment_411025' (line 130)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 4), 'tuple_assignment_411025', None_411141)
    
    # Assigning a Name to a Name (line 130):
    # Getting the type of 'tuple_assignment_411023' (line 130)
    tuple_assignment_411023_411142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 4), 'tuple_assignment_411023')
    # Assigning a type to the variable 'axpy' (line 130)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 4), 'axpy', tuple_assignment_411023_411142)
    
    # Assigning a Name to a Name (line 130):
    # Getting the type of 'tuple_assignment_411024' (line 130)
    tuple_assignment_411024_411143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 4), 'tuple_assignment_411024')
    # Assigning a type to the variable 'dot' (line 130)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 10), 'dot', tuple_assignment_411024_411143)
    
    # Assigning a Name to a Name (line 130):
    # Getting the type of 'tuple_assignment_411025' (line 130)
    tuple_assignment_411025_411144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 4), 'tuple_assignment_411025')
    # Assigning a type to the variable 'scal' (line 130)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 15), 'scal', tuple_assignment_411025_411144)
    
    # Assigning a Call to a Name (line 131):
    
    # Assigning a Call to a Name (line 131):
    
    # Call to get_blas_funcs(...): (line 131)
    # Processing the call arguments (line 131)
    str_411146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 26), 'str', 'nrm2')
    
    # Obtaining an instance of the builtin type 'list' (line 131)
    list_411147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 34), 'list')
    # Adding type elements to the builtin type 'list' instance (line 131)
    # Adding element type (line 131)
    # Getting the type of 'b' (line 131)
    b_411148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 35), 'b', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 34), list_411147, b_411148)
    
    # Processing the call keyword arguments (line 131)
    kwargs_411149 = {}
    # Getting the type of 'get_blas_funcs' (line 131)
    get_blas_funcs_411145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 11), 'get_blas_funcs', False)
    # Calling get_blas_funcs(args, kwargs) (line 131)
    get_blas_funcs_call_result_411150 = invoke(stypy.reporting.localization.Localization(__file__, 131, 11), get_blas_funcs_411145, *[str_411146, list_411147], **kwargs_411149)
    
    # Assigning a type to the variable 'nrm2' (line 131)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 4), 'nrm2', get_blas_funcs_call_result_411150)
    
    # Assigning a Call to a Name (line 133):
    
    # Assigning a Call to a Name (line 133):
    
    # Call to nrm2(...): (line 133)
    # Processing the call arguments (line 133)
    # Getting the type of 'b' (line 133)
    b_411152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 18), 'b', False)
    # Processing the call keyword arguments (line 133)
    kwargs_411153 = {}
    # Getting the type of 'nrm2' (line 133)
    nrm2_411151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 13), 'nrm2', False)
    # Calling nrm2(args, kwargs) (line 133)
    nrm2_call_result_411154 = invoke(stypy.reporting.localization.Localization(__file__, 133, 13), nrm2_411151, *[b_411152], **kwargs_411153)
    
    # Assigning a type to the variable 'b_norm' (line 133)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 4), 'b_norm', nrm2_call_result_411154)
    
    
    # Getting the type of 'b_norm' (line 134)
    b_norm_411155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 7), 'b_norm')
    int_411156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 17), 'int')
    # Applying the binary operator '==' (line 134)
    result_eq_411157 = python_operator(stypy.reporting.localization.Localization(__file__, 134, 7), '==', b_norm_411155, int_411156)
    
    # Testing the type of an if condition (line 134)
    if_condition_411158 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 134, 4), result_eq_411157)
    # Assigning a type to the variable 'if_condition_411158' (line 134)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 4), 'if_condition_411158', if_condition_411158)
    # SSA begins for if statement (line 134)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 135):
    
    # Assigning a Num to a Name (line 135):
    int_411159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 17), 'int')
    # Assigning a type to the variable 'b_norm' (line 135)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'b_norm', int_411159)
    # SSA join for if statement (line 134)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to xrange(...): (line 137)
    # Processing the call arguments (line 137)
    # Getting the type of 'maxiter' (line 137)
    maxiter_411161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 26), 'maxiter', False)
    # Processing the call keyword arguments (line 137)
    kwargs_411162 = {}
    # Getting the type of 'xrange' (line 137)
    xrange_411160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 19), 'xrange', False)
    # Calling xrange(args, kwargs) (line 137)
    xrange_call_result_411163 = invoke(stypy.reporting.localization.Localization(__file__, 137, 19), xrange_411160, *[maxiter_411161], **kwargs_411162)
    
    # Testing the type of a for loop iterable (line 137)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 137, 4), xrange_call_result_411163)
    # Getting the type of the for loop variable (line 137)
    for_loop_var_411164 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 137, 4), xrange_call_result_411163)
    # Assigning a type to the variable 'k_outer' (line 137)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 4), 'k_outer', for_loop_var_411164)
    # SSA begins for a for statement (line 137)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a BinOp to a Name (line 138):
    
    # Assigning a BinOp to a Name (line 138):
    
    # Call to matvec(...): (line 138)
    # Processing the call arguments (line 138)
    # Getting the type of 'x' (line 138)
    x_411166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 25), 'x', False)
    # Processing the call keyword arguments (line 138)
    kwargs_411167 = {}
    # Getting the type of 'matvec' (line 138)
    matvec_411165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 18), 'matvec', False)
    # Calling matvec(args, kwargs) (line 138)
    matvec_call_result_411168 = invoke(stypy.reporting.localization.Localization(__file__, 138, 18), matvec_411165, *[x_411166], **kwargs_411167)
    
    # Getting the type of 'b' (line 138)
    b_411169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 30), 'b')
    # Applying the binary operator '-' (line 138)
    result_sub_411170 = python_operator(stypy.reporting.localization.Localization(__file__, 138, 18), '-', matvec_call_result_411168, b_411169)
    
    # Assigning a type to the variable 'r_outer' (line 138)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 8), 'r_outer', result_sub_411170)
    
    # Type idiom detected: calculating its left and rigth part (line 141)
    # Getting the type of 'callback' (line 141)
    callback_411171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'callback')
    # Getting the type of 'None' (line 141)
    None_411172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 27), 'None')
    
    (may_be_411173, more_types_in_union_411174) = may_not_be_none(callback_411171, None_411172)

    if may_be_411173:

        if more_types_in_union_411174:
            # Runtime conditional SSA (line 141)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to callback(...): (line 142)
        # Processing the call arguments (line 142)
        # Getting the type of 'x' (line 142)
        x_411176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 21), 'x', False)
        # Processing the call keyword arguments (line 142)
        kwargs_411177 = {}
        # Getting the type of 'callback' (line 142)
        callback_411175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 12), 'callback', False)
        # Calling callback(args, kwargs) (line 142)
        callback_call_result_411178 = invoke(stypy.reporting.localization.Localization(__file__, 142, 12), callback_411175, *[x_411176], **kwargs_411177)
        

        if more_types_in_union_411174:
            # SSA join for if statement (line 141)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 145)
    # Getting the type of 'axpy' (line 145)
    axpy_411179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 11), 'axpy')
    # Getting the type of 'None' (line 145)
    None_411180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 19), 'None')
    
    (may_be_411181, more_types_in_union_411182) = may_be_none(axpy_411179, None_411180)

    if may_be_411181:

        if more_types_in_union_411182:
            # Runtime conditional SSA (line 145)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        
        # Evaluating a boolean operation
        
        # Call to iscomplexobj(...): (line 146)
        # Processing the call arguments (line 146)
        # Getting the type of 'r_outer' (line 146)
        r_outer_411185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 31), 'r_outer', False)
        # Processing the call keyword arguments (line 146)
        kwargs_411186 = {}
        # Getting the type of 'np' (line 146)
        np_411183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 15), 'np', False)
        # Obtaining the member 'iscomplexobj' of a type (line 146)
        iscomplexobj_411184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 15), np_411183, 'iscomplexobj')
        # Calling iscomplexobj(args, kwargs) (line 146)
        iscomplexobj_call_result_411187 = invoke(stypy.reporting.localization.Localization(__file__, 146, 15), iscomplexobj_411184, *[r_outer_411185], **kwargs_411186)
        
        
        
        # Call to iscomplexobj(...): (line 146)
        # Processing the call arguments (line 146)
        # Getting the type of 'x' (line 146)
        x_411190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 64), 'x', False)
        # Processing the call keyword arguments (line 146)
        kwargs_411191 = {}
        # Getting the type of 'np' (line 146)
        np_411188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 48), 'np', False)
        # Obtaining the member 'iscomplexobj' of a type (line 146)
        iscomplexobj_411189 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 48), np_411188, 'iscomplexobj')
        # Calling iscomplexobj(args, kwargs) (line 146)
        iscomplexobj_call_result_411192 = invoke(stypy.reporting.localization.Localization(__file__, 146, 48), iscomplexobj_411189, *[x_411190], **kwargs_411191)
        
        # Applying the 'not' unary operator (line 146)
        result_not__411193 = python_operator(stypy.reporting.localization.Localization(__file__, 146, 44), 'not', iscomplexobj_call_result_411192)
        
        # Applying the binary operator 'and' (line 146)
        result_and_keyword_411194 = python_operator(stypy.reporting.localization.Localization(__file__, 146, 15), 'and', iscomplexobj_call_result_411187, result_not__411193)
        
        # Testing the type of an if condition (line 146)
        if_condition_411195 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 146, 12), result_and_keyword_411194)
        # Assigning a type to the variable 'if_condition_411195' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 12), 'if_condition_411195', if_condition_411195)
        # SSA begins for if statement (line 146)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 147):
        
        # Assigning a Call to a Name (line 147):
        
        # Call to astype(...): (line 147)
        # Processing the call arguments (line 147)
        # Getting the type of 'r_outer' (line 147)
        r_outer_411198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 29), 'r_outer', False)
        # Obtaining the member 'dtype' of a type (line 147)
        dtype_411199 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 29), r_outer_411198, 'dtype')
        # Processing the call keyword arguments (line 147)
        kwargs_411200 = {}
        # Getting the type of 'x' (line 147)
        x_411196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 20), 'x', False)
        # Obtaining the member 'astype' of a type (line 147)
        astype_411197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 20), x_411196, 'astype')
        # Calling astype(args, kwargs) (line 147)
        astype_call_result_411201 = invoke(stypy.reporting.localization.Localization(__file__, 147, 20), astype_411197, *[dtype_411199], **kwargs_411200)
        
        # Assigning a type to the variable 'x' (line 147)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 16), 'x', astype_call_result_411201)
        # SSA join for if statement (line 146)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Tuple (line 148):
        
        # Assigning a Subscript to a Name (line 148):
        
        # Obtaining the type of the subscript
        int_411202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 12), 'int')
        
        # Call to get_blas_funcs(...): (line 148)
        # Processing the call arguments (line 148)
        
        # Obtaining an instance of the builtin type 'list' (line 148)
        list_411204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 51), 'list')
        # Adding type elements to the builtin type 'list' instance (line 148)
        # Adding element type (line 148)
        str_411205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 52), 'str', 'axpy')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 148, 51), list_411204, str_411205)
        # Adding element type (line 148)
        str_411206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 60), 'str', 'dot')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 148, 51), list_411204, str_411206)
        # Adding element type (line 148)
        str_411207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 67), 'str', 'scal')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 148, 51), list_411204, str_411207)
        # Adding element type (line 148)
        str_411208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 75), 'str', 'nrm2')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 148, 51), list_411204, str_411208)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 149)
        tuple_411209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 52), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 149)
        # Adding element type (line 149)
        # Getting the type of 'x' (line 149)
        x_411210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 52), 'x', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 149, 52), tuple_411209, x_411210)
        # Adding element type (line 149)
        # Getting the type of 'r_outer' (line 149)
        r_outer_411211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 55), 'r_outer', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 149, 52), tuple_411209, r_outer_411211)
        
        # Processing the call keyword arguments (line 148)
        kwargs_411212 = {}
        # Getting the type of 'get_blas_funcs' (line 148)
        get_blas_funcs_411203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 36), 'get_blas_funcs', False)
        # Calling get_blas_funcs(args, kwargs) (line 148)
        get_blas_funcs_call_result_411213 = invoke(stypy.reporting.localization.Localization(__file__, 148, 36), get_blas_funcs_411203, *[list_411204, tuple_411209], **kwargs_411212)
        
        # Obtaining the member '__getitem__' of a type (line 148)
        getitem___411214 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 12), get_blas_funcs_call_result_411213, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 148)
        subscript_call_result_411215 = invoke(stypy.reporting.localization.Localization(__file__, 148, 12), getitem___411214, int_411202)
        
        # Assigning a type to the variable 'tuple_var_assignment_411026' (line 148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 12), 'tuple_var_assignment_411026', subscript_call_result_411215)
        
        # Assigning a Subscript to a Name (line 148):
        
        # Obtaining the type of the subscript
        int_411216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 12), 'int')
        
        # Call to get_blas_funcs(...): (line 148)
        # Processing the call arguments (line 148)
        
        # Obtaining an instance of the builtin type 'list' (line 148)
        list_411218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 51), 'list')
        # Adding type elements to the builtin type 'list' instance (line 148)
        # Adding element type (line 148)
        str_411219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 52), 'str', 'axpy')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 148, 51), list_411218, str_411219)
        # Adding element type (line 148)
        str_411220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 60), 'str', 'dot')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 148, 51), list_411218, str_411220)
        # Adding element type (line 148)
        str_411221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 67), 'str', 'scal')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 148, 51), list_411218, str_411221)
        # Adding element type (line 148)
        str_411222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 75), 'str', 'nrm2')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 148, 51), list_411218, str_411222)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 149)
        tuple_411223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 52), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 149)
        # Adding element type (line 149)
        # Getting the type of 'x' (line 149)
        x_411224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 52), 'x', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 149, 52), tuple_411223, x_411224)
        # Adding element type (line 149)
        # Getting the type of 'r_outer' (line 149)
        r_outer_411225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 55), 'r_outer', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 149, 52), tuple_411223, r_outer_411225)
        
        # Processing the call keyword arguments (line 148)
        kwargs_411226 = {}
        # Getting the type of 'get_blas_funcs' (line 148)
        get_blas_funcs_411217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 36), 'get_blas_funcs', False)
        # Calling get_blas_funcs(args, kwargs) (line 148)
        get_blas_funcs_call_result_411227 = invoke(stypy.reporting.localization.Localization(__file__, 148, 36), get_blas_funcs_411217, *[list_411218, tuple_411223], **kwargs_411226)
        
        # Obtaining the member '__getitem__' of a type (line 148)
        getitem___411228 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 12), get_blas_funcs_call_result_411227, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 148)
        subscript_call_result_411229 = invoke(stypy.reporting.localization.Localization(__file__, 148, 12), getitem___411228, int_411216)
        
        # Assigning a type to the variable 'tuple_var_assignment_411027' (line 148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 12), 'tuple_var_assignment_411027', subscript_call_result_411229)
        
        # Assigning a Subscript to a Name (line 148):
        
        # Obtaining the type of the subscript
        int_411230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 12), 'int')
        
        # Call to get_blas_funcs(...): (line 148)
        # Processing the call arguments (line 148)
        
        # Obtaining an instance of the builtin type 'list' (line 148)
        list_411232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 51), 'list')
        # Adding type elements to the builtin type 'list' instance (line 148)
        # Adding element type (line 148)
        str_411233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 52), 'str', 'axpy')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 148, 51), list_411232, str_411233)
        # Adding element type (line 148)
        str_411234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 60), 'str', 'dot')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 148, 51), list_411232, str_411234)
        # Adding element type (line 148)
        str_411235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 67), 'str', 'scal')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 148, 51), list_411232, str_411235)
        # Adding element type (line 148)
        str_411236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 75), 'str', 'nrm2')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 148, 51), list_411232, str_411236)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 149)
        tuple_411237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 52), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 149)
        # Adding element type (line 149)
        # Getting the type of 'x' (line 149)
        x_411238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 52), 'x', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 149, 52), tuple_411237, x_411238)
        # Adding element type (line 149)
        # Getting the type of 'r_outer' (line 149)
        r_outer_411239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 55), 'r_outer', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 149, 52), tuple_411237, r_outer_411239)
        
        # Processing the call keyword arguments (line 148)
        kwargs_411240 = {}
        # Getting the type of 'get_blas_funcs' (line 148)
        get_blas_funcs_411231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 36), 'get_blas_funcs', False)
        # Calling get_blas_funcs(args, kwargs) (line 148)
        get_blas_funcs_call_result_411241 = invoke(stypy.reporting.localization.Localization(__file__, 148, 36), get_blas_funcs_411231, *[list_411232, tuple_411237], **kwargs_411240)
        
        # Obtaining the member '__getitem__' of a type (line 148)
        getitem___411242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 12), get_blas_funcs_call_result_411241, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 148)
        subscript_call_result_411243 = invoke(stypy.reporting.localization.Localization(__file__, 148, 12), getitem___411242, int_411230)
        
        # Assigning a type to the variable 'tuple_var_assignment_411028' (line 148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 12), 'tuple_var_assignment_411028', subscript_call_result_411243)
        
        # Assigning a Subscript to a Name (line 148):
        
        # Obtaining the type of the subscript
        int_411244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 12), 'int')
        
        # Call to get_blas_funcs(...): (line 148)
        # Processing the call arguments (line 148)
        
        # Obtaining an instance of the builtin type 'list' (line 148)
        list_411246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 51), 'list')
        # Adding type elements to the builtin type 'list' instance (line 148)
        # Adding element type (line 148)
        str_411247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 52), 'str', 'axpy')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 148, 51), list_411246, str_411247)
        # Adding element type (line 148)
        str_411248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 60), 'str', 'dot')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 148, 51), list_411246, str_411248)
        # Adding element type (line 148)
        str_411249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 67), 'str', 'scal')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 148, 51), list_411246, str_411249)
        # Adding element type (line 148)
        str_411250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 75), 'str', 'nrm2')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 148, 51), list_411246, str_411250)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 149)
        tuple_411251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 52), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 149)
        # Adding element type (line 149)
        # Getting the type of 'x' (line 149)
        x_411252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 52), 'x', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 149, 52), tuple_411251, x_411252)
        # Adding element type (line 149)
        # Getting the type of 'r_outer' (line 149)
        r_outer_411253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 55), 'r_outer', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 149, 52), tuple_411251, r_outer_411253)
        
        # Processing the call keyword arguments (line 148)
        kwargs_411254 = {}
        # Getting the type of 'get_blas_funcs' (line 148)
        get_blas_funcs_411245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 36), 'get_blas_funcs', False)
        # Calling get_blas_funcs(args, kwargs) (line 148)
        get_blas_funcs_call_result_411255 = invoke(stypy.reporting.localization.Localization(__file__, 148, 36), get_blas_funcs_411245, *[list_411246, tuple_411251], **kwargs_411254)
        
        # Obtaining the member '__getitem__' of a type (line 148)
        getitem___411256 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 12), get_blas_funcs_call_result_411255, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 148)
        subscript_call_result_411257 = invoke(stypy.reporting.localization.Localization(__file__, 148, 12), getitem___411256, int_411244)
        
        # Assigning a type to the variable 'tuple_var_assignment_411029' (line 148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 12), 'tuple_var_assignment_411029', subscript_call_result_411257)
        
        # Assigning a Name to a Name (line 148):
        # Getting the type of 'tuple_var_assignment_411026' (line 148)
        tuple_var_assignment_411026_411258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 12), 'tuple_var_assignment_411026')
        # Assigning a type to the variable 'axpy' (line 148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 12), 'axpy', tuple_var_assignment_411026_411258)
        
        # Assigning a Name to a Name (line 148):
        # Getting the type of 'tuple_var_assignment_411027' (line 148)
        tuple_var_assignment_411027_411259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 12), 'tuple_var_assignment_411027')
        # Assigning a type to the variable 'dot' (line 148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 18), 'dot', tuple_var_assignment_411027_411259)
        
        # Assigning a Name to a Name (line 148):
        # Getting the type of 'tuple_var_assignment_411028' (line 148)
        tuple_var_assignment_411028_411260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 12), 'tuple_var_assignment_411028')
        # Assigning a type to the variable 'scal' (line 148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 23), 'scal', tuple_var_assignment_411028_411260)
        
        # Assigning a Name to a Name (line 148):
        # Getting the type of 'tuple_var_assignment_411029' (line 148)
        tuple_var_assignment_411029_411261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 12), 'tuple_var_assignment_411029')
        # Assigning a type to the variable 'nrm2' (line 148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 29), 'nrm2', tuple_var_assignment_411029_411261)
        
        # Assigning a Call to a Name (line 150):
        
        # Assigning a Call to a Name (line 150):
        
        # Call to get_lapack_funcs(...): (line 150)
        # Processing the call arguments (line 150)
        str_411263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 37), 'str', 'trtrs')
        
        # Obtaining an instance of the builtin type 'tuple' (line 150)
        tuple_411264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 47), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 150)
        # Adding element type (line 150)
        # Getting the type of 'x' (line 150)
        x_411265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 47), 'x', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 150, 47), tuple_411264, x_411265)
        # Adding element type (line 150)
        # Getting the type of 'r_outer' (line 150)
        r_outer_411266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 50), 'r_outer', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 150, 47), tuple_411264, r_outer_411266)
        
        # Processing the call keyword arguments (line 150)
        kwargs_411267 = {}
        # Getting the type of 'get_lapack_funcs' (line 150)
        get_lapack_funcs_411262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 20), 'get_lapack_funcs', False)
        # Calling get_lapack_funcs(args, kwargs) (line 150)
        get_lapack_funcs_call_result_411268 = invoke(stypy.reporting.localization.Localization(__file__, 150, 20), get_lapack_funcs_411262, *[str_411263, tuple_411264], **kwargs_411267)
        
        # Assigning a type to the variable 'trtrs' (line 150)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 12), 'trtrs', get_lapack_funcs_call_result_411268)

        if more_types_in_union_411182:
            # SSA join for if statement (line 145)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 153):
    
    # Assigning a Call to a Name (line 153):
    
    # Call to nrm2(...): (line 153)
    # Processing the call arguments (line 153)
    # Getting the type of 'r_outer' (line 153)
    r_outer_411270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 22), 'r_outer', False)
    # Processing the call keyword arguments (line 153)
    kwargs_411271 = {}
    # Getting the type of 'nrm2' (line 153)
    nrm2_411269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 17), 'nrm2', False)
    # Calling nrm2(args, kwargs) (line 153)
    nrm2_call_result_411272 = invoke(stypy.reporting.localization.Localization(__file__, 153, 17), nrm2_411269, *[r_outer_411270], **kwargs_411271)
    
    # Assigning a type to the variable 'r_norm' (line 153)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 8), 'r_norm', nrm2_call_result_411272)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'r_norm' (line 154)
    r_norm_411273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 11), 'r_norm')
    # Getting the type of 'tol' (line 154)
    tol_411274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 21), 'tol')
    # Getting the type of 'b_norm' (line 154)
    b_norm_411275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 27), 'b_norm')
    # Applying the binary operator '*' (line 154)
    result_mul_411276 = python_operator(stypy.reporting.localization.Localization(__file__, 154, 21), '*', tol_411274, b_norm_411275)
    
    # Applying the binary operator '<=' (line 154)
    result_le_411277 = python_operator(stypy.reporting.localization.Localization(__file__, 154, 11), '<=', r_norm_411273, result_mul_411276)
    
    
    # Getting the type of 'r_norm' (line 154)
    r_norm_411278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 37), 'r_norm')
    # Getting the type of 'tol' (line 154)
    tol_411279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 47), 'tol')
    # Applying the binary operator '<=' (line 154)
    result_le_411280 = python_operator(stypy.reporting.localization.Localization(__file__, 154, 37), '<=', r_norm_411278, tol_411279)
    
    # Applying the binary operator 'or' (line 154)
    result_or_keyword_411281 = python_operator(stypy.reporting.localization.Localization(__file__, 154, 11), 'or', result_le_411277, result_le_411280)
    
    # Testing the type of an if condition (line 154)
    if_condition_411282 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 154, 8), result_or_keyword_411281)
    # Assigning a type to the variable 'if_condition_411282' (line 154)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'if_condition_411282', if_condition_411282)
    # SSA begins for if statement (line 154)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 154)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a UnaryOp to a Name (line 158):
    
    # Assigning a UnaryOp to a Name (line 158):
    
    
    # Call to psolve(...): (line 158)
    # Processing the call arguments (line 158)
    # Getting the type of 'r_outer' (line 158)
    r_outer_411284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 21), 'r_outer', False)
    # Processing the call keyword arguments (line 158)
    kwargs_411285 = {}
    # Getting the type of 'psolve' (line 158)
    psolve_411283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 14), 'psolve', False)
    # Calling psolve(args, kwargs) (line 158)
    psolve_call_result_411286 = invoke(stypy.reporting.localization.Localization(__file__, 158, 14), psolve_411283, *[r_outer_411284], **kwargs_411285)
    
    # Applying the 'usub' unary operator (line 158)
    result___neg___411287 = python_operator(stypy.reporting.localization.Localization(__file__, 158, 13), 'usub', psolve_call_result_411286)
    
    # Assigning a type to the variable 'v0' (line 158)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'v0', result___neg___411287)
    
    # Assigning a Call to a Name (line 159):
    
    # Assigning a Call to a Name (line 159):
    
    # Call to nrm2(...): (line 159)
    # Processing the call arguments (line 159)
    # Getting the type of 'v0' (line 159)
    v0_411289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 27), 'v0', False)
    # Processing the call keyword arguments (line 159)
    kwargs_411290 = {}
    # Getting the type of 'nrm2' (line 159)
    nrm2_411288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 22), 'nrm2', False)
    # Calling nrm2(args, kwargs) (line 159)
    nrm2_call_result_411291 = invoke(stypy.reporting.localization.Localization(__file__, 159, 22), nrm2_411288, *[v0_411289], **kwargs_411290)
    
    # Assigning a type to the variable 'inner_res_0' (line 159)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'inner_res_0', nrm2_call_result_411291)
    
    
    # Getting the type of 'inner_res_0' (line 161)
    inner_res_0_411292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 11), 'inner_res_0')
    int_411293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 26), 'int')
    # Applying the binary operator '==' (line 161)
    result_eq_411294 = python_operator(stypy.reporting.localization.Localization(__file__, 161, 11), '==', inner_res_0_411292, int_411293)
    
    # Testing the type of an if condition (line 161)
    if_condition_411295 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 161, 8), result_eq_411294)
    # Assigning a type to the variable 'if_condition_411295' (line 161)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 8), 'if_condition_411295', if_condition_411295)
    # SSA begins for if statement (line 161)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 162):
    
    # Assigning a Call to a Name (line 162):
    
    # Call to nrm2(...): (line 162)
    # Processing the call arguments (line 162)
    # Getting the type of 'r_outer' (line 162)
    r_outer_411297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 25), 'r_outer', False)
    # Processing the call keyword arguments (line 162)
    kwargs_411298 = {}
    # Getting the type of 'nrm2' (line 162)
    nrm2_411296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 20), 'nrm2', False)
    # Calling nrm2(args, kwargs) (line 162)
    nrm2_call_result_411299 = invoke(stypy.reporting.localization.Localization(__file__, 162, 20), nrm2_411296, *[r_outer_411297], **kwargs_411298)
    
    # Assigning a type to the variable 'rnorm' (line 162)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 12), 'rnorm', nrm2_call_result_411299)
    
    # Call to RuntimeError(...): (line 163)
    # Processing the call arguments (line 163)
    str_411301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 31), 'str', 'Preconditioner returned a zero vector; |v| ~ %.1g, |M v| = 0')
    # Getting the type of 'rnorm' (line 164)
    rnorm_411302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 57), 'rnorm', False)
    # Applying the binary operator '%' (line 163)
    result_mod_411303 = python_operator(stypy.reporting.localization.Localization(__file__, 163, 31), '%', str_411301, rnorm_411302)
    
    # Processing the call keyword arguments (line 163)
    kwargs_411304 = {}
    # Getting the type of 'RuntimeError' (line 163)
    RuntimeError_411300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 18), 'RuntimeError', False)
    # Calling RuntimeError(args, kwargs) (line 163)
    RuntimeError_call_result_411305 = invoke(stypy.reporting.localization.Localization(__file__, 163, 18), RuntimeError_411300, *[result_mod_411303], **kwargs_411304)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 163, 12), RuntimeError_call_result_411305, 'raise parameter', BaseException)
    # SSA join for if statement (line 161)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 166):
    
    # Assigning a Call to a Name (line 166):
    
    # Call to scal(...): (line 166)
    # Processing the call arguments (line 166)
    float_411307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 18), 'float')
    # Getting the type of 'inner_res_0' (line 166)
    inner_res_0_411308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 22), 'inner_res_0', False)
    # Applying the binary operator 'div' (line 166)
    result_div_411309 = python_operator(stypy.reporting.localization.Localization(__file__, 166, 18), 'div', float_411307, inner_res_0_411308)
    
    # Getting the type of 'v0' (line 166)
    v0_411310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 35), 'v0', False)
    # Processing the call keyword arguments (line 166)
    kwargs_411311 = {}
    # Getting the type of 'scal' (line 166)
    scal_411306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 13), 'scal', False)
    # Calling scal(args, kwargs) (line 166)
    scal_call_result_411312 = invoke(stypy.reporting.localization.Localization(__file__, 166, 13), scal_411306, *[result_div_411309, v0_411310], **kwargs_411311)
    
    # Assigning a type to the variable 'v0' (line 166)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'v0', scal_call_result_411312)
    
    
    # SSA begins for try-except statement (line 168)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Tuple (line 169):
    
    # Assigning a Subscript to a Name (line 169):
    
    # Obtaining the type of the subscript
    int_411313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 12), 'int')
    
    # Call to _fgmres(...): (line 169)
    # Processing the call arguments (line 169)
    # Getting the type of 'matvec' (line 169)
    matvec_411315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 41), 'matvec', False)
    # Getting the type of 'v0' (line 170)
    v0_411316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 41), 'v0', False)
    # Getting the type of 'inner_m' (line 171)
    inner_m_411317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 41), 'inner_m', False)
    # Processing the call keyword arguments (line 169)
    # Getting the type of 'psolve' (line 172)
    psolve_411318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 49), 'psolve', False)
    keyword_411319 = psolve_411318
    # Getting the type of 'tol' (line 173)
    tol_411320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 46), 'tol', False)
    # Getting the type of 'b_norm' (line 173)
    b_norm_411321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 50), 'b_norm', False)
    # Applying the binary operator '*' (line 173)
    result_mul_411322 = python_operator(stypy.reporting.localization.Localization(__file__, 173, 46), '*', tol_411320, b_norm_411321)
    
    # Getting the type of 'r_norm' (line 173)
    r_norm_411323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 57), 'r_norm', False)
    # Applying the binary operator 'div' (line 173)
    result_div_411324 = python_operator(stypy.reporting.localization.Localization(__file__, 173, 56), 'div', result_mul_411322, r_norm_411323)
    
    keyword_411325 = result_div_411324
    # Getting the type of 'outer_v' (line 174)
    outer_v_411326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 49), 'outer_v', False)
    keyword_411327 = outer_v_411326
    # Getting the type of 'prepend_outer_v' (line 175)
    prepend_outer_v_411328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 57), 'prepend_outer_v', False)
    keyword_411329 = prepend_outer_v_411328
    kwargs_411330 = {'prepend_outer_v': keyword_411329, 'lpsolve': keyword_411319, 'outer_v': keyword_411327, 'atol': keyword_411325}
    # Getting the type of '_fgmres' (line 169)
    _fgmres_411314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 33), '_fgmres', False)
    # Calling _fgmres(args, kwargs) (line 169)
    _fgmres_call_result_411331 = invoke(stypy.reporting.localization.Localization(__file__, 169, 33), _fgmres_411314, *[matvec_411315, v0_411316, inner_m_411317], **kwargs_411330)
    
    # Obtaining the member '__getitem__' of a type (line 169)
    getitem___411332 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 12), _fgmres_call_result_411331, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 169)
    subscript_call_result_411333 = invoke(stypy.reporting.localization.Localization(__file__, 169, 12), getitem___411332, int_411313)
    
    # Assigning a type to the variable 'tuple_var_assignment_411030' (line 169)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 12), 'tuple_var_assignment_411030', subscript_call_result_411333)
    
    # Assigning a Subscript to a Name (line 169):
    
    # Obtaining the type of the subscript
    int_411334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 12), 'int')
    
    # Call to _fgmres(...): (line 169)
    # Processing the call arguments (line 169)
    # Getting the type of 'matvec' (line 169)
    matvec_411336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 41), 'matvec', False)
    # Getting the type of 'v0' (line 170)
    v0_411337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 41), 'v0', False)
    # Getting the type of 'inner_m' (line 171)
    inner_m_411338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 41), 'inner_m', False)
    # Processing the call keyword arguments (line 169)
    # Getting the type of 'psolve' (line 172)
    psolve_411339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 49), 'psolve', False)
    keyword_411340 = psolve_411339
    # Getting the type of 'tol' (line 173)
    tol_411341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 46), 'tol', False)
    # Getting the type of 'b_norm' (line 173)
    b_norm_411342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 50), 'b_norm', False)
    # Applying the binary operator '*' (line 173)
    result_mul_411343 = python_operator(stypy.reporting.localization.Localization(__file__, 173, 46), '*', tol_411341, b_norm_411342)
    
    # Getting the type of 'r_norm' (line 173)
    r_norm_411344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 57), 'r_norm', False)
    # Applying the binary operator 'div' (line 173)
    result_div_411345 = python_operator(stypy.reporting.localization.Localization(__file__, 173, 56), 'div', result_mul_411343, r_norm_411344)
    
    keyword_411346 = result_div_411345
    # Getting the type of 'outer_v' (line 174)
    outer_v_411347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 49), 'outer_v', False)
    keyword_411348 = outer_v_411347
    # Getting the type of 'prepend_outer_v' (line 175)
    prepend_outer_v_411349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 57), 'prepend_outer_v', False)
    keyword_411350 = prepend_outer_v_411349
    kwargs_411351 = {'prepend_outer_v': keyword_411350, 'lpsolve': keyword_411340, 'outer_v': keyword_411348, 'atol': keyword_411346}
    # Getting the type of '_fgmres' (line 169)
    _fgmres_411335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 33), '_fgmres', False)
    # Calling _fgmres(args, kwargs) (line 169)
    _fgmres_call_result_411352 = invoke(stypy.reporting.localization.Localization(__file__, 169, 33), _fgmres_411335, *[matvec_411336, v0_411337, inner_m_411338], **kwargs_411351)
    
    # Obtaining the member '__getitem__' of a type (line 169)
    getitem___411353 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 12), _fgmres_call_result_411352, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 169)
    subscript_call_result_411354 = invoke(stypy.reporting.localization.Localization(__file__, 169, 12), getitem___411353, int_411334)
    
    # Assigning a type to the variable 'tuple_var_assignment_411031' (line 169)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 12), 'tuple_var_assignment_411031', subscript_call_result_411354)
    
    # Assigning a Subscript to a Name (line 169):
    
    # Obtaining the type of the subscript
    int_411355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 12), 'int')
    
    # Call to _fgmres(...): (line 169)
    # Processing the call arguments (line 169)
    # Getting the type of 'matvec' (line 169)
    matvec_411357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 41), 'matvec', False)
    # Getting the type of 'v0' (line 170)
    v0_411358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 41), 'v0', False)
    # Getting the type of 'inner_m' (line 171)
    inner_m_411359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 41), 'inner_m', False)
    # Processing the call keyword arguments (line 169)
    # Getting the type of 'psolve' (line 172)
    psolve_411360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 49), 'psolve', False)
    keyword_411361 = psolve_411360
    # Getting the type of 'tol' (line 173)
    tol_411362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 46), 'tol', False)
    # Getting the type of 'b_norm' (line 173)
    b_norm_411363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 50), 'b_norm', False)
    # Applying the binary operator '*' (line 173)
    result_mul_411364 = python_operator(stypy.reporting.localization.Localization(__file__, 173, 46), '*', tol_411362, b_norm_411363)
    
    # Getting the type of 'r_norm' (line 173)
    r_norm_411365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 57), 'r_norm', False)
    # Applying the binary operator 'div' (line 173)
    result_div_411366 = python_operator(stypy.reporting.localization.Localization(__file__, 173, 56), 'div', result_mul_411364, r_norm_411365)
    
    keyword_411367 = result_div_411366
    # Getting the type of 'outer_v' (line 174)
    outer_v_411368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 49), 'outer_v', False)
    keyword_411369 = outer_v_411368
    # Getting the type of 'prepend_outer_v' (line 175)
    prepend_outer_v_411370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 57), 'prepend_outer_v', False)
    keyword_411371 = prepend_outer_v_411370
    kwargs_411372 = {'prepend_outer_v': keyword_411371, 'lpsolve': keyword_411361, 'outer_v': keyword_411369, 'atol': keyword_411367}
    # Getting the type of '_fgmres' (line 169)
    _fgmres_411356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 33), '_fgmres', False)
    # Calling _fgmres(args, kwargs) (line 169)
    _fgmres_call_result_411373 = invoke(stypy.reporting.localization.Localization(__file__, 169, 33), _fgmres_411356, *[matvec_411357, v0_411358, inner_m_411359], **kwargs_411372)
    
    # Obtaining the member '__getitem__' of a type (line 169)
    getitem___411374 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 12), _fgmres_call_result_411373, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 169)
    subscript_call_result_411375 = invoke(stypy.reporting.localization.Localization(__file__, 169, 12), getitem___411374, int_411355)
    
    # Assigning a type to the variable 'tuple_var_assignment_411032' (line 169)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 12), 'tuple_var_assignment_411032', subscript_call_result_411375)
    
    # Assigning a Subscript to a Name (line 169):
    
    # Obtaining the type of the subscript
    int_411376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 12), 'int')
    
    # Call to _fgmres(...): (line 169)
    # Processing the call arguments (line 169)
    # Getting the type of 'matvec' (line 169)
    matvec_411378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 41), 'matvec', False)
    # Getting the type of 'v0' (line 170)
    v0_411379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 41), 'v0', False)
    # Getting the type of 'inner_m' (line 171)
    inner_m_411380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 41), 'inner_m', False)
    # Processing the call keyword arguments (line 169)
    # Getting the type of 'psolve' (line 172)
    psolve_411381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 49), 'psolve', False)
    keyword_411382 = psolve_411381
    # Getting the type of 'tol' (line 173)
    tol_411383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 46), 'tol', False)
    # Getting the type of 'b_norm' (line 173)
    b_norm_411384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 50), 'b_norm', False)
    # Applying the binary operator '*' (line 173)
    result_mul_411385 = python_operator(stypy.reporting.localization.Localization(__file__, 173, 46), '*', tol_411383, b_norm_411384)
    
    # Getting the type of 'r_norm' (line 173)
    r_norm_411386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 57), 'r_norm', False)
    # Applying the binary operator 'div' (line 173)
    result_div_411387 = python_operator(stypy.reporting.localization.Localization(__file__, 173, 56), 'div', result_mul_411385, r_norm_411386)
    
    keyword_411388 = result_div_411387
    # Getting the type of 'outer_v' (line 174)
    outer_v_411389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 49), 'outer_v', False)
    keyword_411390 = outer_v_411389
    # Getting the type of 'prepend_outer_v' (line 175)
    prepend_outer_v_411391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 57), 'prepend_outer_v', False)
    keyword_411392 = prepend_outer_v_411391
    kwargs_411393 = {'prepend_outer_v': keyword_411392, 'lpsolve': keyword_411382, 'outer_v': keyword_411390, 'atol': keyword_411388}
    # Getting the type of '_fgmres' (line 169)
    _fgmres_411377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 33), '_fgmres', False)
    # Calling _fgmres(args, kwargs) (line 169)
    _fgmres_call_result_411394 = invoke(stypy.reporting.localization.Localization(__file__, 169, 33), _fgmres_411377, *[matvec_411378, v0_411379, inner_m_411380], **kwargs_411393)
    
    # Obtaining the member '__getitem__' of a type (line 169)
    getitem___411395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 12), _fgmres_call_result_411394, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 169)
    subscript_call_result_411396 = invoke(stypy.reporting.localization.Localization(__file__, 169, 12), getitem___411395, int_411376)
    
    # Assigning a type to the variable 'tuple_var_assignment_411033' (line 169)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 12), 'tuple_var_assignment_411033', subscript_call_result_411396)
    
    # Assigning a Subscript to a Name (line 169):
    
    # Obtaining the type of the subscript
    int_411397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 12), 'int')
    
    # Call to _fgmres(...): (line 169)
    # Processing the call arguments (line 169)
    # Getting the type of 'matvec' (line 169)
    matvec_411399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 41), 'matvec', False)
    # Getting the type of 'v0' (line 170)
    v0_411400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 41), 'v0', False)
    # Getting the type of 'inner_m' (line 171)
    inner_m_411401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 41), 'inner_m', False)
    # Processing the call keyword arguments (line 169)
    # Getting the type of 'psolve' (line 172)
    psolve_411402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 49), 'psolve', False)
    keyword_411403 = psolve_411402
    # Getting the type of 'tol' (line 173)
    tol_411404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 46), 'tol', False)
    # Getting the type of 'b_norm' (line 173)
    b_norm_411405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 50), 'b_norm', False)
    # Applying the binary operator '*' (line 173)
    result_mul_411406 = python_operator(stypy.reporting.localization.Localization(__file__, 173, 46), '*', tol_411404, b_norm_411405)
    
    # Getting the type of 'r_norm' (line 173)
    r_norm_411407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 57), 'r_norm', False)
    # Applying the binary operator 'div' (line 173)
    result_div_411408 = python_operator(stypy.reporting.localization.Localization(__file__, 173, 56), 'div', result_mul_411406, r_norm_411407)
    
    keyword_411409 = result_div_411408
    # Getting the type of 'outer_v' (line 174)
    outer_v_411410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 49), 'outer_v', False)
    keyword_411411 = outer_v_411410
    # Getting the type of 'prepend_outer_v' (line 175)
    prepend_outer_v_411412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 57), 'prepend_outer_v', False)
    keyword_411413 = prepend_outer_v_411412
    kwargs_411414 = {'prepend_outer_v': keyword_411413, 'lpsolve': keyword_411403, 'outer_v': keyword_411411, 'atol': keyword_411409}
    # Getting the type of '_fgmres' (line 169)
    _fgmres_411398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 33), '_fgmres', False)
    # Calling _fgmres(args, kwargs) (line 169)
    _fgmres_call_result_411415 = invoke(stypy.reporting.localization.Localization(__file__, 169, 33), _fgmres_411398, *[matvec_411399, v0_411400, inner_m_411401], **kwargs_411414)
    
    # Obtaining the member '__getitem__' of a type (line 169)
    getitem___411416 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 12), _fgmres_call_result_411415, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 169)
    subscript_call_result_411417 = invoke(stypy.reporting.localization.Localization(__file__, 169, 12), getitem___411416, int_411397)
    
    # Assigning a type to the variable 'tuple_var_assignment_411034' (line 169)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 12), 'tuple_var_assignment_411034', subscript_call_result_411417)
    
    # Assigning a Subscript to a Name (line 169):
    
    # Obtaining the type of the subscript
    int_411418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 12), 'int')
    
    # Call to _fgmres(...): (line 169)
    # Processing the call arguments (line 169)
    # Getting the type of 'matvec' (line 169)
    matvec_411420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 41), 'matvec', False)
    # Getting the type of 'v0' (line 170)
    v0_411421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 41), 'v0', False)
    # Getting the type of 'inner_m' (line 171)
    inner_m_411422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 41), 'inner_m', False)
    # Processing the call keyword arguments (line 169)
    # Getting the type of 'psolve' (line 172)
    psolve_411423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 49), 'psolve', False)
    keyword_411424 = psolve_411423
    # Getting the type of 'tol' (line 173)
    tol_411425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 46), 'tol', False)
    # Getting the type of 'b_norm' (line 173)
    b_norm_411426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 50), 'b_norm', False)
    # Applying the binary operator '*' (line 173)
    result_mul_411427 = python_operator(stypy.reporting.localization.Localization(__file__, 173, 46), '*', tol_411425, b_norm_411426)
    
    # Getting the type of 'r_norm' (line 173)
    r_norm_411428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 57), 'r_norm', False)
    # Applying the binary operator 'div' (line 173)
    result_div_411429 = python_operator(stypy.reporting.localization.Localization(__file__, 173, 56), 'div', result_mul_411427, r_norm_411428)
    
    keyword_411430 = result_div_411429
    # Getting the type of 'outer_v' (line 174)
    outer_v_411431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 49), 'outer_v', False)
    keyword_411432 = outer_v_411431
    # Getting the type of 'prepend_outer_v' (line 175)
    prepend_outer_v_411433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 57), 'prepend_outer_v', False)
    keyword_411434 = prepend_outer_v_411433
    kwargs_411435 = {'prepend_outer_v': keyword_411434, 'lpsolve': keyword_411424, 'outer_v': keyword_411432, 'atol': keyword_411430}
    # Getting the type of '_fgmres' (line 169)
    _fgmres_411419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 33), '_fgmres', False)
    # Calling _fgmres(args, kwargs) (line 169)
    _fgmres_call_result_411436 = invoke(stypy.reporting.localization.Localization(__file__, 169, 33), _fgmres_411419, *[matvec_411420, v0_411421, inner_m_411422], **kwargs_411435)
    
    # Obtaining the member '__getitem__' of a type (line 169)
    getitem___411437 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 12), _fgmres_call_result_411436, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 169)
    subscript_call_result_411438 = invoke(stypy.reporting.localization.Localization(__file__, 169, 12), getitem___411437, int_411418)
    
    # Assigning a type to the variable 'tuple_var_assignment_411035' (line 169)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 12), 'tuple_var_assignment_411035', subscript_call_result_411438)
    
    # Assigning a Name to a Name (line 169):
    # Getting the type of 'tuple_var_assignment_411030' (line 169)
    tuple_var_assignment_411030_411439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 12), 'tuple_var_assignment_411030')
    # Assigning a type to the variable 'Q' (line 169)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 12), 'Q', tuple_var_assignment_411030_411439)
    
    # Assigning a Name to a Name (line 169):
    # Getting the type of 'tuple_var_assignment_411031' (line 169)
    tuple_var_assignment_411031_411440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 12), 'tuple_var_assignment_411031')
    # Assigning a type to the variable 'R' (line 169)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 15), 'R', tuple_var_assignment_411031_411440)
    
    # Assigning a Name to a Name (line 169):
    # Getting the type of 'tuple_var_assignment_411032' (line 169)
    tuple_var_assignment_411032_411441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 12), 'tuple_var_assignment_411032')
    # Assigning a type to the variable 'B' (line 169)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 18), 'B', tuple_var_assignment_411032_411441)
    
    # Assigning a Name to a Name (line 169):
    # Getting the type of 'tuple_var_assignment_411033' (line 169)
    tuple_var_assignment_411033_411442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 12), 'tuple_var_assignment_411033')
    # Assigning a type to the variable 'vs' (line 169)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 21), 'vs', tuple_var_assignment_411033_411442)
    
    # Assigning a Name to a Name (line 169):
    # Getting the type of 'tuple_var_assignment_411034' (line 169)
    tuple_var_assignment_411034_411443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 12), 'tuple_var_assignment_411034')
    # Assigning a type to the variable 'zs' (line 169)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 25), 'zs', tuple_var_assignment_411034_411443)
    
    # Assigning a Name to a Name (line 169):
    # Getting the type of 'tuple_var_assignment_411035' (line 169)
    tuple_var_assignment_411035_411444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 12), 'tuple_var_assignment_411035')
    # Assigning a type to the variable 'y' (line 169)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 29), 'y', tuple_var_assignment_411035_411444)
    
    # Getting the type of 'y' (line 176)
    y_411445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 12), 'y')
    # Getting the type of 'inner_res_0' (line 176)
    inner_res_0_411446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 17), 'inner_res_0')
    # Applying the binary operator '*=' (line 176)
    result_imul_411447 = python_operator(stypy.reporting.localization.Localization(__file__, 176, 12), '*=', y_411445, inner_res_0_411446)
    # Assigning a type to the variable 'y' (line 176)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 12), 'y', result_imul_411447)
    
    
    
    
    # Call to all(...): (line 177)
    # Processing the call keyword arguments (line 177)
    kwargs_411454 = {}
    
    # Call to isfinite(...): (line 177)
    # Processing the call arguments (line 177)
    # Getting the type of 'y' (line 177)
    y_411450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 31), 'y', False)
    # Processing the call keyword arguments (line 177)
    kwargs_411451 = {}
    # Getting the type of 'np' (line 177)
    np_411448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 19), 'np', False)
    # Obtaining the member 'isfinite' of a type (line 177)
    isfinite_411449 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 19), np_411448, 'isfinite')
    # Calling isfinite(args, kwargs) (line 177)
    isfinite_call_result_411452 = invoke(stypy.reporting.localization.Localization(__file__, 177, 19), isfinite_411449, *[y_411450], **kwargs_411451)
    
    # Obtaining the member 'all' of a type (line 177)
    all_411453 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 19), isfinite_call_result_411452, 'all')
    # Calling all(args, kwargs) (line 177)
    all_call_result_411455 = invoke(stypy.reporting.localization.Localization(__file__, 177, 19), all_411453, *[], **kwargs_411454)
    
    # Applying the 'not' unary operator (line 177)
    result_not__411456 = python_operator(stypy.reporting.localization.Localization(__file__, 177, 15), 'not', all_call_result_411455)
    
    # Testing the type of an if condition (line 177)
    if_condition_411457 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 177, 12), result_not__411456)
    # Assigning a type to the variable 'if_condition_411457' (line 177)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 12), 'if_condition_411457', if_condition_411457)
    # SSA begins for if statement (line 177)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to LinAlgError(...): (line 180)
    # Processing the call keyword arguments (line 180)
    kwargs_411459 = {}
    # Getting the type of 'LinAlgError' (line 180)
    LinAlgError_411458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 22), 'LinAlgError', False)
    # Calling LinAlgError(args, kwargs) (line 180)
    LinAlgError_call_result_411460 = invoke(stypy.reporting.localization.Localization(__file__, 180, 22), LinAlgError_411458, *[], **kwargs_411459)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 180, 16), LinAlgError_call_result_411460, 'raise parameter', BaseException)
    # SSA join for if statement (line 177)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the except part of a try statement (line 168)
    # SSA branch for the except 'LinAlgError' branch of a try statement (line 168)
    module_type_store.open_ssa_branch('except')
    
    # Obtaining an instance of the builtin type 'tuple' (line 184)
    tuple_411461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 19), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 184)
    # Adding element type (line 184)
    
    # Call to postprocess(...): (line 184)
    # Processing the call arguments (line 184)
    # Getting the type of 'x' (line 184)
    x_411463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 31), 'x', False)
    # Processing the call keyword arguments (line 184)
    kwargs_411464 = {}
    # Getting the type of 'postprocess' (line 184)
    postprocess_411462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 19), 'postprocess', False)
    # Calling postprocess(args, kwargs) (line 184)
    postprocess_call_result_411465 = invoke(stypy.reporting.localization.Localization(__file__, 184, 19), postprocess_411462, *[x_411463], **kwargs_411464)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 184, 19), tuple_411461, postprocess_call_result_411465)
    # Adding element type (line 184)
    # Getting the type of 'k_outer' (line 184)
    k_outer_411466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 35), 'k_outer')
    int_411467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 45), 'int')
    # Applying the binary operator '+' (line 184)
    result_add_411468 = python_operator(stypy.reporting.localization.Localization(__file__, 184, 35), '+', k_outer_411466, int_411467)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 184, 19), tuple_411461, result_add_411468)
    
    # Assigning a type to the variable 'stypy_return_type' (line 184)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 12), 'stypy_return_type', tuple_411461)
    # SSA join for try-except statement (line 168)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 187):
    
    # Assigning a BinOp to a Name (line 187):
    
    # Obtaining the type of the subscript
    int_411469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 16), 'int')
    # Getting the type of 'zs' (line 187)
    zs_411470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 13), 'zs')
    # Obtaining the member '__getitem__' of a type (line 187)
    getitem___411471 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 13), zs_411470, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 187)
    subscript_call_result_411472 = invoke(stypy.reporting.localization.Localization(__file__, 187, 13), getitem___411471, int_411469)
    
    
    # Obtaining the type of the subscript
    int_411473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 21), 'int')
    # Getting the type of 'y' (line 187)
    y_411474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 19), 'y')
    # Obtaining the member '__getitem__' of a type (line 187)
    getitem___411475 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 19), y_411474, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 187)
    subscript_call_result_411476 = invoke(stypy.reporting.localization.Localization(__file__, 187, 19), getitem___411475, int_411473)
    
    # Applying the binary operator '*' (line 187)
    result_mul_411477 = python_operator(stypy.reporting.localization.Localization(__file__, 187, 13), '*', subscript_call_result_411472, subscript_call_result_411476)
    
    # Assigning a type to the variable 'dx' (line 187)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 8), 'dx', result_mul_411477)
    
    
    # Call to zip(...): (line 188)
    # Processing the call arguments (line 188)
    
    # Obtaining the type of the subscript
    int_411479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 28), 'int')
    slice_411480 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 188, 25), int_411479, None, None)
    # Getting the type of 'zs' (line 188)
    zs_411481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 25), 'zs', False)
    # Obtaining the member '__getitem__' of a type (line 188)
    getitem___411482 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 25), zs_411481, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 188)
    subscript_call_result_411483 = invoke(stypy.reporting.localization.Localization(__file__, 188, 25), getitem___411482, slice_411480)
    
    
    # Obtaining the type of the subscript
    int_411484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 35), 'int')
    slice_411485 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 188, 33), int_411484, None, None)
    # Getting the type of 'y' (line 188)
    y_411486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 33), 'y', False)
    # Obtaining the member '__getitem__' of a type (line 188)
    getitem___411487 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 33), y_411486, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 188)
    subscript_call_result_411488 = invoke(stypy.reporting.localization.Localization(__file__, 188, 33), getitem___411487, slice_411485)
    
    # Processing the call keyword arguments (line 188)
    kwargs_411489 = {}
    # Getting the type of 'zip' (line 188)
    zip_411478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 21), 'zip', False)
    # Calling zip(args, kwargs) (line 188)
    zip_call_result_411490 = invoke(stypy.reporting.localization.Localization(__file__, 188, 21), zip_411478, *[subscript_call_result_411483, subscript_call_result_411488], **kwargs_411489)
    
    # Testing the type of a for loop iterable (line 188)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 188, 8), zip_call_result_411490)
    # Getting the type of the for loop variable (line 188)
    for_loop_var_411491 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 188, 8), zip_call_result_411490)
    # Assigning a type to the variable 'w' (line 188)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 8), 'w', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 188, 8), for_loop_var_411491))
    # Assigning a type to the variable 'yc' (line 188)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 8), 'yc', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 188, 8), for_loop_var_411491))
    # SSA begins for a for statement (line 188)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 189):
    
    # Assigning a Call to a Name (line 189):
    
    # Call to axpy(...): (line 189)
    # Processing the call arguments (line 189)
    # Getting the type of 'w' (line 189)
    w_411493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 22), 'w', False)
    # Getting the type of 'dx' (line 189)
    dx_411494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 25), 'dx', False)
    
    # Obtaining the type of the subscript
    int_411495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 38), 'int')
    # Getting the type of 'dx' (line 189)
    dx_411496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 29), 'dx', False)
    # Obtaining the member 'shape' of a type (line 189)
    shape_411497 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 29), dx_411496, 'shape')
    # Obtaining the member '__getitem__' of a type (line 189)
    getitem___411498 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 29), shape_411497, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 189)
    subscript_call_result_411499 = invoke(stypy.reporting.localization.Localization(__file__, 189, 29), getitem___411498, int_411495)
    
    # Getting the type of 'yc' (line 189)
    yc_411500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 42), 'yc', False)
    # Processing the call keyword arguments (line 189)
    kwargs_411501 = {}
    # Getting the type of 'axpy' (line 189)
    axpy_411492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 17), 'axpy', False)
    # Calling axpy(args, kwargs) (line 189)
    axpy_call_result_411502 = invoke(stypy.reporting.localization.Localization(__file__, 189, 17), axpy_411492, *[w_411493, dx_411494, subscript_call_result_411499, yc_411500], **kwargs_411501)
    
    # Assigning a type to the variable 'dx' (line 189)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 12), 'dx', axpy_call_result_411502)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 192):
    
    # Assigning a Call to a Name (line 192):
    
    # Call to nrm2(...): (line 192)
    # Processing the call arguments (line 192)
    # Getting the type of 'dx' (line 192)
    dx_411504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 18), 'dx', False)
    # Processing the call keyword arguments (line 192)
    kwargs_411505 = {}
    # Getting the type of 'nrm2' (line 192)
    nrm2_411503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 13), 'nrm2', False)
    # Calling nrm2(args, kwargs) (line 192)
    nrm2_call_result_411506 = invoke(stypy.reporting.localization.Localization(__file__, 192, 13), nrm2_411503, *[dx_411504], **kwargs_411505)
    
    # Assigning a type to the variable 'nx' (line 192)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 8), 'nx', nrm2_call_result_411506)
    
    
    # Getting the type of 'nx' (line 193)
    nx_411507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 11), 'nx')
    int_411508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 16), 'int')
    # Applying the binary operator '>' (line 193)
    result_gt_411509 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 11), '>', nx_411507, int_411508)
    
    # Testing the type of an if condition (line 193)
    if_condition_411510 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 193, 8), result_gt_411509)
    # Assigning a type to the variable 'if_condition_411510' (line 193)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 8), 'if_condition_411510', if_condition_411510)
    # SSA begins for if statement (line 193)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'store_outer_Av' (line 194)
    store_outer_Av_411511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 15), 'store_outer_Av')
    # Testing the type of an if condition (line 194)
    if_condition_411512 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 194, 12), store_outer_Av_411511)
    # Assigning a type to the variable 'if_condition_411512' (line 194)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 12), 'if_condition_411512', if_condition_411512)
    # SSA begins for if statement (line 194)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 195):
    
    # Assigning a Call to a Name (line 195):
    
    # Call to dot(...): (line 195)
    # Processing the call arguments (line 195)
    
    # Call to dot(...): (line 195)
    # Processing the call arguments (line 195)
    # Getting the type of 'y' (line 195)
    y_411517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 32), 'y', False)
    # Processing the call keyword arguments (line 195)
    kwargs_411518 = {}
    # Getting the type of 'R' (line 195)
    R_411515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 26), 'R', False)
    # Obtaining the member 'dot' of a type (line 195)
    dot_411516 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 26), R_411515, 'dot')
    # Calling dot(args, kwargs) (line 195)
    dot_call_result_411519 = invoke(stypy.reporting.localization.Localization(__file__, 195, 26), dot_411516, *[y_411517], **kwargs_411518)
    
    # Processing the call keyword arguments (line 195)
    kwargs_411520 = {}
    # Getting the type of 'Q' (line 195)
    Q_411513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 20), 'Q', False)
    # Obtaining the member 'dot' of a type (line 195)
    dot_411514 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 20), Q_411513, 'dot')
    # Calling dot(args, kwargs) (line 195)
    dot_call_result_411521 = invoke(stypy.reporting.localization.Localization(__file__, 195, 20), dot_411514, *[dot_call_result_411519], **kwargs_411520)
    
    # Assigning a type to the variable 'q' (line 195)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 16), 'q', dot_call_result_411521)
    
    # Assigning a BinOp to a Name (line 196):
    
    # Assigning a BinOp to a Name (line 196):
    
    # Obtaining the type of the subscript
    int_411522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 24), 'int')
    # Getting the type of 'vs' (line 196)
    vs_411523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 21), 'vs')
    # Obtaining the member '__getitem__' of a type (line 196)
    getitem___411524 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 21), vs_411523, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 196)
    subscript_call_result_411525 = invoke(stypy.reporting.localization.Localization(__file__, 196, 21), getitem___411524, int_411522)
    
    
    # Obtaining the type of the subscript
    int_411526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 29), 'int')
    # Getting the type of 'q' (line 196)
    q_411527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 27), 'q')
    # Obtaining the member '__getitem__' of a type (line 196)
    getitem___411528 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 27), q_411527, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 196)
    subscript_call_result_411529 = invoke(stypy.reporting.localization.Localization(__file__, 196, 27), getitem___411528, int_411526)
    
    # Applying the binary operator '*' (line 196)
    result_mul_411530 = python_operator(stypy.reporting.localization.Localization(__file__, 196, 21), '*', subscript_call_result_411525, subscript_call_result_411529)
    
    # Assigning a type to the variable 'ax' (line 196)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 16), 'ax', result_mul_411530)
    
    
    # Call to zip(...): (line 197)
    # Processing the call arguments (line 197)
    
    # Obtaining the type of the subscript
    int_411532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 36), 'int')
    slice_411533 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 197, 33), int_411532, None, None)
    # Getting the type of 'vs' (line 197)
    vs_411534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 33), 'vs', False)
    # Obtaining the member '__getitem__' of a type (line 197)
    getitem___411535 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 33), vs_411534, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 197)
    subscript_call_result_411536 = invoke(stypy.reporting.localization.Localization(__file__, 197, 33), getitem___411535, slice_411533)
    
    
    # Obtaining the type of the subscript
    int_411537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 43), 'int')
    slice_411538 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 197, 41), int_411537, None, None)
    # Getting the type of 'q' (line 197)
    q_411539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 41), 'q', False)
    # Obtaining the member '__getitem__' of a type (line 197)
    getitem___411540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 41), q_411539, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 197)
    subscript_call_result_411541 = invoke(stypy.reporting.localization.Localization(__file__, 197, 41), getitem___411540, slice_411538)
    
    # Processing the call keyword arguments (line 197)
    kwargs_411542 = {}
    # Getting the type of 'zip' (line 197)
    zip_411531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 29), 'zip', False)
    # Calling zip(args, kwargs) (line 197)
    zip_call_result_411543 = invoke(stypy.reporting.localization.Localization(__file__, 197, 29), zip_411531, *[subscript_call_result_411536, subscript_call_result_411541], **kwargs_411542)
    
    # Testing the type of a for loop iterable (line 197)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 197, 16), zip_call_result_411543)
    # Getting the type of the for loop variable (line 197)
    for_loop_var_411544 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 197, 16), zip_call_result_411543)
    # Assigning a type to the variable 'v' (line 197)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 16), 'v', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 197, 16), for_loop_var_411544))
    # Assigning a type to the variable 'qc' (line 197)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 16), 'qc', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 197, 16), for_loop_var_411544))
    # SSA begins for a for statement (line 197)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 198):
    
    # Assigning a Call to a Name (line 198):
    
    # Call to axpy(...): (line 198)
    # Processing the call arguments (line 198)
    # Getting the type of 'v' (line 198)
    v_411546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 30), 'v', False)
    # Getting the type of 'ax' (line 198)
    ax_411547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 33), 'ax', False)
    
    # Obtaining the type of the subscript
    int_411548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 46), 'int')
    # Getting the type of 'ax' (line 198)
    ax_411549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 37), 'ax', False)
    # Obtaining the member 'shape' of a type (line 198)
    shape_411550 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 37), ax_411549, 'shape')
    # Obtaining the member '__getitem__' of a type (line 198)
    getitem___411551 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 37), shape_411550, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 198)
    subscript_call_result_411552 = invoke(stypy.reporting.localization.Localization(__file__, 198, 37), getitem___411551, int_411548)
    
    # Getting the type of 'qc' (line 198)
    qc_411553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 50), 'qc', False)
    # Processing the call keyword arguments (line 198)
    kwargs_411554 = {}
    # Getting the type of 'axpy' (line 198)
    axpy_411545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 25), 'axpy', False)
    # Calling axpy(args, kwargs) (line 198)
    axpy_call_result_411555 = invoke(stypy.reporting.localization.Localization(__file__, 198, 25), axpy_411545, *[v_411546, ax_411547, subscript_call_result_411552, qc_411553], **kwargs_411554)
    
    # Assigning a type to the variable 'ax' (line 198)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 20), 'ax', axpy_call_result_411555)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to append(...): (line 199)
    # Processing the call arguments (line 199)
    
    # Obtaining an instance of the builtin type 'tuple' (line 199)
    tuple_411558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 32), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 199)
    # Adding element type (line 199)
    # Getting the type of 'dx' (line 199)
    dx_411559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 32), 'dx', False)
    # Getting the type of 'nx' (line 199)
    nx_411560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 35), 'nx', False)
    # Applying the binary operator 'div' (line 199)
    result_div_411561 = python_operator(stypy.reporting.localization.Localization(__file__, 199, 32), 'div', dx_411559, nx_411560)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 32), tuple_411558, result_div_411561)
    # Adding element type (line 199)
    # Getting the type of 'ax' (line 199)
    ax_411562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 39), 'ax', False)
    # Getting the type of 'nx' (line 199)
    nx_411563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 42), 'nx', False)
    # Applying the binary operator 'div' (line 199)
    result_div_411564 = python_operator(stypy.reporting.localization.Localization(__file__, 199, 39), 'div', ax_411562, nx_411563)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 32), tuple_411558, result_div_411564)
    
    # Processing the call keyword arguments (line 199)
    kwargs_411565 = {}
    # Getting the type of 'outer_v' (line 199)
    outer_v_411556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 16), 'outer_v', False)
    # Obtaining the member 'append' of a type (line 199)
    append_411557 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 16), outer_v_411556, 'append')
    # Calling append(args, kwargs) (line 199)
    append_call_result_411566 = invoke(stypy.reporting.localization.Localization(__file__, 199, 16), append_411557, *[tuple_411558], **kwargs_411565)
    
    # SSA branch for the else part of an if statement (line 194)
    module_type_store.open_ssa_branch('else')
    
    # Call to append(...): (line 201)
    # Processing the call arguments (line 201)
    
    # Obtaining an instance of the builtin type 'tuple' (line 201)
    tuple_411569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 32), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 201)
    # Adding element type (line 201)
    # Getting the type of 'dx' (line 201)
    dx_411570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 32), 'dx', False)
    # Getting the type of 'nx' (line 201)
    nx_411571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 35), 'nx', False)
    # Applying the binary operator 'div' (line 201)
    result_div_411572 = python_operator(stypy.reporting.localization.Localization(__file__, 201, 32), 'div', dx_411570, nx_411571)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 201, 32), tuple_411569, result_div_411572)
    # Adding element type (line 201)
    # Getting the type of 'None' (line 201)
    None_411573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 39), 'None', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 201, 32), tuple_411569, None_411573)
    
    # Processing the call keyword arguments (line 201)
    kwargs_411574 = {}
    # Getting the type of 'outer_v' (line 201)
    outer_v_411567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 16), 'outer_v', False)
    # Obtaining the member 'append' of a type (line 201)
    append_411568 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 16), outer_v_411567, 'append')
    # Calling append(args, kwargs) (line 201)
    append_call_result_411575 = invoke(stypy.reporting.localization.Localization(__file__, 201, 16), append_411568, *[tuple_411569], **kwargs_411574)
    
    # SSA join for if statement (line 194)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 193)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to len(...): (line 204)
    # Processing the call arguments (line 204)
    # Getting the type of 'outer_v' (line 204)
    outer_v_411577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 18), 'outer_v', False)
    # Processing the call keyword arguments (line 204)
    kwargs_411578 = {}
    # Getting the type of 'len' (line 204)
    len_411576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 14), 'len', False)
    # Calling len(args, kwargs) (line 204)
    len_call_result_411579 = invoke(stypy.reporting.localization.Localization(__file__, 204, 14), len_411576, *[outer_v_411577], **kwargs_411578)
    
    # Getting the type of 'outer_k' (line 204)
    outer_k_411580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 29), 'outer_k')
    # Applying the binary operator '>' (line 204)
    result_gt_411581 = python_operator(stypy.reporting.localization.Localization(__file__, 204, 14), '>', len_call_result_411579, outer_k_411580)
    
    # Testing the type of an if condition (line 204)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 204, 8), result_gt_411581)
    # SSA begins for while statement (line 204)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    # Deleting a member
    # Getting the type of 'outer_v' (line 205)
    outer_v_411582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 16), 'outer_v')
    
    # Obtaining the type of the subscript
    int_411583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 24), 'int')
    # Getting the type of 'outer_v' (line 205)
    outer_v_411584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 16), 'outer_v')
    # Obtaining the member '__getitem__' of a type (line 205)
    getitem___411585 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 16), outer_v_411584, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 205)
    subscript_call_result_411586 = invoke(stypy.reporting.localization.Localization(__file__, 205, 16), getitem___411585, int_411583)
    
    del_contained_elements_type(stypy.reporting.localization.Localization(__file__, 205, 12), outer_v_411582, subscript_call_result_411586)
    # SSA join for while statement (line 204)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'x' (line 208)
    x_411587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 8), 'x')
    # Getting the type of 'dx' (line 208)
    dx_411588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 13), 'dx')
    # Applying the binary operator '+=' (line 208)
    result_iadd_411589 = python_operator(stypy.reporting.localization.Localization(__file__, 208, 8), '+=', x_411587, dx_411588)
    # Assigning a type to the variable 'x' (line 208)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 8), 'x', result_iadd_411589)
    
    # SSA branch for the else part of a for statement (line 137)
    module_type_store.open_ssa_branch('for loop else')
    
    # Obtaining an instance of the builtin type 'tuple' (line 211)
    tuple_411590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 211)
    # Adding element type (line 211)
    
    # Call to postprocess(...): (line 211)
    # Processing the call arguments (line 211)
    # Getting the type of 'x' (line 211)
    x_411592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 27), 'x', False)
    # Processing the call keyword arguments (line 211)
    kwargs_411593 = {}
    # Getting the type of 'postprocess' (line 211)
    postprocess_411591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 15), 'postprocess', False)
    # Calling postprocess(args, kwargs) (line 211)
    postprocess_call_result_411594 = invoke(stypy.reporting.localization.Localization(__file__, 211, 15), postprocess_411591, *[x_411592], **kwargs_411593)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 211, 15), tuple_411590, postprocess_call_result_411594)
    # Adding element type (line 211)
    # Getting the type of 'maxiter' (line 211)
    maxiter_411595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 31), 'maxiter')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 211, 15), tuple_411590, maxiter_411595)
    
    # Assigning a type to the variable 'stypy_return_type' (line 211)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 8), 'stypy_return_type', tuple_411590)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 213)
    tuple_411596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 213)
    # Adding element type (line 213)
    
    # Call to postprocess(...): (line 213)
    # Processing the call arguments (line 213)
    # Getting the type of 'x' (line 213)
    x_411598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 23), 'x', False)
    # Processing the call keyword arguments (line 213)
    kwargs_411599 = {}
    # Getting the type of 'postprocess' (line 213)
    postprocess_411597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 11), 'postprocess', False)
    # Calling postprocess(args, kwargs) (line 213)
    postprocess_call_result_411600 = invoke(stypy.reporting.localization.Localization(__file__, 213, 11), postprocess_411597, *[x_411598], **kwargs_411599)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 213, 11), tuple_411596, postprocess_call_result_411600)
    # Adding element type (line 213)
    int_411601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 213, 11), tuple_411596, int_411601)
    
    # Assigning a type to the variable 'stypy_return_type' (line 213)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 4), 'stypy_return_type', tuple_411596)
    
    # ################# End of 'lgmres(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'lgmres' in the type store
    # Getting the type of 'stypy_return_type' (line 17)
    stypy_return_type_411602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_411602)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'lgmres'
    return stypy_return_type_411602

# Assigning a type to the variable 'lgmres' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'lgmres', lgmres)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
