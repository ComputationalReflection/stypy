
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Matrix square root for general matrices and for upper triangular matrices.
3: 
4: This module exists to avoid cyclic imports.
5: 
6: '''
7: from __future__ import division, print_function, absolute_import
8: 
9: __all__ = ['sqrtm']
10: 
11: import numpy as np
12: 
13: from scipy._lib._util import _asarray_validated
14: 
15: 
16: # Local imports
17: from .misc import norm
18: from .lapack import ztrsyl, dtrsyl
19: from .decomp_schur import schur, rsf2csf
20: 
21: 
22: class SqrtmError(np.linalg.LinAlgError):
23:     pass
24: 
25: 
26: def _sqrtm_triu(T, blocksize=64):
27:     '''
28:     Matrix square root of an upper triangular matrix.
29: 
30:     This is a helper function for `sqrtm` and `logm`.
31: 
32:     Parameters
33:     ----------
34:     T : (N, N) array_like upper triangular
35:         Matrix whose square root to evaluate
36:     blocksize : int, optional
37:         If the blocksize is not degenerate with respect to the
38:         size of the input array, then use a blocked algorithm. (Default: 64)
39: 
40:     Returns
41:     -------
42:     sqrtm : (N, N) ndarray
43:         Value of the sqrt function at `T`
44: 
45:     References
46:     ----------
47:     .. [1] Edvin Deadman, Nicholas J. Higham, Rui Ralha (2013)
48:            "Blocked Schur Algorithms for Computing the Matrix Square Root,
49:            Lecture Notes in Computer Science, 7782. pp. 171-182.
50: 
51:     '''
52:     T_diag = np.diag(T)
53:     keep_it_real = np.isrealobj(T) and np.min(T_diag) >= 0
54:     if not keep_it_real:
55:         T_diag = T_diag.astype(complex)
56:     R = np.diag(np.sqrt(T_diag))
57: 
58:     # Compute the number of blocks to use; use at least one block.
59:     n, n = T.shape
60:     nblocks = max(n // blocksize, 1)
61: 
62:     # Compute the smaller of the two sizes of blocks that
63:     # we will actually use, and compute the number of large blocks.
64:     bsmall, nlarge = divmod(n, nblocks)
65:     blarge = bsmall + 1
66:     nsmall = nblocks - nlarge
67:     if nsmall * bsmall + nlarge * blarge != n:
68:         raise Exception('internal inconsistency')
69: 
70:     # Define the index range covered by each block.
71:     start_stop_pairs = []
72:     start = 0
73:     for count, size in ((nsmall, bsmall), (nlarge, blarge)):
74:         for i in range(count):
75:             start_stop_pairs.append((start, start + size))
76:             start += size
77: 
78:     # Within-block interactions.
79:     for start, stop in start_stop_pairs:
80:         for j in range(start, stop):
81:             for i in range(j-1, start-1, -1):
82:                 s = 0
83:                 if j - i > 1:
84:                     s = R[i, i+1:j].dot(R[i+1:j, j])
85:                 denom = R[i, i] + R[j, j]
86:                 if not denom:
87:                     raise SqrtmError('failed to find the matrix square root')
88:                 R[i, j] = (T[i, j] - s) / denom
89: 
90:     # Between-block interactions.
91:     for j in range(nblocks):
92:         jstart, jstop = start_stop_pairs[j]
93:         for i in range(j-1, -1, -1):
94:             istart, istop = start_stop_pairs[i]
95:             S = T[istart:istop, jstart:jstop]
96:             if j - i > 1:
97:                 S = S - R[istart:istop, istop:jstart].dot(R[istop:jstart,
98:                                                             jstart:jstop])
99: 
100:             # Invoke LAPACK.
101:             # For more details, see the solve_sylvester implemention
102:             # and the fortran dtrsyl and ztrsyl docs.
103:             Rii = R[istart:istop, istart:istop]
104:             Rjj = R[jstart:jstop, jstart:jstop]
105:             if keep_it_real:
106:                 x, scale, info = dtrsyl(Rii, Rjj, S)
107:             else:
108:                 x, scale, info = ztrsyl(Rii, Rjj, S)
109:             R[istart:istop, jstart:jstop] = x * scale
110: 
111:     # Return the matrix square root.
112:     return R
113: 
114: 
115: def sqrtm(A, disp=True, blocksize=64):
116:     '''
117:     Matrix square root.
118: 
119:     Parameters
120:     ----------
121:     A : (N, N) array_like
122:         Matrix whose square root to evaluate
123:     disp : bool, optional
124:         Print warning if error in the result is estimated large
125:         instead of returning estimated error. (Default: True)
126:     blocksize : integer, optional
127:         If the blocksize is not degenerate with respect to the
128:         size of the input array, then use a blocked algorithm. (Default: 64)
129: 
130:     Returns
131:     -------
132:     sqrtm : (N, N) ndarray
133:         Value of the sqrt function at `A`
134: 
135:     errest : float
136:         (if disp == False)
137: 
138:         Frobenius norm of the estimated error, ||err||_F / ||A||_F
139: 
140:     References
141:     ----------
142:     .. [1] Edvin Deadman, Nicholas J. Higham, Rui Ralha (2013)
143:            "Blocked Schur Algorithms for Computing the Matrix Square Root,
144:            Lecture Notes in Computer Science, 7782. pp. 171-182.
145: 
146:     Examples
147:     --------
148:     >>> from scipy.linalg import sqrtm
149:     >>> a = np.array([[1.0, 3.0], [1.0, 4.0]])
150:     >>> r = sqrtm(a)
151:     >>> r
152:     array([[ 0.75592895,  1.13389342],
153:            [ 0.37796447,  1.88982237]])
154:     >>> r.dot(r)
155:     array([[ 1.,  3.],
156:            [ 1.,  4.]])
157: 
158:     '''
159:     A = _asarray_validated(A, check_finite=True, as_inexact=True)
160:     if len(A.shape) != 2:
161:         raise ValueError("Non-matrix input to matrix function.")
162:     if blocksize < 1:
163:         raise ValueError("The blocksize should be at least 1.")
164:     keep_it_real = np.isrealobj(A)
165:     if keep_it_real:
166:         T, Z = schur(A)
167:         if not np.array_equal(T, np.triu(T)):
168:             T, Z = rsf2csf(T, Z)
169:     else:
170:         T, Z = schur(A, output='complex')
171:     failflag = False
172:     try:
173:         R = _sqrtm_triu(T, blocksize=blocksize)
174:         ZH = np.conjugate(Z).T
175:         X = Z.dot(R).dot(ZH)
176:     except SqrtmError:
177:         failflag = True
178:         X = np.empty_like(A)
179:         X.fill(np.nan)
180: 
181:     if disp:
182:         nzeig = np.any(np.diag(T) == 0)
183:         if nzeig:
184:             print("Matrix is singular and may not have a square root.")
185:         elif failflag:
186:             print("Failed to find a square root.")
187:         return X
188:     else:
189:         try:
190:             arg2 = norm(X.dot(X) - A, 'fro')**2 / norm(A, 'fro')
191:         except ValueError:
192:             # NaNs in matrix
193:             arg2 = np.inf
194: 
195:         return X, arg2
196: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_35096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, (-1)), 'str', '\nMatrix square root for general matrices and for upper triangular matrices.\n\nThis module exists to avoid cyclic imports.\n\n')

# Assigning a List to a Name (line 9):

# Assigning a List to a Name (line 9):
__all__ = ['sqrtm']
module_type_store.set_exportable_members(['sqrtm'])

# Obtaining an instance of the builtin type 'list' (line 9)
list_35097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 9)
# Adding element type (line 9)
str_35098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 11), 'str', 'sqrtm')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 10), list_35097, str_35098)

# Assigning a type to the variable '__all__' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), '__all__', list_35097)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'import numpy' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_35099 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy')

if (type(import_35099) is not StypyTypeError):

    if (import_35099 != 'pyd_module'):
        __import__(import_35099)
        sys_modules_35100 = sys.modules[import_35099]
        import_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'np', sys_modules_35100.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy', import_35099)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from scipy._lib._util import _asarray_validated' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_35101 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy._lib._util')

if (type(import_35101) is not StypyTypeError):

    if (import_35101 != 'pyd_module'):
        __import__(import_35101)
        sys_modules_35102 = sys.modules[import_35101]
        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy._lib._util', sys_modules_35102.module_type_store, module_type_store, ['_asarray_validated'])
        nest_module(stypy.reporting.localization.Localization(__file__, 13, 0), __file__, sys_modules_35102, sys_modules_35102.module_type_store, module_type_store)
    else:
        from scipy._lib._util import _asarray_validated

        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy._lib._util', None, module_type_store, ['_asarray_validated'], [_asarray_validated])

else:
    # Assigning a type to the variable 'scipy._lib._util' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy._lib._util', import_35101)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 17, 0))

# 'from scipy.linalg.misc import norm' statement (line 17)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_35103 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'scipy.linalg.misc')

if (type(import_35103) is not StypyTypeError):

    if (import_35103 != 'pyd_module'):
        __import__(import_35103)
        sys_modules_35104 = sys.modules[import_35103]
        import_from_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'scipy.linalg.misc', sys_modules_35104.module_type_store, module_type_store, ['norm'])
        nest_module(stypy.reporting.localization.Localization(__file__, 17, 0), __file__, sys_modules_35104, sys_modules_35104.module_type_store, module_type_store)
    else:
        from scipy.linalg.misc import norm

        import_from_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'scipy.linalg.misc', None, module_type_store, ['norm'], [norm])

else:
    # Assigning a type to the variable 'scipy.linalg.misc' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'scipy.linalg.misc', import_35103)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 18, 0))

# 'from scipy.linalg.lapack import ztrsyl, dtrsyl' statement (line 18)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_35105 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'scipy.linalg.lapack')

if (type(import_35105) is not StypyTypeError):

    if (import_35105 != 'pyd_module'):
        __import__(import_35105)
        sys_modules_35106 = sys.modules[import_35105]
        import_from_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'scipy.linalg.lapack', sys_modules_35106.module_type_store, module_type_store, ['ztrsyl', 'dtrsyl'])
        nest_module(stypy.reporting.localization.Localization(__file__, 18, 0), __file__, sys_modules_35106, sys_modules_35106.module_type_store, module_type_store)
    else:
        from scipy.linalg.lapack import ztrsyl, dtrsyl

        import_from_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'scipy.linalg.lapack', None, module_type_store, ['ztrsyl', 'dtrsyl'], [ztrsyl, dtrsyl])

else:
    # Assigning a type to the variable 'scipy.linalg.lapack' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'scipy.linalg.lapack', import_35105)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 19, 0))

# 'from scipy.linalg.decomp_schur import schur, rsf2csf' statement (line 19)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_35107 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'scipy.linalg.decomp_schur')

if (type(import_35107) is not StypyTypeError):

    if (import_35107 != 'pyd_module'):
        __import__(import_35107)
        sys_modules_35108 = sys.modules[import_35107]
        import_from_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'scipy.linalg.decomp_schur', sys_modules_35108.module_type_store, module_type_store, ['schur', 'rsf2csf'])
        nest_module(stypy.reporting.localization.Localization(__file__, 19, 0), __file__, sys_modules_35108, sys_modules_35108.module_type_store, module_type_store)
    else:
        from scipy.linalg.decomp_schur import schur, rsf2csf

        import_from_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'scipy.linalg.decomp_schur', None, module_type_store, ['schur', 'rsf2csf'], [schur, rsf2csf])

else:
    # Assigning a type to the variable 'scipy.linalg.decomp_schur' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'scipy.linalg.decomp_schur', import_35107)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')

# Declaration of the 'SqrtmError' class
# Getting the type of 'np' (line 22)
np_35109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 17), 'np')
# Obtaining the member 'linalg' of a type (line 22)
linalg_35110 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 17), np_35109, 'linalg')
# Obtaining the member 'LinAlgError' of a type (line 22)
LinAlgError_35111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 17), linalg_35110, 'LinAlgError')

class SqrtmError(LinAlgError_35111, ):
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 22, 0, False)
        # Assigning a type to the variable 'self' (line 23)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SqrtmError.__init__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        pass
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'SqrtmError' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'SqrtmError', SqrtmError)

@norecursion
def _sqrtm_triu(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_35112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 29), 'int')
    defaults = [int_35112]
    # Create a new context for function '_sqrtm_triu'
    module_type_store = module_type_store.open_function_context('_sqrtm_triu', 26, 0, False)
    
    # Passed parameters checking function
    _sqrtm_triu.stypy_localization = localization
    _sqrtm_triu.stypy_type_of_self = None
    _sqrtm_triu.stypy_type_store = module_type_store
    _sqrtm_triu.stypy_function_name = '_sqrtm_triu'
    _sqrtm_triu.stypy_param_names_list = ['T', 'blocksize']
    _sqrtm_triu.stypy_varargs_param_name = None
    _sqrtm_triu.stypy_kwargs_param_name = None
    _sqrtm_triu.stypy_call_defaults = defaults
    _sqrtm_triu.stypy_call_varargs = varargs
    _sqrtm_triu.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_sqrtm_triu', ['T', 'blocksize'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_sqrtm_triu', localization, ['T', 'blocksize'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_sqrtm_triu(...)' code ##################

    str_35113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, (-1)), 'str', '\n    Matrix square root of an upper triangular matrix.\n\n    This is a helper function for `sqrtm` and `logm`.\n\n    Parameters\n    ----------\n    T : (N, N) array_like upper triangular\n        Matrix whose square root to evaluate\n    blocksize : int, optional\n        If the blocksize is not degenerate with respect to the\n        size of the input array, then use a blocked algorithm. (Default: 64)\n\n    Returns\n    -------\n    sqrtm : (N, N) ndarray\n        Value of the sqrt function at `T`\n\n    References\n    ----------\n    .. [1] Edvin Deadman, Nicholas J. Higham, Rui Ralha (2013)\n           "Blocked Schur Algorithms for Computing the Matrix Square Root,\n           Lecture Notes in Computer Science, 7782. pp. 171-182.\n\n    ')
    
    # Assigning a Call to a Name (line 52):
    
    # Assigning a Call to a Name (line 52):
    
    # Call to diag(...): (line 52)
    # Processing the call arguments (line 52)
    # Getting the type of 'T' (line 52)
    T_35116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 21), 'T', False)
    # Processing the call keyword arguments (line 52)
    kwargs_35117 = {}
    # Getting the type of 'np' (line 52)
    np_35114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 13), 'np', False)
    # Obtaining the member 'diag' of a type (line 52)
    diag_35115 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 13), np_35114, 'diag')
    # Calling diag(args, kwargs) (line 52)
    diag_call_result_35118 = invoke(stypy.reporting.localization.Localization(__file__, 52, 13), diag_35115, *[T_35116], **kwargs_35117)
    
    # Assigning a type to the variable 'T_diag' (line 52)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'T_diag', diag_call_result_35118)
    
    # Assigning a BoolOp to a Name (line 53):
    
    # Assigning a BoolOp to a Name (line 53):
    
    # Evaluating a boolean operation
    
    # Call to isrealobj(...): (line 53)
    # Processing the call arguments (line 53)
    # Getting the type of 'T' (line 53)
    T_35121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 32), 'T', False)
    # Processing the call keyword arguments (line 53)
    kwargs_35122 = {}
    # Getting the type of 'np' (line 53)
    np_35119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 19), 'np', False)
    # Obtaining the member 'isrealobj' of a type (line 53)
    isrealobj_35120 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 19), np_35119, 'isrealobj')
    # Calling isrealobj(args, kwargs) (line 53)
    isrealobj_call_result_35123 = invoke(stypy.reporting.localization.Localization(__file__, 53, 19), isrealobj_35120, *[T_35121], **kwargs_35122)
    
    
    
    # Call to min(...): (line 53)
    # Processing the call arguments (line 53)
    # Getting the type of 'T_diag' (line 53)
    T_diag_35126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 46), 'T_diag', False)
    # Processing the call keyword arguments (line 53)
    kwargs_35127 = {}
    # Getting the type of 'np' (line 53)
    np_35124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 39), 'np', False)
    # Obtaining the member 'min' of a type (line 53)
    min_35125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 39), np_35124, 'min')
    # Calling min(args, kwargs) (line 53)
    min_call_result_35128 = invoke(stypy.reporting.localization.Localization(__file__, 53, 39), min_35125, *[T_diag_35126], **kwargs_35127)
    
    int_35129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 57), 'int')
    # Applying the binary operator '>=' (line 53)
    result_ge_35130 = python_operator(stypy.reporting.localization.Localization(__file__, 53, 39), '>=', min_call_result_35128, int_35129)
    
    # Applying the binary operator 'and' (line 53)
    result_and_keyword_35131 = python_operator(stypy.reporting.localization.Localization(__file__, 53, 19), 'and', isrealobj_call_result_35123, result_ge_35130)
    
    # Assigning a type to the variable 'keep_it_real' (line 53)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'keep_it_real', result_and_keyword_35131)
    
    
    # Getting the type of 'keep_it_real' (line 54)
    keep_it_real_35132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 11), 'keep_it_real')
    # Applying the 'not' unary operator (line 54)
    result_not__35133 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 7), 'not', keep_it_real_35132)
    
    # Testing the type of an if condition (line 54)
    if_condition_35134 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 54, 4), result_not__35133)
    # Assigning a type to the variable 'if_condition_35134' (line 54)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'if_condition_35134', if_condition_35134)
    # SSA begins for if statement (line 54)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 55):
    
    # Assigning a Call to a Name (line 55):
    
    # Call to astype(...): (line 55)
    # Processing the call arguments (line 55)
    # Getting the type of 'complex' (line 55)
    complex_35137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 31), 'complex', False)
    # Processing the call keyword arguments (line 55)
    kwargs_35138 = {}
    # Getting the type of 'T_diag' (line 55)
    T_diag_35135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 17), 'T_diag', False)
    # Obtaining the member 'astype' of a type (line 55)
    astype_35136 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 17), T_diag_35135, 'astype')
    # Calling astype(args, kwargs) (line 55)
    astype_call_result_35139 = invoke(stypy.reporting.localization.Localization(__file__, 55, 17), astype_35136, *[complex_35137], **kwargs_35138)
    
    # Assigning a type to the variable 'T_diag' (line 55)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'T_diag', astype_call_result_35139)
    # SSA join for if statement (line 54)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 56):
    
    # Assigning a Call to a Name (line 56):
    
    # Call to diag(...): (line 56)
    # Processing the call arguments (line 56)
    
    # Call to sqrt(...): (line 56)
    # Processing the call arguments (line 56)
    # Getting the type of 'T_diag' (line 56)
    T_diag_35144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 24), 'T_diag', False)
    # Processing the call keyword arguments (line 56)
    kwargs_35145 = {}
    # Getting the type of 'np' (line 56)
    np_35142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 16), 'np', False)
    # Obtaining the member 'sqrt' of a type (line 56)
    sqrt_35143 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 16), np_35142, 'sqrt')
    # Calling sqrt(args, kwargs) (line 56)
    sqrt_call_result_35146 = invoke(stypy.reporting.localization.Localization(__file__, 56, 16), sqrt_35143, *[T_diag_35144], **kwargs_35145)
    
    # Processing the call keyword arguments (line 56)
    kwargs_35147 = {}
    # Getting the type of 'np' (line 56)
    np_35140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'np', False)
    # Obtaining the member 'diag' of a type (line 56)
    diag_35141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 8), np_35140, 'diag')
    # Calling diag(args, kwargs) (line 56)
    diag_call_result_35148 = invoke(stypy.reporting.localization.Localization(__file__, 56, 8), diag_35141, *[sqrt_call_result_35146], **kwargs_35147)
    
    # Assigning a type to the variable 'R' (line 56)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'R', diag_call_result_35148)
    
    # Assigning a Attribute to a Tuple (line 59):
    
    # Assigning a Subscript to a Name (line 59):
    
    # Obtaining the type of the subscript
    int_35149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 4), 'int')
    # Getting the type of 'T' (line 59)
    T_35150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 11), 'T')
    # Obtaining the member 'shape' of a type (line 59)
    shape_35151 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 11), T_35150, 'shape')
    # Obtaining the member '__getitem__' of a type (line 59)
    getitem___35152 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 4), shape_35151, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 59)
    subscript_call_result_35153 = invoke(stypy.reporting.localization.Localization(__file__, 59, 4), getitem___35152, int_35149)
    
    # Assigning a type to the variable 'tuple_var_assignment_35076' (line 59)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'tuple_var_assignment_35076', subscript_call_result_35153)
    
    # Assigning a Subscript to a Name (line 59):
    
    # Obtaining the type of the subscript
    int_35154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 4), 'int')
    # Getting the type of 'T' (line 59)
    T_35155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 11), 'T')
    # Obtaining the member 'shape' of a type (line 59)
    shape_35156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 11), T_35155, 'shape')
    # Obtaining the member '__getitem__' of a type (line 59)
    getitem___35157 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 4), shape_35156, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 59)
    subscript_call_result_35158 = invoke(stypy.reporting.localization.Localization(__file__, 59, 4), getitem___35157, int_35154)
    
    # Assigning a type to the variable 'tuple_var_assignment_35077' (line 59)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'tuple_var_assignment_35077', subscript_call_result_35158)
    
    # Assigning a Name to a Name (line 59):
    # Getting the type of 'tuple_var_assignment_35076' (line 59)
    tuple_var_assignment_35076_35159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'tuple_var_assignment_35076')
    # Assigning a type to the variable 'n' (line 59)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'n', tuple_var_assignment_35076_35159)
    
    # Assigning a Name to a Name (line 59):
    # Getting the type of 'tuple_var_assignment_35077' (line 59)
    tuple_var_assignment_35077_35160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'tuple_var_assignment_35077')
    # Assigning a type to the variable 'n' (line 59)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 7), 'n', tuple_var_assignment_35077_35160)
    
    # Assigning a Call to a Name (line 60):
    
    # Assigning a Call to a Name (line 60):
    
    # Call to max(...): (line 60)
    # Processing the call arguments (line 60)
    # Getting the type of 'n' (line 60)
    n_35162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 18), 'n', False)
    # Getting the type of 'blocksize' (line 60)
    blocksize_35163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 23), 'blocksize', False)
    # Applying the binary operator '//' (line 60)
    result_floordiv_35164 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 18), '//', n_35162, blocksize_35163)
    
    int_35165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 34), 'int')
    # Processing the call keyword arguments (line 60)
    kwargs_35166 = {}
    # Getting the type of 'max' (line 60)
    max_35161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 14), 'max', False)
    # Calling max(args, kwargs) (line 60)
    max_call_result_35167 = invoke(stypy.reporting.localization.Localization(__file__, 60, 14), max_35161, *[result_floordiv_35164, int_35165], **kwargs_35166)
    
    # Assigning a type to the variable 'nblocks' (line 60)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'nblocks', max_call_result_35167)
    
    # Assigning a Call to a Tuple (line 64):
    
    # Assigning a Subscript to a Name (line 64):
    
    # Obtaining the type of the subscript
    int_35168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 4), 'int')
    
    # Call to divmod(...): (line 64)
    # Processing the call arguments (line 64)
    # Getting the type of 'n' (line 64)
    n_35170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 28), 'n', False)
    # Getting the type of 'nblocks' (line 64)
    nblocks_35171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 31), 'nblocks', False)
    # Processing the call keyword arguments (line 64)
    kwargs_35172 = {}
    # Getting the type of 'divmod' (line 64)
    divmod_35169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 21), 'divmod', False)
    # Calling divmod(args, kwargs) (line 64)
    divmod_call_result_35173 = invoke(stypy.reporting.localization.Localization(__file__, 64, 21), divmod_35169, *[n_35170, nblocks_35171], **kwargs_35172)
    
    # Obtaining the member '__getitem__' of a type (line 64)
    getitem___35174 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 4), divmod_call_result_35173, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 64)
    subscript_call_result_35175 = invoke(stypy.reporting.localization.Localization(__file__, 64, 4), getitem___35174, int_35168)
    
    # Assigning a type to the variable 'tuple_var_assignment_35078' (line 64)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'tuple_var_assignment_35078', subscript_call_result_35175)
    
    # Assigning a Subscript to a Name (line 64):
    
    # Obtaining the type of the subscript
    int_35176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 4), 'int')
    
    # Call to divmod(...): (line 64)
    # Processing the call arguments (line 64)
    # Getting the type of 'n' (line 64)
    n_35178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 28), 'n', False)
    # Getting the type of 'nblocks' (line 64)
    nblocks_35179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 31), 'nblocks', False)
    # Processing the call keyword arguments (line 64)
    kwargs_35180 = {}
    # Getting the type of 'divmod' (line 64)
    divmod_35177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 21), 'divmod', False)
    # Calling divmod(args, kwargs) (line 64)
    divmod_call_result_35181 = invoke(stypy.reporting.localization.Localization(__file__, 64, 21), divmod_35177, *[n_35178, nblocks_35179], **kwargs_35180)
    
    # Obtaining the member '__getitem__' of a type (line 64)
    getitem___35182 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 4), divmod_call_result_35181, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 64)
    subscript_call_result_35183 = invoke(stypy.reporting.localization.Localization(__file__, 64, 4), getitem___35182, int_35176)
    
    # Assigning a type to the variable 'tuple_var_assignment_35079' (line 64)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'tuple_var_assignment_35079', subscript_call_result_35183)
    
    # Assigning a Name to a Name (line 64):
    # Getting the type of 'tuple_var_assignment_35078' (line 64)
    tuple_var_assignment_35078_35184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'tuple_var_assignment_35078')
    # Assigning a type to the variable 'bsmall' (line 64)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'bsmall', tuple_var_assignment_35078_35184)
    
    # Assigning a Name to a Name (line 64):
    # Getting the type of 'tuple_var_assignment_35079' (line 64)
    tuple_var_assignment_35079_35185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'tuple_var_assignment_35079')
    # Assigning a type to the variable 'nlarge' (line 64)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 12), 'nlarge', tuple_var_assignment_35079_35185)
    
    # Assigning a BinOp to a Name (line 65):
    
    # Assigning a BinOp to a Name (line 65):
    # Getting the type of 'bsmall' (line 65)
    bsmall_35186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 13), 'bsmall')
    int_35187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 22), 'int')
    # Applying the binary operator '+' (line 65)
    result_add_35188 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 13), '+', bsmall_35186, int_35187)
    
    # Assigning a type to the variable 'blarge' (line 65)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'blarge', result_add_35188)
    
    # Assigning a BinOp to a Name (line 66):
    
    # Assigning a BinOp to a Name (line 66):
    # Getting the type of 'nblocks' (line 66)
    nblocks_35189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 13), 'nblocks')
    # Getting the type of 'nlarge' (line 66)
    nlarge_35190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 23), 'nlarge')
    # Applying the binary operator '-' (line 66)
    result_sub_35191 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 13), '-', nblocks_35189, nlarge_35190)
    
    # Assigning a type to the variable 'nsmall' (line 66)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'nsmall', result_sub_35191)
    
    
    # Getting the type of 'nsmall' (line 67)
    nsmall_35192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 7), 'nsmall')
    # Getting the type of 'bsmall' (line 67)
    bsmall_35193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 16), 'bsmall')
    # Applying the binary operator '*' (line 67)
    result_mul_35194 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 7), '*', nsmall_35192, bsmall_35193)
    
    # Getting the type of 'nlarge' (line 67)
    nlarge_35195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 25), 'nlarge')
    # Getting the type of 'blarge' (line 67)
    blarge_35196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 34), 'blarge')
    # Applying the binary operator '*' (line 67)
    result_mul_35197 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 25), '*', nlarge_35195, blarge_35196)
    
    # Applying the binary operator '+' (line 67)
    result_add_35198 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 7), '+', result_mul_35194, result_mul_35197)
    
    # Getting the type of 'n' (line 67)
    n_35199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 44), 'n')
    # Applying the binary operator '!=' (line 67)
    result_ne_35200 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 7), '!=', result_add_35198, n_35199)
    
    # Testing the type of an if condition (line 67)
    if_condition_35201 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 67, 4), result_ne_35200)
    # Assigning a type to the variable 'if_condition_35201' (line 67)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'if_condition_35201', if_condition_35201)
    # SSA begins for if statement (line 67)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to Exception(...): (line 68)
    # Processing the call arguments (line 68)
    str_35203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 24), 'str', 'internal inconsistency')
    # Processing the call keyword arguments (line 68)
    kwargs_35204 = {}
    # Getting the type of 'Exception' (line 68)
    Exception_35202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 14), 'Exception', False)
    # Calling Exception(args, kwargs) (line 68)
    Exception_call_result_35205 = invoke(stypy.reporting.localization.Localization(__file__, 68, 14), Exception_35202, *[str_35203], **kwargs_35204)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 68, 8), Exception_call_result_35205, 'raise parameter', BaseException)
    # SSA join for if statement (line 67)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a List to a Name (line 71):
    
    # Assigning a List to a Name (line 71):
    
    # Obtaining an instance of the builtin type 'list' (line 71)
    list_35206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 71)
    
    # Assigning a type to the variable 'start_stop_pairs' (line 71)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 4), 'start_stop_pairs', list_35206)
    
    # Assigning a Num to a Name (line 72):
    
    # Assigning a Num to a Name (line 72):
    int_35207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 12), 'int')
    # Assigning a type to the variable 'start' (line 72)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'start', int_35207)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 73)
    tuple_35208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 24), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 73)
    # Adding element type (line 73)
    
    # Obtaining an instance of the builtin type 'tuple' (line 73)
    tuple_35209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 25), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 73)
    # Adding element type (line 73)
    # Getting the type of 'nsmall' (line 73)
    nsmall_35210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 25), 'nsmall')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 25), tuple_35209, nsmall_35210)
    # Adding element type (line 73)
    # Getting the type of 'bsmall' (line 73)
    bsmall_35211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 33), 'bsmall')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 25), tuple_35209, bsmall_35211)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 24), tuple_35208, tuple_35209)
    # Adding element type (line 73)
    
    # Obtaining an instance of the builtin type 'tuple' (line 73)
    tuple_35212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 43), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 73)
    # Adding element type (line 73)
    # Getting the type of 'nlarge' (line 73)
    nlarge_35213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 43), 'nlarge')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 43), tuple_35212, nlarge_35213)
    # Adding element type (line 73)
    # Getting the type of 'blarge' (line 73)
    blarge_35214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 51), 'blarge')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 43), tuple_35212, blarge_35214)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 24), tuple_35208, tuple_35212)
    
    # Testing the type of a for loop iterable (line 73)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 73, 4), tuple_35208)
    # Getting the type of the for loop variable (line 73)
    for_loop_var_35215 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 73, 4), tuple_35208)
    # Assigning a type to the variable 'count' (line 73)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 4), 'count', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 4), for_loop_var_35215))
    # Assigning a type to the variable 'size' (line 73)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 4), 'size', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 4), for_loop_var_35215))
    # SSA begins for a for statement (line 73)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Call to range(...): (line 74)
    # Processing the call arguments (line 74)
    # Getting the type of 'count' (line 74)
    count_35217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 23), 'count', False)
    # Processing the call keyword arguments (line 74)
    kwargs_35218 = {}
    # Getting the type of 'range' (line 74)
    range_35216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 17), 'range', False)
    # Calling range(args, kwargs) (line 74)
    range_call_result_35219 = invoke(stypy.reporting.localization.Localization(__file__, 74, 17), range_35216, *[count_35217], **kwargs_35218)
    
    # Testing the type of a for loop iterable (line 74)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 74, 8), range_call_result_35219)
    # Getting the type of the for loop variable (line 74)
    for_loop_var_35220 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 74, 8), range_call_result_35219)
    # Assigning a type to the variable 'i' (line 74)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'i', for_loop_var_35220)
    # SSA begins for a for statement (line 74)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to append(...): (line 75)
    # Processing the call arguments (line 75)
    
    # Obtaining an instance of the builtin type 'tuple' (line 75)
    tuple_35223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 37), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 75)
    # Adding element type (line 75)
    # Getting the type of 'start' (line 75)
    start_35224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 37), 'start', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 37), tuple_35223, start_35224)
    # Adding element type (line 75)
    # Getting the type of 'start' (line 75)
    start_35225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 44), 'start', False)
    # Getting the type of 'size' (line 75)
    size_35226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 52), 'size', False)
    # Applying the binary operator '+' (line 75)
    result_add_35227 = python_operator(stypy.reporting.localization.Localization(__file__, 75, 44), '+', start_35225, size_35226)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 37), tuple_35223, result_add_35227)
    
    # Processing the call keyword arguments (line 75)
    kwargs_35228 = {}
    # Getting the type of 'start_stop_pairs' (line 75)
    start_stop_pairs_35221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 12), 'start_stop_pairs', False)
    # Obtaining the member 'append' of a type (line 75)
    append_35222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 12), start_stop_pairs_35221, 'append')
    # Calling append(args, kwargs) (line 75)
    append_call_result_35229 = invoke(stypy.reporting.localization.Localization(__file__, 75, 12), append_35222, *[tuple_35223], **kwargs_35228)
    
    
    # Getting the type of 'start' (line 76)
    start_35230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'start')
    # Getting the type of 'size' (line 76)
    size_35231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 21), 'size')
    # Applying the binary operator '+=' (line 76)
    result_iadd_35232 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 12), '+=', start_35230, size_35231)
    # Assigning a type to the variable 'start' (line 76)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'start', result_iadd_35232)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'start_stop_pairs' (line 79)
    start_stop_pairs_35233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 23), 'start_stop_pairs')
    # Testing the type of a for loop iterable (line 79)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 79, 4), start_stop_pairs_35233)
    # Getting the type of the for loop variable (line 79)
    for_loop_var_35234 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 79, 4), start_stop_pairs_35233)
    # Assigning a type to the variable 'start' (line 79)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'start', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 4), for_loop_var_35234))
    # Assigning a type to the variable 'stop' (line 79)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'stop', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 4), for_loop_var_35234))
    # SSA begins for a for statement (line 79)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Call to range(...): (line 80)
    # Processing the call arguments (line 80)
    # Getting the type of 'start' (line 80)
    start_35236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 23), 'start', False)
    # Getting the type of 'stop' (line 80)
    stop_35237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 30), 'stop', False)
    # Processing the call keyword arguments (line 80)
    kwargs_35238 = {}
    # Getting the type of 'range' (line 80)
    range_35235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 17), 'range', False)
    # Calling range(args, kwargs) (line 80)
    range_call_result_35239 = invoke(stypy.reporting.localization.Localization(__file__, 80, 17), range_35235, *[start_35236, stop_35237], **kwargs_35238)
    
    # Testing the type of a for loop iterable (line 80)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 80, 8), range_call_result_35239)
    # Getting the type of the for loop variable (line 80)
    for_loop_var_35240 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 80, 8), range_call_result_35239)
    # Assigning a type to the variable 'j' (line 80)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'j', for_loop_var_35240)
    # SSA begins for a for statement (line 80)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Call to range(...): (line 81)
    # Processing the call arguments (line 81)
    # Getting the type of 'j' (line 81)
    j_35242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 27), 'j', False)
    int_35243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 29), 'int')
    # Applying the binary operator '-' (line 81)
    result_sub_35244 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 27), '-', j_35242, int_35243)
    
    # Getting the type of 'start' (line 81)
    start_35245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 32), 'start', False)
    int_35246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 38), 'int')
    # Applying the binary operator '-' (line 81)
    result_sub_35247 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 32), '-', start_35245, int_35246)
    
    int_35248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 41), 'int')
    # Processing the call keyword arguments (line 81)
    kwargs_35249 = {}
    # Getting the type of 'range' (line 81)
    range_35241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 21), 'range', False)
    # Calling range(args, kwargs) (line 81)
    range_call_result_35250 = invoke(stypy.reporting.localization.Localization(__file__, 81, 21), range_35241, *[result_sub_35244, result_sub_35247, int_35248], **kwargs_35249)
    
    # Testing the type of a for loop iterable (line 81)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 81, 12), range_call_result_35250)
    # Getting the type of the for loop variable (line 81)
    for_loop_var_35251 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 81, 12), range_call_result_35250)
    # Assigning a type to the variable 'i' (line 81)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 12), 'i', for_loop_var_35251)
    # SSA begins for a for statement (line 81)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Num to a Name (line 82):
    
    # Assigning a Num to a Name (line 82):
    int_35252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 20), 'int')
    # Assigning a type to the variable 's' (line 82)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 16), 's', int_35252)
    
    
    # Getting the type of 'j' (line 83)
    j_35253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 19), 'j')
    # Getting the type of 'i' (line 83)
    i_35254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 23), 'i')
    # Applying the binary operator '-' (line 83)
    result_sub_35255 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 19), '-', j_35253, i_35254)
    
    int_35256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 27), 'int')
    # Applying the binary operator '>' (line 83)
    result_gt_35257 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 19), '>', result_sub_35255, int_35256)
    
    # Testing the type of an if condition (line 83)
    if_condition_35258 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 83, 16), result_gt_35257)
    # Assigning a type to the variable 'if_condition_35258' (line 83)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 16), 'if_condition_35258', if_condition_35258)
    # SSA begins for if statement (line 83)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 84):
    
    # Assigning a Call to a Name (line 84):
    
    # Call to dot(...): (line 84)
    # Processing the call arguments (line 84)
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 84)
    i_35269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 42), 'i', False)
    int_35270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 44), 'int')
    # Applying the binary operator '+' (line 84)
    result_add_35271 = python_operator(stypy.reporting.localization.Localization(__file__, 84, 42), '+', i_35269, int_35270)
    
    # Getting the type of 'j' (line 84)
    j_35272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 46), 'j', False)
    slice_35273 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 84, 40), result_add_35271, j_35272, None)
    # Getting the type of 'j' (line 84)
    j_35274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 49), 'j', False)
    # Getting the type of 'R' (line 84)
    R_35275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 40), 'R', False)
    # Obtaining the member '__getitem__' of a type (line 84)
    getitem___35276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 40), R_35275, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 84)
    subscript_call_result_35277 = invoke(stypy.reporting.localization.Localization(__file__, 84, 40), getitem___35276, (slice_35273, j_35274))
    
    # Processing the call keyword arguments (line 84)
    kwargs_35278 = {}
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 84)
    i_35259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 26), 'i', False)
    # Getting the type of 'i' (line 84)
    i_35260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 29), 'i', False)
    int_35261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 31), 'int')
    # Applying the binary operator '+' (line 84)
    result_add_35262 = python_operator(stypy.reporting.localization.Localization(__file__, 84, 29), '+', i_35260, int_35261)
    
    # Getting the type of 'j' (line 84)
    j_35263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 33), 'j', False)
    slice_35264 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 84, 24), result_add_35262, j_35263, None)
    # Getting the type of 'R' (line 84)
    R_35265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 24), 'R', False)
    # Obtaining the member '__getitem__' of a type (line 84)
    getitem___35266 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 24), R_35265, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 84)
    subscript_call_result_35267 = invoke(stypy.reporting.localization.Localization(__file__, 84, 24), getitem___35266, (i_35259, slice_35264))
    
    # Obtaining the member 'dot' of a type (line 84)
    dot_35268 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 24), subscript_call_result_35267, 'dot')
    # Calling dot(args, kwargs) (line 84)
    dot_call_result_35279 = invoke(stypy.reporting.localization.Localization(__file__, 84, 24), dot_35268, *[subscript_call_result_35277], **kwargs_35278)
    
    # Assigning a type to the variable 's' (line 84)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 20), 's', dot_call_result_35279)
    # SSA join for if statement (line 83)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 85):
    
    # Assigning a BinOp to a Name (line 85):
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 85)
    tuple_35280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 26), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 85)
    # Adding element type (line 85)
    # Getting the type of 'i' (line 85)
    i_35281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 26), 'i')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 26), tuple_35280, i_35281)
    # Adding element type (line 85)
    # Getting the type of 'i' (line 85)
    i_35282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 29), 'i')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 26), tuple_35280, i_35282)
    
    # Getting the type of 'R' (line 85)
    R_35283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 24), 'R')
    # Obtaining the member '__getitem__' of a type (line 85)
    getitem___35284 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 24), R_35283, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 85)
    subscript_call_result_35285 = invoke(stypy.reporting.localization.Localization(__file__, 85, 24), getitem___35284, tuple_35280)
    
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 85)
    tuple_35286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 36), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 85)
    # Adding element type (line 85)
    # Getting the type of 'j' (line 85)
    j_35287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 36), 'j')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 36), tuple_35286, j_35287)
    # Adding element type (line 85)
    # Getting the type of 'j' (line 85)
    j_35288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 39), 'j')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 36), tuple_35286, j_35288)
    
    # Getting the type of 'R' (line 85)
    R_35289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 34), 'R')
    # Obtaining the member '__getitem__' of a type (line 85)
    getitem___35290 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 34), R_35289, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 85)
    subscript_call_result_35291 = invoke(stypy.reporting.localization.Localization(__file__, 85, 34), getitem___35290, tuple_35286)
    
    # Applying the binary operator '+' (line 85)
    result_add_35292 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 24), '+', subscript_call_result_35285, subscript_call_result_35291)
    
    # Assigning a type to the variable 'denom' (line 85)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 16), 'denom', result_add_35292)
    
    
    # Getting the type of 'denom' (line 86)
    denom_35293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 23), 'denom')
    # Applying the 'not' unary operator (line 86)
    result_not__35294 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 19), 'not', denom_35293)
    
    # Testing the type of an if condition (line 86)
    if_condition_35295 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 86, 16), result_not__35294)
    # Assigning a type to the variable 'if_condition_35295' (line 86)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 16), 'if_condition_35295', if_condition_35295)
    # SSA begins for if statement (line 86)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to SqrtmError(...): (line 87)
    # Processing the call arguments (line 87)
    str_35297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 37), 'str', 'failed to find the matrix square root')
    # Processing the call keyword arguments (line 87)
    kwargs_35298 = {}
    # Getting the type of 'SqrtmError' (line 87)
    SqrtmError_35296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 26), 'SqrtmError', False)
    # Calling SqrtmError(args, kwargs) (line 87)
    SqrtmError_call_result_35299 = invoke(stypy.reporting.localization.Localization(__file__, 87, 26), SqrtmError_35296, *[str_35297], **kwargs_35298)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 87, 20), SqrtmError_call_result_35299, 'raise parameter', BaseException)
    # SSA join for if statement (line 86)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Subscript (line 88):
    
    # Assigning a BinOp to a Subscript (line 88):
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 88)
    tuple_35300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 29), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 88)
    # Adding element type (line 88)
    # Getting the type of 'i' (line 88)
    i_35301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 29), 'i')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 29), tuple_35300, i_35301)
    # Adding element type (line 88)
    # Getting the type of 'j' (line 88)
    j_35302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 32), 'j')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 29), tuple_35300, j_35302)
    
    # Getting the type of 'T' (line 88)
    T_35303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 27), 'T')
    # Obtaining the member '__getitem__' of a type (line 88)
    getitem___35304 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 27), T_35303, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 88)
    subscript_call_result_35305 = invoke(stypy.reporting.localization.Localization(__file__, 88, 27), getitem___35304, tuple_35300)
    
    # Getting the type of 's' (line 88)
    s_35306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 37), 's')
    # Applying the binary operator '-' (line 88)
    result_sub_35307 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 27), '-', subscript_call_result_35305, s_35306)
    
    # Getting the type of 'denom' (line 88)
    denom_35308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 42), 'denom')
    # Applying the binary operator 'div' (line 88)
    result_div_35309 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 26), 'div', result_sub_35307, denom_35308)
    
    # Getting the type of 'R' (line 88)
    R_35310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 16), 'R')
    
    # Obtaining an instance of the builtin type 'tuple' (line 88)
    tuple_35311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 18), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 88)
    # Adding element type (line 88)
    # Getting the type of 'i' (line 88)
    i_35312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 18), 'i')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 18), tuple_35311, i_35312)
    # Adding element type (line 88)
    # Getting the type of 'j' (line 88)
    j_35313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 21), 'j')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 18), tuple_35311, j_35313)
    
    # Storing an element on a container (line 88)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 16), R_35310, (tuple_35311, result_div_35309))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to range(...): (line 91)
    # Processing the call arguments (line 91)
    # Getting the type of 'nblocks' (line 91)
    nblocks_35315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 19), 'nblocks', False)
    # Processing the call keyword arguments (line 91)
    kwargs_35316 = {}
    # Getting the type of 'range' (line 91)
    range_35314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 13), 'range', False)
    # Calling range(args, kwargs) (line 91)
    range_call_result_35317 = invoke(stypy.reporting.localization.Localization(__file__, 91, 13), range_35314, *[nblocks_35315], **kwargs_35316)
    
    # Testing the type of a for loop iterable (line 91)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 91, 4), range_call_result_35317)
    # Getting the type of the for loop variable (line 91)
    for_loop_var_35318 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 91, 4), range_call_result_35317)
    # Assigning a type to the variable 'j' (line 91)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 4), 'j', for_loop_var_35318)
    # SSA begins for a for statement (line 91)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Subscript to a Tuple (line 92):
    
    # Assigning a Subscript to a Name (line 92):
    
    # Obtaining the type of the subscript
    int_35319 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 8), 'int')
    
    # Obtaining the type of the subscript
    # Getting the type of 'j' (line 92)
    j_35320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 41), 'j')
    # Getting the type of 'start_stop_pairs' (line 92)
    start_stop_pairs_35321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 24), 'start_stop_pairs')
    # Obtaining the member '__getitem__' of a type (line 92)
    getitem___35322 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 24), start_stop_pairs_35321, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 92)
    subscript_call_result_35323 = invoke(stypy.reporting.localization.Localization(__file__, 92, 24), getitem___35322, j_35320)
    
    # Obtaining the member '__getitem__' of a type (line 92)
    getitem___35324 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 8), subscript_call_result_35323, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 92)
    subscript_call_result_35325 = invoke(stypy.reporting.localization.Localization(__file__, 92, 8), getitem___35324, int_35319)
    
    # Assigning a type to the variable 'tuple_var_assignment_35080' (line 92)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'tuple_var_assignment_35080', subscript_call_result_35325)
    
    # Assigning a Subscript to a Name (line 92):
    
    # Obtaining the type of the subscript
    int_35326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 8), 'int')
    
    # Obtaining the type of the subscript
    # Getting the type of 'j' (line 92)
    j_35327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 41), 'j')
    # Getting the type of 'start_stop_pairs' (line 92)
    start_stop_pairs_35328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 24), 'start_stop_pairs')
    # Obtaining the member '__getitem__' of a type (line 92)
    getitem___35329 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 24), start_stop_pairs_35328, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 92)
    subscript_call_result_35330 = invoke(stypy.reporting.localization.Localization(__file__, 92, 24), getitem___35329, j_35327)
    
    # Obtaining the member '__getitem__' of a type (line 92)
    getitem___35331 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 8), subscript_call_result_35330, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 92)
    subscript_call_result_35332 = invoke(stypy.reporting.localization.Localization(__file__, 92, 8), getitem___35331, int_35326)
    
    # Assigning a type to the variable 'tuple_var_assignment_35081' (line 92)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'tuple_var_assignment_35081', subscript_call_result_35332)
    
    # Assigning a Name to a Name (line 92):
    # Getting the type of 'tuple_var_assignment_35080' (line 92)
    tuple_var_assignment_35080_35333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'tuple_var_assignment_35080')
    # Assigning a type to the variable 'jstart' (line 92)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'jstart', tuple_var_assignment_35080_35333)
    
    # Assigning a Name to a Name (line 92):
    # Getting the type of 'tuple_var_assignment_35081' (line 92)
    tuple_var_assignment_35081_35334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'tuple_var_assignment_35081')
    # Assigning a type to the variable 'jstop' (line 92)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 16), 'jstop', tuple_var_assignment_35081_35334)
    
    
    # Call to range(...): (line 93)
    # Processing the call arguments (line 93)
    # Getting the type of 'j' (line 93)
    j_35336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 23), 'j', False)
    int_35337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 25), 'int')
    # Applying the binary operator '-' (line 93)
    result_sub_35338 = python_operator(stypy.reporting.localization.Localization(__file__, 93, 23), '-', j_35336, int_35337)
    
    int_35339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 28), 'int')
    int_35340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 32), 'int')
    # Processing the call keyword arguments (line 93)
    kwargs_35341 = {}
    # Getting the type of 'range' (line 93)
    range_35335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 17), 'range', False)
    # Calling range(args, kwargs) (line 93)
    range_call_result_35342 = invoke(stypy.reporting.localization.Localization(__file__, 93, 17), range_35335, *[result_sub_35338, int_35339, int_35340], **kwargs_35341)
    
    # Testing the type of a for loop iterable (line 93)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 93, 8), range_call_result_35342)
    # Getting the type of the for loop variable (line 93)
    for_loop_var_35343 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 93, 8), range_call_result_35342)
    # Assigning a type to the variable 'i' (line 93)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'i', for_loop_var_35343)
    # SSA begins for a for statement (line 93)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Subscript to a Tuple (line 94):
    
    # Assigning a Subscript to a Name (line 94):
    
    # Obtaining the type of the subscript
    int_35344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 12), 'int')
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 94)
    i_35345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 45), 'i')
    # Getting the type of 'start_stop_pairs' (line 94)
    start_stop_pairs_35346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 28), 'start_stop_pairs')
    # Obtaining the member '__getitem__' of a type (line 94)
    getitem___35347 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 28), start_stop_pairs_35346, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 94)
    subscript_call_result_35348 = invoke(stypy.reporting.localization.Localization(__file__, 94, 28), getitem___35347, i_35345)
    
    # Obtaining the member '__getitem__' of a type (line 94)
    getitem___35349 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 12), subscript_call_result_35348, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 94)
    subscript_call_result_35350 = invoke(stypy.reporting.localization.Localization(__file__, 94, 12), getitem___35349, int_35344)
    
    # Assigning a type to the variable 'tuple_var_assignment_35082' (line 94)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 12), 'tuple_var_assignment_35082', subscript_call_result_35350)
    
    # Assigning a Subscript to a Name (line 94):
    
    # Obtaining the type of the subscript
    int_35351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 12), 'int')
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 94)
    i_35352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 45), 'i')
    # Getting the type of 'start_stop_pairs' (line 94)
    start_stop_pairs_35353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 28), 'start_stop_pairs')
    # Obtaining the member '__getitem__' of a type (line 94)
    getitem___35354 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 28), start_stop_pairs_35353, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 94)
    subscript_call_result_35355 = invoke(stypy.reporting.localization.Localization(__file__, 94, 28), getitem___35354, i_35352)
    
    # Obtaining the member '__getitem__' of a type (line 94)
    getitem___35356 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 12), subscript_call_result_35355, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 94)
    subscript_call_result_35357 = invoke(stypy.reporting.localization.Localization(__file__, 94, 12), getitem___35356, int_35351)
    
    # Assigning a type to the variable 'tuple_var_assignment_35083' (line 94)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 12), 'tuple_var_assignment_35083', subscript_call_result_35357)
    
    # Assigning a Name to a Name (line 94):
    # Getting the type of 'tuple_var_assignment_35082' (line 94)
    tuple_var_assignment_35082_35358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 12), 'tuple_var_assignment_35082')
    # Assigning a type to the variable 'istart' (line 94)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 12), 'istart', tuple_var_assignment_35082_35358)
    
    # Assigning a Name to a Name (line 94):
    # Getting the type of 'tuple_var_assignment_35083' (line 94)
    tuple_var_assignment_35083_35359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 12), 'tuple_var_assignment_35083')
    # Assigning a type to the variable 'istop' (line 94)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 20), 'istop', tuple_var_assignment_35083_35359)
    
    # Assigning a Subscript to a Name (line 95):
    
    # Assigning a Subscript to a Name (line 95):
    
    # Obtaining the type of the subscript
    # Getting the type of 'istart' (line 95)
    istart_35360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 18), 'istart')
    # Getting the type of 'istop' (line 95)
    istop_35361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 25), 'istop')
    slice_35362 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 95, 16), istart_35360, istop_35361, None)
    # Getting the type of 'jstart' (line 95)
    jstart_35363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 32), 'jstart')
    # Getting the type of 'jstop' (line 95)
    jstop_35364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 39), 'jstop')
    slice_35365 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 95, 16), jstart_35363, jstop_35364, None)
    # Getting the type of 'T' (line 95)
    T_35366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 16), 'T')
    # Obtaining the member '__getitem__' of a type (line 95)
    getitem___35367 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 16), T_35366, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 95)
    subscript_call_result_35368 = invoke(stypy.reporting.localization.Localization(__file__, 95, 16), getitem___35367, (slice_35362, slice_35365))
    
    # Assigning a type to the variable 'S' (line 95)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 12), 'S', subscript_call_result_35368)
    
    
    # Getting the type of 'j' (line 96)
    j_35369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 15), 'j')
    # Getting the type of 'i' (line 96)
    i_35370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 19), 'i')
    # Applying the binary operator '-' (line 96)
    result_sub_35371 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 15), '-', j_35369, i_35370)
    
    int_35372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 23), 'int')
    # Applying the binary operator '>' (line 96)
    result_gt_35373 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 15), '>', result_sub_35371, int_35372)
    
    # Testing the type of an if condition (line 96)
    if_condition_35374 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 96, 12), result_gt_35373)
    # Assigning a type to the variable 'if_condition_35374' (line 96)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 12), 'if_condition_35374', if_condition_35374)
    # SSA begins for if statement (line 96)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 97):
    
    # Assigning a BinOp to a Name (line 97):
    # Getting the type of 'S' (line 97)
    S_35375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 20), 'S')
    
    # Call to dot(...): (line 97)
    # Processing the call arguments (line 97)
    
    # Obtaining the type of the subscript
    # Getting the type of 'istop' (line 97)
    istop_35386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 60), 'istop', False)
    # Getting the type of 'jstart' (line 97)
    jstart_35387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 66), 'jstart', False)
    slice_35388 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 97, 58), istop_35386, jstart_35387, None)
    # Getting the type of 'jstart' (line 98)
    jstart_35389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 60), 'jstart', False)
    # Getting the type of 'jstop' (line 98)
    jstop_35390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 67), 'jstop', False)
    slice_35391 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 97, 58), jstart_35389, jstop_35390, None)
    # Getting the type of 'R' (line 97)
    R_35392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 58), 'R', False)
    # Obtaining the member '__getitem__' of a type (line 97)
    getitem___35393 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 58), R_35392, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 97)
    subscript_call_result_35394 = invoke(stypy.reporting.localization.Localization(__file__, 97, 58), getitem___35393, (slice_35388, slice_35391))
    
    # Processing the call keyword arguments (line 97)
    kwargs_35395 = {}
    
    # Obtaining the type of the subscript
    # Getting the type of 'istart' (line 97)
    istart_35376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 26), 'istart', False)
    # Getting the type of 'istop' (line 97)
    istop_35377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 33), 'istop', False)
    slice_35378 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 97, 24), istart_35376, istop_35377, None)
    # Getting the type of 'istop' (line 97)
    istop_35379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 40), 'istop', False)
    # Getting the type of 'jstart' (line 97)
    jstart_35380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 46), 'jstart', False)
    slice_35381 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 97, 24), istop_35379, jstart_35380, None)
    # Getting the type of 'R' (line 97)
    R_35382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 24), 'R', False)
    # Obtaining the member '__getitem__' of a type (line 97)
    getitem___35383 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 24), R_35382, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 97)
    subscript_call_result_35384 = invoke(stypy.reporting.localization.Localization(__file__, 97, 24), getitem___35383, (slice_35378, slice_35381))
    
    # Obtaining the member 'dot' of a type (line 97)
    dot_35385 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 24), subscript_call_result_35384, 'dot')
    # Calling dot(args, kwargs) (line 97)
    dot_call_result_35396 = invoke(stypy.reporting.localization.Localization(__file__, 97, 24), dot_35385, *[subscript_call_result_35394], **kwargs_35395)
    
    # Applying the binary operator '-' (line 97)
    result_sub_35397 = python_operator(stypy.reporting.localization.Localization(__file__, 97, 20), '-', S_35375, dot_call_result_35396)
    
    # Assigning a type to the variable 'S' (line 97)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 16), 'S', result_sub_35397)
    # SSA join for if statement (line 96)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Subscript to a Name (line 103):
    
    # Assigning a Subscript to a Name (line 103):
    
    # Obtaining the type of the subscript
    # Getting the type of 'istart' (line 103)
    istart_35398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 20), 'istart')
    # Getting the type of 'istop' (line 103)
    istop_35399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 27), 'istop')
    slice_35400 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 103, 18), istart_35398, istop_35399, None)
    # Getting the type of 'istart' (line 103)
    istart_35401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 34), 'istart')
    # Getting the type of 'istop' (line 103)
    istop_35402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 41), 'istop')
    slice_35403 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 103, 18), istart_35401, istop_35402, None)
    # Getting the type of 'R' (line 103)
    R_35404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 18), 'R')
    # Obtaining the member '__getitem__' of a type (line 103)
    getitem___35405 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 18), R_35404, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 103)
    subscript_call_result_35406 = invoke(stypy.reporting.localization.Localization(__file__, 103, 18), getitem___35405, (slice_35400, slice_35403))
    
    # Assigning a type to the variable 'Rii' (line 103)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 12), 'Rii', subscript_call_result_35406)
    
    # Assigning a Subscript to a Name (line 104):
    
    # Assigning a Subscript to a Name (line 104):
    
    # Obtaining the type of the subscript
    # Getting the type of 'jstart' (line 104)
    jstart_35407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 20), 'jstart')
    # Getting the type of 'jstop' (line 104)
    jstop_35408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 27), 'jstop')
    slice_35409 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 104, 18), jstart_35407, jstop_35408, None)
    # Getting the type of 'jstart' (line 104)
    jstart_35410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 34), 'jstart')
    # Getting the type of 'jstop' (line 104)
    jstop_35411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 41), 'jstop')
    slice_35412 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 104, 18), jstart_35410, jstop_35411, None)
    # Getting the type of 'R' (line 104)
    R_35413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 18), 'R')
    # Obtaining the member '__getitem__' of a type (line 104)
    getitem___35414 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 18), R_35413, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 104)
    subscript_call_result_35415 = invoke(stypy.reporting.localization.Localization(__file__, 104, 18), getitem___35414, (slice_35409, slice_35412))
    
    # Assigning a type to the variable 'Rjj' (line 104)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 12), 'Rjj', subscript_call_result_35415)
    
    # Getting the type of 'keep_it_real' (line 105)
    keep_it_real_35416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 15), 'keep_it_real')
    # Testing the type of an if condition (line 105)
    if_condition_35417 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 105, 12), keep_it_real_35416)
    # Assigning a type to the variable 'if_condition_35417' (line 105)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 12), 'if_condition_35417', if_condition_35417)
    # SSA begins for if statement (line 105)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 106):
    
    # Assigning a Subscript to a Name (line 106):
    
    # Obtaining the type of the subscript
    int_35418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 16), 'int')
    
    # Call to dtrsyl(...): (line 106)
    # Processing the call arguments (line 106)
    # Getting the type of 'Rii' (line 106)
    Rii_35420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 40), 'Rii', False)
    # Getting the type of 'Rjj' (line 106)
    Rjj_35421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 45), 'Rjj', False)
    # Getting the type of 'S' (line 106)
    S_35422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 50), 'S', False)
    # Processing the call keyword arguments (line 106)
    kwargs_35423 = {}
    # Getting the type of 'dtrsyl' (line 106)
    dtrsyl_35419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 33), 'dtrsyl', False)
    # Calling dtrsyl(args, kwargs) (line 106)
    dtrsyl_call_result_35424 = invoke(stypy.reporting.localization.Localization(__file__, 106, 33), dtrsyl_35419, *[Rii_35420, Rjj_35421, S_35422], **kwargs_35423)
    
    # Obtaining the member '__getitem__' of a type (line 106)
    getitem___35425 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 16), dtrsyl_call_result_35424, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 106)
    subscript_call_result_35426 = invoke(stypy.reporting.localization.Localization(__file__, 106, 16), getitem___35425, int_35418)
    
    # Assigning a type to the variable 'tuple_var_assignment_35084' (line 106)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 16), 'tuple_var_assignment_35084', subscript_call_result_35426)
    
    # Assigning a Subscript to a Name (line 106):
    
    # Obtaining the type of the subscript
    int_35427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 16), 'int')
    
    # Call to dtrsyl(...): (line 106)
    # Processing the call arguments (line 106)
    # Getting the type of 'Rii' (line 106)
    Rii_35429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 40), 'Rii', False)
    # Getting the type of 'Rjj' (line 106)
    Rjj_35430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 45), 'Rjj', False)
    # Getting the type of 'S' (line 106)
    S_35431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 50), 'S', False)
    # Processing the call keyword arguments (line 106)
    kwargs_35432 = {}
    # Getting the type of 'dtrsyl' (line 106)
    dtrsyl_35428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 33), 'dtrsyl', False)
    # Calling dtrsyl(args, kwargs) (line 106)
    dtrsyl_call_result_35433 = invoke(stypy.reporting.localization.Localization(__file__, 106, 33), dtrsyl_35428, *[Rii_35429, Rjj_35430, S_35431], **kwargs_35432)
    
    # Obtaining the member '__getitem__' of a type (line 106)
    getitem___35434 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 16), dtrsyl_call_result_35433, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 106)
    subscript_call_result_35435 = invoke(stypy.reporting.localization.Localization(__file__, 106, 16), getitem___35434, int_35427)
    
    # Assigning a type to the variable 'tuple_var_assignment_35085' (line 106)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 16), 'tuple_var_assignment_35085', subscript_call_result_35435)
    
    # Assigning a Subscript to a Name (line 106):
    
    # Obtaining the type of the subscript
    int_35436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 16), 'int')
    
    # Call to dtrsyl(...): (line 106)
    # Processing the call arguments (line 106)
    # Getting the type of 'Rii' (line 106)
    Rii_35438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 40), 'Rii', False)
    # Getting the type of 'Rjj' (line 106)
    Rjj_35439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 45), 'Rjj', False)
    # Getting the type of 'S' (line 106)
    S_35440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 50), 'S', False)
    # Processing the call keyword arguments (line 106)
    kwargs_35441 = {}
    # Getting the type of 'dtrsyl' (line 106)
    dtrsyl_35437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 33), 'dtrsyl', False)
    # Calling dtrsyl(args, kwargs) (line 106)
    dtrsyl_call_result_35442 = invoke(stypy.reporting.localization.Localization(__file__, 106, 33), dtrsyl_35437, *[Rii_35438, Rjj_35439, S_35440], **kwargs_35441)
    
    # Obtaining the member '__getitem__' of a type (line 106)
    getitem___35443 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 16), dtrsyl_call_result_35442, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 106)
    subscript_call_result_35444 = invoke(stypy.reporting.localization.Localization(__file__, 106, 16), getitem___35443, int_35436)
    
    # Assigning a type to the variable 'tuple_var_assignment_35086' (line 106)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 16), 'tuple_var_assignment_35086', subscript_call_result_35444)
    
    # Assigning a Name to a Name (line 106):
    # Getting the type of 'tuple_var_assignment_35084' (line 106)
    tuple_var_assignment_35084_35445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 16), 'tuple_var_assignment_35084')
    # Assigning a type to the variable 'x' (line 106)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 16), 'x', tuple_var_assignment_35084_35445)
    
    # Assigning a Name to a Name (line 106):
    # Getting the type of 'tuple_var_assignment_35085' (line 106)
    tuple_var_assignment_35085_35446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 16), 'tuple_var_assignment_35085')
    # Assigning a type to the variable 'scale' (line 106)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 19), 'scale', tuple_var_assignment_35085_35446)
    
    # Assigning a Name to a Name (line 106):
    # Getting the type of 'tuple_var_assignment_35086' (line 106)
    tuple_var_assignment_35086_35447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 16), 'tuple_var_assignment_35086')
    # Assigning a type to the variable 'info' (line 106)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 26), 'info', tuple_var_assignment_35086_35447)
    # SSA branch for the else part of an if statement (line 105)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Tuple (line 108):
    
    # Assigning a Subscript to a Name (line 108):
    
    # Obtaining the type of the subscript
    int_35448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 16), 'int')
    
    # Call to ztrsyl(...): (line 108)
    # Processing the call arguments (line 108)
    # Getting the type of 'Rii' (line 108)
    Rii_35450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 40), 'Rii', False)
    # Getting the type of 'Rjj' (line 108)
    Rjj_35451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 45), 'Rjj', False)
    # Getting the type of 'S' (line 108)
    S_35452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 50), 'S', False)
    # Processing the call keyword arguments (line 108)
    kwargs_35453 = {}
    # Getting the type of 'ztrsyl' (line 108)
    ztrsyl_35449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 33), 'ztrsyl', False)
    # Calling ztrsyl(args, kwargs) (line 108)
    ztrsyl_call_result_35454 = invoke(stypy.reporting.localization.Localization(__file__, 108, 33), ztrsyl_35449, *[Rii_35450, Rjj_35451, S_35452], **kwargs_35453)
    
    # Obtaining the member '__getitem__' of a type (line 108)
    getitem___35455 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 16), ztrsyl_call_result_35454, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 108)
    subscript_call_result_35456 = invoke(stypy.reporting.localization.Localization(__file__, 108, 16), getitem___35455, int_35448)
    
    # Assigning a type to the variable 'tuple_var_assignment_35087' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 16), 'tuple_var_assignment_35087', subscript_call_result_35456)
    
    # Assigning a Subscript to a Name (line 108):
    
    # Obtaining the type of the subscript
    int_35457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 16), 'int')
    
    # Call to ztrsyl(...): (line 108)
    # Processing the call arguments (line 108)
    # Getting the type of 'Rii' (line 108)
    Rii_35459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 40), 'Rii', False)
    # Getting the type of 'Rjj' (line 108)
    Rjj_35460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 45), 'Rjj', False)
    # Getting the type of 'S' (line 108)
    S_35461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 50), 'S', False)
    # Processing the call keyword arguments (line 108)
    kwargs_35462 = {}
    # Getting the type of 'ztrsyl' (line 108)
    ztrsyl_35458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 33), 'ztrsyl', False)
    # Calling ztrsyl(args, kwargs) (line 108)
    ztrsyl_call_result_35463 = invoke(stypy.reporting.localization.Localization(__file__, 108, 33), ztrsyl_35458, *[Rii_35459, Rjj_35460, S_35461], **kwargs_35462)
    
    # Obtaining the member '__getitem__' of a type (line 108)
    getitem___35464 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 16), ztrsyl_call_result_35463, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 108)
    subscript_call_result_35465 = invoke(stypy.reporting.localization.Localization(__file__, 108, 16), getitem___35464, int_35457)
    
    # Assigning a type to the variable 'tuple_var_assignment_35088' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 16), 'tuple_var_assignment_35088', subscript_call_result_35465)
    
    # Assigning a Subscript to a Name (line 108):
    
    # Obtaining the type of the subscript
    int_35466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 16), 'int')
    
    # Call to ztrsyl(...): (line 108)
    # Processing the call arguments (line 108)
    # Getting the type of 'Rii' (line 108)
    Rii_35468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 40), 'Rii', False)
    # Getting the type of 'Rjj' (line 108)
    Rjj_35469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 45), 'Rjj', False)
    # Getting the type of 'S' (line 108)
    S_35470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 50), 'S', False)
    # Processing the call keyword arguments (line 108)
    kwargs_35471 = {}
    # Getting the type of 'ztrsyl' (line 108)
    ztrsyl_35467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 33), 'ztrsyl', False)
    # Calling ztrsyl(args, kwargs) (line 108)
    ztrsyl_call_result_35472 = invoke(stypy.reporting.localization.Localization(__file__, 108, 33), ztrsyl_35467, *[Rii_35468, Rjj_35469, S_35470], **kwargs_35471)
    
    # Obtaining the member '__getitem__' of a type (line 108)
    getitem___35473 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 16), ztrsyl_call_result_35472, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 108)
    subscript_call_result_35474 = invoke(stypy.reporting.localization.Localization(__file__, 108, 16), getitem___35473, int_35466)
    
    # Assigning a type to the variable 'tuple_var_assignment_35089' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 16), 'tuple_var_assignment_35089', subscript_call_result_35474)
    
    # Assigning a Name to a Name (line 108):
    # Getting the type of 'tuple_var_assignment_35087' (line 108)
    tuple_var_assignment_35087_35475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 16), 'tuple_var_assignment_35087')
    # Assigning a type to the variable 'x' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 16), 'x', tuple_var_assignment_35087_35475)
    
    # Assigning a Name to a Name (line 108):
    # Getting the type of 'tuple_var_assignment_35088' (line 108)
    tuple_var_assignment_35088_35476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 16), 'tuple_var_assignment_35088')
    # Assigning a type to the variable 'scale' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 19), 'scale', tuple_var_assignment_35088_35476)
    
    # Assigning a Name to a Name (line 108):
    # Getting the type of 'tuple_var_assignment_35089' (line 108)
    tuple_var_assignment_35089_35477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 16), 'tuple_var_assignment_35089')
    # Assigning a type to the variable 'info' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 26), 'info', tuple_var_assignment_35089_35477)
    # SSA join for if statement (line 105)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Subscript (line 109):
    
    # Assigning a BinOp to a Subscript (line 109):
    # Getting the type of 'x' (line 109)
    x_35478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 44), 'x')
    # Getting the type of 'scale' (line 109)
    scale_35479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 48), 'scale')
    # Applying the binary operator '*' (line 109)
    result_mul_35480 = python_operator(stypy.reporting.localization.Localization(__file__, 109, 44), '*', x_35478, scale_35479)
    
    # Getting the type of 'R' (line 109)
    R_35481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 12), 'R')
    # Getting the type of 'istart' (line 109)
    istart_35482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 14), 'istart')
    # Getting the type of 'istop' (line 109)
    istop_35483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 21), 'istop')
    slice_35484 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 109, 12), istart_35482, istop_35483, None)
    # Getting the type of 'jstart' (line 109)
    jstart_35485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 28), 'jstart')
    # Getting the type of 'jstop' (line 109)
    jstop_35486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 35), 'jstop')
    slice_35487 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 109, 12), jstart_35485, jstop_35486, None)
    # Storing an element on a container (line 109)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 12), R_35481, ((slice_35484, slice_35487), result_mul_35480))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'R' (line 112)
    R_35488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 11), 'R')
    # Assigning a type to the variable 'stypy_return_type' (line 112)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'stypy_return_type', R_35488)
    
    # ################# End of '_sqrtm_triu(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_sqrtm_triu' in the type store
    # Getting the type of 'stypy_return_type' (line 26)
    stypy_return_type_35489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_35489)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_sqrtm_triu'
    return stypy_return_type_35489

# Assigning a type to the variable '_sqrtm_triu' (line 26)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), '_sqrtm_triu', _sqrtm_triu)

@norecursion
def sqrtm(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'True' (line 115)
    True_35490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 18), 'True')
    int_35491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 34), 'int')
    defaults = [True_35490, int_35491]
    # Create a new context for function 'sqrtm'
    module_type_store = module_type_store.open_function_context('sqrtm', 115, 0, False)
    
    # Passed parameters checking function
    sqrtm.stypy_localization = localization
    sqrtm.stypy_type_of_self = None
    sqrtm.stypy_type_store = module_type_store
    sqrtm.stypy_function_name = 'sqrtm'
    sqrtm.stypy_param_names_list = ['A', 'disp', 'blocksize']
    sqrtm.stypy_varargs_param_name = None
    sqrtm.stypy_kwargs_param_name = None
    sqrtm.stypy_call_defaults = defaults
    sqrtm.stypy_call_varargs = varargs
    sqrtm.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'sqrtm', ['A', 'disp', 'blocksize'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'sqrtm', localization, ['A', 'disp', 'blocksize'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'sqrtm(...)' code ##################

    str_35492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, (-1)), 'str', '\n    Matrix square root.\n\n    Parameters\n    ----------\n    A : (N, N) array_like\n        Matrix whose square root to evaluate\n    disp : bool, optional\n        Print warning if error in the result is estimated large\n        instead of returning estimated error. (Default: True)\n    blocksize : integer, optional\n        If the blocksize is not degenerate with respect to the\n        size of the input array, then use a blocked algorithm. (Default: 64)\n\n    Returns\n    -------\n    sqrtm : (N, N) ndarray\n        Value of the sqrt function at `A`\n\n    errest : float\n        (if disp == False)\n\n        Frobenius norm of the estimated error, ||err||_F / ||A||_F\n\n    References\n    ----------\n    .. [1] Edvin Deadman, Nicholas J. Higham, Rui Ralha (2013)\n           "Blocked Schur Algorithms for Computing the Matrix Square Root,\n           Lecture Notes in Computer Science, 7782. pp. 171-182.\n\n    Examples\n    --------\n    >>> from scipy.linalg import sqrtm\n    >>> a = np.array([[1.0, 3.0], [1.0, 4.0]])\n    >>> r = sqrtm(a)\n    >>> r\n    array([[ 0.75592895,  1.13389342],\n           [ 0.37796447,  1.88982237]])\n    >>> r.dot(r)\n    array([[ 1.,  3.],\n           [ 1.,  4.]])\n\n    ')
    
    # Assigning a Call to a Name (line 159):
    
    # Assigning a Call to a Name (line 159):
    
    # Call to _asarray_validated(...): (line 159)
    # Processing the call arguments (line 159)
    # Getting the type of 'A' (line 159)
    A_35494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 27), 'A', False)
    # Processing the call keyword arguments (line 159)
    # Getting the type of 'True' (line 159)
    True_35495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 43), 'True', False)
    keyword_35496 = True_35495
    # Getting the type of 'True' (line 159)
    True_35497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 60), 'True', False)
    keyword_35498 = True_35497
    kwargs_35499 = {'as_inexact': keyword_35498, 'check_finite': keyword_35496}
    # Getting the type of '_asarray_validated' (line 159)
    _asarray_validated_35493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), '_asarray_validated', False)
    # Calling _asarray_validated(args, kwargs) (line 159)
    _asarray_validated_call_result_35500 = invoke(stypy.reporting.localization.Localization(__file__, 159, 8), _asarray_validated_35493, *[A_35494], **kwargs_35499)
    
    # Assigning a type to the variable 'A' (line 159)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 4), 'A', _asarray_validated_call_result_35500)
    
    
    
    # Call to len(...): (line 160)
    # Processing the call arguments (line 160)
    # Getting the type of 'A' (line 160)
    A_35502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 11), 'A', False)
    # Obtaining the member 'shape' of a type (line 160)
    shape_35503 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 11), A_35502, 'shape')
    # Processing the call keyword arguments (line 160)
    kwargs_35504 = {}
    # Getting the type of 'len' (line 160)
    len_35501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 7), 'len', False)
    # Calling len(args, kwargs) (line 160)
    len_call_result_35505 = invoke(stypy.reporting.localization.Localization(__file__, 160, 7), len_35501, *[shape_35503], **kwargs_35504)
    
    int_35506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 23), 'int')
    # Applying the binary operator '!=' (line 160)
    result_ne_35507 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 7), '!=', len_call_result_35505, int_35506)
    
    # Testing the type of an if condition (line 160)
    if_condition_35508 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 160, 4), result_ne_35507)
    # Assigning a type to the variable 'if_condition_35508' (line 160)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 4), 'if_condition_35508', if_condition_35508)
    # SSA begins for if statement (line 160)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 161)
    # Processing the call arguments (line 161)
    str_35510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 25), 'str', 'Non-matrix input to matrix function.')
    # Processing the call keyword arguments (line 161)
    kwargs_35511 = {}
    # Getting the type of 'ValueError' (line 161)
    ValueError_35509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 161)
    ValueError_call_result_35512 = invoke(stypy.reporting.localization.Localization(__file__, 161, 14), ValueError_35509, *[str_35510], **kwargs_35511)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 161, 8), ValueError_call_result_35512, 'raise parameter', BaseException)
    # SSA join for if statement (line 160)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'blocksize' (line 162)
    blocksize_35513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 7), 'blocksize')
    int_35514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 19), 'int')
    # Applying the binary operator '<' (line 162)
    result_lt_35515 = python_operator(stypy.reporting.localization.Localization(__file__, 162, 7), '<', blocksize_35513, int_35514)
    
    # Testing the type of an if condition (line 162)
    if_condition_35516 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 162, 4), result_lt_35515)
    # Assigning a type to the variable 'if_condition_35516' (line 162)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 4), 'if_condition_35516', if_condition_35516)
    # SSA begins for if statement (line 162)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 163)
    # Processing the call arguments (line 163)
    str_35518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 25), 'str', 'The blocksize should be at least 1.')
    # Processing the call keyword arguments (line 163)
    kwargs_35519 = {}
    # Getting the type of 'ValueError' (line 163)
    ValueError_35517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 163)
    ValueError_call_result_35520 = invoke(stypy.reporting.localization.Localization(__file__, 163, 14), ValueError_35517, *[str_35518], **kwargs_35519)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 163, 8), ValueError_call_result_35520, 'raise parameter', BaseException)
    # SSA join for if statement (line 162)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 164):
    
    # Assigning a Call to a Name (line 164):
    
    # Call to isrealobj(...): (line 164)
    # Processing the call arguments (line 164)
    # Getting the type of 'A' (line 164)
    A_35523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 32), 'A', False)
    # Processing the call keyword arguments (line 164)
    kwargs_35524 = {}
    # Getting the type of 'np' (line 164)
    np_35521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 19), 'np', False)
    # Obtaining the member 'isrealobj' of a type (line 164)
    isrealobj_35522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 19), np_35521, 'isrealobj')
    # Calling isrealobj(args, kwargs) (line 164)
    isrealobj_call_result_35525 = invoke(stypy.reporting.localization.Localization(__file__, 164, 19), isrealobj_35522, *[A_35523], **kwargs_35524)
    
    # Assigning a type to the variable 'keep_it_real' (line 164)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 4), 'keep_it_real', isrealobj_call_result_35525)
    
    # Getting the type of 'keep_it_real' (line 165)
    keep_it_real_35526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 7), 'keep_it_real')
    # Testing the type of an if condition (line 165)
    if_condition_35527 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 165, 4), keep_it_real_35526)
    # Assigning a type to the variable 'if_condition_35527' (line 165)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 4), 'if_condition_35527', if_condition_35527)
    # SSA begins for if statement (line 165)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 166):
    
    # Assigning a Subscript to a Name (line 166):
    
    # Obtaining the type of the subscript
    int_35528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 8), 'int')
    
    # Call to schur(...): (line 166)
    # Processing the call arguments (line 166)
    # Getting the type of 'A' (line 166)
    A_35530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 21), 'A', False)
    # Processing the call keyword arguments (line 166)
    kwargs_35531 = {}
    # Getting the type of 'schur' (line 166)
    schur_35529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 15), 'schur', False)
    # Calling schur(args, kwargs) (line 166)
    schur_call_result_35532 = invoke(stypy.reporting.localization.Localization(__file__, 166, 15), schur_35529, *[A_35530], **kwargs_35531)
    
    # Obtaining the member '__getitem__' of a type (line 166)
    getitem___35533 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 8), schur_call_result_35532, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 166)
    subscript_call_result_35534 = invoke(stypy.reporting.localization.Localization(__file__, 166, 8), getitem___35533, int_35528)
    
    # Assigning a type to the variable 'tuple_var_assignment_35090' (line 166)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'tuple_var_assignment_35090', subscript_call_result_35534)
    
    # Assigning a Subscript to a Name (line 166):
    
    # Obtaining the type of the subscript
    int_35535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 8), 'int')
    
    # Call to schur(...): (line 166)
    # Processing the call arguments (line 166)
    # Getting the type of 'A' (line 166)
    A_35537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 21), 'A', False)
    # Processing the call keyword arguments (line 166)
    kwargs_35538 = {}
    # Getting the type of 'schur' (line 166)
    schur_35536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 15), 'schur', False)
    # Calling schur(args, kwargs) (line 166)
    schur_call_result_35539 = invoke(stypy.reporting.localization.Localization(__file__, 166, 15), schur_35536, *[A_35537], **kwargs_35538)
    
    # Obtaining the member '__getitem__' of a type (line 166)
    getitem___35540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 8), schur_call_result_35539, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 166)
    subscript_call_result_35541 = invoke(stypy.reporting.localization.Localization(__file__, 166, 8), getitem___35540, int_35535)
    
    # Assigning a type to the variable 'tuple_var_assignment_35091' (line 166)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'tuple_var_assignment_35091', subscript_call_result_35541)
    
    # Assigning a Name to a Name (line 166):
    # Getting the type of 'tuple_var_assignment_35090' (line 166)
    tuple_var_assignment_35090_35542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'tuple_var_assignment_35090')
    # Assigning a type to the variable 'T' (line 166)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'T', tuple_var_assignment_35090_35542)
    
    # Assigning a Name to a Name (line 166):
    # Getting the type of 'tuple_var_assignment_35091' (line 166)
    tuple_var_assignment_35091_35543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'tuple_var_assignment_35091')
    # Assigning a type to the variable 'Z' (line 166)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 11), 'Z', tuple_var_assignment_35091_35543)
    
    
    
    # Call to array_equal(...): (line 167)
    # Processing the call arguments (line 167)
    # Getting the type of 'T' (line 167)
    T_35546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 30), 'T', False)
    
    # Call to triu(...): (line 167)
    # Processing the call arguments (line 167)
    # Getting the type of 'T' (line 167)
    T_35549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 41), 'T', False)
    # Processing the call keyword arguments (line 167)
    kwargs_35550 = {}
    # Getting the type of 'np' (line 167)
    np_35547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 33), 'np', False)
    # Obtaining the member 'triu' of a type (line 167)
    triu_35548 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 33), np_35547, 'triu')
    # Calling triu(args, kwargs) (line 167)
    triu_call_result_35551 = invoke(stypy.reporting.localization.Localization(__file__, 167, 33), triu_35548, *[T_35549], **kwargs_35550)
    
    # Processing the call keyword arguments (line 167)
    kwargs_35552 = {}
    # Getting the type of 'np' (line 167)
    np_35544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 15), 'np', False)
    # Obtaining the member 'array_equal' of a type (line 167)
    array_equal_35545 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 15), np_35544, 'array_equal')
    # Calling array_equal(args, kwargs) (line 167)
    array_equal_call_result_35553 = invoke(stypy.reporting.localization.Localization(__file__, 167, 15), array_equal_35545, *[T_35546, triu_call_result_35551], **kwargs_35552)
    
    # Applying the 'not' unary operator (line 167)
    result_not__35554 = python_operator(stypy.reporting.localization.Localization(__file__, 167, 11), 'not', array_equal_call_result_35553)
    
    # Testing the type of an if condition (line 167)
    if_condition_35555 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 167, 8), result_not__35554)
    # Assigning a type to the variable 'if_condition_35555' (line 167)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'if_condition_35555', if_condition_35555)
    # SSA begins for if statement (line 167)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 168):
    
    # Assigning a Subscript to a Name (line 168):
    
    # Obtaining the type of the subscript
    int_35556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 12), 'int')
    
    # Call to rsf2csf(...): (line 168)
    # Processing the call arguments (line 168)
    # Getting the type of 'T' (line 168)
    T_35558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 27), 'T', False)
    # Getting the type of 'Z' (line 168)
    Z_35559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 30), 'Z', False)
    # Processing the call keyword arguments (line 168)
    kwargs_35560 = {}
    # Getting the type of 'rsf2csf' (line 168)
    rsf2csf_35557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 19), 'rsf2csf', False)
    # Calling rsf2csf(args, kwargs) (line 168)
    rsf2csf_call_result_35561 = invoke(stypy.reporting.localization.Localization(__file__, 168, 19), rsf2csf_35557, *[T_35558, Z_35559], **kwargs_35560)
    
    # Obtaining the member '__getitem__' of a type (line 168)
    getitem___35562 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 12), rsf2csf_call_result_35561, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 168)
    subscript_call_result_35563 = invoke(stypy.reporting.localization.Localization(__file__, 168, 12), getitem___35562, int_35556)
    
    # Assigning a type to the variable 'tuple_var_assignment_35092' (line 168)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 12), 'tuple_var_assignment_35092', subscript_call_result_35563)
    
    # Assigning a Subscript to a Name (line 168):
    
    # Obtaining the type of the subscript
    int_35564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 12), 'int')
    
    # Call to rsf2csf(...): (line 168)
    # Processing the call arguments (line 168)
    # Getting the type of 'T' (line 168)
    T_35566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 27), 'T', False)
    # Getting the type of 'Z' (line 168)
    Z_35567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 30), 'Z', False)
    # Processing the call keyword arguments (line 168)
    kwargs_35568 = {}
    # Getting the type of 'rsf2csf' (line 168)
    rsf2csf_35565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 19), 'rsf2csf', False)
    # Calling rsf2csf(args, kwargs) (line 168)
    rsf2csf_call_result_35569 = invoke(stypy.reporting.localization.Localization(__file__, 168, 19), rsf2csf_35565, *[T_35566, Z_35567], **kwargs_35568)
    
    # Obtaining the member '__getitem__' of a type (line 168)
    getitem___35570 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 12), rsf2csf_call_result_35569, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 168)
    subscript_call_result_35571 = invoke(stypy.reporting.localization.Localization(__file__, 168, 12), getitem___35570, int_35564)
    
    # Assigning a type to the variable 'tuple_var_assignment_35093' (line 168)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 12), 'tuple_var_assignment_35093', subscript_call_result_35571)
    
    # Assigning a Name to a Name (line 168):
    # Getting the type of 'tuple_var_assignment_35092' (line 168)
    tuple_var_assignment_35092_35572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 12), 'tuple_var_assignment_35092')
    # Assigning a type to the variable 'T' (line 168)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 12), 'T', tuple_var_assignment_35092_35572)
    
    # Assigning a Name to a Name (line 168):
    # Getting the type of 'tuple_var_assignment_35093' (line 168)
    tuple_var_assignment_35093_35573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 12), 'tuple_var_assignment_35093')
    # Assigning a type to the variable 'Z' (line 168)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 15), 'Z', tuple_var_assignment_35093_35573)
    # SSA join for if statement (line 167)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 165)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Tuple (line 170):
    
    # Assigning a Subscript to a Name (line 170):
    
    # Obtaining the type of the subscript
    int_35574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 8), 'int')
    
    # Call to schur(...): (line 170)
    # Processing the call arguments (line 170)
    # Getting the type of 'A' (line 170)
    A_35576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 21), 'A', False)
    # Processing the call keyword arguments (line 170)
    str_35577 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 31), 'str', 'complex')
    keyword_35578 = str_35577
    kwargs_35579 = {'output': keyword_35578}
    # Getting the type of 'schur' (line 170)
    schur_35575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 15), 'schur', False)
    # Calling schur(args, kwargs) (line 170)
    schur_call_result_35580 = invoke(stypy.reporting.localization.Localization(__file__, 170, 15), schur_35575, *[A_35576], **kwargs_35579)
    
    # Obtaining the member '__getitem__' of a type (line 170)
    getitem___35581 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 8), schur_call_result_35580, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 170)
    subscript_call_result_35582 = invoke(stypy.reporting.localization.Localization(__file__, 170, 8), getitem___35581, int_35574)
    
    # Assigning a type to the variable 'tuple_var_assignment_35094' (line 170)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 8), 'tuple_var_assignment_35094', subscript_call_result_35582)
    
    # Assigning a Subscript to a Name (line 170):
    
    # Obtaining the type of the subscript
    int_35583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 8), 'int')
    
    # Call to schur(...): (line 170)
    # Processing the call arguments (line 170)
    # Getting the type of 'A' (line 170)
    A_35585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 21), 'A', False)
    # Processing the call keyword arguments (line 170)
    str_35586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 31), 'str', 'complex')
    keyword_35587 = str_35586
    kwargs_35588 = {'output': keyword_35587}
    # Getting the type of 'schur' (line 170)
    schur_35584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 15), 'schur', False)
    # Calling schur(args, kwargs) (line 170)
    schur_call_result_35589 = invoke(stypy.reporting.localization.Localization(__file__, 170, 15), schur_35584, *[A_35585], **kwargs_35588)
    
    # Obtaining the member '__getitem__' of a type (line 170)
    getitem___35590 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 8), schur_call_result_35589, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 170)
    subscript_call_result_35591 = invoke(stypy.reporting.localization.Localization(__file__, 170, 8), getitem___35590, int_35583)
    
    # Assigning a type to the variable 'tuple_var_assignment_35095' (line 170)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 8), 'tuple_var_assignment_35095', subscript_call_result_35591)
    
    # Assigning a Name to a Name (line 170):
    # Getting the type of 'tuple_var_assignment_35094' (line 170)
    tuple_var_assignment_35094_35592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 8), 'tuple_var_assignment_35094')
    # Assigning a type to the variable 'T' (line 170)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 8), 'T', tuple_var_assignment_35094_35592)
    
    # Assigning a Name to a Name (line 170):
    # Getting the type of 'tuple_var_assignment_35095' (line 170)
    tuple_var_assignment_35095_35593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 8), 'tuple_var_assignment_35095')
    # Assigning a type to the variable 'Z' (line 170)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 11), 'Z', tuple_var_assignment_35095_35593)
    # SSA join for if statement (line 165)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Name (line 171):
    
    # Assigning a Name to a Name (line 171):
    # Getting the type of 'False' (line 171)
    False_35594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 15), 'False')
    # Assigning a type to the variable 'failflag' (line 171)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 4), 'failflag', False_35594)
    
    
    # SSA begins for try-except statement (line 172)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 173):
    
    # Assigning a Call to a Name (line 173):
    
    # Call to _sqrtm_triu(...): (line 173)
    # Processing the call arguments (line 173)
    # Getting the type of 'T' (line 173)
    T_35596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 24), 'T', False)
    # Processing the call keyword arguments (line 173)
    # Getting the type of 'blocksize' (line 173)
    blocksize_35597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 37), 'blocksize', False)
    keyword_35598 = blocksize_35597
    kwargs_35599 = {'blocksize': keyword_35598}
    # Getting the type of '_sqrtm_triu' (line 173)
    _sqrtm_triu_35595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 12), '_sqrtm_triu', False)
    # Calling _sqrtm_triu(args, kwargs) (line 173)
    _sqrtm_triu_call_result_35600 = invoke(stypy.reporting.localization.Localization(__file__, 173, 12), _sqrtm_triu_35595, *[T_35596], **kwargs_35599)
    
    # Assigning a type to the variable 'R' (line 173)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 8), 'R', _sqrtm_triu_call_result_35600)
    
    # Assigning a Attribute to a Name (line 174):
    
    # Assigning a Attribute to a Name (line 174):
    
    # Call to conjugate(...): (line 174)
    # Processing the call arguments (line 174)
    # Getting the type of 'Z' (line 174)
    Z_35603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 26), 'Z', False)
    # Processing the call keyword arguments (line 174)
    kwargs_35604 = {}
    # Getting the type of 'np' (line 174)
    np_35601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 13), 'np', False)
    # Obtaining the member 'conjugate' of a type (line 174)
    conjugate_35602 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 13), np_35601, 'conjugate')
    # Calling conjugate(args, kwargs) (line 174)
    conjugate_call_result_35605 = invoke(stypy.reporting.localization.Localization(__file__, 174, 13), conjugate_35602, *[Z_35603], **kwargs_35604)
    
    # Obtaining the member 'T' of a type (line 174)
    T_35606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 13), conjugate_call_result_35605, 'T')
    # Assigning a type to the variable 'ZH' (line 174)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'ZH', T_35606)
    
    # Assigning a Call to a Name (line 175):
    
    # Assigning a Call to a Name (line 175):
    
    # Call to dot(...): (line 175)
    # Processing the call arguments (line 175)
    # Getting the type of 'ZH' (line 175)
    ZH_35613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 25), 'ZH', False)
    # Processing the call keyword arguments (line 175)
    kwargs_35614 = {}
    
    # Call to dot(...): (line 175)
    # Processing the call arguments (line 175)
    # Getting the type of 'R' (line 175)
    R_35609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 18), 'R', False)
    # Processing the call keyword arguments (line 175)
    kwargs_35610 = {}
    # Getting the type of 'Z' (line 175)
    Z_35607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 12), 'Z', False)
    # Obtaining the member 'dot' of a type (line 175)
    dot_35608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 12), Z_35607, 'dot')
    # Calling dot(args, kwargs) (line 175)
    dot_call_result_35611 = invoke(stypy.reporting.localization.Localization(__file__, 175, 12), dot_35608, *[R_35609], **kwargs_35610)
    
    # Obtaining the member 'dot' of a type (line 175)
    dot_35612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 12), dot_call_result_35611, 'dot')
    # Calling dot(args, kwargs) (line 175)
    dot_call_result_35615 = invoke(stypy.reporting.localization.Localization(__file__, 175, 12), dot_35612, *[ZH_35613], **kwargs_35614)
    
    # Assigning a type to the variable 'X' (line 175)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'X', dot_call_result_35615)
    # SSA branch for the except part of a try statement (line 172)
    # SSA branch for the except 'SqrtmError' branch of a try statement (line 172)
    module_type_store.open_ssa_branch('except')
    
    # Assigning a Name to a Name (line 177):
    
    # Assigning a Name to a Name (line 177):
    # Getting the type of 'True' (line 177)
    True_35616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 19), 'True')
    # Assigning a type to the variable 'failflag' (line 177)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'failflag', True_35616)
    
    # Assigning a Call to a Name (line 178):
    
    # Assigning a Call to a Name (line 178):
    
    # Call to empty_like(...): (line 178)
    # Processing the call arguments (line 178)
    # Getting the type of 'A' (line 178)
    A_35619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 26), 'A', False)
    # Processing the call keyword arguments (line 178)
    kwargs_35620 = {}
    # Getting the type of 'np' (line 178)
    np_35617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 12), 'np', False)
    # Obtaining the member 'empty_like' of a type (line 178)
    empty_like_35618 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 12), np_35617, 'empty_like')
    # Calling empty_like(args, kwargs) (line 178)
    empty_like_call_result_35621 = invoke(stypy.reporting.localization.Localization(__file__, 178, 12), empty_like_35618, *[A_35619], **kwargs_35620)
    
    # Assigning a type to the variable 'X' (line 178)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'X', empty_like_call_result_35621)
    
    # Call to fill(...): (line 179)
    # Processing the call arguments (line 179)
    # Getting the type of 'np' (line 179)
    np_35624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 15), 'np', False)
    # Obtaining the member 'nan' of a type (line 179)
    nan_35625 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 15), np_35624, 'nan')
    # Processing the call keyword arguments (line 179)
    kwargs_35626 = {}
    # Getting the type of 'X' (line 179)
    X_35622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 8), 'X', False)
    # Obtaining the member 'fill' of a type (line 179)
    fill_35623 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 8), X_35622, 'fill')
    # Calling fill(args, kwargs) (line 179)
    fill_call_result_35627 = invoke(stypy.reporting.localization.Localization(__file__, 179, 8), fill_35623, *[nan_35625], **kwargs_35626)
    
    # SSA join for try-except statement (line 172)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'disp' (line 181)
    disp_35628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 7), 'disp')
    # Testing the type of an if condition (line 181)
    if_condition_35629 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 181, 4), disp_35628)
    # Assigning a type to the variable 'if_condition_35629' (line 181)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 4), 'if_condition_35629', if_condition_35629)
    # SSA begins for if statement (line 181)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 182):
    
    # Assigning a Call to a Name (line 182):
    
    # Call to any(...): (line 182)
    # Processing the call arguments (line 182)
    
    
    # Call to diag(...): (line 182)
    # Processing the call arguments (line 182)
    # Getting the type of 'T' (line 182)
    T_35634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 31), 'T', False)
    # Processing the call keyword arguments (line 182)
    kwargs_35635 = {}
    # Getting the type of 'np' (line 182)
    np_35632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 23), 'np', False)
    # Obtaining the member 'diag' of a type (line 182)
    diag_35633 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 23), np_35632, 'diag')
    # Calling diag(args, kwargs) (line 182)
    diag_call_result_35636 = invoke(stypy.reporting.localization.Localization(__file__, 182, 23), diag_35633, *[T_35634], **kwargs_35635)
    
    int_35637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 37), 'int')
    # Applying the binary operator '==' (line 182)
    result_eq_35638 = python_operator(stypy.reporting.localization.Localization(__file__, 182, 23), '==', diag_call_result_35636, int_35637)
    
    # Processing the call keyword arguments (line 182)
    kwargs_35639 = {}
    # Getting the type of 'np' (line 182)
    np_35630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 16), 'np', False)
    # Obtaining the member 'any' of a type (line 182)
    any_35631 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 16), np_35630, 'any')
    # Calling any(args, kwargs) (line 182)
    any_call_result_35640 = invoke(stypy.reporting.localization.Localization(__file__, 182, 16), any_35631, *[result_eq_35638], **kwargs_35639)
    
    # Assigning a type to the variable 'nzeig' (line 182)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'nzeig', any_call_result_35640)
    
    # Getting the type of 'nzeig' (line 183)
    nzeig_35641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 11), 'nzeig')
    # Testing the type of an if condition (line 183)
    if_condition_35642 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 183, 8), nzeig_35641)
    # Assigning a type to the variable 'if_condition_35642' (line 183)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 8), 'if_condition_35642', if_condition_35642)
    # SSA begins for if statement (line 183)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to print(...): (line 184)
    # Processing the call arguments (line 184)
    str_35644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 18), 'str', 'Matrix is singular and may not have a square root.')
    # Processing the call keyword arguments (line 184)
    kwargs_35645 = {}
    # Getting the type of 'print' (line 184)
    print_35643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 12), 'print', False)
    # Calling print(args, kwargs) (line 184)
    print_call_result_35646 = invoke(stypy.reporting.localization.Localization(__file__, 184, 12), print_35643, *[str_35644], **kwargs_35645)
    
    # SSA branch for the else part of an if statement (line 183)
    module_type_store.open_ssa_branch('else')
    
    # Getting the type of 'failflag' (line 185)
    failflag_35647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 13), 'failflag')
    # Testing the type of an if condition (line 185)
    if_condition_35648 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 185, 13), failflag_35647)
    # Assigning a type to the variable 'if_condition_35648' (line 185)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 13), 'if_condition_35648', if_condition_35648)
    # SSA begins for if statement (line 185)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to print(...): (line 186)
    # Processing the call arguments (line 186)
    str_35650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 18), 'str', 'Failed to find a square root.')
    # Processing the call keyword arguments (line 186)
    kwargs_35651 = {}
    # Getting the type of 'print' (line 186)
    print_35649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 12), 'print', False)
    # Calling print(args, kwargs) (line 186)
    print_call_result_35652 = invoke(stypy.reporting.localization.Localization(__file__, 186, 12), print_35649, *[str_35650], **kwargs_35651)
    
    # SSA join for if statement (line 185)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 183)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'X' (line 187)
    X_35653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 15), 'X')
    # Assigning a type to the variable 'stypy_return_type' (line 187)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 8), 'stypy_return_type', X_35653)
    # SSA branch for the else part of an if statement (line 181)
    module_type_store.open_ssa_branch('else')
    
    
    # SSA begins for try-except statement (line 189)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a BinOp to a Name (line 190):
    
    # Assigning a BinOp to a Name (line 190):
    
    # Call to norm(...): (line 190)
    # Processing the call arguments (line 190)
    
    # Call to dot(...): (line 190)
    # Processing the call arguments (line 190)
    # Getting the type of 'X' (line 190)
    X_35657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 30), 'X', False)
    # Processing the call keyword arguments (line 190)
    kwargs_35658 = {}
    # Getting the type of 'X' (line 190)
    X_35655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 24), 'X', False)
    # Obtaining the member 'dot' of a type (line 190)
    dot_35656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 24), X_35655, 'dot')
    # Calling dot(args, kwargs) (line 190)
    dot_call_result_35659 = invoke(stypy.reporting.localization.Localization(__file__, 190, 24), dot_35656, *[X_35657], **kwargs_35658)
    
    # Getting the type of 'A' (line 190)
    A_35660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 35), 'A', False)
    # Applying the binary operator '-' (line 190)
    result_sub_35661 = python_operator(stypy.reporting.localization.Localization(__file__, 190, 24), '-', dot_call_result_35659, A_35660)
    
    str_35662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 38), 'str', 'fro')
    # Processing the call keyword arguments (line 190)
    kwargs_35663 = {}
    # Getting the type of 'norm' (line 190)
    norm_35654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 19), 'norm', False)
    # Calling norm(args, kwargs) (line 190)
    norm_call_result_35664 = invoke(stypy.reporting.localization.Localization(__file__, 190, 19), norm_35654, *[result_sub_35661, str_35662], **kwargs_35663)
    
    int_35665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 46), 'int')
    # Applying the binary operator '**' (line 190)
    result_pow_35666 = python_operator(stypy.reporting.localization.Localization(__file__, 190, 19), '**', norm_call_result_35664, int_35665)
    
    
    # Call to norm(...): (line 190)
    # Processing the call arguments (line 190)
    # Getting the type of 'A' (line 190)
    A_35668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 55), 'A', False)
    str_35669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 58), 'str', 'fro')
    # Processing the call keyword arguments (line 190)
    kwargs_35670 = {}
    # Getting the type of 'norm' (line 190)
    norm_35667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 50), 'norm', False)
    # Calling norm(args, kwargs) (line 190)
    norm_call_result_35671 = invoke(stypy.reporting.localization.Localization(__file__, 190, 50), norm_35667, *[A_35668, str_35669], **kwargs_35670)
    
    # Applying the binary operator 'div' (line 190)
    result_div_35672 = python_operator(stypy.reporting.localization.Localization(__file__, 190, 19), 'div', result_pow_35666, norm_call_result_35671)
    
    # Assigning a type to the variable 'arg2' (line 190)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 12), 'arg2', result_div_35672)
    # SSA branch for the except part of a try statement (line 189)
    # SSA branch for the except 'ValueError' branch of a try statement (line 189)
    module_type_store.open_ssa_branch('except')
    
    # Assigning a Attribute to a Name (line 193):
    
    # Assigning a Attribute to a Name (line 193):
    # Getting the type of 'np' (line 193)
    np_35673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 19), 'np')
    # Obtaining the member 'inf' of a type (line 193)
    inf_35674 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 19), np_35673, 'inf')
    # Assigning a type to the variable 'arg2' (line 193)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 12), 'arg2', inf_35674)
    # SSA join for try-except statement (line 189)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 195)
    tuple_35675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 195)
    # Adding element type (line 195)
    # Getting the type of 'X' (line 195)
    X_35676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 15), 'X')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 15), tuple_35675, X_35676)
    # Adding element type (line 195)
    # Getting the type of 'arg2' (line 195)
    arg2_35677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 18), 'arg2')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 15), tuple_35675, arg2_35677)
    
    # Assigning a type to the variable 'stypy_return_type' (line 195)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 8), 'stypy_return_type', tuple_35675)
    # SSA join for if statement (line 181)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'sqrtm(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'sqrtm' in the type store
    # Getting the type of 'stypy_return_type' (line 115)
    stypy_return_type_35678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_35678)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'sqrtm'
    return stypy_return_type_35678

# Assigning a type to the variable 'sqrtm' (line 115)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 0), 'sqrtm', sqrtm)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
