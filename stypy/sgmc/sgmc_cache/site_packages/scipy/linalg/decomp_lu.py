
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''LU decomposition functions.'''
2: 
3: from __future__ import division, print_function, absolute_import
4: 
5: from warnings import warn
6: 
7: from numpy import asarray, asarray_chkfinite
8: 
9: # Local imports
10: from .misc import _datacopied
11: from .lapack import get_lapack_funcs
12: from .flinalg import get_flinalg_funcs
13: 
14: __all__ = ['lu', 'lu_solve', 'lu_factor']
15: 
16: 
17: def lu_factor(a, overwrite_a=False, check_finite=True):
18:     '''
19:     Compute pivoted LU decomposition of a matrix.
20: 
21:     The decomposition is::
22: 
23:         A = P L U
24: 
25:     where P is a permutation matrix, L lower triangular with unit
26:     diagonal elements, and U upper triangular.
27: 
28:     Parameters
29:     ----------
30:     a : (M, M) array_like
31:         Matrix to decompose
32:     overwrite_a : bool, optional
33:         Whether to overwrite data in A (may increase performance)
34:     check_finite : bool, optional
35:         Whether to check that the input matrix contains only finite numbers.
36:         Disabling may give a performance gain, but may result in problems
37:         (crashes, non-termination) if the inputs do contain infinities or NaNs.
38: 
39:     Returns
40:     -------
41:     lu : (N, N) ndarray
42:         Matrix containing U in its upper triangle, and L in its lower triangle.
43:         The unit diagonal elements of L are not stored.
44:     piv : (N,) ndarray
45:         Pivot indices representing the permutation matrix P:
46:         row i of matrix was interchanged with row piv[i].
47: 
48:     See also
49:     --------
50:     lu_solve : solve an equation system using the LU factorization of a matrix
51: 
52:     Notes
53:     -----
54:     This is a wrapper to the ``*GETRF`` routines from LAPACK.
55: 
56:     '''
57:     if check_finite:
58:         a1 = asarray_chkfinite(a)
59:     else:
60:         a1 = asarray(a)
61:     if len(a1.shape) != 2 or (a1.shape[0] != a1.shape[1]):
62:         raise ValueError('expected square matrix')
63:     overwrite_a = overwrite_a or (_datacopied(a1, a))
64:     getrf, = get_lapack_funcs(('getrf',), (a1,))
65:     lu, piv, info = getrf(a1, overwrite_a=overwrite_a)
66:     if info < 0:
67:         raise ValueError('illegal value in %d-th argument of '
68:                                 'internal getrf (lu_factor)' % -info)
69:     if info > 0:
70:         warn("Diagonal number %d is exactly zero. Singular matrix." % info,
71:                     RuntimeWarning)
72:     return lu, piv
73: 
74: 
75: def lu_solve(lu_and_piv, b, trans=0, overwrite_b=False, check_finite=True):
76:     '''Solve an equation system, a x = b, given the LU factorization of a
77: 
78:     Parameters
79:     ----------
80:     (lu, piv)
81:         Factorization of the coefficient matrix a, as given by lu_factor
82:     b : array
83:         Right-hand side
84:     trans : {0, 1, 2}, optional
85:         Type of system to solve:
86: 
87:         =====  =========
88:         trans  system
89:         =====  =========
90:         0      a x   = b
91:         1      a^T x = b
92:         2      a^H x = b
93:         =====  =========
94:     overwrite_b : bool, optional
95:         Whether to overwrite data in b (may increase performance)
96:     check_finite : bool, optional
97:         Whether to check that the input matrices contain only finite numbers.
98:         Disabling may give a performance gain, but may result in problems
99:         (crashes, non-termination) if the inputs do contain infinities or NaNs.
100: 
101:     Returns
102:     -------
103:     x : array
104:         Solution to the system
105: 
106:     See also
107:     --------
108:     lu_factor : LU factorize a matrix
109: 
110:     '''
111:     (lu, piv) = lu_and_piv
112:     if check_finite:
113:         b1 = asarray_chkfinite(b)
114:     else:
115:         b1 = asarray(b)
116:     overwrite_b = overwrite_b or _datacopied(b1, b)
117:     if lu.shape[0] != b1.shape[0]:
118:         raise ValueError("incompatible dimensions.")
119: 
120:     getrs, = get_lapack_funcs(('getrs',), (lu, b1))
121:     x,info = getrs(lu, piv, b1, trans=trans, overwrite_b=overwrite_b)
122:     if info == 0:
123:         return x
124:     raise ValueError('illegal value in %d-th argument of internal gesv|posv'
125:                                                                     % -info)
126: 
127: 
128: def lu(a, permute_l=False, overwrite_a=False, check_finite=True):
129:     '''
130:     Compute pivoted LU decomposition of a matrix.
131: 
132:     The decomposition is::
133: 
134:         A = P L U
135: 
136:     where P is a permutation matrix, L lower triangular with unit
137:     diagonal elements, and U upper triangular.
138: 
139:     Parameters
140:     ----------
141:     a : (M, N) array_like
142:         Array to decompose
143:     permute_l : bool, optional
144:         Perform the multiplication P*L  (Default: do not permute)
145:     overwrite_a : bool, optional
146:         Whether to overwrite data in a (may improve performance)
147:     check_finite : bool, optional
148:         Whether to check that the input matrix contains only finite numbers.
149:         Disabling may give a performance gain, but may result in problems
150:         (crashes, non-termination) if the inputs do contain infinities or NaNs.
151: 
152:     Returns
153:     -------
154:     **(If permute_l == False)**
155: 
156:     p : (M, M) ndarray
157:         Permutation matrix
158:     l : (M, K) ndarray
159:         Lower triangular or trapezoidal matrix with unit diagonal.
160:         K = min(M, N)
161:     u : (K, N) ndarray
162:         Upper triangular or trapezoidal matrix
163: 
164:     **(If permute_l == True)**
165: 
166:     pl : (M, K) ndarray
167:         Permuted L matrix.
168:         K = min(M, N)
169:     u : (K, N) ndarray
170:         Upper triangular or trapezoidal matrix
171: 
172:     Notes
173:     -----
174:     This is a LU factorization routine written for Scipy.
175: 
176:     '''
177:     if check_finite:
178:         a1 = asarray_chkfinite(a)
179:     else:
180:         a1 = asarray(a)
181:     if len(a1.shape) != 2:
182:         raise ValueError('expected matrix')
183:     overwrite_a = overwrite_a or (_datacopied(a1, a))
184:     flu, = get_flinalg_funcs(('lu',), (a1,))
185:     p, l, u, info = flu(a1, permute_l=permute_l, overwrite_a=overwrite_a)
186:     if info < 0:
187:         raise ValueError('illegal value in %d-th argument of '
188:                                             'internal lu.getrf' % -info)
189:     if permute_l:
190:         return l, u
191:     return p, l, u
192: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_18037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 0), 'str', 'LU decomposition functions.')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from warnings import warn' statement (line 5)
try:
    from warnings import warn

except:
    warn = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'warnings', None, module_type_store, ['warn'], [warn])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from numpy import asarray, asarray_chkfinite' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_18038 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy')

if (type(import_18038) is not StypyTypeError):

    if (import_18038 != 'pyd_module'):
        __import__(import_18038)
        sys_modules_18039 = sys.modules[import_18038]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy', sys_modules_18039.module_type_store, module_type_store, ['asarray', 'asarray_chkfinite'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_18039, sys_modules_18039.module_type_store, module_type_store)
    else:
        from numpy import asarray, asarray_chkfinite

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy', None, module_type_store, ['asarray', 'asarray_chkfinite'], [asarray, asarray_chkfinite])

else:
    # Assigning a type to the variable 'numpy' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy', import_18038)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from scipy.linalg.misc import _datacopied' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_18040 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.linalg.misc')

if (type(import_18040) is not StypyTypeError):

    if (import_18040 != 'pyd_module'):
        __import__(import_18040)
        sys_modules_18041 = sys.modules[import_18040]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.linalg.misc', sys_modules_18041.module_type_store, module_type_store, ['_datacopied'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_18041, sys_modules_18041.module_type_store, module_type_store)
    else:
        from scipy.linalg.misc import _datacopied

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.linalg.misc', None, module_type_store, ['_datacopied'], [_datacopied])

else:
    # Assigning a type to the variable 'scipy.linalg.misc' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.linalg.misc', import_18040)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from scipy.linalg.lapack import get_lapack_funcs' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_18042 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.linalg.lapack')

if (type(import_18042) is not StypyTypeError):

    if (import_18042 != 'pyd_module'):
        __import__(import_18042)
        sys_modules_18043 = sys.modules[import_18042]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.linalg.lapack', sys_modules_18043.module_type_store, module_type_store, ['get_lapack_funcs'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_18043, sys_modules_18043.module_type_store, module_type_store)
    else:
        from scipy.linalg.lapack import get_lapack_funcs

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.linalg.lapack', None, module_type_store, ['get_lapack_funcs'], [get_lapack_funcs])

else:
    # Assigning a type to the variable 'scipy.linalg.lapack' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.linalg.lapack', import_18042)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from scipy.linalg.flinalg import get_flinalg_funcs' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_18044 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.linalg.flinalg')

if (type(import_18044) is not StypyTypeError):

    if (import_18044 != 'pyd_module'):
        __import__(import_18044)
        sys_modules_18045 = sys.modules[import_18044]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.linalg.flinalg', sys_modules_18045.module_type_store, module_type_store, ['get_flinalg_funcs'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_18045, sys_modules_18045.module_type_store, module_type_store)
    else:
        from scipy.linalg.flinalg import get_flinalg_funcs

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.linalg.flinalg', None, module_type_store, ['get_flinalg_funcs'], [get_flinalg_funcs])

else:
    # Assigning a type to the variable 'scipy.linalg.flinalg' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.linalg.flinalg', import_18044)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')


# Assigning a List to a Name (line 14):

# Assigning a List to a Name (line 14):
__all__ = ['lu', 'lu_solve', 'lu_factor']
module_type_store.set_exportable_members(['lu', 'lu_solve', 'lu_factor'])

# Obtaining an instance of the builtin type 'list' (line 14)
list_18046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 14)
# Adding element type (line 14)
str_18047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 11), 'str', 'lu')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 10), list_18046, str_18047)
# Adding element type (line 14)
str_18048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 17), 'str', 'lu_solve')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 10), list_18046, str_18048)
# Adding element type (line 14)
str_18049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 29), 'str', 'lu_factor')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 10), list_18046, str_18049)

# Assigning a type to the variable '__all__' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), '__all__', list_18046)

@norecursion
def lu_factor(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'False' (line 17)
    False_18050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 29), 'False')
    # Getting the type of 'True' (line 17)
    True_18051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 49), 'True')
    defaults = [False_18050, True_18051]
    # Create a new context for function 'lu_factor'
    module_type_store = module_type_store.open_function_context('lu_factor', 17, 0, False)
    
    # Passed parameters checking function
    lu_factor.stypy_localization = localization
    lu_factor.stypy_type_of_self = None
    lu_factor.stypy_type_store = module_type_store
    lu_factor.stypy_function_name = 'lu_factor'
    lu_factor.stypy_param_names_list = ['a', 'overwrite_a', 'check_finite']
    lu_factor.stypy_varargs_param_name = None
    lu_factor.stypy_kwargs_param_name = None
    lu_factor.stypy_call_defaults = defaults
    lu_factor.stypy_call_varargs = varargs
    lu_factor.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'lu_factor', ['a', 'overwrite_a', 'check_finite'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'lu_factor', localization, ['a', 'overwrite_a', 'check_finite'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'lu_factor(...)' code ##################

    str_18052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, (-1)), 'str', '\n    Compute pivoted LU decomposition of a matrix.\n\n    The decomposition is::\n\n        A = P L U\n\n    where P is a permutation matrix, L lower triangular with unit\n    diagonal elements, and U upper triangular.\n\n    Parameters\n    ----------\n    a : (M, M) array_like\n        Matrix to decompose\n    overwrite_a : bool, optional\n        Whether to overwrite data in A (may increase performance)\n    check_finite : bool, optional\n        Whether to check that the input matrix contains only finite numbers.\n        Disabling may give a performance gain, but may result in problems\n        (crashes, non-termination) if the inputs do contain infinities or NaNs.\n\n    Returns\n    -------\n    lu : (N, N) ndarray\n        Matrix containing U in its upper triangle, and L in its lower triangle.\n        The unit diagonal elements of L are not stored.\n    piv : (N,) ndarray\n        Pivot indices representing the permutation matrix P:\n        row i of matrix was interchanged with row piv[i].\n\n    See also\n    --------\n    lu_solve : solve an equation system using the LU factorization of a matrix\n\n    Notes\n    -----\n    This is a wrapper to the ``*GETRF`` routines from LAPACK.\n\n    ')
    
    # Getting the type of 'check_finite' (line 57)
    check_finite_18053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 7), 'check_finite')
    # Testing the type of an if condition (line 57)
    if_condition_18054 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 57, 4), check_finite_18053)
    # Assigning a type to the variable 'if_condition_18054' (line 57)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'if_condition_18054', if_condition_18054)
    # SSA begins for if statement (line 57)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 58):
    
    # Assigning a Call to a Name (line 58):
    
    # Call to asarray_chkfinite(...): (line 58)
    # Processing the call arguments (line 58)
    # Getting the type of 'a' (line 58)
    a_18056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 31), 'a', False)
    # Processing the call keyword arguments (line 58)
    kwargs_18057 = {}
    # Getting the type of 'asarray_chkfinite' (line 58)
    asarray_chkfinite_18055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 13), 'asarray_chkfinite', False)
    # Calling asarray_chkfinite(args, kwargs) (line 58)
    asarray_chkfinite_call_result_18058 = invoke(stypy.reporting.localization.Localization(__file__, 58, 13), asarray_chkfinite_18055, *[a_18056], **kwargs_18057)
    
    # Assigning a type to the variable 'a1' (line 58)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'a1', asarray_chkfinite_call_result_18058)
    # SSA branch for the else part of an if statement (line 57)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 60):
    
    # Assigning a Call to a Name (line 60):
    
    # Call to asarray(...): (line 60)
    # Processing the call arguments (line 60)
    # Getting the type of 'a' (line 60)
    a_18060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 21), 'a', False)
    # Processing the call keyword arguments (line 60)
    kwargs_18061 = {}
    # Getting the type of 'asarray' (line 60)
    asarray_18059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 13), 'asarray', False)
    # Calling asarray(args, kwargs) (line 60)
    asarray_call_result_18062 = invoke(stypy.reporting.localization.Localization(__file__, 60, 13), asarray_18059, *[a_18060], **kwargs_18061)
    
    # Assigning a type to the variable 'a1' (line 60)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'a1', asarray_call_result_18062)
    # SSA join for if statement (line 57)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    
    # Call to len(...): (line 61)
    # Processing the call arguments (line 61)
    # Getting the type of 'a1' (line 61)
    a1_18064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 11), 'a1', False)
    # Obtaining the member 'shape' of a type (line 61)
    shape_18065 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 11), a1_18064, 'shape')
    # Processing the call keyword arguments (line 61)
    kwargs_18066 = {}
    # Getting the type of 'len' (line 61)
    len_18063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 7), 'len', False)
    # Calling len(args, kwargs) (line 61)
    len_call_result_18067 = invoke(stypy.reporting.localization.Localization(__file__, 61, 7), len_18063, *[shape_18065], **kwargs_18066)
    
    int_18068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 24), 'int')
    # Applying the binary operator '!=' (line 61)
    result_ne_18069 = python_operator(stypy.reporting.localization.Localization(__file__, 61, 7), '!=', len_call_result_18067, int_18068)
    
    
    
    # Obtaining the type of the subscript
    int_18070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 39), 'int')
    # Getting the type of 'a1' (line 61)
    a1_18071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 30), 'a1')
    # Obtaining the member 'shape' of a type (line 61)
    shape_18072 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 30), a1_18071, 'shape')
    # Obtaining the member '__getitem__' of a type (line 61)
    getitem___18073 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 30), shape_18072, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 61)
    subscript_call_result_18074 = invoke(stypy.reporting.localization.Localization(__file__, 61, 30), getitem___18073, int_18070)
    
    
    # Obtaining the type of the subscript
    int_18075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 54), 'int')
    # Getting the type of 'a1' (line 61)
    a1_18076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 45), 'a1')
    # Obtaining the member 'shape' of a type (line 61)
    shape_18077 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 45), a1_18076, 'shape')
    # Obtaining the member '__getitem__' of a type (line 61)
    getitem___18078 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 45), shape_18077, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 61)
    subscript_call_result_18079 = invoke(stypy.reporting.localization.Localization(__file__, 61, 45), getitem___18078, int_18075)
    
    # Applying the binary operator '!=' (line 61)
    result_ne_18080 = python_operator(stypy.reporting.localization.Localization(__file__, 61, 30), '!=', subscript_call_result_18074, subscript_call_result_18079)
    
    # Applying the binary operator 'or' (line 61)
    result_or_keyword_18081 = python_operator(stypy.reporting.localization.Localization(__file__, 61, 7), 'or', result_ne_18069, result_ne_18080)
    
    # Testing the type of an if condition (line 61)
    if_condition_18082 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 61, 4), result_or_keyword_18081)
    # Assigning a type to the variable 'if_condition_18082' (line 61)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), 'if_condition_18082', if_condition_18082)
    # SSA begins for if statement (line 61)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 62)
    # Processing the call arguments (line 62)
    str_18084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 25), 'str', 'expected square matrix')
    # Processing the call keyword arguments (line 62)
    kwargs_18085 = {}
    # Getting the type of 'ValueError' (line 62)
    ValueError_18083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 62)
    ValueError_call_result_18086 = invoke(stypy.reporting.localization.Localization(__file__, 62, 14), ValueError_18083, *[str_18084], **kwargs_18085)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 62, 8), ValueError_call_result_18086, 'raise parameter', BaseException)
    # SSA join for if statement (line 61)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BoolOp to a Name (line 63):
    
    # Assigning a BoolOp to a Name (line 63):
    
    # Evaluating a boolean operation
    # Getting the type of 'overwrite_a' (line 63)
    overwrite_a_18087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 18), 'overwrite_a')
    
    # Call to _datacopied(...): (line 63)
    # Processing the call arguments (line 63)
    # Getting the type of 'a1' (line 63)
    a1_18089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 46), 'a1', False)
    # Getting the type of 'a' (line 63)
    a_18090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 50), 'a', False)
    # Processing the call keyword arguments (line 63)
    kwargs_18091 = {}
    # Getting the type of '_datacopied' (line 63)
    _datacopied_18088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 34), '_datacopied', False)
    # Calling _datacopied(args, kwargs) (line 63)
    _datacopied_call_result_18092 = invoke(stypy.reporting.localization.Localization(__file__, 63, 34), _datacopied_18088, *[a1_18089, a_18090], **kwargs_18091)
    
    # Applying the binary operator 'or' (line 63)
    result_or_keyword_18093 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 18), 'or', overwrite_a_18087, _datacopied_call_result_18092)
    
    # Assigning a type to the variable 'overwrite_a' (line 63)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'overwrite_a', result_or_keyword_18093)
    
    # Assigning a Call to a Tuple (line 64):
    
    # Assigning a Subscript to a Name (line 64):
    
    # Obtaining the type of the subscript
    int_18094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 4), 'int')
    
    # Call to get_lapack_funcs(...): (line 64)
    # Processing the call arguments (line 64)
    
    # Obtaining an instance of the builtin type 'tuple' (line 64)
    tuple_18096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 31), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 64)
    # Adding element type (line 64)
    str_18097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 31), 'str', 'getrf')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 31), tuple_18096, str_18097)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 64)
    tuple_18098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 43), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 64)
    # Adding element type (line 64)
    # Getting the type of 'a1' (line 64)
    a1_18099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 43), 'a1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 43), tuple_18098, a1_18099)
    
    # Processing the call keyword arguments (line 64)
    kwargs_18100 = {}
    # Getting the type of 'get_lapack_funcs' (line 64)
    get_lapack_funcs_18095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 13), 'get_lapack_funcs', False)
    # Calling get_lapack_funcs(args, kwargs) (line 64)
    get_lapack_funcs_call_result_18101 = invoke(stypy.reporting.localization.Localization(__file__, 64, 13), get_lapack_funcs_18095, *[tuple_18096, tuple_18098], **kwargs_18100)
    
    # Obtaining the member '__getitem__' of a type (line 64)
    getitem___18102 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 4), get_lapack_funcs_call_result_18101, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 64)
    subscript_call_result_18103 = invoke(stypy.reporting.localization.Localization(__file__, 64, 4), getitem___18102, int_18094)
    
    # Assigning a type to the variable 'tuple_var_assignment_18023' (line 64)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'tuple_var_assignment_18023', subscript_call_result_18103)
    
    # Assigning a Name to a Name (line 64):
    # Getting the type of 'tuple_var_assignment_18023' (line 64)
    tuple_var_assignment_18023_18104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'tuple_var_assignment_18023')
    # Assigning a type to the variable 'getrf' (line 64)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'getrf', tuple_var_assignment_18023_18104)
    
    # Assigning a Call to a Tuple (line 65):
    
    # Assigning a Subscript to a Name (line 65):
    
    # Obtaining the type of the subscript
    int_18105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 4), 'int')
    
    # Call to getrf(...): (line 65)
    # Processing the call arguments (line 65)
    # Getting the type of 'a1' (line 65)
    a1_18107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 26), 'a1', False)
    # Processing the call keyword arguments (line 65)
    # Getting the type of 'overwrite_a' (line 65)
    overwrite_a_18108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 42), 'overwrite_a', False)
    keyword_18109 = overwrite_a_18108
    kwargs_18110 = {'overwrite_a': keyword_18109}
    # Getting the type of 'getrf' (line 65)
    getrf_18106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 20), 'getrf', False)
    # Calling getrf(args, kwargs) (line 65)
    getrf_call_result_18111 = invoke(stypy.reporting.localization.Localization(__file__, 65, 20), getrf_18106, *[a1_18107], **kwargs_18110)
    
    # Obtaining the member '__getitem__' of a type (line 65)
    getitem___18112 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 4), getrf_call_result_18111, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 65)
    subscript_call_result_18113 = invoke(stypy.reporting.localization.Localization(__file__, 65, 4), getitem___18112, int_18105)
    
    # Assigning a type to the variable 'tuple_var_assignment_18024' (line 65)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'tuple_var_assignment_18024', subscript_call_result_18113)
    
    # Assigning a Subscript to a Name (line 65):
    
    # Obtaining the type of the subscript
    int_18114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 4), 'int')
    
    # Call to getrf(...): (line 65)
    # Processing the call arguments (line 65)
    # Getting the type of 'a1' (line 65)
    a1_18116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 26), 'a1', False)
    # Processing the call keyword arguments (line 65)
    # Getting the type of 'overwrite_a' (line 65)
    overwrite_a_18117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 42), 'overwrite_a', False)
    keyword_18118 = overwrite_a_18117
    kwargs_18119 = {'overwrite_a': keyword_18118}
    # Getting the type of 'getrf' (line 65)
    getrf_18115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 20), 'getrf', False)
    # Calling getrf(args, kwargs) (line 65)
    getrf_call_result_18120 = invoke(stypy.reporting.localization.Localization(__file__, 65, 20), getrf_18115, *[a1_18116], **kwargs_18119)
    
    # Obtaining the member '__getitem__' of a type (line 65)
    getitem___18121 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 4), getrf_call_result_18120, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 65)
    subscript_call_result_18122 = invoke(stypy.reporting.localization.Localization(__file__, 65, 4), getitem___18121, int_18114)
    
    # Assigning a type to the variable 'tuple_var_assignment_18025' (line 65)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'tuple_var_assignment_18025', subscript_call_result_18122)
    
    # Assigning a Subscript to a Name (line 65):
    
    # Obtaining the type of the subscript
    int_18123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 4), 'int')
    
    # Call to getrf(...): (line 65)
    # Processing the call arguments (line 65)
    # Getting the type of 'a1' (line 65)
    a1_18125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 26), 'a1', False)
    # Processing the call keyword arguments (line 65)
    # Getting the type of 'overwrite_a' (line 65)
    overwrite_a_18126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 42), 'overwrite_a', False)
    keyword_18127 = overwrite_a_18126
    kwargs_18128 = {'overwrite_a': keyword_18127}
    # Getting the type of 'getrf' (line 65)
    getrf_18124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 20), 'getrf', False)
    # Calling getrf(args, kwargs) (line 65)
    getrf_call_result_18129 = invoke(stypy.reporting.localization.Localization(__file__, 65, 20), getrf_18124, *[a1_18125], **kwargs_18128)
    
    # Obtaining the member '__getitem__' of a type (line 65)
    getitem___18130 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 4), getrf_call_result_18129, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 65)
    subscript_call_result_18131 = invoke(stypy.reporting.localization.Localization(__file__, 65, 4), getitem___18130, int_18123)
    
    # Assigning a type to the variable 'tuple_var_assignment_18026' (line 65)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'tuple_var_assignment_18026', subscript_call_result_18131)
    
    # Assigning a Name to a Name (line 65):
    # Getting the type of 'tuple_var_assignment_18024' (line 65)
    tuple_var_assignment_18024_18132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'tuple_var_assignment_18024')
    # Assigning a type to the variable 'lu' (line 65)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'lu', tuple_var_assignment_18024_18132)
    
    # Assigning a Name to a Name (line 65):
    # Getting the type of 'tuple_var_assignment_18025' (line 65)
    tuple_var_assignment_18025_18133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'tuple_var_assignment_18025')
    # Assigning a type to the variable 'piv' (line 65)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'piv', tuple_var_assignment_18025_18133)
    
    # Assigning a Name to a Name (line 65):
    # Getting the type of 'tuple_var_assignment_18026' (line 65)
    tuple_var_assignment_18026_18134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'tuple_var_assignment_18026')
    # Assigning a type to the variable 'info' (line 65)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 13), 'info', tuple_var_assignment_18026_18134)
    
    
    # Getting the type of 'info' (line 66)
    info_18135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 7), 'info')
    int_18136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 14), 'int')
    # Applying the binary operator '<' (line 66)
    result_lt_18137 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 7), '<', info_18135, int_18136)
    
    # Testing the type of an if condition (line 66)
    if_condition_18138 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 66, 4), result_lt_18137)
    # Assigning a type to the variable 'if_condition_18138' (line 66)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'if_condition_18138', if_condition_18138)
    # SSA begins for if statement (line 66)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 67)
    # Processing the call arguments (line 67)
    str_18140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 25), 'str', 'illegal value in %d-th argument of internal getrf (lu_factor)')
    
    # Getting the type of 'info' (line 68)
    info_18141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 64), 'info', False)
    # Applying the 'usub' unary operator (line 68)
    result___neg___18142 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 63), 'usub', info_18141)
    
    # Applying the binary operator '%' (line 67)
    result_mod_18143 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 25), '%', str_18140, result___neg___18142)
    
    # Processing the call keyword arguments (line 67)
    kwargs_18144 = {}
    # Getting the type of 'ValueError' (line 67)
    ValueError_18139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 67)
    ValueError_call_result_18145 = invoke(stypy.reporting.localization.Localization(__file__, 67, 14), ValueError_18139, *[result_mod_18143], **kwargs_18144)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 67, 8), ValueError_call_result_18145, 'raise parameter', BaseException)
    # SSA join for if statement (line 66)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'info' (line 69)
    info_18146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 7), 'info')
    int_18147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 14), 'int')
    # Applying the binary operator '>' (line 69)
    result_gt_18148 = python_operator(stypy.reporting.localization.Localization(__file__, 69, 7), '>', info_18146, int_18147)
    
    # Testing the type of an if condition (line 69)
    if_condition_18149 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 69, 4), result_gt_18148)
    # Assigning a type to the variable 'if_condition_18149' (line 69)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'if_condition_18149', if_condition_18149)
    # SSA begins for if statement (line 69)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to warn(...): (line 70)
    # Processing the call arguments (line 70)
    str_18151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 13), 'str', 'Diagonal number %d is exactly zero. Singular matrix.')
    # Getting the type of 'info' (line 70)
    info_18152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 70), 'info', False)
    # Applying the binary operator '%' (line 70)
    result_mod_18153 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 13), '%', str_18151, info_18152)
    
    # Getting the type of 'RuntimeWarning' (line 71)
    RuntimeWarning_18154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 20), 'RuntimeWarning', False)
    # Processing the call keyword arguments (line 70)
    kwargs_18155 = {}
    # Getting the type of 'warn' (line 70)
    warn_18150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'warn', False)
    # Calling warn(args, kwargs) (line 70)
    warn_call_result_18156 = invoke(stypy.reporting.localization.Localization(__file__, 70, 8), warn_18150, *[result_mod_18153, RuntimeWarning_18154], **kwargs_18155)
    
    # SSA join for if statement (line 69)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 72)
    tuple_18157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 72)
    # Adding element type (line 72)
    # Getting the type of 'lu' (line 72)
    lu_18158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 11), 'lu')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 11), tuple_18157, lu_18158)
    # Adding element type (line 72)
    # Getting the type of 'piv' (line 72)
    piv_18159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 15), 'piv')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 11), tuple_18157, piv_18159)
    
    # Assigning a type to the variable 'stypy_return_type' (line 72)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'stypy_return_type', tuple_18157)
    
    # ################# End of 'lu_factor(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'lu_factor' in the type store
    # Getting the type of 'stypy_return_type' (line 17)
    stypy_return_type_18160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_18160)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'lu_factor'
    return stypy_return_type_18160

# Assigning a type to the variable 'lu_factor' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'lu_factor', lu_factor)

@norecursion
def lu_solve(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_18161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 34), 'int')
    # Getting the type of 'False' (line 75)
    False_18162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 49), 'False')
    # Getting the type of 'True' (line 75)
    True_18163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 69), 'True')
    defaults = [int_18161, False_18162, True_18163]
    # Create a new context for function 'lu_solve'
    module_type_store = module_type_store.open_function_context('lu_solve', 75, 0, False)
    
    # Passed parameters checking function
    lu_solve.stypy_localization = localization
    lu_solve.stypy_type_of_self = None
    lu_solve.stypy_type_store = module_type_store
    lu_solve.stypy_function_name = 'lu_solve'
    lu_solve.stypy_param_names_list = ['lu_and_piv', 'b', 'trans', 'overwrite_b', 'check_finite']
    lu_solve.stypy_varargs_param_name = None
    lu_solve.stypy_kwargs_param_name = None
    lu_solve.stypy_call_defaults = defaults
    lu_solve.stypy_call_varargs = varargs
    lu_solve.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'lu_solve', ['lu_and_piv', 'b', 'trans', 'overwrite_b', 'check_finite'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'lu_solve', localization, ['lu_and_piv', 'b', 'trans', 'overwrite_b', 'check_finite'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'lu_solve(...)' code ##################

    str_18164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, (-1)), 'str', 'Solve an equation system, a x = b, given the LU factorization of a\n\n    Parameters\n    ----------\n    (lu, piv)\n        Factorization of the coefficient matrix a, as given by lu_factor\n    b : array\n        Right-hand side\n    trans : {0, 1, 2}, optional\n        Type of system to solve:\n\n        =====  =========\n        trans  system\n        =====  =========\n        0      a x   = b\n        1      a^T x = b\n        2      a^H x = b\n        =====  =========\n    overwrite_b : bool, optional\n        Whether to overwrite data in b (may increase performance)\n    check_finite : bool, optional\n        Whether to check that the input matrices contain only finite numbers.\n        Disabling may give a performance gain, but may result in problems\n        (crashes, non-termination) if the inputs do contain infinities or NaNs.\n\n    Returns\n    -------\n    x : array\n        Solution to the system\n\n    See also\n    --------\n    lu_factor : LU factorize a matrix\n\n    ')
    
    # Assigning a Name to a Tuple (line 111):
    
    # Assigning a Subscript to a Name (line 111):
    
    # Obtaining the type of the subscript
    int_18165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 4), 'int')
    # Getting the type of 'lu_and_piv' (line 111)
    lu_and_piv_18166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 16), 'lu_and_piv')
    # Obtaining the member '__getitem__' of a type (line 111)
    getitem___18167 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 4), lu_and_piv_18166, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 111)
    subscript_call_result_18168 = invoke(stypy.reporting.localization.Localization(__file__, 111, 4), getitem___18167, int_18165)
    
    # Assigning a type to the variable 'tuple_var_assignment_18027' (line 111)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 4), 'tuple_var_assignment_18027', subscript_call_result_18168)
    
    # Assigning a Subscript to a Name (line 111):
    
    # Obtaining the type of the subscript
    int_18169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 4), 'int')
    # Getting the type of 'lu_and_piv' (line 111)
    lu_and_piv_18170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 16), 'lu_and_piv')
    # Obtaining the member '__getitem__' of a type (line 111)
    getitem___18171 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 4), lu_and_piv_18170, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 111)
    subscript_call_result_18172 = invoke(stypy.reporting.localization.Localization(__file__, 111, 4), getitem___18171, int_18169)
    
    # Assigning a type to the variable 'tuple_var_assignment_18028' (line 111)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 4), 'tuple_var_assignment_18028', subscript_call_result_18172)
    
    # Assigning a Name to a Name (line 111):
    # Getting the type of 'tuple_var_assignment_18027' (line 111)
    tuple_var_assignment_18027_18173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 4), 'tuple_var_assignment_18027')
    # Assigning a type to the variable 'lu' (line 111)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 5), 'lu', tuple_var_assignment_18027_18173)
    
    # Assigning a Name to a Name (line 111):
    # Getting the type of 'tuple_var_assignment_18028' (line 111)
    tuple_var_assignment_18028_18174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 4), 'tuple_var_assignment_18028')
    # Assigning a type to the variable 'piv' (line 111)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 9), 'piv', tuple_var_assignment_18028_18174)
    
    # Getting the type of 'check_finite' (line 112)
    check_finite_18175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 7), 'check_finite')
    # Testing the type of an if condition (line 112)
    if_condition_18176 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 112, 4), check_finite_18175)
    # Assigning a type to the variable 'if_condition_18176' (line 112)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'if_condition_18176', if_condition_18176)
    # SSA begins for if statement (line 112)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 113):
    
    # Assigning a Call to a Name (line 113):
    
    # Call to asarray_chkfinite(...): (line 113)
    # Processing the call arguments (line 113)
    # Getting the type of 'b' (line 113)
    b_18178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 31), 'b', False)
    # Processing the call keyword arguments (line 113)
    kwargs_18179 = {}
    # Getting the type of 'asarray_chkfinite' (line 113)
    asarray_chkfinite_18177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 13), 'asarray_chkfinite', False)
    # Calling asarray_chkfinite(args, kwargs) (line 113)
    asarray_chkfinite_call_result_18180 = invoke(stypy.reporting.localization.Localization(__file__, 113, 13), asarray_chkfinite_18177, *[b_18178], **kwargs_18179)
    
    # Assigning a type to the variable 'b1' (line 113)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'b1', asarray_chkfinite_call_result_18180)
    # SSA branch for the else part of an if statement (line 112)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 115):
    
    # Assigning a Call to a Name (line 115):
    
    # Call to asarray(...): (line 115)
    # Processing the call arguments (line 115)
    # Getting the type of 'b' (line 115)
    b_18182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 21), 'b', False)
    # Processing the call keyword arguments (line 115)
    kwargs_18183 = {}
    # Getting the type of 'asarray' (line 115)
    asarray_18181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 13), 'asarray', False)
    # Calling asarray(args, kwargs) (line 115)
    asarray_call_result_18184 = invoke(stypy.reporting.localization.Localization(__file__, 115, 13), asarray_18181, *[b_18182], **kwargs_18183)
    
    # Assigning a type to the variable 'b1' (line 115)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 8), 'b1', asarray_call_result_18184)
    # SSA join for if statement (line 112)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BoolOp to a Name (line 116):
    
    # Assigning a BoolOp to a Name (line 116):
    
    # Evaluating a boolean operation
    # Getting the type of 'overwrite_b' (line 116)
    overwrite_b_18185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 18), 'overwrite_b')
    
    # Call to _datacopied(...): (line 116)
    # Processing the call arguments (line 116)
    # Getting the type of 'b1' (line 116)
    b1_18187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 45), 'b1', False)
    # Getting the type of 'b' (line 116)
    b_18188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 49), 'b', False)
    # Processing the call keyword arguments (line 116)
    kwargs_18189 = {}
    # Getting the type of '_datacopied' (line 116)
    _datacopied_18186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 33), '_datacopied', False)
    # Calling _datacopied(args, kwargs) (line 116)
    _datacopied_call_result_18190 = invoke(stypy.reporting.localization.Localization(__file__, 116, 33), _datacopied_18186, *[b1_18187, b_18188], **kwargs_18189)
    
    # Applying the binary operator 'or' (line 116)
    result_or_keyword_18191 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 18), 'or', overwrite_b_18185, _datacopied_call_result_18190)
    
    # Assigning a type to the variable 'overwrite_b' (line 116)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 4), 'overwrite_b', result_or_keyword_18191)
    
    
    
    # Obtaining the type of the subscript
    int_18192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 16), 'int')
    # Getting the type of 'lu' (line 117)
    lu_18193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 7), 'lu')
    # Obtaining the member 'shape' of a type (line 117)
    shape_18194 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 7), lu_18193, 'shape')
    # Obtaining the member '__getitem__' of a type (line 117)
    getitem___18195 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 7), shape_18194, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 117)
    subscript_call_result_18196 = invoke(stypy.reporting.localization.Localization(__file__, 117, 7), getitem___18195, int_18192)
    
    
    # Obtaining the type of the subscript
    int_18197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 31), 'int')
    # Getting the type of 'b1' (line 117)
    b1_18198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 22), 'b1')
    # Obtaining the member 'shape' of a type (line 117)
    shape_18199 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 22), b1_18198, 'shape')
    # Obtaining the member '__getitem__' of a type (line 117)
    getitem___18200 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 22), shape_18199, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 117)
    subscript_call_result_18201 = invoke(stypy.reporting.localization.Localization(__file__, 117, 22), getitem___18200, int_18197)
    
    # Applying the binary operator '!=' (line 117)
    result_ne_18202 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 7), '!=', subscript_call_result_18196, subscript_call_result_18201)
    
    # Testing the type of an if condition (line 117)
    if_condition_18203 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 117, 4), result_ne_18202)
    # Assigning a type to the variable 'if_condition_18203' (line 117)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 4), 'if_condition_18203', if_condition_18203)
    # SSA begins for if statement (line 117)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 118)
    # Processing the call arguments (line 118)
    str_18205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 25), 'str', 'incompatible dimensions.')
    # Processing the call keyword arguments (line 118)
    kwargs_18206 = {}
    # Getting the type of 'ValueError' (line 118)
    ValueError_18204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 118)
    ValueError_call_result_18207 = invoke(stypy.reporting.localization.Localization(__file__, 118, 14), ValueError_18204, *[str_18205], **kwargs_18206)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 118, 8), ValueError_call_result_18207, 'raise parameter', BaseException)
    # SSA join for if statement (line 117)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 120):
    
    # Assigning a Subscript to a Name (line 120):
    
    # Obtaining the type of the subscript
    int_18208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 4), 'int')
    
    # Call to get_lapack_funcs(...): (line 120)
    # Processing the call arguments (line 120)
    
    # Obtaining an instance of the builtin type 'tuple' (line 120)
    tuple_18210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 31), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 120)
    # Adding element type (line 120)
    str_18211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 31), 'str', 'getrs')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 31), tuple_18210, str_18211)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 120)
    tuple_18212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 43), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 120)
    # Adding element type (line 120)
    # Getting the type of 'lu' (line 120)
    lu_18213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 43), 'lu', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 43), tuple_18212, lu_18213)
    # Adding element type (line 120)
    # Getting the type of 'b1' (line 120)
    b1_18214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 47), 'b1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 43), tuple_18212, b1_18214)
    
    # Processing the call keyword arguments (line 120)
    kwargs_18215 = {}
    # Getting the type of 'get_lapack_funcs' (line 120)
    get_lapack_funcs_18209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 13), 'get_lapack_funcs', False)
    # Calling get_lapack_funcs(args, kwargs) (line 120)
    get_lapack_funcs_call_result_18216 = invoke(stypy.reporting.localization.Localization(__file__, 120, 13), get_lapack_funcs_18209, *[tuple_18210, tuple_18212], **kwargs_18215)
    
    # Obtaining the member '__getitem__' of a type (line 120)
    getitem___18217 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 4), get_lapack_funcs_call_result_18216, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 120)
    subscript_call_result_18218 = invoke(stypy.reporting.localization.Localization(__file__, 120, 4), getitem___18217, int_18208)
    
    # Assigning a type to the variable 'tuple_var_assignment_18029' (line 120)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 4), 'tuple_var_assignment_18029', subscript_call_result_18218)
    
    # Assigning a Name to a Name (line 120):
    # Getting the type of 'tuple_var_assignment_18029' (line 120)
    tuple_var_assignment_18029_18219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 4), 'tuple_var_assignment_18029')
    # Assigning a type to the variable 'getrs' (line 120)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 4), 'getrs', tuple_var_assignment_18029_18219)
    
    # Assigning a Call to a Tuple (line 121):
    
    # Assigning a Subscript to a Name (line 121):
    
    # Obtaining the type of the subscript
    int_18220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 4), 'int')
    
    # Call to getrs(...): (line 121)
    # Processing the call arguments (line 121)
    # Getting the type of 'lu' (line 121)
    lu_18222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 19), 'lu', False)
    # Getting the type of 'piv' (line 121)
    piv_18223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 23), 'piv', False)
    # Getting the type of 'b1' (line 121)
    b1_18224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 28), 'b1', False)
    # Processing the call keyword arguments (line 121)
    # Getting the type of 'trans' (line 121)
    trans_18225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 38), 'trans', False)
    keyword_18226 = trans_18225
    # Getting the type of 'overwrite_b' (line 121)
    overwrite_b_18227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 57), 'overwrite_b', False)
    keyword_18228 = overwrite_b_18227
    kwargs_18229 = {'trans': keyword_18226, 'overwrite_b': keyword_18228}
    # Getting the type of 'getrs' (line 121)
    getrs_18221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 13), 'getrs', False)
    # Calling getrs(args, kwargs) (line 121)
    getrs_call_result_18230 = invoke(stypy.reporting.localization.Localization(__file__, 121, 13), getrs_18221, *[lu_18222, piv_18223, b1_18224], **kwargs_18229)
    
    # Obtaining the member '__getitem__' of a type (line 121)
    getitem___18231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 4), getrs_call_result_18230, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 121)
    subscript_call_result_18232 = invoke(stypy.reporting.localization.Localization(__file__, 121, 4), getitem___18231, int_18220)
    
    # Assigning a type to the variable 'tuple_var_assignment_18030' (line 121)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 4), 'tuple_var_assignment_18030', subscript_call_result_18232)
    
    # Assigning a Subscript to a Name (line 121):
    
    # Obtaining the type of the subscript
    int_18233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 4), 'int')
    
    # Call to getrs(...): (line 121)
    # Processing the call arguments (line 121)
    # Getting the type of 'lu' (line 121)
    lu_18235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 19), 'lu', False)
    # Getting the type of 'piv' (line 121)
    piv_18236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 23), 'piv', False)
    # Getting the type of 'b1' (line 121)
    b1_18237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 28), 'b1', False)
    # Processing the call keyword arguments (line 121)
    # Getting the type of 'trans' (line 121)
    trans_18238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 38), 'trans', False)
    keyword_18239 = trans_18238
    # Getting the type of 'overwrite_b' (line 121)
    overwrite_b_18240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 57), 'overwrite_b', False)
    keyword_18241 = overwrite_b_18240
    kwargs_18242 = {'trans': keyword_18239, 'overwrite_b': keyword_18241}
    # Getting the type of 'getrs' (line 121)
    getrs_18234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 13), 'getrs', False)
    # Calling getrs(args, kwargs) (line 121)
    getrs_call_result_18243 = invoke(stypy.reporting.localization.Localization(__file__, 121, 13), getrs_18234, *[lu_18235, piv_18236, b1_18237], **kwargs_18242)
    
    # Obtaining the member '__getitem__' of a type (line 121)
    getitem___18244 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 4), getrs_call_result_18243, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 121)
    subscript_call_result_18245 = invoke(stypy.reporting.localization.Localization(__file__, 121, 4), getitem___18244, int_18233)
    
    # Assigning a type to the variable 'tuple_var_assignment_18031' (line 121)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 4), 'tuple_var_assignment_18031', subscript_call_result_18245)
    
    # Assigning a Name to a Name (line 121):
    # Getting the type of 'tuple_var_assignment_18030' (line 121)
    tuple_var_assignment_18030_18246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 4), 'tuple_var_assignment_18030')
    # Assigning a type to the variable 'x' (line 121)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 4), 'x', tuple_var_assignment_18030_18246)
    
    # Assigning a Name to a Name (line 121):
    # Getting the type of 'tuple_var_assignment_18031' (line 121)
    tuple_var_assignment_18031_18247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 4), 'tuple_var_assignment_18031')
    # Assigning a type to the variable 'info' (line 121)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 6), 'info', tuple_var_assignment_18031_18247)
    
    
    # Getting the type of 'info' (line 122)
    info_18248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 7), 'info')
    int_18249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 15), 'int')
    # Applying the binary operator '==' (line 122)
    result_eq_18250 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 7), '==', info_18248, int_18249)
    
    # Testing the type of an if condition (line 122)
    if_condition_18251 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 122, 4), result_eq_18250)
    # Assigning a type to the variable 'if_condition_18251' (line 122)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 4), 'if_condition_18251', if_condition_18251)
    # SSA begins for if statement (line 122)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'x' (line 123)
    x_18252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 15), 'x')
    # Assigning a type to the variable 'stypy_return_type' (line 123)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'stypy_return_type', x_18252)
    # SSA join for if statement (line 122)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to ValueError(...): (line 124)
    # Processing the call arguments (line 124)
    str_18254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 21), 'str', 'illegal value in %d-th argument of internal gesv|posv')
    
    # Getting the type of 'info' (line 125)
    info_18255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 71), 'info', False)
    # Applying the 'usub' unary operator (line 125)
    result___neg___18256 = python_operator(stypy.reporting.localization.Localization(__file__, 125, 70), 'usub', info_18255)
    
    # Applying the binary operator '%' (line 124)
    result_mod_18257 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 21), '%', str_18254, result___neg___18256)
    
    # Processing the call keyword arguments (line 124)
    kwargs_18258 = {}
    # Getting the type of 'ValueError' (line 124)
    ValueError_18253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 10), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 124)
    ValueError_call_result_18259 = invoke(stypy.reporting.localization.Localization(__file__, 124, 10), ValueError_18253, *[result_mod_18257], **kwargs_18258)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 124, 4), ValueError_call_result_18259, 'raise parameter', BaseException)
    
    # ################# End of 'lu_solve(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'lu_solve' in the type store
    # Getting the type of 'stypy_return_type' (line 75)
    stypy_return_type_18260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_18260)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'lu_solve'
    return stypy_return_type_18260

# Assigning a type to the variable 'lu_solve' (line 75)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 0), 'lu_solve', lu_solve)

@norecursion
def lu(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'False' (line 128)
    False_18261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 20), 'False')
    # Getting the type of 'False' (line 128)
    False_18262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 39), 'False')
    # Getting the type of 'True' (line 128)
    True_18263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 59), 'True')
    defaults = [False_18261, False_18262, True_18263]
    # Create a new context for function 'lu'
    module_type_store = module_type_store.open_function_context('lu', 128, 0, False)
    
    # Passed parameters checking function
    lu.stypy_localization = localization
    lu.stypy_type_of_self = None
    lu.stypy_type_store = module_type_store
    lu.stypy_function_name = 'lu'
    lu.stypy_param_names_list = ['a', 'permute_l', 'overwrite_a', 'check_finite']
    lu.stypy_varargs_param_name = None
    lu.stypy_kwargs_param_name = None
    lu.stypy_call_defaults = defaults
    lu.stypy_call_varargs = varargs
    lu.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'lu', ['a', 'permute_l', 'overwrite_a', 'check_finite'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'lu', localization, ['a', 'permute_l', 'overwrite_a', 'check_finite'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'lu(...)' code ##################

    str_18264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, (-1)), 'str', '\n    Compute pivoted LU decomposition of a matrix.\n\n    The decomposition is::\n\n        A = P L U\n\n    where P is a permutation matrix, L lower triangular with unit\n    diagonal elements, and U upper triangular.\n\n    Parameters\n    ----------\n    a : (M, N) array_like\n        Array to decompose\n    permute_l : bool, optional\n        Perform the multiplication P*L  (Default: do not permute)\n    overwrite_a : bool, optional\n        Whether to overwrite data in a (may improve performance)\n    check_finite : bool, optional\n        Whether to check that the input matrix contains only finite numbers.\n        Disabling may give a performance gain, but may result in problems\n        (crashes, non-termination) if the inputs do contain infinities or NaNs.\n\n    Returns\n    -------\n    **(If permute_l == False)**\n\n    p : (M, M) ndarray\n        Permutation matrix\n    l : (M, K) ndarray\n        Lower triangular or trapezoidal matrix with unit diagonal.\n        K = min(M, N)\n    u : (K, N) ndarray\n        Upper triangular or trapezoidal matrix\n\n    **(If permute_l == True)**\n\n    pl : (M, K) ndarray\n        Permuted L matrix.\n        K = min(M, N)\n    u : (K, N) ndarray\n        Upper triangular or trapezoidal matrix\n\n    Notes\n    -----\n    This is a LU factorization routine written for Scipy.\n\n    ')
    
    # Getting the type of 'check_finite' (line 177)
    check_finite_18265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 7), 'check_finite')
    # Testing the type of an if condition (line 177)
    if_condition_18266 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 177, 4), check_finite_18265)
    # Assigning a type to the variable 'if_condition_18266' (line 177)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 4), 'if_condition_18266', if_condition_18266)
    # SSA begins for if statement (line 177)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 178):
    
    # Assigning a Call to a Name (line 178):
    
    # Call to asarray_chkfinite(...): (line 178)
    # Processing the call arguments (line 178)
    # Getting the type of 'a' (line 178)
    a_18268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 31), 'a', False)
    # Processing the call keyword arguments (line 178)
    kwargs_18269 = {}
    # Getting the type of 'asarray_chkfinite' (line 178)
    asarray_chkfinite_18267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 13), 'asarray_chkfinite', False)
    # Calling asarray_chkfinite(args, kwargs) (line 178)
    asarray_chkfinite_call_result_18270 = invoke(stypy.reporting.localization.Localization(__file__, 178, 13), asarray_chkfinite_18267, *[a_18268], **kwargs_18269)
    
    # Assigning a type to the variable 'a1' (line 178)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'a1', asarray_chkfinite_call_result_18270)
    # SSA branch for the else part of an if statement (line 177)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 180):
    
    # Assigning a Call to a Name (line 180):
    
    # Call to asarray(...): (line 180)
    # Processing the call arguments (line 180)
    # Getting the type of 'a' (line 180)
    a_18272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 21), 'a', False)
    # Processing the call keyword arguments (line 180)
    kwargs_18273 = {}
    # Getting the type of 'asarray' (line 180)
    asarray_18271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 13), 'asarray', False)
    # Calling asarray(args, kwargs) (line 180)
    asarray_call_result_18274 = invoke(stypy.reporting.localization.Localization(__file__, 180, 13), asarray_18271, *[a_18272], **kwargs_18273)
    
    # Assigning a type to the variable 'a1' (line 180)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 8), 'a1', asarray_call_result_18274)
    # SSA join for if statement (line 177)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to len(...): (line 181)
    # Processing the call arguments (line 181)
    # Getting the type of 'a1' (line 181)
    a1_18276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 11), 'a1', False)
    # Obtaining the member 'shape' of a type (line 181)
    shape_18277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 11), a1_18276, 'shape')
    # Processing the call keyword arguments (line 181)
    kwargs_18278 = {}
    # Getting the type of 'len' (line 181)
    len_18275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 7), 'len', False)
    # Calling len(args, kwargs) (line 181)
    len_call_result_18279 = invoke(stypy.reporting.localization.Localization(__file__, 181, 7), len_18275, *[shape_18277], **kwargs_18278)
    
    int_18280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 24), 'int')
    # Applying the binary operator '!=' (line 181)
    result_ne_18281 = python_operator(stypy.reporting.localization.Localization(__file__, 181, 7), '!=', len_call_result_18279, int_18280)
    
    # Testing the type of an if condition (line 181)
    if_condition_18282 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 181, 4), result_ne_18281)
    # Assigning a type to the variable 'if_condition_18282' (line 181)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 4), 'if_condition_18282', if_condition_18282)
    # SSA begins for if statement (line 181)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 182)
    # Processing the call arguments (line 182)
    str_18284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 25), 'str', 'expected matrix')
    # Processing the call keyword arguments (line 182)
    kwargs_18285 = {}
    # Getting the type of 'ValueError' (line 182)
    ValueError_18283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 182)
    ValueError_call_result_18286 = invoke(stypy.reporting.localization.Localization(__file__, 182, 14), ValueError_18283, *[str_18284], **kwargs_18285)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 182, 8), ValueError_call_result_18286, 'raise parameter', BaseException)
    # SSA join for if statement (line 181)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BoolOp to a Name (line 183):
    
    # Assigning a BoolOp to a Name (line 183):
    
    # Evaluating a boolean operation
    # Getting the type of 'overwrite_a' (line 183)
    overwrite_a_18287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 18), 'overwrite_a')
    
    # Call to _datacopied(...): (line 183)
    # Processing the call arguments (line 183)
    # Getting the type of 'a1' (line 183)
    a1_18289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 46), 'a1', False)
    # Getting the type of 'a' (line 183)
    a_18290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 50), 'a', False)
    # Processing the call keyword arguments (line 183)
    kwargs_18291 = {}
    # Getting the type of '_datacopied' (line 183)
    _datacopied_18288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 34), '_datacopied', False)
    # Calling _datacopied(args, kwargs) (line 183)
    _datacopied_call_result_18292 = invoke(stypy.reporting.localization.Localization(__file__, 183, 34), _datacopied_18288, *[a1_18289, a_18290], **kwargs_18291)
    
    # Applying the binary operator 'or' (line 183)
    result_or_keyword_18293 = python_operator(stypy.reporting.localization.Localization(__file__, 183, 18), 'or', overwrite_a_18287, _datacopied_call_result_18292)
    
    # Assigning a type to the variable 'overwrite_a' (line 183)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 4), 'overwrite_a', result_or_keyword_18293)
    
    # Assigning a Call to a Tuple (line 184):
    
    # Assigning a Subscript to a Name (line 184):
    
    # Obtaining the type of the subscript
    int_18294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 4), 'int')
    
    # Call to get_flinalg_funcs(...): (line 184)
    # Processing the call arguments (line 184)
    
    # Obtaining an instance of the builtin type 'tuple' (line 184)
    tuple_18296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 30), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 184)
    # Adding element type (line 184)
    str_18297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 30), 'str', 'lu')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 184, 30), tuple_18296, str_18297)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 184)
    tuple_18298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 39), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 184)
    # Adding element type (line 184)
    # Getting the type of 'a1' (line 184)
    a1_18299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 39), 'a1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 184, 39), tuple_18298, a1_18299)
    
    # Processing the call keyword arguments (line 184)
    kwargs_18300 = {}
    # Getting the type of 'get_flinalg_funcs' (line 184)
    get_flinalg_funcs_18295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 11), 'get_flinalg_funcs', False)
    # Calling get_flinalg_funcs(args, kwargs) (line 184)
    get_flinalg_funcs_call_result_18301 = invoke(stypy.reporting.localization.Localization(__file__, 184, 11), get_flinalg_funcs_18295, *[tuple_18296, tuple_18298], **kwargs_18300)
    
    # Obtaining the member '__getitem__' of a type (line 184)
    getitem___18302 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 4), get_flinalg_funcs_call_result_18301, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 184)
    subscript_call_result_18303 = invoke(stypy.reporting.localization.Localization(__file__, 184, 4), getitem___18302, int_18294)
    
    # Assigning a type to the variable 'tuple_var_assignment_18032' (line 184)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 4), 'tuple_var_assignment_18032', subscript_call_result_18303)
    
    # Assigning a Name to a Name (line 184):
    # Getting the type of 'tuple_var_assignment_18032' (line 184)
    tuple_var_assignment_18032_18304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 4), 'tuple_var_assignment_18032')
    # Assigning a type to the variable 'flu' (line 184)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 4), 'flu', tuple_var_assignment_18032_18304)
    
    # Assigning a Call to a Tuple (line 185):
    
    # Assigning a Subscript to a Name (line 185):
    
    # Obtaining the type of the subscript
    int_18305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 4), 'int')
    
    # Call to flu(...): (line 185)
    # Processing the call arguments (line 185)
    # Getting the type of 'a1' (line 185)
    a1_18307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 24), 'a1', False)
    # Processing the call keyword arguments (line 185)
    # Getting the type of 'permute_l' (line 185)
    permute_l_18308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 38), 'permute_l', False)
    keyword_18309 = permute_l_18308
    # Getting the type of 'overwrite_a' (line 185)
    overwrite_a_18310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 61), 'overwrite_a', False)
    keyword_18311 = overwrite_a_18310
    kwargs_18312 = {'overwrite_a': keyword_18311, 'permute_l': keyword_18309}
    # Getting the type of 'flu' (line 185)
    flu_18306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 20), 'flu', False)
    # Calling flu(args, kwargs) (line 185)
    flu_call_result_18313 = invoke(stypy.reporting.localization.Localization(__file__, 185, 20), flu_18306, *[a1_18307], **kwargs_18312)
    
    # Obtaining the member '__getitem__' of a type (line 185)
    getitem___18314 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 4), flu_call_result_18313, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 185)
    subscript_call_result_18315 = invoke(stypy.reporting.localization.Localization(__file__, 185, 4), getitem___18314, int_18305)
    
    # Assigning a type to the variable 'tuple_var_assignment_18033' (line 185)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 4), 'tuple_var_assignment_18033', subscript_call_result_18315)
    
    # Assigning a Subscript to a Name (line 185):
    
    # Obtaining the type of the subscript
    int_18316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 4), 'int')
    
    # Call to flu(...): (line 185)
    # Processing the call arguments (line 185)
    # Getting the type of 'a1' (line 185)
    a1_18318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 24), 'a1', False)
    # Processing the call keyword arguments (line 185)
    # Getting the type of 'permute_l' (line 185)
    permute_l_18319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 38), 'permute_l', False)
    keyword_18320 = permute_l_18319
    # Getting the type of 'overwrite_a' (line 185)
    overwrite_a_18321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 61), 'overwrite_a', False)
    keyword_18322 = overwrite_a_18321
    kwargs_18323 = {'overwrite_a': keyword_18322, 'permute_l': keyword_18320}
    # Getting the type of 'flu' (line 185)
    flu_18317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 20), 'flu', False)
    # Calling flu(args, kwargs) (line 185)
    flu_call_result_18324 = invoke(stypy.reporting.localization.Localization(__file__, 185, 20), flu_18317, *[a1_18318], **kwargs_18323)
    
    # Obtaining the member '__getitem__' of a type (line 185)
    getitem___18325 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 4), flu_call_result_18324, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 185)
    subscript_call_result_18326 = invoke(stypy.reporting.localization.Localization(__file__, 185, 4), getitem___18325, int_18316)
    
    # Assigning a type to the variable 'tuple_var_assignment_18034' (line 185)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 4), 'tuple_var_assignment_18034', subscript_call_result_18326)
    
    # Assigning a Subscript to a Name (line 185):
    
    # Obtaining the type of the subscript
    int_18327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 4), 'int')
    
    # Call to flu(...): (line 185)
    # Processing the call arguments (line 185)
    # Getting the type of 'a1' (line 185)
    a1_18329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 24), 'a1', False)
    # Processing the call keyword arguments (line 185)
    # Getting the type of 'permute_l' (line 185)
    permute_l_18330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 38), 'permute_l', False)
    keyword_18331 = permute_l_18330
    # Getting the type of 'overwrite_a' (line 185)
    overwrite_a_18332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 61), 'overwrite_a', False)
    keyword_18333 = overwrite_a_18332
    kwargs_18334 = {'overwrite_a': keyword_18333, 'permute_l': keyword_18331}
    # Getting the type of 'flu' (line 185)
    flu_18328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 20), 'flu', False)
    # Calling flu(args, kwargs) (line 185)
    flu_call_result_18335 = invoke(stypy.reporting.localization.Localization(__file__, 185, 20), flu_18328, *[a1_18329], **kwargs_18334)
    
    # Obtaining the member '__getitem__' of a type (line 185)
    getitem___18336 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 4), flu_call_result_18335, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 185)
    subscript_call_result_18337 = invoke(stypy.reporting.localization.Localization(__file__, 185, 4), getitem___18336, int_18327)
    
    # Assigning a type to the variable 'tuple_var_assignment_18035' (line 185)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 4), 'tuple_var_assignment_18035', subscript_call_result_18337)
    
    # Assigning a Subscript to a Name (line 185):
    
    # Obtaining the type of the subscript
    int_18338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 4), 'int')
    
    # Call to flu(...): (line 185)
    # Processing the call arguments (line 185)
    # Getting the type of 'a1' (line 185)
    a1_18340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 24), 'a1', False)
    # Processing the call keyword arguments (line 185)
    # Getting the type of 'permute_l' (line 185)
    permute_l_18341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 38), 'permute_l', False)
    keyword_18342 = permute_l_18341
    # Getting the type of 'overwrite_a' (line 185)
    overwrite_a_18343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 61), 'overwrite_a', False)
    keyword_18344 = overwrite_a_18343
    kwargs_18345 = {'overwrite_a': keyword_18344, 'permute_l': keyword_18342}
    # Getting the type of 'flu' (line 185)
    flu_18339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 20), 'flu', False)
    # Calling flu(args, kwargs) (line 185)
    flu_call_result_18346 = invoke(stypy.reporting.localization.Localization(__file__, 185, 20), flu_18339, *[a1_18340], **kwargs_18345)
    
    # Obtaining the member '__getitem__' of a type (line 185)
    getitem___18347 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 4), flu_call_result_18346, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 185)
    subscript_call_result_18348 = invoke(stypy.reporting.localization.Localization(__file__, 185, 4), getitem___18347, int_18338)
    
    # Assigning a type to the variable 'tuple_var_assignment_18036' (line 185)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 4), 'tuple_var_assignment_18036', subscript_call_result_18348)
    
    # Assigning a Name to a Name (line 185):
    # Getting the type of 'tuple_var_assignment_18033' (line 185)
    tuple_var_assignment_18033_18349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 4), 'tuple_var_assignment_18033')
    # Assigning a type to the variable 'p' (line 185)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 4), 'p', tuple_var_assignment_18033_18349)
    
    # Assigning a Name to a Name (line 185):
    # Getting the type of 'tuple_var_assignment_18034' (line 185)
    tuple_var_assignment_18034_18350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 4), 'tuple_var_assignment_18034')
    # Assigning a type to the variable 'l' (line 185)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 7), 'l', tuple_var_assignment_18034_18350)
    
    # Assigning a Name to a Name (line 185):
    # Getting the type of 'tuple_var_assignment_18035' (line 185)
    tuple_var_assignment_18035_18351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 4), 'tuple_var_assignment_18035')
    # Assigning a type to the variable 'u' (line 185)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 10), 'u', tuple_var_assignment_18035_18351)
    
    # Assigning a Name to a Name (line 185):
    # Getting the type of 'tuple_var_assignment_18036' (line 185)
    tuple_var_assignment_18036_18352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 4), 'tuple_var_assignment_18036')
    # Assigning a type to the variable 'info' (line 185)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 13), 'info', tuple_var_assignment_18036_18352)
    
    
    # Getting the type of 'info' (line 186)
    info_18353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 7), 'info')
    int_18354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 14), 'int')
    # Applying the binary operator '<' (line 186)
    result_lt_18355 = python_operator(stypy.reporting.localization.Localization(__file__, 186, 7), '<', info_18353, int_18354)
    
    # Testing the type of an if condition (line 186)
    if_condition_18356 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 186, 4), result_lt_18355)
    # Assigning a type to the variable 'if_condition_18356' (line 186)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 4), 'if_condition_18356', if_condition_18356)
    # SSA begins for if statement (line 186)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 187)
    # Processing the call arguments (line 187)
    str_18358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 25), 'str', 'illegal value in %d-th argument of internal lu.getrf')
    
    # Getting the type of 'info' (line 188)
    info_18359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 67), 'info', False)
    # Applying the 'usub' unary operator (line 188)
    result___neg___18360 = python_operator(stypy.reporting.localization.Localization(__file__, 188, 66), 'usub', info_18359)
    
    # Applying the binary operator '%' (line 187)
    result_mod_18361 = python_operator(stypy.reporting.localization.Localization(__file__, 187, 25), '%', str_18358, result___neg___18360)
    
    # Processing the call keyword arguments (line 187)
    kwargs_18362 = {}
    # Getting the type of 'ValueError' (line 187)
    ValueError_18357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 187)
    ValueError_call_result_18363 = invoke(stypy.reporting.localization.Localization(__file__, 187, 14), ValueError_18357, *[result_mod_18361], **kwargs_18362)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 187, 8), ValueError_call_result_18363, 'raise parameter', BaseException)
    # SSA join for if statement (line 186)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'permute_l' (line 189)
    permute_l_18364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 7), 'permute_l')
    # Testing the type of an if condition (line 189)
    if_condition_18365 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 189, 4), permute_l_18364)
    # Assigning a type to the variable 'if_condition_18365' (line 189)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 4), 'if_condition_18365', if_condition_18365)
    # SSA begins for if statement (line 189)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'tuple' (line 190)
    tuple_18366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 190)
    # Adding element type (line 190)
    # Getting the type of 'l' (line 190)
    l_18367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 15), 'l')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 190, 15), tuple_18366, l_18367)
    # Adding element type (line 190)
    # Getting the type of 'u' (line 190)
    u_18368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 18), 'u')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 190, 15), tuple_18366, u_18368)
    
    # Assigning a type to the variable 'stypy_return_type' (line 190)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'stypy_return_type', tuple_18366)
    # SSA join for if statement (line 189)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 191)
    tuple_18369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 191)
    # Adding element type (line 191)
    # Getting the type of 'p' (line 191)
    p_18370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 11), 'p')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 191, 11), tuple_18369, p_18370)
    # Adding element type (line 191)
    # Getting the type of 'l' (line 191)
    l_18371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 14), 'l')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 191, 11), tuple_18369, l_18371)
    # Adding element type (line 191)
    # Getting the type of 'u' (line 191)
    u_18372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 17), 'u')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 191, 11), tuple_18369, u_18372)
    
    # Assigning a type to the variable 'stypy_return_type' (line 191)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 4), 'stypy_return_type', tuple_18369)
    
    # ################# End of 'lu(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'lu' in the type store
    # Getting the type of 'stypy_return_type' (line 128)
    stypy_return_type_18373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_18373)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'lu'
    return stypy_return_type_18373

# Assigning a type to the variable 'lu' (line 128)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 0), 'lu', lu)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
