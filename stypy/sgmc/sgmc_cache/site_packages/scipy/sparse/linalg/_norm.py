
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Sparse matrix norms.
2: 
3: '''
4: from __future__ import division, print_function, absolute_import
5: 
6: import numpy as np
7: from scipy.sparse import issparse
8: 
9: from numpy.core import Inf, sqrt, abs
10: 
11: __all__ = ['norm']
12: 
13: 
14: def _sparse_frobenius_norm(x):
15:     if np.issubdtype(x.dtype, np.complexfloating):
16:         sqnorm = abs(x).power(2).sum()
17:     else:
18:         sqnorm = x.power(2).sum()
19:     return sqrt(sqnorm)
20: 
21: 
22: def norm(x, ord=None, axis=None):
23:     '''
24:     Norm of a sparse matrix
25: 
26:     This function is able to return one of seven different matrix norms,
27:     depending on the value of the ``ord`` parameter.
28: 
29:     Parameters
30:     ----------
31:     x : a sparse matrix
32:         Input sparse matrix.
33:     ord : {non-zero int, inf, -inf, 'fro'}, optional
34:         Order of the norm (see table under ``Notes``). inf means numpy's
35:         `inf` object.
36:     axis : {int, 2-tuple of ints, None}, optional
37:         If `axis` is an integer, it specifies the axis of `x` along which to
38:         compute the vector norms.  If `axis` is a 2-tuple, it specifies the
39:         axes that hold 2-D matrices, and the matrix norms of these matrices
40:         are computed.  If `axis` is None then either a vector norm (when `x`
41:         is 1-D) or a matrix norm (when `x` is 2-D) is returned.
42: 
43:     Returns
44:     -------
45:     n : float or ndarray
46: 
47:     Notes
48:     -----
49:     Some of the ord are not implemented because some associated functions like, 
50:     _multi_svd_norm, are not yet available for sparse matrix. 
51: 
52:     This docstring is modified based on numpy.linalg.norm. 
53:     https://github.com/numpy/numpy/blob/master/numpy/linalg/linalg.py 
54: 
55:     The following norms can be calculated:
56: 
57:     =====  ============================  
58:     ord    norm for sparse matrices             
59:     =====  ============================  
60:     None   Frobenius norm                
61:     'fro'  Frobenius norm                
62:     inf    max(sum(abs(x), axis=1))      
63:     -inf   min(sum(abs(x), axis=1))      
64:     0      abs(x).sum(axis=axis)                           
65:     1      max(sum(abs(x), axis=0))      
66:     -1     min(sum(abs(x), axis=0))      
67:     2      Not implemented  
68:     -2     Not implemented      
69:     other  Not implemented                               
70:     =====  ============================  
71: 
72:     The Frobenius norm is given by [1]_:
73: 
74:         :math:`||A||_F = [\\sum_{i,j} abs(a_{i,j})^2]^{1/2}`
75: 
76:     References
77:     ----------
78:     .. [1] G. H. Golub and C. F. Van Loan, *Matrix Computations*,
79:         Baltimore, MD, Johns Hopkins University Press, 1985, pg. 15
80: 
81:     Examples
82:     --------
83:     >>> from scipy.sparse import *
84:     >>> import numpy as np
85:     >>> from scipy.sparse.linalg import norm
86:     >>> a = np.arange(9) - 4
87:     >>> a
88:     array([-4, -3, -2, -1, 0, 1, 2, 3, 4])
89:     >>> b = a.reshape((3, 3))
90:     >>> b
91:     array([[-4, -3, -2],
92:            [-1, 0, 1],
93:            [ 2, 3, 4]])
94: 
95:     >>> b = csr_matrix(b)
96:     >>> norm(b)
97:     7.745966692414834
98:     >>> norm(b, 'fro')
99:     7.745966692414834
100:     >>> norm(b, np.inf)
101:     9
102:     >>> norm(b, -np.inf)
103:     2
104:     >>> norm(b, 1)
105:     7
106:     >>> norm(b, -1)
107:     6
108: 
109:     '''
110:     if not issparse(x):
111:         raise TypeError("input is not sparse. use numpy.linalg.norm")
112: 
113:     # Check the default case first and handle it immediately.
114:     if axis is None and ord in (None, 'fro', 'f'):
115:         return _sparse_frobenius_norm(x)
116: 
117:     # Some norms require functions that are not implemented for all types.
118:     x = x.tocsr()
119: 
120:     if axis is None:
121:         axis = (0, 1)
122:     elif not isinstance(axis, tuple):
123:         msg = "'axis' must be None, an integer or a tuple of integers"
124:         try:
125:             int_axis = int(axis)
126:         except TypeError:
127:             raise TypeError(msg)
128:         if axis != int_axis:
129:             raise TypeError(msg)
130:         axis = (int_axis,)
131: 
132:     nd = 2
133:     if len(axis) == 2:
134:         row_axis, col_axis = axis
135:         if not (-nd <= row_axis < nd and -nd <= col_axis < nd):
136:             raise ValueError('Invalid axis %r for an array with shape %r' %
137:                              (axis, x.shape))
138:         if row_axis % nd == col_axis % nd:
139:             raise ValueError('Duplicate axes given.')
140:         if ord == 2:
141:             raise NotImplementedError
142:             #return _multi_svd_norm(x, row_axis, col_axis, amax)
143:         elif ord == -2:
144:             raise NotImplementedError
145:             #return _multi_svd_norm(x, row_axis, col_axis, amin)
146:         elif ord == 1:
147:             return abs(x).sum(axis=row_axis).max(axis=col_axis)[0,0]
148:         elif ord == Inf:
149:             return abs(x).sum(axis=col_axis).max(axis=row_axis)[0,0]
150:         elif ord == -1:
151:             return abs(x).sum(axis=row_axis).min(axis=col_axis)[0,0]
152:         elif ord == -Inf:
153:             return abs(x).sum(axis=col_axis).min(axis=row_axis)[0,0]
154:         elif ord in (None, 'f', 'fro'):
155:             # The axis order does not matter for this norm.
156:             return _sparse_frobenius_norm(x)
157:         else:
158:             raise ValueError("Invalid norm order for matrices.")
159:     elif len(axis) == 1:
160:         a, = axis
161:         if not (-nd <= a < nd):
162:             raise ValueError('Invalid axis %r for an array with shape %r' %
163:                              (axis, x.shape))
164:         if ord == Inf:
165:             M = abs(x).max(axis=a)
166:         elif ord == -Inf:
167:             M = abs(x).min(axis=a)
168:         elif ord == 0:
169:             # Zero norm
170:             M = (x != 0).sum(axis=a)
171:         elif ord == 1:
172:             # special case for speedup
173:             M = abs(x).sum(axis=a)
174:         elif ord in (2, None):
175:             M = sqrt(abs(x).power(2).sum(axis=a))
176:         else:
177:             try:
178:                 ord + 1
179:             except TypeError:
180:                 raise ValueError('Invalid norm order for vectors.')
181:             M = np.power(abs(x).power(ord).sum(axis=a), 1 / ord)
182:         return M.A.ravel()
183:     else:
184:         raise ValueError("Improper number of dimensions to norm.")
185: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_390106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, (-1)), 'str', 'Sparse matrix norms.\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import numpy' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/')
import_390107 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy')

if (type(import_390107) is not StypyTypeError):

    if (import_390107 != 'pyd_module'):
        __import__(import_390107)
        sys_modules_390108 = sys.modules[import_390107]
        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'np', sys_modules_390108.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy', import_390107)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from scipy.sparse import issparse' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/')
import_390109 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.sparse')

if (type(import_390109) is not StypyTypeError):

    if (import_390109 != 'pyd_module'):
        __import__(import_390109)
        sys_modules_390110 = sys.modules[import_390109]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.sparse', sys_modules_390110.module_type_store, module_type_store, ['issparse'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_390110, sys_modules_390110.module_type_store, module_type_store)
    else:
        from scipy.sparse import issparse

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.sparse', None, module_type_store, ['issparse'], [issparse])

else:
    # Assigning a type to the variable 'scipy.sparse' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.sparse', import_390109)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from numpy.core import Inf, sqrt, abs' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/')
import_390111 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy.core')

if (type(import_390111) is not StypyTypeError):

    if (import_390111 != 'pyd_module'):
        __import__(import_390111)
        sys_modules_390112 = sys.modules[import_390111]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy.core', sys_modules_390112.module_type_store, module_type_store, ['Inf', 'sqrt', 'abs'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_390112, sys_modules_390112.module_type_store, module_type_store)
    else:
        from numpy.core import Inf, sqrt, abs

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy.core', None, module_type_store, ['Inf', 'sqrt', 'abs'], [Inf, sqrt, abs])

else:
    # Assigning a type to the variable 'numpy.core' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy.core', import_390111)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/')


# Assigning a List to a Name (line 11):

# Assigning a List to a Name (line 11):
__all__ = ['norm']
module_type_store.set_exportable_members(['norm'])

# Obtaining an instance of the builtin type 'list' (line 11)
list_390113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 11)
# Adding element type (line 11)
str_390114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 11), 'str', 'norm')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 10), list_390113, str_390114)

# Assigning a type to the variable '__all__' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), '__all__', list_390113)

@norecursion
def _sparse_frobenius_norm(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_sparse_frobenius_norm'
    module_type_store = module_type_store.open_function_context('_sparse_frobenius_norm', 14, 0, False)
    
    # Passed parameters checking function
    _sparse_frobenius_norm.stypy_localization = localization
    _sparse_frobenius_norm.stypy_type_of_self = None
    _sparse_frobenius_norm.stypy_type_store = module_type_store
    _sparse_frobenius_norm.stypy_function_name = '_sparse_frobenius_norm'
    _sparse_frobenius_norm.stypy_param_names_list = ['x']
    _sparse_frobenius_norm.stypy_varargs_param_name = None
    _sparse_frobenius_norm.stypy_kwargs_param_name = None
    _sparse_frobenius_norm.stypy_call_defaults = defaults
    _sparse_frobenius_norm.stypy_call_varargs = varargs
    _sparse_frobenius_norm.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_sparse_frobenius_norm', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_sparse_frobenius_norm', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_sparse_frobenius_norm(...)' code ##################

    
    
    # Call to issubdtype(...): (line 15)
    # Processing the call arguments (line 15)
    # Getting the type of 'x' (line 15)
    x_390117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 21), 'x', False)
    # Obtaining the member 'dtype' of a type (line 15)
    dtype_390118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 21), x_390117, 'dtype')
    # Getting the type of 'np' (line 15)
    np_390119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 30), 'np', False)
    # Obtaining the member 'complexfloating' of a type (line 15)
    complexfloating_390120 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 30), np_390119, 'complexfloating')
    # Processing the call keyword arguments (line 15)
    kwargs_390121 = {}
    # Getting the type of 'np' (line 15)
    np_390115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 7), 'np', False)
    # Obtaining the member 'issubdtype' of a type (line 15)
    issubdtype_390116 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 7), np_390115, 'issubdtype')
    # Calling issubdtype(args, kwargs) (line 15)
    issubdtype_call_result_390122 = invoke(stypy.reporting.localization.Localization(__file__, 15, 7), issubdtype_390116, *[dtype_390118, complexfloating_390120], **kwargs_390121)
    
    # Testing the type of an if condition (line 15)
    if_condition_390123 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 15, 4), issubdtype_call_result_390122)
    # Assigning a type to the variable 'if_condition_390123' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'if_condition_390123', if_condition_390123)
    # SSA begins for if statement (line 15)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 16):
    
    # Assigning a Call to a Name (line 16):
    
    # Call to sum(...): (line 16)
    # Processing the call keyword arguments (line 16)
    kwargs_390133 = {}
    
    # Call to power(...): (line 16)
    # Processing the call arguments (line 16)
    int_390129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 30), 'int')
    # Processing the call keyword arguments (line 16)
    kwargs_390130 = {}
    
    # Call to abs(...): (line 16)
    # Processing the call arguments (line 16)
    # Getting the type of 'x' (line 16)
    x_390125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 21), 'x', False)
    # Processing the call keyword arguments (line 16)
    kwargs_390126 = {}
    # Getting the type of 'abs' (line 16)
    abs_390124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 17), 'abs', False)
    # Calling abs(args, kwargs) (line 16)
    abs_call_result_390127 = invoke(stypy.reporting.localization.Localization(__file__, 16, 17), abs_390124, *[x_390125], **kwargs_390126)
    
    # Obtaining the member 'power' of a type (line 16)
    power_390128 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 17), abs_call_result_390127, 'power')
    # Calling power(args, kwargs) (line 16)
    power_call_result_390131 = invoke(stypy.reporting.localization.Localization(__file__, 16, 17), power_390128, *[int_390129], **kwargs_390130)
    
    # Obtaining the member 'sum' of a type (line 16)
    sum_390132 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 17), power_call_result_390131, 'sum')
    # Calling sum(args, kwargs) (line 16)
    sum_call_result_390134 = invoke(stypy.reporting.localization.Localization(__file__, 16, 17), sum_390132, *[], **kwargs_390133)
    
    # Assigning a type to the variable 'sqnorm' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'sqnorm', sum_call_result_390134)
    # SSA branch for the else part of an if statement (line 15)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 18):
    
    # Assigning a Call to a Name (line 18):
    
    # Call to sum(...): (line 18)
    # Processing the call keyword arguments (line 18)
    kwargs_390141 = {}
    
    # Call to power(...): (line 18)
    # Processing the call arguments (line 18)
    int_390137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 25), 'int')
    # Processing the call keyword arguments (line 18)
    kwargs_390138 = {}
    # Getting the type of 'x' (line 18)
    x_390135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 17), 'x', False)
    # Obtaining the member 'power' of a type (line 18)
    power_390136 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 17), x_390135, 'power')
    # Calling power(args, kwargs) (line 18)
    power_call_result_390139 = invoke(stypy.reporting.localization.Localization(__file__, 18, 17), power_390136, *[int_390137], **kwargs_390138)
    
    # Obtaining the member 'sum' of a type (line 18)
    sum_390140 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 17), power_call_result_390139, 'sum')
    # Calling sum(args, kwargs) (line 18)
    sum_call_result_390142 = invoke(stypy.reporting.localization.Localization(__file__, 18, 17), sum_390140, *[], **kwargs_390141)
    
    # Assigning a type to the variable 'sqnorm' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'sqnorm', sum_call_result_390142)
    # SSA join for if statement (line 15)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to sqrt(...): (line 19)
    # Processing the call arguments (line 19)
    # Getting the type of 'sqnorm' (line 19)
    sqnorm_390144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 16), 'sqnorm', False)
    # Processing the call keyword arguments (line 19)
    kwargs_390145 = {}
    # Getting the type of 'sqrt' (line 19)
    sqrt_390143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 11), 'sqrt', False)
    # Calling sqrt(args, kwargs) (line 19)
    sqrt_call_result_390146 = invoke(stypy.reporting.localization.Localization(__file__, 19, 11), sqrt_390143, *[sqnorm_390144], **kwargs_390145)
    
    # Assigning a type to the variable 'stypy_return_type' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'stypy_return_type', sqrt_call_result_390146)
    
    # ################# End of '_sparse_frobenius_norm(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_sparse_frobenius_norm' in the type store
    # Getting the type of 'stypy_return_type' (line 14)
    stypy_return_type_390147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_390147)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_sparse_frobenius_norm'
    return stypy_return_type_390147

# Assigning a type to the variable '_sparse_frobenius_norm' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), '_sparse_frobenius_norm', _sparse_frobenius_norm)

@norecursion
def norm(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 22)
    None_390148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 16), 'None')
    # Getting the type of 'None' (line 22)
    None_390149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 27), 'None')
    defaults = [None_390148, None_390149]
    # Create a new context for function 'norm'
    module_type_store = module_type_store.open_function_context('norm', 22, 0, False)
    
    # Passed parameters checking function
    norm.stypy_localization = localization
    norm.stypy_type_of_self = None
    norm.stypy_type_store = module_type_store
    norm.stypy_function_name = 'norm'
    norm.stypy_param_names_list = ['x', 'ord', 'axis']
    norm.stypy_varargs_param_name = None
    norm.stypy_kwargs_param_name = None
    norm.stypy_call_defaults = defaults
    norm.stypy_call_varargs = varargs
    norm.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'norm', ['x', 'ord', 'axis'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'norm', localization, ['x', 'ord', 'axis'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'norm(...)' code ##################

    str_390150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, (-1)), 'str', "\n    Norm of a sparse matrix\n\n    This function is able to return one of seven different matrix norms,\n    depending on the value of the ``ord`` parameter.\n\n    Parameters\n    ----------\n    x : a sparse matrix\n        Input sparse matrix.\n    ord : {non-zero int, inf, -inf, 'fro'}, optional\n        Order of the norm (see table under ``Notes``). inf means numpy's\n        `inf` object.\n    axis : {int, 2-tuple of ints, None}, optional\n        If `axis` is an integer, it specifies the axis of `x` along which to\n        compute the vector norms.  If `axis` is a 2-tuple, it specifies the\n        axes that hold 2-D matrices, and the matrix norms of these matrices\n        are computed.  If `axis` is None then either a vector norm (when `x`\n        is 1-D) or a matrix norm (when `x` is 2-D) is returned.\n\n    Returns\n    -------\n    n : float or ndarray\n\n    Notes\n    -----\n    Some of the ord are not implemented because some associated functions like, \n    _multi_svd_norm, are not yet available for sparse matrix. \n\n    This docstring is modified based on numpy.linalg.norm. \n    https://github.com/numpy/numpy/blob/master/numpy/linalg/linalg.py \n\n    The following norms can be calculated:\n\n    =====  ============================  \n    ord    norm for sparse matrices             \n    =====  ============================  \n    None   Frobenius norm                \n    'fro'  Frobenius norm                \n    inf    max(sum(abs(x), axis=1))      \n    -inf   min(sum(abs(x), axis=1))      \n    0      abs(x).sum(axis=axis)                           \n    1      max(sum(abs(x), axis=0))      \n    -1     min(sum(abs(x), axis=0))      \n    2      Not implemented  \n    -2     Not implemented      \n    other  Not implemented                               \n    =====  ============================  \n\n    The Frobenius norm is given by [1]_:\n\n        :math:`||A||_F = [\\sum_{i,j} abs(a_{i,j})^2]^{1/2}`\n\n    References\n    ----------\n    .. [1] G. H. Golub and C. F. Van Loan, *Matrix Computations*,\n        Baltimore, MD, Johns Hopkins University Press, 1985, pg. 15\n\n    Examples\n    --------\n    >>> from scipy.sparse import *\n    >>> import numpy as np\n    >>> from scipy.sparse.linalg import norm\n    >>> a = np.arange(9) - 4\n    >>> a\n    array([-4, -3, -2, -1, 0, 1, 2, 3, 4])\n    >>> b = a.reshape((3, 3))\n    >>> b\n    array([[-4, -3, -2],\n           [-1, 0, 1],\n           [ 2, 3, 4]])\n\n    >>> b = csr_matrix(b)\n    >>> norm(b)\n    7.745966692414834\n    >>> norm(b, 'fro')\n    7.745966692414834\n    >>> norm(b, np.inf)\n    9\n    >>> norm(b, -np.inf)\n    2\n    >>> norm(b, 1)\n    7\n    >>> norm(b, -1)\n    6\n\n    ")
    
    
    
    # Call to issparse(...): (line 110)
    # Processing the call arguments (line 110)
    # Getting the type of 'x' (line 110)
    x_390152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 20), 'x', False)
    # Processing the call keyword arguments (line 110)
    kwargs_390153 = {}
    # Getting the type of 'issparse' (line 110)
    issparse_390151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 11), 'issparse', False)
    # Calling issparse(args, kwargs) (line 110)
    issparse_call_result_390154 = invoke(stypy.reporting.localization.Localization(__file__, 110, 11), issparse_390151, *[x_390152], **kwargs_390153)
    
    # Applying the 'not' unary operator (line 110)
    result_not__390155 = python_operator(stypy.reporting.localization.Localization(__file__, 110, 7), 'not', issparse_call_result_390154)
    
    # Testing the type of an if condition (line 110)
    if_condition_390156 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 110, 4), result_not__390155)
    # Assigning a type to the variable 'if_condition_390156' (line 110)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 4), 'if_condition_390156', if_condition_390156)
    # SSA begins for if statement (line 110)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 111)
    # Processing the call arguments (line 111)
    str_390158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 24), 'str', 'input is not sparse. use numpy.linalg.norm')
    # Processing the call keyword arguments (line 111)
    kwargs_390159 = {}
    # Getting the type of 'TypeError' (line 111)
    TypeError_390157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 14), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 111)
    TypeError_call_result_390160 = invoke(stypy.reporting.localization.Localization(__file__, 111, 14), TypeError_390157, *[str_390158], **kwargs_390159)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 111, 8), TypeError_call_result_390160, 'raise parameter', BaseException)
    # SSA join for if statement (line 110)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'axis' (line 114)
    axis_390161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 7), 'axis')
    # Getting the type of 'None' (line 114)
    None_390162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 15), 'None')
    # Applying the binary operator 'is' (line 114)
    result_is__390163 = python_operator(stypy.reporting.localization.Localization(__file__, 114, 7), 'is', axis_390161, None_390162)
    
    
    # Getting the type of 'ord' (line 114)
    ord_390164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 24), 'ord')
    
    # Obtaining an instance of the builtin type 'tuple' (line 114)
    tuple_390165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 32), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 114)
    # Adding element type (line 114)
    # Getting the type of 'None' (line 114)
    None_390166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 32), 'None')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 32), tuple_390165, None_390166)
    # Adding element type (line 114)
    str_390167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 38), 'str', 'fro')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 32), tuple_390165, str_390167)
    # Adding element type (line 114)
    str_390168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 45), 'str', 'f')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 32), tuple_390165, str_390168)
    
    # Applying the binary operator 'in' (line 114)
    result_contains_390169 = python_operator(stypy.reporting.localization.Localization(__file__, 114, 24), 'in', ord_390164, tuple_390165)
    
    # Applying the binary operator 'and' (line 114)
    result_and_keyword_390170 = python_operator(stypy.reporting.localization.Localization(__file__, 114, 7), 'and', result_is__390163, result_contains_390169)
    
    # Testing the type of an if condition (line 114)
    if_condition_390171 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 114, 4), result_and_keyword_390170)
    # Assigning a type to the variable 'if_condition_390171' (line 114)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 4), 'if_condition_390171', if_condition_390171)
    # SSA begins for if statement (line 114)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to _sparse_frobenius_norm(...): (line 115)
    # Processing the call arguments (line 115)
    # Getting the type of 'x' (line 115)
    x_390173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 38), 'x', False)
    # Processing the call keyword arguments (line 115)
    kwargs_390174 = {}
    # Getting the type of '_sparse_frobenius_norm' (line 115)
    _sparse_frobenius_norm_390172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 15), '_sparse_frobenius_norm', False)
    # Calling _sparse_frobenius_norm(args, kwargs) (line 115)
    _sparse_frobenius_norm_call_result_390175 = invoke(stypy.reporting.localization.Localization(__file__, 115, 15), _sparse_frobenius_norm_390172, *[x_390173], **kwargs_390174)
    
    # Assigning a type to the variable 'stypy_return_type' (line 115)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 8), 'stypy_return_type', _sparse_frobenius_norm_call_result_390175)
    # SSA join for if statement (line 114)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 118):
    
    # Assigning a Call to a Name (line 118):
    
    # Call to tocsr(...): (line 118)
    # Processing the call keyword arguments (line 118)
    kwargs_390178 = {}
    # Getting the type of 'x' (line 118)
    x_390176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'x', False)
    # Obtaining the member 'tocsr' of a type (line 118)
    tocsr_390177 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 8), x_390176, 'tocsr')
    # Calling tocsr(args, kwargs) (line 118)
    tocsr_call_result_390179 = invoke(stypy.reporting.localization.Localization(__file__, 118, 8), tocsr_390177, *[], **kwargs_390178)
    
    # Assigning a type to the variable 'x' (line 118)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 4), 'x', tocsr_call_result_390179)
    
    # Type idiom detected: calculating its left and rigth part (line 120)
    # Getting the type of 'axis' (line 120)
    axis_390180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 7), 'axis')
    # Getting the type of 'None' (line 120)
    None_390181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 15), 'None')
    
    (may_be_390182, more_types_in_union_390183) = may_be_none(axis_390180, None_390181)

    if may_be_390182:

        if more_types_in_union_390183:
            # Runtime conditional SSA (line 120)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Tuple to a Name (line 121):
        
        # Assigning a Tuple to a Name (line 121):
        
        # Obtaining an instance of the builtin type 'tuple' (line 121)
        tuple_390184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 16), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 121)
        # Adding element type (line 121)
        int_390185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 16), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 16), tuple_390184, int_390185)
        # Adding element type (line 121)
        int_390186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 16), tuple_390184, int_390186)
        
        # Assigning a type to the variable 'axis' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'axis', tuple_390184)

        if more_types_in_union_390183:
            # Runtime conditional SSA for else branch (line 120)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_390182) or more_types_in_union_390183):
        
        # Type idiom detected: calculating its left and rigth part (line 122)
        # Getting the type of 'tuple' (line 122)
        tuple_390187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 30), 'tuple')
        # Getting the type of 'axis' (line 122)
        axis_390188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 24), 'axis')
        
        (may_be_390189, more_types_in_union_390190) = may_not_be_subtype(tuple_390187, axis_390188)

        if may_be_390189:

            if more_types_in_union_390190:
                # Runtime conditional SSA (line 122)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'axis' (line 122)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 9), 'axis', remove_subtype_from_union(axis_390188, tuple))
            
            # Assigning a Str to a Name (line 123):
            
            # Assigning a Str to a Name (line 123):
            str_390191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 14), 'str', "'axis' must be None, an integer or a tuple of integers")
            # Assigning a type to the variable 'msg' (line 123)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'msg', str_390191)
            
            
            # SSA begins for try-except statement (line 124)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
            
            # Assigning a Call to a Name (line 125):
            
            # Assigning a Call to a Name (line 125):
            
            # Call to int(...): (line 125)
            # Processing the call arguments (line 125)
            # Getting the type of 'axis' (line 125)
            axis_390193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 27), 'axis', False)
            # Processing the call keyword arguments (line 125)
            kwargs_390194 = {}
            # Getting the type of 'int' (line 125)
            int_390192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 23), 'int', False)
            # Calling int(args, kwargs) (line 125)
            int_call_result_390195 = invoke(stypy.reporting.localization.Localization(__file__, 125, 23), int_390192, *[axis_390193], **kwargs_390194)
            
            # Assigning a type to the variable 'int_axis' (line 125)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 12), 'int_axis', int_call_result_390195)
            # SSA branch for the except part of a try statement (line 124)
            # SSA branch for the except 'TypeError' branch of a try statement (line 124)
            module_type_store.open_ssa_branch('except')
            
            # Call to TypeError(...): (line 127)
            # Processing the call arguments (line 127)
            # Getting the type of 'msg' (line 127)
            msg_390197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 28), 'msg', False)
            # Processing the call keyword arguments (line 127)
            kwargs_390198 = {}
            # Getting the type of 'TypeError' (line 127)
            TypeError_390196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 18), 'TypeError', False)
            # Calling TypeError(args, kwargs) (line 127)
            TypeError_call_result_390199 = invoke(stypy.reporting.localization.Localization(__file__, 127, 18), TypeError_390196, *[msg_390197], **kwargs_390198)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 127, 12), TypeError_call_result_390199, 'raise parameter', BaseException)
            # SSA join for try-except statement (line 124)
            module_type_store = module_type_store.join_ssa_context()
            
            
            
            # Getting the type of 'axis' (line 128)
            axis_390200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 11), 'axis')
            # Getting the type of 'int_axis' (line 128)
            int_axis_390201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 19), 'int_axis')
            # Applying the binary operator '!=' (line 128)
            result_ne_390202 = python_operator(stypy.reporting.localization.Localization(__file__, 128, 11), '!=', axis_390200, int_axis_390201)
            
            # Testing the type of an if condition (line 128)
            if_condition_390203 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 128, 8), result_ne_390202)
            # Assigning a type to the variable 'if_condition_390203' (line 128)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'if_condition_390203', if_condition_390203)
            # SSA begins for if statement (line 128)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to TypeError(...): (line 129)
            # Processing the call arguments (line 129)
            # Getting the type of 'msg' (line 129)
            msg_390205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 28), 'msg', False)
            # Processing the call keyword arguments (line 129)
            kwargs_390206 = {}
            # Getting the type of 'TypeError' (line 129)
            TypeError_390204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 18), 'TypeError', False)
            # Calling TypeError(args, kwargs) (line 129)
            TypeError_call_result_390207 = invoke(stypy.reporting.localization.Localization(__file__, 129, 18), TypeError_390204, *[msg_390205], **kwargs_390206)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 129, 12), TypeError_call_result_390207, 'raise parameter', BaseException)
            # SSA join for if statement (line 128)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Assigning a Tuple to a Name (line 130):
            
            # Assigning a Tuple to a Name (line 130):
            
            # Obtaining an instance of the builtin type 'tuple' (line 130)
            tuple_390208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 16), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 130)
            # Adding element type (line 130)
            # Getting the type of 'int_axis' (line 130)
            int_axis_390209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 16), 'int_axis')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 130, 16), tuple_390208, int_axis_390209)
            
            # Assigning a type to the variable 'axis' (line 130)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 8), 'axis', tuple_390208)

            if more_types_in_union_390190:
                # SSA join for if statement (line 122)
                module_type_store = module_type_store.join_ssa_context()


        

        if (may_be_390182 and more_types_in_union_390183):
            # SSA join for if statement (line 120)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Num to a Name (line 132):
    
    # Assigning a Num to a Name (line 132):
    int_390210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 9), 'int')
    # Assigning a type to the variable 'nd' (line 132)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 4), 'nd', int_390210)
    
    
    
    # Call to len(...): (line 133)
    # Processing the call arguments (line 133)
    # Getting the type of 'axis' (line 133)
    axis_390212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 11), 'axis', False)
    # Processing the call keyword arguments (line 133)
    kwargs_390213 = {}
    # Getting the type of 'len' (line 133)
    len_390211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 7), 'len', False)
    # Calling len(args, kwargs) (line 133)
    len_call_result_390214 = invoke(stypy.reporting.localization.Localization(__file__, 133, 7), len_390211, *[axis_390212], **kwargs_390213)
    
    int_390215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 20), 'int')
    # Applying the binary operator '==' (line 133)
    result_eq_390216 = python_operator(stypy.reporting.localization.Localization(__file__, 133, 7), '==', len_call_result_390214, int_390215)
    
    # Testing the type of an if condition (line 133)
    if_condition_390217 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 133, 4), result_eq_390216)
    # Assigning a type to the variable 'if_condition_390217' (line 133)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 4), 'if_condition_390217', if_condition_390217)
    # SSA begins for if statement (line 133)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Tuple (line 134):
    
    # Assigning a Subscript to a Name (line 134):
    
    # Obtaining the type of the subscript
    int_390218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 8), 'int')
    # Getting the type of 'axis' (line 134)
    axis_390219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 29), 'axis')
    # Obtaining the member '__getitem__' of a type (line 134)
    getitem___390220 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 8), axis_390219, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 134)
    subscript_call_result_390221 = invoke(stypy.reporting.localization.Localization(__file__, 134, 8), getitem___390220, int_390218)
    
    # Assigning a type to the variable 'tuple_var_assignment_390103' (line 134)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'tuple_var_assignment_390103', subscript_call_result_390221)
    
    # Assigning a Subscript to a Name (line 134):
    
    # Obtaining the type of the subscript
    int_390222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 8), 'int')
    # Getting the type of 'axis' (line 134)
    axis_390223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 29), 'axis')
    # Obtaining the member '__getitem__' of a type (line 134)
    getitem___390224 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 8), axis_390223, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 134)
    subscript_call_result_390225 = invoke(stypy.reporting.localization.Localization(__file__, 134, 8), getitem___390224, int_390222)
    
    # Assigning a type to the variable 'tuple_var_assignment_390104' (line 134)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'tuple_var_assignment_390104', subscript_call_result_390225)
    
    # Assigning a Name to a Name (line 134):
    # Getting the type of 'tuple_var_assignment_390103' (line 134)
    tuple_var_assignment_390103_390226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'tuple_var_assignment_390103')
    # Assigning a type to the variable 'row_axis' (line 134)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'row_axis', tuple_var_assignment_390103_390226)
    
    # Assigning a Name to a Name (line 134):
    # Getting the type of 'tuple_var_assignment_390104' (line 134)
    tuple_var_assignment_390104_390227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'tuple_var_assignment_390104')
    # Assigning a type to the variable 'col_axis' (line 134)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 18), 'col_axis', tuple_var_assignment_390104_390227)
    
    
    
    # Evaluating a boolean operation
    
    
    # Getting the type of 'nd' (line 135)
    nd_390228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 17), 'nd')
    # Applying the 'usub' unary operator (line 135)
    result___neg___390229 = python_operator(stypy.reporting.localization.Localization(__file__, 135, 16), 'usub', nd_390228)
    
    # Getting the type of 'row_axis' (line 135)
    row_axis_390230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 23), 'row_axis')
    # Applying the binary operator '<=' (line 135)
    result_le_390231 = python_operator(stypy.reporting.localization.Localization(__file__, 135, 16), '<=', result___neg___390229, row_axis_390230)
    # Getting the type of 'nd' (line 135)
    nd_390232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 34), 'nd')
    # Applying the binary operator '<' (line 135)
    result_lt_390233 = python_operator(stypy.reporting.localization.Localization(__file__, 135, 16), '<', row_axis_390230, nd_390232)
    # Applying the binary operator '&' (line 135)
    result_and__390234 = python_operator(stypy.reporting.localization.Localization(__file__, 135, 16), '&', result_le_390231, result_lt_390233)
    
    
    
    # Getting the type of 'nd' (line 135)
    nd_390235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 42), 'nd')
    # Applying the 'usub' unary operator (line 135)
    result___neg___390236 = python_operator(stypy.reporting.localization.Localization(__file__, 135, 41), 'usub', nd_390235)
    
    # Getting the type of 'col_axis' (line 135)
    col_axis_390237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 48), 'col_axis')
    # Applying the binary operator '<=' (line 135)
    result_le_390238 = python_operator(stypy.reporting.localization.Localization(__file__, 135, 41), '<=', result___neg___390236, col_axis_390237)
    # Getting the type of 'nd' (line 135)
    nd_390239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 59), 'nd')
    # Applying the binary operator '<' (line 135)
    result_lt_390240 = python_operator(stypy.reporting.localization.Localization(__file__, 135, 41), '<', col_axis_390237, nd_390239)
    # Applying the binary operator '&' (line 135)
    result_and__390241 = python_operator(stypy.reporting.localization.Localization(__file__, 135, 41), '&', result_le_390238, result_lt_390240)
    
    # Applying the binary operator 'and' (line 135)
    result_and_keyword_390242 = python_operator(stypy.reporting.localization.Localization(__file__, 135, 16), 'and', result_and__390234, result_and__390241)
    
    # Applying the 'not' unary operator (line 135)
    result_not__390243 = python_operator(stypy.reporting.localization.Localization(__file__, 135, 11), 'not', result_and_keyword_390242)
    
    # Testing the type of an if condition (line 135)
    if_condition_390244 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 135, 8), result_not__390243)
    # Assigning a type to the variable 'if_condition_390244' (line 135)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'if_condition_390244', if_condition_390244)
    # SSA begins for if statement (line 135)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 136)
    # Processing the call arguments (line 136)
    str_390246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 29), 'str', 'Invalid axis %r for an array with shape %r')
    
    # Obtaining an instance of the builtin type 'tuple' (line 137)
    tuple_390247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 30), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 137)
    # Adding element type (line 137)
    # Getting the type of 'axis' (line 137)
    axis_390248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 30), 'axis', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 30), tuple_390247, axis_390248)
    # Adding element type (line 137)
    # Getting the type of 'x' (line 137)
    x_390249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 36), 'x', False)
    # Obtaining the member 'shape' of a type (line 137)
    shape_390250 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 36), x_390249, 'shape')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 30), tuple_390247, shape_390250)
    
    # Applying the binary operator '%' (line 136)
    result_mod_390251 = python_operator(stypy.reporting.localization.Localization(__file__, 136, 29), '%', str_390246, tuple_390247)
    
    # Processing the call keyword arguments (line 136)
    kwargs_390252 = {}
    # Getting the type of 'ValueError' (line 136)
    ValueError_390245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 136)
    ValueError_call_result_390253 = invoke(stypy.reporting.localization.Localization(__file__, 136, 18), ValueError_390245, *[result_mod_390251], **kwargs_390252)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 136, 12), ValueError_call_result_390253, 'raise parameter', BaseException)
    # SSA join for if statement (line 135)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'row_axis' (line 138)
    row_axis_390254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 11), 'row_axis')
    # Getting the type of 'nd' (line 138)
    nd_390255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 22), 'nd')
    # Applying the binary operator '%' (line 138)
    result_mod_390256 = python_operator(stypy.reporting.localization.Localization(__file__, 138, 11), '%', row_axis_390254, nd_390255)
    
    # Getting the type of 'col_axis' (line 138)
    col_axis_390257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 28), 'col_axis')
    # Getting the type of 'nd' (line 138)
    nd_390258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 39), 'nd')
    # Applying the binary operator '%' (line 138)
    result_mod_390259 = python_operator(stypy.reporting.localization.Localization(__file__, 138, 28), '%', col_axis_390257, nd_390258)
    
    # Applying the binary operator '==' (line 138)
    result_eq_390260 = python_operator(stypy.reporting.localization.Localization(__file__, 138, 11), '==', result_mod_390256, result_mod_390259)
    
    # Testing the type of an if condition (line 138)
    if_condition_390261 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 138, 8), result_eq_390260)
    # Assigning a type to the variable 'if_condition_390261' (line 138)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 8), 'if_condition_390261', if_condition_390261)
    # SSA begins for if statement (line 138)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 139)
    # Processing the call arguments (line 139)
    str_390263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 29), 'str', 'Duplicate axes given.')
    # Processing the call keyword arguments (line 139)
    kwargs_390264 = {}
    # Getting the type of 'ValueError' (line 139)
    ValueError_390262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 139)
    ValueError_call_result_390265 = invoke(stypy.reporting.localization.Localization(__file__, 139, 18), ValueError_390262, *[str_390263], **kwargs_390264)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 139, 12), ValueError_call_result_390265, 'raise parameter', BaseException)
    # SSA join for if statement (line 138)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'ord' (line 140)
    ord_390266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 11), 'ord')
    int_390267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 18), 'int')
    # Applying the binary operator '==' (line 140)
    result_eq_390268 = python_operator(stypy.reporting.localization.Localization(__file__, 140, 11), '==', ord_390266, int_390267)
    
    # Testing the type of an if condition (line 140)
    if_condition_390269 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 140, 8), result_eq_390268)
    # Assigning a type to the variable 'if_condition_390269' (line 140)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 8), 'if_condition_390269', if_condition_390269)
    # SSA begins for if statement (line 140)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'NotImplementedError' (line 141)
    NotImplementedError_390270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 18), 'NotImplementedError')
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 141, 12), NotImplementedError_390270, 'raise parameter', BaseException)
    # SSA branch for the else part of an if statement (line 140)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'ord' (line 143)
    ord_390271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 13), 'ord')
    int_390272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 20), 'int')
    # Applying the binary operator '==' (line 143)
    result_eq_390273 = python_operator(stypy.reporting.localization.Localization(__file__, 143, 13), '==', ord_390271, int_390272)
    
    # Testing the type of an if condition (line 143)
    if_condition_390274 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 143, 13), result_eq_390273)
    # Assigning a type to the variable 'if_condition_390274' (line 143)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 13), 'if_condition_390274', if_condition_390274)
    # SSA begins for if statement (line 143)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'NotImplementedError' (line 144)
    NotImplementedError_390275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 18), 'NotImplementedError')
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 144, 12), NotImplementedError_390275, 'raise parameter', BaseException)
    # SSA branch for the else part of an if statement (line 143)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'ord' (line 146)
    ord_390276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 13), 'ord')
    int_390277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 20), 'int')
    # Applying the binary operator '==' (line 146)
    result_eq_390278 = python_operator(stypy.reporting.localization.Localization(__file__, 146, 13), '==', ord_390276, int_390277)
    
    # Testing the type of an if condition (line 146)
    if_condition_390279 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 146, 13), result_eq_390278)
    # Assigning a type to the variable 'if_condition_390279' (line 146)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 13), 'if_condition_390279', if_condition_390279)
    # SSA begins for if statement (line 146)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 147)
    tuple_390280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 64), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 147)
    # Adding element type (line 147)
    int_390281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 64), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 64), tuple_390280, int_390281)
    # Adding element type (line 147)
    int_390282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 66), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 64), tuple_390280, int_390282)
    
    
    # Call to max(...): (line 147)
    # Processing the call keyword arguments (line 147)
    # Getting the type of 'col_axis' (line 147)
    col_axis_390293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 54), 'col_axis', False)
    keyword_390294 = col_axis_390293
    kwargs_390295 = {'axis': keyword_390294}
    
    # Call to sum(...): (line 147)
    # Processing the call keyword arguments (line 147)
    # Getting the type of 'row_axis' (line 147)
    row_axis_390288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 35), 'row_axis', False)
    keyword_390289 = row_axis_390288
    kwargs_390290 = {'axis': keyword_390289}
    
    # Call to abs(...): (line 147)
    # Processing the call arguments (line 147)
    # Getting the type of 'x' (line 147)
    x_390284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 23), 'x', False)
    # Processing the call keyword arguments (line 147)
    kwargs_390285 = {}
    # Getting the type of 'abs' (line 147)
    abs_390283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 19), 'abs', False)
    # Calling abs(args, kwargs) (line 147)
    abs_call_result_390286 = invoke(stypy.reporting.localization.Localization(__file__, 147, 19), abs_390283, *[x_390284], **kwargs_390285)
    
    # Obtaining the member 'sum' of a type (line 147)
    sum_390287 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 19), abs_call_result_390286, 'sum')
    # Calling sum(args, kwargs) (line 147)
    sum_call_result_390291 = invoke(stypy.reporting.localization.Localization(__file__, 147, 19), sum_390287, *[], **kwargs_390290)
    
    # Obtaining the member 'max' of a type (line 147)
    max_390292 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 19), sum_call_result_390291, 'max')
    # Calling max(args, kwargs) (line 147)
    max_call_result_390296 = invoke(stypy.reporting.localization.Localization(__file__, 147, 19), max_390292, *[], **kwargs_390295)
    
    # Obtaining the member '__getitem__' of a type (line 147)
    getitem___390297 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 19), max_call_result_390296, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 147)
    subscript_call_result_390298 = invoke(stypy.reporting.localization.Localization(__file__, 147, 19), getitem___390297, tuple_390280)
    
    # Assigning a type to the variable 'stypy_return_type' (line 147)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 12), 'stypy_return_type', subscript_call_result_390298)
    # SSA branch for the else part of an if statement (line 146)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'ord' (line 148)
    ord_390299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 13), 'ord')
    # Getting the type of 'Inf' (line 148)
    Inf_390300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 20), 'Inf')
    # Applying the binary operator '==' (line 148)
    result_eq_390301 = python_operator(stypy.reporting.localization.Localization(__file__, 148, 13), '==', ord_390299, Inf_390300)
    
    # Testing the type of an if condition (line 148)
    if_condition_390302 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 148, 13), result_eq_390301)
    # Assigning a type to the variable 'if_condition_390302' (line 148)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 13), 'if_condition_390302', if_condition_390302)
    # SSA begins for if statement (line 148)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 149)
    tuple_390303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 64), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 149)
    # Adding element type (line 149)
    int_390304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 64), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 149, 64), tuple_390303, int_390304)
    # Adding element type (line 149)
    int_390305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 66), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 149, 64), tuple_390303, int_390305)
    
    
    # Call to max(...): (line 149)
    # Processing the call keyword arguments (line 149)
    # Getting the type of 'row_axis' (line 149)
    row_axis_390316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 54), 'row_axis', False)
    keyword_390317 = row_axis_390316
    kwargs_390318 = {'axis': keyword_390317}
    
    # Call to sum(...): (line 149)
    # Processing the call keyword arguments (line 149)
    # Getting the type of 'col_axis' (line 149)
    col_axis_390311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 35), 'col_axis', False)
    keyword_390312 = col_axis_390311
    kwargs_390313 = {'axis': keyword_390312}
    
    # Call to abs(...): (line 149)
    # Processing the call arguments (line 149)
    # Getting the type of 'x' (line 149)
    x_390307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 23), 'x', False)
    # Processing the call keyword arguments (line 149)
    kwargs_390308 = {}
    # Getting the type of 'abs' (line 149)
    abs_390306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 19), 'abs', False)
    # Calling abs(args, kwargs) (line 149)
    abs_call_result_390309 = invoke(stypy.reporting.localization.Localization(__file__, 149, 19), abs_390306, *[x_390307], **kwargs_390308)
    
    # Obtaining the member 'sum' of a type (line 149)
    sum_390310 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 19), abs_call_result_390309, 'sum')
    # Calling sum(args, kwargs) (line 149)
    sum_call_result_390314 = invoke(stypy.reporting.localization.Localization(__file__, 149, 19), sum_390310, *[], **kwargs_390313)
    
    # Obtaining the member 'max' of a type (line 149)
    max_390315 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 19), sum_call_result_390314, 'max')
    # Calling max(args, kwargs) (line 149)
    max_call_result_390319 = invoke(stypy.reporting.localization.Localization(__file__, 149, 19), max_390315, *[], **kwargs_390318)
    
    # Obtaining the member '__getitem__' of a type (line 149)
    getitem___390320 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 19), max_call_result_390319, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 149)
    subscript_call_result_390321 = invoke(stypy.reporting.localization.Localization(__file__, 149, 19), getitem___390320, tuple_390303)
    
    # Assigning a type to the variable 'stypy_return_type' (line 149)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 12), 'stypy_return_type', subscript_call_result_390321)
    # SSA branch for the else part of an if statement (line 148)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'ord' (line 150)
    ord_390322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 13), 'ord')
    int_390323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 20), 'int')
    # Applying the binary operator '==' (line 150)
    result_eq_390324 = python_operator(stypy.reporting.localization.Localization(__file__, 150, 13), '==', ord_390322, int_390323)
    
    # Testing the type of an if condition (line 150)
    if_condition_390325 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 150, 13), result_eq_390324)
    # Assigning a type to the variable 'if_condition_390325' (line 150)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 13), 'if_condition_390325', if_condition_390325)
    # SSA begins for if statement (line 150)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 151)
    tuple_390326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 64), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 151)
    # Adding element type (line 151)
    int_390327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 64), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 151, 64), tuple_390326, int_390327)
    # Adding element type (line 151)
    int_390328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 66), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 151, 64), tuple_390326, int_390328)
    
    
    # Call to min(...): (line 151)
    # Processing the call keyword arguments (line 151)
    # Getting the type of 'col_axis' (line 151)
    col_axis_390339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 54), 'col_axis', False)
    keyword_390340 = col_axis_390339
    kwargs_390341 = {'axis': keyword_390340}
    
    # Call to sum(...): (line 151)
    # Processing the call keyword arguments (line 151)
    # Getting the type of 'row_axis' (line 151)
    row_axis_390334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 35), 'row_axis', False)
    keyword_390335 = row_axis_390334
    kwargs_390336 = {'axis': keyword_390335}
    
    # Call to abs(...): (line 151)
    # Processing the call arguments (line 151)
    # Getting the type of 'x' (line 151)
    x_390330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 23), 'x', False)
    # Processing the call keyword arguments (line 151)
    kwargs_390331 = {}
    # Getting the type of 'abs' (line 151)
    abs_390329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 19), 'abs', False)
    # Calling abs(args, kwargs) (line 151)
    abs_call_result_390332 = invoke(stypy.reporting.localization.Localization(__file__, 151, 19), abs_390329, *[x_390330], **kwargs_390331)
    
    # Obtaining the member 'sum' of a type (line 151)
    sum_390333 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 19), abs_call_result_390332, 'sum')
    # Calling sum(args, kwargs) (line 151)
    sum_call_result_390337 = invoke(stypy.reporting.localization.Localization(__file__, 151, 19), sum_390333, *[], **kwargs_390336)
    
    # Obtaining the member 'min' of a type (line 151)
    min_390338 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 19), sum_call_result_390337, 'min')
    # Calling min(args, kwargs) (line 151)
    min_call_result_390342 = invoke(stypy.reporting.localization.Localization(__file__, 151, 19), min_390338, *[], **kwargs_390341)
    
    # Obtaining the member '__getitem__' of a type (line 151)
    getitem___390343 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 19), min_call_result_390342, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 151)
    subscript_call_result_390344 = invoke(stypy.reporting.localization.Localization(__file__, 151, 19), getitem___390343, tuple_390326)
    
    # Assigning a type to the variable 'stypy_return_type' (line 151)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 12), 'stypy_return_type', subscript_call_result_390344)
    # SSA branch for the else part of an if statement (line 150)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'ord' (line 152)
    ord_390345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 13), 'ord')
    
    # Getting the type of 'Inf' (line 152)
    Inf_390346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 21), 'Inf')
    # Applying the 'usub' unary operator (line 152)
    result___neg___390347 = python_operator(stypy.reporting.localization.Localization(__file__, 152, 20), 'usub', Inf_390346)
    
    # Applying the binary operator '==' (line 152)
    result_eq_390348 = python_operator(stypy.reporting.localization.Localization(__file__, 152, 13), '==', ord_390345, result___neg___390347)
    
    # Testing the type of an if condition (line 152)
    if_condition_390349 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 152, 13), result_eq_390348)
    # Assigning a type to the variable 'if_condition_390349' (line 152)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 13), 'if_condition_390349', if_condition_390349)
    # SSA begins for if statement (line 152)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 153)
    tuple_390350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 64), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 153)
    # Adding element type (line 153)
    int_390351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 64), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 153, 64), tuple_390350, int_390351)
    # Adding element type (line 153)
    int_390352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 66), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 153, 64), tuple_390350, int_390352)
    
    
    # Call to min(...): (line 153)
    # Processing the call keyword arguments (line 153)
    # Getting the type of 'row_axis' (line 153)
    row_axis_390363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 54), 'row_axis', False)
    keyword_390364 = row_axis_390363
    kwargs_390365 = {'axis': keyword_390364}
    
    # Call to sum(...): (line 153)
    # Processing the call keyword arguments (line 153)
    # Getting the type of 'col_axis' (line 153)
    col_axis_390358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 35), 'col_axis', False)
    keyword_390359 = col_axis_390358
    kwargs_390360 = {'axis': keyword_390359}
    
    # Call to abs(...): (line 153)
    # Processing the call arguments (line 153)
    # Getting the type of 'x' (line 153)
    x_390354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 23), 'x', False)
    # Processing the call keyword arguments (line 153)
    kwargs_390355 = {}
    # Getting the type of 'abs' (line 153)
    abs_390353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 19), 'abs', False)
    # Calling abs(args, kwargs) (line 153)
    abs_call_result_390356 = invoke(stypy.reporting.localization.Localization(__file__, 153, 19), abs_390353, *[x_390354], **kwargs_390355)
    
    # Obtaining the member 'sum' of a type (line 153)
    sum_390357 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 19), abs_call_result_390356, 'sum')
    # Calling sum(args, kwargs) (line 153)
    sum_call_result_390361 = invoke(stypy.reporting.localization.Localization(__file__, 153, 19), sum_390357, *[], **kwargs_390360)
    
    # Obtaining the member 'min' of a type (line 153)
    min_390362 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 19), sum_call_result_390361, 'min')
    # Calling min(args, kwargs) (line 153)
    min_call_result_390366 = invoke(stypy.reporting.localization.Localization(__file__, 153, 19), min_390362, *[], **kwargs_390365)
    
    # Obtaining the member '__getitem__' of a type (line 153)
    getitem___390367 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 19), min_call_result_390366, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 153)
    subscript_call_result_390368 = invoke(stypy.reporting.localization.Localization(__file__, 153, 19), getitem___390367, tuple_390350)
    
    # Assigning a type to the variable 'stypy_return_type' (line 153)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 12), 'stypy_return_type', subscript_call_result_390368)
    # SSA branch for the else part of an if statement (line 152)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'ord' (line 154)
    ord_390369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 13), 'ord')
    
    # Obtaining an instance of the builtin type 'tuple' (line 154)
    tuple_390370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 21), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 154)
    # Adding element type (line 154)
    # Getting the type of 'None' (line 154)
    None_390371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 21), 'None')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 154, 21), tuple_390370, None_390371)
    # Adding element type (line 154)
    str_390372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 27), 'str', 'f')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 154, 21), tuple_390370, str_390372)
    # Adding element type (line 154)
    str_390373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 32), 'str', 'fro')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 154, 21), tuple_390370, str_390373)
    
    # Applying the binary operator 'in' (line 154)
    result_contains_390374 = python_operator(stypy.reporting.localization.Localization(__file__, 154, 13), 'in', ord_390369, tuple_390370)
    
    # Testing the type of an if condition (line 154)
    if_condition_390375 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 154, 13), result_contains_390374)
    # Assigning a type to the variable 'if_condition_390375' (line 154)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 13), 'if_condition_390375', if_condition_390375)
    # SSA begins for if statement (line 154)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to _sparse_frobenius_norm(...): (line 156)
    # Processing the call arguments (line 156)
    # Getting the type of 'x' (line 156)
    x_390377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 42), 'x', False)
    # Processing the call keyword arguments (line 156)
    kwargs_390378 = {}
    # Getting the type of '_sparse_frobenius_norm' (line 156)
    _sparse_frobenius_norm_390376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 19), '_sparse_frobenius_norm', False)
    # Calling _sparse_frobenius_norm(args, kwargs) (line 156)
    _sparse_frobenius_norm_call_result_390379 = invoke(stypy.reporting.localization.Localization(__file__, 156, 19), _sparse_frobenius_norm_390376, *[x_390377], **kwargs_390378)
    
    # Assigning a type to the variable 'stypy_return_type' (line 156)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 12), 'stypy_return_type', _sparse_frobenius_norm_call_result_390379)
    # SSA branch for the else part of an if statement (line 154)
    module_type_store.open_ssa_branch('else')
    
    # Call to ValueError(...): (line 158)
    # Processing the call arguments (line 158)
    str_390381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 29), 'str', 'Invalid norm order for matrices.')
    # Processing the call keyword arguments (line 158)
    kwargs_390382 = {}
    # Getting the type of 'ValueError' (line 158)
    ValueError_390380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 158)
    ValueError_call_result_390383 = invoke(stypy.reporting.localization.Localization(__file__, 158, 18), ValueError_390380, *[str_390381], **kwargs_390382)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 158, 12), ValueError_call_result_390383, 'raise parameter', BaseException)
    # SSA join for if statement (line 154)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 152)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 150)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 148)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 146)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 143)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 140)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 133)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Call to len(...): (line 159)
    # Processing the call arguments (line 159)
    # Getting the type of 'axis' (line 159)
    axis_390385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 13), 'axis', False)
    # Processing the call keyword arguments (line 159)
    kwargs_390386 = {}
    # Getting the type of 'len' (line 159)
    len_390384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 9), 'len', False)
    # Calling len(args, kwargs) (line 159)
    len_call_result_390387 = invoke(stypy.reporting.localization.Localization(__file__, 159, 9), len_390384, *[axis_390385], **kwargs_390386)
    
    int_390388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 22), 'int')
    # Applying the binary operator '==' (line 159)
    result_eq_390389 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 9), '==', len_call_result_390387, int_390388)
    
    # Testing the type of an if condition (line 159)
    if_condition_390390 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 159, 9), result_eq_390389)
    # Assigning a type to the variable 'if_condition_390390' (line 159)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 9), 'if_condition_390390', if_condition_390390)
    # SSA begins for if statement (line 159)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Tuple (line 160):
    
    # Assigning a Subscript to a Name (line 160):
    
    # Obtaining the type of the subscript
    int_390391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 8), 'int')
    # Getting the type of 'axis' (line 160)
    axis_390392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 13), 'axis')
    # Obtaining the member '__getitem__' of a type (line 160)
    getitem___390393 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 8), axis_390392, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 160)
    subscript_call_result_390394 = invoke(stypy.reporting.localization.Localization(__file__, 160, 8), getitem___390393, int_390391)
    
    # Assigning a type to the variable 'tuple_var_assignment_390105' (line 160)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'tuple_var_assignment_390105', subscript_call_result_390394)
    
    # Assigning a Name to a Name (line 160):
    # Getting the type of 'tuple_var_assignment_390105' (line 160)
    tuple_var_assignment_390105_390395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'tuple_var_assignment_390105')
    # Assigning a type to the variable 'a' (line 160)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'a', tuple_var_assignment_390105_390395)
    
    
    
    
    # Getting the type of 'nd' (line 161)
    nd_390396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 17), 'nd')
    # Applying the 'usub' unary operator (line 161)
    result___neg___390397 = python_operator(stypy.reporting.localization.Localization(__file__, 161, 16), 'usub', nd_390396)
    
    # Getting the type of 'a' (line 161)
    a_390398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 23), 'a')
    # Applying the binary operator '<=' (line 161)
    result_le_390399 = python_operator(stypy.reporting.localization.Localization(__file__, 161, 16), '<=', result___neg___390397, a_390398)
    # Getting the type of 'nd' (line 161)
    nd_390400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 27), 'nd')
    # Applying the binary operator '<' (line 161)
    result_lt_390401 = python_operator(stypy.reporting.localization.Localization(__file__, 161, 16), '<', a_390398, nd_390400)
    # Applying the binary operator '&' (line 161)
    result_and__390402 = python_operator(stypy.reporting.localization.Localization(__file__, 161, 16), '&', result_le_390399, result_lt_390401)
    
    # Applying the 'not' unary operator (line 161)
    result_not__390403 = python_operator(stypy.reporting.localization.Localization(__file__, 161, 11), 'not', result_and__390402)
    
    # Testing the type of an if condition (line 161)
    if_condition_390404 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 161, 8), result_not__390403)
    # Assigning a type to the variable 'if_condition_390404' (line 161)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 8), 'if_condition_390404', if_condition_390404)
    # SSA begins for if statement (line 161)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 162)
    # Processing the call arguments (line 162)
    str_390406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 29), 'str', 'Invalid axis %r for an array with shape %r')
    
    # Obtaining an instance of the builtin type 'tuple' (line 163)
    tuple_390407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 30), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 163)
    # Adding element type (line 163)
    # Getting the type of 'axis' (line 163)
    axis_390408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 30), 'axis', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 163, 30), tuple_390407, axis_390408)
    # Adding element type (line 163)
    # Getting the type of 'x' (line 163)
    x_390409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 36), 'x', False)
    # Obtaining the member 'shape' of a type (line 163)
    shape_390410 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 36), x_390409, 'shape')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 163, 30), tuple_390407, shape_390410)
    
    # Applying the binary operator '%' (line 162)
    result_mod_390411 = python_operator(stypy.reporting.localization.Localization(__file__, 162, 29), '%', str_390406, tuple_390407)
    
    # Processing the call keyword arguments (line 162)
    kwargs_390412 = {}
    # Getting the type of 'ValueError' (line 162)
    ValueError_390405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 162)
    ValueError_call_result_390413 = invoke(stypy.reporting.localization.Localization(__file__, 162, 18), ValueError_390405, *[result_mod_390411], **kwargs_390412)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 162, 12), ValueError_call_result_390413, 'raise parameter', BaseException)
    # SSA join for if statement (line 161)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'ord' (line 164)
    ord_390414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 11), 'ord')
    # Getting the type of 'Inf' (line 164)
    Inf_390415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 18), 'Inf')
    # Applying the binary operator '==' (line 164)
    result_eq_390416 = python_operator(stypy.reporting.localization.Localization(__file__, 164, 11), '==', ord_390414, Inf_390415)
    
    # Testing the type of an if condition (line 164)
    if_condition_390417 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 164, 8), result_eq_390416)
    # Assigning a type to the variable 'if_condition_390417' (line 164)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 8), 'if_condition_390417', if_condition_390417)
    # SSA begins for if statement (line 164)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 165):
    
    # Assigning a Call to a Name (line 165):
    
    # Call to max(...): (line 165)
    # Processing the call keyword arguments (line 165)
    # Getting the type of 'a' (line 165)
    a_390423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 32), 'a', False)
    keyword_390424 = a_390423
    kwargs_390425 = {'axis': keyword_390424}
    
    # Call to abs(...): (line 165)
    # Processing the call arguments (line 165)
    # Getting the type of 'x' (line 165)
    x_390419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 20), 'x', False)
    # Processing the call keyword arguments (line 165)
    kwargs_390420 = {}
    # Getting the type of 'abs' (line 165)
    abs_390418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 16), 'abs', False)
    # Calling abs(args, kwargs) (line 165)
    abs_call_result_390421 = invoke(stypy.reporting.localization.Localization(__file__, 165, 16), abs_390418, *[x_390419], **kwargs_390420)
    
    # Obtaining the member 'max' of a type (line 165)
    max_390422 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 16), abs_call_result_390421, 'max')
    # Calling max(args, kwargs) (line 165)
    max_call_result_390426 = invoke(stypy.reporting.localization.Localization(__file__, 165, 16), max_390422, *[], **kwargs_390425)
    
    # Assigning a type to the variable 'M' (line 165)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 12), 'M', max_call_result_390426)
    # SSA branch for the else part of an if statement (line 164)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'ord' (line 166)
    ord_390427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 13), 'ord')
    
    # Getting the type of 'Inf' (line 166)
    Inf_390428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 21), 'Inf')
    # Applying the 'usub' unary operator (line 166)
    result___neg___390429 = python_operator(stypy.reporting.localization.Localization(__file__, 166, 20), 'usub', Inf_390428)
    
    # Applying the binary operator '==' (line 166)
    result_eq_390430 = python_operator(stypy.reporting.localization.Localization(__file__, 166, 13), '==', ord_390427, result___neg___390429)
    
    # Testing the type of an if condition (line 166)
    if_condition_390431 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 166, 13), result_eq_390430)
    # Assigning a type to the variable 'if_condition_390431' (line 166)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 13), 'if_condition_390431', if_condition_390431)
    # SSA begins for if statement (line 166)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 167):
    
    # Assigning a Call to a Name (line 167):
    
    # Call to min(...): (line 167)
    # Processing the call keyword arguments (line 167)
    # Getting the type of 'a' (line 167)
    a_390437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 32), 'a', False)
    keyword_390438 = a_390437
    kwargs_390439 = {'axis': keyword_390438}
    
    # Call to abs(...): (line 167)
    # Processing the call arguments (line 167)
    # Getting the type of 'x' (line 167)
    x_390433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 20), 'x', False)
    # Processing the call keyword arguments (line 167)
    kwargs_390434 = {}
    # Getting the type of 'abs' (line 167)
    abs_390432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 16), 'abs', False)
    # Calling abs(args, kwargs) (line 167)
    abs_call_result_390435 = invoke(stypy.reporting.localization.Localization(__file__, 167, 16), abs_390432, *[x_390433], **kwargs_390434)
    
    # Obtaining the member 'min' of a type (line 167)
    min_390436 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 16), abs_call_result_390435, 'min')
    # Calling min(args, kwargs) (line 167)
    min_call_result_390440 = invoke(stypy.reporting.localization.Localization(__file__, 167, 16), min_390436, *[], **kwargs_390439)
    
    # Assigning a type to the variable 'M' (line 167)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 12), 'M', min_call_result_390440)
    # SSA branch for the else part of an if statement (line 166)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'ord' (line 168)
    ord_390441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 13), 'ord')
    int_390442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 20), 'int')
    # Applying the binary operator '==' (line 168)
    result_eq_390443 = python_operator(stypy.reporting.localization.Localization(__file__, 168, 13), '==', ord_390441, int_390442)
    
    # Testing the type of an if condition (line 168)
    if_condition_390444 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 168, 13), result_eq_390443)
    # Assigning a type to the variable 'if_condition_390444' (line 168)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 13), 'if_condition_390444', if_condition_390444)
    # SSA begins for if statement (line 168)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 170):
    
    # Assigning a Call to a Name (line 170):
    
    # Call to sum(...): (line 170)
    # Processing the call keyword arguments (line 170)
    # Getting the type of 'a' (line 170)
    a_390449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 34), 'a', False)
    keyword_390450 = a_390449
    kwargs_390451 = {'axis': keyword_390450}
    
    # Getting the type of 'x' (line 170)
    x_390445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 17), 'x', False)
    int_390446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 22), 'int')
    # Applying the binary operator '!=' (line 170)
    result_ne_390447 = python_operator(stypy.reporting.localization.Localization(__file__, 170, 17), '!=', x_390445, int_390446)
    
    # Obtaining the member 'sum' of a type (line 170)
    sum_390448 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 17), result_ne_390447, 'sum')
    # Calling sum(args, kwargs) (line 170)
    sum_call_result_390452 = invoke(stypy.reporting.localization.Localization(__file__, 170, 17), sum_390448, *[], **kwargs_390451)
    
    # Assigning a type to the variable 'M' (line 170)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 12), 'M', sum_call_result_390452)
    # SSA branch for the else part of an if statement (line 168)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'ord' (line 171)
    ord_390453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 13), 'ord')
    int_390454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 20), 'int')
    # Applying the binary operator '==' (line 171)
    result_eq_390455 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 13), '==', ord_390453, int_390454)
    
    # Testing the type of an if condition (line 171)
    if_condition_390456 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 171, 13), result_eq_390455)
    # Assigning a type to the variable 'if_condition_390456' (line 171)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 13), 'if_condition_390456', if_condition_390456)
    # SSA begins for if statement (line 171)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 173):
    
    # Assigning a Call to a Name (line 173):
    
    # Call to sum(...): (line 173)
    # Processing the call keyword arguments (line 173)
    # Getting the type of 'a' (line 173)
    a_390462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 32), 'a', False)
    keyword_390463 = a_390462
    kwargs_390464 = {'axis': keyword_390463}
    
    # Call to abs(...): (line 173)
    # Processing the call arguments (line 173)
    # Getting the type of 'x' (line 173)
    x_390458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 20), 'x', False)
    # Processing the call keyword arguments (line 173)
    kwargs_390459 = {}
    # Getting the type of 'abs' (line 173)
    abs_390457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 16), 'abs', False)
    # Calling abs(args, kwargs) (line 173)
    abs_call_result_390460 = invoke(stypy.reporting.localization.Localization(__file__, 173, 16), abs_390457, *[x_390458], **kwargs_390459)
    
    # Obtaining the member 'sum' of a type (line 173)
    sum_390461 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 16), abs_call_result_390460, 'sum')
    # Calling sum(args, kwargs) (line 173)
    sum_call_result_390465 = invoke(stypy.reporting.localization.Localization(__file__, 173, 16), sum_390461, *[], **kwargs_390464)
    
    # Assigning a type to the variable 'M' (line 173)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 12), 'M', sum_call_result_390465)
    # SSA branch for the else part of an if statement (line 171)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'ord' (line 174)
    ord_390466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 13), 'ord')
    
    # Obtaining an instance of the builtin type 'tuple' (line 174)
    tuple_390467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 21), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 174)
    # Adding element type (line 174)
    int_390468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 174, 21), tuple_390467, int_390468)
    # Adding element type (line 174)
    # Getting the type of 'None' (line 174)
    None_390469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 24), 'None')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 174, 21), tuple_390467, None_390469)
    
    # Applying the binary operator 'in' (line 174)
    result_contains_390470 = python_operator(stypy.reporting.localization.Localization(__file__, 174, 13), 'in', ord_390466, tuple_390467)
    
    # Testing the type of an if condition (line 174)
    if_condition_390471 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 174, 13), result_contains_390470)
    # Assigning a type to the variable 'if_condition_390471' (line 174)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 13), 'if_condition_390471', if_condition_390471)
    # SSA begins for if statement (line 174)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 175):
    
    # Assigning a Call to a Name (line 175):
    
    # Call to sqrt(...): (line 175)
    # Processing the call arguments (line 175)
    
    # Call to sum(...): (line 175)
    # Processing the call keyword arguments (line 175)
    # Getting the type of 'a' (line 175)
    a_390482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 46), 'a', False)
    keyword_390483 = a_390482
    kwargs_390484 = {'axis': keyword_390483}
    
    # Call to power(...): (line 175)
    # Processing the call arguments (line 175)
    int_390478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 34), 'int')
    # Processing the call keyword arguments (line 175)
    kwargs_390479 = {}
    
    # Call to abs(...): (line 175)
    # Processing the call arguments (line 175)
    # Getting the type of 'x' (line 175)
    x_390474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 25), 'x', False)
    # Processing the call keyword arguments (line 175)
    kwargs_390475 = {}
    # Getting the type of 'abs' (line 175)
    abs_390473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 21), 'abs', False)
    # Calling abs(args, kwargs) (line 175)
    abs_call_result_390476 = invoke(stypy.reporting.localization.Localization(__file__, 175, 21), abs_390473, *[x_390474], **kwargs_390475)
    
    # Obtaining the member 'power' of a type (line 175)
    power_390477 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 21), abs_call_result_390476, 'power')
    # Calling power(args, kwargs) (line 175)
    power_call_result_390480 = invoke(stypy.reporting.localization.Localization(__file__, 175, 21), power_390477, *[int_390478], **kwargs_390479)
    
    # Obtaining the member 'sum' of a type (line 175)
    sum_390481 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 21), power_call_result_390480, 'sum')
    # Calling sum(args, kwargs) (line 175)
    sum_call_result_390485 = invoke(stypy.reporting.localization.Localization(__file__, 175, 21), sum_390481, *[], **kwargs_390484)
    
    # Processing the call keyword arguments (line 175)
    kwargs_390486 = {}
    # Getting the type of 'sqrt' (line 175)
    sqrt_390472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 16), 'sqrt', False)
    # Calling sqrt(args, kwargs) (line 175)
    sqrt_call_result_390487 = invoke(stypy.reporting.localization.Localization(__file__, 175, 16), sqrt_390472, *[sum_call_result_390485], **kwargs_390486)
    
    # Assigning a type to the variable 'M' (line 175)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 12), 'M', sqrt_call_result_390487)
    # SSA branch for the else part of an if statement (line 174)
    module_type_store.open_ssa_branch('else')
    
    
    # SSA begins for try-except statement (line 177)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    # Getting the type of 'ord' (line 178)
    ord_390488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 16), 'ord')
    int_390489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 22), 'int')
    # Applying the binary operator '+' (line 178)
    result_add_390490 = python_operator(stypy.reporting.localization.Localization(__file__, 178, 16), '+', ord_390488, int_390489)
    
    # SSA branch for the except part of a try statement (line 177)
    # SSA branch for the except 'TypeError' branch of a try statement (line 177)
    module_type_store.open_ssa_branch('except')
    
    # Call to ValueError(...): (line 180)
    # Processing the call arguments (line 180)
    str_390492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 33), 'str', 'Invalid norm order for vectors.')
    # Processing the call keyword arguments (line 180)
    kwargs_390493 = {}
    # Getting the type of 'ValueError' (line 180)
    ValueError_390491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 22), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 180)
    ValueError_call_result_390494 = invoke(stypy.reporting.localization.Localization(__file__, 180, 22), ValueError_390491, *[str_390492], **kwargs_390493)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 180, 16), ValueError_call_result_390494, 'raise parameter', BaseException)
    # SSA join for try-except statement (line 177)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 181):
    
    # Assigning a Call to a Name (line 181):
    
    # Call to power(...): (line 181)
    # Processing the call arguments (line 181)
    
    # Call to sum(...): (line 181)
    # Processing the call keyword arguments (line 181)
    # Getting the type of 'a' (line 181)
    a_390506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 52), 'a', False)
    keyword_390507 = a_390506
    kwargs_390508 = {'axis': keyword_390507}
    
    # Call to power(...): (line 181)
    # Processing the call arguments (line 181)
    # Getting the type of 'ord' (line 181)
    ord_390502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 38), 'ord', False)
    # Processing the call keyword arguments (line 181)
    kwargs_390503 = {}
    
    # Call to abs(...): (line 181)
    # Processing the call arguments (line 181)
    # Getting the type of 'x' (line 181)
    x_390498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 29), 'x', False)
    # Processing the call keyword arguments (line 181)
    kwargs_390499 = {}
    # Getting the type of 'abs' (line 181)
    abs_390497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 25), 'abs', False)
    # Calling abs(args, kwargs) (line 181)
    abs_call_result_390500 = invoke(stypy.reporting.localization.Localization(__file__, 181, 25), abs_390497, *[x_390498], **kwargs_390499)
    
    # Obtaining the member 'power' of a type (line 181)
    power_390501 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 25), abs_call_result_390500, 'power')
    # Calling power(args, kwargs) (line 181)
    power_call_result_390504 = invoke(stypy.reporting.localization.Localization(__file__, 181, 25), power_390501, *[ord_390502], **kwargs_390503)
    
    # Obtaining the member 'sum' of a type (line 181)
    sum_390505 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 25), power_call_result_390504, 'sum')
    # Calling sum(args, kwargs) (line 181)
    sum_call_result_390509 = invoke(stypy.reporting.localization.Localization(__file__, 181, 25), sum_390505, *[], **kwargs_390508)
    
    int_390510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 56), 'int')
    # Getting the type of 'ord' (line 181)
    ord_390511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 60), 'ord', False)
    # Applying the binary operator 'div' (line 181)
    result_div_390512 = python_operator(stypy.reporting.localization.Localization(__file__, 181, 56), 'div', int_390510, ord_390511)
    
    # Processing the call keyword arguments (line 181)
    kwargs_390513 = {}
    # Getting the type of 'np' (line 181)
    np_390495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 16), 'np', False)
    # Obtaining the member 'power' of a type (line 181)
    power_390496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 16), np_390495, 'power')
    # Calling power(args, kwargs) (line 181)
    power_call_result_390514 = invoke(stypy.reporting.localization.Localization(__file__, 181, 16), power_390496, *[sum_call_result_390509, result_div_390512], **kwargs_390513)
    
    # Assigning a type to the variable 'M' (line 181)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 12), 'M', power_call_result_390514)
    # SSA join for if statement (line 174)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 171)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 168)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 166)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 164)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to ravel(...): (line 182)
    # Processing the call keyword arguments (line 182)
    kwargs_390518 = {}
    # Getting the type of 'M' (line 182)
    M_390515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 15), 'M', False)
    # Obtaining the member 'A' of a type (line 182)
    A_390516 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 15), M_390515, 'A')
    # Obtaining the member 'ravel' of a type (line 182)
    ravel_390517 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 15), A_390516, 'ravel')
    # Calling ravel(args, kwargs) (line 182)
    ravel_call_result_390519 = invoke(stypy.reporting.localization.Localization(__file__, 182, 15), ravel_390517, *[], **kwargs_390518)
    
    # Assigning a type to the variable 'stypy_return_type' (line 182)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'stypy_return_type', ravel_call_result_390519)
    # SSA branch for the else part of an if statement (line 159)
    module_type_store.open_ssa_branch('else')
    
    # Call to ValueError(...): (line 184)
    # Processing the call arguments (line 184)
    str_390521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 25), 'str', 'Improper number of dimensions to norm.')
    # Processing the call keyword arguments (line 184)
    kwargs_390522 = {}
    # Getting the type of 'ValueError' (line 184)
    ValueError_390520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 184)
    ValueError_call_result_390523 = invoke(stypy.reporting.localization.Localization(__file__, 184, 14), ValueError_390520, *[str_390521], **kwargs_390522)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 184, 8), ValueError_call_result_390523, 'raise parameter', BaseException)
    # SSA join for if statement (line 159)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 133)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'norm(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'norm' in the type store
    # Getting the type of 'stypy_return_type' (line 22)
    stypy_return_type_390524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_390524)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'norm'
    return stypy_return_type_390524

# Assigning a type to the variable 'norm' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'norm', norm)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
