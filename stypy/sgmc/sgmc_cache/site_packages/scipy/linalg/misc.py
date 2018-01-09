
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: import numpy as np
4: from numpy.linalg import LinAlgError
5: from .blas import get_blas_funcs
6: from .lapack import get_lapack_funcs
7: 
8: __all__ = ['LinAlgError', 'norm']
9: 
10: 
11: def norm(a, ord=None, axis=None, keepdims=False):
12:     '''
13:     Matrix or vector norm.
14: 
15:     This function is able to return one of seven different matrix norms,
16:     or one of an infinite number of vector norms (described below), depending
17:     on the value of the ``ord`` parameter.
18: 
19:     Parameters
20:     ----------
21:     a : (M,) or (M, N) array_like
22:         Input array.  If `axis` is None, `a` must be 1-D or 2-D.
23:     ord : {non-zero int, inf, -inf, 'fro'}, optional
24:         Order of the norm (see table under ``Notes``). inf means numpy's
25:         `inf` object
26:     axis : {int, 2-tuple of ints, None}, optional
27:         If `axis` is an integer, it specifies the axis of `a` along which to
28:         compute the vector norms.  If `axis` is a 2-tuple, it specifies the
29:         axes that hold 2-D matrices, and the matrix norms of these matrices
30:         are computed.  If `axis` is None then either a vector norm (when `a`
31:         is 1-D) or a matrix norm (when `a` is 2-D) is returned.
32:     keepdims : bool, optional
33:         If this is set to True, the axes which are normed over are left in the
34:         result as dimensions with size one.  With this option the result will
35:         broadcast correctly against the original `a`.
36: 
37:     Returns
38:     -------
39:     n : float or ndarray
40:         Norm of the matrix or vector(s).
41: 
42:     Notes
43:     -----
44:     For values of ``ord <= 0``, the result is, strictly speaking, not a
45:     mathematical 'norm', but it may still be useful for various numerical
46:     purposes.
47: 
48:     The following norms can be calculated:
49: 
50:     =====  ============================  ==========================
51:     ord    norm for matrices             norm for vectors
52:     =====  ============================  ==========================
53:     None   Frobenius norm                2-norm
54:     'fro'  Frobenius norm                --
55:     inf    max(sum(abs(x), axis=1))      max(abs(x))
56:     -inf   min(sum(abs(x), axis=1))      min(abs(x))
57:     0      --                            sum(x != 0)
58:     1      max(sum(abs(x), axis=0))      as below
59:     -1     min(sum(abs(x), axis=0))      as below
60:     2      2-norm (largest sing. value)  as below
61:     -2     smallest singular value       as below
62:     other  --                            sum(abs(x)**ord)**(1./ord)
63:     =====  ============================  ==========================
64: 
65:     The Frobenius norm is given by [1]_:
66: 
67:         :math:`||A||_F = [\\sum_{i,j} abs(a_{i,j})^2]^{1/2}`
68: 
69:     The ``axis`` and ``keepdims`` arguments are passed directly to
70:     ``numpy.linalg.norm`` and are only usable if they are supported
71:     by the version of numpy in use.
72: 
73:     References
74:     ----------
75:     .. [1] G. H. Golub and C. F. Van Loan, *Matrix Computations*,
76:            Baltimore, MD, Johns Hopkins University Press, 1985, pg. 15
77: 
78:     Examples
79:     --------
80:     >>> from scipy.linalg import norm
81:     >>> a = np.arange(9) - 4.0
82:     >>> a
83:     array([-4., -3., -2., -1.,  0.,  1.,  2.,  3.,  4.])
84:     >>> b = a.reshape((3, 3))
85:     >>> b
86:     array([[-4., -3., -2.],
87:            [-1.,  0.,  1.],
88:            [ 2.,  3.,  4.]])
89: 
90:     >>> norm(a)
91:     7.745966692414834
92:     >>> norm(b)
93:     7.745966692414834
94:     >>> norm(b, 'fro')
95:     7.745966692414834
96:     >>> norm(a, np.inf)
97:     4
98:     >>> norm(b, np.inf)
99:     9
100:     >>> norm(a, -np.inf)
101:     0
102:     >>> norm(b, -np.inf)
103:     2
104: 
105:     >>> norm(a, 1)
106:     20
107:     >>> norm(b, 1)
108:     7
109:     >>> norm(a, -1)
110:     -4.6566128774142013e-010
111:     >>> norm(b, -1)
112:     6
113:     >>> norm(a, 2)
114:     7.745966692414834
115:     >>> norm(b, 2)
116:     7.3484692283495345
117: 
118:     >>> norm(a, -2)
119:     0
120:     >>> norm(b, -2)
121:     1.8570331885190563e-016
122:     >>> norm(a, 3)
123:     5.8480354764257312
124:     >>> norm(a, -3)
125:     0
126: 
127:     '''
128:     # Differs from numpy only in non-finite handling and the use of blas.
129:     a = np.asarray_chkfinite(a)
130: 
131:     # Only use optimized norms if axis and keepdims are not specified.
132:     if a.dtype.char in 'fdFD' and axis is None and not keepdims:
133: 
134:         if ord in (None, 2) and (a.ndim == 1):
135:             # use blas for fast and stable euclidean norm
136:             nrm2 = get_blas_funcs('nrm2', dtype=a.dtype)
137:             return nrm2(a)
138: 
139:         if a.ndim == 2 and axis is None and not keepdims:
140:             # Use lapack for a couple fast matrix norms.
141:             # For some reason the *lange frobenius norm is slow.
142:             lange_args = None
143:             # Make sure this works if the user uses the axis keywords
144:             # to apply the norm to the transpose.
145:             if ord == 1:
146:                 if np.isfortran(a):
147:                     lange_args = '1', a
148:                 elif np.isfortran(a.T):
149:                     lange_args = 'i', a.T
150:             elif ord == np.inf:
151:                 if np.isfortran(a):
152:                     lange_args = 'i', a
153:                 elif np.isfortran(a.T):
154:                     lange_args = '1', a.T
155:             if lange_args:
156:                 lange = get_lapack_funcs('lange', dtype=a.dtype)
157:                 return lange(*lange_args)
158: 
159:     # Filter out the axis and keepdims arguments if they aren't used so they
160:     # are never inadvertently passed to a version of numpy that doesn't
161:     # support them.
162:     if axis is not None:
163:         if keepdims:
164:             return np.linalg.norm(a, ord=ord, axis=axis, keepdims=keepdims)
165:         return np.linalg.norm(a, ord=ord, axis=axis)
166:     return np.linalg.norm(a, ord=ord)
167: 
168: 
169: def _datacopied(arr, original):
170:     '''
171:     Strict check for `arr` not sharing any data with `original`,
172:     under the assumption that arr = asarray(original)
173: 
174:     '''
175:     if arr is original:
176:         return False
177:     if not isinstance(original, np.ndarray) and hasattr(original, '__array__'):
178:         return False
179:     return arr.base is None
180: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_23428 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_23428) is not StypyTypeError):

    if (import_23428 != 'pyd_module'):
        __import__(import_23428)
        sys_modules_23429 = sys.modules[import_23428]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_23429.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_23428)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from numpy.linalg import LinAlgError' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_23430 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.linalg')

if (type(import_23430) is not StypyTypeError):

    if (import_23430 != 'pyd_module'):
        __import__(import_23430)
        sys_modules_23431 = sys.modules[import_23430]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.linalg', sys_modules_23431.module_type_store, module_type_store, ['LinAlgError'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_23431, sys_modules_23431.module_type_store, module_type_store)
    else:
        from numpy.linalg import LinAlgError

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.linalg', None, module_type_store, ['LinAlgError'], [LinAlgError])

else:
    # Assigning a type to the variable 'numpy.linalg' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.linalg', import_23430)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from scipy.linalg.blas import get_blas_funcs' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_23432 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.linalg.blas')

if (type(import_23432) is not StypyTypeError):

    if (import_23432 != 'pyd_module'):
        __import__(import_23432)
        sys_modules_23433 = sys.modules[import_23432]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.linalg.blas', sys_modules_23433.module_type_store, module_type_store, ['get_blas_funcs'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_23433, sys_modules_23433.module_type_store, module_type_store)
    else:
        from scipy.linalg.blas import get_blas_funcs

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.linalg.blas', None, module_type_store, ['get_blas_funcs'], [get_blas_funcs])

else:
    # Assigning a type to the variable 'scipy.linalg.blas' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.linalg.blas', import_23432)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from scipy.linalg.lapack import get_lapack_funcs' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_23434 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.linalg.lapack')

if (type(import_23434) is not StypyTypeError):

    if (import_23434 != 'pyd_module'):
        __import__(import_23434)
        sys_modules_23435 = sys.modules[import_23434]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.linalg.lapack', sys_modules_23435.module_type_store, module_type_store, ['get_lapack_funcs'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_23435, sys_modules_23435.module_type_store, module_type_store)
    else:
        from scipy.linalg.lapack import get_lapack_funcs

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.linalg.lapack', None, module_type_store, ['get_lapack_funcs'], [get_lapack_funcs])

else:
    # Assigning a type to the variable 'scipy.linalg.lapack' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.linalg.lapack', import_23434)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')


# Assigning a List to a Name (line 8):
__all__ = ['LinAlgError', 'norm']
module_type_store.set_exportable_members(['LinAlgError', 'norm'])

# Obtaining an instance of the builtin type 'list' (line 8)
list_23436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 8)
# Adding element type (line 8)
str_23437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 11), 'str', 'LinAlgError')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 10), list_23436, str_23437)
# Adding element type (line 8)
str_23438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 26), 'str', 'norm')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 10), list_23436, str_23438)

# Assigning a type to the variable '__all__' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), '__all__', list_23436)

@norecursion
def norm(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 11)
    None_23439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 16), 'None')
    # Getting the type of 'None' (line 11)
    None_23440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 27), 'None')
    # Getting the type of 'False' (line 11)
    False_23441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 42), 'False')
    defaults = [None_23439, None_23440, False_23441]
    # Create a new context for function 'norm'
    module_type_store = module_type_store.open_function_context('norm', 11, 0, False)
    
    # Passed parameters checking function
    norm.stypy_localization = localization
    norm.stypy_type_of_self = None
    norm.stypy_type_store = module_type_store
    norm.stypy_function_name = 'norm'
    norm.stypy_param_names_list = ['a', 'ord', 'axis', 'keepdims']
    norm.stypy_varargs_param_name = None
    norm.stypy_kwargs_param_name = None
    norm.stypy_call_defaults = defaults
    norm.stypy_call_varargs = varargs
    norm.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'norm', ['a', 'ord', 'axis', 'keepdims'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'norm', localization, ['a', 'ord', 'axis', 'keepdims'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'norm(...)' code ##################

    str_23442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, (-1)), 'str', "\n    Matrix or vector norm.\n\n    This function is able to return one of seven different matrix norms,\n    or one of an infinite number of vector norms (described below), depending\n    on the value of the ``ord`` parameter.\n\n    Parameters\n    ----------\n    a : (M,) or (M, N) array_like\n        Input array.  If `axis` is None, `a` must be 1-D or 2-D.\n    ord : {non-zero int, inf, -inf, 'fro'}, optional\n        Order of the norm (see table under ``Notes``). inf means numpy's\n        `inf` object\n    axis : {int, 2-tuple of ints, None}, optional\n        If `axis` is an integer, it specifies the axis of `a` along which to\n        compute the vector norms.  If `axis` is a 2-tuple, it specifies the\n        axes that hold 2-D matrices, and the matrix norms of these matrices\n        are computed.  If `axis` is None then either a vector norm (when `a`\n        is 1-D) or a matrix norm (when `a` is 2-D) is returned.\n    keepdims : bool, optional\n        If this is set to True, the axes which are normed over are left in the\n        result as dimensions with size one.  With this option the result will\n        broadcast correctly against the original `a`.\n\n    Returns\n    -------\n    n : float or ndarray\n        Norm of the matrix or vector(s).\n\n    Notes\n    -----\n    For values of ``ord <= 0``, the result is, strictly speaking, not a\n    mathematical 'norm', but it may still be useful for various numerical\n    purposes.\n\n    The following norms can be calculated:\n\n    =====  ============================  ==========================\n    ord    norm for matrices             norm for vectors\n    =====  ============================  ==========================\n    None   Frobenius norm                2-norm\n    'fro'  Frobenius norm                --\n    inf    max(sum(abs(x), axis=1))      max(abs(x))\n    -inf   min(sum(abs(x), axis=1))      min(abs(x))\n    0      --                            sum(x != 0)\n    1      max(sum(abs(x), axis=0))      as below\n    -1     min(sum(abs(x), axis=0))      as below\n    2      2-norm (largest sing. value)  as below\n    -2     smallest singular value       as below\n    other  --                            sum(abs(x)**ord)**(1./ord)\n    =====  ============================  ==========================\n\n    The Frobenius norm is given by [1]_:\n\n        :math:`||A||_F = [\\sum_{i,j} abs(a_{i,j})^2]^{1/2}`\n\n    The ``axis`` and ``keepdims`` arguments are passed directly to\n    ``numpy.linalg.norm`` and are only usable if they are supported\n    by the version of numpy in use.\n\n    References\n    ----------\n    .. [1] G. H. Golub and C. F. Van Loan, *Matrix Computations*,\n           Baltimore, MD, Johns Hopkins University Press, 1985, pg. 15\n\n    Examples\n    --------\n    >>> from scipy.linalg import norm\n    >>> a = np.arange(9) - 4.0\n    >>> a\n    array([-4., -3., -2., -1.,  0.,  1.,  2.,  3.,  4.])\n    >>> b = a.reshape((3, 3))\n    >>> b\n    array([[-4., -3., -2.],\n           [-1.,  0.,  1.],\n           [ 2.,  3.,  4.]])\n\n    >>> norm(a)\n    7.745966692414834\n    >>> norm(b)\n    7.745966692414834\n    >>> norm(b, 'fro')\n    7.745966692414834\n    >>> norm(a, np.inf)\n    4\n    >>> norm(b, np.inf)\n    9\n    >>> norm(a, -np.inf)\n    0\n    >>> norm(b, -np.inf)\n    2\n\n    >>> norm(a, 1)\n    20\n    >>> norm(b, 1)\n    7\n    >>> norm(a, -1)\n    -4.6566128774142013e-010\n    >>> norm(b, -1)\n    6\n    >>> norm(a, 2)\n    7.745966692414834\n    >>> norm(b, 2)\n    7.3484692283495345\n\n    >>> norm(a, -2)\n    0\n    >>> norm(b, -2)\n    1.8570331885190563e-016\n    >>> norm(a, 3)\n    5.8480354764257312\n    >>> norm(a, -3)\n    0\n\n    ")
    
    # Assigning a Call to a Name (line 129):
    
    # Call to asarray_chkfinite(...): (line 129)
    # Processing the call arguments (line 129)
    # Getting the type of 'a' (line 129)
    a_23445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 29), 'a', False)
    # Processing the call keyword arguments (line 129)
    kwargs_23446 = {}
    # Getting the type of 'np' (line 129)
    np_23443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'np', False)
    # Obtaining the member 'asarray_chkfinite' of a type (line 129)
    asarray_chkfinite_23444 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 8), np_23443, 'asarray_chkfinite')
    # Calling asarray_chkfinite(args, kwargs) (line 129)
    asarray_chkfinite_call_result_23447 = invoke(stypy.reporting.localization.Localization(__file__, 129, 8), asarray_chkfinite_23444, *[a_23445], **kwargs_23446)
    
    # Assigning a type to the variable 'a' (line 129)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 4), 'a', asarray_chkfinite_call_result_23447)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'a' (line 132)
    a_23448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 7), 'a')
    # Obtaining the member 'dtype' of a type (line 132)
    dtype_23449 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 7), a_23448, 'dtype')
    # Obtaining the member 'char' of a type (line 132)
    char_23450 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 7), dtype_23449, 'char')
    str_23451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 23), 'str', 'fdFD')
    # Applying the binary operator 'in' (line 132)
    result_contains_23452 = python_operator(stypy.reporting.localization.Localization(__file__, 132, 7), 'in', char_23450, str_23451)
    
    
    # Getting the type of 'axis' (line 132)
    axis_23453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 34), 'axis')
    # Getting the type of 'None' (line 132)
    None_23454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 42), 'None')
    # Applying the binary operator 'is' (line 132)
    result_is__23455 = python_operator(stypy.reporting.localization.Localization(__file__, 132, 34), 'is', axis_23453, None_23454)
    
    # Applying the binary operator 'and' (line 132)
    result_and_keyword_23456 = python_operator(stypy.reporting.localization.Localization(__file__, 132, 7), 'and', result_contains_23452, result_is__23455)
    
    # Getting the type of 'keepdims' (line 132)
    keepdims_23457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 55), 'keepdims')
    # Applying the 'not' unary operator (line 132)
    result_not__23458 = python_operator(stypy.reporting.localization.Localization(__file__, 132, 51), 'not', keepdims_23457)
    
    # Applying the binary operator 'and' (line 132)
    result_and_keyword_23459 = python_operator(stypy.reporting.localization.Localization(__file__, 132, 7), 'and', result_and_keyword_23456, result_not__23458)
    
    # Testing the type of an if condition (line 132)
    if_condition_23460 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 132, 4), result_and_keyword_23459)
    # Assigning a type to the variable 'if_condition_23460' (line 132)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 4), 'if_condition_23460', if_condition_23460)
    # SSA begins for if statement (line 132)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'ord' (line 134)
    ord_23461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 11), 'ord')
    
    # Obtaining an instance of the builtin type 'tuple' (line 134)
    tuple_23462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 19), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 134)
    # Adding element type (line 134)
    # Getting the type of 'None' (line 134)
    None_23463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 19), 'None')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 134, 19), tuple_23462, None_23463)
    # Adding element type (line 134)
    int_23464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 134, 19), tuple_23462, int_23464)
    
    # Applying the binary operator 'in' (line 134)
    result_contains_23465 = python_operator(stypy.reporting.localization.Localization(__file__, 134, 11), 'in', ord_23461, tuple_23462)
    
    
    # Getting the type of 'a' (line 134)
    a_23466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 33), 'a')
    # Obtaining the member 'ndim' of a type (line 134)
    ndim_23467 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 33), a_23466, 'ndim')
    int_23468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 43), 'int')
    # Applying the binary operator '==' (line 134)
    result_eq_23469 = python_operator(stypy.reporting.localization.Localization(__file__, 134, 33), '==', ndim_23467, int_23468)
    
    # Applying the binary operator 'and' (line 134)
    result_and_keyword_23470 = python_operator(stypy.reporting.localization.Localization(__file__, 134, 11), 'and', result_contains_23465, result_eq_23469)
    
    # Testing the type of an if condition (line 134)
    if_condition_23471 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 134, 8), result_and_keyword_23470)
    # Assigning a type to the variable 'if_condition_23471' (line 134)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'if_condition_23471', if_condition_23471)
    # SSA begins for if statement (line 134)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 136):
    
    # Call to get_blas_funcs(...): (line 136)
    # Processing the call arguments (line 136)
    str_23473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 34), 'str', 'nrm2')
    # Processing the call keyword arguments (line 136)
    # Getting the type of 'a' (line 136)
    a_23474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 48), 'a', False)
    # Obtaining the member 'dtype' of a type (line 136)
    dtype_23475 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 48), a_23474, 'dtype')
    keyword_23476 = dtype_23475
    kwargs_23477 = {'dtype': keyword_23476}
    # Getting the type of 'get_blas_funcs' (line 136)
    get_blas_funcs_23472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 19), 'get_blas_funcs', False)
    # Calling get_blas_funcs(args, kwargs) (line 136)
    get_blas_funcs_call_result_23478 = invoke(stypy.reporting.localization.Localization(__file__, 136, 19), get_blas_funcs_23472, *[str_23473], **kwargs_23477)
    
    # Assigning a type to the variable 'nrm2' (line 136)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 12), 'nrm2', get_blas_funcs_call_result_23478)
    
    # Call to nrm2(...): (line 137)
    # Processing the call arguments (line 137)
    # Getting the type of 'a' (line 137)
    a_23480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 24), 'a', False)
    # Processing the call keyword arguments (line 137)
    kwargs_23481 = {}
    # Getting the type of 'nrm2' (line 137)
    nrm2_23479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 19), 'nrm2', False)
    # Calling nrm2(args, kwargs) (line 137)
    nrm2_call_result_23482 = invoke(stypy.reporting.localization.Localization(__file__, 137, 19), nrm2_23479, *[a_23480], **kwargs_23481)
    
    # Assigning a type to the variable 'stypy_return_type' (line 137)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 12), 'stypy_return_type', nrm2_call_result_23482)
    # SSA join for if statement (line 134)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'a' (line 139)
    a_23483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 11), 'a')
    # Obtaining the member 'ndim' of a type (line 139)
    ndim_23484 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 11), a_23483, 'ndim')
    int_23485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 21), 'int')
    # Applying the binary operator '==' (line 139)
    result_eq_23486 = python_operator(stypy.reporting.localization.Localization(__file__, 139, 11), '==', ndim_23484, int_23485)
    
    
    # Getting the type of 'axis' (line 139)
    axis_23487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 27), 'axis')
    # Getting the type of 'None' (line 139)
    None_23488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 35), 'None')
    # Applying the binary operator 'is' (line 139)
    result_is__23489 = python_operator(stypy.reporting.localization.Localization(__file__, 139, 27), 'is', axis_23487, None_23488)
    
    # Applying the binary operator 'and' (line 139)
    result_and_keyword_23490 = python_operator(stypy.reporting.localization.Localization(__file__, 139, 11), 'and', result_eq_23486, result_is__23489)
    
    # Getting the type of 'keepdims' (line 139)
    keepdims_23491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 48), 'keepdims')
    # Applying the 'not' unary operator (line 139)
    result_not__23492 = python_operator(stypy.reporting.localization.Localization(__file__, 139, 44), 'not', keepdims_23491)
    
    # Applying the binary operator 'and' (line 139)
    result_and_keyword_23493 = python_operator(stypy.reporting.localization.Localization(__file__, 139, 11), 'and', result_and_keyword_23490, result_not__23492)
    
    # Testing the type of an if condition (line 139)
    if_condition_23494 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 139, 8), result_and_keyword_23493)
    # Assigning a type to the variable 'if_condition_23494' (line 139)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 8), 'if_condition_23494', if_condition_23494)
    # SSA begins for if statement (line 139)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 142):
    # Getting the type of 'None' (line 142)
    None_23495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 25), 'None')
    # Assigning a type to the variable 'lange_args' (line 142)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 12), 'lange_args', None_23495)
    
    
    # Getting the type of 'ord' (line 145)
    ord_23496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 15), 'ord')
    int_23497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 22), 'int')
    # Applying the binary operator '==' (line 145)
    result_eq_23498 = python_operator(stypy.reporting.localization.Localization(__file__, 145, 15), '==', ord_23496, int_23497)
    
    # Testing the type of an if condition (line 145)
    if_condition_23499 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 145, 12), result_eq_23498)
    # Assigning a type to the variable 'if_condition_23499' (line 145)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 12), 'if_condition_23499', if_condition_23499)
    # SSA begins for if statement (line 145)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Call to isfortran(...): (line 146)
    # Processing the call arguments (line 146)
    # Getting the type of 'a' (line 146)
    a_23502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 32), 'a', False)
    # Processing the call keyword arguments (line 146)
    kwargs_23503 = {}
    # Getting the type of 'np' (line 146)
    np_23500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 19), 'np', False)
    # Obtaining the member 'isfortran' of a type (line 146)
    isfortran_23501 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 19), np_23500, 'isfortran')
    # Calling isfortran(args, kwargs) (line 146)
    isfortran_call_result_23504 = invoke(stypy.reporting.localization.Localization(__file__, 146, 19), isfortran_23501, *[a_23502], **kwargs_23503)
    
    # Testing the type of an if condition (line 146)
    if_condition_23505 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 146, 16), isfortran_call_result_23504)
    # Assigning a type to the variable 'if_condition_23505' (line 146)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 16), 'if_condition_23505', if_condition_23505)
    # SSA begins for if statement (line 146)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Tuple to a Name (line 147):
    
    # Obtaining an instance of the builtin type 'tuple' (line 147)
    tuple_23506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 33), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 147)
    # Adding element type (line 147)
    str_23507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 33), 'str', '1')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 33), tuple_23506, str_23507)
    # Adding element type (line 147)
    # Getting the type of 'a' (line 147)
    a_23508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 38), 'a')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 33), tuple_23506, a_23508)
    
    # Assigning a type to the variable 'lange_args' (line 147)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 20), 'lange_args', tuple_23506)
    # SSA branch for the else part of an if statement (line 146)
    module_type_store.open_ssa_branch('else')
    
    
    # Call to isfortran(...): (line 148)
    # Processing the call arguments (line 148)
    # Getting the type of 'a' (line 148)
    a_23511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 34), 'a', False)
    # Obtaining the member 'T' of a type (line 148)
    T_23512 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 34), a_23511, 'T')
    # Processing the call keyword arguments (line 148)
    kwargs_23513 = {}
    # Getting the type of 'np' (line 148)
    np_23509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 21), 'np', False)
    # Obtaining the member 'isfortran' of a type (line 148)
    isfortran_23510 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 21), np_23509, 'isfortran')
    # Calling isfortran(args, kwargs) (line 148)
    isfortran_call_result_23514 = invoke(stypy.reporting.localization.Localization(__file__, 148, 21), isfortran_23510, *[T_23512], **kwargs_23513)
    
    # Testing the type of an if condition (line 148)
    if_condition_23515 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 148, 21), isfortran_call_result_23514)
    # Assigning a type to the variable 'if_condition_23515' (line 148)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 21), 'if_condition_23515', if_condition_23515)
    # SSA begins for if statement (line 148)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Tuple to a Name (line 149):
    
    # Obtaining an instance of the builtin type 'tuple' (line 149)
    tuple_23516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 33), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 149)
    # Adding element type (line 149)
    str_23517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 33), 'str', 'i')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 149, 33), tuple_23516, str_23517)
    # Adding element type (line 149)
    # Getting the type of 'a' (line 149)
    a_23518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 38), 'a')
    # Obtaining the member 'T' of a type (line 149)
    T_23519 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 38), a_23518, 'T')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 149, 33), tuple_23516, T_23519)
    
    # Assigning a type to the variable 'lange_args' (line 149)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 20), 'lange_args', tuple_23516)
    # SSA join for if statement (line 148)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 146)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 145)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'ord' (line 150)
    ord_23520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 17), 'ord')
    # Getting the type of 'np' (line 150)
    np_23521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 24), 'np')
    # Obtaining the member 'inf' of a type (line 150)
    inf_23522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 24), np_23521, 'inf')
    # Applying the binary operator '==' (line 150)
    result_eq_23523 = python_operator(stypy.reporting.localization.Localization(__file__, 150, 17), '==', ord_23520, inf_23522)
    
    # Testing the type of an if condition (line 150)
    if_condition_23524 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 150, 17), result_eq_23523)
    # Assigning a type to the variable 'if_condition_23524' (line 150)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 17), 'if_condition_23524', if_condition_23524)
    # SSA begins for if statement (line 150)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Call to isfortran(...): (line 151)
    # Processing the call arguments (line 151)
    # Getting the type of 'a' (line 151)
    a_23527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 32), 'a', False)
    # Processing the call keyword arguments (line 151)
    kwargs_23528 = {}
    # Getting the type of 'np' (line 151)
    np_23525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 19), 'np', False)
    # Obtaining the member 'isfortran' of a type (line 151)
    isfortran_23526 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 19), np_23525, 'isfortran')
    # Calling isfortran(args, kwargs) (line 151)
    isfortran_call_result_23529 = invoke(stypy.reporting.localization.Localization(__file__, 151, 19), isfortran_23526, *[a_23527], **kwargs_23528)
    
    # Testing the type of an if condition (line 151)
    if_condition_23530 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 151, 16), isfortran_call_result_23529)
    # Assigning a type to the variable 'if_condition_23530' (line 151)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 16), 'if_condition_23530', if_condition_23530)
    # SSA begins for if statement (line 151)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Tuple to a Name (line 152):
    
    # Obtaining an instance of the builtin type 'tuple' (line 152)
    tuple_23531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 33), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 152)
    # Adding element type (line 152)
    str_23532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 33), 'str', 'i')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 33), tuple_23531, str_23532)
    # Adding element type (line 152)
    # Getting the type of 'a' (line 152)
    a_23533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 38), 'a')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 33), tuple_23531, a_23533)
    
    # Assigning a type to the variable 'lange_args' (line 152)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 20), 'lange_args', tuple_23531)
    # SSA branch for the else part of an if statement (line 151)
    module_type_store.open_ssa_branch('else')
    
    
    # Call to isfortran(...): (line 153)
    # Processing the call arguments (line 153)
    # Getting the type of 'a' (line 153)
    a_23536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 34), 'a', False)
    # Obtaining the member 'T' of a type (line 153)
    T_23537 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 34), a_23536, 'T')
    # Processing the call keyword arguments (line 153)
    kwargs_23538 = {}
    # Getting the type of 'np' (line 153)
    np_23534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 21), 'np', False)
    # Obtaining the member 'isfortran' of a type (line 153)
    isfortran_23535 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 21), np_23534, 'isfortran')
    # Calling isfortran(args, kwargs) (line 153)
    isfortran_call_result_23539 = invoke(stypy.reporting.localization.Localization(__file__, 153, 21), isfortran_23535, *[T_23537], **kwargs_23538)
    
    # Testing the type of an if condition (line 153)
    if_condition_23540 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 153, 21), isfortran_call_result_23539)
    # Assigning a type to the variable 'if_condition_23540' (line 153)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 21), 'if_condition_23540', if_condition_23540)
    # SSA begins for if statement (line 153)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Tuple to a Name (line 154):
    
    # Obtaining an instance of the builtin type 'tuple' (line 154)
    tuple_23541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 33), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 154)
    # Adding element type (line 154)
    str_23542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 33), 'str', '1')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 154, 33), tuple_23541, str_23542)
    # Adding element type (line 154)
    # Getting the type of 'a' (line 154)
    a_23543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 38), 'a')
    # Obtaining the member 'T' of a type (line 154)
    T_23544 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 38), a_23543, 'T')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 154, 33), tuple_23541, T_23544)
    
    # Assigning a type to the variable 'lange_args' (line 154)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 20), 'lange_args', tuple_23541)
    # SSA join for if statement (line 153)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 151)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 150)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 145)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'lange_args' (line 155)
    lange_args_23545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 15), 'lange_args')
    # Testing the type of an if condition (line 155)
    if_condition_23546 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 155, 12), lange_args_23545)
    # Assigning a type to the variable 'if_condition_23546' (line 155)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 12), 'if_condition_23546', if_condition_23546)
    # SSA begins for if statement (line 155)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 156):
    
    # Call to get_lapack_funcs(...): (line 156)
    # Processing the call arguments (line 156)
    str_23548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 41), 'str', 'lange')
    # Processing the call keyword arguments (line 156)
    # Getting the type of 'a' (line 156)
    a_23549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 56), 'a', False)
    # Obtaining the member 'dtype' of a type (line 156)
    dtype_23550 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 56), a_23549, 'dtype')
    keyword_23551 = dtype_23550
    kwargs_23552 = {'dtype': keyword_23551}
    # Getting the type of 'get_lapack_funcs' (line 156)
    get_lapack_funcs_23547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 24), 'get_lapack_funcs', False)
    # Calling get_lapack_funcs(args, kwargs) (line 156)
    get_lapack_funcs_call_result_23553 = invoke(stypy.reporting.localization.Localization(__file__, 156, 24), get_lapack_funcs_23547, *[str_23548], **kwargs_23552)
    
    # Assigning a type to the variable 'lange' (line 156)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 16), 'lange', get_lapack_funcs_call_result_23553)
    
    # Call to lange(...): (line 157)
    # Getting the type of 'lange_args' (line 157)
    lange_args_23555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 30), 'lange_args', False)
    # Processing the call keyword arguments (line 157)
    kwargs_23556 = {}
    # Getting the type of 'lange' (line 157)
    lange_23554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 23), 'lange', False)
    # Calling lange(args, kwargs) (line 157)
    lange_call_result_23557 = invoke(stypy.reporting.localization.Localization(__file__, 157, 23), lange_23554, *[lange_args_23555], **kwargs_23556)
    
    # Assigning a type to the variable 'stypy_return_type' (line 157)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 16), 'stypy_return_type', lange_call_result_23557)
    # SSA join for if statement (line 155)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 139)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 132)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 162)
    # Getting the type of 'axis' (line 162)
    axis_23558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 4), 'axis')
    # Getting the type of 'None' (line 162)
    None_23559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 19), 'None')
    
    (may_be_23560, more_types_in_union_23561) = may_not_be_none(axis_23558, None_23559)

    if may_be_23560:

        if more_types_in_union_23561:
            # Runtime conditional SSA (line 162)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Getting the type of 'keepdims' (line 163)
        keepdims_23562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 11), 'keepdims')
        # Testing the type of an if condition (line 163)
        if_condition_23563 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 163, 8), keepdims_23562)
        # Assigning a type to the variable 'if_condition_23563' (line 163)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 8), 'if_condition_23563', if_condition_23563)
        # SSA begins for if statement (line 163)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to norm(...): (line 164)
        # Processing the call arguments (line 164)
        # Getting the type of 'a' (line 164)
        a_23567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 34), 'a', False)
        # Processing the call keyword arguments (line 164)
        # Getting the type of 'ord' (line 164)
        ord_23568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 41), 'ord', False)
        keyword_23569 = ord_23568
        # Getting the type of 'axis' (line 164)
        axis_23570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 51), 'axis', False)
        keyword_23571 = axis_23570
        # Getting the type of 'keepdims' (line 164)
        keepdims_23572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 66), 'keepdims', False)
        keyword_23573 = keepdims_23572
        kwargs_23574 = {'ord': keyword_23569, 'keepdims': keyword_23573, 'axis': keyword_23571}
        # Getting the type of 'np' (line 164)
        np_23564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 19), 'np', False)
        # Obtaining the member 'linalg' of a type (line 164)
        linalg_23565 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 19), np_23564, 'linalg')
        # Obtaining the member 'norm' of a type (line 164)
        norm_23566 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 19), linalg_23565, 'norm')
        # Calling norm(args, kwargs) (line 164)
        norm_call_result_23575 = invoke(stypy.reporting.localization.Localization(__file__, 164, 19), norm_23566, *[a_23567], **kwargs_23574)
        
        # Assigning a type to the variable 'stypy_return_type' (line 164)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 12), 'stypy_return_type', norm_call_result_23575)
        # SSA join for if statement (line 163)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to norm(...): (line 165)
        # Processing the call arguments (line 165)
        # Getting the type of 'a' (line 165)
        a_23579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 30), 'a', False)
        # Processing the call keyword arguments (line 165)
        # Getting the type of 'ord' (line 165)
        ord_23580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 37), 'ord', False)
        keyword_23581 = ord_23580
        # Getting the type of 'axis' (line 165)
        axis_23582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 47), 'axis', False)
        keyword_23583 = axis_23582
        kwargs_23584 = {'ord': keyword_23581, 'axis': keyword_23583}
        # Getting the type of 'np' (line 165)
        np_23576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 15), 'np', False)
        # Obtaining the member 'linalg' of a type (line 165)
        linalg_23577 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 15), np_23576, 'linalg')
        # Obtaining the member 'norm' of a type (line 165)
        norm_23578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 15), linalg_23577, 'norm')
        # Calling norm(args, kwargs) (line 165)
        norm_call_result_23585 = invoke(stypy.reporting.localization.Localization(__file__, 165, 15), norm_23578, *[a_23579], **kwargs_23584)
        
        # Assigning a type to the variable 'stypy_return_type' (line 165)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 8), 'stypy_return_type', norm_call_result_23585)

        if more_types_in_union_23561:
            # SSA join for if statement (line 162)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Call to norm(...): (line 166)
    # Processing the call arguments (line 166)
    # Getting the type of 'a' (line 166)
    a_23589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 26), 'a', False)
    # Processing the call keyword arguments (line 166)
    # Getting the type of 'ord' (line 166)
    ord_23590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 33), 'ord', False)
    keyword_23591 = ord_23590
    kwargs_23592 = {'ord': keyword_23591}
    # Getting the type of 'np' (line 166)
    np_23586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 11), 'np', False)
    # Obtaining the member 'linalg' of a type (line 166)
    linalg_23587 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 11), np_23586, 'linalg')
    # Obtaining the member 'norm' of a type (line 166)
    norm_23588 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 11), linalg_23587, 'norm')
    # Calling norm(args, kwargs) (line 166)
    norm_call_result_23593 = invoke(stypy.reporting.localization.Localization(__file__, 166, 11), norm_23588, *[a_23589], **kwargs_23592)
    
    # Assigning a type to the variable 'stypy_return_type' (line 166)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 4), 'stypy_return_type', norm_call_result_23593)
    
    # ################# End of 'norm(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'norm' in the type store
    # Getting the type of 'stypy_return_type' (line 11)
    stypy_return_type_23594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_23594)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'norm'
    return stypy_return_type_23594

# Assigning a type to the variable 'norm' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'norm', norm)

@norecursion
def _datacopied(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_datacopied'
    module_type_store = module_type_store.open_function_context('_datacopied', 169, 0, False)
    
    # Passed parameters checking function
    _datacopied.stypy_localization = localization
    _datacopied.stypy_type_of_self = None
    _datacopied.stypy_type_store = module_type_store
    _datacopied.stypy_function_name = '_datacopied'
    _datacopied.stypy_param_names_list = ['arr', 'original']
    _datacopied.stypy_varargs_param_name = None
    _datacopied.stypy_kwargs_param_name = None
    _datacopied.stypy_call_defaults = defaults
    _datacopied.stypy_call_varargs = varargs
    _datacopied.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_datacopied', ['arr', 'original'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_datacopied', localization, ['arr', 'original'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_datacopied(...)' code ##################

    str_23595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, (-1)), 'str', '\n    Strict check for `arr` not sharing any data with `original`,\n    under the assumption that arr = asarray(original)\n\n    ')
    
    
    # Getting the type of 'arr' (line 175)
    arr_23596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 7), 'arr')
    # Getting the type of 'original' (line 175)
    original_23597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 14), 'original')
    # Applying the binary operator 'is' (line 175)
    result_is__23598 = python_operator(stypy.reporting.localization.Localization(__file__, 175, 7), 'is', arr_23596, original_23597)
    
    # Testing the type of an if condition (line 175)
    if_condition_23599 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 175, 4), result_is__23598)
    # Assigning a type to the variable 'if_condition_23599' (line 175)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 4), 'if_condition_23599', if_condition_23599)
    # SSA begins for if statement (line 175)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'False' (line 176)
    False_23600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 15), 'False')
    # Assigning a type to the variable 'stypy_return_type' (line 176)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'stypy_return_type', False_23600)
    # SSA join for if statement (line 175)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    
    # Call to isinstance(...): (line 177)
    # Processing the call arguments (line 177)
    # Getting the type of 'original' (line 177)
    original_23602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 22), 'original', False)
    # Getting the type of 'np' (line 177)
    np_23603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 32), 'np', False)
    # Obtaining the member 'ndarray' of a type (line 177)
    ndarray_23604 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 32), np_23603, 'ndarray')
    # Processing the call keyword arguments (line 177)
    kwargs_23605 = {}
    # Getting the type of 'isinstance' (line 177)
    isinstance_23601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 11), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 177)
    isinstance_call_result_23606 = invoke(stypy.reporting.localization.Localization(__file__, 177, 11), isinstance_23601, *[original_23602, ndarray_23604], **kwargs_23605)
    
    # Applying the 'not' unary operator (line 177)
    result_not__23607 = python_operator(stypy.reporting.localization.Localization(__file__, 177, 7), 'not', isinstance_call_result_23606)
    
    
    # Call to hasattr(...): (line 177)
    # Processing the call arguments (line 177)
    # Getting the type of 'original' (line 177)
    original_23609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 56), 'original', False)
    str_23610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 66), 'str', '__array__')
    # Processing the call keyword arguments (line 177)
    kwargs_23611 = {}
    # Getting the type of 'hasattr' (line 177)
    hasattr_23608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 48), 'hasattr', False)
    # Calling hasattr(args, kwargs) (line 177)
    hasattr_call_result_23612 = invoke(stypy.reporting.localization.Localization(__file__, 177, 48), hasattr_23608, *[original_23609, str_23610], **kwargs_23611)
    
    # Applying the binary operator 'and' (line 177)
    result_and_keyword_23613 = python_operator(stypy.reporting.localization.Localization(__file__, 177, 7), 'and', result_not__23607, hasattr_call_result_23612)
    
    # Testing the type of an if condition (line 177)
    if_condition_23614 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 177, 4), result_and_keyword_23613)
    # Assigning a type to the variable 'if_condition_23614' (line 177)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 4), 'if_condition_23614', if_condition_23614)
    # SSA begins for if statement (line 177)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'False' (line 178)
    False_23615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 15), 'False')
    # Assigning a type to the variable 'stypy_return_type' (line 178)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'stypy_return_type', False_23615)
    # SSA join for if statement (line 177)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'arr' (line 179)
    arr_23616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 11), 'arr')
    # Obtaining the member 'base' of a type (line 179)
    base_23617 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 11), arr_23616, 'base')
    # Getting the type of 'None' (line 179)
    None_23618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 23), 'None')
    # Applying the binary operator 'is' (line 179)
    result_is__23619 = python_operator(stypy.reporting.localization.Localization(__file__, 179, 11), 'is', base_23617, None_23618)
    
    # Assigning a type to the variable 'stypy_return_type' (line 179)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 4), 'stypy_return_type', result_is__23619)
    
    # ################# End of '_datacopied(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_datacopied' in the type store
    # Getting the type of 'stypy_return_type' (line 169)
    stypy_return_type_23620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_23620)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_datacopied'
    return stypy_return_type_23620

# Assigning a type to the variable '_datacopied' (line 169)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 0), '_datacopied', _datacopied)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
