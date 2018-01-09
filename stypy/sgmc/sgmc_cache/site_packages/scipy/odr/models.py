
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: ''' Collection of Model instances for use with the odrpack fitting package.
2: '''
3: from __future__ import division, print_function, absolute_import
4: 
5: import numpy as np
6: from scipy.odr.odrpack import Model
7: 
8: __all__ = ['Model', 'exponential', 'multilinear', 'unilinear', 'quadratic',
9:            'polynomial']
10: 
11: 
12: def _lin_fcn(B, x):
13:     a, b = B[0], B[1:]
14:     b.shape = (b.shape[0], 1)
15: 
16:     return a + (x*b).sum(axis=0)
17: 
18: 
19: def _lin_fjb(B, x):
20:     a = np.ones(x.shape[-1], float)
21:     res = np.concatenate((a, x.ravel()))
22:     res.shape = (B.shape[-1], x.shape[-1])
23:     return res
24: 
25: 
26: def _lin_fjd(B, x):
27:     b = B[1:]
28:     b = np.repeat(b, (x.shape[-1],)*b.shape[-1],axis=0)
29:     b.shape = x.shape
30:     return b
31: 
32: 
33: def _lin_est(data):
34:     # Eh. The answer is analytical, so just return all ones.
35:     # Don't return zeros since that will interfere with
36:     # ODRPACK's auto-scaling procedures.
37: 
38:     if len(data.x.shape) == 2:
39:         m = data.x.shape[0]
40:     else:
41:         m = 1
42: 
43:     return np.ones((m + 1,), float)
44: 
45: 
46: def _poly_fcn(B, x, powers):
47:     a, b = B[0], B[1:]
48:     b.shape = (b.shape[0], 1)
49: 
50:     return a + np.sum(b * np.power(x, powers), axis=0)
51: 
52: 
53: def _poly_fjacb(B, x, powers):
54:     res = np.concatenate((np.ones(x.shape[-1], float), np.power(x,
55:         powers).flat))
56:     res.shape = (B.shape[-1], x.shape[-1])
57:     return res
58: 
59: 
60: def _poly_fjacd(B, x, powers):
61:     b = B[1:]
62:     b.shape = (b.shape[0], 1)
63: 
64:     b = b * powers
65: 
66:     return np.sum(b * np.power(x, powers-1),axis=0)
67: 
68: 
69: def _exp_fcn(B, x):
70:     return B[0] + np.exp(B[1] * x)
71: 
72: 
73: def _exp_fjd(B, x):
74:     return B[1] * np.exp(B[1] * x)
75: 
76: 
77: def _exp_fjb(B, x):
78:     res = np.concatenate((np.ones(x.shape[-1], float), x * np.exp(B[1] * x)))
79:     res.shape = (2, x.shape[-1])
80:     return res
81: 
82: 
83: def _exp_est(data):
84:     # Eh.
85:     return np.array([1., 1.])
86: 
87: multilinear = Model(_lin_fcn, fjacb=_lin_fjb,
88:                fjacd=_lin_fjd, estimate=_lin_est,
89:                meta={'name': 'Arbitrary-dimensional Linear',
90:                      'equ':'y = B_0 + Sum[i=1..m, B_i * x_i]',
91:                      'TeXequ':r'$y=\beta_0 + \sum_{i=1}^m \beta_i x_i$'})
92: 
93: 
94: def polynomial(order):
95:     '''
96:     Factory function for a general polynomial model.
97: 
98:     Parameters
99:     ----------
100:     order : int or sequence
101:         If an integer, it becomes the order of the polynomial to fit. If
102:         a sequence of numbers, then these are the explicit powers in the
103:         polynomial.
104:         A constant term (power 0) is always included, so don't include 0.
105:         Thus, polynomial(n) is equivalent to polynomial(range(1, n+1)).
106: 
107:     Returns
108:     -------
109:     polynomial : Model instance
110:         Model instance.
111: 
112:     '''
113: 
114:     powers = np.asarray(order)
115:     if powers.shape == ():
116:         # Scalar.
117:         powers = np.arange(1, powers + 1)
118: 
119:     powers.shape = (len(powers), 1)
120:     len_beta = len(powers) + 1
121: 
122:     def _poly_est(data, len_beta=len_beta):
123:         # Eh. Ignore data and return all ones.
124:         return np.ones((len_beta,), float)
125: 
126:     return Model(_poly_fcn, fjacd=_poly_fjacd, fjacb=_poly_fjacb,
127:                  estimate=_poly_est, extra_args=(powers,),
128:                  meta={'name': 'Sorta-general Polynomial',
129:                  'equ': 'y = B_0 + Sum[i=1..%s, B_i * (x**i)]' % (len_beta-1),
130:                  'TeXequ': r'$y=\beta_0 + \sum_{i=1}^{%s} \beta_i x^i$' %
131:                         (len_beta-1)})
132: 
133: exponential = Model(_exp_fcn, fjacd=_exp_fjd, fjacb=_exp_fjb,
134:                     estimate=_exp_est, meta={'name':'Exponential',
135:                     'equ': 'y= B_0 + exp(B_1 * x)',
136:                     'TeXequ': r'$y=\beta_0 + e^{\beta_1 x}$'})
137: 
138: 
139: def _unilin(B, x):
140:     return x*B[0] + B[1]
141: 
142: 
143: def _unilin_fjd(B, x):
144:     return np.ones(x.shape, float) * B[0]
145: 
146: 
147: def _unilin_fjb(B, x):
148:     _ret = np.concatenate((x, np.ones(x.shape, float)))
149:     _ret.shape = (2,) + x.shape
150: 
151:     return _ret
152: 
153: 
154: def _unilin_est(data):
155:     return (1., 1.)
156: 
157: 
158: def _quadratic(B, x):
159:     return x*(x*B[0] + B[1]) + B[2]
160: 
161: 
162: def _quad_fjd(B, x):
163:     return 2*x*B[0] + B[1]
164: 
165: 
166: def _quad_fjb(B, x):
167:     _ret = np.concatenate((x*x, x, np.ones(x.shape, float)))
168:     _ret.shape = (3,) + x.shape
169: 
170:     return _ret
171: 
172: 
173: def _quad_est(data):
174:     return (1.,1.,1.)
175: 
176: unilinear = Model(_unilin, fjacd=_unilin_fjd, fjacb=_unilin_fjb,
177:                   estimate=_unilin_est, meta={'name': 'Univariate Linear',
178:                   'equ': 'y = B_0 * x + B_1',
179:                   'TeXequ': '$y = \\beta_0 x + \\beta_1$'})
180: 
181: quadratic = Model(_quadratic, fjacd=_quad_fjd, fjacb=_quad_fjb,
182:                   estimate=_quad_est, meta={'name': 'Quadratic',
183:                   'equ': 'y = B_0*x**2 + B_1*x + B_2',
184:                   'TeXequ': '$y = \\beta_0 x^2 + \\beta_1 x + \\beta_2'})
185: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_162968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, (-1)), 'str', ' Collection of Model instances for use with the odrpack fitting package.\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import numpy' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/odr/')
import_162969 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy')

if (type(import_162969) is not StypyTypeError):

    if (import_162969 != 'pyd_module'):
        __import__(import_162969)
        sys_modules_162970 = sys.modules[import_162969]
        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'np', sys_modules_162970.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy', import_162969)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/odr/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from scipy.odr.odrpack import Model' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/odr/')
import_162971 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.odr.odrpack')

if (type(import_162971) is not StypyTypeError):

    if (import_162971 != 'pyd_module'):
        __import__(import_162971)
        sys_modules_162972 = sys.modules[import_162971]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.odr.odrpack', sys_modules_162972.module_type_store, module_type_store, ['Model'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_162972, sys_modules_162972.module_type_store, module_type_store)
    else:
        from scipy.odr.odrpack import Model

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.odr.odrpack', None, module_type_store, ['Model'], [Model])

else:
    # Assigning a type to the variable 'scipy.odr.odrpack' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.odr.odrpack', import_162971)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/odr/')


# Assigning a List to a Name (line 8):

# Assigning a List to a Name (line 8):
__all__ = ['Model', 'exponential', 'multilinear', 'unilinear', 'quadratic', 'polynomial']
module_type_store.set_exportable_members(['Model', 'exponential', 'multilinear', 'unilinear', 'quadratic', 'polynomial'])

# Obtaining an instance of the builtin type 'list' (line 8)
list_162973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 8)
# Adding element type (line 8)
str_162974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 11), 'str', 'Model')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 10), list_162973, str_162974)
# Adding element type (line 8)
str_162975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 20), 'str', 'exponential')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 10), list_162973, str_162975)
# Adding element type (line 8)
str_162976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 35), 'str', 'multilinear')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 10), list_162973, str_162976)
# Adding element type (line 8)
str_162977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 50), 'str', 'unilinear')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 10), list_162973, str_162977)
# Adding element type (line 8)
str_162978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 63), 'str', 'quadratic')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 10), list_162973, str_162978)
# Adding element type (line 8)
str_162979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 11), 'str', 'polynomial')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 10), list_162973, str_162979)

# Assigning a type to the variable '__all__' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), '__all__', list_162973)

@norecursion
def _lin_fcn(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_lin_fcn'
    module_type_store = module_type_store.open_function_context('_lin_fcn', 12, 0, False)
    
    # Passed parameters checking function
    _lin_fcn.stypy_localization = localization
    _lin_fcn.stypy_type_of_self = None
    _lin_fcn.stypy_type_store = module_type_store
    _lin_fcn.stypy_function_name = '_lin_fcn'
    _lin_fcn.stypy_param_names_list = ['B', 'x']
    _lin_fcn.stypy_varargs_param_name = None
    _lin_fcn.stypy_kwargs_param_name = None
    _lin_fcn.stypy_call_defaults = defaults
    _lin_fcn.stypy_call_varargs = varargs
    _lin_fcn.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_lin_fcn', ['B', 'x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_lin_fcn', localization, ['B', 'x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_lin_fcn(...)' code ##################

    
    # Assigning a Tuple to a Tuple (line 13):
    
    # Assigning a Subscript to a Name (line 13):
    
    # Obtaining the type of the subscript
    int_162980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 13), 'int')
    # Getting the type of 'B' (line 13)
    B_162981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 11), 'B')
    # Obtaining the member '__getitem__' of a type (line 13)
    getitem___162982 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 11), B_162981, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 13)
    subscript_call_result_162983 = invoke(stypy.reporting.localization.Localization(__file__, 13, 11), getitem___162982, int_162980)
    
    # Assigning a type to the variable 'tuple_assignment_162964' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'tuple_assignment_162964', subscript_call_result_162983)
    
    # Assigning a Subscript to a Name (line 13):
    
    # Obtaining the type of the subscript
    int_162984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 19), 'int')
    slice_162985 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 13, 17), int_162984, None, None)
    # Getting the type of 'B' (line 13)
    B_162986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 17), 'B')
    # Obtaining the member '__getitem__' of a type (line 13)
    getitem___162987 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 17), B_162986, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 13)
    subscript_call_result_162988 = invoke(stypy.reporting.localization.Localization(__file__, 13, 17), getitem___162987, slice_162985)
    
    # Assigning a type to the variable 'tuple_assignment_162965' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'tuple_assignment_162965', subscript_call_result_162988)
    
    # Assigning a Name to a Name (line 13):
    # Getting the type of 'tuple_assignment_162964' (line 13)
    tuple_assignment_162964_162989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'tuple_assignment_162964')
    # Assigning a type to the variable 'a' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'a', tuple_assignment_162964_162989)
    
    # Assigning a Name to a Name (line 13):
    # Getting the type of 'tuple_assignment_162965' (line 13)
    tuple_assignment_162965_162990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'tuple_assignment_162965')
    # Assigning a type to the variable 'b' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 7), 'b', tuple_assignment_162965_162990)
    
    # Assigning a Tuple to a Attribute (line 14):
    
    # Assigning a Tuple to a Attribute (line 14):
    
    # Obtaining an instance of the builtin type 'tuple' (line 14)
    tuple_162991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 14)
    # Adding element type (line 14)
    
    # Obtaining the type of the subscript
    int_162992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 23), 'int')
    # Getting the type of 'b' (line 14)
    b_162993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 15), 'b')
    # Obtaining the member 'shape' of a type (line 14)
    shape_162994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 15), b_162993, 'shape')
    # Obtaining the member '__getitem__' of a type (line 14)
    getitem___162995 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 15), shape_162994, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 14)
    subscript_call_result_162996 = invoke(stypy.reporting.localization.Localization(__file__, 14, 15), getitem___162995, int_162992)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 15), tuple_162991, subscript_call_result_162996)
    # Adding element type (line 14)
    int_162997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 15), tuple_162991, int_162997)
    
    # Getting the type of 'b' (line 14)
    b_162998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'b')
    # Setting the type of the member 'shape' of a type (line 14)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 4), b_162998, 'shape', tuple_162991)
    # Getting the type of 'a' (line 16)
    a_162999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 11), 'a')
    
    # Call to sum(...): (line 16)
    # Processing the call keyword arguments (line 16)
    int_163004 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 30), 'int')
    keyword_163005 = int_163004
    kwargs_163006 = {'axis': keyword_163005}
    # Getting the type of 'x' (line 16)
    x_163000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 16), 'x', False)
    # Getting the type of 'b' (line 16)
    b_163001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 18), 'b', False)
    # Applying the binary operator '*' (line 16)
    result_mul_163002 = python_operator(stypy.reporting.localization.Localization(__file__, 16, 16), '*', x_163000, b_163001)
    
    # Obtaining the member 'sum' of a type (line 16)
    sum_163003 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 16), result_mul_163002, 'sum')
    # Calling sum(args, kwargs) (line 16)
    sum_call_result_163007 = invoke(stypy.reporting.localization.Localization(__file__, 16, 16), sum_163003, *[], **kwargs_163006)
    
    # Applying the binary operator '+' (line 16)
    result_add_163008 = python_operator(stypy.reporting.localization.Localization(__file__, 16, 11), '+', a_162999, sum_call_result_163007)
    
    # Assigning a type to the variable 'stypy_return_type' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'stypy_return_type', result_add_163008)
    
    # ################# End of '_lin_fcn(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_lin_fcn' in the type store
    # Getting the type of 'stypy_return_type' (line 12)
    stypy_return_type_163009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_163009)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_lin_fcn'
    return stypy_return_type_163009

# Assigning a type to the variable '_lin_fcn' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), '_lin_fcn', _lin_fcn)

@norecursion
def _lin_fjb(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_lin_fjb'
    module_type_store = module_type_store.open_function_context('_lin_fjb', 19, 0, False)
    
    # Passed parameters checking function
    _lin_fjb.stypy_localization = localization
    _lin_fjb.stypy_type_of_self = None
    _lin_fjb.stypy_type_store = module_type_store
    _lin_fjb.stypy_function_name = '_lin_fjb'
    _lin_fjb.stypy_param_names_list = ['B', 'x']
    _lin_fjb.stypy_varargs_param_name = None
    _lin_fjb.stypy_kwargs_param_name = None
    _lin_fjb.stypy_call_defaults = defaults
    _lin_fjb.stypy_call_varargs = varargs
    _lin_fjb.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_lin_fjb', ['B', 'x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_lin_fjb', localization, ['B', 'x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_lin_fjb(...)' code ##################

    
    # Assigning a Call to a Name (line 20):
    
    # Assigning a Call to a Name (line 20):
    
    # Call to ones(...): (line 20)
    # Processing the call arguments (line 20)
    
    # Obtaining the type of the subscript
    int_163012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 24), 'int')
    # Getting the type of 'x' (line 20)
    x_163013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 16), 'x', False)
    # Obtaining the member 'shape' of a type (line 20)
    shape_163014 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 16), x_163013, 'shape')
    # Obtaining the member '__getitem__' of a type (line 20)
    getitem___163015 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 16), shape_163014, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 20)
    subscript_call_result_163016 = invoke(stypy.reporting.localization.Localization(__file__, 20, 16), getitem___163015, int_163012)
    
    # Getting the type of 'float' (line 20)
    float_163017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 29), 'float', False)
    # Processing the call keyword arguments (line 20)
    kwargs_163018 = {}
    # Getting the type of 'np' (line 20)
    np_163010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'np', False)
    # Obtaining the member 'ones' of a type (line 20)
    ones_163011 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 8), np_163010, 'ones')
    # Calling ones(args, kwargs) (line 20)
    ones_call_result_163019 = invoke(stypy.reporting.localization.Localization(__file__, 20, 8), ones_163011, *[subscript_call_result_163016, float_163017], **kwargs_163018)
    
    # Assigning a type to the variable 'a' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'a', ones_call_result_163019)
    
    # Assigning a Call to a Name (line 21):
    
    # Assigning a Call to a Name (line 21):
    
    # Call to concatenate(...): (line 21)
    # Processing the call arguments (line 21)
    
    # Obtaining an instance of the builtin type 'tuple' (line 21)
    tuple_163022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 26), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 21)
    # Adding element type (line 21)
    # Getting the type of 'a' (line 21)
    a_163023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 26), 'a', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 26), tuple_163022, a_163023)
    # Adding element type (line 21)
    
    # Call to ravel(...): (line 21)
    # Processing the call keyword arguments (line 21)
    kwargs_163026 = {}
    # Getting the type of 'x' (line 21)
    x_163024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 29), 'x', False)
    # Obtaining the member 'ravel' of a type (line 21)
    ravel_163025 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 29), x_163024, 'ravel')
    # Calling ravel(args, kwargs) (line 21)
    ravel_call_result_163027 = invoke(stypy.reporting.localization.Localization(__file__, 21, 29), ravel_163025, *[], **kwargs_163026)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 26), tuple_163022, ravel_call_result_163027)
    
    # Processing the call keyword arguments (line 21)
    kwargs_163028 = {}
    # Getting the type of 'np' (line 21)
    np_163020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 10), 'np', False)
    # Obtaining the member 'concatenate' of a type (line 21)
    concatenate_163021 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 10), np_163020, 'concatenate')
    # Calling concatenate(args, kwargs) (line 21)
    concatenate_call_result_163029 = invoke(stypy.reporting.localization.Localization(__file__, 21, 10), concatenate_163021, *[tuple_163022], **kwargs_163028)
    
    # Assigning a type to the variable 'res' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'res', concatenate_call_result_163029)
    
    # Assigning a Tuple to a Attribute (line 22):
    
    # Assigning a Tuple to a Attribute (line 22):
    
    # Obtaining an instance of the builtin type 'tuple' (line 22)
    tuple_163030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 17), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 22)
    # Adding element type (line 22)
    
    # Obtaining the type of the subscript
    int_163031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 25), 'int')
    # Getting the type of 'B' (line 22)
    B_163032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 17), 'B')
    # Obtaining the member 'shape' of a type (line 22)
    shape_163033 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 17), B_163032, 'shape')
    # Obtaining the member '__getitem__' of a type (line 22)
    getitem___163034 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 17), shape_163033, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 22)
    subscript_call_result_163035 = invoke(stypy.reporting.localization.Localization(__file__, 22, 17), getitem___163034, int_163031)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 17), tuple_163030, subscript_call_result_163035)
    # Adding element type (line 22)
    
    # Obtaining the type of the subscript
    int_163036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 38), 'int')
    # Getting the type of 'x' (line 22)
    x_163037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 30), 'x')
    # Obtaining the member 'shape' of a type (line 22)
    shape_163038 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 30), x_163037, 'shape')
    # Obtaining the member '__getitem__' of a type (line 22)
    getitem___163039 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 30), shape_163038, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 22)
    subscript_call_result_163040 = invoke(stypy.reporting.localization.Localization(__file__, 22, 30), getitem___163039, int_163036)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 17), tuple_163030, subscript_call_result_163040)
    
    # Getting the type of 'res' (line 22)
    res_163041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'res')
    # Setting the type of the member 'shape' of a type (line 22)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 4), res_163041, 'shape', tuple_163030)
    # Getting the type of 'res' (line 23)
    res_163042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 11), 'res')
    # Assigning a type to the variable 'stypy_return_type' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'stypy_return_type', res_163042)
    
    # ################# End of '_lin_fjb(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_lin_fjb' in the type store
    # Getting the type of 'stypy_return_type' (line 19)
    stypy_return_type_163043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_163043)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_lin_fjb'
    return stypy_return_type_163043

# Assigning a type to the variable '_lin_fjb' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), '_lin_fjb', _lin_fjb)

@norecursion
def _lin_fjd(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_lin_fjd'
    module_type_store = module_type_store.open_function_context('_lin_fjd', 26, 0, False)
    
    # Passed parameters checking function
    _lin_fjd.stypy_localization = localization
    _lin_fjd.stypy_type_of_self = None
    _lin_fjd.stypy_type_store = module_type_store
    _lin_fjd.stypy_function_name = '_lin_fjd'
    _lin_fjd.stypy_param_names_list = ['B', 'x']
    _lin_fjd.stypy_varargs_param_name = None
    _lin_fjd.stypy_kwargs_param_name = None
    _lin_fjd.stypy_call_defaults = defaults
    _lin_fjd.stypy_call_varargs = varargs
    _lin_fjd.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_lin_fjd', ['B', 'x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_lin_fjd', localization, ['B', 'x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_lin_fjd(...)' code ##################

    
    # Assigning a Subscript to a Name (line 27):
    
    # Assigning a Subscript to a Name (line 27):
    
    # Obtaining the type of the subscript
    int_163044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 10), 'int')
    slice_163045 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 27, 8), int_163044, None, None)
    # Getting the type of 'B' (line 27)
    B_163046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'B')
    # Obtaining the member '__getitem__' of a type (line 27)
    getitem___163047 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 8), B_163046, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 27)
    subscript_call_result_163048 = invoke(stypy.reporting.localization.Localization(__file__, 27, 8), getitem___163047, slice_163045)
    
    # Assigning a type to the variable 'b' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'b', subscript_call_result_163048)
    
    # Assigning a Call to a Name (line 28):
    
    # Assigning a Call to a Name (line 28):
    
    # Call to repeat(...): (line 28)
    # Processing the call arguments (line 28)
    # Getting the type of 'b' (line 28)
    b_163051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 18), 'b', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 28)
    tuple_163052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 22), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 28)
    # Adding element type (line 28)
    
    # Obtaining the type of the subscript
    int_163053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 30), 'int')
    # Getting the type of 'x' (line 28)
    x_163054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 22), 'x', False)
    # Obtaining the member 'shape' of a type (line 28)
    shape_163055 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 22), x_163054, 'shape')
    # Obtaining the member '__getitem__' of a type (line 28)
    getitem___163056 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 22), shape_163055, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 28)
    subscript_call_result_163057 = invoke(stypy.reporting.localization.Localization(__file__, 28, 22), getitem___163056, int_163053)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 22), tuple_163052, subscript_call_result_163057)
    
    
    # Obtaining the type of the subscript
    int_163058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 44), 'int')
    # Getting the type of 'b' (line 28)
    b_163059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 36), 'b', False)
    # Obtaining the member 'shape' of a type (line 28)
    shape_163060 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 36), b_163059, 'shape')
    # Obtaining the member '__getitem__' of a type (line 28)
    getitem___163061 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 36), shape_163060, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 28)
    subscript_call_result_163062 = invoke(stypy.reporting.localization.Localization(__file__, 28, 36), getitem___163061, int_163058)
    
    # Applying the binary operator '*' (line 28)
    result_mul_163063 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 21), '*', tuple_163052, subscript_call_result_163062)
    
    # Processing the call keyword arguments (line 28)
    int_163064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 53), 'int')
    keyword_163065 = int_163064
    kwargs_163066 = {'axis': keyword_163065}
    # Getting the type of 'np' (line 28)
    np_163049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'np', False)
    # Obtaining the member 'repeat' of a type (line 28)
    repeat_163050 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 8), np_163049, 'repeat')
    # Calling repeat(args, kwargs) (line 28)
    repeat_call_result_163067 = invoke(stypy.reporting.localization.Localization(__file__, 28, 8), repeat_163050, *[b_163051, result_mul_163063], **kwargs_163066)
    
    # Assigning a type to the variable 'b' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'b', repeat_call_result_163067)
    
    # Assigning a Attribute to a Attribute (line 29):
    
    # Assigning a Attribute to a Attribute (line 29):
    # Getting the type of 'x' (line 29)
    x_163068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 14), 'x')
    # Obtaining the member 'shape' of a type (line 29)
    shape_163069 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 14), x_163068, 'shape')
    # Getting the type of 'b' (line 29)
    b_163070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'b')
    # Setting the type of the member 'shape' of a type (line 29)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 4), b_163070, 'shape', shape_163069)
    # Getting the type of 'b' (line 30)
    b_163071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 11), 'b')
    # Assigning a type to the variable 'stypy_return_type' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'stypy_return_type', b_163071)
    
    # ################# End of '_lin_fjd(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_lin_fjd' in the type store
    # Getting the type of 'stypy_return_type' (line 26)
    stypy_return_type_163072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_163072)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_lin_fjd'
    return stypy_return_type_163072

# Assigning a type to the variable '_lin_fjd' (line 26)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), '_lin_fjd', _lin_fjd)

@norecursion
def _lin_est(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_lin_est'
    module_type_store = module_type_store.open_function_context('_lin_est', 33, 0, False)
    
    # Passed parameters checking function
    _lin_est.stypy_localization = localization
    _lin_est.stypy_type_of_self = None
    _lin_est.stypy_type_store = module_type_store
    _lin_est.stypy_function_name = '_lin_est'
    _lin_est.stypy_param_names_list = ['data']
    _lin_est.stypy_varargs_param_name = None
    _lin_est.stypy_kwargs_param_name = None
    _lin_est.stypy_call_defaults = defaults
    _lin_est.stypy_call_varargs = varargs
    _lin_est.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_lin_est', ['data'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_lin_est', localization, ['data'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_lin_est(...)' code ##################

    
    
    
    # Call to len(...): (line 38)
    # Processing the call arguments (line 38)
    # Getting the type of 'data' (line 38)
    data_163074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 11), 'data', False)
    # Obtaining the member 'x' of a type (line 38)
    x_163075 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 11), data_163074, 'x')
    # Obtaining the member 'shape' of a type (line 38)
    shape_163076 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 11), x_163075, 'shape')
    # Processing the call keyword arguments (line 38)
    kwargs_163077 = {}
    # Getting the type of 'len' (line 38)
    len_163073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 7), 'len', False)
    # Calling len(args, kwargs) (line 38)
    len_call_result_163078 = invoke(stypy.reporting.localization.Localization(__file__, 38, 7), len_163073, *[shape_163076], **kwargs_163077)
    
    int_163079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 28), 'int')
    # Applying the binary operator '==' (line 38)
    result_eq_163080 = python_operator(stypy.reporting.localization.Localization(__file__, 38, 7), '==', len_call_result_163078, int_163079)
    
    # Testing the type of an if condition (line 38)
    if_condition_163081 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 38, 4), result_eq_163080)
    # Assigning a type to the variable 'if_condition_163081' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'if_condition_163081', if_condition_163081)
    # SSA begins for if statement (line 38)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 39):
    
    # Assigning a Subscript to a Name (line 39):
    
    # Obtaining the type of the subscript
    int_163082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 25), 'int')
    # Getting the type of 'data' (line 39)
    data_163083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 12), 'data')
    # Obtaining the member 'x' of a type (line 39)
    x_163084 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 12), data_163083, 'x')
    # Obtaining the member 'shape' of a type (line 39)
    shape_163085 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 12), x_163084, 'shape')
    # Obtaining the member '__getitem__' of a type (line 39)
    getitem___163086 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 12), shape_163085, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 39)
    subscript_call_result_163087 = invoke(stypy.reporting.localization.Localization(__file__, 39, 12), getitem___163086, int_163082)
    
    # Assigning a type to the variable 'm' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'm', subscript_call_result_163087)
    # SSA branch for the else part of an if statement (line 38)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Num to a Name (line 41):
    
    # Assigning a Num to a Name (line 41):
    int_163088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 12), 'int')
    # Assigning a type to the variable 'm' (line 41)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'm', int_163088)
    # SSA join for if statement (line 38)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to ones(...): (line 43)
    # Processing the call arguments (line 43)
    
    # Obtaining an instance of the builtin type 'tuple' (line 43)
    tuple_163091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 20), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 43)
    # Adding element type (line 43)
    # Getting the type of 'm' (line 43)
    m_163092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 20), 'm', False)
    int_163093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 24), 'int')
    # Applying the binary operator '+' (line 43)
    result_add_163094 = python_operator(stypy.reporting.localization.Localization(__file__, 43, 20), '+', m_163092, int_163093)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 20), tuple_163091, result_add_163094)
    
    # Getting the type of 'float' (line 43)
    float_163095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 29), 'float', False)
    # Processing the call keyword arguments (line 43)
    kwargs_163096 = {}
    # Getting the type of 'np' (line 43)
    np_163089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 11), 'np', False)
    # Obtaining the member 'ones' of a type (line 43)
    ones_163090 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 11), np_163089, 'ones')
    # Calling ones(args, kwargs) (line 43)
    ones_call_result_163097 = invoke(stypy.reporting.localization.Localization(__file__, 43, 11), ones_163090, *[tuple_163091, float_163095], **kwargs_163096)
    
    # Assigning a type to the variable 'stypy_return_type' (line 43)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'stypy_return_type', ones_call_result_163097)
    
    # ################# End of '_lin_est(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_lin_est' in the type store
    # Getting the type of 'stypy_return_type' (line 33)
    stypy_return_type_163098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_163098)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_lin_est'
    return stypy_return_type_163098

# Assigning a type to the variable '_lin_est' (line 33)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 0), '_lin_est', _lin_est)

@norecursion
def _poly_fcn(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_poly_fcn'
    module_type_store = module_type_store.open_function_context('_poly_fcn', 46, 0, False)
    
    # Passed parameters checking function
    _poly_fcn.stypy_localization = localization
    _poly_fcn.stypy_type_of_self = None
    _poly_fcn.stypy_type_store = module_type_store
    _poly_fcn.stypy_function_name = '_poly_fcn'
    _poly_fcn.stypy_param_names_list = ['B', 'x', 'powers']
    _poly_fcn.stypy_varargs_param_name = None
    _poly_fcn.stypy_kwargs_param_name = None
    _poly_fcn.stypy_call_defaults = defaults
    _poly_fcn.stypy_call_varargs = varargs
    _poly_fcn.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_poly_fcn', ['B', 'x', 'powers'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_poly_fcn', localization, ['B', 'x', 'powers'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_poly_fcn(...)' code ##################

    
    # Assigning a Tuple to a Tuple (line 47):
    
    # Assigning a Subscript to a Name (line 47):
    
    # Obtaining the type of the subscript
    int_163099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 13), 'int')
    # Getting the type of 'B' (line 47)
    B_163100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 11), 'B')
    # Obtaining the member '__getitem__' of a type (line 47)
    getitem___163101 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 11), B_163100, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 47)
    subscript_call_result_163102 = invoke(stypy.reporting.localization.Localization(__file__, 47, 11), getitem___163101, int_163099)
    
    # Assigning a type to the variable 'tuple_assignment_162966' (line 47)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'tuple_assignment_162966', subscript_call_result_163102)
    
    # Assigning a Subscript to a Name (line 47):
    
    # Obtaining the type of the subscript
    int_163103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 19), 'int')
    slice_163104 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 47, 17), int_163103, None, None)
    # Getting the type of 'B' (line 47)
    B_163105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 17), 'B')
    # Obtaining the member '__getitem__' of a type (line 47)
    getitem___163106 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 17), B_163105, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 47)
    subscript_call_result_163107 = invoke(stypy.reporting.localization.Localization(__file__, 47, 17), getitem___163106, slice_163104)
    
    # Assigning a type to the variable 'tuple_assignment_162967' (line 47)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'tuple_assignment_162967', subscript_call_result_163107)
    
    # Assigning a Name to a Name (line 47):
    # Getting the type of 'tuple_assignment_162966' (line 47)
    tuple_assignment_162966_163108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'tuple_assignment_162966')
    # Assigning a type to the variable 'a' (line 47)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'a', tuple_assignment_162966_163108)
    
    # Assigning a Name to a Name (line 47):
    # Getting the type of 'tuple_assignment_162967' (line 47)
    tuple_assignment_162967_163109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'tuple_assignment_162967')
    # Assigning a type to the variable 'b' (line 47)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 7), 'b', tuple_assignment_162967_163109)
    
    # Assigning a Tuple to a Attribute (line 48):
    
    # Assigning a Tuple to a Attribute (line 48):
    
    # Obtaining an instance of the builtin type 'tuple' (line 48)
    tuple_163110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 48)
    # Adding element type (line 48)
    
    # Obtaining the type of the subscript
    int_163111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 23), 'int')
    # Getting the type of 'b' (line 48)
    b_163112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 15), 'b')
    # Obtaining the member 'shape' of a type (line 48)
    shape_163113 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 15), b_163112, 'shape')
    # Obtaining the member '__getitem__' of a type (line 48)
    getitem___163114 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 15), shape_163113, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 48)
    subscript_call_result_163115 = invoke(stypy.reporting.localization.Localization(__file__, 48, 15), getitem___163114, int_163111)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 15), tuple_163110, subscript_call_result_163115)
    # Adding element type (line 48)
    int_163116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 15), tuple_163110, int_163116)
    
    # Getting the type of 'b' (line 48)
    b_163117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'b')
    # Setting the type of the member 'shape' of a type (line 48)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 4), b_163117, 'shape', tuple_163110)
    # Getting the type of 'a' (line 50)
    a_163118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 11), 'a')
    
    # Call to sum(...): (line 50)
    # Processing the call arguments (line 50)
    # Getting the type of 'b' (line 50)
    b_163121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 22), 'b', False)
    
    # Call to power(...): (line 50)
    # Processing the call arguments (line 50)
    # Getting the type of 'x' (line 50)
    x_163124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 35), 'x', False)
    # Getting the type of 'powers' (line 50)
    powers_163125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 38), 'powers', False)
    # Processing the call keyword arguments (line 50)
    kwargs_163126 = {}
    # Getting the type of 'np' (line 50)
    np_163122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 26), 'np', False)
    # Obtaining the member 'power' of a type (line 50)
    power_163123 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 26), np_163122, 'power')
    # Calling power(args, kwargs) (line 50)
    power_call_result_163127 = invoke(stypy.reporting.localization.Localization(__file__, 50, 26), power_163123, *[x_163124, powers_163125], **kwargs_163126)
    
    # Applying the binary operator '*' (line 50)
    result_mul_163128 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 22), '*', b_163121, power_call_result_163127)
    
    # Processing the call keyword arguments (line 50)
    int_163129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 52), 'int')
    keyword_163130 = int_163129
    kwargs_163131 = {'axis': keyword_163130}
    # Getting the type of 'np' (line 50)
    np_163119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 15), 'np', False)
    # Obtaining the member 'sum' of a type (line 50)
    sum_163120 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 15), np_163119, 'sum')
    # Calling sum(args, kwargs) (line 50)
    sum_call_result_163132 = invoke(stypy.reporting.localization.Localization(__file__, 50, 15), sum_163120, *[result_mul_163128], **kwargs_163131)
    
    # Applying the binary operator '+' (line 50)
    result_add_163133 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 11), '+', a_163118, sum_call_result_163132)
    
    # Assigning a type to the variable 'stypy_return_type' (line 50)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'stypy_return_type', result_add_163133)
    
    # ################# End of '_poly_fcn(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_poly_fcn' in the type store
    # Getting the type of 'stypy_return_type' (line 46)
    stypy_return_type_163134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_163134)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_poly_fcn'
    return stypy_return_type_163134

# Assigning a type to the variable '_poly_fcn' (line 46)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 0), '_poly_fcn', _poly_fcn)

@norecursion
def _poly_fjacb(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_poly_fjacb'
    module_type_store = module_type_store.open_function_context('_poly_fjacb', 53, 0, False)
    
    # Passed parameters checking function
    _poly_fjacb.stypy_localization = localization
    _poly_fjacb.stypy_type_of_self = None
    _poly_fjacb.stypy_type_store = module_type_store
    _poly_fjacb.stypy_function_name = '_poly_fjacb'
    _poly_fjacb.stypy_param_names_list = ['B', 'x', 'powers']
    _poly_fjacb.stypy_varargs_param_name = None
    _poly_fjacb.stypy_kwargs_param_name = None
    _poly_fjacb.stypy_call_defaults = defaults
    _poly_fjacb.stypy_call_varargs = varargs
    _poly_fjacb.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_poly_fjacb', ['B', 'x', 'powers'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_poly_fjacb', localization, ['B', 'x', 'powers'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_poly_fjacb(...)' code ##################

    
    # Assigning a Call to a Name (line 54):
    
    # Assigning a Call to a Name (line 54):
    
    # Call to concatenate(...): (line 54)
    # Processing the call arguments (line 54)
    
    # Obtaining an instance of the builtin type 'tuple' (line 54)
    tuple_163137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 26), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 54)
    # Adding element type (line 54)
    
    # Call to ones(...): (line 54)
    # Processing the call arguments (line 54)
    
    # Obtaining the type of the subscript
    int_163140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 42), 'int')
    # Getting the type of 'x' (line 54)
    x_163141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 34), 'x', False)
    # Obtaining the member 'shape' of a type (line 54)
    shape_163142 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 34), x_163141, 'shape')
    # Obtaining the member '__getitem__' of a type (line 54)
    getitem___163143 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 34), shape_163142, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 54)
    subscript_call_result_163144 = invoke(stypy.reporting.localization.Localization(__file__, 54, 34), getitem___163143, int_163140)
    
    # Getting the type of 'float' (line 54)
    float_163145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 47), 'float', False)
    # Processing the call keyword arguments (line 54)
    kwargs_163146 = {}
    # Getting the type of 'np' (line 54)
    np_163138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 26), 'np', False)
    # Obtaining the member 'ones' of a type (line 54)
    ones_163139 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 26), np_163138, 'ones')
    # Calling ones(args, kwargs) (line 54)
    ones_call_result_163147 = invoke(stypy.reporting.localization.Localization(__file__, 54, 26), ones_163139, *[subscript_call_result_163144, float_163145], **kwargs_163146)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 26), tuple_163137, ones_call_result_163147)
    # Adding element type (line 54)
    
    # Call to power(...): (line 54)
    # Processing the call arguments (line 54)
    # Getting the type of 'x' (line 54)
    x_163150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 64), 'x', False)
    # Getting the type of 'powers' (line 55)
    powers_163151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'powers', False)
    # Processing the call keyword arguments (line 54)
    kwargs_163152 = {}
    # Getting the type of 'np' (line 54)
    np_163148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 55), 'np', False)
    # Obtaining the member 'power' of a type (line 54)
    power_163149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 55), np_163148, 'power')
    # Calling power(args, kwargs) (line 54)
    power_call_result_163153 = invoke(stypy.reporting.localization.Localization(__file__, 54, 55), power_163149, *[x_163150, powers_163151], **kwargs_163152)
    
    # Obtaining the member 'flat' of a type (line 54)
    flat_163154 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 55), power_call_result_163153, 'flat')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 26), tuple_163137, flat_163154)
    
    # Processing the call keyword arguments (line 54)
    kwargs_163155 = {}
    # Getting the type of 'np' (line 54)
    np_163135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 10), 'np', False)
    # Obtaining the member 'concatenate' of a type (line 54)
    concatenate_163136 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 10), np_163135, 'concatenate')
    # Calling concatenate(args, kwargs) (line 54)
    concatenate_call_result_163156 = invoke(stypy.reporting.localization.Localization(__file__, 54, 10), concatenate_163136, *[tuple_163137], **kwargs_163155)
    
    # Assigning a type to the variable 'res' (line 54)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'res', concatenate_call_result_163156)
    
    # Assigning a Tuple to a Attribute (line 56):
    
    # Assigning a Tuple to a Attribute (line 56):
    
    # Obtaining an instance of the builtin type 'tuple' (line 56)
    tuple_163157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 17), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 56)
    # Adding element type (line 56)
    
    # Obtaining the type of the subscript
    int_163158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 25), 'int')
    # Getting the type of 'B' (line 56)
    B_163159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 17), 'B')
    # Obtaining the member 'shape' of a type (line 56)
    shape_163160 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 17), B_163159, 'shape')
    # Obtaining the member '__getitem__' of a type (line 56)
    getitem___163161 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 17), shape_163160, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 56)
    subscript_call_result_163162 = invoke(stypy.reporting.localization.Localization(__file__, 56, 17), getitem___163161, int_163158)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 17), tuple_163157, subscript_call_result_163162)
    # Adding element type (line 56)
    
    # Obtaining the type of the subscript
    int_163163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 38), 'int')
    # Getting the type of 'x' (line 56)
    x_163164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 30), 'x')
    # Obtaining the member 'shape' of a type (line 56)
    shape_163165 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 30), x_163164, 'shape')
    # Obtaining the member '__getitem__' of a type (line 56)
    getitem___163166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 30), shape_163165, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 56)
    subscript_call_result_163167 = invoke(stypy.reporting.localization.Localization(__file__, 56, 30), getitem___163166, int_163163)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 17), tuple_163157, subscript_call_result_163167)
    
    # Getting the type of 'res' (line 56)
    res_163168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'res')
    # Setting the type of the member 'shape' of a type (line 56)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 4), res_163168, 'shape', tuple_163157)
    # Getting the type of 'res' (line 57)
    res_163169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 11), 'res')
    # Assigning a type to the variable 'stypy_return_type' (line 57)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'stypy_return_type', res_163169)
    
    # ################# End of '_poly_fjacb(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_poly_fjacb' in the type store
    # Getting the type of 'stypy_return_type' (line 53)
    stypy_return_type_163170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_163170)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_poly_fjacb'
    return stypy_return_type_163170

# Assigning a type to the variable '_poly_fjacb' (line 53)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 0), '_poly_fjacb', _poly_fjacb)

@norecursion
def _poly_fjacd(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_poly_fjacd'
    module_type_store = module_type_store.open_function_context('_poly_fjacd', 60, 0, False)
    
    # Passed parameters checking function
    _poly_fjacd.stypy_localization = localization
    _poly_fjacd.stypy_type_of_self = None
    _poly_fjacd.stypy_type_store = module_type_store
    _poly_fjacd.stypy_function_name = '_poly_fjacd'
    _poly_fjacd.stypy_param_names_list = ['B', 'x', 'powers']
    _poly_fjacd.stypy_varargs_param_name = None
    _poly_fjacd.stypy_kwargs_param_name = None
    _poly_fjacd.stypy_call_defaults = defaults
    _poly_fjacd.stypy_call_varargs = varargs
    _poly_fjacd.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_poly_fjacd', ['B', 'x', 'powers'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_poly_fjacd', localization, ['B', 'x', 'powers'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_poly_fjacd(...)' code ##################

    
    # Assigning a Subscript to a Name (line 61):
    
    # Assigning a Subscript to a Name (line 61):
    
    # Obtaining the type of the subscript
    int_163171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 10), 'int')
    slice_163172 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 61, 8), int_163171, None, None)
    # Getting the type of 'B' (line 61)
    B_163173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'B')
    # Obtaining the member '__getitem__' of a type (line 61)
    getitem___163174 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 8), B_163173, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 61)
    subscript_call_result_163175 = invoke(stypy.reporting.localization.Localization(__file__, 61, 8), getitem___163174, slice_163172)
    
    # Assigning a type to the variable 'b' (line 61)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), 'b', subscript_call_result_163175)
    
    # Assigning a Tuple to a Attribute (line 62):
    
    # Assigning a Tuple to a Attribute (line 62):
    
    # Obtaining an instance of the builtin type 'tuple' (line 62)
    tuple_163176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 62)
    # Adding element type (line 62)
    
    # Obtaining the type of the subscript
    int_163177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 23), 'int')
    # Getting the type of 'b' (line 62)
    b_163178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 15), 'b')
    # Obtaining the member 'shape' of a type (line 62)
    shape_163179 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 15), b_163178, 'shape')
    # Obtaining the member '__getitem__' of a type (line 62)
    getitem___163180 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 15), shape_163179, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 62)
    subscript_call_result_163181 = invoke(stypy.reporting.localization.Localization(__file__, 62, 15), getitem___163180, int_163177)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 15), tuple_163176, subscript_call_result_163181)
    # Adding element type (line 62)
    int_163182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 15), tuple_163176, int_163182)
    
    # Getting the type of 'b' (line 62)
    b_163183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), 'b')
    # Setting the type of the member 'shape' of a type (line 62)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 4), b_163183, 'shape', tuple_163176)
    
    # Assigning a BinOp to a Name (line 64):
    
    # Assigning a BinOp to a Name (line 64):
    # Getting the type of 'b' (line 64)
    b_163184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'b')
    # Getting the type of 'powers' (line 64)
    powers_163185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 12), 'powers')
    # Applying the binary operator '*' (line 64)
    result_mul_163186 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 8), '*', b_163184, powers_163185)
    
    # Assigning a type to the variable 'b' (line 64)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'b', result_mul_163186)
    
    # Call to sum(...): (line 66)
    # Processing the call arguments (line 66)
    # Getting the type of 'b' (line 66)
    b_163189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 18), 'b', False)
    
    # Call to power(...): (line 66)
    # Processing the call arguments (line 66)
    # Getting the type of 'x' (line 66)
    x_163192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 31), 'x', False)
    # Getting the type of 'powers' (line 66)
    powers_163193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 34), 'powers', False)
    int_163194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 41), 'int')
    # Applying the binary operator '-' (line 66)
    result_sub_163195 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 34), '-', powers_163193, int_163194)
    
    # Processing the call keyword arguments (line 66)
    kwargs_163196 = {}
    # Getting the type of 'np' (line 66)
    np_163190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 22), 'np', False)
    # Obtaining the member 'power' of a type (line 66)
    power_163191 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 22), np_163190, 'power')
    # Calling power(args, kwargs) (line 66)
    power_call_result_163197 = invoke(stypy.reporting.localization.Localization(__file__, 66, 22), power_163191, *[x_163192, result_sub_163195], **kwargs_163196)
    
    # Applying the binary operator '*' (line 66)
    result_mul_163198 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 18), '*', b_163189, power_call_result_163197)
    
    # Processing the call keyword arguments (line 66)
    int_163199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 49), 'int')
    keyword_163200 = int_163199
    kwargs_163201 = {'axis': keyword_163200}
    # Getting the type of 'np' (line 66)
    np_163187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 11), 'np', False)
    # Obtaining the member 'sum' of a type (line 66)
    sum_163188 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 11), np_163187, 'sum')
    # Calling sum(args, kwargs) (line 66)
    sum_call_result_163202 = invoke(stypy.reporting.localization.Localization(__file__, 66, 11), sum_163188, *[result_mul_163198], **kwargs_163201)
    
    # Assigning a type to the variable 'stypy_return_type' (line 66)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'stypy_return_type', sum_call_result_163202)
    
    # ################# End of '_poly_fjacd(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_poly_fjacd' in the type store
    # Getting the type of 'stypy_return_type' (line 60)
    stypy_return_type_163203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_163203)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_poly_fjacd'
    return stypy_return_type_163203

# Assigning a type to the variable '_poly_fjacd' (line 60)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 0), '_poly_fjacd', _poly_fjacd)

@norecursion
def _exp_fcn(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_exp_fcn'
    module_type_store = module_type_store.open_function_context('_exp_fcn', 69, 0, False)
    
    # Passed parameters checking function
    _exp_fcn.stypy_localization = localization
    _exp_fcn.stypy_type_of_self = None
    _exp_fcn.stypy_type_store = module_type_store
    _exp_fcn.stypy_function_name = '_exp_fcn'
    _exp_fcn.stypy_param_names_list = ['B', 'x']
    _exp_fcn.stypy_varargs_param_name = None
    _exp_fcn.stypy_kwargs_param_name = None
    _exp_fcn.stypy_call_defaults = defaults
    _exp_fcn.stypy_call_varargs = varargs
    _exp_fcn.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_exp_fcn', ['B', 'x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_exp_fcn', localization, ['B', 'x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_exp_fcn(...)' code ##################

    
    # Obtaining the type of the subscript
    int_163204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 13), 'int')
    # Getting the type of 'B' (line 70)
    B_163205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 11), 'B')
    # Obtaining the member '__getitem__' of a type (line 70)
    getitem___163206 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 11), B_163205, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 70)
    subscript_call_result_163207 = invoke(stypy.reporting.localization.Localization(__file__, 70, 11), getitem___163206, int_163204)
    
    
    # Call to exp(...): (line 70)
    # Processing the call arguments (line 70)
    
    # Obtaining the type of the subscript
    int_163210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 27), 'int')
    # Getting the type of 'B' (line 70)
    B_163211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 25), 'B', False)
    # Obtaining the member '__getitem__' of a type (line 70)
    getitem___163212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 25), B_163211, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 70)
    subscript_call_result_163213 = invoke(stypy.reporting.localization.Localization(__file__, 70, 25), getitem___163212, int_163210)
    
    # Getting the type of 'x' (line 70)
    x_163214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 32), 'x', False)
    # Applying the binary operator '*' (line 70)
    result_mul_163215 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 25), '*', subscript_call_result_163213, x_163214)
    
    # Processing the call keyword arguments (line 70)
    kwargs_163216 = {}
    # Getting the type of 'np' (line 70)
    np_163208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 18), 'np', False)
    # Obtaining the member 'exp' of a type (line 70)
    exp_163209 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 18), np_163208, 'exp')
    # Calling exp(args, kwargs) (line 70)
    exp_call_result_163217 = invoke(stypy.reporting.localization.Localization(__file__, 70, 18), exp_163209, *[result_mul_163215], **kwargs_163216)
    
    # Applying the binary operator '+' (line 70)
    result_add_163218 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 11), '+', subscript_call_result_163207, exp_call_result_163217)
    
    # Assigning a type to the variable 'stypy_return_type' (line 70)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'stypy_return_type', result_add_163218)
    
    # ################# End of '_exp_fcn(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_exp_fcn' in the type store
    # Getting the type of 'stypy_return_type' (line 69)
    stypy_return_type_163219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_163219)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_exp_fcn'
    return stypy_return_type_163219

# Assigning a type to the variable '_exp_fcn' (line 69)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 0), '_exp_fcn', _exp_fcn)

@norecursion
def _exp_fjd(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_exp_fjd'
    module_type_store = module_type_store.open_function_context('_exp_fjd', 73, 0, False)
    
    # Passed parameters checking function
    _exp_fjd.stypy_localization = localization
    _exp_fjd.stypy_type_of_self = None
    _exp_fjd.stypy_type_store = module_type_store
    _exp_fjd.stypy_function_name = '_exp_fjd'
    _exp_fjd.stypy_param_names_list = ['B', 'x']
    _exp_fjd.stypy_varargs_param_name = None
    _exp_fjd.stypy_kwargs_param_name = None
    _exp_fjd.stypy_call_defaults = defaults
    _exp_fjd.stypy_call_varargs = varargs
    _exp_fjd.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_exp_fjd', ['B', 'x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_exp_fjd', localization, ['B', 'x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_exp_fjd(...)' code ##################

    
    # Obtaining the type of the subscript
    int_163220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 13), 'int')
    # Getting the type of 'B' (line 74)
    B_163221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 11), 'B')
    # Obtaining the member '__getitem__' of a type (line 74)
    getitem___163222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 11), B_163221, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 74)
    subscript_call_result_163223 = invoke(stypy.reporting.localization.Localization(__file__, 74, 11), getitem___163222, int_163220)
    
    
    # Call to exp(...): (line 74)
    # Processing the call arguments (line 74)
    
    # Obtaining the type of the subscript
    int_163226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 27), 'int')
    # Getting the type of 'B' (line 74)
    B_163227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 25), 'B', False)
    # Obtaining the member '__getitem__' of a type (line 74)
    getitem___163228 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 25), B_163227, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 74)
    subscript_call_result_163229 = invoke(stypy.reporting.localization.Localization(__file__, 74, 25), getitem___163228, int_163226)
    
    # Getting the type of 'x' (line 74)
    x_163230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 32), 'x', False)
    # Applying the binary operator '*' (line 74)
    result_mul_163231 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 25), '*', subscript_call_result_163229, x_163230)
    
    # Processing the call keyword arguments (line 74)
    kwargs_163232 = {}
    # Getting the type of 'np' (line 74)
    np_163224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 18), 'np', False)
    # Obtaining the member 'exp' of a type (line 74)
    exp_163225 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 18), np_163224, 'exp')
    # Calling exp(args, kwargs) (line 74)
    exp_call_result_163233 = invoke(stypy.reporting.localization.Localization(__file__, 74, 18), exp_163225, *[result_mul_163231], **kwargs_163232)
    
    # Applying the binary operator '*' (line 74)
    result_mul_163234 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 11), '*', subscript_call_result_163223, exp_call_result_163233)
    
    # Assigning a type to the variable 'stypy_return_type' (line 74)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'stypy_return_type', result_mul_163234)
    
    # ################# End of '_exp_fjd(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_exp_fjd' in the type store
    # Getting the type of 'stypy_return_type' (line 73)
    stypy_return_type_163235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_163235)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_exp_fjd'
    return stypy_return_type_163235

# Assigning a type to the variable '_exp_fjd' (line 73)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 0), '_exp_fjd', _exp_fjd)

@norecursion
def _exp_fjb(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_exp_fjb'
    module_type_store = module_type_store.open_function_context('_exp_fjb', 77, 0, False)
    
    # Passed parameters checking function
    _exp_fjb.stypy_localization = localization
    _exp_fjb.stypy_type_of_self = None
    _exp_fjb.stypy_type_store = module_type_store
    _exp_fjb.stypy_function_name = '_exp_fjb'
    _exp_fjb.stypy_param_names_list = ['B', 'x']
    _exp_fjb.stypy_varargs_param_name = None
    _exp_fjb.stypy_kwargs_param_name = None
    _exp_fjb.stypy_call_defaults = defaults
    _exp_fjb.stypy_call_varargs = varargs
    _exp_fjb.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_exp_fjb', ['B', 'x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_exp_fjb', localization, ['B', 'x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_exp_fjb(...)' code ##################

    
    # Assigning a Call to a Name (line 78):
    
    # Assigning a Call to a Name (line 78):
    
    # Call to concatenate(...): (line 78)
    # Processing the call arguments (line 78)
    
    # Obtaining an instance of the builtin type 'tuple' (line 78)
    tuple_163238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 26), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 78)
    # Adding element type (line 78)
    
    # Call to ones(...): (line 78)
    # Processing the call arguments (line 78)
    
    # Obtaining the type of the subscript
    int_163241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 42), 'int')
    # Getting the type of 'x' (line 78)
    x_163242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 34), 'x', False)
    # Obtaining the member 'shape' of a type (line 78)
    shape_163243 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 34), x_163242, 'shape')
    # Obtaining the member '__getitem__' of a type (line 78)
    getitem___163244 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 34), shape_163243, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 78)
    subscript_call_result_163245 = invoke(stypy.reporting.localization.Localization(__file__, 78, 34), getitem___163244, int_163241)
    
    # Getting the type of 'float' (line 78)
    float_163246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 47), 'float', False)
    # Processing the call keyword arguments (line 78)
    kwargs_163247 = {}
    # Getting the type of 'np' (line 78)
    np_163239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 26), 'np', False)
    # Obtaining the member 'ones' of a type (line 78)
    ones_163240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 26), np_163239, 'ones')
    # Calling ones(args, kwargs) (line 78)
    ones_call_result_163248 = invoke(stypy.reporting.localization.Localization(__file__, 78, 26), ones_163240, *[subscript_call_result_163245, float_163246], **kwargs_163247)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 26), tuple_163238, ones_call_result_163248)
    # Adding element type (line 78)
    # Getting the type of 'x' (line 78)
    x_163249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 55), 'x', False)
    
    # Call to exp(...): (line 78)
    # Processing the call arguments (line 78)
    
    # Obtaining the type of the subscript
    int_163252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 68), 'int')
    # Getting the type of 'B' (line 78)
    B_163253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 66), 'B', False)
    # Obtaining the member '__getitem__' of a type (line 78)
    getitem___163254 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 66), B_163253, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 78)
    subscript_call_result_163255 = invoke(stypy.reporting.localization.Localization(__file__, 78, 66), getitem___163254, int_163252)
    
    # Getting the type of 'x' (line 78)
    x_163256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 73), 'x', False)
    # Applying the binary operator '*' (line 78)
    result_mul_163257 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 66), '*', subscript_call_result_163255, x_163256)
    
    # Processing the call keyword arguments (line 78)
    kwargs_163258 = {}
    # Getting the type of 'np' (line 78)
    np_163250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 59), 'np', False)
    # Obtaining the member 'exp' of a type (line 78)
    exp_163251 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 59), np_163250, 'exp')
    # Calling exp(args, kwargs) (line 78)
    exp_call_result_163259 = invoke(stypy.reporting.localization.Localization(__file__, 78, 59), exp_163251, *[result_mul_163257], **kwargs_163258)
    
    # Applying the binary operator '*' (line 78)
    result_mul_163260 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 55), '*', x_163249, exp_call_result_163259)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 26), tuple_163238, result_mul_163260)
    
    # Processing the call keyword arguments (line 78)
    kwargs_163261 = {}
    # Getting the type of 'np' (line 78)
    np_163236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 10), 'np', False)
    # Obtaining the member 'concatenate' of a type (line 78)
    concatenate_163237 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 10), np_163236, 'concatenate')
    # Calling concatenate(args, kwargs) (line 78)
    concatenate_call_result_163262 = invoke(stypy.reporting.localization.Localization(__file__, 78, 10), concatenate_163237, *[tuple_163238], **kwargs_163261)
    
    # Assigning a type to the variable 'res' (line 78)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'res', concatenate_call_result_163262)
    
    # Assigning a Tuple to a Attribute (line 79):
    
    # Assigning a Tuple to a Attribute (line 79):
    
    # Obtaining an instance of the builtin type 'tuple' (line 79)
    tuple_163263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 17), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 79)
    # Adding element type (line 79)
    int_163264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 17), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 17), tuple_163263, int_163264)
    # Adding element type (line 79)
    
    # Obtaining the type of the subscript
    int_163265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 28), 'int')
    # Getting the type of 'x' (line 79)
    x_163266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 20), 'x')
    # Obtaining the member 'shape' of a type (line 79)
    shape_163267 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 20), x_163266, 'shape')
    # Obtaining the member '__getitem__' of a type (line 79)
    getitem___163268 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 20), shape_163267, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 79)
    subscript_call_result_163269 = invoke(stypy.reporting.localization.Localization(__file__, 79, 20), getitem___163268, int_163265)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 17), tuple_163263, subscript_call_result_163269)
    
    # Getting the type of 'res' (line 79)
    res_163270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'res')
    # Setting the type of the member 'shape' of a type (line 79)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 4), res_163270, 'shape', tuple_163263)
    # Getting the type of 'res' (line 80)
    res_163271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 11), 'res')
    # Assigning a type to the variable 'stypy_return_type' (line 80)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'stypy_return_type', res_163271)
    
    # ################# End of '_exp_fjb(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_exp_fjb' in the type store
    # Getting the type of 'stypy_return_type' (line 77)
    stypy_return_type_163272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_163272)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_exp_fjb'
    return stypy_return_type_163272

# Assigning a type to the variable '_exp_fjb' (line 77)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 0), '_exp_fjb', _exp_fjb)

@norecursion
def _exp_est(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_exp_est'
    module_type_store = module_type_store.open_function_context('_exp_est', 83, 0, False)
    
    # Passed parameters checking function
    _exp_est.stypy_localization = localization
    _exp_est.stypy_type_of_self = None
    _exp_est.stypy_type_store = module_type_store
    _exp_est.stypy_function_name = '_exp_est'
    _exp_est.stypy_param_names_list = ['data']
    _exp_est.stypy_varargs_param_name = None
    _exp_est.stypy_kwargs_param_name = None
    _exp_est.stypy_call_defaults = defaults
    _exp_est.stypy_call_varargs = varargs
    _exp_est.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_exp_est', ['data'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_exp_est', localization, ['data'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_exp_est(...)' code ##################

    
    # Call to array(...): (line 85)
    # Processing the call arguments (line 85)
    
    # Obtaining an instance of the builtin type 'list' (line 85)
    list_163275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 85)
    # Adding element type (line 85)
    float_163276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 21), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 20), list_163275, float_163276)
    # Adding element type (line 85)
    float_163277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 25), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 20), list_163275, float_163277)
    
    # Processing the call keyword arguments (line 85)
    kwargs_163278 = {}
    # Getting the type of 'np' (line 85)
    np_163273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 11), 'np', False)
    # Obtaining the member 'array' of a type (line 85)
    array_163274 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 11), np_163273, 'array')
    # Calling array(args, kwargs) (line 85)
    array_call_result_163279 = invoke(stypy.reporting.localization.Localization(__file__, 85, 11), array_163274, *[list_163275], **kwargs_163278)
    
    # Assigning a type to the variable 'stypy_return_type' (line 85)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'stypy_return_type', array_call_result_163279)
    
    # ################# End of '_exp_est(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_exp_est' in the type store
    # Getting the type of 'stypy_return_type' (line 83)
    stypy_return_type_163280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_163280)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_exp_est'
    return stypy_return_type_163280

# Assigning a type to the variable '_exp_est' (line 83)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 0), '_exp_est', _exp_est)

# Assigning a Call to a Name (line 87):

# Assigning a Call to a Name (line 87):

# Call to Model(...): (line 87)
# Processing the call arguments (line 87)
# Getting the type of '_lin_fcn' (line 87)
_lin_fcn_163282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 20), '_lin_fcn', False)
# Processing the call keyword arguments (line 87)
# Getting the type of '_lin_fjb' (line 87)
_lin_fjb_163283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 36), '_lin_fjb', False)
keyword_163284 = _lin_fjb_163283
# Getting the type of '_lin_fjd' (line 88)
_lin_fjd_163285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 21), '_lin_fjd', False)
keyword_163286 = _lin_fjd_163285
# Getting the type of '_lin_est' (line 88)
_lin_est_163287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 40), '_lin_est', False)
keyword_163288 = _lin_est_163287

# Obtaining an instance of the builtin type 'dict' (line 89)
dict_163289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 20), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 89)
# Adding element type (key, value) (line 89)
str_163290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 21), 'str', 'name')
str_163291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 29), 'str', 'Arbitrary-dimensional Linear')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 20), dict_163289, (str_163290, str_163291))
# Adding element type (key, value) (line 89)
str_163292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 21), 'str', 'equ')
str_163293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 27), 'str', 'y = B_0 + Sum[i=1..m, B_i * x_i]')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 20), dict_163289, (str_163292, str_163293))
# Adding element type (key, value) (line 89)
str_163294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 21), 'str', 'TeXequ')
str_163295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 30), 'str', '$y=\\beta_0 + \\sum_{i=1}^m \\beta_i x_i$')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 20), dict_163289, (str_163294, str_163295))

keyword_163296 = dict_163289
kwargs_163297 = {'fjacd': keyword_163286, 'estimate': keyword_163288, 'meta': keyword_163296, 'fjacb': keyword_163284}
# Getting the type of 'Model' (line 87)
Model_163281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 14), 'Model', False)
# Calling Model(args, kwargs) (line 87)
Model_call_result_163298 = invoke(stypy.reporting.localization.Localization(__file__, 87, 14), Model_163281, *[_lin_fcn_163282], **kwargs_163297)

# Assigning a type to the variable 'multilinear' (line 87)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 0), 'multilinear', Model_call_result_163298)

@norecursion
def polynomial(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'polynomial'
    module_type_store = module_type_store.open_function_context('polynomial', 94, 0, False)
    
    # Passed parameters checking function
    polynomial.stypy_localization = localization
    polynomial.stypy_type_of_self = None
    polynomial.stypy_type_store = module_type_store
    polynomial.stypy_function_name = 'polynomial'
    polynomial.stypy_param_names_list = ['order']
    polynomial.stypy_varargs_param_name = None
    polynomial.stypy_kwargs_param_name = None
    polynomial.stypy_call_defaults = defaults
    polynomial.stypy_call_varargs = varargs
    polynomial.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'polynomial', ['order'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'polynomial', localization, ['order'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'polynomial(...)' code ##################

    str_163299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, (-1)), 'str', "\n    Factory function for a general polynomial model.\n\n    Parameters\n    ----------\n    order : int or sequence\n        If an integer, it becomes the order of the polynomial to fit. If\n        a sequence of numbers, then these are the explicit powers in the\n        polynomial.\n        A constant term (power 0) is always included, so don't include 0.\n        Thus, polynomial(n) is equivalent to polynomial(range(1, n+1)).\n\n    Returns\n    -------\n    polynomial : Model instance\n        Model instance.\n\n    ")
    
    # Assigning a Call to a Name (line 114):
    
    # Assigning a Call to a Name (line 114):
    
    # Call to asarray(...): (line 114)
    # Processing the call arguments (line 114)
    # Getting the type of 'order' (line 114)
    order_163302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 24), 'order', False)
    # Processing the call keyword arguments (line 114)
    kwargs_163303 = {}
    # Getting the type of 'np' (line 114)
    np_163300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 13), 'np', False)
    # Obtaining the member 'asarray' of a type (line 114)
    asarray_163301 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 13), np_163300, 'asarray')
    # Calling asarray(args, kwargs) (line 114)
    asarray_call_result_163304 = invoke(stypy.reporting.localization.Localization(__file__, 114, 13), asarray_163301, *[order_163302], **kwargs_163303)
    
    # Assigning a type to the variable 'powers' (line 114)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 4), 'powers', asarray_call_result_163304)
    
    
    # Getting the type of 'powers' (line 115)
    powers_163305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 7), 'powers')
    # Obtaining the member 'shape' of a type (line 115)
    shape_163306 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 7), powers_163305, 'shape')
    
    # Obtaining an instance of the builtin type 'tuple' (line 115)
    tuple_163307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 23), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 115)
    
    # Applying the binary operator '==' (line 115)
    result_eq_163308 = python_operator(stypy.reporting.localization.Localization(__file__, 115, 7), '==', shape_163306, tuple_163307)
    
    # Testing the type of an if condition (line 115)
    if_condition_163309 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 115, 4), result_eq_163308)
    # Assigning a type to the variable 'if_condition_163309' (line 115)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 4), 'if_condition_163309', if_condition_163309)
    # SSA begins for if statement (line 115)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 117):
    
    # Assigning a Call to a Name (line 117):
    
    # Call to arange(...): (line 117)
    # Processing the call arguments (line 117)
    int_163312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 27), 'int')
    # Getting the type of 'powers' (line 117)
    powers_163313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 30), 'powers', False)
    int_163314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 39), 'int')
    # Applying the binary operator '+' (line 117)
    result_add_163315 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 30), '+', powers_163313, int_163314)
    
    # Processing the call keyword arguments (line 117)
    kwargs_163316 = {}
    # Getting the type of 'np' (line 117)
    np_163310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 17), 'np', False)
    # Obtaining the member 'arange' of a type (line 117)
    arange_163311 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 17), np_163310, 'arange')
    # Calling arange(args, kwargs) (line 117)
    arange_call_result_163317 = invoke(stypy.reporting.localization.Localization(__file__, 117, 17), arange_163311, *[int_163312, result_add_163315], **kwargs_163316)
    
    # Assigning a type to the variable 'powers' (line 117)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'powers', arange_call_result_163317)
    # SSA join for if statement (line 115)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Tuple to a Attribute (line 119):
    
    # Assigning a Tuple to a Attribute (line 119):
    
    # Obtaining an instance of the builtin type 'tuple' (line 119)
    tuple_163318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 20), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 119)
    # Adding element type (line 119)
    
    # Call to len(...): (line 119)
    # Processing the call arguments (line 119)
    # Getting the type of 'powers' (line 119)
    powers_163320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 24), 'powers', False)
    # Processing the call keyword arguments (line 119)
    kwargs_163321 = {}
    # Getting the type of 'len' (line 119)
    len_163319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 20), 'len', False)
    # Calling len(args, kwargs) (line 119)
    len_call_result_163322 = invoke(stypy.reporting.localization.Localization(__file__, 119, 20), len_163319, *[powers_163320], **kwargs_163321)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 20), tuple_163318, len_call_result_163322)
    # Adding element type (line 119)
    int_163323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 20), tuple_163318, int_163323)
    
    # Getting the type of 'powers' (line 119)
    powers_163324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 4), 'powers')
    # Setting the type of the member 'shape' of a type (line 119)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 4), powers_163324, 'shape', tuple_163318)
    
    # Assigning a BinOp to a Name (line 120):
    
    # Assigning a BinOp to a Name (line 120):
    
    # Call to len(...): (line 120)
    # Processing the call arguments (line 120)
    # Getting the type of 'powers' (line 120)
    powers_163326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 19), 'powers', False)
    # Processing the call keyword arguments (line 120)
    kwargs_163327 = {}
    # Getting the type of 'len' (line 120)
    len_163325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 15), 'len', False)
    # Calling len(args, kwargs) (line 120)
    len_call_result_163328 = invoke(stypy.reporting.localization.Localization(__file__, 120, 15), len_163325, *[powers_163326], **kwargs_163327)
    
    int_163329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 29), 'int')
    # Applying the binary operator '+' (line 120)
    result_add_163330 = python_operator(stypy.reporting.localization.Localization(__file__, 120, 15), '+', len_call_result_163328, int_163329)
    
    # Assigning a type to the variable 'len_beta' (line 120)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 4), 'len_beta', result_add_163330)

    @norecursion
    def _poly_est(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'len_beta' (line 122)
        len_beta_163331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 33), 'len_beta')
        defaults = [len_beta_163331]
        # Create a new context for function '_poly_est'
        module_type_store = module_type_store.open_function_context('_poly_est', 122, 4, False)
        
        # Passed parameters checking function
        _poly_est.stypy_localization = localization
        _poly_est.stypy_type_of_self = None
        _poly_est.stypy_type_store = module_type_store
        _poly_est.stypy_function_name = '_poly_est'
        _poly_est.stypy_param_names_list = ['data', 'len_beta']
        _poly_est.stypy_varargs_param_name = None
        _poly_est.stypy_kwargs_param_name = None
        _poly_est.stypy_call_defaults = defaults
        _poly_est.stypy_call_varargs = varargs
        _poly_est.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_poly_est', ['data', 'len_beta'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_poly_est', localization, ['data', 'len_beta'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_poly_est(...)' code ##################

        
        # Call to ones(...): (line 124)
        # Processing the call arguments (line 124)
        
        # Obtaining an instance of the builtin type 'tuple' (line 124)
        tuple_163334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 24), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 124)
        # Adding element type (line 124)
        # Getting the type of 'len_beta' (line 124)
        len_beta_163335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 24), 'len_beta', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 24), tuple_163334, len_beta_163335)
        
        # Getting the type of 'float' (line 124)
        float_163336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 36), 'float', False)
        # Processing the call keyword arguments (line 124)
        kwargs_163337 = {}
        # Getting the type of 'np' (line 124)
        np_163332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 15), 'np', False)
        # Obtaining the member 'ones' of a type (line 124)
        ones_163333 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 15), np_163332, 'ones')
        # Calling ones(args, kwargs) (line 124)
        ones_call_result_163338 = invoke(stypy.reporting.localization.Localization(__file__, 124, 15), ones_163333, *[tuple_163334, float_163336], **kwargs_163337)
        
        # Assigning a type to the variable 'stypy_return_type' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'stypy_return_type', ones_call_result_163338)
        
        # ################# End of '_poly_est(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_poly_est' in the type store
        # Getting the type of 'stypy_return_type' (line 122)
        stypy_return_type_163339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_163339)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_poly_est'
        return stypy_return_type_163339

    # Assigning a type to the variable '_poly_est' (line 122)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 4), '_poly_est', _poly_est)
    
    # Call to Model(...): (line 126)
    # Processing the call arguments (line 126)
    # Getting the type of '_poly_fcn' (line 126)
    _poly_fcn_163341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 17), '_poly_fcn', False)
    # Processing the call keyword arguments (line 126)
    # Getting the type of '_poly_fjacd' (line 126)
    _poly_fjacd_163342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 34), '_poly_fjacd', False)
    keyword_163343 = _poly_fjacd_163342
    # Getting the type of '_poly_fjacb' (line 126)
    _poly_fjacb_163344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 53), '_poly_fjacb', False)
    keyword_163345 = _poly_fjacb_163344
    # Getting the type of '_poly_est' (line 127)
    _poly_est_163346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 26), '_poly_est', False)
    keyword_163347 = _poly_est_163346
    
    # Obtaining an instance of the builtin type 'tuple' (line 127)
    tuple_163348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 49), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 127)
    # Adding element type (line 127)
    # Getting the type of 'powers' (line 127)
    powers_163349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 49), 'powers', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 49), tuple_163348, powers_163349)
    
    keyword_163350 = tuple_163348
    
    # Obtaining an instance of the builtin type 'dict' (line 128)
    dict_163351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 22), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 128)
    # Adding element type (key, value) (line 128)
    str_163352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 23), 'str', 'name')
    str_163353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 31), 'str', 'Sorta-general Polynomial')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 22), dict_163351, (str_163352, str_163353))
    # Adding element type (key, value) (line 128)
    str_163354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 17), 'str', 'equ')
    str_163355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 24), 'str', 'y = B_0 + Sum[i=1..%s, B_i * (x**i)]')
    # Getting the type of 'len_beta' (line 129)
    len_beta_163356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 66), 'len_beta', False)
    int_163357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 75), 'int')
    # Applying the binary operator '-' (line 129)
    result_sub_163358 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 66), '-', len_beta_163356, int_163357)
    
    # Applying the binary operator '%' (line 129)
    result_mod_163359 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 24), '%', str_163355, result_sub_163358)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 22), dict_163351, (str_163354, result_mod_163359))
    # Adding element type (key, value) (line 128)
    str_163360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 17), 'str', 'TeXequ')
    str_163361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 27), 'str', '$y=\\beta_0 + \\sum_{i=1}^{%s} \\beta_i x^i$')
    # Getting the type of 'len_beta' (line 131)
    len_beta_163362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 25), 'len_beta', False)
    int_163363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 34), 'int')
    # Applying the binary operator '-' (line 131)
    result_sub_163364 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 25), '-', len_beta_163362, int_163363)
    
    # Applying the binary operator '%' (line 130)
    result_mod_163365 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 27), '%', str_163361, result_sub_163364)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 22), dict_163351, (str_163360, result_mod_163365))
    
    keyword_163366 = dict_163351
    kwargs_163367 = {'extra_args': keyword_163350, 'fjacd': keyword_163343, 'estimate': keyword_163347, 'meta': keyword_163366, 'fjacb': keyword_163345}
    # Getting the type of 'Model' (line 126)
    Model_163340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 11), 'Model', False)
    # Calling Model(args, kwargs) (line 126)
    Model_call_result_163368 = invoke(stypy.reporting.localization.Localization(__file__, 126, 11), Model_163340, *[_poly_fcn_163341], **kwargs_163367)
    
    # Assigning a type to the variable 'stypy_return_type' (line 126)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 4), 'stypy_return_type', Model_call_result_163368)
    
    # ################# End of 'polynomial(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'polynomial' in the type store
    # Getting the type of 'stypy_return_type' (line 94)
    stypy_return_type_163369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_163369)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'polynomial'
    return stypy_return_type_163369

# Assigning a type to the variable 'polynomial' (line 94)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 0), 'polynomial', polynomial)

# Assigning a Call to a Name (line 133):

# Assigning a Call to a Name (line 133):

# Call to Model(...): (line 133)
# Processing the call arguments (line 133)
# Getting the type of '_exp_fcn' (line 133)
_exp_fcn_163371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 20), '_exp_fcn', False)
# Processing the call keyword arguments (line 133)
# Getting the type of '_exp_fjd' (line 133)
_exp_fjd_163372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 36), '_exp_fjd', False)
keyword_163373 = _exp_fjd_163372
# Getting the type of '_exp_fjb' (line 133)
_exp_fjb_163374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 52), '_exp_fjb', False)
keyword_163375 = _exp_fjb_163374
# Getting the type of '_exp_est' (line 134)
_exp_est_163376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 29), '_exp_est', False)
keyword_163377 = _exp_est_163376

# Obtaining an instance of the builtin type 'dict' (line 134)
dict_163378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 44), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 134)
# Adding element type (key, value) (line 134)
str_163379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 45), 'str', 'name')
str_163380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 52), 'str', 'Exponential')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 134, 44), dict_163378, (str_163379, str_163380))
# Adding element type (key, value) (line 134)
str_163381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 20), 'str', 'equ')
str_163382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 27), 'str', 'y= B_0 + exp(B_1 * x)')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 134, 44), dict_163378, (str_163381, str_163382))
# Adding element type (key, value) (line 134)
str_163383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 20), 'str', 'TeXequ')
str_163384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 30), 'str', '$y=\\beta_0 + e^{\\beta_1 x}$')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 134, 44), dict_163378, (str_163383, str_163384))

keyword_163385 = dict_163378
kwargs_163386 = {'fjacd': keyword_163373, 'estimate': keyword_163377, 'meta': keyword_163385, 'fjacb': keyword_163375}
# Getting the type of 'Model' (line 133)
Model_163370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 14), 'Model', False)
# Calling Model(args, kwargs) (line 133)
Model_call_result_163387 = invoke(stypy.reporting.localization.Localization(__file__, 133, 14), Model_163370, *[_exp_fcn_163371], **kwargs_163386)

# Assigning a type to the variable 'exponential' (line 133)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 0), 'exponential', Model_call_result_163387)

@norecursion
def _unilin(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_unilin'
    module_type_store = module_type_store.open_function_context('_unilin', 139, 0, False)
    
    # Passed parameters checking function
    _unilin.stypy_localization = localization
    _unilin.stypy_type_of_self = None
    _unilin.stypy_type_store = module_type_store
    _unilin.stypy_function_name = '_unilin'
    _unilin.stypy_param_names_list = ['B', 'x']
    _unilin.stypy_varargs_param_name = None
    _unilin.stypy_kwargs_param_name = None
    _unilin.stypy_call_defaults = defaults
    _unilin.stypy_call_varargs = varargs
    _unilin.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_unilin', ['B', 'x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_unilin', localization, ['B', 'x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_unilin(...)' code ##################

    # Getting the type of 'x' (line 140)
    x_163388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 11), 'x')
    
    # Obtaining the type of the subscript
    int_163389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 15), 'int')
    # Getting the type of 'B' (line 140)
    B_163390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 13), 'B')
    # Obtaining the member '__getitem__' of a type (line 140)
    getitem___163391 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 13), B_163390, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 140)
    subscript_call_result_163392 = invoke(stypy.reporting.localization.Localization(__file__, 140, 13), getitem___163391, int_163389)
    
    # Applying the binary operator '*' (line 140)
    result_mul_163393 = python_operator(stypy.reporting.localization.Localization(__file__, 140, 11), '*', x_163388, subscript_call_result_163392)
    
    
    # Obtaining the type of the subscript
    int_163394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 22), 'int')
    # Getting the type of 'B' (line 140)
    B_163395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 20), 'B')
    # Obtaining the member '__getitem__' of a type (line 140)
    getitem___163396 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 20), B_163395, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 140)
    subscript_call_result_163397 = invoke(stypy.reporting.localization.Localization(__file__, 140, 20), getitem___163396, int_163394)
    
    # Applying the binary operator '+' (line 140)
    result_add_163398 = python_operator(stypy.reporting.localization.Localization(__file__, 140, 11), '+', result_mul_163393, subscript_call_result_163397)
    
    # Assigning a type to the variable 'stypy_return_type' (line 140)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 4), 'stypy_return_type', result_add_163398)
    
    # ################# End of '_unilin(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_unilin' in the type store
    # Getting the type of 'stypy_return_type' (line 139)
    stypy_return_type_163399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_163399)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_unilin'
    return stypy_return_type_163399

# Assigning a type to the variable '_unilin' (line 139)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 0), '_unilin', _unilin)

@norecursion
def _unilin_fjd(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_unilin_fjd'
    module_type_store = module_type_store.open_function_context('_unilin_fjd', 143, 0, False)
    
    # Passed parameters checking function
    _unilin_fjd.stypy_localization = localization
    _unilin_fjd.stypy_type_of_self = None
    _unilin_fjd.stypy_type_store = module_type_store
    _unilin_fjd.stypy_function_name = '_unilin_fjd'
    _unilin_fjd.stypy_param_names_list = ['B', 'x']
    _unilin_fjd.stypy_varargs_param_name = None
    _unilin_fjd.stypy_kwargs_param_name = None
    _unilin_fjd.stypy_call_defaults = defaults
    _unilin_fjd.stypy_call_varargs = varargs
    _unilin_fjd.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_unilin_fjd', ['B', 'x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_unilin_fjd', localization, ['B', 'x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_unilin_fjd(...)' code ##################

    
    # Call to ones(...): (line 144)
    # Processing the call arguments (line 144)
    # Getting the type of 'x' (line 144)
    x_163402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 19), 'x', False)
    # Obtaining the member 'shape' of a type (line 144)
    shape_163403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 19), x_163402, 'shape')
    # Getting the type of 'float' (line 144)
    float_163404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 28), 'float', False)
    # Processing the call keyword arguments (line 144)
    kwargs_163405 = {}
    # Getting the type of 'np' (line 144)
    np_163400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 11), 'np', False)
    # Obtaining the member 'ones' of a type (line 144)
    ones_163401 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 11), np_163400, 'ones')
    # Calling ones(args, kwargs) (line 144)
    ones_call_result_163406 = invoke(stypy.reporting.localization.Localization(__file__, 144, 11), ones_163401, *[shape_163403, float_163404], **kwargs_163405)
    
    
    # Obtaining the type of the subscript
    int_163407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 39), 'int')
    # Getting the type of 'B' (line 144)
    B_163408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 37), 'B')
    # Obtaining the member '__getitem__' of a type (line 144)
    getitem___163409 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 37), B_163408, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 144)
    subscript_call_result_163410 = invoke(stypy.reporting.localization.Localization(__file__, 144, 37), getitem___163409, int_163407)
    
    # Applying the binary operator '*' (line 144)
    result_mul_163411 = python_operator(stypy.reporting.localization.Localization(__file__, 144, 11), '*', ones_call_result_163406, subscript_call_result_163410)
    
    # Assigning a type to the variable 'stypy_return_type' (line 144)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 4), 'stypy_return_type', result_mul_163411)
    
    # ################# End of '_unilin_fjd(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_unilin_fjd' in the type store
    # Getting the type of 'stypy_return_type' (line 143)
    stypy_return_type_163412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_163412)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_unilin_fjd'
    return stypy_return_type_163412

# Assigning a type to the variable '_unilin_fjd' (line 143)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 0), '_unilin_fjd', _unilin_fjd)

@norecursion
def _unilin_fjb(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_unilin_fjb'
    module_type_store = module_type_store.open_function_context('_unilin_fjb', 147, 0, False)
    
    # Passed parameters checking function
    _unilin_fjb.stypy_localization = localization
    _unilin_fjb.stypy_type_of_self = None
    _unilin_fjb.stypy_type_store = module_type_store
    _unilin_fjb.stypy_function_name = '_unilin_fjb'
    _unilin_fjb.stypy_param_names_list = ['B', 'x']
    _unilin_fjb.stypy_varargs_param_name = None
    _unilin_fjb.stypy_kwargs_param_name = None
    _unilin_fjb.stypy_call_defaults = defaults
    _unilin_fjb.stypy_call_varargs = varargs
    _unilin_fjb.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_unilin_fjb', ['B', 'x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_unilin_fjb', localization, ['B', 'x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_unilin_fjb(...)' code ##################

    
    # Assigning a Call to a Name (line 148):
    
    # Assigning a Call to a Name (line 148):
    
    # Call to concatenate(...): (line 148)
    # Processing the call arguments (line 148)
    
    # Obtaining an instance of the builtin type 'tuple' (line 148)
    tuple_163415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 148)
    # Adding element type (line 148)
    # Getting the type of 'x' (line 148)
    x_163416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 27), 'x', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 148, 27), tuple_163415, x_163416)
    # Adding element type (line 148)
    
    # Call to ones(...): (line 148)
    # Processing the call arguments (line 148)
    # Getting the type of 'x' (line 148)
    x_163419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 38), 'x', False)
    # Obtaining the member 'shape' of a type (line 148)
    shape_163420 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 38), x_163419, 'shape')
    # Getting the type of 'float' (line 148)
    float_163421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 47), 'float', False)
    # Processing the call keyword arguments (line 148)
    kwargs_163422 = {}
    # Getting the type of 'np' (line 148)
    np_163417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 30), 'np', False)
    # Obtaining the member 'ones' of a type (line 148)
    ones_163418 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 30), np_163417, 'ones')
    # Calling ones(args, kwargs) (line 148)
    ones_call_result_163423 = invoke(stypy.reporting.localization.Localization(__file__, 148, 30), ones_163418, *[shape_163420, float_163421], **kwargs_163422)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 148, 27), tuple_163415, ones_call_result_163423)
    
    # Processing the call keyword arguments (line 148)
    kwargs_163424 = {}
    # Getting the type of 'np' (line 148)
    np_163413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 11), 'np', False)
    # Obtaining the member 'concatenate' of a type (line 148)
    concatenate_163414 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 11), np_163413, 'concatenate')
    # Calling concatenate(args, kwargs) (line 148)
    concatenate_call_result_163425 = invoke(stypy.reporting.localization.Localization(__file__, 148, 11), concatenate_163414, *[tuple_163415], **kwargs_163424)
    
    # Assigning a type to the variable '_ret' (line 148)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 4), '_ret', concatenate_call_result_163425)
    
    # Assigning a BinOp to a Attribute (line 149):
    
    # Assigning a BinOp to a Attribute (line 149):
    
    # Obtaining an instance of the builtin type 'tuple' (line 149)
    tuple_163426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 18), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 149)
    # Adding element type (line 149)
    int_163427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 149, 18), tuple_163426, int_163427)
    
    # Getting the type of 'x' (line 149)
    x_163428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 24), 'x')
    # Obtaining the member 'shape' of a type (line 149)
    shape_163429 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 24), x_163428, 'shape')
    # Applying the binary operator '+' (line 149)
    result_add_163430 = python_operator(stypy.reporting.localization.Localization(__file__, 149, 17), '+', tuple_163426, shape_163429)
    
    # Getting the type of '_ret' (line 149)
    _ret_163431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 4), '_ret')
    # Setting the type of the member 'shape' of a type (line 149)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 4), _ret_163431, 'shape', result_add_163430)
    # Getting the type of '_ret' (line 151)
    _ret_163432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 11), '_ret')
    # Assigning a type to the variable 'stypy_return_type' (line 151)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 4), 'stypy_return_type', _ret_163432)
    
    # ################# End of '_unilin_fjb(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_unilin_fjb' in the type store
    # Getting the type of 'stypy_return_type' (line 147)
    stypy_return_type_163433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_163433)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_unilin_fjb'
    return stypy_return_type_163433

# Assigning a type to the variable '_unilin_fjb' (line 147)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 0), '_unilin_fjb', _unilin_fjb)

@norecursion
def _unilin_est(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_unilin_est'
    module_type_store = module_type_store.open_function_context('_unilin_est', 154, 0, False)
    
    # Passed parameters checking function
    _unilin_est.stypy_localization = localization
    _unilin_est.stypy_type_of_self = None
    _unilin_est.stypy_type_store = module_type_store
    _unilin_est.stypy_function_name = '_unilin_est'
    _unilin_est.stypy_param_names_list = ['data']
    _unilin_est.stypy_varargs_param_name = None
    _unilin_est.stypy_kwargs_param_name = None
    _unilin_est.stypy_call_defaults = defaults
    _unilin_est.stypy_call_varargs = varargs
    _unilin_est.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_unilin_est', ['data'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_unilin_est', localization, ['data'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_unilin_est(...)' code ##################

    
    # Obtaining an instance of the builtin type 'tuple' (line 155)
    tuple_163434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 12), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 155)
    # Adding element type (line 155)
    float_163435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 12), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 155, 12), tuple_163434, float_163435)
    # Adding element type (line 155)
    float_163436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 16), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 155, 12), tuple_163434, float_163436)
    
    # Assigning a type to the variable 'stypy_return_type' (line 155)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 4), 'stypy_return_type', tuple_163434)
    
    # ################# End of '_unilin_est(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_unilin_est' in the type store
    # Getting the type of 'stypy_return_type' (line 154)
    stypy_return_type_163437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_163437)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_unilin_est'
    return stypy_return_type_163437

# Assigning a type to the variable '_unilin_est' (line 154)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 0), '_unilin_est', _unilin_est)

@norecursion
def _quadratic(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_quadratic'
    module_type_store = module_type_store.open_function_context('_quadratic', 158, 0, False)
    
    # Passed parameters checking function
    _quadratic.stypy_localization = localization
    _quadratic.stypy_type_of_self = None
    _quadratic.stypy_type_store = module_type_store
    _quadratic.stypy_function_name = '_quadratic'
    _quadratic.stypy_param_names_list = ['B', 'x']
    _quadratic.stypy_varargs_param_name = None
    _quadratic.stypy_kwargs_param_name = None
    _quadratic.stypy_call_defaults = defaults
    _quadratic.stypy_call_varargs = varargs
    _quadratic.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_quadratic', ['B', 'x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_quadratic', localization, ['B', 'x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_quadratic(...)' code ##################

    # Getting the type of 'x' (line 159)
    x_163438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 11), 'x')
    # Getting the type of 'x' (line 159)
    x_163439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 14), 'x')
    
    # Obtaining the type of the subscript
    int_163440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 18), 'int')
    # Getting the type of 'B' (line 159)
    B_163441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 16), 'B')
    # Obtaining the member '__getitem__' of a type (line 159)
    getitem___163442 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 16), B_163441, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 159)
    subscript_call_result_163443 = invoke(stypy.reporting.localization.Localization(__file__, 159, 16), getitem___163442, int_163440)
    
    # Applying the binary operator '*' (line 159)
    result_mul_163444 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 14), '*', x_163439, subscript_call_result_163443)
    
    
    # Obtaining the type of the subscript
    int_163445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 25), 'int')
    # Getting the type of 'B' (line 159)
    B_163446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 23), 'B')
    # Obtaining the member '__getitem__' of a type (line 159)
    getitem___163447 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 23), B_163446, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 159)
    subscript_call_result_163448 = invoke(stypy.reporting.localization.Localization(__file__, 159, 23), getitem___163447, int_163445)
    
    # Applying the binary operator '+' (line 159)
    result_add_163449 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 14), '+', result_mul_163444, subscript_call_result_163448)
    
    # Applying the binary operator '*' (line 159)
    result_mul_163450 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 11), '*', x_163438, result_add_163449)
    
    
    # Obtaining the type of the subscript
    int_163451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 33), 'int')
    # Getting the type of 'B' (line 159)
    B_163452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 31), 'B')
    # Obtaining the member '__getitem__' of a type (line 159)
    getitem___163453 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 31), B_163452, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 159)
    subscript_call_result_163454 = invoke(stypy.reporting.localization.Localization(__file__, 159, 31), getitem___163453, int_163451)
    
    # Applying the binary operator '+' (line 159)
    result_add_163455 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 11), '+', result_mul_163450, subscript_call_result_163454)
    
    # Assigning a type to the variable 'stypy_return_type' (line 159)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 4), 'stypy_return_type', result_add_163455)
    
    # ################# End of '_quadratic(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_quadratic' in the type store
    # Getting the type of 'stypy_return_type' (line 158)
    stypy_return_type_163456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_163456)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_quadratic'
    return stypy_return_type_163456

# Assigning a type to the variable '_quadratic' (line 158)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 0), '_quadratic', _quadratic)

@norecursion
def _quad_fjd(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_quad_fjd'
    module_type_store = module_type_store.open_function_context('_quad_fjd', 162, 0, False)
    
    # Passed parameters checking function
    _quad_fjd.stypy_localization = localization
    _quad_fjd.stypy_type_of_self = None
    _quad_fjd.stypy_type_store = module_type_store
    _quad_fjd.stypy_function_name = '_quad_fjd'
    _quad_fjd.stypy_param_names_list = ['B', 'x']
    _quad_fjd.stypy_varargs_param_name = None
    _quad_fjd.stypy_kwargs_param_name = None
    _quad_fjd.stypy_call_defaults = defaults
    _quad_fjd.stypy_call_varargs = varargs
    _quad_fjd.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_quad_fjd', ['B', 'x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_quad_fjd', localization, ['B', 'x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_quad_fjd(...)' code ##################

    int_163457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 11), 'int')
    # Getting the type of 'x' (line 163)
    x_163458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 13), 'x')
    # Applying the binary operator '*' (line 163)
    result_mul_163459 = python_operator(stypy.reporting.localization.Localization(__file__, 163, 11), '*', int_163457, x_163458)
    
    
    # Obtaining the type of the subscript
    int_163460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 17), 'int')
    # Getting the type of 'B' (line 163)
    B_163461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 15), 'B')
    # Obtaining the member '__getitem__' of a type (line 163)
    getitem___163462 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 15), B_163461, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 163)
    subscript_call_result_163463 = invoke(stypy.reporting.localization.Localization(__file__, 163, 15), getitem___163462, int_163460)
    
    # Applying the binary operator '*' (line 163)
    result_mul_163464 = python_operator(stypy.reporting.localization.Localization(__file__, 163, 14), '*', result_mul_163459, subscript_call_result_163463)
    
    
    # Obtaining the type of the subscript
    int_163465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 24), 'int')
    # Getting the type of 'B' (line 163)
    B_163466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 22), 'B')
    # Obtaining the member '__getitem__' of a type (line 163)
    getitem___163467 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 22), B_163466, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 163)
    subscript_call_result_163468 = invoke(stypy.reporting.localization.Localization(__file__, 163, 22), getitem___163467, int_163465)
    
    # Applying the binary operator '+' (line 163)
    result_add_163469 = python_operator(stypy.reporting.localization.Localization(__file__, 163, 11), '+', result_mul_163464, subscript_call_result_163468)
    
    # Assigning a type to the variable 'stypy_return_type' (line 163)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 4), 'stypy_return_type', result_add_163469)
    
    # ################# End of '_quad_fjd(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_quad_fjd' in the type store
    # Getting the type of 'stypy_return_type' (line 162)
    stypy_return_type_163470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_163470)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_quad_fjd'
    return stypy_return_type_163470

# Assigning a type to the variable '_quad_fjd' (line 162)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 0), '_quad_fjd', _quad_fjd)

@norecursion
def _quad_fjb(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_quad_fjb'
    module_type_store = module_type_store.open_function_context('_quad_fjb', 166, 0, False)
    
    # Passed parameters checking function
    _quad_fjb.stypy_localization = localization
    _quad_fjb.stypy_type_of_self = None
    _quad_fjb.stypy_type_store = module_type_store
    _quad_fjb.stypy_function_name = '_quad_fjb'
    _quad_fjb.stypy_param_names_list = ['B', 'x']
    _quad_fjb.stypy_varargs_param_name = None
    _quad_fjb.stypy_kwargs_param_name = None
    _quad_fjb.stypy_call_defaults = defaults
    _quad_fjb.stypy_call_varargs = varargs
    _quad_fjb.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_quad_fjb', ['B', 'x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_quad_fjb', localization, ['B', 'x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_quad_fjb(...)' code ##################

    
    # Assigning a Call to a Name (line 167):
    
    # Assigning a Call to a Name (line 167):
    
    # Call to concatenate(...): (line 167)
    # Processing the call arguments (line 167)
    
    # Obtaining an instance of the builtin type 'tuple' (line 167)
    tuple_163473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 167)
    # Adding element type (line 167)
    # Getting the type of 'x' (line 167)
    x_163474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 27), 'x', False)
    # Getting the type of 'x' (line 167)
    x_163475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 29), 'x', False)
    # Applying the binary operator '*' (line 167)
    result_mul_163476 = python_operator(stypy.reporting.localization.Localization(__file__, 167, 27), '*', x_163474, x_163475)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 27), tuple_163473, result_mul_163476)
    # Adding element type (line 167)
    # Getting the type of 'x' (line 167)
    x_163477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 32), 'x', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 27), tuple_163473, x_163477)
    # Adding element type (line 167)
    
    # Call to ones(...): (line 167)
    # Processing the call arguments (line 167)
    # Getting the type of 'x' (line 167)
    x_163480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 43), 'x', False)
    # Obtaining the member 'shape' of a type (line 167)
    shape_163481 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 43), x_163480, 'shape')
    # Getting the type of 'float' (line 167)
    float_163482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 52), 'float', False)
    # Processing the call keyword arguments (line 167)
    kwargs_163483 = {}
    # Getting the type of 'np' (line 167)
    np_163478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 35), 'np', False)
    # Obtaining the member 'ones' of a type (line 167)
    ones_163479 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 35), np_163478, 'ones')
    # Calling ones(args, kwargs) (line 167)
    ones_call_result_163484 = invoke(stypy.reporting.localization.Localization(__file__, 167, 35), ones_163479, *[shape_163481, float_163482], **kwargs_163483)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 27), tuple_163473, ones_call_result_163484)
    
    # Processing the call keyword arguments (line 167)
    kwargs_163485 = {}
    # Getting the type of 'np' (line 167)
    np_163471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 11), 'np', False)
    # Obtaining the member 'concatenate' of a type (line 167)
    concatenate_163472 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 11), np_163471, 'concatenate')
    # Calling concatenate(args, kwargs) (line 167)
    concatenate_call_result_163486 = invoke(stypy.reporting.localization.Localization(__file__, 167, 11), concatenate_163472, *[tuple_163473], **kwargs_163485)
    
    # Assigning a type to the variable '_ret' (line 167)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 4), '_ret', concatenate_call_result_163486)
    
    # Assigning a BinOp to a Attribute (line 168):
    
    # Assigning a BinOp to a Attribute (line 168):
    
    # Obtaining an instance of the builtin type 'tuple' (line 168)
    tuple_163487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 18), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 168)
    # Adding element type (line 168)
    int_163488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 18), tuple_163487, int_163488)
    
    # Getting the type of 'x' (line 168)
    x_163489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 24), 'x')
    # Obtaining the member 'shape' of a type (line 168)
    shape_163490 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 24), x_163489, 'shape')
    # Applying the binary operator '+' (line 168)
    result_add_163491 = python_operator(stypy.reporting.localization.Localization(__file__, 168, 17), '+', tuple_163487, shape_163490)
    
    # Getting the type of '_ret' (line 168)
    _ret_163492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 4), '_ret')
    # Setting the type of the member 'shape' of a type (line 168)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 4), _ret_163492, 'shape', result_add_163491)
    # Getting the type of '_ret' (line 170)
    _ret_163493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 11), '_ret')
    # Assigning a type to the variable 'stypy_return_type' (line 170)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 4), 'stypy_return_type', _ret_163493)
    
    # ################# End of '_quad_fjb(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_quad_fjb' in the type store
    # Getting the type of 'stypy_return_type' (line 166)
    stypy_return_type_163494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_163494)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_quad_fjb'
    return stypy_return_type_163494

# Assigning a type to the variable '_quad_fjb' (line 166)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 0), '_quad_fjb', _quad_fjb)

@norecursion
def _quad_est(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_quad_est'
    module_type_store = module_type_store.open_function_context('_quad_est', 173, 0, False)
    
    # Passed parameters checking function
    _quad_est.stypy_localization = localization
    _quad_est.stypy_type_of_self = None
    _quad_est.stypy_type_store = module_type_store
    _quad_est.stypy_function_name = '_quad_est'
    _quad_est.stypy_param_names_list = ['data']
    _quad_est.stypy_varargs_param_name = None
    _quad_est.stypy_kwargs_param_name = None
    _quad_est.stypy_call_defaults = defaults
    _quad_est.stypy_call_varargs = varargs
    _quad_est.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_quad_est', ['data'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_quad_est', localization, ['data'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_quad_est(...)' code ##################

    
    # Obtaining an instance of the builtin type 'tuple' (line 174)
    tuple_163495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 12), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 174)
    # Adding element type (line 174)
    float_163496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 12), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 174, 12), tuple_163495, float_163496)
    # Adding element type (line 174)
    float_163497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 15), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 174, 12), tuple_163495, float_163497)
    # Adding element type (line 174)
    float_163498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 18), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 174, 12), tuple_163495, float_163498)
    
    # Assigning a type to the variable 'stypy_return_type' (line 174)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 4), 'stypy_return_type', tuple_163495)
    
    # ################# End of '_quad_est(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_quad_est' in the type store
    # Getting the type of 'stypy_return_type' (line 173)
    stypy_return_type_163499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_163499)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_quad_est'
    return stypy_return_type_163499

# Assigning a type to the variable '_quad_est' (line 173)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 0), '_quad_est', _quad_est)

# Assigning a Call to a Name (line 176):

# Assigning a Call to a Name (line 176):

# Call to Model(...): (line 176)
# Processing the call arguments (line 176)
# Getting the type of '_unilin' (line 176)
_unilin_163501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 18), '_unilin', False)
# Processing the call keyword arguments (line 176)
# Getting the type of '_unilin_fjd' (line 176)
_unilin_fjd_163502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 33), '_unilin_fjd', False)
keyword_163503 = _unilin_fjd_163502
# Getting the type of '_unilin_fjb' (line 176)
_unilin_fjb_163504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 52), '_unilin_fjb', False)
keyword_163505 = _unilin_fjb_163504
# Getting the type of '_unilin_est' (line 177)
_unilin_est_163506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 27), '_unilin_est', False)
keyword_163507 = _unilin_est_163506

# Obtaining an instance of the builtin type 'dict' (line 177)
dict_163508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 45), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 177)
# Adding element type (key, value) (line 177)
str_163509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 46), 'str', 'name')
str_163510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 54), 'str', 'Univariate Linear')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 45), dict_163508, (str_163509, str_163510))
# Adding element type (key, value) (line 177)
str_163511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 18), 'str', 'equ')
str_163512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 25), 'str', 'y = B_0 * x + B_1')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 45), dict_163508, (str_163511, str_163512))
# Adding element type (key, value) (line 177)
str_163513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 18), 'str', 'TeXequ')
str_163514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 28), 'str', '$y = \\beta_0 x + \\beta_1$')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 45), dict_163508, (str_163513, str_163514))

keyword_163515 = dict_163508
kwargs_163516 = {'fjacd': keyword_163503, 'estimate': keyword_163507, 'meta': keyword_163515, 'fjacb': keyword_163505}
# Getting the type of 'Model' (line 176)
Model_163500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 12), 'Model', False)
# Calling Model(args, kwargs) (line 176)
Model_call_result_163517 = invoke(stypy.reporting.localization.Localization(__file__, 176, 12), Model_163500, *[_unilin_163501], **kwargs_163516)

# Assigning a type to the variable 'unilinear' (line 176)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 0), 'unilinear', Model_call_result_163517)

# Assigning a Call to a Name (line 181):

# Assigning a Call to a Name (line 181):

# Call to Model(...): (line 181)
# Processing the call arguments (line 181)
# Getting the type of '_quadratic' (line 181)
_quadratic_163519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 18), '_quadratic', False)
# Processing the call keyword arguments (line 181)
# Getting the type of '_quad_fjd' (line 181)
_quad_fjd_163520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 36), '_quad_fjd', False)
keyword_163521 = _quad_fjd_163520
# Getting the type of '_quad_fjb' (line 181)
_quad_fjb_163522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 53), '_quad_fjb', False)
keyword_163523 = _quad_fjb_163522
# Getting the type of '_quad_est' (line 182)
_quad_est_163524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 27), '_quad_est', False)
keyword_163525 = _quad_est_163524

# Obtaining an instance of the builtin type 'dict' (line 182)
dict_163526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 43), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 182)
# Adding element type (key, value) (line 182)
str_163527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 44), 'str', 'name')
str_163528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 52), 'str', 'Quadratic')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 43), dict_163526, (str_163527, str_163528))
# Adding element type (key, value) (line 182)
str_163529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 18), 'str', 'equ')
str_163530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 25), 'str', 'y = B_0*x**2 + B_1*x + B_2')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 43), dict_163526, (str_163529, str_163530))
# Adding element type (key, value) (line 182)
str_163531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 18), 'str', 'TeXequ')
str_163532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 28), 'str', '$y = \\beta_0 x^2 + \\beta_1 x + \\beta_2')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 43), dict_163526, (str_163531, str_163532))

keyword_163533 = dict_163526
kwargs_163534 = {'fjacd': keyword_163521, 'estimate': keyword_163525, 'meta': keyword_163533, 'fjacb': keyword_163523}
# Getting the type of 'Model' (line 181)
Model_163518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 12), 'Model', False)
# Calling Model(args, kwargs) (line 181)
Model_call_result_163535 = invoke(stypy.reporting.localization.Localization(__file__, 181, 12), Model_163518, *[_quadratic_163519], **kwargs_163534)

# Assigning a type to the variable 'quadratic' (line 181)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 0), 'quadratic', Model_call_result_163535)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
