
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: ''' helper_funcs.py.
2:     scavenged from enthought,interpolate
3: '''
4: from __future__ import division, print_function, absolute_import
5: 
6: import numpy as np
7: from . import _interpolate  # C extension.  Does all the real work.
8: 
9: 
10: def atleast_1d_and_contiguous(ary, dtype=np.float64):
11:     return np.atleast_1d(np.ascontiguousarray(ary, dtype))
12: 
13: 
14: @np.deprecate(message="'nearest' is deprecated in SciPy 1.0.0")
15: def nearest(x, y, new_x):
16:     '''
17:     Rounds each new x to nearest input x and returns corresponding input y.
18: 
19:     Parameters
20:     ----------
21:     x : array_like
22:         Independent values.
23:     y : array_like
24:         Dependent values.
25:     new_x : array_like
26:         The x values to return the interpolate y values.
27: 
28:     Returns
29:     -------
30:     nearest : ndarray
31:         Rounds each `new_x` to nearest `x` and returns the corresponding `y`.
32: 
33:     '''
34:     shifted_x = np.concatenate((np.array([x[0]-1]), x[0:-1]))
35: 
36:     midpoints_of_x = atleast_1d_and_contiguous(.5*(x + shifted_x))
37:     new_x = atleast_1d_and_contiguous(new_x)
38: 
39:     TINY = 1e-10
40:     indices = np.searchsorted(midpoints_of_x, new_x+TINY)-1
41:     indices = np.atleast_1d(np.clip(indices, 0, np.Inf).astype(int))
42:     new_y = np.take(y, indices, axis=-1)
43: 
44:     return new_y
45: 
46: 
47: @np.deprecate(message="'linear' is deprecated in SciPy 1.0.0")
48: def linear(x, y, new_x):
49:     '''
50:     Linearly interpolates values in new_x based on the values in x and y
51: 
52:     Parameters
53:     ----------
54:     x : array_like
55:         Independent values
56:     y : array_like
57:         Dependent values
58:     new_x : array_like
59:         The x values to return the interpolated y values.
60: 
61:     '''
62:     x = atleast_1d_and_contiguous(x, np.float64)
63:     y = atleast_1d_and_contiguous(y, np.float64)
64:     new_x = atleast_1d_and_contiguous(new_x, np.float64)
65: 
66:     if y.ndim > 2:
67:         raise ValueError("`linear` only works with 1-D or 2-D arrays.")
68:     if len(y.shape) == 2:
69:         new_y = np.zeros((y.shape[0], len(new_x)), np.float64)
70:         for i in range(len(new_y)):  # for each row
71:             _interpolate.linear_dddd(x, y[i], new_x, new_y[i])
72:     else:
73:         new_y = np.zeros(len(new_x), np.float64)
74:         _interpolate.linear_dddd(x, y, new_x, new_y)
75: 
76:     return new_y
77: 
78: 
79: @np.deprecate(message="'logarithmic' is deprecated in SciPy 1.0.0")
80: def logarithmic(x, y, new_x):
81:     '''
82:     Linearly interpolates values in new_x based in the log space of y.
83: 
84:     Parameters
85:     ----------
86:     x : array_like
87:         Independent values.
88:     y : array_like
89:         Dependent values.
90:     new_x : array_like
91:         The x values to return interpolated y values at.
92: 
93:     '''
94:     x = atleast_1d_and_contiguous(x, np.float64)
95:     y = atleast_1d_and_contiguous(y, np.float64)
96:     new_x = atleast_1d_and_contiguous(new_x, np.float64)
97: 
98:     if y.ndim > 2:
99:         raise ValueError("`linear` only works with 1-D or 2-D arrays.")
100:     if len(y.shape) == 2:
101:         new_y = np.zeros((y.shape[0], len(new_x)), np.float64)
102:         for i in range(len(new_y)):
103:             _interpolate.loginterp_dddd(x, y[i], new_x, new_y[i])
104:     else:
105:         new_y = np.zeros(len(new_x), np.float64)
106:         _interpolate.loginterp_dddd(x, y, new_x, new_y)
107: 
108:     return new_y
109: 
110: 
111: @np.deprecate(message="'block_average_above' is deprecated in SciPy 1.0.0")
112: def block_average_above(x, y, new_x):
113:     '''
114:     Linearly interpolates values in new_x based on the values in x and y.
115: 
116:     Parameters
117:     ----------
118:     x : array_like
119:         Independent values.
120:     y : array_like
121:         Dependent values.
122:     new_x : array_like
123:         The x values to interpolate y values.
124: 
125:     '''
126:     bad_index = None
127:     x = atleast_1d_and_contiguous(x, np.float64)
128:     y = atleast_1d_and_contiguous(y, np.float64)
129:     new_x = atleast_1d_and_contiguous(new_x, np.float64)
130: 
131:     if y.ndim > 2:
132:         raise ValueError("`linear` only works with 1-D or 2-D arrays.")
133:     if len(y.shape) == 2:
134:         new_y = np.zeros((y.shape[0], len(new_x)), np.float64)
135:         for i in range(len(new_y)):
136:             bad_index = _interpolate.block_averave_above_dddd(x, y[i],
137:                                                             new_x, new_y[i])
138:             if bad_index is not None:
139:                 break
140:     else:
141:         new_y = np.zeros(len(new_x), np.float64)
142:         bad_index = _interpolate.block_average_above_dddd(x, y, new_x, new_y)
143: 
144:     if bad_index is not None:
145:         msg = "block_average_above cannot extrapolate and new_x[%d]=%f "\
146:               "is out of the x range (%f, %f)" % \
147:               (bad_index, new_x[bad_index], x[0], x[-1])
148:         raise ValueError(msg)
149: 
150:     return new_y
151: 
152: 
153: @np.deprecate(message="'block' is deprecated in SciPy 1.0.0")
154: def block(x, y, new_x):
155:     '''
156:     Essentially a step function.
157: 
158:     For each `new_x`, finds largest j such that``x[j] < new_x[j]`` and
159:     returns ``y[j]``.
160: 
161:     Parameters
162:     ----------
163:     x : array_like
164:         Independent values.
165:     y : array_like
166:         Dependent values.
167:     new_x : array_like
168:         The x values used to calculate the interpolated y.
169: 
170:     Returns
171:     -------
172:     block : ndarray
173:         Return array, of same length as `x_new`.
174: 
175:     '''
176:     # find index of values in x that precede values in x
177:     # This code is a little strange -- we really want a routine that
178:     # returns the index of values where x[j] < x[index]
179:     TINY = 1e-10
180:     indices = np.searchsorted(x, new_x+TINY)-1
181: 
182:     # If the value is at the front of the list, it'll have -1.
183:     # In this case, we will use the first (0), element in the array.
184:     # take requires the index array to be an Int
185:     indices = np.atleast_1d(np.clip(indices, 0, np.Inf).astype(int))
186:     new_y = np.take(y, indices, axis=-1)
187:     return new_y
188: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_70719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, (-1)), 'str', ' helper_funcs.py.\n    scavenged from enthought,interpolate\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import numpy' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/interpolate/')
import_70720 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy')

if (type(import_70720) is not StypyTypeError):

    if (import_70720 != 'pyd_module'):
        __import__(import_70720)
        sys_modules_70721 = sys.modules[import_70720]
        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'np', sys_modules_70721.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy', import_70720)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/interpolate/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from scipy.interpolate import _interpolate' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/interpolate/')
import_70722 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.interpolate')

if (type(import_70722) is not StypyTypeError):

    if (import_70722 != 'pyd_module'):
        __import__(import_70722)
        sys_modules_70723 = sys.modules[import_70722]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.interpolate', sys_modules_70723.module_type_store, module_type_store, ['_interpolate'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_70723, sys_modules_70723.module_type_store, module_type_store)
    else:
        from scipy.interpolate import _interpolate

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.interpolate', None, module_type_store, ['_interpolate'], [_interpolate])

else:
    # Assigning a type to the variable 'scipy.interpolate' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.interpolate', import_70722)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/interpolate/')


@norecursion
def atleast_1d_and_contiguous(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'np' (line 10)
    np_70724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 41), 'np')
    # Obtaining the member 'float64' of a type (line 10)
    float64_70725 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 41), np_70724, 'float64')
    defaults = [float64_70725]
    # Create a new context for function 'atleast_1d_and_contiguous'
    module_type_store = module_type_store.open_function_context('atleast_1d_and_contiguous', 10, 0, False)
    
    # Passed parameters checking function
    atleast_1d_and_contiguous.stypy_localization = localization
    atleast_1d_and_contiguous.stypy_type_of_self = None
    atleast_1d_and_contiguous.stypy_type_store = module_type_store
    atleast_1d_and_contiguous.stypy_function_name = 'atleast_1d_and_contiguous'
    atleast_1d_and_contiguous.stypy_param_names_list = ['ary', 'dtype']
    atleast_1d_and_contiguous.stypy_varargs_param_name = None
    atleast_1d_and_contiguous.stypy_kwargs_param_name = None
    atleast_1d_and_contiguous.stypy_call_defaults = defaults
    atleast_1d_and_contiguous.stypy_call_varargs = varargs
    atleast_1d_and_contiguous.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'atleast_1d_and_contiguous', ['ary', 'dtype'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'atleast_1d_and_contiguous', localization, ['ary', 'dtype'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'atleast_1d_and_contiguous(...)' code ##################

    
    # Call to atleast_1d(...): (line 11)
    # Processing the call arguments (line 11)
    
    # Call to ascontiguousarray(...): (line 11)
    # Processing the call arguments (line 11)
    # Getting the type of 'ary' (line 11)
    ary_70730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 46), 'ary', False)
    # Getting the type of 'dtype' (line 11)
    dtype_70731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 51), 'dtype', False)
    # Processing the call keyword arguments (line 11)
    kwargs_70732 = {}
    # Getting the type of 'np' (line 11)
    np_70728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 25), 'np', False)
    # Obtaining the member 'ascontiguousarray' of a type (line 11)
    ascontiguousarray_70729 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 25), np_70728, 'ascontiguousarray')
    # Calling ascontiguousarray(args, kwargs) (line 11)
    ascontiguousarray_call_result_70733 = invoke(stypy.reporting.localization.Localization(__file__, 11, 25), ascontiguousarray_70729, *[ary_70730, dtype_70731], **kwargs_70732)
    
    # Processing the call keyword arguments (line 11)
    kwargs_70734 = {}
    # Getting the type of 'np' (line 11)
    np_70726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 11), 'np', False)
    # Obtaining the member 'atleast_1d' of a type (line 11)
    atleast_1d_70727 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 11), np_70726, 'atleast_1d')
    # Calling atleast_1d(args, kwargs) (line 11)
    atleast_1d_call_result_70735 = invoke(stypy.reporting.localization.Localization(__file__, 11, 11), atleast_1d_70727, *[ascontiguousarray_call_result_70733], **kwargs_70734)
    
    # Assigning a type to the variable 'stypy_return_type' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'stypy_return_type', atleast_1d_call_result_70735)
    
    # ################# End of 'atleast_1d_and_contiguous(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'atleast_1d_and_contiguous' in the type store
    # Getting the type of 'stypy_return_type' (line 10)
    stypy_return_type_70736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_70736)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'atleast_1d_and_contiguous'
    return stypy_return_type_70736

# Assigning a type to the variable 'atleast_1d_and_contiguous' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'atleast_1d_and_contiguous', atleast_1d_and_contiguous)

@norecursion
def nearest(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'nearest'
    module_type_store = module_type_store.open_function_context('nearest', 14, 0, False)
    
    # Passed parameters checking function
    nearest.stypy_localization = localization
    nearest.stypy_type_of_self = None
    nearest.stypy_type_store = module_type_store
    nearest.stypy_function_name = 'nearest'
    nearest.stypy_param_names_list = ['x', 'y', 'new_x']
    nearest.stypy_varargs_param_name = None
    nearest.stypy_kwargs_param_name = None
    nearest.stypy_call_defaults = defaults
    nearest.stypy_call_varargs = varargs
    nearest.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'nearest', ['x', 'y', 'new_x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'nearest', localization, ['x', 'y', 'new_x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'nearest(...)' code ##################

    str_70737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, (-1)), 'str', '\n    Rounds each new x to nearest input x and returns corresponding input y.\n\n    Parameters\n    ----------\n    x : array_like\n        Independent values.\n    y : array_like\n        Dependent values.\n    new_x : array_like\n        The x values to return the interpolate y values.\n\n    Returns\n    -------\n    nearest : ndarray\n        Rounds each `new_x` to nearest `x` and returns the corresponding `y`.\n\n    ')
    
    # Assigning a Call to a Name (line 34):
    
    # Call to concatenate(...): (line 34)
    # Processing the call arguments (line 34)
    
    # Obtaining an instance of the builtin type 'tuple' (line 34)
    tuple_70740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 32), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 34)
    # Adding element type (line 34)
    
    # Call to array(...): (line 34)
    # Processing the call arguments (line 34)
    
    # Obtaining an instance of the builtin type 'list' (line 34)
    list_70743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 41), 'list')
    # Adding type elements to the builtin type 'list' instance (line 34)
    # Adding element type (line 34)
    
    # Obtaining the type of the subscript
    int_70744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 44), 'int')
    # Getting the type of 'x' (line 34)
    x_70745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 42), 'x', False)
    # Obtaining the member '__getitem__' of a type (line 34)
    getitem___70746 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 42), x_70745, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 34)
    subscript_call_result_70747 = invoke(stypy.reporting.localization.Localization(__file__, 34, 42), getitem___70746, int_70744)
    
    int_70748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 47), 'int')
    # Applying the binary operator '-' (line 34)
    result_sub_70749 = python_operator(stypy.reporting.localization.Localization(__file__, 34, 42), '-', subscript_call_result_70747, int_70748)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 41), list_70743, result_sub_70749)
    
    # Processing the call keyword arguments (line 34)
    kwargs_70750 = {}
    # Getting the type of 'np' (line 34)
    np_70741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 32), 'np', False)
    # Obtaining the member 'array' of a type (line 34)
    array_70742 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 32), np_70741, 'array')
    # Calling array(args, kwargs) (line 34)
    array_call_result_70751 = invoke(stypy.reporting.localization.Localization(__file__, 34, 32), array_70742, *[list_70743], **kwargs_70750)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 32), tuple_70740, array_call_result_70751)
    # Adding element type (line 34)
    
    # Obtaining the type of the subscript
    int_70752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 54), 'int')
    int_70753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 56), 'int')
    slice_70754 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 34, 52), int_70752, int_70753, None)
    # Getting the type of 'x' (line 34)
    x_70755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 52), 'x', False)
    # Obtaining the member '__getitem__' of a type (line 34)
    getitem___70756 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 52), x_70755, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 34)
    subscript_call_result_70757 = invoke(stypy.reporting.localization.Localization(__file__, 34, 52), getitem___70756, slice_70754)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 32), tuple_70740, subscript_call_result_70757)
    
    # Processing the call keyword arguments (line 34)
    kwargs_70758 = {}
    # Getting the type of 'np' (line 34)
    np_70738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 16), 'np', False)
    # Obtaining the member 'concatenate' of a type (line 34)
    concatenate_70739 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 16), np_70738, 'concatenate')
    # Calling concatenate(args, kwargs) (line 34)
    concatenate_call_result_70759 = invoke(stypy.reporting.localization.Localization(__file__, 34, 16), concatenate_70739, *[tuple_70740], **kwargs_70758)
    
    # Assigning a type to the variable 'shifted_x' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'shifted_x', concatenate_call_result_70759)
    
    # Assigning a Call to a Name (line 36):
    
    # Call to atleast_1d_and_contiguous(...): (line 36)
    # Processing the call arguments (line 36)
    float_70761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 47), 'float')
    # Getting the type of 'x' (line 36)
    x_70762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 51), 'x', False)
    # Getting the type of 'shifted_x' (line 36)
    shifted_x_70763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 55), 'shifted_x', False)
    # Applying the binary operator '+' (line 36)
    result_add_70764 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 51), '+', x_70762, shifted_x_70763)
    
    # Applying the binary operator '*' (line 36)
    result_mul_70765 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 47), '*', float_70761, result_add_70764)
    
    # Processing the call keyword arguments (line 36)
    kwargs_70766 = {}
    # Getting the type of 'atleast_1d_and_contiguous' (line 36)
    atleast_1d_and_contiguous_70760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 21), 'atleast_1d_and_contiguous', False)
    # Calling atleast_1d_and_contiguous(args, kwargs) (line 36)
    atleast_1d_and_contiguous_call_result_70767 = invoke(stypy.reporting.localization.Localization(__file__, 36, 21), atleast_1d_and_contiguous_70760, *[result_mul_70765], **kwargs_70766)
    
    # Assigning a type to the variable 'midpoints_of_x' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'midpoints_of_x', atleast_1d_and_contiguous_call_result_70767)
    
    # Assigning a Call to a Name (line 37):
    
    # Call to atleast_1d_and_contiguous(...): (line 37)
    # Processing the call arguments (line 37)
    # Getting the type of 'new_x' (line 37)
    new_x_70769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 38), 'new_x', False)
    # Processing the call keyword arguments (line 37)
    kwargs_70770 = {}
    # Getting the type of 'atleast_1d_and_contiguous' (line 37)
    atleast_1d_and_contiguous_70768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 12), 'atleast_1d_and_contiguous', False)
    # Calling atleast_1d_and_contiguous(args, kwargs) (line 37)
    atleast_1d_and_contiguous_call_result_70771 = invoke(stypy.reporting.localization.Localization(__file__, 37, 12), atleast_1d_and_contiguous_70768, *[new_x_70769], **kwargs_70770)
    
    # Assigning a type to the variable 'new_x' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'new_x', atleast_1d_and_contiguous_call_result_70771)
    
    # Assigning a Num to a Name (line 39):
    float_70772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 11), 'float')
    # Assigning a type to the variable 'TINY' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'TINY', float_70772)
    
    # Assigning a BinOp to a Name (line 40):
    
    # Call to searchsorted(...): (line 40)
    # Processing the call arguments (line 40)
    # Getting the type of 'midpoints_of_x' (line 40)
    midpoints_of_x_70775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 30), 'midpoints_of_x', False)
    # Getting the type of 'new_x' (line 40)
    new_x_70776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 46), 'new_x', False)
    # Getting the type of 'TINY' (line 40)
    TINY_70777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 52), 'TINY', False)
    # Applying the binary operator '+' (line 40)
    result_add_70778 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 46), '+', new_x_70776, TINY_70777)
    
    # Processing the call keyword arguments (line 40)
    kwargs_70779 = {}
    # Getting the type of 'np' (line 40)
    np_70773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 14), 'np', False)
    # Obtaining the member 'searchsorted' of a type (line 40)
    searchsorted_70774 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 14), np_70773, 'searchsorted')
    # Calling searchsorted(args, kwargs) (line 40)
    searchsorted_call_result_70780 = invoke(stypy.reporting.localization.Localization(__file__, 40, 14), searchsorted_70774, *[midpoints_of_x_70775, result_add_70778], **kwargs_70779)
    
    int_70781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 58), 'int')
    # Applying the binary operator '-' (line 40)
    result_sub_70782 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 14), '-', searchsorted_call_result_70780, int_70781)
    
    # Assigning a type to the variable 'indices' (line 40)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'indices', result_sub_70782)
    
    # Assigning a Call to a Name (line 41):
    
    # Call to atleast_1d(...): (line 41)
    # Processing the call arguments (line 41)
    
    # Call to astype(...): (line 41)
    # Processing the call arguments (line 41)
    # Getting the type of 'int' (line 41)
    int_70794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 63), 'int', False)
    # Processing the call keyword arguments (line 41)
    kwargs_70795 = {}
    
    # Call to clip(...): (line 41)
    # Processing the call arguments (line 41)
    # Getting the type of 'indices' (line 41)
    indices_70787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 36), 'indices', False)
    int_70788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 45), 'int')
    # Getting the type of 'np' (line 41)
    np_70789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 48), 'np', False)
    # Obtaining the member 'Inf' of a type (line 41)
    Inf_70790 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 48), np_70789, 'Inf')
    # Processing the call keyword arguments (line 41)
    kwargs_70791 = {}
    # Getting the type of 'np' (line 41)
    np_70785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 28), 'np', False)
    # Obtaining the member 'clip' of a type (line 41)
    clip_70786 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 28), np_70785, 'clip')
    # Calling clip(args, kwargs) (line 41)
    clip_call_result_70792 = invoke(stypy.reporting.localization.Localization(__file__, 41, 28), clip_70786, *[indices_70787, int_70788, Inf_70790], **kwargs_70791)
    
    # Obtaining the member 'astype' of a type (line 41)
    astype_70793 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 28), clip_call_result_70792, 'astype')
    # Calling astype(args, kwargs) (line 41)
    astype_call_result_70796 = invoke(stypy.reporting.localization.Localization(__file__, 41, 28), astype_70793, *[int_70794], **kwargs_70795)
    
    # Processing the call keyword arguments (line 41)
    kwargs_70797 = {}
    # Getting the type of 'np' (line 41)
    np_70783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 14), 'np', False)
    # Obtaining the member 'atleast_1d' of a type (line 41)
    atleast_1d_70784 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 14), np_70783, 'atleast_1d')
    # Calling atleast_1d(args, kwargs) (line 41)
    atleast_1d_call_result_70798 = invoke(stypy.reporting.localization.Localization(__file__, 41, 14), atleast_1d_70784, *[astype_call_result_70796], **kwargs_70797)
    
    # Assigning a type to the variable 'indices' (line 41)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'indices', atleast_1d_call_result_70798)
    
    # Assigning a Call to a Name (line 42):
    
    # Call to take(...): (line 42)
    # Processing the call arguments (line 42)
    # Getting the type of 'y' (line 42)
    y_70801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 20), 'y', False)
    # Getting the type of 'indices' (line 42)
    indices_70802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 23), 'indices', False)
    # Processing the call keyword arguments (line 42)
    int_70803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 37), 'int')
    keyword_70804 = int_70803
    kwargs_70805 = {'axis': keyword_70804}
    # Getting the type of 'np' (line 42)
    np_70799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 12), 'np', False)
    # Obtaining the member 'take' of a type (line 42)
    take_70800 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 12), np_70799, 'take')
    # Calling take(args, kwargs) (line 42)
    take_call_result_70806 = invoke(stypy.reporting.localization.Localization(__file__, 42, 12), take_70800, *[y_70801, indices_70802], **kwargs_70805)
    
    # Assigning a type to the variable 'new_y' (line 42)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'new_y', take_call_result_70806)
    # Getting the type of 'new_y' (line 44)
    new_y_70807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 11), 'new_y')
    # Assigning a type to the variable 'stypy_return_type' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'stypy_return_type', new_y_70807)
    
    # ################# End of 'nearest(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'nearest' in the type store
    # Getting the type of 'stypy_return_type' (line 14)
    stypy_return_type_70808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_70808)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'nearest'
    return stypy_return_type_70808

# Assigning a type to the variable 'nearest' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'nearest', nearest)

@norecursion
def linear(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'linear'
    module_type_store = module_type_store.open_function_context('linear', 47, 0, False)
    
    # Passed parameters checking function
    linear.stypy_localization = localization
    linear.stypy_type_of_self = None
    linear.stypy_type_store = module_type_store
    linear.stypy_function_name = 'linear'
    linear.stypy_param_names_list = ['x', 'y', 'new_x']
    linear.stypy_varargs_param_name = None
    linear.stypy_kwargs_param_name = None
    linear.stypy_call_defaults = defaults
    linear.stypy_call_varargs = varargs
    linear.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'linear', ['x', 'y', 'new_x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'linear', localization, ['x', 'y', 'new_x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'linear(...)' code ##################

    str_70809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, (-1)), 'str', '\n    Linearly interpolates values in new_x based on the values in x and y\n\n    Parameters\n    ----------\n    x : array_like\n        Independent values\n    y : array_like\n        Dependent values\n    new_x : array_like\n        The x values to return the interpolated y values.\n\n    ')
    
    # Assigning a Call to a Name (line 62):
    
    # Call to atleast_1d_and_contiguous(...): (line 62)
    # Processing the call arguments (line 62)
    # Getting the type of 'x' (line 62)
    x_70811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 34), 'x', False)
    # Getting the type of 'np' (line 62)
    np_70812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 37), 'np', False)
    # Obtaining the member 'float64' of a type (line 62)
    float64_70813 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 37), np_70812, 'float64')
    # Processing the call keyword arguments (line 62)
    kwargs_70814 = {}
    # Getting the type of 'atleast_1d_and_contiguous' (line 62)
    atleast_1d_and_contiguous_70810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'atleast_1d_and_contiguous', False)
    # Calling atleast_1d_and_contiguous(args, kwargs) (line 62)
    atleast_1d_and_contiguous_call_result_70815 = invoke(stypy.reporting.localization.Localization(__file__, 62, 8), atleast_1d_and_contiguous_70810, *[x_70811, float64_70813], **kwargs_70814)
    
    # Assigning a type to the variable 'x' (line 62)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), 'x', atleast_1d_and_contiguous_call_result_70815)
    
    # Assigning a Call to a Name (line 63):
    
    # Call to atleast_1d_and_contiguous(...): (line 63)
    # Processing the call arguments (line 63)
    # Getting the type of 'y' (line 63)
    y_70817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 34), 'y', False)
    # Getting the type of 'np' (line 63)
    np_70818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 37), 'np', False)
    # Obtaining the member 'float64' of a type (line 63)
    float64_70819 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 37), np_70818, 'float64')
    # Processing the call keyword arguments (line 63)
    kwargs_70820 = {}
    # Getting the type of 'atleast_1d_and_contiguous' (line 63)
    atleast_1d_and_contiguous_70816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'atleast_1d_and_contiguous', False)
    # Calling atleast_1d_and_contiguous(args, kwargs) (line 63)
    atleast_1d_and_contiguous_call_result_70821 = invoke(stypy.reporting.localization.Localization(__file__, 63, 8), atleast_1d_and_contiguous_70816, *[y_70817, float64_70819], **kwargs_70820)
    
    # Assigning a type to the variable 'y' (line 63)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'y', atleast_1d_and_contiguous_call_result_70821)
    
    # Assigning a Call to a Name (line 64):
    
    # Call to atleast_1d_and_contiguous(...): (line 64)
    # Processing the call arguments (line 64)
    # Getting the type of 'new_x' (line 64)
    new_x_70823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 38), 'new_x', False)
    # Getting the type of 'np' (line 64)
    np_70824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 45), 'np', False)
    # Obtaining the member 'float64' of a type (line 64)
    float64_70825 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 45), np_70824, 'float64')
    # Processing the call keyword arguments (line 64)
    kwargs_70826 = {}
    # Getting the type of 'atleast_1d_and_contiguous' (line 64)
    atleast_1d_and_contiguous_70822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 12), 'atleast_1d_and_contiguous', False)
    # Calling atleast_1d_and_contiguous(args, kwargs) (line 64)
    atleast_1d_and_contiguous_call_result_70827 = invoke(stypy.reporting.localization.Localization(__file__, 64, 12), atleast_1d_and_contiguous_70822, *[new_x_70823, float64_70825], **kwargs_70826)
    
    # Assigning a type to the variable 'new_x' (line 64)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'new_x', atleast_1d_and_contiguous_call_result_70827)
    
    
    # Getting the type of 'y' (line 66)
    y_70828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 7), 'y')
    # Obtaining the member 'ndim' of a type (line 66)
    ndim_70829 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 7), y_70828, 'ndim')
    int_70830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 16), 'int')
    # Applying the binary operator '>' (line 66)
    result_gt_70831 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 7), '>', ndim_70829, int_70830)
    
    # Testing the type of an if condition (line 66)
    if_condition_70832 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 66, 4), result_gt_70831)
    # Assigning a type to the variable 'if_condition_70832' (line 66)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'if_condition_70832', if_condition_70832)
    # SSA begins for if statement (line 66)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 67)
    # Processing the call arguments (line 67)
    str_70834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 25), 'str', '`linear` only works with 1-D or 2-D arrays.')
    # Processing the call keyword arguments (line 67)
    kwargs_70835 = {}
    # Getting the type of 'ValueError' (line 67)
    ValueError_70833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 67)
    ValueError_call_result_70836 = invoke(stypy.reporting.localization.Localization(__file__, 67, 14), ValueError_70833, *[str_70834], **kwargs_70835)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 67, 8), ValueError_call_result_70836, 'raise parameter', BaseException)
    # SSA join for if statement (line 66)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to len(...): (line 68)
    # Processing the call arguments (line 68)
    # Getting the type of 'y' (line 68)
    y_70838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 11), 'y', False)
    # Obtaining the member 'shape' of a type (line 68)
    shape_70839 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 11), y_70838, 'shape')
    # Processing the call keyword arguments (line 68)
    kwargs_70840 = {}
    # Getting the type of 'len' (line 68)
    len_70837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 7), 'len', False)
    # Calling len(args, kwargs) (line 68)
    len_call_result_70841 = invoke(stypy.reporting.localization.Localization(__file__, 68, 7), len_70837, *[shape_70839], **kwargs_70840)
    
    int_70842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 23), 'int')
    # Applying the binary operator '==' (line 68)
    result_eq_70843 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 7), '==', len_call_result_70841, int_70842)
    
    # Testing the type of an if condition (line 68)
    if_condition_70844 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 68, 4), result_eq_70843)
    # Assigning a type to the variable 'if_condition_70844' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'if_condition_70844', if_condition_70844)
    # SSA begins for if statement (line 68)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 69):
    
    # Call to zeros(...): (line 69)
    # Processing the call arguments (line 69)
    
    # Obtaining an instance of the builtin type 'tuple' (line 69)
    tuple_70847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 26), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 69)
    # Adding element type (line 69)
    
    # Obtaining the type of the subscript
    int_70848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 34), 'int')
    # Getting the type of 'y' (line 69)
    y_70849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 26), 'y', False)
    # Obtaining the member 'shape' of a type (line 69)
    shape_70850 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 26), y_70849, 'shape')
    # Obtaining the member '__getitem__' of a type (line 69)
    getitem___70851 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 26), shape_70850, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 69)
    subscript_call_result_70852 = invoke(stypy.reporting.localization.Localization(__file__, 69, 26), getitem___70851, int_70848)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 26), tuple_70847, subscript_call_result_70852)
    # Adding element type (line 69)
    
    # Call to len(...): (line 69)
    # Processing the call arguments (line 69)
    # Getting the type of 'new_x' (line 69)
    new_x_70854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 42), 'new_x', False)
    # Processing the call keyword arguments (line 69)
    kwargs_70855 = {}
    # Getting the type of 'len' (line 69)
    len_70853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 38), 'len', False)
    # Calling len(args, kwargs) (line 69)
    len_call_result_70856 = invoke(stypy.reporting.localization.Localization(__file__, 69, 38), len_70853, *[new_x_70854], **kwargs_70855)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 26), tuple_70847, len_call_result_70856)
    
    # Getting the type of 'np' (line 69)
    np_70857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 51), 'np', False)
    # Obtaining the member 'float64' of a type (line 69)
    float64_70858 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 51), np_70857, 'float64')
    # Processing the call keyword arguments (line 69)
    kwargs_70859 = {}
    # Getting the type of 'np' (line 69)
    np_70845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 16), 'np', False)
    # Obtaining the member 'zeros' of a type (line 69)
    zeros_70846 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 16), np_70845, 'zeros')
    # Calling zeros(args, kwargs) (line 69)
    zeros_call_result_70860 = invoke(stypy.reporting.localization.Localization(__file__, 69, 16), zeros_70846, *[tuple_70847, float64_70858], **kwargs_70859)
    
    # Assigning a type to the variable 'new_y' (line 69)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'new_y', zeros_call_result_70860)
    
    
    # Call to range(...): (line 70)
    # Processing the call arguments (line 70)
    
    # Call to len(...): (line 70)
    # Processing the call arguments (line 70)
    # Getting the type of 'new_y' (line 70)
    new_y_70863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 27), 'new_y', False)
    # Processing the call keyword arguments (line 70)
    kwargs_70864 = {}
    # Getting the type of 'len' (line 70)
    len_70862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 23), 'len', False)
    # Calling len(args, kwargs) (line 70)
    len_call_result_70865 = invoke(stypy.reporting.localization.Localization(__file__, 70, 23), len_70862, *[new_y_70863], **kwargs_70864)
    
    # Processing the call keyword arguments (line 70)
    kwargs_70866 = {}
    # Getting the type of 'range' (line 70)
    range_70861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 17), 'range', False)
    # Calling range(args, kwargs) (line 70)
    range_call_result_70867 = invoke(stypy.reporting.localization.Localization(__file__, 70, 17), range_70861, *[len_call_result_70865], **kwargs_70866)
    
    # Testing the type of a for loop iterable (line 70)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 70, 8), range_call_result_70867)
    # Getting the type of the for loop variable (line 70)
    for_loop_var_70868 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 70, 8), range_call_result_70867)
    # Assigning a type to the variable 'i' (line 70)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'i', for_loop_var_70868)
    # SSA begins for a for statement (line 70)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to linear_dddd(...): (line 71)
    # Processing the call arguments (line 71)
    # Getting the type of 'x' (line 71)
    x_70871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 37), 'x', False)
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 71)
    i_70872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 42), 'i', False)
    # Getting the type of 'y' (line 71)
    y_70873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 40), 'y', False)
    # Obtaining the member '__getitem__' of a type (line 71)
    getitem___70874 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 40), y_70873, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 71)
    subscript_call_result_70875 = invoke(stypy.reporting.localization.Localization(__file__, 71, 40), getitem___70874, i_70872)
    
    # Getting the type of 'new_x' (line 71)
    new_x_70876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 46), 'new_x', False)
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 71)
    i_70877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 59), 'i', False)
    # Getting the type of 'new_y' (line 71)
    new_y_70878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 53), 'new_y', False)
    # Obtaining the member '__getitem__' of a type (line 71)
    getitem___70879 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 53), new_y_70878, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 71)
    subscript_call_result_70880 = invoke(stypy.reporting.localization.Localization(__file__, 71, 53), getitem___70879, i_70877)
    
    # Processing the call keyword arguments (line 71)
    kwargs_70881 = {}
    # Getting the type of '_interpolate' (line 71)
    _interpolate_70869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 12), '_interpolate', False)
    # Obtaining the member 'linear_dddd' of a type (line 71)
    linear_dddd_70870 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 12), _interpolate_70869, 'linear_dddd')
    # Calling linear_dddd(args, kwargs) (line 71)
    linear_dddd_call_result_70882 = invoke(stypy.reporting.localization.Localization(__file__, 71, 12), linear_dddd_70870, *[x_70871, subscript_call_result_70875, new_x_70876, subscript_call_result_70880], **kwargs_70881)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 68)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 73):
    
    # Call to zeros(...): (line 73)
    # Processing the call arguments (line 73)
    
    # Call to len(...): (line 73)
    # Processing the call arguments (line 73)
    # Getting the type of 'new_x' (line 73)
    new_x_70886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 29), 'new_x', False)
    # Processing the call keyword arguments (line 73)
    kwargs_70887 = {}
    # Getting the type of 'len' (line 73)
    len_70885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 25), 'len', False)
    # Calling len(args, kwargs) (line 73)
    len_call_result_70888 = invoke(stypy.reporting.localization.Localization(__file__, 73, 25), len_70885, *[new_x_70886], **kwargs_70887)
    
    # Getting the type of 'np' (line 73)
    np_70889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 37), 'np', False)
    # Obtaining the member 'float64' of a type (line 73)
    float64_70890 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 37), np_70889, 'float64')
    # Processing the call keyword arguments (line 73)
    kwargs_70891 = {}
    # Getting the type of 'np' (line 73)
    np_70883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 16), 'np', False)
    # Obtaining the member 'zeros' of a type (line 73)
    zeros_70884 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 16), np_70883, 'zeros')
    # Calling zeros(args, kwargs) (line 73)
    zeros_call_result_70892 = invoke(stypy.reporting.localization.Localization(__file__, 73, 16), zeros_70884, *[len_call_result_70888, float64_70890], **kwargs_70891)
    
    # Assigning a type to the variable 'new_y' (line 73)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'new_y', zeros_call_result_70892)
    
    # Call to linear_dddd(...): (line 74)
    # Processing the call arguments (line 74)
    # Getting the type of 'x' (line 74)
    x_70895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 33), 'x', False)
    # Getting the type of 'y' (line 74)
    y_70896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 36), 'y', False)
    # Getting the type of 'new_x' (line 74)
    new_x_70897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 39), 'new_x', False)
    # Getting the type of 'new_y' (line 74)
    new_y_70898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 46), 'new_y', False)
    # Processing the call keyword arguments (line 74)
    kwargs_70899 = {}
    # Getting the type of '_interpolate' (line 74)
    _interpolate_70893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), '_interpolate', False)
    # Obtaining the member 'linear_dddd' of a type (line 74)
    linear_dddd_70894 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 8), _interpolate_70893, 'linear_dddd')
    # Calling linear_dddd(args, kwargs) (line 74)
    linear_dddd_call_result_70900 = invoke(stypy.reporting.localization.Localization(__file__, 74, 8), linear_dddd_70894, *[x_70895, y_70896, new_x_70897, new_y_70898], **kwargs_70899)
    
    # SSA join for if statement (line 68)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'new_y' (line 76)
    new_y_70901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 11), 'new_y')
    # Assigning a type to the variable 'stypy_return_type' (line 76)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'stypy_return_type', new_y_70901)
    
    # ################# End of 'linear(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'linear' in the type store
    # Getting the type of 'stypy_return_type' (line 47)
    stypy_return_type_70902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_70902)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'linear'
    return stypy_return_type_70902

# Assigning a type to the variable 'linear' (line 47)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 0), 'linear', linear)

@norecursion
def logarithmic(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'logarithmic'
    module_type_store = module_type_store.open_function_context('logarithmic', 79, 0, False)
    
    # Passed parameters checking function
    logarithmic.stypy_localization = localization
    logarithmic.stypy_type_of_self = None
    logarithmic.stypy_type_store = module_type_store
    logarithmic.stypy_function_name = 'logarithmic'
    logarithmic.stypy_param_names_list = ['x', 'y', 'new_x']
    logarithmic.stypy_varargs_param_name = None
    logarithmic.stypy_kwargs_param_name = None
    logarithmic.stypy_call_defaults = defaults
    logarithmic.stypy_call_varargs = varargs
    logarithmic.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'logarithmic', ['x', 'y', 'new_x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'logarithmic', localization, ['x', 'y', 'new_x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'logarithmic(...)' code ##################

    str_70903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, (-1)), 'str', '\n    Linearly interpolates values in new_x based in the log space of y.\n\n    Parameters\n    ----------\n    x : array_like\n        Independent values.\n    y : array_like\n        Dependent values.\n    new_x : array_like\n        The x values to return interpolated y values at.\n\n    ')
    
    # Assigning a Call to a Name (line 94):
    
    # Call to atleast_1d_and_contiguous(...): (line 94)
    # Processing the call arguments (line 94)
    # Getting the type of 'x' (line 94)
    x_70905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 34), 'x', False)
    # Getting the type of 'np' (line 94)
    np_70906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 37), 'np', False)
    # Obtaining the member 'float64' of a type (line 94)
    float64_70907 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 37), np_70906, 'float64')
    # Processing the call keyword arguments (line 94)
    kwargs_70908 = {}
    # Getting the type of 'atleast_1d_and_contiguous' (line 94)
    atleast_1d_and_contiguous_70904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'atleast_1d_and_contiguous', False)
    # Calling atleast_1d_and_contiguous(args, kwargs) (line 94)
    atleast_1d_and_contiguous_call_result_70909 = invoke(stypy.reporting.localization.Localization(__file__, 94, 8), atleast_1d_and_contiguous_70904, *[x_70905, float64_70907], **kwargs_70908)
    
    # Assigning a type to the variable 'x' (line 94)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'x', atleast_1d_and_contiguous_call_result_70909)
    
    # Assigning a Call to a Name (line 95):
    
    # Call to atleast_1d_and_contiguous(...): (line 95)
    # Processing the call arguments (line 95)
    # Getting the type of 'y' (line 95)
    y_70911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 34), 'y', False)
    # Getting the type of 'np' (line 95)
    np_70912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 37), 'np', False)
    # Obtaining the member 'float64' of a type (line 95)
    float64_70913 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 37), np_70912, 'float64')
    # Processing the call keyword arguments (line 95)
    kwargs_70914 = {}
    # Getting the type of 'atleast_1d_and_contiguous' (line 95)
    atleast_1d_and_contiguous_70910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'atleast_1d_and_contiguous', False)
    # Calling atleast_1d_and_contiguous(args, kwargs) (line 95)
    atleast_1d_and_contiguous_call_result_70915 = invoke(stypy.reporting.localization.Localization(__file__, 95, 8), atleast_1d_and_contiguous_70910, *[y_70911, float64_70913], **kwargs_70914)
    
    # Assigning a type to the variable 'y' (line 95)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 4), 'y', atleast_1d_and_contiguous_call_result_70915)
    
    # Assigning a Call to a Name (line 96):
    
    # Call to atleast_1d_and_contiguous(...): (line 96)
    # Processing the call arguments (line 96)
    # Getting the type of 'new_x' (line 96)
    new_x_70917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 38), 'new_x', False)
    # Getting the type of 'np' (line 96)
    np_70918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 45), 'np', False)
    # Obtaining the member 'float64' of a type (line 96)
    float64_70919 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 45), np_70918, 'float64')
    # Processing the call keyword arguments (line 96)
    kwargs_70920 = {}
    # Getting the type of 'atleast_1d_and_contiguous' (line 96)
    atleast_1d_and_contiguous_70916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 12), 'atleast_1d_and_contiguous', False)
    # Calling atleast_1d_and_contiguous(args, kwargs) (line 96)
    atleast_1d_and_contiguous_call_result_70921 = invoke(stypy.reporting.localization.Localization(__file__, 96, 12), atleast_1d_and_contiguous_70916, *[new_x_70917, float64_70919], **kwargs_70920)
    
    # Assigning a type to the variable 'new_x' (line 96)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 4), 'new_x', atleast_1d_and_contiguous_call_result_70921)
    
    
    # Getting the type of 'y' (line 98)
    y_70922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 7), 'y')
    # Obtaining the member 'ndim' of a type (line 98)
    ndim_70923 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 7), y_70922, 'ndim')
    int_70924 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 16), 'int')
    # Applying the binary operator '>' (line 98)
    result_gt_70925 = python_operator(stypy.reporting.localization.Localization(__file__, 98, 7), '>', ndim_70923, int_70924)
    
    # Testing the type of an if condition (line 98)
    if_condition_70926 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 98, 4), result_gt_70925)
    # Assigning a type to the variable 'if_condition_70926' (line 98)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'if_condition_70926', if_condition_70926)
    # SSA begins for if statement (line 98)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 99)
    # Processing the call arguments (line 99)
    str_70928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 25), 'str', '`linear` only works with 1-D or 2-D arrays.')
    # Processing the call keyword arguments (line 99)
    kwargs_70929 = {}
    # Getting the type of 'ValueError' (line 99)
    ValueError_70927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 99)
    ValueError_call_result_70930 = invoke(stypy.reporting.localization.Localization(__file__, 99, 14), ValueError_70927, *[str_70928], **kwargs_70929)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 99, 8), ValueError_call_result_70930, 'raise parameter', BaseException)
    # SSA join for if statement (line 98)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to len(...): (line 100)
    # Processing the call arguments (line 100)
    # Getting the type of 'y' (line 100)
    y_70932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 11), 'y', False)
    # Obtaining the member 'shape' of a type (line 100)
    shape_70933 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 11), y_70932, 'shape')
    # Processing the call keyword arguments (line 100)
    kwargs_70934 = {}
    # Getting the type of 'len' (line 100)
    len_70931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 7), 'len', False)
    # Calling len(args, kwargs) (line 100)
    len_call_result_70935 = invoke(stypy.reporting.localization.Localization(__file__, 100, 7), len_70931, *[shape_70933], **kwargs_70934)
    
    int_70936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 23), 'int')
    # Applying the binary operator '==' (line 100)
    result_eq_70937 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 7), '==', len_call_result_70935, int_70936)
    
    # Testing the type of an if condition (line 100)
    if_condition_70938 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 100, 4), result_eq_70937)
    # Assigning a type to the variable 'if_condition_70938' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'if_condition_70938', if_condition_70938)
    # SSA begins for if statement (line 100)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 101):
    
    # Call to zeros(...): (line 101)
    # Processing the call arguments (line 101)
    
    # Obtaining an instance of the builtin type 'tuple' (line 101)
    tuple_70941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 26), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 101)
    # Adding element type (line 101)
    
    # Obtaining the type of the subscript
    int_70942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 34), 'int')
    # Getting the type of 'y' (line 101)
    y_70943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 26), 'y', False)
    # Obtaining the member 'shape' of a type (line 101)
    shape_70944 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 26), y_70943, 'shape')
    # Obtaining the member '__getitem__' of a type (line 101)
    getitem___70945 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 26), shape_70944, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 101)
    subscript_call_result_70946 = invoke(stypy.reporting.localization.Localization(__file__, 101, 26), getitem___70945, int_70942)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 26), tuple_70941, subscript_call_result_70946)
    # Adding element type (line 101)
    
    # Call to len(...): (line 101)
    # Processing the call arguments (line 101)
    # Getting the type of 'new_x' (line 101)
    new_x_70948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 42), 'new_x', False)
    # Processing the call keyword arguments (line 101)
    kwargs_70949 = {}
    # Getting the type of 'len' (line 101)
    len_70947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 38), 'len', False)
    # Calling len(args, kwargs) (line 101)
    len_call_result_70950 = invoke(stypy.reporting.localization.Localization(__file__, 101, 38), len_70947, *[new_x_70948], **kwargs_70949)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 26), tuple_70941, len_call_result_70950)
    
    # Getting the type of 'np' (line 101)
    np_70951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 51), 'np', False)
    # Obtaining the member 'float64' of a type (line 101)
    float64_70952 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 51), np_70951, 'float64')
    # Processing the call keyword arguments (line 101)
    kwargs_70953 = {}
    # Getting the type of 'np' (line 101)
    np_70939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 16), 'np', False)
    # Obtaining the member 'zeros' of a type (line 101)
    zeros_70940 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 16), np_70939, 'zeros')
    # Calling zeros(args, kwargs) (line 101)
    zeros_call_result_70954 = invoke(stypy.reporting.localization.Localization(__file__, 101, 16), zeros_70940, *[tuple_70941, float64_70952], **kwargs_70953)
    
    # Assigning a type to the variable 'new_y' (line 101)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'new_y', zeros_call_result_70954)
    
    
    # Call to range(...): (line 102)
    # Processing the call arguments (line 102)
    
    # Call to len(...): (line 102)
    # Processing the call arguments (line 102)
    # Getting the type of 'new_y' (line 102)
    new_y_70957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 27), 'new_y', False)
    # Processing the call keyword arguments (line 102)
    kwargs_70958 = {}
    # Getting the type of 'len' (line 102)
    len_70956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 23), 'len', False)
    # Calling len(args, kwargs) (line 102)
    len_call_result_70959 = invoke(stypy.reporting.localization.Localization(__file__, 102, 23), len_70956, *[new_y_70957], **kwargs_70958)
    
    # Processing the call keyword arguments (line 102)
    kwargs_70960 = {}
    # Getting the type of 'range' (line 102)
    range_70955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 17), 'range', False)
    # Calling range(args, kwargs) (line 102)
    range_call_result_70961 = invoke(stypy.reporting.localization.Localization(__file__, 102, 17), range_70955, *[len_call_result_70959], **kwargs_70960)
    
    # Testing the type of a for loop iterable (line 102)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 102, 8), range_call_result_70961)
    # Getting the type of the for loop variable (line 102)
    for_loop_var_70962 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 102, 8), range_call_result_70961)
    # Assigning a type to the variable 'i' (line 102)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'i', for_loop_var_70962)
    # SSA begins for a for statement (line 102)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to loginterp_dddd(...): (line 103)
    # Processing the call arguments (line 103)
    # Getting the type of 'x' (line 103)
    x_70965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 40), 'x', False)
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 103)
    i_70966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 45), 'i', False)
    # Getting the type of 'y' (line 103)
    y_70967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 43), 'y', False)
    # Obtaining the member '__getitem__' of a type (line 103)
    getitem___70968 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 43), y_70967, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 103)
    subscript_call_result_70969 = invoke(stypy.reporting.localization.Localization(__file__, 103, 43), getitem___70968, i_70966)
    
    # Getting the type of 'new_x' (line 103)
    new_x_70970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 49), 'new_x', False)
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 103)
    i_70971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 62), 'i', False)
    # Getting the type of 'new_y' (line 103)
    new_y_70972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 56), 'new_y', False)
    # Obtaining the member '__getitem__' of a type (line 103)
    getitem___70973 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 56), new_y_70972, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 103)
    subscript_call_result_70974 = invoke(stypy.reporting.localization.Localization(__file__, 103, 56), getitem___70973, i_70971)
    
    # Processing the call keyword arguments (line 103)
    kwargs_70975 = {}
    # Getting the type of '_interpolate' (line 103)
    _interpolate_70963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 12), '_interpolate', False)
    # Obtaining the member 'loginterp_dddd' of a type (line 103)
    loginterp_dddd_70964 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 12), _interpolate_70963, 'loginterp_dddd')
    # Calling loginterp_dddd(args, kwargs) (line 103)
    loginterp_dddd_call_result_70976 = invoke(stypy.reporting.localization.Localization(__file__, 103, 12), loginterp_dddd_70964, *[x_70965, subscript_call_result_70969, new_x_70970, subscript_call_result_70974], **kwargs_70975)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 100)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 105):
    
    # Call to zeros(...): (line 105)
    # Processing the call arguments (line 105)
    
    # Call to len(...): (line 105)
    # Processing the call arguments (line 105)
    # Getting the type of 'new_x' (line 105)
    new_x_70980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 29), 'new_x', False)
    # Processing the call keyword arguments (line 105)
    kwargs_70981 = {}
    # Getting the type of 'len' (line 105)
    len_70979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 25), 'len', False)
    # Calling len(args, kwargs) (line 105)
    len_call_result_70982 = invoke(stypy.reporting.localization.Localization(__file__, 105, 25), len_70979, *[new_x_70980], **kwargs_70981)
    
    # Getting the type of 'np' (line 105)
    np_70983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 37), 'np', False)
    # Obtaining the member 'float64' of a type (line 105)
    float64_70984 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 37), np_70983, 'float64')
    # Processing the call keyword arguments (line 105)
    kwargs_70985 = {}
    # Getting the type of 'np' (line 105)
    np_70977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 16), 'np', False)
    # Obtaining the member 'zeros' of a type (line 105)
    zeros_70978 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 16), np_70977, 'zeros')
    # Calling zeros(args, kwargs) (line 105)
    zeros_call_result_70986 = invoke(stypy.reporting.localization.Localization(__file__, 105, 16), zeros_70978, *[len_call_result_70982, float64_70984], **kwargs_70985)
    
    # Assigning a type to the variable 'new_y' (line 105)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'new_y', zeros_call_result_70986)
    
    # Call to loginterp_dddd(...): (line 106)
    # Processing the call arguments (line 106)
    # Getting the type of 'x' (line 106)
    x_70989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 36), 'x', False)
    # Getting the type of 'y' (line 106)
    y_70990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 39), 'y', False)
    # Getting the type of 'new_x' (line 106)
    new_x_70991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 42), 'new_x', False)
    # Getting the type of 'new_y' (line 106)
    new_y_70992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 49), 'new_y', False)
    # Processing the call keyword arguments (line 106)
    kwargs_70993 = {}
    # Getting the type of '_interpolate' (line 106)
    _interpolate_70987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), '_interpolate', False)
    # Obtaining the member 'loginterp_dddd' of a type (line 106)
    loginterp_dddd_70988 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 8), _interpolate_70987, 'loginterp_dddd')
    # Calling loginterp_dddd(args, kwargs) (line 106)
    loginterp_dddd_call_result_70994 = invoke(stypy.reporting.localization.Localization(__file__, 106, 8), loginterp_dddd_70988, *[x_70989, y_70990, new_x_70991, new_y_70992], **kwargs_70993)
    
    # SSA join for if statement (line 100)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'new_y' (line 108)
    new_y_70995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 11), 'new_y')
    # Assigning a type to the variable 'stypy_return_type' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'stypy_return_type', new_y_70995)
    
    # ################# End of 'logarithmic(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'logarithmic' in the type store
    # Getting the type of 'stypy_return_type' (line 79)
    stypy_return_type_70996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_70996)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'logarithmic'
    return stypy_return_type_70996

# Assigning a type to the variable 'logarithmic' (line 79)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 0), 'logarithmic', logarithmic)

@norecursion
def block_average_above(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'block_average_above'
    module_type_store = module_type_store.open_function_context('block_average_above', 111, 0, False)
    
    # Passed parameters checking function
    block_average_above.stypy_localization = localization
    block_average_above.stypy_type_of_self = None
    block_average_above.stypy_type_store = module_type_store
    block_average_above.stypy_function_name = 'block_average_above'
    block_average_above.stypy_param_names_list = ['x', 'y', 'new_x']
    block_average_above.stypy_varargs_param_name = None
    block_average_above.stypy_kwargs_param_name = None
    block_average_above.stypy_call_defaults = defaults
    block_average_above.stypy_call_varargs = varargs
    block_average_above.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'block_average_above', ['x', 'y', 'new_x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'block_average_above', localization, ['x', 'y', 'new_x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'block_average_above(...)' code ##################

    str_70997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, (-1)), 'str', '\n    Linearly interpolates values in new_x based on the values in x and y.\n\n    Parameters\n    ----------\n    x : array_like\n        Independent values.\n    y : array_like\n        Dependent values.\n    new_x : array_like\n        The x values to interpolate y values.\n\n    ')
    
    # Assigning a Name to a Name (line 126):
    # Getting the type of 'None' (line 126)
    None_70998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 16), 'None')
    # Assigning a type to the variable 'bad_index' (line 126)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 4), 'bad_index', None_70998)
    
    # Assigning a Call to a Name (line 127):
    
    # Call to atleast_1d_and_contiguous(...): (line 127)
    # Processing the call arguments (line 127)
    # Getting the type of 'x' (line 127)
    x_71000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 34), 'x', False)
    # Getting the type of 'np' (line 127)
    np_71001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 37), 'np', False)
    # Obtaining the member 'float64' of a type (line 127)
    float64_71002 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 37), np_71001, 'float64')
    # Processing the call keyword arguments (line 127)
    kwargs_71003 = {}
    # Getting the type of 'atleast_1d_and_contiguous' (line 127)
    atleast_1d_and_contiguous_70999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'atleast_1d_and_contiguous', False)
    # Calling atleast_1d_and_contiguous(args, kwargs) (line 127)
    atleast_1d_and_contiguous_call_result_71004 = invoke(stypy.reporting.localization.Localization(__file__, 127, 8), atleast_1d_and_contiguous_70999, *[x_71000, float64_71002], **kwargs_71003)
    
    # Assigning a type to the variable 'x' (line 127)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 4), 'x', atleast_1d_and_contiguous_call_result_71004)
    
    # Assigning a Call to a Name (line 128):
    
    # Call to atleast_1d_and_contiguous(...): (line 128)
    # Processing the call arguments (line 128)
    # Getting the type of 'y' (line 128)
    y_71006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 34), 'y', False)
    # Getting the type of 'np' (line 128)
    np_71007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 37), 'np', False)
    # Obtaining the member 'float64' of a type (line 128)
    float64_71008 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 37), np_71007, 'float64')
    # Processing the call keyword arguments (line 128)
    kwargs_71009 = {}
    # Getting the type of 'atleast_1d_and_contiguous' (line 128)
    atleast_1d_and_contiguous_71005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'atleast_1d_and_contiguous', False)
    # Calling atleast_1d_and_contiguous(args, kwargs) (line 128)
    atleast_1d_and_contiguous_call_result_71010 = invoke(stypy.reporting.localization.Localization(__file__, 128, 8), atleast_1d_and_contiguous_71005, *[y_71006, float64_71008], **kwargs_71009)
    
    # Assigning a type to the variable 'y' (line 128)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 4), 'y', atleast_1d_and_contiguous_call_result_71010)
    
    # Assigning a Call to a Name (line 129):
    
    # Call to atleast_1d_and_contiguous(...): (line 129)
    # Processing the call arguments (line 129)
    # Getting the type of 'new_x' (line 129)
    new_x_71012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 38), 'new_x', False)
    # Getting the type of 'np' (line 129)
    np_71013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 45), 'np', False)
    # Obtaining the member 'float64' of a type (line 129)
    float64_71014 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 45), np_71013, 'float64')
    # Processing the call keyword arguments (line 129)
    kwargs_71015 = {}
    # Getting the type of 'atleast_1d_and_contiguous' (line 129)
    atleast_1d_and_contiguous_71011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 12), 'atleast_1d_and_contiguous', False)
    # Calling atleast_1d_and_contiguous(args, kwargs) (line 129)
    atleast_1d_and_contiguous_call_result_71016 = invoke(stypy.reporting.localization.Localization(__file__, 129, 12), atleast_1d_and_contiguous_71011, *[new_x_71012, float64_71014], **kwargs_71015)
    
    # Assigning a type to the variable 'new_x' (line 129)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 4), 'new_x', atleast_1d_and_contiguous_call_result_71016)
    
    
    # Getting the type of 'y' (line 131)
    y_71017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 7), 'y')
    # Obtaining the member 'ndim' of a type (line 131)
    ndim_71018 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 7), y_71017, 'ndim')
    int_71019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 16), 'int')
    # Applying the binary operator '>' (line 131)
    result_gt_71020 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 7), '>', ndim_71018, int_71019)
    
    # Testing the type of an if condition (line 131)
    if_condition_71021 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 131, 4), result_gt_71020)
    # Assigning a type to the variable 'if_condition_71021' (line 131)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 4), 'if_condition_71021', if_condition_71021)
    # SSA begins for if statement (line 131)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 132)
    # Processing the call arguments (line 132)
    str_71023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 25), 'str', '`linear` only works with 1-D or 2-D arrays.')
    # Processing the call keyword arguments (line 132)
    kwargs_71024 = {}
    # Getting the type of 'ValueError' (line 132)
    ValueError_71022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 132)
    ValueError_call_result_71025 = invoke(stypy.reporting.localization.Localization(__file__, 132, 14), ValueError_71022, *[str_71023], **kwargs_71024)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 132, 8), ValueError_call_result_71025, 'raise parameter', BaseException)
    # SSA join for if statement (line 131)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to len(...): (line 133)
    # Processing the call arguments (line 133)
    # Getting the type of 'y' (line 133)
    y_71027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 11), 'y', False)
    # Obtaining the member 'shape' of a type (line 133)
    shape_71028 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 11), y_71027, 'shape')
    # Processing the call keyword arguments (line 133)
    kwargs_71029 = {}
    # Getting the type of 'len' (line 133)
    len_71026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 7), 'len', False)
    # Calling len(args, kwargs) (line 133)
    len_call_result_71030 = invoke(stypy.reporting.localization.Localization(__file__, 133, 7), len_71026, *[shape_71028], **kwargs_71029)
    
    int_71031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 23), 'int')
    # Applying the binary operator '==' (line 133)
    result_eq_71032 = python_operator(stypy.reporting.localization.Localization(__file__, 133, 7), '==', len_call_result_71030, int_71031)
    
    # Testing the type of an if condition (line 133)
    if_condition_71033 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 133, 4), result_eq_71032)
    # Assigning a type to the variable 'if_condition_71033' (line 133)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 4), 'if_condition_71033', if_condition_71033)
    # SSA begins for if statement (line 133)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 134):
    
    # Call to zeros(...): (line 134)
    # Processing the call arguments (line 134)
    
    # Obtaining an instance of the builtin type 'tuple' (line 134)
    tuple_71036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 26), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 134)
    # Adding element type (line 134)
    
    # Obtaining the type of the subscript
    int_71037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 34), 'int')
    # Getting the type of 'y' (line 134)
    y_71038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 26), 'y', False)
    # Obtaining the member 'shape' of a type (line 134)
    shape_71039 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 26), y_71038, 'shape')
    # Obtaining the member '__getitem__' of a type (line 134)
    getitem___71040 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 26), shape_71039, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 134)
    subscript_call_result_71041 = invoke(stypy.reporting.localization.Localization(__file__, 134, 26), getitem___71040, int_71037)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 134, 26), tuple_71036, subscript_call_result_71041)
    # Adding element type (line 134)
    
    # Call to len(...): (line 134)
    # Processing the call arguments (line 134)
    # Getting the type of 'new_x' (line 134)
    new_x_71043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 42), 'new_x', False)
    # Processing the call keyword arguments (line 134)
    kwargs_71044 = {}
    # Getting the type of 'len' (line 134)
    len_71042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 38), 'len', False)
    # Calling len(args, kwargs) (line 134)
    len_call_result_71045 = invoke(stypy.reporting.localization.Localization(__file__, 134, 38), len_71042, *[new_x_71043], **kwargs_71044)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 134, 26), tuple_71036, len_call_result_71045)
    
    # Getting the type of 'np' (line 134)
    np_71046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 51), 'np', False)
    # Obtaining the member 'float64' of a type (line 134)
    float64_71047 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 51), np_71046, 'float64')
    # Processing the call keyword arguments (line 134)
    kwargs_71048 = {}
    # Getting the type of 'np' (line 134)
    np_71034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 16), 'np', False)
    # Obtaining the member 'zeros' of a type (line 134)
    zeros_71035 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 16), np_71034, 'zeros')
    # Calling zeros(args, kwargs) (line 134)
    zeros_call_result_71049 = invoke(stypy.reporting.localization.Localization(__file__, 134, 16), zeros_71035, *[tuple_71036, float64_71047], **kwargs_71048)
    
    # Assigning a type to the variable 'new_y' (line 134)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'new_y', zeros_call_result_71049)
    
    
    # Call to range(...): (line 135)
    # Processing the call arguments (line 135)
    
    # Call to len(...): (line 135)
    # Processing the call arguments (line 135)
    # Getting the type of 'new_y' (line 135)
    new_y_71052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 27), 'new_y', False)
    # Processing the call keyword arguments (line 135)
    kwargs_71053 = {}
    # Getting the type of 'len' (line 135)
    len_71051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 23), 'len', False)
    # Calling len(args, kwargs) (line 135)
    len_call_result_71054 = invoke(stypy.reporting.localization.Localization(__file__, 135, 23), len_71051, *[new_y_71052], **kwargs_71053)
    
    # Processing the call keyword arguments (line 135)
    kwargs_71055 = {}
    # Getting the type of 'range' (line 135)
    range_71050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 17), 'range', False)
    # Calling range(args, kwargs) (line 135)
    range_call_result_71056 = invoke(stypy.reporting.localization.Localization(__file__, 135, 17), range_71050, *[len_call_result_71054], **kwargs_71055)
    
    # Testing the type of a for loop iterable (line 135)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 135, 8), range_call_result_71056)
    # Getting the type of the for loop variable (line 135)
    for_loop_var_71057 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 135, 8), range_call_result_71056)
    # Assigning a type to the variable 'i' (line 135)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'i', for_loop_var_71057)
    # SSA begins for a for statement (line 135)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 136):
    
    # Call to block_averave_above_dddd(...): (line 136)
    # Processing the call arguments (line 136)
    # Getting the type of 'x' (line 136)
    x_71060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 62), 'x', False)
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 136)
    i_71061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 67), 'i', False)
    # Getting the type of 'y' (line 136)
    y_71062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 65), 'y', False)
    # Obtaining the member '__getitem__' of a type (line 136)
    getitem___71063 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 65), y_71062, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 136)
    subscript_call_result_71064 = invoke(stypy.reporting.localization.Localization(__file__, 136, 65), getitem___71063, i_71061)
    
    # Getting the type of 'new_x' (line 137)
    new_x_71065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 60), 'new_x', False)
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 137)
    i_71066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 73), 'i', False)
    # Getting the type of 'new_y' (line 137)
    new_y_71067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 67), 'new_y', False)
    # Obtaining the member '__getitem__' of a type (line 137)
    getitem___71068 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 67), new_y_71067, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 137)
    subscript_call_result_71069 = invoke(stypy.reporting.localization.Localization(__file__, 137, 67), getitem___71068, i_71066)
    
    # Processing the call keyword arguments (line 136)
    kwargs_71070 = {}
    # Getting the type of '_interpolate' (line 136)
    _interpolate_71058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 24), '_interpolate', False)
    # Obtaining the member 'block_averave_above_dddd' of a type (line 136)
    block_averave_above_dddd_71059 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 24), _interpolate_71058, 'block_averave_above_dddd')
    # Calling block_averave_above_dddd(args, kwargs) (line 136)
    block_averave_above_dddd_call_result_71071 = invoke(stypy.reporting.localization.Localization(__file__, 136, 24), block_averave_above_dddd_71059, *[x_71060, subscript_call_result_71064, new_x_71065, subscript_call_result_71069], **kwargs_71070)
    
    # Assigning a type to the variable 'bad_index' (line 136)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 12), 'bad_index', block_averave_above_dddd_call_result_71071)
    
    # Type idiom detected: calculating its left and rigth part (line 138)
    # Getting the type of 'bad_index' (line 138)
    bad_index_71072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 12), 'bad_index')
    # Getting the type of 'None' (line 138)
    None_71073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 32), 'None')
    
    (may_be_71074, more_types_in_union_71075) = may_not_be_none(bad_index_71072, None_71073)

    if may_be_71074:

        if more_types_in_union_71075:
            # Runtime conditional SSA (line 138)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store


        if more_types_in_union_71075:
            # SSA join for if statement (line 138)
            module_type_store = module_type_store.join_ssa_context()


    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 133)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 141):
    
    # Call to zeros(...): (line 141)
    # Processing the call arguments (line 141)
    
    # Call to len(...): (line 141)
    # Processing the call arguments (line 141)
    # Getting the type of 'new_x' (line 141)
    new_x_71079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 29), 'new_x', False)
    # Processing the call keyword arguments (line 141)
    kwargs_71080 = {}
    # Getting the type of 'len' (line 141)
    len_71078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 25), 'len', False)
    # Calling len(args, kwargs) (line 141)
    len_call_result_71081 = invoke(stypy.reporting.localization.Localization(__file__, 141, 25), len_71078, *[new_x_71079], **kwargs_71080)
    
    # Getting the type of 'np' (line 141)
    np_71082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 37), 'np', False)
    # Obtaining the member 'float64' of a type (line 141)
    float64_71083 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 37), np_71082, 'float64')
    # Processing the call keyword arguments (line 141)
    kwargs_71084 = {}
    # Getting the type of 'np' (line 141)
    np_71076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 16), 'np', False)
    # Obtaining the member 'zeros' of a type (line 141)
    zeros_71077 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 16), np_71076, 'zeros')
    # Calling zeros(args, kwargs) (line 141)
    zeros_call_result_71085 = invoke(stypy.reporting.localization.Localization(__file__, 141, 16), zeros_71077, *[len_call_result_71081, float64_71083], **kwargs_71084)
    
    # Assigning a type to the variable 'new_y' (line 141)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'new_y', zeros_call_result_71085)
    
    # Assigning a Call to a Name (line 142):
    
    # Call to block_average_above_dddd(...): (line 142)
    # Processing the call arguments (line 142)
    # Getting the type of 'x' (line 142)
    x_71088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 58), 'x', False)
    # Getting the type of 'y' (line 142)
    y_71089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 61), 'y', False)
    # Getting the type of 'new_x' (line 142)
    new_x_71090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 64), 'new_x', False)
    # Getting the type of 'new_y' (line 142)
    new_y_71091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 71), 'new_y', False)
    # Processing the call keyword arguments (line 142)
    kwargs_71092 = {}
    # Getting the type of '_interpolate' (line 142)
    _interpolate_71086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 20), '_interpolate', False)
    # Obtaining the member 'block_average_above_dddd' of a type (line 142)
    block_average_above_dddd_71087 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 20), _interpolate_71086, 'block_average_above_dddd')
    # Calling block_average_above_dddd(args, kwargs) (line 142)
    block_average_above_dddd_call_result_71093 = invoke(stypy.reporting.localization.Localization(__file__, 142, 20), block_average_above_dddd_71087, *[x_71088, y_71089, new_x_71090, new_y_71091], **kwargs_71092)
    
    # Assigning a type to the variable 'bad_index' (line 142)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'bad_index', block_average_above_dddd_call_result_71093)
    # SSA join for if statement (line 133)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 144)
    # Getting the type of 'bad_index' (line 144)
    bad_index_71094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 4), 'bad_index')
    # Getting the type of 'None' (line 144)
    None_71095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 24), 'None')
    
    (may_be_71096, more_types_in_union_71097) = may_not_be_none(bad_index_71094, None_71095)

    if may_be_71096:

        if more_types_in_union_71097:
            # Runtime conditional SSA (line 144)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a BinOp to a Name (line 145):
        str_71098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 14), 'str', 'block_average_above cannot extrapolate and new_x[%d]=%f is out of the x range (%f, %f)')
        
        # Obtaining an instance of the builtin type 'tuple' (line 147)
        tuple_71099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 147)
        # Adding element type (line 147)
        # Getting the type of 'bad_index' (line 147)
        bad_index_71100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 15), 'bad_index')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 15), tuple_71099, bad_index_71100)
        # Adding element type (line 147)
        
        # Obtaining the type of the subscript
        # Getting the type of 'bad_index' (line 147)
        bad_index_71101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 32), 'bad_index')
        # Getting the type of 'new_x' (line 147)
        new_x_71102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 26), 'new_x')
        # Obtaining the member '__getitem__' of a type (line 147)
        getitem___71103 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 26), new_x_71102, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 147)
        subscript_call_result_71104 = invoke(stypy.reporting.localization.Localization(__file__, 147, 26), getitem___71103, bad_index_71101)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 15), tuple_71099, subscript_call_result_71104)
        # Adding element type (line 147)
        
        # Obtaining the type of the subscript
        int_71105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 46), 'int')
        # Getting the type of 'x' (line 147)
        x_71106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 44), 'x')
        # Obtaining the member '__getitem__' of a type (line 147)
        getitem___71107 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 44), x_71106, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 147)
        subscript_call_result_71108 = invoke(stypy.reporting.localization.Localization(__file__, 147, 44), getitem___71107, int_71105)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 15), tuple_71099, subscript_call_result_71108)
        # Adding element type (line 147)
        
        # Obtaining the type of the subscript
        int_71109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 52), 'int')
        # Getting the type of 'x' (line 147)
        x_71110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 50), 'x')
        # Obtaining the member '__getitem__' of a type (line 147)
        getitem___71111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 50), x_71110, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 147)
        subscript_call_result_71112 = invoke(stypy.reporting.localization.Localization(__file__, 147, 50), getitem___71111, int_71109)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 15), tuple_71099, subscript_call_result_71112)
        
        # Applying the binary operator '%' (line 145)
        result_mod_71113 = python_operator(stypy.reporting.localization.Localization(__file__, 145, 14), '%', str_71098, tuple_71099)
        
        # Assigning a type to the variable 'msg' (line 145)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 8), 'msg', result_mod_71113)
        
        # Call to ValueError(...): (line 148)
        # Processing the call arguments (line 148)
        # Getting the type of 'msg' (line 148)
        msg_71115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 25), 'msg', False)
        # Processing the call keyword arguments (line 148)
        kwargs_71116 = {}
        # Getting the type of 'ValueError' (line 148)
        ValueError_71114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 14), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 148)
        ValueError_call_result_71117 = invoke(stypy.reporting.localization.Localization(__file__, 148, 14), ValueError_71114, *[msg_71115], **kwargs_71116)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 148, 8), ValueError_call_result_71117, 'raise parameter', BaseException)

        if more_types_in_union_71097:
            # SSA join for if statement (line 144)
            module_type_store = module_type_store.join_ssa_context()


    
    # Getting the type of 'new_y' (line 150)
    new_y_71118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 11), 'new_y')
    # Assigning a type to the variable 'stypy_return_type' (line 150)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 4), 'stypy_return_type', new_y_71118)
    
    # ################# End of 'block_average_above(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'block_average_above' in the type store
    # Getting the type of 'stypy_return_type' (line 111)
    stypy_return_type_71119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_71119)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'block_average_above'
    return stypy_return_type_71119

# Assigning a type to the variable 'block_average_above' (line 111)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 0), 'block_average_above', block_average_above)

@norecursion
def block(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'block'
    module_type_store = module_type_store.open_function_context('block', 153, 0, False)
    
    # Passed parameters checking function
    block.stypy_localization = localization
    block.stypy_type_of_self = None
    block.stypy_type_store = module_type_store
    block.stypy_function_name = 'block'
    block.stypy_param_names_list = ['x', 'y', 'new_x']
    block.stypy_varargs_param_name = None
    block.stypy_kwargs_param_name = None
    block.stypy_call_defaults = defaults
    block.stypy_call_varargs = varargs
    block.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'block', ['x', 'y', 'new_x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'block', localization, ['x', 'y', 'new_x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'block(...)' code ##################

    str_71120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, (-1)), 'str', '\n    Essentially a step function.\n\n    For each `new_x`, finds largest j such that``x[j] < new_x[j]`` and\n    returns ``y[j]``.\n\n    Parameters\n    ----------\n    x : array_like\n        Independent values.\n    y : array_like\n        Dependent values.\n    new_x : array_like\n        The x values used to calculate the interpolated y.\n\n    Returns\n    -------\n    block : ndarray\n        Return array, of same length as `x_new`.\n\n    ')
    
    # Assigning a Num to a Name (line 179):
    float_71121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 11), 'float')
    # Assigning a type to the variable 'TINY' (line 179)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 4), 'TINY', float_71121)
    
    # Assigning a BinOp to a Name (line 180):
    
    # Call to searchsorted(...): (line 180)
    # Processing the call arguments (line 180)
    # Getting the type of 'x' (line 180)
    x_71124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 30), 'x', False)
    # Getting the type of 'new_x' (line 180)
    new_x_71125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 33), 'new_x', False)
    # Getting the type of 'TINY' (line 180)
    TINY_71126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 39), 'TINY', False)
    # Applying the binary operator '+' (line 180)
    result_add_71127 = python_operator(stypy.reporting.localization.Localization(__file__, 180, 33), '+', new_x_71125, TINY_71126)
    
    # Processing the call keyword arguments (line 180)
    kwargs_71128 = {}
    # Getting the type of 'np' (line 180)
    np_71122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 14), 'np', False)
    # Obtaining the member 'searchsorted' of a type (line 180)
    searchsorted_71123 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 14), np_71122, 'searchsorted')
    # Calling searchsorted(args, kwargs) (line 180)
    searchsorted_call_result_71129 = invoke(stypy.reporting.localization.Localization(__file__, 180, 14), searchsorted_71123, *[x_71124, result_add_71127], **kwargs_71128)
    
    int_71130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 45), 'int')
    # Applying the binary operator '-' (line 180)
    result_sub_71131 = python_operator(stypy.reporting.localization.Localization(__file__, 180, 14), '-', searchsorted_call_result_71129, int_71130)
    
    # Assigning a type to the variable 'indices' (line 180)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 4), 'indices', result_sub_71131)
    
    # Assigning a Call to a Name (line 185):
    
    # Call to atleast_1d(...): (line 185)
    # Processing the call arguments (line 185)
    
    # Call to astype(...): (line 185)
    # Processing the call arguments (line 185)
    # Getting the type of 'int' (line 185)
    int_71143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 63), 'int', False)
    # Processing the call keyword arguments (line 185)
    kwargs_71144 = {}
    
    # Call to clip(...): (line 185)
    # Processing the call arguments (line 185)
    # Getting the type of 'indices' (line 185)
    indices_71136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 36), 'indices', False)
    int_71137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 45), 'int')
    # Getting the type of 'np' (line 185)
    np_71138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 48), 'np', False)
    # Obtaining the member 'Inf' of a type (line 185)
    Inf_71139 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 48), np_71138, 'Inf')
    # Processing the call keyword arguments (line 185)
    kwargs_71140 = {}
    # Getting the type of 'np' (line 185)
    np_71134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 28), 'np', False)
    # Obtaining the member 'clip' of a type (line 185)
    clip_71135 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 28), np_71134, 'clip')
    # Calling clip(args, kwargs) (line 185)
    clip_call_result_71141 = invoke(stypy.reporting.localization.Localization(__file__, 185, 28), clip_71135, *[indices_71136, int_71137, Inf_71139], **kwargs_71140)
    
    # Obtaining the member 'astype' of a type (line 185)
    astype_71142 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 28), clip_call_result_71141, 'astype')
    # Calling astype(args, kwargs) (line 185)
    astype_call_result_71145 = invoke(stypy.reporting.localization.Localization(__file__, 185, 28), astype_71142, *[int_71143], **kwargs_71144)
    
    # Processing the call keyword arguments (line 185)
    kwargs_71146 = {}
    # Getting the type of 'np' (line 185)
    np_71132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 14), 'np', False)
    # Obtaining the member 'atleast_1d' of a type (line 185)
    atleast_1d_71133 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 14), np_71132, 'atleast_1d')
    # Calling atleast_1d(args, kwargs) (line 185)
    atleast_1d_call_result_71147 = invoke(stypy.reporting.localization.Localization(__file__, 185, 14), atleast_1d_71133, *[astype_call_result_71145], **kwargs_71146)
    
    # Assigning a type to the variable 'indices' (line 185)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 4), 'indices', atleast_1d_call_result_71147)
    
    # Assigning a Call to a Name (line 186):
    
    # Call to take(...): (line 186)
    # Processing the call arguments (line 186)
    # Getting the type of 'y' (line 186)
    y_71150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 20), 'y', False)
    # Getting the type of 'indices' (line 186)
    indices_71151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 23), 'indices', False)
    # Processing the call keyword arguments (line 186)
    int_71152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 37), 'int')
    keyword_71153 = int_71152
    kwargs_71154 = {'axis': keyword_71153}
    # Getting the type of 'np' (line 186)
    np_71148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 12), 'np', False)
    # Obtaining the member 'take' of a type (line 186)
    take_71149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 12), np_71148, 'take')
    # Calling take(args, kwargs) (line 186)
    take_call_result_71155 = invoke(stypy.reporting.localization.Localization(__file__, 186, 12), take_71149, *[y_71150, indices_71151], **kwargs_71154)
    
    # Assigning a type to the variable 'new_y' (line 186)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 4), 'new_y', take_call_result_71155)
    # Getting the type of 'new_y' (line 187)
    new_y_71156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 11), 'new_y')
    # Assigning a type to the variable 'stypy_return_type' (line 187)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 4), 'stypy_return_type', new_y_71156)
    
    # ################# End of 'block(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'block' in the type store
    # Getting the type of 'stypy_return_type' (line 153)
    stypy_return_type_71157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_71157)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'block'
    return stypy_return_type_71157

# Assigning a type to the variable 'block' (line 153)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 0), 'block', block)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
