
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Module of functions that are like ufuncs in acting on arrays and optionally
3: storing results in an output array.
4: 
5: '''
6: from __future__ import division, absolute_import, print_function
7: 
8: __all__ = ['fix', 'isneginf', 'isposinf']
9: 
10: import numpy.core.numeric as nx
11: 
12: def fix(x, y=None):
13:     '''
14:     Round to nearest integer towards zero.
15: 
16:     Round an array of floats element-wise to nearest integer towards zero.
17:     The rounded values are returned as floats.
18: 
19:     Parameters
20:     ----------
21:     x : array_like
22:         An array of floats to be rounded
23:     y : ndarray, optional
24:         Output array
25: 
26:     Returns
27:     -------
28:     out : ndarray of floats
29:         The array of rounded numbers
30: 
31:     See Also
32:     --------
33:     trunc, floor, ceil
34:     around : Round to given number of decimals
35: 
36:     Examples
37:     --------
38:     >>> np.fix(3.14)
39:     3.0
40:     >>> np.fix(3)
41:     3.0
42:     >>> np.fix([2.1, 2.9, -2.1, -2.9])
43:     array([ 2.,  2., -2., -2.])
44: 
45:     '''
46:     x = nx.asanyarray(x)
47:     y1 = nx.floor(x)
48:     y2 = nx.ceil(x)
49:     if y is None:
50:         y = nx.asanyarray(y1)
51:     y[...] = nx.where(x >= 0, y1, y2)
52:     return y
53: 
54: def isposinf(x, y=None):
55:     '''
56:     Test element-wise for positive infinity, return result as bool array.
57: 
58:     Parameters
59:     ----------
60:     x : array_like
61:         The input array.
62:     y : array_like, optional
63:         A boolean array with the same shape as `x` to store the result.
64: 
65:     Returns
66:     -------
67:     y : ndarray
68:         A boolean array with the same dimensions as the input.
69:         If second argument is not supplied then a boolean array is returned
70:         with values True where the corresponding element of the input is
71:         positive infinity and values False where the element of the input is
72:         not positive infinity.
73: 
74:         If a second argument is supplied the result is stored there. If the
75:         type of that array is a numeric type the result is represented as zeros
76:         and ones, if the type is boolean then as False and True.
77:         The return value `y` is then a reference to that array.
78: 
79:     See Also
80:     --------
81:     isinf, isneginf, isfinite, isnan
82: 
83:     Notes
84:     -----
85:     Numpy uses the IEEE Standard for Binary Floating-Point for Arithmetic
86:     (IEEE 754).
87: 
88:     Errors result if the second argument is also supplied when `x` is a
89:     scalar input, or if first and second arguments have different shapes.
90: 
91:     Examples
92:     --------
93:     >>> np.isposinf(np.PINF)
94:     array(True, dtype=bool)
95:     >>> np.isposinf(np.inf)
96:     array(True, dtype=bool)
97:     >>> np.isposinf(np.NINF)
98:     array(False, dtype=bool)
99:     >>> np.isposinf([-np.inf, 0., np.inf])
100:     array([False, False,  True], dtype=bool)
101: 
102:     >>> x = np.array([-np.inf, 0., np.inf])
103:     >>> y = np.array([2, 2, 2])
104:     >>> np.isposinf(x, y)
105:     array([0, 0, 1])
106:     >>> y
107:     array([0, 0, 1])
108: 
109:     '''
110:     if y is None:
111:         x = nx.asarray(x)
112:         y = nx.empty(x.shape, dtype=nx.bool_)
113:     nx.logical_and(nx.isinf(x), ~nx.signbit(x), y)
114:     return y
115: 
116: def isneginf(x, y=None):
117:     '''
118:     Test element-wise for negative infinity, return result as bool array.
119: 
120:     Parameters
121:     ----------
122:     x : array_like
123:         The input array.
124:     y : array_like, optional
125:         A boolean array with the same shape and type as `x` to store the
126:         result.
127: 
128:     Returns
129:     -------
130:     y : ndarray
131:         A boolean array with the same dimensions as the input.
132:         If second argument is not supplied then a numpy boolean array is
133:         returned with values True where the corresponding element of the
134:         input is negative infinity and values False where the element of
135:         the input is not negative infinity.
136: 
137:         If a second argument is supplied the result is stored there. If the
138:         type of that array is a numeric type the result is represented as
139:         zeros and ones, if the type is boolean then as False and True. The
140:         return value `y` is then a reference to that array.
141: 
142:     See Also
143:     --------
144:     isinf, isposinf, isnan, isfinite
145: 
146:     Notes
147:     -----
148:     Numpy uses the IEEE Standard for Binary Floating-Point for Arithmetic
149:     (IEEE 754).
150: 
151:     Errors result if the second argument is also supplied when x is a scalar
152:     input, or if first and second arguments have different shapes.
153: 
154:     Examples
155:     --------
156:     >>> np.isneginf(np.NINF)
157:     array(True, dtype=bool)
158:     >>> np.isneginf(np.inf)
159:     array(False, dtype=bool)
160:     >>> np.isneginf(np.PINF)
161:     array(False, dtype=bool)
162:     >>> np.isneginf([-np.inf, 0., np.inf])
163:     array([ True, False, False], dtype=bool)
164: 
165:     >>> x = np.array([-np.inf, 0., np.inf])
166:     >>> y = np.array([2, 2, 2])
167:     >>> np.isneginf(x, y)
168:     array([1, 0, 0])
169:     >>> y
170:     array([1, 0, 0])
171: 
172:     '''
173:     if y is None:
174:         x = nx.asarray(x)
175:         y = nx.empty(x.shape, dtype=nx.bool_)
176:     nx.logical_and(nx.isinf(x), nx.signbit(x), y)
177:     return y
178: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_127889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, (-1)), 'str', '\nModule of functions that are like ufuncs in acting on arrays and optionally\nstoring results in an output array.\n\n')

# Assigning a List to a Name (line 8):
__all__ = ['fix', 'isneginf', 'isposinf']
module_type_store.set_exportable_members(['fix', 'isneginf', 'isposinf'])

# Obtaining an instance of the builtin type 'list' (line 8)
list_127890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 8)
# Adding element type (line 8)
str_127891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 11), 'str', 'fix')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 10), list_127890, str_127891)
# Adding element type (line 8)
str_127892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 18), 'str', 'isneginf')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 10), list_127890, str_127892)
# Adding element type (line 8)
str_127893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 30), 'str', 'isposinf')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 10), list_127890, str_127893)

# Assigning a type to the variable '__all__' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), '__all__', list_127890)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'import numpy.core.numeric' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/lib/')
import_127894 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy.core.numeric')

if (type(import_127894) is not StypyTypeError):

    if (import_127894 != 'pyd_module'):
        __import__(import_127894)
        sys_modules_127895 = sys.modules[import_127894]
        import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'nx', sys_modules_127895.module_type_store, module_type_store)
    else:
        import numpy.core.numeric as nx

        import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'nx', numpy.core.numeric, module_type_store)

else:
    # Assigning a type to the variable 'numpy.core.numeric' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy.core.numeric', import_127894)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/lib/')


@norecursion
def fix(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 12)
    None_127896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 13), 'None')
    defaults = [None_127896]
    # Create a new context for function 'fix'
    module_type_store = module_type_store.open_function_context('fix', 12, 0, False)
    
    # Passed parameters checking function
    fix.stypy_localization = localization
    fix.stypy_type_of_self = None
    fix.stypy_type_store = module_type_store
    fix.stypy_function_name = 'fix'
    fix.stypy_param_names_list = ['x', 'y']
    fix.stypy_varargs_param_name = None
    fix.stypy_kwargs_param_name = None
    fix.stypy_call_defaults = defaults
    fix.stypy_call_varargs = varargs
    fix.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'fix', ['x', 'y'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'fix', localization, ['x', 'y'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'fix(...)' code ##################

    str_127897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, (-1)), 'str', '\n    Round to nearest integer towards zero.\n\n    Round an array of floats element-wise to nearest integer towards zero.\n    The rounded values are returned as floats.\n\n    Parameters\n    ----------\n    x : array_like\n        An array of floats to be rounded\n    y : ndarray, optional\n        Output array\n\n    Returns\n    -------\n    out : ndarray of floats\n        The array of rounded numbers\n\n    See Also\n    --------\n    trunc, floor, ceil\n    around : Round to given number of decimals\n\n    Examples\n    --------\n    >>> np.fix(3.14)\n    3.0\n    >>> np.fix(3)\n    3.0\n    >>> np.fix([2.1, 2.9, -2.1, -2.9])\n    array([ 2.,  2., -2., -2.])\n\n    ')
    
    # Assigning a Call to a Name (line 46):
    
    # Call to asanyarray(...): (line 46)
    # Processing the call arguments (line 46)
    # Getting the type of 'x' (line 46)
    x_127900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 22), 'x', False)
    # Processing the call keyword arguments (line 46)
    kwargs_127901 = {}
    # Getting the type of 'nx' (line 46)
    nx_127898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'nx', False)
    # Obtaining the member 'asanyarray' of a type (line 46)
    asanyarray_127899 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 8), nx_127898, 'asanyarray')
    # Calling asanyarray(args, kwargs) (line 46)
    asanyarray_call_result_127902 = invoke(stypy.reporting.localization.Localization(__file__, 46, 8), asanyarray_127899, *[x_127900], **kwargs_127901)
    
    # Assigning a type to the variable 'x' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'x', asanyarray_call_result_127902)
    
    # Assigning a Call to a Name (line 47):
    
    # Call to floor(...): (line 47)
    # Processing the call arguments (line 47)
    # Getting the type of 'x' (line 47)
    x_127905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 18), 'x', False)
    # Processing the call keyword arguments (line 47)
    kwargs_127906 = {}
    # Getting the type of 'nx' (line 47)
    nx_127903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 9), 'nx', False)
    # Obtaining the member 'floor' of a type (line 47)
    floor_127904 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 9), nx_127903, 'floor')
    # Calling floor(args, kwargs) (line 47)
    floor_call_result_127907 = invoke(stypy.reporting.localization.Localization(__file__, 47, 9), floor_127904, *[x_127905], **kwargs_127906)
    
    # Assigning a type to the variable 'y1' (line 47)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'y1', floor_call_result_127907)
    
    # Assigning a Call to a Name (line 48):
    
    # Call to ceil(...): (line 48)
    # Processing the call arguments (line 48)
    # Getting the type of 'x' (line 48)
    x_127910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 17), 'x', False)
    # Processing the call keyword arguments (line 48)
    kwargs_127911 = {}
    # Getting the type of 'nx' (line 48)
    nx_127908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 9), 'nx', False)
    # Obtaining the member 'ceil' of a type (line 48)
    ceil_127909 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 9), nx_127908, 'ceil')
    # Calling ceil(args, kwargs) (line 48)
    ceil_call_result_127912 = invoke(stypy.reporting.localization.Localization(__file__, 48, 9), ceil_127909, *[x_127910], **kwargs_127911)
    
    # Assigning a type to the variable 'y2' (line 48)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'y2', ceil_call_result_127912)
    
    # Type idiom detected: calculating its left and rigth part (line 49)
    # Getting the type of 'y' (line 49)
    y_127913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 7), 'y')
    # Getting the type of 'None' (line 49)
    None_127914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 12), 'None')
    
    (may_be_127915, more_types_in_union_127916) = may_be_none(y_127913, None_127914)

    if may_be_127915:

        if more_types_in_union_127916:
            # Runtime conditional SSA (line 49)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 50):
        
        # Call to asanyarray(...): (line 50)
        # Processing the call arguments (line 50)
        # Getting the type of 'y1' (line 50)
        y1_127919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 26), 'y1', False)
        # Processing the call keyword arguments (line 50)
        kwargs_127920 = {}
        # Getting the type of 'nx' (line 50)
        nx_127917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 12), 'nx', False)
        # Obtaining the member 'asanyarray' of a type (line 50)
        asanyarray_127918 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 12), nx_127917, 'asanyarray')
        # Calling asanyarray(args, kwargs) (line 50)
        asanyarray_call_result_127921 = invoke(stypy.reporting.localization.Localization(__file__, 50, 12), asanyarray_127918, *[y1_127919], **kwargs_127920)
        
        # Assigning a type to the variable 'y' (line 50)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'y', asanyarray_call_result_127921)

        if more_types_in_union_127916:
            # SSA join for if statement (line 49)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Subscript (line 51):
    
    # Call to where(...): (line 51)
    # Processing the call arguments (line 51)
    
    # Getting the type of 'x' (line 51)
    x_127924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 22), 'x', False)
    int_127925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 27), 'int')
    # Applying the binary operator '>=' (line 51)
    result_ge_127926 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 22), '>=', x_127924, int_127925)
    
    # Getting the type of 'y1' (line 51)
    y1_127927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 30), 'y1', False)
    # Getting the type of 'y2' (line 51)
    y2_127928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 34), 'y2', False)
    # Processing the call keyword arguments (line 51)
    kwargs_127929 = {}
    # Getting the type of 'nx' (line 51)
    nx_127922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 13), 'nx', False)
    # Obtaining the member 'where' of a type (line 51)
    where_127923 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 13), nx_127922, 'where')
    # Calling where(args, kwargs) (line 51)
    where_call_result_127930 = invoke(stypy.reporting.localization.Localization(__file__, 51, 13), where_127923, *[result_ge_127926, y1_127927, y2_127928], **kwargs_127929)
    
    # Getting the type of 'y' (line 51)
    y_127931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'y')
    Ellipsis_127932 = Ellipsis
    # Storing an element on a container (line 51)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 4), y_127931, (Ellipsis_127932, where_call_result_127930))
    # Getting the type of 'y' (line 52)
    y_127933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 11), 'y')
    # Assigning a type to the variable 'stypy_return_type' (line 52)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'stypy_return_type', y_127933)
    
    # ################# End of 'fix(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'fix' in the type store
    # Getting the type of 'stypy_return_type' (line 12)
    stypy_return_type_127934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_127934)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'fix'
    return stypy_return_type_127934

# Assigning a type to the variable 'fix' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'fix', fix)

@norecursion
def isposinf(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 54)
    None_127935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 18), 'None')
    defaults = [None_127935]
    # Create a new context for function 'isposinf'
    module_type_store = module_type_store.open_function_context('isposinf', 54, 0, False)
    
    # Passed parameters checking function
    isposinf.stypy_localization = localization
    isposinf.stypy_type_of_self = None
    isposinf.stypy_type_store = module_type_store
    isposinf.stypy_function_name = 'isposinf'
    isposinf.stypy_param_names_list = ['x', 'y']
    isposinf.stypy_varargs_param_name = None
    isposinf.stypy_kwargs_param_name = None
    isposinf.stypy_call_defaults = defaults
    isposinf.stypy_call_varargs = varargs
    isposinf.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'isposinf', ['x', 'y'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'isposinf', localization, ['x', 'y'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'isposinf(...)' code ##################

    str_127936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, (-1)), 'str', '\n    Test element-wise for positive infinity, return result as bool array.\n\n    Parameters\n    ----------\n    x : array_like\n        The input array.\n    y : array_like, optional\n        A boolean array with the same shape as `x` to store the result.\n\n    Returns\n    -------\n    y : ndarray\n        A boolean array with the same dimensions as the input.\n        If second argument is not supplied then a boolean array is returned\n        with values True where the corresponding element of the input is\n        positive infinity and values False where the element of the input is\n        not positive infinity.\n\n        If a second argument is supplied the result is stored there. If the\n        type of that array is a numeric type the result is represented as zeros\n        and ones, if the type is boolean then as False and True.\n        The return value `y` is then a reference to that array.\n\n    See Also\n    --------\n    isinf, isneginf, isfinite, isnan\n\n    Notes\n    -----\n    Numpy uses the IEEE Standard for Binary Floating-Point for Arithmetic\n    (IEEE 754).\n\n    Errors result if the second argument is also supplied when `x` is a\n    scalar input, or if first and second arguments have different shapes.\n\n    Examples\n    --------\n    >>> np.isposinf(np.PINF)\n    array(True, dtype=bool)\n    >>> np.isposinf(np.inf)\n    array(True, dtype=bool)\n    >>> np.isposinf(np.NINF)\n    array(False, dtype=bool)\n    >>> np.isposinf([-np.inf, 0., np.inf])\n    array([False, False,  True], dtype=bool)\n\n    >>> x = np.array([-np.inf, 0., np.inf])\n    >>> y = np.array([2, 2, 2])\n    >>> np.isposinf(x, y)\n    array([0, 0, 1])\n    >>> y\n    array([0, 0, 1])\n\n    ')
    
    # Type idiom detected: calculating its left and rigth part (line 110)
    # Getting the type of 'y' (line 110)
    y_127937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 7), 'y')
    # Getting the type of 'None' (line 110)
    None_127938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 12), 'None')
    
    (may_be_127939, more_types_in_union_127940) = may_be_none(y_127937, None_127938)

    if may_be_127939:

        if more_types_in_union_127940:
            # Runtime conditional SSA (line 110)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 111):
        
        # Call to asarray(...): (line 111)
        # Processing the call arguments (line 111)
        # Getting the type of 'x' (line 111)
        x_127943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 23), 'x', False)
        # Processing the call keyword arguments (line 111)
        kwargs_127944 = {}
        # Getting the type of 'nx' (line 111)
        nx_127941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'nx', False)
        # Obtaining the member 'asarray' of a type (line 111)
        asarray_127942 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 12), nx_127941, 'asarray')
        # Calling asarray(args, kwargs) (line 111)
        asarray_call_result_127945 = invoke(stypy.reporting.localization.Localization(__file__, 111, 12), asarray_127942, *[x_127943], **kwargs_127944)
        
        # Assigning a type to the variable 'x' (line 111)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'x', asarray_call_result_127945)
        
        # Assigning a Call to a Name (line 112):
        
        # Call to empty(...): (line 112)
        # Processing the call arguments (line 112)
        # Getting the type of 'x' (line 112)
        x_127948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 21), 'x', False)
        # Obtaining the member 'shape' of a type (line 112)
        shape_127949 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 21), x_127948, 'shape')
        # Processing the call keyword arguments (line 112)
        # Getting the type of 'nx' (line 112)
        nx_127950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 36), 'nx', False)
        # Obtaining the member 'bool_' of a type (line 112)
        bool__127951 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 36), nx_127950, 'bool_')
        keyword_127952 = bool__127951
        kwargs_127953 = {'dtype': keyword_127952}
        # Getting the type of 'nx' (line 112)
        nx_127946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 12), 'nx', False)
        # Obtaining the member 'empty' of a type (line 112)
        empty_127947 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 12), nx_127946, 'empty')
        # Calling empty(args, kwargs) (line 112)
        empty_call_result_127954 = invoke(stypy.reporting.localization.Localization(__file__, 112, 12), empty_127947, *[shape_127949], **kwargs_127953)
        
        # Assigning a type to the variable 'y' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'y', empty_call_result_127954)

        if more_types_in_union_127940:
            # SSA join for if statement (line 110)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Call to logical_and(...): (line 113)
    # Processing the call arguments (line 113)
    
    # Call to isinf(...): (line 113)
    # Processing the call arguments (line 113)
    # Getting the type of 'x' (line 113)
    x_127959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 28), 'x', False)
    # Processing the call keyword arguments (line 113)
    kwargs_127960 = {}
    # Getting the type of 'nx' (line 113)
    nx_127957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 19), 'nx', False)
    # Obtaining the member 'isinf' of a type (line 113)
    isinf_127958 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 19), nx_127957, 'isinf')
    # Calling isinf(args, kwargs) (line 113)
    isinf_call_result_127961 = invoke(stypy.reporting.localization.Localization(__file__, 113, 19), isinf_127958, *[x_127959], **kwargs_127960)
    
    
    
    # Call to signbit(...): (line 113)
    # Processing the call arguments (line 113)
    # Getting the type of 'x' (line 113)
    x_127964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 44), 'x', False)
    # Processing the call keyword arguments (line 113)
    kwargs_127965 = {}
    # Getting the type of 'nx' (line 113)
    nx_127962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 33), 'nx', False)
    # Obtaining the member 'signbit' of a type (line 113)
    signbit_127963 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 33), nx_127962, 'signbit')
    # Calling signbit(args, kwargs) (line 113)
    signbit_call_result_127966 = invoke(stypy.reporting.localization.Localization(__file__, 113, 33), signbit_127963, *[x_127964], **kwargs_127965)
    
    # Applying the '~' unary operator (line 113)
    result_inv_127967 = python_operator(stypy.reporting.localization.Localization(__file__, 113, 32), '~', signbit_call_result_127966)
    
    # Getting the type of 'y' (line 113)
    y_127968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 48), 'y', False)
    # Processing the call keyword arguments (line 113)
    kwargs_127969 = {}
    # Getting the type of 'nx' (line 113)
    nx_127955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 4), 'nx', False)
    # Obtaining the member 'logical_and' of a type (line 113)
    logical_and_127956 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 4), nx_127955, 'logical_and')
    # Calling logical_and(args, kwargs) (line 113)
    logical_and_call_result_127970 = invoke(stypy.reporting.localization.Localization(__file__, 113, 4), logical_and_127956, *[isinf_call_result_127961, result_inv_127967, y_127968], **kwargs_127969)
    
    # Getting the type of 'y' (line 114)
    y_127971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 11), 'y')
    # Assigning a type to the variable 'stypy_return_type' (line 114)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 4), 'stypy_return_type', y_127971)
    
    # ################# End of 'isposinf(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'isposinf' in the type store
    # Getting the type of 'stypy_return_type' (line 54)
    stypy_return_type_127972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_127972)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'isposinf'
    return stypy_return_type_127972

# Assigning a type to the variable 'isposinf' (line 54)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 0), 'isposinf', isposinf)

@norecursion
def isneginf(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 116)
    None_127973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 18), 'None')
    defaults = [None_127973]
    # Create a new context for function 'isneginf'
    module_type_store = module_type_store.open_function_context('isneginf', 116, 0, False)
    
    # Passed parameters checking function
    isneginf.stypy_localization = localization
    isneginf.stypy_type_of_self = None
    isneginf.stypy_type_store = module_type_store
    isneginf.stypy_function_name = 'isneginf'
    isneginf.stypy_param_names_list = ['x', 'y']
    isneginf.stypy_varargs_param_name = None
    isneginf.stypy_kwargs_param_name = None
    isneginf.stypy_call_defaults = defaults
    isneginf.stypy_call_varargs = varargs
    isneginf.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'isneginf', ['x', 'y'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'isneginf', localization, ['x', 'y'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'isneginf(...)' code ##################

    str_127974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, (-1)), 'str', '\n    Test element-wise for negative infinity, return result as bool array.\n\n    Parameters\n    ----------\n    x : array_like\n        The input array.\n    y : array_like, optional\n        A boolean array with the same shape and type as `x` to store the\n        result.\n\n    Returns\n    -------\n    y : ndarray\n        A boolean array with the same dimensions as the input.\n        If second argument is not supplied then a numpy boolean array is\n        returned with values True where the corresponding element of the\n        input is negative infinity and values False where the element of\n        the input is not negative infinity.\n\n        If a second argument is supplied the result is stored there. If the\n        type of that array is a numeric type the result is represented as\n        zeros and ones, if the type is boolean then as False and True. The\n        return value `y` is then a reference to that array.\n\n    See Also\n    --------\n    isinf, isposinf, isnan, isfinite\n\n    Notes\n    -----\n    Numpy uses the IEEE Standard for Binary Floating-Point for Arithmetic\n    (IEEE 754).\n\n    Errors result if the second argument is also supplied when x is a scalar\n    input, or if first and second arguments have different shapes.\n\n    Examples\n    --------\n    >>> np.isneginf(np.NINF)\n    array(True, dtype=bool)\n    >>> np.isneginf(np.inf)\n    array(False, dtype=bool)\n    >>> np.isneginf(np.PINF)\n    array(False, dtype=bool)\n    >>> np.isneginf([-np.inf, 0., np.inf])\n    array([ True, False, False], dtype=bool)\n\n    >>> x = np.array([-np.inf, 0., np.inf])\n    >>> y = np.array([2, 2, 2])\n    >>> np.isneginf(x, y)\n    array([1, 0, 0])\n    >>> y\n    array([1, 0, 0])\n\n    ')
    
    # Type idiom detected: calculating its left and rigth part (line 173)
    # Getting the type of 'y' (line 173)
    y_127975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 7), 'y')
    # Getting the type of 'None' (line 173)
    None_127976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 12), 'None')
    
    (may_be_127977, more_types_in_union_127978) = may_be_none(y_127975, None_127976)

    if may_be_127977:

        if more_types_in_union_127978:
            # Runtime conditional SSA (line 173)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 174):
        
        # Call to asarray(...): (line 174)
        # Processing the call arguments (line 174)
        # Getting the type of 'x' (line 174)
        x_127981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 23), 'x', False)
        # Processing the call keyword arguments (line 174)
        kwargs_127982 = {}
        # Getting the type of 'nx' (line 174)
        nx_127979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 12), 'nx', False)
        # Obtaining the member 'asarray' of a type (line 174)
        asarray_127980 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 12), nx_127979, 'asarray')
        # Calling asarray(args, kwargs) (line 174)
        asarray_call_result_127983 = invoke(stypy.reporting.localization.Localization(__file__, 174, 12), asarray_127980, *[x_127981], **kwargs_127982)
        
        # Assigning a type to the variable 'x' (line 174)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'x', asarray_call_result_127983)
        
        # Assigning a Call to a Name (line 175):
        
        # Call to empty(...): (line 175)
        # Processing the call arguments (line 175)
        # Getting the type of 'x' (line 175)
        x_127986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 21), 'x', False)
        # Obtaining the member 'shape' of a type (line 175)
        shape_127987 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 21), x_127986, 'shape')
        # Processing the call keyword arguments (line 175)
        # Getting the type of 'nx' (line 175)
        nx_127988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 36), 'nx', False)
        # Obtaining the member 'bool_' of a type (line 175)
        bool__127989 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 36), nx_127988, 'bool_')
        keyword_127990 = bool__127989
        kwargs_127991 = {'dtype': keyword_127990}
        # Getting the type of 'nx' (line 175)
        nx_127984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 12), 'nx', False)
        # Obtaining the member 'empty' of a type (line 175)
        empty_127985 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 12), nx_127984, 'empty')
        # Calling empty(args, kwargs) (line 175)
        empty_call_result_127992 = invoke(stypy.reporting.localization.Localization(__file__, 175, 12), empty_127985, *[shape_127987], **kwargs_127991)
        
        # Assigning a type to the variable 'y' (line 175)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'y', empty_call_result_127992)

        if more_types_in_union_127978:
            # SSA join for if statement (line 173)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Call to logical_and(...): (line 176)
    # Processing the call arguments (line 176)
    
    # Call to isinf(...): (line 176)
    # Processing the call arguments (line 176)
    # Getting the type of 'x' (line 176)
    x_127997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 28), 'x', False)
    # Processing the call keyword arguments (line 176)
    kwargs_127998 = {}
    # Getting the type of 'nx' (line 176)
    nx_127995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 19), 'nx', False)
    # Obtaining the member 'isinf' of a type (line 176)
    isinf_127996 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 19), nx_127995, 'isinf')
    # Calling isinf(args, kwargs) (line 176)
    isinf_call_result_127999 = invoke(stypy.reporting.localization.Localization(__file__, 176, 19), isinf_127996, *[x_127997], **kwargs_127998)
    
    
    # Call to signbit(...): (line 176)
    # Processing the call arguments (line 176)
    # Getting the type of 'x' (line 176)
    x_128002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 43), 'x', False)
    # Processing the call keyword arguments (line 176)
    kwargs_128003 = {}
    # Getting the type of 'nx' (line 176)
    nx_128000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 32), 'nx', False)
    # Obtaining the member 'signbit' of a type (line 176)
    signbit_128001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 32), nx_128000, 'signbit')
    # Calling signbit(args, kwargs) (line 176)
    signbit_call_result_128004 = invoke(stypy.reporting.localization.Localization(__file__, 176, 32), signbit_128001, *[x_128002], **kwargs_128003)
    
    # Getting the type of 'y' (line 176)
    y_128005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 47), 'y', False)
    # Processing the call keyword arguments (line 176)
    kwargs_128006 = {}
    # Getting the type of 'nx' (line 176)
    nx_127993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 4), 'nx', False)
    # Obtaining the member 'logical_and' of a type (line 176)
    logical_and_127994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 4), nx_127993, 'logical_and')
    # Calling logical_and(args, kwargs) (line 176)
    logical_and_call_result_128007 = invoke(stypy.reporting.localization.Localization(__file__, 176, 4), logical_and_127994, *[isinf_call_result_127999, signbit_call_result_128004, y_128005], **kwargs_128006)
    
    # Getting the type of 'y' (line 177)
    y_128008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 11), 'y')
    # Assigning a type to the variable 'stypy_return_type' (line 177)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 4), 'stypy_return_type', y_128008)
    
    # ################# End of 'isneginf(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'isneginf' in the type store
    # Getting the type of 'stypy_return_type' (line 116)
    stypy_return_type_128009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_128009)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'isneginf'
    return stypy_return_type_128009

# Assigning a type to the variable 'isneginf' (line 116)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 0), 'isneginf', isneginf)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
