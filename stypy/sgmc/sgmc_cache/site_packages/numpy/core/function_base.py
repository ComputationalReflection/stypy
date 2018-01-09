
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, absolute_import, print_function
2: 
3: __all__ = ['logspace', 'linspace']
4: 
5: from . import numeric as _nx
6: from .numeric import result_type, NaN, shares_memory, MAY_SHARE_BOUNDS, TooHardError
7: 
8: 
9: def linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None):
10:     '''
11:     Return evenly spaced numbers over a specified interval.
12: 
13:     Returns `num` evenly spaced samples, calculated over the
14:     interval [`start`, `stop`].
15: 
16:     The endpoint of the interval can optionally be excluded.
17: 
18:     Parameters
19:     ----------
20:     start : scalar
21:         The starting value of the sequence.
22:     stop : scalar
23:         The end value of the sequence, unless `endpoint` is set to False.
24:         In that case, the sequence consists of all but the last of ``num + 1``
25:         evenly spaced samples, so that `stop` is excluded.  Note that the step
26:         size changes when `endpoint` is False.
27:     num : int, optional
28:         Number of samples to generate. Default is 50. Must be non-negative.
29:     endpoint : bool, optional
30:         If True, `stop` is the last sample. Otherwise, it is not included.
31:         Default is True.
32:     retstep : bool, optional
33:         If True, return (`samples`, `step`), where `step` is the spacing
34:         between samples.
35:     dtype : dtype, optional
36:         The type of the output array.  If `dtype` is not given, infer the data
37:         type from the other input arguments.
38: 
39:         .. versionadded:: 1.9.0
40: 
41:     Returns
42:     -------
43:     samples : ndarray
44:         There are `num` equally spaced samples in the closed interval
45:         ``[start, stop]`` or the half-open interval ``[start, stop)``
46:         (depending on whether `endpoint` is True or False).
47:     step : float
48:         Only returned if `retstep` is True
49: 
50:         Size of spacing between samples.
51: 
52: 
53:     See Also
54:     --------
55:     arange : Similar to `linspace`, but uses a step size (instead of the
56:              number of samples).
57:     logspace : Samples uniformly distributed in log space.
58: 
59:     Examples
60:     --------
61:     >>> np.linspace(2.0, 3.0, num=5)
62:         array([ 2.  ,  2.25,  2.5 ,  2.75,  3.  ])
63:     >>> np.linspace(2.0, 3.0, num=5, endpoint=False)
64:         array([ 2. ,  2.2,  2.4,  2.6,  2.8])
65:     >>> np.linspace(2.0, 3.0, num=5, retstep=True)
66:         (array([ 2.  ,  2.25,  2.5 ,  2.75,  3.  ]), 0.25)
67: 
68:     Graphical illustration:
69: 
70:     >>> import matplotlib.pyplot as plt
71:     >>> N = 8
72:     >>> y = np.zeros(N)
73:     >>> x1 = np.linspace(0, 10, N, endpoint=True)
74:     >>> x2 = np.linspace(0, 10, N, endpoint=False)
75:     >>> plt.plot(x1, y, 'o')
76:     [<matplotlib.lines.Line2D object at 0x...>]
77:     >>> plt.plot(x2, y + 0.5, 'o')
78:     [<matplotlib.lines.Line2D object at 0x...>]
79:     >>> plt.ylim([-0.5, 1])
80:     (-0.5, 1)
81:     >>> plt.show()
82: 
83:     '''
84:     num = int(num)
85:     if num < 0:
86:         raise ValueError("Number of samples, %s, must be non-negative." % num)
87:     div = (num - 1) if endpoint else num
88: 
89:     # Convert float/complex array scalars to float, gh-3504
90:     start = start * 1.
91:     stop = stop * 1.
92: 
93:     dt = result_type(start, stop, float(num))
94:     if dtype is None:
95:         dtype = dt
96: 
97:     y = _nx.arange(0, num, dtype=dt)
98: 
99:     delta = stop - start
100:     if num > 1:
101:         step = delta / div
102:         if step == 0:
103:             # Special handling for denormal numbers, gh-5437
104:             y /= div
105:             y = y * delta
106:         else:
107:             # One might be tempted to use faster, in-place multiplication here,
108:             # but this prevents step from overriding what class is produced,
109:             # and thus prevents, e.g., use of Quantities; see gh-7142.
110:             y = y * step
111:     else:
112:         # 0 and 1 item long sequences have an undefined step
113:         step = NaN
114:         # Multiply with delta to allow possible override of output class.
115:         y = y * delta
116: 
117:     y += start
118: 
119:     if endpoint and num > 1:
120:         y[-1] = stop
121: 
122:     if retstep:
123:         return y.astype(dtype, copy=False), step
124:     else:
125:         return y.astype(dtype, copy=False)
126: 
127: 
128: def logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None):
129:     '''
130:     Return numbers spaced evenly on a log scale.
131: 
132:     In linear space, the sequence starts at ``base ** start``
133:     (`base` to the power of `start`) and ends with ``base ** stop``
134:     (see `endpoint` below).
135: 
136:     Parameters
137:     ----------
138:     start : float
139:         ``base ** start`` is the starting value of the sequence.
140:     stop : float
141:         ``base ** stop`` is the final value of the sequence, unless `endpoint`
142:         is False.  In that case, ``num + 1`` values are spaced over the
143:         interval in log-space, of which all but the last (a sequence of
144:         length ``num``) are returned.
145:     num : integer, optional
146:         Number of samples to generate.  Default is 50.
147:     endpoint : boolean, optional
148:         If true, `stop` is the last sample. Otherwise, it is not included.
149:         Default is True.
150:     base : float, optional
151:         The base of the log space. The step size between the elements in
152:         ``ln(samples) / ln(base)`` (or ``log_base(samples)``) is uniform.
153:         Default is 10.0.
154:     dtype : dtype
155:         The type of the output array.  If `dtype` is not given, infer the data
156:         type from the other input arguments.
157: 
158:     Returns
159:     -------
160:     samples : ndarray
161:         `num` samples, equally spaced on a log scale.
162: 
163:     See Also
164:     --------
165:     arange : Similar to linspace, with the step size specified instead of the
166:              number of samples. Note that, when used with a float endpoint, the
167:              endpoint may or may not be included.
168:     linspace : Similar to logspace, but with the samples uniformly distributed
169:                in linear space, instead of log space.
170: 
171:     Notes
172:     -----
173:     Logspace is equivalent to the code
174: 
175:     >>> y = np.linspace(start, stop, num=num, endpoint=endpoint)
176:     ... # doctest: +SKIP
177:     >>> power(base, y).astype(dtype)
178:     ... # doctest: +SKIP
179: 
180:     Examples
181:     --------
182:     >>> np.logspace(2.0, 3.0, num=4)
183:         array([  100.        ,   215.443469  ,   464.15888336,  1000.        ])
184:     >>> np.logspace(2.0, 3.0, num=4, endpoint=False)
185:         array([ 100.        ,  177.827941  ,  316.22776602,  562.34132519])
186:     >>> np.logspace(2.0, 3.0, num=4, base=2.0)
187:         array([ 4.        ,  5.0396842 ,  6.34960421,  8.        ])
188: 
189:     Graphical illustration:
190: 
191:     >>> import matplotlib.pyplot as plt
192:     >>> N = 10
193:     >>> x1 = np.logspace(0.1, 1, N, endpoint=True)
194:     >>> x2 = np.logspace(0.1, 1, N, endpoint=False)
195:     >>> y = np.zeros(N)
196:     >>> plt.plot(x1, y, 'o')
197:     [<matplotlib.lines.Line2D object at 0x...>]
198:     >>> plt.plot(x2, y + 0.5, 'o')
199:     [<matplotlib.lines.Line2D object at 0x...>]
200:     >>> plt.ylim([-0.5, 1])
201:     (-0.5, 1)
202:     >>> plt.show()
203: 
204:     '''
205:     y = linspace(start, stop, num=num, endpoint=endpoint)
206:     if dtype is None:
207:         return _nx.power(base, y)
208:     return _nx.power(base, y).astype(dtype)
209: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a List to a Name (line 3):
__all__ = ['logspace', 'linspace']
module_type_store.set_exportable_members(['logspace', 'linspace'])

# Obtaining an instance of the builtin type 'list' (line 3)
list_5253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 3)
# Adding element type (line 3)
str_5254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 11), 'str', 'logspace')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3, 10), list_5253, str_5254)
# Adding element type (line 3)
str_5255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 23), 'str', 'linspace')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3, 10), list_5253, str_5255)

# Assigning a type to the variable '__all__' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), '__all__', list_5253)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from numpy.core import _nx' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/core/')
import_5256 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy.core')

if (type(import_5256) is not StypyTypeError):

    if (import_5256 != 'pyd_module'):
        __import__(import_5256)
        sys_modules_5257 = sys.modules[import_5256]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy.core', sys_modules_5257.module_type_store, module_type_store, ['numeric'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_5257, sys_modules_5257.module_type_store, module_type_store)
    else:
        from numpy.core import numeric as _nx

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy.core', None, module_type_store, ['numeric'], [_nx])

else:
    # Assigning a type to the variable 'numpy.core' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy.core', import_5256)

# Adding an alias
module_type_store.add_alias('_nx', 'numeric')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/core/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from numpy.core.numeric import result_type, NaN, shares_memory, MAY_SHARE_BOUNDS, TooHardError' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/core/')
import_5258 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.core.numeric')

if (type(import_5258) is not StypyTypeError):

    if (import_5258 != 'pyd_module'):
        __import__(import_5258)
        sys_modules_5259 = sys.modules[import_5258]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.core.numeric', sys_modules_5259.module_type_store, module_type_store, ['result_type', 'NaN', 'shares_memory', 'MAY_SHARE_BOUNDS', 'TooHardError'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_5259, sys_modules_5259.module_type_store, module_type_store)
    else:
        from numpy.core.numeric import result_type, NaN, shares_memory, MAY_SHARE_BOUNDS, TooHardError

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.core.numeric', None, module_type_store, ['result_type', 'NaN', 'shares_memory', 'MAY_SHARE_BOUNDS', 'TooHardError'], [result_type, NaN, shares_memory, MAY_SHARE_BOUNDS, TooHardError])

else:
    # Assigning a type to the variable 'numpy.core.numeric' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.core.numeric', import_5258)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/core/')


@norecursion
def linspace(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_5260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 30), 'int')
    # Getting the type of 'True' (line 9)
    True_5261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 43), 'True')
    # Getting the type of 'False' (line 9)
    False_5262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 57), 'False')
    # Getting the type of 'None' (line 9)
    None_5263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 70), 'None')
    defaults = [int_5260, True_5261, False_5262, None_5263]
    # Create a new context for function 'linspace'
    module_type_store = module_type_store.open_function_context('linspace', 9, 0, False)
    
    # Passed parameters checking function
    linspace.stypy_localization = localization
    linspace.stypy_type_of_self = None
    linspace.stypy_type_store = module_type_store
    linspace.stypy_function_name = 'linspace'
    linspace.stypy_param_names_list = ['start', 'stop', 'num', 'endpoint', 'retstep', 'dtype']
    linspace.stypy_varargs_param_name = None
    linspace.stypy_kwargs_param_name = None
    linspace.stypy_call_defaults = defaults
    linspace.stypy_call_varargs = varargs
    linspace.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'linspace', ['start', 'stop', 'num', 'endpoint', 'retstep', 'dtype'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'linspace', localization, ['start', 'stop', 'num', 'endpoint', 'retstep', 'dtype'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'linspace(...)' code ##################

    str_5264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, (-1)), 'str', "\n    Return evenly spaced numbers over a specified interval.\n\n    Returns `num` evenly spaced samples, calculated over the\n    interval [`start`, `stop`].\n\n    The endpoint of the interval can optionally be excluded.\n\n    Parameters\n    ----------\n    start : scalar\n        The starting value of the sequence.\n    stop : scalar\n        The end value of the sequence, unless `endpoint` is set to False.\n        In that case, the sequence consists of all but the last of ``num + 1``\n        evenly spaced samples, so that `stop` is excluded.  Note that the step\n        size changes when `endpoint` is False.\n    num : int, optional\n        Number of samples to generate. Default is 50. Must be non-negative.\n    endpoint : bool, optional\n        If True, `stop` is the last sample. Otherwise, it is not included.\n        Default is True.\n    retstep : bool, optional\n        If True, return (`samples`, `step`), where `step` is the spacing\n        between samples.\n    dtype : dtype, optional\n        The type of the output array.  If `dtype` is not given, infer the data\n        type from the other input arguments.\n\n        .. versionadded:: 1.9.0\n\n    Returns\n    -------\n    samples : ndarray\n        There are `num` equally spaced samples in the closed interval\n        ``[start, stop]`` or the half-open interval ``[start, stop)``\n        (depending on whether `endpoint` is True or False).\n    step : float\n        Only returned if `retstep` is True\n\n        Size of spacing between samples.\n\n\n    See Also\n    --------\n    arange : Similar to `linspace`, but uses a step size (instead of the\n             number of samples).\n    logspace : Samples uniformly distributed in log space.\n\n    Examples\n    --------\n    >>> np.linspace(2.0, 3.0, num=5)\n        array([ 2.  ,  2.25,  2.5 ,  2.75,  3.  ])\n    >>> np.linspace(2.0, 3.0, num=5, endpoint=False)\n        array([ 2. ,  2.2,  2.4,  2.6,  2.8])\n    >>> np.linspace(2.0, 3.0, num=5, retstep=True)\n        (array([ 2.  ,  2.25,  2.5 ,  2.75,  3.  ]), 0.25)\n\n    Graphical illustration:\n\n    >>> import matplotlib.pyplot as plt\n    >>> N = 8\n    >>> y = np.zeros(N)\n    >>> x1 = np.linspace(0, 10, N, endpoint=True)\n    >>> x2 = np.linspace(0, 10, N, endpoint=False)\n    >>> plt.plot(x1, y, 'o')\n    [<matplotlib.lines.Line2D object at 0x...>]\n    >>> plt.plot(x2, y + 0.5, 'o')\n    [<matplotlib.lines.Line2D object at 0x...>]\n    >>> plt.ylim([-0.5, 1])\n    (-0.5, 1)\n    >>> plt.show()\n\n    ")
    
    # Assigning a Call to a Name (line 84):
    
    # Call to int(...): (line 84)
    # Processing the call arguments (line 84)
    # Getting the type of 'num' (line 84)
    num_5266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 14), 'num', False)
    # Processing the call keyword arguments (line 84)
    kwargs_5267 = {}
    # Getting the type of 'int' (line 84)
    int_5265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 10), 'int', False)
    # Calling int(args, kwargs) (line 84)
    int_call_result_5268 = invoke(stypy.reporting.localization.Localization(__file__, 84, 10), int_5265, *[num_5266], **kwargs_5267)
    
    # Assigning a type to the variable 'num' (line 84)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 4), 'num', int_call_result_5268)
    
    
    # Getting the type of 'num' (line 85)
    num_5269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 7), 'num')
    int_5270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 13), 'int')
    # Applying the binary operator '<' (line 85)
    result_lt_5271 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 7), '<', num_5269, int_5270)
    
    # Testing the type of an if condition (line 85)
    if_condition_5272 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 85, 4), result_lt_5271)
    # Assigning a type to the variable 'if_condition_5272' (line 85)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'if_condition_5272', if_condition_5272)
    # SSA begins for if statement (line 85)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 86)
    # Processing the call arguments (line 86)
    str_5274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 25), 'str', 'Number of samples, %s, must be non-negative.')
    # Getting the type of 'num' (line 86)
    num_5275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 74), 'num', False)
    # Applying the binary operator '%' (line 86)
    result_mod_5276 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 25), '%', str_5274, num_5275)
    
    # Processing the call keyword arguments (line 86)
    kwargs_5277 = {}
    # Getting the type of 'ValueError' (line 86)
    ValueError_5273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 86)
    ValueError_call_result_5278 = invoke(stypy.reporting.localization.Localization(__file__, 86, 14), ValueError_5273, *[result_mod_5276], **kwargs_5277)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 86, 8), ValueError_call_result_5278, 'raise parameter', BaseException)
    # SSA join for if statement (line 85)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a IfExp to a Name (line 87):
    
    # Getting the type of 'endpoint' (line 87)
    endpoint_5279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 23), 'endpoint')
    # Testing the type of an if expression (line 87)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 87, 10), endpoint_5279)
    # SSA begins for if expression (line 87)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    # Getting the type of 'num' (line 87)
    num_5280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 11), 'num')
    int_5281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 17), 'int')
    # Applying the binary operator '-' (line 87)
    result_sub_5282 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 11), '-', num_5280, int_5281)
    
    # SSA branch for the else part of an if expression (line 87)
    module_type_store.open_ssa_branch('if expression else')
    # Getting the type of 'num' (line 87)
    num_5283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 37), 'num')
    # SSA join for if expression (line 87)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_5284 = union_type.UnionType.add(result_sub_5282, num_5283)
    
    # Assigning a type to the variable 'div' (line 87)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 4), 'div', if_exp_5284)
    
    # Assigning a BinOp to a Name (line 90):
    # Getting the type of 'start' (line 90)
    start_5285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 12), 'start')
    float_5286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 20), 'float')
    # Applying the binary operator '*' (line 90)
    result_mul_5287 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 12), '*', start_5285, float_5286)
    
    # Assigning a type to the variable 'start' (line 90)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 4), 'start', result_mul_5287)
    
    # Assigning a BinOp to a Name (line 91):
    # Getting the type of 'stop' (line 91)
    stop_5288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 11), 'stop')
    float_5289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 18), 'float')
    # Applying the binary operator '*' (line 91)
    result_mul_5290 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 11), '*', stop_5288, float_5289)
    
    # Assigning a type to the variable 'stop' (line 91)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 4), 'stop', result_mul_5290)
    
    # Assigning a Call to a Name (line 93):
    
    # Call to result_type(...): (line 93)
    # Processing the call arguments (line 93)
    # Getting the type of 'start' (line 93)
    start_5292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 21), 'start', False)
    # Getting the type of 'stop' (line 93)
    stop_5293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 28), 'stop', False)
    
    # Call to float(...): (line 93)
    # Processing the call arguments (line 93)
    # Getting the type of 'num' (line 93)
    num_5295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 40), 'num', False)
    # Processing the call keyword arguments (line 93)
    kwargs_5296 = {}
    # Getting the type of 'float' (line 93)
    float_5294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 34), 'float', False)
    # Calling float(args, kwargs) (line 93)
    float_call_result_5297 = invoke(stypy.reporting.localization.Localization(__file__, 93, 34), float_5294, *[num_5295], **kwargs_5296)
    
    # Processing the call keyword arguments (line 93)
    kwargs_5298 = {}
    # Getting the type of 'result_type' (line 93)
    result_type_5291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 9), 'result_type', False)
    # Calling result_type(args, kwargs) (line 93)
    result_type_call_result_5299 = invoke(stypy.reporting.localization.Localization(__file__, 93, 9), result_type_5291, *[start_5292, stop_5293, float_call_result_5297], **kwargs_5298)
    
    # Assigning a type to the variable 'dt' (line 93)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'dt', result_type_call_result_5299)
    
    # Type idiom detected: calculating its left and rigth part (line 94)
    # Getting the type of 'dtype' (line 94)
    dtype_5300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 7), 'dtype')
    # Getting the type of 'None' (line 94)
    None_5301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 16), 'None')
    
    (may_be_5302, more_types_in_union_5303) = may_be_none(dtype_5300, None_5301)

    if may_be_5302:

        if more_types_in_union_5303:
            # Runtime conditional SSA (line 94)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Name to a Name (line 95):
        # Getting the type of 'dt' (line 95)
        dt_5304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 16), 'dt')
        # Assigning a type to the variable 'dtype' (line 95)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'dtype', dt_5304)

        if more_types_in_union_5303:
            # SSA join for if statement (line 94)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 97):
    
    # Call to arange(...): (line 97)
    # Processing the call arguments (line 97)
    int_5307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 19), 'int')
    # Getting the type of 'num' (line 97)
    num_5308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 22), 'num', False)
    # Processing the call keyword arguments (line 97)
    # Getting the type of 'dt' (line 97)
    dt_5309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 33), 'dt', False)
    keyword_5310 = dt_5309
    kwargs_5311 = {'dtype': keyword_5310}
    # Getting the type of '_nx' (line 97)
    _nx_5305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), '_nx', False)
    # Obtaining the member 'arange' of a type (line 97)
    arange_5306 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 8), _nx_5305, 'arange')
    # Calling arange(args, kwargs) (line 97)
    arange_call_result_5312 = invoke(stypy.reporting.localization.Localization(__file__, 97, 8), arange_5306, *[int_5307, num_5308], **kwargs_5311)
    
    # Assigning a type to the variable 'y' (line 97)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 4), 'y', arange_call_result_5312)
    
    # Assigning a BinOp to a Name (line 99):
    # Getting the type of 'stop' (line 99)
    stop_5313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 12), 'stop')
    # Getting the type of 'start' (line 99)
    start_5314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 19), 'start')
    # Applying the binary operator '-' (line 99)
    result_sub_5315 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 12), '-', stop_5313, start_5314)
    
    # Assigning a type to the variable 'delta' (line 99)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'delta', result_sub_5315)
    
    
    # Getting the type of 'num' (line 100)
    num_5316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 7), 'num')
    int_5317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 13), 'int')
    # Applying the binary operator '>' (line 100)
    result_gt_5318 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 7), '>', num_5316, int_5317)
    
    # Testing the type of an if condition (line 100)
    if_condition_5319 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 100, 4), result_gt_5318)
    # Assigning a type to the variable 'if_condition_5319' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'if_condition_5319', if_condition_5319)
    # SSA begins for if statement (line 100)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 101):
    # Getting the type of 'delta' (line 101)
    delta_5320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 15), 'delta')
    # Getting the type of 'div' (line 101)
    div_5321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 23), 'div')
    # Applying the binary operator 'div' (line 101)
    result_div_5322 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 15), 'div', delta_5320, div_5321)
    
    # Assigning a type to the variable 'step' (line 101)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'step', result_div_5322)
    
    
    # Getting the type of 'step' (line 102)
    step_5323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 11), 'step')
    int_5324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 19), 'int')
    # Applying the binary operator '==' (line 102)
    result_eq_5325 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 11), '==', step_5323, int_5324)
    
    # Testing the type of an if condition (line 102)
    if_condition_5326 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 102, 8), result_eq_5325)
    # Assigning a type to the variable 'if_condition_5326' (line 102)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'if_condition_5326', if_condition_5326)
    # SSA begins for if statement (line 102)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'y' (line 104)
    y_5327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 12), 'y')
    # Getting the type of 'div' (line 104)
    div_5328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 17), 'div')
    # Applying the binary operator 'div=' (line 104)
    result_div_5329 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 12), 'div=', y_5327, div_5328)
    # Assigning a type to the variable 'y' (line 104)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 12), 'y', result_div_5329)
    
    
    # Assigning a BinOp to a Name (line 105):
    # Getting the type of 'y' (line 105)
    y_5330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 16), 'y')
    # Getting the type of 'delta' (line 105)
    delta_5331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 20), 'delta')
    # Applying the binary operator '*' (line 105)
    result_mul_5332 = python_operator(stypy.reporting.localization.Localization(__file__, 105, 16), '*', y_5330, delta_5331)
    
    # Assigning a type to the variable 'y' (line 105)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 12), 'y', result_mul_5332)
    # SSA branch for the else part of an if statement (line 102)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a BinOp to a Name (line 110):
    # Getting the type of 'y' (line 110)
    y_5333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 16), 'y')
    # Getting the type of 'step' (line 110)
    step_5334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 20), 'step')
    # Applying the binary operator '*' (line 110)
    result_mul_5335 = python_operator(stypy.reporting.localization.Localization(__file__, 110, 16), '*', y_5333, step_5334)
    
    # Assigning a type to the variable 'y' (line 110)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 12), 'y', result_mul_5335)
    # SSA join for if statement (line 102)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 100)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Name (line 113):
    # Getting the type of 'NaN' (line 113)
    NaN_5336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 15), 'NaN')
    # Assigning a type to the variable 'step' (line 113)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'step', NaN_5336)
    
    # Assigning a BinOp to a Name (line 115):
    # Getting the type of 'y' (line 115)
    y_5337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 12), 'y')
    # Getting the type of 'delta' (line 115)
    delta_5338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 16), 'delta')
    # Applying the binary operator '*' (line 115)
    result_mul_5339 = python_operator(stypy.reporting.localization.Localization(__file__, 115, 12), '*', y_5337, delta_5338)
    
    # Assigning a type to the variable 'y' (line 115)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 8), 'y', result_mul_5339)
    # SSA join for if statement (line 100)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'y' (line 117)
    y_5340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 4), 'y')
    # Getting the type of 'start' (line 117)
    start_5341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 9), 'start')
    # Applying the binary operator '+=' (line 117)
    result_iadd_5342 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 4), '+=', y_5340, start_5341)
    # Assigning a type to the variable 'y' (line 117)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 4), 'y', result_iadd_5342)
    
    
    
    # Evaluating a boolean operation
    # Getting the type of 'endpoint' (line 119)
    endpoint_5343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 7), 'endpoint')
    
    # Getting the type of 'num' (line 119)
    num_5344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 20), 'num')
    int_5345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 26), 'int')
    # Applying the binary operator '>' (line 119)
    result_gt_5346 = python_operator(stypy.reporting.localization.Localization(__file__, 119, 20), '>', num_5344, int_5345)
    
    # Applying the binary operator 'and' (line 119)
    result_and_keyword_5347 = python_operator(stypy.reporting.localization.Localization(__file__, 119, 7), 'and', endpoint_5343, result_gt_5346)
    
    # Testing the type of an if condition (line 119)
    if_condition_5348 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 119, 4), result_and_keyword_5347)
    # Assigning a type to the variable 'if_condition_5348' (line 119)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 4), 'if_condition_5348', if_condition_5348)
    # SSA begins for if statement (line 119)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Subscript (line 120):
    # Getting the type of 'stop' (line 120)
    stop_5349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 16), 'stop')
    # Getting the type of 'y' (line 120)
    y_5350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'y')
    int_5351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 10), 'int')
    # Storing an element on a container (line 120)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 8), y_5350, (int_5351, stop_5349))
    # SSA join for if statement (line 119)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'retstep' (line 122)
    retstep_5352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 7), 'retstep')
    # Testing the type of an if condition (line 122)
    if_condition_5353 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 122, 4), retstep_5352)
    # Assigning a type to the variable 'if_condition_5353' (line 122)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 4), 'if_condition_5353', if_condition_5353)
    # SSA begins for if statement (line 122)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'tuple' (line 123)
    tuple_5354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 123)
    # Adding element type (line 123)
    
    # Call to astype(...): (line 123)
    # Processing the call arguments (line 123)
    # Getting the type of 'dtype' (line 123)
    dtype_5357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 24), 'dtype', False)
    # Processing the call keyword arguments (line 123)
    # Getting the type of 'False' (line 123)
    False_5358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 36), 'False', False)
    keyword_5359 = False_5358
    kwargs_5360 = {'copy': keyword_5359}
    # Getting the type of 'y' (line 123)
    y_5355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 15), 'y', False)
    # Obtaining the member 'astype' of a type (line 123)
    astype_5356 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 15), y_5355, 'astype')
    # Calling astype(args, kwargs) (line 123)
    astype_call_result_5361 = invoke(stypy.reporting.localization.Localization(__file__, 123, 15), astype_5356, *[dtype_5357], **kwargs_5360)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 15), tuple_5354, astype_call_result_5361)
    # Adding element type (line 123)
    # Getting the type of 'step' (line 123)
    step_5362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 44), 'step')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 15), tuple_5354, step_5362)
    
    # Assigning a type to the variable 'stypy_return_type' (line 123)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'stypy_return_type', tuple_5354)
    # SSA branch for the else part of an if statement (line 122)
    module_type_store.open_ssa_branch('else')
    
    # Call to astype(...): (line 125)
    # Processing the call arguments (line 125)
    # Getting the type of 'dtype' (line 125)
    dtype_5365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 24), 'dtype', False)
    # Processing the call keyword arguments (line 125)
    # Getting the type of 'False' (line 125)
    False_5366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 36), 'False', False)
    keyword_5367 = False_5366
    kwargs_5368 = {'copy': keyword_5367}
    # Getting the type of 'y' (line 125)
    y_5363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 15), 'y', False)
    # Obtaining the member 'astype' of a type (line 125)
    astype_5364 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 15), y_5363, 'astype')
    # Calling astype(args, kwargs) (line 125)
    astype_call_result_5369 = invoke(stypy.reporting.localization.Localization(__file__, 125, 15), astype_5364, *[dtype_5365], **kwargs_5368)
    
    # Assigning a type to the variable 'stypy_return_type' (line 125)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'stypy_return_type', astype_call_result_5369)
    # SSA join for if statement (line 122)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'linspace(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'linspace' in the type store
    # Getting the type of 'stypy_return_type' (line 9)
    stypy_return_type_5370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_5370)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'linspace'
    return stypy_return_type_5370

# Assigning a type to the variable 'linspace' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'linspace', linspace)

@norecursion
def logspace(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_5371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 30), 'int')
    # Getting the type of 'True' (line 128)
    True_5372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 43), 'True')
    float_5373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 54), 'float')
    # Getting the type of 'None' (line 128)
    None_5374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 66), 'None')
    defaults = [int_5371, True_5372, float_5373, None_5374]
    # Create a new context for function 'logspace'
    module_type_store = module_type_store.open_function_context('logspace', 128, 0, False)
    
    # Passed parameters checking function
    logspace.stypy_localization = localization
    logspace.stypy_type_of_self = None
    logspace.stypy_type_store = module_type_store
    logspace.stypy_function_name = 'logspace'
    logspace.stypy_param_names_list = ['start', 'stop', 'num', 'endpoint', 'base', 'dtype']
    logspace.stypy_varargs_param_name = None
    logspace.stypy_kwargs_param_name = None
    logspace.stypy_call_defaults = defaults
    logspace.stypy_call_varargs = varargs
    logspace.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'logspace', ['start', 'stop', 'num', 'endpoint', 'base', 'dtype'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'logspace', localization, ['start', 'stop', 'num', 'endpoint', 'base', 'dtype'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'logspace(...)' code ##################

    str_5375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, (-1)), 'str', "\n    Return numbers spaced evenly on a log scale.\n\n    In linear space, the sequence starts at ``base ** start``\n    (`base` to the power of `start`) and ends with ``base ** stop``\n    (see `endpoint` below).\n\n    Parameters\n    ----------\n    start : float\n        ``base ** start`` is the starting value of the sequence.\n    stop : float\n        ``base ** stop`` is the final value of the sequence, unless `endpoint`\n        is False.  In that case, ``num + 1`` values are spaced over the\n        interval in log-space, of which all but the last (a sequence of\n        length ``num``) are returned.\n    num : integer, optional\n        Number of samples to generate.  Default is 50.\n    endpoint : boolean, optional\n        If true, `stop` is the last sample. Otherwise, it is not included.\n        Default is True.\n    base : float, optional\n        The base of the log space. The step size between the elements in\n        ``ln(samples) / ln(base)`` (or ``log_base(samples)``) is uniform.\n        Default is 10.0.\n    dtype : dtype\n        The type of the output array.  If `dtype` is not given, infer the data\n        type from the other input arguments.\n\n    Returns\n    -------\n    samples : ndarray\n        `num` samples, equally spaced on a log scale.\n\n    See Also\n    --------\n    arange : Similar to linspace, with the step size specified instead of the\n             number of samples. Note that, when used with a float endpoint, the\n             endpoint may or may not be included.\n    linspace : Similar to logspace, but with the samples uniformly distributed\n               in linear space, instead of log space.\n\n    Notes\n    -----\n    Logspace is equivalent to the code\n\n    >>> y = np.linspace(start, stop, num=num, endpoint=endpoint)\n    ... # doctest: +SKIP\n    >>> power(base, y).astype(dtype)\n    ... # doctest: +SKIP\n\n    Examples\n    --------\n    >>> np.logspace(2.0, 3.0, num=4)\n        array([  100.        ,   215.443469  ,   464.15888336,  1000.        ])\n    >>> np.logspace(2.0, 3.0, num=4, endpoint=False)\n        array([ 100.        ,  177.827941  ,  316.22776602,  562.34132519])\n    >>> np.logspace(2.0, 3.0, num=4, base=2.0)\n        array([ 4.        ,  5.0396842 ,  6.34960421,  8.        ])\n\n    Graphical illustration:\n\n    >>> import matplotlib.pyplot as plt\n    >>> N = 10\n    >>> x1 = np.logspace(0.1, 1, N, endpoint=True)\n    >>> x2 = np.logspace(0.1, 1, N, endpoint=False)\n    >>> y = np.zeros(N)\n    >>> plt.plot(x1, y, 'o')\n    [<matplotlib.lines.Line2D object at 0x...>]\n    >>> plt.plot(x2, y + 0.5, 'o')\n    [<matplotlib.lines.Line2D object at 0x...>]\n    >>> plt.ylim([-0.5, 1])\n    (-0.5, 1)\n    >>> plt.show()\n\n    ")
    
    # Assigning a Call to a Name (line 205):
    
    # Call to linspace(...): (line 205)
    # Processing the call arguments (line 205)
    # Getting the type of 'start' (line 205)
    start_5377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 17), 'start', False)
    # Getting the type of 'stop' (line 205)
    stop_5378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 24), 'stop', False)
    # Processing the call keyword arguments (line 205)
    # Getting the type of 'num' (line 205)
    num_5379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 34), 'num', False)
    keyword_5380 = num_5379
    # Getting the type of 'endpoint' (line 205)
    endpoint_5381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 48), 'endpoint', False)
    keyword_5382 = endpoint_5381
    kwargs_5383 = {'num': keyword_5380, 'endpoint': keyword_5382}
    # Getting the type of 'linspace' (line 205)
    linspace_5376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 8), 'linspace', False)
    # Calling linspace(args, kwargs) (line 205)
    linspace_call_result_5384 = invoke(stypy.reporting.localization.Localization(__file__, 205, 8), linspace_5376, *[start_5377, stop_5378], **kwargs_5383)
    
    # Assigning a type to the variable 'y' (line 205)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 4), 'y', linspace_call_result_5384)
    
    # Type idiom detected: calculating its left and rigth part (line 206)
    # Getting the type of 'dtype' (line 206)
    dtype_5385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 7), 'dtype')
    # Getting the type of 'None' (line 206)
    None_5386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 16), 'None')
    
    (may_be_5387, more_types_in_union_5388) = may_be_none(dtype_5385, None_5386)

    if may_be_5387:

        if more_types_in_union_5388:
            # Runtime conditional SSA (line 206)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to power(...): (line 207)
        # Processing the call arguments (line 207)
        # Getting the type of 'base' (line 207)
        base_5391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 25), 'base', False)
        # Getting the type of 'y' (line 207)
        y_5392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 31), 'y', False)
        # Processing the call keyword arguments (line 207)
        kwargs_5393 = {}
        # Getting the type of '_nx' (line 207)
        _nx_5389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 15), '_nx', False)
        # Obtaining the member 'power' of a type (line 207)
        power_5390 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 15), _nx_5389, 'power')
        # Calling power(args, kwargs) (line 207)
        power_call_result_5394 = invoke(stypy.reporting.localization.Localization(__file__, 207, 15), power_5390, *[base_5391, y_5392], **kwargs_5393)
        
        # Assigning a type to the variable 'stypy_return_type' (line 207)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 8), 'stypy_return_type', power_call_result_5394)

        if more_types_in_union_5388:
            # SSA join for if statement (line 206)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Call to astype(...): (line 208)
    # Processing the call arguments (line 208)
    # Getting the type of 'dtype' (line 208)
    dtype_5402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 37), 'dtype', False)
    # Processing the call keyword arguments (line 208)
    kwargs_5403 = {}
    
    # Call to power(...): (line 208)
    # Processing the call arguments (line 208)
    # Getting the type of 'base' (line 208)
    base_5397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 21), 'base', False)
    # Getting the type of 'y' (line 208)
    y_5398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 27), 'y', False)
    # Processing the call keyword arguments (line 208)
    kwargs_5399 = {}
    # Getting the type of '_nx' (line 208)
    _nx_5395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 11), '_nx', False)
    # Obtaining the member 'power' of a type (line 208)
    power_5396 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 11), _nx_5395, 'power')
    # Calling power(args, kwargs) (line 208)
    power_call_result_5400 = invoke(stypy.reporting.localization.Localization(__file__, 208, 11), power_5396, *[base_5397, y_5398], **kwargs_5399)
    
    # Obtaining the member 'astype' of a type (line 208)
    astype_5401 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 11), power_call_result_5400, 'astype')
    # Calling astype(args, kwargs) (line 208)
    astype_call_result_5404 = invoke(stypy.reporting.localization.Localization(__file__, 208, 11), astype_5401, *[dtype_5402], **kwargs_5403)
    
    # Assigning a type to the variable 'stypy_return_type' (line 208)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 4), 'stypy_return_type', astype_call_result_5404)
    
    # ################# End of 'logspace(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'logspace' in the type store
    # Getting the type of 'stypy_return_type' (line 128)
    stypy_return_type_5405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_5405)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'logspace'
    return stypy_return_type_5405

# Assigning a type to the variable 'logspace' (line 128)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 0), 'logspace', logspace)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
