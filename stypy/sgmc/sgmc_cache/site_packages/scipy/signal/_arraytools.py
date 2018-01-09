
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Functions for acting on a axis of an array.
3: '''
4: from __future__ import division, print_function, absolute_import
5: 
6: import numpy as np
7: 
8: 
9: def axis_slice(a, start=None, stop=None, step=None, axis=-1):
10:     '''Take a slice along axis 'axis' from 'a'.
11: 
12:     Parameters
13:     ----------
14:     a : numpy.ndarray
15:         The array to be sliced.
16:     start, stop, step : int or None
17:         The slice parameters.
18:     axis : int, optional
19:         The axis of `a` to be sliced.
20: 
21:     Examples
22:     --------
23:     >>> a = array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
24:     >>> axis_slice(a, start=0, stop=1, axis=1)
25:     array([[1],
26:            [4],
27:            [7]])
28:     >>> axis_slice(a, start=1, axis=0)
29:     array([[4, 5, 6],
30:            [7, 8, 9]])
31: 
32:     Notes
33:     -----
34:     The keyword arguments start, stop and step are used by calling
35:     slice(start, stop, step).  This implies axis_slice() does not
36:     handle its arguments the exacty the same as indexing.  To select
37:     a single index k, for example, use
38:         axis_slice(a, start=k, stop=k+1)
39:     In this case, the length of the axis 'axis' in the result will
40:     be 1; the trivial dimension is not removed. (Use numpy.squeeze()
41:     to remove trivial axes.)
42:     '''
43:     a_slice = [slice(None)] * a.ndim
44:     a_slice[axis] = slice(start, stop, step)
45:     b = a[a_slice]
46:     return b
47: 
48: 
49: def axis_reverse(a, axis=-1):
50:     '''Reverse the 1-d slices of `a` along axis `axis`.
51: 
52:     Returns axis_slice(a, step=-1, axis=axis).
53:     '''
54:     return axis_slice(a, step=-1, axis=axis)
55: 
56: 
57: def odd_ext(x, n, axis=-1):
58:     '''
59:     Odd extension at the boundaries of an array
60: 
61:     Generate a new ndarray by making an odd extension of `x` along an axis.
62: 
63:     Parameters
64:     ----------
65:     x : ndarray
66:         The array to be extended.
67:     n : int
68:         The number of elements by which to extend `x` at each end of the axis.
69:     axis : int, optional
70:         The axis along which to extend `x`.  Default is -1.
71: 
72:     Examples
73:     --------
74:     >>> from scipy.signal._arraytools import odd_ext
75:     >>> a = np.array([[1, 2, 3, 4, 5], [0, 1, 4, 9, 16]])
76:     >>> odd_ext(a, 2)
77:     array([[-1,  0,  1,  2,  3,  4,  5,  6,  7],
78:            [-4, -1,  0,  1,  4,  9, 16, 23, 28]])
79: 
80:     Odd extension is a "180 degree rotation" at the endpoints of the original
81:     array:
82: 
83:     >>> t = np.linspace(0, 1.5, 100)
84:     >>> a = 0.9 * np.sin(2 * np.pi * t**2)
85:     >>> b = odd_ext(a, 40)
86:     >>> import matplotlib.pyplot as plt
87:     >>> plt.plot(arange(-40, 140), b, 'b', lw=1, label='odd extension')
88:     >>> plt.plot(arange(100), a, 'r', lw=2, label='original')
89:     >>> plt.legend(loc='best')
90:     >>> plt.show()
91:     '''
92:     if n < 1:
93:         return x
94:     if n > x.shape[axis] - 1:
95:         raise ValueError(("The extension length n (%d) is too big. " +
96:                          "It must not exceed x.shape[axis]-1, which is %d.")
97:                          % (n, x.shape[axis] - 1))
98:     left_end = axis_slice(x, start=0, stop=1, axis=axis)
99:     left_ext = axis_slice(x, start=n, stop=0, step=-1, axis=axis)
100:     right_end = axis_slice(x, start=-1, axis=axis)
101:     right_ext = axis_slice(x, start=-2, stop=-(n + 2), step=-1, axis=axis)
102:     ext = np.concatenate((2 * left_end - left_ext,
103:                           x,
104:                           2 * right_end - right_ext),
105:                          axis=axis)
106:     return ext
107: 
108: 
109: def even_ext(x, n, axis=-1):
110:     '''
111:     Even extension at the boundaries of an array
112: 
113:     Generate a new ndarray by making an even extension of `x` along an axis.
114: 
115:     Parameters
116:     ----------
117:     x : ndarray
118:         The array to be extended.
119:     n : int
120:         The number of elements by which to extend `x` at each end of the axis.
121:     axis : int, optional
122:         The axis along which to extend `x`.  Default is -1.
123: 
124:     Examples
125:     --------
126:     >>> from scipy.signal._arraytools import even_ext
127:     >>> a = np.array([[1, 2, 3, 4, 5], [0, 1, 4, 9, 16]])
128:     >>> even_ext(a, 2)
129:     array([[ 3,  2,  1,  2,  3,  4,  5,  4,  3],
130:            [ 4,  1,  0,  1,  4,  9, 16,  9,  4]])
131: 
132:     Even extension is a "mirror image" at the boundaries of the original array:
133: 
134:     >>> t = np.linspace(0, 1.5, 100)
135:     >>> a = 0.9 * np.sin(2 * np.pi * t**2)
136:     >>> b = even_ext(a, 40)
137:     >>> import matplotlib.pyplot as plt
138:     >>> plt.plot(arange(-40, 140), b, 'b', lw=1, label='even extension')
139:     >>> plt.plot(arange(100), a, 'r', lw=2, label='original')
140:     >>> plt.legend(loc='best')
141:     >>> plt.show()
142:     '''
143:     if n < 1:
144:         return x
145:     if n > x.shape[axis] - 1:
146:         raise ValueError(("The extension length n (%d) is too big. " +
147:                          "It must not exceed x.shape[axis]-1, which is %d.")
148:                          % (n, x.shape[axis] - 1))
149:     left_ext = axis_slice(x, start=n, stop=0, step=-1, axis=axis)
150:     right_ext = axis_slice(x, start=-2, stop=-(n + 2), step=-1, axis=axis)
151:     ext = np.concatenate((left_ext,
152:                           x,
153:                           right_ext),
154:                          axis=axis)
155:     return ext
156: 
157: 
158: def const_ext(x, n, axis=-1):
159:     '''
160:     Constant extension at the boundaries of an array
161: 
162:     Generate a new ndarray that is a constant extension of `x` along an axis.
163: 
164:     The extension repeats the values at the first and last element of
165:     the axis.
166: 
167:     Parameters
168:     ----------
169:     x : ndarray
170:         The array to be extended.
171:     n : int
172:         The number of elements by which to extend `x` at each end of the axis.
173:     axis : int, optional
174:         The axis along which to extend `x`.  Default is -1.
175: 
176:     Examples
177:     --------
178:     >>> from scipy.signal._arraytools import const_ext
179:     >>> a = np.array([[1, 2, 3, 4, 5], [0, 1, 4, 9, 16]])
180:     >>> const_ext(a, 2)
181:     array([[ 1,  1,  1,  2,  3,  4,  5,  5,  5],
182:            [ 0,  0,  0,  1,  4,  9, 16, 16, 16]])
183: 
184:     Constant extension continues with the same values as the endpoints of the
185:     array:
186: 
187:     >>> t = np.linspace(0, 1.5, 100)
188:     >>> a = 0.9 * np.sin(2 * np.pi * t**2)
189:     >>> b = const_ext(a, 40)
190:     >>> import matplotlib.pyplot as plt
191:     >>> plt.plot(arange(-40, 140), b, 'b', lw=1, label='constant extension')
192:     >>> plt.plot(arange(100), a, 'r', lw=2, label='original')
193:     >>> plt.legend(loc='best')
194:     >>> plt.show()
195:     '''
196:     if n < 1:
197:         return x
198:     left_end = axis_slice(x, start=0, stop=1, axis=axis)
199:     ones_shape = [1] * x.ndim
200:     ones_shape[axis] = n
201:     ones = np.ones(ones_shape, dtype=x.dtype)
202:     left_ext = ones * left_end
203:     right_end = axis_slice(x, start=-1, axis=axis)
204:     right_ext = ones * right_end
205:     ext = np.concatenate((left_ext,
206:                           x,
207:                           right_ext),
208:                          axis=axis)
209:     return ext
210: 
211: 
212: def zero_ext(x, n, axis=-1):
213:     '''
214:     Zero padding at the boundaries of an array
215: 
216:     Generate a new ndarray that is a zero padded extension of `x` along
217:     an axis.
218: 
219:     Parameters
220:     ----------
221:     x : ndarray
222:         The array to be extended.
223:     n : int
224:         The number of elements by which to extend `x` at each end of the
225:         axis.
226:     axis : int, optional
227:         The axis along which to extend `x`.  Default is -1.
228: 
229:     Examples
230:     --------
231:     >>> from scipy.signal._arraytools import zero_ext
232:     >>> a = np.array([[1, 2, 3, 4, 5], [0, 1, 4, 9, 16]])
233:     >>> zero_ext(a, 2)
234:     array([[ 0,  0,  1,  2,  3,  4,  5,  0,  0],
235:            [ 0,  0,  0,  1,  4,  9, 16,  0,  0]])
236:     '''
237:     if n < 1:
238:         return x
239:     zeros_shape = list(x.shape)
240:     zeros_shape[axis] = n
241:     zeros = np.zeros(zeros_shape, dtype=x.dtype)
242:     ext = np.concatenate((zeros, x, zeros), axis=axis)
243:     return ext
244: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_286805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, (-1)), 'str', '\nFunctions for acting on a axis of an array.\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import numpy' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/')
import_286806 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy')

if (type(import_286806) is not StypyTypeError):

    if (import_286806 != 'pyd_module'):
        __import__(import_286806)
        sys_modules_286807 = sys.modules[import_286806]
        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'np', sys_modules_286807.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy', import_286806)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/')


@norecursion
def axis_slice(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 9)
    None_286808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 24), 'None')
    # Getting the type of 'None' (line 9)
    None_286809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 35), 'None')
    # Getting the type of 'None' (line 9)
    None_286810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 46), 'None')
    int_286811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 57), 'int')
    defaults = [None_286808, None_286809, None_286810, int_286811]
    # Create a new context for function 'axis_slice'
    module_type_store = module_type_store.open_function_context('axis_slice', 9, 0, False)
    
    # Passed parameters checking function
    axis_slice.stypy_localization = localization
    axis_slice.stypy_type_of_self = None
    axis_slice.stypy_type_store = module_type_store
    axis_slice.stypy_function_name = 'axis_slice'
    axis_slice.stypy_param_names_list = ['a', 'start', 'stop', 'step', 'axis']
    axis_slice.stypy_varargs_param_name = None
    axis_slice.stypy_kwargs_param_name = None
    axis_slice.stypy_call_defaults = defaults
    axis_slice.stypy_call_varargs = varargs
    axis_slice.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'axis_slice', ['a', 'start', 'stop', 'step', 'axis'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'axis_slice', localization, ['a', 'start', 'stop', 'step', 'axis'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'axis_slice(...)' code ##################

    str_286812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, (-1)), 'str', "Take a slice along axis 'axis' from 'a'.\n\n    Parameters\n    ----------\n    a : numpy.ndarray\n        The array to be sliced.\n    start, stop, step : int or None\n        The slice parameters.\n    axis : int, optional\n        The axis of `a` to be sliced.\n\n    Examples\n    --------\n    >>> a = array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])\n    >>> axis_slice(a, start=0, stop=1, axis=1)\n    array([[1],\n           [4],\n           [7]])\n    >>> axis_slice(a, start=1, axis=0)\n    array([[4, 5, 6],\n           [7, 8, 9]])\n\n    Notes\n    -----\n    The keyword arguments start, stop and step are used by calling\n    slice(start, stop, step).  This implies axis_slice() does not\n    handle its arguments the exacty the same as indexing.  To select\n    a single index k, for example, use\n        axis_slice(a, start=k, stop=k+1)\n    In this case, the length of the axis 'axis' in the result will\n    be 1; the trivial dimension is not removed. (Use numpy.squeeze()\n    to remove trivial axes.)\n    ")
    
    # Assigning a BinOp to a Name (line 43):
    
    # Obtaining an instance of the builtin type 'list' (line 43)
    list_286813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 43)
    # Adding element type (line 43)
    
    # Call to slice(...): (line 43)
    # Processing the call arguments (line 43)
    # Getting the type of 'None' (line 43)
    None_286815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 21), 'None', False)
    # Processing the call keyword arguments (line 43)
    kwargs_286816 = {}
    # Getting the type of 'slice' (line 43)
    slice_286814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 15), 'slice', False)
    # Calling slice(args, kwargs) (line 43)
    slice_call_result_286817 = invoke(stypy.reporting.localization.Localization(__file__, 43, 15), slice_286814, *[None_286815], **kwargs_286816)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), list_286813, slice_call_result_286817)
    
    # Getting the type of 'a' (line 43)
    a_286818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 30), 'a')
    # Obtaining the member 'ndim' of a type (line 43)
    ndim_286819 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 30), a_286818, 'ndim')
    # Applying the binary operator '*' (line 43)
    result_mul_286820 = python_operator(stypy.reporting.localization.Localization(__file__, 43, 14), '*', list_286813, ndim_286819)
    
    # Assigning a type to the variable 'a_slice' (line 43)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'a_slice', result_mul_286820)
    
    # Assigning a Call to a Subscript (line 44):
    
    # Call to slice(...): (line 44)
    # Processing the call arguments (line 44)
    # Getting the type of 'start' (line 44)
    start_286822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 26), 'start', False)
    # Getting the type of 'stop' (line 44)
    stop_286823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 33), 'stop', False)
    # Getting the type of 'step' (line 44)
    step_286824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 39), 'step', False)
    # Processing the call keyword arguments (line 44)
    kwargs_286825 = {}
    # Getting the type of 'slice' (line 44)
    slice_286821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 20), 'slice', False)
    # Calling slice(args, kwargs) (line 44)
    slice_call_result_286826 = invoke(stypy.reporting.localization.Localization(__file__, 44, 20), slice_286821, *[start_286822, stop_286823, step_286824], **kwargs_286825)
    
    # Getting the type of 'a_slice' (line 44)
    a_slice_286827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'a_slice')
    # Getting the type of 'axis' (line 44)
    axis_286828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 12), 'axis')
    # Storing an element on a container (line 44)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 4), a_slice_286827, (axis_286828, slice_call_result_286826))
    
    # Assigning a Subscript to a Name (line 45):
    
    # Obtaining the type of the subscript
    # Getting the type of 'a_slice' (line 45)
    a_slice_286829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 10), 'a_slice')
    # Getting the type of 'a' (line 45)
    a_286830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'a')
    # Obtaining the member '__getitem__' of a type (line 45)
    getitem___286831 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 8), a_286830, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 45)
    subscript_call_result_286832 = invoke(stypy.reporting.localization.Localization(__file__, 45, 8), getitem___286831, a_slice_286829)
    
    # Assigning a type to the variable 'b' (line 45)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'b', subscript_call_result_286832)
    # Getting the type of 'b' (line 46)
    b_286833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 11), 'b')
    # Assigning a type to the variable 'stypy_return_type' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'stypy_return_type', b_286833)
    
    # ################# End of 'axis_slice(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'axis_slice' in the type store
    # Getting the type of 'stypy_return_type' (line 9)
    stypy_return_type_286834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_286834)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'axis_slice'
    return stypy_return_type_286834

# Assigning a type to the variable 'axis_slice' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'axis_slice', axis_slice)

@norecursion
def axis_reverse(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_286835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 25), 'int')
    defaults = [int_286835]
    # Create a new context for function 'axis_reverse'
    module_type_store = module_type_store.open_function_context('axis_reverse', 49, 0, False)
    
    # Passed parameters checking function
    axis_reverse.stypy_localization = localization
    axis_reverse.stypy_type_of_self = None
    axis_reverse.stypy_type_store = module_type_store
    axis_reverse.stypy_function_name = 'axis_reverse'
    axis_reverse.stypy_param_names_list = ['a', 'axis']
    axis_reverse.stypy_varargs_param_name = None
    axis_reverse.stypy_kwargs_param_name = None
    axis_reverse.stypy_call_defaults = defaults
    axis_reverse.stypy_call_varargs = varargs
    axis_reverse.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'axis_reverse', ['a', 'axis'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'axis_reverse', localization, ['a', 'axis'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'axis_reverse(...)' code ##################

    str_286836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, (-1)), 'str', 'Reverse the 1-d slices of `a` along axis `axis`.\n\n    Returns axis_slice(a, step=-1, axis=axis).\n    ')
    
    # Call to axis_slice(...): (line 54)
    # Processing the call arguments (line 54)
    # Getting the type of 'a' (line 54)
    a_286838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 22), 'a', False)
    # Processing the call keyword arguments (line 54)
    int_286839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 30), 'int')
    keyword_286840 = int_286839
    # Getting the type of 'axis' (line 54)
    axis_286841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 39), 'axis', False)
    keyword_286842 = axis_286841
    kwargs_286843 = {'step': keyword_286840, 'axis': keyword_286842}
    # Getting the type of 'axis_slice' (line 54)
    axis_slice_286837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 11), 'axis_slice', False)
    # Calling axis_slice(args, kwargs) (line 54)
    axis_slice_call_result_286844 = invoke(stypy.reporting.localization.Localization(__file__, 54, 11), axis_slice_286837, *[a_286838], **kwargs_286843)
    
    # Assigning a type to the variable 'stypy_return_type' (line 54)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'stypy_return_type', axis_slice_call_result_286844)
    
    # ################# End of 'axis_reverse(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'axis_reverse' in the type store
    # Getting the type of 'stypy_return_type' (line 49)
    stypy_return_type_286845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_286845)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'axis_reverse'
    return stypy_return_type_286845

# Assigning a type to the variable 'axis_reverse' (line 49)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 0), 'axis_reverse', axis_reverse)

@norecursion
def odd_ext(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_286846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 23), 'int')
    defaults = [int_286846]
    # Create a new context for function 'odd_ext'
    module_type_store = module_type_store.open_function_context('odd_ext', 57, 0, False)
    
    # Passed parameters checking function
    odd_ext.stypy_localization = localization
    odd_ext.stypy_type_of_self = None
    odd_ext.stypy_type_store = module_type_store
    odd_ext.stypy_function_name = 'odd_ext'
    odd_ext.stypy_param_names_list = ['x', 'n', 'axis']
    odd_ext.stypy_varargs_param_name = None
    odd_ext.stypy_kwargs_param_name = None
    odd_ext.stypy_call_defaults = defaults
    odd_ext.stypy_call_varargs = varargs
    odd_ext.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'odd_ext', ['x', 'n', 'axis'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'odd_ext', localization, ['x', 'n', 'axis'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'odd_ext(...)' code ##################

    str_286847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, (-1)), 'str', '\n    Odd extension at the boundaries of an array\n\n    Generate a new ndarray by making an odd extension of `x` along an axis.\n\n    Parameters\n    ----------\n    x : ndarray\n        The array to be extended.\n    n : int\n        The number of elements by which to extend `x` at each end of the axis.\n    axis : int, optional\n        The axis along which to extend `x`.  Default is -1.\n\n    Examples\n    --------\n    >>> from scipy.signal._arraytools import odd_ext\n    >>> a = np.array([[1, 2, 3, 4, 5], [0, 1, 4, 9, 16]])\n    >>> odd_ext(a, 2)\n    array([[-1,  0,  1,  2,  3,  4,  5,  6,  7],\n           [-4, -1,  0,  1,  4,  9, 16, 23, 28]])\n\n    Odd extension is a "180 degree rotation" at the endpoints of the original\n    array:\n\n    >>> t = np.linspace(0, 1.5, 100)\n    >>> a = 0.9 * np.sin(2 * np.pi * t**2)\n    >>> b = odd_ext(a, 40)\n    >>> import matplotlib.pyplot as plt\n    >>> plt.plot(arange(-40, 140), b, \'b\', lw=1, label=\'odd extension\')\n    >>> plt.plot(arange(100), a, \'r\', lw=2, label=\'original\')\n    >>> plt.legend(loc=\'best\')\n    >>> plt.show()\n    ')
    
    
    # Getting the type of 'n' (line 92)
    n_286848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 7), 'n')
    int_286849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 11), 'int')
    # Applying the binary operator '<' (line 92)
    result_lt_286850 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 7), '<', n_286848, int_286849)
    
    # Testing the type of an if condition (line 92)
    if_condition_286851 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 92, 4), result_lt_286850)
    # Assigning a type to the variable 'if_condition_286851' (line 92)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), 'if_condition_286851', if_condition_286851)
    # SSA begins for if statement (line 92)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'x' (line 93)
    x_286852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 15), 'x')
    # Assigning a type to the variable 'stypy_return_type' (line 93)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'stypy_return_type', x_286852)
    # SSA join for if statement (line 92)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'n' (line 94)
    n_286853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 7), 'n')
    
    # Obtaining the type of the subscript
    # Getting the type of 'axis' (line 94)
    axis_286854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 19), 'axis')
    # Getting the type of 'x' (line 94)
    x_286855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 11), 'x')
    # Obtaining the member 'shape' of a type (line 94)
    shape_286856 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 11), x_286855, 'shape')
    # Obtaining the member '__getitem__' of a type (line 94)
    getitem___286857 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 11), shape_286856, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 94)
    subscript_call_result_286858 = invoke(stypy.reporting.localization.Localization(__file__, 94, 11), getitem___286857, axis_286854)
    
    int_286859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 27), 'int')
    # Applying the binary operator '-' (line 94)
    result_sub_286860 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 11), '-', subscript_call_result_286858, int_286859)
    
    # Applying the binary operator '>' (line 94)
    result_gt_286861 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 7), '>', n_286853, result_sub_286860)
    
    # Testing the type of an if condition (line 94)
    if_condition_286862 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 94, 4), result_gt_286861)
    # Assigning a type to the variable 'if_condition_286862' (line 94)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'if_condition_286862', if_condition_286862)
    # SSA begins for if statement (line 94)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 95)
    # Processing the call arguments (line 95)
    str_286864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 26), 'str', 'The extension length n (%d) is too big. ')
    str_286865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 25), 'str', 'It must not exceed x.shape[axis]-1, which is %d.')
    # Applying the binary operator '+' (line 95)
    result_add_286866 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 26), '+', str_286864, str_286865)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 97)
    tuple_286867 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 28), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 97)
    # Adding element type (line 97)
    # Getting the type of 'n' (line 97)
    n_286868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 28), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 28), tuple_286867, n_286868)
    # Adding element type (line 97)
    
    # Obtaining the type of the subscript
    # Getting the type of 'axis' (line 97)
    axis_286869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 39), 'axis', False)
    # Getting the type of 'x' (line 97)
    x_286870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 31), 'x', False)
    # Obtaining the member 'shape' of a type (line 97)
    shape_286871 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 31), x_286870, 'shape')
    # Obtaining the member '__getitem__' of a type (line 97)
    getitem___286872 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 31), shape_286871, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 97)
    subscript_call_result_286873 = invoke(stypy.reporting.localization.Localization(__file__, 97, 31), getitem___286872, axis_286869)
    
    int_286874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 47), 'int')
    # Applying the binary operator '-' (line 97)
    result_sub_286875 = python_operator(stypy.reporting.localization.Localization(__file__, 97, 31), '-', subscript_call_result_286873, int_286874)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 28), tuple_286867, result_sub_286875)
    
    # Applying the binary operator '%' (line 95)
    result_mod_286876 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 25), '%', result_add_286866, tuple_286867)
    
    # Processing the call keyword arguments (line 95)
    kwargs_286877 = {}
    # Getting the type of 'ValueError' (line 95)
    ValueError_286863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 95)
    ValueError_call_result_286878 = invoke(stypy.reporting.localization.Localization(__file__, 95, 14), ValueError_286863, *[result_mod_286876], **kwargs_286877)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 95, 8), ValueError_call_result_286878, 'raise parameter', BaseException)
    # SSA join for if statement (line 94)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 98):
    
    # Call to axis_slice(...): (line 98)
    # Processing the call arguments (line 98)
    # Getting the type of 'x' (line 98)
    x_286880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 26), 'x', False)
    # Processing the call keyword arguments (line 98)
    int_286881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 35), 'int')
    keyword_286882 = int_286881
    int_286883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 43), 'int')
    keyword_286884 = int_286883
    # Getting the type of 'axis' (line 98)
    axis_286885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 51), 'axis', False)
    keyword_286886 = axis_286885
    kwargs_286887 = {'start': keyword_286882, 'stop': keyword_286884, 'axis': keyword_286886}
    # Getting the type of 'axis_slice' (line 98)
    axis_slice_286879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 15), 'axis_slice', False)
    # Calling axis_slice(args, kwargs) (line 98)
    axis_slice_call_result_286888 = invoke(stypy.reporting.localization.Localization(__file__, 98, 15), axis_slice_286879, *[x_286880], **kwargs_286887)
    
    # Assigning a type to the variable 'left_end' (line 98)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'left_end', axis_slice_call_result_286888)
    
    # Assigning a Call to a Name (line 99):
    
    # Call to axis_slice(...): (line 99)
    # Processing the call arguments (line 99)
    # Getting the type of 'x' (line 99)
    x_286890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 26), 'x', False)
    # Processing the call keyword arguments (line 99)
    # Getting the type of 'n' (line 99)
    n_286891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 35), 'n', False)
    keyword_286892 = n_286891
    int_286893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 43), 'int')
    keyword_286894 = int_286893
    int_286895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 51), 'int')
    keyword_286896 = int_286895
    # Getting the type of 'axis' (line 99)
    axis_286897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 60), 'axis', False)
    keyword_286898 = axis_286897
    kwargs_286899 = {'start': keyword_286892, 'step': keyword_286896, 'stop': keyword_286894, 'axis': keyword_286898}
    # Getting the type of 'axis_slice' (line 99)
    axis_slice_286889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 15), 'axis_slice', False)
    # Calling axis_slice(args, kwargs) (line 99)
    axis_slice_call_result_286900 = invoke(stypy.reporting.localization.Localization(__file__, 99, 15), axis_slice_286889, *[x_286890], **kwargs_286899)
    
    # Assigning a type to the variable 'left_ext' (line 99)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'left_ext', axis_slice_call_result_286900)
    
    # Assigning a Call to a Name (line 100):
    
    # Call to axis_slice(...): (line 100)
    # Processing the call arguments (line 100)
    # Getting the type of 'x' (line 100)
    x_286902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 27), 'x', False)
    # Processing the call keyword arguments (line 100)
    int_286903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 36), 'int')
    keyword_286904 = int_286903
    # Getting the type of 'axis' (line 100)
    axis_286905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 45), 'axis', False)
    keyword_286906 = axis_286905
    kwargs_286907 = {'start': keyword_286904, 'axis': keyword_286906}
    # Getting the type of 'axis_slice' (line 100)
    axis_slice_286901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 16), 'axis_slice', False)
    # Calling axis_slice(args, kwargs) (line 100)
    axis_slice_call_result_286908 = invoke(stypy.reporting.localization.Localization(__file__, 100, 16), axis_slice_286901, *[x_286902], **kwargs_286907)
    
    # Assigning a type to the variable 'right_end' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'right_end', axis_slice_call_result_286908)
    
    # Assigning a Call to a Name (line 101):
    
    # Call to axis_slice(...): (line 101)
    # Processing the call arguments (line 101)
    # Getting the type of 'x' (line 101)
    x_286910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 27), 'x', False)
    # Processing the call keyword arguments (line 101)
    int_286911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 36), 'int')
    keyword_286912 = int_286911
    
    # Getting the type of 'n' (line 101)
    n_286913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 47), 'n', False)
    int_286914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 51), 'int')
    # Applying the binary operator '+' (line 101)
    result_add_286915 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 47), '+', n_286913, int_286914)
    
    # Applying the 'usub' unary operator (line 101)
    result___neg___286916 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 45), 'usub', result_add_286915)
    
    keyword_286917 = result___neg___286916
    int_286918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 60), 'int')
    keyword_286919 = int_286918
    # Getting the type of 'axis' (line 101)
    axis_286920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 69), 'axis', False)
    keyword_286921 = axis_286920
    kwargs_286922 = {'start': keyword_286912, 'step': keyword_286919, 'stop': keyword_286917, 'axis': keyword_286921}
    # Getting the type of 'axis_slice' (line 101)
    axis_slice_286909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 16), 'axis_slice', False)
    # Calling axis_slice(args, kwargs) (line 101)
    axis_slice_call_result_286923 = invoke(stypy.reporting.localization.Localization(__file__, 101, 16), axis_slice_286909, *[x_286910], **kwargs_286922)
    
    # Assigning a type to the variable 'right_ext' (line 101)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'right_ext', axis_slice_call_result_286923)
    
    # Assigning a Call to a Name (line 102):
    
    # Call to concatenate(...): (line 102)
    # Processing the call arguments (line 102)
    
    # Obtaining an instance of the builtin type 'tuple' (line 102)
    tuple_286926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 26), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 102)
    # Adding element type (line 102)
    int_286927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 26), 'int')
    # Getting the type of 'left_end' (line 102)
    left_end_286928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 30), 'left_end', False)
    # Applying the binary operator '*' (line 102)
    result_mul_286929 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 26), '*', int_286927, left_end_286928)
    
    # Getting the type of 'left_ext' (line 102)
    left_ext_286930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 41), 'left_ext', False)
    # Applying the binary operator '-' (line 102)
    result_sub_286931 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 26), '-', result_mul_286929, left_ext_286930)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 26), tuple_286926, result_sub_286931)
    # Adding element type (line 102)
    # Getting the type of 'x' (line 103)
    x_286932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 26), 'x', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 26), tuple_286926, x_286932)
    # Adding element type (line 102)
    int_286933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 26), 'int')
    # Getting the type of 'right_end' (line 104)
    right_end_286934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 30), 'right_end', False)
    # Applying the binary operator '*' (line 104)
    result_mul_286935 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 26), '*', int_286933, right_end_286934)
    
    # Getting the type of 'right_ext' (line 104)
    right_ext_286936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 42), 'right_ext', False)
    # Applying the binary operator '-' (line 104)
    result_sub_286937 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 26), '-', result_mul_286935, right_ext_286936)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 26), tuple_286926, result_sub_286937)
    
    # Processing the call keyword arguments (line 102)
    # Getting the type of 'axis' (line 105)
    axis_286938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 30), 'axis', False)
    keyword_286939 = axis_286938
    kwargs_286940 = {'axis': keyword_286939}
    # Getting the type of 'np' (line 102)
    np_286924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 10), 'np', False)
    # Obtaining the member 'concatenate' of a type (line 102)
    concatenate_286925 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 10), np_286924, 'concatenate')
    # Calling concatenate(args, kwargs) (line 102)
    concatenate_call_result_286941 = invoke(stypy.reporting.localization.Localization(__file__, 102, 10), concatenate_286925, *[tuple_286926], **kwargs_286940)
    
    # Assigning a type to the variable 'ext' (line 102)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 4), 'ext', concatenate_call_result_286941)
    # Getting the type of 'ext' (line 106)
    ext_286942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 11), 'ext')
    # Assigning a type to the variable 'stypy_return_type' (line 106)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 4), 'stypy_return_type', ext_286942)
    
    # ################# End of 'odd_ext(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'odd_ext' in the type store
    # Getting the type of 'stypy_return_type' (line 57)
    stypy_return_type_286943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_286943)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'odd_ext'
    return stypy_return_type_286943

# Assigning a type to the variable 'odd_ext' (line 57)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 0), 'odd_ext', odd_ext)

@norecursion
def even_ext(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_286944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 24), 'int')
    defaults = [int_286944]
    # Create a new context for function 'even_ext'
    module_type_store = module_type_store.open_function_context('even_ext', 109, 0, False)
    
    # Passed parameters checking function
    even_ext.stypy_localization = localization
    even_ext.stypy_type_of_self = None
    even_ext.stypy_type_store = module_type_store
    even_ext.stypy_function_name = 'even_ext'
    even_ext.stypy_param_names_list = ['x', 'n', 'axis']
    even_ext.stypy_varargs_param_name = None
    even_ext.stypy_kwargs_param_name = None
    even_ext.stypy_call_defaults = defaults
    even_ext.stypy_call_varargs = varargs
    even_ext.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'even_ext', ['x', 'n', 'axis'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'even_ext', localization, ['x', 'n', 'axis'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'even_ext(...)' code ##################

    str_286945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, (-1)), 'str', '\n    Even extension at the boundaries of an array\n\n    Generate a new ndarray by making an even extension of `x` along an axis.\n\n    Parameters\n    ----------\n    x : ndarray\n        The array to be extended.\n    n : int\n        The number of elements by which to extend `x` at each end of the axis.\n    axis : int, optional\n        The axis along which to extend `x`.  Default is -1.\n\n    Examples\n    --------\n    >>> from scipy.signal._arraytools import even_ext\n    >>> a = np.array([[1, 2, 3, 4, 5], [0, 1, 4, 9, 16]])\n    >>> even_ext(a, 2)\n    array([[ 3,  2,  1,  2,  3,  4,  5,  4,  3],\n           [ 4,  1,  0,  1,  4,  9, 16,  9,  4]])\n\n    Even extension is a "mirror image" at the boundaries of the original array:\n\n    >>> t = np.linspace(0, 1.5, 100)\n    >>> a = 0.9 * np.sin(2 * np.pi * t**2)\n    >>> b = even_ext(a, 40)\n    >>> import matplotlib.pyplot as plt\n    >>> plt.plot(arange(-40, 140), b, \'b\', lw=1, label=\'even extension\')\n    >>> plt.plot(arange(100), a, \'r\', lw=2, label=\'original\')\n    >>> plt.legend(loc=\'best\')\n    >>> plt.show()\n    ')
    
    
    # Getting the type of 'n' (line 143)
    n_286946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 7), 'n')
    int_286947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 11), 'int')
    # Applying the binary operator '<' (line 143)
    result_lt_286948 = python_operator(stypy.reporting.localization.Localization(__file__, 143, 7), '<', n_286946, int_286947)
    
    # Testing the type of an if condition (line 143)
    if_condition_286949 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 143, 4), result_lt_286948)
    # Assigning a type to the variable 'if_condition_286949' (line 143)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 4), 'if_condition_286949', if_condition_286949)
    # SSA begins for if statement (line 143)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'x' (line 144)
    x_286950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 15), 'x')
    # Assigning a type to the variable 'stypy_return_type' (line 144)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'stypy_return_type', x_286950)
    # SSA join for if statement (line 143)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'n' (line 145)
    n_286951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 7), 'n')
    
    # Obtaining the type of the subscript
    # Getting the type of 'axis' (line 145)
    axis_286952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 19), 'axis')
    # Getting the type of 'x' (line 145)
    x_286953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 11), 'x')
    # Obtaining the member 'shape' of a type (line 145)
    shape_286954 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 11), x_286953, 'shape')
    # Obtaining the member '__getitem__' of a type (line 145)
    getitem___286955 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 11), shape_286954, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 145)
    subscript_call_result_286956 = invoke(stypy.reporting.localization.Localization(__file__, 145, 11), getitem___286955, axis_286952)
    
    int_286957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 27), 'int')
    # Applying the binary operator '-' (line 145)
    result_sub_286958 = python_operator(stypy.reporting.localization.Localization(__file__, 145, 11), '-', subscript_call_result_286956, int_286957)
    
    # Applying the binary operator '>' (line 145)
    result_gt_286959 = python_operator(stypy.reporting.localization.Localization(__file__, 145, 7), '>', n_286951, result_sub_286958)
    
    # Testing the type of an if condition (line 145)
    if_condition_286960 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 145, 4), result_gt_286959)
    # Assigning a type to the variable 'if_condition_286960' (line 145)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 4), 'if_condition_286960', if_condition_286960)
    # SSA begins for if statement (line 145)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 146)
    # Processing the call arguments (line 146)
    str_286962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 26), 'str', 'The extension length n (%d) is too big. ')
    str_286963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 25), 'str', 'It must not exceed x.shape[axis]-1, which is %d.')
    # Applying the binary operator '+' (line 146)
    result_add_286964 = python_operator(stypy.reporting.localization.Localization(__file__, 146, 26), '+', str_286962, str_286963)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 148)
    tuple_286965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 28), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 148)
    # Adding element type (line 148)
    # Getting the type of 'n' (line 148)
    n_286966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 28), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 148, 28), tuple_286965, n_286966)
    # Adding element type (line 148)
    
    # Obtaining the type of the subscript
    # Getting the type of 'axis' (line 148)
    axis_286967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 39), 'axis', False)
    # Getting the type of 'x' (line 148)
    x_286968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 31), 'x', False)
    # Obtaining the member 'shape' of a type (line 148)
    shape_286969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 31), x_286968, 'shape')
    # Obtaining the member '__getitem__' of a type (line 148)
    getitem___286970 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 31), shape_286969, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 148)
    subscript_call_result_286971 = invoke(stypy.reporting.localization.Localization(__file__, 148, 31), getitem___286970, axis_286967)
    
    int_286972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 47), 'int')
    # Applying the binary operator '-' (line 148)
    result_sub_286973 = python_operator(stypy.reporting.localization.Localization(__file__, 148, 31), '-', subscript_call_result_286971, int_286972)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 148, 28), tuple_286965, result_sub_286973)
    
    # Applying the binary operator '%' (line 146)
    result_mod_286974 = python_operator(stypy.reporting.localization.Localization(__file__, 146, 25), '%', result_add_286964, tuple_286965)
    
    # Processing the call keyword arguments (line 146)
    kwargs_286975 = {}
    # Getting the type of 'ValueError' (line 146)
    ValueError_286961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 146)
    ValueError_call_result_286976 = invoke(stypy.reporting.localization.Localization(__file__, 146, 14), ValueError_286961, *[result_mod_286974], **kwargs_286975)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 146, 8), ValueError_call_result_286976, 'raise parameter', BaseException)
    # SSA join for if statement (line 145)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 149):
    
    # Call to axis_slice(...): (line 149)
    # Processing the call arguments (line 149)
    # Getting the type of 'x' (line 149)
    x_286978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 26), 'x', False)
    # Processing the call keyword arguments (line 149)
    # Getting the type of 'n' (line 149)
    n_286979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 35), 'n', False)
    keyword_286980 = n_286979
    int_286981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 43), 'int')
    keyword_286982 = int_286981
    int_286983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 51), 'int')
    keyword_286984 = int_286983
    # Getting the type of 'axis' (line 149)
    axis_286985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 60), 'axis', False)
    keyword_286986 = axis_286985
    kwargs_286987 = {'start': keyword_286980, 'step': keyword_286984, 'stop': keyword_286982, 'axis': keyword_286986}
    # Getting the type of 'axis_slice' (line 149)
    axis_slice_286977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 15), 'axis_slice', False)
    # Calling axis_slice(args, kwargs) (line 149)
    axis_slice_call_result_286988 = invoke(stypy.reporting.localization.Localization(__file__, 149, 15), axis_slice_286977, *[x_286978], **kwargs_286987)
    
    # Assigning a type to the variable 'left_ext' (line 149)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 4), 'left_ext', axis_slice_call_result_286988)
    
    # Assigning a Call to a Name (line 150):
    
    # Call to axis_slice(...): (line 150)
    # Processing the call arguments (line 150)
    # Getting the type of 'x' (line 150)
    x_286990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 27), 'x', False)
    # Processing the call keyword arguments (line 150)
    int_286991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 36), 'int')
    keyword_286992 = int_286991
    
    # Getting the type of 'n' (line 150)
    n_286993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 47), 'n', False)
    int_286994 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 51), 'int')
    # Applying the binary operator '+' (line 150)
    result_add_286995 = python_operator(stypy.reporting.localization.Localization(__file__, 150, 47), '+', n_286993, int_286994)
    
    # Applying the 'usub' unary operator (line 150)
    result___neg___286996 = python_operator(stypy.reporting.localization.Localization(__file__, 150, 45), 'usub', result_add_286995)
    
    keyword_286997 = result___neg___286996
    int_286998 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 60), 'int')
    keyword_286999 = int_286998
    # Getting the type of 'axis' (line 150)
    axis_287000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 69), 'axis', False)
    keyword_287001 = axis_287000
    kwargs_287002 = {'start': keyword_286992, 'step': keyword_286999, 'stop': keyword_286997, 'axis': keyword_287001}
    # Getting the type of 'axis_slice' (line 150)
    axis_slice_286989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 16), 'axis_slice', False)
    # Calling axis_slice(args, kwargs) (line 150)
    axis_slice_call_result_287003 = invoke(stypy.reporting.localization.Localization(__file__, 150, 16), axis_slice_286989, *[x_286990], **kwargs_287002)
    
    # Assigning a type to the variable 'right_ext' (line 150)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 4), 'right_ext', axis_slice_call_result_287003)
    
    # Assigning a Call to a Name (line 151):
    
    # Call to concatenate(...): (line 151)
    # Processing the call arguments (line 151)
    
    # Obtaining an instance of the builtin type 'tuple' (line 151)
    tuple_287006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 26), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 151)
    # Adding element type (line 151)
    # Getting the type of 'left_ext' (line 151)
    left_ext_287007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 26), 'left_ext', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 151, 26), tuple_287006, left_ext_287007)
    # Adding element type (line 151)
    # Getting the type of 'x' (line 152)
    x_287008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 26), 'x', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 151, 26), tuple_287006, x_287008)
    # Adding element type (line 151)
    # Getting the type of 'right_ext' (line 153)
    right_ext_287009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 26), 'right_ext', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 151, 26), tuple_287006, right_ext_287009)
    
    # Processing the call keyword arguments (line 151)
    # Getting the type of 'axis' (line 154)
    axis_287010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 30), 'axis', False)
    keyword_287011 = axis_287010
    kwargs_287012 = {'axis': keyword_287011}
    # Getting the type of 'np' (line 151)
    np_287004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 10), 'np', False)
    # Obtaining the member 'concatenate' of a type (line 151)
    concatenate_287005 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 10), np_287004, 'concatenate')
    # Calling concatenate(args, kwargs) (line 151)
    concatenate_call_result_287013 = invoke(stypy.reporting.localization.Localization(__file__, 151, 10), concatenate_287005, *[tuple_287006], **kwargs_287012)
    
    # Assigning a type to the variable 'ext' (line 151)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 4), 'ext', concatenate_call_result_287013)
    # Getting the type of 'ext' (line 155)
    ext_287014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 11), 'ext')
    # Assigning a type to the variable 'stypy_return_type' (line 155)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 4), 'stypy_return_type', ext_287014)
    
    # ################# End of 'even_ext(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'even_ext' in the type store
    # Getting the type of 'stypy_return_type' (line 109)
    stypy_return_type_287015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_287015)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'even_ext'
    return stypy_return_type_287015

# Assigning a type to the variable 'even_ext' (line 109)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 0), 'even_ext', even_ext)

@norecursion
def const_ext(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_287016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 25), 'int')
    defaults = [int_287016]
    # Create a new context for function 'const_ext'
    module_type_store = module_type_store.open_function_context('const_ext', 158, 0, False)
    
    # Passed parameters checking function
    const_ext.stypy_localization = localization
    const_ext.stypy_type_of_self = None
    const_ext.stypy_type_store = module_type_store
    const_ext.stypy_function_name = 'const_ext'
    const_ext.stypy_param_names_list = ['x', 'n', 'axis']
    const_ext.stypy_varargs_param_name = None
    const_ext.stypy_kwargs_param_name = None
    const_ext.stypy_call_defaults = defaults
    const_ext.stypy_call_varargs = varargs
    const_ext.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'const_ext', ['x', 'n', 'axis'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'const_ext', localization, ['x', 'n', 'axis'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'const_ext(...)' code ##################

    str_287017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, (-1)), 'str', "\n    Constant extension at the boundaries of an array\n\n    Generate a new ndarray that is a constant extension of `x` along an axis.\n\n    The extension repeats the values at the first and last element of\n    the axis.\n\n    Parameters\n    ----------\n    x : ndarray\n        The array to be extended.\n    n : int\n        The number of elements by which to extend `x` at each end of the axis.\n    axis : int, optional\n        The axis along which to extend `x`.  Default is -1.\n\n    Examples\n    --------\n    >>> from scipy.signal._arraytools import const_ext\n    >>> a = np.array([[1, 2, 3, 4, 5], [0, 1, 4, 9, 16]])\n    >>> const_ext(a, 2)\n    array([[ 1,  1,  1,  2,  3,  4,  5,  5,  5],\n           [ 0,  0,  0,  1,  4,  9, 16, 16, 16]])\n\n    Constant extension continues with the same values as the endpoints of the\n    array:\n\n    >>> t = np.linspace(0, 1.5, 100)\n    >>> a = 0.9 * np.sin(2 * np.pi * t**2)\n    >>> b = const_ext(a, 40)\n    >>> import matplotlib.pyplot as plt\n    >>> plt.plot(arange(-40, 140), b, 'b', lw=1, label='constant extension')\n    >>> plt.plot(arange(100), a, 'r', lw=2, label='original')\n    >>> plt.legend(loc='best')\n    >>> plt.show()\n    ")
    
    
    # Getting the type of 'n' (line 196)
    n_287018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 7), 'n')
    int_287019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 11), 'int')
    # Applying the binary operator '<' (line 196)
    result_lt_287020 = python_operator(stypy.reporting.localization.Localization(__file__, 196, 7), '<', n_287018, int_287019)
    
    # Testing the type of an if condition (line 196)
    if_condition_287021 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 196, 4), result_lt_287020)
    # Assigning a type to the variable 'if_condition_287021' (line 196)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 4), 'if_condition_287021', if_condition_287021)
    # SSA begins for if statement (line 196)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'x' (line 197)
    x_287022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 15), 'x')
    # Assigning a type to the variable 'stypy_return_type' (line 197)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 8), 'stypy_return_type', x_287022)
    # SSA join for if statement (line 196)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 198):
    
    # Call to axis_slice(...): (line 198)
    # Processing the call arguments (line 198)
    # Getting the type of 'x' (line 198)
    x_287024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 26), 'x', False)
    # Processing the call keyword arguments (line 198)
    int_287025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 35), 'int')
    keyword_287026 = int_287025
    int_287027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 43), 'int')
    keyword_287028 = int_287027
    # Getting the type of 'axis' (line 198)
    axis_287029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 51), 'axis', False)
    keyword_287030 = axis_287029
    kwargs_287031 = {'start': keyword_287026, 'stop': keyword_287028, 'axis': keyword_287030}
    # Getting the type of 'axis_slice' (line 198)
    axis_slice_287023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 15), 'axis_slice', False)
    # Calling axis_slice(args, kwargs) (line 198)
    axis_slice_call_result_287032 = invoke(stypy.reporting.localization.Localization(__file__, 198, 15), axis_slice_287023, *[x_287024], **kwargs_287031)
    
    # Assigning a type to the variable 'left_end' (line 198)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 4), 'left_end', axis_slice_call_result_287032)
    
    # Assigning a BinOp to a Name (line 199):
    
    # Obtaining an instance of the builtin type 'list' (line 199)
    list_287033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 199)
    # Adding element type (line 199)
    int_287034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 17), list_287033, int_287034)
    
    # Getting the type of 'x' (line 199)
    x_287035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 23), 'x')
    # Obtaining the member 'ndim' of a type (line 199)
    ndim_287036 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 23), x_287035, 'ndim')
    # Applying the binary operator '*' (line 199)
    result_mul_287037 = python_operator(stypy.reporting.localization.Localization(__file__, 199, 17), '*', list_287033, ndim_287036)
    
    # Assigning a type to the variable 'ones_shape' (line 199)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 4), 'ones_shape', result_mul_287037)
    
    # Assigning a Name to a Subscript (line 200):
    # Getting the type of 'n' (line 200)
    n_287038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 23), 'n')
    # Getting the type of 'ones_shape' (line 200)
    ones_shape_287039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 4), 'ones_shape')
    # Getting the type of 'axis' (line 200)
    axis_287040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 15), 'axis')
    # Storing an element on a container (line 200)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 4), ones_shape_287039, (axis_287040, n_287038))
    
    # Assigning a Call to a Name (line 201):
    
    # Call to ones(...): (line 201)
    # Processing the call arguments (line 201)
    # Getting the type of 'ones_shape' (line 201)
    ones_shape_287043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 19), 'ones_shape', False)
    # Processing the call keyword arguments (line 201)
    # Getting the type of 'x' (line 201)
    x_287044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 37), 'x', False)
    # Obtaining the member 'dtype' of a type (line 201)
    dtype_287045 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 37), x_287044, 'dtype')
    keyword_287046 = dtype_287045
    kwargs_287047 = {'dtype': keyword_287046}
    # Getting the type of 'np' (line 201)
    np_287041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 11), 'np', False)
    # Obtaining the member 'ones' of a type (line 201)
    ones_287042 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 11), np_287041, 'ones')
    # Calling ones(args, kwargs) (line 201)
    ones_call_result_287048 = invoke(stypy.reporting.localization.Localization(__file__, 201, 11), ones_287042, *[ones_shape_287043], **kwargs_287047)
    
    # Assigning a type to the variable 'ones' (line 201)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 4), 'ones', ones_call_result_287048)
    
    # Assigning a BinOp to a Name (line 202):
    # Getting the type of 'ones' (line 202)
    ones_287049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 15), 'ones')
    # Getting the type of 'left_end' (line 202)
    left_end_287050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 22), 'left_end')
    # Applying the binary operator '*' (line 202)
    result_mul_287051 = python_operator(stypy.reporting.localization.Localization(__file__, 202, 15), '*', ones_287049, left_end_287050)
    
    # Assigning a type to the variable 'left_ext' (line 202)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 4), 'left_ext', result_mul_287051)
    
    # Assigning a Call to a Name (line 203):
    
    # Call to axis_slice(...): (line 203)
    # Processing the call arguments (line 203)
    # Getting the type of 'x' (line 203)
    x_287053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 27), 'x', False)
    # Processing the call keyword arguments (line 203)
    int_287054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 36), 'int')
    keyword_287055 = int_287054
    # Getting the type of 'axis' (line 203)
    axis_287056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 45), 'axis', False)
    keyword_287057 = axis_287056
    kwargs_287058 = {'start': keyword_287055, 'axis': keyword_287057}
    # Getting the type of 'axis_slice' (line 203)
    axis_slice_287052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 16), 'axis_slice', False)
    # Calling axis_slice(args, kwargs) (line 203)
    axis_slice_call_result_287059 = invoke(stypy.reporting.localization.Localization(__file__, 203, 16), axis_slice_287052, *[x_287053], **kwargs_287058)
    
    # Assigning a type to the variable 'right_end' (line 203)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 4), 'right_end', axis_slice_call_result_287059)
    
    # Assigning a BinOp to a Name (line 204):
    # Getting the type of 'ones' (line 204)
    ones_287060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 16), 'ones')
    # Getting the type of 'right_end' (line 204)
    right_end_287061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 23), 'right_end')
    # Applying the binary operator '*' (line 204)
    result_mul_287062 = python_operator(stypy.reporting.localization.Localization(__file__, 204, 16), '*', ones_287060, right_end_287061)
    
    # Assigning a type to the variable 'right_ext' (line 204)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 4), 'right_ext', result_mul_287062)
    
    # Assigning a Call to a Name (line 205):
    
    # Call to concatenate(...): (line 205)
    # Processing the call arguments (line 205)
    
    # Obtaining an instance of the builtin type 'tuple' (line 205)
    tuple_287065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 26), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 205)
    # Adding element type (line 205)
    # Getting the type of 'left_ext' (line 205)
    left_ext_287066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 26), 'left_ext', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 205, 26), tuple_287065, left_ext_287066)
    # Adding element type (line 205)
    # Getting the type of 'x' (line 206)
    x_287067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 26), 'x', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 205, 26), tuple_287065, x_287067)
    # Adding element type (line 205)
    # Getting the type of 'right_ext' (line 207)
    right_ext_287068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 26), 'right_ext', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 205, 26), tuple_287065, right_ext_287068)
    
    # Processing the call keyword arguments (line 205)
    # Getting the type of 'axis' (line 208)
    axis_287069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 30), 'axis', False)
    keyword_287070 = axis_287069
    kwargs_287071 = {'axis': keyword_287070}
    # Getting the type of 'np' (line 205)
    np_287063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 10), 'np', False)
    # Obtaining the member 'concatenate' of a type (line 205)
    concatenate_287064 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 10), np_287063, 'concatenate')
    # Calling concatenate(args, kwargs) (line 205)
    concatenate_call_result_287072 = invoke(stypy.reporting.localization.Localization(__file__, 205, 10), concatenate_287064, *[tuple_287065], **kwargs_287071)
    
    # Assigning a type to the variable 'ext' (line 205)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 4), 'ext', concatenate_call_result_287072)
    # Getting the type of 'ext' (line 209)
    ext_287073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 11), 'ext')
    # Assigning a type to the variable 'stypy_return_type' (line 209)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 4), 'stypy_return_type', ext_287073)
    
    # ################# End of 'const_ext(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'const_ext' in the type store
    # Getting the type of 'stypy_return_type' (line 158)
    stypy_return_type_287074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_287074)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'const_ext'
    return stypy_return_type_287074

# Assigning a type to the variable 'const_ext' (line 158)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 0), 'const_ext', const_ext)

@norecursion
def zero_ext(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_287075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 24), 'int')
    defaults = [int_287075]
    # Create a new context for function 'zero_ext'
    module_type_store = module_type_store.open_function_context('zero_ext', 212, 0, False)
    
    # Passed parameters checking function
    zero_ext.stypy_localization = localization
    zero_ext.stypy_type_of_self = None
    zero_ext.stypy_type_store = module_type_store
    zero_ext.stypy_function_name = 'zero_ext'
    zero_ext.stypy_param_names_list = ['x', 'n', 'axis']
    zero_ext.stypy_varargs_param_name = None
    zero_ext.stypy_kwargs_param_name = None
    zero_ext.stypy_call_defaults = defaults
    zero_ext.stypy_call_varargs = varargs
    zero_ext.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'zero_ext', ['x', 'n', 'axis'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'zero_ext', localization, ['x', 'n', 'axis'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'zero_ext(...)' code ##################

    str_287076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, (-1)), 'str', '\n    Zero padding at the boundaries of an array\n\n    Generate a new ndarray that is a zero padded extension of `x` along\n    an axis.\n\n    Parameters\n    ----------\n    x : ndarray\n        The array to be extended.\n    n : int\n        The number of elements by which to extend `x` at each end of the\n        axis.\n    axis : int, optional\n        The axis along which to extend `x`.  Default is -1.\n\n    Examples\n    --------\n    >>> from scipy.signal._arraytools import zero_ext\n    >>> a = np.array([[1, 2, 3, 4, 5], [0, 1, 4, 9, 16]])\n    >>> zero_ext(a, 2)\n    array([[ 0,  0,  1,  2,  3,  4,  5,  0,  0],\n           [ 0,  0,  0,  1,  4,  9, 16,  0,  0]])\n    ')
    
    
    # Getting the type of 'n' (line 237)
    n_287077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 7), 'n')
    int_287078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 11), 'int')
    # Applying the binary operator '<' (line 237)
    result_lt_287079 = python_operator(stypy.reporting.localization.Localization(__file__, 237, 7), '<', n_287077, int_287078)
    
    # Testing the type of an if condition (line 237)
    if_condition_287080 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 237, 4), result_lt_287079)
    # Assigning a type to the variable 'if_condition_287080' (line 237)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 4), 'if_condition_287080', if_condition_287080)
    # SSA begins for if statement (line 237)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'x' (line 238)
    x_287081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 15), 'x')
    # Assigning a type to the variable 'stypy_return_type' (line 238)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 8), 'stypy_return_type', x_287081)
    # SSA join for if statement (line 237)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 239):
    
    # Call to list(...): (line 239)
    # Processing the call arguments (line 239)
    # Getting the type of 'x' (line 239)
    x_287083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 23), 'x', False)
    # Obtaining the member 'shape' of a type (line 239)
    shape_287084 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 23), x_287083, 'shape')
    # Processing the call keyword arguments (line 239)
    kwargs_287085 = {}
    # Getting the type of 'list' (line 239)
    list_287082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 18), 'list', False)
    # Calling list(args, kwargs) (line 239)
    list_call_result_287086 = invoke(stypy.reporting.localization.Localization(__file__, 239, 18), list_287082, *[shape_287084], **kwargs_287085)
    
    # Assigning a type to the variable 'zeros_shape' (line 239)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 4), 'zeros_shape', list_call_result_287086)
    
    # Assigning a Name to a Subscript (line 240):
    # Getting the type of 'n' (line 240)
    n_287087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 24), 'n')
    # Getting the type of 'zeros_shape' (line 240)
    zeros_shape_287088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 4), 'zeros_shape')
    # Getting the type of 'axis' (line 240)
    axis_287089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 16), 'axis')
    # Storing an element on a container (line 240)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 240, 4), zeros_shape_287088, (axis_287089, n_287087))
    
    # Assigning a Call to a Name (line 241):
    
    # Call to zeros(...): (line 241)
    # Processing the call arguments (line 241)
    # Getting the type of 'zeros_shape' (line 241)
    zeros_shape_287092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 21), 'zeros_shape', False)
    # Processing the call keyword arguments (line 241)
    # Getting the type of 'x' (line 241)
    x_287093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 40), 'x', False)
    # Obtaining the member 'dtype' of a type (line 241)
    dtype_287094 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 40), x_287093, 'dtype')
    keyword_287095 = dtype_287094
    kwargs_287096 = {'dtype': keyword_287095}
    # Getting the type of 'np' (line 241)
    np_287090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 12), 'np', False)
    # Obtaining the member 'zeros' of a type (line 241)
    zeros_287091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 12), np_287090, 'zeros')
    # Calling zeros(args, kwargs) (line 241)
    zeros_call_result_287097 = invoke(stypy.reporting.localization.Localization(__file__, 241, 12), zeros_287091, *[zeros_shape_287092], **kwargs_287096)
    
    # Assigning a type to the variable 'zeros' (line 241)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 4), 'zeros', zeros_call_result_287097)
    
    # Assigning a Call to a Name (line 242):
    
    # Call to concatenate(...): (line 242)
    # Processing the call arguments (line 242)
    
    # Obtaining an instance of the builtin type 'tuple' (line 242)
    tuple_287100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 26), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 242)
    # Adding element type (line 242)
    # Getting the type of 'zeros' (line 242)
    zeros_287101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 26), 'zeros', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 242, 26), tuple_287100, zeros_287101)
    # Adding element type (line 242)
    # Getting the type of 'x' (line 242)
    x_287102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 33), 'x', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 242, 26), tuple_287100, x_287102)
    # Adding element type (line 242)
    # Getting the type of 'zeros' (line 242)
    zeros_287103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 36), 'zeros', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 242, 26), tuple_287100, zeros_287103)
    
    # Processing the call keyword arguments (line 242)
    # Getting the type of 'axis' (line 242)
    axis_287104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 49), 'axis', False)
    keyword_287105 = axis_287104
    kwargs_287106 = {'axis': keyword_287105}
    # Getting the type of 'np' (line 242)
    np_287098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 10), 'np', False)
    # Obtaining the member 'concatenate' of a type (line 242)
    concatenate_287099 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 10), np_287098, 'concatenate')
    # Calling concatenate(args, kwargs) (line 242)
    concatenate_call_result_287107 = invoke(stypy.reporting.localization.Localization(__file__, 242, 10), concatenate_287099, *[tuple_287100], **kwargs_287106)
    
    # Assigning a type to the variable 'ext' (line 242)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 4), 'ext', concatenate_call_result_287107)
    # Getting the type of 'ext' (line 243)
    ext_287108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 11), 'ext')
    # Assigning a type to the variable 'stypy_return_type' (line 243)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 4), 'stypy_return_type', ext_287108)
    
    # ################# End of 'zero_ext(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'zero_ext' in the type store
    # Getting the type of 'stypy_return_type' (line 212)
    stypy_return_type_287109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_287109)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'zero_ext'
    return stypy_return_type_287109

# Assigning a type to the variable 'zero_ext' (line 212)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 0), 'zero_ext', zero_ext)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
