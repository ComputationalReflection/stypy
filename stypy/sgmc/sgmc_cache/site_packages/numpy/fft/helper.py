
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Discrete Fourier Transforms - helper.py
3: 
4: '''
5: from __future__ import division, absolute_import, print_function
6: 
7: from numpy.compat import integer_types
8: from numpy.core import (
9:         asarray, concatenate, arange, take, integer, empty
10:         )
11: 
12: # Created by Pearu Peterson, September 2002
13: 
14: __all__ = ['fftshift', 'ifftshift', 'fftfreq', 'rfftfreq']
15: 
16: integer_types = integer_types + (integer,)
17: 
18: 
19: def fftshift(x, axes=None):
20:     '''
21:     Shift the zero-frequency component to the center of the spectrum.
22: 
23:     This function swaps half-spaces for all axes listed (defaults to all).
24:     Note that ``y[0]`` is the Nyquist component only if ``len(x)`` is even.
25: 
26:     Parameters
27:     ----------
28:     x : array_like
29:         Input array.
30:     axes : int or shape tuple, optional
31:         Axes over which to shift.  Default is None, which shifts all axes.
32: 
33:     Returns
34:     -------
35:     y : ndarray
36:         The shifted array.
37: 
38:     See Also
39:     --------
40:     ifftshift : The inverse of `fftshift`.
41: 
42:     Examples
43:     --------
44:     >>> freqs = np.fft.fftfreq(10, 0.1)
45:     >>> freqs
46:     array([ 0.,  1.,  2.,  3.,  4., -5., -4., -3., -2., -1.])
47:     >>> np.fft.fftshift(freqs)
48:     array([-5., -4., -3., -2., -1.,  0.,  1.,  2.,  3.,  4.])
49: 
50:     Shift the zero-frequency component only along the second axis:
51: 
52:     >>> freqs = np.fft.fftfreq(9, d=1./9).reshape(3, 3)
53:     >>> freqs
54:     array([[ 0.,  1.,  2.],
55:            [ 3.,  4., -4.],
56:            [-3., -2., -1.]])
57:     >>> np.fft.fftshift(freqs, axes=(1,))
58:     array([[ 2.,  0.,  1.],
59:            [-4.,  3.,  4.],
60:            [-1., -3., -2.]])
61: 
62:     '''
63:     tmp = asarray(x)
64:     ndim = len(tmp.shape)
65:     if axes is None:
66:         axes = list(range(ndim))
67:     elif isinstance(axes, integer_types):
68:         axes = (axes,)
69:     y = tmp
70:     for k in axes:
71:         n = tmp.shape[k]
72:         p2 = (n+1)//2
73:         mylist = concatenate((arange(p2, n), arange(p2)))
74:         y = take(y, mylist, k)
75:     return y
76: 
77: 
78: def ifftshift(x, axes=None):
79:     '''
80:     The inverse of `fftshift`. Although identical for even-length `x`, the
81:     functions differ by one sample for odd-length `x`.
82: 
83:     Parameters
84:     ----------
85:     x : array_like
86:         Input array.
87:     axes : int or shape tuple, optional
88:         Axes over which to calculate.  Defaults to None, which shifts all axes.
89: 
90:     Returns
91:     -------
92:     y : ndarray
93:         The shifted array.
94: 
95:     See Also
96:     --------
97:     fftshift : Shift zero-frequency component to the center of the spectrum.
98: 
99:     Examples
100:     --------
101:     >>> freqs = np.fft.fftfreq(9, d=1./9).reshape(3, 3)
102:     >>> freqs
103:     array([[ 0.,  1.,  2.],
104:            [ 3.,  4., -4.],
105:            [-3., -2., -1.]])
106:     >>> np.fft.ifftshift(np.fft.fftshift(freqs))
107:     array([[ 0.,  1.,  2.],
108:            [ 3.,  4., -4.],
109:            [-3., -2., -1.]])
110: 
111:     '''
112:     tmp = asarray(x)
113:     ndim = len(tmp.shape)
114:     if axes is None:
115:         axes = list(range(ndim))
116:     elif isinstance(axes, integer_types):
117:         axes = (axes,)
118:     y = tmp
119:     for k in axes:
120:         n = tmp.shape[k]
121:         p2 = n-(n+1)//2
122:         mylist = concatenate((arange(p2, n), arange(p2)))
123:         y = take(y, mylist, k)
124:     return y
125: 
126: 
127: def fftfreq(n, d=1.0):
128:     '''
129:     Return the Discrete Fourier Transform sample frequencies.
130: 
131:     The returned float array `f` contains the frequency bin centers in cycles
132:     per unit of the sample spacing (with zero at the start).  For instance, if
133:     the sample spacing is in seconds, then the frequency unit is cycles/second.
134: 
135:     Given a window length `n` and a sample spacing `d`::
136: 
137:       f = [0, 1, ...,   n/2-1,     -n/2, ..., -1] / (d*n)   if n is even
138:       f = [0, 1, ..., (n-1)/2, -(n-1)/2, ..., -1] / (d*n)   if n is odd
139: 
140:     Parameters
141:     ----------
142:     n : int
143:         Window length.
144:     d : scalar, optional
145:         Sample spacing (inverse of the sampling rate). Defaults to 1.
146: 
147:     Returns
148:     -------
149:     f : ndarray
150:         Array of length `n` containing the sample frequencies.
151: 
152:     Examples
153:     --------
154:     >>> signal = np.array([-2, 8, 6, 4, 1, 0, 3, 5], dtype=float)
155:     >>> fourier = np.fft.fft(signal)
156:     >>> n = signal.size
157:     >>> timestep = 0.1
158:     >>> freq = np.fft.fftfreq(n, d=timestep)
159:     >>> freq
160:     array([ 0.  ,  1.25,  2.5 ,  3.75, -5.  , -3.75, -2.5 , -1.25])
161: 
162:     '''
163:     if not isinstance(n, integer_types):
164:         raise ValueError("n should be an integer")
165:     val = 1.0 / (n * d)
166:     results = empty(n, int)
167:     N = (n-1)//2 + 1
168:     p1 = arange(0, N, dtype=int)
169:     results[:N] = p1
170:     p2 = arange(-(n//2), 0, dtype=int)
171:     results[N:] = p2
172:     return results * val
173:     #return hstack((arange(0,(n-1)/2 + 1), arange(-(n/2),0))) / (n*d)
174: 
175: 
176: def rfftfreq(n, d=1.0):
177:     '''
178:     Return the Discrete Fourier Transform sample frequencies
179:     (for usage with rfft, irfft).
180: 
181:     The returned float array `f` contains the frequency bin centers in cycles
182:     per unit of the sample spacing (with zero at the start).  For instance, if
183:     the sample spacing is in seconds, then the frequency unit is cycles/second.
184: 
185:     Given a window length `n` and a sample spacing `d`::
186: 
187:       f = [0, 1, ...,     n/2-1,     n/2] / (d*n)   if n is even
188:       f = [0, 1, ..., (n-1)/2-1, (n-1)/2] / (d*n)   if n is odd
189: 
190:     Unlike `fftfreq` (but like `scipy.fftpack.rfftfreq`)
191:     the Nyquist frequency component is considered to be positive.
192: 
193:     Parameters
194:     ----------
195:     n : int
196:         Window length.
197:     d : scalar, optional
198:         Sample spacing (inverse of the sampling rate). Defaults to 1.
199: 
200:     Returns
201:     -------
202:     f : ndarray
203:         Array of length ``n//2 + 1`` containing the sample frequencies.
204: 
205:     Examples
206:     --------
207:     >>> signal = np.array([-2, 8, 6, 4, 1, 0, 3, 5, -3, 4], dtype=float)
208:     >>> fourier = np.fft.rfft(signal)
209:     >>> n = signal.size
210:     >>> sample_rate = 100
211:     >>> freq = np.fft.fftfreq(n, d=1./sample_rate)
212:     >>> freq
213:     array([  0.,  10.,  20.,  30.,  40., -50., -40., -30., -20., -10.])
214:     >>> freq = np.fft.rfftfreq(n, d=1./sample_rate)
215:     >>> freq
216:     array([  0.,  10.,  20.,  30.,  40.,  50.])
217: 
218:     '''
219:     if not isinstance(n, integer_types):
220:         raise ValueError("n should be an integer")
221:     val = 1.0/(n*d)
222:     N = n//2 + 1
223:     results = arange(0, N, dtype=int)
224:     return results * val
225: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_100844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, (-1)), 'str', '\nDiscrete Fourier Transforms - helper.py\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from numpy.compat import integer_types' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/fft/')
import_100845 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.compat')

if (type(import_100845) is not StypyTypeError):

    if (import_100845 != 'pyd_module'):
        __import__(import_100845)
        sys_modules_100846 = sys.modules[import_100845]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.compat', sys_modules_100846.module_type_store, module_type_store, ['integer_types'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_100846, sys_modules_100846.module_type_store, module_type_store)
    else:
        from numpy.compat import integer_types

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.compat', None, module_type_store, ['integer_types'], [integer_types])

else:
    # Assigning a type to the variable 'numpy.compat' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.compat', import_100845)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/fft/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from numpy.core import asarray, concatenate, arange, take, integer, empty' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/fft/')
import_100847 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy.core')

if (type(import_100847) is not StypyTypeError):

    if (import_100847 != 'pyd_module'):
        __import__(import_100847)
        sys_modules_100848 = sys.modules[import_100847]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy.core', sys_modules_100848.module_type_store, module_type_store, ['asarray', 'concatenate', 'arange', 'take', 'integer', 'empty'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_100848, sys_modules_100848.module_type_store, module_type_store)
    else:
        from numpy.core import asarray, concatenate, arange, take, integer, empty

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy.core', None, module_type_store, ['asarray', 'concatenate', 'arange', 'take', 'integer', 'empty'], [asarray, concatenate, arange, take, integer, empty])

else:
    # Assigning a type to the variable 'numpy.core' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy.core', import_100847)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/fft/')


# Assigning a List to a Name (line 14):
__all__ = ['fftshift', 'ifftshift', 'fftfreq', 'rfftfreq']
module_type_store.set_exportable_members(['fftshift', 'ifftshift', 'fftfreq', 'rfftfreq'])

# Obtaining an instance of the builtin type 'list' (line 14)
list_100849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 14)
# Adding element type (line 14)
str_100850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 11), 'str', 'fftshift')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 10), list_100849, str_100850)
# Adding element type (line 14)
str_100851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 23), 'str', 'ifftshift')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 10), list_100849, str_100851)
# Adding element type (line 14)
str_100852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 36), 'str', 'fftfreq')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 10), list_100849, str_100852)
# Adding element type (line 14)
str_100853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 47), 'str', 'rfftfreq')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 10), list_100849, str_100853)

# Assigning a type to the variable '__all__' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), '__all__', list_100849)

# Assigning a BinOp to a Name (line 16):
# Getting the type of 'integer_types' (line 16)
integer_types_100854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 16), 'integer_types')

# Obtaining an instance of the builtin type 'tuple' (line 16)
tuple_100855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 33), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 16)
# Adding element type (line 16)
# Getting the type of 'integer' (line 16)
integer_100856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 33), 'integer')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 33), tuple_100855, integer_100856)

# Applying the binary operator '+' (line 16)
result_add_100857 = python_operator(stypy.reporting.localization.Localization(__file__, 16, 16), '+', integer_types_100854, tuple_100855)

# Assigning a type to the variable 'integer_types' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'integer_types', result_add_100857)

@norecursion
def fftshift(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 19)
    None_100858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 21), 'None')
    defaults = [None_100858]
    # Create a new context for function 'fftshift'
    module_type_store = module_type_store.open_function_context('fftshift', 19, 0, False)
    
    # Passed parameters checking function
    fftshift.stypy_localization = localization
    fftshift.stypy_type_of_self = None
    fftshift.stypy_type_store = module_type_store
    fftshift.stypy_function_name = 'fftshift'
    fftshift.stypy_param_names_list = ['x', 'axes']
    fftshift.stypy_varargs_param_name = None
    fftshift.stypy_kwargs_param_name = None
    fftshift.stypy_call_defaults = defaults
    fftshift.stypy_call_varargs = varargs
    fftshift.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'fftshift', ['x', 'axes'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'fftshift', localization, ['x', 'axes'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'fftshift(...)' code ##################

    str_100859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, (-1)), 'str', '\n    Shift the zero-frequency component to the center of the spectrum.\n\n    This function swaps half-spaces for all axes listed (defaults to all).\n    Note that ``y[0]`` is the Nyquist component only if ``len(x)`` is even.\n\n    Parameters\n    ----------\n    x : array_like\n        Input array.\n    axes : int or shape tuple, optional\n        Axes over which to shift.  Default is None, which shifts all axes.\n\n    Returns\n    -------\n    y : ndarray\n        The shifted array.\n\n    See Also\n    --------\n    ifftshift : The inverse of `fftshift`.\n\n    Examples\n    --------\n    >>> freqs = np.fft.fftfreq(10, 0.1)\n    >>> freqs\n    array([ 0.,  1.,  2.,  3.,  4., -5., -4., -3., -2., -1.])\n    >>> np.fft.fftshift(freqs)\n    array([-5., -4., -3., -2., -1.,  0.,  1.,  2.,  3.,  4.])\n\n    Shift the zero-frequency component only along the second axis:\n\n    >>> freqs = np.fft.fftfreq(9, d=1./9).reshape(3, 3)\n    >>> freqs\n    array([[ 0.,  1.,  2.],\n           [ 3.,  4., -4.],\n           [-3., -2., -1.]])\n    >>> np.fft.fftshift(freqs, axes=(1,))\n    array([[ 2.,  0.,  1.],\n           [-4.,  3.,  4.],\n           [-1., -3., -2.]])\n\n    ')
    
    # Assigning a Call to a Name (line 63):
    
    # Call to asarray(...): (line 63)
    # Processing the call arguments (line 63)
    # Getting the type of 'x' (line 63)
    x_100861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 18), 'x', False)
    # Processing the call keyword arguments (line 63)
    kwargs_100862 = {}
    # Getting the type of 'asarray' (line 63)
    asarray_100860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 10), 'asarray', False)
    # Calling asarray(args, kwargs) (line 63)
    asarray_call_result_100863 = invoke(stypy.reporting.localization.Localization(__file__, 63, 10), asarray_100860, *[x_100861], **kwargs_100862)
    
    # Assigning a type to the variable 'tmp' (line 63)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'tmp', asarray_call_result_100863)
    
    # Assigning a Call to a Name (line 64):
    
    # Call to len(...): (line 64)
    # Processing the call arguments (line 64)
    # Getting the type of 'tmp' (line 64)
    tmp_100865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 15), 'tmp', False)
    # Obtaining the member 'shape' of a type (line 64)
    shape_100866 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 15), tmp_100865, 'shape')
    # Processing the call keyword arguments (line 64)
    kwargs_100867 = {}
    # Getting the type of 'len' (line 64)
    len_100864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 11), 'len', False)
    # Calling len(args, kwargs) (line 64)
    len_call_result_100868 = invoke(stypy.reporting.localization.Localization(__file__, 64, 11), len_100864, *[shape_100866], **kwargs_100867)
    
    # Assigning a type to the variable 'ndim' (line 64)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'ndim', len_call_result_100868)
    
    # Type idiom detected: calculating its left and rigth part (line 65)
    # Getting the type of 'axes' (line 65)
    axes_100869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 7), 'axes')
    # Getting the type of 'None' (line 65)
    None_100870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 15), 'None')
    
    (may_be_100871, more_types_in_union_100872) = may_be_none(axes_100869, None_100870)

    if may_be_100871:

        if more_types_in_union_100872:
            # Runtime conditional SSA (line 65)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 66):
        
        # Call to list(...): (line 66)
        # Processing the call arguments (line 66)
        
        # Call to range(...): (line 66)
        # Processing the call arguments (line 66)
        # Getting the type of 'ndim' (line 66)
        ndim_100875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 26), 'ndim', False)
        # Processing the call keyword arguments (line 66)
        kwargs_100876 = {}
        # Getting the type of 'range' (line 66)
        range_100874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 20), 'range', False)
        # Calling range(args, kwargs) (line 66)
        range_call_result_100877 = invoke(stypy.reporting.localization.Localization(__file__, 66, 20), range_100874, *[ndim_100875], **kwargs_100876)
        
        # Processing the call keyword arguments (line 66)
        kwargs_100878 = {}
        # Getting the type of 'list' (line 66)
        list_100873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 15), 'list', False)
        # Calling list(args, kwargs) (line 66)
        list_call_result_100879 = invoke(stypy.reporting.localization.Localization(__file__, 66, 15), list_100873, *[range_call_result_100877], **kwargs_100878)
        
        # Assigning a type to the variable 'axes' (line 66)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'axes', list_call_result_100879)

        if more_types_in_union_100872:
            # Runtime conditional SSA for else branch (line 65)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_100871) or more_types_in_union_100872):
        
        
        # Call to isinstance(...): (line 67)
        # Processing the call arguments (line 67)
        # Getting the type of 'axes' (line 67)
        axes_100881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 20), 'axes', False)
        # Getting the type of 'integer_types' (line 67)
        integer_types_100882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 26), 'integer_types', False)
        # Processing the call keyword arguments (line 67)
        kwargs_100883 = {}
        # Getting the type of 'isinstance' (line 67)
        isinstance_100880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 9), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 67)
        isinstance_call_result_100884 = invoke(stypy.reporting.localization.Localization(__file__, 67, 9), isinstance_100880, *[axes_100881, integer_types_100882], **kwargs_100883)
        
        # Testing the type of an if condition (line 67)
        if_condition_100885 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 67, 9), isinstance_call_result_100884)
        # Assigning a type to the variable 'if_condition_100885' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 9), 'if_condition_100885', if_condition_100885)
        # SSA begins for if statement (line 67)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Tuple to a Name (line 68):
        
        # Obtaining an instance of the builtin type 'tuple' (line 68)
        tuple_100886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 16), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 68)
        # Adding element type (line 68)
        # Getting the type of 'axes' (line 68)
        axes_100887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 16), 'axes')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 16), tuple_100886, axes_100887)
        
        # Assigning a type to the variable 'axes' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'axes', tuple_100886)
        # SSA join for if statement (line 67)
        module_type_store = module_type_store.join_ssa_context()
        

        if (may_be_100871 and more_types_in_union_100872):
            # SSA join for if statement (line 65)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Name to a Name (line 69):
    # Getting the type of 'tmp' (line 69)
    tmp_100888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'tmp')
    # Assigning a type to the variable 'y' (line 69)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'y', tmp_100888)
    
    # Getting the type of 'axes' (line 70)
    axes_100889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 13), 'axes')
    # Testing the type of a for loop iterable (line 70)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 70, 4), axes_100889)
    # Getting the type of the for loop variable (line 70)
    for_loop_var_100890 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 70, 4), axes_100889)
    # Assigning a type to the variable 'k' (line 70)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'k', for_loop_var_100890)
    # SSA begins for a for statement (line 70)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Subscript to a Name (line 71):
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 71)
    k_100891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 22), 'k')
    # Getting the type of 'tmp' (line 71)
    tmp_100892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 12), 'tmp')
    # Obtaining the member 'shape' of a type (line 71)
    shape_100893 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 12), tmp_100892, 'shape')
    # Obtaining the member '__getitem__' of a type (line 71)
    getitem___100894 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 12), shape_100893, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 71)
    subscript_call_result_100895 = invoke(stypy.reporting.localization.Localization(__file__, 71, 12), getitem___100894, k_100891)
    
    # Assigning a type to the variable 'n' (line 71)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'n', subscript_call_result_100895)
    
    # Assigning a BinOp to a Name (line 72):
    # Getting the type of 'n' (line 72)
    n_100896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 14), 'n')
    int_100897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 16), 'int')
    # Applying the binary operator '+' (line 72)
    result_add_100898 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 14), '+', n_100896, int_100897)
    
    int_100899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 20), 'int')
    # Applying the binary operator '//' (line 72)
    result_floordiv_100900 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 13), '//', result_add_100898, int_100899)
    
    # Assigning a type to the variable 'p2' (line 72)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'p2', result_floordiv_100900)
    
    # Assigning a Call to a Name (line 73):
    
    # Call to concatenate(...): (line 73)
    # Processing the call arguments (line 73)
    
    # Obtaining an instance of the builtin type 'tuple' (line 73)
    tuple_100902 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 30), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 73)
    # Adding element type (line 73)
    
    # Call to arange(...): (line 73)
    # Processing the call arguments (line 73)
    # Getting the type of 'p2' (line 73)
    p2_100904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 37), 'p2', False)
    # Getting the type of 'n' (line 73)
    n_100905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 41), 'n', False)
    # Processing the call keyword arguments (line 73)
    kwargs_100906 = {}
    # Getting the type of 'arange' (line 73)
    arange_100903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 30), 'arange', False)
    # Calling arange(args, kwargs) (line 73)
    arange_call_result_100907 = invoke(stypy.reporting.localization.Localization(__file__, 73, 30), arange_100903, *[p2_100904, n_100905], **kwargs_100906)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 30), tuple_100902, arange_call_result_100907)
    # Adding element type (line 73)
    
    # Call to arange(...): (line 73)
    # Processing the call arguments (line 73)
    # Getting the type of 'p2' (line 73)
    p2_100909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 52), 'p2', False)
    # Processing the call keyword arguments (line 73)
    kwargs_100910 = {}
    # Getting the type of 'arange' (line 73)
    arange_100908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 45), 'arange', False)
    # Calling arange(args, kwargs) (line 73)
    arange_call_result_100911 = invoke(stypy.reporting.localization.Localization(__file__, 73, 45), arange_100908, *[p2_100909], **kwargs_100910)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 30), tuple_100902, arange_call_result_100911)
    
    # Processing the call keyword arguments (line 73)
    kwargs_100912 = {}
    # Getting the type of 'concatenate' (line 73)
    concatenate_100901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 17), 'concatenate', False)
    # Calling concatenate(args, kwargs) (line 73)
    concatenate_call_result_100913 = invoke(stypy.reporting.localization.Localization(__file__, 73, 17), concatenate_100901, *[tuple_100902], **kwargs_100912)
    
    # Assigning a type to the variable 'mylist' (line 73)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'mylist', concatenate_call_result_100913)
    
    # Assigning a Call to a Name (line 74):
    
    # Call to take(...): (line 74)
    # Processing the call arguments (line 74)
    # Getting the type of 'y' (line 74)
    y_100915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 17), 'y', False)
    # Getting the type of 'mylist' (line 74)
    mylist_100916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 20), 'mylist', False)
    # Getting the type of 'k' (line 74)
    k_100917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 28), 'k', False)
    # Processing the call keyword arguments (line 74)
    kwargs_100918 = {}
    # Getting the type of 'take' (line 74)
    take_100914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 12), 'take', False)
    # Calling take(args, kwargs) (line 74)
    take_call_result_100919 = invoke(stypy.reporting.localization.Localization(__file__, 74, 12), take_100914, *[y_100915, mylist_100916, k_100917], **kwargs_100918)
    
    # Assigning a type to the variable 'y' (line 74)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'y', take_call_result_100919)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'y' (line 75)
    y_100920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 11), 'y')
    # Assigning a type to the variable 'stypy_return_type' (line 75)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'stypy_return_type', y_100920)
    
    # ################# End of 'fftshift(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'fftshift' in the type store
    # Getting the type of 'stypy_return_type' (line 19)
    stypy_return_type_100921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_100921)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'fftshift'
    return stypy_return_type_100921

# Assigning a type to the variable 'fftshift' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'fftshift', fftshift)

@norecursion
def ifftshift(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 78)
    None_100922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 22), 'None')
    defaults = [None_100922]
    # Create a new context for function 'ifftshift'
    module_type_store = module_type_store.open_function_context('ifftshift', 78, 0, False)
    
    # Passed parameters checking function
    ifftshift.stypy_localization = localization
    ifftshift.stypy_type_of_self = None
    ifftshift.stypy_type_store = module_type_store
    ifftshift.stypy_function_name = 'ifftshift'
    ifftshift.stypy_param_names_list = ['x', 'axes']
    ifftshift.stypy_varargs_param_name = None
    ifftshift.stypy_kwargs_param_name = None
    ifftshift.stypy_call_defaults = defaults
    ifftshift.stypy_call_varargs = varargs
    ifftshift.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'ifftshift', ['x', 'axes'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'ifftshift', localization, ['x', 'axes'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'ifftshift(...)' code ##################

    str_100923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, (-1)), 'str', '\n    The inverse of `fftshift`. Although identical for even-length `x`, the\n    functions differ by one sample for odd-length `x`.\n\n    Parameters\n    ----------\n    x : array_like\n        Input array.\n    axes : int or shape tuple, optional\n        Axes over which to calculate.  Defaults to None, which shifts all axes.\n\n    Returns\n    -------\n    y : ndarray\n        The shifted array.\n\n    See Also\n    --------\n    fftshift : Shift zero-frequency component to the center of the spectrum.\n\n    Examples\n    --------\n    >>> freqs = np.fft.fftfreq(9, d=1./9).reshape(3, 3)\n    >>> freqs\n    array([[ 0.,  1.,  2.],\n           [ 3.,  4., -4.],\n           [-3., -2., -1.]])\n    >>> np.fft.ifftshift(np.fft.fftshift(freqs))\n    array([[ 0.,  1.,  2.],\n           [ 3.,  4., -4.],\n           [-3., -2., -1.]])\n\n    ')
    
    # Assigning a Call to a Name (line 112):
    
    # Call to asarray(...): (line 112)
    # Processing the call arguments (line 112)
    # Getting the type of 'x' (line 112)
    x_100925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 18), 'x', False)
    # Processing the call keyword arguments (line 112)
    kwargs_100926 = {}
    # Getting the type of 'asarray' (line 112)
    asarray_100924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 10), 'asarray', False)
    # Calling asarray(args, kwargs) (line 112)
    asarray_call_result_100927 = invoke(stypy.reporting.localization.Localization(__file__, 112, 10), asarray_100924, *[x_100925], **kwargs_100926)
    
    # Assigning a type to the variable 'tmp' (line 112)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'tmp', asarray_call_result_100927)
    
    # Assigning a Call to a Name (line 113):
    
    # Call to len(...): (line 113)
    # Processing the call arguments (line 113)
    # Getting the type of 'tmp' (line 113)
    tmp_100929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 15), 'tmp', False)
    # Obtaining the member 'shape' of a type (line 113)
    shape_100930 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 15), tmp_100929, 'shape')
    # Processing the call keyword arguments (line 113)
    kwargs_100931 = {}
    # Getting the type of 'len' (line 113)
    len_100928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 11), 'len', False)
    # Calling len(args, kwargs) (line 113)
    len_call_result_100932 = invoke(stypy.reporting.localization.Localization(__file__, 113, 11), len_100928, *[shape_100930], **kwargs_100931)
    
    # Assigning a type to the variable 'ndim' (line 113)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 4), 'ndim', len_call_result_100932)
    
    # Type idiom detected: calculating its left and rigth part (line 114)
    # Getting the type of 'axes' (line 114)
    axes_100933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 7), 'axes')
    # Getting the type of 'None' (line 114)
    None_100934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 15), 'None')
    
    (may_be_100935, more_types_in_union_100936) = may_be_none(axes_100933, None_100934)

    if may_be_100935:

        if more_types_in_union_100936:
            # Runtime conditional SSA (line 114)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 115):
        
        # Call to list(...): (line 115)
        # Processing the call arguments (line 115)
        
        # Call to range(...): (line 115)
        # Processing the call arguments (line 115)
        # Getting the type of 'ndim' (line 115)
        ndim_100939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 26), 'ndim', False)
        # Processing the call keyword arguments (line 115)
        kwargs_100940 = {}
        # Getting the type of 'range' (line 115)
        range_100938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 20), 'range', False)
        # Calling range(args, kwargs) (line 115)
        range_call_result_100941 = invoke(stypy.reporting.localization.Localization(__file__, 115, 20), range_100938, *[ndim_100939], **kwargs_100940)
        
        # Processing the call keyword arguments (line 115)
        kwargs_100942 = {}
        # Getting the type of 'list' (line 115)
        list_100937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 15), 'list', False)
        # Calling list(args, kwargs) (line 115)
        list_call_result_100943 = invoke(stypy.reporting.localization.Localization(__file__, 115, 15), list_100937, *[range_call_result_100941], **kwargs_100942)
        
        # Assigning a type to the variable 'axes' (line 115)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 8), 'axes', list_call_result_100943)

        if more_types_in_union_100936:
            # Runtime conditional SSA for else branch (line 114)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_100935) or more_types_in_union_100936):
        
        
        # Call to isinstance(...): (line 116)
        # Processing the call arguments (line 116)
        # Getting the type of 'axes' (line 116)
        axes_100945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 20), 'axes', False)
        # Getting the type of 'integer_types' (line 116)
        integer_types_100946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 26), 'integer_types', False)
        # Processing the call keyword arguments (line 116)
        kwargs_100947 = {}
        # Getting the type of 'isinstance' (line 116)
        isinstance_100944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 9), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 116)
        isinstance_call_result_100948 = invoke(stypy.reporting.localization.Localization(__file__, 116, 9), isinstance_100944, *[axes_100945, integer_types_100946], **kwargs_100947)
        
        # Testing the type of an if condition (line 116)
        if_condition_100949 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 116, 9), isinstance_call_result_100948)
        # Assigning a type to the variable 'if_condition_100949' (line 116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 9), 'if_condition_100949', if_condition_100949)
        # SSA begins for if statement (line 116)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Tuple to a Name (line 117):
        
        # Obtaining an instance of the builtin type 'tuple' (line 117)
        tuple_100950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 16), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 117)
        # Adding element type (line 117)
        # Getting the type of 'axes' (line 117)
        axes_100951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 16), 'axes')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 16), tuple_100950, axes_100951)
        
        # Assigning a type to the variable 'axes' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'axes', tuple_100950)
        # SSA join for if statement (line 116)
        module_type_store = module_type_store.join_ssa_context()
        

        if (may_be_100935 and more_types_in_union_100936):
            # SSA join for if statement (line 114)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Name to a Name (line 118):
    # Getting the type of 'tmp' (line 118)
    tmp_100952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'tmp')
    # Assigning a type to the variable 'y' (line 118)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 4), 'y', tmp_100952)
    
    # Getting the type of 'axes' (line 119)
    axes_100953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 13), 'axes')
    # Testing the type of a for loop iterable (line 119)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 119, 4), axes_100953)
    # Getting the type of the for loop variable (line 119)
    for_loop_var_100954 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 119, 4), axes_100953)
    # Assigning a type to the variable 'k' (line 119)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 4), 'k', for_loop_var_100954)
    # SSA begins for a for statement (line 119)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Subscript to a Name (line 120):
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 120)
    k_100955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 22), 'k')
    # Getting the type of 'tmp' (line 120)
    tmp_100956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 12), 'tmp')
    # Obtaining the member 'shape' of a type (line 120)
    shape_100957 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 12), tmp_100956, 'shape')
    # Obtaining the member '__getitem__' of a type (line 120)
    getitem___100958 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 12), shape_100957, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 120)
    subscript_call_result_100959 = invoke(stypy.reporting.localization.Localization(__file__, 120, 12), getitem___100958, k_100955)
    
    # Assigning a type to the variable 'n' (line 120)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'n', subscript_call_result_100959)
    
    # Assigning a BinOp to a Name (line 121):
    # Getting the type of 'n' (line 121)
    n_100960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 13), 'n')
    # Getting the type of 'n' (line 121)
    n_100961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 16), 'n')
    int_100962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 18), 'int')
    # Applying the binary operator '+' (line 121)
    result_add_100963 = python_operator(stypy.reporting.localization.Localization(__file__, 121, 16), '+', n_100961, int_100962)
    
    int_100964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 22), 'int')
    # Applying the binary operator '//' (line 121)
    result_floordiv_100965 = python_operator(stypy.reporting.localization.Localization(__file__, 121, 15), '//', result_add_100963, int_100964)
    
    # Applying the binary operator '-' (line 121)
    result_sub_100966 = python_operator(stypy.reporting.localization.Localization(__file__, 121, 13), '-', n_100960, result_floordiv_100965)
    
    # Assigning a type to the variable 'p2' (line 121)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'p2', result_sub_100966)
    
    # Assigning a Call to a Name (line 122):
    
    # Call to concatenate(...): (line 122)
    # Processing the call arguments (line 122)
    
    # Obtaining an instance of the builtin type 'tuple' (line 122)
    tuple_100968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 30), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 122)
    # Adding element type (line 122)
    
    # Call to arange(...): (line 122)
    # Processing the call arguments (line 122)
    # Getting the type of 'p2' (line 122)
    p2_100970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 37), 'p2', False)
    # Getting the type of 'n' (line 122)
    n_100971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 41), 'n', False)
    # Processing the call keyword arguments (line 122)
    kwargs_100972 = {}
    # Getting the type of 'arange' (line 122)
    arange_100969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 30), 'arange', False)
    # Calling arange(args, kwargs) (line 122)
    arange_call_result_100973 = invoke(stypy.reporting.localization.Localization(__file__, 122, 30), arange_100969, *[p2_100970, n_100971], **kwargs_100972)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 30), tuple_100968, arange_call_result_100973)
    # Adding element type (line 122)
    
    # Call to arange(...): (line 122)
    # Processing the call arguments (line 122)
    # Getting the type of 'p2' (line 122)
    p2_100975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 52), 'p2', False)
    # Processing the call keyword arguments (line 122)
    kwargs_100976 = {}
    # Getting the type of 'arange' (line 122)
    arange_100974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 45), 'arange', False)
    # Calling arange(args, kwargs) (line 122)
    arange_call_result_100977 = invoke(stypy.reporting.localization.Localization(__file__, 122, 45), arange_100974, *[p2_100975], **kwargs_100976)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 30), tuple_100968, arange_call_result_100977)
    
    # Processing the call keyword arguments (line 122)
    kwargs_100978 = {}
    # Getting the type of 'concatenate' (line 122)
    concatenate_100967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 17), 'concatenate', False)
    # Calling concatenate(args, kwargs) (line 122)
    concatenate_call_result_100979 = invoke(stypy.reporting.localization.Localization(__file__, 122, 17), concatenate_100967, *[tuple_100968], **kwargs_100978)
    
    # Assigning a type to the variable 'mylist' (line 122)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'mylist', concatenate_call_result_100979)
    
    # Assigning a Call to a Name (line 123):
    
    # Call to take(...): (line 123)
    # Processing the call arguments (line 123)
    # Getting the type of 'y' (line 123)
    y_100981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 17), 'y', False)
    # Getting the type of 'mylist' (line 123)
    mylist_100982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 20), 'mylist', False)
    # Getting the type of 'k' (line 123)
    k_100983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 28), 'k', False)
    # Processing the call keyword arguments (line 123)
    kwargs_100984 = {}
    # Getting the type of 'take' (line 123)
    take_100980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 12), 'take', False)
    # Calling take(args, kwargs) (line 123)
    take_call_result_100985 = invoke(stypy.reporting.localization.Localization(__file__, 123, 12), take_100980, *[y_100981, mylist_100982, k_100983], **kwargs_100984)
    
    # Assigning a type to the variable 'y' (line 123)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'y', take_call_result_100985)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'y' (line 124)
    y_100986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 11), 'y')
    # Assigning a type to the variable 'stypy_return_type' (line 124)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 4), 'stypy_return_type', y_100986)
    
    # ################# End of 'ifftshift(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'ifftshift' in the type store
    # Getting the type of 'stypy_return_type' (line 78)
    stypy_return_type_100987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_100987)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'ifftshift'
    return stypy_return_type_100987

# Assigning a type to the variable 'ifftshift' (line 78)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 0), 'ifftshift', ifftshift)

@norecursion
def fftfreq(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    float_100988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 17), 'float')
    defaults = [float_100988]
    # Create a new context for function 'fftfreq'
    module_type_store = module_type_store.open_function_context('fftfreq', 127, 0, False)
    
    # Passed parameters checking function
    fftfreq.stypy_localization = localization
    fftfreq.stypy_type_of_self = None
    fftfreq.stypy_type_store = module_type_store
    fftfreq.stypy_function_name = 'fftfreq'
    fftfreq.stypy_param_names_list = ['n', 'd']
    fftfreq.stypy_varargs_param_name = None
    fftfreq.stypy_kwargs_param_name = None
    fftfreq.stypy_call_defaults = defaults
    fftfreq.stypy_call_varargs = varargs
    fftfreq.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'fftfreq', ['n', 'd'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'fftfreq', localization, ['n', 'd'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'fftfreq(...)' code ##################

    str_100989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, (-1)), 'str', '\n    Return the Discrete Fourier Transform sample frequencies.\n\n    The returned float array `f` contains the frequency bin centers in cycles\n    per unit of the sample spacing (with zero at the start).  For instance, if\n    the sample spacing is in seconds, then the frequency unit is cycles/second.\n\n    Given a window length `n` and a sample spacing `d`::\n\n      f = [0, 1, ...,   n/2-1,     -n/2, ..., -1] / (d*n)   if n is even\n      f = [0, 1, ..., (n-1)/2, -(n-1)/2, ..., -1] / (d*n)   if n is odd\n\n    Parameters\n    ----------\n    n : int\n        Window length.\n    d : scalar, optional\n        Sample spacing (inverse of the sampling rate). Defaults to 1.\n\n    Returns\n    -------\n    f : ndarray\n        Array of length `n` containing the sample frequencies.\n\n    Examples\n    --------\n    >>> signal = np.array([-2, 8, 6, 4, 1, 0, 3, 5], dtype=float)\n    >>> fourier = np.fft.fft(signal)\n    >>> n = signal.size\n    >>> timestep = 0.1\n    >>> freq = np.fft.fftfreq(n, d=timestep)\n    >>> freq\n    array([ 0.  ,  1.25,  2.5 ,  3.75, -5.  , -3.75, -2.5 , -1.25])\n\n    ')
    
    
    
    # Call to isinstance(...): (line 163)
    # Processing the call arguments (line 163)
    # Getting the type of 'n' (line 163)
    n_100991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 22), 'n', False)
    # Getting the type of 'integer_types' (line 163)
    integer_types_100992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 25), 'integer_types', False)
    # Processing the call keyword arguments (line 163)
    kwargs_100993 = {}
    # Getting the type of 'isinstance' (line 163)
    isinstance_100990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 11), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 163)
    isinstance_call_result_100994 = invoke(stypy.reporting.localization.Localization(__file__, 163, 11), isinstance_100990, *[n_100991, integer_types_100992], **kwargs_100993)
    
    # Applying the 'not' unary operator (line 163)
    result_not__100995 = python_operator(stypy.reporting.localization.Localization(__file__, 163, 7), 'not', isinstance_call_result_100994)
    
    # Testing the type of an if condition (line 163)
    if_condition_100996 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 163, 4), result_not__100995)
    # Assigning a type to the variable 'if_condition_100996' (line 163)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 4), 'if_condition_100996', if_condition_100996)
    # SSA begins for if statement (line 163)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 164)
    # Processing the call arguments (line 164)
    str_100998 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 25), 'str', 'n should be an integer')
    # Processing the call keyword arguments (line 164)
    kwargs_100999 = {}
    # Getting the type of 'ValueError' (line 164)
    ValueError_100997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 164)
    ValueError_call_result_101000 = invoke(stypy.reporting.localization.Localization(__file__, 164, 14), ValueError_100997, *[str_100998], **kwargs_100999)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 164, 8), ValueError_call_result_101000, 'raise parameter', BaseException)
    # SSA join for if statement (line 163)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 165):
    float_101001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 10), 'float')
    # Getting the type of 'n' (line 165)
    n_101002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 17), 'n')
    # Getting the type of 'd' (line 165)
    d_101003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 21), 'd')
    # Applying the binary operator '*' (line 165)
    result_mul_101004 = python_operator(stypy.reporting.localization.Localization(__file__, 165, 17), '*', n_101002, d_101003)
    
    # Applying the binary operator 'div' (line 165)
    result_div_101005 = python_operator(stypy.reporting.localization.Localization(__file__, 165, 10), 'div', float_101001, result_mul_101004)
    
    # Assigning a type to the variable 'val' (line 165)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 4), 'val', result_div_101005)
    
    # Assigning a Call to a Name (line 166):
    
    # Call to empty(...): (line 166)
    # Processing the call arguments (line 166)
    # Getting the type of 'n' (line 166)
    n_101007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 20), 'n', False)
    # Getting the type of 'int' (line 166)
    int_101008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 23), 'int', False)
    # Processing the call keyword arguments (line 166)
    kwargs_101009 = {}
    # Getting the type of 'empty' (line 166)
    empty_101006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 14), 'empty', False)
    # Calling empty(args, kwargs) (line 166)
    empty_call_result_101010 = invoke(stypy.reporting.localization.Localization(__file__, 166, 14), empty_101006, *[n_101007, int_101008], **kwargs_101009)
    
    # Assigning a type to the variable 'results' (line 166)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 4), 'results', empty_call_result_101010)
    
    # Assigning a BinOp to a Name (line 167):
    # Getting the type of 'n' (line 167)
    n_101011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 9), 'n')
    int_101012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 11), 'int')
    # Applying the binary operator '-' (line 167)
    result_sub_101013 = python_operator(stypy.reporting.localization.Localization(__file__, 167, 9), '-', n_101011, int_101012)
    
    int_101014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 15), 'int')
    # Applying the binary operator '//' (line 167)
    result_floordiv_101015 = python_operator(stypy.reporting.localization.Localization(__file__, 167, 8), '//', result_sub_101013, int_101014)
    
    int_101016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 19), 'int')
    # Applying the binary operator '+' (line 167)
    result_add_101017 = python_operator(stypy.reporting.localization.Localization(__file__, 167, 8), '+', result_floordiv_101015, int_101016)
    
    # Assigning a type to the variable 'N' (line 167)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 4), 'N', result_add_101017)
    
    # Assigning a Call to a Name (line 168):
    
    # Call to arange(...): (line 168)
    # Processing the call arguments (line 168)
    int_101019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 16), 'int')
    # Getting the type of 'N' (line 168)
    N_101020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 19), 'N', False)
    # Processing the call keyword arguments (line 168)
    # Getting the type of 'int' (line 168)
    int_101021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 28), 'int', False)
    keyword_101022 = int_101021
    kwargs_101023 = {'dtype': keyword_101022}
    # Getting the type of 'arange' (line 168)
    arange_101018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 9), 'arange', False)
    # Calling arange(args, kwargs) (line 168)
    arange_call_result_101024 = invoke(stypy.reporting.localization.Localization(__file__, 168, 9), arange_101018, *[int_101019, N_101020], **kwargs_101023)
    
    # Assigning a type to the variable 'p1' (line 168)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 4), 'p1', arange_call_result_101024)
    
    # Assigning a Name to a Subscript (line 169):
    # Getting the type of 'p1' (line 169)
    p1_101025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 18), 'p1')
    # Getting the type of 'results' (line 169)
    results_101026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 4), 'results')
    # Getting the type of 'N' (line 169)
    N_101027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 13), 'N')
    slice_101028 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 169, 4), None, N_101027, None)
    # Storing an element on a container (line 169)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 169, 4), results_101026, (slice_101028, p1_101025))
    
    # Assigning a Call to a Name (line 170):
    
    # Call to arange(...): (line 170)
    # Processing the call arguments (line 170)
    
    # Getting the type of 'n' (line 170)
    n_101030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 18), 'n', False)
    int_101031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 21), 'int')
    # Applying the binary operator '//' (line 170)
    result_floordiv_101032 = python_operator(stypy.reporting.localization.Localization(__file__, 170, 18), '//', n_101030, int_101031)
    
    # Applying the 'usub' unary operator (line 170)
    result___neg___101033 = python_operator(stypy.reporting.localization.Localization(__file__, 170, 16), 'usub', result_floordiv_101032)
    
    int_101034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 25), 'int')
    # Processing the call keyword arguments (line 170)
    # Getting the type of 'int' (line 170)
    int_101035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 34), 'int', False)
    keyword_101036 = int_101035
    kwargs_101037 = {'dtype': keyword_101036}
    # Getting the type of 'arange' (line 170)
    arange_101029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 9), 'arange', False)
    # Calling arange(args, kwargs) (line 170)
    arange_call_result_101038 = invoke(stypy.reporting.localization.Localization(__file__, 170, 9), arange_101029, *[result___neg___101033, int_101034], **kwargs_101037)
    
    # Assigning a type to the variable 'p2' (line 170)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 4), 'p2', arange_call_result_101038)
    
    # Assigning a Name to a Subscript (line 171):
    # Getting the type of 'p2' (line 171)
    p2_101039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 18), 'p2')
    # Getting the type of 'results' (line 171)
    results_101040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 4), 'results')
    # Getting the type of 'N' (line 171)
    N_101041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 12), 'N')
    slice_101042 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 171, 4), N_101041, None, None)
    # Storing an element on a container (line 171)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 4), results_101040, (slice_101042, p2_101039))
    # Getting the type of 'results' (line 172)
    results_101043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 11), 'results')
    # Getting the type of 'val' (line 172)
    val_101044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 21), 'val')
    # Applying the binary operator '*' (line 172)
    result_mul_101045 = python_operator(stypy.reporting.localization.Localization(__file__, 172, 11), '*', results_101043, val_101044)
    
    # Assigning a type to the variable 'stypy_return_type' (line 172)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 4), 'stypy_return_type', result_mul_101045)
    
    # ################# End of 'fftfreq(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'fftfreq' in the type store
    # Getting the type of 'stypy_return_type' (line 127)
    stypy_return_type_101046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_101046)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'fftfreq'
    return stypy_return_type_101046

# Assigning a type to the variable 'fftfreq' (line 127)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 0), 'fftfreq', fftfreq)

@norecursion
def rfftfreq(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    float_101047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 18), 'float')
    defaults = [float_101047]
    # Create a new context for function 'rfftfreq'
    module_type_store = module_type_store.open_function_context('rfftfreq', 176, 0, False)
    
    # Passed parameters checking function
    rfftfreq.stypy_localization = localization
    rfftfreq.stypy_type_of_self = None
    rfftfreq.stypy_type_store = module_type_store
    rfftfreq.stypy_function_name = 'rfftfreq'
    rfftfreq.stypy_param_names_list = ['n', 'd']
    rfftfreq.stypy_varargs_param_name = None
    rfftfreq.stypy_kwargs_param_name = None
    rfftfreq.stypy_call_defaults = defaults
    rfftfreq.stypy_call_varargs = varargs
    rfftfreq.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'rfftfreq', ['n', 'd'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'rfftfreq', localization, ['n', 'd'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'rfftfreq(...)' code ##################

    str_101048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, (-1)), 'str', '\n    Return the Discrete Fourier Transform sample frequencies\n    (for usage with rfft, irfft).\n\n    The returned float array `f` contains the frequency bin centers in cycles\n    per unit of the sample spacing (with zero at the start).  For instance, if\n    the sample spacing is in seconds, then the frequency unit is cycles/second.\n\n    Given a window length `n` and a sample spacing `d`::\n\n      f = [0, 1, ...,     n/2-1,     n/2] / (d*n)   if n is even\n      f = [0, 1, ..., (n-1)/2-1, (n-1)/2] / (d*n)   if n is odd\n\n    Unlike `fftfreq` (but like `scipy.fftpack.rfftfreq`)\n    the Nyquist frequency component is considered to be positive.\n\n    Parameters\n    ----------\n    n : int\n        Window length.\n    d : scalar, optional\n        Sample spacing (inverse of the sampling rate). Defaults to 1.\n\n    Returns\n    -------\n    f : ndarray\n        Array of length ``n//2 + 1`` containing the sample frequencies.\n\n    Examples\n    --------\n    >>> signal = np.array([-2, 8, 6, 4, 1, 0, 3, 5, -3, 4], dtype=float)\n    >>> fourier = np.fft.rfft(signal)\n    >>> n = signal.size\n    >>> sample_rate = 100\n    >>> freq = np.fft.fftfreq(n, d=1./sample_rate)\n    >>> freq\n    array([  0.,  10.,  20.,  30.,  40., -50., -40., -30., -20., -10.])\n    >>> freq = np.fft.rfftfreq(n, d=1./sample_rate)\n    >>> freq\n    array([  0.,  10.,  20.,  30.,  40.,  50.])\n\n    ')
    
    
    
    # Call to isinstance(...): (line 219)
    # Processing the call arguments (line 219)
    # Getting the type of 'n' (line 219)
    n_101050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 22), 'n', False)
    # Getting the type of 'integer_types' (line 219)
    integer_types_101051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 25), 'integer_types', False)
    # Processing the call keyword arguments (line 219)
    kwargs_101052 = {}
    # Getting the type of 'isinstance' (line 219)
    isinstance_101049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 11), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 219)
    isinstance_call_result_101053 = invoke(stypy.reporting.localization.Localization(__file__, 219, 11), isinstance_101049, *[n_101050, integer_types_101051], **kwargs_101052)
    
    # Applying the 'not' unary operator (line 219)
    result_not__101054 = python_operator(stypy.reporting.localization.Localization(__file__, 219, 7), 'not', isinstance_call_result_101053)
    
    # Testing the type of an if condition (line 219)
    if_condition_101055 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 219, 4), result_not__101054)
    # Assigning a type to the variable 'if_condition_101055' (line 219)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 4), 'if_condition_101055', if_condition_101055)
    # SSA begins for if statement (line 219)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 220)
    # Processing the call arguments (line 220)
    str_101057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 25), 'str', 'n should be an integer')
    # Processing the call keyword arguments (line 220)
    kwargs_101058 = {}
    # Getting the type of 'ValueError' (line 220)
    ValueError_101056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 220)
    ValueError_call_result_101059 = invoke(stypy.reporting.localization.Localization(__file__, 220, 14), ValueError_101056, *[str_101057], **kwargs_101058)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 220, 8), ValueError_call_result_101059, 'raise parameter', BaseException)
    # SSA join for if statement (line 219)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 221):
    float_101060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 10), 'float')
    # Getting the type of 'n' (line 221)
    n_101061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 15), 'n')
    # Getting the type of 'd' (line 221)
    d_101062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 17), 'd')
    # Applying the binary operator '*' (line 221)
    result_mul_101063 = python_operator(stypy.reporting.localization.Localization(__file__, 221, 15), '*', n_101061, d_101062)
    
    # Applying the binary operator 'div' (line 221)
    result_div_101064 = python_operator(stypy.reporting.localization.Localization(__file__, 221, 10), 'div', float_101060, result_mul_101063)
    
    # Assigning a type to the variable 'val' (line 221)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 4), 'val', result_div_101064)
    
    # Assigning a BinOp to a Name (line 222):
    # Getting the type of 'n' (line 222)
    n_101065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 8), 'n')
    int_101066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 11), 'int')
    # Applying the binary operator '//' (line 222)
    result_floordiv_101067 = python_operator(stypy.reporting.localization.Localization(__file__, 222, 8), '//', n_101065, int_101066)
    
    int_101068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 15), 'int')
    # Applying the binary operator '+' (line 222)
    result_add_101069 = python_operator(stypy.reporting.localization.Localization(__file__, 222, 8), '+', result_floordiv_101067, int_101068)
    
    # Assigning a type to the variable 'N' (line 222)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 4), 'N', result_add_101069)
    
    # Assigning a Call to a Name (line 223):
    
    # Call to arange(...): (line 223)
    # Processing the call arguments (line 223)
    int_101071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 21), 'int')
    # Getting the type of 'N' (line 223)
    N_101072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 24), 'N', False)
    # Processing the call keyword arguments (line 223)
    # Getting the type of 'int' (line 223)
    int_101073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 33), 'int', False)
    keyword_101074 = int_101073
    kwargs_101075 = {'dtype': keyword_101074}
    # Getting the type of 'arange' (line 223)
    arange_101070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 14), 'arange', False)
    # Calling arange(args, kwargs) (line 223)
    arange_call_result_101076 = invoke(stypy.reporting.localization.Localization(__file__, 223, 14), arange_101070, *[int_101071, N_101072], **kwargs_101075)
    
    # Assigning a type to the variable 'results' (line 223)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 4), 'results', arange_call_result_101076)
    # Getting the type of 'results' (line 224)
    results_101077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 11), 'results')
    # Getting the type of 'val' (line 224)
    val_101078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 21), 'val')
    # Applying the binary operator '*' (line 224)
    result_mul_101079 = python_operator(stypy.reporting.localization.Localization(__file__, 224, 11), '*', results_101077, val_101078)
    
    # Assigning a type to the variable 'stypy_return_type' (line 224)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 4), 'stypy_return_type', result_mul_101079)
    
    # ################# End of 'rfftfreq(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'rfftfreq' in the type store
    # Getting the type of 'stypy_return_type' (line 176)
    stypy_return_type_101080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_101080)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'rfftfreq'
    return stypy_return_type_101080

# Assigning a type to the variable 'rfftfreq' (line 176)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 0), 'rfftfreq', rfftfreq)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
