
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: import operator
4: from numpy import arange
5: from numpy.fft.helper import fftshift, ifftshift, fftfreq
6: from bisect import bisect_left
7: 
8: __all__ = ['fftshift', 'ifftshift', 'fftfreq', 'rfftfreq', 'next_fast_len']
9: 
10: 
11: def rfftfreq(n, d=1.0):
12:     '''DFT sample frequencies (for usage with rfft, irfft).
13: 
14:     The returned float array contains the frequency bins in
15:     cycles/unit (with zero at the start) given a window length `n` and a
16:     sample spacing `d`::
17: 
18:       f = [0,1,1,2,2,...,n/2-1,n/2-1,n/2]/(d*n)   if n is even
19:       f = [0,1,1,2,2,...,n/2-1,n/2-1,n/2,n/2]/(d*n)   if n is odd
20: 
21:     Parameters
22:     ----------
23:     n : int
24:         Window length.
25:     d : scalar, optional
26:         Sample spacing. Default is 1.
27: 
28:     Returns
29:     -------
30:     out : ndarray
31:         The array of length `n`, containing the sample frequencies.
32: 
33:     Examples
34:     --------
35:     >>> from scipy import fftpack
36:     >>> sig = np.array([-2, 8, 6, 4, 1, 0, 3, 5], dtype=float)
37:     >>> sig_fft = fftpack.rfft(sig)
38:     >>> n = sig_fft.size
39:     >>> timestep = 0.1
40:     >>> freq = fftpack.rfftfreq(n, d=timestep)
41:     >>> freq
42:     array([ 0.  ,  1.25,  1.25,  2.5 ,  2.5 ,  3.75,  3.75,  5.  ])
43: 
44:     '''
45:     n = operator.index(n)
46:     if n < 0:
47:         raise ValueError("n = %s is not valid. "
48:                          "n must be a nonnegative integer." % n)
49: 
50:     return (arange(1, n + 1, dtype=int) // 2) / float(n * d)
51: 
52: 
53: def next_fast_len(target):
54:     '''
55:     Find the next fast size of input data to `fft`, for zero-padding, etc.
56: 
57:     SciPy's FFTPACK has efficient functions for radix {2, 3, 4, 5}, so this
58:     returns the next composite of the prime factors 2, 3, and 5 which is
59:     greater than or equal to `target`. (These are also known as 5-smooth
60:     numbers, regular numbers, or Hamming numbers.)
61: 
62:     Parameters
63:     ----------
64:     target : int
65:         Length to start searching from.  Must be a positive integer.
66: 
67:     Returns
68:     -------
69:     out : int
70:         The first 5-smooth number greater than or equal to `target`.
71: 
72:     Notes
73:     -----
74:     .. versionadded:: 0.18.0
75: 
76:     Examples
77:     --------
78:     On a particular machine, an FFT of prime length takes 133 ms:
79: 
80:     >>> from scipy import fftpack
81:     >>> min_len = 10007  # prime length is worst case for speed
82:     >>> a = np.random.randn(min_len)
83:     >>> b = fftpack.fft(a)
84: 
85:     Zero-padding to the next 5-smooth length reduces computation time to
86:     211 us, a speedup of 630 times:
87: 
88:     >>> fftpack.helper.next_fast_len(min_len)
89:     10125
90:     >>> b = fftpack.fft(a, 10125)
91: 
92:     Rounding up to the next power of 2 is not optimal, taking 367 us to
93:     compute, 1.7 times as long as the 5-smooth size:
94: 
95:     >>> b = fftpack.fft(a, 16384)
96: 
97:     '''
98:     hams = (8, 9, 10, 12, 15, 16, 18, 20, 24, 25, 27, 30, 32, 36, 40, 45, 48,
99:             50, 54, 60, 64, 72, 75, 80, 81, 90, 96, 100, 108, 120, 125, 128,
100:             135, 144, 150, 160, 162, 180, 192, 200, 216, 225, 240, 243, 250,
101:             256, 270, 288, 300, 320, 324, 360, 375, 384, 400, 405, 432, 450,
102:             480, 486, 500, 512, 540, 576, 600, 625, 640, 648, 675, 720, 729,
103:             750, 768, 800, 810, 864, 900, 960, 972, 1000, 1024, 1080, 1125,
104:             1152, 1200, 1215, 1250, 1280, 1296, 1350, 1440, 1458, 1500, 1536,
105:             1600, 1620, 1728, 1800, 1875, 1920, 1944, 2000, 2025, 2048, 2160,
106:             2187, 2250, 2304, 2400, 2430, 2500, 2560, 2592, 2700, 2880, 2916,
107:             3000, 3072, 3125, 3200, 3240, 3375, 3456, 3600, 3645, 3750, 3840,
108:             3888, 4000, 4050, 4096, 4320, 4374, 4500, 4608, 4800, 4860, 5000,
109:             5120, 5184, 5400, 5625, 5760, 5832, 6000, 6075, 6144, 6250, 6400,
110:             6480, 6561, 6750, 6912, 7200, 7290, 7500, 7680, 7776, 8000, 8100,
111:             8192, 8640, 8748, 9000, 9216, 9375, 9600, 9720, 10000)
112: 
113:     if target <= 6:
114:         return target
115: 
116:     # Quickly check if it's already a power of 2
117:     if not (target & (target-1)):
118:         return target
119: 
120:     # Get result quickly for small sizes, since FFT itself is similarly fast.
121:     if target <= hams[-1]:
122:         return hams[bisect_left(hams, target)]
123: 
124:     match = float('inf')  # Anything found will be smaller
125:     p5 = 1
126:     while p5 < target:
127:         p35 = p5
128:         while p35 < target:
129:             # Ceiling integer division, avoiding conversion to float
130:             # (quotient = ceil(target / p35))
131:             quotient = -(-target // p35)
132: 
133:             # Quickly find next power of 2 >= quotient
134:             p2 = 2**((quotient - 1).bit_length())
135: 
136:             N = p2 * p35
137:             if N == target:
138:                 return N
139:             elif N < match:
140:                 match = N
141:             p35 *= 3
142:             if p35 == target:
143:                 return p35
144:         if p35 < match:
145:             match = p35
146:         p5 *= 5
147:         if p5 == target:
148:             return p5
149:     if p5 < match:
150:         match = p5
151:     return match
152: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import operator' statement (line 3)
import operator

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'operator', operator, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from numpy import arange' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/fftpack/')
import_16179 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy')

if (type(import_16179) is not StypyTypeError):

    if (import_16179 != 'pyd_module'):
        __import__(import_16179)
        sys_modules_16180 = sys.modules[import_16179]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy', sys_modules_16180.module_type_store, module_type_store, ['arange'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_16180, sys_modules_16180.module_type_store, module_type_store)
    else:
        from numpy import arange

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy', None, module_type_store, ['arange'], [arange])

else:
    # Assigning a type to the variable 'numpy' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy', import_16179)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/fftpack/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from numpy.fft.helper import fftshift, ifftshift, fftfreq' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/fftpack/')
import_16181 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy.fft.helper')

if (type(import_16181) is not StypyTypeError):

    if (import_16181 != 'pyd_module'):
        __import__(import_16181)
        sys_modules_16182 = sys.modules[import_16181]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy.fft.helper', sys_modules_16182.module_type_store, module_type_store, ['fftshift', 'ifftshift', 'fftfreq'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_16182, sys_modules_16182.module_type_store, module_type_store)
    else:
        from numpy.fft.helper import fftshift, ifftshift, fftfreq

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy.fft.helper', None, module_type_store, ['fftshift', 'ifftshift', 'fftfreq'], [fftshift, ifftshift, fftfreq])

else:
    # Assigning a type to the variable 'numpy.fft.helper' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy.fft.helper', import_16181)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/fftpack/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from bisect import bisect_left' statement (line 6)
try:
    from bisect import bisect_left

except:
    bisect_left = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'bisect', None, module_type_store, ['bisect_left'], [bisect_left])


# Assigning a List to a Name (line 8):
__all__ = ['fftshift', 'ifftshift', 'fftfreq', 'rfftfreq', 'next_fast_len']
module_type_store.set_exportable_members(['fftshift', 'ifftshift', 'fftfreq', 'rfftfreq', 'next_fast_len'])

# Obtaining an instance of the builtin type 'list' (line 8)
list_16183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 8)
# Adding element type (line 8)
str_16184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 11), 'str', 'fftshift')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 10), list_16183, str_16184)
# Adding element type (line 8)
str_16185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 23), 'str', 'ifftshift')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 10), list_16183, str_16185)
# Adding element type (line 8)
str_16186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 36), 'str', 'fftfreq')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 10), list_16183, str_16186)
# Adding element type (line 8)
str_16187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 47), 'str', 'rfftfreq')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 10), list_16183, str_16187)
# Adding element type (line 8)
str_16188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 59), 'str', 'next_fast_len')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 10), list_16183, str_16188)

# Assigning a type to the variable '__all__' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), '__all__', list_16183)

@norecursion
def rfftfreq(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    float_16189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 18), 'float')
    defaults = [float_16189]
    # Create a new context for function 'rfftfreq'
    module_type_store = module_type_store.open_function_context('rfftfreq', 11, 0, False)
    
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

    str_16190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, (-1)), 'str', 'DFT sample frequencies (for usage with rfft, irfft).\n\n    The returned float array contains the frequency bins in\n    cycles/unit (with zero at the start) given a window length `n` and a\n    sample spacing `d`::\n\n      f = [0,1,1,2,2,...,n/2-1,n/2-1,n/2]/(d*n)   if n is even\n      f = [0,1,1,2,2,...,n/2-1,n/2-1,n/2,n/2]/(d*n)   if n is odd\n\n    Parameters\n    ----------\n    n : int\n        Window length.\n    d : scalar, optional\n        Sample spacing. Default is 1.\n\n    Returns\n    -------\n    out : ndarray\n        The array of length `n`, containing the sample frequencies.\n\n    Examples\n    --------\n    >>> from scipy import fftpack\n    >>> sig = np.array([-2, 8, 6, 4, 1, 0, 3, 5], dtype=float)\n    >>> sig_fft = fftpack.rfft(sig)\n    >>> n = sig_fft.size\n    >>> timestep = 0.1\n    >>> freq = fftpack.rfftfreq(n, d=timestep)\n    >>> freq\n    array([ 0.  ,  1.25,  1.25,  2.5 ,  2.5 ,  3.75,  3.75,  5.  ])\n\n    ')
    
    # Assigning a Call to a Name (line 45):
    
    # Call to index(...): (line 45)
    # Processing the call arguments (line 45)
    # Getting the type of 'n' (line 45)
    n_16193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 23), 'n', False)
    # Processing the call keyword arguments (line 45)
    kwargs_16194 = {}
    # Getting the type of 'operator' (line 45)
    operator_16191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'operator', False)
    # Obtaining the member 'index' of a type (line 45)
    index_16192 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 8), operator_16191, 'index')
    # Calling index(args, kwargs) (line 45)
    index_call_result_16195 = invoke(stypy.reporting.localization.Localization(__file__, 45, 8), index_16192, *[n_16193], **kwargs_16194)
    
    # Assigning a type to the variable 'n' (line 45)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'n', index_call_result_16195)
    
    
    # Getting the type of 'n' (line 46)
    n_16196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 7), 'n')
    int_16197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 11), 'int')
    # Applying the binary operator '<' (line 46)
    result_lt_16198 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 7), '<', n_16196, int_16197)
    
    # Testing the type of an if condition (line 46)
    if_condition_16199 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 46, 4), result_lt_16198)
    # Assigning a type to the variable 'if_condition_16199' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'if_condition_16199', if_condition_16199)
    # SSA begins for if statement (line 46)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 47)
    # Processing the call arguments (line 47)
    str_16201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 25), 'str', 'n = %s is not valid. n must be a nonnegative integer.')
    # Getting the type of 'n' (line 48)
    n_16202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 62), 'n', False)
    # Applying the binary operator '%' (line 47)
    result_mod_16203 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 25), '%', str_16201, n_16202)
    
    # Processing the call keyword arguments (line 47)
    kwargs_16204 = {}
    # Getting the type of 'ValueError' (line 47)
    ValueError_16200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 47)
    ValueError_call_result_16205 = invoke(stypy.reporting.localization.Localization(__file__, 47, 14), ValueError_16200, *[result_mod_16203], **kwargs_16204)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 47, 8), ValueError_call_result_16205, 'raise parameter', BaseException)
    # SSA join for if statement (line 46)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to arange(...): (line 50)
    # Processing the call arguments (line 50)
    int_16207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 19), 'int')
    # Getting the type of 'n' (line 50)
    n_16208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 22), 'n', False)
    int_16209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 26), 'int')
    # Applying the binary operator '+' (line 50)
    result_add_16210 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 22), '+', n_16208, int_16209)
    
    # Processing the call keyword arguments (line 50)
    # Getting the type of 'int' (line 50)
    int_16211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 35), 'int', False)
    keyword_16212 = int_16211
    kwargs_16213 = {'dtype': keyword_16212}
    # Getting the type of 'arange' (line 50)
    arange_16206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 12), 'arange', False)
    # Calling arange(args, kwargs) (line 50)
    arange_call_result_16214 = invoke(stypy.reporting.localization.Localization(__file__, 50, 12), arange_16206, *[int_16207, result_add_16210], **kwargs_16213)
    
    int_16215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 43), 'int')
    # Applying the binary operator '//' (line 50)
    result_floordiv_16216 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 12), '//', arange_call_result_16214, int_16215)
    
    
    # Call to float(...): (line 50)
    # Processing the call arguments (line 50)
    # Getting the type of 'n' (line 50)
    n_16218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 54), 'n', False)
    # Getting the type of 'd' (line 50)
    d_16219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 58), 'd', False)
    # Applying the binary operator '*' (line 50)
    result_mul_16220 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 54), '*', n_16218, d_16219)
    
    # Processing the call keyword arguments (line 50)
    kwargs_16221 = {}
    # Getting the type of 'float' (line 50)
    float_16217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 48), 'float', False)
    # Calling float(args, kwargs) (line 50)
    float_call_result_16222 = invoke(stypy.reporting.localization.Localization(__file__, 50, 48), float_16217, *[result_mul_16220], **kwargs_16221)
    
    # Applying the binary operator 'div' (line 50)
    result_div_16223 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 11), 'div', result_floordiv_16216, float_call_result_16222)
    
    # Assigning a type to the variable 'stypy_return_type' (line 50)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'stypy_return_type', result_div_16223)
    
    # ################# End of 'rfftfreq(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'rfftfreq' in the type store
    # Getting the type of 'stypy_return_type' (line 11)
    stypy_return_type_16224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_16224)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'rfftfreq'
    return stypy_return_type_16224

# Assigning a type to the variable 'rfftfreq' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'rfftfreq', rfftfreq)

@norecursion
def next_fast_len(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'next_fast_len'
    module_type_store = module_type_store.open_function_context('next_fast_len', 53, 0, False)
    
    # Passed parameters checking function
    next_fast_len.stypy_localization = localization
    next_fast_len.stypy_type_of_self = None
    next_fast_len.stypy_type_store = module_type_store
    next_fast_len.stypy_function_name = 'next_fast_len'
    next_fast_len.stypy_param_names_list = ['target']
    next_fast_len.stypy_varargs_param_name = None
    next_fast_len.stypy_kwargs_param_name = None
    next_fast_len.stypy_call_defaults = defaults
    next_fast_len.stypy_call_varargs = varargs
    next_fast_len.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'next_fast_len', ['target'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'next_fast_len', localization, ['target'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'next_fast_len(...)' code ##################

    str_16225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, (-1)), 'str', "\n    Find the next fast size of input data to `fft`, for zero-padding, etc.\n\n    SciPy's FFTPACK has efficient functions for radix {2, 3, 4, 5}, so this\n    returns the next composite of the prime factors 2, 3, and 5 which is\n    greater than or equal to `target`. (These are also known as 5-smooth\n    numbers, regular numbers, or Hamming numbers.)\n\n    Parameters\n    ----------\n    target : int\n        Length to start searching from.  Must be a positive integer.\n\n    Returns\n    -------\n    out : int\n        The first 5-smooth number greater than or equal to `target`.\n\n    Notes\n    -----\n    .. versionadded:: 0.18.0\n\n    Examples\n    --------\n    On a particular machine, an FFT of prime length takes 133 ms:\n\n    >>> from scipy import fftpack\n    >>> min_len = 10007  # prime length is worst case for speed\n    >>> a = np.random.randn(min_len)\n    >>> b = fftpack.fft(a)\n\n    Zero-padding to the next 5-smooth length reduces computation time to\n    211 us, a speedup of 630 times:\n\n    >>> fftpack.helper.next_fast_len(min_len)\n    10125\n    >>> b = fftpack.fft(a, 10125)\n\n    Rounding up to the next power of 2 is not optimal, taking 367 us to\n    compute, 1.7 times as long as the 5-smooth size:\n\n    >>> b = fftpack.fft(a, 16384)\n\n    ")
    
    # Assigning a Tuple to a Name (line 98):
    
    # Obtaining an instance of the builtin type 'tuple' (line 98)
    tuple_16226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 12), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 98)
    # Adding element type (line 98)
    int_16227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 12), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16227)
    # Adding element type (line 98)
    int_16228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 15), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16228)
    # Adding element type (line 98)
    int_16229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16229)
    # Adding element type (line 98)
    int_16230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16230)
    # Adding element type (line 98)
    int_16231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 26), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16231)
    # Adding element type (line 98)
    int_16232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16232)
    # Adding element type (line 98)
    int_16233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 34), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16233)
    # Adding element type (line 98)
    int_16234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 38), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16234)
    # Adding element type (line 98)
    int_16235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 42), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16235)
    # Adding element type (line 98)
    int_16236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 46), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16236)
    # Adding element type (line 98)
    int_16237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 50), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16237)
    # Adding element type (line 98)
    int_16238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 54), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16238)
    # Adding element type (line 98)
    int_16239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 58), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16239)
    # Adding element type (line 98)
    int_16240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 62), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16240)
    # Adding element type (line 98)
    int_16241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 66), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16241)
    # Adding element type (line 98)
    int_16242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 70), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16242)
    # Adding element type (line 98)
    int_16243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 74), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16243)
    # Adding element type (line 98)
    int_16244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 12), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16244)
    # Adding element type (line 98)
    int_16245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 16), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16245)
    # Adding element type (line 98)
    int_16246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16246)
    # Adding element type (line 98)
    int_16247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16247)
    # Adding element type (line 98)
    int_16248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16248)
    # Adding element type (line 98)
    int_16249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 32), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16249)
    # Adding element type (line 98)
    int_16250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 36), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16250)
    # Adding element type (line 98)
    int_16251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 40), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16251)
    # Adding element type (line 98)
    int_16252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 44), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16252)
    # Adding element type (line 98)
    int_16253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 48), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16253)
    # Adding element type (line 98)
    int_16254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 52), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16254)
    # Adding element type (line 98)
    int_16255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 57), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16255)
    # Adding element type (line 98)
    int_16256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 62), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16256)
    # Adding element type (line 98)
    int_16257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 67), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16257)
    # Adding element type (line 98)
    int_16258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 72), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16258)
    # Adding element type (line 98)
    int_16259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 12), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16259)
    # Adding element type (line 98)
    int_16260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 17), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16260)
    # Adding element type (line 98)
    int_16261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16261)
    # Adding element type (line 98)
    int_16262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16262)
    # Adding element type (line 98)
    int_16263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 32), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16263)
    # Adding element type (line 98)
    int_16264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 37), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16264)
    # Adding element type (line 98)
    int_16265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 42), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16265)
    # Adding element type (line 98)
    int_16266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 47), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16266)
    # Adding element type (line 98)
    int_16267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 52), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16267)
    # Adding element type (line 98)
    int_16268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 57), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16268)
    # Adding element type (line 98)
    int_16269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 62), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16269)
    # Adding element type (line 98)
    int_16270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 67), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16270)
    # Adding element type (line 98)
    int_16271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 72), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16271)
    # Adding element type (line 98)
    int_16272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 12), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16272)
    # Adding element type (line 98)
    int_16273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 17), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16273)
    # Adding element type (line 98)
    int_16274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16274)
    # Adding element type (line 98)
    int_16275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16275)
    # Adding element type (line 98)
    int_16276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 32), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16276)
    # Adding element type (line 98)
    int_16277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 37), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16277)
    # Adding element type (line 98)
    int_16278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 42), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16278)
    # Adding element type (line 98)
    int_16279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 47), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16279)
    # Adding element type (line 98)
    int_16280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 52), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16280)
    # Adding element type (line 98)
    int_16281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 57), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16281)
    # Adding element type (line 98)
    int_16282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 62), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16282)
    # Adding element type (line 98)
    int_16283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 67), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16283)
    # Adding element type (line 98)
    int_16284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 72), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16284)
    # Adding element type (line 98)
    int_16285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 12), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16285)
    # Adding element type (line 98)
    int_16286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 17), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16286)
    # Adding element type (line 98)
    int_16287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16287)
    # Adding element type (line 98)
    int_16288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16288)
    # Adding element type (line 98)
    int_16289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 32), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16289)
    # Adding element type (line 98)
    int_16290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 37), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16290)
    # Adding element type (line 98)
    int_16291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 42), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16291)
    # Adding element type (line 98)
    int_16292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 47), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16292)
    # Adding element type (line 98)
    int_16293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 52), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16293)
    # Adding element type (line 98)
    int_16294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 57), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16294)
    # Adding element type (line 98)
    int_16295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 62), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16295)
    # Adding element type (line 98)
    int_16296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 67), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16296)
    # Adding element type (line 98)
    int_16297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 72), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16297)
    # Adding element type (line 98)
    int_16298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 12), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16298)
    # Adding element type (line 98)
    int_16299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 17), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16299)
    # Adding element type (line 98)
    int_16300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16300)
    # Adding element type (line 98)
    int_16301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16301)
    # Adding element type (line 98)
    int_16302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 32), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16302)
    # Adding element type (line 98)
    int_16303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 37), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16303)
    # Adding element type (line 98)
    int_16304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 42), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16304)
    # Adding element type (line 98)
    int_16305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 47), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16305)
    # Adding element type (line 98)
    int_16306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 52), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16306)
    # Adding element type (line 98)
    int_16307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 58), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16307)
    # Adding element type (line 98)
    int_16308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 64), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16308)
    # Adding element type (line 98)
    int_16309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 70), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16309)
    # Adding element type (line 98)
    int_16310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 12), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16310)
    # Adding element type (line 98)
    int_16311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16311)
    # Adding element type (line 98)
    int_16312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16312)
    # Adding element type (line 98)
    int_16313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16313)
    # Adding element type (line 98)
    int_16314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 36), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16314)
    # Adding element type (line 98)
    int_16315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 42), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16315)
    # Adding element type (line 98)
    int_16316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 48), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16316)
    # Adding element type (line 98)
    int_16317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 54), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16317)
    # Adding element type (line 98)
    int_16318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 60), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16318)
    # Adding element type (line 98)
    int_16319 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 66), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16319)
    # Adding element type (line 98)
    int_16320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 72), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16320)
    # Adding element type (line 98)
    int_16321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 12), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16321)
    # Adding element type (line 98)
    int_16322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16322)
    # Adding element type (line 98)
    int_16323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16323)
    # Adding element type (line 98)
    int_16324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16324)
    # Adding element type (line 98)
    int_16325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 36), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16325)
    # Adding element type (line 98)
    int_16326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 42), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16326)
    # Adding element type (line 98)
    int_16327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 48), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16327)
    # Adding element type (line 98)
    int_16328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 54), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16328)
    # Adding element type (line 98)
    int_16329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 60), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16329)
    # Adding element type (line 98)
    int_16330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 66), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16330)
    # Adding element type (line 98)
    int_16331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 72), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16331)
    # Adding element type (line 98)
    int_16332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 12), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16332)
    # Adding element type (line 98)
    int_16333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16333)
    # Adding element type (line 98)
    int_16334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16334)
    # Adding element type (line 98)
    int_16335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16335)
    # Adding element type (line 98)
    int_16336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 36), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16336)
    # Adding element type (line 98)
    int_16337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 42), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16337)
    # Adding element type (line 98)
    int_16338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 48), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16338)
    # Adding element type (line 98)
    int_16339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 54), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16339)
    # Adding element type (line 98)
    int_16340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 60), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16340)
    # Adding element type (line 98)
    int_16341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 66), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16341)
    # Adding element type (line 98)
    int_16342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 72), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16342)
    # Adding element type (line 98)
    int_16343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 12), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16343)
    # Adding element type (line 98)
    int_16344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16344)
    # Adding element type (line 98)
    int_16345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16345)
    # Adding element type (line 98)
    int_16346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16346)
    # Adding element type (line 98)
    int_16347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 36), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16347)
    # Adding element type (line 98)
    int_16348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 42), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16348)
    # Adding element type (line 98)
    int_16349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 48), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16349)
    # Adding element type (line 98)
    int_16350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 54), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16350)
    # Adding element type (line 98)
    int_16351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 60), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16351)
    # Adding element type (line 98)
    int_16352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 66), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16352)
    # Adding element type (line 98)
    int_16353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 72), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16353)
    # Adding element type (line 98)
    int_16354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 12), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16354)
    # Adding element type (line 98)
    int_16355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16355)
    # Adding element type (line 98)
    int_16356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16356)
    # Adding element type (line 98)
    int_16357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16357)
    # Adding element type (line 98)
    int_16358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 36), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16358)
    # Adding element type (line 98)
    int_16359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 42), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16359)
    # Adding element type (line 98)
    int_16360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 48), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16360)
    # Adding element type (line 98)
    int_16361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 54), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16361)
    # Adding element type (line 98)
    int_16362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 60), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16362)
    # Adding element type (line 98)
    int_16363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 66), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16363)
    # Adding element type (line 98)
    int_16364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 72), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16364)
    # Adding element type (line 98)
    int_16365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 12), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16365)
    # Adding element type (line 98)
    int_16366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16366)
    # Adding element type (line 98)
    int_16367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16367)
    # Adding element type (line 98)
    int_16368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16368)
    # Adding element type (line 98)
    int_16369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 36), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16369)
    # Adding element type (line 98)
    int_16370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 42), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16370)
    # Adding element type (line 98)
    int_16371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 48), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16371)
    # Adding element type (line 98)
    int_16372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 54), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16372)
    # Adding element type (line 98)
    int_16373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 60), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16373)
    # Adding element type (line 98)
    int_16374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 66), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16374)
    # Adding element type (line 98)
    int_16375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 72), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16375)
    # Adding element type (line 98)
    int_16376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 12), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16376)
    # Adding element type (line 98)
    int_16377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16377)
    # Adding element type (line 98)
    int_16378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16378)
    # Adding element type (line 98)
    int_16379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16379)
    # Adding element type (line 98)
    int_16380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 36), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16380)
    # Adding element type (line 98)
    int_16381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 42), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16381)
    # Adding element type (line 98)
    int_16382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 48), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16382)
    # Adding element type (line 98)
    int_16383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 54), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16383)
    # Adding element type (line 98)
    int_16384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 60), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16384)
    # Adding element type (line 98)
    int_16385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 66), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16385)
    # Adding element type (line 98)
    int_16386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 72), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16386)
    # Adding element type (line 98)
    int_16387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 12), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16387)
    # Adding element type (line 98)
    int_16388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16388)
    # Adding element type (line 98)
    int_16389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16389)
    # Adding element type (line 98)
    int_16390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16390)
    # Adding element type (line 98)
    int_16391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 36), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16391)
    # Adding element type (line 98)
    int_16392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 42), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16392)
    # Adding element type (line 98)
    int_16393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 48), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16393)
    # Adding element type (line 98)
    int_16394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 54), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16394)
    # Adding element type (line 98)
    int_16395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 60), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), tuple_16226, int_16395)
    
    # Assigning a type to the variable 'hams' (line 98)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'hams', tuple_16226)
    
    
    # Getting the type of 'target' (line 113)
    target_16396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 7), 'target')
    int_16397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 17), 'int')
    # Applying the binary operator '<=' (line 113)
    result_le_16398 = python_operator(stypy.reporting.localization.Localization(__file__, 113, 7), '<=', target_16396, int_16397)
    
    # Testing the type of an if condition (line 113)
    if_condition_16399 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 113, 4), result_le_16398)
    # Assigning a type to the variable 'if_condition_16399' (line 113)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 4), 'if_condition_16399', if_condition_16399)
    # SSA begins for if statement (line 113)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'target' (line 114)
    target_16400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 15), 'target')
    # Assigning a type to the variable 'stypy_return_type' (line 114)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'stypy_return_type', target_16400)
    # SSA join for if statement (line 113)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'target' (line 117)
    target_16401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 12), 'target')
    # Getting the type of 'target' (line 117)
    target_16402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 22), 'target')
    int_16403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 29), 'int')
    # Applying the binary operator '-' (line 117)
    result_sub_16404 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 22), '-', target_16402, int_16403)
    
    # Applying the binary operator '&' (line 117)
    result_and__16405 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 12), '&', target_16401, result_sub_16404)
    
    # Applying the 'not' unary operator (line 117)
    result_not__16406 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 7), 'not', result_and__16405)
    
    # Testing the type of an if condition (line 117)
    if_condition_16407 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 117, 4), result_not__16406)
    # Assigning a type to the variable 'if_condition_16407' (line 117)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 4), 'if_condition_16407', if_condition_16407)
    # SSA begins for if statement (line 117)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'target' (line 118)
    target_16408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 15), 'target')
    # Assigning a type to the variable 'stypy_return_type' (line 118)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'stypy_return_type', target_16408)
    # SSA join for if statement (line 117)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'target' (line 121)
    target_16409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 7), 'target')
    
    # Obtaining the type of the subscript
    int_16410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 22), 'int')
    # Getting the type of 'hams' (line 121)
    hams_16411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 17), 'hams')
    # Obtaining the member '__getitem__' of a type (line 121)
    getitem___16412 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 17), hams_16411, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 121)
    subscript_call_result_16413 = invoke(stypy.reporting.localization.Localization(__file__, 121, 17), getitem___16412, int_16410)
    
    # Applying the binary operator '<=' (line 121)
    result_le_16414 = python_operator(stypy.reporting.localization.Localization(__file__, 121, 7), '<=', target_16409, subscript_call_result_16413)
    
    # Testing the type of an if condition (line 121)
    if_condition_16415 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 121, 4), result_le_16414)
    # Assigning a type to the variable 'if_condition_16415' (line 121)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 4), 'if_condition_16415', if_condition_16415)
    # SSA begins for if statement (line 121)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining the type of the subscript
    
    # Call to bisect_left(...): (line 122)
    # Processing the call arguments (line 122)
    # Getting the type of 'hams' (line 122)
    hams_16417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 32), 'hams', False)
    # Getting the type of 'target' (line 122)
    target_16418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 38), 'target', False)
    # Processing the call keyword arguments (line 122)
    kwargs_16419 = {}
    # Getting the type of 'bisect_left' (line 122)
    bisect_left_16416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 20), 'bisect_left', False)
    # Calling bisect_left(args, kwargs) (line 122)
    bisect_left_call_result_16420 = invoke(stypy.reporting.localization.Localization(__file__, 122, 20), bisect_left_16416, *[hams_16417, target_16418], **kwargs_16419)
    
    # Getting the type of 'hams' (line 122)
    hams_16421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 15), 'hams')
    # Obtaining the member '__getitem__' of a type (line 122)
    getitem___16422 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 15), hams_16421, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 122)
    subscript_call_result_16423 = invoke(stypy.reporting.localization.Localization(__file__, 122, 15), getitem___16422, bisect_left_call_result_16420)
    
    # Assigning a type to the variable 'stypy_return_type' (line 122)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'stypy_return_type', subscript_call_result_16423)
    # SSA join for if statement (line 121)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 124):
    
    # Call to float(...): (line 124)
    # Processing the call arguments (line 124)
    str_16425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 18), 'str', 'inf')
    # Processing the call keyword arguments (line 124)
    kwargs_16426 = {}
    # Getting the type of 'float' (line 124)
    float_16424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 12), 'float', False)
    # Calling float(args, kwargs) (line 124)
    float_call_result_16427 = invoke(stypy.reporting.localization.Localization(__file__, 124, 12), float_16424, *[str_16425], **kwargs_16426)
    
    # Assigning a type to the variable 'match' (line 124)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 4), 'match', float_call_result_16427)
    
    # Assigning a Num to a Name (line 125):
    int_16428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 9), 'int')
    # Assigning a type to the variable 'p5' (line 125)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 4), 'p5', int_16428)
    
    
    # Getting the type of 'p5' (line 126)
    p5_16429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 10), 'p5')
    # Getting the type of 'target' (line 126)
    target_16430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 15), 'target')
    # Applying the binary operator '<' (line 126)
    result_lt_16431 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 10), '<', p5_16429, target_16430)
    
    # Testing the type of an if condition (line 126)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 126, 4), result_lt_16431)
    # SSA begins for while statement (line 126)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Assigning a Name to a Name (line 127):
    # Getting the type of 'p5' (line 127)
    p5_16432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 14), 'p5')
    # Assigning a type to the variable 'p35' (line 127)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'p35', p5_16432)
    
    
    # Getting the type of 'p35' (line 128)
    p35_16433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 14), 'p35')
    # Getting the type of 'target' (line 128)
    target_16434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 20), 'target')
    # Applying the binary operator '<' (line 128)
    result_lt_16435 = python_operator(stypy.reporting.localization.Localization(__file__, 128, 14), '<', p35_16433, target_16434)
    
    # Testing the type of an if condition (line 128)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 128, 8), result_lt_16435)
    # SSA begins for while statement (line 128)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Assigning a UnaryOp to a Name (line 131):
    
    
    # Getting the type of 'target' (line 131)
    target_16436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 26), 'target')
    # Applying the 'usub' unary operator (line 131)
    result___neg___16437 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 25), 'usub', target_16436)
    
    # Getting the type of 'p35' (line 131)
    p35_16438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 36), 'p35')
    # Applying the binary operator '//' (line 131)
    result_floordiv_16439 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 25), '//', result___neg___16437, p35_16438)
    
    # Applying the 'usub' unary operator (line 131)
    result___neg___16440 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 23), 'usub', result_floordiv_16439)
    
    # Assigning a type to the variable 'quotient' (line 131)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 12), 'quotient', result___neg___16440)
    
    # Assigning a BinOp to a Name (line 134):
    int_16441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 17), 'int')
    
    # Call to bit_length(...): (line 134)
    # Processing the call keyword arguments (line 134)
    kwargs_16446 = {}
    # Getting the type of 'quotient' (line 134)
    quotient_16442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 22), 'quotient', False)
    int_16443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 33), 'int')
    # Applying the binary operator '-' (line 134)
    result_sub_16444 = python_operator(stypy.reporting.localization.Localization(__file__, 134, 22), '-', quotient_16442, int_16443)
    
    # Obtaining the member 'bit_length' of a type (line 134)
    bit_length_16445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 22), result_sub_16444, 'bit_length')
    # Calling bit_length(args, kwargs) (line 134)
    bit_length_call_result_16447 = invoke(stypy.reporting.localization.Localization(__file__, 134, 22), bit_length_16445, *[], **kwargs_16446)
    
    # Applying the binary operator '**' (line 134)
    result_pow_16448 = python_operator(stypy.reporting.localization.Localization(__file__, 134, 17), '**', int_16441, bit_length_call_result_16447)
    
    # Assigning a type to the variable 'p2' (line 134)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 12), 'p2', result_pow_16448)
    
    # Assigning a BinOp to a Name (line 136):
    # Getting the type of 'p2' (line 136)
    p2_16449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 16), 'p2')
    # Getting the type of 'p35' (line 136)
    p35_16450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 21), 'p35')
    # Applying the binary operator '*' (line 136)
    result_mul_16451 = python_operator(stypy.reporting.localization.Localization(__file__, 136, 16), '*', p2_16449, p35_16450)
    
    # Assigning a type to the variable 'N' (line 136)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 12), 'N', result_mul_16451)
    
    
    # Getting the type of 'N' (line 137)
    N_16452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 15), 'N')
    # Getting the type of 'target' (line 137)
    target_16453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 20), 'target')
    # Applying the binary operator '==' (line 137)
    result_eq_16454 = python_operator(stypy.reporting.localization.Localization(__file__, 137, 15), '==', N_16452, target_16453)
    
    # Testing the type of an if condition (line 137)
    if_condition_16455 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 137, 12), result_eq_16454)
    # Assigning a type to the variable 'if_condition_16455' (line 137)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 12), 'if_condition_16455', if_condition_16455)
    # SSA begins for if statement (line 137)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'N' (line 138)
    N_16456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 23), 'N')
    # Assigning a type to the variable 'stypy_return_type' (line 138)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 16), 'stypy_return_type', N_16456)
    # SSA branch for the else part of an if statement (line 137)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'N' (line 139)
    N_16457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 17), 'N')
    # Getting the type of 'match' (line 139)
    match_16458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 21), 'match')
    # Applying the binary operator '<' (line 139)
    result_lt_16459 = python_operator(stypy.reporting.localization.Localization(__file__, 139, 17), '<', N_16457, match_16458)
    
    # Testing the type of an if condition (line 139)
    if_condition_16460 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 139, 17), result_lt_16459)
    # Assigning a type to the variable 'if_condition_16460' (line 139)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 17), 'if_condition_16460', if_condition_16460)
    # SSA begins for if statement (line 139)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 140):
    # Getting the type of 'N' (line 140)
    N_16461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 24), 'N')
    # Assigning a type to the variable 'match' (line 140)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 16), 'match', N_16461)
    # SSA join for if statement (line 139)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 137)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'p35' (line 141)
    p35_16462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 12), 'p35')
    int_16463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 19), 'int')
    # Applying the binary operator '*=' (line 141)
    result_imul_16464 = python_operator(stypy.reporting.localization.Localization(__file__, 141, 12), '*=', p35_16462, int_16463)
    # Assigning a type to the variable 'p35' (line 141)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 12), 'p35', result_imul_16464)
    
    
    
    # Getting the type of 'p35' (line 142)
    p35_16465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 15), 'p35')
    # Getting the type of 'target' (line 142)
    target_16466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 22), 'target')
    # Applying the binary operator '==' (line 142)
    result_eq_16467 = python_operator(stypy.reporting.localization.Localization(__file__, 142, 15), '==', p35_16465, target_16466)
    
    # Testing the type of an if condition (line 142)
    if_condition_16468 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 142, 12), result_eq_16467)
    # Assigning a type to the variable 'if_condition_16468' (line 142)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 12), 'if_condition_16468', if_condition_16468)
    # SSA begins for if statement (line 142)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'p35' (line 143)
    p35_16469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 23), 'p35')
    # Assigning a type to the variable 'stypy_return_type' (line 143)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 16), 'stypy_return_type', p35_16469)
    # SSA join for if statement (line 142)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for while statement (line 128)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'p35' (line 144)
    p35_16470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 11), 'p35')
    # Getting the type of 'match' (line 144)
    match_16471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 17), 'match')
    # Applying the binary operator '<' (line 144)
    result_lt_16472 = python_operator(stypy.reporting.localization.Localization(__file__, 144, 11), '<', p35_16470, match_16471)
    
    # Testing the type of an if condition (line 144)
    if_condition_16473 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 144, 8), result_lt_16472)
    # Assigning a type to the variable 'if_condition_16473' (line 144)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'if_condition_16473', if_condition_16473)
    # SSA begins for if statement (line 144)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 145):
    # Getting the type of 'p35' (line 145)
    p35_16474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 20), 'p35')
    # Assigning a type to the variable 'match' (line 145)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 12), 'match', p35_16474)
    # SSA join for if statement (line 144)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'p5' (line 146)
    p5_16475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'p5')
    int_16476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 14), 'int')
    # Applying the binary operator '*=' (line 146)
    result_imul_16477 = python_operator(stypy.reporting.localization.Localization(__file__, 146, 8), '*=', p5_16475, int_16476)
    # Assigning a type to the variable 'p5' (line 146)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'p5', result_imul_16477)
    
    
    
    # Getting the type of 'p5' (line 147)
    p5_16478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 11), 'p5')
    # Getting the type of 'target' (line 147)
    target_16479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 17), 'target')
    # Applying the binary operator '==' (line 147)
    result_eq_16480 = python_operator(stypy.reporting.localization.Localization(__file__, 147, 11), '==', p5_16478, target_16479)
    
    # Testing the type of an if condition (line 147)
    if_condition_16481 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 147, 8), result_eq_16480)
    # Assigning a type to the variable 'if_condition_16481' (line 147)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'if_condition_16481', if_condition_16481)
    # SSA begins for if statement (line 147)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'p5' (line 148)
    p5_16482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 19), 'p5')
    # Assigning a type to the variable 'stypy_return_type' (line 148)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 12), 'stypy_return_type', p5_16482)
    # SSA join for if statement (line 147)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for while statement (line 126)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'p5' (line 149)
    p5_16483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 7), 'p5')
    # Getting the type of 'match' (line 149)
    match_16484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 12), 'match')
    # Applying the binary operator '<' (line 149)
    result_lt_16485 = python_operator(stypy.reporting.localization.Localization(__file__, 149, 7), '<', p5_16483, match_16484)
    
    # Testing the type of an if condition (line 149)
    if_condition_16486 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 149, 4), result_lt_16485)
    # Assigning a type to the variable 'if_condition_16486' (line 149)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 4), 'if_condition_16486', if_condition_16486)
    # SSA begins for if statement (line 149)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 150):
    # Getting the type of 'p5' (line 150)
    p5_16487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 16), 'p5')
    # Assigning a type to the variable 'match' (line 150)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 8), 'match', p5_16487)
    # SSA join for if statement (line 149)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'match' (line 151)
    match_16488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 11), 'match')
    # Assigning a type to the variable 'stypy_return_type' (line 151)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 4), 'stypy_return_type', match_16488)
    
    # ################# End of 'next_fast_len(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'next_fast_len' in the type store
    # Getting the type of 'stypy_return_type' (line 53)
    stypy_return_type_16489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_16489)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'next_fast_len'
    return stypy_return_type_16489

# Assigning a type to the variable 'next_fast_len' (line 53)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 0), 'next_fast_len', next_fast_len)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
