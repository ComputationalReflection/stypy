
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # Code adapted from "upfirdn" python library with permission:
2: #
3: # Copyright (c) 2009, Motorola, Inc
4: #
5: # All Rights Reserved.
6: #
7: # Redistribution and use in source and binary forms, with or without
8: # modification, are permitted provided that the following conditions are
9: # met:
10: #
11: # * Redistributions of source code must retain the above copyright notice,
12: # this list of conditions and the following disclaimer.
13: #
14: # * Redistributions in binary form must reproduce the above copyright
15: # notice, this list of conditions and the following disclaimer in the
16: # documentation and/or other materials provided with the distribution.
17: #
18: # * Neither the name of Motorola nor the names of its contributors may be
19: # used to endorse or promote products derived from this software without
20: # specific prior written permission.
21: #
22: # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
23: # IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
24: # THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
25: # PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
26: # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
27: # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
28: # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
29: # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
30: # LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
31: # NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
32: # SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
33: 
34: 
35: import numpy as np
36: from itertools import product
37: 
38: from numpy.testing import assert_equal, assert_allclose
39: from pytest import raises as assert_raises
40: 
41: from scipy.signal import upfirdn, firwin, lfilter
42: from scipy.signal._upfirdn import _output_len
43: 
44: 
45: def upfirdn_naive(x, h, up=1, down=1):
46:     '''Naive upfirdn processing in Python
47: 
48:     Note: arg order (x, h) differs to facilitate apply_along_axis use.
49:     '''
50:     h = np.asarray(h)
51:     out = np.zeros(len(x) * up, x.dtype)
52:     out[::up] = x
53:     out = np.convolve(h, out)[::down][:_output_len(len(h), len(x), up, down)]
54:     return out
55: 
56: 
57: class UpFIRDnCase(object):
58:     '''Test _UpFIRDn object'''
59:     def __init__(self, up, down, h, x_dtype):
60:         self.up = up
61:         self.down = down
62:         self.h = np.atleast_1d(h)
63:         self.x_dtype = x_dtype
64:         self.rng = np.random.RandomState(17)
65: 
66:     def __call__(self):
67:         # tiny signal
68:         self.scrub(np.ones(1, self.x_dtype))
69:         # ones
70:         self.scrub(np.ones(10, self.x_dtype))  # ones
71:         # randn
72:         x = self.rng.randn(10).astype(self.x_dtype)
73:         if self.x_dtype in (np.complex64, np.complex128):
74:             x += 1j * self.rng.randn(10)
75:         self.scrub(x)
76:         # ramp
77:         self.scrub(np.arange(10).astype(self.x_dtype))
78:         # 3D, random
79:         size = (2, 3, 5)
80:         x = self.rng.randn(*size).astype(self.x_dtype)
81:         if self.x_dtype in (np.complex64, np.complex128):
82:             x += 1j * self.rng.randn(*size)
83:         for axis in range(len(size)):
84:             self.scrub(x, axis=axis)
85:         x = x[:, ::2, 1::3].T
86:         for axis in range(len(size)):
87:             self.scrub(x, axis=axis)
88: 
89:     def scrub(self, x, axis=-1):
90:         yr = np.apply_along_axis(upfirdn_naive, axis, x,
91:                                  self.h, self.up, self.down)
92:         y = upfirdn(self.h, x, self.up, self.down, axis=axis)
93:         dtypes = (self.h.dtype, x.dtype)
94:         if all(d == np.complex64 for d in dtypes):
95:             assert_equal(y.dtype, np.complex64)
96:         elif np.complex64 in dtypes and np.float32 in dtypes:
97:             assert_equal(y.dtype, np.complex64)
98:         elif all(d == np.float32 for d in dtypes):
99:             assert_equal(y.dtype, np.float32)
100:         elif np.complex128 in dtypes or np.complex64 in dtypes:
101:             assert_equal(y.dtype, np.complex128)
102:         else:
103:             assert_equal(y.dtype, np.float64)
104:         assert_allclose(yr, y)
105: 
106: 
107: class TestUpfirdn(object):
108: 
109:     def test_valid_input(self):
110:         assert_raises(ValueError, upfirdn, [1], [1], 1, 0)  # up or down < 1
111:         assert_raises(ValueError, upfirdn, [], [1], 1, 1)  # h.ndim != 1
112:         assert_raises(ValueError, upfirdn, [[1]], [1], 1, 1)
113: 
114:     def test_vs_lfilter(self):
115:         # Check that up=1.0 gives same answer as lfilter + slicing
116:         random_state = np.random.RandomState(17)
117:         try_types = (int, np.float32, np.complex64, float, complex)
118:         size = 10000
119:         down_factors = [2, 11, 79]
120: 
121:         for dtype in try_types:
122:             x = random_state.randn(size).astype(dtype)
123:             if dtype in (np.complex64, np.complex128):
124:                 x += 1j * random_state.randn(size)
125: 
126:             for down in down_factors:
127:                 h = firwin(31, 1. / down, window='hamming')
128:                 yl = lfilter(h, 1.0, x)[::down]
129:                 y = upfirdn(h, x, up=1, down=down)
130:                 assert_allclose(yl, y[:yl.size], atol=1e-7, rtol=1e-7)
131: 
132:     def test_vs_naive(self):
133:         tests = []
134:         try_types = (int, np.float32, np.complex64, float, complex)
135: 
136:         # Simple combinations of factors
137:         for x_dtype, h in product(try_types, (1., 1j)):
138:                 tests.append(UpFIRDnCase(1, 1, h, x_dtype))
139:                 tests.append(UpFIRDnCase(2, 2, h, x_dtype))
140:                 tests.append(UpFIRDnCase(3, 2, h, x_dtype))
141:                 tests.append(UpFIRDnCase(2, 3, h, x_dtype))
142: 
143:         # mixture of big, small, and both directions (net up and net down)
144:         # use all combinations of data and filter dtypes
145:         factors = (100, 10)  # up/down factors
146:         cases = product(factors, factors, try_types, try_types)
147:         for case in cases:
148:             tests += self._random_factors(*case)
149: 
150:         for test in tests:
151:             test()
152: 
153:     def _random_factors(self, p_max, q_max, h_dtype, x_dtype):
154:         n_rep = 3
155:         longest_h = 25
156:         random_state = np.random.RandomState(17)
157:         tests = []
158: 
159:         for _ in range(n_rep):
160:             # Randomize the up/down factors somewhat
161:             p_add = q_max if p_max > q_max else 1
162:             q_add = p_max if q_max > p_max else 1
163:             p = random_state.randint(p_max) + p_add
164:             q = random_state.randint(q_max) + q_add
165: 
166:             # Generate random FIR coefficients
167:             len_h = random_state.randint(longest_h) + 1
168:             h = np.atleast_1d(random_state.randint(len_h))
169:             h = h.astype(h_dtype)
170:             if h_dtype == complex:
171:                 h += 1j * random_state.randint(len_h)
172: 
173:             tests.append(UpFIRDnCase(p, q, h, x_dtype))
174: 
175:         return tests
176: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 35, 0))

# 'import numpy' statement (line 35)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/tests/')
import_350772 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 35, 0), 'numpy')

if (type(import_350772) is not StypyTypeError):

    if (import_350772 != 'pyd_module'):
        __import__(import_350772)
        sys_modules_350773 = sys.modules[import_350772]
        import_module(stypy.reporting.localization.Localization(__file__, 35, 0), 'np', sys_modules_350773.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 35, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 0), 'numpy', import_350772)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 36, 0))

# 'from itertools import product' statement (line 36)
try:
    from itertools import product

except:
    product = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 36, 0), 'itertools', None, module_type_store, ['product'], [product])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 38, 0))

# 'from numpy.testing import assert_equal, assert_allclose' statement (line 38)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/tests/')
import_350774 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 38, 0), 'numpy.testing')

if (type(import_350774) is not StypyTypeError):

    if (import_350774 != 'pyd_module'):
        __import__(import_350774)
        sys_modules_350775 = sys.modules[import_350774]
        import_from_module(stypy.reporting.localization.Localization(__file__, 38, 0), 'numpy.testing', sys_modules_350775.module_type_store, module_type_store, ['assert_equal', 'assert_allclose'])
        nest_module(stypy.reporting.localization.Localization(__file__, 38, 0), __file__, sys_modules_350775, sys_modules_350775.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_equal, assert_allclose

        import_from_module(stypy.reporting.localization.Localization(__file__, 38, 0), 'numpy.testing', None, module_type_store, ['assert_equal', 'assert_allclose'], [assert_equal, assert_allclose])

else:
    # Assigning a type to the variable 'numpy.testing' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 0), 'numpy.testing', import_350774)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 39, 0))

# 'from pytest import assert_raises' statement (line 39)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/tests/')
import_350776 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 39, 0), 'pytest')

if (type(import_350776) is not StypyTypeError):

    if (import_350776 != 'pyd_module'):
        __import__(import_350776)
        sys_modules_350777 = sys.modules[import_350776]
        import_from_module(stypy.reporting.localization.Localization(__file__, 39, 0), 'pytest', sys_modules_350777.module_type_store, module_type_store, ['raises'])
        nest_module(stypy.reporting.localization.Localization(__file__, 39, 0), __file__, sys_modules_350777, sys_modules_350777.module_type_store, module_type_store)
    else:
        from pytest import raises as assert_raises

        import_from_module(stypy.reporting.localization.Localization(__file__, 39, 0), 'pytest', None, module_type_store, ['raises'], [assert_raises])

else:
    # Assigning a type to the variable 'pytest' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 0), 'pytest', import_350776)

# Adding an alias
module_type_store.add_alias('assert_raises', 'raises')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 41, 0))

# 'from scipy.signal import upfirdn, firwin, lfilter' statement (line 41)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/tests/')
import_350778 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 41, 0), 'scipy.signal')

if (type(import_350778) is not StypyTypeError):

    if (import_350778 != 'pyd_module'):
        __import__(import_350778)
        sys_modules_350779 = sys.modules[import_350778]
        import_from_module(stypy.reporting.localization.Localization(__file__, 41, 0), 'scipy.signal', sys_modules_350779.module_type_store, module_type_store, ['upfirdn', 'firwin', 'lfilter'])
        nest_module(stypy.reporting.localization.Localization(__file__, 41, 0), __file__, sys_modules_350779, sys_modules_350779.module_type_store, module_type_store)
    else:
        from scipy.signal import upfirdn, firwin, lfilter

        import_from_module(stypy.reporting.localization.Localization(__file__, 41, 0), 'scipy.signal', None, module_type_store, ['upfirdn', 'firwin', 'lfilter'], [upfirdn, firwin, lfilter])

else:
    # Assigning a type to the variable 'scipy.signal' (line 41)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 0), 'scipy.signal', import_350778)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 42, 0))

# 'from scipy.signal._upfirdn import _output_len' statement (line 42)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/tests/')
import_350780 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 42, 0), 'scipy.signal._upfirdn')

if (type(import_350780) is not StypyTypeError):

    if (import_350780 != 'pyd_module'):
        __import__(import_350780)
        sys_modules_350781 = sys.modules[import_350780]
        import_from_module(stypy.reporting.localization.Localization(__file__, 42, 0), 'scipy.signal._upfirdn', sys_modules_350781.module_type_store, module_type_store, ['_output_len'])
        nest_module(stypy.reporting.localization.Localization(__file__, 42, 0), __file__, sys_modules_350781, sys_modules_350781.module_type_store, module_type_store)
    else:
        from scipy.signal._upfirdn import _output_len

        import_from_module(stypy.reporting.localization.Localization(__file__, 42, 0), 'scipy.signal._upfirdn', None, module_type_store, ['_output_len'], [_output_len])

else:
    # Assigning a type to the variable 'scipy.signal._upfirdn' (line 42)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 0), 'scipy.signal._upfirdn', import_350780)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/tests/')


@norecursion
def upfirdn_naive(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_350782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 27), 'int')
    int_350783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 35), 'int')
    defaults = [int_350782, int_350783]
    # Create a new context for function 'upfirdn_naive'
    module_type_store = module_type_store.open_function_context('upfirdn_naive', 45, 0, False)
    
    # Passed parameters checking function
    upfirdn_naive.stypy_localization = localization
    upfirdn_naive.stypy_type_of_self = None
    upfirdn_naive.stypy_type_store = module_type_store
    upfirdn_naive.stypy_function_name = 'upfirdn_naive'
    upfirdn_naive.stypy_param_names_list = ['x', 'h', 'up', 'down']
    upfirdn_naive.stypy_varargs_param_name = None
    upfirdn_naive.stypy_kwargs_param_name = None
    upfirdn_naive.stypy_call_defaults = defaults
    upfirdn_naive.stypy_call_varargs = varargs
    upfirdn_naive.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'upfirdn_naive', ['x', 'h', 'up', 'down'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'upfirdn_naive', localization, ['x', 'h', 'up', 'down'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'upfirdn_naive(...)' code ##################

    str_350784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, (-1)), 'str', 'Naive upfirdn processing in Python\n\n    Note: arg order (x, h) differs to facilitate apply_along_axis use.\n    ')
    
    # Assigning a Call to a Name (line 50):
    
    # Call to asarray(...): (line 50)
    # Processing the call arguments (line 50)
    # Getting the type of 'h' (line 50)
    h_350787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 19), 'h', False)
    # Processing the call keyword arguments (line 50)
    kwargs_350788 = {}
    # Getting the type of 'np' (line 50)
    np_350785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'np', False)
    # Obtaining the member 'asarray' of a type (line 50)
    asarray_350786 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 8), np_350785, 'asarray')
    # Calling asarray(args, kwargs) (line 50)
    asarray_call_result_350789 = invoke(stypy.reporting.localization.Localization(__file__, 50, 8), asarray_350786, *[h_350787], **kwargs_350788)
    
    # Assigning a type to the variable 'h' (line 50)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'h', asarray_call_result_350789)
    
    # Assigning a Call to a Name (line 51):
    
    # Call to zeros(...): (line 51)
    # Processing the call arguments (line 51)
    
    # Call to len(...): (line 51)
    # Processing the call arguments (line 51)
    # Getting the type of 'x' (line 51)
    x_350793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 23), 'x', False)
    # Processing the call keyword arguments (line 51)
    kwargs_350794 = {}
    # Getting the type of 'len' (line 51)
    len_350792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 19), 'len', False)
    # Calling len(args, kwargs) (line 51)
    len_call_result_350795 = invoke(stypy.reporting.localization.Localization(__file__, 51, 19), len_350792, *[x_350793], **kwargs_350794)
    
    # Getting the type of 'up' (line 51)
    up_350796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 28), 'up', False)
    # Applying the binary operator '*' (line 51)
    result_mul_350797 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 19), '*', len_call_result_350795, up_350796)
    
    # Getting the type of 'x' (line 51)
    x_350798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 32), 'x', False)
    # Obtaining the member 'dtype' of a type (line 51)
    dtype_350799 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 32), x_350798, 'dtype')
    # Processing the call keyword arguments (line 51)
    kwargs_350800 = {}
    # Getting the type of 'np' (line 51)
    np_350790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 10), 'np', False)
    # Obtaining the member 'zeros' of a type (line 51)
    zeros_350791 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 10), np_350790, 'zeros')
    # Calling zeros(args, kwargs) (line 51)
    zeros_call_result_350801 = invoke(stypy.reporting.localization.Localization(__file__, 51, 10), zeros_350791, *[result_mul_350797, dtype_350799], **kwargs_350800)
    
    # Assigning a type to the variable 'out' (line 51)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'out', zeros_call_result_350801)
    
    # Assigning a Name to a Subscript (line 52):
    # Getting the type of 'x' (line 52)
    x_350802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 16), 'x')
    # Getting the type of 'out' (line 52)
    out_350803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'out')
    # Getting the type of 'up' (line 52)
    up_350804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 10), 'up')
    slice_350805 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 52, 4), None, None, up_350804)
    # Storing an element on a container (line 52)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 4), out_350803, (slice_350805, x_350802))
    
    # Assigning a Subscript to a Name (line 53):
    
    # Obtaining the type of the subscript
    
    # Call to _output_len(...): (line 53)
    # Processing the call arguments (line 53)
    
    # Call to len(...): (line 53)
    # Processing the call arguments (line 53)
    # Getting the type of 'h' (line 53)
    h_350808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 55), 'h', False)
    # Processing the call keyword arguments (line 53)
    kwargs_350809 = {}
    # Getting the type of 'len' (line 53)
    len_350807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 51), 'len', False)
    # Calling len(args, kwargs) (line 53)
    len_call_result_350810 = invoke(stypy.reporting.localization.Localization(__file__, 53, 51), len_350807, *[h_350808], **kwargs_350809)
    
    
    # Call to len(...): (line 53)
    # Processing the call arguments (line 53)
    # Getting the type of 'x' (line 53)
    x_350812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 63), 'x', False)
    # Processing the call keyword arguments (line 53)
    kwargs_350813 = {}
    # Getting the type of 'len' (line 53)
    len_350811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 59), 'len', False)
    # Calling len(args, kwargs) (line 53)
    len_call_result_350814 = invoke(stypy.reporting.localization.Localization(__file__, 53, 59), len_350811, *[x_350812], **kwargs_350813)
    
    # Getting the type of 'up' (line 53)
    up_350815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 67), 'up', False)
    # Getting the type of 'down' (line 53)
    down_350816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 71), 'down', False)
    # Processing the call keyword arguments (line 53)
    kwargs_350817 = {}
    # Getting the type of '_output_len' (line 53)
    _output_len_350806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 39), '_output_len', False)
    # Calling _output_len(args, kwargs) (line 53)
    _output_len_call_result_350818 = invoke(stypy.reporting.localization.Localization(__file__, 53, 39), _output_len_350806, *[len_call_result_350810, len_call_result_350814, up_350815, down_350816], **kwargs_350817)
    
    slice_350819 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 53, 10), None, _output_len_call_result_350818, None)
    
    # Obtaining the type of the subscript
    # Getting the type of 'down' (line 53)
    down_350820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 32), 'down')
    slice_350821 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 53, 10), None, None, down_350820)
    
    # Call to convolve(...): (line 53)
    # Processing the call arguments (line 53)
    # Getting the type of 'h' (line 53)
    h_350824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 22), 'h', False)
    # Getting the type of 'out' (line 53)
    out_350825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 25), 'out', False)
    # Processing the call keyword arguments (line 53)
    kwargs_350826 = {}
    # Getting the type of 'np' (line 53)
    np_350822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 10), 'np', False)
    # Obtaining the member 'convolve' of a type (line 53)
    convolve_350823 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 10), np_350822, 'convolve')
    # Calling convolve(args, kwargs) (line 53)
    convolve_call_result_350827 = invoke(stypy.reporting.localization.Localization(__file__, 53, 10), convolve_350823, *[h_350824, out_350825], **kwargs_350826)
    
    # Obtaining the member '__getitem__' of a type (line 53)
    getitem___350828 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 10), convolve_call_result_350827, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 53)
    subscript_call_result_350829 = invoke(stypy.reporting.localization.Localization(__file__, 53, 10), getitem___350828, slice_350821)
    
    # Obtaining the member '__getitem__' of a type (line 53)
    getitem___350830 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 10), subscript_call_result_350829, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 53)
    subscript_call_result_350831 = invoke(stypy.reporting.localization.Localization(__file__, 53, 10), getitem___350830, slice_350819)
    
    # Assigning a type to the variable 'out' (line 53)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'out', subscript_call_result_350831)
    # Getting the type of 'out' (line 54)
    out_350832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 11), 'out')
    # Assigning a type to the variable 'stypy_return_type' (line 54)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'stypy_return_type', out_350832)
    
    # ################# End of 'upfirdn_naive(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'upfirdn_naive' in the type store
    # Getting the type of 'stypy_return_type' (line 45)
    stypy_return_type_350833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_350833)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'upfirdn_naive'
    return stypy_return_type_350833

# Assigning a type to the variable 'upfirdn_naive' (line 45)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 0), 'upfirdn_naive', upfirdn_naive)
# Declaration of the 'UpFIRDnCase' class

class UpFIRDnCase(object, ):
    str_350834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 4), 'str', 'Test _UpFIRDn object')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 59, 4, False)
        # Assigning a type to the variable 'self' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'UpFIRDnCase.__init__', ['up', 'down', 'h', 'x_dtype'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['up', 'down', 'h', 'x_dtype'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 60):
        # Getting the type of 'up' (line 60)
        up_350835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 18), 'up')
        # Getting the type of 'self' (line 60)
        self_350836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'self')
        # Setting the type of the member 'up' of a type (line 60)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 8), self_350836, 'up', up_350835)
        
        # Assigning a Name to a Attribute (line 61):
        # Getting the type of 'down' (line 61)
        down_350837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 20), 'down')
        # Getting the type of 'self' (line 61)
        self_350838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'self')
        # Setting the type of the member 'down' of a type (line 61)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 8), self_350838, 'down', down_350837)
        
        # Assigning a Call to a Attribute (line 62):
        
        # Call to atleast_1d(...): (line 62)
        # Processing the call arguments (line 62)
        # Getting the type of 'h' (line 62)
        h_350841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 31), 'h', False)
        # Processing the call keyword arguments (line 62)
        kwargs_350842 = {}
        # Getting the type of 'np' (line 62)
        np_350839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 17), 'np', False)
        # Obtaining the member 'atleast_1d' of a type (line 62)
        atleast_1d_350840 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 17), np_350839, 'atleast_1d')
        # Calling atleast_1d(args, kwargs) (line 62)
        atleast_1d_call_result_350843 = invoke(stypy.reporting.localization.Localization(__file__, 62, 17), atleast_1d_350840, *[h_350841], **kwargs_350842)
        
        # Getting the type of 'self' (line 62)
        self_350844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'self')
        # Setting the type of the member 'h' of a type (line 62)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 8), self_350844, 'h', atleast_1d_call_result_350843)
        
        # Assigning a Name to a Attribute (line 63):
        # Getting the type of 'x_dtype' (line 63)
        x_dtype_350845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 23), 'x_dtype')
        # Getting the type of 'self' (line 63)
        self_350846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'self')
        # Setting the type of the member 'x_dtype' of a type (line 63)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 8), self_350846, 'x_dtype', x_dtype_350845)
        
        # Assigning a Call to a Attribute (line 64):
        
        # Call to RandomState(...): (line 64)
        # Processing the call arguments (line 64)
        int_350850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 41), 'int')
        # Processing the call keyword arguments (line 64)
        kwargs_350851 = {}
        # Getting the type of 'np' (line 64)
        np_350847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 19), 'np', False)
        # Obtaining the member 'random' of a type (line 64)
        random_350848 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 19), np_350847, 'random')
        # Obtaining the member 'RandomState' of a type (line 64)
        RandomState_350849 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 19), random_350848, 'RandomState')
        # Calling RandomState(args, kwargs) (line 64)
        RandomState_call_result_350852 = invoke(stypy.reporting.localization.Localization(__file__, 64, 19), RandomState_350849, *[int_350850], **kwargs_350851)
        
        # Getting the type of 'self' (line 64)
        self_350853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'self')
        # Setting the type of the member 'rng' of a type (line 64)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 8), self_350853, 'rng', RandomState_call_result_350852)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def __call__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__call__'
        module_type_store = module_type_store.open_function_context('__call__', 66, 4, False)
        # Assigning a type to the variable 'self' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        UpFIRDnCase.__call__.__dict__.__setitem__('stypy_localization', localization)
        UpFIRDnCase.__call__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        UpFIRDnCase.__call__.__dict__.__setitem__('stypy_type_store', module_type_store)
        UpFIRDnCase.__call__.__dict__.__setitem__('stypy_function_name', 'UpFIRDnCase.__call__')
        UpFIRDnCase.__call__.__dict__.__setitem__('stypy_param_names_list', [])
        UpFIRDnCase.__call__.__dict__.__setitem__('stypy_varargs_param_name', None)
        UpFIRDnCase.__call__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        UpFIRDnCase.__call__.__dict__.__setitem__('stypy_call_defaults', defaults)
        UpFIRDnCase.__call__.__dict__.__setitem__('stypy_call_varargs', varargs)
        UpFIRDnCase.__call__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        UpFIRDnCase.__call__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'UpFIRDnCase.__call__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__call__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__call__(...)' code ##################

        
        # Call to scrub(...): (line 68)
        # Processing the call arguments (line 68)
        
        # Call to ones(...): (line 68)
        # Processing the call arguments (line 68)
        int_350858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 27), 'int')
        # Getting the type of 'self' (line 68)
        self_350859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 30), 'self', False)
        # Obtaining the member 'x_dtype' of a type (line 68)
        x_dtype_350860 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 30), self_350859, 'x_dtype')
        # Processing the call keyword arguments (line 68)
        kwargs_350861 = {}
        # Getting the type of 'np' (line 68)
        np_350856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 19), 'np', False)
        # Obtaining the member 'ones' of a type (line 68)
        ones_350857 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 19), np_350856, 'ones')
        # Calling ones(args, kwargs) (line 68)
        ones_call_result_350862 = invoke(stypy.reporting.localization.Localization(__file__, 68, 19), ones_350857, *[int_350858, x_dtype_350860], **kwargs_350861)
        
        # Processing the call keyword arguments (line 68)
        kwargs_350863 = {}
        # Getting the type of 'self' (line 68)
        self_350854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'self', False)
        # Obtaining the member 'scrub' of a type (line 68)
        scrub_350855 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 8), self_350854, 'scrub')
        # Calling scrub(args, kwargs) (line 68)
        scrub_call_result_350864 = invoke(stypy.reporting.localization.Localization(__file__, 68, 8), scrub_350855, *[ones_call_result_350862], **kwargs_350863)
        
        
        # Call to scrub(...): (line 70)
        # Processing the call arguments (line 70)
        
        # Call to ones(...): (line 70)
        # Processing the call arguments (line 70)
        int_350869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 27), 'int')
        # Getting the type of 'self' (line 70)
        self_350870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 31), 'self', False)
        # Obtaining the member 'x_dtype' of a type (line 70)
        x_dtype_350871 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 31), self_350870, 'x_dtype')
        # Processing the call keyword arguments (line 70)
        kwargs_350872 = {}
        # Getting the type of 'np' (line 70)
        np_350867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 19), 'np', False)
        # Obtaining the member 'ones' of a type (line 70)
        ones_350868 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 19), np_350867, 'ones')
        # Calling ones(args, kwargs) (line 70)
        ones_call_result_350873 = invoke(stypy.reporting.localization.Localization(__file__, 70, 19), ones_350868, *[int_350869, x_dtype_350871], **kwargs_350872)
        
        # Processing the call keyword arguments (line 70)
        kwargs_350874 = {}
        # Getting the type of 'self' (line 70)
        self_350865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'self', False)
        # Obtaining the member 'scrub' of a type (line 70)
        scrub_350866 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 8), self_350865, 'scrub')
        # Calling scrub(args, kwargs) (line 70)
        scrub_call_result_350875 = invoke(stypy.reporting.localization.Localization(__file__, 70, 8), scrub_350866, *[ones_call_result_350873], **kwargs_350874)
        
        
        # Assigning a Call to a Name (line 72):
        
        # Call to astype(...): (line 72)
        # Processing the call arguments (line 72)
        # Getting the type of 'self' (line 72)
        self_350883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 38), 'self', False)
        # Obtaining the member 'x_dtype' of a type (line 72)
        x_dtype_350884 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 38), self_350883, 'x_dtype')
        # Processing the call keyword arguments (line 72)
        kwargs_350885 = {}
        
        # Call to randn(...): (line 72)
        # Processing the call arguments (line 72)
        int_350879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 27), 'int')
        # Processing the call keyword arguments (line 72)
        kwargs_350880 = {}
        # Getting the type of 'self' (line 72)
        self_350876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 12), 'self', False)
        # Obtaining the member 'rng' of a type (line 72)
        rng_350877 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 12), self_350876, 'rng')
        # Obtaining the member 'randn' of a type (line 72)
        randn_350878 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 12), rng_350877, 'randn')
        # Calling randn(args, kwargs) (line 72)
        randn_call_result_350881 = invoke(stypy.reporting.localization.Localization(__file__, 72, 12), randn_350878, *[int_350879], **kwargs_350880)
        
        # Obtaining the member 'astype' of a type (line 72)
        astype_350882 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 12), randn_call_result_350881, 'astype')
        # Calling astype(args, kwargs) (line 72)
        astype_call_result_350886 = invoke(stypy.reporting.localization.Localization(__file__, 72, 12), astype_350882, *[x_dtype_350884], **kwargs_350885)
        
        # Assigning a type to the variable 'x' (line 72)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'x', astype_call_result_350886)
        
        
        # Getting the type of 'self' (line 73)
        self_350887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 11), 'self')
        # Obtaining the member 'x_dtype' of a type (line 73)
        x_dtype_350888 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 11), self_350887, 'x_dtype')
        
        # Obtaining an instance of the builtin type 'tuple' (line 73)
        tuple_350889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 73)
        # Adding element type (line 73)
        # Getting the type of 'np' (line 73)
        np_350890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 28), 'np')
        # Obtaining the member 'complex64' of a type (line 73)
        complex64_350891 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 28), np_350890, 'complex64')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 28), tuple_350889, complex64_350891)
        # Adding element type (line 73)
        # Getting the type of 'np' (line 73)
        np_350892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 42), 'np')
        # Obtaining the member 'complex128' of a type (line 73)
        complex128_350893 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 42), np_350892, 'complex128')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 28), tuple_350889, complex128_350893)
        
        # Applying the binary operator 'in' (line 73)
        result_contains_350894 = python_operator(stypy.reporting.localization.Localization(__file__, 73, 11), 'in', x_dtype_350888, tuple_350889)
        
        # Testing the type of an if condition (line 73)
        if_condition_350895 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 73, 8), result_contains_350894)
        # Assigning a type to the variable 'if_condition_350895' (line 73)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'if_condition_350895', if_condition_350895)
        # SSA begins for if statement (line 73)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'x' (line 74)
        x_350896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 12), 'x')
        complex_350897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 17), 'complex')
        
        # Call to randn(...): (line 74)
        # Processing the call arguments (line 74)
        int_350901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 37), 'int')
        # Processing the call keyword arguments (line 74)
        kwargs_350902 = {}
        # Getting the type of 'self' (line 74)
        self_350898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 22), 'self', False)
        # Obtaining the member 'rng' of a type (line 74)
        rng_350899 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 22), self_350898, 'rng')
        # Obtaining the member 'randn' of a type (line 74)
        randn_350900 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 22), rng_350899, 'randn')
        # Calling randn(args, kwargs) (line 74)
        randn_call_result_350903 = invoke(stypy.reporting.localization.Localization(__file__, 74, 22), randn_350900, *[int_350901], **kwargs_350902)
        
        # Applying the binary operator '*' (line 74)
        result_mul_350904 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 17), '*', complex_350897, randn_call_result_350903)
        
        # Applying the binary operator '+=' (line 74)
        result_iadd_350905 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 12), '+=', x_350896, result_mul_350904)
        # Assigning a type to the variable 'x' (line 74)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 12), 'x', result_iadd_350905)
        
        # SSA join for if statement (line 73)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to scrub(...): (line 75)
        # Processing the call arguments (line 75)
        # Getting the type of 'x' (line 75)
        x_350908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 19), 'x', False)
        # Processing the call keyword arguments (line 75)
        kwargs_350909 = {}
        # Getting the type of 'self' (line 75)
        self_350906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'self', False)
        # Obtaining the member 'scrub' of a type (line 75)
        scrub_350907 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 8), self_350906, 'scrub')
        # Calling scrub(args, kwargs) (line 75)
        scrub_call_result_350910 = invoke(stypy.reporting.localization.Localization(__file__, 75, 8), scrub_350907, *[x_350908], **kwargs_350909)
        
        
        # Call to scrub(...): (line 77)
        # Processing the call arguments (line 77)
        
        # Call to astype(...): (line 77)
        # Processing the call arguments (line 77)
        # Getting the type of 'self' (line 77)
        self_350919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 40), 'self', False)
        # Obtaining the member 'x_dtype' of a type (line 77)
        x_dtype_350920 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 40), self_350919, 'x_dtype')
        # Processing the call keyword arguments (line 77)
        kwargs_350921 = {}
        
        # Call to arange(...): (line 77)
        # Processing the call arguments (line 77)
        int_350915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 29), 'int')
        # Processing the call keyword arguments (line 77)
        kwargs_350916 = {}
        # Getting the type of 'np' (line 77)
        np_350913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 19), 'np', False)
        # Obtaining the member 'arange' of a type (line 77)
        arange_350914 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 19), np_350913, 'arange')
        # Calling arange(args, kwargs) (line 77)
        arange_call_result_350917 = invoke(stypy.reporting.localization.Localization(__file__, 77, 19), arange_350914, *[int_350915], **kwargs_350916)
        
        # Obtaining the member 'astype' of a type (line 77)
        astype_350918 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 19), arange_call_result_350917, 'astype')
        # Calling astype(args, kwargs) (line 77)
        astype_call_result_350922 = invoke(stypy.reporting.localization.Localization(__file__, 77, 19), astype_350918, *[x_dtype_350920], **kwargs_350921)
        
        # Processing the call keyword arguments (line 77)
        kwargs_350923 = {}
        # Getting the type of 'self' (line 77)
        self_350911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'self', False)
        # Obtaining the member 'scrub' of a type (line 77)
        scrub_350912 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 8), self_350911, 'scrub')
        # Calling scrub(args, kwargs) (line 77)
        scrub_call_result_350924 = invoke(stypy.reporting.localization.Localization(__file__, 77, 8), scrub_350912, *[astype_call_result_350922], **kwargs_350923)
        
        
        # Assigning a Tuple to a Name (line 79):
        
        # Obtaining an instance of the builtin type 'tuple' (line 79)
        tuple_350925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 16), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 79)
        # Adding element type (line 79)
        int_350926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 16), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 16), tuple_350925, int_350926)
        # Adding element type (line 79)
        int_350927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 16), tuple_350925, int_350927)
        # Adding element type (line 79)
        int_350928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 16), tuple_350925, int_350928)
        
        # Assigning a type to the variable 'size' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'size', tuple_350925)
        
        # Assigning a Call to a Name (line 80):
        
        # Call to astype(...): (line 80)
        # Processing the call arguments (line 80)
        # Getting the type of 'self' (line 80)
        self_350936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 41), 'self', False)
        # Obtaining the member 'x_dtype' of a type (line 80)
        x_dtype_350937 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 41), self_350936, 'x_dtype')
        # Processing the call keyword arguments (line 80)
        kwargs_350938 = {}
        
        # Call to randn(...): (line 80)
        # Getting the type of 'size' (line 80)
        size_350932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 28), 'size', False)
        # Processing the call keyword arguments (line 80)
        kwargs_350933 = {}
        # Getting the type of 'self' (line 80)
        self_350929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 12), 'self', False)
        # Obtaining the member 'rng' of a type (line 80)
        rng_350930 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 12), self_350929, 'rng')
        # Obtaining the member 'randn' of a type (line 80)
        randn_350931 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 12), rng_350930, 'randn')
        # Calling randn(args, kwargs) (line 80)
        randn_call_result_350934 = invoke(stypy.reporting.localization.Localization(__file__, 80, 12), randn_350931, *[size_350932], **kwargs_350933)
        
        # Obtaining the member 'astype' of a type (line 80)
        astype_350935 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 12), randn_call_result_350934, 'astype')
        # Calling astype(args, kwargs) (line 80)
        astype_call_result_350939 = invoke(stypy.reporting.localization.Localization(__file__, 80, 12), astype_350935, *[x_dtype_350937], **kwargs_350938)
        
        # Assigning a type to the variable 'x' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'x', astype_call_result_350939)
        
        
        # Getting the type of 'self' (line 81)
        self_350940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 11), 'self')
        # Obtaining the member 'x_dtype' of a type (line 81)
        x_dtype_350941 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 11), self_350940, 'x_dtype')
        
        # Obtaining an instance of the builtin type 'tuple' (line 81)
        tuple_350942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 81)
        # Adding element type (line 81)
        # Getting the type of 'np' (line 81)
        np_350943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 28), 'np')
        # Obtaining the member 'complex64' of a type (line 81)
        complex64_350944 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 28), np_350943, 'complex64')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 28), tuple_350942, complex64_350944)
        # Adding element type (line 81)
        # Getting the type of 'np' (line 81)
        np_350945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 42), 'np')
        # Obtaining the member 'complex128' of a type (line 81)
        complex128_350946 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 42), np_350945, 'complex128')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 28), tuple_350942, complex128_350946)
        
        # Applying the binary operator 'in' (line 81)
        result_contains_350947 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 11), 'in', x_dtype_350941, tuple_350942)
        
        # Testing the type of an if condition (line 81)
        if_condition_350948 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 81, 8), result_contains_350947)
        # Assigning a type to the variable 'if_condition_350948' (line 81)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'if_condition_350948', if_condition_350948)
        # SSA begins for if statement (line 81)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'x' (line 82)
        x_350949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 12), 'x')
        complex_350950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 17), 'complex')
        
        # Call to randn(...): (line 82)
        # Getting the type of 'size' (line 82)
        size_350954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 38), 'size', False)
        # Processing the call keyword arguments (line 82)
        kwargs_350955 = {}
        # Getting the type of 'self' (line 82)
        self_350951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 22), 'self', False)
        # Obtaining the member 'rng' of a type (line 82)
        rng_350952 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 22), self_350951, 'rng')
        # Obtaining the member 'randn' of a type (line 82)
        randn_350953 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 22), rng_350952, 'randn')
        # Calling randn(args, kwargs) (line 82)
        randn_call_result_350956 = invoke(stypy.reporting.localization.Localization(__file__, 82, 22), randn_350953, *[size_350954], **kwargs_350955)
        
        # Applying the binary operator '*' (line 82)
        result_mul_350957 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 17), '*', complex_350950, randn_call_result_350956)
        
        # Applying the binary operator '+=' (line 82)
        result_iadd_350958 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 12), '+=', x_350949, result_mul_350957)
        # Assigning a type to the variable 'x' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 12), 'x', result_iadd_350958)
        
        # SSA join for if statement (line 81)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to range(...): (line 83)
        # Processing the call arguments (line 83)
        
        # Call to len(...): (line 83)
        # Processing the call arguments (line 83)
        # Getting the type of 'size' (line 83)
        size_350961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 30), 'size', False)
        # Processing the call keyword arguments (line 83)
        kwargs_350962 = {}
        # Getting the type of 'len' (line 83)
        len_350960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 26), 'len', False)
        # Calling len(args, kwargs) (line 83)
        len_call_result_350963 = invoke(stypy.reporting.localization.Localization(__file__, 83, 26), len_350960, *[size_350961], **kwargs_350962)
        
        # Processing the call keyword arguments (line 83)
        kwargs_350964 = {}
        # Getting the type of 'range' (line 83)
        range_350959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 20), 'range', False)
        # Calling range(args, kwargs) (line 83)
        range_call_result_350965 = invoke(stypy.reporting.localization.Localization(__file__, 83, 20), range_350959, *[len_call_result_350963], **kwargs_350964)
        
        # Testing the type of a for loop iterable (line 83)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 83, 8), range_call_result_350965)
        # Getting the type of the for loop variable (line 83)
        for_loop_var_350966 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 83, 8), range_call_result_350965)
        # Assigning a type to the variable 'axis' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'axis', for_loop_var_350966)
        # SSA begins for a for statement (line 83)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to scrub(...): (line 84)
        # Processing the call arguments (line 84)
        # Getting the type of 'x' (line 84)
        x_350969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 23), 'x', False)
        # Processing the call keyword arguments (line 84)
        # Getting the type of 'axis' (line 84)
        axis_350970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 31), 'axis', False)
        keyword_350971 = axis_350970
        kwargs_350972 = {'axis': keyword_350971}
        # Getting the type of 'self' (line 84)
        self_350967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 12), 'self', False)
        # Obtaining the member 'scrub' of a type (line 84)
        scrub_350968 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 12), self_350967, 'scrub')
        # Calling scrub(args, kwargs) (line 84)
        scrub_call_result_350973 = invoke(stypy.reporting.localization.Localization(__file__, 84, 12), scrub_350968, *[x_350969], **kwargs_350972)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Attribute to a Name (line 85):
        
        # Obtaining the type of the subscript
        slice_350974 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 85, 12), None, None, None)
        int_350975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 19), 'int')
        slice_350976 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 85, 12), None, None, int_350975)
        int_350977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 22), 'int')
        int_350978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 25), 'int')
        slice_350979 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 85, 12), int_350977, None, int_350978)
        # Getting the type of 'x' (line 85)
        x_350980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 12), 'x')
        # Obtaining the member '__getitem__' of a type (line 85)
        getitem___350981 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 12), x_350980, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 85)
        subscript_call_result_350982 = invoke(stypy.reporting.localization.Localization(__file__, 85, 12), getitem___350981, (slice_350974, slice_350976, slice_350979))
        
        # Obtaining the member 'T' of a type (line 85)
        T_350983 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 12), subscript_call_result_350982, 'T')
        # Assigning a type to the variable 'x' (line 85)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'x', T_350983)
        
        
        # Call to range(...): (line 86)
        # Processing the call arguments (line 86)
        
        # Call to len(...): (line 86)
        # Processing the call arguments (line 86)
        # Getting the type of 'size' (line 86)
        size_350986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 30), 'size', False)
        # Processing the call keyword arguments (line 86)
        kwargs_350987 = {}
        # Getting the type of 'len' (line 86)
        len_350985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 26), 'len', False)
        # Calling len(args, kwargs) (line 86)
        len_call_result_350988 = invoke(stypy.reporting.localization.Localization(__file__, 86, 26), len_350985, *[size_350986], **kwargs_350987)
        
        # Processing the call keyword arguments (line 86)
        kwargs_350989 = {}
        # Getting the type of 'range' (line 86)
        range_350984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 20), 'range', False)
        # Calling range(args, kwargs) (line 86)
        range_call_result_350990 = invoke(stypy.reporting.localization.Localization(__file__, 86, 20), range_350984, *[len_call_result_350988], **kwargs_350989)
        
        # Testing the type of a for loop iterable (line 86)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 86, 8), range_call_result_350990)
        # Getting the type of the for loop variable (line 86)
        for_loop_var_350991 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 86, 8), range_call_result_350990)
        # Assigning a type to the variable 'axis' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'axis', for_loop_var_350991)
        # SSA begins for a for statement (line 86)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to scrub(...): (line 87)
        # Processing the call arguments (line 87)
        # Getting the type of 'x' (line 87)
        x_350994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 23), 'x', False)
        # Processing the call keyword arguments (line 87)
        # Getting the type of 'axis' (line 87)
        axis_350995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 31), 'axis', False)
        keyword_350996 = axis_350995
        kwargs_350997 = {'axis': keyword_350996}
        # Getting the type of 'self' (line 87)
        self_350992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 12), 'self', False)
        # Obtaining the member 'scrub' of a type (line 87)
        scrub_350993 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 12), self_350992, 'scrub')
        # Calling scrub(args, kwargs) (line 87)
        scrub_call_result_350998 = invoke(stypy.reporting.localization.Localization(__file__, 87, 12), scrub_350993, *[x_350994], **kwargs_350997)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__call__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__call__' in the type store
        # Getting the type of 'stypy_return_type' (line 66)
        stypy_return_type_350999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_350999)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__call__'
        return stypy_return_type_350999


    @norecursion
    def scrub(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_351000 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 28), 'int')
        defaults = [int_351000]
        # Create a new context for function 'scrub'
        module_type_store = module_type_store.open_function_context('scrub', 89, 4, False)
        # Assigning a type to the variable 'self' (line 90)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        UpFIRDnCase.scrub.__dict__.__setitem__('stypy_localization', localization)
        UpFIRDnCase.scrub.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        UpFIRDnCase.scrub.__dict__.__setitem__('stypy_type_store', module_type_store)
        UpFIRDnCase.scrub.__dict__.__setitem__('stypy_function_name', 'UpFIRDnCase.scrub')
        UpFIRDnCase.scrub.__dict__.__setitem__('stypy_param_names_list', ['x', 'axis'])
        UpFIRDnCase.scrub.__dict__.__setitem__('stypy_varargs_param_name', None)
        UpFIRDnCase.scrub.__dict__.__setitem__('stypy_kwargs_param_name', None)
        UpFIRDnCase.scrub.__dict__.__setitem__('stypy_call_defaults', defaults)
        UpFIRDnCase.scrub.__dict__.__setitem__('stypy_call_varargs', varargs)
        UpFIRDnCase.scrub.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        UpFIRDnCase.scrub.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'UpFIRDnCase.scrub', ['x', 'axis'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'scrub', localization, ['x', 'axis'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'scrub(...)' code ##################

        
        # Assigning a Call to a Name (line 90):
        
        # Call to apply_along_axis(...): (line 90)
        # Processing the call arguments (line 90)
        # Getting the type of 'upfirdn_naive' (line 90)
        upfirdn_naive_351003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 33), 'upfirdn_naive', False)
        # Getting the type of 'axis' (line 90)
        axis_351004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 48), 'axis', False)
        # Getting the type of 'x' (line 90)
        x_351005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 54), 'x', False)
        # Getting the type of 'self' (line 91)
        self_351006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 33), 'self', False)
        # Obtaining the member 'h' of a type (line 91)
        h_351007 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 33), self_351006, 'h')
        # Getting the type of 'self' (line 91)
        self_351008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 41), 'self', False)
        # Obtaining the member 'up' of a type (line 91)
        up_351009 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 41), self_351008, 'up')
        # Getting the type of 'self' (line 91)
        self_351010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 50), 'self', False)
        # Obtaining the member 'down' of a type (line 91)
        down_351011 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 50), self_351010, 'down')
        # Processing the call keyword arguments (line 90)
        kwargs_351012 = {}
        # Getting the type of 'np' (line 90)
        np_351001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 13), 'np', False)
        # Obtaining the member 'apply_along_axis' of a type (line 90)
        apply_along_axis_351002 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 13), np_351001, 'apply_along_axis')
        # Calling apply_along_axis(args, kwargs) (line 90)
        apply_along_axis_call_result_351013 = invoke(stypy.reporting.localization.Localization(__file__, 90, 13), apply_along_axis_351002, *[upfirdn_naive_351003, axis_351004, x_351005, h_351007, up_351009, down_351011], **kwargs_351012)
        
        # Assigning a type to the variable 'yr' (line 90)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'yr', apply_along_axis_call_result_351013)
        
        # Assigning a Call to a Name (line 92):
        
        # Call to upfirdn(...): (line 92)
        # Processing the call arguments (line 92)
        # Getting the type of 'self' (line 92)
        self_351015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 20), 'self', False)
        # Obtaining the member 'h' of a type (line 92)
        h_351016 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 20), self_351015, 'h')
        # Getting the type of 'x' (line 92)
        x_351017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 28), 'x', False)
        # Getting the type of 'self' (line 92)
        self_351018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 31), 'self', False)
        # Obtaining the member 'up' of a type (line 92)
        up_351019 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 31), self_351018, 'up')
        # Getting the type of 'self' (line 92)
        self_351020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 40), 'self', False)
        # Obtaining the member 'down' of a type (line 92)
        down_351021 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 40), self_351020, 'down')
        # Processing the call keyword arguments (line 92)
        # Getting the type of 'axis' (line 92)
        axis_351022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 56), 'axis', False)
        keyword_351023 = axis_351022
        kwargs_351024 = {'axis': keyword_351023}
        # Getting the type of 'upfirdn' (line 92)
        upfirdn_351014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 12), 'upfirdn', False)
        # Calling upfirdn(args, kwargs) (line 92)
        upfirdn_call_result_351025 = invoke(stypy.reporting.localization.Localization(__file__, 92, 12), upfirdn_351014, *[h_351016, x_351017, up_351019, down_351021], **kwargs_351024)
        
        # Assigning a type to the variable 'y' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'y', upfirdn_call_result_351025)
        
        # Assigning a Tuple to a Name (line 93):
        
        # Obtaining an instance of the builtin type 'tuple' (line 93)
        tuple_351026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 18), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 93)
        # Adding element type (line 93)
        # Getting the type of 'self' (line 93)
        self_351027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 18), 'self')
        # Obtaining the member 'h' of a type (line 93)
        h_351028 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 18), self_351027, 'h')
        # Obtaining the member 'dtype' of a type (line 93)
        dtype_351029 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 18), h_351028, 'dtype')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 18), tuple_351026, dtype_351029)
        # Adding element type (line 93)
        # Getting the type of 'x' (line 93)
        x_351030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 32), 'x')
        # Obtaining the member 'dtype' of a type (line 93)
        dtype_351031 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 32), x_351030, 'dtype')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 18), tuple_351026, dtype_351031)
        
        # Assigning a type to the variable 'dtypes' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'dtypes', tuple_351026)
        
        
        # Call to all(...): (line 94)
        # Processing the call arguments (line 94)
        # Calculating generator expression
        module_type_store = module_type_store.open_function_context('list comprehension expression', 94, 15, True)
        # Calculating comprehension expression
        # Getting the type of 'dtypes' (line 94)
        dtypes_351037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 42), 'dtypes', False)
        comprehension_351038 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 94, 15), dtypes_351037)
        # Assigning a type to the variable 'd' (line 94)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 15), 'd', comprehension_351038)
        
        # Getting the type of 'd' (line 94)
        d_351033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 15), 'd', False)
        # Getting the type of 'np' (line 94)
        np_351034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 20), 'np', False)
        # Obtaining the member 'complex64' of a type (line 94)
        complex64_351035 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 20), np_351034, 'complex64')
        # Applying the binary operator '==' (line 94)
        result_eq_351036 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 15), '==', d_351033, complex64_351035)
        
        list_351039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 15), 'list')
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 94, 15), list_351039, result_eq_351036)
        # Processing the call keyword arguments (line 94)
        kwargs_351040 = {}
        # Getting the type of 'all' (line 94)
        all_351032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 11), 'all', False)
        # Calling all(args, kwargs) (line 94)
        all_call_result_351041 = invoke(stypy.reporting.localization.Localization(__file__, 94, 11), all_351032, *[list_351039], **kwargs_351040)
        
        # Testing the type of an if condition (line 94)
        if_condition_351042 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 94, 8), all_call_result_351041)
        # Assigning a type to the variable 'if_condition_351042' (line 94)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'if_condition_351042', if_condition_351042)
        # SSA begins for if statement (line 94)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to assert_equal(...): (line 95)
        # Processing the call arguments (line 95)
        # Getting the type of 'y' (line 95)
        y_351044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 25), 'y', False)
        # Obtaining the member 'dtype' of a type (line 95)
        dtype_351045 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 25), y_351044, 'dtype')
        # Getting the type of 'np' (line 95)
        np_351046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 34), 'np', False)
        # Obtaining the member 'complex64' of a type (line 95)
        complex64_351047 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 34), np_351046, 'complex64')
        # Processing the call keyword arguments (line 95)
        kwargs_351048 = {}
        # Getting the type of 'assert_equal' (line 95)
        assert_equal_351043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 95)
        assert_equal_call_result_351049 = invoke(stypy.reporting.localization.Localization(__file__, 95, 12), assert_equal_351043, *[dtype_351045, complex64_351047], **kwargs_351048)
        
        # SSA branch for the else part of an if statement (line 94)
        module_type_store.open_ssa_branch('else')
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'np' (line 96)
        np_351050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 13), 'np')
        # Obtaining the member 'complex64' of a type (line 96)
        complex64_351051 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 13), np_351050, 'complex64')
        # Getting the type of 'dtypes' (line 96)
        dtypes_351052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 29), 'dtypes')
        # Applying the binary operator 'in' (line 96)
        result_contains_351053 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 13), 'in', complex64_351051, dtypes_351052)
        
        
        # Getting the type of 'np' (line 96)
        np_351054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 40), 'np')
        # Obtaining the member 'float32' of a type (line 96)
        float32_351055 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 40), np_351054, 'float32')
        # Getting the type of 'dtypes' (line 96)
        dtypes_351056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 54), 'dtypes')
        # Applying the binary operator 'in' (line 96)
        result_contains_351057 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 40), 'in', float32_351055, dtypes_351056)
        
        # Applying the binary operator 'and' (line 96)
        result_and_keyword_351058 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 13), 'and', result_contains_351053, result_contains_351057)
        
        # Testing the type of an if condition (line 96)
        if_condition_351059 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 96, 13), result_and_keyword_351058)
        # Assigning a type to the variable 'if_condition_351059' (line 96)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 13), 'if_condition_351059', if_condition_351059)
        # SSA begins for if statement (line 96)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to assert_equal(...): (line 97)
        # Processing the call arguments (line 97)
        # Getting the type of 'y' (line 97)
        y_351061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 25), 'y', False)
        # Obtaining the member 'dtype' of a type (line 97)
        dtype_351062 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 25), y_351061, 'dtype')
        # Getting the type of 'np' (line 97)
        np_351063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 34), 'np', False)
        # Obtaining the member 'complex64' of a type (line 97)
        complex64_351064 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 34), np_351063, 'complex64')
        # Processing the call keyword arguments (line 97)
        kwargs_351065 = {}
        # Getting the type of 'assert_equal' (line 97)
        assert_equal_351060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 97)
        assert_equal_call_result_351066 = invoke(stypy.reporting.localization.Localization(__file__, 97, 12), assert_equal_351060, *[dtype_351062, complex64_351064], **kwargs_351065)
        
        # SSA branch for the else part of an if statement (line 96)
        module_type_store.open_ssa_branch('else')
        
        
        # Call to all(...): (line 98)
        # Processing the call arguments (line 98)
        # Calculating generator expression
        module_type_store = module_type_store.open_function_context('list comprehension expression', 98, 17, True)
        # Calculating comprehension expression
        # Getting the type of 'dtypes' (line 98)
        dtypes_351072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 42), 'dtypes', False)
        comprehension_351073 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 17), dtypes_351072)
        # Assigning a type to the variable 'd' (line 98)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 17), 'd', comprehension_351073)
        
        # Getting the type of 'd' (line 98)
        d_351068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 17), 'd', False)
        # Getting the type of 'np' (line 98)
        np_351069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 22), 'np', False)
        # Obtaining the member 'float32' of a type (line 98)
        float32_351070 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 22), np_351069, 'float32')
        # Applying the binary operator '==' (line 98)
        result_eq_351071 = python_operator(stypy.reporting.localization.Localization(__file__, 98, 17), '==', d_351068, float32_351070)
        
        list_351074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 17), 'list')
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 17), list_351074, result_eq_351071)
        # Processing the call keyword arguments (line 98)
        kwargs_351075 = {}
        # Getting the type of 'all' (line 98)
        all_351067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 13), 'all', False)
        # Calling all(args, kwargs) (line 98)
        all_call_result_351076 = invoke(stypy.reporting.localization.Localization(__file__, 98, 13), all_351067, *[list_351074], **kwargs_351075)
        
        # Testing the type of an if condition (line 98)
        if_condition_351077 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 98, 13), all_call_result_351076)
        # Assigning a type to the variable 'if_condition_351077' (line 98)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 13), 'if_condition_351077', if_condition_351077)
        # SSA begins for if statement (line 98)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to assert_equal(...): (line 99)
        # Processing the call arguments (line 99)
        # Getting the type of 'y' (line 99)
        y_351079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 25), 'y', False)
        # Obtaining the member 'dtype' of a type (line 99)
        dtype_351080 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 25), y_351079, 'dtype')
        # Getting the type of 'np' (line 99)
        np_351081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 34), 'np', False)
        # Obtaining the member 'float32' of a type (line 99)
        float32_351082 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 34), np_351081, 'float32')
        # Processing the call keyword arguments (line 99)
        kwargs_351083 = {}
        # Getting the type of 'assert_equal' (line 99)
        assert_equal_351078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 99)
        assert_equal_call_result_351084 = invoke(stypy.reporting.localization.Localization(__file__, 99, 12), assert_equal_351078, *[dtype_351080, float32_351082], **kwargs_351083)
        
        # SSA branch for the else part of an if statement (line 98)
        module_type_store.open_ssa_branch('else')
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'np' (line 100)
        np_351085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 13), 'np')
        # Obtaining the member 'complex128' of a type (line 100)
        complex128_351086 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 13), np_351085, 'complex128')
        # Getting the type of 'dtypes' (line 100)
        dtypes_351087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 30), 'dtypes')
        # Applying the binary operator 'in' (line 100)
        result_contains_351088 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 13), 'in', complex128_351086, dtypes_351087)
        
        
        # Getting the type of 'np' (line 100)
        np_351089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 40), 'np')
        # Obtaining the member 'complex64' of a type (line 100)
        complex64_351090 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 40), np_351089, 'complex64')
        # Getting the type of 'dtypes' (line 100)
        dtypes_351091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 56), 'dtypes')
        # Applying the binary operator 'in' (line 100)
        result_contains_351092 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 40), 'in', complex64_351090, dtypes_351091)
        
        # Applying the binary operator 'or' (line 100)
        result_or_keyword_351093 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 13), 'or', result_contains_351088, result_contains_351092)
        
        # Testing the type of an if condition (line 100)
        if_condition_351094 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 100, 13), result_or_keyword_351093)
        # Assigning a type to the variable 'if_condition_351094' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 13), 'if_condition_351094', if_condition_351094)
        # SSA begins for if statement (line 100)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to assert_equal(...): (line 101)
        # Processing the call arguments (line 101)
        # Getting the type of 'y' (line 101)
        y_351096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 25), 'y', False)
        # Obtaining the member 'dtype' of a type (line 101)
        dtype_351097 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 25), y_351096, 'dtype')
        # Getting the type of 'np' (line 101)
        np_351098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 34), 'np', False)
        # Obtaining the member 'complex128' of a type (line 101)
        complex128_351099 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 34), np_351098, 'complex128')
        # Processing the call keyword arguments (line 101)
        kwargs_351100 = {}
        # Getting the type of 'assert_equal' (line 101)
        assert_equal_351095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 101)
        assert_equal_call_result_351101 = invoke(stypy.reporting.localization.Localization(__file__, 101, 12), assert_equal_351095, *[dtype_351097, complex128_351099], **kwargs_351100)
        
        # SSA branch for the else part of an if statement (line 100)
        module_type_store.open_ssa_branch('else')
        
        # Call to assert_equal(...): (line 103)
        # Processing the call arguments (line 103)
        # Getting the type of 'y' (line 103)
        y_351103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 25), 'y', False)
        # Obtaining the member 'dtype' of a type (line 103)
        dtype_351104 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 25), y_351103, 'dtype')
        # Getting the type of 'np' (line 103)
        np_351105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 34), 'np', False)
        # Obtaining the member 'float64' of a type (line 103)
        float64_351106 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 34), np_351105, 'float64')
        # Processing the call keyword arguments (line 103)
        kwargs_351107 = {}
        # Getting the type of 'assert_equal' (line 103)
        assert_equal_351102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 103)
        assert_equal_call_result_351108 = invoke(stypy.reporting.localization.Localization(__file__, 103, 12), assert_equal_351102, *[dtype_351104, float64_351106], **kwargs_351107)
        
        # SSA join for if statement (line 100)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 98)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 96)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 94)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to assert_allclose(...): (line 104)
        # Processing the call arguments (line 104)
        # Getting the type of 'yr' (line 104)
        yr_351110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 24), 'yr', False)
        # Getting the type of 'y' (line 104)
        y_351111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 28), 'y', False)
        # Processing the call keyword arguments (line 104)
        kwargs_351112 = {}
        # Getting the type of 'assert_allclose' (line 104)
        assert_allclose_351109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 104)
        assert_allclose_call_result_351113 = invoke(stypy.reporting.localization.Localization(__file__, 104, 8), assert_allclose_351109, *[yr_351110, y_351111], **kwargs_351112)
        
        
        # ################# End of 'scrub(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'scrub' in the type store
        # Getting the type of 'stypy_return_type' (line 89)
        stypy_return_type_351114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_351114)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'scrub'
        return stypy_return_type_351114


# Assigning a type to the variable 'UpFIRDnCase' (line 57)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 0), 'UpFIRDnCase', UpFIRDnCase)
# Declaration of the 'TestUpfirdn' class

class TestUpfirdn(object, ):

    @norecursion
    def test_valid_input(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_valid_input'
        module_type_store = module_type_store.open_function_context('test_valid_input', 109, 4, False)
        # Assigning a type to the variable 'self' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestUpfirdn.test_valid_input.__dict__.__setitem__('stypy_localization', localization)
        TestUpfirdn.test_valid_input.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestUpfirdn.test_valid_input.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestUpfirdn.test_valid_input.__dict__.__setitem__('stypy_function_name', 'TestUpfirdn.test_valid_input')
        TestUpfirdn.test_valid_input.__dict__.__setitem__('stypy_param_names_list', [])
        TestUpfirdn.test_valid_input.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestUpfirdn.test_valid_input.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestUpfirdn.test_valid_input.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestUpfirdn.test_valid_input.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestUpfirdn.test_valid_input.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestUpfirdn.test_valid_input.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestUpfirdn.test_valid_input', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_valid_input', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_valid_input(...)' code ##################

        
        # Call to assert_raises(...): (line 110)
        # Processing the call arguments (line 110)
        # Getting the type of 'ValueError' (line 110)
        ValueError_351116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 22), 'ValueError', False)
        # Getting the type of 'upfirdn' (line 110)
        upfirdn_351117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 34), 'upfirdn', False)
        
        # Obtaining an instance of the builtin type 'list' (line 110)
        list_351118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 43), 'list')
        # Adding type elements to the builtin type 'list' instance (line 110)
        # Adding element type (line 110)
        int_351119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 43), list_351118, int_351119)
        
        
        # Obtaining an instance of the builtin type 'list' (line 110)
        list_351120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 48), 'list')
        # Adding type elements to the builtin type 'list' instance (line 110)
        # Adding element type (line 110)
        int_351121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 49), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 48), list_351120, int_351121)
        
        int_351122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 53), 'int')
        int_351123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 56), 'int')
        # Processing the call keyword arguments (line 110)
        kwargs_351124 = {}
        # Getting the type of 'assert_raises' (line 110)
        assert_raises_351115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 110)
        assert_raises_call_result_351125 = invoke(stypy.reporting.localization.Localization(__file__, 110, 8), assert_raises_351115, *[ValueError_351116, upfirdn_351117, list_351118, list_351120, int_351122, int_351123], **kwargs_351124)
        
        
        # Call to assert_raises(...): (line 111)
        # Processing the call arguments (line 111)
        # Getting the type of 'ValueError' (line 111)
        ValueError_351127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 22), 'ValueError', False)
        # Getting the type of 'upfirdn' (line 111)
        upfirdn_351128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 34), 'upfirdn', False)
        
        # Obtaining an instance of the builtin type 'list' (line 111)
        list_351129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 43), 'list')
        # Adding type elements to the builtin type 'list' instance (line 111)
        
        
        # Obtaining an instance of the builtin type 'list' (line 111)
        list_351130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 47), 'list')
        # Adding type elements to the builtin type 'list' instance (line 111)
        # Adding element type (line 111)
        int_351131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 48), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 47), list_351130, int_351131)
        
        int_351132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 52), 'int')
        int_351133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 55), 'int')
        # Processing the call keyword arguments (line 111)
        kwargs_351134 = {}
        # Getting the type of 'assert_raises' (line 111)
        assert_raises_351126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 111)
        assert_raises_call_result_351135 = invoke(stypy.reporting.localization.Localization(__file__, 111, 8), assert_raises_351126, *[ValueError_351127, upfirdn_351128, list_351129, list_351130, int_351132, int_351133], **kwargs_351134)
        
        
        # Call to assert_raises(...): (line 112)
        # Processing the call arguments (line 112)
        # Getting the type of 'ValueError' (line 112)
        ValueError_351137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 22), 'ValueError', False)
        # Getting the type of 'upfirdn' (line 112)
        upfirdn_351138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 34), 'upfirdn', False)
        
        # Obtaining an instance of the builtin type 'list' (line 112)
        list_351139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 43), 'list')
        # Adding type elements to the builtin type 'list' instance (line 112)
        # Adding element type (line 112)
        
        # Obtaining an instance of the builtin type 'list' (line 112)
        list_351140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 44), 'list')
        # Adding type elements to the builtin type 'list' instance (line 112)
        # Adding element type (line 112)
        int_351141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 44), list_351140, int_351141)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 43), list_351139, list_351140)
        
        
        # Obtaining an instance of the builtin type 'list' (line 112)
        list_351142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 50), 'list')
        # Adding type elements to the builtin type 'list' instance (line 112)
        # Adding element type (line 112)
        int_351143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 51), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 50), list_351142, int_351143)
        
        int_351144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 55), 'int')
        int_351145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 58), 'int')
        # Processing the call keyword arguments (line 112)
        kwargs_351146 = {}
        # Getting the type of 'assert_raises' (line 112)
        assert_raises_351136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 112)
        assert_raises_call_result_351147 = invoke(stypy.reporting.localization.Localization(__file__, 112, 8), assert_raises_351136, *[ValueError_351137, upfirdn_351138, list_351139, list_351142, int_351144, int_351145], **kwargs_351146)
        
        
        # ################# End of 'test_valid_input(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_valid_input' in the type store
        # Getting the type of 'stypy_return_type' (line 109)
        stypy_return_type_351148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_351148)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_valid_input'
        return stypy_return_type_351148


    @norecursion
    def test_vs_lfilter(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_vs_lfilter'
        module_type_store = module_type_store.open_function_context('test_vs_lfilter', 114, 4, False)
        # Assigning a type to the variable 'self' (line 115)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestUpfirdn.test_vs_lfilter.__dict__.__setitem__('stypy_localization', localization)
        TestUpfirdn.test_vs_lfilter.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestUpfirdn.test_vs_lfilter.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestUpfirdn.test_vs_lfilter.__dict__.__setitem__('stypy_function_name', 'TestUpfirdn.test_vs_lfilter')
        TestUpfirdn.test_vs_lfilter.__dict__.__setitem__('stypy_param_names_list', [])
        TestUpfirdn.test_vs_lfilter.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestUpfirdn.test_vs_lfilter.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestUpfirdn.test_vs_lfilter.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestUpfirdn.test_vs_lfilter.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestUpfirdn.test_vs_lfilter.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestUpfirdn.test_vs_lfilter.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestUpfirdn.test_vs_lfilter', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_vs_lfilter', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_vs_lfilter(...)' code ##################

        
        # Assigning a Call to a Name (line 116):
        
        # Call to RandomState(...): (line 116)
        # Processing the call arguments (line 116)
        int_351152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 45), 'int')
        # Processing the call keyword arguments (line 116)
        kwargs_351153 = {}
        # Getting the type of 'np' (line 116)
        np_351149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 23), 'np', False)
        # Obtaining the member 'random' of a type (line 116)
        random_351150 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 23), np_351149, 'random')
        # Obtaining the member 'RandomState' of a type (line 116)
        RandomState_351151 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 23), random_351150, 'RandomState')
        # Calling RandomState(args, kwargs) (line 116)
        RandomState_call_result_351154 = invoke(stypy.reporting.localization.Localization(__file__, 116, 23), RandomState_351151, *[int_351152], **kwargs_351153)
        
        # Assigning a type to the variable 'random_state' (line 116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'random_state', RandomState_call_result_351154)
        
        # Assigning a Tuple to a Name (line 117):
        
        # Obtaining an instance of the builtin type 'tuple' (line 117)
        tuple_351155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 21), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 117)
        # Adding element type (line 117)
        # Getting the type of 'int' (line 117)
        int_351156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 21), tuple_351155, int_351156)
        # Adding element type (line 117)
        # Getting the type of 'np' (line 117)
        np_351157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 26), 'np')
        # Obtaining the member 'float32' of a type (line 117)
        float32_351158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 26), np_351157, 'float32')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 21), tuple_351155, float32_351158)
        # Adding element type (line 117)
        # Getting the type of 'np' (line 117)
        np_351159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 38), 'np')
        # Obtaining the member 'complex64' of a type (line 117)
        complex64_351160 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 38), np_351159, 'complex64')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 21), tuple_351155, complex64_351160)
        # Adding element type (line 117)
        # Getting the type of 'float' (line 117)
        float_351161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 52), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 21), tuple_351155, float_351161)
        # Adding element type (line 117)
        # Getting the type of 'complex' (line 117)
        complex_351162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 59), 'complex')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 21), tuple_351155, complex_351162)
        
        # Assigning a type to the variable 'try_types' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'try_types', tuple_351155)
        
        # Assigning a Num to a Name (line 118):
        int_351163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 15), 'int')
        # Assigning a type to the variable 'size' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'size', int_351163)
        
        # Assigning a List to a Name (line 119):
        
        # Obtaining an instance of the builtin type 'list' (line 119)
        list_351164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 119)
        # Adding element type (line 119)
        int_351165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 23), list_351164, int_351165)
        # Adding element type (line 119)
        int_351166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 23), list_351164, int_351166)
        # Adding element type (line 119)
        int_351167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 23), list_351164, int_351167)
        
        # Assigning a type to the variable 'down_factors' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'down_factors', list_351164)
        
        # Getting the type of 'try_types' (line 121)
        try_types_351168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 21), 'try_types')
        # Testing the type of a for loop iterable (line 121)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 121, 8), try_types_351168)
        # Getting the type of the for loop variable (line 121)
        for_loop_var_351169 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 121, 8), try_types_351168)
        # Assigning a type to the variable 'dtype' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'dtype', for_loop_var_351169)
        # SSA begins for a for statement (line 121)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 122):
        
        # Call to astype(...): (line 122)
        # Processing the call arguments (line 122)
        # Getting the type of 'dtype' (line 122)
        dtype_351176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 48), 'dtype', False)
        # Processing the call keyword arguments (line 122)
        kwargs_351177 = {}
        
        # Call to randn(...): (line 122)
        # Processing the call arguments (line 122)
        # Getting the type of 'size' (line 122)
        size_351172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 35), 'size', False)
        # Processing the call keyword arguments (line 122)
        kwargs_351173 = {}
        # Getting the type of 'random_state' (line 122)
        random_state_351170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 16), 'random_state', False)
        # Obtaining the member 'randn' of a type (line 122)
        randn_351171 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 16), random_state_351170, 'randn')
        # Calling randn(args, kwargs) (line 122)
        randn_call_result_351174 = invoke(stypy.reporting.localization.Localization(__file__, 122, 16), randn_351171, *[size_351172], **kwargs_351173)
        
        # Obtaining the member 'astype' of a type (line 122)
        astype_351175 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 16), randn_call_result_351174, 'astype')
        # Calling astype(args, kwargs) (line 122)
        astype_call_result_351178 = invoke(stypy.reporting.localization.Localization(__file__, 122, 16), astype_351175, *[dtype_351176], **kwargs_351177)
        
        # Assigning a type to the variable 'x' (line 122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 12), 'x', astype_call_result_351178)
        
        
        # Getting the type of 'dtype' (line 123)
        dtype_351179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 15), 'dtype')
        
        # Obtaining an instance of the builtin type 'tuple' (line 123)
        tuple_351180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 25), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 123)
        # Adding element type (line 123)
        # Getting the type of 'np' (line 123)
        np_351181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 25), 'np')
        # Obtaining the member 'complex64' of a type (line 123)
        complex64_351182 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 25), np_351181, 'complex64')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 25), tuple_351180, complex64_351182)
        # Adding element type (line 123)
        # Getting the type of 'np' (line 123)
        np_351183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 39), 'np')
        # Obtaining the member 'complex128' of a type (line 123)
        complex128_351184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 39), np_351183, 'complex128')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 25), tuple_351180, complex128_351184)
        
        # Applying the binary operator 'in' (line 123)
        result_contains_351185 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 15), 'in', dtype_351179, tuple_351180)
        
        # Testing the type of an if condition (line 123)
        if_condition_351186 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 123, 12), result_contains_351185)
        # Assigning a type to the variable 'if_condition_351186' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 12), 'if_condition_351186', if_condition_351186)
        # SSA begins for if statement (line 123)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'x' (line 124)
        x_351187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 16), 'x')
        complex_351188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 21), 'complex')
        
        # Call to randn(...): (line 124)
        # Processing the call arguments (line 124)
        # Getting the type of 'size' (line 124)
        size_351191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 45), 'size', False)
        # Processing the call keyword arguments (line 124)
        kwargs_351192 = {}
        # Getting the type of 'random_state' (line 124)
        random_state_351189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 26), 'random_state', False)
        # Obtaining the member 'randn' of a type (line 124)
        randn_351190 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 26), random_state_351189, 'randn')
        # Calling randn(args, kwargs) (line 124)
        randn_call_result_351193 = invoke(stypy.reporting.localization.Localization(__file__, 124, 26), randn_351190, *[size_351191], **kwargs_351192)
        
        # Applying the binary operator '*' (line 124)
        result_mul_351194 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 21), '*', complex_351188, randn_call_result_351193)
        
        # Applying the binary operator '+=' (line 124)
        result_iadd_351195 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 16), '+=', x_351187, result_mul_351194)
        # Assigning a type to the variable 'x' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 16), 'x', result_iadd_351195)
        
        # SSA join for if statement (line 123)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'down_factors' (line 126)
        down_factors_351196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 24), 'down_factors')
        # Testing the type of a for loop iterable (line 126)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 126, 12), down_factors_351196)
        # Getting the type of the for loop variable (line 126)
        for_loop_var_351197 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 126, 12), down_factors_351196)
        # Assigning a type to the variable 'down' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 12), 'down', for_loop_var_351197)
        # SSA begins for a for statement (line 126)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 127):
        
        # Call to firwin(...): (line 127)
        # Processing the call arguments (line 127)
        int_351199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 27), 'int')
        float_351200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 31), 'float')
        # Getting the type of 'down' (line 127)
        down_351201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 36), 'down', False)
        # Applying the binary operator 'div' (line 127)
        result_div_351202 = python_operator(stypy.reporting.localization.Localization(__file__, 127, 31), 'div', float_351200, down_351201)
        
        # Processing the call keyword arguments (line 127)
        str_351203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 49), 'str', 'hamming')
        keyword_351204 = str_351203
        kwargs_351205 = {'window': keyword_351204}
        # Getting the type of 'firwin' (line 127)
        firwin_351198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 20), 'firwin', False)
        # Calling firwin(args, kwargs) (line 127)
        firwin_call_result_351206 = invoke(stypy.reporting.localization.Localization(__file__, 127, 20), firwin_351198, *[int_351199, result_div_351202], **kwargs_351205)
        
        # Assigning a type to the variable 'h' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 16), 'h', firwin_call_result_351206)
        
        # Assigning a Subscript to a Name (line 128):
        
        # Obtaining the type of the subscript
        # Getting the type of 'down' (line 128)
        down_351207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 42), 'down')
        slice_351208 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 128, 21), None, None, down_351207)
        
        # Call to lfilter(...): (line 128)
        # Processing the call arguments (line 128)
        # Getting the type of 'h' (line 128)
        h_351210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 29), 'h', False)
        float_351211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 32), 'float')
        # Getting the type of 'x' (line 128)
        x_351212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 37), 'x', False)
        # Processing the call keyword arguments (line 128)
        kwargs_351213 = {}
        # Getting the type of 'lfilter' (line 128)
        lfilter_351209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 21), 'lfilter', False)
        # Calling lfilter(args, kwargs) (line 128)
        lfilter_call_result_351214 = invoke(stypy.reporting.localization.Localization(__file__, 128, 21), lfilter_351209, *[h_351210, float_351211, x_351212], **kwargs_351213)
        
        # Obtaining the member '__getitem__' of a type (line 128)
        getitem___351215 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 21), lfilter_call_result_351214, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 128)
        subscript_call_result_351216 = invoke(stypy.reporting.localization.Localization(__file__, 128, 21), getitem___351215, slice_351208)
        
        # Assigning a type to the variable 'yl' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 16), 'yl', subscript_call_result_351216)
        
        # Assigning a Call to a Name (line 129):
        
        # Call to upfirdn(...): (line 129)
        # Processing the call arguments (line 129)
        # Getting the type of 'h' (line 129)
        h_351218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 28), 'h', False)
        # Getting the type of 'x' (line 129)
        x_351219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 31), 'x', False)
        # Processing the call keyword arguments (line 129)
        int_351220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 37), 'int')
        keyword_351221 = int_351220
        # Getting the type of 'down' (line 129)
        down_351222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 45), 'down', False)
        keyword_351223 = down_351222
        kwargs_351224 = {'down': keyword_351223, 'up': keyword_351221}
        # Getting the type of 'upfirdn' (line 129)
        upfirdn_351217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 20), 'upfirdn', False)
        # Calling upfirdn(args, kwargs) (line 129)
        upfirdn_call_result_351225 = invoke(stypy.reporting.localization.Localization(__file__, 129, 20), upfirdn_351217, *[h_351218, x_351219], **kwargs_351224)
        
        # Assigning a type to the variable 'y' (line 129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 16), 'y', upfirdn_call_result_351225)
        
        # Call to assert_allclose(...): (line 130)
        # Processing the call arguments (line 130)
        # Getting the type of 'yl' (line 130)
        yl_351227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 32), 'yl', False)
        
        # Obtaining the type of the subscript
        # Getting the type of 'yl' (line 130)
        yl_351228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 39), 'yl', False)
        # Obtaining the member 'size' of a type (line 130)
        size_351229 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 39), yl_351228, 'size')
        slice_351230 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 130, 36), None, size_351229, None)
        # Getting the type of 'y' (line 130)
        y_351231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 36), 'y', False)
        # Obtaining the member '__getitem__' of a type (line 130)
        getitem___351232 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 36), y_351231, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 130)
        subscript_call_result_351233 = invoke(stypy.reporting.localization.Localization(__file__, 130, 36), getitem___351232, slice_351230)
        
        # Processing the call keyword arguments (line 130)
        float_351234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 54), 'float')
        keyword_351235 = float_351234
        float_351236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 65), 'float')
        keyword_351237 = float_351236
        kwargs_351238 = {'rtol': keyword_351237, 'atol': keyword_351235}
        # Getting the type of 'assert_allclose' (line 130)
        assert_allclose_351226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 16), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 130)
        assert_allclose_call_result_351239 = invoke(stypy.reporting.localization.Localization(__file__, 130, 16), assert_allclose_351226, *[yl_351227, subscript_call_result_351233], **kwargs_351238)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_vs_lfilter(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_vs_lfilter' in the type store
        # Getting the type of 'stypy_return_type' (line 114)
        stypy_return_type_351240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_351240)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_vs_lfilter'
        return stypy_return_type_351240


    @norecursion
    def test_vs_naive(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_vs_naive'
        module_type_store = module_type_store.open_function_context('test_vs_naive', 132, 4, False)
        # Assigning a type to the variable 'self' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestUpfirdn.test_vs_naive.__dict__.__setitem__('stypy_localization', localization)
        TestUpfirdn.test_vs_naive.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestUpfirdn.test_vs_naive.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestUpfirdn.test_vs_naive.__dict__.__setitem__('stypy_function_name', 'TestUpfirdn.test_vs_naive')
        TestUpfirdn.test_vs_naive.__dict__.__setitem__('stypy_param_names_list', [])
        TestUpfirdn.test_vs_naive.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestUpfirdn.test_vs_naive.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestUpfirdn.test_vs_naive.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestUpfirdn.test_vs_naive.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestUpfirdn.test_vs_naive.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestUpfirdn.test_vs_naive.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestUpfirdn.test_vs_naive', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_vs_naive', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_vs_naive(...)' code ##################

        
        # Assigning a List to a Name (line 133):
        
        # Obtaining an instance of the builtin type 'list' (line 133)
        list_351241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 133)
        
        # Assigning a type to the variable 'tests' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'tests', list_351241)
        
        # Assigning a Tuple to a Name (line 134):
        
        # Obtaining an instance of the builtin type 'tuple' (line 134)
        tuple_351242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 21), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 134)
        # Adding element type (line 134)
        # Getting the type of 'int' (line 134)
        int_351243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 134, 21), tuple_351242, int_351243)
        # Adding element type (line 134)
        # Getting the type of 'np' (line 134)
        np_351244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 26), 'np')
        # Obtaining the member 'float32' of a type (line 134)
        float32_351245 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 26), np_351244, 'float32')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 134, 21), tuple_351242, float32_351245)
        # Adding element type (line 134)
        # Getting the type of 'np' (line 134)
        np_351246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 38), 'np')
        # Obtaining the member 'complex64' of a type (line 134)
        complex64_351247 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 38), np_351246, 'complex64')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 134, 21), tuple_351242, complex64_351247)
        # Adding element type (line 134)
        # Getting the type of 'float' (line 134)
        float_351248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 52), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 134, 21), tuple_351242, float_351248)
        # Adding element type (line 134)
        # Getting the type of 'complex' (line 134)
        complex_351249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 59), 'complex')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 134, 21), tuple_351242, complex_351249)
        
        # Assigning a type to the variable 'try_types' (line 134)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'try_types', tuple_351242)
        
        
        # Call to product(...): (line 137)
        # Processing the call arguments (line 137)
        # Getting the type of 'try_types' (line 137)
        try_types_351251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 34), 'try_types', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 137)
        tuple_351252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 46), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 137)
        # Adding element type (line 137)
        float_351253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 46), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 46), tuple_351252, float_351253)
        # Adding element type (line 137)
        complex_351254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 50), 'complex')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 46), tuple_351252, complex_351254)
        
        # Processing the call keyword arguments (line 137)
        kwargs_351255 = {}
        # Getting the type of 'product' (line 137)
        product_351250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 26), 'product', False)
        # Calling product(args, kwargs) (line 137)
        product_call_result_351256 = invoke(stypy.reporting.localization.Localization(__file__, 137, 26), product_351250, *[try_types_351251, tuple_351252], **kwargs_351255)
        
        # Testing the type of a for loop iterable (line 137)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 137, 8), product_call_result_351256)
        # Getting the type of the for loop variable (line 137)
        for_loop_var_351257 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 137, 8), product_call_result_351256)
        # Assigning a type to the variable 'x_dtype' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'x_dtype', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 8), for_loop_var_351257))
        # Assigning a type to the variable 'h' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'h', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 8), for_loop_var_351257))
        # SSA begins for a for statement (line 137)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to append(...): (line 138)
        # Processing the call arguments (line 138)
        
        # Call to UpFIRDnCase(...): (line 138)
        # Processing the call arguments (line 138)
        int_351261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 41), 'int')
        int_351262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 44), 'int')
        # Getting the type of 'h' (line 138)
        h_351263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 47), 'h', False)
        # Getting the type of 'x_dtype' (line 138)
        x_dtype_351264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 50), 'x_dtype', False)
        # Processing the call keyword arguments (line 138)
        kwargs_351265 = {}
        # Getting the type of 'UpFIRDnCase' (line 138)
        UpFIRDnCase_351260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 29), 'UpFIRDnCase', False)
        # Calling UpFIRDnCase(args, kwargs) (line 138)
        UpFIRDnCase_call_result_351266 = invoke(stypy.reporting.localization.Localization(__file__, 138, 29), UpFIRDnCase_351260, *[int_351261, int_351262, h_351263, x_dtype_351264], **kwargs_351265)
        
        # Processing the call keyword arguments (line 138)
        kwargs_351267 = {}
        # Getting the type of 'tests' (line 138)
        tests_351258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 16), 'tests', False)
        # Obtaining the member 'append' of a type (line 138)
        append_351259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 16), tests_351258, 'append')
        # Calling append(args, kwargs) (line 138)
        append_call_result_351268 = invoke(stypy.reporting.localization.Localization(__file__, 138, 16), append_351259, *[UpFIRDnCase_call_result_351266], **kwargs_351267)
        
        
        # Call to append(...): (line 139)
        # Processing the call arguments (line 139)
        
        # Call to UpFIRDnCase(...): (line 139)
        # Processing the call arguments (line 139)
        int_351272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 41), 'int')
        int_351273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 44), 'int')
        # Getting the type of 'h' (line 139)
        h_351274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 47), 'h', False)
        # Getting the type of 'x_dtype' (line 139)
        x_dtype_351275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 50), 'x_dtype', False)
        # Processing the call keyword arguments (line 139)
        kwargs_351276 = {}
        # Getting the type of 'UpFIRDnCase' (line 139)
        UpFIRDnCase_351271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 29), 'UpFIRDnCase', False)
        # Calling UpFIRDnCase(args, kwargs) (line 139)
        UpFIRDnCase_call_result_351277 = invoke(stypy.reporting.localization.Localization(__file__, 139, 29), UpFIRDnCase_351271, *[int_351272, int_351273, h_351274, x_dtype_351275], **kwargs_351276)
        
        # Processing the call keyword arguments (line 139)
        kwargs_351278 = {}
        # Getting the type of 'tests' (line 139)
        tests_351269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 16), 'tests', False)
        # Obtaining the member 'append' of a type (line 139)
        append_351270 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 16), tests_351269, 'append')
        # Calling append(args, kwargs) (line 139)
        append_call_result_351279 = invoke(stypy.reporting.localization.Localization(__file__, 139, 16), append_351270, *[UpFIRDnCase_call_result_351277], **kwargs_351278)
        
        
        # Call to append(...): (line 140)
        # Processing the call arguments (line 140)
        
        # Call to UpFIRDnCase(...): (line 140)
        # Processing the call arguments (line 140)
        int_351283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 41), 'int')
        int_351284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 44), 'int')
        # Getting the type of 'h' (line 140)
        h_351285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 47), 'h', False)
        # Getting the type of 'x_dtype' (line 140)
        x_dtype_351286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 50), 'x_dtype', False)
        # Processing the call keyword arguments (line 140)
        kwargs_351287 = {}
        # Getting the type of 'UpFIRDnCase' (line 140)
        UpFIRDnCase_351282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 29), 'UpFIRDnCase', False)
        # Calling UpFIRDnCase(args, kwargs) (line 140)
        UpFIRDnCase_call_result_351288 = invoke(stypy.reporting.localization.Localization(__file__, 140, 29), UpFIRDnCase_351282, *[int_351283, int_351284, h_351285, x_dtype_351286], **kwargs_351287)
        
        # Processing the call keyword arguments (line 140)
        kwargs_351289 = {}
        # Getting the type of 'tests' (line 140)
        tests_351280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 16), 'tests', False)
        # Obtaining the member 'append' of a type (line 140)
        append_351281 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 16), tests_351280, 'append')
        # Calling append(args, kwargs) (line 140)
        append_call_result_351290 = invoke(stypy.reporting.localization.Localization(__file__, 140, 16), append_351281, *[UpFIRDnCase_call_result_351288], **kwargs_351289)
        
        
        # Call to append(...): (line 141)
        # Processing the call arguments (line 141)
        
        # Call to UpFIRDnCase(...): (line 141)
        # Processing the call arguments (line 141)
        int_351294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 41), 'int')
        int_351295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 44), 'int')
        # Getting the type of 'h' (line 141)
        h_351296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 47), 'h', False)
        # Getting the type of 'x_dtype' (line 141)
        x_dtype_351297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 50), 'x_dtype', False)
        # Processing the call keyword arguments (line 141)
        kwargs_351298 = {}
        # Getting the type of 'UpFIRDnCase' (line 141)
        UpFIRDnCase_351293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 29), 'UpFIRDnCase', False)
        # Calling UpFIRDnCase(args, kwargs) (line 141)
        UpFIRDnCase_call_result_351299 = invoke(stypy.reporting.localization.Localization(__file__, 141, 29), UpFIRDnCase_351293, *[int_351294, int_351295, h_351296, x_dtype_351297], **kwargs_351298)
        
        # Processing the call keyword arguments (line 141)
        kwargs_351300 = {}
        # Getting the type of 'tests' (line 141)
        tests_351291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 16), 'tests', False)
        # Obtaining the member 'append' of a type (line 141)
        append_351292 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 16), tests_351291, 'append')
        # Calling append(args, kwargs) (line 141)
        append_call_result_351301 = invoke(stypy.reporting.localization.Localization(__file__, 141, 16), append_351292, *[UpFIRDnCase_call_result_351299], **kwargs_351300)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Tuple to a Name (line 145):
        
        # Obtaining an instance of the builtin type 'tuple' (line 145)
        tuple_351302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 145)
        # Adding element type (line 145)
        int_351303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 19), tuple_351302, int_351303)
        # Adding element type (line 145)
        int_351304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 19), tuple_351302, int_351304)
        
        # Assigning a type to the variable 'factors' (line 145)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 8), 'factors', tuple_351302)
        
        # Assigning a Call to a Name (line 146):
        
        # Call to product(...): (line 146)
        # Processing the call arguments (line 146)
        # Getting the type of 'factors' (line 146)
        factors_351306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 24), 'factors', False)
        # Getting the type of 'factors' (line 146)
        factors_351307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 33), 'factors', False)
        # Getting the type of 'try_types' (line 146)
        try_types_351308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 42), 'try_types', False)
        # Getting the type of 'try_types' (line 146)
        try_types_351309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 53), 'try_types', False)
        # Processing the call keyword arguments (line 146)
        kwargs_351310 = {}
        # Getting the type of 'product' (line 146)
        product_351305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 16), 'product', False)
        # Calling product(args, kwargs) (line 146)
        product_call_result_351311 = invoke(stypy.reporting.localization.Localization(__file__, 146, 16), product_351305, *[factors_351306, factors_351307, try_types_351308, try_types_351309], **kwargs_351310)
        
        # Assigning a type to the variable 'cases' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'cases', product_call_result_351311)
        
        # Getting the type of 'cases' (line 147)
        cases_351312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 20), 'cases')
        # Testing the type of a for loop iterable (line 147)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 147, 8), cases_351312)
        # Getting the type of the for loop variable (line 147)
        for_loop_var_351313 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 147, 8), cases_351312)
        # Assigning a type to the variable 'case' (line 147)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'case', for_loop_var_351313)
        # SSA begins for a for statement (line 147)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'tests' (line 148)
        tests_351314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 12), 'tests')
        
        # Call to _random_factors(...): (line 148)
        # Getting the type of 'case' (line 148)
        case_351317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 43), 'case', False)
        # Processing the call keyword arguments (line 148)
        kwargs_351318 = {}
        # Getting the type of 'self' (line 148)
        self_351315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 21), 'self', False)
        # Obtaining the member '_random_factors' of a type (line 148)
        _random_factors_351316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 21), self_351315, '_random_factors')
        # Calling _random_factors(args, kwargs) (line 148)
        _random_factors_call_result_351319 = invoke(stypy.reporting.localization.Localization(__file__, 148, 21), _random_factors_351316, *[case_351317], **kwargs_351318)
        
        # Applying the binary operator '+=' (line 148)
        result_iadd_351320 = python_operator(stypy.reporting.localization.Localization(__file__, 148, 12), '+=', tests_351314, _random_factors_call_result_351319)
        # Assigning a type to the variable 'tests' (line 148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 12), 'tests', result_iadd_351320)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'tests' (line 150)
        tests_351321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 20), 'tests')
        # Testing the type of a for loop iterable (line 150)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 150, 8), tests_351321)
        # Getting the type of the for loop variable (line 150)
        for_loop_var_351322 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 150, 8), tests_351321)
        # Assigning a type to the variable 'test' (line 150)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 8), 'test', for_loop_var_351322)
        # SSA begins for a for statement (line 150)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to test(...): (line 151)
        # Processing the call keyword arguments (line 151)
        kwargs_351324 = {}
        # Getting the type of 'test' (line 151)
        test_351323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 12), 'test', False)
        # Calling test(args, kwargs) (line 151)
        test_call_result_351325 = invoke(stypy.reporting.localization.Localization(__file__, 151, 12), test_351323, *[], **kwargs_351324)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_vs_naive(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_vs_naive' in the type store
        # Getting the type of 'stypy_return_type' (line 132)
        stypy_return_type_351326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_351326)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_vs_naive'
        return stypy_return_type_351326


    @norecursion
    def _random_factors(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_random_factors'
        module_type_store = module_type_store.open_function_context('_random_factors', 153, 4, False)
        # Assigning a type to the variable 'self' (line 154)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestUpfirdn._random_factors.__dict__.__setitem__('stypy_localization', localization)
        TestUpfirdn._random_factors.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestUpfirdn._random_factors.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestUpfirdn._random_factors.__dict__.__setitem__('stypy_function_name', 'TestUpfirdn._random_factors')
        TestUpfirdn._random_factors.__dict__.__setitem__('stypy_param_names_list', ['p_max', 'q_max', 'h_dtype', 'x_dtype'])
        TestUpfirdn._random_factors.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestUpfirdn._random_factors.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestUpfirdn._random_factors.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestUpfirdn._random_factors.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestUpfirdn._random_factors.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestUpfirdn._random_factors.__dict__.__setitem__('stypy_declared_arg_number', 5)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestUpfirdn._random_factors', ['p_max', 'q_max', 'h_dtype', 'x_dtype'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_random_factors', localization, ['p_max', 'q_max', 'h_dtype', 'x_dtype'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_random_factors(...)' code ##################

        
        # Assigning a Num to a Name (line 154):
        int_351327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 16), 'int')
        # Assigning a type to the variable 'n_rep' (line 154)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'n_rep', int_351327)
        
        # Assigning a Num to a Name (line 155):
        int_351328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 20), 'int')
        # Assigning a type to the variable 'longest_h' (line 155)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 8), 'longest_h', int_351328)
        
        # Assigning a Call to a Name (line 156):
        
        # Call to RandomState(...): (line 156)
        # Processing the call arguments (line 156)
        int_351332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 45), 'int')
        # Processing the call keyword arguments (line 156)
        kwargs_351333 = {}
        # Getting the type of 'np' (line 156)
        np_351329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 23), 'np', False)
        # Obtaining the member 'random' of a type (line 156)
        random_351330 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 23), np_351329, 'random')
        # Obtaining the member 'RandomState' of a type (line 156)
        RandomState_351331 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 23), random_351330, 'RandomState')
        # Calling RandomState(args, kwargs) (line 156)
        RandomState_call_result_351334 = invoke(stypy.reporting.localization.Localization(__file__, 156, 23), RandomState_351331, *[int_351332], **kwargs_351333)
        
        # Assigning a type to the variable 'random_state' (line 156)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 8), 'random_state', RandomState_call_result_351334)
        
        # Assigning a List to a Name (line 157):
        
        # Obtaining an instance of the builtin type 'list' (line 157)
        list_351335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 157)
        
        # Assigning a type to the variable 'tests' (line 157)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 8), 'tests', list_351335)
        
        
        # Call to range(...): (line 159)
        # Processing the call arguments (line 159)
        # Getting the type of 'n_rep' (line 159)
        n_rep_351337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 23), 'n_rep', False)
        # Processing the call keyword arguments (line 159)
        kwargs_351338 = {}
        # Getting the type of 'range' (line 159)
        range_351336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 17), 'range', False)
        # Calling range(args, kwargs) (line 159)
        range_call_result_351339 = invoke(stypy.reporting.localization.Localization(__file__, 159, 17), range_351336, *[n_rep_351337], **kwargs_351338)
        
        # Testing the type of a for loop iterable (line 159)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 159, 8), range_call_result_351339)
        # Getting the type of the for loop variable (line 159)
        for_loop_var_351340 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 159, 8), range_call_result_351339)
        # Assigning a type to the variable '_' (line 159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), '_', for_loop_var_351340)
        # SSA begins for a for statement (line 159)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a IfExp to a Name (line 161):
        
        
        # Getting the type of 'p_max' (line 161)
        p_max_351341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 29), 'p_max')
        # Getting the type of 'q_max' (line 161)
        q_max_351342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 37), 'q_max')
        # Applying the binary operator '>' (line 161)
        result_gt_351343 = python_operator(stypy.reporting.localization.Localization(__file__, 161, 29), '>', p_max_351341, q_max_351342)
        
        # Testing the type of an if expression (line 161)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 161, 20), result_gt_351343)
        # SSA begins for if expression (line 161)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
        # Getting the type of 'q_max' (line 161)
        q_max_351344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 20), 'q_max')
        # SSA branch for the else part of an if expression (line 161)
        module_type_store.open_ssa_branch('if expression else')
        int_351345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 48), 'int')
        # SSA join for if expression (line 161)
        module_type_store = module_type_store.join_ssa_context()
        if_exp_351346 = union_type.UnionType.add(q_max_351344, int_351345)
        
        # Assigning a type to the variable 'p_add' (line 161)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 12), 'p_add', if_exp_351346)
        
        # Assigning a IfExp to a Name (line 162):
        
        
        # Getting the type of 'q_max' (line 162)
        q_max_351347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 29), 'q_max')
        # Getting the type of 'p_max' (line 162)
        p_max_351348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 37), 'p_max')
        # Applying the binary operator '>' (line 162)
        result_gt_351349 = python_operator(stypy.reporting.localization.Localization(__file__, 162, 29), '>', q_max_351347, p_max_351348)
        
        # Testing the type of an if expression (line 162)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 162, 20), result_gt_351349)
        # SSA begins for if expression (line 162)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
        # Getting the type of 'p_max' (line 162)
        p_max_351350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 20), 'p_max')
        # SSA branch for the else part of an if expression (line 162)
        module_type_store.open_ssa_branch('if expression else')
        int_351351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 48), 'int')
        # SSA join for if expression (line 162)
        module_type_store = module_type_store.join_ssa_context()
        if_exp_351352 = union_type.UnionType.add(p_max_351350, int_351351)
        
        # Assigning a type to the variable 'q_add' (line 162)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 12), 'q_add', if_exp_351352)
        
        # Assigning a BinOp to a Name (line 163):
        
        # Call to randint(...): (line 163)
        # Processing the call arguments (line 163)
        # Getting the type of 'p_max' (line 163)
        p_max_351355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 37), 'p_max', False)
        # Processing the call keyword arguments (line 163)
        kwargs_351356 = {}
        # Getting the type of 'random_state' (line 163)
        random_state_351353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 16), 'random_state', False)
        # Obtaining the member 'randint' of a type (line 163)
        randint_351354 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 16), random_state_351353, 'randint')
        # Calling randint(args, kwargs) (line 163)
        randint_call_result_351357 = invoke(stypy.reporting.localization.Localization(__file__, 163, 16), randint_351354, *[p_max_351355], **kwargs_351356)
        
        # Getting the type of 'p_add' (line 163)
        p_add_351358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 46), 'p_add')
        # Applying the binary operator '+' (line 163)
        result_add_351359 = python_operator(stypy.reporting.localization.Localization(__file__, 163, 16), '+', randint_call_result_351357, p_add_351358)
        
        # Assigning a type to the variable 'p' (line 163)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 12), 'p', result_add_351359)
        
        # Assigning a BinOp to a Name (line 164):
        
        # Call to randint(...): (line 164)
        # Processing the call arguments (line 164)
        # Getting the type of 'q_max' (line 164)
        q_max_351362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 37), 'q_max', False)
        # Processing the call keyword arguments (line 164)
        kwargs_351363 = {}
        # Getting the type of 'random_state' (line 164)
        random_state_351360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 16), 'random_state', False)
        # Obtaining the member 'randint' of a type (line 164)
        randint_351361 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 16), random_state_351360, 'randint')
        # Calling randint(args, kwargs) (line 164)
        randint_call_result_351364 = invoke(stypy.reporting.localization.Localization(__file__, 164, 16), randint_351361, *[q_max_351362], **kwargs_351363)
        
        # Getting the type of 'q_add' (line 164)
        q_add_351365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 46), 'q_add')
        # Applying the binary operator '+' (line 164)
        result_add_351366 = python_operator(stypy.reporting.localization.Localization(__file__, 164, 16), '+', randint_call_result_351364, q_add_351365)
        
        # Assigning a type to the variable 'q' (line 164)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 12), 'q', result_add_351366)
        
        # Assigning a BinOp to a Name (line 167):
        
        # Call to randint(...): (line 167)
        # Processing the call arguments (line 167)
        # Getting the type of 'longest_h' (line 167)
        longest_h_351369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 41), 'longest_h', False)
        # Processing the call keyword arguments (line 167)
        kwargs_351370 = {}
        # Getting the type of 'random_state' (line 167)
        random_state_351367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 20), 'random_state', False)
        # Obtaining the member 'randint' of a type (line 167)
        randint_351368 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 20), random_state_351367, 'randint')
        # Calling randint(args, kwargs) (line 167)
        randint_call_result_351371 = invoke(stypy.reporting.localization.Localization(__file__, 167, 20), randint_351368, *[longest_h_351369], **kwargs_351370)
        
        int_351372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 54), 'int')
        # Applying the binary operator '+' (line 167)
        result_add_351373 = python_operator(stypy.reporting.localization.Localization(__file__, 167, 20), '+', randint_call_result_351371, int_351372)
        
        # Assigning a type to the variable 'len_h' (line 167)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 12), 'len_h', result_add_351373)
        
        # Assigning a Call to a Name (line 168):
        
        # Call to atleast_1d(...): (line 168)
        # Processing the call arguments (line 168)
        
        # Call to randint(...): (line 168)
        # Processing the call arguments (line 168)
        # Getting the type of 'len_h' (line 168)
        len_h_351378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 51), 'len_h', False)
        # Processing the call keyword arguments (line 168)
        kwargs_351379 = {}
        # Getting the type of 'random_state' (line 168)
        random_state_351376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 30), 'random_state', False)
        # Obtaining the member 'randint' of a type (line 168)
        randint_351377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 30), random_state_351376, 'randint')
        # Calling randint(args, kwargs) (line 168)
        randint_call_result_351380 = invoke(stypy.reporting.localization.Localization(__file__, 168, 30), randint_351377, *[len_h_351378], **kwargs_351379)
        
        # Processing the call keyword arguments (line 168)
        kwargs_351381 = {}
        # Getting the type of 'np' (line 168)
        np_351374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 16), 'np', False)
        # Obtaining the member 'atleast_1d' of a type (line 168)
        atleast_1d_351375 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 16), np_351374, 'atleast_1d')
        # Calling atleast_1d(args, kwargs) (line 168)
        atleast_1d_call_result_351382 = invoke(stypy.reporting.localization.Localization(__file__, 168, 16), atleast_1d_351375, *[randint_call_result_351380], **kwargs_351381)
        
        # Assigning a type to the variable 'h' (line 168)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 12), 'h', atleast_1d_call_result_351382)
        
        # Assigning a Call to a Name (line 169):
        
        # Call to astype(...): (line 169)
        # Processing the call arguments (line 169)
        # Getting the type of 'h_dtype' (line 169)
        h_dtype_351385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 25), 'h_dtype', False)
        # Processing the call keyword arguments (line 169)
        kwargs_351386 = {}
        # Getting the type of 'h' (line 169)
        h_351383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 16), 'h', False)
        # Obtaining the member 'astype' of a type (line 169)
        astype_351384 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 16), h_351383, 'astype')
        # Calling astype(args, kwargs) (line 169)
        astype_call_result_351387 = invoke(stypy.reporting.localization.Localization(__file__, 169, 16), astype_351384, *[h_dtype_351385], **kwargs_351386)
        
        # Assigning a type to the variable 'h' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 12), 'h', astype_call_result_351387)
        
        
        # Getting the type of 'h_dtype' (line 170)
        h_dtype_351388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 15), 'h_dtype')
        # Getting the type of 'complex' (line 170)
        complex_351389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 26), 'complex')
        # Applying the binary operator '==' (line 170)
        result_eq_351390 = python_operator(stypy.reporting.localization.Localization(__file__, 170, 15), '==', h_dtype_351388, complex_351389)
        
        # Testing the type of an if condition (line 170)
        if_condition_351391 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 170, 12), result_eq_351390)
        # Assigning a type to the variable 'if_condition_351391' (line 170)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 12), 'if_condition_351391', if_condition_351391)
        # SSA begins for if statement (line 170)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'h' (line 171)
        h_351392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 16), 'h')
        complex_351393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 21), 'complex')
        
        # Call to randint(...): (line 171)
        # Processing the call arguments (line 171)
        # Getting the type of 'len_h' (line 171)
        len_h_351396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 47), 'len_h', False)
        # Processing the call keyword arguments (line 171)
        kwargs_351397 = {}
        # Getting the type of 'random_state' (line 171)
        random_state_351394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 26), 'random_state', False)
        # Obtaining the member 'randint' of a type (line 171)
        randint_351395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 26), random_state_351394, 'randint')
        # Calling randint(args, kwargs) (line 171)
        randint_call_result_351398 = invoke(stypy.reporting.localization.Localization(__file__, 171, 26), randint_351395, *[len_h_351396], **kwargs_351397)
        
        # Applying the binary operator '*' (line 171)
        result_mul_351399 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 21), '*', complex_351393, randint_call_result_351398)
        
        # Applying the binary operator '+=' (line 171)
        result_iadd_351400 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 16), '+=', h_351392, result_mul_351399)
        # Assigning a type to the variable 'h' (line 171)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 16), 'h', result_iadd_351400)
        
        # SSA join for if statement (line 170)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to append(...): (line 173)
        # Processing the call arguments (line 173)
        
        # Call to UpFIRDnCase(...): (line 173)
        # Processing the call arguments (line 173)
        # Getting the type of 'p' (line 173)
        p_351404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 37), 'p', False)
        # Getting the type of 'q' (line 173)
        q_351405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 40), 'q', False)
        # Getting the type of 'h' (line 173)
        h_351406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 43), 'h', False)
        # Getting the type of 'x_dtype' (line 173)
        x_dtype_351407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 46), 'x_dtype', False)
        # Processing the call keyword arguments (line 173)
        kwargs_351408 = {}
        # Getting the type of 'UpFIRDnCase' (line 173)
        UpFIRDnCase_351403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 25), 'UpFIRDnCase', False)
        # Calling UpFIRDnCase(args, kwargs) (line 173)
        UpFIRDnCase_call_result_351409 = invoke(stypy.reporting.localization.Localization(__file__, 173, 25), UpFIRDnCase_351403, *[p_351404, q_351405, h_351406, x_dtype_351407], **kwargs_351408)
        
        # Processing the call keyword arguments (line 173)
        kwargs_351410 = {}
        # Getting the type of 'tests' (line 173)
        tests_351401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 12), 'tests', False)
        # Obtaining the member 'append' of a type (line 173)
        append_351402 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 12), tests_351401, 'append')
        # Calling append(args, kwargs) (line 173)
        append_call_result_351411 = invoke(stypy.reporting.localization.Localization(__file__, 173, 12), append_351402, *[UpFIRDnCase_call_result_351409], **kwargs_351410)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'tests' (line 175)
        tests_351412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 15), 'tests')
        # Assigning a type to the variable 'stypy_return_type' (line 175)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'stypy_return_type', tests_351412)
        
        # ################# End of '_random_factors(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_random_factors' in the type store
        # Getting the type of 'stypy_return_type' (line 153)
        stypy_return_type_351413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_351413)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_random_factors'
        return stypy_return_type_351413


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 107, 0, False)
        # Assigning a type to the variable 'self' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestUpfirdn.__init__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        pass
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'TestUpfirdn' (line 107)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 0), 'TestUpfirdn', TestUpfirdn)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
