
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: import numpy as np
4: from numpy.testing import (assert_allclose, assert_equal,
5:                            assert_almost_equal, assert_array_equal,
6:                            assert_array_almost_equal)
7: 
8: from scipy.ndimage import convolve1d
9: 
10: from scipy.signal import savgol_coeffs, savgol_filter
11: from scipy.signal._savitzky_golay import _polyder
12: 
13: 
14: def check_polyder(p, m, expected):
15:     dp = _polyder(p, m)
16:     assert_array_equal(dp, expected)
17: 
18: 
19: def test_polyder():
20:     cases = [
21:         ([5], 0, [5]),
22:         ([5], 1, [0]),
23:         ([3, 2, 1], 0, [3, 2, 1]),
24:         ([3, 2, 1], 1, [6, 2]),
25:         ([3, 2, 1], 2, [6]),
26:         ([3, 2, 1], 3, [0]),
27:         ([[3, 2, 1], [5, 6, 7]], 0, [[3, 2, 1], [5, 6, 7]]),
28:         ([[3, 2, 1], [5, 6, 7]], 1, [[6, 2], [10, 6]]),
29:         ([[3, 2, 1], [5, 6, 7]], 2, [[6], [10]]),
30:         ([[3, 2, 1], [5, 6, 7]], 3, [[0], [0]]),
31:     ]
32:     for p, m, expected in cases:
33:         check_polyder(np.array(p).T, m, np.array(expected).T)
34: 
35: 
36: #--------------------------------------------------------------------
37: # savgol_coeffs tests
38: #--------------------------------------------------------------------
39: 
40: def alt_sg_coeffs(window_length, polyorder, pos):
41:     '''This is an alternative implementation of the SG coefficients.
42: 
43:     It uses numpy.polyfit and numpy.polyval.  The results should be
44:     equivalent to those of savgol_coeffs(), but this implementation
45:     is slower.
46: 
47:     window_length should be odd.
48: 
49:     '''
50:     if pos is None:
51:         pos = window_length // 2
52:     t = np.arange(window_length)
53:     unit = (t == pos).astype(int)
54:     h = np.polyval(np.polyfit(t, unit, polyorder), t)
55:     return h
56: 
57: 
58: def test_sg_coeffs_trivial():
59:     # Test a trivial case of savgol_coeffs: polyorder = window_length - 1
60:     h = savgol_coeffs(1, 0)
61:     assert_allclose(h, [1])
62: 
63:     h = savgol_coeffs(3, 2)
64:     assert_allclose(h, [0, 1, 0], atol=1e-10)
65: 
66:     h = savgol_coeffs(5, 4)
67:     assert_allclose(h, [0, 0, 1, 0, 0], atol=1e-10)
68: 
69:     h = savgol_coeffs(5, 4, pos=1)
70:     assert_allclose(h, [0, 0, 0, 1, 0], atol=1e-10)
71: 
72:     h = savgol_coeffs(5, 4, pos=1, use='dot')
73:     assert_allclose(h, [0, 1, 0, 0, 0], atol=1e-10)
74: 
75: 
76: def compare_coeffs_to_alt(window_length, order):
77:     # For the given window_length and order, compare the results
78:     # of savgol_coeffs and alt_sg_coeffs for pos from 0 to window_length - 1.
79:     # Also include pos=None.
80:     for pos in [None] + list(range(window_length)):
81:         h1 = savgol_coeffs(window_length, order, pos=pos, use='dot')
82:         h2 = alt_sg_coeffs(window_length, order, pos=pos)
83:         assert_allclose(h1, h2, atol=1e-10,
84:                         err_msg=("window_length = %d, order = %d, pos = %s" %
85:                                  (window_length, order, pos)))
86: 
87: 
88: def test_sg_coeffs_compare():
89:     # Compare savgol_coeffs() to alt_sg_coeffs().
90:     for window_length in range(1, 8, 2):
91:         for order in range(window_length):
92:             compare_coeffs_to_alt(window_length, order)
93: 
94: 
95: def test_sg_coeffs_exact():
96:     polyorder = 4
97:     window_length = 9
98:     halflen = window_length // 2
99: 
100:     x = np.linspace(0, 21, 43)
101:     delta = x[1] - x[0]
102: 
103:     # The data is a cubic polynomial.  We'll use an order 4
104:     # SG filter, so the filtered values should equal the input data
105:     # (except within half window_length of the edges).
106:     y = 0.5 * x ** 3 - x
107:     h = savgol_coeffs(window_length, polyorder)
108:     y0 = convolve1d(y, h)
109:     assert_allclose(y0[halflen:-halflen], y[halflen:-halflen])
110: 
111:     # Check the same input, but use deriv=1.  dy is the exact result.
112:     dy = 1.5 * x ** 2 - 1
113:     h = savgol_coeffs(window_length, polyorder, deriv=1, delta=delta)
114:     y1 = convolve1d(y, h)
115:     assert_allclose(y1[halflen:-halflen], dy[halflen:-halflen])
116: 
117:     # Check the same input, but use deriv=2. d2y is the exact result.
118:     d2y = 3.0 * x
119:     h = savgol_coeffs(window_length, polyorder, deriv=2, delta=delta)
120:     y2 = convolve1d(y, h)
121:     assert_allclose(y2[halflen:-halflen], d2y[halflen:-halflen])
122: 
123: 
124: def test_sg_coeffs_deriv():
125:     # The data in `x` is a sampled parabola, so using savgol_coeffs with an
126:     # order 2 or higher polynomial should give exact results.
127:     i = np.array([-2.0, 0.0, 2.0, 4.0, 6.0])
128:     x = i ** 2 / 4
129:     dx = i / 2
130:     d2x = 0.5 * np.ones_like(i)
131:     for pos in range(x.size):
132:         coeffs0 = savgol_coeffs(5, 3, pos=pos, delta=2.0, use='dot')
133:         assert_allclose(coeffs0.dot(x), x[pos], atol=1e-10)
134:         coeffs1 = savgol_coeffs(5, 3, pos=pos, delta=2.0, use='dot', deriv=1)
135:         assert_allclose(coeffs1.dot(x), dx[pos], atol=1e-10)
136:         coeffs2 = savgol_coeffs(5, 3, pos=pos, delta=2.0, use='dot', deriv=2)
137:         assert_allclose(coeffs2.dot(x), d2x[pos], atol=1e-10)
138: 
139: 
140: def test_sg_coeffs_large():
141:     # Test that for large values of window_length and polyorder the array of
142:     # coefficients returned is symmetric. The aim is to ensure that
143:     # no potential numeric overflow occurs.
144:     coeffs0 = savgol_coeffs(31, 9)
145:     assert_array_almost_equal(coeffs0, coeffs0[::-1])
146:     coeffs1 = savgol_coeffs(31, 9, deriv=1)
147:     assert_array_almost_equal(coeffs1, -coeffs1[::-1])
148: 
149: 
150: #--------------------------------------------------------------------
151: # savgol_filter tests
152: #--------------------------------------------------------------------
153: 
154: 
155: def test_sg_filter_trivial():
156:     ''' Test some trivial edge cases for savgol_filter().'''
157:     x = np.array([1.0])
158:     y = savgol_filter(x, 1, 0)
159:     assert_equal(y, [1.0])
160: 
161:     # Input is a single value.  With a window length of 3 and polyorder 1,
162:     # the value in y is from the straight-line fit of (-1,0), (0,3) and
163:     # (1, 0) at 0. This is just the average of the three values, hence 1.0.
164:     x = np.array([3.0])
165:     y = savgol_filter(x, 3, 1, mode='constant')
166:     assert_almost_equal(y, [1.0], decimal=15)
167: 
168:     x = np.array([3.0])
169:     y = savgol_filter(x, 3, 1, mode='nearest')
170:     assert_almost_equal(y, [3.0], decimal=15)
171: 
172:     x = np.array([1.0] * 3)
173:     y = savgol_filter(x, 3, 1, mode='wrap')
174:     assert_almost_equal(y, [1.0, 1.0, 1.0], decimal=15)
175: 
176: 
177: def test_sg_filter_basic():
178:     # Some basic test cases for savgol_filter().
179:     x = np.array([1.0, 2.0, 1.0])
180:     y = savgol_filter(x, 3, 1, mode='constant')
181:     assert_allclose(y, [1.0, 4.0 / 3, 1.0])
182: 
183:     y = savgol_filter(x, 3, 1, mode='mirror')
184:     assert_allclose(y, [5.0 / 3, 4.0 / 3, 5.0 / 3])
185: 
186:     y = savgol_filter(x, 3, 1, mode='wrap')
187:     assert_allclose(y, [4.0 / 3, 4.0 / 3, 4.0 / 3])
188: 
189: 
190: def test_sg_filter_2d():
191:     x = np.array([[1.0, 2.0, 1.0],
192:                   [2.0, 4.0, 2.0]])
193:     expected = np.array([[1.0, 4.0 / 3, 1.0],
194:                          [2.0, 8.0 / 3, 2.0]])
195:     y = savgol_filter(x, 3, 1, mode='constant')
196:     assert_allclose(y, expected)
197: 
198:     y = savgol_filter(x.T, 3, 1, mode='constant', axis=0)
199:     assert_allclose(y, expected.T)
200: 
201: 
202: def test_sg_filter_interp_edges():
203:     # Another test with low degree polynomial data, for which we can easily
204:     # give the exact results.  In this test, we use mode='interp', so
205:     # savgol_filter should match the exact solution for the entire data set,
206:     # including the edges.
207:     t = np.linspace(-5, 5, 21)
208:     delta = t[1] - t[0]
209:     # Polynomial test data.
210:     x = np.array([t,
211:                   3 * t ** 2,
212:                   t ** 3 - t])
213:     dx = np.array([np.ones_like(t),
214:                    6 * t,
215:                    3 * t ** 2 - 1.0])
216:     d2x = np.array([np.zeros_like(t),
217:                     6 * np.ones_like(t),
218:                     6 * t])
219: 
220:     window_length = 7
221: 
222:     y = savgol_filter(x, window_length, 3, axis=-1, mode='interp')
223:     assert_allclose(y, x, atol=1e-12)
224: 
225:     y1 = savgol_filter(x, window_length, 3, axis=-1, mode='interp',
226:                        deriv=1, delta=delta)
227:     assert_allclose(y1, dx, atol=1e-12)
228: 
229:     y2 = savgol_filter(x, window_length, 3, axis=-1, mode='interp',
230:                        deriv=2, delta=delta)
231:     assert_allclose(y2, d2x, atol=1e-12)
232: 
233:     # Transpose everything, and test again with axis=0.
234: 
235:     x = x.T
236:     dx = dx.T
237:     d2x = d2x.T
238: 
239:     y = savgol_filter(x, window_length, 3, axis=0, mode='interp')
240:     assert_allclose(y, x, atol=1e-12)
241: 
242:     y1 = savgol_filter(x, window_length, 3, axis=0, mode='interp',
243:                        deriv=1, delta=delta)
244:     assert_allclose(y1, dx, atol=1e-12)
245: 
246:     y2 = savgol_filter(x, window_length, 3, axis=0, mode='interp',
247:                        deriv=2, delta=delta)
248:     assert_allclose(y2, d2x, atol=1e-12)
249: 
250: 
251: def test_sg_filter_interp_edges_3d():
252:     # Test mode='interp' with a 3-D array.
253:     t = np.linspace(-5, 5, 21)
254:     delta = t[1] - t[0]
255:     x1 = np.array([t, -t])
256:     x2 = np.array([t ** 2, 3 * t ** 2 + 5])
257:     x3 = np.array([t ** 3, 2 * t ** 3 + t ** 2 - 0.5 * t])
258:     dx1 = np.array([np.ones_like(t), -np.ones_like(t)])
259:     dx2 = np.array([2 * t, 6 * t])
260:     dx3 = np.array([3 * t ** 2, 6 * t ** 2 + 2 * t - 0.5])
261: 
262:     # z has shape (3, 2, 21)
263:     z = np.array([x1, x2, x3])
264:     dz = np.array([dx1, dx2, dx3])
265: 
266:     y = savgol_filter(z, 7, 3, axis=-1, mode='interp', delta=delta)
267:     assert_allclose(y, z, atol=1e-10)
268: 
269:     dy = savgol_filter(z, 7, 3, axis=-1, mode='interp', deriv=1, delta=delta)
270:     assert_allclose(dy, dz, atol=1e-10)
271: 
272:     # z has shape (3, 21, 2)
273:     z = np.array([x1.T, x2.T, x3.T])
274:     dz = np.array([dx1.T, dx2.T, dx3.T])
275: 
276:     y = savgol_filter(z, 7, 3, axis=1, mode='interp', delta=delta)
277:     assert_allclose(y, z, atol=1e-10)
278: 
279:     dy = savgol_filter(z, 7, 3, axis=1, mode='interp', deriv=1, delta=delta)
280:     assert_allclose(dy, dz, atol=1e-10)
281: 
282:     # z has shape (21, 3, 2)
283:     z = z.swapaxes(0, 1).copy()
284:     dz = dz.swapaxes(0, 1).copy()
285: 
286:     y = savgol_filter(z, 7, 3, axis=0, mode='interp', delta=delta)
287:     assert_allclose(y, z, atol=1e-10)
288: 
289:     dy = savgol_filter(z, 7, 3, axis=0, mode='interp', deriv=1, delta=delta)
290:     assert_allclose(dy, dz, atol=1e-10)
291: 
292: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/tests/')
import_326460 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_326460) is not StypyTypeError):

    if (import_326460 != 'pyd_module'):
        __import__(import_326460)
        sys_modules_326461 = sys.modules[import_326460]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_326461.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_326460)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from numpy.testing import assert_allclose, assert_equal, assert_almost_equal, assert_array_equal, assert_array_almost_equal' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/tests/')
import_326462 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing')

if (type(import_326462) is not StypyTypeError):

    if (import_326462 != 'pyd_module'):
        __import__(import_326462)
        sys_modules_326463 = sys.modules[import_326462]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', sys_modules_326463.module_type_store, module_type_store, ['assert_allclose', 'assert_equal', 'assert_almost_equal', 'assert_array_equal', 'assert_array_almost_equal'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_326463, sys_modules_326463.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_allclose, assert_equal, assert_almost_equal, assert_array_equal, assert_array_almost_equal

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', None, module_type_store, ['assert_allclose', 'assert_equal', 'assert_almost_equal', 'assert_array_equal', 'assert_array_almost_equal'], [assert_allclose, assert_equal, assert_almost_equal, assert_array_equal, assert_array_almost_equal])

else:
    # Assigning a type to the variable 'numpy.testing' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', import_326462)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from scipy.ndimage import convolve1d' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/tests/')
import_326464 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.ndimage')

if (type(import_326464) is not StypyTypeError):

    if (import_326464 != 'pyd_module'):
        __import__(import_326464)
        sys_modules_326465 = sys.modules[import_326464]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.ndimage', sys_modules_326465.module_type_store, module_type_store, ['convolve1d'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_326465, sys_modules_326465.module_type_store, module_type_store)
    else:
        from scipy.ndimage import convolve1d

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.ndimage', None, module_type_store, ['convolve1d'], [convolve1d])

else:
    # Assigning a type to the variable 'scipy.ndimage' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.ndimage', import_326464)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from scipy.signal import savgol_coeffs, savgol_filter' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/tests/')
import_326466 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.signal')

if (type(import_326466) is not StypyTypeError):

    if (import_326466 != 'pyd_module'):
        __import__(import_326466)
        sys_modules_326467 = sys.modules[import_326466]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.signal', sys_modules_326467.module_type_store, module_type_store, ['savgol_coeffs', 'savgol_filter'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_326467, sys_modules_326467.module_type_store, module_type_store)
    else:
        from scipy.signal import savgol_coeffs, savgol_filter

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.signal', None, module_type_store, ['savgol_coeffs', 'savgol_filter'], [savgol_coeffs, savgol_filter])

else:
    # Assigning a type to the variable 'scipy.signal' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.signal', import_326466)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from scipy.signal._savitzky_golay import _polyder' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/tests/')
import_326468 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.signal._savitzky_golay')

if (type(import_326468) is not StypyTypeError):

    if (import_326468 != 'pyd_module'):
        __import__(import_326468)
        sys_modules_326469 = sys.modules[import_326468]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.signal._savitzky_golay', sys_modules_326469.module_type_store, module_type_store, ['_polyder'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_326469, sys_modules_326469.module_type_store, module_type_store)
    else:
        from scipy.signal._savitzky_golay import _polyder

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.signal._savitzky_golay', None, module_type_store, ['_polyder'], [_polyder])

else:
    # Assigning a type to the variable 'scipy.signal._savitzky_golay' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.signal._savitzky_golay', import_326468)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/tests/')


@norecursion
def check_polyder(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'check_polyder'
    module_type_store = module_type_store.open_function_context('check_polyder', 14, 0, False)
    
    # Passed parameters checking function
    check_polyder.stypy_localization = localization
    check_polyder.stypy_type_of_self = None
    check_polyder.stypy_type_store = module_type_store
    check_polyder.stypy_function_name = 'check_polyder'
    check_polyder.stypy_param_names_list = ['p', 'm', 'expected']
    check_polyder.stypy_varargs_param_name = None
    check_polyder.stypy_kwargs_param_name = None
    check_polyder.stypy_call_defaults = defaults
    check_polyder.stypy_call_varargs = varargs
    check_polyder.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'check_polyder', ['p', 'm', 'expected'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'check_polyder', localization, ['p', 'm', 'expected'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'check_polyder(...)' code ##################

    
    # Assigning a Call to a Name (line 15):
    
    # Call to _polyder(...): (line 15)
    # Processing the call arguments (line 15)
    # Getting the type of 'p' (line 15)
    p_326471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 18), 'p', False)
    # Getting the type of 'm' (line 15)
    m_326472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 21), 'm', False)
    # Processing the call keyword arguments (line 15)
    kwargs_326473 = {}
    # Getting the type of '_polyder' (line 15)
    _polyder_326470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 9), '_polyder', False)
    # Calling _polyder(args, kwargs) (line 15)
    _polyder_call_result_326474 = invoke(stypy.reporting.localization.Localization(__file__, 15, 9), _polyder_326470, *[p_326471, m_326472], **kwargs_326473)
    
    # Assigning a type to the variable 'dp' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'dp', _polyder_call_result_326474)
    
    # Call to assert_array_equal(...): (line 16)
    # Processing the call arguments (line 16)
    # Getting the type of 'dp' (line 16)
    dp_326476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 23), 'dp', False)
    # Getting the type of 'expected' (line 16)
    expected_326477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 27), 'expected', False)
    # Processing the call keyword arguments (line 16)
    kwargs_326478 = {}
    # Getting the type of 'assert_array_equal' (line 16)
    assert_array_equal_326475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'assert_array_equal', False)
    # Calling assert_array_equal(args, kwargs) (line 16)
    assert_array_equal_call_result_326479 = invoke(stypy.reporting.localization.Localization(__file__, 16, 4), assert_array_equal_326475, *[dp_326476, expected_326477], **kwargs_326478)
    
    
    # ################# End of 'check_polyder(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'check_polyder' in the type store
    # Getting the type of 'stypy_return_type' (line 14)
    stypy_return_type_326480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_326480)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'check_polyder'
    return stypy_return_type_326480

# Assigning a type to the variable 'check_polyder' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'check_polyder', check_polyder)

@norecursion
def test_polyder(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_polyder'
    module_type_store = module_type_store.open_function_context('test_polyder', 19, 0, False)
    
    # Passed parameters checking function
    test_polyder.stypy_localization = localization
    test_polyder.stypy_type_of_self = None
    test_polyder.stypy_type_store = module_type_store
    test_polyder.stypy_function_name = 'test_polyder'
    test_polyder.stypy_param_names_list = []
    test_polyder.stypy_varargs_param_name = None
    test_polyder.stypy_kwargs_param_name = None
    test_polyder.stypy_call_defaults = defaults
    test_polyder.stypy_call_varargs = varargs
    test_polyder.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_polyder', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_polyder', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_polyder(...)' code ##################

    
    # Assigning a List to a Name (line 20):
    
    # Obtaining an instance of the builtin type 'list' (line 20)
    list_326481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 12), 'list')
    # Adding type elements to the builtin type 'list' instance (line 20)
    # Adding element type (line 20)
    
    # Obtaining an instance of the builtin type 'tuple' (line 21)
    tuple_326482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 21)
    # Adding element type (line 21)
    
    # Obtaining an instance of the builtin type 'list' (line 21)
    list_326483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 9), 'list')
    # Adding type elements to the builtin type 'list' instance (line 21)
    # Adding element type (line 21)
    int_326484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 10), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 9), list_326483, int_326484)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 9), tuple_326482, list_326483)
    # Adding element type (line 21)
    int_326485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 14), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 9), tuple_326482, int_326485)
    # Adding element type (line 21)
    
    # Obtaining an instance of the builtin type 'list' (line 21)
    list_326486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 21)
    # Adding element type (line 21)
    int_326487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 17), list_326486, int_326487)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 9), tuple_326482, list_326486)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 12), list_326481, tuple_326482)
    # Adding element type (line 20)
    
    # Obtaining an instance of the builtin type 'tuple' (line 22)
    tuple_326488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 22)
    # Adding element type (line 22)
    
    # Obtaining an instance of the builtin type 'list' (line 22)
    list_326489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 9), 'list')
    # Adding type elements to the builtin type 'list' instance (line 22)
    # Adding element type (line 22)
    int_326490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 10), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 9), list_326489, int_326490)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 9), tuple_326488, list_326489)
    # Adding element type (line 22)
    int_326491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 14), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 9), tuple_326488, int_326491)
    # Adding element type (line 22)
    
    # Obtaining an instance of the builtin type 'list' (line 22)
    list_326492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 22)
    # Adding element type (line 22)
    int_326493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 17), list_326492, int_326493)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 9), tuple_326488, list_326492)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 12), list_326481, tuple_326488)
    # Adding element type (line 20)
    
    # Obtaining an instance of the builtin type 'tuple' (line 23)
    tuple_326494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 23)
    # Adding element type (line 23)
    
    # Obtaining an instance of the builtin type 'list' (line 23)
    list_326495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 9), 'list')
    # Adding type elements to the builtin type 'list' instance (line 23)
    # Adding element type (line 23)
    int_326496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 10), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 9), list_326495, int_326496)
    # Adding element type (line 23)
    int_326497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 13), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 9), list_326495, int_326497)
    # Adding element type (line 23)
    int_326498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 16), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 9), list_326495, int_326498)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 9), tuple_326494, list_326495)
    # Adding element type (line 23)
    int_326499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 9), tuple_326494, int_326499)
    # Adding element type (line 23)
    
    # Obtaining an instance of the builtin type 'list' (line 23)
    list_326500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 23)
    # Adding element type (line 23)
    int_326501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 23), list_326500, int_326501)
    # Adding element type (line 23)
    int_326502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 23), list_326500, int_326502)
    # Adding element type (line 23)
    int_326503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 23), list_326500, int_326503)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 9), tuple_326494, list_326500)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 12), list_326481, tuple_326494)
    # Adding element type (line 20)
    
    # Obtaining an instance of the builtin type 'tuple' (line 24)
    tuple_326504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 24)
    # Adding element type (line 24)
    
    # Obtaining an instance of the builtin type 'list' (line 24)
    list_326505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 9), 'list')
    # Adding type elements to the builtin type 'list' instance (line 24)
    # Adding element type (line 24)
    int_326506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 10), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 9), list_326505, int_326506)
    # Adding element type (line 24)
    int_326507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 13), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 9), list_326505, int_326507)
    # Adding element type (line 24)
    int_326508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 16), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 9), list_326505, int_326508)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 9), tuple_326504, list_326505)
    # Adding element type (line 24)
    int_326509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 9), tuple_326504, int_326509)
    # Adding element type (line 24)
    
    # Obtaining an instance of the builtin type 'list' (line 24)
    list_326510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 24)
    # Adding element type (line 24)
    int_326511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 23), list_326510, int_326511)
    # Adding element type (line 24)
    int_326512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 23), list_326510, int_326512)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 9), tuple_326504, list_326510)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 12), list_326481, tuple_326504)
    # Adding element type (line 20)
    
    # Obtaining an instance of the builtin type 'tuple' (line 25)
    tuple_326513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 25)
    # Adding element type (line 25)
    
    # Obtaining an instance of the builtin type 'list' (line 25)
    list_326514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 9), 'list')
    # Adding type elements to the builtin type 'list' instance (line 25)
    # Adding element type (line 25)
    int_326515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 10), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 9), list_326514, int_326515)
    # Adding element type (line 25)
    int_326516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 13), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 9), list_326514, int_326516)
    # Adding element type (line 25)
    int_326517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 16), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 9), list_326514, int_326517)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 9), tuple_326513, list_326514)
    # Adding element type (line 25)
    int_326518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 9), tuple_326513, int_326518)
    # Adding element type (line 25)
    
    # Obtaining an instance of the builtin type 'list' (line 25)
    list_326519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 25)
    # Adding element type (line 25)
    int_326520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 23), list_326519, int_326520)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 9), tuple_326513, list_326519)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 12), list_326481, tuple_326513)
    # Adding element type (line 20)
    
    # Obtaining an instance of the builtin type 'tuple' (line 26)
    tuple_326521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 26)
    # Adding element type (line 26)
    
    # Obtaining an instance of the builtin type 'list' (line 26)
    list_326522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 9), 'list')
    # Adding type elements to the builtin type 'list' instance (line 26)
    # Adding element type (line 26)
    int_326523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 10), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 9), list_326522, int_326523)
    # Adding element type (line 26)
    int_326524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 13), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 9), list_326522, int_326524)
    # Adding element type (line 26)
    int_326525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 16), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 9), list_326522, int_326525)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 9), tuple_326521, list_326522)
    # Adding element type (line 26)
    int_326526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 9), tuple_326521, int_326526)
    # Adding element type (line 26)
    
    # Obtaining an instance of the builtin type 'list' (line 26)
    list_326527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 26)
    # Adding element type (line 26)
    int_326528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 23), list_326527, int_326528)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 9), tuple_326521, list_326527)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 12), list_326481, tuple_326521)
    # Adding element type (line 20)
    
    # Obtaining an instance of the builtin type 'tuple' (line 27)
    tuple_326529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 27)
    # Adding element type (line 27)
    
    # Obtaining an instance of the builtin type 'list' (line 27)
    list_326530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 9), 'list')
    # Adding type elements to the builtin type 'list' instance (line 27)
    # Adding element type (line 27)
    
    # Obtaining an instance of the builtin type 'list' (line 27)
    list_326531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 10), 'list')
    # Adding type elements to the builtin type 'list' instance (line 27)
    # Adding element type (line 27)
    int_326532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 11), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 10), list_326531, int_326532)
    # Adding element type (line 27)
    int_326533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 14), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 10), list_326531, int_326533)
    # Adding element type (line 27)
    int_326534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 17), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 10), list_326531, int_326534)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 9), list_326530, list_326531)
    # Adding element type (line 27)
    
    # Obtaining an instance of the builtin type 'list' (line 27)
    list_326535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 21), 'list')
    # Adding type elements to the builtin type 'list' instance (line 27)
    # Adding element type (line 27)
    int_326536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 21), list_326535, int_326536)
    # Adding element type (line 27)
    int_326537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 21), list_326535, int_326537)
    # Adding element type (line 27)
    int_326538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 21), list_326535, int_326538)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 9), list_326530, list_326535)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 9), tuple_326529, list_326530)
    # Adding element type (line 27)
    int_326539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 9), tuple_326529, int_326539)
    # Adding element type (line 27)
    
    # Obtaining an instance of the builtin type 'list' (line 27)
    list_326540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 36), 'list')
    # Adding type elements to the builtin type 'list' instance (line 27)
    # Adding element type (line 27)
    
    # Obtaining an instance of the builtin type 'list' (line 27)
    list_326541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 37), 'list')
    # Adding type elements to the builtin type 'list' instance (line 27)
    # Adding element type (line 27)
    int_326542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 38), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 37), list_326541, int_326542)
    # Adding element type (line 27)
    int_326543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 41), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 37), list_326541, int_326543)
    # Adding element type (line 27)
    int_326544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 44), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 37), list_326541, int_326544)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 36), list_326540, list_326541)
    # Adding element type (line 27)
    
    # Obtaining an instance of the builtin type 'list' (line 27)
    list_326545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 48), 'list')
    # Adding type elements to the builtin type 'list' instance (line 27)
    # Adding element type (line 27)
    int_326546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 49), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 48), list_326545, int_326546)
    # Adding element type (line 27)
    int_326547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 52), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 48), list_326545, int_326547)
    # Adding element type (line 27)
    int_326548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 55), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 48), list_326545, int_326548)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 36), list_326540, list_326545)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 9), tuple_326529, list_326540)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 12), list_326481, tuple_326529)
    # Adding element type (line 20)
    
    # Obtaining an instance of the builtin type 'tuple' (line 28)
    tuple_326549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 28)
    # Adding element type (line 28)
    
    # Obtaining an instance of the builtin type 'list' (line 28)
    list_326550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 9), 'list')
    # Adding type elements to the builtin type 'list' instance (line 28)
    # Adding element type (line 28)
    
    # Obtaining an instance of the builtin type 'list' (line 28)
    list_326551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 10), 'list')
    # Adding type elements to the builtin type 'list' instance (line 28)
    # Adding element type (line 28)
    int_326552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 11), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 10), list_326551, int_326552)
    # Adding element type (line 28)
    int_326553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 14), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 10), list_326551, int_326553)
    # Adding element type (line 28)
    int_326554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 17), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 10), list_326551, int_326554)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 9), list_326550, list_326551)
    # Adding element type (line 28)
    
    # Obtaining an instance of the builtin type 'list' (line 28)
    list_326555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 21), 'list')
    # Adding type elements to the builtin type 'list' instance (line 28)
    # Adding element type (line 28)
    int_326556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 21), list_326555, int_326556)
    # Adding element type (line 28)
    int_326557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 21), list_326555, int_326557)
    # Adding element type (line 28)
    int_326558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 21), list_326555, int_326558)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 9), list_326550, list_326555)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 9), tuple_326549, list_326550)
    # Adding element type (line 28)
    int_326559 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 9), tuple_326549, int_326559)
    # Adding element type (line 28)
    
    # Obtaining an instance of the builtin type 'list' (line 28)
    list_326560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 36), 'list')
    # Adding type elements to the builtin type 'list' instance (line 28)
    # Adding element type (line 28)
    
    # Obtaining an instance of the builtin type 'list' (line 28)
    list_326561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 37), 'list')
    # Adding type elements to the builtin type 'list' instance (line 28)
    # Adding element type (line 28)
    int_326562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 38), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 37), list_326561, int_326562)
    # Adding element type (line 28)
    int_326563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 41), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 37), list_326561, int_326563)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 36), list_326560, list_326561)
    # Adding element type (line 28)
    
    # Obtaining an instance of the builtin type 'list' (line 28)
    list_326564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 45), 'list')
    # Adding type elements to the builtin type 'list' instance (line 28)
    # Adding element type (line 28)
    int_326565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 46), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 45), list_326564, int_326565)
    # Adding element type (line 28)
    int_326566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 50), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 45), list_326564, int_326566)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 36), list_326560, list_326564)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 9), tuple_326549, list_326560)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 12), list_326481, tuple_326549)
    # Adding element type (line 20)
    
    # Obtaining an instance of the builtin type 'tuple' (line 29)
    tuple_326567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 29)
    # Adding element type (line 29)
    
    # Obtaining an instance of the builtin type 'list' (line 29)
    list_326568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 9), 'list')
    # Adding type elements to the builtin type 'list' instance (line 29)
    # Adding element type (line 29)
    
    # Obtaining an instance of the builtin type 'list' (line 29)
    list_326569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 10), 'list')
    # Adding type elements to the builtin type 'list' instance (line 29)
    # Adding element type (line 29)
    int_326570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 11), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 10), list_326569, int_326570)
    # Adding element type (line 29)
    int_326571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 14), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 10), list_326569, int_326571)
    # Adding element type (line 29)
    int_326572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 17), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 10), list_326569, int_326572)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 9), list_326568, list_326569)
    # Adding element type (line 29)
    
    # Obtaining an instance of the builtin type 'list' (line 29)
    list_326573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 21), 'list')
    # Adding type elements to the builtin type 'list' instance (line 29)
    # Adding element type (line 29)
    int_326574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 21), list_326573, int_326574)
    # Adding element type (line 29)
    int_326575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 21), list_326573, int_326575)
    # Adding element type (line 29)
    int_326576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 21), list_326573, int_326576)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 9), list_326568, list_326573)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 9), tuple_326567, list_326568)
    # Adding element type (line 29)
    int_326577 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 9), tuple_326567, int_326577)
    # Adding element type (line 29)
    
    # Obtaining an instance of the builtin type 'list' (line 29)
    list_326578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 36), 'list')
    # Adding type elements to the builtin type 'list' instance (line 29)
    # Adding element type (line 29)
    
    # Obtaining an instance of the builtin type 'list' (line 29)
    list_326579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 37), 'list')
    # Adding type elements to the builtin type 'list' instance (line 29)
    # Adding element type (line 29)
    int_326580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 38), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 37), list_326579, int_326580)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 36), list_326578, list_326579)
    # Adding element type (line 29)
    
    # Obtaining an instance of the builtin type 'list' (line 29)
    list_326581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 42), 'list')
    # Adding type elements to the builtin type 'list' instance (line 29)
    # Adding element type (line 29)
    int_326582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 43), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 42), list_326581, int_326582)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 36), list_326578, list_326581)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 9), tuple_326567, list_326578)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 12), list_326481, tuple_326567)
    # Adding element type (line 20)
    
    # Obtaining an instance of the builtin type 'tuple' (line 30)
    tuple_326583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 30)
    # Adding element type (line 30)
    
    # Obtaining an instance of the builtin type 'list' (line 30)
    list_326584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 9), 'list')
    # Adding type elements to the builtin type 'list' instance (line 30)
    # Adding element type (line 30)
    
    # Obtaining an instance of the builtin type 'list' (line 30)
    list_326585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 10), 'list')
    # Adding type elements to the builtin type 'list' instance (line 30)
    # Adding element type (line 30)
    int_326586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 11), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 10), list_326585, int_326586)
    # Adding element type (line 30)
    int_326587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 14), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 10), list_326585, int_326587)
    # Adding element type (line 30)
    int_326588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 17), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 10), list_326585, int_326588)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 9), list_326584, list_326585)
    # Adding element type (line 30)
    
    # Obtaining an instance of the builtin type 'list' (line 30)
    list_326589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 21), 'list')
    # Adding type elements to the builtin type 'list' instance (line 30)
    # Adding element type (line 30)
    int_326590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 21), list_326589, int_326590)
    # Adding element type (line 30)
    int_326591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 21), list_326589, int_326591)
    # Adding element type (line 30)
    int_326592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 21), list_326589, int_326592)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 9), list_326584, list_326589)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 9), tuple_326583, list_326584)
    # Adding element type (line 30)
    int_326593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 9), tuple_326583, int_326593)
    # Adding element type (line 30)
    
    # Obtaining an instance of the builtin type 'list' (line 30)
    list_326594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 36), 'list')
    # Adding type elements to the builtin type 'list' instance (line 30)
    # Adding element type (line 30)
    
    # Obtaining an instance of the builtin type 'list' (line 30)
    list_326595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 37), 'list')
    # Adding type elements to the builtin type 'list' instance (line 30)
    # Adding element type (line 30)
    int_326596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 38), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 37), list_326595, int_326596)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 36), list_326594, list_326595)
    # Adding element type (line 30)
    
    # Obtaining an instance of the builtin type 'list' (line 30)
    list_326597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 42), 'list')
    # Adding type elements to the builtin type 'list' instance (line 30)
    # Adding element type (line 30)
    int_326598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 43), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 42), list_326597, int_326598)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 36), list_326594, list_326597)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 9), tuple_326583, list_326594)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 12), list_326481, tuple_326583)
    
    # Assigning a type to the variable 'cases' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'cases', list_326481)
    
    # Getting the type of 'cases' (line 32)
    cases_326599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 26), 'cases')
    # Testing the type of a for loop iterable (line 32)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 32, 4), cases_326599)
    # Getting the type of the for loop variable (line 32)
    for_loop_var_326600 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 32, 4), cases_326599)
    # Assigning a type to the variable 'p' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'p', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 4), for_loop_var_326600))
    # Assigning a type to the variable 'm' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'm', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 4), for_loop_var_326600))
    # Assigning a type to the variable 'expected' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'expected', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 4), for_loop_var_326600))
    # SSA begins for a for statement (line 32)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to check_polyder(...): (line 33)
    # Processing the call arguments (line 33)
    
    # Call to array(...): (line 33)
    # Processing the call arguments (line 33)
    # Getting the type of 'p' (line 33)
    p_326604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 31), 'p', False)
    # Processing the call keyword arguments (line 33)
    kwargs_326605 = {}
    # Getting the type of 'np' (line 33)
    np_326602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 22), 'np', False)
    # Obtaining the member 'array' of a type (line 33)
    array_326603 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 22), np_326602, 'array')
    # Calling array(args, kwargs) (line 33)
    array_call_result_326606 = invoke(stypy.reporting.localization.Localization(__file__, 33, 22), array_326603, *[p_326604], **kwargs_326605)
    
    # Obtaining the member 'T' of a type (line 33)
    T_326607 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 22), array_call_result_326606, 'T')
    # Getting the type of 'm' (line 33)
    m_326608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 37), 'm', False)
    
    # Call to array(...): (line 33)
    # Processing the call arguments (line 33)
    # Getting the type of 'expected' (line 33)
    expected_326611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 49), 'expected', False)
    # Processing the call keyword arguments (line 33)
    kwargs_326612 = {}
    # Getting the type of 'np' (line 33)
    np_326609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 40), 'np', False)
    # Obtaining the member 'array' of a type (line 33)
    array_326610 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 40), np_326609, 'array')
    # Calling array(args, kwargs) (line 33)
    array_call_result_326613 = invoke(stypy.reporting.localization.Localization(__file__, 33, 40), array_326610, *[expected_326611], **kwargs_326612)
    
    # Obtaining the member 'T' of a type (line 33)
    T_326614 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 40), array_call_result_326613, 'T')
    # Processing the call keyword arguments (line 33)
    kwargs_326615 = {}
    # Getting the type of 'check_polyder' (line 33)
    check_polyder_326601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'check_polyder', False)
    # Calling check_polyder(args, kwargs) (line 33)
    check_polyder_call_result_326616 = invoke(stypy.reporting.localization.Localization(__file__, 33, 8), check_polyder_326601, *[T_326607, m_326608, T_326614], **kwargs_326615)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_polyder(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_polyder' in the type store
    # Getting the type of 'stypy_return_type' (line 19)
    stypy_return_type_326617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_326617)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_polyder'
    return stypy_return_type_326617

# Assigning a type to the variable 'test_polyder' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'test_polyder', test_polyder)

@norecursion
def alt_sg_coeffs(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'alt_sg_coeffs'
    module_type_store = module_type_store.open_function_context('alt_sg_coeffs', 40, 0, False)
    
    # Passed parameters checking function
    alt_sg_coeffs.stypy_localization = localization
    alt_sg_coeffs.stypy_type_of_self = None
    alt_sg_coeffs.stypy_type_store = module_type_store
    alt_sg_coeffs.stypy_function_name = 'alt_sg_coeffs'
    alt_sg_coeffs.stypy_param_names_list = ['window_length', 'polyorder', 'pos']
    alt_sg_coeffs.stypy_varargs_param_name = None
    alt_sg_coeffs.stypy_kwargs_param_name = None
    alt_sg_coeffs.stypy_call_defaults = defaults
    alt_sg_coeffs.stypy_call_varargs = varargs
    alt_sg_coeffs.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'alt_sg_coeffs', ['window_length', 'polyorder', 'pos'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'alt_sg_coeffs', localization, ['window_length', 'polyorder', 'pos'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'alt_sg_coeffs(...)' code ##################

    str_326618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, (-1)), 'str', 'This is an alternative implementation of the SG coefficients.\n\n    It uses numpy.polyfit and numpy.polyval.  The results should be\n    equivalent to those of savgol_coeffs(), but this implementation\n    is slower.\n\n    window_length should be odd.\n\n    ')
    
    # Type idiom detected: calculating its left and rigth part (line 50)
    # Getting the type of 'pos' (line 50)
    pos_326619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 7), 'pos')
    # Getting the type of 'None' (line 50)
    None_326620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 14), 'None')
    
    (may_be_326621, more_types_in_union_326622) = may_be_none(pos_326619, None_326620)

    if may_be_326621:

        if more_types_in_union_326622:
            # Runtime conditional SSA (line 50)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a BinOp to a Name (line 51):
        # Getting the type of 'window_length' (line 51)
        window_length_326623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 14), 'window_length')
        int_326624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 31), 'int')
        # Applying the binary operator '//' (line 51)
        result_floordiv_326625 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 14), '//', window_length_326623, int_326624)
        
        # Assigning a type to the variable 'pos' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'pos', result_floordiv_326625)

        if more_types_in_union_326622:
            # SSA join for if statement (line 50)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 52):
    
    # Call to arange(...): (line 52)
    # Processing the call arguments (line 52)
    # Getting the type of 'window_length' (line 52)
    window_length_326628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 18), 'window_length', False)
    # Processing the call keyword arguments (line 52)
    kwargs_326629 = {}
    # Getting the type of 'np' (line 52)
    np_326626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'np', False)
    # Obtaining the member 'arange' of a type (line 52)
    arange_326627 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 8), np_326626, 'arange')
    # Calling arange(args, kwargs) (line 52)
    arange_call_result_326630 = invoke(stypy.reporting.localization.Localization(__file__, 52, 8), arange_326627, *[window_length_326628], **kwargs_326629)
    
    # Assigning a type to the variable 't' (line 52)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 't', arange_call_result_326630)
    
    # Assigning a Call to a Name (line 53):
    
    # Call to astype(...): (line 53)
    # Processing the call arguments (line 53)
    # Getting the type of 'int' (line 53)
    int_326635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 29), 'int', False)
    # Processing the call keyword arguments (line 53)
    kwargs_326636 = {}
    
    # Getting the type of 't' (line 53)
    t_326631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 12), 't', False)
    # Getting the type of 'pos' (line 53)
    pos_326632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 17), 'pos', False)
    # Applying the binary operator '==' (line 53)
    result_eq_326633 = python_operator(stypy.reporting.localization.Localization(__file__, 53, 12), '==', t_326631, pos_326632)
    
    # Obtaining the member 'astype' of a type (line 53)
    astype_326634 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 12), result_eq_326633, 'astype')
    # Calling astype(args, kwargs) (line 53)
    astype_call_result_326637 = invoke(stypy.reporting.localization.Localization(__file__, 53, 12), astype_326634, *[int_326635], **kwargs_326636)
    
    # Assigning a type to the variable 'unit' (line 53)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'unit', astype_call_result_326637)
    
    # Assigning a Call to a Name (line 54):
    
    # Call to polyval(...): (line 54)
    # Processing the call arguments (line 54)
    
    # Call to polyfit(...): (line 54)
    # Processing the call arguments (line 54)
    # Getting the type of 't' (line 54)
    t_326642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 30), 't', False)
    # Getting the type of 'unit' (line 54)
    unit_326643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 33), 'unit', False)
    # Getting the type of 'polyorder' (line 54)
    polyorder_326644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 39), 'polyorder', False)
    # Processing the call keyword arguments (line 54)
    kwargs_326645 = {}
    # Getting the type of 'np' (line 54)
    np_326640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 19), 'np', False)
    # Obtaining the member 'polyfit' of a type (line 54)
    polyfit_326641 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 19), np_326640, 'polyfit')
    # Calling polyfit(args, kwargs) (line 54)
    polyfit_call_result_326646 = invoke(stypy.reporting.localization.Localization(__file__, 54, 19), polyfit_326641, *[t_326642, unit_326643, polyorder_326644], **kwargs_326645)
    
    # Getting the type of 't' (line 54)
    t_326647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 51), 't', False)
    # Processing the call keyword arguments (line 54)
    kwargs_326648 = {}
    # Getting the type of 'np' (line 54)
    np_326638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'np', False)
    # Obtaining the member 'polyval' of a type (line 54)
    polyval_326639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 8), np_326638, 'polyval')
    # Calling polyval(args, kwargs) (line 54)
    polyval_call_result_326649 = invoke(stypy.reporting.localization.Localization(__file__, 54, 8), polyval_326639, *[polyfit_call_result_326646, t_326647], **kwargs_326648)
    
    # Assigning a type to the variable 'h' (line 54)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'h', polyval_call_result_326649)
    # Getting the type of 'h' (line 55)
    h_326650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 11), 'h')
    # Assigning a type to the variable 'stypy_return_type' (line 55)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'stypy_return_type', h_326650)
    
    # ################# End of 'alt_sg_coeffs(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'alt_sg_coeffs' in the type store
    # Getting the type of 'stypy_return_type' (line 40)
    stypy_return_type_326651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_326651)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'alt_sg_coeffs'
    return stypy_return_type_326651

# Assigning a type to the variable 'alt_sg_coeffs' (line 40)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 0), 'alt_sg_coeffs', alt_sg_coeffs)

@norecursion
def test_sg_coeffs_trivial(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_sg_coeffs_trivial'
    module_type_store = module_type_store.open_function_context('test_sg_coeffs_trivial', 58, 0, False)
    
    # Passed parameters checking function
    test_sg_coeffs_trivial.stypy_localization = localization
    test_sg_coeffs_trivial.stypy_type_of_self = None
    test_sg_coeffs_trivial.stypy_type_store = module_type_store
    test_sg_coeffs_trivial.stypy_function_name = 'test_sg_coeffs_trivial'
    test_sg_coeffs_trivial.stypy_param_names_list = []
    test_sg_coeffs_trivial.stypy_varargs_param_name = None
    test_sg_coeffs_trivial.stypy_kwargs_param_name = None
    test_sg_coeffs_trivial.stypy_call_defaults = defaults
    test_sg_coeffs_trivial.stypy_call_varargs = varargs
    test_sg_coeffs_trivial.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_sg_coeffs_trivial', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_sg_coeffs_trivial', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_sg_coeffs_trivial(...)' code ##################

    
    # Assigning a Call to a Name (line 60):
    
    # Call to savgol_coeffs(...): (line 60)
    # Processing the call arguments (line 60)
    int_326653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 22), 'int')
    int_326654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 25), 'int')
    # Processing the call keyword arguments (line 60)
    kwargs_326655 = {}
    # Getting the type of 'savgol_coeffs' (line 60)
    savgol_coeffs_326652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'savgol_coeffs', False)
    # Calling savgol_coeffs(args, kwargs) (line 60)
    savgol_coeffs_call_result_326656 = invoke(stypy.reporting.localization.Localization(__file__, 60, 8), savgol_coeffs_326652, *[int_326653, int_326654], **kwargs_326655)
    
    # Assigning a type to the variable 'h' (line 60)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'h', savgol_coeffs_call_result_326656)
    
    # Call to assert_allclose(...): (line 61)
    # Processing the call arguments (line 61)
    # Getting the type of 'h' (line 61)
    h_326658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 20), 'h', False)
    
    # Obtaining an instance of the builtin type 'list' (line 61)
    list_326659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 61)
    # Adding element type (line 61)
    int_326660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 23), list_326659, int_326660)
    
    # Processing the call keyword arguments (line 61)
    kwargs_326661 = {}
    # Getting the type of 'assert_allclose' (line 61)
    assert_allclose_326657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 61)
    assert_allclose_call_result_326662 = invoke(stypy.reporting.localization.Localization(__file__, 61, 4), assert_allclose_326657, *[h_326658, list_326659], **kwargs_326661)
    
    
    # Assigning a Call to a Name (line 63):
    
    # Call to savgol_coeffs(...): (line 63)
    # Processing the call arguments (line 63)
    int_326664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 22), 'int')
    int_326665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 25), 'int')
    # Processing the call keyword arguments (line 63)
    kwargs_326666 = {}
    # Getting the type of 'savgol_coeffs' (line 63)
    savgol_coeffs_326663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'savgol_coeffs', False)
    # Calling savgol_coeffs(args, kwargs) (line 63)
    savgol_coeffs_call_result_326667 = invoke(stypy.reporting.localization.Localization(__file__, 63, 8), savgol_coeffs_326663, *[int_326664, int_326665], **kwargs_326666)
    
    # Assigning a type to the variable 'h' (line 63)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'h', savgol_coeffs_call_result_326667)
    
    # Call to assert_allclose(...): (line 64)
    # Processing the call arguments (line 64)
    # Getting the type of 'h' (line 64)
    h_326669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 20), 'h', False)
    
    # Obtaining an instance of the builtin type 'list' (line 64)
    list_326670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 64)
    # Adding element type (line 64)
    int_326671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 23), list_326670, int_326671)
    # Adding element type (line 64)
    int_326672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 23), list_326670, int_326672)
    # Adding element type (line 64)
    int_326673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 23), list_326670, int_326673)
    
    # Processing the call keyword arguments (line 64)
    float_326674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 39), 'float')
    keyword_326675 = float_326674
    kwargs_326676 = {'atol': keyword_326675}
    # Getting the type of 'assert_allclose' (line 64)
    assert_allclose_326668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 64)
    assert_allclose_call_result_326677 = invoke(stypy.reporting.localization.Localization(__file__, 64, 4), assert_allclose_326668, *[h_326669, list_326670], **kwargs_326676)
    
    
    # Assigning a Call to a Name (line 66):
    
    # Call to savgol_coeffs(...): (line 66)
    # Processing the call arguments (line 66)
    int_326679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 22), 'int')
    int_326680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 25), 'int')
    # Processing the call keyword arguments (line 66)
    kwargs_326681 = {}
    # Getting the type of 'savgol_coeffs' (line 66)
    savgol_coeffs_326678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'savgol_coeffs', False)
    # Calling savgol_coeffs(args, kwargs) (line 66)
    savgol_coeffs_call_result_326682 = invoke(stypy.reporting.localization.Localization(__file__, 66, 8), savgol_coeffs_326678, *[int_326679, int_326680], **kwargs_326681)
    
    # Assigning a type to the variable 'h' (line 66)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'h', savgol_coeffs_call_result_326682)
    
    # Call to assert_allclose(...): (line 67)
    # Processing the call arguments (line 67)
    # Getting the type of 'h' (line 67)
    h_326684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 20), 'h', False)
    
    # Obtaining an instance of the builtin type 'list' (line 67)
    list_326685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 67)
    # Adding element type (line 67)
    int_326686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 23), list_326685, int_326686)
    # Adding element type (line 67)
    int_326687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 23), list_326685, int_326687)
    # Adding element type (line 67)
    int_326688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 23), list_326685, int_326688)
    # Adding element type (line 67)
    int_326689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 23), list_326685, int_326689)
    # Adding element type (line 67)
    int_326690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 36), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 23), list_326685, int_326690)
    
    # Processing the call keyword arguments (line 67)
    float_326691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 45), 'float')
    keyword_326692 = float_326691
    kwargs_326693 = {'atol': keyword_326692}
    # Getting the type of 'assert_allclose' (line 67)
    assert_allclose_326683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 67)
    assert_allclose_call_result_326694 = invoke(stypy.reporting.localization.Localization(__file__, 67, 4), assert_allclose_326683, *[h_326684, list_326685], **kwargs_326693)
    
    
    # Assigning a Call to a Name (line 69):
    
    # Call to savgol_coeffs(...): (line 69)
    # Processing the call arguments (line 69)
    int_326696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 22), 'int')
    int_326697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 25), 'int')
    # Processing the call keyword arguments (line 69)
    int_326698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 32), 'int')
    keyword_326699 = int_326698
    kwargs_326700 = {'pos': keyword_326699}
    # Getting the type of 'savgol_coeffs' (line 69)
    savgol_coeffs_326695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'savgol_coeffs', False)
    # Calling savgol_coeffs(args, kwargs) (line 69)
    savgol_coeffs_call_result_326701 = invoke(stypy.reporting.localization.Localization(__file__, 69, 8), savgol_coeffs_326695, *[int_326696, int_326697], **kwargs_326700)
    
    # Assigning a type to the variable 'h' (line 69)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'h', savgol_coeffs_call_result_326701)
    
    # Call to assert_allclose(...): (line 70)
    # Processing the call arguments (line 70)
    # Getting the type of 'h' (line 70)
    h_326703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 20), 'h', False)
    
    # Obtaining an instance of the builtin type 'list' (line 70)
    list_326704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 70)
    # Adding element type (line 70)
    int_326705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 23), list_326704, int_326705)
    # Adding element type (line 70)
    int_326706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 23), list_326704, int_326706)
    # Adding element type (line 70)
    int_326707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 23), list_326704, int_326707)
    # Adding element type (line 70)
    int_326708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 23), list_326704, int_326708)
    # Adding element type (line 70)
    int_326709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 36), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 23), list_326704, int_326709)
    
    # Processing the call keyword arguments (line 70)
    float_326710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 45), 'float')
    keyword_326711 = float_326710
    kwargs_326712 = {'atol': keyword_326711}
    # Getting the type of 'assert_allclose' (line 70)
    assert_allclose_326702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 70)
    assert_allclose_call_result_326713 = invoke(stypy.reporting.localization.Localization(__file__, 70, 4), assert_allclose_326702, *[h_326703, list_326704], **kwargs_326712)
    
    
    # Assigning a Call to a Name (line 72):
    
    # Call to savgol_coeffs(...): (line 72)
    # Processing the call arguments (line 72)
    int_326715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 22), 'int')
    int_326716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 25), 'int')
    # Processing the call keyword arguments (line 72)
    int_326717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 32), 'int')
    keyword_326718 = int_326717
    str_326719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 39), 'str', 'dot')
    keyword_326720 = str_326719
    kwargs_326721 = {'use': keyword_326720, 'pos': keyword_326718}
    # Getting the type of 'savgol_coeffs' (line 72)
    savgol_coeffs_326714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'savgol_coeffs', False)
    # Calling savgol_coeffs(args, kwargs) (line 72)
    savgol_coeffs_call_result_326722 = invoke(stypy.reporting.localization.Localization(__file__, 72, 8), savgol_coeffs_326714, *[int_326715, int_326716], **kwargs_326721)
    
    # Assigning a type to the variable 'h' (line 72)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'h', savgol_coeffs_call_result_326722)
    
    # Call to assert_allclose(...): (line 73)
    # Processing the call arguments (line 73)
    # Getting the type of 'h' (line 73)
    h_326724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 20), 'h', False)
    
    # Obtaining an instance of the builtin type 'list' (line 73)
    list_326725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 73)
    # Adding element type (line 73)
    int_326726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 23), list_326725, int_326726)
    # Adding element type (line 73)
    int_326727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 23), list_326725, int_326727)
    # Adding element type (line 73)
    int_326728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 23), list_326725, int_326728)
    # Adding element type (line 73)
    int_326729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 23), list_326725, int_326729)
    # Adding element type (line 73)
    int_326730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 36), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 23), list_326725, int_326730)
    
    # Processing the call keyword arguments (line 73)
    float_326731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 45), 'float')
    keyword_326732 = float_326731
    kwargs_326733 = {'atol': keyword_326732}
    # Getting the type of 'assert_allclose' (line 73)
    assert_allclose_326723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 73)
    assert_allclose_call_result_326734 = invoke(stypy.reporting.localization.Localization(__file__, 73, 4), assert_allclose_326723, *[h_326724, list_326725], **kwargs_326733)
    
    
    # ################# End of 'test_sg_coeffs_trivial(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_sg_coeffs_trivial' in the type store
    # Getting the type of 'stypy_return_type' (line 58)
    stypy_return_type_326735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_326735)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_sg_coeffs_trivial'
    return stypy_return_type_326735

# Assigning a type to the variable 'test_sg_coeffs_trivial' (line 58)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 0), 'test_sg_coeffs_trivial', test_sg_coeffs_trivial)

@norecursion
def compare_coeffs_to_alt(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'compare_coeffs_to_alt'
    module_type_store = module_type_store.open_function_context('compare_coeffs_to_alt', 76, 0, False)
    
    # Passed parameters checking function
    compare_coeffs_to_alt.stypy_localization = localization
    compare_coeffs_to_alt.stypy_type_of_self = None
    compare_coeffs_to_alt.stypy_type_store = module_type_store
    compare_coeffs_to_alt.stypy_function_name = 'compare_coeffs_to_alt'
    compare_coeffs_to_alt.stypy_param_names_list = ['window_length', 'order']
    compare_coeffs_to_alt.stypy_varargs_param_name = None
    compare_coeffs_to_alt.stypy_kwargs_param_name = None
    compare_coeffs_to_alt.stypy_call_defaults = defaults
    compare_coeffs_to_alt.stypy_call_varargs = varargs
    compare_coeffs_to_alt.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'compare_coeffs_to_alt', ['window_length', 'order'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'compare_coeffs_to_alt', localization, ['window_length', 'order'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'compare_coeffs_to_alt(...)' code ##################

    
    
    # Obtaining an instance of the builtin type 'list' (line 80)
    list_326736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 80)
    # Adding element type (line 80)
    # Getting the type of 'None' (line 80)
    None_326737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 16), 'None')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 15), list_326736, None_326737)
    
    
    # Call to list(...): (line 80)
    # Processing the call arguments (line 80)
    
    # Call to range(...): (line 80)
    # Processing the call arguments (line 80)
    # Getting the type of 'window_length' (line 80)
    window_length_326740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 35), 'window_length', False)
    # Processing the call keyword arguments (line 80)
    kwargs_326741 = {}
    # Getting the type of 'range' (line 80)
    range_326739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 29), 'range', False)
    # Calling range(args, kwargs) (line 80)
    range_call_result_326742 = invoke(stypy.reporting.localization.Localization(__file__, 80, 29), range_326739, *[window_length_326740], **kwargs_326741)
    
    # Processing the call keyword arguments (line 80)
    kwargs_326743 = {}
    # Getting the type of 'list' (line 80)
    list_326738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 24), 'list', False)
    # Calling list(args, kwargs) (line 80)
    list_call_result_326744 = invoke(stypy.reporting.localization.Localization(__file__, 80, 24), list_326738, *[range_call_result_326742], **kwargs_326743)
    
    # Applying the binary operator '+' (line 80)
    result_add_326745 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 15), '+', list_326736, list_call_result_326744)
    
    # Testing the type of a for loop iterable (line 80)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 80, 4), result_add_326745)
    # Getting the type of the for loop variable (line 80)
    for_loop_var_326746 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 80, 4), result_add_326745)
    # Assigning a type to the variable 'pos' (line 80)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'pos', for_loop_var_326746)
    # SSA begins for a for statement (line 80)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 81):
    
    # Call to savgol_coeffs(...): (line 81)
    # Processing the call arguments (line 81)
    # Getting the type of 'window_length' (line 81)
    window_length_326748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 27), 'window_length', False)
    # Getting the type of 'order' (line 81)
    order_326749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 42), 'order', False)
    # Processing the call keyword arguments (line 81)
    # Getting the type of 'pos' (line 81)
    pos_326750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 53), 'pos', False)
    keyword_326751 = pos_326750
    str_326752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 62), 'str', 'dot')
    keyword_326753 = str_326752
    kwargs_326754 = {'use': keyword_326753, 'pos': keyword_326751}
    # Getting the type of 'savgol_coeffs' (line 81)
    savgol_coeffs_326747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 13), 'savgol_coeffs', False)
    # Calling savgol_coeffs(args, kwargs) (line 81)
    savgol_coeffs_call_result_326755 = invoke(stypy.reporting.localization.Localization(__file__, 81, 13), savgol_coeffs_326747, *[window_length_326748, order_326749], **kwargs_326754)
    
    # Assigning a type to the variable 'h1' (line 81)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'h1', savgol_coeffs_call_result_326755)
    
    # Assigning a Call to a Name (line 82):
    
    # Call to alt_sg_coeffs(...): (line 82)
    # Processing the call arguments (line 82)
    # Getting the type of 'window_length' (line 82)
    window_length_326757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 27), 'window_length', False)
    # Getting the type of 'order' (line 82)
    order_326758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 42), 'order', False)
    # Processing the call keyword arguments (line 82)
    # Getting the type of 'pos' (line 82)
    pos_326759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 53), 'pos', False)
    keyword_326760 = pos_326759
    kwargs_326761 = {'pos': keyword_326760}
    # Getting the type of 'alt_sg_coeffs' (line 82)
    alt_sg_coeffs_326756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 13), 'alt_sg_coeffs', False)
    # Calling alt_sg_coeffs(args, kwargs) (line 82)
    alt_sg_coeffs_call_result_326762 = invoke(stypy.reporting.localization.Localization(__file__, 82, 13), alt_sg_coeffs_326756, *[window_length_326757, order_326758], **kwargs_326761)
    
    # Assigning a type to the variable 'h2' (line 82)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'h2', alt_sg_coeffs_call_result_326762)
    
    # Call to assert_allclose(...): (line 83)
    # Processing the call arguments (line 83)
    # Getting the type of 'h1' (line 83)
    h1_326764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 24), 'h1', False)
    # Getting the type of 'h2' (line 83)
    h2_326765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 28), 'h2', False)
    # Processing the call keyword arguments (line 83)
    float_326766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 37), 'float')
    keyword_326767 = float_326766
    str_326768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 33), 'str', 'window_length = %d, order = %d, pos = %s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 85)
    tuple_326769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 34), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 85)
    # Adding element type (line 85)
    # Getting the type of 'window_length' (line 85)
    window_length_326770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 34), 'window_length', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 34), tuple_326769, window_length_326770)
    # Adding element type (line 85)
    # Getting the type of 'order' (line 85)
    order_326771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 49), 'order', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 34), tuple_326769, order_326771)
    # Adding element type (line 85)
    # Getting the type of 'pos' (line 85)
    pos_326772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 56), 'pos', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 34), tuple_326769, pos_326772)
    
    # Applying the binary operator '%' (line 84)
    result_mod_326773 = python_operator(stypy.reporting.localization.Localization(__file__, 84, 33), '%', str_326768, tuple_326769)
    
    keyword_326774 = result_mod_326773
    kwargs_326775 = {'err_msg': keyword_326774, 'atol': keyword_326767}
    # Getting the type of 'assert_allclose' (line 83)
    assert_allclose_326763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 83)
    assert_allclose_call_result_326776 = invoke(stypy.reporting.localization.Localization(__file__, 83, 8), assert_allclose_326763, *[h1_326764, h2_326765], **kwargs_326775)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'compare_coeffs_to_alt(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'compare_coeffs_to_alt' in the type store
    # Getting the type of 'stypy_return_type' (line 76)
    stypy_return_type_326777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_326777)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'compare_coeffs_to_alt'
    return stypy_return_type_326777

# Assigning a type to the variable 'compare_coeffs_to_alt' (line 76)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 0), 'compare_coeffs_to_alt', compare_coeffs_to_alt)

@norecursion
def test_sg_coeffs_compare(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_sg_coeffs_compare'
    module_type_store = module_type_store.open_function_context('test_sg_coeffs_compare', 88, 0, False)
    
    # Passed parameters checking function
    test_sg_coeffs_compare.stypy_localization = localization
    test_sg_coeffs_compare.stypy_type_of_self = None
    test_sg_coeffs_compare.stypy_type_store = module_type_store
    test_sg_coeffs_compare.stypy_function_name = 'test_sg_coeffs_compare'
    test_sg_coeffs_compare.stypy_param_names_list = []
    test_sg_coeffs_compare.stypy_varargs_param_name = None
    test_sg_coeffs_compare.stypy_kwargs_param_name = None
    test_sg_coeffs_compare.stypy_call_defaults = defaults
    test_sg_coeffs_compare.stypy_call_varargs = varargs
    test_sg_coeffs_compare.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_sg_coeffs_compare', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_sg_coeffs_compare', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_sg_coeffs_compare(...)' code ##################

    
    
    # Call to range(...): (line 90)
    # Processing the call arguments (line 90)
    int_326779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 31), 'int')
    int_326780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 34), 'int')
    int_326781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 37), 'int')
    # Processing the call keyword arguments (line 90)
    kwargs_326782 = {}
    # Getting the type of 'range' (line 90)
    range_326778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 25), 'range', False)
    # Calling range(args, kwargs) (line 90)
    range_call_result_326783 = invoke(stypy.reporting.localization.Localization(__file__, 90, 25), range_326778, *[int_326779, int_326780, int_326781], **kwargs_326782)
    
    # Testing the type of a for loop iterable (line 90)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 90, 4), range_call_result_326783)
    # Getting the type of the for loop variable (line 90)
    for_loop_var_326784 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 90, 4), range_call_result_326783)
    # Assigning a type to the variable 'window_length' (line 90)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 4), 'window_length', for_loop_var_326784)
    # SSA begins for a for statement (line 90)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Call to range(...): (line 91)
    # Processing the call arguments (line 91)
    # Getting the type of 'window_length' (line 91)
    window_length_326786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 27), 'window_length', False)
    # Processing the call keyword arguments (line 91)
    kwargs_326787 = {}
    # Getting the type of 'range' (line 91)
    range_326785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 21), 'range', False)
    # Calling range(args, kwargs) (line 91)
    range_call_result_326788 = invoke(stypy.reporting.localization.Localization(__file__, 91, 21), range_326785, *[window_length_326786], **kwargs_326787)
    
    # Testing the type of a for loop iterable (line 91)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 91, 8), range_call_result_326788)
    # Getting the type of the for loop variable (line 91)
    for_loop_var_326789 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 91, 8), range_call_result_326788)
    # Assigning a type to the variable 'order' (line 91)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'order', for_loop_var_326789)
    # SSA begins for a for statement (line 91)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to compare_coeffs_to_alt(...): (line 92)
    # Processing the call arguments (line 92)
    # Getting the type of 'window_length' (line 92)
    window_length_326791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 34), 'window_length', False)
    # Getting the type of 'order' (line 92)
    order_326792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 49), 'order', False)
    # Processing the call keyword arguments (line 92)
    kwargs_326793 = {}
    # Getting the type of 'compare_coeffs_to_alt' (line 92)
    compare_coeffs_to_alt_326790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 12), 'compare_coeffs_to_alt', False)
    # Calling compare_coeffs_to_alt(args, kwargs) (line 92)
    compare_coeffs_to_alt_call_result_326794 = invoke(stypy.reporting.localization.Localization(__file__, 92, 12), compare_coeffs_to_alt_326790, *[window_length_326791, order_326792], **kwargs_326793)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_sg_coeffs_compare(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_sg_coeffs_compare' in the type store
    # Getting the type of 'stypy_return_type' (line 88)
    stypy_return_type_326795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_326795)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_sg_coeffs_compare'
    return stypy_return_type_326795

# Assigning a type to the variable 'test_sg_coeffs_compare' (line 88)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 0), 'test_sg_coeffs_compare', test_sg_coeffs_compare)

@norecursion
def test_sg_coeffs_exact(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_sg_coeffs_exact'
    module_type_store = module_type_store.open_function_context('test_sg_coeffs_exact', 95, 0, False)
    
    # Passed parameters checking function
    test_sg_coeffs_exact.stypy_localization = localization
    test_sg_coeffs_exact.stypy_type_of_self = None
    test_sg_coeffs_exact.stypy_type_store = module_type_store
    test_sg_coeffs_exact.stypy_function_name = 'test_sg_coeffs_exact'
    test_sg_coeffs_exact.stypy_param_names_list = []
    test_sg_coeffs_exact.stypy_varargs_param_name = None
    test_sg_coeffs_exact.stypy_kwargs_param_name = None
    test_sg_coeffs_exact.stypy_call_defaults = defaults
    test_sg_coeffs_exact.stypy_call_varargs = varargs
    test_sg_coeffs_exact.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_sg_coeffs_exact', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_sg_coeffs_exact', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_sg_coeffs_exact(...)' code ##################

    
    # Assigning a Num to a Name (line 96):
    int_326796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 16), 'int')
    # Assigning a type to the variable 'polyorder' (line 96)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 4), 'polyorder', int_326796)
    
    # Assigning a Num to a Name (line 97):
    int_326797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 20), 'int')
    # Assigning a type to the variable 'window_length' (line 97)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 4), 'window_length', int_326797)
    
    # Assigning a BinOp to a Name (line 98):
    # Getting the type of 'window_length' (line 98)
    window_length_326798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 14), 'window_length')
    int_326799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 31), 'int')
    # Applying the binary operator '//' (line 98)
    result_floordiv_326800 = python_operator(stypy.reporting.localization.Localization(__file__, 98, 14), '//', window_length_326798, int_326799)
    
    # Assigning a type to the variable 'halflen' (line 98)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'halflen', result_floordiv_326800)
    
    # Assigning a Call to a Name (line 100):
    
    # Call to linspace(...): (line 100)
    # Processing the call arguments (line 100)
    int_326803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 20), 'int')
    int_326804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 23), 'int')
    int_326805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 27), 'int')
    # Processing the call keyword arguments (line 100)
    kwargs_326806 = {}
    # Getting the type of 'np' (line 100)
    np_326801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'np', False)
    # Obtaining the member 'linspace' of a type (line 100)
    linspace_326802 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 8), np_326801, 'linspace')
    # Calling linspace(args, kwargs) (line 100)
    linspace_call_result_326807 = invoke(stypy.reporting.localization.Localization(__file__, 100, 8), linspace_326802, *[int_326803, int_326804, int_326805], **kwargs_326806)
    
    # Assigning a type to the variable 'x' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'x', linspace_call_result_326807)
    
    # Assigning a BinOp to a Name (line 101):
    
    # Obtaining the type of the subscript
    int_326808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 14), 'int')
    # Getting the type of 'x' (line 101)
    x_326809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 12), 'x')
    # Obtaining the member '__getitem__' of a type (line 101)
    getitem___326810 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 12), x_326809, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 101)
    subscript_call_result_326811 = invoke(stypy.reporting.localization.Localization(__file__, 101, 12), getitem___326810, int_326808)
    
    
    # Obtaining the type of the subscript
    int_326812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 21), 'int')
    # Getting the type of 'x' (line 101)
    x_326813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 19), 'x')
    # Obtaining the member '__getitem__' of a type (line 101)
    getitem___326814 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 19), x_326813, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 101)
    subscript_call_result_326815 = invoke(stypy.reporting.localization.Localization(__file__, 101, 19), getitem___326814, int_326812)
    
    # Applying the binary operator '-' (line 101)
    result_sub_326816 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 12), '-', subscript_call_result_326811, subscript_call_result_326815)
    
    # Assigning a type to the variable 'delta' (line 101)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'delta', result_sub_326816)
    
    # Assigning a BinOp to a Name (line 106):
    float_326817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 8), 'float')
    # Getting the type of 'x' (line 106)
    x_326818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 14), 'x')
    int_326819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 19), 'int')
    # Applying the binary operator '**' (line 106)
    result_pow_326820 = python_operator(stypy.reporting.localization.Localization(__file__, 106, 14), '**', x_326818, int_326819)
    
    # Applying the binary operator '*' (line 106)
    result_mul_326821 = python_operator(stypy.reporting.localization.Localization(__file__, 106, 8), '*', float_326817, result_pow_326820)
    
    # Getting the type of 'x' (line 106)
    x_326822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 23), 'x')
    # Applying the binary operator '-' (line 106)
    result_sub_326823 = python_operator(stypy.reporting.localization.Localization(__file__, 106, 8), '-', result_mul_326821, x_326822)
    
    # Assigning a type to the variable 'y' (line 106)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 4), 'y', result_sub_326823)
    
    # Assigning a Call to a Name (line 107):
    
    # Call to savgol_coeffs(...): (line 107)
    # Processing the call arguments (line 107)
    # Getting the type of 'window_length' (line 107)
    window_length_326825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 22), 'window_length', False)
    # Getting the type of 'polyorder' (line 107)
    polyorder_326826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 37), 'polyorder', False)
    # Processing the call keyword arguments (line 107)
    kwargs_326827 = {}
    # Getting the type of 'savgol_coeffs' (line 107)
    savgol_coeffs_326824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 8), 'savgol_coeffs', False)
    # Calling savgol_coeffs(args, kwargs) (line 107)
    savgol_coeffs_call_result_326828 = invoke(stypy.reporting.localization.Localization(__file__, 107, 8), savgol_coeffs_326824, *[window_length_326825, polyorder_326826], **kwargs_326827)
    
    # Assigning a type to the variable 'h' (line 107)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 4), 'h', savgol_coeffs_call_result_326828)
    
    # Assigning a Call to a Name (line 108):
    
    # Call to convolve1d(...): (line 108)
    # Processing the call arguments (line 108)
    # Getting the type of 'y' (line 108)
    y_326830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 20), 'y', False)
    # Getting the type of 'h' (line 108)
    h_326831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 23), 'h', False)
    # Processing the call keyword arguments (line 108)
    kwargs_326832 = {}
    # Getting the type of 'convolve1d' (line 108)
    convolve1d_326829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 9), 'convolve1d', False)
    # Calling convolve1d(args, kwargs) (line 108)
    convolve1d_call_result_326833 = invoke(stypy.reporting.localization.Localization(__file__, 108, 9), convolve1d_326829, *[y_326830, h_326831], **kwargs_326832)
    
    # Assigning a type to the variable 'y0' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'y0', convolve1d_call_result_326833)
    
    # Call to assert_allclose(...): (line 109)
    # Processing the call arguments (line 109)
    
    # Obtaining the type of the subscript
    # Getting the type of 'halflen' (line 109)
    halflen_326835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 23), 'halflen', False)
    
    # Getting the type of 'halflen' (line 109)
    halflen_326836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 32), 'halflen', False)
    # Applying the 'usub' unary operator (line 109)
    result___neg___326837 = python_operator(stypy.reporting.localization.Localization(__file__, 109, 31), 'usub', halflen_326836)
    
    slice_326838 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 109, 20), halflen_326835, result___neg___326837, None)
    # Getting the type of 'y0' (line 109)
    y0_326839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 20), 'y0', False)
    # Obtaining the member '__getitem__' of a type (line 109)
    getitem___326840 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 20), y0_326839, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 109)
    subscript_call_result_326841 = invoke(stypy.reporting.localization.Localization(__file__, 109, 20), getitem___326840, slice_326838)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'halflen' (line 109)
    halflen_326842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 44), 'halflen', False)
    
    # Getting the type of 'halflen' (line 109)
    halflen_326843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 53), 'halflen', False)
    # Applying the 'usub' unary operator (line 109)
    result___neg___326844 = python_operator(stypy.reporting.localization.Localization(__file__, 109, 52), 'usub', halflen_326843)
    
    slice_326845 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 109, 42), halflen_326842, result___neg___326844, None)
    # Getting the type of 'y' (line 109)
    y_326846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 42), 'y', False)
    # Obtaining the member '__getitem__' of a type (line 109)
    getitem___326847 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 42), y_326846, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 109)
    subscript_call_result_326848 = invoke(stypy.reporting.localization.Localization(__file__, 109, 42), getitem___326847, slice_326845)
    
    # Processing the call keyword arguments (line 109)
    kwargs_326849 = {}
    # Getting the type of 'assert_allclose' (line 109)
    assert_allclose_326834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 109)
    assert_allclose_call_result_326850 = invoke(stypy.reporting.localization.Localization(__file__, 109, 4), assert_allclose_326834, *[subscript_call_result_326841, subscript_call_result_326848], **kwargs_326849)
    
    
    # Assigning a BinOp to a Name (line 112):
    float_326851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 9), 'float')
    # Getting the type of 'x' (line 112)
    x_326852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 15), 'x')
    int_326853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 20), 'int')
    # Applying the binary operator '**' (line 112)
    result_pow_326854 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 15), '**', x_326852, int_326853)
    
    # Applying the binary operator '*' (line 112)
    result_mul_326855 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 9), '*', float_326851, result_pow_326854)
    
    int_326856 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 24), 'int')
    # Applying the binary operator '-' (line 112)
    result_sub_326857 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 9), '-', result_mul_326855, int_326856)
    
    # Assigning a type to the variable 'dy' (line 112)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'dy', result_sub_326857)
    
    # Assigning a Call to a Name (line 113):
    
    # Call to savgol_coeffs(...): (line 113)
    # Processing the call arguments (line 113)
    # Getting the type of 'window_length' (line 113)
    window_length_326859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 22), 'window_length', False)
    # Getting the type of 'polyorder' (line 113)
    polyorder_326860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 37), 'polyorder', False)
    # Processing the call keyword arguments (line 113)
    int_326861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 54), 'int')
    keyword_326862 = int_326861
    # Getting the type of 'delta' (line 113)
    delta_326863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 63), 'delta', False)
    keyword_326864 = delta_326863
    kwargs_326865 = {'deriv': keyword_326862, 'delta': keyword_326864}
    # Getting the type of 'savgol_coeffs' (line 113)
    savgol_coeffs_326858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'savgol_coeffs', False)
    # Calling savgol_coeffs(args, kwargs) (line 113)
    savgol_coeffs_call_result_326866 = invoke(stypy.reporting.localization.Localization(__file__, 113, 8), savgol_coeffs_326858, *[window_length_326859, polyorder_326860], **kwargs_326865)
    
    # Assigning a type to the variable 'h' (line 113)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 4), 'h', savgol_coeffs_call_result_326866)
    
    # Assigning a Call to a Name (line 114):
    
    # Call to convolve1d(...): (line 114)
    # Processing the call arguments (line 114)
    # Getting the type of 'y' (line 114)
    y_326868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 20), 'y', False)
    # Getting the type of 'h' (line 114)
    h_326869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 23), 'h', False)
    # Processing the call keyword arguments (line 114)
    kwargs_326870 = {}
    # Getting the type of 'convolve1d' (line 114)
    convolve1d_326867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 9), 'convolve1d', False)
    # Calling convolve1d(args, kwargs) (line 114)
    convolve1d_call_result_326871 = invoke(stypy.reporting.localization.Localization(__file__, 114, 9), convolve1d_326867, *[y_326868, h_326869], **kwargs_326870)
    
    # Assigning a type to the variable 'y1' (line 114)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 4), 'y1', convolve1d_call_result_326871)
    
    # Call to assert_allclose(...): (line 115)
    # Processing the call arguments (line 115)
    
    # Obtaining the type of the subscript
    # Getting the type of 'halflen' (line 115)
    halflen_326873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 23), 'halflen', False)
    
    # Getting the type of 'halflen' (line 115)
    halflen_326874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 32), 'halflen', False)
    # Applying the 'usub' unary operator (line 115)
    result___neg___326875 = python_operator(stypy.reporting.localization.Localization(__file__, 115, 31), 'usub', halflen_326874)
    
    slice_326876 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 115, 20), halflen_326873, result___neg___326875, None)
    # Getting the type of 'y1' (line 115)
    y1_326877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 20), 'y1', False)
    # Obtaining the member '__getitem__' of a type (line 115)
    getitem___326878 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 20), y1_326877, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 115)
    subscript_call_result_326879 = invoke(stypy.reporting.localization.Localization(__file__, 115, 20), getitem___326878, slice_326876)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'halflen' (line 115)
    halflen_326880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 45), 'halflen', False)
    
    # Getting the type of 'halflen' (line 115)
    halflen_326881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 54), 'halflen', False)
    # Applying the 'usub' unary operator (line 115)
    result___neg___326882 = python_operator(stypy.reporting.localization.Localization(__file__, 115, 53), 'usub', halflen_326881)
    
    slice_326883 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 115, 42), halflen_326880, result___neg___326882, None)
    # Getting the type of 'dy' (line 115)
    dy_326884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 42), 'dy', False)
    # Obtaining the member '__getitem__' of a type (line 115)
    getitem___326885 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 42), dy_326884, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 115)
    subscript_call_result_326886 = invoke(stypy.reporting.localization.Localization(__file__, 115, 42), getitem___326885, slice_326883)
    
    # Processing the call keyword arguments (line 115)
    kwargs_326887 = {}
    # Getting the type of 'assert_allclose' (line 115)
    assert_allclose_326872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 115)
    assert_allclose_call_result_326888 = invoke(stypy.reporting.localization.Localization(__file__, 115, 4), assert_allclose_326872, *[subscript_call_result_326879, subscript_call_result_326886], **kwargs_326887)
    
    
    # Assigning a BinOp to a Name (line 118):
    float_326889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 10), 'float')
    # Getting the type of 'x' (line 118)
    x_326890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 16), 'x')
    # Applying the binary operator '*' (line 118)
    result_mul_326891 = python_operator(stypy.reporting.localization.Localization(__file__, 118, 10), '*', float_326889, x_326890)
    
    # Assigning a type to the variable 'd2y' (line 118)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 4), 'd2y', result_mul_326891)
    
    # Assigning a Call to a Name (line 119):
    
    # Call to savgol_coeffs(...): (line 119)
    # Processing the call arguments (line 119)
    # Getting the type of 'window_length' (line 119)
    window_length_326893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 22), 'window_length', False)
    # Getting the type of 'polyorder' (line 119)
    polyorder_326894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 37), 'polyorder', False)
    # Processing the call keyword arguments (line 119)
    int_326895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 54), 'int')
    keyword_326896 = int_326895
    # Getting the type of 'delta' (line 119)
    delta_326897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 63), 'delta', False)
    keyword_326898 = delta_326897
    kwargs_326899 = {'deriv': keyword_326896, 'delta': keyword_326898}
    # Getting the type of 'savgol_coeffs' (line 119)
    savgol_coeffs_326892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'savgol_coeffs', False)
    # Calling savgol_coeffs(args, kwargs) (line 119)
    savgol_coeffs_call_result_326900 = invoke(stypy.reporting.localization.Localization(__file__, 119, 8), savgol_coeffs_326892, *[window_length_326893, polyorder_326894], **kwargs_326899)
    
    # Assigning a type to the variable 'h' (line 119)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 4), 'h', savgol_coeffs_call_result_326900)
    
    # Assigning a Call to a Name (line 120):
    
    # Call to convolve1d(...): (line 120)
    # Processing the call arguments (line 120)
    # Getting the type of 'y' (line 120)
    y_326902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 20), 'y', False)
    # Getting the type of 'h' (line 120)
    h_326903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 23), 'h', False)
    # Processing the call keyword arguments (line 120)
    kwargs_326904 = {}
    # Getting the type of 'convolve1d' (line 120)
    convolve1d_326901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 9), 'convolve1d', False)
    # Calling convolve1d(args, kwargs) (line 120)
    convolve1d_call_result_326905 = invoke(stypy.reporting.localization.Localization(__file__, 120, 9), convolve1d_326901, *[y_326902, h_326903], **kwargs_326904)
    
    # Assigning a type to the variable 'y2' (line 120)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 4), 'y2', convolve1d_call_result_326905)
    
    # Call to assert_allclose(...): (line 121)
    # Processing the call arguments (line 121)
    
    # Obtaining the type of the subscript
    # Getting the type of 'halflen' (line 121)
    halflen_326907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 23), 'halflen', False)
    
    # Getting the type of 'halflen' (line 121)
    halflen_326908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 32), 'halflen', False)
    # Applying the 'usub' unary operator (line 121)
    result___neg___326909 = python_operator(stypy.reporting.localization.Localization(__file__, 121, 31), 'usub', halflen_326908)
    
    slice_326910 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 121, 20), halflen_326907, result___neg___326909, None)
    # Getting the type of 'y2' (line 121)
    y2_326911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 20), 'y2', False)
    # Obtaining the member '__getitem__' of a type (line 121)
    getitem___326912 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 20), y2_326911, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 121)
    subscript_call_result_326913 = invoke(stypy.reporting.localization.Localization(__file__, 121, 20), getitem___326912, slice_326910)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'halflen' (line 121)
    halflen_326914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 46), 'halflen', False)
    
    # Getting the type of 'halflen' (line 121)
    halflen_326915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 55), 'halflen', False)
    # Applying the 'usub' unary operator (line 121)
    result___neg___326916 = python_operator(stypy.reporting.localization.Localization(__file__, 121, 54), 'usub', halflen_326915)
    
    slice_326917 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 121, 42), halflen_326914, result___neg___326916, None)
    # Getting the type of 'd2y' (line 121)
    d2y_326918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 42), 'd2y', False)
    # Obtaining the member '__getitem__' of a type (line 121)
    getitem___326919 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 42), d2y_326918, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 121)
    subscript_call_result_326920 = invoke(stypy.reporting.localization.Localization(__file__, 121, 42), getitem___326919, slice_326917)
    
    # Processing the call keyword arguments (line 121)
    kwargs_326921 = {}
    # Getting the type of 'assert_allclose' (line 121)
    assert_allclose_326906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 121)
    assert_allclose_call_result_326922 = invoke(stypy.reporting.localization.Localization(__file__, 121, 4), assert_allclose_326906, *[subscript_call_result_326913, subscript_call_result_326920], **kwargs_326921)
    
    
    # ################# End of 'test_sg_coeffs_exact(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_sg_coeffs_exact' in the type store
    # Getting the type of 'stypy_return_type' (line 95)
    stypy_return_type_326923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_326923)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_sg_coeffs_exact'
    return stypy_return_type_326923

# Assigning a type to the variable 'test_sg_coeffs_exact' (line 95)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 0), 'test_sg_coeffs_exact', test_sg_coeffs_exact)

@norecursion
def test_sg_coeffs_deriv(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_sg_coeffs_deriv'
    module_type_store = module_type_store.open_function_context('test_sg_coeffs_deriv', 124, 0, False)
    
    # Passed parameters checking function
    test_sg_coeffs_deriv.stypy_localization = localization
    test_sg_coeffs_deriv.stypy_type_of_self = None
    test_sg_coeffs_deriv.stypy_type_store = module_type_store
    test_sg_coeffs_deriv.stypy_function_name = 'test_sg_coeffs_deriv'
    test_sg_coeffs_deriv.stypy_param_names_list = []
    test_sg_coeffs_deriv.stypy_varargs_param_name = None
    test_sg_coeffs_deriv.stypy_kwargs_param_name = None
    test_sg_coeffs_deriv.stypy_call_defaults = defaults
    test_sg_coeffs_deriv.stypy_call_varargs = varargs
    test_sg_coeffs_deriv.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_sg_coeffs_deriv', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_sg_coeffs_deriv', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_sg_coeffs_deriv(...)' code ##################

    
    # Assigning a Call to a Name (line 127):
    
    # Call to array(...): (line 127)
    # Processing the call arguments (line 127)
    
    # Obtaining an instance of the builtin type 'list' (line 127)
    list_326926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 127)
    # Adding element type (line 127)
    float_326927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 18), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 17), list_326926, float_326927)
    # Adding element type (line 127)
    float_326928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 24), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 17), list_326926, float_326928)
    # Adding element type (line 127)
    float_326929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 29), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 17), list_326926, float_326929)
    # Adding element type (line 127)
    float_326930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 34), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 17), list_326926, float_326930)
    # Adding element type (line 127)
    float_326931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 39), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 17), list_326926, float_326931)
    
    # Processing the call keyword arguments (line 127)
    kwargs_326932 = {}
    # Getting the type of 'np' (line 127)
    np_326924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'np', False)
    # Obtaining the member 'array' of a type (line 127)
    array_326925 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 8), np_326924, 'array')
    # Calling array(args, kwargs) (line 127)
    array_call_result_326933 = invoke(stypy.reporting.localization.Localization(__file__, 127, 8), array_326925, *[list_326926], **kwargs_326932)
    
    # Assigning a type to the variable 'i' (line 127)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 4), 'i', array_call_result_326933)
    
    # Assigning a BinOp to a Name (line 128):
    # Getting the type of 'i' (line 128)
    i_326934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'i')
    int_326935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 13), 'int')
    # Applying the binary operator '**' (line 128)
    result_pow_326936 = python_operator(stypy.reporting.localization.Localization(__file__, 128, 8), '**', i_326934, int_326935)
    
    int_326937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 17), 'int')
    # Applying the binary operator 'div' (line 128)
    result_div_326938 = python_operator(stypy.reporting.localization.Localization(__file__, 128, 8), 'div', result_pow_326936, int_326937)
    
    # Assigning a type to the variable 'x' (line 128)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 4), 'x', result_div_326938)
    
    # Assigning a BinOp to a Name (line 129):
    # Getting the type of 'i' (line 129)
    i_326939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 9), 'i')
    int_326940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 13), 'int')
    # Applying the binary operator 'div' (line 129)
    result_div_326941 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 9), 'div', i_326939, int_326940)
    
    # Assigning a type to the variable 'dx' (line 129)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 4), 'dx', result_div_326941)
    
    # Assigning a BinOp to a Name (line 130):
    float_326942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 10), 'float')
    
    # Call to ones_like(...): (line 130)
    # Processing the call arguments (line 130)
    # Getting the type of 'i' (line 130)
    i_326945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 29), 'i', False)
    # Processing the call keyword arguments (line 130)
    kwargs_326946 = {}
    # Getting the type of 'np' (line 130)
    np_326943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 16), 'np', False)
    # Obtaining the member 'ones_like' of a type (line 130)
    ones_like_326944 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 16), np_326943, 'ones_like')
    # Calling ones_like(args, kwargs) (line 130)
    ones_like_call_result_326947 = invoke(stypy.reporting.localization.Localization(__file__, 130, 16), ones_like_326944, *[i_326945], **kwargs_326946)
    
    # Applying the binary operator '*' (line 130)
    result_mul_326948 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 10), '*', float_326942, ones_like_call_result_326947)
    
    # Assigning a type to the variable 'd2x' (line 130)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 4), 'd2x', result_mul_326948)
    
    
    # Call to range(...): (line 131)
    # Processing the call arguments (line 131)
    # Getting the type of 'x' (line 131)
    x_326950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 21), 'x', False)
    # Obtaining the member 'size' of a type (line 131)
    size_326951 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 21), x_326950, 'size')
    # Processing the call keyword arguments (line 131)
    kwargs_326952 = {}
    # Getting the type of 'range' (line 131)
    range_326949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 15), 'range', False)
    # Calling range(args, kwargs) (line 131)
    range_call_result_326953 = invoke(stypy.reporting.localization.Localization(__file__, 131, 15), range_326949, *[size_326951], **kwargs_326952)
    
    # Testing the type of a for loop iterable (line 131)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 131, 4), range_call_result_326953)
    # Getting the type of the for loop variable (line 131)
    for_loop_var_326954 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 131, 4), range_call_result_326953)
    # Assigning a type to the variable 'pos' (line 131)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 4), 'pos', for_loop_var_326954)
    # SSA begins for a for statement (line 131)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 132):
    
    # Call to savgol_coeffs(...): (line 132)
    # Processing the call arguments (line 132)
    int_326956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 32), 'int')
    int_326957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 35), 'int')
    # Processing the call keyword arguments (line 132)
    # Getting the type of 'pos' (line 132)
    pos_326958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 42), 'pos', False)
    keyword_326959 = pos_326958
    float_326960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 53), 'float')
    keyword_326961 = float_326960
    str_326962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 62), 'str', 'dot')
    keyword_326963 = str_326962
    kwargs_326964 = {'use': keyword_326963, 'pos': keyword_326959, 'delta': keyword_326961}
    # Getting the type of 'savgol_coeffs' (line 132)
    savgol_coeffs_326955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 18), 'savgol_coeffs', False)
    # Calling savgol_coeffs(args, kwargs) (line 132)
    savgol_coeffs_call_result_326965 = invoke(stypy.reporting.localization.Localization(__file__, 132, 18), savgol_coeffs_326955, *[int_326956, int_326957], **kwargs_326964)
    
    # Assigning a type to the variable 'coeffs0' (line 132)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 8), 'coeffs0', savgol_coeffs_call_result_326965)
    
    # Call to assert_allclose(...): (line 133)
    # Processing the call arguments (line 133)
    
    # Call to dot(...): (line 133)
    # Processing the call arguments (line 133)
    # Getting the type of 'x' (line 133)
    x_326969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 36), 'x', False)
    # Processing the call keyword arguments (line 133)
    kwargs_326970 = {}
    # Getting the type of 'coeffs0' (line 133)
    coeffs0_326967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 24), 'coeffs0', False)
    # Obtaining the member 'dot' of a type (line 133)
    dot_326968 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 24), coeffs0_326967, 'dot')
    # Calling dot(args, kwargs) (line 133)
    dot_call_result_326971 = invoke(stypy.reporting.localization.Localization(__file__, 133, 24), dot_326968, *[x_326969], **kwargs_326970)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'pos' (line 133)
    pos_326972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 42), 'pos', False)
    # Getting the type of 'x' (line 133)
    x_326973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 40), 'x', False)
    # Obtaining the member '__getitem__' of a type (line 133)
    getitem___326974 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 40), x_326973, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 133)
    subscript_call_result_326975 = invoke(stypy.reporting.localization.Localization(__file__, 133, 40), getitem___326974, pos_326972)
    
    # Processing the call keyword arguments (line 133)
    float_326976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 53), 'float')
    keyword_326977 = float_326976
    kwargs_326978 = {'atol': keyword_326977}
    # Getting the type of 'assert_allclose' (line 133)
    assert_allclose_326966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 133)
    assert_allclose_call_result_326979 = invoke(stypy.reporting.localization.Localization(__file__, 133, 8), assert_allclose_326966, *[dot_call_result_326971, subscript_call_result_326975], **kwargs_326978)
    
    
    # Assigning a Call to a Name (line 134):
    
    # Call to savgol_coeffs(...): (line 134)
    # Processing the call arguments (line 134)
    int_326981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 32), 'int')
    int_326982 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 35), 'int')
    # Processing the call keyword arguments (line 134)
    # Getting the type of 'pos' (line 134)
    pos_326983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 42), 'pos', False)
    keyword_326984 = pos_326983
    float_326985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 53), 'float')
    keyword_326986 = float_326985
    str_326987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 62), 'str', 'dot')
    keyword_326988 = str_326987
    int_326989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 75), 'int')
    keyword_326990 = int_326989
    kwargs_326991 = {'use': keyword_326988, 'deriv': keyword_326990, 'pos': keyword_326984, 'delta': keyword_326986}
    # Getting the type of 'savgol_coeffs' (line 134)
    savgol_coeffs_326980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 18), 'savgol_coeffs', False)
    # Calling savgol_coeffs(args, kwargs) (line 134)
    savgol_coeffs_call_result_326992 = invoke(stypy.reporting.localization.Localization(__file__, 134, 18), savgol_coeffs_326980, *[int_326981, int_326982], **kwargs_326991)
    
    # Assigning a type to the variable 'coeffs1' (line 134)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'coeffs1', savgol_coeffs_call_result_326992)
    
    # Call to assert_allclose(...): (line 135)
    # Processing the call arguments (line 135)
    
    # Call to dot(...): (line 135)
    # Processing the call arguments (line 135)
    # Getting the type of 'x' (line 135)
    x_326996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 36), 'x', False)
    # Processing the call keyword arguments (line 135)
    kwargs_326997 = {}
    # Getting the type of 'coeffs1' (line 135)
    coeffs1_326994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 24), 'coeffs1', False)
    # Obtaining the member 'dot' of a type (line 135)
    dot_326995 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 24), coeffs1_326994, 'dot')
    # Calling dot(args, kwargs) (line 135)
    dot_call_result_326998 = invoke(stypy.reporting.localization.Localization(__file__, 135, 24), dot_326995, *[x_326996], **kwargs_326997)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'pos' (line 135)
    pos_326999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 43), 'pos', False)
    # Getting the type of 'dx' (line 135)
    dx_327000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 40), 'dx', False)
    # Obtaining the member '__getitem__' of a type (line 135)
    getitem___327001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 40), dx_327000, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 135)
    subscript_call_result_327002 = invoke(stypy.reporting.localization.Localization(__file__, 135, 40), getitem___327001, pos_326999)
    
    # Processing the call keyword arguments (line 135)
    float_327003 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 54), 'float')
    keyword_327004 = float_327003
    kwargs_327005 = {'atol': keyword_327004}
    # Getting the type of 'assert_allclose' (line 135)
    assert_allclose_326993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 135)
    assert_allclose_call_result_327006 = invoke(stypy.reporting.localization.Localization(__file__, 135, 8), assert_allclose_326993, *[dot_call_result_326998, subscript_call_result_327002], **kwargs_327005)
    
    
    # Assigning a Call to a Name (line 136):
    
    # Call to savgol_coeffs(...): (line 136)
    # Processing the call arguments (line 136)
    int_327008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 32), 'int')
    int_327009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 35), 'int')
    # Processing the call keyword arguments (line 136)
    # Getting the type of 'pos' (line 136)
    pos_327010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 42), 'pos', False)
    keyword_327011 = pos_327010
    float_327012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 53), 'float')
    keyword_327013 = float_327012
    str_327014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 62), 'str', 'dot')
    keyword_327015 = str_327014
    int_327016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 75), 'int')
    keyword_327017 = int_327016
    kwargs_327018 = {'use': keyword_327015, 'deriv': keyword_327017, 'pos': keyword_327011, 'delta': keyword_327013}
    # Getting the type of 'savgol_coeffs' (line 136)
    savgol_coeffs_327007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 18), 'savgol_coeffs', False)
    # Calling savgol_coeffs(args, kwargs) (line 136)
    savgol_coeffs_call_result_327019 = invoke(stypy.reporting.localization.Localization(__file__, 136, 18), savgol_coeffs_327007, *[int_327008, int_327009], **kwargs_327018)
    
    # Assigning a type to the variable 'coeffs2' (line 136)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'coeffs2', savgol_coeffs_call_result_327019)
    
    # Call to assert_allclose(...): (line 137)
    # Processing the call arguments (line 137)
    
    # Call to dot(...): (line 137)
    # Processing the call arguments (line 137)
    # Getting the type of 'x' (line 137)
    x_327023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 36), 'x', False)
    # Processing the call keyword arguments (line 137)
    kwargs_327024 = {}
    # Getting the type of 'coeffs2' (line 137)
    coeffs2_327021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 24), 'coeffs2', False)
    # Obtaining the member 'dot' of a type (line 137)
    dot_327022 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 24), coeffs2_327021, 'dot')
    # Calling dot(args, kwargs) (line 137)
    dot_call_result_327025 = invoke(stypy.reporting.localization.Localization(__file__, 137, 24), dot_327022, *[x_327023], **kwargs_327024)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'pos' (line 137)
    pos_327026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 44), 'pos', False)
    # Getting the type of 'd2x' (line 137)
    d2x_327027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 40), 'd2x', False)
    # Obtaining the member '__getitem__' of a type (line 137)
    getitem___327028 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 40), d2x_327027, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 137)
    subscript_call_result_327029 = invoke(stypy.reporting.localization.Localization(__file__, 137, 40), getitem___327028, pos_327026)
    
    # Processing the call keyword arguments (line 137)
    float_327030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 55), 'float')
    keyword_327031 = float_327030
    kwargs_327032 = {'atol': keyword_327031}
    # Getting the type of 'assert_allclose' (line 137)
    assert_allclose_327020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 137)
    assert_allclose_call_result_327033 = invoke(stypy.reporting.localization.Localization(__file__, 137, 8), assert_allclose_327020, *[dot_call_result_327025, subscript_call_result_327029], **kwargs_327032)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_sg_coeffs_deriv(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_sg_coeffs_deriv' in the type store
    # Getting the type of 'stypy_return_type' (line 124)
    stypy_return_type_327034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_327034)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_sg_coeffs_deriv'
    return stypy_return_type_327034

# Assigning a type to the variable 'test_sg_coeffs_deriv' (line 124)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 0), 'test_sg_coeffs_deriv', test_sg_coeffs_deriv)

@norecursion
def test_sg_coeffs_large(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_sg_coeffs_large'
    module_type_store = module_type_store.open_function_context('test_sg_coeffs_large', 140, 0, False)
    
    # Passed parameters checking function
    test_sg_coeffs_large.stypy_localization = localization
    test_sg_coeffs_large.stypy_type_of_self = None
    test_sg_coeffs_large.stypy_type_store = module_type_store
    test_sg_coeffs_large.stypy_function_name = 'test_sg_coeffs_large'
    test_sg_coeffs_large.stypy_param_names_list = []
    test_sg_coeffs_large.stypy_varargs_param_name = None
    test_sg_coeffs_large.stypy_kwargs_param_name = None
    test_sg_coeffs_large.stypy_call_defaults = defaults
    test_sg_coeffs_large.stypy_call_varargs = varargs
    test_sg_coeffs_large.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_sg_coeffs_large', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_sg_coeffs_large', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_sg_coeffs_large(...)' code ##################

    
    # Assigning a Call to a Name (line 144):
    
    # Call to savgol_coeffs(...): (line 144)
    # Processing the call arguments (line 144)
    int_327036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 28), 'int')
    int_327037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 32), 'int')
    # Processing the call keyword arguments (line 144)
    kwargs_327038 = {}
    # Getting the type of 'savgol_coeffs' (line 144)
    savgol_coeffs_327035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 14), 'savgol_coeffs', False)
    # Calling savgol_coeffs(args, kwargs) (line 144)
    savgol_coeffs_call_result_327039 = invoke(stypy.reporting.localization.Localization(__file__, 144, 14), savgol_coeffs_327035, *[int_327036, int_327037], **kwargs_327038)
    
    # Assigning a type to the variable 'coeffs0' (line 144)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 4), 'coeffs0', savgol_coeffs_call_result_327039)
    
    # Call to assert_array_almost_equal(...): (line 145)
    # Processing the call arguments (line 145)
    # Getting the type of 'coeffs0' (line 145)
    coeffs0_327041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 30), 'coeffs0', False)
    
    # Obtaining the type of the subscript
    int_327042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 49), 'int')
    slice_327043 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 145, 39), None, None, int_327042)
    # Getting the type of 'coeffs0' (line 145)
    coeffs0_327044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 39), 'coeffs0', False)
    # Obtaining the member '__getitem__' of a type (line 145)
    getitem___327045 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 39), coeffs0_327044, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 145)
    subscript_call_result_327046 = invoke(stypy.reporting.localization.Localization(__file__, 145, 39), getitem___327045, slice_327043)
    
    # Processing the call keyword arguments (line 145)
    kwargs_327047 = {}
    # Getting the type of 'assert_array_almost_equal' (line 145)
    assert_array_almost_equal_327040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 4), 'assert_array_almost_equal', False)
    # Calling assert_array_almost_equal(args, kwargs) (line 145)
    assert_array_almost_equal_call_result_327048 = invoke(stypy.reporting.localization.Localization(__file__, 145, 4), assert_array_almost_equal_327040, *[coeffs0_327041, subscript_call_result_327046], **kwargs_327047)
    
    
    # Assigning a Call to a Name (line 146):
    
    # Call to savgol_coeffs(...): (line 146)
    # Processing the call arguments (line 146)
    int_327050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 28), 'int')
    int_327051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 32), 'int')
    # Processing the call keyword arguments (line 146)
    int_327052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 41), 'int')
    keyword_327053 = int_327052
    kwargs_327054 = {'deriv': keyword_327053}
    # Getting the type of 'savgol_coeffs' (line 146)
    savgol_coeffs_327049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 14), 'savgol_coeffs', False)
    # Calling savgol_coeffs(args, kwargs) (line 146)
    savgol_coeffs_call_result_327055 = invoke(stypy.reporting.localization.Localization(__file__, 146, 14), savgol_coeffs_327049, *[int_327050, int_327051], **kwargs_327054)
    
    # Assigning a type to the variable 'coeffs1' (line 146)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 4), 'coeffs1', savgol_coeffs_call_result_327055)
    
    # Call to assert_array_almost_equal(...): (line 147)
    # Processing the call arguments (line 147)
    # Getting the type of 'coeffs1' (line 147)
    coeffs1_327057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 30), 'coeffs1', False)
    
    
    # Obtaining the type of the subscript
    int_327058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 50), 'int')
    slice_327059 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 147, 40), None, None, int_327058)
    # Getting the type of 'coeffs1' (line 147)
    coeffs1_327060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 40), 'coeffs1', False)
    # Obtaining the member '__getitem__' of a type (line 147)
    getitem___327061 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 40), coeffs1_327060, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 147)
    subscript_call_result_327062 = invoke(stypy.reporting.localization.Localization(__file__, 147, 40), getitem___327061, slice_327059)
    
    # Applying the 'usub' unary operator (line 147)
    result___neg___327063 = python_operator(stypy.reporting.localization.Localization(__file__, 147, 39), 'usub', subscript_call_result_327062)
    
    # Processing the call keyword arguments (line 147)
    kwargs_327064 = {}
    # Getting the type of 'assert_array_almost_equal' (line 147)
    assert_array_almost_equal_327056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 4), 'assert_array_almost_equal', False)
    # Calling assert_array_almost_equal(args, kwargs) (line 147)
    assert_array_almost_equal_call_result_327065 = invoke(stypy.reporting.localization.Localization(__file__, 147, 4), assert_array_almost_equal_327056, *[coeffs1_327057, result___neg___327063], **kwargs_327064)
    
    
    # ################# End of 'test_sg_coeffs_large(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_sg_coeffs_large' in the type store
    # Getting the type of 'stypy_return_type' (line 140)
    stypy_return_type_327066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_327066)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_sg_coeffs_large'
    return stypy_return_type_327066

# Assigning a type to the variable 'test_sg_coeffs_large' (line 140)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 0), 'test_sg_coeffs_large', test_sg_coeffs_large)

@norecursion
def test_sg_filter_trivial(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_sg_filter_trivial'
    module_type_store = module_type_store.open_function_context('test_sg_filter_trivial', 155, 0, False)
    
    # Passed parameters checking function
    test_sg_filter_trivial.stypy_localization = localization
    test_sg_filter_trivial.stypy_type_of_self = None
    test_sg_filter_trivial.stypy_type_store = module_type_store
    test_sg_filter_trivial.stypy_function_name = 'test_sg_filter_trivial'
    test_sg_filter_trivial.stypy_param_names_list = []
    test_sg_filter_trivial.stypy_varargs_param_name = None
    test_sg_filter_trivial.stypy_kwargs_param_name = None
    test_sg_filter_trivial.stypy_call_defaults = defaults
    test_sg_filter_trivial.stypy_call_varargs = varargs
    test_sg_filter_trivial.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_sg_filter_trivial', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_sg_filter_trivial', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_sg_filter_trivial(...)' code ##################

    str_327067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 4), 'str', ' Test some trivial edge cases for savgol_filter().')
    
    # Assigning a Call to a Name (line 157):
    
    # Call to array(...): (line 157)
    # Processing the call arguments (line 157)
    
    # Obtaining an instance of the builtin type 'list' (line 157)
    list_327070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 157)
    # Adding element type (line 157)
    float_327071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 18), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 157, 17), list_327070, float_327071)
    
    # Processing the call keyword arguments (line 157)
    kwargs_327072 = {}
    # Getting the type of 'np' (line 157)
    np_327068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 8), 'np', False)
    # Obtaining the member 'array' of a type (line 157)
    array_327069 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 8), np_327068, 'array')
    # Calling array(args, kwargs) (line 157)
    array_call_result_327073 = invoke(stypy.reporting.localization.Localization(__file__, 157, 8), array_327069, *[list_327070], **kwargs_327072)
    
    # Assigning a type to the variable 'x' (line 157)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 4), 'x', array_call_result_327073)
    
    # Assigning a Call to a Name (line 158):
    
    # Call to savgol_filter(...): (line 158)
    # Processing the call arguments (line 158)
    # Getting the type of 'x' (line 158)
    x_327075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 22), 'x', False)
    int_327076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 25), 'int')
    int_327077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 28), 'int')
    # Processing the call keyword arguments (line 158)
    kwargs_327078 = {}
    # Getting the type of 'savgol_filter' (line 158)
    savgol_filter_327074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'savgol_filter', False)
    # Calling savgol_filter(args, kwargs) (line 158)
    savgol_filter_call_result_327079 = invoke(stypy.reporting.localization.Localization(__file__, 158, 8), savgol_filter_327074, *[x_327075, int_327076, int_327077], **kwargs_327078)
    
    # Assigning a type to the variable 'y' (line 158)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 4), 'y', savgol_filter_call_result_327079)
    
    # Call to assert_equal(...): (line 159)
    # Processing the call arguments (line 159)
    # Getting the type of 'y' (line 159)
    y_327081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 17), 'y', False)
    
    # Obtaining an instance of the builtin type 'list' (line 159)
    list_327082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 159)
    # Adding element type (line 159)
    float_327083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 21), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 20), list_327082, float_327083)
    
    # Processing the call keyword arguments (line 159)
    kwargs_327084 = {}
    # Getting the type of 'assert_equal' (line 159)
    assert_equal_327080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 159)
    assert_equal_call_result_327085 = invoke(stypy.reporting.localization.Localization(__file__, 159, 4), assert_equal_327080, *[y_327081, list_327082], **kwargs_327084)
    
    
    # Assigning a Call to a Name (line 164):
    
    # Call to array(...): (line 164)
    # Processing the call arguments (line 164)
    
    # Obtaining an instance of the builtin type 'list' (line 164)
    list_327088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 164)
    # Adding element type (line 164)
    float_327089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 18), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 17), list_327088, float_327089)
    
    # Processing the call keyword arguments (line 164)
    kwargs_327090 = {}
    # Getting the type of 'np' (line 164)
    np_327086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 8), 'np', False)
    # Obtaining the member 'array' of a type (line 164)
    array_327087 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 8), np_327086, 'array')
    # Calling array(args, kwargs) (line 164)
    array_call_result_327091 = invoke(stypy.reporting.localization.Localization(__file__, 164, 8), array_327087, *[list_327088], **kwargs_327090)
    
    # Assigning a type to the variable 'x' (line 164)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 4), 'x', array_call_result_327091)
    
    # Assigning a Call to a Name (line 165):
    
    # Call to savgol_filter(...): (line 165)
    # Processing the call arguments (line 165)
    # Getting the type of 'x' (line 165)
    x_327093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 22), 'x', False)
    int_327094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 25), 'int')
    int_327095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 28), 'int')
    # Processing the call keyword arguments (line 165)
    str_327096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 36), 'str', 'constant')
    keyword_327097 = str_327096
    kwargs_327098 = {'mode': keyword_327097}
    # Getting the type of 'savgol_filter' (line 165)
    savgol_filter_327092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 8), 'savgol_filter', False)
    # Calling savgol_filter(args, kwargs) (line 165)
    savgol_filter_call_result_327099 = invoke(stypy.reporting.localization.Localization(__file__, 165, 8), savgol_filter_327092, *[x_327093, int_327094, int_327095], **kwargs_327098)
    
    # Assigning a type to the variable 'y' (line 165)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 4), 'y', savgol_filter_call_result_327099)
    
    # Call to assert_almost_equal(...): (line 166)
    # Processing the call arguments (line 166)
    # Getting the type of 'y' (line 166)
    y_327101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 24), 'y', False)
    
    # Obtaining an instance of the builtin type 'list' (line 166)
    list_327102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 27), 'list')
    # Adding type elements to the builtin type 'list' instance (line 166)
    # Adding element type (line 166)
    float_327103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 28), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 166, 27), list_327102, float_327103)
    
    # Processing the call keyword arguments (line 166)
    int_327104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 42), 'int')
    keyword_327105 = int_327104
    kwargs_327106 = {'decimal': keyword_327105}
    # Getting the type of 'assert_almost_equal' (line 166)
    assert_almost_equal_327100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 4), 'assert_almost_equal', False)
    # Calling assert_almost_equal(args, kwargs) (line 166)
    assert_almost_equal_call_result_327107 = invoke(stypy.reporting.localization.Localization(__file__, 166, 4), assert_almost_equal_327100, *[y_327101, list_327102], **kwargs_327106)
    
    
    # Assigning a Call to a Name (line 168):
    
    # Call to array(...): (line 168)
    # Processing the call arguments (line 168)
    
    # Obtaining an instance of the builtin type 'list' (line 168)
    list_327110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 168)
    # Adding element type (line 168)
    float_327111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 18), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 17), list_327110, float_327111)
    
    # Processing the call keyword arguments (line 168)
    kwargs_327112 = {}
    # Getting the type of 'np' (line 168)
    np_327108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), 'np', False)
    # Obtaining the member 'array' of a type (line 168)
    array_327109 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 8), np_327108, 'array')
    # Calling array(args, kwargs) (line 168)
    array_call_result_327113 = invoke(stypy.reporting.localization.Localization(__file__, 168, 8), array_327109, *[list_327110], **kwargs_327112)
    
    # Assigning a type to the variable 'x' (line 168)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 4), 'x', array_call_result_327113)
    
    # Assigning a Call to a Name (line 169):
    
    # Call to savgol_filter(...): (line 169)
    # Processing the call arguments (line 169)
    # Getting the type of 'x' (line 169)
    x_327115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 22), 'x', False)
    int_327116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 25), 'int')
    int_327117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 28), 'int')
    # Processing the call keyword arguments (line 169)
    str_327118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 36), 'str', 'nearest')
    keyword_327119 = str_327118
    kwargs_327120 = {'mode': keyword_327119}
    # Getting the type of 'savgol_filter' (line 169)
    savgol_filter_327114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'savgol_filter', False)
    # Calling savgol_filter(args, kwargs) (line 169)
    savgol_filter_call_result_327121 = invoke(stypy.reporting.localization.Localization(__file__, 169, 8), savgol_filter_327114, *[x_327115, int_327116, int_327117], **kwargs_327120)
    
    # Assigning a type to the variable 'y' (line 169)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 4), 'y', savgol_filter_call_result_327121)
    
    # Call to assert_almost_equal(...): (line 170)
    # Processing the call arguments (line 170)
    # Getting the type of 'y' (line 170)
    y_327123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 24), 'y', False)
    
    # Obtaining an instance of the builtin type 'list' (line 170)
    list_327124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 27), 'list')
    # Adding type elements to the builtin type 'list' instance (line 170)
    # Adding element type (line 170)
    float_327125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 28), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 170, 27), list_327124, float_327125)
    
    # Processing the call keyword arguments (line 170)
    int_327126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 42), 'int')
    keyword_327127 = int_327126
    kwargs_327128 = {'decimal': keyword_327127}
    # Getting the type of 'assert_almost_equal' (line 170)
    assert_almost_equal_327122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 4), 'assert_almost_equal', False)
    # Calling assert_almost_equal(args, kwargs) (line 170)
    assert_almost_equal_call_result_327129 = invoke(stypy.reporting.localization.Localization(__file__, 170, 4), assert_almost_equal_327122, *[y_327123, list_327124], **kwargs_327128)
    
    
    # Assigning a Call to a Name (line 172):
    
    # Call to array(...): (line 172)
    # Processing the call arguments (line 172)
    
    # Obtaining an instance of the builtin type 'list' (line 172)
    list_327132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 172)
    # Adding element type (line 172)
    float_327133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 18), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 172, 17), list_327132, float_327133)
    
    int_327134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 25), 'int')
    # Applying the binary operator '*' (line 172)
    result_mul_327135 = python_operator(stypy.reporting.localization.Localization(__file__, 172, 17), '*', list_327132, int_327134)
    
    # Processing the call keyword arguments (line 172)
    kwargs_327136 = {}
    # Getting the type of 'np' (line 172)
    np_327130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 8), 'np', False)
    # Obtaining the member 'array' of a type (line 172)
    array_327131 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 8), np_327130, 'array')
    # Calling array(args, kwargs) (line 172)
    array_call_result_327137 = invoke(stypy.reporting.localization.Localization(__file__, 172, 8), array_327131, *[result_mul_327135], **kwargs_327136)
    
    # Assigning a type to the variable 'x' (line 172)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 4), 'x', array_call_result_327137)
    
    # Assigning a Call to a Name (line 173):
    
    # Call to savgol_filter(...): (line 173)
    # Processing the call arguments (line 173)
    # Getting the type of 'x' (line 173)
    x_327139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 22), 'x', False)
    int_327140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 25), 'int')
    int_327141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 28), 'int')
    # Processing the call keyword arguments (line 173)
    str_327142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 36), 'str', 'wrap')
    keyword_327143 = str_327142
    kwargs_327144 = {'mode': keyword_327143}
    # Getting the type of 'savgol_filter' (line 173)
    savgol_filter_327138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 8), 'savgol_filter', False)
    # Calling savgol_filter(args, kwargs) (line 173)
    savgol_filter_call_result_327145 = invoke(stypy.reporting.localization.Localization(__file__, 173, 8), savgol_filter_327138, *[x_327139, int_327140, int_327141], **kwargs_327144)
    
    # Assigning a type to the variable 'y' (line 173)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 4), 'y', savgol_filter_call_result_327145)
    
    # Call to assert_almost_equal(...): (line 174)
    # Processing the call arguments (line 174)
    # Getting the type of 'y' (line 174)
    y_327147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 24), 'y', False)
    
    # Obtaining an instance of the builtin type 'list' (line 174)
    list_327148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 27), 'list')
    # Adding type elements to the builtin type 'list' instance (line 174)
    # Adding element type (line 174)
    float_327149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 28), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 174, 27), list_327148, float_327149)
    # Adding element type (line 174)
    float_327150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 33), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 174, 27), list_327148, float_327150)
    # Adding element type (line 174)
    float_327151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 38), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 174, 27), list_327148, float_327151)
    
    # Processing the call keyword arguments (line 174)
    int_327152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 52), 'int')
    keyword_327153 = int_327152
    kwargs_327154 = {'decimal': keyword_327153}
    # Getting the type of 'assert_almost_equal' (line 174)
    assert_almost_equal_327146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 4), 'assert_almost_equal', False)
    # Calling assert_almost_equal(args, kwargs) (line 174)
    assert_almost_equal_call_result_327155 = invoke(stypy.reporting.localization.Localization(__file__, 174, 4), assert_almost_equal_327146, *[y_327147, list_327148], **kwargs_327154)
    
    
    # ################# End of 'test_sg_filter_trivial(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_sg_filter_trivial' in the type store
    # Getting the type of 'stypy_return_type' (line 155)
    stypy_return_type_327156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_327156)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_sg_filter_trivial'
    return stypy_return_type_327156

# Assigning a type to the variable 'test_sg_filter_trivial' (line 155)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 0), 'test_sg_filter_trivial', test_sg_filter_trivial)

@norecursion
def test_sg_filter_basic(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_sg_filter_basic'
    module_type_store = module_type_store.open_function_context('test_sg_filter_basic', 177, 0, False)
    
    # Passed parameters checking function
    test_sg_filter_basic.stypy_localization = localization
    test_sg_filter_basic.stypy_type_of_self = None
    test_sg_filter_basic.stypy_type_store = module_type_store
    test_sg_filter_basic.stypy_function_name = 'test_sg_filter_basic'
    test_sg_filter_basic.stypy_param_names_list = []
    test_sg_filter_basic.stypy_varargs_param_name = None
    test_sg_filter_basic.stypy_kwargs_param_name = None
    test_sg_filter_basic.stypy_call_defaults = defaults
    test_sg_filter_basic.stypy_call_varargs = varargs
    test_sg_filter_basic.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_sg_filter_basic', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_sg_filter_basic', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_sg_filter_basic(...)' code ##################

    
    # Assigning a Call to a Name (line 179):
    
    # Call to array(...): (line 179)
    # Processing the call arguments (line 179)
    
    # Obtaining an instance of the builtin type 'list' (line 179)
    list_327159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 179)
    # Adding element type (line 179)
    float_327160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 18), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 17), list_327159, float_327160)
    # Adding element type (line 179)
    float_327161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 23), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 17), list_327159, float_327161)
    # Adding element type (line 179)
    float_327162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 28), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 17), list_327159, float_327162)
    
    # Processing the call keyword arguments (line 179)
    kwargs_327163 = {}
    # Getting the type of 'np' (line 179)
    np_327157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 8), 'np', False)
    # Obtaining the member 'array' of a type (line 179)
    array_327158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 8), np_327157, 'array')
    # Calling array(args, kwargs) (line 179)
    array_call_result_327164 = invoke(stypy.reporting.localization.Localization(__file__, 179, 8), array_327158, *[list_327159], **kwargs_327163)
    
    # Assigning a type to the variable 'x' (line 179)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 4), 'x', array_call_result_327164)
    
    # Assigning a Call to a Name (line 180):
    
    # Call to savgol_filter(...): (line 180)
    # Processing the call arguments (line 180)
    # Getting the type of 'x' (line 180)
    x_327166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 22), 'x', False)
    int_327167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 25), 'int')
    int_327168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 28), 'int')
    # Processing the call keyword arguments (line 180)
    str_327169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 36), 'str', 'constant')
    keyword_327170 = str_327169
    kwargs_327171 = {'mode': keyword_327170}
    # Getting the type of 'savgol_filter' (line 180)
    savgol_filter_327165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 8), 'savgol_filter', False)
    # Calling savgol_filter(args, kwargs) (line 180)
    savgol_filter_call_result_327172 = invoke(stypy.reporting.localization.Localization(__file__, 180, 8), savgol_filter_327165, *[x_327166, int_327167, int_327168], **kwargs_327171)
    
    # Assigning a type to the variable 'y' (line 180)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 4), 'y', savgol_filter_call_result_327172)
    
    # Call to assert_allclose(...): (line 181)
    # Processing the call arguments (line 181)
    # Getting the type of 'y' (line 181)
    y_327174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 20), 'y', False)
    
    # Obtaining an instance of the builtin type 'list' (line 181)
    list_327175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 181)
    # Adding element type (line 181)
    float_327176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 24), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 181, 23), list_327175, float_327176)
    # Adding element type (line 181)
    float_327177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 29), 'float')
    int_327178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 35), 'int')
    # Applying the binary operator 'div' (line 181)
    result_div_327179 = python_operator(stypy.reporting.localization.Localization(__file__, 181, 29), 'div', float_327177, int_327178)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 181, 23), list_327175, result_div_327179)
    # Adding element type (line 181)
    float_327180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 38), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 181, 23), list_327175, float_327180)
    
    # Processing the call keyword arguments (line 181)
    kwargs_327181 = {}
    # Getting the type of 'assert_allclose' (line 181)
    assert_allclose_327173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 181)
    assert_allclose_call_result_327182 = invoke(stypy.reporting.localization.Localization(__file__, 181, 4), assert_allclose_327173, *[y_327174, list_327175], **kwargs_327181)
    
    
    # Assigning a Call to a Name (line 183):
    
    # Call to savgol_filter(...): (line 183)
    # Processing the call arguments (line 183)
    # Getting the type of 'x' (line 183)
    x_327184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 22), 'x', False)
    int_327185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 25), 'int')
    int_327186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 28), 'int')
    # Processing the call keyword arguments (line 183)
    str_327187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 36), 'str', 'mirror')
    keyword_327188 = str_327187
    kwargs_327189 = {'mode': keyword_327188}
    # Getting the type of 'savgol_filter' (line 183)
    savgol_filter_327183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 8), 'savgol_filter', False)
    # Calling savgol_filter(args, kwargs) (line 183)
    savgol_filter_call_result_327190 = invoke(stypy.reporting.localization.Localization(__file__, 183, 8), savgol_filter_327183, *[x_327184, int_327185, int_327186], **kwargs_327189)
    
    # Assigning a type to the variable 'y' (line 183)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 4), 'y', savgol_filter_call_result_327190)
    
    # Call to assert_allclose(...): (line 184)
    # Processing the call arguments (line 184)
    # Getting the type of 'y' (line 184)
    y_327192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 20), 'y', False)
    
    # Obtaining an instance of the builtin type 'list' (line 184)
    list_327193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 184)
    # Adding element type (line 184)
    float_327194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 24), 'float')
    int_327195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 30), 'int')
    # Applying the binary operator 'div' (line 184)
    result_div_327196 = python_operator(stypy.reporting.localization.Localization(__file__, 184, 24), 'div', float_327194, int_327195)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 184, 23), list_327193, result_div_327196)
    # Adding element type (line 184)
    float_327197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 33), 'float')
    int_327198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 39), 'int')
    # Applying the binary operator 'div' (line 184)
    result_div_327199 = python_operator(stypy.reporting.localization.Localization(__file__, 184, 33), 'div', float_327197, int_327198)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 184, 23), list_327193, result_div_327199)
    # Adding element type (line 184)
    float_327200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 42), 'float')
    int_327201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 48), 'int')
    # Applying the binary operator 'div' (line 184)
    result_div_327202 = python_operator(stypy.reporting.localization.Localization(__file__, 184, 42), 'div', float_327200, int_327201)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 184, 23), list_327193, result_div_327202)
    
    # Processing the call keyword arguments (line 184)
    kwargs_327203 = {}
    # Getting the type of 'assert_allclose' (line 184)
    assert_allclose_327191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 184)
    assert_allclose_call_result_327204 = invoke(stypy.reporting.localization.Localization(__file__, 184, 4), assert_allclose_327191, *[y_327192, list_327193], **kwargs_327203)
    
    
    # Assigning a Call to a Name (line 186):
    
    # Call to savgol_filter(...): (line 186)
    # Processing the call arguments (line 186)
    # Getting the type of 'x' (line 186)
    x_327206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 22), 'x', False)
    int_327207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 25), 'int')
    int_327208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 28), 'int')
    # Processing the call keyword arguments (line 186)
    str_327209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 36), 'str', 'wrap')
    keyword_327210 = str_327209
    kwargs_327211 = {'mode': keyword_327210}
    # Getting the type of 'savgol_filter' (line 186)
    savgol_filter_327205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 8), 'savgol_filter', False)
    # Calling savgol_filter(args, kwargs) (line 186)
    savgol_filter_call_result_327212 = invoke(stypy.reporting.localization.Localization(__file__, 186, 8), savgol_filter_327205, *[x_327206, int_327207, int_327208], **kwargs_327211)
    
    # Assigning a type to the variable 'y' (line 186)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 4), 'y', savgol_filter_call_result_327212)
    
    # Call to assert_allclose(...): (line 187)
    # Processing the call arguments (line 187)
    # Getting the type of 'y' (line 187)
    y_327214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 20), 'y', False)
    
    # Obtaining an instance of the builtin type 'list' (line 187)
    list_327215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 187)
    # Adding element type (line 187)
    float_327216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 24), 'float')
    int_327217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 30), 'int')
    # Applying the binary operator 'div' (line 187)
    result_div_327218 = python_operator(stypy.reporting.localization.Localization(__file__, 187, 24), 'div', float_327216, int_327217)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 23), list_327215, result_div_327218)
    # Adding element type (line 187)
    float_327219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 33), 'float')
    int_327220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 39), 'int')
    # Applying the binary operator 'div' (line 187)
    result_div_327221 = python_operator(stypy.reporting.localization.Localization(__file__, 187, 33), 'div', float_327219, int_327220)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 23), list_327215, result_div_327221)
    # Adding element type (line 187)
    float_327222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 42), 'float')
    int_327223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 48), 'int')
    # Applying the binary operator 'div' (line 187)
    result_div_327224 = python_operator(stypy.reporting.localization.Localization(__file__, 187, 42), 'div', float_327222, int_327223)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 23), list_327215, result_div_327224)
    
    # Processing the call keyword arguments (line 187)
    kwargs_327225 = {}
    # Getting the type of 'assert_allclose' (line 187)
    assert_allclose_327213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 187)
    assert_allclose_call_result_327226 = invoke(stypy.reporting.localization.Localization(__file__, 187, 4), assert_allclose_327213, *[y_327214, list_327215], **kwargs_327225)
    
    
    # ################# End of 'test_sg_filter_basic(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_sg_filter_basic' in the type store
    # Getting the type of 'stypy_return_type' (line 177)
    stypy_return_type_327227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_327227)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_sg_filter_basic'
    return stypy_return_type_327227

# Assigning a type to the variable 'test_sg_filter_basic' (line 177)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 0), 'test_sg_filter_basic', test_sg_filter_basic)

@norecursion
def test_sg_filter_2d(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_sg_filter_2d'
    module_type_store = module_type_store.open_function_context('test_sg_filter_2d', 190, 0, False)
    
    # Passed parameters checking function
    test_sg_filter_2d.stypy_localization = localization
    test_sg_filter_2d.stypy_type_of_self = None
    test_sg_filter_2d.stypy_type_store = module_type_store
    test_sg_filter_2d.stypy_function_name = 'test_sg_filter_2d'
    test_sg_filter_2d.stypy_param_names_list = []
    test_sg_filter_2d.stypy_varargs_param_name = None
    test_sg_filter_2d.stypy_kwargs_param_name = None
    test_sg_filter_2d.stypy_call_defaults = defaults
    test_sg_filter_2d.stypy_call_varargs = varargs
    test_sg_filter_2d.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_sg_filter_2d', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_sg_filter_2d', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_sg_filter_2d(...)' code ##################

    
    # Assigning a Call to a Name (line 191):
    
    # Call to array(...): (line 191)
    # Processing the call arguments (line 191)
    
    # Obtaining an instance of the builtin type 'list' (line 191)
    list_327230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 191)
    # Adding element type (line 191)
    
    # Obtaining an instance of the builtin type 'list' (line 191)
    list_327231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 191)
    # Adding element type (line 191)
    float_327232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 19), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 191, 18), list_327231, float_327232)
    # Adding element type (line 191)
    float_327233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 24), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 191, 18), list_327231, float_327233)
    # Adding element type (line 191)
    float_327234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 29), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 191, 18), list_327231, float_327234)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 191, 17), list_327230, list_327231)
    # Adding element type (line 191)
    
    # Obtaining an instance of the builtin type 'list' (line 192)
    list_327235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 192)
    # Adding element type (line 192)
    float_327236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 19), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 192, 18), list_327235, float_327236)
    # Adding element type (line 192)
    float_327237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 24), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 192, 18), list_327235, float_327237)
    # Adding element type (line 192)
    float_327238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 29), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 192, 18), list_327235, float_327238)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 191, 17), list_327230, list_327235)
    
    # Processing the call keyword arguments (line 191)
    kwargs_327239 = {}
    # Getting the type of 'np' (line 191)
    np_327228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 8), 'np', False)
    # Obtaining the member 'array' of a type (line 191)
    array_327229 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 8), np_327228, 'array')
    # Calling array(args, kwargs) (line 191)
    array_call_result_327240 = invoke(stypy.reporting.localization.Localization(__file__, 191, 8), array_327229, *[list_327230], **kwargs_327239)
    
    # Assigning a type to the variable 'x' (line 191)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 4), 'x', array_call_result_327240)
    
    # Assigning a Call to a Name (line 193):
    
    # Call to array(...): (line 193)
    # Processing the call arguments (line 193)
    
    # Obtaining an instance of the builtin type 'list' (line 193)
    list_327243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 193)
    # Adding element type (line 193)
    
    # Obtaining an instance of the builtin type 'list' (line 193)
    list_327244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 25), 'list')
    # Adding type elements to the builtin type 'list' instance (line 193)
    # Adding element type (line 193)
    float_327245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 26), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 193, 25), list_327244, float_327245)
    # Adding element type (line 193)
    float_327246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 31), 'float')
    int_327247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 37), 'int')
    # Applying the binary operator 'div' (line 193)
    result_div_327248 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 31), 'div', float_327246, int_327247)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 193, 25), list_327244, result_div_327248)
    # Adding element type (line 193)
    float_327249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 40), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 193, 25), list_327244, float_327249)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 193, 24), list_327243, list_327244)
    # Adding element type (line 193)
    
    # Obtaining an instance of the builtin type 'list' (line 194)
    list_327250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 25), 'list')
    # Adding type elements to the builtin type 'list' instance (line 194)
    # Adding element type (line 194)
    float_327251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 26), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 25), list_327250, float_327251)
    # Adding element type (line 194)
    float_327252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 31), 'float')
    int_327253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 37), 'int')
    # Applying the binary operator 'div' (line 194)
    result_div_327254 = python_operator(stypy.reporting.localization.Localization(__file__, 194, 31), 'div', float_327252, int_327253)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 25), list_327250, result_div_327254)
    # Adding element type (line 194)
    float_327255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 40), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 25), list_327250, float_327255)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 193, 24), list_327243, list_327250)
    
    # Processing the call keyword arguments (line 193)
    kwargs_327256 = {}
    # Getting the type of 'np' (line 193)
    np_327241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 15), 'np', False)
    # Obtaining the member 'array' of a type (line 193)
    array_327242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 15), np_327241, 'array')
    # Calling array(args, kwargs) (line 193)
    array_call_result_327257 = invoke(stypy.reporting.localization.Localization(__file__, 193, 15), array_327242, *[list_327243], **kwargs_327256)
    
    # Assigning a type to the variable 'expected' (line 193)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 4), 'expected', array_call_result_327257)
    
    # Assigning a Call to a Name (line 195):
    
    # Call to savgol_filter(...): (line 195)
    # Processing the call arguments (line 195)
    # Getting the type of 'x' (line 195)
    x_327259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 22), 'x', False)
    int_327260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 25), 'int')
    int_327261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 28), 'int')
    # Processing the call keyword arguments (line 195)
    str_327262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 36), 'str', 'constant')
    keyword_327263 = str_327262
    kwargs_327264 = {'mode': keyword_327263}
    # Getting the type of 'savgol_filter' (line 195)
    savgol_filter_327258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 8), 'savgol_filter', False)
    # Calling savgol_filter(args, kwargs) (line 195)
    savgol_filter_call_result_327265 = invoke(stypy.reporting.localization.Localization(__file__, 195, 8), savgol_filter_327258, *[x_327259, int_327260, int_327261], **kwargs_327264)
    
    # Assigning a type to the variable 'y' (line 195)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 4), 'y', savgol_filter_call_result_327265)
    
    # Call to assert_allclose(...): (line 196)
    # Processing the call arguments (line 196)
    # Getting the type of 'y' (line 196)
    y_327267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 20), 'y', False)
    # Getting the type of 'expected' (line 196)
    expected_327268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 23), 'expected', False)
    # Processing the call keyword arguments (line 196)
    kwargs_327269 = {}
    # Getting the type of 'assert_allclose' (line 196)
    assert_allclose_327266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 196)
    assert_allclose_call_result_327270 = invoke(stypy.reporting.localization.Localization(__file__, 196, 4), assert_allclose_327266, *[y_327267, expected_327268], **kwargs_327269)
    
    
    # Assigning a Call to a Name (line 198):
    
    # Call to savgol_filter(...): (line 198)
    # Processing the call arguments (line 198)
    # Getting the type of 'x' (line 198)
    x_327272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 22), 'x', False)
    # Obtaining the member 'T' of a type (line 198)
    T_327273 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 22), x_327272, 'T')
    int_327274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 27), 'int')
    int_327275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 30), 'int')
    # Processing the call keyword arguments (line 198)
    str_327276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 38), 'str', 'constant')
    keyword_327277 = str_327276
    int_327278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 55), 'int')
    keyword_327279 = int_327278
    kwargs_327280 = {'mode': keyword_327277, 'axis': keyword_327279}
    # Getting the type of 'savgol_filter' (line 198)
    savgol_filter_327271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 'savgol_filter', False)
    # Calling savgol_filter(args, kwargs) (line 198)
    savgol_filter_call_result_327281 = invoke(stypy.reporting.localization.Localization(__file__, 198, 8), savgol_filter_327271, *[T_327273, int_327274, int_327275], **kwargs_327280)
    
    # Assigning a type to the variable 'y' (line 198)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 4), 'y', savgol_filter_call_result_327281)
    
    # Call to assert_allclose(...): (line 199)
    # Processing the call arguments (line 199)
    # Getting the type of 'y' (line 199)
    y_327283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 20), 'y', False)
    # Getting the type of 'expected' (line 199)
    expected_327284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 23), 'expected', False)
    # Obtaining the member 'T' of a type (line 199)
    T_327285 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 23), expected_327284, 'T')
    # Processing the call keyword arguments (line 199)
    kwargs_327286 = {}
    # Getting the type of 'assert_allclose' (line 199)
    assert_allclose_327282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 199)
    assert_allclose_call_result_327287 = invoke(stypy.reporting.localization.Localization(__file__, 199, 4), assert_allclose_327282, *[y_327283, T_327285], **kwargs_327286)
    
    
    # ################# End of 'test_sg_filter_2d(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_sg_filter_2d' in the type store
    # Getting the type of 'stypy_return_type' (line 190)
    stypy_return_type_327288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_327288)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_sg_filter_2d'
    return stypy_return_type_327288

# Assigning a type to the variable 'test_sg_filter_2d' (line 190)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 0), 'test_sg_filter_2d', test_sg_filter_2d)

@norecursion
def test_sg_filter_interp_edges(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_sg_filter_interp_edges'
    module_type_store = module_type_store.open_function_context('test_sg_filter_interp_edges', 202, 0, False)
    
    # Passed parameters checking function
    test_sg_filter_interp_edges.stypy_localization = localization
    test_sg_filter_interp_edges.stypy_type_of_self = None
    test_sg_filter_interp_edges.stypy_type_store = module_type_store
    test_sg_filter_interp_edges.stypy_function_name = 'test_sg_filter_interp_edges'
    test_sg_filter_interp_edges.stypy_param_names_list = []
    test_sg_filter_interp_edges.stypy_varargs_param_name = None
    test_sg_filter_interp_edges.stypy_kwargs_param_name = None
    test_sg_filter_interp_edges.stypy_call_defaults = defaults
    test_sg_filter_interp_edges.stypy_call_varargs = varargs
    test_sg_filter_interp_edges.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_sg_filter_interp_edges', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_sg_filter_interp_edges', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_sg_filter_interp_edges(...)' code ##################

    
    # Assigning a Call to a Name (line 207):
    
    # Call to linspace(...): (line 207)
    # Processing the call arguments (line 207)
    int_327291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 20), 'int')
    int_327292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 24), 'int')
    int_327293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 27), 'int')
    # Processing the call keyword arguments (line 207)
    kwargs_327294 = {}
    # Getting the type of 'np' (line 207)
    np_327289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 8), 'np', False)
    # Obtaining the member 'linspace' of a type (line 207)
    linspace_327290 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 8), np_327289, 'linspace')
    # Calling linspace(args, kwargs) (line 207)
    linspace_call_result_327295 = invoke(stypy.reporting.localization.Localization(__file__, 207, 8), linspace_327290, *[int_327291, int_327292, int_327293], **kwargs_327294)
    
    # Assigning a type to the variable 't' (line 207)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 4), 't', linspace_call_result_327295)
    
    # Assigning a BinOp to a Name (line 208):
    
    # Obtaining the type of the subscript
    int_327296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 14), 'int')
    # Getting the type of 't' (line 208)
    t_327297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 12), 't')
    # Obtaining the member '__getitem__' of a type (line 208)
    getitem___327298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 12), t_327297, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 208)
    subscript_call_result_327299 = invoke(stypy.reporting.localization.Localization(__file__, 208, 12), getitem___327298, int_327296)
    
    
    # Obtaining the type of the subscript
    int_327300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 21), 'int')
    # Getting the type of 't' (line 208)
    t_327301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 19), 't')
    # Obtaining the member '__getitem__' of a type (line 208)
    getitem___327302 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 19), t_327301, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 208)
    subscript_call_result_327303 = invoke(stypy.reporting.localization.Localization(__file__, 208, 19), getitem___327302, int_327300)
    
    # Applying the binary operator '-' (line 208)
    result_sub_327304 = python_operator(stypy.reporting.localization.Localization(__file__, 208, 12), '-', subscript_call_result_327299, subscript_call_result_327303)
    
    # Assigning a type to the variable 'delta' (line 208)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 4), 'delta', result_sub_327304)
    
    # Assigning a Call to a Name (line 210):
    
    # Call to array(...): (line 210)
    # Processing the call arguments (line 210)
    
    # Obtaining an instance of the builtin type 'list' (line 210)
    list_327307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 210)
    # Adding element type (line 210)
    # Getting the type of 't' (line 210)
    t_327308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 18), 't', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 210, 17), list_327307, t_327308)
    # Adding element type (line 210)
    int_327309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 18), 'int')
    # Getting the type of 't' (line 211)
    t_327310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 22), 't', False)
    int_327311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 27), 'int')
    # Applying the binary operator '**' (line 211)
    result_pow_327312 = python_operator(stypy.reporting.localization.Localization(__file__, 211, 22), '**', t_327310, int_327311)
    
    # Applying the binary operator '*' (line 211)
    result_mul_327313 = python_operator(stypy.reporting.localization.Localization(__file__, 211, 18), '*', int_327309, result_pow_327312)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 210, 17), list_327307, result_mul_327313)
    # Adding element type (line 210)
    # Getting the type of 't' (line 212)
    t_327314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 18), 't', False)
    int_327315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 23), 'int')
    # Applying the binary operator '**' (line 212)
    result_pow_327316 = python_operator(stypy.reporting.localization.Localization(__file__, 212, 18), '**', t_327314, int_327315)
    
    # Getting the type of 't' (line 212)
    t_327317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 27), 't', False)
    # Applying the binary operator '-' (line 212)
    result_sub_327318 = python_operator(stypy.reporting.localization.Localization(__file__, 212, 18), '-', result_pow_327316, t_327317)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 210, 17), list_327307, result_sub_327318)
    
    # Processing the call keyword arguments (line 210)
    kwargs_327319 = {}
    # Getting the type of 'np' (line 210)
    np_327305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 8), 'np', False)
    # Obtaining the member 'array' of a type (line 210)
    array_327306 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 8), np_327305, 'array')
    # Calling array(args, kwargs) (line 210)
    array_call_result_327320 = invoke(stypy.reporting.localization.Localization(__file__, 210, 8), array_327306, *[list_327307], **kwargs_327319)
    
    # Assigning a type to the variable 'x' (line 210)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 4), 'x', array_call_result_327320)
    
    # Assigning a Call to a Name (line 213):
    
    # Call to array(...): (line 213)
    # Processing the call arguments (line 213)
    
    # Obtaining an instance of the builtin type 'list' (line 213)
    list_327323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 213)
    # Adding element type (line 213)
    
    # Call to ones_like(...): (line 213)
    # Processing the call arguments (line 213)
    # Getting the type of 't' (line 213)
    t_327326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 32), 't', False)
    # Processing the call keyword arguments (line 213)
    kwargs_327327 = {}
    # Getting the type of 'np' (line 213)
    np_327324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 19), 'np', False)
    # Obtaining the member 'ones_like' of a type (line 213)
    ones_like_327325 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 19), np_327324, 'ones_like')
    # Calling ones_like(args, kwargs) (line 213)
    ones_like_call_result_327328 = invoke(stypy.reporting.localization.Localization(__file__, 213, 19), ones_like_327325, *[t_327326], **kwargs_327327)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 213, 18), list_327323, ones_like_call_result_327328)
    # Adding element type (line 213)
    int_327329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 19), 'int')
    # Getting the type of 't' (line 214)
    t_327330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 23), 't', False)
    # Applying the binary operator '*' (line 214)
    result_mul_327331 = python_operator(stypy.reporting.localization.Localization(__file__, 214, 19), '*', int_327329, t_327330)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 213, 18), list_327323, result_mul_327331)
    # Adding element type (line 213)
    int_327332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 19), 'int')
    # Getting the type of 't' (line 215)
    t_327333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 23), 't', False)
    int_327334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 28), 'int')
    # Applying the binary operator '**' (line 215)
    result_pow_327335 = python_operator(stypy.reporting.localization.Localization(__file__, 215, 23), '**', t_327333, int_327334)
    
    # Applying the binary operator '*' (line 215)
    result_mul_327336 = python_operator(stypy.reporting.localization.Localization(__file__, 215, 19), '*', int_327332, result_pow_327335)
    
    float_327337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 32), 'float')
    # Applying the binary operator '-' (line 215)
    result_sub_327338 = python_operator(stypy.reporting.localization.Localization(__file__, 215, 19), '-', result_mul_327336, float_327337)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 213, 18), list_327323, result_sub_327338)
    
    # Processing the call keyword arguments (line 213)
    kwargs_327339 = {}
    # Getting the type of 'np' (line 213)
    np_327321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 9), 'np', False)
    # Obtaining the member 'array' of a type (line 213)
    array_327322 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 9), np_327321, 'array')
    # Calling array(args, kwargs) (line 213)
    array_call_result_327340 = invoke(stypy.reporting.localization.Localization(__file__, 213, 9), array_327322, *[list_327323], **kwargs_327339)
    
    # Assigning a type to the variable 'dx' (line 213)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 4), 'dx', array_call_result_327340)
    
    # Assigning a Call to a Name (line 216):
    
    # Call to array(...): (line 216)
    # Processing the call arguments (line 216)
    
    # Obtaining an instance of the builtin type 'list' (line 216)
    list_327343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 216)
    # Adding element type (line 216)
    
    # Call to zeros_like(...): (line 216)
    # Processing the call arguments (line 216)
    # Getting the type of 't' (line 216)
    t_327346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 34), 't', False)
    # Processing the call keyword arguments (line 216)
    kwargs_327347 = {}
    # Getting the type of 'np' (line 216)
    np_327344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 20), 'np', False)
    # Obtaining the member 'zeros_like' of a type (line 216)
    zeros_like_327345 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 20), np_327344, 'zeros_like')
    # Calling zeros_like(args, kwargs) (line 216)
    zeros_like_call_result_327348 = invoke(stypy.reporting.localization.Localization(__file__, 216, 20), zeros_like_327345, *[t_327346], **kwargs_327347)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 216, 19), list_327343, zeros_like_call_result_327348)
    # Adding element type (line 216)
    int_327349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 20), 'int')
    
    # Call to ones_like(...): (line 217)
    # Processing the call arguments (line 217)
    # Getting the type of 't' (line 217)
    t_327352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 37), 't', False)
    # Processing the call keyword arguments (line 217)
    kwargs_327353 = {}
    # Getting the type of 'np' (line 217)
    np_327350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 24), 'np', False)
    # Obtaining the member 'ones_like' of a type (line 217)
    ones_like_327351 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 24), np_327350, 'ones_like')
    # Calling ones_like(args, kwargs) (line 217)
    ones_like_call_result_327354 = invoke(stypy.reporting.localization.Localization(__file__, 217, 24), ones_like_327351, *[t_327352], **kwargs_327353)
    
    # Applying the binary operator '*' (line 217)
    result_mul_327355 = python_operator(stypy.reporting.localization.Localization(__file__, 217, 20), '*', int_327349, ones_like_call_result_327354)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 216, 19), list_327343, result_mul_327355)
    # Adding element type (line 216)
    int_327356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 20), 'int')
    # Getting the type of 't' (line 218)
    t_327357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 24), 't', False)
    # Applying the binary operator '*' (line 218)
    result_mul_327358 = python_operator(stypy.reporting.localization.Localization(__file__, 218, 20), '*', int_327356, t_327357)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 216, 19), list_327343, result_mul_327358)
    
    # Processing the call keyword arguments (line 216)
    kwargs_327359 = {}
    # Getting the type of 'np' (line 216)
    np_327341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 10), 'np', False)
    # Obtaining the member 'array' of a type (line 216)
    array_327342 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 10), np_327341, 'array')
    # Calling array(args, kwargs) (line 216)
    array_call_result_327360 = invoke(stypy.reporting.localization.Localization(__file__, 216, 10), array_327342, *[list_327343], **kwargs_327359)
    
    # Assigning a type to the variable 'd2x' (line 216)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 4), 'd2x', array_call_result_327360)
    
    # Assigning a Num to a Name (line 220):
    int_327361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 20), 'int')
    # Assigning a type to the variable 'window_length' (line 220)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 4), 'window_length', int_327361)
    
    # Assigning a Call to a Name (line 222):
    
    # Call to savgol_filter(...): (line 222)
    # Processing the call arguments (line 222)
    # Getting the type of 'x' (line 222)
    x_327363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 22), 'x', False)
    # Getting the type of 'window_length' (line 222)
    window_length_327364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 25), 'window_length', False)
    int_327365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 40), 'int')
    # Processing the call keyword arguments (line 222)
    int_327366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 48), 'int')
    keyword_327367 = int_327366
    str_327368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 57), 'str', 'interp')
    keyword_327369 = str_327368
    kwargs_327370 = {'mode': keyword_327369, 'axis': keyword_327367}
    # Getting the type of 'savgol_filter' (line 222)
    savgol_filter_327362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 8), 'savgol_filter', False)
    # Calling savgol_filter(args, kwargs) (line 222)
    savgol_filter_call_result_327371 = invoke(stypy.reporting.localization.Localization(__file__, 222, 8), savgol_filter_327362, *[x_327363, window_length_327364, int_327365], **kwargs_327370)
    
    # Assigning a type to the variable 'y' (line 222)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 4), 'y', savgol_filter_call_result_327371)
    
    # Call to assert_allclose(...): (line 223)
    # Processing the call arguments (line 223)
    # Getting the type of 'y' (line 223)
    y_327373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 20), 'y', False)
    # Getting the type of 'x' (line 223)
    x_327374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 23), 'x', False)
    # Processing the call keyword arguments (line 223)
    float_327375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 31), 'float')
    keyword_327376 = float_327375
    kwargs_327377 = {'atol': keyword_327376}
    # Getting the type of 'assert_allclose' (line 223)
    assert_allclose_327372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 223)
    assert_allclose_call_result_327378 = invoke(stypy.reporting.localization.Localization(__file__, 223, 4), assert_allclose_327372, *[y_327373, x_327374], **kwargs_327377)
    
    
    # Assigning a Call to a Name (line 225):
    
    # Call to savgol_filter(...): (line 225)
    # Processing the call arguments (line 225)
    # Getting the type of 'x' (line 225)
    x_327380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 23), 'x', False)
    # Getting the type of 'window_length' (line 225)
    window_length_327381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 26), 'window_length', False)
    int_327382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 41), 'int')
    # Processing the call keyword arguments (line 225)
    int_327383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 49), 'int')
    keyword_327384 = int_327383
    str_327385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 58), 'str', 'interp')
    keyword_327386 = str_327385
    int_327387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 29), 'int')
    keyword_327388 = int_327387
    # Getting the type of 'delta' (line 226)
    delta_327389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 38), 'delta', False)
    keyword_327390 = delta_327389
    kwargs_327391 = {'deriv': keyword_327388, 'delta': keyword_327390, 'mode': keyword_327386, 'axis': keyword_327384}
    # Getting the type of 'savgol_filter' (line 225)
    savgol_filter_327379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 9), 'savgol_filter', False)
    # Calling savgol_filter(args, kwargs) (line 225)
    savgol_filter_call_result_327392 = invoke(stypy.reporting.localization.Localization(__file__, 225, 9), savgol_filter_327379, *[x_327380, window_length_327381, int_327382], **kwargs_327391)
    
    # Assigning a type to the variable 'y1' (line 225)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 4), 'y1', savgol_filter_call_result_327392)
    
    # Call to assert_allclose(...): (line 227)
    # Processing the call arguments (line 227)
    # Getting the type of 'y1' (line 227)
    y1_327394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 20), 'y1', False)
    # Getting the type of 'dx' (line 227)
    dx_327395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 24), 'dx', False)
    # Processing the call keyword arguments (line 227)
    float_327396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 33), 'float')
    keyword_327397 = float_327396
    kwargs_327398 = {'atol': keyword_327397}
    # Getting the type of 'assert_allclose' (line 227)
    assert_allclose_327393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 227)
    assert_allclose_call_result_327399 = invoke(stypy.reporting.localization.Localization(__file__, 227, 4), assert_allclose_327393, *[y1_327394, dx_327395], **kwargs_327398)
    
    
    # Assigning a Call to a Name (line 229):
    
    # Call to savgol_filter(...): (line 229)
    # Processing the call arguments (line 229)
    # Getting the type of 'x' (line 229)
    x_327401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 23), 'x', False)
    # Getting the type of 'window_length' (line 229)
    window_length_327402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 26), 'window_length', False)
    int_327403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 41), 'int')
    # Processing the call keyword arguments (line 229)
    int_327404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 49), 'int')
    keyword_327405 = int_327404
    str_327406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 58), 'str', 'interp')
    keyword_327407 = str_327406
    int_327408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 29), 'int')
    keyword_327409 = int_327408
    # Getting the type of 'delta' (line 230)
    delta_327410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 38), 'delta', False)
    keyword_327411 = delta_327410
    kwargs_327412 = {'deriv': keyword_327409, 'delta': keyword_327411, 'mode': keyword_327407, 'axis': keyword_327405}
    # Getting the type of 'savgol_filter' (line 229)
    savgol_filter_327400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 9), 'savgol_filter', False)
    # Calling savgol_filter(args, kwargs) (line 229)
    savgol_filter_call_result_327413 = invoke(stypy.reporting.localization.Localization(__file__, 229, 9), savgol_filter_327400, *[x_327401, window_length_327402, int_327403], **kwargs_327412)
    
    # Assigning a type to the variable 'y2' (line 229)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 4), 'y2', savgol_filter_call_result_327413)
    
    # Call to assert_allclose(...): (line 231)
    # Processing the call arguments (line 231)
    # Getting the type of 'y2' (line 231)
    y2_327415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 20), 'y2', False)
    # Getting the type of 'd2x' (line 231)
    d2x_327416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 24), 'd2x', False)
    # Processing the call keyword arguments (line 231)
    float_327417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 34), 'float')
    keyword_327418 = float_327417
    kwargs_327419 = {'atol': keyword_327418}
    # Getting the type of 'assert_allclose' (line 231)
    assert_allclose_327414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 231)
    assert_allclose_call_result_327420 = invoke(stypy.reporting.localization.Localization(__file__, 231, 4), assert_allclose_327414, *[y2_327415, d2x_327416], **kwargs_327419)
    
    
    # Assigning a Attribute to a Name (line 235):
    # Getting the type of 'x' (line 235)
    x_327421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 8), 'x')
    # Obtaining the member 'T' of a type (line 235)
    T_327422 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 8), x_327421, 'T')
    # Assigning a type to the variable 'x' (line 235)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 4), 'x', T_327422)
    
    # Assigning a Attribute to a Name (line 236):
    # Getting the type of 'dx' (line 236)
    dx_327423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 9), 'dx')
    # Obtaining the member 'T' of a type (line 236)
    T_327424 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 9), dx_327423, 'T')
    # Assigning a type to the variable 'dx' (line 236)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 4), 'dx', T_327424)
    
    # Assigning a Attribute to a Name (line 237):
    # Getting the type of 'd2x' (line 237)
    d2x_327425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 10), 'd2x')
    # Obtaining the member 'T' of a type (line 237)
    T_327426 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 10), d2x_327425, 'T')
    # Assigning a type to the variable 'd2x' (line 237)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 4), 'd2x', T_327426)
    
    # Assigning a Call to a Name (line 239):
    
    # Call to savgol_filter(...): (line 239)
    # Processing the call arguments (line 239)
    # Getting the type of 'x' (line 239)
    x_327428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 22), 'x', False)
    # Getting the type of 'window_length' (line 239)
    window_length_327429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 25), 'window_length', False)
    int_327430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 40), 'int')
    # Processing the call keyword arguments (line 239)
    int_327431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 48), 'int')
    keyword_327432 = int_327431
    str_327433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 56), 'str', 'interp')
    keyword_327434 = str_327433
    kwargs_327435 = {'mode': keyword_327434, 'axis': keyword_327432}
    # Getting the type of 'savgol_filter' (line 239)
    savgol_filter_327427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 8), 'savgol_filter', False)
    # Calling savgol_filter(args, kwargs) (line 239)
    savgol_filter_call_result_327436 = invoke(stypy.reporting.localization.Localization(__file__, 239, 8), savgol_filter_327427, *[x_327428, window_length_327429, int_327430], **kwargs_327435)
    
    # Assigning a type to the variable 'y' (line 239)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 4), 'y', savgol_filter_call_result_327436)
    
    # Call to assert_allclose(...): (line 240)
    # Processing the call arguments (line 240)
    # Getting the type of 'y' (line 240)
    y_327438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 20), 'y', False)
    # Getting the type of 'x' (line 240)
    x_327439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 23), 'x', False)
    # Processing the call keyword arguments (line 240)
    float_327440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 31), 'float')
    keyword_327441 = float_327440
    kwargs_327442 = {'atol': keyword_327441}
    # Getting the type of 'assert_allclose' (line 240)
    assert_allclose_327437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 240)
    assert_allclose_call_result_327443 = invoke(stypy.reporting.localization.Localization(__file__, 240, 4), assert_allclose_327437, *[y_327438, x_327439], **kwargs_327442)
    
    
    # Assigning a Call to a Name (line 242):
    
    # Call to savgol_filter(...): (line 242)
    # Processing the call arguments (line 242)
    # Getting the type of 'x' (line 242)
    x_327445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 23), 'x', False)
    # Getting the type of 'window_length' (line 242)
    window_length_327446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 26), 'window_length', False)
    int_327447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 41), 'int')
    # Processing the call keyword arguments (line 242)
    int_327448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 49), 'int')
    keyword_327449 = int_327448
    str_327450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 57), 'str', 'interp')
    keyword_327451 = str_327450
    int_327452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 29), 'int')
    keyword_327453 = int_327452
    # Getting the type of 'delta' (line 243)
    delta_327454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 38), 'delta', False)
    keyword_327455 = delta_327454
    kwargs_327456 = {'deriv': keyword_327453, 'delta': keyword_327455, 'mode': keyword_327451, 'axis': keyword_327449}
    # Getting the type of 'savgol_filter' (line 242)
    savgol_filter_327444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 9), 'savgol_filter', False)
    # Calling savgol_filter(args, kwargs) (line 242)
    savgol_filter_call_result_327457 = invoke(stypy.reporting.localization.Localization(__file__, 242, 9), savgol_filter_327444, *[x_327445, window_length_327446, int_327447], **kwargs_327456)
    
    # Assigning a type to the variable 'y1' (line 242)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 4), 'y1', savgol_filter_call_result_327457)
    
    # Call to assert_allclose(...): (line 244)
    # Processing the call arguments (line 244)
    # Getting the type of 'y1' (line 244)
    y1_327459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 20), 'y1', False)
    # Getting the type of 'dx' (line 244)
    dx_327460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 24), 'dx', False)
    # Processing the call keyword arguments (line 244)
    float_327461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 33), 'float')
    keyword_327462 = float_327461
    kwargs_327463 = {'atol': keyword_327462}
    # Getting the type of 'assert_allclose' (line 244)
    assert_allclose_327458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 244)
    assert_allclose_call_result_327464 = invoke(stypy.reporting.localization.Localization(__file__, 244, 4), assert_allclose_327458, *[y1_327459, dx_327460], **kwargs_327463)
    
    
    # Assigning a Call to a Name (line 246):
    
    # Call to savgol_filter(...): (line 246)
    # Processing the call arguments (line 246)
    # Getting the type of 'x' (line 246)
    x_327466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 23), 'x', False)
    # Getting the type of 'window_length' (line 246)
    window_length_327467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 26), 'window_length', False)
    int_327468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 41), 'int')
    # Processing the call keyword arguments (line 246)
    int_327469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 49), 'int')
    keyword_327470 = int_327469
    str_327471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 57), 'str', 'interp')
    keyword_327472 = str_327471
    int_327473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 29), 'int')
    keyword_327474 = int_327473
    # Getting the type of 'delta' (line 247)
    delta_327475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 38), 'delta', False)
    keyword_327476 = delta_327475
    kwargs_327477 = {'deriv': keyword_327474, 'delta': keyword_327476, 'mode': keyword_327472, 'axis': keyword_327470}
    # Getting the type of 'savgol_filter' (line 246)
    savgol_filter_327465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 9), 'savgol_filter', False)
    # Calling savgol_filter(args, kwargs) (line 246)
    savgol_filter_call_result_327478 = invoke(stypy.reporting.localization.Localization(__file__, 246, 9), savgol_filter_327465, *[x_327466, window_length_327467, int_327468], **kwargs_327477)
    
    # Assigning a type to the variable 'y2' (line 246)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 4), 'y2', savgol_filter_call_result_327478)
    
    # Call to assert_allclose(...): (line 248)
    # Processing the call arguments (line 248)
    # Getting the type of 'y2' (line 248)
    y2_327480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 20), 'y2', False)
    # Getting the type of 'd2x' (line 248)
    d2x_327481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 24), 'd2x', False)
    # Processing the call keyword arguments (line 248)
    float_327482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 34), 'float')
    keyword_327483 = float_327482
    kwargs_327484 = {'atol': keyword_327483}
    # Getting the type of 'assert_allclose' (line 248)
    assert_allclose_327479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 248)
    assert_allclose_call_result_327485 = invoke(stypy.reporting.localization.Localization(__file__, 248, 4), assert_allclose_327479, *[y2_327480, d2x_327481], **kwargs_327484)
    
    
    # ################# End of 'test_sg_filter_interp_edges(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_sg_filter_interp_edges' in the type store
    # Getting the type of 'stypy_return_type' (line 202)
    stypy_return_type_327486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_327486)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_sg_filter_interp_edges'
    return stypy_return_type_327486

# Assigning a type to the variable 'test_sg_filter_interp_edges' (line 202)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 0), 'test_sg_filter_interp_edges', test_sg_filter_interp_edges)

@norecursion
def test_sg_filter_interp_edges_3d(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_sg_filter_interp_edges_3d'
    module_type_store = module_type_store.open_function_context('test_sg_filter_interp_edges_3d', 251, 0, False)
    
    # Passed parameters checking function
    test_sg_filter_interp_edges_3d.stypy_localization = localization
    test_sg_filter_interp_edges_3d.stypy_type_of_self = None
    test_sg_filter_interp_edges_3d.stypy_type_store = module_type_store
    test_sg_filter_interp_edges_3d.stypy_function_name = 'test_sg_filter_interp_edges_3d'
    test_sg_filter_interp_edges_3d.stypy_param_names_list = []
    test_sg_filter_interp_edges_3d.stypy_varargs_param_name = None
    test_sg_filter_interp_edges_3d.stypy_kwargs_param_name = None
    test_sg_filter_interp_edges_3d.stypy_call_defaults = defaults
    test_sg_filter_interp_edges_3d.stypy_call_varargs = varargs
    test_sg_filter_interp_edges_3d.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_sg_filter_interp_edges_3d', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_sg_filter_interp_edges_3d', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_sg_filter_interp_edges_3d(...)' code ##################

    
    # Assigning a Call to a Name (line 253):
    
    # Call to linspace(...): (line 253)
    # Processing the call arguments (line 253)
    int_327489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 20), 'int')
    int_327490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 24), 'int')
    int_327491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 27), 'int')
    # Processing the call keyword arguments (line 253)
    kwargs_327492 = {}
    # Getting the type of 'np' (line 253)
    np_327487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 8), 'np', False)
    # Obtaining the member 'linspace' of a type (line 253)
    linspace_327488 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 8), np_327487, 'linspace')
    # Calling linspace(args, kwargs) (line 253)
    linspace_call_result_327493 = invoke(stypy.reporting.localization.Localization(__file__, 253, 8), linspace_327488, *[int_327489, int_327490, int_327491], **kwargs_327492)
    
    # Assigning a type to the variable 't' (line 253)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 4), 't', linspace_call_result_327493)
    
    # Assigning a BinOp to a Name (line 254):
    
    # Obtaining the type of the subscript
    int_327494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 14), 'int')
    # Getting the type of 't' (line 254)
    t_327495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 12), 't')
    # Obtaining the member '__getitem__' of a type (line 254)
    getitem___327496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 12), t_327495, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 254)
    subscript_call_result_327497 = invoke(stypy.reporting.localization.Localization(__file__, 254, 12), getitem___327496, int_327494)
    
    
    # Obtaining the type of the subscript
    int_327498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 21), 'int')
    # Getting the type of 't' (line 254)
    t_327499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 19), 't')
    # Obtaining the member '__getitem__' of a type (line 254)
    getitem___327500 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 19), t_327499, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 254)
    subscript_call_result_327501 = invoke(stypy.reporting.localization.Localization(__file__, 254, 19), getitem___327500, int_327498)
    
    # Applying the binary operator '-' (line 254)
    result_sub_327502 = python_operator(stypy.reporting.localization.Localization(__file__, 254, 12), '-', subscript_call_result_327497, subscript_call_result_327501)
    
    # Assigning a type to the variable 'delta' (line 254)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 4), 'delta', result_sub_327502)
    
    # Assigning a Call to a Name (line 255):
    
    # Call to array(...): (line 255)
    # Processing the call arguments (line 255)
    
    # Obtaining an instance of the builtin type 'list' (line 255)
    list_327505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 255)
    # Adding element type (line 255)
    # Getting the type of 't' (line 255)
    t_327506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 19), 't', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 255, 18), list_327505, t_327506)
    # Adding element type (line 255)
    
    # Getting the type of 't' (line 255)
    t_327507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 23), 't', False)
    # Applying the 'usub' unary operator (line 255)
    result___neg___327508 = python_operator(stypy.reporting.localization.Localization(__file__, 255, 22), 'usub', t_327507)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 255, 18), list_327505, result___neg___327508)
    
    # Processing the call keyword arguments (line 255)
    kwargs_327509 = {}
    # Getting the type of 'np' (line 255)
    np_327503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 9), 'np', False)
    # Obtaining the member 'array' of a type (line 255)
    array_327504 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 9), np_327503, 'array')
    # Calling array(args, kwargs) (line 255)
    array_call_result_327510 = invoke(stypy.reporting.localization.Localization(__file__, 255, 9), array_327504, *[list_327505], **kwargs_327509)
    
    # Assigning a type to the variable 'x1' (line 255)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 4), 'x1', array_call_result_327510)
    
    # Assigning a Call to a Name (line 256):
    
    # Call to array(...): (line 256)
    # Processing the call arguments (line 256)
    
    # Obtaining an instance of the builtin type 'list' (line 256)
    list_327513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 256)
    # Adding element type (line 256)
    # Getting the type of 't' (line 256)
    t_327514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 19), 't', False)
    int_327515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 24), 'int')
    # Applying the binary operator '**' (line 256)
    result_pow_327516 = python_operator(stypy.reporting.localization.Localization(__file__, 256, 19), '**', t_327514, int_327515)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 256, 18), list_327513, result_pow_327516)
    # Adding element type (line 256)
    int_327517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 27), 'int')
    # Getting the type of 't' (line 256)
    t_327518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 31), 't', False)
    int_327519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 36), 'int')
    # Applying the binary operator '**' (line 256)
    result_pow_327520 = python_operator(stypy.reporting.localization.Localization(__file__, 256, 31), '**', t_327518, int_327519)
    
    # Applying the binary operator '*' (line 256)
    result_mul_327521 = python_operator(stypy.reporting.localization.Localization(__file__, 256, 27), '*', int_327517, result_pow_327520)
    
    int_327522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 40), 'int')
    # Applying the binary operator '+' (line 256)
    result_add_327523 = python_operator(stypy.reporting.localization.Localization(__file__, 256, 27), '+', result_mul_327521, int_327522)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 256, 18), list_327513, result_add_327523)
    
    # Processing the call keyword arguments (line 256)
    kwargs_327524 = {}
    # Getting the type of 'np' (line 256)
    np_327511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 9), 'np', False)
    # Obtaining the member 'array' of a type (line 256)
    array_327512 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 9), np_327511, 'array')
    # Calling array(args, kwargs) (line 256)
    array_call_result_327525 = invoke(stypy.reporting.localization.Localization(__file__, 256, 9), array_327512, *[list_327513], **kwargs_327524)
    
    # Assigning a type to the variable 'x2' (line 256)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 4), 'x2', array_call_result_327525)
    
    # Assigning a Call to a Name (line 257):
    
    # Call to array(...): (line 257)
    # Processing the call arguments (line 257)
    
    # Obtaining an instance of the builtin type 'list' (line 257)
    list_327528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 257)
    # Adding element type (line 257)
    # Getting the type of 't' (line 257)
    t_327529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 19), 't', False)
    int_327530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 24), 'int')
    # Applying the binary operator '**' (line 257)
    result_pow_327531 = python_operator(stypy.reporting.localization.Localization(__file__, 257, 19), '**', t_327529, int_327530)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 18), list_327528, result_pow_327531)
    # Adding element type (line 257)
    int_327532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 27), 'int')
    # Getting the type of 't' (line 257)
    t_327533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 31), 't', False)
    int_327534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 36), 'int')
    # Applying the binary operator '**' (line 257)
    result_pow_327535 = python_operator(stypy.reporting.localization.Localization(__file__, 257, 31), '**', t_327533, int_327534)
    
    # Applying the binary operator '*' (line 257)
    result_mul_327536 = python_operator(stypy.reporting.localization.Localization(__file__, 257, 27), '*', int_327532, result_pow_327535)
    
    # Getting the type of 't' (line 257)
    t_327537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 40), 't', False)
    int_327538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 45), 'int')
    # Applying the binary operator '**' (line 257)
    result_pow_327539 = python_operator(stypy.reporting.localization.Localization(__file__, 257, 40), '**', t_327537, int_327538)
    
    # Applying the binary operator '+' (line 257)
    result_add_327540 = python_operator(stypy.reporting.localization.Localization(__file__, 257, 27), '+', result_mul_327536, result_pow_327539)
    
    float_327541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 49), 'float')
    # Getting the type of 't' (line 257)
    t_327542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 55), 't', False)
    # Applying the binary operator '*' (line 257)
    result_mul_327543 = python_operator(stypy.reporting.localization.Localization(__file__, 257, 49), '*', float_327541, t_327542)
    
    # Applying the binary operator '-' (line 257)
    result_sub_327544 = python_operator(stypy.reporting.localization.Localization(__file__, 257, 47), '-', result_add_327540, result_mul_327543)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 18), list_327528, result_sub_327544)
    
    # Processing the call keyword arguments (line 257)
    kwargs_327545 = {}
    # Getting the type of 'np' (line 257)
    np_327526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 9), 'np', False)
    # Obtaining the member 'array' of a type (line 257)
    array_327527 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 9), np_327526, 'array')
    # Calling array(args, kwargs) (line 257)
    array_call_result_327546 = invoke(stypy.reporting.localization.Localization(__file__, 257, 9), array_327527, *[list_327528], **kwargs_327545)
    
    # Assigning a type to the variable 'x3' (line 257)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 4), 'x3', array_call_result_327546)
    
    # Assigning a Call to a Name (line 258):
    
    # Call to array(...): (line 258)
    # Processing the call arguments (line 258)
    
    # Obtaining an instance of the builtin type 'list' (line 258)
    list_327549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 258)
    # Adding element type (line 258)
    
    # Call to ones_like(...): (line 258)
    # Processing the call arguments (line 258)
    # Getting the type of 't' (line 258)
    t_327552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 33), 't', False)
    # Processing the call keyword arguments (line 258)
    kwargs_327553 = {}
    # Getting the type of 'np' (line 258)
    np_327550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 20), 'np', False)
    # Obtaining the member 'ones_like' of a type (line 258)
    ones_like_327551 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 20), np_327550, 'ones_like')
    # Calling ones_like(args, kwargs) (line 258)
    ones_like_call_result_327554 = invoke(stypy.reporting.localization.Localization(__file__, 258, 20), ones_like_327551, *[t_327552], **kwargs_327553)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 19), list_327549, ones_like_call_result_327554)
    # Adding element type (line 258)
    
    
    # Call to ones_like(...): (line 258)
    # Processing the call arguments (line 258)
    # Getting the type of 't' (line 258)
    t_327557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 51), 't', False)
    # Processing the call keyword arguments (line 258)
    kwargs_327558 = {}
    # Getting the type of 'np' (line 258)
    np_327555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 38), 'np', False)
    # Obtaining the member 'ones_like' of a type (line 258)
    ones_like_327556 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 38), np_327555, 'ones_like')
    # Calling ones_like(args, kwargs) (line 258)
    ones_like_call_result_327559 = invoke(stypy.reporting.localization.Localization(__file__, 258, 38), ones_like_327556, *[t_327557], **kwargs_327558)
    
    # Applying the 'usub' unary operator (line 258)
    result___neg___327560 = python_operator(stypy.reporting.localization.Localization(__file__, 258, 37), 'usub', ones_like_call_result_327559)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 19), list_327549, result___neg___327560)
    
    # Processing the call keyword arguments (line 258)
    kwargs_327561 = {}
    # Getting the type of 'np' (line 258)
    np_327547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 10), 'np', False)
    # Obtaining the member 'array' of a type (line 258)
    array_327548 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 10), np_327547, 'array')
    # Calling array(args, kwargs) (line 258)
    array_call_result_327562 = invoke(stypy.reporting.localization.Localization(__file__, 258, 10), array_327548, *[list_327549], **kwargs_327561)
    
    # Assigning a type to the variable 'dx1' (line 258)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 4), 'dx1', array_call_result_327562)
    
    # Assigning a Call to a Name (line 259):
    
    # Call to array(...): (line 259)
    # Processing the call arguments (line 259)
    
    # Obtaining an instance of the builtin type 'list' (line 259)
    list_327565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 259)
    # Adding element type (line 259)
    int_327566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 20), 'int')
    # Getting the type of 't' (line 259)
    t_327567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 24), 't', False)
    # Applying the binary operator '*' (line 259)
    result_mul_327568 = python_operator(stypy.reporting.localization.Localization(__file__, 259, 20), '*', int_327566, t_327567)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 19), list_327565, result_mul_327568)
    # Adding element type (line 259)
    int_327569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 27), 'int')
    # Getting the type of 't' (line 259)
    t_327570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 31), 't', False)
    # Applying the binary operator '*' (line 259)
    result_mul_327571 = python_operator(stypy.reporting.localization.Localization(__file__, 259, 27), '*', int_327569, t_327570)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 19), list_327565, result_mul_327571)
    
    # Processing the call keyword arguments (line 259)
    kwargs_327572 = {}
    # Getting the type of 'np' (line 259)
    np_327563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 10), 'np', False)
    # Obtaining the member 'array' of a type (line 259)
    array_327564 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 10), np_327563, 'array')
    # Calling array(args, kwargs) (line 259)
    array_call_result_327573 = invoke(stypy.reporting.localization.Localization(__file__, 259, 10), array_327564, *[list_327565], **kwargs_327572)
    
    # Assigning a type to the variable 'dx2' (line 259)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 4), 'dx2', array_call_result_327573)
    
    # Assigning a Call to a Name (line 260):
    
    # Call to array(...): (line 260)
    # Processing the call arguments (line 260)
    
    # Obtaining an instance of the builtin type 'list' (line 260)
    list_327576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 260)
    # Adding element type (line 260)
    int_327577 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 20), 'int')
    # Getting the type of 't' (line 260)
    t_327578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 24), 't', False)
    int_327579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 29), 'int')
    # Applying the binary operator '**' (line 260)
    result_pow_327580 = python_operator(stypy.reporting.localization.Localization(__file__, 260, 24), '**', t_327578, int_327579)
    
    # Applying the binary operator '*' (line 260)
    result_mul_327581 = python_operator(stypy.reporting.localization.Localization(__file__, 260, 20), '*', int_327577, result_pow_327580)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 260, 19), list_327576, result_mul_327581)
    # Adding element type (line 260)
    int_327582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 32), 'int')
    # Getting the type of 't' (line 260)
    t_327583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 36), 't', False)
    int_327584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 41), 'int')
    # Applying the binary operator '**' (line 260)
    result_pow_327585 = python_operator(stypy.reporting.localization.Localization(__file__, 260, 36), '**', t_327583, int_327584)
    
    # Applying the binary operator '*' (line 260)
    result_mul_327586 = python_operator(stypy.reporting.localization.Localization(__file__, 260, 32), '*', int_327582, result_pow_327585)
    
    int_327587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 45), 'int')
    # Getting the type of 't' (line 260)
    t_327588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 49), 't', False)
    # Applying the binary operator '*' (line 260)
    result_mul_327589 = python_operator(stypy.reporting.localization.Localization(__file__, 260, 45), '*', int_327587, t_327588)
    
    # Applying the binary operator '+' (line 260)
    result_add_327590 = python_operator(stypy.reporting.localization.Localization(__file__, 260, 32), '+', result_mul_327586, result_mul_327589)
    
    float_327591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 53), 'float')
    # Applying the binary operator '-' (line 260)
    result_sub_327592 = python_operator(stypy.reporting.localization.Localization(__file__, 260, 51), '-', result_add_327590, float_327591)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 260, 19), list_327576, result_sub_327592)
    
    # Processing the call keyword arguments (line 260)
    kwargs_327593 = {}
    # Getting the type of 'np' (line 260)
    np_327574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 10), 'np', False)
    # Obtaining the member 'array' of a type (line 260)
    array_327575 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 10), np_327574, 'array')
    # Calling array(args, kwargs) (line 260)
    array_call_result_327594 = invoke(stypy.reporting.localization.Localization(__file__, 260, 10), array_327575, *[list_327576], **kwargs_327593)
    
    # Assigning a type to the variable 'dx3' (line 260)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 4), 'dx3', array_call_result_327594)
    
    # Assigning a Call to a Name (line 263):
    
    # Call to array(...): (line 263)
    # Processing the call arguments (line 263)
    
    # Obtaining an instance of the builtin type 'list' (line 263)
    list_327597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 263)
    # Adding element type (line 263)
    # Getting the type of 'x1' (line 263)
    x1_327598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 18), 'x1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 263, 17), list_327597, x1_327598)
    # Adding element type (line 263)
    # Getting the type of 'x2' (line 263)
    x2_327599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 22), 'x2', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 263, 17), list_327597, x2_327599)
    # Adding element type (line 263)
    # Getting the type of 'x3' (line 263)
    x3_327600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 26), 'x3', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 263, 17), list_327597, x3_327600)
    
    # Processing the call keyword arguments (line 263)
    kwargs_327601 = {}
    # Getting the type of 'np' (line 263)
    np_327595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 8), 'np', False)
    # Obtaining the member 'array' of a type (line 263)
    array_327596 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 8), np_327595, 'array')
    # Calling array(args, kwargs) (line 263)
    array_call_result_327602 = invoke(stypy.reporting.localization.Localization(__file__, 263, 8), array_327596, *[list_327597], **kwargs_327601)
    
    # Assigning a type to the variable 'z' (line 263)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 4), 'z', array_call_result_327602)
    
    # Assigning a Call to a Name (line 264):
    
    # Call to array(...): (line 264)
    # Processing the call arguments (line 264)
    
    # Obtaining an instance of the builtin type 'list' (line 264)
    list_327605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 264)
    # Adding element type (line 264)
    # Getting the type of 'dx1' (line 264)
    dx1_327606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 19), 'dx1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 264, 18), list_327605, dx1_327606)
    # Adding element type (line 264)
    # Getting the type of 'dx2' (line 264)
    dx2_327607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 24), 'dx2', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 264, 18), list_327605, dx2_327607)
    # Adding element type (line 264)
    # Getting the type of 'dx3' (line 264)
    dx3_327608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 29), 'dx3', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 264, 18), list_327605, dx3_327608)
    
    # Processing the call keyword arguments (line 264)
    kwargs_327609 = {}
    # Getting the type of 'np' (line 264)
    np_327603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 9), 'np', False)
    # Obtaining the member 'array' of a type (line 264)
    array_327604 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 9), np_327603, 'array')
    # Calling array(args, kwargs) (line 264)
    array_call_result_327610 = invoke(stypy.reporting.localization.Localization(__file__, 264, 9), array_327604, *[list_327605], **kwargs_327609)
    
    # Assigning a type to the variable 'dz' (line 264)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 4), 'dz', array_call_result_327610)
    
    # Assigning a Call to a Name (line 266):
    
    # Call to savgol_filter(...): (line 266)
    # Processing the call arguments (line 266)
    # Getting the type of 'z' (line 266)
    z_327612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 22), 'z', False)
    int_327613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 25), 'int')
    int_327614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 28), 'int')
    # Processing the call keyword arguments (line 266)
    int_327615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 36), 'int')
    keyword_327616 = int_327615
    str_327617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 45), 'str', 'interp')
    keyword_327618 = str_327617
    # Getting the type of 'delta' (line 266)
    delta_327619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 61), 'delta', False)
    keyword_327620 = delta_327619
    kwargs_327621 = {'delta': keyword_327620, 'mode': keyword_327618, 'axis': keyword_327616}
    # Getting the type of 'savgol_filter' (line 266)
    savgol_filter_327611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 8), 'savgol_filter', False)
    # Calling savgol_filter(args, kwargs) (line 266)
    savgol_filter_call_result_327622 = invoke(stypy.reporting.localization.Localization(__file__, 266, 8), savgol_filter_327611, *[z_327612, int_327613, int_327614], **kwargs_327621)
    
    # Assigning a type to the variable 'y' (line 266)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 4), 'y', savgol_filter_call_result_327622)
    
    # Call to assert_allclose(...): (line 267)
    # Processing the call arguments (line 267)
    # Getting the type of 'y' (line 267)
    y_327624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 20), 'y', False)
    # Getting the type of 'z' (line 267)
    z_327625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 23), 'z', False)
    # Processing the call keyword arguments (line 267)
    float_327626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 31), 'float')
    keyword_327627 = float_327626
    kwargs_327628 = {'atol': keyword_327627}
    # Getting the type of 'assert_allclose' (line 267)
    assert_allclose_327623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 267)
    assert_allclose_call_result_327629 = invoke(stypy.reporting.localization.Localization(__file__, 267, 4), assert_allclose_327623, *[y_327624, z_327625], **kwargs_327628)
    
    
    # Assigning a Call to a Name (line 269):
    
    # Call to savgol_filter(...): (line 269)
    # Processing the call arguments (line 269)
    # Getting the type of 'z' (line 269)
    z_327631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 23), 'z', False)
    int_327632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 26), 'int')
    int_327633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 29), 'int')
    # Processing the call keyword arguments (line 269)
    int_327634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 37), 'int')
    keyword_327635 = int_327634
    str_327636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 46), 'str', 'interp')
    keyword_327637 = str_327636
    int_327638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 62), 'int')
    keyword_327639 = int_327638
    # Getting the type of 'delta' (line 269)
    delta_327640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 71), 'delta', False)
    keyword_327641 = delta_327640
    kwargs_327642 = {'deriv': keyword_327639, 'delta': keyword_327641, 'mode': keyword_327637, 'axis': keyword_327635}
    # Getting the type of 'savgol_filter' (line 269)
    savgol_filter_327630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 9), 'savgol_filter', False)
    # Calling savgol_filter(args, kwargs) (line 269)
    savgol_filter_call_result_327643 = invoke(stypy.reporting.localization.Localization(__file__, 269, 9), savgol_filter_327630, *[z_327631, int_327632, int_327633], **kwargs_327642)
    
    # Assigning a type to the variable 'dy' (line 269)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 4), 'dy', savgol_filter_call_result_327643)
    
    # Call to assert_allclose(...): (line 270)
    # Processing the call arguments (line 270)
    # Getting the type of 'dy' (line 270)
    dy_327645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 20), 'dy', False)
    # Getting the type of 'dz' (line 270)
    dz_327646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 24), 'dz', False)
    # Processing the call keyword arguments (line 270)
    float_327647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 33), 'float')
    keyword_327648 = float_327647
    kwargs_327649 = {'atol': keyword_327648}
    # Getting the type of 'assert_allclose' (line 270)
    assert_allclose_327644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 270)
    assert_allclose_call_result_327650 = invoke(stypy.reporting.localization.Localization(__file__, 270, 4), assert_allclose_327644, *[dy_327645, dz_327646], **kwargs_327649)
    
    
    # Assigning a Call to a Name (line 273):
    
    # Call to array(...): (line 273)
    # Processing the call arguments (line 273)
    
    # Obtaining an instance of the builtin type 'list' (line 273)
    list_327653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 273)
    # Adding element type (line 273)
    # Getting the type of 'x1' (line 273)
    x1_327654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 18), 'x1', False)
    # Obtaining the member 'T' of a type (line 273)
    T_327655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 18), x1_327654, 'T')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 273, 17), list_327653, T_327655)
    # Adding element type (line 273)
    # Getting the type of 'x2' (line 273)
    x2_327656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 24), 'x2', False)
    # Obtaining the member 'T' of a type (line 273)
    T_327657 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 24), x2_327656, 'T')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 273, 17), list_327653, T_327657)
    # Adding element type (line 273)
    # Getting the type of 'x3' (line 273)
    x3_327658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 30), 'x3', False)
    # Obtaining the member 'T' of a type (line 273)
    T_327659 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 30), x3_327658, 'T')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 273, 17), list_327653, T_327659)
    
    # Processing the call keyword arguments (line 273)
    kwargs_327660 = {}
    # Getting the type of 'np' (line 273)
    np_327651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 8), 'np', False)
    # Obtaining the member 'array' of a type (line 273)
    array_327652 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 8), np_327651, 'array')
    # Calling array(args, kwargs) (line 273)
    array_call_result_327661 = invoke(stypy.reporting.localization.Localization(__file__, 273, 8), array_327652, *[list_327653], **kwargs_327660)
    
    # Assigning a type to the variable 'z' (line 273)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 4), 'z', array_call_result_327661)
    
    # Assigning a Call to a Name (line 274):
    
    # Call to array(...): (line 274)
    # Processing the call arguments (line 274)
    
    # Obtaining an instance of the builtin type 'list' (line 274)
    list_327664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 274)
    # Adding element type (line 274)
    # Getting the type of 'dx1' (line 274)
    dx1_327665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 19), 'dx1', False)
    # Obtaining the member 'T' of a type (line 274)
    T_327666 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 19), dx1_327665, 'T')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 274, 18), list_327664, T_327666)
    # Adding element type (line 274)
    # Getting the type of 'dx2' (line 274)
    dx2_327667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 26), 'dx2', False)
    # Obtaining the member 'T' of a type (line 274)
    T_327668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 26), dx2_327667, 'T')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 274, 18), list_327664, T_327668)
    # Adding element type (line 274)
    # Getting the type of 'dx3' (line 274)
    dx3_327669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 33), 'dx3', False)
    # Obtaining the member 'T' of a type (line 274)
    T_327670 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 33), dx3_327669, 'T')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 274, 18), list_327664, T_327670)
    
    # Processing the call keyword arguments (line 274)
    kwargs_327671 = {}
    # Getting the type of 'np' (line 274)
    np_327662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 9), 'np', False)
    # Obtaining the member 'array' of a type (line 274)
    array_327663 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 9), np_327662, 'array')
    # Calling array(args, kwargs) (line 274)
    array_call_result_327672 = invoke(stypy.reporting.localization.Localization(__file__, 274, 9), array_327663, *[list_327664], **kwargs_327671)
    
    # Assigning a type to the variable 'dz' (line 274)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 4), 'dz', array_call_result_327672)
    
    # Assigning a Call to a Name (line 276):
    
    # Call to savgol_filter(...): (line 276)
    # Processing the call arguments (line 276)
    # Getting the type of 'z' (line 276)
    z_327674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 22), 'z', False)
    int_327675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 25), 'int')
    int_327676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 28), 'int')
    # Processing the call keyword arguments (line 276)
    int_327677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 36), 'int')
    keyword_327678 = int_327677
    str_327679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 44), 'str', 'interp')
    keyword_327680 = str_327679
    # Getting the type of 'delta' (line 276)
    delta_327681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 60), 'delta', False)
    keyword_327682 = delta_327681
    kwargs_327683 = {'delta': keyword_327682, 'mode': keyword_327680, 'axis': keyword_327678}
    # Getting the type of 'savgol_filter' (line 276)
    savgol_filter_327673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 8), 'savgol_filter', False)
    # Calling savgol_filter(args, kwargs) (line 276)
    savgol_filter_call_result_327684 = invoke(stypy.reporting.localization.Localization(__file__, 276, 8), savgol_filter_327673, *[z_327674, int_327675, int_327676], **kwargs_327683)
    
    # Assigning a type to the variable 'y' (line 276)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 4), 'y', savgol_filter_call_result_327684)
    
    # Call to assert_allclose(...): (line 277)
    # Processing the call arguments (line 277)
    # Getting the type of 'y' (line 277)
    y_327686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 20), 'y', False)
    # Getting the type of 'z' (line 277)
    z_327687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 23), 'z', False)
    # Processing the call keyword arguments (line 277)
    float_327688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 31), 'float')
    keyword_327689 = float_327688
    kwargs_327690 = {'atol': keyword_327689}
    # Getting the type of 'assert_allclose' (line 277)
    assert_allclose_327685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 277)
    assert_allclose_call_result_327691 = invoke(stypy.reporting.localization.Localization(__file__, 277, 4), assert_allclose_327685, *[y_327686, z_327687], **kwargs_327690)
    
    
    # Assigning a Call to a Name (line 279):
    
    # Call to savgol_filter(...): (line 279)
    # Processing the call arguments (line 279)
    # Getting the type of 'z' (line 279)
    z_327693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 23), 'z', False)
    int_327694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 26), 'int')
    int_327695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 29), 'int')
    # Processing the call keyword arguments (line 279)
    int_327696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 37), 'int')
    keyword_327697 = int_327696
    str_327698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 45), 'str', 'interp')
    keyword_327699 = str_327698
    int_327700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 61), 'int')
    keyword_327701 = int_327700
    # Getting the type of 'delta' (line 279)
    delta_327702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 70), 'delta', False)
    keyword_327703 = delta_327702
    kwargs_327704 = {'deriv': keyword_327701, 'delta': keyword_327703, 'mode': keyword_327699, 'axis': keyword_327697}
    # Getting the type of 'savgol_filter' (line 279)
    savgol_filter_327692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 9), 'savgol_filter', False)
    # Calling savgol_filter(args, kwargs) (line 279)
    savgol_filter_call_result_327705 = invoke(stypy.reporting.localization.Localization(__file__, 279, 9), savgol_filter_327692, *[z_327693, int_327694, int_327695], **kwargs_327704)
    
    # Assigning a type to the variable 'dy' (line 279)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 4), 'dy', savgol_filter_call_result_327705)
    
    # Call to assert_allclose(...): (line 280)
    # Processing the call arguments (line 280)
    # Getting the type of 'dy' (line 280)
    dy_327707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 20), 'dy', False)
    # Getting the type of 'dz' (line 280)
    dz_327708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 24), 'dz', False)
    # Processing the call keyword arguments (line 280)
    float_327709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 33), 'float')
    keyword_327710 = float_327709
    kwargs_327711 = {'atol': keyword_327710}
    # Getting the type of 'assert_allclose' (line 280)
    assert_allclose_327706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 280)
    assert_allclose_call_result_327712 = invoke(stypy.reporting.localization.Localization(__file__, 280, 4), assert_allclose_327706, *[dy_327707, dz_327708], **kwargs_327711)
    
    
    # Assigning a Call to a Name (line 283):
    
    # Call to copy(...): (line 283)
    # Processing the call keyword arguments (line 283)
    kwargs_327720 = {}
    
    # Call to swapaxes(...): (line 283)
    # Processing the call arguments (line 283)
    int_327715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 19), 'int')
    int_327716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 22), 'int')
    # Processing the call keyword arguments (line 283)
    kwargs_327717 = {}
    # Getting the type of 'z' (line 283)
    z_327713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 8), 'z', False)
    # Obtaining the member 'swapaxes' of a type (line 283)
    swapaxes_327714 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 8), z_327713, 'swapaxes')
    # Calling swapaxes(args, kwargs) (line 283)
    swapaxes_call_result_327718 = invoke(stypy.reporting.localization.Localization(__file__, 283, 8), swapaxes_327714, *[int_327715, int_327716], **kwargs_327717)
    
    # Obtaining the member 'copy' of a type (line 283)
    copy_327719 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 8), swapaxes_call_result_327718, 'copy')
    # Calling copy(args, kwargs) (line 283)
    copy_call_result_327721 = invoke(stypy.reporting.localization.Localization(__file__, 283, 8), copy_327719, *[], **kwargs_327720)
    
    # Assigning a type to the variable 'z' (line 283)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 4), 'z', copy_call_result_327721)
    
    # Assigning a Call to a Name (line 284):
    
    # Call to copy(...): (line 284)
    # Processing the call keyword arguments (line 284)
    kwargs_327729 = {}
    
    # Call to swapaxes(...): (line 284)
    # Processing the call arguments (line 284)
    int_327724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 21), 'int')
    int_327725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 24), 'int')
    # Processing the call keyword arguments (line 284)
    kwargs_327726 = {}
    # Getting the type of 'dz' (line 284)
    dz_327722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 9), 'dz', False)
    # Obtaining the member 'swapaxes' of a type (line 284)
    swapaxes_327723 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 9), dz_327722, 'swapaxes')
    # Calling swapaxes(args, kwargs) (line 284)
    swapaxes_call_result_327727 = invoke(stypy.reporting.localization.Localization(__file__, 284, 9), swapaxes_327723, *[int_327724, int_327725], **kwargs_327726)
    
    # Obtaining the member 'copy' of a type (line 284)
    copy_327728 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 9), swapaxes_call_result_327727, 'copy')
    # Calling copy(args, kwargs) (line 284)
    copy_call_result_327730 = invoke(stypy.reporting.localization.Localization(__file__, 284, 9), copy_327728, *[], **kwargs_327729)
    
    # Assigning a type to the variable 'dz' (line 284)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 4), 'dz', copy_call_result_327730)
    
    # Assigning a Call to a Name (line 286):
    
    # Call to savgol_filter(...): (line 286)
    # Processing the call arguments (line 286)
    # Getting the type of 'z' (line 286)
    z_327732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 22), 'z', False)
    int_327733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 25), 'int')
    int_327734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 28), 'int')
    # Processing the call keyword arguments (line 286)
    int_327735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 36), 'int')
    keyword_327736 = int_327735
    str_327737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 44), 'str', 'interp')
    keyword_327738 = str_327737
    # Getting the type of 'delta' (line 286)
    delta_327739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 60), 'delta', False)
    keyword_327740 = delta_327739
    kwargs_327741 = {'delta': keyword_327740, 'mode': keyword_327738, 'axis': keyword_327736}
    # Getting the type of 'savgol_filter' (line 286)
    savgol_filter_327731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 8), 'savgol_filter', False)
    # Calling savgol_filter(args, kwargs) (line 286)
    savgol_filter_call_result_327742 = invoke(stypy.reporting.localization.Localization(__file__, 286, 8), savgol_filter_327731, *[z_327732, int_327733, int_327734], **kwargs_327741)
    
    # Assigning a type to the variable 'y' (line 286)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 4), 'y', savgol_filter_call_result_327742)
    
    # Call to assert_allclose(...): (line 287)
    # Processing the call arguments (line 287)
    # Getting the type of 'y' (line 287)
    y_327744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 20), 'y', False)
    # Getting the type of 'z' (line 287)
    z_327745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 23), 'z', False)
    # Processing the call keyword arguments (line 287)
    float_327746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 31), 'float')
    keyword_327747 = float_327746
    kwargs_327748 = {'atol': keyword_327747}
    # Getting the type of 'assert_allclose' (line 287)
    assert_allclose_327743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 287)
    assert_allclose_call_result_327749 = invoke(stypy.reporting.localization.Localization(__file__, 287, 4), assert_allclose_327743, *[y_327744, z_327745], **kwargs_327748)
    
    
    # Assigning a Call to a Name (line 289):
    
    # Call to savgol_filter(...): (line 289)
    # Processing the call arguments (line 289)
    # Getting the type of 'z' (line 289)
    z_327751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 23), 'z', False)
    int_327752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 26), 'int')
    int_327753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 29), 'int')
    # Processing the call keyword arguments (line 289)
    int_327754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 37), 'int')
    keyword_327755 = int_327754
    str_327756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 45), 'str', 'interp')
    keyword_327757 = str_327756
    int_327758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 61), 'int')
    keyword_327759 = int_327758
    # Getting the type of 'delta' (line 289)
    delta_327760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 70), 'delta', False)
    keyword_327761 = delta_327760
    kwargs_327762 = {'deriv': keyword_327759, 'delta': keyword_327761, 'mode': keyword_327757, 'axis': keyword_327755}
    # Getting the type of 'savgol_filter' (line 289)
    savgol_filter_327750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 9), 'savgol_filter', False)
    # Calling savgol_filter(args, kwargs) (line 289)
    savgol_filter_call_result_327763 = invoke(stypy.reporting.localization.Localization(__file__, 289, 9), savgol_filter_327750, *[z_327751, int_327752, int_327753], **kwargs_327762)
    
    # Assigning a type to the variable 'dy' (line 289)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 4), 'dy', savgol_filter_call_result_327763)
    
    # Call to assert_allclose(...): (line 290)
    # Processing the call arguments (line 290)
    # Getting the type of 'dy' (line 290)
    dy_327765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 20), 'dy', False)
    # Getting the type of 'dz' (line 290)
    dz_327766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 24), 'dz', False)
    # Processing the call keyword arguments (line 290)
    float_327767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 33), 'float')
    keyword_327768 = float_327767
    kwargs_327769 = {'atol': keyword_327768}
    # Getting the type of 'assert_allclose' (line 290)
    assert_allclose_327764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 290)
    assert_allclose_call_result_327770 = invoke(stypy.reporting.localization.Localization(__file__, 290, 4), assert_allclose_327764, *[dy_327765, dz_327766], **kwargs_327769)
    
    
    # ################# End of 'test_sg_filter_interp_edges_3d(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_sg_filter_interp_edges_3d' in the type store
    # Getting the type of 'stypy_return_type' (line 251)
    stypy_return_type_327771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_327771)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_sg_filter_interp_edges_3d'
    return stypy_return_type_327771

# Assigning a type to the variable 'test_sg_filter_interp_edges_3d' (line 251)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 0), 'test_sg_filter_interp_edges_3d', test_sg_filter_interp_edges_3d)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
