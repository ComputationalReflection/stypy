
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: ''' Some tests for filters '''
2: from __future__ import division, print_function, absolute_import
3: 
4: import sys
5: import numpy as np
6: 
7: from numpy.testing import (assert_equal, assert_allclose,
8:                            assert_array_equal, assert_almost_equal)
9: from pytest import raises as assert_raises
10: 
11: import scipy.ndimage as sndi
12: from scipy.ndimage.filters import _gaussian_kernel1d
13: 
14: 
15: def test_ticket_701():
16:     # Test generic filter sizes
17:     arr = np.arange(4).reshape((2,2))
18:     func = lambda x: np.min(x)
19:     res = sndi.generic_filter(arr, func, size=(1,1))
20:     # The following raises an error unless ticket 701 is fixed
21:     res2 = sndi.generic_filter(arr, func, size=1)
22:     assert_equal(res, res2)
23: 
24: 
25: def test_gh_5430():
26:     # At least one of these raises an error unless gh-5430 is
27:     # fixed. In py2k an int is implemented using a C long, so
28:     # which one fails depends on your system. In py3k there is only
29:     # one arbitrary precision integer type, so both should fail.
30:     sigma = np.int32(1)
31:     out = sndi._ni_support._normalize_sequence(sigma, 1)
32:     assert_equal(out, [sigma])
33:     sigma = np.int64(1)
34:     out = sndi._ni_support._normalize_sequence(sigma, 1)
35:     assert_equal(out, [sigma])
36:     # This worked before; make sure it still works
37:     sigma = 1
38:     out = sndi._ni_support._normalize_sequence(sigma, 1)
39:     assert_equal(out, [sigma])
40:     # This worked before; make sure it still works
41:     sigma = [1, 1]
42:     out = sndi._ni_support._normalize_sequence(sigma, 2)
43:     assert_equal(out, sigma)
44:     # Also include the OPs original example to make sure we fixed the issue
45:     x = np.random.normal(size=(256, 256))
46:     perlin = np.zeros_like(x)
47:     for i in 2**np.arange(6):
48:         perlin += sndi.filters.gaussian_filter(x, i, mode="wrap") * i**2
49:     # This also fixes gh-4106, show that the OPs example now runs.
50:     x = np.int64(21)
51:     sndi._ni_support._normalize_sequence(x, 0)
52: 
53: 
54: def test_gaussian_kernel1d():
55:     radius = 10
56:     sigma = 2
57:     sigma2 = sigma * sigma
58:     x = np.arange(-radius, radius + 1, dtype=np.double)
59:     phi_x = np.exp(-0.5 * x * x / sigma2)
60:     phi_x /= phi_x.sum()
61:     assert_allclose(phi_x, _gaussian_kernel1d(sigma, 0, radius))
62:     assert_allclose(-phi_x * x / sigma2, _gaussian_kernel1d(sigma, 1, radius))
63:     assert_allclose(phi_x * (x * x / sigma2 - 1) / sigma2,
64:                     _gaussian_kernel1d(sigma, 2, radius))
65:     assert_allclose(phi_x * (3 - x * x / sigma2) * x / (sigma2 * sigma2),
66:                     _gaussian_kernel1d(sigma, 3, radius))
67: 
68: 
69: def test_orders_gauss():
70:     # Check order inputs to Gaussians
71:     arr = np.zeros((1,))
72:     assert_equal(0, sndi.gaussian_filter(arr, 1, order=0))
73:     assert_equal(0, sndi.gaussian_filter(arr, 1, order=3))
74:     assert_raises(ValueError, sndi.gaussian_filter, arr, 1, -1)
75:     assert_equal(0, sndi.gaussian_filter1d(arr, 1, axis=-1, order=0))
76:     assert_equal(0, sndi.gaussian_filter1d(arr, 1, axis=-1, order=3))
77:     assert_raises(ValueError, sndi.gaussian_filter1d, arr, 1, -1, -1)
78: 
79: 
80: def test_valid_origins():
81:     '''Regression test for #1311.'''
82:     func = lambda x: np.mean(x)
83:     data = np.array([1,2,3,4,5], dtype=np.float64)
84:     assert_raises(ValueError, sndi.generic_filter, data, func, size=3,
85:                   origin=2)
86:     func2 = lambda x, y: np.mean(x + y)
87:     assert_raises(ValueError, sndi.generic_filter1d, data, func,
88:                   filter_size=3, origin=2)
89:     assert_raises(ValueError, sndi.percentile_filter, data, 0.2, size=3,
90:                   origin=2)
91: 
92:     for filter in [sndi.uniform_filter, sndi.minimum_filter,
93:                    sndi.maximum_filter, sndi.maximum_filter1d,
94:                    sndi.median_filter, sndi.minimum_filter1d]:
95:         # This should work, since for size == 3, the valid range for origin is
96:         # -1 to 1.
97:         list(filter(data, 3, origin=-1))
98:         list(filter(data, 3, origin=1))
99:         # Just check this raises an error instead of silently accepting or
100:         # segfaulting.
101:         assert_raises(ValueError, filter, data, 3, origin=2)
102: 
103: 
104: def test_multiple_modes():
105:     # Test that the filters with multiple mode cababilities for different
106:     # dimensions give the same result as applying a single mode.
107:     arr = np.array([[1., 0., 0.],
108:                     [1., 1., 0.],
109:                     [0., 0., 0.]])
110: 
111:     mode1 = 'reflect'
112:     mode2 = ['reflect', 'reflect']
113: 
114:     assert_equal(sndi.gaussian_filter(arr, 1, mode=mode1),
115:                  sndi.gaussian_filter(arr, 1, mode=mode2))
116:     assert_equal(sndi.prewitt(arr, mode=mode1),
117:                  sndi.prewitt(arr, mode=mode2))
118:     assert_equal(sndi.sobel(arr, mode=mode1),
119:                  sndi.sobel(arr, mode=mode2))
120:     assert_equal(sndi.laplace(arr, mode=mode1),
121:                  sndi.laplace(arr, mode=mode2))
122:     assert_equal(sndi.gaussian_laplace(arr, 1, mode=mode1),
123:                  sndi.gaussian_laplace(arr, 1, mode=mode2))
124:     assert_equal(sndi.maximum_filter(arr, size=5, mode=mode1),
125:                  sndi.maximum_filter(arr, size=5, mode=mode2))
126:     assert_equal(sndi.minimum_filter(arr, size=5, mode=mode1),
127:                  sndi.minimum_filter(arr, size=5, mode=mode2))
128:     assert_equal(sndi.gaussian_gradient_magnitude(arr, 1, mode=mode1),
129:                  sndi.gaussian_gradient_magnitude(arr, 1, mode=mode2))
130:     assert_equal(sndi.uniform_filter(arr, 5, mode=mode1),
131:                  sndi.uniform_filter(arr, 5, mode=mode2))
132: 
133: 
134: def test_multiple_modes_sequentially():
135:     # Test that the filters with multiple mode cababilities for different
136:     # dimensions give the same result as applying the filters with
137:     # different modes sequentially
138:     arr = np.array([[1., 0., 0.],
139:                     [1., 1., 0.],
140:                     [0., 0., 0.]])
141: 
142:     modes = ['reflect', 'wrap']
143: 
144:     expected = sndi.gaussian_filter1d(arr, 1, axis=0, mode=modes[0])
145:     expected = sndi.gaussian_filter1d(expected, 1, axis=1, mode=modes[1])
146:     assert_equal(expected,
147:                  sndi.gaussian_filter(arr, 1, mode=modes))
148: 
149:     expected = sndi.uniform_filter1d(arr, 5, axis=0, mode=modes[0])
150:     expected = sndi.uniform_filter1d(expected, 5, axis=1, mode=modes[1])
151:     assert_equal(expected,
152:                  sndi.uniform_filter(arr, 5, mode=modes))
153: 
154:     expected = sndi.maximum_filter1d(arr, size=5, axis=0, mode=modes[0])
155:     expected = sndi.maximum_filter1d(expected, size=5, axis=1, mode=modes[1])
156:     assert_equal(expected,
157:                  sndi.maximum_filter(arr, size=5, mode=modes))
158: 
159:     expected = sndi.minimum_filter1d(arr, size=5, axis=0, mode=modes[0])
160:     expected = sndi.minimum_filter1d(expected, size=5, axis=1, mode=modes[1])
161:     assert_equal(expected,
162:                  sndi.minimum_filter(arr, size=5, mode=modes))
163: 
164: 
165: def test_multiple_modes_prewitt():
166:     # Test prewitt filter for multiple extrapolation modes
167:     arr = np.array([[1., 0., 0.],
168:                     [1., 1., 0.],
169:                     [0., 0., 0.]])
170: 
171:     expected = np.array([[1., -3., 2.],
172:                          [1., -2., 1.],
173:                          [1., -1., 0.]])
174: 
175:     modes = ['reflect', 'wrap']
176: 
177:     assert_equal(expected,
178:                  sndi.prewitt(arr, mode=modes))
179: 
180: 
181: def test_multiple_modes_sobel():
182:     # Test sobel filter for multiple extrapolation modes
183:     arr = np.array([[1., 0., 0.],
184:                     [1., 1., 0.],
185:                     [0., 0., 0.]])
186: 
187:     expected = np.array([[1., -4., 3.],
188:                          [2., -3., 1.],
189:                          [1., -1., 0.]])
190: 
191:     modes = ['reflect', 'wrap']
192: 
193:     assert_equal(expected,
194:                  sndi.sobel(arr, mode=modes))
195: 
196: 
197: def test_multiple_modes_laplace():
198:     # Test laplace filter for multiple extrapolation modes
199:     arr = np.array([[1., 0., 0.],
200:                     [1., 1., 0.],
201:                     [0., 0., 0.]])
202: 
203:     expected = np.array([[-2., 2., 1.],
204:                          [-2., -3., 2.],
205:                          [1., 1., 0.]])
206: 
207:     modes = ['reflect', 'wrap']
208: 
209:     assert_equal(expected,
210:                  sndi.laplace(arr, mode=modes))
211: 
212: 
213: def test_multiple_modes_gaussian_laplace():
214:     # Test gaussian_laplace filter for multiple extrapolation modes
215:     arr = np.array([[1., 0., 0.],
216:                     [1., 1., 0.],
217:                     [0., 0., 0.]])
218: 
219:     expected = np.array([[-0.28438687, 0.01559809, 0.19773499],
220:                          [-0.36630503, -0.20069774, 0.07483620],
221:                          [0.15849176, 0.18495566, 0.21934094]])
222: 
223:     modes = ['reflect', 'wrap']
224: 
225:     assert_almost_equal(expected,
226:                         sndi.gaussian_laplace(arr, 1, mode=modes))
227: 
228: 
229: def test_multiple_modes_gaussian_gradient_magnitude():
230:     # Test gaussian_gradient_magnitude filter for multiple
231:     # extrapolation modes
232:     arr = np.array([[1., 0., 0.],
233:                     [1., 1., 0.],
234:                     [0., 0., 0.]])
235: 
236:     expected = np.array([[0.04928965, 0.09745625, 0.06405368],
237:                          [0.23056905, 0.14025305, 0.04550846],
238:                          [0.19894369, 0.14950060, 0.06796850]])
239: 
240:     modes = ['reflect', 'wrap']
241: 
242:     calculated = sndi.gaussian_gradient_magnitude(arr, 1, mode=modes)
243: 
244:     assert_almost_equal(expected, calculated)
245: 
246: 
247: def test_multiple_modes_uniform():
248:     # Test uniform filter for multiple extrapolation modes
249:     arr = np.array([[1., 0., 0.],
250:                     [1., 1., 0.],
251:                     [0., 0., 0.]])
252: 
253:     expected = np.array([[0.32, 0.40, 0.48],
254:                          [0.20, 0.28, 0.32],
255:                          [0.28, 0.32, 0.40]])
256: 
257:     modes = ['reflect', 'wrap']
258: 
259:     assert_almost_equal(expected,
260:                         sndi.uniform_filter(arr, 5, mode=modes))
261: 
262: 
263: def test_gaussian_truncate():
264:     # Test that Gaussian filters can be truncated at different widths.
265:     # These tests only check that the result has the expected number
266:     # of nonzero elements.
267:     arr = np.zeros((100, 100), float)
268:     arr[50, 50] = 1
269:     num_nonzeros_2 = (sndi.gaussian_filter(arr, 5, truncate=2) > 0).sum()
270:     assert_equal(num_nonzeros_2, 21**2)
271:     num_nonzeros_5 = (sndi.gaussian_filter(arr, 5, truncate=5) > 0).sum()
272:     assert_equal(num_nonzeros_5, 51**2)
273: 
274:     # Test truncate when sigma is a sequence.
275:     f = sndi.gaussian_filter(arr, [0.5, 2.5], truncate=3.5)
276:     fpos = f > 0
277:     n0 = fpos.any(axis=0).sum()
278:     # n0 should be 2*int(2.5*3.5 + 0.5) + 1
279:     assert_equal(n0, 19)
280:     n1 = fpos.any(axis=1).sum()
281:     # n1 should be 2*int(0.5*3.5 + 0.5) + 1
282:     assert_equal(n1, 5)
283: 
284:     # Test gaussian_filter1d.
285:     x = np.zeros(51)
286:     x[25] = 1
287:     f = sndi.gaussian_filter1d(x, sigma=2, truncate=3.5)
288:     n = (f > 0).sum()
289:     assert_equal(n, 15)
290: 
291:     # Test gaussian_laplace
292:     y = sndi.gaussian_laplace(x, sigma=2, truncate=3.5)
293:     nonzero_indices = np.where(y != 0)[0]
294:     n = nonzero_indices.ptp() + 1
295:     assert_equal(n, 15)
296: 
297:     # Test gaussian_gradient_magnitude
298:     y = sndi.gaussian_gradient_magnitude(x, sigma=2, truncate=3.5)
299:     nonzero_indices = np.where(y != 0)[0]
300:     n = nonzero_indices.ptp() + 1
301:     assert_equal(n, 15)
302: 
303: 
304: class TestThreading(object):
305:     def check_func_thread(self, n, fun, args, out):
306:         from threading import Thread
307:         thrds = [Thread(target=fun, args=args, kwargs={'output': out[x]}) for x in range(n)]
308:         [t.start() for t in thrds]
309:         [t.join() for t in thrds]
310: 
311:     def check_func_serial(self, n, fun, args, out):
312:         for i in range(n):
313:             fun(*args, output=out[i])
314: 
315:     def test_correlate1d(self):
316:         d = np.random.randn(5000)
317:         os = np.empty((4, d.size))
318:         ot = np.empty_like(os)
319:         self.check_func_serial(4, sndi.correlate1d, (d, np.arange(5)), os)
320:         self.check_func_thread(4, sndi.correlate1d, (d, np.arange(5)), ot)
321:         assert_array_equal(os, ot)
322: 
323:     def test_correlate(self):
324:         d = np.random.randn(500, 500)
325:         k = np.random.randn(10, 10)
326:         os = np.empty([4] + list(d.shape))
327:         ot = np.empty_like(os)
328:         self.check_func_serial(4, sndi.correlate, (d, k), os)
329:         self.check_func_thread(4, sndi.correlate, (d, k), ot)
330:         assert_array_equal(os, ot)
331: 
332:     def test_median_filter(self):
333:         d = np.random.randn(500, 500)
334:         os = np.empty([4] + list(d.shape))
335:         ot = np.empty_like(os)
336:         self.check_func_serial(4, sndi.median_filter, (d, 3), os)
337:         self.check_func_thread(4, sndi.median_filter, (d, 3), ot)
338:         assert_array_equal(os, ot)
339: 
340:     def test_uniform_filter1d(self):
341:         d = np.random.randn(5000)
342:         os = np.empty((4, d.size))
343:         ot = np.empty_like(os)
344:         self.check_func_serial(4, sndi.uniform_filter1d, (d, 5), os)
345:         self.check_func_thread(4, sndi.uniform_filter1d, (d, 5), ot)
346:         assert_array_equal(os, ot)
347: 
348:     def test_minmax_filter(self):
349:         d = np.random.randn(500, 500)
350:         os = np.empty([4] + list(d.shape))
351:         ot = np.empty_like(os)
352:         self.check_func_serial(4, sndi.maximum_filter, (d, 3), os)
353:         self.check_func_thread(4, sndi.maximum_filter, (d, 3), ot)
354:         assert_array_equal(os, ot)
355:         self.check_func_serial(4, sndi.minimum_filter, (d, 3), os)
356:         self.check_func_thread(4, sndi.minimum_filter, (d, 3), ot)
357:         assert_array_equal(os, ot)
358: 
359: 
360: def test_minmaximum_filter1d():
361:     # Regression gh-3898
362:     in_ = np.arange(10)
363:     out = sndi.minimum_filter1d(in_, 1)
364:     assert_equal(in_, out)
365:     out = sndi.maximum_filter1d(in_, 1)
366:     assert_equal(in_, out)
367:     # Test reflect
368:     out = sndi.minimum_filter1d(in_, 5, mode='reflect')
369:     assert_equal([0, 0, 0, 1, 2, 3, 4, 5, 6, 7], out)
370:     out = sndi.maximum_filter1d(in_, 5, mode='reflect')
371:     assert_equal([2, 3, 4, 5, 6, 7, 8, 9, 9, 9], out)
372:     #Test constant
373:     out = sndi.minimum_filter1d(in_, 5, mode='constant', cval=-1)
374:     assert_equal([-1, -1, 0, 1, 2, 3, 4, 5, -1, -1], out)
375:     out = sndi.maximum_filter1d(in_, 5, mode='constant', cval=10)
376:     assert_equal([10, 10, 4, 5, 6, 7, 8, 9, 10, 10], out)
377:     # Test nearest
378:     out = sndi.minimum_filter1d(in_, 5, mode='nearest')
379:     assert_equal([0, 0, 0, 1, 2, 3, 4, 5, 6, 7], out)
380:     out = sndi.maximum_filter1d(in_, 5, mode='nearest')
381:     assert_equal([2, 3, 4, 5, 6, 7, 8, 9, 9, 9], out)
382:     # Test wrap
383:     out = sndi.minimum_filter1d(in_, 5, mode='wrap')
384:     assert_equal([0, 0, 0, 1, 2, 3, 4, 5, 0, 0], out)
385:     out = sndi.maximum_filter1d(in_, 5, mode='wrap')
386:     assert_equal([9, 9, 4, 5, 6, 7, 8, 9, 9, 9], out)
387: 
388: 
389: def test_uniform_filter1d_roundoff_errors():
390:     # gh-6930
391:     in_ = np.repeat([0, 1, 0], [9, 9, 9])
392:     for filter_size in range(3, 10):
393:         out = sndi.uniform_filter1d(in_, filter_size)
394:         assert_equal(out.sum(), 10 - filter_size)
395: 
396: 
397: def test_footprint_all_zeros():
398:     # regression test for gh-6876: footprint of all zeros segfaults
399:     arr = np.random.randint(0, 100, (100, 100))
400:     kernel = np.zeros((3, 3), bool)
401:     with assert_raises(ValueError):
402:         sndi.maximum_filter(arr, footprint=kernel)
403: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_127529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 0), 'str', ' Some tests for filters ')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import sys' statement (line 4)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import numpy' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/ndimage/tests/')
import_127530 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy')

if (type(import_127530) is not StypyTypeError):

    if (import_127530 != 'pyd_module'):
        __import__(import_127530)
        sys_modules_127531 = sys.modules[import_127530]
        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'np', sys_modules_127531.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy', import_127530)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/ndimage/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from numpy.testing import assert_equal, assert_allclose, assert_array_equal, assert_almost_equal' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/ndimage/tests/')
import_127532 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.testing')

if (type(import_127532) is not StypyTypeError):

    if (import_127532 != 'pyd_module'):
        __import__(import_127532)
        sys_modules_127533 = sys.modules[import_127532]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.testing', sys_modules_127533.module_type_store, module_type_store, ['assert_equal', 'assert_allclose', 'assert_array_equal', 'assert_almost_equal'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_127533, sys_modules_127533.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_equal, assert_allclose, assert_array_equal, assert_almost_equal

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.testing', None, module_type_store, ['assert_equal', 'assert_allclose', 'assert_array_equal', 'assert_almost_equal'], [assert_equal, assert_allclose, assert_array_equal, assert_almost_equal])

else:
    # Assigning a type to the variable 'numpy.testing' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.testing', import_127532)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/ndimage/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from pytest import assert_raises' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/ndimage/tests/')
import_127534 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'pytest')

if (type(import_127534) is not StypyTypeError):

    if (import_127534 != 'pyd_module'):
        __import__(import_127534)
        sys_modules_127535 = sys.modules[import_127534]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'pytest', sys_modules_127535.module_type_store, module_type_store, ['raises'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_127535, sys_modules_127535.module_type_store, module_type_store)
    else:
        from pytest import raises as assert_raises

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'pytest', None, module_type_store, ['raises'], [assert_raises])

else:
    # Assigning a type to the variable 'pytest' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'pytest', import_127534)

# Adding an alias
module_type_store.add_alias('assert_raises', 'raises')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/ndimage/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'import scipy.ndimage' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/ndimage/tests/')
import_127536 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.ndimage')

if (type(import_127536) is not StypyTypeError):

    if (import_127536 != 'pyd_module'):
        __import__(import_127536)
        sys_modules_127537 = sys.modules[import_127536]
        import_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'sndi', sys_modules_127537.module_type_store, module_type_store)
    else:
        import scipy.ndimage as sndi

        import_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'sndi', scipy.ndimage, module_type_store)

else:
    # Assigning a type to the variable 'scipy.ndimage' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.ndimage', import_127536)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/ndimage/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from scipy.ndimage.filters import _gaussian_kernel1d' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/ndimage/tests/')
import_127538 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.ndimage.filters')

if (type(import_127538) is not StypyTypeError):

    if (import_127538 != 'pyd_module'):
        __import__(import_127538)
        sys_modules_127539 = sys.modules[import_127538]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.ndimage.filters', sys_modules_127539.module_type_store, module_type_store, ['_gaussian_kernel1d'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_127539, sys_modules_127539.module_type_store, module_type_store)
    else:
        from scipy.ndimage.filters import _gaussian_kernel1d

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.ndimage.filters', None, module_type_store, ['_gaussian_kernel1d'], [_gaussian_kernel1d])

else:
    # Assigning a type to the variable 'scipy.ndimage.filters' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.ndimage.filters', import_127538)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/ndimage/tests/')


@norecursion
def test_ticket_701(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_ticket_701'
    module_type_store = module_type_store.open_function_context('test_ticket_701', 15, 0, False)
    
    # Passed parameters checking function
    test_ticket_701.stypy_localization = localization
    test_ticket_701.stypy_type_of_self = None
    test_ticket_701.stypy_type_store = module_type_store
    test_ticket_701.stypy_function_name = 'test_ticket_701'
    test_ticket_701.stypy_param_names_list = []
    test_ticket_701.stypy_varargs_param_name = None
    test_ticket_701.stypy_kwargs_param_name = None
    test_ticket_701.stypy_call_defaults = defaults
    test_ticket_701.stypy_call_varargs = varargs
    test_ticket_701.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_ticket_701', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_ticket_701', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_ticket_701(...)' code ##################

    
    # Assigning a Call to a Name (line 17):
    
    # Call to reshape(...): (line 17)
    # Processing the call arguments (line 17)
    
    # Obtaining an instance of the builtin type 'tuple' (line 17)
    tuple_127546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 32), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 17)
    # Adding element type (line 17)
    int_127547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 32), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 32), tuple_127546, int_127547)
    # Adding element type (line 17)
    int_127548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 34), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 32), tuple_127546, int_127548)
    
    # Processing the call keyword arguments (line 17)
    kwargs_127549 = {}
    
    # Call to arange(...): (line 17)
    # Processing the call arguments (line 17)
    int_127542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 20), 'int')
    # Processing the call keyword arguments (line 17)
    kwargs_127543 = {}
    # Getting the type of 'np' (line 17)
    np_127540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 10), 'np', False)
    # Obtaining the member 'arange' of a type (line 17)
    arange_127541 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 10), np_127540, 'arange')
    # Calling arange(args, kwargs) (line 17)
    arange_call_result_127544 = invoke(stypy.reporting.localization.Localization(__file__, 17, 10), arange_127541, *[int_127542], **kwargs_127543)
    
    # Obtaining the member 'reshape' of a type (line 17)
    reshape_127545 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 10), arange_call_result_127544, 'reshape')
    # Calling reshape(args, kwargs) (line 17)
    reshape_call_result_127550 = invoke(stypy.reporting.localization.Localization(__file__, 17, 10), reshape_127545, *[tuple_127546], **kwargs_127549)
    
    # Assigning a type to the variable 'arr' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'arr', reshape_call_result_127550)
    
    # Assigning a Lambda to a Name (line 18):

    @norecursion
    def _stypy_temp_lambda_45(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_45'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_45', 18, 11, True)
        # Passed parameters checking function
        _stypy_temp_lambda_45.stypy_localization = localization
        _stypy_temp_lambda_45.stypy_type_of_self = None
        _stypy_temp_lambda_45.stypy_type_store = module_type_store
        _stypy_temp_lambda_45.stypy_function_name = '_stypy_temp_lambda_45'
        _stypy_temp_lambda_45.stypy_param_names_list = ['x']
        _stypy_temp_lambda_45.stypy_varargs_param_name = None
        _stypy_temp_lambda_45.stypy_kwargs_param_name = None
        _stypy_temp_lambda_45.stypy_call_defaults = defaults
        _stypy_temp_lambda_45.stypy_call_varargs = varargs
        _stypy_temp_lambda_45.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_45', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Stacktrace push for error reporting
        localization.set_stack_trace('_stypy_temp_lambda_45', ['x'], arguments)
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of the lambda function code ##################

        
        # Call to min(...): (line 18)
        # Processing the call arguments (line 18)
        # Getting the type of 'x' (line 18)
        x_127553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 28), 'x', False)
        # Processing the call keyword arguments (line 18)
        kwargs_127554 = {}
        # Getting the type of 'np' (line 18)
        np_127551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 21), 'np', False)
        # Obtaining the member 'min' of a type (line 18)
        min_127552 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 21), np_127551, 'min')
        # Calling min(args, kwargs) (line 18)
        min_call_result_127555 = invoke(stypy.reporting.localization.Localization(__file__, 18, 21), min_127552, *[x_127553], **kwargs_127554)
        
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 11), 'stypy_return_type', min_call_result_127555)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_45' in the type store
        # Getting the type of 'stypy_return_type' (line 18)
        stypy_return_type_127556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 11), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_127556)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_45'
        return stypy_return_type_127556

    # Assigning a type to the variable '_stypy_temp_lambda_45' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 11), '_stypy_temp_lambda_45', _stypy_temp_lambda_45)
    # Getting the type of '_stypy_temp_lambda_45' (line 18)
    _stypy_temp_lambda_45_127557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 11), '_stypy_temp_lambda_45')
    # Assigning a type to the variable 'func' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'func', _stypy_temp_lambda_45_127557)
    
    # Assigning a Call to a Name (line 19):
    
    # Call to generic_filter(...): (line 19)
    # Processing the call arguments (line 19)
    # Getting the type of 'arr' (line 19)
    arr_127560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 30), 'arr', False)
    # Getting the type of 'func' (line 19)
    func_127561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 35), 'func', False)
    # Processing the call keyword arguments (line 19)
    
    # Obtaining an instance of the builtin type 'tuple' (line 19)
    tuple_127562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 47), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 19)
    # Adding element type (line 19)
    int_127563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 47), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 47), tuple_127562, int_127563)
    # Adding element type (line 19)
    int_127564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 49), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 47), tuple_127562, int_127564)
    
    keyword_127565 = tuple_127562
    kwargs_127566 = {'size': keyword_127565}
    # Getting the type of 'sndi' (line 19)
    sndi_127558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 10), 'sndi', False)
    # Obtaining the member 'generic_filter' of a type (line 19)
    generic_filter_127559 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 10), sndi_127558, 'generic_filter')
    # Calling generic_filter(args, kwargs) (line 19)
    generic_filter_call_result_127567 = invoke(stypy.reporting.localization.Localization(__file__, 19, 10), generic_filter_127559, *[arr_127560, func_127561], **kwargs_127566)
    
    # Assigning a type to the variable 'res' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'res', generic_filter_call_result_127567)
    
    # Assigning a Call to a Name (line 21):
    
    # Call to generic_filter(...): (line 21)
    # Processing the call arguments (line 21)
    # Getting the type of 'arr' (line 21)
    arr_127570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 31), 'arr', False)
    # Getting the type of 'func' (line 21)
    func_127571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 36), 'func', False)
    # Processing the call keyword arguments (line 21)
    int_127572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 47), 'int')
    keyword_127573 = int_127572
    kwargs_127574 = {'size': keyword_127573}
    # Getting the type of 'sndi' (line 21)
    sndi_127568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 11), 'sndi', False)
    # Obtaining the member 'generic_filter' of a type (line 21)
    generic_filter_127569 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 11), sndi_127568, 'generic_filter')
    # Calling generic_filter(args, kwargs) (line 21)
    generic_filter_call_result_127575 = invoke(stypy.reporting.localization.Localization(__file__, 21, 11), generic_filter_127569, *[arr_127570, func_127571], **kwargs_127574)
    
    # Assigning a type to the variable 'res2' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'res2', generic_filter_call_result_127575)
    
    # Call to assert_equal(...): (line 22)
    # Processing the call arguments (line 22)
    # Getting the type of 'res' (line 22)
    res_127577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 17), 'res', False)
    # Getting the type of 'res2' (line 22)
    res2_127578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 22), 'res2', False)
    # Processing the call keyword arguments (line 22)
    kwargs_127579 = {}
    # Getting the type of 'assert_equal' (line 22)
    assert_equal_127576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 22)
    assert_equal_call_result_127580 = invoke(stypy.reporting.localization.Localization(__file__, 22, 4), assert_equal_127576, *[res_127577, res2_127578], **kwargs_127579)
    
    
    # ################# End of 'test_ticket_701(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_ticket_701' in the type store
    # Getting the type of 'stypy_return_type' (line 15)
    stypy_return_type_127581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_127581)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_ticket_701'
    return stypy_return_type_127581

# Assigning a type to the variable 'test_ticket_701' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'test_ticket_701', test_ticket_701)

@norecursion
def test_gh_5430(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_gh_5430'
    module_type_store = module_type_store.open_function_context('test_gh_5430', 25, 0, False)
    
    # Passed parameters checking function
    test_gh_5430.stypy_localization = localization
    test_gh_5430.stypy_type_of_self = None
    test_gh_5430.stypy_type_store = module_type_store
    test_gh_5430.stypy_function_name = 'test_gh_5430'
    test_gh_5430.stypy_param_names_list = []
    test_gh_5430.stypy_varargs_param_name = None
    test_gh_5430.stypy_kwargs_param_name = None
    test_gh_5430.stypy_call_defaults = defaults
    test_gh_5430.stypy_call_varargs = varargs
    test_gh_5430.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_gh_5430', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_gh_5430', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_gh_5430(...)' code ##################

    
    # Assigning a Call to a Name (line 30):
    
    # Call to int32(...): (line 30)
    # Processing the call arguments (line 30)
    int_127584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 21), 'int')
    # Processing the call keyword arguments (line 30)
    kwargs_127585 = {}
    # Getting the type of 'np' (line 30)
    np_127582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 12), 'np', False)
    # Obtaining the member 'int32' of a type (line 30)
    int32_127583 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 12), np_127582, 'int32')
    # Calling int32(args, kwargs) (line 30)
    int32_call_result_127586 = invoke(stypy.reporting.localization.Localization(__file__, 30, 12), int32_127583, *[int_127584], **kwargs_127585)
    
    # Assigning a type to the variable 'sigma' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'sigma', int32_call_result_127586)
    
    # Assigning a Call to a Name (line 31):
    
    # Call to _normalize_sequence(...): (line 31)
    # Processing the call arguments (line 31)
    # Getting the type of 'sigma' (line 31)
    sigma_127590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 47), 'sigma', False)
    int_127591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 54), 'int')
    # Processing the call keyword arguments (line 31)
    kwargs_127592 = {}
    # Getting the type of 'sndi' (line 31)
    sndi_127587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 10), 'sndi', False)
    # Obtaining the member '_ni_support' of a type (line 31)
    _ni_support_127588 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 10), sndi_127587, '_ni_support')
    # Obtaining the member '_normalize_sequence' of a type (line 31)
    _normalize_sequence_127589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 10), _ni_support_127588, '_normalize_sequence')
    # Calling _normalize_sequence(args, kwargs) (line 31)
    _normalize_sequence_call_result_127593 = invoke(stypy.reporting.localization.Localization(__file__, 31, 10), _normalize_sequence_127589, *[sigma_127590, int_127591], **kwargs_127592)
    
    # Assigning a type to the variable 'out' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'out', _normalize_sequence_call_result_127593)
    
    # Call to assert_equal(...): (line 32)
    # Processing the call arguments (line 32)
    # Getting the type of 'out' (line 32)
    out_127595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 17), 'out', False)
    
    # Obtaining an instance of the builtin type 'list' (line 32)
    list_127596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 32)
    # Adding element type (line 32)
    # Getting the type of 'sigma' (line 32)
    sigma_127597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 23), 'sigma', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 22), list_127596, sigma_127597)
    
    # Processing the call keyword arguments (line 32)
    kwargs_127598 = {}
    # Getting the type of 'assert_equal' (line 32)
    assert_equal_127594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 32)
    assert_equal_call_result_127599 = invoke(stypy.reporting.localization.Localization(__file__, 32, 4), assert_equal_127594, *[out_127595, list_127596], **kwargs_127598)
    
    
    # Assigning a Call to a Name (line 33):
    
    # Call to int64(...): (line 33)
    # Processing the call arguments (line 33)
    int_127602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 21), 'int')
    # Processing the call keyword arguments (line 33)
    kwargs_127603 = {}
    # Getting the type of 'np' (line 33)
    np_127600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 12), 'np', False)
    # Obtaining the member 'int64' of a type (line 33)
    int64_127601 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 12), np_127600, 'int64')
    # Calling int64(args, kwargs) (line 33)
    int64_call_result_127604 = invoke(stypy.reporting.localization.Localization(__file__, 33, 12), int64_127601, *[int_127602], **kwargs_127603)
    
    # Assigning a type to the variable 'sigma' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'sigma', int64_call_result_127604)
    
    # Assigning a Call to a Name (line 34):
    
    # Call to _normalize_sequence(...): (line 34)
    # Processing the call arguments (line 34)
    # Getting the type of 'sigma' (line 34)
    sigma_127608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 47), 'sigma', False)
    int_127609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 54), 'int')
    # Processing the call keyword arguments (line 34)
    kwargs_127610 = {}
    # Getting the type of 'sndi' (line 34)
    sndi_127605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 10), 'sndi', False)
    # Obtaining the member '_ni_support' of a type (line 34)
    _ni_support_127606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 10), sndi_127605, '_ni_support')
    # Obtaining the member '_normalize_sequence' of a type (line 34)
    _normalize_sequence_127607 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 10), _ni_support_127606, '_normalize_sequence')
    # Calling _normalize_sequence(args, kwargs) (line 34)
    _normalize_sequence_call_result_127611 = invoke(stypy.reporting.localization.Localization(__file__, 34, 10), _normalize_sequence_127607, *[sigma_127608, int_127609], **kwargs_127610)
    
    # Assigning a type to the variable 'out' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'out', _normalize_sequence_call_result_127611)
    
    # Call to assert_equal(...): (line 35)
    # Processing the call arguments (line 35)
    # Getting the type of 'out' (line 35)
    out_127613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 17), 'out', False)
    
    # Obtaining an instance of the builtin type 'list' (line 35)
    list_127614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 35)
    # Adding element type (line 35)
    # Getting the type of 'sigma' (line 35)
    sigma_127615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 23), 'sigma', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 22), list_127614, sigma_127615)
    
    # Processing the call keyword arguments (line 35)
    kwargs_127616 = {}
    # Getting the type of 'assert_equal' (line 35)
    assert_equal_127612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 35)
    assert_equal_call_result_127617 = invoke(stypy.reporting.localization.Localization(__file__, 35, 4), assert_equal_127612, *[out_127613, list_127614], **kwargs_127616)
    
    
    # Assigning a Num to a Name (line 37):
    int_127618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 12), 'int')
    # Assigning a type to the variable 'sigma' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'sigma', int_127618)
    
    # Assigning a Call to a Name (line 38):
    
    # Call to _normalize_sequence(...): (line 38)
    # Processing the call arguments (line 38)
    # Getting the type of 'sigma' (line 38)
    sigma_127622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 47), 'sigma', False)
    int_127623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 54), 'int')
    # Processing the call keyword arguments (line 38)
    kwargs_127624 = {}
    # Getting the type of 'sndi' (line 38)
    sndi_127619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 10), 'sndi', False)
    # Obtaining the member '_ni_support' of a type (line 38)
    _ni_support_127620 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 10), sndi_127619, '_ni_support')
    # Obtaining the member '_normalize_sequence' of a type (line 38)
    _normalize_sequence_127621 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 10), _ni_support_127620, '_normalize_sequence')
    # Calling _normalize_sequence(args, kwargs) (line 38)
    _normalize_sequence_call_result_127625 = invoke(stypy.reporting.localization.Localization(__file__, 38, 10), _normalize_sequence_127621, *[sigma_127622, int_127623], **kwargs_127624)
    
    # Assigning a type to the variable 'out' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'out', _normalize_sequence_call_result_127625)
    
    # Call to assert_equal(...): (line 39)
    # Processing the call arguments (line 39)
    # Getting the type of 'out' (line 39)
    out_127627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 17), 'out', False)
    
    # Obtaining an instance of the builtin type 'list' (line 39)
    list_127628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 39)
    # Adding element type (line 39)
    # Getting the type of 'sigma' (line 39)
    sigma_127629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 23), 'sigma', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 22), list_127628, sigma_127629)
    
    # Processing the call keyword arguments (line 39)
    kwargs_127630 = {}
    # Getting the type of 'assert_equal' (line 39)
    assert_equal_127626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 39)
    assert_equal_call_result_127631 = invoke(stypy.reporting.localization.Localization(__file__, 39, 4), assert_equal_127626, *[out_127627, list_127628], **kwargs_127630)
    
    
    # Assigning a List to a Name (line 41):
    
    # Obtaining an instance of the builtin type 'list' (line 41)
    list_127632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 12), 'list')
    # Adding type elements to the builtin type 'list' instance (line 41)
    # Adding element type (line 41)
    int_127633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 13), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 12), list_127632, int_127633)
    # Adding element type (line 41)
    int_127634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 16), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 12), list_127632, int_127634)
    
    # Assigning a type to the variable 'sigma' (line 41)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'sigma', list_127632)
    
    # Assigning a Call to a Name (line 42):
    
    # Call to _normalize_sequence(...): (line 42)
    # Processing the call arguments (line 42)
    # Getting the type of 'sigma' (line 42)
    sigma_127638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 47), 'sigma', False)
    int_127639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 54), 'int')
    # Processing the call keyword arguments (line 42)
    kwargs_127640 = {}
    # Getting the type of 'sndi' (line 42)
    sndi_127635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 10), 'sndi', False)
    # Obtaining the member '_ni_support' of a type (line 42)
    _ni_support_127636 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 10), sndi_127635, '_ni_support')
    # Obtaining the member '_normalize_sequence' of a type (line 42)
    _normalize_sequence_127637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 10), _ni_support_127636, '_normalize_sequence')
    # Calling _normalize_sequence(args, kwargs) (line 42)
    _normalize_sequence_call_result_127641 = invoke(stypy.reporting.localization.Localization(__file__, 42, 10), _normalize_sequence_127637, *[sigma_127638, int_127639], **kwargs_127640)
    
    # Assigning a type to the variable 'out' (line 42)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'out', _normalize_sequence_call_result_127641)
    
    # Call to assert_equal(...): (line 43)
    # Processing the call arguments (line 43)
    # Getting the type of 'out' (line 43)
    out_127643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 17), 'out', False)
    # Getting the type of 'sigma' (line 43)
    sigma_127644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 22), 'sigma', False)
    # Processing the call keyword arguments (line 43)
    kwargs_127645 = {}
    # Getting the type of 'assert_equal' (line 43)
    assert_equal_127642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 43)
    assert_equal_call_result_127646 = invoke(stypy.reporting.localization.Localization(__file__, 43, 4), assert_equal_127642, *[out_127643, sigma_127644], **kwargs_127645)
    
    
    # Assigning a Call to a Name (line 45):
    
    # Call to normal(...): (line 45)
    # Processing the call keyword arguments (line 45)
    
    # Obtaining an instance of the builtin type 'tuple' (line 45)
    tuple_127650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 31), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 45)
    # Adding element type (line 45)
    int_127651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 31), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 31), tuple_127650, int_127651)
    # Adding element type (line 45)
    int_127652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 36), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 31), tuple_127650, int_127652)
    
    keyword_127653 = tuple_127650
    kwargs_127654 = {'size': keyword_127653}
    # Getting the type of 'np' (line 45)
    np_127647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'np', False)
    # Obtaining the member 'random' of a type (line 45)
    random_127648 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 8), np_127647, 'random')
    # Obtaining the member 'normal' of a type (line 45)
    normal_127649 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 8), random_127648, 'normal')
    # Calling normal(args, kwargs) (line 45)
    normal_call_result_127655 = invoke(stypy.reporting.localization.Localization(__file__, 45, 8), normal_127649, *[], **kwargs_127654)
    
    # Assigning a type to the variable 'x' (line 45)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'x', normal_call_result_127655)
    
    # Assigning a Call to a Name (line 46):
    
    # Call to zeros_like(...): (line 46)
    # Processing the call arguments (line 46)
    # Getting the type of 'x' (line 46)
    x_127658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 27), 'x', False)
    # Processing the call keyword arguments (line 46)
    kwargs_127659 = {}
    # Getting the type of 'np' (line 46)
    np_127656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 13), 'np', False)
    # Obtaining the member 'zeros_like' of a type (line 46)
    zeros_like_127657 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 13), np_127656, 'zeros_like')
    # Calling zeros_like(args, kwargs) (line 46)
    zeros_like_call_result_127660 = invoke(stypy.reporting.localization.Localization(__file__, 46, 13), zeros_like_127657, *[x_127658], **kwargs_127659)
    
    # Assigning a type to the variable 'perlin' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'perlin', zeros_like_call_result_127660)
    
    int_127661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 13), 'int')
    
    # Call to arange(...): (line 47)
    # Processing the call arguments (line 47)
    int_127664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 26), 'int')
    # Processing the call keyword arguments (line 47)
    kwargs_127665 = {}
    # Getting the type of 'np' (line 47)
    np_127662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 16), 'np', False)
    # Obtaining the member 'arange' of a type (line 47)
    arange_127663 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 16), np_127662, 'arange')
    # Calling arange(args, kwargs) (line 47)
    arange_call_result_127666 = invoke(stypy.reporting.localization.Localization(__file__, 47, 16), arange_127663, *[int_127664], **kwargs_127665)
    
    # Applying the binary operator '**' (line 47)
    result_pow_127667 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 13), '**', int_127661, arange_call_result_127666)
    
    # Testing the type of a for loop iterable (line 47)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 47, 4), result_pow_127667)
    # Getting the type of the for loop variable (line 47)
    for_loop_var_127668 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 47, 4), result_pow_127667)
    # Assigning a type to the variable 'i' (line 47)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'i', for_loop_var_127668)
    # SSA begins for a for statement (line 47)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Getting the type of 'perlin' (line 48)
    perlin_127669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'perlin')
    
    # Call to gaussian_filter(...): (line 48)
    # Processing the call arguments (line 48)
    # Getting the type of 'x' (line 48)
    x_127673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 47), 'x', False)
    # Getting the type of 'i' (line 48)
    i_127674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 50), 'i', False)
    # Processing the call keyword arguments (line 48)
    str_127675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 58), 'str', 'wrap')
    keyword_127676 = str_127675
    kwargs_127677 = {'mode': keyword_127676}
    # Getting the type of 'sndi' (line 48)
    sndi_127670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 18), 'sndi', False)
    # Obtaining the member 'filters' of a type (line 48)
    filters_127671 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 18), sndi_127670, 'filters')
    # Obtaining the member 'gaussian_filter' of a type (line 48)
    gaussian_filter_127672 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 18), filters_127671, 'gaussian_filter')
    # Calling gaussian_filter(args, kwargs) (line 48)
    gaussian_filter_call_result_127678 = invoke(stypy.reporting.localization.Localization(__file__, 48, 18), gaussian_filter_127672, *[x_127673, i_127674], **kwargs_127677)
    
    # Getting the type of 'i' (line 48)
    i_127679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 68), 'i')
    int_127680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 71), 'int')
    # Applying the binary operator '**' (line 48)
    result_pow_127681 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 68), '**', i_127679, int_127680)
    
    # Applying the binary operator '*' (line 48)
    result_mul_127682 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 18), '*', gaussian_filter_call_result_127678, result_pow_127681)
    
    # Applying the binary operator '+=' (line 48)
    result_iadd_127683 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 8), '+=', perlin_127669, result_mul_127682)
    # Assigning a type to the variable 'perlin' (line 48)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'perlin', result_iadd_127683)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 50):
    
    # Call to int64(...): (line 50)
    # Processing the call arguments (line 50)
    int_127686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 17), 'int')
    # Processing the call keyword arguments (line 50)
    kwargs_127687 = {}
    # Getting the type of 'np' (line 50)
    np_127684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'np', False)
    # Obtaining the member 'int64' of a type (line 50)
    int64_127685 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 8), np_127684, 'int64')
    # Calling int64(args, kwargs) (line 50)
    int64_call_result_127688 = invoke(stypy.reporting.localization.Localization(__file__, 50, 8), int64_127685, *[int_127686], **kwargs_127687)
    
    # Assigning a type to the variable 'x' (line 50)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'x', int64_call_result_127688)
    
    # Call to _normalize_sequence(...): (line 51)
    # Processing the call arguments (line 51)
    # Getting the type of 'x' (line 51)
    x_127692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 41), 'x', False)
    int_127693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 44), 'int')
    # Processing the call keyword arguments (line 51)
    kwargs_127694 = {}
    # Getting the type of 'sndi' (line 51)
    sndi_127689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'sndi', False)
    # Obtaining the member '_ni_support' of a type (line 51)
    _ni_support_127690 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 4), sndi_127689, '_ni_support')
    # Obtaining the member '_normalize_sequence' of a type (line 51)
    _normalize_sequence_127691 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 4), _ni_support_127690, '_normalize_sequence')
    # Calling _normalize_sequence(args, kwargs) (line 51)
    _normalize_sequence_call_result_127695 = invoke(stypy.reporting.localization.Localization(__file__, 51, 4), _normalize_sequence_127691, *[x_127692, int_127693], **kwargs_127694)
    
    
    # ################# End of 'test_gh_5430(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_gh_5430' in the type store
    # Getting the type of 'stypy_return_type' (line 25)
    stypy_return_type_127696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_127696)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_gh_5430'
    return stypy_return_type_127696

# Assigning a type to the variable 'test_gh_5430' (line 25)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'test_gh_5430', test_gh_5430)

@norecursion
def test_gaussian_kernel1d(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_gaussian_kernel1d'
    module_type_store = module_type_store.open_function_context('test_gaussian_kernel1d', 54, 0, False)
    
    # Passed parameters checking function
    test_gaussian_kernel1d.stypy_localization = localization
    test_gaussian_kernel1d.stypy_type_of_self = None
    test_gaussian_kernel1d.stypy_type_store = module_type_store
    test_gaussian_kernel1d.stypy_function_name = 'test_gaussian_kernel1d'
    test_gaussian_kernel1d.stypy_param_names_list = []
    test_gaussian_kernel1d.stypy_varargs_param_name = None
    test_gaussian_kernel1d.stypy_kwargs_param_name = None
    test_gaussian_kernel1d.stypy_call_defaults = defaults
    test_gaussian_kernel1d.stypy_call_varargs = varargs
    test_gaussian_kernel1d.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_gaussian_kernel1d', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_gaussian_kernel1d', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_gaussian_kernel1d(...)' code ##################

    
    # Assigning a Num to a Name (line 55):
    int_127697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 13), 'int')
    # Assigning a type to the variable 'radius' (line 55)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'radius', int_127697)
    
    # Assigning a Num to a Name (line 56):
    int_127698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 12), 'int')
    # Assigning a type to the variable 'sigma' (line 56)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'sigma', int_127698)
    
    # Assigning a BinOp to a Name (line 57):
    # Getting the type of 'sigma' (line 57)
    sigma_127699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 13), 'sigma')
    # Getting the type of 'sigma' (line 57)
    sigma_127700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 21), 'sigma')
    # Applying the binary operator '*' (line 57)
    result_mul_127701 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 13), '*', sigma_127699, sigma_127700)
    
    # Assigning a type to the variable 'sigma2' (line 57)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'sigma2', result_mul_127701)
    
    # Assigning a Call to a Name (line 58):
    
    # Call to arange(...): (line 58)
    # Processing the call arguments (line 58)
    
    # Getting the type of 'radius' (line 58)
    radius_127704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 19), 'radius', False)
    # Applying the 'usub' unary operator (line 58)
    result___neg___127705 = python_operator(stypy.reporting.localization.Localization(__file__, 58, 18), 'usub', radius_127704)
    
    # Getting the type of 'radius' (line 58)
    radius_127706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 27), 'radius', False)
    int_127707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 36), 'int')
    # Applying the binary operator '+' (line 58)
    result_add_127708 = python_operator(stypy.reporting.localization.Localization(__file__, 58, 27), '+', radius_127706, int_127707)
    
    # Processing the call keyword arguments (line 58)
    # Getting the type of 'np' (line 58)
    np_127709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 45), 'np', False)
    # Obtaining the member 'double' of a type (line 58)
    double_127710 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 45), np_127709, 'double')
    keyword_127711 = double_127710
    kwargs_127712 = {'dtype': keyword_127711}
    # Getting the type of 'np' (line 58)
    np_127702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'np', False)
    # Obtaining the member 'arange' of a type (line 58)
    arange_127703 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 8), np_127702, 'arange')
    # Calling arange(args, kwargs) (line 58)
    arange_call_result_127713 = invoke(stypy.reporting.localization.Localization(__file__, 58, 8), arange_127703, *[result___neg___127705, result_add_127708], **kwargs_127712)
    
    # Assigning a type to the variable 'x' (line 58)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'x', arange_call_result_127713)
    
    # Assigning a Call to a Name (line 59):
    
    # Call to exp(...): (line 59)
    # Processing the call arguments (line 59)
    float_127716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 19), 'float')
    # Getting the type of 'x' (line 59)
    x_127717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 26), 'x', False)
    # Applying the binary operator '*' (line 59)
    result_mul_127718 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 19), '*', float_127716, x_127717)
    
    # Getting the type of 'x' (line 59)
    x_127719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 30), 'x', False)
    # Applying the binary operator '*' (line 59)
    result_mul_127720 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 28), '*', result_mul_127718, x_127719)
    
    # Getting the type of 'sigma2' (line 59)
    sigma2_127721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 34), 'sigma2', False)
    # Applying the binary operator 'div' (line 59)
    result_div_127722 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 32), 'div', result_mul_127720, sigma2_127721)
    
    # Processing the call keyword arguments (line 59)
    kwargs_127723 = {}
    # Getting the type of 'np' (line 59)
    np_127714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 12), 'np', False)
    # Obtaining the member 'exp' of a type (line 59)
    exp_127715 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 12), np_127714, 'exp')
    # Calling exp(args, kwargs) (line 59)
    exp_call_result_127724 = invoke(stypy.reporting.localization.Localization(__file__, 59, 12), exp_127715, *[result_div_127722], **kwargs_127723)
    
    # Assigning a type to the variable 'phi_x' (line 59)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'phi_x', exp_call_result_127724)
    
    # Getting the type of 'phi_x' (line 60)
    phi_x_127725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'phi_x')
    
    # Call to sum(...): (line 60)
    # Processing the call keyword arguments (line 60)
    kwargs_127728 = {}
    # Getting the type of 'phi_x' (line 60)
    phi_x_127726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 13), 'phi_x', False)
    # Obtaining the member 'sum' of a type (line 60)
    sum_127727 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 13), phi_x_127726, 'sum')
    # Calling sum(args, kwargs) (line 60)
    sum_call_result_127729 = invoke(stypy.reporting.localization.Localization(__file__, 60, 13), sum_127727, *[], **kwargs_127728)
    
    # Applying the binary operator 'div=' (line 60)
    result_div_127730 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 4), 'div=', phi_x_127725, sum_call_result_127729)
    # Assigning a type to the variable 'phi_x' (line 60)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'phi_x', result_div_127730)
    
    
    # Call to assert_allclose(...): (line 61)
    # Processing the call arguments (line 61)
    # Getting the type of 'phi_x' (line 61)
    phi_x_127732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 20), 'phi_x', False)
    
    # Call to _gaussian_kernel1d(...): (line 61)
    # Processing the call arguments (line 61)
    # Getting the type of 'sigma' (line 61)
    sigma_127734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 46), 'sigma', False)
    int_127735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 53), 'int')
    # Getting the type of 'radius' (line 61)
    radius_127736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 56), 'radius', False)
    # Processing the call keyword arguments (line 61)
    kwargs_127737 = {}
    # Getting the type of '_gaussian_kernel1d' (line 61)
    _gaussian_kernel1d_127733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 27), '_gaussian_kernel1d', False)
    # Calling _gaussian_kernel1d(args, kwargs) (line 61)
    _gaussian_kernel1d_call_result_127738 = invoke(stypy.reporting.localization.Localization(__file__, 61, 27), _gaussian_kernel1d_127733, *[sigma_127734, int_127735, radius_127736], **kwargs_127737)
    
    # Processing the call keyword arguments (line 61)
    kwargs_127739 = {}
    # Getting the type of 'assert_allclose' (line 61)
    assert_allclose_127731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 61)
    assert_allclose_call_result_127740 = invoke(stypy.reporting.localization.Localization(__file__, 61, 4), assert_allclose_127731, *[phi_x_127732, _gaussian_kernel1d_call_result_127738], **kwargs_127739)
    
    
    # Call to assert_allclose(...): (line 62)
    # Processing the call arguments (line 62)
    
    # Getting the type of 'phi_x' (line 62)
    phi_x_127742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 21), 'phi_x', False)
    # Applying the 'usub' unary operator (line 62)
    result___neg___127743 = python_operator(stypy.reporting.localization.Localization(__file__, 62, 20), 'usub', phi_x_127742)
    
    # Getting the type of 'x' (line 62)
    x_127744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 29), 'x', False)
    # Applying the binary operator '*' (line 62)
    result_mul_127745 = python_operator(stypy.reporting.localization.Localization(__file__, 62, 20), '*', result___neg___127743, x_127744)
    
    # Getting the type of 'sigma2' (line 62)
    sigma2_127746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 33), 'sigma2', False)
    # Applying the binary operator 'div' (line 62)
    result_div_127747 = python_operator(stypy.reporting.localization.Localization(__file__, 62, 31), 'div', result_mul_127745, sigma2_127746)
    
    
    # Call to _gaussian_kernel1d(...): (line 62)
    # Processing the call arguments (line 62)
    # Getting the type of 'sigma' (line 62)
    sigma_127749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 60), 'sigma', False)
    int_127750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 67), 'int')
    # Getting the type of 'radius' (line 62)
    radius_127751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 70), 'radius', False)
    # Processing the call keyword arguments (line 62)
    kwargs_127752 = {}
    # Getting the type of '_gaussian_kernel1d' (line 62)
    _gaussian_kernel1d_127748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 41), '_gaussian_kernel1d', False)
    # Calling _gaussian_kernel1d(args, kwargs) (line 62)
    _gaussian_kernel1d_call_result_127753 = invoke(stypy.reporting.localization.Localization(__file__, 62, 41), _gaussian_kernel1d_127748, *[sigma_127749, int_127750, radius_127751], **kwargs_127752)
    
    # Processing the call keyword arguments (line 62)
    kwargs_127754 = {}
    # Getting the type of 'assert_allclose' (line 62)
    assert_allclose_127741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 62)
    assert_allclose_call_result_127755 = invoke(stypy.reporting.localization.Localization(__file__, 62, 4), assert_allclose_127741, *[result_div_127747, _gaussian_kernel1d_call_result_127753], **kwargs_127754)
    
    
    # Call to assert_allclose(...): (line 63)
    # Processing the call arguments (line 63)
    # Getting the type of 'phi_x' (line 63)
    phi_x_127757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 20), 'phi_x', False)
    # Getting the type of 'x' (line 63)
    x_127758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 29), 'x', False)
    # Getting the type of 'x' (line 63)
    x_127759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 33), 'x', False)
    # Applying the binary operator '*' (line 63)
    result_mul_127760 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 29), '*', x_127758, x_127759)
    
    # Getting the type of 'sigma2' (line 63)
    sigma2_127761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 37), 'sigma2', False)
    # Applying the binary operator 'div' (line 63)
    result_div_127762 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 35), 'div', result_mul_127760, sigma2_127761)
    
    int_127763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 46), 'int')
    # Applying the binary operator '-' (line 63)
    result_sub_127764 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 29), '-', result_div_127762, int_127763)
    
    # Applying the binary operator '*' (line 63)
    result_mul_127765 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 20), '*', phi_x_127757, result_sub_127764)
    
    # Getting the type of 'sigma2' (line 63)
    sigma2_127766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 51), 'sigma2', False)
    # Applying the binary operator 'div' (line 63)
    result_div_127767 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 49), 'div', result_mul_127765, sigma2_127766)
    
    
    # Call to _gaussian_kernel1d(...): (line 64)
    # Processing the call arguments (line 64)
    # Getting the type of 'sigma' (line 64)
    sigma_127769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 39), 'sigma', False)
    int_127770 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 46), 'int')
    # Getting the type of 'radius' (line 64)
    radius_127771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 49), 'radius', False)
    # Processing the call keyword arguments (line 64)
    kwargs_127772 = {}
    # Getting the type of '_gaussian_kernel1d' (line 64)
    _gaussian_kernel1d_127768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 20), '_gaussian_kernel1d', False)
    # Calling _gaussian_kernel1d(args, kwargs) (line 64)
    _gaussian_kernel1d_call_result_127773 = invoke(stypy.reporting.localization.Localization(__file__, 64, 20), _gaussian_kernel1d_127768, *[sigma_127769, int_127770, radius_127771], **kwargs_127772)
    
    # Processing the call keyword arguments (line 63)
    kwargs_127774 = {}
    # Getting the type of 'assert_allclose' (line 63)
    assert_allclose_127756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 63)
    assert_allclose_call_result_127775 = invoke(stypy.reporting.localization.Localization(__file__, 63, 4), assert_allclose_127756, *[result_div_127767, _gaussian_kernel1d_call_result_127773], **kwargs_127774)
    
    
    # Call to assert_allclose(...): (line 65)
    # Processing the call arguments (line 65)
    # Getting the type of 'phi_x' (line 65)
    phi_x_127777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 20), 'phi_x', False)
    int_127778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 29), 'int')
    # Getting the type of 'x' (line 65)
    x_127779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 33), 'x', False)
    # Getting the type of 'x' (line 65)
    x_127780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 37), 'x', False)
    # Applying the binary operator '*' (line 65)
    result_mul_127781 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 33), '*', x_127779, x_127780)
    
    # Getting the type of 'sigma2' (line 65)
    sigma2_127782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 41), 'sigma2', False)
    # Applying the binary operator 'div' (line 65)
    result_div_127783 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 39), 'div', result_mul_127781, sigma2_127782)
    
    # Applying the binary operator '-' (line 65)
    result_sub_127784 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 29), '-', int_127778, result_div_127783)
    
    # Applying the binary operator '*' (line 65)
    result_mul_127785 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 20), '*', phi_x_127777, result_sub_127784)
    
    # Getting the type of 'x' (line 65)
    x_127786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 51), 'x', False)
    # Applying the binary operator '*' (line 65)
    result_mul_127787 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 49), '*', result_mul_127785, x_127786)
    
    # Getting the type of 'sigma2' (line 65)
    sigma2_127788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 56), 'sigma2', False)
    # Getting the type of 'sigma2' (line 65)
    sigma2_127789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 65), 'sigma2', False)
    # Applying the binary operator '*' (line 65)
    result_mul_127790 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 56), '*', sigma2_127788, sigma2_127789)
    
    # Applying the binary operator 'div' (line 65)
    result_div_127791 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 53), 'div', result_mul_127787, result_mul_127790)
    
    
    # Call to _gaussian_kernel1d(...): (line 66)
    # Processing the call arguments (line 66)
    # Getting the type of 'sigma' (line 66)
    sigma_127793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 39), 'sigma', False)
    int_127794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 46), 'int')
    # Getting the type of 'radius' (line 66)
    radius_127795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 49), 'radius', False)
    # Processing the call keyword arguments (line 66)
    kwargs_127796 = {}
    # Getting the type of '_gaussian_kernel1d' (line 66)
    _gaussian_kernel1d_127792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 20), '_gaussian_kernel1d', False)
    # Calling _gaussian_kernel1d(args, kwargs) (line 66)
    _gaussian_kernel1d_call_result_127797 = invoke(stypy.reporting.localization.Localization(__file__, 66, 20), _gaussian_kernel1d_127792, *[sigma_127793, int_127794, radius_127795], **kwargs_127796)
    
    # Processing the call keyword arguments (line 65)
    kwargs_127798 = {}
    # Getting the type of 'assert_allclose' (line 65)
    assert_allclose_127776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 65)
    assert_allclose_call_result_127799 = invoke(stypy.reporting.localization.Localization(__file__, 65, 4), assert_allclose_127776, *[result_div_127791, _gaussian_kernel1d_call_result_127797], **kwargs_127798)
    
    
    # ################# End of 'test_gaussian_kernel1d(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_gaussian_kernel1d' in the type store
    # Getting the type of 'stypy_return_type' (line 54)
    stypy_return_type_127800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_127800)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_gaussian_kernel1d'
    return stypy_return_type_127800

# Assigning a type to the variable 'test_gaussian_kernel1d' (line 54)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 0), 'test_gaussian_kernel1d', test_gaussian_kernel1d)

@norecursion
def test_orders_gauss(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_orders_gauss'
    module_type_store = module_type_store.open_function_context('test_orders_gauss', 69, 0, False)
    
    # Passed parameters checking function
    test_orders_gauss.stypy_localization = localization
    test_orders_gauss.stypy_type_of_self = None
    test_orders_gauss.stypy_type_store = module_type_store
    test_orders_gauss.stypy_function_name = 'test_orders_gauss'
    test_orders_gauss.stypy_param_names_list = []
    test_orders_gauss.stypy_varargs_param_name = None
    test_orders_gauss.stypy_kwargs_param_name = None
    test_orders_gauss.stypy_call_defaults = defaults
    test_orders_gauss.stypy_call_varargs = varargs
    test_orders_gauss.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_orders_gauss', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_orders_gauss', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_orders_gauss(...)' code ##################

    
    # Assigning a Call to a Name (line 71):
    
    # Call to zeros(...): (line 71)
    # Processing the call arguments (line 71)
    
    # Obtaining an instance of the builtin type 'tuple' (line 71)
    tuple_127803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 20), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 71)
    # Adding element type (line 71)
    int_127804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 20), tuple_127803, int_127804)
    
    # Processing the call keyword arguments (line 71)
    kwargs_127805 = {}
    # Getting the type of 'np' (line 71)
    np_127801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 10), 'np', False)
    # Obtaining the member 'zeros' of a type (line 71)
    zeros_127802 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 10), np_127801, 'zeros')
    # Calling zeros(args, kwargs) (line 71)
    zeros_call_result_127806 = invoke(stypy.reporting.localization.Localization(__file__, 71, 10), zeros_127802, *[tuple_127803], **kwargs_127805)
    
    # Assigning a type to the variable 'arr' (line 71)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 4), 'arr', zeros_call_result_127806)
    
    # Call to assert_equal(...): (line 72)
    # Processing the call arguments (line 72)
    int_127808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 17), 'int')
    
    # Call to gaussian_filter(...): (line 72)
    # Processing the call arguments (line 72)
    # Getting the type of 'arr' (line 72)
    arr_127811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 41), 'arr', False)
    int_127812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 46), 'int')
    # Processing the call keyword arguments (line 72)
    int_127813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 55), 'int')
    keyword_127814 = int_127813
    kwargs_127815 = {'order': keyword_127814}
    # Getting the type of 'sndi' (line 72)
    sndi_127809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 20), 'sndi', False)
    # Obtaining the member 'gaussian_filter' of a type (line 72)
    gaussian_filter_127810 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 20), sndi_127809, 'gaussian_filter')
    # Calling gaussian_filter(args, kwargs) (line 72)
    gaussian_filter_call_result_127816 = invoke(stypy.reporting.localization.Localization(__file__, 72, 20), gaussian_filter_127810, *[arr_127811, int_127812], **kwargs_127815)
    
    # Processing the call keyword arguments (line 72)
    kwargs_127817 = {}
    # Getting the type of 'assert_equal' (line 72)
    assert_equal_127807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 72)
    assert_equal_call_result_127818 = invoke(stypy.reporting.localization.Localization(__file__, 72, 4), assert_equal_127807, *[int_127808, gaussian_filter_call_result_127816], **kwargs_127817)
    
    
    # Call to assert_equal(...): (line 73)
    # Processing the call arguments (line 73)
    int_127820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 17), 'int')
    
    # Call to gaussian_filter(...): (line 73)
    # Processing the call arguments (line 73)
    # Getting the type of 'arr' (line 73)
    arr_127823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 41), 'arr', False)
    int_127824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 46), 'int')
    # Processing the call keyword arguments (line 73)
    int_127825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 55), 'int')
    keyword_127826 = int_127825
    kwargs_127827 = {'order': keyword_127826}
    # Getting the type of 'sndi' (line 73)
    sndi_127821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 20), 'sndi', False)
    # Obtaining the member 'gaussian_filter' of a type (line 73)
    gaussian_filter_127822 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 20), sndi_127821, 'gaussian_filter')
    # Calling gaussian_filter(args, kwargs) (line 73)
    gaussian_filter_call_result_127828 = invoke(stypy.reporting.localization.Localization(__file__, 73, 20), gaussian_filter_127822, *[arr_127823, int_127824], **kwargs_127827)
    
    # Processing the call keyword arguments (line 73)
    kwargs_127829 = {}
    # Getting the type of 'assert_equal' (line 73)
    assert_equal_127819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 73)
    assert_equal_call_result_127830 = invoke(stypy.reporting.localization.Localization(__file__, 73, 4), assert_equal_127819, *[int_127820, gaussian_filter_call_result_127828], **kwargs_127829)
    
    
    # Call to assert_raises(...): (line 74)
    # Processing the call arguments (line 74)
    # Getting the type of 'ValueError' (line 74)
    ValueError_127832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 18), 'ValueError', False)
    # Getting the type of 'sndi' (line 74)
    sndi_127833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 30), 'sndi', False)
    # Obtaining the member 'gaussian_filter' of a type (line 74)
    gaussian_filter_127834 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 30), sndi_127833, 'gaussian_filter')
    # Getting the type of 'arr' (line 74)
    arr_127835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 52), 'arr', False)
    int_127836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 57), 'int')
    int_127837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 60), 'int')
    # Processing the call keyword arguments (line 74)
    kwargs_127838 = {}
    # Getting the type of 'assert_raises' (line 74)
    assert_raises_127831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 74)
    assert_raises_call_result_127839 = invoke(stypy.reporting.localization.Localization(__file__, 74, 4), assert_raises_127831, *[ValueError_127832, gaussian_filter_127834, arr_127835, int_127836, int_127837], **kwargs_127838)
    
    
    # Call to assert_equal(...): (line 75)
    # Processing the call arguments (line 75)
    int_127841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 17), 'int')
    
    # Call to gaussian_filter1d(...): (line 75)
    # Processing the call arguments (line 75)
    # Getting the type of 'arr' (line 75)
    arr_127844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 43), 'arr', False)
    int_127845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 48), 'int')
    # Processing the call keyword arguments (line 75)
    int_127846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 56), 'int')
    keyword_127847 = int_127846
    int_127848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 66), 'int')
    keyword_127849 = int_127848
    kwargs_127850 = {'order': keyword_127849, 'axis': keyword_127847}
    # Getting the type of 'sndi' (line 75)
    sndi_127842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 20), 'sndi', False)
    # Obtaining the member 'gaussian_filter1d' of a type (line 75)
    gaussian_filter1d_127843 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 20), sndi_127842, 'gaussian_filter1d')
    # Calling gaussian_filter1d(args, kwargs) (line 75)
    gaussian_filter1d_call_result_127851 = invoke(stypy.reporting.localization.Localization(__file__, 75, 20), gaussian_filter1d_127843, *[arr_127844, int_127845], **kwargs_127850)
    
    # Processing the call keyword arguments (line 75)
    kwargs_127852 = {}
    # Getting the type of 'assert_equal' (line 75)
    assert_equal_127840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 75)
    assert_equal_call_result_127853 = invoke(stypy.reporting.localization.Localization(__file__, 75, 4), assert_equal_127840, *[int_127841, gaussian_filter1d_call_result_127851], **kwargs_127852)
    
    
    # Call to assert_equal(...): (line 76)
    # Processing the call arguments (line 76)
    int_127855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 17), 'int')
    
    # Call to gaussian_filter1d(...): (line 76)
    # Processing the call arguments (line 76)
    # Getting the type of 'arr' (line 76)
    arr_127858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 43), 'arr', False)
    int_127859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 48), 'int')
    # Processing the call keyword arguments (line 76)
    int_127860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 56), 'int')
    keyword_127861 = int_127860
    int_127862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 66), 'int')
    keyword_127863 = int_127862
    kwargs_127864 = {'order': keyword_127863, 'axis': keyword_127861}
    # Getting the type of 'sndi' (line 76)
    sndi_127856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 20), 'sndi', False)
    # Obtaining the member 'gaussian_filter1d' of a type (line 76)
    gaussian_filter1d_127857 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 20), sndi_127856, 'gaussian_filter1d')
    # Calling gaussian_filter1d(args, kwargs) (line 76)
    gaussian_filter1d_call_result_127865 = invoke(stypy.reporting.localization.Localization(__file__, 76, 20), gaussian_filter1d_127857, *[arr_127858, int_127859], **kwargs_127864)
    
    # Processing the call keyword arguments (line 76)
    kwargs_127866 = {}
    # Getting the type of 'assert_equal' (line 76)
    assert_equal_127854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 76)
    assert_equal_call_result_127867 = invoke(stypy.reporting.localization.Localization(__file__, 76, 4), assert_equal_127854, *[int_127855, gaussian_filter1d_call_result_127865], **kwargs_127866)
    
    
    # Call to assert_raises(...): (line 77)
    # Processing the call arguments (line 77)
    # Getting the type of 'ValueError' (line 77)
    ValueError_127869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 18), 'ValueError', False)
    # Getting the type of 'sndi' (line 77)
    sndi_127870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 30), 'sndi', False)
    # Obtaining the member 'gaussian_filter1d' of a type (line 77)
    gaussian_filter1d_127871 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 30), sndi_127870, 'gaussian_filter1d')
    # Getting the type of 'arr' (line 77)
    arr_127872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 54), 'arr', False)
    int_127873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 59), 'int')
    int_127874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 62), 'int')
    int_127875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 66), 'int')
    # Processing the call keyword arguments (line 77)
    kwargs_127876 = {}
    # Getting the type of 'assert_raises' (line 77)
    assert_raises_127868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 4), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 77)
    assert_raises_call_result_127877 = invoke(stypy.reporting.localization.Localization(__file__, 77, 4), assert_raises_127868, *[ValueError_127869, gaussian_filter1d_127871, arr_127872, int_127873, int_127874, int_127875], **kwargs_127876)
    
    
    # ################# End of 'test_orders_gauss(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_orders_gauss' in the type store
    # Getting the type of 'stypy_return_type' (line 69)
    stypy_return_type_127878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_127878)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_orders_gauss'
    return stypy_return_type_127878

# Assigning a type to the variable 'test_orders_gauss' (line 69)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 0), 'test_orders_gauss', test_orders_gauss)

@norecursion
def test_valid_origins(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_valid_origins'
    module_type_store = module_type_store.open_function_context('test_valid_origins', 80, 0, False)
    
    # Passed parameters checking function
    test_valid_origins.stypy_localization = localization
    test_valid_origins.stypy_type_of_self = None
    test_valid_origins.stypy_type_store = module_type_store
    test_valid_origins.stypy_function_name = 'test_valid_origins'
    test_valid_origins.stypy_param_names_list = []
    test_valid_origins.stypy_varargs_param_name = None
    test_valid_origins.stypy_kwargs_param_name = None
    test_valid_origins.stypy_call_defaults = defaults
    test_valid_origins.stypy_call_varargs = varargs
    test_valid_origins.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_valid_origins', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_valid_origins', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_valid_origins(...)' code ##################

    str_127879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 4), 'str', 'Regression test for #1311.')
    
    # Assigning a Lambda to a Name (line 82):

    @norecursion
    def _stypy_temp_lambda_46(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_46'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_46', 82, 11, True)
        # Passed parameters checking function
        _stypy_temp_lambda_46.stypy_localization = localization
        _stypy_temp_lambda_46.stypy_type_of_self = None
        _stypy_temp_lambda_46.stypy_type_store = module_type_store
        _stypy_temp_lambda_46.stypy_function_name = '_stypy_temp_lambda_46'
        _stypy_temp_lambda_46.stypy_param_names_list = ['x']
        _stypy_temp_lambda_46.stypy_varargs_param_name = None
        _stypy_temp_lambda_46.stypy_kwargs_param_name = None
        _stypy_temp_lambda_46.stypy_call_defaults = defaults
        _stypy_temp_lambda_46.stypy_call_varargs = varargs
        _stypy_temp_lambda_46.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_46', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Stacktrace push for error reporting
        localization.set_stack_trace('_stypy_temp_lambda_46', ['x'], arguments)
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of the lambda function code ##################

        
        # Call to mean(...): (line 82)
        # Processing the call arguments (line 82)
        # Getting the type of 'x' (line 82)
        x_127882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 29), 'x', False)
        # Processing the call keyword arguments (line 82)
        kwargs_127883 = {}
        # Getting the type of 'np' (line 82)
        np_127880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 21), 'np', False)
        # Obtaining the member 'mean' of a type (line 82)
        mean_127881 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 21), np_127880, 'mean')
        # Calling mean(args, kwargs) (line 82)
        mean_call_result_127884 = invoke(stypy.reporting.localization.Localization(__file__, 82, 21), mean_127881, *[x_127882], **kwargs_127883)
        
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 11), 'stypy_return_type', mean_call_result_127884)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_46' in the type store
        # Getting the type of 'stypy_return_type' (line 82)
        stypy_return_type_127885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 11), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_127885)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_46'
        return stypy_return_type_127885

    # Assigning a type to the variable '_stypy_temp_lambda_46' (line 82)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 11), '_stypy_temp_lambda_46', _stypy_temp_lambda_46)
    # Getting the type of '_stypy_temp_lambda_46' (line 82)
    _stypy_temp_lambda_46_127886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 11), '_stypy_temp_lambda_46')
    # Assigning a type to the variable 'func' (line 82)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'func', _stypy_temp_lambda_46_127886)
    
    # Assigning a Call to a Name (line 83):
    
    # Call to array(...): (line 83)
    # Processing the call arguments (line 83)
    
    # Obtaining an instance of the builtin type 'list' (line 83)
    list_127889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 83)
    # Adding element type (line 83)
    int_127890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 20), list_127889, int_127890)
    # Adding element type (line 83)
    int_127891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 20), list_127889, int_127891)
    # Adding element type (line 83)
    int_127892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 20), list_127889, int_127892)
    # Adding element type (line 83)
    int_127893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 20), list_127889, int_127893)
    # Adding element type (line 83)
    int_127894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 29), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 20), list_127889, int_127894)
    
    # Processing the call keyword arguments (line 83)
    # Getting the type of 'np' (line 83)
    np_127895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 39), 'np', False)
    # Obtaining the member 'float64' of a type (line 83)
    float64_127896 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 39), np_127895, 'float64')
    keyword_127897 = float64_127896
    kwargs_127898 = {'dtype': keyword_127897}
    # Getting the type of 'np' (line 83)
    np_127887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 11), 'np', False)
    # Obtaining the member 'array' of a type (line 83)
    array_127888 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 11), np_127887, 'array')
    # Calling array(args, kwargs) (line 83)
    array_call_result_127899 = invoke(stypy.reporting.localization.Localization(__file__, 83, 11), array_127888, *[list_127889], **kwargs_127898)
    
    # Assigning a type to the variable 'data' (line 83)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 4), 'data', array_call_result_127899)
    
    # Call to assert_raises(...): (line 84)
    # Processing the call arguments (line 84)
    # Getting the type of 'ValueError' (line 84)
    ValueError_127901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 18), 'ValueError', False)
    # Getting the type of 'sndi' (line 84)
    sndi_127902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 30), 'sndi', False)
    # Obtaining the member 'generic_filter' of a type (line 84)
    generic_filter_127903 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 30), sndi_127902, 'generic_filter')
    # Getting the type of 'data' (line 84)
    data_127904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 51), 'data', False)
    # Getting the type of 'func' (line 84)
    func_127905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 57), 'func', False)
    # Processing the call keyword arguments (line 84)
    int_127906 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 68), 'int')
    keyword_127907 = int_127906
    int_127908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 25), 'int')
    keyword_127909 = int_127908
    kwargs_127910 = {'origin': keyword_127909, 'size': keyword_127907}
    # Getting the type of 'assert_raises' (line 84)
    assert_raises_127900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 4), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 84)
    assert_raises_call_result_127911 = invoke(stypy.reporting.localization.Localization(__file__, 84, 4), assert_raises_127900, *[ValueError_127901, generic_filter_127903, data_127904, func_127905], **kwargs_127910)
    
    
    # Assigning a Lambda to a Name (line 86):

    @norecursion
    def _stypy_temp_lambda_47(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_47'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_47', 86, 12, True)
        # Passed parameters checking function
        _stypy_temp_lambda_47.stypy_localization = localization
        _stypy_temp_lambda_47.stypy_type_of_self = None
        _stypy_temp_lambda_47.stypy_type_store = module_type_store
        _stypy_temp_lambda_47.stypy_function_name = '_stypy_temp_lambda_47'
        _stypy_temp_lambda_47.stypy_param_names_list = ['x', 'y']
        _stypy_temp_lambda_47.stypy_varargs_param_name = None
        _stypy_temp_lambda_47.stypy_kwargs_param_name = None
        _stypy_temp_lambda_47.stypy_call_defaults = defaults
        _stypy_temp_lambda_47.stypy_call_varargs = varargs
        _stypy_temp_lambda_47.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_47', ['x', 'y'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Stacktrace push for error reporting
        localization.set_stack_trace('_stypy_temp_lambda_47', ['x', 'y'], arguments)
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of the lambda function code ##################

        
        # Call to mean(...): (line 86)
        # Processing the call arguments (line 86)
        # Getting the type of 'x' (line 86)
        x_127914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 33), 'x', False)
        # Getting the type of 'y' (line 86)
        y_127915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 37), 'y', False)
        # Applying the binary operator '+' (line 86)
        result_add_127916 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 33), '+', x_127914, y_127915)
        
        # Processing the call keyword arguments (line 86)
        kwargs_127917 = {}
        # Getting the type of 'np' (line 86)
        np_127912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 25), 'np', False)
        # Obtaining the member 'mean' of a type (line 86)
        mean_127913 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 25), np_127912, 'mean')
        # Calling mean(args, kwargs) (line 86)
        mean_call_result_127918 = invoke(stypy.reporting.localization.Localization(__file__, 86, 25), mean_127913, *[result_add_127916], **kwargs_127917)
        
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 12), 'stypy_return_type', mean_call_result_127918)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_47' in the type store
        # Getting the type of 'stypy_return_type' (line 86)
        stypy_return_type_127919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 12), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_127919)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_47'
        return stypy_return_type_127919

    # Assigning a type to the variable '_stypy_temp_lambda_47' (line 86)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 12), '_stypy_temp_lambda_47', _stypy_temp_lambda_47)
    # Getting the type of '_stypy_temp_lambda_47' (line 86)
    _stypy_temp_lambda_47_127920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 12), '_stypy_temp_lambda_47')
    # Assigning a type to the variable 'func2' (line 86)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'func2', _stypy_temp_lambda_47_127920)
    
    # Call to assert_raises(...): (line 87)
    # Processing the call arguments (line 87)
    # Getting the type of 'ValueError' (line 87)
    ValueError_127922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 18), 'ValueError', False)
    # Getting the type of 'sndi' (line 87)
    sndi_127923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 30), 'sndi', False)
    # Obtaining the member 'generic_filter1d' of a type (line 87)
    generic_filter1d_127924 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 30), sndi_127923, 'generic_filter1d')
    # Getting the type of 'data' (line 87)
    data_127925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 53), 'data', False)
    # Getting the type of 'func' (line 87)
    func_127926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 59), 'func', False)
    # Processing the call keyword arguments (line 87)
    int_127927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 30), 'int')
    keyword_127928 = int_127927
    int_127929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 40), 'int')
    keyword_127930 = int_127929
    kwargs_127931 = {'origin': keyword_127930, 'filter_size': keyword_127928}
    # Getting the type of 'assert_raises' (line 87)
    assert_raises_127921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 4), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 87)
    assert_raises_call_result_127932 = invoke(stypy.reporting.localization.Localization(__file__, 87, 4), assert_raises_127921, *[ValueError_127922, generic_filter1d_127924, data_127925, func_127926], **kwargs_127931)
    
    
    # Call to assert_raises(...): (line 89)
    # Processing the call arguments (line 89)
    # Getting the type of 'ValueError' (line 89)
    ValueError_127934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 18), 'ValueError', False)
    # Getting the type of 'sndi' (line 89)
    sndi_127935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 30), 'sndi', False)
    # Obtaining the member 'percentile_filter' of a type (line 89)
    percentile_filter_127936 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 30), sndi_127935, 'percentile_filter')
    # Getting the type of 'data' (line 89)
    data_127937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 54), 'data', False)
    float_127938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 60), 'float')
    # Processing the call keyword arguments (line 89)
    int_127939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 70), 'int')
    keyword_127940 = int_127939
    int_127941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 25), 'int')
    keyword_127942 = int_127941
    kwargs_127943 = {'origin': keyword_127942, 'size': keyword_127940}
    # Getting the type of 'assert_raises' (line 89)
    assert_raises_127933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 89)
    assert_raises_call_result_127944 = invoke(stypy.reporting.localization.Localization(__file__, 89, 4), assert_raises_127933, *[ValueError_127934, percentile_filter_127936, data_127937, float_127938], **kwargs_127943)
    
    
    
    # Obtaining an instance of the builtin type 'list' (line 92)
    list_127945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 92)
    # Adding element type (line 92)
    # Getting the type of 'sndi' (line 92)
    sndi_127946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 19), 'sndi')
    # Obtaining the member 'uniform_filter' of a type (line 92)
    uniform_filter_127947 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 19), sndi_127946, 'uniform_filter')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 18), list_127945, uniform_filter_127947)
    # Adding element type (line 92)
    # Getting the type of 'sndi' (line 92)
    sndi_127948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 40), 'sndi')
    # Obtaining the member 'minimum_filter' of a type (line 92)
    minimum_filter_127949 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 40), sndi_127948, 'minimum_filter')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 18), list_127945, minimum_filter_127949)
    # Adding element type (line 92)
    # Getting the type of 'sndi' (line 93)
    sndi_127950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 19), 'sndi')
    # Obtaining the member 'maximum_filter' of a type (line 93)
    maximum_filter_127951 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 19), sndi_127950, 'maximum_filter')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 18), list_127945, maximum_filter_127951)
    # Adding element type (line 92)
    # Getting the type of 'sndi' (line 93)
    sndi_127952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 40), 'sndi')
    # Obtaining the member 'maximum_filter1d' of a type (line 93)
    maximum_filter1d_127953 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 40), sndi_127952, 'maximum_filter1d')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 18), list_127945, maximum_filter1d_127953)
    # Adding element type (line 92)
    # Getting the type of 'sndi' (line 94)
    sndi_127954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 19), 'sndi')
    # Obtaining the member 'median_filter' of a type (line 94)
    median_filter_127955 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 19), sndi_127954, 'median_filter')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 18), list_127945, median_filter_127955)
    # Adding element type (line 92)
    # Getting the type of 'sndi' (line 94)
    sndi_127956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 39), 'sndi')
    # Obtaining the member 'minimum_filter1d' of a type (line 94)
    minimum_filter1d_127957 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 39), sndi_127956, 'minimum_filter1d')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 18), list_127945, minimum_filter1d_127957)
    
    # Testing the type of a for loop iterable (line 92)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 92, 4), list_127945)
    # Getting the type of the for loop variable (line 92)
    for_loop_var_127958 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 92, 4), list_127945)
    # Assigning a type to the variable 'filter' (line 92)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), 'filter', for_loop_var_127958)
    # SSA begins for a for statement (line 92)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to list(...): (line 97)
    # Processing the call arguments (line 97)
    
    # Call to filter(...): (line 97)
    # Processing the call arguments (line 97)
    # Getting the type of 'data' (line 97)
    data_127961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 20), 'data', False)
    int_127962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 26), 'int')
    # Processing the call keyword arguments (line 97)
    int_127963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 36), 'int')
    keyword_127964 = int_127963
    kwargs_127965 = {'origin': keyword_127964}
    # Getting the type of 'filter' (line 97)
    filter_127960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 13), 'filter', False)
    # Calling filter(args, kwargs) (line 97)
    filter_call_result_127966 = invoke(stypy.reporting.localization.Localization(__file__, 97, 13), filter_127960, *[data_127961, int_127962], **kwargs_127965)
    
    # Processing the call keyword arguments (line 97)
    kwargs_127967 = {}
    # Getting the type of 'list' (line 97)
    list_127959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'list', False)
    # Calling list(args, kwargs) (line 97)
    list_call_result_127968 = invoke(stypy.reporting.localization.Localization(__file__, 97, 8), list_127959, *[filter_call_result_127966], **kwargs_127967)
    
    
    # Call to list(...): (line 98)
    # Processing the call arguments (line 98)
    
    # Call to filter(...): (line 98)
    # Processing the call arguments (line 98)
    # Getting the type of 'data' (line 98)
    data_127971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 20), 'data', False)
    int_127972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 26), 'int')
    # Processing the call keyword arguments (line 98)
    int_127973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 36), 'int')
    keyword_127974 = int_127973
    kwargs_127975 = {'origin': keyword_127974}
    # Getting the type of 'filter' (line 98)
    filter_127970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 13), 'filter', False)
    # Calling filter(args, kwargs) (line 98)
    filter_call_result_127976 = invoke(stypy.reporting.localization.Localization(__file__, 98, 13), filter_127970, *[data_127971, int_127972], **kwargs_127975)
    
    # Processing the call keyword arguments (line 98)
    kwargs_127977 = {}
    # Getting the type of 'list' (line 98)
    list_127969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'list', False)
    # Calling list(args, kwargs) (line 98)
    list_call_result_127978 = invoke(stypy.reporting.localization.Localization(__file__, 98, 8), list_127969, *[filter_call_result_127976], **kwargs_127977)
    
    
    # Call to assert_raises(...): (line 101)
    # Processing the call arguments (line 101)
    # Getting the type of 'ValueError' (line 101)
    ValueError_127980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 22), 'ValueError', False)
    # Getting the type of 'filter' (line 101)
    filter_127981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 34), 'filter', False)
    # Getting the type of 'data' (line 101)
    data_127982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 42), 'data', False)
    int_127983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 48), 'int')
    # Processing the call keyword arguments (line 101)
    int_127984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 58), 'int')
    keyword_127985 = int_127984
    kwargs_127986 = {'origin': keyword_127985}
    # Getting the type of 'assert_raises' (line 101)
    assert_raises_127979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 101)
    assert_raises_call_result_127987 = invoke(stypy.reporting.localization.Localization(__file__, 101, 8), assert_raises_127979, *[ValueError_127980, filter_127981, data_127982, int_127983], **kwargs_127986)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_valid_origins(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_valid_origins' in the type store
    # Getting the type of 'stypy_return_type' (line 80)
    stypy_return_type_127988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_127988)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_valid_origins'
    return stypy_return_type_127988

# Assigning a type to the variable 'test_valid_origins' (line 80)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 0), 'test_valid_origins', test_valid_origins)

@norecursion
def test_multiple_modes(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_multiple_modes'
    module_type_store = module_type_store.open_function_context('test_multiple_modes', 104, 0, False)
    
    # Passed parameters checking function
    test_multiple_modes.stypy_localization = localization
    test_multiple_modes.stypy_type_of_self = None
    test_multiple_modes.stypy_type_store = module_type_store
    test_multiple_modes.stypy_function_name = 'test_multiple_modes'
    test_multiple_modes.stypy_param_names_list = []
    test_multiple_modes.stypy_varargs_param_name = None
    test_multiple_modes.stypy_kwargs_param_name = None
    test_multiple_modes.stypy_call_defaults = defaults
    test_multiple_modes.stypy_call_varargs = varargs
    test_multiple_modes.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_multiple_modes', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_multiple_modes', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_multiple_modes(...)' code ##################

    
    # Assigning a Call to a Name (line 107):
    
    # Call to array(...): (line 107)
    # Processing the call arguments (line 107)
    
    # Obtaining an instance of the builtin type 'list' (line 107)
    list_127991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 107)
    # Adding element type (line 107)
    
    # Obtaining an instance of the builtin type 'list' (line 107)
    list_127992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 107)
    # Adding element type (line 107)
    float_127993 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 21), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 20), list_127992, float_127993)
    # Adding element type (line 107)
    float_127994 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 25), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 20), list_127992, float_127994)
    # Adding element type (line 107)
    float_127995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 29), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 20), list_127992, float_127995)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 19), list_127991, list_127992)
    # Adding element type (line 107)
    
    # Obtaining an instance of the builtin type 'list' (line 108)
    list_127996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 108)
    # Adding element type (line 108)
    float_127997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 21), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 20), list_127996, float_127997)
    # Adding element type (line 108)
    float_127998 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 25), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 20), list_127996, float_127998)
    # Adding element type (line 108)
    float_127999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 29), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 20), list_127996, float_127999)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 19), list_127991, list_127996)
    # Adding element type (line 107)
    
    # Obtaining an instance of the builtin type 'list' (line 109)
    list_128000 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 109)
    # Adding element type (line 109)
    float_128001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 21), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 20), list_128000, float_128001)
    # Adding element type (line 109)
    float_128002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 25), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 20), list_128000, float_128002)
    # Adding element type (line 109)
    float_128003 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 29), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 20), list_128000, float_128003)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 19), list_127991, list_128000)
    
    # Processing the call keyword arguments (line 107)
    kwargs_128004 = {}
    # Getting the type of 'np' (line 107)
    np_127989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 10), 'np', False)
    # Obtaining the member 'array' of a type (line 107)
    array_127990 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 10), np_127989, 'array')
    # Calling array(args, kwargs) (line 107)
    array_call_result_128005 = invoke(stypy.reporting.localization.Localization(__file__, 107, 10), array_127990, *[list_127991], **kwargs_128004)
    
    # Assigning a type to the variable 'arr' (line 107)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 4), 'arr', array_call_result_128005)
    
    # Assigning a Str to a Name (line 111):
    str_128006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 12), 'str', 'reflect')
    # Assigning a type to the variable 'mode1' (line 111)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 4), 'mode1', str_128006)
    
    # Assigning a List to a Name (line 112):
    
    # Obtaining an instance of the builtin type 'list' (line 112)
    list_128007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 12), 'list')
    # Adding type elements to the builtin type 'list' instance (line 112)
    # Adding element type (line 112)
    str_128008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 13), 'str', 'reflect')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 12), list_128007, str_128008)
    # Adding element type (line 112)
    str_128009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 24), 'str', 'reflect')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 12), list_128007, str_128009)
    
    # Assigning a type to the variable 'mode2' (line 112)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'mode2', list_128007)
    
    # Call to assert_equal(...): (line 114)
    # Processing the call arguments (line 114)
    
    # Call to gaussian_filter(...): (line 114)
    # Processing the call arguments (line 114)
    # Getting the type of 'arr' (line 114)
    arr_128013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 38), 'arr', False)
    int_128014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 43), 'int')
    # Processing the call keyword arguments (line 114)
    # Getting the type of 'mode1' (line 114)
    mode1_128015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 51), 'mode1', False)
    keyword_128016 = mode1_128015
    kwargs_128017 = {'mode': keyword_128016}
    # Getting the type of 'sndi' (line 114)
    sndi_128011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 17), 'sndi', False)
    # Obtaining the member 'gaussian_filter' of a type (line 114)
    gaussian_filter_128012 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 17), sndi_128011, 'gaussian_filter')
    # Calling gaussian_filter(args, kwargs) (line 114)
    gaussian_filter_call_result_128018 = invoke(stypy.reporting.localization.Localization(__file__, 114, 17), gaussian_filter_128012, *[arr_128013, int_128014], **kwargs_128017)
    
    
    # Call to gaussian_filter(...): (line 115)
    # Processing the call arguments (line 115)
    # Getting the type of 'arr' (line 115)
    arr_128021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 38), 'arr', False)
    int_128022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 43), 'int')
    # Processing the call keyword arguments (line 115)
    # Getting the type of 'mode2' (line 115)
    mode2_128023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 51), 'mode2', False)
    keyword_128024 = mode2_128023
    kwargs_128025 = {'mode': keyword_128024}
    # Getting the type of 'sndi' (line 115)
    sndi_128019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 17), 'sndi', False)
    # Obtaining the member 'gaussian_filter' of a type (line 115)
    gaussian_filter_128020 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 17), sndi_128019, 'gaussian_filter')
    # Calling gaussian_filter(args, kwargs) (line 115)
    gaussian_filter_call_result_128026 = invoke(stypy.reporting.localization.Localization(__file__, 115, 17), gaussian_filter_128020, *[arr_128021, int_128022], **kwargs_128025)
    
    # Processing the call keyword arguments (line 114)
    kwargs_128027 = {}
    # Getting the type of 'assert_equal' (line 114)
    assert_equal_128010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 114)
    assert_equal_call_result_128028 = invoke(stypy.reporting.localization.Localization(__file__, 114, 4), assert_equal_128010, *[gaussian_filter_call_result_128018, gaussian_filter_call_result_128026], **kwargs_128027)
    
    
    # Call to assert_equal(...): (line 116)
    # Processing the call arguments (line 116)
    
    # Call to prewitt(...): (line 116)
    # Processing the call arguments (line 116)
    # Getting the type of 'arr' (line 116)
    arr_128032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 30), 'arr', False)
    # Processing the call keyword arguments (line 116)
    # Getting the type of 'mode1' (line 116)
    mode1_128033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 40), 'mode1', False)
    keyword_128034 = mode1_128033
    kwargs_128035 = {'mode': keyword_128034}
    # Getting the type of 'sndi' (line 116)
    sndi_128030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 17), 'sndi', False)
    # Obtaining the member 'prewitt' of a type (line 116)
    prewitt_128031 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 17), sndi_128030, 'prewitt')
    # Calling prewitt(args, kwargs) (line 116)
    prewitt_call_result_128036 = invoke(stypy.reporting.localization.Localization(__file__, 116, 17), prewitt_128031, *[arr_128032], **kwargs_128035)
    
    
    # Call to prewitt(...): (line 117)
    # Processing the call arguments (line 117)
    # Getting the type of 'arr' (line 117)
    arr_128039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 30), 'arr', False)
    # Processing the call keyword arguments (line 117)
    # Getting the type of 'mode2' (line 117)
    mode2_128040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 40), 'mode2', False)
    keyword_128041 = mode2_128040
    kwargs_128042 = {'mode': keyword_128041}
    # Getting the type of 'sndi' (line 117)
    sndi_128037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 17), 'sndi', False)
    # Obtaining the member 'prewitt' of a type (line 117)
    prewitt_128038 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 17), sndi_128037, 'prewitt')
    # Calling prewitt(args, kwargs) (line 117)
    prewitt_call_result_128043 = invoke(stypy.reporting.localization.Localization(__file__, 117, 17), prewitt_128038, *[arr_128039], **kwargs_128042)
    
    # Processing the call keyword arguments (line 116)
    kwargs_128044 = {}
    # Getting the type of 'assert_equal' (line 116)
    assert_equal_128029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 116)
    assert_equal_call_result_128045 = invoke(stypy.reporting.localization.Localization(__file__, 116, 4), assert_equal_128029, *[prewitt_call_result_128036, prewitt_call_result_128043], **kwargs_128044)
    
    
    # Call to assert_equal(...): (line 118)
    # Processing the call arguments (line 118)
    
    # Call to sobel(...): (line 118)
    # Processing the call arguments (line 118)
    # Getting the type of 'arr' (line 118)
    arr_128049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 28), 'arr', False)
    # Processing the call keyword arguments (line 118)
    # Getting the type of 'mode1' (line 118)
    mode1_128050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 38), 'mode1', False)
    keyword_128051 = mode1_128050
    kwargs_128052 = {'mode': keyword_128051}
    # Getting the type of 'sndi' (line 118)
    sndi_128047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 17), 'sndi', False)
    # Obtaining the member 'sobel' of a type (line 118)
    sobel_128048 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 17), sndi_128047, 'sobel')
    # Calling sobel(args, kwargs) (line 118)
    sobel_call_result_128053 = invoke(stypy.reporting.localization.Localization(__file__, 118, 17), sobel_128048, *[arr_128049], **kwargs_128052)
    
    
    # Call to sobel(...): (line 119)
    # Processing the call arguments (line 119)
    # Getting the type of 'arr' (line 119)
    arr_128056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 28), 'arr', False)
    # Processing the call keyword arguments (line 119)
    # Getting the type of 'mode2' (line 119)
    mode2_128057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 38), 'mode2', False)
    keyword_128058 = mode2_128057
    kwargs_128059 = {'mode': keyword_128058}
    # Getting the type of 'sndi' (line 119)
    sndi_128054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 17), 'sndi', False)
    # Obtaining the member 'sobel' of a type (line 119)
    sobel_128055 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 17), sndi_128054, 'sobel')
    # Calling sobel(args, kwargs) (line 119)
    sobel_call_result_128060 = invoke(stypy.reporting.localization.Localization(__file__, 119, 17), sobel_128055, *[arr_128056], **kwargs_128059)
    
    # Processing the call keyword arguments (line 118)
    kwargs_128061 = {}
    # Getting the type of 'assert_equal' (line 118)
    assert_equal_128046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 118)
    assert_equal_call_result_128062 = invoke(stypy.reporting.localization.Localization(__file__, 118, 4), assert_equal_128046, *[sobel_call_result_128053, sobel_call_result_128060], **kwargs_128061)
    
    
    # Call to assert_equal(...): (line 120)
    # Processing the call arguments (line 120)
    
    # Call to laplace(...): (line 120)
    # Processing the call arguments (line 120)
    # Getting the type of 'arr' (line 120)
    arr_128066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 30), 'arr', False)
    # Processing the call keyword arguments (line 120)
    # Getting the type of 'mode1' (line 120)
    mode1_128067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 40), 'mode1', False)
    keyword_128068 = mode1_128067
    kwargs_128069 = {'mode': keyword_128068}
    # Getting the type of 'sndi' (line 120)
    sndi_128064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 17), 'sndi', False)
    # Obtaining the member 'laplace' of a type (line 120)
    laplace_128065 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 17), sndi_128064, 'laplace')
    # Calling laplace(args, kwargs) (line 120)
    laplace_call_result_128070 = invoke(stypy.reporting.localization.Localization(__file__, 120, 17), laplace_128065, *[arr_128066], **kwargs_128069)
    
    
    # Call to laplace(...): (line 121)
    # Processing the call arguments (line 121)
    # Getting the type of 'arr' (line 121)
    arr_128073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 30), 'arr', False)
    # Processing the call keyword arguments (line 121)
    # Getting the type of 'mode2' (line 121)
    mode2_128074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 40), 'mode2', False)
    keyword_128075 = mode2_128074
    kwargs_128076 = {'mode': keyword_128075}
    # Getting the type of 'sndi' (line 121)
    sndi_128071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 17), 'sndi', False)
    # Obtaining the member 'laplace' of a type (line 121)
    laplace_128072 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 17), sndi_128071, 'laplace')
    # Calling laplace(args, kwargs) (line 121)
    laplace_call_result_128077 = invoke(stypy.reporting.localization.Localization(__file__, 121, 17), laplace_128072, *[arr_128073], **kwargs_128076)
    
    # Processing the call keyword arguments (line 120)
    kwargs_128078 = {}
    # Getting the type of 'assert_equal' (line 120)
    assert_equal_128063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 120)
    assert_equal_call_result_128079 = invoke(stypy.reporting.localization.Localization(__file__, 120, 4), assert_equal_128063, *[laplace_call_result_128070, laplace_call_result_128077], **kwargs_128078)
    
    
    # Call to assert_equal(...): (line 122)
    # Processing the call arguments (line 122)
    
    # Call to gaussian_laplace(...): (line 122)
    # Processing the call arguments (line 122)
    # Getting the type of 'arr' (line 122)
    arr_128083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 39), 'arr', False)
    int_128084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 44), 'int')
    # Processing the call keyword arguments (line 122)
    # Getting the type of 'mode1' (line 122)
    mode1_128085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 52), 'mode1', False)
    keyword_128086 = mode1_128085
    kwargs_128087 = {'mode': keyword_128086}
    # Getting the type of 'sndi' (line 122)
    sndi_128081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 17), 'sndi', False)
    # Obtaining the member 'gaussian_laplace' of a type (line 122)
    gaussian_laplace_128082 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 17), sndi_128081, 'gaussian_laplace')
    # Calling gaussian_laplace(args, kwargs) (line 122)
    gaussian_laplace_call_result_128088 = invoke(stypy.reporting.localization.Localization(__file__, 122, 17), gaussian_laplace_128082, *[arr_128083, int_128084], **kwargs_128087)
    
    
    # Call to gaussian_laplace(...): (line 123)
    # Processing the call arguments (line 123)
    # Getting the type of 'arr' (line 123)
    arr_128091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 39), 'arr', False)
    int_128092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 44), 'int')
    # Processing the call keyword arguments (line 123)
    # Getting the type of 'mode2' (line 123)
    mode2_128093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 52), 'mode2', False)
    keyword_128094 = mode2_128093
    kwargs_128095 = {'mode': keyword_128094}
    # Getting the type of 'sndi' (line 123)
    sndi_128089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 17), 'sndi', False)
    # Obtaining the member 'gaussian_laplace' of a type (line 123)
    gaussian_laplace_128090 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 17), sndi_128089, 'gaussian_laplace')
    # Calling gaussian_laplace(args, kwargs) (line 123)
    gaussian_laplace_call_result_128096 = invoke(stypy.reporting.localization.Localization(__file__, 123, 17), gaussian_laplace_128090, *[arr_128091, int_128092], **kwargs_128095)
    
    # Processing the call keyword arguments (line 122)
    kwargs_128097 = {}
    # Getting the type of 'assert_equal' (line 122)
    assert_equal_128080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 122)
    assert_equal_call_result_128098 = invoke(stypy.reporting.localization.Localization(__file__, 122, 4), assert_equal_128080, *[gaussian_laplace_call_result_128088, gaussian_laplace_call_result_128096], **kwargs_128097)
    
    
    # Call to assert_equal(...): (line 124)
    # Processing the call arguments (line 124)
    
    # Call to maximum_filter(...): (line 124)
    # Processing the call arguments (line 124)
    # Getting the type of 'arr' (line 124)
    arr_128102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 37), 'arr', False)
    # Processing the call keyword arguments (line 124)
    int_128103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 47), 'int')
    keyword_128104 = int_128103
    # Getting the type of 'mode1' (line 124)
    mode1_128105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 55), 'mode1', False)
    keyword_128106 = mode1_128105
    kwargs_128107 = {'mode': keyword_128106, 'size': keyword_128104}
    # Getting the type of 'sndi' (line 124)
    sndi_128100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 17), 'sndi', False)
    # Obtaining the member 'maximum_filter' of a type (line 124)
    maximum_filter_128101 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 17), sndi_128100, 'maximum_filter')
    # Calling maximum_filter(args, kwargs) (line 124)
    maximum_filter_call_result_128108 = invoke(stypy.reporting.localization.Localization(__file__, 124, 17), maximum_filter_128101, *[arr_128102], **kwargs_128107)
    
    
    # Call to maximum_filter(...): (line 125)
    # Processing the call arguments (line 125)
    # Getting the type of 'arr' (line 125)
    arr_128111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 37), 'arr', False)
    # Processing the call keyword arguments (line 125)
    int_128112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 47), 'int')
    keyword_128113 = int_128112
    # Getting the type of 'mode2' (line 125)
    mode2_128114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 55), 'mode2', False)
    keyword_128115 = mode2_128114
    kwargs_128116 = {'mode': keyword_128115, 'size': keyword_128113}
    # Getting the type of 'sndi' (line 125)
    sndi_128109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 17), 'sndi', False)
    # Obtaining the member 'maximum_filter' of a type (line 125)
    maximum_filter_128110 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 17), sndi_128109, 'maximum_filter')
    # Calling maximum_filter(args, kwargs) (line 125)
    maximum_filter_call_result_128117 = invoke(stypy.reporting.localization.Localization(__file__, 125, 17), maximum_filter_128110, *[arr_128111], **kwargs_128116)
    
    # Processing the call keyword arguments (line 124)
    kwargs_128118 = {}
    # Getting the type of 'assert_equal' (line 124)
    assert_equal_128099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 124)
    assert_equal_call_result_128119 = invoke(stypy.reporting.localization.Localization(__file__, 124, 4), assert_equal_128099, *[maximum_filter_call_result_128108, maximum_filter_call_result_128117], **kwargs_128118)
    
    
    # Call to assert_equal(...): (line 126)
    # Processing the call arguments (line 126)
    
    # Call to minimum_filter(...): (line 126)
    # Processing the call arguments (line 126)
    # Getting the type of 'arr' (line 126)
    arr_128123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 37), 'arr', False)
    # Processing the call keyword arguments (line 126)
    int_128124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 47), 'int')
    keyword_128125 = int_128124
    # Getting the type of 'mode1' (line 126)
    mode1_128126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 55), 'mode1', False)
    keyword_128127 = mode1_128126
    kwargs_128128 = {'mode': keyword_128127, 'size': keyword_128125}
    # Getting the type of 'sndi' (line 126)
    sndi_128121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 17), 'sndi', False)
    # Obtaining the member 'minimum_filter' of a type (line 126)
    minimum_filter_128122 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 17), sndi_128121, 'minimum_filter')
    # Calling minimum_filter(args, kwargs) (line 126)
    minimum_filter_call_result_128129 = invoke(stypy.reporting.localization.Localization(__file__, 126, 17), minimum_filter_128122, *[arr_128123], **kwargs_128128)
    
    
    # Call to minimum_filter(...): (line 127)
    # Processing the call arguments (line 127)
    # Getting the type of 'arr' (line 127)
    arr_128132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 37), 'arr', False)
    # Processing the call keyword arguments (line 127)
    int_128133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 47), 'int')
    keyword_128134 = int_128133
    # Getting the type of 'mode2' (line 127)
    mode2_128135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 55), 'mode2', False)
    keyword_128136 = mode2_128135
    kwargs_128137 = {'mode': keyword_128136, 'size': keyword_128134}
    # Getting the type of 'sndi' (line 127)
    sndi_128130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 17), 'sndi', False)
    # Obtaining the member 'minimum_filter' of a type (line 127)
    minimum_filter_128131 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 17), sndi_128130, 'minimum_filter')
    # Calling minimum_filter(args, kwargs) (line 127)
    minimum_filter_call_result_128138 = invoke(stypy.reporting.localization.Localization(__file__, 127, 17), minimum_filter_128131, *[arr_128132], **kwargs_128137)
    
    # Processing the call keyword arguments (line 126)
    kwargs_128139 = {}
    # Getting the type of 'assert_equal' (line 126)
    assert_equal_128120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 126)
    assert_equal_call_result_128140 = invoke(stypy.reporting.localization.Localization(__file__, 126, 4), assert_equal_128120, *[minimum_filter_call_result_128129, minimum_filter_call_result_128138], **kwargs_128139)
    
    
    # Call to assert_equal(...): (line 128)
    # Processing the call arguments (line 128)
    
    # Call to gaussian_gradient_magnitude(...): (line 128)
    # Processing the call arguments (line 128)
    # Getting the type of 'arr' (line 128)
    arr_128144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 50), 'arr', False)
    int_128145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 55), 'int')
    # Processing the call keyword arguments (line 128)
    # Getting the type of 'mode1' (line 128)
    mode1_128146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 63), 'mode1', False)
    keyword_128147 = mode1_128146
    kwargs_128148 = {'mode': keyword_128147}
    # Getting the type of 'sndi' (line 128)
    sndi_128142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 17), 'sndi', False)
    # Obtaining the member 'gaussian_gradient_magnitude' of a type (line 128)
    gaussian_gradient_magnitude_128143 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 17), sndi_128142, 'gaussian_gradient_magnitude')
    # Calling gaussian_gradient_magnitude(args, kwargs) (line 128)
    gaussian_gradient_magnitude_call_result_128149 = invoke(stypy.reporting.localization.Localization(__file__, 128, 17), gaussian_gradient_magnitude_128143, *[arr_128144, int_128145], **kwargs_128148)
    
    
    # Call to gaussian_gradient_magnitude(...): (line 129)
    # Processing the call arguments (line 129)
    # Getting the type of 'arr' (line 129)
    arr_128152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 50), 'arr', False)
    int_128153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 55), 'int')
    # Processing the call keyword arguments (line 129)
    # Getting the type of 'mode2' (line 129)
    mode2_128154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 63), 'mode2', False)
    keyword_128155 = mode2_128154
    kwargs_128156 = {'mode': keyword_128155}
    # Getting the type of 'sndi' (line 129)
    sndi_128150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 17), 'sndi', False)
    # Obtaining the member 'gaussian_gradient_magnitude' of a type (line 129)
    gaussian_gradient_magnitude_128151 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 17), sndi_128150, 'gaussian_gradient_magnitude')
    # Calling gaussian_gradient_magnitude(args, kwargs) (line 129)
    gaussian_gradient_magnitude_call_result_128157 = invoke(stypy.reporting.localization.Localization(__file__, 129, 17), gaussian_gradient_magnitude_128151, *[arr_128152, int_128153], **kwargs_128156)
    
    # Processing the call keyword arguments (line 128)
    kwargs_128158 = {}
    # Getting the type of 'assert_equal' (line 128)
    assert_equal_128141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 128)
    assert_equal_call_result_128159 = invoke(stypy.reporting.localization.Localization(__file__, 128, 4), assert_equal_128141, *[gaussian_gradient_magnitude_call_result_128149, gaussian_gradient_magnitude_call_result_128157], **kwargs_128158)
    
    
    # Call to assert_equal(...): (line 130)
    # Processing the call arguments (line 130)
    
    # Call to uniform_filter(...): (line 130)
    # Processing the call arguments (line 130)
    # Getting the type of 'arr' (line 130)
    arr_128163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 37), 'arr', False)
    int_128164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 42), 'int')
    # Processing the call keyword arguments (line 130)
    # Getting the type of 'mode1' (line 130)
    mode1_128165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 50), 'mode1', False)
    keyword_128166 = mode1_128165
    kwargs_128167 = {'mode': keyword_128166}
    # Getting the type of 'sndi' (line 130)
    sndi_128161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 17), 'sndi', False)
    # Obtaining the member 'uniform_filter' of a type (line 130)
    uniform_filter_128162 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 17), sndi_128161, 'uniform_filter')
    # Calling uniform_filter(args, kwargs) (line 130)
    uniform_filter_call_result_128168 = invoke(stypy.reporting.localization.Localization(__file__, 130, 17), uniform_filter_128162, *[arr_128163, int_128164], **kwargs_128167)
    
    
    # Call to uniform_filter(...): (line 131)
    # Processing the call arguments (line 131)
    # Getting the type of 'arr' (line 131)
    arr_128171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 37), 'arr', False)
    int_128172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 42), 'int')
    # Processing the call keyword arguments (line 131)
    # Getting the type of 'mode2' (line 131)
    mode2_128173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 50), 'mode2', False)
    keyword_128174 = mode2_128173
    kwargs_128175 = {'mode': keyword_128174}
    # Getting the type of 'sndi' (line 131)
    sndi_128169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 17), 'sndi', False)
    # Obtaining the member 'uniform_filter' of a type (line 131)
    uniform_filter_128170 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 17), sndi_128169, 'uniform_filter')
    # Calling uniform_filter(args, kwargs) (line 131)
    uniform_filter_call_result_128176 = invoke(stypy.reporting.localization.Localization(__file__, 131, 17), uniform_filter_128170, *[arr_128171, int_128172], **kwargs_128175)
    
    # Processing the call keyword arguments (line 130)
    kwargs_128177 = {}
    # Getting the type of 'assert_equal' (line 130)
    assert_equal_128160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 130)
    assert_equal_call_result_128178 = invoke(stypy.reporting.localization.Localization(__file__, 130, 4), assert_equal_128160, *[uniform_filter_call_result_128168, uniform_filter_call_result_128176], **kwargs_128177)
    
    
    # ################# End of 'test_multiple_modes(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_multiple_modes' in the type store
    # Getting the type of 'stypy_return_type' (line 104)
    stypy_return_type_128179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_128179)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_multiple_modes'
    return stypy_return_type_128179

# Assigning a type to the variable 'test_multiple_modes' (line 104)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 0), 'test_multiple_modes', test_multiple_modes)

@norecursion
def test_multiple_modes_sequentially(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_multiple_modes_sequentially'
    module_type_store = module_type_store.open_function_context('test_multiple_modes_sequentially', 134, 0, False)
    
    # Passed parameters checking function
    test_multiple_modes_sequentially.stypy_localization = localization
    test_multiple_modes_sequentially.stypy_type_of_self = None
    test_multiple_modes_sequentially.stypy_type_store = module_type_store
    test_multiple_modes_sequentially.stypy_function_name = 'test_multiple_modes_sequentially'
    test_multiple_modes_sequentially.stypy_param_names_list = []
    test_multiple_modes_sequentially.stypy_varargs_param_name = None
    test_multiple_modes_sequentially.stypy_kwargs_param_name = None
    test_multiple_modes_sequentially.stypy_call_defaults = defaults
    test_multiple_modes_sequentially.stypy_call_varargs = varargs
    test_multiple_modes_sequentially.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_multiple_modes_sequentially', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_multiple_modes_sequentially', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_multiple_modes_sequentially(...)' code ##################

    
    # Assigning a Call to a Name (line 138):
    
    # Call to array(...): (line 138)
    # Processing the call arguments (line 138)
    
    # Obtaining an instance of the builtin type 'list' (line 138)
    list_128182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 138)
    # Adding element type (line 138)
    
    # Obtaining an instance of the builtin type 'list' (line 138)
    list_128183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 138)
    # Adding element type (line 138)
    float_128184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 21), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 20), list_128183, float_128184)
    # Adding element type (line 138)
    float_128185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 25), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 20), list_128183, float_128185)
    # Adding element type (line 138)
    float_128186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 29), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 20), list_128183, float_128186)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 19), list_128182, list_128183)
    # Adding element type (line 138)
    
    # Obtaining an instance of the builtin type 'list' (line 139)
    list_128187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 139)
    # Adding element type (line 139)
    float_128188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 21), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 139, 20), list_128187, float_128188)
    # Adding element type (line 139)
    float_128189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 25), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 139, 20), list_128187, float_128189)
    # Adding element type (line 139)
    float_128190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 29), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 139, 20), list_128187, float_128190)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 19), list_128182, list_128187)
    # Adding element type (line 138)
    
    # Obtaining an instance of the builtin type 'list' (line 140)
    list_128191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 140)
    # Adding element type (line 140)
    float_128192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 21), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 20), list_128191, float_128192)
    # Adding element type (line 140)
    float_128193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 25), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 20), list_128191, float_128193)
    # Adding element type (line 140)
    float_128194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 29), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 20), list_128191, float_128194)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 19), list_128182, list_128191)
    
    # Processing the call keyword arguments (line 138)
    kwargs_128195 = {}
    # Getting the type of 'np' (line 138)
    np_128180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 10), 'np', False)
    # Obtaining the member 'array' of a type (line 138)
    array_128181 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 10), np_128180, 'array')
    # Calling array(args, kwargs) (line 138)
    array_call_result_128196 = invoke(stypy.reporting.localization.Localization(__file__, 138, 10), array_128181, *[list_128182], **kwargs_128195)
    
    # Assigning a type to the variable 'arr' (line 138)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 4), 'arr', array_call_result_128196)
    
    # Assigning a List to a Name (line 142):
    
    # Obtaining an instance of the builtin type 'list' (line 142)
    list_128197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 12), 'list')
    # Adding type elements to the builtin type 'list' instance (line 142)
    # Adding element type (line 142)
    str_128198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 13), 'str', 'reflect')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 142, 12), list_128197, str_128198)
    # Adding element type (line 142)
    str_128199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 24), 'str', 'wrap')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 142, 12), list_128197, str_128199)
    
    # Assigning a type to the variable 'modes' (line 142)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 4), 'modes', list_128197)
    
    # Assigning a Call to a Name (line 144):
    
    # Call to gaussian_filter1d(...): (line 144)
    # Processing the call arguments (line 144)
    # Getting the type of 'arr' (line 144)
    arr_128202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 38), 'arr', False)
    int_128203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 43), 'int')
    # Processing the call keyword arguments (line 144)
    int_128204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 51), 'int')
    keyword_128205 = int_128204
    
    # Obtaining the type of the subscript
    int_128206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 65), 'int')
    # Getting the type of 'modes' (line 144)
    modes_128207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 59), 'modes', False)
    # Obtaining the member '__getitem__' of a type (line 144)
    getitem___128208 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 59), modes_128207, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 144)
    subscript_call_result_128209 = invoke(stypy.reporting.localization.Localization(__file__, 144, 59), getitem___128208, int_128206)
    
    keyword_128210 = subscript_call_result_128209
    kwargs_128211 = {'mode': keyword_128210, 'axis': keyword_128205}
    # Getting the type of 'sndi' (line 144)
    sndi_128200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 15), 'sndi', False)
    # Obtaining the member 'gaussian_filter1d' of a type (line 144)
    gaussian_filter1d_128201 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 15), sndi_128200, 'gaussian_filter1d')
    # Calling gaussian_filter1d(args, kwargs) (line 144)
    gaussian_filter1d_call_result_128212 = invoke(stypy.reporting.localization.Localization(__file__, 144, 15), gaussian_filter1d_128201, *[arr_128202, int_128203], **kwargs_128211)
    
    # Assigning a type to the variable 'expected' (line 144)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 4), 'expected', gaussian_filter1d_call_result_128212)
    
    # Assigning a Call to a Name (line 145):
    
    # Call to gaussian_filter1d(...): (line 145)
    # Processing the call arguments (line 145)
    # Getting the type of 'expected' (line 145)
    expected_128215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 38), 'expected', False)
    int_128216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 48), 'int')
    # Processing the call keyword arguments (line 145)
    int_128217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 56), 'int')
    keyword_128218 = int_128217
    
    # Obtaining the type of the subscript
    int_128219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 70), 'int')
    # Getting the type of 'modes' (line 145)
    modes_128220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 64), 'modes', False)
    # Obtaining the member '__getitem__' of a type (line 145)
    getitem___128221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 64), modes_128220, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 145)
    subscript_call_result_128222 = invoke(stypy.reporting.localization.Localization(__file__, 145, 64), getitem___128221, int_128219)
    
    keyword_128223 = subscript_call_result_128222
    kwargs_128224 = {'mode': keyword_128223, 'axis': keyword_128218}
    # Getting the type of 'sndi' (line 145)
    sndi_128213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 15), 'sndi', False)
    # Obtaining the member 'gaussian_filter1d' of a type (line 145)
    gaussian_filter1d_128214 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 15), sndi_128213, 'gaussian_filter1d')
    # Calling gaussian_filter1d(args, kwargs) (line 145)
    gaussian_filter1d_call_result_128225 = invoke(stypy.reporting.localization.Localization(__file__, 145, 15), gaussian_filter1d_128214, *[expected_128215, int_128216], **kwargs_128224)
    
    # Assigning a type to the variable 'expected' (line 145)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 4), 'expected', gaussian_filter1d_call_result_128225)
    
    # Call to assert_equal(...): (line 146)
    # Processing the call arguments (line 146)
    # Getting the type of 'expected' (line 146)
    expected_128227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 17), 'expected', False)
    
    # Call to gaussian_filter(...): (line 147)
    # Processing the call arguments (line 147)
    # Getting the type of 'arr' (line 147)
    arr_128230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 38), 'arr', False)
    int_128231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 43), 'int')
    # Processing the call keyword arguments (line 147)
    # Getting the type of 'modes' (line 147)
    modes_128232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 51), 'modes', False)
    keyword_128233 = modes_128232
    kwargs_128234 = {'mode': keyword_128233}
    # Getting the type of 'sndi' (line 147)
    sndi_128228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 17), 'sndi', False)
    # Obtaining the member 'gaussian_filter' of a type (line 147)
    gaussian_filter_128229 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 17), sndi_128228, 'gaussian_filter')
    # Calling gaussian_filter(args, kwargs) (line 147)
    gaussian_filter_call_result_128235 = invoke(stypy.reporting.localization.Localization(__file__, 147, 17), gaussian_filter_128229, *[arr_128230, int_128231], **kwargs_128234)
    
    # Processing the call keyword arguments (line 146)
    kwargs_128236 = {}
    # Getting the type of 'assert_equal' (line 146)
    assert_equal_128226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 146)
    assert_equal_call_result_128237 = invoke(stypy.reporting.localization.Localization(__file__, 146, 4), assert_equal_128226, *[expected_128227, gaussian_filter_call_result_128235], **kwargs_128236)
    
    
    # Assigning a Call to a Name (line 149):
    
    # Call to uniform_filter1d(...): (line 149)
    # Processing the call arguments (line 149)
    # Getting the type of 'arr' (line 149)
    arr_128240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 37), 'arr', False)
    int_128241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 42), 'int')
    # Processing the call keyword arguments (line 149)
    int_128242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 50), 'int')
    keyword_128243 = int_128242
    
    # Obtaining the type of the subscript
    int_128244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 64), 'int')
    # Getting the type of 'modes' (line 149)
    modes_128245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 58), 'modes', False)
    # Obtaining the member '__getitem__' of a type (line 149)
    getitem___128246 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 58), modes_128245, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 149)
    subscript_call_result_128247 = invoke(stypy.reporting.localization.Localization(__file__, 149, 58), getitem___128246, int_128244)
    
    keyword_128248 = subscript_call_result_128247
    kwargs_128249 = {'mode': keyword_128248, 'axis': keyword_128243}
    # Getting the type of 'sndi' (line 149)
    sndi_128238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 15), 'sndi', False)
    # Obtaining the member 'uniform_filter1d' of a type (line 149)
    uniform_filter1d_128239 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 15), sndi_128238, 'uniform_filter1d')
    # Calling uniform_filter1d(args, kwargs) (line 149)
    uniform_filter1d_call_result_128250 = invoke(stypy.reporting.localization.Localization(__file__, 149, 15), uniform_filter1d_128239, *[arr_128240, int_128241], **kwargs_128249)
    
    # Assigning a type to the variable 'expected' (line 149)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 4), 'expected', uniform_filter1d_call_result_128250)
    
    # Assigning a Call to a Name (line 150):
    
    # Call to uniform_filter1d(...): (line 150)
    # Processing the call arguments (line 150)
    # Getting the type of 'expected' (line 150)
    expected_128253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 37), 'expected', False)
    int_128254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 47), 'int')
    # Processing the call keyword arguments (line 150)
    int_128255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 55), 'int')
    keyword_128256 = int_128255
    
    # Obtaining the type of the subscript
    int_128257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 69), 'int')
    # Getting the type of 'modes' (line 150)
    modes_128258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 63), 'modes', False)
    # Obtaining the member '__getitem__' of a type (line 150)
    getitem___128259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 63), modes_128258, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 150)
    subscript_call_result_128260 = invoke(stypy.reporting.localization.Localization(__file__, 150, 63), getitem___128259, int_128257)
    
    keyword_128261 = subscript_call_result_128260
    kwargs_128262 = {'mode': keyword_128261, 'axis': keyword_128256}
    # Getting the type of 'sndi' (line 150)
    sndi_128251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 15), 'sndi', False)
    # Obtaining the member 'uniform_filter1d' of a type (line 150)
    uniform_filter1d_128252 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 15), sndi_128251, 'uniform_filter1d')
    # Calling uniform_filter1d(args, kwargs) (line 150)
    uniform_filter1d_call_result_128263 = invoke(stypy.reporting.localization.Localization(__file__, 150, 15), uniform_filter1d_128252, *[expected_128253, int_128254], **kwargs_128262)
    
    # Assigning a type to the variable 'expected' (line 150)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 4), 'expected', uniform_filter1d_call_result_128263)
    
    # Call to assert_equal(...): (line 151)
    # Processing the call arguments (line 151)
    # Getting the type of 'expected' (line 151)
    expected_128265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 17), 'expected', False)
    
    # Call to uniform_filter(...): (line 152)
    # Processing the call arguments (line 152)
    # Getting the type of 'arr' (line 152)
    arr_128268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 37), 'arr', False)
    int_128269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 42), 'int')
    # Processing the call keyword arguments (line 152)
    # Getting the type of 'modes' (line 152)
    modes_128270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 50), 'modes', False)
    keyword_128271 = modes_128270
    kwargs_128272 = {'mode': keyword_128271}
    # Getting the type of 'sndi' (line 152)
    sndi_128266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 17), 'sndi', False)
    # Obtaining the member 'uniform_filter' of a type (line 152)
    uniform_filter_128267 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 17), sndi_128266, 'uniform_filter')
    # Calling uniform_filter(args, kwargs) (line 152)
    uniform_filter_call_result_128273 = invoke(stypy.reporting.localization.Localization(__file__, 152, 17), uniform_filter_128267, *[arr_128268, int_128269], **kwargs_128272)
    
    # Processing the call keyword arguments (line 151)
    kwargs_128274 = {}
    # Getting the type of 'assert_equal' (line 151)
    assert_equal_128264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 151)
    assert_equal_call_result_128275 = invoke(stypy.reporting.localization.Localization(__file__, 151, 4), assert_equal_128264, *[expected_128265, uniform_filter_call_result_128273], **kwargs_128274)
    
    
    # Assigning a Call to a Name (line 154):
    
    # Call to maximum_filter1d(...): (line 154)
    # Processing the call arguments (line 154)
    # Getting the type of 'arr' (line 154)
    arr_128278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 37), 'arr', False)
    # Processing the call keyword arguments (line 154)
    int_128279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 47), 'int')
    keyword_128280 = int_128279
    int_128281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 55), 'int')
    keyword_128282 = int_128281
    
    # Obtaining the type of the subscript
    int_128283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 69), 'int')
    # Getting the type of 'modes' (line 154)
    modes_128284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 63), 'modes', False)
    # Obtaining the member '__getitem__' of a type (line 154)
    getitem___128285 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 63), modes_128284, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 154)
    subscript_call_result_128286 = invoke(stypy.reporting.localization.Localization(__file__, 154, 63), getitem___128285, int_128283)
    
    keyword_128287 = subscript_call_result_128286
    kwargs_128288 = {'axis': keyword_128282, 'mode': keyword_128287, 'size': keyword_128280}
    # Getting the type of 'sndi' (line 154)
    sndi_128276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 15), 'sndi', False)
    # Obtaining the member 'maximum_filter1d' of a type (line 154)
    maximum_filter1d_128277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 15), sndi_128276, 'maximum_filter1d')
    # Calling maximum_filter1d(args, kwargs) (line 154)
    maximum_filter1d_call_result_128289 = invoke(stypy.reporting.localization.Localization(__file__, 154, 15), maximum_filter1d_128277, *[arr_128278], **kwargs_128288)
    
    # Assigning a type to the variable 'expected' (line 154)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 4), 'expected', maximum_filter1d_call_result_128289)
    
    # Assigning a Call to a Name (line 155):
    
    # Call to maximum_filter1d(...): (line 155)
    # Processing the call arguments (line 155)
    # Getting the type of 'expected' (line 155)
    expected_128292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 37), 'expected', False)
    # Processing the call keyword arguments (line 155)
    int_128293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 52), 'int')
    keyword_128294 = int_128293
    int_128295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 60), 'int')
    keyword_128296 = int_128295
    
    # Obtaining the type of the subscript
    int_128297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 74), 'int')
    # Getting the type of 'modes' (line 155)
    modes_128298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 68), 'modes', False)
    # Obtaining the member '__getitem__' of a type (line 155)
    getitem___128299 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 68), modes_128298, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 155)
    subscript_call_result_128300 = invoke(stypy.reporting.localization.Localization(__file__, 155, 68), getitem___128299, int_128297)
    
    keyword_128301 = subscript_call_result_128300
    kwargs_128302 = {'axis': keyword_128296, 'mode': keyword_128301, 'size': keyword_128294}
    # Getting the type of 'sndi' (line 155)
    sndi_128290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 15), 'sndi', False)
    # Obtaining the member 'maximum_filter1d' of a type (line 155)
    maximum_filter1d_128291 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 15), sndi_128290, 'maximum_filter1d')
    # Calling maximum_filter1d(args, kwargs) (line 155)
    maximum_filter1d_call_result_128303 = invoke(stypy.reporting.localization.Localization(__file__, 155, 15), maximum_filter1d_128291, *[expected_128292], **kwargs_128302)
    
    # Assigning a type to the variable 'expected' (line 155)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 4), 'expected', maximum_filter1d_call_result_128303)
    
    # Call to assert_equal(...): (line 156)
    # Processing the call arguments (line 156)
    # Getting the type of 'expected' (line 156)
    expected_128305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 17), 'expected', False)
    
    # Call to maximum_filter(...): (line 157)
    # Processing the call arguments (line 157)
    # Getting the type of 'arr' (line 157)
    arr_128308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 37), 'arr', False)
    # Processing the call keyword arguments (line 157)
    int_128309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 47), 'int')
    keyword_128310 = int_128309
    # Getting the type of 'modes' (line 157)
    modes_128311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 55), 'modes', False)
    keyword_128312 = modes_128311
    kwargs_128313 = {'mode': keyword_128312, 'size': keyword_128310}
    # Getting the type of 'sndi' (line 157)
    sndi_128306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 17), 'sndi', False)
    # Obtaining the member 'maximum_filter' of a type (line 157)
    maximum_filter_128307 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 17), sndi_128306, 'maximum_filter')
    # Calling maximum_filter(args, kwargs) (line 157)
    maximum_filter_call_result_128314 = invoke(stypy.reporting.localization.Localization(__file__, 157, 17), maximum_filter_128307, *[arr_128308], **kwargs_128313)
    
    # Processing the call keyword arguments (line 156)
    kwargs_128315 = {}
    # Getting the type of 'assert_equal' (line 156)
    assert_equal_128304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 156)
    assert_equal_call_result_128316 = invoke(stypy.reporting.localization.Localization(__file__, 156, 4), assert_equal_128304, *[expected_128305, maximum_filter_call_result_128314], **kwargs_128315)
    
    
    # Assigning a Call to a Name (line 159):
    
    # Call to minimum_filter1d(...): (line 159)
    # Processing the call arguments (line 159)
    # Getting the type of 'arr' (line 159)
    arr_128319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 37), 'arr', False)
    # Processing the call keyword arguments (line 159)
    int_128320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 47), 'int')
    keyword_128321 = int_128320
    int_128322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 55), 'int')
    keyword_128323 = int_128322
    
    # Obtaining the type of the subscript
    int_128324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 69), 'int')
    # Getting the type of 'modes' (line 159)
    modes_128325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 63), 'modes', False)
    # Obtaining the member '__getitem__' of a type (line 159)
    getitem___128326 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 63), modes_128325, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 159)
    subscript_call_result_128327 = invoke(stypy.reporting.localization.Localization(__file__, 159, 63), getitem___128326, int_128324)
    
    keyword_128328 = subscript_call_result_128327
    kwargs_128329 = {'axis': keyword_128323, 'mode': keyword_128328, 'size': keyword_128321}
    # Getting the type of 'sndi' (line 159)
    sndi_128317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 15), 'sndi', False)
    # Obtaining the member 'minimum_filter1d' of a type (line 159)
    minimum_filter1d_128318 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 15), sndi_128317, 'minimum_filter1d')
    # Calling minimum_filter1d(args, kwargs) (line 159)
    minimum_filter1d_call_result_128330 = invoke(stypy.reporting.localization.Localization(__file__, 159, 15), minimum_filter1d_128318, *[arr_128319], **kwargs_128329)
    
    # Assigning a type to the variable 'expected' (line 159)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 4), 'expected', minimum_filter1d_call_result_128330)
    
    # Assigning a Call to a Name (line 160):
    
    # Call to minimum_filter1d(...): (line 160)
    # Processing the call arguments (line 160)
    # Getting the type of 'expected' (line 160)
    expected_128333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 37), 'expected', False)
    # Processing the call keyword arguments (line 160)
    int_128334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 52), 'int')
    keyword_128335 = int_128334
    int_128336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 60), 'int')
    keyword_128337 = int_128336
    
    # Obtaining the type of the subscript
    int_128338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 74), 'int')
    # Getting the type of 'modes' (line 160)
    modes_128339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 68), 'modes', False)
    # Obtaining the member '__getitem__' of a type (line 160)
    getitem___128340 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 68), modes_128339, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 160)
    subscript_call_result_128341 = invoke(stypy.reporting.localization.Localization(__file__, 160, 68), getitem___128340, int_128338)
    
    keyword_128342 = subscript_call_result_128341
    kwargs_128343 = {'axis': keyword_128337, 'mode': keyword_128342, 'size': keyword_128335}
    # Getting the type of 'sndi' (line 160)
    sndi_128331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 15), 'sndi', False)
    # Obtaining the member 'minimum_filter1d' of a type (line 160)
    minimum_filter1d_128332 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 15), sndi_128331, 'minimum_filter1d')
    # Calling minimum_filter1d(args, kwargs) (line 160)
    minimum_filter1d_call_result_128344 = invoke(stypy.reporting.localization.Localization(__file__, 160, 15), minimum_filter1d_128332, *[expected_128333], **kwargs_128343)
    
    # Assigning a type to the variable 'expected' (line 160)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 4), 'expected', minimum_filter1d_call_result_128344)
    
    # Call to assert_equal(...): (line 161)
    # Processing the call arguments (line 161)
    # Getting the type of 'expected' (line 161)
    expected_128346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 17), 'expected', False)
    
    # Call to minimum_filter(...): (line 162)
    # Processing the call arguments (line 162)
    # Getting the type of 'arr' (line 162)
    arr_128349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 37), 'arr', False)
    # Processing the call keyword arguments (line 162)
    int_128350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 47), 'int')
    keyword_128351 = int_128350
    # Getting the type of 'modes' (line 162)
    modes_128352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 55), 'modes', False)
    keyword_128353 = modes_128352
    kwargs_128354 = {'mode': keyword_128353, 'size': keyword_128351}
    # Getting the type of 'sndi' (line 162)
    sndi_128347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 17), 'sndi', False)
    # Obtaining the member 'minimum_filter' of a type (line 162)
    minimum_filter_128348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 17), sndi_128347, 'minimum_filter')
    # Calling minimum_filter(args, kwargs) (line 162)
    minimum_filter_call_result_128355 = invoke(stypy.reporting.localization.Localization(__file__, 162, 17), minimum_filter_128348, *[arr_128349], **kwargs_128354)
    
    # Processing the call keyword arguments (line 161)
    kwargs_128356 = {}
    # Getting the type of 'assert_equal' (line 161)
    assert_equal_128345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 161)
    assert_equal_call_result_128357 = invoke(stypy.reporting.localization.Localization(__file__, 161, 4), assert_equal_128345, *[expected_128346, minimum_filter_call_result_128355], **kwargs_128356)
    
    
    # ################# End of 'test_multiple_modes_sequentially(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_multiple_modes_sequentially' in the type store
    # Getting the type of 'stypy_return_type' (line 134)
    stypy_return_type_128358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_128358)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_multiple_modes_sequentially'
    return stypy_return_type_128358

# Assigning a type to the variable 'test_multiple_modes_sequentially' (line 134)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 0), 'test_multiple_modes_sequentially', test_multiple_modes_sequentially)

@norecursion
def test_multiple_modes_prewitt(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_multiple_modes_prewitt'
    module_type_store = module_type_store.open_function_context('test_multiple_modes_prewitt', 165, 0, False)
    
    # Passed parameters checking function
    test_multiple_modes_prewitt.stypy_localization = localization
    test_multiple_modes_prewitt.stypy_type_of_self = None
    test_multiple_modes_prewitt.stypy_type_store = module_type_store
    test_multiple_modes_prewitt.stypy_function_name = 'test_multiple_modes_prewitt'
    test_multiple_modes_prewitt.stypy_param_names_list = []
    test_multiple_modes_prewitt.stypy_varargs_param_name = None
    test_multiple_modes_prewitt.stypy_kwargs_param_name = None
    test_multiple_modes_prewitt.stypy_call_defaults = defaults
    test_multiple_modes_prewitt.stypy_call_varargs = varargs
    test_multiple_modes_prewitt.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_multiple_modes_prewitt', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_multiple_modes_prewitt', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_multiple_modes_prewitt(...)' code ##################

    
    # Assigning a Call to a Name (line 167):
    
    # Call to array(...): (line 167)
    # Processing the call arguments (line 167)
    
    # Obtaining an instance of the builtin type 'list' (line 167)
    list_128361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 167)
    # Adding element type (line 167)
    
    # Obtaining an instance of the builtin type 'list' (line 167)
    list_128362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 167)
    # Adding element type (line 167)
    float_128363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 21), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 20), list_128362, float_128363)
    # Adding element type (line 167)
    float_128364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 25), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 20), list_128362, float_128364)
    # Adding element type (line 167)
    float_128365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 29), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 20), list_128362, float_128365)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 19), list_128361, list_128362)
    # Adding element type (line 167)
    
    # Obtaining an instance of the builtin type 'list' (line 168)
    list_128366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 168)
    # Adding element type (line 168)
    float_128367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 21), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 20), list_128366, float_128367)
    # Adding element type (line 168)
    float_128368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 25), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 20), list_128366, float_128368)
    # Adding element type (line 168)
    float_128369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 29), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 20), list_128366, float_128369)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 19), list_128361, list_128366)
    # Adding element type (line 167)
    
    # Obtaining an instance of the builtin type 'list' (line 169)
    list_128370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 169)
    # Adding element type (line 169)
    float_128371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 21), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 169, 20), list_128370, float_128371)
    # Adding element type (line 169)
    float_128372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 25), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 169, 20), list_128370, float_128372)
    # Adding element type (line 169)
    float_128373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 29), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 169, 20), list_128370, float_128373)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 19), list_128361, list_128370)
    
    # Processing the call keyword arguments (line 167)
    kwargs_128374 = {}
    # Getting the type of 'np' (line 167)
    np_128359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 10), 'np', False)
    # Obtaining the member 'array' of a type (line 167)
    array_128360 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 10), np_128359, 'array')
    # Calling array(args, kwargs) (line 167)
    array_call_result_128375 = invoke(stypy.reporting.localization.Localization(__file__, 167, 10), array_128360, *[list_128361], **kwargs_128374)
    
    # Assigning a type to the variable 'arr' (line 167)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 4), 'arr', array_call_result_128375)
    
    # Assigning a Call to a Name (line 171):
    
    # Call to array(...): (line 171)
    # Processing the call arguments (line 171)
    
    # Obtaining an instance of the builtin type 'list' (line 171)
    list_128378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 171)
    # Adding element type (line 171)
    
    # Obtaining an instance of the builtin type 'list' (line 171)
    list_128379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 25), 'list')
    # Adding type elements to the builtin type 'list' instance (line 171)
    # Adding element type (line 171)
    float_128380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 26), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 25), list_128379, float_128380)
    # Adding element type (line 171)
    float_128381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 30), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 25), list_128379, float_128381)
    # Adding element type (line 171)
    float_128382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 35), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 25), list_128379, float_128382)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 24), list_128378, list_128379)
    # Adding element type (line 171)
    
    # Obtaining an instance of the builtin type 'list' (line 172)
    list_128383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 25), 'list')
    # Adding type elements to the builtin type 'list' instance (line 172)
    # Adding element type (line 172)
    float_128384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 26), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 172, 25), list_128383, float_128384)
    # Adding element type (line 172)
    float_128385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 30), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 172, 25), list_128383, float_128385)
    # Adding element type (line 172)
    float_128386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 35), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 172, 25), list_128383, float_128386)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 24), list_128378, list_128383)
    # Adding element type (line 171)
    
    # Obtaining an instance of the builtin type 'list' (line 173)
    list_128387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 25), 'list')
    # Adding type elements to the builtin type 'list' instance (line 173)
    # Adding element type (line 173)
    float_128388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 26), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 173, 25), list_128387, float_128388)
    # Adding element type (line 173)
    float_128389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 30), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 173, 25), list_128387, float_128389)
    # Adding element type (line 173)
    float_128390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 35), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 173, 25), list_128387, float_128390)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 24), list_128378, list_128387)
    
    # Processing the call keyword arguments (line 171)
    kwargs_128391 = {}
    # Getting the type of 'np' (line 171)
    np_128376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 15), 'np', False)
    # Obtaining the member 'array' of a type (line 171)
    array_128377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 15), np_128376, 'array')
    # Calling array(args, kwargs) (line 171)
    array_call_result_128392 = invoke(stypy.reporting.localization.Localization(__file__, 171, 15), array_128377, *[list_128378], **kwargs_128391)
    
    # Assigning a type to the variable 'expected' (line 171)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 4), 'expected', array_call_result_128392)
    
    # Assigning a List to a Name (line 175):
    
    # Obtaining an instance of the builtin type 'list' (line 175)
    list_128393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 12), 'list')
    # Adding type elements to the builtin type 'list' instance (line 175)
    # Adding element type (line 175)
    str_128394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 13), 'str', 'reflect')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 175, 12), list_128393, str_128394)
    # Adding element type (line 175)
    str_128395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 24), 'str', 'wrap')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 175, 12), list_128393, str_128395)
    
    # Assigning a type to the variable 'modes' (line 175)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 4), 'modes', list_128393)
    
    # Call to assert_equal(...): (line 177)
    # Processing the call arguments (line 177)
    # Getting the type of 'expected' (line 177)
    expected_128397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 17), 'expected', False)
    
    # Call to prewitt(...): (line 178)
    # Processing the call arguments (line 178)
    # Getting the type of 'arr' (line 178)
    arr_128400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 30), 'arr', False)
    # Processing the call keyword arguments (line 178)
    # Getting the type of 'modes' (line 178)
    modes_128401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 40), 'modes', False)
    keyword_128402 = modes_128401
    kwargs_128403 = {'mode': keyword_128402}
    # Getting the type of 'sndi' (line 178)
    sndi_128398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 17), 'sndi', False)
    # Obtaining the member 'prewitt' of a type (line 178)
    prewitt_128399 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 17), sndi_128398, 'prewitt')
    # Calling prewitt(args, kwargs) (line 178)
    prewitt_call_result_128404 = invoke(stypy.reporting.localization.Localization(__file__, 178, 17), prewitt_128399, *[arr_128400], **kwargs_128403)
    
    # Processing the call keyword arguments (line 177)
    kwargs_128405 = {}
    # Getting the type of 'assert_equal' (line 177)
    assert_equal_128396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 177)
    assert_equal_call_result_128406 = invoke(stypy.reporting.localization.Localization(__file__, 177, 4), assert_equal_128396, *[expected_128397, prewitt_call_result_128404], **kwargs_128405)
    
    
    # ################# End of 'test_multiple_modes_prewitt(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_multiple_modes_prewitt' in the type store
    # Getting the type of 'stypy_return_type' (line 165)
    stypy_return_type_128407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_128407)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_multiple_modes_prewitt'
    return stypy_return_type_128407

# Assigning a type to the variable 'test_multiple_modes_prewitt' (line 165)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 0), 'test_multiple_modes_prewitt', test_multiple_modes_prewitt)

@norecursion
def test_multiple_modes_sobel(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_multiple_modes_sobel'
    module_type_store = module_type_store.open_function_context('test_multiple_modes_sobel', 181, 0, False)
    
    # Passed parameters checking function
    test_multiple_modes_sobel.stypy_localization = localization
    test_multiple_modes_sobel.stypy_type_of_self = None
    test_multiple_modes_sobel.stypy_type_store = module_type_store
    test_multiple_modes_sobel.stypy_function_name = 'test_multiple_modes_sobel'
    test_multiple_modes_sobel.stypy_param_names_list = []
    test_multiple_modes_sobel.stypy_varargs_param_name = None
    test_multiple_modes_sobel.stypy_kwargs_param_name = None
    test_multiple_modes_sobel.stypy_call_defaults = defaults
    test_multiple_modes_sobel.stypy_call_varargs = varargs
    test_multiple_modes_sobel.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_multiple_modes_sobel', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_multiple_modes_sobel', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_multiple_modes_sobel(...)' code ##################

    
    # Assigning a Call to a Name (line 183):
    
    # Call to array(...): (line 183)
    # Processing the call arguments (line 183)
    
    # Obtaining an instance of the builtin type 'list' (line 183)
    list_128410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 183)
    # Adding element type (line 183)
    
    # Obtaining an instance of the builtin type 'list' (line 183)
    list_128411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 183)
    # Adding element type (line 183)
    float_128412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 21), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 20), list_128411, float_128412)
    # Adding element type (line 183)
    float_128413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 25), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 20), list_128411, float_128413)
    # Adding element type (line 183)
    float_128414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 29), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 20), list_128411, float_128414)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 19), list_128410, list_128411)
    # Adding element type (line 183)
    
    # Obtaining an instance of the builtin type 'list' (line 184)
    list_128415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 184)
    # Adding element type (line 184)
    float_128416 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 21), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 184, 20), list_128415, float_128416)
    # Adding element type (line 184)
    float_128417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 25), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 184, 20), list_128415, float_128417)
    # Adding element type (line 184)
    float_128418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 29), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 184, 20), list_128415, float_128418)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 19), list_128410, list_128415)
    # Adding element type (line 183)
    
    # Obtaining an instance of the builtin type 'list' (line 185)
    list_128419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 185)
    # Adding element type (line 185)
    float_128420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 21), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 20), list_128419, float_128420)
    # Adding element type (line 185)
    float_128421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 25), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 20), list_128419, float_128421)
    # Adding element type (line 185)
    float_128422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 29), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 20), list_128419, float_128422)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 19), list_128410, list_128419)
    
    # Processing the call keyword arguments (line 183)
    kwargs_128423 = {}
    # Getting the type of 'np' (line 183)
    np_128408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 10), 'np', False)
    # Obtaining the member 'array' of a type (line 183)
    array_128409 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 10), np_128408, 'array')
    # Calling array(args, kwargs) (line 183)
    array_call_result_128424 = invoke(stypy.reporting.localization.Localization(__file__, 183, 10), array_128409, *[list_128410], **kwargs_128423)
    
    # Assigning a type to the variable 'arr' (line 183)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 4), 'arr', array_call_result_128424)
    
    # Assigning a Call to a Name (line 187):
    
    # Call to array(...): (line 187)
    # Processing the call arguments (line 187)
    
    # Obtaining an instance of the builtin type 'list' (line 187)
    list_128427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 187)
    # Adding element type (line 187)
    
    # Obtaining an instance of the builtin type 'list' (line 187)
    list_128428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 25), 'list')
    # Adding type elements to the builtin type 'list' instance (line 187)
    # Adding element type (line 187)
    float_128429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 26), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 25), list_128428, float_128429)
    # Adding element type (line 187)
    float_128430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 30), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 25), list_128428, float_128430)
    # Adding element type (line 187)
    float_128431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 35), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 25), list_128428, float_128431)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 24), list_128427, list_128428)
    # Adding element type (line 187)
    
    # Obtaining an instance of the builtin type 'list' (line 188)
    list_128432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 25), 'list')
    # Adding type elements to the builtin type 'list' instance (line 188)
    # Adding element type (line 188)
    float_128433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 26), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 188, 25), list_128432, float_128433)
    # Adding element type (line 188)
    float_128434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 30), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 188, 25), list_128432, float_128434)
    # Adding element type (line 188)
    float_128435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 35), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 188, 25), list_128432, float_128435)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 24), list_128427, list_128432)
    # Adding element type (line 187)
    
    # Obtaining an instance of the builtin type 'list' (line 189)
    list_128436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 25), 'list')
    # Adding type elements to the builtin type 'list' instance (line 189)
    # Adding element type (line 189)
    float_128437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 26), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 189, 25), list_128436, float_128437)
    # Adding element type (line 189)
    float_128438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 30), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 189, 25), list_128436, float_128438)
    # Adding element type (line 189)
    float_128439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 35), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 189, 25), list_128436, float_128439)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 24), list_128427, list_128436)
    
    # Processing the call keyword arguments (line 187)
    kwargs_128440 = {}
    # Getting the type of 'np' (line 187)
    np_128425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 15), 'np', False)
    # Obtaining the member 'array' of a type (line 187)
    array_128426 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 15), np_128425, 'array')
    # Calling array(args, kwargs) (line 187)
    array_call_result_128441 = invoke(stypy.reporting.localization.Localization(__file__, 187, 15), array_128426, *[list_128427], **kwargs_128440)
    
    # Assigning a type to the variable 'expected' (line 187)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 4), 'expected', array_call_result_128441)
    
    # Assigning a List to a Name (line 191):
    
    # Obtaining an instance of the builtin type 'list' (line 191)
    list_128442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 12), 'list')
    # Adding type elements to the builtin type 'list' instance (line 191)
    # Adding element type (line 191)
    str_128443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 13), 'str', 'reflect')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 191, 12), list_128442, str_128443)
    # Adding element type (line 191)
    str_128444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 24), 'str', 'wrap')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 191, 12), list_128442, str_128444)
    
    # Assigning a type to the variable 'modes' (line 191)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 4), 'modes', list_128442)
    
    # Call to assert_equal(...): (line 193)
    # Processing the call arguments (line 193)
    # Getting the type of 'expected' (line 193)
    expected_128446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 17), 'expected', False)
    
    # Call to sobel(...): (line 194)
    # Processing the call arguments (line 194)
    # Getting the type of 'arr' (line 194)
    arr_128449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 28), 'arr', False)
    # Processing the call keyword arguments (line 194)
    # Getting the type of 'modes' (line 194)
    modes_128450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 38), 'modes', False)
    keyword_128451 = modes_128450
    kwargs_128452 = {'mode': keyword_128451}
    # Getting the type of 'sndi' (line 194)
    sndi_128447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 17), 'sndi', False)
    # Obtaining the member 'sobel' of a type (line 194)
    sobel_128448 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 17), sndi_128447, 'sobel')
    # Calling sobel(args, kwargs) (line 194)
    sobel_call_result_128453 = invoke(stypy.reporting.localization.Localization(__file__, 194, 17), sobel_128448, *[arr_128449], **kwargs_128452)
    
    # Processing the call keyword arguments (line 193)
    kwargs_128454 = {}
    # Getting the type of 'assert_equal' (line 193)
    assert_equal_128445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 193)
    assert_equal_call_result_128455 = invoke(stypy.reporting.localization.Localization(__file__, 193, 4), assert_equal_128445, *[expected_128446, sobel_call_result_128453], **kwargs_128454)
    
    
    # ################# End of 'test_multiple_modes_sobel(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_multiple_modes_sobel' in the type store
    # Getting the type of 'stypy_return_type' (line 181)
    stypy_return_type_128456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_128456)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_multiple_modes_sobel'
    return stypy_return_type_128456

# Assigning a type to the variable 'test_multiple_modes_sobel' (line 181)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 0), 'test_multiple_modes_sobel', test_multiple_modes_sobel)

@norecursion
def test_multiple_modes_laplace(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_multiple_modes_laplace'
    module_type_store = module_type_store.open_function_context('test_multiple_modes_laplace', 197, 0, False)
    
    # Passed parameters checking function
    test_multiple_modes_laplace.stypy_localization = localization
    test_multiple_modes_laplace.stypy_type_of_self = None
    test_multiple_modes_laplace.stypy_type_store = module_type_store
    test_multiple_modes_laplace.stypy_function_name = 'test_multiple_modes_laplace'
    test_multiple_modes_laplace.stypy_param_names_list = []
    test_multiple_modes_laplace.stypy_varargs_param_name = None
    test_multiple_modes_laplace.stypy_kwargs_param_name = None
    test_multiple_modes_laplace.stypy_call_defaults = defaults
    test_multiple_modes_laplace.stypy_call_varargs = varargs
    test_multiple_modes_laplace.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_multiple_modes_laplace', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_multiple_modes_laplace', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_multiple_modes_laplace(...)' code ##################

    
    # Assigning a Call to a Name (line 199):
    
    # Call to array(...): (line 199)
    # Processing the call arguments (line 199)
    
    # Obtaining an instance of the builtin type 'list' (line 199)
    list_128459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 199)
    # Adding element type (line 199)
    
    # Obtaining an instance of the builtin type 'list' (line 199)
    list_128460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 199)
    # Adding element type (line 199)
    float_128461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 21), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 20), list_128460, float_128461)
    # Adding element type (line 199)
    float_128462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 25), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 20), list_128460, float_128462)
    # Adding element type (line 199)
    float_128463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 29), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 20), list_128460, float_128463)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 19), list_128459, list_128460)
    # Adding element type (line 199)
    
    # Obtaining an instance of the builtin type 'list' (line 200)
    list_128464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 200)
    # Adding element type (line 200)
    float_128465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 21), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 20), list_128464, float_128465)
    # Adding element type (line 200)
    float_128466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 25), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 20), list_128464, float_128466)
    # Adding element type (line 200)
    float_128467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 29), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 20), list_128464, float_128467)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 19), list_128459, list_128464)
    # Adding element type (line 199)
    
    # Obtaining an instance of the builtin type 'list' (line 201)
    list_128468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 201)
    # Adding element type (line 201)
    float_128469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 21), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 201, 20), list_128468, float_128469)
    # Adding element type (line 201)
    float_128470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 25), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 201, 20), list_128468, float_128470)
    # Adding element type (line 201)
    float_128471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 29), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 201, 20), list_128468, float_128471)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 19), list_128459, list_128468)
    
    # Processing the call keyword arguments (line 199)
    kwargs_128472 = {}
    # Getting the type of 'np' (line 199)
    np_128457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 10), 'np', False)
    # Obtaining the member 'array' of a type (line 199)
    array_128458 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 10), np_128457, 'array')
    # Calling array(args, kwargs) (line 199)
    array_call_result_128473 = invoke(stypy.reporting.localization.Localization(__file__, 199, 10), array_128458, *[list_128459], **kwargs_128472)
    
    # Assigning a type to the variable 'arr' (line 199)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 4), 'arr', array_call_result_128473)
    
    # Assigning a Call to a Name (line 203):
    
    # Call to array(...): (line 203)
    # Processing the call arguments (line 203)
    
    # Obtaining an instance of the builtin type 'list' (line 203)
    list_128476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 203)
    # Adding element type (line 203)
    
    # Obtaining an instance of the builtin type 'list' (line 203)
    list_128477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 25), 'list')
    # Adding type elements to the builtin type 'list' instance (line 203)
    # Adding element type (line 203)
    float_128478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 26), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 203, 25), list_128477, float_128478)
    # Adding element type (line 203)
    float_128479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 31), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 203, 25), list_128477, float_128479)
    # Adding element type (line 203)
    float_128480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 35), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 203, 25), list_128477, float_128480)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 203, 24), list_128476, list_128477)
    # Adding element type (line 203)
    
    # Obtaining an instance of the builtin type 'list' (line 204)
    list_128481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 25), 'list')
    # Adding type elements to the builtin type 'list' instance (line 204)
    # Adding element type (line 204)
    float_128482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 26), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 204, 25), list_128481, float_128482)
    # Adding element type (line 204)
    float_128483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 31), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 204, 25), list_128481, float_128483)
    # Adding element type (line 204)
    float_128484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 36), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 204, 25), list_128481, float_128484)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 203, 24), list_128476, list_128481)
    # Adding element type (line 203)
    
    # Obtaining an instance of the builtin type 'list' (line 205)
    list_128485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 25), 'list')
    # Adding type elements to the builtin type 'list' instance (line 205)
    # Adding element type (line 205)
    float_128486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 26), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 205, 25), list_128485, float_128486)
    # Adding element type (line 205)
    float_128487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 30), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 205, 25), list_128485, float_128487)
    # Adding element type (line 205)
    float_128488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 34), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 205, 25), list_128485, float_128488)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 203, 24), list_128476, list_128485)
    
    # Processing the call keyword arguments (line 203)
    kwargs_128489 = {}
    # Getting the type of 'np' (line 203)
    np_128474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 15), 'np', False)
    # Obtaining the member 'array' of a type (line 203)
    array_128475 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 15), np_128474, 'array')
    # Calling array(args, kwargs) (line 203)
    array_call_result_128490 = invoke(stypy.reporting.localization.Localization(__file__, 203, 15), array_128475, *[list_128476], **kwargs_128489)
    
    # Assigning a type to the variable 'expected' (line 203)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 4), 'expected', array_call_result_128490)
    
    # Assigning a List to a Name (line 207):
    
    # Obtaining an instance of the builtin type 'list' (line 207)
    list_128491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 12), 'list')
    # Adding type elements to the builtin type 'list' instance (line 207)
    # Adding element type (line 207)
    str_128492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 13), 'str', 'reflect')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 207, 12), list_128491, str_128492)
    # Adding element type (line 207)
    str_128493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 24), 'str', 'wrap')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 207, 12), list_128491, str_128493)
    
    # Assigning a type to the variable 'modes' (line 207)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 4), 'modes', list_128491)
    
    # Call to assert_equal(...): (line 209)
    # Processing the call arguments (line 209)
    # Getting the type of 'expected' (line 209)
    expected_128495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 17), 'expected', False)
    
    # Call to laplace(...): (line 210)
    # Processing the call arguments (line 210)
    # Getting the type of 'arr' (line 210)
    arr_128498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 30), 'arr', False)
    # Processing the call keyword arguments (line 210)
    # Getting the type of 'modes' (line 210)
    modes_128499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 40), 'modes', False)
    keyword_128500 = modes_128499
    kwargs_128501 = {'mode': keyword_128500}
    # Getting the type of 'sndi' (line 210)
    sndi_128496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 17), 'sndi', False)
    # Obtaining the member 'laplace' of a type (line 210)
    laplace_128497 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 17), sndi_128496, 'laplace')
    # Calling laplace(args, kwargs) (line 210)
    laplace_call_result_128502 = invoke(stypy.reporting.localization.Localization(__file__, 210, 17), laplace_128497, *[arr_128498], **kwargs_128501)
    
    # Processing the call keyword arguments (line 209)
    kwargs_128503 = {}
    # Getting the type of 'assert_equal' (line 209)
    assert_equal_128494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 209)
    assert_equal_call_result_128504 = invoke(stypy.reporting.localization.Localization(__file__, 209, 4), assert_equal_128494, *[expected_128495, laplace_call_result_128502], **kwargs_128503)
    
    
    # ################# End of 'test_multiple_modes_laplace(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_multiple_modes_laplace' in the type store
    # Getting the type of 'stypy_return_type' (line 197)
    stypy_return_type_128505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_128505)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_multiple_modes_laplace'
    return stypy_return_type_128505

# Assigning a type to the variable 'test_multiple_modes_laplace' (line 197)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 0), 'test_multiple_modes_laplace', test_multiple_modes_laplace)

@norecursion
def test_multiple_modes_gaussian_laplace(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_multiple_modes_gaussian_laplace'
    module_type_store = module_type_store.open_function_context('test_multiple_modes_gaussian_laplace', 213, 0, False)
    
    # Passed parameters checking function
    test_multiple_modes_gaussian_laplace.stypy_localization = localization
    test_multiple_modes_gaussian_laplace.stypy_type_of_self = None
    test_multiple_modes_gaussian_laplace.stypy_type_store = module_type_store
    test_multiple_modes_gaussian_laplace.stypy_function_name = 'test_multiple_modes_gaussian_laplace'
    test_multiple_modes_gaussian_laplace.stypy_param_names_list = []
    test_multiple_modes_gaussian_laplace.stypy_varargs_param_name = None
    test_multiple_modes_gaussian_laplace.stypy_kwargs_param_name = None
    test_multiple_modes_gaussian_laplace.stypy_call_defaults = defaults
    test_multiple_modes_gaussian_laplace.stypy_call_varargs = varargs
    test_multiple_modes_gaussian_laplace.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_multiple_modes_gaussian_laplace', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_multiple_modes_gaussian_laplace', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_multiple_modes_gaussian_laplace(...)' code ##################

    
    # Assigning a Call to a Name (line 215):
    
    # Call to array(...): (line 215)
    # Processing the call arguments (line 215)
    
    # Obtaining an instance of the builtin type 'list' (line 215)
    list_128508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 215)
    # Adding element type (line 215)
    
    # Obtaining an instance of the builtin type 'list' (line 215)
    list_128509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 215)
    # Adding element type (line 215)
    float_128510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 21), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 215, 20), list_128509, float_128510)
    # Adding element type (line 215)
    float_128511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 25), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 215, 20), list_128509, float_128511)
    # Adding element type (line 215)
    float_128512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 29), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 215, 20), list_128509, float_128512)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 215, 19), list_128508, list_128509)
    # Adding element type (line 215)
    
    # Obtaining an instance of the builtin type 'list' (line 216)
    list_128513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 216)
    # Adding element type (line 216)
    float_128514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 21), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 216, 20), list_128513, float_128514)
    # Adding element type (line 216)
    float_128515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 25), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 216, 20), list_128513, float_128515)
    # Adding element type (line 216)
    float_128516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 29), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 216, 20), list_128513, float_128516)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 215, 19), list_128508, list_128513)
    # Adding element type (line 215)
    
    # Obtaining an instance of the builtin type 'list' (line 217)
    list_128517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 217)
    # Adding element type (line 217)
    float_128518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 21), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 20), list_128517, float_128518)
    # Adding element type (line 217)
    float_128519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 25), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 20), list_128517, float_128519)
    # Adding element type (line 217)
    float_128520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 29), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 20), list_128517, float_128520)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 215, 19), list_128508, list_128517)
    
    # Processing the call keyword arguments (line 215)
    kwargs_128521 = {}
    # Getting the type of 'np' (line 215)
    np_128506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 10), 'np', False)
    # Obtaining the member 'array' of a type (line 215)
    array_128507 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 10), np_128506, 'array')
    # Calling array(args, kwargs) (line 215)
    array_call_result_128522 = invoke(stypy.reporting.localization.Localization(__file__, 215, 10), array_128507, *[list_128508], **kwargs_128521)
    
    # Assigning a type to the variable 'arr' (line 215)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 4), 'arr', array_call_result_128522)
    
    # Assigning a Call to a Name (line 219):
    
    # Call to array(...): (line 219)
    # Processing the call arguments (line 219)
    
    # Obtaining an instance of the builtin type 'list' (line 219)
    list_128525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 219)
    # Adding element type (line 219)
    
    # Obtaining an instance of the builtin type 'list' (line 219)
    list_128526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 25), 'list')
    # Adding type elements to the builtin type 'list' instance (line 219)
    # Adding element type (line 219)
    float_128527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 26), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 219, 25), list_128526, float_128527)
    # Adding element type (line 219)
    float_128528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 39), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 219, 25), list_128526, float_128528)
    # Adding element type (line 219)
    float_128529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 51), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 219, 25), list_128526, float_128529)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 219, 24), list_128525, list_128526)
    # Adding element type (line 219)
    
    # Obtaining an instance of the builtin type 'list' (line 220)
    list_128530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 25), 'list')
    # Adding type elements to the builtin type 'list' instance (line 220)
    # Adding element type (line 220)
    float_128531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 26), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 220, 25), list_128530, float_128531)
    # Adding element type (line 220)
    float_128532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 39), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 220, 25), list_128530, float_128532)
    # Adding element type (line 220)
    float_128533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 52), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 220, 25), list_128530, float_128533)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 219, 24), list_128525, list_128530)
    # Adding element type (line 219)
    
    # Obtaining an instance of the builtin type 'list' (line 221)
    list_128534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 25), 'list')
    # Adding type elements to the builtin type 'list' instance (line 221)
    # Adding element type (line 221)
    float_128535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 26), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 221, 25), list_128534, float_128535)
    # Adding element type (line 221)
    float_128536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 38), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 221, 25), list_128534, float_128536)
    # Adding element type (line 221)
    float_128537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 50), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 221, 25), list_128534, float_128537)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 219, 24), list_128525, list_128534)
    
    # Processing the call keyword arguments (line 219)
    kwargs_128538 = {}
    # Getting the type of 'np' (line 219)
    np_128523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 15), 'np', False)
    # Obtaining the member 'array' of a type (line 219)
    array_128524 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 15), np_128523, 'array')
    # Calling array(args, kwargs) (line 219)
    array_call_result_128539 = invoke(stypy.reporting.localization.Localization(__file__, 219, 15), array_128524, *[list_128525], **kwargs_128538)
    
    # Assigning a type to the variable 'expected' (line 219)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 4), 'expected', array_call_result_128539)
    
    # Assigning a List to a Name (line 223):
    
    # Obtaining an instance of the builtin type 'list' (line 223)
    list_128540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 12), 'list')
    # Adding type elements to the builtin type 'list' instance (line 223)
    # Adding element type (line 223)
    str_128541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 13), 'str', 'reflect')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 223, 12), list_128540, str_128541)
    # Adding element type (line 223)
    str_128542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 24), 'str', 'wrap')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 223, 12), list_128540, str_128542)
    
    # Assigning a type to the variable 'modes' (line 223)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 4), 'modes', list_128540)
    
    # Call to assert_almost_equal(...): (line 225)
    # Processing the call arguments (line 225)
    # Getting the type of 'expected' (line 225)
    expected_128544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 24), 'expected', False)
    
    # Call to gaussian_laplace(...): (line 226)
    # Processing the call arguments (line 226)
    # Getting the type of 'arr' (line 226)
    arr_128547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 46), 'arr', False)
    int_128548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 51), 'int')
    # Processing the call keyword arguments (line 226)
    # Getting the type of 'modes' (line 226)
    modes_128549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 59), 'modes', False)
    keyword_128550 = modes_128549
    kwargs_128551 = {'mode': keyword_128550}
    # Getting the type of 'sndi' (line 226)
    sndi_128545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 24), 'sndi', False)
    # Obtaining the member 'gaussian_laplace' of a type (line 226)
    gaussian_laplace_128546 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 24), sndi_128545, 'gaussian_laplace')
    # Calling gaussian_laplace(args, kwargs) (line 226)
    gaussian_laplace_call_result_128552 = invoke(stypy.reporting.localization.Localization(__file__, 226, 24), gaussian_laplace_128546, *[arr_128547, int_128548], **kwargs_128551)
    
    # Processing the call keyword arguments (line 225)
    kwargs_128553 = {}
    # Getting the type of 'assert_almost_equal' (line 225)
    assert_almost_equal_128543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 4), 'assert_almost_equal', False)
    # Calling assert_almost_equal(args, kwargs) (line 225)
    assert_almost_equal_call_result_128554 = invoke(stypy.reporting.localization.Localization(__file__, 225, 4), assert_almost_equal_128543, *[expected_128544, gaussian_laplace_call_result_128552], **kwargs_128553)
    
    
    # ################# End of 'test_multiple_modes_gaussian_laplace(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_multiple_modes_gaussian_laplace' in the type store
    # Getting the type of 'stypy_return_type' (line 213)
    stypy_return_type_128555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_128555)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_multiple_modes_gaussian_laplace'
    return stypy_return_type_128555

# Assigning a type to the variable 'test_multiple_modes_gaussian_laplace' (line 213)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 0), 'test_multiple_modes_gaussian_laplace', test_multiple_modes_gaussian_laplace)

@norecursion
def test_multiple_modes_gaussian_gradient_magnitude(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_multiple_modes_gaussian_gradient_magnitude'
    module_type_store = module_type_store.open_function_context('test_multiple_modes_gaussian_gradient_magnitude', 229, 0, False)
    
    # Passed parameters checking function
    test_multiple_modes_gaussian_gradient_magnitude.stypy_localization = localization
    test_multiple_modes_gaussian_gradient_magnitude.stypy_type_of_self = None
    test_multiple_modes_gaussian_gradient_magnitude.stypy_type_store = module_type_store
    test_multiple_modes_gaussian_gradient_magnitude.stypy_function_name = 'test_multiple_modes_gaussian_gradient_magnitude'
    test_multiple_modes_gaussian_gradient_magnitude.stypy_param_names_list = []
    test_multiple_modes_gaussian_gradient_magnitude.stypy_varargs_param_name = None
    test_multiple_modes_gaussian_gradient_magnitude.stypy_kwargs_param_name = None
    test_multiple_modes_gaussian_gradient_magnitude.stypy_call_defaults = defaults
    test_multiple_modes_gaussian_gradient_magnitude.stypy_call_varargs = varargs
    test_multiple_modes_gaussian_gradient_magnitude.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_multiple_modes_gaussian_gradient_magnitude', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_multiple_modes_gaussian_gradient_magnitude', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_multiple_modes_gaussian_gradient_magnitude(...)' code ##################

    
    # Assigning a Call to a Name (line 232):
    
    # Call to array(...): (line 232)
    # Processing the call arguments (line 232)
    
    # Obtaining an instance of the builtin type 'list' (line 232)
    list_128558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 232)
    # Adding element type (line 232)
    
    # Obtaining an instance of the builtin type 'list' (line 232)
    list_128559 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 232)
    # Adding element type (line 232)
    float_128560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 21), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 232, 20), list_128559, float_128560)
    # Adding element type (line 232)
    float_128561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 25), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 232, 20), list_128559, float_128561)
    # Adding element type (line 232)
    float_128562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 29), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 232, 20), list_128559, float_128562)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 232, 19), list_128558, list_128559)
    # Adding element type (line 232)
    
    # Obtaining an instance of the builtin type 'list' (line 233)
    list_128563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 233)
    # Adding element type (line 233)
    float_128564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 21), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 233, 20), list_128563, float_128564)
    # Adding element type (line 233)
    float_128565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 25), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 233, 20), list_128563, float_128565)
    # Adding element type (line 233)
    float_128566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 29), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 233, 20), list_128563, float_128566)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 232, 19), list_128558, list_128563)
    # Adding element type (line 232)
    
    # Obtaining an instance of the builtin type 'list' (line 234)
    list_128567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 234)
    # Adding element type (line 234)
    float_128568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 21), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 234, 20), list_128567, float_128568)
    # Adding element type (line 234)
    float_128569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 25), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 234, 20), list_128567, float_128569)
    # Adding element type (line 234)
    float_128570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 29), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 234, 20), list_128567, float_128570)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 232, 19), list_128558, list_128567)
    
    # Processing the call keyword arguments (line 232)
    kwargs_128571 = {}
    # Getting the type of 'np' (line 232)
    np_128556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 10), 'np', False)
    # Obtaining the member 'array' of a type (line 232)
    array_128557 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 10), np_128556, 'array')
    # Calling array(args, kwargs) (line 232)
    array_call_result_128572 = invoke(stypy.reporting.localization.Localization(__file__, 232, 10), array_128557, *[list_128558], **kwargs_128571)
    
    # Assigning a type to the variable 'arr' (line 232)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 4), 'arr', array_call_result_128572)
    
    # Assigning a Call to a Name (line 236):
    
    # Call to array(...): (line 236)
    # Processing the call arguments (line 236)
    
    # Obtaining an instance of the builtin type 'list' (line 236)
    list_128575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 236)
    # Adding element type (line 236)
    
    # Obtaining an instance of the builtin type 'list' (line 236)
    list_128576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 25), 'list')
    # Adding type elements to the builtin type 'list' instance (line 236)
    # Adding element type (line 236)
    float_128577 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 26), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 236, 25), list_128576, float_128577)
    # Adding element type (line 236)
    float_128578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 38), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 236, 25), list_128576, float_128578)
    # Adding element type (line 236)
    float_128579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 50), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 236, 25), list_128576, float_128579)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 236, 24), list_128575, list_128576)
    # Adding element type (line 236)
    
    # Obtaining an instance of the builtin type 'list' (line 237)
    list_128580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 25), 'list')
    # Adding type elements to the builtin type 'list' instance (line 237)
    # Adding element type (line 237)
    float_128581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 26), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 237, 25), list_128580, float_128581)
    # Adding element type (line 237)
    float_128582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 38), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 237, 25), list_128580, float_128582)
    # Adding element type (line 237)
    float_128583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 50), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 237, 25), list_128580, float_128583)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 236, 24), list_128575, list_128580)
    # Adding element type (line 236)
    
    # Obtaining an instance of the builtin type 'list' (line 238)
    list_128584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 25), 'list')
    # Adding type elements to the builtin type 'list' instance (line 238)
    # Adding element type (line 238)
    float_128585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 26), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 238, 25), list_128584, float_128585)
    # Adding element type (line 238)
    float_128586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 38), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 238, 25), list_128584, float_128586)
    # Adding element type (line 238)
    float_128587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 50), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 238, 25), list_128584, float_128587)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 236, 24), list_128575, list_128584)
    
    # Processing the call keyword arguments (line 236)
    kwargs_128588 = {}
    # Getting the type of 'np' (line 236)
    np_128573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 15), 'np', False)
    # Obtaining the member 'array' of a type (line 236)
    array_128574 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 15), np_128573, 'array')
    # Calling array(args, kwargs) (line 236)
    array_call_result_128589 = invoke(stypy.reporting.localization.Localization(__file__, 236, 15), array_128574, *[list_128575], **kwargs_128588)
    
    # Assigning a type to the variable 'expected' (line 236)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 4), 'expected', array_call_result_128589)
    
    # Assigning a List to a Name (line 240):
    
    # Obtaining an instance of the builtin type 'list' (line 240)
    list_128590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 12), 'list')
    # Adding type elements to the builtin type 'list' instance (line 240)
    # Adding element type (line 240)
    str_128591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 13), 'str', 'reflect')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 240, 12), list_128590, str_128591)
    # Adding element type (line 240)
    str_128592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 24), 'str', 'wrap')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 240, 12), list_128590, str_128592)
    
    # Assigning a type to the variable 'modes' (line 240)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 4), 'modes', list_128590)
    
    # Assigning a Call to a Name (line 242):
    
    # Call to gaussian_gradient_magnitude(...): (line 242)
    # Processing the call arguments (line 242)
    # Getting the type of 'arr' (line 242)
    arr_128595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 50), 'arr', False)
    int_128596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 55), 'int')
    # Processing the call keyword arguments (line 242)
    # Getting the type of 'modes' (line 242)
    modes_128597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 63), 'modes', False)
    keyword_128598 = modes_128597
    kwargs_128599 = {'mode': keyword_128598}
    # Getting the type of 'sndi' (line 242)
    sndi_128593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 17), 'sndi', False)
    # Obtaining the member 'gaussian_gradient_magnitude' of a type (line 242)
    gaussian_gradient_magnitude_128594 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 17), sndi_128593, 'gaussian_gradient_magnitude')
    # Calling gaussian_gradient_magnitude(args, kwargs) (line 242)
    gaussian_gradient_magnitude_call_result_128600 = invoke(stypy.reporting.localization.Localization(__file__, 242, 17), gaussian_gradient_magnitude_128594, *[arr_128595, int_128596], **kwargs_128599)
    
    # Assigning a type to the variable 'calculated' (line 242)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 4), 'calculated', gaussian_gradient_magnitude_call_result_128600)
    
    # Call to assert_almost_equal(...): (line 244)
    # Processing the call arguments (line 244)
    # Getting the type of 'expected' (line 244)
    expected_128602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 24), 'expected', False)
    # Getting the type of 'calculated' (line 244)
    calculated_128603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 34), 'calculated', False)
    # Processing the call keyword arguments (line 244)
    kwargs_128604 = {}
    # Getting the type of 'assert_almost_equal' (line 244)
    assert_almost_equal_128601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 4), 'assert_almost_equal', False)
    # Calling assert_almost_equal(args, kwargs) (line 244)
    assert_almost_equal_call_result_128605 = invoke(stypy.reporting.localization.Localization(__file__, 244, 4), assert_almost_equal_128601, *[expected_128602, calculated_128603], **kwargs_128604)
    
    
    # ################# End of 'test_multiple_modes_gaussian_gradient_magnitude(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_multiple_modes_gaussian_gradient_magnitude' in the type store
    # Getting the type of 'stypy_return_type' (line 229)
    stypy_return_type_128606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_128606)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_multiple_modes_gaussian_gradient_magnitude'
    return stypy_return_type_128606

# Assigning a type to the variable 'test_multiple_modes_gaussian_gradient_magnitude' (line 229)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 0), 'test_multiple_modes_gaussian_gradient_magnitude', test_multiple_modes_gaussian_gradient_magnitude)

@norecursion
def test_multiple_modes_uniform(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_multiple_modes_uniform'
    module_type_store = module_type_store.open_function_context('test_multiple_modes_uniform', 247, 0, False)
    
    # Passed parameters checking function
    test_multiple_modes_uniform.stypy_localization = localization
    test_multiple_modes_uniform.stypy_type_of_self = None
    test_multiple_modes_uniform.stypy_type_store = module_type_store
    test_multiple_modes_uniform.stypy_function_name = 'test_multiple_modes_uniform'
    test_multiple_modes_uniform.stypy_param_names_list = []
    test_multiple_modes_uniform.stypy_varargs_param_name = None
    test_multiple_modes_uniform.stypy_kwargs_param_name = None
    test_multiple_modes_uniform.stypy_call_defaults = defaults
    test_multiple_modes_uniform.stypy_call_varargs = varargs
    test_multiple_modes_uniform.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_multiple_modes_uniform', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_multiple_modes_uniform', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_multiple_modes_uniform(...)' code ##################

    
    # Assigning a Call to a Name (line 249):
    
    # Call to array(...): (line 249)
    # Processing the call arguments (line 249)
    
    # Obtaining an instance of the builtin type 'list' (line 249)
    list_128609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 249)
    # Adding element type (line 249)
    
    # Obtaining an instance of the builtin type 'list' (line 249)
    list_128610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 249)
    # Adding element type (line 249)
    float_128611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 21), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 249, 20), list_128610, float_128611)
    # Adding element type (line 249)
    float_128612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 25), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 249, 20), list_128610, float_128612)
    # Adding element type (line 249)
    float_128613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 29), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 249, 20), list_128610, float_128613)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 249, 19), list_128609, list_128610)
    # Adding element type (line 249)
    
    # Obtaining an instance of the builtin type 'list' (line 250)
    list_128614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 250)
    # Adding element type (line 250)
    float_128615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 21), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 250, 20), list_128614, float_128615)
    # Adding element type (line 250)
    float_128616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 25), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 250, 20), list_128614, float_128616)
    # Adding element type (line 250)
    float_128617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 29), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 250, 20), list_128614, float_128617)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 249, 19), list_128609, list_128614)
    # Adding element type (line 249)
    
    # Obtaining an instance of the builtin type 'list' (line 251)
    list_128618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 251)
    # Adding element type (line 251)
    float_128619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 21), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 20), list_128618, float_128619)
    # Adding element type (line 251)
    float_128620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 25), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 20), list_128618, float_128620)
    # Adding element type (line 251)
    float_128621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 29), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 20), list_128618, float_128621)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 249, 19), list_128609, list_128618)
    
    # Processing the call keyword arguments (line 249)
    kwargs_128622 = {}
    # Getting the type of 'np' (line 249)
    np_128607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 10), 'np', False)
    # Obtaining the member 'array' of a type (line 249)
    array_128608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 10), np_128607, 'array')
    # Calling array(args, kwargs) (line 249)
    array_call_result_128623 = invoke(stypy.reporting.localization.Localization(__file__, 249, 10), array_128608, *[list_128609], **kwargs_128622)
    
    # Assigning a type to the variable 'arr' (line 249)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 4), 'arr', array_call_result_128623)
    
    # Assigning a Call to a Name (line 253):
    
    # Call to array(...): (line 253)
    # Processing the call arguments (line 253)
    
    # Obtaining an instance of the builtin type 'list' (line 253)
    list_128626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 253)
    # Adding element type (line 253)
    
    # Obtaining an instance of the builtin type 'list' (line 253)
    list_128627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 25), 'list')
    # Adding type elements to the builtin type 'list' instance (line 253)
    # Adding element type (line 253)
    float_128628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 26), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 253, 25), list_128627, float_128628)
    # Adding element type (line 253)
    float_128629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 32), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 253, 25), list_128627, float_128629)
    # Adding element type (line 253)
    float_128630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 38), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 253, 25), list_128627, float_128630)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 253, 24), list_128626, list_128627)
    # Adding element type (line 253)
    
    # Obtaining an instance of the builtin type 'list' (line 254)
    list_128631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 25), 'list')
    # Adding type elements to the builtin type 'list' instance (line 254)
    # Adding element type (line 254)
    float_128632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 26), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 25), list_128631, float_128632)
    # Adding element type (line 254)
    float_128633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 32), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 25), list_128631, float_128633)
    # Adding element type (line 254)
    float_128634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 38), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 25), list_128631, float_128634)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 253, 24), list_128626, list_128631)
    # Adding element type (line 253)
    
    # Obtaining an instance of the builtin type 'list' (line 255)
    list_128635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 25), 'list')
    # Adding type elements to the builtin type 'list' instance (line 255)
    # Adding element type (line 255)
    float_128636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 26), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 255, 25), list_128635, float_128636)
    # Adding element type (line 255)
    float_128637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 32), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 255, 25), list_128635, float_128637)
    # Adding element type (line 255)
    float_128638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 38), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 255, 25), list_128635, float_128638)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 253, 24), list_128626, list_128635)
    
    # Processing the call keyword arguments (line 253)
    kwargs_128639 = {}
    # Getting the type of 'np' (line 253)
    np_128624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 15), 'np', False)
    # Obtaining the member 'array' of a type (line 253)
    array_128625 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 15), np_128624, 'array')
    # Calling array(args, kwargs) (line 253)
    array_call_result_128640 = invoke(stypy.reporting.localization.Localization(__file__, 253, 15), array_128625, *[list_128626], **kwargs_128639)
    
    # Assigning a type to the variable 'expected' (line 253)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 4), 'expected', array_call_result_128640)
    
    # Assigning a List to a Name (line 257):
    
    # Obtaining an instance of the builtin type 'list' (line 257)
    list_128641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 12), 'list')
    # Adding type elements to the builtin type 'list' instance (line 257)
    # Adding element type (line 257)
    str_128642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 13), 'str', 'reflect')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 12), list_128641, str_128642)
    # Adding element type (line 257)
    str_128643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 24), 'str', 'wrap')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 12), list_128641, str_128643)
    
    # Assigning a type to the variable 'modes' (line 257)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 4), 'modes', list_128641)
    
    # Call to assert_almost_equal(...): (line 259)
    # Processing the call arguments (line 259)
    # Getting the type of 'expected' (line 259)
    expected_128645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 24), 'expected', False)
    
    # Call to uniform_filter(...): (line 260)
    # Processing the call arguments (line 260)
    # Getting the type of 'arr' (line 260)
    arr_128648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 44), 'arr', False)
    int_128649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 49), 'int')
    # Processing the call keyword arguments (line 260)
    # Getting the type of 'modes' (line 260)
    modes_128650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 57), 'modes', False)
    keyword_128651 = modes_128650
    kwargs_128652 = {'mode': keyword_128651}
    # Getting the type of 'sndi' (line 260)
    sndi_128646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 24), 'sndi', False)
    # Obtaining the member 'uniform_filter' of a type (line 260)
    uniform_filter_128647 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 24), sndi_128646, 'uniform_filter')
    # Calling uniform_filter(args, kwargs) (line 260)
    uniform_filter_call_result_128653 = invoke(stypy.reporting.localization.Localization(__file__, 260, 24), uniform_filter_128647, *[arr_128648, int_128649], **kwargs_128652)
    
    # Processing the call keyword arguments (line 259)
    kwargs_128654 = {}
    # Getting the type of 'assert_almost_equal' (line 259)
    assert_almost_equal_128644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 4), 'assert_almost_equal', False)
    # Calling assert_almost_equal(args, kwargs) (line 259)
    assert_almost_equal_call_result_128655 = invoke(stypy.reporting.localization.Localization(__file__, 259, 4), assert_almost_equal_128644, *[expected_128645, uniform_filter_call_result_128653], **kwargs_128654)
    
    
    # ################# End of 'test_multiple_modes_uniform(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_multiple_modes_uniform' in the type store
    # Getting the type of 'stypy_return_type' (line 247)
    stypy_return_type_128656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_128656)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_multiple_modes_uniform'
    return stypy_return_type_128656

# Assigning a type to the variable 'test_multiple_modes_uniform' (line 247)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 0), 'test_multiple_modes_uniform', test_multiple_modes_uniform)

@norecursion
def test_gaussian_truncate(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_gaussian_truncate'
    module_type_store = module_type_store.open_function_context('test_gaussian_truncate', 263, 0, False)
    
    # Passed parameters checking function
    test_gaussian_truncate.stypy_localization = localization
    test_gaussian_truncate.stypy_type_of_self = None
    test_gaussian_truncate.stypy_type_store = module_type_store
    test_gaussian_truncate.stypy_function_name = 'test_gaussian_truncate'
    test_gaussian_truncate.stypy_param_names_list = []
    test_gaussian_truncate.stypy_varargs_param_name = None
    test_gaussian_truncate.stypy_kwargs_param_name = None
    test_gaussian_truncate.stypy_call_defaults = defaults
    test_gaussian_truncate.stypy_call_varargs = varargs
    test_gaussian_truncate.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_gaussian_truncate', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_gaussian_truncate', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_gaussian_truncate(...)' code ##################

    
    # Assigning a Call to a Name (line 267):
    
    # Call to zeros(...): (line 267)
    # Processing the call arguments (line 267)
    
    # Obtaining an instance of the builtin type 'tuple' (line 267)
    tuple_128659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 20), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 267)
    # Adding element type (line 267)
    int_128660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 267, 20), tuple_128659, int_128660)
    # Adding element type (line 267)
    int_128661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 267, 20), tuple_128659, int_128661)
    
    # Getting the type of 'float' (line 267)
    float_128662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 31), 'float', False)
    # Processing the call keyword arguments (line 267)
    kwargs_128663 = {}
    # Getting the type of 'np' (line 267)
    np_128657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 10), 'np', False)
    # Obtaining the member 'zeros' of a type (line 267)
    zeros_128658 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 267, 10), np_128657, 'zeros')
    # Calling zeros(args, kwargs) (line 267)
    zeros_call_result_128664 = invoke(stypy.reporting.localization.Localization(__file__, 267, 10), zeros_128658, *[tuple_128659, float_128662], **kwargs_128663)
    
    # Assigning a type to the variable 'arr' (line 267)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 4), 'arr', zeros_call_result_128664)
    
    # Assigning a Num to a Subscript (line 268):
    int_128665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 18), 'int')
    # Getting the type of 'arr' (line 268)
    arr_128666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 4), 'arr')
    
    # Obtaining an instance of the builtin type 'tuple' (line 268)
    tuple_128667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 8), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 268)
    # Adding element type (line 268)
    int_128668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 8), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 268, 8), tuple_128667, int_128668)
    # Adding element type (line 268)
    int_128669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 12), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 268, 8), tuple_128667, int_128669)
    
    # Storing an element on a container (line 268)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 268, 4), arr_128666, (tuple_128667, int_128665))
    
    # Assigning a Call to a Name (line 269):
    
    # Call to sum(...): (line 269)
    # Processing the call keyword arguments (line 269)
    kwargs_128681 = {}
    
    
    # Call to gaussian_filter(...): (line 269)
    # Processing the call arguments (line 269)
    # Getting the type of 'arr' (line 269)
    arr_128672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 43), 'arr', False)
    int_128673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 48), 'int')
    # Processing the call keyword arguments (line 269)
    int_128674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 60), 'int')
    keyword_128675 = int_128674
    kwargs_128676 = {'truncate': keyword_128675}
    # Getting the type of 'sndi' (line 269)
    sndi_128670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 22), 'sndi', False)
    # Obtaining the member 'gaussian_filter' of a type (line 269)
    gaussian_filter_128671 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 22), sndi_128670, 'gaussian_filter')
    # Calling gaussian_filter(args, kwargs) (line 269)
    gaussian_filter_call_result_128677 = invoke(stypy.reporting.localization.Localization(__file__, 269, 22), gaussian_filter_128671, *[arr_128672, int_128673], **kwargs_128676)
    
    int_128678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 65), 'int')
    # Applying the binary operator '>' (line 269)
    result_gt_128679 = python_operator(stypy.reporting.localization.Localization(__file__, 269, 22), '>', gaussian_filter_call_result_128677, int_128678)
    
    # Obtaining the member 'sum' of a type (line 269)
    sum_128680 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 22), result_gt_128679, 'sum')
    # Calling sum(args, kwargs) (line 269)
    sum_call_result_128682 = invoke(stypy.reporting.localization.Localization(__file__, 269, 22), sum_128680, *[], **kwargs_128681)
    
    # Assigning a type to the variable 'num_nonzeros_2' (line 269)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 4), 'num_nonzeros_2', sum_call_result_128682)
    
    # Call to assert_equal(...): (line 270)
    # Processing the call arguments (line 270)
    # Getting the type of 'num_nonzeros_2' (line 270)
    num_nonzeros_2_128684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 17), 'num_nonzeros_2', False)
    int_128685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 33), 'int')
    int_128686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 37), 'int')
    # Applying the binary operator '**' (line 270)
    result_pow_128687 = python_operator(stypy.reporting.localization.Localization(__file__, 270, 33), '**', int_128685, int_128686)
    
    # Processing the call keyword arguments (line 270)
    kwargs_128688 = {}
    # Getting the type of 'assert_equal' (line 270)
    assert_equal_128683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 270)
    assert_equal_call_result_128689 = invoke(stypy.reporting.localization.Localization(__file__, 270, 4), assert_equal_128683, *[num_nonzeros_2_128684, result_pow_128687], **kwargs_128688)
    
    
    # Assigning a Call to a Name (line 271):
    
    # Call to sum(...): (line 271)
    # Processing the call keyword arguments (line 271)
    kwargs_128701 = {}
    
    
    # Call to gaussian_filter(...): (line 271)
    # Processing the call arguments (line 271)
    # Getting the type of 'arr' (line 271)
    arr_128692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 43), 'arr', False)
    int_128693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 48), 'int')
    # Processing the call keyword arguments (line 271)
    int_128694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 60), 'int')
    keyword_128695 = int_128694
    kwargs_128696 = {'truncate': keyword_128695}
    # Getting the type of 'sndi' (line 271)
    sndi_128690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 22), 'sndi', False)
    # Obtaining the member 'gaussian_filter' of a type (line 271)
    gaussian_filter_128691 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 22), sndi_128690, 'gaussian_filter')
    # Calling gaussian_filter(args, kwargs) (line 271)
    gaussian_filter_call_result_128697 = invoke(stypy.reporting.localization.Localization(__file__, 271, 22), gaussian_filter_128691, *[arr_128692, int_128693], **kwargs_128696)
    
    int_128698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 65), 'int')
    # Applying the binary operator '>' (line 271)
    result_gt_128699 = python_operator(stypy.reporting.localization.Localization(__file__, 271, 22), '>', gaussian_filter_call_result_128697, int_128698)
    
    # Obtaining the member 'sum' of a type (line 271)
    sum_128700 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 22), result_gt_128699, 'sum')
    # Calling sum(args, kwargs) (line 271)
    sum_call_result_128702 = invoke(stypy.reporting.localization.Localization(__file__, 271, 22), sum_128700, *[], **kwargs_128701)
    
    # Assigning a type to the variable 'num_nonzeros_5' (line 271)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 4), 'num_nonzeros_5', sum_call_result_128702)
    
    # Call to assert_equal(...): (line 272)
    # Processing the call arguments (line 272)
    # Getting the type of 'num_nonzeros_5' (line 272)
    num_nonzeros_5_128704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 17), 'num_nonzeros_5', False)
    int_128705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 33), 'int')
    int_128706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 37), 'int')
    # Applying the binary operator '**' (line 272)
    result_pow_128707 = python_operator(stypy.reporting.localization.Localization(__file__, 272, 33), '**', int_128705, int_128706)
    
    # Processing the call keyword arguments (line 272)
    kwargs_128708 = {}
    # Getting the type of 'assert_equal' (line 272)
    assert_equal_128703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 272)
    assert_equal_call_result_128709 = invoke(stypy.reporting.localization.Localization(__file__, 272, 4), assert_equal_128703, *[num_nonzeros_5_128704, result_pow_128707], **kwargs_128708)
    
    
    # Assigning a Call to a Name (line 275):
    
    # Call to gaussian_filter(...): (line 275)
    # Processing the call arguments (line 275)
    # Getting the type of 'arr' (line 275)
    arr_128712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 29), 'arr', False)
    
    # Obtaining an instance of the builtin type 'list' (line 275)
    list_128713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 34), 'list')
    # Adding type elements to the builtin type 'list' instance (line 275)
    # Adding element type (line 275)
    float_128714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 35), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 34), list_128713, float_128714)
    # Adding element type (line 275)
    float_128715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 40), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 34), list_128713, float_128715)
    
    # Processing the call keyword arguments (line 275)
    float_128716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 55), 'float')
    keyword_128717 = float_128716
    kwargs_128718 = {'truncate': keyword_128717}
    # Getting the type of 'sndi' (line 275)
    sndi_128710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 8), 'sndi', False)
    # Obtaining the member 'gaussian_filter' of a type (line 275)
    gaussian_filter_128711 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 8), sndi_128710, 'gaussian_filter')
    # Calling gaussian_filter(args, kwargs) (line 275)
    gaussian_filter_call_result_128719 = invoke(stypy.reporting.localization.Localization(__file__, 275, 8), gaussian_filter_128711, *[arr_128712, list_128713], **kwargs_128718)
    
    # Assigning a type to the variable 'f' (line 275)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 4), 'f', gaussian_filter_call_result_128719)
    
    # Assigning a Compare to a Name (line 276):
    
    # Getting the type of 'f' (line 276)
    f_128720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 11), 'f')
    int_128721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 15), 'int')
    # Applying the binary operator '>' (line 276)
    result_gt_128722 = python_operator(stypy.reporting.localization.Localization(__file__, 276, 11), '>', f_128720, int_128721)
    
    # Assigning a type to the variable 'fpos' (line 276)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 4), 'fpos', result_gt_128722)
    
    # Assigning a Call to a Name (line 277):
    
    # Call to sum(...): (line 277)
    # Processing the call keyword arguments (line 277)
    kwargs_128730 = {}
    
    # Call to any(...): (line 277)
    # Processing the call keyword arguments (line 277)
    int_128725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 23), 'int')
    keyword_128726 = int_128725
    kwargs_128727 = {'axis': keyword_128726}
    # Getting the type of 'fpos' (line 277)
    fpos_128723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 9), 'fpos', False)
    # Obtaining the member 'any' of a type (line 277)
    any_128724 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 9), fpos_128723, 'any')
    # Calling any(args, kwargs) (line 277)
    any_call_result_128728 = invoke(stypy.reporting.localization.Localization(__file__, 277, 9), any_128724, *[], **kwargs_128727)
    
    # Obtaining the member 'sum' of a type (line 277)
    sum_128729 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 9), any_call_result_128728, 'sum')
    # Calling sum(args, kwargs) (line 277)
    sum_call_result_128731 = invoke(stypy.reporting.localization.Localization(__file__, 277, 9), sum_128729, *[], **kwargs_128730)
    
    # Assigning a type to the variable 'n0' (line 277)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 4), 'n0', sum_call_result_128731)
    
    # Call to assert_equal(...): (line 279)
    # Processing the call arguments (line 279)
    # Getting the type of 'n0' (line 279)
    n0_128733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 17), 'n0', False)
    int_128734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 21), 'int')
    # Processing the call keyword arguments (line 279)
    kwargs_128735 = {}
    # Getting the type of 'assert_equal' (line 279)
    assert_equal_128732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 279)
    assert_equal_call_result_128736 = invoke(stypy.reporting.localization.Localization(__file__, 279, 4), assert_equal_128732, *[n0_128733, int_128734], **kwargs_128735)
    
    
    # Assigning a Call to a Name (line 280):
    
    # Call to sum(...): (line 280)
    # Processing the call keyword arguments (line 280)
    kwargs_128744 = {}
    
    # Call to any(...): (line 280)
    # Processing the call keyword arguments (line 280)
    int_128739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 23), 'int')
    keyword_128740 = int_128739
    kwargs_128741 = {'axis': keyword_128740}
    # Getting the type of 'fpos' (line 280)
    fpos_128737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 9), 'fpos', False)
    # Obtaining the member 'any' of a type (line 280)
    any_128738 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 9), fpos_128737, 'any')
    # Calling any(args, kwargs) (line 280)
    any_call_result_128742 = invoke(stypy.reporting.localization.Localization(__file__, 280, 9), any_128738, *[], **kwargs_128741)
    
    # Obtaining the member 'sum' of a type (line 280)
    sum_128743 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 9), any_call_result_128742, 'sum')
    # Calling sum(args, kwargs) (line 280)
    sum_call_result_128745 = invoke(stypy.reporting.localization.Localization(__file__, 280, 9), sum_128743, *[], **kwargs_128744)
    
    # Assigning a type to the variable 'n1' (line 280)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 4), 'n1', sum_call_result_128745)
    
    # Call to assert_equal(...): (line 282)
    # Processing the call arguments (line 282)
    # Getting the type of 'n1' (line 282)
    n1_128747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 17), 'n1', False)
    int_128748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 21), 'int')
    # Processing the call keyword arguments (line 282)
    kwargs_128749 = {}
    # Getting the type of 'assert_equal' (line 282)
    assert_equal_128746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 282)
    assert_equal_call_result_128750 = invoke(stypy.reporting.localization.Localization(__file__, 282, 4), assert_equal_128746, *[n1_128747, int_128748], **kwargs_128749)
    
    
    # Assigning a Call to a Name (line 285):
    
    # Call to zeros(...): (line 285)
    # Processing the call arguments (line 285)
    int_128753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 17), 'int')
    # Processing the call keyword arguments (line 285)
    kwargs_128754 = {}
    # Getting the type of 'np' (line 285)
    np_128751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 8), 'np', False)
    # Obtaining the member 'zeros' of a type (line 285)
    zeros_128752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 8), np_128751, 'zeros')
    # Calling zeros(args, kwargs) (line 285)
    zeros_call_result_128755 = invoke(stypy.reporting.localization.Localization(__file__, 285, 8), zeros_128752, *[int_128753], **kwargs_128754)
    
    # Assigning a type to the variable 'x' (line 285)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 4), 'x', zeros_call_result_128755)
    
    # Assigning a Num to a Subscript (line 286):
    int_128756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 12), 'int')
    # Getting the type of 'x' (line 286)
    x_128757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 4), 'x')
    int_128758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 6), 'int')
    # Storing an element on a container (line 286)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 286, 4), x_128757, (int_128758, int_128756))
    
    # Assigning a Call to a Name (line 287):
    
    # Call to gaussian_filter1d(...): (line 287)
    # Processing the call arguments (line 287)
    # Getting the type of 'x' (line 287)
    x_128761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 31), 'x', False)
    # Processing the call keyword arguments (line 287)
    int_128762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 40), 'int')
    keyword_128763 = int_128762
    float_128764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 52), 'float')
    keyword_128765 = float_128764
    kwargs_128766 = {'sigma': keyword_128763, 'truncate': keyword_128765}
    # Getting the type of 'sndi' (line 287)
    sndi_128759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 8), 'sndi', False)
    # Obtaining the member 'gaussian_filter1d' of a type (line 287)
    gaussian_filter1d_128760 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 8), sndi_128759, 'gaussian_filter1d')
    # Calling gaussian_filter1d(args, kwargs) (line 287)
    gaussian_filter1d_call_result_128767 = invoke(stypy.reporting.localization.Localization(__file__, 287, 8), gaussian_filter1d_128760, *[x_128761], **kwargs_128766)
    
    # Assigning a type to the variable 'f' (line 287)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 4), 'f', gaussian_filter1d_call_result_128767)
    
    # Assigning a Call to a Name (line 288):
    
    # Call to sum(...): (line 288)
    # Processing the call keyword arguments (line 288)
    kwargs_128772 = {}
    
    # Getting the type of 'f' (line 288)
    f_128768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 9), 'f', False)
    int_128769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 13), 'int')
    # Applying the binary operator '>' (line 288)
    result_gt_128770 = python_operator(stypy.reporting.localization.Localization(__file__, 288, 9), '>', f_128768, int_128769)
    
    # Obtaining the member 'sum' of a type (line 288)
    sum_128771 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 9), result_gt_128770, 'sum')
    # Calling sum(args, kwargs) (line 288)
    sum_call_result_128773 = invoke(stypy.reporting.localization.Localization(__file__, 288, 9), sum_128771, *[], **kwargs_128772)
    
    # Assigning a type to the variable 'n' (line 288)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 4), 'n', sum_call_result_128773)
    
    # Call to assert_equal(...): (line 289)
    # Processing the call arguments (line 289)
    # Getting the type of 'n' (line 289)
    n_128775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 17), 'n', False)
    int_128776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 20), 'int')
    # Processing the call keyword arguments (line 289)
    kwargs_128777 = {}
    # Getting the type of 'assert_equal' (line 289)
    assert_equal_128774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 289)
    assert_equal_call_result_128778 = invoke(stypy.reporting.localization.Localization(__file__, 289, 4), assert_equal_128774, *[n_128775, int_128776], **kwargs_128777)
    
    
    # Assigning a Call to a Name (line 292):
    
    # Call to gaussian_laplace(...): (line 292)
    # Processing the call arguments (line 292)
    # Getting the type of 'x' (line 292)
    x_128781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 30), 'x', False)
    # Processing the call keyword arguments (line 292)
    int_128782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 39), 'int')
    keyword_128783 = int_128782
    float_128784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 51), 'float')
    keyword_128785 = float_128784
    kwargs_128786 = {'sigma': keyword_128783, 'truncate': keyword_128785}
    # Getting the type of 'sndi' (line 292)
    sndi_128779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'sndi', False)
    # Obtaining the member 'gaussian_laplace' of a type (line 292)
    gaussian_laplace_128780 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 8), sndi_128779, 'gaussian_laplace')
    # Calling gaussian_laplace(args, kwargs) (line 292)
    gaussian_laplace_call_result_128787 = invoke(stypy.reporting.localization.Localization(__file__, 292, 8), gaussian_laplace_128780, *[x_128781], **kwargs_128786)
    
    # Assigning a type to the variable 'y' (line 292)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 4), 'y', gaussian_laplace_call_result_128787)
    
    # Assigning a Subscript to a Name (line 293):
    
    # Obtaining the type of the subscript
    int_128788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 39), 'int')
    
    # Call to where(...): (line 293)
    # Processing the call arguments (line 293)
    
    # Getting the type of 'y' (line 293)
    y_128791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 31), 'y', False)
    int_128792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 36), 'int')
    # Applying the binary operator '!=' (line 293)
    result_ne_128793 = python_operator(stypy.reporting.localization.Localization(__file__, 293, 31), '!=', y_128791, int_128792)
    
    # Processing the call keyword arguments (line 293)
    kwargs_128794 = {}
    # Getting the type of 'np' (line 293)
    np_128789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 22), 'np', False)
    # Obtaining the member 'where' of a type (line 293)
    where_128790 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 22), np_128789, 'where')
    # Calling where(args, kwargs) (line 293)
    where_call_result_128795 = invoke(stypy.reporting.localization.Localization(__file__, 293, 22), where_128790, *[result_ne_128793], **kwargs_128794)
    
    # Obtaining the member '__getitem__' of a type (line 293)
    getitem___128796 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 22), where_call_result_128795, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 293)
    subscript_call_result_128797 = invoke(stypy.reporting.localization.Localization(__file__, 293, 22), getitem___128796, int_128788)
    
    # Assigning a type to the variable 'nonzero_indices' (line 293)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 4), 'nonzero_indices', subscript_call_result_128797)
    
    # Assigning a BinOp to a Name (line 294):
    
    # Call to ptp(...): (line 294)
    # Processing the call keyword arguments (line 294)
    kwargs_128800 = {}
    # Getting the type of 'nonzero_indices' (line 294)
    nonzero_indices_128798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 8), 'nonzero_indices', False)
    # Obtaining the member 'ptp' of a type (line 294)
    ptp_128799 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 8), nonzero_indices_128798, 'ptp')
    # Calling ptp(args, kwargs) (line 294)
    ptp_call_result_128801 = invoke(stypy.reporting.localization.Localization(__file__, 294, 8), ptp_128799, *[], **kwargs_128800)
    
    int_128802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 32), 'int')
    # Applying the binary operator '+' (line 294)
    result_add_128803 = python_operator(stypy.reporting.localization.Localization(__file__, 294, 8), '+', ptp_call_result_128801, int_128802)
    
    # Assigning a type to the variable 'n' (line 294)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 4), 'n', result_add_128803)
    
    # Call to assert_equal(...): (line 295)
    # Processing the call arguments (line 295)
    # Getting the type of 'n' (line 295)
    n_128805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 17), 'n', False)
    int_128806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 20), 'int')
    # Processing the call keyword arguments (line 295)
    kwargs_128807 = {}
    # Getting the type of 'assert_equal' (line 295)
    assert_equal_128804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 295)
    assert_equal_call_result_128808 = invoke(stypy.reporting.localization.Localization(__file__, 295, 4), assert_equal_128804, *[n_128805, int_128806], **kwargs_128807)
    
    
    # Assigning a Call to a Name (line 298):
    
    # Call to gaussian_gradient_magnitude(...): (line 298)
    # Processing the call arguments (line 298)
    # Getting the type of 'x' (line 298)
    x_128811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 41), 'x', False)
    # Processing the call keyword arguments (line 298)
    int_128812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 50), 'int')
    keyword_128813 = int_128812
    float_128814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 62), 'float')
    keyword_128815 = float_128814
    kwargs_128816 = {'sigma': keyword_128813, 'truncate': keyword_128815}
    # Getting the type of 'sndi' (line 298)
    sndi_128809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 8), 'sndi', False)
    # Obtaining the member 'gaussian_gradient_magnitude' of a type (line 298)
    gaussian_gradient_magnitude_128810 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 8), sndi_128809, 'gaussian_gradient_magnitude')
    # Calling gaussian_gradient_magnitude(args, kwargs) (line 298)
    gaussian_gradient_magnitude_call_result_128817 = invoke(stypy.reporting.localization.Localization(__file__, 298, 8), gaussian_gradient_magnitude_128810, *[x_128811], **kwargs_128816)
    
    # Assigning a type to the variable 'y' (line 298)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 4), 'y', gaussian_gradient_magnitude_call_result_128817)
    
    # Assigning a Subscript to a Name (line 299):
    
    # Obtaining the type of the subscript
    int_128818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 39), 'int')
    
    # Call to where(...): (line 299)
    # Processing the call arguments (line 299)
    
    # Getting the type of 'y' (line 299)
    y_128821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 31), 'y', False)
    int_128822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 36), 'int')
    # Applying the binary operator '!=' (line 299)
    result_ne_128823 = python_operator(stypy.reporting.localization.Localization(__file__, 299, 31), '!=', y_128821, int_128822)
    
    # Processing the call keyword arguments (line 299)
    kwargs_128824 = {}
    # Getting the type of 'np' (line 299)
    np_128819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 22), 'np', False)
    # Obtaining the member 'where' of a type (line 299)
    where_128820 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 22), np_128819, 'where')
    # Calling where(args, kwargs) (line 299)
    where_call_result_128825 = invoke(stypy.reporting.localization.Localization(__file__, 299, 22), where_128820, *[result_ne_128823], **kwargs_128824)
    
    # Obtaining the member '__getitem__' of a type (line 299)
    getitem___128826 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 22), where_call_result_128825, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 299)
    subscript_call_result_128827 = invoke(stypy.reporting.localization.Localization(__file__, 299, 22), getitem___128826, int_128818)
    
    # Assigning a type to the variable 'nonzero_indices' (line 299)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 4), 'nonzero_indices', subscript_call_result_128827)
    
    # Assigning a BinOp to a Name (line 300):
    
    # Call to ptp(...): (line 300)
    # Processing the call keyword arguments (line 300)
    kwargs_128830 = {}
    # Getting the type of 'nonzero_indices' (line 300)
    nonzero_indices_128828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 8), 'nonzero_indices', False)
    # Obtaining the member 'ptp' of a type (line 300)
    ptp_128829 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 8), nonzero_indices_128828, 'ptp')
    # Calling ptp(args, kwargs) (line 300)
    ptp_call_result_128831 = invoke(stypy.reporting.localization.Localization(__file__, 300, 8), ptp_128829, *[], **kwargs_128830)
    
    int_128832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 32), 'int')
    # Applying the binary operator '+' (line 300)
    result_add_128833 = python_operator(stypy.reporting.localization.Localization(__file__, 300, 8), '+', ptp_call_result_128831, int_128832)
    
    # Assigning a type to the variable 'n' (line 300)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 4), 'n', result_add_128833)
    
    # Call to assert_equal(...): (line 301)
    # Processing the call arguments (line 301)
    # Getting the type of 'n' (line 301)
    n_128835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 17), 'n', False)
    int_128836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 20), 'int')
    # Processing the call keyword arguments (line 301)
    kwargs_128837 = {}
    # Getting the type of 'assert_equal' (line 301)
    assert_equal_128834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 301)
    assert_equal_call_result_128838 = invoke(stypy.reporting.localization.Localization(__file__, 301, 4), assert_equal_128834, *[n_128835, int_128836], **kwargs_128837)
    
    
    # ################# End of 'test_gaussian_truncate(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_gaussian_truncate' in the type store
    # Getting the type of 'stypy_return_type' (line 263)
    stypy_return_type_128839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_128839)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_gaussian_truncate'
    return stypy_return_type_128839

# Assigning a type to the variable 'test_gaussian_truncate' (line 263)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 0), 'test_gaussian_truncate', test_gaussian_truncate)
# Declaration of the 'TestThreading' class

class TestThreading(object, ):

    @norecursion
    def check_func_thread(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'check_func_thread'
        module_type_store = module_type_store.open_function_context('check_func_thread', 305, 4, False)
        # Assigning a type to the variable 'self' (line 306)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestThreading.check_func_thread.__dict__.__setitem__('stypy_localization', localization)
        TestThreading.check_func_thread.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestThreading.check_func_thread.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestThreading.check_func_thread.__dict__.__setitem__('stypy_function_name', 'TestThreading.check_func_thread')
        TestThreading.check_func_thread.__dict__.__setitem__('stypy_param_names_list', ['n', 'fun', 'args', 'out'])
        TestThreading.check_func_thread.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestThreading.check_func_thread.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestThreading.check_func_thread.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestThreading.check_func_thread.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestThreading.check_func_thread.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestThreading.check_func_thread.__dict__.__setitem__('stypy_declared_arg_number', 5)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestThreading.check_func_thread', ['n', 'fun', 'args', 'out'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'check_func_thread', localization, ['n', 'fun', 'args', 'out'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'check_func_thread(...)' code ##################

        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 306, 8))
        
        # 'from threading import Thread' statement (line 306)
        try:
            from threading import Thread

        except:
            Thread = UndefinedType
        import_from_module(stypy.reporting.localization.Localization(__file__, 306, 8), 'threading', None, module_type_store, ['Thread'], [Thread])
        
        
        # Assigning a ListComp to a Name (line 307):
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to range(...): (line 307)
        # Processing the call arguments (line 307)
        # Getting the type of 'n' (line 307)
        n_128855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 89), 'n', False)
        # Processing the call keyword arguments (line 307)
        kwargs_128856 = {}
        # Getting the type of 'range' (line 307)
        range_128854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 83), 'range', False)
        # Calling range(args, kwargs) (line 307)
        range_call_result_128857 = invoke(stypy.reporting.localization.Localization(__file__, 307, 83), range_128854, *[n_128855], **kwargs_128856)
        
        comprehension_128858 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 307, 17), range_call_result_128857)
        # Assigning a type to the variable 'x' (line 307)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 17), 'x', comprehension_128858)
        
        # Call to Thread(...): (line 307)
        # Processing the call keyword arguments (line 307)
        # Getting the type of 'fun' (line 307)
        fun_128841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 31), 'fun', False)
        keyword_128842 = fun_128841
        # Getting the type of 'args' (line 307)
        args_128843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 41), 'args', False)
        keyword_128844 = args_128843
        
        # Obtaining an instance of the builtin type 'dict' (line 307)
        dict_128845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 54), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 307)
        # Adding element type (key, value) (line 307)
        str_128846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 55), 'str', 'output')
        
        # Obtaining the type of the subscript
        # Getting the type of 'x' (line 307)
        x_128847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 69), 'x', False)
        # Getting the type of 'out' (line 307)
        out_128848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 65), 'out', False)
        # Obtaining the member '__getitem__' of a type (line 307)
        getitem___128849 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 65), out_128848, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 307)
        subscript_call_result_128850 = invoke(stypy.reporting.localization.Localization(__file__, 307, 65), getitem___128849, x_128847)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 307, 54), dict_128845, (str_128846, subscript_call_result_128850))
        
        keyword_128851 = dict_128845
        kwargs_128852 = {'args': keyword_128844, 'target': keyword_128842, 'kwargs': keyword_128851}
        # Getting the type of 'Thread' (line 307)
        Thread_128840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 17), 'Thread', False)
        # Calling Thread(args, kwargs) (line 307)
        Thread_call_result_128853 = invoke(stypy.reporting.localization.Localization(__file__, 307, 17), Thread_128840, *[], **kwargs_128852)
        
        list_128859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 17), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 307, 17), list_128859, Thread_call_result_128853)
        # Assigning a type to the variable 'thrds' (line 307)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 8), 'thrds', list_128859)
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'thrds' (line 308)
        thrds_128864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 28), 'thrds')
        comprehension_128865 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 308, 9), thrds_128864)
        # Assigning a type to the variable 't' (line 308)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 308, 9), 't', comprehension_128865)
        
        # Call to start(...): (line 308)
        # Processing the call keyword arguments (line 308)
        kwargs_128862 = {}
        # Getting the type of 't' (line 308)
        t_128860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 9), 't', False)
        # Obtaining the member 'start' of a type (line 308)
        start_128861 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 9), t_128860, 'start')
        # Calling start(args, kwargs) (line 308)
        start_call_result_128863 = invoke(stypy.reporting.localization.Localization(__file__, 308, 9), start_128861, *[], **kwargs_128862)
        
        list_128866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 9), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 308, 9), list_128866, start_call_result_128863)
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'thrds' (line 309)
        thrds_128871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 27), 'thrds')
        comprehension_128872 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 309, 9), thrds_128871)
        # Assigning a type to the variable 't' (line 309)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 9), 't', comprehension_128872)
        
        # Call to join(...): (line 309)
        # Processing the call keyword arguments (line 309)
        kwargs_128869 = {}
        # Getting the type of 't' (line 309)
        t_128867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 9), 't', False)
        # Obtaining the member 'join' of a type (line 309)
        join_128868 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 9), t_128867, 'join')
        # Calling join(args, kwargs) (line 309)
        join_call_result_128870 = invoke(stypy.reporting.localization.Localization(__file__, 309, 9), join_128868, *[], **kwargs_128869)
        
        list_128873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 9), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 309, 9), list_128873, join_call_result_128870)
        
        # ################# End of 'check_func_thread(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'check_func_thread' in the type store
        # Getting the type of 'stypy_return_type' (line 305)
        stypy_return_type_128874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_128874)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'check_func_thread'
        return stypy_return_type_128874


    @norecursion
    def check_func_serial(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'check_func_serial'
        module_type_store = module_type_store.open_function_context('check_func_serial', 311, 4, False)
        # Assigning a type to the variable 'self' (line 312)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestThreading.check_func_serial.__dict__.__setitem__('stypy_localization', localization)
        TestThreading.check_func_serial.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestThreading.check_func_serial.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestThreading.check_func_serial.__dict__.__setitem__('stypy_function_name', 'TestThreading.check_func_serial')
        TestThreading.check_func_serial.__dict__.__setitem__('stypy_param_names_list', ['n', 'fun', 'args', 'out'])
        TestThreading.check_func_serial.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestThreading.check_func_serial.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestThreading.check_func_serial.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestThreading.check_func_serial.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestThreading.check_func_serial.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestThreading.check_func_serial.__dict__.__setitem__('stypy_declared_arg_number', 5)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestThreading.check_func_serial', ['n', 'fun', 'args', 'out'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'check_func_serial', localization, ['n', 'fun', 'args', 'out'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'check_func_serial(...)' code ##################

        
        
        # Call to range(...): (line 312)
        # Processing the call arguments (line 312)
        # Getting the type of 'n' (line 312)
        n_128876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 23), 'n', False)
        # Processing the call keyword arguments (line 312)
        kwargs_128877 = {}
        # Getting the type of 'range' (line 312)
        range_128875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 17), 'range', False)
        # Calling range(args, kwargs) (line 312)
        range_call_result_128878 = invoke(stypy.reporting.localization.Localization(__file__, 312, 17), range_128875, *[n_128876], **kwargs_128877)
        
        # Testing the type of a for loop iterable (line 312)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 312, 8), range_call_result_128878)
        # Getting the type of the for loop variable (line 312)
        for_loop_var_128879 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 312, 8), range_call_result_128878)
        # Assigning a type to the variable 'i' (line 312)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 8), 'i', for_loop_var_128879)
        # SSA begins for a for statement (line 312)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to fun(...): (line 313)
        # Getting the type of 'args' (line 313)
        args_128881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 17), 'args', False)
        # Processing the call keyword arguments (line 313)
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 313)
        i_128882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 34), 'i', False)
        # Getting the type of 'out' (line 313)
        out_128883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 30), 'out', False)
        # Obtaining the member '__getitem__' of a type (line 313)
        getitem___128884 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 30), out_128883, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 313)
        subscript_call_result_128885 = invoke(stypy.reporting.localization.Localization(__file__, 313, 30), getitem___128884, i_128882)
        
        keyword_128886 = subscript_call_result_128885
        kwargs_128887 = {'output': keyword_128886}
        # Getting the type of 'fun' (line 313)
        fun_128880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 12), 'fun', False)
        # Calling fun(args, kwargs) (line 313)
        fun_call_result_128888 = invoke(stypy.reporting.localization.Localization(__file__, 313, 12), fun_128880, *[args_128881], **kwargs_128887)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'check_func_serial(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'check_func_serial' in the type store
        # Getting the type of 'stypy_return_type' (line 311)
        stypy_return_type_128889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_128889)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'check_func_serial'
        return stypy_return_type_128889


    @norecursion
    def test_correlate1d(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_correlate1d'
        module_type_store = module_type_store.open_function_context('test_correlate1d', 315, 4, False)
        # Assigning a type to the variable 'self' (line 316)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 316, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestThreading.test_correlate1d.__dict__.__setitem__('stypy_localization', localization)
        TestThreading.test_correlate1d.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestThreading.test_correlate1d.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestThreading.test_correlate1d.__dict__.__setitem__('stypy_function_name', 'TestThreading.test_correlate1d')
        TestThreading.test_correlate1d.__dict__.__setitem__('stypy_param_names_list', [])
        TestThreading.test_correlate1d.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestThreading.test_correlate1d.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestThreading.test_correlate1d.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestThreading.test_correlate1d.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestThreading.test_correlate1d.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestThreading.test_correlate1d.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestThreading.test_correlate1d', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_correlate1d', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_correlate1d(...)' code ##################

        
        # Assigning a Call to a Name (line 316):
        
        # Call to randn(...): (line 316)
        # Processing the call arguments (line 316)
        int_128893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 28), 'int')
        # Processing the call keyword arguments (line 316)
        kwargs_128894 = {}
        # Getting the type of 'np' (line 316)
        np_128890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 12), 'np', False)
        # Obtaining the member 'random' of a type (line 316)
        random_128891 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 12), np_128890, 'random')
        # Obtaining the member 'randn' of a type (line 316)
        randn_128892 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 12), random_128891, 'randn')
        # Calling randn(args, kwargs) (line 316)
        randn_call_result_128895 = invoke(stypy.reporting.localization.Localization(__file__, 316, 12), randn_128892, *[int_128893], **kwargs_128894)
        
        # Assigning a type to the variable 'd' (line 316)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 316, 8), 'd', randn_call_result_128895)
        
        # Assigning a Call to a Name (line 317):
        
        # Call to empty(...): (line 317)
        # Processing the call arguments (line 317)
        
        # Obtaining an instance of the builtin type 'tuple' (line 317)
        tuple_128898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 317)
        # Adding element type (line 317)
        int_128899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 317, 23), tuple_128898, int_128899)
        # Adding element type (line 317)
        # Getting the type of 'd' (line 317)
        d_128900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 26), 'd', False)
        # Obtaining the member 'size' of a type (line 317)
        size_128901 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 26), d_128900, 'size')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 317, 23), tuple_128898, size_128901)
        
        # Processing the call keyword arguments (line 317)
        kwargs_128902 = {}
        # Getting the type of 'np' (line 317)
        np_128896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 13), 'np', False)
        # Obtaining the member 'empty' of a type (line 317)
        empty_128897 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 13), np_128896, 'empty')
        # Calling empty(args, kwargs) (line 317)
        empty_call_result_128903 = invoke(stypy.reporting.localization.Localization(__file__, 317, 13), empty_128897, *[tuple_128898], **kwargs_128902)
        
        # Assigning a type to the variable 'os' (line 317)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 317, 8), 'os', empty_call_result_128903)
        
        # Assigning a Call to a Name (line 318):
        
        # Call to empty_like(...): (line 318)
        # Processing the call arguments (line 318)
        # Getting the type of 'os' (line 318)
        os_128906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 27), 'os', False)
        # Processing the call keyword arguments (line 318)
        kwargs_128907 = {}
        # Getting the type of 'np' (line 318)
        np_128904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 13), 'np', False)
        # Obtaining the member 'empty_like' of a type (line 318)
        empty_like_128905 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 13), np_128904, 'empty_like')
        # Calling empty_like(args, kwargs) (line 318)
        empty_like_call_result_128908 = invoke(stypy.reporting.localization.Localization(__file__, 318, 13), empty_like_128905, *[os_128906], **kwargs_128907)
        
        # Assigning a type to the variable 'ot' (line 318)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 8), 'ot', empty_like_call_result_128908)
        
        # Call to check_func_serial(...): (line 319)
        # Processing the call arguments (line 319)
        int_128911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 31), 'int')
        # Getting the type of 'sndi' (line 319)
        sndi_128912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 34), 'sndi', False)
        # Obtaining the member 'correlate1d' of a type (line 319)
        correlate1d_128913 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 319, 34), sndi_128912, 'correlate1d')
        
        # Obtaining an instance of the builtin type 'tuple' (line 319)
        tuple_128914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 53), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 319)
        # Adding element type (line 319)
        # Getting the type of 'd' (line 319)
        d_128915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 53), 'd', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 319, 53), tuple_128914, d_128915)
        # Adding element type (line 319)
        
        # Call to arange(...): (line 319)
        # Processing the call arguments (line 319)
        int_128918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 66), 'int')
        # Processing the call keyword arguments (line 319)
        kwargs_128919 = {}
        # Getting the type of 'np' (line 319)
        np_128916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 56), 'np', False)
        # Obtaining the member 'arange' of a type (line 319)
        arange_128917 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 319, 56), np_128916, 'arange')
        # Calling arange(args, kwargs) (line 319)
        arange_call_result_128920 = invoke(stypy.reporting.localization.Localization(__file__, 319, 56), arange_128917, *[int_128918], **kwargs_128919)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 319, 53), tuple_128914, arange_call_result_128920)
        
        # Getting the type of 'os' (line 319)
        os_128921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 71), 'os', False)
        # Processing the call keyword arguments (line 319)
        kwargs_128922 = {}
        # Getting the type of 'self' (line 319)
        self_128909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 8), 'self', False)
        # Obtaining the member 'check_func_serial' of a type (line 319)
        check_func_serial_128910 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 319, 8), self_128909, 'check_func_serial')
        # Calling check_func_serial(args, kwargs) (line 319)
        check_func_serial_call_result_128923 = invoke(stypy.reporting.localization.Localization(__file__, 319, 8), check_func_serial_128910, *[int_128911, correlate1d_128913, tuple_128914, os_128921], **kwargs_128922)
        
        
        # Call to check_func_thread(...): (line 320)
        # Processing the call arguments (line 320)
        int_128926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 31), 'int')
        # Getting the type of 'sndi' (line 320)
        sndi_128927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 34), 'sndi', False)
        # Obtaining the member 'correlate1d' of a type (line 320)
        correlate1d_128928 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 34), sndi_128927, 'correlate1d')
        
        # Obtaining an instance of the builtin type 'tuple' (line 320)
        tuple_128929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 53), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 320)
        # Adding element type (line 320)
        # Getting the type of 'd' (line 320)
        d_128930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 53), 'd', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 320, 53), tuple_128929, d_128930)
        # Adding element type (line 320)
        
        # Call to arange(...): (line 320)
        # Processing the call arguments (line 320)
        int_128933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 66), 'int')
        # Processing the call keyword arguments (line 320)
        kwargs_128934 = {}
        # Getting the type of 'np' (line 320)
        np_128931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 56), 'np', False)
        # Obtaining the member 'arange' of a type (line 320)
        arange_128932 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 56), np_128931, 'arange')
        # Calling arange(args, kwargs) (line 320)
        arange_call_result_128935 = invoke(stypy.reporting.localization.Localization(__file__, 320, 56), arange_128932, *[int_128933], **kwargs_128934)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 320, 53), tuple_128929, arange_call_result_128935)
        
        # Getting the type of 'ot' (line 320)
        ot_128936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 71), 'ot', False)
        # Processing the call keyword arguments (line 320)
        kwargs_128937 = {}
        # Getting the type of 'self' (line 320)
        self_128924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 8), 'self', False)
        # Obtaining the member 'check_func_thread' of a type (line 320)
        check_func_thread_128925 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 8), self_128924, 'check_func_thread')
        # Calling check_func_thread(args, kwargs) (line 320)
        check_func_thread_call_result_128938 = invoke(stypy.reporting.localization.Localization(__file__, 320, 8), check_func_thread_128925, *[int_128926, correlate1d_128928, tuple_128929, ot_128936], **kwargs_128937)
        
        
        # Call to assert_array_equal(...): (line 321)
        # Processing the call arguments (line 321)
        # Getting the type of 'os' (line 321)
        os_128940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 27), 'os', False)
        # Getting the type of 'ot' (line 321)
        ot_128941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 31), 'ot', False)
        # Processing the call keyword arguments (line 321)
        kwargs_128942 = {}
        # Getting the type of 'assert_array_equal' (line 321)
        assert_array_equal_128939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 321)
        assert_array_equal_call_result_128943 = invoke(stypy.reporting.localization.Localization(__file__, 321, 8), assert_array_equal_128939, *[os_128940, ot_128941], **kwargs_128942)
        
        
        # ################# End of 'test_correlate1d(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_correlate1d' in the type store
        # Getting the type of 'stypy_return_type' (line 315)
        stypy_return_type_128944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_128944)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_correlate1d'
        return stypy_return_type_128944


    @norecursion
    def test_correlate(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_correlate'
        module_type_store = module_type_store.open_function_context('test_correlate', 323, 4, False)
        # Assigning a type to the variable 'self' (line 324)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestThreading.test_correlate.__dict__.__setitem__('stypy_localization', localization)
        TestThreading.test_correlate.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestThreading.test_correlate.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestThreading.test_correlate.__dict__.__setitem__('stypy_function_name', 'TestThreading.test_correlate')
        TestThreading.test_correlate.__dict__.__setitem__('stypy_param_names_list', [])
        TestThreading.test_correlate.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestThreading.test_correlate.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestThreading.test_correlate.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestThreading.test_correlate.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestThreading.test_correlate.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestThreading.test_correlate.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestThreading.test_correlate', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_correlate', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_correlate(...)' code ##################

        
        # Assigning a Call to a Name (line 324):
        
        # Call to randn(...): (line 324)
        # Processing the call arguments (line 324)
        int_128948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 28), 'int')
        int_128949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 33), 'int')
        # Processing the call keyword arguments (line 324)
        kwargs_128950 = {}
        # Getting the type of 'np' (line 324)
        np_128945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 12), 'np', False)
        # Obtaining the member 'random' of a type (line 324)
        random_128946 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 12), np_128945, 'random')
        # Obtaining the member 'randn' of a type (line 324)
        randn_128947 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 12), random_128946, 'randn')
        # Calling randn(args, kwargs) (line 324)
        randn_call_result_128951 = invoke(stypy.reporting.localization.Localization(__file__, 324, 12), randn_128947, *[int_128948, int_128949], **kwargs_128950)
        
        # Assigning a type to the variable 'd' (line 324)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 8), 'd', randn_call_result_128951)
        
        # Assigning a Call to a Name (line 325):
        
        # Call to randn(...): (line 325)
        # Processing the call arguments (line 325)
        int_128955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 28), 'int')
        int_128956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 32), 'int')
        # Processing the call keyword arguments (line 325)
        kwargs_128957 = {}
        # Getting the type of 'np' (line 325)
        np_128952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 12), 'np', False)
        # Obtaining the member 'random' of a type (line 325)
        random_128953 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 12), np_128952, 'random')
        # Obtaining the member 'randn' of a type (line 325)
        randn_128954 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 12), random_128953, 'randn')
        # Calling randn(args, kwargs) (line 325)
        randn_call_result_128958 = invoke(stypy.reporting.localization.Localization(__file__, 325, 12), randn_128954, *[int_128955, int_128956], **kwargs_128957)
        
        # Assigning a type to the variable 'k' (line 325)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 8), 'k', randn_call_result_128958)
        
        # Assigning a Call to a Name (line 326):
        
        # Call to empty(...): (line 326)
        # Processing the call arguments (line 326)
        
        # Obtaining an instance of the builtin type 'list' (line 326)
        list_128961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 326)
        # Adding element type (line 326)
        int_128962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 326, 22), list_128961, int_128962)
        
        
        # Call to list(...): (line 326)
        # Processing the call arguments (line 326)
        # Getting the type of 'd' (line 326)
        d_128964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 33), 'd', False)
        # Obtaining the member 'shape' of a type (line 326)
        shape_128965 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 33), d_128964, 'shape')
        # Processing the call keyword arguments (line 326)
        kwargs_128966 = {}
        # Getting the type of 'list' (line 326)
        list_128963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 28), 'list', False)
        # Calling list(args, kwargs) (line 326)
        list_call_result_128967 = invoke(stypy.reporting.localization.Localization(__file__, 326, 28), list_128963, *[shape_128965], **kwargs_128966)
        
        # Applying the binary operator '+' (line 326)
        result_add_128968 = python_operator(stypy.reporting.localization.Localization(__file__, 326, 22), '+', list_128961, list_call_result_128967)
        
        # Processing the call keyword arguments (line 326)
        kwargs_128969 = {}
        # Getting the type of 'np' (line 326)
        np_128959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 13), 'np', False)
        # Obtaining the member 'empty' of a type (line 326)
        empty_128960 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 13), np_128959, 'empty')
        # Calling empty(args, kwargs) (line 326)
        empty_call_result_128970 = invoke(stypy.reporting.localization.Localization(__file__, 326, 13), empty_128960, *[result_add_128968], **kwargs_128969)
        
        # Assigning a type to the variable 'os' (line 326)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 8), 'os', empty_call_result_128970)
        
        # Assigning a Call to a Name (line 327):
        
        # Call to empty_like(...): (line 327)
        # Processing the call arguments (line 327)
        # Getting the type of 'os' (line 327)
        os_128973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 27), 'os', False)
        # Processing the call keyword arguments (line 327)
        kwargs_128974 = {}
        # Getting the type of 'np' (line 327)
        np_128971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 13), 'np', False)
        # Obtaining the member 'empty_like' of a type (line 327)
        empty_like_128972 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 13), np_128971, 'empty_like')
        # Calling empty_like(args, kwargs) (line 327)
        empty_like_call_result_128975 = invoke(stypy.reporting.localization.Localization(__file__, 327, 13), empty_like_128972, *[os_128973], **kwargs_128974)
        
        # Assigning a type to the variable 'ot' (line 327)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 327, 8), 'ot', empty_like_call_result_128975)
        
        # Call to check_func_serial(...): (line 328)
        # Processing the call arguments (line 328)
        int_128978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 31), 'int')
        # Getting the type of 'sndi' (line 328)
        sndi_128979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 34), 'sndi', False)
        # Obtaining the member 'correlate' of a type (line 328)
        correlate_128980 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 328, 34), sndi_128979, 'correlate')
        
        # Obtaining an instance of the builtin type 'tuple' (line 328)
        tuple_128981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 51), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 328)
        # Adding element type (line 328)
        # Getting the type of 'd' (line 328)
        d_128982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 51), 'd', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 328, 51), tuple_128981, d_128982)
        # Adding element type (line 328)
        # Getting the type of 'k' (line 328)
        k_128983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 54), 'k', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 328, 51), tuple_128981, k_128983)
        
        # Getting the type of 'os' (line 328)
        os_128984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 58), 'os', False)
        # Processing the call keyword arguments (line 328)
        kwargs_128985 = {}
        # Getting the type of 'self' (line 328)
        self_128976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 8), 'self', False)
        # Obtaining the member 'check_func_serial' of a type (line 328)
        check_func_serial_128977 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 328, 8), self_128976, 'check_func_serial')
        # Calling check_func_serial(args, kwargs) (line 328)
        check_func_serial_call_result_128986 = invoke(stypy.reporting.localization.Localization(__file__, 328, 8), check_func_serial_128977, *[int_128978, correlate_128980, tuple_128981, os_128984], **kwargs_128985)
        
        
        # Call to check_func_thread(...): (line 329)
        # Processing the call arguments (line 329)
        int_128989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 31), 'int')
        # Getting the type of 'sndi' (line 329)
        sndi_128990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 34), 'sndi', False)
        # Obtaining the member 'correlate' of a type (line 329)
        correlate_128991 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 34), sndi_128990, 'correlate')
        
        # Obtaining an instance of the builtin type 'tuple' (line 329)
        tuple_128992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 51), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 329)
        # Adding element type (line 329)
        # Getting the type of 'd' (line 329)
        d_128993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 51), 'd', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 329, 51), tuple_128992, d_128993)
        # Adding element type (line 329)
        # Getting the type of 'k' (line 329)
        k_128994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 54), 'k', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 329, 51), tuple_128992, k_128994)
        
        # Getting the type of 'ot' (line 329)
        ot_128995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 58), 'ot', False)
        # Processing the call keyword arguments (line 329)
        kwargs_128996 = {}
        # Getting the type of 'self' (line 329)
        self_128987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 8), 'self', False)
        # Obtaining the member 'check_func_thread' of a type (line 329)
        check_func_thread_128988 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 8), self_128987, 'check_func_thread')
        # Calling check_func_thread(args, kwargs) (line 329)
        check_func_thread_call_result_128997 = invoke(stypy.reporting.localization.Localization(__file__, 329, 8), check_func_thread_128988, *[int_128989, correlate_128991, tuple_128992, ot_128995], **kwargs_128996)
        
        
        # Call to assert_array_equal(...): (line 330)
        # Processing the call arguments (line 330)
        # Getting the type of 'os' (line 330)
        os_128999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 27), 'os', False)
        # Getting the type of 'ot' (line 330)
        ot_129000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 31), 'ot', False)
        # Processing the call keyword arguments (line 330)
        kwargs_129001 = {}
        # Getting the type of 'assert_array_equal' (line 330)
        assert_array_equal_128998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 330)
        assert_array_equal_call_result_129002 = invoke(stypy.reporting.localization.Localization(__file__, 330, 8), assert_array_equal_128998, *[os_128999, ot_129000], **kwargs_129001)
        
        
        # ################# End of 'test_correlate(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_correlate' in the type store
        # Getting the type of 'stypy_return_type' (line 323)
        stypy_return_type_129003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_129003)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_correlate'
        return stypy_return_type_129003


    @norecursion
    def test_median_filter(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_median_filter'
        module_type_store = module_type_store.open_function_context('test_median_filter', 332, 4, False)
        # Assigning a type to the variable 'self' (line 333)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 333, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestThreading.test_median_filter.__dict__.__setitem__('stypy_localization', localization)
        TestThreading.test_median_filter.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestThreading.test_median_filter.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestThreading.test_median_filter.__dict__.__setitem__('stypy_function_name', 'TestThreading.test_median_filter')
        TestThreading.test_median_filter.__dict__.__setitem__('stypy_param_names_list', [])
        TestThreading.test_median_filter.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestThreading.test_median_filter.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestThreading.test_median_filter.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestThreading.test_median_filter.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestThreading.test_median_filter.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestThreading.test_median_filter.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestThreading.test_median_filter', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_median_filter', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_median_filter(...)' code ##################

        
        # Assigning a Call to a Name (line 333):
        
        # Call to randn(...): (line 333)
        # Processing the call arguments (line 333)
        int_129007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 28), 'int')
        int_129008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 33), 'int')
        # Processing the call keyword arguments (line 333)
        kwargs_129009 = {}
        # Getting the type of 'np' (line 333)
        np_129004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 12), 'np', False)
        # Obtaining the member 'random' of a type (line 333)
        random_129005 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 12), np_129004, 'random')
        # Obtaining the member 'randn' of a type (line 333)
        randn_129006 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 12), random_129005, 'randn')
        # Calling randn(args, kwargs) (line 333)
        randn_call_result_129010 = invoke(stypy.reporting.localization.Localization(__file__, 333, 12), randn_129006, *[int_129007, int_129008], **kwargs_129009)
        
        # Assigning a type to the variable 'd' (line 333)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 333, 8), 'd', randn_call_result_129010)
        
        # Assigning a Call to a Name (line 334):
        
        # Call to empty(...): (line 334)
        # Processing the call arguments (line 334)
        
        # Obtaining an instance of the builtin type 'list' (line 334)
        list_129013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 334)
        # Adding element type (line 334)
        int_129014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 334, 22), list_129013, int_129014)
        
        
        # Call to list(...): (line 334)
        # Processing the call arguments (line 334)
        # Getting the type of 'd' (line 334)
        d_129016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 33), 'd', False)
        # Obtaining the member 'shape' of a type (line 334)
        shape_129017 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 33), d_129016, 'shape')
        # Processing the call keyword arguments (line 334)
        kwargs_129018 = {}
        # Getting the type of 'list' (line 334)
        list_129015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 28), 'list', False)
        # Calling list(args, kwargs) (line 334)
        list_call_result_129019 = invoke(stypy.reporting.localization.Localization(__file__, 334, 28), list_129015, *[shape_129017], **kwargs_129018)
        
        # Applying the binary operator '+' (line 334)
        result_add_129020 = python_operator(stypy.reporting.localization.Localization(__file__, 334, 22), '+', list_129013, list_call_result_129019)
        
        # Processing the call keyword arguments (line 334)
        kwargs_129021 = {}
        # Getting the type of 'np' (line 334)
        np_129011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 13), 'np', False)
        # Obtaining the member 'empty' of a type (line 334)
        empty_129012 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 13), np_129011, 'empty')
        # Calling empty(args, kwargs) (line 334)
        empty_call_result_129022 = invoke(stypy.reporting.localization.Localization(__file__, 334, 13), empty_129012, *[result_add_129020], **kwargs_129021)
        
        # Assigning a type to the variable 'os' (line 334)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 8), 'os', empty_call_result_129022)
        
        # Assigning a Call to a Name (line 335):
        
        # Call to empty_like(...): (line 335)
        # Processing the call arguments (line 335)
        # Getting the type of 'os' (line 335)
        os_129025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 27), 'os', False)
        # Processing the call keyword arguments (line 335)
        kwargs_129026 = {}
        # Getting the type of 'np' (line 335)
        np_129023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 13), 'np', False)
        # Obtaining the member 'empty_like' of a type (line 335)
        empty_like_129024 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 13), np_129023, 'empty_like')
        # Calling empty_like(args, kwargs) (line 335)
        empty_like_call_result_129027 = invoke(stypy.reporting.localization.Localization(__file__, 335, 13), empty_like_129024, *[os_129025], **kwargs_129026)
        
        # Assigning a type to the variable 'ot' (line 335)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 8), 'ot', empty_like_call_result_129027)
        
        # Call to check_func_serial(...): (line 336)
        # Processing the call arguments (line 336)
        int_129030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 31), 'int')
        # Getting the type of 'sndi' (line 336)
        sndi_129031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 34), 'sndi', False)
        # Obtaining the member 'median_filter' of a type (line 336)
        median_filter_129032 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 34), sndi_129031, 'median_filter')
        
        # Obtaining an instance of the builtin type 'tuple' (line 336)
        tuple_129033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 55), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 336)
        # Adding element type (line 336)
        # Getting the type of 'd' (line 336)
        d_129034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 55), 'd', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 336, 55), tuple_129033, d_129034)
        # Adding element type (line 336)
        int_129035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 58), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 336, 55), tuple_129033, int_129035)
        
        # Getting the type of 'os' (line 336)
        os_129036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 62), 'os', False)
        # Processing the call keyword arguments (line 336)
        kwargs_129037 = {}
        # Getting the type of 'self' (line 336)
        self_129028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 8), 'self', False)
        # Obtaining the member 'check_func_serial' of a type (line 336)
        check_func_serial_129029 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 8), self_129028, 'check_func_serial')
        # Calling check_func_serial(args, kwargs) (line 336)
        check_func_serial_call_result_129038 = invoke(stypy.reporting.localization.Localization(__file__, 336, 8), check_func_serial_129029, *[int_129030, median_filter_129032, tuple_129033, os_129036], **kwargs_129037)
        
        
        # Call to check_func_thread(...): (line 337)
        # Processing the call arguments (line 337)
        int_129041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 31), 'int')
        # Getting the type of 'sndi' (line 337)
        sndi_129042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 34), 'sndi', False)
        # Obtaining the member 'median_filter' of a type (line 337)
        median_filter_129043 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 34), sndi_129042, 'median_filter')
        
        # Obtaining an instance of the builtin type 'tuple' (line 337)
        tuple_129044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 55), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 337)
        # Adding element type (line 337)
        # Getting the type of 'd' (line 337)
        d_129045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 55), 'd', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 337, 55), tuple_129044, d_129045)
        # Adding element type (line 337)
        int_129046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 58), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 337, 55), tuple_129044, int_129046)
        
        # Getting the type of 'ot' (line 337)
        ot_129047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 62), 'ot', False)
        # Processing the call keyword arguments (line 337)
        kwargs_129048 = {}
        # Getting the type of 'self' (line 337)
        self_129039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 8), 'self', False)
        # Obtaining the member 'check_func_thread' of a type (line 337)
        check_func_thread_129040 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 8), self_129039, 'check_func_thread')
        # Calling check_func_thread(args, kwargs) (line 337)
        check_func_thread_call_result_129049 = invoke(stypy.reporting.localization.Localization(__file__, 337, 8), check_func_thread_129040, *[int_129041, median_filter_129043, tuple_129044, ot_129047], **kwargs_129048)
        
        
        # Call to assert_array_equal(...): (line 338)
        # Processing the call arguments (line 338)
        # Getting the type of 'os' (line 338)
        os_129051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 27), 'os', False)
        # Getting the type of 'ot' (line 338)
        ot_129052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 31), 'ot', False)
        # Processing the call keyword arguments (line 338)
        kwargs_129053 = {}
        # Getting the type of 'assert_array_equal' (line 338)
        assert_array_equal_129050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 338)
        assert_array_equal_call_result_129054 = invoke(stypy.reporting.localization.Localization(__file__, 338, 8), assert_array_equal_129050, *[os_129051, ot_129052], **kwargs_129053)
        
        
        # ################# End of 'test_median_filter(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_median_filter' in the type store
        # Getting the type of 'stypy_return_type' (line 332)
        stypy_return_type_129055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_129055)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_median_filter'
        return stypy_return_type_129055


    @norecursion
    def test_uniform_filter1d(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_uniform_filter1d'
        module_type_store = module_type_store.open_function_context('test_uniform_filter1d', 340, 4, False)
        # Assigning a type to the variable 'self' (line 341)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestThreading.test_uniform_filter1d.__dict__.__setitem__('stypy_localization', localization)
        TestThreading.test_uniform_filter1d.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestThreading.test_uniform_filter1d.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestThreading.test_uniform_filter1d.__dict__.__setitem__('stypy_function_name', 'TestThreading.test_uniform_filter1d')
        TestThreading.test_uniform_filter1d.__dict__.__setitem__('stypy_param_names_list', [])
        TestThreading.test_uniform_filter1d.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestThreading.test_uniform_filter1d.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestThreading.test_uniform_filter1d.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestThreading.test_uniform_filter1d.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestThreading.test_uniform_filter1d.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestThreading.test_uniform_filter1d.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestThreading.test_uniform_filter1d', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_uniform_filter1d', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_uniform_filter1d(...)' code ##################

        
        # Assigning a Call to a Name (line 341):
        
        # Call to randn(...): (line 341)
        # Processing the call arguments (line 341)
        int_129059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 28), 'int')
        # Processing the call keyword arguments (line 341)
        kwargs_129060 = {}
        # Getting the type of 'np' (line 341)
        np_129056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 12), 'np', False)
        # Obtaining the member 'random' of a type (line 341)
        random_129057 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 12), np_129056, 'random')
        # Obtaining the member 'randn' of a type (line 341)
        randn_129058 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 12), random_129057, 'randn')
        # Calling randn(args, kwargs) (line 341)
        randn_call_result_129061 = invoke(stypy.reporting.localization.Localization(__file__, 341, 12), randn_129058, *[int_129059], **kwargs_129060)
        
        # Assigning a type to the variable 'd' (line 341)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 8), 'd', randn_call_result_129061)
        
        # Assigning a Call to a Name (line 342):
        
        # Call to empty(...): (line 342)
        # Processing the call arguments (line 342)
        
        # Obtaining an instance of the builtin type 'tuple' (line 342)
        tuple_129064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 342)
        # Adding element type (line 342)
        int_129065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 342, 23), tuple_129064, int_129065)
        # Adding element type (line 342)
        # Getting the type of 'd' (line 342)
        d_129066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 26), 'd', False)
        # Obtaining the member 'size' of a type (line 342)
        size_129067 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 26), d_129066, 'size')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 342, 23), tuple_129064, size_129067)
        
        # Processing the call keyword arguments (line 342)
        kwargs_129068 = {}
        # Getting the type of 'np' (line 342)
        np_129062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 13), 'np', False)
        # Obtaining the member 'empty' of a type (line 342)
        empty_129063 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 13), np_129062, 'empty')
        # Calling empty(args, kwargs) (line 342)
        empty_call_result_129069 = invoke(stypy.reporting.localization.Localization(__file__, 342, 13), empty_129063, *[tuple_129064], **kwargs_129068)
        
        # Assigning a type to the variable 'os' (line 342)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 8), 'os', empty_call_result_129069)
        
        # Assigning a Call to a Name (line 343):
        
        # Call to empty_like(...): (line 343)
        # Processing the call arguments (line 343)
        # Getting the type of 'os' (line 343)
        os_129072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 27), 'os', False)
        # Processing the call keyword arguments (line 343)
        kwargs_129073 = {}
        # Getting the type of 'np' (line 343)
        np_129070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 13), 'np', False)
        # Obtaining the member 'empty_like' of a type (line 343)
        empty_like_129071 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 13), np_129070, 'empty_like')
        # Calling empty_like(args, kwargs) (line 343)
        empty_like_call_result_129074 = invoke(stypy.reporting.localization.Localization(__file__, 343, 13), empty_like_129071, *[os_129072], **kwargs_129073)
        
        # Assigning a type to the variable 'ot' (line 343)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 8), 'ot', empty_like_call_result_129074)
        
        # Call to check_func_serial(...): (line 344)
        # Processing the call arguments (line 344)
        int_129077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 31), 'int')
        # Getting the type of 'sndi' (line 344)
        sndi_129078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 34), 'sndi', False)
        # Obtaining the member 'uniform_filter1d' of a type (line 344)
        uniform_filter1d_129079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 34), sndi_129078, 'uniform_filter1d')
        
        # Obtaining an instance of the builtin type 'tuple' (line 344)
        tuple_129080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 58), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 344)
        # Adding element type (line 344)
        # Getting the type of 'd' (line 344)
        d_129081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 58), 'd', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 344, 58), tuple_129080, d_129081)
        # Adding element type (line 344)
        int_129082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 61), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 344, 58), tuple_129080, int_129082)
        
        # Getting the type of 'os' (line 344)
        os_129083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 65), 'os', False)
        # Processing the call keyword arguments (line 344)
        kwargs_129084 = {}
        # Getting the type of 'self' (line 344)
        self_129075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 8), 'self', False)
        # Obtaining the member 'check_func_serial' of a type (line 344)
        check_func_serial_129076 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 8), self_129075, 'check_func_serial')
        # Calling check_func_serial(args, kwargs) (line 344)
        check_func_serial_call_result_129085 = invoke(stypy.reporting.localization.Localization(__file__, 344, 8), check_func_serial_129076, *[int_129077, uniform_filter1d_129079, tuple_129080, os_129083], **kwargs_129084)
        
        
        # Call to check_func_thread(...): (line 345)
        # Processing the call arguments (line 345)
        int_129088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 31), 'int')
        # Getting the type of 'sndi' (line 345)
        sndi_129089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 34), 'sndi', False)
        # Obtaining the member 'uniform_filter1d' of a type (line 345)
        uniform_filter1d_129090 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 345, 34), sndi_129089, 'uniform_filter1d')
        
        # Obtaining an instance of the builtin type 'tuple' (line 345)
        tuple_129091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 58), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 345)
        # Adding element type (line 345)
        # Getting the type of 'd' (line 345)
        d_129092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 58), 'd', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 345, 58), tuple_129091, d_129092)
        # Adding element type (line 345)
        int_129093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 61), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 345, 58), tuple_129091, int_129093)
        
        # Getting the type of 'ot' (line 345)
        ot_129094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 65), 'ot', False)
        # Processing the call keyword arguments (line 345)
        kwargs_129095 = {}
        # Getting the type of 'self' (line 345)
        self_129086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 8), 'self', False)
        # Obtaining the member 'check_func_thread' of a type (line 345)
        check_func_thread_129087 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 345, 8), self_129086, 'check_func_thread')
        # Calling check_func_thread(args, kwargs) (line 345)
        check_func_thread_call_result_129096 = invoke(stypy.reporting.localization.Localization(__file__, 345, 8), check_func_thread_129087, *[int_129088, uniform_filter1d_129090, tuple_129091, ot_129094], **kwargs_129095)
        
        
        # Call to assert_array_equal(...): (line 346)
        # Processing the call arguments (line 346)
        # Getting the type of 'os' (line 346)
        os_129098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 27), 'os', False)
        # Getting the type of 'ot' (line 346)
        ot_129099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 31), 'ot', False)
        # Processing the call keyword arguments (line 346)
        kwargs_129100 = {}
        # Getting the type of 'assert_array_equal' (line 346)
        assert_array_equal_129097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 346)
        assert_array_equal_call_result_129101 = invoke(stypy.reporting.localization.Localization(__file__, 346, 8), assert_array_equal_129097, *[os_129098, ot_129099], **kwargs_129100)
        
        
        # ################# End of 'test_uniform_filter1d(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_uniform_filter1d' in the type store
        # Getting the type of 'stypy_return_type' (line 340)
        stypy_return_type_129102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_129102)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_uniform_filter1d'
        return stypy_return_type_129102


    @norecursion
    def test_minmax_filter(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_minmax_filter'
        module_type_store = module_type_store.open_function_context('test_minmax_filter', 348, 4, False)
        # Assigning a type to the variable 'self' (line 349)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestThreading.test_minmax_filter.__dict__.__setitem__('stypy_localization', localization)
        TestThreading.test_minmax_filter.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestThreading.test_minmax_filter.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestThreading.test_minmax_filter.__dict__.__setitem__('stypy_function_name', 'TestThreading.test_minmax_filter')
        TestThreading.test_minmax_filter.__dict__.__setitem__('stypy_param_names_list', [])
        TestThreading.test_minmax_filter.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestThreading.test_minmax_filter.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestThreading.test_minmax_filter.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestThreading.test_minmax_filter.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestThreading.test_minmax_filter.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestThreading.test_minmax_filter.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestThreading.test_minmax_filter', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_minmax_filter', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_minmax_filter(...)' code ##################

        
        # Assigning a Call to a Name (line 349):
        
        # Call to randn(...): (line 349)
        # Processing the call arguments (line 349)
        int_129106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 28), 'int')
        int_129107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 33), 'int')
        # Processing the call keyword arguments (line 349)
        kwargs_129108 = {}
        # Getting the type of 'np' (line 349)
        np_129103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 12), 'np', False)
        # Obtaining the member 'random' of a type (line 349)
        random_129104 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 12), np_129103, 'random')
        # Obtaining the member 'randn' of a type (line 349)
        randn_129105 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 12), random_129104, 'randn')
        # Calling randn(args, kwargs) (line 349)
        randn_call_result_129109 = invoke(stypy.reporting.localization.Localization(__file__, 349, 12), randn_129105, *[int_129106, int_129107], **kwargs_129108)
        
        # Assigning a type to the variable 'd' (line 349)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 8), 'd', randn_call_result_129109)
        
        # Assigning a Call to a Name (line 350):
        
        # Call to empty(...): (line 350)
        # Processing the call arguments (line 350)
        
        # Obtaining an instance of the builtin type 'list' (line 350)
        list_129112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 350)
        # Adding element type (line 350)
        int_129113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 350, 22), list_129112, int_129113)
        
        
        # Call to list(...): (line 350)
        # Processing the call arguments (line 350)
        # Getting the type of 'd' (line 350)
        d_129115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 33), 'd', False)
        # Obtaining the member 'shape' of a type (line 350)
        shape_129116 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 33), d_129115, 'shape')
        # Processing the call keyword arguments (line 350)
        kwargs_129117 = {}
        # Getting the type of 'list' (line 350)
        list_129114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 28), 'list', False)
        # Calling list(args, kwargs) (line 350)
        list_call_result_129118 = invoke(stypy.reporting.localization.Localization(__file__, 350, 28), list_129114, *[shape_129116], **kwargs_129117)
        
        # Applying the binary operator '+' (line 350)
        result_add_129119 = python_operator(stypy.reporting.localization.Localization(__file__, 350, 22), '+', list_129112, list_call_result_129118)
        
        # Processing the call keyword arguments (line 350)
        kwargs_129120 = {}
        # Getting the type of 'np' (line 350)
        np_129110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 13), 'np', False)
        # Obtaining the member 'empty' of a type (line 350)
        empty_129111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 13), np_129110, 'empty')
        # Calling empty(args, kwargs) (line 350)
        empty_call_result_129121 = invoke(stypy.reporting.localization.Localization(__file__, 350, 13), empty_129111, *[result_add_129119], **kwargs_129120)
        
        # Assigning a type to the variable 'os' (line 350)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 8), 'os', empty_call_result_129121)
        
        # Assigning a Call to a Name (line 351):
        
        # Call to empty_like(...): (line 351)
        # Processing the call arguments (line 351)
        # Getting the type of 'os' (line 351)
        os_129124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 27), 'os', False)
        # Processing the call keyword arguments (line 351)
        kwargs_129125 = {}
        # Getting the type of 'np' (line 351)
        np_129122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 13), 'np', False)
        # Obtaining the member 'empty_like' of a type (line 351)
        empty_like_129123 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 351, 13), np_129122, 'empty_like')
        # Calling empty_like(args, kwargs) (line 351)
        empty_like_call_result_129126 = invoke(stypy.reporting.localization.Localization(__file__, 351, 13), empty_like_129123, *[os_129124], **kwargs_129125)
        
        # Assigning a type to the variable 'ot' (line 351)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 351, 8), 'ot', empty_like_call_result_129126)
        
        # Call to check_func_serial(...): (line 352)
        # Processing the call arguments (line 352)
        int_129129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 31), 'int')
        # Getting the type of 'sndi' (line 352)
        sndi_129130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 34), 'sndi', False)
        # Obtaining the member 'maximum_filter' of a type (line 352)
        maximum_filter_129131 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 352, 34), sndi_129130, 'maximum_filter')
        
        # Obtaining an instance of the builtin type 'tuple' (line 352)
        tuple_129132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 56), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 352)
        # Adding element type (line 352)
        # Getting the type of 'd' (line 352)
        d_129133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 56), 'd', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 352, 56), tuple_129132, d_129133)
        # Adding element type (line 352)
        int_129134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 59), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 352, 56), tuple_129132, int_129134)
        
        # Getting the type of 'os' (line 352)
        os_129135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 63), 'os', False)
        # Processing the call keyword arguments (line 352)
        kwargs_129136 = {}
        # Getting the type of 'self' (line 352)
        self_129127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 8), 'self', False)
        # Obtaining the member 'check_func_serial' of a type (line 352)
        check_func_serial_129128 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 352, 8), self_129127, 'check_func_serial')
        # Calling check_func_serial(args, kwargs) (line 352)
        check_func_serial_call_result_129137 = invoke(stypy.reporting.localization.Localization(__file__, 352, 8), check_func_serial_129128, *[int_129129, maximum_filter_129131, tuple_129132, os_129135], **kwargs_129136)
        
        
        # Call to check_func_thread(...): (line 353)
        # Processing the call arguments (line 353)
        int_129140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, 31), 'int')
        # Getting the type of 'sndi' (line 353)
        sndi_129141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 34), 'sndi', False)
        # Obtaining the member 'maximum_filter' of a type (line 353)
        maximum_filter_129142 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 353, 34), sndi_129141, 'maximum_filter')
        
        # Obtaining an instance of the builtin type 'tuple' (line 353)
        tuple_129143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, 56), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 353)
        # Adding element type (line 353)
        # Getting the type of 'd' (line 353)
        d_129144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 56), 'd', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 353, 56), tuple_129143, d_129144)
        # Adding element type (line 353)
        int_129145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, 59), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 353, 56), tuple_129143, int_129145)
        
        # Getting the type of 'ot' (line 353)
        ot_129146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 63), 'ot', False)
        # Processing the call keyword arguments (line 353)
        kwargs_129147 = {}
        # Getting the type of 'self' (line 353)
        self_129138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 8), 'self', False)
        # Obtaining the member 'check_func_thread' of a type (line 353)
        check_func_thread_129139 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 353, 8), self_129138, 'check_func_thread')
        # Calling check_func_thread(args, kwargs) (line 353)
        check_func_thread_call_result_129148 = invoke(stypy.reporting.localization.Localization(__file__, 353, 8), check_func_thread_129139, *[int_129140, maximum_filter_129142, tuple_129143, ot_129146], **kwargs_129147)
        
        
        # Call to assert_array_equal(...): (line 354)
        # Processing the call arguments (line 354)
        # Getting the type of 'os' (line 354)
        os_129150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 27), 'os', False)
        # Getting the type of 'ot' (line 354)
        ot_129151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 31), 'ot', False)
        # Processing the call keyword arguments (line 354)
        kwargs_129152 = {}
        # Getting the type of 'assert_array_equal' (line 354)
        assert_array_equal_129149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 354)
        assert_array_equal_call_result_129153 = invoke(stypy.reporting.localization.Localization(__file__, 354, 8), assert_array_equal_129149, *[os_129150, ot_129151], **kwargs_129152)
        
        
        # Call to check_func_serial(...): (line 355)
        # Processing the call arguments (line 355)
        int_129156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 31), 'int')
        # Getting the type of 'sndi' (line 355)
        sndi_129157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 34), 'sndi', False)
        # Obtaining the member 'minimum_filter' of a type (line 355)
        minimum_filter_129158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 355, 34), sndi_129157, 'minimum_filter')
        
        # Obtaining an instance of the builtin type 'tuple' (line 355)
        tuple_129159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 56), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 355)
        # Adding element type (line 355)
        # Getting the type of 'd' (line 355)
        d_129160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 56), 'd', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 355, 56), tuple_129159, d_129160)
        # Adding element type (line 355)
        int_129161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 59), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 355, 56), tuple_129159, int_129161)
        
        # Getting the type of 'os' (line 355)
        os_129162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 63), 'os', False)
        # Processing the call keyword arguments (line 355)
        kwargs_129163 = {}
        # Getting the type of 'self' (line 355)
        self_129154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 8), 'self', False)
        # Obtaining the member 'check_func_serial' of a type (line 355)
        check_func_serial_129155 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 355, 8), self_129154, 'check_func_serial')
        # Calling check_func_serial(args, kwargs) (line 355)
        check_func_serial_call_result_129164 = invoke(stypy.reporting.localization.Localization(__file__, 355, 8), check_func_serial_129155, *[int_129156, minimum_filter_129158, tuple_129159, os_129162], **kwargs_129163)
        
        
        # Call to check_func_thread(...): (line 356)
        # Processing the call arguments (line 356)
        int_129167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 356, 31), 'int')
        # Getting the type of 'sndi' (line 356)
        sndi_129168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 34), 'sndi', False)
        # Obtaining the member 'minimum_filter' of a type (line 356)
        minimum_filter_129169 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 34), sndi_129168, 'minimum_filter')
        
        # Obtaining an instance of the builtin type 'tuple' (line 356)
        tuple_129170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 356, 56), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 356)
        # Adding element type (line 356)
        # Getting the type of 'd' (line 356)
        d_129171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 56), 'd', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 356, 56), tuple_129170, d_129171)
        # Adding element type (line 356)
        int_129172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 356, 59), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 356, 56), tuple_129170, int_129172)
        
        # Getting the type of 'ot' (line 356)
        ot_129173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 63), 'ot', False)
        # Processing the call keyword arguments (line 356)
        kwargs_129174 = {}
        # Getting the type of 'self' (line 356)
        self_129165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 8), 'self', False)
        # Obtaining the member 'check_func_thread' of a type (line 356)
        check_func_thread_129166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 8), self_129165, 'check_func_thread')
        # Calling check_func_thread(args, kwargs) (line 356)
        check_func_thread_call_result_129175 = invoke(stypy.reporting.localization.Localization(__file__, 356, 8), check_func_thread_129166, *[int_129167, minimum_filter_129169, tuple_129170, ot_129173], **kwargs_129174)
        
        
        # Call to assert_array_equal(...): (line 357)
        # Processing the call arguments (line 357)
        # Getting the type of 'os' (line 357)
        os_129177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 27), 'os', False)
        # Getting the type of 'ot' (line 357)
        ot_129178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 31), 'ot', False)
        # Processing the call keyword arguments (line 357)
        kwargs_129179 = {}
        # Getting the type of 'assert_array_equal' (line 357)
        assert_array_equal_129176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 357)
        assert_array_equal_call_result_129180 = invoke(stypy.reporting.localization.Localization(__file__, 357, 8), assert_array_equal_129176, *[os_129177, ot_129178], **kwargs_129179)
        
        
        # ################# End of 'test_minmax_filter(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_minmax_filter' in the type store
        # Getting the type of 'stypy_return_type' (line 348)
        stypy_return_type_129181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_129181)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_minmax_filter'
        return stypy_return_type_129181


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 304, 0, False)
        # Assigning a type to the variable 'self' (line 305)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 305, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestThreading.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestThreading' (line 304)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 0), 'TestThreading', TestThreading)

@norecursion
def test_minmaximum_filter1d(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_minmaximum_filter1d'
    module_type_store = module_type_store.open_function_context('test_minmaximum_filter1d', 360, 0, False)
    
    # Passed parameters checking function
    test_minmaximum_filter1d.stypy_localization = localization
    test_minmaximum_filter1d.stypy_type_of_self = None
    test_minmaximum_filter1d.stypy_type_store = module_type_store
    test_minmaximum_filter1d.stypy_function_name = 'test_minmaximum_filter1d'
    test_minmaximum_filter1d.stypy_param_names_list = []
    test_minmaximum_filter1d.stypy_varargs_param_name = None
    test_minmaximum_filter1d.stypy_kwargs_param_name = None
    test_minmaximum_filter1d.stypy_call_defaults = defaults
    test_minmaximum_filter1d.stypy_call_varargs = varargs
    test_minmaximum_filter1d.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_minmaximum_filter1d', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_minmaximum_filter1d', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_minmaximum_filter1d(...)' code ##################

    
    # Assigning a Call to a Name (line 362):
    
    # Call to arange(...): (line 362)
    # Processing the call arguments (line 362)
    int_129184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 20), 'int')
    # Processing the call keyword arguments (line 362)
    kwargs_129185 = {}
    # Getting the type of 'np' (line 362)
    np_129182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 10), 'np', False)
    # Obtaining the member 'arange' of a type (line 362)
    arange_129183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 10), np_129182, 'arange')
    # Calling arange(args, kwargs) (line 362)
    arange_call_result_129186 = invoke(stypy.reporting.localization.Localization(__file__, 362, 10), arange_129183, *[int_129184], **kwargs_129185)
    
    # Assigning a type to the variable 'in_' (line 362)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 4), 'in_', arange_call_result_129186)
    
    # Assigning a Call to a Name (line 363):
    
    # Call to minimum_filter1d(...): (line 363)
    # Processing the call arguments (line 363)
    # Getting the type of 'in_' (line 363)
    in__129189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 32), 'in_', False)
    int_129190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, 37), 'int')
    # Processing the call keyword arguments (line 363)
    kwargs_129191 = {}
    # Getting the type of 'sndi' (line 363)
    sndi_129187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 10), 'sndi', False)
    # Obtaining the member 'minimum_filter1d' of a type (line 363)
    minimum_filter1d_129188 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 10), sndi_129187, 'minimum_filter1d')
    # Calling minimum_filter1d(args, kwargs) (line 363)
    minimum_filter1d_call_result_129192 = invoke(stypy.reporting.localization.Localization(__file__, 363, 10), minimum_filter1d_129188, *[in__129189, int_129190], **kwargs_129191)
    
    # Assigning a type to the variable 'out' (line 363)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 4), 'out', minimum_filter1d_call_result_129192)
    
    # Call to assert_equal(...): (line 364)
    # Processing the call arguments (line 364)
    # Getting the type of 'in_' (line 364)
    in__129194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 17), 'in_', False)
    # Getting the type of 'out' (line 364)
    out_129195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 22), 'out', False)
    # Processing the call keyword arguments (line 364)
    kwargs_129196 = {}
    # Getting the type of 'assert_equal' (line 364)
    assert_equal_129193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 364)
    assert_equal_call_result_129197 = invoke(stypy.reporting.localization.Localization(__file__, 364, 4), assert_equal_129193, *[in__129194, out_129195], **kwargs_129196)
    
    
    # Assigning a Call to a Name (line 365):
    
    # Call to maximum_filter1d(...): (line 365)
    # Processing the call arguments (line 365)
    # Getting the type of 'in_' (line 365)
    in__129200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 32), 'in_', False)
    int_129201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 365, 37), 'int')
    # Processing the call keyword arguments (line 365)
    kwargs_129202 = {}
    # Getting the type of 'sndi' (line 365)
    sndi_129198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 10), 'sndi', False)
    # Obtaining the member 'maximum_filter1d' of a type (line 365)
    maximum_filter1d_129199 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 365, 10), sndi_129198, 'maximum_filter1d')
    # Calling maximum_filter1d(args, kwargs) (line 365)
    maximum_filter1d_call_result_129203 = invoke(stypy.reporting.localization.Localization(__file__, 365, 10), maximum_filter1d_129199, *[in__129200, int_129201], **kwargs_129202)
    
    # Assigning a type to the variable 'out' (line 365)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 365, 4), 'out', maximum_filter1d_call_result_129203)
    
    # Call to assert_equal(...): (line 366)
    # Processing the call arguments (line 366)
    # Getting the type of 'in_' (line 366)
    in__129205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 17), 'in_', False)
    # Getting the type of 'out' (line 366)
    out_129206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 22), 'out', False)
    # Processing the call keyword arguments (line 366)
    kwargs_129207 = {}
    # Getting the type of 'assert_equal' (line 366)
    assert_equal_129204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 366)
    assert_equal_call_result_129208 = invoke(stypy.reporting.localization.Localization(__file__, 366, 4), assert_equal_129204, *[in__129205, out_129206], **kwargs_129207)
    
    
    # Assigning a Call to a Name (line 368):
    
    # Call to minimum_filter1d(...): (line 368)
    # Processing the call arguments (line 368)
    # Getting the type of 'in_' (line 368)
    in__129211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 32), 'in_', False)
    int_129212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, 37), 'int')
    # Processing the call keyword arguments (line 368)
    str_129213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, 45), 'str', 'reflect')
    keyword_129214 = str_129213
    kwargs_129215 = {'mode': keyword_129214}
    # Getting the type of 'sndi' (line 368)
    sndi_129209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 10), 'sndi', False)
    # Obtaining the member 'minimum_filter1d' of a type (line 368)
    minimum_filter1d_129210 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 10), sndi_129209, 'minimum_filter1d')
    # Calling minimum_filter1d(args, kwargs) (line 368)
    minimum_filter1d_call_result_129216 = invoke(stypy.reporting.localization.Localization(__file__, 368, 10), minimum_filter1d_129210, *[in__129211, int_129212], **kwargs_129215)
    
    # Assigning a type to the variable 'out' (line 368)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 4), 'out', minimum_filter1d_call_result_129216)
    
    # Call to assert_equal(...): (line 369)
    # Processing the call arguments (line 369)
    
    # Obtaining an instance of the builtin type 'list' (line 369)
    list_129218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 369, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 369)
    # Adding element type (line 369)
    int_129219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 369, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 369, 17), list_129218, int_129219)
    # Adding element type (line 369)
    int_129220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 369, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 369, 17), list_129218, int_129220)
    # Adding element type (line 369)
    int_129221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 369, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 369, 17), list_129218, int_129221)
    # Adding element type (line 369)
    int_129222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 369, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 369, 17), list_129218, int_129222)
    # Adding element type (line 369)
    int_129223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 369, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 369, 17), list_129218, int_129223)
    # Adding element type (line 369)
    int_129224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 369, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 369, 17), list_129218, int_129224)
    # Adding element type (line 369)
    int_129225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 369, 36), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 369, 17), list_129218, int_129225)
    # Adding element type (line 369)
    int_129226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 369, 39), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 369, 17), list_129218, int_129226)
    # Adding element type (line 369)
    int_129227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 369, 42), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 369, 17), list_129218, int_129227)
    # Adding element type (line 369)
    int_129228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 369, 45), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 369, 17), list_129218, int_129228)
    
    # Getting the type of 'out' (line 369)
    out_129229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 49), 'out', False)
    # Processing the call keyword arguments (line 369)
    kwargs_129230 = {}
    # Getting the type of 'assert_equal' (line 369)
    assert_equal_129217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 369)
    assert_equal_call_result_129231 = invoke(stypy.reporting.localization.Localization(__file__, 369, 4), assert_equal_129217, *[list_129218, out_129229], **kwargs_129230)
    
    
    # Assigning a Call to a Name (line 370):
    
    # Call to maximum_filter1d(...): (line 370)
    # Processing the call arguments (line 370)
    # Getting the type of 'in_' (line 370)
    in__129234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 32), 'in_', False)
    int_129235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 37), 'int')
    # Processing the call keyword arguments (line 370)
    str_129236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 45), 'str', 'reflect')
    keyword_129237 = str_129236
    kwargs_129238 = {'mode': keyword_129237}
    # Getting the type of 'sndi' (line 370)
    sndi_129232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 10), 'sndi', False)
    # Obtaining the member 'maximum_filter1d' of a type (line 370)
    maximum_filter1d_129233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 10), sndi_129232, 'maximum_filter1d')
    # Calling maximum_filter1d(args, kwargs) (line 370)
    maximum_filter1d_call_result_129239 = invoke(stypy.reporting.localization.Localization(__file__, 370, 10), maximum_filter1d_129233, *[in__129234, int_129235], **kwargs_129238)
    
    # Assigning a type to the variable 'out' (line 370)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 4), 'out', maximum_filter1d_call_result_129239)
    
    # Call to assert_equal(...): (line 371)
    # Processing the call arguments (line 371)
    
    # Obtaining an instance of the builtin type 'list' (line 371)
    list_129241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 371, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 371)
    # Adding element type (line 371)
    int_129242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 371, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 371, 17), list_129241, int_129242)
    # Adding element type (line 371)
    int_129243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 371, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 371, 17), list_129241, int_129243)
    # Adding element type (line 371)
    int_129244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 371, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 371, 17), list_129241, int_129244)
    # Adding element type (line 371)
    int_129245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 371, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 371, 17), list_129241, int_129245)
    # Adding element type (line 371)
    int_129246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 371, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 371, 17), list_129241, int_129246)
    # Adding element type (line 371)
    int_129247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 371, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 371, 17), list_129241, int_129247)
    # Adding element type (line 371)
    int_129248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 371, 36), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 371, 17), list_129241, int_129248)
    # Adding element type (line 371)
    int_129249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 371, 39), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 371, 17), list_129241, int_129249)
    # Adding element type (line 371)
    int_129250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 371, 42), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 371, 17), list_129241, int_129250)
    # Adding element type (line 371)
    int_129251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 371, 45), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 371, 17), list_129241, int_129251)
    
    # Getting the type of 'out' (line 371)
    out_129252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 49), 'out', False)
    # Processing the call keyword arguments (line 371)
    kwargs_129253 = {}
    # Getting the type of 'assert_equal' (line 371)
    assert_equal_129240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 371)
    assert_equal_call_result_129254 = invoke(stypy.reporting.localization.Localization(__file__, 371, 4), assert_equal_129240, *[list_129241, out_129252], **kwargs_129253)
    
    
    # Assigning a Call to a Name (line 373):
    
    # Call to minimum_filter1d(...): (line 373)
    # Processing the call arguments (line 373)
    # Getting the type of 'in_' (line 373)
    in__129257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 32), 'in_', False)
    int_129258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 37), 'int')
    # Processing the call keyword arguments (line 373)
    str_129259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 45), 'str', 'constant')
    keyword_129260 = str_129259
    int_129261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 62), 'int')
    keyword_129262 = int_129261
    kwargs_129263 = {'cval': keyword_129262, 'mode': keyword_129260}
    # Getting the type of 'sndi' (line 373)
    sndi_129255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 10), 'sndi', False)
    # Obtaining the member 'minimum_filter1d' of a type (line 373)
    minimum_filter1d_129256 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 10), sndi_129255, 'minimum_filter1d')
    # Calling minimum_filter1d(args, kwargs) (line 373)
    minimum_filter1d_call_result_129264 = invoke(stypy.reporting.localization.Localization(__file__, 373, 10), minimum_filter1d_129256, *[in__129257, int_129258], **kwargs_129263)
    
    # Assigning a type to the variable 'out' (line 373)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 373, 4), 'out', minimum_filter1d_call_result_129264)
    
    # Call to assert_equal(...): (line 374)
    # Processing the call arguments (line 374)
    
    # Obtaining an instance of the builtin type 'list' (line 374)
    list_129266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 374)
    # Adding element type (line 374)
    int_129267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 374, 17), list_129266, int_129267)
    # Adding element type (line 374)
    int_129268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 374, 17), list_129266, int_129268)
    # Adding element type (line 374)
    int_129269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 26), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 374, 17), list_129266, int_129269)
    # Adding element type (line 374)
    int_129270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 29), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 374, 17), list_129266, int_129270)
    # Adding element type (line 374)
    int_129271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 32), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 374, 17), list_129266, int_129271)
    # Adding element type (line 374)
    int_129272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 35), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 374, 17), list_129266, int_129272)
    # Adding element type (line 374)
    int_129273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 38), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 374, 17), list_129266, int_129273)
    # Adding element type (line 374)
    int_129274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 41), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 374, 17), list_129266, int_129274)
    # Adding element type (line 374)
    int_129275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 44), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 374, 17), list_129266, int_129275)
    # Adding element type (line 374)
    int_129276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 48), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 374, 17), list_129266, int_129276)
    
    # Getting the type of 'out' (line 374)
    out_129277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 53), 'out', False)
    # Processing the call keyword arguments (line 374)
    kwargs_129278 = {}
    # Getting the type of 'assert_equal' (line 374)
    assert_equal_129265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 374)
    assert_equal_call_result_129279 = invoke(stypy.reporting.localization.Localization(__file__, 374, 4), assert_equal_129265, *[list_129266, out_129277], **kwargs_129278)
    
    
    # Assigning a Call to a Name (line 375):
    
    # Call to maximum_filter1d(...): (line 375)
    # Processing the call arguments (line 375)
    # Getting the type of 'in_' (line 375)
    in__129282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 32), 'in_', False)
    int_129283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 37), 'int')
    # Processing the call keyword arguments (line 375)
    str_129284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 45), 'str', 'constant')
    keyword_129285 = str_129284
    int_129286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 62), 'int')
    keyword_129287 = int_129286
    kwargs_129288 = {'cval': keyword_129287, 'mode': keyword_129285}
    # Getting the type of 'sndi' (line 375)
    sndi_129280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 10), 'sndi', False)
    # Obtaining the member 'maximum_filter1d' of a type (line 375)
    maximum_filter1d_129281 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 375, 10), sndi_129280, 'maximum_filter1d')
    # Calling maximum_filter1d(args, kwargs) (line 375)
    maximum_filter1d_call_result_129289 = invoke(stypy.reporting.localization.Localization(__file__, 375, 10), maximum_filter1d_129281, *[in__129282, int_129283], **kwargs_129288)
    
    # Assigning a type to the variable 'out' (line 375)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 375, 4), 'out', maximum_filter1d_call_result_129289)
    
    # Call to assert_equal(...): (line 376)
    # Processing the call arguments (line 376)
    
    # Obtaining an instance of the builtin type 'list' (line 376)
    list_129291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 376, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 376)
    # Adding element type (line 376)
    int_129292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 376, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 376, 17), list_129291, int_129292)
    # Adding element type (line 376)
    int_129293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 376, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 376, 17), list_129291, int_129293)
    # Adding element type (line 376)
    int_129294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 376, 26), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 376, 17), list_129291, int_129294)
    # Adding element type (line 376)
    int_129295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 376, 29), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 376, 17), list_129291, int_129295)
    # Adding element type (line 376)
    int_129296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 376, 32), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 376, 17), list_129291, int_129296)
    # Adding element type (line 376)
    int_129297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 376, 35), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 376, 17), list_129291, int_129297)
    # Adding element type (line 376)
    int_129298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 376, 38), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 376, 17), list_129291, int_129298)
    # Adding element type (line 376)
    int_129299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 376, 41), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 376, 17), list_129291, int_129299)
    # Adding element type (line 376)
    int_129300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 376, 44), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 376, 17), list_129291, int_129300)
    # Adding element type (line 376)
    int_129301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 376, 48), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 376, 17), list_129291, int_129301)
    
    # Getting the type of 'out' (line 376)
    out_129302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 53), 'out', False)
    # Processing the call keyword arguments (line 376)
    kwargs_129303 = {}
    # Getting the type of 'assert_equal' (line 376)
    assert_equal_129290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 376)
    assert_equal_call_result_129304 = invoke(stypy.reporting.localization.Localization(__file__, 376, 4), assert_equal_129290, *[list_129291, out_129302], **kwargs_129303)
    
    
    # Assigning a Call to a Name (line 378):
    
    # Call to minimum_filter1d(...): (line 378)
    # Processing the call arguments (line 378)
    # Getting the type of 'in_' (line 378)
    in__129307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 32), 'in_', False)
    int_129308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 378, 37), 'int')
    # Processing the call keyword arguments (line 378)
    str_129309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 378, 45), 'str', 'nearest')
    keyword_129310 = str_129309
    kwargs_129311 = {'mode': keyword_129310}
    # Getting the type of 'sndi' (line 378)
    sndi_129305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 10), 'sndi', False)
    # Obtaining the member 'minimum_filter1d' of a type (line 378)
    minimum_filter1d_129306 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 378, 10), sndi_129305, 'minimum_filter1d')
    # Calling minimum_filter1d(args, kwargs) (line 378)
    minimum_filter1d_call_result_129312 = invoke(stypy.reporting.localization.Localization(__file__, 378, 10), minimum_filter1d_129306, *[in__129307, int_129308], **kwargs_129311)
    
    # Assigning a type to the variable 'out' (line 378)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 378, 4), 'out', minimum_filter1d_call_result_129312)
    
    # Call to assert_equal(...): (line 379)
    # Processing the call arguments (line 379)
    
    # Obtaining an instance of the builtin type 'list' (line 379)
    list_129314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 379, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 379)
    # Adding element type (line 379)
    int_129315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 379, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 379, 17), list_129314, int_129315)
    # Adding element type (line 379)
    int_129316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 379, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 379, 17), list_129314, int_129316)
    # Adding element type (line 379)
    int_129317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 379, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 379, 17), list_129314, int_129317)
    # Adding element type (line 379)
    int_129318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 379, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 379, 17), list_129314, int_129318)
    # Adding element type (line 379)
    int_129319 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 379, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 379, 17), list_129314, int_129319)
    # Adding element type (line 379)
    int_129320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 379, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 379, 17), list_129314, int_129320)
    # Adding element type (line 379)
    int_129321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 379, 36), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 379, 17), list_129314, int_129321)
    # Adding element type (line 379)
    int_129322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 379, 39), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 379, 17), list_129314, int_129322)
    # Adding element type (line 379)
    int_129323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 379, 42), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 379, 17), list_129314, int_129323)
    # Adding element type (line 379)
    int_129324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 379, 45), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 379, 17), list_129314, int_129324)
    
    # Getting the type of 'out' (line 379)
    out_129325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 49), 'out', False)
    # Processing the call keyword arguments (line 379)
    kwargs_129326 = {}
    # Getting the type of 'assert_equal' (line 379)
    assert_equal_129313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 379)
    assert_equal_call_result_129327 = invoke(stypy.reporting.localization.Localization(__file__, 379, 4), assert_equal_129313, *[list_129314, out_129325], **kwargs_129326)
    
    
    # Assigning a Call to a Name (line 380):
    
    # Call to maximum_filter1d(...): (line 380)
    # Processing the call arguments (line 380)
    # Getting the type of 'in_' (line 380)
    in__129330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 32), 'in_', False)
    int_129331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 380, 37), 'int')
    # Processing the call keyword arguments (line 380)
    str_129332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 380, 45), 'str', 'nearest')
    keyword_129333 = str_129332
    kwargs_129334 = {'mode': keyword_129333}
    # Getting the type of 'sndi' (line 380)
    sndi_129328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 10), 'sndi', False)
    # Obtaining the member 'maximum_filter1d' of a type (line 380)
    maximum_filter1d_129329 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 380, 10), sndi_129328, 'maximum_filter1d')
    # Calling maximum_filter1d(args, kwargs) (line 380)
    maximum_filter1d_call_result_129335 = invoke(stypy.reporting.localization.Localization(__file__, 380, 10), maximum_filter1d_129329, *[in__129330, int_129331], **kwargs_129334)
    
    # Assigning a type to the variable 'out' (line 380)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 380, 4), 'out', maximum_filter1d_call_result_129335)
    
    # Call to assert_equal(...): (line 381)
    # Processing the call arguments (line 381)
    
    # Obtaining an instance of the builtin type 'list' (line 381)
    list_129337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 381)
    # Adding element type (line 381)
    int_129338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 381, 17), list_129337, int_129338)
    # Adding element type (line 381)
    int_129339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 381, 17), list_129337, int_129339)
    # Adding element type (line 381)
    int_129340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 381, 17), list_129337, int_129340)
    # Adding element type (line 381)
    int_129341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 381, 17), list_129337, int_129341)
    # Adding element type (line 381)
    int_129342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 381, 17), list_129337, int_129342)
    # Adding element type (line 381)
    int_129343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 381, 17), list_129337, int_129343)
    # Adding element type (line 381)
    int_129344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 36), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 381, 17), list_129337, int_129344)
    # Adding element type (line 381)
    int_129345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 39), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 381, 17), list_129337, int_129345)
    # Adding element type (line 381)
    int_129346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 42), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 381, 17), list_129337, int_129346)
    # Adding element type (line 381)
    int_129347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 45), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 381, 17), list_129337, int_129347)
    
    # Getting the type of 'out' (line 381)
    out_129348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 49), 'out', False)
    # Processing the call keyword arguments (line 381)
    kwargs_129349 = {}
    # Getting the type of 'assert_equal' (line 381)
    assert_equal_129336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 381)
    assert_equal_call_result_129350 = invoke(stypy.reporting.localization.Localization(__file__, 381, 4), assert_equal_129336, *[list_129337, out_129348], **kwargs_129349)
    
    
    # Assigning a Call to a Name (line 383):
    
    # Call to minimum_filter1d(...): (line 383)
    # Processing the call arguments (line 383)
    # Getting the type of 'in_' (line 383)
    in__129353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 32), 'in_', False)
    int_129354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 37), 'int')
    # Processing the call keyword arguments (line 383)
    str_129355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 45), 'str', 'wrap')
    keyword_129356 = str_129355
    kwargs_129357 = {'mode': keyword_129356}
    # Getting the type of 'sndi' (line 383)
    sndi_129351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 10), 'sndi', False)
    # Obtaining the member 'minimum_filter1d' of a type (line 383)
    minimum_filter1d_129352 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 383, 10), sndi_129351, 'minimum_filter1d')
    # Calling minimum_filter1d(args, kwargs) (line 383)
    minimum_filter1d_call_result_129358 = invoke(stypy.reporting.localization.Localization(__file__, 383, 10), minimum_filter1d_129352, *[in__129353, int_129354], **kwargs_129357)
    
    # Assigning a type to the variable 'out' (line 383)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 4), 'out', minimum_filter1d_call_result_129358)
    
    # Call to assert_equal(...): (line 384)
    # Processing the call arguments (line 384)
    
    # Obtaining an instance of the builtin type 'list' (line 384)
    list_129360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 384)
    # Adding element type (line 384)
    int_129361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 384, 17), list_129360, int_129361)
    # Adding element type (line 384)
    int_129362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 384, 17), list_129360, int_129362)
    # Adding element type (line 384)
    int_129363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 384, 17), list_129360, int_129363)
    # Adding element type (line 384)
    int_129364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 384, 17), list_129360, int_129364)
    # Adding element type (line 384)
    int_129365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 384, 17), list_129360, int_129365)
    # Adding element type (line 384)
    int_129366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 384, 17), list_129360, int_129366)
    # Adding element type (line 384)
    int_129367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 36), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 384, 17), list_129360, int_129367)
    # Adding element type (line 384)
    int_129368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 39), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 384, 17), list_129360, int_129368)
    # Adding element type (line 384)
    int_129369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 42), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 384, 17), list_129360, int_129369)
    # Adding element type (line 384)
    int_129370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 45), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 384, 17), list_129360, int_129370)
    
    # Getting the type of 'out' (line 384)
    out_129371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 49), 'out', False)
    # Processing the call keyword arguments (line 384)
    kwargs_129372 = {}
    # Getting the type of 'assert_equal' (line 384)
    assert_equal_129359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 384)
    assert_equal_call_result_129373 = invoke(stypy.reporting.localization.Localization(__file__, 384, 4), assert_equal_129359, *[list_129360, out_129371], **kwargs_129372)
    
    
    # Assigning a Call to a Name (line 385):
    
    # Call to maximum_filter1d(...): (line 385)
    # Processing the call arguments (line 385)
    # Getting the type of 'in_' (line 385)
    in__129376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 32), 'in_', False)
    int_129377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, 37), 'int')
    # Processing the call keyword arguments (line 385)
    str_129378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, 45), 'str', 'wrap')
    keyword_129379 = str_129378
    kwargs_129380 = {'mode': keyword_129379}
    # Getting the type of 'sndi' (line 385)
    sndi_129374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 10), 'sndi', False)
    # Obtaining the member 'maximum_filter1d' of a type (line 385)
    maximum_filter1d_129375 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 385, 10), sndi_129374, 'maximum_filter1d')
    # Calling maximum_filter1d(args, kwargs) (line 385)
    maximum_filter1d_call_result_129381 = invoke(stypy.reporting.localization.Localization(__file__, 385, 10), maximum_filter1d_129375, *[in__129376, int_129377], **kwargs_129380)
    
    # Assigning a type to the variable 'out' (line 385)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 385, 4), 'out', maximum_filter1d_call_result_129381)
    
    # Call to assert_equal(...): (line 386)
    # Processing the call arguments (line 386)
    
    # Obtaining an instance of the builtin type 'list' (line 386)
    list_129383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 386)
    # Adding element type (line 386)
    int_129384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 386, 17), list_129383, int_129384)
    # Adding element type (line 386)
    int_129385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 386, 17), list_129383, int_129385)
    # Adding element type (line 386)
    int_129386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 386, 17), list_129383, int_129386)
    # Adding element type (line 386)
    int_129387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 386, 17), list_129383, int_129387)
    # Adding element type (line 386)
    int_129388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 386, 17), list_129383, int_129388)
    # Adding element type (line 386)
    int_129389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 386, 17), list_129383, int_129389)
    # Adding element type (line 386)
    int_129390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 36), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 386, 17), list_129383, int_129390)
    # Adding element type (line 386)
    int_129391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 39), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 386, 17), list_129383, int_129391)
    # Adding element type (line 386)
    int_129392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 42), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 386, 17), list_129383, int_129392)
    # Adding element type (line 386)
    int_129393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 45), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 386, 17), list_129383, int_129393)
    
    # Getting the type of 'out' (line 386)
    out_129394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 49), 'out', False)
    # Processing the call keyword arguments (line 386)
    kwargs_129395 = {}
    # Getting the type of 'assert_equal' (line 386)
    assert_equal_129382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 386)
    assert_equal_call_result_129396 = invoke(stypy.reporting.localization.Localization(__file__, 386, 4), assert_equal_129382, *[list_129383, out_129394], **kwargs_129395)
    
    
    # ################# End of 'test_minmaximum_filter1d(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_minmaximum_filter1d' in the type store
    # Getting the type of 'stypy_return_type' (line 360)
    stypy_return_type_129397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_129397)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_minmaximum_filter1d'
    return stypy_return_type_129397

# Assigning a type to the variable 'test_minmaximum_filter1d' (line 360)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 360, 0), 'test_minmaximum_filter1d', test_minmaximum_filter1d)

@norecursion
def test_uniform_filter1d_roundoff_errors(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_uniform_filter1d_roundoff_errors'
    module_type_store = module_type_store.open_function_context('test_uniform_filter1d_roundoff_errors', 389, 0, False)
    
    # Passed parameters checking function
    test_uniform_filter1d_roundoff_errors.stypy_localization = localization
    test_uniform_filter1d_roundoff_errors.stypy_type_of_self = None
    test_uniform_filter1d_roundoff_errors.stypy_type_store = module_type_store
    test_uniform_filter1d_roundoff_errors.stypy_function_name = 'test_uniform_filter1d_roundoff_errors'
    test_uniform_filter1d_roundoff_errors.stypy_param_names_list = []
    test_uniform_filter1d_roundoff_errors.stypy_varargs_param_name = None
    test_uniform_filter1d_roundoff_errors.stypy_kwargs_param_name = None
    test_uniform_filter1d_roundoff_errors.stypy_call_defaults = defaults
    test_uniform_filter1d_roundoff_errors.stypy_call_varargs = varargs
    test_uniform_filter1d_roundoff_errors.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_uniform_filter1d_roundoff_errors', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_uniform_filter1d_roundoff_errors', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_uniform_filter1d_roundoff_errors(...)' code ##################

    
    # Assigning a Call to a Name (line 391):
    
    # Call to repeat(...): (line 391)
    # Processing the call arguments (line 391)
    
    # Obtaining an instance of the builtin type 'list' (line 391)
    list_129400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 391, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 391)
    # Adding element type (line 391)
    int_129401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 391, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 391, 20), list_129400, int_129401)
    # Adding element type (line 391)
    int_129402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 391, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 391, 20), list_129400, int_129402)
    # Adding element type (line 391)
    int_129403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 391, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 391, 20), list_129400, int_129403)
    
    
    # Obtaining an instance of the builtin type 'list' (line 391)
    list_129404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 391, 31), 'list')
    # Adding type elements to the builtin type 'list' instance (line 391)
    # Adding element type (line 391)
    int_129405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 391, 32), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 391, 31), list_129404, int_129405)
    # Adding element type (line 391)
    int_129406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 391, 35), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 391, 31), list_129404, int_129406)
    # Adding element type (line 391)
    int_129407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 391, 38), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 391, 31), list_129404, int_129407)
    
    # Processing the call keyword arguments (line 391)
    kwargs_129408 = {}
    # Getting the type of 'np' (line 391)
    np_129398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 10), 'np', False)
    # Obtaining the member 'repeat' of a type (line 391)
    repeat_129399 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 391, 10), np_129398, 'repeat')
    # Calling repeat(args, kwargs) (line 391)
    repeat_call_result_129409 = invoke(stypy.reporting.localization.Localization(__file__, 391, 10), repeat_129399, *[list_129400, list_129404], **kwargs_129408)
    
    # Assigning a type to the variable 'in_' (line 391)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 391, 4), 'in_', repeat_call_result_129409)
    
    
    # Call to range(...): (line 392)
    # Processing the call arguments (line 392)
    int_129411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, 29), 'int')
    int_129412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, 32), 'int')
    # Processing the call keyword arguments (line 392)
    kwargs_129413 = {}
    # Getting the type of 'range' (line 392)
    range_129410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 23), 'range', False)
    # Calling range(args, kwargs) (line 392)
    range_call_result_129414 = invoke(stypy.reporting.localization.Localization(__file__, 392, 23), range_129410, *[int_129411, int_129412], **kwargs_129413)
    
    # Testing the type of a for loop iterable (line 392)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 392, 4), range_call_result_129414)
    # Getting the type of the for loop variable (line 392)
    for_loop_var_129415 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 392, 4), range_call_result_129414)
    # Assigning a type to the variable 'filter_size' (line 392)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 392, 4), 'filter_size', for_loop_var_129415)
    # SSA begins for a for statement (line 392)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 393):
    
    # Call to uniform_filter1d(...): (line 393)
    # Processing the call arguments (line 393)
    # Getting the type of 'in_' (line 393)
    in__129418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 36), 'in_', False)
    # Getting the type of 'filter_size' (line 393)
    filter_size_129419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 41), 'filter_size', False)
    # Processing the call keyword arguments (line 393)
    kwargs_129420 = {}
    # Getting the type of 'sndi' (line 393)
    sndi_129416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 14), 'sndi', False)
    # Obtaining the member 'uniform_filter1d' of a type (line 393)
    uniform_filter1d_129417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 14), sndi_129416, 'uniform_filter1d')
    # Calling uniform_filter1d(args, kwargs) (line 393)
    uniform_filter1d_call_result_129421 = invoke(stypy.reporting.localization.Localization(__file__, 393, 14), uniform_filter1d_129417, *[in__129418, filter_size_129419], **kwargs_129420)
    
    # Assigning a type to the variable 'out' (line 393)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 8), 'out', uniform_filter1d_call_result_129421)
    
    # Call to assert_equal(...): (line 394)
    # Processing the call arguments (line 394)
    
    # Call to sum(...): (line 394)
    # Processing the call keyword arguments (line 394)
    kwargs_129425 = {}
    # Getting the type of 'out' (line 394)
    out_129423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 21), 'out', False)
    # Obtaining the member 'sum' of a type (line 394)
    sum_129424 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 21), out_129423, 'sum')
    # Calling sum(args, kwargs) (line 394)
    sum_call_result_129426 = invoke(stypy.reporting.localization.Localization(__file__, 394, 21), sum_129424, *[], **kwargs_129425)
    
    int_129427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 32), 'int')
    # Getting the type of 'filter_size' (line 394)
    filter_size_129428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 37), 'filter_size', False)
    # Applying the binary operator '-' (line 394)
    result_sub_129429 = python_operator(stypy.reporting.localization.Localization(__file__, 394, 32), '-', int_129427, filter_size_129428)
    
    # Processing the call keyword arguments (line 394)
    kwargs_129430 = {}
    # Getting the type of 'assert_equal' (line 394)
    assert_equal_129422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 8), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 394)
    assert_equal_call_result_129431 = invoke(stypy.reporting.localization.Localization(__file__, 394, 8), assert_equal_129422, *[sum_call_result_129426, result_sub_129429], **kwargs_129430)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_uniform_filter1d_roundoff_errors(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_uniform_filter1d_roundoff_errors' in the type store
    # Getting the type of 'stypy_return_type' (line 389)
    stypy_return_type_129432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_129432)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_uniform_filter1d_roundoff_errors'
    return stypy_return_type_129432

# Assigning a type to the variable 'test_uniform_filter1d_roundoff_errors' (line 389)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 0), 'test_uniform_filter1d_roundoff_errors', test_uniform_filter1d_roundoff_errors)

@norecursion
def test_footprint_all_zeros(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_footprint_all_zeros'
    module_type_store = module_type_store.open_function_context('test_footprint_all_zeros', 397, 0, False)
    
    # Passed parameters checking function
    test_footprint_all_zeros.stypy_localization = localization
    test_footprint_all_zeros.stypy_type_of_self = None
    test_footprint_all_zeros.stypy_type_store = module_type_store
    test_footprint_all_zeros.stypy_function_name = 'test_footprint_all_zeros'
    test_footprint_all_zeros.stypy_param_names_list = []
    test_footprint_all_zeros.stypy_varargs_param_name = None
    test_footprint_all_zeros.stypy_kwargs_param_name = None
    test_footprint_all_zeros.stypy_call_defaults = defaults
    test_footprint_all_zeros.stypy_call_varargs = varargs
    test_footprint_all_zeros.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_footprint_all_zeros', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_footprint_all_zeros', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_footprint_all_zeros(...)' code ##################

    
    # Assigning a Call to a Name (line 399):
    
    # Call to randint(...): (line 399)
    # Processing the call arguments (line 399)
    int_129436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 28), 'int')
    int_129437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 31), 'int')
    
    # Obtaining an instance of the builtin type 'tuple' (line 399)
    tuple_129438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 37), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 399)
    # Adding element type (line 399)
    int_129439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 37), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 399, 37), tuple_129438, int_129439)
    # Adding element type (line 399)
    int_129440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 42), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 399, 37), tuple_129438, int_129440)
    
    # Processing the call keyword arguments (line 399)
    kwargs_129441 = {}
    # Getting the type of 'np' (line 399)
    np_129433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 10), 'np', False)
    # Obtaining the member 'random' of a type (line 399)
    random_129434 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 10), np_129433, 'random')
    # Obtaining the member 'randint' of a type (line 399)
    randint_129435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 10), random_129434, 'randint')
    # Calling randint(args, kwargs) (line 399)
    randint_call_result_129442 = invoke(stypy.reporting.localization.Localization(__file__, 399, 10), randint_129435, *[int_129436, int_129437, tuple_129438], **kwargs_129441)
    
    # Assigning a type to the variable 'arr' (line 399)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 399, 4), 'arr', randint_call_result_129442)
    
    # Assigning a Call to a Name (line 400):
    
    # Call to zeros(...): (line 400)
    # Processing the call arguments (line 400)
    
    # Obtaining an instance of the builtin type 'tuple' (line 400)
    tuple_129445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 23), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 400)
    # Adding element type (line 400)
    int_129446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 400, 23), tuple_129445, int_129446)
    # Adding element type (line 400)
    int_129447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 26), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 400, 23), tuple_129445, int_129447)
    
    # Getting the type of 'bool' (line 400)
    bool_129448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 30), 'bool', False)
    # Processing the call keyword arguments (line 400)
    kwargs_129449 = {}
    # Getting the type of 'np' (line 400)
    np_129443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 13), 'np', False)
    # Obtaining the member 'zeros' of a type (line 400)
    zeros_129444 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 400, 13), np_129443, 'zeros')
    # Calling zeros(args, kwargs) (line 400)
    zeros_call_result_129450 = invoke(stypy.reporting.localization.Localization(__file__, 400, 13), zeros_129444, *[tuple_129445, bool_129448], **kwargs_129449)
    
    # Assigning a type to the variable 'kernel' (line 400)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 400, 4), 'kernel', zeros_call_result_129450)
    
    # Call to assert_raises(...): (line 401)
    # Processing the call arguments (line 401)
    # Getting the type of 'ValueError' (line 401)
    ValueError_129452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 23), 'ValueError', False)
    # Processing the call keyword arguments (line 401)
    kwargs_129453 = {}
    # Getting the type of 'assert_raises' (line 401)
    assert_raises_129451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 9), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 401)
    assert_raises_call_result_129454 = invoke(stypy.reporting.localization.Localization(__file__, 401, 9), assert_raises_129451, *[ValueError_129452], **kwargs_129453)
    
    with_129455 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 401, 9), assert_raises_call_result_129454, 'with parameter', '__enter__', '__exit__')

    if with_129455:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 401)
        enter___129456 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 401, 9), assert_raises_call_result_129454, '__enter__')
        with_enter_129457 = invoke(stypy.reporting.localization.Localization(__file__, 401, 9), enter___129456)
        
        # Call to maximum_filter(...): (line 402)
        # Processing the call arguments (line 402)
        # Getting the type of 'arr' (line 402)
        arr_129460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 28), 'arr', False)
        # Processing the call keyword arguments (line 402)
        # Getting the type of 'kernel' (line 402)
        kernel_129461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 43), 'kernel', False)
        keyword_129462 = kernel_129461
        kwargs_129463 = {'footprint': keyword_129462}
        # Getting the type of 'sndi' (line 402)
        sndi_129458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 8), 'sndi', False)
        # Obtaining the member 'maximum_filter' of a type (line 402)
        maximum_filter_129459 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 402, 8), sndi_129458, 'maximum_filter')
        # Calling maximum_filter(args, kwargs) (line 402)
        maximum_filter_call_result_129464 = invoke(stypy.reporting.localization.Localization(__file__, 402, 8), maximum_filter_129459, *[arr_129460], **kwargs_129463)
        
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 401)
        exit___129465 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 401, 9), assert_raises_call_result_129454, '__exit__')
        with_exit_129466 = invoke(stypy.reporting.localization.Localization(__file__, 401, 9), exit___129465, None, None, None)

    
    # ################# End of 'test_footprint_all_zeros(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_footprint_all_zeros' in the type store
    # Getting the type of 'stypy_return_type' (line 397)
    stypy_return_type_129467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_129467)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_footprint_all_zeros'
    return stypy_return_type_129467

# Assigning a type to the variable 'test_footprint_all_zeros' (line 397)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 397, 0), 'test_footprint_all_zeros', test_footprint_all_zeros)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
