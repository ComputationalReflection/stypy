
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: import sys
4: import math
5: import numpy as np
6: from numpy import sqrt, cos, sin, arctan, exp, log, pi, Inf
7: from numpy.testing import (assert_,
8:         assert_allclose, assert_array_less, assert_almost_equal)
9: import pytest
10: from pytest import raises as assert_raises
11: 
12: from scipy.integrate import quad, dblquad, tplquad, nquad
13: from scipy._lib.six import xrange
14: from scipy._lib._ccallback import LowLevelCallable
15: 
16: import ctypes
17: import ctypes.util
18: from scipy._lib._ccallback_c import sine_ctypes
19: 
20: import scipy.integrate._test_multivariate as clib_test
21: 
22: 
23: def assert_quad(value_and_err, tabled_value, errTol=1.5e-8):
24:     value, err = value_and_err
25:     assert_allclose(value, tabled_value, atol=err, rtol=0)
26:     if errTol is not None:
27:         assert_array_less(err, errTol)
28: 
29: 
30: class TestCtypesQuad(object):
31:     def setup_method(self):
32:         if sys.platform == 'win32':
33:             if sys.version_info < (3, 5):
34:                 files = [ctypes.util.find_msvcrt()]
35:             else:
36:                 files = ['api-ms-win-crt-math-l1-1-0.dll']
37:         elif sys.platform == 'darwin':
38:             files = ['libm.dylib']
39:         else:
40:             files = ['libm.so', 'libm.so.6']
41: 
42:         for file in files:
43:             try:
44:                 self.lib = ctypes.CDLL(file)
45:                 break
46:             except OSError:
47:                 pass
48:         else:
49:             # This test doesn't work on some Linux platforms (Fedora for
50:             # example) that put an ld script in libm.so - see gh-5370
51:             self.skipTest("Ctypes can't import libm.so")
52: 
53:         restype = ctypes.c_double
54:         argtypes = (ctypes.c_double,)
55:         for name in ['sin', 'cos', 'tan']:
56:             func = getattr(self.lib, name)
57:             func.restype = restype
58:             func.argtypes = argtypes
59: 
60:     def test_typical(self):
61:         assert_quad(quad(self.lib.sin, 0, 5), quad(math.sin, 0, 5)[0])
62:         assert_quad(quad(self.lib.cos, 0, 5), quad(math.cos, 0, 5)[0])
63:         assert_quad(quad(self.lib.tan, 0, 1), quad(math.tan, 0, 1)[0])
64: 
65:     def test_ctypes_sine(self):
66:         quad(LowLevelCallable(sine_ctypes), 0, 1)
67: 
68:     def test_ctypes_variants(self):
69:         lib = ctypes.CDLL(clib_test.__file__)
70: 
71:         sin_0 = lib._sin_0
72:         sin_0.restype = ctypes.c_double
73:         sin_0.argtypes = [ctypes.c_double, ctypes.c_void_p]
74: 
75:         sin_1 = lib._sin_1
76:         sin_1.restype = ctypes.c_double
77:         sin_1.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_void_p]
78: 
79:         sin_2 = lib._sin_2
80:         sin_2.restype = ctypes.c_double
81:         sin_2.argtypes = [ctypes.c_double]
82: 
83:         sin_3 = lib._sin_3
84:         sin_3.restype = ctypes.c_double
85:         sin_3.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
86: 
87:         sin_4 = lib._sin_3
88:         sin_4.restype = ctypes.c_double
89:         sin_4.argtypes = [ctypes.c_int, ctypes.c_double]
90: 
91:         all_sigs = [sin_0, sin_1, sin_2, sin_3, sin_4]
92:         legacy_sigs = [sin_2, sin_4]
93:         legacy_only_sigs = [sin_4]
94: 
95:         # LowLevelCallables work for new signatures
96:         for j, func in enumerate(all_sigs):
97:             callback = LowLevelCallable(func)
98:             if func in legacy_only_sigs:
99:                 assert_raises(ValueError, quad, callback, 0, pi)
100:             else:
101:                 assert_allclose(quad(callback, 0, pi)[0], 2.0)
102: 
103:         # Plain ctypes items work only for legacy signatures
104:         for j, func in enumerate(legacy_sigs):
105:             if func in legacy_sigs:
106:                 assert_allclose(quad(func, 0, pi)[0], 2.0)
107:             else:
108:                 assert_raises(ValueError, quad, func, 0, pi)
109: 
110: 
111: class TestMultivariateCtypesQuad(object):
112:     def setup_method(self):
113:         self.lib = ctypes.CDLL(clib_test.__file__)
114:         restype = ctypes.c_double
115:         argtypes = (ctypes.c_int, ctypes.c_double)
116:         for name in ['_multivariate_typical', '_multivariate_indefinite',
117:                      '_multivariate_sin']:
118:             func = getattr(self.lib, name)
119:             func.restype = restype
120:             func.argtypes = argtypes
121: 
122:     def test_typical(self):
123:         # 1) Typical function with two extra arguments:
124:         assert_quad(quad(self.lib._multivariate_typical, 0, pi, (2, 1.8)),
125:                     0.30614353532540296487)
126: 
127:     def test_indefinite(self):
128:         # 2) Infinite integration limits --- Euler's constant
129:         assert_quad(quad(self.lib._multivariate_indefinite, 0, Inf),
130:                     0.577215664901532860606512)
131: 
132:     def test_threadsafety(self):
133:         # Ensure multivariate ctypes are threadsafe
134:         def threadsafety(y):
135:             return y + quad(self.lib._multivariate_sin, 0, 1)[0]
136:         assert_quad(quad(threadsafety, 0, 1), 0.9596976941318602)
137: 
138: 
139: class TestQuad(object):
140:     def test_typical(self):
141:         # 1) Typical function with two extra arguments:
142:         def myfunc(x, n, z):       # Bessel function integrand
143:             return cos(n*x-z*sin(x))/pi
144:         assert_quad(quad(myfunc, 0, pi, (2, 1.8)), 0.30614353532540296487)
145: 
146:     def test_indefinite(self):
147:         # 2) Infinite integration limits --- Euler's constant
148:         def myfunc(x):           # Euler's constant integrand
149:             return -exp(-x)*log(x)
150:         assert_quad(quad(myfunc, 0, Inf), 0.577215664901532860606512)
151: 
152:     def test_singular(self):
153:         # 3) Singular points in region of integration.
154:         def myfunc(x):
155:             if 0 < x < 2.5:
156:                 return sin(x)
157:             elif 2.5 <= x <= 5.0:
158:                 return exp(-x)
159:             else:
160:                 return 0.0
161: 
162:         assert_quad(quad(myfunc, 0, 10, points=[2.5, 5.0]),
163:                     1 - cos(2.5) + exp(-2.5) - exp(-5.0))
164: 
165:     def test_sine_weighted_finite(self):
166:         # 4) Sine weighted integral (finite limits)
167:         def myfunc(x, a):
168:             return exp(a*(x-1))
169: 
170:         ome = 2.0**3.4
171:         assert_quad(quad(myfunc, 0, 1, args=20, weight='sin', wvar=ome),
172:                     (20*sin(ome)-ome*cos(ome)+ome*exp(-20))/(20**2 + ome**2))
173: 
174:     def test_sine_weighted_infinite(self):
175:         # 5) Sine weighted integral (infinite limits)
176:         def myfunc(x, a):
177:             return exp(-x*a)
178: 
179:         a = 4.0
180:         ome = 3.0
181:         assert_quad(quad(myfunc, 0, Inf, args=a, weight='sin', wvar=ome),
182:                     ome/(a**2 + ome**2))
183: 
184:     def test_cosine_weighted_infinite(self):
185:         # 6) Cosine weighted integral (negative infinite limits)
186:         def myfunc(x, a):
187:             return exp(x*a)
188: 
189:         a = 2.5
190:         ome = 2.3
191:         assert_quad(quad(myfunc, -Inf, 0, args=a, weight='cos', wvar=ome),
192:                     a/(a**2 + ome**2))
193: 
194:     def test_algebraic_log_weight(self):
195:         # 6) Algebraic-logarithmic weight.
196:         def myfunc(x, a):
197:             return 1/(1+x+2**(-a))
198: 
199:         a = 1.5
200:         assert_quad(quad(myfunc, -1, 1, args=a, weight='alg',
201:                          wvar=(-0.5, -0.5)),
202:                     pi/sqrt((1+2**(-a))**2 - 1))
203: 
204:     def test_cauchypv_weight(self):
205:         # 7) Cauchy prinicpal value weighting w(x) = 1/(x-c)
206:         def myfunc(x, a):
207:             return 2.0**(-a)/((x-1)**2+4.0**(-a))
208: 
209:         a = 0.4
210:         tabledValue = ((2.0**(-0.4)*log(1.5) -
211:                         2.0**(-1.4)*log((4.0**(-a)+16) / (4.0**(-a)+1)) -
212:                         arctan(2.0**(a+2)) -
213:                         arctan(2.0**a)) /
214:                        (4.0**(-a) + 1))
215:         assert_quad(quad(myfunc, 0, 5, args=0.4, weight='cauchy', wvar=2.0),
216:                     tabledValue, errTol=1.9e-8)
217: 
218:     def test_double_integral(self):
219:         # 8) Double Integral test
220:         def simpfunc(y, x):       # Note order of arguments.
221:             return x+y
222: 
223:         a, b = 1.0, 2.0
224:         assert_quad(dblquad(simpfunc, a, b, lambda x: x, lambda x: 2*x),
225:                     5/6.0 * (b**3.0-a**3.0))
226: 
227:     def test_double_integral2(self):
228:         def func(x0, x1, t0, t1):
229:             return x0 + x1 + t0 + t1
230:         g = lambda x: x
231:         h = lambda x: 2 * x
232:         args = 1, 2
233:         assert_quad(dblquad(func, 1, 2, g, h, args=args),35./6 + 9*.5)
234: 
235:     def test_triple_integral(self):
236:         # 9) Triple Integral test
237:         def simpfunc(z, y, x, t):      # Note order of arguments.
238:             return (x+y+z)*t
239: 
240:         a, b = 1.0, 2.0
241:         assert_quad(tplquad(simpfunc, a, b,
242:                             lambda x: x, lambda x: 2*x,
243:                             lambda x, y: x - y, lambda x, y: x + y,
244:                             (2.,)),
245:                      2*8/3.0 * (b**4.0 - a**4.0))
246: 
247: 
248: class TestNQuad(object):
249:     def test_fixed_limits(self):
250:         def func1(x0, x1, x2, x3):
251:             val = (x0**2 + x1*x2 - x3**3 + np.sin(x0) +
252:                    (1 if (x0 - 0.2*x3 - 0.5 - 0.25*x1 > 0) else 0))
253:             return val
254: 
255:         def opts_basic(*args):
256:             return {'points': [0.2*args[2] + 0.5 + 0.25*args[0]]}
257: 
258:         res = nquad(func1, [[0, 1], [-1, 1], [.13, .8], [-.15, 1]],
259:                     opts=[opts_basic, {}, {}, {}], full_output=True)
260:         assert_quad(res[:-1], 1.5267454070738635)
261:         assert_(res[-1]['neval'] > 0 and res[-1]['neval'] < 4e5) 
262:         
263:     def test_variable_limits(self):
264:         scale = .1
265: 
266:         def func2(x0, x1, x2, x3, t0, t1):
267:             val = (x0*x1*x3**2 + np.sin(x2) + 1 +
268:                    (1 if x0 + t1*x1 - t0 > 0 else 0))
269:             return val
270: 
271:         def lim0(x1, x2, x3, t0, t1):
272:             return [scale * (x1**2 + x2 + np.cos(x3)*t0*t1 + 1) - 1,
273:                     scale * (x1**2 + x2 + np.cos(x3)*t0*t1 + 1) + 1]
274: 
275:         def lim1(x2, x3, t0, t1):
276:             return [scale * (t0*x2 + t1*x3) - 1,
277:                     scale * (t0*x2 + t1*x3) + 1]
278: 
279:         def lim2(x3, t0, t1):
280:             return [scale * (x3 + t0**2*t1**3) - 1,
281:                     scale * (x3 + t0**2*t1**3) + 1]
282: 
283:         def lim3(t0, t1):
284:             return [scale * (t0 + t1) - 1, scale * (t0 + t1) + 1]
285: 
286:         def opts0(x1, x2, x3, t0, t1):
287:             return {'points': [t0 - t1*x1]}
288: 
289:         def opts1(x2, x3, t0, t1):
290:             return {}
291: 
292:         def opts2(x3, t0, t1):
293:             return {}
294: 
295:         def opts3(t0, t1):
296:             return {}
297: 
298:         res = nquad(func2, [lim0, lim1, lim2, lim3], args=(0, 0),
299:                     opts=[opts0, opts1, opts2, opts3])
300:         assert_quad(res, 25.066666666666663)
301: 
302:     def test_square_separate_ranges_and_opts(self):
303:         def f(y, x):
304:             return 1.0
305: 
306:         assert_quad(nquad(f, [[-1, 1], [-1, 1]], opts=[{}, {}]), 4.0)
307: 
308:     def test_square_aliased_ranges_and_opts(self):
309:         def f(y, x):
310:             return 1.0
311: 
312:         r = [-1, 1]
313:         opt = {}
314:         assert_quad(nquad(f, [r, r], opts=[opt, opt]), 4.0)
315: 
316:     def test_square_separate_fn_ranges_and_opts(self):
317:         def f(y, x):
318:             return 1.0
319: 
320:         def fn_range0(*args):
321:             return (-1, 1)
322: 
323:         def fn_range1(*args):
324:             return (-1, 1)
325: 
326:         def fn_opt0(*args):
327:             return {}
328: 
329:         def fn_opt1(*args):
330:             return {}
331: 
332:         ranges = [fn_range0, fn_range1]
333:         opts = [fn_opt0, fn_opt1]
334:         assert_quad(nquad(f, ranges, opts=opts), 4.0)
335: 
336:     def test_square_aliased_fn_ranges_and_opts(self):
337:         def f(y, x):
338:             return 1.0
339: 
340:         def fn_range(*args):
341:             return (-1, 1)
342: 
343:         def fn_opt(*args):
344:             return {}
345: 
346:         ranges = [fn_range, fn_range]
347:         opts = [fn_opt, fn_opt]
348:         assert_quad(nquad(f, ranges, opts=opts), 4.0)
349: 
350:     def test_matching_quad(self):
351:         def func(x):
352:             return x**2 + 1
353: 
354:         res, reserr = quad(func, 0, 4)
355:         res2, reserr2 = nquad(func, ranges=[[0, 4]])
356:         assert_almost_equal(res, res2)
357:         assert_almost_equal(reserr, reserr2)
358: 
359:     def test_matching_dblquad(self):
360:         def func2d(x0, x1):
361:             return x0**2 + x1**3 - x0 * x1 + 1
362: 
363:         res, reserr = dblquad(func2d, -2, 2, lambda x: -3, lambda x: 3)
364:         res2, reserr2 = nquad(func2d, [[-3, 3], (-2, 2)])
365:         assert_almost_equal(res, res2)
366:         assert_almost_equal(reserr, reserr2)
367: 
368:     def test_matching_tplquad(self):
369:         def func3d(x0, x1, x2, c0, c1):
370:             return x0**2 + c0 * x1**3 - x0 * x1 + 1 + c1 * np.sin(x2)
371: 
372:         res = tplquad(func3d, -1, 2, lambda x: -2, lambda x: 2,
373:                       lambda x, y: -np.pi, lambda x, y: np.pi,
374:                       args=(2, 3))
375:         res2 = nquad(func3d, [[-np.pi, np.pi], [-2, 2], (-1, 2)], args=(2, 3))
376:         assert_almost_equal(res, res2)
377: 
378:     def test_dict_as_opts(self):
379:         try:
380:             out = nquad(lambda x, y: x * y, [[0, 1], [0, 1]], opts={'epsrel': 0.0001})
381:         except(TypeError):
382:             assert False
383: 
384: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import sys' statement (line 3)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import math' statement (line 4)
import math

import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'math', math, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import numpy' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/tests/')
import_49292 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy')

if (type(import_49292) is not StypyTypeError):

    if (import_49292 != 'pyd_module'):
        __import__(import_49292)
        sys_modules_49293 = sys.modules[import_49292]
        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'np', sys_modules_49293.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy', import_49292)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from numpy import sqrt, cos, sin, arctan, exp, log, pi, Inf' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/tests/')
import_49294 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy')

if (type(import_49294) is not StypyTypeError):

    if (import_49294 != 'pyd_module'):
        __import__(import_49294)
        sys_modules_49295 = sys.modules[import_49294]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy', sys_modules_49295.module_type_store, module_type_store, ['sqrt', 'cos', 'sin', 'arctan', 'exp', 'log', 'pi', 'Inf'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_49295, sys_modules_49295.module_type_store, module_type_store)
    else:
        from numpy import sqrt, cos, sin, arctan, exp, log, pi, Inf

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy', None, module_type_store, ['sqrt', 'cos', 'sin', 'arctan', 'exp', 'log', 'pi', 'Inf'], [sqrt, cos, sin, arctan, exp, log, pi, Inf])

else:
    # Assigning a type to the variable 'numpy' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy', import_49294)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from numpy.testing import assert_, assert_allclose, assert_array_less, assert_almost_equal' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/tests/')
import_49296 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.testing')

if (type(import_49296) is not StypyTypeError):

    if (import_49296 != 'pyd_module'):
        __import__(import_49296)
        sys_modules_49297 = sys.modules[import_49296]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.testing', sys_modules_49297.module_type_store, module_type_store, ['assert_', 'assert_allclose', 'assert_array_less', 'assert_almost_equal'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_49297, sys_modules_49297.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_, assert_allclose, assert_array_less, assert_almost_equal

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.testing', None, module_type_store, ['assert_', 'assert_allclose', 'assert_array_less', 'assert_almost_equal'], [assert_, assert_allclose, assert_array_less, assert_almost_equal])

else:
    # Assigning a type to the variable 'numpy.testing' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.testing', import_49296)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'import pytest' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/tests/')
import_49298 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'pytest')

if (type(import_49298) is not StypyTypeError):

    if (import_49298 != 'pyd_module'):
        __import__(import_49298)
        sys_modules_49299 = sys.modules[import_49298]
        import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'pytest', sys_modules_49299.module_type_store, module_type_store)
    else:
        import pytest

        import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'pytest', pytest, module_type_store)

else:
    # Assigning a type to the variable 'pytest' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'pytest', import_49298)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from pytest import assert_raises' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/tests/')
import_49300 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'pytest')

if (type(import_49300) is not StypyTypeError):

    if (import_49300 != 'pyd_module'):
        __import__(import_49300)
        sys_modules_49301 = sys.modules[import_49300]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'pytest', sys_modules_49301.module_type_store, module_type_store, ['raises'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_49301, sys_modules_49301.module_type_store, module_type_store)
    else:
        from pytest import raises as assert_raises

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'pytest', None, module_type_store, ['raises'], [assert_raises])

else:
    # Assigning a type to the variable 'pytest' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'pytest', import_49300)

# Adding an alias
module_type_store.add_alias('assert_raises', 'raises')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from scipy.integrate import quad, dblquad, tplquad, nquad' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/tests/')
import_49302 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.integrate')

if (type(import_49302) is not StypyTypeError):

    if (import_49302 != 'pyd_module'):
        __import__(import_49302)
        sys_modules_49303 = sys.modules[import_49302]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.integrate', sys_modules_49303.module_type_store, module_type_store, ['quad', 'dblquad', 'tplquad', 'nquad'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_49303, sys_modules_49303.module_type_store, module_type_store)
    else:
        from scipy.integrate import quad, dblquad, tplquad, nquad

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.integrate', None, module_type_store, ['quad', 'dblquad', 'tplquad', 'nquad'], [quad, dblquad, tplquad, nquad])

else:
    # Assigning a type to the variable 'scipy.integrate' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.integrate', import_49302)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from scipy._lib.six import xrange' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/tests/')
import_49304 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy._lib.six')

if (type(import_49304) is not StypyTypeError):

    if (import_49304 != 'pyd_module'):
        __import__(import_49304)
        sys_modules_49305 = sys.modules[import_49304]
        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy._lib.six', sys_modules_49305.module_type_store, module_type_store, ['xrange'])
        nest_module(stypy.reporting.localization.Localization(__file__, 13, 0), __file__, sys_modules_49305, sys_modules_49305.module_type_store, module_type_store)
    else:
        from scipy._lib.six import xrange

        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy._lib.six', None, module_type_store, ['xrange'], [xrange])

else:
    # Assigning a type to the variable 'scipy._lib.six' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy._lib.six', import_49304)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'from scipy._lib._ccallback import LowLevelCallable' statement (line 14)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/tests/')
import_49306 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy._lib._ccallback')

if (type(import_49306) is not StypyTypeError):

    if (import_49306 != 'pyd_module'):
        __import__(import_49306)
        sys_modules_49307 = sys.modules[import_49306]
        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy._lib._ccallback', sys_modules_49307.module_type_store, module_type_store, ['LowLevelCallable'])
        nest_module(stypy.reporting.localization.Localization(__file__, 14, 0), __file__, sys_modules_49307, sys_modules_49307.module_type_store, module_type_store)
    else:
        from scipy._lib._ccallback import LowLevelCallable

        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy._lib._ccallback', None, module_type_store, ['LowLevelCallable'], [LowLevelCallable])

else:
    # Assigning a type to the variable 'scipy._lib._ccallback' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy._lib._ccallback', import_49306)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 16, 0))

# 'import ctypes' statement (line 16)
import ctypes

import_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'ctypes', ctypes, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 17, 0))

# 'import ctypes.util' statement (line 17)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/tests/')
import_49308 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'ctypes.util')

if (type(import_49308) is not StypyTypeError):

    if (import_49308 != 'pyd_module'):
        __import__(import_49308)
        sys_modules_49309 = sys.modules[import_49308]
        import_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'ctypes.util', sys_modules_49309.module_type_store, module_type_store)
    else:
        import ctypes.util

        import_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'ctypes.util', ctypes.util, module_type_store)

else:
    # Assigning a type to the variable 'ctypes.util' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'ctypes.util', import_49308)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 18, 0))

# 'from scipy._lib._ccallback_c import sine_ctypes' statement (line 18)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/tests/')
import_49310 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'scipy._lib._ccallback_c')

if (type(import_49310) is not StypyTypeError):

    if (import_49310 != 'pyd_module'):
        __import__(import_49310)
        sys_modules_49311 = sys.modules[import_49310]
        import_from_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'scipy._lib._ccallback_c', sys_modules_49311.module_type_store, module_type_store, ['sine_ctypes'])
        nest_module(stypy.reporting.localization.Localization(__file__, 18, 0), __file__, sys_modules_49311, sys_modules_49311.module_type_store, module_type_store)
    else:
        from scipy._lib._ccallback_c import sine_ctypes

        import_from_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'scipy._lib._ccallback_c', None, module_type_store, ['sine_ctypes'], [sine_ctypes])

else:
    # Assigning a type to the variable 'scipy._lib._ccallback_c' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'scipy._lib._ccallback_c', import_49310)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 20, 0))

# 'import scipy.integrate._test_multivariate' statement (line 20)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/tests/')
import_49312 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'scipy.integrate._test_multivariate')

if (type(import_49312) is not StypyTypeError):

    if (import_49312 != 'pyd_module'):
        __import__(import_49312)
        sys_modules_49313 = sys.modules[import_49312]
        import_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'clib_test', sys_modules_49313.module_type_store, module_type_store)
    else:
        import scipy.integrate._test_multivariate as clib_test

        import_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'clib_test', scipy.integrate._test_multivariate, module_type_store)

else:
    # Assigning a type to the variable 'scipy.integrate._test_multivariate' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'scipy.integrate._test_multivariate', import_49312)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/tests/')


@norecursion
def assert_quad(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    float_49314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 52), 'float')
    defaults = [float_49314]
    # Create a new context for function 'assert_quad'
    module_type_store = module_type_store.open_function_context('assert_quad', 23, 0, False)
    
    # Passed parameters checking function
    assert_quad.stypy_localization = localization
    assert_quad.stypy_type_of_self = None
    assert_quad.stypy_type_store = module_type_store
    assert_quad.stypy_function_name = 'assert_quad'
    assert_quad.stypy_param_names_list = ['value_and_err', 'tabled_value', 'errTol']
    assert_quad.stypy_varargs_param_name = None
    assert_quad.stypy_kwargs_param_name = None
    assert_quad.stypy_call_defaults = defaults
    assert_quad.stypy_call_varargs = varargs
    assert_quad.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'assert_quad', ['value_and_err', 'tabled_value', 'errTol'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'assert_quad', localization, ['value_and_err', 'tabled_value', 'errTol'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'assert_quad(...)' code ##################

    
    # Assigning a Name to a Tuple (line 24):
    
    # Assigning a Subscript to a Name (line 24):
    
    # Obtaining the type of the subscript
    int_49315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 4), 'int')
    # Getting the type of 'value_and_err' (line 24)
    value_and_err_49316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 17), 'value_and_err')
    # Obtaining the member '__getitem__' of a type (line 24)
    getitem___49317 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 4), value_and_err_49316, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 24)
    subscript_call_result_49318 = invoke(stypy.reporting.localization.Localization(__file__, 24, 4), getitem___49317, int_49315)
    
    # Assigning a type to the variable 'tuple_var_assignment_49278' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'tuple_var_assignment_49278', subscript_call_result_49318)
    
    # Assigning a Subscript to a Name (line 24):
    
    # Obtaining the type of the subscript
    int_49319 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 4), 'int')
    # Getting the type of 'value_and_err' (line 24)
    value_and_err_49320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 17), 'value_and_err')
    # Obtaining the member '__getitem__' of a type (line 24)
    getitem___49321 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 4), value_and_err_49320, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 24)
    subscript_call_result_49322 = invoke(stypy.reporting.localization.Localization(__file__, 24, 4), getitem___49321, int_49319)
    
    # Assigning a type to the variable 'tuple_var_assignment_49279' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'tuple_var_assignment_49279', subscript_call_result_49322)
    
    # Assigning a Name to a Name (line 24):
    # Getting the type of 'tuple_var_assignment_49278' (line 24)
    tuple_var_assignment_49278_49323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'tuple_var_assignment_49278')
    # Assigning a type to the variable 'value' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'value', tuple_var_assignment_49278_49323)
    
    # Assigning a Name to a Name (line 24):
    # Getting the type of 'tuple_var_assignment_49279' (line 24)
    tuple_var_assignment_49279_49324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'tuple_var_assignment_49279')
    # Assigning a type to the variable 'err' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 11), 'err', tuple_var_assignment_49279_49324)
    
    # Call to assert_allclose(...): (line 25)
    # Processing the call arguments (line 25)
    # Getting the type of 'value' (line 25)
    value_49326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 20), 'value', False)
    # Getting the type of 'tabled_value' (line 25)
    tabled_value_49327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 27), 'tabled_value', False)
    # Processing the call keyword arguments (line 25)
    # Getting the type of 'err' (line 25)
    err_49328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 46), 'err', False)
    keyword_49329 = err_49328
    int_49330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 56), 'int')
    keyword_49331 = int_49330
    kwargs_49332 = {'rtol': keyword_49331, 'atol': keyword_49329}
    # Getting the type of 'assert_allclose' (line 25)
    assert_allclose_49325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 25)
    assert_allclose_call_result_49333 = invoke(stypy.reporting.localization.Localization(__file__, 25, 4), assert_allclose_49325, *[value_49326, tabled_value_49327], **kwargs_49332)
    
    
    # Type idiom detected: calculating its left and rigth part (line 26)
    # Getting the type of 'errTol' (line 26)
    errTol_49334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'errTol')
    # Getting the type of 'None' (line 26)
    None_49335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 21), 'None')
    
    (may_be_49336, more_types_in_union_49337) = may_not_be_none(errTol_49334, None_49335)

    if may_be_49336:

        if more_types_in_union_49337:
            # Runtime conditional SSA (line 26)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to assert_array_less(...): (line 27)
        # Processing the call arguments (line 27)
        # Getting the type of 'err' (line 27)
        err_49339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 26), 'err', False)
        # Getting the type of 'errTol' (line 27)
        errTol_49340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 31), 'errTol', False)
        # Processing the call keyword arguments (line 27)
        kwargs_49341 = {}
        # Getting the type of 'assert_array_less' (line 27)
        assert_array_less_49338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'assert_array_less', False)
        # Calling assert_array_less(args, kwargs) (line 27)
        assert_array_less_call_result_49342 = invoke(stypy.reporting.localization.Localization(__file__, 27, 8), assert_array_less_49338, *[err_49339, errTol_49340], **kwargs_49341)
        

        if more_types_in_union_49337:
            # SSA join for if statement (line 26)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # ################# End of 'assert_quad(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'assert_quad' in the type store
    # Getting the type of 'stypy_return_type' (line 23)
    stypy_return_type_49343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_49343)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'assert_quad'
    return stypy_return_type_49343

# Assigning a type to the variable 'assert_quad' (line 23)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'assert_quad', assert_quad)
# Declaration of the 'TestCtypesQuad' class

class TestCtypesQuad(object, ):

    @norecursion
    def setup_method(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setup_method'
        module_type_store = module_type_store.open_function_context('setup_method', 31, 4, False)
        # Assigning a type to the variable 'self' (line 32)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCtypesQuad.setup_method.__dict__.__setitem__('stypy_localization', localization)
        TestCtypesQuad.setup_method.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCtypesQuad.setup_method.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCtypesQuad.setup_method.__dict__.__setitem__('stypy_function_name', 'TestCtypesQuad.setup_method')
        TestCtypesQuad.setup_method.__dict__.__setitem__('stypy_param_names_list', [])
        TestCtypesQuad.setup_method.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCtypesQuad.setup_method.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCtypesQuad.setup_method.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCtypesQuad.setup_method.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCtypesQuad.setup_method.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCtypesQuad.setup_method.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCtypesQuad.setup_method', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'setup_method', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'setup_method(...)' code ##################

        
        
        # Getting the type of 'sys' (line 32)
        sys_49344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 11), 'sys')
        # Obtaining the member 'platform' of a type (line 32)
        platform_49345 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 11), sys_49344, 'platform')
        str_49346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 27), 'str', 'win32')
        # Applying the binary operator '==' (line 32)
        result_eq_49347 = python_operator(stypy.reporting.localization.Localization(__file__, 32, 11), '==', platform_49345, str_49346)
        
        # Testing the type of an if condition (line 32)
        if_condition_49348 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 32, 8), result_eq_49347)
        # Assigning a type to the variable 'if_condition_49348' (line 32)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'if_condition_49348', if_condition_49348)
        # SSA begins for if statement (line 32)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Getting the type of 'sys' (line 33)
        sys_49349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 15), 'sys')
        # Obtaining the member 'version_info' of a type (line 33)
        version_info_49350 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 15), sys_49349, 'version_info')
        
        # Obtaining an instance of the builtin type 'tuple' (line 33)
        tuple_49351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 35), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 33)
        # Adding element type (line 33)
        int_49352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 35), tuple_49351, int_49352)
        # Adding element type (line 33)
        int_49353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 35), tuple_49351, int_49353)
        
        # Applying the binary operator '<' (line 33)
        result_lt_49354 = python_operator(stypy.reporting.localization.Localization(__file__, 33, 15), '<', version_info_49350, tuple_49351)
        
        # Testing the type of an if condition (line 33)
        if_condition_49355 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 33, 12), result_lt_49354)
        # Assigning a type to the variable 'if_condition_49355' (line 33)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 12), 'if_condition_49355', if_condition_49355)
        # SSA begins for if statement (line 33)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a List to a Name (line 34):
        
        # Assigning a List to a Name (line 34):
        
        # Obtaining an instance of the builtin type 'list' (line 34)
        list_49356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 34)
        # Adding element type (line 34)
        
        # Call to find_msvcrt(...): (line 34)
        # Processing the call keyword arguments (line 34)
        kwargs_49360 = {}
        # Getting the type of 'ctypes' (line 34)
        ctypes_49357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 25), 'ctypes', False)
        # Obtaining the member 'util' of a type (line 34)
        util_49358 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 25), ctypes_49357, 'util')
        # Obtaining the member 'find_msvcrt' of a type (line 34)
        find_msvcrt_49359 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 25), util_49358, 'find_msvcrt')
        # Calling find_msvcrt(args, kwargs) (line 34)
        find_msvcrt_call_result_49361 = invoke(stypy.reporting.localization.Localization(__file__, 34, 25), find_msvcrt_49359, *[], **kwargs_49360)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 24), list_49356, find_msvcrt_call_result_49361)
        
        # Assigning a type to the variable 'files' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 16), 'files', list_49356)
        # SSA branch for the else part of an if statement (line 33)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a List to a Name (line 36):
        
        # Assigning a List to a Name (line 36):
        
        # Obtaining an instance of the builtin type 'list' (line 36)
        list_49362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 36)
        # Adding element type (line 36)
        str_49363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 25), 'str', 'api-ms-win-crt-math-l1-1-0.dll')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 24), list_49362, str_49363)
        
        # Assigning a type to the variable 'files' (line 36)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 16), 'files', list_49362)
        # SSA join for if statement (line 33)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 32)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'sys' (line 37)
        sys_49364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 13), 'sys')
        # Obtaining the member 'platform' of a type (line 37)
        platform_49365 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 13), sys_49364, 'platform')
        str_49366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 29), 'str', 'darwin')
        # Applying the binary operator '==' (line 37)
        result_eq_49367 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 13), '==', platform_49365, str_49366)
        
        # Testing the type of an if condition (line 37)
        if_condition_49368 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 37, 13), result_eq_49367)
        # Assigning a type to the variable 'if_condition_49368' (line 37)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 13), 'if_condition_49368', if_condition_49368)
        # SSA begins for if statement (line 37)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a List to a Name (line 38):
        
        # Assigning a List to a Name (line 38):
        
        # Obtaining an instance of the builtin type 'list' (line 38)
        list_49369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 38)
        # Adding element type (line 38)
        str_49370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 21), 'str', 'libm.dylib')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 20), list_49369, str_49370)
        
        # Assigning a type to the variable 'files' (line 38)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 12), 'files', list_49369)
        # SSA branch for the else part of an if statement (line 37)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a List to a Name (line 40):
        
        # Assigning a List to a Name (line 40):
        
        # Obtaining an instance of the builtin type 'list' (line 40)
        list_49371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 40)
        # Adding element type (line 40)
        str_49372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 21), 'str', 'libm.so')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 20), list_49371, str_49372)
        # Adding element type (line 40)
        str_49373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 32), 'str', 'libm.so.6')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 20), list_49371, str_49373)
        
        # Assigning a type to the variable 'files' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 12), 'files', list_49371)
        # SSA join for if statement (line 37)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 32)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'files' (line 42)
        files_49374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 20), 'files')
        # Testing the type of a for loop iterable (line 42)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 42, 8), files_49374)
        # Getting the type of the for loop variable (line 42)
        for_loop_var_49375 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 42, 8), files_49374)
        # Assigning a type to the variable 'file' (line 42)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'file', for_loop_var_49375)
        # SSA begins for a for statement (line 42)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # SSA begins for try-except statement (line 43)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Attribute (line 44):
        
        # Assigning a Call to a Attribute (line 44):
        
        # Call to CDLL(...): (line 44)
        # Processing the call arguments (line 44)
        # Getting the type of 'file' (line 44)
        file_49378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 39), 'file', False)
        # Processing the call keyword arguments (line 44)
        kwargs_49379 = {}
        # Getting the type of 'ctypes' (line 44)
        ctypes_49376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 27), 'ctypes', False)
        # Obtaining the member 'CDLL' of a type (line 44)
        CDLL_49377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 27), ctypes_49376, 'CDLL')
        # Calling CDLL(args, kwargs) (line 44)
        CDLL_call_result_49380 = invoke(stypy.reporting.localization.Localization(__file__, 44, 27), CDLL_49377, *[file_49378], **kwargs_49379)
        
        # Getting the type of 'self' (line 44)
        self_49381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 16), 'self')
        # Setting the type of the member 'lib' of a type (line 44)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 16), self_49381, 'lib', CDLL_call_result_49380)
        # SSA branch for the except part of a try statement (line 43)
        # SSA branch for the except 'OSError' branch of a try statement (line 43)
        module_type_store.open_ssa_branch('except')
        pass
        # SSA join for try-except statement (line 43)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of a for statement (line 42)
        module_type_store.open_ssa_branch('for loop else')
        
        # Call to skipTest(...): (line 51)
        # Processing the call arguments (line 51)
        str_49384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 26), 'str', "Ctypes can't import libm.so")
        # Processing the call keyword arguments (line 51)
        kwargs_49385 = {}
        # Getting the type of 'self' (line 51)
        self_49382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 12), 'self', False)
        # Obtaining the member 'skipTest' of a type (line 51)
        skipTest_49383 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 12), self_49382, 'skipTest')
        # Calling skipTest(args, kwargs) (line 51)
        skipTest_call_result_49386 = invoke(stypy.reporting.localization.Localization(__file__, 51, 12), skipTest_49383, *[str_49384], **kwargs_49385)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Attribute to a Name (line 53):
        
        # Assigning a Attribute to a Name (line 53):
        # Getting the type of 'ctypes' (line 53)
        ctypes_49387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 18), 'ctypes')
        # Obtaining the member 'c_double' of a type (line 53)
        c_double_49388 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 18), ctypes_49387, 'c_double')
        # Assigning a type to the variable 'restype' (line 53)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'restype', c_double_49388)
        
        # Assigning a Tuple to a Name (line 54):
        
        # Assigning a Tuple to a Name (line 54):
        
        # Obtaining an instance of the builtin type 'tuple' (line 54)
        tuple_49389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 54)
        # Adding element type (line 54)
        # Getting the type of 'ctypes' (line 54)
        ctypes_49390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 20), 'ctypes')
        # Obtaining the member 'c_double' of a type (line 54)
        c_double_49391 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 20), ctypes_49390, 'c_double')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 20), tuple_49389, c_double_49391)
        
        # Assigning a type to the variable 'argtypes' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'argtypes', tuple_49389)
        
        
        # Obtaining an instance of the builtin type 'list' (line 55)
        list_49392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 55)
        # Adding element type (line 55)
        str_49393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 21), 'str', 'sin')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 20), list_49392, str_49393)
        # Adding element type (line 55)
        str_49394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 28), 'str', 'cos')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 20), list_49392, str_49394)
        # Adding element type (line 55)
        str_49395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 35), 'str', 'tan')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 20), list_49392, str_49395)
        
        # Testing the type of a for loop iterable (line 55)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 55, 8), list_49392)
        # Getting the type of the for loop variable (line 55)
        for_loop_var_49396 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 55, 8), list_49392)
        # Assigning a type to the variable 'name' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'name', for_loop_var_49396)
        # SSA begins for a for statement (line 55)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 56):
        
        # Assigning a Call to a Name (line 56):
        
        # Call to getattr(...): (line 56)
        # Processing the call arguments (line 56)
        # Getting the type of 'self' (line 56)
        self_49398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 27), 'self', False)
        # Obtaining the member 'lib' of a type (line 56)
        lib_49399 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 27), self_49398, 'lib')
        # Getting the type of 'name' (line 56)
        name_49400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 37), 'name', False)
        # Processing the call keyword arguments (line 56)
        kwargs_49401 = {}
        # Getting the type of 'getattr' (line 56)
        getattr_49397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 19), 'getattr', False)
        # Calling getattr(args, kwargs) (line 56)
        getattr_call_result_49402 = invoke(stypy.reporting.localization.Localization(__file__, 56, 19), getattr_49397, *[lib_49399, name_49400], **kwargs_49401)
        
        # Assigning a type to the variable 'func' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 12), 'func', getattr_call_result_49402)
        
        # Assigning a Name to a Attribute (line 57):
        
        # Assigning a Name to a Attribute (line 57):
        # Getting the type of 'restype' (line 57)
        restype_49403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 27), 'restype')
        # Getting the type of 'func' (line 57)
        func_49404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 12), 'func')
        # Setting the type of the member 'restype' of a type (line 57)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 12), func_49404, 'restype', restype_49403)
        
        # Assigning a Name to a Attribute (line 58):
        
        # Assigning a Name to a Attribute (line 58):
        # Getting the type of 'argtypes' (line 58)
        argtypes_49405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 28), 'argtypes')
        # Getting the type of 'func' (line 58)
        func_49406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 12), 'func')
        # Setting the type of the member 'argtypes' of a type (line 58)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 12), func_49406, 'argtypes', argtypes_49405)
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'setup_method(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setup_method' in the type store
        # Getting the type of 'stypy_return_type' (line 31)
        stypy_return_type_49407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_49407)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setup_method'
        return stypy_return_type_49407


    @norecursion
    def test_typical(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_typical'
        module_type_store = module_type_store.open_function_context('test_typical', 60, 4, False)
        # Assigning a type to the variable 'self' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCtypesQuad.test_typical.__dict__.__setitem__('stypy_localization', localization)
        TestCtypesQuad.test_typical.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCtypesQuad.test_typical.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCtypesQuad.test_typical.__dict__.__setitem__('stypy_function_name', 'TestCtypesQuad.test_typical')
        TestCtypesQuad.test_typical.__dict__.__setitem__('stypy_param_names_list', [])
        TestCtypesQuad.test_typical.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCtypesQuad.test_typical.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCtypesQuad.test_typical.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCtypesQuad.test_typical.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCtypesQuad.test_typical.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCtypesQuad.test_typical.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCtypesQuad.test_typical', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_typical', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_typical(...)' code ##################

        
        # Call to assert_quad(...): (line 61)
        # Processing the call arguments (line 61)
        
        # Call to quad(...): (line 61)
        # Processing the call arguments (line 61)
        # Getting the type of 'self' (line 61)
        self_49410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 25), 'self', False)
        # Obtaining the member 'lib' of a type (line 61)
        lib_49411 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 25), self_49410, 'lib')
        # Obtaining the member 'sin' of a type (line 61)
        sin_49412 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 25), lib_49411, 'sin')
        int_49413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 39), 'int')
        int_49414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 42), 'int')
        # Processing the call keyword arguments (line 61)
        kwargs_49415 = {}
        # Getting the type of 'quad' (line 61)
        quad_49409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 20), 'quad', False)
        # Calling quad(args, kwargs) (line 61)
        quad_call_result_49416 = invoke(stypy.reporting.localization.Localization(__file__, 61, 20), quad_49409, *[sin_49412, int_49413, int_49414], **kwargs_49415)
        
        
        # Obtaining the type of the subscript
        int_49417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 67), 'int')
        
        # Call to quad(...): (line 61)
        # Processing the call arguments (line 61)
        # Getting the type of 'math' (line 61)
        math_49419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 51), 'math', False)
        # Obtaining the member 'sin' of a type (line 61)
        sin_49420 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 51), math_49419, 'sin')
        int_49421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 61), 'int')
        int_49422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 64), 'int')
        # Processing the call keyword arguments (line 61)
        kwargs_49423 = {}
        # Getting the type of 'quad' (line 61)
        quad_49418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 46), 'quad', False)
        # Calling quad(args, kwargs) (line 61)
        quad_call_result_49424 = invoke(stypy.reporting.localization.Localization(__file__, 61, 46), quad_49418, *[sin_49420, int_49421, int_49422], **kwargs_49423)
        
        # Obtaining the member '__getitem__' of a type (line 61)
        getitem___49425 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 46), quad_call_result_49424, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 61)
        subscript_call_result_49426 = invoke(stypy.reporting.localization.Localization(__file__, 61, 46), getitem___49425, int_49417)
        
        # Processing the call keyword arguments (line 61)
        kwargs_49427 = {}
        # Getting the type of 'assert_quad' (line 61)
        assert_quad_49408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'assert_quad', False)
        # Calling assert_quad(args, kwargs) (line 61)
        assert_quad_call_result_49428 = invoke(stypy.reporting.localization.Localization(__file__, 61, 8), assert_quad_49408, *[quad_call_result_49416, subscript_call_result_49426], **kwargs_49427)
        
        
        # Call to assert_quad(...): (line 62)
        # Processing the call arguments (line 62)
        
        # Call to quad(...): (line 62)
        # Processing the call arguments (line 62)
        # Getting the type of 'self' (line 62)
        self_49431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 25), 'self', False)
        # Obtaining the member 'lib' of a type (line 62)
        lib_49432 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 25), self_49431, 'lib')
        # Obtaining the member 'cos' of a type (line 62)
        cos_49433 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 25), lib_49432, 'cos')
        int_49434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 39), 'int')
        int_49435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 42), 'int')
        # Processing the call keyword arguments (line 62)
        kwargs_49436 = {}
        # Getting the type of 'quad' (line 62)
        quad_49430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 20), 'quad', False)
        # Calling quad(args, kwargs) (line 62)
        quad_call_result_49437 = invoke(stypy.reporting.localization.Localization(__file__, 62, 20), quad_49430, *[cos_49433, int_49434, int_49435], **kwargs_49436)
        
        
        # Obtaining the type of the subscript
        int_49438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 67), 'int')
        
        # Call to quad(...): (line 62)
        # Processing the call arguments (line 62)
        # Getting the type of 'math' (line 62)
        math_49440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 51), 'math', False)
        # Obtaining the member 'cos' of a type (line 62)
        cos_49441 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 51), math_49440, 'cos')
        int_49442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 61), 'int')
        int_49443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 64), 'int')
        # Processing the call keyword arguments (line 62)
        kwargs_49444 = {}
        # Getting the type of 'quad' (line 62)
        quad_49439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 46), 'quad', False)
        # Calling quad(args, kwargs) (line 62)
        quad_call_result_49445 = invoke(stypy.reporting.localization.Localization(__file__, 62, 46), quad_49439, *[cos_49441, int_49442, int_49443], **kwargs_49444)
        
        # Obtaining the member '__getitem__' of a type (line 62)
        getitem___49446 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 46), quad_call_result_49445, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 62)
        subscript_call_result_49447 = invoke(stypy.reporting.localization.Localization(__file__, 62, 46), getitem___49446, int_49438)
        
        # Processing the call keyword arguments (line 62)
        kwargs_49448 = {}
        # Getting the type of 'assert_quad' (line 62)
        assert_quad_49429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'assert_quad', False)
        # Calling assert_quad(args, kwargs) (line 62)
        assert_quad_call_result_49449 = invoke(stypy.reporting.localization.Localization(__file__, 62, 8), assert_quad_49429, *[quad_call_result_49437, subscript_call_result_49447], **kwargs_49448)
        
        
        # Call to assert_quad(...): (line 63)
        # Processing the call arguments (line 63)
        
        # Call to quad(...): (line 63)
        # Processing the call arguments (line 63)
        # Getting the type of 'self' (line 63)
        self_49452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 25), 'self', False)
        # Obtaining the member 'lib' of a type (line 63)
        lib_49453 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 25), self_49452, 'lib')
        # Obtaining the member 'tan' of a type (line 63)
        tan_49454 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 25), lib_49453, 'tan')
        int_49455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 39), 'int')
        int_49456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 42), 'int')
        # Processing the call keyword arguments (line 63)
        kwargs_49457 = {}
        # Getting the type of 'quad' (line 63)
        quad_49451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 20), 'quad', False)
        # Calling quad(args, kwargs) (line 63)
        quad_call_result_49458 = invoke(stypy.reporting.localization.Localization(__file__, 63, 20), quad_49451, *[tan_49454, int_49455, int_49456], **kwargs_49457)
        
        
        # Obtaining the type of the subscript
        int_49459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 67), 'int')
        
        # Call to quad(...): (line 63)
        # Processing the call arguments (line 63)
        # Getting the type of 'math' (line 63)
        math_49461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 51), 'math', False)
        # Obtaining the member 'tan' of a type (line 63)
        tan_49462 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 51), math_49461, 'tan')
        int_49463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 61), 'int')
        int_49464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 64), 'int')
        # Processing the call keyword arguments (line 63)
        kwargs_49465 = {}
        # Getting the type of 'quad' (line 63)
        quad_49460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 46), 'quad', False)
        # Calling quad(args, kwargs) (line 63)
        quad_call_result_49466 = invoke(stypy.reporting.localization.Localization(__file__, 63, 46), quad_49460, *[tan_49462, int_49463, int_49464], **kwargs_49465)
        
        # Obtaining the member '__getitem__' of a type (line 63)
        getitem___49467 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 46), quad_call_result_49466, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 63)
        subscript_call_result_49468 = invoke(stypy.reporting.localization.Localization(__file__, 63, 46), getitem___49467, int_49459)
        
        # Processing the call keyword arguments (line 63)
        kwargs_49469 = {}
        # Getting the type of 'assert_quad' (line 63)
        assert_quad_49450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'assert_quad', False)
        # Calling assert_quad(args, kwargs) (line 63)
        assert_quad_call_result_49470 = invoke(stypy.reporting.localization.Localization(__file__, 63, 8), assert_quad_49450, *[quad_call_result_49458, subscript_call_result_49468], **kwargs_49469)
        
        
        # ################# End of 'test_typical(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_typical' in the type store
        # Getting the type of 'stypy_return_type' (line 60)
        stypy_return_type_49471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_49471)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_typical'
        return stypy_return_type_49471


    @norecursion
    def test_ctypes_sine(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_ctypes_sine'
        module_type_store = module_type_store.open_function_context('test_ctypes_sine', 65, 4, False)
        # Assigning a type to the variable 'self' (line 66)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCtypesQuad.test_ctypes_sine.__dict__.__setitem__('stypy_localization', localization)
        TestCtypesQuad.test_ctypes_sine.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCtypesQuad.test_ctypes_sine.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCtypesQuad.test_ctypes_sine.__dict__.__setitem__('stypy_function_name', 'TestCtypesQuad.test_ctypes_sine')
        TestCtypesQuad.test_ctypes_sine.__dict__.__setitem__('stypy_param_names_list', [])
        TestCtypesQuad.test_ctypes_sine.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCtypesQuad.test_ctypes_sine.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCtypesQuad.test_ctypes_sine.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCtypesQuad.test_ctypes_sine.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCtypesQuad.test_ctypes_sine.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCtypesQuad.test_ctypes_sine.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCtypesQuad.test_ctypes_sine', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_ctypes_sine', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_ctypes_sine(...)' code ##################

        
        # Call to quad(...): (line 66)
        # Processing the call arguments (line 66)
        
        # Call to LowLevelCallable(...): (line 66)
        # Processing the call arguments (line 66)
        # Getting the type of 'sine_ctypes' (line 66)
        sine_ctypes_49474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 30), 'sine_ctypes', False)
        # Processing the call keyword arguments (line 66)
        kwargs_49475 = {}
        # Getting the type of 'LowLevelCallable' (line 66)
        LowLevelCallable_49473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 13), 'LowLevelCallable', False)
        # Calling LowLevelCallable(args, kwargs) (line 66)
        LowLevelCallable_call_result_49476 = invoke(stypy.reporting.localization.Localization(__file__, 66, 13), LowLevelCallable_49473, *[sine_ctypes_49474], **kwargs_49475)
        
        int_49477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 44), 'int')
        int_49478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 47), 'int')
        # Processing the call keyword arguments (line 66)
        kwargs_49479 = {}
        # Getting the type of 'quad' (line 66)
        quad_49472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'quad', False)
        # Calling quad(args, kwargs) (line 66)
        quad_call_result_49480 = invoke(stypy.reporting.localization.Localization(__file__, 66, 8), quad_49472, *[LowLevelCallable_call_result_49476, int_49477, int_49478], **kwargs_49479)
        
        
        # ################# End of 'test_ctypes_sine(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_ctypes_sine' in the type store
        # Getting the type of 'stypy_return_type' (line 65)
        stypy_return_type_49481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_49481)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_ctypes_sine'
        return stypy_return_type_49481


    @norecursion
    def test_ctypes_variants(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_ctypes_variants'
        module_type_store = module_type_store.open_function_context('test_ctypes_variants', 68, 4, False)
        # Assigning a type to the variable 'self' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCtypesQuad.test_ctypes_variants.__dict__.__setitem__('stypy_localization', localization)
        TestCtypesQuad.test_ctypes_variants.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCtypesQuad.test_ctypes_variants.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCtypesQuad.test_ctypes_variants.__dict__.__setitem__('stypy_function_name', 'TestCtypesQuad.test_ctypes_variants')
        TestCtypesQuad.test_ctypes_variants.__dict__.__setitem__('stypy_param_names_list', [])
        TestCtypesQuad.test_ctypes_variants.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCtypesQuad.test_ctypes_variants.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCtypesQuad.test_ctypes_variants.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCtypesQuad.test_ctypes_variants.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCtypesQuad.test_ctypes_variants.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCtypesQuad.test_ctypes_variants.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCtypesQuad.test_ctypes_variants', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_ctypes_variants', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_ctypes_variants(...)' code ##################

        
        # Assigning a Call to a Name (line 69):
        
        # Assigning a Call to a Name (line 69):
        
        # Call to CDLL(...): (line 69)
        # Processing the call arguments (line 69)
        # Getting the type of 'clib_test' (line 69)
        clib_test_49484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 26), 'clib_test', False)
        # Obtaining the member '__file__' of a type (line 69)
        file___49485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 26), clib_test_49484, '__file__')
        # Processing the call keyword arguments (line 69)
        kwargs_49486 = {}
        # Getting the type of 'ctypes' (line 69)
        ctypes_49482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 14), 'ctypes', False)
        # Obtaining the member 'CDLL' of a type (line 69)
        CDLL_49483 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 14), ctypes_49482, 'CDLL')
        # Calling CDLL(args, kwargs) (line 69)
        CDLL_call_result_49487 = invoke(stypy.reporting.localization.Localization(__file__, 69, 14), CDLL_49483, *[file___49485], **kwargs_49486)
        
        # Assigning a type to the variable 'lib' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'lib', CDLL_call_result_49487)
        
        # Assigning a Attribute to a Name (line 71):
        
        # Assigning a Attribute to a Name (line 71):
        # Getting the type of 'lib' (line 71)
        lib_49488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 16), 'lib')
        # Obtaining the member '_sin_0' of a type (line 71)
        _sin_0_49489 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 16), lib_49488, '_sin_0')
        # Assigning a type to the variable 'sin_0' (line 71)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'sin_0', _sin_0_49489)
        
        # Assigning a Attribute to a Attribute (line 72):
        
        # Assigning a Attribute to a Attribute (line 72):
        # Getting the type of 'ctypes' (line 72)
        ctypes_49490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 24), 'ctypes')
        # Obtaining the member 'c_double' of a type (line 72)
        c_double_49491 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 24), ctypes_49490, 'c_double')
        # Getting the type of 'sin_0' (line 72)
        sin_0_49492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'sin_0')
        # Setting the type of the member 'restype' of a type (line 72)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 8), sin_0_49492, 'restype', c_double_49491)
        
        # Assigning a List to a Attribute (line 73):
        
        # Assigning a List to a Attribute (line 73):
        
        # Obtaining an instance of the builtin type 'list' (line 73)
        list_49493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 73)
        # Adding element type (line 73)
        # Getting the type of 'ctypes' (line 73)
        ctypes_49494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 26), 'ctypes')
        # Obtaining the member 'c_double' of a type (line 73)
        c_double_49495 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 26), ctypes_49494, 'c_double')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 25), list_49493, c_double_49495)
        # Adding element type (line 73)
        # Getting the type of 'ctypes' (line 73)
        ctypes_49496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 43), 'ctypes')
        # Obtaining the member 'c_void_p' of a type (line 73)
        c_void_p_49497 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 43), ctypes_49496, 'c_void_p')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 25), list_49493, c_void_p_49497)
        
        # Getting the type of 'sin_0' (line 73)
        sin_0_49498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'sin_0')
        # Setting the type of the member 'argtypes' of a type (line 73)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 8), sin_0_49498, 'argtypes', list_49493)
        
        # Assigning a Attribute to a Name (line 75):
        
        # Assigning a Attribute to a Name (line 75):
        # Getting the type of 'lib' (line 75)
        lib_49499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 16), 'lib')
        # Obtaining the member '_sin_1' of a type (line 75)
        _sin_1_49500 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 16), lib_49499, '_sin_1')
        # Assigning a type to the variable 'sin_1' (line 75)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'sin_1', _sin_1_49500)
        
        # Assigning a Attribute to a Attribute (line 76):
        
        # Assigning a Attribute to a Attribute (line 76):
        # Getting the type of 'ctypes' (line 76)
        ctypes_49501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 24), 'ctypes')
        # Obtaining the member 'c_double' of a type (line 76)
        c_double_49502 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 24), ctypes_49501, 'c_double')
        # Getting the type of 'sin_1' (line 76)
        sin_1_49503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'sin_1')
        # Setting the type of the member 'restype' of a type (line 76)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 8), sin_1_49503, 'restype', c_double_49502)
        
        # Assigning a List to a Attribute (line 77):
        
        # Assigning a List to a Attribute (line 77):
        
        # Obtaining an instance of the builtin type 'list' (line 77)
        list_49504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 77)
        # Adding element type (line 77)
        # Getting the type of 'ctypes' (line 77)
        ctypes_49505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 26), 'ctypes')
        # Obtaining the member 'c_int' of a type (line 77)
        c_int_49506 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 26), ctypes_49505, 'c_int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 25), list_49504, c_int_49506)
        # Adding element type (line 77)
        
        # Call to POINTER(...): (line 77)
        # Processing the call arguments (line 77)
        # Getting the type of 'ctypes' (line 77)
        ctypes_49509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 55), 'ctypes', False)
        # Obtaining the member 'c_double' of a type (line 77)
        c_double_49510 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 55), ctypes_49509, 'c_double')
        # Processing the call keyword arguments (line 77)
        kwargs_49511 = {}
        # Getting the type of 'ctypes' (line 77)
        ctypes_49507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 40), 'ctypes', False)
        # Obtaining the member 'POINTER' of a type (line 77)
        POINTER_49508 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 40), ctypes_49507, 'POINTER')
        # Calling POINTER(args, kwargs) (line 77)
        POINTER_call_result_49512 = invoke(stypy.reporting.localization.Localization(__file__, 77, 40), POINTER_49508, *[c_double_49510], **kwargs_49511)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 25), list_49504, POINTER_call_result_49512)
        # Adding element type (line 77)
        # Getting the type of 'ctypes' (line 77)
        ctypes_49513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 73), 'ctypes')
        # Obtaining the member 'c_void_p' of a type (line 77)
        c_void_p_49514 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 73), ctypes_49513, 'c_void_p')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 25), list_49504, c_void_p_49514)
        
        # Getting the type of 'sin_1' (line 77)
        sin_1_49515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'sin_1')
        # Setting the type of the member 'argtypes' of a type (line 77)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 8), sin_1_49515, 'argtypes', list_49504)
        
        # Assigning a Attribute to a Name (line 79):
        
        # Assigning a Attribute to a Name (line 79):
        # Getting the type of 'lib' (line 79)
        lib_49516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 16), 'lib')
        # Obtaining the member '_sin_2' of a type (line 79)
        _sin_2_49517 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 16), lib_49516, '_sin_2')
        # Assigning a type to the variable 'sin_2' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'sin_2', _sin_2_49517)
        
        # Assigning a Attribute to a Attribute (line 80):
        
        # Assigning a Attribute to a Attribute (line 80):
        # Getting the type of 'ctypes' (line 80)
        ctypes_49518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 24), 'ctypes')
        # Obtaining the member 'c_double' of a type (line 80)
        c_double_49519 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 24), ctypes_49518, 'c_double')
        # Getting the type of 'sin_2' (line 80)
        sin_2_49520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'sin_2')
        # Setting the type of the member 'restype' of a type (line 80)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 8), sin_2_49520, 'restype', c_double_49519)
        
        # Assigning a List to a Attribute (line 81):
        
        # Assigning a List to a Attribute (line 81):
        
        # Obtaining an instance of the builtin type 'list' (line 81)
        list_49521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 81)
        # Adding element type (line 81)
        # Getting the type of 'ctypes' (line 81)
        ctypes_49522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 26), 'ctypes')
        # Obtaining the member 'c_double' of a type (line 81)
        c_double_49523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 26), ctypes_49522, 'c_double')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 25), list_49521, c_double_49523)
        
        # Getting the type of 'sin_2' (line 81)
        sin_2_49524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'sin_2')
        # Setting the type of the member 'argtypes' of a type (line 81)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 8), sin_2_49524, 'argtypes', list_49521)
        
        # Assigning a Attribute to a Name (line 83):
        
        # Assigning a Attribute to a Name (line 83):
        # Getting the type of 'lib' (line 83)
        lib_49525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 16), 'lib')
        # Obtaining the member '_sin_3' of a type (line 83)
        _sin_3_49526 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 16), lib_49525, '_sin_3')
        # Assigning a type to the variable 'sin_3' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'sin_3', _sin_3_49526)
        
        # Assigning a Attribute to a Attribute (line 84):
        
        # Assigning a Attribute to a Attribute (line 84):
        # Getting the type of 'ctypes' (line 84)
        ctypes_49527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 24), 'ctypes')
        # Obtaining the member 'c_double' of a type (line 84)
        c_double_49528 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 24), ctypes_49527, 'c_double')
        # Getting the type of 'sin_3' (line 84)
        sin_3_49529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'sin_3')
        # Setting the type of the member 'restype' of a type (line 84)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 8), sin_3_49529, 'restype', c_double_49528)
        
        # Assigning a List to a Attribute (line 85):
        
        # Assigning a List to a Attribute (line 85):
        
        # Obtaining an instance of the builtin type 'list' (line 85)
        list_49530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 85)
        # Adding element type (line 85)
        # Getting the type of 'ctypes' (line 85)
        ctypes_49531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 26), 'ctypes')
        # Obtaining the member 'c_int' of a type (line 85)
        c_int_49532 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 26), ctypes_49531, 'c_int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 25), list_49530, c_int_49532)
        # Adding element type (line 85)
        
        # Call to POINTER(...): (line 85)
        # Processing the call arguments (line 85)
        # Getting the type of 'ctypes' (line 85)
        ctypes_49535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 55), 'ctypes', False)
        # Obtaining the member 'c_double' of a type (line 85)
        c_double_49536 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 55), ctypes_49535, 'c_double')
        # Processing the call keyword arguments (line 85)
        kwargs_49537 = {}
        # Getting the type of 'ctypes' (line 85)
        ctypes_49533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 40), 'ctypes', False)
        # Obtaining the member 'POINTER' of a type (line 85)
        POINTER_49534 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 40), ctypes_49533, 'POINTER')
        # Calling POINTER(args, kwargs) (line 85)
        POINTER_call_result_49538 = invoke(stypy.reporting.localization.Localization(__file__, 85, 40), POINTER_49534, *[c_double_49536], **kwargs_49537)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 25), list_49530, POINTER_call_result_49538)
        
        # Getting the type of 'sin_3' (line 85)
        sin_3_49539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'sin_3')
        # Setting the type of the member 'argtypes' of a type (line 85)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 8), sin_3_49539, 'argtypes', list_49530)
        
        # Assigning a Attribute to a Name (line 87):
        
        # Assigning a Attribute to a Name (line 87):
        # Getting the type of 'lib' (line 87)
        lib_49540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 16), 'lib')
        # Obtaining the member '_sin_3' of a type (line 87)
        _sin_3_49541 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 16), lib_49540, '_sin_3')
        # Assigning a type to the variable 'sin_4' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'sin_4', _sin_3_49541)
        
        # Assigning a Attribute to a Attribute (line 88):
        
        # Assigning a Attribute to a Attribute (line 88):
        # Getting the type of 'ctypes' (line 88)
        ctypes_49542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 24), 'ctypes')
        # Obtaining the member 'c_double' of a type (line 88)
        c_double_49543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 24), ctypes_49542, 'c_double')
        # Getting the type of 'sin_4' (line 88)
        sin_4_49544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'sin_4')
        # Setting the type of the member 'restype' of a type (line 88)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 8), sin_4_49544, 'restype', c_double_49543)
        
        # Assigning a List to a Attribute (line 89):
        
        # Assigning a List to a Attribute (line 89):
        
        # Obtaining an instance of the builtin type 'list' (line 89)
        list_49545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 89)
        # Adding element type (line 89)
        # Getting the type of 'ctypes' (line 89)
        ctypes_49546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 26), 'ctypes')
        # Obtaining the member 'c_int' of a type (line 89)
        c_int_49547 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 26), ctypes_49546, 'c_int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 25), list_49545, c_int_49547)
        # Adding element type (line 89)
        # Getting the type of 'ctypes' (line 89)
        ctypes_49548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 40), 'ctypes')
        # Obtaining the member 'c_double' of a type (line 89)
        c_double_49549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 40), ctypes_49548, 'c_double')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 25), list_49545, c_double_49549)
        
        # Getting the type of 'sin_4' (line 89)
        sin_4_49550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'sin_4')
        # Setting the type of the member 'argtypes' of a type (line 89)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 8), sin_4_49550, 'argtypes', list_49545)
        
        # Assigning a List to a Name (line 91):
        
        # Assigning a List to a Name (line 91):
        
        # Obtaining an instance of the builtin type 'list' (line 91)
        list_49551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 91)
        # Adding element type (line 91)
        # Getting the type of 'sin_0' (line 91)
        sin_0_49552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 20), 'sin_0')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 19), list_49551, sin_0_49552)
        # Adding element type (line 91)
        # Getting the type of 'sin_1' (line 91)
        sin_1_49553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 27), 'sin_1')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 19), list_49551, sin_1_49553)
        # Adding element type (line 91)
        # Getting the type of 'sin_2' (line 91)
        sin_2_49554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 34), 'sin_2')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 19), list_49551, sin_2_49554)
        # Adding element type (line 91)
        # Getting the type of 'sin_3' (line 91)
        sin_3_49555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 41), 'sin_3')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 19), list_49551, sin_3_49555)
        # Adding element type (line 91)
        # Getting the type of 'sin_4' (line 91)
        sin_4_49556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 48), 'sin_4')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 19), list_49551, sin_4_49556)
        
        # Assigning a type to the variable 'all_sigs' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'all_sigs', list_49551)
        
        # Assigning a List to a Name (line 92):
        
        # Assigning a List to a Name (line 92):
        
        # Obtaining an instance of the builtin type 'list' (line 92)
        list_49557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 92)
        # Adding element type (line 92)
        # Getting the type of 'sin_2' (line 92)
        sin_2_49558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 23), 'sin_2')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 22), list_49557, sin_2_49558)
        # Adding element type (line 92)
        # Getting the type of 'sin_4' (line 92)
        sin_4_49559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 30), 'sin_4')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 22), list_49557, sin_4_49559)
        
        # Assigning a type to the variable 'legacy_sigs' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'legacy_sigs', list_49557)
        
        # Assigning a List to a Name (line 93):
        
        # Assigning a List to a Name (line 93):
        
        # Obtaining an instance of the builtin type 'list' (line 93)
        list_49560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 93)
        # Adding element type (line 93)
        # Getting the type of 'sin_4' (line 93)
        sin_4_49561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 28), 'sin_4')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 27), list_49560, sin_4_49561)
        
        # Assigning a type to the variable 'legacy_only_sigs' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'legacy_only_sigs', list_49560)
        
        
        # Call to enumerate(...): (line 96)
        # Processing the call arguments (line 96)
        # Getting the type of 'all_sigs' (line 96)
        all_sigs_49563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 33), 'all_sigs', False)
        # Processing the call keyword arguments (line 96)
        kwargs_49564 = {}
        # Getting the type of 'enumerate' (line 96)
        enumerate_49562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 23), 'enumerate', False)
        # Calling enumerate(args, kwargs) (line 96)
        enumerate_call_result_49565 = invoke(stypy.reporting.localization.Localization(__file__, 96, 23), enumerate_49562, *[all_sigs_49563], **kwargs_49564)
        
        # Testing the type of a for loop iterable (line 96)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 96, 8), enumerate_call_result_49565)
        # Getting the type of the for loop variable (line 96)
        for_loop_var_49566 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 96, 8), enumerate_call_result_49565)
        # Assigning a type to the variable 'j' (line 96)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'j', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 8), for_loop_var_49566))
        # Assigning a type to the variable 'func' (line 96)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'func', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 8), for_loop_var_49566))
        # SSA begins for a for statement (line 96)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 97):
        
        # Assigning a Call to a Name (line 97):
        
        # Call to LowLevelCallable(...): (line 97)
        # Processing the call arguments (line 97)
        # Getting the type of 'func' (line 97)
        func_49568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 40), 'func', False)
        # Processing the call keyword arguments (line 97)
        kwargs_49569 = {}
        # Getting the type of 'LowLevelCallable' (line 97)
        LowLevelCallable_49567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 23), 'LowLevelCallable', False)
        # Calling LowLevelCallable(args, kwargs) (line 97)
        LowLevelCallable_call_result_49570 = invoke(stypy.reporting.localization.Localization(__file__, 97, 23), LowLevelCallable_49567, *[func_49568], **kwargs_49569)
        
        # Assigning a type to the variable 'callback' (line 97)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 12), 'callback', LowLevelCallable_call_result_49570)
        
        
        # Getting the type of 'func' (line 98)
        func_49571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 15), 'func')
        # Getting the type of 'legacy_only_sigs' (line 98)
        legacy_only_sigs_49572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 23), 'legacy_only_sigs')
        # Applying the binary operator 'in' (line 98)
        result_contains_49573 = python_operator(stypy.reporting.localization.Localization(__file__, 98, 15), 'in', func_49571, legacy_only_sigs_49572)
        
        # Testing the type of an if condition (line 98)
        if_condition_49574 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 98, 12), result_contains_49573)
        # Assigning a type to the variable 'if_condition_49574' (line 98)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 12), 'if_condition_49574', if_condition_49574)
        # SSA begins for if statement (line 98)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to assert_raises(...): (line 99)
        # Processing the call arguments (line 99)
        # Getting the type of 'ValueError' (line 99)
        ValueError_49576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 30), 'ValueError', False)
        # Getting the type of 'quad' (line 99)
        quad_49577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 42), 'quad', False)
        # Getting the type of 'callback' (line 99)
        callback_49578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 48), 'callback', False)
        int_49579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 58), 'int')
        # Getting the type of 'pi' (line 99)
        pi_49580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 61), 'pi', False)
        # Processing the call keyword arguments (line 99)
        kwargs_49581 = {}
        # Getting the type of 'assert_raises' (line 99)
        assert_raises_49575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 16), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 99)
        assert_raises_call_result_49582 = invoke(stypy.reporting.localization.Localization(__file__, 99, 16), assert_raises_49575, *[ValueError_49576, quad_49577, callback_49578, int_49579, pi_49580], **kwargs_49581)
        
        # SSA branch for the else part of an if statement (line 98)
        module_type_store.open_ssa_branch('else')
        
        # Call to assert_allclose(...): (line 101)
        # Processing the call arguments (line 101)
        
        # Obtaining the type of the subscript
        int_49584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 54), 'int')
        
        # Call to quad(...): (line 101)
        # Processing the call arguments (line 101)
        # Getting the type of 'callback' (line 101)
        callback_49586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 37), 'callback', False)
        int_49587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 47), 'int')
        # Getting the type of 'pi' (line 101)
        pi_49588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 50), 'pi', False)
        # Processing the call keyword arguments (line 101)
        kwargs_49589 = {}
        # Getting the type of 'quad' (line 101)
        quad_49585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 32), 'quad', False)
        # Calling quad(args, kwargs) (line 101)
        quad_call_result_49590 = invoke(stypy.reporting.localization.Localization(__file__, 101, 32), quad_49585, *[callback_49586, int_49587, pi_49588], **kwargs_49589)
        
        # Obtaining the member '__getitem__' of a type (line 101)
        getitem___49591 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 32), quad_call_result_49590, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 101)
        subscript_call_result_49592 = invoke(stypy.reporting.localization.Localization(__file__, 101, 32), getitem___49591, int_49584)
        
        float_49593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 58), 'float')
        # Processing the call keyword arguments (line 101)
        kwargs_49594 = {}
        # Getting the type of 'assert_allclose' (line 101)
        assert_allclose_49583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 16), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 101)
        assert_allclose_call_result_49595 = invoke(stypy.reporting.localization.Localization(__file__, 101, 16), assert_allclose_49583, *[subscript_call_result_49592, float_49593], **kwargs_49594)
        
        # SSA join for if statement (line 98)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to enumerate(...): (line 104)
        # Processing the call arguments (line 104)
        # Getting the type of 'legacy_sigs' (line 104)
        legacy_sigs_49597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 33), 'legacy_sigs', False)
        # Processing the call keyword arguments (line 104)
        kwargs_49598 = {}
        # Getting the type of 'enumerate' (line 104)
        enumerate_49596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 23), 'enumerate', False)
        # Calling enumerate(args, kwargs) (line 104)
        enumerate_call_result_49599 = invoke(stypy.reporting.localization.Localization(__file__, 104, 23), enumerate_49596, *[legacy_sigs_49597], **kwargs_49598)
        
        # Testing the type of a for loop iterable (line 104)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 104, 8), enumerate_call_result_49599)
        # Getting the type of the for loop variable (line 104)
        for_loop_var_49600 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 104, 8), enumerate_call_result_49599)
        # Assigning a type to the variable 'j' (line 104)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'j', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 8), for_loop_var_49600))
        # Assigning a type to the variable 'func' (line 104)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'func', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 8), for_loop_var_49600))
        # SSA begins for a for statement (line 104)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Getting the type of 'func' (line 105)
        func_49601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 15), 'func')
        # Getting the type of 'legacy_sigs' (line 105)
        legacy_sigs_49602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 23), 'legacy_sigs')
        # Applying the binary operator 'in' (line 105)
        result_contains_49603 = python_operator(stypy.reporting.localization.Localization(__file__, 105, 15), 'in', func_49601, legacy_sigs_49602)
        
        # Testing the type of an if condition (line 105)
        if_condition_49604 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 105, 12), result_contains_49603)
        # Assigning a type to the variable 'if_condition_49604' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 12), 'if_condition_49604', if_condition_49604)
        # SSA begins for if statement (line 105)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to assert_allclose(...): (line 106)
        # Processing the call arguments (line 106)
        
        # Obtaining the type of the subscript
        int_49606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 50), 'int')
        
        # Call to quad(...): (line 106)
        # Processing the call arguments (line 106)
        # Getting the type of 'func' (line 106)
        func_49608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 37), 'func', False)
        int_49609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 43), 'int')
        # Getting the type of 'pi' (line 106)
        pi_49610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 46), 'pi', False)
        # Processing the call keyword arguments (line 106)
        kwargs_49611 = {}
        # Getting the type of 'quad' (line 106)
        quad_49607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 32), 'quad', False)
        # Calling quad(args, kwargs) (line 106)
        quad_call_result_49612 = invoke(stypy.reporting.localization.Localization(__file__, 106, 32), quad_49607, *[func_49608, int_49609, pi_49610], **kwargs_49611)
        
        # Obtaining the member '__getitem__' of a type (line 106)
        getitem___49613 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 32), quad_call_result_49612, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 106)
        subscript_call_result_49614 = invoke(stypy.reporting.localization.Localization(__file__, 106, 32), getitem___49613, int_49606)
        
        float_49615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 54), 'float')
        # Processing the call keyword arguments (line 106)
        kwargs_49616 = {}
        # Getting the type of 'assert_allclose' (line 106)
        assert_allclose_49605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 16), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 106)
        assert_allclose_call_result_49617 = invoke(stypy.reporting.localization.Localization(__file__, 106, 16), assert_allclose_49605, *[subscript_call_result_49614, float_49615], **kwargs_49616)
        
        # SSA branch for the else part of an if statement (line 105)
        module_type_store.open_ssa_branch('else')
        
        # Call to assert_raises(...): (line 108)
        # Processing the call arguments (line 108)
        # Getting the type of 'ValueError' (line 108)
        ValueError_49619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 30), 'ValueError', False)
        # Getting the type of 'quad' (line 108)
        quad_49620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 42), 'quad', False)
        # Getting the type of 'func' (line 108)
        func_49621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 48), 'func', False)
        int_49622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 54), 'int')
        # Getting the type of 'pi' (line 108)
        pi_49623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 57), 'pi', False)
        # Processing the call keyword arguments (line 108)
        kwargs_49624 = {}
        # Getting the type of 'assert_raises' (line 108)
        assert_raises_49618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 16), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 108)
        assert_raises_call_result_49625 = invoke(stypy.reporting.localization.Localization(__file__, 108, 16), assert_raises_49618, *[ValueError_49619, quad_49620, func_49621, int_49622, pi_49623], **kwargs_49624)
        
        # SSA join for if statement (line 105)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_ctypes_variants(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_ctypes_variants' in the type store
        # Getting the type of 'stypy_return_type' (line 68)
        stypy_return_type_49626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_49626)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_ctypes_variants'
        return stypy_return_type_49626


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 30, 0, False)
        # Assigning a type to the variable 'self' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCtypesQuad.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestCtypesQuad' (line 30)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), 'TestCtypesQuad', TestCtypesQuad)
# Declaration of the 'TestMultivariateCtypesQuad' class

class TestMultivariateCtypesQuad(object, ):

    @norecursion
    def setup_method(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setup_method'
        module_type_store = module_type_store.open_function_context('setup_method', 112, 4, False)
        # Assigning a type to the variable 'self' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestMultivariateCtypesQuad.setup_method.__dict__.__setitem__('stypy_localization', localization)
        TestMultivariateCtypesQuad.setup_method.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestMultivariateCtypesQuad.setup_method.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestMultivariateCtypesQuad.setup_method.__dict__.__setitem__('stypy_function_name', 'TestMultivariateCtypesQuad.setup_method')
        TestMultivariateCtypesQuad.setup_method.__dict__.__setitem__('stypy_param_names_list', [])
        TestMultivariateCtypesQuad.setup_method.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestMultivariateCtypesQuad.setup_method.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestMultivariateCtypesQuad.setup_method.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestMultivariateCtypesQuad.setup_method.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestMultivariateCtypesQuad.setup_method.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestMultivariateCtypesQuad.setup_method.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestMultivariateCtypesQuad.setup_method', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'setup_method', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'setup_method(...)' code ##################

        
        # Assigning a Call to a Attribute (line 113):
        
        # Assigning a Call to a Attribute (line 113):
        
        # Call to CDLL(...): (line 113)
        # Processing the call arguments (line 113)
        # Getting the type of 'clib_test' (line 113)
        clib_test_49629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 31), 'clib_test', False)
        # Obtaining the member '__file__' of a type (line 113)
        file___49630 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 31), clib_test_49629, '__file__')
        # Processing the call keyword arguments (line 113)
        kwargs_49631 = {}
        # Getting the type of 'ctypes' (line 113)
        ctypes_49627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 19), 'ctypes', False)
        # Obtaining the member 'CDLL' of a type (line 113)
        CDLL_49628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 19), ctypes_49627, 'CDLL')
        # Calling CDLL(args, kwargs) (line 113)
        CDLL_call_result_49632 = invoke(stypy.reporting.localization.Localization(__file__, 113, 19), CDLL_49628, *[file___49630], **kwargs_49631)
        
        # Getting the type of 'self' (line 113)
        self_49633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'self')
        # Setting the type of the member 'lib' of a type (line 113)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 8), self_49633, 'lib', CDLL_call_result_49632)
        
        # Assigning a Attribute to a Name (line 114):
        
        # Assigning a Attribute to a Name (line 114):
        # Getting the type of 'ctypes' (line 114)
        ctypes_49634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 18), 'ctypes')
        # Obtaining the member 'c_double' of a type (line 114)
        c_double_49635 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 18), ctypes_49634, 'c_double')
        # Assigning a type to the variable 'restype' (line 114)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'restype', c_double_49635)
        
        # Assigning a Tuple to a Name (line 115):
        
        # Assigning a Tuple to a Name (line 115):
        
        # Obtaining an instance of the builtin type 'tuple' (line 115)
        tuple_49636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 115)
        # Adding element type (line 115)
        # Getting the type of 'ctypes' (line 115)
        ctypes_49637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 20), 'ctypes')
        # Obtaining the member 'c_int' of a type (line 115)
        c_int_49638 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 20), ctypes_49637, 'c_int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 115, 20), tuple_49636, c_int_49638)
        # Adding element type (line 115)
        # Getting the type of 'ctypes' (line 115)
        ctypes_49639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 34), 'ctypes')
        # Obtaining the member 'c_double' of a type (line 115)
        c_double_49640 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 34), ctypes_49639, 'c_double')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 115, 20), tuple_49636, c_double_49640)
        
        # Assigning a type to the variable 'argtypes' (line 115)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 8), 'argtypes', tuple_49636)
        
        
        # Obtaining an instance of the builtin type 'list' (line 116)
        list_49641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 116)
        # Adding element type (line 116)
        str_49642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 21), 'str', '_multivariate_typical')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 20), list_49641, str_49642)
        # Adding element type (line 116)
        str_49643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 46), 'str', '_multivariate_indefinite')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 20), list_49641, str_49643)
        # Adding element type (line 116)
        str_49644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 21), 'str', '_multivariate_sin')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 20), list_49641, str_49644)
        
        # Testing the type of a for loop iterable (line 116)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 116, 8), list_49641)
        # Getting the type of the for loop variable (line 116)
        for_loop_var_49645 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 116, 8), list_49641)
        # Assigning a type to the variable 'name' (line 116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'name', for_loop_var_49645)
        # SSA begins for a for statement (line 116)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 118):
        
        # Assigning a Call to a Name (line 118):
        
        # Call to getattr(...): (line 118)
        # Processing the call arguments (line 118)
        # Getting the type of 'self' (line 118)
        self_49647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 27), 'self', False)
        # Obtaining the member 'lib' of a type (line 118)
        lib_49648 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 27), self_49647, 'lib')
        # Getting the type of 'name' (line 118)
        name_49649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 37), 'name', False)
        # Processing the call keyword arguments (line 118)
        kwargs_49650 = {}
        # Getting the type of 'getattr' (line 118)
        getattr_49646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 19), 'getattr', False)
        # Calling getattr(args, kwargs) (line 118)
        getattr_call_result_49651 = invoke(stypy.reporting.localization.Localization(__file__, 118, 19), getattr_49646, *[lib_49648, name_49649], **kwargs_49650)
        
        # Assigning a type to the variable 'func' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 12), 'func', getattr_call_result_49651)
        
        # Assigning a Name to a Attribute (line 119):
        
        # Assigning a Name to a Attribute (line 119):
        # Getting the type of 'restype' (line 119)
        restype_49652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 27), 'restype')
        # Getting the type of 'func' (line 119)
        func_49653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 12), 'func')
        # Setting the type of the member 'restype' of a type (line 119)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 12), func_49653, 'restype', restype_49652)
        
        # Assigning a Name to a Attribute (line 120):
        
        # Assigning a Name to a Attribute (line 120):
        # Getting the type of 'argtypes' (line 120)
        argtypes_49654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 28), 'argtypes')
        # Getting the type of 'func' (line 120)
        func_49655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 12), 'func')
        # Setting the type of the member 'argtypes' of a type (line 120)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 12), func_49655, 'argtypes', argtypes_49654)
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'setup_method(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setup_method' in the type store
        # Getting the type of 'stypy_return_type' (line 112)
        stypy_return_type_49656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_49656)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setup_method'
        return stypy_return_type_49656


    @norecursion
    def test_typical(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_typical'
        module_type_store = module_type_store.open_function_context('test_typical', 122, 4, False)
        # Assigning a type to the variable 'self' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestMultivariateCtypesQuad.test_typical.__dict__.__setitem__('stypy_localization', localization)
        TestMultivariateCtypesQuad.test_typical.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestMultivariateCtypesQuad.test_typical.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestMultivariateCtypesQuad.test_typical.__dict__.__setitem__('stypy_function_name', 'TestMultivariateCtypesQuad.test_typical')
        TestMultivariateCtypesQuad.test_typical.__dict__.__setitem__('stypy_param_names_list', [])
        TestMultivariateCtypesQuad.test_typical.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestMultivariateCtypesQuad.test_typical.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestMultivariateCtypesQuad.test_typical.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestMultivariateCtypesQuad.test_typical.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestMultivariateCtypesQuad.test_typical.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestMultivariateCtypesQuad.test_typical.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestMultivariateCtypesQuad.test_typical', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_typical', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_typical(...)' code ##################

        
        # Call to assert_quad(...): (line 124)
        # Processing the call arguments (line 124)
        
        # Call to quad(...): (line 124)
        # Processing the call arguments (line 124)
        # Getting the type of 'self' (line 124)
        self_49659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 25), 'self', False)
        # Obtaining the member 'lib' of a type (line 124)
        lib_49660 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 25), self_49659, 'lib')
        # Obtaining the member '_multivariate_typical' of a type (line 124)
        _multivariate_typical_49661 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 25), lib_49660, '_multivariate_typical')
        int_49662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 57), 'int')
        # Getting the type of 'pi' (line 124)
        pi_49663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 60), 'pi', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 124)
        tuple_49664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 65), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 124)
        # Adding element type (line 124)
        int_49665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 65), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 65), tuple_49664, int_49665)
        # Adding element type (line 124)
        float_49666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 68), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 65), tuple_49664, float_49666)
        
        # Processing the call keyword arguments (line 124)
        kwargs_49667 = {}
        # Getting the type of 'quad' (line 124)
        quad_49658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 20), 'quad', False)
        # Calling quad(args, kwargs) (line 124)
        quad_call_result_49668 = invoke(stypy.reporting.localization.Localization(__file__, 124, 20), quad_49658, *[_multivariate_typical_49661, int_49662, pi_49663, tuple_49664], **kwargs_49667)
        
        float_49669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 20), 'float')
        # Processing the call keyword arguments (line 124)
        kwargs_49670 = {}
        # Getting the type of 'assert_quad' (line 124)
        assert_quad_49657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'assert_quad', False)
        # Calling assert_quad(args, kwargs) (line 124)
        assert_quad_call_result_49671 = invoke(stypy.reporting.localization.Localization(__file__, 124, 8), assert_quad_49657, *[quad_call_result_49668, float_49669], **kwargs_49670)
        
        
        # ################# End of 'test_typical(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_typical' in the type store
        # Getting the type of 'stypy_return_type' (line 122)
        stypy_return_type_49672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_49672)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_typical'
        return stypy_return_type_49672


    @norecursion
    def test_indefinite(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_indefinite'
        module_type_store = module_type_store.open_function_context('test_indefinite', 127, 4, False)
        # Assigning a type to the variable 'self' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestMultivariateCtypesQuad.test_indefinite.__dict__.__setitem__('stypy_localization', localization)
        TestMultivariateCtypesQuad.test_indefinite.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestMultivariateCtypesQuad.test_indefinite.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestMultivariateCtypesQuad.test_indefinite.__dict__.__setitem__('stypy_function_name', 'TestMultivariateCtypesQuad.test_indefinite')
        TestMultivariateCtypesQuad.test_indefinite.__dict__.__setitem__('stypy_param_names_list', [])
        TestMultivariateCtypesQuad.test_indefinite.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestMultivariateCtypesQuad.test_indefinite.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestMultivariateCtypesQuad.test_indefinite.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestMultivariateCtypesQuad.test_indefinite.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestMultivariateCtypesQuad.test_indefinite.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestMultivariateCtypesQuad.test_indefinite.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestMultivariateCtypesQuad.test_indefinite', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_indefinite', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_indefinite(...)' code ##################

        
        # Call to assert_quad(...): (line 129)
        # Processing the call arguments (line 129)
        
        # Call to quad(...): (line 129)
        # Processing the call arguments (line 129)
        # Getting the type of 'self' (line 129)
        self_49675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 25), 'self', False)
        # Obtaining the member 'lib' of a type (line 129)
        lib_49676 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 25), self_49675, 'lib')
        # Obtaining the member '_multivariate_indefinite' of a type (line 129)
        _multivariate_indefinite_49677 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 25), lib_49676, '_multivariate_indefinite')
        int_49678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 60), 'int')
        # Getting the type of 'Inf' (line 129)
        Inf_49679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 63), 'Inf', False)
        # Processing the call keyword arguments (line 129)
        kwargs_49680 = {}
        # Getting the type of 'quad' (line 129)
        quad_49674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 20), 'quad', False)
        # Calling quad(args, kwargs) (line 129)
        quad_call_result_49681 = invoke(stypy.reporting.localization.Localization(__file__, 129, 20), quad_49674, *[_multivariate_indefinite_49677, int_49678, Inf_49679], **kwargs_49680)
        
        float_49682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 20), 'float')
        # Processing the call keyword arguments (line 129)
        kwargs_49683 = {}
        # Getting the type of 'assert_quad' (line 129)
        assert_quad_49673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'assert_quad', False)
        # Calling assert_quad(args, kwargs) (line 129)
        assert_quad_call_result_49684 = invoke(stypy.reporting.localization.Localization(__file__, 129, 8), assert_quad_49673, *[quad_call_result_49681, float_49682], **kwargs_49683)
        
        
        # ################# End of 'test_indefinite(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_indefinite' in the type store
        # Getting the type of 'stypy_return_type' (line 127)
        stypy_return_type_49685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_49685)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_indefinite'
        return stypy_return_type_49685


    @norecursion
    def test_threadsafety(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_threadsafety'
        module_type_store = module_type_store.open_function_context('test_threadsafety', 132, 4, False)
        # Assigning a type to the variable 'self' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestMultivariateCtypesQuad.test_threadsafety.__dict__.__setitem__('stypy_localization', localization)
        TestMultivariateCtypesQuad.test_threadsafety.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestMultivariateCtypesQuad.test_threadsafety.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestMultivariateCtypesQuad.test_threadsafety.__dict__.__setitem__('stypy_function_name', 'TestMultivariateCtypesQuad.test_threadsafety')
        TestMultivariateCtypesQuad.test_threadsafety.__dict__.__setitem__('stypy_param_names_list', [])
        TestMultivariateCtypesQuad.test_threadsafety.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestMultivariateCtypesQuad.test_threadsafety.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestMultivariateCtypesQuad.test_threadsafety.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestMultivariateCtypesQuad.test_threadsafety.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestMultivariateCtypesQuad.test_threadsafety.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestMultivariateCtypesQuad.test_threadsafety.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestMultivariateCtypesQuad.test_threadsafety', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_threadsafety', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_threadsafety(...)' code ##################


        @norecursion
        def threadsafety(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'threadsafety'
            module_type_store = module_type_store.open_function_context('threadsafety', 134, 8, False)
            
            # Passed parameters checking function
            threadsafety.stypy_localization = localization
            threadsafety.stypy_type_of_self = None
            threadsafety.stypy_type_store = module_type_store
            threadsafety.stypy_function_name = 'threadsafety'
            threadsafety.stypy_param_names_list = ['y']
            threadsafety.stypy_varargs_param_name = None
            threadsafety.stypy_kwargs_param_name = None
            threadsafety.stypy_call_defaults = defaults
            threadsafety.stypy_call_varargs = varargs
            threadsafety.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'threadsafety', ['y'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'threadsafety', localization, ['y'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'threadsafety(...)' code ##################

            # Getting the type of 'y' (line 135)
            y_49686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 19), 'y')
            
            # Obtaining the type of the subscript
            int_49687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 62), 'int')
            
            # Call to quad(...): (line 135)
            # Processing the call arguments (line 135)
            # Getting the type of 'self' (line 135)
            self_49689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 28), 'self', False)
            # Obtaining the member 'lib' of a type (line 135)
            lib_49690 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 28), self_49689, 'lib')
            # Obtaining the member '_multivariate_sin' of a type (line 135)
            _multivariate_sin_49691 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 28), lib_49690, '_multivariate_sin')
            int_49692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 56), 'int')
            int_49693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 59), 'int')
            # Processing the call keyword arguments (line 135)
            kwargs_49694 = {}
            # Getting the type of 'quad' (line 135)
            quad_49688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 23), 'quad', False)
            # Calling quad(args, kwargs) (line 135)
            quad_call_result_49695 = invoke(stypy.reporting.localization.Localization(__file__, 135, 23), quad_49688, *[_multivariate_sin_49691, int_49692, int_49693], **kwargs_49694)
            
            # Obtaining the member '__getitem__' of a type (line 135)
            getitem___49696 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 23), quad_call_result_49695, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 135)
            subscript_call_result_49697 = invoke(stypy.reporting.localization.Localization(__file__, 135, 23), getitem___49696, int_49687)
            
            # Applying the binary operator '+' (line 135)
            result_add_49698 = python_operator(stypy.reporting.localization.Localization(__file__, 135, 19), '+', y_49686, subscript_call_result_49697)
            
            # Assigning a type to the variable 'stypy_return_type' (line 135)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 12), 'stypy_return_type', result_add_49698)
            
            # ################# End of 'threadsafety(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'threadsafety' in the type store
            # Getting the type of 'stypy_return_type' (line 134)
            stypy_return_type_49699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_49699)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'threadsafety'
            return stypy_return_type_49699

        # Assigning a type to the variable 'threadsafety' (line 134)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'threadsafety', threadsafety)
        
        # Call to assert_quad(...): (line 136)
        # Processing the call arguments (line 136)
        
        # Call to quad(...): (line 136)
        # Processing the call arguments (line 136)
        # Getting the type of 'threadsafety' (line 136)
        threadsafety_49702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 25), 'threadsafety', False)
        int_49703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 39), 'int')
        int_49704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 42), 'int')
        # Processing the call keyword arguments (line 136)
        kwargs_49705 = {}
        # Getting the type of 'quad' (line 136)
        quad_49701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 20), 'quad', False)
        # Calling quad(args, kwargs) (line 136)
        quad_call_result_49706 = invoke(stypy.reporting.localization.Localization(__file__, 136, 20), quad_49701, *[threadsafety_49702, int_49703, int_49704], **kwargs_49705)
        
        float_49707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 46), 'float')
        # Processing the call keyword arguments (line 136)
        kwargs_49708 = {}
        # Getting the type of 'assert_quad' (line 136)
        assert_quad_49700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'assert_quad', False)
        # Calling assert_quad(args, kwargs) (line 136)
        assert_quad_call_result_49709 = invoke(stypy.reporting.localization.Localization(__file__, 136, 8), assert_quad_49700, *[quad_call_result_49706, float_49707], **kwargs_49708)
        
        
        # ################# End of 'test_threadsafety(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_threadsafety' in the type store
        # Getting the type of 'stypy_return_type' (line 132)
        stypy_return_type_49710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_49710)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_threadsafety'
        return stypy_return_type_49710


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 111, 0, False)
        # Assigning a type to the variable 'self' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestMultivariateCtypesQuad.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestMultivariateCtypesQuad' (line 111)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 0), 'TestMultivariateCtypesQuad', TestMultivariateCtypesQuad)
# Declaration of the 'TestQuad' class

class TestQuad(object, ):

    @norecursion
    def test_typical(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_typical'
        module_type_store = module_type_store.open_function_context('test_typical', 140, 4, False)
        # Assigning a type to the variable 'self' (line 141)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestQuad.test_typical.__dict__.__setitem__('stypy_localization', localization)
        TestQuad.test_typical.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestQuad.test_typical.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestQuad.test_typical.__dict__.__setitem__('stypy_function_name', 'TestQuad.test_typical')
        TestQuad.test_typical.__dict__.__setitem__('stypy_param_names_list', [])
        TestQuad.test_typical.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestQuad.test_typical.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestQuad.test_typical.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestQuad.test_typical.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestQuad.test_typical.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestQuad.test_typical.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestQuad.test_typical', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_typical', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_typical(...)' code ##################


        @norecursion
        def myfunc(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'myfunc'
            module_type_store = module_type_store.open_function_context('myfunc', 142, 8, False)
            
            # Passed parameters checking function
            myfunc.stypy_localization = localization
            myfunc.stypy_type_of_self = None
            myfunc.stypy_type_store = module_type_store
            myfunc.stypy_function_name = 'myfunc'
            myfunc.stypy_param_names_list = ['x', 'n', 'z']
            myfunc.stypy_varargs_param_name = None
            myfunc.stypy_kwargs_param_name = None
            myfunc.stypy_call_defaults = defaults
            myfunc.stypy_call_varargs = varargs
            myfunc.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'myfunc', ['x', 'n', 'z'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'myfunc', localization, ['x', 'n', 'z'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'myfunc(...)' code ##################

            
            # Call to cos(...): (line 143)
            # Processing the call arguments (line 143)
            # Getting the type of 'n' (line 143)
            n_49712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 23), 'n', False)
            # Getting the type of 'x' (line 143)
            x_49713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 25), 'x', False)
            # Applying the binary operator '*' (line 143)
            result_mul_49714 = python_operator(stypy.reporting.localization.Localization(__file__, 143, 23), '*', n_49712, x_49713)
            
            # Getting the type of 'z' (line 143)
            z_49715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 27), 'z', False)
            
            # Call to sin(...): (line 143)
            # Processing the call arguments (line 143)
            # Getting the type of 'x' (line 143)
            x_49717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 33), 'x', False)
            # Processing the call keyword arguments (line 143)
            kwargs_49718 = {}
            # Getting the type of 'sin' (line 143)
            sin_49716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 29), 'sin', False)
            # Calling sin(args, kwargs) (line 143)
            sin_call_result_49719 = invoke(stypy.reporting.localization.Localization(__file__, 143, 29), sin_49716, *[x_49717], **kwargs_49718)
            
            # Applying the binary operator '*' (line 143)
            result_mul_49720 = python_operator(stypy.reporting.localization.Localization(__file__, 143, 27), '*', z_49715, sin_call_result_49719)
            
            # Applying the binary operator '-' (line 143)
            result_sub_49721 = python_operator(stypy.reporting.localization.Localization(__file__, 143, 23), '-', result_mul_49714, result_mul_49720)
            
            # Processing the call keyword arguments (line 143)
            kwargs_49722 = {}
            # Getting the type of 'cos' (line 143)
            cos_49711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 19), 'cos', False)
            # Calling cos(args, kwargs) (line 143)
            cos_call_result_49723 = invoke(stypy.reporting.localization.Localization(__file__, 143, 19), cos_49711, *[result_sub_49721], **kwargs_49722)
            
            # Getting the type of 'pi' (line 143)
            pi_49724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 37), 'pi')
            # Applying the binary operator 'div' (line 143)
            result_div_49725 = python_operator(stypy.reporting.localization.Localization(__file__, 143, 19), 'div', cos_call_result_49723, pi_49724)
            
            # Assigning a type to the variable 'stypy_return_type' (line 143)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 12), 'stypy_return_type', result_div_49725)
            
            # ################# End of 'myfunc(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'myfunc' in the type store
            # Getting the type of 'stypy_return_type' (line 142)
            stypy_return_type_49726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_49726)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'myfunc'
            return stypy_return_type_49726

        # Assigning a type to the variable 'myfunc' (line 142)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'myfunc', myfunc)
        
        # Call to assert_quad(...): (line 144)
        # Processing the call arguments (line 144)
        
        # Call to quad(...): (line 144)
        # Processing the call arguments (line 144)
        # Getting the type of 'myfunc' (line 144)
        myfunc_49729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 25), 'myfunc', False)
        int_49730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 33), 'int')
        # Getting the type of 'pi' (line 144)
        pi_49731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 36), 'pi', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 144)
        tuple_49732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 41), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 144)
        # Adding element type (line 144)
        int_49733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 41), tuple_49732, int_49733)
        # Adding element type (line 144)
        float_49734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 44), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 41), tuple_49732, float_49734)
        
        # Processing the call keyword arguments (line 144)
        kwargs_49735 = {}
        # Getting the type of 'quad' (line 144)
        quad_49728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 20), 'quad', False)
        # Calling quad(args, kwargs) (line 144)
        quad_call_result_49736 = invoke(stypy.reporting.localization.Localization(__file__, 144, 20), quad_49728, *[myfunc_49729, int_49730, pi_49731, tuple_49732], **kwargs_49735)
        
        float_49737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 51), 'float')
        # Processing the call keyword arguments (line 144)
        kwargs_49738 = {}
        # Getting the type of 'assert_quad' (line 144)
        assert_quad_49727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'assert_quad', False)
        # Calling assert_quad(args, kwargs) (line 144)
        assert_quad_call_result_49739 = invoke(stypy.reporting.localization.Localization(__file__, 144, 8), assert_quad_49727, *[quad_call_result_49736, float_49737], **kwargs_49738)
        
        
        # ################# End of 'test_typical(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_typical' in the type store
        # Getting the type of 'stypy_return_type' (line 140)
        stypy_return_type_49740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_49740)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_typical'
        return stypy_return_type_49740


    @norecursion
    def test_indefinite(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_indefinite'
        module_type_store = module_type_store.open_function_context('test_indefinite', 146, 4, False)
        # Assigning a type to the variable 'self' (line 147)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestQuad.test_indefinite.__dict__.__setitem__('stypy_localization', localization)
        TestQuad.test_indefinite.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestQuad.test_indefinite.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestQuad.test_indefinite.__dict__.__setitem__('stypy_function_name', 'TestQuad.test_indefinite')
        TestQuad.test_indefinite.__dict__.__setitem__('stypy_param_names_list', [])
        TestQuad.test_indefinite.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestQuad.test_indefinite.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestQuad.test_indefinite.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestQuad.test_indefinite.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestQuad.test_indefinite.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestQuad.test_indefinite.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestQuad.test_indefinite', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_indefinite', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_indefinite(...)' code ##################


        @norecursion
        def myfunc(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'myfunc'
            module_type_store = module_type_store.open_function_context('myfunc', 148, 8, False)
            
            # Passed parameters checking function
            myfunc.stypy_localization = localization
            myfunc.stypy_type_of_self = None
            myfunc.stypy_type_store = module_type_store
            myfunc.stypy_function_name = 'myfunc'
            myfunc.stypy_param_names_list = ['x']
            myfunc.stypy_varargs_param_name = None
            myfunc.stypy_kwargs_param_name = None
            myfunc.stypy_call_defaults = defaults
            myfunc.stypy_call_varargs = varargs
            myfunc.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'myfunc', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'myfunc', localization, ['x'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'myfunc(...)' code ##################

            
            
            # Call to exp(...): (line 149)
            # Processing the call arguments (line 149)
            
            # Getting the type of 'x' (line 149)
            x_49742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 25), 'x', False)
            # Applying the 'usub' unary operator (line 149)
            result___neg___49743 = python_operator(stypy.reporting.localization.Localization(__file__, 149, 24), 'usub', x_49742)
            
            # Processing the call keyword arguments (line 149)
            kwargs_49744 = {}
            # Getting the type of 'exp' (line 149)
            exp_49741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 20), 'exp', False)
            # Calling exp(args, kwargs) (line 149)
            exp_call_result_49745 = invoke(stypy.reporting.localization.Localization(__file__, 149, 20), exp_49741, *[result___neg___49743], **kwargs_49744)
            
            # Applying the 'usub' unary operator (line 149)
            result___neg___49746 = python_operator(stypy.reporting.localization.Localization(__file__, 149, 19), 'usub', exp_call_result_49745)
            
            
            # Call to log(...): (line 149)
            # Processing the call arguments (line 149)
            # Getting the type of 'x' (line 149)
            x_49748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 32), 'x', False)
            # Processing the call keyword arguments (line 149)
            kwargs_49749 = {}
            # Getting the type of 'log' (line 149)
            log_49747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 28), 'log', False)
            # Calling log(args, kwargs) (line 149)
            log_call_result_49750 = invoke(stypy.reporting.localization.Localization(__file__, 149, 28), log_49747, *[x_49748], **kwargs_49749)
            
            # Applying the binary operator '*' (line 149)
            result_mul_49751 = python_operator(stypy.reporting.localization.Localization(__file__, 149, 19), '*', result___neg___49746, log_call_result_49750)
            
            # Assigning a type to the variable 'stypy_return_type' (line 149)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 12), 'stypy_return_type', result_mul_49751)
            
            # ################# End of 'myfunc(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'myfunc' in the type store
            # Getting the type of 'stypy_return_type' (line 148)
            stypy_return_type_49752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_49752)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'myfunc'
            return stypy_return_type_49752

        # Assigning a type to the variable 'myfunc' (line 148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 8), 'myfunc', myfunc)
        
        # Call to assert_quad(...): (line 150)
        # Processing the call arguments (line 150)
        
        # Call to quad(...): (line 150)
        # Processing the call arguments (line 150)
        # Getting the type of 'myfunc' (line 150)
        myfunc_49755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 25), 'myfunc', False)
        int_49756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 33), 'int')
        # Getting the type of 'Inf' (line 150)
        Inf_49757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 36), 'Inf', False)
        # Processing the call keyword arguments (line 150)
        kwargs_49758 = {}
        # Getting the type of 'quad' (line 150)
        quad_49754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 20), 'quad', False)
        # Calling quad(args, kwargs) (line 150)
        quad_call_result_49759 = invoke(stypy.reporting.localization.Localization(__file__, 150, 20), quad_49754, *[myfunc_49755, int_49756, Inf_49757], **kwargs_49758)
        
        float_49760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 42), 'float')
        # Processing the call keyword arguments (line 150)
        kwargs_49761 = {}
        # Getting the type of 'assert_quad' (line 150)
        assert_quad_49753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 8), 'assert_quad', False)
        # Calling assert_quad(args, kwargs) (line 150)
        assert_quad_call_result_49762 = invoke(stypy.reporting.localization.Localization(__file__, 150, 8), assert_quad_49753, *[quad_call_result_49759, float_49760], **kwargs_49761)
        
        
        # ################# End of 'test_indefinite(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_indefinite' in the type store
        # Getting the type of 'stypy_return_type' (line 146)
        stypy_return_type_49763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_49763)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_indefinite'
        return stypy_return_type_49763


    @norecursion
    def test_singular(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_singular'
        module_type_store = module_type_store.open_function_context('test_singular', 152, 4, False)
        # Assigning a type to the variable 'self' (line 153)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestQuad.test_singular.__dict__.__setitem__('stypy_localization', localization)
        TestQuad.test_singular.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestQuad.test_singular.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestQuad.test_singular.__dict__.__setitem__('stypy_function_name', 'TestQuad.test_singular')
        TestQuad.test_singular.__dict__.__setitem__('stypy_param_names_list', [])
        TestQuad.test_singular.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestQuad.test_singular.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestQuad.test_singular.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestQuad.test_singular.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestQuad.test_singular.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestQuad.test_singular.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestQuad.test_singular', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_singular', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_singular(...)' code ##################


        @norecursion
        def myfunc(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'myfunc'
            module_type_store = module_type_store.open_function_context('myfunc', 154, 8, False)
            
            # Passed parameters checking function
            myfunc.stypy_localization = localization
            myfunc.stypy_type_of_self = None
            myfunc.stypy_type_store = module_type_store
            myfunc.stypy_function_name = 'myfunc'
            myfunc.stypy_param_names_list = ['x']
            myfunc.stypy_varargs_param_name = None
            myfunc.stypy_kwargs_param_name = None
            myfunc.stypy_call_defaults = defaults
            myfunc.stypy_call_varargs = varargs
            myfunc.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'myfunc', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'myfunc', localization, ['x'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'myfunc(...)' code ##################

            
            
            int_49764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 15), 'int')
            # Getting the type of 'x' (line 155)
            x_49765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 19), 'x')
            # Applying the binary operator '<' (line 155)
            result_lt_49766 = python_operator(stypy.reporting.localization.Localization(__file__, 155, 15), '<', int_49764, x_49765)
            float_49767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 23), 'float')
            # Applying the binary operator '<' (line 155)
            result_lt_49768 = python_operator(stypy.reporting.localization.Localization(__file__, 155, 15), '<', x_49765, float_49767)
            # Applying the binary operator '&' (line 155)
            result_and__49769 = python_operator(stypy.reporting.localization.Localization(__file__, 155, 15), '&', result_lt_49766, result_lt_49768)
            
            # Testing the type of an if condition (line 155)
            if_condition_49770 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 155, 12), result_and__49769)
            # Assigning a type to the variable 'if_condition_49770' (line 155)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 12), 'if_condition_49770', if_condition_49770)
            # SSA begins for if statement (line 155)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to sin(...): (line 156)
            # Processing the call arguments (line 156)
            # Getting the type of 'x' (line 156)
            x_49772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 27), 'x', False)
            # Processing the call keyword arguments (line 156)
            kwargs_49773 = {}
            # Getting the type of 'sin' (line 156)
            sin_49771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 23), 'sin', False)
            # Calling sin(args, kwargs) (line 156)
            sin_call_result_49774 = invoke(stypy.reporting.localization.Localization(__file__, 156, 23), sin_49771, *[x_49772], **kwargs_49773)
            
            # Assigning a type to the variable 'stypy_return_type' (line 156)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 16), 'stypy_return_type', sin_call_result_49774)
            # SSA branch for the else part of an if statement (line 155)
            module_type_store.open_ssa_branch('else')
            
            
            float_49775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 17), 'float')
            # Getting the type of 'x' (line 157)
            x_49776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 24), 'x')
            # Applying the binary operator '<=' (line 157)
            result_le_49777 = python_operator(stypy.reporting.localization.Localization(__file__, 157, 17), '<=', float_49775, x_49776)
            float_49778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 29), 'float')
            # Applying the binary operator '<=' (line 157)
            result_le_49779 = python_operator(stypy.reporting.localization.Localization(__file__, 157, 17), '<=', x_49776, float_49778)
            # Applying the binary operator '&' (line 157)
            result_and__49780 = python_operator(stypy.reporting.localization.Localization(__file__, 157, 17), '&', result_le_49777, result_le_49779)
            
            # Testing the type of an if condition (line 157)
            if_condition_49781 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 157, 17), result_and__49780)
            # Assigning a type to the variable 'if_condition_49781' (line 157)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 17), 'if_condition_49781', if_condition_49781)
            # SSA begins for if statement (line 157)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to exp(...): (line 158)
            # Processing the call arguments (line 158)
            
            # Getting the type of 'x' (line 158)
            x_49783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 28), 'x', False)
            # Applying the 'usub' unary operator (line 158)
            result___neg___49784 = python_operator(stypy.reporting.localization.Localization(__file__, 158, 27), 'usub', x_49783)
            
            # Processing the call keyword arguments (line 158)
            kwargs_49785 = {}
            # Getting the type of 'exp' (line 158)
            exp_49782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 23), 'exp', False)
            # Calling exp(args, kwargs) (line 158)
            exp_call_result_49786 = invoke(stypy.reporting.localization.Localization(__file__, 158, 23), exp_49782, *[result___neg___49784], **kwargs_49785)
            
            # Assigning a type to the variable 'stypy_return_type' (line 158)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 16), 'stypy_return_type', exp_call_result_49786)
            # SSA branch for the else part of an if statement (line 157)
            module_type_store.open_ssa_branch('else')
            float_49787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 23), 'float')
            # Assigning a type to the variable 'stypy_return_type' (line 160)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 16), 'stypy_return_type', float_49787)
            # SSA join for if statement (line 157)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 155)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # ################# End of 'myfunc(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'myfunc' in the type store
            # Getting the type of 'stypy_return_type' (line 154)
            stypy_return_type_49788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_49788)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'myfunc'
            return stypy_return_type_49788

        # Assigning a type to the variable 'myfunc' (line 154)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'myfunc', myfunc)
        
        # Call to assert_quad(...): (line 162)
        # Processing the call arguments (line 162)
        
        # Call to quad(...): (line 162)
        # Processing the call arguments (line 162)
        # Getting the type of 'myfunc' (line 162)
        myfunc_49791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 25), 'myfunc', False)
        int_49792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 33), 'int')
        int_49793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 36), 'int')
        # Processing the call keyword arguments (line 162)
        
        # Obtaining an instance of the builtin type 'list' (line 162)
        list_49794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 47), 'list')
        # Adding type elements to the builtin type 'list' instance (line 162)
        # Adding element type (line 162)
        float_49795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 48), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 47), list_49794, float_49795)
        # Adding element type (line 162)
        float_49796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 53), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 47), list_49794, float_49796)
        
        keyword_49797 = list_49794
        kwargs_49798 = {'points': keyword_49797}
        # Getting the type of 'quad' (line 162)
        quad_49790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 20), 'quad', False)
        # Calling quad(args, kwargs) (line 162)
        quad_call_result_49799 = invoke(stypy.reporting.localization.Localization(__file__, 162, 20), quad_49790, *[myfunc_49791, int_49792, int_49793], **kwargs_49798)
        
        int_49800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 20), 'int')
        
        # Call to cos(...): (line 163)
        # Processing the call arguments (line 163)
        float_49802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 28), 'float')
        # Processing the call keyword arguments (line 163)
        kwargs_49803 = {}
        # Getting the type of 'cos' (line 163)
        cos_49801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 24), 'cos', False)
        # Calling cos(args, kwargs) (line 163)
        cos_call_result_49804 = invoke(stypy.reporting.localization.Localization(__file__, 163, 24), cos_49801, *[float_49802], **kwargs_49803)
        
        # Applying the binary operator '-' (line 163)
        result_sub_49805 = python_operator(stypy.reporting.localization.Localization(__file__, 163, 20), '-', int_49800, cos_call_result_49804)
        
        
        # Call to exp(...): (line 163)
        # Processing the call arguments (line 163)
        float_49807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 39), 'float')
        # Processing the call keyword arguments (line 163)
        kwargs_49808 = {}
        # Getting the type of 'exp' (line 163)
        exp_49806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 35), 'exp', False)
        # Calling exp(args, kwargs) (line 163)
        exp_call_result_49809 = invoke(stypy.reporting.localization.Localization(__file__, 163, 35), exp_49806, *[float_49807], **kwargs_49808)
        
        # Applying the binary operator '+' (line 163)
        result_add_49810 = python_operator(stypy.reporting.localization.Localization(__file__, 163, 33), '+', result_sub_49805, exp_call_result_49809)
        
        
        # Call to exp(...): (line 163)
        # Processing the call arguments (line 163)
        float_49812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 51), 'float')
        # Processing the call keyword arguments (line 163)
        kwargs_49813 = {}
        # Getting the type of 'exp' (line 163)
        exp_49811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 47), 'exp', False)
        # Calling exp(args, kwargs) (line 163)
        exp_call_result_49814 = invoke(stypy.reporting.localization.Localization(__file__, 163, 47), exp_49811, *[float_49812], **kwargs_49813)
        
        # Applying the binary operator '-' (line 163)
        result_sub_49815 = python_operator(stypy.reporting.localization.Localization(__file__, 163, 45), '-', result_add_49810, exp_call_result_49814)
        
        # Processing the call keyword arguments (line 162)
        kwargs_49816 = {}
        # Getting the type of 'assert_quad' (line 162)
        assert_quad_49789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 8), 'assert_quad', False)
        # Calling assert_quad(args, kwargs) (line 162)
        assert_quad_call_result_49817 = invoke(stypy.reporting.localization.Localization(__file__, 162, 8), assert_quad_49789, *[quad_call_result_49799, result_sub_49815], **kwargs_49816)
        
        
        # ################# End of 'test_singular(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_singular' in the type store
        # Getting the type of 'stypy_return_type' (line 152)
        stypy_return_type_49818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_49818)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_singular'
        return stypy_return_type_49818


    @norecursion
    def test_sine_weighted_finite(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_sine_weighted_finite'
        module_type_store = module_type_store.open_function_context('test_sine_weighted_finite', 165, 4, False)
        # Assigning a type to the variable 'self' (line 166)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestQuad.test_sine_weighted_finite.__dict__.__setitem__('stypy_localization', localization)
        TestQuad.test_sine_weighted_finite.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestQuad.test_sine_weighted_finite.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestQuad.test_sine_weighted_finite.__dict__.__setitem__('stypy_function_name', 'TestQuad.test_sine_weighted_finite')
        TestQuad.test_sine_weighted_finite.__dict__.__setitem__('stypy_param_names_list', [])
        TestQuad.test_sine_weighted_finite.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestQuad.test_sine_weighted_finite.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestQuad.test_sine_weighted_finite.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestQuad.test_sine_weighted_finite.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestQuad.test_sine_weighted_finite.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestQuad.test_sine_weighted_finite.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestQuad.test_sine_weighted_finite', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_sine_weighted_finite', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_sine_weighted_finite(...)' code ##################


        @norecursion
        def myfunc(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'myfunc'
            module_type_store = module_type_store.open_function_context('myfunc', 167, 8, False)
            
            # Passed parameters checking function
            myfunc.stypy_localization = localization
            myfunc.stypy_type_of_self = None
            myfunc.stypy_type_store = module_type_store
            myfunc.stypy_function_name = 'myfunc'
            myfunc.stypy_param_names_list = ['x', 'a']
            myfunc.stypy_varargs_param_name = None
            myfunc.stypy_kwargs_param_name = None
            myfunc.stypy_call_defaults = defaults
            myfunc.stypy_call_varargs = varargs
            myfunc.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'myfunc', ['x', 'a'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'myfunc', localization, ['x', 'a'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'myfunc(...)' code ##################

            
            # Call to exp(...): (line 168)
            # Processing the call arguments (line 168)
            # Getting the type of 'a' (line 168)
            a_49820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 23), 'a', False)
            # Getting the type of 'x' (line 168)
            x_49821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 26), 'x', False)
            int_49822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 28), 'int')
            # Applying the binary operator '-' (line 168)
            result_sub_49823 = python_operator(stypy.reporting.localization.Localization(__file__, 168, 26), '-', x_49821, int_49822)
            
            # Applying the binary operator '*' (line 168)
            result_mul_49824 = python_operator(stypy.reporting.localization.Localization(__file__, 168, 23), '*', a_49820, result_sub_49823)
            
            # Processing the call keyword arguments (line 168)
            kwargs_49825 = {}
            # Getting the type of 'exp' (line 168)
            exp_49819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 19), 'exp', False)
            # Calling exp(args, kwargs) (line 168)
            exp_call_result_49826 = invoke(stypy.reporting.localization.Localization(__file__, 168, 19), exp_49819, *[result_mul_49824], **kwargs_49825)
            
            # Assigning a type to the variable 'stypy_return_type' (line 168)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 12), 'stypy_return_type', exp_call_result_49826)
            
            # ################# End of 'myfunc(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'myfunc' in the type store
            # Getting the type of 'stypy_return_type' (line 167)
            stypy_return_type_49827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_49827)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'myfunc'
            return stypy_return_type_49827

        # Assigning a type to the variable 'myfunc' (line 167)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'myfunc', myfunc)
        
        # Assigning a BinOp to a Name (line 170):
        
        # Assigning a BinOp to a Name (line 170):
        float_49828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 14), 'float')
        float_49829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 19), 'float')
        # Applying the binary operator '**' (line 170)
        result_pow_49830 = python_operator(stypy.reporting.localization.Localization(__file__, 170, 14), '**', float_49828, float_49829)
        
        # Assigning a type to the variable 'ome' (line 170)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 8), 'ome', result_pow_49830)
        
        # Call to assert_quad(...): (line 171)
        # Processing the call arguments (line 171)
        
        # Call to quad(...): (line 171)
        # Processing the call arguments (line 171)
        # Getting the type of 'myfunc' (line 171)
        myfunc_49833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 25), 'myfunc', False)
        int_49834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 33), 'int')
        int_49835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 36), 'int')
        # Processing the call keyword arguments (line 171)
        int_49836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 44), 'int')
        keyword_49837 = int_49836
        str_49838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 55), 'str', 'sin')
        keyword_49839 = str_49838
        # Getting the type of 'ome' (line 171)
        ome_49840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 67), 'ome', False)
        keyword_49841 = ome_49840
        kwargs_49842 = {'args': keyword_49837, 'wvar': keyword_49841, 'weight': keyword_49839}
        # Getting the type of 'quad' (line 171)
        quad_49832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 20), 'quad', False)
        # Calling quad(args, kwargs) (line 171)
        quad_call_result_49843 = invoke(stypy.reporting.localization.Localization(__file__, 171, 20), quad_49832, *[myfunc_49833, int_49834, int_49835], **kwargs_49842)
        
        int_49844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 21), 'int')
        
        # Call to sin(...): (line 172)
        # Processing the call arguments (line 172)
        # Getting the type of 'ome' (line 172)
        ome_49846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 28), 'ome', False)
        # Processing the call keyword arguments (line 172)
        kwargs_49847 = {}
        # Getting the type of 'sin' (line 172)
        sin_49845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 24), 'sin', False)
        # Calling sin(args, kwargs) (line 172)
        sin_call_result_49848 = invoke(stypy.reporting.localization.Localization(__file__, 172, 24), sin_49845, *[ome_49846], **kwargs_49847)
        
        # Applying the binary operator '*' (line 172)
        result_mul_49849 = python_operator(stypy.reporting.localization.Localization(__file__, 172, 21), '*', int_49844, sin_call_result_49848)
        
        # Getting the type of 'ome' (line 172)
        ome_49850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 33), 'ome', False)
        
        # Call to cos(...): (line 172)
        # Processing the call arguments (line 172)
        # Getting the type of 'ome' (line 172)
        ome_49852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 41), 'ome', False)
        # Processing the call keyword arguments (line 172)
        kwargs_49853 = {}
        # Getting the type of 'cos' (line 172)
        cos_49851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 37), 'cos', False)
        # Calling cos(args, kwargs) (line 172)
        cos_call_result_49854 = invoke(stypy.reporting.localization.Localization(__file__, 172, 37), cos_49851, *[ome_49852], **kwargs_49853)
        
        # Applying the binary operator '*' (line 172)
        result_mul_49855 = python_operator(stypy.reporting.localization.Localization(__file__, 172, 33), '*', ome_49850, cos_call_result_49854)
        
        # Applying the binary operator '-' (line 172)
        result_sub_49856 = python_operator(stypy.reporting.localization.Localization(__file__, 172, 21), '-', result_mul_49849, result_mul_49855)
        
        # Getting the type of 'ome' (line 172)
        ome_49857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 46), 'ome', False)
        
        # Call to exp(...): (line 172)
        # Processing the call arguments (line 172)
        int_49859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 54), 'int')
        # Processing the call keyword arguments (line 172)
        kwargs_49860 = {}
        # Getting the type of 'exp' (line 172)
        exp_49858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 50), 'exp', False)
        # Calling exp(args, kwargs) (line 172)
        exp_call_result_49861 = invoke(stypy.reporting.localization.Localization(__file__, 172, 50), exp_49858, *[int_49859], **kwargs_49860)
        
        # Applying the binary operator '*' (line 172)
        result_mul_49862 = python_operator(stypy.reporting.localization.Localization(__file__, 172, 46), '*', ome_49857, exp_call_result_49861)
        
        # Applying the binary operator '+' (line 172)
        result_add_49863 = python_operator(stypy.reporting.localization.Localization(__file__, 172, 45), '+', result_sub_49856, result_mul_49862)
        
        int_49864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 61), 'int')
        int_49865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 65), 'int')
        # Applying the binary operator '**' (line 172)
        result_pow_49866 = python_operator(stypy.reporting.localization.Localization(__file__, 172, 61), '**', int_49864, int_49865)
        
        # Getting the type of 'ome' (line 172)
        ome_49867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 69), 'ome', False)
        int_49868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 74), 'int')
        # Applying the binary operator '**' (line 172)
        result_pow_49869 = python_operator(stypy.reporting.localization.Localization(__file__, 172, 69), '**', ome_49867, int_49868)
        
        # Applying the binary operator '+' (line 172)
        result_add_49870 = python_operator(stypy.reporting.localization.Localization(__file__, 172, 61), '+', result_pow_49866, result_pow_49869)
        
        # Applying the binary operator 'div' (line 172)
        result_div_49871 = python_operator(stypy.reporting.localization.Localization(__file__, 172, 20), 'div', result_add_49863, result_add_49870)
        
        # Processing the call keyword arguments (line 171)
        kwargs_49872 = {}
        # Getting the type of 'assert_quad' (line 171)
        assert_quad_49831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 8), 'assert_quad', False)
        # Calling assert_quad(args, kwargs) (line 171)
        assert_quad_call_result_49873 = invoke(stypy.reporting.localization.Localization(__file__, 171, 8), assert_quad_49831, *[quad_call_result_49843, result_div_49871], **kwargs_49872)
        
        
        # ################# End of 'test_sine_weighted_finite(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_sine_weighted_finite' in the type store
        # Getting the type of 'stypy_return_type' (line 165)
        stypy_return_type_49874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_49874)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_sine_weighted_finite'
        return stypy_return_type_49874


    @norecursion
    def test_sine_weighted_infinite(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_sine_weighted_infinite'
        module_type_store = module_type_store.open_function_context('test_sine_weighted_infinite', 174, 4, False)
        # Assigning a type to the variable 'self' (line 175)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestQuad.test_sine_weighted_infinite.__dict__.__setitem__('stypy_localization', localization)
        TestQuad.test_sine_weighted_infinite.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestQuad.test_sine_weighted_infinite.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestQuad.test_sine_weighted_infinite.__dict__.__setitem__('stypy_function_name', 'TestQuad.test_sine_weighted_infinite')
        TestQuad.test_sine_weighted_infinite.__dict__.__setitem__('stypy_param_names_list', [])
        TestQuad.test_sine_weighted_infinite.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestQuad.test_sine_weighted_infinite.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestQuad.test_sine_weighted_infinite.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestQuad.test_sine_weighted_infinite.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestQuad.test_sine_weighted_infinite.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestQuad.test_sine_weighted_infinite.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestQuad.test_sine_weighted_infinite', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_sine_weighted_infinite', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_sine_weighted_infinite(...)' code ##################


        @norecursion
        def myfunc(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'myfunc'
            module_type_store = module_type_store.open_function_context('myfunc', 176, 8, False)
            
            # Passed parameters checking function
            myfunc.stypy_localization = localization
            myfunc.stypy_type_of_self = None
            myfunc.stypy_type_store = module_type_store
            myfunc.stypy_function_name = 'myfunc'
            myfunc.stypy_param_names_list = ['x', 'a']
            myfunc.stypy_varargs_param_name = None
            myfunc.stypy_kwargs_param_name = None
            myfunc.stypy_call_defaults = defaults
            myfunc.stypy_call_varargs = varargs
            myfunc.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'myfunc', ['x', 'a'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'myfunc', localization, ['x', 'a'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'myfunc(...)' code ##################

            
            # Call to exp(...): (line 177)
            # Processing the call arguments (line 177)
            
            # Getting the type of 'x' (line 177)
            x_49876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 24), 'x', False)
            # Applying the 'usub' unary operator (line 177)
            result___neg___49877 = python_operator(stypy.reporting.localization.Localization(__file__, 177, 23), 'usub', x_49876)
            
            # Getting the type of 'a' (line 177)
            a_49878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 26), 'a', False)
            # Applying the binary operator '*' (line 177)
            result_mul_49879 = python_operator(stypy.reporting.localization.Localization(__file__, 177, 23), '*', result___neg___49877, a_49878)
            
            # Processing the call keyword arguments (line 177)
            kwargs_49880 = {}
            # Getting the type of 'exp' (line 177)
            exp_49875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 19), 'exp', False)
            # Calling exp(args, kwargs) (line 177)
            exp_call_result_49881 = invoke(stypy.reporting.localization.Localization(__file__, 177, 19), exp_49875, *[result_mul_49879], **kwargs_49880)
            
            # Assigning a type to the variable 'stypy_return_type' (line 177)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 12), 'stypy_return_type', exp_call_result_49881)
            
            # ################# End of 'myfunc(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'myfunc' in the type store
            # Getting the type of 'stypy_return_type' (line 176)
            stypy_return_type_49882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_49882)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'myfunc'
            return stypy_return_type_49882

        # Assigning a type to the variable 'myfunc' (line 176)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'myfunc', myfunc)
        
        # Assigning a Num to a Name (line 179):
        
        # Assigning a Num to a Name (line 179):
        float_49883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 12), 'float')
        # Assigning a type to the variable 'a' (line 179)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 8), 'a', float_49883)
        
        # Assigning a Num to a Name (line 180):
        
        # Assigning a Num to a Name (line 180):
        float_49884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 14), 'float')
        # Assigning a type to the variable 'ome' (line 180)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 8), 'ome', float_49884)
        
        # Call to assert_quad(...): (line 181)
        # Processing the call arguments (line 181)
        
        # Call to quad(...): (line 181)
        # Processing the call arguments (line 181)
        # Getting the type of 'myfunc' (line 181)
        myfunc_49887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 25), 'myfunc', False)
        int_49888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 33), 'int')
        # Getting the type of 'Inf' (line 181)
        Inf_49889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 36), 'Inf', False)
        # Processing the call keyword arguments (line 181)
        # Getting the type of 'a' (line 181)
        a_49890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 46), 'a', False)
        keyword_49891 = a_49890
        str_49892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 56), 'str', 'sin')
        keyword_49893 = str_49892
        # Getting the type of 'ome' (line 181)
        ome_49894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 68), 'ome', False)
        keyword_49895 = ome_49894
        kwargs_49896 = {'args': keyword_49891, 'wvar': keyword_49895, 'weight': keyword_49893}
        # Getting the type of 'quad' (line 181)
        quad_49886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 20), 'quad', False)
        # Calling quad(args, kwargs) (line 181)
        quad_call_result_49897 = invoke(stypy.reporting.localization.Localization(__file__, 181, 20), quad_49886, *[myfunc_49887, int_49888, Inf_49889], **kwargs_49896)
        
        # Getting the type of 'ome' (line 182)
        ome_49898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 20), 'ome', False)
        # Getting the type of 'a' (line 182)
        a_49899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 25), 'a', False)
        int_49900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 28), 'int')
        # Applying the binary operator '**' (line 182)
        result_pow_49901 = python_operator(stypy.reporting.localization.Localization(__file__, 182, 25), '**', a_49899, int_49900)
        
        # Getting the type of 'ome' (line 182)
        ome_49902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 32), 'ome', False)
        int_49903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 37), 'int')
        # Applying the binary operator '**' (line 182)
        result_pow_49904 = python_operator(stypy.reporting.localization.Localization(__file__, 182, 32), '**', ome_49902, int_49903)
        
        # Applying the binary operator '+' (line 182)
        result_add_49905 = python_operator(stypy.reporting.localization.Localization(__file__, 182, 25), '+', result_pow_49901, result_pow_49904)
        
        # Applying the binary operator 'div' (line 182)
        result_div_49906 = python_operator(stypy.reporting.localization.Localization(__file__, 182, 20), 'div', ome_49898, result_add_49905)
        
        # Processing the call keyword arguments (line 181)
        kwargs_49907 = {}
        # Getting the type of 'assert_quad' (line 181)
        assert_quad_49885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'assert_quad', False)
        # Calling assert_quad(args, kwargs) (line 181)
        assert_quad_call_result_49908 = invoke(stypy.reporting.localization.Localization(__file__, 181, 8), assert_quad_49885, *[quad_call_result_49897, result_div_49906], **kwargs_49907)
        
        
        # ################# End of 'test_sine_weighted_infinite(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_sine_weighted_infinite' in the type store
        # Getting the type of 'stypy_return_type' (line 174)
        stypy_return_type_49909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_49909)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_sine_weighted_infinite'
        return stypy_return_type_49909


    @norecursion
    def test_cosine_weighted_infinite(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_cosine_weighted_infinite'
        module_type_store = module_type_store.open_function_context('test_cosine_weighted_infinite', 184, 4, False)
        # Assigning a type to the variable 'self' (line 185)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestQuad.test_cosine_weighted_infinite.__dict__.__setitem__('stypy_localization', localization)
        TestQuad.test_cosine_weighted_infinite.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestQuad.test_cosine_weighted_infinite.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestQuad.test_cosine_weighted_infinite.__dict__.__setitem__('stypy_function_name', 'TestQuad.test_cosine_weighted_infinite')
        TestQuad.test_cosine_weighted_infinite.__dict__.__setitem__('stypy_param_names_list', [])
        TestQuad.test_cosine_weighted_infinite.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestQuad.test_cosine_weighted_infinite.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestQuad.test_cosine_weighted_infinite.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestQuad.test_cosine_weighted_infinite.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestQuad.test_cosine_weighted_infinite.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestQuad.test_cosine_weighted_infinite.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestQuad.test_cosine_weighted_infinite', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_cosine_weighted_infinite', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_cosine_weighted_infinite(...)' code ##################


        @norecursion
        def myfunc(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'myfunc'
            module_type_store = module_type_store.open_function_context('myfunc', 186, 8, False)
            
            # Passed parameters checking function
            myfunc.stypy_localization = localization
            myfunc.stypy_type_of_self = None
            myfunc.stypy_type_store = module_type_store
            myfunc.stypy_function_name = 'myfunc'
            myfunc.stypy_param_names_list = ['x', 'a']
            myfunc.stypy_varargs_param_name = None
            myfunc.stypy_kwargs_param_name = None
            myfunc.stypy_call_defaults = defaults
            myfunc.stypy_call_varargs = varargs
            myfunc.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'myfunc', ['x', 'a'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'myfunc', localization, ['x', 'a'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'myfunc(...)' code ##################

            
            # Call to exp(...): (line 187)
            # Processing the call arguments (line 187)
            # Getting the type of 'x' (line 187)
            x_49911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 23), 'x', False)
            # Getting the type of 'a' (line 187)
            a_49912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 25), 'a', False)
            # Applying the binary operator '*' (line 187)
            result_mul_49913 = python_operator(stypy.reporting.localization.Localization(__file__, 187, 23), '*', x_49911, a_49912)
            
            # Processing the call keyword arguments (line 187)
            kwargs_49914 = {}
            # Getting the type of 'exp' (line 187)
            exp_49910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 19), 'exp', False)
            # Calling exp(args, kwargs) (line 187)
            exp_call_result_49915 = invoke(stypy.reporting.localization.Localization(__file__, 187, 19), exp_49910, *[result_mul_49913], **kwargs_49914)
            
            # Assigning a type to the variable 'stypy_return_type' (line 187)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 12), 'stypy_return_type', exp_call_result_49915)
            
            # ################# End of 'myfunc(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'myfunc' in the type store
            # Getting the type of 'stypy_return_type' (line 186)
            stypy_return_type_49916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_49916)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'myfunc'
            return stypy_return_type_49916

        # Assigning a type to the variable 'myfunc' (line 186)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 8), 'myfunc', myfunc)
        
        # Assigning a Num to a Name (line 189):
        
        # Assigning a Num to a Name (line 189):
        float_49917 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 12), 'float')
        # Assigning a type to the variable 'a' (line 189)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 8), 'a', float_49917)
        
        # Assigning a Num to a Name (line 190):
        
        # Assigning a Num to a Name (line 190):
        float_49918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 14), 'float')
        # Assigning a type to the variable 'ome' (line 190)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'ome', float_49918)
        
        # Call to assert_quad(...): (line 191)
        # Processing the call arguments (line 191)
        
        # Call to quad(...): (line 191)
        # Processing the call arguments (line 191)
        # Getting the type of 'myfunc' (line 191)
        myfunc_49921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 25), 'myfunc', False)
        
        # Getting the type of 'Inf' (line 191)
        Inf_49922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 34), 'Inf', False)
        # Applying the 'usub' unary operator (line 191)
        result___neg___49923 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 33), 'usub', Inf_49922)
        
        int_49924 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 39), 'int')
        # Processing the call keyword arguments (line 191)
        # Getting the type of 'a' (line 191)
        a_49925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 47), 'a', False)
        keyword_49926 = a_49925
        str_49927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 57), 'str', 'cos')
        keyword_49928 = str_49927
        # Getting the type of 'ome' (line 191)
        ome_49929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 69), 'ome', False)
        keyword_49930 = ome_49929
        kwargs_49931 = {'args': keyword_49926, 'wvar': keyword_49930, 'weight': keyword_49928}
        # Getting the type of 'quad' (line 191)
        quad_49920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 20), 'quad', False)
        # Calling quad(args, kwargs) (line 191)
        quad_call_result_49932 = invoke(stypy.reporting.localization.Localization(__file__, 191, 20), quad_49920, *[myfunc_49921, result___neg___49923, int_49924], **kwargs_49931)
        
        # Getting the type of 'a' (line 192)
        a_49933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 20), 'a', False)
        # Getting the type of 'a' (line 192)
        a_49934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 23), 'a', False)
        int_49935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 26), 'int')
        # Applying the binary operator '**' (line 192)
        result_pow_49936 = python_operator(stypy.reporting.localization.Localization(__file__, 192, 23), '**', a_49934, int_49935)
        
        # Getting the type of 'ome' (line 192)
        ome_49937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 30), 'ome', False)
        int_49938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 35), 'int')
        # Applying the binary operator '**' (line 192)
        result_pow_49939 = python_operator(stypy.reporting.localization.Localization(__file__, 192, 30), '**', ome_49937, int_49938)
        
        # Applying the binary operator '+' (line 192)
        result_add_49940 = python_operator(stypy.reporting.localization.Localization(__file__, 192, 23), '+', result_pow_49936, result_pow_49939)
        
        # Applying the binary operator 'div' (line 192)
        result_div_49941 = python_operator(stypy.reporting.localization.Localization(__file__, 192, 20), 'div', a_49933, result_add_49940)
        
        # Processing the call keyword arguments (line 191)
        kwargs_49942 = {}
        # Getting the type of 'assert_quad' (line 191)
        assert_quad_49919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 8), 'assert_quad', False)
        # Calling assert_quad(args, kwargs) (line 191)
        assert_quad_call_result_49943 = invoke(stypy.reporting.localization.Localization(__file__, 191, 8), assert_quad_49919, *[quad_call_result_49932, result_div_49941], **kwargs_49942)
        
        
        # ################# End of 'test_cosine_weighted_infinite(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_cosine_weighted_infinite' in the type store
        # Getting the type of 'stypy_return_type' (line 184)
        stypy_return_type_49944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_49944)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_cosine_weighted_infinite'
        return stypy_return_type_49944


    @norecursion
    def test_algebraic_log_weight(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_algebraic_log_weight'
        module_type_store = module_type_store.open_function_context('test_algebraic_log_weight', 194, 4, False)
        # Assigning a type to the variable 'self' (line 195)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestQuad.test_algebraic_log_weight.__dict__.__setitem__('stypy_localization', localization)
        TestQuad.test_algebraic_log_weight.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestQuad.test_algebraic_log_weight.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestQuad.test_algebraic_log_weight.__dict__.__setitem__('stypy_function_name', 'TestQuad.test_algebraic_log_weight')
        TestQuad.test_algebraic_log_weight.__dict__.__setitem__('stypy_param_names_list', [])
        TestQuad.test_algebraic_log_weight.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestQuad.test_algebraic_log_weight.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestQuad.test_algebraic_log_weight.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestQuad.test_algebraic_log_weight.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestQuad.test_algebraic_log_weight.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestQuad.test_algebraic_log_weight.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestQuad.test_algebraic_log_weight', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_algebraic_log_weight', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_algebraic_log_weight(...)' code ##################


        @norecursion
        def myfunc(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'myfunc'
            module_type_store = module_type_store.open_function_context('myfunc', 196, 8, False)
            
            # Passed parameters checking function
            myfunc.stypy_localization = localization
            myfunc.stypy_type_of_self = None
            myfunc.stypy_type_store = module_type_store
            myfunc.stypy_function_name = 'myfunc'
            myfunc.stypy_param_names_list = ['x', 'a']
            myfunc.stypy_varargs_param_name = None
            myfunc.stypy_kwargs_param_name = None
            myfunc.stypy_call_defaults = defaults
            myfunc.stypy_call_varargs = varargs
            myfunc.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'myfunc', ['x', 'a'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'myfunc', localization, ['x', 'a'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'myfunc(...)' code ##################

            int_49945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 19), 'int')
            int_49946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 22), 'int')
            # Getting the type of 'x' (line 197)
            x_49947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 24), 'x')
            # Applying the binary operator '+' (line 197)
            result_add_49948 = python_operator(stypy.reporting.localization.Localization(__file__, 197, 22), '+', int_49946, x_49947)
            
            int_49949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 26), 'int')
            
            # Getting the type of 'a' (line 197)
            a_49950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 31), 'a')
            # Applying the 'usub' unary operator (line 197)
            result___neg___49951 = python_operator(stypy.reporting.localization.Localization(__file__, 197, 30), 'usub', a_49950)
            
            # Applying the binary operator '**' (line 197)
            result_pow_49952 = python_operator(stypy.reporting.localization.Localization(__file__, 197, 26), '**', int_49949, result___neg___49951)
            
            # Applying the binary operator '+' (line 197)
            result_add_49953 = python_operator(stypy.reporting.localization.Localization(__file__, 197, 25), '+', result_add_49948, result_pow_49952)
            
            # Applying the binary operator 'div' (line 197)
            result_div_49954 = python_operator(stypy.reporting.localization.Localization(__file__, 197, 19), 'div', int_49945, result_add_49953)
            
            # Assigning a type to the variable 'stypy_return_type' (line 197)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 12), 'stypy_return_type', result_div_49954)
            
            # ################# End of 'myfunc(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'myfunc' in the type store
            # Getting the type of 'stypy_return_type' (line 196)
            stypy_return_type_49955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_49955)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'myfunc'
            return stypy_return_type_49955

        # Assigning a type to the variable 'myfunc' (line 196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'myfunc', myfunc)
        
        # Assigning a Num to a Name (line 199):
        
        # Assigning a Num to a Name (line 199):
        float_49956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 12), 'float')
        # Assigning a type to the variable 'a' (line 199)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'a', float_49956)
        
        # Call to assert_quad(...): (line 200)
        # Processing the call arguments (line 200)
        
        # Call to quad(...): (line 200)
        # Processing the call arguments (line 200)
        # Getting the type of 'myfunc' (line 200)
        myfunc_49959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 25), 'myfunc', False)
        int_49960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 33), 'int')
        int_49961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 37), 'int')
        # Processing the call keyword arguments (line 200)
        # Getting the type of 'a' (line 200)
        a_49962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 45), 'a', False)
        keyword_49963 = a_49962
        str_49964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 55), 'str', 'alg')
        keyword_49965 = str_49964
        
        # Obtaining an instance of the builtin type 'tuple' (line 201)
        tuple_49966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 31), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 201)
        # Adding element type (line 201)
        float_49967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 31), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 201, 31), tuple_49966, float_49967)
        # Adding element type (line 201)
        float_49968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 37), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 201, 31), tuple_49966, float_49968)
        
        keyword_49969 = tuple_49966
        kwargs_49970 = {'args': keyword_49963, 'wvar': keyword_49969, 'weight': keyword_49965}
        # Getting the type of 'quad' (line 200)
        quad_49958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 20), 'quad', False)
        # Calling quad(args, kwargs) (line 200)
        quad_call_result_49971 = invoke(stypy.reporting.localization.Localization(__file__, 200, 20), quad_49958, *[myfunc_49959, int_49960, int_49961], **kwargs_49970)
        
        # Getting the type of 'pi' (line 202)
        pi_49972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 20), 'pi', False)
        
        # Call to sqrt(...): (line 202)
        # Processing the call arguments (line 202)
        int_49974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 29), 'int')
        int_49975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 31), 'int')
        
        # Getting the type of 'a' (line 202)
        a_49976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 36), 'a', False)
        # Applying the 'usub' unary operator (line 202)
        result___neg___49977 = python_operator(stypy.reporting.localization.Localization(__file__, 202, 35), 'usub', a_49976)
        
        # Applying the binary operator '**' (line 202)
        result_pow_49978 = python_operator(stypy.reporting.localization.Localization(__file__, 202, 31), '**', int_49975, result___neg___49977)
        
        # Applying the binary operator '+' (line 202)
        result_add_49979 = python_operator(stypy.reporting.localization.Localization(__file__, 202, 29), '+', int_49974, result_pow_49978)
        
        int_49980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 41), 'int')
        # Applying the binary operator '**' (line 202)
        result_pow_49981 = python_operator(stypy.reporting.localization.Localization(__file__, 202, 28), '**', result_add_49979, int_49980)
        
        int_49982 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 45), 'int')
        # Applying the binary operator '-' (line 202)
        result_sub_49983 = python_operator(stypy.reporting.localization.Localization(__file__, 202, 28), '-', result_pow_49981, int_49982)
        
        # Processing the call keyword arguments (line 202)
        kwargs_49984 = {}
        # Getting the type of 'sqrt' (line 202)
        sqrt_49973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 23), 'sqrt', False)
        # Calling sqrt(args, kwargs) (line 202)
        sqrt_call_result_49985 = invoke(stypy.reporting.localization.Localization(__file__, 202, 23), sqrt_49973, *[result_sub_49983], **kwargs_49984)
        
        # Applying the binary operator 'div' (line 202)
        result_div_49986 = python_operator(stypy.reporting.localization.Localization(__file__, 202, 20), 'div', pi_49972, sqrt_call_result_49985)
        
        # Processing the call keyword arguments (line 200)
        kwargs_49987 = {}
        # Getting the type of 'assert_quad' (line 200)
        assert_quad_49957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'assert_quad', False)
        # Calling assert_quad(args, kwargs) (line 200)
        assert_quad_call_result_49988 = invoke(stypy.reporting.localization.Localization(__file__, 200, 8), assert_quad_49957, *[quad_call_result_49971, result_div_49986], **kwargs_49987)
        
        
        # ################# End of 'test_algebraic_log_weight(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_algebraic_log_weight' in the type store
        # Getting the type of 'stypy_return_type' (line 194)
        stypy_return_type_49989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_49989)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_algebraic_log_weight'
        return stypy_return_type_49989


    @norecursion
    def test_cauchypv_weight(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_cauchypv_weight'
        module_type_store = module_type_store.open_function_context('test_cauchypv_weight', 204, 4, False)
        # Assigning a type to the variable 'self' (line 205)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestQuad.test_cauchypv_weight.__dict__.__setitem__('stypy_localization', localization)
        TestQuad.test_cauchypv_weight.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestQuad.test_cauchypv_weight.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestQuad.test_cauchypv_weight.__dict__.__setitem__('stypy_function_name', 'TestQuad.test_cauchypv_weight')
        TestQuad.test_cauchypv_weight.__dict__.__setitem__('stypy_param_names_list', [])
        TestQuad.test_cauchypv_weight.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestQuad.test_cauchypv_weight.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestQuad.test_cauchypv_weight.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestQuad.test_cauchypv_weight.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestQuad.test_cauchypv_weight.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestQuad.test_cauchypv_weight.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestQuad.test_cauchypv_weight', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_cauchypv_weight', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_cauchypv_weight(...)' code ##################


        @norecursion
        def myfunc(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'myfunc'
            module_type_store = module_type_store.open_function_context('myfunc', 206, 8, False)
            
            # Passed parameters checking function
            myfunc.stypy_localization = localization
            myfunc.stypy_type_of_self = None
            myfunc.stypy_type_store = module_type_store
            myfunc.stypy_function_name = 'myfunc'
            myfunc.stypy_param_names_list = ['x', 'a']
            myfunc.stypy_varargs_param_name = None
            myfunc.stypy_kwargs_param_name = None
            myfunc.stypy_call_defaults = defaults
            myfunc.stypy_call_varargs = varargs
            myfunc.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'myfunc', ['x', 'a'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'myfunc', localization, ['x', 'a'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'myfunc(...)' code ##################

            float_49990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 19), 'float')
            
            # Getting the type of 'a' (line 207)
            a_49991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 26), 'a')
            # Applying the 'usub' unary operator (line 207)
            result___neg___49992 = python_operator(stypy.reporting.localization.Localization(__file__, 207, 25), 'usub', a_49991)
            
            # Applying the binary operator '**' (line 207)
            result_pow_49993 = python_operator(stypy.reporting.localization.Localization(__file__, 207, 19), '**', float_49990, result___neg___49992)
            
            # Getting the type of 'x' (line 207)
            x_49994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 31), 'x')
            int_49995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 33), 'int')
            # Applying the binary operator '-' (line 207)
            result_sub_49996 = python_operator(stypy.reporting.localization.Localization(__file__, 207, 31), '-', x_49994, int_49995)
            
            int_49997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 37), 'int')
            # Applying the binary operator '**' (line 207)
            result_pow_49998 = python_operator(stypy.reporting.localization.Localization(__file__, 207, 30), '**', result_sub_49996, int_49997)
            
            float_49999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 39), 'float')
            
            # Getting the type of 'a' (line 207)
            a_50000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 46), 'a')
            # Applying the 'usub' unary operator (line 207)
            result___neg___50001 = python_operator(stypy.reporting.localization.Localization(__file__, 207, 45), 'usub', a_50000)
            
            # Applying the binary operator '**' (line 207)
            result_pow_50002 = python_operator(stypy.reporting.localization.Localization(__file__, 207, 39), '**', float_49999, result___neg___50001)
            
            # Applying the binary operator '+' (line 207)
            result_add_50003 = python_operator(stypy.reporting.localization.Localization(__file__, 207, 30), '+', result_pow_49998, result_pow_50002)
            
            # Applying the binary operator 'div' (line 207)
            result_div_50004 = python_operator(stypy.reporting.localization.Localization(__file__, 207, 19), 'div', result_pow_49993, result_add_50003)
            
            # Assigning a type to the variable 'stypy_return_type' (line 207)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 12), 'stypy_return_type', result_div_50004)
            
            # ################# End of 'myfunc(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'myfunc' in the type store
            # Getting the type of 'stypy_return_type' (line 206)
            stypy_return_type_50005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_50005)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'myfunc'
            return stypy_return_type_50005

        # Assigning a type to the variable 'myfunc' (line 206)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 8), 'myfunc', myfunc)
        
        # Assigning a Num to a Name (line 209):
        
        # Assigning a Num to a Name (line 209):
        float_50006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 12), 'float')
        # Assigning a type to the variable 'a' (line 209)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 8), 'a', float_50006)
        
        # Assigning a BinOp to a Name (line 210):
        
        # Assigning a BinOp to a Name (line 210):
        float_50007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 24), 'float')
        float_50008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 30), 'float')
        # Applying the binary operator '**' (line 210)
        result_pow_50009 = python_operator(stypy.reporting.localization.Localization(__file__, 210, 24), '**', float_50007, float_50008)
        
        
        # Call to log(...): (line 210)
        # Processing the call arguments (line 210)
        float_50011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 40), 'float')
        # Processing the call keyword arguments (line 210)
        kwargs_50012 = {}
        # Getting the type of 'log' (line 210)
        log_50010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 36), 'log', False)
        # Calling log(args, kwargs) (line 210)
        log_call_result_50013 = invoke(stypy.reporting.localization.Localization(__file__, 210, 36), log_50010, *[float_50011], **kwargs_50012)
        
        # Applying the binary operator '*' (line 210)
        result_mul_50014 = python_operator(stypy.reporting.localization.Localization(__file__, 210, 24), '*', result_pow_50009, log_call_result_50013)
        
        float_50015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 24), 'float')
        float_50016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 30), 'float')
        # Applying the binary operator '**' (line 211)
        result_pow_50017 = python_operator(stypy.reporting.localization.Localization(__file__, 211, 24), '**', float_50015, float_50016)
        
        
        # Call to log(...): (line 211)
        # Processing the call arguments (line 211)
        float_50019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 41), 'float')
        
        # Getting the type of 'a' (line 211)
        a_50020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 48), 'a', False)
        # Applying the 'usub' unary operator (line 211)
        result___neg___50021 = python_operator(stypy.reporting.localization.Localization(__file__, 211, 47), 'usub', a_50020)
        
        # Applying the binary operator '**' (line 211)
        result_pow_50022 = python_operator(stypy.reporting.localization.Localization(__file__, 211, 41), '**', float_50019, result___neg___50021)
        
        int_50023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 51), 'int')
        # Applying the binary operator '+' (line 211)
        result_add_50024 = python_operator(stypy.reporting.localization.Localization(__file__, 211, 41), '+', result_pow_50022, int_50023)
        
        float_50025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 58), 'float')
        
        # Getting the type of 'a' (line 211)
        a_50026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 65), 'a', False)
        # Applying the 'usub' unary operator (line 211)
        result___neg___50027 = python_operator(stypy.reporting.localization.Localization(__file__, 211, 64), 'usub', a_50026)
        
        # Applying the binary operator '**' (line 211)
        result_pow_50028 = python_operator(stypy.reporting.localization.Localization(__file__, 211, 58), '**', float_50025, result___neg___50027)
        
        int_50029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 68), 'int')
        # Applying the binary operator '+' (line 211)
        result_add_50030 = python_operator(stypy.reporting.localization.Localization(__file__, 211, 58), '+', result_pow_50028, int_50029)
        
        # Applying the binary operator 'div' (line 211)
        result_div_50031 = python_operator(stypy.reporting.localization.Localization(__file__, 211, 40), 'div', result_add_50024, result_add_50030)
        
        # Processing the call keyword arguments (line 211)
        kwargs_50032 = {}
        # Getting the type of 'log' (line 211)
        log_50018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 36), 'log', False)
        # Calling log(args, kwargs) (line 211)
        log_call_result_50033 = invoke(stypy.reporting.localization.Localization(__file__, 211, 36), log_50018, *[result_div_50031], **kwargs_50032)
        
        # Applying the binary operator '*' (line 211)
        result_mul_50034 = python_operator(stypy.reporting.localization.Localization(__file__, 211, 24), '*', result_pow_50017, log_call_result_50033)
        
        # Applying the binary operator '-' (line 210)
        result_sub_50035 = python_operator(stypy.reporting.localization.Localization(__file__, 210, 24), '-', result_mul_50014, result_mul_50034)
        
        
        # Call to arctan(...): (line 212)
        # Processing the call arguments (line 212)
        float_50037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 31), 'float')
        # Getting the type of 'a' (line 212)
        a_50038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 37), 'a', False)
        int_50039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 39), 'int')
        # Applying the binary operator '+' (line 212)
        result_add_50040 = python_operator(stypy.reporting.localization.Localization(__file__, 212, 37), '+', a_50038, int_50039)
        
        # Applying the binary operator '**' (line 212)
        result_pow_50041 = python_operator(stypy.reporting.localization.Localization(__file__, 212, 31), '**', float_50037, result_add_50040)
        
        # Processing the call keyword arguments (line 212)
        kwargs_50042 = {}
        # Getting the type of 'arctan' (line 212)
        arctan_50036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 24), 'arctan', False)
        # Calling arctan(args, kwargs) (line 212)
        arctan_call_result_50043 = invoke(stypy.reporting.localization.Localization(__file__, 212, 24), arctan_50036, *[result_pow_50041], **kwargs_50042)
        
        # Applying the binary operator '-' (line 211)
        result_sub_50044 = python_operator(stypy.reporting.localization.Localization(__file__, 211, 72), '-', result_sub_50035, arctan_call_result_50043)
        
        
        # Call to arctan(...): (line 213)
        # Processing the call arguments (line 213)
        float_50046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 31), 'float')
        # Getting the type of 'a' (line 213)
        a_50047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 36), 'a', False)
        # Applying the binary operator '**' (line 213)
        result_pow_50048 = python_operator(stypy.reporting.localization.Localization(__file__, 213, 31), '**', float_50046, a_50047)
        
        # Processing the call keyword arguments (line 213)
        kwargs_50049 = {}
        # Getting the type of 'arctan' (line 213)
        arctan_50045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 24), 'arctan', False)
        # Calling arctan(args, kwargs) (line 213)
        arctan_call_result_50050 = invoke(stypy.reporting.localization.Localization(__file__, 213, 24), arctan_50045, *[result_pow_50048], **kwargs_50049)
        
        # Applying the binary operator '-' (line 212)
        result_sub_50051 = python_operator(stypy.reporting.localization.Localization(__file__, 212, 43), '-', result_sub_50044, arctan_call_result_50050)
        
        float_50052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 24), 'float')
        
        # Getting the type of 'a' (line 214)
        a_50053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 31), 'a')
        # Applying the 'usub' unary operator (line 214)
        result___neg___50054 = python_operator(stypy.reporting.localization.Localization(__file__, 214, 30), 'usub', a_50053)
        
        # Applying the binary operator '**' (line 214)
        result_pow_50055 = python_operator(stypy.reporting.localization.Localization(__file__, 214, 24), '**', float_50052, result___neg___50054)
        
        int_50056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 36), 'int')
        # Applying the binary operator '+' (line 214)
        result_add_50057 = python_operator(stypy.reporting.localization.Localization(__file__, 214, 24), '+', result_pow_50055, int_50056)
        
        # Applying the binary operator 'div' (line 210)
        result_div_50058 = python_operator(stypy.reporting.localization.Localization(__file__, 210, 23), 'div', result_sub_50051, result_add_50057)
        
        # Assigning a type to the variable 'tabledValue' (line 210)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 8), 'tabledValue', result_div_50058)
        
        # Call to assert_quad(...): (line 215)
        # Processing the call arguments (line 215)
        
        # Call to quad(...): (line 215)
        # Processing the call arguments (line 215)
        # Getting the type of 'myfunc' (line 215)
        myfunc_50061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 25), 'myfunc', False)
        int_50062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 33), 'int')
        int_50063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 36), 'int')
        # Processing the call keyword arguments (line 215)
        float_50064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 44), 'float')
        keyword_50065 = float_50064
        str_50066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 56), 'str', 'cauchy')
        keyword_50067 = str_50066
        float_50068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 71), 'float')
        keyword_50069 = float_50068
        kwargs_50070 = {'args': keyword_50065, 'wvar': keyword_50069, 'weight': keyword_50067}
        # Getting the type of 'quad' (line 215)
        quad_50060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 20), 'quad', False)
        # Calling quad(args, kwargs) (line 215)
        quad_call_result_50071 = invoke(stypy.reporting.localization.Localization(__file__, 215, 20), quad_50060, *[myfunc_50061, int_50062, int_50063], **kwargs_50070)
        
        # Getting the type of 'tabledValue' (line 216)
        tabledValue_50072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 20), 'tabledValue', False)
        # Processing the call keyword arguments (line 215)
        float_50073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 40), 'float')
        keyword_50074 = float_50073
        kwargs_50075 = {'errTol': keyword_50074}
        # Getting the type of 'assert_quad' (line 215)
        assert_quad_50059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 8), 'assert_quad', False)
        # Calling assert_quad(args, kwargs) (line 215)
        assert_quad_call_result_50076 = invoke(stypy.reporting.localization.Localization(__file__, 215, 8), assert_quad_50059, *[quad_call_result_50071, tabledValue_50072], **kwargs_50075)
        
        
        # ################# End of 'test_cauchypv_weight(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_cauchypv_weight' in the type store
        # Getting the type of 'stypy_return_type' (line 204)
        stypy_return_type_50077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_50077)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_cauchypv_weight'
        return stypy_return_type_50077


    @norecursion
    def test_double_integral(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_double_integral'
        module_type_store = module_type_store.open_function_context('test_double_integral', 218, 4, False)
        # Assigning a type to the variable 'self' (line 219)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestQuad.test_double_integral.__dict__.__setitem__('stypy_localization', localization)
        TestQuad.test_double_integral.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestQuad.test_double_integral.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestQuad.test_double_integral.__dict__.__setitem__('stypy_function_name', 'TestQuad.test_double_integral')
        TestQuad.test_double_integral.__dict__.__setitem__('stypy_param_names_list', [])
        TestQuad.test_double_integral.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestQuad.test_double_integral.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestQuad.test_double_integral.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestQuad.test_double_integral.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestQuad.test_double_integral.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestQuad.test_double_integral.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestQuad.test_double_integral', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_double_integral', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_double_integral(...)' code ##################


        @norecursion
        def simpfunc(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'simpfunc'
            module_type_store = module_type_store.open_function_context('simpfunc', 220, 8, False)
            
            # Passed parameters checking function
            simpfunc.stypy_localization = localization
            simpfunc.stypy_type_of_self = None
            simpfunc.stypy_type_store = module_type_store
            simpfunc.stypy_function_name = 'simpfunc'
            simpfunc.stypy_param_names_list = ['y', 'x']
            simpfunc.stypy_varargs_param_name = None
            simpfunc.stypy_kwargs_param_name = None
            simpfunc.stypy_call_defaults = defaults
            simpfunc.stypy_call_varargs = varargs
            simpfunc.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'simpfunc', ['y', 'x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'simpfunc', localization, ['y', 'x'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'simpfunc(...)' code ##################

            # Getting the type of 'x' (line 221)
            x_50078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 19), 'x')
            # Getting the type of 'y' (line 221)
            y_50079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 21), 'y')
            # Applying the binary operator '+' (line 221)
            result_add_50080 = python_operator(stypy.reporting.localization.Localization(__file__, 221, 19), '+', x_50078, y_50079)
            
            # Assigning a type to the variable 'stypy_return_type' (line 221)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 12), 'stypy_return_type', result_add_50080)
            
            # ################# End of 'simpfunc(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'simpfunc' in the type store
            # Getting the type of 'stypy_return_type' (line 220)
            stypy_return_type_50081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_50081)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'simpfunc'
            return stypy_return_type_50081

        # Assigning a type to the variable 'simpfunc' (line 220)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 8), 'simpfunc', simpfunc)
        
        # Assigning a Tuple to a Tuple (line 223):
        
        # Assigning a Num to a Name (line 223):
        float_50082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 15), 'float')
        # Assigning a type to the variable 'tuple_assignment_49280' (line 223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'tuple_assignment_49280', float_50082)
        
        # Assigning a Num to a Name (line 223):
        float_50083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 20), 'float')
        # Assigning a type to the variable 'tuple_assignment_49281' (line 223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'tuple_assignment_49281', float_50083)
        
        # Assigning a Name to a Name (line 223):
        # Getting the type of 'tuple_assignment_49280' (line 223)
        tuple_assignment_49280_50084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'tuple_assignment_49280')
        # Assigning a type to the variable 'a' (line 223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'a', tuple_assignment_49280_50084)
        
        # Assigning a Name to a Name (line 223):
        # Getting the type of 'tuple_assignment_49281' (line 223)
        tuple_assignment_49281_50085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'tuple_assignment_49281')
        # Assigning a type to the variable 'b' (line 223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 11), 'b', tuple_assignment_49281_50085)
        
        # Call to assert_quad(...): (line 224)
        # Processing the call arguments (line 224)
        
        # Call to dblquad(...): (line 224)
        # Processing the call arguments (line 224)
        # Getting the type of 'simpfunc' (line 224)
        simpfunc_50088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 28), 'simpfunc', False)
        # Getting the type of 'a' (line 224)
        a_50089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 38), 'a', False)
        # Getting the type of 'b' (line 224)
        b_50090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 41), 'b', False)

        @norecursion
        def _stypy_temp_lambda_26(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_26'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_26', 224, 44, True)
            # Passed parameters checking function
            _stypy_temp_lambda_26.stypy_localization = localization
            _stypy_temp_lambda_26.stypy_type_of_self = None
            _stypy_temp_lambda_26.stypy_type_store = module_type_store
            _stypy_temp_lambda_26.stypy_function_name = '_stypy_temp_lambda_26'
            _stypy_temp_lambda_26.stypy_param_names_list = ['x']
            _stypy_temp_lambda_26.stypy_varargs_param_name = None
            _stypy_temp_lambda_26.stypy_kwargs_param_name = None
            _stypy_temp_lambda_26.stypy_call_defaults = defaults
            _stypy_temp_lambda_26.stypy_call_varargs = varargs
            _stypy_temp_lambda_26.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_26', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_26', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            # Getting the type of 'x' (line 224)
            x_50091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 54), 'x', False)
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 224)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 44), 'stypy_return_type', x_50091)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_26' in the type store
            # Getting the type of 'stypy_return_type' (line 224)
            stypy_return_type_50092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 44), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_50092)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_26'
            return stypy_return_type_50092

        # Assigning a type to the variable '_stypy_temp_lambda_26' (line 224)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 44), '_stypy_temp_lambda_26', _stypy_temp_lambda_26)
        # Getting the type of '_stypy_temp_lambda_26' (line 224)
        _stypy_temp_lambda_26_50093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 44), '_stypy_temp_lambda_26')

        @norecursion
        def _stypy_temp_lambda_27(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_27'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_27', 224, 57, True)
            # Passed parameters checking function
            _stypy_temp_lambda_27.stypy_localization = localization
            _stypy_temp_lambda_27.stypy_type_of_self = None
            _stypy_temp_lambda_27.stypy_type_store = module_type_store
            _stypy_temp_lambda_27.stypy_function_name = '_stypy_temp_lambda_27'
            _stypy_temp_lambda_27.stypy_param_names_list = ['x']
            _stypy_temp_lambda_27.stypy_varargs_param_name = None
            _stypy_temp_lambda_27.stypy_kwargs_param_name = None
            _stypy_temp_lambda_27.stypy_call_defaults = defaults
            _stypy_temp_lambda_27.stypy_call_varargs = varargs
            _stypy_temp_lambda_27.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_27', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_27', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            int_50094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 67), 'int')
            # Getting the type of 'x' (line 224)
            x_50095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 69), 'x', False)
            # Applying the binary operator '*' (line 224)
            result_mul_50096 = python_operator(stypy.reporting.localization.Localization(__file__, 224, 67), '*', int_50094, x_50095)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 224)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 57), 'stypy_return_type', result_mul_50096)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_27' in the type store
            # Getting the type of 'stypy_return_type' (line 224)
            stypy_return_type_50097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 57), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_50097)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_27'
            return stypy_return_type_50097

        # Assigning a type to the variable '_stypy_temp_lambda_27' (line 224)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 57), '_stypy_temp_lambda_27', _stypy_temp_lambda_27)
        # Getting the type of '_stypy_temp_lambda_27' (line 224)
        _stypy_temp_lambda_27_50098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 57), '_stypy_temp_lambda_27')
        # Processing the call keyword arguments (line 224)
        kwargs_50099 = {}
        # Getting the type of 'dblquad' (line 224)
        dblquad_50087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 20), 'dblquad', False)
        # Calling dblquad(args, kwargs) (line 224)
        dblquad_call_result_50100 = invoke(stypy.reporting.localization.Localization(__file__, 224, 20), dblquad_50087, *[simpfunc_50088, a_50089, b_50090, _stypy_temp_lambda_26_50093, _stypy_temp_lambda_27_50098], **kwargs_50099)
        
        int_50101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 20), 'int')
        float_50102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 22), 'float')
        # Applying the binary operator 'div' (line 225)
        result_div_50103 = python_operator(stypy.reporting.localization.Localization(__file__, 225, 20), 'div', int_50101, float_50102)
        
        # Getting the type of 'b' (line 225)
        b_50104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 29), 'b', False)
        float_50105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 32), 'float')
        # Applying the binary operator '**' (line 225)
        result_pow_50106 = python_operator(stypy.reporting.localization.Localization(__file__, 225, 29), '**', b_50104, float_50105)
        
        # Getting the type of 'a' (line 225)
        a_50107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 36), 'a', False)
        float_50108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 39), 'float')
        # Applying the binary operator '**' (line 225)
        result_pow_50109 = python_operator(stypy.reporting.localization.Localization(__file__, 225, 36), '**', a_50107, float_50108)
        
        # Applying the binary operator '-' (line 225)
        result_sub_50110 = python_operator(stypy.reporting.localization.Localization(__file__, 225, 29), '-', result_pow_50106, result_pow_50109)
        
        # Applying the binary operator '*' (line 225)
        result_mul_50111 = python_operator(stypy.reporting.localization.Localization(__file__, 225, 26), '*', result_div_50103, result_sub_50110)
        
        # Processing the call keyword arguments (line 224)
        kwargs_50112 = {}
        # Getting the type of 'assert_quad' (line 224)
        assert_quad_50086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 8), 'assert_quad', False)
        # Calling assert_quad(args, kwargs) (line 224)
        assert_quad_call_result_50113 = invoke(stypy.reporting.localization.Localization(__file__, 224, 8), assert_quad_50086, *[dblquad_call_result_50100, result_mul_50111], **kwargs_50112)
        
        
        # ################# End of 'test_double_integral(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_double_integral' in the type store
        # Getting the type of 'stypy_return_type' (line 218)
        stypy_return_type_50114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_50114)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_double_integral'
        return stypy_return_type_50114


    @norecursion
    def test_double_integral2(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_double_integral2'
        module_type_store = module_type_store.open_function_context('test_double_integral2', 227, 4, False)
        # Assigning a type to the variable 'self' (line 228)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestQuad.test_double_integral2.__dict__.__setitem__('stypy_localization', localization)
        TestQuad.test_double_integral2.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestQuad.test_double_integral2.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestQuad.test_double_integral2.__dict__.__setitem__('stypy_function_name', 'TestQuad.test_double_integral2')
        TestQuad.test_double_integral2.__dict__.__setitem__('stypy_param_names_list', [])
        TestQuad.test_double_integral2.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestQuad.test_double_integral2.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestQuad.test_double_integral2.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestQuad.test_double_integral2.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestQuad.test_double_integral2.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestQuad.test_double_integral2.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestQuad.test_double_integral2', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_double_integral2', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_double_integral2(...)' code ##################


        @norecursion
        def func(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'func'
            module_type_store = module_type_store.open_function_context('func', 228, 8, False)
            
            # Passed parameters checking function
            func.stypy_localization = localization
            func.stypy_type_of_self = None
            func.stypy_type_store = module_type_store
            func.stypy_function_name = 'func'
            func.stypy_param_names_list = ['x0', 'x1', 't0', 't1']
            func.stypy_varargs_param_name = None
            func.stypy_kwargs_param_name = None
            func.stypy_call_defaults = defaults
            func.stypy_call_varargs = varargs
            func.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'func', ['x0', 'x1', 't0', 't1'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'func', localization, ['x0', 'x1', 't0', 't1'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'func(...)' code ##################

            # Getting the type of 'x0' (line 229)
            x0_50115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 19), 'x0')
            # Getting the type of 'x1' (line 229)
            x1_50116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 24), 'x1')
            # Applying the binary operator '+' (line 229)
            result_add_50117 = python_operator(stypy.reporting.localization.Localization(__file__, 229, 19), '+', x0_50115, x1_50116)
            
            # Getting the type of 't0' (line 229)
            t0_50118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 29), 't0')
            # Applying the binary operator '+' (line 229)
            result_add_50119 = python_operator(stypy.reporting.localization.Localization(__file__, 229, 27), '+', result_add_50117, t0_50118)
            
            # Getting the type of 't1' (line 229)
            t1_50120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 34), 't1')
            # Applying the binary operator '+' (line 229)
            result_add_50121 = python_operator(stypy.reporting.localization.Localization(__file__, 229, 32), '+', result_add_50119, t1_50120)
            
            # Assigning a type to the variable 'stypy_return_type' (line 229)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 12), 'stypy_return_type', result_add_50121)
            
            # ################# End of 'func(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'func' in the type store
            # Getting the type of 'stypy_return_type' (line 228)
            stypy_return_type_50122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_50122)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'func'
            return stypy_return_type_50122

        # Assigning a type to the variable 'func' (line 228)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 8), 'func', func)
        
        # Assigning a Lambda to a Name (line 230):
        
        # Assigning a Lambda to a Name (line 230):

        @norecursion
        def _stypy_temp_lambda_28(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_28'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_28', 230, 12, True)
            # Passed parameters checking function
            _stypy_temp_lambda_28.stypy_localization = localization
            _stypy_temp_lambda_28.stypy_type_of_self = None
            _stypy_temp_lambda_28.stypy_type_store = module_type_store
            _stypy_temp_lambda_28.stypy_function_name = '_stypy_temp_lambda_28'
            _stypy_temp_lambda_28.stypy_param_names_list = ['x']
            _stypy_temp_lambda_28.stypy_varargs_param_name = None
            _stypy_temp_lambda_28.stypy_kwargs_param_name = None
            _stypy_temp_lambda_28.stypy_call_defaults = defaults
            _stypy_temp_lambda_28.stypy_call_varargs = varargs
            _stypy_temp_lambda_28.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_28', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_28', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            # Getting the type of 'x' (line 230)
            x_50123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 22), 'x')
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 230)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 12), 'stypy_return_type', x_50123)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_28' in the type store
            # Getting the type of 'stypy_return_type' (line 230)
            stypy_return_type_50124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 12), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_50124)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_28'
            return stypy_return_type_50124

        # Assigning a type to the variable '_stypy_temp_lambda_28' (line 230)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 12), '_stypy_temp_lambda_28', _stypy_temp_lambda_28)
        # Getting the type of '_stypy_temp_lambda_28' (line 230)
        _stypy_temp_lambda_28_50125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 12), '_stypy_temp_lambda_28')
        # Assigning a type to the variable 'g' (line 230)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 8), 'g', _stypy_temp_lambda_28_50125)
        
        # Assigning a Lambda to a Name (line 231):
        
        # Assigning a Lambda to a Name (line 231):

        @norecursion
        def _stypy_temp_lambda_29(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_29'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_29', 231, 12, True)
            # Passed parameters checking function
            _stypy_temp_lambda_29.stypy_localization = localization
            _stypy_temp_lambda_29.stypy_type_of_self = None
            _stypy_temp_lambda_29.stypy_type_store = module_type_store
            _stypy_temp_lambda_29.stypy_function_name = '_stypy_temp_lambda_29'
            _stypy_temp_lambda_29.stypy_param_names_list = ['x']
            _stypy_temp_lambda_29.stypy_varargs_param_name = None
            _stypy_temp_lambda_29.stypy_kwargs_param_name = None
            _stypy_temp_lambda_29.stypy_call_defaults = defaults
            _stypy_temp_lambda_29.stypy_call_varargs = varargs
            _stypy_temp_lambda_29.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_29', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_29', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            int_50126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 22), 'int')
            # Getting the type of 'x' (line 231)
            x_50127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 26), 'x')
            # Applying the binary operator '*' (line 231)
            result_mul_50128 = python_operator(stypy.reporting.localization.Localization(__file__, 231, 22), '*', int_50126, x_50127)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 231)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 12), 'stypy_return_type', result_mul_50128)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_29' in the type store
            # Getting the type of 'stypy_return_type' (line 231)
            stypy_return_type_50129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 12), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_50129)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_29'
            return stypy_return_type_50129

        # Assigning a type to the variable '_stypy_temp_lambda_29' (line 231)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 12), '_stypy_temp_lambda_29', _stypy_temp_lambda_29)
        # Getting the type of '_stypy_temp_lambda_29' (line 231)
        _stypy_temp_lambda_29_50130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 12), '_stypy_temp_lambda_29')
        # Assigning a type to the variable 'h' (line 231)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 8), 'h', _stypy_temp_lambda_29_50130)
        
        # Assigning a Tuple to a Name (line 232):
        
        # Assigning a Tuple to a Name (line 232):
        
        # Obtaining an instance of the builtin type 'tuple' (line 232)
        tuple_50131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 232)
        # Adding element type (line 232)
        int_50132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 15), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 232, 15), tuple_50131, int_50132)
        # Adding element type (line 232)
        int_50133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 18), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 232, 15), tuple_50131, int_50133)
        
        # Assigning a type to the variable 'args' (line 232)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 8), 'args', tuple_50131)
        
        # Call to assert_quad(...): (line 233)
        # Processing the call arguments (line 233)
        
        # Call to dblquad(...): (line 233)
        # Processing the call arguments (line 233)
        # Getting the type of 'func' (line 233)
        func_50136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 28), 'func', False)
        int_50137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 34), 'int')
        int_50138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 37), 'int')
        # Getting the type of 'g' (line 233)
        g_50139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 40), 'g', False)
        # Getting the type of 'h' (line 233)
        h_50140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 43), 'h', False)
        # Processing the call keyword arguments (line 233)
        # Getting the type of 'args' (line 233)
        args_50141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 51), 'args', False)
        keyword_50142 = args_50141
        kwargs_50143 = {'args': keyword_50142}
        # Getting the type of 'dblquad' (line 233)
        dblquad_50135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 20), 'dblquad', False)
        # Calling dblquad(args, kwargs) (line 233)
        dblquad_call_result_50144 = invoke(stypy.reporting.localization.Localization(__file__, 233, 20), dblquad_50135, *[func_50136, int_50137, int_50138, g_50139, h_50140], **kwargs_50143)
        
        float_50145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 57), 'float')
        int_50146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 61), 'int')
        # Applying the binary operator 'div' (line 233)
        result_div_50147 = python_operator(stypy.reporting.localization.Localization(__file__, 233, 57), 'div', float_50145, int_50146)
        
        int_50148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 65), 'int')
        float_50149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 67), 'float')
        # Applying the binary operator '*' (line 233)
        result_mul_50150 = python_operator(stypy.reporting.localization.Localization(__file__, 233, 65), '*', int_50148, float_50149)
        
        # Applying the binary operator '+' (line 233)
        result_add_50151 = python_operator(stypy.reporting.localization.Localization(__file__, 233, 57), '+', result_div_50147, result_mul_50150)
        
        # Processing the call keyword arguments (line 233)
        kwargs_50152 = {}
        # Getting the type of 'assert_quad' (line 233)
        assert_quad_50134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 8), 'assert_quad', False)
        # Calling assert_quad(args, kwargs) (line 233)
        assert_quad_call_result_50153 = invoke(stypy.reporting.localization.Localization(__file__, 233, 8), assert_quad_50134, *[dblquad_call_result_50144, result_add_50151], **kwargs_50152)
        
        
        # ################# End of 'test_double_integral2(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_double_integral2' in the type store
        # Getting the type of 'stypy_return_type' (line 227)
        stypy_return_type_50154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_50154)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_double_integral2'
        return stypy_return_type_50154


    @norecursion
    def test_triple_integral(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_triple_integral'
        module_type_store = module_type_store.open_function_context('test_triple_integral', 235, 4, False)
        # Assigning a type to the variable 'self' (line 236)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestQuad.test_triple_integral.__dict__.__setitem__('stypy_localization', localization)
        TestQuad.test_triple_integral.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestQuad.test_triple_integral.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestQuad.test_triple_integral.__dict__.__setitem__('stypy_function_name', 'TestQuad.test_triple_integral')
        TestQuad.test_triple_integral.__dict__.__setitem__('stypy_param_names_list', [])
        TestQuad.test_triple_integral.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestQuad.test_triple_integral.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestQuad.test_triple_integral.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestQuad.test_triple_integral.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestQuad.test_triple_integral.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestQuad.test_triple_integral.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestQuad.test_triple_integral', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_triple_integral', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_triple_integral(...)' code ##################


        @norecursion
        def simpfunc(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'simpfunc'
            module_type_store = module_type_store.open_function_context('simpfunc', 237, 8, False)
            
            # Passed parameters checking function
            simpfunc.stypy_localization = localization
            simpfunc.stypy_type_of_self = None
            simpfunc.stypy_type_store = module_type_store
            simpfunc.stypy_function_name = 'simpfunc'
            simpfunc.stypy_param_names_list = ['z', 'y', 'x', 't']
            simpfunc.stypy_varargs_param_name = None
            simpfunc.stypy_kwargs_param_name = None
            simpfunc.stypy_call_defaults = defaults
            simpfunc.stypy_call_varargs = varargs
            simpfunc.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'simpfunc', ['z', 'y', 'x', 't'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'simpfunc', localization, ['z', 'y', 'x', 't'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'simpfunc(...)' code ##################

            # Getting the type of 'x' (line 238)
            x_50155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 20), 'x')
            # Getting the type of 'y' (line 238)
            y_50156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 22), 'y')
            # Applying the binary operator '+' (line 238)
            result_add_50157 = python_operator(stypy.reporting.localization.Localization(__file__, 238, 20), '+', x_50155, y_50156)
            
            # Getting the type of 'z' (line 238)
            z_50158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 24), 'z')
            # Applying the binary operator '+' (line 238)
            result_add_50159 = python_operator(stypy.reporting.localization.Localization(__file__, 238, 23), '+', result_add_50157, z_50158)
            
            # Getting the type of 't' (line 238)
            t_50160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 27), 't')
            # Applying the binary operator '*' (line 238)
            result_mul_50161 = python_operator(stypy.reporting.localization.Localization(__file__, 238, 19), '*', result_add_50159, t_50160)
            
            # Assigning a type to the variable 'stypy_return_type' (line 238)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 12), 'stypy_return_type', result_mul_50161)
            
            # ################# End of 'simpfunc(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'simpfunc' in the type store
            # Getting the type of 'stypy_return_type' (line 237)
            stypy_return_type_50162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_50162)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'simpfunc'
            return stypy_return_type_50162

        # Assigning a type to the variable 'simpfunc' (line 237)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 8), 'simpfunc', simpfunc)
        
        # Assigning a Tuple to a Tuple (line 240):
        
        # Assigning a Num to a Name (line 240):
        float_50163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 15), 'float')
        # Assigning a type to the variable 'tuple_assignment_49282' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'tuple_assignment_49282', float_50163)
        
        # Assigning a Num to a Name (line 240):
        float_50164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 20), 'float')
        # Assigning a type to the variable 'tuple_assignment_49283' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'tuple_assignment_49283', float_50164)
        
        # Assigning a Name to a Name (line 240):
        # Getting the type of 'tuple_assignment_49282' (line 240)
        tuple_assignment_49282_50165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'tuple_assignment_49282')
        # Assigning a type to the variable 'a' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'a', tuple_assignment_49282_50165)
        
        # Assigning a Name to a Name (line 240):
        # Getting the type of 'tuple_assignment_49283' (line 240)
        tuple_assignment_49283_50166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'tuple_assignment_49283')
        # Assigning a type to the variable 'b' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 11), 'b', tuple_assignment_49283_50166)
        
        # Call to assert_quad(...): (line 241)
        # Processing the call arguments (line 241)
        
        # Call to tplquad(...): (line 241)
        # Processing the call arguments (line 241)
        # Getting the type of 'simpfunc' (line 241)
        simpfunc_50169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 28), 'simpfunc', False)
        # Getting the type of 'a' (line 241)
        a_50170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 38), 'a', False)
        # Getting the type of 'b' (line 241)
        b_50171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 41), 'b', False)

        @norecursion
        def _stypy_temp_lambda_30(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_30'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_30', 242, 28, True)
            # Passed parameters checking function
            _stypy_temp_lambda_30.stypy_localization = localization
            _stypy_temp_lambda_30.stypy_type_of_self = None
            _stypy_temp_lambda_30.stypy_type_store = module_type_store
            _stypy_temp_lambda_30.stypy_function_name = '_stypy_temp_lambda_30'
            _stypy_temp_lambda_30.stypy_param_names_list = ['x']
            _stypy_temp_lambda_30.stypy_varargs_param_name = None
            _stypy_temp_lambda_30.stypy_kwargs_param_name = None
            _stypy_temp_lambda_30.stypy_call_defaults = defaults
            _stypy_temp_lambda_30.stypy_call_varargs = varargs
            _stypy_temp_lambda_30.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_30', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_30', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            # Getting the type of 'x' (line 242)
            x_50172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 38), 'x', False)
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 242)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 28), 'stypy_return_type', x_50172)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_30' in the type store
            # Getting the type of 'stypy_return_type' (line 242)
            stypy_return_type_50173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 28), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_50173)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_30'
            return stypy_return_type_50173

        # Assigning a type to the variable '_stypy_temp_lambda_30' (line 242)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 28), '_stypy_temp_lambda_30', _stypy_temp_lambda_30)
        # Getting the type of '_stypy_temp_lambda_30' (line 242)
        _stypy_temp_lambda_30_50174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 28), '_stypy_temp_lambda_30')

        @norecursion
        def _stypy_temp_lambda_31(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_31'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_31', 242, 41, True)
            # Passed parameters checking function
            _stypy_temp_lambda_31.stypy_localization = localization
            _stypy_temp_lambda_31.stypy_type_of_self = None
            _stypy_temp_lambda_31.stypy_type_store = module_type_store
            _stypy_temp_lambda_31.stypy_function_name = '_stypy_temp_lambda_31'
            _stypy_temp_lambda_31.stypy_param_names_list = ['x']
            _stypy_temp_lambda_31.stypy_varargs_param_name = None
            _stypy_temp_lambda_31.stypy_kwargs_param_name = None
            _stypy_temp_lambda_31.stypy_call_defaults = defaults
            _stypy_temp_lambda_31.stypy_call_varargs = varargs
            _stypy_temp_lambda_31.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_31', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_31', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            int_50175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 51), 'int')
            # Getting the type of 'x' (line 242)
            x_50176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 53), 'x', False)
            # Applying the binary operator '*' (line 242)
            result_mul_50177 = python_operator(stypy.reporting.localization.Localization(__file__, 242, 51), '*', int_50175, x_50176)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 242)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 41), 'stypy_return_type', result_mul_50177)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_31' in the type store
            # Getting the type of 'stypy_return_type' (line 242)
            stypy_return_type_50178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 41), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_50178)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_31'
            return stypy_return_type_50178

        # Assigning a type to the variable '_stypy_temp_lambda_31' (line 242)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 41), '_stypy_temp_lambda_31', _stypy_temp_lambda_31)
        # Getting the type of '_stypy_temp_lambda_31' (line 242)
        _stypy_temp_lambda_31_50179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 41), '_stypy_temp_lambda_31')

        @norecursion
        def _stypy_temp_lambda_32(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_32'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_32', 243, 28, True)
            # Passed parameters checking function
            _stypy_temp_lambda_32.stypy_localization = localization
            _stypy_temp_lambda_32.stypy_type_of_self = None
            _stypy_temp_lambda_32.stypy_type_store = module_type_store
            _stypy_temp_lambda_32.stypy_function_name = '_stypy_temp_lambda_32'
            _stypy_temp_lambda_32.stypy_param_names_list = ['x', 'y']
            _stypy_temp_lambda_32.stypy_varargs_param_name = None
            _stypy_temp_lambda_32.stypy_kwargs_param_name = None
            _stypy_temp_lambda_32.stypy_call_defaults = defaults
            _stypy_temp_lambda_32.stypy_call_varargs = varargs
            _stypy_temp_lambda_32.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_32', ['x', 'y'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_32', ['x', 'y'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            # Getting the type of 'x' (line 243)
            x_50180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 41), 'x', False)
            # Getting the type of 'y' (line 243)
            y_50181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 45), 'y', False)
            # Applying the binary operator '-' (line 243)
            result_sub_50182 = python_operator(stypy.reporting.localization.Localization(__file__, 243, 41), '-', x_50180, y_50181)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 243)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 28), 'stypy_return_type', result_sub_50182)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_32' in the type store
            # Getting the type of 'stypy_return_type' (line 243)
            stypy_return_type_50183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 28), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_50183)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_32'
            return stypy_return_type_50183

        # Assigning a type to the variable '_stypy_temp_lambda_32' (line 243)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 28), '_stypy_temp_lambda_32', _stypy_temp_lambda_32)
        # Getting the type of '_stypy_temp_lambda_32' (line 243)
        _stypy_temp_lambda_32_50184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 28), '_stypy_temp_lambda_32')

        @norecursion
        def _stypy_temp_lambda_33(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_33'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_33', 243, 48, True)
            # Passed parameters checking function
            _stypy_temp_lambda_33.stypy_localization = localization
            _stypy_temp_lambda_33.stypy_type_of_self = None
            _stypy_temp_lambda_33.stypy_type_store = module_type_store
            _stypy_temp_lambda_33.stypy_function_name = '_stypy_temp_lambda_33'
            _stypy_temp_lambda_33.stypy_param_names_list = ['x', 'y']
            _stypy_temp_lambda_33.stypy_varargs_param_name = None
            _stypy_temp_lambda_33.stypy_kwargs_param_name = None
            _stypy_temp_lambda_33.stypy_call_defaults = defaults
            _stypy_temp_lambda_33.stypy_call_varargs = varargs
            _stypy_temp_lambda_33.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_33', ['x', 'y'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_33', ['x', 'y'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            # Getting the type of 'x' (line 243)
            x_50185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 61), 'x', False)
            # Getting the type of 'y' (line 243)
            y_50186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 65), 'y', False)
            # Applying the binary operator '+' (line 243)
            result_add_50187 = python_operator(stypy.reporting.localization.Localization(__file__, 243, 61), '+', x_50185, y_50186)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 243)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 48), 'stypy_return_type', result_add_50187)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_33' in the type store
            # Getting the type of 'stypy_return_type' (line 243)
            stypy_return_type_50188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 48), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_50188)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_33'
            return stypy_return_type_50188

        # Assigning a type to the variable '_stypy_temp_lambda_33' (line 243)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 48), '_stypy_temp_lambda_33', _stypy_temp_lambda_33)
        # Getting the type of '_stypy_temp_lambda_33' (line 243)
        _stypy_temp_lambda_33_50189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 48), '_stypy_temp_lambda_33')
        
        # Obtaining an instance of the builtin type 'tuple' (line 244)
        tuple_50190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 29), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 244)
        # Adding element type (line 244)
        float_50191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 29), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 244, 29), tuple_50190, float_50191)
        
        # Processing the call keyword arguments (line 241)
        kwargs_50192 = {}
        # Getting the type of 'tplquad' (line 241)
        tplquad_50168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 20), 'tplquad', False)
        # Calling tplquad(args, kwargs) (line 241)
        tplquad_call_result_50193 = invoke(stypy.reporting.localization.Localization(__file__, 241, 20), tplquad_50168, *[simpfunc_50169, a_50170, b_50171, _stypy_temp_lambda_30_50174, _stypy_temp_lambda_31_50179, _stypy_temp_lambda_32_50184, _stypy_temp_lambda_33_50189, tuple_50190], **kwargs_50192)
        
        int_50194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 21), 'int')
        int_50195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 23), 'int')
        # Applying the binary operator '*' (line 245)
        result_mul_50196 = python_operator(stypy.reporting.localization.Localization(__file__, 245, 21), '*', int_50194, int_50195)
        
        float_50197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 25), 'float')
        # Applying the binary operator 'div' (line 245)
        result_div_50198 = python_operator(stypy.reporting.localization.Localization(__file__, 245, 24), 'div', result_mul_50196, float_50197)
        
        # Getting the type of 'b' (line 245)
        b_50199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 32), 'b', False)
        float_50200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 35), 'float')
        # Applying the binary operator '**' (line 245)
        result_pow_50201 = python_operator(stypy.reporting.localization.Localization(__file__, 245, 32), '**', b_50199, float_50200)
        
        # Getting the type of 'a' (line 245)
        a_50202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 41), 'a', False)
        float_50203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 44), 'float')
        # Applying the binary operator '**' (line 245)
        result_pow_50204 = python_operator(stypy.reporting.localization.Localization(__file__, 245, 41), '**', a_50202, float_50203)
        
        # Applying the binary operator '-' (line 245)
        result_sub_50205 = python_operator(stypy.reporting.localization.Localization(__file__, 245, 32), '-', result_pow_50201, result_pow_50204)
        
        # Applying the binary operator '*' (line 245)
        result_mul_50206 = python_operator(stypy.reporting.localization.Localization(__file__, 245, 29), '*', result_div_50198, result_sub_50205)
        
        # Processing the call keyword arguments (line 241)
        kwargs_50207 = {}
        # Getting the type of 'assert_quad' (line 241)
        assert_quad_50167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 8), 'assert_quad', False)
        # Calling assert_quad(args, kwargs) (line 241)
        assert_quad_call_result_50208 = invoke(stypy.reporting.localization.Localization(__file__, 241, 8), assert_quad_50167, *[tplquad_call_result_50193, result_mul_50206], **kwargs_50207)
        
        
        # ################# End of 'test_triple_integral(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_triple_integral' in the type store
        # Getting the type of 'stypy_return_type' (line 235)
        stypy_return_type_50209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_50209)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_triple_integral'
        return stypy_return_type_50209


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 139, 0, False)
        # Assigning a type to the variable 'self' (line 140)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestQuad.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestQuad' (line 139)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 0), 'TestQuad', TestQuad)
# Declaration of the 'TestNQuad' class

class TestNQuad(object, ):

    @norecursion
    def test_fixed_limits(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_fixed_limits'
        module_type_store = module_type_store.open_function_context('test_fixed_limits', 249, 4, False)
        # Assigning a type to the variable 'self' (line 250)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestNQuad.test_fixed_limits.__dict__.__setitem__('stypy_localization', localization)
        TestNQuad.test_fixed_limits.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestNQuad.test_fixed_limits.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestNQuad.test_fixed_limits.__dict__.__setitem__('stypy_function_name', 'TestNQuad.test_fixed_limits')
        TestNQuad.test_fixed_limits.__dict__.__setitem__('stypy_param_names_list', [])
        TestNQuad.test_fixed_limits.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestNQuad.test_fixed_limits.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestNQuad.test_fixed_limits.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestNQuad.test_fixed_limits.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestNQuad.test_fixed_limits.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestNQuad.test_fixed_limits.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestNQuad.test_fixed_limits', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_fixed_limits', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_fixed_limits(...)' code ##################


        @norecursion
        def func1(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'func1'
            module_type_store = module_type_store.open_function_context('func1', 250, 8, False)
            
            # Passed parameters checking function
            func1.stypy_localization = localization
            func1.stypy_type_of_self = None
            func1.stypy_type_store = module_type_store
            func1.stypy_function_name = 'func1'
            func1.stypy_param_names_list = ['x0', 'x1', 'x2', 'x3']
            func1.stypy_varargs_param_name = None
            func1.stypy_kwargs_param_name = None
            func1.stypy_call_defaults = defaults
            func1.stypy_call_varargs = varargs
            func1.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'func1', ['x0', 'x1', 'x2', 'x3'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'func1', localization, ['x0', 'x1', 'x2', 'x3'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'func1(...)' code ##################

            
            # Assigning a BinOp to a Name (line 251):
            
            # Assigning a BinOp to a Name (line 251):
            # Getting the type of 'x0' (line 251)
            x0_50210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 19), 'x0')
            int_50211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 23), 'int')
            # Applying the binary operator '**' (line 251)
            result_pow_50212 = python_operator(stypy.reporting.localization.Localization(__file__, 251, 19), '**', x0_50210, int_50211)
            
            # Getting the type of 'x1' (line 251)
            x1_50213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 27), 'x1')
            # Getting the type of 'x2' (line 251)
            x2_50214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 30), 'x2')
            # Applying the binary operator '*' (line 251)
            result_mul_50215 = python_operator(stypy.reporting.localization.Localization(__file__, 251, 27), '*', x1_50213, x2_50214)
            
            # Applying the binary operator '+' (line 251)
            result_add_50216 = python_operator(stypy.reporting.localization.Localization(__file__, 251, 19), '+', result_pow_50212, result_mul_50215)
            
            # Getting the type of 'x3' (line 251)
            x3_50217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 35), 'x3')
            int_50218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 39), 'int')
            # Applying the binary operator '**' (line 251)
            result_pow_50219 = python_operator(stypy.reporting.localization.Localization(__file__, 251, 35), '**', x3_50217, int_50218)
            
            # Applying the binary operator '-' (line 251)
            result_sub_50220 = python_operator(stypy.reporting.localization.Localization(__file__, 251, 33), '-', result_add_50216, result_pow_50219)
            
            
            # Call to sin(...): (line 251)
            # Processing the call arguments (line 251)
            # Getting the type of 'x0' (line 251)
            x0_50223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 50), 'x0', False)
            # Processing the call keyword arguments (line 251)
            kwargs_50224 = {}
            # Getting the type of 'np' (line 251)
            np_50221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 43), 'np', False)
            # Obtaining the member 'sin' of a type (line 251)
            sin_50222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 43), np_50221, 'sin')
            # Calling sin(args, kwargs) (line 251)
            sin_call_result_50225 = invoke(stypy.reporting.localization.Localization(__file__, 251, 43), sin_50222, *[x0_50223], **kwargs_50224)
            
            # Applying the binary operator '+' (line 251)
            result_add_50226 = python_operator(stypy.reporting.localization.Localization(__file__, 251, 41), '+', result_sub_50220, sin_call_result_50225)
            
            
            
            # Getting the type of 'x0' (line 252)
            x0_50227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 26), 'x0')
            float_50228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 31), 'float')
            # Getting the type of 'x3' (line 252)
            x3_50229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 35), 'x3')
            # Applying the binary operator '*' (line 252)
            result_mul_50230 = python_operator(stypy.reporting.localization.Localization(__file__, 252, 31), '*', float_50228, x3_50229)
            
            # Applying the binary operator '-' (line 252)
            result_sub_50231 = python_operator(stypy.reporting.localization.Localization(__file__, 252, 26), '-', x0_50227, result_mul_50230)
            
            float_50232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 40), 'float')
            # Applying the binary operator '-' (line 252)
            result_sub_50233 = python_operator(stypy.reporting.localization.Localization(__file__, 252, 38), '-', result_sub_50231, float_50232)
            
            float_50234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 46), 'float')
            # Getting the type of 'x1' (line 252)
            x1_50235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 51), 'x1')
            # Applying the binary operator '*' (line 252)
            result_mul_50236 = python_operator(stypy.reporting.localization.Localization(__file__, 252, 46), '*', float_50234, x1_50235)
            
            # Applying the binary operator '-' (line 252)
            result_sub_50237 = python_operator(stypy.reporting.localization.Localization(__file__, 252, 44), '-', result_sub_50233, result_mul_50236)
            
            int_50238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 56), 'int')
            # Applying the binary operator '>' (line 252)
            result_gt_50239 = python_operator(stypy.reporting.localization.Localization(__file__, 252, 26), '>', result_sub_50237, int_50238)
            
            # Testing the type of an if expression (line 252)
            is_suitable_condition(stypy.reporting.localization.Localization(__file__, 252, 20), result_gt_50239)
            # SSA begins for if expression (line 252)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
            int_50240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 20), 'int')
            # SSA branch for the else part of an if expression (line 252)
            module_type_store.open_ssa_branch('if expression else')
            int_50241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 64), 'int')
            # SSA join for if expression (line 252)
            module_type_store = module_type_store.join_ssa_context()
            if_exp_50242 = union_type.UnionType.add(int_50240, int_50241)
            
            # Applying the binary operator '+' (line 251)
            result_add_50243 = python_operator(stypy.reporting.localization.Localization(__file__, 251, 54), '+', result_add_50226, if_exp_50242)
            
            # Assigning a type to the variable 'val' (line 251)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 12), 'val', result_add_50243)
            # Getting the type of 'val' (line 253)
            val_50244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 19), 'val')
            # Assigning a type to the variable 'stypy_return_type' (line 253)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 12), 'stypy_return_type', val_50244)
            
            # ################# End of 'func1(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'func1' in the type store
            # Getting the type of 'stypy_return_type' (line 250)
            stypy_return_type_50245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_50245)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'func1'
            return stypy_return_type_50245

        # Assigning a type to the variable 'func1' (line 250)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 8), 'func1', func1)

        @norecursion
        def opts_basic(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'opts_basic'
            module_type_store = module_type_store.open_function_context('opts_basic', 255, 8, False)
            
            # Passed parameters checking function
            opts_basic.stypy_localization = localization
            opts_basic.stypy_type_of_self = None
            opts_basic.stypy_type_store = module_type_store
            opts_basic.stypy_function_name = 'opts_basic'
            opts_basic.stypy_param_names_list = []
            opts_basic.stypy_varargs_param_name = 'args'
            opts_basic.stypy_kwargs_param_name = None
            opts_basic.stypy_call_defaults = defaults
            opts_basic.stypy_call_varargs = varargs
            opts_basic.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'opts_basic', [], 'args', None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'opts_basic', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'opts_basic(...)' code ##################

            
            # Obtaining an instance of the builtin type 'dict' (line 256)
            dict_50246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 19), 'dict')
            # Adding type elements to the builtin type 'dict' instance (line 256)
            # Adding element type (key, value) (line 256)
            str_50247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 20), 'str', 'points')
            
            # Obtaining an instance of the builtin type 'list' (line 256)
            list_50248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 30), 'list')
            # Adding type elements to the builtin type 'list' instance (line 256)
            # Adding element type (line 256)
            float_50249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 31), 'float')
            
            # Obtaining the type of the subscript
            int_50250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 40), 'int')
            # Getting the type of 'args' (line 256)
            args_50251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 35), 'args')
            # Obtaining the member '__getitem__' of a type (line 256)
            getitem___50252 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 35), args_50251, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 256)
            subscript_call_result_50253 = invoke(stypy.reporting.localization.Localization(__file__, 256, 35), getitem___50252, int_50250)
            
            # Applying the binary operator '*' (line 256)
            result_mul_50254 = python_operator(stypy.reporting.localization.Localization(__file__, 256, 31), '*', float_50249, subscript_call_result_50253)
            
            float_50255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 45), 'float')
            # Applying the binary operator '+' (line 256)
            result_add_50256 = python_operator(stypy.reporting.localization.Localization(__file__, 256, 31), '+', result_mul_50254, float_50255)
            
            float_50257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 51), 'float')
            
            # Obtaining the type of the subscript
            int_50258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 61), 'int')
            # Getting the type of 'args' (line 256)
            args_50259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 56), 'args')
            # Obtaining the member '__getitem__' of a type (line 256)
            getitem___50260 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 56), args_50259, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 256)
            subscript_call_result_50261 = invoke(stypy.reporting.localization.Localization(__file__, 256, 56), getitem___50260, int_50258)
            
            # Applying the binary operator '*' (line 256)
            result_mul_50262 = python_operator(stypy.reporting.localization.Localization(__file__, 256, 51), '*', float_50257, subscript_call_result_50261)
            
            # Applying the binary operator '+' (line 256)
            result_add_50263 = python_operator(stypy.reporting.localization.Localization(__file__, 256, 49), '+', result_add_50256, result_mul_50262)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 256, 30), list_50248, result_add_50263)
            
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 256, 19), dict_50246, (str_50247, list_50248))
            
            # Assigning a type to the variable 'stypy_return_type' (line 256)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 12), 'stypy_return_type', dict_50246)
            
            # ################# End of 'opts_basic(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'opts_basic' in the type store
            # Getting the type of 'stypy_return_type' (line 255)
            stypy_return_type_50264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_50264)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'opts_basic'
            return stypy_return_type_50264

        # Assigning a type to the variable 'opts_basic' (line 255)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 8), 'opts_basic', opts_basic)
        
        # Assigning a Call to a Name (line 258):
        
        # Assigning a Call to a Name (line 258):
        
        # Call to nquad(...): (line 258)
        # Processing the call arguments (line 258)
        # Getting the type of 'func1' (line 258)
        func1_50266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 20), 'func1', False)
        
        # Obtaining an instance of the builtin type 'list' (line 258)
        list_50267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 258)
        # Adding element type (line 258)
        
        # Obtaining an instance of the builtin type 'list' (line 258)
        list_50268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 258)
        # Adding element type (line 258)
        int_50269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 28), list_50268, int_50269)
        # Adding element type (line 258)
        int_50270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 28), list_50268, int_50270)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 27), list_50267, list_50268)
        # Adding element type (line 258)
        
        # Obtaining an instance of the builtin type 'list' (line 258)
        list_50271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 36), 'list')
        # Adding type elements to the builtin type 'list' instance (line 258)
        # Adding element type (line 258)
        int_50272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 36), list_50271, int_50272)
        # Adding element type (line 258)
        int_50273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 36), list_50271, int_50273)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 27), list_50267, list_50271)
        # Adding element type (line 258)
        
        # Obtaining an instance of the builtin type 'list' (line 258)
        list_50274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 45), 'list')
        # Adding type elements to the builtin type 'list' instance (line 258)
        # Adding element type (line 258)
        float_50275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 46), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 45), list_50274, float_50275)
        # Adding element type (line 258)
        float_50276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 51), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 45), list_50274, float_50276)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 27), list_50267, list_50274)
        # Adding element type (line 258)
        
        # Obtaining an instance of the builtin type 'list' (line 258)
        list_50277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 56), 'list')
        # Adding type elements to the builtin type 'list' instance (line 258)
        # Adding element type (line 258)
        float_50278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 57), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 56), list_50277, float_50278)
        # Adding element type (line 258)
        int_50279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 63), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 56), list_50277, int_50279)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 27), list_50267, list_50277)
        
        # Processing the call keyword arguments (line 258)
        
        # Obtaining an instance of the builtin type 'list' (line 259)
        list_50280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 259)
        # Adding element type (line 259)
        # Getting the type of 'opts_basic' (line 259)
        opts_basic_50281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 26), 'opts_basic', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 25), list_50280, opts_basic_50281)
        # Adding element type (line 259)
        
        # Obtaining an instance of the builtin type 'dict' (line 259)
        dict_50282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 38), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 259)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 25), list_50280, dict_50282)
        # Adding element type (line 259)
        
        # Obtaining an instance of the builtin type 'dict' (line 259)
        dict_50283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 42), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 259)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 25), list_50280, dict_50283)
        # Adding element type (line 259)
        
        # Obtaining an instance of the builtin type 'dict' (line 259)
        dict_50284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 46), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 259)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 25), list_50280, dict_50284)
        
        keyword_50285 = list_50280
        # Getting the type of 'True' (line 259)
        True_50286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 63), 'True', False)
        keyword_50287 = True_50286
        kwargs_50288 = {'full_output': keyword_50287, 'opts': keyword_50285}
        # Getting the type of 'nquad' (line 258)
        nquad_50265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 14), 'nquad', False)
        # Calling nquad(args, kwargs) (line 258)
        nquad_call_result_50289 = invoke(stypy.reporting.localization.Localization(__file__, 258, 14), nquad_50265, *[func1_50266, list_50267], **kwargs_50288)
        
        # Assigning a type to the variable 'res' (line 258)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 8), 'res', nquad_call_result_50289)
        
        # Call to assert_quad(...): (line 260)
        # Processing the call arguments (line 260)
        
        # Obtaining the type of the subscript
        int_50291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 25), 'int')
        slice_50292 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 260, 20), None, int_50291, None)
        # Getting the type of 'res' (line 260)
        res_50293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 20), 'res', False)
        # Obtaining the member '__getitem__' of a type (line 260)
        getitem___50294 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 20), res_50293, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 260)
        subscript_call_result_50295 = invoke(stypy.reporting.localization.Localization(__file__, 260, 20), getitem___50294, slice_50292)
        
        float_50296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 30), 'float')
        # Processing the call keyword arguments (line 260)
        kwargs_50297 = {}
        # Getting the type of 'assert_quad' (line 260)
        assert_quad_50290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 8), 'assert_quad', False)
        # Calling assert_quad(args, kwargs) (line 260)
        assert_quad_call_result_50298 = invoke(stypy.reporting.localization.Localization(__file__, 260, 8), assert_quad_50290, *[subscript_call_result_50295, float_50296], **kwargs_50297)
        
        
        # Call to assert_(...): (line 261)
        # Processing the call arguments (line 261)
        
        # Evaluating a boolean operation
        
        
        # Obtaining the type of the subscript
        str_50300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 24), 'str', 'neval')
        
        # Obtaining the type of the subscript
        int_50301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 20), 'int')
        # Getting the type of 'res' (line 261)
        res_50302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 16), 'res', False)
        # Obtaining the member '__getitem__' of a type (line 261)
        getitem___50303 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 16), res_50302, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 261)
        subscript_call_result_50304 = invoke(stypy.reporting.localization.Localization(__file__, 261, 16), getitem___50303, int_50301)
        
        # Obtaining the member '__getitem__' of a type (line 261)
        getitem___50305 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 16), subscript_call_result_50304, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 261)
        subscript_call_result_50306 = invoke(stypy.reporting.localization.Localization(__file__, 261, 16), getitem___50305, str_50300)
        
        int_50307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 35), 'int')
        # Applying the binary operator '>' (line 261)
        result_gt_50308 = python_operator(stypy.reporting.localization.Localization(__file__, 261, 16), '>', subscript_call_result_50306, int_50307)
        
        
        
        # Obtaining the type of the subscript
        str_50309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 49), 'str', 'neval')
        
        # Obtaining the type of the subscript
        int_50310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 45), 'int')
        # Getting the type of 'res' (line 261)
        res_50311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 41), 'res', False)
        # Obtaining the member '__getitem__' of a type (line 261)
        getitem___50312 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 41), res_50311, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 261)
        subscript_call_result_50313 = invoke(stypy.reporting.localization.Localization(__file__, 261, 41), getitem___50312, int_50310)
        
        # Obtaining the member '__getitem__' of a type (line 261)
        getitem___50314 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 41), subscript_call_result_50313, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 261)
        subscript_call_result_50315 = invoke(stypy.reporting.localization.Localization(__file__, 261, 41), getitem___50314, str_50309)
        
        float_50316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 60), 'float')
        # Applying the binary operator '<' (line 261)
        result_lt_50317 = python_operator(stypy.reporting.localization.Localization(__file__, 261, 41), '<', subscript_call_result_50315, float_50316)
        
        # Applying the binary operator 'and' (line 261)
        result_and_keyword_50318 = python_operator(stypy.reporting.localization.Localization(__file__, 261, 16), 'and', result_gt_50308, result_lt_50317)
        
        # Processing the call keyword arguments (line 261)
        kwargs_50319 = {}
        # Getting the type of 'assert_' (line 261)
        assert__50299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 261)
        assert__call_result_50320 = invoke(stypy.reporting.localization.Localization(__file__, 261, 8), assert__50299, *[result_and_keyword_50318], **kwargs_50319)
        
        
        # ################# End of 'test_fixed_limits(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_fixed_limits' in the type store
        # Getting the type of 'stypy_return_type' (line 249)
        stypy_return_type_50321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_50321)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_fixed_limits'
        return stypy_return_type_50321


    @norecursion
    def test_variable_limits(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_variable_limits'
        module_type_store = module_type_store.open_function_context('test_variable_limits', 263, 4, False)
        # Assigning a type to the variable 'self' (line 264)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestNQuad.test_variable_limits.__dict__.__setitem__('stypy_localization', localization)
        TestNQuad.test_variable_limits.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestNQuad.test_variable_limits.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestNQuad.test_variable_limits.__dict__.__setitem__('stypy_function_name', 'TestNQuad.test_variable_limits')
        TestNQuad.test_variable_limits.__dict__.__setitem__('stypy_param_names_list', [])
        TestNQuad.test_variable_limits.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestNQuad.test_variable_limits.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestNQuad.test_variable_limits.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestNQuad.test_variable_limits.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestNQuad.test_variable_limits.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestNQuad.test_variable_limits.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestNQuad.test_variable_limits', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_variable_limits', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_variable_limits(...)' code ##################

        
        # Assigning a Num to a Name (line 264):
        
        # Assigning a Num to a Name (line 264):
        float_50322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 16), 'float')
        # Assigning a type to the variable 'scale' (line 264)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 8), 'scale', float_50322)

        @norecursion
        def func2(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'func2'
            module_type_store = module_type_store.open_function_context('func2', 266, 8, False)
            
            # Passed parameters checking function
            func2.stypy_localization = localization
            func2.stypy_type_of_self = None
            func2.stypy_type_store = module_type_store
            func2.stypy_function_name = 'func2'
            func2.stypy_param_names_list = ['x0', 'x1', 'x2', 'x3', 't0', 't1']
            func2.stypy_varargs_param_name = None
            func2.stypy_kwargs_param_name = None
            func2.stypy_call_defaults = defaults
            func2.stypy_call_varargs = varargs
            func2.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'func2', ['x0', 'x1', 'x2', 'x3', 't0', 't1'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'func2', localization, ['x0', 'x1', 'x2', 'x3', 't0', 't1'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'func2(...)' code ##################

            
            # Assigning a BinOp to a Name (line 267):
            
            # Assigning a BinOp to a Name (line 267):
            # Getting the type of 'x0' (line 267)
            x0_50323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 19), 'x0')
            # Getting the type of 'x1' (line 267)
            x1_50324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 22), 'x1')
            # Applying the binary operator '*' (line 267)
            result_mul_50325 = python_operator(stypy.reporting.localization.Localization(__file__, 267, 19), '*', x0_50323, x1_50324)
            
            # Getting the type of 'x3' (line 267)
            x3_50326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 25), 'x3')
            int_50327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 29), 'int')
            # Applying the binary operator '**' (line 267)
            result_pow_50328 = python_operator(stypy.reporting.localization.Localization(__file__, 267, 25), '**', x3_50326, int_50327)
            
            # Applying the binary operator '*' (line 267)
            result_mul_50329 = python_operator(stypy.reporting.localization.Localization(__file__, 267, 24), '*', result_mul_50325, result_pow_50328)
            
            
            # Call to sin(...): (line 267)
            # Processing the call arguments (line 267)
            # Getting the type of 'x2' (line 267)
            x2_50332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 40), 'x2', False)
            # Processing the call keyword arguments (line 267)
            kwargs_50333 = {}
            # Getting the type of 'np' (line 267)
            np_50330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 33), 'np', False)
            # Obtaining the member 'sin' of a type (line 267)
            sin_50331 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 267, 33), np_50330, 'sin')
            # Calling sin(args, kwargs) (line 267)
            sin_call_result_50334 = invoke(stypy.reporting.localization.Localization(__file__, 267, 33), sin_50331, *[x2_50332], **kwargs_50333)
            
            # Applying the binary operator '+' (line 267)
            result_add_50335 = python_operator(stypy.reporting.localization.Localization(__file__, 267, 19), '+', result_mul_50329, sin_call_result_50334)
            
            int_50336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 46), 'int')
            # Applying the binary operator '+' (line 267)
            result_add_50337 = python_operator(stypy.reporting.localization.Localization(__file__, 267, 44), '+', result_add_50335, int_50336)
            
            
            
            # Getting the type of 'x0' (line 268)
            x0_50338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 25), 'x0')
            # Getting the type of 't1' (line 268)
            t1_50339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 30), 't1')
            # Getting the type of 'x1' (line 268)
            x1_50340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 33), 'x1')
            # Applying the binary operator '*' (line 268)
            result_mul_50341 = python_operator(stypy.reporting.localization.Localization(__file__, 268, 30), '*', t1_50339, x1_50340)
            
            # Applying the binary operator '+' (line 268)
            result_add_50342 = python_operator(stypy.reporting.localization.Localization(__file__, 268, 25), '+', x0_50338, result_mul_50341)
            
            # Getting the type of 't0' (line 268)
            t0_50343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 38), 't0')
            # Applying the binary operator '-' (line 268)
            result_sub_50344 = python_operator(stypy.reporting.localization.Localization(__file__, 268, 36), '-', result_add_50342, t0_50343)
            
            int_50345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 43), 'int')
            # Applying the binary operator '>' (line 268)
            result_gt_50346 = python_operator(stypy.reporting.localization.Localization(__file__, 268, 25), '>', result_sub_50344, int_50345)
            
            # Testing the type of an if expression (line 268)
            is_suitable_condition(stypy.reporting.localization.Localization(__file__, 268, 20), result_gt_50346)
            # SSA begins for if expression (line 268)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
            int_50347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 20), 'int')
            # SSA branch for the else part of an if expression (line 268)
            module_type_store.open_ssa_branch('if expression else')
            int_50348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 50), 'int')
            # SSA join for if expression (line 268)
            module_type_store = module_type_store.join_ssa_context()
            if_exp_50349 = union_type.UnionType.add(int_50347, int_50348)
            
            # Applying the binary operator '+' (line 267)
            result_add_50350 = python_operator(stypy.reporting.localization.Localization(__file__, 267, 48), '+', result_add_50337, if_exp_50349)
            
            # Assigning a type to the variable 'val' (line 267)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 12), 'val', result_add_50350)
            # Getting the type of 'val' (line 269)
            val_50351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 19), 'val')
            # Assigning a type to the variable 'stypy_return_type' (line 269)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 12), 'stypy_return_type', val_50351)
            
            # ################# End of 'func2(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'func2' in the type store
            # Getting the type of 'stypy_return_type' (line 266)
            stypy_return_type_50352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_50352)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'func2'
            return stypy_return_type_50352

        # Assigning a type to the variable 'func2' (line 266)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 8), 'func2', func2)

        @norecursion
        def lim0(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'lim0'
            module_type_store = module_type_store.open_function_context('lim0', 271, 8, False)
            
            # Passed parameters checking function
            lim0.stypy_localization = localization
            lim0.stypy_type_of_self = None
            lim0.stypy_type_store = module_type_store
            lim0.stypy_function_name = 'lim0'
            lim0.stypy_param_names_list = ['x1', 'x2', 'x3', 't0', 't1']
            lim0.stypy_varargs_param_name = None
            lim0.stypy_kwargs_param_name = None
            lim0.stypy_call_defaults = defaults
            lim0.stypy_call_varargs = varargs
            lim0.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'lim0', ['x1', 'x2', 'x3', 't0', 't1'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'lim0', localization, ['x1', 'x2', 'x3', 't0', 't1'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'lim0(...)' code ##################

            
            # Obtaining an instance of the builtin type 'list' (line 272)
            list_50353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 19), 'list')
            # Adding type elements to the builtin type 'list' instance (line 272)
            # Adding element type (line 272)
            # Getting the type of 'scale' (line 272)
            scale_50354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 20), 'scale')
            # Getting the type of 'x1' (line 272)
            x1_50355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 29), 'x1')
            int_50356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 33), 'int')
            # Applying the binary operator '**' (line 272)
            result_pow_50357 = python_operator(stypy.reporting.localization.Localization(__file__, 272, 29), '**', x1_50355, int_50356)
            
            # Getting the type of 'x2' (line 272)
            x2_50358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 37), 'x2')
            # Applying the binary operator '+' (line 272)
            result_add_50359 = python_operator(stypy.reporting.localization.Localization(__file__, 272, 29), '+', result_pow_50357, x2_50358)
            
            
            # Call to cos(...): (line 272)
            # Processing the call arguments (line 272)
            # Getting the type of 'x3' (line 272)
            x3_50362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 49), 'x3', False)
            # Processing the call keyword arguments (line 272)
            kwargs_50363 = {}
            # Getting the type of 'np' (line 272)
            np_50360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 42), 'np', False)
            # Obtaining the member 'cos' of a type (line 272)
            cos_50361 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 42), np_50360, 'cos')
            # Calling cos(args, kwargs) (line 272)
            cos_call_result_50364 = invoke(stypy.reporting.localization.Localization(__file__, 272, 42), cos_50361, *[x3_50362], **kwargs_50363)
            
            # Getting the type of 't0' (line 272)
            t0_50365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 53), 't0')
            # Applying the binary operator '*' (line 272)
            result_mul_50366 = python_operator(stypy.reporting.localization.Localization(__file__, 272, 42), '*', cos_call_result_50364, t0_50365)
            
            # Getting the type of 't1' (line 272)
            t1_50367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 56), 't1')
            # Applying the binary operator '*' (line 272)
            result_mul_50368 = python_operator(stypy.reporting.localization.Localization(__file__, 272, 55), '*', result_mul_50366, t1_50367)
            
            # Applying the binary operator '+' (line 272)
            result_add_50369 = python_operator(stypy.reporting.localization.Localization(__file__, 272, 40), '+', result_add_50359, result_mul_50368)
            
            int_50370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 61), 'int')
            # Applying the binary operator '+' (line 272)
            result_add_50371 = python_operator(stypy.reporting.localization.Localization(__file__, 272, 59), '+', result_add_50369, int_50370)
            
            # Applying the binary operator '*' (line 272)
            result_mul_50372 = python_operator(stypy.reporting.localization.Localization(__file__, 272, 20), '*', scale_50354, result_add_50371)
            
            int_50373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 66), 'int')
            # Applying the binary operator '-' (line 272)
            result_sub_50374 = python_operator(stypy.reporting.localization.Localization(__file__, 272, 20), '-', result_mul_50372, int_50373)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 272, 19), list_50353, result_sub_50374)
            # Adding element type (line 272)
            # Getting the type of 'scale' (line 273)
            scale_50375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 20), 'scale')
            # Getting the type of 'x1' (line 273)
            x1_50376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 29), 'x1')
            int_50377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 33), 'int')
            # Applying the binary operator '**' (line 273)
            result_pow_50378 = python_operator(stypy.reporting.localization.Localization(__file__, 273, 29), '**', x1_50376, int_50377)
            
            # Getting the type of 'x2' (line 273)
            x2_50379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 37), 'x2')
            # Applying the binary operator '+' (line 273)
            result_add_50380 = python_operator(stypy.reporting.localization.Localization(__file__, 273, 29), '+', result_pow_50378, x2_50379)
            
            
            # Call to cos(...): (line 273)
            # Processing the call arguments (line 273)
            # Getting the type of 'x3' (line 273)
            x3_50383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 49), 'x3', False)
            # Processing the call keyword arguments (line 273)
            kwargs_50384 = {}
            # Getting the type of 'np' (line 273)
            np_50381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 42), 'np', False)
            # Obtaining the member 'cos' of a type (line 273)
            cos_50382 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 42), np_50381, 'cos')
            # Calling cos(args, kwargs) (line 273)
            cos_call_result_50385 = invoke(stypy.reporting.localization.Localization(__file__, 273, 42), cos_50382, *[x3_50383], **kwargs_50384)
            
            # Getting the type of 't0' (line 273)
            t0_50386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 53), 't0')
            # Applying the binary operator '*' (line 273)
            result_mul_50387 = python_operator(stypy.reporting.localization.Localization(__file__, 273, 42), '*', cos_call_result_50385, t0_50386)
            
            # Getting the type of 't1' (line 273)
            t1_50388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 56), 't1')
            # Applying the binary operator '*' (line 273)
            result_mul_50389 = python_operator(stypy.reporting.localization.Localization(__file__, 273, 55), '*', result_mul_50387, t1_50388)
            
            # Applying the binary operator '+' (line 273)
            result_add_50390 = python_operator(stypy.reporting.localization.Localization(__file__, 273, 40), '+', result_add_50380, result_mul_50389)
            
            int_50391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 61), 'int')
            # Applying the binary operator '+' (line 273)
            result_add_50392 = python_operator(stypy.reporting.localization.Localization(__file__, 273, 59), '+', result_add_50390, int_50391)
            
            # Applying the binary operator '*' (line 273)
            result_mul_50393 = python_operator(stypy.reporting.localization.Localization(__file__, 273, 20), '*', scale_50375, result_add_50392)
            
            int_50394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 66), 'int')
            # Applying the binary operator '+' (line 273)
            result_add_50395 = python_operator(stypy.reporting.localization.Localization(__file__, 273, 20), '+', result_mul_50393, int_50394)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 272, 19), list_50353, result_add_50395)
            
            # Assigning a type to the variable 'stypy_return_type' (line 272)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 12), 'stypy_return_type', list_50353)
            
            # ################# End of 'lim0(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'lim0' in the type store
            # Getting the type of 'stypy_return_type' (line 271)
            stypy_return_type_50396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_50396)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'lim0'
            return stypy_return_type_50396

        # Assigning a type to the variable 'lim0' (line 271)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 8), 'lim0', lim0)

        @norecursion
        def lim1(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'lim1'
            module_type_store = module_type_store.open_function_context('lim1', 275, 8, False)
            
            # Passed parameters checking function
            lim1.stypy_localization = localization
            lim1.stypy_type_of_self = None
            lim1.stypy_type_store = module_type_store
            lim1.stypy_function_name = 'lim1'
            lim1.stypy_param_names_list = ['x2', 'x3', 't0', 't1']
            lim1.stypy_varargs_param_name = None
            lim1.stypy_kwargs_param_name = None
            lim1.stypy_call_defaults = defaults
            lim1.stypy_call_varargs = varargs
            lim1.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'lim1', ['x2', 'x3', 't0', 't1'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'lim1', localization, ['x2', 'x3', 't0', 't1'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'lim1(...)' code ##################

            
            # Obtaining an instance of the builtin type 'list' (line 276)
            list_50397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 19), 'list')
            # Adding type elements to the builtin type 'list' instance (line 276)
            # Adding element type (line 276)
            # Getting the type of 'scale' (line 276)
            scale_50398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 20), 'scale')
            # Getting the type of 't0' (line 276)
            t0_50399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 29), 't0')
            # Getting the type of 'x2' (line 276)
            x2_50400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 32), 'x2')
            # Applying the binary operator '*' (line 276)
            result_mul_50401 = python_operator(stypy.reporting.localization.Localization(__file__, 276, 29), '*', t0_50399, x2_50400)
            
            # Getting the type of 't1' (line 276)
            t1_50402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 37), 't1')
            # Getting the type of 'x3' (line 276)
            x3_50403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 40), 'x3')
            # Applying the binary operator '*' (line 276)
            result_mul_50404 = python_operator(stypy.reporting.localization.Localization(__file__, 276, 37), '*', t1_50402, x3_50403)
            
            # Applying the binary operator '+' (line 276)
            result_add_50405 = python_operator(stypy.reporting.localization.Localization(__file__, 276, 29), '+', result_mul_50401, result_mul_50404)
            
            # Applying the binary operator '*' (line 276)
            result_mul_50406 = python_operator(stypy.reporting.localization.Localization(__file__, 276, 20), '*', scale_50398, result_add_50405)
            
            int_50407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 46), 'int')
            # Applying the binary operator '-' (line 276)
            result_sub_50408 = python_operator(stypy.reporting.localization.Localization(__file__, 276, 20), '-', result_mul_50406, int_50407)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 276, 19), list_50397, result_sub_50408)
            # Adding element type (line 276)
            # Getting the type of 'scale' (line 277)
            scale_50409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 20), 'scale')
            # Getting the type of 't0' (line 277)
            t0_50410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 29), 't0')
            # Getting the type of 'x2' (line 277)
            x2_50411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 32), 'x2')
            # Applying the binary operator '*' (line 277)
            result_mul_50412 = python_operator(stypy.reporting.localization.Localization(__file__, 277, 29), '*', t0_50410, x2_50411)
            
            # Getting the type of 't1' (line 277)
            t1_50413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 37), 't1')
            # Getting the type of 'x3' (line 277)
            x3_50414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 40), 'x3')
            # Applying the binary operator '*' (line 277)
            result_mul_50415 = python_operator(stypy.reporting.localization.Localization(__file__, 277, 37), '*', t1_50413, x3_50414)
            
            # Applying the binary operator '+' (line 277)
            result_add_50416 = python_operator(stypy.reporting.localization.Localization(__file__, 277, 29), '+', result_mul_50412, result_mul_50415)
            
            # Applying the binary operator '*' (line 277)
            result_mul_50417 = python_operator(stypy.reporting.localization.Localization(__file__, 277, 20), '*', scale_50409, result_add_50416)
            
            int_50418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 46), 'int')
            # Applying the binary operator '+' (line 277)
            result_add_50419 = python_operator(stypy.reporting.localization.Localization(__file__, 277, 20), '+', result_mul_50417, int_50418)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 276, 19), list_50397, result_add_50419)
            
            # Assigning a type to the variable 'stypy_return_type' (line 276)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 12), 'stypy_return_type', list_50397)
            
            # ################# End of 'lim1(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'lim1' in the type store
            # Getting the type of 'stypy_return_type' (line 275)
            stypy_return_type_50420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_50420)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'lim1'
            return stypy_return_type_50420

        # Assigning a type to the variable 'lim1' (line 275)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 8), 'lim1', lim1)

        @norecursion
        def lim2(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'lim2'
            module_type_store = module_type_store.open_function_context('lim2', 279, 8, False)
            
            # Passed parameters checking function
            lim2.stypy_localization = localization
            lim2.stypy_type_of_self = None
            lim2.stypy_type_store = module_type_store
            lim2.stypy_function_name = 'lim2'
            lim2.stypy_param_names_list = ['x3', 't0', 't1']
            lim2.stypy_varargs_param_name = None
            lim2.stypy_kwargs_param_name = None
            lim2.stypy_call_defaults = defaults
            lim2.stypy_call_varargs = varargs
            lim2.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'lim2', ['x3', 't0', 't1'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'lim2', localization, ['x3', 't0', 't1'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'lim2(...)' code ##################

            
            # Obtaining an instance of the builtin type 'list' (line 280)
            list_50421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 19), 'list')
            # Adding type elements to the builtin type 'list' instance (line 280)
            # Adding element type (line 280)
            # Getting the type of 'scale' (line 280)
            scale_50422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 20), 'scale')
            # Getting the type of 'x3' (line 280)
            x3_50423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 29), 'x3')
            # Getting the type of 't0' (line 280)
            t0_50424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 34), 't0')
            int_50425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 38), 'int')
            # Applying the binary operator '**' (line 280)
            result_pow_50426 = python_operator(stypy.reporting.localization.Localization(__file__, 280, 34), '**', t0_50424, int_50425)
            
            # Getting the type of 't1' (line 280)
            t1_50427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 40), 't1')
            int_50428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 44), 'int')
            # Applying the binary operator '**' (line 280)
            result_pow_50429 = python_operator(stypy.reporting.localization.Localization(__file__, 280, 40), '**', t1_50427, int_50428)
            
            # Applying the binary operator '*' (line 280)
            result_mul_50430 = python_operator(stypy.reporting.localization.Localization(__file__, 280, 34), '*', result_pow_50426, result_pow_50429)
            
            # Applying the binary operator '+' (line 280)
            result_add_50431 = python_operator(stypy.reporting.localization.Localization(__file__, 280, 29), '+', x3_50423, result_mul_50430)
            
            # Applying the binary operator '*' (line 280)
            result_mul_50432 = python_operator(stypy.reporting.localization.Localization(__file__, 280, 20), '*', scale_50422, result_add_50431)
            
            int_50433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 49), 'int')
            # Applying the binary operator '-' (line 280)
            result_sub_50434 = python_operator(stypy.reporting.localization.Localization(__file__, 280, 20), '-', result_mul_50432, int_50433)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 280, 19), list_50421, result_sub_50434)
            # Adding element type (line 280)
            # Getting the type of 'scale' (line 281)
            scale_50435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 20), 'scale')
            # Getting the type of 'x3' (line 281)
            x3_50436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 29), 'x3')
            # Getting the type of 't0' (line 281)
            t0_50437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 34), 't0')
            int_50438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 38), 'int')
            # Applying the binary operator '**' (line 281)
            result_pow_50439 = python_operator(stypy.reporting.localization.Localization(__file__, 281, 34), '**', t0_50437, int_50438)
            
            # Getting the type of 't1' (line 281)
            t1_50440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 40), 't1')
            int_50441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 44), 'int')
            # Applying the binary operator '**' (line 281)
            result_pow_50442 = python_operator(stypy.reporting.localization.Localization(__file__, 281, 40), '**', t1_50440, int_50441)
            
            # Applying the binary operator '*' (line 281)
            result_mul_50443 = python_operator(stypy.reporting.localization.Localization(__file__, 281, 34), '*', result_pow_50439, result_pow_50442)
            
            # Applying the binary operator '+' (line 281)
            result_add_50444 = python_operator(stypy.reporting.localization.Localization(__file__, 281, 29), '+', x3_50436, result_mul_50443)
            
            # Applying the binary operator '*' (line 281)
            result_mul_50445 = python_operator(stypy.reporting.localization.Localization(__file__, 281, 20), '*', scale_50435, result_add_50444)
            
            int_50446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 49), 'int')
            # Applying the binary operator '+' (line 281)
            result_add_50447 = python_operator(stypy.reporting.localization.Localization(__file__, 281, 20), '+', result_mul_50445, int_50446)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 280, 19), list_50421, result_add_50447)
            
            # Assigning a type to the variable 'stypy_return_type' (line 280)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 12), 'stypy_return_type', list_50421)
            
            # ################# End of 'lim2(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'lim2' in the type store
            # Getting the type of 'stypy_return_type' (line 279)
            stypy_return_type_50448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_50448)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'lim2'
            return stypy_return_type_50448

        # Assigning a type to the variable 'lim2' (line 279)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 8), 'lim2', lim2)

        @norecursion
        def lim3(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'lim3'
            module_type_store = module_type_store.open_function_context('lim3', 283, 8, False)
            
            # Passed parameters checking function
            lim3.stypy_localization = localization
            lim3.stypy_type_of_self = None
            lim3.stypy_type_store = module_type_store
            lim3.stypy_function_name = 'lim3'
            lim3.stypy_param_names_list = ['t0', 't1']
            lim3.stypy_varargs_param_name = None
            lim3.stypy_kwargs_param_name = None
            lim3.stypy_call_defaults = defaults
            lim3.stypy_call_varargs = varargs
            lim3.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'lim3', ['t0', 't1'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'lim3', localization, ['t0', 't1'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'lim3(...)' code ##################

            
            # Obtaining an instance of the builtin type 'list' (line 284)
            list_50449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 19), 'list')
            # Adding type elements to the builtin type 'list' instance (line 284)
            # Adding element type (line 284)
            # Getting the type of 'scale' (line 284)
            scale_50450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 20), 'scale')
            # Getting the type of 't0' (line 284)
            t0_50451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 29), 't0')
            # Getting the type of 't1' (line 284)
            t1_50452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 34), 't1')
            # Applying the binary operator '+' (line 284)
            result_add_50453 = python_operator(stypy.reporting.localization.Localization(__file__, 284, 29), '+', t0_50451, t1_50452)
            
            # Applying the binary operator '*' (line 284)
            result_mul_50454 = python_operator(stypy.reporting.localization.Localization(__file__, 284, 20), '*', scale_50450, result_add_50453)
            
            int_50455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 40), 'int')
            # Applying the binary operator '-' (line 284)
            result_sub_50456 = python_operator(stypy.reporting.localization.Localization(__file__, 284, 20), '-', result_mul_50454, int_50455)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 284, 19), list_50449, result_sub_50456)
            # Adding element type (line 284)
            # Getting the type of 'scale' (line 284)
            scale_50457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 43), 'scale')
            # Getting the type of 't0' (line 284)
            t0_50458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 52), 't0')
            # Getting the type of 't1' (line 284)
            t1_50459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 57), 't1')
            # Applying the binary operator '+' (line 284)
            result_add_50460 = python_operator(stypy.reporting.localization.Localization(__file__, 284, 52), '+', t0_50458, t1_50459)
            
            # Applying the binary operator '*' (line 284)
            result_mul_50461 = python_operator(stypy.reporting.localization.Localization(__file__, 284, 43), '*', scale_50457, result_add_50460)
            
            int_50462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 63), 'int')
            # Applying the binary operator '+' (line 284)
            result_add_50463 = python_operator(stypy.reporting.localization.Localization(__file__, 284, 43), '+', result_mul_50461, int_50462)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 284, 19), list_50449, result_add_50463)
            
            # Assigning a type to the variable 'stypy_return_type' (line 284)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 12), 'stypy_return_type', list_50449)
            
            # ################# End of 'lim3(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'lim3' in the type store
            # Getting the type of 'stypy_return_type' (line 283)
            stypy_return_type_50464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_50464)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'lim3'
            return stypy_return_type_50464

        # Assigning a type to the variable 'lim3' (line 283)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 8), 'lim3', lim3)

        @norecursion
        def opts0(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'opts0'
            module_type_store = module_type_store.open_function_context('opts0', 286, 8, False)
            
            # Passed parameters checking function
            opts0.stypy_localization = localization
            opts0.stypy_type_of_self = None
            opts0.stypy_type_store = module_type_store
            opts0.stypy_function_name = 'opts0'
            opts0.stypy_param_names_list = ['x1', 'x2', 'x3', 't0', 't1']
            opts0.stypy_varargs_param_name = None
            opts0.stypy_kwargs_param_name = None
            opts0.stypy_call_defaults = defaults
            opts0.stypy_call_varargs = varargs
            opts0.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'opts0', ['x1', 'x2', 'x3', 't0', 't1'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'opts0', localization, ['x1', 'x2', 'x3', 't0', 't1'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'opts0(...)' code ##################

            
            # Obtaining an instance of the builtin type 'dict' (line 287)
            dict_50465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 19), 'dict')
            # Adding type elements to the builtin type 'dict' instance (line 287)
            # Adding element type (key, value) (line 287)
            str_50466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 20), 'str', 'points')
            
            # Obtaining an instance of the builtin type 'list' (line 287)
            list_50467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 30), 'list')
            # Adding type elements to the builtin type 'list' instance (line 287)
            # Adding element type (line 287)
            # Getting the type of 't0' (line 287)
            t0_50468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 31), 't0')
            # Getting the type of 't1' (line 287)
            t1_50469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 36), 't1')
            # Getting the type of 'x1' (line 287)
            x1_50470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 39), 'x1')
            # Applying the binary operator '*' (line 287)
            result_mul_50471 = python_operator(stypy.reporting.localization.Localization(__file__, 287, 36), '*', t1_50469, x1_50470)
            
            # Applying the binary operator '-' (line 287)
            result_sub_50472 = python_operator(stypy.reporting.localization.Localization(__file__, 287, 31), '-', t0_50468, result_mul_50471)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 287, 30), list_50467, result_sub_50472)
            
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 287, 19), dict_50465, (str_50466, list_50467))
            
            # Assigning a type to the variable 'stypy_return_type' (line 287)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 12), 'stypy_return_type', dict_50465)
            
            # ################# End of 'opts0(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'opts0' in the type store
            # Getting the type of 'stypy_return_type' (line 286)
            stypy_return_type_50473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_50473)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'opts0'
            return stypy_return_type_50473

        # Assigning a type to the variable 'opts0' (line 286)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 8), 'opts0', opts0)

        @norecursion
        def opts1(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'opts1'
            module_type_store = module_type_store.open_function_context('opts1', 289, 8, False)
            
            # Passed parameters checking function
            opts1.stypy_localization = localization
            opts1.stypy_type_of_self = None
            opts1.stypy_type_store = module_type_store
            opts1.stypy_function_name = 'opts1'
            opts1.stypy_param_names_list = ['x2', 'x3', 't0', 't1']
            opts1.stypy_varargs_param_name = None
            opts1.stypy_kwargs_param_name = None
            opts1.stypy_call_defaults = defaults
            opts1.stypy_call_varargs = varargs
            opts1.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'opts1', ['x2', 'x3', 't0', 't1'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'opts1', localization, ['x2', 'x3', 't0', 't1'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'opts1(...)' code ##################

            
            # Obtaining an instance of the builtin type 'dict' (line 290)
            dict_50474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 19), 'dict')
            # Adding type elements to the builtin type 'dict' instance (line 290)
            
            # Assigning a type to the variable 'stypy_return_type' (line 290)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 12), 'stypy_return_type', dict_50474)
            
            # ################# End of 'opts1(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'opts1' in the type store
            # Getting the type of 'stypy_return_type' (line 289)
            stypy_return_type_50475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_50475)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'opts1'
            return stypy_return_type_50475

        # Assigning a type to the variable 'opts1' (line 289)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 8), 'opts1', opts1)

        @norecursion
        def opts2(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'opts2'
            module_type_store = module_type_store.open_function_context('opts2', 292, 8, False)
            
            # Passed parameters checking function
            opts2.stypy_localization = localization
            opts2.stypy_type_of_self = None
            opts2.stypy_type_store = module_type_store
            opts2.stypy_function_name = 'opts2'
            opts2.stypy_param_names_list = ['x3', 't0', 't1']
            opts2.stypy_varargs_param_name = None
            opts2.stypy_kwargs_param_name = None
            opts2.stypy_call_defaults = defaults
            opts2.stypy_call_varargs = varargs
            opts2.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'opts2', ['x3', 't0', 't1'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'opts2', localization, ['x3', 't0', 't1'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'opts2(...)' code ##################

            
            # Obtaining an instance of the builtin type 'dict' (line 293)
            dict_50476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 19), 'dict')
            # Adding type elements to the builtin type 'dict' instance (line 293)
            
            # Assigning a type to the variable 'stypy_return_type' (line 293)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 12), 'stypy_return_type', dict_50476)
            
            # ################# End of 'opts2(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'opts2' in the type store
            # Getting the type of 'stypy_return_type' (line 292)
            stypy_return_type_50477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_50477)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'opts2'
            return stypy_return_type_50477

        # Assigning a type to the variable 'opts2' (line 292)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'opts2', opts2)

        @norecursion
        def opts3(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'opts3'
            module_type_store = module_type_store.open_function_context('opts3', 295, 8, False)
            
            # Passed parameters checking function
            opts3.stypy_localization = localization
            opts3.stypy_type_of_self = None
            opts3.stypy_type_store = module_type_store
            opts3.stypy_function_name = 'opts3'
            opts3.stypy_param_names_list = ['t0', 't1']
            opts3.stypy_varargs_param_name = None
            opts3.stypy_kwargs_param_name = None
            opts3.stypy_call_defaults = defaults
            opts3.stypy_call_varargs = varargs
            opts3.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'opts3', ['t0', 't1'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'opts3', localization, ['t0', 't1'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'opts3(...)' code ##################

            
            # Obtaining an instance of the builtin type 'dict' (line 296)
            dict_50478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 19), 'dict')
            # Adding type elements to the builtin type 'dict' instance (line 296)
            
            # Assigning a type to the variable 'stypy_return_type' (line 296)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 296, 12), 'stypy_return_type', dict_50478)
            
            # ################# End of 'opts3(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'opts3' in the type store
            # Getting the type of 'stypy_return_type' (line 295)
            stypy_return_type_50479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_50479)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'opts3'
            return stypy_return_type_50479

        # Assigning a type to the variable 'opts3' (line 295)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 8), 'opts3', opts3)
        
        # Assigning a Call to a Name (line 298):
        
        # Assigning a Call to a Name (line 298):
        
        # Call to nquad(...): (line 298)
        # Processing the call arguments (line 298)
        # Getting the type of 'func2' (line 298)
        func2_50481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 20), 'func2', False)
        
        # Obtaining an instance of the builtin type 'list' (line 298)
        list_50482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 298)
        # Adding element type (line 298)
        # Getting the type of 'lim0' (line 298)
        lim0_50483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 28), 'lim0', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 298, 27), list_50482, lim0_50483)
        # Adding element type (line 298)
        # Getting the type of 'lim1' (line 298)
        lim1_50484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 34), 'lim1', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 298, 27), list_50482, lim1_50484)
        # Adding element type (line 298)
        # Getting the type of 'lim2' (line 298)
        lim2_50485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 40), 'lim2', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 298, 27), list_50482, lim2_50485)
        # Adding element type (line 298)
        # Getting the type of 'lim3' (line 298)
        lim3_50486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 46), 'lim3', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 298, 27), list_50482, lim3_50486)
        
        # Processing the call keyword arguments (line 298)
        
        # Obtaining an instance of the builtin type 'tuple' (line 298)
        tuple_50487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 59), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 298)
        # Adding element type (line 298)
        int_50488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 59), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 298, 59), tuple_50487, int_50488)
        # Adding element type (line 298)
        int_50489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 62), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 298, 59), tuple_50487, int_50489)
        
        keyword_50490 = tuple_50487
        
        # Obtaining an instance of the builtin type 'list' (line 299)
        list_50491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 299)
        # Adding element type (line 299)
        # Getting the type of 'opts0' (line 299)
        opts0_50492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 26), 'opts0', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 299, 25), list_50491, opts0_50492)
        # Adding element type (line 299)
        # Getting the type of 'opts1' (line 299)
        opts1_50493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 33), 'opts1', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 299, 25), list_50491, opts1_50493)
        # Adding element type (line 299)
        # Getting the type of 'opts2' (line 299)
        opts2_50494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 40), 'opts2', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 299, 25), list_50491, opts2_50494)
        # Adding element type (line 299)
        # Getting the type of 'opts3' (line 299)
        opts3_50495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 47), 'opts3', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 299, 25), list_50491, opts3_50495)
        
        keyword_50496 = list_50491
        kwargs_50497 = {'args': keyword_50490, 'opts': keyword_50496}
        # Getting the type of 'nquad' (line 298)
        nquad_50480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 14), 'nquad', False)
        # Calling nquad(args, kwargs) (line 298)
        nquad_call_result_50498 = invoke(stypy.reporting.localization.Localization(__file__, 298, 14), nquad_50480, *[func2_50481, list_50482], **kwargs_50497)
        
        # Assigning a type to the variable 'res' (line 298)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 8), 'res', nquad_call_result_50498)
        
        # Call to assert_quad(...): (line 300)
        # Processing the call arguments (line 300)
        # Getting the type of 'res' (line 300)
        res_50500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 20), 'res', False)
        float_50501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 25), 'float')
        # Processing the call keyword arguments (line 300)
        kwargs_50502 = {}
        # Getting the type of 'assert_quad' (line 300)
        assert_quad_50499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 8), 'assert_quad', False)
        # Calling assert_quad(args, kwargs) (line 300)
        assert_quad_call_result_50503 = invoke(stypy.reporting.localization.Localization(__file__, 300, 8), assert_quad_50499, *[res_50500, float_50501], **kwargs_50502)
        
        
        # ################# End of 'test_variable_limits(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_variable_limits' in the type store
        # Getting the type of 'stypy_return_type' (line 263)
        stypy_return_type_50504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_50504)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_variable_limits'
        return stypy_return_type_50504


    @norecursion
    def test_square_separate_ranges_and_opts(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_square_separate_ranges_and_opts'
        module_type_store = module_type_store.open_function_context('test_square_separate_ranges_and_opts', 302, 4, False)
        # Assigning a type to the variable 'self' (line 303)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestNQuad.test_square_separate_ranges_and_opts.__dict__.__setitem__('stypy_localization', localization)
        TestNQuad.test_square_separate_ranges_and_opts.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestNQuad.test_square_separate_ranges_and_opts.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestNQuad.test_square_separate_ranges_and_opts.__dict__.__setitem__('stypy_function_name', 'TestNQuad.test_square_separate_ranges_and_opts')
        TestNQuad.test_square_separate_ranges_and_opts.__dict__.__setitem__('stypy_param_names_list', [])
        TestNQuad.test_square_separate_ranges_and_opts.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestNQuad.test_square_separate_ranges_and_opts.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestNQuad.test_square_separate_ranges_and_opts.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestNQuad.test_square_separate_ranges_and_opts.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestNQuad.test_square_separate_ranges_and_opts.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestNQuad.test_square_separate_ranges_and_opts.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestNQuad.test_square_separate_ranges_and_opts', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_square_separate_ranges_and_opts', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_square_separate_ranges_and_opts(...)' code ##################


        @norecursion
        def f(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'f'
            module_type_store = module_type_store.open_function_context('f', 303, 8, False)
            
            # Passed parameters checking function
            f.stypy_localization = localization
            f.stypy_type_of_self = None
            f.stypy_type_store = module_type_store
            f.stypy_function_name = 'f'
            f.stypy_param_names_list = ['y', 'x']
            f.stypy_varargs_param_name = None
            f.stypy_kwargs_param_name = None
            f.stypy_call_defaults = defaults
            f.stypy_call_varargs = varargs
            f.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'f', ['y', 'x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'f', localization, ['y', 'x'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'f(...)' code ##################

            float_50505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 19), 'float')
            # Assigning a type to the variable 'stypy_return_type' (line 304)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 12), 'stypy_return_type', float_50505)
            
            # ################# End of 'f(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'f' in the type store
            # Getting the type of 'stypy_return_type' (line 303)
            stypy_return_type_50506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_50506)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'f'
            return stypy_return_type_50506

        # Assigning a type to the variable 'f' (line 303)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 8), 'f', f)
        
        # Call to assert_quad(...): (line 306)
        # Processing the call arguments (line 306)
        
        # Call to nquad(...): (line 306)
        # Processing the call arguments (line 306)
        # Getting the type of 'f' (line 306)
        f_50509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 26), 'f', False)
        
        # Obtaining an instance of the builtin type 'list' (line 306)
        list_50510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 306)
        # Adding element type (line 306)
        
        # Obtaining an instance of the builtin type 'list' (line 306)
        list_50511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 306)
        # Adding element type (line 306)
        int_50512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 306, 30), list_50511, int_50512)
        # Adding element type (line 306)
        int_50513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 306, 30), list_50511, int_50513)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 306, 29), list_50510, list_50511)
        # Adding element type (line 306)
        
        # Obtaining an instance of the builtin type 'list' (line 306)
        list_50514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 39), 'list')
        # Adding type elements to the builtin type 'list' instance (line 306)
        # Adding element type (line 306)
        int_50515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 306, 39), list_50514, int_50515)
        # Adding element type (line 306)
        int_50516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 306, 39), list_50514, int_50516)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 306, 29), list_50510, list_50514)
        
        # Processing the call keyword arguments (line 306)
        
        # Obtaining an instance of the builtin type 'list' (line 306)
        list_50517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 54), 'list')
        # Adding type elements to the builtin type 'list' instance (line 306)
        # Adding element type (line 306)
        
        # Obtaining an instance of the builtin type 'dict' (line 306)
        dict_50518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 55), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 306)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 306, 54), list_50517, dict_50518)
        # Adding element type (line 306)
        
        # Obtaining an instance of the builtin type 'dict' (line 306)
        dict_50519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 59), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 306)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 306, 54), list_50517, dict_50519)
        
        keyword_50520 = list_50517
        kwargs_50521 = {'opts': keyword_50520}
        # Getting the type of 'nquad' (line 306)
        nquad_50508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 20), 'nquad', False)
        # Calling nquad(args, kwargs) (line 306)
        nquad_call_result_50522 = invoke(stypy.reporting.localization.Localization(__file__, 306, 20), nquad_50508, *[f_50509, list_50510], **kwargs_50521)
        
        float_50523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 65), 'float')
        # Processing the call keyword arguments (line 306)
        kwargs_50524 = {}
        # Getting the type of 'assert_quad' (line 306)
        assert_quad_50507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 8), 'assert_quad', False)
        # Calling assert_quad(args, kwargs) (line 306)
        assert_quad_call_result_50525 = invoke(stypy.reporting.localization.Localization(__file__, 306, 8), assert_quad_50507, *[nquad_call_result_50522, float_50523], **kwargs_50524)
        
        
        # ################# End of 'test_square_separate_ranges_and_opts(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_square_separate_ranges_and_opts' in the type store
        # Getting the type of 'stypy_return_type' (line 302)
        stypy_return_type_50526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_50526)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_square_separate_ranges_and_opts'
        return stypy_return_type_50526


    @norecursion
    def test_square_aliased_ranges_and_opts(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_square_aliased_ranges_and_opts'
        module_type_store = module_type_store.open_function_context('test_square_aliased_ranges_and_opts', 308, 4, False)
        # Assigning a type to the variable 'self' (line 309)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestNQuad.test_square_aliased_ranges_and_opts.__dict__.__setitem__('stypy_localization', localization)
        TestNQuad.test_square_aliased_ranges_and_opts.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestNQuad.test_square_aliased_ranges_and_opts.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestNQuad.test_square_aliased_ranges_and_opts.__dict__.__setitem__('stypy_function_name', 'TestNQuad.test_square_aliased_ranges_and_opts')
        TestNQuad.test_square_aliased_ranges_and_opts.__dict__.__setitem__('stypy_param_names_list', [])
        TestNQuad.test_square_aliased_ranges_and_opts.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestNQuad.test_square_aliased_ranges_and_opts.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestNQuad.test_square_aliased_ranges_and_opts.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestNQuad.test_square_aliased_ranges_and_opts.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestNQuad.test_square_aliased_ranges_and_opts.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestNQuad.test_square_aliased_ranges_and_opts.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestNQuad.test_square_aliased_ranges_and_opts', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_square_aliased_ranges_and_opts', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_square_aliased_ranges_and_opts(...)' code ##################


        @norecursion
        def f(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'f'
            module_type_store = module_type_store.open_function_context('f', 309, 8, False)
            
            # Passed parameters checking function
            f.stypy_localization = localization
            f.stypy_type_of_self = None
            f.stypy_type_store = module_type_store
            f.stypy_function_name = 'f'
            f.stypy_param_names_list = ['y', 'x']
            f.stypy_varargs_param_name = None
            f.stypy_kwargs_param_name = None
            f.stypy_call_defaults = defaults
            f.stypy_call_varargs = varargs
            f.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'f', ['y', 'x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'f', localization, ['y', 'x'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'f(...)' code ##################

            float_50527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 19), 'float')
            # Assigning a type to the variable 'stypy_return_type' (line 310)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 12), 'stypy_return_type', float_50527)
            
            # ################# End of 'f(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'f' in the type store
            # Getting the type of 'stypy_return_type' (line 309)
            stypy_return_type_50528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_50528)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'f'
            return stypy_return_type_50528

        # Assigning a type to the variable 'f' (line 309)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 8), 'f', f)
        
        # Assigning a List to a Name (line 312):
        
        # Assigning a List to a Name (line 312):
        
        # Obtaining an instance of the builtin type 'list' (line 312)
        list_50529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 312)
        # Adding element type (line 312)
        int_50530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 13), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 312, 12), list_50529, int_50530)
        # Adding element type (line 312)
        int_50531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 17), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 312, 12), list_50529, int_50531)
        
        # Assigning a type to the variable 'r' (line 312)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 8), 'r', list_50529)
        
        # Assigning a Dict to a Name (line 313):
        
        # Assigning a Dict to a Name (line 313):
        
        # Obtaining an instance of the builtin type 'dict' (line 313)
        dict_50532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 14), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 313)
        
        # Assigning a type to the variable 'opt' (line 313)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 313, 8), 'opt', dict_50532)
        
        # Call to assert_quad(...): (line 314)
        # Processing the call arguments (line 314)
        
        # Call to nquad(...): (line 314)
        # Processing the call arguments (line 314)
        # Getting the type of 'f' (line 314)
        f_50535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 26), 'f', False)
        
        # Obtaining an instance of the builtin type 'list' (line 314)
        list_50536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 314)
        # Adding element type (line 314)
        # Getting the type of 'r' (line 314)
        r_50537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 30), 'r', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 314, 29), list_50536, r_50537)
        # Adding element type (line 314)
        # Getting the type of 'r' (line 314)
        r_50538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 33), 'r', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 314, 29), list_50536, r_50538)
        
        # Processing the call keyword arguments (line 314)
        
        # Obtaining an instance of the builtin type 'list' (line 314)
        list_50539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 42), 'list')
        # Adding type elements to the builtin type 'list' instance (line 314)
        # Adding element type (line 314)
        # Getting the type of 'opt' (line 314)
        opt_50540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 43), 'opt', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 314, 42), list_50539, opt_50540)
        # Adding element type (line 314)
        # Getting the type of 'opt' (line 314)
        opt_50541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 48), 'opt', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 314, 42), list_50539, opt_50541)
        
        keyword_50542 = list_50539
        kwargs_50543 = {'opts': keyword_50542}
        # Getting the type of 'nquad' (line 314)
        nquad_50534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 20), 'nquad', False)
        # Calling nquad(args, kwargs) (line 314)
        nquad_call_result_50544 = invoke(stypy.reporting.localization.Localization(__file__, 314, 20), nquad_50534, *[f_50535, list_50536], **kwargs_50543)
        
        float_50545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 55), 'float')
        # Processing the call keyword arguments (line 314)
        kwargs_50546 = {}
        # Getting the type of 'assert_quad' (line 314)
        assert_quad_50533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 8), 'assert_quad', False)
        # Calling assert_quad(args, kwargs) (line 314)
        assert_quad_call_result_50547 = invoke(stypy.reporting.localization.Localization(__file__, 314, 8), assert_quad_50533, *[nquad_call_result_50544, float_50545], **kwargs_50546)
        
        
        # ################# End of 'test_square_aliased_ranges_and_opts(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_square_aliased_ranges_and_opts' in the type store
        # Getting the type of 'stypy_return_type' (line 308)
        stypy_return_type_50548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_50548)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_square_aliased_ranges_and_opts'
        return stypy_return_type_50548


    @norecursion
    def test_square_separate_fn_ranges_and_opts(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_square_separate_fn_ranges_and_opts'
        module_type_store = module_type_store.open_function_context('test_square_separate_fn_ranges_and_opts', 316, 4, False)
        # Assigning a type to the variable 'self' (line 317)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 317, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestNQuad.test_square_separate_fn_ranges_and_opts.__dict__.__setitem__('stypy_localization', localization)
        TestNQuad.test_square_separate_fn_ranges_and_opts.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestNQuad.test_square_separate_fn_ranges_and_opts.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestNQuad.test_square_separate_fn_ranges_and_opts.__dict__.__setitem__('stypy_function_name', 'TestNQuad.test_square_separate_fn_ranges_and_opts')
        TestNQuad.test_square_separate_fn_ranges_and_opts.__dict__.__setitem__('stypy_param_names_list', [])
        TestNQuad.test_square_separate_fn_ranges_and_opts.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestNQuad.test_square_separate_fn_ranges_and_opts.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestNQuad.test_square_separate_fn_ranges_and_opts.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestNQuad.test_square_separate_fn_ranges_and_opts.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestNQuad.test_square_separate_fn_ranges_and_opts.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestNQuad.test_square_separate_fn_ranges_and_opts.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestNQuad.test_square_separate_fn_ranges_and_opts', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_square_separate_fn_ranges_and_opts', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_square_separate_fn_ranges_and_opts(...)' code ##################


        @norecursion
        def f(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'f'
            module_type_store = module_type_store.open_function_context('f', 317, 8, False)
            
            # Passed parameters checking function
            f.stypy_localization = localization
            f.stypy_type_of_self = None
            f.stypy_type_store = module_type_store
            f.stypy_function_name = 'f'
            f.stypy_param_names_list = ['y', 'x']
            f.stypy_varargs_param_name = None
            f.stypy_kwargs_param_name = None
            f.stypy_call_defaults = defaults
            f.stypy_call_varargs = varargs
            f.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'f', ['y', 'x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'f', localization, ['y', 'x'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'f(...)' code ##################

            float_50549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, 19), 'float')
            # Assigning a type to the variable 'stypy_return_type' (line 318)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 12), 'stypy_return_type', float_50549)
            
            # ################# End of 'f(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'f' in the type store
            # Getting the type of 'stypy_return_type' (line 317)
            stypy_return_type_50550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_50550)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'f'
            return stypy_return_type_50550

        # Assigning a type to the variable 'f' (line 317)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 317, 8), 'f', f)

        @norecursion
        def fn_range0(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'fn_range0'
            module_type_store = module_type_store.open_function_context('fn_range0', 320, 8, False)
            
            # Passed parameters checking function
            fn_range0.stypy_localization = localization
            fn_range0.stypy_type_of_self = None
            fn_range0.stypy_type_store = module_type_store
            fn_range0.stypy_function_name = 'fn_range0'
            fn_range0.stypy_param_names_list = []
            fn_range0.stypy_varargs_param_name = 'args'
            fn_range0.stypy_kwargs_param_name = None
            fn_range0.stypy_call_defaults = defaults
            fn_range0.stypy_call_varargs = varargs
            fn_range0.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'fn_range0', [], 'args', None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'fn_range0', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'fn_range0(...)' code ##################

            
            # Obtaining an instance of the builtin type 'tuple' (line 321)
            tuple_50551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 321, 20), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 321)
            # Adding element type (line 321)
            int_50552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 321, 20), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 321, 20), tuple_50551, int_50552)
            # Adding element type (line 321)
            int_50553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 321, 24), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 321, 20), tuple_50551, int_50553)
            
            # Assigning a type to the variable 'stypy_return_type' (line 321)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 321, 12), 'stypy_return_type', tuple_50551)
            
            # ################# End of 'fn_range0(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'fn_range0' in the type store
            # Getting the type of 'stypy_return_type' (line 320)
            stypy_return_type_50554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_50554)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'fn_range0'
            return stypy_return_type_50554

        # Assigning a type to the variable 'fn_range0' (line 320)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 320, 8), 'fn_range0', fn_range0)

        @norecursion
        def fn_range1(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'fn_range1'
            module_type_store = module_type_store.open_function_context('fn_range1', 323, 8, False)
            
            # Passed parameters checking function
            fn_range1.stypy_localization = localization
            fn_range1.stypy_type_of_self = None
            fn_range1.stypy_type_store = module_type_store
            fn_range1.stypy_function_name = 'fn_range1'
            fn_range1.stypy_param_names_list = []
            fn_range1.stypy_varargs_param_name = 'args'
            fn_range1.stypy_kwargs_param_name = None
            fn_range1.stypy_call_defaults = defaults
            fn_range1.stypy_call_varargs = varargs
            fn_range1.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'fn_range1', [], 'args', None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'fn_range1', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'fn_range1(...)' code ##################

            
            # Obtaining an instance of the builtin type 'tuple' (line 324)
            tuple_50555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 20), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 324)
            # Adding element type (line 324)
            int_50556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 20), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 324, 20), tuple_50555, int_50556)
            # Adding element type (line 324)
            int_50557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 24), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 324, 20), tuple_50555, int_50557)
            
            # Assigning a type to the variable 'stypy_return_type' (line 324)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 12), 'stypy_return_type', tuple_50555)
            
            # ################# End of 'fn_range1(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'fn_range1' in the type store
            # Getting the type of 'stypy_return_type' (line 323)
            stypy_return_type_50558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_50558)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'fn_range1'
            return stypy_return_type_50558

        # Assigning a type to the variable 'fn_range1' (line 323)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 8), 'fn_range1', fn_range1)

        @norecursion
        def fn_opt0(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'fn_opt0'
            module_type_store = module_type_store.open_function_context('fn_opt0', 326, 8, False)
            
            # Passed parameters checking function
            fn_opt0.stypy_localization = localization
            fn_opt0.stypy_type_of_self = None
            fn_opt0.stypy_type_store = module_type_store
            fn_opt0.stypy_function_name = 'fn_opt0'
            fn_opt0.stypy_param_names_list = []
            fn_opt0.stypy_varargs_param_name = 'args'
            fn_opt0.stypy_kwargs_param_name = None
            fn_opt0.stypy_call_defaults = defaults
            fn_opt0.stypy_call_varargs = varargs
            fn_opt0.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'fn_opt0', [], 'args', None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'fn_opt0', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'fn_opt0(...)' code ##################

            
            # Obtaining an instance of the builtin type 'dict' (line 327)
            dict_50559 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 19), 'dict')
            # Adding type elements to the builtin type 'dict' instance (line 327)
            
            # Assigning a type to the variable 'stypy_return_type' (line 327)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 327, 12), 'stypy_return_type', dict_50559)
            
            # ################# End of 'fn_opt0(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'fn_opt0' in the type store
            # Getting the type of 'stypy_return_type' (line 326)
            stypy_return_type_50560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_50560)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'fn_opt0'
            return stypy_return_type_50560

        # Assigning a type to the variable 'fn_opt0' (line 326)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 8), 'fn_opt0', fn_opt0)

        @norecursion
        def fn_opt1(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'fn_opt1'
            module_type_store = module_type_store.open_function_context('fn_opt1', 329, 8, False)
            
            # Passed parameters checking function
            fn_opt1.stypy_localization = localization
            fn_opt1.stypy_type_of_self = None
            fn_opt1.stypy_type_store = module_type_store
            fn_opt1.stypy_function_name = 'fn_opt1'
            fn_opt1.stypy_param_names_list = []
            fn_opt1.stypy_varargs_param_name = 'args'
            fn_opt1.stypy_kwargs_param_name = None
            fn_opt1.stypy_call_defaults = defaults
            fn_opt1.stypy_call_varargs = varargs
            fn_opt1.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'fn_opt1', [], 'args', None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'fn_opt1', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'fn_opt1(...)' code ##################

            
            # Obtaining an instance of the builtin type 'dict' (line 330)
            dict_50561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 19), 'dict')
            # Adding type elements to the builtin type 'dict' instance (line 330)
            
            # Assigning a type to the variable 'stypy_return_type' (line 330)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 12), 'stypy_return_type', dict_50561)
            
            # ################# End of 'fn_opt1(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'fn_opt1' in the type store
            # Getting the type of 'stypy_return_type' (line 329)
            stypy_return_type_50562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_50562)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'fn_opt1'
            return stypy_return_type_50562

        # Assigning a type to the variable 'fn_opt1' (line 329)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 8), 'fn_opt1', fn_opt1)
        
        # Assigning a List to a Name (line 332):
        
        # Assigning a List to a Name (line 332):
        
        # Obtaining an instance of the builtin type 'list' (line 332)
        list_50563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 332)
        # Adding element type (line 332)
        # Getting the type of 'fn_range0' (line 332)
        fn_range0_50564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 18), 'fn_range0')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 332, 17), list_50563, fn_range0_50564)
        # Adding element type (line 332)
        # Getting the type of 'fn_range1' (line 332)
        fn_range1_50565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 29), 'fn_range1')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 332, 17), list_50563, fn_range1_50565)
        
        # Assigning a type to the variable 'ranges' (line 332)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 8), 'ranges', list_50563)
        
        # Assigning a List to a Name (line 333):
        
        # Assigning a List to a Name (line 333):
        
        # Obtaining an instance of the builtin type 'list' (line 333)
        list_50566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 333)
        # Adding element type (line 333)
        # Getting the type of 'fn_opt0' (line 333)
        fn_opt0_50567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 16), 'fn_opt0')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 333, 15), list_50566, fn_opt0_50567)
        # Adding element type (line 333)
        # Getting the type of 'fn_opt1' (line 333)
        fn_opt1_50568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 25), 'fn_opt1')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 333, 15), list_50566, fn_opt1_50568)
        
        # Assigning a type to the variable 'opts' (line 333)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 333, 8), 'opts', list_50566)
        
        # Call to assert_quad(...): (line 334)
        # Processing the call arguments (line 334)
        
        # Call to nquad(...): (line 334)
        # Processing the call arguments (line 334)
        # Getting the type of 'f' (line 334)
        f_50571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 26), 'f', False)
        # Getting the type of 'ranges' (line 334)
        ranges_50572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 29), 'ranges', False)
        # Processing the call keyword arguments (line 334)
        # Getting the type of 'opts' (line 334)
        opts_50573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 42), 'opts', False)
        keyword_50574 = opts_50573
        kwargs_50575 = {'opts': keyword_50574}
        # Getting the type of 'nquad' (line 334)
        nquad_50570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 20), 'nquad', False)
        # Calling nquad(args, kwargs) (line 334)
        nquad_call_result_50576 = invoke(stypy.reporting.localization.Localization(__file__, 334, 20), nquad_50570, *[f_50571, ranges_50572], **kwargs_50575)
        
        float_50577 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 49), 'float')
        # Processing the call keyword arguments (line 334)
        kwargs_50578 = {}
        # Getting the type of 'assert_quad' (line 334)
        assert_quad_50569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 8), 'assert_quad', False)
        # Calling assert_quad(args, kwargs) (line 334)
        assert_quad_call_result_50579 = invoke(stypy.reporting.localization.Localization(__file__, 334, 8), assert_quad_50569, *[nquad_call_result_50576, float_50577], **kwargs_50578)
        
        
        # ################# End of 'test_square_separate_fn_ranges_and_opts(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_square_separate_fn_ranges_and_opts' in the type store
        # Getting the type of 'stypy_return_type' (line 316)
        stypy_return_type_50580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_50580)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_square_separate_fn_ranges_and_opts'
        return stypy_return_type_50580


    @norecursion
    def test_square_aliased_fn_ranges_and_opts(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_square_aliased_fn_ranges_and_opts'
        module_type_store = module_type_store.open_function_context('test_square_aliased_fn_ranges_and_opts', 336, 4, False)
        # Assigning a type to the variable 'self' (line 337)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 337, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestNQuad.test_square_aliased_fn_ranges_and_opts.__dict__.__setitem__('stypy_localization', localization)
        TestNQuad.test_square_aliased_fn_ranges_and_opts.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestNQuad.test_square_aliased_fn_ranges_and_opts.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestNQuad.test_square_aliased_fn_ranges_and_opts.__dict__.__setitem__('stypy_function_name', 'TestNQuad.test_square_aliased_fn_ranges_and_opts')
        TestNQuad.test_square_aliased_fn_ranges_and_opts.__dict__.__setitem__('stypy_param_names_list', [])
        TestNQuad.test_square_aliased_fn_ranges_and_opts.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestNQuad.test_square_aliased_fn_ranges_and_opts.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestNQuad.test_square_aliased_fn_ranges_and_opts.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestNQuad.test_square_aliased_fn_ranges_and_opts.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestNQuad.test_square_aliased_fn_ranges_and_opts.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestNQuad.test_square_aliased_fn_ranges_and_opts.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestNQuad.test_square_aliased_fn_ranges_and_opts', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_square_aliased_fn_ranges_and_opts', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_square_aliased_fn_ranges_and_opts(...)' code ##################


        @norecursion
        def f(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'f'
            module_type_store = module_type_store.open_function_context('f', 337, 8, False)
            
            # Passed parameters checking function
            f.stypy_localization = localization
            f.stypy_type_of_self = None
            f.stypy_type_store = module_type_store
            f.stypy_function_name = 'f'
            f.stypy_param_names_list = ['y', 'x']
            f.stypy_varargs_param_name = None
            f.stypy_kwargs_param_name = None
            f.stypy_call_defaults = defaults
            f.stypy_call_varargs = varargs
            f.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'f', ['y', 'x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'f', localization, ['y', 'x'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'f(...)' code ##################

            float_50581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 19), 'float')
            # Assigning a type to the variable 'stypy_return_type' (line 338)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 338, 12), 'stypy_return_type', float_50581)
            
            # ################# End of 'f(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'f' in the type store
            # Getting the type of 'stypy_return_type' (line 337)
            stypy_return_type_50582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_50582)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'f'
            return stypy_return_type_50582

        # Assigning a type to the variable 'f' (line 337)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 337, 8), 'f', f)

        @norecursion
        def fn_range(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'fn_range'
            module_type_store = module_type_store.open_function_context('fn_range', 340, 8, False)
            
            # Passed parameters checking function
            fn_range.stypy_localization = localization
            fn_range.stypy_type_of_self = None
            fn_range.stypy_type_store = module_type_store
            fn_range.stypy_function_name = 'fn_range'
            fn_range.stypy_param_names_list = []
            fn_range.stypy_varargs_param_name = 'args'
            fn_range.stypy_kwargs_param_name = None
            fn_range.stypy_call_defaults = defaults
            fn_range.stypy_call_varargs = varargs
            fn_range.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'fn_range', [], 'args', None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'fn_range', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'fn_range(...)' code ##################

            
            # Obtaining an instance of the builtin type 'tuple' (line 341)
            tuple_50583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 20), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 341)
            # Adding element type (line 341)
            int_50584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 20), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 341, 20), tuple_50583, int_50584)
            # Adding element type (line 341)
            int_50585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 24), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 341, 20), tuple_50583, int_50585)
            
            # Assigning a type to the variable 'stypy_return_type' (line 341)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 12), 'stypy_return_type', tuple_50583)
            
            # ################# End of 'fn_range(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'fn_range' in the type store
            # Getting the type of 'stypy_return_type' (line 340)
            stypy_return_type_50586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_50586)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'fn_range'
            return stypy_return_type_50586

        # Assigning a type to the variable 'fn_range' (line 340)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 8), 'fn_range', fn_range)

        @norecursion
        def fn_opt(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'fn_opt'
            module_type_store = module_type_store.open_function_context('fn_opt', 343, 8, False)
            
            # Passed parameters checking function
            fn_opt.stypy_localization = localization
            fn_opt.stypy_type_of_self = None
            fn_opt.stypy_type_store = module_type_store
            fn_opt.stypy_function_name = 'fn_opt'
            fn_opt.stypy_param_names_list = []
            fn_opt.stypy_varargs_param_name = 'args'
            fn_opt.stypy_kwargs_param_name = None
            fn_opt.stypy_call_defaults = defaults
            fn_opt.stypy_call_varargs = varargs
            fn_opt.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'fn_opt', [], 'args', None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'fn_opt', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'fn_opt(...)' code ##################

            
            # Obtaining an instance of the builtin type 'dict' (line 344)
            dict_50587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 19), 'dict')
            # Adding type elements to the builtin type 'dict' instance (line 344)
            
            # Assigning a type to the variable 'stypy_return_type' (line 344)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 344, 12), 'stypy_return_type', dict_50587)
            
            # ################# End of 'fn_opt(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'fn_opt' in the type store
            # Getting the type of 'stypy_return_type' (line 343)
            stypy_return_type_50588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_50588)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'fn_opt'
            return stypy_return_type_50588

        # Assigning a type to the variable 'fn_opt' (line 343)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 8), 'fn_opt', fn_opt)
        
        # Assigning a List to a Name (line 346):
        
        # Assigning a List to a Name (line 346):
        
        # Obtaining an instance of the builtin type 'list' (line 346)
        list_50589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 346)
        # Adding element type (line 346)
        # Getting the type of 'fn_range' (line 346)
        fn_range_50590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 18), 'fn_range')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 346, 17), list_50589, fn_range_50590)
        # Adding element type (line 346)
        # Getting the type of 'fn_range' (line 346)
        fn_range_50591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 28), 'fn_range')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 346, 17), list_50589, fn_range_50591)
        
        # Assigning a type to the variable 'ranges' (line 346)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 346, 8), 'ranges', list_50589)
        
        # Assigning a List to a Name (line 347):
        
        # Assigning a List to a Name (line 347):
        
        # Obtaining an instance of the builtin type 'list' (line 347)
        list_50592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 347, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 347)
        # Adding element type (line 347)
        # Getting the type of 'fn_opt' (line 347)
        fn_opt_50593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 16), 'fn_opt')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 347, 15), list_50592, fn_opt_50593)
        # Adding element type (line 347)
        # Getting the type of 'fn_opt' (line 347)
        fn_opt_50594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 24), 'fn_opt')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 347, 15), list_50592, fn_opt_50594)
        
        # Assigning a type to the variable 'opts' (line 347)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 347, 8), 'opts', list_50592)
        
        # Call to assert_quad(...): (line 348)
        # Processing the call arguments (line 348)
        
        # Call to nquad(...): (line 348)
        # Processing the call arguments (line 348)
        # Getting the type of 'f' (line 348)
        f_50597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 26), 'f', False)
        # Getting the type of 'ranges' (line 348)
        ranges_50598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 29), 'ranges', False)
        # Processing the call keyword arguments (line 348)
        # Getting the type of 'opts' (line 348)
        opts_50599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 42), 'opts', False)
        keyword_50600 = opts_50599
        kwargs_50601 = {'opts': keyword_50600}
        # Getting the type of 'nquad' (line 348)
        nquad_50596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 20), 'nquad', False)
        # Calling nquad(args, kwargs) (line 348)
        nquad_call_result_50602 = invoke(stypy.reporting.localization.Localization(__file__, 348, 20), nquad_50596, *[f_50597, ranges_50598], **kwargs_50601)
        
        float_50603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 49), 'float')
        # Processing the call keyword arguments (line 348)
        kwargs_50604 = {}
        # Getting the type of 'assert_quad' (line 348)
        assert_quad_50595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 8), 'assert_quad', False)
        # Calling assert_quad(args, kwargs) (line 348)
        assert_quad_call_result_50605 = invoke(stypy.reporting.localization.Localization(__file__, 348, 8), assert_quad_50595, *[nquad_call_result_50602, float_50603], **kwargs_50604)
        
        
        # ################# End of 'test_square_aliased_fn_ranges_and_opts(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_square_aliased_fn_ranges_and_opts' in the type store
        # Getting the type of 'stypy_return_type' (line 336)
        stypy_return_type_50606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_50606)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_square_aliased_fn_ranges_and_opts'
        return stypy_return_type_50606


    @norecursion
    def test_matching_quad(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_matching_quad'
        module_type_store = module_type_store.open_function_context('test_matching_quad', 350, 4, False)
        # Assigning a type to the variable 'self' (line 351)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 351, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestNQuad.test_matching_quad.__dict__.__setitem__('stypy_localization', localization)
        TestNQuad.test_matching_quad.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestNQuad.test_matching_quad.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestNQuad.test_matching_quad.__dict__.__setitem__('stypy_function_name', 'TestNQuad.test_matching_quad')
        TestNQuad.test_matching_quad.__dict__.__setitem__('stypy_param_names_list', [])
        TestNQuad.test_matching_quad.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestNQuad.test_matching_quad.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestNQuad.test_matching_quad.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestNQuad.test_matching_quad.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestNQuad.test_matching_quad.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestNQuad.test_matching_quad.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestNQuad.test_matching_quad', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_matching_quad', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_matching_quad(...)' code ##################


        @norecursion
        def func(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'func'
            module_type_store = module_type_store.open_function_context('func', 351, 8, False)
            
            # Passed parameters checking function
            func.stypy_localization = localization
            func.stypy_type_of_self = None
            func.stypy_type_store = module_type_store
            func.stypy_function_name = 'func'
            func.stypy_param_names_list = ['x']
            func.stypy_varargs_param_name = None
            func.stypy_kwargs_param_name = None
            func.stypy_call_defaults = defaults
            func.stypy_call_varargs = varargs
            func.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'func', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'func', localization, ['x'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'func(...)' code ##################

            # Getting the type of 'x' (line 352)
            x_50607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 19), 'x')
            int_50608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 22), 'int')
            # Applying the binary operator '**' (line 352)
            result_pow_50609 = python_operator(stypy.reporting.localization.Localization(__file__, 352, 19), '**', x_50607, int_50608)
            
            int_50610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 26), 'int')
            # Applying the binary operator '+' (line 352)
            result_add_50611 = python_operator(stypy.reporting.localization.Localization(__file__, 352, 19), '+', result_pow_50609, int_50610)
            
            # Assigning a type to the variable 'stypy_return_type' (line 352)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 352, 12), 'stypy_return_type', result_add_50611)
            
            # ################# End of 'func(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'func' in the type store
            # Getting the type of 'stypy_return_type' (line 351)
            stypy_return_type_50612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_50612)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'func'
            return stypy_return_type_50612

        # Assigning a type to the variable 'func' (line 351)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 351, 8), 'func', func)
        
        # Assigning a Call to a Tuple (line 354):
        
        # Assigning a Subscript to a Name (line 354):
        
        # Obtaining the type of the subscript
        int_50613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 8), 'int')
        
        # Call to quad(...): (line 354)
        # Processing the call arguments (line 354)
        # Getting the type of 'func' (line 354)
        func_50615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 27), 'func', False)
        int_50616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 33), 'int')
        int_50617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 36), 'int')
        # Processing the call keyword arguments (line 354)
        kwargs_50618 = {}
        # Getting the type of 'quad' (line 354)
        quad_50614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 22), 'quad', False)
        # Calling quad(args, kwargs) (line 354)
        quad_call_result_50619 = invoke(stypy.reporting.localization.Localization(__file__, 354, 22), quad_50614, *[func_50615, int_50616, int_50617], **kwargs_50618)
        
        # Obtaining the member '__getitem__' of a type (line 354)
        getitem___50620 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 8), quad_call_result_50619, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 354)
        subscript_call_result_50621 = invoke(stypy.reporting.localization.Localization(__file__, 354, 8), getitem___50620, int_50613)
        
        # Assigning a type to the variable 'tuple_var_assignment_49284' (line 354)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 8), 'tuple_var_assignment_49284', subscript_call_result_50621)
        
        # Assigning a Subscript to a Name (line 354):
        
        # Obtaining the type of the subscript
        int_50622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 8), 'int')
        
        # Call to quad(...): (line 354)
        # Processing the call arguments (line 354)
        # Getting the type of 'func' (line 354)
        func_50624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 27), 'func', False)
        int_50625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 33), 'int')
        int_50626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 36), 'int')
        # Processing the call keyword arguments (line 354)
        kwargs_50627 = {}
        # Getting the type of 'quad' (line 354)
        quad_50623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 22), 'quad', False)
        # Calling quad(args, kwargs) (line 354)
        quad_call_result_50628 = invoke(stypy.reporting.localization.Localization(__file__, 354, 22), quad_50623, *[func_50624, int_50625, int_50626], **kwargs_50627)
        
        # Obtaining the member '__getitem__' of a type (line 354)
        getitem___50629 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 8), quad_call_result_50628, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 354)
        subscript_call_result_50630 = invoke(stypy.reporting.localization.Localization(__file__, 354, 8), getitem___50629, int_50622)
        
        # Assigning a type to the variable 'tuple_var_assignment_49285' (line 354)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 8), 'tuple_var_assignment_49285', subscript_call_result_50630)
        
        # Assigning a Name to a Name (line 354):
        # Getting the type of 'tuple_var_assignment_49284' (line 354)
        tuple_var_assignment_49284_50631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 8), 'tuple_var_assignment_49284')
        # Assigning a type to the variable 'res' (line 354)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 8), 'res', tuple_var_assignment_49284_50631)
        
        # Assigning a Name to a Name (line 354):
        # Getting the type of 'tuple_var_assignment_49285' (line 354)
        tuple_var_assignment_49285_50632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 8), 'tuple_var_assignment_49285')
        # Assigning a type to the variable 'reserr' (line 354)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 13), 'reserr', tuple_var_assignment_49285_50632)
        
        # Assigning a Call to a Tuple (line 355):
        
        # Assigning a Subscript to a Name (line 355):
        
        # Obtaining the type of the subscript
        int_50633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 8), 'int')
        
        # Call to nquad(...): (line 355)
        # Processing the call arguments (line 355)
        # Getting the type of 'func' (line 355)
        func_50635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 30), 'func', False)
        # Processing the call keyword arguments (line 355)
        
        # Obtaining an instance of the builtin type 'list' (line 355)
        list_50636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 43), 'list')
        # Adding type elements to the builtin type 'list' instance (line 355)
        # Adding element type (line 355)
        
        # Obtaining an instance of the builtin type 'list' (line 355)
        list_50637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 44), 'list')
        # Adding type elements to the builtin type 'list' instance (line 355)
        # Adding element type (line 355)
        int_50638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 355, 44), list_50637, int_50638)
        # Adding element type (line 355)
        int_50639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 48), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 355, 44), list_50637, int_50639)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 355, 43), list_50636, list_50637)
        
        keyword_50640 = list_50636
        kwargs_50641 = {'ranges': keyword_50640}
        # Getting the type of 'nquad' (line 355)
        nquad_50634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 24), 'nquad', False)
        # Calling nquad(args, kwargs) (line 355)
        nquad_call_result_50642 = invoke(stypy.reporting.localization.Localization(__file__, 355, 24), nquad_50634, *[func_50635], **kwargs_50641)
        
        # Obtaining the member '__getitem__' of a type (line 355)
        getitem___50643 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 355, 8), nquad_call_result_50642, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 355)
        subscript_call_result_50644 = invoke(stypy.reporting.localization.Localization(__file__, 355, 8), getitem___50643, int_50633)
        
        # Assigning a type to the variable 'tuple_var_assignment_49286' (line 355)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 355, 8), 'tuple_var_assignment_49286', subscript_call_result_50644)
        
        # Assigning a Subscript to a Name (line 355):
        
        # Obtaining the type of the subscript
        int_50645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 8), 'int')
        
        # Call to nquad(...): (line 355)
        # Processing the call arguments (line 355)
        # Getting the type of 'func' (line 355)
        func_50647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 30), 'func', False)
        # Processing the call keyword arguments (line 355)
        
        # Obtaining an instance of the builtin type 'list' (line 355)
        list_50648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 43), 'list')
        # Adding type elements to the builtin type 'list' instance (line 355)
        # Adding element type (line 355)
        
        # Obtaining an instance of the builtin type 'list' (line 355)
        list_50649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 44), 'list')
        # Adding type elements to the builtin type 'list' instance (line 355)
        # Adding element type (line 355)
        int_50650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 355, 44), list_50649, int_50650)
        # Adding element type (line 355)
        int_50651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 48), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 355, 44), list_50649, int_50651)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 355, 43), list_50648, list_50649)
        
        keyword_50652 = list_50648
        kwargs_50653 = {'ranges': keyword_50652}
        # Getting the type of 'nquad' (line 355)
        nquad_50646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 24), 'nquad', False)
        # Calling nquad(args, kwargs) (line 355)
        nquad_call_result_50654 = invoke(stypy.reporting.localization.Localization(__file__, 355, 24), nquad_50646, *[func_50647], **kwargs_50653)
        
        # Obtaining the member '__getitem__' of a type (line 355)
        getitem___50655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 355, 8), nquad_call_result_50654, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 355)
        subscript_call_result_50656 = invoke(stypy.reporting.localization.Localization(__file__, 355, 8), getitem___50655, int_50645)
        
        # Assigning a type to the variable 'tuple_var_assignment_49287' (line 355)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 355, 8), 'tuple_var_assignment_49287', subscript_call_result_50656)
        
        # Assigning a Name to a Name (line 355):
        # Getting the type of 'tuple_var_assignment_49286' (line 355)
        tuple_var_assignment_49286_50657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 8), 'tuple_var_assignment_49286')
        # Assigning a type to the variable 'res2' (line 355)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 355, 8), 'res2', tuple_var_assignment_49286_50657)
        
        # Assigning a Name to a Name (line 355):
        # Getting the type of 'tuple_var_assignment_49287' (line 355)
        tuple_var_assignment_49287_50658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 8), 'tuple_var_assignment_49287')
        # Assigning a type to the variable 'reserr2' (line 355)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 355, 14), 'reserr2', tuple_var_assignment_49287_50658)
        
        # Call to assert_almost_equal(...): (line 356)
        # Processing the call arguments (line 356)
        # Getting the type of 'res' (line 356)
        res_50660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 28), 'res', False)
        # Getting the type of 'res2' (line 356)
        res2_50661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 33), 'res2', False)
        # Processing the call keyword arguments (line 356)
        kwargs_50662 = {}
        # Getting the type of 'assert_almost_equal' (line 356)
        assert_almost_equal_50659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 356)
        assert_almost_equal_call_result_50663 = invoke(stypy.reporting.localization.Localization(__file__, 356, 8), assert_almost_equal_50659, *[res_50660, res2_50661], **kwargs_50662)
        
        
        # Call to assert_almost_equal(...): (line 357)
        # Processing the call arguments (line 357)
        # Getting the type of 'reserr' (line 357)
        reserr_50665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 28), 'reserr', False)
        # Getting the type of 'reserr2' (line 357)
        reserr2_50666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 36), 'reserr2', False)
        # Processing the call keyword arguments (line 357)
        kwargs_50667 = {}
        # Getting the type of 'assert_almost_equal' (line 357)
        assert_almost_equal_50664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 357)
        assert_almost_equal_call_result_50668 = invoke(stypy.reporting.localization.Localization(__file__, 357, 8), assert_almost_equal_50664, *[reserr_50665, reserr2_50666], **kwargs_50667)
        
        
        # ################# End of 'test_matching_quad(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_matching_quad' in the type store
        # Getting the type of 'stypy_return_type' (line 350)
        stypy_return_type_50669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_50669)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_matching_quad'
        return stypy_return_type_50669


    @norecursion
    def test_matching_dblquad(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_matching_dblquad'
        module_type_store = module_type_store.open_function_context('test_matching_dblquad', 359, 4, False)
        # Assigning a type to the variable 'self' (line 360)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 360, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestNQuad.test_matching_dblquad.__dict__.__setitem__('stypy_localization', localization)
        TestNQuad.test_matching_dblquad.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestNQuad.test_matching_dblquad.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestNQuad.test_matching_dblquad.__dict__.__setitem__('stypy_function_name', 'TestNQuad.test_matching_dblquad')
        TestNQuad.test_matching_dblquad.__dict__.__setitem__('stypy_param_names_list', [])
        TestNQuad.test_matching_dblquad.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestNQuad.test_matching_dblquad.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestNQuad.test_matching_dblquad.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestNQuad.test_matching_dblquad.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestNQuad.test_matching_dblquad.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestNQuad.test_matching_dblquad.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestNQuad.test_matching_dblquad', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_matching_dblquad', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_matching_dblquad(...)' code ##################


        @norecursion
        def func2d(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'func2d'
            module_type_store = module_type_store.open_function_context('func2d', 360, 8, False)
            
            # Passed parameters checking function
            func2d.stypy_localization = localization
            func2d.stypy_type_of_self = None
            func2d.stypy_type_store = module_type_store
            func2d.stypy_function_name = 'func2d'
            func2d.stypy_param_names_list = ['x0', 'x1']
            func2d.stypy_varargs_param_name = None
            func2d.stypy_kwargs_param_name = None
            func2d.stypy_call_defaults = defaults
            func2d.stypy_call_varargs = varargs
            func2d.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'func2d', ['x0', 'x1'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'func2d', localization, ['x0', 'x1'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'func2d(...)' code ##################

            # Getting the type of 'x0' (line 361)
            x0_50670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 19), 'x0')
            int_50671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 23), 'int')
            # Applying the binary operator '**' (line 361)
            result_pow_50672 = python_operator(stypy.reporting.localization.Localization(__file__, 361, 19), '**', x0_50670, int_50671)
            
            # Getting the type of 'x1' (line 361)
            x1_50673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 27), 'x1')
            int_50674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 31), 'int')
            # Applying the binary operator '**' (line 361)
            result_pow_50675 = python_operator(stypy.reporting.localization.Localization(__file__, 361, 27), '**', x1_50673, int_50674)
            
            # Applying the binary operator '+' (line 361)
            result_add_50676 = python_operator(stypy.reporting.localization.Localization(__file__, 361, 19), '+', result_pow_50672, result_pow_50675)
            
            # Getting the type of 'x0' (line 361)
            x0_50677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 35), 'x0')
            # Getting the type of 'x1' (line 361)
            x1_50678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 40), 'x1')
            # Applying the binary operator '*' (line 361)
            result_mul_50679 = python_operator(stypy.reporting.localization.Localization(__file__, 361, 35), '*', x0_50677, x1_50678)
            
            # Applying the binary operator '-' (line 361)
            result_sub_50680 = python_operator(stypy.reporting.localization.Localization(__file__, 361, 33), '-', result_add_50676, result_mul_50679)
            
            int_50681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 45), 'int')
            # Applying the binary operator '+' (line 361)
            result_add_50682 = python_operator(stypy.reporting.localization.Localization(__file__, 361, 43), '+', result_sub_50680, int_50681)
            
            # Assigning a type to the variable 'stypy_return_type' (line 361)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 361, 12), 'stypy_return_type', result_add_50682)
            
            # ################# End of 'func2d(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'func2d' in the type store
            # Getting the type of 'stypy_return_type' (line 360)
            stypy_return_type_50683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_50683)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'func2d'
            return stypy_return_type_50683

        # Assigning a type to the variable 'func2d' (line 360)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 360, 8), 'func2d', func2d)
        
        # Assigning a Call to a Tuple (line 363):
        
        # Assigning a Subscript to a Name (line 363):
        
        # Obtaining the type of the subscript
        int_50684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, 8), 'int')
        
        # Call to dblquad(...): (line 363)
        # Processing the call arguments (line 363)
        # Getting the type of 'func2d' (line 363)
        func2d_50686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 30), 'func2d', False)
        int_50687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, 38), 'int')
        int_50688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, 42), 'int')

        @norecursion
        def _stypy_temp_lambda_34(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_34'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_34', 363, 45, True)
            # Passed parameters checking function
            _stypy_temp_lambda_34.stypy_localization = localization
            _stypy_temp_lambda_34.stypy_type_of_self = None
            _stypy_temp_lambda_34.stypy_type_store = module_type_store
            _stypy_temp_lambda_34.stypy_function_name = '_stypy_temp_lambda_34'
            _stypy_temp_lambda_34.stypy_param_names_list = ['x']
            _stypy_temp_lambda_34.stypy_varargs_param_name = None
            _stypy_temp_lambda_34.stypy_kwargs_param_name = None
            _stypy_temp_lambda_34.stypy_call_defaults = defaults
            _stypy_temp_lambda_34.stypy_call_varargs = varargs
            _stypy_temp_lambda_34.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_34', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_34', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            int_50689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, 55), 'int')
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 363)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 45), 'stypy_return_type', int_50689)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_34' in the type store
            # Getting the type of 'stypy_return_type' (line 363)
            stypy_return_type_50690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 45), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_50690)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_34'
            return stypy_return_type_50690

        # Assigning a type to the variable '_stypy_temp_lambda_34' (line 363)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 45), '_stypy_temp_lambda_34', _stypy_temp_lambda_34)
        # Getting the type of '_stypy_temp_lambda_34' (line 363)
        _stypy_temp_lambda_34_50691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 45), '_stypy_temp_lambda_34')

        @norecursion
        def _stypy_temp_lambda_35(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_35'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_35', 363, 59, True)
            # Passed parameters checking function
            _stypy_temp_lambda_35.stypy_localization = localization
            _stypy_temp_lambda_35.stypy_type_of_self = None
            _stypy_temp_lambda_35.stypy_type_store = module_type_store
            _stypy_temp_lambda_35.stypy_function_name = '_stypy_temp_lambda_35'
            _stypy_temp_lambda_35.stypy_param_names_list = ['x']
            _stypy_temp_lambda_35.stypy_varargs_param_name = None
            _stypy_temp_lambda_35.stypy_kwargs_param_name = None
            _stypy_temp_lambda_35.stypy_call_defaults = defaults
            _stypy_temp_lambda_35.stypy_call_varargs = varargs
            _stypy_temp_lambda_35.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_35', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_35', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            int_50692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, 69), 'int')
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 363)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 59), 'stypy_return_type', int_50692)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_35' in the type store
            # Getting the type of 'stypy_return_type' (line 363)
            stypy_return_type_50693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 59), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_50693)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_35'
            return stypy_return_type_50693

        # Assigning a type to the variable '_stypy_temp_lambda_35' (line 363)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 59), '_stypy_temp_lambda_35', _stypy_temp_lambda_35)
        # Getting the type of '_stypy_temp_lambda_35' (line 363)
        _stypy_temp_lambda_35_50694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 59), '_stypy_temp_lambda_35')
        # Processing the call keyword arguments (line 363)
        kwargs_50695 = {}
        # Getting the type of 'dblquad' (line 363)
        dblquad_50685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 22), 'dblquad', False)
        # Calling dblquad(args, kwargs) (line 363)
        dblquad_call_result_50696 = invoke(stypy.reporting.localization.Localization(__file__, 363, 22), dblquad_50685, *[func2d_50686, int_50687, int_50688, _stypy_temp_lambda_34_50691, _stypy_temp_lambda_35_50694], **kwargs_50695)
        
        # Obtaining the member '__getitem__' of a type (line 363)
        getitem___50697 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 8), dblquad_call_result_50696, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 363)
        subscript_call_result_50698 = invoke(stypy.reporting.localization.Localization(__file__, 363, 8), getitem___50697, int_50684)
        
        # Assigning a type to the variable 'tuple_var_assignment_49288' (line 363)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 8), 'tuple_var_assignment_49288', subscript_call_result_50698)
        
        # Assigning a Subscript to a Name (line 363):
        
        # Obtaining the type of the subscript
        int_50699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, 8), 'int')
        
        # Call to dblquad(...): (line 363)
        # Processing the call arguments (line 363)
        # Getting the type of 'func2d' (line 363)
        func2d_50701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 30), 'func2d', False)
        int_50702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, 38), 'int')
        int_50703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, 42), 'int')

        @norecursion
        def _stypy_temp_lambda_36(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_36'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_36', 363, 45, True)
            # Passed parameters checking function
            _stypy_temp_lambda_36.stypy_localization = localization
            _stypy_temp_lambda_36.stypy_type_of_self = None
            _stypy_temp_lambda_36.stypy_type_store = module_type_store
            _stypy_temp_lambda_36.stypy_function_name = '_stypy_temp_lambda_36'
            _stypy_temp_lambda_36.stypy_param_names_list = ['x']
            _stypy_temp_lambda_36.stypy_varargs_param_name = None
            _stypy_temp_lambda_36.stypy_kwargs_param_name = None
            _stypy_temp_lambda_36.stypy_call_defaults = defaults
            _stypy_temp_lambda_36.stypy_call_varargs = varargs
            _stypy_temp_lambda_36.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_36', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_36', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            int_50704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, 55), 'int')
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 363)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 45), 'stypy_return_type', int_50704)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_36' in the type store
            # Getting the type of 'stypy_return_type' (line 363)
            stypy_return_type_50705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 45), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_50705)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_36'
            return stypy_return_type_50705

        # Assigning a type to the variable '_stypy_temp_lambda_36' (line 363)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 45), '_stypy_temp_lambda_36', _stypy_temp_lambda_36)
        # Getting the type of '_stypy_temp_lambda_36' (line 363)
        _stypy_temp_lambda_36_50706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 45), '_stypy_temp_lambda_36')

        @norecursion
        def _stypy_temp_lambda_37(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_37'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_37', 363, 59, True)
            # Passed parameters checking function
            _stypy_temp_lambda_37.stypy_localization = localization
            _stypy_temp_lambda_37.stypy_type_of_self = None
            _stypy_temp_lambda_37.stypy_type_store = module_type_store
            _stypy_temp_lambda_37.stypy_function_name = '_stypy_temp_lambda_37'
            _stypy_temp_lambda_37.stypy_param_names_list = ['x']
            _stypy_temp_lambda_37.stypy_varargs_param_name = None
            _stypy_temp_lambda_37.stypy_kwargs_param_name = None
            _stypy_temp_lambda_37.stypy_call_defaults = defaults
            _stypy_temp_lambda_37.stypy_call_varargs = varargs
            _stypy_temp_lambda_37.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_37', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_37', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            int_50707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, 69), 'int')
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 363)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 59), 'stypy_return_type', int_50707)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_37' in the type store
            # Getting the type of 'stypy_return_type' (line 363)
            stypy_return_type_50708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 59), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_50708)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_37'
            return stypy_return_type_50708

        # Assigning a type to the variable '_stypy_temp_lambda_37' (line 363)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 59), '_stypy_temp_lambda_37', _stypy_temp_lambda_37)
        # Getting the type of '_stypy_temp_lambda_37' (line 363)
        _stypy_temp_lambda_37_50709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 59), '_stypy_temp_lambda_37')
        # Processing the call keyword arguments (line 363)
        kwargs_50710 = {}
        # Getting the type of 'dblquad' (line 363)
        dblquad_50700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 22), 'dblquad', False)
        # Calling dblquad(args, kwargs) (line 363)
        dblquad_call_result_50711 = invoke(stypy.reporting.localization.Localization(__file__, 363, 22), dblquad_50700, *[func2d_50701, int_50702, int_50703, _stypy_temp_lambda_36_50706, _stypy_temp_lambda_37_50709], **kwargs_50710)
        
        # Obtaining the member '__getitem__' of a type (line 363)
        getitem___50712 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 8), dblquad_call_result_50711, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 363)
        subscript_call_result_50713 = invoke(stypy.reporting.localization.Localization(__file__, 363, 8), getitem___50712, int_50699)
        
        # Assigning a type to the variable 'tuple_var_assignment_49289' (line 363)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 8), 'tuple_var_assignment_49289', subscript_call_result_50713)
        
        # Assigning a Name to a Name (line 363):
        # Getting the type of 'tuple_var_assignment_49288' (line 363)
        tuple_var_assignment_49288_50714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 8), 'tuple_var_assignment_49288')
        # Assigning a type to the variable 'res' (line 363)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 8), 'res', tuple_var_assignment_49288_50714)
        
        # Assigning a Name to a Name (line 363):
        # Getting the type of 'tuple_var_assignment_49289' (line 363)
        tuple_var_assignment_49289_50715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 8), 'tuple_var_assignment_49289')
        # Assigning a type to the variable 'reserr' (line 363)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 13), 'reserr', tuple_var_assignment_49289_50715)
        
        # Assigning a Call to a Tuple (line 364):
        
        # Assigning a Subscript to a Name (line 364):
        
        # Obtaining the type of the subscript
        int_50716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 8), 'int')
        
        # Call to nquad(...): (line 364)
        # Processing the call arguments (line 364)
        # Getting the type of 'func2d' (line 364)
        func2d_50718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 30), 'func2d', False)
        
        # Obtaining an instance of the builtin type 'list' (line 364)
        list_50719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 38), 'list')
        # Adding type elements to the builtin type 'list' instance (line 364)
        # Adding element type (line 364)
        
        # Obtaining an instance of the builtin type 'list' (line 364)
        list_50720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 39), 'list')
        # Adding type elements to the builtin type 'list' instance (line 364)
        # Adding element type (line 364)
        int_50721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 364, 39), list_50720, int_50721)
        # Adding element type (line 364)
        int_50722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 364, 39), list_50720, int_50722)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 364, 38), list_50719, list_50720)
        # Adding element type (line 364)
        
        # Obtaining an instance of the builtin type 'tuple' (line 364)
        tuple_50723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 49), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 364)
        # Adding element type (line 364)
        int_50724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 49), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 364, 49), tuple_50723, int_50724)
        # Adding element type (line 364)
        int_50725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 53), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 364, 49), tuple_50723, int_50725)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 364, 38), list_50719, tuple_50723)
        
        # Processing the call keyword arguments (line 364)
        kwargs_50726 = {}
        # Getting the type of 'nquad' (line 364)
        nquad_50717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 24), 'nquad', False)
        # Calling nquad(args, kwargs) (line 364)
        nquad_call_result_50727 = invoke(stypy.reporting.localization.Localization(__file__, 364, 24), nquad_50717, *[func2d_50718, list_50719], **kwargs_50726)
        
        # Obtaining the member '__getitem__' of a type (line 364)
        getitem___50728 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 364, 8), nquad_call_result_50727, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 364)
        subscript_call_result_50729 = invoke(stypy.reporting.localization.Localization(__file__, 364, 8), getitem___50728, int_50716)
        
        # Assigning a type to the variable 'tuple_var_assignment_49290' (line 364)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 364, 8), 'tuple_var_assignment_49290', subscript_call_result_50729)
        
        # Assigning a Subscript to a Name (line 364):
        
        # Obtaining the type of the subscript
        int_50730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 8), 'int')
        
        # Call to nquad(...): (line 364)
        # Processing the call arguments (line 364)
        # Getting the type of 'func2d' (line 364)
        func2d_50732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 30), 'func2d', False)
        
        # Obtaining an instance of the builtin type 'list' (line 364)
        list_50733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 38), 'list')
        # Adding type elements to the builtin type 'list' instance (line 364)
        # Adding element type (line 364)
        
        # Obtaining an instance of the builtin type 'list' (line 364)
        list_50734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 39), 'list')
        # Adding type elements to the builtin type 'list' instance (line 364)
        # Adding element type (line 364)
        int_50735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 364, 39), list_50734, int_50735)
        # Adding element type (line 364)
        int_50736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 364, 39), list_50734, int_50736)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 364, 38), list_50733, list_50734)
        # Adding element type (line 364)
        
        # Obtaining an instance of the builtin type 'tuple' (line 364)
        tuple_50737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 49), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 364)
        # Adding element type (line 364)
        int_50738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 49), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 364, 49), tuple_50737, int_50738)
        # Adding element type (line 364)
        int_50739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 53), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 364, 49), tuple_50737, int_50739)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 364, 38), list_50733, tuple_50737)
        
        # Processing the call keyword arguments (line 364)
        kwargs_50740 = {}
        # Getting the type of 'nquad' (line 364)
        nquad_50731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 24), 'nquad', False)
        # Calling nquad(args, kwargs) (line 364)
        nquad_call_result_50741 = invoke(stypy.reporting.localization.Localization(__file__, 364, 24), nquad_50731, *[func2d_50732, list_50733], **kwargs_50740)
        
        # Obtaining the member '__getitem__' of a type (line 364)
        getitem___50742 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 364, 8), nquad_call_result_50741, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 364)
        subscript_call_result_50743 = invoke(stypy.reporting.localization.Localization(__file__, 364, 8), getitem___50742, int_50730)
        
        # Assigning a type to the variable 'tuple_var_assignment_49291' (line 364)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 364, 8), 'tuple_var_assignment_49291', subscript_call_result_50743)
        
        # Assigning a Name to a Name (line 364):
        # Getting the type of 'tuple_var_assignment_49290' (line 364)
        tuple_var_assignment_49290_50744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 8), 'tuple_var_assignment_49290')
        # Assigning a type to the variable 'res2' (line 364)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 364, 8), 'res2', tuple_var_assignment_49290_50744)
        
        # Assigning a Name to a Name (line 364):
        # Getting the type of 'tuple_var_assignment_49291' (line 364)
        tuple_var_assignment_49291_50745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 8), 'tuple_var_assignment_49291')
        # Assigning a type to the variable 'reserr2' (line 364)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 364, 14), 'reserr2', tuple_var_assignment_49291_50745)
        
        # Call to assert_almost_equal(...): (line 365)
        # Processing the call arguments (line 365)
        # Getting the type of 'res' (line 365)
        res_50747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 28), 'res', False)
        # Getting the type of 'res2' (line 365)
        res2_50748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 33), 'res2', False)
        # Processing the call keyword arguments (line 365)
        kwargs_50749 = {}
        # Getting the type of 'assert_almost_equal' (line 365)
        assert_almost_equal_50746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 365)
        assert_almost_equal_call_result_50750 = invoke(stypy.reporting.localization.Localization(__file__, 365, 8), assert_almost_equal_50746, *[res_50747, res2_50748], **kwargs_50749)
        
        
        # Call to assert_almost_equal(...): (line 366)
        # Processing the call arguments (line 366)
        # Getting the type of 'reserr' (line 366)
        reserr_50752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 28), 'reserr', False)
        # Getting the type of 'reserr2' (line 366)
        reserr2_50753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 36), 'reserr2', False)
        # Processing the call keyword arguments (line 366)
        kwargs_50754 = {}
        # Getting the type of 'assert_almost_equal' (line 366)
        assert_almost_equal_50751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 366)
        assert_almost_equal_call_result_50755 = invoke(stypy.reporting.localization.Localization(__file__, 366, 8), assert_almost_equal_50751, *[reserr_50752, reserr2_50753], **kwargs_50754)
        
        
        # ################# End of 'test_matching_dblquad(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_matching_dblquad' in the type store
        # Getting the type of 'stypy_return_type' (line 359)
        stypy_return_type_50756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_50756)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_matching_dblquad'
        return stypy_return_type_50756


    @norecursion
    def test_matching_tplquad(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_matching_tplquad'
        module_type_store = module_type_store.open_function_context('test_matching_tplquad', 368, 4, False)
        # Assigning a type to the variable 'self' (line 369)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 369, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestNQuad.test_matching_tplquad.__dict__.__setitem__('stypy_localization', localization)
        TestNQuad.test_matching_tplquad.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestNQuad.test_matching_tplquad.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestNQuad.test_matching_tplquad.__dict__.__setitem__('stypy_function_name', 'TestNQuad.test_matching_tplquad')
        TestNQuad.test_matching_tplquad.__dict__.__setitem__('stypy_param_names_list', [])
        TestNQuad.test_matching_tplquad.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestNQuad.test_matching_tplquad.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestNQuad.test_matching_tplquad.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestNQuad.test_matching_tplquad.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestNQuad.test_matching_tplquad.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestNQuad.test_matching_tplquad.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestNQuad.test_matching_tplquad', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_matching_tplquad', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_matching_tplquad(...)' code ##################


        @norecursion
        def func3d(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'func3d'
            module_type_store = module_type_store.open_function_context('func3d', 369, 8, False)
            
            # Passed parameters checking function
            func3d.stypy_localization = localization
            func3d.stypy_type_of_self = None
            func3d.stypy_type_store = module_type_store
            func3d.stypy_function_name = 'func3d'
            func3d.stypy_param_names_list = ['x0', 'x1', 'x2', 'c0', 'c1']
            func3d.stypy_varargs_param_name = None
            func3d.stypy_kwargs_param_name = None
            func3d.stypy_call_defaults = defaults
            func3d.stypy_call_varargs = varargs
            func3d.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'func3d', ['x0', 'x1', 'x2', 'c0', 'c1'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'func3d', localization, ['x0', 'x1', 'x2', 'c0', 'c1'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'func3d(...)' code ##################

            # Getting the type of 'x0' (line 370)
            x0_50757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 19), 'x0')
            int_50758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 23), 'int')
            # Applying the binary operator '**' (line 370)
            result_pow_50759 = python_operator(stypy.reporting.localization.Localization(__file__, 370, 19), '**', x0_50757, int_50758)
            
            # Getting the type of 'c0' (line 370)
            c0_50760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 27), 'c0')
            # Getting the type of 'x1' (line 370)
            x1_50761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 32), 'x1')
            int_50762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 36), 'int')
            # Applying the binary operator '**' (line 370)
            result_pow_50763 = python_operator(stypy.reporting.localization.Localization(__file__, 370, 32), '**', x1_50761, int_50762)
            
            # Applying the binary operator '*' (line 370)
            result_mul_50764 = python_operator(stypy.reporting.localization.Localization(__file__, 370, 27), '*', c0_50760, result_pow_50763)
            
            # Applying the binary operator '+' (line 370)
            result_add_50765 = python_operator(stypy.reporting.localization.Localization(__file__, 370, 19), '+', result_pow_50759, result_mul_50764)
            
            # Getting the type of 'x0' (line 370)
            x0_50766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 40), 'x0')
            # Getting the type of 'x1' (line 370)
            x1_50767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 45), 'x1')
            # Applying the binary operator '*' (line 370)
            result_mul_50768 = python_operator(stypy.reporting.localization.Localization(__file__, 370, 40), '*', x0_50766, x1_50767)
            
            # Applying the binary operator '-' (line 370)
            result_sub_50769 = python_operator(stypy.reporting.localization.Localization(__file__, 370, 38), '-', result_add_50765, result_mul_50768)
            
            int_50770 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 50), 'int')
            # Applying the binary operator '+' (line 370)
            result_add_50771 = python_operator(stypy.reporting.localization.Localization(__file__, 370, 48), '+', result_sub_50769, int_50770)
            
            # Getting the type of 'c1' (line 370)
            c1_50772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 54), 'c1')
            
            # Call to sin(...): (line 370)
            # Processing the call arguments (line 370)
            # Getting the type of 'x2' (line 370)
            x2_50775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 66), 'x2', False)
            # Processing the call keyword arguments (line 370)
            kwargs_50776 = {}
            # Getting the type of 'np' (line 370)
            np_50773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 59), 'np', False)
            # Obtaining the member 'sin' of a type (line 370)
            sin_50774 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 59), np_50773, 'sin')
            # Calling sin(args, kwargs) (line 370)
            sin_call_result_50777 = invoke(stypy.reporting.localization.Localization(__file__, 370, 59), sin_50774, *[x2_50775], **kwargs_50776)
            
            # Applying the binary operator '*' (line 370)
            result_mul_50778 = python_operator(stypy.reporting.localization.Localization(__file__, 370, 54), '*', c1_50772, sin_call_result_50777)
            
            # Applying the binary operator '+' (line 370)
            result_add_50779 = python_operator(stypy.reporting.localization.Localization(__file__, 370, 52), '+', result_add_50771, result_mul_50778)
            
            # Assigning a type to the variable 'stypy_return_type' (line 370)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 12), 'stypy_return_type', result_add_50779)
            
            # ################# End of 'func3d(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'func3d' in the type store
            # Getting the type of 'stypy_return_type' (line 369)
            stypy_return_type_50780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_50780)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'func3d'
            return stypy_return_type_50780

        # Assigning a type to the variable 'func3d' (line 369)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 369, 8), 'func3d', func3d)
        
        # Assigning a Call to a Name (line 372):
        
        # Assigning a Call to a Name (line 372):
        
        # Call to tplquad(...): (line 372)
        # Processing the call arguments (line 372)
        # Getting the type of 'func3d' (line 372)
        func3d_50782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 22), 'func3d', False)
        int_50783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, 30), 'int')
        int_50784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, 34), 'int')

        @norecursion
        def _stypy_temp_lambda_38(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_38'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_38', 372, 37, True)
            # Passed parameters checking function
            _stypy_temp_lambda_38.stypy_localization = localization
            _stypy_temp_lambda_38.stypy_type_of_self = None
            _stypy_temp_lambda_38.stypy_type_store = module_type_store
            _stypy_temp_lambda_38.stypy_function_name = '_stypy_temp_lambda_38'
            _stypy_temp_lambda_38.stypy_param_names_list = ['x']
            _stypy_temp_lambda_38.stypy_varargs_param_name = None
            _stypy_temp_lambda_38.stypy_kwargs_param_name = None
            _stypy_temp_lambda_38.stypy_call_defaults = defaults
            _stypy_temp_lambda_38.stypy_call_varargs = varargs
            _stypy_temp_lambda_38.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_38', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_38', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            int_50785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, 47), 'int')
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 372)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 372, 37), 'stypy_return_type', int_50785)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_38' in the type store
            # Getting the type of 'stypy_return_type' (line 372)
            stypy_return_type_50786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 37), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_50786)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_38'
            return stypy_return_type_50786

        # Assigning a type to the variable '_stypy_temp_lambda_38' (line 372)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 372, 37), '_stypy_temp_lambda_38', _stypy_temp_lambda_38)
        # Getting the type of '_stypy_temp_lambda_38' (line 372)
        _stypy_temp_lambda_38_50787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 37), '_stypy_temp_lambda_38')

        @norecursion
        def _stypy_temp_lambda_39(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_39'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_39', 372, 51, True)
            # Passed parameters checking function
            _stypy_temp_lambda_39.stypy_localization = localization
            _stypy_temp_lambda_39.stypy_type_of_self = None
            _stypy_temp_lambda_39.stypy_type_store = module_type_store
            _stypy_temp_lambda_39.stypy_function_name = '_stypy_temp_lambda_39'
            _stypy_temp_lambda_39.stypy_param_names_list = ['x']
            _stypy_temp_lambda_39.stypy_varargs_param_name = None
            _stypy_temp_lambda_39.stypy_kwargs_param_name = None
            _stypy_temp_lambda_39.stypy_call_defaults = defaults
            _stypy_temp_lambda_39.stypy_call_varargs = varargs
            _stypy_temp_lambda_39.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_39', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_39', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            int_50788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, 61), 'int')
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 372)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 372, 51), 'stypy_return_type', int_50788)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_39' in the type store
            # Getting the type of 'stypy_return_type' (line 372)
            stypy_return_type_50789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 51), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_50789)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_39'
            return stypy_return_type_50789

        # Assigning a type to the variable '_stypy_temp_lambda_39' (line 372)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 372, 51), '_stypy_temp_lambda_39', _stypy_temp_lambda_39)
        # Getting the type of '_stypy_temp_lambda_39' (line 372)
        _stypy_temp_lambda_39_50790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 51), '_stypy_temp_lambda_39')

        @norecursion
        def _stypy_temp_lambda_40(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_40'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_40', 373, 22, True)
            # Passed parameters checking function
            _stypy_temp_lambda_40.stypy_localization = localization
            _stypy_temp_lambda_40.stypy_type_of_self = None
            _stypy_temp_lambda_40.stypy_type_store = module_type_store
            _stypy_temp_lambda_40.stypy_function_name = '_stypy_temp_lambda_40'
            _stypy_temp_lambda_40.stypy_param_names_list = ['x', 'y']
            _stypy_temp_lambda_40.stypy_varargs_param_name = None
            _stypy_temp_lambda_40.stypy_kwargs_param_name = None
            _stypy_temp_lambda_40.stypy_call_defaults = defaults
            _stypy_temp_lambda_40.stypy_call_varargs = varargs
            _stypy_temp_lambda_40.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_40', ['x', 'y'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_40', ['x', 'y'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Getting the type of 'np' (line 373)
            np_50791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 36), 'np', False)
            # Obtaining the member 'pi' of a type (line 373)
            pi_50792 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 36), np_50791, 'pi')
            # Applying the 'usub' unary operator (line 373)
            result___neg___50793 = python_operator(stypy.reporting.localization.Localization(__file__, 373, 35), 'usub', pi_50792)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 373)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 373, 22), 'stypy_return_type', result___neg___50793)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_40' in the type store
            # Getting the type of 'stypy_return_type' (line 373)
            stypy_return_type_50794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 22), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_50794)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_40'
            return stypy_return_type_50794

        # Assigning a type to the variable '_stypy_temp_lambda_40' (line 373)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 373, 22), '_stypy_temp_lambda_40', _stypy_temp_lambda_40)
        # Getting the type of '_stypy_temp_lambda_40' (line 373)
        _stypy_temp_lambda_40_50795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 22), '_stypy_temp_lambda_40')

        @norecursion
        def _stypy_temp_lambda_41(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_41'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_41', 373, 43, True)
            # Passed parameters checking function
            _stypy_temp_lambda_41.stypy_localization = localization
            _stypy_temp_lambda_41.stypy_type_of_self = None
            _stypy_temp_lambda_41.stypy_type_store = module_type_store
            _stypy_temp_lambda_41.stypy_function_name = '_stypy_temp_lambda_41'
            _stypy_temp_lambda_41.stypy_param_names_list = ['x', 'y']
            _stypy_temp_lambda_41.stypy_varargs_param_name = None
            _stypy_temp_lambda_41.stypy_kwargs_param_name = None
            _stypy_temp_lambda_41.stypy_call_defaults = defaults
            _stypy_temp_lambda_41.stypy_call_varargs = varargs
            _stypy_temp_lambda_41.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_41', ['x', 'y'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_41', ['x', 'y'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            # Getting the type of 'np' (line 373)
            np_50796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 56), 'np', False)
            # Obtaining the member 'pi' of a type (line 373)
            pi_50797 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 56), np_50796, 'pi')
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 373)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 373, 43), 'stypy_return_type', pi_50797)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_41' in the type store
            # Getting the type of 'stypy_return_type' (line 373)
            stypy_return_type_50798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 43), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_50798)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_41'
            return stypy_return_type_50798

        # Assigning a type to the variable '_stypy_temp_lambda_41' (line 373)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 373, 43), '_stypy_temp_lambda_41', _stypy_temp_lambda_41)
        # Getting the type of '_stypy_temp_lambda_41' (line 373)
        _stypy_temp_lambda_41_50799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 43), '_stypy_temp_lambda_41')
        # Processing the call keyword arguments (line 372)
        
        # Obtaining an instance of the builtin type 'tuple' (line 374)
        tuple_50800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 374)
        # Adding element type (line 374)
        int_50801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 374, 28), tuple_50800, int_50801)
        # Adding element type (line 374)
        int_50802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 374, 28), tuple_50800, int_50802)
        
        keyword_50803 = tuple_50800
        kwargs_50804 = {'args': keyword_50803}
        # Getting the type of 'tplquad' (line 372)
        tplquad_50781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 14), 'tplquad', False)
        # Calling tplquad(args, kwargs) (line 372)
        tplquad_call_result_50805 = invoke(stypy.reporting.localization.Localization(__file__, 372, 14), tplquad_50781, *[func3d_50782, int_50783, int_50784, _stypy_temp_lambda_38_50787, _stypy_temp_lambda_39_50790, _stypy_temp_lambda_40_50795, _stypy_temp_lambda_41_50799], **kwargs_50804)
        
        # Assigning a type to the variable 'res' (line 372)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 372, 8), 'res', tplquad_call_result_50805)
        
        # Assigning a Call to a Name (line 375):
        
        # Assigning a Call to a Name (line 375):
        
        # Call to nquad(...): (line 375)
        # Processing the call arguments (line 375)
        # Getting the type of 'func3d' (line 375)
        func3d_50807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 21), 'func3d', False)
        
        # Obtaining an instance of the builtin type 'list' (line 375)
        list_50808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 375)
        # Adding element type (line 375)
        
        # Obtaining an instance of the builtin type 'list' (line 375)
        list_50809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 375)
        # Adding element type (line 375)
        
        # Getting the type of 'np' (line 375)
        np_50810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 32), 'np', False)
        # Obtaining the member 'pi' of a type (line 375)
        pi_50811 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 375, 32), np_50810, 'pi')
        # Applying the 'usub' unary operator (line 375)
        result___neg___50812 = python_operator(stypy.reporting.localization.Localization(__file__, 375, 31), 'usub', pi_50811)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 375, 30), list_50809, result___neg___50812)
        # Adding element type (line 375)
        # Getting the type of 'np' (line 375)
        np_50813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 39), 'np', False)
        # Obtaining the member 'pi' of a type (line 375)
        pi_50814 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 375, 39), np_50813, 'pi')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 375, 30), list_50809, pi_50814)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 375, 29), list_50808, list_50809)
        # Adding element type (line 375)
        
        # Obtaining an instance of the builtin type 'list' (line 375)
        list_50815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 47), 'list')
        # Adding type elements to the builtin type 'list' instance (line 375)
        # Adding element type (line 375)
        int_50816 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 48), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 375, 47), list_50815, int_50816)
        # Adding element type (line 375)
        int_50817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 52), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 375, 47), list_50815, int_50817)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 375, 29), list_50808, list_50815)
        # Adding element type (line 375)
        
        # Obtaining an instance of the builtin type 'tuple' (line 375)
        tuple_50818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 57), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 375)
        # Adding element type (line 375)
        int_50819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 57), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 375, 57), tuple_50818, int_50819)
        # Adding element type (line 375)
        int_50820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 61), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 375, 57), tuple_50818, int_50820)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 375, 29), list_50808, tuple_50818)
        
        # Processing the call keyword arguments (line 375)
        
        # Obtaining an instance of the builtin type 'tuple' (line 375)
        tuple_50821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 72), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 375)
        # Adding element type (line 375)
        int_50822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 72), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 375, 72), tuple_50821, int_50822)
        # Adding element type (line 375)
        int_50823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 75), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 375, 72), tuple_50821, int_50823)
        
        keyword_50824 = tuple_50821
        kwargs_50825 = {'args': keyword_50824}
        # Getting the type of 'nquad' (line 375)
        nquad_50806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 15), 'nquad', False)
        # Calling nquad(args, kwargs) (line 375)
        nquad_call_result_50826 = invoke(stypy.reporting.localization.Localization(__file__, 375, 15), nquad_50806, *[func3d_50807, list_50808], **kwargs_50825)
        
        # Assigning a type to the variable 'res2' (line 375)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 375, 8), 'res2', nquad_call_result_50826)
        
        # Call to assert_almost_equal(...): (line 376)
        # Processing the call arguments (line 376)
        # Getting the type of 'res' (line 376)
        res_50828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 28), 'res', False)
        # Getting the type of 'res2' (line 376)
        res2_50829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 33), 'res2', False)
        # Processing the call keyword arguments (line 376)
        kwargs_50830 = {}
        # Getting the type of 'assert_almost_equal' (line 376)
        assert_almost_equal_50827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 376)
        assert_almost_equal_call_result_50831 = invoke(stypy.reporting.localization.Localization(__file__, 376, 8), assert_almost_equal_50827, *[res_50828, res2_50829], **kwargs_50830)
        
        
        # ################# End of 'test_matching_tplquad(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_matching_tplquad' in the type store
        # Getting the type of 'stypy_return_type' (line 368)
        stypy_return_type_50832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_50832)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_matching_tplquad'
        return stypy_return_type_50832


    @norecursion
    def test_dict_as_opts(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_dict_as_opts'
        module_type_store = module_type_store.open_function_context('test_dict_as_opts', 378, 4, False)
        # Assigning a type to the variable 'self' (line 379)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 379, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestNQuad.test_dict_as_opts.__dict__.__setitem__('stypy_localization', localization)
        TestNQuad.test_dict_as_opts.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestNQuad.test_dict_as_opts.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestNQuad.test_dict_as_opts.__dict__.__setitem__('stypy_function_name', 'TestNQuad.test_dict_as_opts')
        TestNQuad.test_dict_as_opts.__dict__.__setitem__('stypy_param_names_list', [])
        TestNQuad.test_dict_as_opts.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestNQuad.test_dict_as_opts.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestNQuad.test_dict_as_opts.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestNQuad.test_dict_as_opts.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestNQuad.test_dict_as_opts.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestNQuad.test_dict_as_opts.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestNQuad.test_dict_as_opts', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_dict_as_opts', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_dict_as_opts(...)' code ##################

        
        
        # SSA begins for try-except statement (line 379)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Name (line 380):
        
        # Assigning a Call to a Name (line 380):
        
        # Call to nquad(...): (line 380)
        # Processing the call arguments (line 380)

        @norecursion
        def _stypy_temp_lambda_42(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_42'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_42', 380, 24, True)
            # Passed parameters checking function
            _stypy_temp_lambda_42.stypy_localization = localization
            _stypy_temp_lambda_42.stypy_type_of_self = None
            _stypy_temp_lambda_42.stypy_type_store = module_type_store
            _stypy_temp_lambda_42.stypy_function_name = '_stypy_temp_lambda_42'
            _stypy_temp_lambda_42.stypy_param_names_list = ['x', 'y']
            _stypy_temp_lambda_42.stypy_varargs_param_name = None
            _stypy_temp_lambda_42.stypy_kwargs_param_name = None
            _stypy_temp_lambda_42.stypy_call_defaults = defaults
            _stypy_temp_lambda_42.stypy_call_varargs = varargs
            _stypy_temp_lambda_42.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_42', ['x', 'y'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_42', ['x', 'y'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            # Getting the type of 'x' (line 380)
            x_50834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 37), 'x', False)
            # Getting the type of 'y' (line 380)
            y_50835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 41), 'y', False)
            # Applying the binary operator '*' (line 380)
            result_mul_50836 = python_operator(stypy.reporting.localization.Localization(__file__, 380, 37), '*', x_50834, y_50835)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 380)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 380, 24), 'stypy_return_type', result_mul_50836)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_42' in the type store
            # Getting the type of 'stypy_return_type' (line 380)
            stypy_return_type_50837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 24), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_50837)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_42'
            return stypy_return_type_50837

        # Assigning a type to the variable '_stypy_temp_lambda_42' (line 380)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 380, 24), '_stypy_temp_lambda_42', _stypy_temp_lambda_42)
        # Getting the type of '_stypy_temp_lambda_42' (line 380)
        _stypy_temp_lambda_42_50838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 24), '_stypy_temp_lambda_42')
        
        # Obtaining an instance of the builtin type 'list' (line 380)
        list_50839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 380, 44), 'list')
        # Adding type elements to the builtin type 'list' instance (line 380)
        # Adding element type (line 380)
        
        # Obtaining an instance of the builtin type 'list' (line 380)
        list_50840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 380, 45), 'list')
        # Adding type elements to the builtin type 'list' instance (line 380)
        # Adding element type (line 380)
        int_50841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 380, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 380, 45), list_50840, int_50841)
        # Adding element type (line 380)
        int_50842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 380, 49), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 380, 45), list_50840, int_50842)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 380, 44), list_50839, list_50840)
        # Adding element type (line 380)
        
        # Obtaining an instance of the builtin type 'list' (line 380)
        list_50843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 380, 53), 'list')
        # Adding type elements to the builtin type 'list' instance (line 380)
        # Adding element type (line 380)
        int_50844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 380, 54), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 380, 53), list_50843, int_50844)
        # Adding element type (line 380)
        int_50845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 380, 57), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 380, 53), list_50843, int_50845)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 380, 44), list_50839, list_50843)
        
        # Processing the call keyword arguments (line 380)
        
        # Obtaining an instance of the builtin type 'dict' (line 380)
        dict_50846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 380, 67), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 380)
        # Adding element type (key, value) (line 380)
        str_50847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 380, 68), 'str', 'epsrel')
        float_50848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 380, 78), 'float')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 380, 67), dict_50846, (str_50847, float_50848))
        
        keyword_50849 = dict_50846
        kwargs_50850 = {'opts': keyword_50849}
        # Getting the type of 'nquad' (line 380)
        nquad_50833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 18), 'nquad', False)
        # Calling nquad(args, kwargs) (line 380)
        nquad_call_result_50851 = invoke(stypy.reporting.localization.Localization(__file__, 380, 18), nquad_50833, *[_stypy_temp_lambda_42_50838, list_50839], **kwargs_50850)
        
        # Assigning a type to the variable 'out' (line 380)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 380, 12), 'out', nquad_call_result_50851)
        # SSA branch for the except part of a try statement (line 379)
        # SSA branch for the except 'TypeError' branch of a try statement (line 379)
        module_type_store.open_ssa_branch('except')
        # Evaluating assert statement condition
        # Getting the type of 'False' (line 382)
        False_50852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 19), 'False')
        # SSA join for try-except statement (line 379)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_dict_as_opts(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_dict_as_opts' in the type store
        # Getting the type of 'stypy_return_type' (line 378)
        stypy_return_type_50853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_50853)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_dict_as_opts'
        return stypy_return_type_50853


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 248, 0, False)
        # Assigning a type to the variable 'self' (line 249)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestNQuad.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestNQuad' (line 248)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 0), 'TestNQuad', TestNQuad)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
