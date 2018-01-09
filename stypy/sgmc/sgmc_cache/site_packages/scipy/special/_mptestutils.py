
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: import os
4: import sys
5: import time
6: 
7: import numpy as np
8: from numpy.testing import assert_
9: import pytest
10: 
11: from scipy._lib.six import reraise
12: from scipy.special._testutils import assert_func_equal
13: 
14: try:
15:     import mpmath
16: except ImportError:
17:     pass
18: 
19: 
20: # ------------------------------------------------------------------------------
21: # Machinery for systematic tests with mpmath
22: # ------------------------------------------------------------------------------
23: 
24: class Arg(object):
25:     '''
26:     Generate a set of numbers on the real axis, concentrating on
27:     'interesting' regions and covering all orders of magnitude.
28:     '''
29: 
30:     def __init__(self, a=-np.inf, b=np.inf, inclusive_a=True, inclusive_b=True):
31:         self.a = a
32:         self.b = b
33:         self.inclusive_a = inclusive_a
34:         self.inclusive_b = inclusive_b
35:         if self.a == -np.inf:
36:             self.a = -np.finfo(float).max/2
37:         if self.b == np.inf:
38:             self.b = np.finfo(float).max/2
39: 
40:     def values(self, n):
41:         '''Return an array containing approximatively `n` numbers.'''
42:         n1 = max(2, int(0.3*n))
43:         n2 = max(2, int(0.2*n))
44:         n3 = max(8, n - n1 - n2)
45: 
46:         v1 = np.linspace(-1, 1, n1)
47:         v2 = np.r_[np.linspace(-10, 10, max(0, n2-4)),
48:                    -9, -5.5, 5.5, 9]
49:         if self.a >= 0 and self.b > 0:
50:             v3 = np.r_[
51:                 np.logspace(-30, -1, 2 + n3//4),
52:                 np.logspace(5, np.log10(self.b), 1 + n3//4),
53:                 ]
54:             v4 = np.logspace(1, 5, 1 + n3//2)
55:         elif self.a < 0 < self.b:
56:             v3 = np.r_[
57:                 np.logspace(-30, -1, 2 + n3//8),
58:                 np.logspace(5, np.log10(self.b), 1 + n3//8),
59:                 -np.logspace(-30, -1, 2 + n3//8),
60:                 -np.logspace(5, np.log10(-self.a), 1 + n3//8)
61:                 ]
62:             v4 = np.r_[
63:                 np.logspace(1, 5, 1 + n3//4),
64:                 -np.logspace(1, 5, 1 + n3//4)
65:                 ]
66:         elif self.b < 0:
67:             v3 = np.r_[
68:                 -np.logspace(-30, -1, 2 + n3//4),
69:                 -np.logspace(5, np.log10(-self.b), 1 + n3//4),
70:                 ]
71:             v4 = -np.logspace(1, 5, 1 + n3//2)
72:         else:
73:             v3 = []
74:             v4 = []
75:         v = np.r_[v1, v2, v3, v4, 0]
76:         if self.inclusive_a:
77:             v = v[v >= self.a]
78:         else:
79:             v = v[v > self.a]
80:         if self.inclusive_b:
81:             v = v[v <= self.b]
82:         else:
83:             v = v[v < self.b]
84:         return np.unique(v)
85: 
86: 
87: class FixedArg(object):
88:     def __init__(self, values):
89:         self._values = np.asarray(values)
90: 
91:     def values(self, n):
92:         return self._values
93: 
94: 
95: class ComplexArg(object):
96:     def __init__(self, a=complex(-np.inf, -np.inf), b=complex(np.inf, np.inf)):
97:         self.real = Arg(a.real, b.real)
98:         self.imag = Arg(a.imag, b.imag)
99: 
100:     def values(self, n):
101:         m = max(2, int(np.sqrt(n)))
102:         x = self.real.values(m)
103:         y = self.imag.values(m)
104:         return (x[:,None] + 1j*y[None,:]).ravel()
105: 
106: 
107: class IntArg(object):
108:     def __init__(self, a=-1000, b=1000):
109:         self.a = a
110:         self.b = b
111: 
112:     def values(self, n):
113:         v1 = Arg(self.a, self.b).values(max(1 + n//2, n-5)).astype(int)
114:         v2 = np.arange(-5, 5)
115:         v = np.unique(np.r_[v1, v2])
116:         v = v[(v >= self.a) & (v < self.b)]
117:         return v
118: 
119: 
120: def get_args(argspec, n):
121:     if isinstance(argspec, np.ndarray):
122:         args = argspec.copy()
123:     else:
124:         nargs = len(argspec)
125:         ms = np.asarray([1.5 if isinstance(spec, ComplexArg) else 1.0 for spec in argspec])
126:         ms = (n**(ms/sum(ms))).astype(int) + 1
127: 
128:         args = []
129:         for spec, m in zip(argspec, ms):
130:             args.append(spec.values(m))
131:         args = np.array(np.broadcast_arrays(*np.ix_(*args))).reshape(nargs, -1).T
132: 
133:     return args
134: 
135: 
136: class MpmathData(object):
137:     def __init__(self, scipy_func, mpmath_func, arg_spec, name=None,
138:                  dps=None, prec=None, n=None, rtol=1e-7, atol=1e-300,
139:                  ignore_inf_sign=False, distinguish_nan_and_inf=True,
140:                  nan_ok=True, param_filter=None):
141: 
142:         # mpmath tests are really slow (see gh-6989).  Use a small number of
143:         # points by default, increase back to 5000 (old default) if XSLOW is
144:         # set
145:         if n is None:
146:             try:
147:                 is_xslow = int(os.environ.get('SCIPY_XSLOW', '0'))
148:             except ValueError:
149:                 is_xslow = False
150: 
151:             n = 5000 if is_xslow else 500
152: 
153:         self.scipy_func = scipy_func
154:         self.mpmath_func = mpmath_func
155:         self.arg_spec = arg_spec
156:         self.dps = dps
157:         self.prec = prec
158:         self.n = n
159:         self.rtol = rtol
160:         self.atol = atol
161:         self.ignore_inf_sign = ignore_inf_sign
162:         self.nan_ok = nan_ok
163:         if isinstance(self.arg_spec, np.ndarray):
164:             self.is_complex = np.issubdtype(self.arg_spec.dtype, np.complexfloating)
165:         else:
166:             self.is_complex = any([isinstance(arg, ComplexArg) for arg in self.arg_spec])
167:         self.ignore_inf_sign = ignore_inf_sign
168:         self.distinguish_nan_and_inf = distinguish_nan_and_inf
169:         if not name or name == '<lambda>':
170:             name = getattr(scipy_func, '__name__', None)
171:         if not name or name == '<lambda>':
172:             name = getattr(mpmath_func, '__name__', None)
173:         self.name = name
174:         self.param_filter = param_filter
175: 
176:     def check(self):
177:         np.random.seed(1234)
178: 
179:         # Generate values for the arguments
180:         argarr = get_args(self.arg_spec, self.n)
181: 
182:         # Check
183:         old_dps, old_prec = mpmath.mp.dps, mpmath.mp.prec
184:         try:
185:             if self.dps is not None:
186:                 dps_list = [self.dps]
187:             else:
188:                 dps_list = [20]
189:             if self.prec is not None:
190:                 mpmath.mp.prec = self.prec
191: 
192:             # Proper casting of mpmath input and output types. Using
193:             # native mpmath types as inputs gives improved precision
194:             # in some cases.
195:             if np.issubdtype(argarr.dtype, np.complexfloating):
196:                 pytype = mpc2complex
197: 
198:                 def mptype(x):
199:                     return mpmath.mpc(complex(x))
200:             else:
201:                 def mptype(x):
202:                     return mpmath.mpf(float(x))
203: 
204:                 def pytype(x):
205:                     if abs(x.imag) > 1e-16*(1 + abs(x.real)):
206:                         return np.nan
207:                     else:
208:                         return mpf2float(x.real)
209: 
210:             # Try out different dps until one (or none) works
211:             for j, dps in enumerate(dps_list):
212:                 mpmath.mp.dps = dps
213: 
214:                 try:
215:                     assert_func_equal(self.scipy_func,
216:                                       lambda *a: pytype(self.mpmath_func(*map(mptype, a))),
217:                                       argarr,
218:                                       vectorized=False,
219:                                       rtol=self.rtol, atol=self.atol,
220:                                       ignore_inf_sign=self.ignore_inf_sign,
221:                                       distinguish_nan_and_inf=self.distinguish_nan_and_inf,
222:                                       nan_ok=self.nan_ok,
223:                                       param_filter=self.param_filter)
224:                     break
225:                 except AssertionError:
226:                     if j >= len(dps_list)-1:
227:                         reraise(*sys.exc_info())
228:         finally:
229:             mpmath.mp.dps, mpmath.mp.prec = old_dps, old_prec
230: 
231:     def __repr__(self):
232:         if self.is_complex:
233:             return "<MpmathData: %s (complex)>" % (self.name,)
234:         else:
235:             return "<MpmathData: %s>" % (self.name,)
236: 
237: 
238: def assert_mpmath_equal(*a, **kw):
239:     d = MpmathData(*a, **kw)
240:     d.check()
241: 
242: 
243: def nonfunctional_tooslow(func):
244:     return pytest.mark.skip(reason="    Test not yet functional (too slow), needs more work.")(func)
245: 
246: 
247: # ------------------------------------------------------------------------------
248: # Tools for dealing with mpmath quirks
249: # ------------------------------------------------------------------------------
250: 
251: def mpf2float(x):
252:     '''
253:     Convert an mpf to the nearest floating point number. Just using
254:     float directly doesn't work because of results like this:
255: 
256:     with mp.workdps(50):
257:         float(mpf("0.99999999999999999")) = 0.9999999999999999
258: 
259:     '''
260:     return float(mpmath.nstr(x, 17, min_fixed=0, max_fixed=0))
261: 
262: 
263: def mpc2complex(x):
264:     return complex(mpf2float(x.real), mpf2float(x.imag))
265: 
266: 
267: def trace_args(func):
268:     def tofloat(x):
269:         if isinstance(x, mpmath.mpc):
270:             return complex(x)
271:         else:
272:             return float(x)
273: 
274:     def wrap(*a, **kw):
275:         sys.stderr.write("%r: " % (tuple(map(tofloat, a)),))
276:         sys.stderr.flush()
277:         try:
278:             r = func(*a, **kw)
279:             sys.stderr.write("-> %r" % r)
280:         finally:
281:             sys.stderr.write("\n")
282:             sys.stderr.flush()
283:         return r
284:     return wrap
285: 
286: try:
287:     import posix
288:     import signal
289:     POSIX = ('setitimer' in dir(signal))
290: except ImportError:
291:     POSIX = False
292: 
293: 
294: class TimeoutError(Exception):
295:     pass
296: 
297: 
298: def time_limited(timeout=0.5, return_val=np.nan, use_sigalrm=True):
299:     '''
300:     Decorator for setting a timeout for pure-Python functions.
301: 
302:     If the function does not return within `timeout` seconds, the
303:     value `return_val` is returned instead.
304: 
305:     On POSIX this uses SIGALRM by default. On non-POSIX, settrace is
306:     used. Do not use this with threads: the SIGALRM implementation
307:     does probably not work well. The settrace implementation only
308:     traces the current thread.
309: 
310:     The settrace implementation slows down execution speed. Slowdown
311:     by a factor around 10 is probably typical.
312:     '''
313:     if POSIX and use_sigalrm:
314:         def sigalrm_handler(signum, frame):
315:             raise TimeoutError()
316: 
317:         def deco(func):
318:             def wrap(*a, **kw):
319:                 old_handler = signal.signal(signal.SIGALRM, sigalrm_handler)
320:                 signal.setitimer(signal.ITIMER_REAL, timeout)
321:                 try:
322:                     return func(*a, **kw)
323:                 except TimeoutError:
324:                     return return_val
325:                 finally:
326:                     signal.setitimer(signal.ITIMER_REAL, 0)
327:                     signal.signal(signal.SIGALRM, old_handler)
328:             return wrap
329:     else:
330:         def deco(func):
331:             def wrap(*a, **kw):
332:                 start_time = time.time()
333: 
334:                 def trace(frame, event, arg):
335:                     if time.time() - start_time > timeout:
336:                         raise TimeoutError()
337:                     return trace
338:                 sys.settrace(trace)
339:                 try:
340:                     return func(*a, **kw)
341:                 except TimeoutError:
342:                     sys.settrace(None)
343:                     return return_val
344:                 finally:
345:                     sys.settrace(None)
346:             return wrap
347:     return deco
348: 
349: 
350: def exception_to_nan(func):
351:     '''Decorate function to return nan if it raises an exception'''
352:     def wrap(*a, **kw):
353:         try:
354:             return func(*a, **kw)
355:         except Exception:
356:             return np.nan
357:     return wrap
358: 
359: 
360: def inf_to_nan(func):
361:     '''Decorate function to return nan if it returns inf'''
362:     def wrap(*a, **kw):
363:         v = func(*a, **kw)
364:         if not np.isfinite(v):
365:             return np.nan
366:         return v
367:     return wrap
368: 
369: 
370: def mp_assert_allclose(res, std, atol=0, rtol=1e-17):
371:     '''
372:     Compare lists of mpmath.mpf's or mpmath.mpc's directly so that it
373:     can be done to higher precision than double.
374: 
375:     '''
376:     try:
377:         len(res)
378:     except TypeError:
379:         res = list(res)
380: 
381:     n = len(std)
382:     if len(res) != n:
383:         raise AssertionError("Lengths of inputs not equal.")
384: 
385:     failures = []
386:     for k in range(n):
387:         try:
388:             assert_(mpmath.fabs(res[k] - std[k]) <= atol + rtol*mpmath.fabs(std[k]))
389:         except AssertionError:
390:             failures.append(k)
391: 
392:     ndigits = int(abs(np.log10(rtol)))
393:     msg = [""]
394:     msg.append("Bad results ({} out of {}) for the following points:"
395:                .format(len(failures), n))
396:     for k in failures:
397:         resrep = mpmath.nstr(res[k], ndigits, min_fixed=0, max_fixed=0)
398:         stdrep = mpmath.nstr(std[k], ndigits, min_fixed=0, max_fixed=0)
399:         if std[k] == 0:
400:             rdiff = "inf"
401:         else:
402:             rdiff = mpmath.fabs((res[k] - std[k])/std[k])
403:             rdiff = mpmath.nstr(rdiff, 3)
404:         msg.append("{}: {} != {} (rdiff {})".format(k, resrep, stdrep, rdiff))
405:     if failures:
406:         assert_(False, "\n".join(msg))
407: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import os' statement (line 3)
import os

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import sys' statement (line 4)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import time' statement (line 5)
import time

import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'time', time, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'import numpy' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/')
import_509630 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy')

if (type(import_509630) is not StypyTypeError):

    if (import_509630 != 'pyd_module'):
        __import__(import_509630)
        sys_modules_509631 = sys.modules[import_509630]
        import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'np', sys_modules_509631.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy', import_509630)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from numpy.testing import assert_' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/')
import_509632 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy.testing')

if (type(import_509632) is not StypyTypeError):

    if (import_509632 != 'pyd_module'):
        __import__(import_509632)
        sys_modules_509633 = sys.modules[import_509632]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy.testing', sys_modules_509633.module_type_store, module_type_store, ['assert_'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_509633, sys_modules_509633.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy.testing', None, module_type_store, ['assert_'], [assert_])

else:
    # Assigning a type to the variable 'numpy.testing' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy.testing', import_509632)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'import pytest' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/')
import_509634 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'pytest')

if (type(import_509634) is not StypyTypeError):

    if (import_509634 != 'pyd_module'):
        __import__(import_509634)
        sys_modules_509635 = sys.modules[import_509634]
        import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'pytest', sys_modules_509635.module_type_store, module_type_store)
    else:
        import pytest

        import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'pytest', pytest, module_type_store)

else:
    # Assigning a type to the variable 'pytest' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'pytest', import_509634)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from scipy._lib.six import reraise' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/')
import_509636 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy._lib.six')

if (type(import_509636) is not StypyTypeError):

    if (import_509636 != 'pyd_module'):
        __import__(import_509636)
        sys_modules_509637 = sys.modules[import_509636]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy._lib.six', sys_modules_509637.module_type_store, module_type_store, ['reraise'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_509637, sys_modules_509637.module_type_store, module_type_store)
    else:
        from scipy._lib.six import reraise

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy._lib.six', None, module_type_store, ['reraise'], [reraise])

else:
    # Assigning a type to the variable 'scipy._lib.six' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy._lib.six', import_509636)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from scipy.special._testutils import assert_func_equal' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/')
import_509638 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.special._testutils')

if (type(import_509638) is not StypyTypeError):

    if (import_509638 != 'pyd_module'):
        __import__(import_509638)
        sys_modules_509639 = sys.modules[import_509638]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.special._testutils', sys_modules_509639.module_type_store, module_type_store, ['assert_func_equal'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_509639, sys_modules_509639.module_type_store, module_type_store)
    else:
        from scipy.special._testutils import assert_func_equal

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.special._testutils', None, module_type_store, ['assert_func_equal'], [assert_func_equal])

else:
    # Assigning a type to the variable 'scipy.special._testutils' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.special._testutils', import_509638)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/')



# SSA begins for try-except statement (line 14)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 4))

# 'import mpmath' statement (line 15)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/')
import_509640 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 15, 4), 'mpmath')

if (type(import_509640) is not StypyTypeError):

    if (import_509640 != 'pyd_module'):
        __import__(import_509640)
        sys_modules_509641 = sys.modules[import_509640]
        import_module(stypy.reporting.localization.Localization(__file__, 15, 4), 'mpmath', sys_modules_509641.module_type_store, module_type_store)
    else:
        import mpmath

        import_module(stypy.reporting.localization.Localization(__file__, 15, 4), 'mpmath', mpmath, module_type_store)

else:
    # Assigning a type to the variable 'mpmath' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'mpmath', import_509640)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/')

# SSA branch for the except part of a try statement (line 14)
# SSA branch for the except 'ImportError' branch of a try statement (line 14)
module_type_store.open_ssa_branch('except')
pass
# SSA join for try-except statement (line 14)
module_type_store = module_type_store.join_ssa_context()

# Declaration of the 'Arg' class

class Arg(object, ):
    str_509642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, (-1)), 'str', "\n    Generate a set of numbers on the real axis, concentrating on\n    'interesting' regions and covering all orders of magnitude.\n    ")

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        
        # Getting the type of 'np' (line 30)
        np_509643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 26), 'np')
        # Obtaining the member 'inf' of a type (line 30)
        inf_509644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 26), np_509643, 'inf')
        # Applying the 'usub' unary operator (line 30)
        result___neg___509645 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 25), 'usub', inf_509644)
        
        # Getting the type of 'np' (line 30)
        np_509646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 36), 'np')
        # Obtaining the member 'inf' of a type (line 30)
        inf_509647 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 36), np_509646, 'inf')
        # Getting the type of 'True' (line 30)
        True_509648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 56), 'True')
        # Getting the type of 'True' (line 30)
        True_509649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 74), 'True')
        defaults = [result___neg___509645, inf_509647, True_509648, True_509649]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 30, 4, False)
        # Assigning a type to the variable 'self' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Arg.__init__', ['a', 'b', 'inclusive_a', 'inclusive_b'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['a', 'b', 'inclusive_a', 'inclusive_b'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 31):
        
        # Assigning a Name to a Attribute (line 31):
        # Getting the type of 'a' (line 31)
        a_509650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 17), 'a')
        # Getting the type of 'self' (line 31)
        self_509651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'self')
        # Setting the type of the member 'a' of a type (line 31)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 8), self_509651, 'a', a_509650)
        
        # Assigning a Name to a Attribute (line 32):
        
        # Assigning a Name to a Attribute (line 32):
        # Getting the type of 'b' (line 32)
        b_509652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 17), 'b')
        # Getting the type of 'self' (line 32)
        self_509653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'self')
        # Setting the type of the member 'b' of a type (line 32)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 8), self_509653, 'b', b_509652)
        
        # Assigning a Name to a Attribute (line 33):
        
        # Assigning a Name to a Attribute (line 33):
        # Getting the type of 'inclusive_a' (line 33)
        inclusive_a_509654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 27), 'inclusive_a')
        # Getting the type of 'self' (line 33)
        self_509655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'self')
        # Setting the type of the member 'inclusive_a' of a type (line 33)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 8), self_509655, 'inclusive_a', inclusive_a_509654)
        
        # Assigning a Name to a Attribute (line 34):
        
        # Assigning a Name to a Attribute (line 34):
        # Getting the type of 'inclusive_b' (line 34)
        inclusive_b_509656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 27), 'inclusive_b')
        # Getting the type of 'self' (line 34)
        self_509657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'self')
        # Setting the type of the member 'inclusive_b' of a type (line 34)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 8), self_509657, 'inclusive_b', inclusive_b_509656)
        
        
        # Getting the type of 'self' (line 35)
        self_509658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 11), 'self')
        # Obtaining the member 'a' of a type (line 35)
        a_509659 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 11), self_509658, 'a')
        
        # Getting the type of 'np' (line 35)
        np_509660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 22), 'np')
        # Obtaining the member 'inf' of a type (line 35)
        inf_509661 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 22), np_509660, 'inf')
        # Applying the 'usub' unary operator (line 35)
        result___neg___509662 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 21), 'usub', inf_509661)
        
        # Applying the binary operator '==' (line 35)
        result_eq_509663 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 11), '==', a_509659, result___neg___509662)
        
        # Testing the type of an if condition (line 35)
        if_condition_509664 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 35, 8), result_eq_509663)
        # Assigning a type to the variable 'if_condition_509664' (line 35)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'if_condition_509664', if_condition_509664)
        # SSA begins for if statement (line 35)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Attribute (line 36):
        
        # Assigning a BinOp to a Attribute (line 36):
        
        
        # Call to finfo(...): (line 36)
        # Processing the call arguments (line 36)
        # Getting the type of 'float' (line 36)
        float_509667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 31), 'float', False)
        # Processing the call keyword arguments (line 36)
        kwargs_509668 = {}
        # Getting the type of 'np' (line 36)
        np_509665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 22), 'np', False)
        # Obtaining the member 'finfo' of a type (line 36)
        finfo_509666 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 22), np_509665, 'finfo')
        # Calling finfo(args, kwargs) (line 36)
        finfo_call_result_509669 = invoke(stypy.reporting.localization.Localization(__file__, 36, 22), finfo_509666, *[float_509667], **kwargs_509668)
        
        # Obtaining the member 'max' of a type (line 36)
        max_509670 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 22), finfo_call_result_509669, 'max')
        # Applying the 'usub' unary operator (line 36)
        result___neg___509671 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 21), 'usub', max_509670)
        
        int_509672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 42), 'int')
        # Applying the binary operator 'div' (line 36)
        result_div_509673 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 21), 'div', result___neg___509671, int_509672)
        
        # Getting the type of 'self' (line 36)
        self_509674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 12), 'self')
        # Setting the type of the member 'a' of a type (line 36)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 12), self_509674, 'a', result_div_509673)
        # SSA join for if statement (line 35)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'self' (line 37)
        self_509675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 11), 'self')
        # Obtaining the member 'b' of a type (line 37)
        b_509676 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 11), self_509675, 'b')
        # Getting the type of 'np' (line 37)
        np_509677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 21), 'np')
        # Obtaining the member 'inf' of a type (line 37)
        inf_509678 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 21), np_509677, 'inf')
        # Applying the binary operator '==' (line 37)
        result_eq_509679 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 11), '==', b_509676, inf_509678)
        
        # Testing the type of an if condition (line 37)
        if_condition_509680 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 37, 8), result_eq_509679)
        # Assigning a type to the variable 'if_condition_509680' (line 37)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'if_condition_509680', if_condition_509680)
        # SSA begins for if statement (line 37)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Attribute (line 38):
        
        # Assigning a BinOp to a Attribute (line 38):
        
        # Call to finfo(...): (line 38)
        # Processing the call arguments (line 38)
        # Getting the type of 'float' (line 38)
        float_509683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 30), 'float', False)
        # Processing the call keyword arguments (line 38)
        kwargs_509684 = {}
        # Getting the type of 'np' (line 38)
        np_509681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 21), 'np', False)
        # Obtaining the member 'finfo' of a type (line 38)
        finfo_509682 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 21), np_509681, 'finfo')
        # Calling finfo(args, kwargs) (line 38)
        finfo_call_result_509685 = invoke(stypy.reporting.localization.Localization(__file__, 38, 21), finfo_509682, *[float_509683], **kwargs_509684)
        
        # Obtaining the member 'max' of a type (line 38)
        max_509686 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 21), finfo_call_result_509685, 'max')
        int_509687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 41), 'int')
        # Applying the binary operator 'div' (line 38)
        result_div_509688 = python_operator(stypy.reporting.localization.Localization(__file__, 38, 21), 'div', max_509686, int_509687)
        
        # Getting the type of 'self' (line 38)
        self_509689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 12), 'self')
        # Setting the type of the member 'b' of a type (line 38)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 12), self_509689, 'b', result_div_509688)
        # SSA join for if statement (line 37)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def values(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'values'
        module_type_store = module_type_store.open_function_context('values', 40, 4, False)
        # Assigning a type to the variable 'self' (line 41)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Arg.values.__dict__.__setitem__('stypy_localization', localization)
        Arg.values.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Arg.values.__dict__.__setitem__('stypy_type_store', module_type_store)
        Arg.values.__dict__.__setitem__('stypy_function_name', 'Arg.values')
        Arg.values.__dict__.__setitem__('stypy_param_names_list', ['n'])
        Arg.values.__dict__.__setitem__('stypy_varargs_param_name', None)
        Arg.values.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Arg.values.__dict__.__setitem__('stypy_call_defaults', defaults)
        Arg.values.__dict__.__setitem__('stypy_call_varargs', varargs)
        Arg.values.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Arg.values.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Arg.values', ['n'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'values', localization, ['n'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'values(...)' code ##################

        str_509690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 8), 'str', 'Return an array containing approximatively `n` numbers.')
        
        # Assigning a Call to a Name (line 42):
        
        # Assigning a Call to a Name (line 42):
        
        # Call to max(...): (line 42)
        # Processing the call arguments (line 42)
        int_509692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 17), 'int')
        
        # Call to int(...): (line 42)
        # Processing the call arguments (line 42)
        float_509694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 24), 'float')
        # Getting the type of 'n' (line 42)
        n_509695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 28), 'n', False)
        # Applying the binary operator '*' (line 42)
        result_mul_509696 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 24), '*', float_509694, n_509695)
        
        # Processing the call keyword arguments (line 42)
        kwargs_509697 = {}
        # Getting the type of 'int' (line 42)
        int_509693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 20), 'int', False)
        # Calling int(args, kwargs) (line 42)
        int_call_result_509698 = invoke(stypy.reporting.localization.Localization(__file__, 42, 20), int_509693, *[result_mul_509696], **kwargs_509697)
        
        # Processing the call keyword arguments (line 42)
        kwargs_509699 = {}
        # Getting the type of 'max' (line 42)
        max_509691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 13), 'max', False)
        # Calling max(args, kwargs) (line 42)
        max_call_result_509700 = invoke(stypy.reporting.localization.Localization(__file__, 42, 13), max_509691, *[int_509692, int_call_result_509698], **kwargs_509699)
        
        # Assigning a type to the variable 'n1' (line 42)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'n1', max_call_result_509700)
        
        # Assigning a Call to a Name (line 43):
        
        # Assigning a Call to a Name (line 43):
        
        # Call to max(...): (line 43)
        # Processing the call arguments (line 43)
        int_509702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 17), 'int')
        
        # Call to int(...): (line 43)
        # Processing the call arguments (line 43)
        float_509704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 24), 'float')
        # Getting the type of 'n' (line 43)
        n_509705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 28), 'n', False)
        # Applying the binary operator '*' (line 43)
        result_mul_509706 = python_operator(stypy.reporting.localization.Localization(__file__, 43, 24), '*', float_509704, n_509705)
        
        # Processing the call keyword arguments (line 43)
        kwargs_509707 = {}
        # Getting the type of 'int' (line 43)
        int_509703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 20), 'int', False)
        # Calling int(args, kwargs) (line 43)
        int_call_result_509708 = invoke(stypy.reporting.localization.Localization(__file__, 43, 20), int_509703, *[result_mul_509706], **kwargs_509707)
        
        # Processing the call keyword arguments (line 43)
        kwargs_509709 = {}
        # Getting the type of 'max' (line 43)
        max_509701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 13), 'max', False)
        # Calling max(args, kwargs) (line 43)
        max_call_result_509710 = invoke(stypy.reporting.localization.Localization(__file__, 43, 13), max_509701, *[int_509702, int_call_result_509708], **kwargs_509709)
        
        # Assigning a type to the variable 'n2' (line 43)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'n2', max_call_result_509710)
        
        # Assigning a Call to a Name (line 44):
        
        # Assigning a Call to a Name (line 44):
        
        # Call to max(...): (line 44)
        # Processing the call arguments (line 44)
        int_509712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 17), 'int')
        # Getting the type of 'n' (line 44)
        n_509713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 20), 'n', False)
        # Getting the type of 'n1' (line 44)
        n1_509714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 24), 'n1', False)
        # Applying the binary operator '-' (line 44)
        result_sub_509715 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 20), '-', n_509713, n1_509714)
        
        # Getting the type of 'n2' (line 44)
        n2_509716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 29), 'n2', False)
        # Applying the binary operator '-' (line 44)
        result_sub_509717 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 27), '-', result_sub_509715, n2_509716)
        
        # Processing the call keyword arguments (line 44)
        kwargs_509718 = {}
        # Getting the type of 'max' (line 44)
        max_509711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 13), 'max', False)
        # Calling max(args, kwargs) (line 44)
        max_call_result_509719 = invoke(stypy.reporting.localization.Localization(__file__, 44, 13), max_509711, *[int_509712, result_sub_509717], **kwargs_509718)
        
        # Assigning a type to the variable 'n3' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'n3', max_call_result_509719)
        
        # Assigning a Call to a Name (line 46):
        
        # Assigning a Call to a Name (line 46):
        
        # Call to linspace(...): (line 46)
        # Processing the call arguments (line 46)
        int_509722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 25), 'int')
        int_509723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 29), 'int')
        # Getting the type of 'n1' (line 46)
        n1_509724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 32), 'n1', False)
        # Processing the call keyword arguments (line 46)
        kwargs_509725 = {}
        # Getting the type of 'np' (line 46)
        np_509720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 13), 'np', False)
        # Obtaining the member 'linspace' of a type (line 46)
        linspace_509721 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 13), np_509720, 'linspace')
        # Calling linspace(args, kwargs) (line 46)
        linspace_call_result_509726 = invoke(stypy.reporting.localization.Localization(__file__, 46, 13), linspace_509721, *[int_509722, int_509723, n1_509724], **kwargs_509725)
        
        # Assigning a type to the variable 'v1' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'v1', linspace_call_result_509726)
        
        # Assigning a Subscript to a Name (line 47):
        
        # Assigning a Subscript to a Name (line 47):
        
        # Obtaining the type of the subscript
        
        # Obtaining an instance of the builtin type 'tuple' (line 47)
        tuple_509727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 47)
        # Adding element type (line 47)
        
        # Call to linspace(...): (line 47)
        # Processing the call arguments (line 47)
        int_509730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 31), 'int')
        int_509731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 36), 'int')
        
        # Call to max(...): (line 47)
        # Processing the call arguments (line 47)
        int_509733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 44), 'int')
        # Getting the type of 'n2' (line 47)
        n2_509734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 47), 'n2', False)
        int_509735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 50), 'int')
        # Applying the binary operator '-' (line 47)
        result_sub_509736 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 47), '-', n2_509734, int_509735)
        
        # Processing the call keyword arguments (line 47)
        kwargs_509737 = {}
        # Getting the type of 'max' (line 47)
        max_509732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 40), 'max', False)
        # Calling max(args, kwargs) (line 47)
        max_call_result_509738 = invoke(stypy.reporting.localization.Localization(__file__, 47, 40), max_509732, *[int_509733, result_sub_509736], **kwargs_509737)
        
        # Processing the call keyword arguments (line 47)
        kwargs_509739 = {}
        # Getting the type of 'np' (line 47)
        np_509728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 19), 'np', False)
        # Obtaining the member 'linspace' of a type (line 47)
        linspace_509729 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 19), np_509728, 'linspace')
        # Calling linspace(args, kwargs) (line 47)
        linspace_call_result_509740 = invoke(stypy.reporting.localization.Localization(__file__, 47, 19), linspace_509729, *[int_509730, int_509731, max_call_result_509738], **kwargs_509739)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 19), tuple_509727, linspace_call_result_509740)
        # Adding element type (line 47)
        int_509741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 19), tuple_509727, int_509741)
        # Adding element type (line 47)
        float_509742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 19), tuple_509727, float_509742)
        # Adding element type (line 47)
        float_509743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 29), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 19), tuple_509727, float_509743)
        # Adding element type (line 47)
        int_509744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 19), tuple_509727, int_509744)
        
        # Getting the type of 'np' (line 47)
        np_509745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 13), 'np')
        # Obtaining the member 'r_' of a type (line 47)
        r__509746 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 13), np_509745, 'r_')
        # Obtaining the member '__getitem__' of a type (line 47)
        getitem___509747 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 13), r__509746, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 47)
        subscript_call_result_509748 = invoke(stypy.reporting.localization.Localization(__file__, 47, 13), getitem___509747, tuple_509727)
        
        # Assigning a type to the variable 'v2' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'v2', subscript_call_result_509748)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'self' (line 49)
        self_509749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 11), 'self')
        # Obtaining the member 'a' of a type (line 49)
        a_509750 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 11), self_509749, 'a')
        int_509751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 21), 'int')
        # Applying the binary operator '>=' (line 49)
        result_ge_509752 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 11), '>=', a_509750, int_509751)
        
        
        # Getting the type of 'self' (line 49)
        self_509753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 27), 'self')
        # Obtaining the member 'b' of a type (line 49)
        b_509754 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 27), self_509753, 'b')
        int_509755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 36), 'int')
        # Applying the binary operator '>' (line 49)
        result_gt_509756 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 27), '>', b_509754, int_509755)
        
        # Applying the binary operator 'and' (line 49)
        result_and_keyword_509757 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 11), 'and', result_ge_509752, result_gt_509756)
        
        # Testing the type of an if condition (line 49)
        if_condition_509758 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 49, 8), result_and_keyword_509757)
        # Assigning a type to the variable 'if_condition_509758' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'if_condition_509758', if_condition_509758)
        # SSA begins for if statement (line 49)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Name (line 50):
        
        # Assigning a Subscript to a Name (line 50):
        
        # Obtaining the type of the subscript
        
        # Obtaining an instance of the builtin type 'tuple' (line 51)
        tuple_509759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 16), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 51)
        # Adding element type (line 51)
        
        # Call to logspace(...): (line 51)
        # Processing the call arguments (line 51)
        int_509762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 28), 'int')
        int_509763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 33), 'int')
        int_509764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 37), 'int')
        # Getting the type of 'n3' (line 51)
        n3_509765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 41), 'n3', False)
        int_509766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 45), 'int')
        # Applying the binary operator '//' (line 51)
        result_floordiv_509767 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 41), '//', n3_509765, int_509766)
        
        # Applying the binary operator '+' (line 51)
        result_add_509768 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 37), '+', int_509764, result_floordiv_509767)
        
        # Processing the call keyword arguments (line 51)
        kwargs_509769 = {}
        # Getting the type of 'np' (line 51)
        np_509760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 16), 'np', False)
        # Obtaining the member 'logspace' of a type (line 51)
        logspace_509761 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 16), np_509760, 'logspace')
        # Calling logspace(args, kwargs) (line 51)
        logspace_call_result_509770 = invoke(stypy.reporting.localization.Localization(__file__, 51, 16), logspace_509761, *[int_509762, int_509763, result_add_509768], **kwargs_509769)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 16), tuple_509759, logspace_call_result_509770)
        # Adding element type (line 51)
        
        # Call to logspace(...): (line 52)
        # Processing the call arguments (line 52)
        int_509773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 28), 'int')
        
        # Call to log10(...): (line 52)
        # Processing the call arguments (line 52)
        # Getting the type of 'self' (line 52)
        self_509776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 40), 'self', False)
        # Obtaining the member 'b' of a type (line 52)
        b_509777 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 40), self_509776, 'b')
        # Processing the call keyword arguments (line 52)
        kwargs_509778 = {}
        # Getting the type of 'np' (line 52)
        np_509774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 31), 'np', False)
        # Obtaining the member 'log10' of a type (line 52)
        log10_509775 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 31), np_509774, 'log10')
        # Calling log10(args, kwargs) (line 52)
        log10_call_result_509779 = invoke(stypy.reporting.localization.Localization(__file__, 52, 31), log10_509775, *[b_509777], **kwargs_509778)
        
        int_509780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 49), 'int')
        # Getting the type of 'n3' (line 52)
        n3_509781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 53), 'n3', False)
        int_509782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 57), 'int')
        # Applying the binary operator '//' (line 52)
        result_floordiv_509783 = python_operator(stypy.reporting.localization.Localization(__file__, 52, 53), '//', n3_509781, int_509782)
        
        # Applying the binary operator '+' (line 52)
        result_add_509784 = python_operator(stypy.reporting.localization.Localization(__file__, 52, 49), '+', int_509780, result_floordiv_509783)
        
        # Processing the call keyword arguments (line 52)
        kwargs_509785 = {}
        # Getting the type of 'np' (line 52)
        np_509771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 16), 'np', False)
        # Obtaining the member 'logspace' of a type (line 52)
        logspace_509772 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 16), np_509771, 'logspace')
        # Calling logspace(args, kwargs) (line 52)
        logspace_call_result_509786 = invoke(stypy.reporting.localization.Localization(__file__, 52, 16), logspace_509772, *[int_509773, log10_call_result_509779, result_add_509784], **kwargs_509785)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 16), tuple_509759, logspace_call_result_509786)
        
        # Getting the type of 'np' (line 50)
        np_509787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 17), 'np')
        # Obtaining the member 'r_' of a type (line 50)
        r__509788 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 17), np_509787, 'r_')
        # Obtaining the member '__getitem__' of a type (line 50)
        getitem___509789 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 17), r__509788, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 50)
        subscript_call_result_509790 = invoke(stypy.reporting.localization.Localization(__file__, 50, 17), getitem___509789, tuple_509759)
        
        # Assigning a type to the variable 'v3' (line 50)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 12), 'v3', subscript_call_result_509790)
        
        # Assigning a Call to a Name (line 54):
        
        # Assigning a Call to a Name (line 54):
        
        # Call to logspace(...): (line 54)
        # Processing the call arguments (line 54)
        int_509793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 29), 'int')
        int_509794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 32), 'int')
        int_509795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 35), 'int')
        # Getting the type of 'n3' (line 54)
        n3_509796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 39), 'n3', False)
        int_509797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 43), 'int')
        # Applying the binary operator '//' (line 54)
        result_floordiv_509798 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 39), '//', n3_509796, int_509797)
        
        # Applying the binary operator '+' (line 54)
        result_add_509799 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 35), '+', int_509795, result_floordiv_509798)
        
        # Processing the call keyword arguments (line 54)
        kwargs_509800 = {}
        # Getting the type of 'np' (line 54)
        np_509791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 17), 'np', False)
        # Obtaining the member 'logspace' of a type (line 54)
        logspace_509792 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 17), np_509791, 'logspace')
        # Calling logspace(args, kwargs) (line 54)
        logspace_call_result_509801 = invoke(stypy.reporting.localization.Localization(__file__, 54, 17), logspace_509792, *[int_509793, int_509794, result_add_509799], **kwargs_509800)
        
        # Assigning a type to the variable 'v4' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 12), 'v4', logspace_call_result_509801)
        # SSA branch for the else part of an if statement (line 49)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'self' (line 55)
        self_509802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 13), 'self')
        # Obtaining the member 'a' of a type (line 55)
        a_509803 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 13), self_509802, 'a')
        int_509804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 22), 'int')
        # Applying the binary operator '<' (line 55)
        result_lt_509805 = python_operator(stypy.reporting.localization.Localization(__file__, 55, 13), '<', a_509803, int_509804)
        # Getting the type of 'self' (line 55)
        self_509806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 26), 'self')
        # Obtaining the member 'b' of a type (line 55)
        b_509807 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 26), self_509806, 'b')
        # Applying the binary operator '<' (line 55)
        result_lt_509808 = python_operator(stypy.reporting.localization.Localization(__file__, 55, 13), '<', int_509804, b_509807)
        # Applying the binary operator '&' (line 55)
        result_and__509809 = python_operator(stypy.reporting.localization.Localization(__file__, 55, 13), '&', result_lt_509805, result_lt_509808)
        
        # Testing the type of an if condition (line 55)
        if_condition_509810 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 55, 13), result_and__509809)
        # Assigning a type to the variable 'if_condition_509810' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 13), 'if_condition_509810', if_condition_509810)
        # SSA begins for if statement (line 55)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Name (line 56):
        
        # Assigning a Subscript to a Name (line 56):
        
        # Obtaining the type of the subscript
        
        # Obtaining an instance of the builtin type 'tuple' (line 57)
        tuple_509811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 16), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 57)
        # Adding element type (line 57)
        
        # Call to logspace(...): (line 57)
        # Processing the call arguments (line 57)
        int_509814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 28), 'int')
        int_509815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 33), 'int')
        int_509816 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 37), 'int')
        # Getting the type of 'n3' (line 57)
        n3_509817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 41), 'n3', False)
        int_509818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 45), 'int')
        # Applying the binary operator '//' (line 57)
        result_floordiv_509819 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 41), '//', n3_509817, int_509818)
        
        # Applying the binary operator '+' (line 57)
        result_add_509820 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 37), '+', int_509816, result_floordiv_509819)
        
        # Processing the call keyword arguments (line 57)
        kwargs_509821 = {}
        # Getting the type of 'np' (line 57)
        np_509812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 16), 'np', False)
        # Obtaining the member 'logspace' of a type (line 57)
        logspace_509813 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 16), np_509812, 'logspace')
        # Calling logspace(args, kwargs) (line 57)
        logspace_call_result_509822 = invoke(stypy.reporting.localization.Localization(__file__, 57, 16), logspace_509813, *[int_509814, int_509815, result_add_509820], **kwargs_509821)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 16), tuple_509811, logspace_call_result_509822)
        # Adding element type (line 57)
        
        # Call to logspace(...): (line 58)
        # Processing the call arguments (line 58)
        int_509825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 28), 'int')
        
        # Call to log10(...): (line 58)
        # Processing the call arguments (line 58)
        # Getting the type of 'self' (line 58)
        self_509828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 40), 'self', False)
        # Obtaining the member 'b' of a type (line 58)
        b_509829 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 40), self_509828, 'b')
        # Processing the call keyword arguments (line 58)
        kwargs_509830 = {}
        # Getting the type of 'np' (line 58)
        np_509826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 31), 'np', False)
        # Obtaining the member 'log10' of a type (line 58)
        log10_509827 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 31), np_509826, 'log10')
        # Calling log10(args, kwargs) (line 58)
        log10_call_result_509831 = invoke(stypy.reporting.localization.Localization(__file__, 58, 31), log10_509827, *[b_509829], **kwargs_509830)
        
        int_509832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 49), 'int')
        # Getting the type of 'n3' (line 58)
        n3_509833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 53), 'n3', False)
        int_509834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 57), 'int')
        # Applying the binary operator '//' (line 58)
        result_floordiv_509835 = python_operator(stypy.reporting.localization.Localization(__file__, 58, 53), '//', n3_509833, int_509834)
        
        # Applying the binary operator '+' (line 58)
        result_add_509836 = python_operator(stypy.reporting.localization.Localization(__file__, 58, 49), '+', int_509832, result_floordiv_509835)
        
        # Processing the call keyword arguments (line 58)
        kwargs_509837 = {}
        # Getting the type of 'np' (line 58)
        np_509823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 16), 'np', False)
        # Obtaining the member 'logspace' of a type (line 58)
        logspace_509824 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 16), np_509823, 'logspace')
        # Calling logspace(args, kwargs) (line 58)
        logspace_call_result_509838 = invoke(stypy.reporting.localization.Localization(__file__, 58, 16), logspace_509824, *[int_509825, log10_call_result_509831, result_add_509836], **kwargs_509837)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 16), tuple_509811, logspace_call_result_509838)
        # Adding element type (line 57)
        
        
        # Call to logspace(...): (line 59)
        # Processing the call arguments (line 59)
        int_509841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 29), 'int')
        int_509842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 34), 'int')
        int_509843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 38), 'int')
        # Getting the type of 'n3' (line 59)
        n3_509844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 42), 'n3', False)
        int_509845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 46), 'int')
        # Applying the binary operator '//' (line 59)
        result_floordiv_509846 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 42), '//', n3_509844, int_509845)
        
        # Applying the binary operator '+' (line 59)
        result_add_509847 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 38), '+', int_509843, result_floordiv_509846)
        
        # Processing the call keyword arguments (line 59)
        kwargs_509848 = {}
        # Getting the type of 'np' (line 59)
        np_509839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 17), 'np', False)
        # Obtaining the member 'logspace' of a type (line 59)
        logspace_509840 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 17), np_509839, 'logspace')
        # Calling logspace(args, kwargs) (line 59)
        logspace_call_result_509849 = invoke(stypy.reporting.localization.Localization(__file__, 59, 17), logspace_509840, *[int_509841, int_509842, result_add_509847], **kwargs_509848)
        
        # Applying the 'usub' unary operator (line 59)
        result___neg___509850 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 16), 'usub', logspace_call_result_509849)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 16), tuple_509811, result___neg___509850)
        # Adding element type (line 57)
        
        
        # Call to logspace(...): (line 60)
        # Processing the call arguments (line 60)
        int_509853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 29), 'int')
        
        # Call to log10(...): (line 60)
        # Processing the call arguments (line 60)
        
        # Getting the type of 'self' (line 60)
        self_509856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 42), 'self', False)
        # Obtaining the member 'a' of a type (line 60)
        a_509857 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 42), self_509856, 'a')
        # Applying the 'usub' unary operator (line 60)
        result___neg___509858 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 41), 'usub', a_509857)
        
        # Processing the call keyword arguments (line 60)
        kwargs_509859 = {}
        # Getting the type of 'np' (line 60)
        np_509854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 32), 'np', False)
        # Obtaining the member 'log10' of a type (line 60)
        log10_509855 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 32), np_509854, 'log10')
        # Calling log10(args, kwargs) (line 60)
        log10_call_result_509860 = invoke(stypy.reporting.localization.Localization(__file__, 60, 32), log10_509855, *[result___neg___509858], **kwargs_509859)
        
        int_509861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 51), 'int')
        # Getting the type of 'n3' (line 60)
        n3_509862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 55), 'n3', False)
        int_509863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 59), 'int')
        # Applying the binary operator '//' (line 60)
        result_floordiv_509864 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 55), '//', n3_509862, int_509863)
        
        # Applying the binary operator '+' (line 60)
        result_add_509865 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 51), '+', int_509861, result_floordiv_509864)
        
        # Processing the call keyword arguments (line 60)
        kwargs_509866 = {}
        # Getting the type of 'np' (line 60)
        np_509851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 17), 'np', False)
        # Obtaining the member 'logspace' of a type (line 60)
        logspace_509852 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 17), np_509851, 'logspace')
        # Calling logspace(args, kwargs) (line 60)
        logspace_call_result_509867 = invoke(stypy.reporting.localization.Localization(__file__, 60, 17), logspace_509852, *[int_509853, log10_call_result_509860, result_add_509865], **kwargs_509866)
        
        # Applying the 'usub' unary operator (line 60)
        result___neg___509868 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 16), 'usub', logspace_call_result_509867)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 16), tuple_509811, result___neg___509868)
        
        # Getting the type of 'np' (line 56)
        np_509869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 17), 'np')
        # Obtaining the member 'r_' of a type (line 56)
        r__509870 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 17), np_509869, 'r_')
        # Obtaining the member '__getitem__' of a type (line 56)
        getitem___509871 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 17), r__509870, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 56)
        subscript_call_result_509872 = invoke(stypy.reporting.localization.Localization(__file__, 56, 17), getitem___509871, tuple_509811)
        
        # Assigning a type to the variable 'v3' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 12), 'v3', subscript_call_result_509872)
        
        # Assigning a Subscript to a Name (line 62):
        
        # Assigning a Subscript to a Name (line 62):
        
        # Obtaining the type of the subscript
        
        # Obtaining an instance of the builtin type 'tuple' (line 63)
        tuple_509873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 16), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 63)
        # Adding element type (line 63)
        
        # Call to logspace(...): (line 63)
        # Processing the call arguments (line 63)
        int_509876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 28), 'int')
        int_509877 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 31), 'int')
        int_509878 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 34), 'int')
        # Getting the type of 'n3' (line 63)
        n3_509879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 38), 'n3', False)
        int_509880 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 42), 'int')
        # Applying the binary operator '//' (line 63)
        result_floordiv_509881 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 38), '//', n3_509879, int_509880)
        
        # Applying the binary operator '+' (line 63)
        result_add_509882 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 34), '+', int_509878, result_floordiv_509881)
        
        # Processing the call keyword arguments (line 63)
        kwargs_509883 = {}
        # Getting the type of 'np' (line 63)
        np_509874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 16), 'np', False)
        # Obtaining the member 'logspace' of a type (line 63)
        logspace_509875 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 16), np_509874, 'logspace')
        # Calling logspace(args, kwargs) (line 63)
        logspace_call_result_509884 = invoke(stypy.reporting.localization.Localization(__file__, 63, 16), logspace_509875, *[int_509876, int_509877, result_add_509882], **kwargs_509883)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 16), tuple_509873, logspace_call_result_509884)
        # Adding element type (line 63)
        
        
        # Call to logspace(...): (line 64)
        # Processing the call arguments (line 64)
        int_509887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 29), 'int')
        int_509888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 32), 'int')
        int_509889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 35), 'int')
        # Getting the type of 'n3' (line 64)
        n3_509890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 39), 'n3', False)
        int_509891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 43), 'int')
        # Applying the binary operator '//' (line 64)
        result_floordiv_509892 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 39), '//', n3_509890, int_509891)
        
        # Applying the binary operator '+' (line 64)
        result_add_509893 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 35), '+', int_509889, result_floordiv_509892)
        
        # Processing the call keyword arguments (line 64)
        kwargs_509894 = {}
        # Getting the type of 'np' (line 64)
        np_509885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 17), 'np', False)
        # Obtaining the member 'logspace' of a type (line 64)
        logspace_509886 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 17), np_509885, 'logspace')
        # Calling logspace(args, kwargs) (line 64)
        logspace_call_result_509895 = invoke(stypy.reporting.localization.Localization(__file__, 64, 17), logspace_509886, *[int_509887, int_509888, result_add_509893], **kwargs_509894)
        
        # Applying the 'usub' unary operator (line 64)
        result___neg___509896 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 16), 'usub', logspace_call_result_509895)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 16), tuple_509873, result___neg___509896)
        
        # Getting the type of 'np' (line 62)
        np_509897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 17), 'np')
        # Obtaining the member 'r_' of a type (line 62)
        r__509898 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 17), np_509897, 'r_')
        # Obtaining the member '__getitem__' of a type (line 62)
        getitem___509899 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 17), r__509898, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 62)
        subscript_call_result_509900 = invoke(stypy.reporting.localization.Localization(__file__, 62, 17), getitem___509899, tuple_509873)
        
        # Assigning a type to the variable 'v4' (line 62)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 12), 'v4', subscript_call_result_509900)
        # SSA branch for the else part of an if statement (line 55)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'self' (line 66)
        self_509901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 13), 'self')
        # Obtaining the member 'b' of a type (line 66)
        b_509902 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 13), self_509901, 'b')
        int_509903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 22), 'int')
        # Applying the binary operator '<' (line 66)
        result_lt_509904 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 13), '<', b_509902, int_509903)
        
        # Testing the type of an if condition (line 66)
        if_condition_509905 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 66, 13), result_lt_509904)
        # Assigning a type to the variable 'if_condition_509905' (line 66)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 13), 'if_condition_509905', if_condition_509905)
        # SSA begins for if statement (line 66)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Name (line 67):
        
        # Assigning a Subscript to a Name (line 67):
        
        # Obtaining the type of the subscript
        
        # Obtaining an instance of the builtin type 'tuple' (line 68)
        tuple_509906 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 16), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 68)
        # Adding element type (line 68)
        
        
        # Call to logspace(...): (line 68)
        # Processing the call arguments (line 68)
        int_509909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 29), 'int')
        int_509910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 34), 'int')
        int_509911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 38), 'int')
        # Getting the type of 'n3' (line 68)
        n3_509912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 42), 'n3', False)
        int_509913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 46), 'int')
        # Applying the binary operator '//' (line 68)
        result_floordiv_509914 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 42), '//', n3_509912, int_509913)
        
        # Applying the binary operator '+' (line 68)
        result_add_509915 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 38), '+', int_509911, result_floordiv_509914)
        
        # Processing the call keyword arguments (line 68)
        kwargs_509916 = {}
        # Getting the type of 'np' (line 68)
        np_509907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 17), 'np', False)
        # Obtaining the member 'logspace' of a type (line 68)
        logspace_509908 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 17), np_509907, 'logspace')
        # Calling logspace(args, kwargs) (line 68)
        logspace_call_result_509917 = invoke(stypy.reporting.localization.Localization(__file__, 68, 17), logspace_509908, *[int_509909, int_509910, result_add_509915], **kwargs_509916)
        
        # Applying the 'usub' unary operator (line 68)
        result___neg___509918 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 16), 'usub', logspace_call_result_509917)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 16), tuple_509906, result___neg___509918)
        # Adding element type (line 68)
        
        
        # Call to logspace(...): (line 69)
        # Processing the call arguments (line 69)
        int_509921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 29), 'int')
        
        # Call to log10(...): (line 69)
        # Processing the call arguments (line 69)
        
        # Getting the type of 'self' (line 69)
        self_509924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 42), 'self', False)
        # Obtaining the member 'b' of a type (line 69)
        b_509925 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 42), self_509924, 'b')
        # Applying the 'usub' unary operator (line 69)
        result___neg___509926 = python_operator(stypy.reporting.localization.Localization(__file__, 69, 41), 'usub', b_509925)
        
        # Processing the call keyword arguments (line 69)
        kwargs_509927 = {}
        # Getting the type of 'np' (line 69)
        np_509922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 32), 'np', False)
        # Obtaining the member 'log10' of a type (line 69)
        log10_509923 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 32), np_509922, 'log10')
        # Calling log10(args, kwargs) (line 69)
        log10_call_result_509928 = invoke(stypy.reporting.localization.Localization(__file__, 69, 32), log10_509923, *[result___neg___509926], **kwargs_509927)
        
        int_509929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 51), 'int')
        # Getting the type of 'n3' (line 69)
        n3_509930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 55), 'n3', False)
        int_509931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 59), 'int')
        # Applying the binary operator '//' (line 69)
        result_floordiv_509932 = python_operator(stypy.reporting.localization.Localization(__file__, 69, 55), '//', n3_509930, int_509931)
        
        # Applying the binary operator '+' (line 69)
        result_add_509933 = python_operator(stypy.reporting.localization.Localization(__file__, 69, 51), '+', int_509929, result_floordiv_509932)
        
        # Processing the call keyword arguments (line 69)
        kwargs_509934 = {}
        # Getting the type of 'np' (line 69)
        np_509919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 17), 'np', False)
        # Obtaining the member 'logspace' of a type (line 69)
        logspace_509920 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 17), np_509919, 'logspace')
        # Calling logspace(args, kwargs) (line 69)
        logspace_call_result_509935 = invoke(stypy.reporting.localization.Localization(__file__, 69, 17), logspace_509920, *[int_509921, log10_call_result_509928, result_add_509933], **kwargs_509934)
        
        # Applying the 'usub' unary operator (line 69)
        result___neg___509936 = python_operator(stypy.reporting.localization.Localization(__file__, 69, 16), 'usub', logspace_call_result_509935)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 16), tuple_509906, result___neg___509936)
        
        # Getting the type of 'np' (line 67)
        np_509937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 17), 'np')
        # Obtaining the member 'r_' of a type (line 67)
        r__509938 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 17), np_509937, 'r_')
        # Obtaining the member '__getitem__' of a type (line 67)
        getitem___509939 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 17), r__509938, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 67)
        subscript_call_result_509940 = invoke(stypy.reporting.localization.Localization(__file__, 67, 17), getitem___509939, tuple_509906)
        
        # Assigning a type to the variable 'v3' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 12), 'v3', subscript_call_result_509940)
        
        # Assigning a UnaryOp to a Name (line 71):
        
        # Assigning a UnaryOp to a Name (line 71):
        
        
        # Call to logspace(...): (line 71)
        # Processing the call arguments (line 71)
        int_509943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 30), 'int')
        int_509944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 33), 'int')
        int_509945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 36), 'int')
        # Getting the type of 'n3' (line 71)
        n3_509946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 40), 'n3', False)
        int_509947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 44), 'int')
        # Applying the binary operator '//' (line 71)
        result_floordiv_509948 = python_operator(stypy.reporting.localization.Localization(__file__, 71, 40), '//', n3_509946, int_509947)
        
        # Applying the binary operator '+' (line 71)
        result_add_509949 = python_operator(stypy.reporting.localization.Localization(__file__, 71, 36), '+', int_509945, result_floordiv_509948)
        
        # Processing the call keyword arguments (line 71)
        kwargs_509950 = {}
        # Getting the type of 'np' (line 71)
        np_509941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 18), 'np', False)
        # Obtaining the member 'logspace' of a type (line 71)
        logspace_509942 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 18), np_509941, 'logspace')
        # Calling logspace(args, kwargs) (line 71)
        logspace_call_result_509951 = invoke(stypy.reporting.localization.Localization(__file__, 71, 18), logspace_509942, *[int_509943, int_509944, result_add_509949], **kwargs_509950)
        
        # Applying the 'usub' unary operator (line 71)
        result___neg___509952 = python_operator(stypy.reporting.localization.Localization(__file__, 71, 17), 'usub', logspace_call_result_509951)
        
        # Assigning a type to the variable 'v4' (line 71)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 12), 'v4', result___neg___509952)
        # SSA branch for the else part of an if statement (line 66)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a List to a Name (line 73):
        
        # Assigning a List to a Name (line 73):
        
        # Obtaining an instance of the builtin type 'list' (line 73)
        list_509953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 73)
        
        # Assigning a type to the variable 'v3' (line 73)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 12), 'v3', list_509953)
        
        # Assigning a List to a Name (line 74):
        
        # Assigning a List to a Name (line 74):
        
        # Obtaining an instance of the builtin type 'list' (line 74)
        list_509954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 74)
        
        # Assigning a type to the variable 'v4' (line 74)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 12), 'v4', list_509954)
        # SSA join for if statement (line 66)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 55)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 49)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Subscript to a Name (line 75):
        
        # Assigning a Subscript to a Name (line 75):
        
        # Obtaining the type of the subscript
        
        # Obtaining an instance of the builtin type 'tuple' (line 75)
        tuple_509955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 18), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 75)
        # Adding element type (line 75)
        # Getting the type of 'v1' (line 75)
        v1_509956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 18), 'v1')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 18), tuple_509955, v1_509956)
        # Adding element type (line 75)
        # Getting the type of 'v2' (line 75)
        v2_509957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 22), 'v2')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 18), tuple_509955, v2_509957)
        # Adding element type (line 75)
        # Getting the type of 'v3' (line 75)
        v3_509958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 26), 'v3')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 18), tuple_509955, v3_509958)
        # Adding element type (line 75)
        # Getting the type of 'v4' (line 75)
        v4_509959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 30), 'v4')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 18), tuple_509955, v4_509959)
        # Adding element type (line 75)
        int_509960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 18), tuple_509955, int_509960)
        
        # Getting the type of 'np' (line 75)
        np_509961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 12), 'np')
        # Obtaining the member 'r_' of a type (line 75)
        r__509962 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 12), np_509961, 'r_')
        # Obtaining the member '__getitem__' of a type (line 75)
        getitem___509963 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 12), r__509962, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 75)
        subscript_call_result_509964 = invoke(stypy.reporting.localization.Localization(__file__, 75, 12), getitem___509963, tuple_509955)
        
        # Assigning a type to the variable 'v' (line 75)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'v', subscript_call_result_509964)
        
        # Getting the type of 'self' (line 76)
        self_509965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 11), 'self')
        # Obtaining the member 'inclusive_a' of a type (line 76)
        inclusive_a_509966 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 11), self_509965, 'inclusive_a')
        # Testing the type of an if condition (line 76)
        if_condition_509967 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 76, 8), inclusive_a_509966)
        # Assigning a type to the variable 'if_condition_509967' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'if_condition_509967', if_condition_509967)
        # SSA begins for if statement (line 76)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Name (line 77):
        
        # Assigning a Subscript to a Name (line 77):
        
        # Obtaining the type of the subscript
        
        # Getting the type of 'v' (line 77)
        v_509968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 18), 'v')
        # Getting the type of 'self' (line 77)
        self_509969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 23), 'self')
        # Obtaining the member 'a' of a type (line 77)
        a_509970 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 23), self_509969, 'a')
        # Applying the binary operator '>=' (line 77)
        result_ge_509971 = python_operator(stypy.reporting.localization.Localization(__file__, 77, 18), '>=', v_509968, a_509970)
        
        # Getting the type of 'v' (line 77)
        v_509972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 16), 'v')
        # Obtaining the member '__getitem__' of a type (line 77)
        getitem___509973 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 16), v_509972, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 77)
        subscript_call_result_509974 = invoke(stypy.reporting.localization.Localization(__file__, 77, 16), getitem___509973, result_ge_509971)
        
        # Assigning a type to the variable 'v' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 12), 'v', subscript_call_result_509974)
        # SSA branch for the else part of an if statement (line 76)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Subscript to a Name (line 79):
        
        # Assigning a Subscript to a Name (line 79):
        
        # Obtaining the type of the subscript
        
        # Getting the type of 'v' (line 79)
        v_509975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 18), 'v')
        # Getting the type of 'self' (line 79)
        self_509976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 22), 'self')
        # Obtaining the member 'a' of a type (line 79)
        a_509977 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 22), self_509976, 'a')
        # Applying the binary operator '>' (line 79)
        result_gt_509978 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 18), '>', v_509975, a_509977)
        
        # Getting the type of 'v' (line 79)
        v_509979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 16), 'v')
        # Obtaining the member '__getitem__' of a type (line 79)
        getitem___509980 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 16), v_509979, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 79)
        subscript_call_result_509981 = invoke(stypy.reporting.localization.Localization(__file__, 79, 16), getitem___509980, result_gt_509978)
        
        # Assigning a type to the variable 'v' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 12), 'v', subscript_call_result_509981)
        # SSA join for if statement (line 76)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'self' (line 80)
        self_509982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 11), 'self')
        # Obtaining the member 'inclusive_b' of a type (line 80)
        inclusive_b_509983 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 11), self_509982, 'inclusive_b')
        # Testing the type of an if condition (line 80)
        if_condition_509984 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 80, 8), inclusive_b_509983)
        # Assigning a type to the variable 'if_condition_509984' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'if_condition_509984', if_condition_509984)
        # SSA begins for if statement (line 80)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Name (line 81):
        
        # Assigning a Subscript to a Name (line 81):
        
        # Obtaining the type of the subscript
        
        # Getting the type of 'v' (line 81)
        v_509985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 18), 'v')
        # Getting the type of 'self' (line 81)
        self_509986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 23), 'self')
        # Obtaining the member 'b' of a type (line 81)
        b_509987 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 23), self_509986, 'b')
        # Applying the binary operator '<=' (line 81)
        result_le_509988 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 18), '<=', v_509985, b_509987)
        
        # Getting the type of 'v' (line 81)
        v_509989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 16), 'v')
        # Obtaining the member '__getitem__' of a type (line 81)
        getitem___509990 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 16), v_509989, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 81)
        subscript_call_result_509991 = invoke(stypy.reporting.localization.Localization(__file__, 81, 16), getitem___509990, result_le_509988)
        
        # Assigning a type to the variable 'v' (line 81)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 12), 'v', subscript_call_result_509991)
        # SSA branch for the else part of an if statement (line 80)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Subscript to a Name (line 83):
        
        # Assigning a Subscript to a Name (line 83):
        
        # Obtaining the type of the subscript
        
        # Getting the type of 'v' (line 83)
        v_509992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 18), 'v')
        # Getting the type of 'self' (line 83)
        self_509993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 22), 'self')
        # Obtaining the member 'b' of a type (line 83)
        b_509994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 22), self_509993, 'b')
        # Applying the binary operator '<' (line 83)
        result_lt_509995 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 18), '<', v_509992, b_509994)
        
        # Getting the type of 'v' (line 83)
        v_509996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 16), 'v')
        # Obtaining the member '__getitem__' of a type (line 83)
        getitem___509997 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 16), v_509996, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 83)
        subscript_call_result_509998 = invoke(stypy.reporting.localization.Localization(__file__, 83, 16), getitem___509997, result_lt_509995)
        
        # Assigning a type to the variable 'v' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 12), 'v', subscript_call_result_509998)
        # SSA join for if statement (line 80)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to unique(...): (line 84)
        # Processing the call arguments (line 84)
        # Getting the type of 'v' (line 84)
        v_510001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 25), 'v', False)
        # Processing the call keyword arguments (line 84)
        kwargs_510002 = {}
        # Getting the type of 'np' (line 84)
        np_509999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 15), 'np', False)
        # Obtaining the member 'unique' of a type (line 84)
        unique_510000 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 15), np_509999, 'unique')
        # Calling unique(args, kwargs) (line 84)
        unique_call_result_510003 = invoke(stypy.reporting.localization.Localization(__file__, 84, 15), unique_510000, *[v_510001], **kwargs_510002)
        
        # Assigning a type to the variable 'stypy_return_type' (line 84)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'stypy_return_type', unique_call_result_510003)
        
        # ################# End of 'values(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'values' in the type store
        # Getting the type of 'stypy_return_type' (line 40)
        stypy_return_type_510004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_510004)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'values'
        return stypy_return_type_510004


# Assigning a type to the variable 'Arg' (line 24)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'Arg', Arg)
# Declaration of the 'FixedArg' class

class FixedArg(object, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 88, 4, False)
        # Assigning a type to the variable 'self' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FixedArg.__init__', ['values'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['values'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Call to a Attribute (line 89):
        
        # Assigning a Call to a Attribute (line 89):
        
        # Call to asarray(...): (line 89)
        # Processing the call arguments (line 89)
        # Getting the type of 'values' (line 89)
        values_510007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 34), 'values', False)
        # Processing the call keyword arguments (line 89)
        kwargs_510008 = {}
        # Getting the type of 'np' (line 89)
        np_510005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 23), 'np', False)
        # Obtaining the member 'asarray' of a type (line 89)
        asarray_510006 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 23), np_510005, 'asarray')
        # Calling asarray(args, kwargs) (line 89)
        asarray_call_result_510009 = invoke(stypy.reporting.localization.Localization(__file__, 89, 23), asarray_510006, *[values_510007], **kwargs_510008)
        
        # Getting the type of 'self' (line 89)
        self_510010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'self')
        # Setting the type of the member '_values' of a type (line 89)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 8), self_510010, '_values', asarray_call_result_510009)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def values(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'values'
        module_type_store = module_type_store.open_function_context('values', 91, 4, False)
        # Assigning a type to the variable 'self' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FixedArg.values.__dict__.__setitem__('stypy_localization', localization)
        FixedArg.values.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FixedArg.values.__dict__.__setitem__('stypy_type_store', module_type_store)
        FixedArg.values.__dict__.__setitem__('stypy_function_name', 'FixedArg.values')
        FixedArg.values.__dict__.__setitem__('stypy_param_names_list', ['n'])
        FixedArg.values.__dict__.__setitem__('stypy_varargs_param_name', None)
        FixedArg.values.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FixedArg.values.__dict__.__setitem__('stypy_call_defaults', defaults)
        FixedArg.values.__dict__.__setitem__('stypy_call_varargs', varargs)
        FixedArg.values.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FixedArg.values.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FixedArg.values', ['n'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'values', localization, ['n'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'values(...)' code ##################

        # Getting the type of 'self' (line 92)
        self_510011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 15), 'self')
        # Obtaining the member '_values' of a type (line 92)
        _values_510012 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 15), self_510011, '_values')
        # Assigning a type to the variable 'stypy_return_type' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'stypy_return_type', _values_510012)
        
        # ################# End of 'values(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'values' in the type store
        # Getting the type of 'stypy_return_type' (line 91)
        stypy_return_type_510013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_510013)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'values'
        return stypy_return_type_510013


# Assigning a type to the variable 'FixedArg' (line 87)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 0), 'FixedArg', FixedArg)
# Declaration of the 'ComplexArg' class

class ComplexArg(object, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        
        # Call to complex(...): (line 96)
        # Processing the call arguments (line 96)
        
        # Getting the type of 'np' (line 96)
        np_510015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 34), 'np', False)
        # Obtaining the member 'inf' of a type (line 96)
        inf_510016 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 34), np_510015, 'inf')
        # Applying the 'usub' unary operator (line 96)
        result___neg___510017 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 33), 'usub', inf_510016)
        
        
        # Getting the type of 'np' (line 96)
        np_510018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 43), 'np', False)
        # Obtaining the member 'inf' of a type (line 96)
        inf_510019 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 43), np_510018, 'inf')
        # Applying the 'usub' unary operator (line 96)
        result___neg___510020 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 42), 'usub', inf_510019)
        
        # Processing the call keyword arguments (line 96)
        kwargs_510021 = {}
        # Getting the type of 'complex' (line 96)
        complex_510014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 25), 'complex', False)
        # Calling complex(args, kwargs) (line 96)
        complex_call_result_510022 = invoke(stypy.reporting.localization.Localization(__file__, 96, 25), complex_510014, *[result___neg___510017, result___neg___510020], **kwargs_510021)
        
        
        # Call to complex(...): (line 96)
        # Processing the call arguments (line 96)
        # Getting the type of 'np' (line 96)
        np_510024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 62), 'np', False)
        # Obtaining the member 'inf' of a type (line 96)
        inf_510025 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 62), np_510024, 'inf')
        # Getting the type of 'np' (line 96)
        np_510026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 70), 'np', False)
        # Obtaining the member 'inf' of a type (line 96)
        inf_510027 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 70), np_510026, 'inf')
        # Processing the call keyword arguments (line 96)
        kwargs_510028 = {}
        # Getting the type of 'complex' (line 96)
        complex_510023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 54), 'complex', False)
        # Calling complex(args, kwargs) (line 96)
        complex_call_result_510029 = invoke(stypy.reporting.localization.Localization(__file__, 96, 54), complex_510023, *[inf_510025, inf_510027], **kwargs_510028)
        
        defaults = [complex_call_result_510022, complex_call_result_510029]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 96, 4, False)
        # Assigning a type to the variable 'self' (line 97)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ComplexArg.__init__', ['a', 'b'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['a', 'b'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Call to a Attribute (line 97):
        
        # Assigning a Call to a Attribute (line 97):
        
        # Call to Arg(...): (line 97)
        # Processing the call arguments (line 97)
        # Getting the type of 'a' (line 97)
        a_510031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 24), 'a', False)
        # Obtaining the member 'real' of a type (line 97)
        real_510032 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 24), a_510031, 'real')
        # Getting the type of 'b' (line 97)
        b_510033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 32), 'b', False)
        # Obtaining the member 'real' of a type (line 97)
        real_510034 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 32), b_510033, 'real')
        # Processing the call keyword arguments (line 97)
        kwargs_510035 = {}
        # Getting the type of 'Arg' (line 97)
        Arg_510030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 20), 'Arg', False)
        # Calling Arg(args, kwargs) (line 97)
        Arg_call_result_510036 = invoke(stypy.reporting.localization.Localization(__file__, 97, 20), Arg_510030, *[real_510032, real_510034], **kwargs_510035)
        
        # Getting the type of 'self' (line 97)
        self_510037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'self')
        # Setting the type of the member 'real' of a type (line 97)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 8), self_510037, 'real', Arg_call_result_510036)
        
        # Assigning a Call to a Attribute (line 98):
        
        # Assigning a Call to a Attribute (line 98):
        
        # Call to Arg(...): (line 98)
        # Processing the call arguments (line 98)
        # Getting the type of 'a' (line 98)
        a_510039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 24), 'a', False)
        # Obtaining the member 'imag' of a type (line 98)
        imag_510040 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 24), a_510039, 'imag')
        # Getting the type of 'b' (line 98)
        b_510041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 32), 'b', False)
        # Obtaining the member 'imag' of a type (line 98)
        imag_510042 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 32), b_510041, 'imag')
        # Processing the call keyword arguments (line 98)
        kwargs_510043 = {}
        # Getting the type of 'Arg' (line 98)
        Arg_510038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 20), 'Arg', False)
        # Calling Arg(args, kwargs) (line 98)
        Arg_call_result_510044 = invoke(stypy.reporting.localization.Localization(__file__, 98, 20), Arg_510038, *[imag_510040, imag_510042], **kwargs_510043)
        
        # Getting the type of 'self' (line 98)
        self_510045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'self')
        # Setting the type of the member 'imag' of a type (line 98)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 8), self_510045, 'imag', Arg_call_result_510044)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def values(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'values'
        module_type_store = module_type_store.open_function_context('values', 100, 4, False)
        # Assigning a type to the variable 'self' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ComplexArg.values.__dict__.__setitem__('stypy_localization', localization)
        ComplexArg.values.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ComplexArg.values.__dict__.__setitem__('stypy_type_store', module_type_store)
        ComplexArg.values.__dict__.__setitem__('stypy_function_name', 'ComplexArg.values')
        ComplexArg.values.__dict__.__setitem__('stypy_param_names_list', ['n'])
        ComplexArg.values.__dict__.__setitem__('stypy_varargs_param_name', None)
        ComplexArg.values.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ComplexArg.values.__dict__.__setitem__('stypy_call_defaults', defaults)
        ComplexArg.values.__dict__.__setitem__('stypy_call_varargs', varargs)
        ComplexArg.values.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ComplexArg.values.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ComplexArg.values', ['n'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'values', localization, ['n'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'values(...)' code ##################

        
        # Assigning a Call to a Name (line 101):
        
        # Assigning a Call to a Name (line 101):
        
        # Call to max(...): (line 101)
        # Processing the call arguments (line 101)
        int_510047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 16), 'int')
        
        # Call to int(...): (line 101)
        # Processing the call arguments (line 101)
        
        # Call to sqrt(...): (line 101)
        # Processing the call arguments (line 101)
        # Getting the type of 'n' (line 101)
        n_510051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 31), 'n', False)
        # Processing the call keyword arguments (line 101)
        kwargs_510052 = {}
        # Getting the type of 'np' (line 101)
        np_510049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 23), 'np', False)
        # Obtaining the member 'sqrt' of a type (line 101)
        sqrt_510050 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 23), np_510049, 'sqrt')
        # Calling sqrt(args, kwargs) (line 101)
        sqrt_call_result_510053 = invoke(stypy.reporting.localization.Localization(__file__, 101, 23), sqrt_510050, *[n_510051], **kwargs_510052)
        
        # Processing the call keyword arguments (line 101)
        kwargs_510054 = {}
        # Getting the type of 'int' (line 101)
        int_510048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 19), 'int', False)
        # Calling int(args, kwargs) (line 101)
        int_call_result_510055 = invoke(stypy.reporting.localization.Localization(__file__, 101, 19), int_510048, *[sqrt_call_result_510053], **kwargs_510054)
        
        # Processing the call keyword arguments (line 101)
        kwargs_510056 = {}
        # Getting the type of 'max' (line 101)
        max_510046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 12), 'max', False)
        # Calling max(args, kwargs) (line 101)
        max_call_result_510057 = invoke(stypy.reporting.localization.Localization(__file__, 101, 12), max_510046, *[int_510047, int_call_result_510055], **kwargs_510056)
        
        # Assigning a type to the variable 'm' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'm', max_call_result_510057)
        
        # Assigning a Call to a Name (line 102):
        
        # Assigning a Call to a Name (line 102):
        
        # Call to values(...): (line 102)
        # Processing the call arguments (line 102)
        # Getting the type of 'm' (line 102)
        m_510061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 29), 'm', False)
        # Processing the call keyword arguments (line 102)
        kwargs_510062 = {}
        # Getting the type of 'self' (line 102)
        self_510058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 12), 'self', False)
        # Obtaining the member 'real' of a type (line 102)
        real_510059 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 12), self_510058, 'real')
        # Obtaining the member 'values' of a type (line 102)
        values_510060 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 12), real_510059, 'values')
        # Calling values(args, kwargs) (line 102)
        values_call_result_510063 = invoke(stypy.reporting.localization.Localization(__file__, 102, 12), values_510060, *[m_510061], **kwargs_510062)
        
        # Assigning a type to the variable 'x' (line 102)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'x', values_call_result_510063)
        
        # Assigning a Call to a Name (line 103):
        
        # Assigning a Call to a Name (line 103):
        
        # Call to values(...): (line 103)
        # Processing the call arguments (line 103)
        # Getting the type of 'm' (line 103)
        m_510067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 29), 'm', False)
        # Processing the call keyword arguments (line 103)
        kwargs_510068 = {}
        # Getting the type of 'self' (line 103)
        self_510064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 12), 'self', False)
        # Obtaining the member 'imag' of a type (line 103)
        imag_510065 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 12), self_510064, 'imag')
        # Obtaining the member 'values' of a type (line 103)
        values_510066 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 12), imag_510065, 'values')
        # Calling values(args, kwargs) (line 103)
        values_call_result_510069 = invoke(stypy.reporting.localization.Localization(__file__, 103, 12), values_510066, *[m_510067], **kwargs_510068)
        
        # Assigning a type to the variable 'y' (line 103)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'y', values_call_result_510069)
        
        # Call to ravel(...): (line 104)
        # Processing the call keyword arguments (line 104)
        kwargs_510084 = {}
        
        # Obtaining the type of the subscript
        slice_510070 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 104, 16), None, None, None)
        # Getting the type of 'None' (line 104)
        None_510071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 20), 'None', False)
        # Getting the type of 'x' (line 104)
        x_510072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 16), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 104)
        getitem___510073 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 16), x_510072, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 104)
        subscript_call_result_510074 = invoke(stypy.reporting.localization.Localization(__file__, 104, 16), getitem___510073, (slice_510070, None_510071))
        
        complex_510075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 28), 'complex')
        
        # Obtaining the type of the subscript
        # Getting the type of 'None' (line 104)
        None_510076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 33), 'None', False)
        slice_510077 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 104, 31), None, None, None)
        # Getting the type of 'y' (line 104)
        y_510078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 31), 'y', False)
        # Obtaining the member '__getitem__' of a type (line 104)
        getitem___510079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 31), y_510078, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 104)
        subscript_call_result_510080 = invoke(stypy.reporting.localization.Localization(__file__, 104, 31), getitem___510079, (None_510076, slice_510077))
        
        # Applying the binary operator '*' (line 104)
        result_mul_510081 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 28), '*', complex_510075, subscript_call_result_510080)
        
        # Applying the binary operator '+' (line 104)
        result_add_510082 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 16), '+', subscript_call_result_510074, result_mul_510081)
        
        # Obtaining the member 'ravel' of a type (line 104)
        ravel_510083 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 16), result_add_510082, 'ravel')
        # Calling ravel(args, kwargs) (line 104)
        ravel_call_result_510085 = invoke(stypy.reporting.localization.Localization(__file__, 104, 16), ravel_510083, *[], **kwargs_510084)
        
        # Assigning a type to the variable 'stypy_return_type' (line 104)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'stypy_return_type', ravel_call_result_510085)
        
        # ################# End of 'values(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'values' in the type store
        # Getting the type of 'stypy_return_type' (line 100)
        stypy_return_type_510086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_510086)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'values'
        return stypy_return_type_510086


# Assigning a type to the variable 'ComplexArg' (line 95)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 0), 'ComplexArg', ComplexArg)
# Declaration of the 'IntArg' class

class IntArg(object, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_510087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 25), 'int')
        int_510088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 34), 'int')
        defaults = [int_510087, int_510088]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 108, 4, False)
        # Assigning a type to the variable 'self' (line 109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'IntArg.__init__', ['a', 'b'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['a', 'b'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 109):
        
        # Assigning a Name to a Attribute (line 109):
        # Getting the type of 'a' (line 109)
        a_510089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 17), 'a')
        # Getting the type of 'self' (line 109)
        self_510090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'self')
        # Setting the type of the member 'a' of a type (line 109)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 8), self_510090, 'a', a_510089)
        
        # Assigning a Name to a Attribute (line 110):
        
        # Assigning a Name to a Attribute (line 110):
        # Getting the type of 'b' (line 110)
        b_510091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 17), 'b')
        # Getting the type of 'self' (line 110)
        self_510092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'self')
        # Setting the type of the member 'b' of a type (line 110)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 8), self_510092, 'b', b_510091)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def values(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'values'
        module_type_store = module_type_store.open_function_context('values', 112, 4, False)
        # Assigning a type to the variable 'self' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        IntArg.values.__dict__.__setitem__('stypy_localization', localization)
        IntArg.values.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        IntArg.values.__dict__.__setitem__('stypy_type_store', module_type_store)
        IntArg.values.__dict__.__setitem__('stypy_function_name', 'IntArg.values')
        IntArg.values.__dict__.__setitem__('stypy_param_names_list', ['n'])
        IntArg.values.__dict__.__setitem__('stypy_varargs_param_name', None)
        IntArg.values.__dict__.__setitem__('stypy_kwargs_param_name', None)
        IntArg.values.__dict__.__setitem__('stypy_call_defaults', defaults)
        IntArg.values.__dict__.__setitem__('stypy_call_varargs', varargs)
        IntArg.values.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        IntArg.values.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'IntArg.values', ['n'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'values', localization, ['n'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'values(...)' code ##################

        
        # Assigning a Call to a Name (line 113):
        
        # Assigning a Call to a Name (line 113):
        
        # Call to astype(...): (line 113)
        # Processing the call arguments (line 113)
        # Getting the type of 'int' (line 113)
        int_510115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 67), 'int', False)
        # Processing the call keyword arguments (line 113)
        kwargs_510116 = {}
        
        # Call to values(...): (line 113)
        # Processing the call arguments (line 113)
        
        # Call to max(...): (line 113)
        # Processing the call arguments (line 113)
        int_510102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 44), 'int')
        # Getting the type of 'n' (line 113)
        n_510103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 48), 'n', False)
        int_510104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 51), 'int')
        # Applying the binary operator '//' (line 113)
        result_floordiv_510105 = python_operator(stypy.reporting.localization.Localization(__file__, 113, 48), '//', n_510103, int_510104)
        
        # Applying the binary operator '+' (line 113)
        result_add_510106 = python_operator(stypy.reporting.localization.Localization(__file__, 113, 44), '+', int_510102, result_floordiv_510105)
        
        # Getting the type of 'n' (line 113)
        n_510107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 54), 'n', False)
        int_510108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 56), 'int')
        # Applying the binary operator '-' (line 113)
        result_sub_510109 = python_operator(stypy.reporting.localization.Localization(__file__, 113, 54), '-', n_510107, int_510108)
        
        # Processing the call keyword arguments (line 113)
        kwargs_510110 = {}
        # Getting the type of 'max' (line 113)
        max_510101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 40), 'max', False)
        # Calling max(args, kwargs) (line 113)
        max_call_result_510111 = invoke(stypy.reporting.localization.Localization(__file__, 113, 40), max_510101, *[result_add_510106, result_sub_510109], **kwargs_510110)
        
        # Processing the call keyword arguments (line 113)
        kwargs_510112 = {}
        
        # Call to Arg(...): (line 113)
        # Processing the call arguments (line 113)
        # Getting the type of 'self' (line 113)
        self_510094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 17), 'self', False)
        # Obtaining the member 'a' of a type (line 113)
        a_510095 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 17), self_510094, 'a')
        # Getting the type of 'self' (line 113)
        self_510096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 25), 'self', False)
        # Obtaining the member 'b' of a type (line 113)
        b_510097 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 25), self_510096, 'b')
        # Processing the call keyword arguments (line 113)
        kwargs_510098 = {}
        # Getting the type of 'Arg' (line 113)
        Arg_510093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 13), 'Arg', False)
        # Calling Arg(args, kwargs) (line 113)
        Arg_call_result_510099 = invoke(stypy.reporting.localization.Localization(__file__, 113, 13), Arg_510093, *[a_510095, b_510097], **kwargs_510098)
        
        # Obtaining the member 'values' of a type (line 113)
        values_510100 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 13), Arg_call_result_510099, 'values')
        # Calling values(args, kwargs) (line 113)
        values_call_result_510113 = invoke(stypy.reporting.localization.Localization(__file__, 113, 13), values_510100, *[max_call_result_510111], **kwargs_510112)
        
        # Obtaining the member 'astype' of a type (line 113)
        astype_510114 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 13), values_call_result_510113, 'astype')
        # Calling astype(args, kwargs) (line 113)
        astype_call_result_510117 = invoke(stypy.reporting.localization.Localization(__file__, 113, 13), astype_510114, *[int_510115], **kwargs_510116)
        
        # Assigning a type to the variable 'v1' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'v1', astype_call_result_510117)
        
        # Assigning a Call to a Name (line 114):
        
        # Assigning a Call to a Name (line 114):
        
        # Call to arange(...): (line 114)
        # Processing the call arguments (line 114)
        int_510120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 23), 'int')
        int_510121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 27), 'int')
        # Processing the call keyword arguments (line 114)
        kwargs_510122 = {}
        # Getting the type of 'np' (line 114)
        np_510118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 13), 'np', False)
        # Obtaining the member 'arange' of a type (line 114)
        arange_510119 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 13), np_510118, 'arange')
        # Calling arange(args, kwargs) (line 114)
        arange_call_result_510123 = invoke(stypy.reporting.localization.Localization(__file__, 114, 13), arange_510119, *[int_510120, int_510121], **kwargs_510122)
        
        # Assigning a type to the variable 'v2' (line 114)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'v2', arange_call_result_510123)
        
        # Assigning a Call to a Name (line 115):
        
        # Assigning a Call to a Name (line 115):
        
        # Call to unique(...): (line 115)
        # Processing the call arguments (line 115)
        
        # Obtaining the type of the subscript
        
        # Obtaining an instance of the builtin type 'tuple' (line 115)
        tuple_510126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 115)
        # Adding element type (line 115)
        # Getting the type of 'v1' (line 115)
        v1_510127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 28), 'v1', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 115, 28), tuple_510126, v1_510127)
        # Adding element type (line 115)
        # Getting the type of 'v2' (line 115)
        v2_510128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 32), 'v2', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 115, 28), tuple_510126, v2_510128)
        
        # Getting the type of 'np' (line 115)
        np_510129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 22), 'np', False)
        # Obtaining the member 'r_' of a type (line 115)
        r__510130 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 22), np_510129, 'r_')
        # Obtaining the member '__getitem__' of a type (line 115)
        getitem___510131 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 22), r__510130, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 115)
        subscript_call_result_510132 = invoke(stypy.reporting.localization.Localization(__file__, 115, 22), getitem___510131, tuple_510126)
        
        # Processing the call keyword arguments (line 115)
        kwargs_510133 = {}
        # Getting the type of 'np' (line 115)
        np_510124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 12), 'np', False)
        # Obtaining the member 'unique' of a type (line 115)
        unique_510125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 12), np_510124, 'unique')
        # Calling unique(args, kwargs) (line 115)
        unique_call_result_510134 = invoke(stypy.reporting.localization.Localization(__file__, 115, 12), unique_510125, *[subscript_call_result_510132], **kwargs_510133)
        
        # Assigning a type to the variable 'v' (line 115)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 8), 'v', unique_call_result_510134)
        
        # Assigning a Subscript to a Name (line 116):
        
        # Assigning a Subscript to a Name (line 116):
        
        # Obtaining the type of the subscript
        
        # Getting the type of 'v' (line 116)
        v_510135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 15), 'v')
        # Getting the type of 'self' (line 116)
        self_510136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 20), 'self')
        # Obtaining the member 'a' of a type (line 116)
        a_510137 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 20), self_510136, 'a')
        # Applying the binary operator '>=' (line 116)
        result_ge_510138 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 15), '>=', v_510135, a_510137)
        
        
        # Getting the type of 'v' (line 116)
        v_510139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 31), 'v')
        # Getting the type of 'self' (line 116)
        self_510140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 35), 'self')
        # Obtaining the member 'b' of a type (line 116)
        b_510141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 35), self_510140, 'b')
        # Applying the binary operator '<' (line 116)
        result_lt_510142 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 31), '<', v_510139, b_510141)
        
        # Applying the binary operator '&' (line 116)
        result_and__510143 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 14), '&', result_ge_510138, result_lt_510142)
        
        # Getting the type of 'v' (line 116)
        v_510144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 12), 'v')
        # Obtaining the member '__getitem__' of a type (line 116)
        getitem___510145 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 12), v_510144, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 116)
        subscript_call_result_510146 = invoke(stypy.reporting.localization.Localization(__file__, 116, 12), getitem___510145, result_and__510143)
        
        # Assigning a type to the variable 'v' (line 116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'v', subscript_call_result_510146)
        # Getting the type of 'v' (line 117)
        v_510147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 15), 'v')
        # Assigning a type to the variable 'stypy_return_type' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'stypy_return_type', v_510147)
        
        # ################# End of 'values(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'values' in the type store
        # Getting the type of 'stypy_return_type' (line 112)
        stypy_return_type_510148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_510148)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'values'
        return stypy_return_type_510148


# Assigning a type to the variable 'IntArg' (line 107)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 0), 'IntArg', IntArg)

@norecursion
def get_args(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'get_args'
    module_type_store = module_type_store.open_function_context('get_args', 120, 0, False)
    
    # Passed parameters checking function
    get_args.stypy_localization = localization
    get_args.stypy_type_of_self = None
    get_args.stypy_type_store = module_type_store
    get_args.stypy_function_name = 'get_args'
    get_args.stypy_param_names_list = ['argspec', 'n']
    get_args.stypy_varargs_param_name = None
    get_args.stypy_kwargs_param_name = None
    get_args.stypy_call_defaults = defaults
    get_args.stypy_call_varargs = varargs
    get_args.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'get_args', ['argspec', 'n'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'get_args', localization, ['argspec', 'n'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'get_args(...)' code ##################

    
    
    # Call to isinstance(...): (line 121)
    # Processing the call arguments (line 121)
    # Getting the type of 'argspec' (line 121)
    argspec_510150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 18), 'argspec', False)
    # Getting the type of 'np' (line 121)
    np_510151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 27), 'np', False)
    # Obtaining the member 'ndarray' of a type (line 121)
    ndarray_510152 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 27), np_510151, 'ndarray')
    # Processing the call keyword arguments (line 121)
    kwargs_510153 = {}
    # Getting the type of 'isinstance' (line 121)
    isinstance_510149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 7), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 121)
    isinstance_call_result_510154 = invoke(stypy.reporting.localization.Localization(__file__, 121, 7), isinstance_510149, *[argspec_510150, ndarray_510152], **kwargs_510153)
    
    # Testing the type of an if condition (line 121)
    if_condition_510155 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 121, 4), isinstance_call_result_510154)
    # Assigning a type to the variable 'if_condition_510155' (line 121)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 4), 'if_condition_510155', if_condition_510155)
    # SSA begins for if statement (line 121)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 122):
    
    # Assigning a Call to a Name (line 122):
    
    # Call to copy(...): (line 122)
    # Processing the call keyword arguments (line 122)
    kwargs_510158 = {}
    # Getting the type of 'argspec' (line 122)
    argspec_510156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 15), 'argspec', False)
    # Obtaining the member 'copy' of a type (line 122)
    copy_510157 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 15), argspec_510156, 'copy')
    # Calling copy(args, kwargs) (line 122)
    copy_call_result_510159 = invoke(stypy.reporting.localization.Localization(__file__, 122, 15), copy_510157, *[], **kwargs_510158)
    
    # Assigning a type to the variable 'args' (line 122)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'args', copy_call_result_510159)
    # SSA branch for the else part of an if statement (line 121)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 124):
    
    # Assigning a Call to a Name (line 124):
    
    # Call to len(...): (line 124)
    # Processing the call arguments (line 124)
    # Getting the type of 'argspec' (line 124)
    argspec_510161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 20), 'argspec', False)
    # Processing the call keyword arguments (line 124)
    kwargs_510162 = {}
    # Getting the type of 'len' (line 124)
    len_510160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 16), 'len', False)
    # Calling len(args, kwargs) (line 124)
    len_call_result_510163 = invoke(stypy.reporting.localization.Localization(__file__, 124, 16), len_510160, *[argspec_510161], **kwargs_510162)
    
    # Assigning a type to the variable 'nargs' (line 124)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'nargs', len_call_result_510163)
    
    # Assigning a Call to a Name (line 125):
    
    # Assigning a Call to a Name (line 125):
    
    # Call to asarray(...): (line 125)
    # Processing the call arguments (line 125)
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'argspec' (line 125)
    argspec_510174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 82), 'argspec', False)
    comprehension_510175 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 25), argspec_510174)
    # Assigning a type to the variable 'spec' (line 125)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 25), 'spec', comprehension_510175)
    
    
    # Call to isinstance(...): (line 125)
    # Processing the call arguments (line 125)
    # Getting the type of 'spec' (line 125)
    spec_510167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 43), 'spec', False)
    # Getting the type of 'ComplexArg' (line 125)
    ComplexArg_510168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 49), 'ComplexArg', False)
    # Processing the call keyword arguments (line 125)
    kwargs_510169 = {}
    # Getting the type of 'isinstance' (line 125)
    isinstance_510166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 32), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 125)
    isinstance_call_result_510170 = invoke(stypy.reporting.localization.Localization(__file__, 125, 32), isinstance_510166, *[spec_510167, ComplexArg_510168], **kwargs_510169)
    
    # Testing the type of an if expression (line 125)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 125, 25), isinstance_call_result_510170)
    # SSA begins for if expression (line 125)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    float_510171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 25), 'float')
    # SSA branch for the else part of an if expression (line 125)
    module_type_store.open_ssa_branch('if expression else')
    float_510172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 66), 'float')
    # SSA join for if expression (line 125)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_510173 = union_type.UnionType.add(float_510171, float_510172)
    
    list_510176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 25), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 25), list_510176, if_exp_510173)
    # Processing the call keyword arguments (line 125)
    kwargs_510177 = {}
    # Getting the type of 'np' (line 125)
    np_510164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 13), 'np', False)
    # Obtaining the member 'asarray' of a type (line 125)
    asarray_510165 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 13), np_510164, 'asarray')
    # Calling asarray(args, kwargs) (line 125)
    asarray_call_result_510178 = invoke(stypy.reporting.localization.Localization(__file__, 125, 13), asarray_510165, *[list_510176], **kwargs_510177)
    
    # Assigning a type to the variable 'ms' (line 125)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'ms', asarray_call_result_510178)
    
    # Assigning a BinOp to a Name (line 126):
    
    # Assigning a BinOp to a Name (line 126):
    
    # Call to astype(...): (line 126)
    # Processing the call arguments (line 126)
    # Getting the type of 'int' (line 126)
    int_510188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 38), 'int', False)
    # Processing the call keyword arguments (line 126)
    kwargs_510189 = {}
    # Getting the type of 'n' (line 126)
    n_510179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 14), 'n', False)
    # Getting the type of 'ms' (line 126)
    ms_510180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 18), 'ms', False)
    
    # Call to sum(...): (line 126)
    # Processing the call arguments (line 126)
    # Getting the type of 'ms' (line 126)
    ms_510182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 25), 'ms', False)
    # Processing the call keyword arguments (line 126)
    kwargs_510183 = {}
    # Getting the type of 'sum' (line 126)
    sum_510181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 21), 'sum', False)
    # Calling sum(args, kwargs) (line 126)
    sum_call_result_510184 = invoke(stypy.reporting.localization.Localization(__file__, 126, 21), sum_510181, *[ms_510182], **kwargs_510183)
    
    # Applying the binary operator 'div' (line 126)
    result_div_510185 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 18), 'div', ms_510180, sum_call_result_510184)
    
    # Applying the binary operator '**' (line 126)
    result_pow_510186 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 14), '**', n_510179, result_div_510185)
    
    # Obtaining the member 'astype' of a type (line 126)
    astype_510187 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 14), result_pow_510186, 'astype')
    # Calling astype(args, kwargs) (line 126)
    astype_call_result_510190 = invoke(stypy.reporting.localization.Localization(__file__, 126, 14), astype_510187, *[int_510188], **kwargs_510189)
    
    int_510191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 45), 'int')
    # Applying the binary operator '+' (line 126)
    result_add_510192 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 13), '+', astype_call_result_510190, int_510191)
    
    # Assigning a type to the variable 'ms' (line 126)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'ms', result_add_510192)
    
    # Assigning a List to a Name (line 128):
    
    # Assigning a List to a Name (line 128):
    
    # Obtaining an instance of the builtin type 'list' (line 128)
    list_510193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 128)
    
    # Assigning a type to the variable 'args' (line 128)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'args', list_510193)
    
    
    # Call to zip(...): (line 129)
    # Processing the call arguments (line 129)
    # Getting the type of 'argspec' (line 129)
    argspec_510195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 27), 'argspec', False)
    # Getting the type of 'ms' (line 129)
    ms_510196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 36), 'ms', False)
    # Processing the call keyword arguments (line 129)
    kwargs_510197 = {}
    # Getting the type of 'zip' (line 129)
    zip_510194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 23), 'zip', False)
    # Calling zip(args, kwargs) (line 129)
    zip_call_result_510198 = invoke(stypy.reporting.localization.Localization(__file__, 129, 23), zip_510194, *[argspec_510195, ms_510196], **kwargs_510197)
    
    # Testing the type of a for loop iterable (line 129)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 129, 8), zip_call_result_510198)
    # Getting the type of the for loop variable (line 129)
    for_loop_var_510199 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 129, 8), zip_call_result_510198)
    # Assigning a type to the variable 'spec' (line 129)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'spec', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 8), for_loop_var_510199))
    # Assigning a type to the variable 'm' (line 129)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'm', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 8), for_loop_var_510199))
    # SSA begins for a for statement (line 129)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to append(...): (line 130)
    # Processing the call arguments (line 130)
    
    # Call to values(...): (line 130)
    # Processing the call arguments (line 130)
    # Getting the type of 'm' (line 130)
    m_510204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 36), 'm', False)
    # Processing the call keyword arguments (line 130)
    kwargs_510205 = {}
    # Getting the type of 'spec' (line 130)
    spec_510202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 24), 'spec', False)
    # Obtaining the member 'values' of a type (line 130)
    values_510203 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 24), spec_510202, 'values')
    # Calling values(args, kwargs) (line 130)
    values_call_result_510206 = invoke(stypy.reporting.localization.Localization(__file__, 130, 24), values_510203, *[m_510204], **kwargs_510205)
    
    # Processing the call keyword arguments (line 130)
    kwargs_510207 = {}
    # Getting the type of 'args' (line 130)
    args_510200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 12), 'args', False)
    # Obtaining the member 'append' of a type (line 130)
    append_510201 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 12), args_510200, 'append')
    # Calling append(args, kwargs) (line 130)
    append_call_result_510208 = invoke(stypy.reporting.localization.Localization(__file__, 130, 12), append_510201, *[values_call_result_510206], **kwargs_510207)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Attribute to a Name (line 131):
    
    # Assigning a Attribute to a Name (line 131):
    
    # Call to reshape(...): (line 131)
    # Processing the call arguments (line 131)
    # Getting the type of 'nargs' (line 131)
    nargs_510223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 69), 'nargs', False)
    int_510224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 76), 'int')
    # Processing the call keyword arguments (line 131)
    kwargs_510225 = {}
    
    # Call to array(...): (line 131)
    # Processing the call arguments (line 131)
    
    # Call to broadcast_arrays(...): (line 131)
    
    # Call to ix_(...): (line 131)
    # Getting the type of 'args' (line 131)
    args_510215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 53), 'args', False)
    # Processing the call keyword arguments (line 131)
    kwargs_510216 = {}
    # Getting the type of 'np' (line 131)
    np_510213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 45), 'np', False)
    # Obtaining the member 'ix_' of a type (line 131)
    ix__510214 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 45), np_510213, 'ix_')
    # Calling ix_(args, kwargs) (line 131)
    ix__call_result_510217 = invoke(stypy.reporting.localization.Localization(__file__, 131, 45), ix__510214, *[args_510215], **kwargs_510216)
    
    # Processing the call keyword arguments (line 131)
    kwargs_510218 = {}
    # Getting the type of 'np' (line 131)
    np_510211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 24), 'np', False)
    # Obtaining the member 'broadcast_arrays' of a type (line 131)
    broadcast_arrays_510212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 24), np_510211, 'broadcast_arrays')
    # Calling broadcast_arrays(args, kwargs) (line 131)
    broadcast_arrays_call_result_510219 = invoke(stypy.reporting.localization.Localization(__file__, 131, 24), broadcast_arrays_510212, *[ix__call_result_510217], **kwargs_510218)
    
    # Processing the call keyword arguments (line 131)
    kwargs_510220 = {}
    # Getting the type of 'np' (line 131)
    np_510209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 15), 'np', False)
    # Obtaining the member 'array' of a type (line 131)
    array_510210 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 15), np_510209, 'array')
    # Calling array(args, kwargs) (line 131)
    array_call_result_510221 = invoke(stypy.reporting.localization.Localization(__file__, 131, 15), array_510210, *[broadcast_arrays_call_result_510219], **kwargs_510220)
    
    # Obtaining the member 'reshape' of a type (line 131)
    reshape_510222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 15), array_call_result_510221, 'reshape')
    # Calling reshape(args, kwargs) (line 131)
    reshape_call_result_510226 = invoke(stypy.reporting.localization.Localization(__file__, 131, 15), reshape_510222, *[nargs_510223, int_510224], **kwargs_510225)
    
    # Obtaining the member 'T' of a type (line 131)
    T_510227 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 15), reshape_call_result_510226, 'T')
    # Assigning a type to the variable 'args' (line 131)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 8), 'args', T_510227)
    # SSA join for if statement (line 121)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'args' (line 133)
    args_510228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 11), 'args')
    # Assigning a type to the variable 'stypy_return_type' (line 133)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 4), 'stypy_return_type', args_510228)
    
    # ################# End of 'get_args(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_args' in the type store
    # Getting the type of 'stypy_return_type' (line 120)
    stypy_return_type_510229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_510229)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_args'
    return stypy_return_type_510229

# Assigning a type to the variable 'get_args' (line 120)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 0), 'get_args', get_args)
# Declaration of the 'MpmathData' class

class MpmathData(object, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 137)
        None_510230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 63), 'None')
        # Getting the type of 'None' (line 138)
        None_510231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 21), 'None')
        # Getting the type of 'None' (line 138)
        None_510232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 32), 'None')
        # Getting the type of 'None' (line 138)
        None_510233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 40), 'None')
        float_510234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 51), 'float')
        float_510235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 62), 'float')
        # Getting the type of 'False' (line 139)
        False_510236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 33), 'False')
        # Getting the type of 'True' (line 139)
        True_510237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 64), 'True')
        # Getting the type of 'True' (line 140)
        True_510238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 24), 'True')
        # Getting the type of 'None' (line 140)
        None_510239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 43), 'None')
        defaults = [None_510230, None_510231, None_510232, None_510233, float_510234, float_510235, False_510236, True_510237, True_510238, None_510239]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 137, 4, False)
        # Assigning a type to the variable 'self' (line 138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MpmathData.__init__', ['scipy_func', 'mpmath_func', 'arg_spec', 'name', 'dps', 'prec', 'n', 'rtol', 'atol', 'ignore_inf_sign', 'distinguish_nan_and_inf', 'nan_ok', 'param_filter'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['scipy_func', 'mpmath_func', 'arg_spec', 'name', 'dps', 'prec', 'n', 'rtol', 'atol', 'ignore_inf_sign', 'distinguish_nan_and_inf', 'nan_ok', 'param_filter'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Type idiom detected: calculating its left and rigth part (line 145)
        # Getting the type of 'n' (line 145)
        n_510240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 11), 'n')
        # Getting the type of 'None' (line 145)
        None_510241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 16), 'None')
        
        (may_be_510242, more_types_in_union_510243) = may_be_none(n_510240, None_510241)

        if may_be_510242:

            if more_types_in_union_510243:
                # Runtime conditional SSA (line 145)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            
            # SSA begins for try-except statement (line 146)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
            
            # Assigning a Call to a Name (line 147):
            
            # Assigning a Call to a Name (line 147):
            
            # Call to int(...): (line 147)
            # Processing the call arguments (line 147)
            
            # Call to get(...): (line 147)
            # Processing the call arguments (line 147)
            str_510248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 46), 'str', 'SCIPY_XSLOW')
            str_510249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 61), 'str', '0')
            # Processing the call keyword arguments (line 147)
            kwargs_510250 = {}
            # Getting the type of 'os' (line 147)
            os_510245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 31), 'os', False)
            # Obtaining the member 'environ' of a type (line 147)
            environ_510246 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 31), os_510245, 'environ')
            # Obtaining the member 'get' of a type (line 147)
            get_510247 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 31), environ_510246, 'get')
            # Calling get(args, kwargs) (line 147)
            get_call_result_510251 = invoke(stypy.reporting.localization.Localization(__file__, 147, 31), get_510247, *[str_510248, str_510249], **kwargs_510250)
            
            # Processing the call keyword arguments (line 147)
            kwargs_510252 = {}
            # Getting the type of 'int' (line 147)
            int_510244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 27), 'int', False)
            # Calling int(args, kwargs) (line 147)
            int_call_result_510253 = invoke(stypy.reporting.localization.Localization(__file__, 147, 27), int_510244, *[get_call_result_510251], **kwargs_510252)
            
            # Assigning a type to the variable 'is_xslow' (line 147)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 16), 'is_xslow', int_call_result_510253)
            # SSA branch for the except part of a try statement (line 146)
            # SSA branch for the except 'ValueError' branch of a try statement (line 146)
            module_type_store.open_ssa_branch('except')
            
            # Assigning a Name to a Name (line 149):
            
            # Assigning a Name to a Name (line 149):
            # Getting the type of 'False' (line 149)
            False_510254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 27), 'False')
            # Assigning a type to the variable 'is_xslow' (line 149)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 16), 'is_xslow', False_510254)
            # SSA join for try-except statement (line 146)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Assigning a IfExp to a Name (line 151):
            
            # Assigning a IfExp to a Name (line 151):
            
            # Getting the type of 'is_xslow' (line 151)
            is_xslow_510255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 24), 'is_xslow')
            # Testing the type of an if expression (line 151)
            is_suitable_condition(stypy.reporting.localization.Localization(__file__, 151, 16), is_xslow_510255)
            # SSA begins for if expression (line 151)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
            int_510256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 16), 'int')
            # SSA branch for the else part of an if expression (line 151)
            module_type_store.open_ssa_branch('if expression else')
            int_510257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 38), 'int')
            # SSA join for if expression (line 151)
            module_type_store = module_type_store.join_ssa_context()
            if_exp_510258 = union_type.UnionType.add(int_510256, int_510257)
            
            # Assigning a type to the variable 'n' (line 151)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 12), 'n', if_exp_510258)

            if more_types_in_union_510243:
                # SSA join for if statement (line 145)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Name to a Attribute (line 153):
        
        # Assigning a Name to a Attribute (line 153):
        # Getting the type of 'scipy_func' (line 153)
        scipy_func_510259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 26), 'scipy_func')
        # Getting the type of 'self' (line 153)
        self_510260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 8), 'self')
        # Setting the type of the member 'scipy_func' of a type (line 153)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 8), self_510260, 'scipy_func', scipy_func_510259)
        
        # Assigning a Name to a Attribute (line 154):
        
        # Assigning a Name to a Attribute (line 154):
        # Getting the type of 'mpmath_func' (line 154)
        mpmath_func_510261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 27), 'mpmath_func')
        # Getting the type of 'self' (line 154)
        self_510262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'self')
        # Setting the type of the member 'mpmath_func' of a type (line 154)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 8), self_510262, 'mpmath_func', mpmath_func_510261)
        
        # Assigning a Name to a Attribute (line 155):
        
        # Assigning a Name to a Attribute (line 155):
        # Getting the type of 'arg_spec' (line 155)
        arg_spec_510263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 24), 'arg_spec')
        # Getting the type of 'self' (line 155)
        self_510264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 8), 'self')
        # Setting the type of the member 'arg_spec' of a type (line 155)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 8), self_510264, 'arg_spec', arg_spec_510263)
        
        # Assigning a Name to a Attribute (line 156):
        
        # Assigning a Name to a Attribute (line 156):
        # Getting the type of 'dps' (line 156)
        dps_510265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 19), 'dps')
        # Getting the type of 'self' (line 156)
        self_510266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 8), 'self')
        # Setting the type of the member 'dps' of a type (line 156)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 8), self_510266, 'dps', dps_510265)
        
        # Assigning a Name to a Attribute (line 157):
        
        # Assigning a Name to a Attribute (line 157):
        # Getting the type of 'prec' (line 157)
        prec_510267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 20), 'prec')
        # Getting the type of 'self' (line 157)
        self_510268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 8), 'self')
        # Setting the type of the member 'prec' of a type (line 157)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 8), self_510268, 'prec', prec_510267)
        
        # Assigning a Name to a Attribute (line 158):
        
        # Assigning a Name to a Attribute (line 158):
        # Getting the type of 'n' (line 158)
        n_510269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 17), 'n')
        # Getting the type of 'self' (line 158)
        self_510270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'self')
        # Setting the type of the member 'n' of a type (line 158)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 8), self_510270, 'n', n_510269)
        
        # Assigning a Name to a Attribute (line 159):
        
        # Assigning a Name to a Attribute (line 159):
        # Getting the type of 'rtol' (line 159)
        rtol_510271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 20), 'rtol')
        # Getting the type of 'self' (line 159)
        self_510272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'self')
        # Setting the type of the member 'rtol' of a type (line 159)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 8), self_510272, 'rtol', rtol_510271)
        
        # Assigning a Name to a Attribute (line 160):
        
        # Assigning a Name to a Attribute (line 160):
        # Getting the type of 'atol' (line 160)
        atol_510273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 20), 'atol')
        # Getting the type of 'self' (line 160)
        self_510274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'self')
        # Setting the type of the member 'atol' of a type (line 160)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 8), self_510274, 'atol', atol_510273)
        
        # Assigning a Name to a Attribute (line 161):
        
        # Assigning a Name to a Attribute (line 161):
        # Getting the type of 'ignore_inf_sign' (line 161)
        ignore_inf_sign_510275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 31), 'ignore_inf_sign')
        # Getting the type of 'self' (line 161)
        self_510276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 8), 'self')
        # Setting the type of the member 'ignore_inf_sign' of a type (line 161)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 8), self_510276, 'ignore_inf_sign', ignore_inf_sign_510275)
        
        # Assigning a Name to a Attribute (line 162):
        
        # Assigning a Name to a Attribute (line 162):
        # Getting the type of 'nan_ok' (line 162)
        nan_ok_510277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 22), 'nan_ok')
        # Getting the type of 'self' (line 162)
        self_510278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 8), 'self')
        # Setting the type of the member 'nan_ok' of a type (line 162)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 8), self_510278, 'nan_ok', nan_ok_510277)
        
        
        # Call to isinstance(...): (line 163)
        # Processing the call arguments (line 163)
        # Getting the type of 'self' (line 163)
        self_510280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 22), 'self', False)
        # Obtaining the member 'arg_spec' of a type (line 163)
        arg_spec_510281 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 22), self_510280, 'arg_spec')
        # Getting the type of 'np' (line 163)
        np_510282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 37), 'np', False)
        # Obtaining the member 'ndarray' of a type (line 163)
        ndarray_510283 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 37), np_510282, 'ndarray')
        # Processing the call keyword arguments (line 163)
        kwargs_510284 = {}
        # Getting the type of 'isinstance' (line 163)
        isinstance_510279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 163)
        isinstance_call_result_510285 = invoke(stypy.reporting.localization.Localization(__file__, 163, 11), isinstance_510279, *[arg_spec_510281, ndarray_510283], **kwargs_510284)
        
        # Testing the type of an if condition (line 163)
        if_condition_510286 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 163, 8), isinstance_call_result_510285)
        # Assigning a type to the variable 'if_condition_510286' (line 163)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 8), 'if_condition_510286', if_condition_510286)
        # SSA begins for if statement (line 163)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Attribute (line 164):
        
        # Assigning a Call to a Attribute (line 164):
        
        # Call to issubdtype(...): (line 164)
        # Processing the call arguments (line 164)
        # Getting the type of 'self' (line 164)
        self_510289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 44), 'self', False)
        # Obtaining the member 'arg_spec' of a type (line 164)
        arg_spec_510290 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 44), self_510289, 'arg_spec')
        # Obtaining the member 'dtype' of a type (line 164)
        dtype_510291 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 44), arg_spec_510290, 'dtype')
        # Getting the type of 'np' (line 164)
        np_510292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 65), 'np', False)
        # Obtaining the member 'complexfloating' of a type (line 164)
        complexfloating_510293 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 65), np_510292, 'complexfloating')
        # Processing the call keyword arguments (line 164)
        kwargs_510294 = {}
        # Getting the type of 'np' (line 164)
        np_510287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 30), 'np', False)
        # Obtaining the member 'issubdtype' of a type (line 164)
        issubdtype_510288 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 30), np_510287, 'issubdtype')
        # Calling issubdtype(args, kwargs) (line 164)
        issubdtype_call_result_510295 = invoke(stypy.reporting.localization.Localization(__file__, 164, 30), issubdtype_510288, *[dtype_510291, complexfloating_510293], **kwargs_510294)
        
        # Getting the type of 'self' (line 164)
        self_510296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 12), 'self')
        # Setting the type of the member 'is_complex' of a type (line 164)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 12), self_510296, 'is_complex', issubdtype_call_result_510295)
        # SSA branch for the else part of an if statement (line 163)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Attribute (line 166):
        
        # Assigning a Call to a Attribute (line 166):
        
        # Call to any(...): (line 166)
        # Processing the call arguments (line 166)
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'self' (line 166)
        self_510303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 74), 'self', False)
        # Obtaining the member 'arg_spec' of a type (line 166)
        arg_spec_510304 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 74), self_510303, 'arg_spec')
        comprehension_510305 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 166, 35), arg_spec_510304)
        # Assigning a type to the variable 'arg' (line 166)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 35), 'arg', comprehension_510305)
        
        # Call to isinstance(...): (line 166)
        # Processing the call arguments (line 166)
        # Getting the type of 'arg' (line 166)
        arg_510299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 46), 'arg', False)
        # Getting the type of 'ComplexArg' (line 166)
        ComplexArg_510300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 51), 'ComplexArg', False)
        # Processing the call keyword arguments (line 166)
        kwargs_510301 = {}
        # Getting the type of 'isinstance' (line 166)
        isinstance_510298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 35), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 166)
        isinstance_call_result_510302 = invoke(stypy.reporting.localization.Localization(__file__, 166, 35), isinstance_510298, *[arg_510299, ComplexArg_510300], **kwargs_510301)
        
        list_510306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 35), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 166, 35), list_510306, isinstance_call_result_510302)
        # Processing the call keyword arguments (line 166)
        kwargs_510307 = {}
        # Getting the type of 'any' (line 166)
        any_510297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 30), 'any', False)
        # Calling any(args, kwargs) (line 166)
        any_call_result_510308 = invoke(stypy.reporting.localization.Localization(__file__, 166, 30), any_510297, *[list_510306], **kwargs_510307)
        
        # Getting the type of 'self' (line 166)
        self_510309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 12), 'self')
        # Setting the type of the member 'is_complex' of a type (line 166)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 12), self_510309, 'is_complex', any_call_result_510308)
        # SSA join for if statement (line 163)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 167):
        
        # Assigning a Name to a Attribute (line 167):
        # Getting the type of 'ignore_inf_sign' (line 167)
        ignore_inf_sign_510310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 31), 'ignore_inf_sign')
        # Getting the type of 'self' (line 167)
        self_510311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'self')
        # Setting the type of the member 'ignore_inf_sign' of a type (line 167)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 8), self_510311, 'ignore_inf_sign', ignore_inf_sign_510310)
        
        # Assigning a Name to a Attribute (line 168):
        
        # Assigning a Name to a Attribute (line 168):
        # Getting the type of 'distinguish_nan_and_inf' (line 168)
        distinguish_nan_and_inf_510312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 39), 'distinguish_nan_and_inf')
        # Getting the type of 'self' (line 168)
        self_510313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), 'self')
        # Setting the type of the member 'distinguish_nan_and_inf' of a type (line 168)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 8), self_510313, 'distinguish_nan_and_inf', distinguish_nan_and_inf_510312)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'name' (line 169)
        name_510314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 15), 'name')
        # Applying the 'not' unary operator (line 169)
        result_not__510315 = python_operator(stypy.reporting.localization.Localization(__file__, 169, 11), 'not', name_510314)
        
        
        # Getting the type of 'name' (line 169)
        name_510316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 23), 'name')
        str_510317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 31), 'str', '<lambda>')
        # Applying the binary operator '==' (line 169)
        result_eq_510318 = python_operator(stypy.reporting.localization.Localization(__file__, 169, 23), '==', name_510316, str_510317)
        
        # Applying the binary operator 'or' (line 169)
        result_or_keyword_510319 = python_operator(stypy.reporting.localization.Localization(__file__, 169, 11), 'or', result_not__510315, result_eq_510318)
        
        # Testing the type of an if condition (line 169)
        if_condition_510320 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 169, 8), result_or_keyword_510319)
        # Assigning a type to the variable 'if_condition_510320' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'if_condition_510320', if_condition_510320)
        # SSA begins for if statement (line 169)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 170):
        
        # Assigning a Call to a Name (line 170):
        
        # Call to getattr(...): (line 170)
        # Processing the call arguments (line 170)
        # Getting the type of 'scipy_func' (line 170)
        scipy_func_510322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 27), 'scipy_func', False)
        str_510323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 39), 'str', '__name__')
        # Getting the type of 'None' (line 170)
        None_510324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 51), 'None', False)
        # Processing the call keyword arguments (line 170)
        kwargs_510325 = {}
        # Getting the type of 'getattr' (line 170)
        getattr_510321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 19), 'getattr', False)
        # Calling getattr(args, kwargs) (line 170)
        getattr_call_result_510326 = invoke(stypy.reporting.localization.Localization(__file__, 170, 19), getattr_510321, *[scipy_func_510322, str_510323, None_510324], **kwargs_510325)
        
        # Assigning a type to the variable 'name' (line 170)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 12), 'name', getattr_call_result_510326)
        # SSA join for if statement (line 169)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'name' (line 171)
        name_510327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 15), 'name')
        # Applying the 'not' unary operator (line 171)
        result_not__510328 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 11), 'not', name_510327)
        
        
        # Getting the type of 'name' (line 171)
        name_510329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 23), 'name')
        str_510330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 31), 'str', '<lambda>')
        # Applying the binary operator '==' (line 171)
        result_eq_510331 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 23), '==', name_510329, str_510330)
        
        # Applying the binary operator 'or' (line 171)
        result_or_keyword_510332 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 11), 'or', result_not__510328, result_eq_510331)
        
        # Testing the type of an if condition (line 171)
        if_condition_510333 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 171, 8), result_or_keyword_510332)
        # Assigning a type to the variable 'if_condition_510333' (line 171)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 8), 'if_condition_510333', if_condition_510333)
        # SSA begins for if statement (line 171)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 172):
        
        # Assigning a Call to a Name (line 172):
        
        # Call to getattr(...): (line 172)
        # Processing the call arguments (line 172)
        # Getting the type of 'mpmath_func' (line 172)
        mpmath_func_510335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 27), 'mpmath_func', False)
        str_510336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 40), 'str', '__name__')
        # Getting the type of 'None' (line 172)
        None_510337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 52), 'None', False)
        # Processing the call keyword arguments (line 172)
        kwargs_510338 = {}
        # Getting the type of 'getattr' (line 172)
        getattr_510334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 19), 'getattr', False)
        # Calling getattr(args, kwargs) (line 172)
        getattr_call_result_510339 = invoke(stypy.reporting.localization.Localization(__file__, 172, 19), getattr_510334, *[mpmath_func_510335, str_510336, None_510337], **kwargs_510338)
        
        # Assigning a type to the variable 'name' (line 172)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 12), 'name', getattr_call_result_510339)
        # SSA join for if statement (line 171)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 173):
        
        # Assigning a Name to a Attribute (line 173):
        # Getting the type of 'name' (line 173)
        name_510340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 20), 'name')
        # Getting the type of 'self' (line 173)
        self_510341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 8), 'self')
        # Setting the type of the member 'name' of a type (line 173)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 8), self_510341, 'name', name_510340)
        
        # Assigning a Name to a Attribute (line 174):
        
        # Assigning a Name to a Attribute (line 174):
        # Getting the type of 'param_filter' (line 174)
        param_filter_510342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 28), 'param_filter')
        # Getting the type of 'self' (line 174)
        self_510343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'self')
        # Setting the type of the member 'param_filter' of a type (line 174)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 8), self_510343, 'param_filter', param_filter_510342)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def check(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'check'
        module_type_store = module_type_store.open_function_context('check', 176, 4, False)
        # Assigning a type to the variable 'self' (line 177)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MpmathData.check.__dict__.__setitem__('stypy_localization', localization)
        MpmathData.check.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MpmathData.check.__dict__.__setitem__('stypy_type_store', module_type_store)
        MpmathData.check.__dict__.__setitem__('stypy_function_name', 'MpmathData.check')
        MpmathData.check.__dict__.__setitem__('stypy_param_names_list', [])
        MpmathData.check.__dict__.__setitem__('stypy_varargs_param_name', None)
        MpmathData.check.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MpmathData.check.__dict__.__setitem__('stypy_call_defaults', defaults)
        MpmathData.check.__dict__.__setitem__('stypy_call_varargs', varargs)
        MpmathData.check.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MpmathData.check.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MpmathData.check', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'check', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'check(...)' code ##################

        
        # Call to seed(...): (line 177)
        # Processing the call arguments (line 177)
        int_510347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 23), 'int')
        # Processing the call keyword arguments (line 177)
        kwargs_510348 = {}
        # Getting the type of 'np' (line 177)
        np_510344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'np', False)
        # Obtaining the member 'random' of a type (line 177)
        random_510345 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 8), np_510344, 'random')
        # Obtaining the member 'seed' of a type (line 177)
        seed_510346 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 8), random_510345, 'seed')
        # Calling seed(args, kwargs) (line 177)
        seed_call_result_510349 = invoke(stypy.reporting.localization.Localization(__file__, 177, 8), seed_510346, *[int_510347], **kwargs_510348)
        
        
        # Assigning a Call to a Name (line 180):
        
        # Assigning a Call to a Name (line 180):
        
        # Call to get_args(...): (line 180)
        # Processing the call arguments (line 180)
        # Getting the type of 'self' (line 180)
        self_510351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 26), 'self', False)
        # Obtaining the member 'arg_spec' of a type (line 180)
        arg_spec_510352 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 26), self_510351, 'arg_spec')
        # Getting the type of 'self' (line 180)
        self_510353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 41), 'self', False)
        # Obtaining the member 'n' of a type (line 180)
        n_510354 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 41), self_510353, 'n')
        # Processing the call keyword arguments (line 180)
        kwargs_510355 = {}
        # Getting the type of 'get_args' (line 180)
        get_args_510350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 17), 'get_args', False)
        # Calling get_args(args, kwargs) (line 180)
        get_args_call_result_510356 = invoke(stypy.reporting.localization.Localization(__file__, 180, 17), get_args_510350, *[arg_spec_510352, n_510354], **kwargs_510355)
        
        # Assigning a type to the variable 'argarr' (line 180)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 8), 'argarr', get_args_call_result_510356)
        
        # Assigning a Tuple to a Tuple (line 183):
        
        # Assigning a Attribute to a Name (line 183):
        # Getting the type of 'mpmath' (line 183)
        mpmath_510357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 28), 'mpmath')
        # Obtaining the member 'mp' of a type (line 183)
        mp_510358 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 28), mpmath_510357, 'mp')
        # Obtaining the member 'dps' of a type (line 183)
        dps_510359 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 28), mp_510358, 'dps')
        # Assigning a type to the variable 'tuple_assignment_509626' (line 183)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 8), 'tuple_assignment_509626', dps_510359)
        
        # Assigning a Attribute to a Name (line 183):
        # Getting the type of 'mpmath' (line 183)
        mpmath_510360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 43), 'mpmath')
        # Obtaining the member 'mp' of a type (line 183)
        mp_510361 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 43), mpmath_510360, 'mp')
        # Obtaining the member 'prec' of a type (line 183)
        prec_510362 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 43), mp_510361, 'prec')
        # Assigning a type to the variable 'tuple_assignment_509627' (line 183)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 8), 'tuple_assignment_509627', prec_510362)
        
        # Assigning a Name to a Name (line 183):
        # Getting the type of 'tuple_assignment_509626' (line 183)
        tuple_assignment_509626_510363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 8), 'tuple_assignment_509626')
        # Assigning a type to the variable 'old_dps' (line 183)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 8), 'old_dps', tuple_assignment_509626_510363)
        
        # Assigning a Name to a Name (line 183):
        # Getting the type of 'tuple_assignment_509627' (line 183)
        tuple_assignment_509627_510364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 8), 'tuple_assignment_509627')
        # Assigning a type to the variable 'old_prec' (line 183)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 17), 'old_prec', tuple_assignment_509627_510364)
        
        # Try-finally block (line 184)
        
        
        # Getting the type of 'self' (line 185)
        self_510365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 15), 'self')
        # Obtaining the member 'dps' of a type (line 185)
        dps_510366 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 15), self_510365, 'dps')
        # Getting the type of 'None' (line 185)
        None_510367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 31), 'None')
        # Applying the binary operator 'isnot' (line 185)
        result_is_not_510368 = python_operator(stypy.reporting.localization.Localization(__file__, 185, 15), 'isnot', dps_510366, None_510367)
        
        # Testing the type of an if condition (line 185)
        if_condition_510369 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 185, 12), result_is_not_510368)
        # Assigning a type to the variable 'if_condition_510369' (line 185)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 12), 'if_condition_510369', if_condition_510369)
        # SSA begins for if statement (line 185)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a List to a Name (line 186):
        
        # Assigning a List to a Name (line 186):
        
        # Obtaining an instance of the builtin type 'list' (line 186)
        list_510370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 186)
        # Adding element type (line 186)
        # Getting the type of 'self' (line 186)
        self_510371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 28), 'self')
        # Obtaining the member 'dps' of a type (line 186)
        dps_510372 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 28), self_510371, 'dps')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 186, 27), list_510370, dps_510372)
        
        # Assigning a type to the variable 'dps_list' (line 186)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 16), 'dps_list', list_510370)
        # SSA branch for the else part of an if statement (line 185)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a List to a Name (line 188):
        
        # Assigning a List to a Name (line 188):
        
        # Obtaining an instance of the builtin type 'list' (line 188)
        list_510373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 188)
        # Adding element type (line 188)
        int_510374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 188, 27), list_510373, int_510374)
        
        # Assigning a type to the variable 'dps_list' (line 188)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 16), 'dps_list', list_510373)
        # SSA join for if statement (line 185)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'self' (line 189)
        self_510375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 15), 'self')
        # Obtaining the member 'prec' of a type (line 189)
        prec_510376 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 15), self_510375, 'prec')
        # Getting the type of 'None' (line 189)
        None_510377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 32), 'None')
        # Applying the binary operator 'isnot' (line 189)
        result_is_not_510378 = python_operator(stypy.reporting.localization.Localization(__file__, 189, 15), 'isnot', prec_510376, None_510377)
        
        # Testing the type of an if condition (line 189)
        if_condition_510379 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 189, 12), result_is_not_510378)
        # Assigning a type to the variable 'if_condition_510379' (line 189)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 12), 'if_condition_510379', if_condition_510379)
        # SSA begins for if statement (line 189)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Attribute (line 190):
        
        # Assigning a Attribute to a Attribute (line 190):
        # Getting the type of 'self' (line 190)
        self_510380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 33), 'self')
        # Obtaining the member 'prec' of a type (line 190)
        prec_510381 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 33), self_510380, 'prec')
        # Getting the type of 'mpmath' (line 190)
        mpmath_510382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 16), 'mpmath')
        # Obtaining the member 'mp' of a type (line 190)
        mp_510383 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 16), mpmath_510382, 'mp')
        # Setting the type of the member 'prec' of a type (line 190)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 16), mp_510383, 'prec', prec_510381)
        # SSA join for if statement (line 189)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to issubdtype(...): (line 195)
        # Processing the call arguments (line 195)
        # Getting the type of 'argarr' (line 195)
        argarr_510386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 29), 'argarr', False)
        # Obtaining the member 'dtype' of a type (line 195)
        dtype_510387 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 29), argarr_510386, 'dtype')
        # Getting the type of 'np' (line 195)
        np_510388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 43), 'np', False)
        # Obtaining the member 'complexfloating' of a type (line 195)
        complexfloating_510389 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 43), np_510388, 'complexfloating')
        # Processing the call keyword arguments (line 195)
        kwargs_510390 = {}
        # Getting the type of 'np' (line 195)
        np_510384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 15), 'np', False)
        # Obtaining the member 'issubdtype' of a type (line 195)
        issubdtype_510385 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 15), np_510384, 'issubdtype')
        # Calling issubdtype(args, kwargs) (line 195)
        issubdtype_call_result_510391 = invoke(stypy.reporting.localization.Localization(__file__, 195, 15), issubdtype_510385, *[dtype_510387, complexfloating_510389], **kwargs_510390)
        
        # Testing the type of an if condition (line 195)
        if_condition_510392 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 195, 12), issubdtype_call_result_510391)
        # Assigning a type to the variable 'if_condition_510392' (line 195)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 12), 'if_condition_510392', if_condition_510392)
        # SSA begins for if statement (line 195)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 196):
        
        # Assigning a Name to a Name (line 196):
        # Getting the type of 'mpc2complex' (line 196)
        mpc2complex_510393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 25), 'mpc2complex')
        # Assigning a type to the variable 'pytype' (line 196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 16), 'pytype', mpc2complex_510393)

        @norecursion
        def mptype(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'mptype'
            module_type_store = module_type_store.open_function_context('mptype', 198, 16, False)
            
            # Passed parameters checking function
            mptype.stypy_localization = localization
            mptype.stypy_type_of_self = None
            mptype.stypy_type_store = module_type_store
            mptype.stypy_function_name = 'mptype'
            mptype.stypy_param_names_list = ['x']
            mptype.stypy_varargs_param_name = None
            mptype.stypy_kwargs_param_name = None
            mptype.stypy_call_defaults = defaults
            mptype.stypy_call_varargs = varargs
            mptype.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'mptype', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'mptype', localization, ['x'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'mptype(...)' code ##################

            
            # Call to mpc(...): (line 199)
            # Processing the call arguments (line 199)
            
            # Call to complex(...): (line 199)
            # Processing the call arguments (line 199)
            # Getting the type of 'x' (line 199)
            x_510397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 46), 'x', False)
            # Processing the call keyword arguments (line 199)
            kwargs_510398 = {}
            # Getting the type of 'complex' (line 199)
            complex_510396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 38), 'complex', False)
            # Calling complex(args, kwargs) (line 199)
            complex_call_result_510399 = invoke(stypy.reporting.localization.Localization(__file__, 199, 38), complex_510396, *[x_510397], **kwargs_510398)
            
            # Processing the call keyword arguments (line 199)
            kwargs_510400 = {}
            # Getting the type of 'mpmath' (line 199)
            mpmath_510394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 27), 'mpmath', False)
            # Obtaining the member 'mpc' of a type (line 199)
            mpc_510395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 27), mpmath_510394, 'mpc')
            # Calling mpc(args, kwargs) (line 199)
            mpc_call_result_510401 = invoke(stypy.reporting.localization.Localization(__file__, 199, 27), mpc_510395, *[complex_call_result_510399], **kwargs_510400)
            
            # Assigning a type to the variable 'stypy_return_type' (line 199)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 20), 'stypy_return_type', mpc_call_result_510401)
            
            # ################# End of 'mptype(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'mptype' in the type store
            # Getting the type of 'stypy_return_type' (line 198)
            stypy_return_type_510402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 16), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_510402)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'mptype'
            return stypy_return_type_510402

        # Assigning a type to the variable 'mptype' (line 198)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 16), 'mptype', mptype)
        # SSA branch for the else part of an if statement (line 195)
        module_type_store.open_ssa_branch('else')

        @norecursion
        def mptype(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'mptype'
            module_type_store = module_type_store.open_function_context('mptype', 201, 16, False)
            
            # Passed parameters checking function
            mptype.stypy_localization = localization
            mptype.stypy_type_of_self = None
            mptype.stypy_type_store = module_type_store
            mptype.stypy_function_name = 'mptype'
            mptype.stypy_param_names_list = ['x']
            mptype.stypy_varargs_param_name = None
            mptype.stypy_kwargs_param_name = None
            mptype.stypy_call_defaults = defaults
            mptype.stypy_call_varargs = varargs
            mptype.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'mptype', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'mptype', localization, ['x'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'mptype(...)' code ##################

            
            # Call to mpf(...): (line 202)
            # Processing the call arguments (line 202)
            
            # Call to float(...): (line 202)
            # Processing the call arguments (line 202)
            # Getting the type of 'x' (line 202)
            x_510406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 44), 'x', False)
            # Processing the call keyword arguments (line 202)
            kwargs_510407 = {}
            # Getting the type of 'float' (line 202)
            float_510405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 38), 'float', False)
            # Calling float(args, kwargs) (line 202)
            float_call_result_510408 = invoke(stypy.reporting.localization.Localization(__file__, 202, 38), float_510405, *[x_510406], **kwargs_510407)
            
            # Processing the call keyword arguments (line 202)
            kwargs_510409 = {}
            # Getting the type of 'mpmath' (line 202)
            mpmath_510403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 27), 'mpmath', False)
            # Obtaining the member 'mpf' of a type (line 202)
            mpf_510404 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 27), mpmath_510403, 'mpf')
            # Calling mpf(args, kwargs) (line 202)
            mpf_call_result_510410 = invoke(stypy.reporting.localization.Localization(__file__, 202, 27), mpf_510404, *[float_call_result_510408], **kwargs_510409)
            
            # Assigning a type to the variable 'stypy_return_type' (line 202)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 20), 'stypy_return_type', mpf_call_result_510410)
            
            # ################# End of 'mptype(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'mptype' in the type store
            # Getting the type of 'stypy_return_type' (line 201)
            stypy_return_type_510411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 16), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_510411)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'mptype'
            return stypy_return_type_510411

        # Assigning a type to the variable 'mptype' (line 201)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 16), 'mptype', mptype)

        @norecursion
        def pytype(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'pytype'
            module_type_store = module_type_store.open_function_context('pytype', 204, 16, False)
            
            # Passed parameters checking function
            pytype.stypy_localization = localization
            pytype.stypy_type_of_self = None
            pytype.stypy_type_store = module_type_store
            pytype.stypy_function_name = 'pytype'
            pytype.stypy_param_names_list = ['x']
            pytype.stypy_varargs_param_name = None
            pytype.stypy_kwargs_param_name = None
            pytype.stypy_call_defaults = defaults
            pytype.stypy_call_varargs = varargs
            pytype.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'pytype', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'pytype', localization, ['x'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'pytype(...)' code ##################

            
            
            
            # Call to abs(...): (line 205)
            # Processing the call arguments (line 205)
            # Getting the type of 'x' (line 205)
            x_510413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 27), 'x', False)
            # Obtaining the member 'imag' of a type (line 205)
            imag_510414 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 27), x_510413, 'imag')
            # Processing the call keyword arguments (line 205)
            kwargs_510415 = {}
            # Getting the type of 'abs' (line 205)
            abs_510412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 23), 'abs', False)
            # Calling abs(args, kwargs) (line 205)
            abs_call_result_510416 = invoke(stypy.reporting.localization.Localization(__file__, 205, 23), abs_510412, *[imag_510414], **kwargs_510415)
            
            float_510417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 37), 'float')
            int_510418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 44), 'int')
            
            # Call to abs(...): (line 205)
            # Processing the call arguments (line 205)
            # Getting the type of 'x' (line 205)
            x_510420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 52), 'x', False)
            # Obtaining the member 'real' of a type (line 205)
            real_510421 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 52), x_510420, 'real')
            # Processing the call keyword arguments (line 205)
            kwargs_510422 = {}
            # Getting the type of 'abs' (line 205)
            abs_510419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 48), 'abs', False)
            # Calling abs(args, kwargs) (line 205)
            abs_call_result_510423 = invoke(stypy.reporting.localization.Localization(__file__, 205, 48), abs_510419, *[real_510421], **kwargs_510422)
            
            # Applying the binary operator '+' (line 205)
            result_add_510424 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 44), '+', int_510418, abs_call_result_510423)
            
            # Applying the binary operator '*' (line 205)
            result_mul_510425 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 37), '*', float_510417, result_add_510424)
            
            # Applying the binary operator '>' (line 205)
            result_gt_510426 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 23), '>', abs_call_result_510416, result_mul_510425)
            
            # Testing the type of an if condition (line 205)
            if_condition_510427 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 205, 20), result_gt_510426)
            # Assigning a type to the variable 'if_condition_510427' (line 205)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 20), 'if_condition_510427', if_condition_510427)
            # SSA begins for if statement (line 205)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'np' (line 206)
            np_510428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 31), 'np')
            # Obtaining the member 'nan' of a type (line 206)
            nan_510429 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 31), np_510428, 'nan')
            # Assigning a type to the variable 'stypy_return_type' (line 206)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 24), 'stypy_return_type', nan_510429)
            # SSA branch for the else part of an if statement (line 205)
            module_type_store.open_ssa_branch('else')
            
            # Call to mpf2float(...): (line 208)
            # Processing the call arguments (line 208)
            # Getting the type of 'x' (line 208)
            x_510431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 41), 'x', False)
            # Obtaining the member 'real' of a type (line 208)
            real_510432 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 41), x_510431, 'real')
            # Processing the call keyword arguments (line 208)
            kwargs_510433 = {}
            # Getting the type of 'mpf2float' (line 208)
            mpf2float_510430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 31), 'mpf2float', False)
            # Calling mpf2float(args, kwargs) (line 208)
            mpf2float_call_result_510434 = invoke(stypy.reporting.localization.Localization(__file__, 208, 31), mpf2float_510430, *[real_510432], **kwargs_510433)
            
            # Assigning a type to the variable 'stypy_return_type' (line 208)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 24), 'stypy_return_type', mpf2float_call_result_510434)
            # SSA join for if statement (line 205)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # ################# End of 'pytype(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'pytype' in the type store
            # Getting the type of 'stypy_return_type' (line 204)
            stypy_return_type_510435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 16), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_510435)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'pytype'
            return stypy_return_type_510435

        # Assigning a type to the variable 'pytype' (line 204)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 16), 'pytype', pytype)
        # SSA join for if statement (line 195)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to enumerate(...): (line 211)
        # Processing the call arguments (line 211)
        # Getting the type of 'dps_list' (line 211)
        dps_list_510437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 36), 'dps_list', False)
        # Processing the call keyword arguments (line 211)
        kwargs_510438 = {}
        # Getting the type of 'enumerate' (line 211)
        enumerate_510436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 26), 'enumerate', False)
        # Calling enumerate(args, kwargs) (line 211)
        enumerate_call_result_510439 = invoke(stypy.reporting.localization.Localization(__file__, 211, 26), enumerate_510436, *[dps_list_510437], **kwargs_510438)
        
        # Testing the type of a for loop iterable (line 211)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 211, 12), enumerate_call_result_510439)
        # Getting the type of the for loop variable (line 211)
        for_loop_var_510440 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 211, 12), enumerate_call_result_510439)
        # Assigning a type to the variable 'j' (line 211)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 12), 'j', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 211, 12), for_loop_var_510440))
        # Assigning a type to the variable 'dps' (line 211)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 12), 'dps', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 211, 12), for_loop_var_510440))
        # SSA begins for a for statement (line 211)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Name to a Attribute (line 212):
        
        # Assigning a Name to a Attribute (line 212):
        # Getting the type of 'dps' (line 212)
        dps_510441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 32), 'dps')
        # Getting the type of 'mpmath' (line 212)
        mpmath_510442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 16), 'mpmath')
        # Obtaining the member 'mp' of a type (line 212)
        mp_510443 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 16), mpmath_510442, 'mp')
        # Setting the type of the member 'dps' of a type (line 212)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 16), mp_510443, 'dps', dps_510441)
        
        
        # SSA begins for try-except statement (line 214)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to assert_func_equal(...): (line 215)
        # Processing the call arguments (line 215)
        # Getting the type of 'self' (line 215)
        self_510445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 38), 'self', False)
        # Obtaining the member 'scipy_func' of a type (line 215)
        scipy_func_510446 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 38), self_510445, 'scipy_func')

        @norecursion
        def _stypy_temp_lambda_309(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_309'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_309', 216, 38, True)
            # Passed parameters checking function
            _stypy_temp_lambda_309.stypy_localization = localization
            _stypy_temp_lambda_309.stypy_type_of_self = None
            _stypy_temp_lambda_309.stypy_type_store = module_type_store
            _stypy_temp_lambda_309.stypy_function_name = '_stypy_temp_lambda_309'
            _stypy_temp_lambda_309.stypy_param_names_list = []
            _stypy_temp_lambda_309.stypy_varargs_param_name = 'a'
            _stypy_temp_lambda_309.stypy_kwargs_param_name = None
            _stypy_temp_lambda_309.stypy_call_defaults = defaults
            _stypy_temp_lambda_309.stypy_call_varargs = varargs
            _stypy_temp_lambda_309.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_309', [], 'a', None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_309', [], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Call to pytype(...): (line 216)
            # Processing the call arguments (line 216)
            
            # Call to mpmath_func(...): (line 216)
            
            # Call to map(...): (line 216)
            # Processing the call arguments (line 216)
            # Getting the type of 'mptype' (line 216)
            mptype_510451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 78), 'mptype', False)
            # Getting the type of 'a' (line 216)
            a_510452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 86), 'a', False)
            # Processing the call keyword arguments (line 216)
            kwargs_510453 = {}
            # Getting the type of 'map' (line 216)
            map_510450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 74), 'map', False)
            # Calling map(args, kwargs) (line 216)
            map_call_result_510454 = invoke(stypy.reporting.localization.Localization(__file__, 216, 74), map_510450, *[mptype_510451, a_510452], **kwargs_510453)
            
            # Processing the call keyword arguments (line 216)
            kwargs_510455 = {}
            # Getting the type of 'self' (line 216)
            self_510448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 56), 'self', False)
            # Obtaining the member 'mpmath_func' of a type (line 216)
            mpmath_func_510449 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 56), self_510448, 'mpmath_func')
            # Calling mpmath_func(args, kwargs) (line 216)
            mpmath_func_call_result_510456 = invoke(stypy.reporting.localization.Localization(__file__, 216, 56), mpmath_func_510449, *[map_call_result_510454], **kwargs_510455)
            
            # Processing the call keyword arguments (line 216)
            kwargs_510457 = {}
            # Getting the type of 'pytype' (line 216)
            pytype_510447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 49), 'pytype', False)
            # Calling pytype(args, kwargs) (line 216)
            pytype_call_result_510458 = invoke(stypy.reporting.localization.Localization(__file__, 216, 49), pytype_510447, *[mpmath_func_call_result_510456], **kwargs_510457)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 216)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 38), 'stypy_return_type', pytype_call_result_510458)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_309' in the type store
            # Getting the type of 'stypy_return_type' (line 216)
            stypy_return_type_510459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 38), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_510459)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_309'
            return stypy_return_type_510459

        # Assigning a type to the variable '_stypy_temp_lambda_309' (line 216)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 38), '_stypy_temp_lambda_309', _stypy_temp_lambda_309)
        # Getting the type of '_stypy_temp_lambda_309' (line 216)
        _stypy_temp_lambda_309_510460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 38), '_stypy_temp_lambda_309')
        # Getting the type of 'argarr' (line 217)
        argarr_510461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 38), 'argarr', False)
        # Processing the call keyword arguments (line 215)
        # Getting the type of 'False' (line 218)
        False_510462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 49), 'False', False)
        keyword_510463 = False_510462
        # Getting the type of 'self' (line 219)
        self_510464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 43), 'self', False)
        # Obtaining the member 'rtol' of a type (line 219)
        rtol_510465 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 43), self_510464, 'rtol')
        keyword_510466 = rtol_510465
        # Getting the type of 'self' (line 219)
        self_510467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 59), 'self', False)
        # Obtaining the member 'atol' of a type (line 219)
        atol_510468 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 59), self_510467, 'atol')
        keyword_510469 = atol_510468
        # Getting the type of 'self' (line 220)
        self_510470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 54), 'self', False)
        # Obtaining the member 'ignore_inf_sign' of a type (line 220)
        ignore_inf_sign_510471 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 54), self_510470, 'ignore_inf_sign')
        keyword_510472 = ignore_inf_sign_510471
        # Getting the type of 'self' (line 221)
        self_510473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 62), 'self', False)
        # Obtaining the member 'distinguish_nan_and_inf' of a type (line 221)
        distinguish_nan_and_inf_510474 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 62), self_510473, 'distinguish_nan_and_inf')
        keyword_510475 = distinguish_nan_and_inf_510474
        # Getting the type of 'self' (line 222)
        self_510476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 45), 'self', False)
        # Obtaining the member 'nan_ok' of a type (line 222)
        nan_ok_510477 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 45), self_510476, 'nan_ok')
        keyword_510478 = nan_ok_510477
        # Getting the type of 'self' (line 223)
        self_510479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 51), 'self', False)
        # Obtaining the member 'param_filter' of a type (line 223)
        param_filter_510480 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 51), self_510479, 'param_filter')
        keyword_510481 = param_filter_510480
        kwargs_510482 = {'param_filter': keyword_510481, 'nan_ok': keyword_510478, 'vectorized': keyword_510463, 'distinguish_nan_and_inf': keyword_510475, 'rtol': keyword_510466, 'ignore_inf_sign': keyword_510472, 'atol': keyword_510469}
        # Getting the type of 'assert_func_equal' (line 215)
        assert_func_equal_510444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 20), 'assert_func_equal', False)
        # Calling assert_func_equal(args, kwargs) (line 215)
        assert_func_equal_call_result_510483 = invoke(stypy.reporting.localization.Localization(__file__, 215, 20), assert_func_equal_510444, *[scipy_func_510446, _stypy_temp_lambda_309_510460, argarr_510461], **kwargs_510482)
        
        # SSA branch for the except part of a try statement (line 214)
        # SSA branch for the except 'AssertionError' branch of a try statement (line 214)
        module_type_store.open_ssa_branch('except')
        
        
        # Getting the type of 'j' (line 226)
        j_510484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 23), 'j')
        
        # Call to len(...): (line 226)
        # Processing the call arguments (line 226)
        # Getting the type of 'dps_list' (line 226)
        dps_list_510486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 32), 'dps_list', False)
        # Processing the call keyword arguments (line 226)
        kwargs_510487 = {}
        # Getting the type of 'len' (line 226)
        len_510485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 28), 'len', False)
        # Calling len(args, kwargs) (line 226)
        len_call_result_510488 = invoke(stypy.reporting.localization.Localization(__file__, 226, 28), len_510485, *[dps_list_510486], **kwargs_510487)
        
        int_510489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 42), 'int')
        # Applying the binary operator '-' (line 226)
        result_sub_510490 = python_operator(stypy.reporting.localization.Localization(__file__, 226, 28), '-', len_call_result_510488, int_510489)
        
        # Applying the binary operator '>=' (line 226)
        result_ge_510491 = python_operator(stypy.reporting.localization.Localization(__file__, 226, 23), '>=', j_510484, result_sub_510490)
        
        # Testing the type of an if condition (line 226)
        if_condition_510492 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 226, 20), result_ge_510491)
        # Assigning a type to the variable 'if_condition_510492' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 20), 'if_condition_510492', if_condition_510492)
        # SSA begins for if statement (line 226)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to reraise(...): (line 227)
        
        # Call to exc_info(...): (line 227)
        # Processing the call keyword arguments (line 227)
        kwargs_510496 = {}
        # Getting the type of 'sys' (line 227)
        sys_510494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 33), 'sys', False)
        # Obtaining the member 'exc_info' of a type (line 227)
        exc_info_510495 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 33), sys_510494, 'exc_info')
        # Calling exc_info(args, kwargs) (line 227)
        exc_info_call_result_510497 = invoke(stypy.reporting.localization.Localization(__file__, 227, 33), exc_info_510495, *[], **kwargs_510496)
        
        # Processing the call keyword arguments (line 227)
        kwargs_510498 = {}
        # Getting the type of 'reraise' (line 227)
        reraise_510493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 24), 'reraise', False)
        # Calling reraise(args, kwargs) (line 227)
        reraise_call_result_510499 = invoke(stypy.reporting.localization.Localization(__file__, 227, 24), reraise_510493, *[exc_info_call_result_510497], **kwargs_510498)
        
        # SSA join for if statement (line 226)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for try-except statement (line 214)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # finally branch of the try-finally block (line 184)
        
        # Assigning a Tuple to a Tuple (line 229):
        
        # Assigning a Name to a Name (line 229):
        # Getting the type of 'old_dps' (line 229)
        old_dps_510500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 44), 'old_dps')
        # Assigning a type to the variable 'tuple_assignment_509628' (line 229)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 12), 'tuple_assignment_509628', old_dps_510500)
        
        # Assigning a Name to a Name (line 229):
        # Getting the type of 'old_prec' (line 229)
        old_prec_510501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 53), 'old_prec')
        # Assigning a type to the variable 'tuple_assignment_509629' (line 229)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 12), 'tuple_assignment_509629', old_prec_510501)
        
        # Assigning a Name to a Attribute (line 229):
        # Getting the type of 'tuple_assignment_509628' (line 229)
        tuple_assignment_509628_510502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 12), 'tuple_assignment_509628')
        # Getting the type of 'mpmath' (line 229)
        mpmath_510503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 12), 'mpmath')
        # Obtaining the member 'mp' of a type (line 229)
        mp_510504 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 12), mpmath_510503, 'mp')
        # Setting the type of the member 'dps' of a type (line 229)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 12), mp_510504, 'dps', tuple_assignment_509628_510502)
        
        # Assigning a Name to a Attribute (line 229):
        # Getting the type of 'tuple_assignment_509629' (line 229)
        tuple_assignment_509629_510505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 12), 'tuple_assignment_509629')
        # Getting the type of 'mpmath' (line 229)
        mpmath_510506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 27), 'mpmath')
        # Obtaining the member 'mp' of a type (line 229)
        mp_510507 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 27), mpmath_510506, 'mp')
        # Setting the type of the member 'prec' of a type (line 229)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 27), mp_510507, 'prec', tuple_assignment_509629_510505)
        
        
        # ################# End of 'check(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'check' in the type store
        # Getting the type of 'stypy_return_type' (line 176)
        stypy_return_type_510508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_510508)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'check'
        return stypy_return_type_510508


    @norecursion
    def stypy__repr__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__repr__'
        module_type_store = module_type_store.open_function_context('__repr__', 231, 4, False)
        # Assigning a type to the variable 'self' (line 232)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MpmathData.stypy__repr__.__dict__.__setitem__('stypy_localization', localization)
        MpmathData.stypy__repr__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MpmathData.stypy__repr__.__dict__.__setitem__('stypy_type_store', module_type_store)
        MpmathData.stypy__repr__.__dict__.__setitem__('stypy_function_name', 'MpmathData.stypy__repr__')
        MpmathData.stypy__repr__.__dict__.__setitem__('stypy_param_names_list', [])
        MpmathData.stypy__repr__.__dict__.__setitem__('stypy_varargs_param_name', None)
        MpmathData.stypy__repr__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MpmathData.stypy__repr__.__dict__.__setitem__('stypy_call_defaults', defaults)
        MpmathData.stypy__repr__.__dict__.__setitem__('stypy_call_varargs', varargs)
        MpmathData.stypy__repr__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MpmathData.stypy__repr__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MpmathData.stypy__repr__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__repr__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__repr__(...)' code ##################

        
        # Getting the type of 'self' (line 232)
        self_510509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 11), 'self')
        # Obtaining the member 'is_complex' of a type (line 232)
        is_complex_510510 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 11), self_510509, 'is_complex')
        # Testing the type of an if condition (line 232)
        if_condition_510511 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 232, 8), is_complex_510510)
        # Assigning a type to the variable 'if_condition_510511' (line 232)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 8), 'if_condition_510511', if_condition_510511)
        # SSA begins for if statement (line 232)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        str_510512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 19), 'str', '<MpmathData: %s (complex)>')
        
        # Obtaining an instance of the builtin type 'tuple' (line 233)
        tuple_510513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 51), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 233)
        # Adding element type (line 233)
        # Getting the type of 'self' (line 233)
        self_510514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 51), 'self')
        # Obtaining the member 'name' of a type (line 233)
        name_510515 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 51), self_510514, 'name')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 233, 51), tuple_510513, name_510515)
        
        # Applying the binary operator '%' (line 233)
        result_mod_510516 = python_operator(stypy.reporting.localization.Localization(__file__, 233, 19), '%', str_510512, tuple_510513)
        
        # Assigning a type to the variable 'stypy_return_type' (line 233)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 12), 'stypy_return_type', result_mod_510516)
        # SSA branch for the else part of an if statement (line 232)
        module_type_store.open_ssa_branch('else')
        str_510517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 19), 'str', '<MpmathData: %s>')
        
        # Obtaining an instance of the builtin type 'tuple' (line 235)
        tuple_510518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 41), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 235)
        # Adding element type (line 235)
        # Getting the type of 'self' (line 235)
        self_510519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 41), 'self')
        # Obtaining the member 'name' of a type (line 235)
        name_510520 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 41), self_510519, 'name')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 41), tuple_510518, name_510520)
        
        # Applying the binary operator '%' (line 235)
        result_mod_510521 = python_operator(stypy.reporting.localization.Localization(__file__, 235, 19), '%', str_510517, tuple_510518)
        
        # Assigning a type to the variable 'stypy_return_type' (line 235)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 12), 'stypy_return_type', result_mod_510521)
        # SSA join for if statement (line 232)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__repr__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__repr__' in the type store
        # Getting the type of 'stypy_return_type' (line 231)
        stypy_return_type_510522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_510522)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__repr__'
        return stypy_return_type_510522


# Assigning a type to the variable 'MpmathData' (line 136)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 0), 'MpmathData', MpmathData)

@norecursion
def assert_mpmath_equal(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'assert_mpmath_equal'
    module_type_store = module_type_store.open_function_context('assert_mpmath_equal', 238, 0, False)
    
    # Passed parameters checking function
    assert_mpmath_equal.stypy_localization = localization
    assert_mpmath_equal.stypy_type_of_self = None
    assert_mpmath_equal.stypy_type_store = module_type_store
    assert_mpmath_equal.stypy_function_name = 'assert_mpmath_equal'
    assert_mpmath_equal.stypy_param_names_list = []
    assert_mpmath_equal.stypy_varargs_param_name = 'a'
    assert_mpmath_equal.stypy_kwargs_param_name = 'kw'
    assert_mpmath_equal.stypy_call_defaults = defaults
    assert_mpmath_equal.stypy_call_varargs = varargs
    assert_mpmath_equal.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'assert_mpmath_equal', [], 'a', 'kw', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'assert_mpmath_equal', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'assert_mpmath_equal(...)' code ##################

    
    # Assigning a Call to a Name (line 239):
    
    # Assigning a Call to a Name (line 239):
    
    # Call to MpmathData(...): (line 239)
    # Getting the type of 'a' (line 239)
    a_510524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 20), 'a', False)
    # Processing the call keyword arguments (line 239)
    # Getting the type of 'kw' (line 239)
    kw_510525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 25), 'kw', False)
    kwargs_510526 = {'kw_510525': kw_510525}
    # Getting the type of 'MpmathData' (line 239)
    MpmathData_510523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 8), 'MpmathData', False)
    # Calling MpmathData(args, kwargs) (line 239)
    MpmathData_call_result_510527 = invoke(stypy.reporting.localization.Localization(__file__, 239, 8), MpmathData_510523, *[a_510524], **kwargs_510526)
    
    # Assigning a type to the variable 'd' (line 239)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 4), 'd', MpmathData_call_result_510527)
    
    # Call to check(...): (line 240)
    # Processing the call keyword arguments (line 240)
    kwargs_510530 = {}
    # Getting the type of 'd' (line 240)
    d_510528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 4), 'd', False)
    # Obtaining the member 'check' of a type (line 240)
    check_510529 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 4), d_510528, 'check')
    # Calling check(args, kwargs) (line 240)
    check_call_result_510531 = invoke(stypy.reporting.localization.Localization(__file__, 240, 4), check_510529, *[], **kwargs_510530)
    
    
    # ################# End of 'assert_mpmath_equal(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'assert_mpmath_equal' in the type store
    # Getting the type of 'stypy_return_type' (line 238)
    stypy_return_type_510532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_510532)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'assert_mpmath_equal'
    return stypy_return_type_510532

# Assigning a type to the variable 'assert_mpmath_equal' (line 238)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 0), 'assert_mpmath_equal', assert_mpmath_equal)

@norecursion
def nonfunctional_tooslow(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'nonfunctional_tooslow'
    module_type_store = module_type_store.open_function_context('nonfunctional_tooslow', 243, 0, False)
    
    # Passed parameters checking function
    nonfunctional_tooslow.stypy_localization = localization
    nonfunctional_tooslow.stypy_type_of_self = None
    nonfunctional_tooslow.stypy_type_store = module_type_store
    nonfunctional_tooslow.stypy_function_name = 'nonfunctional_tooslow'
    nonfunctional_tooslow.stypy_param_names_list = ['func']
    nonfunctional_tooslow.stypy_varargs_param_name = None
    nonfunctional_tooslow.stypy_kwargs_param_name = None
    nonfunctional_tooslow.stypy_call_defaults = defaults
    nonfunctional_tooslow.stypy_call_varargs = varargs
    nonfunctional_tooslow.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'nonfunctional_tooslow', ['func'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'nonfunctional_tooslow', localization, ['func'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'nonfunctional_tooslow(...)' code ##################

    
    # Call to (...): (line 244)
    # Processing the call arguments (line 244)
    # Getting the type of 'func' (line 244)
    func_510540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 95), 'func', False)
    # Processing the call keyword arguments (line 244)
    kwargs_510541 = {}
    
    # Call to skip(...): (line 244)
    # Processing the call keyword arguments (line 244)
    str_510536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 35), 'str', '    Test not yet functional (too slow), needs more work.')
    keyword_510537 = str_510536
    kwargs_510538 = {'reason': keyword_510537}
    # Getting the type of 'pytest' (line 244)
    pytest_510533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 11), 'pytest', False)
    # Obtaining the member 'mark' of a type (line 244)
    mark_510534 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 11), pytest_510533, 'mark')
    # Obtaining the member 'skip' of a type (line 244)
    skip_510535 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 11), mark_510534, 'skip')
    # Calling skip(args, kwargs) (line 244)
    skip_call_result_510539 = invoke(stypy.reporting.localization.Localization(__file__, 244, 11), skip_510535, *[], **kwargs_510538)
    
    # Calling (args, kwargs) (line 244)
    _call_result_510542 = invoke(stypy.reporting.localization.Localization(__file__, 244, 11), skip_call_result_510539, *[func_510540], **kwargs_510541)
    
    # Assigning a type to the variable 'stypy_return_type' (line 244)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 4), 'stypy_return_type', _call_result_510542)
    
    # ################# End of 'nonfunctional_tooslow(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'nonfunctional_tooslow' in the type store
    # Getting the type of 'stypy_return_type' (line 243)
    stypy_return_type_510543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_510543)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'nonfunctional_tooslow'
    return stypy_return_type_510543

# Assigning a type to the variable 'nonfunctional_tooslow' (line 243)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 0), 'nonfunctional_tooslow', nonfunctional_tooslow)

@norecursion
def mpf2float(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'mpf2float'
    module_type_store = module_type_store.open_function_context('mpf2float', 251, 0, False)
    
    # Passed parameters checking function
    mpf2float.stypy_localization = localization
    mpf2float.stypy_type_of_self = None
    mpf2float.stypy_type_store = module_type_store
    mpf2float.stypy_function_name = 'mpf2float'
    mpf2float.stypy_param_names_list = ['x']
    mpf2float.stypy_varargs_param_name = None
    mpf2float.stypy_kwargs_param_name = None
    mpf2float.stypy_call_defaults = defaults
    mpf2float.stypy_call_varargs = varargs
    mpf2float.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'mpf2float', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'mpf2float', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'mpf2float(...)' code ##################

    str_510544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, (-1)), 'str', '\n    Convert an mpf to the nearest floating point number. Just using\n    float directly doesn\'t work because of results like this:\n\n    with mp.workdps(50):\n        float(mpf("0.99999999999999999")) = 0.9999999999999999\n\n    ')
    
    # Call to float(...): (line 260)
    # Processing the call arguments (line 260)
    
    # Call to nstr(...): (line 260)
    # Processing the call arguments (line 260)
    # Getting the type of 'x' (line 260)
    x_510548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 29), 'x', False)
    int_510549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 32), 'int')
    # Processing the call keyword arguments (line 260)
    int_510550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 46), 'int')
    keyword_510551 = int_510550
    int_510552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 59), 'int')
    keyword_510553 = int_510552
    kwargs_510554 = {'max_fixed': keyword_510553, 'min_fixed': keyword_510551}
    # Getting the type of 'mpmath' (line 260)
    mpmath_510546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 17), 'mpmath', False)
    # Obtaining the member 'nstr' of a type (line 260)
    nstr_510547 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 17), mpmath_510546, 'nstr')
    # Calling nstr(args, kwargs) (line 260)
    nstr_call_result_510555 = invoke(stypy.reporting.localization.Localization(__file__, 260, 17), nstr_510547, *[x_510548, int_510549], **kwargs_510554)
    
    # Processing the call keyword arguments (line 260)
    kwargs_510556 = {}
    # Getting the type of 'float' (line 260)
    float_510545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 11), 'float', False)
    # Calling float(args, kwargs) (line 260)
    float_call_result_510557 = invoke(stypy.reporting.localization.Localization(__file__, 260, 11), float_510545, *[nstr_call_result_510555], **kwargs_510556)
    
    # Assigning a type to the variable 'stypy_return_type' (line 260)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 4), 'stypy_return_type', float_call_result_510557)
    
    # ################# End of 'mpf2float(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'mpf2float' in the type store
    # Getting the type of 'stypy_return_type' (line 251)
    stypy_return_type_510558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_510558)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'mpf2float'
    return stypy_return_type_510558

# Assigning a type to the variable 'mpf2float' (line 251)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 0), 'mpf2float', mpf2float)

@norecursion
def mpc2complex(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'mpc2complex'
    module_type_store = module_type_store.open_function_context('mpc2complex', 263, 0, False)
    
    # Passed parameters checking function
    mpc2complex.stypy_localization = localization
    mpc2complex.stypy_type_of_self = None
    mpc2complex.stypy_type_store = module_type_store
    mpc2complex.stypy_function_name = 'mpc2complex'
    mpc2complex.stypy_param_names_list = ['x']
    mpc2complex.stypy_varargs_param_name = None
    mpc2complex.stypy_kwargs_param_name = None
    mpc2complex.stypy_call_defaults = defaults
    mpc2complex.stypy_call_varargs = varargs
    mpc2complex.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'mpc2complex', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'mpc2complex', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'mpc2complex(...)' code ##################

    
    # Call to complex(...): (line 264)
    # Processing the call arguments (line 264)
    
    # Call to mpf2float(...): (line 264)
    # Processing the call arguments (line 264)
    # Getting the type of 'x' (line 264)
    x_510561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 29), 'x', False)
    # Obtaining the member 'real' of a type (line 264)
    real_510562 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 29), x_510561, 'real')
    # Processing the call keyword arguments (line 264)
    kwargs_510563 = {}
    # Getting the type of 'mpf2float' (line 264)
    mpf2float_510560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 19), 'mpf2float', False)
    # Calling mpf2float(args, kwargs) (line 264)
    mpf2float_call_result_510564 = invoke(stypy.reporting.localization.Localization(__file__, 264, 19), mpf2float_510560, *[real_510562], **kwargs_510563)
    
    
    # Call to mpf2float(...): (line 264)
    # Processing the call arguments (line 264)
    # Getting the type of 'x' (line 264)
    x_510566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 48), 'x', False)
    # Obtaining the member 'imag' of a type (line 264)
    imag_510567 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 48), x_510566, 'imag')
    # Processing the call keyword arguments (line 264)
    kwargs_510568 = {}
    # Getting the type of 'mpf2float' (line 264)
    mpf2float_510565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 38), 'mpf2float', False)
    # Calling mpf2float(args, kwargs) (line 264)
    mpf2float_call_result_510569 = invoke(stypy.reporting.localization.Localization(__file__, 264, 38), mpf2float_510565, *[imag_510567], **kwargs_510568)
    
    # Processing the call keyword arguments (line 264)
    kwargs_510570 = {}
    # Getting the type of 'complex' (line 264)
    complex_510559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 11), 'complex', False)
    # Calling complex(args, kwargs) (line 264)
    complex_call_result_510571 = invoke(stypy.reporting.localization.Localization(__file__, 264, 11), complex_510559, *[mpf2float_call_result_510564, mpf2float_call_result_510569], **kwargs_510570)
    
    # Assigning a type to the variable 'stypy_return_type' (line 264)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 4), 'stypy_return_type', complex_call_result_510571)
    
    # ################# End of 'mpc2complex(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'mpc2complex' in the type store
    # Getting the type of 'stypy_return_type' (line 263)
    stypy_return_type_510572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_510572)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'mpc2complex'
    return stypy_return_type_510572

# Assigning a type to the variable 'mpc2complex' (line 263)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 0), 'mpc2complex', mpc2complex)

@norecursion
def trace_args(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'trace_args'
    module_type_store = module_type_store.open_function_context('trace_args', 267, 0, False)
    
    # Passed parameters checking function
    trace_args.stypy_localization = localization
    trace_args.stypy_type_of_self = None
    trace_args.stypy_type_store = module_type_store
    trace_args.stypy_function_name = 'trace_args'
    trace_args.stypy_param_names_list = ['func']
    trace_args.stypy_varargs_param_name = None
    trace_args.stypy_kwargs_param_name = None
    trace_args.stypy_call_defaults = defaults
    trace_args.stypy_call_varargs = varargs
    trace_args.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'trace_args', ['func'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'trace_args', localization, ['func'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'trace_args(...)' code ##################


    @norecursion
    def tofloat(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'tofloat'
        module_type_store = module_type_store.open_function_context('tofloat', 268, 4, False)
        
        # Passed parameters checking function
        tofloat.stypy_localization = localization
        tofloat.stypy_type_of_self = None
        tofloat.stypy_type_store = module_type_store
        tofloat.stypy_function_name = 'tofloat'
        tofloat.stypy_param_names_list = ['x']
        tofloat.stypy_varargs_param_name = None
        tofloat.stypy_kwargs_param_name = None
        tofloat.stypy_call_defaults = defaults
        tofloat.stypy_call_varargs = varargs
        tofloat.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'tofloat', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'tofloat', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'tofloat(...)' code ##################

        
        
        # Call to isinstance(...): (line 269)
        # Processing the call arguments (line 269)
        # Getting the type of 'x' (line 269)
        x_510574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 22), 'x', False)
        # Getting the type of 'mpmath' (line 269)
        mpmath_510575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 25), 'mpmath', False)
        # Obtaining the member 'mpc' of a type (line 269)
        mpc_510576 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 25), mpmath_510575, 'mpc')
        # Processing the call keyword arguments (line 269)
        kwargs_510577 = {}
        # Getting the type of 'isinstance' (line 269)
        isinstance_510573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 269)
        isinstance_call_result_510578 = invoke(stypy.reporting.localization.Localization(__file__, 269, 11), isinstance_510573, *[x_510574, mpc_510576], **kwargs_510577)
        
        # Testing the type of an if condition (line 269)
        if_condition_510579 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 269, 8), isinstance_call_result_510578)
        # Assigning a type to the variable 'if_condition_510579' (line 269)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 8), 'if_condition_510579', if_condition_510579)
        # SSA begins for if statement (line 269)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to complex(...): (line 270)
        # Processing the call arguments (line 270)
        # Getting the type of 'x' (line 270)
        x_510581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 27), 'x', False)
        # Processing the call keyword arguments (line 270)
        kwargs_510582 = {}
        # Getting the type of 'complex' (line 270)
        complex_510580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 19), 'complex', False)
        # Calling complex(args, kwargs) (line 270)
        complex_call_result_510583 = invoke(stypy.reporting.localization.Localization(__file__, 270, 19), complex_510580, *[x_510581], **kwargs_510582)
        
        # Assigning a type to the variable 'stypy_return_type' (line 270)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 12), 'stypy_return_type', complex_call_result_510583)
        # SSA branch for the else part of an if statement (line 269)
        module_type_store.open_ssa_branch('else')
        
        # Call to float(...): (line 272)
        # Processing the call arguments (line 272)
        # Getting the type of 'x' (line 272)
        x_510585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 25), 'x', False)
        # Processing the call keyword arguments (line 272)
        kwargs_510586 = {}
        # Getting the type of 'float' (line 272)
        float_510584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 19), 'float', False)
        # Calling float(args, kwargs) (line 272)
        float_call_result_510587 = invoke(stypy.reporting.localization.Localization(__file__, 272, 19), float_510584, *[x_510585], **kwargs_510586)
        
        # Assigning a type to the variable 'stypy_return_type' (line 272)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 12), 'stypy_return_type', float_call_result_510587)
        # SSA join for if statement (line 269)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'tofloat(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'tofloat' in the type store
        # Getting the type of 'stypy_return_type' (line 268)
        stypy_return_type_510588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_510588)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'tofloat'
        return stypy_return_type_510588

    # Assigning a type to the variable 'tofloat' (line 268)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 4), 'tofloat', tofloat)

    @norecursion
    def wrap(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'wrap'
        module_type_store = module_type_store.open_function_context('wrap', 274, 4, False)
        
        # Passed parameters checking function
        wrap.stypy_localization = localization
        wrap.stypy_type_of_self = None
        wrap.stypy_type_store = module_type_store
        wrap.stypy_function_name = 'wrap'
        wrap.stypy_param_names_list = []
        wrap.stypy_varargs_param_name = 'a'
        wrap.stypy_kwargs_param_name = 'kw'
        wrap.stypy_call_defaults = defaults
        wrap.stypy_call_varargs = varargs
        wrap.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'wrap', [], 'a', 'kw', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'wrap', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'wrap(...)' code ##################

        
        # Call to write(...): (line 275)
        # Processing the call arguments (line 275)
        str_510592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 25), 'str', '%r: ')
        
        # Obtaining an instance of the builtin type 'tuple' (line 275)
        tuple_510593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 35), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 275)
        # Adding element type (line 275)
        
        # Call to tuple(...): (line 275)
        # Processing the call arguments (line 275)
        
        # Call to map(...): (line 275)
        # Processing the call arguments (line 275)
        # Getting the type of 'tofloat' (line 275)
        tofloat_510596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 45), 'tofloat', False)
        # Getting the type of 'a' (line 275)
        a_510597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 54), 'a', False)
        # Processing the call keyword arguments (line 275)
        kwargs_510598 = {}
        # Getting the type of 'map' (line 275)
        map_510595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 41), 'map', False)
        # Calling map(args, kwargs) (line 275)
        map_call_result_510599 = invoke(stypy.reporting.localization.Localization(__file__, 275, 41), map_510595, *[tofloat_510596, a_510597], **kwargs_510598)
        
        # Processing the call keyword arguments (line 275)
        kwargs_510600 = {}
        # Getting the type of 'tuple' (line 275)
        tuple_510594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 35), 'tuple', False)
        # Calling tuple(args, kwargs) (line 275)
        tuple_call_result_510601 = invoke(stypy.reporting.localization.Localization(__file__, 275, 35), tuple_510594, *[map_call_result_510599], **kwargs_510600)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 35), tuple_510593, tuple_call_result_510601)
        
        # Applying the binary operator '%' (line 275)
        result_mod_510602 = python_operator(stypy.reporting.localization.Localization(__file__, 275, 25), '%', str_510592, tuple_510593)
        
        # Processing the call keyword arguments (line 275)
        kwargs_510603 = {}
        # Getting the type of 'sys' (line 275)
        sys_510589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 8), 'sys', False)
        # Obtaining the member 'stderr' of a type (line 275)
        stderr_510590 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 8), sys_510589, 'stderr')
        # Obtaining the member 'write' of a type (line 275)
        write_510591 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 8), stderr_510590, 'write')
        # Calling write(args, kwargs) (line 275)
        write_call_result_510604 = invoke(stypy.reporting.localization.Localization(__file__, 275, 8), write_510591, *[result_mod_510602], **kwargs_510603)
        
        
        # Call to flush(...): (line 276)
        # Processing the call keyword arguments (line 276)
        kwargs_510608 = {}
        # Getting the type of 'sys' (line 276)
        sys_510605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 8), 'sys', False)
        # Obtaining the member 'stderr' of a type (line 276)
        stderr_510606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 8), sys_510605, 'stderr')
        # Obtaining the member 'flush' of a type (line 276)
        flush_510607 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 8), stderr_510606, 'flush')
        # Calling flush(args, kwargs) (line 276)
        flush_call_result_510609 = invoke(stypy.reporting.localization.Localization(__file__, 276, 8), flush_510607, *[], **kwargs_510608)
        
        
        # Try-finally block (line 277)
        
        # Assigning a Call to a Name (line 278):
        
        # Assigning a Call to a Name (line 278):
        
        # Call to func(...): (line 278)
        # Getting the type of 'a' (line 278)
        a_510611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 22), 'a', False)
        # Processing the call keyword arguments (line 278)
        # Getting the type of 'kw' (line 278)
        kw_510612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 27), 'kw', False)
        kwargs_510613 = {'kw_510612': kw_510612}
        # Getting the type of 'func' (line 278)
        func_510610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 16), 'func', False)
        # Calling func(args, kwargs) (line 278)
        func_call_result_510614 = invoke(stypy.reporting.localization.Localization(__file__, 278, 16), func_510610, *[a_510611], **kwargs_510613)
        
        # Assigning a type to the variable 'r' (line 278)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 12), 'r', func_call_result_510614)
        
        # Call to write(...): (line 279)
        # Processing the call arguments (line 279)
        str_510618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 29), 'str', '-> %r')
        # Getting the type of 'r' (line 279)
        r_510619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 39), 'r', False)
        # Applying the binary operator '%' (line 279)
        result_mod_510620 = python_operator(stypy.reporting.localization.Localization(__file__, 279, 29), '%', str_510618, r_510619)
        
        # Processing the call keyword arguments (line 279)
        kwargs_510621 = {}
        # Getting the type of 'sys' (line 279)
        sys_510615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 12), 'sys', False)
        # Obtaining the member 'stderr' of a type (line 279)
        stderr_510616 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 12), sys_510615, 'stderr')
        # Obtaining the member 'write' of a type (line 279)
        write_510617 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 12), stderr_510616, 'write')
        # Calling write(args, kwargs) (line 279)
        write_call_result_510622 = invoke(stypy.reporting.localization.Localization(__file__, 279, 12), write_510617, *[result_mod_510620], **kwargs_510621)
        
        
        # finally branch of the try-finally block (line 277)
        
        # Call to write(...): (line 281)
        # Processing the call arguments (line 281)
        str_510626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 29), 'str', '\n')
        # Processing the call keyword arguments (line 281)
        kwargs_510627 = {}
        # Getting the type of 'sys' (line 281)
        sys_510623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 12), 'sys', False)
        # Obtaining the member 'stderr' of a type (line 281)
        stderr_510624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 12), sys_510623, 'stderr')
        # Obtaining the member 'write' of a type (line 281)
        write_510625 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 12), stderr_510624, 'write')
        # Calling write(args, kwargs) (line 281)
        write_call_result_510628 = invoke(stypy.reporting.localization.Localization(__file__, 281, 12), write_510625, *[str_510626], **kwargs_510627)
        
        
        # Call to flush(...): (line 282)
        # Processing the call keyword arguments (line 282)
        kwargs_510632 = {}
        # Getting the type of 'sys' (line 282)
        sys_510629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 12), 'sys', False)
        # Obtaining the member 'stderr' of a type (line 282)
        stderr_510630 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 12), sys_510629, 'stderr')
        # Obtaining the member 'flush' of a type (line 282)
        flush_510631 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 12), stderr_510630, 'flush')
        # Calling flush(args, kwargs) (line 282)
        flush_call_result_510633 = invoke(stypy.reporting.localization.Localization(__file__, 282, 12), flush_510631, *[], **kwargs_510632)
        
        
        # Getting the type of 'r' (line 283)
        r_510634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 15), 'r')
        # Assigning a type to the variable 'stypy_return_type' (line 283)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 8), 'stypy_return_type', r_510634)
        
        # ################# End of 'wrap(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'wrap' in the type store
        # Getting the type of 'stypy_return_type' (line 274)
        stypy_return_type_510635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_510635)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'wrap'
        return stypy_return_type_510635

    # Assigning a type to the variable 'wrap' (line 274)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 4), 'wrap', wrap)
    # Getting the type of 'wrap' (line 284)
    wrap_510636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 11), 'wrap')
    # Assigning a type to the variable 'stypy_return_type' (line 284)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 4), 'stypy_return_type', wrap_510636)
    
    # ################# End of 'trace_args(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'trace_args' in the type store
    # Getting the type of 'stypy_return_type' (line 267)
    stypy_return_type_510637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_510637)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'trace_args'
    return stypy_return_type_510637

# Assigning a type to the variable 'trace_args' (line 267)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 0), 'trace_args', trace_args)


# SSA begins for try-except statement (line 286)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 287, 4))

# 'import posix' statement (line 287)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/')
import_510638 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 287, 4), 'posix')

if (type(import_510638) is not StypyTypeError):

    if (import_510638 != 'pyd_module'):
        __import__(import_510638)
        sys_modules_510639 = sys.modules[import_510638]
        import_module(stypy.reporting.localization.Localization(__file__, 287, 4), 'posix', sys_modules_510639.module_type_store, module_type_store)
    else:
        import posix

        import_module(stypy.reporting.localization.Localization(__file__, 287, 4), 'posix', posix, module_type_store)

else:
    # Assigning a type to the variable 'posix' (line 287)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 4), 'posix', import_510638)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 288, 4))

# 'import signal' statement (line 288)
import signal

import_module(stypy.reporting.localization.Localization(__file__, 288, 4), 'signal', signal, module_type_store)


# Assigning a Compare to a Name (line 289):

# Assigning a Compare to a Name (line 289):

str_510640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 13), 'str', 'setitimer')

# Call to dir(...): (line 289)
# Processing the call arguments (line 289)
# Getting the type of 'signal' (line 289)
signal_510642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 32), 'signal', False)
# Processing the call keyword arguments (line 289)
kwargs_510643 = {}
# Getting the type of 'dir' (line 289)
dir_510641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 28), 'dir', False)
# Calling dir(args, kwargs) (line 289)
dir_call_result_510644 = invoke(stypy.reporting.localization.Localization(__file__, 289, 28), dir_510641, *[signal_510642], **kwargs_510643)

# Applying the binary operator 'in' (line 289)
result_contains_510645 = python_operator(stypy.reporting.localization.Localization(__file__, 289, 13), 'in', str_510640, dir_call_result_510644)

# Assigning a type to the variable 'POSIX' (line 289)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 4), 'POSIX', result_contains_510645)
# SSA branch for the except part of a try statement (line 286)
# SSA branch for the except 'ImportError' branch of a try statement (line 286)
module_type_store.open_ssa_branch('except')

# Assigning a Name to a Name (line 291):

# Assigning a Name to a Name (line 291):
# Getting the type of 'False' (line 291)
False_510646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 12), 'False')
# Assigning a type to the variable 'POSIX' (line 291)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 4), 'POSIX', False_510646)
# SSA join for try-except statement (line 286)
module_type_store = module_type_store.join_ssa_context()

# Declaration of the 'TimeoutError' class
# Getting the type of 'Exception' (line 294)
Exception_510647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 19), 'Exception')

class TimeoutError(Exception_510647, ):
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 294, 0, False)
        # Assigning a type to the variable 'self' (line 295)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TimeoutError.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TimeoutError' (line 294)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 0), 'TimeoutError', TimeoutError)

@norecursion
def time_limited(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    float_510648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 25), 'float')
    # Getting the type of 'np' (line 298)
    np_510649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 41), 'np')
    # Obtaining the member 'nan' of a type (line 298)
    nan_510650 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 41), np_510649, 'nan')
    # Getting the type of 'True' (line 298)
    True_510651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 61), 'True')
    defaults = [float_510648, nan_510650, True_510651]
    # Create a new context for function 'time_limited'
    module_type_store = module_type_store.open_function_context('time_limited', 298, 0, False)
    
    # Passed parameters checking function
    time_limited.stypy_localization = localization
    time_limited.stypy_type_of_self = None
    time_limited.stypy_type_store = module_type_store
    time_limited.stypy_function_name = 'time_limited'
    time_limited.stypy_param_names_list = ['timeout', 'return_val', 'use_sigalrm']
    time_limited.stypy_varargs_param_name = None
    time_limited.stypy_kwargs_param_name = None
    time_limited.stypy_call_defaults = defaults
    time_limited.stypy_call_varargs = varargs
    time_limited.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'time_limited', ['timeout', 'return_val', 'use_sigalrm'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'time_limited', localization, ['timeout', 'return_val', 'use_sigalrm'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'time_limited(...)' code ##################

    str_510652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, (-1)), 'str', '\n    Decorator for setting a timeout for pure-Python functions.\n\n    If the function does not return within `timeout` seconds, the\n    value `return_val` is returned instead.\n\n    On POSIX this uses SIGALRM by default. On non-POSIX, settrace is\n    used. Do not use this with threads: the SIGALRM implementation\n    does probably not work well. The settrace implementation only\n    traces the current thread.\n\n    The settrace implementation slows down execution speed. Slowdown\n    by a factor around 10 is probably typical.\n    ')
    
    
    # Evaluating a boolean operation
    # Getting the type of 'POSIX' (line 313)
    POSIX_510653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 7), 'POSIX')
    # Getting the type of 'use_sigalrm' (line 313)
    use_sigalrm_510654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 17), 'use_sigalrm')
    # Applying the binary operator 'and' (line 313)
    result_and_keyword_510655 = python_operator(stypy.reporting.localization.Localization(__file__, 313, 7), 'and', POSIX_510653, use_sigalrm_510654)
    
    # Testing the type of an if condition (line 313)
    if_condition_510656 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 313, 4), result_and_keyword_510655)
    # Assigning a type to the variable 'if_condition_510656' (line 313)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 313, 4), 'if_condition_510656', if_condition_510656)
    # SSA begins for if statement (line 313)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

    @norecursion
    def sigalrm_handler(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'sigalrm_handler'
        module_type_store = module_type_store.open_function_context('sigalrm_handler', 314, 8, False)
        
        # Passed parameters checking function
        sigalrm_handler.stypy_localization = localization
        sigalrm_handler.stypy_type_of_self = None
        sigalrm_handler.stypy_type_store = module_type_store
        sigalrm_handler.stypy_function_name = 'sigalrm_handler'
        sigalrm_handler.stypy_param_names_list = ['signum', 'frame']
        sigalrm_handler.stypy_varargs_param_name = None
        sigalrm_handler.stypy_kwargs_param_name = None
        sigalrm_handler.stypy_call_defaults = defaults
        sigalrm_handler.stypy_call_varargs = varargs
        sigalrm_handler.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'sigalrm_handler', ['signum', 'frame'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'sigalrm_handler', localization, ['signum', 'frame'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'sigalrm_handler(...)' code ##################

        
        # Call to TimeoutError(...): (line 315)
        # Processing the call keyword arguments (line 315)
        kwargs_510658 = {}
        # Getting the type of 'TimeoutError' (line 315)
        TimeoutError_510657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 18), 'TimeoutError', False)
        # Calling TimeoutError(args, kwargs) (line 315)
        TimeoutError_call_result_510659 = invoke(stypy.reporting.localization.Localization(__file__, 315, 18), TimeoutError_510657, *[], **kwargs_510658)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 315, 12), TimeoutError_call_result_510659, 'raise parameter', BaseException)
        
        # ################# End of 'sigalrm_handler(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'sigalrm_handler' in the type store
        # Getting the type of 'stypy_return_type' (line 314)
        stypy_return_type_510660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 8), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_510660)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'sigalrm_handler'
        return stypy_return_type_510660

    # Assigning a type to the variable 'sigalrm_handler' (line 314)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 8), 'sigalrm_handler', sigalrm_handler)

    @norecursion
    def deco(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'deco'
        module_type_store = module_type_store.open_function_context('deco', 317, 8, False)
        
        # Passed parameters checking function
        deco.stypy_localization = localization
        deco.stypy_type_of_self = None
        deco.stypy_type_store = module_type_store
        deco.stypy_function_name = 'deco'
        deco.stypy_param_names_list = ['func']
        deco.stypy_varargs_param_name = None
        deco.stypy_kwargs_param_name = None
        deco.stypy_call_defaults = defaults
        deco.stypy_call_varargs = varargs
        deco.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'deco', ['func'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'deco', localization, ['func'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'deco(...)' code ##################


        @norecursion
        def wrap(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'wrap'
            module_type_store = module_type_store.open_function_context('wrap', 318, 12, False)
            
            # Passed parameters checking function
            wrap.stypy_localization = localization
            wrap.stypy_type_of_self = None
            wrap.stypy_type_store = module_type_store
            wrap.stypy_function_name = 'wrap'
            wrap.stypy_param_names_list = []
            wrap.stypy_varargs_param_name = 'a'
            wrap.stypy_kwargs_param_name = 'kw'
            wrap.stypy_call_defaults = defaults
            wrap.stypy_call_varargs = varargs
            wrap.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'wrap', [], 'a', 'kw', defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'wrap', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'wrap(...)' code ##################

            
            # Assigning a Call to a Name (line 319):
            
            # Assigning a Call to a Name (line 319):
            
            # Call to signal(...): (line 319)
            # Processing the call arguments (line 319)
            # Getting the type of 'signal' (line 319)
            signal_510663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 44), 'signal', False)
            # Obtaining the member 'SIGALRM' of a type (line 319)
            SIGALRM_510664 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 319, 44), signal_510663, 'SIGALRM')
            # Getting the type of 'sigalrm_handler' (line 319)
            sigalrm_handler_510665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 60), 'sigalrm_handler', False)
            # Processing the call keyword arguments (line 319)
            kwargs_510666 = {}
            # Getting the type of 'signal' (line 319)
            signal_510661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 30), 'signal', False)
            # Obtaining the member 'signal' of a type (line 319)
            signal_510662 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 319, 30), signal_510661, 'signal')
            # Calling signal(args, kwargs) (line 319)
            signal_call_result_510667 = invoke(stypy.reporting.localization.Localization(__file__, 319, 30), signal_510662, *[SIGALRM_510664, sigalrm_handler_510665], **kwargs_510666)
            
            # Assigning a type to the variable 'old_handler' (line 319)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 319, 16), 'old_handler', signal_call_result_510667)
            
            # Call to setitimer(...): (line 320)
            # Processing the call arguments (line 320)
            # Getting the type of 'signal' (line 320)
            signal_510670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 33), 'signal', False)
            # Obtaining the member 'ITIMER_REAL' of a type (line 320)
            ITIMER_REAL_510671 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 33), signal_510670, 'ITIMER_REAL')
            # Getting the type of 'timeout' (line 320)
            timeout_510672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 53), 'timeout', False)
            # Processing the call keyword arguments (line 320)
            kwargs_510673 = {}
            # Getting the type of 'signal' (line 320)
            signal_510668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 16), 'signal', False)
            # Obtaining the member 'setitimer' of a type (line 320)
            setitimer_510669 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 16), signal_510668, 'setitimer')
            # Calling setitimer(args, kwargs) (line 320)
            setitimer_call_result_510674 = invoke(stypy.reporting.localization.Localization(__file__, 320, 16), setitimer_510669, *[ITIMER_REAL_510671, timeout_510672], **kwargs_510673)
            
            
            # Try-finally block (line 321)
            
            
            # SSA begins for try-except statement (line 321)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
            
            # Call to func(...): (line 322)
            # Getting the type of 'a' (line 322)
            a_510676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 33), 'a', False)
            # Processing the call keyword arguments (line 322)
            # Getting the type of 'kw' (line 322)
            kw_510677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 38), 'kw', False)
            kwargs_510678 = {'kw_510677': kw_510677}
            # Getting the type of 'func' (line 322)
            func_510675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 27), 'func', False)
            # Calling func(args, kwargs) (line 322)
            func_call_result_510679 = invoke(stypy.reporting.localization.Localization(__file__, 322, 27), func_510675, *[a_510676], **kwargs_510678)
            
            # Assigning a type to the variable 'stypy_return_type' (line 322)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 322, 20), 'stypy_return_type', func_call_result_510679)
            # SSA branch for the except part of a try statement (line 321)
            # SSA branch for the except 'TimeoutError' branch of a try statement (line 321)
            module_type_store.open_ssa_branch('except')
            # Getting the type of 'return_val' (line 324)
            return_val_510680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 27), 'return_val')
            # Assigning a type to the variable 'stypy_return_type' (line 324)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 20), 'stypy_return_type', return_val_510680)
            # SSA join for try-except statement (line 321)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # finally branch of the try-finally block (line 321)
            
            # Call to setitimer(...): (line 326)
            # Processing the call arguments (line 326)
            # Getting the type of 'signal' (line 326)
            signal_510683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 37), 'signal', False)
            # Obtaining the member 'ITIMER_REAL' of a type (line 326)
            ITIMER_REAL_510684 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 37), signal_510683, 'ITIMER_REAL')
            int_510685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 57), 'int')
            # Processing the call keyword arguments (line 326)
            kwargs_510686 = {}
            # Getting the type of 'signal' (line 326)
            signal_510681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 20), 'signal', False)
            # Obtaining the member 'setitimer' of a type (line 326)
            setitimer_510682 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 20), signal_510681, 'setitimer')
            # Calling setitimer(args, kwargs) (line 326)
            setitimer_call_result_510687 = invoke(stypy.reporting.localization.Localization(__file__, 326, 20), setitimer_510682, *[ITIMER_REAL_510684, int_510685], **kwargs_510686)
            
            
            # Call to signal(...): (line 327)
            # Processing the call arguments (line 327)
            # Getting the type of 'signal' (line 327)
            signal_510690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 34), 'signal', False)
            # Obtaining the member 'SIGALRM' of a type (line 327)
            SIGALRM_510691 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 34), signal_510690, 'SIGALRM')
            # Getting the type of 'old_handler' (line 327)
            old_handler_510692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 50), 'old_handler', False)
            # Processing the call keyword arguments (line 327)
            kwargs_510693 = {}
            # Getting the type of 'signal' (line 327)
            signal_510688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 20), 'signal', False)
            # Obtaining the member 'signal' of a type (line 327)
            signal_510689 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 20), signal_510688, 'signal')
            # Calling signal(args, kwargs) (line 327)
            signal_call_result_510694 = invoke(stypy.reporting.localization.Localization(__file__, 327, 20), signal_510689, *[SIGALRM_510691, old_handler_510692], **kwargs_510693)
            
            
            
            # ################# End of 'wrap(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'wrap' in the type store
            # Getting the type of 'stypy_return_type' (line 318)
            stypy_return_type_510695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 12), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_510695)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'wrap'
            return stypy_return_type_510695

        # Assigning a type to the variable 'wrap' (line 318)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 12), 'wrap', wrap)
        # Getting the type of 'wrap' (line 328)
        wrap_510696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 19), 'wrap')
        # Assigning a type to the variable 'stypy_return_type' (line 328)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 12), 'stypy_return_type', wrap_510696)
        
        # ################# End of 'deco(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'deco' in the type store
        # Getting the type of 'stypy_return_type' (line 317)
        stypy_return_type_510697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 8), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_510697)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'deco'
        return stypy_return_type_510697

    # Assigning a type to the variable 'deco' (line 317)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 317, 8), 'deco', deco)
    # SSA branch for the else part of an if statement (line 313)
    module_type_store.open_ssa_branch('else')

    @norecursion
    def deco(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'deco'
        module_type_store = module_type_store.open_function_context('deco', 330, 8, False)
        
        # Passed parameters checking function
        deco.stypy_localization = localization
        deco.stypy_type_of_self = None
        deco.stypy_type_store = module_type_store
        deco.stypy_function_name = 'deco'
        deco.stypy_param_names_list = ['func']
        deco.stypy_varargs_param_name = None
        deco.stypy_kwargs_param_name = None
        deco.stypy_call_defaults = defaults
        deco.stypy_call_varargs = varargs
        deco.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'deco', ['func'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'deco', localization, ['func'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'deco(...)' code ##################


        @norecursion
        def wrap(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'wrap'
            module_type_store = module_type_store.open_function_context('wrap', 331, 12, False)
            
            # Passed parameters checking function
            wrap.stypy_localization = localization
            wrap.stypy_type_of_self = None
            wrap.stypy_type_store = module_type_store
            wrap.stypy_function_name = 'wrap'
            wrap.stypy_param_names_list = []
            wrap.stypy_varargs_param_name = 'a'
            wrap.stypy_kwargs_param_name = 'kw'
            wrap.stypy_call_defaults = defaults
            wrap.stypy_call_varargs = varargs
            wrap.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'wrap', [], 'a', 'kw', defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'wrap', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'wrap(...)' code ##################

            
            # Assigning a Call to a Name (line 332):
            
            # Assigning a Call to a Name (line 332):
            
            # Call to time(...): (line 332)
            # Processing the call keyword arguments (line 332)
            kwargs_510700 = {}
            # Getting the type of 'time' (line 332)
            time_510698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 29), 'time', False)
            # Obtaining the member 'time' of a type (line 332)
            time_510699 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 29), time_510698, 'time')
            # Calling time(args, kwargs) (line 332)
            time_call_result_510701 = invoke(stypy.reporting.localization.Localization(__file__, 332, 29), time_510699, *[], **kwargs_510700)
            
            # Assigning a type to the variable 'start_time' (line 332)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 16), 'start_time', time_call_result_510701)

            @norecursion
            def trace(localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'trace'
                module_type_store = module_type_store.open_function_context('trace', 334, 16, False)
                
                # Passed parameters checking function
                trace.stypy_localization = localization
                trace.stypy_type_of_self = None
                trace.stypy_type_store = module_type_store
                trace.stypy_function_name = 'trace'
                trace.stypy_param_names_list = ['frame', 'event', 'arg']
                trace.stypy_varargs_param_name = None
                trace.stypy_kwargs_param_name = None
                trace.stypy_call_defaults = defaults
                trace.stypy_call_varargs = varargs
                trace.stypy_call_kwargs = kwargs
                arguments = process_argument_values(localization, None, module_type_store, 'trace', ['frame', 'event', 'arg'], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'trace', localization, ['frame', 'event', 'arg'], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'trace(...)' code ##################

                
                
                
                # Call to time(...): (line 335)
                # Processing the call keyword arguments (line 335)
                kwargs_510704 = {}
                # Getting the type of 'time' (line 335)
                time_510702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 23), 'time', False)
                # Obtaining the member 'time' of a type (line 335)
                time_510703 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 23), time_510702, 'time')
                # Calling time(args, kwargs) (line 335)
                time_call_result_510705 = invoke(stypy.reporting.localization.Localization(__file__, 335, 23), time_510703, *[], **kwargs_510704)
                
                # Getting the type of 'start_time' (line 335)
                start_time_510706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 37), 'start_time')
                # Applying the binary operator '-' (line 335)
                result_sub_510707 = python_operator(stypy.reporting.localization.Localization(__file__, 335, 23), '-', time_call_result_510705, start_time_510706)
                
                # Getting the type of 'timeout' (line 335)
                timeout_510708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 50), 'timeout')
                # Applying the binary operator '>' (line 335)
                result_gt_510709 = python_operator(stypy.reporting.localization.Localization(__file__, 335, 23), '>', result_sub_510707, timeout_510708)
                
                # Testing the type of an if condition (line 335)
                if_condition_510710 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 335, 20), result_gt_510709)
                # Assigning a type to the variable 'if_condition_510710' (line 335)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 20), 'if_condition_510710', if_condition_510710)
                # SSA begins for if statement (line 335)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to TimeoutError(...): (line 336)
                # Processing the call keyword arguments (line 336)
                kwargs_510712 = {}
                # Getting the type of 'TimeoutError' (line 336)
                TimeoutError_510711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 30), 'TimeoutError', False)
                # Calling TimeoutError(args, kwargs) (line 336)
                TimeoutError_call_result_510713 = invoke(stypy.reporting.localization.Localization(__file__, 336, 30), TimeoutError_510711, *[], **kwargs_510712)
                
                ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 336, 24), TimeoutError_call_result_510713, 'raise parameter', BaseException)
                # SSA join for if statement (line 335)
                module_type_store = module_type_store.join_ssa_context()
                
                # Getting the type of 'trace' (line 337)
                trace_510714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 27), 'trace')
                # Assigning a type to the variable 'stypy_return_type' (line 337)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 337, 20), 'stypy_return_type', trace_510714)
                
                # ################# End of 'trace(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'trace' in the type store
                # Getting the type of 'stypy_return_type' (line 334)
                stypy_return_type_510715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 16), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_510715)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'trace'
                return stypy_return_type_510715

            # Assigning a type to the variable 'trace' (line 334)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 16), 'trace', trace)
            
            # Call to settrace(...): (line 338)
            # Processing the call arguments (line 338)
            # Getting the type of 'trace' (line 338)
            trace_510718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 29), 'trace', False)
            # Processing the call keyword arguments (line 338)
            kwargs_510719 = {}
            # Getting the type of 'sys' (line 338)
            sys_510716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 16), 'sys', False)
            # Obtaining the member 'settrace' of a type (line 338)
            settrace_510717 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 16), sys_510716, 'settrace')
            # Calling settrace(args, kwargs) (line 338)
            settrace_call_result_510720 = invoke(stypy.reporting.localization.Localization(__file__, 338, 16), settrace_510717, *[trace_510718], **kwargs_510719)
            
            
            # Try-finally block (line 339)
            
            
            # SSA begins for try-except statement (line 339)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
            
            # Call to func(...): (line 340)
            # Getting the type of 'a' (line 340)
            a_510722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 33), 'a', False)
            # Processing the call keyword arguments (line 340)
            # Getting the type of 'kw' (line 340)
            kw_510723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 38), 'kw', False)
            kwargs_510724 = {'kw_510723': kw_510723}
            # Getting the type of 'func' (line 340)
            func_510721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 27), 'func', False)
            # Calling func(args, kwargs) (line 340)
            func_call_result_510725 = invoke(stypy.reporting.localization.Localization(__file__, 340, 27), func_510721, *[a_510722], **kwargs_510724)
            
            # Assigning a type to the variable 'stypy_return_type' (line 340)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 20), 'stypy_return_type', func_call_result_510725)
            # SSA branch for the except part of a try statement (line 339)
            # SSA branch for the except 'TimeoutError' branch of a try statement (line 339)
            module_type_store.open_ssa_branch('except')
            
            # Call to settrace(...): (line 342)
            # Processing the call arguments (line 342)
            # Getting the type of 'None' (line 342)
            None_510728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 33), 'None', False)
            # Processing the call keyword arguments (line 342)
            kwargs_510729 = {}
            # Getting the type of 'sys' (line 342)
            sys_510726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 20), 'sys', False)
            # Obtaining the member 'settrace' of a type (line 342)
            settrace_510727 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 20), sys_510726, 'settrace')
            # Calling settrace(args, kwargs) (line 342)
            settrace_call_result_510730 = invoke(stypy.reporting.localization.Localization(__file__, 342, 20), settrace_510727, *[None_510728], **kwargs_510729)
            
            # Getting the type of 'return_val' (line 343)
            return_val_510731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 27), 'return_val')
            # Assigning a type to the variable 'stypy_return_type' (line 343)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 20), 'stypy_return_type', return_val_510731)
            # SSA join for try-except statement (line 339)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # finally branch of the try-finally block (line 339)
            
            # Call to settrace(...): (line 345)
            # Processing the call arguments (line 345)
            # Getting the type of 'None' (line 345)
            None_510734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 33), 'None', False)
            # Processing the call keyword arguments (line 345)
            kwargs_510735 = {}
            # Getting the type of 'sys' (line 345)
            sys_510732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 20), 'sys', False)
            # Obtaining the member 'settrace' of a type (line 345)
            settrace_510733 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 345, 20), sys_510732, 'settrace')
            # Calling settrace(args, kwargs) (line 345)
            settrace_call_result_510736 = invoke(stypy.reporting.localization.Localization(__file__, 345, 20), settrace_510733, *[None_510734], **kwargs_510735)
            
            
            
            # ################# End of 'wrap(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'wrap' in the type store
            # Getting the type of 'stypy_return_type' (line 331)
            stypy_return_type_510737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 12), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_510737)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'wrap'
            return stypy_return_type_510737

        # Assigning a type to the variable 'wrap' (line 331)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 12), 'wrap', wrap)
        # Getting the type of 'wrap' (line 346)
        wrap_510738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 19), 'wrap')
        # Assigning a type to the variable 'stypy_return_type' (line 346)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 346, 12), 'stypy_return_type', wrap_510738)
        
        # ################# End of 'deco(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'deco' in the type store
        # Getting the type of 'stypy_return_type' (line 330)
        stypy_return_type_510739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 8), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_510739)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'deco'
        return stypy_return_type_510739

    # Assigning a type to the variable 'deco' (line 330)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 8), 'deco', deco)
    # SSA join for if statement (line 313)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'deco' (line 347)
    deco_510740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 11), 'deco')
    # Assigning a type to the variable 'stypy_return_type' (line 347)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 347, 4), 'stypy_return_type', deco_510740)
    
    # ################# End of 'time_limited(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'time_limited' in the type store
    # Getting the type of 'stypy_return_type' (line 298)
    stypy_return_type_510741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_510741)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'time_limited'
    return stypy_return_type_510741

# Assigning a type to the variable 'time_limited' (line 298)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 0), 'time_limited', time_limited)

@norecursion
def exception_to_nan(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'exception_to_nan'
    module_type_store = module_type_store.open_function_context('exception_to_nan', 350, 0, False)
    
    # Passed parameters checking function
    exception_to_nan.stypy_localization = localization
    exception_to_nan.stypy_type_of_self = None
    exception_to_nan.stypy_type_store = module_type_store
    exception_to_nan.stypy_function_name = 'exception_to_nan'
    exception_to_nan.stypy_param_names_list = ['func']
    exception_to_nan.stypy_varargs_param_name = None
    exception_to_nan.stypy_kwargs_param_name = None
    exception_to_nan.stypy_call_defaults = defaults
    exception_to_nan.stypy_call_varargs = varargs
    exception_to_nan.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'exception_to_nan', ['func'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'exception_to_nan', localization, ['func'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'exception_to_nan(...)' code ##################

    str_510742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 351, 4), 'str', 'Decorate function to return nan if it raises an exception')

    @norecursion
    def wrap(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'wrap'
        module_type_store = module_type_store.open_function_context('wrap', 352, 4, False)
        
        # Passed parameters checking function
        wrap.stypy_localization = localization
        wrap.stypy_type_of_self = None
        wrap.stypy_type_store = module_type_store
        wrap.stypy_function_name = 'wrap'
        wrap.stypy_param_names_list = []
        wrap.stypy_varargs_param_name = 'a'
        wrap.stypy_kwargs_param_name = 'kw'
        wrap.stypy_call_defaults = defaults
        wrap.stypy_call_varargs = varargs
        wrap.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'wrap', [], 'a', 'kw', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'wrap', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'wrap(...)' code ##################

        
        
        # SSA begins for try-except statement (line 353)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to func(...): (line 354)
        # Getting the type of 'a' (line 354)
        a_510744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 25), 'a', False)
        # Processing the call keyword arguments (line 354)
        # Getting the type of 'kw' (line 354)
        kw_510745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 30), 'kw', False)
        kwargs_510746 = {'kw_510745': kw_510745}
        # Getting the type of 'func' (line 354)
        func_510743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 19), 'func', False)
        # Calling func(args, kwargs) (line 354)
        func_call_result_510747 = invoke(stypy.reporting.localization.Localization(__file__, 354, 19), func_510743, *[a_510744], **kwargs_510746)
        
        # Assigning a type to the variable 'stypy_return_type' (line 354)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 12), 'stypy_return_type', func_call_result_510747)
        # SSA branch for the except part of a try statement (line 353)
        # SSA branch for the except 'Exception' branch of a try statement (line 353)
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'np' (line 356)
        np_510748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 19), 'np')
        # Obtaining the member 'nan' of a type (line 356)
        nan_510749 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 19), np_510748, 'nan')
        # Assigning a type to the variable 'stypy_return_type' (line 356)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 356, 12), 'stypy_return_type', nan_510749)
        # SSA join for try-except statement (line 353)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'wrap(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'wrap' in the type store
        # Getting the type of 'stypy_return_type' (line 352)
        stypy_return_type_510750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_510750)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'wrap'
        return stypy_return_type_510750

    # Assigning a type to the variable 'wrap' (line 352)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 352, 4), 'wrap', wrap)
    # Getting the type of 'wrap' (line 357)
    wrap_510751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 11), 'wrap')
    # Assigning a type to the variable 'stypy_return_type' (line 357)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 357, 4), 'stypy_return_type', wrap_510751)
    
    # ################# End of 'exception_to_nan(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'exception_to_nan' in the type store
    # Getting the type of 'stypy_return_type' (line 350)
    stypy_return_type_510752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_510752)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'exception_to_nan'
    return stypy_return_type_510752

# Assigning a type to the variable 'exception_to_nan' (line 350)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 0), 'exception_to_nan', exception_to_nan)

@norecursion
def inf_to_nan(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'inf_to_nan'
    module_type_store = module_type_store.open_function_context('inf_to_nan', 360, 0, False)
    
    # Passed parameters checking function
    inf_to_nan.stypy_localization = localization
    inf_to_nan.stypy_type_of_self = None
    inf_to_nan.stypy_type_store = module_type_store
    inf_to_nan.stypy_function_name = 'inf_to_nan'
    inf_to_nan.stypy_param_names_list = ['func']
    inf_to_nan.stypy_varargs_param_name = None
    inf_to_nan.stypy_kwargs_param_name = None
    inf_to_nan.stypy_call_defaults = defaults
    inf_to_nan.stypy_call_varargs = varargs
    inf_to_nan.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'inf_to_nan', ['func'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'inf_to_nan', localization, ['func'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'inf_to_nan(...)' code ##################

    str_510753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 4), 'str', 'Decorate function to return nan if it returns inf')

    @norecursion
    def wrap(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'wrap'
        module_type_store = module_type_store.open_function_context('wrap', 362, 4, False)
        
        # Passed parameters checking function
        wrap.stypy_localization = localization
        wrap.stypy_type_of_self = None
        wrap.stypy_type_store = module_type_store
        wrap.stypy_function_name = 'wrap'
        wrap.stypy_param_names_list = []
        wrap.stypy_varargs_param_name = 'a'
        wrap.stypy_kwargs_param_name = 'kw'
        wrap.stypy_call_defaults = defaults
        wrap.stypy_call_varargs = varargs
        wrap.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'wrap', [], 'a', 'kw', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'wrap', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'wrap(...)' code ##################

        
        # Assigning a Call to a Name (line 363):
        
        # Assigning a Call to a Name (line 363):
        
        # Call to func(...): (line 363)
        # Getting the type of 'a' (line 363)
        a_510755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 18), 'a', False)
        # Processing the call keyword arguments (line 363)
        # Getting the type of 'kw' (line 363)
        kw_510756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 23), 'kw', False)
        kwargs_510757 = {'kw_510756': kw_510756}
        # Getting the type of 'func' (line 363)
        func_510754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 12), 'func', False)
        # Calling func(args, kwargs) (line 363)
        func_call_result_510758 = invoke(stypy.reporting.localization.Localization(__file__, 363, 12), func_510754, *[a_510755], **kwargs_510757)
        
        # Assigning a type to the variable 'v' (line 363)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 8), 'v', func_call_result_510758)
        
        
        
        # Call to isfinite(...): (line 364)
        # Processing the call arguments (line 364)
        # Getting the type of 'v' (line 364)
        v_510761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 27), 'v', False)
        # Processing the call keyword arguments (line 364)
        kwargs_510762 = {}
        # Getting the type of 'np' (line 364)
        np_510759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 15), 'np', False)
        # Obtaining the member 'isfinite' of a type (line 364)
        isfinite_510760 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 364, 15), np_510759, 'isfinite')
        # Calling isfinite(args, kwargs) (line 364)
        isfinite_call_result_510763 = invoke(stypy.reporting.localization.Localization(__file__, 364, 15), isfinite_510760, *[v_510761], **kwargs_510762)
        
        # Applying the 'not' unary operator (line 364)
        result_not__510764 = python_operator(stypy.reporting.localization.Localization(__file__, 364, 11), 'not', isfinite_call_result_510763)
        
        # Testing the type of an if condition (line 364)
        if_condition_510765 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 364, 8), result_not__510764)
        # Assigning a type to the variable 'if_condition_510765' (line 364)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 364, 8), 'if_condition_510765', if_condition_510765)
        # SSA begins for if statement (line 364)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'np' (line 365)
        np_510766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 19), 'np')
        # Obtaining the member 'nan' of a type (line 365)
        nan_510767 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 365, 19), np_510766, 'nan')
        # Assigning a type to the variable 'stypy_return_type' (line 365)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 365, 12), 'stypy_return_type', nan_510767)
        # SSA join for if statement (line 364)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'v' (line 366)
        v_510768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 15), 'v')
        # Assigning a type to the variable 'stypy_return_type' (line 366)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 366, 8), 'stypy_return_type', v_510768)
        
        # ################# End of 'wrap(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'wrap' in the type store
        # Getting the type of 'stypy_return_type' (line 362)
        stypy_return_type_510769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_510769)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'wrap'
        return stypy_return_type_510769

    # Assigning a type to the variable 'wrap' (line 362)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 4), 'wrap', wrap)
    # Getting the type of 'wrap' (line 367)
    wrap_510770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 11), 'wrap')
    # Assigning a type to the variable 'stypy_return_type' (line 367)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 367, 4), 'stypy_return_type', wrap_510770)
    
    # ################# End of 'inf_to_nan(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'inf_to_nan' in the type store
    # Getting the type of 'stypy_return_type' (line 360)
    stypy_return_type_510771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_510771)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'inf_to_nan'
    return stypy_return_type_510771

# Assigning a type to the variable 'inf_to_nan' (line 360)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 360, 0), 'inf_to_nan', inf_to_nan)

@norecursion
def mp_assert_allclose(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_510772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 38), 'int')
    float_510773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 46), 'float')
    defaults = [int_510772, float_510773]
    # Create a new context for function 'mp_assert_allclose'
    module_type_store = module_type_store.open_function_context('mp_assert_allclose', 370, 0, False)
    
    # Passed parameters checking function
    mp_assert_allclose.stypy_localization = localization
    mp_assert_allclose.stypy_type_of_self = None
    mp_assert_allclose.stypy_type_store = module_type_store
    mp_assert_allclose.stypy_function_name = 'mp_assert_allclose'
    mp_assert_allclose.stypy_param_names_list = ['res', 'std', 'atol', 'rtol']
    mp_assert_allclose.stypy_varargs_param_name = None
    mp_assert_allclose.stypy_kwargs_param_name = None
    mp_assert_allclose.stypy_call_defaults = defaults
    mp_assert_allclose.stypy_call_varargs = varargs
    mp_assert_allclose.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'mp_assert_allclose', ['res', 'std', 'atol', 'rtol'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'mp_assert_allclose', localization, ['res', 'std', 'atol', 'rtol'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'mp_assert_allclose(...)' code ##################

    str_510774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, (-1)), 'str', "\n    Compare lists of mpmath.mpf's or mpmath.mpc's directly so that it\n    can be done to higher precision than double.\n\n    ")
    
    
    # SSA begins for try-except statement (line 376)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Call to len(...): (line 377)
    # Processing the call arguments (line 377)
    # Getting the type of 'res' (line 377)
    res_510776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 12), 'res', False)
    # Processing the call keyword arguments (line 377)
    kwargs_510777 = {}
    # Getting the type of 'len' (line 377)
    len_510775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 8), 'len', False)
    # Calling len(args, kwargs) (line 377)
    len_call_result_510778 = invoke(stypy.reporting.localization.Localization(__file__, 377, 8), len_510775, *[res_510776], **kwargs_510777)
    
    # SSA branch for the except part of a try statement (line 376)
    # SSA branch for the except 'TypeError' branch of a try statement (line 376)
    module_type_store.open_ssa_branch('except')
    
    # Assigning a Call to a Name (line 379):
    
    # Assigning a Call to a Name (line 379):
    
    # Call to list(...): (line 379)
    # Processing the call arguments (line 379)
    # Getting the type of 'res' (line 379)
    res_510780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 19), 'res', False)
    # Processing the call keyword arguments (line 379)
    kwargs_510781 = {}
    # Getting the type of 'list' (line 379)
    list_510779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 14), 'list', False)
    # Calling list(args, kwargs) (line 379)
    list_call_result_510782 = invoke(stypy.reporting.localization.Localization(__file__, 379, 14), list_510779, *[res_510780], **kwargs_510781)
    
    # Assigning a type to the variable 'res' (line 379)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 379, 8), 'res', list_call_result_510782)
    # SSA join for try-except statement (line 376)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 381):
    
    # Assigning a Call to a Name (line 381):
    
    # Call to len(...): (line 381)
    # Processing the call arguments (line 381)
    # Getting the type of 'std' (line 381)
    std_510784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 12), 'std', False)
    # Processing the call keyword arguments (line 381)
    kwargs_510785 = {}
    # Getting the type of 'len' (line 381)
    len_510783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 8), 'len', False)
    # Calling len(args, kwargs) (line 381)
    len_call_result_510786 = invoke(stypy.reporting.localization.Localization(__file__, 381, 8), len_510783, *[std_510784], **kwargs_510785)
    
    # Assigning a type to the variable 'n' (line 381)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 381, 4), 'n', len_call_result_510786)
    
    
    
    # Call to len(...): (line 382)
    # Processing the call arguments (line 382)
    # Getting the type of 'res' (line 382)
    res_510788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 11), 'res', False)
    # Processing the call keyword arguments (line 382)
    kwargs_510789 = {}
    # Getting the type of 'len' (line 382)
    len_510787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 7), 'len', False)
    # Calling len(args, kwargs) (line 382)
    len_call_result_510790 = invoke(stypy.reporting.localization.Localization(__file__, 382, 7), len_510787, *[res_510788], **kwargs_510789)
    
    # Getting the type of 'n' (line 382)
    n_510791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 19), 'n')
    # Applying the binary operator '!=' (line 382)
    result_ne_510792 = python_operator(stypy.reporting.localization.Localization(__file__, 382, 7), '!=', len_call_result_510790, n_510791)
    
    # Testing the type of an if condition (line 382)
    if_condition_510793 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 382, 4), result_ne_510792)
    # Assigning a type to the variable 'if_condition_510793' (line 382)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 382, 4), 'if_condition_510793', if_condition_510793)
    # SSA begins for if statement (line 382)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to AssertionError(...): (line 383)
    # Processing the call arguments (line 383)
    str_510795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 29), 'str', 'Lengths of inputs not equal.')
    # Processing the call keyword arguments (line 383)
    kwargs_510796 = {}
    # Getting the type of 'AssertionError' (line 383)
    AssertionError_510794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 14), 'AssertionError', False)
    # Calling AssertionError(args, kwargs) (line 383)
    AssertionError_call_result_510797 = invoke(stypy.reporting.localization.Localization(__file__, 383, 14), AssertionError_510794, *[str_510795], **kwargs_510796)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 383, 8), AssertionError_call_result_510797, 'raise parameter', BaseException)
    # SSA join for if statement (line 382)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a List to a Name (line 385):
    
    # Assigning a List to a Name (line 385):
    
    # Obtaining an instance of the builtin type 'list' (line 385)
    list_510798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 385)
    
    # Assigning a type to the variable 'failures' (line 385)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 385, 4), 'failures', list_510798)
    
    
    # Call to range(...): (line 386)
    # Processing the call arguments (line 386)
    # Getting the type of 'n' (line 386)
    n_510800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 19), 'n', False)
    # Processing the call keyword arguments (line 386)
    kwargs_510801 = {}
    # Getting the type of 'range' (line 386)
    range_510799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 13), 'range', False)
    # Calling range(args, kwargs) (line 386)
    range_call_result_510802 = invoke(stypy.reporting.localization.Localization(__file__, 386, 13), range_510799, *[n_510800], **kwargs_510801)
    
    # Testing the type of a for loop iterable (line 386)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 386, 4), range_call_result_510802)
    # Getting the type of the for loop variable (line 386)
    for_loop_var_510803 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 386, 4), range_call_result_510802)
    # Assigning a type to the variable 'k' (line 386)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 386, 4), 'k', for_loop_var_510803)
    # SSA begins for a for statement (line 386)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # SSA begins for try-except statement (line 387)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Call to assert_(...): (line 388)
    # Processing the call arguments (line 388)
    
    
    # Call to fabs(...): (line 388)
    # Processing the call arguments (line 388)
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 388)
    k_510807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 36), 'k', False)
    # Getting the type of 'res' (line 388)
    res_510808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 32), 'res', False)
    # Obtaining the member '__getitem__' of a type (line 388)
    getitem___510809 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 32), res_510808, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 388)
    subscript_call_result_510810 = invoke(stypy.reporting.localization.Localization(__file__, 388, 32), getitem___510809, k_510807)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 388)
    k_510811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 45), 'k', False)
    # Getting the type of 'std' (line 388)
    std_510812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 41), 'std', False)
    # Obtaining the member '__getitem__' of a type (line 388)
    getitem___510813 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 41), std_510812, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 388)
    subscript_call_result_510814 = invoke(stypy.reporting.localization.Localization(__file__, 388, 41), getitem___510813, k_510811)
    
    # Applying the binary operator '-' (line 388)
    result_sub_510815 = python_operator(stypy.reporting.localization.Localization(__file__, 388, 32), '-', subscript_call_result_510810, subscript_call_result_510814)
    
    # Processing the call keyword arguments (line 388)
    kwargs_510816 = {}
    # Getting the type of 'mpmath' (line 388)
    mpmath_510805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 20), 'mpmath', False)
    # Obtaining the member 'fabs' of a type (line 388)
    fabs_510806 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 20), mpmath_510805, 'fabs')
    # Calling fabs(args, kwargs) (line 388)
    fabs_call_result_510817 = invoke(stypy.reporting.localization.Localization(__file__, 388, 20), fabs_510806, *[result_sub_510815], **kwargs_510816)
    
    # Getting the type of 'atol' (line 388)
    atol_510818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 52), 'atol', False)
    # Getting the type of 'rtol' (line 388)
    rtol_510819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 59), 'rtol', False)
    
    # Call to fabs(...): (line 388)
    # Processing the call arguments (line 388)
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 388)
    k_510822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 80), 'k', False)
    # Getting the type of 'std' (line 388)
    std_510823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 76), 'std', False)
    # Obtaining the member '__getitem__' of a type (line 388)
    getitem___510824 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 76), std_510823, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 388)
    subscript_call_result_510825 = invoke(stypy.reporting.localization.Localization(__file__, 388, 76), getitem___510824, k_510822)
    
    # Processing the call keyword arguments (line 388)
    kwargs_510826 = {}
    # Getting the type of 'mpmath' (line 388)
    mpmath_510820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 64), 'mpmath', False)
    # Obtaining the member 'fabs' of a type (line 388)
    fabs_510821 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 64), mpmath_510820, 'fabs')
    # Calling fabs(args, kwargs) (line 388)
    fabs_call_result_510827 = invoke(stypy.reporting.localization.Localization(__file__, 388, 64), fabs_510821, *[subscript_call_result_510825], **kwargs_510826)
    
    # Applying the binary operator '*' (line 388)
    result_mul_510828 = python_operator(stypy.reporting.localization.Localization(__file__, 388, 59), '*', rtol_510819, fabs_call_result_510827)
    
    # Applying the binary operator '+' (line 388)
    result_add_510829 = python_operator(stypy.reporting.localization.Localization(__file__, 388, 52), '+', atol_510818, result_mul_510828)
    
    # Applying the binary operator '<=' (line 388)
    result_le_510830 = python_operator(stypy.reporting.localization.Localization(__file__, 388, 20), '<=', fabs_call_result_510817, result_add_510829)
    
    # Processing the call keyword arguments (line 388)
    kwargs_510831 = {}
    # Getting the type of 'assert_' (line 388)
    assert__510804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 12), 'assert_', False)
    # Calling assert_(args, kwargs) (line 388)
    assert__call_result_510832 = invoke(stypy.reporting.localization.Localization(__file__, 388, 12), assert__510804, *[result_le_510830], **kwargs_510831)
    
    # SSA branch for the except part of a try statement (line 387)
    # SSA branch for the except 'AssertionError' branch of a try statement (line 387)
    module_type_store.open_ssa_branch('except')
    
    # Call to append(...): (line 390)
    # Processing the call arguments (line 390)
    # Getting the type of 'k' (line 390)
    k_510835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 28), 'k', False)
    # Processing the call keyword arguments (line 390)
    kwargs_510836 = {}
    # Getting the type of 'failures' (line 390)
    failures_510833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 12), 'failures', False)
    # Obtaining the member 'append' of a type (line 390)
    append_510834 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 12), failures_510833, 'append')
    # Calling append(args, kwargs) (line 390)
    append_call_result_510837 = invoke(stypy.reporting.localization.Localization(__file__, 390, 12), append_510834, *[k_510835], **kwargs_510836)
    
    # SSA join for try-except statement (line 387)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 392):
    
    # Assigning a Call to a Name (line 392):
    
    # Call to int(...): (line 392)
    # Processing the call arguments (line 392)
    
    # Call to abs(...): (line 392)
    # Processing the call arguments (line 392)
    
    # Call to log10(...): (line 392)
    # Processing the call arguments (line 392)
    # Getting the type of 'rtol' (line 392)
    rtol_510842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 31), 'rtol', False)
    # Processing the call keyword arguments (line 392)
    kwargs_510843 = {}
    # Getting the type of 'np' (line 392)
    np_510840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 22), 'np', False)
    # Obtaining the member 'log10' of a type (line 392)
    log10_510841 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 392, 22), np_510840, 'log10')
    # Calling log10(args, kwargs) (line 392)
    log10_call_result_510844 = invoke(stypy.reporting.localization.Localization(__file__, 392, 22), log10_510841, *[rtol_510842], **kwargs_510843)
    
    # Processing the call keyword arguments (line 392)
    kwargs_510845 = {}
    # Getting the type of 'abs' (line 392)
    abs_510839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 18), 'abs', False)
    # Calling abs(args, kwargs) (line 392)
    abs_call_result_510846 = invoke(stypy.reporting.localization.Localization(__file__, 392, 18), abs_510839, *[log10_call_result_510844], **kwargs_510845)
    
    # Processing the call keyword arguments (line 392)
    kwargs_510847 = {}
    # Getting the type of 'int' (line 392)
    int_510838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 14), 'int', False)
    # Calling int(args, kwargs) (line 392)
    int_call_result_510848 = invoke(stypy.reporting.localization.Localization(__file__, 392, 14), int_510838, *[abs_call_result_510846], **kwargs_510847)
    
    # Assigning a type to the variable 'ndigits' (line 392)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 392, 4), 'ndigits', int_call_result_510848)
    
    # Assigning a List to a Name (line 393):
    
    # Assigning a List to a Name (line 393):
    
    # Obtaining an instance of the builtin type 'list' (line 393)
    list_510849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 10), 'list')
    # Adding type elements to the builtin type 'list' instance (line 393)
    # Adding element type (line 393)
    str_510850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 11), 'str', '')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 393, 10), list_510849, str_510850)
    
    # Assigning a type to the variable 'msg' (line 393)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 4), 'msg', list_510849)
    
    # Call to append(...): (line 394)
    # Processing the call arguments (line 394)
    
    # Call to format(...): (line 394)
    # Processing the call arguments (line 394)
    
    # Call to len(...): (line 395)
    # Processing the call arguments (line 395)
    # Getting the type of 'failures' (line 395)
    failures_510856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 27), 'failures', False)
    # Processing the call keyword arguments (line 395)
    kwargs_510857 = {}
    # Getting the type of 'len' (line 395)
    len_510855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 23), 'len', False)
    # Calling len(args, kwargs) (line 395)
    len_call_result_510858 = invoke(stypy.reporting.localization.Localization(__file__, 395, 23), len_510855, *[failures_510856], **kwargs_510857)
    
    # Getting the type of 'n' (line 395)
    n_510859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 38), 'n', False)
    # Processing the call keyword arguments (line 394)
    kwargs_510860 = {}
    str_510853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 15), 'str', 'Bad results ({} out of {}) for the following points:')
    # Obtaining the member 'format' of a type (line 394)
    format_510854 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 15), str_510853, 'format')
    # Calling format(args, kwargs) (line 394)
    format_call_result_510861 = invoke(stypy.reporting.localization.Localization(__file__, 394, 15), format_510854, *[len_call_result_510858, n_510859], **kwargs_510860)
    
    # Processing the call keyword arguments (line 394)
    kwargs_510862 = {}
    # Getting the type of 'msg' (line 394)
    msg_510851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 4), 'msg', False)
    # Obtaining the member 'append' of a type (line 394)
    append_510852 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 4), msg_510851, 'append')
    # Calling append(args, kwargs) (line 394)
    append_call_result_510863 = invoke(stypy.reporting.localization.Localization(__file__, 394, 4), append_510852, *[format_call_result_510861], **kwargs_510862)
    
    
    # Getting the type of 'failures' (line 396)
    failures_510864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 13), 'failures')
    # Testing the type of a for loop iterable (line 396)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 396, 4), failures_510864)
    # Getting the type of the for loop variable (line 396)
    for_loop_var_510865 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 396, 4), failures_510864)
    # Assigning a type to the variable 'k' (line 396)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 396, 4), 'k', for_loop_var_510865)
    # SSA begins for a for statement (line 396)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 397):
    
    # Assigning a Call to a Name (line 397):
    
    # Call to nstr(...): (line 397)
    # Processing the call arguments (line 397)
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 397)
    k_510868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 33), 'k', False)
    # Getting the type of 'res' (line 397)
    res_510869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 29), 'res', False)
    # Obtaining the member '__getitem__' of a type (line 397)
    getitem___510870 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 397, 29), res_510869, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 397)
    subscript_call_result_510871 = invoke(stypy.reporting.localization.Localization(__file__, 397, 29), getitem___510870, k_510868)
    
    # Getting the type of 'ndigits' (line 397)
    ndigits_510872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 37), 'ndigits', False)
    # Processing the call keyword arguments (line 397)
    int_510873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 397, 56), 'int')
    keyword_510874 = int_510873
    int_510875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 397, 69), 'int')
    keyword_510876 = int_510875
    kwargs_510877 = {'max_fixed': keyword_510876, 'min_fixed': keyword_510874}
    # Getting the type of 'mpmath' (line 397)
    mpmath_510866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 17), 'mpmath', False)
    # Obtaining the member 'nstr' of a type (line 397)
    nstr_510867 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 397, 17), mpmath_510866, 'nstr')
    # Calling nstr(args, kwargs) (line 397)
    nstr_call_result_510878 = invoke(stypy.reporting.localization.Localization(__file__, 397, 17), nstr_510867, *[subscript_call_result_510871, ndigits_510872], **kwargs_510877)
    
    # Assigning a type to the variable 'resrep' (line 397)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 397, 8), 'resrep', nstr_call_result_510878)
    
    # Assigning a Call to a Name (line 398):
    
    # Assigning a Call to a Name (line 398):
    
    # Call to nstr(...): (line 398)
    # Processing the call arguments (line 398)
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 398)
    k_510881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 33), 'k', False)
    # Getting the type of 'std' (line 398)
    std_510882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 29), 'std', False)
    # Obtaining the member '__getitem__' of a type (line 398)
    getitem___510883 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 29), std_510882, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 398)
    subscript_call_result_510884 = invoke(stypy.reporting.localization.Localization(__file__, 398, 29), getitem___510883, k_510881)
    
    # Getting the type of 'ndigits' (line 398)
    ndigits_510885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 37), 'ndigits', False)
    # Processing the call keyword arguments (line 398)
    int_510886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 56), 'int')
    keyword_510887 = int_510886
    int_510888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 69), 'int')
    keyword_510889 = int_510888
    kwargs_510890 = {'max_fixed': keyword_510889, 'min_fixed': keyword_510887}
    # Getting the type of 'mpmath' (line 398)
    mpmath_510879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 17), 'mpmath', False)
    # Obtaining the member 'nstr' of a type (line 398)
    nstr_510880 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 17), mpmath_510879, 'nstr')
    # Calling nstr(args, kwargs) (line 398)
    nstr_call_result_510891 = invoke(stypy.reporting.localization.Localization(__file__, 398, 17), nstr_510880, *[subscript_call_result_510884, ndigits_510885], **kwargs_510890)
    
    # Assigning a type to the variable 'stdrep' (line 398)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 8), 'stdrep', nstr_call_result_510891)
    
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 399)
    k_510892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 15), 'k')
    # Getting the type of 'std' (line 399)
    std_510893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 11), 'std')
    # Obtaining the member '__getitem__' of a type (line 399)
    getitem___510894 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 11), std_510893, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 399)
    subscript_call_result_510895 = invoke(stypy.reporting.localization.Localization(__file__, 399, 11), getitem___510894, k_510892)
    
    int_510896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 21), 'int')
    # Applying the binary operator '==' (line 399)
    result_eq_510897 = python_operator(stypy.reporting.localization.Localization(__file__, 399, 11), '==', subscript_call_result_510895, int_510896)
    
    # Testing the type of an if condition (line 399)
    if_condition_510898 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 399, 8), result_eq_510897)
    # Assigning a type to the variable 'if_condition_510898' (line 399)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 399, 8), 'if_condition_510898', if_condition_510898)
    # SSA begins for if statement (line 399)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 400):
    
    # Assigning a Str to a Name (line 400):
    str_510899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 20), 'str', 'inf')
    # Assigning a type to the variable 'rdiff' (line 400)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 400, 12), 'rdiff', str_510899)
    # SSA branch for the else part of an if statement (line 399)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 402):
    
    # Assigning a Call to a Name (line 402):
    
    # Call to fabs(...): (line 402)
    # Processing the call arguments (line 402)
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 402)
    k_510902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 37), 'k', False)
    # Getting the type of 'res' (line 402)
    res_510903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 33), 'res', False)
    # Obtaining the member '__getitem__' of a type (line 402)
    getitem___510904 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 402, 33), res_510903, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 402)
    subscript_call_result_510905 = invoke(stypy.reporting.localization.Localization(__file__, 402, 33), getitem___510904, k_510902)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 402)
    k_510906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 46), 'k', False)
    # Getting the type of 'std' (line 402)
    std_510907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 42), 'std', False)
    # Obtaining the member '__getitem__' of a type (line 402)
    getitem___510908 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 402, 42), std_510907, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 402)
    subscript_call_result_510909 = invoke(stypy.reporting.localization.Localization(__file__, 402, 42), getitem___510908, k_510906)
    
    # Applying the binary operator '-' (line 402)
    result_sub_510910 = python_operator(stypy.reporting.localization.Localization(__file__, 402, 33), '-', subscript_call_result_510905, subscript_call_result_510909)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 402)
    k_510911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 54), 'k', False)
    # Getting the type of 'std' (line 402)
    std_510912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 50), 'std', False)
    # Obtaining the member '__getitem__' of a type (line 402)
    getitem___510913 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 402, 50), std_510912, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 402)
    subscript_call_result_510914 = invoke(stypy.reporting.localization.Localization(__file__, 402, 50), getitem___510913, k_510911)
    
    # Applying the binary operator 'div' (line 402)
    result_div_510915 = python_operator(stypy.reporting.localization.Localization(__file__, 402, 32), 'div', result_sub_510910, subscript_call_result_510914)
    
    # Processing the call keyword arguments (line 402)
    kwargs_510916 = {}
    # Getting the type of 'mpmath' (line 402)
    mpmath_510900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 20), 'mpmath', False)
    # Obtaining the member 'fabs' of a type (line 402)
    fabs_510901 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 402, 20), mpmath_510900, 'fabs')
    # Calling fabs(args, kwargs) (line 402)
    fabs_call_result_510917 = invoke(stypy.reporting.localization.Localization(__file__, 402, 20), fabs_510901, *[result_div_510915], **kwargs_510916)
    
    # Assigning a type to the variable 'rdiff' (line 402)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 402, 12), 'rdiff', fabs_call_result_510917)
    
    # Assigning a Call to a Name (line 403):
    
    # Assigning a Call to a Name (line 403):
    
    # Call to nstr(...): (line 403)
    # Processing the call arguments (line 403)
    # Getting the type of 'rdiff' (line 403)
    rdiff_510920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 32), 'rdiff', False)
    int_510921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 403, 39), 'int')
    # Processing the call keyword arguments (line 403)
    kwargs_510922 = {}
    # Getting the type of 'mpmath' (line 403)
    mpmath_510918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 20), 'mpmath', False)
    # Obtaining the member 'nstr' of a type (line 403)
    nstr_510919 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 403, 20), mpmath_510918, 'nstr')
    # Calling nstr(args, kwargs) (line 403)
    nstr_call_result_510923 = invoke(stypy.reporting.localization.Localization(__file__, 403, 20), nstr_510919, *[rdiff_510920, int_510921], **kwargs_510922)
    
    # Assigning a type to the variable 'rdiff' (line 403)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 403, 12), 'rdiff', nstr_call_result_510923)
    # SSA join for if statement (line 399)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to append(...): (line 404)
    # Processing the call arguments (line 404)
    
    # Call to format(...): (line 404)
    # Processing the call arguments (line 404)
    # Getting the type of 'k' (line 404)
    k_510928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 52), 'k', False)
    # Getting the type of 'resrep' (line 404)
    resrep_510929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 55), 'resrep', False)
    # Getting the type of 'stdrep' (line 404)
    stdrep_510930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 63), 'stdrep', False)
    # Getting the type of 'rdiff' (line 404)
    rdiff_510931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 71), 'rdiff', False)
    # Processing the call keyword arguments (line 404)
    kwargs_510932 = {}
    str_510926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 19), 'str', '{}: {} != {} (rdiff {})')
    # Obtaining the member 'format' of a type (line 404)
    format_510927 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 404, 19), str_510926, 'format')
    # Calling format(args, kwargs) (line 404)
    format_call_result_510933 = invoke(stypy.reporting.localization.Localization(__file__, 404, 19), format_510927, *[k_510928, resrep_510929, stdrep_510930, rdiff_510931], **kwargs_510932)
    
    # Processing the call keyword arguments (line 404)
    kwargs_510934 = {}
    # Getting the type of 'msg' (line 404)
    msg_510924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 8), 'msg', False)
    # Obtaining the member 'append' of a type (line 404)
    append_510925 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 404, 8), msg_510924, 'append')
    # Calling append(args, kwargs) (line 404)
    append_call_result_510935 = invoke(stypy.reporting.localization.Localization(__file__, 404, 8), append_510925, *[format_call_result_510933], **kwargs_510934)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'failures' (line 405)
    failures_510936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 7), 'failures')
    # Testing the type of an if condition (line 405)
    if_condition_510937 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 405, 4), failures_510936)
    # Assigning a type to the variable 'if_condition_510937' (line 405)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 405, 4), 'if_condition_510937', if_condition_510937)
    # SSA begins for if statement (line 405)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to assert_(...): (line 406)
    # Processing the call arguments (line 406)
    # Getting the type of 'False' (line 406)
    False_510939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 16), 'False', False)
    
    # Call to join(...): (line 406)
    # Processing the call arguments (line 406)
    # Getting the type of 'msg' (line 406)
    msg_510942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 33), 'msg', False)
    # Processing the call keyword arguments (line 406)
    kwargs_510943 = {}
    str_510940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 23), 'str', '\n')
    # Obtaining the member 'join' of a type (line 406)
    join_510941 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 23), str_510940, 'join')
    # Calling join(args, kwargs) (line 406)
    join_call_result_510944 = invoke(stypy.reporting.localization.Localization(__file__, 406, 23), join_510941, *[msg_510942], **kwargs_510943)
    
    # Processing the call keyword arguments (line 406)
    kwargs_510945 = {}
    # Getting the type of 'assert_' (line 406)
    assert__510938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 8), 'assert_', False)
    # Calling assert_(args, kwargs) (line 406)
    assert__call_result_510946 = invoke(stypy.reporting.localization.Localization(__file__, 406, 8), assert__510938, *[False_510939, join_call_result_510944], **kwargs_510945)
    
    # SSA join for if statement (line 405)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'mp_assert_allclose(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'mp_assert_allclose' in the type store
    # Getting the type of 'stypy_return_type' (line 370)
    stypy_return_type_510947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_510947)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'mp_assert_allclose'
    return stypy_return_type_510947

# Assigning a type to the variable 'mp_assert_allclose' (line 370)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 0), 'mp_assert_allclose', mp_assert_allclose)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
