
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Test cdflib functions versus mpmath, if available.
3: 
4: The following functions still need tests:
5: 
6: - ncfdtr
7: - ncfdtri
8: - ncfdtridfn
9: - ncfdtridfd
10: - ncfdtrinc
11: - nbdtrik
12: - nbdtrin
13: - nrdtrimn
14: - nrdtrisd
15: - pdtrik
16: - nctdtr
17: - nctdtrit
18: - nctdtridf
19: - nctdtrinc
20: 
21: '''
22: from __future__ import division, print_function, absolute_import
23: 
24: import itertools
25: 
26: import numpy as np
27: from numpy.testing import assert_equal
28: import pytest
29: 
30: import scipy.special as sp
31: from scipy._lib.six import with_metaclass
32: from scipy.special._testutils import (
33:     MissingModule, check_version, FuncData)
34: from scipy.special._mptestutils import (
35:     Arg, IntArg, get_args, mpf2float, assert_mpmath_equal)
36: 
37: try:
38:     import mpmath
39: except ImportError:
40:     mpmath = MissingModule('mpmath')
41: 
42: 
43: class ProbArg(object):
44:     '''Generate a set of probabilities on [0, 1].'''
45:     def __init__(self):
46:         # Include the endpoints for compatibility with Arg et. al.
47:         self.a = 0
48:         self.b = 1
49:     
50:     def values(self, n):
51:         '''Return an array containing approximatively n numbers.'''
52:         m = max(1, n//3)
53:         v1 = np.logspace(-30, np.log10(0.3), m)
54:         v2 = np.linspace(0.3, 0.7, m + 1, endpoint=False)[1:]
55:         v3 = 1 - np.logspace(np.log10(0.3), -15, m)
56:         v = np.r_[v1, v2, v3]
57:         return np.unique(v)
58: 
59: 
60: class EndpointFilter(object):
61:     def __init__(self, a, b, rtol, atol):
62:         self.a = a
63:         self.b = b
64:         self.rtol = rtol
65:         self.atol = atol
66: 
67:     def __call__(self, x):
68:         mask1 = np.abs(x - self.a) < self.rtol*np.abs(self.a) + self.atol
69:         mask2 = np.abs(x - self.b) < self.rtol*np.abs(self.b) + self.atol
70:         return np.where(mask1 | mask2, False, True)
71:             
72: 
73: class _CDFData(object):
74:     def __init__(self, spfunc, mpfunc, index, argspec, spfunc_first=True,
75:                  dps=20, n=5000, rtol=None, atol=None,
76:                  endpt_rtol=None, endpt_atol=None):
77:         self.spfunc = spfunc
78:         self.mpfunc = mpfunc
79:         self.index = index
80:         self.argspec = argspec
81:         self.spfunc_first = spfunc_first
82:         self.dps = dps
83:         self.n = n
84:         self.rtol = rtol
85:         self.atol = atol
86:         
87:         if not isinstance(argspec, list):
88:             self.endpt_rtol = None
89:             self.endpt_atol = None
90:         elif endpt_rtol is not None or endpt_atol is not None:
91:             if isinstance(endpt_rtol, list):
92:                 self.endpt_rtol = endpt_rtol
93:             else:
94:                 self.endpt_rtol = [endpt_rtol]*len(self.argspec)
95:             if isinstance(endpt_atol, list):
96:                 self.endpt_atol = endpt_atol
97:             else:
98:                 self.endpt_atol = [endpt_atol]*len(self.argspec)
99:         else:
100:             self.endpt_rtol = None
101:             self.endpt_atol = None
102: 
103:     def idmap(self, *args):
104:         if self.spfunc_first:
105:             res = self.spfunc(*args)
106:             if np.isnan(res):
107:                 return np.nan
108:             args = list(args)
109:             args[self.index] = res
110:             with mpmath.workdps(self.dps):
111:                 res = self.mpfunc(*tuple(args))
112:                 # Imaginary parts are spurious
113:                 res = mpf2float(res.real)
114:         else:
115:             with mpmath.workdps(self.dps):
116:                 res = self.mpfunc(*args)
117:                 res = mpf2float(res.real)
118:             args = list(args)
119:             args[self.index] = res
120:             res = self.spfunc(*tuple(args))
121:         return res
122: 
123:     def get_param_filter(self):
124:         if self.endpt_rtol is None and self.endpt_atol is None:
125:             return None
126:         
127:         filters = []
128:         for rtol, atol, spec in zip(self.endpt_rtol, self.endpt_atol, self.argspec):
129:             if rtol is None and atol is None:
130:                 filters.append(None)
131:                 continue
132:             elif rtol is None:
133:                 rtol = 0.0
134:             elif atol is None:
135:                 atol = 0.0
136: 
137:             filters.append(EndpointFilter(spec.a, spec.b, rtol, atol))
138:         return filters
139:         
140:     def check(self):
141:         # Generate values for the arguments
142:         args = get_args(self.argspec, self.n)
143:         param_filter = self.get_param_filter()
144:         param_columns = tuple(range(args.shape[1]))
145:         result_columns = args.shape[1]
146:         args = np.hstack((args, args[:,self.index].reshape(args.shape[0], 1)))
147:         FuncData(self.idmap, args,
148:                  param_columns=param_columns, result_columns=result_columns,
149:                  rtol=self.rtol, atol=self.atol, vectorized=False,
150:                  param_filter=param_filter).check()
151: 
152: 
153: def _assert_inverts(*a, **kw):
154:     d = _CDFData(*a, **kw)
155:     d.check()
156: 
157: 
158: def _binomial_cdf(k, n, p):
159:     k, n, p = mpmath.mpf(k), mpmath.mpf(n), mpmath.mpf(p)
160:     if k <= 0:
161:         return mpmath.mpf(0)
162:     elif k >= n:
163:         return mpmath.mpf(1)
164: 
165:     onemp = mpmath.fsub(1, p, exact=True)
166:     return mpmath.betainc(n - k, k + 1, x2=onemp, regularized=True)
167: 
168: 
169: def _f_cdf(dfn, dfd, x):
170:     if x < 0:
171:         return mpmath.mpf(0)
172:     dfn, dfd, x = mpmath.mpf(dfn), mpmath.mpf(dfd), mpmath.mpf(x)
173:     ub = dfn*x/(dfn*x + dfd)
174:     res = mpmath.betainc(dfn/2, dfd/2, x2=ub, regularized=True)
175:     return res
176:     
177: 
178: def _student_t_cdf(df, t, dps=None):
179:     if dps is None:
180:         dps = mpmath.mp.dps
181:     with mpmath.workdps(dps):
182:         df, t = mpmath.mpf(df), mpmath.mpf(t)
183:         fac = mpmath.hyp2f1(0.5, 0.5*(df + 1), 1.5, -t**2/df)
184:         fac *= t*mpmath.gamma(0.5*(df + 1))
185:         fac /= mpmath.sqrt(mpmath.pi*df)*mpmath.gamma(0.5*df)
186:         return 0.5 + fac
187: 
188: 
189: def _noncentral_chi_pdf(t, df, nc):
190:     res = mpmath.besseli(df/2 - 1, mpmath.sqrt(nc*t))
191:     res *= mpmath.exp(-(t + nc)/2)*(t/nc)**(df/4 - 1/2)/2
192:     return res
193: 
194: 
195: def _noncentral_chi_cdf(x, df, nc, dps=None):
196:     if dps is None:
197:         dps = mpmath.mp.dps
198:     x, df, nc = mpmath.mpf(x), mpmath.mpf(df), mpmath.mpf(nc)
199:     with mpmath.workdps(dps):
200:         res = mpmath.quad(lambda t: _noncentral_chi_pdf(t, df, nc), [0, x])
201:         return res
202: 
203: 
204: def _tukey_lmbda_quantile(p, lmbda):
205:     # For lmbda != 0
206:     return (p**lmbda - (1 - p)**lmbda)/lmbda
207: 
208: 
209: @pytest.mark.slow
210: @check_version(mpmath, '0.19')
211: class TestCDFlib(object):
212: 
213:     @pytest.mark.xfail(run=False)
214:     def test_bdtrik(self):
215:         _assert_inverts(
216:             sp.bdtrik,
217:             _binomial_cdf,
218:             0, [ProbArg(), IntArg(1, 1000), ProbArg()],
219:             rtol=1e-4)
220: 
221:     def test_bdtrin(self):
222:         _assert_inverts(
223:             sp.bdtrin,
224:             _binomial_cdf,
225:             1, [IntArg(1, 1000), ProbArg(), ProbArg()],
226:             rtol=1e-4, endpt_atol=[None, None, 1e-6])
227:     
228:     def test_btdtria(self):
229:         _assert_inverts(
230:             sp.btdtria,
231:             lambda a, b, x: mpmath.betainc(a, b, x2=x, regularized=True),
232:             0, [ProbArg(), Arg(0, 1e3, inclusive_a=False),
233:                 Arg(0, 1, inclusive_a=False, inclusive_b=False)],
234:             rtol=1e-6)
235: 
236:     def test_btdtrib(self):
237:         # Use small values of a or mpmath doesn't converge
238:         _assert_inverts(
239:             sp.btdtrib,
240:             lambda a, b, x: mpmath.betainc(a, b, x2=x, regularized=True),
241:             1, [Arg(0, 1e2, inclusive_a=False), ProbArg(),
242:              Arg(0, 1, inclusive_a=False, inclusive_b=False)],
243:             rtol=1e-7, endpt_atol=[None, 1e-20, 1e-20])
244: 
245:     @pytest.mark.xfail(run=False)
246:     def test_fdtridfd(self):
247:         _assert_inverts(
248:             sp.fdtridfd,
249:             _f_cdf,
250:             1, [IntArg(1, 100), ProbArg(), Arg(0, 100, inclusive_a=False)],
251:             rtol=1e-7)
252:         
253:     def test_gdtria(self):
254:         _assert_inverts(
255:             sp.gdtria,
256:             lambda a, b, x: mpmath.gammainc(b, b=a*x, regularized=True),
257:             0, [ProbArg(), Arg(0, 1e3, inclusive_a=False),
258:                 Arg(0, 1e4, inclusive_a=False)], rtol=1e-7,
259:             endpt_atol=[None, 1e-10, 1e-10])
260: 
261:     def test_gdtrib(self):
262:         # Use small values of a and x or mpmath doesn't converge
263:         _assert_inverts(
264:             sp.gdtrib,
265:             lambda a, b, x: mpmath.gammainc(b, b=a*x, regularized=True),
266:             1, [Arg(0, 1e2, inclusive_a=False), ProbArg(),
267:                 Arg(0, 1e3, inclusive_a=False)], rtol=1e-5)
268: 
269:     def test_gdtrix(self):
270:         _assert_inverts(
271:             sp.gdtrix,
272:             lambda a, b, x: mpmath.gammainc(b, b=a*x, regularized=True),
273:             2, [Arg(0, 1e3, inclusive_a=False), Arg(0, 1e3, inclusive_a=False),
274:                 ProbArg()], rtol=1e-7,
275:             endpt_atol=[None, 1e-10, 1e-10])
276: 
277:     def test_stdtr(self):
278:         # Ideally the left endpoint for Arg() should be 0.
279:         assert_mpmath_equal(
280:             sp.stdtr,
281:             _student_t_cdf,
282:             [IntArg(1, 100), Arg(1e-10, np.inf)], rtol=1e-7)
283: 
284:     @pytest.mark.xfail(run=False)
285:     def test_stdtridf(self):
286:         _assert_inverts(
287:             sp.stdtridf,
288:             _student_t_cdf,
289:             0, [ProbArg(), Arg()], rtol=1e-7)
290: 
291:     def test_stdtrit(self):
292:         _assert_inverts(
293:             sp.stdtrit,
294:             _student_t_cdf,
295:             1, [IntArg(1, 100), ProbArg()], rtol=1e-7,
296:             endpt_atol=[None, 1e-10])
297: 
298:     def test_chdtriv(self):
299:         _assert_inverts(
300:             sp.chdtriv,
301:             lambda v, x: mpmath.gammainc(v/2, b=x/2, regularized=True),
302:             0, [ProbArg(), IntArg(1, 100)], rtol=1e-4)
303: 
304:     @pytest.mark.xfail(run=False)
305:     def test_chndtridf(self):
306:         # Use a larger atol since mpmath is doing numerical integration
307:         _assert_inverts(
308:             sp.chndtridf,
309:             _noncentral_chi_cdf,
310:             1, [Arg(0, 100, inclusive_a=False), ProbArg(),
311:                 Arg(0, 100, inclusive_a=False)],
312:             n=1000, rtol=1e-4, atol=1e-15)
313: 
314:     @pytest.mark.xfail(run=False)
315:     def test_chndtrinc(self):
316:         # Use a larger atol since mpmath is doing numerical integration
317:         _assert_inverts(
318:             sp.chndtrinc,
319:             _noncentral_chi_cdf,
320:             2, [Arg(0, 100, inclusive_a=False), IntArg(1, 100), ProbArg()],
321:             n=1000, rtol=1e-4, atol=1e-15)
322:         
323:     def test_chndtrix(self):
324:         # Use a larger atol since mpmath is doing numerical integration
325:         _assert_inverts(
326:             sp.chndtrix,
327:             _noncentral_chi_cdf,
328:             0, [ProbArg(), IntArg(1, 100), Arg(0, 100, inclusive_a=False)],
329:             n=1000, rtol=1e-4, atol=1e-15,
330:             endpt_atol=[1e-6, None, None])
331: 
332:     def test_tklmbda_zero_shape(self):
333:         # When lmbda = 0 the CDF has a simple closed form
334:         one = mpmath.mpf(1)
335:         assert_mpmath_equal(
336:             lambda x: sp.tklmbda(x, 0),
337:             lambda x: one/(mpmath.exp(-x) + one),
338:             [Arg()], rtol=1e-7)
339: 
340:     def test_tklmbda_neg_shape(self):
341:         _assert_inverts(
342:             sp.tklmbda,
343:             _tukey_lmbda_quantile,
344:             0, [ProbArg(), Arg(-np.inf, 0, inclusive_b=False)],
345:             spfunc_first=False, rtol=1e-5,
346:             endpt_atol=[1e-9, None])
347: 
348:     @pytest.mark.xfail(run=False)
349:     def test_tklmbda_pos_shape(self):
350:         _assert_inverts(
351:             sp.tklmbda,
352:             _tukey_lmbda_quantile,
353:             0, [ProbArg(), Arg(0, 100, inclusive_a=False)],
354:             spfunc_first=False, rtol=1e-5)
355: 
356: 
357: def test_nonfinite():
358:     funcs = [
359:         ("btdtria", 3),
360:         ("btdtrib", 3),
361:         ("bdtrik", 3),
362:         ("bdtrin", 3),
363:         ("chdtriv", 2),
364:         ("chndtr", 3),
365:         ("chndtrix", 3),
366:         ("chndtridf", 3),
367:         ("chndtrinc", 3),
368:         ("fdtridfd", 3),
369:         ("ncfdtr", 4),
370:         ("ncfdtri", 4),
371:         ("ncfdtridfn", 4),
372:         ("ncfdtridfd", 4),
373:         ("ncfdtrinc", 4),
374:         ("gdtrix", 3),
375:         ("gdtrib", 3),
376:         ("gdtria", 3),
377:         ("nbdtrik", 3),
378:         ("nbdtrin", 3),
379:         ("nrdtrimn", 3),
380:         ("nrdtrisd", 3),
381:         ("pdtrik", 2),
382:         ("stdtr", 2),
383:         ("stdtrit", 2),
384:         ("stdtridf", 2),
385:         ("nctdtr", 3),
386:         ("nctdtrit", 3),
387:         ("nctdtridf", 3),
388:         ("nctdtrinc", 3),
389:         ("tklmbda", 2),
390:     ]
391: 
392:     np.random.seed(1)
393: 
394:     for func, numargs in funcs:
395:         func = getattr(sp, func)
396: 
397:         args_choices = [(float(x), np.nan, np.inf, -np.inf) for x in
398:                         np.random.rand(numargs)]
399: 
400:         for args in itertools.product(*args_choices):
401:             res = func(*args)
402: 
403:             if any(np.isnan(x) for x in args):
404:                 # Nan inputs should result to nan output
405:                 assert_equal(res, np.nan)
406:             else:
407:                 # All other inputs should return something (but not
408:                 # raise exceptions or cause hangs)
409:                 pass
410: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_530889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, (-1)), 'str', '\nTest cdflib functions versus mpmath, if available.\n\nThe following functions still need tests:\n\n- ncfdtr\n- ncfdtri\n- ncfdtridfn\n- ncfdtridfd\n- ncfdtrinc\n- nbdtrik\n- nbdtrin\n- nrdtrimn\n- nrdtrisd\n- pdtrik\n- nctdtr\n- nctdtrit\n- nctdtridf\n- nctdtrinc\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 24, 0))

# 'import itertools' statement (line 24)
import itertools

import_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'itertools', itertools, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 26, 0))

# 'import numpy' statement (line 26)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_530890 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'numpy')

if (type(import_530890) is not StypyTypeError):

    if (import_530890 != 'pyd_module'):
        __import__(import_530890)
        sys_modules_530891 = sys.modules[import_530890]
        import_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'np', sys_modules_530891.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'numpy', import_530890)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 27, 0))

# 'from numpy.testing import assert_equal' statement (line 27)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_530892 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 27, 0), 'numpy.testing')

if (type(import_530892) is not StypyTypeError):

    if (import_530892 != 'pyd_module'):
        __import__(import_530892)
        sys_modules_530893 = sys.modules[import_530892]
        import_from_module(stypy.reporting.localization.Localization(__file__, 27, 0), 'numpy.testing', sys_modules_530893.module_type_store, module_type_store, ['assert_equal'])
        nest_module(stypy.reporting.localization.Localization(__file__, 27, 0), __file__, sys_modules_530893, sys_modules_530893.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_equal

        import_from_module(stypy.reporting.localization.Localization(__file__, 27, 0), 'numpy.testing', None, module_type_store, ['assert_equal'], [assert_equal])

else:
    # Assigning a type to the variable 'numpy.testing' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 0), 'numpy.testing', import_530892)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 28, 0))

# 'import pytest' statement (line 28)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_530894 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 28, 0), 'pytest')

if (type(import_530894) is not StypyTypeError):

    if (import_530894 != 'pyd_module'):
        __import__(import_530894)
        sys_modules_530895 = sys.modules[import_530894]
        import_module(stypy.reporting.localization.Localization(__file__, 28, 0), 'pytest', sys_modules_530895.module_type_store, module_type_store)
    else:
        import pytest

        import_module(stypy.reporting.localization.Localization(__file__, 28, 0), 'pytest', pytest, module_type_store)

else:
    # Assigning a type to the variable 'pytest' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'pytest', import_530894)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 30, 0))

# 'import scipy.special' statement (line 30)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_530896 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 30, 0), 'scipy.special')

if (type(import_530896) is not StypyTypeError):

    if (import_530896 != 'pyd_module'):
        __import__(import_530896)
        sys_modules_530897 = sys.modules[import_530896]
        import_module(stypy.reporting.localization.Localization(__file__, 30, 0), 'sp', sys_modules_530897.module_type_store, module_type_store)
    else:
        import scipy.special as sp

        import_module(stypy.reporting.localization.Localization(__file__, 30, 0), 'sp', scipy.special, module_type_store)

else:
    # Assigning a type to the variable 'scipy.special' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), 'scipy.special', import_530896)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 31, 0))

# 'from scipy._lib.six import with_metaclass' statement (line 31)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_530898 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 31, 0), 'scipy._lib.six')

if (type(import_530898) is not StypyTypeError):

    if (import_530898 != 'pyd_module'):
        __import__(import_530898)
        sys_modules_530899 = sys.modules[import_530898]
        import_from_module(stypy.reporting.localization.Localization(__file__, 31, 0), 'scipy._lib.six', sys_modules_530899.module_type_store, module_type_store, ['with_metaclass'])
        nest_module(stypy.reporting.localization.Localization(__file__, 31, 0), __file__, sys_modules_530899, sys_modules_530899.module_type_store, module_type_store)
    else:
        from scipy._lib.six import with_metaclass

        import_from_module(stypy.reporting.localization.Localization(__file__, 31, 0), 'scipy._lib.six', None, module_type_store, ['with_metaclass'], [with_metaclass])

else:
    # Assigning a type to the variable 'scipy._lib.six' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), 'scipy._lib.six', import_530898)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 32, 0))

# 'from scipy.special._testutils import MissingModule, check_version, FuncData' statement (line 32)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_530900 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 32, 0), 'scipy.special._testutils')

if (type(import_530900) is not StypyTypeError):

    if (import_530900 != 'pyd_module'):
        __import__(import_530900)
        sys_modules_530901 = sys.modules[import_530900]
        import_from_module(stypy.reporting.localization.Localization(__file__, 32, 0), 'scipy.special._testutils', sys_modules_530901.module_type_store, module_type_store, ['MissingModule', 'check_version', 'FuncData'])
        nest_module(stypy.reporting.localization.Localization(__file__, 32, 0), __file__, sys_modules_530901, sys_modules_530901.module_type_store, module_type_store)
    else:
        from scipy.special._testutils import MissingModule, check_version, FuncData

        import_from_module(stypy.reporting.localization.Localization(__file__, 32, 0), 'scipy.special._testutils', None, module_type_store, ['MissingModule', 'check_version', 'FuncData'], [MissingModule, check_version, FuncData])

else:
    # Assigning a type to the variable 'scipy.special._testutils' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 0), 'scipy.special._testutils', import_530900)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 34, 0))

# 'from scipy.special._mptestutils import Arg, IntArg, get_args, mpf2float, assert_mpmath_equal' statement (line 34)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_530902 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 34, 0), 'scipy.special._mptestutils')

if (type(import_530902) is not StypyTypeError):

    if (import_530902 != 'pyd_module'):
        __import__(import_530902)
        sys_modules_530903 = sys.modules[import_530902]
        import_from_module(stypy.reporting.localization.Localization(__file__, 34, 0), 'scipy.special._mptestutils', sys_modules_530903.module_type_store, module_type_store, ['Arg', 'IntArg', 'get_args', 'mpf2float', 'assert_mpmath_equal'])
        nest_module(stypy.reporting.localization.Localization(__file__, 34, 0), __file__, sys_modules_530903, sys_modules_530903.module_type_store, module_type_store)
    else:
        from scipy.special._mptestutils import Arg, IntArg, get_args, mpf2float, assert_mpmath_equal

        import_from_module(stypy.reporting.localization.Localization(__file__, 34, 0), 'scipy.special._mptestutils', None, module_type_store, ['Arg', 'IntArg', 'get_args', 'mpf2float', 'assert_mpmath_equal'], [Arg, IntArg, get_args, mpf2float, assert_mpmath_equal])

else:
    # Assigning a type to the variable 'scipy.special._mptestutils' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), 'scipy.special._mptestutils', import_530902)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')



# SSA begins for try-except statement (line 37)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 38, 4))

# 'import mpmath' statement (line 38)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_530904 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 38, 4), 'mpmath')

if (type(import_530904) is not StypyTypeError):

    if (import_530904 != 'pyd_module'):
        __import__(import_530904)
        sys_modules_530905 = sys.modules[import_530904]
        import_module(stypy.reporting.localization.Localization(__file__, 38, 4), 'mpmath', sys_modules_530905.module_type_store, module_type_store)
    else:
        import mpmath

        import_module(stypy.reporting.localization.Localization(__file__, 38, 4), 'mpmath', mpmath, module_type_store)

else:
    # Assigning a type to the variable 'mpmath' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'mpmath', import_530904)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')

# SSA branch for the except part of a try statement (line 37)
# SSA branch for the except 'ImportError' branch of a try statement (line 37)
module_type_store.open_ssa_branch('except')

# Assigning a Call to a Name (line 40):

# Assigning a Call to a Name (line 40):

# Call to MissingModule(...): (line 40)
# Processing the call arguments (line 40)
str_530907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 27), 'str', 'mpmath')
# Processing the call keyword arguments (line 40)
kwargs_530908 = {}
# Getting the type of 'MissingModule' (line 40)
MissingModule_530906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 13), 'MissingModule', False)
# Calling MissingModule(args, kwargs) (line 40)
MissingModule_call_result_530909 = invoke(stypy.reporting.localization.Localization(__file__, 40, 13), MissingModule_530906, *[str_530907], **kwargs_530908)

# Assigning a type to the variable 'mpmath' (line 40)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'mpmath', MissingModule_call_result_530909)
# SSA join for try-except statement (line 37)
module_type_store = module_type_store.join_ssa_context()

# Declaration of the 'ProbArg' class

class ProbArg(object, ):
    str_530910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 4), 'str', 'Generate a set of probabilities on [0, 1].')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 45, 4, False)
        # Assigning a type to the variable 'self' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ProbArg.__init__', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Num to a Attribute (line 47):
        
        # Assigning a Num to a Attribute (line 47):
        int_530911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 17), 'int')
        # Getting the type of 'self' (line 47)
        self_530912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'self')
        # Setting the type of the member 'a' of a type (line 47)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 8), self_530912, 'a', int_530911)
        
        # Assigning a Num to a Attribute (line 48):
        
        # Assigning a Num to a Attribute (line 48):
        int_530913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 17), 'int')
        # Getting the type of 'self' (line 48)
        self_530914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'self')
        # Setting the type of the member 'b' of a type (line 48)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 8), self_530914, 'b', int_530913)
        
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
        module_type_store = module_type_store.open_function_context('values', 50, 4, False)
        # Assigning a type to the variable 'self' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ProbArg.values.__dict__.__setitem__('stypy_localization', localization)
        ProbArg.values.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ProbArg.values.__dict__.__setitem__('stypy_type_store', module_type_store)
        ProbArg.values.__dict__.__setitem__('stypy_function_name', 'ProbArg.values')
        ProbArg.values.__dict__.__setitem__('stypy_param_names_list', ['n'])
        ProbArg.values.__dict__.__setitem__('stypy_varargs_param_name', None)
        ProbArg.values.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ProbArg.values.__dict__.__setitem__('stypy_call_defaults', defaults)
        ProbArg.values.__dict__.__setitem__('stypy_call_varargs', varargs)
        ProbArg.values.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ProbArg.values.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ProbArg.values', ['n'], None, None, defaults, varargs, kwargs)

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

        str_530915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 8), 'str', 'Return an array containing approximatively n numbers.')
        
        # Assigning a Call to a Name (line 52):
        
        # Assigning a Call to a Name (line 52):
        
        # Call to max(...): (line 52)
        # Processing the call arguments (line 52)
        int_530917 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 16), 'int')
        # Getting the type of 'n' (line 52)
        n_530918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 19), 'n', False)
        int_530919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 22), 'int')
        # Applying the binary operator '//' (line 52)
        result_floordiv_530920 = python_operator(stypy.reporting.localization.Localization(__file__, 52, 19), '//', n_530918, int_530919)
        
        # Processing the call keyword arguments (line 52)
        kwargs_530921 = {}
        # Getting the type of 'max' (line 52)
        max_530916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 12), 'max', False)
        # Calling max(args, kwargs) (line 52)
        max_call_result_530922 = invoke(stypy.reporting.localization.Localization(__file__, 52, 12), max_530916, *[int_530917, result_floordiv_530920], **kwargs_530921)
        
        # Assigning a type to the variable 'm' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'm', max_call_result_530922)
        
        # Assigning a Call to a Name (line 53):
        
        # Assigning a Call to a Name (line 53):
        
        # Call to logspace(...): (line 53)
        # Processing the call arguments (line 53)
        int_530925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 25), 'int')
        
        # Call to log10(...): (line 53)
        # Processing the call arguments (line 53)
        float_530928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 39), 'float')
        # Processing the call keyword arguments (line 53)
        kwargs_530929 = {}
        # Getting the type of 'np' (line 53)
        np_530926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 30), 'np', False)
        # Obtaining the member 'log10' of a type (line 53)
        log10_530927 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 30), np_530926, 'log10')
        # Calling log10(args, kwargs) (line 53)
        log10_call_result_530930 = invoke(stypy.reporting.localization.Localization(__file__, 53, 30), log10_530927, *[float_530928], **kwargs_530929)
        
        # Getting the type of 'm' (line 53)
        m_530931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 45), 'm', False)
        # Processing the call keyword arguments (line 53)
        kwargs_530932 = {}
        # Getting the type of 'np' (line 53)
        np_530923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 13), 'np', False)
        # Obtaining the member 'logspace' of a type (line 53)
        logspace_530924 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 13), np_530923, 'logspace')
        # Calling logspace(args, kwargs) (line 53)
        logspace_call_result_530933 = invoke(stypy.reporting.localization.Localization(__file__, 53, 13), logspace_530924, *[int_530925, log10_call_result_530930, m_530931], **kwargs_530932)
        
        # Assigning a type to the variable 'v1' (line 53)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'v1', logspace_call_result_530933)
        
        # Assigning a Subscript to a Name (line 54):
        
        # Assigning a Subscript to a Name (line 54):
        
        # Obtaining the type of the subscript
        int_530934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 58), 'int')
        slice_530935 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 54, 13), int_530934, None, None)
        
        # Call to linspace(...): (line 54)
        # Processing the call arguments (line 54)
        float_530938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 25), 'float')
        float_530939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 30), 'float')
        # Getting the type of 'm' (line 54)
        m_530940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 35), 'm', False)
        int_530941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 39), 'int')
        # Applying the binary operator '+' (line 54)
        result_add_530942 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 35), '+', m_530940, int_530941)
        
        # Processing the call keyword arguments (line 54)
        # Getting the type of 'False' (line 54)
        False_530943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 51), 'False', False)
        keyword_530944 = False_530943
        kwargs_530945 = {'endpoint': keyword_530944}
        # Getting the type of 'np' (line 54)
        np_530936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 13), 'np', False)
        # Obtaining the member 'linspace' of a type (line 54)
        linspace_530937 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 13), np_530936, 'linspace')
        # Calling linspace(args, kwargs) (line 54)
        linspace_call_result_530946 = invoke(stypy.reporting.localization.Localization(__file__, 54, 13), linspace_530937, *[float_530938, float_530939, result_add_530942], **kwargs_530945)
        
        # Obtaining the member '__getitem__' of a type (line 54)
        getitem___530947 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 13), linspace_call_result_530946, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 54)
        subscript_call_result_530948 = invoke(stypy.reporting.localization.Localization(__file__, 54, 13), getitem___530947, slice_530935)
        
        # Assigning a type to the variable 'v2' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'v2', subscript_call_result_530948)
        
        # Assigning a BinOp to a Name (line 55):
        
        # Assigning a BinOp to a Name (line 55):
        int_530949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 13), 'int')
        
        # Call to logspace(...): (line 55)
        # Processing the call arguments (line 55)
        
        # Call to log10(...): (line 55)
        # Processing the call arguments (line 55)
        float_530954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 38), 'float')
        # Processing the call keyword arguments (line 55)
        kwargs_530955 = {}
        # Getting the type of 'np' (line 55)
        np_530952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 29), 'np', False)
        # Obtaining the member 'log10' of a type (line 55)
        log10_530953 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 29), np_530952, 'log10')
        # Calling log10(args, kwargs) (line 55)
        log10_call_result_530956 = invoke(stypy.reporting.localization.Localization(__file__, 55, 29), log10_530953, *[float_530954], **kwargs_530955)
        
        int_530957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 44), 'int')
        # Getting the type of 'm' (line 55)
        m_530958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 49), 'm', False)
        # Processing the call keyword arguments (line 55)
        kwargs_530959 = {}
        # Getting the type of 'np' (line 55)
        np_530950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 17), 'np', False)
        # Obtaining the member 'logspace' of a type (line 55)
        logspace_530951 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 17), np_530950, 'logspace')
        # Calling logspace(args, kwargs) (line 55)
        logspace_call_result_530960 = invoke(stypy.reporting.localization.Localization(__file__, 55, 17), logspace_530951, *[log10_call_result_530956, int_530957, m_530958], **kwargs_530959)
        
        # Applying the binary operator '-' (line 55)
        result_sub_530961 = python_operator(stypy.reporting.localization.Localization(__file__, 55, 13), '-', int_530949, logspace_call_result_530960)
        
        # Assigning a type to the variable 'v3' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'v3', result_sub_530961)
        
        # Assigning a Subscript to a Name (line 56):
        
        # Assigning a Subscript to a Name (line 56):
        
        # Obtaining the type of the subscript
        
        # Obtaining an instance of the builtin type 'tuple' (line 56)
        tuple_530962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 18), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 56)
        # Adding element type (line 56)
        # Getting the type of 'v1' (line 56)
        v1_530963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 18), 'v1')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 18), tuple_530962, v1_530963)
        # Adding element type (line 56)
        # Getting the type of 'v2' (line 56)
        v2_530964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 22), 'v2')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 18), tuple_530962, v2_530964)
        # Adding element type (line 56)
        # Getting the type of 'v3' (line 56)
        v3_530965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 26), 'v3')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 18), tuple_530962, v3_530965)
        
        # Getting the type of 'np' (line 56)
        np_530966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 12), 'np')
        # Obtaining the member 'r_' of a type (line 56)
        r__530967 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 12), np_530966, 'r_')
        # Obtaining the member '__getitem__' of a type (line 56)
        getitem___530968 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 12), r__530967, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 56)
        subscript_call_result_530969 = invoke(stypy.reporting.localization.Localization(__file__, 56, 12), getitem___530968, tuple_530962)
        
        # Assigning a type to the variable 'v' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'v', subscript_call_result_530969)
        
        # Call to unique(...): (line 57)
        # Processing the call arguments (line 57)
        # Getting the type of 'v' (line 57)
        v_530972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 25), 'v', False)
        # Processing the call keyword arguments (line 57)
        kwargs_530973 = {}
        # Getting the type of 'np' (line 57)
        np_530970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 15), 'np', False)
        # Obtaining the member 'unique' of a type (line 57)
        unique_530971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 15), np_530970, 'unique')
        # Calling unique(args, kwargs) (line 57)
        unique_call_result_530974 = invoke(stypy.reporting.localization.Localization(__file__, 57, 15), unique_530971, *[v_530972], **kwargs_530973)
        
        # Assigning a type to the variable 'stypy_return_type' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'stypy_return_type', unique_call_result_530974)
        
        # ################# End of 'values(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'values' in the type store
        # Getting the type of 'stypy_return_type' (line 50)
        stypy_return_type_530975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_530975)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'values'
        return stypy_return_type_530975


# Assigning a type to the variable 'ProbArg' (line 43)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 0), 'ProbArg', ProbArg)
# Declaration of the 'EndpointFilter' class

class EndpointFilter(object, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 61, 4, False)
        # Assigning a type to the variable 'self' (line 62)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'EndpointFilter.__init__', ['a', 'b', 'rtol', 'atol'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['a', 'b', 'rtol', 'atol'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 62):
        
        # Assigning a Name to a Attribute (line 62):
        # Getting the type of 'a' (line 62)
        a_530976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 17), 'a')
        # Getting the type of 'self' (line 62)
        self_530977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'self')
        # Setting the type of the member 'a' of a type (line 62)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 8), self_530977, 'a', a_530976)
        
        # Assigning a Name to a Attribute (line 63):
        
        # Assigning a Name to a Attribute (line 63):
        # Getting the type of 'b' (line 63)
        b_530978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 17), 'b')
        # Getting the type of 'self' (line 63)
        self_530979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'self')
        # Setting the type of the member 'b' of a type (line 63)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 8), self_530979, 'b', b_530978)
        
        # Assigning a Name to a Attribute (line 64):
        
        # Assigning a Name to a Attribute (line 64):
        # Getting the type of 'rtol' (line 64)
        rtol_530980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 20), 'rtol')
        # Getting the type of 'self' (line 64)
        self_530981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'self')
        # Setting the type of the member 'rtol' of a type (line 64)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 8), self_530981, 'rtol', rtol_530980)
        
        # Assigning a Name to a Attribute (line 65):
        
        # Assigning a Name to a Attribute (line 65):
        # Getting the type of 'atol' (line 65)
        atol_530982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 20), 'atol')
        # Getting the type of 'self' (line 65)
        self_530983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'self')
        # Setting the type of the member 'atol' of a type (line 65)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 8), self_530983, 'atol', atol_530982)
        
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
        module_type_store = module_type_store.open_function_context('__call__', 67, 4, False)
        # Assigning a type to the variable 'self' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        EndpointFilter.__call__.__dict__.__setitem__('stypy_localization', localization)
        EndpointFilter.__call__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        EndpointFilter.__call__.__dict__.__setitem__('stypy_type_store', module_type_store)
        EndpointFilter.__call__.__dict__.__setitem__('stypy_function_name', 'EndpointFilter.__call__')
        EndpointFilter.__call__.__dict__.__setitem__('stypy_param_names_list', ['x'])
        EndpointFilter.__call__.__dict__.__setitem__('stypy_varargs_param_name', None)
        EndpointFilter.__call__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        EndpointFilter.__call__.__dict__.__setitem__('stypy_call_defaults', defaults)
        EndpointFilter.__call__.__dict__.__setitem__('stypy_call_varargs', varargs)
        EndpointFilter.__call__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        EndpointFilter.__call__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'EndpointFilter.__call__', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__call__', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__call__(...)' code ##################

        
        # Assigning a Compare to a Name (line 68):
        
        # Assigning a Compare to a Name (line 68):
        
        
        # Call to abs(...): (line 68)
        # Processing the call arguments (line 68)
        # Getting the type of 'x' (line 68)
        x_530986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 23), 'x', False)
        # Getting the type of 'self' (line 68)
        self_530987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 27), 'self', False)
        # Obtaining the member 'a' of a type (line 68)
        a_530988 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 27), self_530987, 'a')
        # Applying the binary operator '-' (line 68)
        result_sub_530989 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 23), '-', x_530986, a_530988)
        
        # Processing the call keyword arguments (line 68)
        kwargs_530990 = {}
        # Getting the type of 'np' (line 68)
        np_530984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 16), 'np', False)
        # Obtaining the member 'abs' of a type (line 68)
        abs_530985 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 16), np_530984, 'abs')
        # Calling abs(args, kwargs) (line 68)
        abs_call_result_530991 = invoke(stypy.reporting.localization.Localization(__file__, 68, 16), abs_530985, *[result_sub_530989], **kwargs_530990)
        
        # Getting the type of 'self' (line 68)
        self_530992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 37), 'self')
        # Obtaining the member 'rtol' of a type (line 68)
        rtol_530993 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 37), self_530992, 'rtol')
        
        # Call to abs(...): (line 68)
        # Processing the call arguments (line 68)
        # Getting the type of 'self' (line 68)
        self_530996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 54), 'self', False)
        # Obtaining the member 'a' of a type (line 68)
        a_530997 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 54), self_530996, 'a')
        # Processing the call keyword arguments (line 68)
        kwargs_530998 = {}
        # Getting the type of 'np' (line 68)
        np_530994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 47), 'np', False)
        # Obtaining the member 'abs' of a type (line 68)
        abs_530995 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 47), np_530994, 'abs')
        # Calling abs(args, kwargs) (line 68)
        abs_call_result_530999 = invoke(stypy.reporting.localization.Localization(__file__, 68, 47), abs_530995, *[a_530997], **kwargs_530998)
        
        # Applying the binary operator '*' (line 68)
        result_mul_531000 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 37), '*', rtol_530993, abs_call_result_530999)
        
        # Getting the type of 'self' (line 68)
        self_531001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 64), 'self')
        # Obtaining the member 'atol' of a type (line 68)
        atol_531002 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 64), self_531001, 'atol')
        # Applying the binary operator '+' (line 68)
        result_add_531003 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 37), '+', result_mul_531000, atol_531002)
        
        # Applying the binary operator '<' (line 68)
        result_lt_531004 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 16), '<', abs_call_result_530991, result_add_531003)
        
        # Assigning a type to the variable 'mask1' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'mask1', result_lt_531004)
        
        # Assigning a Compare to a Name (line 69):
        
        # Assigning a Compare to a Name (line 69):
        
        
        # Call to abs(...): (line 69)
        # Processing the call arguments (line 69)
        # Getting the type of 'x' (line 69)
        x_531007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 23), 'x', False)
        # Getting the type of 'self' (line 69)
        self_531008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 27), 'self', False)
        # Obtaining the member 'b' of a type (line 69)
        b_531009 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 27), self_531008, 'b')
        # Applying the binary operator '-' (line 69)
        result_sub_531010 = python_operator(stypy.reporting.localization.Localization(__file__, 69, 23), '-', x_531007, b_531009)
        
        # Processing the call keyword arguments (line 69)
        kwargs_531011 = {}
        # Getting the type of 'np' (line 69)
        np_531005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 16), 'np', False)
        # Obtaining the member 'abs' of a type (line 69)
        abs_531006 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 16), np_531005, 'abs')
        # Calling abs(args, kwargs) (line 69)
        abs_call_result_531012 = invoke(stypy.reporting.localization.Localization(__file__, 69, 16), abs_531006, *[result_sub_531010], **kwargs_531011)
        
        # Getting the type of 'self' (line 69)
        self_531013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 37), 'self')
        # Obtaining the member 'rtol' of a type (line 69)
        rtol_531014 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 37), self_531013, 'rtol')
        
        # Call to abs(...): (line 69)
        # Processing the call arguments (line 69)
        # Getting the type of 'self' (line 69)
        self_531017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 54), 'self', False)
        # Obtaining the member 'b' of a type (line 69)
        b_531018 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 54), self_531017, 'b')
        # Processing the call keyword arguments (line 69)
        kwargs_531019 = {}
        # Getting the type of 'np' (line 69)
        np_531015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 47), 'np', False)
        # Obtaining the member 'abs' of a type (line 69)
        abs_531016 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 47), np_531015, 'abs')
        # Calling abs(args, kwargs) (line 69)
        abs_call_result_531020 = invoke(stypy.reporting.localization.Localization(__file__, 69, 47), abs_531016, *[b_531018], **kwargs_531019)
        
        # Applying the binary operator '*' (line 69)
        result_mul_531021 = python_operator(stypy.reporting.localization.Localization(__file__, 69, 37), '*', rtol_531014, abs_call_result_531020)
        
        # Getting the type of 'self' (line 69)
        self_531022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 64), 'self')
        # Obtaining the member 'atol' of a type (line 69)
        atol_531023 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 64), self_531022, 'atol')
        # Applying the binary operator '+' (line 69)
        result_add_531024 = python_operator(stypy.reporting.localization.Localization(__file__, 69, 37), '+', result_mul_531021, atol_531023)
        
        # Applying the binary operator '<' (line 69)
        result_lt_531025 = python_operator(stypy.reporting.localization.Localization(__file__, 69, 16), '<', abs_call_result_531012, result_add_531024)
        
        # Assigning a type to the variable 'mask2' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'mask2', result_lt_531025)
        
        # Call to where(...): (line 70)
        # Processing the call arguments (line 70)
        # Getting the type of 'mask1' (line 70)
        mask1_531028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 24), 'mask1', False)
        # Getting the type of 'mask2' (line 70)
        mask2_531029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 32), 'mask2', False)
        # Applying the binary operator '|' (line 70)
        result_or__531030 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 24), '|', mask1_531028, mask2_531029)
        
        # Getting the type of 'False' (line 70)
        False_531031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 39), 'False', False)
        # Getting the type of 'True' (line 70)
        True_531032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 46), 'True', False)
        # Processing the call keyword arguments (line 70)
        kwargs_531033 = {}
        # Getting the type of 'np' (line 70)
        np_531026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 15), 'np', False)
        # Obtaining the member 'where' of a type (line 70)
        where_531027 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 15), np_531026, 'where')
        # Calling where(args, kwargs) (line 70)
        where_call_result_531034 = invoke(stypy.reporting.localization.Localization(__file__, 70, 15), where_531027, *[result_or__531030, False_531031, True_531032], **kwargs_531033)
        
        # Assigning a type to the variable 'stypy_return_type' (line 70)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'stypy_return_type', where_call_result_531034)
        
        # ################# End of '__call__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__call__' in the type store
        # Getting the type of 'stypy_return_type' (line 67)
        stypy_return_type_531035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_531035)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__call__'
        return stypy_return_type_531035


# Assigning a type to the variable 'EndpointFilter' (line 60)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 0), 'EndpointFilter', EndpointFilter)
# Declaration of the '_CDFData' class

class _CDFData(object, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'True' (line 74)
        True_531036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 68), 'True')
        int_531037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 21), 'int')
        int_531038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 27), 'int')
        # Getting the type of 'None' (line 75)
        None_531039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 38), 'None')
        # Getting the type of 'None' (line 75)
        None_531040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 49), 'None')
        # Getting the type of 'None' (line 76)
        None_531041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 28), 'None')
        # Getting the type of 'None' (line 76)
        None_531042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 45), 'None')
        defaults = [True_531036, int_531037, int_531038, None_531039, None_531040, None_531041, None_531042]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 74, 4, False)
        # Assigning a type to the variable 'self' (line 75)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_CDFData.__init__', ['spfunc', 'mpfunc', 'index', 'argspec', 'spfunc_first', 'dps', 'n', 'rtol', 'atol', 'endpt_rtol', 'endpt_atol'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['spfunc', 'mpfunc', 'index', 'argspec', 'spfunc_first', 'dps', 'n', 'rtol', 'atol', 'endpt_rtol', 'endpt_atol'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 77):
        
        # Assigning a Name to a Attribute (line 77):
        # Getting the type of 'spfunc' (line 77)
        spfunc_531043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 22), 'spfunc')
        # Getting the type of 'self' (line 77)
        self_531044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'self')
        # Setting the type of the member 'spfunc' of a type (line 77)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 8), self_531044, 'spfunc', spfunc_531043)
        
        # Assigning a Name to a Attribute (line 78):
        
        # Assigning a Name to a Attribute (line 78):
        # Getting the type of 'mpfunc' (line 78)
        mpfunc_531045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 22), 'mpfunc')
        # Getting the type of 'self' (line 78)
        self_531046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'self')
        # Setting the type of the member 'mpfunc' of a type (line 78)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 8), self_531046, 'mpfunc', mpfunc_531045)
        
        # Assigning a Name to a Attribute (line 79):
        
        # Assigning a Name to a Attribute (line 79):
        # Getting the type of 'index' (line 79)
        index_531047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 21), 'index')
        # Getting the type of 'self' (line 79)
        self_531048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'self')
        # Setting the type of the member 'index' of a type (line 79)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 8), self_531048, 'index', index_531047)
        
        # Assigning a Name to a Attribute (line 80):
        
        # Assigning a Name to a Attribute (line 80):
        # Getting the type of 'argspec' (line 80)
        argspec_531049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 23), 'argspec')
        # Getting the type of 'self' (line 80)
        self_531050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'self')
        # Setting the type of the member 'argspec' of a type (line 80)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 8), self_531050, 'argspec', argspec_531049)
        
        # Assigning a Name to a Attribute (line 81):
        
        # Assigning a Name to a Attribute (line 81):
        # Getting the type of 'spfunc_first' (line 81)
        spfunc_first_531051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 28), 'spfunc_first')
        # Getting the type of 'self' (line 81)
        self_531052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'self')
        # Setting the type of the member 'spfunc_first' of a type (line 81)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 8), self_531052, 'spfunc_first', spfunc_first_531051)
        
        # Assigning a Name to a Attribute (line 82):
        
        # Assigning a Name to a Attribute (line 82):
        # Getting the type of 'dps' (line 82)
        dps_531053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 19), 'dps')
        # Getting the type of 'self' (line 82)
        self_531054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'self')
        # Setting the type of the member 'dps' of a type (line 82)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 8), self_531054, 'dps', dps_531053)
        
        # Assigning a Name to a Attribute (line 83):
        
        # Assigning a Name to a Attribute (line 83):
        # Getting the type of 'n' (line 83)
        n_531055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 17), 'n')
        # Getting the type of 'self' (line 83)
        self_531056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'self')
        # Setting the type of the member 'n' of a type (line 83)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 8), self_531056, 'n', n_531055)
        
        # Assigning a Name to a Attribute (line 84):
        
        # Assigning a Name to a Attribute (line 84):
        # Getting the type of 'rtol' (line 84)
        rtol_531057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 20), 'rtol')
        # Getting the type of 'self' (line 84)
        self_531058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'self')
        # Setting the type of the member 'rtol' of a type (line 84)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 8), self_531058, 'rtol', rtol_531057)
        
        # Assigning a Name to a Attribute (line 85):
        
        # Assigning a Name to a Attribute (line 85):
        # Getting the type of 'atol' (line 85)
        atol_531059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 20), 'atol')
        # Getting the type of 'self' (line 85)
        self_531060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'self')
        # Setting the type of the member 'atol' of a type (line 85)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 8), self_531060, 'atol', atol_531059)
        
        # Type idiom detected: calculating its left and rigth part (line 87)
        # Getting the type of 'list' (line 87)
        list_531061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 35), 'list')
        # Getting the type of 'argspec' (line 87)
        argspec_531062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 26), 'argspec')
        
        (may_be_531063, more_types_in_union_531064) = may_not_be_subtype(list_531061, argspec_531062)

        if may_be_531063:

            if more_types_in_union_531064:
                # Runtime conditional SSA (line 87)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'argspec' (line 87)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'argspec', remove_subtype_from_union(argspec_531062, list))
            
            # Assigning a Name to a Attribute (line 88):
            
            # Assigning a Name to a Attribute (line 88):
            # Getting the type of 'None' (line 88)
            None_531065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 30), 'None')
            # Getting the type of 'self' (line 88)
            self_531066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 12), 'self')
            # Setting the type of the member 'endpt_rtol' of a type (line 88)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 12), self_531066, 'endpt_rtol', None_531065)
            
            # Assigning a Name to a Attribute (line 89):
            
            # Assigning a Name to a Attribute (line 89):
            # Getting the type of 'None' (line 89)
            None_531067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 30), 'None')
            # Getting the type of 'self' (line 89)
            self_531068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 12), 'self')
            # Setting the type of the member 'endpt_atol' of a type (line 89)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 12), self_531068, 'endpt_atol', None_531067)

            if more_types_in_union_531064:
                # Runtime conditional SSA for else branch (line 87)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_531063) or more_types_in_union_531064):
            # Assigning a type to the variable 'argspec' (line 87)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'argspec', remove_not_subtype_from_union(argspec_531062, list))
            
            
            # Evaluating a boolean operation
            
            # Getting the type of 'endpt_rtol' (line 90)
            endpt_rtol_531069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 13), 'endpt_rtol')
            # Getting the type of 'None' (line 90)
            None_531070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 31), 'None')
            # Applying the binary operator 'isnot' (line 90)
            result_is_not_531071 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 13), 'isnot', endpt_rtol_531069, None_531070)
            
            
            # Getting the type of 'endpt_atol' (line 90)
            endpt_atol_531072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 39), 'endpt_atol')
            # Getting the type of 'None' (line 90)
            None_531073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 57), 'None')
            # Applying the binary operator 'isnot' (line 90)
            result_is_not_531074 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 39), 'isnot', endpt_atol_531072, None_531073)
            
            # Applying the binary operator 'or' (line 90)
            result_or_keyword_531075 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 13), 'or', result_is_not_531071, result_is_not_531074)
            
            # Testing the type of an if condition (line 90)
            if_condition_531076 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 90, 13), result_or_keyword_531075)
            # Assigning a type to the variable 'if_condition_531076' (line 90)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 13), 'if_condition_531076', if_condition_531076)
            # SSA begins for if statement (line 90)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Type idiom detected: calculating its left and rigth part (line 91)
            # Getting the type of 'list' (line 91)
            list_531077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 38), 'list')
            # Getting the type of 'endpt_rtol' (line 91)
            endpt_rtol_531078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 26), 'endpt_rtol')
            
            (may_be_531079, more_types_in_union_531080) = may_be_subtype(list_531077, endpt_rtol_531078)

            if may_be_531079:

                if more_types_in_union_531080:
                    # Runtime conditional SSA (line 91)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Assigning a type to the variable 'endpt_rtol' (line 91)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 12), 'endpt_rtol', remove_not_subtype_from_union(endpt_rtol_531078, list))
                
                # Assigning a Name to a Attribute (line 92):
                
                # Assigning a Name to a Attribute (line 92):
                # Getting the type of 'endpt_rtol' (line 92)
                endpt_rtol_531081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 34), 'endpt_rtol')
                # Getting the type of 'self' (line 92)
                self_531082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 16), 'self')
                # Setting the type of the member 'endpt_rtol' of a type (line 92)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 16), self_531082, 'endpt_rtol', endpt_rtol_531081)

                if more_types_in_union_531080:
                    # Runtime conditional SSA for else branch (line 91)
                    module_type_store.open_ssa_branch('idiom else')



            if ((not may_be_531079) or more_types_in_union_531080):
                # Assigning a type to the variable 'endpt_rtol' (line 91)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 12), 'endpt_rtol', remove_subtype_from_union(endpt_rtol_531078, list))
                
                # Assigning a BinOp to a Attribute (line 94):
                
                # Assigning a BinOp to a Attribute (line 94):
                
                # Obtaining an instance of the builtin type 'list' (line 94)
                list_531083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 34), 'list')
                # Adding type elements to the builtin type 'list' instance (line 94)
                # Adding element type (line 94)
                # Getting the type of 'endpt_rtol' (line 94)
                endpt_rtol_531084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 35), 'endpt_rtol')
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 94, 34), list_531083, endpt_rtol_531084)
                
                
                # Call to len(...): (line 94)
                # Processing the call arguments (line 94)
                # Getting the type of 'self' (line 94)
                self_531086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 51), 'self', False)
                # Obtaining the member 'argspec' of a type (line 94)
                argspec_531087 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 51), self_531086, 'argspec')
                # Processing the call keyword arguments (line 94)
                kwargs_531088 = {}
                # Getting the type of 'len' (line 94)
                len_531085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 47), 'len', False)
                # Calling len(args, kwargs) (line 94)
                len_call_result_531089 = invoke(stypy.reporting.localization.Localization(__file__, 94, 47), len_531085, *[argspec_531087], **kwargs_531088)
                
                # Applying the binary operator '*' (line 94)
                result_mul_531090 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 34), '*', list_531083, len_call_result_531089)
                
                # Getting the type of 'self' (line 94)
                self_531091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 16), 'self')
                # Setting the type of the member 'endpt_rtol' of a type (line 94)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 16), self_531091, 'endpt_rtol', result_mul_531090)

                if (may_be_531079 and more_types_in_union_531080):
                    # SSA join for if statement (line 91)
                    module_type_store = module_type_store.join_ssa_context()


            
            
            # Type idiom detected: calculating its left and rigth part (line 95)
            # Getting the type of 'list' (line 95)
            list_531092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 38), 'list')
            # Getting the type of 'endpt_atol' (line 95)
            endpt_atol_531093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 26), 'endpt_atol')
            
            (may_be_531094, more_types_in_union_531095) = may_be_subtype(list_531092, endpt_atol_531093)

            if may_be_531094:

                if more_types_in_union_531095:
                    # Runtime conditional SSA (line 95)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Assigning a type to the variable 'endpt_atol' (line 95)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 12), 'endpt_atol', remove_not_subtype_from_union(endpt_atol_531093, list))
                
                # Assigning a Name to a Attribute (line 96):
                
                # Assigning a Name to a Attribute (line 96):
                # Getting the type of 'endpt_atol' (line 96)
                endpt_atol_531096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 34), 'endpt_atol')
                # Getting the type of 'self' (line 96)
                self_531097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 16), 'self')
                # Setting the type of the member 'endpt_atol' of a type (line 96)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 16), self_531097, 'endpt_atol', endpt_atol_531096)

                if more_types_in_union_531095:
                    # Runtime conditional SSA for else branch (line 95)
                    module_type_store.open_ssa_branch('idiom else')



            if ((not may_be_531094) or more_types_in_union_531095):
                # Assigning a type to the variable 'endpt_atol' (line 95)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 12), 'endpt_atol', remove_subtype_from_union(endpt_atol_531093, list))
                
                # Assigning a BinOp to a Attribute (line 98):
                
                # Assigning a BinOp to a Attribute (line 98):
                
                # Obtaining an instance of the builtin type 'list' (line 98)
                list_531098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 34), 'list')
                # Adding type elements to the builtin type 'list' instance (line 98)
                # Adding element type (line 98)
                # Getting the type of 'endpt_atol' (line 98)
                endpt_atol_531099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 35), 'endpt_atol')
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 34), list_531098, endpt_atol_531099)
                
                
                # Call to len(...): (line 98)
                # Processing the call arguments (line 98)
                # Getting the type of 'self' (line 98)
                self_531101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 51), 'self', False)
                # Obtaining the member 'argspec' of a type (line 98)
                argspec_531102 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 51), self_531101, 'argspec')
                # Processing the call keyword arguments (line 98)
                kwargs_531103 = {}
                # Getting the type of 'len' (line 98)
                len_531100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 47), 'len', False)
                # Calling len(args, kwargs) (line 98)
                len_call_result_531104 = invoke(stypy.reporting.localization.Localization(__file__, 98, 47), len_531100, *[argspec_531102], **kwargs_531103)
                
                # Applying the binary operator '*' (line 98)
                result_mul_531105 = python_operator(stypy.reporting.localization.Localization(__file__, 98, 34), '*', list_531098, len_call_result_531104)
                
                # Getting the type of 'self' (line 98)
                self_531106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 16), 'self')
                # Setting the type of the member 'endpt_atol' of a type (line 98)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 16), self_531106, 'endpt_atol', result_mul_531105)

                if (may_be_531094 and more_types_in_union_531095):
                    # SSA join for if statement (line 95)
                    module_type_store = module_type_store.join_ssa_context()


            
            # SSA branch for the else part of an if statement (line 90)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Name to a Attribute (line 100):
            
            # Assigning a Name to a Attribute (line 100):
            # Getting the type of 'None' (line 100)
            None_531107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 30), 'None')
            # Getting the type of 'self' (line 100)
            self_531108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 12), 'self')
            # Setting the type of the member 'endpt_rtol' of a type (line 100)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 12), self_531108, 'endpt_rtol', None_531107)
            
            # Assigning a Name to a Attribute (line 101):
            
            # Assigning a Name to a Attribute (line 101):
            # Getting the type of 'None' (line 101)
            None_531109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 30), 'None')
            # Getting the type of 'self' (line 101)
            self_531110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 12), 'self')
            # Setting the type of the member 'endpt_atol' of a type (line 101)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 12), self_531110, 'endpt_atol', None_531109)
            # SSA join for if statement (line 90)
            module_type_store = module_type_store.join_ssa_context()
            

            if (may_be_531063 and more_types_in_union_531064):
                # SSA join for if statement (line 87)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def idmap(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'idmap'
        module_type_store = module_type_store.open_function_context('idmap', 103, 4, False)
        # Assigning a type to the variable 'self' (line 104)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _CDFData.idmap.__dict__.__setitem__('stypy_localization', localization)
        _CDFData.idmap.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _CDFData.idmap.__dict__.__setitem__('stypy_type_store', module_type_store)
        _CDFData.idmap.__dict__.__setitem__('stypy_function_name', '_CDFData.idmap')
        _CDFData.idmap.__dict__.__setitem__('stypy_param_names_list', [])
        _CDFData.idmap.__dict__.__setitem__('stypy_varargs_param_name', 'args')
        _CDFData.idmap.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _CDFData.idmap.__dict__.__setitem__('stypy_call_defaults', defaults)
        _CDFData.idmap.__dict__.__setitem__('stypy_call_varargs', varargs)
        _CDFData.idmap.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _CDFData.idmap.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_CDFData.idmap', [], 'args', None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'idmap', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'idmap(...)' code ##################

        
        # Getting the type of 'self' (line 104)
        self_531111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 11), 'self')
        # Obtaining the member 'spfunc_first' of a type (line 104)
        spfunc_first_531112 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 11), self_531111, 'spfunc_first')
        # Testing the type of an if condition (line 104)
        if_condition_531113 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 104, 8), spfunc_first_531112)
        # Assigning a type to the variable 'if_condition_531113' (line 104)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'if_condition_531113', if_condition_531113)
        # SSA begins for if statement (line 104)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 105):
        
        # Assigning a Call to a Name (line 105):
        
        # Call to spfunc(...): (line 105)
        # Getting the type of 'args' (line 105)
        args_531116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 31), 'args', False)
        # Processing the call keyword arguments (line 105)
        kwargs_531117 = {}
        # Getting the type of 'self' (line 105)
        self_531114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 18), 'self', False)
        # Obtaining the member 'spfunc' of a type (line 105)
        spfunc_531115 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 18), self_531114, 'spfunc')
        # Calling spfunc(args, kwargs) (line 105)
        spfunc_call_result_531118 = invoke(stypy.reporting.localization.Localization(__file__, 105, 18), spfunc_531115, *[args_531116], **kwargs_531117)
        
        # Assigning a type to the variable 'res' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 12), 'res', spfunc_call_result_531118)
        
        
        # Call to isnan(...): (line 106)
        # Processing the call arguments (line 106)
        # Getting the type of 'res' (line 106)
        res_531121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 24), 'res', False)
        # Processing the call keyword arguments (line 106)
        kwargs_531122 = {}
        # Getting the type of 'np' (line 106)
        np_531119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 15), 'np', False)
        # Obtaining the member 'isnan' of a type (line 106)
        isnan_531120 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 15), np_531119, 'isnan')
        # Calling isnan(args, kwargs) (line 106)
        isnan_call_result_531123 = invoke(stypy.reporting.localization.Localization(__file__, 106, 15), isnan_531120, *[res_531121], **kwargs_531122)
        
        # Testing the type of an if condition (line 106)
        if_condition_531124 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 106, 12), isnan_call_result_531123)
        # Assigning a type to the variable 'if_condition_531124' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 12), 'if_condition_531124', if_condition_531124)
        # SSA begins for if statement (line 106)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'np' (line 107)
        np_531125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 23), 'np')
        # Obtaining the member 'nan' of a type (line 107)
        nan_531126 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 23), np_531125, 'nan')
        # Assigning a type to the variable 'stypy_return_type' (line 107)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 16), 'stypy_return_type', nan_531126)
        # SSA join for if statement (line 106)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 108):
        
        # Assigning a Call to a Name (line 108):
        
        # Call to list(...): (line 108)
        # Processing the call arguments (line 108)
        # Getting the type of 'args' (line 108)
        args_531128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 24), 'args', False)
        # Processing the call keyword arguments (line 108)
        kwargs_531129 = {}
        # Getting the type of 'list' (line 108)
        list_531127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 19), 'list', False)
        # Calling list(args, kwargs) (line 108)
        list_call_result_531130 = invoke(stypy.reporting.localization.Localization(__file__, 108, 19), list_531127, *[args_531128], **kwargs_531129)
        
        # Assigning a type to the variable 'args' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 12), 'args', list_call_result_531130)
        
        # Assigning a Name to a Subscript (line 109):
        
        # Assigning a Name to a Subscript (line 109):
        # Getting the type of 'res' (line 109)
        res_531131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 31), 'res')
        # Getting the type of 'args' (line 109)
        args_531132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 12), 'args')
        # Getting the type of 'self' (line 109)
        self_531133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 17), 'self')
        # Obtaining the member 'index' of a type (line 109)
        index_531134 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 17), self_531133, 'index')
        # Storing an element on a container (line 109)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 12), args_531132, (index_531134, res_531131))
        
        # Call to workdps(...): (line 110)
        # Processing the call arguments (line 110)
        # Getting the type of 'self' (line 110)
        self_531137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 32), 'self', False)
        # Obtaining the member 'dps' of a type (line 110)
        dps_531138 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 32), self_531137, 'dps')
        # Processing the call keyword arguments (line 110)
        kwargs_531139 = {}
        # Getting the type of 'mpmath' (line 110)
        mpmath_531135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 17), 'mpmath', False)
        # Obtaining the member 'workdps' of a type (line 110)
        workdps_531136 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 17), mpmath_531135, 'workdps')
        # Calling workdps(args, kwargs) (line 110)
        workdps_call_result_531140 = invoke(stypy.reporting.localization.Localization(__file__, 110, 17), workdps_531136, *[dps_531138], **kwargs_531139)
        
        with_531141 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 110, 17), workdps_call_result_531140, 'with parameter', '__enter__', '__exit__')

        if with_531141:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 110)
            enter___531142 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 17), workdps_call_result_531140, '__enter__')
            with_enter_531143 = invoke(stypy.reporting.localization.Localization(__file__, 110, 17), enter___531142)
            
            # Assigning a Call to a Name (line 111):
            
            # Assigning a Call to a Name (line 111):
            
            # Call to mpfunc(...): (line 111)
            
            # Call to tuple(...): (line 111)
            # Processing the call arguments (line 111)
            # Getting the type of 'args' (line 111)
            args_531147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 41), 'args', False)
            # Processing the call keyword arguments (line 111)
            kwargs_531148 = {}
            # Getting the type of 'tuple' (line 111)
            tuple_531146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 35), 'tuple', False)
            # Calling tuple(args, kwargs) (line 111)
            tuple_call_result_531149 = invoke(stypy.reporting.localization.Localization(__file__, 111, 35), tuple_531146, *[args_531147], **kwargs_531148)
            
            # Processing the call keyword arguments (line 111)
            kwargs_531150 = {}
            # Getting the type of 'self' (line 111)
            self_531144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 22), 'self', False)
            # Obtaining the member 'mpfunc' of a type (line 111)
            mpfunc_531145 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 22), self_531144, 'mpfunc')
            # Calling mpfunc(args, kwargs) (line 111)
            mpfunc_call_result_531151 = invoke(stypy.reporting.localization.Localization(__file__, 111, 22), mpfunc_531145, *[tuple_call_result_531149], **kwargs_531150)
            
            # Assigning a type to the variable 'res' (line 111)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 16), 'res', mpfunc_call_result_531151)
            
            # Assigning a Call to a Name (line 113):
            
            # Assigning a Call to a Name (line 113):
            
            # Call to mpf2float(...): (line 113)
            # Processing the call arguments (line 113)
            # Getting the type of 'res' (line 113)
            res_531153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 32), 'res', False)
            # Obtaining the member 'real' of a type (line 113)
            real_531154 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 32), res_531153, 'real')
            # Processing the call keyword arguments (line 113)
            kwargs_531155 = {}
            # Getting the type of 'mpf2float' (line 113)
            mpf2float_531152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 22), 'mpf2float', False)
            # Calling mpf2float(args, kwargs) (line 113)
            mpf2float_call_result_531156 = invoke(stypy.reporting.localization.Localization(__file__, 113, 22), mpf2float_531152, *[real_531154], **kwargs_531155)
            
            # Assigning a type to the variable 'res' (line 113)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 16), 'res', mpf2float_call_result_531156)
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 110)
            exit___531157 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 17), workdps_call_result_531140, '__exit__')
            with_exit_531158 = invoke(stypy.reporting.localization.Localization(__file__, 110, 17), exit___531157, None, None, None)

        # SSA branch for the else part of an if statement (line 104)
        module_type_store.open_ssa_branch('else')
        
        # Call to workdps(...): (line 115)
        # Processing the call arguments (line 115)
        # Getting the type of 'self' (line 115)
        self_531161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 32), 'self', False)
        # Obtaining the member 'dps' of a type (line 115)
        dps_531162 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 32), self_531161, 'dps')
        # Processing the call keyword arguments (line 115)
        kwargs_531163 = {}
        # Getting the type of 'mpmath' (line 115)
        mpmath_531159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 17), 'mpmath', False)
        # Obtaining the member 'workdps' of a type (line 115)
        workdps_531160 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 17), mpmath_531159, 'workdps')
        # Calling workdps(args, kwargs) (line 115)
        workdps_call_result_531164 = invoke(stypy.reporting.localization.Localization(__file__, 115, 17), workdps_531160, *[dps_531162], **kwargs_531163)
        
        with_531165 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 115, 17), workdps_call_result_531164, 'with parameter', '__enter__', '__exit__')

        if with_531165:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 115)
            enter___531166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 17), workdps_call_result_531164, '__enter__')
            with_enter_531167 = invoke(stypy.reporting.localization.Localization(__file__, 115, 17), enter___531166)
            
            # Assigning a Call to a Name (line 116):
            
            # Assigning a Call to a Name (line 116):
            
            # Call to mpfunc(...): (line 116)
            # Getting the type of 'args' (line 116)
            args_531170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 35), 'args', False)
            # Processing the call keyword arguments (line 116)
            kwargs_531171 = {}
            # Getting the type of 'self' (line 116)
            self_531168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 22), 'self', False)
            # Obtaining the member 'mpfunc' of a type (line 116)
            mpfunc_531169 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 22), self_531168, 'mpfunc')
            # Calling mpfunc(args, kwargs) (line 116)
            mpfunc_call_result_531172 = invoke(stypy.reporting.localization.Localization(__file__, 116, 22), mpfunc_531169, *[args_531170], **kwargs_531171)
            
            # Assigning a type to the variable 'res' (line 116)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 16), 'res', mpfunc_call_result_531172)
            
            # Assigning a Call to a Name (line 117):
            
            # Assigning a Call to a Name (line 117):
            
            # Call to mpf2float(...): (line 117)
            # Processing the call arguments (line 117)
            # Getting the type of 'res' (line 117)
            res_531174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 32), 'res', False)
            # Obtaining the member 'real' of a type (line 117)
            real_531175 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 32), res_531174, 'real')
            # Processing the call keyword arguments (line 117)
            kwargs_531176 = {}
            # Getting the type of 'mpf2float' (line 117)
            mpf2float_531173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 22), 'mpf2float', False)
            # Calling mpf2float(args, kwargs) (line 117)
            mpf2float_call_result_531177 = invoke(stypy.reporting.localization.Localization(__file__, 117, 22), mpf2float_531173, *[real_531175], **kwargs_531176)
            
            # Assigning a type to the variable 'res' (line 117)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 16), 'res', mpf2float_call_result_531177)
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 115)
            exit___531178 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 17), workdps_call_result_531164, '__exit__')
            with_exit_531179 = invoke(stypy.reporting.localization.Localization(__file__, 115, 17), exit___531178, None, None, None)

        
        # Assigning a Call to a Name (line 118):
        
        # Assigning a Call to a Name (line 118):
        
        # Call to list(...): (line 118)
        # Processing the call arguments (line 118)
        # Getting the type of 'args' (line 118)
        args_531181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 24), 'args', False)
        # Processing the call keyword arguments (line 118)
        kwargs_531182 = {}
        # Getting the type of 'list' (line 118)
        list_531180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 19), 'list', False)
        # Calling list(args, kwargs) (line 118)
        list_call_result_531183 = invoke(stypy.reporting.localization.Localization(__file__, 118, 19), list_531180, *[args_531181], **kwargs_531182)
        
        # Assigning a type to the variable 'args' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 12), 'args', list_call_result_531183)
        
        # Assigning a Name to a Subscript (line 119):
        
        # Assigning a Name to a Subscript (line 119):
        # Getting the type of 'res' (line 119)
        res_531184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 31), 'res')
        # Getting the type of 'args' (line 119)
        args_531185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 12), 'args')
        # Getting the type of 'self' (line 119)
        self_531186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 17), 'self')
        # Obtaining the member 'index' of a type (line 119)
        index_531187 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 17), self_531186, 'index')
        # Storing an element on a container (line 119)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 12), args_531185, (index_531187, res_531184))
        
        # Assigning a Call to a Name (line 120):
        
        # Assigning a Call to a Name (line 120):
        
        # Call to spfunc(...): (line 120)
        
        # Call to tuple(...): (line 120)
        # Processing the call arguments (line 120)
        # Getting the type of 'args' (line 120)
        args_531191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 37), 'args', False)
        # Processing the call keyword arguments (line 120)
        kwargs_531192 = {}
        # Getting the type of 'tuple' (line 120)
        tuple_531190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 31), 'tuple', False)
        # Calling tuple(args, kwargs) (line 120)
        tuple_call_result_531193 = invoke(stypy.reporting.localization.Localization(__file__, 120, 31), tuple_531190, *[args_531191], **kwargs_531192)
        
        # Processing the call keyword arguments (line 120)
        kwargs_531194 = {}
        # Getting the type of 'self' (line 120)
        self_531188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 18), 'self', False)
        # Obtaining the member 'spfunc' of a type (line 120)
        spfunc_531189 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 18), self_531188, 'spfunc')
        # Calling spfunc(args, kwargs) (line 120)
        spfunc_call_result_531195 = invoke(stypy.reporting.localization.Localization(__file__, 120, 18), spfunc_531189, *[tuple_call_result_531193], **kwargs_531194)
        
        # Assigning a type to the variable 'res' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 12), 'res', spfunc_call_result_531195)
        # SSA join for if statement (line 104)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'res' (line 121)
        res_531196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 15), 'res')
        # Assigning a type to the variable 'stypy_return_type' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'stypy_return_type', res_531196)
        
        # ################# End of 'idmap(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'idmap' in the type store
        # Getting the type of 'stypy_return_type' (line 103)
        stypy_return_type_531197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_531197)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'idmap'
        return stypy_return_type_531197


    @norecursion
    def get_param_filter(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_param_filter'
        module_type_store = module_type_store.open_function_context('get_param_filter', 123, 4, False)
        # Assigning a type to the variable 'self' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _CDFData.get_param_filter.__dict__.__setitem__('stypy_localization', localization)
        _CDFData.get_param_filter.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _CDFData.get_param_filter.__dict__.__setitem__('stypy_type_store', module_type_store)
        _CDFData.get_param_filter.__dict__.__setitem__('stypy_function_name', '_CDFData.get_param_filter')
        _CDFData.get_param_filter.__dict__.__setitem__('stypy_param_names_list', [])
        _CDFData.get_param_filter.__dict__.__setitem__('stypy_varargs_param_name', None)
        _CDFData.get_param_filter.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _CDFData.get_param_filter.__dict__.__setitem__('stypy_call_defaults', defaults)
        _CDFData.get_param_filter.__dict__.__setitem__('stypy_call_varargs', varargs)
        _CDFData.get_param_filter.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _CDFData.get_param_filter.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_CDFData.get_param_filter', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_param_filter', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_param_filter(...)' code ##################

        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'self' (line 124)
        self_531198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 11), 'self')
        # Obtaining the member 'endpt_rtol' of a type (line 124)
        endpt_rtol_531199 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 11), self_531198, 'endpt_rtol')
        # Getting the type of 'None' (line 124)
        None_531200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 30), 'None')
        # Applying the binary operator 'is' (line 124)
        result_is__531201 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 11), 'is', endpt_rtol_531199, None_531200)
        
        
        # Getting the type of 'self' (line 124)
        self_531202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 39), 'self')
        # Obtaining the member 'endpt_atol' of a type (line 124)
        endpt_atol_531203 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 39), self_531202, 'endpt_atol')
        # Getting the type of 'None' (line 124)
        None_531204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 58), 'None')
        # Applying the binary operator 'is' (line 124)
        result_is__531205 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 39), 'is', endpt_atol_531203, None_531204)
        
        # Applying the binary operator 'and' (line 124)
        result_and_keyword_531206 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 11), 'and', result_is__531201, result_is__531205)
        
        # Testing the type of an if condition (line 124)
        if_condition_531207 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 124, 8), result_and_keyword_531206)
        # Assigning a type to the variable 'if_condition_531207' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'if_condition_531207', if_condition_531207)
        # SSA begins for if statement (line 124)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'None' (line 125)
        None_531208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 19), 'None')
        # Assigning a type to the variable 'stypy_return_type' (line 125)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 12), 'stypy_return_type', None_531208)
        # SSA join for if statement (line 124)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a List to a Name (line 127):
        
        # Assigning a List to a Name (line 127):
        
        # Obtaining an instance of the builtin type 'list' (line 127)
        list_531209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 127)
        
        # Assigning a type to the variable 'filters' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'filters', list_531209)
        
        
        # Call to zip(...): (line 128)
        # Processing the call arguments (line 128)
        # Getting the type of 'self' (line 128)
        self_531211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 36), 'self', False)
        # Obtaining the member 'endpt_rtol' of a type (line 128)
        endpt_rtol_531212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 36), self_531211, 'endpt_rtol')
        # Getting the type of 'self' (line 128)
        self_531213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 53), 'self', False)
        # Obtaining the member 'endpt_atol' of a type (line 128)
        endpt_atol_531214 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 53), self_531213, 'endpt_atol')
        # Getting the type of 'self' (line 128)
        self_531215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 70), 'self', False)
        # Obtaining the member 'argspec' of a type (line 128)
        argspec_531216 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 70), self_531215, 'argspec')
        # Processing the call keyword arguments (line 128)
        kwargs_531217 = {}
        # Getting the type of 'zip' (line 128)
        zip_531210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 32), 'zip', False)
        # Calling zip(args, kwargs) (line 128)
        zip_call_result_531218 = invoke(stypy.reporting.localization.Localization(__file__, 128, 32), zip_531210, *[endpt_rtol_531212, endpt_atol_531214, argspec_531216], **kwargs_531217)
        
        # Testing the type of a for loop iterable (line 128)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 128, 8), zip_call_result_531218)
        # Getting the type of the for loop variable (line 128)
        for_loop_var_531219 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 128, 8), zip_call_result_531218)
        # Assigning a type to the variable 'rtol' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'rtol', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 8), for_loop_var_531219))
        # Assigning a type to the variable 'atol' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'atol', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 8), for_loop_var_531219))
        # Assigning a type to the variable 'spec' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'spec', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 8), for_loop_var_531219))
        # SSA begins for a for statement (line 128)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'rtol' (line 129)
        rtol_531220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 15), 'rtol')
        # Getting the type of 'None' (line 129)
        None_531221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 23), 'None')
        # Applying the binary operator 'is' (line 129)
        result_is__531222 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 15), 'is', rtol_531220, None_531221)
        
        
        # Getting the type of 'atol' (line 129)
        atol_531223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 32), 'atol')
        # Getting the type of 'None' (line 129)
        None_531224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 40), 'None')
        # Applying the binary operator 'is' (line 129)
        result_is__531225 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 32), 'is', atol_531223, None_531224)
        
        # Applying the binary operator 'and' (line 129)
        result_and_keyword_531226 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 15), 'and', result_is__531222, result_is__531225)
        
        # Testing the type of an if condition (line 129)
        if_condition_531227 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 129, 12), result_and_keyword_531226)
        # Assigning a type to the variable 'if_condition_531227' (line 129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 12), 'if_condition_531227', if_condition_531227)
        # SSA begins for if statement (line 129)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 130)
        # Processing the call arguments (line 130)
        # Getting the type of 'None' (line 130)
        None_531230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 31), 'None', False)
        # Processing the call keyword arguments (line 130)
        kwargs_531231 = {}
        # Getting the type of 'filters' (line 130)
        filters_531228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 16), 'filters', False)
        # Obtaining the member 'append' of a type (line 130)
        append_531229 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 16), filters_531228, 'append')
        # Calling append(args, kwargs) (line 130)
        append_call_result_531232 = invoke(stypy.reporting.localization.Localization(__file__, 130, 16), append_531229, *[None_531230], **kwargs_531231)
        
        # SSA branch for the else part of an if statement (line 129)
        module_type_store.open_ssa_branch('else')
        
        # Type idiom detected: calculating its left and rigth part (line 132)
        # Getting the type of 'rtol' (line 132)
        rtol_531233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 17), 'rtol')
        # Getting the type of 'None' (line 132)
        None_531234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 25), 'None')
        
        (may_be_531235, more_types_in_union_531236) = may_be_none(rtol_531233, None_531234)

        if may_be_531235:

            if more_types_in_union_531236:
                # Runtime conditional SSA (line 132)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Num to a Name (line 133):
            
            # Assigning a Num to a Name (line 133):
            float_531237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 23), 'float')
            # Assigning a type to the variable 'rtol' (line 133)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 16), 'rtol', float_531237)

            if more_types_in_union_531236:
                # Runtime conditional SSA for else branch (line 132)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_531235) or more_types_in_union_531236):
            
            # Type idiom detected: calculating its left and rigth part (line 134)
            # Getting the type of 'atol' (line 134)
            atol_531238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 17), 'atol')
            # Getting the type of 'None' (line 134)
            None_531239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 25), 'None')
            
            (may_be_531240, more_types_in_union_531241) = may_be_none(atol_531238, None_531239)

            if may_be_531240:

                if more_types_in_union_531241:
                    # Runtime conditional SSA (line 134)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                
                # Assigning a Num to a Name (line 135):
                
                # Assigning a Num to a Name (line 135):
                float_531242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 23), 'float')
                # Assigning a type to the variable 'atol' (line 135)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 16), 'atol', float_531242)

                if more_types_in_union_531241:
                    # SSA join for if statement (line 134)
                    module_type_store = module_type_store.join_ssa_context()


            

            if (may_be_531235 and more_types_in_union_531236):
                # SSA join for if statement (line 132)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for if statement (line 129)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to append(...): (line 137)
        # Processing the call arguments (line 137)
        
        # Call to EndpointFilter(...): (line 137)
        # Processing the call arguments (line 137)
        # Getting the type of 'spec' (line 137)
        spec_531246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 42), 'spec', False)
        # Obtaining the member 'a' of a type (line 137)
        a_531247 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 42), spec_531246, 'a')
        # Getting the type of 'spec' (line 137)
        spec_531248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 50), 'spec', False)
        # Obtaining the member 'b' of a type (line 137)
        b_531249 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 50), spec_531248, 'b')
        # Getting the type of 'rtol' (line 137)
        rtol_531250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 58), 'rtol', False)
        # Getting the type of 'atol' (line 137)
        atol_531251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 64), 'atol', False)
        # Processing the call keyword arguments (line 137)
        kwargs_531252 = {}
        # Getting the type of 'EndpointFilter' (line 137)
        EndpointFilter_531245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 27), 'EndpointFilter', False)
        # Calling EndpointFilter(args, kwargs) (line 137)
        EndpointFilter_call_result_531253 = invoke(stypy.reporting.localization.Localization(__file__, 137, 27), EndpointFilter_531245, *[a_531247, b_531249, rtol_531250, atol_531251], **kwargs_531252)
        
        # Processing the call keyword arguments (line 137)
        kwargs_531254 = {}
        # Getting the type of 'filters' (line 137)
        filters_531243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 12), 'filters', False)
        # Obtaining the member 'append' of a type (line 137)
        append_531244 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 12), filters_531243, 'append')
        # Calling append(args, kwargs) (line 137)
        append_call_result_531255 = invoke(stypy.reporting.localization.Localization(__file__, 137, 12), append_531244, *[EndpointFilter_call_result_531253], **kwargs_531254)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'filters' (line 138)
        filters_531256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 15), 'filters')
        # Assigning a type to the variable 'stypy_return_type' (line 138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 8), 'stypy_return_type', filters_531256)
        
        # ################# End of 'get_param_filter(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_param_filter' in the type store
        # Getting the type of 'stypy_return_type' (line 123)
        stypy_return_type_531257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_531257)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_param_filter'
        return stypy_return_type_531257


    @norecursion
    def check(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'check'
        module_type_store = module_type_store.open_function_context('check', 140, 4, False)
        # Assigning a type to the variable 'self' (line 141)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _CDFData.check.__dict__.__setitem__('stypy_localization', localization)
        _CDFData.check.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _CDFData.check.__dict__.__setitem__('stypy_type_store', module_type_store)
        _CDFData.check.__dict__.__setitem__('stypy_function_name', '_CDFData.check')
        _CDFData.check.__dict__.__setitem__('stypy_param_names_list', [])
        _CDFData.check.__dict__.__setitem__('stypy_varargs_param_name', None)
        _CDFData.check.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _CDFData.check.__dict__.__setitem__('stypy_call_defaults', defaults)
        _CDFData.check.__dict__.__setitem__('stypy_call_varargs', varargs)
        _CDFData.check.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _CDFData.check.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_CDFData.check', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Call to a Name (line 142):
        
        # Assigning a Call to a Name (line 142):
        
        # Call to get_args(...): (line 142)
        # Processing the call arguments (line 142)
        # Getting the type of 'self' (line 142)
        self_531259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 24), 'self', False)
        # Obtaining the member 'argspec' of a type (line 142)
        argspec_531260 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 24), self_531259, 'argspec')
        # Getting the type of 'self' (line 142)
        self_531261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 38), 'self', False)
        # Obtaining the member 'n' of a type (line 142)
        n_531262 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 38), self_531261, 'n')
        # Processing the call keyword arguments (line 142)
        kwargs_531263 = {}
        # Getting the type of 'get_args' (line 142)
        get_args_531258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 15), 'get_args', False)
        # Calling get_args(args, kwargs) (line 142)
        get_args_call_result_531264 = invoke(stypy.reporting.localization.Localization(__file__, 142, 15), get_args_531258, *[argspec_531260, n_531262], **kwargs_531263)
        
        # Assigning a type to the variable 'args' (line 142)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'args', get_args_call_result_531264)
        
        # Assigning a Call to a Name (line 143):
        
        # Assigning a Call to a Name (line 143):
        
        # Call to get_param_filter(...): (line 143)
        # Processing the call keyword arguments (line 143)
        kwargs_531267 = {}
        # Getting the type of 'self' (line 143)
        self_531265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 23), 'self', False)
        # Obtaining the member 'get_param_filter' of a type (line 143)
        get_param_filter_531266 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 23), self_531265, 'get_param_filter')
        # Calling get_param_filter(args, kwargs) (line 143)
        get_param_filter_call_result_531268 = invoke(stypy.reporting.localization.Localization(__file__, 143, 23), get_param_filter_531266, *[], **kwargs_531267)
        
        # Assigning a type to the variable 'param_filter' (line 143)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 8), 'param_filter', get_param_filter_call_result_531268)
        
        # Assigning a Call to a Name (line 144):
        
        # Assigning a Call to a Name (line 144):
        
        # Call to tuple(...): (line 144)
        # Processing the call arguments (line 144)
        
        # Call to range(...): (line 144)
        # Processing the call arguments (line 144)
        
        # Obtaining the type of the subscript
        int_531271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 47), 'int')
        # Getting the type of 'args' (line 144)
        args_531272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 36), 'args', False)
        # Obtaining the member 'shape' of a type (line 144)
        shape_531273 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 36), args_531272, 'shape')
        # Obtaining the member '__getitem__' of a type (line 144)
        getitem___531274 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 36), shape_531273, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 144)
        subscript_call_result_531275 = invoke(stypy.reporting.localization.Localization(__file__, 144, 36), getitem___531274, int_531271)
        
        # Processing the call keyword arguments (line 144)
        kwargs_531276 = {}
        # Getting the type of 'range' (line 144)
        range_531270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 30), 'range', False)
        # Calling range(args, kwargs) (line 144)
        range_call_result_531277 = invoke(stypy.reporting.localization.Localization(__file__, 144, 30), range_531270, *[subscript_call_result_531275], **kwargs_531276)
        
        # Processing the call keyword arguments (line 144)
        kwargs_531278 = {}
        # Getting the type of 'tuple' (line 144)
        tuple_531269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 24), 'tuple', False)
        # Calling tuple(args, kwargs) (line 144)
        tuple_call_result_531279 = invoke(stypy.reporting.localization.Localization(__file__, 144, 24), tuple_531269, *[range_call_result_531277], **kwargs_531278)
        
        # Assigning a type to the variable 'param_columns' (line 144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'param_columns', tuple_call_result_531279)
        
        # Assigning a Subscript to a Name (line 145):
        
        # Assigning a Subscript to a Name (line 145):
        
        # Obtaining the type of the subscript
        int_531280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 36), 'int')
        # Getting the type of 'args' (line 145)
        args_531281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 25), 'args')
        # Obtaining the member 'shape' of a type (line 145)
        shape_531282 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 25), args_531281, 'shape')
        # Obtaining the member '__getitem__' of a type (line 145)
        getitem___531283 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 25), shape_531282, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 145)
        subscript_call_result_531284 = invoke(stypy.reporting.localization.Localization(__file__, 145, 25), getitem___531283, int_531280)
        
        # Assigning a type to the variable 'result_columns' (line 145)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 8), 'result_columns', subscript_call_result_531284)
        
        # Assigning a Call to a Name (line 146):
        
        # Assigning a Call to a Name (line 146):
        
        # Call to hstack(...): (line 146)
        # Processing the call arguments (line 146)
        
        # Obtaining an instance of the builtin type 'tuple' (line 146)
        tuple_531287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 146)
        # Adding element type (line 146)
        # Getting the type of 'args' (line 146)
        args_531288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 26), 'args', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 146, 26), tuple_531287, args_531288)
        # Adding element type (line 146)
        
        # Call to reshape(...): (line 146)
        # Processing the call arguments (line 146)
        
        # Obtaining the type of the subscript
        int_531296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 70), 'int')
        # Getting the type of 'args' (line 146)
        args_531297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 59), 'args', False)
        # Obtaining the member 'shape' of a type (line 146)
        shape_531298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 59), args_531297, 'shape')
        # Obtaining the member '__getitem__' of a type (line 146)
        getitem___531299 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 59), shape_531298, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 146)
        subscript_call_result_531300 = invoke(stypy.reporting.localization.Localization(__file__, 146, 59), getitem___531299, int_531296)
        
        int_531301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 74), 'int')
        # Processing the call keyword arguments (line 146)
        kwargs_531302 = {}
        
        # Obtaining the type of the subscript
        slice_531289 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 146, 32), None, None, None)
        # Getting the type of 'self' (line 146)
        self_531290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 39), 'self', False)
        # Obtaining the member 'index' of a type (line 146)
        index_531291 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 39), self_531290, 'index')
        # Getting the type of 'args' (line 146)
        args_531292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 32), 'args', False)
        # Obtaining the member '__getitem__' of a type (line 146)
        getitem___531293 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 32), args_531292, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 146)
        subscript_call_result_531294 = invoke(stypy.reporting.localization.Localization(__file__, 146, 32), getitem___531293, (slice_531289, index_531291))
        
        # Obtaining the member 'reshape' of a type (line 146)
        reshape_531295 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 32), subscript_call_result_531294, 'reshape')
        # Calling reshape(args, kwargs) (line 146)
        reshape_call_result_531303 = invoke(stypy.reporting.localization.Localization(__file__, 146, 32), reshape_531295, *[subscript_call_result_531300, int_531301], **kwargs_531302)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 146, 26), tuple_531287, reshape_call_result_531303)
        
        # Processing the call keyword arguments (line 146)
        kwargs_531304 = {}
        # Getting the type of 'np' (line 146)
        np_531285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 15), 'np', False)
        # Obtaining the member 'hstack' of a type (line 146)
        hstack_531286 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 15), np_531285, 'hstack')
        # Calling hstack(args, kwargs) (line 146)
        hstack_call_result_531305 = invoke(stypy.reporting.localization.Localization(__file__, 146, 15), hstack_531286, *[tuple_531287], **kwargs_531304)
        
        # Assigning a type to the variable 'args' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'args', hstack_call_result_531305)
        
        # Call to check(...): (line 147)
        # Processing the call keyword arguments (line 147)
        kwargs_531327 = {}
        
        # Call to FuncData(...): (line 147)
        # Processing the call arguments (line 147)
        # Getting the type of 'self' (line 147)
        self_531307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 17), 'self', False)
        # Obtaining the member 'idmap' of a type (line 147)
        idmap_531308 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 17), self_531307, 'idmap')
        # Getting the type of 'args' (line 147)
        args_531309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 29), 'args', False)
        # Processing the call keyword arguments (line 147)
        # Getting the type of 'param_columns' (line 148)
        param_columns_531310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 31), 'param_columns', False)
        keyword_531311 = param_columns_531310
        # Getting the type of 'result_columns' (line 148)
        result_columns_531312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 61), 'result_columns', False)
        keyword_531313 = result_columns_531312
        # Getting the type of 'self' (line 149)
        self_531314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 22), 'self', False)
        # Obtaining the member 'rtol' of a type (line 149)
        rtol_531315 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 22), self_531314, 'rtol')
        keyword_531316 = rtol_531315
        # Getting the type of 'self' (line 149)
        self_531317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 38), 'self', False)
        # Obtaining the member 'atol' of a type (line 149)
        atol_531318 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 38), self_531317, 'atol')
        keyword_531319 = atol_531318
        # Getting the type of 'False' (line 149)
        False_531320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 60), 'False', False)
        keyword_531321 = False_531320
        # Getting the type of 'param_filter' (line 150)
        param_filter_531322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 30), 'param_filter', False)
        keyword_531323 = param_filter_531322
        kwargs_531324 = {'param_filter': keyword_531323, 'param_columns': keyword_531311, 'vectorized': keyword_531321, 'rtol': keyword_531316, 'atol': keyword_531319, 'result_columns': keyword_531313}
        # Getting the type of 'FuncData' (line 147)
        FuncData_531306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'FuncData', False)
        # Calling FuncData(args, kwargs) (line 147)
        FuncData_call_result_531325 = invoke(stypy.reporting.localization.Localization(__file__, 147, 8), FuncData_531306, *[idmap_531308, args_531309], **kwargs_531324)
        
        # Obtaining the member 'check' of a type (line 147)
        check_531326 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 8), FuncData_call_result_531325, 'check')
        # Calling check(args, kwargs) (line 147)
        check_call_result_531328 = invoke(stypy.reporting.localization.Localization(__file__, 147, 8), check_531326, *[], **kwargs_531327)
        
        
        # ################# End of 'check(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'check' in the type store
        # Getting the type of 'stypy_return_type' (line 140)
        stypy_return_type_531329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_531329)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'check'
        return stypy_return_type_531329


# Assigning a type to the variable '_CDFData' (line 73)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 0), '_CDFData', _CDFData)

@norecursion
def _assert_inverts(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_assert_inverts'
    module_type_store = module_type_store.open_function_context('_assert_inverts', 153, 0, False)
    
    # Passed parameters checking function
    _assert_inverts.stypy_localization = localization
    _assert_inverts.stypy_type_of_self = None
    _assert_inverts.stypy_type_store = module_type_store
    _assert_inverts.stypy_function_name = '_assert_inverts'
    _assert_inverts.stypy_param_names_list = []
    _assert_inverts.stypy_varargs_param_name = 'a'
    _assert_inverts.stypy_kwargs_param_name = 'kw'
    _assert_inverts.stypy_call_defaults = defaults
    _assert_inverts.stypy_call_varargs = varargs
    _assert_inverts.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_assert_inverts', [], 'a', 'kw', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_assert_inverts', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_assert_inverts(...)' code ##################

    
    # Assigning a Call to a Name (line 154):
    
    # Assigning a Call to a Name (line 154):
    
    # Call to _CDFData(...): (line 154)
    # Getting the type of 'a' (line 154)
    a_531331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 18), 'a', False)
    # Processing the call keyword arguments (line 154)
    # Getting the type of 'kw' (line 154)
    kw_531332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 23), 'kw', False)
    kwargs_531333 = {'kw_531332': kw_531332}
    # Getting the type of '_CDFData' (line 154)
    _CDFData_531330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), '_CDFData', False)
    # Calling _CDFData(args, kwargs) (line 154)
    _CDFData_call_result_531334 = invoke(stypy.reporting.localization.Localization(__file__, 154, 8), _CDFData_531330, *[a_531331], **kwargs_531333)
    
    # Assigning a type to the variable 'd' (line 154)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 4), 'd', _CDFData_call_result_531334)
    
    # Call to check(...): (line 155)
    # Processing the call keyword arguments (line 155)
    kwargs_531337 = {}
    # Getting the type of 'd' (line 155)
    d_531335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 4), 'd', False)
    # Obtaining the member 'check' of a type (line 155)
    check_531336 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 4), d_531335, 'check')
    # Calling check(args, kwargs) (line 155)
    check_call_result_531338 = invoke(stypy.reporting.localization.Localization(__file__, 155, 4), check_531336, *[], **kwargs_531337)
    
    
    # ################# End of '_assert_inverts(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_assert_inverts' in the type store
    # Getting the type of 'stypy_return_type' (line 153)
    stypy_return_type_531339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_531339)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_assert_inverts'
    return stypy_return_type_531339

# Assigning a type to the variable '_assert_inverts' (line 153)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 0), '_assert_inverts', _assert_inverts)

@norecursion
def _binomial_cdf(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_binomial_cdf'
    module_type_store = module_type_store.open_function_context('_binomial_cdf', 158, 0, False)
    
    # Passed parameters checking function
    _binomial_cdf.stypy_localization = localization
    _binomial_cdf.stypy_type_of_self = None
    _binomial_cdf.stypy_type_store = module_type_store
    _binomial_cdf.stypy_function_name = '_binomial_cdf'
    _binomial_cdf.stypy_param_names_list = ['k', 'n', 'p']
    _binomial_cdf.stypy_varargs_param_name = None
    _binomial_cdf.stypy_kwargs_param_name = None
    _binomial_cdf.stypy_call_defaults = defaults
    _binomial_cdf.stypy_call_varargs = varargs
    _binomial_cdf.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_binomial_cdf', ['k', 'n', 'p'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_binomial_cdf', localization, ['k', 'n', 'p'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_binomial_cdf(...)' code ##################

    
    # Assigning a Tuple to a Tuple (line 159):
    
    # Assigning a Call to a Name (line 159):
    
    # Call to mpf(...): (line 159)
    # Processing the call arguments (line 159)
    # Getting the type of 'k' (line 159)
    k_531342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 25), 'k', False)
    # Processing the call keyword arguments (line 159)
    kwargs_531343 = {}
    # Getting the type of 'mpmath' (line 159)
    mpmath_531340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 14), 'mpmath', False)
    # Obtaining the member 'mpf' of a type (line 159)
    mpf_531341 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 14), mpmath_531340, 'mpf')
    # Calling mpf(args, kwargs) (line 159)
    mpf_call_result_531344 = invoke(stypy.reporting.localization.Localization(__file__, 159, 14), mpf_531341, *[k_531342], **kwargs_531343)
    
    # Assigning a type to the variable 'tuple_assignment_530878' (line 159)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 4), 'tuple_assignment_530878', mpf_call_result_531344)
    
    # Assigning a Call to a Name (line 159):
    
    # Call to mpf(...): (line 159)
    # Processing the call arguments (line 159)
    # Getting the type of 'n' (line 159)
    n_531347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 40), 'n', False)
    # Processing the call keyword arguments (line 159)
    kwargs_531348 = {}
    # Getting the type of 'mpmath' (line 159)
    mpmath_531345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 29), 'mpmath', False)
    # Obtaining the member 'mpf' of a type (line 159)
    mpf_531346 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 29), mpmath_531345, 'mpf')
    # Calling mpf(args, kwargs) (line 159)
    mpf_call_result_531349 = invoke(stypy.reporting.localization.Localization(__file__, 159, 29), mpf_531346, *[n_531347], **kwargs_531348)
    
    # Assigning a type to the variable 'tuple_assignment_530879' (line 159)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 4), 'tuple_assignment_530879', mpf_call_result_531349)
    
    # Assigning a Call to a Name (line 159):
    
    # Call to mpf(...): (line 159)
    # Processing the call arguments (line 159)
    # Getting the type of 'p' (line 159)
    p_531352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 55), 'p', False)
    # Processing the call keyword arguments (line 159)
    kwargs_531353 = {}
    # Getting the type of 'mpmath' (line 159)
    mpmath_531350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 44), 'mpmath', False)
    # Obtaining the member 'mpf' of a type (line 159)
    mpf_531351 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 44), mpmath_531350, 'mpf')
    # Calling mpf(args, kwargs) (line 159)
    mpf_call_result_531354 = invoke(stypy.reporting.localization.Localization(__file__, 159, 44), mpf_531351, *[p_531352], **kwargs_531353)
    
    # Assigning a type to the variable 'tuple_assignment_530880' (line 159)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 4), 'tuple_assignment_530880', mpf_call_result_531354)
    
    # Assigning a Name to a Name (line 159):
    # Getting the type of 'tuple_assignment_530878' (line 159)
    tuple_assignment_530878_531355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 4), 'tuple_assignment_530878')
    # Assigning a type to the variable 'k' (line 159)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 4), 'k', tuple_assignment_530878_531355)
    
    # Assigning a Name to a Name (line 159):
    # Getting the type of 'tuple_assignment_530879' (line 159)
    tuple_assignment_530879_531356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 4), 'tuple_assignment_530879')
    # Assigning a type to the variable 'n' (line 159)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 7), 'n', tuple_assignment_530879_531356)
    
    # Assigning a Name to a Name (line 159):
    # Getting the type of 'tuple_assignment_530880' (line 159)
    tuple_assignment_530880_531357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 4), 'tuple_assignment_530880')
    # Assigning a type to the variable 'p' (line 159)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 10), 'p', tuple_assignment_530880_531357)
    
    
    # Getting the type of 'k' (line 160)
    k_531358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 7), 'k')
    int_531359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 12), 'int')
    # Applying the binary operator '<=' (line 160)
    result_le_531360 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 7), '<=', k_531358, int_531359)
    
    # Testing the type of an if condition (line 160)
    if_condition_531361 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 160, 4), result_le_531360)
    # Assigning a type to the variable 'if_condition_531361' (line 160)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 4), 'if_condition_531361', if_condition_531361)
    # SSA begins for if statement (line 160)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to mpf(...): (line 161)
    # Processing the call arguments (line 161)
    int_531364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 26), 'int')
    # Processing the call keyword arguments (line 161)
    kwargs_531365 = {}
    # Getting the type of 'mpmath' (line 161)
    mpmath_531362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 15), 'mpmath', False)
    # Obtaining the member 'mpf' of a type (line 161)
    mpf_531363 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 15), mpmath_531362, 'mpf')
    # Calling mpf(args, kwargs) (line 161)
    mpf_call_result_531366 = invoke(stypy.reporting.localization.Localization(__file__, 161, 15), mpf_531363, *[int_531364], **kwargs_531365)
    
    # Assigning a type to the variable 'stypy_return_type' (line 161)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 8), 'stypy_return_type', mpf_call_result_531366)
    # SSA branch for the else part of an if statement (line 160)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'k' (line 162)
    k_531367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 9), 'k')
    # Getting the type of 'n' (line 162)
    n_531368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 14), 'n')
    # Applying the binary operator '>=' (line 162)
    result_ge_531369 = python_operator(stypy.reporting.localization.Localization(__file__, 162, 9), '>=', k_531367, n_531368)
    
    # Testing the type of an if condition (line 162)
    if_condition_531370 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 162, 9), result_ge_531369)
    # Assigning a type to the variable 'if_condition_531370' (line 162)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 9), 'if_condition_531370', if_condition_531370)
    # SSA begins for if statement (line 162)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to mpf(...): (line 163)
    # Processing the call arguments (line 163)
    int_531373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 26), 'int')
    # Processing the call keyword arguments (line 163)
    kwargs_531374 = {}
    # Getting the type of 'mpmath' (line 163)
    mpmath_531371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 15), 'mpmath', False)
    # Obtaining the member 'mpf' of a type (line 163)
    mpf_531372 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 15), mpmath_531371, 'mpf')
    # Calling mpf(args, kwargs) (line 163)
    mpf_call_result_531375 = invoke(stypy.reporting.localization.Localization(__file__, 163, 15), mpf_531372, *[int_531373], **kwargs_531374)
    
    # Assigning a type to the variable 'stypy_return_type' (line 163)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 8), 'stypy_return_type', mpf_call_result_531375)
    # SSA join for if statement (line 162)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 160)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 165):
    
    # Assigning a Call to a Name (line 165):
    
    # Call to fsub(...): (line 165)
    # Processing the call arguments (line 165)
    int_531378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 24), 'int')
    # Getting the type of 'p' (line 165)
    p_531379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 27), 'p', False)
    # Processing the call keyword arguments (line 165)
    # Getting the type of 'True' (line 165)
    True_531380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 36), 'True', False)
    keyword_531381 = True_531380
    kwargs_531382 = {'exact': keyword_531381}
    # Getting the type of 'mpmath' (line 165)
    mpmath_531376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 12), 'mpmath', False)
    # Obtaining the member 'fsub' of a type (line 165)
    fsub_531377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 12), mpmath_531376, 'fsub')
    # Calling fsub(args, kwargs) (line 165)
    fsub_call_result_531383 = invoke(stypy.reporting.localization.Localization(__file__, 165, 12), fsub_531377, *[int_531378, p_531379], **kwargs_531382)
    
    # Assigning a type to the variable 'onemp' (line 165)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 4), 'onemp', fsub_call_result_531383)
    
    # Call to betainc(...): (line 166)
    # Processing the call arguments (line 166)
    # Getting the type of 'n' (line 166)
    n_531386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 26), 'n', False)
    # Getting the type of 'k' (line 166)
    k_531387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 30), 'k', False)
    # Applying the binary operator '-' (line 166)
    result_sub_531388 = python_operator(stypy.reporting.localization.Localization(__file__, 166, 26), '-', n_531386, k_531387)
    
    # Getting the type of 'k' (line 166)
    k_531389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 33), 'k', False)
    int_531390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 37), 'int')
    # Applying the binary operator '+' (line 166)
    result_add_531391 = python_operator(stypy.reporting.localization.Localization(__file__, 166, 33), '+', k_531389, int_531390)
    
    # Processing the call keyword arguments (line 166)
    # Getting the type of 'onemp' (line 166)
    onemp_531392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 43), 'onemp', False)
    keyword_531393 = onemp_531392
    # Getting the type of 'True' (line 166)
    True_531394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 62), 'True', False)
    keyword_531395 = True_531394
    kwargs_531396 = {'x2': keyword_531393, 'regularized': keyword_531395}
    # Getting the type of 'mpmath' (line 166)
    mpmath_531384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 11), 'mpmath', False)
    # Obtaining the member 'betainc' of a type (line 166)
    betainc_531385 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 11), mpmath_531384, 'betainc')
    # Calling betainc(args, kwargs) (line 166)
    betainc_call_result_531397 = invoke(stypy.reporting.localization.Localization(__file__, 166, 11), betainc_531385, *[result_sub_531388, result_add_531391], **kwargs_531396)
    
    # Assigning a type to the variable 'stypy_return_type' (line 166)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 4), 'stypy_return_type', betainc_call_result_531397)
    
    # ################# End of '_binomial_cdf(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_binomial_cdf' in the type store
    # Getting the type of 'stypy_return_type' (line 158)
    stypy_return_type_531398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_531398)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_binomial_cdf'
    return stypy_return_type_531398

# Assigning a type to the variable '_binomial_cdf' (line 158)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 0), '_binomial_cdf', _binomial_cdf)

@norecursion
def _f_cdf(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_f_cdf'
    module_type_store = module_type_store.open_function_context('_f_cdf', 169, 0, False)
    
    # Passed parameters checking function
    _f_cdf.stypy_localization = localization
    _f_cdf.stypy_type_of_self = None
    _f_cdf.stypy_type_store = module_type_store
    _f_cdf.stypy_function_name = '_f_cdf'
    _f_cdf.stypy_param_names_list = ['dfn', 'dfd', 'x']
    _f_cdf.stypy_varargs_param_name = None
    _f_cdf.stypy_kwargs_param_name = None
    _f_cdf.stypy_call_defaults = defaults
    _f_cdf.stypy_call_varargs = varargs
    _f_cdf.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_f_cdf', ['dfn', 'dfd', 'x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_f_cdf', localization, ['dfn', 'dfd', 'x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_f_cdf(...)' code ##################

    
    
    # Getting the type of 'x' (line 170)
    x_531399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 7), 'x')
    int_531400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 11), 'int')
    # Applying the binary operator '<' (line 170)
    result_lt_531401 = python_operator(stypy.reporting.localization.Localization(__file__, 170, 7), '<', x_531399, int_531400)
    
    # Testing the type of an if condition (line 170)
    if_condition_531402 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 170, 4), result_lt_531401)
    # Assigning a type to the variable 'if_condition_531402' (line 170)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 4), 'if_condition_531402', if_condition_531402)
    # SSA begins for if statement (line 170)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to mpf(...): (line 171)
    # Processing the call arguments (line 171)
    int_531405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 26), 'int')
    # Processing the call keyword arguments (line 171)
    kwargs_531406 = {}
    # Getting the type of 'mpmath' (line 171)
    mpmath_531403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 15), 'mpmath', False)
    # Obtaining the member 'mpf' of a type (line 171)
    mpf_531404 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 15), mpmath_531403, 'mpf')
    # Calling mpf(args, kwargs) (line 171)
    mpf_call_result_531407 = invoke(stypy.reporting.localization.Localization(__file__, 171, 15), mpf_531404, *[int_531405], **kwargs_531406)
    
    # Assigning a type to the variable 'stypy_return_type' (line 171)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 8), 'stypy_return_type', mpf_call_result_531407)
    # SSA join for if statement (line 170)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Tuple to a Tuple (line 172):
    
    # Assigning a Call to a Name (line 172):
    
    # Call to mpf(...): (line 172)
    # Processing the call arguments (line 172)
    # Getting the type of 'dfn' (line 172)
    dfn_531410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 29), 'dfn', False)
    # Processing the call keyword arguments (line 172)
    kwargs_531411 = {}
    # Getting the type of 'mpmath' (line 172)
    mpmath_531408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 18), 'mpmath', False)
    # Obtaining the member 'mpf' of a type (line 172)
    mpf_531409 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 18), mpmath_531408, 'mpf')
    # Calling mpf(args, kwargs) (line 172)
    mpf_call_result_531412 = invoke(stypy.reporting.localization.Localization(__file__, 172, 18), mpf_531409, *[dfn_531410], **kwargs_531411)
    
    # Assigning a type to the variable 'tuple_assignment_530881' (line 172)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 4), 'tuple_assignment_530881', mpf_call_result_531412)
    
    # Assigning a Call to a Name (line 172):
    
    # Call to mpf(...): (line 172)
    # Processing the call arguments (line 172)
    # Getting the type of 'dfd' (line 172)
    dfd_531415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 46), 'dfd', False)
    # Processing the call keyword arguments (line 172)
    kwargs_531416 = {}
    # Getting the type of 'mpmath' (line 172)
    mpmath_531413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 35), 'mpmath', False)
    # Obtaining the member 'mpf' of a type (line 172)
    mpf_531414 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 35), mpmath_531413, 'mpf')
    # Calling mpf(args, kwargs) (line 172)
    mpf_call_result_531417 = invoke(stypy.reporting.localization.Localization(__file__, 172, 35), mpf_531414, *[dfd_531415], **kwargs_531416)
    
    # Assigning a type to the variable 'tuple_assignment_530882' (line 172)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 4), 'tuple_assignment_530882', mpf_call_result_531417)
    
    # Assigning a Call to a Name (line 172):
    
    # Call to mpf(...): (line 172)
    # Processing the call arguments (line 172)
    # Getting the type of 'x' (line 172)
    x_531420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 63), 'x', False)
    # Processing the call keyword arguments (line 172)
    kwargs_531421 = {}
    # Getting the type of 'mpmath' (line 172)
    mpmath_531418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 52), 'mpmath', False)
    # Obtaining the member 'mpf' of a type (line 172)
    mpf_531419 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 52), mpmath_531418, 'mpf')
    # Calling mpf(args, kwargs) (line 172)
    mpf_call_result_531422 = invoke(stypy.reporting.localization.Localization(__file__, 172, 52), mpf_531419, *[x_531420], **kwargs_531421)
    
    # Assigning a type to the variable 'tuple_assignment_530883' (line 172)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 4), 'tuple_assignment_530883', mpf_call_result_531422)
    
    # Assigning a Name to a Name (line 172):
    # Getting the type of 'tuple_assignment_530881' (line 172)
    tuple_assignment_530881_531423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 4), 'tuple_assignment_530881')
    # Assigning a type to the variable 'dfn' (line 172)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 4), 'dfn', tuple_assignment_530881_531423)
    
    # Assigning a Name to a Name (line 172):
    # Getting the type of 'tuple_assignment_530882' (line 172)
    tuple_assignment_530882_531424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 4), 'tuple_assignment_530882')
    # Assigning a type to the variable 'dfd' (line 172)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 9), 'dfd', tuple_assignment_530882_531424)
    
    # Assigning a Name to a Name (line 172):
    # Getting the type of 'tuple_assignment_530883' (line 172)
    tuple_assignment_530883_531425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 4), 'tuple_assignment_530883')
    # Assigning a type to the variable 'x' (line 172)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 14), 'x', tuple_assignment_530883_531425)
    
    # Assigning a BinOp to a Name (line 173):
    
    # Assigning a BinOp to a Name (line 173):
    # Getting the type of 'dfn' (line 173)
    dfn_531426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 9), 'dfn')
    # Getting the type of 'x' (line 173)
    x_531427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 13), 'x')
    # Applying the binary operator '*' (line 173)
    result_mul_531428 = python_operator(stypy.reporting.localization.Localization(__file__, 173, 9), '*', dfn_531426, x_531427)
    
    # Getting the type of 'dfn' (line 173)
    dfn_531429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 16), 'dfn')
    # Getting the type of 'x' (line 173)
    x_531430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 20), 'x')
    # Applying the binary operator '*' (line 173)
    result_mul_531431 = python_operator(stypy.reporting.localization.Localization(__file__, 173, 16), '*', dfn_531429, x_531430)
    
    # Getting the type of 'dfd' (line 173)
    dfd_531432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 24), 'dfd')
    # Applying the binary operator '+' (line 173)
    result_add_531433 = python_operator(stypy.reporting.localization.Localization(__file__, 173, 16), '+', result_mul_531431, dfd_531432)
    
    # Applying the binary operator 'div' (line 173)
    result_div_531434 = python_operator(stypy.reporting.localization.Localization(__file__, 173, 14), 'div', result_mul_531428, result_add_531433)
    
    # Assigning a type to the variable 'ub' (line 173)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 4), 'ub', result_div_531434)
    
    # Assigning a Call to a Name (line 174):
    
    # Assigning a Call to a Name (line 174):
    
    # Call to betainc(...): (line 174)
    # Processing the call arguments (line 174)
    # Getting the type of 'dfn' (line 174)
    dfn_531437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 25), 'dfn', False)
    int_531438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 29), 'int')
    # Applying the binary operator 'div' (line 174)
    result_div_531439 = python_operator(stypy.reporting.localization.Localization(__file__, 174, 25), 'div', dfn_531437, int_531438)
    
    # Getting the type of 'dfd' (line 174)
    dfd_531440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 32), 'dfd', False)
    int_531441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 36), 'int')
    # Applying the binary operator 'div' (line 174)
    result_div_531442 = python_operator(stypy.reporting.localization.Localization(__file__, 174, 32), 'div', dfd_531440, int_531441)
    
    # Processing the call keyword arguments (line 174)
    # Getting the type of 'ub' (line 174)
    ub_531443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 42), 'ub', False)
    keyword_531444 = ub_531443
    # Getting the type of 'True' (line 174)
    True_531445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 58), 'True', False)
    keyword_531446 = True_531445
    kwargs_531447 = {'x2': keyword_531444, 'regularized': keyword_531446}
    # Getting the type of 'mpmath' (line 174)
    mpmath_531435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 10), 'mpmath', False)
    # Obtaining the member 'betainc' of a type (line 174)
    betainc_531436 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 10), mpmath_531435, 'betainc')
    # Calling betainc(args, kwargs) (line 174)
    betainc_call_result_531448 = invoke(stypy.reporting.localization.Localization(__file__, 174, 10), betainc_531436, *[result_div_531439, result_div_531442], **kwargs_531447)
    
    # Assigning a type to the variable 'res' (line 174)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 4), 'res', betainc_call_result_531448)
    # Getting the type of 'res' (line 175)
    res_531449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 11), 'res')
    # Assigning a type to the variable 'stypy_return_type' (line 175)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 4), 'stypy_return_type', res_531449)
    
    # ################# End of '_f_cdf(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_f_cdf' in the type store
    # Getting the type of 'stypy_return_type' (line 169)
    stypy_return_type_531450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_531450)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_f_cdf'
    return stypy_return_type_531450

# Assigning a type to the variable '_f_cdf' (line 169)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 0), '_f_cdf', _f_cdf)

@norecursion
def _student_t_cdf(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 178)
    None_531451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 30), 'None')
    defaults = [None_531451]
    # Create a new context for function '_student_t_cdf'
    module_type_store = module_type_store.open_function_context('_student_t_cdf', 178, 0, False)
    
    # Passed parameters checking function
    _student_t_cdf.stypy_localization = localization
    _student_t_cdf.stypy_type_of_self = None
    _student_t_cdf.stypy_type_store = module_type_store
    _student_t_cdf.stypy_function_name = '_student_t_cdf'
    _student_t_cdf.stypy_param_names_list = ['df', 't', 'dps']
    _student_t_cdf.stypy_varargs_param_name = None
    _student_t_cdf.stypy_kwargs_param_name = None
    _student_t_cdf.stypy_call_defaults = defaults
    _student_t_cdf.stypy_call_varargs = varargs
    _student_t_cdf.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_student_t_cdf', ['df', 't', 'dps'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_student_t_cdf', localization, ['df', 't', 'dps'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_student_t_cdf(...)' code ##################

    
    # Type idiom detected: calculating its left and rigth part (line 179)
    # Getting the type of 'dps' (line 179)
    dps_531452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 7), 'dps')
    # Getting the type of 'None' (line 179)
    None_531453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 14), 'None')
    
    (may_be_531454, more_types_in_union_531455) = may_be_none(dps_531452, None_531453)

    if may_be_531454:

        if more_types_in_union_531455:
            # Runtime conditional SSA (line 179)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Attribute to a Name (line 180):
        
        # Assigning a Attribute to a Name (line 180):
        # Getting the type of 'mpmath' (line 180)
        mpmath_531456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 14), 'mpmath')
        # Obtaining the member 'mp' of a type (line 180)
        mp_531457 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 14), mpmath_531456, 'mp')
        # Obtaining the member 'dps' of a type (line 180)
        dps_531458 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 14), mp_531457, 'dps')
        # Assigning a type to the variable 'dps' (line 180)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 8), 'dps', dps_531458)

        if more_types_in_union_531455:
            # SSA join for if statement (line 179)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Call to workdps(...): (line 181)
    # Processing the call arguments (line 181)
    # Getting the type of 'dps' (line 181)
    dps_531461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 24), 'dps', False)
    # Processing the call keyword arguments (line 181)
    kwargs_531462 = {}
    # Getting the type of 'mpmath' (line 181)
    mpmath_531459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 9), 'mpmath', False)
    # Obtaining the member 'workdps' of a type (line 181)
    workdps_531460 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 9), mpmath_531459, 'workdps')
    # Calling workdps(args, kwargs) (line 181)
    workdps_call_result_531463 = invoke(stypy.reporting.localization.Localization(__file__, 181, 9), workdps_531460, *[dps_531461], **kwargs_531462)
    
    with_531464 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 181, 9), workdps_call_result_531463, 'with parameter', '__enter__', '__exit__')

    if with_531464:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 181)
        enter___531465 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 9), workdps_call_result_531463, '__enter__')
        with_enter_531466 = invoke(stypy.reporting.localization.Localization(__file__, 181, 9), enter___531465)
        
        # Assigning a Tuple to a Tuple (line 182):
        
        # Assigning a Call to a Name (line 182):
        
        # Call to mpf(...): (line 182)
        # Processing the call arguments (line 182)
        # Getting the type of 'df' (line 182)
        df_531469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 27), 'df', False)
        # Processing the call keyword arguments (line 182)
        kwargs_531470 = {}
        # Getting the type of 'mpmath' (line 182)
        mpmath_531467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 16), 'mpmath', False)
        # Obtaining the member 'mpf' of a type (line 182)
        mpf_531468 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 16), mpmath_531467, 'mpf')
        # Calling mpf(args, kwargs) (line 182)
        mpf_call_result_531471 = invoke(stypy.reporting.localization.Localization(__file__, 182, 16), mpf_531468, *[df_531469], **kwargs_531470)
        
        # Assigning a type to the variable 'tuple_assignment_530884' (line 182)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'tuple_assignment_530884', mpf_call_result_531471)
        
        # Assigning a Call to a Name (line 182):
        
        # Call to mpf(...): (line 182)
        # Processing the call arguments (line 182)
        # Getting the type of 't' (line 182)
        t_531474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 43), 't', False)
        # Processing the call keyword arguments (line 182)
        kwargs_531475 = {}
        # Getting the type of 'mpmath' (line 182)
        mpmath_531472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 32), 'mpmath', False)
        # Obtaining the member 'mpf' of a type (line 182)
        mpf_531473 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 32), mpmath_531472, 'mpf')
        # Calling mpf(args, kwargs) (line 182)
        mpf_call_result_531476 = invoke(stypy.reporting.localization.Localization(__file__, 182, 32), mpf_531473, *[t_531474], **kwargs_531475)
        
        # Assigning a type to the variable 'tuple_assignment_530885' (line 182)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'tuple_assignment_530885', mpf_call_result_531476)
        
        # Assigning a Name to a Name (line 182):
        # Getting the type of 'tuple_assignment_530884' (line 182)
        tuple_assignment_530884_531477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'tuple_assignment_530884')
        # Assigning a type to the variable 'df' (line 182)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'df', tuple_assignment_530884_531477)
        
        # Assigning a Name to a Name (line 182):
        # Getting the type of 'tuple_assignment_530885' (line 182)
        tuple_assignment_530885_531478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'tuple_assignment_530885')
        # Assigning a type to the variable 't' (line 182)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 12), 't', tuple_assignment_530885_531478)
        
        # Assigning a Call to a Name (line 183):
        
        # Assigning a Call to a Name (line 183):
        
        # Call to hyp2f1(...): (line 183)
        # Processing the call arguments (line 183)
        float_531481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 28), 'float')
        float_531482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 33), 'float')
        # Getting the type of 'df' (line 183)
        df_531483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 38), 'df', False)
        int_531484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 43), 'int')
        # Applying the binary operator '+' (line 183)
        result_add_531485 = python_operator(stypy.reporting.localization.Localization(__file__, 183, 38), '+', df_531483, int_531484)
        
        # Applying the binary operator '*' (line 183)
        result_mul_531486 = python_operator(stypy.reporting.localization.Localization(__file__, 183, 33), '*', float_531482, result_add_531485)
        
        float_531487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 47), 'float')
        
        # Getting the type of 't' (line 183)
        t_531488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 53), 't', False)
        int_531489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 56), 'int')
        # Applying the binary operator '**' (line 183)
        result_pow_531490 = python_operator(stypy.reporting.localization.Localization(__file__, 183, 53), '**', t_531488, int_531489)
        
        # Applying the 'usub' unary operator (line 183)
        result___neg___531491 = python_operator(stypy.reporting.localization.Localization(__file__, 183, 52), 'usub', result_pow_531490)
        
        # Getting the type of 'df' (line 183)
        df_531492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 58), 'df', False)
        # Applying the binary operator 'div' (line 183)
        result_div_531493 = python_operator(stypy.reporting.localization.Localization(__file__, 183, 52), 'div', result___neg___531491, df_531492)
        
        # Processing the call keyword arguments (line 183)
        kwargs_531494 = {}
        # Getting the type of 'mpmath' (line 183)
        mpmath_531479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 14), 'mpmath', False)
        # Obtaining the member 'hyp2f1' of a type (line 183)
        hyp2f1_531480 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 14), mpmath_531479, 'hyp2f1')
        # Calling hyp2f1(args, kwargs) (line 183)
        hyp2f1_call_result_531495 = invoke(stypy.reporting.localization.Localization(__file__, 183, 14), hyp2f1_531480, *[float_531481, result_mul_531486, float_531487, result_div_531493], **kwargs_531494)
        
        # Assigning a type to the variable 'fac' (line 183)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 8), 'fac', hyp2f1_call_result_531495)
        
        # Getting the type of 'fac' (line 184)
        fac_531496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'fac')
        # Getting the type of 't' (line 184)
        t_531497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 15), 't')
        
        # Call to gamma(...): (line 184)
        # Processing the call arguments (line 184)
        float_531500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 30), 'float')
        # Getting the type of 'df' (line 184)
        df_531501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 35), 'df', False)
        int_531502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 40), 'int')
        # Applying the binary operator '+' (line 184)
        result_add_531503 = python_operator(stypy.reporting.localization.Localization(__file__, 184, 35), '+', df_531501, int_531502)
        
        # Applying the binary operator '*' (line 184)
        result_mul_531504 = python_operator(stypy.reporting.localization.Localization(__file__, 184, 30), '*', float_531500, result_add_531503)
        
        # Processing the call keyword arguments (line 184)
        kwargs_531505 = {}
        # Getting the type of 'mpmath' (line 184)
        mpmath_531498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 17), 'mpmath', False)
        # Obtaining the member 'gamma' of a type (line 184)
        gamma_531499 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 17), mpmath_531498, 'gamma')
        # Calling gamma(args, kwargs) (line 184)
        gamma_call_result_531506 = invoke(stypy.reporting.localization.Localization(__file__, 184, 17), gamma_531499, *[result_mul_531504], **kwargs_531505)
        
        # Applying the binary operator '*' (line 184)
        result_mul_531507 = python_operator(stypy.reporting.localization.Localization(__file__, 184, 15), '*', t_531497, gamma_call_result_531506)
        
        # Applying the binary operator '*=' (line 184)
        result_imul_531508 = python_operator(stypy.reporting.localization.Localization(__file__, 184, 8), '*=', fac_531496, result_mul_531507)
        # Assigning a type to the variable 'fac' (line 184)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'fac', result_imul_531508)
        
        
        # Getting the type of 'fac' (line 185)
        fac_531509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 8), 'fac')
        
        # Call to sqrt(...): (line 185)
        # Processing the call arguments (line 185)
        # Getting the type of 'mpmath' (line 185)
        mpmath_531512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 27), 'mpmath', False)
        # Obtaining the member 'pi' of a type (line 185)
        pi_531513 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 27), mpmath_531512, 'pi')
        # Getting the type of 'df' (line 185)
        df_531514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 37), 'df', False)
        # Applying the binary operator '*' (line 185)
        result_mul_531515 = python_operator(stypy.reporting.localization.Localization(__file__, 185, 27), '*', pi_531513, df_531514)
        
        # Processing the call keyword arguments (line 185)
        kwargs_531516 = {}
        # Getting the type of 'mpmath' (line 185)
        mpmath_531510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 15), 'mpmath', False)
        # Obtaining the member 'sqrt' of a type (line 185)
        sqrt_531511 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 15), mpmath_531510, 'sqrt')
        # Calling sqrt(args, kwargs) (line 185)
        sqrt_call_result_531517 = invoke(stypy.reporting.localization.Localization(__file__, 185, 15), sqrt_531511, *[result_mul_531515], **kwargs_531516)
        
        
        # Call to gamma(...): (line 185)
        # Processing the call arguments (line 185)
        float_531520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 54), 'float')
        # Getting the type of 'df' (line 185)
        df_531521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 58), 'df', False)
        # Applying the binary operator '*' (line 185)
        result_mul_531522 = python_operator(stypy.reporting.localization.Localization(__file__, 185, 54), '*', float_531520, df_531521)
        
        # Processing the call keyword arguments (line 185)
        kwargs_531523 = {}
        # Getting the type of 'mpmath' (line 185)
        mpmath_531518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 41), 'mpmath', False)
        # Obtaining the member 'gamma' of a type (line 185)
        gamma_531519 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 41), mpmath_531518, 'gamma')
        # Calling gamma(args, kwargs) (line 185)
        gamma_call_result_531524 = invoke(stypy.reporting.localization.Localization(__file__, 185, 41), gamma_531519, *[result_mul_531522], **kwargs_531523)
        
        # Applying the binary operator '*' (line 185)
        result_mul_531525 = python_operator(stypy.reporting.localization.Localization(__file__, 185, 15), '*', sqrt_call_result_531517, gamma_call_result_531524)
        
        # Applying the binary operator 'div=' (line 185)
        result_div_531526 = python_operator(stypy.reporting.localization.Localization(__file__, 185, 8), 'div=', fac_531509, result_mul_531525)
        # Assigning a type to the variable 'fac' (line 185)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 8), 'fac', result_div_531526)
        
        float_531527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 15), 'float')
        # Getting the type of 'fac' (line 186)
        fac_531528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 21), 'fac')
        # Applying the binary operator '+' (line 186)
        result_add_531529 = python_operator(stypy.reporting.localization.Localization(__file__, 186, 15), '+', float_531527, fac_531528)
        
        # Assigning a type to the variable 'stypy_return_type' (line 186)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 8), 'stypy_return_type', result_add_531529)
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 181)
        exit___531530 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 9), workdps_call_result_531463, '__exit__')
        with_exit_531531 = invoke(stypy.reporting.localization.Localization(__file__, 181, 9), exit___531530, None, None, None)

    
    # ################# End of '_student_t_cdf(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_student_t_cdf' in the type store
    # Getting the type of 'stypy_return_type' (line 178)
    stypy_return_type_531532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_531532)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_student_t_cdf'
    return stypy_return_type_531532

# Assigning a type to the variable '_student_t_cdf' (line 178)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 0), '_student_t_cdf', _student_t_cdf)

@norecursion
def _noncentral_chi_pdf(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_noncentral_chi_pdf'
    module_type_store = module_type_store.open_function_context('_noncentral_chi_pdf', 189, 0, False)
    
    # Passed parameters checking function
    _noncentral_chi_pdf.stypy_localization = localization
    _noncentral_chi_pdf.stypy_type_of_self = None
    _noncentral_chi_pdf.stypy_type_store = module_type_store
    _noncentral_chi_pdf.stypy_function_name = '_noncentral_chi_pdf'
    _noncentral_chi_pdf.stypy_param_names_list = ['t', 'df', 'nc']
    _noncentral_chi_pdf.stypy_varargs_param_name = None
    _noncentral_chi_pdf.stypy_kwargs_param_name = None
    _noncentral_chi_pdf.stypy_call_defaults = defaults
    _noncentral_chi_pdf.stypy_call_varargs = varargs
    _noncentral_chi_pdf.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_noncentral_chi_pdf', ['t', 'df', 'nc'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_noncentral_chi_pdf', localization, ['t', 'df', 'nc'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_noncentral_chi_pdf(...)' code ##################

    
    # Assigning a Call to a Name (line 190):
    
    # Assigning a Call to a Name (line 190):
    
    # Call to besseli(...): (line 190)
    # Processing the call arguments (line 190)
    # Getting the type of 'df' (line 190)
    df_531535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 25), 'df', False)
    int_531536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 28), 'int')
    # Applying the binary operator 'div' (line 190)
    result_div_531537 = python_operator(stypy.reporting.localization.Localization(__file__, 190, 25), 'div', df_531535, int_531536)
    
    int_531538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 32), 'int')
    # Applying the binary operator '-' (line 190)
    result_sub_531539 = python_operator(stypy.reporting.localization.Localization(__file__, 190, 25), '-', result_div_531537, int_531538)
    
    
    # Call to sqrt(...): (line 190)
    # Processing the call arguments (line 190)
    # Getting the type of 'nc' (line 190)
    nc_531542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 47), 'nc', False)
    # Getting the type of 't' (line 190)
    t_531543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 50), 't', False)
    # Applying the binary operator '*' (line 190)
    result_mul_531544 = python_operator(stypy.reporting.localization.Localization(__file__, 190, 47), '*', nc_531542, t_531543)
    
    # Processing the call keyword arguments (line 190)
    kwargs_531545 = {}
    # Getting the type of 'mpmath' (line 190)
    mpmath_531540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 35), 'mpmath', False)
    # Obtaining the member 'sqrt' of a type (line 190)
    sqrt_531541 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 35), mpmath_531540, 'sqrt')
    # Calling sqrt(args, kwargs) (line 190)
    sqrt_call_result_531546 = invoke(stypy.reporting.localization.Localization(__file__, 190, 35), sqrt_531541, *[result_mul_531544], **kwargs_531545)
    
    # Processing the call keyword arguments (line 190)
    kwargs_531547 = {}
    # Getting the type of 'mpmath' (line 190)
    mpmath_531533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 10), 'mpmath', False)
    # Obtaining the member 'besseli' of a type (line 190)
    besseli_531534 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 10), mpmath_531533, 'besseli')
    # Calling besseli(args, kwargs) (line 190)
    besseli_call_result_531548 = invoke(stypy.reporting.localization.Localization(__file__, 190, 10), besseli_531534, *[result_sub_531539, sqrt_call_result_531546], **kwargs_531547)
    
    # Assigning a type to the variable 'res' (line 190)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 4), 'res', besseli_call_result_531548)
    
    # Getting the type of 'res' (line 191)
    res_531549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 4), 'res')
    
    # Call to exp(...): (line 191)
    # Processing the call arguments (line 191)
    
    # Getting the type of 't' (line 191)
    t_531552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 24), 't', False)
    # Getting the type of 'nc' (line 191)
    nc_531553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 28), 'nc', False)
    # Applying the binary operator '+' (line 191)
    result_add_531554 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 24), '+', t_531552, nc_531553)
    
    # Applying the 'usub' unary operator (line 191)
    result___neg___531555 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 22), 'usub', result_add_531554)
    
    int_531556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 32), 'int')
    # Applying the binary operator 'div' (line 191)
    result_div_531557 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 22), 'div', result___neg___531555, int_531556)
    
    # Processing the call keyword arguments (line 191)
    kwargs_531558 = {}
    # Getting the type of 'mpmath' (line 191)
    mpmath_531550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 11), 'mpmath', False)
    # Obtaining the member 'exp' of a type (line 191)
    exp_531551 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 11), mpmath_531550, 'exp')
    # Calling exp(args, kwargs) (line 191)
    exp_call_result_531559 = invoke(stypy.reporting.localization.Localization(__file__, 191, 11), exp_531551, *[result_div_531557], **kwargs_531558)
    
    # Getting the type of 't' (line 191)
    t_531560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 36), 't')
    # Getting the type of 'nc' (line 191)
    nc_531561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 38), 'nc')
    # Applying the binary operator 'div' (line 191)
    result_div_531562 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 36), 'div', t_531560, nc_531561)
    
    # Getting the type of 'df' (line 191)
    df_531563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 44), 'df')
    int_531564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 47), 'int')
    # Applying the binary operator 'div' (line 191)
    result_div_531565 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 44), 'div', df_531563, int_531564)
    
    int_531566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 51), 'int')
    int_531567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 53), 'int')
    # Applying the binary operator 'div' (line 191)
    result_div_531568 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 51), 'div', int_531566, int_531567)
    
    # Applying the binary operator '-' (line 191)
    result_sub_531569 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 44), '-', result_div_531565, result_div_531568)
    
    # Applying the binary operator '**' (line 191)
    result_pow_531570 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 35), '**', result_div_531562, result_sub_531569)
    
    # Applying the binary operator '*' (line 191)
    result_mul_531571 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 11), '*', exp_call_result_531559, result_pow_531570)
    
    int_531572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 56), 'int')
    # Applying the binary operator 'div' (line 191)
    result_div_531573 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 55), 'div', result_mul_531571, int_531572)
    
    # Applying the binary operator '*=' (line 191)
    result_imul_531574 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 4), '*=', res_531549, result_div_531573)
    # Assigning a type to the variable 'res' (line 191)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 4), 'res', result_imul_531574)
    
    # Getting the type of 'res' (line 192)
    res_531575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 11), 'res')
    # Assigning a type to the variable 'stypy_return_type' (line 192)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 4), 'stypy_return_type', res_531575)
    
    # ################# End of '_noncentral_chi_pdf(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_noncentral_chi_pdf' in the type store
    # Getting the type of 'stypy_return_type' (line 189)
    stypy_return_type_531576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_531576)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_noncentral_chi_pdf'
    return stypy_return_type_531576

# Assigning a type to the variable '_noncentral_chi_pdf' (line 189)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 0), '_noncentral_chi_pdf', _noncentral_chi_pdf)

@norecursion
def _noncentral_chi_cdf(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 195)
    None_531577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 39), 'None')
    defaults = [None_531577]
    # Create a new context for function '_noncentral_chi_cdf'
    module_type_store = module_type_store.open_function_context('_noncentral_chi_cdf', 195, 0, False)
    
    # Passed parameters checking function
    _noncentral_chi_cdf.stypy_localization = localization
    _noncentral_chi_cdf.stypy_type_of_self = None
    _noncentral_chi_cdf.stypy_type_store = module_type_store
    _noncentral_chi_cdf.stypy_function_name = '_noncentral_chi_cdf'
    _noncentral_chi_cdf.stypy_param_names_list = ['x', 'df', 'nc', 'dps']
    _noncentral_chi_cdf.stypy_varargs_param_name = None
    _noncentral_chi_cdf.stypy_kwargs_param_name = None
    _noncentral_chi_cdf.stypy_call_defaults = defaults
    _noncentral_chi_cdf.stypy_call_varargs = varargs
    _noncentral_chi_cdf.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_noncentral_chi_cdf', ['x', 'df', 'nc', 'dps'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_noncentral_chi_cdf', localization, ['x', 'df', 'nc', 'dps'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_noncentral_chi_cdf(...)' code ##################

    
    # Type idiom detected: calculating its left and rigth part (line 196)
    # Getting the type of 'dps' (line 196)
    dps_531578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 7), 'dps')
    # Getting the type of 'None' (line 196)
    None_531579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 14), 'None')
    
    (may_be_531580, more_types_in_union_531581) = may_be_none(dps_531578, None_531579)

    if may_be_531580:

        if more_types_in_union_531581:
            # Runtime conditional SSA (line 196)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Attribute to a Name (line 197):
        
        # Assigning a Attribute to a Name (line 197):
        # Getting the type of 'mpmath' (line 197)
        mpmath_531582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 14), 'mpmath')
        # Obtaining the member 'mp' of a type (line 197)
        mp_531583 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 14), mpmath_531582, 'mp')
        # Obtaining the member 'dps' of a type (line 197)
        dps_531584 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 14), mp_531583, 'dps')
        # Assigning a type to the variable 'dps' (line 197)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 8), 'dps', dps_531584)

        if more_types_in_union_531581:
            # SSA join for if statement (line 196)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Tuple to a Tuple (line 198):
    
    # Assigning a Call to a Name (line 198):
    
    # Call to mpf(...): (line 198)
    # Processing the call arguments (line 198)
    # Getting the type of 'x' (line 198)
    x_531587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 27), 'x', False)
    # Processing the call keyword arguments (line 198)
    kwargs_531588 = {}
    # Getting the type of 'mpmath' (line 198)
    mpmath_531585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 16), 'mpmath', False)
    # Obtaining the member 'mpf' of a type (line 198)
    mpf_531586 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 16), mpmath_531585, 'mpf')
    # Calling mpf(args, kwargs) (line 198)
    mpf_call_result_531589 = invoke(stypy.reporting.localization.Localization(__file__, 198, 16), mpf_531586, *[x_531587], **kwargs_531588)
    
    # Assigning a type to the variable 'tuple_assignment_530886' (line 198)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 4), 'tuple_assignment_530886', mpf_call_result_531589)
    
    # Assigning a Call to a Name (line 198):
    
    # Call to mpf(...): (line 198)
    # Processing the call arguments (line 198)
    # Getting the type of 'df' (line 198)
    df_531592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 42), 'df', False)
    # Processing the call keyword arguments (line 198)
    kwargs_531593 = {}
    # Getting the type of 'mpmath' (line 198)
    mpmath_531590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 31), 'mpmath', False)
    # Obtaining the member 'mpf' of a type (line 198)
    mpf_531591 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 31), mpmath_531590, 'mpf')
    # Calling mpf(args, kwargs) (line 198)
    mpf_call_result_531594 = invoke(stypy.reporting.localization.Localization(__file__, 198, 31), mpf_531591, *[df_531592], **kwargs_531593)
    
    # Assigning a type to the variable 'tuple_assignment_530887' (line 198)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 4), 'tuple_assignment_530887', mpf_call_result_531594)
    
    # Assigning a Call to a Name (line 198):
    
    # Call to mpf(...): (line 198)
    # Processing the call arguments (line 198)
    # Getting the type of 'nc' (line 198)
    nc_531597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 58), 'nc', False)
    # Processing the call keyword arguments (line 198)
    kwargs_531598 = {}
    # Getting the type of 'mpmath' (line 198)
    mpmath_531595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 47), 'mpmath', False)
    # Obtaining the member 'mpf' of a type (line 198)
    mpf_531596 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 47), mpmath_531595, 'mpf')
    # Calling mpf(args, kwargs) (line 198)
    mpf_call_result_531599 = invoke(stypy.reporting.localization.Localization(__file__, 198, 47), mpf_531596, *[nc_531597], **kwargs_531598)
    
    # Assigning a type to the variable 'tuple_assignment_530888' (line 198)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 4), 'tuple_assignment_530888', mpf_call_result_531599)
    
    # Assigning a Name to a Name (line 198):
    # Getting the type of 'tuple_assignment_530886' (line 198)
    tuple_assignment_530886_531600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 4), 'tuple_assignment_530886')
    # Assigning a type to the variable 'x' (line 198)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 4), 'x', tuple_assignment_530886_531600)
    
    # Assigning a Name to a Name (line 198):
    # Getting the type of 'tuple_assignment_530887' (line 198)
    tuple_assignment_530887_531601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 4), 'tuple_assignment_530887')
    # Assigning a type to the variable 'df' (line 198)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 7), 'df', tuple_assignment_530887_531601)
    
    # Assigning a Name to a Name (line 198):
    # Getting the type of 'tuple_assignment_530888' (line 198)
    tuple_assignment_530888_531602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 4), 'tuple_assignment_530888')
    # Assigning a type to the variable 'nc' (line 198)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 11), 'nc', tuple_assignment_530888_531602)
    
    # Call to workdps(...): (line 199)
    # Processing the call arguments (line 199)
    # Getting the type of 'dps' (line 199)
    dps_531605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 24), 'dps', False)
    # Processing the call keyword arguments (line 199)
    kwargs_531606 = {}
    # Getting the type of 'mpmath' (line 199)
    mpmath_531603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 9), 'mpmath', False)
    # Obtaining the member 'workdps' of a type (line 199)
    workdps_531604 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 9), mpmath_531603, 'workdps')
    # Calling workdps(args, kwargs) (line 199)
    workdps_call_result_531607 = invoke(stypy.reporting.localization.Localization(__file__, 199, 9), workdps_531604, *[dps_531605], **kwargs_531606)
    
    with_531608 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 199, 9), workdps_call_result_531607, 'with parameter', '__enter__', '__exit__')

    if with_531608:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 199)
        enter___531609 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 9), workdps_call_result_531607, '__enter__')
        with_enter_531610 = invoke(stypy.reporting.localization.Localization(__file__, 199, 9), enter___531609)
        
        # Assigning a Call to a Name (line 200):
        
        # Assigning a Call to a Name (line 200):
        
        # Call to quad(...): (line 200)
        # Processing the call arguments (line 200)

        @norecursion
        def _stypy_temp_lambda_316(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_316'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_316', 200, 26, True)
            # Passed parameters checking function
            _stypy_temp_lambda_316.stypy_localization = localization
            _stypy_temp_lambda_316.stypy_type_of_self = None
            _stypy_temp_lambda_316.stypy_type_store = module_type_store
            _stypy_temp_lambda_316.stypy_function_name = '_stypy_temp_lambda_316'
            _stypy_temp_lambda_316.stypy_param_names_list = ['t']
            _stypy_temp_lambda_316.stypy_varargs_param_name = None
            _stypy_temp_lambda_316.stypy_kwargs_param_name = None
            _stypy_temp_lambda_316.stypy_call_defaults = defaults
            _stypy_temp_lambda_316.stypy_call_varargs = varargs
            _stypy_temp_lambda_316.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_316', ['t'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_316', ['t'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Call to _noncentral_chi_pdf(...): (line 200)
            # Processing the call arguments (line 200)
            # Getting the type of 't' (line 200)
            t_531614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 56), 't', False)
            # Getting the type of 'df' (line 200)
            df_531615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 59), 'df', False)
            # Getting the type of 'nc' (line 200)
            nc_531616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 63), 'nc', False)
            # Processing the call keyword arguments (line 200)
            kwargs_531617 = {}
            # Getting the type of '_noncentral_chi_pdf' (line 200)
            _noncentral_chi_pdf_531613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 36), '_noncentral_chi_pdf', False)
            # Calling _noncentral_chi_pdf(args, kwargs) (line 200)
            _noncentral_chi_pdf_call_result_531618 = invoke(stypy.reporting.localization.Localization(__file__, 200, 36), _noncentral_chi_pdf_531613, *[t_531614, df_531615, nc_531616], **kwargs_531617)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 200)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 26), 'stypy_return_type', _noncentral_chi_pdf_call_result_531618)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_316' in the type store
            # Getting the type of 'stypy_return_type' (line 200)
            stypy_return_type_531619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 26), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_531619)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_316'
            return stypy_return_type_531619

        # Assigning a type to the variable '_stypy_temp_lambda_316' (line 200)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 26), '_stypy_temp_lambda_316', _stypy_temp_lambda_316)
        # Getting the type of '_stypy_temp_lambda_316' (line 200)
        _stypy_temp_lambda_316_531620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 26), '_stypy_temp_lambda_316')
        
        # Obtaining an instance of the builtin type 'list' (line 200)
        list_531621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 68), 'list')
        # Adding type elements to the builtin type 'list' instance (line 200)
        # Adding element type (line 200)
        int_531622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 69), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 68), list_531621, int_531622)
        # Adding element type (line 200)
        # Getting the type of 'x' (line 200)
        x_531623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 72), 'x', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 68), list_531621, x_531623)
        
        # Processing the call keyword arguments (line 200)
        kwargs_531624 = {}
        # Getting the type of 'mpmath' (line 200)
        mpmath_531611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 14), 'mpmath', False)
        # Obtaining the member 'quad' of a type (line 200)
        quad_531612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 14), mpmath_531611, 'quad')
        # Calling quad(args, kwargs) (line 200)
        quad_call_result_531625 = invoke(stypy.reporting.localization.Localization(__file__, 200, 14), quad_531612, *[_stypy_temp_lambda_316_531620, list_531621], **kwargs_531624)
        
        # Assigning a type to the variable 'res' (line 200)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'res', quad_call_result_531625)
        # Getting the type of 'res' (line 201)
        res_531626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 15), 'res')
        # Assigning a type to the variable 'stypy_return_type' (line 201)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 8), 'stypy_return_type', res_531626)
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 199)
        exit___531627 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 9), workdps_call_result_531607, '__exit__')
        with_exit_531628 = invoke(stypy.reporting.localization.Localization(__file__, 199, 9), exit___531627, None, None, None)

    
    # ################# End of '_noncentral_chi_cdf(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_noncentral_chi_cdf' in the type store
    # Getting the type of 'stypy_return_type' (line 195)
    stypy_return_type_531629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_531629)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_noncentral_chi_cdf'
    return stypy_return_type_531629

# Assigning a type to the variable '_noncentral_chi_cdf' (line 195)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 0), '_noncentral_chi_cdf', _noncentral_chi_cdf)

@norecursion
def _tukey_lmbda_quantile(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_tukey_lmbda_quantile'
    module_type_store = module_type_store.open_function_context('_tukey_lmbda_quantile', 204, 0, False)
    
    # Passed parameters checking function
    _tukey_lmbda_quantile.stypy_localization = localization
    _tukey_lmbda_quantile.stypy_type_of_self = None
    _tukey_lmbda_quantile.stypy_type_store = module_type_store
    _tukey_lmbda_quantile.stypy_function_name = '_tukey_lmbda_quantile'
    _tukey_lmbda_quantile.stypy_param_names_list = ['p', 'lmbda']
    _tukey_lmbda_quantile.stypy_varargs_param_name = None
    _tukey_lmbda_quantile.stypy_kwargs_param_name = None
    _tukey_lmbda_quantile.stypy_call_defaults = defaults
    _tukey_lmbda_quantile.stypy_call_varargs = varargs
    _tukey_lmbda_quantile.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_tukey_lmbda_quantile', ['p', 'lmbda'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_tukey_lmbda_quantile', localization, ['p', 'lmbda'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_tukey_lmbda_quantile(...)' code ##################

    # Getting the type of 'p' (line 206)
    p_531630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 12), 'p')
    # Getting the type of 'lmbda' (line 206)
    lmbda_531631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 15), 'lmbda')
    # Applying the binary operator '**' (line 206)
    result_pow_531632 = python_operator(stypy.reporting.localization.Localization(__file__, 206, 12), '**', p_531630, lmbda_531631)
    
    int_531633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 24), 'int')
    # Getting the type of 'p' (line 206)
    p_531634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 28), 'p')
    # Applying the binary operator '-' (line 206)
    result_sub_531635 = python_operator(stypy.reporting.localization.Localization(__file__, 206, 24), '-', int_531633, p_531634)
    
    # Getting the type of 'lmbda' (line 206)
    lmbda_531636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 32), 'lmbda')
    # Applying the binary operator '**' (line 206)
    result_pow_531637 = python_operator(stypy.reporting.localization.Localization(__file__, 206, 23), '**', result_sub_531635, lmbda_531636)
    
    # Applying the binary operator '-' (line 206)
    result_sub_531638 = python_operator(stypy.reporting.localization.Localization(__file__, 206, 12), '-', result_pow_531632, result_pow_531637)
    
    # Getting the type of 'lmbda' (line 206)
    lmbda_531639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 39), 'lmbda')
    # Applying the binary operator 'div' (line 206)
    result_div_531640 = python_operator(stypy.reporting.localization.Localization(__file__, 206, 11), 'div', result_sub_531638, lmbda_531639)
    
    # Assigning a type to the variable 'stypy_return_type' (line 206)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 4), 'stypy_return_type', result_div_531640)
    
    # ################# End of '_tukey_lmbda_quantile(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_tukey_lmbda_quantile' in the type store
    # Getting the type of 'stypy_return_type' (line 204)
    stypy_return_type_531641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_531641)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_tukey_lmbda_quantile'
    return stypy_return_type_531641

# Assigning a type to the variable '_tukey_lmbda_quantile' (line 204)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 0), '_tukey_lmbda_quantile', _tukey_lmbda_quantile)
# Declaration of the 'TestCDFlib' class

class TestCDFlib(object, ):

    @norecursion
    def test_bdtrik(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_bdtrik'
        module_type_store = module_type_store.open_function_context('test_bdtrik', 213, 4, False)
        # Assigning a type to the variable 'self' (line 214)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCDFlib.test_bdtrik.__dict__.__setitem__('stypy_localization', localization)
        TestCDFlib.test_bdtrik.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCDFlib.test_bdtrik.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCDFlib.test_bdtrik.__dict__.__setitem__('stypy_function_name', 'TestCDFlib.test_bdtrik')
        TestCDFlib.test_bdtrik.__dict__.__setitem__('stypy_param_names_list', [])
        TestCDFlib.test_bdtrik.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCDFlib.test_bdtrik.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCDFlib.test_bdtrik.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCDFlib.test_bdtrik.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCDFlib.test_bdtrik.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCDFlib.test_bdtrik.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCDFlib.test_bdtrik', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_bdtrik', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_bdtrik(...)' code ##################

        
        # Call to _assert_inverts(...): (line 215)
        # Processing the call arguments (line 215)
        # Getting the type of 'sp' (line 216)
        sp_531643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 12), 'sp', False)
        # Obtaining the member 'bdtrik' of a type (line 216)
        bdtrik_531644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 12), sp_531643, 'bdtrik')
        # Getting the type of '_binomial_cdf' (line 217)
        _binomial_cdf_531645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 12), '_binomial_cdf', False)
        int_531646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 12), 'int')
        
        # Obtaining an instance of the builtin type 'list' (line 218)
        list_531647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 218)
        # Adding element type (line 218)
        
        # Call to ProbArg(...): (line 218)
        # Processing the call keyword arguments (line 218)
        kwargs_531649 = {}
        # Getting the type of 'ProbArg' (line 218)
        ProbArg_531648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 16), 'ProbArg', False)
        # Calling ProbArg(args, kwargs) (line 218)
        ProbArg_call_result_531650 = invoke(stypy.reporting.localization.Localization(__file__, 218, 16), ProbArg_531648, *[], **kwargs_531649)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 15), list_531647, ProbArg_call_result_531650)
        # Adding element type (line 218)
        
        # Call to IntArg(...): (line 218)
        # Processing the call arguments (line 218)
        int_531652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 34), 'int')
        int_531653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 37), 'int')
        # Processing the call keyword arguments (line 218)
        kwargs_531654 = {}
        # Getting the type of 'IntArg' (line 218)
        IntArg_531651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 27), 'IntArg', False)
        # Calling IntArg(args, kwargs) (line 218)
        IntArg_call_result_531655 = invoke(stypy.reporting.localization.Localization(__file__, 218, 27), IntArg_531651, *[int_531652, int_531653], **kwargs_531654)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 15), list_531647, IntArg_call_result_531655)
        # Adding element type (line 218)
        
        # Call to ProbArg(...): (line 218)
        # Processing the call keyword arguments (line 218)
        kwargs_531657 = {}
        # Getting the type of 'ProbArg' (line 218)
        ProbArg_531656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 44), 'ProbArg', False)
        # Calling ProbArg(args, kwargs) (line 218)
        ProbArg_call_result_531658 = invoke(stypy.reporting.localization.Localization(__file__, 218, 44), ProbArg_531656, *[], **kwargs_531657)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 15), list_531647, ProbArg_call_result_531658)
        
        # Processing the call keyword arguments (line 215)
        float_531659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 17), 'float')
        keyword_531660 = float_531659
        kwargs_531661 = {'rtol': keyword_531660}
        # Getting the type of '_assert_inverts' (line 215)
        _assert_inverts_531642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 8), '_assert_inverts', False)
        # Calling _assert_inverts(args, kwargs) (line 215)
        _assert_inverts_call_result_531662 = invoke(stypy.reporting.localization.Localization(__file__, 215, 8), _assert_inverts_531642, *[bdtrik_531644, _binomial_cdf_531645, int_531646, list_531647], **kwargs_531661)
        
        
        # ################# End of 'test_bdtrik(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_bdtrik' in the type store
        # Getting the type of 'stypy_return_type' (line 213)
        stypy_return_type_531663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_531663)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_bdtrik'
        return stypy_return_type_531663


    @norecursion
    def test_bdtrin(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_bdtrin'
        module_type_store = module_type_store.open_function_context('test_bdtrin', 221, 4, False)
        # Assigning a type to the variable 'self' (line 222)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCDFlib.test_bdtrin.__dict__.__setitem__('stypy_localization', localization)
        TestCDFlib.test_bdtrin.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCDFlib.test_bdtrin.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCDFlib.test_bdtrin.__dict__.__setitem__('stypy_function_name', 'TestCDFlib.test_bdtrin')
        TestCDFlib.test_bdtrin.__dict__.__setitem__('stypy_param_names_list', [])
        TestCDFlib.test_bdtrin.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCDFlib.test_bdtrin.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCDFlib.test_bdtrin.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCDFlib.test_bdtrin.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCDFlib.test_bdtrin.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCDFlib.test_bdtrin.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCDFlib.test_bdtrin', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_bdtrin', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_bdtrin(...)' code ##################

        
        # Call to _assert_inverts(...): (line 222)
        # Processing the call arguments (line 222)
        # Getting the type of 'sp' (line 223)
        sp_531665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 12), 'sp', False)
        # Obtaining the member 'bdtrin' of a type (line 223)
        bdtrin_531666 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 12), sp_531665, 'bdtrin')
        # Getting the type of '_binomial_cdf' (line 224)
        _binomial_cdf_531667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 12), '_binomial_cdf', False)
        int_531668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 12), 'int')
        
        # Obtaining an instance of the builtin type 'list' (line 225)
        list_531669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 225)
        # Adding element type (line 225)
        
        # Call to IntArg(...): (line 225)
        # Processing the call arguments (line 225)
        int_531671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 23), 'int')
        int_531672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 26), 'int')
        # Processing the call keyword arguments (line 225)
        kwargs_531673 = {}
        # Getting the type of 'IntArg' (line 225)
        IntArg_531670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 16), 'IntArg', False)
        # Calling IntArg(args, kwargs) (line 225)
        IntArg_call_result_531674 = invoke(stypy.reporting.localization.Localization(__file__, 225, 16), IntArg_531670, *[int_531671, int_531672], **kwargs_531673)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 225, 15), list_531669, IntArg_call_result_531674)
        # Adding element type (line 225)
        
        # Call to ProbArg(...): (line 225)
        # Processing the call keyword arguments (line 225)
        kwargs_531676 = {}
        # Getting the type of 'ProbArg' (line 225)
        ProbArg_531675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 33), 'ProbArg', False)
        # Calling ProbArg(args, kwargs) (line 225)
        ProbArg_call_result_531677 = invoke(stypy.reporting.localization.Localization(__file__, 225, 33), ProbArg_531675, *[], **kwargs_531676)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 225, 15), list_531669, ProbArg_call_result_531677)
        # Adding element type (line 225)
        
        # Call to ProbArg(...): (line 225)
        # Processing the call keyword arguments (line 225)
        kwargs_531679 = {}
        # Getting the type of 'ProbArg' (line 225)
        ProbArg_531678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 44), 'ProbArg', False)
        # Calling ProbArg(args, kwargs) (line 225)
        ProbArg_call_result_531680 = invoke(stypy.reporting.localization.Localization(__file__, 225, 44), ProbArg_531678, *[], **kwargs_531679)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 225, 15), list_531669, ProbArg_call_result_531680)
        
        # Processing the call keyword arguments (line 222)
        float_531681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 17), 'float')
        keyword_531682 = float_531681
        
        # Obtaining an instance of the builtin type 'list' (line 226)
        list_531683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 34), 'list')
        # Adding type elements to the builtin type 'list' instance (line 226)
        # Adding element type (line 226)
        # Getting the type of 'None' (line 226)
        None_531684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 35), 'None', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 34), list_531683, None_531684)
        # Adding element type (line 226)
        # Getting the type of 'None' (line 226)
        None_531685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 41), 'None', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 34), list_531683, None_531685)
        # Adding element type (line 226)
        float_531686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 47), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 34), list_531683, float_531686)
        
        keyword_531687 = list_531683
        kwargs_531688 = {'rtol': keyword_531682, 'endpt_atol': keyword_531687}
        # Getting the type of '_assert_inverts' (line 222)
        _assert_inverts_531664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 8), '_assert_inverts', False)
        # Calling _assert_inverts(args, kwargs) (line 222)
        _assert_inverts_call_result_531689 = invoke(stypy.reporting.localization.Localization(__file__, 222, 8), _assert_inverts_531664, *[bdtrin_531666, _binomial_cdf_531667, int_531668, list_531669], **kwargs_531688)
        
        
        # ################# End of 'test_bdtrin(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_bdtrin' in the type store
        # Getting the type of 'stypy_return_type' (line 221)
        stypy_return_type_531690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_531690)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_bdtrin'
        return stypy_return_type_531690


    @norecursion
    def test_btdtria(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_btdtria'
        module_type_store = module_type_store.open_function_context('test_btdtria', 228, 4, False)
        # Assigning a type to the variable 'self' (line 229)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCDFlib.test_btdtria.__dict__.__setitem__('stypy_localization', localization)
        TestCDFlib.test_btdtria.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCDFlib.test_btdtria.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCDFlib.test_btdtria.__dict__.__setitem__('stypy_function_name', 'TestCDFlib.test_btdtria')
        TestCDFlib.test_btdtria.__dict__.__setitem__('stypy_param_names_list', [])
        TestCDFlib.test_btdtria.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCDFlib.test_btdtria.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCDFlib.test_btdtria.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCDFlib.test_btdtria.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCDFlib.test_btdtria.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCDFlib.test_btdtria.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCDFlib.test_btdtria', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_btdtria', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_btdtria(...)' code ##################

        
        # Call to _assert_inverts(...): (line 229)
        # Processing the call arguments (line 229)
        # Getting the type of 'sp' (line 230)
        sp_531692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 12), 'sp', False)
        # Obtaining the member 'btdtria' of a type (line 230)
        btdtria_531693 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 12), sp_531692, 'btdtria')

        @norecursion
        def _stypy_temp_lambda_317(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_317'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_317', 231, 12, True)
            # Passed parameters checking function
            _stypy_temp_lambda_317.stypy_localization = localization
            _stypy_temp_lambda_317.stypy_type_of_self = None
            _stypy_temp_lambda_317.stypy_type_store = module_type_store
            _stypy_temp_lambda_317.stypy_function_name = '_stypy_temp_lambda_317'
            _stypy_temp_lambda_317.stypy_param_names_list = ['a', 'b', 'x']
            _stypy_temp_lambda_317.stypy_varargs_param_name = None
            _stypy_temp_lambda_317.stypy_kwargs_param_name = None
            _stypy_temp_lambda_317.stypy_call_defaults = defaults
            _stypy_temp_lambda_317.stypy_call_varargs = varargs
            _stypy_temp_lambda_317.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_317', ['a', 'b', 'x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_317', ['a', 'b', 'x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Call to betainc(...): (line 231)
            # Processing the call arguments (line 231)
            # Getting the type of 'a' (line 231)
            a_531696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 43), 'a', False)
            # Getting the type of 'b' (line 231)
            b_531697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 46), 'b', False)
            # Processing the call keyword arguments (line 231)
            # Getting the type of 'x' (line 231)
            x_531698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 52), 'x', False)
            keyword_531699 = x_531698
            # Getting the type of 'True' (line 231)
            True_531700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 67), 'True', False)
            keyword_531701 = True_531700
            kwargs_531702 = {'x2': keyword_531699, 'regularized': keyword_531701}
            # Getting the type of 'mpmath' (line 231)
            mpmath_531694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 28), 'mpmath', False)
            # Obtaining the member 'betainc' of a type (line 231)
            betainc_531695 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 28), mpmath_531694, 'betainc')
            # Calling betainc(args, kwargs) (line 231)
            betainc_call_result_531703 = invoke(stypy.reporting.localization.Localization(__file__, 231, 28), betainc_531695, *[a_531696, b_531697], **kwargs_531702)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 231)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 12), 'stypy_return_type', betainc_call_result_531703)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_317' in the type store
            # Getting the type of 'stypy_return_type' (line 231)
            stypy_return_type_531704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 12), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_531704)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_317'
            return stypy_return_type_531704

        # Assigning a type to the variable '_stypy_temp_lambda_317' (line 231)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 12), '_stypy_temp_lambda_317', _stypy_temp_lambda_317)
        # Getting the type of '_stypy_temp_lambda_317' (line 231)
        _stypy_temp_lambda_317_531705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 12), '_stypy_temp_lambda_317')
        int_531706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 12), 'int')
        
        # Obtaining an instance of the builtin type 'list' (line 232)
        list_531707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 232)
        # Adding element type (line 232)
        
        # Call to ProbArg(...): (line 232)
        # Processing the call keyword arguments (line 232)
        kwargs_531709 = {}
        # Getting the type of 'ProbArg' (line 232)
        ProbArg_531708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 16), 'ProbArg', False)
        # Calling ProbArg(args, kwargs) (line 232)
        ProbArg_call_result_531710 = invoke(stypy.reporting.localization.Localization(__file__, 232, 16), ProbArg_531708, *[], **kwargs_531709)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 232, 15), list_531707, ProbArg_call_result_531710)
        # Adding element type (line 232)
        
        # Call to Arg(...): (line 232)
        # Processing the call arguments (line 232)
        int_531712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 31), 'int')
        float_531713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 34), 'float')
        # Processing the call keyword arguments (line 232)
        # Getting the type of 'False' (line 232)
        False_531714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 51), 'False', False)
        keyword_531715 = False_531714
        kwargs_531716 = {'inclusive_a': keyword_531715}
        # Getting the type of 'Arg' (line 232)
        Arg_531711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 27), 'Arg', False)
        # Calling Arg(args, kwargs) (line 232)
        Arg_call_result_531717 = invoke(stypy.reporting.localization.Localization(__file__, 232, 27), Arg_531711, *[int_531712, float_531713], **kwargs_531716)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 232, 15), list_531707, Arg_call_result_531717)
        # Adding element type (line 232)
        
        # Call to Arg(...): (line 233)
        # Processing the call arguments (line 233)
        int_531719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 20), 'int')
        int_531720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 23), 'int')
        # Processing the call keyword arguments (line 233)
        # Getting the type of 'False' (line 233)
        False_531721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 38), 'False', False)
        keyword_531722 = False_531721
        # Getting the type of 'False' (line 233)
        False_531723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 57), 'False', False)
        keyword_531724 = False_531723
        kwargs_531725 = {'inclusive_a': keyword_531722, 'inclusive_b': keyword_531724}
        # Getting the type of 'Arg' (line 233)
        Arg_531718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 16), 'Arg', False)
        # Calling Arg(args, kwargs) (line 233)
        Arg_call_result_531726 = invoke(stypy.reporting.localization.Localization(__file__, 233, 16), Arg_531718, *[int_531719, int_531720], **kwargs_531725)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 232, 15), list_531707, Arg_call_result_531726)
        
        # Processing the call keyword arguments (line 229)
        float_531727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 17), 'float')
        keyword_531728 = float_531727
        kwargs_531729 = {'rtol': keyword_531728}
        # Getting the type of '_assert_inverts' (line 229)
        _assert_inverts_531691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 8), '_assert_inverts', False)
        # Calling _assert_inverts(args, kwargs) (line 229)
        _assert_inverts_call_result_531730 = invoke(stypy.reporting.localization.Localization(__file__, 229, 8), _assert_inverts_531691, *[btdtria_531693, _stypy_temp_lambda_317_531705, int_531706, list_531707], **kwargs_531729)
        
        
        # ################# End of 'test_btdtria(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_btdtria' in the type store
        # Getting the type of 'stypy_return_type' (line 228)
        stypy_return_type_531731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_531731)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_btdtria'
        return stypy_return_type_531731


    @norecursion
    def test_btdtrib(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_btdtrib'
        module_type_store = module_type_store.open_function_context('test_btdtrib', 236, 4, False)
        # Assigning a type to the variable 'self' (line 237)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCDFlib.test_btdtrib.__dict__.__setitem__('stypy_localization', localization)
        TestCDFlib.test_btdtrib.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCDFlib.test_btdtrib.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCDFlib.test_btdtrib.__dict__.__setitem__('stypy_function_name', 'TestCDFlib.test_btdtrib')
        TestCDFlib.test_btdtrib.__dict__.__setitem__('stypy_param_names_list', [])
        TestCDFlib.test_btdtrib.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCDFlib.test_btdtrib.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCDFlib.test_btdtrib.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCDFlib.test_btdtrib.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCDFlib.test_btdtrib.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCDFlib.test_btdtrib.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCDFlib.test_btdtrib', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_btdtrib', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_btdtrib(...)' code ##################

        
        # Call to _assert_inverts(...): (line 238)
        # Processing the call arguments (line 238)
        # Getting the type of 'sp' (line 239)
        sp_531733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 12), 'sp', False)
        # Obtaining the member 'btdtrib' of a type (line 239)
        btdtrib_531734 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 12), sp_531733, 'btdtrib')

        @norecursion
        def _stypy_temp_lambda_318(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_318'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_318', 240, 12, True)
            # Passed parameters checking function
            _stypy_temp_lambda_318.stypy_localization = localization
            _stypy_temp_lambda_318.stypy_type_of_self = None
            _stypy_temp_lambda_318.stypy_type_store = module_type_store
            _stypy_temp_lambda_318.stypy_function_name = '_stypy_temp_lambda_318'
            _stypy_temp_lambda_318.stypy_param_names_list = ['a', 'b', 'x']
            _stypy_temp_lambda_318.stypy_varargs_param_name = None
            _stypy_temp_lambda_318.stypy_kwargs_param_name = None
            _stypy_temp_lambda_318.stypy_call_defaults = defaults
            _stypy_temp_lambda_318.stypy_call_varargs = varargs
            _stypy_temp_lambda_318.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_318', ['a', 'b', 'x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_318', ['a', 'b', 'x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Call to betainc(...): (line 240)
            # Processing the call arguments (line 240)
            # Getting the type of 'a' (line 240)
            a_531737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 43), 'a', False)
            # Getting the type of 'b' (line 240)
            b_531738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 46), 'b', False)
            # Processing the call keyword arguments (line 240)
            # Getting the type of 'x' (line 240)
            x_531739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 52), 'x', False)
            keyword_531740 = x_531739
            # Getting the type of 'True' (line 240)
            True_531741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 67), 'True', False)
            keyword_531742 = True_531741
            kwargs_531743 = {'x2': keyword_531740, 'regularized': keyword_531742}
            # Getting the type of 'mpmath' (line 240)
            mpmath_531735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 28), 'mpmath', False)
            # Obtaining the member 'betainc' of a type (line 240)
            betainc_531736 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 28), mpmath_531735, 'betainc')
            # Calling betainc(args, kwargs) (line 240)
            betainc_call_result_531744 = invoke(stypy.reporting.localization.Localization(__file__, 240, 28), betainc_531736, *[a_531737, b_531738], **kwargs_531743)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 240)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 12), 'stypy_return_type', betainc_call_result_531744)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_318' in the type store
            # Getting the type of 'stypy_return_type' (line 240)
            stypy_return_type_531745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 12), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_531745)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_318'
            return stypy_return_type_531745

        # Assigning a type to the variable '_stypy_temp_lambda_318' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 12), '_stypy_temp_lambda_318', _stypy_temp_lambda_318)
        # Getting the type of '_stypy_temp_lambda_318' (line 240)
        _stypy_temp_lambda_318_531746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 12), '_stypy_temp_lambda_318')
        int_531747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 12), 'int')
        
        # Obtaining an instance of the builtin type 'list' (line 241)
        list_531748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 241)
        # Adding element type (line 241)
        
        # Call to Arg(...): (line 241)
        # Processing the call arguments (line 241)
        int_531750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 20), 'int')
        float_531751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 23), 'float')
        # Processing the call keyword arguments (line 241)
        # Getting the type of 'False' (line 241)
        False_531752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 40), 'False', False)
        keyword_531753 = False_531752
        kwargs_531754 = {'inclusive_a': keyword_531753}
        # Getting the type of 'Arg' (line 241)
        Arg_531749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 16), 'Arg', False)
        # Calling Arg(args, kwargs) (line 241)
        Arg_call_result_531755 = invoke(stypy.reporting.localization.Localization(__file__, 241, 16), Arg_531749, *[int_531750, float_531751], **kwargs_531754)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 241, 15), list_531748, Arg_call_result_531755)
        # Adding element type (line 241)
        
        # Call to ProbArg(...): (line 241)
        # Processing the call keyword arguments (line 241)
        kwargs_531757 = {}
        # Getting the type of 'ProbArg' (line 241)
        ProbArg_531756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 48), 'ProbArg', False)
        # Calling ProbArg(args, kwargs) (line 241)
        ProbArg_call_result_531758 = invoke(stypy.reporting.localization.Localization(__file__, 241, 48), ProbArg_531756, *[], **kwargs_531757)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 241, 15), list_531748, ProbArg_call_result_531758)
        # Adding element type (line 241)
        
        # Call to Arg(...): (line 242)
        # Processing the call arguments (line 242)
        int_531760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 17), 'int')
        int_531761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 20), 'int')
        # Processing the call keyword arguments (line 242)
        # Getting the type of 'False' (line 242)
        False_531762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 35), 'False', False)
        keyword_531763 = False_531762
        # Getting the type of 'False' (line 242)
        False_531764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 54), 'False', False)
        keyword_531765 = False_531764
        kwargs_531766 = {'inclusive_a': keyword_531763, 'inclusive_b': keyword_531765}
        # Getting the type of 'Arg' (line 242)
        Arg_531759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 13), 'Arg', False)
        # Calling Arg(args, kwargs) (line 242)
        Arg_call_result_531767 = invoke(stypy.reporting.localization.Localization(__file__, 242, 13), Arg_531759, *[int_531760, int_531761], **kwargs_531766)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 241, 15), list_531748, Arg_call_result_531767)
        
        # Processing the call keyword arguments (line 238)
        float_531768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 17), 'float')
        keyword_531769 = float_531768
        
        # Obtaining an instance of the builtin type 'list' (line 243)
        list_531770 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 34), 'list')
        # Adding type elements to the builtin type 'list' instance (line 243)
        # Adding element type (line 243)
        # Getting the type of 'None' (line 243)
        None_531771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 35), 'None', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 243, 34), list_531770, None_531771)
        # Adding element type (line 243)
        float_531772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 41), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 243, 34), list_531770, float_531772)
        # Adding element type (line 243)
        float_531773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 48), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 243, 34), list_531770, float_531773)
        
        keyword_531774 = list_531770
        kwargs_531775 = {'rtol': keyword_531769, 'endpt_atol': keyword_531774}
        # Getting the type of '_assert_inverts' (line 238)
        _assert_inverts_531732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 8), '_assert_inverts', False)
        # Calling _assert_inverts(args, kwargs) (line 238)
        _assert_inverts_call_result_531776 = invoke(stypy.reporting.localization.Localization(__file__, 238, 8), _assert_inverts_531732, *[btdtrib_531734, _stypy_temp_lambda_318_531746, int_531747, list_531748], **kwargs_531775)
        
        
        # ################# End of 'test_btdtrib(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_btdtrib' in the type store
        # Getting the type of 'stypy_return_type' (line 236)
        stypy_return_type_531777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_531777)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_btdtrib'
        return stypy_return_type_531777


    @norecursion
    def test_fdtridfd(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_fdtridfd'
        module_type_store = module_type_store.open_function_context('test_fdtridfd', 245, 4, False)
        # Assigning a type to the variable 'self' (line 246)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCDFlib.test_fdtridfd.__dict__.__setitem__('stypy_localization', localization)
        TestCDFlib.test_fdtridfd.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCDFlib.test_fdtridfd.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCDFlib.test_fdtridfd.__dict__.__setitem__('stypy_function_name', 'TestCDFlib.test_fdtridfd')
        TestCDFlib.test_fdtridfd.__dict__.__setitem__('stypy_param_names_list', [])
        TestCDFlib.test_fdtridfd.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCDFlib.test_fdtridfd.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCDFlib.test_fdtridfd.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCDFlib.test_fdtridfd.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCDFlib.test_fdtridfd.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCDFlib.test_fdtridfd.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCDFlib.test_fdtridfd', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_fdtridfd', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_fdtridfd(...)' code ##################

        
        # Call to _assert_inverts(...): (line 247)
        # Processing the call arguments (line 247)
        # Getting the type of 'sp' (line 248)
        sp_531779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 12), 'sp', False)
        # Obtaining the member 'fdtridfd' of a type (line 248)
        fdtridfd_531780 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 12), sp_531779, 'fdtridfd')
        # Getting the type of '_f_cdf' (line 249)
        _f_cdf_531781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 12), '_f_cdf', False)
        int_531782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 12), 'int')
        
        # Obtaining an instance of the builtin type 'list' (line 250)
        list_531783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 250)
        # Adding element type (line 250)
        
        # Call to IntArg(...): (line 250)
        # Processing the call arguments (line 250)
        int_531785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 23), 'int')
        int_531786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 26), 'int')
        # Processing the call keyword arguments (line 250)
        kwargs_531787 = {}
        # Getting the type of 'IntArg' (line 250)
        IntArg_531784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 16), 'IntArg', False)
        # Calling IntArg(args, kwargs) (line 250)
        IntArg_call_result_531788 = invoke(stypy.reporting.localization.Localization(__file__, 250, 16), IntArg_531784, *[int_531785, int_531786], **kwargs_531787)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 250, 15), list_531783, IntArg_call_result_531788)
        # Adding element type (line 250)
        
        # Call to ProbArg(...): (line 250)
        # Processing the call keyword arguments (line 250)
        kwargs_531790 = {}
        # Getting the type of 'ProbArg' (line 250)
        ProbArg_531789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 32), 'ProbArg', False)
        # Calling ProbArg(args, kwargs) (line 250)
        ProbArg_call_result_531791 = invoke(stypy.reporting.localization.Localization(__file__, 250, 32), ProbArg_531789, *[], **kwargs_531790)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 250, 15), list_531783, ProbArg_call_result_531791)
        # Adding element type (line 250)
        
        # Call to Arg(...): (line 250)
        # Processing the call arguments (line 250)
        int_531793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 47), 'int')
        int_531794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 50), 'int')
        # Processing the call keyword arguments (line 250)
        # Getting the type of 'False' (line 250)
        False_531795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 67), 'False', False)
        keyword_531796 = False_531795
        kwargs_531797 = {'inclusive_a': keyword_531796}
        # Getting the type of 'Arg' (line 250)
        Arg_531792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 43), 'Arg', False)
        # Calling Arg(args, kwargs) (line 250)
        Arg_call_result_531798 = invoke(stypy.reporting.localization.Localization(__file__, 250, 43), Arg_531792, *[int_531793, int_531794], **kwargs_531797)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 250, 15), list_531783, Arg_call_result_531798)
        
        # Processing the call keyword arguments (line 247)
        float_531799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 17), 'float')
        keyword_531800 = float_531799
        kwargs_531801 = {'rtol': keyword_531800}
        # Getting the type of '_assert_inverts' (line 247)
        _assert_inverts_531778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 8), '_assert_inverts', False)
        # Calling _assert_inverts(args, kwargs) (line 247)
        _assert_inverts_call_result_531802 = invoke(stypy.reporting.localization.Localization(__file__, 247, 8), _assert_inverts_531778, *[fdtridfd_531780, _f_cdf_531781, int_531782, list_531783], **kwargs_531801)
        
        
        # ################# End of 'test_fdtridfd(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_fdtridfd' in the type store
        # Getting the type of 'stypy_return_type' (line 245)
        stypy_return_type_531803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_531803)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_fdtridfd'
        return stypy_return_type_531803


    @norecursion
    def test_gdtria(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_gdtria'
        module_type_store = module_type_store.open_function_context('test_gdtria', 253, 4, False)
        # Assigning a type to the variable 'self' (line 254)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCDFlib.test_gdtria.__dict__.__setitem__('stypy_localization', localization)
        TestCDFlib.test_gdtria.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCDFlib.test_gdtria.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCDFlib.test_gdtria.__dict__.__setitem__('stypy_function_name', 'TestCDFlib.test_gdtria')
        TestCDFlib.test_gdtria.__dict__.__setitem__('stypy_param_names_list', [])
        TestCDFlib.test_gdtria.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCDFlib.test_gdtria.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCDFlib.test_gdtria.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCDFlib.test_gdtria.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCDFlib.test_gdtria.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCDFlib.test_gdtria.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCDFlib.test_gdtria', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_gdtria', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_gdtria(...)' code ##################

        
        # Call to _assert_inverts(...): (line 254)
        # Processing the call arguments (line 254)
        # Getting the type of 'sp' (line 255)
        sp_531805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 12), 'sp', False)
        # Obtaining the member 'gdtria' of a type (line 255)
        gdtria_531806 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 12), sp_531805, 'gdtria')

        @norecursion
        def _stypy_temp_lambda_319(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_319'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_319', 256, 12, True)
            # Passed parameters checking function
            _stypy_temp_lambda_319.stypy_localization = localization
            _stypy_temp_lambda_319.stypy_type_of_self = None
            _stypy_temp_lambda_319.stypy_type_store = module_type_store
            _stypy_temp_lambda_319.stypy_function_name = '_stypy_temp_lambda_319'
            _stypy_temp_lambda_319.stypy_param_names_list = ['a', 'b', 'x']
            _stypy_temp_lambda_319.stypy_varargs_param_name = None
            _stypy_temp_lambda_319.stypy_kwargs_param_name = None
            _stypy_temp_lambda_319.stypy_call_defaults = defaults
            _stypy_temp_lambda_319.stypy_call_varargs = varargs
            _stypy_temp_lambda_319.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_319', ['a', 'b', 'x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_319', ['a', 'b', 'x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Call to gammainc(...): (line 256)
            # Processing the call arguments (line 256)
            # Getting the type of 'b' (line 256)
            b_531809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 44), 'b', False)
            # Processing the call keyword arguments (line 256)
            # Getting the type of 'a' (line 256)
            a_531810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 49), 'a', False)
            # Getting the type of 'x' (line 256)
            x_531811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 51), 'x', False)
            # Applying the binary operator '*' (line 256)
            result_mul_531812 = python_operator(stypy.reporting.localization.Localization(__file__, 256, 49), '*', a_531810, x_531811)
            
            keyword_531813 = result_mul_531812
            # Getting the type of 'True' (line 256)
            True_531814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 66), 'True', False)
            keyword_531815 = True_531814
            kwargs_531816 = {'regularized': keyword_531815, 'b': keyword_531813}
            # Getting the type of 'mpmath' (line 256)
            mpmath_531807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 28), 'mpmath', False)
            # Obtaining the member 'gammainc' of a type (line 256)
            gammainc_531808 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 28), mpmath_531807, 'gammainc')
            # Calling gammainc(args, kwargs) (line 256)
            gammainc_call_result_531817 = invoke(stypy.reporting.localization.Localization(__file__, 256, 28), gammainc_531808, *[b_531809], **kwargs_531816)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 256)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 12), 'stypy_return_type', gammainc_call_result_531817)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_319' in the type store
            # Getting the type of 'stypy_return_type' (line 256)
            stypy_return_type_531818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 12), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_531818)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_319'
            return stypy_return_type_531818

        # Assigning a type to the variable '_stypy_temp_lambda_319' (line 256)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 12), '_stypy_temp_lambda_319', _stypy_temp_lambda_319)
        # Getting the type of '_stypy_temp_lambda_319' (line 256)
        _stypy_temp_lambda_319_531819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 12), '_stypy_temp_lambda_319')
        int_531820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 12), 'int')
        
        # Obtaining an instance of the builtin type 'list' (line 257)
        list_531821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 257)
        # Adding element type (line 257)
        
        # Call to ProbArg(...): (line 257)
        # Processing the call keyword arguments (line 257)
        kwargs_531823 = {}
        # Getting the type of 'ProbArg' (line 257)
        ProbArg_531822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 16), 'ProbArg', False)
        # Calling ProbArg(args, kwargs) (line 257)
        ProbArg_call_result_531824 = invoke(stypy.reporting.localization.Localization(__file__, 257, 16), ProbArg_531822, *[], **kwargs_531823)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 15), list_531821, ProbArg_call_result_531824)
        # Adding element type (line 257)
        
        # Call to Arg(...): (line 257)
        # Processing the call arguments (line 257)
        int_531826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 31), 'int')
        float_531827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 34), 'float')
        # Processing the call keyword arguments (line 257)
        # Getting the type of 'False' (line 257)
        False_531828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 51), 'False', False)
        keyword_531829 = False_531828
        kwargs_531830 = {'inclusive_a': keyword_531829}
        # Getting the type of 'Arg' (line 257)
        Arg_531825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 27), 'Arg', False)
        # Calling Arg(args, kwargs) (line 257)
        Arg_call_result_531831 = invoke(stypy.reporting.localization.Localization(__file__, 257, 27), Arg_531825, *[int_531826, float_531827], **kwargs_531830)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 15), list_531821, Arg_call_result_531831)
        # Adding element type (line 257)
        
        # Call to Arg(...): (line 258)
        # Processing the call arguments (line 258)
        int_531833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 20), 'int')
        float_531834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 23), 'float')
        # Processing the call keyword arguments (line 258)
        # Getting the type of 'False' (line 258)
        False_531835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 40), 'False', False)
        keyword_531836 = False_531835
        kwargs_531837 = {'inclusive_a': keyword_531836}
        # Getting the type of 'Arg' (line 258)
        Arg_531832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 16), 'Arg', False)
        # Calling Arg(args, kwargs) (line 258)
        Arg_call_result_531838 = invoke(stypy.reporting.localization.Localization(__file__, 258, 16), Arg_531832, *[int_531833, float_531834], **kwargs_531837)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 15), list_531821, Arg_call_result_531838)
        
        # Processing the call keyword arguments (line 254)
        float_531839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 54), 'float')
        keyword_531840 = float_531839
        
        # Obtaining an instance of the builtin type 'list' (line 259)
        list_531841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 259)
        # Adding element type (line 259)
        # Getting the type of 'None' (line 259)
        None_531842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 24), 'None', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 23), list_531841, None_531842)
        # Adding element type (line 259)
        float_531843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 23), list_531841, float_531843)
        # Adding element type (line 259)
        float_531844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 37), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 23), list_531841, float_531844)
        
        keyword_531845 = list_531841
        kwargs_531846 = {'rtol': keyword_531840, 'endpt_atol': keyword_531845}
        # Getting the type of '_assert_inverts' (line 254)
        _assert_inverts_531804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 8), '_assert_inverts', False)
        # Calling _assert_inverts(args, kwargs) (line 254)
        _assert_inverts_call_result_531847 = invoke(stypy.reporting.localization.Localization(__file__, 254, 8), _assert_inverts_531804, *[gdtria_531806, _stypy_temp_lambda_319_531819, int_531820, list_531821], **kwargs_531846)
        
        
        # ################# End of 'test_gdtria(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_gdtria' in the type store
        # Getting the type of 'stypy_return_type' (line 253)
        stypy_return_type_531848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_531848)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_gdtria'
        return stypy_return_type_531848


    @norecursion
    def test_gdtrib(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_gdtrib'
        module_type_store = module_type_store.open_function_context('test_gdtrib', 261, 4, False)
        # Assigning a type to the variable 'self' (line 262)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCDFlib.test_gdtrib.__dict__.__setitem__('stypy_localization', localization)
        TestCDFlib.test_gdtrib.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCDFlib.test_gdtrib.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCDFlib.test_gdtrib.__dict__.__setitem__('stypy_function_name', 'TestCDFlib.test_gdtrib')
        TestCDFlib.test_gdtrib.__dict__.__setitem__('stypy_param_names_list', [])
        TestCDFlib.test_gdtrib.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCDFlib.test_gdtrib.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCDFlib.test_gdtrib.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCDFlib.test_gdtrib.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCDFlib.test_gdtrib.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCDFlib.test_gdtrib.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCDFlib.test_gdtrib', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_gdtrib', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_gdtrib(...)' code ##################

        
        # Call to _assert_inverts(...): (line 263)
        # Processing the call arguments (line 263)
        # Getting the type of 'sp' (line 264)
        sp_531850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 12), 'sp', False)
        # Obtaining the member 'gdtrib' of a type (line 264)
        gdtrib_531851 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 12), sp_531850, 'gdtrib')

        @norecursion
        def _stypy_temp_lambda_320(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_320'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_320', 265, 12, True)
            # Passed parameters checking function
            _stypy_temp_lambda_320.stypy_localization = localization
            _stypy_temp_lambda_320.stypy_type_of_self = None
            _stypy_temp_lambda_320.stypy_type_store = module_type_store
            _stypy_temp_lambda_320.stypy_function_name = '_stypy_temp_lambda_320'
            _stypy_temp_lambda_320.stypy_param_names_list = ['a', 'b', 'x']
            _stypy_temp_lambda_320.stypy_varargs_param_name = None
            _stypy_temp_lambda_320.stypy_kwargs_param_name = None
            _stypy_temp_lambda_320.stypy_call_defaults = defaults
            _stypy_temp_lambda_320.stypy_call_varargs = varargs
            _stypy_temp_lambda_320.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_320', ['a', 'b', 'x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_320', ['a', 'b', 'x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Call to gammainc(...): (line 265)
            # Processing the call arguments (line 265)
            # Getting the type of 'b' (line 265)
            b_531854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 44), 'b', False)
            # Processing the call keyword arguments (line 265)
            # Getting the type of 'a' (line 265)
            a_531855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 49), 'a', False)
            # Getting the type of 'x' (line 265)
            x_531856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 51), 'x', False)
            # Applying the binary operator '*' (line 265)
            result_mul_531857 = python_operator(stypy.reporting.localization.Localization(__file__, 265, 49), '*', a_531855, x_531856)
            
            keyword_531858 = result_mul_531857
            # Getting the type of 'True' (line 265)
            True_531859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 66), 'True', False)
            keyword_531860 = True_531859
            kwargs_531861 = {'regularized': keyword_531860, 'b': keyword_531858}
            # Getting the type of 'mpmath' (line 265)
            mpmath_531852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 28), 'mpmath', False)
            # Obtaining the member 'gammainc' of a type (line 265)
            gammainc_531853 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 28), mpmath_531852, 'gammainc')
            # Calling gammainc(args, kwargs) (line 265)
            gammainc_call_result_531862 = invoke(stypy.reporting.localization.Localization(__file__, 265, 28), gammainc_531853, *[b_531854], **kwargs_531861)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 265)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 12), 'stypy_return_type', gammainc_call_result_531862)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_320' in the type store
            # Getting the type of 'stypy_return_type' (line 265)
            stypy_return_type_531863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 12), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_531863)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_320'
            return stypy_return_type_531863

        # Assigning a type to the variable '_stypy_temp_lambda_320' (line 265)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 12), '_stypy_temp_lambda_320', _stypy_temp_lambda_320)
        # Getting the type of '_stypy_temp_lambda_320' (line 265)
        _stypy_temp_lambda_320_531864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 12), '_stypy_temp_lambda_320')
        int_531865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 12), 'int')
        
        # Obtaining an instance of the builtin type 'list' (line 266)
        list_531866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 266)
        # Adding element type (line 266)
        
        # Call to Arg(...): (line 266)
        # Processing the call arguments (line 266)
        int_531868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 20), 'int')
        float_531869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 23), 'float')
        # Processing the call keyword arguments (line 266)
        # Getting the type of 'False' (line 266)
        False_531870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 40), 'False', False)
        keyword_531871 = False_531870
        kwargs_531872 = {'inclusive_a': keyword_531871}
        # Getting the type of 'Arg' (line 266)
        Arg_531867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 16), 'Arg', False)
        # Calling Arg(args, kwargs) (line 266)
        Arg_call_result_531873 = invoke(stypy.reporting.localization.Localization(__file__, 266, 16), Arg_531867, *[int_531868, float_531869], **kwargs_531872)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 266, 15), list_531866, Arg_call_result_531873)
        # Adding element type (line 266)
        
        # Call to ProbArg(...): (line 266)
        # Processing the call keyword arguments (line 266)
        kwargs_531875 = {}
        # Getting the type of 'ProbArg' (line 266)
        ProbArg_531874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 48), 'ProbArg', False)
        # Calling ProbArg(args, kwargs) (line 266)
        ProbArg_call_result_531876 = invoke(stypy.reporting.localization.Localization(__file__, 266, 48), ProbArg_531874, *[], **kwargs_531875)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 266, 15), list_531866, ProbArg_call_result_531876)
        # Adding element type (line 266)
        
        # Call to Arg(...): (line 267)
        # Processing the call arguments (line 267)
        int_531878 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 20), 'int')
        float_531879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 23), 'float')
        # Processing the call keyword arguments (line 267)
        # Getting the type of 'False' (line 267)
        False_531880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 40), 'False', False)
        keyword_531881 = False_531880
        kwargs_531882 = {'inclusive_a': keyword_531881}
        # Getting the type of 'Arg' (line 267)
        Arg_531877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 16), 'Arg', False)
        # Calling Arg(args, kwargs) (line 267)
        Arg_call_result_531883 = invoke(stypy.reporting.localization.Localization(__file__, 267, 16), Arg_531877, *[int_531878, float_531879], **kwargs_531882)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 266, 15), list_531866, Arg_call_result_531883)
        
        # Processing the call keyword arguments (line 263)
        float_531884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 54), 'float')
        keyword_531885 = float_531884
        kwargs_531886 = {'rtol': keyword_531885}
        # Getting the type of '_assert_inverts' (line 263)
        _assert_inverts_531849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 8), '_assert_inverts', False)
        # Calling _assert_inverts(args, kwargs) (line 263)
        _assert_inverts_call_result_531887 = invoke(stypy.reporting.localization.Localization(__file__, 263, 8), _assert_inverts_531849, *[gdtrib_531851, _stypy_temp_lambda_320_531864, int_531865, list_531866], **kwargs_531886)
        
        
        # ################# End of 'test_gdtrib(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_gdtrib' in the type store
        # Getting the type of 'stypy_return_type' (line 261)
        stypy_return_type_531888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_531888)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_gdtrib'
        return stypy_return_type_531888


    @norecursion
    def test_gdtrix(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_gdtrix'
        module_type_store = module_type_store.open_function_context('test_gdtrix', 269, 4, False)
        # Assigning a type to the variable 'self' (line 270)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCDFlib.test_gdtrix.__dict__.__setitem__('stypy_localization', localization)
        TestCDFlib.test_gdtrix.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCDFlib.test_gdtrix.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCDFlib.test_gdtrix.__dict__.__setitem__('stypy_function_name', 'TestCDFlib.test_gdtrix')
        TestCDFlib.test_gdtrix.__dict__.__setitem__('stypy_param_names_list', [])
        TestCDFlib.test_gdtrix.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCDFlib.test_gdtrix.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCDFlib.test_gdtrix.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCDFlib.test_gdtrix.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCDFlib.test_gdtrix.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCDFlib.test_gdtrix.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCDFlib.test_gdtrix', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_gdtrix', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_gdtrix(...)' code ##################

        
        # Call to _assert_inverts(...): (line 270)
        # Processing the call arguments (line 270)
        # Getting the type of 'sp' (line 271)
        sp_531890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 12), 'sp', False)
        # Obtaining the member 'gdtrix' of a type (line 271)
        gdtrix_531891 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 12), sp_531890, 'gdtrix')

        @norecursion
        def _stypy_temp_lambda_321(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_321'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_321', 272, 12, True)
            # Passed parameters checking function
            _stypy_temp_lambda_321.stypy_localization = localization
            _stypy_temp_lambda_321.stypy_type_of_self = None
            _stypy_temp_lambda_321.stypy_type_store = module_type_store
            _stypy_temp_lambda_321.stypy_function_name = '_stypy_temp_lambda_321'
            _stypy_temp_lambda_321.stypy_param_names_list = ['a', 'b', 'x']
            _stypy_temp_lambda_321.stypy_varargs_param_name = None
            _stypy_temp_lambda_321.stypy_kwargs_param_name = None
            _stypy_temp_lambda_321.stypy_call_defaults = defaults
            _stypy_temp_lambda_321.stypy_call_varargs = varargs
            _stypy_temp_lambda_321.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_321', ['a', 'b', 'x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_321', ['a', 'b', 'x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Call to gammainc(...): (line 272)
            # Processing the call arguments (line 272)
            # Getting the type of 'b' (line 272)
            b_531894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 44), 'b', False)
            # Processing the call keyword arguments (line 272)
            # Getting the type of 'a' (line 272)
            a_531895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 49), 'a', False)
            # Getting the type of 'x' (line 272)
            x_531896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 51), 'x', False)
            # Applying the binary operator '*' (line 272)
            result_mul_531897 = python_operator(stypy.reporting.localization.Localization(__file__, 272, 49), '*', a_531895, x_531896)
            
            keyword_531898 = result_mul_531897
            # Getting the type of 'True' (line 272)
            True_531899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 66), 'True', False)
            keyword_531900 = True_531899
            kwargs_531901 = {'regularized': keyword_531900, 'b': keyword_531898}
            # Getting the type of 'mpmath' (line 272)
            mpmath_531892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 28), 'mpmath', False)
            # Obtaining the member 'gammainc' of a type (line 272)
            gammainc_531893 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 28), mpmath_531892, 'gammainc')
            # Calling gammainc(args, kwargs) (line 272)
            gammainc_call_result_531902 = invoke(stypy.reporting.localization.Localization(__file__, 272, 28), gammainc_531893, *[b_531894], **kwargs_531901)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 272)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 12), 'stypy_return_type', gammainc_call_result_531902)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_321' in the type store
            # Getting the type of 'stypy_return_type' (line 272)
            stypy_return_type_531903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 12), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_531903)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_321'
            return stypy_return_type_531903

        # Assigning a type to the variable '_stypy_temp_lambda_321' (line 272)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 12), '_stypy_temp_lambda_321', _stypy_temp_lambda_321)
        # Getting the type of '_stypy_temp_lambda_321' (line 272)
        _stypy_temp_lambda_321_531904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 12), '_stypy_temp_lambda_321')
        int_531905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 12), 'int')
        
        # Obtaining an instance of the builtin type 'list' (line 273)
        list_531906 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 273)
        # Adding element type (line 273)
        
        # Call to Arg(...): (line 273)
        # Processing the call arguments (line 273)
        int_531908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 20), 'int')
        float_531909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 23), 'float')
        # Processing the call keyword arguments (line 273)
        # Getting the type of 'False' (line 273)
        False_531910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 40), 'False', False)
        keyword_531911 = False_531910
        kwargs_531912 = {'inclusive_a': keyword_531911}
        # Getting the type of 'Arg' (line 273)
        Arg_531907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 16), 'Arg', False)
        # Calling Arg(args, kwargs) (line 273)
        Arg_call_result_531913 = invoke(stypy.reporting.localization.Localization(__file__, 273, 16), Arg_531907, *[int_531908, float_531909], **kwargs_531912)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 273, 15), list_531906, Arg_call_result_531913)
        # Adding element type (line 273)
        
        # Call to Arg(...): (line 273)
        # Processing the call arguments (line 273)
        int_531915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 52), 'int')
        float_531916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 55), 'float')
        # Processing the call keyword arguments (line 273)
        # Getting the type of 'False' (line 273)
        False_531917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 72), 'False', False)
        keyword_531918 = False_531917
        kwargs_531919 = {'inclusive_a': keyword_531918}
        # Getting the type of 'Arg' (line 273)
        Arg_531914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 48), 'Arg', False)
        # Calling Arg(args, kwargs) (line 273)
        Arg_call_result_531920 = invoke(stypy.reporting.localization.Localization(__file__, 273, 48), Arg_531914, *[int_531915, float_531916], **kwargs_531919)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 273, 15), list_531906, Arg_call_result_531920)
        # Adding element type (line 273)
        
        # Call to ProbArg(...): (line 274)
        # Processing the call keyword arguments (line 274)
        kwargs_531922 = {}
        # Getting the type of 'ProbArg' (line 274)
        ProbArg_531921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 16), 'ProbArg', False)
        # Calling ProbArg(args, kwargs) (line 274)
        ProbArg_call_result_531923 = invoke(stypy.reporting.localization.Localization(__file__, 274, 16), ProbArg_531921, *[], **kwargs_531922)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 273, 15), list_531906, ProbArg_call_result_531923)
        
        # Processing the call keyword arguments (line 270)
        float_531924 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 33), 'float')
        keyword_531925 = float_531924
        
        # Obtaining an instance of the builtin type 'list' (line 275)
        list_531926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 275)
        # Adding element type (line 275)
        # Getting the type of 'None' (line 275)
        None_531927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 24), 'None', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 23), list_531926, None_531927)
        # Adding element type (line 275)
        float_531928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 23), list_531926, float_531928)
        # Adding element type (line 275)
        float_531929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 37), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 23), list_531926, float_531929)
        
        keyword_531930 = list_531926
        kwargs_531931 = {'rtol': keyword_531925, 'endpt_atol': keyword_531930}
        # Getting the type of '_assert_inverts' (line 270)
        _assert_inverts_531889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 8), '_assert_inverts', False)
        # Calling _assert_inverts(args, kwargs) (line 270)
        _assert_inverts_call_result_531932 = invoke(stypy.reporting.localization.Localization(__file__, 270, 8), _assert_inverts_531889, *[gdtrix_531891, _stypy_temp_lambda_321_531904, int_531905, list_531906], **kwargs_531931)
        
        
        # ################# End of 'test_gdtrix(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_gdtrix' in the type store
        # Getting the type of 'stypy_return_type' (line 269)
        stypy_return_type_531933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_531933)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_gdtrix'
        return stypy_return_type_531933


    @norecursion
    def test_stdtr(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_stdtr'
        module_type_store = module_type_store.open_function_context('test_stdtr', 277, 4, False)
        # Assigning a type to the variable 'self' (line 278)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCDFlib.test_stdtr.__dict__.__setitem__('stypy_localization', localization)
        TestCDFlib.test_stdtr.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCDFlib.test_stdtr.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCDFlib.test_stdtr.__dict__.__setitem__('stypy_function_name', 'TestCDFlib.test_stdtr')
        TestCDFlib.test_stdtr.__dict__.__setitem__('stypy_param_names_list', [])
        TestCDFlib.test_stdtr.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCDFlib.test_stdtr.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCDFlib.test_stdtr.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCDFlib.test_stdtr.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCDFlib.test_stdtr.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCDFlib.test_stdtr.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCDFlib.test_stdtr', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_stdtr', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_stdtr(...)' code ##################

        
        # Call to assert_mpmath_equal(...): (line 279)
        # Processing the call arguments (line 279)
        # Getting the type of 'sp' (line 280)
        sp_531935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 12), 'sp', False)
        # Obtaining the member 'stdtr' of a type (line 280)
        stdtr_531936 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 12), sp_531935, 'stdtr')
        # Getting the type of '_student_t_cdf' (line 281)
        _student_t_cdf_531937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 12), '_student_t_cdf', False)
        
        # Obtaining an instance of the builtin type 'list' (line 282)
        list_531938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 282)
        # Adding element type (line 282)
        
        # Call to IntArg(...): (line 282)
        # Processing the call arguments (line 282)
        int_531940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 20), 'int')
        int_531941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 23), 'int')
        # Processing the call keyword arguments (line 282)
        kwargs_531942 = {}
        # Getting the type of 'IntArg' (line 282)
        IntArg_531939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 13), 'IntArg', False)
        # Calling IntArg(args, kwargs) (line 282)
        IntArg_call_result_531943 = invoke(stypy.reporting.localization.Localization(__file__, 282, 13), IntArg_531939, *[int_531940, int_531941], **kwargs_531942)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 282, 12), list_531938, IntArg_call_result_531943)
        # Adding element type (line 282)
        
        # Call to Arg(...): (line 282)
        # Processing the call arguments (line 282)
        float_531945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 33), 'float')
        # Getting the type of 'np' (line 282)
        np_531946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 40), 'np', False)
        # Obtaining the member 'inf' of a type (line 282)
        inf_531947 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 40), np_531946, 'inf')
        # Processing the call keyword arguments (line 282)
        kwargs_531948 = {}
        # Getting the type of 'Arg' (line 282)
        Arg_531944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 29), 'Arg', False)
        # Calling Arg(args, kwargs) (line 282)
        Arg_call_result_531949 = invoke(stypy.reporting.localization.Localization(__file__, 282, 29), Arg_531944, *[float_531945, inf_531947], **kwargs_531948)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 282, 12), list_531938, Arg_call_result_531949)
        
        # Processing the call keyword arguments (line 279)
        float_531950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 55), 'float')
        keyword_531951 = float_531950
        kwargs_531952 = {'rtol': keyword_531951}
        # Getting the type of 'assert_mpmath_equal' (line 279)
        assert_mpmath_equal_531934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 8), 'assert_mpmath_equal', False)
        # Calling assert_mpmath_equal(args, kwargs) (line 279)
        assert_mpmath_equal_call_result_531953 = invoke(stypy.reporting.localization.Localization(__file__, 279, 8), assert_mpmath_equal_531934, *[stdtr_531936, _student_t_cdf_531937, list_531938], **kwargs_531952)
        
        
        # ################# End of 'test_stdtr(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_stdtr' in the type store
        # Getting the type of 'stypy_return_type' (line 277)
        stypy_return_type_531954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_531954)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_stdtr'
        return stypy_return_type_531954


    @norecursion
    def test_stdtridf(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_stdtridf'
        module_type_store = module_type_store.open_function_context('test_stdtridf', 284, 4, False)
        # Assigning a type to the variable 'self' (line 285)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCDFlib.test_stdtridf.__dict__.__setitem__('stypy_localization', localization)
        TestCDFlib.test_stdtridf.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCDFlib.test_stdtridf.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCDFlib.test_stdtridf.__dict__.__setitem__('stypy_function_name', 'TestCDFlib.test_stdtridf')
        TestCDFlib.test_stdtridf.__dict__.__setitem__('stypy_param_names_list', [])
        TestCDFlib.test_stdtridf.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCDFlib.test_stdtridf.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCDFlib.test_stdtridf.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCDFlib.test_stdtridf.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCDFlib.test_stdtridf.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCDFlib.test_stdtridf.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCDFlib.test_stdtridf', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_stdtridf', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_stdtridf(...)' code ##################

        
        # Call to _assert_inverts(...): (line 286)
        # Processing the call arguments (line 286)
        # Getting the type of 'sp' (line 287)
        sp_531956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 12), 'sp', False)
        # Obtaining the member 'stdtridf' of a type (line 287)
        stdtridf_531957 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 12), sp_531956, 'stdtridf')
        # Getting the type of '_student_t_cdf' (line 288)
        _student_t_cdf_531958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 12), '_student_t_cdf', False)
        int_531959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 12), 'int')
        
        # Obtaining an instance of the builtin type 'list' (line 289)
        list_531960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 289)
        # Adding element type (line 289)
        
        # Call to ProbArg(...): (line 289)
        # Processing the call keyword arguments (line 289)
        kwargs_531962 = {}
        # Getting the type of 'ProbArg' (line 289)
        ProbArg_531961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 16), 'ProbArg', False)
        # Calling ProbArg(args, kwargs) (line 289)
        ProbArg_call_result_531963 = invoke(stypy.reporting.localization.Localization(__file__, 289, 16), ProbArg_531961, *[], **kwargs_531962)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 289, 15), list_531960, ProbArg_call_result_531963)
        # Adding element type (line 289)
        
        # Call to Arg(...): (line 289)
        # Processing the call keyword arguments (line 289)
        kwargs_531965 = {}
        # Getting the type of 'Arg' (line 289)
        Arg_531964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 27), 'Arg', False)
        # Calling Arg(args, kwargs) (line 289)
        Arg_call_result_531966 = invoke(stypy.reporting.localization.Localization(__file__, 289, 27), Arg_531964, *[], **kwargs_531965)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 289, 15), list_531960, Arg_call_result_531966)
        
        # Processing the call keyword arguments (line 286)
        float_531967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 40), 'float')
        keyword_531968 = float_531967
        kwargs_531969 = {'rtol': keyword_531968}
        # Getting the type of '_assert_inverts' (line 286)
        _assert_inverts_531955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 8), '_assert_inverts', False)
        # Calling _assert_inverts(args, kwargs) (line 286)
        _assert_inverts_call_result_531970 = invoke(stypy.reporting.localization.Localization(__file__, 286, 8), _assert_inverts_531955, *[stdtridf_531957, _student_t_cdf_531958, int_531959, list_531960], **kwargs_531969)
        
        
        # ################# End of 'test_stdtridf(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_stdtridf' in the type store
        # Getting the type of 'stypy_return_type' (line 284)
        stypy_return_type_531971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_531971)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_stdtridf'
        return stypy_return_type_531971


    @norecursion
    def test_stdtrit(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_stdtrit'
        module_type_store = module_type_store.open_function_context('test_stdtrit', 291, 4, False)
        # Assigning a type to the variable 'self' (line 292)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCDFlib.test_stdtrit.__dict__.__setitem__('stypy_localization', localization)
        TestCDFlib.test_stdtrit.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCDFlib.test_stdtrit.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCDFlib.test_stdtrit.__dict__.__setitem__('stypy_function_name', 'TestCDFlib.test_stdtrit')
        TestCDFlib.test_stdtrit.__dict__.__setitem__('stypy_param_names_list', [])
        TestCDFlib.test_stdtrit.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCDFlib.test_stdtrit.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCDFlib.test_stdtrit.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCDFlib.test_stdtrit.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCDFlib.test_stdtrit.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCDFlib.test_stdtrit.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCDFlib.test_stdtrit', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_stdtrit', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_stdtrit(...)' code ##################

        
        # Call to _assert_inverts(...): (line 292)
        # Processing the call arguments (line 292)
        # Getting the type of 'sp' (line 293)
        sp_531973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 12), 'sp', False)
        # Obtaining the member 'stdtrit' of a type (line 293)
        stdtrit_531974 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 12), sp_531973, 'stdtrit')
        # Getting the type of '_student_t_cdf' (line 294)
        _student_t_cdf_531975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 12), '_student_t_cdf', False)
        int_531976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 12), 'int')
        
        # Obtaining an instance of the builtin type 'list' (line 295)
        list_531977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 295)
        # Adding element type (line 295)
        
        # Call to IntArg(...): (line 295)
        # Processing the call arguments (line 295)
        int_531979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 23), 'int')
        int_531980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 26), 'int')
        # Processing the call keyword arguments (line 295)
        kwargs_531981 = {}
        # Getting the type of 'IntArg' (line 295)
        IntArg_531978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 16), 'IntArg', False)
        # Calling IntArg(args, kwargs) (line 295)
        IntArg_call_result_531982 = invoke(stypy.reporting.localization.Localization(__file__, 295, 16), IntArg_531978, *[int_531979, int_531980], **kwargs_531981)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 15), list_531977, IntArg_call_result_531982)
        # Adding element type (line 295)
        
        # Call to ProbArg(...): (line 295)
        # Processing the call keyword arguments (line 295)
        kwargs_531984 = {}
        # Getting the type of 'ProbArg' (line 295)
        ProbArg_531983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 32), 'ProbArg', False)
        # Calling ProbArg(args, kwargs) (line 295)
        ProbArg_call_result_531985 = invoke(stypy.reporting.localization.Localization(__file__, 295, 32), ProbArg_531983, *[], **kwargs_531984)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 15), list_531977, ProbArg_call_result_531985)
        
        # Processing the call keyword arguments (line 292)
        float_531986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 49), 'float')
        keyword_531987 = float_531986
        
        # Obtaining an instance of the builtin type 'list' (line 296)
        list_531988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 296)
        # Adding element type (line 296)
        # Getting the type of 'None' (line 296)
        None_531989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 24), 'None', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 296, 23), list_531988, None_531989)
        # Adding element type (line 296)
        float_531990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 296, 23), list_531988, float_531990)
        
        keyword_531991 = list_531988
        kwargs_531992 = {'rtol': keyword_531987, 'endpt_atol': keyword_531991}
        # Getting the type of '_assert_inverts' (line 292)
        _assert_inverts_531972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), '_assert_inverts', False)
        # Calling _assert_inverts(args, kwargs) (line 292)
        _assert_inverts_call_result_531993 = invoke(stypy.reporting.localization.Localization(__file__, 292, 8), _assert_inverts_531972, *[stdtrit_531974, _student_t_cdf_531975, int_531976, list_531977], **kwargs_531992)
        
        
        # ################# End of 'test_stdtrit(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_stdtrit' in the type store
        # Getting the type of 'stypy_return_type' (line 291)
        stypy_return_type_531994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_531994)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_stdtrit'
        return stypy_return_type_531994


    @norecursion
    def test_chdtriv(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_chdtriv'
        module_type_store = module_type_store.open_function_context('test_chdtriv', 298, 4, False)
        # Assigning a type to the variable 'self' (line 299)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCDFlib.test_chdtriv.__dict__.__setitem__('stypy_localization', localization)
        TestCDFlib.test_chdtriv.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCDFlib.test_chdtriv.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCDFlib.test_chdtriv.__dict__.__setitem__('stypy_function_name', 'TestCDFlib.test_chdtriv')
        TestCDFlib.test_chdtriv.__dict__.__setitem__('stypy_param_names_list', [])
        TestCDFlib.test_chdtriv.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCDFlib.test_chdtriv.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCDFlib.test_chdtriv.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCDFlib.test_chdtriv.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCDFlib.test_chdtriv.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCDFlib.test_chdtriv.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCDFlib.test_chdtriv', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_chdtriv', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_chdtriv(...)' code ##################

        
        # Call to _assert_inverts(...): (line 299)
        # Processing the call arguments (line 299)
        # Getting the type of 'sp' (line 300)
        sp_531996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 12), 'sp', False)
        # Obtaining the member 'chdtriv' of a type (line 300)
        chdtriv_531997 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 12), sp_531996, 'chdtriv')

        @norecursion
        def _stypy_temp_lambda_322(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_322'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_322', 301, 12, True)
            # Passed parameters checking function
            _stypy_temp_lambda_322.stypy_localization = localization
            _stypy_temp_lambda_322.stypy_type_of_self = None
            _stypy_temp_lambda_322.stypy_type_store = module_type_store
            _stypy_temp_lambda_322.stypy_function_name = '_stypy_temp_lambda_322'
            _stypy_temp_lambda_322.stypy_param_names_list = ['v', 'x']
            _stypy_temp_lambda_322.stypy_varargs_param_name = None
            _stypy_temp_lambda_322.stypy_kwargs_param_name = None
            _stypy_temp_lambda_322.stypy_call_defaults = defaults
            _stypy_temp_lambda_322.stypy_call_varargs = varargs
            _stypy_temp_lambda_322.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_322', ['v', 'x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_322', ['v', 'x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Call to gammainc(...): (line 301)
            # Processing the call arguments (line 301)
            # Getting the type of 'v' (line 301)
            v_532000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 41), 'v', False)
            int_532001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 43), 'int')
            # Applying the binary operator 'div' (line 301)
            result_div_532002 = python_operator(stypy.reporting.localization.Localization(__file__, 301, 41), 'div', v_532000, int_532001)
            
            # Processing the call keyword arguments (line 301)
            # Getting the type of 'x' (line 301)
            x_532003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 48), 'x', False)
            int_532004 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 50), 'int')
            # Applying the binary operator 'div' (line 301)
            result_div_532005 = python_operator(stypy.reporting.localization.Localization(__file__, 301, 48), 'div', x_532003, int_532004)
            
            keyword_532006 = result_div_532005
            # Getting the type of 'True' (line 301)
            True_532007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 65), 'True', False)
            keyword_532008 = True_532007
            kwargs_532009 = {'regularized': keyword_532008, 'b': keyword_532006}
            # Getting the type of 'mpmath' (line 301)
            mpmath_531998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 25), 'mpmath', False)
            # Obtaining the member 'gammainc' of a type (line 301)
            gammainc_531999 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 25), mpmath_531998, 'gammainc')
            # Calling gammainc(args, kwargs) (line 301)
            gammainc_call_result_532010 = invoke(stypy.reporting.localization.Localization(__file__, 301, 25), gammainc_531999, *[result_div_532002], **kwargs_532009)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 301)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 12), 'stypy_return_type', gammainc_call_result_532010)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_322' in the type store
            # Getting the type of 'stypy_return_type' (line 301)
            stypy_return_type_532011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 12), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_532011)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_322'
            return stypy_return_type_532011

        # Assigning a type to the variable '_stypy_temp_lambda_322' (line 301)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 12), '_stypy_temp_lambda_322', _stypy_temp_lambda_322)
        # Getting the type of '_stypy_temp_lambda_322' (line 301)
        _stypy_temp_lambda_322_532012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 12), '_stypy_temp_lambda_322')
        int_532013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 12), 'int')
        
        # Obtaining an instance of the builtin type 'list' (line 302)
        list_532014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 302)
        # Adding element type (line 302)
        
        # Call to ProbArg(...): (line 302)
        # Processing the call keyword arguments (line 302)
        kwargs_532016 = {}
        # Getting the type of 'ProbArg' (line 302)
        ProbArg_532015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 16), 'ProbArg', False)
        # Calling ProbArg(args, kwargs) (line 302)
        ProbArg_call_result_532017 = invoke(stypy.reporting.localization.Localization(__file__, 302, 16), ProbArg_532015, *[], **kwargs_532016)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 302, 15), list_532014, ProbArg_call_result_532017)
        # Adding element type (line 302)
        
        # Call to IntArg(...): (line 302)
        # Processing the call arguments (line 302)
        int_532019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 34), 'int')
        int_532020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 37), 'int')
        # Processing the call keyword arguments (line 302)
        kwargs_532021 = {}
        # Getting the type of 'IntArg' (line 302)
        IntArg_532018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 27), 'IntArg', False)
        # Calling IntArg(args, kwargs) (line 302)
        IntArg_call_result_532022 = invoke(stypy.reporting.localization.Localization(__file__, 302, 27), IntArg_532018, *[int_532019, int_532020], **kwargs_532021)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 302, 15), list_532014, IntArg_call_result_532022)
        
        # Processing the call keyword arguments (line 299)
        float_532023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 49), 'float')
        keyword_532024 = float_532023
        kwargs_532025 = {'rtol': keyword_532024}
        # Getting the type of '_assert_inverts' (line 299)
        _assert_inverts_531995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 8), '_assert_inverts', False)
        # Calling _assert_inverts(args, kwargs) (line 299)
        _assert_inverts_call_result_532026 = invoke(stypy.reporting.localization.Localization(__file__, 299, 8), _assert_inverts_531995, *[chdtriv_531997, _stypy_temp_lambda_322_532012, int_532013, list_532014], **kwargs_532025)
        
        
        # ################# End of 'test_chdtriv(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_chdtriv' in the type store
        # Getting the type of 'stypy_return_type' (line 298)
        stypy_return_type_532027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_532027)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_chdtriv'
        return stypy_return_type_532027


    @norecursion
    def test_chndtridf(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_chndtridf'
        module_type_store = module_type_store.open_function_context('test_chndtridf', 304, 4, False)
        # Assigning a type to the variable 'self' (line 305)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 305, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCDFlib.test_chndtridf.__dict__.__setitem__('stypy_localization', localization)
        TestCDFlib.test_chndtridf.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCDFlib.test_chndtridf.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCDFlib.test_chndtridf.__dict__.__setitem__('stypy_function_name', 'TestCDFlib.test_chndtridf')
        TestCDFlib.test_chndtridf.__dict__.__setitem__('stypy_param_names_list', [])
        TestCDFlib.test_chndtridf.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCDFlib.test_chndtridf.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCDFlib.test_chndtridf.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCDFlib.test_chndtridf.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCDFlib.test_chndtridf.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCDFlib.test_chndtridf.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCDFlib.test_chndtridf', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_chndtridf', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_chndtridf(...)' code ##################

        
        # Call to _assert_inverts(...): (line 307)
        # Processing the call arguments (line 307)
        # Getting the type of 'sp' (line 308)
        sp_532029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 12), 'sp', False)
        # Obtaining the member 'chndtridf' of a type (line 308)
        chndtridf_532030 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 12), sp_532029, 'chndtridf')
        # Getting the type of '_noncentral_chi_cdf' (line 309)
        _noncentral_chi_cdf_532031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 12), '_noncentral_chi_cdf', False)
        int_532032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 12), 'int')
        
        # Obtaining an instance of the builtin type 'list' (line 310)
        list_532033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 310)
        # Adding element type (line 310)
        
        # Call to Arg(...): (line 310)
        # Processing the call arguments (line 310)
        int_532035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 20), 'int')
        int_532036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 23), 'int')
        # Processing the call keyword arguments (line 310)
        # Getting the type of 'False' (line 310)
        False_532037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 40), 'False', False)
        keyword_532038 = False_532037
        kwargs_532039 = {'inclusive_a': keyword_532038}
        # Getting the type of 'Arg' (line 310)
        Arg_532034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 16), 'Arg', False)
        # Calling Arg(args, kwargs) (line 310)
        Arg_call_result_532040 = invoke(stypy.reporting.localization.Localization(__file__, 310, 16), Arg_532034, *[int_532035, int_532036], **kwargs_532039)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 310, 15), list_532033, Arg_call_result_532040)
        # Adding element type (line 310)
        
        # Call to ProbArg(...): (line 310)
        # Processing the call keyword arguments (line 310)
        kwargs_532042 = {}
        # Getting the type of 'ProbArg' (line 310)
        ProbArg_532041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 48), 'ProbArg', False)
        # Calling ProbArg(args, kwargs) (line 310)
        ProbArg_call_result_532043 = invoke(stypy.reporting.localization.Localization(__file__, 310, 48), ProbArg_532041, *[], **kwargs_532042)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 310, 15), list_532033, ProbArg_call_result_532043)
        # Adding element type (line 310)
        
        # Call to Arg(...): (line 311)
        # Processing the call arguments (line 311)
        int_532045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 20), 'int')
        int_532046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 23), 'int')
        # Processing the call keyword arguments (line 311)
        # Getting the type of 'False' (line 311)
        False_532047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 40), 'False', False)
        keyword_532048 = False_532047
        kwargs_532049 = {'inclusive_a': keyword_532048}
        # Getting the type of 'Arg' (line 311)
        Arg_532044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 16), 'Arg', False)
        # Calling Arg(args, kwargs) (line 311)
        Arg_call_result_532050 = invoke(stypy.reporting.localization.Localization(__file__, 311, 16), Arg_532044, *[int_532045, int_532046], **kwargs_532049)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 310, 15), list_532033, Arg_call_result_532050)
        
        # Processing the call keyword arguments (line 307)
        int_532051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 14), 'int')
        keyword_532052 = int_532051
        float_532053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 25), 'float')
        keyword_532054 = float_532053
        float_532055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 36), 'float')
        keyword_532056 = float_532055
        kwargs_532057 = {'rtol': keyword_532054, 'atol': keyword_532056, 'n': keyword_532052}
        # Getting the type of '_assert_inverts' (line 307)
        _assert_inverts_532028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 8), '_assert_inverts', False)
        # Calling _assert_inverts(args, kwargs) (line 307)
        _assert_inverts_call_result_532058 = invoke(stypy.reporting.localization.Localization(__file__, 307, 8), _assert_inverts_532028, *[chndtridf_532030, _noncentral_chi_cdf_532031, int_532032, list_532033], **kwargs_532057)
        
        
        # ################# End of 'test_chndtridf(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_chndtridf' in the type store
        # Getting the type of 'stypy_return_type' (line 304)
        stypy_return_type_532059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_532059)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_chndtridf'
        return stypy_return_type_532059


    @norecursion
    def test_chndtrinc(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_chndtrinc'
        module_type_store = module_type_store.open_function_context('test_chndtrinc', 314, 4, False)
        # Assigning a type to the variable 'self' (line 315)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCDFlib.test_chndtrinc.__dict__.__setitem__('stypy_localization', localization)
        TestCDFlib.test_chndtrinc.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCDFlib.test_chndtrinc.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCDFlib.test_chndtrinc.__dict__.__setitem__('stypy_function_name', 'TestCDFlib.test_chndtrinc')
        TestCDFlib.test_chndtrinc.__dict__.__setitem__('stypy_param_names_list', [])
        TestCDFlib.test_chndtrinc.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCDFlib.test_chndtrinc.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCDFlib.test_chndtrinc.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCDFlib.test_chndtrinc.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCDFlib.test_chndtrinc.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCDFlib.test_chndtrinc.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCDFlib.test_chndtrinc', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_chndtrinc', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_chndtrinc(...)' code ##################

        
        # Call to _assert_inverts(...): (line 317)
        # Processing the call arguments (line 317)
        # Getting the type of 'sp' (line 318)
        sp_532061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 12), 'sp', False)
        # Obtaining the member 'chndtrinc' of a type (line 318)
        chndtrinc_532062 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 12), sp_532061, 'chndtrinc')
        # Getting the type of '_noncentral_chi_cdf' (line 319)
        _noncentral_chi_cdf_532063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 12), '_noncentral_chi_cdf', False)
        int_532064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 12), 'int')
        
        # Obtaining an instance of the builtin type 'list' (line 320)
        list_532065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 320)
        # Adding element type (line 320)
        
        # Call to Arg(...): (line 320)
        # Processing the call arguments (line 320)
        int_532067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 20), 'int')
        int_532068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 23), 'int')
        # Processing the call keyword arguments (line 320)
        # Getting the type of 'False' (line 320)
        False_532069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 40), 'False', False)
        keyword_532070 = False_532069
        kwargs_532071 = {'inclusive_a': keyword_532070}
        # Getting the type of 'Arg' (line 320)
        Arg_532066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 16), 'Arg', False)
        # Calling Arg(args, kwargs) (line 320)
        Arg_call_result_532072 = invoke(stypy.reporting.localization.Localization(__file__, 320, 16), Arg_532066, *[int_532067, int_532068], **kwargs_532071)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 320, 15), list_532065, Arg_call_result_532072)
        # Adding element type (line 320)
        
        # Call to IntArg(...): (line 320)
        # Processing the call arguments (line 320)
        int_532074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 55), 'int')
        int_532075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 58), 'int')
        # Processing the call keyword arguments (line 320)
        kwargs_532076 = {}
        # Getting the type of 'IntArg' (line 320)
        IntArg_532073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 48), 'IntArg', False)
        # Calling IntArg(args, kwargs) (line 320)
        IntArg_call_result_532077 = invoke(stypy.reporting.localization.Localization(__file__, 320, 48), IntArg_532073, *[int_532074, int_532075], **kwargs_532076)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 320, 15), list_532065, IntArg_call_result_532077)
        # Adding element type (line 320)
        
        # Call to ProbArg(...): (line 320)
        # Processing the call keyword arguments (line 320)
        kwargs_532079 = {}
        # Getting the type of 'ProbArg' (line 320)
        ProbArg_532078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 64), 'ProbArg', False)
        # Calling ProbArg(args, kwargs) (line 320)
        ProbArg_call_result_532080 = invoke(stypy.reporting.localization.Localization(__file__, 320, 64), ProbArg_532078, *[], **kwargs_532079)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 320, 15), list_532065, ProbArg_call_result_532080)
        
        # Processing the call keyword arguments (line 317)
        int_532081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 321, 14), 'int')
        keyword_532082 = int_532081
        float_532083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 321, 25), 'float')
        keyword_532084 = float_532083
        float_532085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 321, 36), 'float')
        keyword_532086 = float_532085
        kwargs_532087 = {'rtol': keyword_532084, 'atol': keyword_532086, 'n': keyword_532082}
        # Getting the type of '_assert_inverts' (line 317)
        _assert_inverts_532060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 8), '_assert_inverts', False)
        # Calling _assert_inverts(args, kwargs) (line 317)
        _assert_inverts_call_result_532088 = invoke(stypy.reporting.localization.Localization(__file__, 317, 8), _assert_inverts_532060, *[chndtrinc_532062, _noncentral_chi_cdf_532063, int_532064, list_532065], **kwargs_532087)
        
        
        # ################# End of 'test_chndtrinc(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_chndtrinc' in the type store
        # Getting the type of 'stypy_return_type' (line 314)
        stypy_return_type_532089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_532089)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_chndtrinc'
        return stypy_return_type_532089


    @norecursion
    def test_chndtrix(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_chndtrix'
        module_type_store = module_type_store.open_function_context('test_chndtrix', 323, 4, False)
        # Assigning a type to the variable 'self' (line 324)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCDFlib.test_chndtrix.__dict__.__setitem__('stypy_localization', localization)
        TestCDFlib.test_chndtrix.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCDFlib.test_chndtrix.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCDFlib.test_chndtrix.__dict__.__setitem__('stypy_function_name', 'TestCDFlib.test_chndtrix')
        TestCDFlib.test_chndtrix.__dict__.__setitem__('stypy_param_names_list', [])
        TestCDFlib.test_chndtrix.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCDFlib.test_chndtrix.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCDFlib.test_chndtrix.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCDFlib.test_chndtrix.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCDFlib.test_chndtrix.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCDFlib.test_chndtrix.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCDFlib.test_chndtrix', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_chndtrix', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_chndtrix(...)' code ##################

        
        # Call to _assert_inverts(...): (line 325)
        # Processing the call arguments (line 325)
        # Getting the type of 'sp' (line 326)
        sp_532091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 12), 'sp', False)
        # Obtaining the member 'chndtrix' of a type (line 326)
        chndtrix_532092 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 12), sp_532091, 'chndtrix')
        # Getting the type of '_noncentral_chi_cdf' (line 327)
        _noncentral_chi_cdf_532093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 12), '_noncentral_chi_cdf', False)
        int_532094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 12), 'int')
        
        # Obtaining an instance of the builtin type 'list' (line 328)
        list_532095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 328)
        # Adding element type (line 328)
        
        # Call to ProbArg(...): (line 328)
        # Processing the call keyword arguments (line 328)
        kwargs_532097 = {}
        # Getting the type of 'ProbArg' (line 328)
        ProbArg_532096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 16), 'ProbArg', False)
        # Calling ProbArg(args, kwargs) (line 328)
        ProbArg_call_result_532098 = invoke(stypy.reporting.localization.Localization(__file__, 328, 16), ProbArg_532096, *[], **kwargs_532097)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 328, 15), list_532095, ProbArg_call_result_532098)
        # Adding element type (line 328)
        
        # Call to IntArg(...): (line 328)
        # Processing the call arguments (line 328)
        int_532100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 34), 'int')
        int_532101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 37), 'int')
        # Processing the call keyword arguments (line 328)
        kwargs_532102 = {}
        # Getting the type of 'IntArg' (line 328)
        IntArg_532099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 27), 'IntArg', False)
        # Calling IntArg(args, kwargs) (line 328)
        IntArg_call_result_532103 = invoke(stypy.reporting.localization.Localization(__file__, 328, 27), IntArg_532099, *[int_532100, int_532101], **kwargs_532102)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 328, 15), list_532095, IntArg_call_result_532103)
        # Adding element type (line 328)
        
        # Call to Arg(...): (line 328)
        # Processing the call arguments (line 328)
        int_532105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 47), 'int')
        int_532106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 50), 'int')
        # Processing the call keyword arguments (line 328)
        # Getting the type of 'False' (line 328)
        False_532107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 67), 'False', False)
        keyword_532108 = False_532107
        kwargs_532109 = {'inclusive_a': keyword_532108}
        # Getting the type of 'Arg' (line 328)
        Arg_532104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 43), 'Arg', False)
        # Calling Arg(args, kwargs) (line 328)
        Arg_call_result_532110 = invoke(stypy.reporting.localization.Localization(__file__, 328, 43), Arg_532104, *[int_532105, int_532106], **kwargs_532109)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 328, 15), list_532095, Arg_call_result_532110)
        
        # Processing the call keyword arguments (line 325)
        int_532111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 14), 'int')
        keyword_532112 = int_532111
        float_532113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 25), 'float')
        keyword_532114 = float_532113
        float_532115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 36), 'float')
        keyword_532116 = float_532115
        
        # Obtaining an instance of the builtin type 'list' (line 330)
        list_532117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 330)
        # Adding element type (line 330)
        float_532118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 24), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 330, 23), list_532117, float_532118)
        # Adding element type (line 330)
        # Getting the type of 'None' (line 330)
        None_532119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 30), 'None', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 330, 23), list_532117, None_532119)
        # Adding element type (line 330)
        # Getting the type of 'None' (line 330)
        None_532120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 36), 'None', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 330, 23), list_532117, None_532120)
        
        keyword_532121 = list_532117
        kwargs_532122 = {'rtol': keyword_532114, 'endpt_atol': keyword_532121, 'atol': keyword_532116, 'n': keyword_532112}
        # Getting the type of '_assert_inverts' (line 325)
        _assert_inverts_532090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 8), '_assert_inverts', False)
        # Calling _assert_inverts(args, kwargs) (line 325)
        _assert_inverts_call_result_532123 = invoke(stypy.reporting.localization.Localization(__file__, 325, 8), _assert_inverts_532090, *[chndtrix_532092, _noncentral_chi_cdf_532093, int_532094, list_532095], **kwargs_532122)
        
        
        # ################# End of 'test_chndtrix(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_chndtrix' in the type store
        # Getting the type of 'stypy_return_type' (line 323)
        stypy_return_type_532124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_532124)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_chndtrix'
        return stypy_return_type_532124


    @norecursion
    def test_tklmbda_zero_shape(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_tklmbda_zero_shape'
        module_type_store = module_type_store.open_function_context('test_tklmbda_zero_shape', 332, 4, False)
        # Assigning a type to the variable 'self' (line 333)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 333, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCDFlib.test_tklmbda_zero_shape.__dict__.__setitem__('stypy_localization', localization)
        TestCDFlib.test_tklmbda_zero_shape.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCDFlib.test_tklmbda_zero_shape.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCDFlib.test_tklmbda_zero_shape.__dict__.__setitem__('stypy_function_name', 'TestCDFlib.test_tklmbda_zero_shape')
        TestCDFlib.test_tklmbda_zero_shape.__dict__.__setitem__('stypy_param_names_list', [])
        TestCDFlib.test_tklmbda_zero_shape.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCDFlib.test_tklmbda_zero_shape.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCDFlib.test_tklmbda_zero_shape.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCDFlib.test_tklmbda_zero_shape.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCDFlib.test_tklmbda_zero_shape.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCDFlib.test_tklmbda_zero_shape.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCDFlib.test_tklmbda_zero_shape', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_tklmbda_zero_shape', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_tklmbda_zero_shape(...)' code ##################

        
        # Assigning a Call to a Name (line 334):
        
        # Assigning a Call to a Name (line 334):
        
        # Call to mpf(...): (line 334)
        # Processing the call arguments (line 334)
        int_532127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 25), 'int')
        # Processing the call keyword arguments (line 334)
        kwargs_532128 = {}
        # Getting the type of 'mpmath' (line 334)
        mpmath_532125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 14), 'mpmath', False)
        # Obtaining the member 'mpf' of a type (line 334)
        mpf_532126 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 14), mpmath_532125, 'mpf')
        # Calling mpf(args, kwargs) (line 334)
        mpf_call_result_532129 = invoke(stypy.reporting.localization.Localization(__file__, 334, 14), mpf_532126, *[int_532127], **kwargs_532128)
        
        # Assigning a type to the variable 'one' (line 334)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 8), 'one', mpf_call_result_532129)
        
        # Call to assert_mpmath_equal(...): (line 335)
        # Processing the call arguments (line 335)

        @norecursion
        def _stypy_temp_lambda_323(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_323'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_323', 336, 12, True)
            # Passed parameters checking function
            _stypy_temp_lambda_323.stypy_localization = localization
            _stypy_temp_lambda_323.stypy_type_of_self = None
            _stypy_temp_lambda_323.stypy_type_store = module_type_store
            _stypy_temp_lambda_323.stypy_function_name = '_stypy_temp_lambda_323'
            _stypy_temp_lambda_323.stypy_param_names_list = ['x']
            _stypy_temp_lambda_323.stypy_varargs_param_name = None
            _stypy_temp_lambda_323.stypy_kwargs_param_name = None
            _stypy_temp_lambda_323.stypy_call_defaults = defaults
            _stypy_temp_lambda_323.stypy_call_varargs = varargs
            _stypy_temp_lambda_323.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_323', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_323', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Call to tklmbda(...): (line 336)
            # Processing the call arguments (line 336)
            # Getting the type of 'x' (line 336)
            x_532133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 33), 'x', False)
            int_532134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 36), 'int')
            # Processing the call keyword arguments (line 336)
            kwargs_532135 = {}
            # Getting the type of 'sp' (line 336)
            sp_532131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 22), 'sp', False)
            # Obtaining the member 'tklmbda' of a type (line 336)
            tklmbda_532132 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 22), sp_532131, 'tklmbda')
            # Calling tklmbda(args, kwargs) (line 336)
            tklmbda_call_result_532136 = invoke(stypy.reporting.localization.Localization(__file__, 336, 22), tklmbda_532132, *[x_532133, int_532134], **kwargs_532135)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 336)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 12), 'stypy_return_type', tklmbda_call_result_532136)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_323' in the type store
            # Getting the type of 'stypy_return_type' (line 336)
            stypy_return_type_532137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 12), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_532137)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_323'
            return stypy_return_type_532137

        # Assigning a type to the variable '_stypy_temp_lambda_323' (line 336)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 12), '_stypy_temp_lambda_323', _stypy_temp_lambda_323)
        # Getting the type of '_stypy_temp_lambda_323' (line 336)
        _stypy_temp_lambda_323_532138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 12), '_stypy_temp_lambda_323')

        @norecursion
        def _stypy_temp_lambda_324(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_324'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_324', 337, 12, True)
            # Passed parameters checking function
            _stypy_temp_lambda_324.stypy_localization = localization
            _stypy_temp_lambda_324.stypy_type_of_self = None
            _stypy_temp_lambda_324.stypy_type_store = module_type_store
            _stypy_temp_lambda_324.stypy_function_name = '_stypy_temp_lambda_324'
            _stypy_temp_lambda_324.stypy_param_names_list = ['x']
            _stypy_temp_lambda_324.stypy_varargs_param_name = None
            _stypy_temp_lambda_324.stypy_kwargs_param_name = None
            _stypy_temp_lambda_324.stypy_call_defaults = defaults
            _stypy_temp_lambda_324.stypy_call_varargs = varargs
            _stypy_temp_lambda_324.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_324', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_324', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            # Getting the type of 'one' (line 337)
            one_532139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 22), 'one', False)
            
            # Call to exp(...): (line 337)
            # Processing the call arguments (line 337)
            
            # Getting the type of 'x' (line 337)
            x_532142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 39), 'x', False)
            # Applying the 'usub' unary operator (line 337)
            result___neg___532143 = python_operator(stypy.reporting.localization.Localization(__file__, 337, 38), 'usub', x_532142)
            
            # Processing the call keyword arguments (line 337)
            kwargs_532144 = {}
            # Getting the type of 'mpmath' (line 337)
            mpmath_532140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 27), 'mpmath', False)
            # Obtaining the member 'exp' of a type (line 337)
            exp_532141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 27), mpmath_532140, 'exp')
            # Calling exp(args, kwargs) (line 337)
            exp_call_result_532145 = invoke(stypy.reporting.localization.Localization(__file__, 337, 27), exp_532141, *[result___neg___532143], **kwargs_532144)
            
            # Getting the type of 'one' (line 337)
            one_532146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 44), 'one', False)
            # Applying the binary operator '+' (line 337)
            result_add_532147 = python_operator(stypy.reporting.localization.Localization(__file__, 337, 27), '+', exp_call_result_532145, one_532146)
            
            # Applying the binary operator 'div' (line 337)
            result_div_532148 = python_operator(stypy.reporting.localization.Localization(__file__, 337, 22), 'div', one_532139, result_add_532147)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 337)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 337, 12), 'stypy_return_type', result_div_532148)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_324' in the type store
            # Getting the type of 'stypy_return_type' (line 337)
            stypy_return_type_532149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 12), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_532149)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_324'
            return stypy_return_type_532149

        # Assigning a type to the variable '_stypy_temp_lambda_324' (line 337)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 337, 12), '_stypy_temp_lambda_324', _stypy_temp_lambda_324)
        # Getting the type of '_stypy_temp_lambda_324' (line 337)
        _stypy_temp_lambda_324_532150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 12), '_stypy_temp_lambda_324')
        
        # Obtaining an instance of the builtin type 'list' (line 338)
        list_532151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 338)
        # Adding element type (line 338)
        
        # Call to Arg(...): (line 338)
        # Processing the call keyword arguments (line 338)
        kwargs_532153 = {}
        # Getting the type of 'Arg' (line 338)
        Arg_532152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 13), 'Arg', False)
        # Calling Arg(args, kwargs) (line 338)
        Arg_call_result_532154 = invoke(stypy.reporting.localization.Localization(__file__, 338, 13), Arg_532152, *[], **kwargs_532153)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 338, 12), list_532151, Arg_call_result_532154)
        
        # Processing the call keyword arguments (line 335)
        float_532155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 26), 'float')
        keyword_532156 = float_532155
        kwargs_532157 = {'rtol': keyword_532156}
        # Getting the type of 'assert_mpmath_equal' (line 335)
        assert_mpmath_equal_532130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 8), 'assert_mpmath_equal', False)
        # Calling assert_mpmath_equal(args, kwargs) (line 335)
        assert_mpmath_equal_call_result_532158 = invoke(stypy.reporting.localization.Localization(__file__, 335, 8), assert_mpmath_equal_532130, *[_stypy_temp_lambda_323_532138, _stypy_temp_lambda_324_532150, list_532151], **kwargs_532157)
        
        
        # ################# End of 'test_tklmbda_zero_shape(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_tklmbda_zero_shape' in the type store
        # Getting the type of 'stypy_return_type' (line 332)
        stypy_return_type_532159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_532159)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_tklmbda_zero_shape'
        return stypy_return_type_532159


    @norecursion
    def test_tklmbda_neg_shape(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_tklmbda_neg_shape'
        module_type_store = module_type_store.open_function_context('test_tklmbda_neg_shape', 340, 4, False)
        # Assigning a type to the variable 'self' (line 341)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCDFlib.test_tklmbda_neg_shape.__dict__.__setitem__('stypy_localization', localization)
        TestCDFlib.test_tklmbda_neg_shape.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCDFlib.test_tklmbda_neg_shape.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCDFlib.test_tklmbda_neg_shape.__dict__.__setitem__('stypy_function_name', 'TestCDFlib.test_tklmbda_neg_shape')
        TestCDFlib.test_tklmbda_neg_shape.__dict__.__setitem__('stypy_param_names_list', [])
        TestCDFlib.test_tklmbda_neg_shape.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCDFlib.test_tklmbda_neg_shape.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCDFlib.test_tklmbda_neg_shape.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCDFlib.test_tklmbda_neg_shape.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCDFlib.test_tklmbda_neg_shape.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCDFlib.test_tklmbda_neg_shape.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCDFlib.test_tklmbda_neg_shape', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_tklmbda_neg_shape', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_tklmbda_neg_shape(...)' code ##################

        
        # Call to _assert_inverts(...): (line 341)
        # Processing the call arguments (line 341)
        # Getting the type of 'sp' (line 342)
        sp_532161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 12), 'sp', False)
        # Obtaining the member 'tklmbda' of a type (line 342)
        tklmbda_532162 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 12), sp_532161, 'tklmbda')
        # Getting the type of '_tukey_lmbda_quantile' (line 343)
        _tukey_lmbda_quantile_532163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 12), '_tukey_lmbda_quantile', False)
        int_532164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 12), 'int')
        
        # Obtaining an instance of the builtin type 'list' (line 344)
        list_532165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 344)
        # Adding element type (line 344)
        
        # Call to ProbArg(...): (line 344)
        # Processing the call keyword arguments (line 344)
        kwargs_532167 = {}
        # Getting the type of 'ProbArg' (line 344)
        ProbArg_532166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 16), 'ProbArg', False)
        # Calling ProbArg(args, kwargs) (line 344)
        ProbArg_call_result_532168 = invoke(stypy.reporting.localization.Localization(__file__, 344, 16), ProbArg_532166, *[], **kwargs_532167)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 344, 15), list_532165, ProbArg_call_result_532168)
        # Adding element type (line 344)
        
        # Call to Arg(...): (line 344)
        # Processing the call arguments (line 344)
        
        # Getting the type of 'np' (line 344)
        np_532170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 32), 'np', False)
        # Obtaining the member 'inf' of a type (line 344)
        inf_532171 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 32), np_532170, 'inf')
        # Applying the 'usub' unary operator (line 344)
        result___neg___532172 = python_operator(stypy.reporting.localization.Localization(__file__, 344, 31), 'usub', inf_532171)
        
        int_532173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 40), 'int')
        # Processing the call keyword arguments (line 344)
        # Getting the type of 'False' (line 344)
        False_532174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 55), 'False', False)
        keyword_532175 = False_532174
        kwargs_532176 = {'inclusive_b': keyword_532175}
        # Getting the type of 'Arg' (line 344)
        Arg_532169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 27), 'Arg', False)
        # Calling Arg(args, kwargs) (line 344)
        Arg_call_result_532177 = invoke(stypy.reporting.localization.Localization(__file__, 344, 27), Arg_532169, *[result___neg___532172, int_532173], **kwargs_532176)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 344, 15), list_532165, Arg_call_result_532177)
        
        # Processing the call keyword arguments (line 341)
        # Getting the type of 'False' (line 345)
        False_532178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 25), 'False', False)
        keyword_532179 = False_532178
        float_532180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 37), 'float')
        keyword_532181 = float_532180
        
        # Obtaining an instance of the builtin type 'list' (line 346)
        list_532182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 346)
        # Adding element type (line 346)
        float_532183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, 24), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 346, 23), list_532182, float_532183)
        # Adding element type (line 346)
        # Getting the type of 'None' (line 346)
        None_532184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 30), 'None', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 346, 23), list_532182, None_532184)
        
        keyword_532185 = list_532182
        kwargs_532186 = {'rtol': keyword_532181, 'spfunc_first': keyword_532179, 'endpt_atol': keyword_532185}
        # Getting the type of '_assert_inverts' (line 341)
        _assert_inverts_532160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 8), '_assert_inverts', False)
        # Calling _assert_inverts(args, kwargs) (line 341)
        _assert_inverts_call_result_532187 = invoke(stypy.reporting.localization.Localization(__file__, 341, 8), _assert_inverts_532160, *[tklmbda_532162, _tukey_lmbda_quantile_532163, int_532164, list_532165], **kwargs_532186)
        
        
        # ################# End of 'test_tklmbda_neg_shape(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_tklmbda_neg_shape' in the type store
        # Getting the type of 'stypy_return_type' (line 340)
        stypy_return_type_532188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_532188)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_tklmbda_neg_shape'
        return stypy_return_type_532188


    @norecursion
    def test_tklmbda_pos_shape(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_tklmbda_pos_shape'
        module_type_store = module_type_store.open_function_context('test_tklmbda_pos_shape', 348, 4, False)
        # Assigning a type to the variable 'self' (line 349)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCDFlib.test_tklmbda_pos_shape.__dict__.__setitem__('stypy_localization', localization)
        TestCDFlib.test_tklmbda_pos_shape.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCDFlib.test_tklmbda_pos_shape.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCDFlib.test_tklmbda_pos_shape.__dict__.__setitem__('stypy_function_name', 'TestCDFlib.test_tklmbda_pos_shape')
        TestCDFlib.test_tklmbda_pos_shape.__dict__.__setitem__('stypy_param_names_list', [])
        TestCDFlib.test_tklmbda_pos_shape.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCDFlib.test_tklmbda_pos_shape.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCDFlib.test_tklmbda_pos_shape.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCDFlib.test_tklmbda_pos_shape.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCDFlib.test_tklmbda_pos_shape.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCDFlib.test_tklmbda_pos_shape.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCDFlib.test_tklmbda_pos_shape', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_tklmbda_pos_shape', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_tklmbda_pos_shape(...)' code ##################

        
        # Call to _assert_inverts(...): (line 350)
        # Processing the call arguments (line 350)
        # Getting the type of 'sp' (line 351)
        sp_532190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 12), 'sp', False)
        # Obtaining the member 'tklmbda' of a type (line 351)
        tklmbda_532191 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 351, 12), sp_532190, 'tklmbda')
        # Getting the type of '_tukey_lmbda_quantile' (line 352)
        _tukey_lmbda_quantile_532192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 12), '_tukey_lmbda_quantile', False)
        int_532193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, 12), 'int')
        
        # Obtaining an instance of the builtin type 'list' (line 353)
        list_532194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 353)
        # Adding element type (line 353)
        
        # Call to ProbArg(...): (line 353)
        # Processing the call keyword arguments (line 353)
        kwargs_532196 = {}
        # Getting the type of 'ProbArg' (line 353)
        ProbArg_532195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 16), 'ProbArg', False)
        # Calling ProbArg(args, kwargs) (line 353)
        ProbArg_call_result_532197 = invoke(stypy.reporting.localization.Localization(__file__, 353, 16), ProbArg_532195, *[], **kwargs_532196)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 353, 15), list_532194, ProbArg_call_result_532197)
        # Adding element type (line 353)
        
        # Call to Arg(...): (line 353)
        # Processing the call arguments (line 353)
        int_532199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, 31), 'int')
        int_532200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, 34), 'int')
        # Processing the call keyword arguments (line 353)
        # Getting the type of 'False' (line 353)
        False_532201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 51), 'False', False)
        keyword_532202 = False_532201
        kwargs_532203 = {'inclusive_a': keyword_532202}
        # Getting the type of 'Arg' (line 353)
        Arg_532198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 27), 'Arg', False)
        # Calling Arg(args, kwargs) (line 353)
        Arg_call_result_532204 = invoke(stypy.reporting.localization.Localization(__file__, 353, 27), Arg_532198, *[int_532199, int_532200], **kwargs_532203)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 353, 15), list_532194, Arg_call_result_532204)
        
        # Processing the call keyword arguments (line 350)
        # Getting the type of 'False' (line 354)
        False_532205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 25), 'False', False)
        keyword_532206 = False_532205
        float_532207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 37), 'float')
        keyword_532208 = float_532207
        kwargs_532209 = {'rtol': keyword_532208, 'spfunc_first': keyword_532206}
        # Getting the type of '_assert_inverts' (line 350)
        _assert_inverts_532189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 8), '_assert_inverts', False)
        # Calling _assert_inverts(args, kwargs) (line 350)
        _assert_inverts_call_result_532210 = invoke(stypy.reporting.localization.Localization(__file__, 350, 8), _assert_inverts_532189, *[tklmbda_532191, _tukey_lmbda_quantile_532192, int_532193, list_532194], **kwargs_532209)
        
        
        # ################# End of 'test_tklmbda_pos_shape(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_tklmbda_pos_shape' in the type store
        # Getting the type of 'stypy_return_type' (line 348)
        stypy_return_type_532211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_532211)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_tklmbda_pos_shape'
        return stypy_return_type_532211


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 209, 0, False)
        # Assigning a type to the variable 'self' (line 210)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCDFlib.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestCDFlib' (line 209)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 0), 'TestCDFlib', TestCDFlib)

@norecursion
def test_nonfinite(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_nonfinite'
    module_type_store = module_type_store.open_function_context('test_nonfinite', 357, 0, False)
    
    # Passed parameters checking function
    test_nonfinite.stypy_localization = localization
    test_nonfinite.stypy_type_of_self = None
    test_nonfinite.stypy_type_store = module_type_store
    test_nonfinite.stypy_function_name = 'test_nonfinite'
    test_nonfinite.stypy_param_names_list = []
    test_nonfinite.stypy_varargs_param_name = None
    test_nonfinite.stypy_kwargs_param_name = None
    test_nonfinite.stypy_call_defaults = defaults
    test_nonfinite.stypy_call_varargs = varargs
    test_nonfinite.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_nonfinite', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_nonfinite', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_nonfinite(...)' code ##################

    
    # Assigning a List to a Name (line 358):
    
    # Assigning a List to a Name (line 358):
    
    # Obtaining an instance of the builtin type 'list' (line 358)
    list_532212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 12), 'list')
    # Adding type elements to the builtin type 'list' instance (line 358)
    # Adding element type (line 358)
    
    # Obtaining an instance of the builtin type 'tuple' (line 359)
    tuple_532213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 359)
    # Adding element type (line 359)
    str_532214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 9), 'str', 'btdtria')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 359, 9), tuple_532213, str_532214)
    # Adding element type (line 359)
    int_532215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 359, 9), tuple_532213, int_532215)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 358, 12), list_532212, tuple_532213)
    # Adding element type (line 358)
    
    # Obtaining an instance of the builtin type 'tuple' (line 360)
    tuple_532216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 360, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 360)
    # Adding element type (line 360)
    str_532217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 360, 9), 'str', 'btdtrib')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 360, 9), tuple_532216, str_532217)
    # Adding element type (line 360)
    int_532218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 360, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 360, 9), tuple_532216, int_532218)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 358, 12), list_532212, tuple_532216)
    # Adding element type (line 358)
    
    # Obtaining an instance of the builtin type 'tuple' (line 361)
    tuple_532219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 361)
    # Adding element type (line 361)
    str_532220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 9), 'str', 'bdtrik')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 361, 9), tuple_532219, str_532220)
    # Adding element type (line 361)
    int_532221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 361, 9), tuple_532219, int_532221)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 358, 12), list_532212, tuple_532219)
    # Adding element type (line 358)
    
    # Obtaining an instance of the builtin type 'tuple' (line 362)
    tuple_532222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 362)
    # Adding element type (line 362)
    str_532223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 9), 'str', 'bdtrin')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 362, 9), tuple_532222, str_532223)
    # Adding element type (line 362)
    int_532224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 362, 9), tuple_532222, int_532224)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 358, 12), list_532212, tuple_532222)
    # Adding element type (line 358)
    
    # Obtaining an instance of the builtin type 'tuple' (line 363)
    tuple_532225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 363)
    # Adding element type (line 363)
    str_532226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, 9), 'str', 'chdtriv')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 363, 9), tuple_532225, str_532226)
    # Adding element type (line 363)
    int_532227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 363, 9), tuple_532225, int_532227)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 358, 12), list_532212, tuple_532225)
    # Adding element type (line 358)
    
    # Obtaining an instance of the builtin type 'tuple' (line 364)
    tuple_532228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 364)
    # Adding element type (line 364)
    str_532229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 9), 'str', 'chndtr')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 364, 9), tuple_532228, str_532229)
    # Adding element type (line 364)
    int_532230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 364, 9), tuple_532228, int_532230)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 358, 12), list_532212, tuple_532228)
    # Adding element type (line 358)
    
    # Obtaining an instance of the builtin type 'tuple' (line 365)
    tuple_532231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 365, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 365)
    # Adding element type (line 365)
    str_532232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 365, 9), 'str', 'chndtrix')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 365, 9), tuple_532231, str_532232)
    # Adding element type (line 365)
    int_532233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 365, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 365, 9), tuple_532231, int_532233)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 358, 12), list_532212, tuple_532231)
    # Adding element type (line 358)
    
    # Obtaining an instance of the builtin type 'tuple' (line 366)
    tuple_532234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 366)
    # Adding element type (line 366)
    str_532235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 9), 'str', 'chndtridf')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 366, 9), tuple_532234, str_532235)
    # Adding element type (line 366)
    int_532236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 366, 9), tuple_532234, int_532236)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 358, 12), list_532212, tuple_532234)
    # Adding element type (line 358)
    
    # Obtaining an instance of the builtin type 'tuple' (line 367)
    tuple_532237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 367, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 367)
    # Adding element type (line 367)
    str_532238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 367, 9), 'str', 'chndtrinc')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 367, 9), tuple_532237, str_532238)
    # Adding element type (line 367)
    int_532239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 367, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 367, 9), tuple_532237, int_532239)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 358, 12), list_532212, tuple_532237)
    # Adding element type (line 358)
    
    # Obtaining an instance of the builtin type 'tuple' (line 368)
    tuple_532240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 368)
    # Adding element type (line 368)
    str_532241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, 9), 'str', 'fdtridfd')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 368, 9), tuple_532240, str_532241)
    # Adding element type (line 368)
    int_532242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 368, 9), tuple_532240, int_532242)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 358, 12), list_532212, tuple_532240)
    # Adding element type (line 358)
    
    # Obtaining an instance of the builtin type 'tuple' (line 369)
    tuple_532243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 369, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 369)
    # Adding element type (line 369)
    str_532244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 369, 9), 'str', 'ncfdtr')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 369, 9), tuple_532243, str_532244)
    # Adding element type (line 369)
    int_532245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 369, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 369, 9), tuple_532243, int_532245)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 358, 12), list_532212, tuple_532243)
    # Adding element type (line 358)
    
    # Obtaining an instance of the builtin type 'tuple' (line 370)
    tuple_532246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 370)
    # Adding element type (line 370)
    str_532247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 9), 'str', 'ncfdtri')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 370, 9), tuple_532246, str_532247)
    # Adding element type (line 370)
    int_532248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 370, 9), tuple_532246, int_532248)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 358, 12), list_532212, tuple_532246)
    # Adding element type (line 358)
    
    # Obtaining an instance of the builtin type 'tuple' (line 371)
    tuple_532249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 371, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 371)
    # Adding element type (line 371)
    str_532250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 371, 9), 'str', 'ncfdtridfn')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 371, 9), tuple_532249, str_532250)
    # Adding element type (line 371)
    int_532251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 371, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 371, 9), tuple_532249, int_532251)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 358, 12), list_532212, tuple_532249)
    # Adding element type (line 358)
    
    # Obtaining an instance of the builtin type 'tuple' (line 372)
    tuple_532252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 372)
    # Adding element type (line 372)
    str_532253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, 9), 'str', 'ncfdtridfd')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 372, 9), tuple_532252, str_532253)
    # Adding element type (line 372)
    int_532254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 372, 9), tuple_532252, int_532254)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 358, 12), list_532212, tuple_532252)
    # Adding element type (line 358)
    
    # Obtaining an instance of the builtin type 'tuple' (line 373)
    tuple_532255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 373)
    # Adding element type (line 373)
    str_532256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 9), 'str', 'ncfdtrinc')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 373, 9), tuple_532255, str_532256)
    # Adding element type (line 373)
    int_532257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 373, 9), tuple_532255, int_532257)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 358, 12), list_532212, tuple_532255)
    # Adding element type (line 358)
    
    # Obtaining an instance of the builtin type 'tuple' (line 374)
    tuple_532258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 374)
    # Adding element type (line 374)
    str_532259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 9), 'str', 'gdtrix')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 374, 9), tuple_532258, str_532259)
    # Adding element type (line 374)
    int_532260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 374, 9), tuple_532258, int_532260)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 358, 12), list_532212, tuple_532258)
    # Adding element type (line 358)
    
    # Obtaining an instance of the builtin type 'tuple' (line 375)
    tuple_532261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 375)
    # Adding element type (line 375)
    str_532262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 9), 'str', 'gdtrib')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 375, 9), tuple_532261, str_532262)
    # Adding element type (line 375)
    int_532263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 375, 9), tuple_532261, int_532263)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 358, 12), list_532212, tuple_532261)
    # Adding element type (line 358)
    
    # Obtaining an instance of the builtin type 'tuple' (line 376)
    tuple_532264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 376, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 376)
    # Adding element type (line 376)
    str_532265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 376, 9), 'str', 'gdtria')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 376, 9), tuple_532264, str_532265)
    # Adding element type (line 376)
    int_532266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 376, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 376, 9), tuple_532264, int_532266)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 358, 12), list_532212, tuple_532264)
    # Adding element type (line 358)
    
    # Obtaining an instance of the builtin type 'tuple' (line 377)
    tuple_532267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 377, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 377)
    # Adding element type (line 377)
    str_532268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 377, 9), 'str', 'nbdtrik')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 377, 9), tuple_532267, str_532268)
    # Adding element type (line 377)
    int_532269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 377, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 377, 9), tuple_532267, int_532269)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 358, 12), list_532212, tuple_532267)
    # Adding element type (line 358)
    
    # Obtaining an instance of the builtin type 'tuple' (line 378)
    tuple_532270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 378, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 378)
    # Adding element type (line 378)
    str_532271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 378, 9), 'str', 'nbdtrin')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 378, 9), tuple_532270, str_532271)
    # Adding element type (line 378)
    int_532272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 378, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 378, 9), tuple_532270, int_532272)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 358, 12), list_532212, tuple_532270)
    # Adding element type (line 358)
    
    # Obtaining an instance of the builtin type 'tuple' (line 379)
    tuple_532273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 379, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 379)
    # Adding element type (line 379)
    str_532274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 379, 9), 'str', 'nrdtrimn')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 379, 9), tuple_532273, str_532274)
    # Adding element type (line 379)
    int_532275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 379, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 379, 9), tuple_532273, int_532275)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 358, 12), list_532212, tuple_532273)
    # Adding element type (line 358)
    
    # Obtaining an instance of the builtin type 'tuple' (line 380)
    tuple_532276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 380, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 380)
    # Adding element type (line 380)
    str_532277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 380, 9), 'str', 'nrdtrisd')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 380, 9), tuple_532276, str_532277)
    # Adding element type (line 380)
    int_532278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 380, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 380, 9), tuple_532276, int_532278)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 358, 12), list_532212, tuple_532276)
    # Adding element type (line 358)
    
    # Obtaining an instance of the builtin type 'tuple' (line 381)
    tuple_532279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 381)
    # Adding element type (line 381)
    str_532280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 9), 'str', 'pdtrik')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 381, 9), tuple_532279, str_532280)
    # Adding element type (line 381)
    int_532281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 381, 9), tuple_532279, int_532281)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 358, 12), list_532212, tuple_532279)
    # Adding element type (line 358)
    
    # Obtaining an instance of the builtin type 'tuple' (line 382)
    tuple_532282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 382)
    # Adding element type (line 382)
    str_532283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, 9), 'str', 'stdtr')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 382, 9), tuple_532282, str_532283)
    # Adding element type (line 382)
    int_532284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 382, 9), tuple_532282, int_532284)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 358, 12), list_532212, tuple_532282)
    # Adding element type (line 358)
    
    # Obtaining an instance of the builtin type 'tuple' (line 383)
    tuple_532285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 383)
    # Adding element type (line 383)
    str_532286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 9), 'str', 'stdtrit')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 383, 9), tuple_532285, str_532286)
    # Adding element type (line 383)
    int_532287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 383, 9), tuple_532285, int_532287)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 358, 12), list_532212, tuple_532285)
    # Adding element type (line 358)
    
    # Obtaining an instance of the builtin type 'tuple' (line 384)
    tuple_532288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 384)
    # Adding element type (line 384)
    str_532289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 9), 'str', 'stdtridf')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 384, 9), tuple_532288, str_532289)
    # Adding element type (line 384)
    int_532290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 384, 9), tuple_532288, int_532290)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 358, 12), list_532212, tuple_532288)
    # Adding element type (line 358)
    
    # Obtaining an instance of the builtin type 'tuple' (line 385)
    tuple_532291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 385)
    # Adding element type (line 385)
    str_532292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, 9), 'str', 'nctdtr')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 385, 9), tuple_532291, str_532292)
    # Adding element type (line 385)
    int_532293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 385, 9), tuple_532291, int_532293)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 358, 12), list_532212, tuple_532291)
    # Adding element type (line 358)
    
    # Obtaining an instance of the builtin type 'tuple' (line 386)
    tuple_532294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 386)
    # Adding element type (line 386)
    str_532295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 9), 'str', 'nctdtrit')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 386, 9), tuple_532294, str_532295)
    # Adding element type (line 386)
    int_532296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 386, 9), tuple_532294, int_532296)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 358, 12), list_532212, tuple_532294)
    # Adding element type (line 358)
    
    # Obtaining an instance of the builtin type 'tuple' (line 387)
    tuple_532297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 387, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 387)
    # Adding element type (line 387)
    str_532298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 387, 9), 'str', 'nctdtridf')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 387, 9), tuple_532297, str_532298)
    # Adding element type (line 387)
    int_532299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 387, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 387, 9), tuple_532297, int_532299)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 358, 12), list_532212, tuple_532297)
    # Adding element type (line 358)
    
    # Obtaining an instance of the builtin type 'tuple' (line 388)
    tuple_532300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 388, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 388)
    # Adding element type (line 388)
    str_532301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 388, 9), 'str', 'nctdtrinc')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 388, 9), tuple_532300, str_532301)
    # Adding element type (line 388)
    int_532302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 388, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 388, 9), tuple_532300, int_532302)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 358, 12), list_532212, tuple_532300)
    # Adding element type (line 358)
    
    # Obtaining an instance of the builtin type 'tuple' (line 389)
    tuple_532303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 389)
    # Adding element type (line 389)
    str_532304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 9), 'str', 'tklmbda')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 389, 9), tuple_532303, str_532304)
    # Adding element type (line 389)
    int_532305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 389, 9), tuple_532303, int_532305)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 358, 12), list_532212, tuple_532303)
    
    # Assigning a type to the variable 'funcs' (line 358)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 358, 4), 'funcs', list_532212)
    
    # Call to seed(...): (line 392)
    # Processing the call arguments (line 392)
    int_532309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, 19), 'int')
    # Processing the call keyword arguments (line 392)
    kwargs_532310 = {}
    # Getting the type of 'np' (line 392)
    np_532306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 4), 'np', False)
    # Obtaining the member 'random' of a type (line 392)
    random_532307 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 392, 4), np_532306, 'random')
    # Obtaining the member 'seed' of a type (line 392)
    seed_532308 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 392, 4), random_532307, 'seed')
    # Calling seed(args, kwargs) (line 392)
    seed_call_result_532311 = invoke(stypy.reporting.localization.Localization(__file__, 392, 4), seed_532308, *[int_532309], **kwargs_532310)
    
    
    # Getting the type of 'funcs' (line 394)
    funcs_532312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 25), 'funcs')
    # Testing the type of a for loop iterable (line 394)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 394, 4), funcs_532312)
    # Getting the type of the for loop variable (line 394)
    for_loop_var_532313 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 394, 4), funcs_532312)
    # Assigning a type to the variable 'func' (line 394)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 394, 4), 'func', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 394, 4), for_loop_var_532313))
    # Assigning a type to the variable 'numargs' (line 394)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 394, 4), 'numargs', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 394, 4), for_loop_var_532313))
    # SSA begins for a for statement (line 394)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 395):
    
    # Assigning a Call to a Name (line 395):
    
    # Call to getattr(...): (line 395)
    # Processing the call arguments (line 395)
    # Getting the type of 'sp' (line 395)
    sp_532315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 23), 'sp', False)
    # Getting the type of 'func' (line 395)
    func_532316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 27), 'func', False)
    # Processing the call keyword arguments (line 395)
    kwargs_532317 = {}
    # Getting the type of 'getattr' (line 395)
    getattr_532314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 15), 'getattr', False)
    # Calling getattr(args, kwargs) (line 395)
    getattr_call_result_532318 = invoke(stypy.reporting.localization.Localization(__file__, 395, 15), getattr_532314, *[sp_532315, func_532316], **kwargs_532317)
    
    # Assigning a type to the variable 'func' (line 395)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 395, 8), 'func', getattr_call_result_532318)
    
    # Assigning a ListComp to a Name (line 397):
    
    # Assigning a ListComp to a Name (line 397):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to rand(...): (line 398)
    # Processing the call arguments (line 398)
    # Getting the type of 'numargs' (line 398)
    numargs_532334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 39), 'numargs', False)
    # Processing the call keyword arguments (line 398)
    kwargs_532335 = {}
    # Getting the type of 'np' (line 398)
    np_532331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 24), 'np', False)
    # Obtaining the member 'random' of a type (line 398)
    random_532332 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 24), np_532331, 'random')
    # Obtaining the member 'rand' of a type (line 398)
    rand_532333 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 24), random_532332, 'rand')
    # Calling rand(args, kwargs) (line 398)
    rand_call_result_532336 = invoke(stypy.reporting.localization.Localization(__file__, 398, 24), rand_532333, *[numargs_532334], **kwargs_532335)
    
    comprehension_532337 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 397, 24), rand_call_result_532336)
    # Assigning a type to the variable 'x' (line 397)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 397, 24), 'x', comprehension_532337)
    
    # Obtaining an instance of the builtin type 'tuple' (line 397)
    tuple_532319 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 397, 25), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 397)
    # Adding element type (line 397)
    
    # Call to float(...): (line 397)
    # Processing the call arguments (line 397)
    # Getting the type of 'x' (line 397)
    x_532321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 31), 'x', False)
    # Processing the call keyword arguments (line 397)
    kwargs_532322 = {}
    # Getting the type of 'float' (line 397)
    float_532320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 25), 'float', False)
    # Calling float(args, kwargs) (line 397)
    float_call_result_532323 = invoke(stypy.reporting.localization.Localization(__file__, 397, 25), float_532320, *[x_532321], **kwargs_532322)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 397, 25), tuple_532319, float_call_result_532323)
    # Adding element type (line 397)
    # Getting the type of 'np' (line 397)
    np_532324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 35), 'np')
    # Obtaining the member 'nan' of a type (line 397)
    nan_532325 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 397, 35), np_532324, 'nan')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 397, 25), tuple_532319, nan_532325)
    # Adding element type (line 397)
    # Getting the type of 'np' (line 397)
    np_532326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 43), 'np')
    # Obtaining the member 'inf' of a type (line 397)
    inf_532327 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 397, 43), np_532326, 'inf')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 397, 25), tuple_532319, inf_532327)
    # Adding element type (line 397)
    
    # Getting the type of 'np' (line 397)
    np_532328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 52), 'np')
    # Obtaining the member 'inf' of a type (line 397)
    inf_532329 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 397, 52), np_532328, 'inf')
    # Applying the 'usub' unary operator (line 397)
    result___neg___532330 = python_operator(stypy.reporting.localization.Localization(__file__, 397, 51), 'usub', inf_532329)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 397, 25), tuple_532319, result___neg___532330)
    
    list_532338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 397, 24), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 397, 24), list_532338, tuple_532319)
    # Assigning a type to the variable 'args_choices' (line 397)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 397, 8), 'args_choices', list_532338)
    
    
    # Call to product(...): (line 400)
    # Getting the type of 'args_choices' (line 400)
    args_choices_532341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 39), 'args_choices', False)
    # Processing the call keyword arguments (line 400)
    kwargs_532342 = {}
    # Getting the type of 'itertools' (line 400)
    itertools_532339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 20), 'itertools', False)
    # Obtaining the member 'product' of a type (line 400)
    product_532340 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 400, 20), itertools_532339, 'product')
    # Calling product(args, kwargs) (line 400)
    product_call_result_532343 = invoke(stypy.reporting.localization.Localization(__file__, 400, 20), product_532340, *[args_choices_532341], **kwargs_532342)
    
    # Testing the type of a for loop iterable (line 400)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 400, 8), product_call_result_532343)
    # Getting the type of the for loop variable (line 400)
    for_loop_var_532344 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 400, 8), product_call_result_532343)
    # Assigning a type to the variable 'args' (line 400)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 400, 8), 'args', for_loop_var_532344)
    # SSA begins for a for statement (line 400)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 401):
    
    # Assigning a Call to a Name (line 401):
    
    # Call to func(...): (line 401)
    # Getting the type of 'args' (line 401)
    args_532346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 24), 'args', False)
    # Processing the call keyword arguments (line 401)
    kwargs_532347 = {}
    # Getting the type of 'func' (line 401)
    func_532345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 18), 'func', False)
    # Calling func(args, kwargs) (line 401)
    func_call_result_532348 = invoke(stypy.reporting.localization.Localization(__file__, 401, 18), func_532345, *[args_532346], **kwargs_532347)
    
    # Assigning a type to the variable 'res' (line 401)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 401, 12), 'res', func_call_result_532348)
    
    
    # Call to any(...): (line 403)
    # Processing the call arguments (line 403)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 403, 19, True)
    # Calculating comprehension expression
    # Getting the type of 'args' (line 403)
    args_532355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 40), 'args', False)
    comprehension_532356 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 403, 19), args_532355)
    # Assigning a type to the variable 'x' (line 403)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 403, 19), 'x', comprehension_532356)
    
    # Call to isnan(...): (line 403)
    # Processing the call arguments (line 403)
    # Getting the type of 'x' (line 403)
    x_532352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 28), 'x', False)
    # Processing the call keyword arguments (line 403)
    kwargs_532353 = {}
    # Getting the type of 'np' (line 403)
    np_532350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 19), 'np', False)
    # Obtaining the member 'isnan' of a type (line 403)
    isnan_532351 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 403, 19), np_532350, 'isnan')
    # Calling isnan(args, kwargs) (line 403)
    isnan_call_result_532354 = invoke(stypy.reporting.localization.Localization(__file__, 403, 19), isnan_532351, *[x_532352], **kwargs_532353)
    
    list_532357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 403, 19), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 403, 19), list_532357, isnan_call_result_532354)
    # Processing the call keyword arguments (line 403)
    kwargs_532358 = {}
    # Getting the type of 'any' (line 403)
    any_532349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 15), 'any', False)
    # Calling any(args, kwargs) (line 403)
    any_call_result_532359 = invoke(stypy.reporting.localization.Localization(__file__, 403, 15), any_532349, *[list_532357], **kwargs_532358)
    
    # Testing the type of an if condition (line 403)
    if_condition_532360 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 403, 12), any_call_result_532359)
    # Assigning a type to the variable 'if_condition_532360' (line 403)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 403, 12), 'if_condition_532360', if_condition_532360)
    # SSA begins for if statement (line 403)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to assert_equal(...): (line 405)
    # Processing the call arguments (line 405)
    # Getting the type of 'res' (line 405)
    res_532362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 29), 'res', False)
    # Getting the type of 'np' (line 405)
    np_532363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 34), 'np', False)
    # Obtaining the member 'nan' of a type (line 405)
    nan_532364 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 405, 34), np_532363, 'nan')
    # Processing the call keyword arguments (line 405)
    kwargs_532365 = {}
    # Getting the type of 'assert_equal' (line 405)
    assert_equal_532361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 16), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 405)
    assert_equal_call_result_532366 = invoke(stypy.reporting.localization.Localization(__file__, 405, 16), assert_equal_532361, *[res_532362, nan_532364], **kwargs_532365)
    
    # SSA branch for the else part of an if statement (line 403)
    module_type_store.open_ssa_branch('else')
    pass
    # SSA join for if statement (line 403)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_nonfinite(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_nonfinite' in the type store
    # Getting the type of 'stypy_return_type' (line 357)
    stypy_return_type_532367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_532367)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_nonfinite'
    return stypy_return_type_532367

# Assigning a type to the variable 'test_nonfinite' (line 357)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 357, 0), 'test_nonfinite', test_nonfinite)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
