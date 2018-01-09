
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: from scipy._lib.six import xrange
4: from numpy import (logical_and, asarray, pi, zeros_like,
5:                    piecewise, array, arctan2, tan, zeros, arange, floor)
6: from numpy.core.umath import (sqrt, exp, greater, less, cos, add, sin,
7:                               less_equal, greater_equal)
8: 
9: # From splinemodule.c
10: from .spline import cspline2d, sepfir2d
11: 
12: from scipy.special import comb, gamma
13: 
14: __all__ = ['spline_filter', 'bspline', 'gauss_spline', 'cubic', 'quadratic',
15:            'cspline1d', 'qspline1d', 'cspline1d_eval', 'qspline1d_eval']
16: 
17: 
18: def factorial(n):
19:     return gamma(n + 1)
20: 
21: 
22: def spline_filter(Iin, lmbda=5.0):
23:     '''Smoothing spline (cubic) filtering of a rank-2 array.
24: 
25:     Filter an input data set, `Iin`, using a (cubic) smoothing spline of
26:     fall-off `lmbda`.
27:     '''
28:     intype = Iin.dtype.char
29:     hcol = array([1.0, 4.0, 1.0], 'f') / 6.0
30:     if intype in ['F', 'D']:
31:         Iin = Iin.astype('F')
32:         ckr = cspline2d(Iin.real, lmbda)
33:         cki = cspline2d(Iin.imag, lmbda)
34:         outr = sepfir2d(ckr, hcol, hcol)
35:         outi = sepfir2d(cki, hcol, hcol)
36:         out = (outr + 1j * outi).astype(intype)
37:     elif intype in ['f', 'd']:
38:         ckr = cspline2d(Iin, lmbda)
39:         out = sepfir2d(ckr, hcol, hcol)
40:         out = out.astype(intype)
41:     else:
42:         raise TypeError("Invalid data type for Iin")
43:     return out
44: 
45: 
46: _splinefunc_cache = {}
47: 
48: 
49: def _bspline_piecefunctions(order):
50:     '''Returns the function defined over the left-side pieces for a bspline of
51:     a given order.
52: 
53:     The 0th piece is the first one less than 0.  The last piece is a function
54:     identical to 0 (returned as the constant 0).  (There are order//2 + 2 total
55:     pieces).
56: 
57:     Also returns the condition functions that when evaluated return boolean
58:     arrays for use with `numpy.piecewise`.
59:     '''
60:     try:
61:         return _splinefunc_cache[order]
62:     except KeyError:
63:         pass
64: 
65:     def condfuncgen(num, val1, val2):
66:         if num == 0:
67:             return lambda x: logical_and(less_equal(x, val1),
68:                                          greater_equal(x, val2))
69:         elif num == 2:
70:             return lambda x: less_equal(x, val2)
71:         else:
72:             return lambda x: logical_and(less(x, val1),
73:                                          greater_equal(x, val2))
74: 
75:     last = order // 2 + 2
76:     if order % 2:
77:         startbound = -1.0
78:     else:
79:         startbound = -0.5
80:     condfuncs = [condfuncgen(0, 0, startbound)]
81:     bound = startbound
82:     for num in xrange(1, last - 1):
83:         condfuncs.append(condfuncgen(1, bound, bound - 1))
84:         bound = bound - 1
85:     condfuncs.append(condfuncgen(2, 0, -(order + 1) / 2.0))
86: 
87:     # final value of bound is used in piecefuncgen below
88: 
89:     # the functions to evaluate are taken from the left-hand-side
90:     #  in the general expression derived from the central difference
91:     #  operator (because they involve fewer terms).
92: 
93:     fval = factorial(order)
94: 
95:     def piecefuncgen(num):
96:         Mk = order // 2 - num
97:         if (Mk < 0):
98:             return 0  # final function is 0
99:         coeffs = [(1 - 2 * (k % 2)) * float(comb(order + 1, k, exact=1)) / fval
100:                   for k in xrange(Mk + 1)]
101:         shifts = [-bound - k for k in xrange(Mk + 1)]
102: 
103:         def thefunc(x):
104:             res = 0.0
105:             for k in range(Mk + 1):
106:                 res += coeffs[k] * (x + shifts[k]) ** order
107:             return res
108:         return thefunc
109: 
110:     funclist = [piecefuncgen(k) for k in xrange(last)]
111: 
112:     _splinefunc_cache[order] = (funclist, condfuncs)
113: 
114:     return funclist, condfuncs
115: 
116: 
117: def bspline(x, n):
118:     '''B-spline basis function of order n.
119: 
120:     Notes
121:     -----
122:     Uses numpy.piecewise and automatic function-generator.
123: 
124:     '''
125:     ax = -abs(asarray(x))
126:     # number of pieces on the left-side is (n+1)/2
127:     funclist, condfuncs = _bspline_piecefunctions(n)
128:     condlist = [func(ax) for func in condfuncs]
129:     return piecewise(ax, condlist, funclist)
130: 
131: 
132: def gauss_spline(x, n):
133:     '''Gaussian approximation to B-spline basis function of order n.
134:     '''
135:     signsq = (n + 1) / 12.0
136:     return 1 / sqrt(2 * pi * signsq) * exp(-x ** 2 / 2 / signsq)
137: 
138: 
139: def cubic(x):
140:     '''A cubic B-spline.
141: 
142:     This is a special case of `bspline`, and equivalent to ``bspline(x, 3)``.
143:     '''
144:     ax = abs(asarray(x))
145:     res = zeros_like(ax)
146:     cond1 = less(ax, 1)
147:     if cond1.any():
148:         ax1 = ax[cond1]
149:         res[cond1] = 2.0 / 3 - 1.0 / 2 * ax1 ** 2 * (2 - ax1)
150:     cond2 = ~cond1 & less(ax, 2)
151:     if cond2.any():
152:         ax2 = ax[cond2]
153:         res[cond2] = 1.0 / 6 * (2 - ax2) ** 3
154:     return res
155: 
156: 
157: def quadratic(x):
158:     '''A quadratic B-spline.
159: 
160:     This is a special case of `bspline`, and equivalent to ``bspline(x, 2)``.
161:     '''
162:     ax = abs(asarray(x))
163:     res = zeros_like(ax)
164:     cond1 = less(ax, 0.5)
165:     if cond1.any():
166:         ax1 = ax[cond1]
167:         res[cond1] = 0.75 - ax1 ** 2
168:     cond2 = ~cond1 & less(ax, 1.5)
169:     if cond2.any():
170:         ax2 = ax[cond2]
171:         res[cond2] = (ax2 - 1.5) ** 2 / 2.0
172:     return res
173: 
174: 
175: def _coeff_smooth(lam):
176:     xi = 1 - 96 * lam + 24 * lam * sqrt(3 + 144 * lam)
177:     omeg = arctan2(sqrt(144 * lam - 1), sqrt(xi))
178:     rho = (24 * lam - 1 - sqrt(xi)) / (24 * lam)
179:     rho = rho * sqrt((48 * lam + 24 * lam * sqrt(3 + 144 * lam)) / xi)
180:     return rho, omeg
181: 
182: 
183: def _hc(k, cs, rho, omega):
184:     return (cs / sin(omega) * (rho ** k) * sin(omega * (k + 1)) *
185:             greater(k, -1))
186: 
187: 
188: def _hs(k, cs, rho, omega):
189:     c0 = (cs * cs * (1 + rho * rho) / (1 - rho * rho) /
190:           (1 - 2 * rho * rho * cos(2 * omega) + rho ** 4))
191:     gamma = (1 - rho * rho) / (1 + rho * rho) / tan(omega)
192:     ak = abs(k)
193:     return c0 * rho ** ak * (cos(omega * ak) + gamma * sin(omega * ak))
194: 
195: 
196: def _cubic_smooth_coeff(signal, lamb):
197:     rho, omega = _coeff_smooth(lamb)
198:     cs = 1 - 2 * rho * cos(omega) + rho * rho
199:     K = len(signal)
200:     yp = zeros((K,), signal.dtype.char)
201:     k = arange(K)
202:     yp[0] = (_hc(0, cs, rho, omega) * signal[0] +
203:              add.reduce(_hc(k + 1, cs, rho, omega) * signal))
204: 
205:     yp[1] = (_hc(0, cs, rho, omega) * signal[0] +
206:              _hc(1, cs, rho, omega) * signal[1] +
207:              add.reduce(_hc(k + 2, cs, rho, omega) * signal))
208: 
209:     for n in range(2, K):
210:         yp[n] = (cs * signal[n] + 2 * rho * cos(omega) * yp[n - 1] -
211:                  rho * rho * yp[n - 2])
212: 
213:     y = zeros((K,), signal.dtype.char)
214: 
215:     y[K - 1] = add.reduce((_hs(k, cs, rho, omega) +
216:                            _hs(k + 1, cs, rho, omega)) * signal[::-1])
217:     y[K - 2] = add.reduce((_hs(k - 1, cs, rho, omega) +
218:                            _hs(k + 2, cs, rho, omega)) * signal[::-1])
219: 
220:     for n in range(K - 3, -1, -1):
221:         y[n] = (cs * yp[n] + 2 * rho * cos(omega) * y[n + 1] -
222:                 rho * rho * y[n + 2])
223: 
224:     return y
225: 
226: 
227: def _cubic_coeff(signal):
228:     zi = -2 + sqrt(3)
229:     K = len(signal)
230:     yplus = zeros((K,), signal.dtype.char)
231:     powers = zi ** arange(K)
232:     yplus[0] = signal[0] + zi * add.reduce(powers * signal)
233:     for k in range(1, K):
234:         yplus[k] = signal[k] + zi * yplus[k - 1]
235:     output = zeros((K,), signal.dtype)
236:     output[K - 1] = zi / (zi - 1) * yplus[K - 1]
237:     for k in range(K - 2, -1, -1):
238:         output[k] = zi * (output[k + 1] - yplus[k])
239:     return output * 6.0
240: 
241: 
242: def _quadratic_coeff(signal):
243:     zi = -3 + 2 * sqrt(2.0)
244:     K = len(signal)
245:     yplus = zeros((K,), signal.dtype.char)
246:     powers = zi ** arange(K)
247:     yplus[0] = signal[0] + zi * add.reduce(powers * signal)
248:     for k in range(1, K):
249:         yplus[k] = signal[k] + zi * yplus[k - 1]
250:     output = zeros((K,), signal.dtype.char)
251:     output[K - 1] = zi / (zi - 1) * yplus[K - 1]
252:     for k in range(K - 2, -1, -1):
253:         output[k] = zi * (output[k + 1] - yplus[k])
254:     return output * 8.0
255: 
256: 
257: def cspline1d(signal, lamb=0.0):
258:     '''
259:     Compute cubic spline coefficients for rank-1 array.
260: 
261:     Find the cubic spline coefficients for a 1-D signal assuming
262:     mirror-symmetric boundary conditions.   To obtain the signal back from the
263:     spline representation mirror-symmetric-convolve these coefficients with a
264:     length 3 FIR window [1.0, 4.0, 1.0]/ 6.0 .
265: 
266:     Parameters
267:     ----------
268:     signal : ndarray
269:         A rank-1 array representing samples of a signal.
270:     lamb : float, optional
271:         Smoothing coefficient, default is 0.0.
272: 
273:     Returns
274:     -------
275:     c : ndarray
276:         Cubic spline coefficients.
277: 
278:     '''
279:     if lamb != 0.0:
280:         return _cubic_smooth_coeff(signal, lamb)
281:     else:
282:         return _cubic_coeff(signal)
283: 
284: 
285: def qspline1d(signal, lamb=0.0):
286:     '''Compute quadratic spline coefficients for rank-1 array.
287: 
288:     Find the quadratic spline coefficients for a 1-D signal assuming
289:     mirror-symmetric boundary conditions.   To obtain the signal back from the
290:     spline representation mirror-symmetric-convolve these coefficients with a
291:     length 3 FIR window [1.0, 6.0, 1.0]/ 8.0 .
292: 
293:     Parameters
294:     ----------
295:     signal : ndarray
296:         A rank-1 array representing samples of a signal.
297:     lamb : float, optional
298:         Smoothing coefficient (must be zero for now).
299: 
300:     Returns
301:     -------
302:     c : ndarray
303:         Cubic spline coefficients.
304: 
305:     '''
306:     if lamb != 0.0:
307:         raise ValueError("Smoothing quadratic splines not supported yet.")
308:     else:
309:         return _quadratic_coeff(signal)
310: 
311: 
312: def cspline1d_eval(cj, newx, dx=1.0, x0=0):
313:     '''Evaluate a spline at the new set of points.
314: 
315:     `dx` is the old sample-spacing while `x0` was the old origin.  In
316:     other-words the old-sample points (knot-points) for which the `cj`
317:     represent spline coefficients were at equally-spaced points of:
318: 
319:       oldx = x0 + j*dx  j=0...N-1, with N=len(cj)
320: 
321:     Edges are handled using mirror-symmetric boundary conditions.
322: 
323:     '''
324:     newx = (asarray(newx) - x0) / float(dx)
325:     res = zeros_like(newx, dtype=cj.dtype)
326:     if res.size == 0:
327:         return res
328:     N = len(cj)
329:     cond1 = newx < 0
330:     cond2 = newx > (N - 1)
331:     cond3 = ~(cond1 | cond2)
332:     # handle general mirror-symmetry
333:     res[cond1] = cspline1d_eval(cj, -newx[cond1])
334:     res[cond2] = cspline1d_eval(cj, 2 * (N - 1) - newx[cond2])
335:     newx = newx[cond3]
336:     if newx.size == 0:
337:         return res
338:     result = zeros_like(newx, dtype=cj.dtype)
339:     jlower = floor(newx - 2).astype(int) + 1
340:     for i in range(4):
341:         thisj = jlower + i
342:         indj = thisj.clip(0, N - 1)  # handle edge cases
343:         result += cj[indj] * cubic(newx - thisj)
344:     res[cond3] = result
345:     return res
346: 
347: 
348: def qspline1d_eval(cj, newx, dx=1.0, x0=0):
349:     '''Evaluate a quadratic spline at the new set of points.
350: 
351:     `dx` is the old sample-spacing while `x0` was the old origin.  In
352:     other-words the old-sample points (knot-points) for which the `cj`
353:     represent spline coefficients were at equally-spaced points of::
354: 
355:       oldx = x0 + j*dx  j=0...N-1, with N=len(cj)
356: 
357:     Edges are handled using mirror-symmetric boundary conditions.
358: 
359:     '''
360:     newx = (asarray(newx) - x0) / dx
361:     res = zeros_like(newx)
362:     if res.size == 0:
363:         return res
364:     N = len(cj)
365:     cond1 = newx < 0
366:     cond2 = newx > (N - 1)
367:     cond3 = ~(cond1 | cond2)
368:     # handle general mirror-symmetry
369:     res[cond1] = qspline1d_eval(cj, -newx[cond1])
370:     res[cond2] = qspline1d_eval(cj, 2 * (N - 1) - newx[cond2])
371:     newx = newx[cond3]
372:     if newx.size == 0:
373:         return res
374:     result = zeros_like(newx)
375:     jlower = floor(newx - 1.5).astype(int) + 1
376:     for i in range(3):
377:         thisj = jlower + i
378:         indj = thisj.clip(0, N - 1)  # handle edge cases
379:         result += cj[indj] * quadratic(newx - thisj)
380:     res[cond3] = result
381:     return res
382: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'from scipy._lib.six import xrange' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/')
import_255808 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'scipy._lib.six')

if (type(import_255808) is not StypyTypeError):

    if (import_255808 != 'pyd_module'):
        __import__(import_255808)
        sys_modules_255809 = sys.modules[import_255808]
        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'scipy._lib.six', sys_modules_255809.module_type_store, module_type_store, ['xrange'])
        nest_module(stypy.reporting.localization.Localization(__file__, 3, 0), __file__, sys_modules_255809, sys_modules_255809.module_type_store, module_type_store)
    else:
        from scipy._lib.six import xrange

        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'scipy._lib.six', None, module_type_store, ['xrange'], [xrange])

else:
    # Assigning a type to the variable 'scipy._lib.six' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'scipy._lib.six', import_255808)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from numpy import logical_and, asarray, pi, zeros_like, piecewise, array, arctan2, tan, zeros, arange, floor' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/')
import_255810 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy')

if (type(import_255810) is not StypyTypeError):

    if (import_255810 != 'pyd_module'):
        __import__(import_255810)
        sys_modules_255811 = sys.modules[import_255810]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy', sys_modules_255811.module_type_store, module_type_store, ['logical_and', 'asarray', 'pi', 'zeros_like', 'piecewise', 'array', 'arctan2', 'tan', 'zeros', 'arange', 'floor'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_255811, sys_modules_255811.module_type_store, module_type_store)
    else:
        from numpy import logical_and, asarray, pi, zeros_like, piecewise, array, arctan2, tan, zeros, arange, floor

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy', None, module_type_store, ['logical_and', 'asarray', 'pi', 'zeros_like', 'piecewise', 'array', 'arctan2', 'tan', 'zeros', 'arange', 'floor'], [logical_and, asarray, pi, zeros_like, piecewise, array, arctan2, tan, zeros, arange, floor])

else:
    # Assigning a type to the variable 'numpy' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy', import_255810)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from numpy.core.umath import sqrt, exp, greater, less, cos, add, sin, less_equal, greater_equal' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/')
import_255812 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.core.umath')

if (type(import_255812) is not StypyTypeError):

    if (import_255812 != 'pyd_module'):
        __import__(import_255812)
        sys_modules_255813 = sys.modules[import_255812]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.core.umath', sys_modules_255813.module_type_store, module_type_store, ['sqrt', 'exp', 'greater', 'less', 'cos', 'add', 'sin', 'less_equal', 'greater_equal'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_255813, sys_modules_255813.module_type_store, module_type_store)
    else:
        from numpy.core.umath import sqrt, exp, greater, less, cos, add, sin, less_equal, greater_equal

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.core.umath', None, module_type_store, ['sqrt', 'exp', 'greater', 'less', 'cos', 'add', 'sin', 'less_equal', 'greater_equal'], [sqrt, exp, greater, less, cos, add, sin, less_equal, greater_equal])

else:
    # Assigning a type to the variable 'numpy.core.umath' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.core.umath', import_255812)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from scipy.signal.spline import cspline2d, sepfir2d' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/')
import_255814 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.signal.spline')

if (type(import_255814) is not StypyTypeError):

    if (import_255814 != 'pyd_module'):
        __import__(import_255814)
        sys_modules_255815 = sys.modules[import_255814]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.signal.spline', sys_modules_255815.module_type_store, module_type_store, ['cspline2d', 'sepfir2d'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_255815, sys_modules_255815.module_type_store, module_type_store)
    else:
        from scipy.signal.spline import cspline2d, sepfir2d

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.signal.spline', None, module_type_store, ['cspline2d', 'sepfir2d'], [cspline2d, sepfir2d])

else:
    # Assigning a type to the variable 'scipy.signal.spline' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.signal.spline', import_255814)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from scipy.special import comb, gamma' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/')
import_255816 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.special')

if (type(import_255816) is not StypyTypeError):

    if (import_255816 != 'pyd_module'):
        __import__(import_255816)
        sys_modules_255817 = sys.modules[import_255816]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.special', sys_modules_255817.module_type_store, module_type_store, ['comb', 'gamma'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_255817, sys_modules_255817.module_type_store, module_type_store)
    else:
        from scipy.special import comb, gamma

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.special', None, module_type_store, ['comb', 'gamma'], [comb, gamma])

else:
    # Assigning a type to the variable 'scipy.special' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.special', import_255816)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/')


# Assigning a List to a Name (line 14):

# Assigning a List to a Name (line 14):
__all__ = ['spline_filter', 'bspline', 'gauss_spline', 'cubic', 'quadratic', 'cspline1d', 'qspline1d', 'cspline1d_eval', 'qspline1d_eval']
module_type_store.set_exportable_members(['spline_filter', 'bspline', 'gauss_spline', 'cubic', 'quadratic', 'cspline1d', 'qspline1d', 'cspline1d_eval', 'qspline1d_eval'])

# Obtaining an instance of the builtin type 'list' (line 14)
list_255818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 14)
# Adding element type (line 14)
str_255819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 11), 'str', 'spline_filter')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 10), list_255818, str_255819)
# Adding element type (line 14)
str_255820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 28), 'str', 'bspline')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 10), list_255818, str_255820)
# Adding element type (line 14)
str_255821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 39), 'str', 'gauss_spline')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 10), list_255818, str_255821)
# Adding element type (line 14)
str_255822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 55), 'str', 'cubic')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 10), list_255818, str_255822)
# Adding element type (line 14)
str_255823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 64), 'str', 'quadratic')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 10), list_255818, str_255823)
# Adding element type (line 14)
str_255824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 11), 'str', 'cspline1d')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 10), list_255818, str_255824)
# Adding element type (line 14)
str_255825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 24), 'str', 'qspline1d')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 10), list_255818, str_255825)
# Adding element type (line 14)
str_255826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 37), 'str', 'cspline1d_eval')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 10), list_255818, str_255826)
# Adding element type (line 14)
str_255827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 55), 'str', 'qspline1d_eval')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 10), list_255818, str_255827)

# Assigning a type to the variable '__all__' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), '__all__', list_255818)

@norecursion
def factorial(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'factorial'
    module_type_store = module_type_store.open_function_context('factorial', 18, 0, False)
    
    # Passed parameters checking function
    factorial.stypy_localization = localization
    factorial.stypy_type_of_self = None
    factorial.stypy_type_store = module_type_store
    factorial.stypy_function_name = 'factorial'
    factorial.stypy_param_names_list = ['n']
    factorial.stypy_varargs_param_name = None
    factorial.stypy_kwargs_param_name = None
    factorial.stypy_call_defaults = defaults
    factorial.stypy_call_varargs = varargs
    factorial.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'factorial', ['n'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'factorial', localization, ['n'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'factorial(...)' code ##################

    
    # Call to gamma(...): (line 19)
    # Processing the call arguments (line 19)
    # Getting the type of 'n' (line 19)
    n_255829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 17), 'n', False)
    int_255830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 21), 'int')
    # Applying the binary operator '+' (line 19)
    result_add_255831 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 17), '+', n_255829, int_255830)
    
    # Processing the call keyword arguments (line 19)
    kwargs_255832 = {}
    # Getting the type of 'gamma' (line 19)
    gamma_255828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 11), 'gamma', False)
    # Calling gamma(args, kwargs) (line 19)
    gamma_call_result_255833 = invoke(stypy.reporting.localization.Localization(__file__, 19, 11), gamma_255828, *[result_add_255831], **kwargs_255832)
    
    # Assigning a type to the variable 'stypy_return_type' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'stypy_return_type', gamma_call_result_255833)
    
    # ################# End of 'factorial(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'factorial' in the type store
    # Getting the type of 'stypy_return_type' (line 18)
    stypy_return_type_255834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_255834)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'factorial'
    return stypy_return_type_255834

# Assigning a type to the variable 'factorial' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'factorial', factorial)

@norecursion
def spline_filter(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    float_255835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 29), 'float')
    defaults = [float_255835]
    # Create a new context for function 'spline_filter'
    module_type_store = module_type_store.open_function_context('spline_filter', 22, 0, False)
    
    # Passed parameters checking function
    spline_filter.stypy_localization = localization
    spline_filter.stypy_type_of_self = None
    spline_filter.stypy_type_store = module_type_store
    spline_filter.stypy_function_name = 'spline_filter'
    spline_filter.stypy_param_names_list = ['Iin', 'lmbda']
    spline_filter.stypy_varargs_param_name = None
    spline_filter.stypy_kwargs_param_name = None
    spline_filter.stypy_call_defaults = defaults
    spline_filter.stypy_call_varargs = varargs
    spline_filter.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'spline_filter', ['Iin', 'lmbda'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'spline_filter', localization, ['Iin', 'lmbda'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'spline_filter(...)' code ##################

    str_255836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, (-1)), 'str', 'Smoothing spline (cubic) filtering of a rank-2 array.\n\n    Filter an input data set, `Iin`, using a (cubic) smoothing spline of\n    fall-off `lmbda`.\n    ')
    
    # Assigning a Attribute to a Name (line 28):
    
    # Assigning a Attribute to a Name (line 28):
    # Getting the type of 'Iin' (line 28)
    Iin_255837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 13), 'Iin')
    # Obtaining the member 'dtype' of a type (line 28)
    dtype_255838 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 13), Iin_255837, 'dtype')
    # Obtaining the member 'char' of a type (line 28)
    char_255839 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 13), dtype_255838, 'char')
    # Assigning a type to the variable 'intype' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'intype', char_255839)
    
    # Assigning a BinOp to a Name (line 29):
    
    # Assigning a BinOp to a Name (line 29):
    
    # Call to array(...): (line 29)
    # Processing the call arguments (line 29)
    
    # Obtaining an instance of the builtin type 'list' (line 29)
    list_255841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 29)
    # Adding element type (line 29)
    float_255842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 18), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 17), list_255841, float_255842)
    # Adding element type (line 29)
    float_255843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 23), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 17), list_255841, float_255843)
    # Adding element type (line 29)
    float_255844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 28), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 17), list_255841, float_255844)
    
    str_255845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 34), 'str', 'f')
    # Processing the call keyword arguments (line 29)
    kwargs_255846 = {}
    # Getting the type of 'array' (line 29)
    array_255840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 11), 'array', False)
    # Calling array(args, kwargs) (line 29)
    array_call_result_255847 = invoke(stypy.reporting.localization.Localization(__file__, 29, 11), array_255840, *[list_255841, str_255845], **kwargs_255846)
    
    float_255848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 41), 'float')
    # Applying the binary operator 'div' (line 29)
    result_div_255849 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 11), 'div', array_call_result_255847, float_255848)
    
    # Assigning a type to the variable 'hcol' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'hcol', result_div_255849)
    
    
    # Getting the type of 'intype' (line 30)
    intype_255850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 7), 'intype')
    
    # Obtaining an instance of the builtin type 'list' (line 30)
    list_255851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 30)
    # Adding element type (line 30)
    str_255852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 18), 'str', 'F')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 17), list_255851, str_255852)
    # Adding element type (line 30)
    str_255853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 23), 'str', 'D')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 17), list_255851, str_255853)
    
    # Applying the binary operator 'in' (line 30)
    result_contains_255854 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 7), 'in', intype_255850, list_255851)
    
    # Testing the type of an if condition (line 30)
    if_condition_255855 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 30, 4), result_contains_255854)
    # Assigning a type to the variable 'if_condition_255855' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'if_condition_255855', if_condition_255855)
    # SSA begins for if statement (line 30)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 31):
    
    # Assigning a Call to a Name (line 31):
    
    # Call to astype(...): (line 31)
    # Processing the call arguments (line 31)
    str_255858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 25), 'str', 'F')
    # Processing the call keyword arguments (line 31)
    kwargs_255859 = {}
    # Getting the type of 'Iin' (line 31)
    Iin_255856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 14), 'Iin', False)
    # Obtaining the member 'astype' of a type (line 31)
    astype_255857 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 14), Iin_255856, 'astype')
    # Calling astype(args, kwargs) (line 31)
    astype_call_result_255860 = invoke(stypy.reporting.localization.Localization(__file__, 31, 14), astype_255857, *[str_255858], **kwargs_255859)
    
    # Assigning a type to the variable 'Iin' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'Iin', astype_call_result_255860)
    
    # Assigning a Call to a Name (line 32):
    
    # Assigning a Call to a Name (line 32):
    
    # Call to cspline2d(...): (line 32)
    # Processing the call arguments (line 32)
    # Getting the type of 'Iin' (line 32)
    Iin_255862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 24), 'Iin', False)
    # Obtaining the member 'real' of a type (line 32)
    real_255863 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 24), Iin_255862, 'real')
    # Getting the type of 'lmbda' (line 32)
    lmbda_255864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 34), 'lmbda', False)
    # Processing the call keyword arguments (line 32)
    kwargs_255865 = {}
    # Getting the type of 'cspline2d' (line 32)
    cspline2d_255861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 14), 'cspline2d', False)
    # Calling cspline2d(args, kwargs) (line 32)
    cspline2d_call_result_255866 = invoke(stypy.reporting.localization.Localization(__file__, 32, 14), cspline2d_255861, *[real_255863, lmbda_255864], **kwargs_255865)
    
    # Assigning a type to the variable 'ckr' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'ckr', cspline2d_call_result_255866)
    
    # Assigning a Call to a Name (line 33):
    
    # Assigning a Call to a Name (line 33):
    
    # Call to cspline2d(...): (line 33)
    # Processing the call arguments (line 33)
    # Getting the type of 'Iin' (line 33)
    Iin_255868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 24), 'Iin', False)
    # Obtaining the member 'imag' of a type (line 33)
    imag_255869 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 24), Iin_255868, 'imag')
    # Getting the type of 'lmbda' (line 33)
    lmbda_255870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 34), 'lmbda', False)
    # Processing the call keyword arguments (line 33)
    kwargs_255871 = {}
    # Getting the type of 'cspline2d' (line 33)
    cspline2d_255867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 14), 'cspline2d', False)
    # Calling cspline2d(args, kwargs) (line 33)
    cspline2d_call_result_255872 = invoke(stypy.reporting.localization.Localization(__file__, 33, 14), cspline2d_255867, *[imag_255869, lmbda_255870], **kwargs_255871)
    
    # Assigning a type to the variable 'cki' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'cki', cspline2d_call_result_255872)
    
    # Assigning a Call to a Name (line 34):
    
    # Assigning a Call to a Name (line 34):
    
    # Call to sepfir2d(...): (line 34)
    # Processing the call arguments (line 34)
    # Getting the type of 'ckr' (line 34)
    ckr_255874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 24), 'ckr', False)
    # Getting the type of 'hcol' (line 34)
    hcol_255875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 29), 'hcol', False)
    # Getting the type of 'hcol' (line 34)
    hcol_255876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 35), 'hcol', False)
    # Processing the call keyword arguments (line 34)
    kwargs_255877 = {}
    # Getting the type of 'sepfir2d' (line 34)
    sepfir2d_255873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 15), 'sepfir2d', False)
    # Calling sepfir2d(args, kwargs) (line 34)
    sepfir2d_call_result_255878 = invoke(stypy.reporting.localization.Localization(__file__, 34, 15), sepfir2d_255873, *[ckr_255874, hcol_255875, hcol_255876], **kwargs_255877)
    
    # Assigning a type to the variable 'outr' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'outr', sepfir2d_call_result_255878)
    
    # Assigning a Call to a Name (line 35):
    
    # Assigning a Call to a Name (line 35):
    
    # Call to sepfir2d(...): (line 35)
    # Processing the call arguments (line 35)
    # Getting the type of 'cki' (line 35)
    cki_255880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 24), 'cki', False)
    # Getting the type of 'hcol' (line 35)
    hcol_255881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 29), 'hcol', False)
    # Getting the type of 'hcol' (line 35)
    hcol_255882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 35), 'hcol', False)
    # Processing the call keyword arguments (line 35)
    kwargs_255883 = {}
    # Getting the type of 'sepfir2d' (line 35)
    sepfir2d_255879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 15), 'sepfir2d', False)
    # Calling sepfir2d(args, kwargs) (line 35)
    sepfir2d_call_result_255884 = invoke(stypy.reporting.localization.Localization(__file__, 35, 15), sepfir2d_255879, *[cki_255880, hcol_255881, hcol_255882], **kwargs_255883)
    
    # Assigning a type to the variable 'outi' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'outi', sepfir2d_call_result_255884)
    
    # Assigning a Call to a Name (line 36):
    
    # Assigning a Call to a Name (line 36):
    
    # Call to astype(...): (line 36)
    # Processing the call arguments (line 36)
    # Getting the type of 'intype' (line 36)
    intype_255891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 40), 'intype', False)
    # Processing the call keyword arguments (line 36)
    kwargs_255892 = {}
    # Getting the type of 'outr' (line 36)
    outr_255885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 15), 'outr', False)
    complex_255886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 22), 'complex')
    # Getting the type of 'outi' (line 36)
    outi_255887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 27), 'outi', False)
    # Applying the binary operator '*' (line 36)
    result_mul_255888 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 22), '*', complex_255886, outi_255887)
    
    # Applying the binary operator '+' (line 36)
    result_add_255889 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 15), '+', outr_255885, result_mul_255888)
    
    # Obtaining the member 'astype' of a type (line 36)
    astype_255890 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 15), result_add_255889, 'astype')
    # Calling astype(args, kwargs) (line 36)
    astype_call_result_255893 = invoke(stypy.reporting.localization.Localization(__file__, 36, 15), astype_255890, *[intype_255891], **kwargs_255892)
    
    # Assigning a type to the variable 'out' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'out', astype_call_result_255893)
    # SSA branch for the else part of an if statement (line 30)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'intype' (line 37)
    intype_255894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 9), 'intype')
    
    # Obtaining an instance of the builtin type 'list' (line 37)
    list_255895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 37)
    # Adding element type (line 37)
    str_255896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 20), 'str', 'f')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 19), list_255895, str_255896)
    # Adding element type (line 37)
    str_255897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 25), 'str', 'd')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 19), list_255895, str_255897)
    
    # Applying the binary operator 'in' (line 37)
    result_contains_255898 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 9), 'in', intype_255894, list_255895)
    
    # Testing the type of an if condition (line 37)
    if_condition_255899 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 37, 9), result_contains_255898)
    # Assigning a type to the variable 'if_condition_255899' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 9), 'if_condition_255899', if_condition_255899)
    # SSA begins for if statement (line 37)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 38):
    
    # Assigning a Call to a Name (line 38):
    
    # Call to cspline2d(...): (line 38)
    # Processing the call arguments (line 38)
    # Getting the type of 'Iin' (line 38)
    Iin_255901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 24), 'Iin', False)
    # Getting the type of 'lmbda' (line 38)
    lmbda_255902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 29), 'lmbda', False)
    # Processing the call keyword arguments (line 38)
    kwargs_255903 = {}
    # Getting the type of 'cspline2d' (line 38)
    cspline2d_255900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 14), 'cspline2d', False)
    # Calling cspline2d(args, kwargs) (line 38)
    cspline2d_call_result_255904 = invoke(stypy.reporting.localization.Localization(__file__, 38, 14), cspline2d_255900, *[Iin_255901, lmbda_255902], **kwargs_255903)
    
    # Assigning a type to the variable 'ckr' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'ckr', cspline2d_call_result_255904)
    
    # Assigning a Call to a Name (line 39):
    
    # Assigning a Call to a Name (line 39):
    
    # Call to sepfir2d(...): (line 39)
    # Processing the call arguments (line 39)
    # Getting the type of 'ckr' (line 39)
    ckr_255906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 23), 'ckr', False)
    # Getting the type of 'hcol' (line 39)
    hcol_255907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 28), 'hcol', False)
    # Getting the type of 'hcol' (line 39)
    hcol_255908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 34), 'hcol', False)
    # Processing the call keyword arguments (line 39)
    kwargs_255909 = {}
    # Getting the type of 'sepfir2d' (line 39)
    sepfir2d_255905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 14), 'sepfir2d', False)
    # Calling sepfir2d(args, kwargs) (line 39)
    sepfir2d_call_result_255910 = invoke(stypy.reporting.localization.Localization(__file__, 39, 14), sepfir2d_255905, *[ckr_255906, hcol_255907, hcol_255908], **kwargs_255909)
    
    # Assigning a type to the variable 'out' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'out', sepfir2d_call_result_255910)
    
    # Assigning a Call to a Name (line 40):
    
    # Assigning a Call to a Name (line 40):
    
    # Call to astype(...): (line 40)
    # Processing the call arguments (line 40)
    # Getting the type of 'intype' (line 40)
    intype_255913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 25), 'intype', False)
    # Processing the call keyword arguments (line 40)
    kwargs_255914 = {}
    # Getting the type of 'out' (line 40)
    out_255911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 14), 'out', False)
    # Obtaining the member 'astype' of a type (line 40)
    astype_255912 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 14), out_255911, 'astype')
    # Calling astype(args, kwargs) (line 40)
    astype_call_result_255915 = invoke(stypy.reporting.localization.Localization(__file__, 40, 14), astype_255912, *[intype_255913], **kwargs_255914)
    
    # Assigning a type to the variable 'out' (line 40)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'out', astype_call_result_255915)
    # SSA branch for the else part of an if statement (line 37)
    module_type_store.open_ssa_branch('else')
    
    # Call to TypeError(...): (line 42)
    # Processing the call arguments (line 42)
    str_255917 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 24), 'str', 'Invalid data type for Iin')
    # Processing the call keyword arguments (line 42)
    kwargs_255918 = {}
    # Getting the type of 'TypeError' (line 42)
    TypeError_255916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 14), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 42)
    TypeError_call_result_255919 = invoke(stypy.reporting.localization.Localization(__file__, 42, 14), TypeError_255916, *[str_255917], **kwargs_255918)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 42, 8), TypeError_call_result_255919, 'raise parameter', BaseException)
    # SSA join for if statement (line 37)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 30)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'out' (line 43)
    out_255920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 11), 'out')
    # Assigning a type to the variable 'stypy_return_type' (line 43)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'stypy_return_type', out_255920)
    
    # ################# End of 'spline_filter(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'spline_filter' in the type store
    # Getting the type of 'stypy_return_type' (line 22)
    stypy_return_type_255921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_255921)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'spline_filter'
    return stypy_return_type_255921

# Assigning a type to the variable 'spline_filter' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'spline_filter', spline_filter)

# Assigning a Dict to a Name (line 46):

# Assigning a Dict to a Name (line 46):

# Obtaining an instance of the builtin type 'dict' (line 46)
dict_255922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 20), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 46)

# Assigning a type to the variable '_splinefunc_cache' (line 46)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 0), '_splinefunc_cache', dict_255922)

@norecursion
def _bspline_piecefunctions(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_bspline_piecefunctions'
    module_type_store = module_type_store.open_function_context('_bspline_piecefunctions', 49, 0, False)
    
    # Passed parameters checking function
    _bspline_piecefunctions.stypy_localization = localization
    _bspline_piecefunctions.stypy_type_of_self = None
    _bspline_piecefunctions.stypy_type_store = module_type_store
    _bspline_piecefunctions.stypy_function_name = '_bspline_piecefunctions'
    _bspline_piecefunctions.stypy_param_names_list = ['order']
    _bspline_piecefunctions.stypy_varargs_param_name = None
    _bspline_piecefunctions.stypy_kwargs_param_name = None
    _bspline_piecefunctions.stypy_call_defaults = defaults
    _bspline_piecefunctions.stypy_call_varargs = varargs
    _bspline_piecefunctions.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_bspline_piecefunctions', ['order'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_bspline_piecefunctions', localization, ['order'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_bspline_piecefunctions(...)' code ##################

    str_255923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, (-1)), 'str', 'Returns the function defined over the left-side pieces for a bspline of\n    a given order.\n\n    The 0th piece is the first one less than 0.  The last piece is a function\n    identical to 0 (returned as the constant 0).  (There are order//2 + 2 total\n    pieces).\n\n    Also returns the condition functions that when evaluated return boolean\n    arrays for use with `numpy.piecewise`.\n    ')
    
    
    # SSA begins for try-except statement (line 60)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Obtaining the type of the subscript
    # Getting the type of 'order' (line 61)
    order_255924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 33), 'order')
    # Getting the type of '_splinefunc_cache' (line 61)
    _splinefunc_cache_255925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 15), '_splinefunc_cache')
    # Obtaining the member '__getitem__' of a type (line 61)
    getitem___255926 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 15), _splinefunc_cache_255925, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 61)
    subscript_call_result_255927 = invoke(stypy.reporting.localization.Localization(__file__, 61, 15), getitem___255926, order_255924)
    
    # Assigning a type to the variable 'stypy_return_type' (line 61)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'stypy_return_type', subscript_call_result_255927)
    # SSA branch for the except part of a try statement (line 60)
    # SSA branch for the except 'KeyError' branch of a try statement (line 60)
    module_type_store.open_ssa_branch('except')
    pass
    # SSA join for try-except statement (line 60)
    module_type_store = module_type_store.join_ssa_context()
    

    @norecursion
    def condfuncgen(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'condfuncgen'
        module_type_store = module_type_store.open_function_context('condfuncgen', 65, 4, False)
        
        # Passed parameters checking function
        condfuncgen.stypy_localization = localization
        condfuncgen.stypy_type_of_self = None
        condfuncgen.stypy_type_store = module_type_store
        condfuncgen.stypy_function_name = 'condfuncgen'
        condfuncgen.stypy_param_names_list = ['num', 'val1', 'val2']
        condfuncgen.stypy_varargs_param_name = None
        condfuncgen.stypy_kwargs_param_name = None
        condfuncgen.stypy_call_defaults = defaults
        condfuncgen.stypy_call_varargs = varargs
        condfuncgen.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'condfuncgen', ['num', 'val1', 'val2'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'condfuncgen', localization, ['num', 'val1', 'val2'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'condfuncgen(...)' code ##################

        
        
        # Getting the type of 'num' (line 66)
        num_255928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 11), 'num')
        int_255929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 18), 'int')
        # Applying the binary operator '==' (line 66)
        result_eq_255930 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 11), '==', num_255928, int_255929)
        
        # Testing the type of an if condition (line 66)
        if_condition_255931 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 66, 8), result_eq_255930)
        # Assigning a type to the variable 'if_condition_255931' (line 66)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'if_condition_255931', if_condition_255931)
        # SSA begins for if statement (line 66)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

        @norecursion
        def _stypy_temp_lambda_167(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_167'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_167', 67, 19, True)
            # Passed parameters checking function
            _stypy_temp_lambda_167.stypy_localization = localization
            _stypy_temp_lambda_167.stypy_type_of_self = None
            _stypy_temp_lambda_167.stypy_type_store = module_type_store
            _stypy_temp_lambda_167.stypy_function_name = '_stypy_temp_lambda_167'
            _stypy_temp_lambda_167.stypy_param_names_list = ['x']
            _stypy_temp_lambda_167.stypy_varargs_param_name = None
            _stypy_temp_lambda_167.stypy_kwargs_param_name = None
            _stypy_temp_lambda_167.stypy_call_defaults = defaults
            _stypy_temp_lambda_167.stypy_call_varargs = varargs
            _stypy_temp_lambda_167.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_167', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_167', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Call to logical_and(...): (line 67)
            # Processing the call arguments (line 67)
            
            # Call to less_equal(...): (line 67)
            # Processing the call arguments (line 67)
            # Getting the type of 'x' (line 67)
            x_255934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 52), 'x', False)
            # Getting the type of 'val1' (line 67)
            val1_255935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 55), 'val1', False)
            # Processing the call keyword arguments (line 67)
            kwargs_255936 = {}
            # Getting the type of 'less_equal' (line 67)
            less_equal_255933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 41), 'less_equal', False)
            # Calling less_equal(args, kwargs) (line 67)
            less_equal_call_result_255937 = invoke(stypy.reporting.localization.Localization(__file__, 67, 41), less_equal_255933, *[x_255934, val1_255935], **kwargs_255936)
            
            
            # Call to greater_equal(...): (line 68)
            # Processing the call arguments (line 68)
            # Getting the type of 'x' (line 68)
            x_255939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 55), 'x', False)
            # Getting the type of 'val2' (line 68)
            val2_255940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 58), 'val2', False)
            # Processing the call keyword arguments (line 68)
            kwargs_255941 = {}
            # Getting the type of 'greater_equal' (line 68)
            greater_equal_255938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 41), 'greater_equal', False)
            # Calling greater_equal(args, kwargs) (line 68)
            greater_equal_call_result_255942 = invoke(stypy.reporting.localization.Localization(__file__, 68, 41), greater_equal_255938, *[x_255939, val2_255940], **kwargs_255941)
            
            # Processing the call keyword arguments (line 67)
            kwargs_255943 = {}
            # Getting the type of 'logical_and' (line 67)
            logical_and_255932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 29), 'logical_and', False)
            # Calling logical_and(args, kwargs) (line 67)
            logical_and_call_result_255944 = invoke(stypy.reporting.localization.Localization(__file__, 67, 29), logical_and_255932, *[less_equal_call_result_255937, greater_equal_call_result_255942], **kwargs_255943)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 67)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 19), 'stypy_return_type', logical_and_call_result_255944)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_167' in the type store
            # Getting the type of 'stypy_return_type' (line 67)
            stypy_return_type_255945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 19), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_255945)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_167'
            return stypy_return_type_255945

        # Assigning a type to the variable '_stypy_temp_lambda_167' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 19), '_stypy_temp_lambda_167', _stypy_temp_lambda_167)
        # Getting the type of '_stypy_temp_lambda_167' (line 67)
        _stypy_temp_lambda_167_255946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 19), '_stypy_temp_lambda_167')
        # Assigning a type to the variable 'stypy_return_type' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 12), 'stypy_return_type', _stypy_temp_lambda_167_255946)
        # SSA branch for the else part of an if statement (line 66)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'num' (line 69)
        num_255947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 13), 'num')
        int_255948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 20), 'int')
        # Applying the binary operator '==' (line 69)
        result_eq_255949 = python_operator(stypy.reporting.localization.Localization(__file__, 69, 13), '==', num_255947, int_255948)
        
        # Testing the type of an if condition (line 69)
        if_condition_255950 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 69, 13), result_eq_255949)
        # Assigning a type to the variable 'if_condition_255950' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 13), 'if_condition_255950', if_condition_255950)
        # SSA begins for if statement (line 69)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

        @norecursion
        def _stypy_temp_lambda_168(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_168'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_168', 70, 19, True)
            # Passed parameters checking function
            _stypy_temp_lambda_168.stypy_localization = localization
            _stypy_temp_lambda_168.stypy_type_of_self = None
            _stypy_temp_lambda_168.stypy_type_store = module_type_store
            _stypy_temp_lambda_168.stypy_function_name = '_stypy_temp_lambda_168'
            _stypy_temp_lambda_168.stypy_param_names_list = ['x']
            _stypy_temp_lambda_168.stypy_varargs_param_name = None
            _stypy_temp_lambda_168.stypy_kwargs_param_name = None
            _stypy_temp_lambda_168.stypy_call_defaults = defaults
            _stypy_temp_lambda_168.stypy_call_varargs = varargs
            _stypy_temp_lambda_168.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_168', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_168', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Call to less_equal(...): (line 70)
            # Processing the call arguments (line 70)
            # Getting the type of 'x' (line 70)
            x_255952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 40), 'x', False)
            # Getting the type of 'val2' (line 70)
            val2_255953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 43), 'val2', False)
            # Processing the call keyword arguments (line 70)
            kwargs_255954 = {}
            # Getting the type of 'less_equal' (line 70)
            less_equal_255951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 29), 'less_equal', False)
            # Calling less_equal(args, kwargs) (line 70)
            less_equal_call_result_255955 = invoke(stypy.reporting.localization.Localization(__file__, 70, 29), less_equal_255951, *[x_255952, val2_255953], **kwargs_255954)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 70)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 19), 'stypy_return_type', less_equal_call_result_255955)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_168' in the type store
            # Getting the type of 'stypy_return_type' (line 70)
            stypy_return_type_255956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 19), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_255956)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_168'
            return stypy_return_type_255956

        # Assigning a type to the variable '_stypy_temp_lambda_168' (line 70)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 19), '_stypy_temp_lambda_168', _stypy_temp_lambda_168)
        # Getting the type of '_stypy_temp_lambda_168' (line 70)
        _stypy_temp_lambda_168_255957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 19), '_stypy_temp_lambda_168')
        # Assigning a type to the variable 'stypy_return_type' (line 70)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 12), 'stypy_return_type', _stypy_temp_lambda_168_255957)
        # SSA branch for the else part of an if statement (line 69)
        module_type_store.open_ssa_branch('else')

        @norecursion
        def _stypy_temp_lambda_169(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_169'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_169', 72, 19, True)
            # Passed parameters checking function
            _stypy_temp_lambda_169.stypy_localization = localization
            _stypy_temp_lambda_169.stypy_type_of_self = None
            _stypy_temp_lambda_169.stypy_type_store = module_type_store
            _stypy_temp_lambda_169.stypy_function_name = '_stypy_temp_lambda_169'
            _stypy_temp_lambda_169.stypy_param_names_list = ['x']
            _stypy_temp_lambda_169.stypy_varargs_param_name = None
            _stypy_temp_lambda_169.stypy_kwargs_param_name = None
            _stypy_temp_lambda_169.stypy_call_defaults = defaults
            _stypy_temp_lambda_169.stypy_call_varargs = varargs
            _stypy_temp_lambda_169.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_169', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_169', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Call to logical_and(...): (line 72)
            # Processing the call arguments (line 72)
            
            # Call to less(...): (line 72)
            # Processing the call arguments (line 72)
            # Getting the type of 'x' (line 72)
            x_255960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 46), 'x', False)
            # Getting the type of 'val1' (line 72)
            val1_255961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 49), 'val1', False)
            # Processing the call keyword arguments (line 72)
            kwargs_255962 = {}
            # Getting the type of 'less' (line 72)
            less_255959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 41), 'less', False)
            # Calling less(args, kwargs) (line 72)
            less_call_result_255963 = invoke(stypy.reporting.localization.Localization(__file__, 72, 41), less_255959, *[x_255960, val1_255961], **kwargs_255962)
            
            
            # Call to greater_equal(...): (line 73)
            # Processing the call arguments (line 73)
            # Getting the type of 'x' (line 73)
            x_255965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 55), 'x', False)
            # Getting the type of 'val2' (line 73)
            val2_255966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 58), 'val2', False)
            # Processing the call keyword arguments (line 73)
            kwargs_255967 = {}
            # Getting the type of 'greater_equal' (line 73)
            greater_equal_255964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 41), 'greater_equal', False)
            # Calling greater_equal(args, kwargs) (line 73)
            greater_equal_call_result_255968 = invoke(stypy.reporting.localization.Localization(__file__, 73, 41), greater_equal_255964, *[x_255965, val2_255966], **kwargs_255967)
            
            # Processing the call keyword arguments (line 72)
            kwargs_255969 = {}
            # Getting the type of 'logical_and' (line 72)
            logical_and_255958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 29), 'logical_and', False)
            # Calling logical_and(args, kwargs) (line 72)
            logical_and_call_result_255970 = invoke(stypy.reporting.localization.Localization(__file__, 72, 29), logical_and_255958, *[less_call_result_255963, greater_equal_call_result_255968], **kwargs_255969)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 72)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 19), 'stypy_return_type', logical_and_call_result_255970)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_169' in the type store
            # Getting the type of 'stypy_return_type' (line 72)
            stypy_return_type_255971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 19), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_255971)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_169'
            return stypy_return_type_255971

        # Assigning a type to the variable '_stypy_temp_lambda_169' (line 72)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 19), '_stypy_temp_lambda_169', _stypy_temp_lambda_169)
        # Getting the type of '_stypy_temp_lambda_169' (line 72)
        _stypy_temp_lambda_169_255972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 19), '_stypy_temp_lambda_169')
        # Assigning a type to the variable 'stypy_return_type' (line 72)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 12), 'stypy_return_type', _stypy_temp_lambda_169_255972)
        # SSA join for if statement (line 69)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 66)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'condfuncgen(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'condfuncgen' in the type store
        # Getting the type of 'stypy_return_type' (line 65)
        stypy_return_type_255973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_255973)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'condfuncgen'
        return stypy_return_type_255973

    # Assigning a type to the variable 'condfuncgen' (line 65)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'condfuncgen', condfuncgen)
    
    # Assigning a BinOp to a Name (line 75):
    
    # Assigning a BinOp to a Name (line 75):
    # Getting the type of 'order' (line 75)
    order_255974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 11), 'order')
    int_255975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 20), 'int')
    # Applying the binary operator '//' (line 75)
    result_floordiv_255976 = python_operator(stypy.reporting.localization.Localization(__file__, 75, 11), '//', order_255974, int_255975)
    
    int_255977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 24), 'int')
    # Applying the binary operator '+' (line 75)
    result_add_255978 = python_operator(stypy.reporting.localization.Localization(__file__, 75, 11), '+', result_floordiv_255976, int_255977)
    
    # Assigning a type to the variable 'last' (line 75)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'last', result_add_255978)
    
    # Getting the type of 'order' (line 76)
    order_255979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 7), 'order')
    int_255980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 15), 'int')
    # Applying the binary operator '%' (line 76)
    result_mod_255981 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 7), '%', order_255979, int_255980)
    
    # Testing the type of an if condition (line 76)
    if_condition_255982 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 76, 4), result_mod_255981)
    # Assigning a type to the variable 'if_condition_255982' (line 76)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'if_condition_255982', if_condition_255982)
    # SSA begins for if statement (line 76)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 77):
    
    # Assigning a Num to a Name (line 77):
    float_255983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 21), 'float')
    # Assigning a type to the variable 'startbound' (line 77)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'startbound', float_255983)
    # SSA branch for the else part of an if statement (line 76)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Num to a Name (line 79):
    
    # Assigning a Num to a Name (line 79):
    float_255984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 21), 'float')
    # Assigning a type to the variable 'startbound' (line 79)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'startbound', float_255984)
    # SSA join for if statement (line 76)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a List to a Name (line 80):
    
    # Assigning a List to a Name (line 80):
    
    # Obtaining an instance of the builtin type 'list' (line 80)
    list_255985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 80)
    # Adding element type (line 80)
    
    # Call to condfuncgen(...): (line 80)
    # Processing the call arguments (line 80)
    int_255987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 29), 'int')
    int_255988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 32), 'int')
    # Getting the type of 'startbound' (line 80)
    startbound_255989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 35), 'startbound', False)
    # Processing the call keyword arguments (line 80)
    kwargs_255990 = {}
    # Getting the type of 'condfuncgen' (line 80)
    condfuncgen_255986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 17), 'condfuncgen', False)
    # Calling condfuncgen(args, kwargs) (line 80)
    condfuncgen_call_result_255991 = invoke(stypy.reporting.localization.Localization(__file__, 80, 17), condfuncgen_255986, *[int_255987, int_255988, startbound_255989], **kwargs_255990)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 16), list_255985, condfuncgen_call_result_255991)
    
    # Assigning a type to the variable 'condfuncs' (line 80)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'condfuncs', list_255985)
    
    # Assigning a Name to a Name (line 81):
    
    # Assigning a Name to a Name (line 81):
    # Getting the type of 'startbound' (line 81)
    startbound_255992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 12), 'startbound')
    # Assigning a type to the variable 'bound' (line 81)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'bound', startbound_255992)
    
    
    # Call to xrange(...): (line 82)
    # Processing the call arguments (line 82)
    int_255994 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 22), 'int')
    # Getting the type of 'last' (line 82)
    last_255995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 25), 'last', False)
    int_255996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 32), 'int')
    # Applying the binary operator '-' (line 82)
    result_sub_255997 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 25), '-', last_255995, int_255996)
    
    # Processing the call keyword arguments (line 82)
    kwargs_255998 = {}
    # Getting the type of 'xrange' (line 82)
    xrange_255993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 15), 'xrange', False)
    # Calling xrange(args, kwargs) (line 82)
    xrange_call_result_255999 = invoke(stypy.reporting.localization.Localization(__file__, 82, 15), xrange_255993, *[int_255994, result_sub_255997], **kwargs_255998)
    
    # Testing the type of a for loop iterable (line 82)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 82, 4), xrange_call_result_255999)
    # Getting the type of the for loop variable (line 82)
    for_loop_var_256000 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 82, 4), xrange_call_result_255999)
    # Assigning a type to the variable 'num' (line 82)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'num', for_loop_var_256000)
    # SSA begins for a for statement (line 82)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to append(...): (line 83)
    # Processing the call arguments (line 83)
    
    # Call to condfuncgen(...): (line 83)
    # Processing the call arguments (line 83)
    int_256004 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 37), 'int')
    # Getting the type of 'bound' (line 83)
    bound_256005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 40), 'bound', False)
    # Getting the type of 'bound' (line 83)
    bound_256006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 47), 'bound', False)
    int_256007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 55), 'int')
    # Applying the binary operator '-' (line 83)
    result_sub_256008 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 47), '-', bound_256006, int_256007)
    
    # Processing the call keyword arguments (line 83)
    kwargs_256009 = {}
    # Getting the type of 'condfuncgen' (line 83)
    condfuncgen_256003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 25), 'condfuncgen', False)
    # Calling condfuncgen(args, kwargs) (line 83)
    condfuncgen_call_result_256010 = invoke(stypy.reporting.localization.Localization(__file__, 83, 25), condfuncgen_256003, *[int_256004, bound_256005, result_sub_256008], **kwargs_256009)
    
    # Processing the call keyword arguments (line 83)
    kwargs_256011 = {}
    # Getting the type of 'condfuncs' (line 83)
    condfuncs_256001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'condfuncs', False)
    # Obtaining the member 'append' of a type (line 83)
    append_256002 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 8), condfuncs_256001, 'append')
    # Calling append(args, kwargs) (line 83)
    append_call_result_256012 = invoke(stypy.reporting.localization.Localization(__file__, 83, 8), append_256002, *[condfuncgen_call_result_256010], **kwargs_256011)
    
    
    # Assigning a BinOp to a Name (line 84):
    
    # Assigning a BinOp to a Name (line 84):
    # Getting the type of 'bound' (line 84)
    bound_256013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 16), 'bound')
    int_256014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 24), 'int')
    # Applying the binary operator '-' (line 84)
    result_sub_256015 = python_operator(stypy.reporting.localization.Localization(__file__, 84, 16), '-', bound_256013, int_256014)
    
    # Assigning a type to the variable 'bound' (line 84)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'bound', result_sub_256015)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to append(...): (line 85)
    # Processing the call arguments (line 85)
    
    # Call to condfuncgen(...): (line 85)
    # Processing the call arguments (line 85)
    int_256019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 33), 'int')
    int_256020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 36), 'int')
    
    # Getting the type of 'order' (line 85)
    order_256021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 41), 'order', False)
    int_256022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 49), 'int')
    # Applying the binary operator '+' (line 85)
    result_add_256023 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 41), '+', order_256021, int_256022)
    
    # Applying the 'usub' unary operator (line 85)
    result___neg___256024 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 39), 'usub', result_add_256023)
    
    float_256025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 54), 'float')
    # Applying the binary operator 'div' (line 85)
    result_div_256026 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 39), 'div', result___neg___256024, float_256025)
    
    # Processing the call keyword arguments (line 85)
    kwargs_256027 = {}
    # Getting the type of 'condfuncgen' (line 85)
    condfuncgen_256018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 21), 'condfuncgen', False)
    # Calling condfuncgen(args, kwargs) (line 85)
    condfuncgen_call_result_256028 = invoke(stypy.reporting.localization.Localization(__file__, 85, 21), condfuncgen_256018, *[int_256019, int_256020, result_div_256026], **kwargs_256027)
    
    # Processing the call keyword arguments (line 85)
    kwargs_256029 = {}
    # Getting the type of 'condfuncs' (line 85)
    condfuncs_256016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'condfuncs', False)
    # Obtaining the member 'append' of a type (line 85)
    append_256017 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 4), condfuncs_256016, 'append')
    # Calling append(args, kwargs) (line 85)
    append_call_result_256030 = invoke(stypy.reporting.localization.Localization(__file__, 85, 4), append_256017, *[condfuncgen_call_result_256028], **kwargs_256029)
    
    
    # Assigning a Call to a Name (line 93):
    
    # Assigning a Call to a Name (line 93):
    
    # Call to factorial(...): (line 93)
    # Processing the call arguments (line 93)
    # Getting the type of 'order' (line 93)
    order_256032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 21), 'order', False)
    # Processing the call keyword arguments (line 93)
    kwargs_256033 = {}
    # Getting the type of 'factorial' (line 93)
    factorial_256031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 11), 'factorial', False)
    # Calling factorial(args, kwargs) (line 93)
    factorial_call_result_256034 = invoke(stypy.reporting.localization.Localization(__file__, 93, 11), factorial_256031, *[order_256032], **kwargs_256033)
    
    # Assigning a type to the variable 'fval' (line 93)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'fval', factorial_call_result_256034)

    @norecursion
    def piecefuncgen(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'piecefuncgen'
        module_type_store = module_type_store.open_function_context('piecefuncgen', 95, 4, False)
        
        # Passed parameters checking function
        piecefuncgen.stypy_localization = localization
        piecefuncgen.stypy_type_of_self = None
        piecefuncgen.stypy_type_store = module_type_store
        piecefuncgen.stypy_function_name = 'piecefuncgen'
        piecefuncgen.stypy_param_names_list = ['num']
        piecefuncgen.stypy_varargs_param_name = None
        piecefuncgen.stypy_kwargs_param_name = None
        piecefuncgen.stypy_call_defaults = defaults
        piecefuncgen.stypy_call_varargs = varargs
        piecefuncgen.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'piecefuncgen', ['num'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'piecefuncgen', localization, ['num'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'piecefuncgen(...)' code ##################

        
        # Assigning a BinOp to a Name (line 96):
        
        # Assigning a BinOp to a Name (line 96):
        # Getting the type of 'order' (line 96)
        order_256035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 13), 'order')
        int_256036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 22), 'int')
        # Applying the binary operator '//' (line 96)
        result_floordiv_256037 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 13), '//', order_256035, int_256036)
        
        # Getting the type of 'num' (line 96)
        num_256038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 26), 'num')
        # Applying the binary operator '-' (line 96)
        result_sub_256039 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 13), '-', result_floordiv_256037, num_256038)
        
        # Assigning a type to the variable 'Mk' (line 96)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'Mk', result_sub_256039)
        
        
        # Getting the type of 'Mk' (line 97)
        Mk_256040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 12), 'Mk')
        int_256041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 17), 'int')
        # Applying the binary operator '<' (line 97)
        result_lt_256042 = python_operator(stypy.reporting.localization.Localization(__file__, 97, 12), '<', Mk_256040, int_256041)
        
        # Testing the type of an if condition (line 97)
        if_condition_256043 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 97, 8), result_lt_256042)
        # Assigning a type to the variable 'if_condition_256043' (line 97)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'if_condition_256043', if_condition_256043)
        # SSA begins for if statement (line 97)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        int_256044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 19), 'int')
        # Assigning a type to the variable 'stypy_return_type' (line 98)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 12), 'stypy_return_type', int_256044)
        # SSA join for if statement (line 97)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a ListComp to a Name (line 99):
        
        # Assigning a ListComp to a Name (line 99):
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to xrange(...): (line 100)
        # Processing the call arguments (line 100)
        # Getting the type of 'Mk' (line 100)
        Mk_256068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 34), 'Mk', False)
        int_256069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 39), 'int')
        # Applying the binary operator '+' (line 100)
        result_add_256070 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 34), '+', Mk_256068, int_256069)
        
        # Processing the call keyword arguments (line 100)
        kwargs_256071 = {}
        # Getting the type of 'xrange' (line 100)
        xrange_256067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 27), 'xrange', False)
        # Calling xrange(args, kwargs) (line 100)
        xrange_call_result_256072 = invoke(stypy.reporting.localization.Localization(__file__, 100, 27), xrange_256067, *[result_add_256070], **kwargs_256071)
        
        comprehension_256073 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 18), xrange_call_result_256072)
        # Assigning a type to the variable 'k' (line 99)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 18), 'k', comprehension_256073)
        int_256045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 19), 'int')
        int_256046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 23), 'int')
        # Getting the type of 'k' (line 99)
        k_256047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 28), 'k')
        int_256048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 32), 'int')
        # Applying the binary operator '%' (line 99)
        result_mod_256049 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 28), '%', k_256047, int_256048)
        
        # Applying the binary operator '*' (line 99)
        result_mul_256050 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 23), '*', int_256046, result_mod_256049)
        
        # Applying the binary operator '-' (line 99)
        result_sub_256051 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 19), '-', int_256045, result_mul_256050)
        
        
        # Call to float(...): (line 99)
        # Processing the call arguments (line 99)
        
        # Call to comb(...): (line 99)
        # Processing the call arguments (line 99)
        # Getting the type of 'order' (line 99)
        order_256054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 49), 'order', False)
        int_256055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 57), 'int')
        # Applying the binary operator '+' (line 99)
        result_add_256056 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 49), '+', order_256054, int_256055)
        
        # Getting the type of 'k' (line 99)
        k_256057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 60), 'k', False)
        # Processing the call keyword arguments (line 99)
        int_256058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 69), 'int')
        keyword_256059 = int_256058
        kwargs_256060 = {'exact': keyword_256059}
        # Getting the type of 'comb' (line 99)
        comb_256053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 44), 'comb', False)
        # Calling comb(args, kwargs) (line 99)
        comb_call_result_256061 = invoke(stypy.reporting.localization.Localization(__file__, 99, 44), comb_256053, *[result_add_256056, k_256057], **kwargs_256060)
        
        # Processing the call keyword arguments (line 99)
        kwargs_256062 = {}
        # Getting the type of 'float' (line 99)
        float_256052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 38), 'float', False)
        # Calling float(args, kwargs) (line 99)
        float_call_result_256063 = invoke(stypy.reporting.localization.Localization(__file__, 99, 38), float_256052, *[comb_call_result_256061], **kwargs_256062)
        
        # Applying the binary operator '*' (line 99)
        result_mul_256064 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 18), '*', result_sub_256051, float_call_result_256063)
        
        # Getting the type of 'fval' (line 99)
        fval_256065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 75), 'fval')
        # Applying the binary operator 'div' (line 99)
        result_div_256066 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 73), 'div', result_mul_256064, fval_256065)
        
        list_256074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 18), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 18), list_256074, result_div_256066)
        # Assigning a type to the variable 'coeffs' (line 99)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'coeffs', list_256074)
        
        # Assigning a ListComp to a Name (line 101):
        
        # Assigning a ListComp to a Name (line 101):
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to xrange(...): (line 101)
        # Processing the call arguments (line 101)
        # Getting the type of 'Mk' (line 101)
        Mk_256080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 45), 'Mk', False)
        int_256081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 50), 'int')
        # Applying the binary operator '+' (line 101)
        result_add_256082 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 45), '+', Mk_256080, int_256081)
        
        # Processing the call keyword arguments (line 101)
        kwargs_256083 = {}
        # Getting the type of 'xrange' (line 101)
        xrange_256079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 38), 'xrange', False)
        # Calling xrange(args, kwargs) (line 101)
        xrange_call_result_256084 = invoke(stypy.reporting.localization.Localization(__file__, 101, 38), xrange_256079, *[result_add_256082], **kwargs_256083)
        
        comprehension_256085 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 18), xrange_call_result_256084)
        # Assigning a type to the variable 'k' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 18), 'k', comprehension_256085)
        
        # Getting the type of 'bound' (line 101)
        bound_256075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 19), 'bound')
        # Applying the 'usub' unary operator (line 101)
        result___neg___256076 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 18), 'usub', bound_256075)
        
        # Getting the type of 'k' (line 101)
        k_256077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 27), 'k')
        # Applying the binary operator '-' (line 101)
        result_sub_256078 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 18), '-', result___neg___256076, k_256077)
        
        list_256086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 18), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 18), list_256086, result_sub_256078)
        # Assigning a type to the variable 'shifts' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'shifts', list_256086)

        @norecursion
        def thefunc(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'thefunc'
            module_type_store = module_type_store.open_function_context('thefunc', 103, 8, False)
            
            # Passed parameters checking function
            thefunc.stypy_localization = localization
            thefunc.stypy_type_of_self = None
            thefunc.stypy_type_store = module_type_store
            thefunc.stypy_function_name = 'thefunc'
            thefunc.stypy_param_names_list = ['x']
            thefunc.stypy_varargs_param_name = None
            thefunc.stypy_kwargs_param_name = None
            thefunc.stypy_call_defaults = defaults
            thefunc.stypy_call_varargs = varargs
            thefunc.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'thefunc', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'thefunc', localization, ['x'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'thefunc(...)' code ##################

            
            # Assigning a Num to a Name (line 104):
            
            # Assigning a Num to a Name (line 104):
            float_256087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 18), 'float')
            # Assigning a type to the variable 'res' (line 104)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 12), 'res', float_256087)
            
            
            # Call to range(...): (line 105)
            # Processing the call arguments (line 105)
            # Getting the type of 'Mk' (line 105)
            Mk_256089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 27), 'Mk', False)
            int_256090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 32), 'int')
            # Applying the binary operator '+' (line 105)
            result_add_256091 = python_operator(stypy.reporting.localization.Localization(__file__, 105, 27), '+', Mk_256089, int_256090)
            
            # Processing the call keyword arguments (line 105)
            kwargs_256092 = {}
            # Getting the type of 'range' (line 105)
            range_256088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 21), 'range', False)
            # Calling range(args, kwargs) (line 105)
            range_call_result_256093 = invoke(stypy.reporting.localization.Localization(__file__, 105, 21), range_256088, *[result_add_256091], **kwargs_256092)
            
            # Testing the type of a for loop iterable (line 105)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 105, 12), range_call_result_256093)
            # Getting the type of the for loop variable (line 105)
            for_loop_var_256094 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 105, 12), range_call_result_256093)
            # Assigning a type to the variable 'k' (line 105)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 12), 'k', for_loop_var_256094)
            # SSA begins for a for statement (line 105)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Getting the type of 'res' (line 106)
            res_256095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 16), 'res')
            
            # Obtaining the type of the subscript
            # Getting the type of 'k' (line 106)
            k_256096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 30), 'k')
            # Getting the type of 'coeffs' (line 106)
            coeffs_256097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 23), 'coeffs')
            # Obtaining the member '__getitem__' of a type (line 106)
            getitem___256098 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 23), coeffs_256097, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 106)
            subscript_call_result_256099 = invoke(stypy.reporting.localization.Localization(__file__, 106, 23), getitem___256098, k_256096)
            
            # Getting the type of 'x' (line 106)
            x_256100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 36), 'x')
            
            # Obtaining the type of the subscript
            # Getting the type of 'k' (line 106)
            k_256101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 47), 'k')
            # Getting the type of 'shifts' (line 106)
            shifts_256102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 40), 'shifts')
            # Obtaining the member '__getitem__' of a type (line 106)
            getitem___256103 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 40), shifts_256102, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 106)
            subscript_call_result_256104 = invoke(stypy.reporting.localization.Localization(__file__, 106, 40), getitem___256103, k_256101)
            
            # Applying the binary operator '+' (line 106)
            result_add_256105 = python_operator(stypy.reporting.localization.Localization(__file__, 106, 36), '+', x_256100, subscript_call_result_256104)
            
            # Getting the type of 'order' (line 106)
            order_256106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 54), 'order')
            # Applying the binary operator '**' (line 106)
            result_pow_256107 = python_operator(stypy.reporting.localization.Localization(__file__, 106, 35), '**', result_add_256105, order_256106)
            
            # Applying the binary operator '*' (line 106)
            result_mul_256108 = python_operator(stypy.reporting.localization.Localization(__file__, 106, 23), '*', subscript_call_result_256099, result_pow_256107)
            
            # Applying the binary operator '+=' (line 106)
            result_iadd_256109 = python_operator(stypy.reporting.localization.Localization(__file__, 106, 16), '+=', res_256095, result_mul_256108)
            # Assigning a type to the variable 'res' (line 106)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 16), 'res', result_iadd_256109)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()
            
            # Getting the type of 'res' (line 107)
            res_256110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 19), 'res')
            # Assigning a type to the variable 'stypy_return_type' (line 107)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 12), 'stypy_return_type', res_256110)
            
            # ################# End of 'thefunc(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'thefunc' in the type store
            # Getting the type of 'stypy_return_type' (line 103)
            stypy_return_type_256111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_256111)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'thefunc'
            return stypy_return_type_256111

        # Assigning a type to the variable 'thefunc' (line 103)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'thefunc', thefunc)
        # Getting the type of 'thefunc' (line 108)
        thefunc_256112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 15), 'thefunc')
        # Assigning a type to the variable 'stypy_return_type' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'stypy_return_type', thefunc_256112)
        
        # ################# End of 'piecefuncgen(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'piecefuncgen' in the type store
        # Getting the type of 'stypy_return_type' (line 95)
        stypy_return_type_256113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_256113)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'piecefuncgen'
        return stypy_return_type_256113

    # Assigning a type to the variable 'piecefuncgen' (line 95)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 4), 'piecefuncgen', piecefuncgen)
    
    # Assigning a ListComp to a Name (line 110):
    
    # Assigning a ListComp to a Name (line 110):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to xrange(...): (line 110)
    # Processing the call arguments (line 110)
    # Getting the type of 'last' (line 110)
    last_256119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 48), 'last', False)
    # Processing the call keyword arguments (line 110)
    kwargs_256120 = {}
    # Getting the type of 'xrange' (line 110)
    xrange_256118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 41), 'xrange', False)
    # Calling xrange(args, kwargs) (line 110)
    xrange_call_result_256121 = invoke(stypy.reporting.localization.Localization(__file__, 110, 41), xrange_256118, *[last_256119], **kwargs_256120)
    
    comprehension_256122 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 16), xrange_call_result_256121)
    # Assigning a type to the variable 'k' (line 110)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 16), 'k', comprehension_256122)
    
    # Call to piecefuncgen(...): (line 110)
    # Processing the call arguments (line 110)
    # Getting the type of 'k' (line 110)
    k_256115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 29), 'k', False)
    # Processing the call keyword arguments (line 110)
    kwargs_256116 = {}
    # Getting the type of 'piecefuncgen' (line 110)
    piecefuncgen_256114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 16), 'piecefuncgen', False)
    # Calling piecefuncgen(args, kwargs) (line 110)
    piecefuncgen_call_result_256117 = invoke(stypy.reporting.localization.Localization(__file__, 110, 16), piecefuncgen_256114, *[k_256115], **kwargs_256116)
    
    list_256123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 16), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 16), list_256123, piecefuncgen_call_result_256117)
    # Assigning a type to the variable 'funclist' (line 110)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 4), 'funclist', list_256123)
    
    # Assigning a Tuple to a Subscript (line 112):
    
    # Assigning a Tuple to a Subscript (line 112):
    
    # Obtaining an instance of the builtin type 'tuple' (line 112)
    tuple_256124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 32), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 112)
    # Adding element type (line 112)
    # Getting the type of 'funclist' (line 112)
    funclist_256125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 32), 'funclist')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 32), tuple_256124, funclist_256125)
    # Adding element type (line 112)
    # Getting the type of 'condfuncs' (line 112)
    condfuncs_256126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 42), 'condfuncs')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 32), tuple_256124, condfuncs_256126)
    
    # Getting the type of '_splinefunc_cache' (line 112)
    _splinefunc_cache_256127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), '_splinefunc_cache')
    # Getting the type of 'order' (line 112)
    order_256128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 22), 'order')
    # Storing an element on a container (line 112)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 4), _splinefunc_cache_256127, (order_256128, tuple_256124))
    
    # Obtaining an instance of the builtin type 'tuple' (line 114)
    tuple_256129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 114)
    # Adding element type (line 114)
    # Getting the type of 'funclist' (line 114)
    funclist_256130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 11), 'funclist')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 11), tuple_256129, funclist_256130)
    # Adding element type (line 114)
    # Getting the type of 'condfuncs' (line 114)
    condfuncs_256131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 21), 'condfuncs')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 11), tuple_256129, condfuncs_256131)
    
    # Assigning a type to the variable 'stypy_return_type' (line 114)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 4), 'stypy_return_type', tuple_256129)
    
    # ################# End of '_bspline_piecefunctions(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_bspline_piecefunctions' in the type store
    # Getting the type of 'stypy_return_type' (line 49)
    stypy_return_type_256132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_256132)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_bspline_piecefunctions'
    return stypy_return_type_256132

# Assigning a type to the variable '_bspline_piecefunctions' (line 49)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 0), '_bspline_piecefunctions', _bspline_piecefunctions)

@norecursion
def bspline(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'bspline'
    module_type_store = module_type_store.open_function_context('bspline', 117, 0, False)
    
    # Passed parameters checking function
    bspline.stypy_localization = localization
    bspline.stypy_type_of_self = None
    bspline.stypy_type_store = module_type_store
    bspline.stypy_function_name = 'bspline'
    bspline.stypy_param_names_list = ['x', 'n']
    bspline.stypy_varargs_param_name = None
    bspline.stypy_kwargs_param_name = None
    bspline.stypy_call_defaults = defaults
    bspline.stypy_call_varargs = varargs
    bspline.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'bspline', ['x', 'n'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'bspline', localization, ['x', 'n'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'bspline(...)' code ##################

    str_256133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, (-1)), 'str', 'B-spline basis function of order n.\n\n    Notes\n    -----\n    Uses numpy.piecewise and automatic function-generator.\n\n    ')
    
    # Assigning a UnaryOp to a Name (line 125):
    
    # Assigning a UnaryOp to a Name (line 125):
    
    
    # Call to abs(...): (line 125)
    # Processing the call arguments (line 125)
    
    # Call to asarray(...): (line 125)
    # Processing the call arguments (line 125)
    # Getting the type of 'x' (line 125)
    x_256136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 22), 'x', False)
    # Processing the call keyword arguments (line 125)
    kwargs_256137 = {}
    # Getting the type of 'asarray' (line 125)
    asarray_256135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 14), 'asarray', False)
    # Calling asarray(args, kwargs) (line 125)
    asarray_call_result_256138 = invoke(stypy.reporting.localization.Localization(__file__, 125, 14), asarray_256135, *[x_256136], **kwargs_256137)
    
    # Processing the call keyword arguments (line 125)
    kwargs_256139 = {}
    # Getting the type of 'abs' (line 125)
    abs_256134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 10), 'abs', False)
    # Calling abs(args, kwargs) (line 125)
    abs_call_result_256140 = invoke(stypy.reporting.localization.Localization(__file__, 125, 10), abs_256134, *[asarray_call_result_256138], **kwargs_256139)
    
    # Applying the 'usub' unary operator (line 125)
    result___neg___256141 = python_operator(stypy.reporting.localization.Localization(__file__, 125, 9), 'usub', abs_call_result_256140)
    
    # Assigning a type to the variable 'ax' (line 125)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 4), 'ax', result___neg___256141)
    
    # Assigning a Call to a Tuple (line 127):
    
    # Assigning a Subscript to a Name (line 127):
    
    # Obtaining the type of the subscript
    int_256142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 4), 'int')
    
    # Call to _bspline_piecefunctions(...): (line 127)
    # Processing the call arguments (line 127)
    # Getting the type of 'n' (line 127)
    n_256144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 50), 'n', False)
    # Processing the call keyword arguments (line 127)
    kwargs_256145 = {}
    # Getting the type of '_bspline_piecefunctions' (line 127)
    _bspline_piecefunctions_256143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 26), '_bspline_piecefunctions', False)
    # Calling _bspline_piecefunctions(args, kwargs) (line 127)
    _bspline_piecefunctions_call_result_256146 = invoke(stypy.reporting.localization.Localization(__file__, 127, 26), _bspline_piecefunctions_256143, *[n_256144], **kwargs_256145)
    
    # Obtaining the member '__getitem__' of a type (line 127)
    getitem___256147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 4), _bspline_piecefunctions_call_result_256146, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 127)
    subscript_call_result_256148 = invoke(stypy.reporting.localization.Localization(__file__, 127, 4), getitem___256147, int_256142)
    
    # Assigning a type to the variable 'tuple_var_assignment_255804' (line 127)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 4), 'tuple_var_assignment_255804', subscript_call_result_256148)
    
    # Assigning a Subscript to a Name (line 127):
    
    # Obtaining the type of the subscript
    int_256149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 4), 'int')
    
    # Call to _bspline_piecefunctions(...): (line 127)
    # Processing the call arguments (line 127)
    # Getting the type of 'n' (line 127)
    n_256151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 50), 'n', False)
    # Processing the call keyword arguments (line 127)
    kwargs_256152 = {}
    # Getting the type of '_bspline_piecefunctions' (line 127)
    _bspline_piecefunctions_256150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 26), '_bspline_piecefunctions', False)
    # Calling _bspline_piecefunctions(args, kwargs) (line 127)
    _bspline_piecefunctions_call_result_256153 = invoke(stypy.reporting.localization.Localization(__file__, 127, 26), _bspline_piecefunctions_256150, *[n_256151], **kwargs_256152)
    
    # Obtaining the member '__getitem__' of a type (line 127)
    getitem___256154 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 4), _bspline_piecefunctions_call_result_256153, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 127)
    subscript_call_result_256155 = invoke(stypy.reporting.localization.Localization(__file__, 127, 4), getitem___256154, int_256149)
    
    # Assigning a type to the variable 'tuple_var_assignment_255805' (line 127)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 4), 'tuple_var_assignment_255805', subscript_call_result_256155)
    
    # Assigning a Name to a Name (line 127):
    # Getting the type of 'tuple_var_assignment_255804' (line 127)
    tuple_var_assignment_255804_256156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 4), 'tuple_var_assignment_255804')
    # Assigning a type to the variable 'funclist' (line 127)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 4), 'funclist', tuple_var_assignment_255804_256156)
    
    # Assigning a Name to a Name (line 127):
    # Getting the type of 'tuple_var_assignment_255805' (line 127)
    tuple_var_assignment_255805_256157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 4), 'tuple_var_assignment_255805')
    # Assigning a type to the variable 'condfuncs' (line 127)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 14), 'condfuncs', tuple_var_assignment_255805_256157)
    
    # Assigning a ListComp to a Name (line 128):
    
    # Assigning a ListComp to a Name (line 128):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'condfuncs' (line 128)
    condfuncs_256162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 37), 'condfuncs')
    comprehension_256163 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 16), condfuncs_256162)
    # Assigning a type to the variable 'func' (line 128)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 16), 'func', comprehension_256163)
    
    # Call to func(...): (line 128)
    # Processing the call arguments (line 128)
    # Getting the type of 'ax' (line 128)
    ax_256159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 21), 'ax', False)
    # Processing the call keyword arguments (line 128)
    kwargs_256160 = {}
    # Getting the type of 'func' (line 128)
    func_256158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 16), 'func', False)
    # Calling func(args, kwargs) (line 128)
    func_call_result_256161 = invoke(stypy.reporting.localization.Localization(__file__, 128, 16), func_256158, *[ax_256159], **kwargs_256160)
    
    list_256164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 16), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 16), list_256164, func_call_result_256161)
    # Assigning a type to the variable 'condlist' (line 128)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 4), 'condlist', list_256164)
    
    # Call to piecewise(...): (line 129)
    # Processing the call arguments (line 129)
    # Getting the type of 'ax' (line 129)
    ax_256166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 21), 'ax', False)
    # Getting the type of 'condlist' (line 129)
    condlist_256167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 25), 'condlist', False)
    # Getting the type of 'funclist' (line 129)
    funclist_256168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 35), 'funclist', False)
    # Processing the call keyword arguments (line 129)
    kwargs_256169 = {}
    # Getting the type of 'piecewise' (line 129)
    piecewise_256165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 11), 'piecewise', False)
    # Calling piecewise(args, kwargs) (line 129)
    piecewise_call_result_256170 = invoke(stypy.reporting.localization.Localization(__file__, 129, 11), piecewise_256165, *[ax_256166, condlist_256167, funclist_256168], **kwargs_256169)
    
    # Assigning a type to the variable 'stypy_return_type' (line 129)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 4), 'stypy_return_type', piecewise_call_result_256170)
    
    # ################# End of 'bspline(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'bspline' in the type store
    # Getting the type of 'stypy_return_type' (line 117)
    stypy_return_type_256171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_256171)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'bspline'
    return stypy_return_type_256171

# Assigning a type to the variable 'bspline' (line 117)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 0), 'bspline', bspline)

@norecursion
def gauss_spline(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'gauss_spline'
    module_type_store = module_type_store.open_function_context('gauss_spline', 132, 0, False)
    
    # Passed parameters checking function
    gauss_spline.stypy_localization = localization
    gauss_spline.stypy_type_of_self = None
    gauss_spline.stypy_type_store = module_type_store
    gauss_spline.stypy_function_name = 'gauss_spline'
    gauss_spline.stypy_param_names_list = ['x', 'n']
    gauss_spline.stypy_varargs_param_name = None
    gauss_spline.stypy_kwargs_param_name = None
    gauss_spline.stypy_call_defaults = defaults
    gauss_spline.stypy_call_varargs = varargs
    gauss_spline.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'gauss_spline', ['x', 'n'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'gauss_spline', localization, ['x', 'n'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'gauss_spline(...)' code ##################

    str_256172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, (-1)), 'str', 'Gaussian approximation to B-spline basis function of order n.\n    ')
    
    # Assigning a BinOp to a Name (line 135):
    
    # Assigning a BinOp to a Name (line 135):
    # Getting the type of 'n' (line 135)
    n_256173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 14), 'n')
    int_256174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 18), 'int')
    # Applying the binary operator '+' (line 135)
    result_add_256175 = python_operator(stypy.reporting.localization.Localization(__file__, 135, 14), '+', n_256173, int_256174)
    
    float_256176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 23), 'float')
    # Applying the binary operator 'div' (line 135)
    result_div_256177 = python_operator(stypy.reporting.localization.Localization(__file__, 135, 13), 'div', result_add_256175, float_256176)
    
    # Assigning a type to the variable 'signsq' (line 135)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 4), 'signsq', result_div_256177)
    int_256178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 11), 'int')
    
    # Call to sqrt(...): (line 136)
    # Processing the call arguments (line 136)
    int_256180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 20), 'int')
    # Getting the type of 'pi' (line 136)
    pi_256181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 24), 'pi', False)
    # Applying the binary operator '*' (line 136)
    result_mul_256182 = python_operator(stypy.reporting.localization.Localization(__file__, 136, 20), '*', int_256180, pi_256181)
    
    # Getting the type of 'signsq' (line 136)
    signsq_256183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 29), 'signsq', False)
    # Applying the binary operator '*' (line 136)
    result_mul_256184 = python_operator(stypy.reporting.localization.Localization(__file__, 136, 27), '*', result_mul_256182, signsq_256183)
    
    # Processing the call keyword arguments (line 136)
    kwargs_256185 = {}
    # Getting the type of 'sqrt' (line 136)
    sqrt_256179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 15), 'sqrt', False)
    # Calling sqrt(args, kwargs) (line 136)
    sqrt_call_result_256186 = invoke(stypy.reporting.localization.Localization(__file__, 136, 15), sqrt_256179, *[result_mul_256184], **kwargs_256185)
    
    # Applying the binary operator 'div' (line 136)
    result_div_256187 = python_operator(stypy.reporting.localization.Localization(__file__, 136, 11), 'div', int_256178, sqrt_call_result_256186)
    
    
    # Call to exp(...): (line 136)
    # Processing the call arguments (line 136)
    
    # Getting the type of 'x' (line 136)
    x_256189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 44), 'x', False)
    int_256190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 49), 'int')
    # Applying the binary operator '**' (line 136)
    result_pow_256191 = python_operator(stypy.reporting.localization.Localization(__file__, 136, 44), '**', x_256189, int_256190)
    
    # Applying the 'usub' unary operator (line 136)
    result___neg___256192 = python_operator(stypy.reporting.localization.Localization(__file__, 136, 43), 'usub', result_pow_256191)
    
    int_256193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 53), 'int')
    # Applying the binary operator 'div' (line 136)
    result_div_256194 = python_operator(stypy.reporting.localization.Localization(__file__, 136, 43), 'div', result___neg___256192, int_256193)
    
    # Getting the type of 'signsq' (line 136)
    signsq_256195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 57), 'signsq', False)
    # Applying the binary operator 'div' (line 136)
    result_div_256196 = python_operator(stypy.reporting.localization.Localization(__file__, 136, 55), 'div', result_div_256194, signsq_256195)
    
    # Processing the call keyword arguments (line 136)
    kwargs_256197 = {}
    # Getting the type of 'exp' (line 136)
    exp_256188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 39), 'exp', False)
    # Calling exp(args, kwargs) (line 136)
    exp_call_result_256198 = invoke(stypy.reporting.localization.Localization(__file__, 136, 39), exp_256188, *[result_div_256196], **kwargs_256197)
    
    # Applying the binary operator '*' (line 136)
    result_mul_256199 = python_operator(stypy.reporting.localization.Localization(__file__, 136, 37), '*', result_div_256187, exp_call_result_256198)
    
    # Assigning a type to the variable 'stypy_return_type' (line 136)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 4), 'stypy_return_type', result_mul_256199)
    
    # ################# End of 'gauss_spline(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'gauss_spline' in the type store
    # Getting the type of 'stypy_return_type' (line 132)
    stypy_return_type_256200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_256200)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'gauss_spline'
    return stypy_return_type_256200

# Assigning a type to the variable 'gauss_spline' (line 132)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 0), 'gauss_spline', gauss_spline)

@norecursion
def cubic(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'cubic'
    module_type_store = module_type_store.open_function_context('cubic', 139, 0, False)
    
    # Passed parameters checking function
    cubic.stypy_localization = localization
    cubic.stypy_type_of_self = None
    cubic.stypy_type_store = module_type_store
    cubic.stypy_function_name = 'cubic'
    cubic.stypy_param_names_list = ['x']
    cubic.stypy_varargs_param_name = None
    cubic.stypy_kwargs_param_name = None
    cubic.stypy_call_defaults = defaults
    cubic.stypy_call_varargs = varargs
    cubic.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'cubic', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'cubic', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'cubic(...)' code ##################

    str_256201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, (-1)), 'str', 'A cubic B-spline.\n\n    This is a special case of `bspline`, and equivalent to ``bspline(x, 3)``.\n    ')
    
    # Assigning a Call to a Name (line 144):
    
    # Assigning a Call to a Name (line 144):
    
    # Call to abs(...): (line 144)
    # Processing the call arguments (line 144)
    
    # Call to asarray(...): (line 144)
    # Processing the call arguments (line 144)
    # Getting the type of 'x' (line 144)
    x_256204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 21), 'x', False)
    # Processing the call keyword arguments (line 144)
    kwargs_256205 = {}
    # Getting the type of 'asarray' (line 144)
    asarray_256203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 13), 'asarray', False)
    # Calling asarray(args, kwargs) (line 144)
    asarray_call_result_256206 = invoke(stypy.reporting.localization.Localization(__file__, 144, 13), asarray_256203, *[x_256204], **kwargs_256205)
    
    # Processing the call keyword arguments (line 144)
    kwargs_256207 = {}
    # Getting the type of 'abs' (line 144)
    abs_256202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 9), 'abs', False)
    # Calling abs(args, kwargs) (line 144)
    abs_call_result_256208 = invoke(stypy.reporting.localization.Localization(__file__, 144, 9), abs_256202, *[asarray_call_result_256206], **kwargs_256207)
    
    # Assigning a type to the variable 'ax' (line 144)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 4), 'ax', abs_call_result_256208)
    
    # Assigning a Call to a Name (line 145):
    
    # Assigning a Call to a Name (line 145):
    
    # Call to zeros_like(...): (line 145)
    # Processing the call arguments (line 145)
    # Getting the type of 'ax' (line 145)
    ax_256210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 21), 'ax', False)
    # Processing the call keyword arguments (line 145)
    kwargs_256211 = {}
    # Getting the type of 'zeros_like' (line 145)
    zeros_like_256209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 10), 'zeros_like', False)
    # Calling zeros_like(args, kwargs) (line 145)
    zeros_like_call_result_256212 = invoke(stypy.reporting.localization.Localization(__file__, 145, 10), zeros_like_256209, *[ax_256210], **kwargs_256211)
    
    # Assigning a type to the variable 'res' (line 145)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 4), 'res', zeros_like_call_result_256212)
    
    # Assigning a Call to a Name (line 146):
    
    # Assigning a Call to a Name (line 146):
    
    # Call to less(...): (line 146)
    # Processing the call arguments (line 146)
    # Getting the type of 'ax' (line 146)
    ax_256214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 17), 'ax', False)
    int_256215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 21), 'int')
    # Processing the call keyword arguments (line 146)
    kwargs_256216 = {}
    # Getting the type of 'less' (line 146)
    less_256213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 12), 'less', False)
    # Calling less(args, kwargs) (line 146)
    less_call_result_256217 = invoke(stypy.reporting.localization.Localization(__file__, 146, 12), less_256213, *[ax_256214, int_256215], **kwargs_256216)
    
    # Assigning a type to the variable 'cond1' (line 146)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 4), 'cond1', less_call_result_256217)
    
    
    # Call to any(...): (line 147)
    # Processing the call keyword arguments (line 147)
    kwargs_256220 = {}
    # Getting the type of 'cond1' (line 147)
    cond1_256218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 7), 'cond1', False)
    # Obtaining the member 'any' of a type (line 147)
    any_256219 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 7), cond1_256218, 'any')
    # Calling any(args, kwargs) (line 147)
    any_call_result_256221 = invoke(stypy.reporting.localization.Localization(__file__, 147, 7), any_256219, *[], **kwargs_256220)
    
    # Testing the type of an if condition (line 147)
    if_condition_256222 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 147, 4), any_call_result_256221)
    # Assigning a type to the variable 'if_condition_256222' (line 147)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 4), 'if_condition_256222', if_condition_256222)
    # SSA begins for if statement (line 147)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 148):
    
    # Assigning a Subscript to a Name (line 148):
    
    # Obtaining the type of the subscript
    # Getting the type of 'cond1' (line 148)
    cond1_256223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 17), 'cond1')
    # Getting the type of 'ax' (line 148)
    ax_256224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 14), 'ax')
    # Obtaining the member '__getitem__' of a type (line 148)
    getitem___256225 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 14), ax_256224, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 148)
    subscript_call_result_256226 = invoke(stypy.reporting.localization.Localization(__file__, 148, 14), getitem___256225, cond1_256223)
    
    # Assigning a type to the variable 'ax1' (line 148)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 8), 'ax1', subscript_call_result_256226)
    
    # Assigning a BinOp to a Subscript (line 149):
    
    # Assigning a BinOp to a Subscript (line 149):
    float_256227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 21), 'float')
    int_256228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 27), 'int')
    # Applying the binary operator 'div' (line 149)
    result_div_256229 = python_operator(stypy.reporting.localization.Localization(__file__, 149, 21), 'div', float_256227, int_256228)
    
    float_256230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 31), 'float')
    int_256231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 37), 'int')
    # Applying the binary operator 'div' (line 149)
    result_div_256232 = python_operator(stypy.reporting.localization.Localization(__file__, 149, 31), 'div', float_256230, int_256231)
    
    # Getting the type of 'ax1' (line 149)
    ax1_256233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 41), 'ax1')
    int_256234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 48), 'int')
    # Applying the binary operator '**' (line 149)
    result_pow_256235 = python_operator(stypy.reporting.localization.Localization(__file__, 149, 41), '**', ax1_256233, int_256234)
    
    # Applying the binary operator '*' (line 149)
    result_mul_256236 = python_operator(stypy.reporting.localization.Localization(__file__, 149, 39), '*', result_div_256232, result_pow_256235)
    
    int_256237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 53), 'int')
    # Getting the type of 'ax1' (line 149)
    ax1_256238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 57), 'ax1')
    # Applying the binary operator '-' (line 149)
    result_sub_256239 = python_operator(stypy.reporting.localization.Localization(__file__, 149, 53), '-', int_256237, ax1_256238)
    
    # Applying the binary operator '*' (line 149)
    result_mul_256240 = python_operator(stypy.reporting.localization.Localization(__file__, 149, 50), '*', result_mul_256236, result_sub_256239)
    
    # Applying the binary operator '-' (line 149)
    result_sub_256241 = python_operator(stypy.reporting.localization.Localization(__file__, 149, 21), '-', result_div_256229, result_mul_256240)
    
    # Getting the type of 'res' (line 149)
    res_256242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'res')
    # Getting the type of 'cond1' (line 149)
    cond1_256243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 12), 'cond1')
    # Storing an element on a container (line 149)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 149, 8), res_256242, (cond1_256243, result_sub_256241))
    # SSA join for if statement (line 147)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 150):
    
    # Assigning a BinOp to a Name (line 150):
    
    # Getting the type of 'cond1' (line 150)
    cond1_256244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 13), 'cond1')
    # Applying the '~' unary operator (line 150)
    result_inv_256245 = python_operator(stypy.reporting.localization.Localization(__file__, 150, 12), '~', cond1_256244)
    
    
    # Call to less(...): (line 150)
    # Processing the call arguments (line 150)
    # Getting the type of 'ax' (line 150)
    ax_256247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 26), 'ax', False)
    int_256248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 30), 'int')
    # Processing the call keyword arguments (line 150)
    kwargs_256249 = {}
    # Getting the type of 'less' (line 150)
    less_256246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 21), 'less', False)
    # Calling less(args, kwargs) (line 150)
    less_call_result_256250 = invoke(stypy.reporting.localization.Localization(__file__, 150, 21), less_256246, *[ax_256247, int_256248], **kwargs_256249)
    
    # Applying the binary operator '&' (line 150)
    result_and__256251 = python_operator(stypy.reporting.localization.Localization(__file__, 150, 12), '&', result_inv_256245, less_call_result_256250)
    
    # Assigning a type to the variable 'cond2' (line 150)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 4), 'cond2', result_and__256251)
    
    
    # Call to any(...): (line 151)
    # Processing the call keyword arguments (line 151)
    kwargs_256254 = {}
    # Getting the type of 'cond2' (line 151)
    cond2_256252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 7), 'cond2', False)
    # Obtaining the member 'any' of a type (line 151)
    any_256253 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 7), cond2_256252, 'any')
    # Calling any(args, kwargs) (line 151)
    any_call_result_256255 = invoke(stypy.reporting.localization.Localization(__file__, 151, 7), any_256253, *[], **kwargs_256254)
    
    # Testing the type of an if condition (line 151)
    if_condition_256256 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 151, 4), any_call_result_256255)
    # Assigning a type to the variable 'if_condition_256256' (line 151)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 4), 'if_condition_256256', if_condition_256256)
    # SSA begins for if statement (line 151)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 152):
    
    # Assigning a Subscript to a Name (line 152):
    
    # Obtaining the type of the subscript
    # Getting the type of 'cond2' (line 152)
    cond2_256257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 17), 'cond2')
    # Getting the type of 'ax' (line 152)
    ax_256258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 14), 'ax')
    # Obtaining the member '__getitem__' of a type (line 152)
    getitem___256259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 14), ax_256258, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 152)
    subscript_call_result_256260 = invoke(stypy.reporting.localization.Localization(__file__, 152, 14), getitem___256259, cond2_256257)
    
    # Assigning a type to the variable 'ax2' (line 152)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'ax2', subscript_call_result_256260)
    
    # Assigning a BinOp to a Subscript (line 153):
    
    # Assigning a BinOp to a Subscript (line 153):
    float_256261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 21), 'float')
    int_256262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 27), 'int')
    # Applying the binary operator 'div' (line 153)
    result_div_256263 = python_operator(stypy.reporting.localization.Localization(__file__, 153, 21), 'div', float_256261, int_256262)
    
    int_256264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 32), 'int')
    # Getting the type of 'ax2' (line 153)
    ax2_256265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 36), 'ax2')
    # Applying the binary operator '-' (line 153)
    result_sub_256266 = python_operator(stypy.reporting.localization.Localization(__file__, 153, 32), '-', int_256264, ax2_256265)
    
    int_256267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 44), 'int')
    # Applying the binary operator '**' (line 153)
    result_pow_256268 = python_operator(stypy.reporting.localization.Localization(__file__, 153, 31), '**', result_sub_256266, int_256267)
    
    # Applying the binary operator '*' (line 153)
    result_mul_256269 = python_operator(stypy.reporting.localization.Localization(__file__, 153, 29), '*', result_div_256263, result_pow_256268)
    
    # Getting the type of 'res' (line 153)
    res_256270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 8), 'res')
    # Getting the type of 'cond2' (line 153)
    cond2_256271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 12), 'cond2')
    # Storing an element on a container (line 153)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 153, 8), res_256270, (cond2_256271, result_mul_256269))
    # SSA join for if statement (line 151)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'res' (line 154)
    res_256272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 11), 'res')
    # Assigning a type to the variable 'stypy_return_type' (line 154)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 4), 'stypy_return_type', res_256272)
    
    # ################# End of 'cubic(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'cubic' in the type store
    # Getting the type of 'stypy_return_type' (line 139)
    stypy_return_type_256273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_256273)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'cubic'
    return stypy_return_type_256273

# Assigning a type to the variable 'cubic' (line 139)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 0), 'cubic', cubic)

@norecursion
def quadratic(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'quadratic'
    module_type_store = module_type_store.open_function_context('quadratic', 157, 0, False)
    
    # Passed parameters checking function
    quadratic.stypy_localization = localization
    quadratic.stypy_type_of_self = None
    quadratic.stypy_type_store = module_type_store
    quadratic.stypy_function_name = 'quadratic'
    quadratic.stypy_param_names_list = ['x']
    quadratic.stypy_varargs_param_name = None
    quadratic.stypy_kwargs_param_name = None
    quadratic.stypy_call_defaults = defaults
    quadratic.stypy_call_varargs = varargs
    quadratic.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'quadratic', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'quadratic', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'quadratic(...)' code ##################

    str_256274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, (-1)), 'str', 'A quadratic B-spline.\n\n    This is a special case of `bspline`, and equivalent to ``bspline(x, 2)``.\n    ')
    
    # Assigning a Call to a Name (line 162):
    
    # Assigning a Call to a Name (line 162):
    
    # Call to abs(...): (line 162)
    # Processing the call arguments (line 162)
    
    # Call to asarray(...): (line 162)
    # Processing the call arguments (line 162)
    # Getting the type of 'x' (line 162)
    x_256277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 21), 'x', False)
    # Processing the call keyword arguments (line 162)
    kwargs_256278 = {}
    # Getting the type of 'asarray' (line 162)
    asarray_256276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 13), 'asarray', False)
    # Calling asarray(args, kwargs) (line 162)
    asarray_call_result_256279 = invoke(stypy.reporting.localization.Localization(__file__, 162, 13), asarray_256276, *[x_256277], **kwargs_256278)
    
    # Processing the call keyword arguments (line 162)
    kwargs_256280 = {}
    # Getting the type of 'abs' (line 162)
    abs_256275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 9), 'abs', False)
    # Calling abs(args, kwargs) (line 162)
    abs_call_result_256281 = invoke(stypy.reporting.localization.Localization(__file__, 162, 9), abs_256275, *[asarray_call_result_256279], **kwargs_256280)
    
    # Assigning a type to the variable 'ax' (line 162)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 4), 'ax', abs_call_result_256281)
    
    # Assigning a Call to a Name (line 163):
    
    # Assigning a Call to a Name (line 163):
    
    # Call to zeros_like(...): (line 163)
    # Processing the call arguments (line 163)
    # Getting the type of 'ax' (line 163)
    ax_256283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 21), 'ax', False)
    # Processing the call keyword arguments (line 163)
    kwargs_256284 = {}
    # Getting the type of 'zeros_like' (line 163)
    zeros_like_256282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 10), 'zeros_like', False)
    # Calling zeros_like(args, kwargs) (line 163)
    zeros_like_call_result_256285 = invoke(stypy.reporting.localization.Localization(__file__, 163, 10), zeros_like_256282, *[ax_256283], **kwargs_256284)
    
    # Assigning a type to the variable 'res' (line 163)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 4), 'res', zeros_like_call_result_256285)
    
    # Assigning a Call to a Name (line 164):
    
    # Assigning a Call to a Name (line 164):
    
    # Call to less(...): (line 164)
    # Processing the call arguments (line 164)
    # Getting the type of 'ax' (line 164)
    ax_256287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 17), 'ax', False)
    float_256288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 21), 'float')
    # Processing the call keyword arguments (line 164)
    kwargs_256289 = {}
    # Getting the type of 'less' (line 164)
    less_256286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 12), 'less', False)
    # Calling less(args, kwargs) (line 164)
    less_call_result_256290 = invoke(stypy.reporting.localization.Localization(__file__, 164, 12), less_256286, *[ax_256287, float_256288], **kwargs_256289)
    
    # Assigning a type to the variable 'cond1' (line 164)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 4), 'cond1', less_call_result_256290)
    
    
    # Call to any(...): (line 165)
    # Processing the call keyword arguments (line 165)
    kwargs_256293 = {}
    # Getting the type of 'cond1' (line 165)
    cond1_256291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 7), 'cond1', False)
    # Obtaining the member 'any' of a type (line 165)
    any_256292 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 7), cond1_256291, 'any')
    # Calling any(args, kwargs) (line 165)
    any_call_result_256294 = invoke(stypy.reporting.localization.Localization(__file__, 165, 7), any_256292, *[], **kwargs_256293)
    
    # Testing the type of an if condition (line 165)
    if_condition_256295 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 165, 4), any_call_result_256294)
    # Assigning a type to the variable 'if_condition_256295' (line 165)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 4), 'if_condition_256295', if_condition_256295)
    # SSA begins for if statement (line 165)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 166):
    
    # Assigning a Subscript to a Name (line 166):
    
    # Obtaining the type of the subscript
    # Getting the type of 'cond1' (line 166)
    cond1_256296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 17), 'cond1')
    # Getting the type of 'ax' (line 166)
    ax_256297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 14), 'ax')
    # Obtaining the member '__getitem__' of a type (line 166)
    getitem___256298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 14), ax_256297, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 166)
    subscript_call_result_256299 = invoke(stypy.reporting.localization.Localization(__file__, 166, 14), getitem___256298, cond1_256296)
    
    # Assigning a type to the variable 'ax1' (line 166)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'ax1', subscript_call_result_256299)
    
    # Assigning a BinOp to a Subscript (line 167):
    
    # Assigning a BinOp to a Subscript (line 167):
    float_256300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 21), 'float')
    # Getting the type of 'ax1' (line 167)
    ax1_256301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 28), 'ax1')
    int_256302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 35), 'int')
    # Applying the binary operator '**' (line 167)
    result_pow_256303 = python_operator(stypy.reporting.localization.Localization(__file__, 167, 28), '**', ax1_256301, int_256302)
    
    # Applying the binary operator '-' (line 167)
    result_sub_256304 = python_operator(stypy.reporting.localization.Localization(__file__, 167, 21), '-', float_256300, result_pow_256303)
    
    # Getting the type of 'res' (line 167)
    res_256305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'res')
    # Getting the type of 'cond1' (line 167)
    cond1_256306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 12), 'cond1')
    # Storing an element on a container (line 167)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 8), res_256305, (cond1_256306, result_sub_256304))
    # SSA join for if statement (line 165)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 168):
    
    # Assigning a BinOp to a Name (line 168):
    
    # Getting the type of 'cond1' (line 168)
    cond1_256307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 13), 'cond1')
    # Applying the '~' unary operator (line 168)
    result_inv_256308 = python_operator(stypy.reporting.localization.Localization(__file__, 168, 12), '~', cond1_256307)
    
    
    # Call to less(...): (line 168)
    # Processing the call arguments (line 168)
    # Getting the type of 'ax' (line 168)
    ax_256310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 26), 'ax', False)
    float_256311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 30), 'float')
    # Processing the call keyword arguments (line 168)
    kwargs_256312 = {}
    # Getting the type of 'less' (line 168)
    less_256309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 21), 'less', False)
    # Calling less(args, kwargs) (line 168)
    less_call_result_256313 = invoke(stypy.reporting.localization.Localization(__file__, 168, 21), less_256309, *[ax_256310, float_256311], **kwargs_256312)
    
    # Applying the binary operator '&' (line 168)
    result_and__256314 = python_operator(stypy.reporting.localization.Localization(__file__, 168, 12), '&', result_inv_256308, less_call_result_256313)
    
    # Assigning a type to the variable 'cond2' (line 168)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 4), 'cond2', result_and__256314)
    
    
    # Call to any(...): (line 169)
    # Processing the call keyword arguments (line 169)
    kwargs_256317 = {}
    # Getting the type of 'cond2' (line 169)
    cond2_256315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 7), 'cond2', False)
    # Obtaining the member 'any' of a type (line 169)
    any_256316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 7), cond2_256315, 'any')
    # Calling any(args, kwargs) (line 169)
    any_call_result_256318 = invoke(stypy.reporting.localization.Localization(__file__, 169, 7), any_256316, *[], **kwargs_256317)
    
    # Testing the type of an if condition (line 169)
    if_condition_256319 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 169, 4), any_call_result_256318)
    # Assigning a type to the variable 'if_condition_256319' (line 169)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 4), 'if_condition_256319', if_condition_256319)
    # SSA begins for if statement (line 169)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 170):
    
    # Assigning a Subscript to a Name (line 170):
    
    # Obtaining the type of the subscript
    # Getting the type of 'cond2' (line 170)
    cond2_256320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 17), 'cond2')
    # Getting the type of 'ax' (line 170)
    ax_256321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 14), 'ax')
    # Obtaining the member '__getitem__' of a type (line 170)
    getitem___256322 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 14), ax_256321, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 170)
    subscript_call_result_256323 = invoke(stypy.reporting.localization.Localization(__file__, 170, 14), getitem___256322, cond2_256320)
    
    # Assigning a type to the variable 'ax2' (line 170)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 8), 'ax2', subscript_call_result_256323)
    
    # Assigning a BinOp to a Subscript (line 171):
    
    # Assigning a BinOp to a Subscript (line 171):
    # Getting the type of 'ax2' (line 171)
    ax2_256324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 22), 'ax2')
    float_256325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 28), 'float')
    # Applying the binary operator '-' (line 171)
    result_sub_256326 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 22), '-', ax2_256324, float_256325)
    
    int_256327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 36), 'int')
    # Applying the binary operator '**' (line 171)
    result_pow_256328 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 21), '**', result_sub_256326, int_256327)
    
    float_256329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 40), 'float')
    # Applying the binary operator 'div' (line 171)
    result_div_256330 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 21), 'div', result_pow_256328, float_256329)
    
    # Getting the type of 'res' (line 171)
    res_256331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 8), 'res')
    # Getting the type of 'cond2' (line 171)
    cond2_256332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 12), 'cond2')
    # Storing an element on a container (line 171)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 8), res_256331, (cond2_256332, result_div_256330))
    # SSA join for if statement (line 169)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'res' (line 172)
    res_256333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 11), 'res')
    # Assigning a type to the variable 'stypy_return_type' (line 172)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 4), 'stypy_return_type', res_256333)
    
    # ################# End of 'quadratic(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'quadratic' in the type store
    # Getting the type of 'stypy_return_type' (line 157)
    stypy_return_type_256334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_256334)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'quadratic'
    return stypy_return_type_256334

# Assigning a type to the variable 'quadratic' (line 157)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 0), 'quadratic', quadratic)

@norecursion
def _coeff_smooth(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_coeff_smooth'
    module_type_store = module_type_store.open_function_context('_coeff_smooth', 175, 0, False)
    
    # Passed parameters checking function
    _coeff_smooth.stypy_localization = localization
    _coeff_smooth.stypy_type_of_self = None
    _coeff_smooth.stypy_type_store = module_type_store
    _coeff_smooth.stypy_function_name = '_coeff_smooth'
    _coeff_smooth.stypy_param_names_list = ['lam']
    _coeff_smooth.stypy_varargs_param_name = None
    _coeff_smooth.stypy_kwargs_param_name = None
    _coeff_smooth.stypy_call_defaults = defaults
    _coeff_smooth.stypy_call_varargs = varargs
    _coeff_smooth.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_coeff_smooth', ['lam'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_coeff_smooth', localization, ['lam'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_coeff_smooth(...)' code ##################

    
    # Assigning a BinOp to a Name (line 176):
    
    # Assigning a BinOp to a Name (line 176):
    int_256335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 9), 'int')
    int_256336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 13), 'int')
    # Getting the type of 'lam' (line 176)
    lam_256337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 18), 'lam')
    # Applying the binary operator '*' (line 176)
    result_mul_256338 = python_operator(stypy.reporting.localization.Localization(__file__, 176, 13), '*', int_256336, lam_256337)
    
    # Applying the binary operator '-' (line 176)
    result_sub_256339 = python_operator(stypy.reporting.localization.Localization(__file__, 176, 9), '-', int_256335, result_mul_256338)
    
    int_256340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 24), 'int')
    # Getting the type of 'lam' (line 176)
    lam_256341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 29), 'lam')
    # Applying the binary operator '*' (line 176)
    result_mul_256342 = python_operator(stypy.reporting.localization.Localization(__file__, 176, 24), '*', int_256340, lam_256341)
    
    
    # Call to sqrt(...): (line 176)
    # Processing the call arguments (line 176)
    int_256344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 40), 'int')
    int_256345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 44), 'int')
    # Getting the type of 'lam' (line 176)
    lam_256346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 50), 'lam', False)
    # Applying the binary operator '*' (line 176)
    result_mul_256347 = python_operator(stypy.reporting.localization.Localization(__file__, 176, 44), '*', int_256345, lam_256346)
    
    # Applying the binary operator '+' (line 176)
    result_add_256348 = python_operator(stypy.reporting.localization.Localization(__file__, 176, 40), '+', int_256344, result_mul_256347)
    
    # Processing the call keyword arguments (line 176)
    kwargs_256349 = {}
    # Getting the type of 'sqrt' (line 176)
    sqrt_256343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 35), 'sqrt', False)
    # Calling sqrt(args, kwargs) (line 176)
    sqrt_call_result_256350 = invoke(stypy.reporting.localization.Localization(__file__, 176, 35), sqrt_256343, *[result_add_256348], **kwargs_256349)
    
    # Applying the binary operator '*' (line 176)
    result_mul_256351 = python_operator(stypy.reporting.localization.Localization(__file__, 176, 33), '*', result_mul_256342, sqrt_call_result_256350)
    
    # Applying the binary operator '+' (line 176)
    result_add_256352 = python_operator(stypy.reporting.localization.Localization(__file__, 176, 22), '+', result_sub_256339, result_mul_256351)
    
    # Assigning a type to the variable 'xi' (line 176)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 4), 'xi', result_add_256352)
    
    # Assigning a Call to a Name (line 177):
    
    # Assigning a Call to a Name (line 177):
    
    # Call to arctan2(...): (line 177)
    # Processing the call arguments (line 177)
    
    # Call to sqrt(...): (line 177)
    # Processing the call arguments (line 177)
    int_256355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 24), 'int')
    # Getting the type of 'lam' (line 177)
    lam_256356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 30), 'lam', False)
    # Applying the binary operator '*' (line 177)
    result_mul_256357 = python_operator(stypy.reporting.localization.Localization(__file__, 177, 24), '*', int_256355, lam_256356)
    
    int_256358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 36), 'int')
    # Applying the binary operator '-' (line 177)
    result_sub_256359 = python_operator(stypy.reporting.localization.Localization(__file__, 177, 24), '-', result_mul_256357, int_256358)
    
    # Processing the call keyword arguments (line 177)
    kwargs_256360 = {}
    # Getting the type of 'sqrt' (line 177)
    sqrt_256354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 19), 'sqrt', False)
    # Calling sqrt(args, kwargs) (line 177)
    sqrt_call_result_256361 = invoke(stypy.reporting.localization.Localization(__file__, 177, 19), sqrt_256354, *[result_sub_256359], **kwargs_256360)
    
    
    # Call to sqrt(...): (line 177)
    # Processing the call arguments (line 177)
    # Getting the type of 'xi' (line 177)
    xi_256363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 45), 'xi', False)
    # Processing the call keyword arguments (line 177)
    kwargs_256364 = {}
    # Getting the type of 'sqrt' (line 177)
    sqrt_256362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 40), 'sqrt', False)
    # Calling sqrt(args, kwargs) (line 177)
    sqrt_call_result_256365 = invoke(stypy.reporting.localization.Localization(__file__, 177, 40), sqrt_256362, *[xi_256363], **kwargs_256364)
    
    # Processing the call keyword arguments (line 177)
    kwargs_256366 = {}
    # Getting the type of 'arctan2' (line 177)
    arctan2_256353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 11), 'arctan2', False)
    # Calling arctan2(args, kwargs) (line 177)
    arctan2_call_result_256367 = invoke(stypy.reporting.localization.Localization(__file__, 177, 11), arctan2_256353, *[sqrt_call_result_256361, sqrt_call_result_256365], **kwargs_256366)
    
    # Assigning a type to the variable 'omeg' (line 177)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 4), 'omeg', arctan2_call_result_256367)
    
    # Assigning a BinOp to a Name (line 178):
    
    # Assigning a BinOp to a Name (line 178):
    int_256368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 11), 'int')
    # Getting the type of 'lam' (line 178)
    lam_256369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 16), 'lam')
    # Applying the binary operator '*' (line 178)
    result_mul_256370 = python_operator(stypy.reporting.localization.Localization(__file__, 178, 11), '*', int_256368, lam_256369)
    
    int_256371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 22), 'int')
    # Applying the binary operator '-' (line 178)
    result_sub_256372 = python_operator(stypy.reporting.localization.Localization(__file__, 178, 11), '-', result_mul_256370, int_256371)
    
    
    # Call to sqrt(...): (line 178)
    # Processing the call arguments (line 178)
    # Getting the type of 'xi' (line 178)
    xi_256374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 31), 'xi', False)
    # Processing the call keyword arguments (line 178)
    kwargs_256375 = {}
    # Getting the type of 'sqrt' (line 178)
    sqrt_256373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 26), 'sqrt', False)
    # Calling sqrt(args, kwargs) (line 178)
    sqrt_call_result_256376 = invoke(stypy.reporting.localization.Localization(__file__, 178, 26), sqrt_256373, *[xi_256374], **kwargs_256375)
    
    # Applying the binary operator '-' (line 178)
    result_sub_256377 = python_operator(stypy.reporting.localization.Localization(__file__, 178, 24), '-', result_sub_256372, sqrt_call_result_256376)
    
    int_256378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 39), 'int')
    # Getting the type of 'lam' (line 178)
    lam_256379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 44), 'lam')
    # Applying the binary operator '*' (line 178)
    result_mul_256380 = python_operator(stypy.reporting.localization.Localization(__file__, 178, 39), '*', int_256378, lam_256379)
    
    # Applying the binary operator 'div' (line 178)
    result_div_256381 = python_operator(stypy.reporting.localization.Localization(__file__, 178, 10), 'div', result_sub_256377, result_mul_256380)
    
    # Assigning a type to the variable 'rho' (line 178)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 4), 'rho', result_div_256381)
    
    # Assigning a BinOp to a Name (line 179):
    
    # Assigning a BinOp to a Name (line 179):
    # Getting the type of 'rho' (line 179)
    rho_256382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 10), 'rho')
    
    # Call to sqrt(...): (line 179)
    # Processing the call arguments (line 179)
    int_256384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 22), 'int')
    # Getting the type of 'lam' (line 179)
    lam_256385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 27), 'lam', False)
    # Applying the binary operator '*' (line 179)
    result_mul_256386 = python_operator(stypy.reporting.localization.Localization(__file__, 179, 22), '*', int_256384, lam_256385)
    
    int_256387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 33), 'int')
    # Getting the type of 'lam' (line 179)
    lam_256388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 38), 'lam', False)
    # Applying the binary operator '*' (line 179)
    result_mul_256389 = python_operator(stypy.reporting.localization.Localization(__file__, 179, 33), '*', int_256387, lam_256388)
    
    
    # Call to sqrt(...): (line 179)
    # Processing the call arguments (line 179)
    int_256391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 49), 'int')
    int_256392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 53), 'int')
    # Getting the type of 'lam' (line 179)
    lam_256393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 59), 'lam', False)
    # Applying the binary operator '*' (line 179)
    result_mul_256394 = python_operator(stypy.reporting.localization.Localization(__file__, 179, 53), '*', int_256392, lam_256393)
    
    # Applying the binary operator '+' (line 179)
    result_add_256395 = python_operator(stypy.reporting.localization.Localization(__file__, 179, 49), '+', int_256391, result_mul_256394)
    
    # Processing the call keyword arguments (line 179)
    kwargs_256396 = {}
    # Getting the type of 'sqrt' (line 179)
    sqrt_256390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 44), 'sqrt', False)
    # Calling sqrt(args, kwargs) (line 179)
    sqrt_call_result_256397 = invoke(stypy.reporting.localization.Localization(__file__, 179, 44), sqrt_256390, *[result_add_256395], **kwargs_256396)
    
    # Applying the binary operator '*' (line 179)
    result_mul_256398 = python_operator(stypy.reporting.localization.Localization(__file__, 179, 42), '*', result_mul_256389, sqrt_call_result_256397)
    
    # Applying the binary operator '+' (line 179)
    result_add_256399 = python_operator(stypy.reporting.localization.Localization(__file__, 179, 22), '+', result_mul_256386, result_mul_256398)
    
    # Getting the type of 'xi' (line 179)
    xi_256400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 67), 'xi', False)
    # Applying the binary operator 'div' (line 179)
    result_div_256401 = python_operator(stypy.reporting.localization.Localization(__file__, 179, 21), 'div', result_add_256399, xi_256400)
    
    # Processing the call keyword arguments (line 179)
    kwargs_256402 = {}
    # Getting the type of 'sqrt' (line 179)
    sqrt_256383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 16), 'sqrt', False)
    # Calling sqrt(args, kwargs) (line 179)
    sqrt_call_result_256403 = invoke(stypy.reporting.localization.Localization(__file__, 179, 16), sqrt_256383, *[result_div_256401], **kwargs_256402)
    
    # Applying the binary operator '*' (line 179)
    result_mul_256404 = python_operator(stypy.reporting.localization.Localization(__file__, 179, 10), '*', rho_256382, sqrt_call_result_256403)
    
    # Assigning a type to the variable 'rho' (line 179)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 4), 'rho', result_mul_256404)
    
    # Obtaining an instance of the builtin type 'tuple' (line 180)
    tuple_256405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 180)
    # Adding element type (line 180)
    # Getting the type of 'rho' (line 180)
    rho_256406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 11), 'rho')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 180, 11), tuple_256405, rho_256406)
    # Adding element type (line 180)
    # Getting the type of 'omeg' (line 180)
    omeg_256407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 16), 'omeg')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 180, 11), tuple_256405, omeg_256407)
    
    # Assigning a type to the variable 'stypy_return_type' (line 180)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 4), 'stypy_return_type', tuple_256405)
    
    # ################# End of '_coeff_smooth(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_coeff_smooth' in the type store
    # Getting the type of 'stypy_return_type' (line 175)
    stypy_return_type_256408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_256408)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_coeff_smooth'
    return stypy_return_type_256408

# Assigning a type to the variable '_coeff_smooth' (line 175)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 0), '_coeff_smooth', _coeff_smooth)

@norecursion
def _hc(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_hc'
    module_type_store = module_type_store.open_function_context('_hc', 183, 0, False)
    
    # Passed parameters checking function
    _hc.stypy_localization = localization
    _hc.stypy_type_of_self = None
    _hc.stypy_type_store = module_type_store
    _hc.stypy_function_name = '_hc'
    _hc.stypy_param_names_list = ['k', 'cs', 'rho', 'omega']
    _hc.stypy_varargs_param_name = None
    _hc.stypy_kwargs_param_name = None
    _hc.stypy_call_defaults = defaults
    _hc.stypy_call_varargs = varargs
    _hc.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_hc', ['k', 'cs', 'rho', 'omega'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_hc', localization, ['k', 'cs', 'rho', 'omega'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_hc(...)' code ##################

    # Getting the type of 'cs' (line 184)
    cs_256409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 12), 'cs')
    
    # Call to sin(...): (line 184)
    # Processing the call arguments (line 184)
    # Getting the type of 'omega' (line 184)
    omega_256411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 21), 'omega', False)
    # Processing the call keyword arguments (line 184)
    kwargs_256412 = {}
    # Getting the type of 'sin' (line 184)
    sin_256410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 17), 'sin', False)
    # Calling sin(args, kwargs) (line 184)
    sin_call_result_256413 = invoke(stypy.reporting.localization.Localization(__file__, 184, 17), sin_256410, *[omega_256411], **kwargs_256412)
    
    # Applying the binary operator 'div' (line 184)
    result_div_256414 = python_operator(stypy.reporting.localization.Localization(__file__, 184, 12), 'div', cs_256409, sin_call_result_256413)
    
    # Getting the type of 'rho' (line 184)
    rho_256415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 31), 'rho')
    # Getting the type of 'k' (line 184)
    k_256416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 38), 'k')
    # Applying the binary operator '**' (line 184)
    result_pow_256417 = python_operator(stypy.reporting.localization.Localization(__file__, 184, 31), '**', rho_256415, k_256416)
    
    # Applying the binary operator '*' (line 184)
    result_mul_256418 = python_operator(stypy.reporting.localization.Localization(__file__, 184, 28), '*', result_div_256414, result_pow_256417)
    
    
    # Call to sin(...): (line 184)
    # Processing the call arguments (line 184)
    # Getting the type of 'omega' (line 184)
    omega_256420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 47), 'omega', False)
    # Getting the type of 'k' (line 184)
    k_256421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 56), 'k', False)
    int_256422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 60), 'int')
    # Applying the binary operator '+' (line 184)
    result_add_256423 = python_operator(stypy.reporting.localization.Localization(__file__, 184, 56), '+', k_256421, int_256422)
    
    # Applying the binary operator '*' (line 184)
    result_mul_256424 = python_operator(stypy.reporting.localization.Localization(__file__, 184, 47), '*', omega_256420, result_add_256423)
    
    # Processing the call keyword arguments (line 184)
    kwargs_256425 = {}
    # Getting the type of 'sin' (line 184)
    sin_256419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 43), 'sin', False)
    # Calling sin(args, kwargs) (line 184)
    sin_call_result_256426 = invoke(stypy.reporting.localization.Localization(__file__, 184, 43), sin_256419, *[result_mul_256424], **kwargs_256425)
    
    # Applying the binary operator '*' (line 184)
    result_mul_256427 = python_operator(stypy.reporting.localization.Localization(__file__, 184, 41), '*', result_mul_256418, sin_call_result_256426)
    
    
    # Call to greater(...): (line 185)
    # Processing the call arguments (line 185)
    # Getting the type of 'k' (line 185)
    k_256429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 20), 'k', False)
    int_256430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 23), 'int')
    # Processing the call keyword arguments (line 185)
    kwargs_256431 = {}
    # Getting the type of 'greater' (line 185)
    greater_256428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 12), 'greater', False)
    # Calling greater(args, kwargs) (line 185)
    greater_call_result_256432 = invoke(stypy.reporting.localization.Localization(__file__, 185, 12), greater_256428, *[k_256429, int_256430], **kwargs_256431)
    
    # Applying the binary operator '*' (line 184)
    result_mul_256433 = python_operator(stypy.reporting.localization.Localization(__file__, 184, 64), '*', result_mul_256427, greater_call_result_256432)
    
    # Assigning a type to the variable 'stypy_return_type' (line 184)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 4), 'stypy_return_type', result_mul_256433)
    
    # ################# End of '_hc(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_hc' in the type store
    # Getting the type of 'stypy_return_type' (line 183)
    stypy_return_type_256434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_256434)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_hc'
    return stypy_return_type_256434

# Assigning a type to the variable '_hc' (line 183)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 0), '_hc', _hc)

@norecursion
def _hs(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_hs'
    module_type_store = module_type_store.open_function_context('_hs', 188, 0, False)
    
    # Passed parameters checking function
    _hs.stypy_localization = localization
    _hs.stypy_type_of_self = None
    _hs.stypy_type_store = module_type_store
    _hs.stypy_function_name = '_hs'
    _hs.stypy_param_names_list = ['k', 'cs', 'rho', 'omega']
    _hs.stypy_varargs_param_name = None
    _hs.stypy_kwargs_param_name = None
    _hs.stypy_call_defaults = defaults
    _hs.stypy_call_varargs = varargs
    _hs.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_hs', ['k', 'cs', 'rho', 'omega'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_hs', localization, ['k', 'cs', 'rho', 'omega'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_hs(...)' code ##################

    
    # Assigning a BinOp to a Name (line 189):
    
    # Assigning a BinOp to a Name (line 189):
    # Getting the type of 'cs' (line 189)
    cs_256435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 10), 'cs')
    # Getting the type of 'cs' (line 189)
    cs_256436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 15), 'cs')
    # Applying the binary operator '*' (line 189)
    result_mul_256437 = python_operator(stypy.reporting.localization.Localization(__file__, 189, 10), '*', cs_256435, cs_256436)
    
    int_256438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 21), 'int')
    # Getting the type of 'rho' (line 189)
    rho_256439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 25), 'rho')
    # Getting the type of 'rho' (line 189)
    rho_256440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 31), 'rho')
    # Applying the binary operator '*' (line 189)
    result_mul_256441 = python_operator(stypy.reporting.localization.Localization(__file__, 189, 25), '*', rho_256439, rho_256440)
    
    # Applying the binary operator '+' (line 189)
    result_add_256442 = python_operator(stypy.reporting.localization.Localization(__file__, 189, 21), '+', int_256438, result_mul_256441)
    
    # Applying the binary operator '*' (line 189)
    result_mul_256443 = python_operator(stypy.reporting.localization.Localization(__file__, 189, 18), '*', result_mul_256437, result_add_256442)
    
    int_256444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 39), 'int')
    # Getting the type of 'rho' (line 189)
    rho_256445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 43), 'rho')
    # Getting the type of 'rho' (line 189)
    rho_256446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 49), 'rho')
    # Applying the binary operator '*' (line 189)
    result_mul_256447 = python_operator(stypy.reporting.localization.Localization(__file__, 189, 43), '*', rho_256445, rho_256446)
    
    # Applying the binary operator '-' (line 189)
    result_sub_256448 = python_operator(stypy.reporting.localization.Localization(__file__, 189, 39), '-', int_256444, result_mul_256447)
    
    # Applying the binary operator 'div' (line 189)
    result_div_256449 = python_operator(stypy.reporting.localization.Localization(__file__, 189, 36), 'div', result_mul_256443, result_sub_256448)
    
    int_256450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 11), 'int')
    int_256451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 15), 'int')
    # Getting the type of 'rho' (line 190)
    rho_256452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 19), 'rho')
    # Applying the binary operator '*' (line 190)
    result_mul_256453 = python_operator(stypy.reporting.localization.Localization(__file__, 190, 15), '*', int_256451, rho_256452)
    
    # Getting the type of 'rho' (line 190)
    rho_256454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 25), 'rho')
    # Applying the binary operator '*' (line 190)
    result_mul_256455 = python_operator(stypy.reporting.localization.Localization(__file__, 190, 23), '*', result_mul_256453, rho_256454)
    
    
    # Call to cos(...): (line 190)
    # Processing the call arguments (line 190)
    int_256457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 35), 'int')
    # Getting the type of 'omega' (line 190)
    omega_256458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 39), 'omega', False)
    # Applying the binary operator '*' (line 190)
    result_mul_256459 = python_operator(stypy.reporting.localization.Localization(__file__, 190, 35), '*', int_256457, omega_256458)
    
    # Processing the call keyword arguments (line 190)
    kwargs_256460 = {}
    # Getting the type of 'cos' (line 190)
    cos_256456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 31), 'cos', False)
    # Calling cos(args, kwargs) (line 190)
    cos_call_result_256461 = invoke(stypy.reporting.localization.Localization(__file__, 190, 31), cos_256456, *[result_mul_256459], **kwargs_256460)
    
    # Applying the binary operator '*' (line 190)
    result_mul_256462 = python_operator(stypy.reporting.localization.Localization(__file__, 190, 29), '*', result_mul_256455, cos_call_result_256461)
    
    # Applying the binary operator '-' (line 190)
    result_sub_256463 = python_operator(stypy.reporting.localization.Localization(__file__, 190, 11), '-', int_256450, result_mul_256462)
    
    # Getting the type of 'rho' (line 190)
    rho_256464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 48), 'rho')
    int_256465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 55), 'int')
    # Applying the binary operator '**' (line 190)
    result_pow_256466 = python_operator(stypy.reporting.localization.Localization(__file__, 190, 48), '**', rho_256464, int_256465)
    
    # Applying the binary operator '+' (line 190)
    result_add_256467 = python_operator(stypy.reporting.localization.Localization(__file__, 190, 46), '+', result_sub_256463, result_pow_256466)
    
    # Applying the binary operator 'div' (line 189)
    result_div_256468 = python_operator(stypy.reporting.localization.Localization(__file__, 189, 54), 'div', result_div_256449, result_add_256467)
    
    # Assigning a type to the variable 'c0' (line 189)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 4), 'c0', result_div_256468)
    
    # Assigning a BinOp to a Name (line 191):
    
    # Assigning a BinOp to a Name (line 191):
    int_256469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 13), 'int')
    # Getting the type of 'rho' (line 191)
    rho_256470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 17), 'rho')
    # Getting the type of 'rho' (line 191)
    rho_256471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 23), 'rho')
    # Applying the binary operator '*' (line 191)
    result_mul_256472 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 17), '*', rho_256470, rho_256471)
    
    # Applying the binary operator '-' (line 191)
    result_sub_256473 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 13), '-', int_256469, result_mul_256472)
    
    int_256474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 31), 'int')
    # Getting the type of 'rho' (line 191)
    rho_256475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 35), 'rho')
    # Getting the type of 'rho' (line 191)
    rho_256476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 41), 'rho')
    # Applying the binary operator '*' (line 191)
    result_mul_256477 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 35), '*', rho_256475, rho_256476)
    
    # Applying the binary operator '+' (line 191)
    result_add_256478 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 31), '+', int_256474, result_mul_256477)
    
    # Applying the binary operator 'div' (line 191)
    result_div_256479 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 12), 'div', result_sub_256473, result_add_256478)
    
    
    # Call to tan(...): (line 191)
    # Processing the call arguments (line 191)
    # Getting the type of 'omega' (line 191)
    omega_256481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 52), 'omega', False)
    # Processing the call keyword arguments (line 191)
    kwargs_256482 = {}
    # Getting the type of 'tan' (line 191)
    tan_256480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 48), 'tan', False)
    # Calling tan(args, kwargs) (line 191)
    tan_call_result_256483 = invoke(stypy.reporting.localization.Localization(__file__, 191, 48), tan_256480, *[omega_256481], **kwargs_256482)
    
    # Applying the binary operator 'div' (line 191)
    result_div_256484 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 46), 'div', result_div_256479, tan_call_result_256483)
    
    # Assigning a type to the variable 'gamma' (line 191)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 4), 'gamma', result_div_256484)
    
    # Assigning a Call to a Name (line 192):
    
    # Assigning a Call to a Name (line 192):
    
    # Call to abs(...): (line 192)
    # Processing the call arguments (line 192)
    # Getting the type of 'k' (line 192)
    k_256486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 13), 'k', False)
    # Processing the call keyword arguments (line 192)
    kwargs_256487 = {}
    # Getting the type of 'abs' (line 192)
    abs_256485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 9), 'abs', False)
    # Calling abs(args, kwargs) (line 192)
    abs_call_result_256488 = invoke(stypy.reporting.localization.Localization(__file__, 192, 9), abs_256485, *[k_256486], **kwargs_256487)
    
    # Assigning a type to the variable 'ak' (line 192)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 4), 'ak', abs_call_result_256488)
    # Getting the type of 'c0' (line 193)
    c0_256489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 11), 'c0')
    # Getting the type of 'rho' (line 193)
    rho_256490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 16), 'rho')
    # Getting the type of 'ak' (line 193)
    ak_256491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 23), 'ak')
    # Applying the binary operator '**' (line 193)
    result_pow_256492 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 16), '**', rho_256490, ak_256491)
    
    # Applying the binary operator '*' (line 193)
    result_mul_256493 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 11), '*', c0_256489, result_pow_256492)
    
    
    # Call to cos(...): (line 193)
    # Processing the call arguments (line 193)
    # Getting the type of 'omega' (line 193)
    omega_256495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 33), 'omega', False)
    # Getting the type of 'ak' (line 193)
    ak_256496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 41), 'ak', False)
    # Applying the binary operator '*' (line 193)
    result_mul_256497 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 33), '*', omega_256495, ak_256496)
    
    # Processing the call keyword arguments (line 193)
    kwargs_256498 = {}
    # Getting the type of 'cos' (line 193)
    cos_256494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 29), 'cos', False)
    # Calling cos(args, kwargs) (line 193)
    cos_call_result_256499 = invoke(stypy.reporting.localization.Localization(__file__, 193, 29), cos_256494, *[result_mul_256497], **kwargs_256498)
    
    # Getting the type of 'gamma' (line 193)
    gamma_256500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 47), 'gamma')
    
    # Call to sin(...): (line 193)
    # Processing the call arguments (line 193)
    # Getting the type of 'omega' (line 193)
    omega_256502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 59), 'omega', False)
    # Getting the type of 'ak' (line 193)
    ak_256503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 67), 'ak', False)
    # Applying the binary operator '*' (line 193)
    result_mul_256504 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 59), '*', omega_256502, ak_256503)
    
    # Processing the call keyword arguments (line 193)
    kwargs_256505 = {}
    # Getting the type of 'sin' (line 193)
    sin_256501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 55), 'sin', False)
    # Calling sin(args, kwargs) (line 193)
    sin_call_result_256506 = invoke(stypy.reporting.localization.Localization(__file__, 193, 55), sin_256501, *[result_mul_256504], **kwargs_256505)
    
    # Applying the binary operator '*' (line 193)
    result_mul_256507 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 47), '*', gamma_256500, sin_call_result_256506)
    
    # Applying the binary operator '+' (line 193)
    result_add_256508 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 29), '+', cos_call_result_256499, result_mul_256507)
    
    # Applying the binary operator '*' (line 193)
    result_mul_256509 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 26), '*', result_mul_256493, result_add_256508)
    
    # Assigning a type to the variable 'stypy_return_type' (line 193)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 4), 'stypy_return_type', result_mul_256509)
    
    # ################# End of '_hs(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_hs' in the type store
    # Getting the type of 'stypy_return_type' (line 188)
    stypy_return_type_256510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_256510)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_hs'
    return stypy_return_type_256510

# Assigning a type to the variable '_hs' (line 188)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 0), '_hs', _hs)

@norecursion
def _cubic_smooth_coeff(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_cubic_smooth_coeff'
    module_type_store = module_type_store.open_function_context('_cubic_smooth_coeff', 196, 0, False)
    
    # Passed parameters checking function
    _cubic_smooth_coeff.stypy_localization = localization
    _cubic_smooth_coeff.stypy_type_of_self = None
    _cubic_smooth_coeff.stypy_type_store = module_type_store
    _cubic_smooth_coeff.stypy_function_name = '_cubic_smooth_coeff'
    _cubic_smooth_coeff.stypy_param_names_list = ['signal', 'lamb']
    _cubic_smooth_coeff.stypy_varargs_param_name = None
    _cubic_smooth_coeff.stypy_kwargs_param_name = None
    _cubic_smooth_coeff.stypy_call_defaults = defaults
    _cubic_smooth_coeff.stypy_call_varargs = varargs
    _cubic_smooth_coeff.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_cubic_smooth_coeff', ['signal', 'lamb'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_cubic_smooth_coeff', localization, ['signal', 'lamb'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_cubic_smooth_coeff(...)' code ##################

    
    # Assigning a Call to a Tuple (line 197):
    
    # Assigning a Subscript to a Name (line 197):
    
    # Obtaining the type of the subscript
    int_256511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 4), 'int')
    
    # Call to _coeff_smooth(...): (line 197)
    # Processing the call arguments (line 197)
    # Getting the type of 'lamb' (line 197)
    lamb_256513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 31), 'lamb', False)
    # Processing the call keyword arguments (line 197)
    kwargs_256514 = {}
    # Getting the type of '_coeff_smooth' (line 197)
    _coeff_smooth_256512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 17), '_coeff_smooth', False)
    # Calling _coeff_smooth(args, kwargs) (line 197)
    _coeff_smooth_call_result_256515 = invoke(stypy.reporting.localization.Localization(__file__, 197, 17), _coeff_smooth_256512, *[lamb_256513], **kwargs_256514)
    
    # Obtaining the member '__getitem__' of a type (line 197)
    getitem___256516 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 4), _coeff_smooth_call_result_256515, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 197)
    subscript_call_result_256517 = invoke(stypy.reporting.localization.Localization(__file__, 197, 4), getitem___256516, int_256511)
    
    # Assigning a type to the variable 'tuple_var_assignment_255806' (line 197)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 4), 'tuple_var_assignment_255806', subscript_call_result_256517)
    
    # Assigning a Subscript to a Name (line 197):
    
    # Obtaining the type of the subscript
    int_256518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 4), 'int')
    
    # Call to _coeff_smooth(...): (line 197)
    # Processing the call arguments (line 197)
    # Getting the type of 'lamb' (line 197)
    lamb_256520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 31), 'lamb', False)
    # Processing the call keyword arguments (line 197)
    kwargs_256521 = {}
    # Getting the type of '_coeff_smooth' (line 197)
    _coeff_smooth_256519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 17), '_coeff_smooth', False)
    # Calling _coeff_smooth(args, kwargs) (line 197)
    _coeff_smooth_call_result_256522 = invoke(stypy.reporting.localization.Localization(__file__, 197, 17), _coeff_smooth_256519, *[lamb_256520], **kwargs_256521)
    
    # Obtaining the member '__getitem__' of a type (line 197)
    getitem___256523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 4), _coeff_smooth_call_result_256522, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 197)
    subscript_call_result_256524 = invoke(stypy.reporting.localization.Localization(__file__, 197, 4), getitem___256523, int_256518)
    
    # Assigning a type to the variable 'tuple_var_assignment_255807' (line 197)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 4), 'tuple_var_assignment_255807', subscript_call_result_256524)
    
    # Assigning a Name to a Name (line 197):
    # Getting the type of 'tuple_var_assignment_255806' (line 197)
    tuple_var_assignment_255806_256525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 4), 'tuple_var_assignment_255806')
    # Assigning a type to the variable 'rho' (line 197)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 4), 'rho', tuple_var_assignment_255806_256525)
    
    # Assigning a Name to a Name (line 197):
    # Getting the type of 'tuple_var_assignment_255807' (line 197)
    tuple_var_assignment_255807_256526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 4), 'tuple_var_assignment_255807')
    # Assigning a type to the variable 'omega' (line 197)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 9), 'omega', tuple_var_assignment_255807_256526)
    
    # Assigning a BinOp to a Name (line 198):
    
    # Assigning a BinOp to a Name (line 198):
    int_256527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 9), 'int')
    int_256528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 13), 'int')
    # Getting the type of 'rho' (line 198)
    rho_256529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 17), 'rho')
    # Applying the binary operator '*' (line 198)
    result_mul_256530 = python_operator(stypy.reporting.localization.Localization(__file__, 198, 13), '*', int_256528, rho_256529)
    
    
    # Call to cos(...): (line 198)
    # Processing the call arguments (line 198)
    # Getting the type of 'omega' (line 198)
    omega_256532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 27), 'omega', False)
    # Processing the call keyword arguments (line 198)
    kwargs_256533 = {}
    # Getting the type of 'cos' (line 198)
    cos_256531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 23), 'cos', False)
    # Calling cos(args, kwargs) (line 198)
    cos_call_result_256534 = invoke(stypy.reporting.localization.Localization(__file__, 198, 23), cos_256531, *[omega_256532], **kwargs_256533)
    
    # Applying the binary operator '*' (line 198)
    result_mul_256535 = python_operator(stypy.reporting.localization.Localization(__file__, 198, 21), '*', result_mul_256530, cos_call_result_256534)
    
    # Applying the binary operator '-' (line 198)
    result_sub_256536 = python_operator(stypy.reporting.localization.Localization(__file__, 198, 9), '-', int_256527, result_mul_256535)
    
    # Getting the type of 'rho' (line 198)
    rho_256537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 36), 'rho')
    # Getting the type of 'rho' (line 198)
    rho_256538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 42), 'rho')
    # Applying the binary operator '*' (line 198)
    result_mul_256539 = python_operator(stypy.reporting.localization.Localization(__file__, 198, 36), '*', rho_256537, rho_256538)
    
    # Applying the binary operator '+' (line 198)
    result_add_256540 = python_operator(stypy.reporting.localization.Localization(__file__, 198, 34), '+', result_sub_256536, result_mul_256539)
    
    # Assigning a type to the variable 'cs' (line 198)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 4), 'cs', result_add_256540)
    
    # Assigning a Call to a Name (line 199):
    
    # Assigning a Call to a Name (line 199):
    
    # Call to len(...): (line 199)
    # Processing the call arguments (line 199)
    # Getting the type of 'signal' (line 199)
    signal_256542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 12), 'signal', False)
    # Processing the call keyword arguments (line 199)
    kwargs_256543 = {}
    # Getting the type of 'len' (line 199)
    len_256541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'len', False)
    # Calling len(args, kwargs) (line 199)
    len_call_result_256544 = invoke(stypy.reporting.localization.Localization(__file__, 199, 8), len_256541, *[signal_256542], **kwargs_256543)
    
    # Assigning a type to the variable 'K' (line 199)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 4), 'K', len_call_result_256544)
    
    # Assigning a Call to a Name (line 200):
    
    # Assigning a Call to a Name (line 200):
    
    # Call to zeros(...): (line 200)
    # Processing the call arguments (line 200)
    
    # Obtaining an instance of the builtin type 'tuple' (line 200)
    tuple_256546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 16), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 200)
    # Adding element type (line 200)
    # Getting the type of 'K' (line 200)
    K_256547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 16), 'K', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 16), tuple_256546, K_256547)
    
    # Getting the type of 'signal' (line 200)
    signal_256548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 21), 'signal', False)
    # Obtaining the member 'dtype' of a type (line 200)
    dtype_256549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 21), signal_256548, 'dtype')
    # Obtaining the member 'char' of a type (line 200)
    char_256550 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 21), dtype_256549, 'char')
    # Processing the call keyword arguments (line 200)
    kwargs_256551 = {}
    # Getting the type of 'zeros' (line 200)
    zeros_256545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 9), 'zeros', False)
    # Calling zeros(args, kwargs) (line 200)
    zeros_call_result_256552 = invoke(stypy.reporting.localization.Localization(__file__, 200, 9), zeros_256545, *[tuple_256546, char_256550], **kwargs_256551)
    
    # Assigning a type to the variable 'yp' (line 200)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 4), 'yp', zeros_call_result_256552)
    
    # Assigning a Call to a Name (line 201):
    
    # Assigning a Call to a Name (line 201):
    
    # Call to arange(...): (line 201)
    # Processing the call arguments (line 201)
    # Getting the type of 'K' (line 201)
    K_256554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 15), 'K', False)
    # Processing the call keyword arguments (line 201)
    kwargs_256555 = {}
    # Getting the type of 'arange' (line 201)
    arange_256553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 8), 'arange', False)
    # Calling arange(args, kwargs) (line 201)
    arange_call_result_256556 = invoke(stypy.reporting.localization.Localization(__file__, 201, 8), arange_256553, *[K_256554], **kwargs_256555)
    
    # Assigning a type to the variable 'k' (line 201)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 4), 'k', arange_call_result_256556)
    
    # Assigning a BinOp to a Subscript (line 202):
    
    # Assigning a BinOp to a Subscript (line 202):
    
    # Call to _hc(...): (line 202)
    # Processing the call arguments (line 202)
    int_256558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 17), 'int')
    # Getting the type of 'cs' (line 202)
    cs_256559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 20), 'cs', False)
    # Getting the type of 'rho' (line 202)
    rho_256560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 24), 'rho', False)
    # Getting the type of 'omega' (line 202)
    omega_256561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 29), 'omega', False)
    # Processing the call keyword arguments (line 202)
    kwargs_256562 = {}
    # Getting the type of '_hc' (line 202)
    _hc_256557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 13), '_hc', False)
    # Calling _hc(args, kwargs) (line 202)
    _hc_call_result_256563 = invoke(stypy.reporting.localization.Localization(__file__, 202, 13), _hc_256557, *[int_256558, cs_256559, rho_256560, omega_256561], **kwargs_256562)
    
    
    # Obtaining the type of the subscript
    int_256564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 45), 'int')
    # Getting the type of 'signal' (line 202)
    signal_256565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 38), 'signal')
    # Obtaining the member '__getitem__' of a type (line 202)
    getitem___256566 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 38), signal_256565, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 202)
    subscript_call_result_256567 = invoke(stypy.reporting.localization.Localization(__file__, 202, 38), getitem___256566, int_256564)
    
    # Applying the binary operator '*' (line 202)
    result_mul_256568 = python_operator(stypy.reporting.localization.Localization(__file__, 202, 13), '*', _hc_call_result_256563, subscript_call_result_256567)
    
    
    # Call to reduce(...): (line 203)
    # Processing the call arguments (line 203)
    
    # Call to _hc(...): (line 203)
    # Processing the call arguments (line 203)
    # Getting the type of 'k' (line 203)
    k_256572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 28), 'k', False)
    int_256573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 32), 'int')
    # Applying the binary operator '+' (line 203)
    result_add_256574 = python_operator(stypy.reporting.localization.Localization(__file__, 203, 28), '+', k_256572, int_256573)
    
    # Getting the type of 'cs' (line 203)
    cs_256575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 35), 'cs', False)
    # Getting the type of 'rho' (line 203)
    rho_256576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 39), 'rho', False)
    # Getting the type of 'omega' (line 203)
    omega_256577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 44), 'omega', False)
    # Processing the call keyword arguments (line 203)
    kwargs_256578 = {}
    # Getting the type of '_hc' (line 203)
    _hc_256571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 24), '_hc', False)
    # Calling _hc(args, kwargs) (line 203)
    _hc_call_result_256579 = invoke(stypy.reporting.localization.Localization(__file__, 203, 24), _hc_256571, *[result_add_256574, cs_256575, rho_256576, omega_256577], **kwargs_256578)
    
    # Getting the type of 'signal' (line 203)
    signal_256580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 53), 'signal', False)
    # Applying the binary operator '*' (line 203)
    result_mul_256581 = python_operator(stypy.reporting.localization.Localization(__file__, 203, 24), '*', _hc_call_result_256579, signal_256580)
    
    # Processing the call keyword arguments (line 203)
    kwargs_256582 = {}
    # Getting the type of 'add' (line 203)
    add_256569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 13), 'add', False)
    # Obtaining the member 'reduce' of a type (line 203)
    reduce_256570 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 13), add_256569, 'reduce')
    # Calling reduce(args, kwargs) (line 203)
    reduce_call_result_256583 = invoke(stypy.reporting.localization.Localization(__file__, 203, 13), reduce_256570, *[result_mul_256581], **kwargs_256582)
    
    # Applying the binary operator '+' (line 202)
    result_add_256584 = python_operator(stypy.reporting.localization.Localization(__file__, 202, 13), '+', result_mul_256568, reduce_call_result_256583)
    
    # Getting the type of 'yp' (line 202)
    yp_256585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 4), 'yp')
    int_256586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 7), 'int')
    # Storing an element on a container (line 202)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 4), yp_256585, (int_256586, result_add_256584))
    
    # Assigning a BinOp to a Subscript (line 205):
    
    # Assigning a BinOp to a Subscript (line 205):
    
    # Call to _hc(...): (line 205)
    # Processing the call arguments (line 205)
    int_256588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 17), 'int')
    # Getting the type of 'cs' (line 205)
    cs_256589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 20), 'cs', False)
    # Getting the type of 'rho' (line 205)
    rho_256590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 24), 'rho', False)
    # Getting the type of 'omega' (line 205)
    omega_256591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 29), 'omega', False)
    # Processing the call keyword arguments (line 205)
    kwargs_256592 = {}
    # Getting the type of '_hc' (line 205)
    _hc_256587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 13), '_hc', False)
    # Calling _hc(args, kwargs) (line 205)
    _hc_call_result_256593 = invoke(stypy.reporting.localization.Localization(__file__, 205, 13), _hc_256587, *[int_256588, cs_256589, rho_256590, omega_256591], **kwargs_256592)
    
    
    # Obtaining the type of the subscript
    int_256594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 45), 'int')
    # Getting the type of 'signal' (line 205)
    signal_256595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 38), 'signal')
    # Obtaining the member '__getitem__' of a type (line 205)
    getitem___256596 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 38), signal_256595, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 205)
    subscript_call_result_256597 = invoke(stypy.reporting.localization.Localization(__file__, 205, 38), getitem___256596, int_256594)
    
    # Applying the binary operator '*' (line 205)
    result_mul_256598 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 13), '*', _hc_call_result_256593, subscript_call_result_256597)
    
    
    # Call to _hc(...): (line 206)
    # Processing the call arguments (line 206)
    int_256600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 17), 'int')
    # Getting the type of 'cs' (line 206)
    cs_256601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 20), 'cs', False)
    # Getting the type of 'rho' (line 206)
    rho_256602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 24), 'rho', False)
    # Getting the type of 'omega' (line 206)
    omega_256603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 29), 'omega', False)
    # Processing the call keyword arguments (line 206)
    kwargs_256604 = {}
    # Getting the type of '_hc' (line 206)
    _hc_256599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 13), '_hc', False)
    # Calling _hc(args, kwargs) (line 206)
    _hc_call_result_256605 = invoke(stypy.reporting.localization.Localization(__file__, 206, 13), _hc_256599, *[int_256600, cs_256601, rho_256602, omega_256603], **kwargs_256604)
    
    
    # Obtaining the type of the subscript
    int_256606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 45), 'int')
    # Getting the type of 'signal' (line 206)
    signal_256607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 38), 'signal')
    # Obtaining the member '__getitem__' of a type (line 206)
    getitem___256608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 38), signal_256607, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 206)
    subscript_call_result_256609 = invoke(stypy.reporting.localization.Localization(__file__, 206, 38), getitem___256608, int_256606)
    
    # Applying the binary operator '*' (line 206)
    result_mul_256610 = python_operator(stypy.reporting.localization.Localization(__file__, 206, 13), '*', _hc_call_result_256605, subscript_call_result_256609)
    
    # Applying the binary operator '+' (line 205)
    result_add_256611 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 13), '+', result_mul_256598, result_mul_256610)
    
    
    # Call to reduce(...): (line 207)
    # Processing the call arguments (line 207)
    
    # Call to _hc(...): (line 207)
    # Processing the call arguments (line 207)
    # Getting the type of 'k' (line 207)
    k_256615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 28), 'k', False)
    int_256616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 32), 'int')
    # Applying the binary operator '+' (line 207)
    result_add_256617 = python_operator(stypy.reporting.localization.Localization(__file__, 207, 28), '+', k_256615, int_256616)
    
    # Getting the type of 'cs' (line 207)
    cs_256618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 35), 'cs', False)
    # Getting the type of 'rho' (line 207)
    rho_256619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 39), 'rho', False)
    # Getting the type of 'omega' (line 207)
    omega_256620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 44), 'omega', False)
    # Processing the call keyword arguments (line 207)
    kwargs_256621 = {}
    # Getting the type of '_hc' (line 207)
    _hc_256614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 24), '_hc', False)
    # Calling _hc(args, kwargs) (line 207)
    _hc_call_result_256622 = invoke(stypy.reporting.localization.Localization(__file__, 207, 24), _hc_256614, *[result_add_256617, cs_256618, rho_256619, omega_256620], **kwargs_256621)
    
    # Getting the type of 'signal' (line 207)
    signal_256623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 53), 'signal', False)
    # Applying the binary operator '*' (line 207)
    result_mul_256624 = python_operator(stypy.reporting.localization.Localization(__file__, 207, 24), '*', _hc_call_result_256622, signal_256623)
    
    # Processing the call keyword arguments (line 207)
    kwargs_256625 = {}
    # Getting the type of 'add' (line 207)
    add_256612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 13), 'add', False)
    # Obtaining the member 'reduce' of a type (line 207)
    reduce_256613 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 13), add_256612, 'reduce')
    # Calling reduce(args, kwargs) (line 207)
    reduce_call_result_256626 = invoke(stypy.reporting.localization.Localization(__file__, 207, 13), reduce_256613, *[result_mul_256624], **kwargs_256625)
    
    # Applying the binary operator '+' (line 206)
    result_add_256627 = python_operator(stypy.reporting.localization.Localization(__file__, 206, 48), '+', result_add_256611, reduce_call_result_256626)
    
    # Getting the type of 'yp' (line 205)
    yp_256628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 4), 'yp')
    int_256629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 7), 'int')
    # Storing an element on a container (line 205)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 205, 4), yp_256628, (int_256629, result_add_256627))
    
    
    # Call to range(...): (line 209)
    # Processing the call arguments (line 209)
    int_256631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 19), 'int')
    # Getting the type of 'K' (line 209)
    K_256632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 22), 'K', False)
    # Processing the call keyword arguments (line 209)
    kwargs_256633 = {}
    # Getting the type of 'range' (line 209)
    range_256630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 13), 'range', False)
    # Calling range(args, kwargs) (line 209)
    range_call_result_256634 = invoke(stypy.reporting.localization.Localization(__file__, 209, 13), range_256630, *[int_256631, K_256632], **kwargs_256633)
    
    # Testing the type of a for loop iterable (line 209)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 209, 4), range_call_result_256634)
    # Getting the type of the for loop variable (line 209)
    for_loop_var_256635 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 209, 4), range_call_result_256634)
    # Assigning a type to the variable 'n' (line 209)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 4), 'n', for_loop_var_256635)
    # SSA begins for a for statement (line 209)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a BinOp to a Subscript (line 210):
    
    # Assigning a BinOp to a Subscript (line 210):
    # Getting the type of 'cs' (line 210)
    cs_256636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 17), 'cs')
    
    # Obtaining the type of the subscript
    # Getting the type of 'n' (line 210)
    n_256637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 29), 'n')
    # Getting the type of 'signal' (line 210)
    signal_256638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 22), 'signal')
    # Obtaining the member '__getitem__' of a type (line 210)
    getitem___256639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 22), signal_256638, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 210)
    subscript_call_result_256640 = invoke(stypy.reporting.localization.Localization(__file__, 210, 22), getitem___256639, n_256637)
    
    # Applying the binary operator '*' (line 210)
    result_mul_256641 = python_operator(stypy.reporting.localization.Localization(__file__, 210, 17), '*', cs_256636, subscript_call_result_256640)
    
    int_256642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 34), 'int')
    # Getting the type of 'rho' (line 210)
    rho_256643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 38), 'rho')
    # Applying the binary operator '*' (line 210)
    result_mul_256644 = python_operator(stypy.reporting.localization.Localization(__file__, 210, 34), '*', int_256642, rho_256643)
    
    
    # Call to cos(...): (line 210)
    # Processing the call arguments (line 210)
    # Getting the type of 'omega' (line 210)
    omega_256646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 48), 'omega', False)
    # Processing the call keyword arguments (line 210)
    kwargs_256647 = {}
    # Getting the type of 'cos' (line 210)
    cos_256645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 44), 'cos', False)
    # Calling cos(args, kwargs) (line 210)
    cos_call_result_256648 = invoke(stypy.reporting.localization.Localization(__file__, 210, 44), cos_256645, *[omega_256646], **kwargs_256647)
    
    # Applying the binary operator '*' (line 210)
    result_mul_256649 = python_operator(stypy.reporting.localization.Localization(__file__, 210, 42), '*', result_mul_256644, cos_call_result_256648)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'n' (line 210)
    n_256650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 60), 'n')
    int_256651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 64), 'int')
    # Applying the binary operator '-' (line 210)
    result_sub_256652 = python_operator(stypy.reporting.localization.Localization(__file__, 210, 60), '-', n_256650, int_256651)
    
    # Getting the type of 'yp' (line 210)
    yp_256653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 57), 'yp')
    # Obtaining the member '__getitem__' of a type (line 210)
    getitem___256654 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 57), yp_256653, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 210)
    subscript_call_result_256655 = invoke(stypy.reporting.localization.Localization(__file__, 210, 57), getitem___256654, result_sub_256652)
    
    # Applying the binary operator '*' (line 210)
    result_mul_256656 = python_operator(stypy.reporting.localization.Localization(__file__, 210, 55), '*', result_mul_256649, subscript_call_result_256655)
    
    # Applying the binary operator '+' (line 210)
    result_add_256657 = python_operator(stypy.reporting.localization.Localization(__file__, 210, 17), '+', result_mul_256641, result_mul_256656)
    
    # Getting the type of 'rho' (line 211)
    rho_256658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 17), 'rho')
    # Getting the type of 'rho' (line 211)
    rho_256659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 23), 'rho')
    # Applying the binary operator '*' (line 211)
    result_mul_256660 = python_operator(stypy.reporting.localization.Localization(__file__, 211, 17), '*', rho_256658, rho_256659)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'n' (line 211)
    n_256661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 32), 'n')
    int_256662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 36), 'int')
    # Applying the binary operator '-' (line 211)
    result_sub_256663 = python_operator(stypy.reporting.localization.Localization(__file__, 211, 32), '-', n_256661, int_256662)
    
    # Getting the type of 'yp' (line 211)
    yp_256664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 29), 'yp')
    # Obtaining the member '__getitem__' of a type (line 211)
    getitem___256665 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 29), yp_256664, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 211)
    subscript_call_result_256666 = invoke(stypy.reporting.localization.Localization(__file__, 211, 29), getitem___256665, result_sub_256663)
    
    # Applying the binary operator '*' (line 211)
    result_mul_256667 = python_operator(stypy.reporting.localization.Localization(__file__, 211, 27), '*', result_mul_256660, subscript_call_result_256666)
    
    # Applying the binary operator '-' (line 210)
    result_sub_256668 = python_operator(stypy.reporting.localization.Localization(__file__, 210, 67), '-', result_add_256657, result_mul_256667)
    
    # Getting the type of 'yp' (line 210)
    yp_256669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 8), 'yp')
    # Getting the type of 'n' (line 210)
    n_256670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 11), 'n')
    # Storing an element on a container (line 210)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 210, 8), yp_256669, (n_256670, result_sub_256668))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 213):
    
    # Assigning a Call to a Name (line 213):
    
    # Call to zeros(...): (line 213)
    # Processing the call arguments (line 213)
    
    # Obtaining an instance of the builtin type 'tuple' (line 213)
    tuple_256672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 213)
    # Adding element type (line 213)
    # Getting the type of 'K' (line 213)
    K_256673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 15), 'K', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 213, 15), tuple_256672, K_256673)
    
    # Getting the type of 'signal' (line 213)
    signal_256674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 20), 'signal', False)
    # Obtaining the member 'dtype' of a type (line 213)
    dtype_256675 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 20), signal_256674, 'dtype')
    # Obtaining the member 'char' of a type (line 213)
    char_256676 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 20), dtype_256675, 'char')
    # Processing the call keyword arguments (line 213)
    kwargs_256677 = {}
    # Getting the type of 'zeros' (line 213)
    zeros_256671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 8), 'zeros', False)
    # Calling zeros(args, kwargs) (line 213)
    zeros_call_result_256678 = invoke(stypy.reporting.localization.Localization(__file__, 213, 8), zeros_256671, *[tuple_256672, char_256676], **kwargs_256677)
    
    # Assigning a type to the variable 'y' (line 213)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 4), 'y', zeros_call_result_256678)
    
    # Assigning a Call to a Subscript (line 215):
    
    # Assigning a Call to a Subscript (line 215):
    
    # Call to reduce(...): (line 215)
    # Processing the call arguments (line 215)
    
    # Call to _hs(...): (line 215)
    # Processing the call arguments (line 215)
    # Getting the type of 'k' (line 215)
    k_256682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 31), 'k', False)
    # Getting the type of 'cs' (line 215)
    cs_256683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 34), 'cs', False)
    # Getting the type of 'rho' (line 215)
    rho_256684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 38), 'rho', False)
    # Getting the type of 'omega' (line 215)
    omega_256685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 43), 'omega', False)
    # Processing the call keyword arguments (line 215)
    kwargs_256686 = {}
    # Getting the type of '_hs' (line 215)
    _hs_256681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 27), '_hs', False)
    # Calling _hs(args, kwargs) (line 215)
    _hs_call_result_256687 = invoke(stypy.reporting.localization.Localization(__file__, 215, 27), _hs_256681, *[k_256682, cs_256683, rho_256684, omega_256685], **kwargs_256686)
    
    
    # Call to _hs(...): (line 216)
    # Processing the call arguments (line 216)
    # Getting the type of 'k' (line 216)
    k_256689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 31), 'k', False)
    int_256690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 35), 'int')
    # Applying the binary operator '+' (line 216)
    result_add_256691 = python_operator(stypy.reporting.localization.Localization(__file__, 216, 31), '+', k_256689, int_256690)
    
    # Getting the type of 'cs' (line 216)
    cs_256692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 38), 'cs', False)
    # Getting the type of 'rho' (line 216)
    rho_256693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 42), 'rho', False)
    # Getting the type of 'omega' (line 216)
    omega_256694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 47), 'omega', False)
    # Processing the call keyword arguments (line 216)
    kwargs_256695 = {}
    # Getting the type of '_hs' (line 216)
    _hs_256688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 27), '_hs', False)
    # Calling _hs(args, kwargs) (line 216)
    _hs_call_result_256696 = invoke(stypy.reporting.localization.Localization(__file__, 216, 27), _hs_256688, *[result_add_256691, cs_256692, rho_256693, omega_256694], **kwargs_256695)
    
    # Applying the binary operator '+' (line 215)
    result_add_256697 = python_operator(stypy.reporting.localization.Localization(__file__, 215, 27), '+', _hs_call_result_256687, _hs_call_result_256696)
    
    
    # Obtaining the type of the subscript
    int_256698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 66), 'int')
    slice_256699 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 216, 57), None, None, int_256698)
    # Getting the type of 'signal' (line 216)
    signal_256700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 57), 'signal', False)
    # Obtaining the member '__getitem__' of a type (line 216)
    getitem___256701 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 57), signal_256700, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 216)
    subscript_call_result_256702 = invoke(stypy.reporting.localization.Localization(__file__, 216, 57), getitem___256701, slice_256699)
    
    # Applying the binary operator '*' (line 215)
    result_mul_256703 = python_operator(stypy.reporting.localization.Localization(__file__, 215, 26), '*', result_add_256697, subscript_call_result_256702)
    
    # Processing the call keyword arguments (line 215)
    kwargs_256704 = {}
    # Getting the type of 'add' (line 215)
    add_256679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 15), 'add', False)
    # Obtaining the member 'reduce' of a type (line 215)
    reduce_256680 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 15), add_256679, 'reduce')
    # Calling reduce(args, kwargs) (line 215)
    reduce_call_result_256705 = invoke(stypy.reporting.localization.Localization(__file__, 215, 15), reduce_256680, *[result_mul_256703], **kwargs_256704)
    
    # Getting the type of 'y' (line 215)
    y_256706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 4), 'y')
    # Getting the type of 'K' (line 215)
    K_256707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 6), 'K')
    int_256708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 10), 'int')
    # Applying the binary operator '-' (line 215)
    result_sub_256709 = python_operator(stypy.reporting.localization.Localization(__file__, 215, 6), '-', K_256707, int_256708)
    
    # Storing an element on a container (line 215)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 215, 4), y_256706, (result_sub_256709, reduce_call_result_256705))
    
    # Assigning a Call to a Subscript (line 217):
    
    # Assigning a Call to a Subscript (line 217):
    
    # Call to reduce(...): (line 217)
    # Processing the call arguments (line 217)
    
    # Call to _hs(...): (line 217)
    # Processing the call arguments (line 217)
    # Getting the type of 'k' (line 217)
    k_256713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 31), 'k', False)
    int_256714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 35), 'int')
    # Applying the binary operator '-' (line 217)
    result_sub_256715 = python_operator(stypy.reporting.localization.Localization(__file__, 217, 31), '-', k_256713, int_256714)
    
    # Getting the type of 'cs' (line 217)
    cs_256716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 38), 'cs', False)
    # Getting the type of 'rho' (line 217)
    rho_256717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 42), 'rho', False)
    # Getting the type of 'omega' (line 217)
    omega_256718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 47), 'omega', False)
    # Processing the call keyword arguments (line 217)
    kwargs_256719 = {}
    # Getting the type of '_hs' (line 217)
    _hs_256712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 27), '_hs', False)
    # Calling _hs(args, kwargs) (line 217)
    _hs_call_result_256720 = invoke(stypy.reporting.localization.Localization(__file__, 217, 27), _hs_256712, *[result_sub_256715, cs_256716, rho_256717, omega_256718], **kwargs_256719)
    
    
    # Call to _hs(...): (line 218)
    # Processing the call arguments (line 218)
    # Getting the type of 'k' (line 218)
    k_256722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 31), 'k', False)
    int_256723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 35), 'int')
    # Applying the binary operator '+' (line 218)
    result_add_256724 = python_operator(stypy.reporting.localization.Localization(__file__, 218, 31), '+', k_256722, int_256723)
    
    # Getting the type of 'cs' (line 218)
    cs_256725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 38), 'cs', False)
    # Getting the type of 'rho' (line 218)
    rho_256726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 42), 'rho', False)
    # Getting the type of 'omega' (line 218)
    omega_256727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 47), 'omega', False)
    # Processing the call keyword arguments (line 218)
    kwargs_256728 = {}
    # Getting the type of '_hs' (line 218)
    _hs_256721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 27), '_hs', False)
    # Calling _hs(args, kwargs) (line 218)
    _hs_call_result_256729 = invoke(stypy.reporting.localization.Localization(__file__, 218, 27), _hs_256721, *[result_add_256724, cs_256725, rho_256726, omega_256727], **kwargs_256728)
    
    # Applying the binary operator '+' (line 217)
    result_add_256730 = python_operator(stypy.reporting.localization.Localization(__file__, 217, 27), '+', _hs_call_result_256720, _hs_call_result_256729)
    
    
    # Obtaining the type of the subscript
    int_256731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 66), 'int')
    slice_256732 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 218, 57), None, None, int_256731)
    # Getting the type of 'signal' (line 218)
    signal_256733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 57), 'signal', False)
    # Obtaining the member '__getitem__' of a type (line 218)
    getitem___256734 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 57), signal_256733, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 218)
    subscript_call_result_256735 = invoke(stypy.reporting.localization.Localization(__file__, 218, 57), getitem___256734, slice_256732)
    
    # Applying the binary operator '*' (line 217)
    result_mul_256736 = python_operator(stypy.reporting.localization.Localization(__file__, 217, 26), '*', result_add_256730, subscript_call_result_256735)
    
    # Processing the call keyword arguments (line 217)
    kwargs_256737 = {}
    # Getting the type of 'add' (line 217)
    add_256710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 15), 'add', False)
    # Obtaining the member 'reduce' of a type (line 217)
    reduce_256711 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 15), add_256710, 'reduce')
    # Calling reduce(args, kwargs) (line 217)
    reduce_call_result_256738 = invoke(stypy.reporting.localization.Localization(__file__, 217, 15), reduce_256711, *[result_mul_256736], **kwargs_256737)
    
    # Getting the type of 'y' (line 217)
    y_256739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 4), 'y')
    # Getting the type of 'K' (line 217)
    K_256740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 6), 'K')
    int_256741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 10), 'int')
    # Applying the binary operator '-' (line 217)
    result_sub_256742 = python_operator(stypy.reporting.localization.Localization(__file__, 217, 6), '-', K_256740, int_256741)
    
    # Storing an element on a container (line 217)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 4), y_256739, (result_sub_256742, reduce_call_result_256738))
    
    
    # Call to range(...): (line 220)
    # Processing the call arguments (line 220)
    # Getting the type of 'K' (line 220)
    K_256744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 19), 'K', False)
    int_256745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 23), 'int')
    # Applying the binary operator '-' (line 220)
    result_sub_256746 = python_operator(stypy.reporting.localization.Localization(__file__, 220, 19), '-', K_256744, int_256745)
    
    int_256747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 26), 'int')
    int_256748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 30), 'int')
    # Processing the call keyword arguments (line 220)
    kwargs_256749 = {}
    # Getting the type of 'range' (line 220)
    range_256743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 13), 'range', False)
    # Calling range(args, kwargs) (line 220)
    range_call_result_256750 = invoke(stypy.reporting.localization.Localization(__file__, 220, 13), range_256743, *[result_sub_256746, int_256747, int_256748], **kwargs_256749)
    
    # Testing the type of a for loop iterable (line 220)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 220, 4), range_call_result_256750)
    # Getting the type of the for loop variable (line 220)
    for_loop_var_256751 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 220, 4), range_call_result_256750)
    # Assigning a type to the variable 'n' (line 220)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 4), 'n', for_loop_var_256751)
    # SSA begins for a for statement (line 220)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a BinOp to a Subscript (line 221):
    
    # Assigning a BinOp to a Subscript (line 221):
    # Getting the type of 'cs' (line 221)
    cs_256752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 16), 'cs')
    
    # Obtaining the type of the subscript
    # Getting the type of 'n' (line 221)
    n_256753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 24), 'n')
    # Getting the type of 'yp' (line 221)
    yp_256754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 21), 'yp')
    # Obtaining the member '__getitem__' of a type (line 221)
    getitem___256755 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 21), yp_256754, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 221)
    subscript_call_result_256756 = invoke(stypy.reporting.localization.Localization(__file__, 221, 21), getitem___256755, n_256753)
    
    # Applying the binary operator '*' (line 221)
    result_mul_256757 = python_operator(stypy.reporting.localization.Localization(__file__, 221, 16), '*', cs_256752, subscript_call_result_256756)
    
    int_256758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 29), 'int')
    # Getting the type of 'rho' (line 221)
    rho_256759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 33), 'rho')
    # Applying the binary operator '*' (line 221)
    result_mul_256760 = python_operator(stypy.reporting.localization.Localization(__file__, 221, 29), '*', int_256758, rho_256759)
    
    
    # Call to cos(...): (line 221)
    # Processing the call arguments (line 221)
    # Getting the type of 'omega' (line 221)
    omega_256762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 43), 'omega', False)
    # Processing the call keyword arguments (line 221)
    kwargs_256763 = {}
    # Getting the type of 'cos' (line 221)
    cos_256761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 39), 'cos', False)
    # Calling cos(args, kwargs) (line 221)
    cos_call_result_256764 = invoke(stypy.reporting.localization.Localization(__file__, 221, 39), cos_256761, *[omega_256762], **kwargs_256763)
    
    # Applying the binary operator '*' (line 221)
    result_mul_256765 = python_operator(stypy.reporting.localization.Localization(__file__, 221, 37), '*', result_mul_256760, cos_call_result_256764)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'n' (line 221)
    n_256766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 54), 'n')
    int_256767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 58), 'int')
    # Applying the binary operator '+' (line 221)
    result_add_256768 = python_operator(stypy.reporting.localization.Localization(__file__, 221, 54), '+', n_256766, int_256767)
    
    # Getting the type of 'y' (line 221)
    y_256769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 52), 'y')
    # Obtaining the member '__getitem__' of a type (line 221)
    getitem___256770 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 52), y_256769, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 221)
    subscript_call_result_256771 = invoke(stypy.reporting.localization.Localization(__file__, 221, 52), getitem___256770, result_add_256768)
    
    # Applying the binary operator '*' (line 221)
    result_mul_256772 = python_operator(stypy.reporting.localization.Localization(__file__, 221, 50), '*', result_mul_256765, subscript_call_result_256771)
    
    # Applying the binary operator '+' (line 221)
    result_add_256773 = python_operator(stypy.reporting.localization.Localization(__file__, 221, 16), '+', result_mul_256757, result_mul_256772)
    
    # Getting the type of 'rho' (line 222)
    rho_256774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 16), 'rho')
    # Getting the type of 'rho' (line 222)
    rho_256775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 22), 'rho')
    # Applying the binary operator '*' (line 222)
    result_mul_256776 = python_operator(stypy.reporting.localization.Localization(__file__, 222, 16), '*', rho_256774, rho_256775)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'n' (line 222)
    n_256777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 30), 'n')
    int_256778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 34), 'int')
    # Applying the binary operator '+' (line 222)
    result_add_256779 = python_operator(stypy.reporting.localization.Localization(__file__, 222, 30), '+', n_256777, int_256778)
    
    # Getting the type of 'y' (line 222)
    y_256780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 28), 'y')
    # Obtaining the member '__getitem__' of a type (line 222)
    getitem___256781 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 28), y_256780, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 222)
    subscript_call_result_256782 = invoke(stypy.reporting.localization.Localization(__file__, 222, 28), getitem___256781, result_add_256779)
    
    # Applying the binary operator '*' (line 222)
    result_mul_256783 = python_operator(stypy.reporting.localization.Localization(__file__, 222, 26), '*', result_mul_256776, subscript_call_result_256782)
    
    # Applying the binary operator '-' (line 221)
    result_sub_256784 = python_operator(stypy.reporting.localization.Localization(__file__, 221, 61), '-', result_add_256773, result_mul_256783)
    
    # Getting the type of 'y' (line 221)
    y_256785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 8), 'y')
    # Getting the type of 'n' (line 221)
    n_256786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 10), 'n')
    # Storing an element on a container (line 221)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 221, 8), y_256785, (n_256786, result_sub_256784))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'y' (line 224)
    y_256787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 11), 'y')
    # Assigning a type to the variable 'stypy_return_type' (line 224)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 4), 'stypy_return_type', y_256787)
    
    # ################# End of '_cubic_smooth_coeff(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_cubic_smooth_coeff' in the type store
    # Getting the type of 'stypy_return_type' (line 196)
    stypy_return_type_256788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_256788)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_cubic_smooth_coeff'
    return stypy_return_type_256788

# Assigning a type to the variable '_cubic_smooth_coeff' (line 196)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 0), '_cubic_smooth_coeff', _cubic_smooth_coeff)

@norecursion
def _cubic_coeff(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_cubic_coeff'
    module_type_store = module_type_store.open_function_context('_cubic_coeff', 227, 0, False)
    
    # Passed parameters checking function
    _cubic_coeff.stypy_localization = localization
    _cubic_coeff.stypy_type_of_self = None
    _cubic_coeff.stypy_type_store = module_type_store
    _cubic_coeff.stypy_function_name = '_cubic_coeff'
    _cubic_coeff.stypy_param_names_list = ['signal']
    _cubic_coeff.stypy_varargs_param_name = None
    _cubic_coeff.stypy_kwargs_param_name = None
    _cubic_coeff.stypy_call_defaults = defaults
    _cubic_coeff.stypy_call_varargs = varargs
    _cubic_coeff.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_cubic_coeff', ['signal'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_cubic_coeff', localization, ['signal'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_cubic_coeff(...)' code ##################

    
    # Assigning a BinOp to a Name (line 228):
    
    # Assigning a BinOp to a Name (line 228):
    int_256789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 9), 'int')
    
    # Call to sqrt(...): (line 228)
    # Processing the call arguments (line 228)
    int_256791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 19), 'int')
    # Processing the call keyword arguments (line 228)
    kwargs_256792 = {}
    # Getting the type of 'sqrt' (line 228)
    sqrt_256790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 14), 'sqrt', False)
    # Calling sqrt(args, kwargs) (line 228)
    sqrt_call_result_256793 = invoke(stypy.reporting.localization.Localization(__file__, 228, 14), sqrt_256790, *[int_256791], **kwargs_256792)
    
    # Applying the binary operator '+' (line 228)
    result_add_256794 = python_operator(stypy.reporting.localization.Localization(__file__, 228, 9), '+', int_256789, sqrt_call_result_256793)
    
    # Assigning a type to the variable 'zi' (line 228)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 4), 'zi', result_add_256794)
    
    # Assigning a Call to a Name (line 229):
    
    # Assigning a Call to a Name (line 229):
    
    # Call to len(...): (line 229)
    # Processing the call arguments (line 229)
    # Getting the type of 'signal' (line 229)
    signal_256796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 12), 'signal', False)
    # Processing the call keyword arguments (line 229)
    kwargs_256797 = {}
    # Getting the type of 'len' (line 229)
    len_256795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 8), 'len', False)
    # Calling len(args, kwargs) (line 229)
    len_call_result_256798 = invoke(stypy.reporting.localization.Localization(__file__, 229, 8), len_256795, *[signal_256796], **kwargs_256797)
    
    # Assigning a type to the variable 'K' (line 229)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 4), 'K', len_call_result_256798)
    
    # Assigning a Call to a Name (line 230):
    
    # Assigning a Call to a Name (line 230):
    
    # Call to zeros(...): (line 230)
    # Processing the call arguments (line 230)
    
    # Obtaining an instance of the builtin type 'tuple' (line 230)
    tuple_256800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 19), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 230)
    # Adding element type (line 230)
    # Getting the type of 'K' (line 230)
    K_256801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 19), 'K', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 230, 19), tuple_256800, K_256801)
    
    # Getting the type of 'signal' (line 230)
    signal_256802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 24), 'signal', False)
    # Obtaining the member 'dtype' of a type (line 230)
    dtype_256803 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 24), signal_256802, 'dtype')
    # Obtaining the member 'char' of a type (line 230)
    char_256804 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 24), dtype_256803, 'char')
    # Processing the call keyword arguments (line 230)
    kwargs_256805 = {}
    # Getting the type of 'zeros' (line 230)
    zeros_256799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 12), 'zeros', False)
    # Calling zeros(args, kwargs) (line 230)
    zeros_call_result_256806 = invoke(stypy.reporting.localization.Localization(__file__, 230, 12), zeros_256799, *[tuple_256800, char_256804], **kwargs_256805)
    
    # Assigning a type to the variable 'yplus' (line 230)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 4), 'yplus', zeros_call_result_256806)
    
    # Assigning a BinOp to a Name (line 231):
    
    # Assigning a BinOp to a Name (line 231):
    # Getting the type of 'zi' (line 231)
    zi_256807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 13), 'zi')
    
    # Call to arange(...): (line 231)
    # Processing the call arguments (line 231)
    # Getting the type of 'K' (line 231)
    K_256809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 26), 'K', False)
    # Processing the call keyword arguments (line 231)
    kwargs_256810 = {}
    # Getting the type of 'arange' (line 231)
    arange_256808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 19), 'arange', False)
    # Calling arange(args, kwargs) (line 231)
    arange_call_result_256811 = invoke(stypy.reporting.localization.Localization(__file__, 231, 19), arange_256808, *[K_256809], **kwargs_256810)
    
    # Applying the binary operator '**' (line 231)
    result_pow_256812 = python_operator(stypy.reporting.localization.Localization(__file__, 231, 13), '**', zi_256807, arange_call_result_256811)
    
    # Assigning a type to the variable 'powers' (line 231)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 4), 'powers', result_pow_256812)
    
    # Assigning a BinOp to a Subscript (line 232):
    
    # Assigning a BinOp to a Subscript (line 232):
    
    # Obtaining the type of the subscript
    int_256813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 22), 'int')
    # Getting the type of 'signal' (line 232)
    signal_256814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 15), 'signal')
    # Obtaining the member '__getitem__' of a type (line 232)
    getitem___256815 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 15), signal_256814, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 232)
    subscript_call_result_256816 = invoke(stypy.reporting.localization.Localization(__file__, 232, 15), getitem___256815, int_256813)
    
    # Getting the type of 'zi' (line 232)
    zi_256817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 27), 'zi')
    
    # Call to reduce(...): (line 232)
    # Processing the call arguments (line 232)
    # Getting the type of 'powers' (line 232)
    powers_256820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 43), 'powers', False)
    # Getting the type of 'signal' (line 232)
    signal_256821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 52), 'signal', False)
    # Applying the binary operator '*' (line 232)
    result_mul_256822 = python_operator(stypy.reporting.localization.Localization(__file__, 232, 43), '*', powers_256820, signal_256821)
    
    # Processing the call keyword arguments (line 232)
    kwargs_256823 = {}
    # Getting the type of 'add' (line 232)
    add_256818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 32), 'add', False)
    # Obtaining the member 'reduce' of a type (line 232)
    reduce_256819 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 32), add_256818, 'reduce')
    # Calling reduce(args, kwargs) (line 232)
    reduce_call_result_256824 = invoke(stypy.reporting.localization.Localization(__file__, 232, 32), reduce_256819, *[result_mul_256822], **kwargs_256823)
    
    # Applying the binary operator '*' (line 232)
    result_mul_256825 = python_operator(stypy.reporting.localization.Localization(__file__, 232, 27), '*', zi_256817, reduce_call_result_256824)
    
    # Applying the binary operator '+' (line 232)
    result_add_256826 = python_operator(stypy.reporting.localization.Localization(__file__, 232, 15), '+', subscript_call_result_256816, result_mul_256825)
    
    # Getting the type of 'yplus' (line 232)
    yplus_256827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 4), 'yplus')
    int_256828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 10), 'int')
    # Storing an element on a container (line 232)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 232, 4), yplus_256827, (int_256828, result_add_256826))
    
    
    # Call to range(...): (line 233)
    # Processing the call arguments (line 233)
    int_256830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 19), 'int')
    # Getting the type of 'K' (line 233)
    K_256831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 22), 'K', False)
    # Processing the call keyword arguments (line 233)
    kwargs_256832 = {}
    # Getting the type of 'range' (line 233)
    range_256829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 13), 'range', False)
    # Calling range(args, kwargs) (line 233)
    range_call_result_256833 = invoke(stypy.reporting.localization.Localization(__file__, 233, 13), range_256829, *[int_256830, K_256831], **kwargs_256832)
    
    # Testing the type of a for loop iterable (line 233)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 233, 4), range_call_result_256833)
    # Getting the type of the for loop variable (line 233)
    for_loop_var_256834 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 233, 4), range_call_result_256833)
    # Assigning a type to the variable 'k' (line 233)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 4), 'k', for_loop_var_256834)
    # SSA begins for a for statement (line 233)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a BinOp to a Subscript (line 234):
    
    # Assigning a BinOp to a Subscript (line 234):
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 234)
    k_256835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 26), 'k')
    # Getting the type of 'signal' (line 234)
    signal_256836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 19), 'signal')
    # Obtaining the member '__getitem__' of a type (line 234)
    getitem___256837 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 19), signal_256836, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 234)
    subscript_call_result_256838 = invoke(stypy.reporting.localization.Localization(__file__, 234, 19), getitem___256837, k_256835)
    
    # Getting the type of 'zi' (line 234)
    zi_256839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 31), 'zi')
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 234)
    k_256840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 42), 'k')
    int_256841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 46), 'int')
    # Applying the binary operator '-' (line 234)
    result_sub_256842 = python_operator(stypy.reporting.localization.Localization(__file__, 234, 42), '-', k_256840, int_256841)
    
    # Getting the type of 'yplus' (line 234)
    yplus_256843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 36), 'yplus')
    # Obtaining the member '__getitem__' of a type (line 234)
    getitem___256844 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 36), yplus_256843, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 234)
    subscript_call_result_256845 = invoke(stypy.reporting.localization.Localization(__file__, 234, 36), getitem___256844, result_sub_256842)
    
    # Applying the binary operator '*' (line 234)
    result_mul_256846 = python_operator(stypy.reporting.localization.Localization(__file__, 234, 31), '*', zi_256839, subscript_call_result_256845)
    
    # Applying the binary operator '+' (line 234)
    result_add_256847 = python_operator(stypy.reporting.localization.Localization(__file__, 234, 19), '+', subscript_call_result_256838, result_mul_256846)
    
    # Getting the type of 'yplus' (line 234)
    yplus_256848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 8), 'yplus')
    # Getting the type of 'k' (line 234)
    k_256849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 14), 'k')
    # Storing an element on a container (line 234)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 234, 8), yplus_256848, (k_256849, result_add_256847))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 235):
    
    # Assigning a Call to a Name (line 235):
    
    # Call to zeros(...): (line 235)
    # Processing the call arguments (line 235)
    
    # Obtaining an instance of the builtin type 'tuple' (line 235)
    tuple_256851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 20), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 235)
    # Adding element type (line 235)
    # Getting the type of 'K' (line 235)
    K_256852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 20), 'K', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 20), tuple_256851, K_256852)
    
    # Getting the type of 'signal' (line 235)
    signal_256853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 25), 'signal', False)
    # Obtaining the member 'dtype' of a type (line 235)
    dtype_256854 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 25), signal_256853, 'dtype')
    # Processing the call keyword arguments (line 235)
    kwargs_256855 = {}
    # Getting the type of 'zeros' (line 235)
    zeros_256850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 13), 'zeros', False)
    # Calling zeros(args, kwargs) (line 235)
    zeros_call_result_256856 = invoke(stypy.reporting.localization.Localization(__file__, 235, 13), zeros_256850, *[tuple_256851, dtype_256854], **kwargs_256855)
    
    # Assigning a type to the variable 'output' (line 235)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 4), 'output', zeros_call_result_256856)
    
    # Assigning a BinOp to a Subscript (line 236):
    
    # Assigning a BinOp to a Subscript (line 236):
    # Getting the type of 'zi' (line 236)
    zi_256857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 20), 'zi')
    # Getting the type of 'zi' (line 236)
    zi_256858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 26), 'zi')
    int_256859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 31), 'int')
    # Applying the binary operator '-' (line 236)
    result_sub_256860 = python_operator(stypy.reporting.localization.Localization(__file__, 236, 26), '-', zi_256858, int_256859)
    
    # Applying the binary operator 'div' (line 236)
    result_div_256861 = python_operator(stypy.reporting.localization.Localization(__file__, 236, 20), 'div', zi_256857, result_sub_256860)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'K' (line 236)
    K_256862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 42), 'K')
    int_256863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 46), 'int')
    # Applying the binary operator '-' (line 236)
    result_sub_256864 = python_operator(stypy.reporting.localization.Localization(__file__, 236, 42), '-', K_256862, int_256863)
    
    # Getting the type of 'yplus' (line 236)
    yplus_256865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 36), 'yplus')
    # Obtaining the member '__getitem__' of a type (line 236)
    getitem___256866 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 36), yplus_256865, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 236)
    subscript_call_result_256867 = invoke(stypy.reporting.localization.Localization(__file__, 236, 36), getitem___256866, result_sub_256864)
    
    # Applying the binary operator '*' (line 236)
    result_mul_256868 = python_operator(stypy.reporting.localization.Localization(__file__, 236, 34), '*', result_div_256861, subscript_call_result_256867)
    
    # Getting the type of 'output' (line 236)
    output_256869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 4), 'output')
    # Getting the type of 'K' (line 236)
    K_256870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 11), 'K')
    int_256871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 15), 'int')
    # Applying the binary operator '-' (line 236)
    result_sub_256872 = python_operator(stypy.reporting.localization.Localization(__file__, 236, 11), '-', K_256870, int_256871)
    
    # Storing an element on a container (line 236)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 236, 4), output_256869, (result_sub_256872, result_mul_256868))
    
    
    # Call to range(...): (line 237)
    # Processing the call arguments (line 237)
    # Getting the type of 'K' (line 237)
    K_256874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 19), 'K', False)
    int_256875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 23), 'int')
    # Applying the binary operator '-' (line 237)
    result_sub_256876 = python_operator(stypy.reporting.localization.Localization(__file__, 237, 19), '-', K_256874, int_256875)
    
    int_256877 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 26), 'int')
    int_256878 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 30), 'int')
    # Processing the call keyword arguments (line 237)
    kwargs_256879 = {}
    # Getting the type of 'range' (line 237)
    range_256873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 13), 'range', False)
    # Calling range(args, kwargs) (line 237)
    range_call_result_256880 = invoke(stypy.reporting.localization.Localization(__file__, 237, 13), range_256873, *[result_sub_256876, int_256877, int_256878], **kwargs_256879)
    
    # Testing the type of a for loop iterable (line 237)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 237, 4), range_call_result_256880)
    # Getting the type of the for loop variable (line 237)
    for_loop_var_256881 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 237, 4), range_call_result_256880)
    # Assigning a type to the variable 'k' (line 237)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 4), 'k', for_loop_var_256881)
    # SSA begins for a for statement (line 237)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a BinOp to a Subscript (line 238):
    
    # Assigning a BinOp to a Subscript (line 238):
    # Getting the type of 'zi' (line 238)
    zi_256882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 20), 'zi')
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 238)
    k_256883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 33), 'k')
    int_256884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 37), 'int')
    # Applying the binary operator '+' (line 238)
    result_add_256885 = python_operator(stypy.reporting.localization.Localization(__file__, 238, 33), '+', k_256883, int_256884)
    
    # Getting the type of 'output' (line 238)
    output_256886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 26), 'output')
    # Obtaining the member '__getitem__' of a type (line 238)
    getitem___256887 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 26), output_256886, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 238)
    subscript_call_result_256888 = invoke(stypy.reporting.localization.Localization(__file__, 238, 26), getitem___256887, result_add_256885)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 238)
    k_256889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 48), 'k')
    # Getting the type of 'yplus' (line 238)
    yplus_256890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 42), 'yplus')
    # Obtaining the member '__getitem__' of a type (line 238)
    getitem___256891 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 42), yplus_256890, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 238)
    subscript_call_result_256892 = invoke(stypy.reporting.localization.Localization(__file__, 238, 42), getitem___256891, k_256889)
    
    # Applying the binary operator '-' (line 238)
    result_sub_256893 = python_operator(stypy.reporting.localization.Localization(__file__, 238, 26), '-', subscript_call_result_256888, subscript_call_result_256892)
    
    # Applying the binary operator '*' (line 238)
    result_mul_256894 = python_operator(stypy.reporting.localization.Localization(__file__, 238, 20), '*', zi_256882, result_sub_256893)
    
    # Getting the type of 'output' (line 238)
    output_256895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 8), 'output')
    # Getting the type of 'k' (line 238)
    k_256896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 15), 'k')
    # Storing an element on a container (line 238)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 238, 8), output_256895, (k_256896, result_mul_256894))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'output' (line 239)
    output_256897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 11), 'output')
    float_256898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 20), 'float')
    # Applying the binary operator '*' (line 239)
    result_mul_256899 = python_operator(stypy.reporting.localization.Localization(__file__, 239, 11), '*', output_256897, float_256898)
    
    # Assigning a type to the variable 'stypy_return_type' (line 239)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 4), 'stypy_return_type', result_mul_256899)
    
    # ################# End of '_cubic_coeff(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_cubic_coeff' in the type store
    # Getting the type of 'stypy_return_type' (line 227)
    stypy_return_type_256900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_256900)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_cubic_coeff'
    return stypy_return_type_256900

# Assigning a type to the variable '_cubic_coeff' (line 227)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 0), '_cubic_coeff', _cubic_coeff)

@norecursion
def _quadratic_coeff(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_quadratic_coeff'
    module_type_store = module_type_store.open_function_context('_quadratic_coeff', 242, 0, False)
    
    # Passed parameters checking function
    _quadratic_coeff.stypy_localization = localization
    _quadratic_coeff.stypy_type_of_self = None
    _quadratic_coeff.stypy_type_store = module_type_store
    _quadratic_coeff.stypy_function_name = '_quadratic_coeff'
    _quadratic_coeff.stypy_param_names_list = ['signal']
    _quadratic_coeff.stypy_varargs_param_name = None
    _quadratic_coeff.stypy_kwargs_param_name = None
    _quadratic_coeff.stypy_call_defaults = defaults
    _quadratic_coeff.stypy_call_varargs = varargs
    _quadratic_coeff.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_quadratic_coeff', ['signal'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_quadratic_coeff', localization, ['signal'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_quadratic_coeff(...)' code ##################

    
    # Assigning a BinOp to a Name (line 243):
    
    # Assigning a BinOp to a Name (line 243):
    int_256901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 9), 'int')
    int_256902 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 14), 'int')
    
    # Call to sqrt(...): (line 243)
    # Processing the call arguments (line 243)
    float_256904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 23), 'float')
    # Processing the call keyword arguments (line 243)
    kwargs_256905 = {}
    # Getting the type of 'sqrt' (line 243)
    sqrt_256903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 18), 'sqrt', False)
    # Calling sqrt(args, kwargs) (line 243)
    sqrt_call_result_256906 = invoke(stypy.reporting.localization.Localization(__file__, 243, 18), sqrt_256903, *[float_256904], **kwargs_256905)
    
    # Applying the binary operator '*' (line 243)
    result_mul_256907 = python_operator(stypy.reporting.localization.Localization(__file__, 243, 14), '*', int_256902, sqrt_call_result_256906)
    
    # Applying the binary operator '+' (line 243)
    result_add_256908 = python_operator(stypy.reporting.localization.Localization(__file__, 243, 9), '+', int_256901, result_mul_256907)
    
    # Assigning a type to the variable 'zi' (line 243)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 4), 'zi', result_add_256908)
    
    # Assigning a Call to a Name (line 244):
    
    # Assigning a Call to a Name (line 244):
    
    # Call to len(...): (line 244)
    # Processing the call arguments (line 244)
    # Getting the type of 'signal' (line 244)
    signal_256910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 12), 'signal', False)
    # Processing the call keyword arguments (line 244)
    kwargs_256911 = {}
    # Getting the type of 'len' (line 244)
    len_256909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 8), 'len', False)
    # Calling len(args, kwargs) (line 244)
    len_call_result_256912 = invoke(stypy.reporting.localization.Localization(__file__, 244, 8), len_256909, *[signal_256910], **kwargs_256911)
    
    # Assigning a type to the variable 'K' (line 244)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 4), 'K', len_call_result_256912)
    
    # Assigning a Call to a Name (line 245):
    
    # Assigning a Call to a Name (line 245):
    
    # Call to zeros(...): (line 245)
    # Processing the call arguments (line 245)
    
    # Obtaining an instance of the builtin type 'tuple' (line 245)
    tuple_256914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 19), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 245)
    # Adding element type (line 245)
    # Getting the type of 'K' (line 245)
    K_256915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 19), 'K', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 245, 19), tuple_256914, K_256915)
    
    # Getting the type of 'signal' (line 245)
    signal_256916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 24), 'signal', False)
    # Obtaining the member 'dtype' of a type (line 245)
    dtype_256917 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 24), signal_256916, 'dtype')
    # Obtaining the member 'char' of a type (line 245)
    char_256918 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 24), dtype_256917, 'char')
    # Processing the call keyword arguments (line 245)
    kwargs_256919 = {}
    # Getting the type of 'zeros' (line 245)
    zeros_256913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 12), 'zeros', False)
    # Calling zeros(args, kwargs) (line 245)
    zeros_call_result_256920 = invoke(stypy.reporting.localization.Localization(__file__, 245, 12), zeros_256913, *[tuple_256914, char_256918], **kwargs_256919)
    
    # Assigning a type to the variable 'yplus' (line 245)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 4), 'yplus', zeros_call_result_256920)
    
    # Assigning a BinOp to a Name (line 246):
    
    # Assigning a BinOp to a Name (line 246):
    # Getting the type of 'zi' (line 246)
    zi_256921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 13), 'zi')
    
    # Call to arange(...): (line 246)
    # Processing the call arguments (line 246)
    # Getting the type of 'K' (line 246)
    K_256923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 26), 'K', False)
    # Processing the call keyword arguments (line 246)
    kwargs_256924 = {}
    # Getting the type of 'arange' (line 246)
    arange_256922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 19), 'arange', False)
    # Calling arange(args, kwargs) (line 246)
    arange_call_result_256925 = invoke(stypy.reporting.localization.Localization(__file__, 246, 19), arange_256922, *[K_256923], **kwargs_256924)
    
    # Applying the binary operator '**' (line 246)
    result_pow_256926 = python_operator(stypy.reporting.localization.Localization(__file__, 246, 13), '**', zi_256921, arange_call_result_256925)
    
    # Assigning a type to the variable 'powers' (line 246)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 4), 'powers', result_pow_256926)
    
    # Assigning a BinOp to a Subscript (line 247):
    
    # Assigning a BinOp to a Subscript (line 247):
    
    # Obtaining the type of the subscript
    int_256927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 22), 'int')
    # Getting the type of 'signal' (line 247)
    signal_256928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 15), 'signal')
    # Obtaining the member '__getitem__' of a type (line 247)
    getitem___256929 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 15), signal_256928, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 247)
    subscript_call_result_256930 = invoke(stypy.reporting.localization.Localization(__file__, 247, 15), getitem___256929, int_256927)
    
    # Getting the type of 'zi' (line 247)
    zi_256931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 27), 'zi')
    
    # Call to reduce(...): (line 247)
    # Processing the call arguments (line 247)
    # Getting the type of 'powers' (line 247)
    powers_256934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 43), 'powers', False)
    # Getting the type of 'signal' (line 247)
    signal_256935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 52), 'signal', False)
    # Applying the binary operator '*' (line 247)
    result_mul_256936 = python_operator(stypy.reporting.localization.Localization(__file__, 247, 43), '*', powers_256934, signal_256935)
    
    # Processing the call keyword arguments (line 247)
    kwargs_256937 = {}
    # Getting the type of 'add' (line 247)
    add_256932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 32), 'add', False)
    # Obtaining the member 'reduce' of a type (line 247)
    reduce_256933 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 32), add_256932, 'reduce')
    # Calling reduce(args, kwargs) (line 247)
    reduce_call_result_256938 = invoke(stypy.reporting.localization.Localization(__file__, 247, 32), reduce_256933, *[result_mul_256936], **kwargs_256937)
    
    # Applying the binary operator '*' (line 247)
    result_mul_256939 = python_operator(stypy.reporting.localization.Localization(__file__, 247, 27), '*', zi_256931, reduce_call_result_256938)
    
    # Applying the binary operator '+' (line 247)
    result_add_256940 = python_operator(stypy.reporting.localization.Localization(__file__, 247, 15), '+', subscript_call_result_256930, result_mul_256939)
    
    # Getting the type of 'yplus' (line 247)
    yplus_256941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 4), 'yplus')
    int_256942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 10), 'int')
    # Storing an element on a container (line 247)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 247, 4), yplus_256941, (int_256942, result_add_256940))
    
    
    # Call to range(...): (line 248)
    # Processing the call arguments (line 248)
    int_256944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 19), 'int')
    # Getting the type of 'K' (line 248)
    K_256945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 22), 'K', False)
    # Processing the call keyword arguments (line 248)
    kwargs_256946 = {}
    # Getting the type of 'range' (line 248)
    range_256943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 13), 'range', False)
    # Calling range(args, kwargs) (line 248)
    range_call_result_256947 = invoke(stypy.reporting.localization.Localization(__file__, 248, 13), range_256943, *[int_256944, K_256945], **kwargs_256946)
    
    # Testing the type of a for loop iterable (line 248)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 248, 4), range_call_result_256947)
    # Getting the type of the for loop variable (line 248)
    for_loop_var_256948 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 248, 4), range_call_result_256947)
    # Assigning a type to the variable 'k' (line 248)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 4), 'k', for_loop_var_256948)
    # SSA begins for a for statement (line 248)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a BinOp to a Subscript (line 249):
    
    # Assigning a BinOp to a Subscript (line 249):
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 249)
    k_256949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 26), 'k')
    # Getting the type of 'signal' (line 249)
    signal_256950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 19), 'signal')
    # Obtaining the member '__getitem__' of a type (line 249)
    getitem___256951 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 19), signal_256950, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 249)
    subscript_call_result_256952 = invoke(stypy.reporting.localization.Localization(__file__, 249, 19), getitem___256951, k_256949)
    
    # Getting the type of 'zi' (line 249)
    zi_256953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 31), 'zi')
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 249)
    k_256954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 42), 'k')
    int_256955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 46), 'int')
    # Applying the binary operator '-' (line 249)
    result_sub_256956 = python_operator(stypy.reporting.localization.Localization(__file__, 249, 42), '-', k_256954, int_256955)
    
    # Getting the type of 'yplus' (line 249)
    yplus_256957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 36), 'yplus')
    # Obtaining the member '__getitem__' of a type (line 249)
    getitem___256958 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 36), yplus_256957, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 249)
    subscript_call_result_256959 = invoke(stypy.reporting.localization.Localization(__file__, 249, 36), getitem___256958, result_sub_256956)
    
    # Applying the binary operator '*' (line 249)
    result_mul_256960 = python_operator(stypy.reporting.localization.Localization(__file__, 249, 31), '*', zi_256953, subscript_call_result_256959)
    
    # Applying the binary operator '+' (line 249)
    result_add_256961 = python_operator(stypy.reporting.localization.Localization(__file__, 249, 19), '+', subscript_call_result_256952, result_mul_256960)
    
    # Getting the type of 'yplus' (line 249)
    yplus_256962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 8), 'yplus')
    # Getting the type of 'k' (line 249)
    k_256963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 14), 'k')
    # Storing an element on a container (line 249)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 249, 8), yplus_256962, (k_256963, result_add_256961))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 250):
    
    # Assigning a Call to a Name (line 250):
    
    # Call to zeros(...): (line 250)
    # Processing the call arguments (line 250)
    
    # Obtaining an instance of the builtin type 'tuple' (line 250)
    tuple_256965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 20), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 250)
    # Adding element type (line 250)
    # Getting the type of 'K' (line 250)
    K_256966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 20), 'K', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 250, 20), tuple_256965, K_256966)
    
    # Getting the type of 'signal' (line 250)
    signal_256967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 25), 'signal', False)
    # Obtaining the member 'dtype' of a type (line 250)
    dtype_256968 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 25), signal_256967, 'dtype')
    # Obtaining the member 'char' of a type (line 250)
    char_256969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 25), dtype_256968, 'char')
    # Processing the call keyword arguments (line 250)
    kwargs_256970 = {}
    # Getting the type of 'zeros' (line 250)
    zeros_256964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 13), 'zeros', False)
    # Calling zeros(args, kwargs) (line 250)
    zeros_call_result_256971 = invoke(stypy.reporting.localization.Localization(__file__, 250, 13), zeros_256964, *[tuple_256965, char_256969], **kwargs_256970)
    
    # Assigning a type to the variable 'output' (line 250)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 4), 'output', zeros_call_result_256971)
    
    # Assigning a BinOp to a Subscript (line 251):
    
    # Assigning a BinOp to a Subscript (line 251):
    # Getting the type of 'zi' (line 251)
    zi_256972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 20), 'zi')
    # Getting the type of 'zi' (line 251)
    zi_256973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 26), 'zi')
    int_256974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 31), 'int')
    # Applying the binary operator '-' (line 251)
    result_sub_256975 = python_operator(stypy.reporting.localization.Localization(__file__, 251, 26), '-', zi_256973, int_256974)
    
    # Applying the binary operator 'div' (line 251)
    result_div_256976 = python_operator(stypy.reporting.localization.Localization(__file__, 251, 20), 'div', zi_256972, result_sub_256975)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'K' (line 251)
    K_256977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 42), 'K')
    int_256978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 46), 'int')
    # Applying the binary operator '-' (line 251)
    result_sub_256979 = python_operator(stypy.reporting.localization.Localization(__file__, 251, 42), '-', K_256977, int_256978)
    
    # Getting the type of 'yplus' (line 251)
    yplus_256980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 36), 'yplus')
    # Obtaining the member '__getitem__' of a type (line 251)
    getitem___256981 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 36), yplus_256980, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 251)
    subscript_call_result_256982 = invoke(stypy.reporting.localization.Localization(__file__, 251, 36), getitem___256981, result_sub_256979)
    
    # Applying the binary operator '*' (line 251)
    result_mul_256983 = python_operator(stypy.reporting.localization.Localization(__file__, 251, 34), '*', result_div_256976, subscript_call_result_256982)
    
    # Getting the type of 'output' (line 251)
    output_256984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 4), 'output')
    # Getting the type of 'K' (line 251)
    K_256985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 11), 'K')
    int_256986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 15), 'int')
    # Applying the binary operator '-' (line 251)
    result_sub_256987 = python_operator(stypy.reporting.localization.Localization(__file__, 251, 11), '-', K_256985, int_256986)
    
    # Storing an element on a container (line 251)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 4), output_256984, (result_sub_256987, result_mul_256983))
    
    
    # Call to range(...): (line 252)
    # Processing the call arguments (line 252)
    # Getting the type of 'K' (line 252)
    K_256989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 19), 'K', False)
    int_256990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 23), 'int')
    # Applying the binary operator '-' (line 252)
    result_sub_256991 = python_operator(stypy.reporting.localization.Localization(__file__, 252, 19), '-', K_256989, int_256990)
    
    int_256992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 26), 'int')
    int_256993 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 30), 'int')
    # Processing the call keyword arguments (line 252)
    kwargs_256994 = {}
    # Getting the type of 'range' (line 252)
    range_256988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 13), 'range', False)
    # Calling range(args, kwargs) (line 252)
    range_call_result_256995 = invoke(stypy.reporting.localization.Localization(__file__, 252, 13), range_256988, *[result_sub_256991, int_256992, int_256993], **kwargs_256994)
    
    # Testing the type of a for loop iterable (line 252)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 252, 4), range_call_result_256995)
    # Getting the type of the for loop variable (line 252)
    for_loop_var_256996 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 252, 4), range_call_result_256995)
    # Assigning a type to the variable 'k' (line 252)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 4), 'k', for_loop_var_256996)
    # SSA begins for a for statement (line 252)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a BinOp to a Subscript (line 253):
    
    # Assigning a BinOp to a Subscript (line 253):
    # Getting the type of 'zi' (line 253)
    zi_256997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 20), 'zi')
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 253)
    k_256998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 33), 'k')
    int_256999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 37), 'int')
    # Applying the binary operator '+' (line 253)
    result_add_257000 = python_operator(stypy.reporting.localization.Localization(__file__, 253, 33), '+', k_256998, int_256999)
    
    # Getting the type of 'output' (line 253)
    output_257001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 26), 'output')
    # Obtaining the member '__getitem__' of a type (line 253)
    getitem___257002 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 26), output_257001, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 253)
    subscript_call_result_257003 = invoke(stypy.reporting.localization.Localization(__file__, 253, 26), getitem___257002, result_add_257000)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 253)
    k_257004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 48), 'k')
    # Getting the type of 'yplus' (line 253)
    yplus_257005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 42), 'yplus')
    # Obtaining the member '__getitem__' of a type (line 253)
    getitem___257006 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 42), yplus_257005, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 253)
    subscript_call_result_257007 = invoke(stypy.reporting.localization.Localization(__file__, 253, 42), getitem___257006, k_257004)
    
    # Applying the binary operator '-' (line 253)
    result_sub_257008 = python_operator(stypy.reporting.localization.Localization(__file__, 253, 26), '-', subscript_call_result_257003, subscript_call_result_257007)
    
    # Applying the binary operator '*' (line 253)
    result_mul_257009 = python_operator(stypy.reporting.localization.Localization(__file__, 253, 20), '*', zi_256997, result_sub_257008)
    
    # Getting the type of 'output' (line 253)
    output_257010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 8), 'output')
    # Getting the type of 'k' (line 253)
    k_257011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 15), 'k')
    # Storing an element on a container (line 253)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 253, 8), output_257010, (k_257011, result_mul_257009))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'output' (line 254)
    output_257012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 11), 'output')
    float_257013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 20), 'float')
    # Applying the binary operator '*' (line 254)
    result_mul_257014 = python_operator(stypy.reporting.localization.Localization(__file__, 254, 11), '*', output_257012, float_257013)
    
    # Assigning a type to the variable 'stypy_return_type' (line 254)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 4), 'stypy_return_type', result_mul_257014)
    
    # ################# End of '_quadratic_coeff(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_quadratic_coeff' in the type store
    # Getting the type of 'stypy_return_type' (line 242)
    stypy_return_type_257015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_257015)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_quadratic_coeff'
    return stypy_return_type_257015

# Assigning a type to the variable '_quadratic_coeff' (line 242)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 0), '_quadratic_coeff', _quadratic_coeff)

@norecursion
def cspline1d(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    float_257016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 27), 'float')
    defaults = [float_257016]
    # Create a new context for function 'cspline1d'
    module_type_store = module_type_store.open_function_context('cspline1d', 257, 0, False)
    
    # Passed parameters checking function
    cspline1d.stypy_localization = localization
    cspline1d.stypy_type_of_self = None
    cspline1d.stypy_type_store = module_type_store
    cspline1d.stypy_function_name = 'cspline1d'
    cspline1d.stypy_param_names_list = ['signal', 'lamb']
    cspline1d.stypy_varargs_param_name = None
    cspline1d.stypy_kwargs_param_name = None
    cspline1d.stypy_call_defaults = defaults
    cspline1d.stypy_call_varargs = varargs
    cspline1d.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'cspline1d', ['signal', 'lamb'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'cspline1d', localization, ['signal', 'lamb'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'cspline1d(...)' code ##################

    str_257017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, (-1)), 'str', '\n    Compute cubic spline coefficients for rank-1 array.\n\n    Find the cubic spline coefficients for a 1-D signal assuming\n    mirror-symmetric boundary conditions.   To obtain the signal back from the\n    spline representation mirror-symmetric-convolve these coefficients with a\n    length 3 FIR window [1.0, 4.0, 1.0]/ 6.0 .\n\n    Parameters\n    ----------\n    signal : ndarray\n        A rank-1 array representing samples of a signal.\n    lamb : float, optional\n        Smoothing coefficient, default is 0.0.\n\n    Returns\n    -------\n    c : ndarray\n        Cubic spline coefficients.\n\n    ')
    
    
    # Getting the type of 'lamb' (line 279)
    lamb_257018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 7), 'lamb')
    float_257019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 15), 'float')
    # Applying the binary operator '!=' (line 279)
    result_ne_257020 = python_operator(stypy.reporting.localization.Localization(__file__, 279, 7), '!=', lamb_257018, float_257019)
    
    # Testing the type of an if condition (line 279)
    if_condition_257021 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 279, 4), result_ne_257020)
    # Assigning a type to the variable 'if_condition_257021' (line 279)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 4), 'if_condition_257021', if_condition_257021)
    # SSA begins for if statement (line 279)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to _cubic_smooth_coeff(...): (line 280)
    # Processing the call arguments (line 280)
    # Getting the type of 'signal' (line 280)
    signal_257023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 35), 'signal', False)
    # Getting the type of 'lamb' (line 280)
    lamb_257024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 43), 'lamb', False)
    # Processing the call keyword arguments (line 280)
    kwargs_257025 = {}
    # Getting the type of '_cubic_smooth_coeff' (line 280)
    _cubic_smooth_coeff_257022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 15), '_cubic_smooth_coeff', False)
    # Calling _cubic_smooth_coeff(args, kwargs) (line 280)
    _cubic_smooth_coeff_call_result_257026 = invoke(stypy.reporting.localization.Localization(__file__, 280, 15), _cubic_smooth_coeff_257022, *[signal_257023, lamb_257024], **kwargs_257025)
    
    # Assigning a type to the variable 'stypy_return_type' (line 280)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 8), 'stypy_return_type', _cubic_smooth_coeff_call_result_257026)
    # SSA branch for the else part of an if statement (line 279)
    module_type_store.open_ssa_branch('else')
    
    # Call to _cubic_coeff(...): (line 282)
    # Processing the call arguments (line 282)
    # Getting the type of 'signal' (line 282)
    signal_257028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 28), 'signal', False)
    # Processing the call keyword arguments (line 282)
    kwargs_257029 = {}
    # Getting the type of '_cubic_coeff' (line 282)
    _cubic_coeff_257027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 15), '_cubic_coeff', False)
    # Calling _cubic_coeff(args, kwargs) (line 282)
    _cubic_coeff_call_result_257030 = invoke(stypy.reporting.localization.Localization(__file__, 282, 15), _cubic_coeff_257027, *[signal_257028], **kwargs_257029)
    
    # Assigning a type to the variable 'stypy_return_type' (line 282)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 8), 'stypy_return_type', _cubic_coeff_call_result_257030)
    # SSA join for if statement (line 279)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'cspline1d(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'cspline1d' in the type store
    # Getting the type of 'stypy_return_type' (line 257)
    stypy_return_type_257031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_257031)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'cspline1d'
    return stypy_return_type_257031

# Assigning a type to the variable 'cspline1d' (line 257)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 0), 'cspline1d', cspline1d)

@norecursion
def qspline1d(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    float_257032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 27), 'float')
    defaults = [float_257032]
    # Create a new context for function 'qspline1d'
    module_type_store = module_type_store.open_function_context('qspline1d', 285, 0, False)
    
    # Passed parameters checking function
    qspline1d.stypy_localization = localization
    qspline1d.stypy_type_of_self = None
    qspline1d.stypy_type_store = module_type_store
    qspline1d.stypy_function_name = 'qspline1d'
    qspline1d.stypy_param_names_list = ['signal', 'lamb']
    qspline1d.stypy_varargs_param_name = None
    qspline1d.stypy_kwargs_param_name = None
    qspline1d.stypy_call_defaults = defaults
    qspline1d.stypy_call_varargs = varargs
    qspline1d.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'qspline1d', ['signal', 'lamb'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'qspline1d', localization, ['signal', 'lamb'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'qspline1d(...)' code ##################

    str_257033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, (-1)), 'str', 'Compute quadratic spline coefficients for rank-1 array.\n\n    Find the quadratic spline coefficients for a 1-D signal assuming\n    mirror-symmetric boundary conditions.   To obtain the signal back from the\n    spline representation mirror-symmetric-convolve these coefficients with a\n    length 3 FIR window [1.0, 6.0, 1.0]/ 8.0 .\n\n    Parameters\n    ----------\n    signal : ndarray\n        A rank-1 array representing samples of a signal.\n    lamb : float, optional\n        Smoothing coefficient (must be zero for now).\n\n    Returns\n    -------\n    c : ndarray\n        Cubic spline coefficients.\n\n    ')
    
    
    # Getting the type of 'lamb' (line 306)
    lamb_257034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 7), 'lamb')
    float_257035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 15), 'float')
    # Applying the binary operator '!=' (line 306)
    result_ne_257036 = python_operator(stypy.reporting.localization.Localization(__file__, 306, 7), '!=', lamb_257034, float_257035)
    
    # Testing the type of an if condition (line 306)
    if_condition_257037 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 306, 4), result_ne_257036)
    # Assigning a type to the variable 'if_condition_257037' (line 306)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 4), 'if_condition_257037', if_condition_257037)
    # SSA begins for if statement (line 306)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 307)
    # Processing the call arguments (line 307)
    str_257039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 25), 'str', 'Smoothing quadratic splines not supported yet.')
    # Processing the call keyword arguments (line 307)
    kwargs_257040 = {}
    # Getting the type of 'ValueError' (line 307)
    ValueError_257038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 307)
    ValueError_call_result_257041 = invoke(stypy.reporting.localization.Localization(__file__, 307, 14), ValueError_257038, *[str_257039], **kwargs_257040)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 307, 8), ValueError_call_result_257041, 'raise parameter', BaseException)
    # SSA branch for the else part of an if statement (line 306)
    module_type_store.open_ssa_branch('else')
    
    # Call to _quadratic_coeff(...): (line 309)
    # Processing the call arguments (line 309)
    # Getting the type of 'signal' (line 309)
    signal_257043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 32), 'signal', False)
    # Processing the call keyword arguments (line 309)
    kwargs_257044 = {}
    # Getting the type of '_quadratic_coeff' (line 309)
    _quadratic_coeff_257042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 15), '_quadratic_coeff', False)
    # Calling _quadratic_coeff(args, kwargs) (line 309)
    _quadratic_coeff_call_result_257045 = invoke(stypy.reporting.localization.Localization(__file__, 309, 15), _quadratic_coeff_257042, *[signal_257043], **kwargs_257044)
    
    # Assigning a type to the variable 'stypy_return_type' (line 309)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 8), 'stypy_return_type', _quadratic_coeff_call_result_257045)
    # SSA join for if statement (line 306)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'qspline1d(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'qspline1d' in the type store
    # Getting the type of 'stypy_return_type' (line 285)
    stypy_return_type_257046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_257046)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'qspline1d'
    return stypy_return_type_257046

# Assigning a type to the variable 'qspline1d' (line 285)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 0), 'qspline1d', qspline1d)

@norecursion
def cspline1d_eval(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    float_257047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 32), 'float')
    int_257048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 40), 'int')
    defaults = [float_257047, int_257048]
    # Create a new context for function 'cspline1d_eval'
    module_type_store = module_type_store.open_function_context('cspline1d_eval', 312, 0, False)
    
    # Passed parameters checking function
    cspline1d_eval.stypy_localization = localization
    cspline1d_eval.stypy_type_of_self = None
    cspline1d_eval.stypy_type_store = module_type_store
    cspline1d_eval.stypy_function_name = 'cspline1d_eval'
    cspline1d_eval.stypy_param_names_list = ['cj', 'newx', 'dx', 'x0']
    cspline1d_eval.stypy_varargs_param_name = None
    cspline1d_eval.stypy_kwargs_param_name = None
    cspline1d_eval.stypy_call_defaults = defaults
    cspline1d_eval.stypy_call_varargs = varargs
    cspline1d_eval.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'cspline1d_eval', ['cj', 'newx', 'dx', 'x0'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'cspline1d_eval', localization, ['cj', 'newx', 'dx', 'x0'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'cspline1d_eval(...)' code ##################

    str_257049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, (-1)), 'str', 'Evaluate a spline at the new set of points.\n\n    `dx` is the old sample-spacing while `x0` was the old origin.  In\n    other-words the old-sample points (knot-points) for which the `cj`\n    represent spline coefficients were at equally-spaced points of:\n\n      oldx = x0 + j*dx  j=0...N-1, with N=len(cj)\n\n    Edges are handled using mirror-symmetric boundary conditions.\n\n    ')
    
    # Assigning a BinOp to a Name (line 324):
    
    # Assigning a BinOp to a Name (line 324):
    
    # Call to asarray(...): (line 324)
    # Processing the call arguments (line 324)
    # Getting the type of 'newx' (line 324)
    newx_257051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 20), 'newx', False)
    # Processing the call keyword arguments (line 324)
    kwargs_257052 = {}
    # Getting the type of 'asarray' (line 324)
    asarray_257050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 12), 'asarray', False)
    # Calling asarray(args, kwargs) (line 324)
    asarray_call_result_257053 = invoke(stypy.reporting.localization.Localization(__file__, 324, 12), asarray_257050, *[newx_257051], **kwargs_257052)
    
    # Getting the type of 'x0' (line 324)
    x0_257054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 28), 'x0')
    # Applying the binary operator '-' (line 324)
    result_sub_257055 = python_operator(stypy.reporting.localization.Localization(__file__, 324, 12), '-', asarray_call_result_257053, x0_257054)
    
    
    # Call to float(...): (line 324)
    # Processing the call arguments (line 324)
    # Getting the type of 'dx' (line 324)
    dx_257057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 40), 'dx', False)
    # Processing the call keyword arguments (line 324)
    kwargs_257058 = {}
    # Getting the type of 'float' (line 324)
    float_257056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 34), 'float', False)
    # Calling float(args, kwargs) (line 324)
    float_call_result_257059 = invoke(stypy.reporting.localization.Localization(__file__, 324, 34), float_257056, *[dx_257057], **kwargs_257058)
    
    # Applying the binary operator 'div' (line 324)
    result_div_257060 = python_operator(stypy.reporting.localization.Localization(__file__, 324, 11), 'div', result_sub_257055, float_call_result_257059)
    
    # Assigning a type to the variable 'newx' (line 324)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 4), 'newx', result_div_257060)
    
    # Assigning a Call to a Name (line 325):
    
    # Assigning a Call to a Name (line 325):
    
    # Call to zeros_like(...): (line 325)
    # Processing the call arguments (line 325)
    # Getting the type of 'newx' (line 325)
    newx_257062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 21), 'newx', False)
    # Processing the call keyword arguments (line 325)
    # Getting the type of 'cj' (line 325)
    cj_257063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 33), 'cj', False)
    # Obtaining the member 'dtype' of a type (line 325)
    dtype_257064 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 33), cj_257063, 'dtype')
    keyword_257065 = dtype_257064
    kwargs_257066 = {'dtype': keyword_257065}
    # Getting the type of 'zeros_like' (line 325)
    zeros_like_257061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 10), 'zeros_like', False)
    # Calling zeros_like(args, kwargs) (line 325)
    zeros_like_call_result_257067 = invoke(stypy.reporting.localization.Localization(__file__, 325, 10), zeros_like_257061, *[newx_257062], **kwargs_257066)
    
    # Assigning a type to the variable 'res' (line 325)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 4), 'res', zeros_like_call_result_257067)
    
    
    # Getting the type of 'res' (line 326)
    res_257068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 7), 'res')
    # Obtaining the member 'size' of a type (line 326)
    size_257069 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 7), res_257068, 'size')
    int_257070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 19), 'int')
    # Applying the binary operator '==' (line 326)
    result_eq_257071 = python_operator(stypy.reporting.localization.Localization(__file__, 326, 7), '==', size_257069, int_257070)
    
    # Testing the type of an if condition (line 326)
    if_condition_257072 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 326, 4), result_eq_257071)
    # Assigning a type to the variable 'if_condition_257072' (line 326)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 4), 'if_condition_257072', if_condition_257072)
    # SSA begins for if statement (line 326)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'res' (line 327)
    res_257073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 15), 'res')
    # Assigning a type to the variable 'stypy_return_type' (line 327)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 327, 8), 'stypy_return_type', res_257073)
    # SSA join for if statement (line 326)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 328):
    
    # Assigning a Call to a Name (line 328):
    
    # Call to len(...): (line 328)
    # Processing the call arguments (line 328)
    # Getting the type of 'cj' (line 328)
    cj_257075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 12), 'cj', False)
    # Processing the call keyword arguments (line 328)
    kwargs_257076 = {}
    # Getting the type of 'len' (line 328)
    len_257074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 8), 'len', False)
    # Calling len(args, kwargs) (line 328)
    len_call_result_257077 = invoke(stypy.reporting.localization.Localization(__file__, 328, 8), len_257074, *[cj_257075], **kwargs_257076)
    
    # Assigning a type to the variable 'N' (line 328)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 4), 'N', len_call_result_257077)
    
    # Assigning a Compare to a Name (line 329):
    
    # Assigning a Compare to a Name (line 329):
    
    # Getting the type of 'newx' (line 329)
    newx_257078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 12), 'newx')
    int_257079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 19), 'int')
    # Applying the binary operator '<' (line 329)
    result_lt_257080 = python_operator(stypy.reporting.localization.Localization(__file__, 329, 12), '<', newx_257078, int_257079)
    
    # Assigning a type to the variable 'cond1' (line 329)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 4), 'cond1', result_lt_257080)
    
    # Assigning a Compare to a Name (line 330):
    
    # Assigning a Compare to a Name (line 330):
    
    # Getting the type of 'newx' (line 330)
    newx_257081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 12), 'newx')
    # Getting the type of 'N' (line 330)
    N_257082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 20), 'N')
    int_257083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 24), 'int')
    # Applying the binary operator '-' (line 330)
    result_sub_257084 = python_operator(stypy.reporting.localization.Localization(__file__, 330, 20), '-', N_257082, int_257083)
    
    # Applying the binary operator '>' (line 330)
    result_gt_257085 = python_operator(stypy.reporting.localization.Localization(__file__, 330, 12), '>', newx_257081, result_sub_257084)
    
    # Assigning a type to the variable 'cond2' (line 330)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 4), 'cond2', result_gt_257085)
    
    # Assigning a UnaryOp to a Name (line 331):
    
    # Assigning a UnaryOp to a Name (line 331):
    
    # Getting the type of 'cond1' (line 331)
    cond1_257086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 14), 'cond1')
    # Getting the type of 'cond2' (line 331)
    cond2_257087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 22), 'cond2')
    # Applying the binary operator '|' (line 331)
    result_or__257088 = python_operator(stypy.reporting.localization.Localization(__file__, 331, 14), '|', cond1_257086, cond2_257087)
    
    # Applying the '~' unary operator (line 331)
    result_inv_257089 = python_operator(stypy.reporting.localization.Localization(__file__, 331, 12), '~', result_or__257088)
    
    # Assigning a type to the variable 'cond3' (line 331)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 4), 'cond3', result_inv_257089)
    
    # Assigning a Call to a Subscript (line 333):
    
    # Assigning a Call to a Subscript (line 333):
    
    # Call to cspline1d_eval(...): (line 333)
    # Processing the call arguments (line 333)
    # Getting the type of 'cj' (line 333)
    cj_257091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 32), 'cj', False)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'cond1' (line 333)
    cond1_257092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 42), 'cond1', False)
    # Getting the type of 'newx' (line 333)
    newx_257093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 37), 'newx', False)
    # Obtaining the member '__getitem__' of a type (line 333)
    getitem___257094 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 37), newx_257093, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 333)
    subscript_call_result_257095 = invoke(stypy.reporting.localization.Localization(__file__, 333, 37), getitem___257094, cond1_257092)
    
    # Applying the 'usub' unary operator (line 333)
    result___neg___257096 = python_operator(stypy.reporting.localization.Localization(__file__, 333, 36), 'usub', subscript_call_result_257095)
    
    # Processing the call keyword arguments (line 333)
    kwargs_257097 = {}
    # Getting the type of 'cspline1d_eval' (line 333)
    cspline1d_eval_257090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 17), 'cspline1d_eval', False)
    # Calling cspline1d_eval(args, kwargs) (line 333)
    cspline1d_eval_call_result_257098 = invoke(stypy.reporting.localization.Localization(__file__, 333, 17), cspline1d_eval_257090, *[cj_257091, result___neg___257096], **kwargs_257097)
    
    # Getting the type of 'res' (line 333)
    res_257099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 4), 'res')
    # Getting the type of 'cond1' (line 333)
    cond1_257100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 8), 'cond1')
    # Storing an element on a container (line 333)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 333, 4), res_257099, (cond1_257100, cspline1d_eval_call_result_257098))
    
    # Assigning a Call to a Subscript (line 334):
    
    # Assigning a Call to a Subscript (line 334):
    
    # Call to cspline1d_eval(...): (line 334)
    # Processing the call arguments (line 334)
    # Getting the type of 'cj' (line 334)
    cj_257102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 32), 'cj', False)
    int_257103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 36), 'int')
    # Getting the type of 'N' (line 334)
    N_257104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 41), 'N', False)
    int_257105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 45), 'int')
    # Applying the binary operator '-' (line 334)
    result_sub_257106 = python_operator(stypy.reporting.localization.Localization(__file__, 334, 41), '-', N_257104, int_257105)
    
    # Applying the binary operator '*' (line 334)
    result_mul_257107 = python_operator(stypy.reporting.localization.Localization(__file__, 334, 36), '*', int_257103, result_sub_257106)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'cond2' (line 334)
    cond2_257108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 55), 'cond2', False)
    # Getting the type of 'newx' (line 334)
    newx_257109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 50), 'newx', False)
    # Obtaining the member '__getitem__' of a type (line 334)
    getitem___257110 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 50), newx_257109, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 334)
    subscript_call_result_257111 = invoke(stypy.reporting.localization.Localization(__file__, 334, 50), getitem___257110, cond2_257108)
    
    # Applying the binary operator '-' (line 334)
    result_sub_257112 = python_operator(stypy.reporting.localization.Localization(__file__, 334, 36), '-', result_mul_257107, subscript_call_result_257111)
    
    # Processing the call keyword arguments (line 334)
    kwargs_257113 = {}
    # Getting the type of 'cspline1d_eval' (line 334)
    cspline1d_eval_257101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 17), 'cspline1d_eval', False)
    # Calling cspline1d_eval(args, kwargs) (line 334)
    cspline1d_eval_call_result_257114 = invoke(stypy.reporting.localization.Localization(__file__, 334, 17), cspline1d_eval_257101, *[cj_257102, result_sub_257112], **kwargs_257113)
    
    # Getting the type of 'res' (line 334)
    res_257115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 4), 'res')
    # Getting the type of 'cond2' (line 334)
    cond2_257116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 8), 'cond2')
    # Storing an element on a container (line 334)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 334, 4), res_257115, (cond2_257116, cspline1d_eval_call_result_257114))
    
    # Assigning a Subscript to a Name (line 335):
    
    # Assigning a Subscript to a Name (line 335):
    
    # Obtaining the type of the subscript
    # Getting the type of 'cond3' (line 335)
    cond3_257117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 16), 'cond3')
    # Getting the type of 'newx' (line 335)
    newx_257118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 11), 'newx')
    # Obtaining the member '__getitem__' of a type (line 335)
    getitem___257119 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 11), newx_257118, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 335)
    subscript_call_result_257120 = invoke(stypy.reporting.localization.Localization(__file__, 335, 11), getitem___257119, cond3_257117)
    
    # Assigning a type to the variable 'newx' (line 335)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 4), 'newx', subscript_call_result_257120)
    
    
    # Getting the type of 'newx' (line 336)
    newx_257121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 7), 'newx')
    # Obtaining the member 'size' of a type (line 336)
    size_257122 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 7), newx_257121, 'size')
    int_257123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 20), 'int')
    # Applying the binary operator '==' (line 336)
    result_eq_257124 = python_operator(stypy.reporting.localization.Localization(__file__, 336, 7), '==', size_257122, int_257123)
    
    # Testing the type of an if condition (line 336)
    if_condition_257125 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 336, 4), result_eq_257124)
    # Assigning a type to the variable 'if_condition_257125' (line 336)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 4), 'if_condition_257125', if_condition_257125)
    # SSA begins for if statement (line 336)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'res' (line 337)
    res_257126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 15), 'res')
    # Assigning a type to the variable 'stypy_return_type' (line 337)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 337, 8), 'stypy_return_type', res_257126)
    # SSA join for if statement (line 336)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 338):
    
    # Assigning a Call to a Name (line 338):
    
    # Call to zeros_like(...): (line 338)
    # Processing the call arguments (line 338)
    # Getting the type of 'newx' (line 338)
    newx_257128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 24), 'newx', False)
    # Processing the call keyword arguments (line 338)
    # Getting the type of 'cj' (line 338)
    cj_257129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 36), 'cj', False)
    # Obtaining the member 'dtype' of a type (line 338)
    dtype_257130 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 36), cj_257129, 'dtype')
    keyword_257131 = dtype_257130
    kwargs_257132 = {'dtype': keyword_257131}
    # Getting the type of 'zeros_like' (line 338)
    zeros_like_257127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 13), 'zeros_like', False)
    # Calling zeros_like(args, kwargs) (line 338)
    zeros_like_call_result_257133 = invoke(stypy.reporting.localization.Localization(__file__, 338, 13), zeros_like_257127, *[newx_257128], **kwargs_257132)
    
    # Assigning a type to the variable 'result' (line 338)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 338, 4), 'result', zeros_like_call_result_257133)
    
    # Assigning a BinOp to a Name (line 339):
    
    # Assigning a BinOp to a Name (line 339):
    
    # Call to astype(...): (line 339)
    # Processing the call arguments (line 339)
    # Getting the type of 'int' (line 339)
    int_257141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 36), 'int', False)
    # Processing the call keyword arguments (line 339)
    kwargs_257142 = {}
    
    # Call to floor(...): (line 339)
    # Processing the call arguments (line 339)
    # Getting the type of 'newx' (line 339)
    newx_257135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 19), 'newx', False)
    int_257136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 26), 'int')
    # Applying the binary operator '-' (line 339)
    result_sub_257137 = python_operator(stypy.reporting.localization.Localization(__file__, 339, 19), '-', newx_257135, int_257136)
    
    # Processing the call keyword arguments (line 339)
    kwargs_257138 = {}
    # Getting the type of 'floor' (line 339)
    floor_257134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 13), 'floor', False)
    # Calling floor(args, kwargs) (line 339)
    floor_call_result_257139 = invoke(stypy.reporting.localization.Localization(__file__, 339, 13), floor_257134, *[result_sub_257137], **kwargs_257138)
    
    # Obtaining the member 'astype' of a type (line 339)
    astype_257140 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 339, 13), floor_call_result_257139, 'astype')
    # Calling astype(args, kwargs) (line 339)
    astype_call_result_257143 = invoke(stypy.reporting.localization.Localization(__file__, 339, 13), astype_257140, *[int_257141], **kwargs_257142)
    
    int_257144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 43), 'int')
    # Applying the binary operator '+' (line 339)
    result_add_257145 = python_operator(stypy.reporting.localization.Localization(__file__, 339, 13), '+', astype_call_result_257143, int_257144)
    
    # Assigning a type to the variable 'jlower' (line 339)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 339, 4), 'jlower', result_add_257145)
    
    
    # Call to range(...): (line 340)
    # Processing the call arguments (line 340)
    int_257147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 19), 'int')
    # Processing the call keyword arguments (line 340)
    kwargs_257148 = {}
    # Getting the type of 'range' (line 340)
    range_257146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 13), 'range', False)
    # Calling range(args, kwargs) (line 340)
    range_call_result_257149 = invoke(stypy.reporting.localization.Localization(__file__, 340, 13), range_257146, *[int_257147], **kwargs_257148)
    
    # Testing the type of a for loop iterable (line 340)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 340, 4), range_call_result_257149)
    # Getting the type of the for loop variable (line 340)
    for_loop_var_257150 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 340, 4), range_call_result_257149)
    # Assigning a type to the variable 'i' (line 340)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 4), 'i', for_loop_var_257150)
    # SSA begins for a for statement (line 340)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a BinOp to a Name (line 341):
    
    # Assigning a BinOp to a Name (line 341):
    # Getting the type of 'jlower' (line 341)
    jlower_257151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 16), 'jlower')
    # Getting the type of 'i' (line 341)
    i_257152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 25), 'i')
    # Applying the binary operator '+' (line 341)
    result_add_257153 = python_operator(stypy.reporting.localization.Localization(__file__, 341, 16), '+', jlower_257151, i_257152)
    
    # Assigning a type to the variable 'thisj' (line 341)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 8), 'thisj', result_add_257153)
    
    # Assigning a Call to a Name (line 342):
    
    # Assigning a Call to a Name (line 342):
    
    # Call to clip(...): (line 342)
    # Processing the call arguments (line 342)
    int_257156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 26), 'int')
    # Getting the type of 'N' (line 342)
    N_257157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 29), 'N', False)
    int_257158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 33), 'int')
    # Applying the binary operator '-' (line 342)
    result_sub_257159 = python_operator(stypy.reporting.localization.Localization(__file__, 342, 29), '-', N_257157, int_257158)
    
    # Processing the call keyword arguments (line 342)
    kwargs_257160 = {}
    # Getting the type of 'thisj' (line 342)
    thisj_257154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 15), 'thisj', False)
    # Obtaining the member 'clip' of a type (line 342)
    clip_257155 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 15), thisj_257154, 'clip')
    # Calling clip(args, kwargs) (line 342)
    clip_call_result_257161 = invoke(stypy.reporting.localization.Localization(__file__, 342, 15), clip_257155, *[int_257156, result_sub_257159], **kwargs_257160)
    
    # Assigning a type to the variable 'indj' (line 342)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 8), 'indj', clip_call_result_257161)
    
    # Getting the type of 'result' (line 343)
    result_257162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 8), 'result')
    
    # Obtaining the type of the subscript
    # Getting the type of 'indj' (line 343)
    indj_257163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 21), 'indj')
    # Getting the type of 'cj' (line 343)
    cj_257164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 18), 'cj')
    # Obtaining the member '__getitem__' of a type (line 343)
    getitem___257165 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 18), cj_257164, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 343)
    subscript_call_result_257166 = invoke(stypy.reporting.localization.Localization(__file__, 343, 18), getitem___257165, indj_257163)
    
    
    # Call to cubic(...): (line 343)
    # Processing the call arguments (line 343)
    # Getting the type of 'newx' (line 343)
    newx_257168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 35), 'newx', False)
    # Getting the type of 'thisj' (line 343)
    thisj_257169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 42), 'thisj', False)
    # Applying the binary operator '-' (line 343)
    result_sub_257170 = python_operator(stypy.reporting.localization.Localization(__file__, 343, 35), '-', newx_257168, thisj_257169)
    
    # Processing the call keyword arguments (line 343)
    kwargs_257171 = {}
    # Getting the type of 'cubic' (line 343)
    cubic_257167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 29), 'cubic', False)
    # Calling cubic(args, kwargs) (line 343)
    cubic_call_result_257172 = invoke(stypy.reporting.localization.Localization(__file__, 343, 29), cubic_257167, *[result_sub_257170], **kwargs_257171)
    
    # Applying the binary operator '*' (line 343)
    result_mul_257173 = python_operator(stypy.reporting.localization.Localization(__file__, 343, 18), '*', subscript_call_result_257166, cubic_call_result_257172)
    
    # Applying the binary operator '+=' (line 343)
    result_iadd_257174 = python_operator(stypy.reporting.localization.Localization(__file__, 343, 8), '+=', result_257162, result_mul_257173)
    # Assigning a type to the variable 'result' (line 343)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 8), 'result', result_iadd_257174)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Subscript (line 344):
    
    # Assigning a Name to a Subscript (line 344):
    # Getting the type of 'result' (line 344)
    result_257175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 17), 'result')
    # Getting the type of 'res' (line 344)
    res_257176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 4), 'res')
    # Getting the type of 'cond3' (line 344)
    cond3_257177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 8), 'cond3')
    # Storing an element on a container (line 344)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 344, 4), res_257176, (cond3_257177, result_257175))
    # Getting the type of 'res' (line 345)
    res_257178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 11), 'res')
    # Assigning a type to the variable 'stypy_return_type' (line 345)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 345, 4), 'stypy_return_type', res_257178)
    
    # ################# End of 'cspline1d_eval(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'cspline1d_eval' in the type store
    # Getting the type of 'stypy_return_type' (line 312)
    stypy_return_type_257179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_257179)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'cspline1d_eval'
    return stypy_return_type_257179

# Assigning a type to the variable 'cspline1d_eval' (line 312)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 0), 'cspline1d_eval', cspline1d_eval)

@norecursion
def qspline1d_eval(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    float_257180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 32), 'float')
    int_257181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 40), 'int')
    defaults = [float_257180, int_257181]
    # Create a new context for function 'qspline1d_eval'
    module_type_store = module_type_store.open_function_context('qspline1d_eval', 348, 0, False)
    
    # Passed parameters checking function
    qspline1d_eval.stypy_localization = localization
    qspline1d_eval.stypy_type_of_self = None
    qspline1d_eval.stypy_type_store = module_type_store
    qspline1d_eval.stypy_function_name = 'qspline1d_eval'
    qspline1d_eval.stypy_param_names_list = ['cj', 'newx', 'dx', 'x0']
    qspline1d_eval.stypy_varargs_param_name = None
    qspline1d_eval.stypy_kwargs_param_name = None
    qspline1d_eval.stypy_call_defaults = defaults
    qspline1d_eval.stypy_call_varargs = varargs
    qspline1d_eval.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'qspline1d_eval', ['cj', 'newx', 'dx', 'x0'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'qspline1d_eval', localization, ['cj', 'newx', 'dx', 'x0'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'qspline1d_eval(...)' code ##################

    str_257182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, (-1)), 'str', 'Evaluate a quadratic spline at the new set of points.\n\n    `dx` is the old sample-spacing while `x0` was the old origin.  In\n    other-words the old-sample points (knot-points) for which the `cj`\n    represent spline coefficients were at equally-spaced points of::\n\n      oldx = x0 + j*dx  j=0...N-1, with N=len(cj)\n\n    Edges are handled using mirror-symmetric boundary conditions.\n\n    ')
    
    # Assigning a BinOp to a Name (line 360):
    
    # Assigning a BinOp to a Name (line 360):
    
    # Call to asarray(...): (line 360)
    # Processing the call arguments (line 360)
    # Getting the type of 'newx' (line 360)
    newx_257184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 20), 'newx', False)
    # Processing the call keyword arguments (line 360)
    kwargs_257185 = {}
    # Getting the type of 'asarray' (line 360)
    asarray_257183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 12), 'asarray', False)
    # Calling asarray(args, kwargs) (line 360)
    asarray_call_result_257186 = invoke(stypy.reporting.localization.Localization(__file__, 360, 12), asarray_257183, *[newx_257184], **kwargs_257185)
    
    # Getting the type of 'x0' (line 360)
    x0_257187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 28), 'x0')
    # Applying the binary operator '-' (line 360)
    result_sub_257188 = python_operator(stypy.reporting.localization.Localization(__file__, 360, 12), '-', asarray_call_result_257186, x0_257187)
    
    # Getting the type of 'dx' (line 360)
    dx_257189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 34), 'dx')
    # Applying the binary operator 'div' (line 360)
    result_div_257190 = python_operator(stypy.reporting.localization.Localization(__file__, 360, 11), 'div', result_sub_257188, dx_257189)
    
    # Assigning a type to the variable 'newx' (line 360)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 360, 4), 'newx', result_div_257190)
    
    # Assigning a Call to a Name (line 361):
    
    # Assigning a Call to a Name (line 361):
    
    # Call to zeros_like(...): (line 361)
    # Processing the call arguments (line 361)
    # Getting the type of 'newx' (line 361)
    newx_257192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 21), 'newx', False)
    # Processing the call keyword arguments (line 361)
    kwargs_257193 = {}
    # Getting the type of 'zeros_like' (line 361)
    zeros_like_257191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 10), 'zeros_like', False)
    # Calling zeros_like(args, kwargs) (line 361)
    zeros_like_call_result_257194 = invoke(stypy.reporting.localization.Localization(__file__, 361, 10), zeros_like_257191, *[newx_257192], **kwargs_257193)
    
    # Assigning a type to the variable 'res' (line 361)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 361, 4), 'res', zeros_like_call_result_257194)
    
    
    # Getting the type of 'res' (line 362)
    res_257195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 7), 'res')
    # Obtaining the member 'size' of a type (line 362)
    size_257196 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 7), res_257195, 'size')
    int_257197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 19), 'int')
    # Applying the binary operator '==' (line 362)
    result_eq_257198 = python_operator(stypy.reporting.localization.Localization(__file__, 362, 7), '==', size_257196, int_257197)
    
    # Testing the type of an if condition (line 362)
    if_condition_257199 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 362, 4), result_eq_257198)
    # Assigning a type to the variable 'if_condition_257199' (line 362)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 4), 'if_condition_257199', if_condition_257199)
    # SSA begins for if statement (line 362)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'res' (line 363)
    res_257200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 15), 'res')
    # Assigning a type to the variable 'stypy_return_type' (line 363)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 8), 'stypy_return_type', res_257200)
    # SSA join for if statement (line 362)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 364):
    
    # Assigning a Call to a Name (line 364):
    
    # Call to len(...): (line 364)
    # Processing the call arguments (line 364)
    # Getting the type of 'cj' (line 364)
    cj_257202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 12), 'cj', False)
    # Processing the call keyword arguments (line 364)
    kwargs_257203 = {}
    # Getting the type of 'len' (line 364)
    len_257201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 8), 'len', False)
    # Calling len(args, kwargs) (line 364)
    len_call_result_257204 = invoke(stypy.reporting.localization.Localization(__file__, 364, 8), len_257201, *[cj_257202], **kwargs_257203)
    
    # Assigning a type to the variable 'N' (line 364)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 364, 4), 'N', len_call_result_257204)
    
    # Assigning a Compare to a Name (line 365):
    
    # Assigning a Compare to a Name (line 365):
    
    # Getting the type of 'newx' (line 365)
    newx_257205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 12), 'newx')
    int_257206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 365, 19), 'int')
    # Applying the binary operator '<' (line 365)
    result_lt_257207 = python_operator(stypy.reporting.localization.Localization(__file__, 365, 12), '<', newx_257205, int_257206)
    
    # Assigning a type to the variable 'cond1' (line 365)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 365, 4), 'cond1', result_lt_257207)
    
    # Assigning a Compare to a Name (line 366):
    
    # Assigning a Compare to a Name (line 366):
    
    # Getting the type of 'newx' (line 366)
    newx_257208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 12), 'newx')
    # Getting the type of 'N' (line 366)
    N_257209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 20), 'N')
    int_257210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 24), 'int')
    # Applying the binary operator '-' (line 366)
    result_sub_257211 = python_operator(stypy.reporting.localization.Localization(__file__, 366, 20), '-', N_257209, int_257210)
    
    # Applying the binary operator '>' (line 366)
    result_gt_257212 = python_operator(stypy.reporting.localization.Localization(__file__, 366, 12), '>', newx_257208, result_sub_257211)
    
    # Assigning a type to the variable 'cond2' (line 366)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 366, 4), 'cond2', result_gt_257212)
    
    # Assigning a UnaryOp to a Name (line 367):
    
    # Assigning a UnaryOp to a Name (line 367):
    
    # Getting the type of 'cond1' (line 367)
    cond1_257213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 14), 'cond1')
    # Getting the type of 'cond2' (line 367)
    cond2_257214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 22), 'cond2')
    # Applying the binary operator '|' (line 367)
    result_or__257215 = python_operator(stypy.reporting.localization.Localization(__file__, 367, 14), '|', cond1_257213, cond2_257214)
    
    # Applying the '~' unary operator (line 367)
    result_inv_257216 = python_operator(stypy.reporting.localization.Localization(__file__, 367, 12), '~', result_or__257215)
    
    # Assigning a type to the variable 'cond3' (line 367)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 367, 4), 'cond3', result_inv_257216)
    
    # Assigning a Call to a Subscript (line 369):
    
    # Assigning a Call to a Subscript (line 369):
    
    # Call to qspline1d_eval(...): (line 369)
    # Processing the call arguments (line 369)
    # Getting the type of 'cj' (line 369)
    cj_257218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 32), 'cj', False)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'cond1' (line 369)
    cond1_257219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 42), 'cond1', False)
    # Getting the type of 'newx' (line 369)
    newx_257220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 37), 'newx', False)
    # Obtaining the member '__getitem__' of a type (line 369)
    getitem___257221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 369, 37), newx_257220, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 369)
    subscript_call_result_257222 = invoke(stypy.reporting.localization.Localization(__file__, 369, 37), getitem___257221, cond1_257219)
    
    # Applying the 'usub' unary operator (line 369)
    result___neg___257223 = python_operator(stypy.reporting.localization.Localization(__file__, 369, 36), 'usub', subscript_call_result_257222)
    
    # Processing the call keyword arguments (line 369)
    kwargs_257224 = {}
    # Getting the type of 'qspline1d_eval' (line 369)
    qspline1d_eval_257217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 17), 'qspline1d_eval', False)
    # Calling qspline1d_eval(args, kwargs) (line 369)
    qspline1d_eval_call_result_257225 = invoke(stypy.reporting.localization.Localization(__file__, 369, 17), qspline1d_eval_257217, *[cj_257218, result___neg___257223], **kwargs_257224)
    
    # Getting the type of 'res' (line 369)
    res_257226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 4), 'res')
    # Getting the type of 'cond1' (line 369)
    cond1_257227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 8), 'cond1')
    # Storing an element on a container (line 369)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 369, 4), res_257226, (cond1_257227, qspline1d_eval_call_result_257225))
    
    # Assigning a Call to a Subscript (line 370):
    
    # Assigning a Call to a Subscript (line 370):
    
    # Call to qspline1d_eval(...): (line 370)
    # Processing the call arguments (line 370)
    # Getting the type of 'cj' (line 370)
    cj_257229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 32), 'cj', False)
    int_257230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 36), 'int')
    # Getting the type of 'N' (line 370)
    N_257231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 41), 'N', False)
    int_257232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 45), 'int')
    # Applying the binary operator '-' (line 370)
    result_sub_257233 = python_operator(stypy.reporting.localization.Localization(__file__, 370, 41), '-', N_257231, int_257232)
    
    # Applying the binary operator '*' (line 370)
    result_mul_257234 = python_operator(stypy.reporting.localization.Localization(__file__, 370, 36), '*', int_257230, result_sub_257233)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'cond2' (line 370)
    cond2_257235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 55), 'cond2', False)
    # Getting the type of 'newx' (line 370)
    newx_257236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 50), 'newx', False)
    # Obtaining the member '__getitem__' of a type (line 370)
    getitem___257237 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 50), newx_257236, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 370)
    subscript_call_result_257238 = invoke(stypy.reporting.localization.Localization(__file__, 370, 50), getitem___257237, cond2_257235)
    
    # Applying the binary operator '-' (line 370)
    result_sub_257239 = python_operator(stypy.reporting.localization.Localization(__file__, 370, 36), '-', result_mul_257234, subscript_call_result_257238)
    
    # Processing the call keyword arguments (line 370)
    kwargs_257240 = {}
    # Getting the type of 'qspline1d_eval' (line 370)
    qspline1d_eval_257228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 17), 'qspline1d_eval', False)
    # Calling qspline1d_eval(args, kwargs) (line 370)
    qspline1d_eval_call_result_257241 = invoke(stypy.reporting.localization.Localization(__file__, 370, 17), qspline1d_eval_257228, *[cj_257229, result_sub_257239], **kwargs_257240)
    
    # Getting the type of 'res' (line 370)
    res_257242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 4), 'res')
    # Getting the type of 'cond2' (line 370)
    cond2_257243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 8), 'cond2')
    # Storing an element on a container (line 370)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 370, 4), res_257242, (cond2_257243, qspline1d_eval_call_result_257241))
    
    # Assigning a Subscript to a Name (line 371):
    
    # Assigning a Subscript to a Name (line 371):
    
    # Obtaining the type of the subscript
    # Getting the type of 'cond3' (line 371)
    cond3_257244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 16), 'cond3')
    # Getting the type of 'newx' (line 371)
    newx_257245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 11), 'newx')
    # Obtaining the member '__getitem__' of a type (line 371)
    getitem___257246 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 371, 11), newx_257245, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 371)
    subscript_call_result_257247 = invoke(stypy.reporting.localization.Localization(__file__, 371, 11), getitem___257246, cond3_257244)
    
    # Assigning a type to the variable 'newx' (line 371)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 371, 4), 'newx', subscript_call_result_257247)
    
    
    # Getting the type of 'newx' (line 372)
    newx_257248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 7), 'newx')
    # Obtaining the member 'size' of a type (line 372)
    size_257249 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 7), newx_257248, 'size')
    int_257250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, 20), 'int')
    # Applying the binary operator '==' (line 372)
    result_eq_257251 = python_operator(stypy.reporting.localization.Localization(__file__, 372, 7), '==', size_257249, int_257250)
    
    # Testing the type of an if condition (line 372)
    if_condition_257252 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 372, 4), result_eq_257251)
    # Assigning a type to the variable 'if_condition_257252' (line 372)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 372, 4), 'if_condition_257252', if_condition_257252)
    # SSA begins for if statement (line 372)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'res' (line 373)
    res_257253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 15), 'res')
    # Assigning a type to the variable 'stypy_return_type' (line 373)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 373, 8), 'stypy_return_type', res_257253)
    # SSA join for if statement (line 372)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 374):
    
    # Assigning a Call to a Name (line 374):
    
    # Call to zeros_like(...): (line 374)
    # Processing the call arguments (line 374)
    # Getting the type of 'newx' (line 374)
    newx_257255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 24), 'newx', False)
    # Processing the call keyword arguments (line 374)
    kwargs_257256 = {}
    # Getting the type of 'zeros_like' (line 374)
    zeros_like_257254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 13), 'zeros_like', False)
    # Calling zeros_like(args, kwargs) (line 374)
    zeros_like_call_result_257257 = invoke(stypy.reporting.localization.Localization(__file__, 374, 13), zeros_like_257254, *[newx_257255], **kwargs_257256)
    
    # Assigning a type to the variable 'result' (line 374)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 374, 4), 'result', zeros_like_call_result_257257)
    
    # Assigning a BinOp to a Name (line 375):
    
    # Assigning a BinOp to a Name (line 375):
    
    # Call to astype(...): (line 375)
    # Processing the call arguments (line 375)
    # Getting the type of 'int' (line 375)
    int_257265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 38), 'int', False)
    # Processing the call keyword arguments (line 375)
    kwargs_257266 = {}
    
    # Call to floor(...): (line 375)
    # Processing the call arguments (line 375)
    # Getting the type of 'newx' (line 375)
    newx_257259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 19), 'newx', False)
    float_257260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 26), 'float')
    # Applying the binary operator '-' (line 375)
    result_sub_257261 = python_operator(stypy.reporting.localization.Localization(__file__, 375, 19), '-', newx_257259, float_257260)
    
    # Processing the call keyword arguments (line 375)
    kwargs_257262 = {}
    # Getting the type of 'floor' (line 375)
    floor_257258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 13), 'floor', False)
    # Calling floor(args, kwargs) (line 375)
    floor_call_result_257263 = invoke(stypy.reporting.localization.Localization(__file__, 375, 13), floor_257258, *[result_sub_257261], **kwargs_257262)
    
    # Obtaining the member 'astype' of a type (line 375)
    astype_257264 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 375, 13), floor_call_result_257263, 'astype')
    # Calling astype(args, kwargs) (line 375)
    astype_call_result_257267 = invoke(stypy.reporting.localization.Localization(__file__, 375, 13), astype_257264, *[int_257265], **kwargs_257266)
    
    int_257268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 45), 'int')
    # Applying the binary operator '+' (line 375)
    result_add_257269 = python_operator(stypy.reporting.localization.Localization(__file__, 375, 13), '+', astype_call_result_257267, int_257268)
    
    # Assigning a type to the variable 'jlower' (line 375)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 375, 4), 'jlower', result_add_257269)
    
    
    # Call to range(...): (line 376)
    # Processing the call arguments (line 376)
    int_257271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 376, 19), 'int')
    # Processing the call keyword arguments (line 376)
    kwargs_257272 = {}
    # Getting the type of 'range' (line 376)
    range_257270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 13), 'range', False)
    # Calling range(args, kwargs) (line 376)
    range_call_result_257273 = invoke(stypy.reporting.localization.Localization(__file__, 376, 13), range_257270, *[int_257271], **kwargs_257272)
    
    # Testing the type of a for loop iterable (line 376)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 376, 4), range_call_result_257273)
    # Getting the type of the for loop variable (line 376)
    for_loop_var_257274 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 376, 4), range_call_result_257273)
    # Assigning a type to the variable 'i' (line 376)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 376, 4), 'i', for_loop_var_257274)
    # SSA begins for a for statement (line 376)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a BinOp to a Name (line 377):
    
    # Assigning a BinOp to a Name (line 377):
    # Getting the type of 'jlower' (line 377)
    jlower_257275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 16), 'jlower')
    # Getting the type of 'i' (line 377)
    i_257276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 25), 'i')
    # Applying the binary operator '+' (line 377)
    result_add_257277 = python_operator(stypy.reporting.localization.Localization(__file__, 377, 16), '+', jlower_257275, i_257276)
    
    # Assigning a type to the variable 'thisj' (line 377)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 377, 8), 'thisj', result_add_257277)
    
    # Assigning a Call to a Name (line 378):
    
    # Assigning a Call to a Name (line 378):
    
    # Call to clip(...): (line 378)
    # Processing the call arguments (line 378)
    int_257280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 378, 26), 'int')
    # Getting the type of 'N' (line 378)
    N_257281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 29), 'N', False)
    int_257282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 378, 33), 'int')
    # Applying the binary operator '-' (line 378)
    result_sub_257283 = python_operator(stypy.reporting.localization.Localization(__file__, 378, 29), '-', N_257281, int_257282)
    
    # Processing the call keyword arguments (line 378)
    kwargs_257284 = {}
    # Getting the type of 'thisj' (line 378)
    thisj_257278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 15), 'thisj', False)
    # Obtaining the member 'clip' of a type (line 378)
    clip_257279 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 378, 15), thisj_257278, 'clip')
    # Calling clip(args, kwargs) (line 378)
    clip_call_result_257285 = invoke(stypy.reporting.localization.Localization(__file__, 378, 15), clip_257279, *[int_257280, result_sub_257283], **kwargs_257284)
    
    # Assigning a type to the variable 'indj' (line 378)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 378, 8), 'indj', clip_call_result_257285)
    
    # Getting the type of 'result' (line 379)
    result_257286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 8), 'result')
    
    # Obtaining the type of the subscript
    # Getting the type of 'indj' (line 379)
    indj_257287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 21), 'indj')
    # Getting the type of 'cj' (line 379)
    cj_257288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 18), 'cj')
    # Obtaining the member '__getitem__' of a type (line 379)
    getitem___257289 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 379, 18), cj_257288, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 379)
    subscript_call_result_257290 = invoke(stypy.reporting.localization.Localization(__file__, 379, 18), getitem___257289, indj_257287)
    
    
    # Call to quadratic(...): (line 379)
    # Processing the call arguments (line 379)
    # Getting the type of 'newx' (line 379)
    newx_257292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 39), 'newx', False)
    # Getting the type of 'thisj' (line 379)
    thisj_257293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 46), 'thisj', False)
    # Applying the binary operator '-' (line 379)
    result_sub_257294 = python_operator(stypy.reporting.localization.Localization(__file__, 379, 39), '-', newx_257292, thisj_257293)
    
    # Processing the call keyword arguments (line 379)
    kwargs_257295 = {}
    # Getting the type of 'quadratic' (line 379)
    quadratic_257291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 29), 'quadratic', False)
    # Calling quadratic(args, kwargs) (line 379)
    quadratic_call_result_257296 = invoke(stypy.reporting.localization.Localization(__file__, 379, 29), quadratic_257291, *[result_sub_257294], **kwargs_257295)
    
    # Applying the binary operator '*' (line 379)
    result_mul_257297 = python_operator(stypy.reporting.localization.Localization(__file__, 379, 18), '*', subscript_call_result_257290, quadratic_call_result_257296)
    
    # Applying the binary operator '+=' (line 379)
    result_iadd_257298 = python_operator(stypy.reporting.localization.Localization(__file__, 379, 8), '+=', result_257286, result_mul_257297)
    # Assigning a type to the variable 'result' (line 379)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 379, 8), 'result', result_iadd_257298)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Subscript (line 380):
    
    # Assigning a Name to a Subscript (line 380):
    # Getting the type of 'result' (line 380)
    result_257299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 17), 'result')
    # Getting the type of 'res' (line 380)
    res_257300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 4), 'res')
    # Getting the type of 'cond3' (line 380)
    cond3_257301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 8), 'cond3')
    # Storing an element on a container (line 380)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 380, 4), res_257300, (cond3_257301, result_257299))
    # Getting the type of 'res' (line 381)
    res_257302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 11), 'res')
    # Assigning a type to the variable 'stypy_return_type' (line 381)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 381, 4), 'stypy_return_type', res_257302)
    
    # ################# End of 'qspline1d_eval(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'qspline1d_eval' in the type store
    # Getting the type of 'stypy_return_type' (line 348)
    stypy_return_type_257303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_257303)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'qspline1d_eval'
    return stypy_return_type_257303

# Assigning a type to the variable 'qspline1d_eval' (line 348)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 348, 0), 'qspline1d_eval', qspline1d_eval)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
