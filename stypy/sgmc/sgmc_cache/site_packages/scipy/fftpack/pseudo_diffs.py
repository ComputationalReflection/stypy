
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Differential and pseudo-differential operators.
3: '''
4: # Created by Pearu Peterson, September 2002
5: from __future__ import division, print_function, absolute_import
6: 
7: 
8: __all__ = ['diff',
9:            'tilbert','itilbert','hilbert','ihilbert',
10:            'cs_diff','cc_diff','sc_diff','ss_diff',
11:            'shift']
12: 
13: from numpy import pi, asarray, sin, cos, sinh, cosh, tanh, iscomplexobj
14: from . import convolve
15: 
16: from scipy.fftpack.basic import _datacopied
17: 
18: import atexit
19: atexit.register(convolve.destroy_convolve_cache)
20: del atexit
21: 
22: 
23: _cache = {}
24: 
25: 
26: def diff(x,order=1,period=None, _cache=_cache):
27:     '''
28:     Return k-th derivative (or integral) of a periodic sequence x.
29: 
30:     If x_j and y_j are Fourier coefficients of periodic functions x
31:     and y, respectively, then::
32: 
33:       y_j = pow(sqrt(-1)*j*2*pi/period, order) * x_j
34:       y_0 = 0 if order is not 0.
35: 
36:     Parameters
37:     ----------
38:     x : array_like
39:         Input array.
40:     order : int, optional
41:         The order of differentiation. Default order is 1. If order is
42:         negative, then integration is carried out under the assumption
43:         that ``x_0 == 0``.
44:     period : float, optional
45:         The assumed period of the sequence. Default is ``2*pi``.
46: 
47:     Notes
48:     -----
49:     If ``sum(x, axis=0) = 0`` then ``diff(diff(x, k), -k) == x`` (within
50:     numerical accuracy).
51: 
52:     For odd order and even ``len(x)``, the Nyquist mode is taken zero.
53: 
54:     '''
55:     tmp = asarray(x)
56:     if order == 0:
57:         return tmp
58:     if iscomplexobj(tmp):
59:         return diff(tmp.real,order,period)+1j*diff(tmp.imag,order,period)
60:     if period is not None:
61:         c = 2*pi/period
62:     else:
63:         c = 1.0
64:     n = len(x)
65:     omega = _cache.get((n,order,c))
66:     if omega is None:
67:         if len(_cache) > 20:
68:             while _cache:
69:                 _cache.popitem()
70: 
71:         def kernel(k,order=order,c=c):
72:             if k:
73:                 return pow(c*k,order)
74:             return 0
75:         omega = convolve.init_convolution_kernel(n,kernel,d=order,
76:                                                  zero_nyquist=1)
77:         _cache[(n,order,c)] = omega
78:     overwrite_x = _datacopied(tmp, x)
79:     return convolve.convolve(tmp,omega,swap_real_imag=order % 2,
80:                              overwrite_x=overwrite_x)
81: del _cache
82: 
83: 
84: _cache = {}
85: 
86: 
87: def tilbert(x, h, period=None, _cache=_cache):
88:     '''
89:     Return h-Tilbert transform of a periodic sequence x.
90: 
91:     If x_j and y_j are Fourier coefficients of periodic functions x
92:     and y, respectively, then::
93: 
94:         y_j = sqrt(-1)*coth(j*h*2*pi/period) * x_j
95:         y_0 = 0
96: 
97:     Parameters
98:     ----------
99:     x : array_like
100:         The input array to transform.
101:     h : float
102:         Defines the parameter of the Tilbert transform.
103:     period : float, optional
104:         The assumed period of the sequence.  Default period is ``2*pi``.
105: 
106:     Returns
107:     -------
108:     tilbert : ndarray
109:         The result of the transform.
110: 
111:     Notes
112:     -----
113:     If ``sum(x, axis=0) == 0`` and ``n = len(x)`` is odd then
114:     ``tilbert(itilbert(x)) == x``.
115: 
116:     If ``2 * pi * h / period`` is approximately 10 or larger, then
117:     numerically ``tilbert == hilbert``
118:     (theoretically oo-Tilbert == Hilbert).
119: 
120:     For even ``len(x)``, the Nyquist mode of ``x`` is taken zero.
121: 
122:     '''
123:     tmp = asarray(x)
124:     if iscomplexobj(tmp):
125:         return tilbert(tmp.real, h, period) + \
126:                1j * tilbert(tmp.imag, h, period)
127: 
128:     if period is not None:
129:         h = h * 2 * pi / period
130: 
131:     n = len(x)
132:     omega = _cache.get((n, h))
133:     if omega is None:
134:         if len(_cache) > 20:
135:             while _cache:
136:                 _cache.popitem()
137: 
138:         def kernel(k, h=h):
139:             if k:
140:                 return 1.0/tanh(h*k)
141: 
142:             return 0
143: 
144:         omega = convolve.init_convolution_kernel(n, kernel, d=1)
145:         _cache[(n,h)] = omega
146: 
147:     overwrite_x = _datacopied(tmp, x)
148:     return convolve.convolve(tmp,omega,swap_real_imag=1,overwrite_x=overwrite_x)
149: del _cache
150: 
151: 
152: _cache = {}
153: 
154: 
155: def itilbert(x,h,period=None, _cache=_cache):
156:     '''
157:     Return inverse h-Tilbert transform of a periodic sequence x.
158: 
159:     If ``x_j`` and ``y_j`` are Fourier coefficients of periodic functions x
160:     and y, respectively, then::
161: 
162:       y_j = -sqrt(-1)*tanh(j*h*2*pi/period) * x_j
163:       y_0 = 0
164: 
165:     For more details, see `tilbert`.
166: 
167:     '''
168:     tmp = asarray(x)
169:     if iscomplexobj(tmp):
170:         return itilbert(tmp.real,h,period) + \
171:                1j*itilbert(tmp.imag,h,period)
172:     if period is not None:
173:         h = h*2*pi/period
174:     n = len(x)
175:     omega = _cache.get((n,h))
176:     if omega is None:
177:         if len(_cache) > 20:
178:             while _cache:
179:                 _cache.popitem()
180: 
181:         def kernel(k,h=h):
182:             if k:
183:                 return -tanh(h*k)
184:             return 0
185:         omega = convolve.init_convolution_kernel(n,kernel,d=1)
186:         _cache[(n,h)] = omega
187:     overwrite_x = _datacopied(tmp, x)
188:     return convolve.convolve(tmp,omega,swap_real_imag=1,overwrite_x=overwrite_x)
189: del _cache
190: 
191: 
192: _cache = {}
193: 
194: 
195: def hilbert(x, _cache=_cache):
196:     '''
197:     Return Hilbert transform of a periodic sequence x.
198: 
199:     If x_j and y_j are Fourier coefficients of periodic functions x
200:     and y, respectively, then::
201: 
202:       y_j = sqrt(-1)*sign(j) * x_j
203:       y_0 = 0
204: 
205:     Parameters
206:     ----------
207:     x : array_like
208:         The input array, should be periodic.
209:     _cache : dict, optional
210:         Dictionary that contains the kernel used to do a convolution with.
211: 
212:     Returns
213:     -------
214:     y : ndarray
215:         The transformed input.
216: 
217:     See Also
218:     --------
219:     scipy.signal.hilbert : Compute the analytic signal, using the Hilbert
220:                            transform.
221: 
222:     Notes
223:     -----
224:     If ``sum(x, axis=0) == 0`` then ``hilbert(ihilbert(x)) == x``.
225: 
226:     For even len(x), the Nyquist mode of x is taken zero.
227: 
228:     The sign of the returned transform does not have a factor -1 that is more
229:     often than not found in the definition of the Hilbert transform.  Note also
230:     that `scipy.signal.hilbert` does have an extra -1 factor compared to this
231:     function.
232: 
233:     '''
234:     tmp = asarray(x)
235:     if iscomplexobj(tmp):
236:         return hilbert(tmp.real)+1j*hilbert(tmp.imag)
237:     n = len(x)
238:     omega = _cache.get(n)
239:     if omega is None:
240:         if len(_cache) > 20:
241:             while _cache:
242:                 _cache.popitem()
243: 
244:         def kernel(k):
245:             if k > 0:
246:                 return 1.0
247:             elif k < 0:
248:                 return -1.0
249:             return 0.0
250:         omega = convolve.init_convolution_kernel(n,kernel,d=1)
251:         _cache[n] = omega
252:     overwrite_x = _datacopied(tmp, x)
253:     return convolve.convolve(tmp,omega,swap_real_imag=1,overwrite_x=overwrite_x)
254: del _cache
255: 
256: 
257: def ihilbert(x):
258:     '''
259:     Return inverse Hilbert transform of a periodic sequence x.
260: 
261:     If ``x_j`` and ``y_j`` are Fourier coefficients of periodic functions x
262:     and y, respectively, then::
263: 
264:       y_j = -sqrt(-1)*sign(j) * x_j
265:       y_0 = 0
266: 
267:     '''
268:     return -hilbert(x)
269: 
270: 
271: _cache = {}
272: 
273: 
274: def cs_diff(x, a, b, period=None, _cache=_cache):
275:     '''
276:     Return (a,b)-cosh/sinh pseudo-derivative of a periodic sequence.
277: 
278:     If ``x_j`` and ``y_j`` are Fourier coefficients of periodic functions x
279:     and y, respectively, then::
280: 
281:       y_j = -sqrt(-1)*cosh(j*a*2*pi/period)/sinh(j*b*2*pi/period) * x_j
282:       y_0 = 0
283: 
284:     Parameters
285:     ----------
286:     x : array_like
287:         The array to take the pseudo-derivative from.
288:     a, b : float
289:         Defines the parameters of the cosh/sinh pseudo-differential
290:         operator.
291:     period : float, optional
292:         The period of the sequence. Default period is ``2*pi``.
293: 
294:     Returns
295:     -------
296:     cs_diff : ndarray
297:         Pseudo-derivative of periodic sequence `x`.
298: 
299:     Notes
300:     -----
301:     For even len(`x`), the Nyquist mode of `x` is taken as zero.
302: 
303:     '''
304:     tmp = asarray(x)
305:     if iscomplexobj(tmp):
306:         return cs_diff(tmp.real,a,b,period) + \
307:                1j*cs_diff(tmp.imag,a,b,period)
308:     if period is not None:
309:         a = a*2*pi/period
310:         b = b*2*pi/period
311:     n = len(x)
312:     omega = _cache.get((n,a,b))
313:     if omega is None:
314:         if len(_cache) > 20:
315:             while _cache:
316:                 _cache.popitem()
317: 
318:         def kernel(k,a=a,b=b):
319:             if k:
320:                 return -cosh(a*k)/sinh(b*k)
321:             return 0
322:         omega = convolve.init_convolution_kernel(n,kernel,d=1)
323:         _cache[(n,a,b)] = omega
324:     overwrite_x = _datacopied(tmp, x)
325:     return convolve.convolve(tmp,omega,swap_real_imag=1,overwrite_x=overwrite_x)
326: del _cache
327: 
328: 
329: _cache = {}
330: 
331: 
332: def sc_diff(x, a, b, period=None, _cache=_cache):
333:     '''
334:     Return (a,b)-sinh/cosh pseudo-derivative of a periodic sequence x.
335: 
336:     If x_j and y_j are Fourier coefficients of periodic functions x
337:     and y, respectively, then::
338: 
339:       y_j = sqrt(-1)*sinh(j*a*2*pi/period)/cosh(j*b*2*pi/period) * x_j
340:       y_0 = 0
341: 
342:     Parameters
343:     ----------
344:     x : array_like
345:         Input array.
346:     a,b : float
347:         Defines the parameters of the sinh/cosh pseudo-differential
348:         operator.
349:     period : float, optional
350:         The period of the sequence x. Default is 2*pi.
351: 
352:     Notes
353:     -----
354:     ``sc_diff(cs_diff(x,a,b),b,a) == x``
355:     For even ``len(x)``, the Nyquist mode of x is taken as zero.
356: 
357:     '''
358:     tmp = asarray(x)
359:     if iscomplexobj(tmp):
360:         return sc_diff(tmp.real,a,b,period) + \
361:                1j*sc_diff(tmp.imag,a,b,period)
362:     if period is not None:
363:         a = a*2*pi/period
364:         b = b*2*pi/period
365:     n = len(x)
366:     omega = _cache.get((n,a,b))
367:     if omega is None:
368:         if len(_cache) > 20:
369:             while _cache:
370:                 _cache.popitem()
371: 
372:         def kernel(k,a=a,b=b):
373:             if k:
374:                 return sinh(a*k)/cosh(b*k)
375:             return 0
376:         omega = convolve.init_convolution_kernel(n,kernel,d=1)
377:         _cache[(n,a,b)] = omega
378:     overwrite_x = _datacopied(tmp, x)
379:     return convolve.convolve(tmp,omega,swap_real_imag=1,overwrite_x=overwrite_x)
380: del _cache
381: 
382: 
383: _cache = {}
384: 
385: 
386: def ss_diff(x, a, b, period=None, _cache=_cache):
387:     '''
388:     Return (a,b)-sinh/sinh pseudo-derivative of a periodic sequence x.
389: 
390:     If x_j and y_j are Fourier coefficients of periodic functions x
391:     and y, respectively, then::
392: 
393:       y_j = sinh(j*a*2*pi/period)/sinh(j*b*2*pi/period) * x_j
394:       y_0 = a/b * x_0
395: 
396:     Parameters
397:     ----------
398:     x : array_like
399:         The array to take the pseudo-derivative from.
400:     a,b
401:         Defines the parameters of the sinh/sinh pseudo-differential
402:         operator.
403:     period : float, optional
404:         The period of the sequence x. Default is ``2*pi``.
405: 
406:     Notes
407:     -----
408:     ``ss_diff(ss_diff(x,a,b),b,a) == x``
409: 
410:     '''
411:     tmp = asarray(x)
412:     if iscomplexobj(tmp):
413:         return ss_diff(tmp.real,a,b,period) + \
414:                1j*ss_diff(tmp.imag,a,b,period)
415:     if period is not None:
416:         a = a*2*pi/period
417:         b = b*2*pi/period
418:     n = len(x)
419:     omega = _cache.get((n,a,b))
420:     if omega is None:
421:         if len(_cache) > 20:
422:             while _cache:
423:                 _cache.popitem()
424: 
425:         def kernel(k,a=a,b=b):
426:             if k:
427:                 return sinh(a*k)/sinh(b*k)
428:             return float(a)/b
429:         omega = convolve.init_convolution_kernel(n,kernel)
430:         _cache[(n,a,b)] = omega
431:     overwrite_x = _datacopied(tmp, x)
432:     return convolve.convolve(tmp,omega,overwrite_x=overwrite_x)
433: del _cache
434: 
435: 
436: _cache = {}
437: 
438: 
439: def cc_diff(x, a, b, period=None, _cache=_cache):
440:     '''
441:     Return (a,b)-cosh/cosh pseudo-derivative of a periodic sequence.
442: 
443:     If x_j and y_j are Fourier coefficients of periodic functions x
444:     and y, respectively, then::
445: 
446:       y_j = cosh(j*a*2*pi/period)/cosh(j*b*2*pi/period) * x_j
447: 
448:     Parameters
449:     ----------
450:     x : array_like
451:         The array to take the pseudo-derivative from.
452:     a,b : float
453:         Defines the parameters of the sinh/sinh pseudo-differential
454:         operator.
455:     period : float, optional
456:         The period of the sequence x. Default is ``2*pi``.
457: 
458:     Returns
459:     -------
460:     cc_diff : ndarray
461:         Pseudo-derivative of periodic sequence `x`.
462: 
463:     Notes
464:     -----
465:     ``cc_diff(cc_diff(x,a,b),b,a) == x``
466: 
467:     '''
468:     tmp = asarray(x)
469:     if iscomplexobj(tmp):
470:         return cc_diff(tmp.real,a,b,period) + \
471:                1j*cc_diff(tmp.imag,a,b,period)
472:     if period is not None:
473:         a = a*2*pi/period
474:         b = b*2*pi/period
475:     n = len(x)
476:     omega = _cache.get((n,a,b))
477:     if omega is None:
478:         if len(_cache) > 20:
479:             while _cache:
480:                 _cache.popitem()
481: 
482:         def kernel(k,a=a,b=b):
483:             return cosh(a*k)/cosh(b*k)
484:         omega = convolve.init_convolution_kernel(n,kernel)
485:         _cache[(n,a,b)] = omega
486:     overwrite_x = _datacopied(tmp, x)
487:     return convolve.convolve(tmp,omega,overwrite_x=overwrite_x)
488: del _cache
489: 
490: 
491: _cache = {}
492: 
493: 
494: def shift(x, a, period=None, _cache=_cache):
495:     '''
496:     Shift periodic sequence x by a: y(u) = x(u+a).
497: 
498:     If x_j and y_j are Fourier coefficients of periodic functions x
499:     and y, respectively, then::
500: 
501:           y_j = exp(j*a*2*pi/period*sqrt(-1)) * x_f
502: 
503:     Parameters
504:     ----------
505:     x : array_like
506:         The array to take the pseudo-derivative from.
507:     a : float
508:         Defines the parameters of the sinh/sinh pseudo-differential
509:     period : float, optional
510:         The period of the sequences x and y. Default period is ``2*pi``.
511:     '''
512:     tmp = asarray(x)
513:     if iscomplexobj(tmp):
514:         return shift(tmp.real,a,period)+1j*shift(tmp.imag,a,period)
515:     if period is not None:
516:         a = a*2*pi/period
517:     n = len(x)
518:     omega = _cache.get((n,a))
519:     if omega is None:
520:         if len(_cache) > 20:
521:             while _cache:
522:                 _cache.popitem()
523: 
524:         def kernel_real(k,a=a):
525:             return cos(a*k)
526: 
527:         def kernel_imag(k,a=a):
528:             return sin(a*k)
529:         omega_real = convolve.init_convolution_kernel(n,kernel_real,d=0,
530:                                                       zero_nyquist=0)
531:         omega_imag = convolve.init_convolution_kernel(n,kernel_imag,d=1,
532:                                                       zero_nyquist=0)
533:         _cache[(n,a)] = omega_real,omega_imag
534:     else:
535:         omega_real,omega_imag = omega
536:     overwrite_x = _datacopied(tmp, x)
537:     return convolve.convolve_z(tmp,omega_real,omega_imag,
538:                                overwrite_x=overwrite_x)
539: 
540: del _cache
541: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_16492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, (-1)), 'str', '\nDifferential and pseudo-differential operators.\n')

# Assigning a List to a Name (line 8):

# Assigning a List to a Name (line 8):
__all__ = ['diff', 'tilbert', 'itilbert', 'hilbert', 'ihilbert', 'cs_diff', 'cc_diff', 'sc_diff', 'ss_diff', 'shift']
module_type_store.set_exportable_members(['diff', 'tilbert', 'itilbert', 'hilbert', 'ihilbert', 'cs_diff', 'cc_diff', 'sc_diff', 'ss_diff', 'shift'])

# Obtaining an instance of the builtin type 'list' (line 8)
list_16493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 8)
# Adding element type (line 8)
str_16494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 11), 'str', 'diff')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 10), list_16493, str_16494)
# Adding element type (line 8)
str_16495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 11), 'str', 'tilbert')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 10), list_16493, str_16495)
# Adding element type (line 8)
str_16496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 21), 'str', 'itilbert')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 10), list_16493, str_16496)
# Adding element type (line 8)
str_16497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 32), 'str', 'hilbert')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 10), list_16493, str_16497)
# Adding element type (line 8)
str_16498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 42), 'str', 'ihilbert')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 10), list_16493, str_16498)
# Adding element type (line 8)
str_16499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 11), 'str', 'cs_diff')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 10), list_16493, str_16499)
# Adding element type (line 8)
str_16500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 21), 'str', 'cc_diff')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 10), list_16493, str_16500)
# Adding element type (line 8)
str_16501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 31), 'str', 'sc_diff')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 10), list_16493, str_16501)
# Adding element type (line 8)
str_16502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 41), 'str', 'ss_diff')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 10), list_16493, str_16502)
# Adding element type (line 8)
str_16503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 11), 'str', 'shift')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 10), list_16493, str_16503)

# Assigning a type to the variable '__all__' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), '__all__', list_16493)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from numpy import pi, asarray, sin, cos, sinh, cosh, tanh, iscomplexobj' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/fftpack/')
import_16504 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy')

if (type(import_16504) is not StypyTypeError):

    if (import_16504 != 'pyd_module'):
        __import__(import_16504)
        sys_modules_16505 = sys.modules[import_16504]
        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy', sys_modules_16505.module_type_store, module_type_store, ['pi', 'asarray', 'sin', 'cos', 'sinh', 'cosh', 'tanh', 'iscomplexobj'])
        nest_module(stypy.reporting.localization.Localization(__file__, 13, 0), __file__, sys_modules_16505, sys_modules_16505.module_type_store, module_type_store)
    else:
        from numpy import pi, asarray, sin, cos, sinh, cosh, tanh, iscomplexobj

        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy', None, module_type_store, ['pi', 'asarray', 'sin', 'cos', 'sinh', 'cosh', 'tanh', 'iscomplexobj'], [pi, asarray, sin, cos, sinh, cosh, tanh, iscomplexobj])

else:
    # Assigning a type to the variable 'numpy' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy', import_16504)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/fftpack/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'from scipy.fftpack import convolve' statement (line 14)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/fftpack/')
import_16506 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy.fftpack')

if (type(import_16506) is not StypyTypeError):

    if (import_16506 != 'pyd_module'):
        __import__(import_16506)
        sys_modules_16507 = sys.modules[import_16506]
        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy.fftpack', sys_modules_16507.module_type_store, module_type_store, ['convolve'])
        nest_module(stypy.reporting.localization.Localization(__file__, 14, 0), __file__, sys_modules_16507, sys_modules_16507.module_type_store, module_type_store)
    else:
        from scipy.fftpack import convolve

        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy.fftpack', None, module_type_store, ['convolve'], [convolve])

else:
    # Assigning a type to the variable 'scipy.fftpack' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy.fftpack', import_16506)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/fftpack/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 16, 0))

# 'from scipy.fftpack.basic import _datacopied' statement (line 16)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/fftpack/')
import_16508 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy.fftpack.basic')

if (type(import_16508) is not StypyTypeError):

    if (import_16508 != 'pyd_module'):
        __import__(import_16508)
        sys_modules_16509 = sys.modules[import_16508]
        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy.fftpack.basic', sys_modules_16509.module_type_store, module_type_store, ['_datacopied'])
        nest_module(stypy.reporting.localization.Localization(__file__, 16, 0), __file__, sys_modules_16509, sys_modules_16509.module_type_store, module_type_store)
    else:
        from scipy.fftpack.basic import _datacopied

        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy.fftpack.basic', None, module_type_store, ['_datacopied'], [_datacopied])

else:
    # Assigning a type to the variable 'scipy.fftpack.basic' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy.fftpack.basic', import_16508)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/fftpack/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 18, 0))

# 'import atexit' statement (line 18)
import atexit

import_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'atexit', atexit, module_type_store)


# Call to register(...): (line 19)
# Processing the call arguments (line 19)
# Getting the type of 'convolve' (line 19)
convolve_16512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 16), 'convolve', False)
# Obtaining the member 'destroy_convolve_cache' of a type (line 19)
destroy_convolve_cache_16513 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 16), convolve_16512, 'destroy_convolve_cache')
# Processing the call keyword arguments (line 19)
kwargs_16514 = {}
# Getting the type of 'atexit' (line 19)
atexit_16510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'atexit', False)
# Obtaining the member 'register' of a type (line 19)
register_16511 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 0), atexit_16510, 'register')
# Calling register(args, kwargs) (line 19)
register_call_result_16515 = invoke(stypy.reporting.localization.Localization(__file__, 19, 0), register_16511, *[destroy_convolve_cache_16513], **kwargs_16514)

# Deleting a member
module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 20, 0), module_type_store, 'atexit')

# Assigning a Dict to a Name (line 23):

# Assigning a Dict to a Name (line 23):

# Obtaining an instance of the builtin type 'dict' (line 23)
dict_16516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 9), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 23)

# Assigning a type to the variable '_cache' (line 23)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), '_cache', dict_16516)

@norecursion
def diff(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_16517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 17), 'int')
    # Getting the type of 'None' (line 26)
    None_16518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 26), 'None')
    # Getting the type of '_cache' (line 26)
    _cache_16519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 39), '_cache')
    defaults = [int_16517, None_16518, _cache_16519]
    # Create a new context for function 'diff'
    module_type_store = module_type_store.open_function_context('diff', 26, 0, False)
    
    # Passed parameters checking function
    diff.stypy_localization = localization
    diff.stypy_type_of_self = None
    diff.stypy_type_store = module_type_store
    diff.stypy_function_name = 'diff'
    diff.stypy_param_names_list = ['x', 'order', 'period', '_cache']
    diff.stypy_varargs_param_name = None
    diff.stypy_kwargs_param_name = None
    diff.stypy_call_defaults = defaults
    diff.stypy_call_varargs = varargs
    diff.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'diff', ['x', 'order', 'period', '_cache'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'diff', localization, ['x', 'order', 'period', '_cache'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'diff(...)' code ##################

    str_16520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, (-1)), 'str', '\n    Return k-th derivative (or integral) of a periodic sequence x.\n\n    If x_j and y_j are Fourier coefficients of periodic functions x\n    and y, respectively, then::\n\n      y_j = pow(sqrt(-1)*j*2*pi/period, order) * x_j\n      y_0 = 0 if order is not 0.\n\n    Parameters\n    ----------\n    x : array_like\n        Input array.\n    order : int, optional\n        The order of differentiation. Default order is 1. If order is\n        negative, then integration is carried out under the assumption\n        that ``x_0 == 0``.\n    period : float, optional\n        The assumed period of the sequence. Default is ``2*pi``.\n\n    Notes\n    -----\n    If ``sum(x, axis=0) = 0`` then ``diff(diff(x, k), -k) == x`` (within\n    numerical accuracy).\n\n    For odd order and even ``len(x)``, the Nyquist mode is taken zero.\n\n    ')
    
    # Assigning a Call to a Name (line 55):
    
    # Assigning a Call to a Name (line 55):
    
    # Call to asarray(...): (line 55)
    # Processing the call arguments (line 55)
    # Getting the type of 'x' (line 55)
    x_16522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 18), 'x', False)
    # Processing the call keyword arguments (line 55)
    kwargs_16523 = {}
    # Getting the type of 'asarray' (line 55)
    asarray_16521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 10), 'asarray', False)
    # Calling asarray(args, kwargs) (line 55)
    asarray_call_result_16524 = invoke(stypy.reporting.localization.Localization(__file__, 55, 10), asarray_16521, *[x_16522], **kwargs_16523)
    
    # Assigning a type to the variable 'tmp' (line 55)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'tmp', asarray_call_result_16524)
    
    
    # Getting the type of 'order' (line 56)
    order_16525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 7), 'order')
    int_16526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 16), 'int')
    # Applying the binary operator '==' (line 56)
    result_eq_16527 = python_operator(stypy.reporting.localization.Localization(__file__, 56, 7), '==', order_16525, int_16526)
    
    # Testing the type of an if condition (line 56)
    if_condition_16528 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 56, 4), result_eq_16527)
    # Assigning a type to the variable 'if_condition_16528' (line 56)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'if_condition_16528', if_condition_16528)
    # SSA begins for if statement (line 56)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'tmp' (line 57)
    tmp_16529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 15), 'tmp')
    # Assigning a type to the variable 'stypy_return_type' (line 57)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'stypy_return_type', tmp_16529)
    # SSA join for if statement (line 56)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to iscomplexobj(...): (line 58)
    # Processing the call arguments (line 58)
    # Getting the type of 'tmp' (line 58)
    tmp_16531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 20), 'tmp', False)
    # Processing the call keyword arguments (line 58)
    kwargs_16532 = {}
    # Getting the type of 'iscomplexobj' (line 58)
    iscomplexobj_16530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 7), 'iscomplexobj', False)
    # Calling iscomplexobj(args, kwargs) (line 58)
    iscomplexobj_call_result_16533 = invoke(stypy.reporting.localization.Localization(__file__, 58, 7), iscomplexobj_16530, *[tmp_16531], **kwargs_16532)
    
    # Testing the type of an if condition (line 58)
    if_condition_16534 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 58, 4), iscomplexobj_call_result_16533)
    # Assigning a type to the variable 'if_condition_16534' (line 58)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'if_condition_16534', if_condition_16534)
    # SSA begins for if statement (line 58)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to diff(...): (line 59)
    # Processing the call arguments (line 59)
    # Getting the type of 'tmp' (line 59)
    tmp_16536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 20), 'tmp', False)
    # Obtaining the member 'real' of a type (line 59)
    real_16537 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 20), tmp_16536, 'real')
    # Getting the type of 'order' (line 59)
    order_16538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 29), 'order', False)
    # Getting the type of 'period' (line 59)
    period_16539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 35), 'period', False)
    # Processing the call keyword arguments (line 59)
    kwargs_16540 = {}
    # Getting the type of 'diff' (line 59)
    diff_16535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 15), 'diff', False)
    # Calling diff(args, kwargs) (line 59)
    diff_call_result_16541 = invoke(stypy.reporting.localization.Localization(__file__, 59, 15), diff_16535, *[real_16537, order_16538, period_16539], **kwargs_16540)
    
    complex_16542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 43), 'complex')
    
    # Call to diff(...): (line 59)
    # Processing the call arguments (line 59)
    # Getting the type of 'tmp' (line 59)
    tmp_16544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 51), 'tmp', False)
    # Obtaining the member 'imag' of a type (line 59)
    imag_16545 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 51), tmp_16544, 'imag')
    # Getting the type of 'order' (line 59)
    order_16546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 60), 'order', False)
    # Getting the type of 'period' (line 59)
    period_16547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 66), 'period', False)
    # Processing the call keyword arguments (line 59)
    kwargs_16548 = {}
    # Getting the type of 'diff' (line 59)
    diff_16543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 46), 'diff', False)
    # Calling diff(args, kwargs) (line 59)
    diff_call_result_16549 = invoke(stypy.reporting.localization.Localization(__file__, 59, 46), diff_16543, *[imag_16545, order_16546, period_16547], **kwargs_16548)
    
    # Applying the binary operator '*' (line 59)
    result_mul_16550 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 43), '*', complex_16542, diff_call_result_16549)
    
    # Applying the binary operator '+' (line 59)
    result_add_16551 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 15), '+', diff_call_result_16541, result_mul_16550)
    
    # Assigning a type to the variable 'stypy_return_type' (line 59)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'stypy_return_type', result_add_16551)
    # SSA join for if statement (line 58)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 60)
    # Getting the type of 'period' (line 60)
    period_16552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'period')
    # Getting the type of 'None' (line 60)
    None_16553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 21), 'None')
    
    (may_be_16554, more_types_in_union_16555) = may_not_be_none(period_16552, None_16553)

    if may_be_16554:

        if more_types_in_union_16555:
            # Runtime conditional SSA (line 60)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a BinOp to a Name (line 61):
        
        # Assigning a BinOp to a Name (line 61):
        int_16556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 12), 'int')
        # Getting the type of 'pi' (line 61)
        pi_16557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 14), 'pi')
        # Applying the binary operator '*' (line 61)
        result_mul_16558 = python_operator(stypy.reporting.localization.Localization(__file__, 61, 12), '*', int_16556, pi_16557)
        
        # Getting the type of 'period' (line 61)
        period_16559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 17), 'period')
        # Applying the binary operator 'div' (line 61)
        result_div_16560 = python_operator(stypy.reporting.localization.Localization(__file__, 61, 16), 'div', result_mul_16558, period_16559)
        
        # Assigning a type to the variable 'c' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'c', result_div_16560)

        if more_types_in_union_16555:
            # Runtime conditional SSA for else branch (line 60)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_16554) or more_types_in_union_16555):
        
        # Assigning a Num to a Name (line 63):
        
        # Assigning a Num to a Name (line 63):
        float_16561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 12), 'float')
        # Assigning a type to the variable 'c' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'c', float_16561)

        if (may_be_16554 and more_types_in_union_16555):
            # SSA join for if statement (line 60)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 64):
    
    # Assigning a Call to a Name (line 64):
    
    # Call to len(...): (line 64)
    # Processing the call arguments (line 64)
    # Getting the type of 'x' (line 64)
    x_16563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 12), 'x', False)
    # Processing the call keyword arguments (line 64)
    kwargs_16564 = {}
    # Getting the type of 'len' (line 64)
    len_16562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'len', False)
    # Calling len(args, kwargs) (line 64)
    len_call_result_16565 = invoke(stypy.reporting.localization.Localization(__file__, 64, 8), len_16562, *[x_16563], **kwargs_16564)
    
    # Assigning a type to the variable 'n' (line 64)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'n', len_call_result_16565)
    
    # Assigning a Call to a Name (line 65):
    
    # Assigning a Call to a Name (line 65):
    
    # Call to get(...): (line 65)
    # Processing the call arguments (line 65)
    
    # Obtaining an instance of the builtin type 'tuple' (line 65)
    tuple_16568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 24), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 65)
    # Adding element type (line 65)
    # Getting the type of 'n' (line 65)
    n_16569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 24), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 24), tuple_16568, n_16569)
    # Adding element type (line 65)
    # Getting the type of 'order' (line 65)
    order_16570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 26), 'order', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 24), tuple_16568, order_16570)
    # Adding element type (line 65)
    # Getting the type of 'c' (line 65)
    c_16571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 32), 'c', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 24), tuple_16568, c_16571)
    
    # Processing the call keyword arguments (line 65)
    kwargs_16572 = {}
    # Getting the type of '_cache' (line 65)
    _cache_16566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 12), '_cache', False)
    # Obtaining the member 'get' of a type (line 65)
    get_16567 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 12), _cache_16566, 'get')
    # Calling get(args, kwargs) (line 65)
    get_call_result_16573 = invoke(stypy.reporting.localization.Localization(__file__, 65, 12), get_16567, *[tuple_16568], **kwargs_16572)
    
    # Assigning a type to the variable 'omega' (line 65)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'omega', get_call_result_16573)
    
    # Type idiom detected: calculating its left and rigth part (line 66)
    # Getting the type of 'omega' (line 66)
    omega_16574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 7), 'omega')
    # Getting the type of 'None' (line 66)
    None_16575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 16), 'None')
    
    (may_be_16576, more_types_in_union_16577) = may_be_none(omega_16574, None_16575)

    if may_be_16576:

        if more_types_in_union_16577:
            # Runtime conditional SSA (line 66)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        
        
        # Call to len(...): (line 67)
        # Processing the call arguments (line 67)
        # Getting the type of '_cache' (line 67)
        _cache_16579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 15), '_cache', False)
        # Processing the call keyword arguments (line 67)
        kwargs_16580 = {}
        # Getting the type of 'len' (line 67)
        len_16578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 11), 'len', False)
        # Calling len(args, kwargs) (line 67)
        len_call_result_16581 = invoke(stypy.reporting.localization.Localization(__file__, 67, 11), len_16578, *[_cache_16579], **kwargs_16580)
        
        int_16582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 25), 'int')
        # Applying the binary operator '>' (line 67)
        result_gt_16583 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 11), '>', len_call_result_16581, int_16582)
        
        # Testing the type of an if condition (line 67)
        if_condition_16584 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 67, 8), result_gt_16583)
        # Assigning a type to the variable 'if_condition_16584' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'if_condition_16584', if_condition_16584)
        # SSA begins for if statement (line 67)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of '_cache' (line 68)
        _cache_16585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 18), '_cache')
        # Testing the type of an if condition (line 68)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 68, 12), _cache_16585)
        # SSA begins for while statement (line 68)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Call to popitem(...): (line 69)
        # Processing the call keyword arguments (line 69)
        kwargs_16588 = {}
        # Getting the type of '_cache' (line 69)
        _cache_16586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 16), '_cache', False)
        # Obtaining the member 'popitem' of a type (line 69)
        popitem_16587 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 16), _cache_16586, 'popitem')
        # Calling popitem(args, kwargs) (line 69)
        popitem_call_result_16589 = invoke(stypy.reporting.localization.Localization(__file__, 69, 16), popitem_16587, *[], **kwargs_16588)
        
        # SSA join for while statement (line 68)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 67)
        module_type_store = module_type_store.join_ssa_context()
        

        @norecursion
        def kernel(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            # Getting the type of 'order' (line 71)
            order_16590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 27), 'order')
            # Getting the type of 'c' (line 71)
            c_16591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 35), 'c')
            defaults = [order_16590, c_16591]
            # Create a new context for function 'kernel'
            module_type_store = module_type_store.open_function_context('kernel', 71, 8, False)
            
            # Passed parameters checking function
            kernel.stypy_localization = localization
            kernel.stypy_type_of_self = None
            kernel.stypy_type_store = module_type_store
            kernel.stypy_function_name = 'kernel'
            kernel.stypy_param_names_list = ['k', 'order', 'c']
            kernel.stypy_varargs_param_name = None
            kernel.stypy_kwargs_param_name = None
            kernel.stypy_call_defaults = defaults
            kernel.stypy_call_varargs = varargs
            kernel.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'kernel', ['k', 'order', 'c'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'kernel', localization, ['k', 'order', 'c'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'kernel(...)' code ##################

            
            # Getting the type of 'k' (line 72)
            k_16592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 15), 'k')
            # Testing the type of an if condition (line 72)
            if_condition_16593 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 72, 12), k_16592)
            # Assigning a type to the variable 'if_condition_16593' (line 72)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 12), 'if_condition_16593', if_condition_16593)
            # SSA begins for if statement (line 72)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to pow(...): (line 73)
            # Processing the call arguments (line 73)
            # Getting the type of 'c' (line 73)
            c_16595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 27), 'c', False)
            # Getting the type of 'k' (line 73)
            k_16596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 29), 'k', False)
            # Applying the binary operator '*' (line 73)
            result_mul_16597 = python_operator(stypy.reporting.localization.Localization(__file__, 73, 27), '*', c_16595, k_16596)
            
            # Getting the type of 'order' (line 73)
            order_16598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 31), 'order', False)
            # Processing the call keyword arguments (line 73)
            kwargs_16599 = {}
            # Getting the type of 'pow' (line 73)
            pow_16594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 23), 'pow', False)
            # Calling pow(args, kwargs) (line 73)
            pow_call_result_16600 = invoke(stypy.reporting.localization.Localization(__file__, 73, 23), pow_16594, *[result_mul_16597, order_16598], **kwargs_16599)
            
            # Assigning a type to the variable 'stypy_return_type' (line 73)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 16), 'stypy_return_type', pow_call_result_16600)
            # SSA join for if statement (line 72)
            module_type_store = module_type_store.join_ssa_context()
            
            int_16601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 19), 'int')
            # Assigning a type to the variable 'stypy_return_type' (line 74)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 12), 'stypy_return_type', int_16601)
            
            # ################# End of 'kernel(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'kernel' in the type store
            # Getting the type of 'stypy_return_type' (line 71)
            stypy_return_type_16602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_16602)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'kernel'
            return stypy_return_type_16602

        # Assigning a type to the variable 'kernel' (line 71)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'kernel', kernel)
        
        # Assigning a Call to a Name (line 75):
        
        # Assigning a Call to a Name (line 75):
        
        # Call to init_convolution_kernel(...): (line 75)
        # Processing the call arguments (line 75)
        # Getting the type of 'n' (line 75)
        n_16605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 49), 'n', False)
        # Getting the type of 'kernel' (line 75)
        kernel_16606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 51), 'kernel', False)
        # Processing the call keyword arguments (line 75)
        # Getting the type of 'order' (line 75)
        order_16607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 60), 'order', False)
        keyword_16608 = order_16607
        int_16609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 62), 'int')
        keyword_16610 = int_16609
        kwargs_16611 = {'zero_nyquist': keyword_16610, 'd': keyword_16608}
        # Getting the type of 'convolve' (line 75)
        convolve_16603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 16), 'convolve', False)
        # Obtaining the member 'init_convolution_kernel' of a type (line 75)
        init_convolution_kernel_16604 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 16), convolve_16603, 'init_convolution_kernel')
        # Calling init_convolution_kernel(args, kwargs) (line 75)
        init_convolution_kernel_call_result_16612 = invoke(stypy.reporting.localization.Localization(__file__, 75, 16), init_convolution_kernel_16604, *[n_16605, kernel_16606], **kwargs_16611)
        
        # Assigning a type to the variable 'omega' (line 75)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'omega', init_convolution_kernel_call_result_16612)
        
        # Assigning a Name to a Subscript (line 77):
        
        # Assigning a Name to a Subscript (line 77):
        # Getting the type of 'omega' (line 77)
        omega_16613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 30), 'omega')
        # Getting the type of '_cache' (line 77)
        _cache_16614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), '_cache')
        
        # Obtaining an instance of the builtin type 'tuple' (line 77)
        tuple_16615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 16), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 77)
        # Adding element type (line 77)
        # Getting the type of 'n' (line 77)
        n_16616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 16), 'n')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 16), tuple_16615, n_16616)
        # Adding element type (line 77)
        # Getting the type of 'order' (line 77)
        order_16617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 18), 'order')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 16), tuple_16615, order_16617)
        # Adding element type (line 77)
        # Getting the type of 'c' (line 77)
        c_16618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 24), 'c')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 16), tuple_16615, c_16618)
        
        # Storing an element on a container (line 77)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 8), _cache_16614, (tuple_16615, omega_16613))

        if more_types_in_union_16577:
            # SSA join for if statement (line 66)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 78):
    
    # Assigning a Call to a Name (line 78):
    
    # Call to _datacopied(...): (line 78)
    # Processing the call arguments (line 78)
    # Getting the type of 'tmp' (line 78)
    tmp_16620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 30), 'tmp', False)
    # Getting the type of 'x' (line 78)
    x_16621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 35), 'x', False)
    # Processing the call keyword arguments (line 78)
    kwargs_16622 = {}
    # Getting the type of '_datacopied' (line 78)
    _datacopied_16619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 18), '_datacopied', False)
    # Calling _datacopied(args, kwargs) (line 78)
    _datacopied_call_result_16623 = invoke(stypy.reporting.localization.Localization(__file__, 78, 18), _datacopied_16619, *[tmp_16620, x_16621], **kwargs_16622)
    
    # Assigning a type to the variable 'overwrite_x' (line 78)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'overwrite_x', _datacopied_call_result_16623)
    
    # Call to convolve(...): (line 79)
    # Processing the call arguments (line 79)
    # Getting the type of 'tmp' (line 79)
    tmp_16626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 29), 'tmp', False)
    # Getting the type of 'omega' (line 79)
    omega_16627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 33), 'omega', False)
    # Processing the call keyword arguments (line 79)
    # Getting the type of 'order' (line 79)
    order_16628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 54), 'order', False)
    int_16629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 62), 'int')
    # Applying the binary operator '%' (line 79)
    result_mod_16630 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 54), '%', order_16628, int_16629)
    
    keyword_16631 = result_mod_16630
    # Getting the type of 'overwrite_x' (line 80)
    overwrite_x_16632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 41), 'overwrite_x', False)
    keyword_16633 = overwrite_x_16632
    kwargs_16634 = {'overwrite_x': keyword_16633, 'swap_real_imag': keyword_16631}
    # Getting the type of 'convolve' (line 79)
    convolve_16624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 11), 'convolve', False)
    # Obtaining the member 'convolve' of a type (line 79)
    convolve_16625 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 11), convolve_16624, 'convolve')
    # Calling convolve(args, kwargs) (line 79)
    convolve_call_result_16635 = invoke(stypy.reporting.localization.Localization(__file__, 79, 11), convolve_16625, *[tmp_16626, omega_16627], **kwargs_16634)
    
    # Assigning a type to the variable 'stypy_return_type' (line 79)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'stypy_return_type', convolve_call_result_16635)
    
    # ################# End of 'diff(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'diff' in the type store
    # Getting the type of 'stypy_return_type' (line 26)
    stypy_return_type_16636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_16636)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'diff'
    return stypy_return_type_16636

# Assigning a type to the variable 'diff' (line 26)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'diff', diff)
# Deleting a member
module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 81, 0), module_type_store, '_cache')

# Assigning a Dict to a Name (line 84):

# Assigning a Dict to a Name (line 84):

# Obtaining an instance of the builtin type 'dict' (line 84)
dict_16637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 9), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 84)

# Assigning a type to the variable '_cache' (line 84)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 0), '_cache', dict_16637)

@norecursion
def tilbert(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 87)
    None_16638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 25), 'None')
    # Getting the type of '_cache' (line 87)
    _cache_16639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 38), '_cache')
    defaults = [None_16638, _cache_16639]
    # Create a new context for function 'tilbert'
    module_type_store = module_type_store.open_function_context('tilbert', 87, 0, False)
    
    # Passed parameters checking function
    tilbert.stypy_localization = localization
    tilbert.stypy_type_of_self = None
    tilbert.stypy_type_store = module_type_store
    tilbert.stypy_function_name = 'tilbert'
    tilbert.stypy_param_names_list = ['x', 'h', 'period', '_cache']
    tilbert.stypy_varargs_param_name = None
    tilbert.stypy_kwargs_param_name = None
    tilbert.stypy_call_defaults = defaults
    tilbert.stypy_call_varargs = varargs
    tilbert.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'tilbert', ['x', 'h', 'period', '_cache'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'tilbert', localization, ['x', 'h', 'period', '_cache'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'tilbert(...)' code ##################

    str_16640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, (-1)), 'str', '\n    Return h-Tilbert transform of a periodic sequence x.\n\n    If x_j and y_j are Fourier coefficients of periodic functions x\n    and y, respectively, then::\n\n        y_j = sqrt(-1)*coth(j*h*2*pi/period) * x_j\n        y_0 = 0\n\n    Parameters\n    ----------\n    x : array_like\n        The input array to transform.\n    h : float\n        Defines the parameter of the Tilbert transform.\n    period : float, optional\n        The assumed period of the sequence.  Default period is ``2*pi``.\n\n    Returns\n    -------\n    tilbert : ndarray\n        The result of the transform.\n\n    Notes\n    -----\n    If ``sum(x, axis=0) == 0`` and ``n = len(x)`` is odd then\n    ``tilbert(itilbert(x)) == x``.\n\n    If ``2 * pi * h / period`` is approximately 10 or larger, then\n    numerically ``tilbert == hilbert``\n    (theoretically oo-Tilbert == Hilbert).\n\n    For even ``len(x)``, the Nyquist mode of ``x`` is taken zero.\n\n    ')
    
    # Assigning a Call to a Name (line 123):
    
    # Assigning a Call to a Name (line 123):
    
    # Call to asarray(...): (line 123)
    # Processing the call arguments (line 123)
    # Getting the type of 'x' (line 123)
    x_16642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 18), 'x', False)
    # Processing the call keyword arguments (line 123)
    kwargs_16643 = {}
    # Getting the type of 'asarray' (line 123)
    asarray_16641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 10), 'asarray', False)
    # Calling asarray(args, kwargs) (line 123)
    asarray_call_result_16644 = invoke(stypy.reporting.localization.Localization(__file__, 123, 10), asarray_16641, *[x_16642], **kwargs_16643)
    
    # Assigning a type to the variable 'tmp' (line 123)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 4), 'tmp', asarray_call_result_16644)
    
    
    # Call to iscomplexobj(...): (line 124)
    # Processing the call arguments (line 124)
    # Getting the type of 'tmp' (line 124)
    tmp_16646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 20), 'tmp', False)
    # Processing the call keyword arguments (line 124)
    kwargs_16647 = {}
    # Getting the type of 'iscomplexobj' (line 124)
    iscomplexobj_16645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 7), 'iscomplexobj', False)
    # Calling iscomplexobj(args, kwargs) (line 124)
    iscomplexobj_call_result_16648 = invoke(stypy.reporting.localization.Localization(__file__, 124, 7), iscomplexobj_16645, *[tmp_16646], **kwargs_16647)
    
    # Testing the type of an if condition (line 124)
    if_condition_16649 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 124, 4), iscomplexobj_call_result_16648)
    # Assigning a type to the variable 'if_condition_16649' (line 124)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 4), 'if_condition_16649', if_condition_16649)
    # SSA begins for if statement (line 124)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to tilbert(...): (line 125)
    # Processing the call arguments (line 125)
    # Getting the type of 'tmp' (line 125)
    tmp_16651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 23), 'tmp', False)
    # Obtaining the member 'real' of a type (line 125)
    real_16652 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 23), tmp_16651, 'real')
    # Getting the type of 'h' (line 125)
    h_16653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 33), 'h', False)
    # Getting the type of 'period' (line 125)
    period_16654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 36), 'period', False)
    # Processing the call keyword arguments (line 125)
    kwargs_16655 = {}
    # Getting the type of 'tilbert' (line 125)
    tilbert_16650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 15), 'tilbert', False)
    # Calling tilbert(args, kwargs) (line 125)
    tilbert_call_result_16656 = invoke(stypy.reporting.localization.Localization(__file__, 125, 15), tilbert_16650, *[real_16652, h_16653, period_16654], **kwargs_16655)
    
    complex_16657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 15), 'complex')
    
    # Call to tilbert(...): (line 126)
    # Processing the call arguments (line 126)
    # Getting the type of 'tmp' (line 126)
    tmp_16659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 28), 'tmp', False)
    # Obtaining the member 'imag' of a type (line 126)
    imag_16660 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 28), tmp_16659, 'imag')
    # Getting the type of 'h' (line 126)
    h_16661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 38), 'h', False)
    # Getting the type of 'period' (line 126)
    period_16662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 41), 'period', False)
    # Processing the call keyword arguments (line 126)
    kwargs_16663 = {}
    # Getting the type of 'tilbert' (line 126)
    tilbert_16658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 20), 'tilbert', False)
    # Calling tilbert(args, kwargs) (line 126)
    tilbert_call_result_16664 = invoke(stypy.reporting.localization.Localization(__file__, 126, 20), tilbert_16658, *[imag_16660, h_16661, period_16662], **kwargs_16663)
    
    # Applying the binary operator '*' (line 126)
    result_mul_16665 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 15), '*', complex_16657, tilbert_call_result_16664)
    
    # Applying the binary operator '+' (line 125)
    result_add_16666 = python_operator(stypy.reporting.localization.Localization(__file__, 125, 15), '+', tilbert_call_result_16656, result_mul_16665)
    
    # Assigning a type to the variable 'stypy_return_type' (line 125)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'stypy_return_type', result_add_16666)
    # SSA join for if statement (line 124)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 128)
    # Getting the type of 'period' (line 128)
    period_16667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 4), 'period')
    # Getting the type of 'None' (line 128)
    None_16668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 21), 'None')
    
    (may_be_16669, more_types_in_union_16670) = may_not_be_none(period_16667, None_16668)

    if may_be_16669:

        if more_types_in_union_16670:
            # Runtime conditional SSA (line 128)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a BinOp to a Name (line 129):
        
        # Assigning a BinOp to a Name (line 129):
        # Getting the type of 'h' (line 129)
        h_16671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 12), 'h')
        int_16672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 16), 'int')
        # Applying the binary operator '*' (line 129)
        result_mul_16673 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 12), '*', h_16671, int_16672)
        
        # Getting the type of 'pi' (line 129)
        pi_16674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 20), 'pi')
        # Applying the binary operator '*' (line 129)
        result_mul_16675 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 18), '*', result_mul_16673, pi_16674)
        
        # Getting the type of 'period' (line 129)
        period_16676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 25), 'period')
        # Applying the binary operator 'div' (line 129)
        result_div_16677 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 23), 'div', result_mul_16675, period_16676)
        
        # Assigning a type to the variable 'h' (line 129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'h', result_div_16677)

        if more_types_in_union_16670:
            # SSA join for if statement (line 128)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 131):
    
    # Assigning a Call to a Name (line 131):
    
    # Call to len(...): (line 131)
    # Processing the call arguments (line 131)
    # Getting the type of 'x' (line 131)
    x_16679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 12), 'x', False)
    # Processing the call keyword arguments (line 131)
    kwargs_16680 = {}
    # Getting the type of 'len' (line 131)
    len_16678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 8), 'len', False)
    # Calling len(args, kwargs) (line 131)
    len_call_result_16681 = invoke(stypy.reporting.localization.Localization(__file__, 131, 8), len_16678, *[x_16679], **kwargs_16680)
    
    # Assigning a type to the variable 'n' (line 131)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 4), 'n', len_call_result_16681)
    
    # Assigning a Call to a Name (line 132):
    
    # Assigning a Call to a Name (line 132):
    
    # Call to get(...): (line 132)
    # Processing the call arguments (line 132)
    
    # Obtaining an instance of the builtin type 'tuple' (line 132)
    tuple_16684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 24), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 132)
    # Adding element type (line 132)
    # Getting the type of 'n' (line 132)
    n_16685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 24), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 24), tuple_16684, n_16685)
    # Adding element type (line 132)
    # Getting the type of 'h' (line 132)
    h_16686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 27), 'h', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 24), tuple_16684, h_16686)
    
    # Processing the call keyword arguments (line 132)
    kwargs_16687 = {}
    # Getting the type of '_cache' (line 132)
    _cache_16682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 12), '_cache', False)
    # Obtaining the member 'get' of a type (line 132)
    get_16683 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 12), _cache_16682, 'get')
    # Calling get(args, kwargs) (line 132)
    get_call_result_16688 = invoke(stypy.reporting.localization.Localization(__file__, 132, 12), get_16683, *[tuple_16684], **kwargs_16687)
    
    # Assigning a type to the variable 'omega' (line 132)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 4), 'omega', get_call_result_16688)
    
    # Type idiom detected: calculating its left and rigth part (line 133)
    # Getting the type of 'omega' (line 133)
    omega_16689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 7), 'omega')
    # Getting the type of 'None' (line 133)
    None_16690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 16), 'None')
    
    (may_be_16691, more_types_in_union_16692) = may_be_none(omega_16689, None_16690)

    if may_be_16691:

        if more_types_in_union_16692:
            # Runtime conditional SSA (line 133)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        
        
        # Call to len(...): (line 134)
        # Processing the call arguments (line 134)
        # Getting the type of '_cache' (line 134)
        _cache_16694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 15), '_cache', False)
        # Processing the call keyword arguments (line 134)
        kwargs_16695 = {}
        # Getting the type of 'len' (line 134)
        len_16693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 11), 'len', False)
        # Calling len(args, kwargs) (line 134)
        len_call_result_16696 = invoke(stypy.reporting.localization.Localization(__file__, 134, 11), len_16693, *[_cache_16694], **kwargs_16695)
        
        int_16697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 25), 'int')
        # Applying the binary operator '>' (line 134)
        result_gt_16698 = python_operator(stypy.reporting.localization.Localization(__file__, 134, 11), '>', len_call_result_16696, int_16697)
        
        # Testing the type of an if condition (line 134)
        if_condition_16699 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 134, 8), result_gt_16698)
        # Assigning a type to the variable 'if_condition_16699' (line 134)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'if_condition_16699', if_condition_16699)
        # SSA begins for if statement (line 134)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of '_cache' (line 135)
        _cache_16700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 18), '_cache')
        # Testing the type of an if condition (line 135)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 135, 12), _cache_16700)
        # SSA begins for while statement (line 135)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Call to popitem(...): (line 136)
        # Processing the call keyword arguments (line 136)
        kwargs_16703 = {}
        # Getting the type of '_cache' (line 136)
        _cache_16701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 16), '_cache', False)
        # Obtaining the member 'popitem' of a type (line 136)
        popitem_16702 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 16), _cache_16701, 'popitem')
        # Calling popitem(args, kwargs) (line 136)
        popitem_call_result_16704 = invoke(stypy.reporting.localization.Localization(__file__, 136, 16), popitem_16702, *[], **kwargs_16703)
        
        # SSA join for while statement (line 135)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 134)
        module_type_store = module_type_store.join_ssa_context()
        

        @norecursion
        def kernel(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            # Getting the type of 'h' (line 138)
            h_16705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 24), 'h')
            defaults = [h_16705]
            # Create a new context for function 'kernel'
            module_type_store = module_type_store.open_function_context('kernel', 138, 8, False)
            
            # Passed parameters checking function
            kernel.stypy_localization = localization
            kernel.stypy_type_of_self = None
            kernel.stypy_type_store = module_type_store
            kernel.stypy_function_name = 'kernel'
            kernel.stypy_param_names_list = ['k', 'h']
            kernel.stypy_varargs_param_name = None
            kernel.stypy_kwargs_param_name = None
            kernel.stypy_call_defaults = defaults
            kernel.stypy_call_varargs = varargs
            kernel.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'kernel', ['k', 'h'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'kernel', localization, ['k', 'h'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'kernel(...)' code ##################

            
            # Getting the type of 'k' (line 139)
            k_16706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 15), 'k')
            # Testing the type of an if condition (line 139)
            if_condition_16707 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 139, 12), k_16706)
            # Assigning a type to the variable 'if_condition_16707' (line 139)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 12), 'if_condition_16707', if_condition_16707)
            # SSA begins for if statement (line 139)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            float_16708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 23), 'float')
            
            # Call to tanh(...): (line 140)
            # Processing the call arguments (line 140)
            # Getting the type of 'h' (line 140)
            h_16710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 32), 'h', False)
            # Getting the type of 'k' (line 140)
            k_16711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 34), 'k', False)
            # Applying the binary operator '*' (line 140)
            result_mul_16712 = python_operator(stypy.reporting.localization.Localization(__file__, 140, 32), '*', h_16710, k_16711)
            
            # Processing the call keyword arguments (line 140)
            kwargs_16713 = {}
            # Getting the type of 'tanh' (line 140)
            tanh_16709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 27), 'tanh', False)
            # Calling tanh(args, kwargs) (line 140)
            tanh_call_result_16714 = invoke(stypy.reporting.localization.Localization(__file__, 140, 27), tanh_16709, *[result_mul_16712], **kwargs_16713)
            
            # Applying the binary operator 'div' (line 140)
            result_div_16715 = python_operator(stypy.reporting.localization.Localization(__file__, 140, 23), 'div', float_16708, tanh_call_result_16714)
            
            # Assigning a type to the variable 'stypy_return_type' (line 140)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 16), 'stypy_return_type', result_div_16715)
            # SSA join for if statement (line 139)
            module_type_store = module_type_store.join_ssa_context()
            
            int_16716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 19), 'int')
            # Assigning a type to the variable 'stypy_return_type' (line 142)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 12), 'stypy_return_type', int_16716)
            
            # ################# End of 'kernel(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'kernel' in the type store
            # Getting the type of 'stypy_return_type' (line 138)
            stypy_return_type_16717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_16717)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'kernel'
            return stypy_return_type_16717

        # Assigning a type to the variable 'kernel' (line 138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 8), 'kernel', kernel)
        
        # Assigning a Call to a Name (line 144):
        
        # Assigning a Call to a Name (line 144):
        
        # Call to init_convolution_kernel(...): (line 144)
        # Processing the call arguments (line 144)
        # Getting the type of 'n' (line 144)
        n_16720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 49), 'n', False)
        # Getting the type of 'kernel' (line 144)
        kernel_16721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 52), 'kernel', False)
        # Processing the call keyword arguments (line 144)
        int_16722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 62), 'int')
        keyword_16723 = int_16722
        kwargs_16724 = {'d': keyword_16723}
        # Getting the type of 'convolve' (line 144)
        convolve_16718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 16), 'convolve', False)
        # Obtaining the member 'init_convolution_kernel' of a type (line 144)
        init_convolution_kernel_16719 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 16), convolve_16718, 'init_convolution_kernel')
        # Calling init_convolution_kernel(args, kwargs) (line 144)
        init_convolution_kernel_call_result_16725 = invoke(stypy.reporting.localization.Localization(__file__, 144, 16), init_convolution_kernel_16719, *[n_16720, kernel_16721], **kwargs_16724)
        
        # Assigning a type to the variable 'omega' (line 144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'omega', init_convolution_kernel_call_result_16725)
        
        # Assigning a Name to a Subscript (line 145):
        
        # Assigning a Name to a Subscript (line 145):
        # Getting the type of 'omega' (line 145)
        omega_16726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 24), 'omega')
        # Getting the type of '_cache' (line 145)
        _cache_16727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 8), '_cache')
        
        # Obtaining an instance of the builtin type 'tuple' (line 145)
        tuple_16728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 16), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 145)
        # Adding element type (line 145)
        # Getting the type of 'n' (line 145)
        n_16729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 16), 'n')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 16), tuple_16728, n_16729)
        # Adding element type (line 145)
        # Getting the type of 'h' (line 145)
        h_16730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 18), 'h')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 16), tuple_16728, h_16730)
        
        # Storing an element on a container (line 145)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 8), _cache_16727, (tuple_16728, omega_16726))

        if more_types_in_union_16692:
            # SSA join for if statement (line 133)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 147):
    
    # Assigning a Call to a Name (line 147):
    
    # Call to _datacopied(...): (line 147)
    # Processing the call arguments (line 147)
    # Getting the type of 'tmp' (line 147)
    tmp_16732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 30), 'tmp', False)
    # Getting the type of 'x' (line 147)
    x_16733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 35), 'x', False)
    # Processing the call keyword arguments (line 147)
    kwargs_16734 = {}
    # Getting the type of '_datacopied' (line 147)
    _datacopied_16731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 18), '_datacopied', False)
    # Calling _datacopied(args, kwargs) (line 147)
    _datacopied_call_result_16735 = invoke(stypy.reporting.localization.Localization(__file__, 147, 18), _datacopied_16731, *[tmp_16732, x_16733], **kwargs_16734)
    
    # Assigning a type to the variable 'overwrite_x' (line 147)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 4), 'overwrite_x', _datacopied_call_result_16735)
    
    # Call to convolve(...): (line 148)
    # Processing the call arguments (line 148)
    # Getting the type of 'tmp' (line 148)
    tmp_16738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 29), 'tmp', False)
    # Getting the type of 'omega' (line 148)
    omega_16739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 33), 'omega', False)
    # Processing the call keyword arguments (line 148)
    int_16740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 54), 'int')
    keyword_16741 = int_16740
    # Getting the type of 'overwrite_x' (line 148)
    overwrite_x_16742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 68), 'overwrite_x', False)
    keyword_16743 = overwrite_x_16742
    kwargs_16744 = {'overwrite_x': keyword_16743, 'swap_real_imag': keyword_16741}
    # Getting the type of 'convolve' (line 148)
    convolve_16736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 11), 'convolve', False)
    # Obtaining the member 'convolve' of a type (line 148)
    convolve_16737 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 11), convolve_16736, 'convolve')
    # Calling convolve(args, kwargs) (line 148)
    convolve_call_result_16745 = invoke(stypy.reporting.localization.Localization(__file__, 148, 11), convolve_16737, *[tmp_16738, omega_16739], **kwargs_16744)
    
    # Assigning a type to the variable 'stypy_return_type' (line 148)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 4), 'stypy_return_type', convolve_call_result_16745)
    
    # ################# End of 'tilbert(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'tilbert' in the type store
    # Getting the type of 'stypy_return_type' (line 87)
    stypy_return_type_16746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_16746)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'tilbert'
    return stypy_return_type_16746

# Assigning a type to the variable 'tilbert' (line 87)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 0), 'tilbert', tilbert)
# Deleting a member
module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 149, 0), module_type_store, '_cache')

# Assigning a Dict to a Name (line 152):

# Assigning a Dict to a Name (line 152):

# Obtaining an instance of the builtin type 'dict' (line 152)
dict_16747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 9), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 152)

# Assigning a type to the variable '_cache' (line 152)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 0), '_cache', dict_16747)

@norecursion
def itilbert(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 155)
    None_16748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 24), 'None')
    # Getting the type of '_cache' (line 155)
    _cache_16749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 37), '_cache')
    defaults = [None_16748, _cache_16749]
    # Create a new context for function 'itilbert'
    module_type_store = module_type_store.open_function_context('itilbert', 155, 0, False)
    
    # Passed parameters checking function
    itilbert.stypy_localization = localization
    itilbert.stypy_type_of_self = None
    itilbert.stypy_type_store = module_type_store
    itilbert.stypy_function_name = 'itilbert'
    itilbert.stypy_param_names_list = ['x', 'h', 'period', '_cache']
    itilbert.stypy_varargs_param_name = None
    itilbert.stypy_kwargs_param_name = None
    itilbert.stypy_call_defaults = defaults
    itilbert.stypy_call_varargs = varargs
    itilbert.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'itilbert', ['x', 'h', 'period', '_cache'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'itilbert', localization, ['x', 'h', 'period', '_cache'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'itilbert(...)' code ##################

    str_16750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, (-1)), 'str', '\n    Return inverse h-Tilbert transform of a periodic sequence x.\n\n    If ``x_j`` and ``y_j`` are Fourier coefficients of periodic functions x\n    and y, respectively, then::\n\n      y_j = -sqrt(-1)*tanh(j*h*2*pi/period) * x_j\n      y_0 = 0\n\n    For more details, see `tilbert`.\n\n    ')
    
    # Assigning a Call to a Name (line 168):
    
    # Assigning a Call to a Name (line 168):
    
    # Call to asarray(...): (line 168)
    # Processing the call arguments (line 168)
    # Getting the type of 'x' (line 168)
    x_16752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 18), 'x', False)
    # Processing the call keyword arguments (line 168)
    kwargs_16753 = {}
    # Getting the type of 'asarray' (line 168)
    asarray_16751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 10), 'asarray', False)
    # Calling asarray(args, kwargs) (line 168)
    asarray_call_result_16754 = invoke(stypy.reporting.localization.Localization(__file__, 168, 10), asarray_16751, *[x_16752], **kwargs_16753)
    
    # Assigning a type to the variable 'tmp' (line 168)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 4), 'tmp', asarray_call_result_16754)
    
    
    # Call to iscomplexobj(...): (line 169)
    # Processing the call arguments (line 169)
    # Getting the type of 'tmp' (line 169)
    tmp_16756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 20), 'tmp', False)
    # Processing the call keyword arguments (line 169)
    kwargs_16757 = {}
    # Getting the type of 'iscomplexobj' (line 169)
    iscomplexobj_16755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 7), 'iscomplexobj', False)
    # Calling iscomplexobj(args, kwargs) (line 169)
    iscomplexobj_call_result_16758 = invoke(stypy.reporting.localization.Localization(__file__, 169, 7), iscomplexobj_16755, *[tmp_16756], **kwargs_16757)
    
    # Testing the type of an if condition (line 169)
    if_condition_16759 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 169, 4), iscomplexobj_call_result_16758)
    # Assigning a type to the variable 'if_condition_16759' (line 169)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 4), 'if_condition_16759', if_condition_16759)
    # SSA begins for if statement (line 169)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to itilbert(...): (line 170)
    # Processing the call arguments (line 170)
    # Getting the type of 'tmp' (line 170)
    tmp_16761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 24), 'tmp', False)
    # Obtaining the member 'real' of a type (line 170)
    real_16762 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 24), tmp_16761, 'real')
    # Getting the type of 'h' (line 170)
    h_16763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 33), 'h', False)
    # Getting the type of 'period' (line 170)
    period_16764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 35), 'period', False)
    # Processing the call keyword arguments (line 170)
    kwargs_16765 = {}
    # Getting the type of 'itilbert' (line 170)
    itilbert_16760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 15), 'itilbert', False)
    # Calling itilbert(args, kwargs) (line 170)
    itilbert_call_result_16766 = invoke(stypy.reporting.localization.Localization(__file__, 170, 15), itilbert_16760, *[real_16762, h_16763, period_16764], **kwargs_16765)
    
    complex_16767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 15), 'complex')
    
    # Call to itilbert(...): (line 171)
    # Processing the call arguments (line 171)
    # Getting the type of 'tmp' (line 171)
    tmp_16769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 27), 'tmp', False)
    # Obtaining the member 'imag' of a type (line 171)
    imag_16770 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 27), tmp_16769, 'imag')
    # Getting the type of 'h' (line 171)
    h_16771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 36), 'h', False)
    # Getting the type of 'period' (line 171)
    period_16772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 38), 'period', False)
    # Processing the call keyword arguments (line 171)
    kwargs_16773 = {}
    # Getting the type of 'itilbert' (line 171)
    itilbert_16768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 18), 'itilbert', False)
    # Calling itilbert(args, kwargs) (line 171)
    itilbert_call_result_16774 = invoke(stypy.reporting.localization.Localization(__file__, 171, 18), itilbert_16768, *[imag_16770, h_16771, period_16772], **kwargs_16773)
    
    # Applying the binary operator '*' (line 171)
    result_mul_16775 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 15), '*', complex_16767, itilbert_call_result_16774)
    
    # Applying the binary operator '+' (line 170)
    result_add_16776 = python_operator(stypy.reporting.localization.Localization(__file__, 170, 15), '+', itilbert_call_result_16766, result_mul_16775)
    
    # Assigning a type to the variable 'stypy_return_type' (line 170)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 8), 'stypy_return_type', result_add_16776)
    # SSA join for if statement (line 169)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 172)
    # Getting the type of 'period' (line 172)
    period_16777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 4), 'period')
    # Getting the type of 'None' (line 172)
    None_16778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 21), 'None')
    
    (may_be_16779, more_types_in_union_16780) = may_not_be_none(period_16777, None_16778)

    if may_be_16779:

        if more_types_in_union_16780:
            # Runtime conditional SSA (line 172)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a BinOp to a Name (line 173):
        
        # Assigning a BinOp to a Name (line 173):
        # Getting the type of 'h' (line 173)
        h_16781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 12), 'h')
        int_16782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 14), 'int')
        # Applying the binary operator '*' (line 173)
        result_mul_16783 = python_operator(stypy.reporting.localization.Localization(__file__, 173, 12), '*', h_16781, int_16782)
        
        # Getting the type of 'pi' (line 173)
        pi_16784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 16), 'pi')
        # Applying the binary operator '*' (line 173)
        result_mul_16785 = python_operator(stypy.reporting.localization.Localization(__file__, 173, 15), '*', result_mul_16783, pi_16784)
        
        # Getting the type of 'period' (line 173)
        period_16786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 19), 'period')
        # Applying the binary operator 'div' (line 173)
        result_div_16787 = python_operator(stypy.reporting.localization.Localization(__file__, 173, 18), 'div', result_mul_16785, period_16786)
        
        # Assigning a type to the variable 'h' (line 173)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 8), 'h', result_div_16787)

        if more_types_in_union_16780:
            # SSA join for if statement (line 172)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 174):
    
    # Assigning a Call to a Name (line 174):
    
    # Call to len(...): (line 174)
    # Processing the call arguments (line 174)
    # Getting the type of 'x' (line 174)
    x_16789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 12), 'x', False)
    # Processing the call keyword arguments (line 174)
    kwargs_16790 = {}
    # Getting the type of 'len' (line 174)
    len_16788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'len', False)
    # Calling len(args, kwargs) (line 174)
    len_call_result_16791 = invoke(stypy.reporting.localization.Localization(__file__, 174, 8), len_16788, *[x_16789], **kwargs_16790)
    
    # Assigning a type to the variable 'n' (line 174)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 4), 'n', len_call_result_16791)
    
    # Assigning a Call to a Name (line 175):
    
    # Assigning a Call to a Name (line 175):
    
    # Call to get(...): (line 175)
    # Processing the call arguments (line 175)
    
    # Obtaining an instance of the builtin type 'tuple' (line 175)
    tuple_16794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 24), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 175)
    # Adding element type (line 175)
    # Getting the type of 'n' (line 175)
    n_16795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 24), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 175, 24), tuple_16794, n_16795)
    # Adding element type (line 175)
    # Getting the type of 'h' (line 175)
    h_16796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 26), 'h', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 175, 24), tuple_16794, h_16796)
    
    # Processing the call keyword arguments (line 175)
    kwargs_16797 = {}
    # Getting the type of '_cache' (line 175)
    _cache_16792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 12), '_cache', False)
    # Obtaining the member 'get' of a type (line 175)
    get_16793 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 12), _cache_16792, 'get')
    # Calling get(args, kwargs) (line 175)
    get_call_result_16798 = invoke(stypy.reporting.localization.Localization(__file__, 175, 12), get_16793, *[tuple_16794], **kwargs_16797)
    
    # Assigning a type to the variable 'omega' (line 175)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 4), 'omega', get_call_result_16798)
    
    # Type idiom detected: calculating its left and rigth part (line 176)
    # Getting the type of 'omega' (line 176)
    omega_16799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 7), 'omega')
    # Getting the type of 'None' (line 176)
    None_16800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 16), 'None')
    
    (may_be_16801, more_types_in_union_16802) = may_be_none(omega_16799, None_16800)

    if may_be_16801:

        if more_types_in_union_16802:
            # Runtime conditional SSA (line 176)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        
        
        # Call to len(...): (line 177)
        # Processing the call arguments (line 177)
        # Getting the type of '_cache' (line 177)
        _cache_16804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 15), '_cache', False)
        # Processing the call keyword arguments (line 177)
        kwargs_16805 = {}
        # Getting the type of 'len' (line 177)
        len_16803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 11), 'len', False)
        # Calling len(args, kwargs) (line 177)
        len_call_result_16806 = invoke(stypy.reporting.localization.Localization(__file__, 177, 11), len_16803, *[_cache_16804], **kwargs_16805)
        
        int_16807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 25), 'int')
        # Applying the binary operator '>' (line 177)
        result_gt_16808 = python_operator(stypy.reporting.localization.Localization(__file__, 177, 11), '>', len_call_result_16806, int_16807)
        
        # Testing the type of an if condition (line 177)
        if_condition_16809 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 177, 8), result_gt_16808)
        # Assigning a type to the variable 'if_condition_16809' (line 177)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'if_condition_16809', if_condition_16809)
        # SSA begins for if statement (line 177)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of '_cache' (line 178)
        _cache_16810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 18), '_cache')
        # Testing the type of an if condition (line 178)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 178, 12), _cache_16810)
        # SSA begins for while statement (line 178)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Call to popitem(...): (line 179)
        # Processing the call keyword arguments (line 179)
        kwargs_16813 = {}
        # Getting the type of '_cache' (line 179)
        _cache_16811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 16), '_cache', False)
        # Obtaining the member 'popitem' of a type (line 179)
        popitem_16812 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 16), _cache_16811, 'popitem')
        # Calling popitem(args, kwargs) (line 179)
        popitem_call_result_16814 = invoke(stypy.reporting.localization.Localization(__file__, 179, 16), popitem_16812, *[], **kwargs_16813)
        
        # SSA join for while statement (line 178)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 177)
        module_type_store = module_type_store.join_ssa_context()
        

        @norecursion
        def kernel(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            # Getting the type of 'h' (line 181)
            h_16815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 23), 'h')
            defaults = [h_16815]
            # Create a new context for function 'kernel'
            module_type_store = module_type_store.open_function_context('kernel', 181, 8, False)
            
            # Passed parameters checking function
            kernel.stypy_localization = localization
            kernel.stypy_type_of_self = None
            kernel.stypy_type_store = module_type_store
            kernel.stypy_function_name = 'kernel'
            kernel.stypy_param_names_list = ['k', 'h']
            kernel.stypy_varargs_param_name = None
            kernel.stypy_kwargs_param_name = None
            kernel.stypy_call_defaults = defaults
            kernel.stypy_call_varargs = varargs
            kernel.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'kernel', ['k', 'h'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'kernel', localization, ['k', 'h'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'kernel(...)' code ##################

            
            # Getting the type of 'k' (line 182)
            k_16816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 15), 'k')
            # Testing the type of an if condition (line 182)
            if_condition_16817 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 182, 12), k_16816)
            # Assigning a type to the variable 'if_condition_16817' (line 182)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 12), 'if_condition_16817', if_condition_16817)
            # SSA begins for if statement (line 182)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            
            # Call to tanh(...): (line 183)
            # Processing the call arguments (line 183)
            # Getting the type of 'h' (line 183)
            h_16819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 29), 'h', False)
            # Getting the type of 'k' (line 183)
            k_16820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 31), 'k', False)
            # Applying the binary operator '*' (line 183)
            result_mul_16821 = python_operator(stypy.reporting.localization.Localization(__file__, 183, 29), '*', h_16819, k_16820)
            
            # Processing the call keyword arguments (line 183)
            kwargs_16822 = {}
            # Getting the type of 'tanh' (line 183)
            tanh_16818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 24), 'tanh', False)
            # Calling tanh(args, kwargs) (line 183)
            tanh_call_result_16823 = invoke(stypy.reporting.localization.Localization(__file__, 183, 24), tanh_16818, *[result_mul_16821], **kwargs_16822)
            
            # Applying the 'usub' unary operator (line 183)
            result___neg___16824 = python_operator(stypy.reporting.localization.Localization(__file__, 183, 23), 'usub', tanh_call_result_16823)
            
            # Assigning a type to the variable 'stypy_return_type' (line 183)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 16), 'stypy_return_type', result___neg___16824)
            # SSA join for if statement (line 182)
            module_type_store = module_type_store.join_ssa_context()
            
            int_16825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 19), 'int')
            # Assigning a type to the variable 'stypy_return_type' (line 184)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 12), 'stypy_return_type', int_16825)
            
            # ################# End of 'kernel(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'kernel' in the type store
            # Getting the type of 'stypy_return_type' (line 181)
            stypy_return_type_16826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_16826)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'kernel'
            return stypy_return_type_16826

        # Assigning a type to the variable 'kernel' (line 181)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'kernel', kernel)
        
        # Assigning a Call to a Name (line 185):
        
        # Assigning a Call to a Name (line 185):
        
        # Call to init_convolution_kernel(...): (line 185)
        # Processing the call arguments (line 185)
        # Getting the type of 'n' (line 185)
        n_16829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 49), 'n', False)
        # Getting the type of 'kernel' (line 185)
        kernel_16830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 51), 'kernel', False)
        # Processing the call keyword arguments (line 185)
        int_16831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 60), 'int')
        keyword_16832 = int_16831
        kwargs_16833 = {'d': keyword_16832}
        # Getting the type of 'convolve' (line 185)
        convolve_16827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 16), 'convolve', False)
        # Obtaining the member 'init_convolution_kernel' of a type (line 185)
        init_convolution_kernel_16828 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 16), convolve_16827, 'init_convolution_kernel')
        # Calling init_convolution_kernel(args, kwargs) (line 185)
        init_convolution_kernel_call_result_16834 = invoke(stypy.reporting.localization.Localization(__file__, 185, 16), init_convolution_kernel_16828, *[n_16829, kernel_16830], **kwargs_16833)
        
        # Assigning a type to the variable 'omega' (line 185)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 8), 'omega', init_convolution_kernel_call_result_16834)
        
        # Assigning a Name to a Subscript (line 186):
        
        # Assigning a Name to a Subscript (line 186):
        # Getting the type of 'omega' (line 186)
        omega_16835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 24), 'omega')
        # Getting the type of '_cache' (line 186)
        _cache_16836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 8), '_cache')
        
        # Obtaining an instance of the builtin type 'tuple' (line 186)
        tuple_16837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 16), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 186)
        # Adding element type (line 186)
        # Getting the type of 'n' (line 186)
        n_16838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 16), 'n')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 186, 16), tuple_16837, n_16838)
        # Adding element type (line 186)
        # Getting the type of 'h' (line 186)
        h_16839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 18), 'h')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 186, 16), tuple_16837, h_16839)
        
        # Storing an element on a container (line 186)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 186, 8), _cache_16836, (tuple_16837, omega_16835))

        if more_types_in_union_16802:
            # SSA join for if statement (line 176)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 187):
    
    # Assigning a Call to a Name (line 187):
    
    # Call to _datacopied(...): (line 187)
    # Processing the call arguments (line 187)
    # Getting the type of 'tmp' (line 187)
    tmp_16841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 30), 'tmp', False)
    # Getting the type of 'x' (line 187)
    x_16842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 35), 'x', False)
    # Processing the call keyword arguments (line 187)
    kwargs_16843 = {}
    # Getting the type of '_datacopied' (line 187)
    _datacopied_16840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 18), '_datacopied', False)
    # Calling _datacopied(args, kwargs) (line 187)
    _datacopied_call_result_16844 = invoke(stypy.reporting.localization.Localization(__file__, 187, 18), _datacopied_16840, *[tmp_16841, x_16842], **kwargs_16843)
    
    # Assigning a type to the variable 'overwrite_x' (line 187)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 4), 'overwrite_x', _datacopied_call_result_16844)
    
    # Call to convolve(...): (line 188)
    # Processing the call arguments (line 188)
    # Getting the type of 'tmp' (line 188)
    tmp_16847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 29), 'tmp', False)
    # Getting the type of 'omega' (line 188)
    omega_16848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 33), 'omega', False)
    # Processing the call keyword arguments (line 188)
    int_16849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 54), 'int')
    keyword_16850 = int_16849
    # Getting the type of 'overwrite_x' (line 188)
    overwrite_x_16851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 68), 'overwrite_x', False)
    keyword_16852 = overwrite_x_16851
    kwargs_16853 = {'overwrite_x': keyword_16852, 'swap_real_imag': keyword_16850}
    # Getting the type of 'convolve' (line 188)
    convolve_16845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 11), 'convolve', False)
    # Obtaining the member 'convolve' of a type (line 188)
    convolve_16846 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 11), convolve_16845, 'convolve')
    # Calling convolve(args, kwargs) (line 188)
    convolve_call_result_16854 = invoke(stypy.reporting.localization.Localization(__file__, 188, 11), convolve_16846, *[tmp_16847, omega_16848], **kwargs_16853)
    
    # Assigning a type to the variable 'stypy_return_type' (line 188)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 4), 'stypy_return_type', convolve_call_result_16854)
    
    # ################# End of 'itilbert(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'itilbert' in the type store
    # Getting the type of 'stypy_return_type' (line 155)
    stypy_return_type_16855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_16855)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'itilbert'
    return stypy_return_type_16855

# Assigning a type to the variable 'itilbert' (line 155)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 0), 'itilbert', itilbert)
# Deleting a member
module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 189, 0), module_type_store, '_cache')

# Assigning a Dict to a Name (line 192):

# Assigning a Dict to a Name (line 192):

# Obtaining an instance of the builtin type 'dict' (line 192)
dict_16856 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 9), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 192)

# Assigning a type to the variable '_cache' (line 192)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 0), '_cache', dict_16856)

@norecursion
def hilbert(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of '_cache' (line 195)
    _cache_16857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 22), '_cache')
    defaults = [_cache_16857]
    # Create a new context for function 'hilbert'
    module_type_store = module_type_store.open_function_context('hilbert', 195, 0, False)
    
    # Passed parameters checking function
    hilbert.stypy_localization = localization
    hilbert.stypy_type_of_self = None
    hilbert.stypy_type_store = module_type_store
    hilbert.stypy_function_name = 'hilbert'
    hilbert.stypy_param_names_list = ['x', '_cache']
    hilbert.stypy_varargs_param_name = None
    hilbert.stypy_kwargs_param_name = None
    hilbert.stypy_call_defaults = defaults
    hilbert.stypy_call_varargs = varargs
    hilbert.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'hilbert', ['x', '_cache'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'hilbert', localization, ['x', '_cache'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'hilbert(...)' code ##################

    str_16858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, (-1)), 'str', '\n    Return Hilbert transform of a periodic sequence x.\n\n    If x_j and y_j are Fourier coefficients of periodic functions x\n    and y, respectively, then::\n\n      y_j = sqrt(-1)*sign(j) * x_j\n      y_0 = 0\n\n    Parameters\n    ----------\n    x : array_like\n        The input array, should be periodic.\n    _cache : dict, optional\n        Dictionary that contains the kernel used to do a convolution with.\n\n    Returns\n    -------\n    y : ndarray\n        The transformed input.\n\n    See Also\n    --------\n    scipy.signal.hilbert : Compute the analytic signal, using the Hilbert\n                           transform.\n\n    Notes\n    -----\n    If ``sum(x, axis=0) == 0`` then ``hilbert(ihilbert(x)) == x``.\n\n    For even len(x), the Nyquist mode of x is taken zero.\n\n    The sign of the returned transform does not have a factor -1 that is more\n    often than not found in the definition of the Hilbert transform.  Note also\n    that `scipy.signal.hilbert` does have an extra -1 factor compared to this\n    function.\n\n    ')
    
    # Assigning a Call to a Name (line 234):
    
    # Assigning a Call to a Name (line 234):
    
    # Call to asarray(...): (line 234)
    # Processing the call arguments (line 234)
    # Getting the type of 'x' (line 234)
    x_16860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 18), 'x', False)
    # Processing the call keyword arguments (line 234)
    kwargs_16861 = {}
    # Getting the type of 'asarray' (line 234)
    asarray_16859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 10), 'asarray', False)
    # Calling asarray(args, kwargs) (line 234)
    asarray_call_result_16862 = invoke(stypy.reporting.localization.Localization(__file__, 234, 10), asarray_16859, *[x_16860], **kwargs_16861)
    
    # Assigning a type to the variable 'tmp' (line 234)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 4), 'tmp', asarray_call_result_16862)
    
    
    # Call to iscomplexobj(...): (line 235)
    # Processing the call arguments (line 235)
    # Getting the type of 'tmp' (line 235)
    tmp_16864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 20), 'tmp', False)
    # Processing the call keyword arguments (line 235)
    kwargs_16865 = {}
    # Getting the type of 'iscomplexobj' (line 235)
    iscomplexobj_16863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 7), 'iscomplexobj', False)
    # Calling iscomplexobj(args, kwargs) (line 235)
    iscomplexobj_call_result_16866 = invoke(stypy.reporting.localization.Localization(__file__, 235, 7), iscomplexobj_16863, *[tmp_16864], **kwargs_16865)
    
    # Testing the type of an if condition (line 235)
    if_condition_16867 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 235, 4), iscomplexobj_call_result_16866)
    # Assigning a type to the variable 'if_condition_16867' (line 235)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 4), 'if_condition_16867', if_condition_16867)
    # SSA begins for if statement (line 235)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to hilbert(...): (line 236)
    # Processing the call arguments (line 236)
    # Getting the type of 'tmp' (line 236)
    tmp_16869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 23), 'tmp', False)
    # Obtaining the member 'real' of a type (line 236)
    real_16870 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 23), tmp_16869, 'real')
    # Processing the call keyword arguments (line 236)
    kwargs_16871 = {}
    # Getting the type of 'hilbert' (line 236)
    hilbert_16868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 15), 'hilbert', False)
    # Calling hilbert(args, kwargs) (line 236)
    hilbert_call_result_16872 = invoke(stypy.reporting.localization.Localization(__file__, 236, 15), hilbert_16868, *[real_16870], **kwargs_16871)
    
    complex_16873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 33), 'complex')
    
    # Call to hilbert(...): (line 236)
    # Processing the call arguments (line 236)
    # Getting the type of 'tmp' (line 236)
    tmp_16875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 44), 'tmp', False)
    # Obtaining the member 'imag' of a type (line 236)
    imag_16876 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 44), tmp_16875, 'imag')
    # Processing the call keyword arguments (line 236)
    kwargs_16877 = {}
    # Getting the type of 'hilbert' (line 236)
    hilbert_16874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 36), 'hilbert', False)
    # Calling hilbert(args, kwargs) (line 236)
    hilbert_call_result_16878 = invoke(stypy.reporting.localization.Localization(__file__, 236, 36), hilbert_16874, *[imag_16876], **kwargs_16877)
    
    # Applying the binary operator '*' (line 236)
    result_mul_16879 = python_operator(stypy.reporting.localization.Localization(__file__, 236, 33), '*', complex_16873, hilbert_call_result_16878)
    
    # Applying the binary operator '+' (line 236)
    result_add_16880 = python_operator(stypy.reporting.localization.Localization(__file__, 236, 15), '+', hilbert_call_result_16872, result_mul_16879)
    
    # Assigning a type to the variable 'stypy_return_type' (line 236)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 8), 'stypy_return_type', result_add_16880)
    # SSA join for if statement (line 235)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 237):
    
    # Assigning a Call to a Name (line 237):
    
    # Call to len(...): (line 237)
    # Processing the call arguments (line 237)
    # Getting the type of 'x' (line 237)
    x_16882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 12), 'x', False)
    # Processing the call keyword arguments (line 237)
    kwargs_16883 = {}
    # Getting the type of 'len' (line 237)
    len_16881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 8), 'len', False)
    # Calling len(args, kwargs) (line 237)
    len_call_result_16884 = invoke(stypy.reporting.localization.Localization(__file__, 237, 8), len_16881, *[x_16882], **kwargs_16883)
    
    # Assigning a type to the variable 'n' (line 237)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 4), 'n', len_call_result_16884)
    
    # Assigning a Call to a Name (line 238):
    
    # Assigning a Call to a Name (line 238):
    
    # Call to get(...): (line 238)
    # Processing the call arguments (line 238)
    # Getting the type of 'n' (line 238)
    n_16887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 23), 'n', False)
    # Processing the call keyword arguments (line 238)
    kwargs_16888 = {}
    # Getting the type of '_cache' (line 238)
    _cache_16885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 12), '_cache', False)
    # Obtaining the member 'get' of a type (line 238)
    get_16886 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 12), _cache_16885, 'get')
    # Calling get(args, kwargs) (line 238)
    get_call_result_16889 = invoke(stypy.reporting.localization.Localization(__file__, 238, 12), get_16886, *[n_16887], **kwargs_16888)
    
    # Assigning a type to the variable 'omega' (line 238)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 4), 'omega', get_call_result_16889)
    
    # Type idiom detected: calculating its left and rigth part (line 239)
    # Getting the type of 'omega' (line 239)
    omega_16890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 7), 'omega')
    # Getting the type of 'None' (line 239)
    None_16891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 16), 'None')
    
    (may_be_16892, more_types_in_union_16893) = may_be_none(omega_16890, None_16891)

    if may_be_16892:

        if more_types_in_union_16893:
            # Runtime conditional SSA (line 239)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        
        
        # Call to len(...): (line 240)
        # Processing the call arguments (line 240)
        # Getting the type of '_cache' (line 240)
        _cache_16895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 15), '_cache', False)
        # Processing the call keyword arguments (line 240)
        kwargs_16896 = {}
        # Getting the type of 'len' (line 240)
        len_16894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 11), 'len', False)
        # Calling len(args, kwargs) (line 240)
        len_call_result_16897 = invoke(stypy.reporting.localization.Localization(__file__, 240, 11), len_16894, *[_cache_16895], **kwargs_16896)
        
        int_16898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 25), 'int')
        # Applying the binary operator '>' (line 240)
        result_gt_16899 = python_operator(stypy.reporting.localization.Localization(__file__, 240, 11), '>', len_call_result_16897, int_16898)
        
        # Testing the type of an if condition (line 240)
        if_condition_16900 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 240, 8), result_gt_16899)
        # Assigning a type to the variable 'if_condition_16900' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'if_condition_16900', if_condition_16900)
        # SSA begins for if statement (line 240)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of '_cache' (line 241)
        _cache_16901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 18), '_cache')
        # Testing the type of an if condition (line 241)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 241, 12), _cache_16901)
        # SSA begins for while statement (line 241)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Call to popitem(...): (line 242)
        # Processing the call keyword arguments (line 242)
        kwargs_16904 = {}
        # Getting the type of '_cache' (line 242)
        _cache_16902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 16), '_cache', False)
        # Obtaining the member 'popitem' of a type (line 242)
        popitem_16903 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 16), _cache_16902, 'popitem')
        # Calling popitem(args, kwargs) (line 242)
        popitem_call_result_16905 = invoke(stypy.reporting.localization.Localization(__file__, 242, 16), popitem_16903, *[], **kwargs_16904)
        
        # SSA join for while statement (line 241)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 240)
        module_type_store = module_type_store.join_ssa_context()
        

        @norecursion
        def kernel(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'kernel'
            module_type_store = module_type_store.open_function_context('kernel', 244, 8, False)
            
            # Passed parameters checking function
            kernel.stypy_localization = localization
            kernel.stypy_type_of_self = None
            kernel.stypy_type_store = module_type_store
            kernel.stypy_function_name = 'kernel'
            kernel.stypy_param_names_list = ['k']
            kernel.stypy_varargs_param_name = None
            kernel.stypy_kwargs_param_name = None
            kernel.stypy_call_defaults = defaults
            kernel.stypy_call_varargs = varargs
            kernel.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'kernel', ['k'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'kernel', localization, ['k'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'kernel(...)' code ##################

            
            
            # Getting the type of 'k' (line 245)
            k_16906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 15), 'k')
            int_16907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 19), 'int')
            # Applying the binary operator '>' (line 245)
            result_gt_16908 = python_operator(stypy.reporting.localization.Localization(__file__, 245, 15), '>', k_16906, int_16907)
            
            # Testing the type of an if condition (line 245)
            if_condition_16909 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 245, 12), result_gt_16908)
            # Assigning a type to the variable 'if_condition_16909' (line 245)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 12), 'if_condition_16909', if_condition_16909)
            # SSA begins for if statement (line 245)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            float_16910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 23), 'float')
            # Assigning a type to the variable 'stypy_return_type' (line 246)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 16), 'stypy_return_type', float_16910)
            # SSA branch for the else part of an if statement (line 245)
            module_type_store.open_ssa_branch('else')
            
            
            # Getting the type of 'k' (line 247)
            k_16911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 17), 'k')
            int_16912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 21), 'int')
            # Applying the binary operator '<' (line 247)
            result_lt_16913 = python_operator(stypy.reporting.localization.Localization(__file__, 247, 17), '<', k_16911, int_16912)
            
            # Testing the type of an if condition (line 247)
            if_condition_16914 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 247, 17), result_lt_16913)
            # Assigning a type to the variable 'if_condition_16914' (line 247)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 17), 'if_condition_16914', if_condition_16914)
            # SSA begins for if statement (line 247)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            float_16915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 23), 'float')
            # Assigning a type to the variable 'stypy_return_type' (line 248)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 16), 'stypy_return_type', float_16915)
            # SSA join for if statement (line 247)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 245)
            module_type_store = module_type_store.join_ssa_context()
            
            float_16916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 19), 'float')
            # Assigning a type to the variable 'stypy_return_type' (line 249)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 12), 'stypy_return_type', float_16916)
            
            # ################# End of 'kernel(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'kernel' in the type store
            # Getting the type of 'stypy_return_type' (line 244)
            stypy_return_type_16917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_16917)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'kernel'
            return stypy_return_type_16917

        # Assigning a type to the variable 'kernel' (line 244)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 8), 'kernel', kernel)
        
        # Assigning a Call to a Name (line 250):
        
        # Assigning a Call to a Name (line 250):
        
        # Call to init_convolution_kernel(...): (line 250)
        # Processing the call arguments (line 250)
        # Getting the type of 'n' (line 250)
        n_16920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 49), 'n', False)
        # Getting the type of 'kernel' (line 250)
        kernel_16921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 51), 'kernel', False)
        # Processing the call keyword arguments (line 250)
        int_16922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 60), 'int')
        keyword_16923 = int_16922
        kwargs_16924 = {'d': keyword_16923}
        # Getting the type of 'convolve' (line 250)
        convolve_16918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 16), 'convolve', False)
        # Obtaining the member 'init_convolution_kernel' of a type (line 250)
        init_convolution_kernel_16919 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 16), convolve_16918, 'init_convolution_kernel')
        # Calling init_convolution_kernel(args, kwargs) (line 250)
        init_convolution_kernel_call_result_16925 = invoke(stypy.reporting.localization.Localization(__file__, 250, 16), init_convolution_kernel_16919, *[n_16920, kernel_16921], **kwargs_16924)
        
        # Assigning a type to the variable 'omega' (line 250)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 8), 'omega', init_convolution_kernel_call_result_16925)
        
        # Assigning a Name to a Subscript (line 251):
        
        # Assigning a Name to a Subscript (line 251):
        # Getting the type of 'omega' (line 251)
        omega_16926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 20), 'omega')
        # Getting the type of '_cache' (line 251)
        _cache_16927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 8), '_cache')
        # Getting the type of 'n' (line 251)
        n_16928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 15), 'n')
        # Storing an element on a container (line 251)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 8), _cache_16927, (n_16928, omega_16926))

        if more_types_in_union_16893:
            # SSA join for if statement (line 239)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 252):
    
    # Assigning a Call to a Name (line 252):
    
    # Call to _datacopied(...): (line 252)
    # Processing the call arguments (line 252)
    # Getting the type of 'tmp' (line 252)
    tmp_16930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 30), 'tmp', False)
    # Getting the type of 'x' (line 252)
    x_16931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 35), 'x', False)
    # Processing the call keyword arguments (line 252)
    kwargs_16932 = {}
    # Getting the type of '_datacopied' (line 252)
    _datacopied_16929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 18), '_datacopied', False)
    # Calling _datacopied(args, kwargs) (line 252)
    _datacopied_call_result_16933 = invoke(stypy.reporting.localization.Localization(__file__, 252, 18), _datacopied_16929, *[tmp_16930, x_16931], **kwargs_16932)
    
    # Assigning a type to the variable 'overwrite_x' (line 252)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 4), 'overwrite_x', _datacopied_call_result_16933)
    
    # Call to convolve(...): (line 253)
    # Processing the call arguments (line 253)
    # Getting the type of 'tmp' (line 253)
    tmp_16936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 29), 'tmp', False)
    # Getting the type of 'omega' (line 253)
    omega_16937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 33), 'omega', False)
    # Processing the call keyword arguments (line 253)
    int_16938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 54), 'int')
    keyword_16939 = int_16938
    # Getting the type of 'overwrite_x' (line 253)
    overwrite_x_16940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 68), 'overwrite_x', False)
    keyword_16941 = overwrite_x_16940
    kwargs_16942 = {'overwrite_x': keyword_16941, 'swap_real_imag': keyword_16939}
    # Getting the type of 'convolve' (line 253)
    convolve_16934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 11), 'convolve', False)
    # Obtaining the member 'convolve' of a type (line 253)
    convolve_16935 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 11), convolve_16934, 'convolve')
    # Calling convolve(args, kwargs) (line 253)
    convolve_call_result_16943 = invoke(stypy.reporting.localization.Localization(__file__, 253, 11), convolve_16935, *[tmp_16936, omega_16937], **kwargs_16942)
    
    # Assigning a type to the variable 'stypy_return_type' (line 253)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 4), 'stypy_return_type', convolve_call_result_16943)
    
    # ################# End of 'hilbert(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'hilbert' in the type store
    # Getting the type of 'stypy_return_type' (line 195)
    stypy_return_type_16944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_16944)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'hilbert'
    return stypy_return_type_16944

# Assigning a type to the variable 'hilbert' (line 195)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 0), 'hilbert', hilbert)
# Deleting a member
module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 254, 0), module_type_store, '_cache')

@norecursion
def ihilbert(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'ihilbert'
    module_type_store = module_type_store.open_function_context('ihilbert', 257, 0, False)
    
    # Passed parameters checking function
    ihilbert.stypy_localization = localization
    ihilbert.stypy_type_of_self = None
    ihilbert.stypy_type_store = module_type_store
    ihilbert.stypy_function_name = 'ihilbert'
    ihilbert.stypy_param_names_list = ['x']
    ihilbert.stypy_varargs_param_name = None
    ihilbert.stypy_kwargs_param_name = None
    ihilbert.stypy_call_defaults = defaults
    ihilbert.stypy_call_varargs = varargs
    ihilbert.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'ihilbert', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'ihilbert', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'ihilbert(...)' code ##################

    str_16945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, (-1)), 'str', '\n    Return inverse Hilbert transform of a periodic sequence x.\n\n    If ``x_j`` and ``y_j`` are Fourier coefficients of periodic functions x\n    and y, respectively, then::\n\n      y_j = -sqrt(-1)*sign(j) * x_j\n      y_0 = 0\n\n    ')
    
    
    # Call to hilbert(...): (line 268)
    # Processing the call arguments (line 268)
    # Getting the type of 'x' (line 268)
    x_16947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 20), 'x', False)
    # Processing the call keyword arguments (line 268)
    kwargs_16948 = {}
    # Getting the type of 'hilbert' (line 268)
    hilbert_16946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 12), 'hilbert', False)
    # Calling hilbert(args, kwargs) (line 268)
    hilbert_call_result_16949 = invoke(stypy.reporting.localization.Localization(__file__, 268, 12), hilbert_16946, *[x_16947], **kwargs_16948)
    
    # Applying the 'usub' unary operator (line 268)
    result___neg___16950 = python_operator(stypy.reporting.localization.Localization(__file__, 268, 11), 'usub', hilbert_call_result_16949)
    
    # Assigning a type to the variable 'stypy_return_type' (line 268)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 4), 'stypy_return_type', result___neg___16950)
    
    # ################# End of 'ihilbert(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'ihilbert' in the type store
    # Getting the type of 'stypy_return_type' (line 257)
    stypy_return_type_16951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_16951)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'ihilbert'
    return stypy_return_type_16951

# Assigning a type to the variable 'ihilbert' (line 257)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 0), 'ihilbert', ihilbert)

# Assigning a Dict to a Name (line 271):

# Assigning a Dict to a Name (line 271):

# Obtaining an instance of the builtin type 'dict' (line 271)
dict_16952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 9), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 271)

# Assigning a type to the variable '_cache' (line 271)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 0), '_cache', dict_16952)

@norecursion
def cs_diff(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 274)
    None_16953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 28), 'None')
    # Getting the type of '_cache' (line 274)
    _cache_16954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 41), '_cache')
    defaults = [None_16953, _cache_16954]
    # Create a new context for function 'cs_diff'
    module_type_store = module_type_store.open_function_context('cs_diff', 274, 0, False)
    
    # Passed parameters checking function
    cs_diff.stypy_localization = localization
    cs_diff.stypy_type_of_self = None
    cs_diff.stypy_type_store = module_type_store
    cs_diff.stypy_function_name = 'cs_diff'
    cs_diff.stypy_param_names_list = ['x', 'a', 'b', 'period', '_cache']
    cs_diff.stypy_varargs_param_name = None
    cs_diff.stypy_kwargs_param_name = None
    cs_diff.stypy_call_defaults = defaults
    cs_diff.stypy_call_varargs = varargs
    cs_diff.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'cs_diff', ['x', 'a', 'b', 'period', '_cache'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'cs_diff', localization, ['x', 'a', 'b', 'period', '_cache'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'cs_diff(...)' code ##################

    str_16955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, (-1)), 'str', '\n    Return (a,b)-cosh/sinh pseudo-derivative of a periodic sequence.\n\n    If ``x_j`` and ``y_j`` are Fourier coefficients of periodic functions x\n    and y, respectively, then::\n\n      y_j = -sqrt(-1)*cosh(j*a*2*pi/period)/sinh(j*b*2*pi/period) * x_j\n      y_0 = 0\n\n    Parameters\n    ----------\n    x : array_like\n        The array to take the pseudo-derivative from.\n    a, b : float\n        Defines the parameters of the cosh/sinh pseudo-differential\n        operator.\n    period : float, optional\n        The period of the sequence. Default period is ``2*pi``.\n\n    Returns\n    -------\n    cs_diff : ndarray\n        Pseudo-derivative of periodic sequence `x`.\n\n    Notes\n    -----\n    For even len(`x`), the Nyquist mode of `x` is taken as zero.\n\n    ')
    
    # Assigning a Call to a Name (line 304):
    
    # Assigning a Call to a Name (line 304):
    
    # Call to asarray(...): (line 304)
    # Processing the call arguments (line 304)
    # Getting the type of 'x' (line 304)
    x_16957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 18), 'x', False)
    # Processing the call keyword arguments (line 304)
    kwargs_16958 = {}
    # Getting the type of 'asarray' (line 304)
    asarray_16956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 10), 'asarray', False)
    # Calling asarray(args, kwargs) (line 304)
    asarray_call_result_16959 = invoke(stypy.reporting.localization.Localization(__file__, 304, 10), asarray_16956, *[x_16957], **kwargs_16958)
    
    # Assigning a type to the variable 'tmp' (line 304)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 4), 'tmp', asarray_call_result_16959)
    
    
    # Call to iscomplexobj(...): (line 305)
    # Processing the call arguments (line 305)
    # Getting the type of 'tmp' (line 305)
    tmp_16961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 20), 'tmp', False)
    # Processing the call keyword arguments (line 305)
    kwargs_16962 = {}
    # Getting the type of 'iscomplexobj' (line 305)
    iscomplexobj_16960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 7), 'iscomplexobj', False)
    # Calling iscomplexobj(args, kwargs) (line 305)
    iscomplexobj_call_result_16963 = invoke(stypy.reporting.localization.Localization(__file__, 305, 7), iscomplexobj_16960, *[tmp_16961], **kwargs_16962)
    
    # Testing the type of an if condition (line 305)
    if_condition_16964 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 305, 4), iscomplexobj_call_result_16963)
    # Assigning a type to the variable 'if_condition_16964' (line 305)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 305, 4), 'if_condition_16964', if_condition_16964)
    # SSA begins for if statement (line 305)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to cs_diff(...): (line 306)
    # Processing the call arguments (line 306)
    # Getting the type of 'tmp' (line 306)
    tmp_16966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 23), 'tmp', False)
    # Obtaining the member 'real' of a type (line 306)
    real_16967 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 23), tmp_16966, 'real')
    # Getting the type of 'a' (line 306)
    a_16968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 32), 'a', False)
    # Getting the type of 'b' (line 306)
    b_16969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 34), 'b', False)
    # Getting the type of 'period' (line 306)
    period_16970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 36), 'period', False)
    # Processing the call keyword arguments (line 306)
    kwargs_16971 = {}
    # Getting the type of 'cs_diff' (line 306)
    cs_diff_16965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 15), 'cs_diff', False)
    # Calling cs_diff(args, kwargs) (line 306)
    cs_diff_call_result_16972 = invoke(stypy.reporting.localization.Localization(__file__, 306, 15), cs_diff_16965, *[real_16967, a_16968, b_16969, period_16970], **kwargs_16971)
    
    complex_16973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 15), 'complex')
    
    # Call to cs_diff(...): (line 307)
    # Processing the call arguments (line 307)
    # Getting the type of 'tmp' (line 307)
    tmp_16975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 26), 'tmp', False)
    # Obtaining the member 'imag' of a type (line 307)
    imag_16976 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 26), tmp_16975, 'imag')
    # Getting the type of 'a' (line 307)
    a_16977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 35), 'a', False)
    # Getting the type of 'b' (line 307)
    b_16978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 37), 'b', False)
    # Getting the type of 'period' (line 307)
    period_16979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 39), 'period', False)
    # Processing the call keyword arguments (line 307)
    kwargs_16980 = {}
    # Getting the type of 'cs_diff' (line 307)
    cs_diff_16974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 18), 'cs_diff', False)
    # Calling cs_diff(args, kwargs) (line 307)
    cs_diff_call_result_16981 = invoke(stypy.reporting.localization.Localization(__file__, 307, 18), cs_diff_16974, *[imag_16976, a_16977, b_16978, period_16979], **kwargs_16980)
    
    # Applying the binary operator '*' (line 307)
    result_mul_16982 = python_operator(stypy.reporting.localization.Localization(__file__, 307, 15), '*', complex_16973, cs_diff_call_result_16981)
    
    # Applying the binary operator '+' (line 306)
    result_add_16983 = python_operator(stypy.reporting.localization.Localization(__file__, 306, 15), '+', cs_diff_call_result_16972, result_mul_16982)
    
    # Assigning a type to the variable 'stypy_return_type' (line 306)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 8), 'stypy_return_type', result_add_16983)
    # SSA join for if statement (line 305)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 308)
    # Getting the type of 'period' (line 308)
    period_16984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 4), 'period')
    # Getting the type of 'None' (line 308)
    None_16985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 21), 'None')
    
    (may_be_16986, more_types_in_union_16987) = may_not_be_none(period_16984, None_16985)

    if may_be_16986:

        if more_types_in_union_16987:
            # Runtime conditional SSA (line 308)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a BinOp to a Name (line 309):
        
        # Assigning a BinOp to a Name (line 309):
        # Getting the type of 'a' (line 309)
        a_16988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 12), 'a')
        int_16989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 14), 'int')
        # Applying the binary operator '*' (line 309)
        result_mul_16990 = python_operator(stypy.reporting.localization.Localization(__file__, 309, 12), '*', a_16988, int_16989)
        
        # Getting the type of 'pi' (line 309)
        pi_16991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 16), 'pi')
        # Applying the binary operator '*' (line 309)
        result_mul_16992 = python_operator(stypy.reporting.localization.Localization(__file__, 309, 15), '*', result_mul_16990, pi_16991)
        
        # Getting the type of 'period' (line 309)
        period_16993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 19), 'period')
        # Applying the binary operator 'div' (line 309)
        result_div_16994 = python_operator(stypy.reporting.localization.Localization(__file__, 309, 18), 'div', result_mul_16992, period_16993)
        
        # Assigning a type to the variable 'a' (line 309)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 8), 'a', result_div_16994)
        
        # Assigning a BinOp to a Name (line 310):
        
        # Assigning a BinOp to a Name (line 310):
        # Getting the type of 'b' (line 310)
        b_16995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 12), 'b')
        int_16996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 14), 'int')
        # Applying the binary operator '*' (line 310)
        result_mul_16997 = python_operator(stypy.reporting.localization.Localization(__file__, 310, 12), '*', b_16995, int_16996)
        
        # Getting the type of 'pi' (line 310)
        pi_16998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 16), 'pi')
        # Applying the binary operator '*' (line 310)
        result_mul_16999 = python_operator(stypy.reporting.localization.Localization(__file__, 310, 15), '*', result_mul_16997, pi_16998)
        
        # Getting the type of 'period' (line 310)
        period_17000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 19), 'period')
        # Applying the binary operator 'div' (line 310)
        result_div_17001 = python_operator(stypy.reporting.localization.Localization(__file__, 310, 18), 'div', result_mul_16999, period_17000)
        
        # Assigning a type to the variable 'b' (line 310)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 8), 'b', result_div_17001)

        if more_types_in_union_16987:
            # SSA join for if statement (line 308)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 311):
    
    # Assigning a Call to a Name (line 311):
    
    # Call to len(...): (line 311)
    # Processing the call arguments (line 311)
    # Getting the type of 'x' (line 311)
    x_17003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 12), 'x', False)
    # Processing the call keyword arguments (line 311)
    kwargs_17004 = {}
    # Getting the type of 'len' (line 311)
    len_17002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 8), 'len', False)
    # Calling len(args, kwargs) (line 311)
    len_call_result_17005 = invoke(stypy.reporting.localization.Localization(__file__, 311, 8), len_17002, *[x_17003], **kwargs_17004)
    
    # Assigning a type to the variable 'n' (line 311)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 4), 'n', len_call_result_17005)
    
    # Assigning a Call to a Name (line 312):
    
    # Assigning a Call to a Name (line 312):
    
    # Call to get(...): (line 312)
    # Processing the call arguments (line 312)
    
    # Obtaining an instance of the builtin type 'tuple' (line 312)
    tuple_17008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 24), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 312)
    # Adding element type (line 312)
    # Getting the type of 'n' (line 312)
    n_17009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 24), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 312, 24), tuple_17008, n_17009)
    # Adding element type (line 312)
    # Getting the type of 'a' (line 312)
    a_17010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 26), 'a', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 312, 24), tuple_17008, a_17010)
    # Adding element type (line 312)
    # Getting the type of 'b' (line 312)
    b_17011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 28), 'b', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 312, 24), tuple_17008, b_17011)
    
    # Processing the call keyword arguments (line 312)
    kwargs_17012 = {}
    # Getting the type of '_cache' (line 312)
    _cache_17006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 12), '_cache', False)
    # Obtaining the member 'get' of a type (line 312)
    get_17007 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 12), _cache_17006, 'get')
    # Calling get(args, kwargs) (line 312)
    get_call_result_17013 = invoke(stypy.reporting.localization.Localization(__file__, 312, 12), get_17007, *[tuple_17008], **kwargs_17012)
    
    # Assigning a type to the variable 'omega' (line 312)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 4), 'omega', get_call_result_17013)
    
    # Type idiom detected: calculating its left and rigth part (line 313)
    # Getting the type of 'omega' (line 313)
    omega_17014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 7), 'omega')
    # Getting the type of 'None' (line 313)
    None_17015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 16), 'None')
    
    (may_be_17016, more_types_in_union_17017) = may_be_none(omega_17014, None_17015)

    if may_be_17016:

        if more_types_in_union_17017:
            # Runtime conditional SSA (line 313)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        
        
        # Call to len(...): (line 314)
        # Processing the call arguments (line 314)
        # Getting the type of '_cache' (line 314)
        _cache_17019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 15), '_cache', False)
        # Processing the call keyword arguments (line 314)
        kwargs_17020 = {}
        # Getting the type of 'len' (line 314)
        len_17018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 11), 'len', False)
        # Calling len(args, kwargs) (line 314)
        len_call_result_17021 = invoke(stypy.reporting.localization.Localization(__file__, 314, 11), len_17018, *[_cache_17019], **kwargs_17020)
        
        int_17022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 25), 'int')
        # Applying the binary operator '>' (line 314)
        result_gt_17023 = python_operator(stypy.reporting.localization.Localization(__file__, 314, 11), '>', len_call_result_17021, int_17022)
        
        # Testing the type of an if condition (line 314)
        if_condition_17024 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 314, 8), result_gt_17023)
        # Assigning a type to the variable 'if_condition_17024' (line 314)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 8), 'if_condition_17024', if_condition_17024)
        # SSA begins for if statement (line 314)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of '_cache' (line 315)
        _cache_17025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 18), '_cache')
        # Testing the type of an if condition (line 315)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 315, 12), _cache_17025)
        # SSA begins for while statement (line 315)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Call to popitem(...): (line 316)
        # Processing the call keyword arguments (line 316)
        kwargs_17028 = {}
        # Getting the type of '_cache' (line 316)
        _cache_17026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 16), '_cache', False)
        # Obtaining the member 'popitem' of a type (line 316)
        popitem_17027 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 16), _cache_17026, 'popitem')
        # Calling popitem(args, kwargs) (line 316)
        popitem_call_result_17029 = invoke(stypy.reporting.localization.Localization(__file__, 316, 16), popitem_17027, *[], **kwargs_17028)
        
        # SSA join for while statement (line 315)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 314)
        module_type_store = module_type_store.join_ssa_context()
        

        @norecursion
        def kernel(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            # Getting the type of 'a' (line 318)
            a_17030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 23), 'a')
            # Getting the type of 'b' (line 318)
            b_17031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 27), 'b')
            defaults = [a_17030, b_17031]
            # Create a new context for function 'kernel'
            module_type_store = module_type_store.open_function_context('kernel', 318, 8, False)
            
            # Passed parameters checking function
            kernel.stypy_localization = localization
            kernel.stypy_type_of_self = None
            kernel.stypy_type_store = module_type_store
            kernel.stypy_function_name = 'kernel'
            kernel.stypy_param_names_list = ['k', 'a', 'b']
            kernel.stypy_varargs_param_name = None
            kernel.stypy_kwargs_param_name = None
            kernel.stypy_call_defaults = defaults
            kernel.stypy_call_varargs = varargs
            kernel.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'kernel', ['k', 'a', 'b'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'kernel', localization, ['k', 'a', 'b'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'kernel(...)' code ##################

            
            # Getting the type of 'k' (line 319)
            k_17032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 15), 'k')
            # Testing the type of an if condition (line 319)
            if_condition_17033 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 319, 12), k_17032)
            # Assigning a type to the variable 'if_condition_17033' (line 319)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 319, 12), 'if_condition_17033', if_condition_17033)
            # SSA begins for if statement (line 319)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            
            # Call to cosh(...): (line 320)
            # Processing the call arguments (line 320)
            # Getting the type of 'a' (line 320)
            a_17035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 29), 'a', False)
            # Getting the type of 'k' (line 320)
            k_17036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 31), 'k', False)
            # Applying the binary operator '*' (line 320)
            result_mul_17037 = python_operator(stypy.reporting.localization.Localization(__file__, 320, 29), '*', a_17035, k_17036)
            
            # Processing the call keyword arguments (line 320)
            kwargs_17038 = {}
            # Getting the type of 'cosh' (line 320)
            cosh_17034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 24), 'cosh', False)
            # Calling cosh(args, kwargs) (line 320)
            cosh_call_result_17039 = invoke(stypy.reporting.localization.Localization(__file__, 320, 24), cosh_17034, *[result_mul_17037], **kwargs_17038)
            
            # Applying the 'usub' unary operator (line 320)
            result___neg___17040 = python_operator(stypy.reporting.localization.Localization(__file__, 320, 23), 'usub', cosh_call_result_17039)
            
            
            # Call to sinh(...): (line 320)
            # Processing the call arguments (line 320)
            # Getting the type of 'b' (line 320)
            b_17042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 39), 'b', False)
            # Getting the type of 'k' (line 320)
            k_17043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 41), 'k', False)
            # Applying the binary operator '*' (line 320)
            result_mul_17044 = python_operator(stypy.reporting.localization.Localization(__file__, 320, 39), '*', b_17042, k_17043)
            
            # Processing the call keyword arguments (line 320)
            kwargs_17045 = {}
            # Getting the type of 'sinh' (line 320)
            sinh_17041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 34), 'sinh', False)
            # Calling sinh(args, kwargs) (line 320)
            sinh_call_result_17046 = invoke(stypy.reporting.localization.Localization(__file__, 320, 34), sinh_17041, *[result_mul_17044], **kwargs_17045)
            
            # Applying the binary operator 'div' (line 320)
            result_div_17047 = python_operator(stypy.reporting.localization.Localization(__file__, 320, 23), 'div', result___neg___17040, sinh_call_result_17046)
            
            # Assigning a type to the variable 'stypy_return_type' (line 320)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 320, 16), 'stypy_return_type', result_div_17047)
            # SSA join for if statement (line 319)
            module_type_store = module_type_store.join_ssa_context()
            
            int_17048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 321, 19), 'int')
            # Assigning a type to the variable 'stypy_return_type' (line 321)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 321, 12), 'stypy_return_type', int_17048)
            
            # ################# End of 'kernel(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'kernel' in the type store
            # Getting the type of 'stypy_return_type' (line 318)
            stypy_return_type_17049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_17049)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'kernel'
            return stypy_return_type_17049

        # Assigning a type to the variable 'kernel' (line 318)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 8), 'kernel', kernel)
        
        # Assigning a Call to a Name (line 322):
        
        # Assigning a Call to a Name (line 322):
        
        # Call to init_convolution_kernel(...): (line 322)
        # Processing the call arguments (line 322)
        # Getting the type of 'n' (line 322)
        n_17052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 49), 'n', False)
        # Getting the type of 'kernel' (line 322)
        kernel_17053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 51), 'kernel', False)
        # Processing the call keyword arguments (line 322)
        int_17054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 60), 'int')
        keyword_17055 = int_17054
        kwargs_17056 = {'d': keyword_17055}
        # Getting the type of 'convolve' (line 322)
        convolve_17050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 16), 'convolve', False)
        # Obtaining the member 'init_convolution_kernel' of a type (line 322)
        init_convolution_kernel_17051 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 16), convolve_17050, 'init_convolution_kernel')
        # Calling init_convolution_kernel(args, kwargs) (line 322)
        init_convolution_kernel_call_result_17057 = invoke(stypy.reporting.localization.Localization(__file__, 322, 16), init_convolution_kernel_17051, *[n_17052, kernel_17053], **kwargs_17056)
        
        # Assigning a type to the variable 'omega' (line 322)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 322, 8), 'omega', init_convolution_kernel_call_result_17057)
        
        # Assigning a Name to a Subscript (line 323):
        
        # Assigning a Name to a Subscript (line 323):
        # Getting the type of 'omega' (line 323)
        omega_17058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 26), 'omega')
        # Getting the type of '_cache' (line 323)
        _cache_17059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 8), '_cache')
        
        # Obtaining an instance of the builtin type 'tuple' (line 323)
        tuple_17060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 16), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 323)
        # Adding element type (line 323)
        # Getting the type of 'n' (line 323)
        n_17061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 16), 'n')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 323, 16), tuple_17060, n_17061)
        # Adding element type (line 323)
        # Getting the type of 'a' (line 323)
        a_17062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 18), 'a')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 323, 16), tuple_17060, a_17062)
        # Adding element type (line 323)
        # Getting the type of 'b' (line 323)
        b_17063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 20), 'b')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 323, 16), tuple_17060, b_17063)
        
        # Storing an element on a container (line 323)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 323, 8), _cache_17059, (tuple_17060, omega_17058))

        if more_types_in_union_17017:
            # SSA join for if statement (line 313)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 324):
    
    # Assigning a Call to a Name (line 324):
    
    # Call to _datacopied(...): (line 324)
    # Processing the call arguments (line 324)
    # Getting the type of 'tmp' (line 324)
    tmp_17065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 30), 'tmp', False)
    # Getting the type of 'x' (line 324)
    x_17066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 35), 'x', False)
    # Processing the call keyword arguments (line 324)
    kwargs_17067 = {}
    # Getting the type of '_datacopied' (line 324)
    _datacopied_17064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 18), '_datacopied', False)
    # Calling _datacopied(args, kwargs) (line 324)
    _datacopied_call_result_17068 = invoke(stypy.reporting.localization.Localization(__file__, 324, 18), _datacopied_17064, *[tmp_17065, x_17066], **kwargs_17067)
    
    # Assigning a type to the variable 'overwrite_x' (line 324)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 4), 'overwrite_x', _datacopied_call_result_17068)
    
    # Call to convolve(...): (line 325)
    # Processing the call arguments (line 325)
    # Getting the type of 'tmp' (line 325)
    tmp_17071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 29), 'tmp', False)
    # Getting the type of 'omega' (line 325)
    omega_17072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 33), 'omega', False)
    # Processing the call keyword arguments (line 325)
    int_17073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 54), 'int')
    keyword_17074 = int_17073
    # Getting the type of 'overwrite_x' (line 325)
    overwrite_x_17075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 68), 'overwrite_x', False)
    keyword_17076 = overwrite_x_17075
    kwargs_17077 = {'overwrite_x': keyword_17076, 'swap_real_imag': keyword_17074}
    # Getting the type of 'convolve' (line 325)
    convolve_17069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 11), 'convolve', False)
    # Obtaining the member 'convolve' of a type (line 325)
    convolve_17070 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 11), convolve_17069, 'convolve')
    # Calling convolve(args, kwargs) (line 325)
    convolve_call_result_17078 = invoke(stypy.reporting.localization.Localization(__file__, 325, 11), convolve_17070, *[tmp_17071, omega_17072], **kwargs_17077)
    
    # Assigning a type to the variable 'stypy_return_type' (line 325)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 4), 'stypy_return_type', convolve_call_result_17078)
    
    # ################# End of 'cs_diff(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'cs_diff' in the type store
    # Getting the type of 'stypy_return_type' (line 274)
    stypy_return_type_17079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_17079)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'cs_diff'
    return stypy_return_type_17079

# Assigning a type to the variable 'cs_diff' (line 274)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 0), 'cs_diff', cs_diff)
# Deleting a member
module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 326, 0), module_type_store, '_cache')

# Assigning a Dict to a Name (line 329):

# Assigning a Dict to a Name (line 329):

# Obtaining an instance of the builtin type 'dict' (line 329)
dict_17080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 9), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 329)

# Assigning a type to the variable '_cache' (line 329)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 0), '_cache', dict_17080)

@norecursion
def sc_diff(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 332)
    None_17081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 28), 'None')
    # Getting the type of '_cache' (line 332)
    _cache_17082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 41), '_cache')
    defaults = [None_17081, _cache_17082]
    # Create a new context for function 'sc_diff'
    module_type_store = module_type_store.open_function_context('sc_diff', 332, 0, False)
    
    # Passed parameters checking function
    sc_diff.stypy_localization = localization
    sc_diff.stypy_type_of_self = None
    sc_diff.stypy_type_store = module_type_store
    sc_diff.stypy_function_name = 'sc_diff'
    sc_diff.stypy_param_names_list = ['x', 'a', 'b', 'period', '_cache']
    sc_diff.stypy_varargs_param_name = None
    sc_diff.stypy_kwargs_param_name = None
    sc_diff.stypy_call_defaults = defaults
    sc_diff.stypy_call_varargs = varargs
    sc_diff.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'sc_diff', ['x', 'a', 'b', 'period', '_cache'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'sc_diff', localization, ['x', 'a', 'b', 'period', '_cache'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'sc_diff(...)' code ##################

    str_17083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 357, (-1)), 'str', '\n    Return (a,b)-sinh/cosh pseudo-derivative of a periodic sequence x.\n\n    If x_j and y_j are Fourier coefficients of periodic functions x\n    and y, respectively, then::\n\n      y_j = sqrt(-1)*sinh(j*a*2*pi/period)/cosh(j*b*2*pi/period) * x_j\n      y_0 = 0\n\n    Parameters\n    ----------\n    x : array_like\n        Input array.\n    a,b : float\n        Defines the parameters of the sinh/cosh pseudo-differential\n        operator.\n    period : float, optional\n        The period of the sequence x. Default is 2*pi.\n\n    Notes\n    -----\n    ``sc_diff(cs_diff(x,a,b),b,a) == x``\n    For even ``len(x)``, the Nyquist mode of x is taken as zero.\n\n    ')
    
    # Assigning a Call to a Name (line 358):
    
    # Assigning a Call to a Name (line 358):
    
    # Call to asarray(...): (line 358)
    # Processing the call arguments (line 358)
    # Getting the type of 'x' (line 358)
    x_17085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 18), 'x', False)
    # Processing the call keyword arguments (line 358)
    kwargs_17086 = {}
    # Getting the type of 'asarray' (line 358)
    asarray_17084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 10), 'asarray', False)
    # Calling asarray(args, kwargs) (line 358)
    asarray_call_result_17087 = invoke(stypy.reporting.localization.Localization(__file__, 358, 10), asarray_17084, *[x_17085], **kwargs_17086)
    
    # Assigning a type to the variable 'tmp' (line 358)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 358, 4), 'tmp', asarray_call_result_17087)
    
    
    # Call to iscomplexobj(...): (line 359)
    # Processing the call arguments (line 359)
    # Getting the type of 'tmp' (line 359)
    tmp_17089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 20), 'tmp', False)
    # Processing the call keyword arguments (line 359)
    kwargs_17090 = {}
    # Getting the type of 'iscomplexobj' (line 359)
    iscomplexobj_17088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 7), 'iscomplexobj', False)
    # Calling iscomplexobj(args, kwargs) (line 359)
    iscomplexobj_call_result_17091 = invoke(stypy.reporting.localization.Localization(__file__, 359, 7), iscomplexobj_17088, *[tmp_17089], **kwargs_17090)
    
    # Testing the type of an if condition (line 359)
    if_condition_17092 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 359, 4), iscomplexobj_call_result_17091)
    # Assigning a type to the variable 'if_condition_17092' (line 359)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 359, 4), 'if_condition_17092', if_condition_17092)
    # SSA begins for if statement (line 359)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to sc_diff(...): (line 360)
    # Processing the call arguments (line 360)
    # Getting the type of 'tmp' (line 360)
    tmp_17094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 23), 'tmp', False)
    # Obtaining the member 'real' of a type (line 360)
    real_17095 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 23), tmp_17094, 'real')
    # Getting the type of 'a' (line 360)
    a_17096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 32), 'a', False)
    # Getting the type of 'b' (line 360)
    b_17097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 34), 'b', False)
    # Getting the type of 'period' (line 360)
    period_17098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 36), 'period', False)
    # Processing the call keyword arguments (line 360)
    kwargs_17099 = {}
    # Getting the type of 'sc_diff' (line 360)
    sc_diff_17093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 15), 'sc_diff', False)
    # Calling sc_diff(args, kwargs) (line 360)
    sc_diff_call_result_17100 = invoke(stypy.reporting.localization.Localization(__file__, 360, 15), sc_diff_17093, *[real_17095, a_17096, b_17097, period_17098], **kwargs_17099)
    
    complex_17101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 15), 'complex')
    
    # Call to sc_diff(...): (line 361)
    # Processing the call arguments (line 361)
    # Getting the type of 'tmp' (line 361)
    tmp_17103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 26), 'tmp', False)
    # Obtaining the member 'imag' of a type (line 361)
    imag_17104 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 361, 26), tmp_17103, 'imag')
    # Getting the type of 'a' (line 361)
    a_17105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 35), 'a', False)
    # Getting the type of 'b' (line 361)
    b_17106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 37), 'b', False)
    # Getting the type of 'period' (line 361)
    period_17107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 39), 'period', False)
    # Processing the call keyword arguments (line 361)
    kwargs_17108 = {}
    # Getting the type of 'sc_diff' (line 361)
    sc_diff_17102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 18), 'sc_diff', False)
    # Calling sc_diff(args, kwargs) (line 361)
    sc_diff_call_result_17109 = invoke(stypy.reporting.localization.Localization(__file__, 361, 18), sc_diff_17102, *[imag_17104, a_17105, b_17106, period_17107], **kwargs_17108)
    
    # Applying the binary operator '*' (line 361)
    result_mul_17110 = python_operator(stypy.reporting.localization.Localization(__file__, 361, 15), '*', complex_17101, sc_diff_call_result_17109)
    
    # Applying the binary operator '+' (line 360)
    result_add_17111 = python_operator(stypy.reporting.localization.Localization(__file__, 360, 15), '+', sc_diff_call_result_17100, result_mul_17110)
    
    # Assigning a type to the variable 'stypy_return_type' (line 360)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 360, 8), 'stypy_return_type', result_add_17111)
    # SSA join for if statement (line 359)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 362)
    # Getting the type of 'period' (line 362)
    period_17112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 4), 'period')
    # Getting the type of 'None' (line 362)
    None_17113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 21), 'None')
    
    (may_be_17114, more_types_in_union_17115) = may_not_be_none(period_17112, None_17113)

    if may_be_17114:

        if more_types_in_union_17115:
            # Runtime conditional SSA (line 362)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a BinOp to a Name (line 363):
        
        # Assigning a BinOp to a Name (line 363):
        # Getting the type of 'a' (line 363)
        a_17116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 12), 'a')
        int_17117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, 14), 'int')
        # Applying the binary operator '*' (line 363)
        result_mul_17118 = python_operator(stypy.reporting.localization.Localization(__file__, 363, 12), '*', a_17116, int_17117)
        
        # Getting the type of 'pi' (line 363)
        pi_17119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 16), 'pi')
        # Applying the binary operator '*' (line 363)
        result_mul_17120 = python_operator(stypy.reporting.localization.Localization(__file__, 363, 15), '*', result_mul_17118, pi_17119)
        
        # Getting the type of 'period' (line 363)
        period_17121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 19), 'period')
        # Applying the binary operator 'div' (line 363)
        result_div_17122 = python_operator(stypy.reporting.localization.Localization(__file__, 363, 18), 'div', result_mul_17120, period_17121)
        
        # Assigning a type to the variable 'a' (line 363)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 8), 'a', result_div_17122)
        
        # Assigning a BinOp to a Name (line 364):
        
        # Assigning a BinOp to a Name (line 364):
        # Getting the type of 'b' (line 364)
        b_17123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 12), 'b')
        int_17124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 14), 'int')
        # Applying the binary operator '*' (line 364)
        result_mul_17125 = python_operator(stypy.reporting.localization.Localization(__file__, 364, 12), '*', b_17123, int_17124)
        
        # Getting the type of 'pi' (line 364)
        pi_17126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 16), 'pi')
        # Applying the binary operator '*' (line 364)
        result_mul_17127 = python_operator(stypy.reporting.localization.Localization(__file__, 364, 15), '*', result_mul_17125, pi_17126)
        
        # Getting the type of 'period' (line 364)
        period_17128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 19), 'period')
        # Applying the binary operator 'div' (line 364)
        result_div_17129 = python_operator(stypy.reporting.localization.Localization(__file__, 364, 18), 'div', result_mul_17127, period_17128)
        
        # Assigning a type to the variable 'b' (line 364)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 364, 8), 'b', result_div_17129)

        if more_types_in_union_17115:
            # SSA join for if statement (line 362)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 365):
    
    # Assigning a Call to a Name (line 365):
    
    # Call to len(...): (line 365)
    # Processing the call arguments (line 365)
    # Getting the type of 'x' (line 365)
    x_17131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 12), 'x', False)
    # Processing the call keyword arguments (line 365)
    kwargs_17132 = {}
    # Getting the type of 'len' (line 365)
    len_17130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 8), 'len', False)
    # Calling len(args, kwargs) (line 365)
    len_call_result_17133 = invoke(stypy.reporting.localization.Localization(__file__, 365, 8), len_17130, *[x_17131], **kwargs_17132)
    
    # Assigning a type to the variable 'n' (line 365)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 365, 4), 'n', len_call_result_17133)
    
    # Assigning a Call to a Name (line 366):
    
    # Assigning a Call to a Name (line 366):
    
    # Call to get(...): (line 366)
    # Processing the call arguments (line 366)
    
    # Obtaining an instance of the builtin type 'tuple' (line 366)
    tuple_17136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 24), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 366)
    # Adding element type (line 366)
    # Getting the type of 'n' (line 366)
    n_17137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 24), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 366, 24), tuple_17136, n_17137)
    # Adding element type (line 366)
    # Getting the type of 'a' (line 366)
    a_17138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 26), 'a', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 366, 24), tuple_17136, a_17138)
    # Adding element type (line 366)
    # Getting the type of 'b' (line 366)
    b_17139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 28), 'b', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 366, 24), tuple_17136, b_17139)
    
    # Processing the call keyword arguments (line 366)
    kwargs_17140 = {}
    # Getting the type of '_cache' (line 366)
    _cache_17134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 12), '_cache', False)
    # Obtaining the member 'get' of a type (line 366)
    get_17135 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 366, 12), _cache_17134, 'get')
    # Calling get(args, kwargs) (line 366)
    get_call_result_17141 = invoke(stypy.reporting.localization.Localization(__file__, 366, 12), get_17135, *[tuple_17136], **kwargs_17140)
    
    # Assigning a type to the variable 'omega' (line 366)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 366, 4), 'omega', get_call_result_17141)
    
    # Type idiom detected: calculating its left and rigth part (line 367)
    # Getting the type of 'omega' (line 367)
    omega_17142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 7), 'omega')
    # Getting the type of 'None' (line 367)
    None_17143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 16), 'None')
    
    (may_be_17144, more_types_in_union_17145) = may_be_none(omega_17142, None_17143)

    if may_be_17144:

        if more_types_in_union_17145:
            # Runtime conditional SSA (line 367)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        
        
        # Call to len(...): (line 368)
        # Processing the call arguments (line 368)
        # Getting the type of '_cache' (line 368)
        _cache_17147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 15), '_cache', False)
        # Processing the call keyword arguments (line 368)
        kwargs_17148 = {}
        # Getting the type of 'len' (line 368)
        len_17146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 11), 'len', False)
        # Calling len(args, kwargs) (line 368)
        len_call_result_17149 = invoke(stypy.reporting.localization.Localization(__file__, 368, 11), len_17146, *[_cache_17147], **kwargs_17148)
        
        int_17150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, 25), 'int')
        # Applying the binary operator '>' (line 368)
        result_gt_17151 = python_operator(stypy.reporting.localization.Localization(__file__, 368, 11), '>', len_call_result_17149, int_17150)
        
        # Testing the type of an if condition (line 368)
        if_condition_17152 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 368, 8), result_gt_17151)
        # Assigning a type to the variable 'if_condition_17152' (line 368)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 8), 'if_condition_17152', if_condition_17152)
        # SSA begins for if statement (line 368)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of '_cache' (line 369)
        _cache_17153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 18), '_cache')
        # Testing the type of an if condition (line 369)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 369, 12), _cache_17153)
        # SSA begins for while statement (line 369)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Call to popitem(...): (line 370)
        # Processing the call keyword arguments (line 370)
        kwargs_17156 = {}
        # Getting the type of '_cache' (line 370)
        _cache_17154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 16), '_cache', False)
        # Obtaining the member 'popitem' of a type (line 370)
        popitem_17155 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 16), _cache_17154, 'popitem')
        # Calling popitem(args, kwargs) (line 370)
        popitem_call_result_17157 = invoke(stypy.reporting.localization.Localization(__file__, 370, 16), popitem_17155, *[], **kwargs_17156)
        
        # SSA join for while statement (line 369)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 368)
        module_type_store = module_type_store.join_ssa_context()
        

        @norecursion
        def kernel(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            # Getting the type of 'a' (line 372)
            a_17158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 23), 'a')
            # Getting the type of 'b' (line 372)
            b_17159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 27), 'b')
            defaults = [a_17158, b_17159]
            # Create a new context for function 'kernel'
            module_type_store = module_type_store.open_function_context('kernel', 372, 8, False)
            
            # Passed parameters checking function
            kernel.stypy_localization = localization
            kernel.stypy_type_of_self = None
            kernel.stypy_type_store = module_type_store
            kernel.stypy_function_name = 'kernel'
            kernel.stypy_param_names_list = ['k', 'a', 'b']
            kernel.stypy_varargs_param_name = None
            kernel.stypy_kwargs_param_name = None
            kernel.stypy_call_defaults = defaults
            kernel.stypy_call_varargs = varargs
            kernel.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'kernel', ['k', 'a', 'b'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'kernel', localization, ['k', 'a', 'b'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'kernel(...)' code ##################

            
            # Getting the type of 'k' (line 373)
            k_17160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 15), 'k')
            # Testing the type of an if condition (line 373)
            if_condition_17161 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 373, 12), k_17160)
            # Assigning a type to the variable 'if_condition_17161' (line 373)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 373, 12), 'if_condition_17161', if_condition_17161)
            # SSA begins for if statement (line 373)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to sinh(...): (line 374)
            # Processing the call arguments (line 374)
            # Getting the type of 'a' (line 374)
            a_17163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 28), 'a', False)
            # Getting the type of 'k' (line 374)
            k_17164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 30), 'k', False)
            # Applying the binary operator '*' (line 374)
            result_mul_17165 = python_operator(stypy.reporting.localization.Localization(__file__, 374, 28), '*', a_17163, k_17164)
            
            # Processing the call keyword arguments (line 374)
            kwargs_17166 = {}
            # Getting the type of 'sinh' (line 374)
            sinh_17162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 23), 'sinh', False)
            # Calling sinh(args, kwargs) (line 374)
            sinh_call_result_17167 = invoke(stypy.reporting.localization.Localization(__file__, 374, 23), sinh_17162, *[result_mul_17165], **kwargs_17166)
            
            
            # Call to cosh(...): (line 374)
            # Processing the call arguments (line 374)
            # Getting the type of 'b' (line 374)
            b_17169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 38), 'b', False)
            # Getting the type of 'k' (line 374)
            k_17170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 40), 'k', False)
            # Applying the binary operator '*' (line 374)
            result_mul_17171 = python_operator(stypy.reporting.localization.Localization(__file__, 374, 38), '*', b_17169, k_17170)
            
            # Processing the call keyword arguments (line 374)
            kwargs_17172 = {}
            # Getting the type of 'cosh' (line 374)
            cosh_17168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 33), 'cosh', False)
            # Calling cosh(args, kwargs) (line 374)
            cosh_call_result_17173 = invoke(stypy.reporting.localization.Localization(__file__, 374, 33), cosh_17168, *[result_mul_17171], **kwargs_17172)
            
            # Applying the binary operator 'div' (line 374)
            result_div_17174 = python_operator(stypy.reporting.localization.Localization(__file__, 374, 23), 'div', sinh_call_result_17167, cosh_call_result_17173)
            
            # Assigning a type to the variable 'stypy_return_type' (line 374)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 374, 16), 'stypy_return_type', result_div_17174)
            # SSA join for if statement (line 373)
            module_type_store = module_type_store.join_ssa_context()
            
            int_17175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 19), 'int')
            # Assigning a type to the variable 'stypy_return_type' (line 375)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 375, 12), 'stypy_return_type', int_17175)
            
            # ################# End of 'kernel(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'kernel' in the type store
            # Getting the type of 'stypy_return_type' (line 372)
            stypy_return_type_17176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_17176)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'kernel'
            return stypy_return_type_17176

        # Assigning a type to the variable 'kernel' (line 372)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 372, 8), 'kernel', kernel)
        
        # Assigning a Call to a Name (line 376):
        
        # Assigning a Call to a Name (line 376):
        
        # Call to init_convolution_kernel(...): (line 376)
        # Processing the call arguments (line 376)
        # Getting the type of 'n' (line 376)
        n_17179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 49), 'n', False)
        # Getting the type of 'kernel' (line 376)
        kernel_17180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 51), 'kernel', False)
        # Processing the call keyword arguments (line 376)
        int_17181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 376, 60), 'int')
        keyword_17182 = int_17181
        kwargs_17183 = {'d': keyword_17182}
        # Getting the type of 'convolve' (line 376)
        convolve_17177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 16), 'convolve', False)
        # Obtaining the member 'init_convolution_kernel' of a type (line 376)
        init_convolution_kernel_17178 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 376, 16), convolve_17177, 'init_convolution_kernel')
        # Calling init_convolution_kernel(args, kwargs) (line 376)
        init_convolution_kernel_call_result_17184 = invoke(stypy.reporting.localization.Localization(__file__, 376, 16), init_convolution_kernel_17178, *[n_17179, kernel_17180], **kwargs_17183)
        
        # Assigning a type to the variable 'omega' (line 376)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 376, 8), 'omega', init_convolution_kernel_call_result_17184)
        
        # Assigning a Name to a Subscript (line 377):
        
        # Assigning a Name to a Subscript (line 377):
        # Getting the type of 'omega' (line 377)
        omega_17185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 26), 'omega')
        # Getting the type of '_cache' (line 377)
        _cache_17186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 8), '_cache')
        
        # Obtaining an instance of the builtin type 'tuple' (line 377)
        tuple_17187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 377, 16), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 377)
        # Adding element type (line 377)
        # Getting the type of 'n' (line 377)
        n_17188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 16), 'n')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 377, 16), tuple_17187, n_17188)
        # Adding element type (line 377)
        # Getting the type of 'a' (line 377)
        a_17189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 18), 'a')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 377, 16), tuple_17187, a_17189)
        # Adding element type (line 377)
        # Getting the type of 'b' (line 377)
        b_17190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 20), 'b')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 377, 16), tuple_17187, b_17190)
        
        # Storing an element on a container (line 377)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 377, 8), _cache_17186, (tuple_17187, omega_17185))

        if more_types_in_union_17145:
            # SSA join for if statement (line 367)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 378):
    
    # Assigning a Call to a Name (line 378):
    
    # Call to _datacopied(...): (line 378)
    # Processing the call arguments (line 378)
    # Getting the type of 'tmp' (line 378)
    tmp_17192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 30), 'tmp', False)
    # Getting the type of 'x' (line 378)
    x_17193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 35), 'x', False)
    # Processing the call keyword arguments (line 378)
    kwargs_17194 = {}
    # Getting the type of '_datacopied' (line 378)
    _datacopied_17191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 18), '_datacopied', False)
    # Calling _datacopied(args, kwargs) (line 378)
    _datacopied_call_result_17195 = invoke(stypy.reporting.localization.Localization(__file__, 378, 18), _datacopied_17191, *[tmp_17192, x_17193], **kwargs_17194)
    
    # Assigning a type to the variable 'overwrite_x' (line 378)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 378, 4), 'overwrite_x', _datacopied_call_result_17195)
    
    # Call to convolve(...): (line 379)
    # Processing the call arguments (line 379)
    # Getting the type of 'tmp' (line 379)
    tmp_17198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 29), 'tmp', False)
    # Getting the type of 'omega' (line 379)
    omega_17199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 33), 'omega', False)
    # Processing the call keyword arguments (line 379)
    int_17200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 379, 54), 'int')
    keyword_17201 = int_17200
    # Getting the type of 'overwrite_x' (line 379)
    overwrite_x_17202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 68), 'overwrite_x', False)
    keyword_17203 = overwrite_x_17202
    kwargs_17204 = {'overwrite_x': keyword_17203, 'swap_real_imag': keyword_17201}
    # Getting the type of 'convolve' (line 379)
    convolve_17196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 11), 'convolve', False)
    # Obtaining the member 'convolve' of a type (line 379)
    convolve_17197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 379, 11), convolve_17196, 'convolve')
    # Calling convolve(args, kwargs) (line 379)
    convolve_call_result_17205 = invoke(stypy.reporting.localization.Localization(__file__, 379, 11), convolve_17197, *[tmp_17198, omega_17199], **kwargs_17204)
    
    # Assigning a type to the variable 'stypy_return_type' (line 379)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 379, 4), 'stypy_return_type', convolve_call_result_17205)
    
    # ################# End of 'sc_diff(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'sc_diff' in the type store
    # Getting the type of 'stypy_return_type' (line 332)
    stypy_return_type_17206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_17206)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'sc_diff'
    return stypy_return_type_17206

# Assigning a type to the variable 'sc_diff' (line 332)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 0), 'sc_diff', sc_diff)
# Deleting a member
module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 380, 0), module_type_store, '_cache')

# Assigning a Dict to a Name (line 383):

# Assigning a Dict to a Name (line 383):

# Obtaining an instance of the builtin type 'dict' (line 383)
dict_17207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 9), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 383)

# Assigning a type to the variable '_cache' (line 383)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 0), '_cache', dict_17207)

@norecursion
def ss_diff(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 386)
    None_17208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 28), 'None')
    # Getting the type of '_cache' (line 386)
    _cache_17209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 41), '_cache')
    defaults = [None_17208, _cache_17209]
    # Create a new context for function 'ss_diff'
    module_type_store = module_type_store.open_function_context('ss_diff', 386, 0, False)
    
    # Passed parameters checking function
    ss_diff.stypy_localization = localization
    ss_diff.stypy_type_of_self = None
    ss_diff.stypy_type_store = module_type_store
    ss_diff.stypy_function_name = 'ss_diff'
    ss_diff.stypy_param_names_list = ['x', 'a', 'b', 'period', '_cache']
    ss_diff.stypy_varargs_param_name = None
    ss_diff.stypy_kwargs_param_name = None
    ss_diff.stypy_call_defaults = defaults
    ss_diff.stypy_call_varargs = varargs
    ss_diff.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'ss_diff', ['x', 'a', 'b', 'period', '_cache'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'ss_diff', localization, ['x', 'a', 'b', 'period', '_cache'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'ss_diff(...)' code ##################

    str_17210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 410, (-1)), 'str', '\n    Return (a,b)-sinh/sinh pseudo-derivative of a periodic sequence x.\n\n    If x_j and y_j are Fourier coefficients of periodic functions x\n    and y, respectively, then::\n\n      y_j = sinh(j*a*2*pi/period)/sinh(j*b*2*pi/period) * x_j\n      y_0 = a/b * x_0\n\n    Parameters\n    ----------\n    x : array_like\n        The array to take the pseudo-derivative from.\n    a,b\n        Defines the parameters of the sinh/sinh pseudo-differential\n        operator.\n    period : float, optional\n        The period of the sequence x. Default is ``2*pi``.\n\n    Notes\n    -----\n    ``ss_diff(ss_diff(x,a,b),b,a) == x``\n\n    ')
    
    # Assigning a Call to a Name (line 411):
    
    # Assigning a Call to a Name (line 411):
    
    # Call to asarray(...): (line 411)
    # Processing the call arguments (line 411)
    # Getting the type of 'x' (line 411)
    x_17212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 18), 'x', False)
    # Processing the call keyword arguments (line 411)
    kwargs_17213 = {}
    # Getting the type of 'asarray' (line 411)
    asarray_17211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 10), 'asarray', False)
    # Calling asarray(args, kwargs) (line 411)
    asarray_call_result_17214 = invoke(stypy.reporting.localization.Localization(__file__, 411, 10), asarray_17211, *[x_17212], **kwargs_17213)
    
    # Assigning a type to the variable 'tmp' (line 411)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 411, 4), 'tmp', asarray_call_result_17214)
    
    
    # Call to iscomplexobj(...): (line 412)
    # Processing the call arguments (line 412)
    # Getting the type of 'tmp' (line 412)
    tmp_17216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 20), 'tmp', False)
    # Processing the call keyword arguments (line 412)
    kwargs_17217 = {}
    # Getting the type of 'iscomplexobj' (line 412)
    iscomplexobj_17215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 7), 'iscomplexobj', False)
    # Calling iscomplexobj(args, kwargs) (line 412)
    iscomplexobj_call_result_17218 = invoke(stypy.reporting.localization.Localization(__file__, 412, 7), iscomplexobj_17215, *[tmp_17216], **kwargs_17217)
    
    # Testing the type of an if condition (line 412)
    if_condition_17219 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 412, 4), iscomplexobj_call_result_17218)
    # Assigning a type to the variable 'if_condition_17219' (line 412)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 412, 4), 'if_condition_17219', if_condition_17219)
    # SSA begins for if statement (line 412)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ss_diff(...): (line 413)
    # Processing the call arguments (line 413)
    # Getting the type of 'tmp' (line 413)
    tmp_17221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 23), 'tmp', False)
    # Obtaining the member 'real' of a type (line 413)
    real_17222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 413, 23), tmp_17221, 'real')
    # Getting the type of 'a' (line 413)
    a_17223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 32), 'a', False)
    # Getting the type of 'b' (line 413)
    b_17224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 34), 'b', False)
    # Getting the type of 'period' (line 413)
    period_17225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 36), 'period', False)
    # Processing the call keyword arguments (line 413)
    kwargs_17226 = {}
    # Getting the type of 'ss_diff' (line 413)
    ss_diff_17220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 15), 'ss_diff', False)
    # Calling ss_diff(args, kwargs) (line 413)
    ss_diff_call_result_17227 = invoke(stypy.reporting.localization.Localization(__file__, 413, 15), ss_diff_17220, *[real_17222, a_17223, b_17224, period_17225], **kwargs_17226)
    
    complex_17228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 414, 15), 'complex')
    
    # Call to ss_diff(...): (line 414)
    # Processing the call arguments (line 414)
    # Getting the type of 'tmp' (line 414)
    tmp_17230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 26), 'tmp', False)
    # Obtaining the member 'imag' of a type (line 414)
    imag_17231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 414, 26), tmp_17230, 'imag')
    # Getting the type of 'a' (line 414)
    a_17232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 35), 'a', False)
    # Getting the type of 'b' (line 414)
    b_17233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 37), 'b', False)
    # Getting the type of 'period' (line 414)
    period_17234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 39), 'period', False)
    # Processing the call keyword arguments (line 414)
    kwargs_17235 = {}
    # Getting the type of 'ss_diff' (line 414)
    ss_diff_17229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 18), 'ss_diff', False)
    # Calling ss_diff(args, kwargs) (line 414)
    ss_diff_call_result_17236 = invoke(stypy.reporting.localization.Localization(__file__, 414, 18), ss_diff_17229, *[imag_17231, a_17232, b_17233, period_17234], **kwargs_17235)
    
    # Applying the binary operator '*' (line 414)
    result_mul_17237 = python_operator(stypy.reporting.localization.Localization(__file__, 414, 15), '*', complex_17228, ss_diff_call_result_17236)
    
    # Applying the binary operator '+' (line 413)
    result_add_17238 = python_operator(stypy.reporting.localization.Localization(__file__, 413, 15), '+', ss_diff_call_result_17227, result_mul_17237)
    
    # Assigning a type to the variable 'stypy_return_type' (line 413)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 413, 8), 'stypy_return_type', result_add_17238)
    # SSA join for if statement (line 412)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 415)
    # Getting the type of 'period' (line 415)
    period_17239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 4), 'period')
    # Getting the type of 'None' (line 415)
    None_17240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 21), 'None')
    
    (may_be_17241, more_types_in_union_17242) = may_not_be_none(period_17239, None_17240)

    if may_be_17241:

        if more_types_in_union_17242:
            # Runtime conditional SSA (line 415)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a BinOp to a Name (line 416):
        
        # Assigning a BinOp to a Name (line 416):
        # Getting the type of 'a' (line 416)
        a_17243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 12), 'a')
        int_17244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 14), 'int')
        # Applying the binary operator '*' (line 416)
        result_mul_17245 = python_operator(stypy.reporting.localization.Localization(__file__, 416, 12), '*', a_17243, int_17244)
        
        # Getting the type of 'pi' (line 416)
        pi_17246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 16), 'pi')
        # Applying the binary operator '*' (line 416)
        result_mul_17247 = python_operator(stypy.reporting.localization.Localization(__file__, 416, 15), '*', result_mul_17245, pi_17246)
        
        # Getting the type of 'period' (line 416)
        period_17248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 19), 'period')
        # Applying the binary operator 'div' (line 416)
        result_div_17249 = python_operator(stypy.reporting.localization.Localization(__file__, 416, 18), 'div', result_mul_17247, period_17248)
        
        # Assigning a type to the variable 'a' (line 416)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 416, 8), 'a', result_div_17249)
        
        # Assigning a BinOp to a Name (line 417):
        
        # Assigning a BinOp to a Name (line 417):
        # Getting the type of 'b' (line 417)
        b_17250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 12), 'b')
        int_17251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 417, 14), 'int')
        # Applying the binary operator '*' (line 417)
        result_mul_17252 = python_operator(stypy.reporting.localization.Localization(__file__, 417, 12), '*', b_17250, int_17251)
        
        # Getting the type of 'pi' (line 417)
        pi_17253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 16), 'pi')
        # Applying the binary operator '*' (line 417)
        result_mul_17254 = python_operator(stypy.reporting.localization.Localization(__file__, 417, 15), '*', result_mul_17252, pi_17253)
        
        # Getting the type of 'period' (line 417)
        period_17255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 19), 'period')
        # Applying the binary operator 'div' (line 417)
        result_div_17256 = python_operator(stypy.reporting.localization.Localization(__file__, 417, 18), 'div', result_mul_17254, period_17255)
        
        # Assigning a type to the variable 'b' (line 417)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 417, 8), 'b', result_div_17256)

        if more_types_in_union_17242:
            # SSA join for if statement (line 415)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 418):
    
    # Assigning a Call to a Name (line 418):
    
    # Call to len(...): (line 418)
    # Processing the call arguments (line 418)
    # Getting the type of 'x' (line 418)
    x_17258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 12), 'x', False)
    # Processing the call keyword arguments (line 418)
    kwargs_17259 = {}
    # Getting the type of 'len' (line 418)
    len_17257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 8), 'len', False)
    # Calling len(args, kwargs) (line 418)
    len_call_result_17260 = invoke(stypy.reporting.localization.Localization(__file__, 418, 8), len_17257, *[x_17258], **kwargs_17259)
    
    # Assigning a type to the variable 'n' (line 418)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 418, 4), 'n', len_call_result_17260)
    
    # Assigning a Call to a Name (line 419):
    
    # Assigning a Call to a Name (line 419):
    
    # Call to get(...): (line 419)
    # Processing the call arguments (line 419)
    
    # Obtaining an instance of the builtin type 'tuple' (line 419)
    tuple_17263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 419, 24), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 419)
    # Adding element type (line 419)
    # Getting the type of 'n' (line 419)
    n_17264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 24), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 419, 24), tuple_17263, n_17264)
    # Adding element type (line 419)
    # Getting the type of 'a' (line 419)
    a_17265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 26), 'a', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 419, 24), tuple_17263, a_17265)
    # Adding element type (line 419)
    # Getting the type of 'b' (line 419)
    b_17266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 28), 'b', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 419, 24), tuple_17263, b_17266)
    
    # Processing the call keyword arguments (line 419)
    kwargs_17267 = {}
    # Getting the type of '_cache' (line 419)
    _cache_17261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 12), '_cache', False)
    # Obtaining the member 'get' of a type (line 419)
    get_17262 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 419, 12), _cache_17261, 'get')
    # Calling get(args, kwargs) (line 419)
    get_call_result_17268 = invoke(stypy.reporting.localization.Localization(__file__, 419, 12), get_17262, *[tuple_17263], **kwargs_17267)
    
    # Assigning a type to the variable 'omega' (line 419)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 419, 4), 'omega', get_call_result_17268)
    
    # Type idiom detected: calculating its left and rigth part (line 420)
    # Getting the type of 'omega' (line 420)
    omega_17269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 7), 'omega')
    # Getting the type of 'None' (line 420)
    None_17270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 16), 'None')
    
    (may_be_17271, more_types_in_union_17272) = may_be_none(omega_17269, None_17270)

    if may_be_17271:

        if more_types_in_union_17272:
            # Runtime conditional SSA (line 420)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        
        
        # Call to len(...): (line 421)
        # Processing the call arguments (line 421)
        # Getting the type of '_cache' (line 421)
        _cache_17274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 15), '_cache', False)
        # Processing the call keyword arguments (line 421)
        kwargs_17275 = {}
        # Getting the type of 'len' (line 421)
        len_17273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 11), 'len', False)
        # Calling len(args, kwargs) (line 421)
        len_call_result_17276 = invoke(stypy.reporting.localization.Localization(__file__, 421, 11), len_17273, *[_cache_17274], **kwargs_17275)
        
        int_17277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 421, 25), 'int')
        # Applying the binary operator '>' (line 421)
        result_gt_17278 = python_operator(stypy.reporting.localization.Localization(__file__, 421, 11), '>', len_call_result_17276, int_17277)
        
        # Testing the type of an if condition (line 421)
        if_condition_17279 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 421, 8), result_gt_17278)
        # Assigning a type to the variable 'if_condition_17279' (line 421)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 421, 8), 'if_condition_17279', if_condition_17279)
        # SSA begins for if statement (line 421)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of '_cache' (line 422)
        _cache_17280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 18), '_cache')
        # Testing the type of an if condition (line 422)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 422, 12), _cache_17280)
        # SSA begins for while statement (line 422)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Call to popitem(...): (line 423)
        # Processing the call keyword arguments (line 423)
        kwargs_17283 = {}
        # Getting the type of '_cache' (line 423)
        _cache_17281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 16), '_cache', False)
        # Obtaining the member 'popitem' of a type (line 423)
        popitem_17282 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 16), _cache_17281, 'popitem')
        # Calling popitem(args, kwargs) (line 423)
        popitem_call_result_17284 = invoke(stypy.reporting.localization.Localization(__file__, 423, 16), popitem_17282, *[], **kwargs_17283)
        
        # SSA join for while statement (line 422)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 421)
        module_type_store = module_type_store.join_ssa_context()
        

        @norecursion
        def kernel(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            # Getting the type of 'a' (line 425)
            a_17285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 23), 'a')
            # Getting the type of 'b' (line 425)
            b_17286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 27), 'b')
            defaults = [a_17285, b_17286]
            # Create a new context for function 'kernel'
            module_type_store = module_type_store.open_function_context('kernel', 425, 8, False)
            
            # Passed parameters checking function
            kernel.stypy_localization = localization
            kernel.stypy_type_of_self = None
            kernel.stypy_type_store = module_type_store
            kernel.stypy_function_name = 'kernel'
            kernel.stypy_param_names_list = ['k', 'a', 'b']
            kernel.stypy_varargs_param_name = None
            kernel.stypy_kwargs_param_name = None
            kernel.stypy_call_defaults = defaults
            kernel.stypy_call_varargs = varargs
            kernel.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'kernel', ['k', 'a', 'b'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'kernel', localization, ['k', 'a', 'b'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'kernel(...)' code ##################

            
            # Getting the type of 'k' (line 426)
            k_17287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 15), 'k')
            # Testing the type of an if condition (line 426)
            if_condition_17288 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 426, 12), k_17287)
            # Assigning a type to the variable 'if_condition_17288' (line 426)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 426, 12), 'if_condition_17288', if_condition_17288)
            # SSA begins for if statement (line 426)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to sinh(...): (line 427)
            # Processing the call arguments (line 427)
            # Getting the type of 'a' (line 427)
            a_17290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 28), 'a', False)
            # Getting the type of 'k' (line 427)
            k_17291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 30), 'k', False)
            # Applying the binary operator '*' (line 427)
            result_mul_17292 = python_operator(stypy.reporting.localization.Localization(__file__, 427, 28), '*', a_17290, k_17291)
            
            # Processing the call keyword arguments (line 427)
            kwargs_17293 = {}
            # Getting the type of 'sinh' (line 427)
            sinh_17289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 23), 'sinh', False)
            # Calling sinh(args, kwargs) (line 427)
            sinh_call_result_17294 = invoke(stypy.reporting.localization.Localization(__file__, 427, 23), sinh_17289, *[result_mul_17292], **kwargs_17293)
            
            
            # Call to sinh(...): (line 427)
            # Processing the call arguments (line 427)
            # Getting the type of 'b' (line 427)
            b_17296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 38), 'b', False)
            # Getting the type of 'k' (line 427)
            k_17297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 40), 'k', False)
            # Applying the binary operator '*' (line 427)
            result_mul_17298 = python_operator(stypy.reporting.localization.Localization(__file__, 427, 38), '*', b_17296, k_17297)
            
            # Processing the call keyword arguments (line 427)
            kwargs_17299 = {}
            # Getting the type of 'sinh' (line 427)
            sinh_17295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 33), 'sinh', False)
            # Calling sinh(args, kwargs) (line 427)
            sinh_call_result_17300 = invoke(stypy.reporting.localization.Localization(__file__, 427, 33), sinh_17295, *[result_mul_17298], **kwargs_17299)
            
            # Applying the binary operator 'div' (line 427)
            result_div_17301 = python_operator(stypy.reporting.localization.Localization(__file__, 427, 23), 'div', sinh_call_result_17294, sinh_call_result_17300)
            
            # Assigning a type to the variable 'stypy_return_type' (line 427)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 427, 16), 'stypy_return_type', result_div_17301)
            # SSA join for if statement (line 426)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Call to float(...): (line 428)
            # Processing the call arguments (line 428)
            # Getting the type of 'a' (line 428)
            a_17303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 25), 'a', False)
            # Processing the call keyword arguments (line 428)
            kwargs_17304 = {}
            # Getting the type of 'float' (line 428)
            float_17302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 19), 'float', False)
            # Calling float(args, kwargs) (line 428)
            float_call_result_17305 = invoke(stypy.reporting.localization.Localization(__file__, 428, 19), float_17302, *[a_17303], **kwargs_17304)
            
            # Getting the type of 'b' (line 428)
            b_17306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 28), 'b')
            # Applying the binary operator 'div' (line 428)
            result_div_17307 = python_operator(stypy.reporting.localization.Localization(__file__, 428, 19), 'div', float_call_result_17305, b_17306)
            
            # Assigning a type to the variable 'stypy_return_type' (line 428)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 428, 12), 'stypy_return_type', result_div_17307)
            
            # ################# End of 'kernel(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'kernel' in the type store
            # Getting the type of 'stypy_return_type' (line 425)
            stypy_return_type_17308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_17308)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'kernel'
            return stypy_return_type_17308

        # Assigning a type to the variable 'kernel' (line 425)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 425, 8), 'kernel', kernel)
        
        # Assigning a Call to a Name (line 429):
        
        # Assigning a Call to a Name (line 429):
        
        # Call to init_convolution_kernel(...): (line 429)
        # Processing the call arguments (line 429)
        # Getting the type of 'n' (line 429)
        n_17311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 49), 'n', False)
        # Getting the type of 'kernel' (line 429)
        kernel_17312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 51), 'kernel', False)
        # Processing the call keyword arguments (line 429)
        kwargs_17313 = {}
        # Getting the type of 'convolve' (line 429)
        convolve_17309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 16), 'convolve', False)
        # Obtaining the member 'init_convolution_kernel' of a type (line 429)
        init_convolution_kernel_17310 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 429, 16), convolve_17309, 'init_convolution_kernel')
        # Calling init_convolution_kernel(args, kwargs) (line 429)
        init_convolution_kernel_call_result_17314 = invoke(stypy.reporting.localization.Localization(__file__, 429, 16), init_convolution_kernel_17310, *[n_17311, kernel_17312], **kwargs_17313)
        
        # Assigning a type to the variable 'omega' (line 429)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 429, 8), 'omega', init_convolution_kernel_call_result_17314)
        
        # Assigning a Name to a Subscript (line 430):
        
        # Assigning a Name to a Subscript (line 430):
        # Getting the type of 'omega' (line 430)
        omega_17315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 26), 'omega')
        # Getting the type of '_cache' (line 430)
        _cache_17316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 8), '_cache')
        
        # Obtaining an instance of the builtin type 'tuple' (line 430)
        tuple_17317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 430, 16), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 430)
        # Adding element type (line 430)
        # Getting the type of 'n' (line 430)
        n_17318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 16), 'n')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 430, 16), tuple_17317, n_17318)
        # Adding element type (line 430)
        # Getting the type of 'a' (line 430)
        a_17319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 18), 'a')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 430, 16), tuple_17317, a_17319)
        # Adding element type (line 430)
        # Getting the type of 'b' (line 430)
        b_17320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 20), 'b')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 430, 16), tuple_17317, b_17320)
        
        # Storing an element on a container (line 430)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 430, 8), _cache_17316, (tuple_17317, omega_17315))

        if more_types_in_union_17272:
            # SSA join for if statement (line 420)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 431):
    
    # Assigning a Call to a Name (line 431):
    
    # Call to _datacopied(...): (line 431)
    # Processing the call arguments (line 431)
    # Getting the type of 'tmp' (line 431)
    tmp_17322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 30), 'tmp', False)
    # Getting the type of 'x' (line 431)
    x_17323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 35), 'x', False)
    # Processing the call keyword arguments (line 431)
    kwargs_17324 = {}
    # Getting the type of '_datacopied' (line 431)
    _datacopied_17321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 18), '_datacopied', False)
    # Calling _datacopied(args, kwargs) (line 431)
    _datacopied_call_result_17325 = invoke(stypy.reporting.localization.Localization(__file__, 431, 18), _datacopied_17321, *[tmp_17322, x_17323], **kwargs_17324)
    
    # Assigning a type to the variable 'overwrite_x' (line 431)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 431, 4), 'overwrite_x', _datacopied_call_result_17325)
    
    # Call to convolve(...): (line 432)
    # Processing the call arguments (line 432)
    # Getting the type of 'tmp' (line 432)
    tmp_17328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 29), 'tmp', False)
    # Getting the type of 'omega' (line 432)
    omega_17329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 33), 'omega', False)
    # Processing the call keyword arguments (line 432)
    # Getting the type of 'overwrite_x' (line 432)
    overwrite_x_17330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 51), 'overwrite_x', False)
    keyword_17331 = overwrite_x_17330
    kwargs_17332 = {'overwrite_x': keyword_17331}
    # Getting the type of 'convolve' (line 432)
    convolve_17326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 11), 'convolve', False)
    # Obtaining the member 'convolve' of a type (line 432)
    convolve_17327 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 432, 11), convolve_17326, 'convolve')
    # Calling convolve(args, kwargs) (line 432)
    convolve_call_result_17333 = invoke(stypy.reporting.localization.Localization(__file__, 432, 11), convolve_17327, *[tmp_17328, omega_17329], **kwargs_17332)
    
    # Assigning a type to the variable 'stypy_return_type' (line 432)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 432, 4), 'stypy_return_type', convolve_call_result_17333)
    
    # ################# End of 'ss_diff(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'ss_diff' in the type store
    # Getting the type of 'stypy_return_type' (line 386)
    stypy_return_type_17334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_17334)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'ss_diff'
    return stypy_return_type_17334

# Assigning a type to the variable 'ss_diff' (line 386)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 386, 0), 'ss_diff', ss_diff)
# Deleting a member
module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 433, 0), module_type_store, '_cache')

# Assigning a Dict to a Name (line 436):

# Assigning a Dict to a Name (line 436):

# Obtaining an instance of the builtin type 'dict' (line 436)
dict_17335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 436, 9), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 436)

# Assigning a type to the variable '_cache' (line 436)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 436, 0), '_cache', dict_17335)

@norecursion
def cc_diff(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 439)
    None_17336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 28), 'None')
    # Getting the type of '_cache' (line 439)
    _cache_17337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 41), '_cache')
    defaults = [None_17336, _cache_17337]
    # Create a new context for function 'cc_diff'
    module_type_store = module_type_store.open_function_context('cc_diff', 439, 0, False)
    
    # Passed parameters checking function
    cc_diff.stypy_localization = localization
    cc_diff.stypy_type_of_self = None
    cc_diff.stypy_type_store = module_type_store
    cc_diff.stypy_function_name = 'cc_diff'
    cc_diff.stypy_param_names_list = ['x', 'a', 'b', 'period', '_cache']
    cc_diff.stypy_varargs_param_name = None
    cc_diff.stypy_kwargs_param_name = None
    cc_diff.stypy_call_defaults = defaults
    cc_diff.stypy_call_varargs = varargs
    cc_diff.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'cc_diff', ['x', 'a', 'b', 'period', '_cache'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'cc_diff', localization, ['x', 'a', 'b', 'period', '_cache'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'cc_diff(...)' code ##################

    str_17338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 467, (-1)), 'str', '\n    Return (a,b)-cosh/cosh pseudo-derivative of a periodic sequence.\n\n    If x_j and y_j are Fourier coefficients of periodic functions x\n    and y, respectively, then::\n\n      y_j = cosh(j*a*2*pi/period)/cosh(j*b*2*pi/period) * x_j\n\n    Parameters\n    ----------\n    x : array_like\n        The array to take the pseudo-derivative from.\n    a,b : float\n        Defines the parameters of the sinh/sinh pseudo-differential\n        operator.\n    period : float, optional\n        The period of the sequence x. Default is ``2*pi``.\n\n    Returns\n    -------\n    cc_diff : ndarray\n        Pseudo-derivative of periodic sequence `x`.\n\n    Notes\n    -----\n    ``cc_diff(cc_diff(x,a,b),b,a) == x``\n\n    ')
    
    # Assigning a Call to a Name (line 468):
    
    # Assigning a Call to a Name (line 468):
    
    # Call to asarray(...): (line 468)
    # Processing the call arguments (line 468)
    # Getting the type of 'x' (line 468)
    x_17340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 18), 'x', False)
    # Processing the call keyword arguments (line 468)
    kwargs_17341 = {}
    # Getting the type of 'asarray' (line 468)
    asarray_17339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 10), 'asarray', False)
    # Calling asarray(args, kwargs) (line 468)
    asarray_call_result_17342 = invoke(stypy.reporting.localization.Localization(__file__, 468, 10), asarray_17339, *[x_17340], **kwargs_17341)
    
    # Assigning a type to the variable 'tmp' (line 468)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 468, 4), 'tmp', asarray_call_result_17342)
    
    
    # Call to iscomplexobj(...): (line 469)
    # Processing the call arguments (line 469)
    # Getting the type of 'tmp' (line 469)
    tmp_17344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 20), 'tmp', False)
    # Processing the call keyword arguments (line 469)
    kwargs_17345 = {}
    # Getting the type of 'iscomplexobj' (line 469)
    iscomplexobj_17343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 7), 'iscomplexobj', False)
    # Calling iscomplexobj(args, kwargs) (line 469)
    iscomplexobj_call_result_17346 = invoke(stypy.reporting.localization.Localization(__file__, 469, 7), iscomplexobj_17343, *[tmp_17344], **kwargs_17345)
    
    # Testing the type of an if condition (line 469)
    if_condition_17347 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 469, 4), iscomplexobj_call_result_17346)
    # Assigning a type to the variable 'if_condition_17347' (line 469)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 469, 4), 'if_condition_17347', if_condition_17347)
    # SSA begins for if statement (line 469)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to cc_diff(...): (line 470)
    # Processing the call arguments (line 470)
    # Getting the type of 'tmp' (line 470)
    tmp_17349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 23), 'tmp', False)
    # Obtaining the member 'real' of a type (line 470)
    real_17350 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 470, 23), tmp_17349, 'real')
    # Getting the type of 'a' (line 470)
    a_17351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 32), 'a', False)
    # Getting the type of 'b' (line 470)
    b_17352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 34), 'b', False)
    # Getting the type of 'period' (line 470)
    period_17353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 36), 'period', False)
    # Processing the call keyword arguments (line 470)
    kwargs_17354 = {}
    # Getting the type of 'cc_diff' (line 470)
    cc_diff_17348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 15), 'cc_diff', False)
    # Calling cc_diff(args, kwargs) (line 470)
    cc_diff_call_result_17355 = invoke(stypy.reporting.localization.Localization(__file__, 470, 15), cc_diff_17348, *[real_17350, a_17351, b_17352, period_17353], **kwargs_17354)
    
    complex_17356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 471, 15), 'complex')
    
    # Call to cc_diff(...): (line 471)
    # Processing the call arguments (line 471)
    # Getting the type of 'tmp' (line 471)
    tmp_17358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 26), 'tmp', False)
    # Obtaining the member 'imag' of a type (line 471)
    imag_17359 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 471, 26), tmp_17358, 'imag')
    # Getting the type of 'a' (line 471)
    a_17360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 35), 'a', False)
    # Getting the type of 'b' (line 471)
    b_17361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 37), 'b', False)
    # Getting the type of 'period' (line 471)
    period_17362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 39), 'period', False)
    # Processing the call keyword arguments (line 471)
    kwargs_17363 = {}
    # Getting the type of 'cc_diff' (line 471)
    cc_diff_17357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 18), 'cc_diff', False)
    # Calling cc_diff(args, kwargs) (line 471)
    cc_diff_call_result_17364 = invoke(stypy.reporting.localization.Localization(__file__, 471, 18), cc_diff_17357, *[imag_17359, a_17360, b_17361, period_17362], **kwargs_17363)
    
    # Applying the binary operator '*' (line 471)
    result_mul_17365 = python_operator(stypy.reporting.localization.Localization(__file__, 471, 15), '*', complex_17356, cc_diff_call_result_17364)
    
    # Applying the binary operator '+' (line 470)
    result_add_17366 = python_operator(stypy.reporting.localization.Localization(__file__, 470, 15), '+', cc_diff_call_result_17355, result_mul_17365)
    
    # Assigning a type to the variable 'stypy_return_type' (line 470)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 470, 8), 'stypy_return_type', result_add_17366)
    # SSA join for if statement (line 469)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 472)
    # Getting the type of 'period' (line 472)
    period_17367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 4), 'period')
    # Getting the type of 'None' (line 472)
    None_17368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 21), 'None')
    
    (may_be_17369, more_types_in_union_17370) = may_not_be_none(period_17367, None_17368)

    if may_be_17369:

        if more_types_in_union_17370:
            # Runtime conditional SSA (line 472)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a BinOp to a Name (line 473):
        
        # Assigning a BinOp to a Name (line 473):
        # Getting the type of 'a' (line 473)
        a_17371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 12), 'a')
        int_17372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 473, 14), 'int')
        # Applying the binary operator '*' (line 473)
        result_mul_17373 = python_operator(stypy.reporting.localization.Localization(__file__, 473, 12), '*', a_17371, int_17372)
        
        # Getting the type of 'pi' (line 473)
        pi_17374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 16), 'pi')
        # Applying the binary operator '*' (line 473)
        result_mul_17375 = python_operator(stypy.reporting.localization.Localization(__file__, 473, 15), '*', result_mul_17373, pi_17374)
        
        # Getting the type of 'period' (line 473)
        period_17376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 19), 'period')
        # Applying the binary operator 'div' (line 473)
        result_div_17377 = python_operator(stypy.reporting.localization.Localization(__file__, 473, 18), 'div', result_mul_17375, period_17376)
        
        # Assigning a type to the variable 'a' (line 473)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 473, 8), 'a', result_div_17377)
        
        # Assigning a BinOp to a Name (line 474):
        
        # Assigning a BinOp to a Name (line 474):
        # Getting the type of 'b' (line 474)
        b_17378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 12), 'b')
        int_17379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 474, 14), 'int')
        # Applying the binary operator '*' (line 474)
        result_mul_17380 = python_operator(stypy.reporting.localization.Localization(__file__, 474, 12), '*', b_17378, int_17379)
        
        # Getting the type of 'pi' (line 474)
        pi_17381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 16), 'pi')
        # Applying the binary operator '*' (line 474)
        result_mul_17382 = python_operator(stypy.reporting.localization.Localization(__file__, 474, 15), '*', result_mul_17380, pi_17381)
        
        # Getting the type of 'period' (line 474)
        period_17383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 19), 'period')
        # Applying the binary operator 'div' (line 474)
        result_div_17384 = python_operator(stypy.reporting.localization.Localization(__file__, 474, 18), 'div', result_mul_17382, period_17383)
        
        # Assigning a type to the variable 'b' (line 474)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 474, 8), 'b', result_div_17384)

        if more_types_in_union_17370:
            # SSA join for if statement (line 472)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 475):
    
    # Assigning a Call to a Name (line 475):
    
    # Call to len(...): (line 475)
    # Processing the call arguments (line 475)
    # Getting the type of 'x' (line 475)
    x_17386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 12), 'x', False)
    # Processing the call keyword arguments (line 475)
    kwargs_17387 = {}
    # Getting the type of 'len' (line 475)
    len_17385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 8), 'len', False)
    # Calling len(args, kwargs) (line 475)
    len_call_result_17388 = invoke(stypy.reporting.localization.Localization(__file__, 475, 8), len_17385, *[x_17386], **kwargs_17387)
    
    # Assigning a type to the variable 'n' (line 475)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 475, 4), 'n', len_call_result_17388)
    
    # Assigning a Call to a Name (line 476):
    
    # Assigning a Call to a Name (line 476):
    
    # Call to get(...): (line 476)
    # Processing the call arguments (line 476)
    
    # Obtaining an instance of the builtin type 'tuple' (line 476)
    tuple_17391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 476, 24), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 476)
    # Adding element type (line 476)
    # Getting the type of 'n' (line 476)
    n_17392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 24), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 476, 24), tuple_17391, n_17392)
    # Adding element type (line 476)
    # Getting the type of 'a' (line 476)
    a_17393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 26), 'a', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 476, 24), tuple_17391, a_17393)
    # Adding element type (line 476)
    # Getting the type of 'b' (line 476)
    b_17394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 28), 'b', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 476, 24), tuple_17391, b_17394)
    
    # Processing the call keyword arguments (line 476)
    kwargs_17395 = {}
    # Getting the type of '_cache' (line 476)
    _cache_17389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 12), '_cache', False)
    # Obtaining the member 'get' of a type (line 476)
    get_17390 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 476, 12), _cache_17389, 'get')
    # Calling get(args, kwargs) (line 476)
    get_call_result_17396 = invoke(stypy.reporting.localization.Localization(__file__, 476, 12), get_17390, *[tuple_17391], **kwargs_17395)
    
    # Assigning a type to the variable 'omega' (line 476)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 476, 4), 'omega', get_call_result_17396)
    
    # Type idiom detected: calculating its left and rigth part (line 477)
    # Getting the type of 'omega' (line 477)
    omega_17397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 7), 'omega')
    # Getting the type of 'None' (line 477)
    None_17398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 16), 'None')
    
    (may_be_17399, more_types_in_union_17400) = may_be_none(omega_17397, None_17398)

    if may_be_17399:

        if more_types_in_union_17400:
            # Runtime conditional SSA (line 477)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        
        
        # Call to len(...): (line 478)
        # Processing the call arguments (line 478)
        # Getting the type of '_cache' (line 478)
        _cache_17402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 15), '_cache', False)
        # Processing the call keyword arguments (line 478)
        kwargs_17403 = {}
        # Getting the type of 'len' (line 478)
        len_17401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 11), 'len', False)
        # Calling len(args, kwargs) (line 478)
        len_call_result_17404 = invoke(stypy.reporting.localization.Localization(__file__, 478, 11), len_17401, *[_cache_17402], **kwargs_17403)
        
        int_17405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 478, 25), 'int')
        # Applying the binary operator '>' (line 478)
        result_gt_17406 = python_operator(stypy.reporting.localization.Localization(__file__, 478, 11), '>', len_call_result_17404, int_17405)
        
        # Testing the type of an if condition (line 478)
        if_condition_17407 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 478, 8), result_gt_17406)
        # Assigning a type to the variable 'if_condition_17407' (line 478)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 478, 8), 'if_condition_17407', if_condition_17407)
        # SSA begins for if statement (line 478)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of '_cache' (line 479)
        _cache_17408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 18), '_cache')
        # Testing the type of an if condition (line 479)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 479, 12), _cache_17408)
        # SSA begins for while statement (line 479)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Call to popitem(...): (line 480)
        # Processing the call keyword arguments (line 480)
        kwargs_17411 = {}
        # Getting the type of '_cache' (line 480)
        _cache_17409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 16), '_cache', False)
        # Obtaining the member 'popitem' of a type (line 480)
        popitem_17410 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 480, 16), _cache_17409, 'popitem')
        # Calling popitem(args, kwargs) (line 480)
        popitem_call_result_17412 = invoke(stypy.reporting.localization.Localization(__file__, 480, 16), popitem_17410, *[], **kwargs_17411)
        
        # SSA join for while statement (line 479)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 478)
        module_type_store = module_type_store.join_ssa_context()
        

        @norecursion
        def kernel(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            # Getting the type of 'a' (line 482)
            a_17413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 23), 'a')
            # Getting the type of 'b' (line 482)
            b_17414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 27), 'b')
            defaults = [a_17413, b_17414]
            # Create a new context for function 'kernel'
            module_type_store = module_type_store.open_function_context('kernel', 482, 8, False)
            
            # Passed parameters checking function
            kernel.stypy_localization = localization
            kernel.stypy_type_of_self = None
            kernel.stypy_type_store = module_type_store
            kernel.stypy_function_name = 'kernel'
            kernel.stypy_param_names_list = ['k', 'a', 'b']
            kernel.stypy_varargs_param_name = None
            kernel.stypy_kwargs_param_name = None
            kernel.stypy_call_defaults = defaults
            kernel.stypy_call_varargs = varargs
            kernel.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'kernel', ['k', 'a', 'b'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'kernel', localization, ['k', 'a', 'b'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'kernel(...)' code ##################

            
            # Call to cosh(...): (line 483)
            # Processing the call arguments (line 483)
            # Getting the type of 'a' (line 483)
            a_17416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 24), 'a', False)
            # Getting the type of 'k' (line 483)
            k_17417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 26), 'k', False)
            # Applying the binary operator '*' (line 483)
            result_mul_17418 = python_operator(stypy.reporting.localization.Localization(__file__, 483, 24), '*', a_17416, k_17417)
            
            # Processing the call keyword arguments (line 483)
            kwargs_17419 = {}
            # Getting the type of 'cosh' (line 483)
            cosh_17415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 19), 'cosh', False)
            # Calling cosh(args, kwargs) (line 483)
            cosh_call_result_17420 = invoke(stypy.reporting.localization.Localization(__file__, 483, 19), cosh_17415, *[result_mul_17418], **kwargs_17419)
            
            
            # Call to cosh(...): (line 483)
            # Processing the call arguments (line 483)
            # Getting the type of 'b' (line 483)
            b_17422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 34), 'b', False)
            # Getting the type of 'k' (line 483)
            k_17423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 36), 'k', False)
            # Applying the binary operator '*' (line 483)
            result_mul_17424 = python_operator(stypy.reporting.localization.Localization(__file__, 483, 34), '*', b_17422, k_17423)
            
            # Processing the call keyword arguments (line 483)
            kwargs_17425 = {}
            # Getting the type of 'cosh' (line 483)
            cosh_17421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 29), 'cosh', False)
            # Calling cosh(args, kwargs) (line 483)
            cosh_call_result_17426 = invoke(stypy.reporting.localization.Localization(__file__, 483, 29), cosh_17421, *[result_mul_17424], **kwargs_17425)
            
            # Applying the binary operator 'div' (line 483)
            result_div_17427 = python_operator(stypy.reporting.localization.Localization(__file__, 483, 19), 'div', cosh_call_result_17420, cosh_call_result_17426)
            
            # Assigning a type to the variable 'stypy_return_type' (line 483)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 483, 12), 'stypy_return_type', result_div_17427)
            
            # ################# End of 'kernel(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'kernel' in the type store
            # Getting the type of 'stypy_return_type' (line 482)
            stypy_return_type_17428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_17428)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'kernel'
            return stypy_return_type_17428

        # Assigning a type to the variable 'kernel' (line 482)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 482, 8), 'kernel', kernel)
        
        # Assigning a Call to a Name (line 484):
        
        # Assigning a Call to a Name (line 484):
        
        # Call to init_convolution_kernel(...): (line 484)
        # Processing the call arguments (line 484)
        # Getting the type of 'n' (line 484)
        n_17431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 49), 'n', False)
        # Getting the type of 'kernel' (line 484)
        kernel_17432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 51), 'kernel', False)
        # Processing the call keyword arguments (line 484)
        kwargs_17433 = {}
        # Getting the type of 'convolve' (line 484)
        convolve_17429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 16), 'convolve', False)
        # Obtaining the member 'init_convolution_kernel' of a type (line 484)
        init_convolution_kernel_17430 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 484, 16), convolve_17429, 'init_convolution_kernel')
        # Calling init_convolution_kernel(args, kwargs) (line 484)
        init_convolution_kernel_call_result_17434 = invoke(stypy.reporting.localization.Localization(__file__, 484, 16), init_convolution_kernel_17430, *[n_17431, kernel_17432], **kwargs_17433)
        
        # Assigning a type to the variable 'omega' (line 484)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 484, 8), 'omega', init_convolution_kernel_call_result_17434)
        
        # Assigning a Name to a Subscript (line 485):
        
        # Assigning a Name to a Subscript (line 485):
        # Getting the type of 'omega' (line 485)
        omega_17435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 26), 'omega')
        # Getting the type of '_cache' (line 485)
        _cache_17436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 8), '_cache')
        
        # Obtaining an instance of the builtin type 'tuple' (line 485)
        tuple_17437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 485, 16), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 485)
        # Adding element type (line 485)
        # Getting the type of 'n' (line 485)
        n_17438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 16), 'n')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 485, 16), tuple_17437, n_17438)
        # Adding element type (line 485)
        # Getting the type of 'a' (line 485)
        a_17439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 18), 'a')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 485, 16), tuple_17437, a_17439)
        # Adding element type (line 485)
        # Getting the type of 'b' (line 485)
        b_17440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 20), 'b')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 485, 16), tuple_17437, b_17440)
        
        # Storing an element on a container (line 485)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 485, 8), _cache_17436, (tuple_17437, omega_17435))

        if more_types_in_union_17400:
            # SSA join for if statement (line 477)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 486):
    
    # Assigning a Call to a Name (line 486):
    
    # Call to _datacopied(...): (line 486)
    # Processing the call arguments (line 486)
    # Getting the type of 'tmp' (line 486)
    tmp_17442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 30), 'tmp', False)
    # Getting the type of 'x' (line 486)
    x_17443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 35), 'x', False)
    # Processing the call keyword arguments (line 486)
    kwargs_17444 = {}
    # Getting the type of '_datacopied' (line 486)
    _datacopied_17441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 18), '_datacopied', False)
    # Calling _datacopied(args, kwargs) (line 486)
    _datacopied_call_result_17445 = invoke(stypy.reporting.localization.Localization(__file__, 486, 18), _datacopied_17441, *[tmp_17442, x_17443], **kwargs_17444)
    
    # Assigning a type to the variable 'overwrite_x' (line 486)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 486, 4), 'overwrite_x', _datacopied_call_result_17445)
    
    # Call to convolve(...): (line 487)
    # Processing the call arguments (line 487)
    # Getting the type of 'tmp' (line 487)
    tmp_17448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 29), 'tmp', False)
    # Getting the type of 'omega' (line 487)
    omega_17449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 33), 'omega', False)
    # Processing the call keyword arguments (line 487)
    # Getting the type of 'overwrite_x' (line 487)
    overwrite_x_17450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 51), 'overwrite_x', False)
    keyword_17451 = overwrite_x_17450
    kwargs_17452 = {'overwrite_x': keyword_17451}
    # Getting the type of 'convolve' (line 487)
    convolve_17446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 11), 'convolve', False)
    # Obtaining the member 'convolve' of a type (line 487)
    convolve_17447 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 487, 11), convolve_17446, 'convolve')
    # Calling convolve(args, kwargs) (line 487)
    convolve_call_result_17453 = invoke(stypy.reporting.localization.Localization(__file__, 487, 11), convolve_17447, *[tmp_17448, omega_17449], **kwargs_17452)
    
    # Assigning a type to the variable 'stypy_return_type' (line 487)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 487, 4), 'stypy_return_type', convolve_call_result_17453)
    
    # ################# End of 'cc_diff(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'cc_diff' in the type store
    # Getting the type of 'stypy_return_type' (line 439)
    stypy_return_type_17454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_17454)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'cc_diff'
    return stypy_return_type_17454

# Assigning a type to the variable 'cc_diff' (line 439)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 439, 0), 'cc_diff', cc_diff)
# Deleting a member
module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 488, 0), module_type_store, '_cache')

# Assigning a Dict to a Name (line 491):

# Assigning a Dict to a Name (line 491):

# Obtaining an instance of the builtin type 'dict' (line 491)
dict_17455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 491, 9), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 491)

# Assigning a type to the variable '_cache' (line 491)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 491, 0), '_cache', dict_17455)

@norecursion
def shift(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 494)
    None_17456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 23), 'None')
    # Getting the type of '_cache' (line 494)
    _cache_17457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 36), '_cache')
    defaults = [None_17456, _cache_17457]
    # Create a new context for function 'shift'
    module_type_store = module_type_store.open_function_context('shift', 494, 0, False)
    
    # Passed parameters checking function
    shift.stypy_localization = localization
    shift.stypy_type_of_self = None
    shift.stypy_type_store = module_type_store
    shift.stypy_function_name = 'shift'
    shift.stypy_param_names_list = ['x', 'a', 'period', '_cache']
    shift.stypy_varargs_param_name = None
    shift.stypy_kwargs_param_name = None
    shift.stypy_call_defaults = defaults
    shift.stypy_call_varargs = varargs
    shift.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'shift', ['x', 'a', 'period', '_cache'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'shift', localization, ['x', 'a', 'period', '_cache'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'shift(...)' code ##################

    str_17458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 511, (-1)), 'str', '\n    Shift periodic sequence x by a: y(u) = x(u+a).\n\n    If x_j and y_j are Fourier coefficients of periodic functions x\n    and y, respectively, then::\n\n          y_j = exp(j*a*2*pi/period*sqrt(-1)) * x_f\n\n    Parameters\n    ----------\n    x : array_like\n        The array to take the pseudo-derivative from.\n    a : float\n        Defines the parameters of the sinh/sinh pseudo-differential\n    period : float, optional\n        The period of the sequences x and y. Default period is ``2*pi``.\n    ')
    
    # Assigning a Call to a Name (line 512):
    
    # Assigning a Call to a Name (line 512):
    
    # Call to asarray(...): (line 512)
    # Processing the call arguments (line 512)
    # Getting the type of 'x' (line 512)
    x_17460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 18), 'x', False)
    # Processing the call keyword arguments (line 512)
    kwargs_17461 = {}
    # Getting the type of 'asarray' (line 512)
    asarray_17459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 10), 'asarray', False)
    # Calling asarray(args, kwargs) (line 512)
    asarray_call_result_17462 = invoke(stypy.reporting.localization.Localization(__file__, 512, 10), asarray_17459, *[x_17460], **kwargs_17461)
    
    # Assigning a type to the variable 'tmp' (line 512)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 512, 4), 'tmp', asarray_call_result_17462)
    
    
    # Call to iscomplexobj(...): (line 513)
    # Processing the call arguments (line 513)
    # Getting the type of 'tmp' (line 513)
    tmp_17464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 20), 'tmp', False)
    # Processing the call keyword arguments (line 513)
    kwargs_17465 = {}
    # Getting the type of 'iscomplexobj' (line 513)
    iscomplexobj_17463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 7), 'iscomplexobj', False)
    # Calling iscomplexobj(args, kwargs) (line 513)
    iscomplexobj_call_result_17466 = invoke(stypy.reporting.localization.Localization(__file__, 513, 7), iscomplexobj_17463, *[tmp_17464], **kwargs_17465)
    
    # Testing the type of an if condition (line 513)
    if_condition_17467 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 513, 4), iscomplexobj_call_result_17466)
    # Assigning a type to the variable 'if_condition_17467' (line 513)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 513, 4), 'if_condition_17467', if_condition_17467)
    # SSA begins for if statement (line 513)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to shift(...): (line 514)
    # Processing the call arguments (line 514)
    # Getting the type of 'tmp' (line 514)
    tmp_17469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 21), 'tmp', False)
    # Obtaining the member 'real' of a type (line 514)
    real_17470 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 514, 21), tmp_17469, 'real')
    # Getting the type of 'a' (line 514)
    a_17471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 30), 'a', False)
    # Getting the type of 'period' (line 514)
    period_17472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 32), 'period', False)
    # Processing the call keyword arguments (line 514)
    kwargs_17473 = {}
    # Getting the type of 'shift' (line 514)
    shift_17468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 15), 'shift', False)
    # Calling shift(args, kwargs) (line 514)
    shift_call_result_17474 = invoke(stypy.reporting.localization.Localization(__file__, 514, 15), shift_17468, *[real_17470, a_17471, period_17472], **kwargs_17473)
    
    complex_17475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 514, 40), 'complex')
    
    # Call to shift(...): (line 514)
    # Processing the call arguments (line 514)
    # Getting the type of 'tmp' (line 514)
    tmp_17477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 49), 'tmp', False)
    # Obtaining the member 'imag' of a type (line 514)
    imag_17478 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 514, 49), tmp_17477, 'imag')
    # Getting the type of 'a' (line 514)
    a_17479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 58), 'a', False)
    # Getting the type of 'period' (line 514)
    period_17480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 60), 'period', False)
    # Processing the call keyword arguments (line 514)
    kwargs_17481 = {}
    # Getting the type of 'shift' (line 514)
    shift_17476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 43), 'shift', False)
    # Calling shift(args, kwargs) (line 514)
    shift_call_result_17482 = invoke(stypy.reporting.localization.Localization(__file__, 514, 43), shift_17476, *[imag_17478, a_17479, period_17480], **kwargs_17481)
    
    # Applying the binary operator '*' (line 514)
    result_mul_17483 = python_operator(stypy.reporting.localization.Localization(__file__, 514, 40), '*', complex_17475, shift_call_result_17482)
    
    # Applying the binary operator '+' (line 514)
    result_add_17484 = python_operator(stypy.reporting.localization.Localization(__file__, 514, 15), '+', shift_call_result_17474, result_mul_17483)
    
    # Assigning a type to the variable 'stypy_return_type' (line 514)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 514, 8), 'stypy_return_type', result_add_17484)
    # SSA join for if statement (line 513)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 515)
    # Getting the type of 'period' (line 515)
    period_17485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 4), 'period')
    # Getting the type of 'None' (line 515)
    None_17486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 21), 'None')
    
    (may_be_17487, more_types_in_union_17488) = may_not_be_none(period_17485, None_17486)

    if may_be_17487:

        if more_types_in_union_17488:
            # Runtime conditional SSA (line 515)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a BinOp to a Name (line 516):
        
        # Assigning a BinOp to a Name (line 516):
        # Getting the type of 'a' (line 516)
        a_17489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 12), 'a')
        int_17490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 516, 14), 'int')
        # Applying the binary operator '*' (line 516)
        result_mul_17491 = python_operator(stypy.reporting.localization.Localization(__file__, 516, 12), '*', a_17489, int_17490)
        
        # Getting the type of 'pi' (line 516)
        pi_17492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 16), 'pi')
        # Applying the binary operator '*' (line 516)
        result_mul_17493 = python_operator(stypy.reporting.localization.Localization(__file__, 516, 15), '*', result_mul_17491, pi_17492)
        
        # Getting the type of 'period' (line 516)
        period_17494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 19), 'period')
        # Applying the binary operator 'div' (line 516)
        result_div_17495 = python_operator(stypy.reporting.localization.Localization(__file__, 516, 18), 'div', result_mul_17493, period_17494)
        
        # Assigning a type to the variable 'a' (line 516)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 516, 8), 'a', result_div_17495)

        if more_types_in_union_17488:
            # SSA join for if statement (line 515)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 517):
    
    # Assigning a Call to a Name (line 517):
    
    # Call to len(...): (line 517)
    # Processing the call arguments (line 517)
    # Getting the type of 'x' (line 517)
    x_17497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 12), 'x', False)
    # Processing the call keyword arguments (line 517)
    kwargs_17498 = {}
    # Getting the type of 'len' (line 517)
    len_17496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 8), 'len', False)
    # Calling len(args, kwargs) (line 517)
    len_call_result_17499 = invoke(stypy.reporting.localization.Localization(__file__, 517, 8), len_17496, *[x_17497], **kwargs_17498)
    
    # Assigning a type to the variable 'n' (line 517)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 517, 4), 'n', len_call_result_17499)
    
    # Assigning a Call to a Name (line 518):
    
    # Assigning a Call to a Name (line 518):
    
    # Call to get(...): (line 518)
    # Processing the call arguments (line 518)
    
    # Obtaining an instance of the builtin type 'tuple' (line 518)
    tuple_17502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 518, 24), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 518)
    # Adding element type (line 518)
    # Getting the type of 'n' (line 518)
    n_17503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 24), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 518, 24), tuple_17502, n_17503)
    # Adding element type (line 518)
    # Getting the type of 'a' (line 518)
    a_17504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 26), 'a', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 518, 24), tuple_17502, a_17504)
    
    # Processing the call keyword arguments (line 518)
    kwargs_17505 = {}
    # Getting the type of '_cache' (line 518)
    _cache_17500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 12), '_cache', False)
    # Obtaining the member 'get' of a type (line 518)
    get_17501 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 518, 12), _cache_17500, 'get')
    # Calling get(args, kwargs) (line 518)
    get_call_result_17506 = invoke(stypy.reporting.localization.Localization(__file__, 518, 12), get_17501, *[tuple_17502], **kwargs_17505)
    
    # Assigning a type to the variable 'omega' (line 518)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 518, 4), 'omega', get_call_result_17506)
    
    # Type idiom detected: calculating its left and rigth part (line 519)
    # Getting the type of 'omega' (line 519)
    omega_17507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 7), 'omega')
    # Getting the type of 'None' (line 519)
    None_17508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 16), 'None')
    
    (may_be_17509, more_types_in_union_17510) = may_be_none(omega_17507, None_17508)

    if may_be_17509:

        if more_types_in_union_17510:
            # Runtime conditional SSA (line 519)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        
        
        # Call to len(...): (line 520)
        # Processing the call arguments (line 520)
        # Getting the type of '_cache' (line 520)
        _cache_17512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 15), '_cache', False)
        # Processing the call keyword arguments (line 520)
        kwargs_17513 = {}
        # Getting the type of 'len' (line 520)
        len_17511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 11), 'len', False)
        # Calling len(args, kwargs) (line 520)
        len_call_result_17514 = invoke(stypy.reporting.localization.Localization(__file__, 520, 11), len_17511, *[_cache_17512], **kwargs_17513)
        
        int_17515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 520, 25), 'int')
        # Applying the binary operator '>' (line 520)
        result_gt_17516 = python_operator(stypy.reporting.localization.Localization(__file__, 520, 11), '>', len_call_result_17514, int_17515)
        
        # Testing the type of an if condition (line 520)
        if_condition_17517 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 520, 8), result_gt_17516)
        # Assigning a type to the variable 'if_condition_17517' (line 520)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 520, 8), 'if_condition_17517', if_condition_17517)
        # SSA begins for if statement (line 520)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of '_cache' (line 521)
        _cache_17518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 18), '_cache')
        # Testing the type of an if condition (line 521)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 521, 12), _cache_17518)
        # SSA begins for while statement (line 521)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Call to popitem(...): (line 522)
        # Processing the call keyword arguments (line 522)
        kwargs_17521 = {}
        # Getting the type of '_cache' (line 522)
        _cache_17519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 16), '_cache', False)
        # Obtaining the member 'popitem' of a type (line 522)
        popitem_17520 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 522, 16), _cache_17519, 'popitem')
        # Calling popitem(args, kwargs) (line 522)
        popitem_call_result_17522 = invoke(stypy.reporting.localization.Localization(__file__, 522, 16), popitem_17520, *[], **kwargs_17521)
        
        # SSA join for while statement (line 521)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 520)
        module_type_store = module_type_store.join_ssa_context()
        

        @norecursion
        def kernel_real(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            # Getting the type of 'a' (line 524)
            a_17523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 28), 'a')
            defaults = [a_17523]
            # Create a new context for function 'kernel_real'
            module_type_store = module_type_store.open_function_context('kernel_real', 524, 8, False)
            
            # Passed parameters checking function
            kernel_real.stypy_localization = localization
            kernel_real.stypy_type_of_self = None
            kernel_real.stypy_type_store = module_type_store
            kernel_real.stypy_function_name = 'kernel_real'
            kernel_real.stypy_param_names_list = ['k', 'a']
            kernel_real.stypy_varargs_param_name = None
            kernel_real.stypy_kwargs_param_name = None
            kernel_real.stypy_call_defaults = defaults
            kernel_real.stypy_call_varargs = varargs
            kernel_real.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'kernel_real', ['k', 'a'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'kernel_real', localization, ['k', 'a'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'kernel_real(...)' code ##################

            
            # Call to cos(...): (line 525)
            # Processing the call arguments (line 525)
            # Getting the type of 'a' (line 525)
            a_17525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 23), 'a', False)
            # Getting the type of 'k' (line 525)
            k_17526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 25), 'k', False)
            # Applying the binary operator '*' (line 525)
            result_mul_17527 = python_operator(stypy.reporting.localization.Localization(__file__, 525, 23), '*', a_17525, k_17526)
            
            # Processing the call keyword arguments (line 525)
            kwargs_17528 = {}
            # Getting the type of 'cos' (line 525)
            cos_17524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 19), 'cos', False)
            # Calling cos(args, kwargs) (line 525)
            cos_call_result_17529 = invoke(stypy.reporting.localization.Localization(__file__, 525, 19), cos_17524, *[result_mul_17527], **kwargs_17528)
            
            # Assigning a type to the variable 'stypy_return_type' (line 525)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 525, 12), 'stypy_return_type', cos_call_result_17529)
            
            # ################# End of 'kernel_real(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'kernel_real' in the type store
            # Getting the type of 'stypy_return_type' (line 524)
            stypy_return_type_17530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_17530)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'kernel_real'
            return stypy_return_type_17530

        # Assigning a type to the variable 'kernel_real' (line 524)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 524, 8), 'kernel_real', kernel_real)

        @norecursion
        def kernel_imag(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            # Getting the type of 'a' (line 527)
            a_17531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 28), 'a')
            defaults = [a_17531]
            # Create a new context for function 'kernel_imag'
            module_type_store = module_type_store.open_function_context('kernel_imag', 527, 8, False)
            
            # Passed parameters checking function
            kernel_imag.stypy_localization = localization
            kernel_imag.stypy_type_of_self = None
            kernel_imag.stypy_type_store = module_type_store
            kernel_imag.stypy_function_name = 'kernel_imag'
            kernel_imag.stypy_param_names_list = ['k', 'a']
            kernel_imag.stypy_varargs_param_name = None
            kernel_imag.stypy_kwargs_param_name = None
            kernel_imag.stypy_call_defaults = defaults
            kernel_imag.stypy_call_varargs = varargs
            kernel_imag.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'kernel_imag', ['k', 'a'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'kernel_imag', localization, ['k', 'a'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'kernel_imag(...)' code ##################

            
            # Call to sin(...): (line 528)
            # Processing the call arguments (line 528)
            # Getting the type of 'a' (line 528)
            a_17533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 23), 'a', False)
            # Getting the type of 'k' (line 528)
            k_17534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 25), 'k', False)
            # Applying the binary operator '*' (line 528)
            result_mul_17535 = python_operator(stypy.reporting.localization.Localization(__file__, 528, 23), '*', a_17533, k_17534)
            
            # Processing the call keyword arguments (line 528)
            kwargs_17536 = {}
            # Getting the type of 'sin' (line 528)
            sin_17532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 19), 'sin', False)
            # Calling sin(args, kwargs) (line 528)
            sin_call_result_17537 = invoke(stypy.reporting.localization.Localization(__file__, 528, 19), sin_17532, *[result_mul_17535], **kwargs_17536)
            
            # Assigning a type to the variable 'stypy_return_type' (line 528)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 528, 12), 'stypy_return_type', sin_call_result_17537)
            
            # ################# End of 'kernel_imag(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'kernel_imag' in the type store
            # Getting the type of 'stypy_return_type' (line 527)
            stypy_return_type_17538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_17538)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'kernel_imag'
            return stypy_return_type_17538

        # Assigning a type to the variable 'kernel_imag' (line 527)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 527, 8), 'kernel_imag', kernel_imag)
        
        # Assigning a Call to a Name (line 529):
        
        # Assigning a Call to a Name (line 529):
        
        # Call to init_convolution_kernel(...): (line 529)
        # Processing the call arguments (line 529)
        # Getting the type of 'n' (line 529)
        n_17541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 54), 'n', False)
        # Getting the type of 'kernel_real' (line 529)
        kernel_real_17542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 56), 'kernel_real', False)
        # Processing the call keyword arguments (line 529)
        int_17543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 529, 70), 'int')
        keyword_17544 = int_17543
        int_17545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 530, 67), 'int')
        keyword_17546 = int_17545
        kwargs_17547 = {'zero_nyquist': keyword_17546, 'd': keyword_17544}
        # Getting the type of 'convolve' (line 529)
        convolve_17539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 21), 'convolve', False)
        # Obtaining the member 'init_convolution_kernel' of a type (line 529)
        init_convolution_kernel_17540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 529, 21), convolve_17539, 'init_convolution_kernel')
        # Calling init_convolution_kernel(args, kwargs) (line 529)
        init_convolution_kernel_call_result_17548 = invoke(stypy.reporting.localization.Localization(__file__, 529, 21), init_convolution_kernel_17540, *[n_17541, kernel_real_17542], **kwargs_17547)
        
        # Assigning a type to the variable 'omega_real' (line 529)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 529, 8), 'omega_real', init_convolution_kernel_call_result_17548)
        
        # Assigning a Call to a Name (line 531):
        
        # Assigning a Call to a Name (line 531):
        
        # Call to init_convolution_kernel(...): (line 531)
        # Processing the call arguments (line 531)
        # Getting the type of 'n' (line 531)
        n_17551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 54), 'n', False)
        # Getting the type of 'kernel_imag' (line 531)
        kernel_imag_17552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 56), 'kernel_imag', False)
        # Processing the call keyword arguments (line 531)
        int_17553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 531, 70), 'int')
        keyword_17554 = int_17553
        int_17555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 532, 67), 'int')
        keyword_17556 = int_17555
        kwargs_17557 = {'zero_nyquist': keyword_17556, 'd': keyword_17554}
        # Getting the type of 'convolve' (line 531)
        convolve_17549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 21), 'convolve', False)
        # Obtaining the member 'init_convolution_kernel' of a type (line 531)
        init_convolution_kernel_17550 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 531, 21), convolve_17549, 'init_convolution_kernel')
        # Calling init_convolution_kernel(args, kwargs) (line 531)
        init_convolution_kernel_call_result_17558 = invoke(stypy.reporting.localization.Localization(__file__, 531, 21), init_convolution_kernel_17550, *[n_17551, kernel_imag_17552], **kwargs_17557)
        
        # Assigning a type to the variable 'omega_imag' (line 531)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 531, 8), 'omega_imag', init_convolution_kernel_call_result_17558)
        
        # Assigning a Tuple to a Subscript (line 533):
        
        # Assigning a Tuple to a Subscript (line 533):
        
        # Obtaining an instance of the builtin type 'tuple' (line 533)
        tuple_17559 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 533, 24), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 533)
        # Adding element type (line 533)
        # Getting the type of 'omega_real' (line 533)
        omega_real_17560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 24), 'omega_real')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 533, 24), tuple_17559, omega_real_17560)
        # Adding element type (line 533)
        # Getting the type of 'omega_imag' (line 533)
        omega_imag_17561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 35), 'omega_imag')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 533, 24), tuple_17559, omega_imag_17561)
        
        # Getting the type of '_cache' (line 533)
        _cache_17562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 8), '_cache')
        
        # Obtaining an instance of the builtin type 'tuple' (line 533)
        tuple_17563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 533, 16), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 533)
        # Adding element type (line 533)
        # Getting the type of 'n' (line 533)
        n_17564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 16), 'n')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 533, 16), tuple_17563, n_17564)
        # Adding element type (line 533)
        # Getting the type of 'a' (line 533)
        a_17565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 18), 'a')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 533, 16), tuple_17563, a_17565)
        
        # Storing an element on a container (line 533)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 533, 8), _cache_17562, (tuple_17563, tuple_17559))

        if more_types_in_union_17510:
            # Runtime conditional SSA for else branch (line 519)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_17509) or more_types_in_union_17510):
        
        # Assigning a Name to a Tuple (line 535):
        
        # Assigning a Subscript to a Name (line 535):
        
        # Obtaining the type of the subscript
        int_17566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 535, 8), 'int')
        # Getting the type of 'omega' (line 535)
        omega_17567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 32), 'omega')
        # Obtaining the member '__getitem__' of a type (line 535)
        getitem___17568 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 535, 8), omega_17567, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 535)
        subscript_call_result_17569 = invoke(stypy.reporting.localization.Localization(__file__, 535, 8), getitem___17568, int_17566)
        
        # Assigning a type to the variable 'tuple_var_assignment_16490' (line 535)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 535, 8), 'tuple_var_assignment_16490', subscript_call_result_17569)
        
        # Assigning a Subscript to a Name (line 535):
        
        # Obtaining the type of the subscript
        int_17570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 535, 8), 'int')
        # Getting the type of 'omega' (line 535)
        omega_17571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 32), 'omega')
        # Obtaining the member '__getitem__' of a type (line 535)
        getitem___17572 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 535, 8), omega_17571, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 535)
        subscript_call_result_17573 = invoke(stypy.reporting.localization.Localization(__file__, 535, 8), getitem___17572, int_17570)
        
        # Assigning a type to the variable 'tuple_var_assignment_16491' (line 535)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 535, 8), 'tuple_var_assignment_16491', subscript_call_result_17573)
        
        # Assigning a Name to a Name (line 535):
        # Getting the type of 'tuple_var_assignment_16490' (line 535)
        tuple_var_assignment_16490_17574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 8), 'tuple_var_assignment_16490')
        # Assigning a type to the variable 'omega_real' (line 535)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 535, 8), 'omega_real', tuple_var_assignment_16490_17574)
        
        # Assigning a Name to a Name (line 535):
        # Getting the type of 'tuple_var_assignment_16491' (line 535)
        tuple_var_assignment_16491_17575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 8), 'tuple_var_assignment_16491')
        # Assigning a type to the variable 'omega_imag' (line 535)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 535, 19), 'omega_imag', tuple_var_assignment_16491_17575)

        if (may_be_17509 and more_types_in_union_17510):
            # SSA join for if statement (line 519)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 536):
    
    # Assigning a Call to a Name (line 536):
    
    # Call to _datacopied(...): (line 536)
    # Processing the call arguments (line 536)
    # Getting the type of 'tmp' (line 536)
    tmp_17577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 30), 'tmp', False)
    # Getting the type of 'x' (line 536)
    x_17578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 35), 'x', False)
    # Processing the call keyword arguments (line 536)
    kwargs_17579 = {}
    # Getting the type of '_datacopied' (line 536)
    _datacopied_17576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 18), '_datacopied', False)
    # Calling _datacopied(args, kwargs) (line 536)
    _datacopied_call_result_17580 = invoke(stypy.reporting.localization.Localization(__file__, 536, 18), _datacopied_17576, *[tmp_17577, x_17578], **kwargs_17579)
    
    # Assigning a type to the variable 'overwrite_x' (line 536)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 536, 4), 'overwrite_x', _datacopied_call_result_17580)
    
    # Call to convolve_z(...): (line 537)
    # Processing the call arguments (line 537)
    # Getting the type of 'tmp' (line 537)
    tmp_17583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 31), 'tmp', False)
    # Getting the type of 'omega_real' (line 537)
    omega_real_17584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 35), 'omega_real', False)
    # Getting the type of 'omega_imag' (line 537)
    omega_imag_17585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 46), 'omega_imag', False)
    # Processing the call keyword arguments (line 537)
    # Getting the type of 'overwrite_x' (line 538)
    overwrite_x_17586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 43), 'overwrite_x', False)
    keyword_17587 = overwrite_x_17586
    kwargs_17588 = {'overwrite_x': keyword_17587}
    # Getting the type of 'convolve' (line 537)
    convolve_17581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 11), 'convolve', False)
    # Obtaining the member 'convolve_z' of a type (line 537)
    convolve_z_17582 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 537, 11), convolve_17581, 'convolve_z')
    # Calling convolve_z(args, kwargs) (line 537)
    convolve_z_call_result_17589 = invoke(stypy.reporting.localization.Localization(__file__, 537, 11), convolve_z_17582, *[tmp_17583, omega_real_17584, omega_imag_17585], **kwargs_17588)
    
    # Assigning a type to the variable 'stypy_return_type' (line 537)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 537, 4), 'stypy_return_type', convolve_z_call_result_17589)
    
    # ################# End of 'shift(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'shift' in the type store
    # Getting the type of 'stypy_return_type' (line 494)
    stypy_return_type_17590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_17590)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'shift'
    return stypy_return_type_17590

# Assigning a type to the variable 'shift' (line 494)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 494, 0), 'shift', shift)
# Deleting a member
module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 540, 0), module_type_store, '_cache')

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
