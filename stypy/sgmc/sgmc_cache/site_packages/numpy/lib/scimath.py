
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Wrapper functions to more user-friendly calling of certain math functions
3: whose output data-type is different than the input data-type in certain
4: domains of the input.
5: 
6: For example, for functions like `log` with branch cuts, the versions in this
7: module provide the mathematically valid answers in the complex plane::
8: 
9:   >>> import math
10:   >>> from numpy.lib import scimath
11:   >>> scimath.log(-math.exp(1)) == (1+1j*math.pi)
12:   True
13: 
14: Similarly, `sqrt`, other base logarithms, `power` and trig functions are
15: correctly handled.  See their respective docstrings for specific examples.
16: 
17: '''
18: from __future__ import division, absolute_import, print_function
19: 
20: import numpy.core.numeric as nx
21: import numpy.core.numerictypes as nt
22: from numpy.core.numeric import asarray, any
23: from numpy.lib.type_check import isreal
24: 
25: 
26: __all__ = [
27:     'sqrt', 'log', 'log2', 'logn', 'log10', 'power', 'arccos', 'arcsin',
28:     'arctanh'
29:     ]
30: 
31: 
32: _ln2 = nx.log(2.0)
33: 
34: 
35: def _tocomplex(arr):
36:     '''Convert its input `arr` to a complex array.
37: 
38:     The input is returned as a complex array of the smallest type that will fit
39:     the original data: types like single, byte, short, etc. become csingle,
40:     while others become cdouble.
41: 
42:     A copy of the input is always made.
43: 
44:     Parameters
45:     ----------
46:     arr : array
47: 
48:     Returns
49:     -------
50:     array
51:         An array with the same input data as the input but in complex form.
52: 
53:     Examples
54:     --------
55: 
56:     First, consider an input of type short:
57: 
58:     >>> a = np.array([1,2,3],np.short)
59: 
60:     >>> ac = np.lib.scimath._tocomplex(a); ac
61:     array([ 1.+0.j,  2.+0.j,  3.+0.j], dtype=complex64)
62: 
63:     >>> ac.dtype
64:     dtype('complex64')
65: 
66:     If the input is of type double, the output is correspondingly of the
67:     complex double type as well:
68: 
69:     >>> b = np.array([1,2,3],np.double)
70: 
71:     >>> bc = np.lib.scimath._tocomplex(b); bc
72:     array([ 1.+0.j,  2.+0.j,  3.+0.j])
73: 
74:     >>> bc.dtype
75:     dtype('complex128')
76: 
77:     Note that even if the input was complex to begin with, a copy is still
78:     made, since the astype() method always copies:
79: 
80:     >>> c = np.array([1,2,3],np.csingle)
81: 
82:     >>> cc = np.lib.scimath._tocomplex(c); cc
83:     array([ 1.+0.j,  2.+0.j,  3.+0.j], dtype=complex64)
84: 
85:     >>> c *= 2; c
86:     array([ 2.+0.j,  4.+0.j,  6.+0.j], dtype=complex64)
87: 
88:     >>> cc
89:     array([ 1.+0.j,  2.+0.j,  3.+0.j], dtype=complex64)
90:     '''
91:     if issubclass(arr.dtype.type, (nt.single, nt.byte, nt.short, nt.ubyte,
92:                                    nt.ushort, nt.csingle)):
93:         return arr.astype(nt.csingle)
94:     else:
95:         return arr.astype(nt.cdouble)
96: 
97: def _fix_real_lt_zero(x):
98:     '''Convert `x` to complex if it has real, negative components.
99: 
100:     Otherwise, output is just the array version of the input (via asarray).
101: 
102:     Parameters
103:     ----------
104:     x : array_like
105: 
106:     Returns
107:     -------
108:     array
109: 
110:     Examples
111:     --------
112:     >>> np.lib.scimath._fix_real_lt_zero([1,2])
113:     array([1, 2])
114: 
115:     >>> np.lib.scimath._fix_real_lt_zero([-1,2])
116:     array([-1.+0.j,  2.+0.j])
117: 
118:     '''
119:     x = asarray(x)
120:     if any(isreal(x) & (x < 0)):
121:         x = _tocomplex(x)
122:     return x
123: 
124: def _fix_int_lt_zero(x):
125:     '''Convert `x` to double if it has real, negative components.
126: 
127:     Otherwise, output is just the array version of the input (via asarray).
128: 
129:     Parameters
130:     ----------
131:     x : array_like
132: 
133:     Returns
134:     -------
135:     array
136: 
137:     Examples
138:     --------
139:     >>> np.lib.scimath._fix_int_lt_zero([1,2])
140:     array([1, 2])
141: 
142:     >>> np.lib.scimath._fix_int_lt_zero([-1,2])
143:     array([-1.,  2.])
144:     '''
145:     x = asarray(x)
146:     if any(isreal(x) & (x < 0)):
147:         x = x * 1.0
148:     return x
149: 
150: def _fix_real_abs_gt_1(x):
151:     '''Convert `x` to complex if it has real components x_i with abs(x_i)>1.
152: 
153:     Otherwise, output is just the array version of the input (via asarray).
154: 
155:     Parameters
156:     ----------
157:     x : array_like
158: 
159:     Returns
160:     -------
161:     array
162: 
163:     Examples
164:     --------
165:     >>> np.lib.scimath._fix_real_abs_gt_1([0,1])
166:     array([0, 1])
167: 
168:     >>> np.lib.scimath._fix_real_abs_gt_1([0,2])
169:     array([ 0.+0.j,  2.+0.j])
170:     '''
171:     x = asarray(x)
172:     if any(isreal(x) & (abs(x) > 1)):
173:         x = _tocomplex(x)
174:     return x
175: 
176: def sqrt(x):
177:     '''
178:     Compute the square root of x.
179: 
180:     For negative input elements, a complex value is returned
181:     (unlike `numpy.sqrt` which returns NaN).
182: 
183:     Parameters
184:     ----------
185:     x : array_like
186:        The input value(s).
187: 
188:     Returns
189:     -------
190:     out : ndarray or scalar
191:        The square root of `x`. If `x` was a scalar, so is `out`,
192:        otherwise an array is returned.
193: 
194:     See Also
195:     --------
196:     numpy.sqrt
197: 
198:     Examples
199:     --------
200:     For real, non-negative inputs this works just like `numpy.sqrt`:
201: 
202:     >>> np.lib.scimath.sqrt(1)
203:     1.0
204:     >>> np.lib.scimath.sqrt([1, 4])
205:     array([ 1.,  2.])
206: 
207:     But it automatically handles negative inputs:
208: 
209:     >>> np.lib.scimath.sqrt(-1)
210:     (0.0+1.0j)
211:     >>> np.lib.scimath.sqrt([-1,4])
212:     array([ 0.+1.j,  2.+0.j])
213: 
214:     '''
215:     x = _fix_real_lt_zero(x)
216:     return nx.sqrt(x)
217: 
218: def log(x):
219:     '''
220:     Compute the natural logarithm of `x`.
221: 
222:     Return the "principal value" (for a description of this, see `numpy.log`)
223:     of :math:`log_e(x)`. For real `x > 0`, this is a real number (``log(0)``
224:     returns ``-inf`` and ``log(np.inf)`` returns ``inf``). Otherwise, the
225:     complex principle value is returned.
226: 
227:     Parameters
228:     ----------
229:     x : array_like
230:        The value(s) whose log is (are) required.
231: 
232:     Returns
233:     -------
234:     out : ndarray or scalar
235:        The log of the `x` value(s). If `x` was a scalar, so is `out`,
236:        otherwise an array is returned.
237: 
238:     See Also
239:     --------
240:     numpy.log
241: 
242:     Notes
243:     -----
244:     For a log() that returns ``NAN`` when real `x < 0`, use `numpy.log`
245:     (note, however, that otherwise `numpy.log` and this `log` are identical,
246:     i.e., both return ``-inf`` for `x = 0`, ``inf`` for `x = inf`, and,
247:     notably, the complex principle value if ``x.imag != 0``).
248: 
249:     Examples
250:     --------
251:     >>> np.emath.log(np.exp(1))
252:     1.0
253: 
254:     Negative arguments are handled "correctly" (recall that
255:     ``exp(log(x)) == x`` does *not* hold for real ``x < 0``):
256: 
257:     >>> np.emath.log(-np.exp(1)) == (1 + np.pi * 1j)
258:     True
259: 
260:     '''
261:     x = _fix_real_lt_zero(x)
262:     return nx.log(x)
263: 
264: def log10(x):
265:     '''
266:     Compute the logarithm base 10 of `x`.
267: 
268:     Return the "principal value" (for a description of this, see
269:     `numpy.log10`) of :math:`log_{10}(x)`. For real `x > 0`, this
270:     is a real number (``log10(0)`` returns ``-inf`` and ``log10(np.inf)``
271:     returns ``inf``). Otherwise, the complex principle value is returned.
272: 
273:     Parameters
274:     ----------
275:     x : array_like or scalar
276:        The value(s) whose log base 10 is (are) required.
277: 
278:     Returns
279:     -------
280:     out : ndarray or scalar
281:        The log base 10 of the `x` value(s). If `x` was a scalar, so is `out`,
282:        otherwise an array object is returned.
283: 
284:     See Also
285:     --------
286:     numpy.log10
287: 
288:     Notes
289:     -----
290:     For a log10() that returns ``NAN`` when real `x < 0`, use `numpy.log10`
291:     (note, however, that otherwise `numpy.log10` and this `log10` are
292:     identical, i.e., both return ``-inf`` for `x = 0`, ``inf`` for `x = inf`,
293:     and, notably, the complex principle value if ``x.imag != 0``).
294: 
295:     Examples
296:     --------
297: 
298:     (We set the printing precision so the example can be auto-tested)
299: 
300:     >>> np.set_printoptions(precision=4)
301: 
302:     >>> np.emath.log10(10**1)
303:     1.0
304: 
305:     >>> np.emath.log10([-10**1, -10**2, 10**2])
306:     array([ 1.+1.3644j,  2.+1.3644j,  2.+0.j    ])
307: 
308:     '''
309:     x = _fix_real_lt_zero(x)
310:     return nx.log10(x)
311: 
312: def logn(n, x):
313:     '''
314:     Take log base n of x.
315: 
316:     If `x` contains negative inputs, the answer is computed and returned in the
317:     complex domain.
318: 
319:     Parameters
320:     ----------
321:     n : int
322:        The base in which the log is taken.
323:     x : array_like
324:        The value(s) whose log base `n` is (are) required.
325: 
326:     Returns
327:     -------
328:     out : ndarray or scalar
329:        The log base `n` of the `x` value(s). If `x` was a scalar, so is
330:        `out`, otherwise an array is returned.
331: 
332:     Examples
333:     --------
334:     >>> np.set_printoptions(precision=4)
335: 
336:     >>> np.lib.scimath.logn(2, [4, 8])
337:     array([ 2.,  3.])
338:     >>> np.lib.scimath.logn(2, [-4, -8, 8])
339:     array([ 2.+4.5324j,  3.+4.5324j,  3.+0.j    ])
340: 
341:     '''
342:     x = _fix_real_lt_zero(x)
343:     n = _fix_real_lt_zero(n)
344:     return nx.log(x)/nx.log(n)
345: 
346: def log2(x):
347:     '''
348:     Compute the logarithm base 2 of `x`.
349: 
350:     Return the "principal value" (for a description of this, see
351:     `numpy.log2`) of :math:`log_2(x)`. For real `x > 0`, this is
352:     a real number (``log2(0)`` returns ``-inf`` and ``log2(np.inf)`` returns
353:     ``inf``). Otherwise, the complex principle value is returned.
354: 
355:     Parameters
356:     ----------
357:     x : array_like
358:        The value(s) whose log base 2 is (are) required.
359: 
360:     Returns
361:     -------
362:     out : ndarray or scalar
363:        The log base 2 of the `x` value(s). If `x` was a scalar, so is `out`,
364:        otherwise an array is returned.
365: 
366:     See Also
367:     --------
368:     numpy.log2
369: 
370:     Notes
371:     -----
372:     For a log2() that returns ``NAN`` when real `x < 0`, use `numpy.log2`
373:     (note, however, that otherwise `numpy.log2` and this `log2` are
374:     identical, i.e., both return ``-inf`` for `x = 0`, ``inf`` for `x = inf`,
375:     and, notably, the complex principle value if ``x.imag != 0``).
376: 
377:     Examples
378:     --------
379:     We set the printing precision so the example can be auto-tested:
380: 
381:     >>> np.set_printoptions(precision=4)
382: 
383:     >>> np.emath.log2(8)
384:     3.0
385:     >>> np.emath.log2([-4, -8, 8])
386:     array([ 2.+4.5324j,  3.+4.5324j,  3.+0.j    ])
387: 
388:     '''
389:     x = _fix_real_lt_zero(x)
390:     return nx.log2(x)
391: 
392: def power(x, p):
393:     '''
394:     Return x to the power p, (x**p).
395: 
396:     If `x` contains negative values, the output is converted to the
397:     complex domain.
398: 
399:     Parameters
400:     ----------
401:     x : array_like
402:         The input value(s).
403:     p : array_like of ints
404:         The power(s) to which `x` is raised. If `x` contains multiple values,
405:         `p` has to either be a scalar, or contain the same number of values
406:         as `x`. In the latter case, the result is
407:         ``x[0]**p[0], x[1]**p[1], ...``.
408: 
409:     Returns
410:     -------
411:     out : ndarray or scalar
412:         The result of ``x**p``. If `x` and `p` are scalars, so is `out`,
413:         otherwise an array is returned.
414: 
415:     See Also
416:     --------
417:     numpy.power
418: 
419:     Examples
420:     --------
421:     >>> np.set_printoptions(precision=4)
422: 
423:     >>> np.lib.scimath.power([2, 4], 2)
424:     array([ 4, 16])
425:     >>> np.lib.scimath.power([2, 4], -2)
426:     array([ 0.25  ,  0.0625])
427:     >>> np.lib.scimath.power([-2, 4], 2)
428:     array([  4.+0.j,  16.+0.j])
429: 
430:     '''
431:     x = _fix_real_lt_zero(x)
432:     p = _fix_int_lt_zero(p)
433:     return nx.power(x, p)
434: 
435: def arccos(x):
436:     '''
437:     Compute the inverse cosine of x.
438: 
439:     Return the "principal value" (for a description of this, see
440:     `numpy.arccos`) of the inverse cosine of `x`. For real `x` such that
441:     `abs(x) <= 1`, this is a real number in the closed interval
442:     :math:`[0, \\pi]`.  Otherwise, the complex principle value is returned.
443: 
444:     Parameters
445:     ----------
446:     x : array_like or scalar
447:        The value(s) whose arccos is (are) required.
448: 
449:     Returns
450:     -------
451:     out : ndarray or scalar
452:        The inverse cosine(s) of the `x` value(s). If `x` was a scalar, so
453:        is `out`, otherwise an array object is returned.
454: 
455:     See Also
456:     --------
457:     numpy.arccos
458: 
459:     Notes
460:     -----
461:     For an arccos() that returns ``NAN`` when real `x` is not in the
462:     interval ``[-1,1]``, use `numpy.arccos`.
463: 
464:     Examples
465:     --------
466:     >>> np.set_printoptions(precision=4)
467: 
468:     >>> np.emath.arccos(1) # a scalar is returned
469:     0.0
470: 
471:     >>> np.emath.arccos([1,2])
472:     array([ 0.-0.j   ,  0.+1.317j])
473: 
474:     '''
475:     x = _fix_real_abs_gt_1(x)
476:     return nx.arccos(x)
477: 
478: def arcsin(x):
479:     '''
480:     Compute the inverse sine of x.
481: 
482:     Return the "principal value" (for a description of this, see
483:     `numpy.arcsin`) of the inverse sine of `x`. For real `x` such that
484:     `abs(x) <= 1`, this is a real number in the closed interval
485:     :math:`[-\\pi/2, \\pi/2]`.  Otherwise, the complex principle value is
486:     returned.
487: 
488:     Parameters
489:     ----------
490:     x : array_like or scalar
491:        The value(s) whose arcsin is (are) required.
492: 
493:     Returns
494:     -------
495:     out : ndarray or scalar
496:        The inverse sine(s) of the `x` value(s). If `x` was a scalar, so
497:        is `out`, otherwise an array object is returned.
498: 
499:     See Also
500:     --------
501:     numpy.arcsin
502: 
503:     Notes
504:     -----
505:     For an arcsin() that returns ``NAN`` when real `x` is not in the
506:     interval ``[-1,1]``, use `numpy.arcsin`.
507: 
508:     Examples
509:     --------
510:     >>> np.set_printoptions(precision=4)
511: 
512:     >>> np.emath.arcsin(0)
513:     0.0
514: 
515:     >>> np.emath.arcsin([0,1])
516:     array([ 0.    ,  1.5708])
517: 
518:     '''
519:     x = _fix_real_abs_gt_1(x)
520:     return nx.arcsin(x)
521: 
522: def arctanh(x):
523:     '''
524:     Compute the inverse hyperbolic tangent of `x`.
525: 
526:     Return the "principal value" (for a description of this, see
527:     `numpy.arctanh`) of `arctanh(x)`. For real `x` such that
528:     `abs(x) < 1`, this is a real number.  If `abs(x) > 1`, or if `x` is
529:     complex, the result is complex. Finally, `x = 1` returns``inf`` and
530:     `x=-1` returns ``-inf``.
531: 
532:     Parameters
533:     ----------
534:     x : array_like
535:        The value(s) whose arctanh is (are) required.
536: 
537:     Returns
538:     -------
539:     out : ndarray or scalar
540:        The inverse hyperbolic tangent(s) of the `x` value(s). If `x` was
541:        a scalar so is `out`, otherwise an array is returned.
542: 
543: 
544:     See Also
545:     --------
546:     numpy.arctanh
547: 
548:     Notes
549:     -----
550:     For an arctanh() that returns ``NAN`` when real `x` is not in the
551:     interval ``(-1,1)``, use `numpy.arctanh` (this latter, however, does
552:     return +/-inf for `x = +/-1`).
553: 
554:     Examples
555:     --------
556:     >>> np.set_printoptions(precision=4)
557: 
558:     >>> np.emath.arctanh(np.matrix(np.eye(2)))
559:     array([[ Inf,   0.],
560:            [  0.,  Inf]])
561:     >>> np.emath.arctanh([1j])
562:     array([ 0.+0.7854j])
563: 
564:     '''
565:     x = _fix_real_abs_gt_1(x)
566:     return nx.arctanh(x)
567: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_124916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, (-1)), 'str', '\nWrapper functions to more user-friendly calling of certain math functions\nwhose output data-type is different than the input data-type in certain\ndomains of the input.\n\nFor example, for functions like `log` with branch cuts, the versions in this\nmodule provide the mathematically valid answers in the complex plane::\n\n  >>> import math\n  >>> from numpy.lib import scimath\n  >>> scimath.log(-math.exp(1)) == (1+1j*math.pi)\n  True\n\nSimilarly, `sqrt`, other base logarithms, `power` and trig functions are\ncorrectly handled.  See their respective docstrings for specific examples.\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 20, 0))

# 'import numpy.core.numeric' statement (line 20)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/lib/')
import_124917 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'numpy.core.numeric')

if (type(import_124917) is not StypyTypeError):

    if (import_124917 != 'pyd_module'):
        __import__(import_124917)
        sys_modules_124918 = sys.modules[import_124917]
        import_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'nx', sys_modules_124918.module_type_store, module_type_store)
    else:
        import numpy.core.numeric as nx

        import_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'nx', numpy.core.numeric, module_type_store)

else:
    # Assigning a type to the variable 'numpy.core.numeric' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'numpy.core.numeric', import_124917)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/lib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 21, 0))

# 'import numpy.core.numerictypes' statement (line 21)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/lib/')
import_124919 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'numpy.core.numerictypes')

if (type(import_124919) is not StypyTypeError):

    if (import_124919 != 'pyd_module'):
        __import__(import_124919)
        sys_modules_124920 = sys.modules[import_124919]
        import_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'nt', sys_modules_124920.module_type_store, module_type_store)
    else:
        import numpy.core.numerictypes as nt

        import_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'nt', numpy.core.numerictypes, module_type_store)

else:
    # Assigning a type to the variable 'numpy.core.numerictypes' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'numpy.core.numerictypes', import_124919)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/lib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 22, 0))

# 'from numpy.core.numeric import asarray, any' statement (line 22)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/lib/')
import_124921 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'numpy.core.numeric')

if (type(import_124921) is not StypyTypeError):

    if (import_124921 != 'pyd_module'):
        __import__(import_124921)
        sys_modules_124922 = sys.modules[import_124921]
        import_from_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'numpy.core.numeric', sys_modules_124922.module_type_store, module_type_store, ['asarray', 'any'])
        nest_module(stypy.reporting.localization.Localization(__file__, 22, 0), __file__, sys_modules_124922, sys_modules_124922.module_type_store, module_type_store)
    else:
        from numpy.core.numeric import asarray, any

        import_from_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'numpy.core.numeric', None, module_type_store, ['asarray', 'any'], [asarray, any])

else:
    # Assigning a type to the variable 'numpy.core.numeric' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'numpy.core.numeric', import_124921)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/lib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 23, 0))

# 'from numpy.lib.type_check import isreal' statement (line 23)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/lib/')
import_124923 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 23, 0), 'numpy.lib.type_check')

if (type(import_124923) is not StypyTypeError):

    if (import_124923 != 'pyd_module'):
        __import__(import_124923)
        sys_modules_124924 = sys.modules[import_124923]
        import_from_module(stypy.reporting.localization.Localization(__file__, 23, 0), 'numpy.lib.type_check', sys_modules_124924.module_type_store, module_type_store, ['isreal'])
        nest_module(stypy.reporting.localization.Localization(__file__, 23, 0), __file__, sys_modules_124924, sys_modules_124924.module_type_store, module_type_store)
    else:
        from numpy.lib.type_check import isreal

        import_from_module(stypy.reporting.localization.Localization(__file__, 23, 0), 'numpy.lib.type_check', None, module_type_store, ['isreal'], [isreal])

else:
    # Assigning a type to the variable 'numpy.lib.type_check' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'numpy.lib.type_check', import_124923)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/lib/')


# Assigning a List to a Name (line 26):
__all__ = ['sqrt', 'log', 'log2', 'logn', 'log10', 'power', 'arccos', 'arcsin', 'arctanh']
module_type_store.set_exportable_members(['sqrt', 'log', 'log2', 'logn', 'log10', 'power', 'arccos', 'arcsin', 'arctanh'])

# Obtaining an instance of the builtin type 'list' (line 26)
list_124925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 26)
# Adding element type (line 26)
str_124926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 4), 'str', 'sqrt')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 10), list_124925, str_124926)
# Adding element type (line 26)
str_124927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 12), 'str', 'log')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 10), list_124925, str_124927)
# Adding element type (line 26)
str_124928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 19), 'str', 'log2')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 10), list_124925, str_124928)
# Adding element type (line 26)
str_124929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 27), 'str', 'logn')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 10), list_124925, str_124929)
# Adding element type (line 26)
str_124930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 35), 'str', 'log10')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 10), list_124925, str_124930)
# Adding element type (line 26)
str_124931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 44), 'str', 'power')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 10), list_124925, str_124931)
# Adding element type (line 26)
str_124932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 53), 'str', 'arccos')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 10), list_124925, str_124932)
# Adding element type (line 26)
str_124933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 63), 'str', 'arcsin')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 10), list_124925, str_124933)
# Adding element type (line 26)
str_124934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 4), 'str', 'arctanh')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 10), list_124925, str_124934)

# Assigning a type to the variable '__all__' (line 26)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), '__all__', list_124925)

# Assigning a Call to a Name (line 32):

# Call to log(...): (line 32)
# Processing the call arguments (line 32)
float_124937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 14), 'float')
# Processing the call keyword arguments (line 32)
kwargs_124938 = {}
# Getting the type of 'nx' (line 32)
nx_124935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 7), 'nx', False)
# Obtaining the member 'log' of a type (line 32)
log_124936 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 7), nx_124935, 'log')
# Calling log(args, kwargs) (line 32)
log_call_result_124939 = invoke(stypy.reporting.localization.Localization(__file__, 32, 7), log_124936, *[float_124937], **kwargs_124938)

# Assigning a type to the variable '_ln2' (line 32)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 0), '_ln2', log_call_result_124939)

@norecursion
def _tocomplex(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_tocomplex'
    module_type_store = module_type_store.open_function_context('_tocomplex', 35, 0, False)
    
    # Passed parameters checking function
    _tocomplex.stypy_localization = localization
    _tocomplex.stypy_type_of_self = None
    _tocomplex.stypy_type_store = module_type_store
    _tocomplex.stypy_function_name = '_tocomplex'
    _tocomplex.stypy_param_names_list = ['arr']
    _tocomplex.stypy_varargs_param_name = None
    _tocomplex.stypy_kwargs_param_name = None
    _tocomplex.stypy_call_defaults = defaults
    _tocomplex.stypy_call_varargs = varargs
    _tocomplex.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_tocomplex', ['arr'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_tocomplex', localization, ['arr'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_tocomplex(...)' code ##################

    str_124940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, (-1)), 'str', "Convert its input `arr` to a complex array.\n\n    The input is returned as a complex array of the smallest type that will fit\n    the original data: types like single, byte, short, etc. become csingle,\n    while others become cdouble.\n\n    A copy of the input is always made.\n\n    Parameters\n    ----------\n    arr : array\n\n    Returns\n    -------\n    array\n        An array with the same input data as the input but in complex form.\n\n    Examples\n    --------\n\n    First, consider an input of type short:\n\n    >>> a = np.array([1,2,3],np.short)\n\n    >>> ac = np.lib.scimath._tocomplex(a); ac\n    array([ 1.+0.j,  2.+0.j,  3.+0.j], dtype=complex64)\n\n    >>> ac.dtype\n    dtype('complex64')\n\n    If the input is of type double, the output is correspondingly of the\n    complex double type as well:\n\n    >>> b = np.array([1,2,3],np.double)\n\n    >>> bc = np.lib.scimath._tocomplex(b); bc\n    array([ 1.+0.j,  2.+0.j,  3.+0.j])\n\n    >>> bc.dtype\n    dtype('complex128')\n\n    Note that even if the input was complex to begin with, a copy is still\n    made, since the astype() method always copies:\n\n    >>> c = np.array([1,2,3],np.csingle)\n\n    >>> cc = np.lib.scimath._tocomplex(c); cc\n    array([ 1.+0.j,  2.+0.j,  3.+0.j], dtype=complex64)\n\n    >>> c *= 2; c\n    array([ 2.+0.j,  4.+0.j,  6.+0.j], dtype=complex64)\n\n    >>> cc\n    array([ 1.+0.j,  2.+0.j,  3.+0.j], dtype=complex64)\n    ")
    
    
    # Call to issubclass(...): (line 91)
    # Processing the call arguments (line 91)
    # Getting the type of 'arr' (line 91)
    arr_124942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 18), 'arr', False)
    # Obtaining the member 'dtype' of a type (line 91)
    dtype_124943 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 18), arr_124942, 'dtype')
    # Obtaining the member 'type' of a type (line 91)
    type_124944 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 18), dtype_124943, 'type')
    
    # Obtaining an instance of the builtin type 'tuple' (line 91)
    tuple_124945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 35), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 91)
    # Adding element type (line 91)
    # Getting the type of 'nt' (line 91)
    nt_124946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 35), 'nt', False)
    # Obtaining the member 'single' of a type (line 91)
    single_124947 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 35), nt_124946, 'single')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 35), tuple_124945, single_124947)
    # Adding element type (line 91)
    # Getting the type of 'nt' (line 91)
    nt_124948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 46), 'nt', False)
    # Obtaining the member 'byte' of a type (line 91)
    byte_124949 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 46), nt_124948, 'byte')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 35), tuple_124945, byte_124949)
    # Adding element type (line 91)
    # Getting the type of 'nt' (line 91)
    nt_124950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 55), 'nt', False)
    # Obtaining the member 'short' of a type (line 91)
    short_124951 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 55), nt_124950, 'short')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 35), tuple_124945, short_124951)
    # Adding element type (line 91)
    # Getting the type of 'nt' (line 91)
    nt_124952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 65), 'nt', False)
    # Obtaining the member 'ubyte' of a type (line 91)
    ubyte_124953 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 65), nt_124952, 'ubyte')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 35), tuple_124945, ubyte_124953)
    # Adding element type (line 91)
    # Getting the type of 'nt' (line 92)
    nt_124954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 35), 'nt', False)
    # Obtaining the member 'ushort' of a type (line 92)
    ushort_124955 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 35), nt_124954, 'ushort')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 35), tuple_124945, ushort_124955)
    # Adding element type (line 91)
    # Getting the type of 'nt' (line 92)
    nt_124956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 46), 'nt', False)
    # Obtaining the member 'csingle' of a type (line 92)
    csingle_124957 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 46), nt_124956, 'csingle')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 35), tuple_124945, csingle_124957)
    
    # Processing the call keyword arguments (line 91)
    kwargs_124958 = {}
    # Getting the type of 'issubclass' (line 91)
    issubclass_124941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 7), 'issubclass', False)
    # Calling issubclass(args, kwargs) (line 91)
    issubclass_call_result_124959 = invoke(stypy.reporting.localization.Localization(__file__, 91, 7), issubclass_124941, *[type_124944, tuple_124945], **kwargs_124958)
    
    # Testing the type of an if condition (line 91)
    if_condition_124960 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 91, 4), issubclass_call_result_124959)
    # Assigning a type to the variable 'if_condition_124960' (line 91)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 4), 'if_condition_124960', if_condition_124960)
    # SSA begins for if statement (line 91)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to astype(...): (line 93)
    # Processing the call arguments (line 93)
    # Getting the type of 'nt' (line 93)
    nt_124963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 26), 'nt', False)
    # Obtaining the member 'csingle' of a type (line 93)
    csingle_124964 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 26), nt_124963, 'csingle')
    # Processing the call keyword arguments (line 93)
    kwargs_124965 = {}
    # Getting the type of 'arr' (line 93)
    arr_124961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 15), 'arr', False)
    # Obtaining the member 'astype' of a type (line 93)
    astype_124962 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 15), arr_124961, 'astype')
    # Calling astype(args, kwargs) (line 93)
    astype_call_result_124966 = invoke(stypy.reporting.localization.Localization(__file__, 93, 15), astype_124962, *[csingle_124964], **kwargs_124965)
    
    # Assigning a type to the variable 'stypy_return_type' (line 93)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'stypy_return_type', astype_call_result_124966)
    # SSA branch for the else part of an if statement (line 91)
    module_type_store.open_ssa_branch('else')
    
    # Call to astype(...): (line 95)
    # Processing the call arguments (line 95)
    # Getting the type of 'nt' (line 95)
    nt_124969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 26), 'nt', False)
    # Obtaining the member 'cdouble' of a type (line 95)
    cdouble_124970 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 26), nt_124969, 'cdouble')
    # Processing the call keyword arguments (line 95)
    kwargs_124971 = {}
    # Getting the type of 'arr' (line 95)
    arr_124967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 15), 'arr', False)
    # Obtaining the member 'astype' of a type (line 95)
    astype_124968 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 15), arr_124967, 'astype')
    # Calling astype(args, kwargs) (line 95)
    astype_call_result_124972 = invoke(stypy.reporting.localization.Localization(__file__, 95, 15), astype_124968, *[cdouble_124970], **kwargs_124971)
    
    # Assigning a type to the variable 'stypy_return_type' (line 95)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'stypy_return_type', astype_call_result_124972)
    # SSA join for if statement (line 91)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '_tocomplex(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_tocomplex' in the type store
    # Getting the type of 'stypy_return_type' (line 35)
    stypy_return_type_124973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_124973)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_tocomplex'
    return stypy_return_type_124973

# Assigning a type to the variable '_tocomplex' (line 35)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 0), '_tocomplex', _tocomplex)

@norecursion
def _fix_real_lt_zero(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_fix_real_lt_zero'
    module_type_store = module_type_store.open_function_context('_fix_real_lt_zero', 97, 0, False)
    
    # Passed parameters checking function
    _fix_real_lt_zero.stypy_localization = localization
    _fix_real_lt_zero.stypy_type_of_self = None
    _fix_real_lt_zero.stypy_type_store = module_type_store
    _fix_real_lt_zero.stypy_function_name = '_fix_real_lt_zero'
    _fix_real_lt_zero.stypy_param_names_list = ['x']
    _fix_real_lt_zero.stypy_varargs_param_name = None
    _fix_real_lt_zero.stypy_kwargs_param_name = None
    _fix_real_lt_zero.stypy_call_defaults = defaults
    _fix_real_lt_zero.stypy_call_varargs = varargs
    _fix_real_lt_zero.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_fix_real_lt_zero', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_fix_real_lt_zero', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_fix_real_lt_zero(...)' code ##################

    str_124974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, (-1)), 'str', 'Convert `x` to complex if it has real, negative components.\n\n    Otherwise, output is just the array version of the input (via asarray).\n\n    Parameters\n    ----------\n    x : array_like\n\n    Returns\n    -------\n    array\n\n    Examples\n    --------\n    >>> np.lib.scimath._fix_real_lt_zero([1,2])\n    array([1, 2])\n\n    >>> np.lib.scimath._fix_real_lt_zero([-1,2])\n    array([-1.+0.j,  2.+0.j])\n\n    ')
    
    # Assigning a Call to a Name (line 119):
    
    # Call to asarray(...): (line 119)
    # Processing the call arguments (line 119)
    # Getting the type of 'x' (line 119)
    x_124976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 16), 'x', False)
    # Processing the call keyword arguments (line 119)
    kwargs_124977 = {}
    # Getting the type of 'asarray' (line 119)
    asarray_124975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'asarray', False)
    # Calling asarray(args, kwargs) (line 119)
    asarray_call_result_124978 = invoke(stypy.reporting.localization.Localization(__file__, 119, 8), asarray_124975, *[x_124976], **kwargs_124977)
    
    # Assigning a type to the variable 'x' (line 119)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 4), 'x', asarray_call_result_124978)
    
    
    # Call to any(...): (line 120)
    # Processing the call arguments (line 120)
    
    # Call to isreal(...): (line 120)
    # Processing the call arguments (line 120)
    # Getting the type of 'x' (line 120)
    x_124981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 18), 'x', False)
    # Processing the call keyword arguments (line 120)
    kwargs_124982 = {}
    # Getting the type of 'isreal' (line 120)
    isreal_124980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 11), 'isreal', False)
    # Calling isreal(args, kwargs) (line 120)
    isreal_call_result_124983 = invoke(stypy.reporting.localization.Localization(__file__, 120, 11), isreal_124980, *[x_124981], **kwargs_124982)
    
    
    # Getting the type of 'x' (line 120)
    x_124984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 24), 'x', False)
    int_124985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 28), 'int')
    # Applying the binary operator '<' (line 120)
    result_lt_124986 = python_operator(stypy.reporting.localization.Localization(__file__, 120, 24), '<', x_124984, int_124985)
    
    # Applying the binary operator '&' (line 120)
    result_and__124987 = python_operator(stypy.reporting.localization.Localization(__file__, 120, 11), '&', isreal_call_result_124983, result_lt_124986)
    
    # Processing the call keyword arguments (line 120)
    kwargs_124988 = {}
    # Getting the type of 'any' (line 120)
    any_124979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 7), 'any', False)
    # Calling any(args, kwargs) (line 120)
    any_call_result_124989 = invoke(stypy.reporting.localization.Localization(__file__, 120, 7), any_124979, *[result_and__124987], **kwargs_124988)
    
    # Testing the type of an if condition (line 120)
    if_condition_124990 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 120, 4), any_call_result_124989)
    # Assigning a type to the variable 'if_condition_124990' (line 120)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 4), 'if_condition_124990', if_condition_124990)
    # SSA begins for if statement (line 120)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 121):
    
    # Call to _tocomplex(...): (line 121)
    # Processing the call arguments (line 121)
    # Getting the type of 'x' (line 121)
    x_124992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 23), 'x', False)
    # Processing the call keyword arguments (line 121)
    kwargs_124993 = {}
    # Getting the type of '_tocomplex' (line 121)
    _tocomplex_124991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 12), '_tocomplex', False)
    # Calling _tocomplex(args, kwargs) (line 121)
    _tocomplex_call_result_124994 = invoke(stypy.reporting.localization.Localization(__file__, 121, 12), _tocomplex_124991, *[x_124992], **kwargs_124993)
    
    # Assigning a type to the variable 'x' (line 121)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'x', _tocomplex_call_result_124994)
    # SSA join for if statement (line 120)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'x' (line 122)
    x_124995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 11), 'x')
    # Assigning a type to the variable 'stypy_return_type' (line 122)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 4), 'stypy_return_type', x_124995)
    
    # ################# End of '_fix_real_lt_zero(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_fix_real_lt_zero' in the type store
    # Getting the type of 'stypy_return_type' (line 97)
    stypy_return_type_124996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_124996)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_fix_real_lt_zero'
    return stypy_return_type_124996

# Assigning a type to the variable '_fix_real_lt_zero' (line 97)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 0), '_fix_real_lt_zero', _fix_real_lt_zero)

@norecursion
def _fix_int_lt_zero(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_fix_int_lt_zero'
    module_type_store = module_type_store.open_function_context('_fix_int_lt_zero', 124, 0, False)
    
    # Passed parameters checking function
    _fix_int_lt_zero.stypy_localization = localization
    _fix_int_lt_zero.stypy_type_of_self = None
    _fix_int_lt_zero.stypy_type_store = module_type_store
    _fix_int_lt_zero.stypy_function_name = '_fix_int_lt_zero'
    _fix_int_lt_zero.stypy_param_names_list = ['x']
    _fix_int_lt_zero.stypy_varargs_param_name = None
    _fix_int_lt_zero.stypy_kwargs_param_name = None
    _fix_int_lt_zero.stypy_call_defaults = defaults
    _fix_int_lt_zero.stypy_call_varargs = varargs
    _fix_int_lt_zero.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_fix_int_lt_zero', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_fix_int_lt_zero', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_fix_int_lt_zero(...)' code ##################

    str_124997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, (-1)), 'str', 'Convert `x` to double if it has real, negative components.\n\n    Otherwise, output is just the array version of the input (via asarray).\n\n    Parameters\n    ----------\n    x : array_like\n\n    Returns\n    -------\n    array\n\n    Examples\n    --------\n    >>> np.lib.scimath._fix_int_lt_zero([1,2])\n    array([1, 2])\n\n    >>> np.lib.scimath._fix_int_lt_zero([-1,2])\n    array([-1.,  2.])\n    ')
    
    # Assigning a Call to a Name (line 145):
    
    # Call to asarray(...): (line 145)
    # Processing the call arguments (line 145)
    # Getting the type of 'x' (line 145)
    x_124999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 16), 'x', False)
    # Processing the call keyword arguments (line 145)
    kwargs_125000 = {}
    # Getting the type of 'asarray' (line 145)
    asarray_124998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 8), 'asarray', False)
    # Calling asarray(args, kwargs) (line 145)
    asarray_call_result_125001 = invoke(stypy.reporting.localization.Localization(__file__, 145, 8), asarray_124998, *[x_124999], **kwargs_125000)
    
    # Assigning a type to the variable 'x' (line 145)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 4), 'x', asarray_call_result_125001)
    
    
    # Call to any(...): (line 146)
    # Processing the call arguments (line 146)
    
    # Call to isreal(...): (line 146)
    # Processing the call arguments (line 146)
    # Getting the type of 'x' (line 146)
    x_125004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 18), 'x', False)
    # Processing the call keyword arguments (line 146)
    kwargs_125005 = {}
    # Getting the type of 'isreal' (line 146)
    isreal_125003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 11), 'isreal', False)
    # Calling isreal(args, kwargs) (line 146)
    isreal_call_result_125006 = invoke(stypy.reporting.localization.Localization(__file__, 146, 11), isreal_125003, *[x_125004], **kwargs_125005)
    
    
    # Getting the type of 'x' (line 146)
    x_125007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 24), 'x', False)
    int_125008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 28), 'int')
    # Applying the binary operator '<' (line 146)
    result_lt_125009 = python_operator(stypy.reporting.localization.Localization(__file__, 146, 24), '<', x_125007, int_125008)
    
    # Applying the binary operator '&' (line 146)
    result_and__125010 = python_operator(stypy.reporting.localization.Localization(__file__, 146, 11), '&', isreal_call_result_125006, result_lt_125009)
    
    # Processing the call keyword arguments (line 146)
    kwargs_125011 = {}
    # Getting the type of 'any' (line 146)
    any_125002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 7), 'any', False)
    # Calling any(args, kwargs) (line 146)
    any_call_result_125012 = invoke(stypy.reporting.localization.Localization(__file__, 146, 7), any_125002, *[result_and__125010], **kwargs_125011)
    
    # Testing the type of an if condition (line 146)
    if_condition_125013 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 146, 4), any_call_result_125012)
    # Assigning a type to the variable 'if_condition_125013' (line 146)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 4), 'if_condition_125013', if_condition_125013)
    # SSA begins for if statement (line 146)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 147):
    # Getting the type of 'x' (line 147)
    x_125014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 12), 'x')
    float_125015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 16), 'float')
    # Applying the binary operator '*' (line 147)
    result_mul_125016 = python_operator(stypy.reporting.localization.Localization(__file__, 147, 12), '*', x_125014, float_125015)
    
    # Assigning a type to the variable 'x' (line 147)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'x', result_mul_125016)
    # SSA join for if statement (line 146)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'x' (line 148)
    x_125017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 11), 'x')
    # Assigning a type to the variable 'stypy_return_type' (line 148)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 4), 'stypy_return_type', x_125017)
    
    # ################# End of '_fix_int_lt_zero(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_fix_int_lt_zero' in the type store
    # Getting the type of 'stypy_return_type' (line 124)
    stypy_return_type_125018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_125018)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_fix_int_lt_zero'
    return stypy_return_type_125018

# Assigning a type to the variable '_fix_int_lt_zero' (line 124)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 0), '_fix_int_lt_zero', _fix_int_lt_zero)

@norecursion
def _fix_real_abs_gt_1(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_fix_real_abs_gt_1'
    module_type_store = module_type_store.open_function_context('_fix_real_abs_gt_1', 150, 0, False)
    
    # Passed parameters checking function
    _fix_real_abs_gt_1.stypy_localization = localization
    _fix_real_abs_gt_1.stypy_type_of_self = None
    _fix_real_abs_gt_1.stypy_type_store = module_type_store
    _fix_real_abs_gt_1.stypy_function_name = '_fix_real_abs_gt_1'
    _fix_real_abs_gt_1.stypy_param_names_list = ['x']
    _fix_real_abs_gt_1.stypy_varargs_param_name = None
    _fix_real_abs_gt_1.stypy_kwargs_param_name = None
    _fix_real_abs_gt_1.stypy_call_defaults = defaults
    _fix_real_abs_gt_1.stypy_call_varargs = varargs
    _fix_real_abs_gt_1.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_fix_real_abs_gt_1', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_fix_real_abs_gt_1', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_fix_real_abs_gt_1(...)' code ##################

    str_125019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, (-1)), 'str', 'Convert `x` to complex if it has real components x_i with abs(x_i)>1.\n\n    Otherwise, output is just the array version of the input (via asarray).\n\n    Parameters\n    ----------\n    x : array_like\n\n    Returns\n    -------\n    array\n\n    Examples\n    --------\n    >>> np.lib.scimath._fix_real_abs_gt_1([0,1])\n    array([0, 1])\n\n    >>> np.lib.scimath._fix_real_abs_gt_1([0,2])\n    array([ 0.+0.j,  2.+0.j])\n    ')
    
    # Assigning a Call to a Name (line 171):
    
    # Call to asarray(...): (line 171)
    # Processing the call arguments (line 171)
    # Getting the type of 'x' (line 171)
    x_125021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 16), 'x', False)
    # Processing the call keyword arguments (line 171)
    kwargs_125022 = {}
    # Getting the type of 'asarray' (line 171)
    asarray_125020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 8), 'asarray', False)
    # Calling asarray(args, kwargs) (line 171)
    asarray_call_result_125023 = invoke(stypy.reporting.localization.Localization(__file__, 171, 8), asarray_125020, *[x_125021], **kwargs_125022)
    
    # Assigning a type to the variable 'x' (line 171)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 4), 'x', asarray_call_result_125023)
    
    
    # Call to any(...): (line 172)
    # Processing the call arguments (line 172)
    
    # Call to isreal(...): (line 172)
    # Processing the call arguments (line 172)
    # Getting the type of 'x' (line 172)
    x_125026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 18), 'x', False)
    # Processing the call keyword arguments (line 172)
    kwargs_125027 = {}
    # Getting the type of 'isreal' (line 172)
    isreal_125025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 11), 'isreal', False)
    # Calling isreal(args, kwargs) (line 172)
    isreal_call_result_125028 = invoke(stypy.reporting.localization.Localization(__file__, 172, 11), isreal_125025, *[x_125026], **kwargs_125027)
    
    
    
    # Call to abs(...): (line 172)
    # Processing the call arguments (line 172)
    # Getting the type of 'x' (line 172)
    x_125030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 28), 'x', False)
    # Processing the call keyword arguments (line 172)
    kwargs_125031 = {}
    # Getting the type of 'abs' (line 172)
    abs_125029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 24), 'abs', False)
    # Calling abs(args, kwargs) (line 172)
    abs_call_result_125032 = invoke(stypy.reporting.localization.Localization(__file__, 172, 24), abs_125029, *[x_125030], **kwargs_125031)
    
    int_125033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 33), 'int')
    # Applying the binary operator '>' (line 172)
    result_gt_125034 = python_operator(stypy.reporting.localization.Localization(__file__, 172, 24), '>', abs_call_result_125032, int_125033)
    
    # Applying the binary operator '&' (line 172)
    result_and__125035 = python_operator(stypy.reporting.localization.Localization(__file__, 172, 11), '&', isreal_call_result_125028, result_gt_125034)
    
    # Processing the call keyword arguments (line 172)
    kwargs_125036 = {}
    # Getting the type of 'any' (line 172)
    any_125024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 7), 'any', False)
    # Calling any(args, kwargs) (line 172)
    any_call_result_125037 = invoke(stypy.reporting.localization.Localization(__file__, 172, 7), any_125024, *[result_and__125035], **kwargs_125036)
    
    # Testing the type of an if condition (line 172)
    if_condition_125038 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 172, 4), any_call_result_125037)
    # Assigning a type to the variable 'if_condition_125038' (line 172)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 4), 'if_condition_125038', if_condition_125038)
    # SSA begins for if statement (line 172)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 173):
    
    # Call to _tocomplex(...): (line 173)
    # Processing the call arguments (line 173)
    # Getting the type of 'x' (line 173)
    x_125040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 23), 'x', False)
    # Processing the call keyword arguments (line 173)
    kwargs_125041 = {}
    # Getting the type of '_tocomplex' (line 173)
    _tocomplex_125039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 12), '_tocomplex', False)
    # Calling _tocomplex(args, kwargs) (line 173)
    _tocomplex_call_result_125042 = invoke(stypy.reporting.localization.Localization(__file__, 173, 12), _tocomplex_125039, *[x_125040], **kwargs_125041)
    
    # Assigning a type to the variable 'x' (line 173)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 8), 'x', _tocomplex_call_result_125042)
    # SSA join for if statement (line 172)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'x' (line 174)
    x_125043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 11), 'x')
    # Assigning a type to the variable 'stypy_return_type' (line 174)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 4), 'stypy_return_type', x_125043)
    
    # ################# End of '_fix_real_abs_gt_1(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_fix_real_abs_gt_1' in the type store
    # Getting the type of 'stypy_return_type' (line 150)
    stypy_return_type_125044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_125044)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_fix_real_abs_gt_1'
    return stypy_return_type_125044

# Assigning a type to the variable '_fix_real_abs_gt_1' (line 150)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 0), '_fix_real_abs_gt_1', _fix_real_abs_gt_1)

@norecursion
def sqrt(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'sqrt'
    module_type_store = module_type_store.open_function_context('sqrt', 176, 0, False)
    
    # Passed parameters checking function
    sqrt.stypy_localization = localization
    sqrt.stypy_type_of_self = None
    sqrt.stypy_type_store = module_type_store
    sqrt.stypy_function_name = 'sqrt'
    sqrt.stypy_param_names_list = ['x']
    sqrt.stypy_varargs_param_name = None
    sqrt.stypy_kwargs_param_name = None
    sqrt.stypy_call_defaults = defaults
    sqrt.stypy_call_varargs = varargs
    sqrt.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'sqrt', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'sqrt', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'sqrt(...)' code ##################

    str_125045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, (-1)), 'str', '\n    Compute the square root of x.\n\n    For negative input elements, a complex value is returned\n    (unlike `numpy.sqrt` which returns NaN).\n\n    Parameters\n    ----------\n    x : array_like\n       The input value(s).\n\n    Returns\n    -------\n    out : ndarray or scalar\n       The square root of `x`. If `x` was a scalar, so is `out`,\n       otherwise an array is returned.\n\n    See Also\n    --------\n    numpy.sqrt\n\n    Examples\n    --------\n    For real, non-negative inputs this works just like `numpy.sqrt`:\n\n    >>> np.lib.scimath.sqrt(1)\n    1.0\n    >>> np.lib.scimath.sqrt([1, 4])\n    array([ 1.,  2.])\n\n    But it automatically handles negative inputs:\n\n    >>> np.lib.scimath.sqrt(-1)\n    (0.0+1.0j)\n    >>> np.lib.scimath.sqrt([-1,4])\n    array([ 0.+1.j,  2.+0.j])\n\n    ')
    
    # Assigning a Call to a Name (line 215):
    
    # Call to _fix_real_lt_zero(...): (line 215)
    # Processing the call arguments (line 215)
    # Getting the type of 'x' (line 215)
    x_125047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 26), 'x', False)
    # Processing the call keyword arguments (line 215)
    kwargs_125048 = {}
    # Getting the type of '_fix_real_lt_zero' (line 215)
    _fix_real_lt_zero_125046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 8), '_fix_real_lt_zero', False)
    # Calling _fix_real_lt_zero(args, kwargs) (line 215)
    _fix_real_lt_zero_call_result_125049 = invoke(stypy.reporting.localization.Localization(__file__, 215, 8), _fix_real_lt_zero_125046, *[x_125047], **kwargs_125048)
    
    # Assigning a type to the variable 'x' (line 215)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 4), 'x', _fix_real_lt_zero_call_result_125049)
    
    # Call to sqrt(...): (line 216)
    # Processing the call arguments (line 216)
    # Getting the type of 'x' (line 216)
    x_125052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 19), 'x', False)
    # Processing the call keyword arguments (line 216)
    kwargs_125053 = {}
    # Getting the type of 'nx' (line 216)
    nx_125050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 11), 'nx', False)
    # Obtaining the member 'sqrt' of a type (line 216)
    sqrt_125051 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 11), nx_125050, 'sqrt')
    # Calling sqrt(args, kwargs) (line 216)
    sqrt_call_result_125054 = invoke(stypy.reporting.localization.Localization(__file__, 216, 11), sqrt_125051, *[x_125052], **kwargs_125053)
    
    # Assigning a type to the variable 'stypy_return_type' (line 216)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 4), 'stypy_return_type', sqrt_call_result_125054)
    
    # ################# End of 'sqrt(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'sqrt' in the type store
    # Getting the type of 'stypy_return_type' (line 176)
    stypy_return_type_125055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_125055)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'sqrt'
    return stypy_return_type_125055

# Assigning a type to the variable 'sqrt' (line 176)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 0), 'sqrt', sqrt)

@norecursion
def log(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'log'
    module_type_store = module_type_store.open_function_context('log', 218, 0, False)
    
    # Passed parameters checking function
    log.stypy_localization = localization
    log.stypy_type_of_self = None
    log.stypy_type_store = module_type_store
    log.stypy_function_name = 'log'
    log.stypy_param_names_list = ['x']
    log.stypy_varargs_param_name = None
    log.stypy_kwargs_param_name = None
    log.stypy_call_defaults = defaults
    log.stypy_call_varargs = varargs
    log.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'log', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'log', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'log(...)' code ##################

    str_125056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, (-1)), 'str', '\n    Compute the natural logarithm of `x`.\n\n    Return the "principal value" (for a description of this, see `numpy.log`)\n    of :math:`log_e(x)`. For real `x > 0`, this is a real number (``log(0)``\n    returns ``-inf`` and ``log(np.inf)`` returns ``inf``). Otherwise, the\n    complex principle value is returned.\n\n    Parameters\n    ----------\n    x : array_like\n       The value(s) whose log is (are) required.\n\n    Returns\n    -------\n    out : ndarray or scalar\n       The log of the `x` value(s). If `x` was a scalar, so is `out`,\n       otherwise an array is returned.\n\n    See Also\n    --------\n    numpy.log\n\n    Notes\n    -----\n    For a log() that returns ``NAN`` when real `x < 0`, use `numpy.log`\n    (note, however, that otherwise `numpy.log` and this `log` are identical,\n    i.e., both return ``-inf`` for `x = 0`, ``inf`` for `x = inf`, and,\n    notably, the complex principle value if ``x.imag != 0``).\n\n    Examples\n    --------\n    >>> np.emath.log(np.exp(1))\n    1.0\n\n    Negative arguments are handled "correctly" (recall that\n    ``exp(log(x)) == x`` does *not* hold for real ``x < 0``):\n\n    >>> np.emath.log(-np.exp(1)) == (1 + np.pi * 1j)\n    True\n\n    ')
    
    # Assigning a Call to a Name (line 261):
    
    # Call to _fix_real_lt_zero(...): (line 261)
    # Processing the call arguments (line 261)
    # Getting the type of 'x' (line 261)
    x_125058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 26), 'x', False)
    # Processing the call keyword arguments (line 261)
    kwargs_125059 = {}
    # Getting the type of '_fix_real_lt_zero' (line 261)
    _fix_real_lt_zero_125057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 8), '_fix_real_lt_zero', False)
    # Calling _fix_real_lt_zero(args, kwargs) (line 261)
    _fix_real_lt_zero_call_result_125060 = invoke(stypy.reporting.localization.Localization(__file__, 261, 8), _fix_real_lt_zero_125057, *[x_125058], **kwargs_125059)
    
    # Assigning a type to the variable 'x' (line 261)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 4), 'x', _fix_real_lt_zero_call_result_125060)
    
    # Call to log(...): (line 262)
    # Processing the call arguments (line 262)
    # Getting the type of 'x' (line 262)
    x_125063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 18), 'x', False)
    # Processing the call keyword arguments (line 262)
    kwargs_125064 = {}
    # Getting the type of 'nx' (line 262)
    nx_125061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 11), 'nx', False)
    # Obtaining the member 'log' of a type (line 262)
    log_125062 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 11), nx_125061, 'log')
    # Calling log(args, kwargs) (line 262)
    log_call_result_125065 = invoke(stypy.reporting.localization.Localization(__file__, 262, 11), log_125062, *[x_125063], **kwargs_125064)
    
    # Assigning a type to the variable 'stypy_return_type' (line 262)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 4), 'stypy_return_type', log_call_result_125065)
    
    # ################# End of 'log(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'log' in the type store
    # Getting the type of 'stypy_return_type' (line 218)
    stypy_return_type_125066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_125066)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'log'
    return stypy_return_type_125066

# Assigning a type to the variable 'log' (line 218)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 0), 'log', log)

@norecursion
def log10(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'log10'
    module_type_store = module_type_store.open_function_context('log10', 264, 0, False)
    
    # Passed parameters checking function
    log10.stypy_localization = localization
    log10.stypy_type_of_self = None
    log10.stypy_type_store = module_type_store
    log10.stypy_function_name = 'log10'
    log10.stypy_param_names_list = ['x']
    log10.stypy_varargs_param_name = None
    log10.stypy_kwargs_param_name = None
    log10.stypy_call_defaults = defaults
    log10.stypy_call_varargs = varargs
    log10.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'log10', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'log10', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'log10(...)' code ##################

    str_125067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, (-1)), 'str', '\n    Compute the logarithm base 10 of `x`.\n\n    Return the "principal value" (for a description of this, see\n    `numpy.log10`) of :math:`log_{10}(x)`. For real `x > 0`, this\n    is a real number (``log10(0)`` returns ``-inf`` and ``log10(np.inf)``\n    returns ``inf``). Otherwise, the complex principle value is returned.\n\n    Parameters\n    ----------\n    x : array_like or scalar\n       The value(s) whose log base 10 is (are) required.\n\n    Returns\n    -------\n    out : ndarray or scalar\n       The log base 10 of the `x` value(s). If `x` was a scalar, so is `out`,\n       otherwise an array object is returned.\n\n    See Also\n    --------\n    numpy.log10\n\n    Notes\n    -----\n    For a log10() that returns ``NAN`` when real `x < 0`, use `numpy.log10`\n    (note, however, that otherwise `numpy.log10` and this `log10` are\n    identical, i.e., both return ``-inf`` for `x = 0`, ``inf`` for `x = inf`,\n    and, notably, the complex principle value if ``x.imag != 0``).\n\n    Examples\n    --------\n\n    (We set the printing precision so the example can be auto-tested)\n\n    >>> np.set_printoptions(precision=4)\n\n    >>> np.emath.log10(10**1)\n    1.0\n\n    >>> np.emath.log10([-10**1, -10**2, 10**2])\n    array([ 1.+1.3644j,  2.+1.3644j,  2.+0.j    ])\n\n    ')
    
    # Assigning a Call to a Name (line 309):
    
    # Call to _fix_real_lt_zero(...): (line 309)
    # Processing the call arguments (line 309)
    # Getting the type of 'x' (line 309)
    x_125069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 26), 'x', False)
    # Processing the call keyword arguments (line 309)
    kwargs_125070 = {}
    # Getting the type of '_fix_real_lt_zero' (line 309)
    _fix_real_lt_zero_125068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 8), '_fix_real_lt_zero', False)
    # Calling _fix_real_lt_zero(args, kwargs) (line 309)
    _fix_real_lt_zero_call_result_125071 = invoke(stypy.reporting.localization.Localization(__file__, 309, 8), _fix_real_lt_zero_125068, *[x_125069], **kwargs_125070)
    
    # Assigning a type to the variable 'x' (line 309)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 4), 'x', _fix_real_lt_zero_call_result_125071)
    
    # Call to log10(...): (line 310)
    # Processing the call arguments (line 310)
    # Getting the type of 'x' (line 310)
    x_125074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 20), 'x', False)
    # Processing the call keyword arguments (line 310)
    kwargs_125075 = {}
    # Getting the type of 'nx' (line 310)
    nx_125072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 11), 'nx', False)
    # Obtaining the member 'log10' of a type (line 310)
    log10_125073 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 11), nx_125072, 'log10')
    # Calling log10(args, kwargs) (line 310)
    log10_call_result_125076 = invoke(stypy.reporting.localization.Localization(__file__, 310, 11), log10_125073, *[x_125074], **kwargs_125075)
    
    # Assigning a type to the variable 'stypy_return_type' (line 310)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 4), 'stypy_return_type', log10_call_result_125076)
    
    # ################# End of 'log10(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'log10' in the type store
    # Getting the type of 'stypy_return_type' (line 264)
    stypy_return_type_125077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_125077)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'log10'
    return stypy_return_type_125077

# Assigning a type to the variable 'log10' (line 264)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 0), 'log10', log10)

@norecursion
def logn(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'logn'
    module_type_store = module_type_store.open_function_context('logn', 312, 0, False)
    
    # Passed parameters checking function
    logn.stypy_localization = localization
    logn.stypy_type_of_self = None
    logn.stypy_type_store = module_type_store
    logn.stypy_function_name = 'logn'
    logn.stypy_param_names_list = ['n', 'x']
    logn.stypy_varargs_param_name = None
    logn.stypy_kwargs_param_name = None
    logn.stypy_call_defaults = defaults
    logn.stypy_call_varargs = varargs
    logn.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'logn', ['n', 'x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'logn', localization, ['n', 'x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'logn(...)' code ##################

    str_125078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, (-1)), 'str', '\n    Take log base n of x.\n\n    If `x` contains negative inputs, the answer is computed and returned in the\n    complex domain.\n\n    Parameters\n    ----------\n    n : int\n       The base in which the log is taken.\n    x : array_like\n       The value(s) whose log base `n` is (are) required.\n\n    Returns\n    -------\n    out : ndarray or scalar\n       The log base `n` of the `x` value(s). If `x` was a scalar, so is\n       `out`, otherwise an array is returned.\n\n    Examples\n    --------\n    >>> np.set_printoptions(precision=4)\n\n    >>> np.lib.scimath.logn(2, [4, 8])\n    array([ 2.,  3.])\n    >>> np.lib.scimath.logn(2, [-4, -8, 8])\n    array([ 2.+4.5324j,  3.+4.5324j,  3.+0.j    ])\n\n    ')
    
    # Assigning a Call to a Name (line 342):
    
    # Call to _fix_real_lt_zero(...): (line 342)
    # Processing the call arguments (line 342)
    # Getting the type of 'x' (line 342)
    x_125080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 26), 'x', False)
    # Processing the call keyword arguments (line 342)
    kwargs_125081 = {}
    # Getting the type of '_fix_real_lt_zero' (line 342)
    _fix_real_lt_zero_125079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 8), '_fix_real_lt_zero', False)
    # Calling _fix_real_lt_zero(args, kwargs) (line 342)
    _fix_real_lt_zero_call_result_125082 = invoke(stypy.reporting.localization.Localization(__file__, 342, 8), _fix_real_lt_zero_125079, *[x_125080], **kwargs_125081)
    
    # Assigning a type to the variable 'x' (line 342)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 4), 'x', _fix_real_lt_zero_call_result_125082)
    
    # Assigning a Call to a Name (line 343):
    
    # Call to _fix_real_lt_zero(...): (line 343)
    # Processing the call arguments (line 343)
    # Getting the type of 'n' (line 343)
    n_125084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 26), 'n', False)
    # Processing the call keyword arguments (line 343)
    kwargs_125085 = {}
    # Getting the type of '_fix_real_lt_zero' (line 343)
    _fix_real_lt_zero_125083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 8), '_fix_real_lt_zero', False)
    # Calling _fix_real_lt_zero(args, kwargs) (line 343)
    _fix_real_lt_zero_call_result_125086 = invoke(stypy.reporting.localization.Localization(__file__, 343, 8), _fix_real_lt_zero_125083, *[n_125084], **kwargs_125085)
    
    # Assigning a type to the variable 'n' (line 343)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 4), 'n', _fix_real_lt_zero_call_result_125086)
    
    # Call to log(...): (line 344)
    # Processing the call arguments (line 344)
    # Getting the type of 'x' (line 344)
    x_125089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 18), 'x', False)
    # Processing the call keyword arguments (line 344)
    kwargs_125090 = {}
    # Getting the type of 'nx' (line 344)
    nx_125087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 11), 'nx', False)
    # Obtaining the member 'log' of a type (line 344)
    log_125088 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 11), nx_125087, 'log')
    # Calling log(args, kwargs) (line 344)
    log_call_result_125091 = invoke(stypy.reporting.localization.Localization(__file__, 344, 11), log_125088, *[x_125089], **kwargs_125090)
    
    
    # Call to log(...): (line 344)
    # Processing the call arguments (line 344)
    # Getting the type of 'n' (line 344)
    n_125094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 28), 'n', False)
    # Processing the call keyword arguments (line 344)
    kwargs_125095 = {}
    # Getting the type of 'nx' (line 344)
    nx_125092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 21), 'nx', False)
    # Obtaining the member 'log' of a type (line 344)
    log_125093 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 21), nx_125092, 'log')
    # Calling log(args, kwargs) (line 344)
    log_call_result_125096 = invoke(stypy.reporting.localization.Localization(__file__, 344, 21), log_125093, *[n_125094], **kwargs_125095)
    
    # Applying the binary operator 'div' (line 344)
    result_div_125097 = python_operator(stypy.reporting.localization.Localization(__file__, 344, 11), 'div', log_call_result_125091, log_call_result_125096)
    
    # Assigning a type to the variable 'stypy_return_type' (line 344)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 344, 4), 'stypy_return_type', result_div_125097)
    
    # ################# End of 'logn(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'logn' in the type store
    # Getting the type of 'stypy_return_type' (line 312)
    stypy_return_type_125098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_125098)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'logn'
    return stypy_return_type_125098

# Assigning a type to the variable 'logn' (line 312)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 0), 'logn', logn)

@norecursion
def log2(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'log2'
    module_type_store = module_type_store.open_function_context('log2', 346, 0, False)
    
    # Passed parameters checking function
    log2.stypy_localization = localization
    log2.stypy_type_of_self = None
    log2.stypy_type_store = module_type_store
    log2.stypy_function_name = 'log2'
    log2.stypy_param_names_list = ['x']
    log2.stypy_varargs_param_name = None
    log2.stypy_kwargs_param_name = None
    log2.stypy_call_defaults = defaults
    log2.stypy_call_varargs = varargs
    log2.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'log2', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'log2', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'log2(...)' code ##################

    str_125099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 388, (-1)), 'str', '\n    Compute the logarithm base 2 of `x`.\n\n    Return the "principal value" (for a description of this, see\n    `numpy.log2`) of :math:`log_2(x)`. For real `x > 0`, this is\n    a real number (``log2(0)`` returns ``-inf`` and ``log2(np.inf)`` returns\n    ``inf``). Otherwise, the complex principle value is returned.\n\n    Parameters\n    ----------\n    x : array_like\n       The value(s) whose log base 2 is (are) required.\n\n    Returns\n    -------\n    out : ndarray or scalar\n       The log base 2 of the `x` value(s). If `x` was a scalar, so is `out`,\n       otherwise an array is returned.\n\n    See Also\n    --------\n    numpy.log2\n\n    Notes\n    -----\n    For a log2() that returns ``NAN`` when real `x < 0`, use `numpy.log2`\n    (note, however, that otherwise `numpy.log2` and this `log2` are\n    identical, i.e., both return ``-inf`` for `x = 0`, ``inf`` for `x = inf`,\n    and, notably, the complex principle value if ``x.imag != 0``).\n\n    Examples\n    --------\n    We set the printing precision so the example can be auto-tested:\n\n    >>> np.set_printoptions(precision=4)\n\n    >>> np.emath.log2(8)\n    3.0\n    >>> np.emath.log2([-4, -8, 8])\n    array([ 2.+4.5324j,  3.+4.5324j,  3.+0.j    ])\n\n    ')
    
    # Assigning a Call to a Name (line 389):
    
    # Call to _fix_real_lt_zero(...): (line 389)
    # Processing the call arguments (line 389)
    # Getting the type of 'x' (line 389)
    x_125101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 26), 'x', False)
    # Processing the call keyword arguments (line 389)
    kwargs_125102 = {}
    # Getting the type of '_fix_real_lt_zero' (line 389)
    _fix_real_lt_zero_125100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 8), '_fix_real_lt_zero', False)
    # Calling _fix_real_lt_zero(args, kwargs) (line 389)
    _fix_real_lt_zero_call_result_125103 = invoke(stypy.reporting.localization.Localization(__file__, 389, 8), _fix_real_lt_zero_125100, *[x_125101], **kwargs_125102)
    
    # Assigning a type to the variable 'x' (line 389)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 4), 'x', _fix_real_lt_zero_call_result_125103)
    
    # Call to log2(...): (line 390)
    # Processing the call arguments (line 390)
    # Getting the type of 'x' (line 390)
    x_125106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 19), 'x', False)
    # Processing the call keyword arguments (line 390)
    kwargs_125107 = {}
    # Getting the type of 'nx' (line 390)
    nx_125104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 11), 'nx', False)
    # Obtaining the member 'log2' of a type (line 390)
    log2_125105 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 11), nx_125104, 'log2')
    # Calling log2(args, kwargs) (line 390)
    log2_call_result_125108 = invoke(stypy.reporting.localization.Localization(__file__, 390, 11), log2_125105, *[x_125106], **kwargs_125107)
    
    # Assigning a type to the variable 'stypy_return_type' (line 390)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 390, 4), 'stypy_return_type', log2_call_result_125108)
    
    # ################# End of 'log2(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'log2' in the type store
    # Getting the type of 'stypy_return_type' (line 346)
    stypy_return_type_125109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_125109)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'log2'
    return stypy_return_type_125109

# Assigning a type to the variable 'log2' (line 346)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 346, 0), 'log2', log2)

@norecursion
def power(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'power'
    module_type_store = module_type_store.open_function_context('power', 392, 0, False)
    
    # Passed parameters checking function
    power.stypy_localization = localization
    power.stypy_type_of_self = None
    power.stypy_type_store = module_type_store
    power.stypy_function_name = 'power'
    power.stypy_param_names_list = ['x', 'p']
    power.stypy_varargs_param_name = None
    power.stypy_kwargs_param_name = None
    power.stypy_call_defaults = defaults
    power.stypy_call_varargs = varargs
    power.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'power', ['x', 'p'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'power', localization, ['x', 'p'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'power(...)' code ##################

    str_125110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 430, (-1)), 'str', '\n    Return x to the power p, (x**p).\n\n    If `x` contains negative values, the output is converted to the\n    complex domain.\n\n    Parameters\n    ----------\n    x : array_like\n        The input value(s).\n    p : array_like of ints\n        The power(s) to which `x` is raised. If `x` contains multiple values,\n        `p` has to either be a scalar, or contain the same number of values\n        as `x`. In the latter case, the result is\n        ``x[0]**p[0], x[1]**p[1], ...``.\n\n    Returns\n    -------\n    out : ndarray or scalar\n        The result of ``x**p``. If `x` and `p` are scalars, so is `out`,\n        otherwise an array is returned.\n\n    See Also\n    --------\n    numpy.power\n\n    Examples\n    --------\n    >>> np.set_printoptions(precision=4)\n\n    >>> np.lib.scimath.power([2, 4], 2)\n    array([ 4, 16])\n    >>> np.lib.scimath.power([2, 4], -2)\n    array([ 0.25  ,  0.0625])\n    >>> np.lib.scimath.power([-2, 4], 2)\n    array([  4.+0.j,  16.+0.j])\n\n    ')
    
    # Assigning a Call to a Name (line 431):
    
    # Call to _fix_real_lt_zero(...): (line 431)
    # Processing the call arguments (line 431)
    # Getting the type of 'x' (line 431)
    x_125112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 26), 'x', False)
    # Processing the call keyword arguments (line 431)
    kwargs_125113 = {}
    # Getting the type of '_fix_real_lt_zero' (line 431)
    _fix_real_lt_zero_125111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 8), '_fix_real_lt_zero', False)
    # Calling _fix_real_lt_zero(args, kwargs) (line 431)
    _fix_real_lt_zero_call_result_125114 = invoke(stypy.reporting.localization.Localization(__file__, 431, 8), _fix_real_lt_zero_125111, *[x_125112], **kwargs_125113)
    
    # Assigning a type to the variable 'x' (line 431)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 431, 4), 'x', _fix_real_lt_zero_call_result_125114)
    
    # Assigning a Call to a Name (line 432):
    
    # Call to _fix_int_lt_zero(...): (line 432)
    # Processing the call arguments (line 432)
    # Getting the type of 'p' (line 432)
    p_125116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 25), 'p', False)
    # Processing the call keyword arguments (line 432)
    kwargs_125117 = {}
    # Getting the type of '_fix_int_lt_zero' (line 432)
    _fix_int_lt_zero_125115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 8), '_fix_int_lt_zero', False)
    # Calling _fix_int_lt_zero(args, kwargs) (line 432)
    _fix_int_lt_zero_call_result_125118 = invoke(stypy.reporting.localization.Localization(__file__, 432, 8), _fix_int_lt_zero_125115, *[p_125116], **kwargs_125117)
    
    # Assigning a type to the variable 'p' (line 432)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 432, 4), 'p', _fix_int_lt_zero_call_result_125118)
    
    # Call to power(...): (line 433)
    # Processing the call arguments (line 433)
    # Getting the type of 'x' (line 433)
    x_125121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 20), 'x', False)
    # Getting the type of 'p' (line 433)
    p_125122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 23), 'p', False)
    # Processing the call keyword arguments (line 433)
    kwargs_125123 = {}
    # Getting the type of 'nx' (line 433)
    nx_125119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 11), 'nx', False)
    # Obtaining the member 'power' of a type (line 433)
    power_125120 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 433, 11), nx_125119, 'power')
    # Calling power(args, kwargs) (line 433)
    power_call_result_125124 = invoke(stypy.reporting.localization.Localization(__file__, 433, 11), power_125120, *[x_125121, p_125122], **kwargs_125123)
    
    # Assigning a type to the variable 'stypy_return_type' (line 433)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 433, 4), 'stypy_return_type', power_call_result_125124)
    
    # ################# End of 'power(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'power' in the type store
    # Getting the type of 'stypy_return_type' (line 392)
    stypy_return_type_125125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_125125)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'power'
    return stypy_return_type_125125

# Assigning a type to the variable 'power' (line 392)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 392, 0), 'power', power)

@norecursion
def arccos(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'arccos'
    module_type_store = module_type_store.open_function_context('arccos', 435, 0, False)
    
    # Passed parameters checking function
    arccos.stypy_localization = localization
    arccos.stypy_type_of_self = None
    arccos.stypy_type_store = module_type_store
    arccos.stypy_function_name = 'arccos'
    arccos.stypy_param_names_list = ['x']
    arccos.stypy_varargs_param_name = None
    arccos.stypy_kwargs_param_name = None
    arccos.stypy_call_defaults = defaults
    arccos.stypy_call_varargs = varargs
    arccos.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'arccos', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'arccos', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'arccos(...)' code ##################

    str_125126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 474, (-1)), 'str', '\n    Compute the inverse cosine of x.\n\n    Return the "principal value" (for a description of this, see\n    `numpy.arccos`) of the inverse cosine of `x`. For real `x` such that\n    `abs(x) <= 1`, this is a real number in the closed interval\n    :math:`[0, \\pi]`.  Otherwise, the complex principle value is returned.\n\n    Parameters\n    ----------\n    x : array_like or scalar\n       The value(s) whose arccos is (are) required.\n\n    Returns\n    -------\n    out : ndarray or scalar\n       The inverse cosine(s) of the `x` value(s). If `x` was a scalar, so\n       is `out`, otherwise an array object is returned.\n\n    See Also\n    --------\n    numpy.arccos\n\n    Notes\n    -----\n    For an arccos() that returns ``NAN`` when real `x` is not in the\n    interval ``[-1,1]``, use `numpy.arccos`.\n\n    Examples\n    --------\n    >>> np.set_printoptions(precision=4)\n\n    >>> np.emath.arccos(1) # a scalar is returned\n    0.0\n\n    >>> np.emath.arccos([1,2])\n    array([ 0.-0.j   ,  0.+1.317j])\n\n    ')
    
    # Assigning a Call to a Name (line 475):
    
    # Call to _fix_real_abs_gt_1(...): (line 475)
    # Processing the call arguments (line 475)
    # Getting the type of 'x' (line 475)
    x_125128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 27), 'x', False)
    # Processing the call keyword arguments (line 475)
    kwargs_125129 = {}
    # Getting the type of '_fix_real_abs_gt_1' (line 475)
    _fix_real_abs_gt_1_125127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 8), '_fix_real_abs_gt_1', False)
    # Calling _fix_real_abs_gt_1(args, kwargs) (line 475)
    _fix_real_abs_gt_1_call_result_125130 = invoke(stypy.reporting.localization.Localization(__file__, 475, 8), _fix_real_abs_gt_1_125127, *[x_125128], **kwargs_125129)
    
    # Assigning a type to the variable 'x' (line 475)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 475, 4), 'x', _fix_real_abs_gt_1_call_result_125130)
    
    # Call to arccos(...): (line 476)
    # Processing the call arguments (line 476)
    # Getting the type of 'x' (line 476)
    x_125133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 21), 'x', False)
    # Processing the call keyword arguments (line 476)
    kwargs_125134 = {}
    # Getting the type of 'nx' (line 476)
    nx_125131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 11), 'nx', False)
    # Obtaining the member 'arccos' of a type (line 476)
    arccos_125132 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 476, 11), nx_125131, 'arccos')
    # Calling arccos(args, kwargs) (line 476)
    arccos_call_result_125135 = invoke(stypy.reporting.localization.Localization(__file__, 476, 11), arccos_125132, *[x_125133], **kwargs_125134)
    
    # Assigning a type to the variable 'stypy_return_type' (line 476)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 476, 4), 'stypy_return_type', arccos_call_result_125135)
    
    # ################# End of 'arccos(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'arccos' in the type store
    # Getting the type of 'stypy_return_type' (line 435)
    stypy_return_type_125136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_125136)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'arccos'
    return stypy_return_type_125136

# Assigning a type to the variable 'arccos' (line 435)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 435, 0), 'arccos', arccos)

@norecursion
def arcsin(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'arcsin'
    module_type_store = module_type_store.open_function_context('arcsin', 478, 0, False)
    
    # Passed parameters checking function
    arcsin.stypy_localization = localization
    arcsin.stypy_type_of_self = None
    arcsin.stypy_type_store = module_type_store
    arcsin.stypy_function_name = 'arcsin'
    arcsin.stypy_param_names_list = ['x']
    arcsin.stypy_varargs_param_name = None
    arcsin.stypy_kwargs_param_name = None
    arcsin.stypy_call_defaults = defaults
    arcsin.stypy_call_varargs = varargs
    arcsin.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'arcsin', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'arcsin', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'arcsin(...)' code ##################

    str_125137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 518, (-1)), 'str', '\n    Compute the inverse sine of x.\n\n    Return the "principal value" (for a description of this, see\n    `numpy.arcsin`) of the inverse sine of `x`. For real `x` such that\n    `abs(x) <= 1`, this is a real number in the closed interval\n    :math:`[-\\pi/2, \\pi/2]`.  Otherwise, the complex principle value is\n    returned.\n\n    Parameters\n    ----------\n    x : array_like or scalar\n       The value(s) whose arcsin is (are) required.\n\n    Returns\n    -------\n    out : ndarray or scalar\n       The inverse sine(s) of the `x` value(s). If `x` was a scalar, so\n       is `out`, otherwise an array object is returned.\n\n    See Also\n    --------\n    numpy.arcsin\n\n    Notes\n    -----\n    For an arcsin() that returns ``NAN`` when real `x` is not in the\n    interval ``[-1,1]``, use `numpy.arcsin`.\n\n    Examples\n    --------\n    >>> np.set_printoptions(precision=4)\n\n    >>> np.emath.arcsin(0)\n    0.0\n\n    >>> np.emath.arcsin([0,1])\n    array([ 0.    ,  1.5708])\n\n    ')
    
    # Assigning a Call to a Name (line 519):
    
    # Call to _fix_real_abs_gt_1(...): (line 519)
    # Processing the call arguments (line 519)
    # Getting the type of 'x' (line 519)
    x_125139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 27), 'x', False)
    # Processing the call keyword arguments (line 519)
    kwargs_125140 = {}
    # Getting the type of '_fix_real_abs_gt_1' (line 519)
    _fix_real_abs_gt_1_125138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 8), '_fix_real_abs_gt_1', False)
    # Calling _fix_real_abs_gt_1(args, kwargs) (line 519)
    _fix_real_abs_gt_1_call_result_125141 = invoke(stypy.reporting.localization.Localization(__file__, 519, 8), _fix_real_abs_gt_1_125138, *[x_125139], **kwargs_125140)
    
    # Assigning a type to the variable 'x' (line 519)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 519, 4), 'x', _fix_real_abs_gt_1_call_result_125141)
    
    # Call to arcsin(...): (line 520)
    # Processing the call arguments (line 520)
    # Getting the type of 'x' (line 520)
    x_125144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 21), 'x', False)
    # Processing the call keyword arguments (line 520)
    kwargs_125145 = {}
    # Getting the type of 'nx' (line 520)
    nx_125142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 11), 'nx', False)
    # Obtaining the member 'arcsin' of a type (line 520)
    arcsin_125143 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 520, 11), nx_125142, 'arcsin')
    # Calling arcsin(args, kwargs) (line 520)
    arcsin_call_result_125146 = invoke(stypy.reporting.localization.Localization(__file__, 520, 11), arcsin_125143, *[x_125144], **kwargs_125145)
    
    # Assigning a type to the variable 'stypy_return_type' (line 520)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 520, 4), 'stypy_return_type', arcsin_call_result_125146)
    
    # ################# End of 'arcsin(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'arcsin' in the type store
    # Getting the type of 'stypy_return_type' (line 478)
    stypy_return_type_125147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_125147)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'arcsin'
    return stypy_return_type_125147

# Assigning a type to the variable 'arcsin' (line 478)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 478, 0), 'arcsin', arcsin)

@norecursion
def arctanh(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'arctanh'
    module_type_store = module_type_store.open_function_context('arctanh', 522, 0, False)
    
    # Passed parameters checking function
    arctanh.stypy_localization = localization
    arctanh.stypy_type_of_self = None
    arctanh.stypy_type_store = module_type_store
    arctanh.stypy_function_name = 'arctanh'
    arctanh.stypy_param_names_list = ['x']
    arctanh.stypy_varargs_param_name = None
    arctanh.stypy_kwargs_param_name = None
    arctanh.stypy_call_defaults = defaults
    arctanh.stypy_call_varargs = varargs
    arctanh.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'arctanh', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'arctanh', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'arctanh(...)' code ##################

    str_125148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 564, (-1)), 'str', '\n    Compute the inverse hyperbolic tangent of `x`.\n\n    Return the "principal value" (for a description of this, see\n    `numpy.arctanh`) of `arctanh(x)`. For real `x` such that\n    `abs(x) < 1`, this is a real number.  If `abs(x) > 1`, or if `x` is\n    complex, the result is complex. Finally, `x = 1` returns``inf`` and\n    `x=-1` returns ``-inf``.\n\n    Parameters\n    ----------\n    x : array_like\n       The value(s) whose arctanh is (are) required.\n\n    Returns\n    -------\n    out : ndarray or scalar\n       The inverse hyperbolic tangent(s) of the `x` value(s). If `x` was\n       a scalar so is `out`, otherwise an array is returned.\n\n\n    See Also\n    --------\n    numpy.arctanh\n\n    Notes\n    -----\n    For an arctanh() that returns ``NAN`` when real `x` is not in the\n    interval ``(-1,1)``, use `numpy.arctanh` (this latter, however, does\n    return +/-inf for `x = +/-1`).\n\n    Examples\n    --------\n    >>> np.set_printoptions(precision=4)\n\n    >>> np.emath.arctanh(np.matrix(np.eye(2)))\n    array([[ Inf,   0.],\n           [  0.,  Inf]])\n    >>> np.emath.arctanh([1j])\n    array([ 0.+0.7854j])\n\n    ')
    
    # Assigning a Call to a Name (line 565):
    
    # Call to _fix_real_abs_gt_1(...): (line 565)
    # Processing the call arguments (line 565)
    # Getting the type of 'x' (line 565)
    x_125150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 27), 'x', False)
    # Processing the call keyword arguments (line 565)
    kwargs_125151 = {}
    # Getting the type of '_fix_real_abs_gt_1' (line 565)
    _fix_real_abs_gt_1_125149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 8), '_fix_real_abs_gt_1', False)
    # Calling _fix_real_abs_gt_1(args, kwargs) (line 565)
    _fix_real_abs_gt_1_call_result_125152 = invoke(stypy.reporting.localization.Localization(__file__, 565, 8), _fix_real_abs_gt_1_125149, *[x_125150], **kwargs_125151)
    
    # Assigning a type to the variable 'x' (line 565)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 565, 4), 'x', _fix_real_abs_gt_1_call_result_125152)
    
    # Call to arctanh(...): (line 566)
    # Processing the call arguments (line 566)
    # Getting the type of 'x' (line 566)
    x_125155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 22), 'x', False)
    # Processing the call keyword arguments (line 566)
    kwargs_125156 = {}
    # Getting the type of 'nx' (line 566)
    nx_125153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 11), 'nx', False)
    # Obtaining the member 'arctanh' of a type (line 566)
    arctanh_125154 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 566, 11), nx_125153, 'arctanh')
    # Calling arctanh(args, kwargs) (line 566)
    arctanh_call_result_125157 = invoke(stypy.reporting.localization.Localization(__file__, 566, 11), arctanh_125154, *[x_125155], **kwargs_125156)
    
    # Assigning a type to the variable 'stypy_return_type' (line 566)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 566, 4), 'stypy_return_type', arctanh_call_result_125157)
    
    # ################# End of 'arctanh(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'arctanh' in the type store
    # Getting the type of 'stypy_return_type' (line 522)
    stypy_return_type_125158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_125158)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'arctanh'
    return stypy_return_type_125158

# Assigning a type to the variable 'arctanh' (line 522)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 522, 0), 'arctanh', arctanh)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
