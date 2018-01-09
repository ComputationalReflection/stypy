
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Automatically adapted for numpy Sep 19, 2005 by convertcode.py
2: 
3: '''
4: from __future__ import division, absolute_import, print_function
5: 
6: __all__ = ['iscomplexobj', 'isrealobj', 'imag', 'iscomplex',
7:            'isreal', 'nan_to_num', 'real', 'real_if_close',
8:            'typename', 'asfarray', 'mintypecode', 'asscalar',
9:            'common_type']
10: 
11: import numpy.core.numeric as _nx
12: from numpy.core.numeric import asarray, asanyarray, array, isnan, \
13:                 obj2sctype, zeros
14: from .ufunclike import isneginf, isposinf
15: 
16: _typecodes_by_elsize = 'GDFgdfQqLlIiHhBb?'
17: 
18: def mintypecode(typechars,typeset='GDFgdf',default='d'):
19:     '''
20:     Return the character for the minimum-size type to which given types can
21:     be safely cast.
22: 
23:     The returned type character must represent the smallest size dtype such
24:     that an array of the returned type can handle the data from an array of
25:     all types in `typechars` (or if `typechars` is an array, then its
26:     dtype.char).
27: 
28:     Parameters
29:     ----------
30:     typechars : list of str or array_like
31:         If a list of strings, each string should represent a dtype.
32:         If array_like, the character representation of the array dtype is used.
33:     typeset : str or list of str, optional
34:         The set of characters that the returned character is chosen from.
35:         The default set is 'GDFgdf'.
36:     default : str, optional
37:         The default character, this is returned if none of the characters in
38:         `typechars` matches a character in `typeset`.
39: 
40:     Returns
41:     -------
42:     typechar : str
43:         The character representing the minimum-size type that was found.
44: 
45:     See Also
46:     --------
47:     dtype, sctype2char, maximum_sctype
48: 
49:     Examples
50:     --------
51:     >>> np.mintypecode(['d', 'f', 'S'])
52:     'd'
53:     >>> x = np.array([1.1, 2-3.j])
54:     >>> np.mintypecode(x)
55:     'D'
56: 
57:     >>> np.mintypecode('abceh', default='G')
58:     'G'
59: 
60:     '''
61:     typecodes = [(isinstance(t, str) and t) or asarray(t).dtype.char
62:                  for t in typechars]
63:     intersection = [t for t in typecodes if t in typeset]
64:     if not intersection:
65:         return default
66:     if 'F' in intersection and 'd' in intersection:
67:         return 'D'
68:     l = []
69:     for t in intersection:
70:         i = _typecodes_by_elsize.index(t)
71:         l.append((i, t))
72:     l.sort()
73:     return l[0][1]
74: 
75: def asfarray(a, dtype=_nx.float_):
76:     '''
77:     Return an array converted to a float type.
78: 
79:     Parameters
80:     ----------
81:     a : array_like
82:         The input array.
83:     dtype : str or dtype object, optional
84:         Float type code to coerce input array `a`.  If `dtype` is one of the
85:         'int' dtypes, it is replaced with float64.
86: 
87:     Returns
88:     -------
89:     out : ndarray
90:         The input `a` as a float ndarray.
91: 
92:     Examples
93:     --------
94:     >>> np.asfarray([2, 3])
95:     array([ 2.,  3.])
96:     >>> np.asfarray([2, 3], dtype='float')
97:     array([ 2.,  3.])
98:     >>> np.asfarray([2, 3], dtype='int8')
99:     array([ 2.,  3.])
100: 
101:     '''
102:     dtype = _nx.obj2sctype(dtype)
103:     if not issubclass(dtype, _nx.inexact):
104:         dtype = _nx.float_
105:     return asarray(a, dtype=dtype)
106: 
107: def real(val):
108:     '''
109:     Return the real part of the elements of the array.
110: 
111:     Parameters
112:     ----------
113:     val : array_like
114:         Input array.
115: 
116:     Returns
117:     -------
118:     out : ndarray
119:         Output array. If `val` is real, the type of `val` is used for the
120:         output.  If `val` has complex elements, the returned type is float.
121: 
122:     See Also
123:     --------
124:     real_if_close, imag, angle
125: 
126:     Examples
127:     --------
128:     >>> a = np.array([1+2j, 3+4j, 5+6j])
129:     >>> a.real
130:     array([ 1.,  3.,  5.])
131:     >>> a.real = 9
132:     >>> a
133:     array([ 9.+2.j,  9.+4.j,  9.+6.j])
134:     >>> a.real = np.array([9, 8, 7])
135:     >>> a
136:     array([ 9.+2.j,  8.+4.j,  7.+6.j])
137: 
138:     '''
139:     return asanyarray(val).real
140: 
141: def imag(val):
142:     '''
143:     Return the imaginary part of the elements of the array.
144: 
145:     Parameters
146:     ----------
147:     val : array_like
148:         Input array.
149: 
150:     Returns
151:     -------
152:     out : ndarray
153:         Output array. If `val` is real, the type of `val` is used for the
154:         output.  If `val` has complex elements, the returned type is float.
155: 
156:     See Also
157:     --------
158:     real, angle, real_if_close
159: 
160:     Examples
161:     --------
162:     >>> a = np.array([1+2j, 3+4j, 5+6j])
163:     >>> a.imag
164:     array([ 2.,  4.,  6.])
165:     >>> a.imag = np.array([8, 10, 12])
166:     >>> a
167:     array([ 1. +8.j,  3.+10.j,  5.+12.j])
168: 
169:     '''
170:     return asanyarray(val).imag
171: 
172: def iscomplex(x):
173:     '''
174:     Returns a bool array, where True if input element is complex.
175: 
176:     What is tested is whether the input has a non-zero imaginary part, not if
177:     the input type is complex.
178: 
179:     Parameters
180:     ----------
181:     x : array_like
182:         Input array.
183: 
184:     Returns
185:     -------
186:     out : ndarray of bools
187:         Output array.
188: 
189:     See Also
190:     --------
191:     isreal
192:     iscomplexobj : Return True if x is a complex type or an array of complex
193:                    numbers.
194: 
195:     Examples
196:     --------
197:     >>> np.iscomplex([1+1j, 1+0j, 4.5, 3, 2, 2j])
198:     array([ True, False, False, False, False,  True], dtype=bool)
199: 
200:     '''
201:     ax = asanyarray(x)
202:     if issubclass(ax.dtype.type, _nx.complexfloating):
203:         return ax.imag != 0
204:     res = zeros(ax.shape, bool)
205:     return +res  # convet to array-scalar if needed
206: 
207: def isreal(x):
208:     '''
209:     Returns a bool array, where True if input element is real.
210: 
211:     If element has complex type with zero complex part, the return value
212:     for that element is True.
213: 
214:     Parameters
215:     ----------
216:     x : array_like
217:         Input array.
218: 
219:     Returns
220:     -------
221:     out : ndarray, bool
222:         Boolean array of same shape as `x`.
223: 
224:     See Also
225:     --------
226:     iscomplex
227:     isrealobj : Return True if x is not a complex type.
228: 
229:     Examples
230:     --------
231:     >>> np.isreal([1+1j, 1+0j, 4.5, 3, 2, 2j])
232:     array([False,  True,  True,  True,  True, False], dtype=bool)
233: 
234:     '''
235:     return imag(x) == 0
236: 
237: def iscomplexobj(x):
238:     '''
239:     Check for a complex type or an array of complex numbers.
240: 
241:     The type of the input is checked, not the value. Even if the input
242:     has an imaginary part equal to zero, `iscomplexobj` evaluates to True.
243: 
244:     Parameters
245:     ----------
246:     x : any
247:         The input can be of any type and shape.
248: 
249:     Returns
250:     -------
251:     iscomplexobj : bool
252:         The return value, True if `x` is of a complex type or has at least
253:         one complex element.
254: 
255:     See Also
256:     --------
257:     isrealobj, iscomplex
258: 
259:     Examples
260:     --------
261:     >>> np.iscomplexobj(1)
262:     False
263:     >>> np.iscomplexobj(1+0j)
264:     True
265:     >>> np.iscomplexobj([3, 1+0j, True])
266:     True
267: 
268:     '''
269:     return issubclass(asarray(x).dtype.type, _nx.complexfloating)
270: 
271: def isrealobj(x):
272:     '''
273:     Return True if x is a not complex type or an array of complex numbers.
274: 
275:     The type of the input is checked, not the value. So even if the input
276:     has an imaginary part equal to zero, `isrealobj` evaluates to False
277:     if the data type is complex.
278: 
279:     Parameters
280:     ----------
281:     x : any
282:         The input can be of any type and shape.
283: 
284:     Returns
285:     -------
286:     y : bool
287:         The return value, False if `x` is of a complex type.
288: 
289:     See Also
290:     --------
291:     iscomplexobj, isreal
292: 
293:     Examples
294:     --------
295:     >>> np.isrealobj(1)
296:     True
297:     >>> np.isrealobj(1+0j)
298:     False
299:     >>> np.isrealobj([3, 1+0j, True])
300:     False
301: 
302:     '''
303:     return not issubclass(asarray(x).dtype.type, _nx.complexfloating)
304: 
305: #-----------------------------------------------------------------------------
306: 
307: def _getmaxmin(t):
308:     from numpy.core import getlimits
309:     f = getlimits.finfo(t)
310:     return f.max, f.min
311: 
312: def nan_to_num(x):
313:     '''
314:     Replace nan with zero and inf with finite numbers.
315: 
316:     Returns an array or scalar replacing Not a Number (NaN) with zero,
317:     (positive) infinity with a very large number and negative infinity
318:     with a very small (or negative) number.
319: 
320:     Parameters
321:     ----------
322:     x : array_like
323:         Input data.
324: 
325:     Returns
326:     -------
327:     out : ndarray
328:         New Array with the same shape as `x` and dtype of the element in
329:         `x`  with the greatest precision. If `x` is inexact, then NaN is
330:         replaced by zero, and infinity (-infinity) is replaced by the
331:         largest (smallest or most negative) floating point value that fits
332:         in the output dtype. If `x` is not inexact, then a copy of `x` is
333:         returned.
334: 
335:     See Also
336:     --------
337:     isinf : Shows which elements are negative or negative infinity.
338:     isneginf : Shows which elements are negative infinity.
339:     isposinf : Shows which elements are positive infinity.
340:     isnan : Shows which elements are Not a Number (NaN).
341:     isfinite : Shows which elements are finite (not NaN, not infinity)
342: 
343:     Notes
344:     -----
345:     Numpy uses the IEEE Standard for Binary Floating-Point for Arithmetic
346:     (IEEE 754). This means that Not a Number is not equivalent to infinity.
347: 
348: 
349:     Examples
350:     --------
351:     >>> np.set_printoptions(precision=8)
352:     >>> x = np.array([np.inf, -np.inf, np.nan, -128, 128])
353:     >>> np.nan_to_num(x)
354:     array([  1.79769313e+308,  -1.79769313e+308,   0.00000000e+000,
355:             -1.28000000e+002,   1.28000000e+002])
356: 
357:     '''
358:     x = _nx.array(x, subok=True)
359:     xtype = x.dtype.type
360:     if not issubclass(xtype, _nx.inexact):
361:         return x
362: 
363:     iscomplex = issubclass(xtype, _nx.complexfloating)
364:     isscalar = (x.ndim == 0)
365: 
366:     x = x[None] if isscalar else x
367:     dest = (x.real, x.imag) if iscomplex else (x,)
368:     maxf, minf = _getmaxmin(x.real.dtype)
369:     for d in dest:
370:         _nx.copyto(d, 0.0, where=isnan(d))
371:         _nx.copyto(d, maxf, where=isposinf(d))
372:         _nx.copyto(d, minf, where=isneginf(d))
373:     return x[0] if isscalar else x
374: 
375: #-----------------------------------------------------------------------------
376: 
377: def real_if_close(a,tol=100):
378:     '''
379:     If complex input returns a real array if complex parts are close to zero.
380: 
381:     "Close to zero" is defined as `tol` * (machine epsilon of the type for
382:     `a`).
383: 
384:     Parameters
385:     ----------
386:     a : array_like
387:         Input array.
388:     tol : float
389:         Tolerance in machine epsilons for the complex part of the elements
390:         in the array.
391: 
392:     Returns
393:     -------
394:     out : ndarray
395:         If `a` is real, the type of `a` is used for the output.  If `a`
396:         has complex elements, the returned type is float.
397: 
398:     See Also
399:     --------
400:     real, imag, angle
401: 
402:     Notes
403:     -----
404:     Machine epsilon varies from machine to machine and between data types
405:     but Python floats on most platforms have a machine epsilon equal to
406:     2.2204460492503131e-16.  You can use 'np.finfo(np.float).eps' to print
407:     out the machine epsilon for floats.
408: 
409:     Examples
410:     --------
411:     >>> np.finfo(np.float).eps
412:     2.2204460492503131e-16
413: 
414:     >>> np.real_if_close([2.1 + 4e-14j], tol=1000)
415:     array([ 2.1])
416:     >>> np.real_if_close([2.1 + 4e-13j], tol=1000)
417:     array([ 2.1 +4.00000000e-13j])
418: 
419:     '''
420:     a = asanyarray(a)
421:     if not issubclass(a.dtype.type, _nx.complexfloating):
422:         return a
423:     if tol > 1:
424:         from numpy.core import getlimits
425:         f = getlimits.finfo(a.dtype.type)
426:         tol = f.eps * tol
427:     if _nx.allclose(a.imag, 0, atol=tol):
428:         a = a.real
429:     return a
430: 
431: 
432: def asscalar(a):
433:     '''
434:     Convert an array of size 1 to its scalar equivalent.
435: 
436:     Parameters
437:     ----------
438:     a : ndarray
439:         Input array of size 1.
440: 
441:     Returns
442:     -------
443:     out : scalar
444:         Scalar representation of `a`. The output data type is the same type
445:         returned by the input's `item` method.
446: 
447:     Examples
448:     --------
449:     >>> np.asscalar(np.array([24]))
450:     24
451: 
452:     '''
453:     return a.item()
454: 
455: #-----------------------------------------------------------------------------
456: 
457: _namefromtype = {'S1': 'character',
458:                  '?': 'bool',
459:                  'b': 'signed char',
460:                  'B': 'unsigned char',
461:                  'h': 'short',
462:                  'H': 'unsigned short',
463:                  'i': 'integer',
464:                  'I': 'unsigned integer',
465:                  'l': 'long integer',
466:                  'L': 'unsigned long integer',
467:                  'q': 'long long integer',
468:                  'Q': 'unsigned long long integer',
469:                  'f': 'single precision',
470:                  'd': 'double precision',
471:                  'g': 'long precision',
472:                  'F': 'complex single precision',
473:                  'D': 'complex double precision',
474:                  'G': 'complex long double precision',
475:                  'S': 'string',
476:                  'U': 'unicode',
477:                  'V': 'void',
478:                  'O': 'object'
479:                  }
480: 
481: def typename(char):
482:     '''
483:     Return a description for the given data type code.
484: 
485:     Parameters
486:     ----------
487:     char : str
488:         Data type code.
489: 
490:     Returns
491:     -------
492:     out : str
493:         Description of the input data type code.
494: 
495:     See Also
496:     --------
497:     dtype, typecodes
498: 
499:     Examples
500:     --------
501:     >>> typechars = ['S1', '?', 'B', 'D', 'G', 'F', 'I', 'H', 'L', 'O', 'Q',
502:     ...              'S', 'U', 'V', 'b', 'd', 'g', 'f', 'i', 'h', 'l', 'q']
503:     >>> for typechar in typechars:
504:     ...     print(typechar, ' : ', np.typename(typechar))
505:     ...
506:     S1  :  character
507:     ?  :  bool
508:     B  :  unsigned char
509:     D  :  complex double precision
510:     G  :  complex long double precision
511:     F  :  complex single precision
512:     I  :  unsigned integer
513:     H  :  unsigned short
514:     L  :  unsigned long integer
515:     O  :  object
516:     Q  :  unsigned long long integer
517:     S  :  string
518:     U  :  unicode
519:     V  :  void
520:     b  :  signed char
521:     d  :  double precision
522:     g  :  long precision
523:     f  :  single precision
524:     i  :  integer
525:     h  :  short
526:     l  :  long integer
527:     q  :  long long integer
528: 
529:     '''
530:     return _namefromtype[char]
531: 
532: #-----------------------------------------------------------------------------
533: 
534: #determine the "minimum common type" for a group of arrays.
535: array_type = [[_nx.half, _nx.single, _nx.double, _nx.longdouble],
536:               [None, _nx.csingle, _nx.cdouble, _nx.clongdouble]]
537: array_precision = {_nx.half: 0,
538:                    _nx.single: 1,
539:                    _nx.double: 2,
540:                    _nx.longdouble: 3,
541:                    _nx.csingle: 1,
542:                    _nx.cdouble: 2,
543:                    _nx.clongdouble: 3}
544: def common_type(*arrays):
545:     '''
546:     Return a scalar type which is common to the input arrays.
547: 
548:     The return type will always be an inexact (i.e. floating point) scalar
549:     type, even if all the arrays are integer arrays. If one of the inputs is
550:     an integer array, the minimum precision type that is returned is a
551:     64-bit floating point dtype.
552: 
553:     All input arrays can be safely cast to the returned dtype without loss
554:     of information.
555: 
556:     Parameters
557:     ----------
558:     array1, array2, ... : ndarrays
559:         Input arrays.
560: 
561:     Returns
562:     -------
563:     out : data type code
564:         Data type code.
565: 
566:     See Also
567:     --------
568:     dtype, mintypecode
569: 
570:     Examples
571:     --------
572:     >>> np.common_type(np.arange(2, dtype=np.float32))
573:     <type 'numpy.float32'>
574:     >>> np.common_type(np.arange(2, dtype=np.float32), np.arange(2))
575:     <type 'numpy.float64'>
576:     >>> np.common_type(np.arange(4), np.array([45, 6.j]), np.array([45.0]))
577:     <type 'numpy.complex128'>
578: 
579:     '''
580:     is_complex = False
581:     precision = 0
582:     for a in arrays:
583:         t = a.dtype.type
584:         if iscomplexobj(a):
585:             is_complex = True
586:         if issubclass(t, _nx.integer):
587:             p = 2  # array_precision[_nx.double]
588:         else:
589:             p = array_precision.get(t, None)
590:             if p is None:
591:                 raise TypeError("can't get common type for non-numeric array")
592:         precision = max(precision, p)
593:     if is_complex:
594:         return array_type[1][precision]
595:     else:
596:         return array_type[0][precision]
597: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_127376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, (-1)), 'str', 'Automatically adapted for numpy Sep 19, 2005 by convertcode.py\n\n')

# Assigning a List to a Name (line 6):

# Assigning a List to a Name (line 6):
__all__ = ['iscomplexobj', 'isrealobj', 'imag', 'iscomplex', 'isreal', 'nan_to_num', 'real', 'real_if_close', 'typename', 'asfarray', 'mintypecode', 'asscalar', 'common_type']
module_type_store.set_exportable_members(['iscomplexobj', 'isrealobj', 'imag', 'iscomplex', 'isreal', 'nan_to_num', 'real', 'real_if_close', 'typename', 'asfarray', 'mintypecode', 'asscalar', 'common_type'])

# Obtaining an instance of the builtin type 'list' (line 6)
list_127377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 6)
# Adding element type (line 6)
str_127378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 11), 'str', 'iscomplexobj')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 10), list_127377, str_127378)
# Adding element type (line 6)
str_127379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 27), 'str', 'isrealobj')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 10), list_127377, str_127379)
# Adding element type (line 6)
str_127380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 40), 'str', 'imag')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 10), list_127377, str_127380)
# Adding element type (line 6)
str_127381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 48), 'str', 'iscomplex')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 10), list_127377, str_127381)
# Adding element type (line 6)
str_127382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 11), 'str', 'isreal')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 10), list_127377, str_127382)
# Adding element type (line 6)
str_127383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 21), 'str', 'nan_to_num')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 10), list_127377, str_127383)
# Adding element type (line 6)
str_127384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 35), 'str', 'real')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 10), list_127377, str_127384)
# Adding element type (line 6)
str_127385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 43), 'str', 'real_if_close')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 10), list_127377, str_127385)
# Adding element type (line 6)
str_127386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 11), 'str', 'typename')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 10), list_127377, str_127386)
# Adding element type (line 6)
str_127387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 23), 'str', 'asfarray')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 10), list_127377, str_127387)
# Adding element type (line 6)
str_127388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 35), 'str', 'mintypecode')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 10), list_127377, str_127388)
# Adding element type (line 6)
str_127389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 50), 'str', 'asscalar')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 10), list_127377, str_127389)
# Adding element type (line 6)
str_127390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 11), 'str', 'common_type')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 10), list_127377, str_127390)

# Assigning a type to the variable '__all__' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), '__all__', list_127377)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'import numpy.core.numeric' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/lib/')
import_127391 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy.core.numeric')

if (type(import_127391) is not StypyTypeError):

    if (import_127391 != 'pyd_module'):
        __import__(import_127391)
        sys_modules_127392 = sys.modules[import_127391]
        import_module(stypy.reporting.localization.Localization(__file__, 11, 0), '_nx', sys_modules_127392.module_type_store, module_type_store)
    else:
        import numpy.core.numeric as _nx

        import_module(stypy.reporting.localization.Localization(__file__, 11, 0), '_nx', numpy.core.numeric, module_type_store)

else:
    # Assigning a type to the variable 'numpy.core.numeric' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy.core.numeric', import_127391)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/lib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from numpy.core.numeric import asarray, asanyarray, array, isnan, obj2sctype, zeros' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/lib/')
import_127393 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'numpy.core.numeric')

if (type(import_127393) is not StypyTypeError):

    if (import_127393 != 'pyd_module'):
        __import__(import_127393)
        sys_modules_127394 = sys.modules[import_127393]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'numpy.core.numeric', sys_modules_127394.module_type_store, module_type_store, ['asarray', 'asanyarray', 'array', 'isnan', 'obj2sctype', 'zeros'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_127394, sys_modules_127394.module_type_store, module_type_store)
    else:
        from numpy.core.numeric import asarray, asanyarray, array, isnan, obj2sctype, zeros

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'numpy.core.numeric', None, module_type_store, ['asarray', 'asanyarray', 'array', 'isnan', 'obj2sctype', 'zeros'], [asarray, asanyarray, array, isnan, obj2sctype, zeros])

else:
    # Assigning a type to the variable 'numpy.core.numeric' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'numpy.core.numeric', import_127393)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/lib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'from numpy.lib.ufunclike import isneginf, isposinf' statement (line 14)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/lib/')
import_127395 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'numpy.lib.ufunclike')

if (type(import_127395) is not StypyTypeError):

    if (import_127395 != 'pyd_module'):
        __import__(import_127395)
        sys_modules_127396 = sys.modules[import_127395]
        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'numpy.lib.ufunclike', sys_modules_127396.module_type_store, module_type_store, ['isneginf', 'isposinf'])
        nest_module(stypy.reporting.localization.Localization(__file__, 14, 0), __file__, sys_modules_127396, sys_modules_127396.module_type_store, module_type_store)
    else:
        from numpy.lib.ufunclike import isneginf, isposinf

        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'numpy.lib.ufunclike', None, module_type_store, ['isneginf', 'isposinf'], [isneginf, isposinf])

else:
    # Assigning a type to the variable 'numpy.lib.ufunclike' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'numpy.lib.ufunclike', import_127395)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/lib/')


# Assigning a Str to a Name (line 16):

# Assigning a Str to a Name (line 16):
str_127397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 23), 'str', 'GDFgdfQqLlIiHhBb?')
# Assigning a type to the variable '_typecodes_by_elsize' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), '_typecodes_by_elsize', str_127397)

@norecursion
def mintypecode(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_127398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 34), 'str', 'GDFgdf')
    str_127399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 51), 'str', 'd')
    defaults = [str_127398, str_127399]
    # Create a new context for function 'mintypecode'
    module_type_store = module_type_store.open_function_context('mintypecode', 18, 0, False)
    
    # Passed parameters checking function
    mintypecode.stypy_localization = localization
    mintypecode.stypy_type_of_self = None
    mintypecode.stypy_type_store = module_type_store
    mintypecode.stypy_function_name = 'mintypecode'
    mintypecode.stypy_param_names_list = ['typechars', 'typeset', 'default']
    mintypecode.stypy_varargs_param_name = None
    mintypecode.stypy_kwargs_param_name = None
    mintypecode.stypy_call_defaults = defaults
    mintypecode.stypy_call_varargs = varargs
    mintypecode.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'mintypecode', ['typechars', 'typeset', 'default'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'mintypecode', localization, ['typechars', 'typeset', 'default'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'mintypecode(...)' code ##################

    str_127400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, (-1)), 'str', "\n    Return the character for the minimum-size type to which given types can\n    be safely cast.\n\n    The returned type character must represent the smallest size dtype such\n    that an array of the returned type can handle the data from an array of\n    all types in `typechars` (or if `typechars` is an array, then its\n    dtype.char).\n\n    Parameters\n    ----------\n    typechars : list of str or array_like\n        If a list of strings, each string should represent a dtype.\n        If array_like, the character representation of the array dtype is used.\n    typeset : str or list of str, optional\n        The set of characters that the returned character is chosen from.\n        The default set is 'GDFgdf'.\n    default : str, optional\n        The default character, this is returned if none of the characters in\n        `typechars` matches a character in `typeset`.\n\n    Returns\n    -------\n    typechar : str\n        The character representing the minimum-size type that was found.\n\n    See Also\n    --------\n    dtype, sctype2char, maximum_sctype\n\n    Examples\n    --------\n    >>> np.mintypecode(['d', 'f', 'S'])\n    'd'\n    >>> x = np.array([1.1, 2-3.j])\n    >>> np.mintypecode(x)\n    'D'\n\n    >>> np.mintypecode('abceh', default='G')\n    'G'\n\n    ")
    
    # Assigning a ListComp to a Name (line 61):
    
    # Assigning a ListComp to a Name (line 61):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'typechars' (line 62)
    typechars_127415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 26), 'typechars')
    comprehension_127416 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 17), typechars_127415)
    # Assigning a type to the variable 't' (line 61)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 17), 't', comprehension_127416)
    
    # Evaluating a boolean operation
    
    # Evaluating a boolean operation
    
    # Call to isinstance(...): (line 61)
    # Processing the call arguments (line 61)
    # Getting the type of 't' (line 61)
    t_127402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 29), 't', False)
    # Getting the type of 'str' (line 61)
    str_127403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 32), 'str', False)
    # Processing the call keyword arguments (line 61)
    kwargs_127404 = {}
    # Getting the type of 'isinstance' (line 61)
    isinstance_127401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 18), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 61)
    isinstance_call_result_127405 = invoke(stypy.reporting.localization.Localization(__file__, 61, 18), isinstance_127401, *[t_127402, str_127403], **kwargs_127404)
    
    # Getting the type of 't' (line 61)
    t_127406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 41), 't')
    # Applying the binary operator 'and' (line 61)
    result_and_keyword_127407 = python_operator(stypy.reporting.localization.Localization(__file__, 61, 18), 'and', isinstance_call_result_127405, t_127406)
    
    
    # Call to asarray(...): (line 61)
    # Processing the call arguments (line 61)
    # Getting the type of 't' (line 61)
    t_127409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 55), 't', False)
    # Processing the call keyword arguments (line 61)
    kwargs_127410 = {}
    # Getting the type of 'asarray' (line 61)
    asarray_127408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 47), 'asarray', False)
    # Calling asarray(args, kwargs) (line 61)
    asarray_call_result_127411 = invoke(stypy.reporting.localization.Localization(__file__, 61, 47), asarray_127408, *[t_127409], **kwargs_127410)
    
    # Obtaining the member 'dtype' of a type (line 61)
    dtype_127412 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 47), asarray_call_result_127411, 'dtype')
    # Obtaining the member 'char' of a type (line 61)
    char_127413 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 47), dtype_127412, 'char')
    # Applying the binary operator 'or' (line 61)
    result_or_keyword_127414 = python_operator(stypy.reporting.localization.Localization(__file__, 61, 17), 'or', result_and_keyword_127407, char_127413)
    
    list_127417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 17), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 17), list_127417, result_or_keyword_127414)
    # Assigning a type to the variable 'typecodes' (line 61)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), 'typecodes', list_127417)
    
    # Assigning a ListComp to a Name (line 63):
    
    # Assigning a ListComp to a Name (line 63):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'typecodes' (line 63)
    typecodes_127422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 31), 'typecodes')
    comprehension_127423 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 20), typecodes_127422)
    # Assigning a type to the variable 't' (line 63)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 20), 't', comprehension_127423)
    
    # Getting the type of 't' (line 63)
    t_127419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 44), 't')
    # Getting the type of 'typeset' (line 63)
    typeset_127420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 49), 'typeset')
    # Applying the binary operator 'in' (line 63)
    result_contains_127421 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 44), 'in', t_127419, typeset_127420)
    
    # Getting the type of 't' (line 63)
    t_127418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 20), 't')
    list_127424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 20), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 20), list_127424, t_127418)
    # Assigning a type to the variable 'intersection' (line 63)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'intersection', list_127424)
    
    
    # Getting the type of 'intersection' (line 64)
    intersection_127425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 11), 'intersection')
    # Applying the 'not' unary operator (line 64)
    result_not__127426 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 7), 'not', intersection_127425)
    
    # Testing the type of an if condition (line 64)
    if_condition_127427 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 64, 4), result_not__127426)
    # Assigning a type to the variable 'if_condition_127427' (line 64)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'if_condition_127427', if_condition_127427)
    # SSA begins for if statement (line 64)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'default' (line 65)
    default_127428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 15), 'default')
    # Assigning a type to the variable 'stypy_return_type' (line 65)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'stypy_return_type', default_127428)
    # SSA join for if statement (line 64)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    str_127429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 7), 'str', 'F')
    # Getting the type of 'intersection' (line 66)
    intersection_127430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 14), 'intersection')
    # Applying the binary operator 'in' (line 66)
    result_contains_127431 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 7), 'in', str_127429, intersection_127430)
    
    
    str_127432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 31), 'str', 'd')
    # Getting the type of 'intersection' (line 66)
    intersection_127433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 38), 'intersection')
    # Applying the binary operator 'in' (line 66)
    result_contains_127434 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 31), 'in', str_127432, intersection_127433)
    
    # Applying the binary operator 'and' (line 66)
    result_and_keyword_127435 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 7), 'and', result_contains_127431, result_contains_127434)
    
    # Testing the type of an if condition (line 66)
    if_condition_127436 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 66, 4), result_and_keyword_127435)
    # Assigning a type to the variable 'if_condition_127436' (line 66)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'if_condition_127436', if_condition_127436)
    # SSA begins for if statement (line 66)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    str_127437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 15), 'str', 'D')
    # Assigning a type to the variable 'stypy_return_type' (line 67)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'stypy_return_type', str_127437)
    # SSA join for if statement (line 66)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a List to a Name (line 68):
    
    # Assigning a List to a Name (line 68):
    
    # Obtaining an instance of the builtin type 'list' (line 68)
    list_127438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 68)
    
    # Assigning a type to the variable 'l' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'l', list_127438)
    
    # Getting the type of 'intersection' (line 69)
    intersection_127439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 13), 'intersection')
    # Testing the type of a for loop iterable (line 69)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 69, 4), intersection_127439)
    # Getting the type of the for loop variable (line 69)
    for_loop_var_127440 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 69, 4), intersection_127439)
    # Assigning a type to the variable 't' (line 69)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 't', for_loop_var_127440)
    # SSA begins for a for statement (line 69)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 70):
    
    # Assigning a Call to a Name (line 70):
    
    # Call to index(...): (line 70)
    # Processing the call arguments (line 70)
    # Getting the type of 't' (line 70)
    t_127443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 39), 't', False)
    # Processing the call keyword arguments (line 70)
    kwargs_127444 = {}
    # Getting the type of '_typecodes_by_elsize' (line 70)
    _typecodes_by_elsize_127441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 12), '_typecodes_by_elsize', False)
    # Obtaining the member 'index' of a type (line 70)
    index_127442 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 12), _typecodes_by_elsize_127441, 'index')
    # Calling index(args, kwargs) (line 70)
    index_call_result_127445 = invoke(stypy.reporting.localization.Localization(__file__, 70, 12), index_127442, *[t_127443], **kwargs_127444)
    
    # Assigning a type to the variable 'i' (line 70)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'i', index_call_result_127445)
    
    # Call to append(...): (line 71)
    # Processing the call arguments (line 71)
    
    # Obtaining an instance of the builtin type 'tuple' (line 71)
    tuple_127448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 18), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 71)
    # Adding element type (line 71)
    # Getting the type of 'i' (line 71)
    i_127449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 18), 'i', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 18), tuple_127448, i_127449)
    # Adding element type (line 71)
    # Getting the type of 't' (line 71)
    t_127450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 21), 't', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 18), tuple_127448, t_127450)
    
    # Processing the call keyword arguments (line 71)
    kwargs_127451 = {}
    # Getting the type of 'l' (line 71)
    l_127446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'l', False)
    # Obtaining the member 'append' of a type (line 71)
    append_127447 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 8), l_127446, 'append')
    # Calling append(args, kwargs) (line 71)
    append_call_result_127452 = invoke(stypy.reporting.localization.Localization(__file__, 71, 8), append_127447, *[tuple_127448], **kwargs_127451)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to sort(...): (line 72)
    # Processing the call keyword arguments (line 72)
    kwargs_127455 = {}
    # Getting the type of 'l' (line 72)
    l_127453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'l', False)
    # Obtaining the member 'sort' of a type (line 72)
    sort_127454 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 4), l_127453, 'sort')
    # Calling sort(args, kwargs) (line 72)
    sort_call_result_127456 = invoke(stypy.reporting.localization.Localization(__file__, 72, 4), sort_127454, *[], **kwargs_127455)
    
    
    # Obtaining the type of the subscript
    int_127457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 16), 'int')
    
    # Obtaining the type of the subscript
    int_127458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 13), 'int')
    # Getting the type of 'l' (line 73)
    l_127459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 11), 'l')
    # Obtaining the member '__getitem__' of a type (line 73)
    getitem___127460 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 11), l_127459, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 73)
    subscript_call_result_127461 = invoke(stypy.reporting.localization.Localization(__file__, 73, 11), getitem___127460, int_127458)
    
    # Obtaining the member '__getitem__' of a type (line 73)
    getitem___127462 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 11), subscript_call_result_127461, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 73)
    subscript_call_result_127463 = invoke(stypy.reporting.localization.Localization(__file__, 73, 11), getitem___127462, int_127457)
    
    # Assigning a type to the variable 'stypy_return_type' (line 73)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 4), 'stypy_return_type', subscript_call_result_127463)
    
    # ################# End of 'mintypecode(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'mintypecode' in the type store
    # Getting the type of 'stypy_return_type' (line 18)
    stypy_return_type_127464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_127464)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'mintypecode'
    return stypy_return_type_127464

# Assigning a type to the variable 'mintypecode' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'mintypecode', mintypecode)

@norecursion
def asfarray(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of '_nx' (line 75)
    _nx_127465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 22), '_nx')
    # Obtaining the member 'float_' of a type (line 75)
    float__127466 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 22), _nx_127465, 'float_')
    defaults = [float__127466]
    # Create a new context for function 'asfarray'
    module_type_store = module_type_store.open_function_context('asfarray', 75, 0, False)
    
    # Passed parameters checking function
    asfarray.stypy_localization = localization
    asfarray.stypy_type_of_self = None
    asfarray.stypy_type_store = module_type_store
    asfarray.stypy_function_name = 'asfarray'
    asfarray.stypy_param_names_list = ['a', 'dtype']
    asfarray.stypy_varargs_param_name = None
    asfarray.stypy_kwargs_param_name = None
    asfarray.stypy_call_defaults = defaults
    asfarray.stypy_call_varargs = varargs
    asfarray.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'asfarray', ['a', 'dtype'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'asfarray', localization, ['a', 'dtype'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'asfarray(...)' code ##################

    str_127467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, (-1)), 'str', "\n    Return an array converted to a float type.\n\n    Parameters\n    ----------\n    a : array_like\n        The input array.\n    dtype : str or dtype object, optional\n        Float type code to coerce input array `a`.  If `dtype` is one of the\n        'int' dtypes, it is replaced with float64.\n\n    Returns\n    -------\n    out : ndarray\n        The input `a` as a float ndarray.\n\n    Examples\n    --------\n    >>> np.asfarray([2, 3])\n    array([ 2.,  3.])\n    >>> np.asfarray([2, 3], dtype='float')\n    array([ 2.,  3.])\n    >>> np.asfarray([2, 3], dtype='int8')\n    array([ 2.,  3.])\n\n    ")
    
    # Assigning a Call to a Name (line 102):
    
    # Assigning a Call to a Name (line 102):
    
    # Call to obj2sctype(...): (line 102)
    # Processing the call arguments (line 102)
    # Getting the type of 'dtype' (line 102)
    dtype_127470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 27), 'dtype', False)
    # Processing the call keyword arguments (line 102)
    kwargs_127471 = {}
    # Getting the type of '_nx' (line 102)
    _nx_127468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 12), '_nx', False)
    # Obtaining the member 'obj2sctype' of a type (line 102)
    obj2sctype_127469 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 12), _nx_127468, 'obj2sctype')
    # Calling obj2sctype(args, kwargs) (line 102)
    obj2sctype_call_result_127472 = invoke(stypy.reporting.localization.Localization(__file__, 102, 12), obj2sctype_127469, *[dtype_127470], **kwargs_127471)
    
    # Assigning a type to the variable 'dtype' (line 102)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 4), 'dtype', obj2sctype_call_result_127472)
    
    
    
    # Call to issubclass(...): (line 103)
    # Processing the call arguments (line 103)
    # Getting the type of 'dtype' (line 103)
    dtype_127474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 22), 'dtype', False)
    # Getting the type of '_nx' (line 103)
    _nx_127475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 29), '_nx', False)
    # Obtaining the member 'inexact' of a type (line 103)
    inexact_127476 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 29), _nx_127475, 'inexact')
    # Processing the call keyword arguments (line 103)
    kwargs_127477 = {}
    # Getting the type of 'issubclass' (line 103)
    issubclass_127473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 11), 'issubclass', False)
    # Calling issubclass(args, kwargs) (line 103)
    issubclass_call_result_127478 = invoke(stypy.reporting.localization.Localization(__file__, 103, 11), issubclass_127473, *[dtype_127474, inexact_127476], **kwargs_127477)
    
    # Applying the 'not' unary operator (line 103)
    result_not__127479 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 7), 'not', issubclass_call_result_127478)
    
    # Testing the type of an if condition (line 103)
    if_condition_127480 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 103, 4), result_not__127479)
    # Assigning a type to the variable 'if_condition_127480' (line 103)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 4), 'if_condition_127480', if_condition_127480)
    # SSA begins for if statement (line 103)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Attribute to a Name (line 104):
    
    # Assigning a Attribute to a Name (line 104):
    # Getting the type of '_nx' (line 104)
    _nx_127481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 16), '_nx')
    # Obtaining the member 'float_' of a type (line 104)
    float__127482 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 16), _nx_127481, 'float_')
    # Assigning a type to the variable 'dtype' (line 104)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'dtype', float__127482)
    # SSA join for if statement (line 103)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to asarray(...): (line 105)
    # Processing the call arguments (line 105)
    # Getting the type of 'a' (line 105)
    a_127484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 19), 'a', False)
    # Processing the call keyword arguments (line 105)
    # Getting the type of 'dtype' (line 105)
    dtype_127485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 28), 'dtype', False)
    keyword_127486 = dtype_127485
    kwargs_127487 = {'dtype': keyword_127486}
    # Getting the type of 'asarray' (line 105)
    asarray_127483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 11), 'asarray', False)
    # Calling asarray(args, kwargs) (line 105)
    asarray_call_result_127488 = invoke(stypy.reporting.localization.Localization(__file__, 105, 11), asarray_127483, *[a_127484], **kwargs_127487)
    
    # Assigning a type to the variable 'stypy_return_type' (line 105)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 'stypy_return_type', asarray_call_result_127488)
    
    # ################# End of 'asfarray(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'asfarray' in the type store
    # Getting the type of 'stypy_return_type' (line 75)
    stypy_return_type_127489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_127489)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'asfarray'
    return stypy_return_type_127489

# Assigning a type to the variable 'asfarray' (line 75)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 0), 'asfarray', asfarray)

@norecursion
def real(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'real'
    module_type_store = module_type_store.open_function_context('real', 107, 0, False)
    
    # Passed parameters checking function
    real.stypy_localization = localization
    real.stypy_type_of_self = None
    real.stypy_type_store = module_type_store
    real.stypy_function_name = 'real'
    real.stypy_param_names_list = ['val']
    real.stypy_varargs_param_name = None
    real.stypy_kwargs_param_name = None
    real.stypy_call_defaults = defaults
    real.stypy_call_varargs = varargs
    real.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'real', ['val'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'real', localization, ['val'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'real(...)' code ##################

    str_127490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, (-1)), 'str', '\n    Return the real part of the elements of the array.\n\n    Parameters\n    ----------\n    val : array_like\n        Input array.\n\n    Returns\n    -------\n    out : ndarray\n        Output array. If `val` is real, the type of `val` is used for the\n        output.  If `val` has complex elements, the returned type is float.\n\n    See Also\n    --------\n    real_if_close, imag, angle\n\n    Examples\n    --------\n    >>> a = np.array([1+2j, 3+4j, 5+6j])\n    >>> a.real\n    array([ 1.,  3.,  5.])\n    >>> a.real = 9\n    >>> a\n    array([ 9.+2.j,  9.+4.j,  9.+6.j])\n    >>> a.real = np.array([9, 8, 7])\n    >>> a\n    array([ 9.+2.j,  8.+4.j,  7.+6.j])\n\n    ')
    
    # Call to asanyarray(...): (line 139)
    # Processing the call arguments (line 139)
    # Getting the type of 'val' (line 139)
    val_127492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 22), 'val', False)
    # Processing the call keyword arguments (line 139)
    kwargs_127493 = {}
    # Getting the type of 'asanyarray' (line 139)
    asanyarray_127491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 11), 'asanyarray', False)
    # Calling asanyarray(args, kwargs) (line 139)
    asanyarray_call_result_127494 = invoke(stypy.reporting.localization.Localization(__file__, 139, 11), asanyarray_127491, *[val_127492], **kwargs_127493)
    
    # Obtaining the member 'real' of a type (line 139)
    real_127495 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 11), asanyarray_call_result_127494, 'real')
    # Assigning a type to the variable 'stypy_return_type' (line 139)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 4), 'stypy_return_type', real_127495)
    
    # ################# End of 'real(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'real' in the type store
    # Getting the type of 'stypy_return_type' (line 107)
    stypy_return_type_127496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_127496)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'real'
    return stypy_return_type_127496

# Assigning a type to the variable 'real' (line 107)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 0), 'real', real)

@norecursion
def imag(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'imag'
    module_type_store = module_type_store.open_function_context('imag', 141, 0, False)
    
    # Passed parameters checking function
    imag.stypy_localization = localization
    imag.stypy_type_of_self = None
    imag.stypy_type_store = module_type_store
    imag.stypy_function_name = 'imag'
    imag.stypy_param_names_list = ['val']
    imag.stypy_varargs_param_name = None
    imag.stypy_kwargs_param_name = None
    imag.stypy_call_defaults = defaults
    imag.stypy_call_varargs = varargs
    imag.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'imag', ['val'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'imag', localization, ['val'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'imag(...)' code ##################

    str_127497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, (-1)), 'str', '\n    Return the imaginary part of the elements of the array.\n\n    Parameters\n    ----------\n    val : array_like\n        Input array.\n\n    Returns\n    -------\n    out : ndarray\n        Output array. If `val` is real, the type of `val` is used for the\n        output.  If `val` has complex elements, the returned type is float.\n\n    See Also\n    --------\n    real, angle, real_if_close\n\n    Examples\n    --------\n    >>> a = np.array([1+2j, 3+4j, 5+6j])\n    >>> a.imag\n    array([ 2.,  4.,  6.])\n    >>> a.imag = np.array([8, 10, 12])\n    >>> a\n    array([ 1. +8.j,  3.+10.j,  5.+12.j])\n\n    ')
    
    # Call to asanyarray(...): (line 170)
    # Processing the call arguments (line 170)
    # Getting the type of 'val' (line 170)
    val_127499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 22), 'val', False)
    # Processing the call keyword arguments (line 170)
    kwargs_127500 = {}
    # Getting the type of 'asanyarray' (line 170)
    asanyarray_127498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 11), 'asanyarray', False)
    # Calling asanyarray(args, kwargs) (line 170)
    asanyarray_call_result_127501 = invoke(stypy.reporting.localization.Localization(__file__, 170, 11), asanyarray_127498, *[val_127499], **kwargs_127500)
    
    # Obtaining the member 'imag' of a type (line 170)
    imag_127502 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 11), asanyarray_call_result_127501, 'imag')
    # Assigning a type to the variable 'stypy_return_type' (line 170)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 4), 'stypy_return_type', imag_127502)
    
    # ################# End of 'imag(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'imag' in the type store
    # Getting the type of 'stypy_return_type' (line 141)
    stypy_return_type_127503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_127503)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'imag'
    return stypy_return_type_127503

# Assigning a type to the variable 'imag' (line 141)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 0), 'imag', imag)

@norecursion
def iscomplex(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'iscomplex'
    module_type_store = module_type_store.open_function_context('iscomplex', 172, 0, False)
    
    # Passed parameters checking function
    iscomplex.stypy_localization = localization
    iscomplex.stypy_type_of_self = None
    iscomplex.stypy_type_store = module_type_store
    iscomplex.stypy_function_name = 'iscomplex'
    iscomplex.stypy_param_names_list = ['x']
    iscomplex.stypy_varargs_param_name = None
    iscomplex.stypy_kwargs_param_name = None
    iscomplex.stypy_call_defaults = defaults
    iscomplex.stypy_call_varargs = varargs
    iscomplex.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'iscomplex', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'iscomplex', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'iscomplex(...)' code ##################

    str_127504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, (-1)), 'str', '\n    Returns a bool array, where True if input element is complex.\n\n    What is tested is whether the input has a non-zero imaginary part, not if\n    the input type is complex.\n\n    Parameters\n    ----------\n    x : array_like\n        Input array.\n\n    Returns\n    -------\n    out : ndarray of bools\n        Output array.\n\n    See Also\n    --------\n    isreal\n    iscomplexobj : Return True if x is a complex type or an array of complex\n                   numbers.\n\n    Examples\n    --------\n    >>> np.iscomplex([1+1j, 1+0j, 4.5, 3, 2, 2j])\n    array([ True, False, False, False, False,  True], dtype=bool)\n\n    ')
    
    # Assigning a Call to a Name (line 201):
    
    # Assigning a Call to a Name (line 201):
    
    # Call to asanyarray(...): (line 201)
    # Processing the call arguments (line 201)
    # Getting the type of 'x' (line 201)
    x_127506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 20), 'x', False)
    # Processing the call keyword arguments (line 201)
    kwargs_127507 = {}
    # Getting the type of 'asanyarray' (line 201)
    asanyarray_127505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 9), 'asanyarray', False)
    # Calling asanyarray(args, kwargs) (line 201)
    asanyarray_call_result_127508 = invoke(stypy.reporting.localization.Localization(__file__, 201, 9), asanyarray_127505, *[x_127506], **kwargs_127507)
    
    # Assigning a type to the variable 'ax' (line 201)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 4), 'ax', asanyarray_call_result_127508)
    
    
    # Call to issubclass(...): (line 202)
    # Processing the call arguments (line 202)
    # Getting the type of 'ax' (line 202)
    ax_127510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 18), 'ax', False)
    # Obtaining the member 'dtype' of a type (line 202)
    dtype_127511 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 18), ax_127510, 'dtype')
    # Obtaining the member 'type' of a type (line 202)
    type_127512 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 18), dtype_127511, 'type')
    # Getting the type of '_nx' (line 202)
    _nx_127513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 33), '_nx', False)
    # Obtaining the member 'complexfloating' of a type (line 202)
    complexfloating_127514 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 33), _nx_127513, 'complexfloating')
    # Processing the call keyword arguments (line 202)
    kwargs_127515 = {}
    # Getting the type of 'issubclass' (line 202)
    issubclass_127509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 7), 'issubclass', False)
    # Calling issubclass(args, kwargs) (line 202)
    issubclass_call_result_127516 = invoke(stypy.reporting.localization.Localization(__file__, 202, 7), issubclass_127509, *[type_127512, complexfloating_127514], **kwargs_127515)
    
    # Testing the type of an if condition (line 202)
    if_condition_127517 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 202, 4), issubclass_call_result_127516)
    # Assigning a type to the variable 'if_condition_127517' (line 202)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 4), 'if_condition_127517', if_condition_127517)
    # SSA begins for if statement (line 202)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'ax' (line 203)
    ax_127518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 15), 'ax')
    # Obtaining the member 'imag' of a type (line 203)
    imag_127519 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 15), ax_127518, 'imag')
    int_127520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 26), 'int')
    # Applying the binary operator '!=' (line 203)
    result_ne_127521 = python_operator(stypy.reporting.localization.Localization(__file__, 203, 15), '!=', imag_127519, int_127520)
    
    # Assigning a type to the variable 'stypy_return_type' (line 203)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 8), 'stypy_return_type', result_ne_127521)
    # SSA join for if statement (line 202)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 204):
    
    # Assigning a Call to a Name (line 204):
    
    # Call to zeros(...): (line 204)
    # Processing the call arguments (line 204)
    # Getting the type of 'ax' (line 204)
    ax_127523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 16), 'ax', False)
    # Obtaining the member 'shape' of a type (line 204)
    shape_127524 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 16), ax_127523, 'shape')
    # Getting the type of 'bool' (line 204)
    bool_127525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 26), 'bool', False)
    # Processing the call keyword arguments (line 204)
    kwargs_127526 = {}
    # Getting the type of 'zeros' (line 204)
    zeros_127522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 10), 'zeros', False)
    # Calling zeros(args, kwargs) (line 204)
    zeros_call_result_127527 = invoke(stypy.reporting.localization.Localization(__file__, 204, 10), zeros_127522, *[shape_127524, bool_127525], **kwargs_127526)
    
    # Assigning a type to the variable 'res' (line 204)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 4), 'res', zeros_call_result_127527)
    
    # Getting the type of 'res' (line 205)
    res_127528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 12), 'res')
    # Applying the 'uadd' unary operator (line 205)
    result___pos___127529 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 11), 'uadd', res_127528)
    
    # Assigning a type to the variable 'stypy_return_type' (line 205)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 4), 'stypy_return_type', result___pos___127529)
    
    # ################# End of 'iscomplex(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'iscomplex' in the type store
    # Getting the type of 'stypy_return_type' (line 172)
    stypy_return_type_127530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_127530)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'iscomplex'
    return stypy_return_type_127530

# Assigning a type to the variable 'iscomplex' (line 172)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 0), 'iscomplex', iscomplex)

@norecursion
def isreal(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'isreal'
    module_type_store = module_type_store.open_function_context('isreal', 207, 0, False)
    
    # Passed parameters checking function
    isreal.stypy_localization = localization
    isreal.stypy_type_of_self = None
    isreal.stypy_type_store = module_type_store
    isreal.stypy_function_name = 'isreal'
    isreal.stypy_param_names_list = ['x']
    isreal.stypy_varargs_param_name = None
    isreal.stypy_kwargs_param_name = None
    isreal.stypy_call_defaults = defaults
    isreal.stypy_call_varargs = varargs
    isreal.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'isreal', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'isreal', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'isreal(...)' code ##################

    str_127531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, (-1)), 'str', '\n    Returns a bool array, where True if input element is real.\n\n    If element has complex type with zero complex part, the return value\n    for that element is True.\n\n    Parameters\n    ----------\n    x : array_like\n        Input array.\n\n    Returns\n    -------\n    out : ndarray, bool\n        Boolean array of same shape as `x`.\n\n    See Also\n    --------\n    iscomplex\n    isrealobj : Return True if x is not a complex type.\n\n    Examples\n    --------\n    >>> np.isreal([1+1j, 1+0j, 4.5, 3, 2, 2j])\n    array([False,  True,  True,  True,  True, False], dtype=bool)\n\n    ')
    
    
    # Call to imag(...): (line 235)
    # Processing the call arguments (line 235)
    # Getting the type of 'x' (line 235)
    x_127533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 16), 'x', False)
    # Processing the call keyword arguments (line 235)
    kwargs_127534 = {}
    # Getting the type of 'imag' (line 235)
    imag_127532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 11), 'imag', False)
    # Calling imag(args, kwargs) (line 235)
    imag_call_result_127535 = invoke(stypy.reporting.localization.Localization(__file__, 235, 11), imag_127532, *[x_127533], **kwargs_127534)
    
    int_127536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 22), 'int')
    # Applying the binary operator '==' (line 235)
    result_eq_127537 = python_operator(stypy.reporting.localization.Localization(__file__, 235, 11), '==', imag_call_result_127535, int_127536)
    
    # Assigning a type to the variable 'stypy_return_type' (line 235)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 4), 'stypy_return_type', result_eq_127537)
    
    # ################# End of 'isreal(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'isreal' in the type store
    # Getting the type of 'stypy_return_type' (line 207)
    stypy_return_type_127538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_127538)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'isreal'
    return stypy_return_type_127538

# Assigning a type to the variable 'isreal' (line 207)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 0), 'isreal', isreal)

@norecursion
def iscomplexobj(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'iscomplexobj'
    module_type_store = module_type_store.open_function_context('iscomplexobj', 237, 0, False)
    
    # Passed parameters checking function
    iscomplexobj.stypy_localization = localization
    iscomplexobj.stypy_type_of_self = None
    iscomplexobj.stypy_type_store = module_type_store
    iscomplexobj.stypy_function_name = 'iscomplexobj'
    iscomplexobj.stypy_param_names_list = ['x']
    iscomplexobj.stypy_varargs_param_name = None
    iscomplexobj.stypy_kwargs_param_name = None
    iscomplexobj.stypy_call_defaults = defaults
    iscomplexobj.stypy_call_varargs = varargs
    iscomplexobj.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'iscomplexobj', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'iscomplexobj', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'iscomplexobj(...)' code ##################

    str_127539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, (-1)), 'str', '\n    Check for a complex type or an array of complex numbers.\n\n    The type of the input is checked, not the value. Even if the input\n    has an imaginary part equal to zero, `iscomplexobj` evaluates to True.\n\n    Parameters\n    ----------\n    x : any\n        The input can be of any type and shape.\n\n    Returns\n    -------\n    iscomplexobj : bool\n        The return value, True if `x` is of a complex type or has at least\n        one complex element.\n\n    See Also\n    --------\n    isrealobj, iscomplex\n\n    Examples\n    --------\n    >>> np.iscomplexobj(1)\n    False\n    >>> np.iscomplexobj(1+0j)\n    True\n    >>> np.iscomplexobj([3, 1+0j, True])\n    True\n\n    ')
    
    # Call to issubclass(...): (line 269)
    # Processing the call arguments (line 269)
    
    # Call to asarray(...): (line 269)
    # Processing the call arguments (line 269)
    # Getting the type of 'x' (line 269)
    x_127542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 30), 'x', False)
    # Processing the call keyword arguments (line 269)
    kwargs_127543 = {}
    # Getting the type of 'asarray' (line 269)
    asarray_127541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 22), 'asarray', False)
    # Calling asarray(args, kwargs) (line 269)
    asarray_call_result_127544 = invoke(stypy.reporting.localization.Localization(__file__, 269, 22), asarray_127541, *[x_127542], **kwargs_127543)
    
    # Obtaining the member 'dtype' of a type (line 269)
    dtype_127545 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 22), asarray_call_result_127544, 'dtype')
    # Obtaining the member 'type' of a type (line 269)
    type_127546 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 22), dtype_127545, 'type')
    # Getting the type of '_nx' (line 269)
    _nx_127547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 45), '_nx', False)
    # Obtaining the member 'complexfloating' of a type (line 269)
    complexfloating_127548 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 45), _nx_127547, 'complexfloating')
    # Processing the call keyword arguments (line 269)
    kwargs_127549 = {}
    # Getting the type of 'issubclass' (line 269)
    issubclass_127540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 11), 'issubclass', False)
    # Calling issubclass(args, kwargs) (line 269)
    issubclass_call_result_127550 = invoke(stypy.reporting.localization.Localization(__file__, 269, 11), issubclass_127540, *[type_127546, complexfloating_127548], **kwargs_127549)
    
    # Assigning a type to the variable 'stypy_return_type' (line 269)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 4), 'stypy_return_type', issubclass_call_result_127550)
    
    # ################# End of 'iscomplexobj(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'iscomplexobj' in the type store
    # Getting the type of 'stypy_return_type' (line 237)
    stypy_return_type_127551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_127551)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'iscomplexobj'
    return stypy_return_type_127551

# Assigning a type to the variable 'iscomplexobj' (line 237)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 0), 'iscomplexobj', iscomplexobj)

@norecursion
def isrealobj(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'isrealobj'
    module_type_store = module_type_store.open_function_context('isrealobj', 271, 0, False)
    
    # Passed parameters checking function
    isrealobj.stypy_localization = localization
    isrealobj.stypy_type_of_self = None
    isrealobj.stypy_type_store = module_type_store
    isrealobj.stypy_function_name = 'isrealobj'
    isrealobj.stypy_param_names_list = ['x']
    isrealobj.stypy_varargs_param_name = None
    isrealobj.stypy_kwargs_param_name = None
    isrealobj.stypy_call_defaults = defaults
    isrealobj.stypy_call_varargs = varargs
    isrealobj.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'isrealobj', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'isrealobj', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'isrealobj(...)' code ##################

    str_127552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, (-1)), 'str', '\n    Return True if x is a not complex type or an array of complex numbers.\n\n    The type of the input is checked, not the value. So even if the input\n    has an imaginary part equal to zero, `isrealobj` evaluates to False\n    if the data type is complex.\n\n    Parameters\n    ----------\n    x : any\n        The input can be of any type and shape.\n\n    Returns\n    -------\n    y : bool\n        The return value, False if `x` is of a complex type.\n\n    See Also\n    --------\n    iscomplexobj, isreal\n\n    Examples\n    --------\n    >>> np.isrealobj(1)\n    True\n    >>> np.isrealobj(1+0j)\n    False\n    >>> np.isrealobj([3, 1+0j, True])\n    False\n\n    ')
    
    
    # Call to issubclass(...): (line 303)
    # Processing the call arguments (line 303)
    
    # Call to asarray(...): (line 303)
    # Processing the call arguments (line 303)
    # Getting the type of 'x' (line 303)
    x_127555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 34), 'x', False)
    # Processing the call keyword arguments (line 303)
    kwargs_127556 = {}
    # Getting the type of 'asarray' (line 303)
    asarray_127554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 26), 'asarray', False)
    # Calling asarray(args, kwargs) (line 303)
    asarray_call_result_127557 = invoke(stypy.reporting.localization.Localization(__file__, 303, 26), asarray_127554, *[x_127555], **kwargs_127556)
    
    # Obtaining the member 'dtype' of a type (line 303)
    dtype_127558 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 26), asarray_call_result_127557, 'dtype')
    # Obtaining the member 'type' of a type (line 303)
    type_127559 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 26), dtype_127558, 'type')
    # Getting the type of '_nx' (line 303)
    _nx_127560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 49), '_nx', False)
    # Obtaining the member 'complexfloating' of a type (line 303)
    complexfloating_127561 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 49), _nx_127560, 'complexfloating')
    # Processing the call keyword arguments (line 303)
    kwargs_127562 = {}
    # Getting the type of 'issubclass' (line 303)
    issubclass_127553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 15), 'issubclass', False)
    # Calling issubclass(args, kwargs) (line 303)
    issubclass_call_result_127563 = invoke(stypy.reporting.localization.Localization(__file__, 303, 15), issubclass_127553, *[type_127559, complexfloating_127561], **kwargs_127562)
    
    # Applying the 'not' unary operator (line 303)
    result_not__127564 = python_operator(stypy.reporting.localization.Localization(__file__, 303, 11), 'not', issubclass_call_result_127563)
    
    # Assigning a type to the variable 'stypy_return_type' (line 303)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 4), 'stypy_return_type', result_not__127564)
    
    # ################# End of 'isrealobj(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'isrealobj' in the type store
    # Getting the type of 'stypy_return_type' (line 271)
    stypy_return_type_127565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_127565)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'isrealobj'
    return stypy_return_type_127565

# Assigning a type to the variable 'isrealobj' (line 271)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 0), 'isrealobj', isrealobj)

@norecursion
def _getmaxmin(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_getmaxmin'
    module_type_store = module_type_store.open_function_context('_getmaxmin', 307, 0, False)
    
    # Passed parameters checking function
    _getmaxmin.stypy_localization = localization
    _getmaxmin.stypy_type_of_self = None
    _getmaxmin.stypy_type_store = module_type_store
    _getmaxmin.stypy_function_name = '_getmaxmin'
    _getmaxmin.stypy_param_names_list = ['t']
    _getmaxmin.stypy_varargs_param_name = None
    _getmaxmin.stypy_kwargs_param_name = None
    _getmaxmin.stypy_call_defaults = defaults
    _getmaxmin.stypy_call_varargs = varargs
    _getmaxmin.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_getmaxmin', ['t'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_getmaxmin', localization, ['t'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_getmaxmin(...)' code ##################

    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 308, 4))
    
    # 'from numpy.core import getlimits' statement (line 308)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/lib/')
    import_127566 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 308, 4), 'numpy.core')

    if (type(import_127566) is not StypyTypeError):

        if (import_127566 != 'pyd_module'):
            __import__(import_127566)
            sys_modules_127567 = sys.modules[import_127566]
            import_from_module(stypy.reporting.localization.Localization(__file__, 308, 4), 'numpy.core', sys_modules_127567.module_type_store, module_type_store, ['getlimits'])
            nest_module(stypy.reporting.localization.Localization(__file__, 308, 4), __file__, sys_modules_127567, sys_modules_127567.module_type_store, module_type_store)
        else:
            from numpy.core import getlimits

            import_from_module(stypy.reporting.localization.Localization(__file__, 308, 4), 'numpy.core', None, module_type_store, ['getlimits'], [getlimits])

    else:
        # Assigning a type to the variable 'numpy.core' (line 308)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 308, 4), 'numpy.core', import_127566)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/lib/')
    
    
    # Assigning a Call to a Name (line 309):
    
    # Assigning a Call to a Name (line 309):
    
    # Call to finfo(...): (line 309)
    # Processing the call arguments (line 309)
    # Getting the type of 't' (line 309)
    t_127570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 24), 't', False)
    # Processing the call keyword arguments (line 309)
    kwargs_127571 = {}
    # Getting the type of 'getlimits' (line 309)
    getlimits_127568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 8), 'getlimits', False)
    # Obtaining the member 'finfo' of a type (line 309)
    finfo_127569 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 8), getlimits_127568, 'finfo')
    # Calling finfo(args, kwargs) (line 309)
    finfo_call_result_127572 = invoke(stypy.reporting.localization.Localization(__file__, 309, 8), finfo_127569, *[t_127570], **kwargs_127571)
    
    # Assigning a type to the variable 'f' (line 309)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 4), 'f', finfo_call_result_127572)
    
    # Obtaining an instance of the builtin type 'tuple' (line 310)
    tuple_127573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 310)
    # Adding element type (line 310)
    # Getting the type of 'f' (line 310)
    f_127574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 11), 'f')
    # Obtaining the member 'max' of a type (line 310)
    max_127575 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 11), f_127574, 'max')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 310, 11), tuple_127573, max_127575)
    # Adding element type (line 310)
    # Getting the type of 'f' (line 310)
    f_127576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 18), 'f')
    # Obtaining the member 'min' of a type (line 310)
    min_127577 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 18), f_127576, 'min')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 310, 11), tuple_127573, min_127577)
    
    # Assigning a type to the variable 'stypy_return_type' (line 310)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 4), 'stypy_return_type', tuple_127573)
    
    # ################# End of '_getmaxmin(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_getmaxmin' in the type store
    # Getting the type of 'stypy_return_type' (line 307)
    stypy_return_type_127578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_127578)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_getmaxmin'
    return stypy_return_type_127578

# Assigning a type to the variable '_getmaxmin' (line 307)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 0), '_getmaxmin', _getmaxmin)

@norecursion
def nan_to_num(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'nan_to_num'
    module_type_store = module_type_store.open_function_context('nan_to_num', 312, 0, False)
    
    # Passed parameters checking function
    nan_to_num.stypy_localization = localization
    nan_to_num.stypy_type_of_self = None
    nan_to_num.stypy_type_store = module_type_store
    nan_to_num.stypy_function_name = 'nan_to_num'
    nan_to_num.stypy_param_names_list = ['x']
    nan_to_num.stypy_varargs_param_name = None
    nan_to_num.stypy_kwargs_param_name = None
    nan_to_num.stypy_call_defaults = defaults
    nan_to_num.stypy_call_varargs = varargs
    nan_to_num.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'nan_to_num', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'nan_to_num', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'nan_to_num(...)' code ##################

    str_127579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 357, (-1)), 'str', '\n    Replace nan with zero and inf with finite numbers.\n\n    Returns an array or scalar replacing Not a Number (NaN) with zero,\n    (positive) infinity with a very large number and negative infinity\n    with a very small (or negative) number.\n\n    Parameters\n    ----------\n    x : array_like\n        Input data.\n\n    Returns\n    -------\n    out : ndarray\n        New Array with the same shape as `x` and dtype of the element in\n        `x`  with the greatest precision. If `x` is inexact, then NaN is\n        replaced by zero, and infinity (-infinity) is replaced by the\n        largest (smallest or most negative) floating point value that fits\n        in the output dtype. If `x` is not inexact, then a copy of `x` is\n        returned.\n\n    See Also\n    --------\n    isinf : Shows which elements are negative or negative infinity.\n    isneginf : Shows which elements are negative infinity.\n    isposinf : Shows which elements are positive infinity.\n    isnan : Shows which elements are Not a Number (NaN).\n    isfinite : Shows which elements are finite (not NaN, not infinity)\n\n    Notes\n    -----\n    Numpy uses the IEEE Standard for Binary Floating-Point for Arithmetic\n    (IEEE 754). This means that Not a Number is not equivalent to infinity.\n\n\n    Examples\n    --------\n    >>> np.set_printoptions(precision=8)\n    >>> x = np.array([np.inf, -np.inf, np.nan, -128, 128])\n    >>> np.nan_to_num(x)\n    array([  1.79769313e+308,  -1.79769313e+308,   0.00000000e+000,\n            -1.28000000e+002,   1.28000000e+002])\n\n    ')
    
    # Assigning a Call to a Name (line 358):
    
    # Assigning a Call to a Name (line 358):
    
    # Call to array(...): (line 358)
    # Processing the call arguments (line 358)
    # Getting the type of 'x' (line 358)
    x_127582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 18), 'x', False)
    # Processing the call keyword arguments (line 358)
    # Getting the type of 'True' (line 358)
    True_127583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 27), 'True', False)
    keyword_127584 = True_127583
    kwargs_127585 = {'subok': keyword_127584}
    # Getting the type of '_nx' (line 358)
    _nx_127580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 8), '_nx', False)
    # Obtaining the member 'array' of a type (line 358)
    array_127581 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 8), _nx_127580, 'array')
    # Calling array(args, kwargs) (line 358)
    array_call_result_127586 = invoke(stypy.reporting.localization.Localization(__file__, 358, 8), array_127581, *[x_127582], **kwargs_127585)
    
    # Assigning a type to the variable 'x' (line 358)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 358, 4), 'x', array_call_result_127586)
    
    # Assigning a Attribute to a Name (line 359):
    
    # Assigning a Attribute to a Name (line 359):
    # Getting the type of 'x' (line 359)
    x_127587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 12), 'x')
    # Obtaining the member 'dtype' of a type (line 359)
    dtype_127588 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 12), x_127587, 'dtype')
    # Obtaining the member 'type' of a type (line 359)
    type_127589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 12), dtype_127588, 'type')
    # Assigning a type to the variable 'xtype' (line 359)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 359, 4), 'xtype', type_127589)
    
    
    
    # Call to issubclass(...): (line 360)
    # Processing the call arguments (line 360)
    # Getting the type of 'xtype' (line 360)
    xtype_127591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 22), 'xtype', False)
    # Getting the type of '_nx' (line 360)
    _nx_127592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 29), '_nx', False)
    # Obtaining the member 'inexact' of a type (line 360)
    inexact_127593 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 29), _nx_127592, 'inexact')
    # Processing the call keyword arguments (line 360)
    kwargs_127594 = {}
    # Getting the type of 'issubclass' (line 360)
    issubclass_127590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 11), 'issubclass', False)
    # Calling issubclass(args, kwargs) (line 360)
    issubclass_call_result_127595 = invoke(stypy.reporting.localization.Localization(__file__, 360, 11), issubclass_127590, *[xtype_127591, inexact_127593], **kwargs_127594)
    
    # Applying the 'not' unary operator (line 360)
    result_not__127596 = python_operator(stypy.reporting.localization.Localization(__file__, 360, 7), 'not', issubclass_call_result_127595)
    
    # Testing the type of an if condition (line 360)
    if_condition_127597 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 360, 4), result_not__127596)
    # Assigning a type to the variable 'if_condition_127597' (line 360)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 360, 4), 'if_condition_127597', if_condition_127597)
    # SSA begins for if statement (line 360)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'x' (line 361)
    x_127598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 15), 'x')
    # Assigning a type to the variable 'stypy_return_type' (line 361)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 361, 8), 'stypy_return_type', x_127598)
    # SSA join for if statement (line 360)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 363):
    
    # Assigning a Call to a Name (line 363):
    
    # Call to issubclass(...): (line 363)
    # Processing the call arguments (line 363)
    # Getting the type of 'xtype' (line 363)
    xtype_127600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 27), 'xtype', False)
    # Getting the type of '_nx' (line 363)
    _nx_127601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 34), '_nx', False)
    # Obtaining the member 'complexfloating' of a type (line 363)
    complexfloating_127602 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 34), _nx_127601, 'complexfloating')
    # Processing the call keyword arguments (line 363)
    kwargs_127603 = {}
    # Getting the type of 'issubclass' (line 363)
    issubclass_127599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 16), 'issubclass', False)
    # Calling issubclass(args, kwargs) (line 363)
    issubclass_call_result_127604 = invoke(stypy.reporting.localization.Localization(__file__, 363, 16), issubclass_127599, *[xtype_127600, complexfloating_127602], **kwargs_127603)
    
    # Assigning a type to the variable 'iscomplex' (line 363)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 4), 'iscomplex', issubclass_call_result_127604)
    
    # Assigning a Compare to a Name (line 364):
    
    # Assigning a Compare to a Name (line 364):
    
    # Getting the type of 'x' (line 364)
    x_127605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 16), 'x')
    # Obtaining the member 'ndim' of a type (line 364)
    ndim_127606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 364, 16), x_127605, 'ndim')
    int_127607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 26), 'int')
    # Applying the binary operator '==' (line 364)
    result_eq_127608 = python_operator(stypy.reporting.localization.Localization(__file__, 364, 16), '==', ndim_127606, int_127607)
    
    # Assigning a type to the variable 'isscalar' (line 364)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 364, 4), 'isscalar', result_eq_127608)
    
    # Assigning a IfExp to a Name (line 366):
    
    # Assigning a IfExp to a Name (line 366):
    
    # Getting the type of 'isscalar' (line 366)
    isscalar_127609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 19), 'isscalar')
    # Testing the type of an if expression (line 366)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 366, 8), isscalar_127609)
    # SSA begins for if expression (line 366)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    
    # Obtaining the type of the subscript
    # Getting the type of 'None' (line 366)
    None_127610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 10), 'None')
    # Getting the type of 'x' (line 366)
    x_127611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 8), 'x')
    # Obtaining the member '__getitem__' of a type (line 366)
    getitem___127612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 366, 8), x_127611, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 366)
    subscript_call_result_127613 = invoke(stypy.reporting.localization.Localization(__file__, 366, 8), getitem___127612, None_127610)
    
    # SSA branch for the else part of an if expression (line 366)
    module_type_store.open_ssa_branch('if expression else')
    # Getting the type of 'x' (line 366)
    x_127614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 33), 'x')
    # SSA join for if expression (line 366)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_127615 = union_type.UnionType.add(subscript_call_result_127613, x_127614)
    
    # Assigning a type to the variable 'x' (line 366)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 366, 4), 'x', if_exp_127615)
    
    # Assigning a IfExp to a Name (line 367):
    
    # Assigning a IfExp to a Name (line 367):
    
    # Getting the type of 'iscomplex' (line 367)
    iscomplex_127616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 31), 'iscomplex')
    # Testing the type of an if expression (line 367)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 367, 11), iscomplex_127616)
    # SSA begins for if expression (line 367)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    
    # Obtaining an instance of the builtin type 'tuple' (line 367)
    tuple_127617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 367, 12), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 367)
    # Adding element type (line 367)
    # Getting the type of 'x' (line 367)
    x_127618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 12), 'x')
    # Obtaining the member 'real' of a type (line 367)
    real_127619 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 367, 12), x_127618, 'real')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 367, 12), tuple_127617, real_127619)
    # Adding element type (line 367)
    # Getting the type of 'x' (line 367)
    x_127620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 20), 'x')
    # Obtaining the member 'imag' of a type (line 367)
    imag_127621 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 367, 20), x_127620, 'imag')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 367, 12), tuple_127617, imag_127621)
    
    # SSA branch for the else part of an if expression (line 367)
    module_type_store.open_ssa_branch('if expression else')
    
    # Obtaining an instance of the builtin type 'tuple' (line 367)
    tuple_127622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 367, 47), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 367)
    # Adding element type (line 367)
    # Getting the type of 'x' (line 367)
    x_127623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 47), 'x')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 367, 47), tuple_127622, x_127623)
    
    # SSA join for if expression (line 367)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_127624 = union_type.UnionType.add(tuple_127617, tuple_127622)
    
    # Assigning a type to the variable 'dest' (line 367)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 367, 4), 'dest', if_exp_127624)
    
    # Assigning a Call to a Tuple (line 368):
    
    # Assigning a Call to a Name:
    
    # Call to _getmaxmin(...): (line 368)
    # Processing the call arguments (line 368)
    # Getting the type of 'x' (line 368)
    x_127626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 28), 'x', False)
    # Obtaining the member 'real' of a type (line 368)
    real_127627 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 28), x_127626, 'real')
    # Obtaining the member 'dtype' of a type (line 368)
    dtype_127628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 28), real_127627, 'dtype')
    # Processing the call keyword arguments (line 368)
    kwargs_127629 = {}
    # Getting the type of '_getmaxmin' (line 368)
    _getmaxmin_127625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 17), '_getmaxmin', False)
    # Calling _getmaxmin(args, kwargs) (line 368)
    _getmaxmin_call_result_127630 = invoke(stypy.reporting.localization.Localization(__file__, 368, 17), _getmaxmin_127625, *[dtype_127628], **kwargs_127629)
    
    # Assigning a type to the variable 'call_assignment_127373' (line 368)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 4), 'call_assignment_127373', _getmaxmin_call_result_127630)
    
    # Assigning a Call to a Name (line 368):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_127633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, 4), 'int')
    # Processing the call keyword arguments
    kwargs_127634 = {}
    # Getting the type of 'call_assignment_127373' (line 368)
    call_assignment_127373_127631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 4), 'call_assignment_127373', False)
    # Obtaining the member '__getitem__' of a type (line 368)
    getitem___127632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 4), call_assignment_127373_127631, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_127635 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___127632, *[int_127633], **kwargs_127634)
    
    # Assigning a type to the variable 'call_assignment_127374' (line 368)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 4), 'call_assignment_127374', getitem___call_result_127635)
    
    # Assigning a Name to a Name (line 368):
    # Getting the type of 'call_assignment_127374' (line 368)
    call_assignment_127374_127636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 4), 'call_assignment_127374')
    # Assigning a type to the variable 'maxf' (line 368)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 4), 'maxf', call_assignment_127374_127636)
    
    # Assigning a Call to a Name (line 368):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_127639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, 4), 'int')
    # Processing the call keyword arguments
    kwargs_127640 = {}
    # Getting the type of 'call_assignment_127373' (line 368)
    call_assignment_127373_127637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 4), 'call_assignment_127373', False)
    # Obtaining the member '__getitem__' of a type (line 368)
    getitem___127638 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 4), call_assignment_127373_127637, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_127641 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___127638, *[int_127639], **kwargs_127640)
    
    # Assigning a type to the variable 'call_assignment_127375' (line 368)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 4), 'call_assignment_127375', getitem___call_result_127641)
    
    # Assigning a Name to a Name (line 368):
    # Getting the type of 'call_assignment_127375' (line 368)
    call_assignment_127375_127642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 4), 'call_assignment_127375')
    # Assigning a type to the variable 'minf' (line 368)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 10), 'minf', call_assignment_127375_127642)
    
    # Getting the type of 'dest' (line 369)
    dest_127643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 13), 'dest')
    # Testing the type of a for loop iterable (line 369)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 369, 4), dest_127643)
    # Getting the type of the for loop variable (line 369)
    for_loop_var_127644 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 369, 4), dest_127643)
    # Assigning a type to the variable 'd' (line 369)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 369, 4), 'd', for_loop_var_127644)
    # SSA begins for a for statement (line 369)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to copyto(...): (line 370)
    # Processing the call arguments (line 370)
    # Getting the type of 'd' (line 370)
    d_127647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 19), 'd', False)
    float_127648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 22), 'float')
    # Processing the call keyword arguments (line 370)
    
    # Call to isnan(...): (line 370)
    # Processing the call arguments (line 370)
    # Getting the type of 'd' (line 370)
    d_127650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 39), 'd', False)
    # Processing the call keyword arguments (line 370)
    kwargs_127651 = {}
    # Getting the type of 'isnan' (line 370)
    isnan_127649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 33), 'isnan', False)
    # Calling isnan(args, kwargs) (line 370)
    isnan_call_result_127652 = invoke(stypy.reporting.localization.Localization(__file__, 370, 33), isnan_127649, *[d_127650], **kwargs_127651)
    
    keyword_127653 = isnan_call_result_127652
    kwargs_127654 = {'where': keyword_127653}
    # Getting the type of '_nx' (line 370)
    _nx_127645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 8), '_nx', False)
    # Obtaining the member 'copyto' of a type (line 370)
    copyto_127646 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 8), _nx_127645, 'copyto')
    # Calling copyto(args, kwargs) (line 370)
    copyto_call_result_127655 = invoke(stypy.reporting.localization.Localization(__file__, 370, 8), copyto_127646, *[d_127647, float_127648], **kwargs_127654)
    
    
    # Call to copyto(...): (line 371)
    # Processing the call arguments (line 371)
    # Getting the type of 'd' (line 371)
    d_127658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 19), 'd', False)
    # Getting the type of 'maxf' (line 371)
    maxf_127659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 22), 'maxf', False)
    # Processing the call keyword arguments (line 371)
    
    # Call to isposinf(...): (line 371)
    # Processing the call arguments (line 371)
    # Getting the type of 'd' (line 371)
    d_127661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 43), 'd', False)
    # Processing the call keyword arguments (line 371)
    kwargs_127662 = {}
    # Getting the type of 'isposinf' (line 371)
    isposinf_127660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 34), 'isposinf', False)
    # Calling isposinf(args, kwargs) (line 371)
    isposinf_call_result_127663 = invoke(stypy.reporting.localization.Localization(__file__, 371, 34), isposinf_127660, *[d_127661], **kwargs_127662)
    
    keyword_127664 = isposinf_call_result_127663
    kwargs_127665 = {'where': keyword_127664}
    # Getting the type of '_nx' (line 371)
    _nx_127656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 8), '_nx', False)
    # Obtaining the member 'copyto' of a type (line 371)
    copyto_127657 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 371, 8), _nx_127656, 'copyto')
    # Calling copyto(args, kwargs) (line 371)
    copyto_call_result_127666 = invoke(stypy.reporting.localization.Localization(__file__, 371, 8), copyto_127657, *[d_127658, maxf_127659], **kwargs_127665)
    
    
    # Call to copyto(...): (line 372)
    # Processing the call arguments (line 372)
    # Getting the type of 'd' (line 372)
    d_127669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 19), 'd', False)
    # Getting the type of 'minf' (line 372)
    minf_127670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 22), 'minf', False)
    # Processing the call keyword arguments (line 372)
    
    # Call to isneginf(...): (line 372)
    # Processing the call arguments (line 372)
    # Getting the type of 'd' (line 372)
    d_127672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 43), 'd', False)
    # Processing the call keyword arguments (line 372)
    kwargs_127673 = {}
    # Getting the type of 'isneginf' (line 372)
    isneginf_127671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 34), 'isneginf', False)
    # Calling isneginf(args, kwargs) (line 372)
    isneginf_call_result_127674 = invoke(stypy.reporting.localization.Localization(__file__, 372, 34), isneginf_127671, *[d_127672], **kwargs_127673)
    
    keyword_127675 = isneginf_call_result_127674
    kwargs_127676 = {'where': keyword_127675}
    # Getting the type of '_nx' (line 372)
    _nx_127667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 8), '_nx', False)
    # Obtaining the member 'copyto' of a type (line 372)
    copyto_127668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 8), _nx_127667, 'copyto')
    # Calling copyto(args, kwargs) (line 372)
    copyto_call_result_127677 = invoke(stypy.reporting.localization.Localization(__file__, 372, 8), copyto_127668, *[d_127669, minf_127670], **kwargs_127676)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'isscalar' (line 373)
    isscalar_127678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 19), 'isscalar')
    # Testing the type of an if expression (line 373)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 373, 11), isscalar_127678)
    # SSA begins for if expression (line 373)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    
    # Obtaining the type of the subscript
    int_127679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 13), 'int')
    # Getting the type of 'x' (line 373)
    x_127680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 11), 'x')
    # Obtaining the member '__getitem__' of a type (line 373)
    getitem___127681 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 11), x_127680, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 373)
    subscript_call_result_127682 = invoke(stypy.reporting.localization.Localization(__file__, 373, 11), getitem___127681, int_127679)
    
    # SSA branch for the else part of an if expression (line 373)
    module_type_store.open_ssa_branch('if expression else')
    # Getting the type of 'x' (line 373)
    x_127683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 33), 'x')
    # SSA join for if expression (line 373)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_127684 = union_type.UnionType.add(subscript_call_result_127682, x_127683)
    
    # Assigning a type to the variable 'stypy_return_type' (line 373)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 373, 4), 'stypy_return_type', if_exp_127684)
    
    # ################# End of 'nan_to_num(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'nan_to_num' in the type store
    # Getting the type of 'stypy_return_type' (line 312)
    stypy_return_type_127685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_127685)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'nan_to_num'
    return stypy_return_type_127685

# Assigning a type to the variable 'nan_to_num' (line 312)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 0), 'nan_to_num', nan_to_num)

@norecursion
def real_if_close(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_127686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 377, 24), 'int')
    defaults = [int_127686]
    # Create a new context for function 'real_if_close'
    module_type_store = module_type_store.open_function_context('real_if_close', 377, 0, False)
    
    # Passed parameters checking function
    real_if_close.stypy_localization = localization
    real_if_close.stypy_type_of_self = None
    real_if_close.stypy_type_store = module_type_store
    real_if_close.stypy_function_name = 'real_if_close'
    real_if_close.stypy_param_names_list = ['a', 'tol']
    real_if_close.stypy_varargs_param_name = None
    real_if_close.stypy_kwargs_param_name = None
    real_if_close.stypy_call_defaults = defaults
    real_if_close.stypy_call_varargs = varargs
    real_if_close.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'real_if_close', ['a', 'tol'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'real_if_close', localization, ['a', 'tol'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'real_if_close(...)' code ##################

    str_127687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 419, (-1)), 'str', '\n    If complex input returns a real array if complex parts are close to zero.\n\n    "Close to zero" is defined as `tol` * (machine epsilon of the type for\n    `a`).\n\n    Parameters\n    ----------\n    a : array_like\n        Input array.\n    tol : float\n        Tolerance in machine epsilons for the complex part of the elements\n        in the array.\n\n    Returns\n    -------\n    out : ndarray\n        If `a` is real, the type of `a` is used for the output.  If `a`\n        has complex elements, the returned type is float.\n\n    See Also\n    --------\n    real, imag, angle\n\n    Notes\n    -----\n    Machine epsilon varies from machine to machine and between data types\n    but Python floats on most platforms have a machine epsilon equal to\n    2.2204460492503131e-16.  You can use \'np.finfo(np.float).eps\' to print\n    out the machine epsilon for floats.\n\n    Examples\n    --------\n    >>> np.finfo(np.float).eps\n    2.2204460492503131e-16\n\n    >>> np.real_if_close([2.1 + 4e-14j], tol=1000)\n    array([ 2.1])\n    >>> np.real_if_close([2.1 + 4e-13j], tol=1000)\n    array([ 2.1 +4.00000000e-13j])\n\n    ')
    
    # Assigning a Call to a Name (line 420):
    
    # Assigning a Call to a Name (line 420):
    
    # Call to asanyarray(...): (line 420)
    # Processing the call arguments (line 420)
    # Getting the type of 'a' (line 420)
    a_127689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 19), 'a', False)
    # Processing the call keyword arguments (line 420)
    kwargs_127690 = {}
    # Getting the type of 'asanyarray' (line 420)
    asanyarray_127688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 8), 'asanyarray', False)
    # Calling asanyarray(args, kwargs) (line 420)
    asanyarray_call_result_127691 = invoke(stypy.reporting.localization.Localization(__file__, 420, 8), asanyarray_127688, *[a_127689], **kwargs_127690)
    
    # Assigning a type to the variable 'a' (line 420)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 420, 4), 'a', asanyarray_call_result_127691)
    
    
    
    # Call to issubclass(...): (line 421)
    # Processing the call arguments (line 421)
    # Getting the type of 'a' (line 421)
    a_127693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 22), 'a', False)
    # Obtaining the member 'dtype' of a type (line 421)
    dtype_127694 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 421, 22), a_127693, 'dtype')
    # Obtaining the member 'type' of a type (line 421)
    type_127695 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 421, 22), dtype_127694, 'type')
    # Getting the type of '_nx' (line 421)
    _nx_127696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 36), '_nx', False)
    # Obtaining the member 'complexfloating' of a type (line 421)
    complexfloating_127697 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 421, 36), _nx_127696, 'complexfloating')
    # Processing the call keyword arguments (line 421)
    kwargs_127698 = {}
    # Getting the type of 'issubclass' (line 421)
    issubclass_127692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 11), 'issubclass', False)
    # Calling issubclass(args, kwargs) (line 421)
    issubclass_call_result_127699 = invoke(stypy.reporting.localization.Localization(__file__, 421, 11), issubclass_127692, *[type_127695, complexfloating_127697], **kwargs_127698)
    
    # Applying the 'not' unary operator (line 421)
    result_not__127700 = python_operator(stypy.reporting.localization.Localization(__file__, 421, 7), 'not', issubclass_call_result_127699)
    
    # Testing the type of an if condition (line 421)
    if_condition_127701 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 421, 4), result_not__127700)
    # Assigning a type to the variable 'if_condition_127701' (line 421)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 421, 4), 'if_condition_127701', if_condition_127701)
    # SSA begins for if statement (line 421)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'a' (line 422)
    a_127702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 15), 'a')
    # Assigning a type to the variable 'stypy_return_type' (line 422)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 422, 8), 'stypy_return_type', a_127702)
    # SSA join for if statement (line 421)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'tol' (line 423)
    tol_127703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 7), 'tol')
    int_127704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 13), 'int')
    # Applying the binary operator '>' (line 423)
    result_gt_127705 = python_operator(stypy.reporting.localization.Localization(__file__, 423, 7), '>', tol_127703, int_127704)
    
    # Testing the type of an if condition (line 423)
    if_condition_127706 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 423, 4), result_gt_127705)
    # Assigning a type to the variable 'if_condition_127706' (line 423)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 4), 'if_condition_127706', if_condition_127706)
    # SSA begins for if statement (line 423)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 424, 8))
    
    # 'from numpy.core import getlimits' statement (line 424)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/lib/')
    import_127707 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 424, 8), 'numpy.core')

    if (type(import_127707) is not StypyTypeError):

        if (import_127707 != 'pyd_module'):
            __import__(import_127707)
            sys_modules_127708 = sys.modules[import_127707]
            import_from_module(stypy.reporting.localization.Localization(__file__, 424, 8), 'numpy.core', sys_modules_127708.module_type_store, module_type_store, ['getlimits'])
            nest_module(stypy.reporting.localization.Localization(__file__, 424, 8), __file__, sys_modules_127708, sys_modules_127708.module_type_store, module_type_store)
        else:
            from numpy.core import getlimits

            import_from_module(stypy.reporting.localization.Localization(__file__, 424, 8), 'numpy.core', None, module_type_store, ['getlimits'], [getlimits])

    else:
        # Assigning a type to the variable 'numpy.core' (line 424)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 424, 8), 'numpy.core', import_127707)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/lib/')
    
    
    # Assigning a Call to a Name (line 425):
    
    # Assigning a Call to a Name (line 425):
    
    # Call to finfo(...): (line 425)
    # Processing the call arguments (line 425)
    # Getting the type of 'a' (line 425)
    a_127711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 28), 'a', False)
    # Obtaining the member 'dtype' of a type (line 425)
    dtype_127712 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 425, 28), a_127711, 'dtype')
    # Obtaining the member 'type' of a type (line 425)
    type_127713 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 425, 28), dtype_127712, 'type')
    # Processing the call keyword arguments (line 425)
    kwargs_127714 = {}
    # Getting the type of 'getlimits' (line 425)
    getlimits_127709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 12), 'getlimits', False)
    # Obtaining the member 'finfo' of a type (line 425)
    finfo_127710 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 425, 12), getlimits_127709, 'finfo')
    # Calling finfo(args, kwargs) (line 425)
    finfo_call_result_127715 = invoke(stypy.reporting.localization.Localization(__file__, 425, 12), finfo_127710, *[type_127713], **kwargs_127714)
    
    # Assigning a type to the variable 'f' (line 425)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 425, 8), 'f', finfo_call_result_127715)
    
    # Assigning a BinOp to a Name (line 426):
    
    # Assigning a BinOp to a Name (line 426):
    # Getting the type of 'f' (line 426)
    f_127716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 14), 'f')
    # Obtaining the member 'eps' of a type (line 426)
    eps_127717 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 426, 14), f_127716, 'eps')
    # Getting the type of 'tol' (line 426)
    tol_127718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 22), 'tol')
    # Applying the binary operator '*' (line 426)
    result_mul_127719 = python_operator(stypy.reporting.localization.Localization(__file__, 426, 14), '*', eps_127717, tol_127718)
    
    # Assigning a type to the variable 'tol' (line 426)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 426, 8), 'tol', result_mul_127719)
    # SSA join for if statement (line 423)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to allclose(...): (line 427)
    # Processing the call arguments (line 427)
    # Getting the type of 'a' (line 427)
    a_127722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 20), 'a', False)
    # Obtaining the member 'imag' of a type (line 427)
    imag_127723 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 427, 20), a_127722, 'imag')
    int_127724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 427, 28), 'int')
    # Processing the call keyword arguments (line 427)
    # Getting the type of 'tol' (line 427)
    tol_127725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 36), 'tol', False)
    keyword_127726 = tol_127725
    kwargs_127727 = {'atol': keyword_127726}
    # Getting the type of '_nx' (line 427)
    _nx_127720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 7), '_nx', False)
    # Obtaining the member 'allclose' of a type (line 427)
    allclose_127721 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 427, 7), _nx_127720, 'allclose')
    # Calling allclose(args, kwargs) (line 427)
    allclose_call_result_127728 = invoke(stypy.reporting.localization.Localization(__file__, 427, 7), allclose_127721, *[imag_127723, int_127724], **kwargs_127727)
    
    # Testing the type of an if condition (line 427)
    if_condition_127729 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 427, 4), allclose_call_result_127728)
    # Assigning a type to the variable 'if_condition_127729' (line 427)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 427, 4), 'if_condition_127729', if_condition_127729)
    # SSA begins for if statement (line 427)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Attribute to a Name (line 428):
    
    # Assigning a Attribute to a Name (line 428):
    # Getting the type of 'a' (line 428)
    a_127730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 12), 'a')
    # Obtaining the member 'real' of a type (line 428)
    real_127731 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 428, 12), a_127730, 'real')
    # Assigning a type to the variable 'a' (line 428)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 428, 8), 'a', real_127731)
    # SSA join for if statement (line 427)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'a' (line 429)
    a_127732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 11), 'a')
    # Assigning a type to the variable 'stypy_return_type' (line 429)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 429, 4), 'stypy_return_type', a_127732)
    
    # ################# End of 'real_if_close(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'real_if_close' in the type store
    # Getting the type of 'stypy_return_type' (line 377)
    stypy_return_type_127733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_127733)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'real_if_close'
    return stypy_return_type_127733

# Assigning a type to the variable 'real_if_close' (line 377)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 377, 0), 'real_if_close', real_if_close)

@norecursion
def asscalar(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'asscalar'
    module_type_store = module_type_store.open_function_context('asscalar', 432, 0, False)
    
    # Passed parameters checking function
    asscalar.stypy_localization = localization
    asscalar.stypy_type_of_self = None
    asscalar.stypy_type_store = module_type_store
    asscalar.stypy_function_name = 'asscalar'
    asscalar.stypy_param_names_list = ['a']
    asscalar.stypy_varargs_param_name = None
    asscalar.stypy_kwargs_param_name = None
    asscalar.stypy_call_defaults = defaults
    asscalar.stypy_call_varargs = varargs
    asscalar.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'asscalar', ['a'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'asscalar', localization, ['a'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'asscalar(...)' code ##################

    str_127734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 452, (-1)), 'str', "\n    Convert an array of size 1 to its scalar equivalent.\n\n    Parameters\n    ----------\n    a : ndarray\n        Input array of size 1.\n\n    Returns\n    -------\n    out : scalar\n        Scalar representation of `a`. The output data type is the same type\n        returned by the input's `item` method.\n\n    Examples\n    --------\n    >>> np.asscalar(np.array([24]))\n    24\n\n    ")
    
    # Call to item(...): (line 453)
    # Processing the call keyword arguments (line 453)
    kwargs_127737 = {}
    # Getting the type of 'a' (line 453)
    a_127735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 11), 'a', False)
    # Obtaining the member 'item' of a type (line 453)
    item_127736 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 453, 11), a_127735, 'item')
    # Calling item(args, kwargs) (line 453)
    item_call_result_127738 = invoke(stypy.reporting.localization.Localization(__file__, 453, 11), item_127736, *[], **kwargs_127737)
    
    # Assigning a type to the variable 'stypy_return_type' (line 453)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 453, 4), 'stypy_return_type', item_call_result_127738)
    
    # ################# End of 'asscalar(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'asscalar' in the type store
    # Getting the type of 'stypy_return_type' (line 432)
    stypy_return_type_127739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_127739)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'asscalar'
    return stypy_return_type_127739

# Assigning a type to the variable 'asscalar' (line 432)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 432, 0), 'asscalar', asscalar)

# Assigning a Dict to a Name (line 457):

# Assigning a Dict to a Name (line 457):

# Obtaining an instance of the builtin type 'dict' (line 457)
dict_127740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 457, 16), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 457)
# Adding element type (key, value) (line 457)
str_127741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 457, 17), 'str', 'S1')
str_127742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 457, 23), 'str', 'character')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 457, 16), dict_127740, (str_127741, str_127742))
# Adding element type (key, value) (line 457)
str_127743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 458, 17), 'str', '?')
str_127744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 458, 22), 'str', 'bool')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 457, 16), dict_127740, (str_127743, str_127744))
# Adding element type (key, value) (line 457)
str_127745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 459, 17), 'str', 'b')
str_127746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 459, 22), 'str', 'signed char')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 457, 16), dict_127740, (str_127745, str_127746))
# Adding element type (key, value) (line 457)
str_127747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 460, 17), 'str', 'B')
str_127748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 460, 22), 'str', 'unsigned char')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 457, 16), dict_127740, (str_127747, str_127748))
# Adding element type (key, value) (line 457)
str_127749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 461, 17), 'str', 'h')
str_127750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 461, 22), 'str', 'short')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 457, 16), dict_127740, (str_127749, str_127750))
# Adding element type (key, value) (line 457)
str_127751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 462, 17), 'str', 'H')
str_127752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 462, 22), 'str', 'unsigned short')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 457, 16), dict_127740, (str_127751, str_127752))
# Adding element type (key, value) (line 457)
str_127753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 463, 17), 'str', 'i')
str_127754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 463, 22), 'str', 'integer')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 457, 16), dict_127740, (str_127753, str_127754))
# Adding element type (key, value) (line 457)
str_127755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 464, 17), 'str', 'I')
str_127756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 464, 22), 'str', 'unsigned integer')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 457, 16), dict_127740, (str_127755, str_127756))
# Adding element type (key, value) (line 457)
str_127757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 465, 17), 'str', 'l')
str_127758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 465, 22), 'str', 'long integer')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 457, 16), dict_127740, (str_127757, str_127758))
# Adding element type (key, value) (line 457)
str_127759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 466, 17), 'str', 'L')
str_127760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 466, 22), 'str', 'unsigned long integer')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 457, 16), dict_127740, (str_127759, str_127760))
# Adding element type (key, value) (line 457)
str_127761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 467, 17), 'str', 'q')
str_127762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 467, 22), 'str', 'long long integer')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 457, 16), dict_127740, (str_127761, str_127762))
# Adding element type (key, value) (line 457)
str_127763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 468, 17), 'str', 'Q')
str_127764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 468, 22), 'str', 'unsigned long long integer')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 457, 16), dict_127740, (str_127763, str_127764))
# Adding element type (key, value) (line 457)
str_127765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 469, 17), 'str', 'f')
str_127766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 469, 22), 'str', 'single precision')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 457, 16), dict_127740, (str_127765, str_127766))
# Adding element type (key, value) (line 457)
str_127767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 470, 17), 'str', 'd')
str_127768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 470, 22), 'str', 'double precision')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 457, 16), dict_127740, (str_127767, str_127768))
# Adding element type (key, value) (line 457)
str_127769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 471, 17), 'str', 'g')
str_127770 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 471, 22), 'str', 'long precision')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 457, 16), dict_127740, (str_127769, str_127770))
# Adding element type (key, value) (line 457)
str_127771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 472, 17), 'str', 'F')
str_127772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 472, 22), 'str', 'complex single precision')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 457, 16), dict_127740, (str_127771, str_127772))
# Adding element type (key, value) (line 457)
str_127773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 473, 17), 'str', 'D')
str_127774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 473, 22), 'str', 'complex double precision')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 457, 16), dict_127740, (str_127773, str_127774))
# Adding element type (key, value) (line 457)
str_127775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 474, 17), 'str', 'G')
str_127776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 474, 22), 'str', 'complex long double precision')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 457, 16), dict_127740, (str_127775, str_127776))
# Adding element type (key, value) (line 457)
str_127777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 475, 17), 'str', 'S')
str_127778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 475, 22), 'str', 'string')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 457, 16), dict_127740, (str_127777, str_127778))
# Adding element type (key, value) (line 457)
str_127779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 476, 17), 'str', 'U')
str_127780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 476, 22), 'str', 'unicode')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 457, 16), dict_127740, (str_127779, str_127780))
# Adding element type (key, value) (line 457)
str_127781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 477, 17), 'str', 'V')
str_127782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 477, 22), 'str', 'void')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 457, 16), dict_127740, (str_127781, str_127782))
# Adding element type (key, value) (line 457)
str_127783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 478, 17), 'str', 'O')
str_127784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 478, 22), 'str', 'object')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 457, 16), dict_127740, (str_127783, str_127784))

# Assigning a type to the variable '_namefromtype' (line 457)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 457, 0), '_namefromtype', dict_127740)

@norecursion
def typename(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'typename'
    module_type_store = module_type_store.open_function_context('typename', 481, 0, False)
    
    # Passed parameters checking function
    typename.stypy_localization = localization
    typename.stypy_type_of_self = None
    typename.stypy_type_store = module_type_store
    typename.stypy_function_name = 'typename'
    typename.stypy_param_names_list = ['char']
    typename.stypy_varargs_param_name = None
    typename.stypy_kwargs_param_name = None
    typename.stypy_call_defaults = defaults
    typename.stypy_call_varargs = varargs
    typename.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'typename', ['char'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'typename', localization, ['char'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'typename(...)' code ##################

    str_127785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 529, (-1)), 'str', "\n    Return a description for the given data type code.\n\n    Parameters\n    ----------\n    char : str\n        Data type code.\n\n    Returns\n    -------\n    out : str\n        Description of the input data type code.\n\n    See Also\n    --------\n    dtype, typecodes\n\n    Examples\n    --------\n    >>> typechars = ['S1', '?', 'B', 'D', 'G', 'F', 'I', 'H', 'L', 'O', 'Q',\n    ...              'S', 'U', 'V', 'b', 'd', 'g', 'f', 'i', 'h', 'l', 'q']\n    >>> for typechar in typechars:\n    ...     print(typechar, ' : ', np.typename(typechar))\n    ...\n    S1  :  character\n    ?  :  bool\n    B  :  unsigned char\n    D  :  complex double precision\n    G  :  complex long double precision\n    F  :  complex single precision\n    I  :  unsigned integer\n    H  :  unsigned short\n    L  :  unsigned long integer\n    O  :  object\n    Q  :  unsigned long long integer\n    S  :  string\n    U  :  unicode\n    V  :  void\n    b  :  signed char\n    d  :  double precision\n    g  :  long precision\n    f  :  single precision\n    i  :  integer\n    h  :  short\n    l  :  long integer\n    q  :  long long integer\n\n    ")
    
    # Obtaining the type of the subscript
    # Getting the type of 'char' (line 530)
    char_127786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 25), 'char')
    # Getting the type of '_namefromtype' (line 530)
    _namefromtype_127787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 11), '_namefromtype')
    # Obtaining the member '__getitem__' of a type (line 530)
    getitem___127788 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 530, 11), _namefromtype_127787, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 530)
    subscript_call_result_127789 = invoke(stypy.reporting.localization.Localization(__file__, 530, 11), getitem___127788, char_127786)
    
    # Assigning a type to the variable 'stypy_return_type' (line 530)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 530, 4), 'stypy_return_type', subscript_call_result_127789)
    
    # ################# End of 'typename(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'typename' in the type store
    # Getting the type of 'stypy_return_type' (line 481)
    stypy_return_type_127790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_127790)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'typename'
    return stypy_return_type_127790

# Assigning a type to the variable 'typename' (line 481)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 481, 0), 'typename', typename)

# Assigning a List to a Name (line 535):

# Assigning a List to a Name (line 535):

# Obtaining an instance of the builtin type 'list' (line 535)
list_127791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 535, 13), 'list')
# Adding type elements to the builtin type 'list' instance (line 535)
# Adding element type (line 535)

# Obtaining an instance of the builtin type 'list' (line 535)
list_127792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 535, 14), 'list')
# Adding type elements to the builtin type 'list' instance (line 535)
# Adding element type (line 535)
# Getting the type of '_nx' (line 535)
_nx_127793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 15), '_nx')
# Obtaining the member 'half' of a type (line 535)
half_127794 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 535, 15), _nx_127793, 'half')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 535, 14), list_127792, half_127794)
# Adding element type (line 535)
# Getting the type of '_nx' (line 535)
_nx_127795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 25), '_nx')
# Obtaining the member 'single' of a type (line 535)
single_127796 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 535, 25), _nx_127795, 'single')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 535, 14), list_127792, single_127796)
# Adding element type (line 535)
# Getting the type of '_nx' (line 535)
_nx_127797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 37), '_nx')
# Obtaining the member 'double' of a type (line 535)
double_127798 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 535, 37), _nx_127797, 'double')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 535, 14), list_127792, double_127798)
# Adding element type (line 535)
# Getting the type of '_nx' (line 535)
_nx_127799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 49), '_nx')
# Obtaining the member 'longdouble' of a type (line 535)
longdouble_127800 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 535, 49), _nx_127799, 'longdouble')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 535, 14), list_127792, longdouble_127800)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 535, 13), list_127791, list_127792)
# Adding element type (line 535)

# Obtaining an instance of the builtin type 'list' (line 536)
list_127801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 536, 14), 'list')
# Adding type elements to the builtin type 'list' instance (line 536)
# Adding element type (line 536)
# Getting the type of 'None' (line 536)
None_127802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 15), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 536, 14), list_127801, None_127802)
# Adding element type (line 536)
# Getting the type of '_nx' (line 536)
_nx_127803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 21), '_nx')
# Obtaining the member 'csingle' of a type (line 536)
csingle_127804 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 536, 21), _nx_127803, 'csingle')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 536, 14), list_127801, csingle_127804)
# Adding element type (line 536)
# Getting the type of '_nx' (line 536)
_nx_127805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 34), '_nx')
# Obtaining the member 'cdouble' of a type (line 536)
cdouble_127806 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 536, 34), _nx_127805, 'cdouble')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 536, 14), list_127801, cdouble_127806)
# Adding element type (line 536)
# Getting the type of '_nx' (line 536)
_nx_127807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 47), '_nx')
# Obtaining the member 'clongdouble' of a type (line 536)
clongdouble_127808 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 536, 47), _nx_127807, 'clongdouble')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 536, 14), list_127801, clongdouble_127808)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 535, 13), list_127791, list_127801)

# Assigning a type to the variable 'array_type' (line 535)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 535, 0), 'array_type', list_127791)

# Assigning a Dict to a Name (line 537):

# Assigning a Dict to a Name (line 537):

# Obtaining an instance of the builtin type 'dict' (line 537)
dict_127809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 537, 18), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 537)
# Adding element type (key, value) (line 537)
# Getting the type of '_nx' (line 537)
_nx_127810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 19), '_nx')
# Obtaining the member 'half' of a type (line 537)
half_127811 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 537, 19), _nx_127810, 'half')
int_127812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 537, 29), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 537, 18), dict_127809, (half_127811, int_127812))
# Adding element type (key, value) (line 537)
# Getting the type of '_nx' (line 538)
_nx_127813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 19), '_nx')
# Obtaining the member 'single' of a type (line 538)
single_127814 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 538, 19), _nx_127813, 'single')
int_127815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 538, 31), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 537, 18), dict_127809, (single_127814, int_127815))
# Adding element type (key, value) (line 537)
# Getting the type of '_nx' (line 539)
_nx_127816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 19), '_nx')
# Obtaining the member 'double' of a type (line 539)
double_127817 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 539, 19), _nx_127816, 'double')
int_127818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 539, 31), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 537, 18), dict_127809, (double_127817, int_127818))
# Adding element type (key, value) (line 537)
# Getting the type of '_nx' (line 540)
_nx_127819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 19), '_nx')
# Obtaining the member 'longdouble' of a type (line 540)
longdouble_127820 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 540, 19), _nx_127819, 'longdouble')
int_127821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 540, 35), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 537, 18), dict_127809, (longdouble_127820, int_127821))
# Adding element type (key, value) (line 537)
# Getting the type of '_nx' (line 541)
_nx_127822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 19), '_nx')
# Obtaining the member 'csingle' of a type (line 541)
csingle_127823 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 541, 19), _nx_127822, 'csingle')
int_127824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 541, 32), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 537, 18), dict_127809, (csingle_127823, int_127824))
# Adding element type (key, value) (line 537)
# Getting the type of '_nx' (line 542)
_nx_127825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 19), '_nx')
# Obtaining the member 'cdouble' of a type (line 542)
cdouble_127826 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 542, 19), _nx_127825, 'cdouble')
int_127827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 542, 32), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 537, 18), dict_127809, (cdouble_127826, int_127827))
# Adding element type (key, value) (line 537)
# Getting the type of '_nx' (line 543)
_nx_127828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 19), '_nx')
# Obtaining the member 'clongdouble' of a type (line 543)
clongdouble_127829 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 543, 19), _nx_127828, 'clongdouble')
int_127830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 543, 36), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 537, 18), dict_127809, (clongdouble_127829, int_127830))

# Assigning a type to the variable 'array_precision' (line 537)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 537, 0), 'array_precision', dict_127809)

@norecursion
def common_type(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'common_type'
    module_type_store = module_type_store.open_function_context('common_type', 544, 0, False)
    
    # Passed parameters checking function
    common_type.stypy_localization = localization
    common_type.stypy_type_of_self = None
    common_type.stypy_type_store = module_type_store
    common_type.stypy_function_name = 'common_type'
    common_type.stypy_param_names_list = []
    common_type.stypy_varargs_param_name = 'arrays'
    common_type.stypy_kwargs_param_name = None
    common_type.stypy_call_defaults = defaults
    common_type.stypy_call_varargs = varargs
    common_type.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'common_type', [], 'arrays', None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'common_type', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'common_type(...)' code ##################

    str_127831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 579, (-1)), 'str', "\n    Return a scalar type which is common to the input arrays.\n\n    The return type will always be an inexact (i.e. floating point) scalar\n    type, even if all the arrays are integer arrays. If one of the inputs is\n    an integer array, the minimum precision type that is returned is a\n    64-bit floating point dtype.\n\n    All input arrays can be safely cast to the returned dtype without loss\n    of information.\n\n    Parameters\n    ----------\n    array1, array2, ... : ndarrays\n        Input arrays.\n\n    Returns\n    -------\n    out : data type code\n        Data type code.\n\n    See Also\n    --------\n    dtype, mintypecode\n\n    Examples\n    --------\n    >>> np.common_type(np.arange(2, dtype=np.float32))\n    <type 'numpy.float32'>\n    >>> np.common_type(np.arange(2, dtype=np.float32), np.arange(2))\n    <type 'numpy.float64'>\n    >>> np.common_type(np.arange(4), np.array([45, 6.j]), np.array([45.0]))\n    <type 'numpy.complex128'>\n\n    ")
    
    # Assigning a Name to a Name (line 580):
    
    # Assigning a Name to a Name (line 580):
    # Getting the type of 'False' (line 580)
    False_127832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 17), 'False')
    # Assigning a type to the variable 'is_complex' (line 580)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 580, 4), 'is_complex', False_127832)
    
    # Assigning a Num to a Name (line 581):
    
    # Assigning a Num to a Name (line 581):
    int_127833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 581, 16), 'int')
    # Assigning a type to the variable 'precision' (line 581)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 581, 4), 'precision', int_127833)
    
    # Getting the type of 'arrays' (line 582)
    arrays_127834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 13), 'arrays')
    # Testing the type of a for loop iterable (line 582)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 582, 4), arrays_127834)
    # Getting the type of the for loop variable (line 582)
    for_loop_var_127835 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 582, 4), arrays_127834)
    # Assigning a type to the variable 'a' (line 582)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 582, 4), 'a', for_loop_var_127835)
    # SSA begins for a for statement (line 582)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Attribute to a Name (line 583):
    
    # Assigning a Attribute to a Name (line 583):
    # Getting the type of 'a' (line 583)
    a_127836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 12), 'a')
    # Obtaining the member 'dtype' of a type (line 583)
    dtype_127837 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 583, 12), a_127836, 'dtype')
    # Obtaining the member 'type' of a type (line 583)
    type_127838 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 583, 12), dtype_127837, 'type')
    # Assigning a type to the variable 't' (line 583)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 583, 8), 't', type_127838)
    
    
    # Call to iscomplexobj(...): (line 584)
    # Processing the call arguments (line 584)
    # Getting the type of 'a' (line 584)
    a_127840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 24), 'a', False)
    # Processing the call keyword arguments (line 584)
    kwargs_127841 = {}
    # Getting the type of 'iscomplexobj' (line 584)
    iscomplexobj_127839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 11), 'iscomplexobj', False)
    # Calling iscomplexobj(args, kwargs) (line 584)
    iscomplexobj_call_result_127842 = invoke(stypy.reporting.localization.Localization(__file__, 584, 11), iscomplexobj_127839, *[a_127840], **kwargs_127841)
    
    # Testing the type of an if condition (line 584)
    if_condition_127843 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 584, 8), iscomplexobj_call_result_127842)
    # Assigning a type to the variable 'if_condition_127843' (line 584)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 584, 8), 'if_condition_127843', if_condition_127843)
    # SSA begins for if statement (line 584)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 585):
    
    # Assigning a Name to a Name (line 585):
    # Getting the type of 'True' (line 585)
    True_127844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 25), 'True')
    # Assigning a type to the variable 'is_complex' (line 585)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 585, 12), 'is_complex', True_127844)
    # SSA join for if statement (line 584)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to issubclass(...): (line 586)
    # Processing the call arguments (line 586)
    # Getting the type of 't' (line 586)
    t_127846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 22), 't', False)
    # Getting the type of '_nx' (line 586)
    _nx_127847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 25), '_nx', False)
    # Obtaining the member 'integer' of a type (line 586)
    integer_127848 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 586, 25), _nx_127847, 'integer')
    # Processing the call keyword arguments (line 586)
    kwargs_127849 = {}
    # Getting the type of 'issubclass' (line 586)
    issubclass_127845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 11), 'issubclass', False)
    # Calling issubclass(args, kwargs) (line 586)
    issubclass_call_result_127850 = invoke(stypy.reporting.localization.Localization(__file__, 586, 11), issubclass_127845, *[t_127846, integer_127848], **kwargs_127849)
    
    # Testing the type of an if condition (line 586)
    if_condition_127851 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 586, 8), issubclass_call_result_127850)
    # Assigning a type to the variable 'if_condition_127851' (line 586)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 586, 8), 'if_condition_127851', if_condition_127851)
    # SSA begins for if statement (line 586)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 587):
    
    # Assigning a Num to a Name (line 587):
    int_127852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 587, 16), 'int')
    # Assigning a type to the variable 'p' (line 587)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 587, 12), 'p', int_127852)
    # SSA branch for the else part of an if statement (line 586)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 589):
    
    # Assigning a Call to a Name (line 589):
    
    # Call to get(...): (line 589)
    # Processing the call arguments (line 589)
    # Getting the type of 't' (line 589)
    t_127855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 36), 't', False)
    # Getting the type of 'None' (line 589)
    None_127856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 39), 'None', False)
    # Processing the call keyword arguments (line 589)
    kwargs_127857 = {}
    # Getting the type of 'array_precision' (line 589)
    array_precision_127853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 16), 'array_precision', False)
    # Obtaining the member 'get' of a type (line 589)
    get_127854 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 589, 16), array_precision_127853, 'get')
    # Calling get(args, kwargs) (line 589)
    get_call_result_127858 = invoke(stypy.reporting.localization.Localization(__file__, 589, 16), get_127854, *[t_127855, None_127856], **kwargs_127857)
    
    # Assigning a type to the variable 'p' (line 589)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 589, 12), 'p', get_call_result_127858)
    
    # Type idiom detected: calculating its left and rigth part (line 590)
    # Getting the type of 'p' (line 590)
    p_127859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 15), 'p')
    # Getting the type of 'None' (line 590)
    None_127860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 20), 'None')
    
    (may_be_127861, more_types_in_union_127862) = may_be_none(p_127859, None_127860)

    if may_be_127861:

        if more_types_in_union_127862:
            # Runtime conditional SSA (line 590)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to TypeError(...): (line 591)
        # Processing the call arguments (line 591)
        str_127864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 591, 32), 'str', "can't get common type for non-numeric array")
        # Processing the call keyword arguments (line 591)
        kwargs_127865 = {}
        # Getting the type of 'TypeError' (line 591)
        TypeError_127863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 22), 'TypeError', False)
        # Calling TypeError(args, kwargs) (line 591)
        TypeError_call_result_127866 = invoke(stypy.reporting.localization.Localization(__file__, 591, 22), TypeError_127863, *[str_127864], **kwargs_127865)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 591, 16), TypeError_call_result_127866, 'raise parameter', BaseException)

        if more_types_in_union_127862:
            # SSA join for if statement (line 590)
            module_type_store = module_type_store.join_ssa_context()


    
    # SSA join for if statement (line 586)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 592):
    
    # Assigning a Call to a Name (line 592):
    
    # Call to max(...): (line 592)
    # Processing the call arguments (line 592)
    # Getting the type of 'precision' (line 592)
    precision_127868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 24), 'precision', False)
    # Getting the type of 'p' (line 592)
    p_127869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 35), 'p', False)
    # Processing the call keyword arguments (line 592)
    kwargs_127870 = {}
    # Getting the type of 'max' (line 592)
    max_127867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 20), 'max', False)
    # Calling max(args, kwargs) (line 592)
    max_call_result_127871 = invoke(stypy.reporting.localization.Localization(__file__, 592, 20), max_127867, *[precision_127868, p_127869], **kwargs_127870)
    
    # Assigning a type to the variable 'precision' (line 592)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 592, 8), 'precision', max_call_result_127871)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'is_complex' (line 593)
    is_complex_127872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 7), 'is_complex')
    # Testing the type of an if condition (line 593)
    if_condition_127873 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 593, 4), is_complex_127872)
    # Assigning a type to the variable 'if_condition_127873' (line 593)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 593, 4), 'if_condition_127873', if_condition_127873)
    # SSA begins for if statement (line 593)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining the type of the subscript
    # Getting the type of 'precision' (line 594)
    precision_127874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 29), 'precision')
    
    # Obtaining the type of the subscript
    int_127875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 594, 26), 'int')
    # Getting the type of 'array_type' (line 594)
    array_type_127876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 15), 'array_type')
    # Obtaining the member '__getitem__' of a type (line 594)
    getitem___127877 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 594, 15), array_type_127876, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 594)
    subscript_call_result_127878 = invoke(stypy.reporting.localization.Localization(__file__, 594, 15), getitem___127877, int_127875)
    
    # Obtaining the member '__getitem__' of a type (line 594)
    getitem___127879 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 594, 15), subscript_call_result_127878, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 594)
    subscript_call_result_127880 = invoke(stypy.reporting.localization.Localization(__file__, 594, 15), getitem___127879, precision_127874)
    
    # Assigning a type to the variable 'stypy_return_type' (line 594)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 594, 8), 'stypy_return_type', subscript_call_result_127880)
    # SSA branch for the else part of an if statement (line 593)
    module_type_store.open_ssa_branch('else')
    
    # Obtaining the type of the subscript
    # Getting the type of 'precision' (line 596)
    precision_127881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 29), 'precision')
    
    # Obtaining the type of the subscript
    int_127882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 596, 26), 'int')
    # Getting the type of 'array_type' (line 596)
    array_type_127883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 15), 'array_type')
    # Obtaining the member '__getitem__' of a type (line 596)
    getitem___127884 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 596, 15), array_type_127883, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 596)
    subscript_call_result_127885 = invoke(stypy.reporting.localization.Localization(__file__, 596, 15), getitem___127884, int_127882)
    
    # Obtaining the member '__getitem__' of a type (line 596)
    getitem___127886 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 596, 15), subscript_call_result_127885, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 596)
    subscript_call_result_127887 = invoke(stypy.reporting.localization.Localization(__file__, 596, 15), getitem___127886, precision_127881)
    
    # Assigning a type to the variable 'stypy_return_type' (line 596)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 596, 8), 'stypy_return_type', subscript_call_result_127887)
    # SSA join for if statement (line 593)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'common_type(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'common_type' in the type store
    # Getting the type of 'stypy_return_type' (line 544)
    stypy_return_type_127888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_127888)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'common_type'
    return stypy_return_type_127888

# Assigning a type to the variable 'common_type' (line 544)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 544, 0), 'common_type', common_type)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
