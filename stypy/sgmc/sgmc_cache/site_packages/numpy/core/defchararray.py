
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: This module contains a set of functions for vectorized string
3: operations and methods.
4: 
5: .. note::
6:    The `chararray` class exists for backwards compatibility with
7:    Numarray, it is not recommended for new development. Starting from numpy
8:    1.4, if one needs arrays of strings, it is recommended to use arrays of
9:    `dtype` `object_`, `string_` or `unicode_`, and use the free functions
10:    in the `numpy.char` module for fast vectorized string operations.
11: 
12: Some methods will only be available if the corresponding string method is
13: available in your version of Python.
14: 
15: The preferred alias for `defchararray` is `numpy.char`.
16: 
17: '''
18: from __future__ import division, absolute_import, print_function
19: 
20: import sys
21: from .numerictypes import string_, unicode_, integer, object_, bool_, character
22: from .numeric import ndarray, compare_chararrays
23: from .numeric import array as narray
24: from numpy.core.multiarray import _vec_string
25: from numpy.compat import asbytes, long
26: import numpy
27: 
28: __all__ = [
29:     'chararray', 'equal', 'not_equal', 'greater_equal', 'less_equal',
30:     'greater', 'less', 'str_len', 'add', 'multiply', 'mod', 'capitalize',
31:     'center', 'count', 'decode', 'encode', 'endswith', 'expandtabs',
32:     'find', 'index', 'isalnum', 'isalpha', 'isdigit', 'islower', 'isspace',
33:     'istitle', 'isupper', 'join', 'ljust', 'lower', 'lstrip', 'partition',
34:     'replace', 'rfind', 'rindex', 'rjust', 'rpartition', 'rsplit',
35:     'rstrip', 'split', 'splitlines', 'startswith', 'strip', 'swapcase',
36:     'title', 'translate', 'upper', 'zfill', 'isnumeric', 'isdecimal',
37:     'array', 'asarray'
38:     ]
39: 
40: 
41: _globalvar = 0
42: if sys.version_info[0] >= 3:
43:     _unicode = str
44:     _bytes = bytes
45: else:
46:     _unicode = unicode
47:     _bytes = str
48: _len = len
49: 
50: def _use_unicode(*args):
51:     '''
52:     Helper function for determining the output type of some string
53:     operations.
54: 
55:     For an operation on two ndarrays, if at least one is unicode, the
56:     result should be unicode.
57:     '''
58:     for x in args:
59:         if (isinstance(x, _unicode) or
60:                 issubclass(numpy.asarray(x).dtype.type, unicode_)):
61:             return unicode_
62:     return string_
63: 
64: def _to_string_or_unicode_array(result):
65:     '''
66:     Helper function to cast a result back into a string or unicode array
67:     if an object array must be used as an intermediary.
68:     '''
69:     return numpy.asarray(result.tolist())
70: 
71: def _clean_args(*args):
72:     '''
73:     Helper function for delegating arguments to Python string
74:     functions.
75: 
76:     Many of the Python string operations that have optional arguments
77:     do not use 'None' to indicate a default value.  In these cases,
78:     we need to remove all `None` arguments, and those following them.
79:     '''
80:     newargs = []
81:     for chk in args:
82:         if chk is None:
83:             break
84:         newargs.append(chk)
85:     return newargs
86: 
87: def _get_num_chars(a):
88:     '''
89:     Helper function that returns the number of characters per field in
90:     a string or unicode array.  This is to abstract out the fact that
91:     for a unicode array this is itemsize / 4.
92:     '''
93:     if issubclass(a.dtype.type, unicode_):
94:         return a.itemsize // 4
95:     return a.itemsize
96: 
97: 
98: def equal(x1, x2):
99:     '''
100:     Return (x1 == x2) element-wise.
101: 
102:     Unlike `numpy.equal`, this comparison is performed by first
103:     stripping whitespace characters from the end of the string.  This
104:     behavior is provided for backward-compatibility with numarray.
105: 
106:     Parameters
107:     ----------
108:     x1, x2 : array_like of str or unicode
109:         Input arrays of the same shape.
110: 
111:     Returns
112:     -------
113:     out : ndarray or bool
114:         Output array of bools, or a single bool if x1 and x2 are scalars.
115: 
116:     See Also
117:     --------
118:     not_equal, greater_equal, less_equal, greater, less
119:     '''
120:     return compare_chararrays(x1, x2, '==', True)
121: 
122: def not_equal(x1, x2):
123:     '''
124:     Return (x1 != x2) element-wise.
125: 
126:     Unlike `numpy.not_equal`, this comparison is performed by first
127:     stripping whitespace characters from the end of the string.  This
128:     behavior is provided for backward-compatibility with numarray.
129: 
130:     Parameters
131:     ----------
132:     x1, x2 : array_like of str or unicode
133:         Input arrays of the same shape.
134: 
135:     Returns
136:     -------
137:     out : ndarray or bool
138:         Output array of bools, or a single bool if x1 and x2 are scalars.
139: 
140:     See Also
141:     --------
142:     equal, greater_equal, less_equal, greater, less
143:     '''
144:     return compare_chararrays(x1, x2, '!=', True)
145: 
146: def greater_equal(x1, x2):
147:     '''
148:     Return (x1 >= x2) element-wise.
149: 
150:     Unlike `numpy.greater_equal`, this comparison is performed by
151:     first stripping whitespace characters from the end of the string.
152:     This behavior is provided for backward-compatibility with
153:     numarray.
154: 
155:     Parameters
156:     ----------
157:     x1, x2 : array_like of str or unicode
158:         Input arrays of the same shape.
159: 
160:     Returns
161:     -------
162:     out : ndarray or bool
163:         Output array of bools, or a single bool if x1 and x2 are scalars.
164: 
165:     See Also
166:     --------
167:     equal, not_equal, less_equal, greater, less
168:     '''
169:     return compare_chararrays(x1, x2, '>=', True)
170: 
171: def less_equal(x1, x2):
172:     '''
173:     Return (x1 <= x2) element-wise.
174: 
175:     Unlike `numpy.less_equal`, this comparison is performed by first
176:     stripping whitespace characters from the end of the string.  This
177:     behavior is provided for backward-compatibility with numarray.
178: 
179:     Parameters
180:     ----------
181:     x1, x2 : array_like of str or unicode
182:         Input arrays of the same shape.
183: 
184:     Returns
185:     -------
186:     out : ndarray or bool
187:         Output array of bools, or a single bool if x1 and x2 are scalars.
188: 
189:     See Also
190:     --------
191:     equal, not_equal, greater_equal, greater, less
192:     '''
193:     return compare_chararrays(x1, x2, '<=', True)
194: 
195: def greater(x1, x2):
196:     '''
197:     Return (x1 > x2) element-wise.
198: 
199:     Unlike `numpy.greater`, this comparison is performed by first
200:     stripping whitespace characters from the end of the string.  This
201:     behavior is provided for backward-compatibility with numarray.
202: 
203:     Parameters
204:     ----------
205:     x1, x2 : array_like of str or unicode
206:         Input arrays of the same shape.
207: 
208:     Returns
209:     -------
210:     out : ndarray or bool
211:         Output array of bools, or a single bool if x1 and x2 are scalars.
212: 
213:     See Also
214:     --------
215:     equal, not_equal, greater_equal, less_equal, less
216:     '''
217:     return compare_chararrays(x1, x2, '>', True)
218: 
219: def less(x1, x2):
220:     '''
221:     Return (x1 < x2) element-wise.
222: 
223:     Unlike `numpy.greater`, this comparison is performed by first
224:     stripping whitespace characters from the end of the string.  This
225:     behavior is provided for backward-compatibility with numarray.
226: 
227:     Parameters
228:     ----------
229:     x1, x2 : array_like of str or unicode
230:         Input arrays of the same shape.
231: 
232:     Returns
233:     -------
234:     out : ndarray or bool
235:         Output array of bools, or a single bool if x1 and x2 are scalars.
236: 
237:     See Also
238:     --------
239:     equal, not_equal, greater_equal, less_equal, greater
240:     '''
241:     return compare_chararrays(x1, x2, '<', True)
242: 
243: def str_len(a):
244:     '''
245:     Return len(a) element-wise.
246: 
247:     Parameters
248:     ----------
249:     a : array_like of str or unicode
250: 
251:     Returns
252:     -------
253:     out : ndarray
254:         Output array of integers
255: 
256:     See also
257:     --------
258:     __builtin__.len
259:     '''
260:     return _vec_string(a, integer, '__len__')
261: 
262: def add(x1, x2):
263:     '''
264:     Return element-wise string concatenation for two arrays of str or unicode.
265: 
266:     Arrays `x1` and `x2` must have the same shape.
267: 
268:     Parameters
269:     ----------
270:     x1 : array_like of str or unicode
271:         Input array.
272:     x2 : array_like of str or unicode
273:         Input array.
274: 
275:     Returns
276:     -------
277:     add : ndarray
278:         Output array of `string_` or `unicode_`, depending on input types
279:         of the same shape as `x1` and `x2`.
280: 
281:     '''
282:     arr1 = numpy.asarray(x1)
283:     arr2 = numpy.asarray(x2)
284:     out_size = _get_num_chars(arr1) + _get_num_chars(arr2)
285:     dtype = _use_unicode(arr1, arr2)
286:     return _vec_string(arr1, (dtype, out_size), '__add__', (arr2,))
287: 
288: def multiply(a, i):
289:     '''
290:     Return (a * i), that is string multiple concatenation,
291:     element-wise.
292: 
293:     Values in `i` of less than 0 are treated as 0 (which yields an
294:     empty string).
295: 
296:     Parameters
297:     ----------
298:     a : array_like of str or unicode
299: 
300:     i : array_like of ints
301: 
302:     Returns
303:     -------
304:     out : ndarray
305:         Output array of str or unicode, depending on input types
306: 
307:     '''
308:     a_arr = numpy.asarray(a)
309:     i_arr = numpy.asarray(i)
310:     if not issubclass(i_arr.dtype.type, integer):
311:         raise ValueError("Can only multiply by integers")
312:     out_size = _get_num_chars(a_arr) * max(long(i_arr.max()), 0)
313:     return _vec_string(
314:         a_arr, (a_arr.dtype.type, out_size), '__mul__', (i_arr,))
315: 
316: def mod(a, values):
317:     '''
318:     Return (a % i), that is pre-Python 2.6 string formatting
319:     (iterpolation), element-wise for a pair of array_likes of str
320:     or unicode.
321: 
322:     Parameters
323:     ----------
324:     a : array_like of str or unicode
325: 
326:     values : array_like of values
327:        These values will be element-wise interpolated into the string.
328: 
329:     Returns
330:     -------
331:     out : ndarray
332:         Output array of str or unicode, depending on input types
333: 
334:     See also
335:     --------
336:     str.__mod__
337: 
338:     '''
339:     return _to_string_or_unicode_array(
340:         _vec_string(a, object_, '__mod__', (values,)))
341: 
342: def capitalize(a):
343:     '''
344:     Return a copy of `a` with only the first character of each element
345:     capitalized.
346: 
347:     Calls `str.capitalize` element-wise.
348: 
349:     For 8-bit strings, this method is locale-dependent.
350: 
351:     Parameters
352:     ----------
353:     a : array_like of str or unicode
354:         Input array of strings to capitalize.
355: 
356:     Returns
357:     -------
358:     out : ndarray
359:         Output array of str or unicode, depending on input
360:         types
361: 
362:     See also
363:     --------
364:     str.capitalize
365: 
366:     Examples
367:     --------
368:     >>> c = np.array(['a1b2','1b2a','b2a1','2a1b'],'S4'); c
369:     array(['a1b2', '1b2a', 'b2a1', '2a1b'],
370:         dtype='|S4')
371:     >>> np.char.capitalize(c)
372:     array(['A1b2', '1b2a', 'B2a1', '2a1b'],
373:         dtype='|S4')
374: 
375:     '''
376:     a_arr = numpy.asarray(a)
377:     return _vec_string(a_arr, a_arr.dtype, 'capitalize')
378: 
379: 
380: def center(a, width, fillchar=' '):
381:     '''
382:     Return a copy of `a` with its elements centered in a string of
383:     length `width`.
384: 
385:     Calls `str.center` element-wise.
386: 
387:     Parameters
388:     ----------
389:     a : array_like of str or unicode
390: 
391:     width : int
392:         The length of the resulting strings
393:     fillchar : str or unicode, optional
394:         The padding character to use (default is space).
395: 
396:     Returns
397:     -------
398:     out : ndarray
399:         Output array of str or unicode, depending on input
400:         types
401: 
402:     See also
403:     --------
404:     str.center
405: 
406:     '''
407:     a_arr = numpy.asarray(a)
408:     width_arr = numpy.asarray(width)
409:     size = long(numpy.max(width_arr.flat))
410:     if numpy.issubdtype(a_arr.dtype, numpy.string_):
411:         fillchar = asbytes(fillchar)
412:     return _vec_string(
413:         a_arr, (a_arr.dtype.type, size), 'center', (width_arr, fillchar))
414: 
415: 
416: def count(a, sub, start=0, end=None):
417:     '''
418:     Returns an array with the number of non-overlapping occurrences of
419:     substring `sub` in the range [`start`, `end`].
420: 
421:     Calls `str.count` element-wise.
422: 
423:     Parameters
424:     ----------
425:     a : array_like of str or unicode
426: 
427:     sub : str or unicode
428:        The substring to search for.
429: 
430:     start, end : int, optional
431:        Optional arguments `start` and `end` are interpreted as slice
432:        notation to specify the range in which to count.
433: 
434:     Returns
435:     -------
436:     out : ndarray
437:         Output array of ints.
438: 
439:     See also
440:     --------
441:     str.count
442: 
443:     Examples
444:     --------
445:     >>> c = np.array(['aAaAaA', '  aA  ', 'abBABba'])
446:     >>> c
447:     array(['aAaAaA', '  aA  ', 'abBABba'],
448:         dtype='|S7')
449:     >>> np.char.count(c, 'A')
450:     array([3, 1, 1])
451:     >>> np.char.count(c, 'aA')
452:     array([3, 1, 0])
453:     >>> np.char.count(c, 'A', start=1, end=4)
454:     array([2, 1, 1])
455:     >>> np.char.count(c, 'A', start=1, end=3)
456:     array([1, 0, 0])
457: 
458:     '''
459:     return _vec_string(a, integer, 'count', [sub, start] + _clean_args(end))
460: 
461: 
462: def decode(a, encoding=None, errors=None):
463:     '''
464:     Calls `str.decode` element-wise.
465: 
466:     The set of available codecs comes from the Python standard library,
467:     and may be extended at runtime.  For more information, see the
468:     :mod:`codecs` module.
469: 
470:     Parameters
471:     ----------
472:     a : array_like of str or unicode
473: 
474:     encoding : str, optional
475:        The name of an encoding
476: 
477:     errors : str, optional
478:        Specifies how to handle encoding errors
479: 
480:     Returns
481:     -------
482:     out : ndarray
483: 
484:     See also
485:     --------
486:     str.decode
487: 
488:     Notes
489:     -----
490:     The type of the result will depend on the encoding specified.
491: 
492:     Examples
493:     --------
494:     >>> c = np.array(['aAaAaA', '  aA  ', 'abBABba'])
495:     >>> c
496:     array(['aAaAaA', '  aA  ', 'abBABba'],
497:         dtype='|S7')
498:     >>> np.char.encode(c, encoding='cp037')
499:     array(['\\x81\\xc1\\x81\\xc1\\x81\\xc1', '@@\\x81\\xc1@@',
500:         '\\x81\\x82\\xc2\\xc1\\xc2\\x82\\x81'],
501:         dtype='|S7')
502: 
503:     '''
504:     return _to_string_or_unicode_array(
505:         _vec_string(a, object_, 'decode', _clean_args(encoding, errors)))
506: 
507: 
508: def encode(a, encoding=None, errors=None):
509:     '''
510:     Calls `str.encode` element-wise.
511: 
512:     The set of available codecs comes from the Python standard library,
513:     and may be extended at runtime. For more information, see the codecs
514:     module.
515: 
516:     Parameters
517:     ----------
518:     a : array_like of str or unicode
519: 
520:     encoding : str, optional
521:        The name of an encoding
522: 
523:     errors : str, optional
524:        Specifies how to handle encoding errors
525: 
526:     Returns
527:     -------
528:     out : ndarray
529: 
530:     See also
531:     --------
532:     str.encode
533: 
534:     Notes
535:     -----
536:     The type of the result will depend on the encoding specified.
537: 
538:     '''
539:     return _to_string_or_unicode_array(
540:         _vec_string(a, object_, 'encode', _clean_args(encoding, errors)))
541: 
542: 
543: def endswith(a, suffix, start=0, end=None):
544:     '''
545:     Returns a boolean array which is `True` where the string element
546:     in `a` ends with `suffix`, otherwise `False`.
547: 
548:     Calls `str.endswith` element-wise.
549: 
550:     Parameters
551:     ----------
552:     a : array_like of str or unicode
553: 
554:     suffix : str
555: 
556:     start, end : int, optional
557:         With optional `start`, test beginning at that position. With
558:         optional `end`, stop comparing at that position.
559: 
560:     Returns
561:     -------
562:     out : ndarray
563:         Outputs an array of bools.
564: 
565:     See also
566:     --------
567:     str.endswith
568: 
569:     Examples
570:     --------
571:     >>> s = np.array(['foo', 'bar'])
572:     >>> s[0] = 'foo'
573:     >>> s[1] = 'bar'
574:     >>> s
575:     array(['foo', 'bar'],
576:         dtype='|S3')
577:     >>> np.char.endswith(s, 'ar')
578:     array([False,  True], dtype=bool)
579:     >>> np.char.endswith(s, 'a', start=1, end=2)
580:     array([False,  True], dtype=bool)
581: 
582:     '''
583:     return _vec_string(
584:         a, bool_, 'endswith', [suffix, start] + _clean_args(end))
585: 
586: 
587: def expandtabs(a, tabsize=8):
588:     '''
589:     Return a copy of each string element where all tab characters are
590:     replaced by one or more spaces.
591: 
592:     Calls `str.expandtabs` element-wise.
593: 
594:     Return a copy of each string element where all tab characters are
595:     replaced by one or more spaces, depending on the current column
596:     and the given `tabsize`. The column number is reset to zero after
597:     each newline occurring in the string. This doesn't understand other
598:     non-printing characters or escape sequences.
599: 
600:     Parameters
601:     ----------
602:     a : array_like of str or unicode
603:         Input array
604:     tabsize : int, optional
605:         Replace tabs with `tabsize` number of spaces.  If not given defaults
606:         to 8 spaces.
607: 
608:     Returns
609:     -------
610:     out : ndarray
611:         Output array of str or unicode, depending on input type
612: 
613:     See also
614:     --------
615:     str.expandtabs
616: 
617:     '''
618:     return _to_string_or_unicode_array(
619:         _vec_string(a, object_, 'expandtabs', (tabsize,)))
620: 
621: 
622: def find(a, sub, start=0, end=None):
623:     '''
624:     For each element, return the lowest index in the string where
625:     substring `sub` is found.
626: 
627:     Calls `str.find` element-wise.
628: 
629:     For each element, return the lowest index in the string where
630:     substring `sub` is found, such that `sub` is contained in the
631:     range [`start`, `end`].
632: 
633:     Parameters
634:     ----------
635:     a : array_like of str or unicode
636: 
637:     sub : str or unicode
638: 
639:     start, end : int, optional
640:         Optional arguments `start` and `end` are interpreted as in
641:         slice notation.
642: 
643:     Returns
644:     -------
645:     out : ndarray or int
646:         Output array of ints.  Returns -1 if `sub` is not found.
647: 
648:     See also
649:     --------
650:     str.find
651: 
652:     '''
653:     return _vec_string(
654:         a, integer, 'find', [sub, start] + _clean_args(end))
655: 
656: 
657: def index(a, sub, start=0, end=None):
658:     '''
659:     Like `find`, but raises `ValueError` when the substring is not found.
660: 
661:     Calls `str.index` element-wise.
662: 
663:     Parameters
664:     ----------
665:     a : array_like of str or unicode
666: 
667:     sub : str or unicode
668: 
669:     start, end : int, optional
670: 
671:     Returns
672:     -------
673:     out : ndarray
674:         Output array of ints.  Returns -1 if `sub` is not found.
675: 
676:     See also
677:     --------
678:     find, str.find
679: 
680:     '''
681:     return _vec_string(
682:         a, integer, 'index', [sub, start] + _clean_args(end))
683: 
684: def isalnum(a):
685:     '''
686:     Returns true for each element if all characters in the string are
687:     alphanumeric and there is at least one character, false otherwise.
688: 
689:     Calls `str.isalnum` element-wise.
690: 
691:     For 8-bit strings, this method is locale-dependent.
692: 
693:     Parameters
694:     ----------
695:     a : array_like of str or unicode
696: 
697:     Returns
698:     -------
699:     out : ndarray
700:         Output array of str or unicode, depending on input type
701: 
702:     See also
703:     --------
704:     str.isalnum
705:     '''
706:     return _vec_string(a, bool_, 'isalnum')
707: 
708: def isalpha(a):
709:     '''
710:     Returns true for each element if all characters in the string are
711:     alphabetic and there is at least one character, false otherwise.
712: 
713:     Calls `str.isalpha` element-wise.
714: 
715:     For 8-bit strings, this method is locale-dependent.
716: 
717:     Parameters
718:     ----------
719:     a : array_like of str or unicode
720: 
721:     Returns
722:     -------
723:     out : ndarray
724:         Output array of bools
725: 
726:     See also
727:     --------
728:     str.isalpha
729:     '''
730:     return _vec_string(a, bool_, 'isalpha')
731: 
732: def isdigit(a):
733:     '''
734:     Returns true for each element if all characters in the string are
735:     digits and there is at least one character, false otherwise.
736: 
737:     Calls `str.isdigit` element-wise.
738: 
739:     For 8-bit strings, this method is locale-dependent.
740: 
741:     Parameters
742:     ----------
743:     a : array_like of str or unicode
744: 
745:     Returns
746:     -------
747:     out : ndarray
748:         Output array of bools
749: 
750:     See also
751:     --------
752:     str.isdigit
753:     '''
754:     return _vec_string(a, bool_, 'isdigit')
755: 
756: def islower(a):
757:     '''
758:     Returns true for each element if all cased characters in the
759:     string are lowercase and there is at least one cased character,
760:     false otherwise.
761: 
762:     Calls `str.islower` element-wise.
763: 
764:     For 8-bit strings, this method is locale-dependent.
765: 
766:     Parameters
767:     ----------
768:     a : array_like of str or unicode
769: 
770:     Returns
771:     -------
772:     out : ndarray
773:         Output array of bools
774: 
775:     See also
776:     --------
777:     str.islower
778:     '''
779:     return _vec_string(a, bool_, 'islower')
780: 
781: def isspace(a):
782:     '''
783:     Returns true for each element if there are only whitespace
784:     characters in the string and there is at least one character,
785:     false otherwise.
786: 
787:     Calls `str.isspace` element-wise.
788: 
789:     For 8-bit strings, this method is locale-dependent.
790: 
791:     Parameters
792:     ----------
793:     a : array_like of str or unicode
794: 
795:     Returns
796:     -------
797:     out : ndarray
798:         Output array of bools
799: 
800:     See also
801:     --------
802:     str.isspace
803:     '''
804:     return _vec_string(a, bool_, 'isspace')
805: 
806: def istitle(a):
807:     '''
808:     Returns true for each element if the element is a titlecased
809:     string and there is at least one character, false otherwise.
810: 
811:     Call `str.istitle` element-wise.
812: 
813:     For 8-bit strings, this method is locale-dependent.
814: 
815:     Parameters
816:     ----------
817:     a : array_like of str or unicode
818: 
819:     Returns
820:     -------
821:     out : ndarray
822:         Output array of bools
823: 
824:     See also
825:     --------
826:     str.istitle
827:     '''
828:     return _vec_string(a, bool_, 'istitle')
829: 
830: def isupper(a):
831:     '''
832:     Returns true for each element if all cased characters in the
833:     string are uppercase and there is at least one character, false
834:     otherwise.
835: 
836:     Call `str.isupper` element-wise.
837: 
838:     For 8-bit strings, this method is locale-dependent.
839: 
840:     Parameters
841:     ----------
842:     a : array_like of str or unicode
843: 
844:     Returns
845:     -------
846:     out : ndarray
847:         Output array of bools
848: 
849:     See also
850:     --------
851:     str.isupper
852:     '''
853:     return _vec_string(a, bool_, 'isupper')
854: 
855: def join(sep, seq):
856:     '''
857:     Return a string which is the concatenation of the strings in the
858:     sequence `seq`.
859: 
860:     Calls `str.join` element-wise.
861: 
862:     Parameters
863:     ----------
864:     sep : array_like of str or unicode
865:     seq : array_like of str or unicode
866: 
867:     Returns
868:     -------
869:     out : ndarray
870:         Output array of str or unicode, depending on input types
871: 
872:     See also
873:     --------
874:     str.join
875:     '''
876:     return _to_string_or_unicode_array(
877:         _vec_string(sep, object_, 'join', (seq,)))
878: 
879: 
880: def ljust(a, width, fillchar=' '):
881:     '''
882:     Return an array with the elements of `a` left-justified in a
883:     string of length `width`.
884: 
885:     Calls `str.ljust` element-wise.
886: 
887:     Parameters
888:     ----------
889:     a : array_like of str or unicode
890: 
891:     width : int
892:         The length of the resulting strings
893:     fillchar : str or unicode, optional
894:         The character to use for padding
895: 
896:     Returns
897:     -------
898:     out : ndarray
899:         Output array of str or unicode, depending on input type
900: 
901:     See also
902:     --------
903:     str.ljust
904: 
905:     '''
906:     a_arr = numpy.asarray(a)
907:     width_arr = numpy.asarray(width)
908:     size = long(numpy.max(width_arr.flat))
909:     if numpy.issubdtype(a_arr.dtype, numpy.string_):
910:         fillchar = asbytes(fillchar)
911:     return _vec_string(
912:         a_arr, (a_arr.dtype.type, size), 'ljust', (width_arr, fillchar))
913: 
914: 
915: def lower(a):
916:     '''
917:     Return an array with the elements converted to lowercase.
918: 
919:     Call `str.lower` element-wise.
920: 
921:     For 8-bit strings, this method is locale-dependent.
922: 
923:     Parameters
924:     ----------
925:     a : array_like, {str, unicode}
926:         Input array.
927: 
928:     Returns
929:     -------
930:     out : ndarray, {str, unicode}
931:         Output array of str or unicode, depending on input type
932: 
933:     See also
934:     --------
935:     str.lower
936: 
937:     Examples
938:     --------
939:     >>> c = np.array(['A1B C', '1BCA', 'BCA1']); c
940:     array(['A1B C', '1BCA', 'BCA1'],
941:           dtype='|S5')
942:     >>> np.char.lower(c)
943:     array(['a1b c', '1bca', 'bca1'],
944:           dtype='|S5')
945: 
946:     '''
947:     a_arr = numpy.asarray(a)
948:     return _vec_string(a_arr, a_arr.dtype, 'lower')
949: 
950: 
951: def lstrip(a, chars=None):
952:     '''
953:     For each element in `a`, return a copy with the leading characters
954:     removed.
955: 
956:     Calls `str.lstrip` element-wise.
957: 
958:     Parameters
959:     ----------
960:     a : array-like, {str, unicode}
961:         Input array.
962: 
963:     chars : {str, unicode}, optional
964:         The `chars` argument is a string specifying the set of
965:         characters to be removed. If omitted or None, the `chars`
966:         argument defaults to removing whitespace. The `chars` argument
967:         is not a prefix; rather, all combinations of its values are
968:         stripped.
969: 
970:     Returns
971:     -------
972:     out : ndarray, {str, unicode}
973:         Output array of str or unicode, depending on input type
974: 
975:     See also
976:     --------
977:     str.lstrip
978: 
979:     Examples
980:     --------
981:     >>> c = np.array(['aAaAaA', '  aA  ', 'abBABba'])
982:     >>> c
983:     array(['aAaAaA', '  aA  ', 'abBABba'],
984:         dtype='|S7')
985: 
986:     The 'a' variable is unstripped from c[1] because whitespace leading.
987: 
988:     >>> np.char.lstrip(c, 'a')
989:     array(['AaAaA', '  aA  ', 'bBABba'],
990:         dtype='|S7')
991: 
992: 
993:     >>> np.char.lstrip(c, 'A') # leaves c unchanged
994:     array(['aAaAaA', '  aA  ', 'abBABba'],
995:         dtype='|S7')
996:     >>> (np.char.lstrip(c, ' ') == np.char.lstrip(c, '')).all()
997:     ... # XXX: is this a regression? this line now returns False
998:     ... # np.char.lstrip(c,'') does not modify c at all.
999:     True
1000:     >>> (np.char.lstrip(c, ' ') == np.char.lstrip(c, None)).all()
1001:     True
1002: 
1003:     '''
1004:     a_arr = numpy.asarray(a)
1005:     return _vec_string(a_arr, a_arr.dtype, 'lstrip', (chars,))
1006: 
1007: 
1008: def partition(a, sep):
1009:     '''
1010:     Partition each element in `a` around `sep`.
1011: 
1012:     Calls `str.partition` element-wise.
1013: 
1014:     For each element in `a`, split the element as the first
1015:     occurrence of `sep`, and return 3 strings containing the part
1016:     before the separator, the separator itself, and the part after
1017:     the separator. If the separator is not found, return 3 strings
1018:     containing the string itself, followed by two empty strings.
1019: 
1020:     Parameters
1021:     ----------
1022:     a : array_like, {str, unicode}
1023:         Input array
1024:     sep : {str, unicode}
1025:         Separator to split each string element in `a`.
1026: 
1027:     Returns
1028:     -------
1029:     out : ndarray, {str, unicode}
1030:         Output array of str or unicode, depending on input type.
1031:         The output array will have an extra dimension with 3
1032:         elements per input element.
1033: 
1034:     See also
1035:     --------
1036:     str.partition
1037: 
1038:     '''
1039:     return _to_string_or_unicode_array(
1040:         _vec_string(a, object_, 'partition', (sep,)))
1041: 
1042: 
1043: def replace(a, old, new, count=None):
1044:     '''
1045:     For each element in `a`, return a copy of the string with all
1046:     occurrences of substring `old` replaced by `new`.
1047: 
1048:     Calls `str.replace` element-wise.
1049: 
1050:     Parameters
1051:     ----------
1052:     a : array-like of str or unicode
1053: 
1054:     old, new : str or unicode
1055: 
1056:     count : int, optional
1057:         If the optional argument `count` is given, only the first
1058:         `count` occurrences are replaced.
1059: 
1060:     Returns
1061:     -------
1062:     out : ndarray
1063:         Output array of str or unicode, depending on input type
1064: 
1065:     See also
1066:     --------
1067:     str.replace
1068: 
1069:     '''
1070:     return _to_string_or_unicode_array(
1071:         _vec_string(
1072:             a, object_, 'replace', [old, new] + _clean_args(count)))
1073: 
1074: 
1075: def rfind(a, sub, start=0, end=None):
1076:     '''
1077:     For each element in `a`, return the highest index in the string
1078:     where substring `sub` is found, such that `sub` is contained
1079:     within [`start`, `end`].
1080: 
1081:     Calls `str.rfind` element-wise.
1082: 
1083:     Parameters
1084:     ----------
1085:     a : array-like of str or unicode
1086: 
1087:     sub : str or unicode
1088: 
1089:     start, end : int, optional
1090:         Optional arguments `start` and `end` are interpreted as in
1091:         slice notation.
1092: 
1093:     Returns
1094:     -------
1095:     out : ndarray
1096:        Output array of ints.  Return -1 on failure.
1097: 
1098:     See also
1099:     --------
1100:     str.rfind
1101: 
1102:     '''
1103:     return _vec_string(
1104:         a, integer, 'rfind', [sub, start] + _clean_args(end))
1105: 
1106: 
1107: def rindex(a, sub, start=0, end=None):
1108:     '''
1109:     Like `rfind`, but raises `ValueError` when the substring `sub` is
1110:     not found.
1111: 
1112:     Calls `str.rindex` element-wise.
1113: 
1114:     Parameters
1115:     ----------
1116:     a : array-like of str or unicode
1117: 
1118:     sub : str or unicode
1119: 
1120:     start, end : int, optional
1121: 
1122:     Returns
1123:     -------
1124:     out : ndarray
1125:        Output array of ints.
1126: 
1127:     See also
1128:     --------
1129:     rfind, str.rindex
1130: 
1131:     '''
1132:     return _vec_string(
1133:         a, integer, 'rindex', [sub, start] + _clean_args(end))
1134: 
1135: 
1136: def rjust(a, width, fillchar=' '):
1137:     '''
1138:     Return an array with the elements of `a` right-justified in a
1139:     string of length `width`.
1140: 
1141:     Calls `str.rjust` element-wise.
1142: 
1143:     Parameters
1144:     ----------
1145:     a : array_like of str or unicode
1146: 
1147:     width : int
1148:         The length of the resulting strings
1149:     fillchar : str or unicode, optional
1150:         The character to use for padding
1151: 
1152:     Returns
1153:     -------
1154:     out : ndarray
1155:         Output array of str or unicode, depending on input type
1156: 
1157:     See also
1158:     --------
1159:     str.rjust
1160: 
1161:     '''
1162:     a_arr = numpy.asarray(a)
1163:     width_arr = numpy.asarray(width)
1164:     size = long(numpy.max(width_arr.flat))
1165:     if numpy.issubdtype(a_arr.dtype, numpy.string_):
1166:         fillchar = asbytes(fillchar)
1167:     return _vec_string(
1168:         a_arr, (a_arr.dtype.type, size), 'rjust', (width_arr, fillchar))
1169: 
1170: 
1171: def rpartition(a, sep):
1172:     '''
1173:     Partition (split) each element around the right-most separator.
1174: 
1175:     Calls `str.rpartition` element-wise.
1176: 
1177:     For each element in `a`, split the element as the last
1178:     occurrence of `sep`, and return 3 strings containing the part
1179:     before the separator, the separator itself, and the part after
1180:     the separator. If the separator is not found, return 3 strings
1181:     containing the string itself, followed by two empty strings.
1182: 
1183:     Parameters
1184:     ----------
1185:     a : array_like of str or unicode
1186:         Input array
1187:     sep : str or unicode
1188:         Right-most separator to split each element in array.
1189: 
1190:     Returns
1191:     -------
1192:     out : ndarray
1193:         Output array of string or unicode, depending on input
1194:         type.  The output array will have an extra dimension with
1195:         3 elements per input element.
1196: 
1197:     See also
1198:     --------
1199:     str.rpartition
1200: 
1201:     '''
1202:     return _to_string_or_unicode_array(
1203:         _vec_string(a, object_, 'rpartition', (sep,)))
1204: 
1205: 
1206: def rsplit(a, sep=None, maxsplit=None):
1207:     '''
1208:     For each element in `a`, return a list of the words in the
1209:     string, using `sep` as the delimiter string.
1210: 
1211:     Calls `str.rsplit` element-wise.
1212: 
1213:     Except for splitting from the right, `rsplit`
1214:     behaves like `split`.
1215: 
1216:     Parameters
1217:     ----------
1218:     a : array_like of str or unicode
1219: 
1220:     sep : str or unicode, optional
1221:         If `sep` is not specified or `None`, any whitespace string
1222:         is a separator.
1223:     maxsplit : int, optional
1224:         If `maxsplit` is given, at most `maxsplit` splits are done,
1225:         the rightmost ones.
1226: 
1227:     Returns
1228:     -------
1229:     out : ndarray
1230:        Array of list objects
1231: 
1232:     See also
1233:     --------
1234:     str.rsplit, split
1235: 
1236:     '''
1237:     # This will return an array of lists of different sizes, so we
1238:     # leave it as an object array
1239:     return _vec_string(
1240:         a, object_, 'rsplit', [sep] + _clean_args(maxsplit))
1241: 
1242: 
1243: def rstrip(a, chars=None):
1244:     '''
1245:     For each element in `a`, return a copy with the trailing
1246:     characters removed.
1247: 
1248:     Calls `str.rstrip` element-wise.
1249: 
1250:     Parameters
1251:     ----------
1252:     a : array-like of str or unicode
1253: 
1254:     chars : str or unicode, optional
1255:        The `chars` argument is a string specifying the set of
1256:        characters to be removed. If omitted or None, the `chars`
1257:        argument defaults to removing whitespace. The `chars` argument
1258:        is not a suffix; rather, all combinations of its values are
1259:        stripped.
1260: 
1261:     Returns
1262:     -------
1263:     out : ndarray
1264:         Output array of str or unicode, depending on input type
1265: 
1266:     See also
1267:     --------
1268:     str.rstrip
1269: 
1270:     Examples
1271:     --------
1272:     >>> c = np.array(['aAaAaA', 'abBABba'], dtype='S7'); c
1273:     array(['aAaAaA', 'abBABba'],
1274:         dtype='|S7')
1275:     >>> np.char.rstrip(c, 'a')
1276:     array(['aAaAaA', 'abBABb'],
1277:         dtype='|S7')
1278:     >>> np.char.rstrip(c, 'A')
1279:     array(['aAaAa', 'abBABba'],
1280:         dtype='|S7')
1281: 
1282:     '''
1283:     a_arr = numpy.asarray(a)
1284:     return _vec_string(a_arr, a_arr.dtype, 'rstrip', (chars,))
1285: 
1286: 
1287: def split(a, sep=None, maxsplit=None):
1288:     '''
1289:     For each element in `a`, return a list of the words in the
1290:     string, using `sep` as the delimiter string.
1291: 
1292:     Calls `str.rsplit` element-wise.
1293: 
1294:     Parameters
1295:     ----------
1296:     a : array_like of str or unicode
1297: 
1298:     sep : str or unicode, optional
1299:        If `sep` is not specified or `None`, any whitespace string is a
1300:        separator.
1301: 
1302:     maxsplit : int, optional
1303:         If `maxsplit` is given, at most `maxsplit` splits are done.
1304: 
1305:     Returns
1306:     -------
1307:     out : ndarray
1308:         Array of list objects
1309: 
1310:     See also
1311:     --------
1312:     str.split, rsplit
1313: 
1314:     '''
1315:     # This will return an array of lists of different sizes, so we
1316:     # leave it as an object array
1317:     return _vec_string(
1318:         a, object_, 'split', [sep] + _clean_args(maxsplit))
1319: 
1320: 
1321: def splitlines(a, keepends=None):
1322:     '''
1323:     For each element in `a`, return a list of the lines in the
1324:     element, breaking at line boundaries.
1325: 
1326:     Calls `str.splitlines` element-wise.
1327: 
1328:     Parameters
1329:     ----------
1330:     a : array_like of str or unicode
1331: 
1332:     keepends : bool, optional
1333:         Line breaks are not included in the resulting list unless
1334:         keepends is given and true.
1335: 
1336:     Returns
1337:     -------
1338:     out : ndarray
1339:         Array of list objects
1340: 
1341:     See also
1342:     --------
1343:     str.splitlines
1344: 
1345:     '''
1346:     return _vec_string(
1347:         a, object_, 'splitlines', _clean_args(keepends))
1348: 
1349: 
1350: def startswith(a, prefix, start=0, end=None):
1351:     '''
1352:     Returns a boolean array which is `True` where the string element
1353:     in `a` starts with `prefix`, otherwise `False`.
1354: 
1355:     Calls `str.startswith` element-wise.
1356: 
1357:     Parameters
1358:     ----------
1359:     a : array_like of str or unicode
1360: 
1361:     prefix : str
1362: 
1363:     start, end : int, optional
1364:         With optional `start`, test beginning at that position. With
1365:         optional `end`, stop comparing at that position.
1366: 
1367:     Returns
1368:     -------
1369:     out : ndarray
1370:         Array of booleans
1371: 
1372:     See also
1373:     --------
1374:     str.startswith
1375: 
1376:     '''
1377:     return _vec_string(
1378:         a, bool_, 'startswith', [prefix, start] + _clean_args(end))
1379: 
1380: 
1381: def strip(a, chars=None):
1382:     '''
1383:     For each element in `a`, return a copy with the leading and
1384:     trailing characters removed.
1385: 
1386:     Calls `str.rstrip` element-wise.
1387: 
1388:     Parameters
1389:     ----------
1390:     a : array-like of str or unicode
1391: 
1392:     chars : str or unicode, optional
1393:        The `chars` argument is a string specifying the set of
1394:        characters to be removed. If omitted or None, the `chars`
1395:        argument defaults to removing whitespace. The `chars` argument
1396:        is not a prefix or suffix; rather, all combinations of its
1397:        values are stripped.
1398: 
1399:     Returns
1400:     -------
1401:     out : ndarray
1402:         Output array of str or unicode, depending on input type
1403: 
1404:     See also
1405:     --------
1406:     str.strip
1407: 
1408:     Examples
1409:     --------
1410:     >>> c = np.array(['aAaAaA', '  aA  ', 'abBABba'])
1411:     >>> c
1412:     array(['aAaAaA', '  aA  ', 'abBABba'],
1413:         dtype='|S7')
1414:     >>> np.char.strip(c)
1415:     array(['aAaAaA', 'aA', 'abBABba'],
1416:         dtype='|S7')
1417:     >>> np.char.strip(c, 'a') # 'a' unstripped from c[1] because whitespace leads
1418:     array(['AaAaA', '  aA  ', 'bBABb'],
1419:         dtype='|S7')
1420:     >>> np.char.strip(c, 'A') # 'A' unstripped from c[1] because (unprinted) ws trails
1421:     array(['aAaAa', '  aA  ', 'abBABba'],
1422:         dtype='|S7')
1423: 
1424:     '''
1425:     a_arr = numpy.asarray(a)
1426:     return _vec_string(a_arr, a_arr.dtype, 'strip', _clean_args(chars))
1427: 
1428: 
1429: def swapcase(a):
1430:     '''
1431:     Return element-wise a copy of the string with
1432:     uppercase characters converted to lowercase and vice versa.
1433: 
1434:     Calls `str.swapcase` element-wise.
1435: 
1436:     For 8-bit strings, this method is locale-dependent.
1437: 
1438:     Parameters
1439:     ----------
1440:     a : array_like, {str, unicode}
1441:         Input array.
1442: 
1443:     Returns
1444:     -------
1445:     out : ndarray, {str, unicode}
1446:         Output array of str or unicode, depending on input type
1447: 
1448:     See also
1449:     --------
1450:     str.swapcase
1451: 
1452:     Examples
1453:     --------
1454:     >>> c=np.array(['a1B c','1b Ca','b Ca1','cA1b'],'S5'); c
1455:     array(['a1B c', '1b Ca', 'b Ca1', 'cA1b'],
1456:         dtype='|S5')
1457:     >>> np.char.swapcase(c)
1458:     array(['A1b C', '1B cA', 'B cA1', 'Ca1B'],
1459:         dtype='|S5')
1460: 
1461:     '''
1462:     a_arr = numpy.asarray(a)
1463:     return _vec_string(a_arr, a_arr.dtype, 'swapcase')
1464: 
1465: 
1466: def title(a):
1467:     '''
1468:     Return element-wise title cased version of string or unicode.
1469: 
1470:     Title case words start with uppercase characters, all remaining cased
1471:     characters are lowercase.
1472: 
1473:     Calls `str.title` element-wise.
1474: 
1475:     For 8-bit strings, this method is locale-dependent.
1476: 
1477:     Parameters
1478:     ----------
1479:     a : array_like, {str, unicode}
1480:         Input array.
1481: 
1482:     Returns
1483:     -------
1484:     out : ndarray
1485:         Output array of str or unicode, depending on input type
1486: 
1487:     See also
1488:     --------
1489:     str.title
1490: 
1491:     Examples
1492:     --------
1493:     >>> c=np.array(['a1b c','1b ca','b ca1','ca1b'],'S5'); c
1494:     array(['a1b c', '1b ca', 'b ca1', 'ca1b'],
1495:         dtype='|S5')
1496:     >>> np.char.title(c)
1497:     array(['A1B C', '1B Ca', 'B Ca1', 'Ca1B'],
1498:         dtype='|S5')
1499: 
1500:     '''
1501:     a_arr = numpy.asarray(a)
1502:     return _vec_string(a_arr, a_arr.dtype, 'title')
1503: 
1504: 
1505: def translate(a, table, deletechars=None):
1506:     '''
1507:     For each element in `a`, return a copy of the string where all
1508:     characters occurring in the optional argument `deletechars` are
1509:     removed, and the remaining characters have been mapped through the
1510:     given translation table.
1511: 
1512:     Calls `str.translate` element-wise.
1513: 
1514:     Parameters
1515:     ----------
1516:     a : array-like of str or unicode
1517: 
1518:     table : str of length 256
1519: 
1520:     deletechars : str
1521: 
1522:     Returns
1523:     -------
1524:     out : ndarray
1525:         Output array of str or unicode, depending on input type
1526: 
1527:     See also
1528:     --------
1529:     str.translate
1530: 
1531:     '''
1532:     a_arr = numpy.asarray(a)
1533:     if issubclass(a_arr.dtype.type, unicode_):
1534:         return _vec_string(
1535:             a_arr, a_arr.dtype, 'translate', (table,))
1536:     else:
1537:         return _vec_string(
1538:             a_arr, a_arr.dtype, 'translate', [table] + _clean_args(deletechars))
1539: 
1540: 
1541: def upper(a):
1542:     '''
1543:     Return an array with the elements converted to uppercase.
1544: 
1545:     Calls `str.upper` element-wise.
1546: 
1547:     For 8-bit strings, this method is locale-dependent.
1548: 
1549:     Parameters
1550:     ----------
1551:     a : array_like, {str, unicode}
1552:         Input array.
1553: 
1554:     Returns
1555:     -------
1556:     out : ndarray, {str, unicode}
1557:         Output array of str or unicode, depending on input type
1558: 
1559:     See also
1560:     --------
1561:     str.upper
1562: 
1563:     Examples
1564:     --------
1565:     >>> c = np.array(['a1b c', '1bca', 'bca1']); c
1566:     array(['a1b c', '1bca', 'bca1'],
1567:         dtype='|S5')
1568:     >>> np.char.upper(c)
1569:     array(['A1B C', '1BCA', 'BCA1'],
1570:         dtype='|S5')
1571: 
1572:     '''
1573:     a_arr = numpy.asarray(a)
1574:     return _vec_string(a_arr, a_arr.dtype, 'upper')
1575: 
1576: 
1577: def zfill(a, width):
1578:     '''
1579:     Return the numeric string left-filled with zeros
1580: 
1581:     Calls `str.zfill` element-wise.
1582: 
1583:     Parameters
1584:     ----------
1585:     a : array_like, {str, unicode}
1586:         Input array.
1587:     width : int
1588:         Width of string to left-fill elements in `a`.
1589: 
1590:     Returns
1591:     -------
1592:     out : ndarray, {str, unicode}
1593:         Output array of str or unicode, depending on input type
1594: 
1595:     See also
1596:     --------
1597:     str.zfill
1598: 
1599:     '''
1600:     a_arr = numpy.asarray(a)
1601:     width_arr = numpy.asarray(width)
1602:     size = long(numpy.max(width_arr.flat))
1603:     return _vec_string(
1604:         a_arr, (a_arr.dtype.type, size), 'zfill', (width_arr,))
1605: 
1606: 
1607: def isnumeric(a):
1608:     '''
1609:     For each element, return True if there are only numeric
1610:     characters in the element.
1611: 
1612:     Calls `unicode.isnumeric` element-wise.
1613: 
1614:     Numeric characters include digit characters, and all characters
1615:     that have the Unicode numeric value property, e.g. ``U+2155,
1616:     VULGAR FRACTION ONE FIFTH``.
1617: 
1618:     Parameters
1619:     ----------
1620:     a : array_like, unicode
1621:         Input array.
1622: 
1623:     Returns
1624:     -------
1625:     out : ndarray, bool
1626:         Array of booleans of same shape as `a`.
1627: 
1628:     See also
1629:     --------
1630:     unicode.isnumeric
1631: 
1632:     '''
1633:     if _use_unicode(a) != unicode_:
1634:         raise TypeError("isnumeric is only available for Unicode strings and arrays")
1635:     return _vec_string(a, bool_, 'isnumeric')
1636: 
1637: 
1638: def isdecimal(a):
1639:     '''
1640:     For each element, return True if there are only decimal
1641:     characters in the element.
1642: 
1643:     Calls `unicode.isdecimal` element-wise.
1644: 
1645:     Decimal characters include digit characters, and all characters
1646:     that that can be used to form decimal-radix numbers,
1647:     e.g. ``U+0660, ARABIC-INDIC DIGIT ZERO``.
1648: 
1649:     Parameters
1650:     ----------
1651:     a : array_like, unicode
1652:         Input array.
1653: 
1654:     Returns
1655:     -------
1656:     out : ndarray, bool
1657:         Array of booleans identical in shape to `a`.
1658: 
1659:     See also
1660:     --------
1661:     unicode.isdecimal
1662: 
1663:     '''
1664:     if _use_unicode(a) != unicode_:
1665:         raise TypeError("isnumeric is only available for Unicode strings and arrays")
1666:     return _vec_string(a, bool_, 'isdecimal')
1667: 
1668: 
1669: class chararray(ndarray):
1670:     '''
1671:     chararray(shape, itemsize=1, unicode=False, buffer=None, offset=0,
1672:               strides=None, order=None)
1673: 
1674:     Provides a convenient view on arrays of string and unicode values.
1675: 
1676:     .. note::
1677:        The `chararray` class exists for backwards compatibility with
1678:        Numarray, it is not recommended for new development. Starting from numpy
1679:        1.4, if one needs arrays of strings, it is recommended to use arrays of
1680:        `dtype` `object_`, `string_` or `unicode_`, and use the free functions
1681:        in the `numpy.char` module for fast vectorized string operations.
1682: 
1683:     Versus a regular Numpy array of type `str` or `unicode`, this
1684:     class adds the following functionality:
1685: 
1686:       1) values automatically have whitespace removed from the end
1687:          when indexed
1688: 
1689:       2) comparison operators automatically remove whitespace from the
1690:          end when comparing values
1691: 
1692:       3) vectorized string operations are provided as methods
1693:          (e.g. `.endswith`) and infix operators (e.g. ``"+", "*", "%"``)
1694: 
1695:     chararrays should be created using `numpy.char.array` or
1696:     `numpy.char.asarray`, rather than this constructor directly.
1697: 
1698:     This constructor creates the array, using `buffer` (with `offset`
1699:     and `strides`) if it is not ``None``. If `buffer` is ``None``, then
1700:     constructs a new array with `strides` in "C order", unless both
1701:     ``len(shape) >= 2`` and ``order='Fortran'``, in which case `strides`
1702:     is in "Fortran order".
1703: 
1704:     Methods
1705:     -------
1706:     astype
1707:     argsort
1708:     copy
1709:     count
1710:     decode
1711:     dump
1712:     dumps
1713:     encode
1714:     endswith
1715:     expandtabs
1716:     fill
1717:     find
1718:     flatten
1719:     getfield
1720:     index
1721:     isalnum
1722:     isalpha
1723:     isdecimal
1724:     isdigit
1725:     islower
1726:     isnumeric
1727:     isspace
1728:     istitle
1729:     isupper
1730:     item
1731:     join
1732:     ljust
1733:     lower
1734:     lstrip
1735:     nonzero
1736:     put
1737:     ravel
1738:     repeat
1739:     replace
1740:     reshape
1741:     resize
1742:     rfind
1743:     rindex
1744:     rjust
1745:     rsplit
1746:     rstrip
1747:     searchsorted
1748:     setfield
1749:     setflags
1750:     sort
1751:     split
1752:     splitlines
1753:     squeeze
1754:     startswith
1755:     strip
1756:     swapaxes
1757:     swapcase
1758:     take
1759:     title
1760:     tofile
1761:     tolist
1762:     tostring
1763:     translate
1764:     transpose
1765:     upper
1766:     view
1767:     zfill
1768: 
1769:     Parameters
1770:     ----------
1771:     shape : tuple
1772:         Shape of the array.
1773:     itemsize : int, optional
1774:         Length of each array element, in number of characters. Default is 1.
1775:     unicode : bool, optional
1776:         Are the array elements of type unicode (True) or string (False).
1777:         Default is False.
1778:     buffer : int, optional
1779:         Memory address of the start of the array data.  Default is None,
1780:         in which case a new array is created.
1781:     offset : int, optional
1782:         Fixed stride displacement from the beginning of an axis?
1783:         Default is 0. Needs to be >=0.
1784:     strides : array_like of ints, optional
1785:         Strides for the array (see `ndarray.strides` for full description).
1786:         Default is None.
1787:     order : {'C', 'F'}, optional
1788:         The order in which the array data is stored in memory: 'C' ->
1789:         "row major" order (the default), 'F' -> "column major"
1790:         (Fortran) order.
1791: 
1792:     Examples
1793:     --------
1794:     >>> charar = np.chararray((3, 3))
1795:     >>> charar[:] = 'a'
1796:     >>> charar
1797:     chararray([['a', 'a', 'a'],
1798:            ['a', 'a', 'a'],
1799:            ['a', 'a', 'a']],
1800:           dtype='|S1')
1801: 
1802:     >>> charar = np.chararray(charar.shape, itemsize=5)
1803:     >>> charar[:] = 'abc'
1804:     >>> charar
1805:     chararray([['abc', 'abc', 'abc'],
1806:            ['abc', 'abc', 'abc'],
1807:            ['abc', 'abc', 'abc']],
1808:           dtype='|S5')
1809: 
1810:     '''
1811:     def __new__(subtype, shape, itemsize=1, unicode=False, buffer=None,
1812:                 offset=0, strides=None, order='C'):
1813:         global _globalvar
1814: 
1815:         if unicode:
1816:             dtype = unicode_
1817:         else:
1818:             dtype = string_
1819: 
1820:         # force itemsize to be a Python long, since using Numpy integer
1821:         # types results in itemsize.itemsize being used as the size of
1822:         # strings in the new array.
1823:         itemsize = long(itemsize)
1824: 
1825:         if sys.version_info[0] >= 3 and isinstance(buffer, _unicode):
1826:             # On Py3, unicode objects do not have the buffer interface
1827:             filler = buffer
1828:             buffer = None
1829:         else:
1830:             filler = None
1831: 
1832:         _globalvar = 1
1833:         if buffer is None:
1834:             self = ndarray.__new__(subtype, shape, (dtype, itemsize),
1835:                                    order=order)
1836:         else:
1837:             self = ndarray.__new__(subtype, shape, (dtype, itemsize),
1838:                                    buffer=buffer,
1839:                                    offset=offset, strides=strides,
1840:                                    order=order)
1841:         if filler is not None:
1842:             self[...] = filler
1843:         _globalvar = 0
1844:         return self
1845: 
1846:     def __array_finalize__(self, obj):
1847:         # The b is a special case because it is used for reconstructing.
1848:         if not _globalvar and self.dtype.char not in 'SUbc':
1849:             raise ValueError("Can only create a chararray from string data.")
1850: 
1851:     def __getitem__(self, obj):
1852:         val = ndarray.__getitem__(self, obj)
1853: 
1854:         if isinstance(val, character):
1855:             temp = val.rstrip()
1856:             if _len(temp) == 0:
1857:                 val = ''
1858:             else:
1859:                 val = temp
1860: 
1861:         return val
1862: 
1863:     # IMPLEMENTATION NOTE: Most of the methods of this class are
1864:     # direct delegations to the free functions in this module.
1865:     # However, those that return an array of strings should instead
1866:     # return a chararray, so some extra wrapping is required.
1867: 
1868:     def __eq__(self, other):
1869:         '''
1870:         Return (self == other) element-wise.
1871: 
1872:         See also
1873:         --------
1874:         equal
1875:         '''
1876:         return equal(self, other)
1877: 
1878:     def __ne__(self, other):
1879:         '''
1880:         Return (self != other) element-wise.
1881: 
1882:         See also
1883:         --------
1884:         not_equal
1885:         '''
1886:         return not_equal(self, other)
1887: 
1888:     def __ge__(self, other):
1889:         '''
1890:         Return (self >= other) element-wise.
1891: 
1892:         See also
1893:         --------
1894:         greater_equal
1895:         '''
1896:         return greater_equal(self, other)
1897: 
1898:     def __le__(self, other):
1899:         '''
1900:         Return (self <= other) element-wise.
1901: 
1902:         See also
1903:         --------
1904:         less_equal
1905:         '''
1906:         return less_equal(self, other)
1907: 
1908:     def __gt__(self, other):
1909:         '''
1910:         Return (self > other) element-wise.
1911: 
1912:         See also
1913:         --------
1914:         greater
1915:         '''
1916:         return greater(self, other)
1917: 
1918:     def __lt__(self, other):
1919:         '''
1920:         Return (self < other) element-wise.
1921: 
1922:         See also
1923:         --------
1924:         less
1925:         '''
1926:         return less(self, other)
1927: 
1928:     def __add__(self, other):
1929:         '''
1930:         Return (self + other), that is string concatenation,
1931:         element-wise for a pair of array_likes of str or unicode.
1932: 
1933:         See also
1934:         --------
1935:         add
1936:         '''
1937:         return asarray(add(self, other))
1938: 
1939:     def __radd__(self, other):
1940:         '''
1941:         Return (other + self), that is string concatenation,
1942:         element-wise for a pair of array_likes of `string_` or `unicode_`.
1943: 
1944:         See also
1945:         --------
1946:         add
1947:         '''
1948:         return asarray(add(numpy.asarray(other), self))
1949: 
1950:     def __mul__(self, i):
1951:         '''
1952:         Return (self * i), that is string multiple concatenation,
1953:         element-wise.
1954: 
1955:         See also
1956:         --------
1957:         multiply
1958:         '''
1959:         return asarray(multiply(self, i))
1960: 
1961:     def __rmul__(self, i):
1962:         '''
1963:         Return (self * i), that is string multiple concatenation,
1964:         element-wise.
1965: 
1966:         See also
1967:         --------
1968:         multiply
1969:         '''
1970:         return asarray(multiply(self, i))
1971: 
1972:     def __mod__(self, i):
1973:         '''
1974:         Return (self % i), that is pre-Python 2.6 string formatting
1975:         (iterpolation), element-wise for a pair of array_likes of `string_`
1976:         or `unicode_`.
1977: 
1978:         See also
1979:         --------
1980:         mod
1981:         '''
1982:         return asarray(mod(self, i))
1983: 
1984:     def __rmod__(self, other):
1985:         return NotImplemented
1986: 
1987:     def argsort(self, axis=-1, kind='quicksort', order=None):
1988:         '''
1989:         Return the indices that sort the array lexicographically.
1990: 
1991:         For full documentation see `numpy.argsort`, for which this method is
1992:         in fact merely a "thin wrapper."
1993: 
1994:         Examples
1995:         --------
1996:         >>> c = np.array(['a1b c', '1b ca', 'b ca1', 'Ca1b'], 'S5')
1997:         >>> c = c.view(np.chararray); c
1998:         chararray(['a1b c', '1b ca', 'b ca1', 'Ca1b'],
1999:               dtype='|S5')
2000:         >>> c[c.argsort()]
2001:         chararray(['1b ca', 'Ca1b', 'a1b c', 'b ca1'],
2002:               dtype='|S5')
2003: 
2004:         '''
2005:         return self.__array__().argsort(axis, kind, order)
2006:     argsort.__doc__ = ndarray.argsort.__doc__
2007: 
2008:     def capitalize(self):
2009:         '''
2010:         Return a copy of `self` with only the first character of each element
2011:         capitalized.
2012: 
2013:         See also
2014:         --------
2015:         char.capitalize
2016: 
2017:         '''
2018:         return asarray(capitalize(self))
2019: 
2020:     def center(self, width, fillchar=' '):
2021:         '''
2022:         Return a copy of `self` with its elements centered in a
2023:         string of length `width`.
2024: 
2025:         See also
2026:         --------
2027:         center
2028:         '''
2029:         return asarray(center(self, width, fillchar))
2030: 
2031:     def count(self, sub, start=0, end=None):
2032:         '''
2033:         Returns an array with the number of non-overlapping occurrences of
2034:         substring `sub` in the range [`start`, `end`].
2035: 
2036:         See also
2037:         --------
2038:         char.count
2039: 
2040:         '''
2041:         return count(self, sub, start, end)
2042: 
2043:     def decode(self, encoding=None, errors=None):
2044:         '''
2045:         Calls `str.decode` element-wise.
2046: 
2047:         See also
2048:         --------
2049:         char.decode
2050: 
2051:         '''
2052:         return decode(self, encoding, errors)
2053: 
2054:     def encode(self, encoding=None, errors=None):
2055:         '''
2056:         Calls `str.encode` element-wise.
2057: 
2058:         See also
2059:         --------
2060:         char.encode
2061: 
2062:         '''
2063:         return encode(self, encoding, errors)
2064: 
2065:     def endswith(self, suffix, start=0, end=None):
2066:         '''
2067:         Returns a boolean array which is `True` where the string element
2068:         in `self` ends with `suffix`, otherwise `False`.
2069: 
2070:         See also
2071:         --------
2072:         char.endswith
2073: 
2074:         '''
2075:         return endswith(self, suffix, start, end)
2076: 
2077:     def expandtabs(self, tabsize=8):
2078:         '''
2079:         Return a copy of each string element where all tab characters are
2080:         replaced by one or more spaces.
2081: 
2082:         See also
2083:         --------
2084:         char.expandtabs
2085: 
2086:         '''
2087:         return asarray(expandtabs(self, tabsize))
2088: 
2089:     def find(self, sub, start=0, end=None):
2090:         '''
2091:         For each element, return the lowest index in the string where
2092:         substring `sub` is found.
2093: 
2094:         See also
2095:         --------
2096:         char.find
2097: 
2098:         '''
2099:         return find(self, sub, start, end)
2100: 
2101:     def index(self, sub, start=0, end=None):
2102:         '''
2103:         Like `find`, but raises `ValueError` when the substring is not found.
2104: 
2105:         See also
2106:         --------
2107:         char.index
2108: 
2109:         '''
2110:         return index(self, sub, start, end)
2111: 
2112:     def isalnum(self):
2113:         '''
2114:         Returns true for each element if all characters in the string
2115:         are alphanumeric and there is at least one character, false
2116:         otherwise.
2117: 
2118:         See also
2119:         --------
2120:         char.isalnum
2121: 
2122:         '''
2123:         return isalnum(self)
2124: 
2125:     def isalpha(self):
2126:         '''
2127:         Returns true for each element if all characters in the string
2128:         are alphabetic and there is at least one character, false
2129:         otherwise.
2130: 
2131:         See also
2132:         --------
2133:         char.isalpha
2134: 
2135:         '''
2136:         return isalpha(self)
2137: 
2138:     def isdigit(self):
2139:         '''
2140:         Returns true for each element if all characters in the string are
2141:         digits and there is at least one character, false otherwise.
2142: 
2143:         See also
2144:         --------
2145:         char.isdigit
2146: 
2147:         '''
2148:         return isdigit(self)
2149: 
2150:     def islower(self):
2151:         '''
2152:         Returns true for each element if all cased characters in the
2153:         string are lowercase and there is at least one cased character,
2154:         false otherwise.
2155: 
2156:         See also
2157:         --------
2158:         char.islower
2159: 
2160:         '''
2161:         return islower(self)
2162: 
2163:     def isspace(self):
2164:         '''
2165:         Returns true for each element if there are only whitespace
2166:         characters in the string and there is at least one character,
2167:         false otherwise.
2168: 
2169:         See also
2170:         --------
2171:         char.isspace
2172: 
2173:         '''
2174:         return isspace(self)
2175: 
2176:     def istitle(self):
2177:         '''
2178:         Returns true for each element if the element is a titlecased
2179:         string and there is at least one character, false otherwise.
2180: 
2181:         See also
2182:         --------
2183:         char.istitle
2184: 
2185:         '''
2186:         return istitle(self)
2187: 
2188:     def isupper(self):
2189:         '''
2190:         Returns true for each element if all cased characters in the
2191:         string are uppercase and there is at least one character, false
2192:         otherwise.
2193: 
2194:         See also
2195:         --------
2196:         char.isupper
2197: 
2198:         '''
2199:         return isupper(self)
2200: 
2201:     def join(self, seq):
2202:         '''
2203:         Return a string which is the concatenation of the strings in the
2204:         sequence `seq`.
2205: 
2206:         See also
2207:         --------
2208:         char.join
2209: 
2210:         '''
2211:         return join(self, seq)
2212: 
2213:     def ljust(self, width, fillchar=' '):
2214:         '''
2215:         Return an array with the elements of `self` left-justified in a
2216:         string of length `width`.
2217: 
2218:         See also
2219:         --------
2220:         char.ljust
2221: 
2222:         '''
2223:         return asarray(ljust(self, width, fillchar))
2224: 
2225:     def lower(self):
2226:         '''
2227:         Return an array with the elements of `self` converted to
2228:         lowercase.
2229: 
2230:         See also
2231:         --------
2232:         char.lower
2233: 
2234:         '''
2235:         return asarray(lower(self))
2236: 
2237:     def lstrip(self, chars=None):
2238:         '''
2239:         For each element in `self`, return a copy with the leading characters
2240:         removed.
2241: 
2242:         See also
2243:         --------
2244:         char.lstrip
2245: 
2246:         '''
2247:         return asarray(lstrip(self, chars))
2248: 
2249:     def partition(self, sep):
2250:         '''
2251:         Partition each element in `self` around `sep`.
2252: 
2253:         See also
2254:         --------
2255:         partition
2256:         '''
2257:         return asarray(partition(self, sep))
2258: 
2259:     def replace(self, old, new, count=None):
2260:         '''
2261:         For each element in `self`, return a copy of the string with all
2262:         occurrences of substring `old` replaced by `new`.
2263: 
2264:         See also
2265:         --------
2266:         char.replace
2267: 
2268:         '''
2269:         return asarray(replace(self, old, new, count))
2270: 
2271:     def rfind(self, sub, start=0, end=None):
2272:         '''
2273:         For each element in `self`, return the highest index in the string
2274:         where substring `sub` is found, such that `sub` is contained
2275:         within [`start`, `end`].
2276: 
2277:         See also
2278:         --------
2279:         char.rfind
2280: 
2281:         '''
2282:         return rfind(self, sub, start, end)
2283: 
2284:     def rindex(self, sub, start=0, end=None):
2285:         '''
2286:         Like `rfind`, but raises `ValueError` when the substring `sub` is
2287:         not found.
2288: 
2289:         See also
2290:         --------
2291:         char.rindex
2292: 
2293:         '''
2294:         return rindex(self, sub, start, end)
2295: 
2296:     def rjust(self, width, fillchar=' '):
2297:         '''
2298:         Return an array with the elements of `self`
2299:         right-justified in a string of length `width`.
2300: 
2301:         See also
2302:         --------
2303:         char.rjust
2304: 
2305:         '''
2306:         return asarray(rjust(self, width, fillchar))
2307: 
2308:     def rpartition(self, sep):
2309:         '''
2310:         Partition each element in `self` around `sep`.
2311: 
2312:         See also
2313:         --------
2314:         rpartition
2315:         '''
2316:         return asarray(rpartition(self, sep))
2317: 
2318:     def rsplit(self, sep=None, maxsplit=None):
2319:         '''
2320:         For each element in `self`, return a list of the words in
2321:         the string, using `sep` as the delimiter string.
2322: 
2323:         See also
2324:         --------
2325:         char.rsplit
2326: 
2327:         '''
2328:         return rsplit(self, sep, maxsplit)
2329: 
2330:     def rstrip(self, chars=None):
2331:         '''
2332:         For each element in `self`, return a copy with the trailing
2333:         characters removed.
2334: 
2335:         See also
2336:         --------
2337:         char.rstrip
2338: 
2339:         '''
2340:         return asarray(rstrip(self, chars))
2341: 
2342:     def split(self, sep=None, maxsplit=None):
2343:         '''
2344:         For each element in `self`, return a list of the words in the
2345:         string, using `sep` as the delimiter string.
2346: 
2347:         See also
2348:         --------
2349:         char.split
2350: 
2351:         '''
2352:         return split(self, sep, maxsplit)
2353: 
2354:     def splitlines(self, keepends=None):
2355:         '''
2356:         For each element in `self`, return a list of the lines in the
2357:         element, breaking at line boundaries.
2358: 
2359:         See also
2360:         --------
2361:         char.splitlines
2362: 
2363:         '''
2364:         return splitlines(self, keepends)
2365: 
2366:     def startswith(self, prefix, start=0, end=None):
2367:         '''
2368:         Returns a boolean array which is `True` where the string element
2369:         in `self` starts with `prefix`, otherwise `False`.
2370: 
2371:         See also
2372:         --------
2373:         char.startswith
2374: 
2375:         '''
2376:         return startswith(self, prefix, start, end)
2377: 
2378:     def strip(self, chars=None):
2379:         '''
2380:         For each element in `self`, return a copy with the leading and
2381:         trailing characters removed.
2382: 
2383:         See also
2384:         --------
2385:         char.strip
2386: 
2387:         '''
2388:         return asarray(strip(self, chars))
2389: 
2390:     def swapcase(self):
2391:         '''
2392:         For each element in `self`, return a copy of the string with
2393:         uppercase characters converted to lowercase and vice versa.
2394: 
2395:         See also
2396:         --------
2397:         char.swapcase
2398: 
2399:         '''
2400:         return asarray(swapcase(self))
2401: 
2402:     def title(self):
2403:         '''
2404:         For each element in `self`, return a titlecased version of the
2405:         string: words start with uppercase characters, all remaining cased
2406:         characters are lowercase.
2407: 
2408:         See also
2409:         --------
2410:         char.title
2411: 
2412:         '''
2413:         return asarray(title(self))
2414: 
2415:     def translate(self, table, deletechars=None):
2416:         '''
2417:         For each element in `self`, return a copy of the string where
2418:         all characters occurring in the optional argument
2419:         `deletechars` are removed, and the remaining characters have
2420:         been mapped through the given translation table.
2421: 
2422:         See also
2423:         --------
2424:         char.translate
2425: 
2426:         '''
2427:         return asarray(translate(self, table, deletechars))
2428: 
2429:     def upper(self):
2430:         '''
2431:         Return an array with the elements of `self` converted to
2432:         uppercase.
2433: 
2434:         See also
2435:         --------
2436:         char.upper
2437: 
2438:         '''
2439:         return asarray(upper(self))
2440: 
2441:     def zfill(self, width):
2442:         '''
2443:         Return the numeric string left-filled with zeros in a string of
2444:         length `width`.
2445: 
2446:         See also
2447:         --------
2448:         char.zfill
2449: 
2450:         '''
2451:         return asarray(zfill(self, width))
2452: 
2453:     def isnumeric(self):
2454:         '''
2455:         For each element in `self`, return True if there are only
2456:         numeric characters in the element.
2457: 
2458:         See also
2459:         --------
2460:         char.isnumeric
2461: 
2462:         '''
2463:         return isnumeric(self)
2464: 
2465:     def isdecimal(self):
2466:         '''
2467:         For each element in `self`, return True if there are only
2468:         decimal characters in the element.
2469: 
2470:         See also
2471:         --------
2472:         char.isdecimal
2473: 
2474:         '''
2475:         return isdecimal(self)
2476: 
2477: 
2478: def array(obj, itemsize=None, copy=True, unicode=None, order=None):
2479:     '''
2480:     Create a `chararray`.
2481: 
2482:     .. note::
2483:        This class is provided for numarray backward-compatibility.
2484:        New code (not concerned with numarray compatibility) should use
2485:        arrays of type `string_` or `unicode_` and use the free functions
2486:        in :mod:`numpy.char <numpy.core.defchararray>` for fast
2487:        vectorized string operations instead.
2488: 
2489:     Versus a regular Numpy array of type `str` or `unicode`, this
2490:     class adds the following functionality:
2491: 
2492:       1) values automatically have whitespace removed from the end
2493:          when indexed
2494: 
2495:       2) comparison operators automatically remove whitespace from the
2496:          end when comparing values
2497: 
2498:       3) vectorized string operations are provided as methods
2499:          (e.g. `str.endswith`) and infix operators (e.g. ``+, *, %``)
2500: 
2501:     Parameters
2502:     ----------
2503:     obj : array of str or unicode-like
2504: 
2505:     itemsize : int, optional
2506:         `itemsize` is the number of characters per scalar in the
2507:         resulting array.  If `itemsize` is None, and `obj` is an
2508:         object array or a Python list, the `itemsize` will be
2509:         automatically determined.  If `itemsize` is provided and `obj`
2510:         is of type str or unicode, then the `obj` string will be
2511:         chunked into `itemsize` pieces.
2512: 
2513:     copy : bool, optional
2514:         If true (default), then the object is copied.  Otherwise, a copy
2515:         will only be made if __array__ returns a copy, if obj is a
2516:         nested sequence, or if a copy is needed to satisfy any of the other
2517:         requirements (`itemsize`, unicode, `order`, etc.).
2518: 
2519:     unicode : bool, optional
2520:         When true, the resulting `chararray` can contain Unicode
2521:         characters, when false only 8-bit characters.  If unicode is
2522:         `None` and `obj` is one of the following:
2523: 
2524:           - a `chararray`,
2525:           - an ndarray of type `str` or `unicode`
2526:           - a Python str or unicode object,
2527: 
2528:         then the unicode setting of the output array will be
2529:         automatically determined.
2530: 
2531:     order : {'C', 'F', 'A'}, optional
2532:         Specify the order of the array.  If order is 'C' (default), then the
2533:         array will be in C-contiguous order (last-index varies the
2534:         fastest).  If order is 'F', then the returned array
2535:         will be in Fortran-contiguous order (first-index varies the
2536:         fastest).  If order is 'A', then the returned array may
2537:         be in any order (either C-, Fortran-contiguous, or even
2538:         discontiguous).
2539:     '''
2540:     if isinstance(obj, (_bytes, _unicode)):
2541:         if unicode is None:
2542:             if isinstance(obj, _unicode):
2543:                 unicode = True
2544:             else:
2545:                 unicode = False
2546: 
2547:         if itemsize is None:
2548:             itemsize = _len(obj)
2549:         shape = _len(obj) // itemsize
2550: 
2551:         if unicode:
2552:             if sys.maxunicode == 0xffff:
2553:                 # On a narrow Python build, the buffer for Unicode
2554:                 # strings is UCS2, which doesn't match the buffer for
2555:                 # Numpy Unicode types, which is ALWAYS UCS4.
2556:                 # Therefore, we need to convert the buffer.  On Python
2557:                 # 2.6 and later, we can use the utf_32 codec.  Earlier
2558:                 # versions don't have that codec, so we convert to a
2559:                 # numerical array that matches the input buffer, and
2560:                 # then use Numpy to convert it to UCS4.  All of this
2561:                 # should happen in native endianness.
2562:                 if sys.hexversion >= 0x2060000:
2563:                     obj = obj.encode('utf_32')
2564:                 else:
2565:                     if isinstance(obj, str):
2566:                         ascii = numpy.frombuffer(obj, 'u1')
2567:                         ucs4 = numpy.array(ascii, 'u4')
2568:                         obj = ucs4.data
2569:                     else:
2570:                         ucs2 = numpy.frombuffer(obj, 'u2')
2571:                         ucs4 = numpy.array(ucs2, 'u4')
2572:                         obj = ucs4.data
2573:             else:
2574:                 obj = _unicode(obj)
2575:         else:
2576:             # Let the default Unicode -> string encoding (if any) take
2577:             # precedence.
2578:             obj = _bytes(obj)
2579: 
2580:         return chararray(shape, itemsize=itemsize, unicode=unicode,
2581:                          buffer=obj, order=order)
2582: 
2583:     if isinstance(obj, (list, tuple)):
2584:         obj = numpy.asarray(obj)
2585: 
2586:     if isinstance(obj, ndarray) and issubclass(obj.dtype.type, character):
2587:         # If we just have a vanilla chararray, create a chararray
2588:         # view around it.
2589:         if not isinstance(obj, chararray):
2590:             obj = obj.view(chararray)
2591: 
2592:         if itemsize is None:
2593:             itemsize = obj.itemsize
2594:             # itemsize is in 8-bit chars, so for Unicode, we need
2595:             # to divide by the size of a single Unicode character,
2596:             # which for Numpy is always 4
2597:             if issubclass(obj.dtype.type, unicode_):
2598:                 itemsize //= 4
2599: 
2600:         if unicode is None:
2601:             if issubclass(obj.dtype.type, unicode_):
2602:                 unicode = True
2603:             else:
2604:                 unicode = False
2605: 
2606:         if unicode:
2607:             dtype = unicode_
2608:         else:
2609:             dtype = string_
2610: 
2611:         if order is not None:
2612:             obj = numpy.asarray(obj, order=order)
2613:         if (copy or
2614:                 (itemsize != obj.itemsize) or
2615:                 (not unicode and isinstance(obj, unicode_)) or
2616:                 (unicode and isinstance(obj, string_))):
2617:             obj = obj.astype((dtype, long(itemsize)))
2618:         return obj
2619: 
2620:     if isinstance(obj, ndarray) and issubclass(obj.dtype.type, object):
2621:         if itemsize is None:
2622:             # Since no itemsize was specified, convert the input array to
2623:             # a list so the ndarray constructor will automatically
2624:             # determine the itemsize for us.
2625:             obj = obj.tolist()
2626:             # Fall through to the default case
2627: 
2628:     if unicode:
2629:         dtype = unicode_
2630:     else:
2631:         dtype = string_
2632: 
2633:     if itemsize is None:
2634:         val = narray(obj, dtype=dtype, order=order, subok=True)
2635:     else:
2636:         val = narray(obj, dtype=(dtype, itemsize), order=order, subok=True)
2637:     return val.view(chararray)
2638: 
2639: 
2640: def asarray(obj, itemsize=None, unicode=None, order=None):
2641:     '''
2642:     Convert the input to a `chararray`, copying the data only if
2643:     necessary.
2644: 
2645:     Versus a regular Numpy array of type `str` or `unicode`, this
2646:     class adds the following functionality:
2647: 
2648:       1) values automatically have whitespace removed from the end
2649:          when indexed
2650: 
2651:       2) comparison operators automatically remove whitespace from the
2652:          end when comparing values
2653: 
2654:       3) vectorized string operations are provided as methods
2655:          (e.g. `str.endswith`) and infix operators (e.g. ``+``, ``*``,``%``)
2656: 
2657:     Parameters
2658:     ----------
2659:     obj : array of str or unicode-like
2660: 
2661:     itemsize : int, optional
2662:         `itemsize` is the number of characters per scalar in the
2663:         resulting array.  If `itemsize` is None, and `obj` is an
2664:         object array or a Python list, the `itemsize` will be
2665:         automatically determined.  If `itemsize` is provided and `obj`
2666:         is of type str or unicode, then the `obj` string will be
2667:         chunked into `itemsize` pieces.
2668: 
2669:     unicode : bool, optional
2670:         When true, the resulting `chararray` can contain Unicode
2671:         characters, when false only 8-bit characters.  If unicode is
2672:         `None` and `obj` is one of the following:
2673: 
2674:           - a `chararray`,
2675:           - an ndarray of type `str` or 'unicode`
2676:           - a Python str or unicode object,
2677: 
2678:         then the unicode setting of the output array will be
2679:         automatically determined.
2680: 
2681:     order : {'C', 'F'}, optional
2682:         Specify the order of the array.  If order is 'C' (default), then the
2683:         array will be in C-contiguous order (last-index varies the
2684:         fastest).  If order is 'F', then the returned array
2685:         will be in Fortran-contiguous order (first-index varies the
2686:         fastest).
2687:     '''
2688:     return array(obj, itemsize, copy=False,
2689:                  unicode=unicode, order=order)
2690: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_2002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, (-1)), 'str', '\nThis module contains a set of functions for vectorized string\noperations and methods.\n\n.. note::\n   The `chararray` class exists for backwards compatibility with\n   Numarray, it is not recommended for new development. Starting from numpy\n   1.4, if one needs arrays of strings, it is recommended to use arrays of\n   `dtype` `object_`, `string_` or `unicode_`, and use the free functions\n   in the `numpy.char` module for fast vectorized string operations.\n\nSome methods will only be available if the corresponding string method is\navailable in your version of Python.\n\nThe preferred alias for `defchararray` is `numpy.char`.\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 20, 0))

# 'import sys' statement (line 20)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 21, 0))

# 'from numpy.core.numerictypes import string_, unicode_, integer, object_, bool_, character' statement (line 21)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/core/')
import_2003 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'numpy.core.numerictypes')

if (type(import_2003) is not StypyTypeError):

    if (import_2003 != 'pyd_module'):
        __import__(import_2003)
        sys_modules_2004 = sys.modules[import_2003]
        import_from_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'numpy.core.numerictypes', sys_modules_2004.module_type_store, module_type_store, ['string_', 'unicode_', 'integer', 'object_', 'bool_', 'character'])
        nest_module(stypy.reporting.localization.Localization(__file__, 21, 0), __file__, sys_modules_2004, sys_modules_2004.module_type_store, module_type_store)
    else:
        from numpy.core.numerictypes import string_, unicode_, integer, object_, bool_, character

        import_from_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'numpy.core.numerictypes', None, module_type_store, ['string_', 'unicode_', 'integer', 'object_', 'bool_', 'character'], [string_, unicode_, integer, object_, bool_, character])

else:
    # Assigning a type to the variable 'numpy.core.numerictypes' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'numpy.core.numerictypes', import_2003)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/core/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 22, 0))

# 'from numpy.core.numeric import ndarray, compare_chararrays' statement (line 22)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/core/')
import_2005 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'numpy.core.numeric')

if (type(import_2005) is not StypyTypeError):

    if (import_2005 != 'pyd_module'):
        __import__(import_2005)
        sys_modules_2006 = sys.modules[import_2005]
        import_from_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'numpy.core.numeric', sys_modules_2006.module_type_store, module_type_store, ['ndarray', 'compare_chararrays'])
        nest_module(stypy.reporting.localization.Localization(__file__, 22, 0), __file__, sys_modules_2006, sys_modules_2006.module_type_store, module_type_store)
    else:
        from numpy.core.numeric import ndarray, compare_chararrays

        import_from_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'numpy.core.numeric', None, module_type_store, ['ndarray', 'compare_chararrays'], [ndarray, compare_chararrays])

else:
    # Assigning a type to the variable 'numpy.core.numeric' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'numpy.core.numeric', import_2005)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/core/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 23, 0))

# 'from numpy.core.numeric import narray' statement (line 23)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/core/')
import_2007 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 23, 0), 'numpy.core.numeric')

if (type(import_2007) is not StypyTypeError):

    if (import_2007 != 'pyd_module'):
        __import__(import_2007)
        sys_modules_2008 = sys.modules[import_2007]
        import_from_module(stypy.reporting.localization.Localization(__file__, 23, 0), 'numpy.core.numeric', sys_modules_2008.module_type_store, module_type_store, ['array'])
        nest_module(stypy.reporting.localization.Localization(__file__, 23, 0), __file__, sys_modules_2008, sys_modules_2008.module_type_store, module_type_store)
    else:
        from numpy.core.numeric import array as narray

        import_from_module(stypy.reporting.localization.Localization(__file__, 23, 0), 'numpy.core.numeric', None, module_type_store, ['array'], [narray])

else:
    # Assigning a type to the variable 'numpy.core.numeric' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'numpy.core.numeric', import_2007)

# Adding an alias
module_type_store.add_alias('narray', 'array')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/core/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 24, 0))

# 'from numpy.core.multiarray import _vec_string' statement (line 24)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/core/')
import_2009 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'numpy.core.multiarray')

if (type(import_2009) is not StypyTypeError):

    if (import_2009 != 'pyd_module'):
        __import__(import_2009)
        sys_modules_2010 = sys.modules[import_2009]
        import_from_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'numpy.core.multiarray', sys_modules_2010.module_type_store, module_type_store, ['_vec_string'])
        nest_module(stypy.reporting.localization.Localization(__file__, 24, 0), __file__, sys_modules_2010, sys_modules_2010.module_type_store, module_type_store)
    else:
        from numpy.core.multiarray import _vec_string

        import_from_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'numpy.core.multiarray', None, module_type_store, ['_vec_string'], [_vec_string])

else:
    # Assigning a type to the variable 'numpy.core.multiarray' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'numpy.core.multiarray', import_2009)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/core/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 25, 0))

# 'from numpy.compat import asbytes, long' statement (line 25)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/core/')
import_2011 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 25, 0), 'numpy.compat')

if (type(import_2011) is not StypyTypeError):

    if (import_2011 != 'pyd_module'):
        __import__(import_2011)
        sys_modules_2012 = sys.modules[import_2011]
        import_from_module(stypy.reporting.localization.Localization(__file__, 25, 0), 'numpy.compat', sys_modules_2012.module_type_store, module_type_store, ['asbytes', 'long'])
        nest_module(stypy.reporting.localization.Localization(__file__, 25, 0), __file__, sys_modules_2012, sys_modules_2012.module_type_store, module_type_store)
    else:
        from numpy.compat import asbytes, long

        import_from_module(stypy.reporting.localization.Localization(__file__, 25, 0), 'numpy.compat', None, module_type_store, ['asbytes', 'long'], [asbytes, long])

else:
    # Assigning a type to the variable 'numpy.compat' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'numpy.compat', import_2011)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/core/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 26, 0))

# 'import numpy' statement (line 26)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/core/')
import_2013 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'numpy')

if (type(import_2013) is not StypyTypeError):

    if (import_2013 != 'pyd_module'):
        __import__(import_2013)
        sys_modules_2014 = sys.modules[import_2013]
        import_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'numpy', sys_modules_2014.module_type_store, module_type_store)
    else:
        import numpy

        import_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'numpy', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'numpy', import_2013)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/core/')


# Assigning a List to a Name (line 28):
__all__ = ['chararray', 'equal', 'not_equal', 'greater_equal', 'less_equal', 'greater', 'less', 'str_len', 'add', 'multiply', 'mod', 'capitalize', 'center', 'count', 'decode', 'encode', 'endswith', 'expandtabs', 'find', 'index', 'isalnum', 'isalpha', 'isdigit', 'islower', 'isspace', 'istitle', 'isupper', 'join', 'ljust', 'lower', 'lstrip', 'partition', 'replace', 'rfind', 'rindex', 'rjust', 'rpartition', 'rsplit', 'rstrip', 'split', 'splitlines', 'startswith', 'strip', 'swapcase', 'title', 'translate', 'upper', 'zfill', 'isnumeric', 'isdecimal', 'array', 'asarray']
module_type_store.set_exportable_members(['chararray', 'equal', 'not_equal', 'greater_equal', 'less_equal', 'greater', 'less', 'str_len', 'add', 'multiply', 'mod', 'capitalize', 'center', 'count', 'decode', 'encode', 'endswith', 'expandtabs', 'find', 'index', 'isalnum', 'isalpha', 'isdigit', 'islower', 'isspace', 'istitle', 'isupper', 'join', 'ljust', 'lower', 'lstrip', 'partition', 'replace', 'rfind', 'rindex', 'rjust', 'rpartition', 'rsplit', 'rstrip', 'split', 'splitlines', 'startswith', 'strip', 'swapcase', 'title', 'translate', 'upper', 'zfill', 'isnumeric', 'isdecimal', 'array', 'asarray'])

# Obtaining an instance of the builtin type 'list' (line 28)
list_2015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 28)
# Adding element type (line 28)
str_2016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 4), 'str', 'chararray')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 10), list_2015, str_2016)
# Adding element type (line 28)
str_2017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 17), 'str', 'equal')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 10), list_2015, str_2017)
# Adding element type (line 28)
str_2018 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 26), 'str', 'not_equal')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 10), list_2015, str_2018)
# Adding element type (line 28)
str_2019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 39), 'str', 'greater_equal')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 10), list_2015, str_2019)
# Adding element type (line 28)
str_2020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 56), 'str', 'less_equal')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 10), list_2015, str_2020)
# Adding element type (line 28)
str_2021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 4), 'str', 'greater')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 10), list_2015, str_2021)
# Adding element type (line 28)
str_2022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 15), 'str', 'less')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 10), list_2015, str_2022)
# Adding element type (line 28)
str_2023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 23), 'str', 'str_len')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 10), list_2015, str_2023)
# Adding element type (line 28)
str_2024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 34), 'str', 'add')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 10), list_2015, str_2024)
# Adding element type (line 28)
str_2025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 41), 'str', 'multiply')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 10), list_2015, str_2025)
# Adding element type (line 28)
str_2026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 53), 'str', 'mod')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 10), list_2015, str_2026)
# Adding element type (line 28)
str_2027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 60), 'str', 'capitalize')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 10), list_2015, str_2027)
# Adding element type (line 28)
str_2028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 4), 'str', 'center')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 10), list_2015, str_2028)
# Adding element type (line 28)
str_2029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 14), 'str', 'count')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 10), list_2015, str_2029)
# Adding element type (line 28)
str_2030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 23), 'str', 'decode')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 10), list_2015, str_2030)
# Adding element type (line 28)
str_2031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 33), 'str', 'encode')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 10), list_2015, str_2031)
# Adding element type (line 28)
str_2032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 43), 'str', 'endswith')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 10), list_2015, str_2032)
# Adding element type (line 28)
str_2033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 55), 'str', 'expandtabs')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 10), list_2015, str_2033)
# Adding element type (line 28)
str_2034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 4), 'str', 'find')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 10), list_2015, str_2034)
# Adding element type (line 28)
str_2035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 12), 'str', 'index')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 10), list_2015, str_2035)
# Adding element type (line 28)
str_2036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 21), 'str', 'isalnum')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 10), list_2015, str_2036)
# Adding element type (line 28)
str_2037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 32), 'str', 'isalpha')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 10), list_2015, str_2037)
# Adding element type (line 28)
str_2038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 43), 'str', 'isdigit')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 10), list_2015, str_2038)
# Adding element type (line 28)
str_2039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 54), 'str', 'islower')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 10), list_2015, str_2039)
# Adding element type (line 28)
str_2040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 65), 'str', 'isspace')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 10), list_2015, str_2040)
# Adding element type (line 28)
str_2041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 4), 'str', 'istitle')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 10), list_2015, str_2041)
# Adding element type (line 28)
str_2042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 15), 'str', 'isupper')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 10), list_2015, str_2042)
# Adding element type (line 28)
str_2043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 26), 'str', 'join')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 10), list_2015, str_2043)
# Adding element type (line 28)
str_2044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 34), 'str', 'ljust')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 10), list_2015, str_2044)
# Adding element type (line 28)
str_2045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 43), 'str', 'lower')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 10), list_2015, str_2045)
# Adding element type (line 28)
str_2046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 52), 'str', 'lstrip')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 10), list_2015, str_2046)
# Adding element type (line 28)
str_2047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 62), 'str', 'partition')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 10), list_2015, str_2047)
# Adding element type (line 28)
str_2048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 4), 'str', 'replace')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 10), list_2015, str_2048)
# Adding element type (line 28)
str_2049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 15), 'str', 'rfind')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 10), list_2015, str_2049)
# Adding element type (line 28)
str_2050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 24), 'str', 'rindex')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 10), list_2015, str_2050)
# Adding element type (line 28)
str_2051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 34), 'str', 'rjust')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 10), list_2015, str_2051)
# Adding element type (line 28)
str_2052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 43), 'str', 'rpartition')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 10), list_2015, str_2052)
# Adding element type (line 28)
str_2053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 57), 'str', 'rsplit')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 10), list_2015, str_2053)
# Adding element type (line 28)
str_2054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 4), 'str', 'rstrip')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 10), list_2015, str_2054)
# Adding element type (line 28)
str_2055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 14), 'str', 'split')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 10), list_2015, str_2055)
# Adding element type (line 28)
str_2056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 23), 'str', 'splitlines')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 10), list_2015, str_2056)
# Adding element type (line 28)
str_2057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 37), 'str', 'startswith')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 10), list_2015, str_2057)
# Adding element type (line 28)
str_2058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 51), 'str', 'strip')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 10), list_2015, str_2058)
# Adding element type (line 28)
str_2059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 60), 'str', 'swapcase')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 10), list_2015, str_2059)
# Adding element type (line 28)
str_2060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 4), 'str', 'title')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 10), list_2015, str_2060)
# Adding element type (line 28)
str_2061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 13), 'str', 'translate')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 10), list_2015, str_2061)
# Adding element type (line 28)
str_2062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 26), 'str', 'upper')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 10), list_2015, str_2062)
# Adding element type (line 28)
str_2063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 35), 'str', 'zfill')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 10), list_2015, str_2063)
# Adding element type (line 28)
str_2064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 44), 'str', 'isnumeric')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 10), list_2015, str_2064)
# Adding element type (line 28)
str_2065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 57), 'str', 'isdecimal')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 10), list_2015, str_2065)
# Adding element type (line 28)
str_2066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 4), 'str', 'array')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 10), list_2015, str_2066)
# Adding element type (line 28)
str_2067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 13), 'str', 'asarray')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 10), list_2015, str_2067)

# Assigning a type to the variable '__all__' (line 28)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), '__all__', list_2015)

# Assigning a Num to a Name (line 41):
int_2068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 13), 'int')
# Assigning a type to the variable '_globalvar' (line 41)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 0), '_globalvar', int_2068)



# Obtaining the type of the subscript
int_2069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 20), 'int')
# Getting the type of 'sys' (line 42)
sys_2070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 3), 'sys')
# Obtaining the member 'version_info' of a type (line 42)
version_info_2071 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 3), sys_2070, 'version_info')
# Obtaining the member '__getitem__' of a type (line 42)
getitem___2072 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 3), version_info_2071, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 42)
subscript_call_result_2073 = invoke(stypy.reporting.localization.Localization(__file__, 42, 3), getitem___2072, int_2069)

int_2074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 26), 'int')
# Applying the binary operator '>=' (line 42)
result_ge_2075 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 3), '>=', subscript_call_result_2073, int_2074)

# Testing the type of an if condition (line 42)
if_condition_2076 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 42, 0), result_ge_2075)
# Assigning a type to the variable 'if_condition_2076' (line 42)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 0), 'if_condition_2076', if_condition_2076)
# SSA begins for if statement (line 42)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a Name to a Name (line 43):
# Getting the type of 'str' (line 43)
str_2077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 15), 'str')
# Assigning a type to the variable '_unicode' (line 43)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), '_unicode', str_2077)

# Assigning a Name to a Name (line 44):
# Getting the type of 'bytes' (line 44)
bytes_2078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 13), 'bytes')
# Assigning a type to the variable '_bytes' (line 44)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), '_bytes', bytes_2078)
# SSA branch for the else part of an if statement (line 42)
module_type_store.open_ssa_branch('else')

# Assigning a Name to a Name (line 46):
# Getting the type of 'unicode' (line 46)
unicode_2079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 15), 'unicode')
# Assigning a type to the variable '_unicode' (line 46)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), '_unicode', unicode_2079)

# Assigning a Name to a Name (line 47):
# Getting the type of 'str' (line 47)
str_2080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 13), 'str')
# Assigning a type to the variable '_bytes' (line 47)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), '_bytes', str_2080)
# SSA join for if statement (line 42)
module_type_store = module_type_store.join_ssa_context()


# Assigning a Name to a Name (line 48):
# Getting the type of 'len' (line 48)
len_2081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 7), 'len')
# Assigning a type to the variable '_len' (line 48)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 0), '_len', len_2081)

@norecursion
def _use_unicode(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_use_unicode'
    module_type_store = module_type_store.open_function_context('_use_unicode', 50, 0, False)
    
    # Passed parameters checking function
    _use_unicode.stypy_localization = localization
    _use_unicode.stypy_type_of_self = None
    _use_unicode.stypy_type_store = module_type_store
    _use_unicode.stypy_function_name = '_use_unicode'
    _use_unicode.stypy_param_names_list = []
    _use_unicode.stypy_varargs_param_name = 'args'
    _use_unicode.stypy_kwargs_param_name = None
    _use_unicode.stypy_call_defaults = defaults
    _use_unicode.stypy_call_varargs = varargs
    _use_unicode.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_use_unicode', [], 'args', None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_use_unicode', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_use_unicode(...)' code ##################

    str_2082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, (-1)), 'str', '\n    Helper function for determining the output type of some string\n    operations.\n\n    For an operation on two ndarrays, if at least one is unicode, the\n    result should be unicode.\n    ')
    
    # Getting the type of 'args' (line 58)
    args_2083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 13), 'args')
    # Testing the type of a for loop iterable (line 58)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 58, 4), args_2083)
    # Getting the type of the for loop variable (line 58)
    for_loop_var_2084 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 58, 4), args_2083)
    # Assigning a type to the variable 'x' (line 58)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'x', for_loop_var_2084)
    # SSA begins for a for statement (line 58)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Evaluating a boolean operation
    
    # Call to isinstance(...): (line 59)
    # Processing the call arguments (line 59)
    # Getting the type of 'x' (line 59)
    x_2086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 23), 'x', False)
    # Getting the type of '_unicode' (line 59)
    _unicode_2087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 26), '_unicode', False)
    # Processing the call keyword arguments (line 59)
    kwargs_2088 = {}
    # Getting the type of 'isinstance' (line 59)
    isinstance_2085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 12), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 59)
    isinstance_call_result_2089 = invoke(stypy.reporting.localization.Localization(__file__, 59, 12), isinstance_2085, *[x_2086, _unicode_2087], **kwargs_2088)
    
    
    # Call to issubclass(...): (line 60)
    # Processing the call arguments (line 60)
    
    # Call to asarray(...): (line 60)
    # Processing the call arguments (line 60)
    # Getting the type of 'x' (line 60)
    x_2093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 41), 'x', False)
    # Processing the call keyword arguments (line 60)
    kwargs_2094 = {}
    # Getting the type of 'numpy' (line 60)
    numpy_2091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 27), 'numpy', False)
    # Obtaining the member 'asarray' of a type (line 60)
    asarray_2092 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 27), numpy_2091, 'asarray')
    # Calling asarray(args, kwargs) (line 60)
    asarray_call_result_2095 = invoke(stypy.reporting.localization.Localization(__file__, 60, 27), asarray_2092, *[x_2093], **kwargs_2094)
    
    # Obtaining the member 'dtype' of a type (line 60)
    dtype_2096 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 27), asarray_call_result_2095, 'dtype')
    # Obtaining the member 'type' of a type (line 60)
    type_2097 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 27), dtype_2096, 'type')
    # Getting the type of 'unicode_' (line 60)
    unicode__2098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 56), 'unicode_', False)
    # Processing the call keyword arguments (line 60)
    kwargs_2099 = {}
    # Getting the type of 'issubclass' (line 60)
    issubclass_2090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 16), 'issubclass', False)
    # Calling issubclass(args, kwargs) (line 60)
    issubclass_call_result_2100 = invoke(stypy.reporting.localization.Localization(__file__, 60, 16), issubclass_2090, *[type_2097, unicode__2098], **kwargs_2099)
    
    # Applying the binary operator 'or' (line 59)
    result_or_keyword_2101 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 12), 'or', isinstance_call_result_2089, issubclass_call_result_2100)
    
    # Testing the type of an if condition (line 59)
    if_condition_2102 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 59, 8), result_or_keyword_2101)
    # Assigning a type to the variable 'if_condition_2102' (line 59)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'if_condition_2102', if_condition_2102)
    # SSA begins for if statement (line 59)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'unicode_' (line 61)
    unicode__2103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 19), 'unicode_')
    # Assigning a type to the variable 'stypy_return_type' (line 61)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 12), 'stypy_return_type', unicode__2103)
    # SSA join for if statement (line 59)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'string_' (line 62)
    string__2104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 11), 'string_')
    # Assigning a type to the variable 'stypy_return_type' (line 62)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), 'stypy_return_type', string__2104)
    
    # ################# End of '_use_unicode(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_use_unicode' in the type store
    # Getting the type of 'stypy_return_type' (line 50)
    stypy_return_type_2105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_2105)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_use_unicode'
    return stypy_return_type_2105

# Assigning a type to the variable '_use_unicode' (line 50)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 0), '_use_unicode', _use_unicode)

@norecursion
def _to_string_or_unicode_array(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_to_string_or_unicode_array'
    module_type_store = module_type_store.open_function_context('_to_string_or_unicode_array', 64, 0, False)
    
    # Passed parameters checking function
    _to_string_or_unicode_array.stypy_localization = localization
    _to_string_or_unicode_array.stypy_type_of_self = None
    _to_string_or_unicode_array.stypy_type_store = module_type_store
    _to_string_or_unicode_array.stypy_function_name = '_to_string_or_unicode_array'
    _to_string_or_unicode_array.stypy_param_names_list = ['result']
    _to_string_or_unicode_array.stypy_varargs_param_name = None
    _to_string_or_unicode_array.stypy_kwargs_param_name = None
    _to_string_or_unicode_array.stypy_call_defaults = defaults
    _to_string_or_unicode_array.stypy_call_varargs = varargs
    _to_string_or_unicode_array.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_to_string_or_unicode_array', ['result'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_to_string_or_unicode_array', localization, ['result'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_to_string_or_unicode_array(...)' code ##################

    str_2106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, (-1)), 'str', '\n    Helper function to cast a result back into a string or unicode array\n    if an object array must be used as an intermediary.\n    ')
    
    # Call to asarray(...): (line 69)
    # Processing the call arguments (line 69)
    
    # Call to tolist(...): (line 69)
    # Processing the call keyword arguments (line 69)
    kwargs_2111 = {}
    # Getting the type of 'result' (line 69)
    result_2109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 25), 'result', False)
    # Obtaining the member 'tolist' of a type (line 69)
    tolist_2110 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 25), result_2109, 'tolist')
    # Calling tolist(args, kwargs) (line 69)
    tolist_call_result_2112 = invoke(stypy.reporting.localization.Localization(__file__, 69, 25), tolist_2110, *[], **kwargs_2111)
    
    # Processing the call keyword arguments (line 69)
    kwargs_2113 = {}
    # Getting the type of 'numpy' (line 69)
    numpy_2107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 11), 'numpy', False)
    # Obtaining the member 'asarray' of a type (line 69)
    asarray_2108 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 11), numpy_2107, 'asarray')
    # Calling asarray(args, kwargs) (line 69)
    asarray_call_result_2114 = invoke(stypy.reporting.localization.Localization(__file__, 69, 11), asarray_2108, *[tolist_call_result_2112], **kwargs_2113)
    
    # Assigning a type to the variable 'stypy_return_type' (line 69)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'stypy_return_type', asarray_call_result_2114)
    
    # ################# End of '_to_string_or_unicode_array(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_to_string_or_unicode_array' in the type store
    # Getting the type of 'stypy_return_type' (line 64)
    stypy_return_type_2115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_2115)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_to_string_or_unicode_array'
    return stypy_return_type_2115

# Assigning a type to the variable '_to_string_or_unicode_array' (line 64)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 0), '_to_string_or_unicode_array', _to_string_or_unicode_array)

@norecursion
def _clean_args(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_clean_args'
    module_type_store = module_type_store.open_function_context('_clean_args', 71, 0, False)
    
    # Passed parameters checking function
    _clean_args.stypy_localization = localization
    _clean_args.stypy_type_of_self = None
    _clean_args.stypy_type_store = module_type_store
    _clean_args.stypy_function_name = '_clean_args'
    _clean_args.stypy_param_names_list = []
    _clean_args.stypy_varargs_param_name = 'args'
    _clean_args.stypy_kwargs_param_name = None
    _clean_args.stypy_call_defaults = defaults
    _clean_args.stypy_call_varargs = varargs
    _clean_args.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_clean_args', [], 'args', None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_clean_args', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_clean_args(...)' code ##################

    str_2116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, (-1)), 'str', "\n    Helper function for delegating arguments to Python string\n    functions.\n\n    Many of the Python string operations that have optional arguments\n    do not use 'None' to indicate a default value.  In these cases,\n    we need to remove all `None` arguments, and those following them.\n    ")
    
    # Assigning a List to a Name (line 80):
    
    # Obtaining an instance of the builtin type 'list' (line 80)
    list_2117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 80)
    
    # Assigning a type to the variable 'newargs' (line 80)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'newargs', list_2117)
    
    # Getting the type of 'args' (line 81)
    args_2118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 15), 'args')
    # Testing the type of a for loop iterable (line 81)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 81, 4), args_2118)
    # Getting the type of the for loop variable (line 81)
    for_loop_var_2119 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 81, 4), args_2118)
    # Assigning a type to the variable 'chk' (line 81)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'chk', for_loop_var_2119)
    # SSA begins for a for statement (line 81)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Type idiom detected: calculating its left and rigth part (line 82)
    # Getting the type of 'chk' (line 82)
    chk_2120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 11), 'chk')
    # Getting the type of 'None' (line 82)
    None_2121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 18), 'None')
    
    (may_be_2122, more_types_in_union_2123) = may_be_none(chk_2120, None_2121)

    if may_be_2122:

        if more_types_in_union_2123:
            # Runtime conditional SSA (line 82)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store


        if more_types_in_union_2123:
            # SSA join for if statement (line 82)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Call to append(...): (line 84)
    # Processing the call arguments (line 84)
    # Getting the type of 'chk' (line 84)
    chk_2126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 23), 'chk', False)
    # Processing the call keyword arguments (line 84)
    kwargs_2127 = {}
    # Getting the type of 'newargs' (line 84)
    newargs_2124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'newargs', False)
    # Obtaining the member 'append' of a type (line 84)
    append_2125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 8), newargs_2124, 'append')
    # Calling append(args, kwargs) (line 84)
    append_call_result_2128 = invoke(stypy.reporting.localization.Localization(__file__, 84, 8), append_2125, *[chk_2126], **kwargs_2127)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'newargs' (line 85)
    newargs_2129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 11), 'newargs')
    # Assigning a type to the variable 'stypy_return_type' (line 85)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'stypy_return_type', newargs_2129)
    
    # ################# End of '_clean_args(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_clean_args' in the type store
    # Getting the type of 'stypy_return_type' (line 71)
    stypy_return_type_2130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_2130)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_clean_args'
    return stypy_return_type_2130

# Assigning a type to the variable '_clean_args' (line 71)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 0), '_clean_args', _clean_args)

@norecursion
def _get_num_chars(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_get_num_chars'
    module_type_store = module_type_store.open_function_context('_get_num_chars', 87, 0, False)
    
    # Passed parameters checking function
    _get_num_chars.stypy_localization = localization
    _get_num_chars.stypy_type_of_self = None
    _get_num_chars.stypy_type_store = module_type_store
    _get_num_chars.stypy_function_name = '_get_num_chars'
    _get_num_chars.stypy_param_names_list = ['a']
    _get_num_chars.stypy_varargs_param_name = None
    _get_num_chars.stypy_kwargs_param_name = None
    _get_num_chars.stypy_call_defaults = defaults
    _get_num_chars.stypy_call_varargs = varargs
    _get_num_chars.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_get_num_chars', ['a'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_get_num_chars', localization, ['a'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_get_num_chars(...)' code ##################

    str_2131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, (-1)), 'str', '\n    Helper function that returns the number of characters per field in\n    a string or unicode array.  This is to abstract out the fact that\n    for a unicode array this is itemsize / 4.\n    ')
    
    
    # Call to issubclass(...): (line 93)
    # Processing the call arguments (line 93)
    # Getting the type of 'a' (line 93)
    a_2133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 18), 'a', False)
    # Obtaining the member 'dtype' of a type (line 93)
    dtype_2134 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 18), a_2133, 'dtype')
    # Obtaining the member 'type' of a type (line 93)
    type_2135 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 18), dtype_2134, 'type')
    # Getting the type of 'unicode_' (line 93)
    unicode__2136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 32), 'unicode_', False)
    # Processing the call keyword arguments (line 93)
    kwargs_2137 = {}
    # Getting the type of 'issubclass' (line 93)
    issubclass_2132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 7), 'issubclass', False)
    # Calling issubclass(args, kwargs) (line 93)
    issubclass_call_result_2138 = invoke(stypy.reporting.localization.Localization(__file__, 93, 7), issubclass_2132, *[type_2135, unicode__2136], **kwargs_2137)
    
    # Testing the type of an if condition (line 93)
    if_condition_2139 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 93, 4), issubclass_call_result_2138)
    # Assigning a type to the variable 'if_condition_2139' (line 93)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'if_condition_2139', if_condition_2139)
    # SSA begins for if statement (line 93)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'a' (line 94)
    a_2140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 15), 'a')
    # Obtaining the member 'itemsize' of a type (line 94)
    itemsize_2141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 15), a_2140, 'itemsize')
    int_2142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 29), 'int')
    # Applying the binary operator '//' (line 94)
    result_floordiv_2143 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 15), '//', itemsize_2141, int_2142)
    
    # Assigning a type to the variable 'stypy_return_type' (line 94)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'stypy_return_type', result_floordiv_2143)
    # SSA join for if statement (line 93)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'a' (line 95)
    a_2144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 11), 'a')
    # Obtaining the member 'itemsize' of a type (line 95)
    itemsize_2145 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 11), a_2144, 'itemsize')
    # Assigning a type to the variable 'stypy_return_type' (line 95)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 4), 'stypy_return_type', itemsize_2145)
    
    # ################# End of '_get_num_chars(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_get_num_chars' in the type store
    # Getting the type of 'stypy_return_type' (line 87)
    stypy_return_type_2146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_2146)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_get_num_chars'
    return stypy_return_type_2146

# Assigning a type to the variable '_get_num_chars' (line 87)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 0), '_get_num_chars', _get_num_chars)

@norecursion
def equal(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'equal'
    module_type_store = module_type_store.open_function_context('equal', 98, 0, False)
    
    # Passed parameters checking function
    equal.stypy_localization = localization
    equal.stypy_type_of_self = None
    equal.stypy_type_store = module_type_store
    equal.stypy_function_name = 'equal'
    equal.stypy_param_names_list = ['x1', 'x2']
    equal.stypy_varargs_param_name = None
    equal.stypy_kwargs_param_name = None
    equal.stypy_call_defaults = defaults
    equal.stypy_call_varargs = varargs
    equal.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'equal', ['x1', 'x2'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'equal', localization, ['x1', 'x2'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'equal(...)' code ##################

    str_2147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, (-1)), 'str', '\n    Return (x1 == x2) element-wise.\n\n    Unlike `numpy.equal`, this comparison is performed by first\n    stripping whitespace characters from the end of the string.  This\n    behavior is provided for backward-compatibility with numarray.\n\n    Parameters\n    ----------\n    x1, x2 : array_like of str or unicode\n        Input arrays of the same shape.\n\n    Returns\n    -------\n    out : ndarray or bool\n        Output array of bools, or a single bool if x1 and x2 are scalars.\n\n    See Also\n    --------\n    not_equal, greater_equal, less_equal, greater, less\n    ')
    
    # Call to compare_chararrays(...): (line 120)
    # Processing the call arguments (line 120)
    # Getting the type of 'x1' (line 120)
    x1_2149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 30), 'x1', False)
    # Getting the type of 'x2' (line 120)
    x2_2150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 34), 'x2', False)
    str_2151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 38), 'str', '==')
    # Getting the type of 'True' (line 120)
    True_2152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 44), 'True', False)
    # Processing the call keyword arguments (line 120)
    kwargs_2153 = {}
    # Getting the type of 'compare_chararrays' (line 120)
    compare_chararrays_2148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 11), 'compare_chararrays', False)
    # Calling compare_chararrays(args, kwargs) (line 120)
    compare_chararrays_call_result_2154 = invoke(stypy.reporting.localization.Localization(__file__, 120, 11), compare_chararrays_2148, *[x1_2149, x2_2150, str_2151, True_2152], **kwargs_2153)
    
    # Assigning a type to the variable 'stypy_return_type' (line 120)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 4), 'stypy_return_type', compare_chararrays_call_result_2154)
    
    # ################# End of 'equal(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'equal' in the type store
    # Getting the type of 'stypy_return_type' (line 98)
    stypy_return_type_2155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_2155)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'equal'
    return stypy_return_type_2155

# Assigning a type to the variable 'equal' (line 98)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 0), 'equal', equal)

@norecursion
def not_equal(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'not_equal'
    module_type_store = module_type_store.open_function_context('not_equal', 122, 0, False)
    
    # Passed parameters checking function
    not_equal.stypy_localization = localization
    not_equal.stypy_type_of_self = None
    not_equal.stypy_type_store = module_type_store
    not_equal.stypy_function_name = 'not_equal'
    not_equal.stypy_param_names_list = ['x1', 'x2']
    not_equal.stypy_varargs_param_name = None
    not_equal.stypy_kwargs_param_name = None
    not_equal.stypy_call_defaults = defaults
    not_equal.stypy_call_varargs = varargs
    not_equal.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'not_equal', ['x1', 'x2'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'not_equal', localization, ['x1', 'x2'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'not_equal(...)' code ##################

    str_2156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, (-1)), 'str', '\n    Return (x1 != x2) element-wise.\n\n    Unlike `numpy.not_equal`, this comparison is performed by first\n    stripping whitespace characters from the end of the string.  This\n    behavior is provided for backward-compatibility with numarray.\n\n    Parameters\n    ----------\n    x1, x2 : array_like of str or unicode\n        Input arrays of the same shape.\n\n    Returns\n    -------\n    out : ndarray or bool\n        Output array of bools, or a single bool if x1 and x2 are scalars.\n\n    See Also\n    --------\n    equal, greater_equal, less_equal, greater, less\n    ')
    
    # Call to compare_chararrays(...): (line 144)
    # Processing the call arguments (line 144)
    # Getting the type of 'x1' (line 144)
    x1_2158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 30), 'x1', False)
    # Getting the type of 'x2' (line 144)
    x2_2159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 34), 'x2', False)
    str_2160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 38), 'str', '!=')
    # Getting the type of 'True' (line 144)
    True_2161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 44), 'True', False)
    # Processing the call keyword arguments (line 144)
    kwargs_2162 = {}
    # Getting the type of 'compare_chararrays' (line 144)
    compare_chararrays_2157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 11), 'compare_chararrays', False)
    # Calling compare_chararrays(args, kwargs) (line 144)
    compare_chararrays_call_result_2163 = invoke(stypy.reporting.localization.Localization(__file__, 144, 11), compare_chararrays_2157, *[x1_2158, x2_2159, str_2160, True_2161], **kwargs_2162)
    
    # Assigning a type to the variable 'stypy_return_type' (line 144)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 4), 'stypy_return_type', compare_chararrays_call_result_2163)
    
    # ################# End of 'not_equal(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'not_equal' in the type store
    # Getting the type of 'stypy_return_type' (line 122)
    stypy_return_type_2164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_2164)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'not_equal'
    return stypy_return_type_2164

# Assigning a type to the variable 'not_equal' (line 122)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 0), 'not_equal', not_equal)

@norecursion
def greater_equal(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'greater_equal'
    module_type_store = module_type_store.open_function_context('greater_equal', 146, 0, False)
    
    # Passed parameters checking function
    greater_equal.stypy_localization = localization
    greater_equal.stypy_type_of_self = None
    greater_equal.stypy_type_store = module_type_store
    greater_equal.stypy_function_name = 'greater_equal'
    greater_equal.stypy_param_names_list = ['x1', 'x2']
    greater_equal.stypy_varargs_param_name = None
    greater_equal.stypy_kwargs_param_name = None
    greater_equal.stypy_call_defaults = defaults
    greater_equal.stypy_call_varargs = varargs
    greater_equal.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'greater_equal', ['x1', 'x2'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'greater_equal', localization, ['x1', 'x2'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'greater_equal(...)' code ##################

    str_2165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, (-1)), 'str', '\n    Return (x1 >= x2) element-wise.\n\n    Unlike `numpy.greater_equal`, this comparison is performed by\n    first stripping whitespace characters from the end of the string.\n    This behavior is provided for backward-compatibility with\n    numarray.\n\n    Parameters\n    ----------\n    x1, x2 : array_like of str or unicode\n        Input arrays of the same shape.\n\n    Returns\n    -------\n    out : ndarray or bool\n        Output array of bools, or a single bool if x1 and x2 are scalars.\n\n    See Also\n    --------\n    equal, not_equal, less_equal, greater, less\n    ')
    
    # Call to compare_chararrays(...): (line 169)
    # Processing the call arguments (line 169)
    # Getting the type of 'x1' (line 169)
    x1_2167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 30), 'x1', False)
    # Getting the type of 'x2' (line 169)
    x2_2168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 34), 'x2', False)
    str_2169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 38), 'str', '>=')
    # Getting the type of 'True' (line 169)
    True_2170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 44), 'True', False)
    # Processing the call keyword arguments (line 169)
    kwargs_2171 = {}
    # Getting the type of 'compare_chararrays' (line 169)
    compare_chararrays_2166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 11), 'compare_chararrays', False)
    # Calling compare_chararrays(args, kwargs) (line 169)
    compare_chararrays_call_result_2172 = invoke(stypy.reporting.localization.Localization(__file__, 169, 11), compare_chararrays_2166, *[x1_2167, x2_2168, str_2169, True_2170], **kwargs_2171)
    
    # Assigning a type to the variable 'stypy_return_type' (line 169)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 4), 'stypy_return_type', compare_chararrays_call_result_2172)
    
    # ################# End of 'greater_equal(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'greater_equal' in the type store
    # Getting the type of 'stypy_return_type' (line 146)
    stypy_return_type_2173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_2173)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'greater_equal'
    return stypy_return_type_2173

# Assigning a type to the variable 'greater_equal' (line 146)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 0), 'greater_equal', greater_equal)

@norecursion
def less_equal(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'less_equal'
    module_type_store = module_type_store.open_function_context('less_equal', 171, 0, False)
    
    # Passed parameters checking function
    less_equal.stypy_localization = localization
    less_equal.stypy_type_of_self = None
    less_equal.stypy_type_store = module_type_store
    less_equal.stypy_function_name = 'less_equal'
    less_equal.stypy_param_names_list = ['x1', 'x2']
    less_equal.stypy_varargs_param_name = None
    less_equal.stypy_kwargs_param_name = None
    less_equal.stypy_call_defaults = defaults
    less_equal.stypy_call_varargs = varargs
    less_equal.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'less_equal', ['x1', 'x2'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'less_equal', localization, ['x1', 'x2'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'less_equal(...)' code ##################

    str_2174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, (-1)), 'str', '\n    Return (x1 <= x2) element-wise.\n\n    Unlike `numpy.less_equal`, this comparison is performed by first\n    stripping whitespace characters from the end of the string.  This\n    behavior is provided for backward-compatibility with numarray.\n\n    Parameters\n    ----------\n    x1, x2 : array_like of str or unicode\n        Input arrays of the same shape.\n\n    Returns\n    -------\n    out : ndarray or bool\n        Output array of bools, or a single bool if x1 and x2 are scalars.\n\n    See Also\n    --------\n    equal, not_equal, greater_equal, greater, less\n    ')
    
    # Call to compare_chararrays(...): (line 193)
    # Processing the call arguments (line 193)
    # Getting the type of 'x1' (line 193)
    x1_2176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 30), 'x1', False)
    # Getting the type of 'x2' (line 193)
    x2_2177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 34), 'x2', False)
    str_2178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 38), 'str', '<=')
    # Getting the type of 'True' (line 193)
    True_2179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 44), 'True', False)
    # Processing the call keyword arguments (line 193)
    kwargs_2180 = {}
    # Getting the type of 'compare_chararrays' (line 193)
    compare_chararrays_2175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 11), 'compare_chararrays', False)
    # Calling compare_chararrays(args, kwargs) (line 193)
    compare_chararrays_call_result_2181 = invoke(stypy.reporting.localization.Localization(__file__, 193, 11), compare_chararrays_2175, *[x1_2176, x2_2177, str_2178, True_2179], **kwargs_2180)
    
    # Assigning a type to the variable 'stypy_return_type' (line 193)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 4), 'stypy_return_type', compare_chararrays_call_result_2181)
    
    # ################# End of 'less_equal(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'less_equal' in the type store
    # Getting the type of 'stypy_return_type' (line 171)
    stypy_return_type_2182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_2182)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'less_equal'
    return stypy_return_type_2182

# Assigning a type to the variable 'less_equal' (line 171)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 0), 'less_equal', less_equal)

@norecursion
def greater(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'greater'
    module_type_store = module_type_store.open_function_context('greater', 195, 0, False)
    
    # Passed parameters checking function
    greater.stypy_localization = localization
    greater.stypy_type_of_self = None
    greater.stypy_type_store = module_type_store
    greater.stypy_function_name = 'greater'
    greater.stypy_param_names_list = ['x1', 'x2']
    greater.stypy_varargs_param_name = None
    greater.stypy_kwargs_param_name = None
    greater.stypy_call_defaults = defaults
    greater.stypy_call_varargs = varargs
    greater.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'greater', ['x1', 'x2'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'greater', localization, ['x1', 'x2'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'greater(...)' code ##################

    str_2183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, (-1)), 'str', '\n    Return (x1 > x2) element-wise.\n\n    Unlike `numpy.greater`, this comparison is performed by first\n    stripping whitespace characters from the end of the string.  This\n    behavior is provided for backward-compatibility with numarray.\n\n    Parameters\n    ----------\n    x1, x2 : array_like of str or unicode\n        Input arrays of the same shape.\n\n    Returns\n    -------\n    out : ndarray or bool\n        Output array of bools, or a single bool if x1 and x2 are scalars.\n\n    See Also\n    --------\n    equal, not_equal, greater_equal, less_equal, less\n    ')
    
    # Call to compare_chararrays(...): (line 217)
    # Processing the call arguments (line 217)
    # Getting the type of 'x1' (line 217)
    x1_2185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 30), 'x1', False)
    # Getting the type of 'x2' (line 217)
    x2_2186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 34), 'x2', False)
    str_2187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 38), 'str', '>')
    # Getting the type of 'True' (line 217)
    True_2188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 43), 'True', False)
    # Processing the call keyword arguments (line 217)
    kwargs_2189 = {}
    # Getting the type of 'compare_chararrays' (line 217)
    compare_chararrays_2184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 11), 'compare_chararrays', False)
    # Calling compare_chararrays(args, kwargs) (line 217)
    compare_chararrays_call_result_2190 = invoke(stypy.reporting.localization.Localization(__file__, 217, 11), compare_chararrays_2184, *[x1_2185, x2_2186, str_2187, True_2188], **kwargs_2189)
    
    # Assigning a type to the variable 'stypy_return_type' (line 217)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 4), 'stypy_return_type', compare_chararrays_call_result_2190)
    
    # ################# End of 'greater(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'greater' in the type store
    # Getting the type of 'stypy_return_type' (line 195)
    stypy_return_type_2191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_2191)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'greater'
    return stypy_return_type_2191

# Assigning a type to the variable 'greater' (line 195)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 0), 'greater', greater)

@norecursion
def less(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'less'
    module_type_store = module_type_store.open_function_context('less', 219, 0, False)
    
    # Passed parameters checking function
    less.stypy_localization = localization
    less.stypy_type_of_self = None
    less.stypy_type_store = module_type_store
    less.stypy_function_name = 'less'
    less.stypy_param_names_list = ['x1', 'x2']
    less.stypy_varargs_param_name = None
    less.stypy_kwargs_param_name = None
    less.stypy_call_defaults = defaults
    less.stypy_call_varargs = varargs
    less.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'less', ['x1', 'x2'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'less', localization, ['x1', 'x2'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'less(...)' code ##################

    str_2192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, (-1)), 'str', '\n    Return (x1 < x2) element-wise.\n\n    Unlike `numpy.greater`, this comparison is performed by first\n    stripping whitespace characters from the end of the string.  This\n    behavior is provided for backward-compatibility with numarray.\n\n    Parameters\n    ----------\n    x1, x2 : array_like of str or unicode\n        Input arrays of the same shape.\n\n    Returns\n    -------\n    out : ndarray or bool\n        Output array of bools, or a single bool if x1 and x2 are scalars.\n\n    See Also\n    --------\n    equal, not_equal, greater_equal, less_equal, greater\n    ')
    
    # Call to compare_chararrays(...): (line 241)
    # Processing the call arguments (line 241)
    # Getting the type of 'x1' (line 241)
    x1_2194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 30), 'x1', False)
    # Getting the type of 'x2' (line 241)
    x2_2195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 34), 'x2', False)
    str_2196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 38), 'str', '<')
    # Getting the type of 'True' (line 241)
    True_2197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 43), 'True', False)
    # Processing the call keyword arguments (line 241)
    kwargs_2198 = {}
    # Getting the type of 'compare_chararrays' (line 241)
    compare_chararrays_2193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 11), 'compare_chararrays', False)
    # Calling compare_chararrays(args, kwargs) (line 241)
    compare_chararrays_call_result_2199 = invoke(stypy.reporting.localization.Localization(__file__, 241, 11), compare_chararrays_2193, *[x1_2194, x2_2195, str_2196, True_2197], **kwargs_2198)
    
    # Assigning a type to the variable 'stypy_return_type' (line 241)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 4), 'stypy_return_type', compare_chararrays_call_result_2199)
    
    # ################# End of 'less(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'less' in the type store
    # Getting the type of 'stypy_return_type' (line 219)
    stypy_return_type_2200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_2200)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'less'
    return stypy_return_type_2200

# Assigning a type to the variable 'less' (line 219)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 0), 'less', less)

@norecursion
def str_len(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'str_len'
    module_type_store = module_type_store.open_function_context('str_len', 243, 0, False)
    
    # Passed parameters checking function
    str_len.stypy_localization = localization
    str_len.stypy_type_of_self = None
    str_len.stypy_type_store = module_type_store
    str_len.stypy_function_name = 'str_len'
    str_len.stypy_param_names_list = ['a']
    str_len.stypy_varargs_param_name = None
    str_len.stypy_kwargs_param_name = None
    str_len.stypy_call_defaults = defaults
    str_len.stypy_call_varargs = varargs
    str_len.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'str_len', ['a'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'str_len', localization, ['a'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'str_len(...)' code ##################

    str_2201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, (-1)), 'str', '\n    Return len(a) element-wise.\n\n    Parameters\n    ----------\n    a : array_like of str or unicode\n\n    Returns\n    -------\n    out : ndarray\n        Output array of integers\n\n    See also\n    --------\n    __builtin__.len\n    ')
    
    # Call to _vec_string(...): (line 260)
    # Processing the call arguments (line 260)
    # Getting the type of 'a' (line 260)
    a_2203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 23), 'a', False)
    # Getting the type of 'integer' (line 260)
    integer_2204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 26), 'integer', False)
    str_2205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 35), 'str', '__len__')
    # Processing the call keyword arguments (line 260)
    kwargs_2206 = {}
    # Getting the type of '_vec_string' (line 260)
    _vec_string_2202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 11), '_vec_string', False)
    # Calling _vec_string(args, kwargs) (line 260)
    _vec_string_call_result_2207 = invoke(stypy.reporting.localization.Localization(__file__, 260, 11), _vec_string_2202, *[a_2203, integer_2204, str_2205], **kwargs_2206)
    
    # Assigning a type to the variable 'stypy_return_type' (line 260)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 4), 'stypy_return_type', _vec_string_call_result_2207)
    
    # ################# End of 'str_len(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'str_len' in the type store
    # Getting the type of 'stypy_return_type' (line 243)
    stypy_return_type_2208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_2208)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'str_len'
    return stypy_return_type_2208

# Assigning a type to the variable 'str_len' (line 243)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 0), 'str_len', str_len)

@norecursion
def add(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'add'
    module_type_store = module_type_store.open_function_context('add', 262, 0, False)
    
    # Passed parameters checking function
    add.stypy_localization = localization
    add.stypy_type_of_self = None
    add.stypy_type_store = module_type_store
    add.stypy_function_name = 'add'
    add.stypy_param_names_list = ['x1', 'x2']
    add.stypy_varargs_param_name = None
    add.stypy_kwargs_param_name = None
    add.stypy_call_defaults = defaults
    add.stypy_call_varargs = varargs
    add.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'add', ['x1', 'x2'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'add', localization, ['x1', 'x2'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'add(...)' code ##################

    str_2209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, (-1)), 'str', '\n    Return element-wise string concatenation for two arrays of str or unicode.\n\n    Arrays `x1` and `x2` must have the same shape.\n\n    Parameters\n    ----------\n    x1 : array_like of str or unicode\n        Input array.\n    x2 : array_like of str or unicode\n        Input array.\n\n    Returns\n    -------\n    add : ndarray\n        Output array of `string_` or `unicode_`, depending on input types\n        of the same shape as `x1` and `x2`.\n\n    ')
    
    # Assigning a Call to a Name (line 282):
    
    # Call to asarray(...): (line 282)
    # Processing the call arguments (line 282)
    # Getting the type of 'x1' (line 282)
    x1_2212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 25), 'x1', False)
    # Processing the call keyword arguments (line 282)
    kwargs_2213 = {}
    # Getting the type of 'numpy' (line 282)
    numpy_2210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 11), 'numpy', False)
    # Obtaining the member 'asarray' of a type (line 282)
    asarray_2211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 11), numpy_2210, 'asarray')
    # Calling asarray(args, kwargs) (line 282)
    asarray_call_result_2214 = invoke(stypy.reporting.localization.Localization(__file__, 282, 11), asarray_2211, *[x1_2212], **kwargs_2213)
    
    # Assigning a type to the variable 'arr1' (line 282)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 4), 'arr1', asarray_call_result_2214)
    
    # Assigning a Call to a Name (line 283):
    
    # Call to asarray(...): (line 283)
    # Processing the call arguments (line 283)
    # Getting the type of 'x2' (line 283)
    x2_2217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 25), 'x2', False)
    # Processing the call keyword arguments (line 283)
    kwargs_2218 = {}
    # Getting the type of 'numpy' (line 283)
    numpy_2215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 11), 'numpy', False)
    # Obtaining the member 'asarray' of a type (line 283)
    asarray_2216 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 11), numpy_2215, 'asarray')
    # Calling asarray(args, kwargs) (line 283)
    asarray_call_result_2219 = invoke(stypy.reporting.localization.Localization(__file__, 283, 11), asarray_2216, *[x2_2217], **kwargs_2218)
    
    # Assigning a type to the variable 'arr2' (line 283)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 4), 'arr2', asarray_call_result_2219)
    
    # Assigning a BinOp to a Name (line 284):
    
    # Call to _get_num_chars(...): (line 284)
    # Processing the call arguments (line 284)
    # Getting the type of 'arr1' (line 284)
    arr1_2221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 30), 'arr1', False)
    # Processing the call keyword arguments (line 284)
    kwargs_2222 = {}
    # Getting the type of '_get_num_chars' (line 284)
    _get_num_chars_2220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 15), '_get_num_chars', False)
    # Calling _get_num_chars(args, kwargs) (line 284)
    _get_num_chars_call_result_2223 = invoke(stypy.reporting.localization.Localization(__file__, 284, 15), _get_num_chars_2220, *[arr1_2221], **kwargs_2222)
    
    
    # Call to _get_num_chars(...): (line 284)
    # Processing the call arguments (line 284)
    # Getting the type of 'arr2' (line 284)
    arr2_2225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 53), 'arr2', False)
    # Processing the call keyword arguments (line 284)
    kwargs_2226 = {}
    # Getting the type of '_get_num_chars' (line 284)
    _get_num_chars_2224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 38), '_get_num_chars', False)
    # Calling _get_num_chars(args, kwargs) (line 284)
    _get_num_chars_call_result_2227 = invoke(stypy.reporting.localization.Localization(__file__, 284, 38), _get_num_chars_2224, *[arr2_2225], **kwargs_2226)
    
    # Applying the binary operator '+' (line 284)
    result_add_2228 = python_operator(stypy.reporting.localization.Localization(__file__, 284, 15), '+', _get_num_chars_call_result_2223, _get_num_chars_call_result_2227)
    
    # Assigning a type to the variable 'out_size' (line 284)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 4), 'out_size', result_add_2228)
    
    # Assigning a Call to a Name (line 285):
    
    # Call to _use_unicode(...): (line 285)
    # Processing the call arguments (line 285)
    # Getting the type of 'arr1' (line 285)
    arr1_2230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 25), 'arr1', False)
    # Getting the type of 'arr2' (line 285)
    arr2_2231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 31), 'arr2', False)
    # Processing the call keyword arguments (line 285)
    kwargs_2232 = {}
    # Getting the type of '_use_unicode' (line 285)
    _use_unicode_2229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 12), '_use_unicode', False)
    # Calling _use_unicode(args, kwargs) (line 285)
    _use_unicode_call_result_2233 = invoke(stypy.reporting.localization.Localization(__file__, 285, 12), _use_unicode_2229, *[arr1_2230, arr2_2231], **kwargs_2232)
    
    # Assigning a type to the variable 'dtype' (line 285)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 4), 'dtype', _use_unicode_call_result_2233)
    
    # Call to _vec_string(...): (line 286)
    # Processing the call arguments (line 286)
    # Getting the type of 'arr1' (line 286)
    arr1_2235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 23), 'arr1', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 286)
    tuple_2236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 30), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 286)
    # Adding element type (line 286)
    # Getting the type of 'dtype' (line 286)
    dtype_2237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 30), 'dtype', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 286, 30), tuple_2236, dtype_2237)
    # Adding element type (line 286)
    # Getting the type of 'out_size' (line 286)
    out_size_2238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 37), 'out_size', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 286, 30), tuple_2236, out_size_2238)
    
    str_2239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 48), 'str', '__add__')
    
    # Obtaining an instance of the builtin type 'tuple' (line 286)
    tuple_2240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 60), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 286)
    # Adding element type (line 286)
    # Getting the type of 'arr2' (line 286)
    arr2_2241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 60), 'arr2', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 286, 60), tuple_2240, arr2_2241)
    
    # Processing the call keyword arguments (line 286)
    kwargs_2242 = {}
    # Getting the type of '_vec_string' (line 286)
    _vec_string_2234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 11), '_vec_string', False)
    # Calling _vec_string(args, kwargs) (line 286)
    _vec_string_call_result_2243 = invoke(stypy.reporting.localization.Localization(__file__, 286, 11), _vec_string_2234, *[arr1_2235, tuple_2236, str_2239, tuple_2240], **kwargs_2242)
    
    # Assigning a type to the variable 'stypy_return_type' (line 286)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 4), 'stypy_return_type', _vec_string_call_result_2243)
    
    # ################# End of 'add(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'add' in the type store
    # Getting the type of 'stypy_return_type' (line 262)
    stypy_return_type_2244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_2244)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'add'
    return stypy_return_type_2244

# Assigning a type to the variable 'add' (line 262)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 0), 'add', add)

@norecursion
def multiply(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'multiply'
    module_type_store = module_type_store.open_function_context('multiply', 288, 0, False)
    
    # Passed parameters checking function
    multiply.stypy_localization = localization
    multiply.stypy_type_of_self = None
    multiply.stypy_type_store = module_type_store
    multiply.stypy_function_name = 'multiply'
    multiply.stypy_param_names_list = ['a', 'i']
    multiply.stypy_varargs_param_name = None
    multiply.stypy_kwargs_param_name = None
    multiply.stypy_call_defaults = defaults
    multiply.stypy_call_varargs = varargs
    multiply.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'multiply', ['a', 'i'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'multiply', localization, ['a', 'i'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'multiply(...)' code ##################

    str_2245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, (-1)), 'str', '\n    Return (a * i), that is string multiple concatenation,\n    element-wise.\n\n    Values in `i` of less than 0 are treated as 0 (which yields an\n    empty string).\n\n    Parameters\n    ----------\n    a : array_like of str or unicode\n\n    i : array_like of ints\n\n    Returns\n    -------\n    out : ndarray\n        Output array of str or unicode, depending on input types\n\n    ')
    
    # Assigning a Call to a Name (line 308):
    
    # Call to asarray(...): (line 308)
    # Processing the call arguments (line 308)
    # Getting the type of 'a' (line 308)
    a_2248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 26), 'a', False)
    # Processing the call keyword arguments (line 308)
    kwargs_2249 = {}
    # Getting the type of 'numpy' (line 308)
    numpy_2246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 12), 'numpy', False)
    # Obtaining the member 'asarray' of a type (line 308)
    asarray_2247 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 12), numpy_2246, 'asarray')
    # Calling asarray(args, kwargs) (line 308)
    asarray_call_result_2250 = invoke(stypy.reporting.localization.Localization(__file__, 308, 12), asarray_2247, *[a_2248], **kwargs_2249)
    
    # Assigning a type to the variable 'a_arr' (line 308)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 308, 4), 'a_arr', asarray_call_result_2250)
    
    # Assigning a Call to a Name (line 309):
    
    # Call to asarray(...): (line 309)
    # Processing the call arguments (line 309)
    # Getting the type of 'i' (line 309)
    i_2253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 26), 'i', False)
    # Processing the call keyword arguments (line 309)
    kwargs_2254 = {}
    # Getting the type of 'numpy' (line 309)
    numpy_2251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 12), 'numpy', False)
    # Obtaining the member 'asarray' of a type (line 309)
    asarray_2252 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 12), numpy_2251, 'asarray')
    # Calling asarray(args, kwargs) (line 309)
    asarray_call_result_2255 = invoke(stypy.reporting.localization.Localization(__file__, 309, 12), asarray_2252, *[i_2253], **kwargs_2254)
    
    # Assigning a type to the variable 'i_arr' (line 309)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 4), 'i_arr', asarray_call_result_2255)
    
    
    
    # Call to issubclass(...): (line 310)
    # Processing the call arguments (line 310)
    # Getting the type of 'i_arr' (line 310)
    i_arr_2257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 22), 'i_arr', False)
    # Obtaining the member 'dtype' of a type (line 310)
    dtype_2258 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 22), i_arr_2257, 'dtype')
    # Obtaining the member 'type' of a type (line 310)
    type_2259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 22), dtype_2258, 'type')
    # Getting the type of 'integer' (line 310)
    integer_2260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 40), 'integer', False)
    # Processing the call keyword arguments (line 310)
    kwargs_2261 = {}
    # Getting the type of 'issubclass' (line 310)
    issubclass_2256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 11), 'issubclass', False)
    # Calling issubclass(args, kwargs) (line 310)
    issubclass_call_result_2262 = invoke(stypy.reporting.localization.Localization(__file__, 310, 11), issubclass_2256, *[type_2259, integer_2260], **kwargs_2261)
    
    # Applying the 'not' unary operator (line 310)
    result_not__2263 = python_operator(stypy.reporting.localization.Localization(__file__, 310, 7), 'not', issubclass_call_result_2262)
    
    # Testing the type of an if condition (line 310)
    if_condition_2264 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 310, 4), result_not__2263)
    # Assigning a type to the variable 'if_condition_2264' (line 310)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 4), 'if_condition_2264', if_condition_2264)
    # SSA begins for if statement (line 310)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 311)
    # Processing the call arguments (line 311)
    str_2266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 25), 'str', 'Can only multiply by integers')
    # Processing the call keyword arguments (line 311)
    kwargs_2267 = {}
    # Getting the type of 'ValueError' (line 311)
    ValueError_2265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 311)
    ValueError_call_result_2268 = invoke(stypy.reporting.localization.Localization(__file__, 311, 14), ValueError_2265, *[str_2266], **kwargs_2267)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 311, 8), ValueError_call_result_2268, 'raise parameter', BaseException)
    # SSA join for if statement (line 310)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 312):
    
    # Call to _get_num_chars(...): (line 312)
    # Processing the call arguments (line 312)
    # Getting the type of 'a_arr' (line 312)
    a_arr_2270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 30), 'a_arr', False)
    # Processing the call keyword arguments (line 312)
    kwargs_2271 = {}
    # Getting the type of '_get_num_chars' (line 312)
    _get_num_chars_2269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 15), '_get_num_chars', False)
    # Calling _get_num_chars(args, kwargs) (line 312)
    _get_num_chars_call_result_2272 = invoke(stypy.reporting.localization.Localization(__file__, 312, 15), _get_num_chars_2269, *[a_arr_2270], **kwargs_2271)
    
    
    # Call to max(...): (line 312)
    # Processing the call arguments (line 312)
    
    # Call to long(...): (line 312)
    # Processing the call arguments (line 312)
    
    # Call to max(...): (line 312)
    # Processing the call keyword arguments (line 312)
    kwargs_2277 = {}
    # Getting the type of 'i_arr' (line 312)
    i_arr_2275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 48), 'i_arr', False)
    # Obtaining the member 'max' of a type (line 312)
    max_2276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 48), i_arr_2275, 'max')
    # Calling max(args, kwargs) (line 312)
    max_call_result_2278 = invoke(stypy.reporting.localization.Localization(__file__, 312, 48), max_2276, *[], **kwargs_2277)
    
    # Processing the call keyword arguments (line 312)
    kwargs_2279 = {}
    # Getting the type of 'long' (line 312)
    long_2274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 43), 'long', False)
    # Calling long(args, kwargs) (line 312)
    long_call_result_2280 = invoke(stypy.reporting.localization.Localization(__file__, 312, 43), long_2274, *[max_call_result_2278], **kwargs_2279)
    
    int_2281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 62), 'int')
    # Processing the call keyword arguments (line 312)
    kwargs_2282 = {}
    # Getting the type of 'max' (line 312)
    max_2273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 39), 'max', False)
    # Calling max(args, kwargs) (line 312)
    max_call_result_2283 = invoke(stypy.reporting.localization.Localization(__file__, 312, 39), max_2273, *[long_call_result_2280, int_2281], **kwargs_2282)
    
    # Applying the binary operator '*' (line 312)
    result_mul_2284 = python_operator(stypy.reporting.localization.Localization(__file__, 312, 15), '*', _get_num_chars_call_result_2272, max_call_result_2283)
    
    # Assigning a type to the variable 'out_size' (line 312)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 4), 'out_size', result_mul_2284)
    
    # Call to _vec_string(...): (line 313)
    # Processing the call arguments (line 313)
    # Getting the type of 'a_arr' (line 314)
    a_arr_2286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 8), 'a_arr', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 314)
    tuple_2287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 16), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 314)
    # Adding element type (line 314)
    # Getting the type of 'a_arr' (line 314)
    a_arr_2288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 16), 'a_arr', False)
    # Obtaining the member 'dtype' of a type (line 314)
    dtype_2289 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 16), a_arr_2288, 'dtype')
    # Obtaining the member 'type' of a type (line 314)
    type_2290 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 16), dtype_2289, 'type')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 314, 16), tuple_2287, type_2290)
    # Adding element type (line 314)
    # Getting the type of 'out_size' (line 314)
    out_size_2291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 34), 'out_size', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 314, 16), tuple_2287, out_size_2291)
    
    str_2292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 45), 'str', '__mul__')
    
    # Obtaining an instance of the builtin type 'tuple' (line 314)
    tuple_2293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 57), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 314)
    # Adding element type (line 314)
    # Getting the type of 'i_arr' (line 314)
    i_arr_2294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 57), 'i_arr', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 314, 57), tuple_2293, i_arr_2294)
    
    # Processing the call keyword arguments (line 313)
    kwargs_2295 = {}
    # Getting the type of '_vec_string' (line 313)
    _vec_string_2285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 11), '_vec_string', False)
    # Calling _vec_string(args, kwargs) (line 313)
    _vec_string_call_result_2296 = invoke(stypy.reporting.localization.Localization(__file__, 313, 11), _vec_string_2285, *[a_arr_2286, tuple_2287, str_2292, tuple_2293], **kwargs_2295)
    
    # Assigning a type to the variable 'stypy_return_type' (line 313)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 313, 4), 'stypy_return_type', _vec_string_call_result_2296)
    
    # ################# End of 'multiply(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'multiply' in the type store
    # Getting the type of 'stypy_return_type' (line 288)
    stypy_return_type_2297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_2297)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'multiply'
    return stypy_return_type_2297

# Assigning a type to the variable 'multiply' (line 288)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 0), 'multiply', multiply)

@norecursion
def mod(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'mod'
    module_type_store = module_type_store.open_function_context('mod', 316, 0, False)
    
    # Passed parameters checking function
    mod.stypy_localization = localization
    mod.stypy_type_of_self = None
    mod.stypy_type_store = module_type_store
    mod.stypy_function_name = 'mod'
    mod.stypy_param_names_list = ['a', 'values']
    mod.stypy_varargs_param_name = None
    mod.stypy_kwargs_param_name = None
    mod.stypy_call_defaults = defaults
    mod.stypy_call_varargs = varargs
    mod.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'mod', ['a', 'values'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'mod', localization, ['a', 'values'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'mod(...)' code ##################

    str_2298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, (-1)), 'str', '\n    Return (a % i), that is pre-Python 2.6 string formatting\n    (iterpolation), element-wise for a pair of array_likes of str\n    or unicode.\n\n    Parameters\n    ----------\n    a : array_like of str or unicode\n\n    values : array_like of values\n       These values will be element-wise interpolated into the string.\n\n    Returns\n    -------\n    out : ndarray\n        Output array of str or unicode, depending on input types\n\n    See also\n    --------\n    str.__mod__\n\n    ')
    
    # Call to _to_string_or_unicode_array(...): (line 339)
    # Processing the call arguments (line 339)
    
    # Call to _vec_string(...): (line 340)
    # Processing the call arguments (line 340)
    # Getting the type of 'a' (line 340)
    a_2301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 20), 'a', False)
    # Getting the type of 'object_' (line 340)
    object__2302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 23), 'object_', False)
    str_2303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 32), 'str', '__mod__')
    
    # Obtaining an instance of the builtin type 'tuple' (line 340)
    tuple_2304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 44), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 340)
    # Adding element type (line 340)
    # Getting the type of 'values' (line 340)
    values_2305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 44), 'values', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 340, 44), tuple_2304, values_2305)
    
    # Processing the call keyword arguments (line 340)
    kwargs_2306 = {}
    # Getting the type of '_vec_string' (line 340)
    _vec_string_2300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 8), '_vec_string', False)
    # Calling _vec_string(args, kwargs) (line 340)
    _vec_string_call_result_2307 = invoke(stypy.reporting.localization.Localization(__file__, 340, 8), _vec_string_2300, *[a_2301, object__2302, str_2303, tuple_2304], **kwargs_2306)
    
    # Processing the call keyword arguments (line 339)
    kwargs_2308 = {}
    # Getting the type of '_to_string_or_unicode_array' (line 339)
    _to_string_or_unicode_array_2299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 11), '_to_string_or_unicode_array', False)
    # Calling _to_string_or_unicode_array(args, kwargs) (line 339)
    _to_string_or_unicode_array_call_result_2309 = invoke(stypy.reporting.localization.Localization(__file__, 339, 11), _to_string_or_unicode_array_2299, *[_vec_string_call_result_2307], **kwargs_2308)
    
    # Assigning a type to the variable 'stypy_return_type' (line 339)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 339, 4), 'stypy_return_type', _to_string_or_unicode_array_call_result_2309)
    
    # ################# End of 'mod(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'mod' in the type store
    # Getting the type of 'stypy_return_type' (line 316)
    stypy_return_type_2310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_2310)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'mod'
    return stypy_return_type_2310

# Assigning a type to the variable 'mod' (line 316)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 316, 0), 'mod', mod)

@norecursion
def capitalize(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'capitalize'
    module_type_store = module_type_store.open_function_context('capitalize', 342, 0, False)
    
    # Passed parameters checking function
    capitalize.stypy_localization = localization
    capitalize.stypy_type_of_self = None
    capitalize.stypy_type_store = module_type_store
    capitalize.stypy_function_name = 'capitalize'
    capitalize.stypy_param_names_list = ['a']
    capitalize.stypy_varargs_param_name = None
    capitalize.stypy_kwargs_param_name = None
    capitalize.stypy_call_defaults = defaults
    capitalize.stypy_call_varargs = varargs
    capitalize.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'capitalize', ['a'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'capitalize', localization, ['a'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'capitalize(...)' code ##################

    str_2311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, (-1)), 'str', "\n    Return a copy of `a` with only the first character of each element\n    capitalized.\n\n    Calls `str.capitalize` element-wise.\n\n    For 8-bit strings, this method is locale-dependent.\n\n    Parameters\n    ----------\n    a : array_like of str or unicode\n        Input array of strings to capitalize.\n\n    Returns\n    -------\n    out : ndarray\n        Output array of str or unicode, depending on input\n        types\n\n    See also\n    --------\n    str.capitalize\n\n    Examples\n    --------\n    >>> c = np.array(['a1b2','1b2a','b2a1','2a1b'],'S4'); c\n    array(['a1b2', '1b2a', 'b2a1', '2a1b'],\n        dtype='|S4')\n    >>> np.char.capitalize(c)\n    array(['A1b2', '1b2a', 'B2a1', '2a1b'],\n        dtype='|S4')\n\n    ")
    
    # Assigning a Call to a Name (line 376):
    
    # Call to asarray(...): (line 376)
    # Processing the call arguments (line 376)
    # Getting the type of 'a' (line 376)
    a_2314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 26), 'a', False)
    # Processing the call keyword arguments (line 376)
    kwargs_2315 = {}
    # Getting the type of 'numpy' (line 376)
    numpy_2312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 12), 'numpy', False)
    # Obtaining the member 'asarray' of a type (line 376)
    asarray_2313 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 376, 12), numpy_2312, 'asarray')
    # Calling asarray(args, kwargs) (line 376)
    asarray_call_result_2316 = invoke(stypy.reporting.localization.Localization(__file__, 376, 12), asarray_2313, *[a_2314], **kwargs_2315)
    
    # Assigning a type to the variable 'a_arr' (line 376)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 376, 4), 'a_arr', asarray_call_result_2316)
    
    # Call to _vec_string(...): (line 377)
    # Processing the call arguments (line 377)
    # Getting the type of 'a_arr' (line 377)
    a_arr_2318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 23), 'a_arr', False)
    # Getting the type of 'a_arr' (line 377)
    a_arr_2319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 30), 'a_arr', False)
    # Obtaining the member 'dtype' of a type (line 377)
    dtype_2320 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 377, 30), a_arr_2319, 'dtype')
    str_2321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 377, 43), 'str', 'capitalize')
    # Processing the call keyword arguments (line 377)
    kwargs_2322 = {}
    # Getting the type of '_vec_string' (line 377)
    _vec_string_2317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 11), '_vec_string', False)
    # Calling _vec_string(args, kwargs) (line 377)
    _vec_string_call_result_2323 = invoke(stypy.reporting.localization.Localization(__file__, 377, 11), _vec_string_2317, *[a_arr_2318, dtype_2320, str_2321], **kwargs_2322)
    
    # Assigning a type to the variable 'stypy_return_type' (line 377)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 377, 4), 'stypy_return_type', _vec_string_call_result_2323)
    
    # ################# End of 'capitalize(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'capitalize' in the type store
    # Getting the type of 'stypy_return_type' (line 342)
    stypy_return_type_2324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_2324)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'capitalize'
    return stypy_return_type_2324

# Assigning a type to the variable 'capitalize' (line 342)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 0), 'capitalize', capitalize)

@norecursion
def center(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_2325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 380, 30), 'str', ' ')
    defaults = [str_2325]
    # Create a new context for function 'center'
    module_type_store = module_type_store.open_function_context('center', 380, 0, False)
    
    # Passed parameters checking function
    center.stypy_localization = localization
    center.stypy_type_of_self = None
    center.stypy_type_store = module_type_store
    center.stypy_function_name = 'center'
    center.stypy_param_names_list = ['a', 'width', 'fillchar']
    center.stypy_varargs_param_name = None
    center.stypy_kwargs_param_name = None
    center.stypy_call_defaults = defaults
    center.stypy_call_varargs = varargs
    center.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'center', ['a', 'width', 'fillchar'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'center', localization, ['a', 'width', 'fillchar'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'center(...)' code ##################

    str_2326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, (-1)), 'str', '\n    Return a copy of `a` with its elements centered in a string of\n    length `width`.\n\n    Calls `str.center` element-wise.\n\n    Parameters\n    ----------\n    a : array_like of str or unicode\n\n    width : int\n        The length of the resulting strings\n    fillchar : str or unicode, optional\n        The padding character to use (default is space).\n\n    Returns\n    -------\n    out : ndarray\n        Output array of str or unicode, depending on input\n        types\n\n    See also\n    --------\n    str.center\n\n    ')
    
    # Assigning a Call to a Name (line 407):
    
    # Call to asarray(...): (line 407)
    # Processing the call arguments (line 407)
    # Getting the type of 'a' (line 407)
    a_2329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 26), 'a', False)
    # Processing the call keyword arguments (line 407)
    kwargs_2330 = {}
    # Getting the type of 'numpy' (line 407)
    numpy_2327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 12), 'numpy', False)
    # Obtaining the member 'asarray' of a type (line 407)
    asarray_2328 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 407, 12), numpy_2327, 'asarray')
    # Calling asarray(args, kwargs) (line 407)
    asarray_call_result_2331 = invoke(stypy.reporting.localization.Localization(__file__, 407, 12), asarray_2328, *[a_2329], **kwargs_2330)
    
    # Assigning a type to the variable 'a_arr' (line 407)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 407, 4), 'a_arr', asarray_call_result_2331)
    
    # Assigning a Call to a Name (line 408):
    
    # Call to asarray(...): (line 408)
    # Processing the call arguments (line 408)
    # Getting the type of 'width' (line 408)
    width_2334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 30), 'width', False)
    # Processing the call keyword arguments (line 408)
    kwargs_2335 = {}
    # Getting the type of 'numpy' (line 408)
    numpy_2332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 16), 'numpy', False)
    # Obtaining the member 'asarray' of a type (line 408)
    asarray_2333 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 408, 16), numpy_2332, 'asarray')
    # Calling asarray(args, kwargs) (line 408)
    asarray_call_result_2336 = invoke(stypy.reporting.localization.Localization(__file__, 408, 16), asarray_2333, *[width_2334], **kwargs_2335)
    
    # Assigning a type to the variable 'width_arr' (line 408)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 408, 4), 'width_arr', asarray_call_result_2336)
    
    # Assigning a Call to a Name (line 409):
    
    # Call to long(...): (line 409)
    # Processing the call arguments (line 409)
    
    # Call to max(...): (line 409)
    # Processing the call arguments (line 409)
    # Getting the type of 'width_arr' (line 409)
    width_arr_2340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 26), 'width_arr', False)
    # Obtaining the member 'flat' of a type (line 409)
    flat_2341 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 409, 26), width_arr_2340, 'flat')
    # Processing the call keyword arguments (line 409)
    kwargs_2342 = {}
    # Getting the type of 'numpy' (line 409)
    numpy_2338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 16), 'numpy', False)
    # Obtaining the member 'max' of a type (line 409)
    max_2339 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 409, 16), numpy_2338, 'max')
    # Calling max(args, kwargs) (line 409)
    max_call_result_2343 = invoke(stypy.reporting.localization.Localization(__file__, 409, 16), max_2339, *[flat_2341], **kwargs_2342)
    
    # Processing the call keyword arguments (line 409)
    kwargs_2344 = {}
    # Getting the type of 'long' (line 409)
    long_2337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 11), 'long', False)
    # Calling long(args, kwargs) (line 409)
    long_call_result_2345 = invoke(stypy.reporting.localization.Localization(__file__, 409, 11), long_2337, *[max_call_result_2343], **kwargs_2344)
    
    # Assigning a type to the variable 'size' (line 409)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 409, 4), 'size', long_call_result_2345)
    
    
    # Call to issubdtype(...): (line 410)
    # Processing the call arguments (line 410)
    # Getting the type of 'a_arr' (line 410)
    a_arr_2348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 24), 'a_arr', False)
    # Obtaining the member 'dtype' of a type (line 410)
    dtype_2349 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 410, 24), a_arr_2348, 'dtype')
    # Getting the type of 'numpy' (line 410)
    numpy_2350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 37), 'numpy', False)
    # Obtaining the member 'string_' of a type (line 410)
    string__2351 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 410, 37), numpy_2350, 'string_')
    # Processing the call keyword arguments (line 410)
    kwargs_2352 = {}
    # Getting the type of 'numpy' (line 410)
    numpy_2346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 7), 'numpy', False)
    # Obtaining the member 'issubdtype' of a type (line 410)
    issubdtype_2347 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 410, 7), numpy_2346, 'issubdtype')
    # Calling issubdtype(args, kwargs) (line 410)
    issubdtype_call_result_2353 = invoke(stypy.reporting.localization.Localization(__file__, 410, 7), issubdtype_2347, *[dtype_2349, string__2351], **kwargs_2352)
    
    # Testing the type of an if condition (line 410)
    if_condition_2354 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 410, 4), issubdtype_call_result_2353)
    # Assigning a type to the variable 'if_condition_2354' (line 410)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 410, 4), 'if_condition_2354', if_condition_2354)
    # SSA begins for if statement (line 410)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 411):
    
    # Call to asbytes(...): (line 411)
    # Processing the call arguments (line 411)
    # Getting the type of 'fillchar' (line 411)
    fillchar_2356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 27), 'fillchar', False)
    # Processing the call keyword arguments (line 411)
    kwargs_2357 = {}
    # Getting the type of 'asbytes' (line 411)
    asbytes_2355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 19), 'asbytes', False)
    # Calling asbytes(args, kwargs) (line 411)
    asbytes_call_result_2358 = invoke(stypy.reporting.localization.Localization(__file__, 411, 19), asbytes_2355, *[fillchar_2356], **kwargs_2357)
    
    # Assigning a type to the variable 'fillchar' (line 411)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 411, 8), 'fillchar', asbytes_call_result_2358)
    # SSA join for if statement (line 410)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to _vec_string(...): (line 412)
    # Processing the call arguments (line 412)
    # Getting the type of 'a_arr' (line 413)
    a_arr_2360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 8), 'a_arr', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 413)
    tuple_2361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, 16), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 413)
    # Adding element type (line 413)
    # Getting the type of 'a_arr' (line 413)
    a_arr_2362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 16), 'a_arr', False)
    # Obtaining the member 'dtype' of a type (line 413)
    dtype_2363 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 413, 16), a_arr_2362, 'dtype')
    # Obtaining the member 'type' of a type (line 413)
    type_2364 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 413, 16), dtype_2363, 'type')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 413, 16), tuple_2361, type_2364)
    # Adding element type (line 413)
    # Getting the type of 'size' (line 413)
    size_2365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 34), 'size', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 413, 16), tuple_2361, size_2365)
    
    str_2366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, 41), 'str', 'center')
    
    # Obtaining an instance of the builtin type 'tuple' (line 413)
    tuple_2367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, 52), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 413)
    # Adding element type (line 413)
    # Getting the type of 'width_arr' (line 413)
    width_arr_2368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 52), 'width_arr', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 413, 52), tuple_2367, width_arr_2368)
    # Adding element type (line 413)
    # Getting the type of 'fillchar' (line 413)
    fillchar_2369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 63), 'fillchar', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 413, 52), tuple_2367, fillchar_2369)
    
    # Processing the call keyword arguments (line 412)
    kwargs_2370 = {}
    # Getting the type of '_vec_string' (line 412)
    _vec_string_2359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 11), '_vec_string', False)
    # Calling _vec_string(args, kwargs) (line 412)
    _vec_string_call_result_2371 = invoke(stypy.reporting.localization.Localization(__file__, 412, 11), _vec_string_2359, *[a_arr_2360, tuple_2361, str_2366, tuple_2367], **kwargs_2370)
    
    # Assigning a type to the variable 'stypy_return_type' (line 412)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 412, 4), 'stypy_return_type', _vec_string_call_result_2371)
    
    # ################# End of 'center(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'center' in the type store
    # Getting the type of 'stypy_return_type' (line 380)
    stypy_return_type_2372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_2372)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'center'
    return stypy_return_type_2372

# Assigning a type to the variable 'center' (line 380)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 380, 0), 'center', center)

@norecursion
def count(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_2373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 24), 'int')
    # Getting the type of 'None' (line 416)
    None_2374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 31), 'None')
    defaults = [int_2373, None_2374]
    # Create a new context for function 'count'
    module_type_store = module_type_store.open_function_context('count', 416, 0, False)
    
    # Passed parameters checking function
    count.stypy_localization = localization
    count.stypy_type_of_self = None
    count.stypy_type_store = module_type_store
    count.stypy_function_name = 'count'
    count.stypy_param_names_list = ['a', 'sub', 'start', 'end']
    count.stypy_varargs_param_name = None
    count.stypy_kwargs_param_name = None
    count.stypy_call_defaults = defaults
    count.stypy_call_varargs = varargs
    count.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'count', ['a', 'sub', 'start', 'end'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'count', localization, ['a', 'sub', 'start', 'end'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'count(...)' code ##################

    str_2375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 458, (-1)), 'str', "\n    Returns an array with the number of non-overlapping occurrences of\n    substring `sub` in the range [`start`, `end`].\n\n    Calls `str.count` element-wise.\n\n    Parameters\n    ----------\n    a : array_like of str or unicode\n\n    sub : str or unicode\n       The substring to search for.\n\n    start, end : int, optional\n       Optional arguments `start` and `end` are interpreted as slice\n       notation to specify the range in which to count.\n\n    Returns\n    -------\n    out : ndarray\n        Output array of ints.\n\n    See also\n    --------\n    str.count\n\n    Examples\n    --------\n    >>> c = np.array(['aAaAaA', '  aA  ', 'abBABba'])\n    >>> c\n    array(['aAaAaA', '  aA  ', 'abBABba'],\n        dtype='|S7')\n    >>> np.char.count(c, 'A')\n    array([3, 1, 1])\n    >>> np.char.count(c, 'aA')\n    array([3, 1, 0])\n    >>> np.char.count(c, 'A', start=1, end=4)\n    array([2, 1, 1])\n    >>> np.char.count(c, 'A', start=1, end=3)\n    array([1, 0, 0])\n\n    ")
    
    # Call to _vec_string(...): (line 459)
    # Processing the call arguments (line 459)
    # Getting the type of 'a' (line 459)
    a_2377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 23), 'a', False)
    # Getting the type of 'integer' (line 459)
    integer_2378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 26), 'integer', False)
    str_2379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 459, 35), 'str', 'count')
    
    # Obtaining an instance of the builtin type 'list' (line 459)
    list_2380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 459, 44), 'list')
    # Adding type elements to the builtin type 'list' instance (line 459)
    # Adding element type (line 459)
    # Getting the type of 'sub' (line 459)
    sub_2381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 45), 'sub', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 459, 44), list_2380, sub_2381)
    # Adding element type (line 459)
    # Getting the type of 'start' (line 459)
    start_2382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 50), 'start', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 459, 44), list_2380, start_2382)
    
    
    # Call to _clean_args(...): (line 459)
    # Processing the call arguments (line 459)
    # Getting the type of 'end' (line 459)
    end_2384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 71), 'end', False)
    # Processing the call keyword arguments (line 459)
    kwargs_2385 = {}
    # Getting the type of '_clean_args' (line 459)
    _clean_args_2383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 59), '_clean_args', False)
    # Calling _clean_args(args, kwargs) (line 459)
    _clean_args_call_result_2386 = invoke(stypy.reporting.localization.Localization(__file__, 459, 59), _clean_args_2383, *[end_2384], **kwargs_2385)
    
    # Applying the binary operator '+' (line 459)
    result_add_2387 = python_operator(stypy.reporting.localization.Localization(__file__, 459, 44), '+', list_2380, _clean_args_call_result_2386)
    
    # Processing the call keyword arguments (line 459)
    kwargs_2388 = {}
    # Getting the type of '_vec_string' (line 459)
    _vec_string_2376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 11), '_vec_string', False)
    # Calling _vec_string(args, kwargs) (line 459)
    _vec_string_call_result_2389 = invoke(stypy.reporting.localization.Localization(__file__, 459, 11), _vec_string_2376, *[a_2377, integer_2378, str_2379, result_add_2387], **kwargs_2388)
    
    # Assigning a type to the variable 'stypy_return_type' (line 459)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 459, 4), 'stypy_return_type', _vec_string_call_result_2389)
    
    # ################# End of 'count(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'count' in the type store
    # Getting the type of 'stypy_return_type' (line 416)
    stypy_return_type_2390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_2390)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'count'
    return stypy_return_type_2390

# Assigning a type to the variable 'count' (line 416)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 416, 0), 'count', count)

@norecursion
def decode(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 462)
    None_2391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 23), 'None')
    # Getting the type of 'None' (line 462)
    None_2392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 36), 'None')
    defaults = [None_2391, None_2392]
    # Create a new context for function 'decode'
    module_type_store = module_type_store.open_function_context('decode', 462, 0, False)
    
    # Passed parameters checking function
    decode.stypy_localization = localization
    decode.stypy_type_of_self = None
    decode.stypy_type_store = module_type_store
    decode.stypy_function_name = 'decode'
    decode.stypy_param_names_list = ['a', 'encoding', 'errors']
    decode.stypy_varargs_param_name = None
    decode.stypy_kwargs_param_name = None
    decode.stypy_call_defaults = defaults
    decode.stypy_call_varargs = varargs
    decode.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'decode', ['a', 'encoding', 'errors'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'decode', localization, ['a', 'encoding', 'errors'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'decode(...)' code ##################

    str_2393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 503, (-1)), 'str', "\n    Calls `str.decode` element-wise.\n\n    The set of available codecs comes from the Python standard library,\n    and may be extended at runtime.  For more information, see the\n    :mod:`codecs` module.\n\n    Parameters\n    ----------\n    a : array_like of str or unicode\n\n    encoding : str, optional\n       The name of an encoding\n\n    errors : str, optional\n       Specifies how to handle encoding errors\n\n    Returns\n    -------\n    out : ndarray\n\n    See also\n    --------\n    str.decode\n\n    Notes\n    -----\n    The type of the result will depend on the encoding specified.\n\n    Examples\n    --------\n    >>> c = np.array(['aAaAaA', '  aA  ', 'abBABba'])\n    >>> c\n    array(['aAaAaA', '  aA  ', 'abBABba'],\n        dtype='|S7')\n    >>> np.char.encode(c, encoding='cp037')\n    array(['\\x81\\xc1\\x81\\xc1\\x81\\xc1', '@@\\x81\\xc1@@',\n        '\\x81\\x82\\xc2\\xc1\\xc2\\x82\\x81'],\n        dtype='|S7')\n\n    ")
    
    # Call to _to_string_or_unicode_array(...): (line 504)
    # Processing the call arguments (line 504)
    
    # Call to _vec_string(...): (line 505)
    # Processing the call arguments (line 505)
    # Getting the type of 'a' (line 505)
    a_2396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 20), 'a', False)
    # Getting the type of 'object_' (line 505)
    object__2397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 23), 'object_', False)
    str_2398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 505, 32), 'str', 'decode')
    
    # Call to _clean_args(...): (line 505)
    # Processing the call arguments (line 505)
    # Getting the type of 'encoding' (line 505)
    encoding_2400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 54), 'encoding', False)
    # Getting the type of 'errors' (line 505)
    errors_2401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 64), 'errors', False)
    # Processing the call keyword arguments (line 505)
    kwargs_2402 = {}
    # Getting the type of '_clean_args' (line 505)
    _clean_args_2399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 42), '_clean_args', False)
    # Calling _clean_args(args, kwargs) (line 505)
    _clean_args_call_result_2403 = invoke(stypy.reporting.localization.Localization(__file__, 505, 42), _clean_args_2399, *[encoding_2400, errors_2401], **kwargs_2402)
    
    # Processing the call keyword arguments (line 505)
    kwargs_2404 = {}
    # Getting the type of '_vec_string' (line 505)
    _vec_string_2395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 8), '_vec_string', False)
    # Calling _vec_string(args, kwargs) (line 505)
    _vec_string_call_result_2405 = invoke(stypy.reporting.localization.Localization(__file__, 505, 8), _vec_string_2395, *[a_2396, object__2397, str_2398, _clean_args_call_result_2403], **kwargs_2404)
    
    # Processing the call keyword arguments (line 504)
    kwargs_2406 = {}
    # Getting the type of '_to_string_or_unicode_array' (line 504)
    _to_string_or_unicode_array_2394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 11), '_to_string_or_unicode_array', False)
    # Calling _to_string_or_unicode_array(args, kwargs) (line 504)
    _to_string_or_unicode_array_call_result_2407 = invoke(stypy.reporting.localization.Localization(__file__, 504, 11), _to_string_or_unicode_array_2394, *[_vec_string_call_result_2405], **kwargs_2406)
    
    # Assigning a type to the variable 'stypy_return_type' (line 504)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 504, 4), 'stypy_return_type', _to_string_or_unicode_array_call_result_2407)
    
    # ################# End of 'decode(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'decode' in the type store
    # Getting the type of 'stypy_return_type' (line 462)
    stypy_return_type_2408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_2408)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'decode'
    return stypy_return_type_2408

# Assigning a type to the variable 'decode' (line 462)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 462, 0), 'decode', decode)

@norecursion
def encode(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 508)
    None_2409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 23), 'None')
    # Getting the type of 'None' (line 508)
    None_2410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 36), 'None')
    defaults = [None_2409, None_2410]
    # Create a new context for function 'encode'
    module_type_store = module_type_store.open_function_context('encode', 508, 0, False)
    
    # Passed parameters checking function
    encode.stypy_localization = localization
    encode.stypy_type_of_self = None
    encode.stypy_type_store = module_type_store
    encode.stypy_function_name = 'encode'
    encode.stypy_param_names_list = ['a', 'encoding', 'errors']
    encode.stypy_varargs_param_name = None
    encode.stypy_kwargs_param_name = None
    encode.stypy_call_defaults = defaults
    encode.stypy_call_varargs = varargs
    encode.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'encode', ['a', 'encoding', 'errors'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'encode', localization, ['a', 'encoding', 'errors'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'encode(...)' code ##################

    str_2411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 538, (-1)), 'str', '\n    Calls `str.encode` element-wise.\n\n    The set of available codecs comes from the Python standard library,\n    and may be extended at runtime. For more information, see the codecs\n    module.\n\n    Parameters\n    ----------\n    a : array_like of str or unicode\n\n    encoding : str, optional\n       The name of an encoding\n\n    errors : str, optional\n       Specifies how to handle encoding errors\n\n    Returns\n    -------\n    out : ndarray\n\n    See also\n    --------\n    str.encode\n\n    Notes\n    -----\n    The type of the result will depend on the encoding specified.\n\n    ')
    
    # Call to _to_string_or_unicode_array(...): (line 539)
    # Processing the call arguments (line 539)
    
    # Call to _vec_string(...): (line 540)
    # Processing the call arguments (line 540)
    # Getting the type of 'a' (line 540)
    a_2414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 20), 'a', False)
    # Getting the type of 'object_' (line 540)
    object__2415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 23), 'object_', False)
    str_2416 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 540, 32), 'str', 'encode')
    
    # Call to _clean_args(...): (line 540)
    # Processing the call arguments (line 540)
    # Getting the type of 'encoding' (line 540)
    encoding_2418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 54), 'encoding', False)
    # Getting the type of 'errors' (line 540)
    errors_2419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 64), 'errors', False)
    # Processing the call keyword arguments (line 540)
    kwargs_2420 = {}
    # Getting the type of '_clean_args' (line 540)
    _clean_args_2417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 42), '_clean_args', False)
    # Calling _clean_args(args, kwargs) (line 540)
    _clean_args_call_result_2421 = invoke(stypy.reporting.localization.Localization(__file__, 540, 42), _clean_args_2417, *[encoding_2418, errors_2419], **kwargs_2420)
    
    # Processing the call keyword arguments (line 540)
    kwargs_2422 = {}
    # Getting the type of '_vec_string' (line 540)
    _vec_string_2413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 8), '_vec_string', False)
    # Calling _vec_string(args, kwargs) (line 540)
    _vec_string_call_result_2423 = invoke(stypy.reporting.localization.Localization(__file__, 540, 8), _vec_string_2413, *[a_2414, object__2415, str_2416, _clean_args_call_result_2421], **kwargs_2422)
    
    # Processing the call keyword arguments (line 539)
    kwargs_2424 = {}
    # Getting the type of '_to_string_or_unicode_array' (line 539)
    _to_string_or_unicode_array_2412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 11), '_to_string_or_unicode_array', False)
    # Calling _to_string_or_unicode_array(args, kwargs) (line 539)
    _to_string_or_unicode_array_call_result_2425 = invoke(stypy.reporting.localization.Localization(__file__, 539, 11), _to_string_or_unicode_array_2412, *[_vec_string_call_result_2423], **kwargs_2424)
    
    # Assigning a type to the variable 'stypy_return_type' (line 539)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 539, 4), 'stypy_return_type', _to_string_or_unicode_array_call_result_2425)
    
    # ################# End of 'encode(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'encode' in the type store
    # Getting the type of 'stypy_return_type' (line 508)
    stypy_return_type_2426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_2426)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'encode'
    return stypy_return_type_2426

# Assigning a type to the variable 'encode' (line 508)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 508, 0), 'encode', encode)

@norecursion
def endswith(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_2427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 543, 30), 'int')
    # Getting the type of 'None' (line 543)
    None_2428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 37), 'None')
    defaults = [int_2427, None_2428]
    # Create a new context for function 'endswith'
    module_type_store = module_type_store.open_function_context('endswith', 543, 0, False)
    
    # Passed parameters checking function
    endswith.stypy_localization = localization
    endswith.stypy_type_of_self = None
    endswith.stypy_type_store = module_type_store
    endswith.stypy_function_name = 'endswith'
    endswith.stypy_param_names_list = ['a', 'suffix', 'start', 'end']
    endswith.stypy_varargs_param_name = None
    endswith.stypy_kwargs_param_name = None
    endswith.stypy_call_defaults = defaults
    endswith.stypy_call_varargs = varargs
    endswith.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'endswith', ['a', 'suffix', 'start', 'end'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'endswith', localization, ['a', 'suffix', 'start', 'end'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'endswith(...)' code ##################

    str_2429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 582, (-1)), 'str', "\n    Returns a boolean array which is `True` where the string element\n    in `a` ends with `suffix`, otherwise `False`.\n\n    Calls `str.endswith` element-wise.\n\n    Parameters\n    ----------\n    a : array_like of str or unicode\n\n    suffix : str\n\n    start, end : int, optional\n        With optional `start`, test beginning at that position. With\n        optional `end`, stop comparing at that position.\n\n    Returns\n    -------\n    out : ndarray\n        Outputs an array of bools.\n\n    See also\n    --------\n    str.endswith\n\n    Examples\n    --------\n    >>> s = np.array(['foo', 'bar'])\n    >>> s[0] = 'foo'\n    >>> s[1] = 'bar'\n    >>> s\n    array(['foo', 'bar'],\n        dtype='|S3')\n    >>> np.char.endswith(s, 'ar')\n    array([False,  True], dtype=bool)\n    >>> np.char.endswith(s, 'a', start=1, end=2)\n    array([False,  True], dtype=bool)\n\n    ")
    
    # Call to _vec_string(...): (line 583)
    # Processing the call arguments (line 583)
    # Getting the type of 'a' (line 584)
    a_2431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 8), 'a', False)
    # Getting the type of 'bool_' (line 584)
    bool__2432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 11), 'bool_', False)
    str_2433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 584, 18), 'str', 'endswith')
    
    # Obtaining an instance of the builtin type 'list' (line 584)
    list_2434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 584, 30), 'list')
    # Adding type elements to the builtin type 'list' instance (line 584)
    # Adding element type (line 584)
    # Getting the type of 'suffix' (line 584)
    suffix_2435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 31), 'suffix', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 584, 30), list_2434, suffix_2435)
    # Adding element type (line 584)
    # Getting the type of 'start' (line 584)
    start_2436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 39), 'start', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 584, 30), list_2434, start_2436)
    
    
    # Call to _clean_args(...): (line 584)
    # Processing the call arguments (line 584)
    # Getting the type of 'end' (line 584)
    end_2438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 60), 'end', False)
    # Processing the call keyword arguments (line 584)
    kwargs_2439 = {}
    # Getting the type of '_clean_args' (line 584)
    _clean_args_2437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 48), '_clean_args', False)
    # Calling _clean_args(args, kwargs) (line 584)
    _clean_args_call_result_2440 = invoke(stypy.reporting.localization.Localization(__file__, 584, 48), _clean_args_2437, *[end_2438], **kwargs_2439)
    
    # Applying the binary operator '+' (line 584)
    result_add_2441 = python_operator(stypy.reporting.localization.Localization(__file__, 584, 30), '+', list_2434, _clean_args_call_result_2440)
    
    # Processing the call keyword arguments (line 583)
    kwargs_2442 = {}
    # Getting the type of '_vec_string' (line 583)
    _vec_string_2430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 11), '_vec_string', False)
    # Calling _vec_string(args, kwargs) (line 583)
    _vec_string_call_result_2443 = invoke(stypy.reporting.localization.Localization(__file__, 583, 11), _vec_string_2430, *[a_2431, bool__2432, str_2433, result_add_2441], **kwargs_2442)
    
    # Assigning a type to the variable 'stypy_return_type' (line 583)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 583, 4), 'stypy_return_type', _vec_string_call_result_2443)
    
    # ################# End of 'endswith(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'endswith' in the type store
    # Getting the type of 'stypy_return_type' (line 543)
    stypy_return_type_2444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_2444)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'endswith'
    return stypy_return_type_2444

# Assigning a type to the variable 'endswith' (line 543)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 543, 0), 'endswith', endswith)

@norecursion
def expandtabs(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_2445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 587, 26), 'int')
    defaults = [int_2445]
    # Create a new context for function 'expandtabs'
    module_type_store = module_type_store.open_function_context('expandtabs', 587, 0, False)
    
    # Passed parameters checking function
    expandtabs.stypy_localization = localization
    expandtabs.stypy_type_of_self = None
    expandtabs.stypy_type_store = module_type_store
    expandtabs.stypy_function_name = 'expandtabs'
    expandtabs.stypy_param_names_list = ['a', 'tabsize']
    expandtabs.stypy_varargs_param_name = None
    expandtabs.stypy_kwargs_param_name = None
    expandtabs.stypy_call_defaults = defaults
    expandtabs.stypy_call_varargs = varargs
    expandtabs.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'expandtabs', ['a', 'tabsize'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'expandtabs', localization, ['a', 'tabsize'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'expandtabs(...)' code ##################

    str_2446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 617, (-1)), 'str', "\n    Return a copy of each string element where all tab characters are\n    replaced by one or more spaces.\n\n    Calls `str.expandtabs` element-wise.\n\n    Return a copy of each string element where all tab characters are\n    replaced by one or more spaces, depending on the current column\n    and the given `tabsize`. The column number is reset to zero after\n    each newline occurring in the string. This doesn't understand other\n    non-printing characters or escape sequences.\n\n    Parameters\n    ----------\n    a : array_like of str or unicode\n        Input array\n    tabsize : int, optional\n        Replace tabs with `tabsize` number of spaces.  If not given defaults\n        to 8 spaces.\n\n    Returns\n    -------\n    out : ndarray\n        Output array of str or unicode, depending on input type\n\n    See also\n    --------\n    str.expandtabs\n\n    ")
    
    # Call to _to_string_or_unicode_array(...): (line 618)
    # Processing the call arguments (line 618)
    
    # Call to _vec_string(...): (line 619)
    # Processing the call arguments (line 619)
    # Getting the type of 'a' (line 619)
    a_2449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 20), 'a', False)
    # Getting the type of 'object_' (line 619)
    object__2450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 23), 'object_', False)
    str_2451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 619, 32), 'str', 'expandtabs')
    
    # Obtaining an instance of the builtin type 'tuple' (line 619)
    tuple_2452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 619, 47), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 619)
    # Adding element type (line 619)
    # Getting the type of 'tabsize' (line 619)
    tabsize_2453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 47), 'tabsize', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 619, 47), tuple_2452, tabsize_2453)
    
    # Processing the call keyword arguments (line 619)
    kwargs_2454 = {}
    # Getting the type of '_vec_string' (line 619)
    _vec_string_2448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 8), '_vec_string', False)
    # Calling _vec_string(args, kwargs) (line 619)
    _vec_string_call_result_2455 = invoke(stypy.reporting.localization.Localization(__file__, 619, 8), _vec_string_2448, *[a_2449, object__2450, str_2451, tuple_2452], **kwargs_2454)
    
    # Processing the call keyword arguments (line 618)
    kwargs_2456 = {}
    # Getting the type of '_to_string_or_unicode_array' (line 618)
    _to_string_or_unicode_array_2447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 11), '_to_string_or_unicode_array', False)
    # Calling _to_string_or_unicode_array(args, kwargs) (line 618)
    _to_string_or_unicode_array_call_result_2457 = invoke(stypy.reporting.localization.Localization(__file__, 618, 11), _to_string_or_unicode_array_2447, *[_vec_string_call_result_2455], **kwargs_2456)
    
    # Assigning a type to the variable 'stypy_return_type' (line 618)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 618, 4), 'stypy_return_type', _to_string_or_unicode_array_call_result_2457)
    
    # ################# End of 'expandtabs(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'expandtabs' in the type store
    # Getting the type of 'stypy_return_type' (line 587)
    stypy_return_type_2458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_2458)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'expandtabs'
    return stypy_return_type_2458

# Assigning a type to the variable 'expandtabs' (line 587)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 587, 0), 'expandtabs', expandtabs)

@norecursion
def find(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_2459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 622, 23), 'int')
    # Getting the type of 'None' (line 622)
    None_2460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 30), 'None')
    defaults = [int_2459, None_2460]
    # Create a new context for function 'find'
    module_type_store = module_type_store.open_function_context('find', 622, 0, False)
    
    # Passed parameters checking function
    find.stypy_localization = localization
    find.stypy_type_of_self = None
    find.stypy_type_store = module_type_store
    find.stypy_function_name = 'find'
    find.stypy_param_names_list = ['a', 'sub', 'start', 'end']
    find.stypy_varargs_param_name = None
    find.stypy_kwargs_param_name = None
    find.stypy_call_defaults = defaults
    find.stypy_call_varargs = varargs
    find.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'find', ['a', 'sub', 'start', 'end'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'find', localization, ['a', 'sub', 'start', 'end'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'find(...)' code ##################

    str_2461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 652, (-1)), 'str', '\n    For each element, return the lowest index in the string where\n    substring `sub` is found.\n\n    Calls `str.find` element-wise.\n\n    For each element, return the lowest index in the string where\n    substring `sub` is found, such that `sub` is contained in the\n    range [`start`, `end`].\n\n    Parameters\n    ----------\n    a : array_like of str or unicode\n\n    sub : str or unicode\n\n    start, end : int, optional\n        Optional arguments `start` and `end` are interpreted as in\n        slice notation.\n\n    Returns\n    -------\n    out : ndarray or int\n        Output array of ints.  Returns -1 if `sub` is not found.\n\n    See also\n    --------\n    str.find\n\n    ')
    
    # Call to _vec_string(...): (line 653)
    # Processing the call arguments (line 653)
    # Getting the type of 'a' (line 654)
    a_2463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 654, 8), 'a', False)
    # Getting the type of 'integer' (line 654)
    integer_2464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 654, 11), 'integer', False)
    str_2465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 654, 20), 'str', 'find')
    
    # Obtaining an instance of the builtin type 'list' (line 654)
    list_2466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 654, 28), 'list')
    # Adding type elements to the builtin type 'list' instance (line 654)
    # Adding element type (line 654)
    # Getting the type of 'sub' (line 654)
    sub_2467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 654, 29), 'sub', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 654, 28), list_2466, sub_2467)
    # Adding element type (line 654)
    # Getting the type of 'start' (line 654)
    start_2468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 654, 34), 'start', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 654, 28), list_2466, start_2468)
    
    
    # Call to _clean_args(...): (line 654)
    # Processing the call arguments (line 654)
    # Getting the type of 'end' (line 654)
    end_2470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 654, 55), 'end', False)
    # Processing the call keyword arguments (line 654)
    kwargs_2471 = {}
    # Getting the type of '_clean_args' (line 654)
    _clean_args_2469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 654, 43), '_clean_args', False)
    # Calling _clean_args(args, kwargs) (line 654)
    _clean_args_call_result_2472 = invoke(stypy.reporting.localization.Localization(__file__, 654, 43), _clean_args_2469, *[end_2470], **kwargs_2471)
    
    # Applying the binary operator '+' (line 654)
    result_add_2473 = python_operator(stypy.reporting.localization.Localization(__file__, 654, 28), '+', list_2466, _clean_args_call_result_2472)
    
    # Processing the call keyword arguments (line 653)
    kwargs_2474 = {}
    # Getting the type of '_vec_string' (line 653)
    _vec_string_2462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 11), '_vec_string', False)
    # Calling _vec_string(args, kwargs) (line 653)
    _vec_string_call_result_2475 = invoke(stypy.reporting.localization.Localization(__file__, 653, 11), _vec_string_2462, *[a_2463, integer_2464, str_2465, result_add_2473], **kwargs_2474)
    
    # Assigning a type to the variable 'stypy_return_type' (line 653)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 653, 4), 'stypy_return_type', _vec_string_call_result_2475)
    
    # ################# End of 'find(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'find' in the type store
    # Getting the type of 'stypy_return_type' (line 622)
    stypy_return_type_2476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_2476)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'find'
    return stypy_return_type_2476

# Assigning a type to the variable 'find' (line 622)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 622, 0), 'find', find)

@norecursion
def index(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_2477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 657, 24), 'int')
    # Getting the type of 'None' (line 657)
    None_2478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 31), 'None')
    defaults = [int_2477, None_2478]
    # Create a new context for function 'index'
    module_type_store = module_type_store.open_function_context('index', 657, 0, False)
    
    # Passed parameters checking function
    index.stypy_localization = localization
    index.stypy_type_of_self = None
    index.stypy_type_store = module_type_store
    index.stypy_function_name = 'index'
    index.stypy_param_names_list = ['a', 'sub', 'start', 'end']
    index.stypy_varargs_param_name = None
    index.stypy_kwargs_param_name = None
    index.stypy_call_defaults = defaults
    index.stypy_call_varargs = varargs
    index.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'index', ['a', 'sub', 'start', 'end'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'index', localization, ['a', 'sub', 'start', 'end'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'index(...)' code ##################

    str_2479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 680, (-1)), 'str', '\n    Like `find`, but raises `ValueError` when the substring is not found.\n\n    Calls `str.index` element-wise.\n\n    Parameters\n    ----------\n    a : array_like of str or unicode\n\n    sub : str or unicode\n\n    start, end : int, optional\n\n    Returns\n    -------\n    out : ndarray\n        Output array of ints.  Returns -1 if `sub` is not found.\n\n    See also\n    --------\n    find, str.find\n\n    ')
    
    # Call to _vec_string(...): (line 681)
    # Processing the call arguments (line 681)
    # Getting the type of 'a' (line 682)
    a_2481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 682, 8), 'a', False)
    # Getting the type of 'integer' (line 682)
    integer_2482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 682, 11), 'integer', False)
    str_2483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 682, 20), 'str', 'index')
    
    # Obtaining an instance of the builtin type 'list' (line 682)
    list_2484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 682, 29), 'list')
    # Adding type elements to the builtin type 'list' instance (line 682)
    # Adding element type (line 682)
    # Getting the type of 'sub' (line 682)
    sub_2485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 682, 30), 'sub', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 682, 29), list_2484, sub_2485)
    # Adding element type (line 682)
    # Getting the type of 'start' (line 682)
    start_2486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 682, 35), 'start', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 682, 29), list_2484, start_2486)
    
    
    # Call to _clean_args(...): (line 682)
    # Processing the call arguments (line 682)
    # Getting the type of 'end' (line 682)
    end_2488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 682, 56), 'end', False)
    # Processing the call keyword arguments (line 682)
    kwargs_2489 = {}
    # Getting the type of '_clean_args' (line 682)
    _clean_args_2487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 682, 44), '_clean_args', False)
    # Calling _clean_args(args, kwargs) (line 682)
    _clean_args_call_result_2490 = invoke(stypy.reporting.localization.Localization(__file__, 682, 44), _clean_args_2487, *[end_2488], **kwargs_2489)
    
    # Applying the binary operator '+' (line 682)
    result_add_2491 = python_operator(stypy.reporting.localization.Localization(__file__, 682, 29), '+', list_2484, _clean_args_call_result_2490)
    
    # Processing the call keyword arguments (line 681)
    kwargs_2492 = {}
    # Getting the type of '_vec_string' (line 681)
    _vec_string_2480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 11), '_vec_string', False)
    # Calling _vec_string(args, kwargs) (line 681)
    _vec_string_call_result_2493 = invoke(stypy.reporting.localization.Localization(__file__, 681, 11), _vec_string_2480, *[a_2481, integer_2482, str_2483, result_add_2491], **kwargs_2492)
    
    # Assigning a type to the variable 'stypy_return_type' (line 681)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 681, 4), 'stypy_return_type', _vec_string_call_result_2493)
    
    # ################# End of 'index(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'index' in the type store
    # Getting the type of 'stypy_return_type' (line 657)
    stypy_return_type_2494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_2494)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'index'
    return stypy_return_type_2494

# Assigning a type to the variable 'index' (line 657)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 657, 0), 'index', index)

@norecursion
def isalnum(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'isalnum'
    module_type_store = module_type_store.open_function_context('isalnum', 684, 0, False)
    
    # Passed parameters checking function
    isalnum.stypy_localization = localization
    isalnum.stypy_type_of_self = None
    isalnum.stypy_type_store = module_type_store
    isalnum.stypy_function_name = 'isalnum'
    isalnum.stypy_param_names_list = ['a']
    isalnum.stypy_varargs_param_name = None
    isalnum.stypy_kwargs_param_name = None
    isalnum.stypy_call_defaults = defaults
    isalnum.stypy_call_varargs = varargs
    isalnum.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'isalnum', ['a'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'isalnum', localization, ['a'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'isalnum(...)' code ##################

    str_2495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 705, (-1)), 'str', '\n    Returns true for each element if all characters in the string are\n    alphanumeric and there is at least one character, false otherwise.\n\n    Calls `str.isalnum` element-wise.\n\n    For 8-bit strings, this method is locale-dependent.\n\n    Parameters\n    ----------\n    a : array_like of str or unicode\n\n    Returns\n    -------\n    out : ndarray\n        Output array of str or unicode, depending on input type\n\n    See also\n    --------\n    str.isalnum\n    ')
    
    # Call to _vec_string(...): (line 706)
    # Processing the call arguments (line 706)
    # Getting the type of 'a' (line 706)
    a_2497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 706, 23), 'a', False)
    # Getting the type of 'bool_' (line 706)
    bool__2498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 706, 26), 'bool_', False)
    str_2499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 706, 33), 'str', 'isalnum')
    # Processing the call keyword arguments (line 706)
    kwargs_2500 = {}
    # Getting the type of '_vec_string' (line 706)
    _vec_string_2496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 706, 11), '_vec_string', False)
    # Calling _vec_string(args, kwargs) (line 706)
    _vec_string_call_result_2501 = invoke(stypy.reporting.localization.Localization(__file__, 706, 11), _vec_string_2496, *[a_2497, bool__2498, str_2499], **kwargs_2500)
    
    # Assigning a type to the variable 'stypy_return_type' (line 706)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 706, 4), 'stypy_return_type', _vec_string_call_result_2501)
    
    # ################# End of 'isalnum(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'isalnum' in the type store
    # Getting the type of 'stypy_return_type' (line 684)
    stypy_return_type_2502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 684, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_2502)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'isalnum'
    return stypy_return_type_2502

# Assigning a type to the variable 'isalnum' (line 684)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 684, 0), 'isalnum', isalnum)

@norecursion
def isalpha(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'isalpha'
    module_type_store = module_type_store.open_function_context('isalpha', 708, 0, False)
    
    # Passed parameters checking function
    isalpha.stypy_localization = localization
    isalpha.stypy_type_of_self = None
    isalpha.stypy_type_store = module_type_store
    isalpha.stypy_function_name = 'isalpha'
    isalpha.stypy_param_names_list = ['a']
    isalpha.stypy_varargs_param_name = None
    isalpha.stypy_kwargs_param_name = None
    isalpha.stypy_call_defaults = defaults
    isalpha.stypy_call_varargs = varargs
    isalpha.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'isalpha', ['a'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'isalpha', localization, ['a'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'isalpha(...)' code ##################

    str_2503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 729, (-1)), 'str', '\n    Returns true for each element if all characters in the string are\n    alphabetic and there is at least one character, false otherwise.\n\n    Calls `str.isalpha` element-wise.\n\n    For 8-bit strings, this method is locale-dependent.\n\n    Parameters\n    ----------\n    a : array_like of str or unicode\n\n    Returns\n    -------\n    out : ndarray\n        Output array of bools\n\n    See also\n    --------\n    str.isalpha\n    ')
    
    # Call to _vec_string(...): (line 730)
    # Processing the call arguments (line 730)
    # Getting the type of 'a' (line 730)
    a_2505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 730, 23), 'a', False)
    # Getting the type of 'bool_' (line 730)
    bool__2506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 730, 26), 'bool_', False)
    str_2507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 730, 33), 'str', 'isalpha')
    # Processing the call keyword arguments (line 730)
    kwargs_2508 = {}
    # Getting the type of '_vec_string' (line 730)
    _vec_string_2504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 730, 11), '_vec_string', False)
    # Calling _vec_string(args, kwargs) (line 730)
    _vec_string_call_result_2509 = invoke(stypy.reporting.localization.Localization(__file__, 730, 11), _vec_string_2504, *[a_2505, bool__2506, str_2507], **kwargs_2508)
    
    # Assigning a type to the variable 'stypy_return_type' (line 730)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 730, 4), 'stypy_return_type', _vec_string_call_result_2509)
    
    # ################# End of 'isalpha(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'isalpha' in the type store
    # Getting the type of 'stypy_return_type' (line 708)
    stypy_return_type_2510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 708, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_2510)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'isalpha'
    return stypy_return_type_2510

# Assigning a type to the variable 'isalpha' (line 708)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 708, 0), 'isalpha', isalpha)

@norecursion
def isdigit(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'isdigit'
    module_type_store = module_type_store.open_function_context('isdigit', 732, 0, False)
    
    # Passed parameters checking function
    isdigit.stypy_localization = localization
    isdigit.stypy_type_of_self = None
    isdigit.stypy_type_store = module_type_store
    isdigit.stypy_function_name = 'isdigit'
    isdigit.stypy_param_names_list = ['a']
    isdigit.stypy_varargs_param_name = None
    isdigit.stypy_kwargs_param_name = None
    isdigit.stypy_call_defaults = defaults
    isdigit.stypy_call_varargs = varargs
    isdigit.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'isdigit', ['a'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'isdigit', localization, ['a'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'isdigit(...)' code ##################

    str_2511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 753, (-1)), 'str', '\n    Returns true for each element if all characters in the string are\n    digits and there is at least one character, false otherwise.\n\n    Calls `str.isdigit` element-wise.\n\n    For 8-bit strings, this method is locale-dependent.\n\n    Parameters\n    ----------\n    a : array_like of str or unicode\n\n    Returns\n    -------\n    out : ndarray\n        Output array of bools\n\n    See also\n    --------\n    str.isdigit\n    ')
    
    # Call to _vec_string(...): (line 754)
    # Processing the call arguments (line 754)
    # Getting the type of 'a' (line 754)
    a_2513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 754, 23), 'a', False)
    # Getting the type of 'bool_' (line 754)
    bool__2514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 754, 26), 'bool_', False)
    str_2515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 754, 33), 'str', 'isdigit')
    # Processing the call keyword arguments (line 754)
    kwargs_2516 = {}
    # Getting the type of '_vec_string' (line 754)
    _vec_string_2512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 754, 11), '_vec_string', False)
    # Calling _vec_string(args, kwargs) (line 754)
    _vec_string_call_result_2517 = invoke(stypy.reporting.localization.Localization(__file__, 754, 11), _vec_string_2512, *[a_2513, bool__2514, str_2515], **kwargs_2516)
    
    # Assigning a type to the variable 'stypy_return_type' (line 754)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 754, 4), 'stypy_return_type', _vec_string_call_result_2517)
    
    # ################# End of 'isdigit(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'isdigit' in the type store
    # Getting the type of 'stypy_return_type' (line 732)
    stypy_return_type_2518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 732, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_2518)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'isdigit'
    return stypy_return_type_2518

# Assigning a type to the variable 'isdigit' (line 732)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 732, 0), 'isdigit', isdigit)

@norecursion
def islower(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'islower'
    module_type_store = module_type_store.open_function_context('islower', 756, 0, False)
    
    # Passed parameters checking function
    islower.stypy_localization = localization
    islower.stypy_type_of_self = None
    islower.stypy_type_store = module_type_store
    islower.stypy_function_name = 'islower'
    islower.stypy_param_names_list = ['a']
    islower.stypy_varargs_param_name = None
    islower.stypy_kwargs_param_name = None
    islower.stypy_call_defaults = defaults
    islower.stypy_call_varargs = varargs
    islower.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'islower', ['a'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'islower', localization, ['a'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'islower(...)' code ##################

    str_2519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 778, (-1)), 'str', '\n    Returns true for each element if all cased characters in the\n    string are lowercase and there is at least one cased character,\n    false otherwise.\n\n    Calls `str.islower` element-wise.\n\n    For 8-bit strings, this method is locale-dependent.\n\n    Parameters\n    ----------\n    a : array_like of str or unicode\n\n    Returns\n    -------\n    out : ndarray\n        Output array of bools\n\n    See also\n    --------\n    str.islower\n    ')
    
    # Call to _vec_string(...): (line 779)
    # Processing the call arguments (line 779)
    # Getting the type of 'a' (line 779)
    a_2521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 779, 23), 'a', False)
    # Getting the type of 'bool_' (line 779)
    bool__2522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 779, 26), 'bool_', False)
    str_2523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 779, 33), 'str', 'islower')
    # Processing the call keyword arguments (line 779)
    kwargs_2524 = {}
    # Getting the type of '_vec_string' (line 779)
    _vec_string_2520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 779, 11), '_vec_string', False)
    # Calling _vec_string(args, kwargs) (line 779)
    _vec_string_call_result_2525 = invoke(stypy.reporting.localization.Localization(__file__, 779, 11), _vec_string_2520, *[a_2521, bool__2522, str_2523], **kwargs_2524)
    
    # Assigning a type to the variable 'stypy_return_type' (line 779)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 779, 4), 'stypy_return_type', _vec_string_call_result_2525)
    
    # ################# End of 'islower(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'islower' in the type store
    # Getting the type of 'stypy_return_type' (line 756)
    stypy_return_type_2526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 756, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_2526)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'islower'
    return stypy_return_type_2526

# Assigning a type to the variable 'islower' (line 756)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 756, 0), 'islower', islower)

@norecursion
def isspace(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'isspace'
    module_type_store = module_type_store.open_function_context('isspace', 781, 0, False)
    
    # Passed parameters checking function
    isspace.stypy_localization = localization
    isspace.stypy_type_of_self = None
    isspace.stypy_type_store = module_type_store
    isspace.stypy_function_name = 'isspace'
    isspace.stypy_param_names_list = ['a']
    isspace.stypy_varargs_param_name = None
    isspace.stypy_kwargs_param_name = None
    isspace.stypy_call_defaults = defaults
    isspace.stypy_call_varargs = varargs
    isspace.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'isspace', ['a'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'isspace', localization, ['a'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'isspace(...)' code ##################

    str_2527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 803, (-1)), 'str', '\n    Returns true for each element if there are only whitespace\n    characters in the string and there is at least one character,\n    false otherwise.\n\n    Calls `str.isspace` element-wise.\n\n    For 8-bit strings, this method is locale-dependent.\n\n    Parameters\n    ----------\n    a : array_like of str or unicode\n\n    Returns\n    -------\n    out : ndarray\n        Output array of bools\n\n    See also\n    --------\n    str.isspace\n    ')
    
    # Call to _vec_string(...): (line 804)
    # Processing the call arguments (line 804)
    # Getting the type of 'a' (line 804)
    a_2529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 804, 23), 'a', False)
    # Getting the type of 'bool_' (line 804)
    bool__2530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 804, 26), 'bool_', False)
    str_2531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 804, 33), 'str', 'isspace')
    # Processing the call keyword arguments (line 804)
    kwargs_2532 = {}
    # Getting the type of '_vec_string' (line 804)
    _vec_string_2528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 804, 11), '_vec_string', False)
    # Calling _vec_string(args, kwargs) (line 804)
    _vec_string_call_result_2533 = invoke(stypy.reporting.localization.Localization(__file__, 804, 11), _vec_string_2528, *[a_2529, bool__2530, str_2531], **kwargs_2532)
    
    # Assigning a type to the variable 'stypy_return_type' (line 804)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 804, 4), 'stypy_return_type', _vec_string_call_result_2533)
    
    # ################# End of 'isspace(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'isspace' in the type store
    # Getting the type of 'stypy_return_type' (line 781)
    stypy_return_type_2534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 781, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_2534)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'isspace'
    return stypy_return_type_2534

# Assigning a type to the variable 'isspace' (line 781)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 781, 0), 'isspace', isspace)

@norecursion
def istitle(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'istitle'
    module_type_store = module_type_store.open_function_context('istitle', 806, 0, False)
    
    # Passed parameters checking function
    istitle.stypy_localization = localization
    istitle.stypy_type_of_self = None
    istitle.stypy_type_store = module_type_store
    istitle.stypy_function_name = 'istitle'
    istitle.stypy_param_names_list = ['a']
    istitle.stypy_varargs_param_name = None
    istitle.stypy_kwargs_param_name = None
    istitle.stypy_call_defaults = defaults
    istitle.stypy_call_varargs = varargs
    istitle.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'istitle', ['a'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'istitle', localization, ['a'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'istitle(...)' code ##################

    str_2535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 827, (-1)), 'str', '\n    Returns true for each element if the element is a titlecased\n    string and there is at least one character, false otherwise.\n\n    Call `str.istitle` element-wise.\n\n    For 8-bit strings, this method is locale-dependent.\n\n    Parameters\n    ----------\n    a : array_like of str or unicode\n\n    Returns\n    -------\n    out : ndarray\n        Output array of bools\n\n    See also\n    --------\n    str.istitle\n    ')
    
    # Call to _vec_string(...): (line 828)
    # Processing the call arguments (line 828)
    # Getting the type of 'a' (line 828)
    a_2537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 828, 23), 'a', False)
    # Getting the type of 'bool_' (line 828)
    bool__2538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 828, 26), 'bool_', False)
    str_2539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 828, 33), 'str', 'istitle')
    # Processing the call keyword arguments (line 828)
    kwargs_2540 = {}
    # Getting the type of '_vec_string' (line 828)
    _vec_string_2536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 828, 11), '_vec_string', False)
    # Calling _vec_string(args, kwargs) (line 828)
    _vec_string_call_result_2541 = invoke(stypy.reporting.localization.Localization(__file__, 828, 11), _vec_string_2536, *[a_2537, bool__2538, str_2539], **kwargs_2540)
    
    # Assigning a type to the variable 'stypy_return_type' (line 828)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 828, 4), 'stypy_return_type', _vec_string_call_result_2541)
    
    # ################# End of 'istitle(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'istitle' in the type store
    # Getting the type of 'stypy_return_type' (line 806)
    stypy_return_type_2542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 806, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_2542)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'istitle'
    return stypy_return_type_2542

# Assigning a type to the variable 'istitle' (line 806)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 806, 0), 'istitle', istitle)

@norecursion
def isupper(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'isupper'
    module_type_store = module_type_store.open_function_context('isupper', 830, 0, False)
    
    # Passed parameters checking function
    isupper.stypy_localization = localization
    isupper.stypy_type_of_self = None
    isupper.stypy_type_store = module_type_store
    isupper.stypy_function_name = 'isupper'
    isupper.stypy_param_names_list = ['a']
    isupper.stypy_varargs_param_name = None
    isupper.stypy_kwargs_param_name = None
    isupper.stypy_call_defaults = defaults
    isupper.stypy_call_varargs = varargs
    isupper.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'isupper', ['a'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'isupper', localization, ['a'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'isupper(...)' code ##################

    str_2543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 852, (-1)), 'str', '\n    Returns true for each element if all cased characters in the\n    string are uppercase and there is at least one character, false\n    otherwise.\n\n    Call `str.isupper` element-wise.\n\n    For 8-bit strings, this method is locale-dependent.\n\n    Parameters\n    ----------\n    a : array_like of str or unicode\n\n    Returns\n    -------\n    out : ndarray\n        Output array of bools\n\n    See also\n    --------\n    str.isupper\n    ')
    
    # Call to _vec_string(...): (line 853)
    # Processing the call arguments (line 853)
    # Getting the type of 'a' (line 853)
    a_2545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 853, 23), 'a', False)
    # Getting the type of 'bool_' (line 853)
    bool__2546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 853, 26), 'bool_', False)
    str_2547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 853, 33), 'str', 'isupper')
    # Processing the call keyword arguments (line 853)
    kwargs_2548 = {}
    # Getting the type of '_vec_string' (line 853)
    _vec_string_2544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 853, 11), '_vec_string', False)
    # Calling _vec_string(args, kwargs) (line 853)
    _vec_string_call_result_2549 = invoke(stypy.reporting.localization.Localization(__file__, 853, 11), _vec_string_2544, *[a_2545, bool__2546, str_2547], **kwargs_2548)
    
    # Assigning a type to the variable 'stypy_return_type' (line 853)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 853, 4), 'stypy_return_type', _vec_string_call_result_2549)
    
    # ################# End of 'isupper(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'isupper' in the type store
    # Getting the type of 'stypy_return_type' (line 830)
    stypy_return_type_2550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 830, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_2550)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'isupper'
    return stypy_return_type_2550

# Assigning a type to the variable 'isupper' (line 830)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 830, 0), 'isupper', isupper)

@norecursion
def join(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'join'
    module_type_store = module_type_store.open_function_context('join', 855, 0, False)
    
    # Passed parameters checking function
    join.stypy_localization = localization
    join.stypy_type_of_self = None
    join.stypy_type_store = module_type_store
    join.stypy_function_name = 'join'
    join.stypy_param_names_list = ['sep', 'seq']
    join.stypy_varargs_param_name = None
    join.stypy_kwargs_param_name = None
    join.stypy_call_defaults = defaults
    join.stypy_call_varargs = varargs
    join.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'join', ['sep', 'seq'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'join', localization, ['sep', 'seq'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'join(...)' code ##################

    str_2551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 875, (-1)), 'str', '\n    Return a string which is the concatenation of the strings in the\n    sequence `seq`.\n\n    Calls `str.join` element-wise.\n\n    Parameters\n    ----------\n    sep : array_like of str or unicode\n    seq : array_like of str or unicode\n\n    Returns\n    -------\n    out : ndarray\n        Output array of str or unicode, depending on input types\n\n    See also\n    --------\n    str.join\n    ')
    
    # Call to _to_string_or_unicode_array(...): (line 876)
    # Processing the call arguments (line 876)
    
    # Call to _vec_string(...): (line 877)
    # Processing the call arguments (line 877)
    # Getting the type of 'sep' (line 877)
    sep_2554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 877, 20), 'sep', False)
    # Getting the type of 'object_' (line 877)
    object__2555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 877, 25), 'object_', False)
    str_2556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 877, 34), 'str', 'join')
    
    # Obtaining an instance of the builtin type 'tuple' (line 877)
    tuple_2557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 877, 43), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 877)
    # Adding element type (line 877)
    # Getting the type of 'seq' (line 877)
    seq_2558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 877, 43), 'seq', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 877, 43), tuple_2557, seq_2558)
    
    # Processing the call keyword arguments (line 877)
    kwargs_2559 = {}
    # Getting the type of '_vec_string' (line 877)
    _vec_string_2553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 877, 8), '_vec_string', False)
    # Calling _vec_string(args, kwargs) (line 877)
    _vec_string_call_result_2560 = invoke(stypy.reporting.localization.Localization(__file__, 877, 8), _vec_string_2553, *[sep_2554, object__2555, str_2556, tuple_2557], **kwargs_2559)
    
    # Processing the call keyword arguments (line 876)
    kwargs_2561 = {}
    # Getting the type of '_to_string_or_unicode_array' (line 876)
    _to_string_or_unicode_array_2552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 876, 11), '_to_string_or_unicode_array', False)
    # Calling _to_string_or_unicode_array(args, kwargs) (line 876)
    _to_string_or_unicode_array_call_result_2562 = invoke(stypy.reporting.localization.Localization(__file__, 876, 11), _to_string_or_unicode_array_2552, *[_vec_string_call_result_2560], **kwargs_2561)
    
    # Assigning a type to the variable 'stypy_return_type' (line 876)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 876, 4), 'stypy_return_type', _to_string_or_unicode_array_call_result_2562)
    
    # ################# End of 'join(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'join' in the type store
    # Getting the type of 'stypy_return_type' (line 855)
    stypy_return_type_2563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 855, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_2563)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'join'
    return stypy_return_type_2563

# Assigning a type to the variable 'join' (line 855)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 855, 0), 'join', join)

@norecursion
def ljust(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_2564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 880, 29), 'str', ' ')
    defaults = [str_2564]
    # Create a new context for function 'ljust'
    module_type_store = module_type_store.open_function_context('ljust', 880, 0, False)
    
    # Passed parameters checking function
    ljust.stypy_localization = localization
    ljust.stypy_type_of_self = None
    ljust.stypy_type_store = module_type_store
    ljust.stypy_function_name = 'ljust'
    ljust.stypy_param_names_list = ['a', 'width', 'fillchar']
    ljust.stypy_varargs_param_name = None
    ljust.stypy_kwargs_param_name = None
    ljust.stypy_call_defaults = defaults
    ljust.stypy_call_varargs = varargs
    ljust.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'ljust', ['a', 'width', 'fillchar'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'ljust', localization, ['a', 'width', 'fillchar'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'ljust(...)' code ##################

    str_2565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 905, (-1)), 'str', '\n    Return an array with the elements of `a` left-justified in a\n    string of length `width`.\n\n    Calls `str.ljust` element-wise.\n\n    Parameters\n    ----------\n    a : array_like of str or unicode\n\n    width : int\n        The length of the resulting strings\n    fillchar : str or unicode, optional\n        The character to use for padding\n\n    Returns\n    -------\n    out : ndarray\n        Output array of str or unicode, depending on input type\n\n    See also\n    --------\n    str.ljust\n\n    ')
    
    # Assigning a Call to a Name (line 906):
    
    # Call to asarray(...): (line 906)
    # Processing the call arguments (line 906)
    # Getting the type of 'a' (line 906)
    a_2568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 906, 26), 'a', False)
    # Processing the call keyword arguments (line 906)
    kwargs_2569 = {}
    # Getting the type of 'numpy' (line 906)
    numpy_2566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 906, 12), 'numpy', False)
    # Obtaining the member 'asarray' of a type (line 906)
    asarray_2567 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 906, 12), numpy_2566, 'asarray')
    # Calling asarray(args, kwargs) (line 906)
    asarray_call_result_2570 = invoke(stypy.reporting.localization.Localization(__file__, 906, 12), asarray_2567, *[a_2568], **kwargs_2569)
    
    # Assigning a type to the variable 'a_arr' (line 906)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 906, 4), 'a_arr', asarray_call_result_2570)
    
    # Assigning a Call to a Name (line 907):
    
    # Call to asarray(...): (line 907)
    # Processing the call arguments (line 907)
    # Getting the type of 'width' (line 907)
    width_2573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 907, 30), 'width', False)
    # Processing the call keyword arguments (line 907)
    kwargs_2574 = {}
    # Getting the type of 'numpy' (line 907)
    numpy_2571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 907, 16), 'numpy', False)
    # Obtaining the member 'asarray' of a type (line 907)
    asarray_2572 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 907, 16), numpy_2571, 'asarray')
    # Calling asarray(args, kwargs) (line 907)
    asarray_call_result_2575 = invoke(stypy.reporting.localization.Localization(__file__, 907, 16), asarray_2572, *[width_2573], **kwargs_2574)
    
    # Assigning a type to the variable 'width_arr' (line 907)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 907, 4), 'width_arr', asarray_call_result_2575)
    
    # Assigning a Call to a Name (line 908):
    
    # Call to long(...): (line 908)
    # Processing the call arguments (line 908)
    
    # Call to max(...): (line 908)
    # Processing the call arguments (line 908)
    # Getting the type of 'width_arr' (line 908)
    width_arr_2579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 908, 26), 'width_arr', False)
    # Obtaining the member 'flat' of a type (line 908)
    flat_2580 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 908, 26), width_arr_2579, 'flat')
    # Processing the call keyword arguments (line 908)
    kwargs_2581 = {}
    # Getting the type of 'numpy' (line 908)
    numpy_2577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 908, 16), 'numpy', False)
    # Obtaining the member 'max' of a type (line 908)
    max_2578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 908, 16), numpy_2577, 'max')
    # Calling max(args, kwargs) (line 908)
    max_call_result_2582 = invoke(stypy.reporting.localization.Localization(__file__, 908, 16), max_2578, *[flat_2580], **kwargs_2581)
    
    # Processing the call keyword arguments (line 908)
    kwargs_2583 = {}
    # Getting the type of 'long' (line 908)
    long_2576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 908, 11), 'long', False)
    # Calling long(args, kwargs) (line 908)
    long_call_result_2584 = invoke(stypy.reporting.localization.Localization(__file__, 908, 11), long_2576, *[max_call_result_2582], **kwargs_2583)
    
    # Assigning a type to the variable 'size' (line 908)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 908, 4), 'size', long_call_result_2584)
    
    
    # Call to issubdtype(...): (line 909)
    # Processing the call arguments (line 909)
    # Getting the type of 'a_arr' (line 909)
    a_arr_2587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 909, 24), 'a_arr', False)
    # Obtaining the member 'dtype' of a type (line 909)
    dtype_2588 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 909, 24), a_arr_2587, 'dtype')
    # Getting the type of 'numpy' (line 909)
    numpy_2589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 909, 37), 'numpy', False)
    # Obtaining the member 'string_' of a type (line 909)
    string__2590 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 909, 37), numpy_2589, 'string_')
    # Processing the call keyword arguments (line 909)
    kwargs_2591 = {}
    # Getting the type of 'numpy' (line 909)
    numpy_2585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 909, 7), 'numpy', False)
    # Obtaining the member 'issubdtype' of a type (line 909)
    issubdtype_2586 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 909, 7), numpy_2585, 'issubdtype')
    # Calling issubdtype(args, kwargs) (line 909)
    issubdtype_call_result_2592 = invoke(stypy.reporting.localization.Localization(__file__, 909, 7), issubdtype_2586, *[dtype_2588, string__2590], **kwargs_2591)
    
    # Testing the type of an if condition (line 909)
    if_condition_2593 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 909, 4), issubdtype_call_result_2592)
    # Assigning a type to the variable 'if_condition_2593' (line 909)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 909, 4), 'if_condition_2593', if_condition_2593)
    # SSA begins for if statement (line 909)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 910):
    
    # Call to asbytes(...): (line 910)
    # Processing the call arguments (line 910)
    # Getting the type of 'fillchar' (line 910)
    fillchar_2595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 910, 27), 'fillchar', False)
    # Processing the call keyword arguments (line 910)
    kwargs_2596 = {}
    # Getting the type of 'asbytes' (line 910)
    asbytes_2594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 910, 19), 'asbytes', False)
    # Calling asbytes(args, kwargs) (line 910)
    asbytes_call_result_2597 = invoke(stypy.reporting.localization.Localization(__file__, 910, 19), asbytes_2594, *[fillchar_2595], **kwargs_2596)
    
    # Assigning a type to the variable 'fillchar' (line 910)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 910, 8), 'fillchar', asbytes_call_result_2597)
    # SSA join for if statement (line 909)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to _vec_string(...): (line 911)
    # Processing the call arguments (line 911)
    # Getting the type of 'a_arr' (line 912)
    a_arr_2599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 912, 8), 'a_arr', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 912)
    tuple_2600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 912, 16), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 912)
    # Adding element type (line 912)
    # Getting the type of 'a_arr' (line 912)
    a_arr_2601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 912, 16), 'a_arr', False)
    # Obtaining the member 'dtype' of a type (line 912)
    dtype_2602 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 912, 16), a_arr_2601, 'dtype')
    # Obtaining the member 'type' of a type (line 912)
    type_2603 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 912, 16), dtype_2602, 'type')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 912, 16), tuple_2600, type_2603)
    # Adding element type (line 912)
    # Getting the type of 'size' (line 912)
    size_2604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 912, 34), 'size', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 912, 16), tuple_2600, size_2604)
    
    str_2605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 912, 41), 'str', 'ljust')
    
    # Obtaining an instance of the builtin type 'tuple' (line 912)
    tuple_2606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 912, 51), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 912)
    # Adding element type (line 912)
    # Getting the type of 'width_arr' (line 912)
    width_arr_2607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 912, 51), 'width_arr', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 912, 51), tuple_2606, width_arr_2607)
    # Adding element type (line 912)
    # Getting the type of 'fillchar' (line 912)
    fillchar_2608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 912, 62), 'fillchar', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 912, 51), tuple_2606, fillchar_2608)
    
    # Processing the call keyword arguments (line 911)
    kwargs_2609 = {}
    # Getting the type of '_vec_string' (line 911)
    _vec_string_2598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 911, 11), '_vec_string', False)
    # Calling _vec_string(args, kwargs) (line 911)
    _vec_string_call_result_2610 = invoke(stypy.reporting.localization.Localization(__file__, 911, 11), _vec_string_2598, *[a_arr_2599, tuple_2600, str_2605, tuple_2606], **kwargs_2609)
    
    # Assigning a type to the variable 'stypy_return_type' (line 911)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 911, 4), 'stypy_return_type', _vec_string_call_result_2610)
    
    # ################# End of 'ljust(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'ljust' in the type store
    # Getting the type of 'stypy_return_type' (line 880)
    stypy_return_type_2611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 880, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_2611)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'ljust'
    return stypy_return_type_2611

# Assigning a type to the variable 'ljust' (line 880)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 880, 0), 'ljust', ljust)

@norecursion
def lower(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'lower'
    module_type_store = module_type_store.open_function_context('lower', 915, 0, False)
    
    # Passed parameters checking function
    lower.stypy_localization = localization
    lower.stypy_type_of_self = None
    lower.stypy_type_store = module_type_store
    lower.stypy_function_name = 'lower'
    lower.stypy_param_names_list = ['a']
    lower.stypy_varargs_param_name = None
    lower.stypy_kwargs_param_name = None
    lower.stypy_call_defaults = defaults
    lower.stypy_call_varargs = varargs
    lower.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'lower', ['a'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'lower', localization, ['a'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'lower(...)' code ##################

    str_2612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 946, (-1)), 'str', "\n    Return an array with the elements converted to lowercase.\n\n    Call `str.lower` element-wise.\n\n    For 8-bit strings, this method is locale-dependent.\n\n    Parameters\n    ----------\n    a : array_like, {str, unicode}\n        Input array.\n\n    Returns\n    -------\n    out : ndarray, {str, unicode}\n        Output array of str or unicode, depending on input type\n\n    See also\n    --------\n    str.lower\n\n    Examples\n    --------\n    >>> c = np.array(['A1B C', '1BCA', 'BCA1']); c\n    array(['A1B C', '1BCA', 'BCA1'],\n          dtype='|S5')\n    >>> np.char.lower(c)\n    array(['a1b c', '1bca', 'bca1'],\n          dtype='|S5')\n\n    ")
    
    # Assigning a Call to a Name (line 947):
    
    # Call to asarray(...): (line 947)
    # Processing the call arguments (line 947)
    # Getting the type of 'a' (line 947)
    a_2615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 947, 26), 'a', False)
    # Processing the call keyword arguments (line 947)
    kwargs_2616 = {}
    # Getting the type of 'numpy' (line 947)
    numpy_2613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 947, 12), 'numpy', False)
    # Obtaining the member 'asarray' of a type (line 947)
    asarray_2614 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 947, 12), numpy_2613, 'asarray')
    # Calling asarray(args, kwargs) (line 947)
    asarray_call_result_2617 = invoke(stypy.reporting.localization.Localization(__file__, 947, 12), asarray_2614, *[a_2615], **kwargs_2616)
    
    # Assigning a type to the variable 'a_arr' (line 947)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 947, 4), 'a_arr', asarray_call_result_2617)
    
    # Call to _vec_string(...): (line 948)
    # Processing the call arguments (line 948)
    # Getting the type of 'a_arr' (line 948)
    a_arr_2619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 948, 23), 'a_arr', False)
    # Getting the type of 'a_arr' (line 948)
    a_arr_2620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 948, 30), 'a_arr', False)
    # Obtaining the member 'dtype' of a type (line 948)
    dtype_2621 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 948, 30), a_arr_2620, 'dtype')
    str_2622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 948, 43), 'str', 'lower')
    # Processing the call keyword arguments (line 948)
    kwargs_2623 = {}
    # Getting the type of '_vec_string' (line 948)
    _vec_string_2618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 948, 11), '_vec_string', False)
    # Calling _vec_string(args, kwargs) (line 948)
    _vec_string_call_result_2624 = invoke(stypy.reporting.localization.Localization(__file__, 948, 11), _vec_string_2618, *[a_arr_2619, dtype_2621, str_2622], **kwargs_2623)
    
    # Assigning a type to the variable 'stypy_return_type' (line 948)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 948, 4), 'stypy_return_type', _vec_string_call_result_2624)
    
    # ################# End of 'lower(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'lower' in the type store
    # Getting the type of 'stypy_return_type' (line 915)
    stypy_return_type_2625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 915, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_2625)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'lower'
    return stypy_return_type_2625

# Assigning a type to the variable 'lower' (line 915)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 915, 0), 'lower', lower)

@norecursion
def lstrip(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 951)
    None_2626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 951, 20), 'None')
    defaults = [None_2626]
    # Create a new context for function 'lstrip'
    module_type_store = module_type_store.open_function_context('lstrip', 951, 0, False)
    
    # Passed parameters checking function
    lstrip.stypy_localization = localization
    lstrip.stypy_type_of_self = None
    lstrip.stypy_type_store = module_type_store
    lstrip.stypy_function_name = 'lstrip'
    lstrip.stypy_param_names_list = ['a', 'chars']
    lstrip.stypy_varargs_param_name = None
    lstrip.stypy_kwargs_param_name = None
    lstrip.stypy_call_defaults = defaults
    lstrip.stypy_call_varargs = varargs
    lstrip.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'lstrip', ['a', 'chars'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'lstrip', localization, ['a', 'chars'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'lstrip(...)' code ##################

    str_2627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1003, (-1)), 'str', "\n    For each element in `a`, return a copy with the leading characters\n    removed.\n\n    Calls `str.lstrip` element-wise.\n\n    Parameters\n    ----------\n    a : array-like, {str, unicode}\n        Input array.\n\n    chars : {str, unicode}, optional\n        The `chars` argument is a string specifying the set of\n        characters to be removed. If omitted or None, the `chars`\n        argument defaults to removing whitespace. The `chars` argument\n        is not a prefix; rather, all combinations of its values are\n        stripped.\n\n    Returns\n    -------\n    out : ndarray, {str, unicode}\n        Output array of str or unicode, depending on input type\n\n    See also\n    --------\n    str.lstrip\n\n    Examples\n    --------\n    >>> c = np.array(['aAaAaA', '  aA  ', 'abBABba'])\n    >>> c\n    array(['aAaAaA', '  aA  ', 'abBABba'],\n        dtype='|S7')\n\n    The 'a' variable is unstripped from c[1] because whitespace leading.\n\n    >>> np.char.lstrip(c, 'a')\n    array(['AaAaA', '  aA  ', 'bBABba'],\n        dtype='|S7')\n\n\n    >>> np.char.lstrip(c, 'A') # leaves c unchanged\n    array(['aAaAaA', '  aA  ', 'abBABba'],\n        dtype='|S7')\n    >>> (np.char.lstrip(c, ' ') == np.char.lstrip(c, '')).all()\n    ... # XXX: is this a regression? this line now returns False\n    ... # np.char.lstrip(c,'') does not modify c at all.\n    True\n    >>> (np.char.lstrip(c, ' ') == np.char.lstrip(c, None)).all()\n    True\n\n    ")
    
    # Assigning a Call to a Name (line 1004):
    
    # Call to asarray(...): (line 1004)
    # Processing the call arguments (line 1004)
    # Getting the type of 'a' (line 1004)
    a_2630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1004, 26), 'a', False)
    # Processing the call keyword arguments (line 1004)
    kwargs_2631 = {}
    # Getting the type of 'numpy' (line 1004)
    numpy_2628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1004, 12), 'numpy', False)
    # Obtaining the member 'asarray' of a type (line 1004)
    asarray_2629 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1004, 12), numpy_2628, 'asarray')
    # Calling asarray(args, kwargs) (line 1004)
    asarray_call_result_2632 = invoke(stypy.reporting.localization.Localization(__file__, 1004, 12), asarray_2629, *[a_2630], **kwargs_2631)
    
    # Assigning a type to the variable 'a_arr' (line 1004)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1004, 4), 'a_arr', asarray_call_result_2632)
    
    # Call to _vec_string(...): (line 1005)
    # Processing the call arguments (line 1005)
    # Getting the type of 'a_arr' (line 1005)
    a_arr_2634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1005, 23), 'a_arr', False)
    # Getting the type of 'a_arr' (line 1005)
    a_arr_2635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1005, 30), 'a_arr', False)
    # Obtaining the member 'dtype' of a type (line 1005)
    dtype_2636 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1005, 30), a_arr_2635, 'dtype')
    str_2637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1005, 43), 'str', 'lstrip')
    
    # Obtaining an instance of the builtin type 'tuple' (line 1005)
    tuple_2638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1005, 54), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1005)
    # Adding element type (line 1005)
    # Getting the type of 'chars' (line 1005)
    chars_2639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1005, 54), 'chars', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1005, 54), tuple_2638, chars_2639)
    
    # Processing the call keyword arguments (line 1005)
    kwargs_2640 = {}
    # Getting the type of '_vec_string' (line 1005)
    _vec_string_2633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1005, 11), '_vec_string', False)
    # Calling _vec_string(args, kwargs) (line 1005)
    _vec_string_call_result_2641 = invoke(stypy.reporting.localization.Localization(__file__, 1005, 11), _vec_string_2633, *[a_arr_2634, dtype_2636, str_2637, tuple_2638], **kwargs_2640)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1005)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1005, 4), 'stypy_return_type', _vec_string_call_result_2641)
    
    # ################# End of 'lstrip(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'lstrip' in the type store
    # Getting the type of 'stypy_return_type' (line 951)
    stypy_return_type_2642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 951, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_2642)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'lstrip'
    return stypy_return_type_2642

# Assigning a type to the variable 'lstrip' (line 951)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 951, 0), 'lstrip', lstrip)

@norecursion
def partition(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'partition'
    module_type_store = module_type_store.open_function_context('partition', 1008, 0, False)
    
    # Passed parameters checking function
    partition.stypy_localization = localization
    partition.stypy_type_of_self = None
    partition.stypy_type_store = module_type_store
    partition.stypy_function_name = 'partition'
    partition.stypy_param_names_list = ['a', 'sep']
    partition.stypy_varargs_param_name = None
    partition.stypy_kwargs_param_name = None
    partition.stypy_call_defaults = defaults
    partition.stypy_call_varargs = varargs
    partition.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'partition', ['a', 'sep'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'partition', localization, ['a', 'sep'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'partition(...)' code ##################

    str_2643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1038, (-1)), 'str', '\n    Partition each element in `a` around `sep`.\n\n    Calls `str.partition` element-wise.\n\n    For each element in `a`, split the element as the first\n    occurrence of `sep`, and return 3 strings containing the part\n    before the separator, the separator itself, and the part after\n    the separator. If the separator is not found, return 3 strings\n    containing the string itself, followed by two empty strings.\n\n    Parameters\n    ----------\n    a : array_like, {str, unicode}\n        Input array\n    sep : {str, unicode}\n        Separator to split each string element in `a`.\n\n    Returns\n    -------\n    out : ndarray, {str, unicode}\n        Output array of str or unicode, depending on input type.\n        The output array will have an extra dimension with 3\n        elements per input element.\n\n    See also\n    --------\n    str.partition\n\n    ')
    
    # Call to _to_string_or_unicode_array(...): (line 1039)
    # Processing the call arguments (line 1039)
    
    # Call to _vec_string(...): (line 1040)
    # Processing the call arguments (line 1040)
    # Getting the type of 'a' (line 1040)
    a_2646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1040, 20), 'a', False)
    # Getting the type of 'object_' (line 1040)
    object__2647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1040, 23), 'object_', False)
    str_2648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1040, 32), 'str', 'partition')
    
    # Obtaining an instance of the builtin type 'tuple' (line 1040)
    tuple_2649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1040, 46), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1040)
    # Adding element type (line 1040)
    # Getting the type of 'sep' (line 1040)
    sep_2650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1040, 46), 'sep', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1040, 46), tuple_2649, sep_2650)
    
    # Processing the call keyword arguments (line 1040)
    kwargs_2651 = {}
    # Getting the type of '_vec_string' (line 1040)
    _vec_string_2645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1040, 8), '_vec_string', False)
    # Calling _vec_string(args, kwargs) (line 1040)
    _vec_string_call_result_2652 = invoke(stypy.reporting.localization.Localization(__file__, 1040, 8), _vec_string_2645, *[a_2646, object__2647, str_2648, tuple_2649], **kwargs_2651)
    
    # Processing the call keyword arguments (line 1039)
    kwargs_2653 = {}
    # Getting the type of '_to_string_or_unicode_array' (line 1039)
    _to_string_or_unicode_array_2644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1039, 11), '_to_string_or_unicode_array', False)
    # Calling _to_string_or_unicode_array(args, kwargs) (line 1039)
    _to_string_or_unicode_array_call_result_2654 = invoke(stypy.reporting.localization.Localization(__file__, 1039, 11), _to_string_or_unicode_array_2644, *[_vec_string_call_result_2652], **kwargs_2653)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1039)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1039, 4), 'stypy_return_type', _to_string_or_unicode_array_call_result_2654)
    
    # ################# End of 'partition(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'partition' in the type store
    # Getting the type of 'stypy_return_type' (line 1008)
    stypy_return_type_2655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1008, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_2655)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'partition'
    return stypy_return_type_2655

# Assigning a type to the variable 'partition' (line 1008)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1008, 0), 'partition', partition)

@norecursion
def replace(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 1043)
    None_2656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1043, 31), 'None')
    defaults = [None_2656]
    # Create a new context for function 'replace'
    module_type_store = module_type_store.open_function_context('replace', 1043, 0, False)
    
    # Passed parameters checking function
    replace.stypy_localization = localization
    replace.stypy_type_of_self = None
    replace.stypy_type_store = module_type_store
    replace.stypy_function_name = 'replace'
    replace.stypy_param_names_list = ['a', 'old', 'new', 'count']
    replace.stypy_varargs_param_name = None
    replace.stypy_kwargs_param_name = None
    replace.stypy_call_defaults = defaults
    replace.stypy_call_varargs = varargs
    replace.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'replace', ['a', 'old', 'new', 'count'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'replace', localization, ['a', 'old', 'new', 'count'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'replace(...)' code ##################

    str_2657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1069, (-1)), 'str', '\n    For each element in `a`, return a copy of the string with all\n    occurrences of substring `old` replaced by `new`.\n\n    Calls `str.replace` element-wise.\n\n    Parameters\n    ----------\n    a : array-like of str or unicode\n\n    old, new : str or unicode\n\n    count : int, optional\n        If the optional argument `count` is given, only the first\n        `count` occurrences are replaced.\n\n    Returns\n    -------\n    out : ndarray\n        Output array of str or unicode, depending on input type\n\n    See also\n    --------\n    str.replace\n\n    ')
    
    # Call to _to_string_or_unicode_array(...): (line 1070)
    # Processing the call arguments (line 1070)
    
    # Call to _vec_string(...): (line 1071)
    # Processing the call arguments (line 1071)
    # Getting the type of 'a' (line 1072)
    a_2660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1072, 12), 'a', False)
    # Getting the type of 'object_' (line 1072)
    object__2661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1072, 15), 'object_', False)
    str_2662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1072, 24), 'str', 'replace')
    
    # Obtaining an instance of the builtin type 'list' (line 1072)
    list_2663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1072, 35), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1072)
    # Adding element type (line 1072)
    # Getting the type of 'old' (line 1072)
    old_2664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1072, 36), 'old', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1072, 35), list_2663, old_2664)
    # Adding element type (line 1072)
    # Getting the type of 'new' (line 1072)
    new_2665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1072, 41), 'new', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1072, 35), list_2663, new_2665)
    
    
    # Call to _clean_args(...): (line 1072)
    # Processing the call arguments (line 1072)
    # Getting the type of 'count' (line 1072)
    count_2667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1072, 60), 'count', False)
    # Processing the call keyword arguments (line 1072)
    kwargs_2668 = {}
    # Getting the type of '_clean_args' (line 1072)
    _clean_args_2666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1072, 48), '_clean_args', False)
    # Calling _clean_args(args, kwargs) (line 1072)
    _clean_args_call_result_2669 = invoke(stypy.reporting.localization.Localization(__file__, 1072, 48), _clean_args_2666, *[count_2667], **kwargs_2668)
    
    # Applying the binary operator '+' (line 1072)
    result_add_2670 = python_operator(stypy.reporting.localization.Localization(__file__, 1072, 35), '+', list_2663, _clean_args_call_result_2669)
    
    # Processing the call keyword arguments (line 1071)
    kwargs_2671 = {}
    # Getting the type of '_vec_string' (line 1071)
    _vec_string_2659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1071, 8), '_vec_string', False)
    # Calling _vec_string(args, kwargs) (line 1071)
    _vec_string_call_result_2672 = invoke(stypy.reporting.localization.Localization(__file__, 1071, 8), _vec_string_2659, *[a_2660, object__2661, str_2662, result_add_2670], **kwargs_2671)
    
    # Processing the call keyword arguments (line 1070)
    kwargs_2673 = {}
    # Getting the type of '_to_string_or_unicode_array' (line 1070)
    _to_string_or_unicode_array_2658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1070, 11), '_to_string_or_unicode_array', False)
    # Calling _to_string_or_unicode_array(args, kwargs) (line 1070)
    _to_string_or_unicode_array_call_result_2674 = invoke(stypy.reporting.localization.Localization(__file__, 1070, 11), _to_string_or_unicode_array_2658, *[_vec_string_call_result_2672], **kwargs_2673)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1070)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1070, 4), 'stypy_return_type', _to_string_or_unicode_array_call_result_2674)
    
    # ################# End of 'replace(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'replace' in the type store
    # Getting the type of 'stypy_return_type' (line 1043)
    stypy_return_type_2675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1043, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_2675)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'replace'
    return stypy_return_type_2675

# Assigning a type to the variable 'replace' (line 1043)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1043, 0), 'replace', replace)

@norecursion
def rfind(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_2676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1075, 24), 'int')
    # Getting the type of 'None' (line 1075)
    None_2677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1075, 31), 'None')
    defaults = [int_2676, None_2677]
    # Create a new context for function 'rfind'
    module_type_store = module_type_store.open_function_context('rfind', 1075, 0, False)
    
    # Passed parameters checking function
    rfind.stypy_localization = localization
    rfind.stypy_type_of_self = None
    rfind.stypy_type_store = module_type_store
    rfind.stypy_function_name = 'rfind'
    rfind.stypy_param_names_list = ['a', 'sub', 'start', 'end']
    rfind.stypy_varargs_param_name = None
    rfind.stypy_kwargs_param_name = None
    rfind.stypy_call_defaults = defaults
    rfind.stypy_call_varargs = varargs
    rfind.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'rfind', ['a', 'sub', 'start', 'end'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'rfind', localization, ['a', 'sub', 'start', 'end'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'rfind(...)' code ##################

    str_2678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1102, (-1)), 'str', '\n    For each element in `a`, return the highest index in the string\n    where substring `sub` is found, such that `sub` is contained\n    within [`start`, `end`].\n\n    Calls `str.rfind` element-wise.\n\n    Parameters\n    ----------\n    a : array-like of str or unicode\n\n    sub : str or unicode\n\n    start, end : int, optional\n        Optional arguments `start` and `end` are interpreted as in\n        slice notation.\n\n    Returns\n    -------\n    out : ndarray\n       Output array of ints.  Return -1 on failure.\n\n    See also\n    --------\n    str.rfind\n\n    ')
    
    # Call to _vec_string(...): (line 1103)
    # Processing the call arguments (line 1103)
    # Getting the type of 'a' (line 1104)
    a_2680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1104, 8), 'a', False)
    # Getting the type of 'integer' (line 1104)
    integer_2681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1104, 11), 'integer', False)
    str_2682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1104, 20), 'str', 'rfind')
    
    # Obtaining an instance of the builtin type 'list' (line 1104)
    list_2683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1104, 29), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1104)
    # Adding element type (line 1104)
    # Getting the type of 'sub' (line 1104)
    sub_2684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1104, 30), 'sub', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1104, 29), list_2683, sub_2684)
    # Adding element type (line 1104)
    # Getting the type of 'start' (line 1104)
    start_2685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1104, 35), 'start', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1104, 29), list_2683, start_2685)
    
    
    # Call to _clean_args(...): (line 1104)
    # Processing the call arguments (line 1104)
    # Getting the type of 'end' (line 1104)
    end_2687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1104, 56), 'end', False)
    # Processing the call keyword arguments (line 1104)
    kwargs_2688 = {}
    # Getting the type of '_clean_args' (line 1104)
    _clean_args_2686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1104, 44), '_clean_args', False)
    # Calling _clean_args(args, kwargs) (line 1104)
    _clean_args_call_result_2689 = invoke(stypy.reporting.localization.Localization(__file__, 1104, 44), _clean_args_2686, *[end_2687], **kwargs_2688)
    
    # Applying the binary operator '+' (line 1104)
    result_add_2690 = python_operator(stypy.reporting.localization.Localization(__file__, 1104, 29), '+', list_2683, _clean_args_call_result_2689)
    
    # Processing the call keyword arguments (line 1103)
    kwargs_2691 = {}
    # Getting the type of '_vec_string' (line 1103)
    _vec_string_2679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1103, 11), '_vec_string', False)
    # Calling _vec_string(args, kwargs) (line 1103)
    _vec_string_call_result_2692 = invoke(stypy.reporting.localization.Localization(__file__, 1103, 11), _vec_string_2679, *[a_2680, integer_2681, str_2682, result_add_2690], **kwargs_2691)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1103)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1103, 4), 'stypy_return_type', _vec_string_call_result_2692)
    
    # ################# End of 'rfind(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'rfind' in the type store
    # Getting the type of 'stypy_return_type' (line 1075)
    stypy_return_type_2693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1075, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_2693)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'rfind'
    return stypy_return_type_2693

# Assigning a type to the variable 'rfind' (line 1075)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1075, 0), 'rfind', rfind)

@norecursion
def rindex(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_2694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1107, 25), 'int')
    # Getting the type of 'None' (line 1107)
    None_2695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1107, 32), 'None')
    defaults = [int_2694, None_2695]
    # Create a new context for function 'rindex'
    module_type_store = module_type_store.open_function_context('rindex', 1107, 0, False)
    
    # Passed parameters checking function
    rindex.stypy_localization = localization
    rindex.stypy_type_of_self = None
    rindex.stypy_type_store = module_type_store
    rindex.stypy_function_name = 'rindex'
    rindex.stypy_param_names_list = ['a', 'sub', 'start', 'end']
    rindex.stypy_varargs_param_name = None
    rindex.stypy_kwargs_param_name = None
    rindex.stypy_call_defaults = defaults
    rindex.stypy_call_varargs = varargs
    rindex.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'rindex', ['a', 'sub', 'start', 'end'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'rindex', localization, ['a', 'sub', 'start', 'end'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'rindex(...)' code ##################

    str_2696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1131, (-1)), 'str', '\n    Like `rfind`, but raises `ValueError` when the substring `sub` is\n    not found.\n\n    Calls `str.rindex` element-wise.\n\n    Parameters\n    ----------\n    a : array-like of str or unicode\n\n    sub : str or unicode\n\n    start, end : int, optional\n\n    Returns\n    -------\n    out : ndarray\n       Output array of ints.\n\n    See also\n    --------\n    rfind, str.rindex\n\n    ')
    
    # Call to _vec_string(...): (line 1132)
    # Processing the call arguments (line 1132)
    # Getting the type of 'a' (line 1133)
    a_2698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1133, 8), 'a', False)
    # Getting the type of 'integer' (line 1133)
    integer_2699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1133, 11), 'integer', False)
    str_2700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1133, 20), 'str', 'rindex')
    
    # Obtaining an instance of the builtin type 'list' (line 1133)
    list_2701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1133, 30), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1133)
    # Adding element type (line 1133)
    # Getting the type of 'sub' (line 1133)
    sub_2702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1133, 31), 'sub', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1133, 30), list_2701, sub_2702)
    # Adding element type (line 1133)
    # Getting the type of 'start' (line 1133)
    start_2703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1133, 36), 'start', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1133, 30), list_2701, start_2703)
    
    
    # Call to _clean_args(...): (line 1133)
    # Processing the call arguments (line 1133)
    # Getting the type of 'end' (line 1133)
    end_2705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1133, 57), 'end', False)
    # Processing the call keyword arguments (line 1133)
    kwargs_2706 = {}
    # Getting the type of '_clean_args' (line 1133)
    _clean_args_2704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1133, 45), '_clean_args', False)
    # Calling _clean_args(args, kwargs) (line 1133)
    _clean_args_call_result_2707 = invoke(stypy.reporting.localization.Localization(__file__, 1133, 45), _clean_args_2704, *[end_2705], **kwargs_2706)
    
    # Applying the binary operator '+' (line 1133)
    result_add_2708 = python_operator(stypy.reporting.localization.Localization(__file__, 1133, 30), '+', list_2701, _clean_args_call_result_2707)
    
    # Processing the call keyword arguments (line 1132)
    kwargs_2709 = {}
    # Getting the type of '_vec_string' (line 1132)
    _vec_string_2697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1132, 11), '_vec_string', False)
    # Calling _vec_string(args, kwargs) (line 1132)
    _vec_string_call_result_2710 = invoke(stypy.reporting.localization.Localization(__file__, 1132, 11), _vec_string_2697, *[a_2698, integer_2699, str_2700, result_add_2708], **kwargs_2709)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1132)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1132, 4), 'stypy_return_type', _vec_string_call_result_2710)
    
    # ################# End of 'rindex(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'rindex' in the type store
    # Getting the type of 'stypy_return_type' (line 1107)
    stypy_return_type_2711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1107, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_2711)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'rindex'
    return stypy_return_type_2711

# Assigning a type to the variable 'rindex' (line 1107)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1107, 0), 'rindex', rindex)

@norecursion
def rjust(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_2712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1136, 29), 'str', ' ')
    defaults = [str_2712]
    # Create a new context for function 'rjust'
    module_type_store = module_type_store.open_function_context('rjust', 1136, 0, False)
    
    # Passed parameters checking function
    rjust.stypy_localization = localization
    rjust.stypy_type_of_self = None
    rjust.stypy_type_store = module_type_store
    rjust.stypy_function_name = 'rjust'
    rjust.stypy_param_names_list = ['a', 'width', 'fillchar']
    rjust.stypy_varargs_param_name = None
    rjust.stypy_kwargs_param_name = None
    rjust.stypy_call_defaults = defaults
    rjust.stypy_call_varargs = varargs
    rjust.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'rjust', ['a', 'width', 'fillchar'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'rjust', localization, ['a', 'width', 'fillchar'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'rjust(...)' code ##################

    str_2713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1161, (-1)), 'str', '\n    Return an array with the elements of `a` right-justified in a\n    string of length `width`.\n\n    Calls `str.rjust` element-wise.\n\n    Parameters\n    ----------\n    a : array_like of str or unicode\n\n    width : int\n        The length of the resulting strings\n    fillchar : str or unicode, optional\n        The character to use for padding\n\n    Returns\n    -------\n    out : ndarray\n        Output array of str or unicode, depending on input type\n\n    See also\n    --------\n    str.rjust\n\n    ')
    
    # Assigning a Call to a Name (line 1162):
    
    # Call to asarray(...): (line 1162)
    # Processing the call arguments (line 1162)
    # Getting the type of 'a' (line 1162)
    a_2716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1162, 26), 'a', False)
    # Processing the call keyword arguments (line 1162)
    kwargs_2717 = {}
    # Getting the type of 'numpy' (line 1162)
    numpy_2714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1162, 12), 'numpy', False)
    # Obtaining the member 'asarray' of a type (line 1162)
    asarray_2715 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1162, 12), numpy_2714, 'asarray')
    # Calling asarray(args, kwargs) (line 1162)
    asarray_call_result_2718 = invoke(stypy.reporting.localization.Localization(__file__, 1162, 12), asarray_2715, *[a_2716], **kwargs_2717)
    
    # Assigning a type to the variable 'a_arr' (line 1162)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1162, 4), 'a_arr', asarray_call_result_2718)
    
    # Assigning a Call to a Name (line 1163):
    
    # Call to asarray(...): (line 1163)
    # Processing the call arguments (line 1163)
    # Getting the type of 'width' (line 1163)
    width_2721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1163, 30), 'width', False)
    # Processing the call keyword arguments (line 1163)
    kwargs_2722 = {}
    # Getting the type of 'numpy' (line 1163)
    numpy_2719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1163, 16), 'numpy', False)
    # Obtaining the member 'asarray' of a type (line 1163)
    asarray_2720 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1163, 16), numpy_2719, 'asarray')
    # Calling asarray(args, kwargs) (line 1163)
    asarray_call_result_2723 = invoke(stypy.reporting.localization.Localization(__file__, 1163, 16), asarray_2720, *[width_2721], **kwargs_2722)
    
    # Assigning a type to the variable 'width_arr' (line 1163)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1163, 4), 'width_arr', asarray_call_result_2723)
    
    # Assigning a Call to a Name (line 1164):
    
    # Call to long(...): (line 1164)
    # Processing the call arguments (line 1164)
    
    # Call to max(...): (line 1164)
    # Processing the call arguments (line 1164)
    # Getting the type of 'width_arr' (line 1164)
    width_arr_2727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1164, 26), 'width_arr', False)
    # Obtaining the member 'flat' of a type (line 1164)
    flat_2728 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1164, 26), width_arr_2727, 'flat')
    # Processing the call keyword arguments (line 1164)
    kwargs_2729 = {}
    # Getting the type of 'numpy' (line 1164)
    numpy_2725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1164, 16), 'numpy', False)
    # Obtaining the member 'max' of a type (line 1164)
    max_2726 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1164, 16), numpy_2725, 'max')
    # Calling max(args, kwargs) (line 1164)
    max_call_result_2730 = invoke(stypy.reporting.localization.Localization(__file__, 1164, 16), max_2726, *[flat_2728], **kwargs_2729)
    
    # Processing the call keyword arguments (line 1164)
    kwargs_2731 = {}
    # Getting the type of 'long' (line 1164)
    long_2724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1164, 11), 'long', False)
    # Calling long(args, kwargs) (line 1164)
    long_call_result_2732 = invoke(stypy.reporting.localization.Localization(__file__, 1164, 11), long_2724, *[max_call_result_2730], **kwargs_2731)
    
    # Assigning a type to the variable 'size' (line 1164)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1164, 4), 'size', long_call_result_2732)
    
    
    # Call to issubdtype(...): (line 1165)
    # Processing the call arguments (line 1165)
    # Getting the type of 'a_arr' (line 1165)
    a_arr_2735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1165, 24), 'a_arr', False)
    # Obtaining the member 'dtype' of a type (line 1165)
    dtype_2736 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1165, 24), a_arr_2735, 'dtype')
    # Getting the type of 'numpy' (line 1165)
    numpy_2737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1165, 37), 'numpy', False)
    # Obtaining the member 'string_' of a type (line 1165)
    string__2738 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1165, 37), numpy_2737, 'string_')
    # Processing the call keyword arguments (line 1165)
    kwargs_2739 = {}
    # Getting the type of 'numpy' (line 1165)
    numpy_2733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1165, 7), 'numpy', False)
    # Obtaining the member 'issubdtype' of a type (line 1165)
    issubdtype_2734 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1165, 7), numpy_2733, 'issubdtype')
    # Calling issubdtype(args, kwargs) (line 1165)
    issubdtype_call_result_2740 = invoke(stypy.reporting.localization.Localization(__file__, 1165, 7), issubdtype_2734, *[dtype_2736, string__2738], **kwargs_2739)
    
    # Testing the type of an if condition (line 1165)
    if_condition_2741 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1165, 4), issubdtype_call_result_2740)
    # Assigning a type to the variable 'if_condition_2741' (line 1165)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1165, 4), 'if_condition_2741', if_condition_2741)
    # SSA begins for if statement (line 1165)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 1166):
    
    # Call to asbytes(...): (line 1166)
    # Processing the call arguments (line 1166)
    # Getting the type of 'fillchar' (line 1166)
    fillchar_2743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1166, 27), 'fillchar', False)
    # Processing the call keyword arguments (line 1166)
    kwargs_2744 = {}
    # Getting the type of 'asbytes' (line 1166)
    asbytes_2742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1166, 19), 'asbytes', False)
    # Calling asbytes(args, kwargs) (line 1166)
    asbytes_call_result_2745 = invoke(stypy.reporting.localization.Localization(__file__, 1166, 19), asbytes_2742, *[fillchar_2743], **kwargs_2744)
    
    # Assigning a type to the variable 'fillchar' (line 1166)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1166, 8), 'fillchar', asbytes_call_result_2745)
    # SSA join for if statement (line 1165)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to _vec_string(...): (line 1167)
    # Processing the call arguments (line 1167)
    # Getting the type of 'a_arr' (line 1168)
    a_arr_2747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1168, 8), 'a_arr', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1168)
    tuple_2748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1168, 16), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1168)
    # Adding element type (line 1168)
    # Getting the type of 'a_arr' (line 1168)
    a_arr_2749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1168, 16), 'a_arr', False)
    # Obtaining the member 'dtype' of a type (line 1168)
    dtype_2750 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1168, 16), a_arr_2749, 'dtype')
    # Obtaining the member 'type' of a type (line 1168)
    type_2751 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1168, 16), dtype_2750, 'type')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1168, 16), tuple_2748, type_2751)
    # Adding element type (line 1168)
    # Getting the type of 'size' (line 1168)
    size_2752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1168, 34), 'size', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1168, 16), tuple_2748, size_2752)
    
    str_2753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1168, 41), 'str', 'rjust')
    
    # Obtaining an instance of the builtin type 'tuple' (line 1168)
    tuple_2754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1168, 51), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1168)
    # Adding element type (line 1168)
    # Getting the type of 'width_arr' (line 1168)
    width_arr_2755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1168, 51), 'width_arr', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1168, 51), tuple_2754, width_arr_2755)
    # Adding element type (line 1168)
    # Getting the type of 'fillchar' (line 1168)
    fillchar_2756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1168, 62), 'fillchar', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1168, 51), tuple_2754, fillchar_2756)
    
    # Processing the call keyword arguments (line 1167)
    kwargs_2757 = {}
    # Getting the type of '_vec_string' (line 1167)
    _vec_string_2746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1167, 11), '_vec_string', False)
    # Calling _vec_string(args, kwargs) (line 1167)
    _vec_string_call_result_2758 = invoke(stypy.reporting.localization.Localization(__file__, 1167, 11), _vec_string_2746, *[a_arr_2747, tuple_2748, str_2753, tuple_2754], **kwargs_2757)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1167)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1167, 4), 'stypy_return_type', _vec_string_call_result_2758)
    
    # ################# End of 'rjust(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'rjust' in the type store
    # Getting the type of 'stypy_return_type' (line 1136)
    stypy_return_type_2759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1136, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_2759)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'rjust'
    return stypy_return_type_2759

# Assigning a type to the variable 'rjust' (line 1136)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1136, 0), 'rjust', rjust)

@norecursion
def rpartition(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'rpartition'
    module_type_store = module_type_store.open_function_context('rpartition', 1171, 0, False)
    
    # Passed parameters checking function
    rpartition.stypy_localization = localization
    rpartition.stypy_type_of_self = None
    rpartition.stypy_type_store = module_type_store
    rpartition.stypy_function_name = 'rpartition'
    rpartition.stypy_param_names_list = ['a', 'sep']
    rpartition.stypy_varargs_param_name = None
    rpartition.stypy_kwargs_param_name = None
    rpartition.stypy_call_defaults = defaults
    rpartition.stypy_call_varargs = varargs
    rpartition.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'rpartition', ['a', 'sep'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'rpartition', localization, ['a', 'sep'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'rpartition(...)' code ##################

    str_2760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1201, (-1)), 'str', '\n    Partition (split) each element around the right-most separator.\n\n    Calls `str.rpartition` element-wise.\n\n    For each element in `a`, split the element as the last\n    occurrence of `sep`, and return 3 strings containing the part\n    before the separator, the separator itself, and the part after\n    the separator. If the separator is not found, return 3 strings\n    containing the string itself, followed by two empty strings.\n\n    Parameters\n    ----------\n    a : array_like of str or unicode\n        Input array\n    sep : str or unicode\n        Right-most separator to split each element in array.\n\n    Returns\n    -------\n    out : ndarray\n        Output array of string or unicode, depending on input\n        type.  The output array will have an extra dimension with\n        3 elements per input element.\n\n    See also\n    --------\n    str.rpartition\n\n    ')
    
    # Call to _to_string_or_unicode_array(...): (line 1202)
    # Processing the call arguments (line 1202)
    
    # Call to _vec_string(...): (line 1203)
    # Processing the call arguments (line 1203)
    # Getting the type of 'a' (line 1203)
    a_2763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1203, 20), 'a', False)
    # Getting the type of 'object_' (line 1203)
    object__2764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1203, 23), 'object_', False)
    str_2765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1203, 32), 'str', 'rpartition')
    
    # Obtaining an instance of the builtin type 'tuple' (line 1203)
    tuple_2766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1203, 47), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1203)
    # Adding element type (line 1203)
    # Getting the type of 'sep' (line 1203)
    sep_2767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1203, 47), 'sep', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1203, 47), tuple_2766, sep_2767)
    
    # Processing the call keyword arguments (line 1203)
    kwargs_2768 = {}
    # Getting the type of '_vec_string' (line 1203)
    _vec_string_2762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1203, 8), '_vec_string', False)
    # Calling _vec_string(args, kwargs) (line 1203)
    _vec_string_call_result_2769 = invoke(stypy.reporting.localization.Localization(__file__, 1203, 8), _vec_string_2762, *[a_2763, object__2764, str_2765, tuple_2766], **kwargs_2768)
    
    # Processing the call keyword arguments (line 1202)
    kwargs_2770 = {}
    # Getting the type of '_to_string_or_unicode_array' (line 1202)
    _to_string_or_unicode_array_2761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1202, 11), '_to_string_or_unicode_array', False)
    # Calling _to_string_or_unicode_array(args, kwargs) (line 1202)
    _to_string_or_unicode_array_call_result_2771 = invoke(stypy.reporting.localization.Localization(__file__, 1202, 11), _to_string_or_unicode_array_2761, *[_vec_string_call_result_2769], **kwargs_2770)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1202)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1202, 4), 'stypy_return_type', _to_string_or_unicode_array_call_result_2771)
    
    # ################# End of 'rpartition(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'rpartition' in the type store
    # Getting the type of 'stypy_return_type' (line 1171)
    stypy_return_type_2772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1171, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_2772)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'rpartition'
    return stypy_return_type_2772

# Assigning a type to the variable 'rpartition' (line 1171)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1171, 0), 'rpartition', rpartition)

@norecursion
def rsplit(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 1206)
    None_2773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1206, 18), 'None')
    # Getting the type of 'None' (line 1206)
    None_2774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1206, 33), 'None')
    defaults = [None_2773, None_2774]
    # Create a new context for function 'rsplit'
    module_type_store = module_type_store.open_function_context('rsplit', 1206, 0, False)
    
    # Passed parameters checking function
    rsplit.stypy_localization = localization
    rsplit.stypy_type_of_self = None
    rsplit.stypy_type_store = module_type_store
    rsplit.stypy_function_name = 'rsplit'
    rsplit.stypy_param_names_list = ['a', 'sep', 'maxsplit']
    rsplit.stypy_varargs_param_name = None
    rsplit.stypy_kwargs_param_name = None
    rsplit.stypy_call_defaults = defaults
    rsplit.stypy_call_varargs = varargs
    rsplit.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'rsplit', ['a', 'sep', 'maxsplit'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'rsplit', localization, ['a', 'sep', 'maxsplit'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'rsplit(...)' code ##################

    str_2775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1236, (-1)), 'str', '\n    For each element in `a`, return a list of the words in the\n    string, using `sep` as the delimiter string.\n\n    Calls `str.rsplit` element-wise.\n\n    Except for splitting from the right, `rsplit`\n    behaves like `split`.\n\n    Parameters\n    ----------\n    a : array_like of str or unicode\n\n    sep : str or unicode, optional\n        If `sep` is not specified or `None`, any whitespace string\n        is a separator.\n    maxsplit : int, optional\n        If `maxsplit` is given, at most `maxsplit` splits are done,\n        the rightmost ones.\n\n    Returns\n    -------\n    out : ndarray\n       Array of list objects\n\n    See also\n    --------\n    str.rsplit, split\n\n    ')
    
    # Call to _vec_string(...): (line 1239)
    # Processing the call arguments (line 1239)
    # Getting the type of 'a' (line 1240)
    a_2777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1240, 8), 'a', False)
    # Getting the type of 'object_' (line 1240)
    object__2778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1240, 11), 'object_', False)
    str_2779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1240, 20), 'str', 'rsplit')
    
    # Obtaining an instance of the builtin type 'list' (line 1240)
    list_2780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1240, 30), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1240)
    # Adding element type (line 1240)
    # Getting the type of 'sep' (line 1240)
    sep_2781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1240, 31), 'sep', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1240, 30), list_2780, sep_2781)
    
    
    # Call to _clean_args(...): (line 1240)
    # Processing the call arguments (line 1240)
    # Getting the type of 'maxsplit' (line 1240)
    maxsplit_2783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1240, 50), 'maxsplit', False)
    # Processing the call keyword arguments (line 1240)
    kwargs_2784 = {}
    # Getting the type of '_clean_args' (line 1240)
    _clean_args_2782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1240, 38), '_clean_args', False)
    # Calling _clean_args(args, kwargs) (line 1240)
    _clean_args_call_result_2785 = invoke(stypy.reporting.localization.Localization(__file__, 1240, 38), _clean_args_2782, *[maxsplit_2783], **kwargs_2784)
    
    # Applying the binary operator '+' (line 1240)
    result_add_2786 = python_operator(stypy.reporting.localization.Localization(__file__, 1240, 30), '+', list_2780, _clean_args_call_result_2785)
    
    # Processing the call keyword arguments (line 1239)
    kwargs_2787 = {}
    # Getting the type of '_vec_string' (line 1239)
    _vec_string_2776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1239, 11), '_vec_string', False)
    # Calling _vec_string(args, kwargs) (line 1239)
    _vec_string_call_result_2788 = invoke(stypy.reporting.localization.Localization(__file__, 1239, 11), _vec_string_2776, *[a_2777, object__2778, str_2779, result_add_2786], **kwargs_2787)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1239)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1239, 4), 'stypy_return_type', _vec_string_call_result_2788)
    
    # ################# End of 'rsplit(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'rsplit' in the type store
    # Getting the type of 'stypy_return_type' (line 1206)
    stypy_return_type_2789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1206, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_2789)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'rsplit'
    return stypy_return_type_2789

# Assigning a type to the variable 'rsplit' (line 1206)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1206, 0), 'rsplit', rsplit)

@norecursion
def rstrip(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 1243)
    None_2790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1243, 20), 'None')
    defaults = [None_2790]
    # Create a new context for function 'rstrip'
    module_type_store = module_type_store.open_function_context('rstrip', 1243, 0, False)
    
    # Passed parameters checking function
    rstrip.stypy_localization = localization
    rstrip.stypy_type_of_self = None
    rstrip.stypy_type_store = module_type_store
    rstrip.stypy_function_name = 'rstrip'
    rstrip.stypy_param_names_list = ['a', 'chars']
    rstrip.stypy_varargs_param_name = None
    rstrip.stypy_kwargs_param_name = None
    rstrip.stypy_call_defaults = defaults
    rstrip.stypy_call_varargs = varargs
    rstrip.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'rstrip', ['a', 'chars'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'rstrip', localization, ['a', 'chars'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'rstrip(...)' code ##################

    str_2791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1282, (-1)), 'str', "\n    For each element in `a`, return a copy with the trailing\n    characters removed.\n\n    Calls `str.rstrip` element-wise.\n\n    Parameters\n    ----------\n    a : array-like of str or unicode\n\n    chars : str or unicode, optional\n       The `chars` argument is a string specifying the set of\n       characters to be removed. If omitted or None, the `chars`\n       argument defaults to removing whitespace. The `chars` argument\n       is not a suffix; rather, all combinations of its values are\n       stripped.\n\n    Returns\n    -------\n    out : ndarray\n        Output array of str or unicode, depending on input type\n\n    See also\n    --------\n    str.rstrip\n\n    Examples\n    --------\n    >>> c = np.array(['aAaAaA', 'abBABba'], dtype='S7'); c\n    array(['aAaAaA', 'abBABba'],\n        dtype='|S7')\n    >>> np.char.rstrip(c, 'a')\n    array(['aAaAaA', 'abBABb'],\n        dtype='|S7')\n    >>> np.char.rstrip(c, 'A')\n    array(['aAaAa', 'abBABba'],\n        dtype='|S7')\n\n    ")
    
    # Assigning a Call to a Name (line 1283):
    
    # Call to asarray(...): (line 1283)
    # Processing the call arguments (line 1283)
    # Getting the type of 'a' (line 1283)
    a_2794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1283, 26), 'a', False)
    # Processing the call keyword arguments (line 1283)
    kwargs_2795 = {}
    # Getting the type of 'numpy' (line 1283)
    numpy_2792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1283, 12), 'numpy', False)
    # Obtaining the member 'asarray' of a type (line 1283)
    asarray_2793 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1283, 12), numpy_2792, 'asarray')
    # Calling asarray(args, kwargs) (line 1283)
    asarray_call_result_2796 = invoke(stypy.reporting.localization.Localization(__file__, 1283, 12), asarray_2793, *[a_2794], **kwargs_2795)
    
    # Assigning a type to the variable 'a_arr' (line 1283)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1283, 4), 'a_arr', asarray_call_result_2796)
    
    # Call to _vec_string(...): (line 1284)
    # Processing the call arguments (line 1284)
    # Getting the type of 'a_arr' (line 1284)
    a_arr_2798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1284, 23), 'a_arr', False)
    # Getting the type of 'a_arr' (line 1284)
    a_arr_2799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1284, 30), 'a_arr', False)
    # Obtaining the member 'dtype' of a type (line 1284)
    dtype_2800 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1284, 30), a_arr_2799, 'dtype')
    str_2801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1284, 43), 'str', 'rstrip')
    
    # Obtaining an instance of the builtin type 'tuple' (line 1284)
    tuple_2802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1284, 54), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1284)
    # Adding element type (line 1284)
    # Getting the type of 'chars' (line 1284)
    chars_2803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1284, 54), 'chars', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1284, 54), tuple_2802, chars_2803)
    
    # Processing the call keyword arguments (line 1284)
    kwargs_2804 = {}
    # Getting the type of '_vec_string' (line 1284)
    _vec_string_2797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1284, 11), '_vec_string', False)
    # Calling _vec_string(args, kwargs) (line 1284)
    _vec_string_call_result_2805 = invoke(stypy.reporting.localization.Localization(__file__, 1284, 11), _vec_string_2797, *[a_arr_2798, dtype_2800, str_2801, tuple_2802], **kwargs_2804)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1284)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1284, 4), 'stypy_return_type', _vec_string_call_result_2805)
    
    # ################# End of 'rstrip(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'rstrip' in the type store
    # Getting the type of 'stypy_return_type' (line 1243)
    stypy_return_type_2806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1243, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_2806)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'rstrip'
    return stypy_return_type_2806

# Assigning a type to the variable 'rstrip' (line 1243)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1243, 0), 'rstrip', rstrip)

@norecursion
def split(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 1287)
    None_2807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1287, 17), 'None')
    # Getting the type of 'None' (line 1287)
    None_2808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1287, 32), 'None')
    defaults = [None_2807, None_2808]
    # Create a new context for function 'split'
    module_type_store = module_type_store.open_function_context('split', 1287, 0, False)
    
    # Passed parameters checking function
    split.stypy_localization = localization
    split.stypy_type_of_self = None
    split.stypy_type_store = module_type_store
    split.stypy_function_name = 'split'
    split.stypy_param_names_list = ['a', 'sep', 'maxsplit']
    split.stypy_varargs_param_name = None
    split.stypy_kwargs_param_name = None
    split.stypy_call_defaults = defaults
    split.stypy_call_varargs = varargs
    split.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'split', ['a', 'sep', 'maxsplit'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'split', localization, ['a', 'sep', 'maxsplit'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'split(...)' code ##################

    str_2809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1314, (-1)), 'str', '\n    For each element in `a`, return a list of the words in the\n    string, using `sep` as the delimiter string.\n\n    Calls `str.rsplit` element-wise.\n\n    Parameters\n    ----------\n    a : array_like of str or unicode\n\n    sep : str or unicode, optional\n       If `sep` is not specified or `None`, any whitespace string is a\n       separator.\n\n    maxsplit : int, optional\n        If `maxsplit` is given, at most `maxsplit` splits are done.\n\n    Returns\n    -------\n    out : ndarray\n        Array of list objects\n\n    See also\n    --------\n    str.split, rsplit\n\n    ')
    
    # Call to _vec_string(...): (line 1317)
    # Processing the call arguments (line 1317)
    # Getting the type of 'a' (line 1318)
    a_2811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1318, 8), 'a', False)
    # Getting the type of 'object_' (line 1318)
    object__2812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1318, 11), 'object_', False)
    str_2813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1318, 20), 'str', 'split')
    
    # Obtaining an instance of the builtin type 'list' (line 1318)
    list_2814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1318, 29), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1318)
    # Adding element type (line 1318)
    # Getting the type of 'sep' (line 1318)
    sep_2815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1318, 30), 'sep', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1318, 29), list_2814, sep_2815)
    
    
    # Call to _clean_args(...): (line 1318)
    # Processing the call arguments (line 1318)
    # Getting the type of 'maxsplit' (line 1318)
    maxsplit_2817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1318, 49), 'maxsplit', False)
    # Processing the call keyword arguments (line 1318)
    kwargs_2818 = {}
    # Getting the type of '_clean_args' (line 1318)
    _clean_args_2816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1318, 37), '_clean_args', False)
    # Calling _clean_args(args, kwargs) (line 1318)
    _clean_args_call_result_2819 = invoke(stypy.reporting.localization.Localization(__file__, 1318, 37), _clean_args_2816, *[maxsplit_2817], **kwargs_2818)
    
    # Applying the binary operator '+' (line 1318)
    result_add_2820 = python_operator(stypy.reporting.localization.Localization(__file__, 1318, 29), '+', list_2814, _clean_args_call_result_2819)
    
    # Processing the call keyword arguments (line 1317)
    kwargs_2821 = {}
    # Getting the type of '_vec_string' (line 1317)
    _vec_string_2810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1317, 11), '_vec_string', False)
    # Calling _vec_string(args, kwargs) (line 1317)
    _vec_string_call_result_2822 = invoke(stypy.reporting.localization.Localization(__file__, 1317, 11), _vec_string_2810, *[a_2811, object__2812, str_2813, result_add_2820], **kwargs_2821)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1317)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1317, 4), 'stypy_return_type', _vec_string_call_result_2822)
    
    # ################# End of 'split(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'split' in the type store
    # Getting the type of 'stypy_return_type' (line 1287)
    stypy_return_type_2823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1287, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_2823)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'split'
    return stypy_return_type_2823

# Assigning a type to the variable 'split' (line 1287)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1287, 0), 'split', split)

@norecursion
def splitlines(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 1321)
    None_2824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1321, 27), 'None')
    defaults = [None_2824]
    # Create a new context for function 'splitlines'
    module_type_store = module_type_store.open_function_context('splitlines', 1321, 0, False)
    
    # Passed parameters checking function
    splitlines.stypy_localization = localization
    splitlines.stypy_type_of_self = None
    splitlines.stypy_type_store = module_type_store
    splitlines.stypy_function_name = 'splitlines'
    splitlines.stypy_param_names_list = ['a', 'keepends']
    splitlines.stypy_varargs_param_name = None
    splitlines.stypy_kwargs_param_name = None
    splitlines.stypy_call_defaults = defaults
    splitlines.stypy_call_varargs = varargs
    splitlines.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'splitlines', ['a', 'keepends'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'splitlines', localization, ['a', 'keepends'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'splitlines(...)' code ##################

    str_2825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1345, (-1)), 'str', '\n    For each element in `a`, return a list of the lines in the\n    element, breaking at line boundaries.\n\n    Calls `str.splitlines` element-wise.\n\n    Parameters\n    ----------\n    a : array_like of str or unicode\n\n    keepends : bool, optional\n        Line breaks are not included in the resulting list unless\n        keepends is given and true.\n\n    Returns\n    -------\n    out : ndarray\n        Array of list objects\n\n    See also\n    --------\n    str.splitlines\n\n    ')
    
    # Call to _vec_string(...): (line 1346)
    # Processing the call arguments (line 1346)
    # Getting the type of 'a' (line 1347)
    a_2827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1347, 8), 'a', False)
    # Getting the type of 'object_' (line 1347)
    object__2828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1347, 11), 'object_', False)
    str_2829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1347, 20), 'str', 'splitlines')
    
    # Call to _clean_args(...): (line 1347)
    # Processing the call arguments (line 1347)
    # Getting the type of 'keepends' (line 1347)
    keepends_2831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1347, 46), 'keepends', False)
    # Processing the call keyword arguments (line 1347)
    kwargs_2832 = {}
    # Getting the type of '_clean_args' (line 1347)
    _clean_args_2830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1347, 34), '_clean_args', False)
    # Calling _clean_args(args, kwargs) (line 1347)
    _clean_args_call_result_2833 = invoke(stypy.reporting.localization.Localization(__file__, 1347, 34), _clean_args_2830, *[keepends_2831], **kwargs_2832)
    
    # Processing the call keyword arguments (line 1346)
    kwargs_2834 = {}
    # Getting the type of '_vec_string' (line 1346)
    _vec_string_2826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1346, 11), '_vec_string', False)
    # Calling _vec_string(args, kwargs) (line 1346)
    _vec_string_call_result_2835 = invoke(stypy.reporting.localization.Localization(__file__, 1346, 11), _vec_string_2826, *[a_2827, object__2828, str_2829, _clean_args_call_result_2833], **kwargs_2834)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1346)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1346, 4), 'stypy_return_type', _vec_string_call_result_2835)
    
    # ################# End of 'splitlines(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'splitlines' in the type store
    # Getting the type of 'stypy_return_type' (line 1321)
    stypy_return_type_2836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1321, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_2836)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'splitlines'
    return stypy_return_type_2836

# Assigning a type to the variable 'splitlines' (line 1321)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1321, 0), 'splitlines', splitlines)

@norecursion
def startswith(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_2837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1350, 32), 'int')
    # Getting the type of 'None' (line 1350)
    None_2838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1350, 39), 'None')
    defaults = [int_2837, None_2838]
    # Create a new context for function 'startswith'
    module_type_store = module_type_store.open_function_context('startswith', 1350, 0, False)
    
    # Passed parameters checking function
    startswith.stypy_localization = localization
    startswith.stypy_type_of_self = None
    startswith.stypy_type_store = module_type_store
    startswith.stypy_function_name = 'startswith'
    startswith.stypy_param_names_list = ['a', 'prefix', 'start', 'end']
    startswith.stypy_varargs_param_name = None
    startswith.stypy_kwargs_param_name = None
    startswith.stypy_call_defaults = defaults
    startswith.stypy_call_varargs = varargs
    startswith.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'startswith', ['a', 'prefix', 'start', 'end'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'startswith', localization, ['a', 'prefix', 'start', 'end'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'startswith(...)' code ##################

    str_2839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1376, (-1)), 'str', '\n    Returns a boolean array which is `True` where the string element\n    in `a` starts with `prefix`, otherwise `False`.\n\n    Calls `str.startswith` element-wise.\n\n    Parameters\n    ----------\n    a : array_like of str or unicode\n\n    prefix : str\n\n    start, end : int, optional\n        With optional `start`, test beginning at that position. With\n        optional `end`, stop comparing at that position.\n\n    Returns\n    -------\n    out : ndarray\n        Array of booleans\n\n    See also\n    --------\n    str.startswith\n\n    ')
    
    # Call to _vec_string(...): (line 1377)
    # Processing the call arguments (line 1377)
    # Getting the type of 'a' (line 1378)
    a_2841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1378, 8), 'a', False)
    # Getting the type of 'bool_' (line 1378)
    bool__2842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1378, 11), 'bool_', False)
    str_2843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1378, 18), 'str', 'startswith')
    
    # Obtaining an instance of the builtin type 'list' (line 1378)
    list_2844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1378, 32), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1378)
    # Adding element type (line 1378)
    # Getting the type of 'prefix' (line 1378)
    prefix_2845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1378, 33), 'prefix', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1378, 32), list_2844, prefix_2845)
    # Adding element type (line 1378)
    # Getting the type of 'start' (line 1378)
    start_2846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1378, 41), 'start', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1378, 32), list_2844, start_2846)
    
    
    # Call to _clean_args(...): (line 1378)
    # Processing the call arguments (line 1378)
    # Getting the type of 'end' (line 1378)
    end_2848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1378, 62), 'end', False)
    # Processing the call keyword arguments (line 1378)
    kwargs_2849 = {}
    # Getting the type of '_clean_args' (line 1378)
    _clean_args_2847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1378, 50), '_clean_args', False)
    # Calling _clean_args(args, kwargs) (line 1378)
    _clean_args_call_result_2850 = invoke(stypy.reporting.localization.Localization(__file__, 1378, 50), _clean_args_2847, *[end_2848], **kwargs_2849)
    
    # Applying the binary operator '+' (line 1378)
    result_add_2851 = python_operator(stypy.reporting.localization.Localization(__file__, 1378, 32), '+', list_2844, _clean_args_call_result_2850)
    
    # Processing the call keyword arguments (line 1377)
    kwargs_2852 = {}
    # Getting the type of '_vec_string' (line 1377)
    _vec_string_2840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1377, 11), '_vec_string', False)
    # Calling _vec_string(args, kwargs) (line 1377)
    _vec_string_call_result_2853 = invoke(stypy.reporting.localization.Localization(__file__, 1377, 11), _vec_string_2840, *[a_2841, bool__2842, str_2843, result_add_2851], **kwargs_2852)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1377)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1377, 4), 'stypy_return_type', _vec_string_call_result_2853)
    
    # ################# End of 'startswith(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'startswith' in the type store
    # Getting the type of 'stypy_return_type' (line 1350)
    stypy_return_type_2854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1350, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_2854)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'startswith'
    return stypy_return_type_2854

# Assigning a type to the variable 'startswith' (line 1350)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1350, 0), 'startswith', startswith)

@norecursion
def strip(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 1381)
    None_2855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1381, 19), 'None')
    defaults = [None_2855]
    # Create a new context for function 'strip'
    module_type_store = module_type_store.open_function_context('strip', 1381, 0, False)
    
    # Passed parameters checking function
    strip.stypy_localization = localization
    strip.stypy_type_of_self = None
    strip.stypy_type_store = module_type_store
    strip.stypy_function_name = 'strip'
    strip.stypy_param_names_list = ['a', 'chars']
    strip.stypy_varargs_param_name = None
    strip.stypy_kwargs_param_name = None
    strip.stypy_call_defaults = defaults
    strip.stypy_call_varargs = varargs
    strip.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'strip', ['a', 'chars'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'strip', localization, ['a', 'chars'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'strip(...)' code ##################

    str_2856 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1424, (-1)), 'str', "\n    For each element in `a`, return a copy with the leading and\n    trailing characters removed.\n\n    Calls `str.rstrip` element-wise.\n\n    Parameters\n    ----------\n    a : array-like of str or unicode\n\n    chars : str or unicode, optional\n       The `chars` argument is a string specifying the set of\n       characters to be removed. If omitted or None, the `chars`\n       argument defaults to removing whitespace. The `chars` argument\n       is not a prefix or suffix; rather, all combinations of its\n       values are stripped.\n\n    Returns\n    -------\n    out : ndarray\n        Output array of str or unicode, depending on input type\n\n    See also\n    --------\n    str.strip\n\n    Examples\n    --------\n    >>> c = np.array(['aAaAaA', '  aA  ', 'abBABba'])\n    >>> c\n    array(['aAaAaA', '  aA  ', 'abBABba'],\n        dtype='|S7')\n    >>> np.char.strip(c)\n    array(['aAaAaA', 'aA', 'abBABba'],\n        dtype='|S7')\n    >>> np.char.strip(c, 'a') # 'a' unstripped from c[1] because whitespace leads\n    array(['AaAaA', '  aA  ', 'bBABb'],\n        dtype='|S7')\n    >>> np.char.strip(c, 'A') # 'A' unstripped from c[1] because (unprinted) ws trails\n    array(['aAaAa', '  aA  ', 'abBABba'],\n        dtype='|S7')\n\n    ")
    
    # Assigning a Call to a Name (line 1425):
    
    # Call to asarray(...): (line 1425)
    # Processing the call arguments (line 1425)
    # Getting the type of 'a' (line 1425)
    a_2859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1425, 26), 'a', False)
    # Processing the call keyword arguments (line 1425)
    kwargs_2860 = {}
    # Getting the type of 'numpy' (line 1425)
    numpy_2857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1425, 12), 'numpy', False)
    # Obtaining the member 'asarray' of a type (line 1425)
    asarray_2858 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1425, 12), numpy_2857, 'asarray')
    # Calling asarray(args, kwargs) (line 1425)
    asarray_call_result_2861 = invoke(stypy.reporting.localization.Localization(__file__, 1425, 12), asarray_2858, *[a_2859], **kwargs_2860)
    
    # Assigning a type to the variable 'a_arr' (line 1425)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1425, 4), 'a_arr', asarray_call_result_2861)
    
    # Call to _vec_string(...): (line 1426)
    # Processing the call arguments (line 1426)
    # Getting the type of 'a_arr' (line 1426)
    a_arr_2863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1426, 23), 'a_arr', False)
    # Getting the type of 'a_arr' (line 1426)
    a_arr_2864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1426, 30), 'a_arr', False)
    # Obtaining the member 'dtype' of a type (line 1426)
    dtype_2865 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1426, 30), a_arr_2864, 'dtype')
    str_2866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1426, 43), 'str', 'strip')
    
    # Call to _clean_args(...): (line 1426)
    # Processing the call arguments (line 1426)
    # Getting the type of 'chars' (line 1426)
    chars_2868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1426, 64), 'chars', False)
    # Processing the call keyword arguments (line 1426)
    kwargs_2869 = {}
    # Getting the type of '_clean_args' (line 1426)
    _clean_args_2867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1426, 52), '_clean_args', False)
    # Calling _clean_args(args, kwargs) (line 1426)
    _clean_args_call_result_2870 = invoke(stypy.reporting.localization.Localization(__file__, 1426, 52), _clean_args_2867, *[chars_2868], **kwargs_2869)
    
    # Processing the call keyword arguments (line 1426)
    kwargs_2871 = {}
    # Getting the type of '_vec_string' (line 1426)
    _vec_string_2862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1426, 11), '_vec_string', False)
    # Calling _vec_string(args, kwargs) (line 1426)
    _vec_string_call_result_2872 = invoke(stypy.reporting.localization.Localization(__file__, 1426, 11), _vec_string_2862, *[a_arr_2863, dtype_2865, str_2866, _clean_args_call_result_2870], **kwargs_2871)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1426)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1426, 4), 'stypy_return_type', _vec_string_call_result_2872)
    
    # ################# End of 'strip(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'strip' in the type store
    # Getting the type of 'stypy_return_type' (line 1381)
    stypy_return_type_2873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1381, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_2873)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'strip'
    return stypy_return_type_2873

# Assigning a type to the variable 'strip' (line 1381)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1381, 0), 'strip', strip)

@norecursion
def swapcase(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'swapcase'
    module_type_store = module_type_store.open_function_context('swapcase', 1429, 0, False)
    
    # Passed parameters checking function
    swapcase.stypy_localization = localization
    swapcase.stypy_type_of_self = None
    swapcase.stypy_type_store = module_type_store
    swapcase.stypy_function_name = 'swapcase'
    swapcase.stypy_param_names_list = ['a']
    swapcase.stypy_varargs_param_name = None
    swapcase.stypy_kwargs_param_name = None
    swapcase.stypy_call_defaults = defaults
    swapcase.stypy_call_varargs = varargs
    swapcase.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'swapcase', ['a'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'swapcase', localization, ['a'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'swapcase(...)' code ##################

    str_2874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1461, (-1)), 'str', "\n    Return element-wise a copy of the string with\n    uppercase characters converted to lowercase and vice versa.\n\n    Calls `str.swapcase` element-wise.\n\n    For 8-bit strings, this method is locale-dependent.\n\n    Parameters\n    ----------\n    a : array_like, {str, unicode}\n        Input array.\n\n    Returns\n    -------\n    out : ndarray, {str, unicode}\n        Output array of str or unicode, depending on input type\n\n    See also\n    --------\n    str.swapcase\n\n    Examples\n    --------\n    >>> c=np.array(['a1B c','1b Ca','b Ca1','cA1b'],'S5'); c\n    array(['a1B c', '1b Ca', 'b Ca1', 'cA1b'],\n        dtype='|S5')\n    >>> np.char.swapcase(c)\n    array(['A1b C', '1B cA', 'B cA1', 'Ca1B'],\n        dtype='|S5')\n\n    ")
    
    # Assigning a Call to a Name (line 1462):
    
    # Call to asarray(...): (line 1462)
    # Processing the call arguments (line 1462)
    # Getting the type of 'a' (line 1462)
    a_2877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1462, 26), 'a', False)
    # Processing the call keyword arguments (line 1462)
    kwargs_2878 = {}
    # Getting the type of 'numpy' (line 1462)
    numpy_2875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1462, 12), 'numpy', False)
    # Obtaining the member 'asarray' of a type (line 1462)
    asarray_2876 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1462, 12), numpy_2875, 'asarray')
    # Calling asarray(args, kwargs) (line 1462)
    asarray_call_result_2879 = invoke(stypy.reporting.localization.Localization(__file__, 1462, 12), asarray_2876, *[a_2877], **kwargs_2878)
    
    # Assigning a type to the variable 'a_arr' (line 1462)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1462, 4), 'a_arr', asarray_call_result_2879)
    
    # Call to _vec_string(...): (line 1463)
    # Processing the call arguments (line 1463)
    # Getting the type of 'a_arr' (line 1463)
    a_arr_2881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1463, 23), 'a_arr', False)
    # Getting the type of 'a_arr' (line 1463)
    a_arr_2882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1463, 30), 'a_arr', False)
    # Obtaining the member 'dtype' of a type (line 1463)
    dtype_2883 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1463, 30), a_arr_2882, 'dtype')
    str_2884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1463, 43), 'str', 'swapcase')
    # Processing the call keyword arguments (line 1463)
    kwargs_2885 = {}
    # Getting the type of '_vec_string' (line 1463)
    _vec_string_2880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1463, 11), '_vec_string', False)
    # Calling _vec_string(args, kwargs) (line 1463)
    _vec_string_call_result_2886 = invoke(stypy.reporting.localization.Localization(__file__, 1463, 11), _vec_string_2880, *[a_arr_2881, dtype_2883, str_2884], **kwargs_2885)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1463)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1463, 4), 'stypy_return_type', _vec_string_call_result_2886)
    
    # ################# End of 'swapcase(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'swapcase' in the type store
    # Getting the type of 'stypy_return_type' (line 1429)
    stypy_return_type_2887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1429, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_2887)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'swapcase'
    return stypy_return_type_2887

# Assigning a type to the variable 'swapcase' (line 1429)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1429, 0), 'swapcase', swapcase)

@norecursion
def title(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'title'
    module_type_store = module_type_store.open_function_context('title', 1466, 0, False)
    
    # Passed parameters checking function
    title.stypy_localization = localization
    title.stypy_type_of_self = None
    title.stypy_type_store = module_type_store
    title.stypy_function_name = 'title'
    title.stypy_param_names_list = ['a']
    title.stypy_varargs_param_name = None
    title.stypy_kwargs_param_name = None
    title.stypy_call_defaults = defaults
    title.stypy_call_varargs = varargs
    title.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'title', ['a'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'title', localization, ['a'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'title(...)' code ##################

    str_2888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1500, (-1)), 'str', "\n    Return element-wise title cased version of string or unicode.\n\n    Title case words start with uppercase characters, all remaining cased\n    characters are lowercase.\n\n    Calls `str.title` element-wise.\n\n    For 8-bit strings, this method is locale-dependent.\n\n    Parameters\n    ----------\n    a : array_like, {str, unicode}\n        Input array.\n\n    Returns\n    -------\n    out : ndarray\n        Output array of str or unicode, depending on input type\n\n    See also\n    --------\n    str.title\n\n    Examples\n    --------\n    >>> c=np.array(['a1b c','1b ca','b ca1','ca1b'],'S5'); c\n    array(['a1b c', '1b ca', 'b ca1', 'ca1b'],\n        dtype='|S5')\n    >>> np.char.title(c)\n    array(['A1B C', '1B Ca', 'B Ca1', 'Ca1B'],\n        dtype='|S5')\n\n    ")
    
    # Assigning a Call to a Name (line 1501):
    
    # Call to asarray(...): (line 1501)
    # Processing the call arguments (line 1501)
    # Getting the type of 'a' (line 1501)
    a_2891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1501, 26), 'a', False)
    # Processing the call keyword arguments (line 1501)
    kwargs_2892 = {}
    # Getting the type of 'numpy' (line 1501)
    numpy_2889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1501, 12), 'numpy', False)
    # Obtaining the member 'asarray' of a type (line 1501)
    asarray_2890 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1501, 12), numpy_2889, 'asarray')
    # Calling asarray(args, kwargs) (line 1501)
    asarray_call_result_2893 = invoke(stypy.reporting.localization.Localization(__file__, 1501, 12), asarray_2890, *[a_2891], **kwargs_2892)
    
    # Assigning a type to the variable 'a_arr' (line 1501)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1501, 4), 'a_arr', asarray_call_result_2893)
    
    # Call to _vec_string(...): (line 1502)
    # Processing the call arguments (line 1502)
    # Getting the type of 'a_arr' (line 1502)
    a_arr_2895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1502, 23), 'a_arr', False)
    # Getting the type of 'a_arr' (line 1502)
    a_arr_2896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1502, 30), 'a_arr', False)
    # Obtaining the member 'dtype' of a type (line 1502)
    dtype_2897 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1502, 30), a_arr_2896, 'dtype')
    str_2898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1502, 43), 'str', 'title')
    # Processing the call keyword arguments (line 1502)
    kwargs_2899 = {}
    # Getting the type of '_vec_string' (line 1502)
    _vec_string_2894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1502, 11), '_vec_string', False)
    # Calling _vec_string(args, kwargs) (line 1502)
    _vec_string_call_result_2900 = invoke(stypy.reporting.localization.Localization(__file__, 1502, 11), _vec_string_2894, *[a_arr_2895, dtype_2897, str_2898], **kwargs_2899)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1502)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1502, 4), 'stypy_return_type', _vec_string_call_result_2900)
    
    # ################# End of 'title(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'title' in the type store
    # Getting the type of 'stypy_return_type' (line 1466)
    stypy_return_type_2901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1466, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_2901)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'title'
    return stypy_return_type_2901

# Assigning a type to the variable 'title' (line 1466)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1466, 0), 'title', title)

@norecursion
def translate(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 1505)
    None_2902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1505, 36), 'None')
    defaults = [None_2902]
    # Create a new context for function 'translate'
    module_type_store = module_type_store.open_function_context('translate', 1505, 0, False)
    
    # Passed parameters checking function
    translate.stypy_localization = localization
    translate.stypy_type_of_self = None
    translate.stypy_type_store = module_type_store
    translate.stypy_function_name = 'translate'
    translate.stypy_param_names_list = ['a', 'table', 'deletechars']
    translate.stypy_varargs_param_name = None
    translate.stypy_kwargs_param_name = None
    translate.stypy_call_defaults = defaults
    translate.stypy_call_varargs = varargs
    translate.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'translate', ['a', 'table', 'deletechars'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'translate', localization, ['a', 'table', 'deletechars'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'translate(...)' code ##################

    str_2903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1531, (-1)), 'str', '\n    For each element in `a`, return a copy of the string where all\n    characters occurring in the optional argument `deletechars` are\n    removed, and the remaining characters have been mapped through the\n    given translation table.\n\n    Calls `str.translate` element-wise.\n\n    Parameters\n    ----------\n    a : array-like of str or unicode\n\n    table : str of length 256\n\n    deletechars : str\n\n    Returns\n    -------\n    out : ndarray\n        Output array of str or unicode, depending on input type\n\n    See also\n    --------\n    str.translate\n\n    ')
    
    # Assigning a Call to a Name (line 1532):
    
    # Call to asarray(...): (line 1532)
    # Processing the call arguments (line 1532)
    # Getting the type of 'a' (line 1532)
    a_2906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1532, 26), 'a', False)
    # Processing the call keyword arguments (line 1532)
    kwargs_2907 = {}
    # Getting the type of 'numpy' (line 1532)
    numpy_2904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1532, 12), 'numpy', False)
    # Obtaining the member 'asarray' of a type (line 1532)
    asarray_2905 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1532, 12), numpy_2904, 'asarray')
    # Calling asarray(args, kwargs) (line 1532)
    asarray_call_result_2908 = invoke(stypy.reporting.localization.Localization(__file__, 1532, 12), asarray_2905, *[a_2906], **kwargs_2907)
    
    # Assigning a type to the variable 'a_arr' (line 1532)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1532, 4), 'a_arr', asarray_call_result_2908)
    
    
    # Call to issubclass(...): (line 1533)
    # Processing the call arguments (line 1533)
    # Getting the type of 'a_arr' (line 1533)
    a_arr_2910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1533, 18), 'a_arr', False)
    # Obtaining the member 'dtype' of a type (line 1533)
    dtype_2911 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1533, 18), a_arr_2910, 'dtype')
    # Obtaining the member 'type' of a type (line 1533)
    type_2912 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1533, 18), dtype_2911, 'type')
    # Getting the type of 'unicode_' (line 1533)
    unicode__2913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1533, 36), 'unicode_', False)
    # Processing the call keyword arguments (line 1533)
    kwargs_2914 = {}
    # Getting the type of 'issubclass' (line 1533)
    issubclass_2909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1533, 7), 'issubclass', False)
    # Calling issubclass(args, kwargs) (line 1533)
    issubclass_call_result_2915 = invoke(stypy.reporting.localization.Localization(__file__, 1533, 7), issubclass_2909, *[type_2912, unicode__2913], **kwargs_2914)
    
    # Testing the type of an if condition (line 1533)
    if_condition_2916 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1533, 4), issubclass_call_result_2915)
    # Assigning a type to the variable 'if_condition_2916' (line 1533)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1533, 4), 'if_condition_2916', if_condition_2916)
    # SSA begins for if statement (line 1533)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to _vec_string(...): (line 1534)
    # Processing the call arguments (line 1534)
    # Getting the type of 'a_arr' (line 1535)
    a_arr_2918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1535, 12), 'a_arr', False)
    # Getting the type of 'a_arr' (line 1535)
    a_arr_2919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1535, 19), 'a_arr', False)
    # Obtaining the member 'dtype' of a type (line 1535)
    dtype_2920 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1535, 19), a_arr_2919, 'dtype')
    str_2921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1535, 32), 'str', 'translate')
    
    # Obtaining an instance of the builtin type 'tuple' (line 1535)
    tuple_2922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1535, 46), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1535)
    # Adding element type (line 1535)
    # Getting the type of 'table' (line 1535)
    table_2923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1535, 46), 'table', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1535, 46), tuple_2922, table_2923)
    
    # Processing the call keyword arguments (line 1534)
    kwargs_2924 = {}
    # Getting the type of '_vec_string' (line 1534)
    _vec_string_2917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1534, 15), '_vec_string', False)
    # Calling _vec_string(args, kwargs) (line 1534)
    _vec_string_call_result_2925 = invoke(stypy.reporting.localization.Localization(__file__, 1534, 15), _vec_string_2917, *[a_arr_2918, dtype_2920, str_2921, tuple_2922], **kwargs_2924)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1534)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1534, 8), 'stypy_return_type', _vec_string_call_result_2925)
    # SSA branch for the else part of an if statement (line 1533)
    module_type_store.open_ssa_branch('else')
    
    # Call to _vec_string(...): (line 1537)
    # Processing the call arguments (line 1537)
    # Getting the type of 'a_arr' (line 1538)
    a_arr_2927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1538, 12), 'a_arr', False)
    # Getting the type of 'a_arr' (line 1538)
    a_arr_2928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1538, 19), 'a_arr', False)
    # Obtaining the member 'dtype' of a type (line 1538)
    dtype_2929 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1538, 19), a_arr_2928, 'dtype')
    str_2930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1538, 32), 'str', 'translate')
    
    # Obtaining an instance of the builtin type 'list' (line 1538)
    list_2931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1538, 45), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1538)
    # Adding element type (line 1538)
    # Getting the type of 'table' (line 1538)
    table_2932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1538, 46), 'table', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1538, 45), list_2931, table_2932)
    
    
    # Call to _clean_args(...): (line 1538)
    # Processing the call arguments (line 1538)
    # Getting the type of 'deletechars' (line 1538)
    deletechars_2934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1538, 67), 'deletechars', False)
    # Processing the call keyword arguments (line 1538)
    kwargs_2935 = {}
    # Getting the type of '_clean_args' (line 1538)
    _clean_args_2933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1538, 55), '_clean_args', False)
    # Calling _clean_args(args, kwargs) (line 1538)
    _clean_args_call_result_2936 = invoke(stypy.reporting.localization.Localization(__file__, 1538, 55), _clean_args_2933, *[deletechars_2934], **kwargs_2935)
    
    # Applying the binary operator '+' (line 1538)
    result_add_2937 = python_operator(stypy.reporting.localization.Localization(__file__, 1538, 45), '+', list_2931, _clean_args_call_result_2936)
    
    # Processing the call keyword arguments (line 1537)
    kwargs_2938 = {}
    # Getting the type of '_vec_string' (line 1537)
    _vec_string_2926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1537, 15), '_vec_string', False)
    # Calling _vec_string(args, kwargs) (line 1537)
    _vec_string_call_result_2939 = invoke(stypy.reporting.localization.Localization(__file__, 1537, 15), _vec_string_2926, *[a_arr_2927, dtype_2929, str_2930, result_add_2937], **kwargs_2938)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1537)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1537, 8), 'stypy_return_type', _vec_string_call_result_2939)
    # SSA join for if statement (line 1533)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'translate(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'translate' in the type store
    # Getting the type of 'stypy_return_type' (line 1505)
    stypy_return_type_2940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1505, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_2940)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'translate'
    return stypy_return_type_2940

# Assigning a type to the variable 'translate' (line 1505)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1505, 0), 'translate', translate)

@norecursion
def upper(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'upper'
    module_type_store = module_type_store.open_function_context('upper', 1541, 0, False)
    
    # Passed parameters checking function
    upper.stypy_localization = localization
    upper.stypy_type_of_self = None
    upper.stypy_type_store = module_type_store
    upper.stypy_function_name = 'upper'
    upper.stypy_param_names_list = ['a']
    upper.stypy_varargs_param_name = None
    upper.stypy_kwargs_param_name = None
    upper.stypy_call_defaults = defaults
    upper.stypy_call_varargs = varargs
    upper.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'upper', ['a'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'upper', localization, ['a'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'upper(...)' code ##################

    str_2941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1572, (-1)), 'str', "\n    Return an array with the elements converted to uppercase.\n\n    Calls `str.upper` element-wise.\n\n    For 8-bit strings, this method is locale-dependent.\n\n    Parameters\n    ----------\n    a : array_like, {str, unicode}\n        Input array.\n\n    Returns\n    -------\n    out : ndarray, {str, unicode}\n        Output array of str or unicode, depending on input type\n\n    See also\n    --------\n    str.upper\n\n    Examples\n    --------\n    >>> c = np.array(['a1b c', '1bca', 'bca1']); c\n    array(['a1b c', '1bca', 'bca1'],\n        dtype='|S5')\n    >>> np.char.upper(c)\n    array(['A1B C', '1BCA', 'BCA1'],\n        dtype='|S5')\n\n    ")
    
    # Assigning a Call to a Name (line 1573):
    
    # Call to asarray(...): (line 1573)
    # Processing the call arguments (line 1573)
    # Getting the type of 'a' (line 1573)
    a_2944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1573, 26), 'a', False)
    # Processing the call keyword arguments (line 1573)
    kwargs_2945 = {}
    # Getting the type of 'numpy' (line 1573)
    numpy_2942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1573, 12), 'numpy', False)
    # Obtaining the member 'asarray' of a type (line 1573)
    asarray_2943 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1573, 12), numpy_2942, 'asarray')
    # Calling asarray(args, kwargs) (line 1573)
    asarray_call_result_2946 = invoke(stypy.reporting.localization.Localization(__file__, 1573, 12), asarray_2943, *[a_2944], **kwargs_2945)
    
    # Assigning a type to the variable 'a_arr' (line 1573)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1573, 4), 'a_arr', asarray_call_result_2946)
    
    # Call to _vec_string(...): (line 1574)
    # Processing the call arguments (line 1574)
    # Getting the type of 'a_arr' (line 1574)
    a_arr_2948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1574, 23), 'a_arr', False)
    # Getting the type of 'a_arr' (line 1574)
    a_arr_2949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1574, 30), 'a_arr', False)
    # Obtaining the member 'dtype' of a type (line 1574)
    dtype_2950 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1574, 30), a_arr_2949, 'dtype')
    str_2951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1574, 43), 'str', 'upper')
    # Processing the call keyword arguments (line 1574)
    kwargs_2952 = {}
    # Getting the type of '_vec_string' (line 1574)
    _vec_string_2947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1574, 11), '_vec_string', False)
    # Calling _vec_string(args, kwargs) (line 1574)
    _vec_string_call_result_2953 = invoke(stypy.reporting.localization.Localization(__file__, 1574, 11), _vec_string_2947, *[a_arr_2948, dtype_2950, str_2951], **kwargs_2952)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1574)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1574, 4), 'stypy_return_type', _vec_string_call_result_2953)
    
    # ################# End of 'upper(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'upper' in the type store
    # Getting the type of 'stypy_return_type' (line 1541)
    stypy_return_type_2954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1541, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_2954)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'upper'
    return stypy_return_type_2954

# Assigning a type to the variable 'upper' (line 1541)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1541, 0), 'upper', upper)

@norecursion
def zfill(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'zfill'
    module_type_store = module_type_store.open_function_context('zfill', 1577, 0, False)
    
    # Passed parameters checking function
    zfill.stypy_localization = localization
    zfill.stypy_type_of_self = None
    zfill.stypy_type_store = module_type_store
    zfill.stypy_function_name = 'zfill'
    zfill.stypy_param_names_list = ['a', 'width']
    zfill.stypy_varargs_param_name = None
    zfill.stypy_kwargs_param_name = None
    zfill.stypy_call_defaults = defaults
    zfill.stypy_call_varargs = varargs
    zfill.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'zfill', ['a', 'width'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'zfill', localization, ['a', 'width'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'zfill(...)' code ##################

    str_2955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1599, (-1)), 'str', '\n    Return the numeric string left-filled with zeros\n\n    Calls `str.zfill` element-wise.\n\n    Parameters\n    ----------\n    a : array_like, {str, unicode}\n        Input array.\n    width : int\n        Width of string to left-fill elements in `a`.\n\n    Returns\n    -------\n    out : ndarray, {str, unicode}\n        Output array of str or unicode, depending on input type\n\n    See also\n    --------\n    str.zfill\n\n    ')
    
    # Assigning a Call to a Name (line 1600):
    
    # Call to asarray(...): (line 1600)
    # Processing the call arguments (line 1600)
    # Getting the type of 'a' (line 1600)
    a_2958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1600, 26), 'a', False)
    # Processing the call keyword arguments (line 1600)
    kwargs_2959 = {}
    # Getting the type of 'numpy' (line 1600)
    numpy_2956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1600, 12), 'numpy', False)
    # Obtaining the member 'asarray' of a type (line 1600)
    asarray_2957 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1600, 12), numpy_2956, 'asarray')
    # Calling asarray(args, kwargs) (line 1600)
    asarray_call_result_2960 = invoke(stypy.reporting.localization.Localization(__file__, 1600, 12), asarray_2957, *[a_2958], **kwargs_2959)
    
    # Assigning a type to the variable 'a_arr' (line 1600)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1600, 4), 'a_arr', asarray_call_result_2960)
    
    # Assigning a Call to a Name (line 1601):
    
    # Call to asarray(...): (line 1601)
    # Processing the call arguments (line 1601)
    # Getting the type of 'width' (line 1601)
    width_2963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1601, 30), 'width', False)
    # Processing the call keyword arguments (line 1601)
    kwargs_2964 = {}
    # Getting the type of 'numpy' (line 1601)
    numpy_2961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1601, 16), 'numpy', False)
    # Obtaining the member 'asarray' of a type (line 1601)
    asarray_2962 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1601, 16), numpy_2961, 'asarray')
    # Calling asarray(args, kwargs) (line 1601)
    asarray_call_result_2965 = invoke(stypy.reporting.localization.Localization(__file__, 1601, 16), asarray_2962, *[width_2963], **kwargs_2964)
    
    # Assigning a type to the variable 'width_arr' (line 1601)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1601, 4), 'width_arr', asarray_call_result_2965)
    
    # Assigning a Call to a Name (line 1602):
    
    # Call to long(...): (line 1602)
    # Processing the call arguments (line 1602)
    
    # Call to max(...): (line 1602)
    # Processing the call arguments (line 1602)
    # Getting the type of 'width_arr' (line 1602)
    width_arr_2969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1602, 26), 'width_arr', False)
    # Obtaining the member 'flat' of a type (line 1602)
    flat_2970 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1602, 26), width_arr_2969, 'flat')
    # Processing the call keyword arguments (line 1602)
    kwargs_2971 = {}
    # Getting the type of 'numpy' (line 1602)
    numpy_2967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1602, 16), 'numpy', False)
    # Obtaining the member 'max' of a type (line 1602)
    max_2968 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1602, 16), numpy_2967, 'max')
    # Calling max(args, kwargs) (line 1602)
    max_call_result_2972 = invoke(stypy.reporting.localization.Localization(__file__, 1602, 16), max_2968, *[flat_2970], **kwargs_2971)
    
    # Processing the call keyword arguments (line 1602)
    kwargs_2973 = {}
    # Getting the type of 'long' (line 1602)
    long_2966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1602, 11), 'long', False)
    # Calling long(args, kwargs) (line 1602)
    long_call_result_2974 = invoke(stypy.reporting.localization.Localization(__file__, 1602, 11), long_2966, *[max_call_result_2972], **kwargs_2973)
    
    # Assigning a type to the variable 'size' (line 1602)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1602, 4), 'size', long_call_result_2974)
    
    # Call to _vec_string(...): (line 1603)
    # Processing the call arguments (line 1603)
    # Getting the type of 'a_arr' (line 1604)
    a_arr_2976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1604, 8), 'a_arr', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1604)
    tuple_2977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1604, 16), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1604)
    # Adding element type (line 1604)
    # Getting the type of 'a_arr' (line 1604)
    a_arr_2978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1604, 16), 'a_arr', False)
    # Obtaining the member 'dtype' of a type (line 1604)
    dtype_2979 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1604, 16), a_arr_2978, 'dtype')
    # Obtaining the member 'type' of a type (line 1604)
    type_2980 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1604, 16), dtype_2979, 'type')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1604, 16), tuple_2977, type_2980)
    # Adding element type (line 1604)
    # Getting the type of 'size' (line 1604)
    size_2981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1604, 34), 'size', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1604, 16), tuple_2977, size_2981)
    
    str_2982 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1604, 41), 'str', 'zfill')
    
    # Obtaining an instance of the builtin type 'tuple' (line 1604)
    tuple_2983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1604, 51), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1604)
    # Adding element type (line 1604)
    # Getting the type of 'width_arr' (line 1604)
    width_arr_2984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1604, 51), 'width_arr', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1604, 51), tuple_2983, width_arr_2984)
    
    # Processing the call keyword arguments (line 1603)
    kwargs_2985 = {}
    # Getting the type of '_vec_string' (line 1603)
    _vec_string_2975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1603, 11), '_vec_string', False)
    # Calling _vec_string(args, kwargs) (line 1603)
    _vec_string_call_result_2986 = invoke(stypy.reporting.localization.Localization(__file__, 1603, 11), _vec_string_2975, *[a_arr_2976, tuple_2977, str_2982, tuple_2983], **kwargs_2985)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1603)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1603, 4), 'stypy_return_type', _vec_string_call_result_2986)
    
    # ################# End of 'zfill(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'zfill' in the type store
    # Getting the type of 'stypy_return_type' (line 1577)
    stypy_return_type_2987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1577, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_2987)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'zfill'
    return stypy_return_type_2987

# Assigning a type to the variable 'zfill' (line 1577)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1577, 0), 'zfill', zfill)

@norecursion
def isnumeric(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'isnumeric'
    module_type_store = module_type_store.open_function_context('isnumeric', 1607, 0, False)
    
    # Passed parameters checking function
    isnumeric.stypy_localization = localization
    isnumeric.stypy_type_of_self = None
    isnumeric.stypy_type_store = module_type_store
    isnumeric.stypy_function_name = 'isnumeric'
    isnumeric.stypy_param_names_list = ['a']
    isnumeric.stypy_varargs_param_name = None
    isnumeric.stypy_kwargs_param_name = None
    isnumeric.stypy_call_defaults = defaults
    isnumeric.stypy_call_varargs = varargs
    isnumeric.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'isnumeric', ['a'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'isnumeric', localization, ['a'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'isnumeric(...)' code ##################

    str_2988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1632, (-1)), 'str', '\n    For each element, return True if there are only numeric\n    characters in the element.\n\n    Calls `unicode.isnumeric` element-wise.\n\n    Numeric characters include digit characters, and all characters\n    that have the Unicode numeric value property, e.g. ``U+2155,\n    VULGAR FRACTION ONE FIFTH``.\n\n    Parameters\n    ----------\n    a : array_like, unicode\n        Input array.\n\n    Returns\n    -------\n    out : ndarray, bool\n        Array of booleans of same shape as `a`.\n\n    See also\n    --------\n    unicode.isnumeric\n\n    ')
    
    
    
    # Call to _use_unicode(...): (line 1633)
    # Processing the call arguments (line 1633)
    # Getting the type of 'a' (line 1633)
    a_2990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1633, 20), 'a', False)
    # Processing the call keyword arguments (line 1633)
    kwargs_2991 = {}
    # Getting the type of '_use_unicode' (line 1633)
    _use_unicode_2989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1633, 7), '_use_unicode', False)
    # Calling _use_unicode(args, kwargs) (line 1633)
    _use_unicode_call_result_2992 = invoke(stypy.reporting.localization.Localization(__file__, 1633, 7), _use_unicode_2989, *[a_2990], **kwargs_2991)
    
    # Getting the type of 'unicode_' (line 1633)
    unicode__2993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1633, 26), 'unicode_')
    # Applying the binary operator '!=' (line 1633)
    result_ne_2994 = python_operator(stypy.reporting.localization.Localization(__file__, 1633, 7), '!=', _use_unicode_call_result_2992, unicode__2993)
    
    # Testing the type of an if condition (line 1633)
    if_condition_2995 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1633, 4), result_ne_2994)
    # Assigning a type to the variable 'if_condition_2995' (line 1633)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1633, 4), 'if_condition_2995', if_condition_2995)
    # SSA begins for if statement (line 1633)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 1634)
    # Processing the call arguments (line 1634)
    str_2997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1634, 24), 'str', 'isnumeric is only available for Unicode strings and arrays')
    # Processing the call keyword arguments (line 1634)
    kwargs_2998 = {}
    # Getting the type of 'TypeError' (line 1634)
    TypeError_2996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1634, 14), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 1634)
    TypeError_call_result_2999 = invoke(stypy.reporting.localization.Localization(__file__, 1634, 14), TypeError_2996, *[str_2997], **kwargs_2998)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1634, 8), TypeError_call_result_2999, 'raise parameter', BaseException)
    # SSA join for if statement (line 1633)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to _vec_string(...): (line 1635)
    # Processing the call arguments (line 1635)
    # Getting the type of 'a' (line 1635)
    a_3001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1635, 23), 'a', False)
    # Getting the type of 'bool_' (line 1635)
    bool__3002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1635, 26), 'bool_', False)
    str_3003 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1635, 33), 'str', 'isnumeric')
    # Processing the call keyword arguments (line 1635)
    kwargs_3004 = {}
    # Getting the type of '_vec_string' (line 1635)
    _vec_string_3000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1635, 11), '_vec_string', False)
    # Calling _vec_string(args, kwargs) (line 1635)
    _vec_string_call_result_3005 = invoke(stypy.reporting.localization.Localization(__file__, 1635, 11), _vec_string_3000, *[a_3001, bool__3002, str_3003], **kwargs_3004)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1635)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1635, 4), 'stypy_return_type', _vec_string_call_result_3005)
    
    # ################# End of 'isnumeric(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'isnumeric' in the type store
    # Getting the type of 'stypy_return_type' (line 1607)
    stypy_return_type_3006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1607, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_3006)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'isnumeric'
    return stypy_return_type_3006

# Assigning a type to the variable 'isnumeric' (line 1607)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1607, 0), 'isnumeric', isnumeric)

@norecursion
def isdecimal(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'isdecimal'
    module_type_store = module_type_store.open_function_context('isdecimal', 1638, 0, False)
    
    # Passed parameters checking function
    isdecimal.stypy_localization = localization
    isdecimal.stypy_type_of_self = None
    isdecimal.stypy_type_store = module_type_store
    isdecimal.stypy_function_name = 'isdecimal'
    isdecimal.stypy_param_names_list = ['a']
    isdecimal.stypy_varargs_param_name = None
    isdecimal.stypy_kwargs_param_name = None
    isdecimal.stypy_call_defaults = defaults
    isdecimal.stypy_call_varargs = varargs
    isdecimal.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'isdecimal', ['a'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'isdecimal', localization, ['a'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'isdecimal(...)' code ##################

    str_3007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1663, (-1)), 'str', '\n    For each element, return True if there are only decimal\n    characters in the element.\n\n    Calls `unicode.isdecimal` element-wise.\n\n    Decimal characters include digit characters, and all characters\n    that that can be used to form decimal-radix numbers,\n    e.g. ``U+0660, ARABIC-INDIC DIGIT ZERO``.\n\n    Parameters\n    ----------\n    a : array_like, unicode\n        Input array.\n\n    Returns\n    -------\n    out : ndarray, bool\n        Array of booleans identical in shape to `a`.\n\n    See also\n    --------\n    unicode.isdecimal\n\n    ')
    
    
    
    # Call to _use_unicode(...): (line 1664)
    # Processing the call arguments (line 1664)
    # Getting the type of 'a' (line 1664)
    a_3009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1664, 20), 'a', False)
    # Processing the call keyword arguments (line 1664)
    kwargs_3010 = {}
    # Getting the type of '_use_unicode' (line 1664)
    _use_unicode_3008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1664, 7), '_use_unicode', False)
    # Calling _use_unicode(args, kwargs) (line 1664)
    _use_unicode_call_result_3011 = invoke(stypy.reporting.localization.Localization(__file__, 1664, 7), _use_unicode_3008, *[a_3009], **kwargs_3010)
    
    # Getting the type of 'unicode_' (line 1664)
    unicode__3012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1664, 26), 'unicode_')
    # Applying the binary operator '!=' (line 1664)
    result_ne_3013 = python_operator(stypy.reporting.localization.Localization(__file__, 1664, 7), '!=', _use_unicode_call_result_3011, unicode__3012)
    
    # Testing the type of an if condition (line 1664)
    if_condition_3014 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1664, 4), result_ne_3013)
    # Assigning a type to the variable 'if_condition_3014' (line 1664)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1664, 4), 'if_condition_3014', if_condition_3014)
    # SSA begins for if statement (line 1664)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 1665)
    # Processing the call arguments (line 1665)
    str_3016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1665, 24), 'str', 'isnumeric is only available for Unicode strings and arrays')
    # Processing the call keyword arguments (line 1665)
    kwargs_3017 = {}
    # Getting the type of 'TypeError' (line 1665)
    TypeError_3015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1665, 14), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 1665)
    TypeError_call_result_3018 = invoke(stypy.reporting.localization.Localization(__file__, 1665, 14), TypeError_3015, *[str_3016], **kwargs_3017)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1665, 8), TypeError_call_result_3018, 'raise parameter', BaseException)
    # SSA join for if statement (line 1664)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to _vec_string(...): (line 1666)
    # Processing the call arguments (line 1666)
    # Getting the type of 'a' (line 1666)
    a_3020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1666, 23), 'a', False)
    # Getting the type of 'bool_' (line 1666)
    bool__3021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1666, 26), 'bool_', False)
    str_3022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1666, 33), 'str', 'isdecimal')
    # Processing the call keyword arguments (line 1666)
    kwargs_3023 = {}
    # Getting the type of '_vec_string' (line 1666)
    _vec_string_3019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1666, 11), '_vec_string', False)
    # Calling _vec_string(args, kwargs) (line 1666)
    _vec_string_call_result_3024 = invoke(stypy.reporting.localization.Localization(__file__, 1666, 11), _vec_string_3019, *[a_3020, bool__3021, str_3022], **kwargs_3023)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1666)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1666, 4), 'stypy_return_type', _vec_string_call_result_3024)
    
    # ################# End of 'isdecimal(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'isdecimal' in the type store
    # Getting the type of 'stypy_return_type' (line 1638)
    stypy_return_type_3025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1638, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_3025)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'isdecimal'
    return stypy_return_type_3025

# Assigning a type to the variable 'isdecimal' (line 1638)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1638, 0), 'isdecimal', isdecimal)
# Declaration of the 'chararray' class
# Getting the type of 'ndarray' (line 1669)
ndarray_3026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1669, 16), 'ndarray')

class chararray(ndarray_3026, ):
    str_3027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1810, (-1)), 'str', '\n    chararray(shape, itemsize=1, unicode=False, buffer=None, offset=0,\n              strides=None, order=None)\n\n    Provides a convenient view on arrays of string and unicode values.\n\n    .. note::\n       The `chararray` class exists for backwards compatibility with\n       Numarray, it is not recommended for new development. Starting from numpy\n       1.4, if one needs arrays of strings, it is recommended to use arrays of\n       `dtype` `object_`, `string_` or `unicode_`, and use the free functions\n       in the `numpy.char` module for fast vectorized string operations.\n\n    Versus a regular Numpy array of type `str` or `unicode`, this\n    class adds the following functionality:\n\n      1) values automatically have whitespace removed from the end\n         when indexed\n\n      2) comparison operators automatically remove whitespace from the\n         end when comparing values\n\n      3) vectorized string operations are provided as methods\n         (e.g. `.endswith`) and infix operators (e.g. ``"+", "*", "%"``)\n\n    chararrays should be created using `numpy.char.array` or\n    `numpy.char.asarray`, rather than this constructor directly.\n\n    This constructor creates the array, using `buffer` (with `offset`\n    and `strides`) if it is not ``None``. If `buffer` is ``None``, then\n    constructs a new array with `strides` in "C order", unless both\n    ``len(shape) >= 2`` and ``order=\'Fortran\'``, in which case `strides`\n    is in "Fortran order".\n\n    Methods\n    -------\n    astype\n    argsort\n    copy\n    count\n    decode\n    dump\n    dumps\n    encode\n    endswith\n    expandtabs\n    fill\n    find\n    flatten\n    getfield\n    index\n    isalnum\n    isalpha\n    isdecimal\n    isdigit\n    islower\n    isnumeric\n    isspace\n    istitle\n    isupper\n    item\n    join\n    ljust\n    lower\n    lstrip\n    nonzero\n    put\n    ravel\n    repeat\n    replace\n    reshape\n    resize\n    rfind\n    rindex\n    rjust\n    rsplit\n    rstrip\n    searchsorted\n    setfield\n    setflags\n    sort\n    split\n    splitlines\n    squeeze\n    startswith\n    strip\n    swapaxes\n    swapcase\n    take\n    title\n    tofile\n    tolist\n    tostring\n    translate\n    transpose\n    upper\n    view\n    zfill\n\n    Parameters\n    ----------\n    shape : tuple\n        Shape of the array.\n    itemsize : int, optional\n        Length of each array element, in number of characters. Default is 1.\n    unicode : bool, optional\n        Are the array elements of type unicode (True) or string (False).\n        Default is False.\n    buffer : int, optional\n        Memory address of the start of the array data.  Default is None,\n        in which case a new array is created.\n    offset : int, optional\n        Fixed stride displacement from the beginning of an axis?\n        Default is 0. Needs to be >=0.\n    strides : array_like of ints, optional\n        Strides for the array (see `ndarray.strides` for full description).\n        Default is None.\n    order : {\'C\', \'F\'}, optional\n        The order in which the array data is stored in memory: \'C\' ->\n        "row major" order (the default), \'F\' -> "column major"\n        (Fortran) order.\n\n    Examples\n    --------\n    >>> charar = np.chararray((3, 3))\n    >>> charar[:] = \'a\'\n    >>> charar\n    chararray([[\'a\', \'a\', \'a\'],\n           [\'a\', \'a\', \'a\'],\n           [\'a\', \'a\', \'a\']],\n          dtype=\'|S1\')\n\n    >>> charar = np.chararray(charar.shape, itemsize=5)\n    >>> charar[:] = \'abc\'\n    >>> charar\n    chararray([[\'abc\', \'abc\', \'abc\'],\n           [\'abc\', \'abc\', \'abc\'],\n           [\'abc\', \'abc\', \'abc\']],\n          dtype=\'|S5\')\n\n    ')

    @norecursion
    def __new__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_3028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1811, 41), 'int')
        # Getting the type of 'False' (line 1811)
        False_3029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1811, 52), 'False')
        # Getting the type of 'None' (line 1811)
        None_3030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1811, 66), 'None')
        int_3031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1812, 23), 'int')
        # Getting the type of 'None' (line 1812)
        None_3032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1812, 34), 'None')
        str_3033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1812, 46), 'str', 'C')
        defaults = [int_3028, False_3029, None_3030, int_3031, None_3032, str_3033]
        # Create a new context for function '__new__'
        module_type_store = module_type_store.open_function_context('__new__', 1811, 4, False)
        # Assigning a type to the variable 'self' (line 1812)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1812, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        chararray.__new__.__dict__.__setitem__('stypy_localization', localization)
        chararray.__new__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        chararray.__new__.__dict__.__setitem__('stypy_type_store', module_type_store)
        chararray.__new__.__dict__.__setitem__('stypy_function_name', 'chararray.__new__')
        chararray.__new__.__dict__.__setitem__('stypy_param_names_list', ['shape', 'itemsize', 'unicode', 'buffer', 'offset', 'strides', 'order'])
        chararray.__new__.__dict__.__setitem__('stypy_varargs_param_name', None)
        chararray.__new__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        chararray.__new__.__dict__.__setitem__('stypy_call_defaults', defaults)
        chararray.__new__.__dict__.__setitem__('stypy_call_varargs', varargs)
        chararray.__new__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        chararray.__new__.__dict__.__setitem__('stypy_declared_arg_number', 8)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'chararray.__new__', ['shape', 'itemsize', 'unicode', 'buffer', 'offset', 'strides', 'order'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__new__', localization, ['shape', 'itemsize', 'unicode', 'buffer', 'offset', 'strides', 'order'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__new__(...)' code ##################

        # Marking variables as global (line 1813)
        module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 1813, 8), '_globalvar')
        
        # Getting the type of 'unicode' (line 1815)
        unicode_3034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1815, 11), 'unicode')
        # Testing the type of an if condition (line 1815)
        if_condition_3035 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1815, 8), unicode_3034)
        # Assigning a type to the variable 'if_condition_3035' (line 1815)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1815, 8), 'if_condition_3035', if_condition_3035)
        # SSA begins for if statement (line 1815)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 1816):
        # Getting the type of 'unicode_' (line 1816)
        unicode__3036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1816, 20), 'unicode_')
        # Assigning a type to the variable 'dtype' (line 1816)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1816, 12), 'dtype', unicode__3036)
        # SSA branch for the else part of an if statement (line 1815)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Name to a Name (line 1818):
        # Getting the type of 'string_' (line 1818)
        string__3037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1818, 20), 'string_')
        # Assigning a type to the variable 'dtype' (line 1818)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1818, 12), 'dtype', string__3037)
        # SSA join for if statement (line 1815)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 1823):
        
        # Call to long(...): (line 1823)
        # Processing the call arguments (line 1823)
        # Getting the type of 'itemsize' (line 1823)
        itemsize_3039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1823, 24), 'itemsize', False)
        # Processing the call keyword arguments (line 1823)
        kwargs_3040 = {}
        # Getting the type of 'long' (line 1823)
        long_3038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1823, 19), 'long', False)
        # Calling long(args, kwargs) (line 1823)
        long_call_result_3041 = invoke(stypy.reporting.localization.Localization(__file__, 1823, 19), long_3038, *[itemsize_3039], **kwargs_3040)
        
        # Assigning a type to the variable 'itemsize' (line 1823)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1823, 8), 'itemsize', long_call_result_3041)
        
        
        # Evaluating a boolean operation
        
        
        # Obtaining the type of the subscript
        int_3042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1825, 28), 'int')
        # Getting the type of 'sys' (line 1825)
        sys_3043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1825, 11), 'sys')
        # Obtaining the member 'version_info' of a type (line 1825)
        version_info_3044 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1825, 11), sys_3043, 'version_info')
        # Obtaining the member '__getitem__' of a type (line 1825)
        getitem___3045 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1825, 11), version_info_3044, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1825)
        subscript_call_result_3046 = invoke(stypy.reporting.localization.Localization(__file__, 1825, 11), getitem___3045, int_3042)
        
        int_3047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1825, 34), 'int')
        # Applying the binary operator '>=' (line 1825)
        result_ge_3048 = python_operator(stypy.reporting.localization.Localization(__file__, 1825, 11), '>=', subscript_call_result_3046, int_3047)
        
        
        # Call to isinstance(...): (line 1825)
        # Processing the call arguments (line 1825)
        # Getting the type of 'buffer' (line 1825)
        buffer_3050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1825, 51), 'buffer', False)
        # Getting the type of '_unicode' (line 1825)
        _unicode_3051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1825, 59), '_unicode', False)
        # Processing the call keyword arguments (line 1825)
        kwargs_3052 = {}
        # Getting the type of 'isinstance' (line 1825)
        isinstance_3049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1825, 40), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 1825)
        isinstance_call_result_3053 = invoke(stypy.reporting.localization.Localization(__file__, 1825, 40), isinstance_3049, *[buffer_3050, _unicode_3051], **kwargs_3052)
        
        # Applying the binary operator 'and' (line 1825)
        result_and_keyword_3054 = python_operator(stypy.reporting.localization.Localization(__file__, 1825, 11), 'and', result_ge_3048, isinstance_call_result_3053)
        
        # Testing the type of an if condition (line 1825)
        if_condition_3055 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1825, 8), result_and_keyword_3054)
        # Assigning a type to the variable 'if_condition_3055' (line 1825)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1825, 8), 'if_condition_3055', if_condition_3055)
        # SSA begins for if statement (line 1825)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 1827):
        # Getting the type of 'buffer' (line 1827)
        buffer_3056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1827, 21), 'buffer')
        # Assigning a type to the variable 'filler' (line 1827)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1827, 12), 'filler', buffer_3056)
        
        # Assigning a Name to a Name (line 1828):
        # Getting the type of 'None' (line 1828)
        None_3057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1828, 21), 'None')
        # Assigning a type to the variable 'buffer' (line 1828)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1828, 12), 'buffer', None_3057)
        # SSA branch for the else part of an if statement (line 1825)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Name to a Name (line 1830):
        # Getting the type of 'None' (line 1830)
        None_3058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1830, 21), 'None')
        # Assigning a type to the variable 'filler' (line 1830)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1830, 12), 'filler', None_3058)
        # SSA join for if statement (line 1825)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Num to a Name (line 1832):
        int_3059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1832, 21), 'int')
        # Assigning a type to the variable '_globalvar' (line 1832)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1832, 8), '_globalvar', int_3059)
        
        # Type idiom detected: calculating its left and rigth part (line 1833)
        # Getting the type of 'buffer' (line 1833)
        buffer_3060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1833, 11), 'buffer')
        # Getting the type of 'None' (line 1833)
        None_3061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1833, 21), 'None')
        
        (may_be_3062, more_types_in_union_3063) = may_be_none(buffer_3060, None_3061)

        if may_be_3062:

            if more_types_in_union_3063:
                # Runtime conditional SSA (line 1833)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 1834):
            
            # Call to __new__(...): (line 1834)
            # Processing the call arguments (line 1834)
            # Getting the type of 'subtype' (line 1834)
            subtype_3066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1834, 35), 'subtype', False)
            # Getting the type of 'shape' (line 1834)
            shape_3067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1834, 44), 'shape', False)
            
            # Obtaining an instance of the builtin type 'tuple' (line 1834)
            tuple_3068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1834, 52), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 1834)
            # Adding element type (line 1834)
            # Getting the type of 'dtype' (line 1834)
            dtype_3069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1834, 52), 'dtype', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1834, 52), tuple_3068, dtype_3069)
            # Adding element type (line 1834)
            # Getting the type of 'itemsize' (line 1834)
            itemsize_3070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1834, 59), 'itemsize', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1834, 52), tuple_3068, itemsize_3070)
            
            # Processing the call keyword arguments (line 1834)
            # Getting the type of 'order' (line 1835)
            order_3071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1835, 41), 'order', False)
            keyword_3072 = order_3071
            kwargs_3073 = {'order': keyword_3072}
            # Getting the type of 'ndarray' (line 1834)
            ndarray_3064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1834, 19), 'ndarray', False)
            # Obtaining the member '__new__' of a type (line 1834)
            new___3065 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1834, 19), ndarray_3064, '__new__')
            # Calling __new__(args, kwargs) (line 1834)
            new___call_result_3074 = invoke(stypy.reporting.localization.Localization(__file__, 1834, 19), new___3065, *[subtype_3066, shape_3067, tuple_3068], **kwargs_3073)
            
            # Assigning a type to the variable 'self' (line 1834)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1834, 12), 'self', new___call_result_3074)

            if more_types_in_union_3063:
                # Runtime conditional SSA for else branch (line 1833)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_3062) or more_types_in_union_3063):
            
            # Assigning a Call to a Name (line 1837):
            
            # Call to __new__(...): (line 1837)
            # Processing the call arguments (line 1837)
            # Getting the type of 'subtype' (line 1837)
            subtype_3077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1837, 35), 'subtype', False)
            # Getting the type of 'shape' (line 1837)
            shape_3078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1837, 44), 'shape', False)
            
            # Obtaining an instance of the builtin type 'tuple' (line 1837)
            tuple_3079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1837, 52), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 1837)
            # Adding element type (line 1837)
            # Getting the type of 'dtype' (line 1837)
            dtype_3080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1837, 52), 'dtype', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1837, 52), tuple_3079, dtype_3080)
            # Adding element type (line 1837)
            # Getting the type of 'itemsize' (line 1837)
            itemsize_3081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1837, 59), 'itemsize', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1837, 52), tuple_3079, itemsize_3081)
            
            # Processing the call keyword arguments (line 1837)
            # Getting the type of 'buffer' (line 1838)
            buffer_3082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1838, 42), 'buffer', False)
            keyword_3083 = buffer_3082
            # Getting the type of 'offset' (line 1839)
            offset_3084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1839, 42), 'offset', False)
            keyword_3085 = offset_3084
            # Getting the type of 'strides' (line 1839)
            strides_3086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1839, 58), 'strides', False)
            keyword_3087 = strides_3086
            # Getting the type of 'order' (line 1840)
            order_3088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1840, 41), 'order', False)
            keyword_3089 = order_3088
            kwargs_3090 = {'buffer': keyword_3083, 'strides': keyword_3087, 'order': keyword_3089, 'offset': keyword_3085}
            # Getting the type of 'ndarray' (line 1837)
            ndarray_3075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1837, 19), 'ndarray', False)
            # Obtaining the member '__new__' of a type (line 1837)
            new___3076 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1837, 19), ndarray_3075, '__new__')
            # Calling __new__(args, kwargs) (line 1837)
            new___call_result_3091 = invoke(stypy.reporting.localization.Localization(__file__, 1837, 19), new___3076, *[subtype_3077, shape_3078, tuple_3079], **kwargs_3090)
            
            # Assigning a type to the variable 'self' (line 1837)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1837, 12), 'self', new___call_result_3091)

            if (may_be_3062 and more_types_in_union_3063):
                # SSA join for if statement (line 1833)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 1841)
        # Getting the type of 'filler' (line 1841)
        filler_3092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1841, 8), 'filler')
        # Getting the type of 'None' (line 1841)
        None_3093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1841, 25), 'None')
        
        (may_be_3094, more_types_in_union_3095) = may_not_be_none(filler_3092, None_3093)

        if may_be_3094:

            if more_types_in_union_3095:
                # Runtime conditional SSA (line 1841)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Name to a Subscript (line 1842):
            # Getting the type of 'filler' (line 1842)
            filler_3096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1842, 24), 'filler')
            # Getting the type of 'self' (line 1842)
            self_3097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1842, 12), 'self')
            Ellipsis_3098 = Ellipsis
            # Storing an element on a container (line 1842)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1842, 12), self_3097, (Ellipsis_3098, filler_3096))

            if more_types_in_union_3095:
                # SSA join for if statement (line 1841)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Num to a Name (line 1843):
        int_3099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1843, 21), 'int')
        # Assigning a type to the variable '_globalvar' (line 1843)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1843, 8), '_globalvar', int_3099)
        # Getting the type of 'self' (line 1844)
        self_3100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1844, 15), 'self')
        # Assigning a type to the variable 'stypy_return_type' (line 1844)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1844, 8), 'stypy_return_type', self_3100)
        
        # ################# End of '__new__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__new__' in the type store
        # Getting the type of 'stypy_return_type' (line 1811)
        stypy_return_type_3101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1811, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3101)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__new__'
        return stypy_return_type_3101


    @norecursion
    def __array_finalize__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__array_finalize__'
        module_type_store = module_type_store.open_function_context('__array_finalize__', 1846, 4, False)
        # Assigning a type to the variable 'self' (line 1847)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1847, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        chararray.__array_finalize__.__dict__.__setitem__('stypy_localization', localization)
        chararray.__array_finalize__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        chararray.__array_finalize__.__dict__.__setitem__('stypy_type_store', module_type_store)
        chararray.__array_finalize__.__dict__.__setitem__('stypy_function_name', 'chararray.__array_finalize__')
        chararray.__array_finalize__.__dict__.__setitem__('stypy_param_names_list', ['obj'])
        chararray.__array_finalize__.__dict__.__setitem__('stypy_varargs_param_name', None)
        chararray.__array_finalize__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        chararray.__array_finalize__.__dict__.__setitem__('stypy_call_defaults', defaults)
        chararray.__array_finalize__.__dict__.__setitem__('stypy_call_varargs', varargs)
        chararray.__array_finalize__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        chararray.__array_finalize__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'chararray.__array_finalize__', ['obj'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__array_finalize__', localization, ['obj'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__array_finalize__(...)' code ##################

        
        
        # Evaluating a boolean operation
        
        # Getting the type of '_globalvar' (line 1848)
        _globalvar_3102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1848, 15), '_globalvar')
        # Applying the 'not' unary operator (line 1848)
        result_not__3103 = python_operator(stypy.reporting.localization.Localization(__file__, 1848, 11), 'not', _globalvar_3102)
        
        
        # Getting the type of 'self' (line 1848)
        self_3104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1848, 30), 'self')
        # Obtaining the member 'dtype' of a type (line 1848)
        dtype_3105 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1848, 30), self_3104, 'dtype')
        # Obtaining the member 'char' of a type (line 1848)
        char_3106 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1848, 30), dtype_3105, 'char')
        str_3107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1848, 53), 'str', 'SUbc')
        # Applying the binary operator 'notin' (line 1848)
        result_contains_3108 = python_operator(stypy.reporting.localization.Localization(__file__, 1848, 30), 'notin', char_3106, str_3107)
        
        # Applying the binary operator 'and' (line 1848)
        result_and_keyword_3109 = python_operator(stypy.reporting.localization.Localization(__file__, 1848, 11), 'and', result_not__3103, result_contains_3108)
        
        # Testing the type of an if condition (line 1848)
        if_condition_3110 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1848, 8), result_and_keyword_3109)
        # Assigning a type to the variable 'if_condition_3110' (line 1848)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1848, 8), 'if_condition_3110', if_condition_3110)
        # SSA begins for if statement (line 1848)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 1849)
        # Processing the call arguments (line 1849)
        str_3112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1849, 29), 'str', 'Can only create a chararray from string data.')
        # Processing the call keyword arguments (line 1849)
        kwargs_3113 = {}
        # Getting the type of 'ValueError' (line 1849)
        ValueError_3111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1849, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 1849)
        ValueError_call_result_3114 = invoke(stypy.reporting.localization.Localization(__file__, 1849, 18), ValueError_3111, *[str_3112], **kwargs_3113)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1849, 12), ValueError_call_result_3114, 'raise parameter', BaseException)
        # SSA join for if statement (line 1848)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__array_finalize__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__array_finalize__' in the type store
        # Getting the type of 'stypy_return_type' (line 1846)
        stypy_return_type_3115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1846, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3115)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__array_finalize__'
        return stypy_return_type_3115


    @norecursion
    def __getitem__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__getitem__'
        module_type_store = module_type_store.open_function_context('__getitem__', 1851, 4, False)
        # Assigning a type to the variable 'self' (line 1852)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1852, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        chararray.__getitem__.__dict__.__setitem__('stypy_localization', localization)
        chararray.__getitem__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        chararray.__getitem__.__dict__.__setitem__('stypy_type_store', module_type_store)
        chararray.__getitem__.__dict__.__setitem__('stypy_function_name', 'chararray.__getitem__')
        chararray.__getitem__.__dict__.__setitem__('stypy_param_names_list', ['obj'])
        chararray.__getitem__.__dict__.__setitem__('stypy_varargs_param_name', None)
        chararray.__getitem__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        chararray.__getitem__.__dict__.__setitem__('stypy_call_defaults', defaults)
        chararray.__getitem__.__dict__.__setitem__('stypy_call_varargs', varargs)
        chararray.__getitem__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        chararray.__getitem__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'chararray.__getitem__', ['obj'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__getitem__', localization, ['obj'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__getitem__(...)' code ##################

        
        # Assigning a Call to a Name (line 1852):
        
        # Call to __getitem__(...): (line 1852)
        # Processing the call arguments (line 1852)
        # Getting the type of 'self' (line 1852)
        self_3118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1852, 34), 'self', False)
        # Getting the type of 'obj' (line 1852)
        obj_3119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1852, 40), 'obj', False)
        # Processing the call keyword arguments (line 1852)
        kwargs_3120 = {}
        # Getting the type of 'ndarray' (line 1852)
        ndarray_3116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1852, 14), 'ndarray', False)
        # Obtaining the member '__getitem__' of a type (line 1852)
        getitem___3117 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1852, 14), ndarray_3116, '__getitem__')
        # Calling __getitem__(args, kwargs) (line 1852)
        getitem___call_result_3121 = invoke(stypy.reporting.localization.Localization(__file__, 1852, 14), getitem___3117, *[self_3118, obj_3119], **kwargs_3120)
        
        # Assigning a type to the variable 'val' (line 1852)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1852, 8), 'val', getitem___call_result_3121)
        
        
        # Call to isinstance(...): (line 1854)
        # Processing the call arguments (line 1854)
        # Getting the type of 'val' (line 1854)
        val_3123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1854, 22), 'val', False)
        # Getting the type of 'character' (line 1854)
        character_3124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1854, 27), 'character', False)
        # Processing the call keyword arguments (line 1854)
        kwargs_3125 = {}
        # Getting the type of 'isinstance' (line 1854)
        isinstance_3122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1854, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 1854)
        isinstance_call_result_3126 = invoke(stypy.reporting.localization.Localization(__file__, 1854, 11), isinstance_3122, *[val_3123, character_3124], **kwargs_3125)
        
        # Testing the type of an if condition (line 1854)
        if_condition_3127 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1854, 8), isinstance_call_result_3126)
        # Assigning a type to the variable 'if_condition_3127' (line 1854)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1854, 8), 'if_condition_3127', if_condition_3127)
        # SSA begins for if statement (line 1854)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 1855):
        
        # Call to rstrip(...): (line 1855)
        # Processing the call keyword arguments (line 1855)
        kwargs_3130 = {}
        # Getting the type of 'val' (line 1855)
        val_3128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1855, 19), 'val', False)
        # Obtaining the member 'rstrip' of a type (line 1855)
        rstrip_3129 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1855, 19), val_3128, 'rstrip')
        # Calling rstrip(args, kwargs) (line 1855)
        rstrip_call_result_3131 = invoke(stypy.reporting.localization.Localization(__file__, 1855, 19), rstrip_3129, *[], **kwargs_3130)
        
        # Assigning a type to the variable 'temp' (line 1855)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1855, 12), 'temp', rstrip_call_result_3131)
        
        
        
        # Call to _len(...): (line 1856)
        # Processing the call arguments (line 1856)
        # Getting the type of 'temp' (line 1856)
        temp_3133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1856, 20), 'temp', False)
        # Processing the call keyword arguments (line 1856)
        kwargs_3134 = {}
        # Getting the type of '_len' (line 1856)
        _len_3132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1856, 15), '_len', False)
        # Calling _len(args, kwargs) (line 1856)
        _len_call_result_3135 = invoke(stypy.reporting.localization.Localization(__file__, 1856, 15), _len_3132, *[temp_3133], **kwargs_3134)
        
        int_3136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1856, 29), 'int')
        # Applying the binary operator '==' (line 1856)
        result_eq_3137 = python_operator(stypy.reporting.localization.Localization(__file__, 1856, 15), '==', _len_call_result_3135, int_3136)
        
        # Testing the type of an if condition (line 1856)
        if_condition_3138 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1856, 12), result_eq_3137)
        # Assigning a type to the variable 'if_condition_3138' (line 1856)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1856, 12), 'if_condition_3138', if_condition_3138)
        # SSA begins for if statement (line 1856)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Name (line 1857):
        str_3139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1857, 22), 'str', '')
        # Assigning a type to the variable 'val' (line 1857)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1857, 16), 'val', str_3139)
        # SSA branch for the else part of an if statement (line 1856)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Name to a Name (line 1859):
        # Getting the type of 'temp' (line 1859)
        temp_3140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1859, 22), 'temp')
        # Assigning a type to the variable 'val' (line 1859)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1859, 16), 'val', temp_3140)
        # SSA join for if statement (line 1856)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 1854)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'val' (line 1861)
        val_3141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1861, 15), 'val')
        # Assigning a type to the variable 'stypy_return_type' (line 1861)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1861, 8), 'stypy_return_type', val_3141)
        
        # ################# End of '__getitem__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__getitem__' in the type store
        # Getting the type of 'stypy_return_type' (line 1851)
        stypy_return_type_3142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1851, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3142)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__getitem__'
        return stypy_return_type_3142


    @norecursion
    def stypy__eq__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__eq__'
        module_type_store = module_type_store.open_function_context('__eq__', 1868, 4, False)
        # Assigning a type to the variable 'self' (line 1869)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1869, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        chararray.stypy__eq__.__dict__.__setitem__('stypy_localization', localization)
        chararray.stypy__eq__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        chararray.stypy__eq__.__dict__.__setitem__('stypy_type_store', module_type_store)
        chararray.stypy__eq__.__dict__.__setitem__('stypy_function_name', 'chararray.__eq__')
        chararray.stypy__eq__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        chararray.stypy__eq__.__dict__.__setitem__('stypy_varargs_param_name', None)
        chararray.stypy__eq__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        chararray.stypy__eq__.__dict__.__setitem__('stypy_call_defaults', defaults)
        chararray.stypy__eq__.__dict__.__setitem__('stypy_call_varargs', varargs)
        chararray.stypy__eq__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        chararray.stypy__eq__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'chararray.__eq__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__eq__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__eq__(...)' code ##################

        str_3143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1875, (-1)), 'str', '\n        Return (self == other) element-wise.\n\n        See also\n        --------\n        equal\n        ')
        
        # Call to equal(...): (line 1876)
        # Processing the call arguments (line 1876)
        # Getting the type of 'self' (line 1876)
        self_3145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1876, 21), 'self', False)
        # Getting the type of 'other' (line 1876)
        other_3146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1876, 27), 'other', False)
        # Processing the call keyword arguments (line 1876)
        kwargs_3147 = {}
        # Getting the type of 'equal' (line 1876)
        equal_3144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1876, 15), 'equal', False)
        # Calling equal(args, kwargs) (line 1876)
        equal_call_result_3148 = invoke(stypy.reporting.localization.Localization(__file__, 1876, 15), equal_3144, *[self_3145, other_3146], **kwargs_3147)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1876)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1876, 8), 'stypy_return_type', equal_call_result_3148)
        
        # ################# End of '__eq__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__eq__' in the type store
        # Getting the type of 'stypy_return_type' (line 1868)
        stypy_return_type_3149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1868, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3149)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__eq__'
        return stypy_return_type_3149


    @norecursion
    def __ne__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__ne__'
        module_type_store = module_type_store.open_function_context('__ne__', 1878, 4, False)
        # Assigning a type to the variable 'self' (line 1879)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1879, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        chararray.__ne__.__dict__.__setitem__('stypy_localization', localization)
        chararray.__ne__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        chararray.__ne__.__dict__.__setitem__('stypy_type_store', module_type_store)
        chararray.__ne__.__dict__.__setitem__('stypy_function_name', 'chararray.__ne__')
        chararray.__ne__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        chararray.__ne__.__dict__.__setitem__('stypy_varargs_param_name', None)
        chararray.__ne__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        chararray.__ne__.__dict__.__setitem__('stypy_call_defaults', defaults)
        chararray.__ne__.__dict__.__setitem__('stypy_call_varargs', varargs)
        chararray.__ne__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        chararray.__ne__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'chararray.__ne__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__ne__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__ne__(...)' code ##################

        str_3150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1885, (-1)), 'str', '\n        Return (self != other) element-wise.\n\n        See also\n        --------\n        not_equal\n        ')
        
        # Call to not_equal(...): (line 1886)
        # Processing the call arguments (line 1886)
        # Getting the type of 'self' (line 1886)
        self_3152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1886, 25), 'self', False)
        # Getting the type of 'other' (line 1886)
        other_3153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1886, 31), 'other', False)
        # Processing the call keyword arguments (line 1886)
        kwargs_3154 = {}
        # Getting the type of 'not_equal' (line 1886)
        not_equal_3151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1886, 15), 'not_equal', False)
        # Calling not_equal(args, kwargs) (line 1886)
        not_equal_call_result_3155 = invoke(stypy.reporting.localization.Localization(__file__, 1886, 15), not_equal_3151, *[self_3152, other_3153], **kwargs_3154)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1886)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1886, 8), 'stypy_return_type', not_equal_call_result_3155)
        
        # ################# End of '__ne__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__ne__' in the type store
        # Getting the type of 'stypy_return_type' (line 1878)
        stypy_return_type_3156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1878, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3156)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__ne__'
        return stypy_return_type_3156


    @norecursion
    def __ge__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__ge__'
        module_type_store = module_type_store.open_function_context('__ge__', 1888, 4, False)
        # Assigning a type to the variable 'self' (line 1889)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1889, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        chararray.__ge__.__dict__.__setitem__('stypy_localization', localization)
        chararray.__ge__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        chararray.__ge__.__dict__.__setitem__('stypy_type_store', module_type_store)
        chararray.__ge__.__dict__.__setitem__('stypy_function_name', 'chararray.__ge__')
        chararray.__ge__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        chararray.__ge__.__dict__.__setitem__('stypy_varargs_param_name', None)
        chararray.__ge__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        chararray.__ge__.__dict__.__setitem__('stypy_call_defaults', defaults)
        chararray.__ge__.__dict__.__setitem__('stypy_call_varargs', varargs)
        chararray.__ge__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        chararray.__ge__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'chararray.__ge__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__ge__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__ge__(...)' code ##################

        str_3157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1895, (-1)), 'str', '\n        Return (self >= other) element-wise.\n\n        See also\n        --------\n        greater_equal\n        ')
        
        # Call to greater_equal(...): (line 1896)
        # Processing the call arguments (line 1896)
        # Getting the type of 'self' (line 1896)
        self_3159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1896, 29), 'self', False)
        # Getting the type of 'other' (line 1896)
        other_3160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1896, 35), 'other', False)
        # Processing the call keyword arguments (line 1896)
        kwargs_3161 = {}
        # Getting the type of 'greater_equal' (line 1896)
        greater_equal_3158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1896, 15), 'greater_equal', False)
        # Calling greater_equal(args, kwargs) (line 1896)
        greater_equal_call_result_3162 = invoke(stypy.reporting.localization.Localization(__file__, 1896, 15), greater_equal_3158, *[self_3159, other_3160], **kwargs_3161)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1896)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1896, 8), 'stypy_return_type', greater_equal_call_result_3162)
        
        # ################# End of '__ge__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__ge__' in the type store
        # Getting the type of 'stypy_return_type' (line 1888)
        stypy_return_type_3163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1888, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3163)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__ge__'
        return stypy_return_type_3163


    @norecursion
    def __le__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__le__'
        module_type_store = module_type_store.open_function_context('__le__', 1898, 4, False)
        # Assigning a type to the variable 'self' (line 1899)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1899, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        chararray.__le__.__dict__.__setitem__('stypy_localization', localization)
        chararray.__le__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        chararray.__le__.__dict__.__setitem__('stypy_type_store', module_type_store)
        chararray.__le__.__dict__.__setitem__('stypy_function_name', 'chararray.__le__')
        chararray.__le__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        chararray.__le__.__dict__.__setitem__('stypy_varargs_param_name', None)
        chararray.__le__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        chararray.__le__.__dict__.__setitem__('stypy_call_defaults', defaults)
        chararray.__le__.__dict__.__setitem__('stypy_call_varargs', varargs)
        chararray.__le__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        chararray.__le__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'chararray.__le__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__le__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__le__(...)' code ##################

        str_3164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1905, (-1)), 'str', '\n        Return (self <= other) element-wise.\n\n        See also\n        --------\n        less_equal\n        ')
        
        # Call to less_equal(...): (line 1906)
        # Processing the call arguments (line 1906)
        # Getting the type of 'self' (line 1906)
        self_3166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1906, 26), 'self', False)
        # Getting the type of 'other' (line 1906)
        other_3167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1906, 32), 'other', False)
        # Processing the call keyword arguments (line 1906)
        kwargs_3168 = {}
        # Getting the type of 'less_equal' (line 1906)
        less_equal_3165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1906, 15), 'less_equal', False)
        # Calling less_equal(args, kwargs) (line 1906)
        less_equal_call_result_3169 = invoke(stypy.reporting.localization.Localization(__file__, 1906, 15), less_equal_3165, *[self_3166, other_3167], **kwargs_3168)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1906)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1906, 8), 'stypy_return_type', less_equal_call_result_3169)
        
        # ################# End of '__le__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__le__' in the type store
        # Getting the type of 'stypy_return_type' (line 1898)
        stypy_return_type_3170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1898, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3170)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__le__'
        return stypy_return_type_3170


    @norecursion
    def __gt__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__gt__'
        module_type_store = module_type_store.open_function_context('__gt__', 1908, 4, False)
        # Assigning a type to the variable 'self' (line 1909)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1909, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        chararray.__gt__.__dict__.__setitem__('stypy_localization', localization)
        chararray.__gt__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        chararray.__gt__.__dict__.__setitem__('stypy_type_store', module_type_store)
        chararray.__gt__.__dict__.__setitem__('stypy_function_name', 'chararray.__gt__')
        chararray.__gt__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        chararray.__gt__.__dict__.__setitem__('stypy_varargs_param_name', None)
        chararray.__gt__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        chararray.__gt__.__dict__.__setitem__('stypy_call_defaults', defaults)
        chararray.__gt__.__dict__.__setitem__('stypy_call_varargs', varargs)
        chararray.__gt__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        chararray.__gt__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'chararray.__gt__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__gt__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__gt__(...)' code ##################

        str_3171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1915, (-1)), 'str', '\n        Return (self > other) element-wise.\n\n        See also\n        --------\n        greater\n        ')
        
        # Call to greater(...): (line 1916)
        # Processing the call arguments (line 1916)
        # Getting the type of 'self' (line 1916)
        self_3173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1916, 23), 'self', False)
        # Getting the type of 'other' (line 1916)
        other_3174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1916, 29), 'other', False)
        # Processing the call keyword arguments (line 1916)
        kwargs_3175 = {}
        # Getting the type of 'greater' (line 1916)
        greater_3172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1916, 15), 'greater', False)
        # Calling greater(args, kwargs) (line 1916)
        greater_call_result_3176 = invoke(stypy.reporting.localization.Localization(__file__, 1916, 15), greater_3172, *[self_3173, other_3174], **kwargs_3175)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1916)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1916, 8), 'stypy_return_type', greater_call_result_3176)
        
        # ################# End of '__gt__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__gt__' in the type store
        # Getting the type of 'stypy_return_type' (line 1908)
        stypy_return_type_3177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1908, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3177)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__gt__'
        return stypy_return_type_3177


    @norecursion
    def __lt__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__lt__'
        module_type_store = module_type_store.open_function_context('__lt__', 1918, 4, False)
        # Assigning a type to the variable 'self' (line 1919)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1919, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        chararray.__lt__.__dict__.__setitem__('stypy_localization', localization)
        chararray.__lt__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        chararray.__lt__.__dict__.__setitem__('stypy_type_store', module_type_store)
        chararray.__lt__.__dict__.__setitem__('stypy_function_name', 'chararray.__lt__')
        chararray.__lt__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        chararray.__lt__.__dict__.__setitem__('stypy_varargs_param_name', None)
        chararray.__lt__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        chararray.__lt__.__dict__.__setitem__('stypy_call_defaults', defaults)
        chararray.__lt__.__dict__.__setitem__('stypy_call_varargs', varargs)
        chararray.__lt__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        chararray.__lt__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'chararray.__lt__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__lt__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__lt__(...)' code ##################

        str_3178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1925, (-1)), 'str', '\n        Return (self < other) element-wise.\n\n        See also\n        --------\n        less\n        ')
        
        # Call to less(...): (line 1926)
        # Processing the call arguments (line 1926)
        # Getting the type of 'self' (line 1926)
        self_3180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1926, 20), 'self', False)
        # Getting the type of 'other' (line 1926)
        other_3181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1926, 26), 'other', False)
        # Processing the call keyword arguments (line 1926)
        kwargs_3182 = {}
        # Getting the type of 'less' (line 1926)
        less_3179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1926, 15), 'less', False)
        # Calling less(args, kwargs) (line 1926)
        less_call_result_3183 = invoke(stypy.reporting.localization.Localization(__file__, 1926, 15), less_3179, *[self_3180, other_3181], **kwargs_3182)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1926)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1926, 8), 'stypy_return_type', less_call_result_3183)
        
        # ################# End of '__lt__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__lt__' in the type store
        # Getting the type of 'stypy_return_type' (line 1918)
        stypy_return_type_3184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1918, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3184)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__lt__'
        return stypy_return_type_3184


    @norecursion
    def __add__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__add__'
        module_type_store = module_type_store.open_function_context('__add__', 1928, 4, False)
        # Assigning a type to the variable 'self' (line 1929)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1929, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        chararray.__add__.__dict__.__setitem__('stypy_localization', localization)
        chararray.__add__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        chararray.__add__.__dict__.__setitem__('stypy_type_store', module_type_store)
        chararray.__add__.__dict__.__setitem__('stypy_function_name', 'chararray.__add__')
        chararray.__add__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        chararray.__add__.__dict__.__setitem__('stypy_varargs_param_name', None)
        chararray.__add__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        chararray.__add__.__dict__.__setitem__('stypy_call_defaults', defaults)
        chararray.__add__.__dict__.__setitem__('stypy_call_varargs', varargs)
        chararray.__add__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        chararray.__add__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'chararray.__add__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__add__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__add__(...)' code ##################

        str_3185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1936, (-1)), 'str', '\n        Return (self + other), that is string concatenation,\n        element-wise for a pair of array_likes of str or unicode.\n\n        See also\n        --------\n        add\n        ')
        
        # Call to asarray(...): (line 1937)
        # Processing the call arguments (line 1937)
        
        # Call to add(...): (line 1937)
        # Processing the call arguments (line 1937)
        # Getting the type of 'self' (line 1937)
        self_3188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1937, 27), 'self', False)
        # Getting the type of 'other' (line 1937)
        other_3189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1937, 33), 'other', False)
        # Processing the call keyword arguments (line 1937)
        kwargs_3190 = {}
        # Getting the type of 'add' (line 1937)
        add_3187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1937, 23), 'add', False)
        # Calling add(args, kwargs) (line 1937)
        add_call_result_3191 = invoke(stypy.reporting.localization.Localization(__file__, 1937, 23), add_3187, *[self_3188, other_3189], **kwargs_3190)
        
        # Processing the call keyword arguments (line 1937)
        kwargs_3192 = {}
        # Getting the type of 'asarray' (line 1937)
        asarray_3186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1937, 15), 'asarray', False)
        # Calling asarray(args, kwargs) (line 1937)
        asarray_call_result_3193 = invoke(stypy.reporting.localization.Localization(__file__, 1937, 15), asarray_3186, *[add_call_result_3191], **kwargs_3192)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1937)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1937, 8), 'stypy_return_type', asarray_call_result_3193)
        
        # ################# End of '__add__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__add__' in the type store
        # Getting the type of 'stypy_return_type' (line 1928)
        stypy_return_type_3194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1928, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3194)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__add__'
        return stypy_return_type_3194


    @norecursion
    def __radd__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__radd__'
        module_type_store = module_type_store.open_function_context('__radd__', 1939, 4, False)
        # Assigning a type to the variable 'self' (line 1940)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1940, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        chararray.__radd__.__dict__.__setitem__('stypy_localization', localization)
        chararray.__radd__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        chararray.__radd__.__dict__.__setitem__('stypy_type_store', module_type_store)
        chararray.__radd__.__dict__.__setitem__('stypy_function_name', 'chararray.__radd__')
        chararray.__radd__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        chararray.__radd__.__dict__.__setitem__('stypy_varargs_param_name', None)
        chararray.__radd__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        chararray.__radd__.__dict__.__setitem__('stypy_call_defaults', defaults)
        chararray.__radd__.__dict__.__setitem__('stypy_call_varargs', varargs)
        chararray.__radd__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        chararray.__radd__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'chararray.__radd__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__radd__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__radd__(...)' code ##################

        str_3195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1947, (-1)), 'str', '\n        Return (other + self), that is string concatenation,\n        element-wise for a pair of array_likes of `string_` or `unicode_`.\n\n        See also\n        --------\n        add\n        ')
        
        # Call to asarray(...): (line 1948)
        # Processing the call arguments (line 1948)
        
        # Call to add(...): (line 1948)
        # Processing the call arguments (line 1948)
        
        # Call to asarray(...): (line 1948)
        # Processing the call arguments (line 1948)
        # Getting the type of 'other' (line 1948)
        other_3200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1948, 41), 'other', False)
        # Processing the call keyword arguments (line 1948)
        kwargs_3201 = {}
        # Getting the type of 'numpy' (line 1948)
        numpy_3198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1948, 27), 'numpy', False)
        # Obtaining the member 'asarray' of a type (line 1948)
        asarray_3199 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1948, 27), numpy_3198, 'asarray')
        # Calling asarray(args, kwargs) (line 1948)
        asarray_call_result_3202 = invoke(stypy.reporting.localization.Localization(__file__, 1948, 27), asarray_3199, *[other_3200], **kwargs_3201)
        
        # Getting the type of 'self' (line 1948)
        self_3203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1948, 49), 'self', False)
        # Processing the call keyword arguments (line 1948)
        kwargs_3204 = {}
        # Getting the type of 'add' (line 1948)
        add_3197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1948, 23), 'add', False)
        # Calling add(args, kwargs) (line 1948)
        add_call_result_3205 = invoke(stypy.reporting.localization.Localization(__file__, 1948, 23), add_3197, *[asarray_call_result_3202, self_3203], **kwargs_3204)
        
        # Processing the call keyword arguments (line 1948)
        kwargs_3206 = {}
        # Getting the type of 'asarray' (line 1948)
        asarray_3196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1948, 15), 'asarray', False)
        # Calling asarray(args, kwargs) (line 1948)
        asarray_call_result_3207 = invoke(stypy.reporting.localization.Localization(__file__, 1948, 15), asarray_3196, *[add_call_result_3205], **kwargs_3206)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1948)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1948, 8), 'stypy_return_type', asarray_call_result_3207)
        
        # ################# End of '__radd__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__radd__' in the type store
        # Getting the type of 'stypy_return_type' (line 1939)
        stypy_return_type_3208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1939, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3208)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__radd__'
        return stypy_return_type_3208


    @norecursion
    def __mul__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__mul__'
        module_type_store = module_type_store.open_function_context('__mul__', 1950, 4, False)
        # Assigning a type to the variable 'self' (line 1951)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1951, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        chararray.__mul__.__dict__.__setitem__('stypy_localization', localization)
        chararray.__mul__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        chararray.__mul__.__dict__.__setitem__('stypy_type_store', module_type_store)
        chararray.__mul__.__dict__.__setitem__('stypy_function_name', 'chararray.__mul__')
        chararray.__mul__.__dict__.__setitem__('stypy_param_names_list', ['i'])
        chararray.__mul__.__dict__.__setitem__('stypy_varargs_param_name', None)
        chararray.__mul__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        chararray.__mul__.__dict__.__setitem__('stypy_call_defaults', defaults)
        chararray.__mul__.__dict__.__setitem__('stypy_call_varargs', varargs)
        chararray.__mul__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        chararray.__mul__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'chararray.__mul__', ['i'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__mul__', localization, ['i'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__mul__(...)' code ##################

        str_3209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1958, (-1)), 'str', '\n        Return (self * i), that is string multiple concatenation,\n        element-wise.\n\n        See also\n        --------\n        multiply\n        ')
        
        # Call to asarray(...): (line 1959)
        # Processing the call arguments (line 1959)
        
        # Call to multiply(...): (line 1959)
        # Processing the call arguments (line 1959)
        # Getting the type of 'self' (line 1959)
        self_3212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1959, 32), 'self', False)
        # Getting the type of 'i' (line 1959)
        i_3213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1959, 38), 'i', False)
        # Processing the call keyword arguments (line 1959)
        kwargs_3214 = {}
        # Getting the type of 'multiply' (line 1959)
        multiply_3211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1959, 23), 'multiply', False)
        # Calling multiply(args, kwargs) (line 1959)
        multiply_call_result_3215 = invoke(stypy.reporting.localization.Localization(__file__, 1959, 23), multiply_3211, *[self_3212, i_3213], **kwargs_3214)
        
        # Processing the call keyword arguments (line 1959)
        kwargs_3216 = {}
        # Getting the type of 'asarray' (line 1959)
        asarray_3210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1959, 15), 'asarray', False)
        # Calling asarray(args, kwargs) (line 1959)
        asarray_call_result_3217 = invoke(stypy.reporting.localization.Localization(__file__, 1959, 15), asarray_3210, *[multiply_call_result_3215], **kwargs_3216)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1959)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1959, 8), 'stypy_return_type', asarray_call_result_3217)
        
        # ################# End of '__mul__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__mul__' in the type store
        # Getting the type of 'stypy_return_type' (line 1950)
        stypy_return_type_3218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1950, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3218)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__mul__'
        return stypy_return_type_3218


    @norecursion
    def __rmul__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__rmul__'
        module_type_store = module_type_store.open_function_context('__rmul__', 1961, 4, False)
        # Assigning a type to the variable 'self' (line 1962)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1962, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        chararray.__rmul__.__dict__.__setitem__('stypy_localization', localization)
        chararray.__rmul__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        chararray.__rmul__.__dict__.__setitem__('stypy_type_store', module_type_store)
        chararray.__rmul__.__dict__.__setitem__('stypy_function_name', 'chararray.__rmul__')
        chararray.__rmul__.__dict__.__setitem__('stypy_param_names_list', ['i'])
        chararray.__rmul__.__dict__.__setitem__('stypy_varargs_param_name', None)
        chararray.__rmul__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        chararray.__rmul__.__dict__.__setitem__('stypy_call_defaults', defaults)
        chararray.__rmul__.__dict__.__setitem__('stypy_call_varargs', varargs)
        chararray.__rmul__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        chararray.__rmul__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'chararray.__rmul__', ['i'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__rmul__', localization, ['i'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__rmul__(...)' code ##################

        str_3219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1969, (-1)), 'str', '\n        Return (self * i), that is string multiple concatenation,\n        element-wise.\n\n        See also\n        --------\n        multiply\n        ')
        
        # Call to asarray(...): (line 1970)
        # Processing the call arguments (line 1970)
        
        # Call to multiply(...): (line 1970)
        # Processing the call arguments (line 1970)
        # Getting the type of 'self' (line 1970)
        self_3222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1970, 32), 'self', False)
        # Getting the type of 'i' (line 1970)
        i_3223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1970, 38), 'i', False)
        # Processing the call keyword arguments (line 1970)
        kwargs_3224 = {}
        # Getting the type of 'multiply' (line 1970)
        multiply_3221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1970, 23), 'multiply', False)
        # Calling multiply(args, kwargs) (line 1970)
        multiply_call_result_3225 = invoke(stypy.reporting.localization.Localization(__file__, 1970, 23), multiply_3221, *[self_3222, i_3223], **kwargs_3224)
        
        # Processing the call keyword arguments (line 1970)
        kwargs_3226 = {}
        # Getting the type of 'asarray' (line 1970)
        asarray_3220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1970, 15), 'asarray', False)
        # Calling asarray(args, kwargs) (line 1970)
        asarray_call_result_3227 = invoke(stypy.reporting.localization.Localization(__file__, 1970, 15), asarray_3220, *[multiply_call_result_3225], **kwargs_3226)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1970)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1970, 8), 'stypy_return_type', asarray_call_result_3227)
        
        # ################# End of '__rmul__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__rmul__' in the type store
        # Getting the type of 'stypy_return_type' (line 1961)
        stypy_return_type_3228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1961, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3228)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__rmul__'
        return stypy_return_type_3228


    @norecursion
    def __mod__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__mod__'
        module_type_store = module_type_store.open_function_context('__mod__', 1972, 4, False)
        # Assigning a type to the variable 'self' (line 1973)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1973, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        chararray.__mod__.__dict__.__setitem__('stypy_localization', localization)
        chararray.__mod__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        chararray.__mod__.__dict__.__setitem__('stypy_type_store', module_type_store)
        chararray.__mod__.__dict__.__setitem__('stypy_function_name', 'chararray.__mod__')
        chararray.__mod__.__dict__.__setitem__('stypy_param_names_list', ['i'])
        chararray.__mod__.__dict__.__setitem__('stypy_varargs_param_name', None)
        chararray.__mod__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        chararray.__mod__.__dict__.__setitem__('stypy_call_defaults', defaults)
        chararray.__mod__.__dict__.__setitem__('stypy_call_varargs', varargs)
        chararray.__mod__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        chararray.__mod__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'chararray.__mod__', ['i'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__mod__', localization, ['i'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__mod__(...)' code ##################

        str_3229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1981, (-1)), 'str', '\n        Return (self % i), that is pre-Python 2.6 string formatting\n        (iterpolation), element-wise for a pair of array_likes of `string_`\n        or `unicode_`.\n\n        See also\n        --------\n        mod\n        ')
        
        # Call to asarray(...): (line 1982)
        # Processing the call arguments (line 1982)
        
        # Call to mod(...): (line 1982)
        # Processing the call arguments (line 1982)
        # Getting the type of 'self' (line 1982)
        self_3232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1982, 27), 'self', False)
        # Getting the type of 'i' (line 1982)
        i_3233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1982, 33), 'i', False)
        # Processing the call keyword arguments (line 1982)
        kwargs_3234 = {}
        # Getting the type of 'mod' (line 1982)
        mod_3231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1982, 23), 'mod', False)
        # Calling mod(args, kwargs) (line 1982)
        mod_call_result_3235 = invoke(stypy.reporting.localization.Localization(__file__, 1982, 23), mod_3231, *[self_3232, i_3233], **kwargs_3234)
        
        # Processing the call keyword arguments (line 1982)
        kwargs_3236 = {}
        # Getting the type of 'asarray' (line 1982)
        asarray_3230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1982, 15), 'asarray', False)
        # Calling asarray(args, kwargs) (line 1982)
        asarray_call_result_3237 = invoke(stypy.reporting.localization.Localization(__file__, 1982, 15), asarray_3230, *[mod_call_result_3235], **kwargs_3236)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1982)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1982, 8), 'stypy_return_type', asarray_call_result_3237)
        
        # ################# End of '__mod__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__mod__' in the type store
        # Getting the type of 'stypy_return_type' (line 1972)
        stypy_return_type_3238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1972, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3238)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__mod__'
        return stypy_return_type_3238


    @norecursion
    def __rmod__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__rmod__'
        module_type_store = module_type_store.open_function_context('__rmod__', 1984, 4, False)
        # Assigning a type to the variable 'self' (line 1985)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1985, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        chararray.__rmod__.__dict__.__setitem__('stypy_localization', localization)
        chararray.__rmod__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        chararray.__rmod__.__dict__.__setitem__('stypy_type_store', module_type_store)
        chararray.__rmod__.__dict__.__setitem__('stypy_function_name', 'chararray.__rmod__')
        chararray.__rmod__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        chararray.__rmod__.__dict__.__setitem__('stypy_varargs_param_name', None)
        chararray.__rmod__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        chararray.__rmod__.__dict__.__setitem__('stypy_call_defaults', defaults)
        chararray.__rmod__.__dict__.__setitem__('stypy_call_varargs', varargs)
        chararray.__rmod__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        chararray.__rmod__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'chararray.__rmod__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__rmod__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__rmod__(...)' code ##################

        # Getting the type of 'NotImplemented' (line 1985)
        NotImplemented_3239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1985, 15), 'NotImplemented')
        # Assigning a type to the variable 'stypy_return_type' (line 1985)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1985, 8), 'stypy_return_type', NotImplemented_3239)
        
        # ################# End of '__rmod__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__rmod__' in the type store
        # Getting the type of 'stypy_return_type' (line 1984)
        stypy_return_type_3240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1984, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3240)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__rmod__'
        return stypy_return_type_3240


    @norecursion
    def argsort(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_3241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1987, 27), 'int')
        str_3242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1987, 36), 'str', 'quicksort')
        # Getting the type of 'None' (line 1987)
        None_3243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1987, 55), 'None')
        defaults = [int_3241, str_3242, None_3243]
        # Create a new context for function 'argsort'
        module_type_store = module_type_store.open_function_context('argsort', 1987, 4, False)
        # Assigning a type to the variable 'self' (line 1988)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1988, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        chararray.argsort.__dict__.__setitem__('stypy_localization', localization)
        chararray.argsort.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        chararray.argsort.__dict__.__setitem__('stypy_type_store', module_type_store)
        chararray.argsort.__dict__.__setitem__('stypy_function_name', 'chararray.argsort')
        chararray.argsort.__dict__.__setitem__('stypy_param_names_list', ['axis', 'kind', 'order'])
        chararray.argsort.__dict__.__setitem__('stypy_varargs_param_name', None)
        chararray.argsort.__dict__.__setitem__('stypy_kwargs_param_name', None)
        chararray.argsort.__dict__.__setitem__('stypy_call_defaults', defaults)
        chararray.argsort.__dict__.__setitem__('stypy_call_varargs', varargs)
        chararray.argsort.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        chararray.argsort.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'chararray.argsort', ['axis', 'kind', 'order'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'argsort', localization, ['axis', 'kind', 'order'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'argsort(...)' code ##################

        str_3244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2004, (-1)), 'str', '\n        Return the indices that sort the array lexicographically.\n\n        For full documentation see `numpy.argsort`, for which this method is\n        in fact merely a "thin wrapper."\n\n        Examples\n        --------\n        >>> c = np.array([\'a1b c\', \'1b ca\', \'b ca1\', \'Ca1b\'], \'S5\')\n        >>> c = c.view(np.chararray); c\n        chararray([\'a1b c\', \'1b ca\', \'b ca1\', \'Ca1b\'],\n              dtype=\'|S5\')\n        >>> c[c.argsort()]\n        chararray([\'1b ca\', \'Ca1b\', \'a1b c\', \'b ca1\'],\n              dtype=\'|S5\')\n\n        ')
        
        # Call to argsort(...): (line 2005)
        # Processing the call arguments (line 2005)
        # Getting the type of 'axis' (line 2005)
        axis_3250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2005, 40), 'axis', False)
        # Getting the type of 'kind' (line 2005)
        kind_3251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2005, 46), 'kind', False)
        # Getting the type of 'order' (line 2005)
        order_3252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2005, 52), 'order', False)
        # Processing the call keyword arguments (line 2005)
        kwargs_3253 = {}
        
        # Call to __array__(...): (line 2005)
        # Processing the call keyword arguments (line 2005)
        kwargs_3247 = {}
        # Getting the type of 'self' (line 2005)
        self_3245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2005, 15), 'self', False)
        # Obtaining the member '__array__' of a type (line 2005)
        array___3246 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2005, 15), self_3245, '__array__')
        # Calling __array__(args, kwargs) (line 2005)
        array___call_result_3248 = invoke(stypy.reporting.localization.Localization(__file__, 2005, 15), array___3246, *[], **kwargs_3247)
        
        # Obtaining the member 'argsort' of a type (line 2005)
        argsort_3249 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2005, 15), array___call_result_3248, 'argsort')
        # Calling argsort(args, kwargs) (line 2005)
        argsort_call_result_3254 = invoke(stypy.reporting.localization.Localization(__file__, 2005, 15), argsort_3249, *[axis_3250, kind_3251, order_3252], **kwargs_3253)
        
        # Assigning a type to the variable 'stypy_return_type' (line 2005)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2005, 8), 'stypy_return_type', argsort_call_result_3254)
        
        # ################# End of 'argsort(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'argsort' in the type store
        # Getting the type of 'stypy_return_type' (line 1987)
        stypy_return_type_3255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1987, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3255)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'argsort'
        return stypy_return_type_3255


    @norecursion
    def capitalize(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'capitalize'
        module_type_store = module_type_store.open_function_context('capitalize', 2008, 4, False)
        # Assigning a type to the variable 'self' (line 2009)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2009, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        chararray.capitalize.__dict__.__setitem__('stypy_localization', localization)
        chararray.capitalize.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        chararray.capitalize.__dict__.__setitem__('stypy_type_store', module_type_store)
        chararray.capitalize.__dict__.__setitem__('stypy_function_name', 'chararray.capitalize')
        chararray.capitalize.__dict__.__setitem__('stypy_param_names_list', [])
        chararray.capitalize.__dict__.__setitem__('stypy_varargs_param_name', None)
        chararray.capitalize.__dict__.__setitem__('stypy_kwargs_param_name', None)
        chararray.capitalize.__dict__.__setitem__('stypy_call_defaults', defaults)
        chararray.capitalize.__dict__.__setitem__('stypy_call_varargs', varargs)
        chararray.capitalize.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        chararray.capitalize.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'chararray.capitalize', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'capitalize', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'capitalize(...)' code ##################

        str_3256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2017, (-1)), 'str', '\n        Return a copy of `self` with only the first character of each element\n        capitalized.\n\n        See also\n        --------\n        char.capitalize\n\n        ')
        
        # Call to asarray(...): (line 2018)
        # Processing the call arguments (line 2018)
        
        # Call to capitalize(...): (line 2018)
        # Processing the call arguments (line 2018)
        # Getting the type of 'self' (line 2018)
        self_3259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2018, 34), 'self', False)
        # Processing the call keyword arguments (line 2018)
        kwargs_3260 = {}
        # Getting the type of 'capitalize' (line 2018)
        capitalize_3258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2018, 23), 'capitalize', False)
        # Calling capitalize(args, kwargs) (line 2018)
        capitalize_call_result_3261 = invoke(stypy.reporting.localization.Localization(__file__, 2018, 23), capitalize_3258, *[self_3259], **kwargs_3260)
        
        # Processing the call keyword arguments (line 2018)
        kwargs_3262 = {}
        # Getting the type of 'asarray' (line 2018)
        asarray_3257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2018, 15), 'asarray', False)
        # Calling asarray(args, kwargs) (line 2018)
        asarray_call_result_3263 = invoke(stypy.reporting.localization.Localization(__file__, 2018, 15), asarray_3257, *[capitalize_call_result_3261], **kwargs_3262)
        
        # Assigning a type to the variable 'stypy_return_type' (line 2018)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2018, 8), 'stypy_return_type', asarray_call_result_3263)
        
        # ################# End of 'capitalize(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'capitalize' in the type store
        # Getting the type of 'stypy_return_type' (line 2008)
        stypy_return_type_3264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2008, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3264)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'capitalize'
        return stypy_return_type_3264


    @norecursion
    def center(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        str_3265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2020, 37), 'str', ' ')
        defaults = [str_3265]
        # Create a new context for function 'center'
        module_type_store = module_type_store.open_function_context('center', 2020, 4, False)
        # Assigning a type to the variable 'self' (line 2021)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2021, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        chararray.center.__dict__.__setitem__('stypy_localization', localization)
        chararray.center.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        chararray.center.__dict__.__setitem__('stypy_type_store', module_type_store)
        chararray.center.__dict__.__setitem__('stypy_function_name', 'chararray.center')
        chararray.center.__dict__.__setitem__('stypy_param_names_list', ['width', 'fillchar'])
        chararray.center.__dict__.__setitem__('stypy_varargs_param_name', None)
        chararray.center.__dict__.__setitem__('stypy_kwargs_param_name', None)
        chararray.center.__dict__.__setitem__('stypy_call_defaults', defaults)
        chararray.center.__dict__.__setitem__('stypy_call_varargs', varargs)
        chararray.center.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        chararray.center.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'chararray.center', ['width', 'fillchar'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'center', localization, ['width', 'fillchar'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'center(...)' code ##################

        str_3266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2028, (-1)), 'str', '\n        Return a copy of `self` with its elements centered in a\n        string of length `width`.\n\n        See also\n        --------\n        center\n        ')
        
        # Call to asarray(...): (line 2029)
        # Processing the call arguments (line 2029)
        
        # Call to center(...): (line 2029)
        # Processing the call arguments (line 2029)
        # Getting the type of 'self' (line 2029)
        self_3269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2029, 30), 'self', False)
        # Getting the type of 'width' (line 2029)
        width_3270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2029, 36), 'width', False)
        # Getting the type of 'fillchar' (line 2029)
        fillchar_3271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2029, 43), 'fillchar', False)
        # Processing the call keyword arguments (line 2029)
        kwargs_3272 = {}
        # Getting the type of 'center' (line 2029)
        center_3268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2029, 23), 'center', False)
        # Calling center(args, kwargs) (line 2029)
        center_call_result_3273 = invoke(stypy.reporting.localization.Localization(__file__, 2029, 23), center_3268, *[self_3269, width_3270, fillchar_3271], **kwargs_3272)
        
        # Processing the call keyword arguments (line 2029)
        kwargs_3274 = {}
        # Getting the type of 'asarray' (line 2029)
        asarray_3267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2029, 15), 'asarray', False)
        # Calling asarray(args, kwargs) (line 2029)
        asarray_call_result_3275 = invoke(stypy.reporting.localization.Localization(__file__, 2029, 15), asarray_3267, *[center_call_result_3273], **kwargs_3274)
        
        # Assigning a type to the variable 'stypy_return_type' (line 2029)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2029, 8), 'stypy_return_type', asarray_call_result_3275)
        
        # ################# End of 'center(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'center' in the type store
        # Getting the type of 'stypy_return_type' (line 2020)
        stypy_return_type_3276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2020, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3276)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'center'
        return stypy_return_type_3276


    @norecursion
    def count(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_3277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2031, 31), 'int')
        # Getting the type of 'None' (line 2031)
        None_3278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2031, 38), 'None')
        defaults = [int_3277, None_3278]
        # Create a new context for function 'count'
        module_type_store = module_type_store.open_function_context('count', 2031, 4, False)
        # Assigning a type to the variable 'self' (line 2032)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2032, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        chararray.count.__dict__.__setitem__('stypy_localization', localization)
        chararray.count.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        chararray.count.__dict__.__setitem__('stypy_type_store', module_type_store)
        chararray.count.__dict__.__setitem__('stypy_function_name', 'chararray.count')
        chararray.count.__dict__.__setitem__('stypy_param_names_list', ['sub', 'start', 'end'])
        chararray.count.__dict__.__setitem__('stypy_varargs_param_name', None)
        chararray.count.__dict__.__setitem__('stypy_kwargs_param_name', None)
        chararray.count.__dict__.__setitem__('stypy_call_defaults', defaults)
        chararray.count.__dict__.__setitem__('stypy_call_varargs', varargs)
        chararray.count.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        chararray.count.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'chararray.count', ['sub', 'start', 'end'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'count', localization, ['sub', 'start', 'end'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'count(...)' code ##################

        str_3279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2040, (-1)), 'str', '\n        Returns an array with the number of non-overlapping occurrences of\n        substring `sub` in the range [`start`, `end`].\n\n        See also\n        --------\n        char.count\n\n        ')
        
        # Call to count(...): (line 2041)
        # Processing the call arguments (line 2041)
        # Getting the type of 'self' (line 2041)
        self_3281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2041, 21), 'self', False)
        # Getting the type of 'sub' (line 2041)
        sub_3282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2041, 27), 'sub', False)
        # Getting the type of 'start' (line 2041)
        start_3283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2041, 32), 'start', False)
        # Getting the type of 'end' (line 2041)
        end_3284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2041, 39), 'end', False)
        # Processing the call keyword arguments (line 2041)
        kwargs_3285 = {}
        # Getting the type of 'count' (line 2041)
        count_3280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2041, 15), 'count', False)
        # Calling count(args, kwargs) (line 2041)
        count_call_result_3286 = invoke(stypy.reporting.localization.Localization(__file__, 2041, 15), count_3280, *[self_3281, sub_3282, start_3283, end_3284], **kwargs_3285)
        
        # Assigning a type to the variable 'stypy_return_type' (line 2041)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2041, 8), 'stypy_return_type', count_call_result_3286)
        
        # ################# End of 'count(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'count' in the type store
        # Getting the type of 'stypy_return_type' (line 2031)
        stypy_return_type_3287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2031, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3287)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'count'
        return stypy_return_type_3287


    @norecursion
    def decode(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 2043)
        None_3288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2043, 30), 'None')
        # Getting the type of 'None' (line 2043)
        None_3289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2043, 43), 'None')
        defaults = [None_3288, None_3289]
        # Create a new context for function 'decode'
        module_type_store = module_type_store.open_function_context('decode', 2043, 4, False)
        # Assigning a type to the variable 'self' (line 2044)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2044, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        chararray.decode.__dict__.__setitem__('stypy_localization', localization)
        chararray.decode.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        chararray.decode.__dict__.__setitem__('stypy_type_store', module_type_store)
        chararray.decode.__dict__.__setitem__('stypy_function_name', 'chararray.decode')
        chararray.decode.__dict__.__setitem__('stypy_param_names_list', ['encoding', 'errors'])
        chararray.decode.__dict__.__setitem__('stypy_varargs_param_name', None)
        chararray.decode.__dict__.__setitem__('stypy_kwargs_param_name', None)
        chararray.decode.__dict__.__setitem__('stypy_call_defaults', defaults)
        chararray.decode.__dict__.__setitem__('stypy_call_varargs', varargs)
        chararray.decode.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        chararray.decode.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'chararray.decode', ['encoding', 'errors'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'decode', localization, ['encoding', 'errors'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'decode(...)' code ##################

        str_3290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2051, (-1)), 'str', '\n        Calls `str.decode` element-wise.\n\n        See also\n        --------\n        char.decode\n\n        ')
        
        # Call to decode(...): (line 2052)
        # Processing the call arguments (line 2052)
        # Getting the type of 'self' (line 2052)
        self_3292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2052, 22), 'self', False)
        # Getting the type of 'encoding' (line 2052)
        encoding_3293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2052, 28), 'encoding', False)
        # Getting the type of 'errors' (line 2052)
        errors_3294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2052, 38), 'errors', False)
        # Processing the call keyword arguments (line 2052)
        kwargs_3295 = {}
        # Getting the type of 'decode' (line 2052)
        decode_3291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2052, 15), 'decode', False)
        # Calling decode(args, kwargs) (line 2052)
        decode_call_result_3296 = invoke(stypy.reporting.localization.Localization(__file__, 2052, 15), decode_3291, *[self_3292, encoding_3293, errors_3294], **kwargs_3295)
        
        # Assigning a type to the variable 'stypy_return_type' (line 2052)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2052, 8), 'stypy_return_type', decode_call_result_3296)
        
        # ################# End of 'decode(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'decode' in the type store
        # Getting the type of 'stypy_return_type' (line 2043)
        stypy_return_type_3297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2043, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3297)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'decode'
        return stypy_return_type_3297


    @norecursion
    def encode(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 2054)
        None_3298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2054, 30), 'None')
        # Getting the type of 'None' (line 2054)
        None_3299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2054, 43), 'None')
        defaults = [None_3298, None_3299]
        # Create a new context for function 'encode'
        module_type_store = module_type_store.open_function_context('encode', 2054, 4, False)
        # Assigning a type to the variable 'self' (line 2055)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2055, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        chararray.encode.__dict__.__setitem__('stypy_localization', localization)
        chararray.encode.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        chararray.encode.__dict__.__setitem__('stypy_type_store', module_type_store)
        chararray.encode.__dict__.__setitem__('stypy_function_name', 'chararray.encode')
        chararray.encode.__dict__.__setitem__('stypy_param_names_list', ['encoding', 'errors'])
        chararray.encode.__dict__.__setitem__('stypy_varargs_param_name', None)
        chararray.encode.__dict__.__setitem__('stypy_kwargs_param_name', None)
        chararray.encode.__dict__.__setitem__('stypy_call_defaults', defaults)
        chararray.encode.__dict__.__setitem__('stypy_call_varargs', varargs)
        chararray.encode.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        chararray.encode.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'chararray.encode', ['encoding', 'errors'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'encode', localization, ['encoding', 'errors'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'encode(...)' code ##################

        str_3300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2062, (-1)), 'str', '\n        Calls `str.encode` element-wise.\n\n        See also\n        --------\n        char.encode\n\n        ')
        
        # Call to encode(...): (line 2063)
        # Processing the call arguments (line 2063)
        # Getting the type of 'self' (line 2063)
        self_3302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2063, 22), 'self', False)
        # Getting the type of 'encoding' (line 2063)
        encoding_3303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2063, 28), 'encoding', False)
        # Getting the type of 'errors' (line 2063)
        errors_3304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2063, 38), 'errors', False)
        # Processing the call keyword arguments (line 2063)
        kwargs_3305 = {}
        # Getting the type of 'encode' (line 2063)
        encode_3301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2063, 15), 'encode', False)
        # Calling encode(args, kwargs) (line 2063)
        encode_call_result_3306 = invoke(stypy.reporting.localization.Localization(__file__, 2063, 15), encode_3301, *[self_3302, encoding_3303, errors_3304], **kwargs_3305)
        
        # Assigning a type to the variable 'stypy_return_type' (line 2063)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2063, 8), 'stypy_return_type', encode_call_result_3306)
        
        # ################# End of 'encode(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'encode' in the type store
        # Getting the type of 'stypy_return_type' (line 2054)
        stypy_return_type_3307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2054, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3307)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'encode'
        return stypy_return_type_3307


    @norecursion
    def endswith(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_3308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2065, 37), 'int')
        # Getting the type of 'None' (line 2065)
        None_3309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2065, 44), 'None')
        defaults = [int_3308, None_3309]
        # Create a new context for function 'endswith'
        module_type_store = module_type_store.open_function_context('endswith', 2065, 4, False)
        # Assigning a type to the variable 'self' (line 2066)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2066, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        chararray.endswith.__dict__.__setitem__('stypy_localization', localization)
        chararray.endswith.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        chararray.endswith.__dict__.__setitem__('stypy_type_store', module_type_store)
        chararray.endswith.__dict__.__setitem__('stypy_function_name', 'chararray.endswith')
        chararray.endswith.__dict__.__setitem__('stypy_param_names_list', ['suffix', 'start', 'end'])
        chararray.endswith.__dict__.__setitem__('stypy_varargs_param_name', None)
        chararray.endswith.__dict__.__setitem__('stypy_kwargs_param_name', None)
        chararray.endswith.__dict__.__setitem__('stypy_call_defaults', defaults)
        chararray.endswith.__dict__.__setitem__('stypy_call_varargs', varargs)
        chararray.endswith.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        chararray.endswith.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'chararray.endswith', ['suffix', 'start', 'end'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'endswith', localization, ['suffix', 'start', 'end'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'endswith(...)' code ##################

        str_3310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2074, (-1)), 'str', '\n        Returns a boolean array which is `True` where the string element\n        in `self` ends with `suffix`, otherwise `False`.\n\n        See also\n        --------\n        char.endswith\n\n        ')
        
        # Call to endswith(...): (line 2075)
        # Processing the call arguments (line 2075)
        # Getting the type of 'self' (line 2075)
        self_3312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2075, 24), 'self', False)
        # Getting the type of 'suffix' (line 2075)
        suffix_3313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2075, 30), 'suffix', False)
        # Getting the type of 'start' (line 2075)
        start_3314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2075, 38), 'start', False)
        # Getting the type of 'end' (line 2075)
        end_3315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2075, 45), 'end', False)
        # Processing the call keyword arguments (line 2075)
        kwargs_3316 = {}
        # Getting the type of 'endswith' (line 2075)
        endswith_3311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2075, 15), 'endswith', False)
        # Calling endswith(args, kwargs) (line 2075)
        endswith_call_result_3317 = invoke(stypy.reporting.localization.Localization(__file__, 2075, 15), endswith_3311, *[self_3312, suffix_3313, start_3314, end_3315], **kwargs_3316)
        
        # Assigning a type to the variable 'stypy_return_type' (line 2075)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2075, 8), 'stypy_return_type', endswith_call_result_3317)
        
        # ################# End of 'endswith(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'endswith' in the type store
        # Getting the type of 'stypy_return_type' (line 2065)
        stypy_return_type_3318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2065, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3318)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'endswith'
        return stypy_return_type_3318


    @norecursion
    def expandtabs(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_3319 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2077, 33), 'int')
        defaults = [int_3319]
        # Create a new context for function 'expandtabs'
        module_type_store = module_type_store.open_function_context('expandtabs', 2077, 4, False)
        # Assigning a type to the variable 'self' (line 2078)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2078, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        chararray.expandtabs.__dict__.__setitem__('stypy_localization', localization)
        chararray.expandtabs.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        chararray.expandtabs.__dict__.__setitem__('stypy_type_store', module_type_store)
        chararray.expandtabs.__dict__.__setitem__('stypy_function_name', 'chararray.expandtabs')
        chararray.expandtabs.__dict__.__setitem__('stypy_param_names_list', ['tabsize'])
        chararray.expandtabs.__dict__.__setitem__('stypy_varargs_param_name', None)
        chararray.expandtabs.__dict__.__setitem__('stypy_kwargs_param_name', None)
        chararray.expandtabs.__dict__.__setitem__('stypy_call_defaults', defaults)
        chararray.expandtabs.__dict__.__setitem__('stypy_call_varargs', varargs)
        chararray.expandtabs.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        chararray.expandtabs.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'chararray.expandtabs', ['tabsize'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'expandtabs', localization, ['tabsize'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'expandtabs(...)' code ##################

        str_3320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2086, (-1)), 'str', '\n        Return a copy of each string element where all tab characters are\n        replaced by one or more spaces.\n\n        See also\n        --------\n        char.expandtabs\n\n        ')
        
        # Call to asarray(...): (line 2087)
        # Processing the call arguments (line 2087)
        
        # Call to expandtabs(...): (line 2087)
        # Processing the call arguments (line 2087)
        # Getting the type of 'self' (line 2087)
        self_3323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2087, 34), 'self', False)
        # Getting the type of 'tabsize' (line 2087)
        tabsize_3324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2087, 40), 'tabsize', False)
        # Processing the call keyword arguments (line 2087)
        kwargs_3325 = {}
        # Getting the type of 'expandtabs' (line 2087)
        expandtabs_3322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2087, 23), 'expandtabs', False)
        # Calling expandtabs(args, kwargs) (line 2087)
        expandtabs_call_result_3326 = invoke(stypy.reporting.localization.Localization(__file__, 2087, 23), expandtabs_3322, *[self_3323, tabsize_3324], **kwargs_3325)
        
        # Processing the call keyword arguments (line 2087)
        kwargs_3327 = {}
        # Getting the type of 'asarray' (line 2087)
        asarray_3321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2087, 15), 'asarray', False)
        # Calling asarray(args, kwargs) (line 2087)
        asarray_call_result_3328 = invoke(stypy.reporting.localization.Localization(__file__, 2087, 15), asarray_3321, *[expandtabs_call_result_3326], **kwargs_3327)
        
        # Assigning a type to the variable 'stypy_return_type' (line 2087)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2087, 8), 'stypy_return_type', asarray_call_result_3328)
        
        # ################# End of 'expandtabs(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'expandtabs' in the type store
        # Getting the type of 'stypy_return_type' (line 2077)
        stypy_return_type_3329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2077, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3329)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'expandtabs'
        return stypy_return_type_3329


    @norecursion
    def find(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_3330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2089, 30), 'int')
        # Getting the type of 'None' (line 2089)
        None_3331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2089, 37), 'None')
        defaults = [int_3330, None_3331]
        # Create a new context for function 'find'
        module_type_store = module_type_store.open_function_context('find', 2089, 4, False)
        # Assigning a type to the variable 'self' (line 2090)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2090, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        chararray.find.__dict__.__setitem__('stypy_localization', localization)
        chararray.find.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        chararray.find.__dict__.__setitem__('stypy_type_store', module_type_store)
        chararray.find.__dict__.__setitem__('stypy_function_name', 'chararray.find')
        chararray.find.__dict__.__setitem__('stypy_param_names_list', ['sub', 'start', 'end'])
        chararray.find.__dict__.__setitem__('stypy_varargs_param_name', None)
        chararray.find.__dict__.__setitem__('stypy_kwargs_param_name', None)
        chararray.find.__dict__.__setitem__('stypy_call_defaults', defaults)
        chararray.find.__dict__.__setitem__('stypy_call_varargs', varargs)
        chararray.find.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        chararray.find.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'chararray.find', ['sub', 'start', 'end'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'find', localization, ['sub', 'start', 'end'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'find(...)' code ##################

        str_3332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2098, (-1)), 'str', '\n        For each element, return the lowest index in the string where\n        substring `sub` is found.\n\n        See also\n        --------\n        char.find\n\n        ')
        
        # Call to find(...): (line 2099)
        # Processing the call arguments (line 2099)
        # Getting the type of 'self' (line 2099)
        self_3334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2099, 20), 'self', False)
        # Getting the type of 'sub' (line 2099)
        sub_3335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2099, 26), 'sub', False)
        # Getting the type of 'start' (line 2099)
        start_3336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2099, 31), 'start', False)
        # Getting the type of 'end' (line 2099)
        end_3337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2099, 38), 'end', False)
        # Processing the call keyword arguments (line 2099)
        kwargs_3338 = {}
        # Getting the type of 'find' (line 2099)
        find_3333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2099, 15), 'find', False)
        # Calling find(args, kwargs) (line 2099)
        find_call_result_3339 = invoke(stypy.reporting.localization.Localization(__file__, 2099, 15), find_3333, *[self_3334, sub_3335, start_3336, end_3337], **kwargs_3338)
        
        # Assigning a type to the variable 'stypy_return_type' (line 2099)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2099, 8), 'stypy_return_type', find_call_result_3339)
        
        # ################# End of 'find(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'find' in the type store
        # Getting the type of 'stypy_return_type' (line 2089)
        stypy_return_type_3340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2089, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3340)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'find'
        return stypy_return_type_3340


    @norecursion
    def index(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_3341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2101, 31), 'int')
        # Getting the type of 'None' (line 2101)
        None_3342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2101, 38), 'None')
        defaults = [int_3341, None_3342]
        # Create a new context for function 'index'
        module_type_store = module_type_store.open_function_context('index', 2101, 4, False)
        # Assigning a type to the variable 'self' (line 2102)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2102, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        chararray.index.__dict__.__setitem__('stypy_localization', localization)
        chararray.index.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        chararray.index.__dict__.__setitem__('stypy_type_store', module_type_store)
        chararray.index.__dict__.__setitem__('stypy_function_name', 'chararray.index')
        chararray.index.__dict__.__setitem__('stypy_param_names_list', ['sub', 'start', 'end'])
        chararray.index.__dict__.__setitem__('stypy_varargs_param_name', None)
        chararray.index.__dict__.__setitem__('stypy_kwargs_param_name', None)
        chararray.index.__dict__.__setitem__('stypy_call_defaults', defaults)
        chararray.index.__dict__.__setitem__('stypy_call_varargs', varargs)
        chararray.index.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        chararray.index.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'chararray.index', ['sub', 'start', 'end'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'index', localization, ['sub', 'start', 'end'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'index(...)' code ##################

        str_3343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2109, (-1)), 'str', '\n        Like `find`, but raises `ValueError` when the substring is not found.\n\n        See also\n        --------\n        char.index\n\n        ')
        
        # Call to index(...): (line 2110)
        # Processing the call arguments (line 2110)
        # Getting the type of 'self' (line 2110)
        self_3345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2110, 21), 'self', False)
        # Getting the type of 'sub' (line 2110)
        sub_3346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2110, 27), 'sub', False)
        # Getting the type of 'start' (line 2110)
        start_3347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2110, 32), 'start', False)
        # Getting the type of 'end' (line 2110)
        end_3348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2110, 39), 'end', False)
        # Processing the call keyword arguments (line 2110)
        kwargs_3349 = {}
        # Getting the type of 'index' (line 2110)
        index_3344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2110, 15), 'index', False)
        # Calling index(args, kwargs) (line 2110)
        index_call_result_3350 = invoke(stypy.reporting.localization.Localization(__file__, 2110, 15), index_3344, *[self_3345, sub_3346, start_3347, end_3348], **kwargs_3349)
        
        # Assigning a type to the variable 'stypy_return_type' (line 2110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2110, 8), 'stypy_return_type', index_call_result_3350)
        
        # ################# End of 'index(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'index' in the type store
        # Getting the type of 'stypy_return_type' (line 2101)
        stypy_return_type_3351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2101, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3351)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'index'
        return stypy_return_type_3351


    @norecursion
    def isalnum(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'isalnum'
        module_type_store = module_type_store.open_function_context('isalnum', 2112, 4, False)
        # Assigning a type to the variable 'self' (line 2113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2113, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        chararray.isalnum.__dict__.__setitem__('stypy_localization', localization)
        chararray.isalnum.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        chararray.isalnum.__dict__.__setitem__('stypy_type_store', module_type_store)
        chararray.isalnum.__dict__.__setitem__('stypy_function_name', 'chararray.isalnum')
        chararray.isalnum.__dict__.__setitem__('stypy_param_names_list', [])
        chararray.isalnum.__dict__.__setitem__('stypy_varargs_param_name', None)
        chararray.isalnum.__dict__.__setitem__('stypy_kwargs_param_name', None)
        chararray.isalnum.__dict__.__setitem__('stypy_call_defaults', defaults)
        chararray.isalnum.__dict__.__setitem__('stypy_call_varargs', varargs)
        chararray.isalnum.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        chararray.isalnum.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'chararray.isalnum', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'isalnum', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'isalnum(...)' code ##################

        str_3352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2122, (-1)), 'str', '\n        Returns true for each element if all characters in the string\n        are alphanumeric and there is at least one character, false\n        otherwise.\n\n        See also\n        --------\n        char.isalnum\n\n        ')
        
        # Call to isalnum(...): (line 2123)
        # Processing the call arguments (line 2123)
        # Getting the type of 'self' (line 2123)
        self_3354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2123, 23), 'self', False)
        # Processing the call keyword arguments (line 2123)
        kwargs_3355 = {}
        # Getting the type of 'isalnum' (line 2123)
        isalnum_3353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2123, 15), 'isalnum', False)
        # Calling isalnum(args, kwargs) (line 2123)
        isalnum_call_result_3356 = invoke(stypy.reporting.localization.Localization(__file__, 2123, 15), isalnum_3353, *[self_3354], **kwargs_3355)
        
        # Assigning a type to the variable 'stypy_return_type' (line 2123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2123, 8), 'stypy_return_type', isalnum_call_result_3356)
        
        # ################# End of 'isalnum(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'isalnum' in the type store
        # Getting the type of 'stypy_return_type' (line 2112)
        stypy_return_type_3357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2112, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3357)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'isalnum'
        return stypy_return_type_3357


    @norecursion
    def isalpha(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'isalpha'
        module_type_store = module_type_store.open_function_context('isalpha', 2125, 4, False)
        # Assigning a type to the variable 'self' (line 2126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2126, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        chararray.isalpha.__dict__.__setitem__('stypy_localization', localization)
        chararray.isalpha.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        chararray.isalpha.__dict__.__setitem__('stypy_type_store', module_type_store)
        chararray.isalpha.__dict__.__setitem__('stypy_function_name', 'chararray.isalpha')
        chararray.isalpha.__dict__.__setitem__('stypy_param_names_list', [])
        chararray.isalpha.__dict__.__setitem__('stypy_varargs_param_name', None)
        chararray.isalpha.__dict__.__setitem__('stypy_kwargs_param_name', None)
        chararray.isalpha.__dict__.__setitem__('stypy_call_defaults', defaults)
        chararray.isalpha.__dict__.__setitem__('stypy_call_varargs', varargs)
        chararray.isalpha.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        chararray.isalpha.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'chararray.isalpha', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'isalpha', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'isalpha(...)' code ##################

        str_3358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2135, (-1)), 'str', '\n        Returns true for each element if all characters in the string\n        are alphabetic and there is at least one character, false\n        otherwise.\n\n        See also\n        --------\n        char.isalpha\n\n        ')
        
        # Call to isalpha(...): (line 2136)
        # Processing the call arguments (line 2136)
        # Getting the type of 'self' (line 2136)
        self_3360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2136, 23), 'self', False)
        # Processing the call keyword arguments (line 2136)
        kwargs_3361 = {}
        # Getting the type of 'isalpha' (line 2136)
        isalpha_3359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2136, 15), 'isalpha', False)
        # Calling isalpha(args, kwargs) (line 2136)
        isalpha_call_result_3362 = invoke(stypy.reporting.localization.Localization(__file__, 2136, 15), isalpha_3359, *[self_3360], **kwargs_3361)
        
        # Assigning a type to the variable 'stypy_return_type' (line 2136)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2136, 8), 'stypy_return_type', isalpha_call_result_3362)
        
        # ################# End of 'isalpha(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'isalpha' in the type store
        # Getting the type of 'stypy_return_type' (line 2125)
        stypy_return_type_3363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2125, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3363)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'isalpha'
        return stypy_return_type_3363


    @norecursion
    def isdigit(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'isdigit'
        module_type_store = module_type_store.open_function_context('isdigit', 2138, 4, False)
        # Assigning a type to the variable 'self' (line 2139)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2139, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        chararray.isdigit.__dict__.__setitem__('stypy_localization', localization)
        chararray.isdigit.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        chararray.isdigit.__dict__.__setitem__('stypy_type_store', module_type_store)
        chararray.isdigit.__dict__.__setitem__('stypy_function_name', 'chararray.isdigit')
        chararray.isdigit.__dict__.__setitem__('stypy_param_names_list', [])
        chararray.isdigit.__dict__.__setitem__('stypy_varargs_param_name', None)
        chararray.isdigit.__dict__.__setitem__('stypy_kwargs_param_name', None)
        chararray.isdigit.__dict__.__setitem__('stypy_call_defaults', defaults)
        chararray.isdigit.__dict__.__setitem__('stypy_call_varargs', varargs)
        chararray.isdigit.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        chararray.isdigit.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'chararray.isdigit', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'isdigit', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'isdigit(...)' code ##################

        str_3364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2147, (-1)), 'str', '\n        Returns true for each element if all characters in the string are\n        digits and there is at least one character, false otherwise.\n\n        See also\n        --------\n        char.isdigit\n\n        ')
        
        # Call to isdigit(...): (line 2148)
        # Processing the call arguments (line 2148)
        # Getting the type of 'self' (line 2148)
        self_3366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2148, 23), 'self', False)
        # Processing the call keyword arguments (line 2148)
        kwargs_3367 = {}
        # Getting the type of 'isdigit' (line 2148)
        isdigit_3365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2148, 15), 'isdigit', False)
        # Calling isdigit(args, kwargs) (line 2148)
        isdigit_call_result_3368 = invoke(stypy.reporting.localization.Localization(__file__, 2148, 15), isdigit_3365, *[self_3366], **kwargs_3367)
        
        # Assigning a type to the variable 'stypy_return_type' (line 2148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2148, 8), 'stypy_return_type', isdigit_call_result_3368)
        
        # ################# End of 'isdigit(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'isdigit' in the type store
        # Getting the type of 'stypy_return_type' (line 2138)
        stypy_return_type_3369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2138, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3369)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'isdigit'
        return stypy_return_type_3369


    @norecursion
    def islower(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'islower'
        module_type_store = module_type_store.open_function_context('islower', 2150, 4, False)
        # Assigning a type to the variable 'self' (line 2151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2151, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        chararray.islower.__dict__.__setitem__('stypy_localization', localization)
        chararray.islower.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        chararray.islower.__dict__.__setitem__('stypy_type_store', module_type_store)
        chararray.islower.__dict__.__setitem__('stypy_function_name', 'chararray.islower')
        chararray.islower.__dict__.__setitem__('stypy_param_names_list', [])
        chararray.islower.__dict__.__setitem__('stypy_varargs_param_name', None)
        chararray.islower.__dict__.__setitem__('stypy_kwargs_param_name', None)
        chararray.islower.__dict__.__setitem__('stypy_call_defaults', defaults)
        chararray.islower.__dict__.__setitem__('stypy_call_varargs', varargs)
        chararray.islower.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        chararray.islower.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'chararray.islower', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'islower', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'islower(...)' code ##################

        str_3370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2160, (-1)), 'str', '\n        Returns true for each element if all cased characters in the\n        string are lowercase and there is at least one cased character,\n        false otherwise.\n\n        See also\n        --------\n        char.islower\n\n        ')
        
        # Call to islower(...): (line 2161)
        # Processing the call arguments (line 2161)
        # Getting the type of 'self' (line 2161)
        self_3372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2161, 23), 'self', False)
        # Processing the call keyword arguments (line 2161)
        kwargs_3373 = {}
        # Getting the type of 'islower' (line 2161)
        islower_3371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2161, 15), 'islower', False)
        # Calling islower(args, kwargs) (line 2161)
        islower_call_result_3374 = invoke(stypy.reporting.localization.Localization(__file__, 2161, 15), islower_3371, *[self_3372], **kwargs_3373)
        
        # Assigning a type to the variable 'stypy_return_type' (line 2161)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2161, 8), 'stypy_return_type', islower_call_result_3374)
        
        # ################# End of 'islower(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'islower' in the type store
        # Getting the type of 'stypy_return_type' (line 2150)
        stypy_return_type_3375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2150, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3375)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'islower'
        return stypy_return_type_3375


    @norecursion
    def isspace(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'isspace'
        module_type_store = module_type_store.open_function_context('isspace', 2163, 4, False)
        # Assigning a type to the variable 'self' (line 2164)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2164, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        chararray.isspace.__dict__.__setitem__('stypy_localization', localization)
        chararray.isspace.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        chararray.isspace.__dict__.__setitem__('stypy_type_store', module_type_store)
        chararray.isspace.__dict__.__setitem__('stypy_function_name', 'chararray.isspace')
        chararray.isspace.__dict__.__setitem__('stypy_param_names_list', [])
        chararray.isspace.__dict__.__setitem__('stypy_varargs_param_name', None)
        chararray.isspace.__dict__.__setitem__('stypy_kwargs_param_name', None)
        chararray.isspace.__dict__.__setitem__('stypy_call_defaults', defaults)
        chararray.isspace.__dict__.__setitem__('stypy_call_varargs', varargs)
        chararray.isspace.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        chararray.isspace.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'chararray.isspace', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'isspace', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'isspace(...)' code ##################

        str_3376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2173, (-1)), 'str', '\n        Returns true for each element if there are only whitespace\n        characters in the string and there is at least one character,\n        false otherwise.\n\n        See also\n        --------\n        char.isspace\n\n        ')
        
        # Call to isspace(...): (line 2174)
        # Processing the call arguments (line 2174)
        # Getting the type of 'self' (line 2174)
        self_3378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2174, 23), 'self', False)
        # Processing the call keyword arguments (line 2174)
        kwargs_3379 = {}
        # Getting the type of 'isspace' (line 2174)
        isspace_3377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2174, 15), 'isspace', False)
        # Calling isspace(args, kwargs) (line 2174)
        isspace_call_result_3380 = invoke(stypy.reporting.localization.Localization(__file__, 2174, 15), isspace_3377, *[self_3378], **kwargs_3379)
        
        # Assigning a type to the variable 'stypy_return_type' (line 2174)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2174, 8), 'stypy_return_type', isspace_call_result_3380)
        
        # ################# End of 'isspace(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'isspace' in the type store
        # Getting the type of 'stypy_return_type' (line 2163)
        stypy_return_type_3381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2163, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3381)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'isspace'
        return stypy_return_type_3381


    @norecursion
    def istitle(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'istitle'
        module_type_store = module_type_store.open_function_context('istitle', 2176, 4, False)
        # Assigning a type to the variable 'self' (line 2177)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2177, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        chararray.istitle.__dict__.__setitem__('stypy_localization', localization)
        chararray.istitle.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        chararray.istitle.__dict__.__setitem__('stypy_type_store', module_type_store)
        chararray.istitle.__dict__.__setitem__('stypy_function_name', 'chararray.istitle')
        chararray.istitle.__dict__.__setitem__('stypy_param_names_list', [])
        chararray.istitle.__dict__.__setitem__('stypy_varargs_param_name', None)
        chararray.istitle.__dict__.__setitem__('stypy_kwargs_param_name', None)
        chararray.istitle.__dict__.__setitem__('stypy_call_defaults', defaults)
        chararray.istitle.__dict__.__setitem__('stypy_call_varargs', varargs)
        chararray.istitle.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        chararray.istitle.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'chararray.istitle', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'istitle', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'istitle(...)' code ##################

        str_3382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2185, (-1)), 'str', '\n        Returns true for each element if the element is a titlecased\n        string and there is at least one character, false otherwise.\n\n        See also\n        --------\n        char.istitle\n\n        ')
        
        # Call to istitle(...): (line 2186)
        # Processing the call arguments (line 2186)
        # Getting the type of 'self' (line 2186)
        self_3384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2186, 23), 'self', False)
        # Processing the call keyword arguments (line 2186)
        kwargs_3385 = {}
        # Getting the type of 'istitle' (line 2186)
        istitle_3383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2186, 15), 'istitle', False)
        # Calling istitle(args, kwargs) (line 2186)
        istitle_call_result_3386 = invoke(stypy.reporting.localization.Localization(__file__, 2186, 15), istitle_3383, *[self_3384], **kwargs_3385)
        
        # Assigning a type to the variable 'stypy_return_type' (line 2186)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2186, 8), 'stypy_return_type', istitle_call_result_3386)
        
        # ################# End of 'istitle(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'istitle' in the type store
        # Getting the type of 'stypy_return_type' (line 2176)
        stypy_return_type_3387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2176, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3387)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'istitle'
        return stypy_return_type_3387


    @norecursion
    def isupper(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'isupper'
        module_type_store = module_type_store.open_function_context('isupper', 2188, 4, False)
        # Assigning a type to the variable 'self' (line 2189)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2189, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        chararray.isupper.__dict__.__setitem__('stypy_localization', localization)
        chararray.isupper.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        chararray.isupper.__dict__.__setitem__('stypy_type_store', module_type_store)
        chararray.isupper.__dict__.__setitem__('stypy_function_name', 'chararray.isupper')
        chararray.isupper.__dict__.__setitem__('stypy_param_names_list', [])
        chararray.isupper.__dict__.__setitem__('stypy_varargs_param_name', None)
        chararray.isupper.__dict__.__setitem__('stypy_kwargs_param_name', None)
        chararray.isupper.__dict__.__setitem__('stypy_call_defaults', defaults)
        chararray.isupper.__dict__.__setitem__('stypy_call_varargs', varargs)
        chararray.isupper.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        chararray.isupper.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'chararray.isupper', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'isupper', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'isupper(...)' code ##################

        str_3388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2198, (-1)), 'str', '\n        Returns true for each element if all cased characters in the\n        string are uppercase and there is at least one character, false\n        otherwise.\n\n        See also\n        --------\n        char.isupper\n\n        ')
        
        # Call to isupper(...): (line 2199)
        # Processing the call arguments (line 2199)
        # Getting the type of 'self' (line 2199)
        self_3390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2199, 23), 'self', False)
        # Processing the call keyword arguments (line 2199)
        kwargs_3391 = {}
        # Getting the type of 'isupper' (line 2199)
        isupper_3389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2199, 15), 'isupper', False)
        # Calling isupper(args, kwargs) (line 2199)
        isupper_call_result_3392 = invoke(stypy.reporting.localization.Localization(__file__, 2199, 15), isupper_3389, *[self_3390], **kwargs_3391)
        
        # Assigning a type to the variable 'stypy_return_type' (line 2199)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2199, 8), 'stypy_return_type', isupper_call_result_3392)
        
        # ################# End of 'isupper(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'isupper' in the type store
        # Getting the type of 'stypy_return_type' (line 2188)
        stypy_return_type_3393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2188, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3393)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'isupper'
        return stypy_return_type_3393


    @norecursion
    def join(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'join'
        module_type_store = module_type_store.open_function_context('join', 2201, 4, False)
        # Assigning a type to the variable 'self' (line 2202)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2202, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        chararray.join.__dict__.__setitem__('stypy_localization', localization)
        chararray.join.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        chararray.join.__dict__.__setitem__('stypy_type_store', module_type_store)
        chararray.join.__dict__.__setitem__('stypy_function_name', 'chararray.join')
        chararray.join.__dict__.__setitem__('stypy_param_names_list', ['seq'])
        chararray.join.__dict__.__setitem__('stypy_varargs_param_name', None)
        chararray.join.__dict__.__setitem__('stypy_kwargs_param_name', None)
        chararray.join.__dict__.__setitem__('stypy_call_defaults', defaults)
        chararray.join.__dict__.__setitem__('stypy_call_varargs', varargs)
        chararray.join.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        chararray.join.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'chararray.join', ['seq'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'join', localization, ['seq'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'join(...)' code ##################

        str_3394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2210, (-1)), 'str', '\n        Return a string which is the concatenation of the strings in the\n        sequence `seq`.\n\n        See also\n        --------\n        char.join\n\n        ')
        
        # Call to join(...): (line 2211)
        # Processing the call arguments (line 2211)
        # Getting the type of 'self' (line 2211)
        self_3396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2211, 20), 'self', False)
        # Getting the type of 'seq' (line 2211)
        seq_3397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2211, 26), 'seq', False)
        # Processing the call keyword arguments (line 2211)
        kwargs_3398 = {}
        # Getting the type of 'join' (line 2211)
        join_3395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2211, 15), 'join', False)
        # Calling join(args, kwargs) (line 2211)
        join_call_result_3399 = invoke(stypy.reporting.localization.Localization(__file__, 2211, 15), join_3395, *[self_3396, seq_3397], **kwargs_3398)
        
        # Assigning a type to the variable 'stypy_return_type' (line 2211)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2211, 8), 'stypy_return_type', join_call_result_3399)
        
        # ################# End of 'join(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'join' in the type store
        # Getting the type of 'stypy_return_type' (line 2201)
        stypy_return_type_3400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2201, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3400)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'join'
        return stypy_return_type_3400


    @norecursion
    def ljust(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        str_3401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2213, 36), 'str', ' ')
        defaults = [str_3401]
        # Create a new context for function 'ljust'
        module_type_store = module_type_store.open_function_context('ljust', 2213, 4, False)
        # Assigning a type to the variable 'self' (line 2214)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2214, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        chararray.ljust.__dict__.__setitem__('stypy_localization', localization)
        chararray.ljust.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        chararray.ljust.__dict__.__setitem__('stypy_type_store', module_type_store)
        chararray.ljust.__dict__.__setitem__('stypy_function_name', 'chararray.ljust')
        chararray.ljust.__dict__.__setitem__('stypy_param_names_list', ['width', 'fillchar'])
        chararray.ljust.__dict__.__setitem__('stypy_varargs_param_name', None)
        chararray.ljust.__dict__.__setitem__('stypy_kwargs_param_name', None)
        chararray.ljust.__dict__.__setitem__('stypy_call_defaults', defaults)
        chararray.ljust.__dict__.__setitem__('stypy_call_varargs', varargs)
        chararray.ljust.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        chararray.ljust.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'chararray.ljust', ['width', 'fillchar'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'ljust', localization, ['width', 'fillchar'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'ljust(...)' code ##################

        str_3402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2222, (-1)), 'str', '\n        Return an array with the elements of `self` left-justified in a\n        string of length `width`.\n\n        See also\n        --------\n        char.ljust\n\n        ')
        
        # Call to asarray(...): (line 2223)
        # Processing the call arguments (line 2223)
        
        # Call to ljust(...): (line 2223)
        # Processing the call arguments (line 2223)
        # Getting the type of 'self' (line 2223)
        self_3405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2223, 29), 'self', False)
        # Getting the type of 'width' (line 2223)
        width_3406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2223, 35), 'width', False)
        # Getting the type of 'fillchar' (line 2223)
        fillchar_3407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2223, 42), 'fillchar', False)
        # Processing the call keyword arguments (line 2223)
        kwargs_3408 = {}
        # Getting the type of 'ljust' (line 2223)
        ljust_3404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2223, 23), 'ljust', False)
        # Calling ljust(args, kwargs) (line 2223)
        ljust_call_result_3409 = invoke(stypy.reporting.localization.Localization(__file__, 2223, 23), ljust_3404, *[self_3405, width_3406, fillchar_3407], **kwargs_3408)
        
        # Processing the call keyword arguments (line 2223)
        kwargs_3410 = {}
        # Getting the type of 'asarray' (line 2223)
        asarray_3403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2223, 15), 'asarray', False)
        # Calling asarray(args, kwargs) (line 2223)
        asarray_call_result_3411 = invoke(stypy.reporting.localization.Localization(__file__, 2223, 15), asarray_3403, *[ljust_call_result_3409], **kwargs_3410)
        
        # Assigning a type to the variable 'stypy_return_type' (line 2223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2223, 8), 'stypy_return_type', asarray_call_result_3411)
        
        # ################# End of 'ljust(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'ljust' in the type store
        # Getting the type of 'stypy_return_type' (line 2213)
        stypy_return_type_3412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2213, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3412)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'ljust'
        return stypy_return_type_3412


    @norecursion
    def lower(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'lower'
        module_type_store = module_type_store.open_function_context('lower', 2225, 4, False)
        # Assigning a type to the variable 'self' (line 2226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2226, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        chararray.lower.__dict__.__setitem__('stypy_localization', localization)
        chararray.lower.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        chararray.lower.__dict__.__setitem__('stypy_type_store', module_type_store)
        chararray.lower.__dict__.__setitem__('stypy_function_name', 'chararray.lower')
        chararray.lower.__dict__.__setitem__('stypy_param_names_list', [])
        chararray.lower.__dict__.__setitem__('stypy_varargs_param_name', None)
        chararray.lower.__dict__.__setitem__('stypy_kwargs_param_name', None)
        chararray.lower.__dict__.__setitem__('stypy_call_defaults', defaults)
        chararray.lower.__dict__.__setitem__('stypy_call_varargs', varargs)
        chararray.lower.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        chararray.lower.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'chararray.lower', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'lower', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'lower(...)' code ##################

        str_3413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2234, (-1)), 'str', '\n        Return an array with the elements of `self` converted to\n        lowercase.\n\n        See also\n        --------\n        char.lower\n\n        ')
        
        # Call to asarray(...): (line 2235)
        # Processing the call arguments (line 2235)
        
        # Call to lower(...): (line 2235)
        # Processing the call arguments (line 2235)
        # Getting the type of 'self' (line 2235)
        self_3416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2235, 29), 'self', False)
        # Processing the call keyword arguments (line 2235)
        kwargs_3417 = {}
        # Getting the type of 'lower' (line 2235)
        lower_3415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2235, 23), 'lower', False)
        # Calling lower(args, kwargs) (line 2235)
        lower_call_result_3418 = invoke(stypy.reporting.localization.Localization(__file__, 2235, 23), lower_3415, *[self_3416], **kwargs_3417)
        
        # Processing the call keyword arguments (line 2235)
        kwargs_3419 = {}
        # Getting the type of 'asarray' (line 2235)
        asarray_3414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2235, 15), 'asarray', False)
        # Calling asarray(args, kwargs) (line 2235)
        asarray_call_result_3420 = invoke(stypy.reporting.localization.Localization(__file__, 2235, 15), asarray_3414, *[lower_call_result_3418], **kwargs_3419)
        
        # Assigning a type to the variable 'stypy_return_type' (line 2235)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2235, 8), 'stypy_return_type', asarray_call_result_3420)
        
        # ################# End of 'lower(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'lower' in the type store
        # Getting the type of 'stypy_return_type' (line 2225)
        stypy_return_type_3421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2225, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3421)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'lower'
        return stypy_return_type_3421


    @norecursion
    def lstrip(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 2237)
        None_3422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2237, 27), 'None')
        defaults = [None_3422]
        # Create a new context for function 'lstrip'
        module_type_store = module_type_store.open_function_context('lstrip', 2237, 4, False)
        # Assigning a type to the variable 'self' (line 2238)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2238, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        chararray.lstrip.__dict__.__setitem__('stypy_localization', localization)
        chararray.lstrip.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        chararray.lstrip.__dict__.__setitem__('stypy_type_store', module_type_store)
        chararray.lstrip.__dict__.__setitem__('stypy_function_name', 'chararray.lstrip')
        chararray.lstrip.__dict__.__setitem__('stypy_param_names_list', ['chars'])
        chararray.lstrip.__dict__.__setitem__('stypy_varargs_param_name', None)
        chararray.lstrip.__dict__.__setitem__('stypy_kwargs_param_name', None)
        chararray.lstrip.__dict__.__setitem__('stypy_call_defaults', defaults)
        chararray.lstrip.__dict__.__setitem__('stypy_call_varargs', varargs)
        chararray.lstrip.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        chararray.lstrip.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'chararray.lstrip', ['chars'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'lstrip', localization, ['chars'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'lstrip(...)' code ##################

        str_3423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2246, (-1)), 'str', '\n        For each element in `self`, return a copy with the leading characters\n        removed.\n\n        See also\n        --------\n        char.lstrip\n\n        ')
        
        # Call to asarray(...): (line 2247)
        # Processing the call arguments (line 2247)
        
        # Call to lstrip(...): (line 2247)
        # Processing the call arguments (line 2247)
        # Getting the type of 'self' (line 2247)
        self_3426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2247, 30), 'self', False)
        # Getting the type of 'chars' (line 2247)
        chars_3427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2247, 36), 'chars', False)
        # Processing the call keyword arguments (line 2247)
        kwargs_3428 = {}
        # Getting the type of 'lstrip' (line 2247)
        lstrip_3425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2247, 23), 'lstrip', False)
        # Calling lstrip(args, kwargs) (line 2247)
        lstrip_call_result_3429 = invoke(stypy.reporting.localization.Localization(__file__, 2247, 23), lstrip_3425, *[self_3426, chars_3427], **kwargs_3428)
        
        # Processing the call keyword arguments (line 2247)
        kwargs_3430 = {}
        # Getting the type of 'asarray' (line 2247)
        asarray_3424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2247, 15), 'asarray', False)
        # Calling asarray(args, kwargs) (line 2247)
        asarray_call_result_3431 = invoke(stypy.reporting.localization.Localization(__file__, 2247, 15), asarray_3424, *[lstrip_call_result_3429], **kwargs_3430)
        
        # Assigning a type to the variable 'stypy_return_type' (line 2247)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2247, 8), 'stypy_return_type', asarray_call_result_3431)
        
        # ################# End of 'lstrip(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'lstrip' in the type store
        # Getting the type of 'stypy_return_type' (line 2237)
        stypy_return_type_3432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2237, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3432)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'lstrip'
        return stypy_return_type_3432


    @norecursion
    def partition(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'partition'
        module_type_store = module_type_store.open_function_context('partition', 2249, 4, False)
        # Assigning a type to the variable 'self' (line 2250)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2250, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        chararray.partition.__dict__.__setitem__('stypy_localization', localization)
        chararray.partition.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        chararray.partition.__dict__.__setitem__('stypy_type_store', module_type_store)
        chararray.partition.__dict__.__setitem__('stypy_function_name', 'chararray.partition')
        chararray.partition.__dict__.__setitem__('stypy_param_names_list', ['sep'])
        chararray.partition.__dict__.__setitem__('stypy_varargs_param_name', None)
        chararray.partition.__dict__.__setitem__('stypy_kwargs_param_name', None)
        chararray.partition.__dict__.__setitem__('stypy_call_defaults', defaults)
        chararray.partition.__dict__.__setitem__('stypy_call_varargs', varargs)
        chararray.partition.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        chararray.partition.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'chararray.partition', ['sep'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'partition', localization, ['sep'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'partition(...)' code ##################

        str_3433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2256, (-1)), 'str', '\n        Partition each element in `self` around `sep`.\n\n        See also\n        --------\n        partition\n        ')
        
        # Call to asarray(...): (line 2257)
        # Processing the call arguments (line 2257)
        
        # Call to partition(...): (line 2257)
        # Processing the call arguments (line 2257)
        # Getting the type of 'self' (line 2257)
        self_3436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2257, 33), 'self', False)
        # Getting the type of 'sep' (line 2257)
        sep_3437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2257, 39), 'sep', False)
        # Processing the call keyword arguments (line 2257)
        kwargs_3438 = {}
        # Getting the type of 'partition' (line 2257)
        partition_3435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2257, 23), 'partition', False)
        # Calling partition(args, kwargs) (line 2257)
        partition_call_result_3439 = invoke(stypy.reporting.localization.Localization(__file__, 2257, 23), partition_3435, *[self_3436, sep_3437], **kwargs_3438)
        
        # Processing the call keyword arguments (line 2257)
        kwargs_3440 = {}
        # Getting the type of 'asarray' (line 2257)
        asarray_3434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2257, 15), 'asarray', False)
        # Calling asarray(args, kwargs) (line 2257)
        asarray_call_result_3441 = invoke(stypy.reporting.localization.Localization(__file__, 2257, 15), asarray_3434, *[partition_call_result_3439], **kwargs_3440)
        
        # Assigning a type to the variable 'stypy_return_type' (line 2257)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2257, 8), 'stypy_return_type', asarray_call_result_3441)
        
        # ################# End of 'partition(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'partition' in the type store
        # Getting the type of 'stypy_return_type' (line 2249)
        stypy_return_type_3442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2249, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3442)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'partition'
        return stypy_return_type_3442


    @norecursion
    def replace(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 2259)
        None_3443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2259, 38), 'None')
        defaults = [None_3443]
        # Create a new context for function 'replace'
        module_type_store = module_type_store.open_function_context('replace', 2259, 4, False)
        # Assigning a type to the variable 'self' (line 2260)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2260, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        chararray.replace.__dict__.__setitem__('stypy_localization', localization)
        chararray.replace.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        chararray.replace.__dict__.__setitem__('stypy_type_store', module_type_store)
        chararray.replace.__dict__.__setitem__('stypy_function_name', 'chararray.replace')
        chararray.replace.__dict__.__setitem__('stypy_param_names_list', ['old', 'new', 'count'])
        chararray.replace.__dict__.__setitem__('stypy_varargs_param_name', None)
        chararray.replace.__dict__.__setitem__('stypy_kwargs_param_name', None)
        chararray.replace.__dict__.__setitem__('stypy_call_defaults', defaults)
        chararray.replace.__dict__.__setitem__('stypy_call_varargs', varargs)
        chararray.replace.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        chararray.replace.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'chararray.replace', ['old', 'new', 'count'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'replace', localization, ['old', 'new', 'count'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'replace(...)' code ##################

        str_3444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2268, (-1)), 'str', '\n        For each element in `self`, return a copy of the string with all\n        occurrences of substring `old` replaced by `new`.\n\n        See also\n        --------\n        char.replace\n\n        ')
        
        # Call to asarray(...): (line 2269)
        # Processing the call arguments (line 2269)
        
        # Call to replace(...): (line 2269)
        # Processing the call arguments (line 2269)
        # Getting the type of 'self' (line 2269)
        self_3447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2269, 31), 'self', False)
        # Getting the type of 'old' (line 2269)
        old_3448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2269, 37), 'old', False)
        # Getting the type of 'new' (line 2269)
        new_3449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2269, 42), 'new', False)
        # Getting the type of 'count' (line 2269)
        count_3450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2269, 47), 'count', False)
        # Processing the call keyword arguments (line 2269)
        kwargs_3451 = {}
        # Getting the type of 'replace' (line 2269)
        replace_3446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2269, 23), 'replace', False)
        # Calling replace(args, kwargs) (line 2269)
        replace_call_result_3452 = invoke(stypy.reporting.localization.Localization(__file__, 2269, 23), replace_3446, *[self_3447, old_3448, new_3449, count_3450], **kwargs_3451)
        
        # Processing the call keyword arguments (line 2269)
        kwargs_3453 = {}
        # Getting the type of 'asarray' (line 2269)
        asarray_3445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2269, 15), 'asarray', False)
        # Calling asarray(args, kwargs) (line 2269)
        asarray_call_result_3454 = invoke(stypy.reporting.localization.Localization(__file__, 2269, 15), asarray_3445, *[replace_call_result_3452], **kwargs_3453)
        
        # Assigning a type to the variable 'stypy_return_type' (line 2269)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2269, 8), 'stypy_return_type', asarray_call_result_3454)
        
        # ################# End of 'replace(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'replace' in the type store
        # Getting the type of 'stypy_return_type' (line 2259)
        stypy_return_type_3455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2259, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3455)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'replace'
        return stypy_return_type_3455


    @norecursion
    def rfind(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_3456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2271, 31), 'int')
        # Getting the type of 'None' (line 2271)
        None_3457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2271, 38), 'None')
        defaults = [int_3456, None_3457]
        # Create a new context for function 'rfind'
        module_type_store = module_type_store.open_function_context('rfind', 2271, 4, False)
        # Assigning a type to the variable 'self' (line 2272)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2272, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        chararray.rfind.__dict__.__setitem__('stypy_localization', localization)
        chararray.rfind.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        chararray.rfind.__dict__.__setitem__('stypy_type_store', module_type_store)
        chararray.rfind.__dict__.__setitem__('stypy_function_name', 'chararray.rfind')
        chararray.rfind.__dict__.__setitem__('stypy_param_names_list', ['sub', 'start', 'end'])
        chararray.rfind.__dict__.__setitem__('stypy_varargs_param_name', None)
        chararray.rfind.__dict__.__setitem__('stypy_kwargs_param_name', None)
        chararray.rfind.__dict__.__setitem__('stypy_call_defaults', defaults)
        chararray.rfind.__dict__.__setitem__('stypy_call_varargs', varargs)
        chararray.rfind.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        chararray.rfind.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'chararray.rfind', ['sub', 'start', 'end'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'rfind', localization, ['sub', 'start', 'end'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'rfind(...)' code ##################

        str_3458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2281, (-1)), 'str', '\n        For each element in `self`, return the highest index in the string\n        where substring `sub` is found, such that `sub` is contained\n        within [`start`, `end`].\n\n        See also\n        --------\n        char.rfind\n\n        ')
        
        # Call to rfind(...): (line 2282)
        # Processing the call arguments (line 2282)
        # Getting the type of 'self' (line 2282)
        self_3460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2282, 21), 'self', False)
        # Getting the type of 'sub' (line 2282)
        sub_3461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2282, 27), 'sub', False)
        # Getting the type of 'start' (line 2282)
        start_3462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2282, 32), 'start', False)
        # Getting the type of 'end' (line 2282)
        end_3463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2282, 39), 'end', False)
        # Processing the call keyword arguments (line 2282)
        kwargs_3464 = {}
        # Getting the type of 'rfind' (line 2282)
        rfind_3459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2282, 15), 'rfind', False)
        # Calling rfind(args, kwargs) (line 2282)
        rfind_call_result_3465 = invoke(stypy.reporting.localization.Localization(__file__, 2282, 15), rfind_3459, *[self_3460, sub_3461, start_3462, end_3463], **kwargs_3464)
        
        # Assigning a type to the variable 'stypy_return_type' (line 2282)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2282, 8), 'stypy_return_type', rfind_call_result_3465)
        
        # ################# End of 'rfind(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'rfind' in the type store
        # Getting the type of 'stypy_return_type' (line 2271)
        stypy_return_type_3466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2271, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3466)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'rfind'
        return stypy_return_type_3466


    @norecursion
    def rindex(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_3467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2284, 32), 'int')
        # Getting the type of 'None' (line 2284)
        None_3468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2284, 39), 'None')
        defaults = [int_3467, None_3468]
        # Create a new context for function 'rindex'
        module_type_store = module_type_store.open_function_context('rindex', 2284, 4, False)
        # Assigning a type to the variable 'self' (line 2285)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2285, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        chararray.rindex.__dict__.__setitem__('stypy_localization', localization)
        chararray.rindex.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        chararray.rindex.__dict__.__setitem__('stypy_type_store', module_type_store)
        chararray.rindex.__dict__.__setitem__('stypy_function_name', 'chararray.rindex')
        chararray.rindex.__dict__.__setitem__('stypy_param_names_list', ['sub', 'start', 'end'])
        chararray.rindex.__dict__.__setitem__('stypy_varargs_param_name', None)
        chararray.rindex.__dict__.__setitem__('stypy_kwargs_param_name', None)
        chararray.rindex.__dict__.__setitem__('stypy_call_defaults', defaults)
        chararray.rindex.__dict__.__setitem__('stypy_call_varargs', varargs)
        chararray.rindex.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        chararray.rindex.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'chararray.rindex', ['sub', 'start', 'end'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'rindex', localization, ['sub', 'start', 'end'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'rindex(...)' code ##################

        str_3469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2293, (-1)), 'str', '\n        Like `rfind`, but raises `ValueError` when the substring `sub` is\n        not found.\n\n        See also\n        --------\n        char.rindex\n\n        ')
        
        # Call to rindex(...): (line 2294)
        # Processing the call arguments (line 2294)
        # Getting the type of 'self' (line 2294)
        self_3471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2294, 22), 'self', False)
        # Getting the type of 'sub' (line 2294)
        sub_3472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2294, 28), 'sub', False)
        # Getting the type of 'start' (line 2294)
        start_3473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2294, 33), 'start', False)
        # Getting the type of 'end' (line 2294)
        end_3474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2294, 40), 'end', False)
        # Processing the call keyword arguments (line 2294)
        kwargs_3475 = {}
        # Getting the type of 'rindex' (line 2294)
        rindex_3470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2294, 15), 'rindex', False)
        # Calling rindex(args, kwargs) (line 2294)
        rindex_call_result_3476 = invoke(stypy.reporting.localization.Localization(__file__, 2294, 15), rindex_3470, *[self_3471, sub_3472, start_3473, end_3474], **kwargs_3475)
        
        # Assigning a type to the variable 'stypy_return_type' (line 2294)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2294, 8), 'stypy_return_type', rindex_call_result_3476)
        
        # ################# End of 'rindex(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'rindex' in the type store
        # Getting the type of 'stypy_return_type' (line 2284)
        stypy_return_type_3477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2284, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3477)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'rindex'
        return stypy_return_type_3477


    @norecursion
    def rjust(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        str_3478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2296, 36), 'str', ' ')
        defaults = [str_3478]
        # Create a new context for function 'rjust'
        module_type_store = module_type_store.open_function_context('rjust', 2296, 4, False)
        # Assigning a type to the variable 'self' (line 2297)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2297, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        chararray.rjust.__dict__.__setitem__('stypy_localization', localization)
        chararray.rjust.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        chararray.rjust.__dict__.__setitem__('stypy_type_store', module_type_store)
        chararray.rjust.__dict__.__setitem__('stypy_function_name', 'chararray.rjust')
        chararray.rjust.__dict__.__setitem__('stypy_param_names_list', ['width', 'fillchar'])
        chararray.rjust.__dict__.__setitem__('stypy_varargs_param_name', None)
        chararray.rjust.__dict__.__setitem__('stypy_kwargs_param_name', None)
        chararray.rjust.__dict__.__setitem__('stypy_call_defaults', defaults)
        chararray.rjust.__dict__.__setitem__('stypy_call_varargs', varargs)
        chararray.rjust.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        chararray.rjust.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'chararray.rjust', ['width', 'fillchar'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'rjust', localization, ['width', 'fillchar'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'rjust(...)' code ##################

        str_3479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2305, (-1)), 'str', '\n        Return an array with the elements of `self`\n        right-justified in a string of length `width`.\n\n        See also\n        --------\n        char.rjust\n\n        ')
        
        # Call to asarray(...): (line 2306)
        # Processing the call arguments (line 2306)
        
        # Call to rjust(...): (line 2306)
        # Processing the call arguments (line 2306)
        # Getting the type of 'self' (line 2306)
        self_3482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2306, 29), 'self', False)
        # Getting the type of 'width' (line 2306)
        width_3483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2306, 35), 'width', False)
        # Getting the type of 'fillchar' (line 2306)
        fillchar_3484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2306, 42), 'fillchar', False)
        # Processing the call keyword arguments (line 2306)
        kwargs_3485 = {}
        # Getting the type of 'rjust' (line 2306)
        rjust_3481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2306, 23), 'rjust', False)
        # Calling rjust(args, kwargs) (line 2306)
        rjust_call_result_3486 = invoke(stypy.reporting.localization.Localization(__file__, 2306, 23), rjust_3481, *[self_3482, width_3483, fillchar_3484], **kwargs_3485)
        
        # Processing the call keyword arguments (line 2306)
        kwargs_3487 = {}
        # Getting the type of 'asarray' (line 2306)
        asarray_3480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2306, 15), 'asarray', False)
        # Calling asarray(args, kwargs) (line 2306)
        asarray_call_result_3488 = invoke(stypy.reporting.localization.Localization(__file__, 2306, 15), asarray_3480, *[rjust_call_result_3486], **kwargs_3487)
        
        # Assigning a type to the variable 'stypy_return_type' (line 2306)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2306, 8), 'stypy_return_type', asarray_call_result_3488)
        
        # ################# End of 'rjust(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'rjust' in the type store
        # Getting the type of 'stypy_return_type' (line 2296)
        stypy_return_type_3489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2296, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3489)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'rjust'
        return stypy_return_type_3489


    @norecursion
    def rpartition(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'rpartition'
        module_type_store = module_type_store.open_function_context('rpartition', 2308, 4, False)
        # Assigning a type to the variable 'self' (line 2309)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2309, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        chararray.rpartition.__dict__.__setitem__('stypy_localization', localization)
        chararray.rpartition.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        chararray.rpartition.__dict__.__setitem__('stypy_type_store', module_type_store)
        chararray.rpartition.__dict__.__setitem__('stypy_function_name', 'chararray.rpartition')
        chararray.rpartition.__dict__.__setitem__('stypy_param_names_list', ['sep'])
        chararray.rpartition.__dict__.__setitem__('stypy_varargs_param_name', None)
        chararray.rpartition.__dict__.__setitem__('stypy_kwargs_param_name', None)
        chararray.rpartition.__dict__.__setitem__('stypy_call_defaults', defaults)
        chararray.rpartition.__dict__.__setitem__('stypy_call_varargs', varargs)
        chararray.rpartition.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        chararray.rpartition.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'chararray.rpartition', ['sep'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'rpartition', localization, ['sep'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'rpartition(...)' code ##################

        str_3490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2315, (-1)), 'str', '\n        Partition each element in `self` around `sep`.\n\n        See also\n        --------\n        rpartition\n        ')
        
        # Call to asarray(...): (line 2316)
        # Processing the call arguments (line 2316)
        
        # Call to rpartition(...): (line 2316)
        # Processing the call arguments (line 2316)
        # Getting the type of 'self' (line 2316)
        self_3493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2316, 34), 'self', False)
        # Getting the type of 'sep' (line 2316)
        sep_3494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2316, 40), 'sep', False)
        # Processing the call keyword arguments (line 2316)
        kwargs_3495 = {}
        # Getting the type of 'rpartition' (line 2316)
        rpartition_3492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2316, 23), 'rpartition', False)
        # Calling rpartition(args, kwargs) (line 2316)
        rpartition_call_result_3496 = invoke(stypy.reporting.localization.Localization(__file__, 2316, 23), rpartition_3492, *[self_3493, sep_3494], **kwargs_3495)
        
        # Processing the call keyword arguments (line 2316)
        kwargs_3497 = {}
        # Getting the type of 'asarray' (line 2316)
        asarray_3491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2316, 15), 'asarray', False)
        # Calling asarray(args, kwargs) (line 2316)
        asarray_call_result_3498 = invoke(stypy.reporting.localization.Localization(__file__, 2316, 15), asarray_3491, *[rpartition_call_result_3496], **kwargs_3497)
        
        # Assigning a type to the variable 'stypy_return_type' (line 2316)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2316, 8), 'stypy_return_type', asarray_call_result_3498)
        
        # ################# End of 'rpartition(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'rpartition' in the type store
        # Getting the type of 'stypy_return_type' (line 2308)
        stypy_return_type_3499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2308, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3499)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'rpartition'
        return stypy_return_type_3499


    @norecursion
    def rsplit(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 2318)
        None_3500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2318, 25), 'None')
        # Getting the type of 'None' (line 2318)
        None_3501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2318, 40), 'None')
        defaults = [None_3500, None_3501]
        # Create a new context for function 'rsplit'
        module_type_store = module_type_store.open_function_context('rsplit', 2318, 4, False)
        # Assigning a type to the variable 'self' (line 2319)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2319, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        chararray.rsplit.__dict__.__setitem__('stypy_localization', localization)
        chararray.rsplit.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        chararray.rsplit.__dict__.__setitem__('stypy_type_store', module_type_store)
        chararray.rsplit.__dict__.__setitem__('stypy_function_name', 'chararray.rsplit')
        chararray.rsplit.__dict__.__setitem__('stypy_param_names_list', ['sep', 'maxsplit'])
        chararray.rsplit.__dict__.__setitem__('stypy_varargs_param_name', None)
        chararray.rsplit.__dict__.__setitem__('stypy_kwargs_param_name', None)
        chararray.rsplit.__dict__.__setitem__('stypy_call_defaults', defaults)
        chararray.rsplit.__dict__.__setitem__('stypy_call_varargs', varargs)
        chararray.rsplit.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        chararray.rsplit.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'chararray.rsplit', ['sep', 'maxsplit'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'rsplit', localization, ['sep', 'maxsplit'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'rsplit(...)' code ##################

        str_3502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2327, (-1)), 'str', '\n        For each element in `self`, return a list of the words in\n        the string, using `sep` as the delimiter string.\n\n        See also\n        --------\n        char.rsplit\n\n        ')
        
        # Call to rsplit(...): (line 2328)
        # Processing the call arguments (line 2328)
        # Getting the type of 'self' (line 2328)
        self_3504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2328, 22), 'self', False)
        # Getting the type of 'sep' (line 2328)
        sep_3505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2328, 28), 'sep', False)
        # Getting the type of 'maxsplit' (line 2328)
        maxsplit_3506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2328, 33), 'maxsplit', False)
        # Processing the call keyword arguments (line 2328)
        kwargs_3507 = {}
        # Getting the type of 'rsplit' (line 2328)
        rsplit_3503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2328, 15), 'rsplit', False)
        # Calling rsplit(args, kwargs) (line 2328)
        rsplit_call_result_3508 = invoke(stypy.reporting.localization.Localization(__file__, 2328, 15), rsplit_3503, *[self_3504, sep_3505, maxsplit_3506], **kwargs_3507)
        
        # Assigning a type to the variable 'stypy_return_type' (line 2328)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2328, 8), 'stypy_return_type', rsplit_call_result_3508)
        
        # ################# End of 'rsplit(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'rsplit' in the type store
        # Getting the type of 'stypy_return_type' (line 2318)
        stypy_return_type_3509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2318, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3509)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'rsplit'
        return stypy_return_type_3509


    @norecursion
    def rstrip(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 2330)
        None_3510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2330, 27), 'None')
        defaults = [None_3510]
        # Create a new context for function 'rstrip'
        module_type_store = module_type_store.open_function_context('rstrip', 2330, 4, False)
        # Assigning a type to the variable 'self' (line 2331)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2331, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        chararray.rstrip.__dict__.__setitem__('stypy_localization', localization)
        chararray.rstrip.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        chararray.rstrip.__dict__.__setitem__('stypy_type_store', module_type_store)
        chararray.rstrip.__dict__.__setitem__('stypy_function_name', 'chararray.rstrip')
        chararray.rstrip.__dict__.__setitem__('stypy_param_names_list', ['chars'])
        chararray.rstrip.__dict__.__setitem__('stypy_varargs_param_name', None)
        chararray.rstrip.__dict__.__setitem__('stypy_kwargs_param_name', None)
        chararray.rstrip.__dict__.__setitem__('stypy_call_defaults', defaults)
        chararray.rstrip.__dict__.__setitem__('stypy_call_varargs', varargs)
        chararray.rstrip.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        chararray.rstrip.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'chararray.rstrip', ['chars'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'rstrip', localization, ['chars'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'rstrip(...)' code ##################

        str_3511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2339, (-1)), 'str', '\n        For each element in `self`, return a copy with the trailing\n        characters removed.\n\n        See also\n        --------\n        char.rstrip\n\n        ')
        
        # Call to asarray(...): (line 2340)
        # Processing the call arguments (line 2340)
        
        # Call to rstrip(...): (line 2340)
        # Processing the call arguments (line 2340)
        # Getting the type of 'self' (line 2340)
        self_3514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2340, 30), 'self', False)
        # Getting the type of 'chars' (line 2340)
        chars_3515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2340, 36), 'chars', False)
        # Processing the call keyword arguments (line 2340)
        kwargs_3516 = {}
        # Getting the type of 'rstrip' (line 2340)
        rstrip_3513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2340, 23), 'rstrip', False)
        # Calling rstrip(args, kwargs) (line 2340)
        rstrip_call_result_3517 = invoke(stypy.reporting.localization.Localization(__file__, 2340, 23), rstrip_3513, *[self_3514, chars_3515], **kwargs_3516)
        
        # Processing the call keyword arguments (line 2340)
        kwargs_3518 = {}
        # Getting the type of 'asarray' (line 2340)
        asarray_3512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2340, 15), 'asarray', False)
        # Calling asarray(args, kwargs) (line 2340)
        asarray_call_result_3519 = invoke(stypy.reporting.localization.Localization(__file__, 2340, 15), asarray_3512, *[rstrip_call_result_3517], **kwargs_3518)
        
        # Assigning a type to the variable 'stypy_return_type' (line 2340)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2340, 8), 'stypy_return_type', asarray_call_result_3519)
        
        # ################# End of 'rstrip(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'rstrip' in the type store
        # Getting the type of 'stypy_return_type' (line 2330)
        stypy_return_type_3520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2330, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3520)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'rstrip'
        return stypy_return_type_3520


    @norecursion
    def split(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 2342)
        None_3521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2342, 24), 'None')
        # Getting the type of 'None' (line 2342)
        None_3522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2342, 39), 'None')
        defaults = [None_3521, None_3522]
        # Create a new context for function 'split'
        module_type_store = module_type_store.open_function_context('split', 2342, 4, False)
        # Assigning a type to the variable 'self' (line 2343)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2343, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        chararray.split.__dict__.__setitem__('stypy_localization', localization)
        chararray.split.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        chararray.split.__dict__.__setitem__('stypy_type_store', module_type_store)
        chararray.split.__dict__.__setitem__('stypy_function_name', 'chararray.split')
        chararray.split.__dict__.__setitem__('stypy_param_names_list', ['sep', 'maxsplit'])
        chararray.split.__dict__.__setitem__('stypy_varargs_param_name', None)
        chararray.split.__dict__.__setitem__('stypy_kwargs_param_name', None)
        chararray.split.__dict__.__setitem__('stypy_call_defaults', defaults)
        chararray.split.__dict__.__setitem__('stypy_call_varargs', varargs)
        chararray.split.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        chararray.split.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'chararray.split', ['sep', 'maxsplit'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'split', localization, ['sep', 'maxsplit'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'split(...)' code ##################

        str_3523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2351, (-1)), 'str', '\n        For each element in `self`, return a list of the words in the\n        string, using `sep` as the delimiter string.\n\n        See also\n        --------\n        char.split\n\n        ')
        
        # Call to split(...): (line 2352)
        # Processing the call arguments (line 2352)
        # Getting the type of 'self' (line 2352)
        self_3525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2352, 21), 'self', False)
        # Getting the type of 'sep' (line 2352)
        sep_3526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2352, 27), 'sep', False)
        # Getting the type of 'maxsplit' (line 2352)
        maxsplit_3527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2352, 32), 'maxsplit', False)
        # Processing the call keyword arguments (line 2352)
        kwargs_3528 = {}
        # Getting the type of 'split' (line 2352)
        split_3524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2352, 15), 'split', False)
        # Calling split(args, kwargs) (line 2352)
        split_call_result_3529 = invoke(stypy.reporting.localization.Localization(__file__, 2352, 15), split_3524, *[self_3525, sep_3526, maxsplit_3527], **kwargs_3528)
        
        # Assigning a type to the variable 'stypy_return_type' (line 2352)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2352, 8), 'stypy_return_type', split_call_result_3529)
        
        # ################# End of 'split(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'split' in the type store
        # Getting the type of 'stypy_return_type' (line 2342)
        stypy_return_type_3530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2342, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3530)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'split'
        return stypy_return_type_3530


    @norecursion
    def splitlines(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 2354)
        None_3531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2354, 34), 'None')
        defaults = [None_3531]
        # Create a new context for function 'splitlines'
        module_type_store = module_type_store.open_function_context('splitlines', 2354, 4, False)
        # Assigning a type to the variable 'self' (line 2355)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2355, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        chararray.splitlines.__dict__.__setitem__('stypy_localization', localization)
        chararray.splitlines.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        chararray.splitlines.__dict__.__setitem__('stypy_type_store', module_type_store)
        chararray.splitlines.__dict__.__setitem__('stypy_function_name', 'chararray.splitlines')
        chararray.splitlines.__dict__.__setitem__('stypy_param_names_list', ['keepends'])
        chararray.splitlines.__dict__.__setitem__('stypy_varargs_param_name', None)
        chararray.splitlines.__dict__.__setitem__('stypy_kwargs_param_name', None)
        chararray.splitlines.__dict__.__setitem__('stypy_call_defaults', defaults)
        chararray.splitlines.__dict__.__setitem__('stypy_call_varargs', varargs)
        chararray.splitlines.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        chararray.splitlines.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'chararray.splitlines', ['keepends'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'splitlines', localization, ['keepends'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'splitlines(...)' code ##################

        str_3532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2363, (-1)), 'str', '\n        For each element in `self`, return a list of the lines in the\n        element, breaking at line boundaries.\n\n        See also\n        --------\n        char.splitlines\n\n        ')
        
        # Call to splitlines(...): (line 2364)
        # Processing the call arguments (line 2364)
        # Getting the type of 'self' (line 2364)
        self_3534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2364, 26), 'self', False)
        # Getting the type of 'keepends' (line 2364)
        keepends_3535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2364, 32), 'keepends', False)
        # Processing the call keyword arguments (line 2364)
        kwargs_3536 = {}
        # Getting the type of 'splitlines' (line 2364)
        splitlines_3533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2364, 15), 'splitlines', False)
        # Calling splitlines(args, kwargs) (line 2364)
        splitlines_call_result_3537 = invoke(stypy.reporting.localization.Localization(__file__, 2364, 15), splitlines_3533, *[self_3534, keepends_3535], **kwargs_3536)
        
        # Assigning a type to the variable 'stypy_return_type' (line 2364)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2364, 8), 'stypy_return_type', splitlines_call_result_3537)
        
        # ################# End of 'splitlines(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'splitlines' in the type store
        # Getting the type of 'stypy_return_type' (line 2354)
        stypy_return_type_3538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2354, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3538)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'splitlines'
        return stypy_return_type_3538


    @norecursion
    def startswith(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_3539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2366, 39), 'int')
        # Getting the type of 'None' (line 2366)
        None_3540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2366, 46), 'None')
        defaults = [int_3539, None_3540]
        # Create a new context for function 'startswith'
        module_type_store = module_type_store.open_function_context('startswith', 2366, 4, False)
        # Assigning a type to the variable 'self' (line 2367)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2367, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        chararray.startswith.__dict__.__setitem__('stypy_localization', localization)
        chararray.startswith.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        chararray.startswith.__dict__.__setitem__('stypy_type_store', module_type_store)
        chararray.startswith.__dict__.__setitem__('stypy_function_name', 'chararray.startswith')
        chararray.startswith.__dict__.__setitem__('stypy_param_names_list', ['prefix', 'start', 'end'])
        chararray.startswith.__dict__.__setitem__('stypy_varargs_param_name', None)
        chararray.startswith.__dict__.__setitem__('stypy_kwargs_param_name', None)
        chararray.startswith.__dict__.__setitem__('stypy_call_defaults', defaults)
        chararray.startswith.__dict__.__setitem__('stypy_call_varargs', varargs)
        chararray.startswith.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        chararray.startswith.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'chararray.startswith', ['prefix', 'start', 'end'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'startswith', localization, ['prefix', 'start', 'end'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'startswith(...)' code ##################

        str_3541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2375, (-1)), 'str', '\n        Returns a boolean array which is `True` where the string element\n        in `self` starts with `prefix`, otherwise `False`.\n\n        See also\n        --------\n        char.startswith\n\n        ')
        
        # Call to startswith(...): (line 2376)
        # Processing the call arguments (line 2376)
        # Getting the type of 'self' (line 2376)
        self_3543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2376, 26), 'self', False)
        # Getting the type of 'prefix' (line 2376)
        prefix_3544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2376, 32), 'prefix', False)
        # Getting the type of 'start' (line 2376)
        start_3545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2376, 40), 'start', False)
        # Getting the type of 'end' (line 2376)
        end_3546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2376, 47), 'end', False)
        # Processing the call keyword arguments (line 2376)
        kwargs_3547 = {}
        # Getting the type of 'startswith' (line 2376)
        startswith_3542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2376, 15), 'startswith', False)
        # Calling startswith(args, kwargs) (line 2376)
        startswith_call_result_3548 = invoke(stypy.reporting.localization.Localization(__file__, 2376, 15), startswith_3542, *[self_3543, prefix_3544, start_3545, end_3546], **kwargs_3547)
        
        # Assigning a type to the variable 'stypy_return_type' (line 2376)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2376, 8), 'stypy_return_type', startswith_call_result_3548)
        
        # ################# End of 'startswith(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'startswith' in the type store
        # Getting the type of 'stypy_return_type' (line 2366)
        stypy_return_type_3549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2366, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3549)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'startswith'
        return stypy_return_type_3549


    @norecursion
    def strip(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 2378)
        None_3550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2378, 26), 'None')
        defaults = [None_3550]
        # Create a new context for function 'strip'
        module_type_store = module_type_store.open_function_context('strip', 2378, 4, False)
        # Assigning a type to the variable 'self' (line 2379)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2379, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        chararray.strip.__dict__.__setitem__('stypy_localization', localization)
        chararray.strip.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        chararray.strip.__dict__.__setitem__('stypy_type_store', module_type_store)
        chararray.strip.__dict__.__setitem__('stypy_function_name', 'chararray.strip')
        chararray.strip.__dict__.__setitem__('stypy_param_names_list', ['chars'])
        chararray.strip.__dict__.__setitem__('stypy_varargs_param_name', None)
        chararray.strip.__dict__.__setitem__('stypy_kwargs_param_name', None)
        chararray.strip.__dict__.__setitem__('stypy_call_defaults', defaults)
        chararray.strip.__dict__.__setitem__('stypy_call_varargs', varargs)
        chararray.strip.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        chararray.strip.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'chararray.strip', ['chars'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'strip', localization, ['chars'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'strip(...)' code ##################

        str_3551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2387, (-1)), 'str', '\n        For each element in `self`, return a copy with the leading and\n        trailing characters removed.\n\n        See also\n        --------\n        char.strip\n\n        ')
        
        # Call to asarray(...): (line 2388)
        # Processing the call arguments (line 2388)
        
        # Call to strip(...): (line 2388)
        # Processing the call arguments (line 2388)
        # Getting the type of 'self' (line 2388)
        self_3554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2388, 29), 'self', False)
        # Getting the type of 'chars' (line 2388)
        chars_3555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2388, 35), 'chars', False)
        # Processing the call keyword arguments (line 2388)
        kwargs_3556 = {}
        # Getting the type of 'strip' (line 2388)
        strip_3553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2388, 23), 'strip', False)
        # Calling strip(args, kwargs) (line 2388)
        strip_call_result_3557 = invoke(stypy.reporting.localization.Localization(__file__, 2388, 23), strip_3553, *[self_3554, chars_3555], **kwargs_3556)
        
        # Processing the call keyword arguments (line 2388)
        kwargs_3558 = {}
        # Getting the type of 'asarray' (line 2388)
        asarray_3552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2388, 15), 'asarray', False)
        # Calling asarray(args, kwargs) (line 2388)
        asarray_call_result_3559 = invoke(stypy.reporting.localization.Localization(__file__, 2388, 15), asarray_3552, *[strip_call_result_3557], **kwargs_3558)
        
        # Assigning a type to the variable 'stypy_return_type' (line 2388)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2388, 8), 'stypy_return_type', asarray_call_result_3559)
        
        # ################# End of 'strip(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'strip' in the type store
        # Getting the type of 'stypy_return_type' (line 2378)
        stypy_return_type_3560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2378, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3560)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'strip'
        return stypy_return_type_3560


    @norecursion
    def swapcase(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'swapcase'
        module_type_store = module_type_store.open_function_context('swapcase', 2390, 4, False)
        # Assigning a type to the variable 'self' (line 2391)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2391, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        chararray.swapcase.__dict__.__setitem__('stypy_localization', localization)
        chararray.swapcase.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        chararray.swapcase.__dict__.__setitem__('stypy_type_store', module_type_store)
        chararray.swapcase.__dict__.__setitem__('stypy_function_name', 'chararray.swapcase')
        chararray.swapcase.__dict__.__setitem__('stypy_param_names_list', [])
        chararray.swapcase.__dict__.__setitem__('stypy_varargs_param_name', None)
        chararray.swapcase.__dict__.__setitem__('stypy_kwargs_param_name', None)
        chararray.swapcase.__dict__.__setitem__('stypy_call_defaults', defaults)
        chararray.swapcase.__dict__.__setitem__('stypy_call_varargs', varargs)
        chararray.swapcase.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        chararray.swapcase.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'chararray.swapcase', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'swapcase', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'swapcase(...)' code ##################

        str_3561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2399, (-1)), 'str', '\n        For each element in `self`, return a copy of the string with\n        uppercase characters converted to lowercase and vice versa.\n\n        See also\n        --------\n        char.swapcase\n\n        ')
        
        # Call to asarray(...): (line 2400)
        # Processing the call arguments (line 2400)
        
        # Call to swapcase(...): (line 2400)
        # Processing the call arguments (line 2400)
        # Getting the type of 'self' (line 2400)
        self_3564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2400, 32), 'self', False)
        # Processing the call keyword arguments (line 2400)
        kwargs_3565 = {}
        # Getting the type of 'swapcase' (line 2400)
        swapcase_3563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2400, 23), 'swapcase', False)
        # Calling swapcase(args, kwargs) (line 2400)
        swapcase_call_result_3566 = invoke(stypy.reporting.localization.Localization(__file__, 2400, 23), swapcase_3563, *[self_3564], **kwargs_3565)
        
        # Processing the call keyword arguments (line 2400)
        kwargs_3567 = {}
        # Getting the type of 'asarray' (line 2400)
        asarray_3562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2400, 15), 'asarray', False)
        # Calling asarray(args, kwargs) (line 2400)
        asarray_call_result_3568 = invoke(stypy.reporting.localization.Localization(__file__, 2400, 15), asarray_3562, *[swapcase_call_result_3566], **kwargs_3567)
        
        # Assigning a type to the variable 'stypy_return_type' (line 2400)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2400, 8), 'stypy_return_type', asarray_call_result_3568)
        
        # ################# End of 'swapcase(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'swapcase' in the type store
        # Getting the type of 'stypy_return_type' (line 2390)
        stypy_return_type_3569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2390, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3569)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'swapcase'
        return stypy_return_type_3569


    @norecursion
    def title(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'title'
        module_type_store = module_type_store.open_function_context('title', 2402, 4, False)
        # Assigning a type to the variable 'self' (line 2403)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2403, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        chararray.title.__dict__.__setitem__('stypy_localization', localization)
        chararray.title.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        chararray.title.__dict__.__setitem__('stypy_type_store', module_type_store)
        chararray.title.__dict__.__setitem__('stypy_function_name', 'chararray.title')
        chararray.title.__dict__.__setitem__('stypy_param_names_list', [])
        chararray.title.__dict__.__setitem__('stypy_varargs_param_name', None)
        chararray.title.__dict__.__setitem__('stypy_kwargs_param_name', None)
        chararray.title.__dict__.__setitem__('stypy_call_defaults', defaults)
        chararray.title.__dict__.__setitem__('stypy_call_varargs', varargs)
        chararray.title.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        chararray.title.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'chararray.title', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'title', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'title(...)' code ##################

        str_3570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2412, (-1)), 'str', '\n        For each element in `self`, return a titlecased version of the\n        string: words start with uppercase characters, all remaining cased\n        characters are lowercase.\n\n        See also\n        --------\n        char.title\n\n        ')
        
        # Call to asarray(...): (line 2413)
        # Processing the call arguments (line 2413)
        
        # Call to title(...): (line 2413)
        # Processing the call arguments (line 2413)
        # Getting the type of 'self' (line 2413)
        self_3573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2413, 29), 'self', False)
        # Processing the call keyword arguments (line 2413)
        kwargs_3574 = {}
        # Getting the type of 'title' (line 2413)
        title_3572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2413, 23), 'title', False)
        # Calling title(args, kwargs) (line 2413)
        title_call_result_3575 = invoke(stypy.reporting.localization.Localization(__file__, 2413, 23), title_3572, *[self_3573], **kwargs_3574)
        
        # Processing the call keyword arguments (line 2413)
        kwargs_3576 = {}
        # Getting the type of 'asarray' (line 2413)
        asarray_3571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2413, 15), 'asarray', False)
        # Calling asarray(args, kwargs) (line 2413)
        asarray_call_result_3577 = invoke(stypy.reporting.localization.Localization(__file__, 2413, 15), asarray_3571, *[title_call_result_3575], **kwargs_3576)
        
        # Assigning a type to the variable 'stypy_return_type' (line 2413)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2413, 8), 'stypy_return_type', asarray_call_result_3577)
        
        # ################# End of 'title(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'title' in the type store
        # Getting the type of 'stypy_return_type' (line 2402)
        stypy_return_type_3578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2402, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3578)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'title'
        return stypy_return_type_3578


    @norecursion
    def translate(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 2415)
        None_3579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2415, 43), 'None')
        defaults = [None_3579]
        # Create a new context for function 'translate'
        module_type_store = module_type_store.open_function_context('translate', 2415, 4, False)
        # Assigning a type to the variable 'self' (line 2416)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2416, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        chararray.translate.__dict__.__setitem__('stypy_localization', localization)
        chararray.translate.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        chararray.translate.__dict__.__setitem__('stypy_type_store', module_type_store)
        chararray.translate.__dict__.__setitem__('stypy_function_name', 'chararray.translate')
        chararray.translate.__dict__.__setitem__('stypy_param_names_list', ['table', 'deletechars'])
        chararray.translate.__dict__.__setitem__('stypy_varargs_param_name', None)
        chararray.translate.__dict__.__setitem__('stypy_kwargs_param_name', None)
        chararray.translate.__dict__.__setitem__('stypy_call_defaults', defaults)
        chararray.translate.__dict__.__setitem__('stypy_call_varargs', varargs)
        chararray.translate.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        chararray.translate.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'chararray.translate', ['table', 'deletechars'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'translate', localization, ['table', 'deletechars'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'translate(...)' code ##################

        str_3580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2426, (-1)), 'str', '\n        For each element in `self`, return a copy of the string where\n        all characters occurring in the optional argument\n        `deletechars` are removed, and the remaining characters have\n        been mapped through the given translation table.\n\n        See also\n        --------\n        char.translate\n\n        ')
        
        # Call to asarray(...): (line 2427)
        # Processing the call arguments (line 2427)
        
        # Call to translate(...): (line 2427)
        # Processing the call arguments (line 2427)
        # Getting the type of 'self' (line 2427)
        self_3583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2427, 33), 'self', False)
        # Getting the type of 'table' (line 2427)
        table_3584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2427, 39), 'table', False)
        # Getting the type of 'deletechars' (line 2427)
        deletechars_3585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2427, 46), 'deletechars', False)
        # Processing the call keyword arguments (line 2427)
        kwargs_3586 = {}
        # Getting the type of 'translate' (line 2427)
        translate_3582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2427, 23), 'translate', False)
        # Calling translate(args, kwargs) (line 2427)
        translate_call_result_3587 = invoke(stypy.reporting.localization.Localization(__file__, 2427, 23), translate_3582, *[self_3583, table_3584, deletechars_3585], **kwargs_3586)
        
        # Processing the call keyword arguments (line 2427)
        kwargs_3588 = {}
        # Getting the type of 'asarray' (line 2427)
        asarray_3581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2427, 15), 'asarray', False)
        # Calling asarray(args, kwargs) (line 2427)
        asarray_call_result_3589 = invoke(stypy.reporting.localization.Localization(__file__, 2427, 15), asarray_3581, *[translate_call_result_3587], **kwargs_3588)
        
        # Assigning a type to the variable 'stypy_return_type' (line 2427)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2427, 8), 'stypy_return_type', asarray_call_result_3589)
        
        # ################# End of 'translate(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'translate' in the type store
        # Getting the type of 'stypy_return_type' (line 2415)
        stypy_return_type_3590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2415, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3590)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'translate'
        return stypy_return_type_3590


    @norecursion
    def upper(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'upper'
        module_type_store = module_type_store.open_function_context('upper', 2429, 4, False)
        # Assigning a type to the variable 'self' (line 2430)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2430, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        chararray.upper.__dict__.__setitem__('stypy_localization', localization)
        chararray.upper.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        chararray.upper.__dict__.__setitem__('stypy_type_store', module_type_store)
        chararray.upper.__dict__.__setitem__('stypy_function_name', 'chararray.upper')
        chararray.upper.__dict__.__setitem__('stypy_param_names_list', [])
        chararray.upper.__dict__.__setitem__('stypy_varargs_param_name', None)
        chararray.upper.__dict__.__setitem__('stypy_kwargs_param_name', None)
        chararray.upper.__dict__.__setitem__('stypy_call_defaults', defaults)
        chararray.upper.__dict__.__setitem__('stypy_call_varargs', varargs)
        chararray.upper.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        chararray.upper.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'chararray.upper', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'upper', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'upper(...)' code ##################

        str_3591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2438, (-1)), 'str', '\n        Return an array with the elements of `self` converted to\n        uppercase.\n\n        See also\n        --------\n        char.upper\n\n        ')
        
        # Call to asarray(...): (line 2439)
        # Processing the call arguments (line 2439)
        
        # Call to upper(...): (line 2439)
        # Processing the call arguments (line 2439)
        # Getting the type of 'self' (line 2439)
        self_3594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2439, 29), 'self', False)
        # Processing the call keyword arguments (line 2439)
        kwargs_3595 = {}
        # Getting the type of 'upper' (line 2439)
        upper_3593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2439, 23), 'upper', False)
        # Calling upper(args, kwargs) (line 2439)
        upper_call_result_3596 = invoke(stypy.reporting.localization.Localization(__file__, 2439, 23), upper_3593, *[self_3594], **kwargs_3595)
        
        # Processing the call keyword arguments (line 2439)
        kwargs_3597 = {}
        # Getting the type of 'asarray' (line 2439)
        asarray_3592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2439, 15), 'asarray', False)
        # Calling asarray(args, kwargs) (line 2439)
        asarray_call_result_3598 = invoke(stypy.reporting.localization.Localization(__file__, 2439, 15), asarray_3592, *[upper_call_result_3596], **kwargs_3597)
        
        # Assigning a type to the variable 'stypy_return_type' (line 2439)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2439, 8), 'stypy_return_type', asarray_call_result_3598)
        
        # ################# End of 'upper(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'upper' in the type store
        # Getting the type of 'stypy_return_type' (line 2429)
        stypy_return_type_3599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2429, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3599)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'upper'
        return stypy_return_type_3599


    @norecursion
    def zfill(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'zfill'
        module_type_store = module_type_store.open_function_context('zfill', 2441, 4, False)
        # Assigning a type to the variable 'self' (line 2442)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2442, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        chararray.zfill.__dict__.__setitem__('stypy_localization', localization)
        chararray.zfill.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        chararray.zfill.__dict__.__setitem__('stypy_type_store', module_type_store)
        chararray.zfill.__dict__.__setitem__('stypy_function_name', 'chararray.zfill')
        chararray.zfill.__dict__.__setitem__('stypy_param_names_list', ['width'])
        chararray.zfill.__dict__.__setitem__('stypy_varargs_param_name', None)
        chararray.zfill.__dict__.__setitem__('stypy_kwargs_param_name', None)
        chararray.zfill.__dict__.__setitem__('stypy_call_defaults', defaults)
        chararray.zfill.__dict__.__setitem__('stypy_call_varargs', varargs)
        chararray.zfill.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        chararray.zfill.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'chararray.zfill', ['width'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'zfill', localization, ['width'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'zfill(...)' code ##################

        str_3600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2450, (-1)), 'str', '\n        Return the numeric string left-filled with zeros in a string of\n        length `width`.\n\n        See also\n        --------\n        char.zfill\n\n        ')
        
        # Call to asarray(...): (line 2451)
        # Processing the call arguments (line 2451)
        
        # Call to zfill(...): (line 2451)
        # Processing the call arguments (line 2451)
        # Getting the type of 'self' (line 2451)
        self_3603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2451, 29), 'self', False)
        # Getting the type of 'width' (line 2451)
        width_3604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2451, 35), 'width', False)
        # Processing the call keyword arguments (line 2451)
        kwargs_3605 = {}
        # Getting the type of 'zfill' (line 2451)
        zfill_3602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2451, 23), 'zfill', False)
        # Calling zfill(args, kwargs) (line 2451)
        zfill_call_result_3606 = invoke(stypy.reporting.localization.Localization(__file__, 2451, 23), zfill_3602, *[self_3603, width_3604], **kwargs_3605)
        
        # Processing the call keyword arguments (line 2451)
        kwargs_3607 = {}
        # Getting the type of 'asarray' (line 2451)
        asarray_3601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2451, 15), 'asarray', False)
        # Calling asarray(args, kwargs) (line 2451)
        asarray_call_result_3608 = invoke(stypy.reporting.localization.Localization(__file__, 2451, 15), asarray_3601, *[zfill_call_result_3606], **kwargs_3607)
        
        # Assigning a type to the variable 'stypy_return_type' (line 2451)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2451, 8), 'stypy_return_type', asarray_call_result_3608)
        
        # ################# End of 'zfill(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'zfill' in the type store
        # Getting the type of 'stypy_return_type' (line 2441)
        stypy_return_type_3609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2441, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3609)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'zfill'
        return stypy_return_type_3609


    @norecursion
    def isnumeric(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'isnumeric'
        module_type_store = module_type_store.open_function_context('isnumeric', 2453, 4, False)
        # Assigning a type to the variable 'self' (line 2454)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2454, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        chararray.isnumeric.__dict__.__setitem__('stypy_localization', localization)
        chararray.isnumeric.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        chararray.isnumeric.__dict__.__setitem__('stypy_type_store', module_type_store)
        chararray.isnumeric.__dict__.__setitem__('stypy_function_name', 'chararray.isnumeric')
        chararray.isnumeric.__dict__.__setitem__('stypy_param_names_list', [])
        chararray.isnumeric.__dict__.__setitem__('stypy_varargs_param_name', None)
        chararray.isnumeric.__dict__.__setitem__('stypy_kwargs_param_name', None)
        chararray.isnumeric.__dict__.__setitem__('stypy_call_defaults', defaults)
        chararray.isnumeric.__dict__.__setitem__('stypy_call_varargs', varargs)
        chararray.isnumeric.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        chararray.isnumeric.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'chararray.isnumeric', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'isnumeric', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'isnumeric(...)' code ##################

        str_3610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2462, (-1)), 'str', '\n        For each element in `self`, return True if there are only\n        numeric characters in the element.\n\n        See also\n        --------\n        char.isnumeric\n\n        ')
        
        # Call to isnumeric(...): (line 2463)
        # Processing the call arguments (line 2463)
        # Getting the type of 'self' (line 2463)
        self_3612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2463, 25), 'self', False)
        # Processing the call keyword arguments (line 2463)
        kwargs_3613 = {}
        # Getting the type of 'isnumeric' (line 2463)
        isnumeric_3611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2463, 15), 'isnumeric', False)
        # Calling isnumeric(args, kwargs) (line 2463)
        isnumeric_call_result_3614 = invoke(stypy.reporting.localization.Localization(__file__, 2463, 15), isnumeric_3611, *[self_3612], **kwargs_3613)
        
        # Assigning a type to the variable 'stypy_return_type' (line 2463)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2463, 8), 'stypy_return_type', isnumeric_call_result_3614)
        
        # ################# End of 'isnumeric(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'isnumeric' in the type store
        # Getting the type of 'stypy_return_type' (line 2453)
        stypy_return_type_3615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2453, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3615)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'isnumeric'
        return stypy_return_type_3615


    @norecursion
    def isdecimal(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'isdecimal'
        module_type_store = module_type_store.open_function_context('isdecimal', 2465, 4, False)
        # Assigning a type to the variable 'self' (line 2466)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2466, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        chararray.isdecimal.__dict__.__setitem__('stypy_localization', localization)
        chararray.isdecimal.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        chararray.isdecimal.__dict__.__setitem__('stypy_type_store', module_type_store)
        chararray.isdecimal.__dict__.__setitem__('stypy_function_name', 'chararray.isdecimal')
        chararray.isdecimal.__dict__.__setitem__('stypy_param_names_list', [])
        chararray.isdecimal.__dict__.__setitem__('stypy_varargs_param_name', None)
        chararray.isdecimal.__dict__.__setitem__('stypy_kwargs_param_name', None)
        chararray.isdecimal.__dict__.__setitem__('stypy_call_defaults', defaults)
        chararray.isdecimal.__dict__.__setitem__('stypy_call_varargs', varargs)
        chararray.isdecimal.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        chararray.isdecimal.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'chararray.isdecimal', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'isdecimal', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'isdecimal(...)' code ##################

        str_3616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2474, (-1)), 'str', '\n        For each element in `self`, return True if there are only\n        decimal characters in the element.\n\n        See also\n        --------\n        char.isdecimal\n\n        ')
        
        # Call to isdecimal(...): (line 2475)
        # Processing the call arguments (line 2475)
        # Getting the type of 'self' (line 2475)
        self_3618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2475, 25), 'self', False)
        # Processing the call keyword arguments (line 2475)
        kwargs_3619 = {}
        # Getting the type of 'isdecimal' (line 2475)
        isdecimal_3617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2475, 15), 'isdecimal', False)
        # Calling isdecimal(args, kwargs) (line 2475)
        isdecimal_call_result_3620 = invoke(stypy.reporting.localization.Localization(__file__, 2475, 15), isdecimal_3617, *[self_3618], **kwargs_3619)
        
        # Assigning a type to the variable 'stypy_return_type' (line 2475)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2475, 8), 'stypy_return_type', isdecimal_call_result_3620)
        
        # ################# End of 'isdecimal(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'isdecimal' in the type store
        # Getting the type of 'stypy_return_type' (line 2465)
        stypy_return_type_3621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2465, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3621)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'isdecimal'
        return stypy_return_type_3621


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 1669, 0, False)
        # Assigning a type to the variable 'self' (line 1670)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1670, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'chararray.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'chararray' (line 1669)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1669, 0), 'chararray', chararray)

# Assigning a Attribute to a Attribute (line 2006):
# Getting the type of 'ndarray' (line 2006)
ndarray_3622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2006, 22), 'ndarray')
# Obtaining the member 'argsort' of a type (line 2006)
argsort_3623 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2006, 22), ndarray_3622, 'argsort')
# Obtaining the member '__doc__' of a type (line 2006)
doc___3624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2006, 22), argsort_3623, '__doc__')
# Getting the type of 'chararray'
chararray_3625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'chararray')
# Obtaining the member 'argsort' of a type
argsort_3626 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), chararray_3625, 'argsort')
# Setting the type of the member '__doc__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), argsort_3626, '__doc__', doc___3624)

@norecursion
def array(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 2478)
    None_3627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2478, 24), 'None')
    # Getting the type of 'True' (line 2478)
    True_3628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2478, 35), 'True')
    # Getting the type of 'None' (line 2478)
    None_3629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2478, 49), 'None')
    # Getting the type of 'None' (line 2478)
    None_3630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2478, 61), 'None')
    defaults = [None_3627, True_3628, None_3629, None_3630]
    # Create a new context for function 'array'
    module_type_store = module_type_store.open_function_context('array', 2478, 0, False)
    
    # Passed parameters checking function
    array.stypy_localization = localization
    array.stypy_type_of_self = None
    array.stypy_type_store = module_type_store
    array.stypy_function_name = 'array'
    array.stypy_param_names_list = ['obj', 'itemsize', 'copy', 'unicode', 'order']
    array.stypy_varargs_param_name = None
    array.stypy_kwargs_param_name = None
    array.stypy_call_defaults = defaults
    array.stypy_call_varargs = varargs
    array.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'array', ['obj', 'itemsize', 'copy', 'unicode', 'order'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'array', localization, ['obj', 'itemsize', 'copy', 'unicode', 'order'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'array(...)' code ##################

    str_3631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2539, (-1)), 'str', "\n    Create a `chararray`.\n\n    .. note::\n       This class is provided for numarray backward-compatibility.\n       New code (not concerned with numarray compatibility) should use\n       arrays of type `string_` or `unicode_` and use the free functions\n       in :mod:`numpy.char <numpy.core.defchararray>` for fast\n       vectorized string operations instead.\n\n    Versus a regular Numpy array of type `str` or `unicode`, this\n    class adds the following functionality:\n\n      1) values automatically have whitespace removed from the end\n         when indexed\n\n      2) comparison operators automatically remove whitespace from the\n         end when comparing values\n\n      3) vectorized string operations are provided as methods\n         (e.g. `str.endswith`) and infix operators (e.g. ``+, *, %``)\n\n    Parameters\n    ----------\n    obj : array of str or unicode-like\n\n    itemsize : int, optional\n        `itemsize` is the number of characters per scalar in the\n        resulting array.  If `itemsize` is None, and `obj` is an\n        object array or a Python list, the `itemsize` will be\n        automatically determined.  If `itemsize` is provided and `obj`\n        is of type str or unicode, then the `obj` string will be\n        chunked into `itemsize` pieces.\n\n    copy : bool, optional\n        If true (default), then the object is copied.  Otherwise, a copy\n        will only be made if __array__ returns a copy, if obj is a\n        nested sequence, or if a copy is needed to satisfy any of the other\n        requirements (`itemsize`, unicode, `order`, etc.).\n\n    unicode : bool, optional\n        When true, the resulting `chararray` can contain Unicode\n        characters, when false only 8-bit characters.  If unicode is\n        `None` and `obj` is one of the following:\n\n          - a `chararray`,\n          - an ndarray of type `str` or `unicode`\n          - a Python str or unicode object,\n\n        then the unicode setting of the output array will be\n        automatically determined.\n\n    order : {'C', 'F', 'A'}, optional\n        Specify the order of the array.  If order is 'C' (default), then the\n        array will be in C-contiguous order (last-index varies the\n        fastest).  If order is 'F', then the returned array\n        will be in Fortran-contiguous order (first-index varies the\n        fastest).  If order is 'A', then the returned array may\n        be in any order (either C-, Fortran-contiguous, or even\n        discontiguous).\n    ")
    
    
    # Call to isinstance(...): (line 2540)
    # Processing the call arguments (line 2540)
    # Getting the type of 'obj' (line 2540)
    obj_3633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2540, 18), 'obj', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 2540)
    tuple_3634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2540, 24), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 2540)
    # Adding element type (line 2540)
    # Getting the type of '_bytes' (line 2540)
    _bytes_3635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2540, 24), '_bytes', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2540, 24), tuple_3634, _bytes_3635)
    # Adding element type (line 2540)
    # Getting the type of '_unicode' (line 2540)
    _unicode_3636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2540, 32), '_unicode', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2540, 24), tuple_3634, _unicode_3636)
    
    # Processing the call keyword arguments (line 2540)
    kwargs_3637 = {}
    # Getting the type of 'isinstance' (line 2540)
    isinstance_3632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2540, 7), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 2540)
    isinstance_call_result_3638 = invoke(stypy.reporting.localization.Localization(__file__, 2540, 7), isinstance_3632, *[obj_3633, tuple_3634], **kwargs_3637)
    
    # Testing the type of an if condition (line 2540)
    if_condition_3639 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 2540, 4), isinstance_call_result_3638)
    # Assigning a type to the variable 'if_condition_3639' (line 2540)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2540, 4), 'if_condition_3639', if_condition_3639)
    # SSA begins for if statement (line 2540)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Type idiom detected: calculating its left and rigth part (line 2541)
    # Getting the type of 'unicode' (line 2541)
    unicode_3640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2541, 11), 'unicode')
    # Getting the type of 'None' (line 2541)
    None_3641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2541, 22), 'None')
    
    (may_be_3642, more_types_in_union_3643) = may_be_none(unicode_3640, None_3641)

    if may_be_3642:

        if more_types_in_union_3643:
            # Runtime conditional SSA (line 2541)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        
        # Call to isinstance(...): (line 2542)
        # Processing the call arguments (line 2542)
        # Getting the type of 'obj' (line 2542)
        obj_3645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2542, 26), 'obj', False)
        # Getting the type of '_unicode' (line 2542)
        _unicode_3646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2542, 31), '_unicode', False)
        # Processing the call keyword arguments (line 2542)
        kwargs_3647 = {}
        # Getting the type of 'isinstance' (line 2542)
        isinstance_3644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2542, 15), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 2542)
        isinstance_call_result_3648 = invoke(stypy.reporting.localization.Localization(__file__, 2542, 15), isinstance_3644, *[obj_3645, _unicode_3646], **kwargs_3647)
        
        # Testing the type of an if condition (line 2542)
        if_condition_3649 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 2542, 12), isinstance_call_result_3648)
        # Assigning a type to the variable 'if_condition_3649' (line 2542)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2542, 12), 'if_condition_3649', if_condition_3649)
        # SSA begins for if statement (line 2542)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 2543):
        # Getting the type of 'True' (line 2543)
        True_3650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2543, 26), 'True')
        # Assigning a type to the variable 'unicode' (line 2543)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2543, 16), 'unicode', True_3650)
        # SSA branch for the else part of an if statement (line 2542)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Name to a Name (line 2545):
        # Getting the type of 'False' (line 2545)
        False_3651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2545, 26), 'False')
        # Assigning a type to the variable 'unicode' (line 2545)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2545, 16), 'unicode', False_3651)
        # SSA join for if statement (line 2542)
        module_type_store = module_type_store.join_ssa_context()
        

        if more_types_in_union_3643:
            # SSA join for if statement (line 2541)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 2547)
    # Getting the type of 'itemsize' (line 2547)
    itemsize_3652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2547, 11), 'itemsize')
    # Getting the type of 'None' (line 2547)
    None_3653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2547, 23), 'None')
    
    (may_be_3654, more_types_in_union_3655) = may_be_none(itemsize_3652, None_3653)

    if may_be_3654:

        if more_types_in_union_3655:
            # Runtime conditional SSA (line 2547)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 2548):
        
        # Call to _len(...): (line 2548)
        # Processing the call arguments (line 2548)
        # Getting the type of 'obj' (line 2548)
        obj_3657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2548, 28), 'obj', False)
        # Processing the call keyword arguments (line 2548)
        kwargs_3658 = {}
        # Getting the type of '_len' (line 2548)
        _len_3656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2548, 23), '_len', False)
        # Calling _len(args, kwargs) (line 2548)
        _len_call_result_3659 = invoke(stypy.reporting.localization.Localization(__file__, 2548, 23), _len_3656, *[obj_3657], **kwargs_3658)
        
        # Assigning a type to the variable 'itemsize' (line 2548)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2548, 12), 'itemsize', _len_call_result_3659)

        if more_types_in_union_3655:
            # SSA join for if statement (line 2547)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a BinOp to a Name (line 2549):
    
    # Call to _len(...): (line 2549)
    # Processing the call arguments (line 2549)
    # Getting the type of 'obj' (line 2549)
    obj_3661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2549, 21), 'obj', False)
    # Processing the call keyword arguments (line 2549)
    kwargs_3662 = {}
    # Getting the type of '_len' (line 2549)
    _len_3660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2549, 16), '_len', False)
    # Calling _len(args, kwargs) (line 2549)
    _len_call_result_3663 = invoke(stypy.reporting.localization.Localization(__file__, 2549, 16), _len_3660, *[obj_3661], **kwargs_3662)
    
    # Getting the type of 'itemsize' (line 2549)
    itemsize_3664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2549, 29), 'itemsize')
    # Applying the binary operator '//' (line 2549)
    result_floordiv_3665 = python_operator(stypy.reporting.localization.Localization(__file__, 2549, 16), '//', _len_call_result_3663, itemsize_3664)
    
    # Assigning a type to the variable 'shape' (line 2549)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2549, 8), 'shape', result_floordiv_3665)
    
    # Getting the type of 'unicode' (line 2551)
    unicode_3666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2551, 11), 'unicode')
    # Testing the type of an if condition (line 2551)
    if_condition_3667 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 2551, 8), unicode_3666)
    # Assigning a type to the variable 'if_condition_3667' (line 2551)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2551, 8), 'if_condition_3667', if_condition_3667)
    # SSA begins for if statement (line 2551)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Getting the type of 'sys' (line 2552)
    sys_3668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2552, 15), 'sys')
    # Obtaining the member 'maxunicode' of a type (line 2552)
    maxunicode_3669 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2552, 15), sys_3668, 'maxunicode')
    int_3670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2552, 33), 'int')
    # Applying the binary operator '==' (line 2552)
    result_eq_3671 = python_operator(stypy.reporting.localization.Localization(__file__, 2552, 15), '==', maxunicode_3669, int_3670)
    
    # Testing the type of an if condition (line 2552)
    if_condition_3672 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 2552, 12), result_eq_3671)
    # Assigning a type to the variable 'if_condition_3672' (line 2552)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2552, 12), 'if_condition_3672', if_condition_3672)
    # SSA begins for if statement (line 2552)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Getting the type of 'sys' (line 2562)
    sys_3673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2562, 19), 'sys')
    # Obtaining the member 'hexversion' of a type (line 2562)
    hexversion_3674 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2562, 19), sys_3673, 'hexversion')
    int_3675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2562, 37), 'int')
    # Applying the binary operator '>=' (line 2562)
    result_ge_3676 = python_operator(stypy.reporting.localization.Localization(__file__, 2562, 19), '>=', hexversion_3674, int_3675)
    
    # Testing the type of an if condition (line 2562)
    if_condition_3677 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 2562, 16), result_ge_3676)
    # Assigning a type to the variable 'if_condition_3677' (line 2562)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2562, 16), 'if_condition_3677', if_condition_3677)
    # SSA begins for if statement (line 2562)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 2563):
    
    # Call to encode(...): (line 2563)
    # Processing the call arguments (line 2563)
    str_3680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2563, 37), 'str', 'utf_32')
    # Processing the call keyword arguments (line 2563)
    kwargs_3681 = {}
    # Getting the type of 'obj' (line 2563)
    obj_3678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2563, 26), 'obj', False)
    # Obtaining the member 'encode' of a type (line 2563)
    encode_3679 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2563, 26), obj_3678, 'encode')
    # Calling encode(args, kwargs) (line 2563)
    encode_call_result_3682 = invoke(stypy.reporting.localization.Localization(__file__, 2563, 26), encode_3679, *[str_3680], **kwargs_3681)
    
    # Assigning a type to the variable 'obj' (line 2563)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2563, 20), 'obj', encode_call_result_3682)
    # SSA branch for the else part of an if statement (line 2562)
    module_type_store.open_ssa_branch('else')
    
    # Type idiom detected: calculating its left and rigth part (line 2565)
    # Getting the type of 'str' (line 2565)
    str_3683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2565, 39), 'str')
    # Getting the type of 'obj' (line 2565)
    obj_3684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2565, 34), 'obj')
    
    (may_be_3685, more_types_in_union_3686) = may_be_subtype(str_3683, obj_3684)

    if may_be_3685:

        if more_types_in_union_3686:
            # Runtime conditional SSA (line 2565)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'obj' (line 2565)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2565, 20), 'obj', remove_not_subtype_from_union(obj_3684, str))
        
        # Assigning a Call to a Name (line 2566):
        
        # Call to frombuffer(...): (line 2566)
        # Processing the call arguments (line 2566)
        # Getting the type of 'obj' (line 2566)
        obj_3689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2566, 49), 'obj', False)
        str_3690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2566, 54), 'str', 'u1')
        # Processing the call keyword arguments (line 2566)
        kwargs_3691 = {}
        # Getting the type of 'numpy' (line 2566)
        numpy_3687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2566, 32), 'numpy', False)
        # Obtaining the member 'frombuffer' of a type (line 2566)
        frombuffer_3688 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2566, 32), numpy_3687, 'frombuffer')
        # Calling frombuffer(args, kwargs) (line 2566)
        frombuffer_call_result_3692 = invoke(stypy.reporting.localization.Localization(__file__, 2566, 32), frombuffer_3688, *[obj_3689, str_3690], **kwargs_3691)
        
        # Assigning a type to the variable 'ascii' (line 2566)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2566, 24), 'ascii', frombuffer_call_result_3692)
        
        # Assigning a Call to a Name (line 2567):
        
        # Call to array(...): (line 2567)
        # Processing the call arguments (line 2567)
        # Getting the type of 'ascii' (line 2567)
        ascii_3695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2567, 43), 'ascii', False)
        str_3696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2567, 50), 'str', 'u4')
        # Processing the call keyword arguments (line 2567)
        kwargs_3697 = {}
        # Getting the type of 'numpy' (line 2567)
        numpy_3693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2567, 31), 'numpy', False)
        # Obtaining the member 'array' of a type (line 2567)
        array_3694 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2567, 31), numpy_3693, 'array')
        # Calling array(args, kwargs) (line 2567)
        array_call_result_3698 = invoke(stypy.reporting.localization.Localization(__file__, 2567, 31), array_3694, *[ascii_3695, str_3696], **kwargs_3697)
        
        # Assigning a type to the variable 'ucs4' (line 2567)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2567, 24), 'ucs4', array_call_result_3698)
        
        # Assigning a Attribute to a Name (line 2568):
        # Getting the type of 'ucs4' (line 2568)
        ucs4_3699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2568, 30), 'ucs4')
        # Obtaining the member 'data' of a type (line 2568)
        data_3700 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2568, 30), ucs4_3699, 'data')
        # Assigning a type to the variable 'obj' (line 2568)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2568, 24), 'obj', data_3700)

        if more_types_in_union_3686:
            # Runtime conditional SSA for else branch (line 2565)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_3685) or more_types_in_union_3686):
        # Assigning a type to the variable 'obj' (line 2565)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2565, 20), 'obj', remove_subtype_from_union(obj_3684, str))
        
        # Assigning a Call to a Name (line 2570):
        
        # Call to frombuffer(...): (line 2570)
        # Processing the call arguments (line 2570)
        # Getting the type of 'obj' (line 2570)
        obj_3703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2570, 48), 'obj', False)
        str_3704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2570, 53), 'str', 'u2')
        # Processing the call keyword arguments (line 2570)
        kwargs_3705 = {}
        # Getting the type of 'numpy' (line 2570)
        numpy_3701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2570, 31), 'numpy', False)
        # Obtaining the member 'frombuffer' of a type (line 2570)
        frombuffer_3702 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2570, 31), numpy_3701, 'frombuffer')
        # Calling frombuffer(args, kwargs) (line 2570)
        frombuffer_call_result_3706 = invoke(stypy.reporting.localization.Localization(__file__, 2570, 31), frombuffer_3702, *[obj_3703, str_3704], **kwargs_3705)
        
        # Assigning a type to the variable 'ucs2' (line 2570)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2570, 24), 'ucs2', frombuffer_call_result_3706)
        
        # Assigning a Call to a Name (line 2571):
        
        # Call to array(...): (line 2571)
        # Processing the call arguments (line 2571)
        # Getting the type of 'ucs2' (line 2571)
        ucs2_3709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2571, 43), 'ucs2', False)
        str_3710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2571, 49), 'str', 'u4')
        # Processing the call keyword arguments (line 2571)
        kwargs_3711 = {}
        # Getting the type of 'numpy' (line 2571)
        numpy_3707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2571, 31), 'numpy', False)
        # Obtaining the member 'array' of a type (line 2571)
        array_3708 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2571, 31), numpy_3707, 'array')
        # Calling array(args, kwargs) (line 2571)
        array_call_result_3712 = invoke(stypy.reporting.localization.Localization(__file__, 2571, 31), array_3708, *[ucs2_3709, str_3710], **kwargs_3711)
        
        # Assigning a type to the variable 'ucs4' (line 2571)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2571, 24), 'ucs4', array_call_result_3712)
        
        # Assigning a Attribute to a Name (line 2572):
        # Getting the type of 'ucs4' (line 2572)
        ucs4_3713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2572, 30), 'ucs4')
        # Obtaining the member 'data' of a type (line 2572)
        data_3714 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2572, 30), ucs4_3713, 'data')
        # Assigning a type to the variable 'obj' (line 2572)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2572, 24), 'obj', data_3714)

        if (may_be_3685 and more_types_in_union_3686):
            # SSA join for if statement (line 2565)
            module_type_store = module_type_store.join_ssa_context()


    
    # SSA join for if statement (line 2562)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 2552)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 2574):
    
    # Call to _unicode(...): (line 2574)
    # Processing the call arguments (line 2574)
    # Getting the type of 'obj' (line 2574)
    obj_3716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2574, 31), 'obj', False)
    # Processing the call keyword arguments (line 2574)
    kwargs_3717 = {}
    # Getting the type of '_unicode' (line 2574)
    _unicode_3715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2574, 22), '_unicode', False)
    # Calling _unicode(args, kwargs) (line 2574)
    _unicode_call_result_3718 = invoke(stypy.reporting.localization.Localization(__file__, 2574, 22), _unicode_3715, *[obj_3716], **kwargs_3717)
    
    # Assigning a type to the variable 'obj' (line 2574)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2574, 16), 'obj', _unicode_call_result_3718)
    # SSA join for if statement (line 2552)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 2551)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 2578):
    
    # Call to _bytes(...): (line 2578)
    # Processing the call arguments (line 2578)
    # Getting the type of 'obj' (line 2578)
    obj_3720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2578, 25), 'obj', False)
    # Processing the call keyword arguments (line 2578)
    kwargs_3721 = {}
    # Getting the type of '_bytes' (line 2578)
    _bytes_3719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2578, 18), '_bytes', False)
    # Calling _bytes(args, kwargs) (line 2578)
    _bytes_call_result_3722 = invoke(stypy.reporting.localization.Localization(__file__, 2578, 18), _bytes_3719, *[obj_3720], **kwargs_3721)
    
    # Assigning a type to the variable 'obj' (line 2578)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2578, 12), 'obj', _bytes_call_result_3722)
    # SSA join for if statement (line 2551)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to chararray(...): (line 2580)
    # Processing the call arguments (line 2580)
    # Getting the type of 'shape' (line 2580)
    shape_3724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2580, 25), 'shape', False)
    # Processing the call keyword arguments (line 2580)
    # Getting the type of 'itemsize' (line 2580)
    itemsize_3725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2580, 41), 'itemsize', False)
    keyword_3726 = itemsize_3725
    # Getting the type of 'unicode' (line 2580)
    unicode_3727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2580, 59), 'unicode', False)
    keyword_3728 = unicode_3727
    # Getting the type of 'obj' (line 2581)
    obj_3729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2581, 32), 'obj', False)
    keyword_3730 = obj_3729
    # Getting the type of 'order' (line 2581)
    order_3731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2581, 43), 'order', False)
    keyword_3732 = order_3731
    kwargs_3733 = {'buffer': keyword_3730, 'itemsize': keyword_3726, 'order': keyword_3732, 'unicode': keyword_3728}
    # Getting the type of 'chararray' (line 2580)
    chararray_3723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2580, 15), 'chararray', False)
    # Calling chararray(args, kwargs) (line 2580)
    chararray_call_result_3734 = invoke(stypy.reporting.localization.Localization(__file__, 2580, 15), chararray_3723, *[shape_3724], **kwargs_3733)
    
    # Assigning a type to the variable 'stypy_return_type' (line 2580)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2580, 8), 'stypy_return_type', chararray_call_result_3734)
    # SSA join for if statement (line 2540)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to isinstance(...): (line 2583)
    # Processing the call arguments (line 2583)
    # Getting the type of 'obj' (line 2583)
    obj_3736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2583, 18), 'obj', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 2583)
    tuple_3737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2583, 24), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 2583)
    # Adding element type (line 2583)
    # Getting the type of 'list' (line 2583)
    list_3738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2583, 24), 'list', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2583, 24), tuple_3737, list_3738)
    # Adding element type (line 2583)
    # Getting the type of 'tuple' (line 2583)
    tuple_3739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2583, 30), 'tuple', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2583, 24), tuple_3737, tuple_3739)
    
    # Processing the call keyword arguments (line 2583)
    kwargs_3740 = {}
    # Getting the type of 'isinstance' (line 2583)
    isinstance_3735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2583, 7), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 2583)
    isinstance_call_result_3741 = invoke(stypy.reporting.localization.Localization(__file__, 2583, 7), isinstance_3735, *[obj_3736, tuple_3737], **kwargs_3740)
    
    # Testing the type of an if condition (line 2583)
    if_condition_3742 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 2583, 4), isinstance_call_result_3741)
    # Assigning a type to the variable 'if_condition_3742' (line 2583)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2583, 4), 'if_condition_3742', if_condition_3742)
    # SSA begins for if statement (line 2583)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 2584):
    
    # Call to asarray(...): (line 2584)
    # Processing the call arguments (line 2584)
    # Getting the type of 'obj' (line 2584)
    obj_3745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2584, 28), 'obj', False)
    # Processing the call keyword arguments (line 2584)
    kwargs_3746 = {}
    # Getting the type of 'numpy' (line 2584)
    numpy_3743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2584, 14), 'numpy', False)
    # Obtaining the member 'asarray' of a type (line 2584)
    asarray_3744 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2584, 14), numpy_3743, 'asarray')
    # Calling asarray(args, kwargs) (line 2584)
    asarray_call_result_3747 = invoke(stypy.reporting.localization.Localization(__file__, 2584, 14), asarray_3744, *[obj_3745], **kwargs_3746)
    
    # Assigning a type to the variable 'obj' (line 2584)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2584, 8), 'obj', asarray_call_result_3747)
    # SSA join for if statement (line 2583)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Call to isinstance(...): (line 2586)
    # Processing the call arguments (line 2586)
    # Getting the type of 'obj' (line 2586)
    obj_3749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2586, 18), 'obj', False)
    # Getting the type of 'ndarray' (line 2586)
    ndarray_3750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2586, 23), 'ndarray', False)
    # Processing the call keyword arguments (line 2586)
    kwargs_3751 = {}
    # Getting the type of 'isinstance' (line 2586)
    isinstance_3748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2586, 7), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 2586)
    isinstance_call_result_3752 = invoke(stypy.reporting.localization.Localization(__file__, 2586, 7), isinstance_3748, *[obj_3749, ndarray_3750], **kwargs_3751)
    
    
    # Call to issubclass(...): (line 2586)
    # Processing the call arguments (line 2586)
    # Getting the type of 'obj' (line 2586)
    obj_3754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2586, 47), 'obj', False)
    # Obtaining the member 'dtype' of a type (line 2586)
    dtype_3755 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2586, 47), obj_3754, 'dtype')
    # Obtaining the member 'type' of a type (line 2586)
    type_3756 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2586, 47), dtype_3755, 'type')
    # Getting the type of 'character' (line 2586)
    character_3757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2586, 63), 'character', False)
    # Processing the call keyword arguments (line 2586)
    kwargs_3758 = {}
    # Getting the type of 'issubclass' (line 2586)
    issubclass_3753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2586, 36), 'issubclass', False)
    # Calling issubclass(args, kwargs) (line 2586)
    issubclass_call_result_3759 = invoke(stypy.reporting.localization.Localization(__file__, 2586, 36), issubclass_3753, *[type_3756, character_3757], **kwargs_3758)
    
    # Applying the binary operator 'and' (line 2586)
    result_and_keyword_3760 = python_operator(stypy.reporting.localization.Localization(__file__, 2586, 7), 'and', isinstance_call_result_3752, issubclass_call_result_3759)
    
    # Testing the type of an if condition (line 2586)
    if_condition_3761 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 2586, 4), result_and_keyword_3760)
    # Assigning a type to the variable 'if_condition_3761' (line 2586)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2586, 4), 'if_condition_3761', if_condition_3761)
    # SSA begins for if statement (line 2586)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    
    # Call to isinstance(...): (line 2589)
    # Processing the call arguments (line 2589)
    # Getting the type of 'obj' (line 2589)
    obj_3763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2589, 26), 'obj', False)
    # Getting the type of 'chararray' (line 2589)
    chararray_3764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2589, 31), 'chararray', False)
    # Processing the call keyword arguments (line 2589)
    kwargs_3765 = {}
    # Getting the type of 'isinstance' (line 2589)
    isinstance_3762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2589, 15), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 2589)
    isinstance_call_result_3766 = invoke(stypy.reporting.localization.Localization(__file__, 2589, 15), isinstance_3762, *[obj_3763, chararray_3764], **kwargs_3765)
    
    # Applying the 'not' unary operator (line 2589)
    result_not__3767 = python_operator(stypy.reporting.localization.Localization(__file__, 2589, 11), 'not', isinstance_call_result_3766)
    
    # Testing the type of an if condition (line 2589)
    if_condition_3768 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 2589, 8), result_not__3767)
    # Assigning a type to the variable 'if_condition_3768' (line 2589)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2589, 8), 'if_condition_3768', if_condition_3768)
    # SSA begins for if statement (line 2589)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 2590):
    
    # Call to view(...): (line 2590)
    # Processing the call arguments (line 2590)
    # Getting the type of 'chararray' (line 2590)
    chararray_3771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2590, 27), 'chararray', False)
    # Processing the call keyword arguments (line 2590)
    kwargs_3772 = {}
    # Getting the type of 'obj' (line 2590)
    obj_3769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2590, 18), 'obj', False)
    # Obtaining the member 'view' of a type (line 2590)
    view_3770 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2590, 18), obj_3769, 'view')
    # Calling view(args, kwargs) (line 2590)
    view_call_result_3773 = invoke(stypy.reporting.localization.Localization(__file__, 2590, 18), view_3770, *[chararray_3771], **kwargs_3772)
    
    # Assigning a type to the variable 'obj' (line 2590)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2590, 12), 'obj', view_call_result_3773)
    # SSA join for if statement (line 2589)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 2592)
    # Getting the type of 'itemsize' (line 2592)
    itemsize_3774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2592, 11), 'itemsize')
    # Getting the type of 'None' (line 2592)
    None_3775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2592, 23), 'None')
    
    (may_be_3776, more_types_in_union_3777) = may_be_none(itemsize_3774, None_3775)

    if may_be_3776:

        if more_types_in_union_3777:
            # Runtime conditional SSA (line 2592)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Attribute to a Name (line 2593):
        # Getting the type of 'obj' (line 2593)
        obj_3778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2593, 23), 'obj')
        # Obtaining the member 'itemsize' of a type (line 2593)
        itemsize_3779 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2593, 23), obj_3778, 'itemsize')
        # Assigning a type to the variable 'itemsize' (line 2593)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2593, 12), 'itemsize', itemsize_3779)
        
        
        # Call to issubclass(...): (line 2597)
        # Processing the call arguments (line 2597)
        # Getting the type of 'obj' (line 2597)
        obj_3781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2597, 26), 'obj', False)
        # Obtaining the member 'dtype' of a type (line 2597)
        dtype_3782 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2597, 26), obj_3781, 'dtype')
        # Obtaining the member 'type' of a type (line 2597)
        type_3783 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2597, 26), dtype_3782, 'type')
        # Getting the type of 'unicode_' (line 2597)
        unicode__3784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2597, 42), 'unicode_', False)
        # Processing the call keyword arguments (line 2597)
        kwargs_3785 = {}
        # Getting the type of 'issubclass' (line 2597)
        issubclass_3780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2597, 15), 'issubclass', False)
        # Calling issubclass(args, kwargs) (line 2597)
        issubclass_call_result_3786 = invoke(stypy.reporting.localization.Localization(__file__, 2597, 15), issubclass_3780, *[type_3783, unicode__3784], **kwargs_3785)
        
        # Testing the type of an if condition (line 2597)
        if_condition_3787 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 2597, 12), issubclass_call_result_3786)
        # Assigning a type to the variable 'if_condition_3787' (line 2597)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2597, 12), 'if_condition_3787', if_condition_3787)
        # SSA begins for if statement (line 2597)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'itemsize' (line 2598)
        itemsize_3788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2598, 16), 'itemsize')
        int_3789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2598, 29), 'int')
        # Applying the binary operator '//=' (line 2598)
        result_ifloordiv_3790 = python_operator(stypy.reporting.localization.Localization(__file__, 2598, 16), '//=', itemsize_3788, int_3789)
        # Assigning a type to the variable 'itemsize' (line 2598)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2598, 16), 'itemsize', result_ifloordiv_3790)
        
        # SSA join for if statement (line 2597)
        module_type_store = module_type_store.join_ssa_context()
        

        if more_types_in_union_3777:
            # SSA join for if statement (line 2592)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 2600)
    # Getting the type of 'unicode' (line 2600)
    unicode_3791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2600, 11), 'unicode')
    # Getting the type of 'None' (line 2600)
    None_3792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2600, 22), 'None')
    
    (may_be_3793, more_types_in_union_3794) = may_be_none(unicode_3791, None_3792)

    if may_be_3793:

        if more_types_in_union_3794:
            # Runtime conditional SSA (line 2600)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        
        # Call to issubclass(...): (line 2601)
        # Processing the call arguments (line 2601)
        # Getting the type of 'obj' (line 2601)
        obj_3796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2601, 26), 'obj', False)
        # Obtaining the member 'dtype' of a type (line 2601)
        dtype_3797 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2601, 26), obj_3796, 'dtype')
        # Obtaining the member 'type' of a type (line 2601)
        type_3798 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2601, 26), dtype_3797, 'type')
        # Getting the type of 'unicode_' (line 2601)
        unicode__3799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2601, 42), 'unicode_', False)
        # Processing the call keyword arguments (line 2601)
        kwargs_3800 = {}
        # Getting the type of 'issubclass' (line 2601)
        issubclass_3795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2601, 15), 'issubclass', False)
        # Calling issubclass(args, kwargs) (line 2601)
        issubclass_call_result_3801 = invoke(stypy.reporting.localization.Localization(__file__, 2601, 15), issubclass_3795, *[type_3798, unicode__3799], **kwargs_3800)
        
        # Testing the type of an if condition (line 2601)
        if_condition_3802 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 2601, 12), issubclass_call_result_3801)
        # Assigning a type to the variable 'if_condition_3802' (line 2601)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2601, 12), 'if_condition_3802', if_condition_3802)
        # SSA begins for if statement (line 2601)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 2602):
        # Getting the type of 'True' (line 2602)
        True_3803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2602, 26), 'True')
        # Assigning a type to the variable 'unicode' (line 2602)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2602, 16), 'unicode', True_3803)
        # SSA branch for the else part of an if statement (line 2601)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Name to a Name (line 2604):
        # Getting the type of 'False' (line 2604)
        False_3804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2604, 26), 'False')
        # Assigning a type to the variable 'unicode' (line 2604)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2604, 16), 'unicode', False_3804)
        # SSA join for if statement (line 2601)
        module_type_store = module_type_store.join_ssa_context()
        

        if more_types_in_union_3794:
            # SSA join for if statement (line 2600)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Getting the type of 'unicode' (line 2606)
    unicode_3805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2606, 11), 'unicode')
    # Testing the type of an if condition (line 2606)
    if_condition_3806 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 2606, 8), unicode_3805)
    # Assigning a type to the variable 'if_condition_3806' (line 2606)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2606, 8), 'if_condition_3806', if_condition_3806)
    # SSA begins for if statement (line 2606)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 2607):
    # Getting the type of 'unicode_' (line 2607)
    unicode__3807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2607, 20), 'unicode_')
    # Assigning a type to the variable 'dtype' (line 2607)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2607, 12), 'dtype', unicode__3807)
    # SSA branch for the else part of an if statement (line 2606)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Name (line 2609):
    # Getting the type of 'string_' (line 2609)
    string__3808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2609, 20), 'string_')
    # Assigning a type to the variable 'dtype' (line 2609)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2609, 12), 'dtype', string__3808)
    # SSA join for if statement (line 2606)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 2611)
    # Getting the type of 'order' (line 2611)
    order_3809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2611, 8), 'order')
    # Getting the type of 'None' (line 2611)
    None_3810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2611, 24), 'None')
    
    (may_be_3811, more_types_in_union_3812) = may_not_be_none(order_3809, None_3810)

    if may_be_3811:

        if more_types_in_union_3812:
            # Runtime conditional SSA (line 2611)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 2612):
        
        # Call to asarray(...): (line 2612)
        # Processing the call arguments (line 2612)
        # Getting the type of 'obj' (line 2612)
        obj_3815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2612, 32), 'obj', False)
        # Processing the call keyword arguments (line 2612)
        # Getting the type of 'order' (line 2612)
        order_3816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2612, 43), 'order', False)
        keyword_3817 = order_3816
        kwargs_3818 = {'order': keyword_3817}
        # Getting the type of 'numpy' (line 2612)
        numpy_3813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2612, 18), 'numpy', False)
        # Obtaining the member 'asarray' of a type (line 2612)
        asarray_3814 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2612, 18), numpy_3813, 'asarray')
        # Calling asarray(args, kwargs) (line 2612)
        asarray_call_result_3819 = invoke(stypy.reporting.localization.Localization(__file__, 2612, 18), asarray_3814, *[obj_3815], **kwargs_3818)
        
        # Assigning a type to the variable 'obj' (line 2612)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2612, 12), 'obj', asarray_call_result_3819)

        if more_types_in_union_3812:
            # SSA join for if statement (line 2611)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Evaluating a boolean operation
    # Getting the type of 'copy' (line 2613)
    copy_3820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2613, 12), 'copy')
    
    # Getting the type of 'itemsize' (line 2614)
    itemsize_3821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2614, 17), 'itemsize')
    # Getting the type of 'obj' (line 2614)
    obj_3822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2614, 29), 'obj')
    # Obtaining the member 'itemsize' of a type (line 2614)
    itemsize_3823 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2614, 29), obj_3822, 'itemsize')
    # Applying the binary operator '!=' (line 2614)
    result_ne_3824 = python_operator(stypy.reporting.localization.Localization(__file__, 2614, 17), '!=', itemsize_3821, itemsize_3823)
    
    # Applying the binary operator 'or' (line 2613)
    result_or_keyword_3825 = python_operator(stypy.reporting.localization.Localization(__file__, 2613, 12), 'or', copy_3820, result_ne_3824)
    
    # Evaluating a boolean operation
    
    # Getting the type of 'unicode' (line 2615)
    unicode_3826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2615, 21), 'unicode')
    # Applying the 'not' unary operator (line 2615)
    result_not__3827 = python_operator(stypy.reporting.localization.Localization(__file__, 2615, 17), 'not', unicode_3826)
    
    
    # Call to isinstance(...): (line 2615)
    # Processing the call arguments (line 2615)
    # Getting the type of 'obj' (line 2615)
    obj_3829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2615, 44), 'obj', False)
    # Getting the type of 'unicode_' (line 2615)
    unicode__3830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2615, 49), 'unicode_', False)
    # Processing the call keyword arguments (line 2615)
    kwargs_3831 = {}
    # Getting the type of 'isinstance' (line 2615)
    isinstance_3828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2615, 33), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 2615)
    isinstance_call_result_3832 = invoke(stypy.reporting.localization.Localization(__file__, 2615, 33), isinstance_3828, *[obj_3829, unicode__3830], **kwargs_3831)
    
    # Applying the binary operator 'and' (line 2615)
    result_and_keyword_3833 = python_operator(stypy.reporting.localization.Localization(__file__, 2615, 17), 'and', result_not__3827, isinstance_call_result_3832)
    
    # Applying the binary operator 'or' (line 2613)
    result_or_keyword_3834 = python_operator(stypy.reporting.localization.Localization(__file__, 2613, 12), 'or', result_or_keyword_3825, result_and_keyword_3833)
    
    # Evaluating a boolean operation
    # Getting the type of 'unicode' (line 2616)
    unicode_3835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2616, 17), 'unicode')
    
    # Call to isinstance(...): (line 2616)
    # Processing the call arguments (line 2616)
    # Getting the type of 'obj' (line 2616)
    obj_3837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2616, 40), 'obj', False)
    # Getting the type of 'string_' (line 2616)
    string__3838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2616, 45), 'string_', False)
    # Processing the call keyword arguments (line 2616)
    kwargs_3839 = {}
    # Getting the type of 'isinstance' (line 2616)
    isinstance_3836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2616, 29), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 2616)
    isinstance_call_result_3840 = invoke(stypy.reporting.localization.Localization(__file__, 2616, 29), isinstance_3836, *[obj_3837, string__3838], **kwargs_3839)
    
    # Applying the binary operator 'and' (line 2616)
    result_and_keyword_3841 = python_operator(stypy.reporting.localization.Localization(__file__, 2616, 17), 'and', unicode_3835, isinstance_call_result_3840)
    
    # Applying the binary operator 'or' (line 2613)
    result_or_keyword_3842 = python_operator(stypy.reporting.localization.Localization(__file__, 2613, 12), 'or', result_or_keyword_3834, result_and_keyword_3841)
    
    # Testing the type of an if condition (line 2613)
    if_condition_3843 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 2613, 8), result_or_keyword_3842)
    # Assigning a type to the variable 'if_condition_3843' (line 2613)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2613, 8), 'if_condition_3843', if_condition_3843)
    # SSA begins for if statement (line 2613)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 2617):
    
    # Call to astype(...): (line 2617)
    # Processing the call arguments (line 2617)
    
    # Obtaining an instance of the builtin type 'tuple' (line 2617)
    tuple_3846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2617, 30), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 2617)
    # Adding element type (line 2617)
    # Getting the type of 'dtype' (line 2617)
    dtype_3847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2617, 30), 'dtype', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2617, 30), tuple_3846, dtype_3847)
    # Adding element type (line 2617)
    
    # Call to long(...): (line 2617)
    # Processing the call arguments (line 2617)
    # Getting the type of 'itemsize' (line 2617)
    itemsize_3849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2617, 42), 'itemsize', False)
    # Processing the call keyword arguments (line 2617)
    kwargs_3850 = {}
    # Getting the type of 'long' (line 2617)
    long_3848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2617, 37), 'long', False)
    # Calling long(args, kwargs) (line 2617)
    long_call_result_3851 = invoke(stypy.reporting.localization.Localization(__file__, 2617, 37), long_3848, *[itemsize_3849], **kwargs_3850)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2617, 30), tuple_3846, long_call_result_3851)
    
    # Processing the call keyword arguments (line 2617)
    kwargs_3852 = {}
    # Getting the type of 'obj' (line 2617)
    obj_3844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2617, 18), 'obj', False)
    # Obtaining the member 'astype' of a type (line 2617)
    astype_3845 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2617, 18), obj_3844, 'astype')
    # Calling astype(args, kwargs) (line 2617)
    astype_call_result_3853 = invoke(stypy.reporting.localization.Localization(__file__, 2617, 18), astype_3845, *[tuple_3846], **kwargs_3852)
    
    # Assigning a type to the variable 'obj' (line 2617)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2617, 12), 'obj', astype_call_result_3853)
    # SSA join for if statement (line 2613)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'obj' (line 2618)
    obj_3854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2618, 15), 'obj')
    # Assigning a type to the variable 'stypy_return_type' (line 2618)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2618, 8), 'stypy_return_type', obj_3854)
    # SSA join for if statement (line 2586)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Call to isinstance(...): (line 2620)
    # Processing the call arguments (line 2620)
    # Getting the type of 'obj' (line 2620)
    obj_3856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2620, 18), 'obj', False)
    # Getting the type of 'ndarray' (line 2620)
    ndarray_3857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2620, 23), 'ndarray', False)
    # Processing the call keyword arguments (line 2620)
    kwargs_3858 = {}
    # Getting the type of 'isinstance' (line 2620)
    isinstance_3855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2620, 7), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 2620)
    isinstance_call_result_3859 = invoke(stypy.reporting.localization.Localization(__file__, 2620, 7), isinstance_3855, *[obj_3856, ndarray_3857], **kwargs_3858)
    
    
    # Call to issubclass(...): (line 2620)
    # Processing the call arguments (line 2620)
    # Getting the type of 'obj' (line 2620)
    obj_3861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2620, 47), 'obj', False)
    # Obtaining the member 'dtype' of a type (line 2620)
    dtype_3862 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2620, 47), obj_3861, 'dtype')
    # Obtaining the member 'type' of a type (line 2620)
    type_3863 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2620, 47), dtype_3862, 'type')
    # Getting the type of 'object' (line 2620)
    object_3864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2620, 63), 'object', False)
    # Processing the call keyword arguments (line 2620)
    kwargs_3865 = {}
    # Getting the type of 'issubclass' (line 2620)
    issubclass_3860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2620, 36), 'issubclass', False)
    # Calling issubclass(args, kwargs) (line 2620)
    issubclass_call_result_3866 = invoke(stypy.reporting.localization.Localization(__file__, 2620, 36), issubclass_3860, *[type_3863, object_3864], **kwargs_3865)
    
    # Applying the binary operator 'and' (line 2620)
    result_and_keyword_3867 = python_operator(stypy.reporting.localization.Localization(__file__, 2620, 7), 'and', isinstance_call_result_3859, issubclass_call_result_3866)
    
    # Testing the type of an if condition (line 2620)
    if_condition_3868 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 2620, 4), result_and_keyword_3867)
    # Assigning a type to the variable 'if_condition_3868' (line 2620)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2620, 4), 'if_condition_3868', if_condition_3868)
    # SSA begins for if statement (line 2620)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Type idiom detected: calculating its left and rigth part (line 2621)
    # Getting the type of 'itemsize' (line 2621)
    itemsize_3869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2621, 11), 'itemsize')
    # Getting the type of 'None' (line 2621)
    None_3870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2621, 23), 'None')
    
    (may_be_3871, more_types_in_union_3872) = may_be_none(itemsize_3869, None_3870)

    if may_be_3871:

        if more_types_in_union_3872:
            # Runtime conditional SSA (line 2621)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 2625):
        
        # Call to tolist(...): (line 2625)
        # Processing the call keyword arguments (line 2625)
        kwargs_3875 = {}
        # Getting the type of 'obj' (line 2625)
        obj_3873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2625, 18), 'obj', False)
        # Obtaining the member 'tolist' of a type (line 2625)
        tolist_3874 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2625, 18), obj_3873, 'tolist')
        # Calling tolist(args, kwargs) (line 2625)
        tolist_call_result_3876 = invoke(stypy.reporting.localization.Localization(__file__, 2625, 18), tolist_3874, *[], **kwargs_3875)
        
        # Assigning a type to the variable 'obj' (line 2625)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2625, 12), 'obj', tolist_call_result_3876)

        if more_types_in_union_3872:
            # SSA join for if statement (line 2621)
            module_type_store = module_type_store.join_ssa_context()


    
    # SSA join for if statement (line 2620)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'unicode' (line 2628)
    unicode_3877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2628, 7), 'unicode')
    # Testing the type of an if condition (line 2628)
    if_condition_3878 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 2628, 4), unicode_3877)
    # Assigning a type to the variable 'if_condition_3878' (line 2628)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2628, 4), 'if_condition_3878', if_condition_3878)
    # SSA begins for if statement (line 2628)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 2629):
    # Getting the type of 'unicode_' (line 2629)
    unicode__3879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2629, 16), 'unicode_')
    # Assigning a type to the variable 'dtype' (line 2629)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2629, 8), 'dtype', unicode__3879)
    # SSA branch for the else part of an if statement (line 2628)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Name (line 2631):
    # Getting the type of 'string_' (line 2631)
    string__3880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2631, 16), 'string_')
    # Assigning a type to the variable 'dtype' (line 2631)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2631, 8), 'dtype', string__3880)
    # SSA join for if statement (line 2628)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 2633)
    # Getting the type of 'itemsize' (line 2633)
    itemsize_3881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2633, 7), 'itemsize')
    # Getting the type of 'None' (line 2633)
    None_3882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2633, 19), 'None')
    
    (may_be_3883, more_types_in_union_3884) = may_be_none(itemsize_3881, None_3882)

    if may_be_3883:

        if more_types_in_union_3884:
            # Runtime conditional SSA (line 2633)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 2634):
        
        # Call to narray(...): (line 2634)
        # Processing the call arguments (line 2634)
        # Getting the type of 'obj' (line 2634)
        obj_3886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2634, 21), 'obj', False)
        # Processing the call keyword arguments (line 2634)
        # Getting the type of 'dtype' (line 2634)
        dtype_3887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2634, 32), 'dtype', False)
        keyword_3888 = dtype_3887
        # Getting the type of 'order' (line 2634)
        order_3889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2634, 45), 'order', False)
        keyword_3890 = order_3889
        # Getting the type of 'True' (line 2634)
        True_3891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2634, 58), 'True', False)
        keyword_3892 = True_3891
        kwargs_3893 = {'dtype': keyword_3888, 'order': keyword_3890, 'subok': keyword_3892}
        # Getting the type of 'narray' (line 2634)
        narray_3885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2634, 14), 'narray', False)
        # Calling narray(args, kwargs) (line 2634)
        narray_call_result_3894 = invoke(stypy.reporting.localization.Localization(__file__, 2634, 14), narray_3885, *[obj_3886], **kwargs_3893)
        
        # Assigning a type to the variable 'val' (line 2634)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2634, 8), 'val', narray_call_result_3894)

        if more_types_in_union_3884:
            # Runtime conditional SSA for else branch (line 2633)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_3883) or more_types_in_union_3884):
        
        # Assigning a Call to a Name (line 2636):
        
        # Call to narray(...): (line 2636)
        # Processing the call arguments (line 2636)
        # Getting the type of 'obj' (line 2636)
        obj_3896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2636, 21), 'obj', False)
        # Processing the call keyword arguments (line 2636)
        
        # Obtaining an instance of the builtin type 'tuple' (line 2636)
        tuple_3897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2636, 33), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 2636)
        # Adding element type (line 2636)
        # Getting the type of 'dtype' (line 2636)
        dtype_3898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2636, 33), 'dtype', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2636, 33), tuple_3897, dtype_3898)
        # Adding element type (line 2636)
        # Getting the type of 'itemsize' (line 2636)
        itemsize_3899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2636, 40), 'itemsize', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2636, 33), tuple_3897, itemsize_3899)
        
        keyword_3900 = tuple_3897
        # Getting the type of 'order' (line 2636)
        order_3901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2636, 57), 'order', False)
        keyword_3902 = order_3901
        # Getting the type of 'True' (line 2636)
        True_3903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2636, 70), 'True', False)
        keyword_3904 = True_3903
        kwargs_3905 = {'dtype': keyword_3900, 'order': keyword_3902, 'subok': keyword_3904}
        # Getting the type of 'narray' (line 2636)
        narray_3895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2636, 14), 'narray', False)
        # Calling narray(args, kwargs) (line 2636)
        narray_call_result_3906 = invoke(stypy.reporting.localization.Localization(__file__, 2636, 14), narray_3895, *[obj_3896], **kwargs_3905)
        
        # Assigning a type to the variable 'val' (line 2636)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2636, 8), 'val', narray_call_result_3906)

        if (may_be_3883 and more_types_in_union_3884):
            # SSA join for if statement (line 2633)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Call to view(...): (line 2637)
    # Processing the call arguments (line 2637)
    # Getting the type of 'chararray' (line 2637)
    chararray_3909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2637, 20), 'chararray', False)
    # Processing the call keyword arguments (line 2637)
    kwargs_3910 = {}
    # Getting the type of 'val' (line 2637)
    val_3907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2637, 11), 'val', False)
    # Obtaining the member 'view' of a type (line 2637)
    view_3908 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2637, 11), val_3907, 'view')
    # Calling view(args, kwargs) (line 2637)
    view_call_result_3911 = invoke(stypy.reporting.localization.Localization(__file__, 2637, 11), view_3908, *[chararray_3909], **kwargs_3910)
    
    # Assigning a type to the variable 'stypy_return_type' (line 2637)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2637, 4), 'stypy_return_type', view_call_result_3911)
    
    # ################# End of 'array(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'array' in the type store
    # Getting the type of 'stypy_return_type' (line 2478)
    stypy_return_type_3912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2478, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_3912)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'array'
    return stypy_return_type_3912

# Assigning a type to the variable 'array' (line 2478)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2478, 0), 'array', array)

@norecursion
def asarray(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 2640)
    None_3913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2640, 26), 'None')
    # Getting the type of 'None' (line 2640)
    None_3914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2640, 40), 'None')
    # Getting the type of 'None' (line 2640)
    None_3915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2640, 52), 'None')
    defaults = [None_3913, None_3914, None_3915]
    # Create a new context for function 'asarray'
    module_type_store = module_type_store.open_function_context('asarray', 2640, 0, False)
    
    # Passed parameters checking function
    asarray.stypy_localization = localization
    asarray.stypy_type_of_self = None
    asarray.stypy_type_store = module_type_store
    asarray.stypy_function_name = 'asarray'
    asarray.stypy_param_names_list = ['obj', 'itemsize', 'unicode', 'order']
    asarray.stypy_varargs_param_name = None
    asarray.stypy_kwargs_param_name = None
    asarray.stypy_call_defaults = defaults
    asarray.stypy_call_varargs = varargs
    asarray.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'asarray', ['obj', 'itemsize', 'unicode', 'order'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'asarray', localization, ['obj', 'itemsize', 'unicode', 'order'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'asarray(...)' code ##################

    str_3916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2687, (-1)), 'str', "\n    Convert the input to a `chararray`, copying the data only if\n    necessary.\n\n    Versus a regular Numpy array of type `str` or `unicode`, this\n    class adds the following functionality:\n\n      1) values automatically have whitespace removed from the end\n         when indexed\n\n      2) comparison operators automatically remove whitespace from the\n         end when comparing values\n\n      3) vectorized string operations are provided as methods\n         (e.g. `str.endswith`) and infix operators (e.g. ``+``, ``*``,``%``)\n\n    Parameters\n    ----------\n    obj : array of str or unicode-like\n\n    itemsize : int, optional\n        `itemsize` is the number of characters per scalar in the\n        resulting array.  If `itemsize` is None, and `obj` is an\n        object array or a Python list, the `itemsize` will be\n        automatically determined.  If `itemsize` is provided and `obj`\n        is of type str or unicode, then the `obj` string will be\n        chunked into `itemsize` pieces.\n\n    unicode : bool, optional\n        When true, the resulting `chararray` can contain Unicode\n        characters, when false only 8-bit characters.  If unicode is\n        `None` and `obj` is one of the following:\n\n          - a `chararray`,\n          - an ndarray of type `str` or 'unicode`\n          - a Python str or unicode object,\n\n        then the unicode setting of the output array will be\n        automatically determined.\n\n    order : {'C', 'F'}, optional\n        Specify the order of the array.  If order is 'C' (default), then the\n        array will be in C-contiguous order (last-index varies the\n        fastest).  If order is 'F', then the returned array\n        will be in Fortran-contiguous order (first-index varies the\n        fastest).\n    ")
    
    # Call to array(...): (line 2688)
    # Processing the call arguments (line 2688)
    # Getting the type of 'obj' (line 2688)
    obj_3918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2688, 17), 'obj', False)
    # Getting the type of 'itemsize' (line 2688)
    itemsize_3919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2688, 22), 'itemsize', False)
    # Processing the call keyword arguments (line 2688)
    # Getting the type of 'False' (line 2688)
    False_3920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2688, 37), 'False', False)
    keyword_3921 = False_3920
    # Getting the type of 'unicode' (line 2689)
    unicode_3922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2689, 25), 'unicode', False)
    keyword_3923 = unicode_3922
    # Getting the type of 'order' (line 2689)
    order_3924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2689, 40), 'order', False)
    keyword_3925 = order_3924
    kwargs_3926 = {'copy': keyword_3921, 'order': keyword_3925, 'unicode': keyword_3923}
    # Getting the type of 'array' (line 2688)
    array_3917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2688, 11), 'array', False)
    # Calling array(args, kwargs) (line 2688)
    array_call_result_3927 = invoke(stypy.reporting.localization.Localization(__file__, 2688, 11), array_3917, *[obj_3918, itemsize_3919], **kwargs_3926)
    
    # Assigning a type to the variable 'stypy_return_type' (line 2688)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2688, 4), 'stypy_return_type', array_call_result_3927)
    
    # ################# End of 'asarray(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'asarray' in the type store
    # Getting the type of 'stypy_return_type' (line 2640)
    stypy_return_type_3928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2640, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_3928)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'asarray'
    return stypy_return_type_3928

# Assigning a type to the variable 'asarray' (line 2640)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2640, 0), 'asarray', asarray)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
