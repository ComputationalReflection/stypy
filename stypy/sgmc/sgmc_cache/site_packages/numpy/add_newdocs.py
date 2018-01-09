
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: This is only meant to add docs to objects defined in C-extension modules.
3: The purpose is to allow easier editing of the docstrings without
4: requiring a re-compile.
5: 
6: NOTE: Many of the methods of ndarray have corresponding functions.
7:       If you update these docstrings, please keep also the ones in
8:       core/fromnumeric.py, core/defmatrix.py up-to-date.
9: 
10: '''
11: from __future__ import division, absolute_import, print_function
12: 
13: from numpy.lib import add_newdoc
14: 
15: ###############################################################################
16: #
17: # flatiter
18: #
19: # flatiter needs a toplevel description
20: #
21: ###############################################################################
22: 
23: add_newdoc('numpy.core', 'flatiter',
24:     '''
25:     Flat iterator object to iterate over arrays.
26: 
27:     A `flatiter` iterator is returned by ``x.flat`` for any array `x`.
28:     It allows iterating over the array as if it were a 1-D array,
29:     either in a for-loop or by calling its `next` method.
30: 
31:     Iteration is done in row-major, C-style order (the last
32:     index varying the fastest). The iterator can also be indexed using
33:     basic slicing or advanced indexing.
34: 
35:     See Also
36:     --------
37:     ndarray.flat : Return a flat iterator over an array.
38:     ndarray.flatten : Returns a flattened copy of an array.
39: 
40:     Notes
41:     -----
42:     A `flatiter` iterator can not be constructed directly from Python code
43:     by calling the `flatiter` constructor.
44: 
45:     Examples
46:     --------
47:     >>> x = np.arange(6).reshape(2, 3)
48:     >>> fl = x.flat
49:     >>> type(fl)
50:     <type 'numpy.flatiter'>
51:     >>> for item in fl:
52:     ...     print(item)
53:     ...
54:     0
55:     1
56:     2
57:     3
58:     4
59:     5
60: 
61:     >>> fl[2:4]
62:     array([2, 3])
63: 
64:     ''')
65: 
66: # flatiter attributes
67: 
68: add_newdoc('numpy.core', 'flatiter', ('base',
69:     '''
70:     A reference to the array that is iterated over.
71: 
72:     Examples
73:     --------
74:     >>> x = np.arange(5)
75:     >>> fl = x.flat
76:     >>> fl.base is x
77:     True
78: 
79:     '''))
80: 
81: 
82: 
83: add_newdoc('numpy.core', 'flatiter', ('coords',
84:     '''
85:     An N-dimensional tuple of current coordinates.
86: 
87:     Examples
88:     --------
89:     >>> x = np.arange(6).reshape(2, 3)
90:     >>> fl = x.flat
91:     >>> fl.coords
92:     (0, 0)
93:     >>> fl.next()
94:     0
95:     >>> fl.coords
96:     (0, 1)
97: 
98:     '''))
99: 
100: 
101: 
102: add_newdoc('numpy.core', 'flatiter', ('index',
103:     '''
104:     Current flat index into the array.
105: 
106:     Examples
107:     --------
108:     >>> x = np.arange(6).reshape(2, 3)
109:     >>> fl = x.flat
110:     >>> fl.index
111:     0
112:     >>> fl.next()
113:     0
114:     >>> fl.index
115:     1
116: 
117:     '''))
118: 
119: # flatiter functions
120: 
121: add_newdoc('numpy.core', 'flatiter', ('__array__',
122:     '''__array__(type=None) Get array from iterator
123: 
124:     '''))
125: 
126: 
127: add_newdoc('numpy.core', 'flatiter', ('copy',
128:     '''
129:     copy()
130: 
131:     Get a copy of the iterator as a 1-D array.
132: 
133:     Examples
134:     --------
135:     >>> x = np.arange(6).reshape(2, 3)
136:     >>> x
137:     array([[0, 1, 2],
138:            [3, 4, 5]])
139:     >>> fl = x.flat
140:     >>> fl.copy()
141:     array([0, 1, 2, 3, 4, 5])
142: 
143:     '''))
144: 
145: 
146: ###############################################################################
147: #
148: # nditer
149: #
150: ###############################################################################
151: 
152: add_newdoc('numpy.core', 'nditer',
153:     '''
154:     Efficient multi-dimensional iterator object to iterate over arrays.
155:     To get started using this object, see the
156:     :ref:`introductory guide to array iteration <arrays.nditer>`.
157: 
158:     Parameters
159:     ----------
160:     op : ndarray or sequence of array_like
161:         The array(s) to iterate over.
162:     flags : sequence of str, optional
163:         Flags to control the behavior of the iterator.
164: 
165:           * "buffered" enables buffering when required.
166:           * "c_index" causes a C-order index to be tracked.
167:           * "f_index" causes a Fortran-order index to be tracked.
168:           * "multi_index" causes a multi-index, or a tuple of indices
169:             with one per iteration dimension, to be tracked.
170:           * "common_dtype" causes all the operands to be converted to
171:             a common data type, with copying or buffering as necessary.
172:           * "delay_bufalloc" delays allocation of the buffers until
173:             a reset() call is made. Allows "allocate" operands to
174:             be initialized before their values are copied into the buffers.
175:           * "external_loop" causes the `values` given to be
176:             one-dimensional arrays with multiple values instead of
177:             zero-dimensional arrays.
178:           * "grow_inner" allows the `value` array sizes to be made
179:             larger than the buffer size when both "buffered" and
180:             "external_loop" is used.
181:           * "ranged" allows the iterator to be restricted to a sub-range
182:             of the iterindex values.
183:           * "refs_ok" enables iteration of reference types, such as
184:             object arrays.
185:           * "reduce_ok" enables iteration of "readwrite" operands
186:             which are broadcasted, also known as reduction operands.
187:           * "zerosize_ok" allows `itersize` to be zero.
188:     op_flags : list of list of str, optional
189:         This is a list of flags for each operand. At minimum, one of
190:         "readonly", "readwrite", or "writeonly" must be specified.
191: 
192:           * "readonly" indicates the operand will only be read from.
193:           * "readwrite" indicates the operand will be read from and written to.
194:           * "writeonly" indicates the operand will only be written to.
195:           * "no_broadcast" prevents the operand from being broadcasted.
196:           * "contig" forces the operand data to be contiguous.
197:           * "aligned" forces the operand data to be aligned.
198:           * "nbo" forces the operand data to be in native byte order.
199:           * "copy" allows a temporary read-only copy if required.
200:           * "updateifcopy" allows a temporary read-write copy if required.
201:           * "allocate" causes the array to be allocated if it is None
202:             in the `op` parameter.
203:           * "no_subtype" prevents an "allocate" operand from using a subtype.
204:           * "arraymask" indicates that this operand is the mask to use
205:             for selecting elements when writing to operands with the
206:             'writemasked' flag set. The iterator does not enforce this,
207:             but when writing from a buffer back to the array, it only
208:             copies those elements indicated by this mask.
209:           * 'writemasked' indicates that only elements where the chosen
210:             'arraymask' operand is True will be written to.
211:     op_dtypes : dtype or tuple of dtype(s), optional
212:         The required data type(s) of the operands. If copying or buffering
213:         is enabled, the data will be converted to/from their original types.
214:     order : {'C', 'F', 'A', 'K'}, optional
215:         Controls the iteration order. 'C' means C order, 'F' means
216:         Fortran order, 'A' means 'F' order if all the arrays are Fortran
217:         contiguous, 'C' order otherwise, and 'K' means as close to the
218:         order the array elements appear in memory as possible. This also
219:         affects the element memory order of "allocate" operands, as they
220:         are allocated to be compatible with iteration order.
221:         Default is 'K'.
222:     casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
223:         Controls what kind of data casting may occur when making a copy
224:         or buffering.  Setting this to 'unsafe' is not recommended,
225:         as it can adversely affect accumulations.
226: 
227:           * 'no' means the data types should not be cast at all.
228:           * 'equiv' means only byte-order changes are allowed.
229:           * 'safe' means only casts which can preserve values are allowed.
230:           * 'same_kind' means only safe casts or casts within a kind,
231:             like float64 to float32, are allowed.
232:           * 'unsafe' means any data conversions may be done.
233:     op_axes : list of list of ints, optional
234:         If provided, is a list of ints or None for each operands.
235:         The list of axes for an operand is a mapping from the dimensions
236:         of the iterator to the dimensions of the operand. A value of
237:         -1 can be placed for entries, causing that dimension to be
238:         treated as "newaxis".
239:     itershape : tuple of ints, optional
240:         The desired shape of the iterator. This allows "allocate" operands
241:         with a dimension mapped by op_axes not corresponding to a dimension
242:         of a different operand to get a value not equal to 1 for that
243:         dimension.
244:     buffersize : int, optional
245:         When buffering is enabled, controls the size of the temporary
246:         buffers. Set to 0 for the default value.
247: 
248:     Attributes
249:     ----------
250:     dtypes : tuple of dtype(s)
251:         The data types of the values provided in `value`. This may be
252:         different from the operand data types if buffering is enabled.
253:     finished : bool
254:         Whether the iteration over the operands is finished or not.
255:     has_delayed_bufalloc : bool
256:         If True, the iterator was created with the "delay_bufalloc" flag,
257:         and no reset() function was called on it yet.
258:     has_index : bool
259:         If True, the iterator was created with either the "c_index" or
260:         the "f_index" flag, and the property `index` can be used to
261:         retrieve it.
262:     has_multi_index : bool
263:         If True, the iterator was created with the "multi_index" flag,
264:         and the property `multi_index` can be used to retrieve it.
265:     index :
266:         When the "c_index" or "f_index" flag was used, this property
267:         provides access to the index. Raises a ValueError if accessed
268:         and `has_index` is False.
269:     iterationneedsapi : bool
270:         Whether iteration requires access to the Python API, for example
271:         if one of the operands is an object array.
272:     iterindex : int
273:         An index which matches the order of iteration.
274:     itersize : int
275:         Size of the iterator.
276:     itviews :
277:         Structured view(s) of `operands` in memory, matching the reordered
278:         and optimized iterator access pattern.
279:     multi_index :
280:         When the "multi_index" flag was used, this property
281:         provides access to the index. Raises a ValueError if accessed
282:         accessed and `has_multi_index` is False.
283:     ndim : int
284:         The iterator's dimension.
285:     nop : int
286:         The number of iterator operands.
287:     operands : tuple of operand(s)
288:         The array(s) to be iterated over.
289:     shape : tuple of ints
290:         Shape tuple, the shape of the iterator.
291:     value :
292:         Value of `operands` at current iteration. Normally, this is a
293:         tuple of array scalars, but if the flag "external_loop" is used,
294:         it is a tuple of one dimensional arrays.
295: 
296:     Notes
297:     -----
298:     `nditer` supersedes `flatiter`.  The iterator implementation behind
299:     `nditer` is also exposed by the Numpy C API.
300: 
301:     The Python exposure supplies two iteration interfaces, one which follows
302:     the Python iterator protocol, and another which mirrors the C-style
303:     do-while pattern.  The native Python approach is better in most cases, but
304:     if you need the iterator's coordinates or index, use the C-style pattern.
305: 
306:     Examples
307:     --------
308:     Here is how we might write an ``iter_add`` function, using the
309:     Python iterator protocol::
310: 
311:         def iter_add_py(x, y, out=None):
312:             addop = np.add
313:             it = np.nditer([x, y, out], [],
314:                         [['readonly'], ['readonly'], ['writeonly','allocate']])
315:             for (a, b, c) in it:
316:                 addop(a, b, out=c)
317:             return it.operands[2]
318: 
319:     Here is the same function, but following the C-style pattern::
320: 
321:         def iter_add(x, y, out=None):
322:             addop = np.add
323: 
324:             it = np.nditer([x, y, out], [],
325:                         [['readonly'], ['readonly'], ['writeonly','allocate']])
326: 
327:             while not it.finished:
328:                 addop(it[0], it[1], out=it[2])
329:                 it.iternext()
330: 
331:             return it.operands[2]
332: 
333:     Here is an example outer product function::
334: 
335:         def outer_it(x, y, out=None):
336:             mulop = np.multiply
337: 
338:             it = np.nditer([x, y, out], ['external_loop'],
339:                     [['readonly'], ['readonly'], ['writeonly', 'allocate']],
340:                     op_axes=[range(x.ndim)+[-1]*y.ndim,
341:                              [-1]*x.ndim+range(y.ndim),
342:                              None])
343: 
344:             for (a, b, c) in it:
345:                 mulop(a, b, out=c)
346: 
347:             return it.operands[2]
348: 
349:         >>> a = np.arange(2)+1
350:         >>> b = np.arange(3)+1
351:         >>> outer_it(a,b)
352:         array([[1, 2, 3],
353:                [2, 4, 6]])
354: 
355:     Here is an example function which operates like a "lambda" ufunc::
356: 
357:         def luf(lamdaexpr, *args, **kwargs):
358:             "luf(lambdaexpr, op1, ..., opn, out=None, order='K', casting='safe', buffersize=0)"
359:             nargs = len(args)
360:             op = (kwargs.get('out',None),) + args
361:             it = np.nditer(op, ['buffered','external_loop'],
362:                     [['writeonly','allocate','no_broadcast']] +
363:                                     [['readonly','nbo','aligned']]*nargs,
364:                     order=kwargs.get('order','K'),
365:                     casting=kwargs.get('casting','safe'),
366:                     buffersize=kwargs.get('buffersize',0))
367:             while not it.finished:
368:                 it[0] = lamdaexpr(*it[1:])
369:                 it.iternext()
370:             return it.operands[0]
371: 
372:         >>> a = np.arange(5)
373:         >>> b = np.ones(5)
374:         >>> luf(lambda i,j:i*i + j/2, a, b)
375:         array([  0.5,   1.5,   4.5,   9.5,  16.5])
376: 
377:     ''')
378: 
379: # nditer methods
380: 
381: add_newdoc('numpy.core', 'nditer', ('copy',
382:     '''
383:     copy()
384: 
385:     Get a copy of the iterator in its current state.
386: 
387:     Examples
388:     --------
389:     >>> x = np.arange(10)
390:     >>> y = x + 1
391:     >>> it = np.nditer([x, y])
392:     >>> it.next()
393:     (array(0), array(1))
394:     >>> it2 = it.copy()
395:     >>> it2.next()
396:     (array(1), array(2))
397: 
398:     '''))
399: 
400: add_newdoc('numpy.core', 'nditer', ('debug_print',
401:     '''
402:     debug_print()
403: 
404:     Print the current state of the `nditer` instance and debug info to stdout.
405: 
406:     '''))
407: 
408: add_newdoc('numpy.core', 'nditer', ('enable_external_loop',
409:     '''
410:     enable_external_loop()
411: 
412:     When the "external_loop" was not used during construction, but
413:     is desired, this modifies the iterator to behave as if the flag
414:     was specified.
415: 
416:     '''))
417: 
418: add_newdoc('numpy.core', 'nditer', ('iternext',
419:     '''
420:     iternext()
421: 
422:     Check whether iterations are left, and perform a single internal iteration
423:     without returning the result.  Used in the C-style pattern do-while
424:     pattern.  For an example, see `nditer`.
425: 
426:     Returns
427:     -------
428:     iternext : bool
429:         Whether or not there are iterations left.
430: 
431:     '''))
432: 
433: add_newdoc('numpy.core', 'nditer', ('remove_axis',
434:     '''
435:     remove_axis(i)
436: 
437:     Removes axis `i` from the iterator. Requires that the flag "multi_index"
438:     be enabled.
439: 
440:     '''))
441: 
442: add_newdoc('numpy.core', 'nditer', ('remove_multi_index',
443:     '''
444:     remove_multi_index()
445: 
446:     When the "multi_index" flag was specified, this removes it, allowing
447:     the internal iteration structure to be optimized further.
448: 
449:     '''))
450: 
451: add_newdoc('numpy.core', 'nditer', ('reset',
452:     '''
453:     reset()
454: 
455:     Reset the iterator to its initial state.
456: 
457:     '''))
458: 
459: 
460: 
461: ###############################################################################
462: #
463: # broadcast
464: #
465: ###############################################################################
466: 
467: add_newdoc('numpy.core', 'broadcast',
468:     '''
469:     Produce an object that mimics broadcasting.
470: 
471:     Parameters
472:     ----------
473:     in1, in2, ... : array_like
474:         Input parameters.
475: 
476:     Returns
477:     -------
478:     b : broadcast object
479:         Broadcast the input parameters against one another, and
480:         return an object that encapsulates the result.
481:         Amongst others, it has ``shape`` and ``nd`` properties, and
482:         may be used as an iterator.
483: 
484:     Examples
485:     --------
486:     Manually adding two vectors, using broadcasting:
487: 
488:     >>> x = np.array([[1], [2], [3]])
489:     >>> y = np.array([4, 5, 6])
490:     >>> b = np.broadcast(x, y)
491: 
492:     >>> out = np.empty(b.shape)
493:     >>> out.flat = [u+v for (u,v) in b]
494:     >>> out
495:     array([[ 5.,  6.,  7.],
496:            [ 6.,  7.,  8.],
497:            [ 7.,  8.,  9.]])
498: 
499:     Compare against built-in broadcasting:
500: 
501:     >>> x + y
502:     array([[5, 6, 7],
503:            [6, 7, 8],
504:            [7, 8, 9]])
505: 
506:     ''')
507: 
508: # attributes
509: 
510: add_newdoc('numpy.core', 'broadcast', ('index',
511:     '''
512:     current index in broadcasted result
513: 
514:     Examples
515:     --------
516:     >>> x = np.array([[1], [2], [3]])
517:     >>> y = np.array([4, 5, 6])
518:     >>> b = np.broadcast(x, y)
519:     >>> b.index
520:     0
521:     >>> b.next(), b.next(), b.next()
522:     ((1, 4), (1, 5), (1, 6))
523:     >>> b.index
524:     3
525: 
526:     '''))
527: 
528: add_newdoc('numpy.core', 'broadcast', ('iters',
529:     '''
530:     tuple of iterators along ``self``'s "components."
531: 
532:     Returns a tuple of `numpy.flatiter` objects, one for each "component"
533:     of ``self``.
534: 
535:     See Also
536:     --------
537:     numpy.flatiter
538: 
539:     Examples
540:     --------
541:     >>> x = np.array([1, 2, 3])
542:     >>> y = np.array([[4], [5], [6]])
543:     >>> b = np.broadcast(x, y)
544:     >>> row, col = b.iters
545:     >>> row.next(), col.next()
546:     (1, 4)
547: 
548:     '''))
549: 
550: add_newdoc('numpy.core', 'broadcast', ('nd',
551:     '''
552:     Number of dimensions of broadcasted result.
553: 
554:     Examples
555:     --------
556:     >>> x = np.array([1, 2, 3])
557:     >>> y = np.array([[4], [5], [6]])
558:     >>> b = np.broadcast(x, y)
559:     >>> b.nd
560:     2
561: 
562:     '''))
563: 
564: add_newdoc('numpy.core', 'broadcast', ('numiter',
565:     '''
566:     Number of iterators possessed by the broadcasted result.
567: 
568:     Examples
569:     --------
570:     >>> x = np.array([1, 2, 3])
571:     >>> y = np.array([[4], [5], [6]])
572:     >>> b = np.broadcast(x, y)
573:     >>> b.numiter
574:     2
575: 
576:     '''))
577: 
578: add_newdoc('numpy.core', 'broadcast', ('shape',
579:     '''
580:     Shape of broadcasted result.
581: 
582:     Examples
583:     --------
584:     >>> x = np.array([1, 2, 3])
585:     >>> y = np.array([[4], [5], [6]])
586:     >>> b = np.broadcast(x, y)
587:     >>> b.shape
588:     (3, 3)
589: 
590:     '''))
591: 
592: add_newdoc('numpy.core', 'broadcast', ('size',
593:     '''
594:     Total size of broadcasted result.
595: 
596:     Examples
597:     --------
598:     >>> x = np.array([1, 2, 3])
599:     >>> y = np.array([[4], [5], [6]])
600:     >>> b = np.broadcast(x, y)
601:     >>> b.size
602:     9
603: 
604:     '''))
605: 
606: add_newdoc('numpy.core', 'broadcast', ('reset',
607:     '''
608:     reset()
609: 
610:     Reset the broadcasted result's iterator(s).
611: 
612:     Parameters
613:     ----------
614:     None
615: 
616:     Returns
617:     -------
618:     None
619: 
620:     Examples
621:     --------
622:     >>> x = np.array([1, 2, 3])
623:     >>> y = np.array([[4], [5], [6]]
624:     >>> b = np.broadcast(x, y)
625:     >>> b.index
626:     0
627:     >>> b.next(), b.next(), b.next()
628:     ((1, 4), (2, 4), (3, 4))
629:     >>> b.index
630:     3
631:     >>> b.reset()
632:     >>> b.index
633:     0
634: 
635:     '''))
636: 
637: ###############################################################################
638: #
639: # numpy functions
640: #
641: ###############################################################################
642: 
643: add_newdoc('numpy.core.multiarray', 'array',
644:     '''
645:     array(object, dtype=None, copy=True, order=None, subok=False, ndmin=0)
646: 
647:     Create an array.
648: 
649:     Parameters
650:     ----------
651:     object : array_like
652:         An array, any object exposing the array interface, an
653:         object whose __array__ method returns an array, or any
654:         (nested) sequence.
655:     dtype : data-type, optional
656:         The desired data-type for the array.  If not given, then
657:         the type will be determined as the minimum type required
658:         to hold the objects in the sequence.  This argument can only
659:         be used to 'upcast' the array.  For downcasting, use the
660:         .astype(t) method.
661:     copy : bool, optional
662:         If true (default), then the object is copied.  Otherwise, a copy
663:         will only be made if __array__ returns a copy, if obj is a
664:         nested sequence, or if a copy is needed to satisfy any of the other
665:         requirements (`dtype`, `order`, etc.).
666:     order : {'C', 'F', 'A'}, optional
667:         Specify the order of the array.  If order is 'C', then the array
668:         will be in C-contiguous order (last-index varies the fastest).
669:         If order is 'F', then the returned array will be in
670:         Fortran-contiguous order (first-index varies the fastest).
671:         If order is 'A' (default), then the returned array may be
672:         in any order (either C-, Fortran-contiguous, or even discontiguous),
673:         unless a copy is required, in which case it will be C-contiguous.
674:     subok : bool, optional
675:         If True, then sub-classes will be passed-through, otherwise
676:         the returned array will be forced to be a base-class array (default).
677:     ndmin : int, optional
678:         Specifies the minimum number of dimensions that the resulting
679:         array should have.  Ones will be pre-pended to the shape as
680:         needed to meet this requirement.
681: 
682:     Returns
683:     -------
684:     out : ndarray
685:         An array object satisfying the specified requirements.
686: 
687:     See Also
688:     --------
689:     empty, empty_like, zeros, zeros_like, ones, ones_like, fill
690: 
691:     Examples
692:     --------
693:     >>> np.array([1, 2, 3])
694:     array([1, 2, 3])
695: 
696:     Upcasting:
697: 
698:     >>> np.array([1, 2, 3.0])
699:     array([ 1.,  2.,  3.])
700: 
701:     More than one dimension:
702: 
703:     >>> np.array([[1, 2], [3, 4]])
704:     array([[1, 2],
705:            [3, 4]])
706: 
707:     Minimum dimensions 2:
708: 
709:     >>> np.array([1, 2, 3], ndmin=2)
710:     array([[1, 2, 3]])
711: 
712:     Type provided:
713: 
714:     >>> np.array([1, 2, 3], dtype=complex)
715:     array([ 1.+0.j,  2.+0.j,  3.+0.j])
716: 
717:     Data-type consisting of more than one element:
718: 
719:     >>> x = np.array([(1,2),(3,4)],dtype=[('a','<i4'),('b','<i4')])
720:     >>> x['a']
721:     array([1, 3])
722: 
723:     Creating an array from sub-classes:
724: 
725:     >>> np.array(np.mat('1 2; 3 4'))
726:     array([[1, 2],
727:            [3, 4]])
728: 
729:     >>> np.array(np.mat('1 2; 3 4'), subok=True)
730:     matrix([[1, 2],
731:             [3, 4]])
732: 
733:     ''')
734: 
735: add_newdoc('numpy.core.multiarray', 'empty',
736:     '''
737:     empty(shape, dtype=float, order='C')
738: 
739:     Return a new array of given shape and type, without initializing entries.
740: 
741:     Parameters
742:     ----------
743:     shape : int or tuple of int
744:         Shape of the empty array
745:     dtype : data-type, optional
746:         Desired output data-type.
747:     order : {'C', 'F'}, optional
748:         Whether to store multi-dimensional data in row-major
749:         (C-style) or column-major (Fortran-style) order in
750:         memory.
751: 
752:     Returns
753:     -------
754:     out : ndarray
755:         Array of uninitialized (arbitrary) data of the given shape, dtype, and
756:         order.  Object arrays will be initialized to None.
757: 
758:     See Also
759:     --------
760:     empty_like, zeros, ones
761: 
762:     Notes
763:     -----
764:     `empty`, unlike `zeros`, does not set the array values to zero,
765:     and may therefore be marginally faster.  On the other hand, it requires
766:     the user to manually set all the values in the array, and should be
767:     used with caution.
768: 
769:     Examples
770:     --------
771:     >>> np.empty([2, 2])
772:     array([[ -9.74499359e+001,   6.69583040e-309],
773:            [  2.13182611e-314,   3.06959433e-309]])         #random
774: 
775:     >>> np.empty([2, 2], dtype=int)
776:     array([[-1073741821, -1067949133],
777:            [  496041986,    19249760]])                     #random
778: 
779:     ''')
780: 
781: add_newdoc('numpy.core.multiarray', 'empty_like',
782:     '''
783:     empty_like(a, dtype=None, order='K', subok=True)
784: 
785:     Return a new array with the same shape and type as a given array.
786: 
787:     Parameters
788:     ----------
789:     a : array_like
790:         The shape and data-type of `a` define these same attributes of the
791:         returned array.
792:     dtype : data-type, optional
793:         Overrides the data type of the result.
794: 
795:         .. versionadded:: 1.6.0
796:     order : {'C', 'F', 'A', or 'K'}, optional
797:         Overrides the memory layout of the result. 'C' means C-order,
798:         'F' means F-order, 'A' means 'F' if ``a`` is Fortran contiguous,
799:         'C' otherwise. 'K' means match the layout of ``a`` as closely
800:         as possible.
801: 
802:         .. versionadded:: 1.6.0
803:     subok : bool, optional.
804:         If True, then the newly created array will use the sub-class
805:         type of 'a', otherwise it will be a base-class array. Defaults
806:         to True.
807: 
808:     Returns
809:     -------
810:     out : ndarray
811:         Array of uninitialized (arbitrary) data with the same
812:         shape and type as `a`.
813: 
814:     See Also
815:     --------
816:     ones_like : Return an array of ones with shape and type of input.
817:     zeros_like : Return an array of zeros with shape and type of input.
818:     empty : Return a new uninitialized array.
819:     ones : Return a new array setting values to one.
820:     zeros : Return a new array setting values to zero.
821: 
822:     Notes
823:     -----
824:     This function does *not* initialize the returned array; to do that use
825:     `zeros_like` or `ones_like` instead.  It may be marginally faster than
826:     the functions that do set the array values.
827: 
828:     Examples
829:     --------
830:     >>> a = ([1,2,3], [4,5,6])                         # a is array-like
831:     >>> np.empty_like(a)
832:     array([[-1073741821, -1073741821,           3],    #random
833:            [          0,           0, -1073741821]])
834:     >>> a = np.array([[1., 2., 3.],[4.,5.,6.]])
835:     >>> np.empty_like(a)
836:     array([[ -2.00000715e+000,   1.48219694e-323,  -2.00000572e+000],#random
837:            [  4.38791518e-305,  -2.00000715e+000,   4.17269252e-309]])
838: 
839:     ''')
840: 
841: 
842: add_newdoc('numpy.core.multiarray', 'scalar',
843:     '''
844:     scalar(dtype, obj)
845: 
846:     Return a new scalar array of the given type initialized with obj.
847: 
848:     This function is meant mainly for pickle support. `dtype` must be a
849:     valid data-type descriptor. If `dtype` corresponds to an object
850:     descriptor, then `obj` can be any object, otherwise `obj` must be a
851:     string. If `obj` is not given, it will be interpreted as None for object
852:     type and as zeros for all other types.
853: 
854:     ''')
855: 
856: add_newdoc('numpy.core.multiarray', 'zeros',
857:     '''
858:     zeros(shape, dtype=float, order='C')
859: 
860:     Return a new array of given shape and type, filled with zeros.
861: 
862:     Parameters
863:     ----------
864:     shape : int or sequence of ints
865:         Shape of the new array, e.g., ``(2, 3)`` or ``2``.
866:     dtype : data-type, optional
867:         The desired data-type for the array, e.g., `numpy.int8`.  Default is
868:         `numpy.float64`.
869:     order : {'C', 'F'}, optional
870:         Whether to store multidimensional data in C- or Fortran-contiguous
871:         (row- or column-wise) order in memory.
872: 
873:     Returns
874:     -------
875:     out : ndarray
876:         Array of zeros with the given shape, dtype, and order.
877: 
878:     See Also
879:     --------
880:     zeros_like : Return an array of zeros with shape and type of input.
881:     ones_like : Return an array of ones with shape and type of input.
882:     empty_like : Return an empty array with shape and type of input.
883:     ones : Return a new array setting values to one.
884:     empty : Return a new uninitialized array.
885: 
886:     Examples
887:     --------
888:     >>> np.zeros(5)
889:     array([ 0.,  0.,  0.,  0.,  0.])
890: 
891:     >>> np.zeros((5,), dtype=np.int)
892:     array([0, 0, 0, 0, 0])
893: 
894:     >>> np.zeros((2, 1))
895:     array([[ 0.],
896:            [ 0.]])
897: 
898:     >>> s = (2,2)
899:     >>> np.zeros(s)
900:     array([[ 0.,  0.],
901:            [ 0.,  0.]])
902: 
903:     >>> np.zeros((2,), dtype=[('x', 'i4'), ('y', 'i4')]) # custom dtype
904:     array([(0, 0), (0, 0)],
905:           dtype=[('x', '<i4'), ('y', '<i4')])
906: 
907:     ''')
908: 
909: add_newdoc('numpy.core.multiarray', 'count_nonzero',
910:     '''
911:     count_nonzero(a)
912: 
913:     Counts the number of non-zero values in the array ``a``.
914: 
915:     Parameters
916:     ----------
917:     a : array_like
918:         The array for which to count non-zeros.
919: 
920:     Returns
921:     -------
922:     count : int or array of int
923:         Number of non-zero values in the array.
924: 
925:     See Also
926:     --------
927:     nonzero : Return the coordinates of all the non-zero values.
928: 
929:     Examples
930:     --------
931:     >>> np.count_nonzero(np.eye(4))
932:     4
933:     >>> np.count_nonzero([[0,1,7,0,0],[3,0,0,2,19]])
934:     5
935:     ''')
936: 
937: add_newdoc('numpy.core.multiarray', 'set_typeDict',
938:     '''set_typeDict(dict)
939: 
940:     Set the internal dictionary that can look up an array type using a
941:     registered code.
942: 
943:     ''')
944: 
945: add_newdoc('numpy.core.multiarray', 'fromstring',
946:     '''
947:     fromstring(string, dtype=float, count=-1, sep='')
948: 
949:     A new 1-D array initialized from raw binary or text data in a string.
950: 
951:     Parameters
952:     ----------
953:     string : str
954:         A string containing the data.
955:     dtype : data-type, optional
956:         The data type of the array; default: float.  For binary input data,
957:         the data must be in exactly this format.
958:     count : int, optional
959:         Read this number of `dtype` elements from the data.  If this is
960:         negative (the default), the count will be determined from the
961:         length of the data.
962:     sep : str, optional
963:         If not provided or, equivalently, the empty string, the data will
964:         be interpreted as binary data; otherwise, as ASCII text with
965:         decimal numbers.  Also in this latter case, this argument is
966:         interpreted as the string separating numbers in the data; extra
967:         whitespace between elements is also ignored.
968: 
969:     Returns
970:     -------
971:     arr : ndarray
972:         The constructed array.
973: 
974:     Raises
975:     ------
976:     ValueError
977:         If the string is not the correct size to satisfy the requested
978:         `dtype` and `count`.
979: 
980:     See Also
981:     --------
982:     frombuffer, fromfile, fromiter
983: 
984:     Examples
985:     --------
986:     >>> np.fromstring('\\x01\\x02', dtype=np.uint8)
987:     array([1, 2], dtype=uint8)
988:     >>> np.fromstring('1 2', dtype=int, sep=' ')
989:     array([1, 2])
990:     >>> np.fromstring('1, 2', dtype=int, sep=',')
991:     array([1, 2])
992:     >>> np.fromstring('\\x01\\x02\\x03\\x04\\x05', dtype=np.uint8, count=3)
993:     array([1, 2, 3], dtype=uint8)
994: 
995:     ''')
996: 
997: add_newdoc('numpy.core.multiarray', 'fromiter',
998:     '''
999:     fromiter(iterable, dtype, count=-1)
1000: 
1001:     Create a new 1-dimensional array from an iterable object.
1002: 
1003:     Parameters
1004:     ----------
1005:     iterable : iterable object
1006:         An iterable object providing data for the array.
1007:     dtype : data-type
1008:         The data-type of the returned array.
1009:     count : int, optional
1010:         The number of items to read from *iterable*.  The default is -1,
1011:         which means all data is read.
1012: 
1013:     Returns
1014:     -------
1015:     out : ndarray
1016:         The output array.
1017: 
1018:     Notes
1019:     -----
1020:     Specify `count` to improve performance.  It allows ``fromiter`` to
1021:     pre-allocate the output array, instead of resizing it on demand.
1022: 
1023:     Examples
1024:     --------
1025:     >>> iterable = (x*x for x in range(5))
1026:     >>> np.fromiter(iterable, np.float)
1027:     array([  0.,   1.,   4.,   9.,  16.])
1028: 
1029:     ''')
1030: 
1031: add_newdoc('numpy.core.multiarray', 'fromfile',
1032:     '''
1033:     fromfile(file, dtype=float, count=-1, sep='')
1034: 
1035:     Construct an array from data in a text or binary file.
1036: 
1037:     A highly efficient way of reading binary data with a known data-type,
1038:     as well as parsing simply formatted text files.  Data written using the
1039:     `tofile` method can be read using this function.
1040: 
1041:     Parameters
1042:     ----------
1043:     file : file or str
1044:         Open file object or filename.
1045:     dtype : data-type
1046:         Data type of the returned array.
1047:         For binary files, it is used to determine the size and byte-order
1048:         of the items in the file.
1049:     count : int
1050:         Number of items to read. ``-1`` means all items (i.e., the complete
1051:         file).
1052:     sep : str
1053:         Separator between items if file is a text file.
1054:         Empty ("") separator means the file should be treated as binary.
1055:         Spaces (" ") in the separator match zero or more whitespace characters.
1056:         A separator consisting only of spaces must match at least one
1057:         whitespace.
1058: 
1059:     See also
1060:     --------
1061:     load, save
1062:     ndarray.tofile
1063:     loadtxt : More flexible way of loading data from a text file.
1064: 
1065:     Notes
1066:     -----
1067:     Do not rely on the combination of `tofile` and `fromfile` for
1068:     data storage, as the binary files generated are are not platform
1069:     independent.  In particular, no byte-order or data-type information is
1070:     saved.  Data can be stored in the platform independent ``.npy`` format
1071:     using `save` and `load` instead.
1072: 
1073:     Examples
1074:     --------
1075:     Construct an ndarray:
1076: 
1077:     >>> dt = np.dtype([('time', [('min', int), ('sec', int)]),
1078:     ...                ('temp', float)])
1079:     >>> x = np.zeros((1,), dtype=dt)
1080:     >>> x['time']['min'] = 10; x['temp'] = 98.25
1081:     >>> x
1082:     array([((10, 0), 98.25)],
1083:           dtype=[('time', [('min', '<i4'), ('sec', '<i4')]), ('temp', '<f8')])
1084: 
1085:     Save the raw data to disk:
1086: 
1087:     >>> import os
1088:     >>> fname = os.tmpnam()
1089:     >>> x.tofile(fname)
1090: 
1091:     Read the raw data from disk:
1092: 
1093:     >>> np.fromfile(fname, dtype=dt)
1094:     array([((10, 0), 98.25)],
1095:           dtype=[('time', [('min', '<i4'), ('sec', '<i4')]), ('temp', '<f8')])
1096: 
1097:     The recommended way to store and load data:
1098: 
1099:     >>> np.save(fname, x)
1100:     >>> np.load(fname + '.npy')
1101:     array([((10, 0), 98.25)],
1102:           dtype=[('time', [('min', '<i4'), ('sec', '<i4')]), ('temp', '<f8')])
1103: 
1104:     ''')
1105: 
1106: add_newdoc('numpy.core.multiarray', 'frombuffer',
1107:     '''
1108:     frombuffer(buffer, dtype=float, count=-1, offset=0)
1109: 
1110:     Interpret a buffer as a 1-dimensional array.
1111: 
1112:     Parameters
1113:     ----------
1114:     buffer : buffer_like
1115:         An object that exposes the buffer interface.
1116:     dtype : data-type, optional
1117:         Data-type of the returned array; default: float.
1118:     count : int, optional
1119:         Number of items to read. ``-1`` means all data in the buffer.
1120:     offset : int, optional
1121:         Start reading the buffer from this offset; default: 0.
1122: 
1123:     Notes
1124:     -----
1125:     If the buffer has data that is not in machine byte-order, this should
1126:     be specified as part of the data-type, e.g.::
1127: 
1128:       >>> dt = np.dtype(int)
1129:       >>> dt = dt.newbyteorder('>')
1130:       >>> np.frombuffer(buf, dtype=dt)
1131: 
1132:     The data of the resulting array will not be byteswapped, but will be
1133:     interpreted correctly.
1134: 
1135:     Examples
1136:     --------
1137:     >>> s = 'hello world'
1138:     >>> np.frombuffer(s, dtype='S1', count=5, offset=6)
1139:     array(['w', 'o', 'r', 'l', 'd'],
1140:           dtype='|S1')
1141: 
1142:     ''')
1143: 
1144: add_newdoc('numpy.core.multiarray', 'concatenate',
1145:     '''
1146:     concatenate((a1, a2, ...), axis=0)
1147: 
1148:     Join a sequence of arrays along an existing axis.
1149: 
1150:     Parameters
1151:     ----------
1152:     a1, a2, ... : sequence of array_like
1153:         The arrays must have the same shape, except in the dimension
1154:         corresponding to `axis` (the first, by default).
1155:     axis : int, optional
1156:         The axis along which the arrays will be joined.  Default is 0.
1157: 
1158:     Returns
1159:     -------
1160:     res : ndarray
1161:         The concatenated array.
1162: 
1163:     See Also
1164:     --------
1165:     ma.concatenate : Concatenate function that preserves input masks.
1166:     array_split : Split an array into multiple sub-arrays of equal or
1167:                   near-equal size.
1168:     split : Split array into a list of multiple sub-arrays of equal size.
1169:     hsplit : Split array into multiple sub-arrays horizontally (column wise)
1170:     vsplit : Split array into multiple sub-arrays vertically (row wise)
1171:     dsplit : Split array into multiple sub-arrays along the 3rd axis (depth).
1172:     stack : Stack a sequence of arrays along a new axis.
1173:     hstack : Stack arrays in sequence horizontally (column wise)
1174:     vstack : Stack arrays in sequence vertically (row wise)
1175:     dstack : Stack arrays in sequence depth wise (along third dimension)
1176: 
1177:     Notes
1178:     -----
1179:     When one or more of the arrays to be concatenated is a MaskedArray,
1180:     this function will return a MaskedArray object instead of an ndarray,
1181:     but the input masks are *not* preserved. In cases where a MaskedArray
1182:     is expected as input, use the ma.concatenate function from the masked
1183:     array module instead.
1184: 
1185:     Examples
1186:     --------
1187:     >>> a = np.array([[1, 2], [3, 4]])
1188:     >>> b = np.array([[5, 6]])
1189:     >>> np.concatenate((a, b), axis=0)
1190:     array([[1, 2],
1191:            [3, 4],
1192:            [5, 6]])
1193:     >>> np.concatenate((a, b.T), axis=1)
1194:     array([[1, 2, 5],
1195:            [3, 4, 6]])
1196: 
1197:     This function will not preserve masking of MaskedArray inputs.
1198: 
1199:     >>> a = np.ma.arange(3)
1200:     >>> a[1] = np.ma.masked
1201:     >>> b = np.arange(2, 5)
1202:     >>> a
1203:     masked_array(data = [0 -- 2],
1204:                  mask = [False  True False],
1205:            fill_value = 999999)
1206:     >>> b
1207:     array([2, 3, 4])
1208:     >>> np.concatenate([a, b])
1209:     masked_array(data = [0 1 2 2 3 4],
1210:                  mask = False,
1211:            fill_value = 999999)
1212:     >>> np.ma.concatenate([a, b])
1213:     masked_array(data = [0 -- 2 2 3 4],
1214:                  mask = [False  True False False False False],
1215:            fill_value = 999999)
1216: 
1217:     ''')
1218: 
1219: add_newdoc('numpy.core', 'inner',
1220:     '''
1221:     inner(a, b)
1222: 
1223:     Inner product of two arrays.
1224: 
1225:     Ordinary inner product of vectors for 1-D arrays (without complex
1226:     conjugation), in higher dimensions a sum product over the last axes.
1227: 
1228:     Parameters
1229:     ----------
1230:     a, b : array_like
1231:         If `a` and `b` are nonscalar, their last dimensions must match.
1232: 
1233:     Returns
1234:     -------
1235:     out : ndarray
1236:         `out.shape = a.shape[:-1] + b.shape[:-1]`
1237: 
1238:     Raises
1239:     ------
1240:     ValueError
1241:         If the last dimension of `a` and `b` has different size.
1242: 
1243:     See Also
1244:     --------
1245:     tensordot : Sum products over arbitrary axes.
1246:     dot : Generalised matrix product, using second last dimension of `b`.
1247:     einsum : Einstein summation convention.
1248: 
1249:     Notes
1250:     -----
1251:     For vectors (1-D arrays) it computes the ordinary inner-product::
1252: 
1253:         np.inner(a, b) = sum(a[:]*b[:])
1254: 
1255:     More generally, if `ndim(a) = r > 0` and `ndim(b) = s > 0`::
1256: 
1257:         np.inner(a, b) = np.tensordot(a, b, axes=(-1,-1))
1258: 
1259:     or explicitly::
1260: 
1261:         np.inner(a, b)[i0,...,ir-1,j0,...,js-1]
1262:              = sum(a[i0,...,ir-1,:]*b[j0,...,js-1,:])
1263: 
1264:     In addition `a` or `b` may be scalars, in which case::
1265: 
1266:        np.inner(a,b) = a*b
1267: 
1268:     Examples
1269:     --------
1270:     Ordinary inner product for vectors:
1271: 
1272:     >>> a = np.array([1,2,3])
1273:     >>> b = np.array([0,1,0])
1274:     >>> np.inner(a, b)
1275:     2
1276: 
1277:     A multidimensional example:
1278: 
1279:     >>> a = np.arange(24).reshape((2,3,4))
1280:     >>> b = np.arange(4)
1281:     >>> np.inner(a, b)
1282:     array([[ 14,  38,  62],
1283:            [ 86, 110, 134]])
1284: 
1285:     An example where `b` is a scalar:
1286: 
1287:     >>> np.inner(np.eye(2), 7)
1288:     array([[ 7.,  0.],
1289:            [ 0.,  7.]])
1290: 
1291:     ''')
1292: 
1293: add_newdoc('numpy.core', 'fastCopyAndTranspose',
1294:     '''_fastCopyAndTranspose(a)''')
1295: 
1296: add_newdoc('numpy.core.multiarray', 'correlate',
1297:     '''cross_correlate(a,v, mode=0)''')
1298: 
1299: add_newdoc('numpy.core.multiarray', 'arange',
1300:     '''
1301:     arange([start,] stop[, step,], dtype=None)
1302: 
1303:     Return evenly spaced values within a given interval.
1304: 
1305:     Values are generated within the half-open interval ``[start, stop)``
1306:     (in other words, the interval including `start` but excluding `stop`).
1307:     For integer arguments the function is equivalent to the Python built-in
1308:     `range <http://docs.python.org/lib/built-in-funcs.html>`_ function,
1309:     but returns an ndarray rather than a list.
1310: 
1311:     When using a non-integer step, such as 0.1, the results will often not
1312:     be consistent.  It is better to use ``linspace`` for these cases.
1313: 
1314:     Parameters
1315:     ----------
1316:     start : number, optional
1317:         Start of interval.  The interval includes this value.  The default
1318:         start value is 0.
1319:     stop : number
1320:         End of interval.  The interval does not include this value, except
1321:         in some cases where `step` is not an integer and floating point
1322:         round-off affects the length of `out`.
1323:     step : number, optional
1324:         Spacing between values.  For any output `out`, this is the distance
1325:         between two adjacent values, ``out[i+1] - out[i]``.  The default
1326:         step size is 1.  If `step` is specified, `start` must also be given.
1327:     dtype : dtype
1328:         The type of the output array.  If `dtype` is not given, infer the data
1329:         type from the other input arguments.
1330: 
1331:     Returns
1332:     -------
1333:     arange : ndarray
1334:         Array of evenly spaced values.
1335: 
1336:         For floating point arguments, the length of the result is
1337:         ``ceil((stop - start)/step)``.  Because of floating point overflow,
1338:         this rule may result in the last element of `out` being greater
1339:         than `stop`.
1340: 
1341:     See Also
1342:     --------
1343:     linspace : Evenly spaced numbers with careful handling of endpoints.
1344:     ogrid: Arrays of evenly spaced numbers in N-dimensions.
1345:     mgrid: Grid-shaped arrays of evenly spaced numbers in N-dimensions.
1346: 
1347:     Examples
1348:     --------
1349:     >>> np.arange(3)
1350:     array([0, 1, 2])
1351:     >>> np.arange(3.0)
1352:     array([ 0.,  1.,  2.])
1353:     >>> np.arange(3,7)
1354:     array([3, 4, 5, 6])
1355:     >>> np.arange(3,7,2)
1356:     array([3, 5])
1357: 
1358:     ''')
1359: 
1360: add_newdoc('numpy.core.multiarray', '_get_ndarray_c_version',
1361:     '''_get_ndarray_c_version()
1362: 
1363:     Return the compile time NDARRAY_VERSION number.
1364: 
1365:     ''')
1366: 
1367: add_newdoc('numpy.core.multiarray', '_reconstruct',
1368:     '''_reconstruct(subtype, shape, dtype)
1369: 
1370:     Construct an empty array. Used by Pickles.
1371: 
1372:     ''')
1373: 
1374: 
1375: add_newdoc('numpy.core.multiarray', 'set_string_function',
1376:     '''
1377:     set_string_function(f, repr=1)
1378: 
1379:     Internal method to set a function to be used when pretty printing arrays.
1380: 
1381:     ''')
1382: 
1383: add_newdoc('numpy.core.multiarray', 'set_numeric_ops',
1384:     '''
1385:     set_numeric_ops(op1=func1, op2=func2, ...)
1386: 
1387:     Set numerical operators for array objects.
1388: 
1389:     Parameters
1390:     ----------
1391:     op1, op2, ... : callable
1392:         Each ``op = func`` pair describes an operator to be replaced.
1393:         For example, ``add = lambda x, y: np.add(x, y) % 5`` would replace
1394:         addition by modulus 5 addition.
1395: 
1396:     Returns
1397:     -------
1398:     saved_ops : list of callables
1399:         A list of all operators, stored before making replacements.
1400: 
1401:     Notes
1402:     -----
1403:     .. WARNING::
1404:        Use with care!  Incorrect usage may lead to memory errors.
1405: 
1406:     A function replacing an operator cannot make use of that operator.
1407:     For example, when replacing add, you may not use ``+``.  Instead,
1408:     directly call ufuncs.
1409: 
1410:     Examples
1411:     --------
1412:     >>> def add_mod5(x, y):
1413:     ...     return np.add(x, y) % 5
1414:     ...
1415:     >>> old_funcs = np.set_numeric_ops(add=add_mod5)
1416: 
1417:     >>> x = np.arange(12).reshape((3, 4))
1418:     >>> x + x
1419:     array([[0, 2, 4, 1],
1420:            [3, 0, 2, 4],
1421:            [1, 3, 0, 2]])
1422: 
1423:     >>> ignore = np.set_numeric_ops(**old_funcs) # restore operators
1424: 
1425:     ''')
1426: 
1427: add_newdoc('numpy.core.multiarray', 'where',
1428:     '''
1429:     where(condition, [x, y])
1430: 
1431:     Return elements, either from `x` or `y`, depending on `condition`.
1432: 
1433:     If only `condition` is given, return ``condition.nonzero()``.
1434: 
1435:     Parameters
1436:     ----------
1437:     condition : array_like, bool
1438:         When True, yield `x`, otherwise yield `y`.
1439:     x, y : array_like, optional
1440:         Values from which to choose. `x` and `y` need to have the same
1441:         shape as `condition`.
1442: 
1443:     Returns
1444:     -------
1445:     out : ndarray or tuple of ndarrays
1446:         If both `x` and `y` are specified, the output array contains
1447:         elements of `x` where `condition` is True, and elements from
1448:         `y` elsewhere.
1449: 
1450:         If only `condition` is given, return the tuple
1451:         ``condition.nonzero()``, the indices where `condition` is True.
1452: 
1453:     See Also
1454:     --------
1455:     nonzero, choose
1456: 
1457:     Notes
1458:     -----
1459:     If `x` and `y` are given and input arrays are 1-D, `where` is
1460:     equivalent to::
1461: 
1462:         [xv if c else yv for (c,xv,yv) in zip(condition,x,y)]
1463: 
1464:     Examples
1465:     --------
1466:     >>> np.where([[True, False], [True, True]],
1467:     ...          [[1, 2], [3, 4]],
1468:     ...          [[9, 8], [7, 6]])
1469:     array([[1, 8],
1470:            [3, 4]])
1471: 
1472:     >>> np.where([[0, 1], [1, 0]])
1473:     (array([0, 1]), array([1, 0]))
1474: 
1475:     >>> x = np.arange(9.).reshape(3, 3)
1476:     >>> np.where( x > 5 )
1477:     (array([2, 2, 2]), array([0, 1, 2]))
1478:     >>> x[np.where( x > 3.0 )]               # Note: result is 1D.
1479:     array([ 4.,  5.,  6.,  7.,  8.])
1480:     >>> np.where(x < 5, x, -1)               # Note: broadcasting.
1481:     array([[ 0.,  1.,  2.],
1482:            [ 3.,  4., -1.],
1483:            [-1., -1., -1.]])
1484: 
1485:     Find the indices of elements of `x` that are in `goodvalues`.
1486: 
1487:     >>> goodvalues = [3, 4, 7]
1488:     >>> ix = np.in1d(x.ravel(), goodvalues).reshape(x.shape)
1489:     >>> ix
1490:     array([[False, False, False],
1491:            [ True,  True, False],
1492:            [False,  True, False]], dtype=bool)
1493:     >>> np.where(ix)
1494:     (array([1, 1, 2]), array([0, 1, 1]))
1495: 
1496:     ''')
1497: 
1498: 
1499: add_newdoc('numpy.core.multiarray', 'lexsort',
1500:     '''
1501:     lexsort(keys, axis=-1)
1502: 
1503:     Perform an indirect sort using a sequence of keys.
1504: 
1505:     Given multiple sorting keys, which can be interpreted as columns in a
1506:     spreadsheet, lexsort returns an array of integer indices that describes
1507:     the sort order by multiple columns. The last key in the sequence is used
1508:     for the primary sort order, the second-to-last key for the secondary sort
1509:     order, and so on. The keys argument must be a sequence of objects that
1510:     can be converted to arrays of the same shape. If a 2D array is provided
1511:     for the keys argument, it's rows are interpreted as the sorting keys and
1512:     sorting is according to the last row, second last row etc.
1513: 
1514:     Parameters
1515:     ----------
1516:     keys : (k, N) array or tuple containing k (N,)-shaped sequences
1517:         The `k` different "columns" to be sorted.  The last column (or row if
1518:         `keys` is a 2D array) is the primary sort key.
1519:     axis : int, optional
1520:         Axis to be indirectly sorted.  By default, sort over the last axis.
1521: 
1522:     Returns
1523:     -------
1524:     indices : (N,) ndarray of ints
1525:         Array of indices that sort the keys along the specified axis.
1526: 
1527:     See Also
1528:     --------
1529:     argsort : Indirect sort.
1530:     ndarray.sort : In-place sort.
1531:     sort : Return a sorted copy of an array.
1532: 
1533:     Examples
1534:     --------
1535:     Sort names: first by surname, then by name.
1536: 
1537:     >>> surnames =    ('Hertz',    'Galilei', 'Hertz')
1538:     >>> first_names = ('Heinrich', 'Galileo', 'Gustav')
1539:     >>> ind = np.lexsort((first_names, surnames))
1540:     >>> ind
1541:     array([1, 2, 0])
1542: 
1543:     >>> [surnames[i] + ", " + first_names[i] for i in ind]
1544:     ['Galilei, Galileo', 'Hertz, Gustav', 'Hertz, Heinrich']
1545: 
1546:     Sort two columns of numbers:
1547: 
1548:     >>> a = [1,5,1,4,3,4,4] # First column
1549:     >>> b = [9,4,0,4,0,2,1] # Second column
1550:     >>> ind = np.lexsort((b,a)) # Sort by a, then by b
1551:     >>> print(ind)
1552:     [2 0 4 6 5 3 1]
1553: 
1554:     >>> [(a[i],b[i]) for i in ind]
1555:     [(1, 0), (1, 9), (3, 0), (4, 1), (4, 2), (4, 4), (5, 4)]
1556: 
1557:     Note that sorting is first according to the elements of ``a``.
1558:     Secondary sorting is according to the elements of ``b``.
1559: 
1560:     A normal ``argsort`` would have yielded:
1561: 
1562:     >>> [(a[i],b[i]) for i in np.argsort(a)]
1563:     [(1, 9), (1, 0), (3, 0), (4, 4), (4, 2), (4, 1), (5, 4)]
1564: 
1565:     Structured arrays are sorted lexically by ``argsort``:
1566: 
1567:     >>> x = np.array([(1,9), (5,4), (1,0), (4,4), (3,0), (4,2), (4,1)],
1568:     ...              dtype=np.dtype([('x', int), ('y', int)]))
1569: 
1570:     >>> np.argsort(x) # or np.argsort(x, order=('x', 'y'))
1571:     array([2, 0, 4, 6, 5, 3, 1])
1572: 
1573:     ''')
1574: 
1575: add_newdoc('numpy.core.multiarray', 'can_cast',
1576:     '''
1577:     can_cast(from, totype, casting = 'safe')
1578: 
1579:     Returns True if cast between data types can occur according to the
1580:     casting rule.  If from is a scalar or array scalar, also returns
1581:     True if the scalar value can be cast without overflow or truncation
1582:     to an integer.
1583: 
1584:     Parameters
1585:     ----------
1586:     from : dtype, dtype specifier, scalar, or array
1587:         Data type, scalar, or array to cast from.
1588:     totype : dtype or dtype specifier
1589:         Data type to cast to.
1590:     casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
1591:         Controls what kind of data casting may occur.
1592: 
1593:           * 'no' means the data types should not be cast at all.
1594:           * 'equiv' means only byte-order changes are allowed.
1595:           * 'safe' means only casts which can preserve values are allowed.
1596:           * 'same_kind' means only safe casts or casts within a kind,
1597:             like float64 to float32, are allowed.
1598:           * 'unsafe' means any data conversions may be done.
1599: 
1600:     Returns
1601:     -------
1602:     out : bool
1603:         True if cast can occur according to the casting rule.
1604: 
1605:     Notes
1606:     -----
1607:     Starting in NumPy 1.9, can_cast function now returns False in 'safe'
1608:     casting mode for integer/float dtype and string dtype if the string dtype
1609:     length is not long enough to store the max integer/float value converted
1610:     to a string. Previously can_cast in 'safe' mode returned True for
1611:     integer/float dtype and a string dtype of any length.
1612: 
1613:     See also
1614:     --------
1615:     dtype, result_type
1616: 
1617:     Examples
1618:     --------
1619:     Basic examples
1620: 
1621:     >>> np.can_cast(np.int32, np.int64)
1622:     True
1623:     >>> np.can_cast(np.float64, np.complex)
1624:     True
1625:     >>> np.can_cast(np.complex, np.float)
1626:     False
1627: 
1628:     >>> np.can_cast('i8', 'f8')
1629:     True
1630:     >>> np.can_cast('i8', 'f4')
1631:     False
1632:     >>> np.can_cast('i4', 'S4')
1633:     False
1634: 
1635:     Casting scalars
1636: 
1637:     >>> np.can_cast(100, 'i1')
1638:     True
1639:     >>> np.can_cast(150, 'i1')
1640:     False
1641:     >>> np.can_cast(150, 'u1')
1642:     True
1643: 
1644:     >>> np.can_cast(3.5e100, np.float32)
1645:     False
1646:     >>> np.can_cast(1000.0, np.float32)
1647:     True
1648: 
1649:     Array scalar checks the value, array does not
1650: 
1651:     >>> np.can_cast(np.array(1000.0), np.float32)
1652:     True
1653:     >>> np.can_cast(np.array([1000.0]), np.float32)
1654:     False
1655: 
1656:     Using the casting rules
1657: 
1658:     >>> np.can_cast('i8', 'i8', 'no')
1659:     True
1660:     >>> np.can_cast('<i8', '>i8', 'no')
1661:     False
1662: 
1663:     >>> np.can_cast('<i8', '>i8', 'equiv')
1664:     True
1665:     >>> np.can_cast('<i4', '>i8', 'equiv')
1666:     False
1667: 
1668:     >>> np.can_cast('<i4', '>i8', 'safe')
1669:     True
1670:     >>> np.can_cast('<i8', '>i4', 'safe')
1671:     False
1672: 
1673:     >>> np.can_cast('<i8', '>i4', 'same_kind')
1674:     True
1675:     >>> np.can_cast('<i8', '>u4', 'same_kind')
1676:     False
1677: 
1678:     >>> np.can_cast('<i8', '>u4', 'unsafe')
1679:     True
1680: 
1681:     ''')
1682: 
1683: add_newdoc('numpy.core.multiarray', 'promote_types',
1684:     '''
1685:     promote_types(type1, type2)
1686: 
1687:     Returns the data type with the smallest size and smallest scalar
1688:     kind to which both ``type1`` and ``type2`` may be safely cast.
1689:     The returned data type is always in native byte order.
1690: 
1691:     This function is symmetric and associative.
1692: 
1693:     Parameters
1694:     ----------
1695:     type1 : dtype or dtype specifier
1696:         First data type.
1697:     type2 : dtype or dtype specifier
1698:         Second data type.
1699: 
1700:     Returns
1701:     -------
1702:     out : dtype
1703:         The promoted data type.
1704: 
1705:     Notes
1706:     -----
1707:     .. versionadded:: 1.6.0
1708: 
1709:     Starting in NumPy 1.9, promote_types function now returns a valid string
1710:     length when given an integer or float dtype as one argument and a string
1711:     dtype as another argument. Previously it always returned the input string
1712:     dtype, even if it wasn't long enough to store the max integer/float value
1713:     converted to a string.
1714: 
1715:     See Also
1716:     --------
1717:     result_type, dtype, can_cast
1718: 
1719:     Examples
1720:     --------
1721:     >>> np.promote_types('f4', 'f8')
1722:     dtype('float64')
1723: 
1724:     >>> np.promote_types('i8', 'f4')
1725:     dtype('float64')
1726: 
1727:     >>> np.promote_types('>i8', '<c8')
1728:     dtype('complex128')
1729: 
1730:     >>> np.promote_types('i4', 'S8')
1731:     dtype('S11')
1732: 
1733:     ''')
1734: 
1735: add_newdoc('numpy.core.multiarray', 'min_scalar_type',
1736:     '''
1737:     min_scalar_type(a)
1738: 
1739:     For scalar ``a``, returns the data type with the smallest size
1740:     and smallest scalar kind which can hold its value.  For non-scalar
1741:     array ``a``, returns the vector's dtype unmodified.
1742: 
1743:     Floating point values are not demoted to integers,
1744:     and complex values are not demoted to floats.
1745: 
1746:     Parameters
1747:     ----------
1748:     a : scalar or array_like
1749:         The value whose minimal data type is to be found.
1750: 
1751:     Returns
1752:     -------
1753:     out : dtype
1754:         The minimal data type.
1755: 
1756:     Notes
1757:     -----
1758:     .. versionadded:: 1.6.0
1759: 
1760:     See Also
1761:     --------
1762:     result_type, promote_types, dtype, can_cast
1763: 
1764:     Examples
1765:     --------
1766:     >>> np.min_scalar_type(10)
1767:     dtype('uint8')
1768: 
1769:     >>> np.min_scalar_type(-260)
1770:     dtype('int16')
1771: 
1772:     >>> np.min_scalar_type(3.1)
1773:     dtype('float16')
1774: 
1775:     >>> np.min_scalar_type(1e50)
1776:     dtype('float64')
1777: 
1778:     >>> np.min_scalar_type(np.arange(4,dtype='f8'))
1779:     dtype('float64')
1780: 
1781:     ''')
1782: 
1783: add_newdoc('numpy.core.multiarray', 'result_type',
1784:     '''
1785:     result_type(*arrays_and_dtypes)
1786: 
1787:     Returns the type that results from applying the NumPy
1788:     type promotion rules to the arguments.
1789: 
1790:     Type promotion in NumPy works similarly to the rules in languages
1791:     like C++, with some slight differences.  When both scalars and
1792:     arrays are used, the array's type takes precedence and the actual value
1793:     of the scalar is taken into account.
1794: 
1795:     For example, calculating 3*a, where a is an array of 32-bit floats,
1796:     intuitively should result in a 32-bit float output.  If the 3 is a
1797:     32-bit integer, the NumPy rules indicate it can't convert losslessly
1798:     into a 32-bit float, so a 64-bit float should be the result type.
1799:     By examining the value of the constant, '3', we see that it fits in
1800:     an 8-bit integer, which can be cast losslessly into the 32-bit float.
1801: 
1802:     Parameters
1803:     ----------
1804:     arrays_and_dtypes : list of arrays and dtypes
1805:         The operands of some operation whose result type is needed.
1806: 
1807:     Returns
1808:     -------
1809:     out : dtype
1810:         The result type.
1811: 
1812:     See also
1813:     --------
1814:     dtype, promote_types, min_scalar_type, can_cast
1815: 
1816:     Notes
1817:     -----
1818:     .. versionadded:: 1.6.0
1819: 
1820:     The specific algorithm used is as follows.
1821: 
1822:     Categories are determined by first checking which of boolean,
1823:     integer (int/uint), or floating point (float/complex) the maximum
1824:     kind of all the arrays and the scalars are.
1825: 
1826:     If there are only scalars or the maximum category of the scalars
1827:     is higher than the maximum category of the arrays,
1828:     the data types are combined with :func:`promote_types`
1829:     to produce the return value.
1830: 
1831:     Otherwise, `min_scalar_type` is called on each array, and
1832:     the resulting data types are all combined with :func:`promote_types`
1833:     to produce the return value.
1834: 
1835:     The set of int values is not a subset of the uint values for types
1836:     with the same number of bits, something not reflected in
1837:     :func:`min_scalar_type`, but handled as a special case in `result_type`.
1838: 
1839:     Examples
1840:     --------
1841:     >>> np.result_type(3, np.arange(7, dtype='i1'))
1842:     dtype('int8')
1843: 
1844:     >>> np.result_type('i4', 'c8')
1845:     dtype('complex128')
1846: 
1847:     >>> np.result_type(3.0, -2)
1848:     dtype('float64')
1849: 
1850:     ''')
1851: 
1852: add_newdoc('numpy.core.multiarray', 'newbuffer',
1853:     '''
1854:     newbuffer(size)
1855: 
1856:     Return a new uninitialized buffer object.
1857: 
1858:     Parameters
1859:     ----------
1860:     size : int
1861:         Size in bytes of returned buffer object.
1862: 
1863:     Returns
1864:     -------
1865:     newbuffer : buffer object
1866:         Returned, uninitialized buffer object of `size` bytes.
1867: 
1868:     ''')
1869: 
1870: add_newdoc('numpy.core.multiarray', 'getbuffer',
1871:     '''
1872:     getbuffer(obj [,offset[, size]])
1873: 
1874:     Create a buffer object from the given object referencing a slice of
1875:     length size starting at offset.
1876: 
1877:     Default is the entire buffer. A read-write buffer is attempted followed
1878:     by a read-only buffer.
1879: 
1880:     Parameters
1881:     ----------
1882:     obj : object
1883: 
1884:     offset : int, optional
1885: 
1886:     size : int, optional
1887: 
1888:     Returns
1889:     -------
1890:     buffer_obj : buffer
1891: 
1892:     Examples
1893:     --------
1894:     >>> buf = np.getbuffer(np.ones(5), 1, 3)
1895:     >>> len(buf)
1896:     3
1897:     >>> buf[0]
1898:     '\\x00'
1899:     >>> buf
1900:     <read-write buffer for 0x8af1e70, size 3, offset 1 at 0x8ba4ec0>
1901: 
1902:     ''')
1903: 
1904: add_newdoc('numpy.core', 'dot',
1905:     '''
1906:     dot(a, b, out=None)
1907: 
1908:     Dot product of two arrays.
1909: 
1910:     For 2-D arrays it is equivalent to matrix multiplication, and for 1-D
1911:     arrays to inner product of vectors (without complex conjugation). For
1912:     N dimensions it is a sum product over the last axis of `a` and
1913:     the second-to-last of `b`::
1914: 
1915:         dot(a, b)[i,j,k,m] = sum(a[i,j,:] * b[k,:,m])
1916: 
1917:     Parameters
1918:     ----------
1919:     a : array_like
1920:         First argument.
1921:     b : array_like
1922:         Second argument.
1923:     out : ndarray, optional
1924:         Output argument. This must have the exact kind that would be returned
1925:         if it was not used. In particular, it must have the right type, must be
1926:         C-contiguous, and its dtype must be the dtype that would be returned
1927:         for `dot(a,b)`. This is a performance feature. Therefore, if these
1928:         conditions are not met, an exception is raised, instead of attempting
1929:         to be flexible.
1930: 
1931:     Returns
1932:     -------
1933:     output : ndarray
1934:         Returns the dot product of `a` and `b`.  If `a` and `b` are both
1935:         scalars or both 1-D arrays then a scalar is returned; otherwise
1936:         an array is returned.
1937:         If `out` is given, then it is returned.
1938: 
1939:     Raises
1940:     ------
1941:     ValueError
1942:         If the last dimension of `a` is not the same size as
1943:         the second-to-last dimension of `b`.
1944: 
1945:     See Also
1946:     --------
1947:     vdot : Complex-conjugating dot product.
1948:     tensordot : Sum products over arbitrary axes.
1949:     einsum : Einstein summation convention.
1950:     matmul : '@' operator as method with out parameter.
1951: 
1952:     Examples
1953:     --------
1954:     >>> np.dot(3, 4)
1955:     12
1956: 
1957:     Neither argument is complex-conjugated:
1958: 
1959:     >>> np.dot([2j, 3j], [2j, 3j])
1960:     (-13+0j)
1961: 
1962:     For 2-D arrays it is the matrix product:
1963: 
1964:     >>> a = [[1, 0], [0, 1]]
1965:     >>> b = [[4, 1], [2, 2]]
1966:     >>> np.dot(a, b)
1967:     array([[4, 1],
1968:            [2, 2]])
1969: 
1970:     >>> a = np.arange(3*4*5*6).reshape((3,4,5,6))
1971:     >>> b = np.arange(3*4*5*6)[::-1].reshape((5,4,6,3))
1972:     >>> np.dot(a, b)[2,3,2,1,2,2]
1973:     499128
1974:     >>> sum(a[2,3,2,:] * b[1,2,:,2])
1975:     499128
1976: 
1977:     ''')
1978: 
1979: add_newdoc('numpy.core', 'matmul',
1980:     '''
1981:     matmul(a, b, out=None)
1982: 
1983:     Matrix product of two arrays.
1984: 
1985:     The behavior depends on the arguments in the following way.
1986: 
1987:     - If both arguments are 2-D they are multiplied like conventional
1988:       matrices.
1989:     - If either argument is N-D, N > 2, it is treated as a stack of
1990:       matrices residing in the last two indexes and broadcast accordingly.
1991:     - If the first argument is 1-D, it is promoted to a matrix by
1992:       prepending a 1 to its dimensions. After matrix multiplication
1993:       the prepended 1 is removed.
1994:     - If the second argument is 1-D, it is promoted to a matrix by
1995:       appending a 1 to its dimensions. After matrix multiplication
1996:       the appended 1 is removed.
1997: 
1998:     Multiplication by a scalar is not allowed, use ``*`` instead. Note that
1999:     multiplying a stack of matrices with a vector will result in a stack of
2000:     vectors, but matmul will not recognize it as such.
2001: 
2002:     ``matmul`` differs from ``dot`` in two important ways.
2003: 
2004:     - Multiplication by scalars is not allowed.
2005:     - Stacks of matrices are broadcast together as if the matrices
2006:       were elements.
2007: 
2008:     .. warning::
2009:        This function is preliminary and included in Numpy 1.10 for testing
2010:        and documentation. Its semantics will not change, but the number and
2011:        order of the optional arguments will.
2012: 
2013:     .. versionadded:: 1.10.0
2014: 
2015:     Parameters
2016:     ----------
2017:     a : array_like
2018:         First argument.
2019:     b : array_like
2020:         Second argument.
2021:     out : ndarray, optional
2022:         Output argument. This must have the exact kind that would be returned
2023:         if it was not used. In particular, it must have the right type, must be
2024:         C-contiguous, and its dtype must be the dtype that would be returned
2025:         for `dot(a,b)`. This is a performance feature. Therefore, if these
2026:         conditions are not met, an exception is raised, instead of attempting
2027:         to be flexible.
2028: 
2029:     Returns
2030:     -------
2031:     output : ndarray
2032:         Returns the dot product of `a` and `b`.  If `a` and `b` are both
2033:         1-D arrays then a scalar is returned; otherwise an array is
2034:         returned.  If `out` is given, then it is returned.
2035: 
2036:     Raises
2037:     ------
2038:     ValueError
2039:         If the last dimension of `a` is not the same size as
2040:         the second-to-last dimension of `b`.
2041: 
2042:         If scalar value is passed.
2043: 
2044:     See Also
2045:     --------
2046:     vdot : Complex-conjugating dot product.
2047:     tensordot : Sum products over arbitrary axes.
2048:     einsum : Einstein summation convention.
2049:     dot : alternative matrix product with different broadcasting rules.
2050: 
2051:     Notes
2052:     -----
2053:     The matmul function implements the semantics of the `@` operator introduced
2054:     in Python 3.5 following PEP465.
2055: 
2056:     Examples
2057:     --------
2058:     For 2-D arrays it is the matrix product:
2059: 
2060:     >>> a = [[1, 0], [0, 1]]
2061:     >>> b = [[4, 1], [2, 2]]
2062:     >>> np.matmul(a, b)
2063:     array([[4, 1],
2064:            [2, 2]])
2065: 
2066:     For 2-D mixed with 1-D, the result is the usual.
2067: 
2068:     >>> a = [[1, 0], [0, 1]]
2069:     >>> b = [1, 2]
2070:     >>> np.matmul(a, b)
2071:     array([1, 2])
2072:     >>> np.matmul(b, a)
2073:     array([1, 2])
2074: 
2075: 
2076:     Broadcasting is conventional for stacks of arrays
2077: 
2078:     >>> a = np.arange(2*2*4).reshape((2,2,4))
2079:     >>> b = np.arange(2*2*4).reshape((2,4,2))
2080:     >>> np.matmul(a,b).shape
2081:     (2, 2, 2)
2082:     >>> np.matmul(a,b)[0,1,1]
2083:     98
2084:     >>> sum(a[0,1,:] * b[0,:,1])
2085:     98
2086: 
2087:     Vector, vector returns the scalar inner product, but neither argument
2088:     is complex-conjugated:
2089: 
2090:     >>> np.matmul([2j, 3j], [2j, 3j])
2091:     (-13+0j)
2092: 
2093:     Scalar multiplication raises an error.
2094: 
2095:     >>> np.matmul([1,2], 3)
2096:     Traceback (most recent call last):
2097:     ...
2098:     ValueError: Scalar operands are not allowed, use '*' instead
2099: 
2100:     ''')
2101: 
2102: 
2103: add_newdoc('numpy.core', 'einsum',
2104:     '''
2105:     einsum(subscripts, *operands, out=None, dtype=None, order='K', casting='safe')
2106: 
2107:     Evaluates the Einstein summation convention on the operands.
2108: 
2109:     Using the Einstein summation convention, many common multi-dimensional
2110:     array operations can be represented in a simple fashion.  This function
2111:     provides a way to compute such summations. The best way to understand this
2112:     function is to try the examples below, which show how many common NumPy
2113:     functions can be implemented as calls to `einsum`.
2114: 
2115:     Parameters
2116:     ----------
2117:     subscripts : str
2118:         Specifies the subscripts for summation.
2119:     operands : list of array_like
2120:         These are the arrays for the operation.
2121:     out : ndarray, optional
2122:         If provided, the calculation is done into this array.
2123:     dtype : data-type, optional
2124:         If provided, forces the calculation to use the data type specified.
2125:         Note that you may have to also give a more liberal `casting`
2126:         parameter to allow the conversions.
2127:     order : {'C', 'F', 'A', 'K'}, optional
2128:         Controls the memory layout of the output. 'C' means it should
2129:         be C contiguous. 'F' means it should be Fortran contiguous,
2130:         'A' means it should be 'F' if the inputs are all 'F', 'C' otherwise.
2131:         'K' means it should be as close to the layout as the inputs as
2132:         is possible, including arbitrarily permuted axes.
2133:         Default is 'K'.
2134:     casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
2135:         Controls what kind of data casting may occur.  Setting this to
2136:         'unsafe' is not recommended, as it can adversely affect accumulations.
2137: 
2138:           * 'no' means the data types should not be cast at all.
2139:           * 'equiv' means only byte-order changes are allowed.
2140:           * 'safe' means only casts which can preserve values are allowed.
2141:           * 'same_kind' means only safe casts or casts within a kind,
2142:             like float64 to float32, are allowed.
2143:           * 'unsafe' means any data conversions may be done.
2144: 
2145:     Returns
2146:     -------
2147:     output : ndarray
2148:         The calculation based on the Einstein summation convention.
2149: 
2150:     See Also
2151:     --------
2152:     dot, inner, outer, tensordot
2153: 
2154:     Notes
2155:     -----
2156:     .. versionadded:: 1.6.0
2157: 
2158:     The subscripts string is a comma-separated list of subscript labels,
2159:     where each label refers to a dimension of the corresponding operand.
2160:     Repeated subscripts labels in one operand take the diagonal.  For example,
2161:     ``np.einsum('ii', a)`` is equivalent to ``np.trace(a)``.
2162: 
2163:     Whenever a label is repeated, it is summed, so ``np.einsum('i,i', a, b)``
2164:     is equivalent to ``np.inner(a,b)``.  If a label appears only once,
2165:     it is not summed, so ``np.einsum('i', a)`` produces a view of ``a``
2166:     with no changes.
2167: 
2168:     The order of labels in the output is by default alphabetical.  This
2169:     means that ``np.einsum('ij', a)`` doesn't affect a 2D array, while
2170:     ``np.einsum('ji', a)`` takes its transpose.
2171: 
2172:     The output can be controlled by specifying output subscript labels
2173:     as well.  This specifies the label order, and allows summing to
2174:     be disallowed or forced when desired.  The call ``np.einsum('i->', a)``
2175:     is like ``np.sum(a, axis=-1)``, and ``np.einsum('ii->i', a)``
2176:     is like ``np.diag(a)``.  The difference is that `einsum` does not
2177:     allow broadcasting by default.
2178: 
2179:     To enable and control broadcasting, use an ellipsis.  Default
2180:     NumPy-style broadcasting is done by adding an ellipsis
2181:     to the left of each term, like ``np.einsum('...ii->...i', a)``.
2182:     To take the trace along the first and last axes,
2183:     you can do ``np.einsum('i...i', a)``, or to do a matrix-matrix
2184:     product with the left-most indices instead of rightmost, you can do
2185:     ``np.einsum('ij...,jk...->ik...', a, b)``.
2186: 
2187:     When there is only one operand, no axes are summed, and no output
2188:     parameter is provided, a view into the operand is returned instead
2189:     of a new array.  Thus, taking the diagonal as ``np.einsum('ii->i', a)``
2190:     produces a view.
2191: 
2192:     An alternative way to provide the subscripts and operands is as
2193:     ``einsum(op0, sublist0, op1, sublist1, ..., [sublistout])``. The examples
2194:     below have corresponding `einsum` calls with the two parameter methods.
2195: 
2196:     .. versionadded:: 1.10.0
2197: 
2198:     Views returned from einsum are now writeable whenever the input array
2199:     is writeable. For example, ``np.einsum('ijk...->kji...', a)`` will now
2200:     have the same effect as ``np.swapaxes(a, 0, 2)`` and
2201:     ``np.einsum('ii->i', a)`` will return a writeable view of the diagonal
2202:     of a 2D array.
2203: 
2204:     Examples
2205:     --------
2206:     >>> a = np.arange(25).reshape(5,5)
2207:     >>> b = np.arange(5)
2208:     >>> c = np.arange(6).reshape(2,3)
2209: 
2210:     >>> np.einsum('ii', a)
2211:     60
2212:     >>> np.einsum(a, [0,0])
2213:     60
2214:     >>> np.trace(a)
2215:     60
2216: 
2217:     >>> np.einsum('ii->i', a)
2218:     array([ 0,  6, 12, 18, 24])
2219:     >>> np.einsum(a, [0,0], [0])
2220:     array([ 0,  6, 12, 18, 24])
2221:     >>> np.diag(a)
2222:     array([ 0,  6, 12, 18, 24])
2223: 
2224:     >>> np.einsum('ij,j', a, b)
2225:     array([ 30,  80, 130, 180, 230])
2226:     >>> np.einsum(a, [0,1], b, [1])
2227:     array([ 30,  80, 130, 180, 230])
2228:     >>> np.dot(a, b)
2229:     array([ 30,  80, 130, 180, 230])
2230:     >>> np.einsum('...j,j', a, b)
2231:     array([ 30,  80, 130, 180, 230])
2232: 
2233:     >>> np.einsum('ji', c)
2234:     array([[0, 3],
2235:            [1, 4],
2236:            [2, 5]])
2237:     >>> np.einsum(c, [1,0])
2238:     array([[0, 3],
2239:            [1, 4],
2240:            [2, 5]])
2241:     >>> c.T
2242:     array([[0, 3],
2243:            [1, 4],
2244:            [2, 5]])
2245: 
2246:     >>> np.einsum('..., ...', 3, c)
2247:     array([[ 0,  3,  6],
2248:            [ 9, 12, 15]])
2249:     >>> np.einsum(3, [Ellipsis], c, [Ellipsis])
2250:     array([[ 0,  3,  6],
2251:            [ 9, 12, 15]])
2252:     >>> np.multiply(3, c)
2253:     array([[ 0,  3,  6],
2254:            [ 9, 12, 15]])
2255: 
2256:     >>> np.einsum('i,i', b, b)
2257:     30
2258:     >>> np.einsum(b, [0], b, [0])
2259:     30
2260:     >>> np.inner(b,b)
2261:     30
2262: 
2263:     >>> np.einsum('i,j', np.arange(2)+1, b)
2264:     array([[0, 1, 2, 3, 4],
2265:            [0, 2, 4, 6, 8]])
2266:     >>> np.einsum(np.arange(2)+1, [0], b, [1])
2267:     array([[0, 1, 2, 3, 4],
2268:            [0, 2, 4, 6, 8]])
2269:     >>> np.outer(np.arange(2)+1, b)
2270:     array([[0, 1, 2, 3, 4],
2271:            [0, 2, 4, 6, 8]])
2272: 
2273:     >>> np.einsum('i...->...', a)
2274:     array([50, 55, 60, 65, 70])
2275:     >>> np.einsum(a, [0,Ellipsis], [Ellipsis])
2276:     array([50, 55, 60, 65, 70])
2277:     >>> np.sum(a, axis=0)
2278:     array([50, 55, 60, 65, 70])
2279: 
2280:     >>> a = np.arange(60.).reshape(3,4,5)
2281:     >>> b = np.arange(24.).reshape(4,3,2)
2282:     >>> np.einsum('ijk,jil->kl', a, b)
2283:     array([[ 4400.,  4730.],
2284:            [ 4532.,  4874.],
2285:            [ 4664.,  5018.],
2286:            [ 4796.,  5162.],
2287:            [ 4928.,  5306.]])
2288:     >>> np.einsum(a, [0,1,2], b, [1,0,3], [2,3])
2289:     array([[ 4400.,  4730.],
2290:            [ 4532.,  4874.],
2291:            [ 4664.,  5018.],
2292:            [ 4796.,  5162.],
2293:            [ 4928.,  5306.]])
2294:     >>> np.tensordot(a,b, axes=([1,0],[0,1]))
2295:     array([[ 4400.,  4730.],
2296:            [ 4532.,  4874.],
2297:            [ 4664.,  5018.],
2298:            [ 4796.,  5162.],
2299:            [ 4928.,  5306.]])
2300: 
2301:     >>> a = np.arange(6).reshape((3,2))
2302:     >>> b = np.arange(12).reshape((4,3))
2303:     >>> np.einsum('ki,jk->ij', a, b)
2304:     array([[10, 28, 46, 64],
2305:            [13, 40, 67, 94]])
2306:     >>> np.einsum('ki,...k->i...', a, b)
2307:     array([[10, 28, 46, 64],
2308:            [13, 40, 67, 94]])
2309:     >>> np.einsum('k...,jk', a, b)
2310:     array([[10, 28, 46, 64],
2311:            [13, 40, 67, 94]])
2312: 
2313:     >>> # since version 1.10.0
2314:     >>> a = np.zeros((3, 3))
2315:     >>> np.einsum('ii->i', a)[:] = 1
2316:     >>> a
2317:     array([[ 1.,  0.,  0.],
2318:            [ 0.,  1.,  0.],
2319:            [ 0.,  0.,  1.]])
2320: 
2321:     ''')
2322: 
2323: add_newdoc('numpy.core', 'vdot',
2324:     '''
2325:     vdot(a, b)
2326: 
2327:     Return the dot product of two vectors.
2328: 
2329:     The vdot(`a`, `b`) function handles complex numbers differently than
2330:     dot(`a`, `b`).  If the first argument is complex the complex conjugate
2331:     of the first argument is used for the calculation of the dot product.
2332: 
2333:     Note that `vdot` handles multidimensional arrays differently than `dot`:
2334:     it does *not* perform a matrix product, but flattens input arguments
2335:     to 1-D vectors first. Consequently, it should only be used for vectors.
2336: 
2337:     Parameters
2338:     ----------
2339:     a : array_like
2340:         If `a` is complex the complex conjugate is taken before calculation
2341:         of the dot product.
2342:     b : array_like
2343:         Second argument to the dot product.
2344: 
2345:     Returns
2346:     -------
2347:     output : ndarray
2348:         Dot product of `a` and `b`.  Can be an int, float, or
2349:         complex depending on the types of `a` and `b`.
2350: 
2351:     See Also
2352:     --------
2353:     dot : Return the dot product without using the complex conjugate of the
2354:           first argument.
2355: 
2356:     Examples
2357:     --------
2358:     >>> a = np.array([1+2j,3+4j])
2359:     >>> b = np.array([5+6j,7+8j])
2360:     >>> np.vdot(a, b)
2361:     (70-8j)
2362:     >>> np.vdot(b, a)
2363:     (70+8j)
2364: 
2365:     Note that higher-dimensional arrays are flattened!
2366: 
2367:     >>> a = np.array([[1, 4], [5, 6]])
2368:     >>> b = np.array([[4, 1], [2, 2]])
2369:     >>> np.vdot(a, b)
2370:     30
2371:     >>> np.vdot(b, a)
2372:     30
2373:     >>> 1*4 + 4*1 + 5*2 + 6*2
2374:     30
2375: 
2376:     ''')
2377: 
2378: 
2379: ##############################################################################
2380: #
2381: # Documentation for ndarray attributes and methods
2382: #
2383: ##############################################################################
2384: 
2385: 
2386: ##############################################################################
2387: #
2388: # ndarray object
2389: #
2390: ##############################################################################
2391: 
2392: 
2393: add_newdoc('numpy.core.multiarray', 'ndarray',
2394:     '''
2395:     ndarray(shape, dtype=float, buffer=None, offset=0,
2396:             strides=None, order=None)
2397: 
2398:     An array object represents a multidimensional, homogeneous array
2399:     of fixed-size items.  An associated data-type object describes the
2400:     format of each element in the array (its byte-order, how many bytes it
2401:     occupies in memory, whether it is an integer, a floating point number,
2402:     or something else, etc.)
2403: 
2404:     Arrays should be constructed using `array`, `zeros` or `empty` (refer
2405:     to the See Also section below).  The parameters given here refer to
2406:     a low-level method (`ndarray(...)`) for instantiating an array.
2407: 
2408:     For more information, refer to the `numpy` module and examine the
2409:     the methods and attributes of an array.
2410: 
2411:     Parameters
2412:     ----------
2413:     (for the __new__ method; see Notes below)
2414: 
2415:     shape : tuple of ints
2416:         Shape of created array.
2417:     dtype : data-type, optional
2418:         Any object that can be interpreted as a numpy data type.
2419:     buffer : object exposing buffer interface, optional
2420:         Used to fill the array with data.
2421:     offset : int, optional
2422:         Offset of array data in buffer.
2423:     strides : tuple of ints, optional
2424:         Strides of data in memory.
2425:     order : {'C', 'F'}, optional
2426:         Row-major (C-style) or column-major (Fortran-style) order.
2427: 
2428:     Attributes
2429:     ----------
2430:     T : ndarray
2431:         Transpose of the array.
2432:     data : buffer
2433:         The array's elements, in memory.
2434:     dtype : dtype object
2435:         Describes the format of the elements in the array.
2436:     flags : dict
2437:         Dictionary containing information related to memory use, e.g.,
2438:         'C_CONTIGUOUS', 'OWNDATA', 'WRITEABLE', etc.
2439:     flat : numpy.flatiter object
2440:         Flattened version of the array as an iterator.  The iterator
2441:         allows assignments, e.g., ``x.flat = 3`` (See `ndarray.flat` for
2442:         assignment examples; TODO).
2443:     imag : ndarray
2444:         Imaginary part of the array.
2445:     real : ndarray
2446:         Real part of the array.
2447:     size : int
2448:         Number of elements in the array.
2449:     itemsize : int
2450:         The memory use of each array element in bytes.
2451:     nbytes : int
2452:         The total number of bytes required to store the array data,
2453:         i.e., ``itemsize * size``.
2454:     ndim : int
2455:         The array's number of dimensions.
2456:     shape : tuple of ints
2457:         Shape of the array.
2458:     strides : tuple of ints
2459:         The step-size required to move from one element to the next in
2460:         memory. For example, a contiguous ``(3, 4)`` array of type
2461:         ``int16`` in C-order has strides ``(8, 2)``.  This implies that
2462:         to move from element to element in memory requires jumps of 2 bytes.
2463:         To move from row-to-row, one needs to jump 8 bytes at a time
2464:         (``2 * 4``).
2465:     ctypes : ctypes object
2466:         Class containing properties of the array needed for interaction
2467:         with ctypes.
2468:     base : ndarray
2469:         If the array is a view into another array, that array is its `base`
2470:         (unless that array is also a view).  The `base` array is where the
2471:         array data is actually stored.
2472: 
2473:     See Also
2474:     --------
2475:     array : Construct an array.
2476:     zeros : Create an array, each element of which is zero.
2477:     empty : Create an array, but leave its allocated memory unchanged (i.e.,
2478:             it contains "garbage").
2479:     dtype : Create a data-type.
2480: 
2481:     Notes
2482:     -----
2483:     There are two modes of creating an array using ``__new__``:
2484: 
2485:     1. If `buffer` is None, then only `shape`, `dtype`, and `order`
2486:        are used.
2487:     2. If `buffer` is an object exposing the buffer interface, then
2488:        all keywords are interpreted.
2489: 
2490:     No ``__init__`` method is needed because the array is fully initialized
2491:     after the ``__new__`` method.
2492: 
2493:     Examples
2494:     --------
2495:     These examples illustrate the low-level `ndarray` constructor.  Refer
2496:     to the `See Also` section above for easier ways of constructing an
2497:     ndarray.
2498: 
2499:     First mode, `buffer` is None:
2500: 
2501:     >>> np.ndarray(shape=(2,2), dtype=float, order='F')
2502:     array([[ -1.13698227e+002,   4.25087011e-303],
2503:            [  2.88528414e-306,   3.27025015e-309]])         #random
2504: 
2505:     Second mode:
2506: 
2507:     >>> np.ndarray((2,), buffer=np.array([1,2,3]),
2508:     ...            offset=np.int_().itemsize,
2509:     ...            dtype=int) # offset = 1*itemsize, i.e. skip first element
2510:     array([2, 3])
2511: 
2512:     ''')
2513: 
2514: 
2515: ##############################################################################
2516: #
2517: # ndarray attributes
2518: #
2519: ##############################################################################
2520: 
2521: 
2522: add_newdoc('numpy.core.multiarray', 'ndarray', ('__array_interface__',
2523:     '''Array protocol: Python side.'''))
2524: 
2525: 
2526: add_newdoc('numpy.core.multiarray', 'ndarray', ('__array_finalize__',
2527:     '''None.'''))
2528: 
2529: 
2530: add_newdoc('numpy.core.multiarray', 'ndarray', ('__array_priority__',
2531:     '''Array priority.'''))
2532: 
2533: 
2534: add_newdoc('numpy.core.multiarray', 'ndarray', ('__array_struct__',
2535:     '''Array protocol: C-struct side.'''))
2536: 
2537: 
2538: add_newdoc('numpy.core.multiarray', 'ndarray', ('_as_parameter_',
2539:     '''Allow the array to be interpreted as a ctypes object by returning the
2540:     data-memory location as an integer
2541: 
2542:     '''))
2543: 
2544: 
2545: add_newdoc('numpy.core.multiarray', 'ndarray', ('base',
2546:     '''
2547:     Base object if memory is from some other object.
2548: 
2549:     Examples
2550:     --------
2551:     The base of an array that owns its memory is None:
2552: 
2553:     >>> x = np.array([1,2,3,4])
2554:     >>> x.base is None
2555:     True
2556: 
2557:     Slicing creates a view, whose memory is shared with x:
2558: 
2559:     >>> y = x[2:]
2560:     >>> y.base is x
2561:     True
2562: 
2563:     '''))
2564: 
2565: 
2566: add_newdoc('numpy.core.multiarray', 'ndarray', ('ctypes',
2567:     '''
2568:     An object to simplify the interaction of the array with the ctypes
2569:     module.
2570: 
2571:     This attribute creates an object that makes it easier to use arrays
2572:     when calling shared libraries with the ctypes module. The returned
2573:     object has, among others, data, shape, and strides attributes (see
2574:     Notes below) which themselves return ctypes objects that can be used
2575:     as arguments to a shared library.
2576: 
2577:     Parameters
2578:     ----------
2579:     None
2580: 
2581:     Returns
2582:     -------
2583:     c : Python object
2584:         Possessing attributes data, shape, strides, etc.
2585: 
2586:     See Also
2587:     --------
2588:     numpy.ctypeslib
2589: 
2590:     Notes
2591:     -----
2592:     Below are the public attributes of this object which were documented
2593:     in "Guide to NumPy" (we have omitted undocumented public attributes,
2594:     as well as documented private attributes):
2595: 
2596:     * data: A pointer to the memory area of the array as a Python integer.
2597:       This memory area may contain data that is not aligned, or not in correct
2598:       byte-order. The memory area may not even be writeable. The array
2599:       flags and data-type of this array should be respected when passing this
2600:       attribute to arbitrary C-code to avoid trouble that can include Python
2601:       crashing. User Beware! The value of this attribute is exactly the same
2602:       as self._array_interface_['data'][0].
2603: 
2604:     * shape (c_intp*self.ndim): A ctypes array of length self.ndim where
2605:       the basetype is the C-integer corresponding to dtype('p') on this
2606:       platform. This base-type could be c_int, c_long, or c_longlong
2607:       depending on the platform. The c_intp type is defined accordingly in
2608:       numpy.ctypeslib. The ctypes array contains the shape of the underlying
2609:       array.
2610: 
2611:     * strides (c_intp*self.ndim): A ctypes array of length self.ndim where
2612:       the basetype is the same as for the shape attribute. This ctypes array
2613:       contains the strides information from the underlying array. This strides
2614:       information is important for showing how many bytes must be jumped to
2615:       get to the next element in the array.
2616: 
2617:     * data_as(obj): Return the data pointer cast to a particular c-types object.
2618:       For example, calling self._as_parameter_ is equivalent to
2619:       self.data_as(ctypes.c_void_p). Perhaps you want to use the data as a
2620:       pointer to a ctypes array of floating-point data:
2621:       self.data_as(ctypes.POINTER(ctypes.c_double)).
2622: 
2623:     * shape_as(obj): Return the shape tuple as an array of some other c-types
2624:       type. For example: self.shape_as(ctypes.c_short).
2625: 
2626:     * strides_as(obj): Return the strides tuple as an array of some other
2627:       c-types type. For example: self.strides_as(ctypes.c_longlong).
2628: 
2629:     Be careful using the ctypes attribute - especially on temporary
2630:     arrays or arrays constructed on the fly. For example, calling
2631:     ``(a+b).ctypes.data_as(ctypes.c_void_p)`` returns a pointer to memory
2632:     that is invalid because the array created as (a+b) is deallocated
2633:     before the next Python statement. You can avoid this problem using
2634:     either ``c=a+b`` or ``ct=(a+b).ctypes``. In the latter case, ct will
2635:     hold a reference to the array until ct is deleted or re-assigned.
2636: 
2637:     If the ctypes module is not available, then the ctypes attribute
2638:     of array objects still returns something useful, but ctypes objects
2639:     are not returned and errors may be raised instead. In particular,
2640:     the object will still have the as parameter attribute which will
2641:     return an integer equal to the data attribute.
2642: 
2643:     Examples
2644:     --------
2645:     >>> import ctypes
2646:     >>> x
2647:     array([[0, 1],
2648:            [2, 3]])
2649:     >>> x.ctypes.data
2650:     30439712
2651:     >>> x.ctypes.data_as(ctypes.POINTER(ctypes.c_long))
2652:     <ctypes.LP_c_long object at 0x01F01300>
2653:     >>> x.ctypes.data_as(ctypes.POINTER(ctypes.c_long)).contents
2654:     c_long(0)
2655:     >>> x.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)).contents
2656:     c_longlong(4294967296L)
2657:     >>> x.ctypes.shape
2658:     <numpy.core._internal.c_long_Array_2 object at 0x01FFD580>
2659:     >>> x.ctypes.shape_as(ctypes.c_long)
2660:     <numpy.core._internal.c_long_Array_2 object at 0x01FCE620>
2661:     >>> x.ctypes.strides
2662:     <numpy.core._internal.c_long_Array_2 object at 0x01FCE620>
2663:     >>> x.ctypes.strides_as(ctypes.c_longlong)
2664:     <numpy.core._internal.c_longlong_Array_2 object at 0x01F01300>
2665: 
2666:     '''))
2667: 
2668: 
2669: add_newdoc('numpy.core.multiarray', 'ndarray', ('data',
2670:     '''Python buffer object pointing to the start of the array's data.'''))
2671: 
2672: 
2673: add_newdoc('numpy.core.multiarray', 'ndarray', ('dtype',
2674:     '''
2675:     Data-type of the array's elements.
2676: 
2677:     Parameters
2678:     ----------
2679:     None
2680: 
2681:     Returns
2682:     -------
2683:     d : numpy dtype object
2684: 
2685:     See Also
2686:     --------
2687:     numpy.dtype
2688: 
2689:     Examples
2690:     --------
2691:     >>> x
2692:     array([[0, 1],
2693:            [2, 3]])
2694:     >>> x.dtype
2695:     dtype('int32')
2696:     >>> type(x.dtype)
2697:     <type 'numpy.dtype'>
2698: 
2699:     '''))
2700: 
2701: 
2702: add_newdoc('numpy.core.multiarray', 'ndarray', ('imag',
2703:     '''
2704:     The imaginary part of the array.
2705: 
2706:     Examples
2707:     --------
2708:     >>> x = np.sqrt([1+0j, 0+1j])
2709:     >>> x.imag
2710:     array([ 0.        ,  0.70710678])
2711:     >>> x.imag.dtype
2712:     dtype('float64')
2713: 
2714:     '''))
2715: 
2716: 
2717: add_newdoc('numpy.core.multiarray', 'ndarray', ('itemsize',
2718:     '''
2719:     Length of one array element in bytes.
2720: 
2721:     Examples
2722:     --------
2723:     >>> x = np.array([1,2,3], dtype=np.float64)
2724:     >>> x.itemsize
2725:     8
2726:     >>> x = np.array([1,2,3], dtype=np.complex128)
2727:     >>> x.itemsize
2728:     16
2729: 
2730:     '''))
2731: 
2732: 
2733: add_newdoc('numpy.core.multiarray', 'ndarray', ('flags',
2734:     '''
2735:     Information about the memory layout of the array.
2736: 
2737:     Attributes
2738:     ----------
2739:     C_CONTIGUOUS (C)
2740:         The data is in a single, C-style contiguous segment.
2741:     F_CONTIGUOUS (F)
2742:         The data is in a single, Fortran-style contiguous segment.
2743:     OWNDATA (O)
2744:         The array owns the memory it uses or borrows it from another object.
2745:     WRITEABLE (W)
2746:         The data area can be written to.  Setting this to False locks
2747:         the data, making it read-only.  A view (slice, etc.) inherits WRITEABLE
2748:         from its base array at creation time, but a view of a writeable
2749:         array may be subsequently locked while the base array remains writeable.
2750:         (The opposite is not true, in that a view of a locked array may not
2751:         be made writeable.  However, currently, locking a base object does not
2752:         lock any views that already reference it, so under that circumstance it
2753:         is possible to alter the contents of a locked array via a previously
2754:         created writeable view onto it.)  Attempting to change a non-writeable
2755:         array raises a RuntimeError exception.
2756:     ALIGNED (A)
2757:         The data and all elements are aligned appropriately for the hardware.
2758:     UPDATEIFCOPY (U)
2759:         This array is a copy of some other array. When this array is
2760:         deallocated, the base array will be updated with the contents of
2761:         this array.
2762:     FNC
2763:         F_CONTIGUOUS and not C_CONTIGUOUS.
2764:     FORC
2765:         F_CONTIGUOUS or C_CONTIGUOUS (one-segment test).
2766:     BEHAVED (B)
2767:         ALIGNED and WRITEABLE.
2768:     CARRAY (CA)
2769:         BEHAVED and C_CONTIGUOUS.
2770:     FARRAY (FA)
2771:         BEHAVED and F_CONTIGUOUS and not C_CONTIGUOUS.
2772: 
2773:     Notes
2774:     -----
2775:     The `flags` object can be accessed dictionary-like (as in ``a.flags['WRITEABLE']``),
2776:     or by using lowercased attribute names (as in ``a.flags.writeable``). Short flag
2777:     names are only supported in dictionary access.
2778: 
2779:     Only the UPDATEIFCOPY, WRITEABLE, and ALIGNED flags can be changed by
2780:     the user, via direct assignment to the attribute or dictionary entry,
2781:     or by calling `ndarray.setflags`.
2782: 
2783:     The array flags cannot be set arbitrarily:
2784: 
2785:     - UPDATEIFCOPY can only be set ``False``.
2786:     - ALIGNED can only be set ``True`` if the data is truly aligned.
2787:     - WRITEABLE can only be set ``True`` if the array owns its own memory
2788:       or the ultimate owner of the memory exposes a writeable buffer
2789:       interface or is a string.
2790: 
2791:     Arrays can be both C-style and Fortran-style contiguous simultaneously.
2792:     This is clear for 1-dimensional arrays, but can also be true for higher
2793:     dimensional arrays.
2794: 
2795:     Even for contiguous arrays a stride for a given dimension
2796:     ``arr.strides[dim]`` may be *arbitrary* if ``arr.shape[dim] == 1``
2797:     or the array has no elements.
2798:     It does *not* generally hold that ``self.strides[-1] == self.itemsize``
2799:     for C-style contiguous arrays or ``self.strides[0] == self.itemsize`` for
2800:     Fortran-style contiguous arrays is true.
2801:     '''))
2802: 
2803: 
2804: add_newdoc('numpy.core.multiarray', 'ndarray', ('flat',
2805:     '''
2806:     A 1-D iterator over the array.
2807: 
2808:     This is a `numpy.flatiter` instance, which acts similarly to, but is not
2809:     a subclass of, Python's built-in iterator object.
2810: 
2811:     See Also
2812:     --------
2813:     flatten : Return a copy of the array collapsed into one dimension.
2814: 
2815:     flatiter
2816: 
2817:     Examples
2818:     --------
2819:     >>> x = np.arange(1, 7).reshape(2, 3)
2820:     >>> x
2821:     array([[1, 2, 3],
2822:            [4, 5, 6]])
2823:     >>> x.flat[3]
2824:     4
2825:     >>> x.T
2826:     array([[1, 4],
2827:            [2, 5],
2828:            [3, 6]])
2829:     >>> x.T.flat[3]
2830:     5
2831:     >>> type(x.flat)
2832:     <type 'numpy.flatiter'>
2833: 
2834:     An assignment example:
2835: 
2836:     >>> x.flat = 3; x
2837:     array([[3, 3, 3],
2838:            [3, 3, 3]])
2839:     >>> x.flat[[1,4]] = 1; x
2840:     array([[3, 1, 3],
2841:            [3, 1, 3]])
2842: 
2843:     '''))
2844: 
2845: 
2846: add_newdoc('numpy.core.multiarray', 'ndarray', ('nbytes',
2847:     '''
2848:     Total bytes consumed by the elements of the array.
2849: 
2850:     Notes
2851:     -----
2852:     Does not include memory consumed by non-element attributes of the
2853:     array object.
2854: 
2855:     Examples
2856:     --------
2857:     >>> x = np.zeros((3,5,2), dtype=np.complex128)
2858:     >>> x.nbytes
2859:     480
2860:     >>> np.prod(x.shape) * x.itemsize
2861:     480
2862: 
2863:     '''))
2864: 
2865: 
2866: add_newdoc('numpy.core.multiarray', 'ndarray', ('ndim',
2867:     '''
2868:     Number of array dimensions.
2869: 
2870:     Examples
2871:     --------
2872:     >>> x = np.array([1, 2, 3])
2873:     >>> x.ndim
2874:     1
2875:     >>> y = np.zeros((2, 3, 4))
2876:     >>> y.ndim
2877:     3
2878: 
2879:     '''))
2880: 
2881: 
2882: add_newdoc('numpy.core.multiarray', 'ndarray', ('real',
2883:     '''
2884:     The real part of the array.
2885: 
2886:     Examples
2887:     --------
2888:     >>> x = np.sqrt([1+0j, 0+1j])
2889:     >>> x.real
2890:     array([ 1.        ,  0.70710678])
2891:     >>> x.real.dtype
2892:     dtype('float64')
2893: 
2894:     See Also
2895:     --------
2896:     numpy.real : equivalent function
2897: 
2898:     '''))
2899: 
2900: 
2901: add_newdoc('numpy.core.multiarray', 'ndarray', ('shape',
2902:     '''
2903:     Tuple of array dimensions.
2904: 
2905:     Notes
2906:     -----
2907:     May be used to "reshape" the array, as long as this would not
2908:     require a change in the total number of elements
2909: 
2910:     Examples
2911:     --------
2912:     >>> x = np.array([1, 2, 3, 4])
2913:     >>> x.shape
2914:     (4,)
2915:     >>> y = np.zeros((2, 3, 4))
2916:     >>> y.shape
2917:     (2, 3, 4)
2918:     >>> y.shape = (3, 8)
2919:     >>> y
2920:     array([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
2921:            [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
2922:            [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])
2923:     >>> y.shape = (3, 6)
2924:     Traceback (most recent call last):
2925:       File "<stdin>", line 1, in <module>
2926:     ValueError: total size of new array must be unchanged
2927: 
2928:     '''))
2929: 
2930: 
2931: add_newdoc('numpy.core.multiarray', 'ndarray', ('size',
2932:     '''
2933:     Number of elements in the array.
2934: 
2935:     Equivalent to ``np.prod(a.shape)``, i.e., the product of the array's
2936:     dimensions.
2937: 
2938:     Examples
2939:     --------
2940:     >>> x = np.zeros((3, 5, 2), dtype=np.complex128)
2941:     >>> x.size
2942:     30
2943:     >>> np.prod(x.shape)
2944:     30
2945: 
2946:     '''))
2947: 
2948: 
2949: add_newdoc('numpy.core.multiarray', 'ndarray', ('strides',
2950:     '''
2951:     Tuple of bytes to step in each dimension when traversing an array.
2952: 
2953:     The byte offset of element ``(i[0], i[1], ..., i[n])`` in an array `a`
2954:     is::
2955: 
2956:         offset = sum(np.array(i) * a.strides)
2957: 
2958:     A more detailed explanation of strides can be found in the
2959:     "ndarray.rst" file in the NumPy reference guide.
2960: 
2961:     Notes
2962:     -----
2963:     Imagine an array of 32-bit integers (each 4 bytes)::
2964: 
2965:       x = np.array([[0, 1, 2, 3, 4],
2966:                     [5, 6, 7, 8, 9]], dtype=np.int32)
2967: 
2968:     This array is stored in memory as 40 bytes, one after the other
2969:     (known as a contiguous block of memory).  The strides of an array tell
2970:     us how many bytes we have to skip in memory to move to the next position
2971:     along a certain axis.  For example, we have to skip 4 bytes (1 value) to
2972:     move to the next column, but 20 bytes (5 values) to get to the same
2973:     position in the next row.  As such, the strides for the array `x` will be
2974:     ``(20, 4)``.
2975: 
2976:     See Also
2977:     --------
2978:     numpy.lib.stride_tricks.as_strided
2979: 
2980:     Examples
2981:     --------
2982:     >>> y = np.reshape(np.arange(2*3*4), (2,3,4))
2983:     >>> y
2984:     array([[[ 0,  1,  2,  3],
2985:             [ 4,  5,  6,  7],
2986:             [ 8,  9, 10, 11]],
2987:            [[12, 13, 14, 15],
2988:             [16, 17, 18, 19],
2989:             [20, 21, 22, 23]]])
2990:     >>> y.strides
2991:     (48, 16, 4)
2992:     >>> y[1,1,1]
2993:     17
2994:     >>> offset=sum(y.strides * np.array((1,1,1)))
2995:     >>> offset/y.itemsize
2996:     17
2997: 
2998:     >>> x = np.reshape(np.arange(5*6*7*8), (5,6,7,8)).transpose(2,3,1,0)
2999:     >>> x.strides
3000:     (32, 4, 224, 1344)
3001:     >>> i = np.array([3,5,2,2])
3002:     >>> offset = sum(i * x.strides)
3003:     >>> x[3,5,2,2]
3004:     813
3005:     >>> offset / x.itemsize
3006:     813
3007: 
3008:     '''))
3009: 
3010: 
3011: add_newdoc('numpy.core.multiarray', 'ndarray', ('T',
3012:     '''
3013:     Same as self.transpose(), except that self is returned if
3014:     self.ndim < 2.
3015: 
3016:     Examples
3017:     --------
3018:     >>> x = np.array([[1.,2.],[3.,4.]])
3019:     >>> x
3020:     array([[ 1.,  2.],
3021:            [ 3.,  4.]])
3022:     >>> x.T
3023:     array([[ 1.,  3.],
3024:            [ 2.,  4.]])
3025:     >>> x = np.array([1.,2.,3.,4.])
3026:     >>> x
3027:     array([ 1.,  2.,  3.,  4.])
3028:     >>> x.T
3029:     array([ 1.,  2.,  3.,  4.])
3030: 
3031:     '''))
3032: 
3033: 
3034: ##############################################################################
3035: #
3036: # ndarray methods
3037: #
3038: ##############################################################################
3039: 
3040: 
3041: add_newdoc('numpy.core.multiarray', 'ndarray', ('__array__',
3042:     ''' a.__array__(|dtype) -> reference if type unchanged, copy otherwise.
3043: 
3044:     Returns either a new reference to self if dtype is not given or a new array
3045:     of provided data type if dtype is different from the current dtype of the
3046:     array.
3047: 
3048:     '''))
3049: 
3050: 
3051: add_newdoc('numpy.core.multiarray', 'ndarray', ('__array_prepare__',
3052:     '''a.__array_prepare__(obj) -> Object of same type as ndarray object obj.
3053: 
3054:     '''))
3055: 
3056: 
3057: add_newdoc('numpy.core.multiarray', 'ndarray', ('__array_wrap__',
3058:     '''a.__array_wrap__(obj) -> Object of same type as ndarray object a.
3059: 
3060:     '''))
3061: 
3062: 
3063: add_newdoc('numpy.core.multiarray', 'ndarray', ('__copy__',
3064:     '''a.__copy__([order])
3065: 
3066:     Return a copy of the array.
3067: 
3068:     Parameters
3069:     ----------
3070:     order : {'C', 'F', 'A'}, optional
3071:         If order is 'C' (False) then the result is contiguous (default).
3072:         If order is 'Fortran' (True) then the result has fortran order.
3073:         If order is 'Any' (None) then the result has fortran order
3074:         only if the array already is in fortran order.
3075: 
3076:     '''))
3077: 
3078: 
3079: add_newdoc('numpy.core.multiarray', 'ndarray', ('__deepcopy__',
3080:     '''a.__deepcopy__() -> Deep copy of array.
3081: 
3082:     Used if copy.deepcopy is called on an array.
3083: 
3084:     '''))
3085: 
3086: 
3087: add_newdoc('numpy.core.multiarray', 'ndarray', ('__reduce__',
3088:     '''a.__reduce__()
3089: 
3090:     For pickling.
3091: 
3092:     '''))
3093: 
3094: 
3095: add_newdoc('numpy.core.multiarray', 'ndarray', ('__setstate__',
3096:     '''a.__setstate__(version, shape, dtype, isfortran, rawdata)
3097: 
3098:     For unpickling.
3099: 
3100:     Parameters
3101:     ----------
3102:     version : int
3103:         optional pickle version. If omitted defaults to 0.
3104:     shape : tuple
3105:     dtype : data-type
3106:     isFortran : bool
3107:     rawdata : string or list
3108:         a binary string with the data (or a list if 'a' is an object array)
3109: 
3110:     '''))
3111: 
3112: 
3113: add_newdoc('numpy.core.multiarray', 'ndarray', ('all',
3114:     '''
3115:     a.all(axis=None, out=None, keepdims=False)
3116: 
3117:     Returns True if all elements evaluate to True.
3118: 
3119:     Refer to `numpy.all` for full documentation.
3120: 
3121:     See Also
3122:     --------
3123:     numpy.all : equivalent function
3124: 
3125:     '''))
3126: 
3127: 
3128: add_newdoc('numpy.core.multiarray', 'ndarray', ('any',
3129:     '''
3130:     a.any(axis=None, out=None, keepdims=False)
3131: 
3132:     Returns True if any of the elements of `a` evaluate to True.
3133: 
3134:     Refer to `numpy.any` for full documentation.
3135: 
3136:     See Also
3137:     --------
3138:     numpy.any : equivalent function
3139: 
3140:     '''))
3141: 
3142: 
3143: add_newdoc('numpy.core.multiarray', 'ndarray', ('argmax',
3144:     '''
3145:     a.argmax(axis=None, out=None)
3146: 
3147:     Return indices of the maximum values along the given axis.
3148: 
3149:     Refer to `numpy.argmax` for full documentation.
3150: 
3151:     See Also
3152:     --------
3153:     numpy.argmax : equivalent function
3154: 
3155:     '''))
3156: 
3157: 
3158: add_newdoc('numpy.core.multiarray', 'ndarray', ('argmin',
3159:     '''
3160:     a.argmin(axis=None, out=None)
3161: 
3162:     Return indices of the minimum values along the given axis of `a`.
3163: 
3164:     Refer to `numpy.argmin` for detailed documentation.
3165: 
3166:     See Also
3167:     --------
3168:     numpy.argmin : equivalent function
3169: 
3170:     '''))
3171: 
3172: 
3173: add_newdoc('numpy.core.multiarray', 'ndarray', ('argsort',
3174:     '''
3175:     a.argsort(axis=-1, kind='quicksort', order=None)
3176: 
3177:     Returns the indices that would sort this array.
3178: 
3179:     Refer to `numpy.argsort` for full documentation.
3180: 
3181:     See Also
3182:     --------
3183:     numpy.argsort : equivalent function
3184: 
3185:     '''))
3186: 
3187: 
3188: add_newdoc('numpy.core.multiarray', 'ndarray', ('argpartition',
3189:     '''
3190:     a.argpartition(kth, axis=-1, kind='introselect', order=None)
3191: 
3192:     Returns the indices that would partition this array.
3193: 
3194:     Refer to `numpy.argpartition` for full documentation.
3195: 
3196:     .. versionadded:: 1.8.0
3197: 
3198:     See Also
3199:     --------
3200:     numpy.argpartition : equivalent function
3201: 
3202:     '''))
3203: 
3204: 
3205: add_newdoc('numpy.core.multiarray', 'ndarray', ('astype',
3206:     '''
3207:     a.astype(dtype, order='K', casting='unsafe', subok=True, copy=True)
3208: 
3209:     Copy of the array, cast to a specified type.
3210: 
3211:     Parameters
3212:     ----------
3213:     dtype : str or dtype
3214:         Typecode or data-type to which the array is cast.
3215:     order : {'C', 'F', 'A', 'K'}, optional
3216:         Controls the memory layout order of the result.
3217:         'C' means C order, 'F' means Fortran order, 'A'
3218:         means 'F' order if all the arrays are Fortran contiguous,
3219:         'C' order otherwise, and 'K' means as close to the
3220:         order the array elements appear in memory as possible.
3221:         Default is 'K'.
3222:     casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
3223:         Controls what kind of data casting may occur. Defaults to 'unsafe'
3224:         for backwards compatibility.
3225: 
3226:           * 'no' means the data types should not be cast at all.
3227:           * 'equiv' means only byte-order changes are allowed.
3228:           * 'safe' means only casts which can preserve values are allowed.
3229:           * 'same_kind' means only safe casts or casts within a kind,
3230:             like float64 to float32, are allowed.
3231:           * 'unsafe' means any data conversions may be done.
3232:     subok : bool, optional
3233:         If True, then sub-classes will be passed-through (default), otherwise
3234:         the returned array will be forced to be a base-class array.
3235:     copy : bool, optional
3236:         By default, astype always returns a newly allocated array. If this
3237:         is set to false, and the `dtype`, `order`, and `subok`
3238:         requirements are satisfied, the input array is returned instead
3239:         of a copy.
3240: 
3241:     Returns
3242:     -------
3243:     arr_t : ndarray
3244:         Unless `copy` is False and the other conditions for returning the input
3245:         array are satisfied (see description for `copy` input parameter), `arr_t`
3246:         is a new array of the same shape as the input array, with dtype, order
3247:         given by `dtype`, `order`.
3248: 
3249:     Notes
3250:     -----
3251:     Starting in NumPy 1.9, astype method now returns an error if the string
3252:     dtype to cast to is not long enough in 'safe' casting mode to hold the max
3253:     value of integer/float array that is being casted. Previously the casting
3254:     was allowed even if the result was truncated.
3255: 
3256:     Raises
3257:     ------
3258:     ComplexWarning
3259:         When casting from complex to float or int. To avoid this,
3260:         one should use ``a.real.astype(t)``.
3261: 
3262:     Examples
3263:     --------
3264:     >>> x = np.array([1, 2, 2.5])
3265:     >>> x
3266:     array([ 1. ,  2. ,  2.5])
3267: 
3268:     >>> x.astype(int)
3269:     array([1, 2, 2])
3270: 
3271:     '''))
3272: 
3273: 
3274: add_newdoc('numpy.core.multiarray', 'ndarray', ('byteswap',
3275:     '''
3276:     a.byteswap(inplace)
3277: 
3278:     Swap the bytes of the array elements
3279: 
3280:     Toggle between low-endian and big-endian data representation by
3281:     returning a byteswapped array, optionally swapped in-place.
3282: 
3283:     Parameters
3284:     ----------
3285:     inplace : bool, optional
3286:         If ``True``, swap bytes in-place, default is ``False``.
3287: 
3288:     Returns
3289:     -------
3290:     out : ndarray
3291:         The byteswapped array. If `inplace` is ``True``, this is
3292:         a view to self.
3293: 
3294:     Examples
3295:     --------
3296:     >>> A = np.array([1, 256, 8755], dtype=np.int16)
3297:     >>> map(hex, A)
3298:     ['0x1', '0x100', '0x2233']
3299:     >>> A.byteswap(True)
3300:     array([  256,     1, 13090], dtype=int16)
3301:     >>> map(hex, A)
3302:     ['0x100', '0x1', '0x3322']
3303: 
3304:     Arrays of strings are not swapped
3305: 
3306:     >>> A = np.array(['ceg', 'fac'])
3307:     >>> A.byteswap()
3308:     array(['ceg', 'fac'],
3309:           dtype='|S3')
3310: 
3311:     '''))
3312: 
3313: 
3314: add_newdoc('numpy.core.multiarray', 'ndarray', ('choose',
3315:     '''
3316:     a.choose(choices, out=None, mode='raise')
3317: 
3318:     Use an index array to construct a new array from a set of choices.
3319: 
3320:     Refer to `numpy.choose` for full documentation.
3321: 
3322:     See Also
3323:     --------
3324:     numpy.choose : equivalent function
3325: 
3326:     '''))
3327: 
3328: 
3329: add_newdoc('numpy.core.multiarray', 'ndarray', ('clip',
3330:     '''
3331:     a.clip(min=None, max=None, out=None)
3332: 
3333:     Return an array whose values are limited to ``[min, max]``.
3334:     One of max or min must be given.
3335: 
3336:     Refer to `numpy.clip` for full documentation.
3337: 
3338:     See Also
3339:     --------
3340:     numpy.clip : equivalent function
3341: 
3342:     '''))
3343: 
3344: 
3345: add_newdoc('numpy.core.multiarray', 'ndarray', ('compress',
3346:     '''
3347:     a.compress(condition, axis=None, out=None)
3348: 
3349:     Return selected slices of this array along given axis.
3350: 
3351:     Refer to `numpy.compress` for full documentation.
3352: 
3353:     See Also
3354:     --------
3355:     numpy.compress : equivalent function
3356: 
3357:     '''))
3358: 
3359: 
3360: add_newdoc('numpy.core.multiarray', 'ndarray', ('conj',
3361:     '''
3362:     a.conj()
3363: 
3364:     Complex-conjugate all elements.
3365: 
3366:     Refer to `numpy.conjugate` for full documentation.
3367: 
3368:     See Also
3369:     --------
3370:     numpy.conjugate : equivalent function
3371: 
3372:     '''))
3373: 
3374: 
3375: add_newdoc('numpy.core.multiarray', 'ndarray', ('conjugate',
3376:     '''
3377:     a.conjugate()
3378: 
3379:     Return the complex conjugate, element-wise.
3380: 
3381:     Refer to `numpy.conjugate` for full documentation.
3382: 
3383:     See Also
3384:     --------
3385:     numpy.conjugate : equivalent function
3386: 
3387:     '''))
3388: 
3389: 
3390: add_newdoc('numpy.core.multiarray', 'ndarray', ('copy',
3391:     '''
3392:     a.copy(order='C')
3393: 
3394:     Return a copy of the array.
3395: 
3396:     Parameters
3397:     ----------
3398:     order : {'C', 'F', 'A', 'K'}, optional
3399:         Controls the memory layout of the copy. 'C' means C-order,
3400:         'F' means F-order, 'A' means 'F' if `a` is Fortran contiguous,
3401:         'C' otherwise. 'K' means match the layout of `a` as closely
3402:         as possible. (Note that this function and :func:numpy.copy are very
3403:         similar, but have different default values for their order=
3404:         arguments.)
3405: 
3406:     See also
3407:     --------
3408:     numpy.copy
3409:     numpy.copyto
3410: 
3411:     Examples
3412:     --------
3413:     >>> x = np.array([[1,2,3],[4,5,6]], order='F')
3414: 
3415:     >>> y = x.copy()
3416: 
3417:     >>> x.fill(0)
3418: 
3419:     >>> x
3420:     array([[0, 0, 0],
3421:            [0, 0, 0]])
3422: 
3423:     >>> y
3424:     array([[1, 2, 3],
3425:            [4, 5, 6]])
3426: 
3427:     >>> y.flags['C_CONTIGUOUS']
3428:     True
3429: 
3430:     '''))
3431: 
3432: 
3433: add_newdoc('numpy.core.multiarray', 'ndarray', ('cumprod',
3434:     '''
3435:     a.cumprod(axis=None, dtype=None, out=None)
3436: 
3437:     Return the cumulative product of the elements along the given axis.
3438: 
3439:     Refer to `numpy.cumprod` for full documentation.
3440: 
3441:     See Also
3442:     --------
3443:     numpy.cumprod : equivalent function
3444: 
3445:     '''))
3446: 
3447: 
3448: add_newdoc('numpy.core.multiarray', 'ndarray', ('cumsum',
3449:     '''
3450:     a.cumsum(axis=None, dtype=None, out=None)
3451: 
3452:     Return the cumulative sum of the elements along the given axis.
3453: 
3454:     Refer to `numpy.cumsum` for full documentation.
3455: 
3456:     See Also
3457:     --------
3458:     numpy.cumsum : equivalent function
3459: 
3460:     '''))
3461: 
3462: 
3463: add_newdoc('numpy.core.multiarray', 'ndarray', ('diagonal',
3464:     '''
3465:     a.diagonal(offset=0, axis1=0, axis2=1)
3466: 
3467:     Return specified diagonals. In NumPy 1.9 the returned array is a
3468:     read-only view instead of a copy as in previous NumPy versions.  In
3469:     a future version the read-only restriction will be removed.
3470: 
3471:     Refer to :func:`numpy.diagonal` for full documentation.
3472: 
3473:     See Also
3474:     --------
3475:     numpy.diagonal : equivalent function
3476: 
3477:     '''))
3478: 
3479: 
3480: add_newdoc('numpy.core.multiarray', 'ndarray', ('dot',
3481:     '''
3482:     a.dot(b, out=None)
3483: 
3484:     Dot product of two arrays.
3485: 
3486:     Refer to `numpy.dot` for full documentation.
3487: 
3488:     See Also
3489:     --------
3490:     numpy.dot : equivalent function
3491: 
3492:     Examples
3493:     --------
3494:     >>> a = np.eye(2)
3495:     >>> b = np.ones((2, 2)) * 2
3496:     >>> a.dot(b)
3497:     array([[ 2.,  2.],
3498:            [ 2.,  2.]])
3499: 
3500:     This array method can be conveniently chained:
3501: 
3502:     >>> a.dot(b).dot(b)
3503:     array([[ 8.,  8.],
3504:            [ 8.,  8.]])
3505: 
3506:     '''))
3507: 
3508: 
3509: add_newdoc('numpy.core.multiarray', 'ndarray', ('dump',
3510:     '''a.dump(file)
3511: 
3512:     Dump a pickle of the array to the specified file.
3513:     The array can be read back with pickle.load or numpy.load.
3514: 
3515:     Parameters
3516:     ----------
3517:     file : str
3518:         A string naming the dump file.
3519: 
3520:     '''))
3521: 
3522: 
3523: add_newdoc('numpy.core.multiarray', 'ndarray', ('dumps',
3524:     '''
3525:     a.dumps()
3526: 
3527:     Returns the pickle of the array as a string.
3528:     pickle.loads or numpy.loads will convert the string back to an array.
3529: 
3530:     Parameters
3531:     ----------
3532:     None
3533: 
3534:     '''))
3535: 
3536: 
3537: add_newdoc('numpy.core.multiarray', 'ndarray', ('fill',
3538:     '''
3539:     a.fill(value)
3540: 
3541:     Fill the array with a scalar value.
3542: 
3543:     Parameters
3544:     ----------
3545:     value : scalar
3546:         All elements of `a` will be assigned this value.
3547: 
3548:     Examples
3549:     --------
3550:     >>> a = np.array([1, 2])
3551:     >>> a.fill(0)
3552:     >>> a
3553:     array([0, 0])
3554:     >>> a = np.empty(2)
3555:     >>> a.fill(1)
3556:     >>> a
3557:     array([ 1.,  1.])
3558: 
3559:     '''))
3560: 
3561: 
3562: add_newdoc('numpy.core.multiarray', 'ndarray', ('flatten',
3563:     '''
3564:     a.flatten(order='C')
3565: 
3566:     Return a copy of the array collapsed into one dimension.
3567: 
3568:     Parameters
3569:     ----------
3570:     order : {'C', 'F', 'A', 'K'}, optional
3571:         'C' means to flatten in row-major (C-style) order.
3572:         'F' means to flatten in column-major (Fortran-
3573:         style) order. 'A' means to flatten in column-major
3574:         order if `a` is Fortran *contiguous* in memory,
3575:         row-major order otherwise. 'K' means to flatten
3576:         `a` in the order the elements occur in memory.
3577:         The default is 'C'.
3578: 
3579:     Returns
3580:     -------
3581:     y : ndarray
3582:         A copy of the input array, flattened to one dimension.
3583: 
3584:     See Also
3585:     --------
3586:     ravel : Return a flattened array.
3587:     flat : A 1-D flat iterator over the array.
3588: 
3589:     Examples
3590:     --------
3591:     >>> a = np.array([[1,2], [3,4]])
3592:     >>> a.flatten()
3593:     array([1, 2, 3, 4])
3594:     >>> a.flatten('F')
3595:     array([1, 3, 2, 4])
3596: 
3597:     '''))
3598: 
3599: 
3600: add_newdoc('numpy.core.multiarray', 'ndarray', ('getfield',
3601:     '''
3602:     a.getfield(dtype, offset=0)
3603: 
3604:     Returns a field of the given array as a certain type.
3605: 
3606:     A field is a view of the array data with a given data-type. The values in
3607:     the view are determined by the given type and the offset into the current
3608:     array in bytes. The offset needs to be such that the view dtype fits in the
3609:     array dtype; for example an array of dtype complex128 has 16-byte elements.
3610:     If taking a view with a 32-bit integer (4 bytes), the offset needs to be
3611:     between 0 and 12 bytes.
3612: 
3613:     Parameters
3614:     ----------
3615:     dtype : str or dtype
3616:         The data type of the view. The dtype size of the view can not be larger
3617:         than that of the array itself.
3618:     offset : int
3619:         Number of bytes to skip before beginning the element view.
3620: 
3621:     Examples
3622:     --------
3623:     >>> x = np.diag([1.+1.j]*2)
3624:     >>> x[1, 1] = 2 + 4.j
3625:     >>> x
3626:     array([[ 1.+1.j,  0.+0.j],
3627:            [ 0.+0.j,  2.+4.j]])
3628:     >>> x.getfield(np.float64)
3629:     array([[ 1.,  0.],
3630:            [ 0.,  2.]])
3631: 
3632:     By choosing an offset of 8 bytes we can select the complex part of the
3633:     array for our view:
3634: 
3635:     >>> x.getfield(np.float64, offset=8)
3636:     array([[ 1.,  0.],
3637:        [ 0.,  4.]])
3638: 
3639:     '''))
3640: 
3641: 
3642: add_newdoc('numpy.core.multiarray', 'ndarray', ('item',
3643:     '''
3644:     a.item(*args)
3645: 
3646:     Copy an element of an array to a standard Python scalar and return it.
3647: 
3648:     Parameters
3649:     ----------
3650:     \\*args : Arguments (variable number and type)
3651: 
3652:         * none: in this case, the method only works for arrays
3653:           with one element (`a.size == 1`), which element is
3654:           copied into a standard Python scalar object and returned.
3655: 
3656:         * int_type: this argument is interpreted as a flat index into
3657:           the array, specifying which element to copy and return.
3658: 
3659:         * tuple of int_types: functions as does a single int_type argument,
3660:           except that the argument is interpreted as an nd-index into the
3661:           array.
3662: 
3663:     Returns
3664:     -------
3665:     z : Standard Python scalar object
3666:         A copy of the specified element of the array as a suitable
3667:         Python scalar
3668: 
3669:     Notes
3670:     -----
3671:     When the data type of `a` is longdouble or clongdouble, item() returns
3672:     a scalar array object because there is no available Python scalar that
3673:     would not lose information. Void arrays return a buffer object for item(),
3674:     unless fields are defined, in which case a tuple is returned.
3675: 
3676:     `item` is very similar to a[args], except, instead of an array scalar,
3677:     a standard Python scalar is returned. This can be useful for speeding up
3678:     access to elements of the array and doing arithmetic on elements of the
3679:     array using Python's optimized math.
3680: 
3681:     Examples
3682:     --------
3683:     >>> x = np.random.randint(9, size=(3, 3))
3684:     >>> x
3685:     array([[3, 1, 7],
3686:            [2, 8, 3],
3687:            [8, 5, 3]])
3688:     >>> x.item(3)
3689:     2
3690:     >>> x.item(7)
3691:     5
3692:     >>> x.item((0, 1))
3693:     1
3694:     >>> x.item((2, 2))
3695:     3
3696: 
3697:     '''))
3698: 
3699: 
3700: add_newdoc('numpy.core.multiarray', 'ndarray', ('itemset',
3701:     '''
3702:     a.itemset(*args)
3703: 
3704:     Insert scalar into an array (scalar is cast to array's dtype, if possible)
3705: 
3706:     There must be at least 1 argument, and define the last argument
3707:     as *item*.  Then, ``a.itemset(*args)`` is equivalent to but faster
3708:     than ``a[args] = item``.  The item should be a scalar value and `args`
3709:     must select a single item in the array `a`.
3710: 
3711:     Parameters
3712:     ----------
3713:     \*args : Arguments
3714:         If one argument: a scalar, only used in case `a` is of size 1.
3715:         If two arguments: the last argument is the value to be set
3716:         and must be a scalar, the first argument specifies a single array
3717:         element location. It is either an int or a tuple.
3718: 
3719:     Notes
3720:     -----
3721:     Compared to indexing syntax, `itemset` provides some speed increase
3722:     for placing a scalar into a particular location in an `ndarray`,
3723:     if you must do this.  However, generally this is discouraged:
3724:     among other problems, it complicates the appearance of the code.
3725:     Also, when using `itemset` (and `item`) inside a loop, be sure
3726:     to assign the methods to a local variable to avoid the attribute
3727:     look-up at each loop iteration.
3728: 
3729:     Examples
3730:     --------
3731:     >>> x = np.random.randint(9, size=(3, 3))
3732:     >>> x
3733:     array([[3, 1, 7],
3734:            [2, 8, 3],
3735:            [8, 5, 3]])
3736:     >>> x.itemset(4, 0)
3737:     >>> x.itemset((2, 2), 9)
3738:     >>> x
3739:     array([[3, 1, 7],
3740:            [2, 0, 3],
3741:            [8, 5, 9]])
3742: 
3743:     '''))
3744: 
3745: 
3746: add_newdoc('numpy.core.multiarray', 'ndarray', ('max',
3747:     '''
3748:     a.max(axis=None, out=None)
3749: 
3750:     Return the maximum along a given axis.
3751: 
3752:     Refer to `numpy.amax` for full documentation.
3753: 
3754:     See Also
3755:     --------
3756:     numpy.amax : equivalent function
3757: 
3758:     '''))
3759: 
3760: 
3761: add_newdoc('numpy.core.multiarray', 'ndarray', ('mean',
3762:     '''
3763:     a.mean(axis=None, dtype=None, out=None, keepdims=False)
3764: 
3765:     Returns the average of the array elements along given axis.
3766: 
3767:     Refer to `numpy.mean` for full documentation.
3768: 
3769:     See Also
3770:     --------
3771:     numpy.mean : equivalent function
3772: 
3773:     '''))
3774: 
3775: 
3776: add_newdoc('numpy.core.multiarray', 'ndarray', ('min',
3777:     '''
3778:     a.min(axis=None, out=None, keepdims=False)
3779: 
3780:     Return the minimum along a given axis.
3781: 
3782:     Refer to `numpy.amin` for full documentation.
3783: 
3784:     See Also
3785:     --------
3786:     numpy.amin : equivalent function
3787: 
3788:     '''))
3789: 
3790: 
3791: add_newdoc('numpy.core.multiarray', 'shares_memory',
3792:     '''
3793:     shares_memory(a, b, max_work=None)
3794: 
3795:     Determine if two arrays share memory
3796: 
3797:     Parameters
3798:     ----------
3799:     a, b : ndarray
3800:         Input arrays
3801:     max_work : int, optional
3802:         Effort to spend on solving the overlap problem (maximum number
3803:         of candidate solutions to consider). The following special
3804:         values are recognized:
3805: 
3806:         max_work=MAY_SHARE_EXACT  (default)
3807:             The problem is solved exactly. In this case, the function returns
3808:             True only if there is an element shared between the arrays.
3809:         max_work=MAY_SHARE_BOUNDS
3810:             Only the memory bounds of a and b are checked.
3811: 
3812:     Raises
3813:     ------
3814:     numpy.TooHardError
3815:         Exceeded max_work.
3816: 
3817:     Returns
3818:     -------
3819:     out : bool
3820: 
3821:     See Also
3822:     --------
3823:     may_share_memory
3824: 
3825:     Examples
3826:     --------
3827:     >>> np.may_share_memory(np.array([1,2]), np.array([5,8,9]))
3828:     False
3829: 
3830:     ''')
3831: 
3832: 
3833: add_newdoc('numpy.core.multiarray', 'may_share_memory',
3834:     '''
3835:     may_share_memory(a, b, max_work=None)
3836: 
3837:     Determine if two arrays might share memory
3838: 
3839:     A return of True does not necessarily mean that the two arrays
3840:     share any element.  It just means that they *might*.
3841: 
3842:     Only the memory bounds of a and b are checked by default.
3843: 
3844:     Parameters
3845:     ----------
3846:     a, b : ndarray
3847:         Input arrays
3848:     max_work : int, optional
3849:         Effort to spend on solving the overlap problem.  See
3850:         `shares_memory` for details.  Default for ``may_share_memory``
3851:         is to do a bounds check.
3852: 
3853:     Returns
3854:     -------
3855:     out : bool
3856: 
3857:     See Also
3858:     --------
3859:     shares_memory
3860: 
3861:     Examples
3862:     --------
3863:     >>> np.may_share_memory(np.array([1,2]), np.array([5,8,9]))
3864:     False
3865:     >>> x = np.zeros([3, 4])
3866:     >>> np.may_share_memory(x[:,0], x[:,1])
3867:     True
3868: 
3869:     ''')
3870: 
3871: 
3872: add_newdoc('numpy.core.multiarray', 'ndarray', ('newbyteorder',
3873:     '''
3874:     arr.newbyteorder(new_order='S')
3875: 
3876:     Return the array with the same data viewed with a different byte order.
3877: 
3878:     Equivalent to::
3879: 
3880:         arr.view(arr.dtype.newbytorder(new_order))
3881: 
3882:     Changes are also made in all fields and sub-arrays of the array data
3883:     type.
3884: 
3885: 
3886: 
3887:     Parameters
3888:     ----------
3889:     new_order : string, optional
3890:         Byte order to force; a value from the byte order specifications
3891:         below. `new_order` codes can be any of:
3892: 
3893:         * 'S' - swap dtype from current to opposite endian
3894:         * {'<', 'L'} - little endian
3895:         * {'>', 'B'} - big endian
3896:         * {'=', 'N'} - native order
3897:         * {'|', 'I'} - ignore (no change to byte order)
3898: 
3899:         The default value ('S') results in swapping the current
3900:         byte order. The code does a case-insensitive check on the first
3901:         letter of `new_order` for the alternatives above.  For example,
3902:         any of 'B' or 'b' or 'biggish' are valid to specify big-endian.
3903: 
3904: 
3905:     Returns
3906:     -------
3907:     new_arr : array
3908:         New array object with the dtype reflecting given change to the
3909:         byte order.
3910: 
3911:     '''))
3912: 
3913: 
3914: add_newdoc('numpy.core.multiarray', 'ndarray', ('nonzero',
3915:     '''
3916:     a.nonzero()
3917: 
3918:     Return the indices of the elements that are non-zero.
3919: 
3920:     Refer to `numpy.nonzero` for full documentation.
3921: 
3922:     See Also
3923:     --------
3924:     numpy.nonzero : equivalent function
3925: 
3926:     '''))
3927: 
3928: 
3929: add_newdoc('numpy.core.multiarray', 'ndarray', ('prod',
3930:     '''
3931:     a.prod(axis=None, dtype=None, out=None, keepdims=False)
3932: 
3933:     Return the product of the array elements over the given axis
3934: 
3935:     Refer to `numpy.prod` for full documentation.
3936: 
3937:     See Also
3938:     --------
3939:     numpy.prod : equivalent function
3940: 
3941:     '''))
3942: 
3943: 
3944: add_newdoc('numpy.core.multiarray', 'ndarray', ('ptp',
3945:     '''
3946:     a.ptp(axis=None, out=None)
3947: 
3948:     Peak to peak (maximum - minimum) value along a given axis.
3949: 
3950:     Refer to `numpy.ptp` for full documentation.
3951: 
3952:     See Also
3953:     --------
3954:     numpy.ptp : equivalent function
3955: 
3956:     '''))
3957: 
3958: 
3959: add_newdoc('numpy.core.multiarray', 'ndarray', ('put',
3960:     '''
3961:     a.put(indices, values, mode='raise')
3962: 
3963:     Set ``a.flat[n] = values[n]`` for all `n` in indices.
3964: 
3965:     Refer to `numpy.put` for full documentation.
3966: 
3967:     See Also
3968:     --------
3969:     numpy.put : equivalent function
3970: 
3971:     '''))
3972: 
3973: add_newdoc('numpy.core.multiarray', 'copyto',
3974:     '''
3975:     copyto(dst, src, casting='same_kind', where=None)
3976: 
3977:     Copies values from one array to another, broadcasting as necessary.
3978: 
3979:     Raises a TypeError if the `casting` rule is violated, and if
3980:     `where` is provided, it selects which elements to copy.
3981: 
3982:     .. versionadded:: 1.7.0
3983: 
3984:     Parameters
3985:     ----------
3986:     dst : ndarray
3987:         The array into which values are copied.
3988:     src : array_like
3989:         The array from which values are copied.
3990:     casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
3991:         Controls what kind of data casting may occur when copying.
3992: 
3993:           * 'no' means the data types should not be cast at all.
3994:           * 'equiv' means only byte-order changes are allowed.
3995:           * 'safe' means only casts which can preserve values are allowed.
3996:           * 'same_kind' means only safe casts or casts within a kind,
3997:             like float64 to float32, are allowed.
3998:           * 'unsafe' means any data conversions may be done.
3999:     where : array_like of bool, optional
4000:         A boolean array which is broadcasted to match the dimensions
4001:         of `dst`, and selects elements to copy from `src` to `dst`
4002:         wherever it contains the value True.
4003: 
4004:     ''')
4005: 
4006: add_newdoc('numpy.core.multiarray', 'putmask',
4007:     '''
4008:     putmask(a, mask, values)
4009: 
4010:     Changes elements of an array based on conditional and input values.
4011: 
4012:     Sets ``a.flat[n] = values[n]`` for each n where ``mask.flat[n]==True``.
4013: 
4014:     If `values` is not the same size as `a` and `mask` then it will repeat.
4015:     This gives behavior different from ``a[mask] = values``.
4016: 
4017:     Parameters
4018:     ----------
4019:     a : array_like
4020:         Target array.
4021:     mask : array_like
4022:         Boolean mask array. It has to be the same shape as `a`.
4023:     values : array_like
4024:         Values to put into `a` where `mask` is True. If `values` is smaller
4025:         than `a` it will be repeated.
4026: 
4027:     See Also
4028:     --------
4029:     place, put, take, copyto
4030: 
4031:     Examples
4032:     --------
4033:     >>> x = np.arange(6).reshape(2, 3)
4034:     >>> np.putmask(x, x>2, x**2)
4035:     >>> x
4036:     array([[ 0,  1,  2],
4037:            [ 9, 16, 25]])
4038: 
4039:     If `values` is smaller than `a` it is repeated:
4040: 
4041:     >>> x = np.arange(5)
4042:     >>> np.putmask(x, x>1, [-33, -44])
4043:     >>> x
4044:     array([  0,   1, -33, -44, -33])
4045: 
4046:     ''')
4047: 
4048: 
4049: add_newdoc('numpy.core.multiarray', 'ndarray', ('ravel',
4050:     '''
4051:     a.ravel([order])
4052: 
4053:     Return a flattened array.
4054: 
4055:     Refer to `numpy.ravel` for full documentation.
4056: 
4057:     See Also
4058:     --------
4059:     numpy.ravel : equivalent function
4060: 
4061:     ndarray.flat : a flat iterator on the array.
4062: 
4063:     '''))
4064: 
4065: 
4066: add_newdoc('numpy.core.multiarray', 'ndarray', ('repeat',
4067:     '''
4068:     a.repeat(repeats, axis=None)
4069: 
4070:     Repeat elements of an array.
4071: 
4072:     Refer to `numpy.repeat` for full documentation.
4073: 
4074:     See Also
4075:     --------
4076:     numpy.repeat : equivalent function
4077: 
4078:     '''))
4079: 
4080: 
4081: add_newdoc('numpy.core.multiarray', 'ndarray', ('reshape',
4082:     '''
4083:     a.reshape(shape, order='C')
4084: 
4085:     Returns an array containing the same data with a new shape.
4086: 
4087:     Refer to `numpy.reshape` for full documentation.
4088: 
4089:     See Also
4090:     --------
4091:     numpy.reshape : equivalent function
4092: 
4093:     '''))
4094: 
4095: 
4096: add_newdoc('numpy.core.multiarray', 'ndarray', ('resize',
4097:     '''
4098:     a.resize(new_shape, refcheck=True)
4099: 
4100:     Change shape and size of array in-place.
4101: 
4102:     Parameters
4103:     ----------
4104:     new_shape : tuple of ints, or `n` ints
4105:         Shape of resized array.
4106:     refcheck : bool, optional
4107:         If False, reference count will not be checked. Default is True.
4108: 
4109:     Returns
4110:     -------
4111:     None
4112: 
4113:     Raises
4114:     ------
4115:     ValueError
4116:         If `a` does not own its own data or references or views to it exist,
4117:         and the data memory must be changed.
4118: 
4119:     SystemError
4120:         If the `order` keyword argument is specified. This behaviour is a
4121:         bug in NumPy.
4122: 
4123:     See Also
4124:     --------
4125:     resize : Return a new array with the specified shape.
4126: 
4127:     Notes
4128:     -----
4129:     This reallocates space for the data area if necessary.
4130: 
4131:     Only contiguous arrays (data elements consecutive in memory) can be
4132:     resized.
4133: 
4134:     The purpose of the reference count check is to make sure you
4135:     do not use this array as a buffer for another Python object and then
4136:     reallocate the memory. However, reference counts can increase in
4137:     other ways so if you are sure that you have not shared the memory
4138:     for this array with another Python object, then you may safely set
4139:     `refcheck` to False.
4140: 
4141:     Examples
4142:     --------
4143:     Shrinking an array: array is flattened (in the order that the data are
4144:     stored in memory), resized, and reshaped:
4145: 
4146:     >>> a = np.array([[0, 1], [2, 3]], order='C')
4147:     >>> a.resize((2, 1))
4148:     >>> a
4149:     array([[0],
4150:            [1]])
4151: 
4152:     >>> a = np.array([[0, 1], [2, 3]], order='F')
4153:     >>> a.resize((2, 1))
4154:     >>> a
4155:     array([[0],
4156:            [2]])
4157: 
4158:     Enlarging an array: as above, but missing entries are filled with zeros:
4159: 
4160:     >>> b = np.array([[0, 1], [2, 3]])
4161:     >>> b.resize(2, 3) # new_shape parameter doesn't have to be a tuple
4162:     >>> b
4163:     array([[0, 1, 2],
4164:            [3, 0, 0]])
4165: 
4166:     Referencing an array prevents resizing...
4167: 
4168:     >>> c = a
4169:     >>> a.resize((1, 1))
4170:     Traceback (most recent call last):
4171:     ...
4172:     ValueError: cannot resize an array that has been referenced ...
4173: 
4174:     Unless `refcheck` is False:
4175: 
4176:     >>> a.resize((1, 1), refcheck=False)
4177:     >>> a
4178:     array([[0]])
4179:     >>> c
4180:     array([[0]])
4181: 
4182:     '''))
4183: 
4184: 
4185: add_newdoc('numpy.core.multiarray', 'ndarray', ('round',
4186:     '''
4187:     a.round(decimals=0, out=None)
4188: 
4189:     Return `a` with each element rounded to the given number of decimals.
4190: 
4191:     Refer to `numpy.around` for full documentation.
4192: 
4193:     See Also
4194:     --------
4195:     numpy.around : equivalent function
4196: 
4197:     '''))
4198: 
4199: 
4200: add_newdoc('numpy.core.multiarray', 'ndarray', ('searchsorted',
4201:     '''
4202:     a.searchsorted(v, side='left', sorter=None)
4203: 
4204:     Find indices where elements of v should be inserted in a to maintain order.
4205: 
4206:     For full documentation, see `numpy.searchsorted`
4207: 
4208:     See Also
4209:     --------
4210:     numpy.searchsorted : equivalent function
4211: 
4212:     '''))
4213: 
4214: 
4215: add_newdoc('numpy.core.multiarray', 'ndarray', ('setfield',
4216:     '''
4217:     a.setfield(val, dtype, offset=0)
4218: 
4219:     Put a value into a specified place in a field defined by a data-type.
4220: 
4221:     Place `val` into `a`'s field defined by `dtype` and beginning `offset`
4222:     bytes into the field.
4223: 
4224:     Parameters
4225:     ----------
4226:     val : object
4227:         Value to be placed in field.
4228:     dtype : dtype object
4229:         Data-type of the field in which to place `val`.
4230:     offset : int, optional
4231:         The number of bytes into the field at which to place `val`.
4232: 
4233:     Returns
4234:     -------
4235:     None
4236: 
4237:     See Also
4238:     --------
4239:     getfield
4240: 
4241:     Examples
4242:     --------
4243:     >>> x = np.eye(3)
4244:     >>> x.getfield(np.float64)
4245:     array([[ 1.,  0.,  0.],
4246:            [ 0.,  1.,  0.],
4247:            [ 0.,  0.,  1.]])
4248:     >>> x.setfield(3, np.int32)
4249:     >>> x.getfield(np.int32)
4250:     array([[3, 3, 3],
4251:            [3, 3, 3],
4252:            [3, 3, 3]])
4253:     >>> x
4254:     array([[  1.00000000e+000,   1.48219694e-323,   1.48219694e-323],
4255:            [  1.48219694e-323,   1.00000000e+000,   1.48219694e-323],
4256:            [  1.48219694e-323,   1.48219694e-323,   1.00000000e+000]])
4257:     >>> x.setfield(np.eye(3), np.int32)
4258:     >>> x
4259:     array([[ 1.,  0.,  0.],
4260:            [ 0.,  1.,  0.],
4261:            [ 0.,  0.,  1.]])
4262: 
4263:     '''))
4264: 
4265: 
4266: add_newdoc('numpy.core.multiarray', 'ndarray', ('setflags',
4267:     '''
4268:     a.setflags(write=None, align=None, uic=None)
4269: 
4270:     Set array flags WRITEABLE, ALIGNED, and UPDATEIFCOPY, respectively.
4271: 
4272:     These Boolean-valued flags affect how numpy interprets the memory
4273:     area used by `a` (see Notes below). The ALIGNED flag can only
4274:     be set to True if the data is actually aligned according to the type.
4275:     The UPDATEIFCOPY flag can never be set to True. The flag WRITEABLE
4276:     can only be set to True if the array owns its own memory, or the
4277:     ultimate owner of the memory exposes a writeable buffer interface,
4278:     or is a string. (The exception for string is made so that unpickling
4279:     can be done without copying memory.)
4280: 
4281:     Parameters
4282:     ----------
4283:     write : bool, optional
4284:         Describes whether or not `a` can be written to.
4285:     align : bool, optional
4286:         Describes whether or not `a` is aligned properly for its type.
4287:     uic : bool, optional
4288:         Describes whether or not `a` is a copy of another "base" array.
4289: 
4290:     Notes
4291:     -----
4292:     Array flags provide information about how the memory area used
4293:     for the array is to be interpreted. There are 6 Boolean flags
4294:     in use, only three of which can be changed by the user:
4295:     UPDATEIFCOPY, WRITEABLE, and ALIGNED.
4296: 
4297:     WRITEABLE (W) the data area can be written to;
4298: 
4299:     ALIGNED (A) the data and strides are aligned appropriately for the hardware
4300:     (as determined by the compiler);
4301: 
4302:     UPDATEIFCOPY (U) this array is a copy of some other array (referenced
4303:     by .base). When this array is deallocated, the base array will be
4304:     updated with the contents of this array.
4305: 
4306:     All flags can be accessed using their first (upper case) letter as well
4307:     as the full name.
4308: 
4309:     Examples
4310:     --------
4311:     >>> y
4312:     array([[3, 1, 7],
4313:            [2, 0, 0],
4314:            [8, 5, 9]])
4315:     >>> y.flags
4316:       C_CONTIGUOUS : True
4317:       F_CONTIGUOUS : False
4318:       OWNDATA : True
4319:       WRITEABLE : True
4320:       ALIGNED : True
4321:       UPDATEIFCOPY : False
4322:     >>> y.setflags(write=0, align=0)
4323:     >>> y.flags
4324:       C_CONTIGUOUS : True
4325:       F_CONTIGUOUS : False
4326:       OWNDATA : True
4327:       WRITEABLE : False
4328:       ALIGNED : False
4329:       UPDATEIFCOPY : False
4330:     >>> y.setflags(uic=1)
4331:     Traceback (most recent call last):
4332:       File "<stdin>", line 1, in <module>
4333:     ValueError: cannot set UPDATEIFCOPY flag to True
4334: 
4335:     '''))
4336: 
4337: 
4338: add_newdoc('numpy.core.multiarray', 'ndarray', ('sort',
4339:     '''
4340:     a.sort(axis=-1, kind='quicksort', order=None)
4341: 
4342:     Sort an array, in-place.
4343: 
4344:     Parameters
4345:     ----------
4346:     axis : int, optional
4347:         Axis along which to sort. Default is -1, which means sort along the
4348:         last axis.
4349:     kind : {'quicksort', 'mergesort', 'heapsort'}, optional
4350:         Sorting algorithm. Default is 'quicksort'.
4351:     order : str or list of str, optional
4352:         When `a` is an array with fields defined, this argument specifies
4353:         which fields to compare first, second, etc.  A single field can
4354:         be specified as a string, and not all fields need be specified,
4355:         but unspecified fields will still be used, in the order in which
4356:         they come up in the dtype, to break ties.
4357: 
4358:     See Also
4359:     --------
4360:     numpy.sort : Return a sorted copy of an array.
4361:     argsort : Indirect sort.
4362:     lexsort : Indirect stable sort on multiple keys.
4363:     searchsorted : Find elements in sorted array.
4364:     partition: Partial sort.
4365: 
4366:     Notes
4367:     -----
4368:     See ``sort`` for notes on the different sorting algorithms.
4369: 
4370:     Examples
4371:     --------
4372:     >>> a = np.array([[1,4], [3,1]])
4373:     >>> a.sort(axis=1)
4374:     >>> a
4375:     array([[1, 4],
4376:            [1, 3]])
4377:     >>> a.sort(axis=0)
4378:     >>> a
4379:     array([[1, 3],
4380:            [1, 4]])
4381: 
4382:     Use the `order` keyword to specify a field to use when sorting a
4383:     structured array:
4384: 
4385:     >>> a = np.array([('a', 2), ('c', 1)], dtype=[('x', 'S1'), ('y', int)])
4386:     >>> a.sort(order='y')
4387:     >>> a
4388:     array([('c', 1), ('a', 2)],
4389:           dtype=[('x', '|S1'), ('y', '<i4')])
4390: 
4391:     '''))
4392: 
4393: 
4394: add_newdoc('numpy.core.multiarray', 'ndarray', ('partition',
4395:     '''
4396:     a.partition(kth, axis=-1, kind='introselect', order=None)
4397: 
4398:     Rearranges the elements in the array in such a way that value of the
4399:     element in kth position is in the position it would be in a sorted array.
4400:     All elements smaller than the kth element are moved before this element and
4401:     all equal or greater are moved behind it. The ordering of the elements in
4402:     the two partitions is undefined.
4403: 
4404:     .. versionadded:: 1.8.0
4405: 
4406:     Parameters
4407:     ----------
4408:     kth : int or sequence of ints
4409:         Element index to partition by. The kth element value will be in its
4410:         final sorted position and all smaller elements will be moved before it
4411:         and all equal or greater elements behind it.
4412:         The order all elements in the partitions is undefined.
4413:         If provided with a sequence of kth it will partition all elements
4414:         indexed by kth of them into their sorted position at once.
4415:     axis : int, optional
4416:         Axis along which to sort. Default is -1, which means sort along the
4417:         last axis.
4418:     kind : {'introselect'}, optional
4419:         Selection algorithm. Default is 'introselect'.
4420:     order : str or list of str, optional
4421:         When `a` is an array with fields defined, this argument specifies
4422:         which fields to compare first, second, etc.  A single field can
4423:         be specified as a string, and not all fields need be specified,
4424:         but unspecified fields will still be used, in the order in which
4425:         they come up in the dtype, to break ties.
4426: 
4427:     See Also
4428:     --------
4429:     numpy.partition : Return a parititioned copy of an array.
4430:     argpartition : Indirect partition.
4431:     sort : Full sort.
4432: 
4433:     Notes
4434:     -----
4435:     See ``np.partition`` for notes on the different algorithms.
4436: 
4437:     Examples
4438:     --------
4439:     >>> a = np.array([3, 4, 2, 1])
4440:     >>> a.partition(a, 3)
4441:     >>> a
4442:     array([2, 1, 3, 4])
4443: 
4444:     >>> a.partition((1, 3))
4445:     array([1, 2, 3, 4])
4446:     '''))
4447: 
4448: 
4449: add_newdoc('numpy.core.multiarray', 'ndarray', ('squeeze',
4450:     '''
4451:     a.squeeze(axis=None)
4452: 
4453:     Remove single-dimensional entries from the shape of `a`.
4454: 
4455:     Refer to `numpy.squeeze` for full documentation.
4456: 
4457:     See Also
4458:     --------
4459:     numpy.squeeze : equivalent function
4460: 
4461:     '''))
4462: 
4463: 
4464: add_newdoc('numpy.core.multiarray', 'ndarray', ('std',
4465:     '''
4466:     a.std(axis=None, dtype=None, out=None, ddof=0, keepdims=False)
4467: 
4468:     Returns the standard deviation of the array elements along given axis.
4469: 
4470:     Refer to `numpy.std` for full documentation.
4471: 
4472:     See Also
4473:     --------
4474:     numpy.std : equivalent function
4475: 
4476:     '''))
4477: 
4478: 
4479: add_newdoc('numpy.core.multiarray', 'ndarray', ('sum',
4480:     '''
4481:     a.sum(axis=None, dtype=None, out=None, keepdims=False)
4482: 
4483:     Return the sum of the array elements over the given axis.
4484: 
4485:     Refer to `numpy.sum` for full documentation.
4486: 
4487:     See Also
4488:     --------
4489:     numpy.sum : equivalent function
4490: 
4491:     '''))
4492: 
4493: 
4494: add_newdoc('numpy.core.multiarray', 'ndarray', ('swapaxes',
4495:     '''
4496:     a.swapaxes(axis1, axis2)
4497: 
4498:     Return a view of the array with `axis1` and `axis2` interchanged.
4499: 
4500:     Refer to `numpy.swapaxes` for full documentation.
4501: 
4502:     See Also
4503:     --------
4504:     numpy.swapaxes : equivalent function
4505: 
4506:     '''))
4507: 
4508: 
4509: add_newdoc('numpy.core.multiarray', 'ndarray', ('take',
4510:     '''
4511:     a.take(indices, axis=None, out=None, mode='raise')
4512: 
4513:     Return an array formed from the elements of `a` at the given indices.
4514: 
4515:     Refer to `numpy.take` for full documentation.
4516: 
4517:     See Also
4518:     --------
4519:     numpy.take : equivalent function
4520: 
4521:     '''))
4522: 
4523: 
4524: add_newdoc('numpy.core.multiarray', 'ndarray', ('tofile',
4525:     '''
4526:     a.tofile(fid, sep="", format="%s")
4527: 
4528:     Write array to a file as text or binary (default).
4529: 
4530:     Data is always written in 'C' order, independent of the order of `a`.
4531:     The data produced by this method can be recovered using the function
4532:     fromfile().
4533: 
4534:     Parameters
4535:     ----------
4536:     fid : file or str
4537:         An open file object, or a string containing a filename.
4538:     sep : str
4539:         Separator between array items for text output.
4540:         If "" (empty), a binary file is written, equivalent to
4541:         ``file.write(a.tobytes())``.
4542:     format : str
4543:         Format string for text file output.
4544:         Each entry in the array is formatted to text by first converting
4545:         it to the closest Python type, and then using "format" % item.
4546: 
4547:     Notes
4548:     -----
4549:     This is a convenience function for quick storage of array data.
4550:     Information on endianness and precision is lost, so this method is not a
4551:     good choice for files intended to archive data or transport data between
4552:     machines with different endianness. Some of these problems can be overcome
4553:     by outputting the data as text files, at the expense of speed and file
4554:     size.
4555: 
4556:     '''))
4557: 
4558: 
4559: add_newdoc('numpy.core.multiarray', 'ndarray', ('tolist',
4560:     '''
4561:     a.tolist()
4562: 
4563:     Return the array as a (possibly nested) list.
4564: 
4565:     Return a copy of the array data as a (nested) Python list.
4566:     Data items are converted to the nearest compatible Python type.
4567: 
4568:     Parameters
4569:     ----------
4570:     none
4571: 
4572:     Returns
4573:     -------
4574:     y : list
4575:         The possibly nested list of array elements.
4576: 
4577:     Notes
4578:     -----
4579:     The array may be recreated, ``a = np.array(a.tolist())``.
4580: 
4581:     Examples
4582:     --------
4583:     >>> a = np.array([1, 2])
4584:     >>> a.tolist()
4585:     [1, 2]
4586:     >>> a = np.array([[1, 2], [3, 4]])
4587:     >>> list(a)
4588:     [array([1, 2]), array([3, 4])]
4589:     >>> a.tolist()
4590:     [[1, 2], [3, 4]]
4591: 
4592:     '''))
4593: 
4594: 
4595: tobytesdoc = '''
4596:     a.{name}(order='C')
4597: 
4598:     Construct Python bytes containing the raw data bytes in the array.
4599: 
4600:     Constructs Python bytes showing a copy of the raw contents of
4601:     data memory. The bytes object can be produced in either 'C' or 'Fortran',
4602:     or 'Any' order (the default is 'C'-order). 'Any' order means C-order
4603:     unless the F_CONTIGUOUS flag in the array is set, in which case it
4604:     means 'Fortran' order.
4605: 
4606:     {deprecated}
4607: 
4608:     Parameters
4609:     ----------
4610:     order : {{'C', 'F', None}}, optional
4611:         Order of the data for multidimensional arrays:
4612:         C, Fortran, or the same as for the original array.
4613: 
4614:     Returns
4615:     -------
4616:     s : bytes
4617:         Python bytes exhibiting a copy of `a`'s raw data.
4618: 
4619:     Examples
4620:     --------
4621:     >>> x = np.array([[0, 1], [2, 3]])
4622:     >>> x.tobytes()
4623:     b'\\x00\\x00\\x00\\x00\\x01\\x00\\x00\\x00\\x02\\x00\\x00\\x00\\x03\\x00\\x00\\x00'
4624:     >>> x.tobytes('C') == x.tobytes()
4625:     True
4626:     >>> x.tobytes('F')
4627:     b'\\x00\\x00\\x00\\x00\\x02\\x00\\x00\\x00\\x01\\x00\\x00\\x00\\x03\\x00\\x00\\x00'
4628: 
4629:     '''
4630: 
4631: add_newdoc('numpy.core.multiarray', 'ndarray',
4632:            ('tostring', tobytesdoc.format(name='tostring',
4633:                                           deprecated=
4634:                                           'This function is a compatibility '
4635:                                           'alias for tobytes. Despite its '
4636:                                           'name it returns bytes not '
4637:                                           'strings.')))
4638: add_newdoc('numpy.core.multiarray', 'ndarray',
4639:            ('tobytes', tobytesdoc.format(name='tobytes',
4640:                                          deprecated='.. versionadded:: 1.9.0')))
4641: 
4642: add_newdoc('numpy.core.multiarray', 'ndarray', ('trace',
4643:     '''
4644:     a.trace(offset=0, axis1=0, axis2=1, dtype=None, out=None)
4645: 
4646:     Return the sum along diagonals of the array.
4647: 
4648:     Refer to `numpy.trace` for full documentation.
4649: 
4650:     See Also
4651:     --------
4652:     numpy.trace : equivalent function
4653: 
4654:     '''))
4655: 
4656: 
4657: add_newdoc('numpy.core.multiarray', 'ndarray', ('transpose',
4658:     '''
4659:     a.transpose(*axes)
4660: 
4661:     Returns a view of the array with axes transposed.
4662: 
4663:     For a 1-D array, this has no effect. (To change between column and
4664:     row vectors, first cast the 1-D array into a matrix object.)
4665:     For a 2-D array, this is the usual matrix transpose.
4666:     For an n-D array, if axes are given, their order indicates how the
4667:     axes are permuted (see Examples). If axes are not provided and
4668:     ``a.shape = (i[0], i[1], ... i[n-2], i[n-1])``, then
4669:     ``a.transpose().shape = (i[n-1], i[n-2], ... i[1], i[0])``.
4670: 
4671:     Parameters
4672:     ----------
4673:     axes : None, tuple of ints, or `n` ints
4674: 
4675:      * None or no argument: reverses the order of the axes.
4676: 
4677:      * tuple of ints: `i` in the `j`-th place in the tuple means `a`'s
4678:        `i`-th axis becomes `a.transpose()`'s `j`-th axis.
4679: 
4680:      * `n` ints: same as an n-tuple of the same ints (this form is
4681:        intended simply as a "convenience" alternative to the tuple form)
4682: 
4683:     Returns
4684:     -------
4685:     out : ndarray
4686:         View of `a`, with axes suitably permuted.
4687: 
4688:     See Also
4689:     --------
4690:     ndarray.T : Array property returning the array transposed.
4691: 
4692:     Examples
4693:     --------
4694:     >>> a = np.array([[1, 2], [3, 4]])
4695:     >>> a
4696:     array([[1, 2],
4697:            [3, 4]])
4698:     >>> a.transpose()
4699:     array([[1, 3],
4700:            [2, 4]])
4701:     >>> a.transpose((1, 0))
4702:     array([[1, 3],
4703:            [2, 4]])
4704:     >>> a.transpose(1, 0)
4705:     array([[1, 3],
4706:            [2, 4]])
4707: 
4708:     '''))
4709: 
4710: 
4711: add_newdoc('numpy.core.multiarray', 'ndarray', ('var',
4712:     '''
4713:     a.var(axis=None, dtype=None, out=None, ddof=0, keepdims=False)
4714: 
4715:     Returns the variance of the array elements, along given axis.
4716: 
4717:     Refer to `numpy.var` for full documentation.
4718: 
4719:     See Also
4720:     --------
4721:     numpy.var : equivalent function
4722: 
4723:     '''))
4724: 
4725: 
4726: add_newdoc('numpy.core.multiarray', 'ndarray', ('view',
4727:     '''
4728:     a.view(dtype=None, type=None)
4729: 
4730:     New view of array with the same data.
4731: 
4732:     Parameters
4733:     ----------
4734:     dtype : data-type or ndarray sub-class, optional
4735:         Data-type descriptor of the returned view, e.g., float32 or int16. The
4736:         default, None, results in the view having the same data-type as `a`.
4737:         This argument can also be specified as an ndarray sub-class, which
4738:         then specifies the type of the returned object (this is equivalent to
4739:         setting the ``type`` parameter).
4740:     type : Python type, optional
4741:         Type of the returned view, e.g., ndarray or matrix.  Again, the
4742:         default None results in type preservation.
4743: 
4744:     Notes
4745:     -----
4746:     ``a.view()`` is used two different ways:
4747: 
4748:     ``a.view(some_dtype)`` or ``a.view(dtype=some_dtype)`` constructs a view
4749:     of the array's memory with a different data-type.  This can cause a
4750:     reinterpretation of the bytes of memory.
4751: 
4752:     ``a.view(ndarray_subclass)`` or ``a.view(type=ndarray_subclass)`` just
4753:     returns an instance of `ndarray_subclass` that looks at the same array
4754:     (same shape, dtype, etc.)  This does not cause a reinterpretation of the
4755:     memory.
4756: 
4757:     For ``a.view(some_dtype)``, if ``some_dtype`` has a different number of
4758:     bytes per entry than the previous dtype (for example, converting a
4759:     regular array to a structured array), then the behavior of the view
4760:     cannot be predicted just from the superficial appearance of ``a`` (shown
4761:     by ``print(a)``). It also depends on exactly how ``a`` is stored in
4762:     memory. Therefore if ``a`` is C-ordered versus fortran-ordered, versus
4763:     defined as a slice or transpose, etc., the view may give different
4764:     results.
4765: 
4766: 
4767:     Examples
4768:     --------
4769:     >>> x = np.array([(1, 2)], dtype=[('a', np.int8), ('b', np.int8)])
4770: 
4771:     Viewing array data using a different type and dtype:
4772: 
4773:     >>> y = x.view(dtype=np.int16, type=np.matrix)
4774:     >>> y
4775:     matrix([[513]], dtype=int16)
4776:     >>> print(type(y))
4777:     <class 'numpy.matrixlib.defmatrix.matrix'>
4778: 
4779:     Creating a view on a structured array so it can be used in calculations
4780: 
4781:     >>> x = np.array([(1, 2),(3,4)], dtype=[('a', np.int8), ('b', np.int8)])
4782:     >>> xv = x.view(dtype=np.int8).reshape(-1,2)
4783:     >>> xv
4784:     array([[1, 2],
4785:            [3, 4]], dtype=int8)
4786:     >>> xv.mean(0)
4787:     array([ 2.,  3.])
4788: 
4789:     Making changes to the view changes the underlying array
4790: 
4791:     >>> xv[0,1] = 20
4792:     >>> print(x)
4793:     [(1, 20) (3, 4)]
4794: 
4795:     Using a view to convert an array to a recarray:
4796: 
4797:     >>> z = x.view(np.recarray)
4798:     >>> z.a
4799:     array([1], dtype=int8)
4800: 
4801:     Views share data:
4802: 
4803:     >>> x[0] = (9, 10)
4804:     >>> z[0]
4805:     (9, 10)
4806: 
4807:     Views that change the dtype size (bytes per entry) should normally be
4808:     avoided on arrays defined by slices, transposes, fortran-ordering, etc.:
4809: 
4810:     >>> x = np.array([[1,2,3],[4,5,6]], dtype=np.int16)
4811:     >>> y = x[:, 0:2]
4812:     >>> y
4813:     array([[1, 2],
4814:            [4, 5]], dtype=int16)
4815:     >>> y.view(dtype=[('width', np.int16), ('length', np.int16)])
4816:     Traceback (most recent call last):
4817:       File "<stdin>", line 1, in <module>
4818:     ValueError: new type not compatible with array.
4819:     >>> z = y.copy()
4820:     >>> z.view(dtype=[('width', np.int16), ('length', np.int16)])
4821:     array([[(1, 2)],
4822:            [(4, 5)]], dtype=[('width', '<i2'), ('length', '<i2')])
4823:     '''))
4824: 
4825: 
4826: ##############################################################################
4827: #
4828: # umath functions
4829: #
4830: ##############################################################################
4831: 
4832: add_newdoc('numpy.core.umath', 'frompyfunc',
4833:     '''
4834:     frompyfunc(func, nin, nout)
4835: 
4836:     Takes an arbitrary Python function and returns a Numpy ufunc.
4837: 
4838:     Can be used, for example, to add broadcasting to a built-in Python
4839:     function (see Examples section).
4840: 
4841:     Parameters
4842:     ----------
4843:     func : Python function object
4844:         An arbitrary Python function.
4845:     nin : int
4846:         The number of input arguments.
4847:     nout : int
4848:         The number of objects returned by `func`.
4849: 
4850:     Returns
4851:     -------
4852:     out : ufunc
4853:         Returns a Numpy universal function (``ufunc``) object.
4854: 
4855:     Notes
4856:     -----
4857:     The returned ufunc always returns PyObject arrays.
4858: 
4859:     Examples
4860:     --------
4861:     Use frompyfunc to add broadcasting to the Python function ``oct``:
4862: 
4863:     >>> oct_array = np.frompyfunc(oct, 1, 1)
4864:     >>> oct_array(np.array((10, 30, 100)))
4865:     array([012, 036, 0144], dtype=object)
4866:     >>> np.array((oct(10), oct(30), oct(100))) # for comparison
4867:     array(['012', '036', '0144'],
4868:           dtype='|S4')
4869: 
4870:     ''')
4871: 
4872: add_newdoc('numpy.core.umath', 'geterrobj',
4873:     '''
4874:     geterrobj()
4875: 
4876:     Return the current object that defines floating-point error handling.
4877: 
4878:     The error object contains all information that defines the error handling
4879:     behavior in Numpy. `geterrobj` is used internally by the other
4880:     functions that get and set error handling behavior (`geterr`, `seterr`,
4881:     `geterrcall`, `seterrcall`).
4882: 
4883:     Returns
4884:     -------
4885:     errobj : list
4886:         The error object, a list containing three elements:
4887:         [internal numpy buffer size, error mask, error callback function].
4888: 
4889:         The error mask is a single integer that holds the treatment information
4890:         on all four floating point errors. The information for each error type
4891:         is contained in three bits of the integer. If we print it in base 8, we
4892:         can see what treatment is set for "invalid", "under", "over", and
4893:         "divide" (in that order). The printed string can be interpreted with
4894: 
4895:         * 0 : 'ignore'
4896:         * 1 : 'warn'
4897:         * 2 : 'raise'
4898:         * 3 : 'call'
4899:         * 4 : 'print'
4900:         * 5 : 'log'
4901: 
4902:     See Also
4903:     --------
4904:     seterrobj, seterr, geterr, seterrcall, geterrcall
4905:     getbufsize, setbufsize
4906: 
4907:     Notes
4908:     -----
4909:     For complete documentation of the types of floating-point exceptions and
4910:     treatment options, see `seterr`.
4911: 
4912:     Examples
4913:     --------
4914:     >>> np.geterrobj()  # first get the defaults
4915:     [10000, 0, None]
4916: 
4917:     >>> def err_handler(type, flag):
4918:     ...     print("Floating point error (%s), with flag %s" % (type, flag))
4919:     ...
4920:     >>> old_bufsize = np.setbufsize(20000)
4921:     >>> old_err = np.seterr(divide='raise')
4922:     >>> old_handler = np.seterrcall(err_handler)
4923:     >>> np.geterrobj()
4924:     [20000, 2, <function err_handler at 0x91dcaac>]
4925: 
4926:     >>> old_err = np.seterr(all='ignore')
4927:     >>> np.base_repr(np.geterrobj()[1], 8)
4928:     '0'
4929:     >>> old_err = np.seterr(divide='warn', over='log', under='call',
4930:                             invalid='print')
4931:     >>> np.base_repr(np.geterrobj()[1], 8)
4932:     '4351'
4933: 
4934:     ''')
4935: 
4936: add_newdoc('numpy.core.umath', 'seterrobj',
4937:     '''
4938:     seterrobj(errobj)
4939: 
4940:     Set the object that defines floating-point error handling.
4941: 
4942:     The error object contains all information that defines the error handling
4943:     behavior in Numpy. `seterrobj` is used internally by the other
4944:     functions that set error handling behavior (`seterr`, `seterrcall`).
4945: 
4946:     Parameters
4947:     ----------
4948:     errobj : list
4949:         The error object, a list containing three elements:
4950:         [internal numpy buffer size, error mask, error callback function].
4951: 
4952:         The error mask is a single integer that holds the treatment information
4953:         on all four floating point errors. The information for each error type
4954:         is contained in three bits of the integer. If we print it in base 8, we
4955:         can see what treatment is set for "invalid", "under", "over", and
4956:         "divide" (in that order). The printed string can be interpreted with
4957: 
4958:         * 0 : 'ignore'
4959:         * 1 : 'warn'
4960:         * 2 : 'raise'
4961:         * 3 : 'call'
4962:         * 4 : 'print'
4963:         * 5 : 'log'
4964: 
4965:     See Also
4966:     --------
4967:     geterrobj, seterr, geterr, seterrcall, geterrcall
4968:     getbufsize, setbufsize
4969: 
4970:     Notes
4971:     -----
4972:     For complete documentation of the types of floating-point exceptions and
4973:     treatment options, see `seterr`.
4974: 
4975:     Examples
4976:     --------
4977:     >>> old_errobj = np.geterrobj()  # first get the defaults
4978:     >>> old_errobj
4979:     [10000, 0, None]
4980: 
4981:     >>> def err_handler(type, flag):
4982:     ...     print("Floating point error (%s), with flag %s" % (type, flag))
4983:     ...
4984:     >>> new_errobj = [20000, 12, err_handler]
4985:     >>> np.seterrobj(new_errobj)
4986:     >>> np.base_repr(12, 8)  # int for divide=4 ('print') and over=1 ('warn')
4987:     '14'
4988:     >>> np.geterr()
4989:     {'over': 'warn', 'divide': 'print', 'invalid': 'ignore', 'under': 'ignore'}
4990:     >>> np.geterrcall() is err_handler
4991:     True
4992: 
4993:     ''')
4994: 
4995: 
4996: ##############################################################################
4997: #
4998: # compiled_base functions
4999: #
5000: ##############################################################################
5001: 
5002: add_newdoc('numpy.core.multiarray', 'digitize',
5003:     '''
5004:     digitize(x, bins, right=False)
5005: 
5006:     Return the indices of the bins to which each value in input array belongs.
5007: 
5008:     Each index ``i`` returned is such that ``bins[i-1] <= x < bins[i]`` if
5009:     `bins` is monotonically increasing, or ``bins[i-1] > x >= bins[i]`` if
5010:     `bins` is monotonically decreasing. If values in `x` are beyond the
5011:     bounds of `bins`, 0 or ``len(bins)`` is returned as appropriate. If right
5012:     is True, then the right bin is closed so that the index ``i`` is such
5013:     that ``bins[i-1] < x <= bins[i]`` or bins[i-1] >= x > bins[i]`` if `bins`
5014:     is monotonically increasing or decreasing, respectively.
5015: 
5016:     Parameters
5017:     ----------
5018:     x : array_like
5019:         Input array to be binned. Prior to Numpy 1.10.0, this array had to
5020:         be 1-dimensional, but can now have any shape.
5021:     bins : array_like
5022:         Array of bins. It has to be 1-dimensional and monotonic.
5023:     right : bool, optional
5024:         Indicating whether the intervals include the right or the left bin
5025:         edge. Default behavior is (right==False) indicating that the interval
5026:         does not include the right edge. The left bin end is open in this
5027:         case, i.e., bins[i-1] <= x < bins[i] is the default behavior for
5028:         monotonically increasing bins.
5029: 
5030:     Returns
5031:     -------
5032:     out : ndarray of ints
5033:         Output array of indices, of same shape as `x`.
5034: 
5035:     Raises
5036:     ------
5037:     ValueError
5038:         If `bins` is not monotonic.
5039:     TypeError
5040:         If the type of the input is complex.
5041: 
5042:     See Also
5043:     --------
5044:     bincount, histogram, unique
5045: 
5046:     Notes
5047:     -----
5048:     If values in `x` are such that they fall outside the bin range,
5049:     attempting to index `bins` with the indices that `digitize` returns
5050:     will result in an IndexError.
5051: 
5052:     .. versionadded:: 1.10.0
5053: 
5054:     `np.digitize` is  implemented in terms of `np.searchsorted`. This means
5055:     that a binary search is used to bin the values, which scales much better
5056:     for larger number of bins than the previous linear search. It also removes
5057:     the requirement for the input array to be 1-dimensional.
5058: 
5059:     Examples
5060:     --------
5061:     >>> x = np.array([0.2, 6.4, 3.0, 1.6])
5062:     >>> bins = np.array([0.0, 1.0, 2.5, 4.0, 10.0])
5063:     >>> inds = np.digitize(x, bins)
5064:     >>> inds
5065:     array([1, 4, 3, 2])
5066:     >>> for n in range(x.size):
5067:     ...   print(bins[inds[n]-1], "<=", x[n], "<", bins[inds[n]])
5068:     ...
5069:     0.0 <= 0.2 < 1.0
5070:     4.0 <= 6.4 < 10.0
5071:     2.5 <= 3.0 < 4.0
5072:     1.0 <= 1.6 < 2.5
5073: 
5074:     >>> x = np.array([1.2, 10.0, 12.4, 15.5, 20.])
5075:     >>> bins = np.array([0, 5, 10, 15, 20])
5076:     >>> np.digitize(x,bins,right=True)
5077:     array([1, 2, 3, 4, 4])
5078:     >>> np.digitize(x,bins,right=False)
5079:     array([1, 3, 3, 4, 5])
5080:     ''')
5081: 
5082: add_newdoc('numpy.core.multiarray', 'bincount',
5083:     '''
5084:     bincount(x, weights=None, minlength=None)
5085: 
5086:     Count number of occurrences of each value in array of non-negative ints.
5087: 
5088:     The number of bins (of size 1) is one larger than the largest value in
5089:     `x`. If `minlength` is specified, there will be at least this number
5090:     of bins in the output array (though it will be longer if necessary,
5091:     depending on the contents of `x`).
5092:     Each bin gives the number of occurrences of its index value in `x`.
5093:     If `weights` is specified the input array is weighted by it, i.e. if a
5094:     value ``n`` is found at position ``i``, ``out[n] += weight[i]`` instead
5095:     of ``out[n] += 1``.
5096: 
5097:     Parameters
5098:     ----------
5099:     x : array_like, 1 dimension, nonnegative ints
5100:         Input array.
5101:     weights : array_like, optional
5102:         Weights, array of the same shape as `x`.
5103:     minlength : int, optional
5104:         A minimum number of bins for the output array.
5105: 
5106:         .. versionadded:: 1.6.0
5107: 
5108:     Returns
5109:     -------
5110:     out : ndarray of ints
5111:         The result of binning the input array.
5112:         The length of `out` is equal to ``np.amax(x)+1``.
5113: 
5114:     Raises
5115:     ------
5116:     ValueError
5117:         If the input is not 1-dimensional, or contains elements with negative
5118:         values, or if `minlength` is non-positive.
5119:     TypeError
5120:         If the type of the input is float or complex.
5121: 
5122:     See Also
5123:     --------
5124:     histogram, digitize, unique
5125: 
5126:     Examples
5127:     --------
5128:     >>> np.bincount(np.arange(5))
5129:     array([1, 1, 1, 1, 1])
5130:     >>> np.bincount(np.array([0, 1, 1, 3, 2, 1, 7]))
5131:     array([1, 3, 1, 1, 0, 0, 0, 1])
5132: 
5133:     >>> x = np.array([0, 1, 1, 3, 2, 1, 7, 23])
5134:     >>> np.bincount(x).size == np.amax(x)+1
5135:     True
5136: 
5137:     The input array needs to be of integer dtype, otherwise a
5138:     TypeError is raised:
5139: 
5140:     >>> np.bincount(np.arange(5, dtype=np.float))
5141:     Traceback (most recent call last):
5142:       File "<stdin>", line 1, in <module>
5143:     TypeError: array cannot be safely cast to required type
5144: 
5145:     A possible use of ``bincount`` is to perform sums over
5146:     variable-size chunks of an array, using the ``weights`` keyword.
5147: 
5148:     >>> w = np.array([0.3, 0.5, 0.2, 0.7, 1., -0.6]) # weights
5149:     >>> x = np.array([0, 1, 1, 2, 2, 2])
5150:     >>> np.bincount(x,  weights=w)
5151:     array([ 0.3,  0.7,  1.1])
5152: 
5153:     ''')
5154: 
5155: add_newdoc('numpy.core.multiarray', 'ravel_multi_index',
5156:     '''
5157:     ravel_multi_index(multi_index, dims, mode='raise', order='C')
5158: 
5159:     Converts a tuple of index arrays into an array of flat
5160:     indices, applying boundary modes to the multi-index.
5161: 
5162:     Parameters
5163:     ----------
5164:     multi_index : tuple of array_like
5165:         A tuple of integer arrays, one array for each dimension.
5166:     dims : tuple of ints
5167:         The shape of array into which the indices from ``multi_index`` apply.
5168:     mode : {'raise', 'wrap', 'clip'}, optional
5169:         Specifies how out-of-bounds indices are handled.  Can specify
5170:         either one mode or a tuple of modes, one mode per index.
5171: 
5172:         * 'raise' -- raise an error (default)
5173:         * 'wrap' -- wrap around
5174:         * 'clip' -- clip to the range
5175: 
5176:         In 'clip' mode, a negative index which would normally
5177:         wrap will clip to 0 instead.
5178:     order : {'C', 'F'}, optional
5179:         Determines whether the multi-index should be viewed as
5180:         indexing in row-major (C-style) or column-major
5181:         (Fortran-style) order.
5182: 
5183:     Returns
5184:     -------
5185:     raveled_indices : ndarray
5186:         An array of indices into the flattened version of an array
5187:         of dimensions ``dims``.
5188: 
5189:     See Also
5190:     --------
5191:     unravel_index
5192: 
5193:     Notes
5194:     -----
5195:     .. versionadded:: 1.6.0
5196: 
5197:     Examples
5198:     --------
5199:     >>> arr = np.array([[3,6,6],[4,5,1]])
5200:     >>> np.ravel_multi_index(arr, (7,6))
5201:     array([22, 41, 37])
5202:     >>> np.ravel_multi_index(arr, (7,6), order='F')
5203:     array([31, 41, 13])
5204:     >>> np.ravel_multi_index(arr, (4,6), mode='clip')
5205:     array([22, 23, 19])
5206:     >>> np.ravel_multi_index(arr, (4,4), mode=('clip','wrap'))
5207:     array([12, 13, 13])
5208: 
5209:     >>> np.ravel_multi_index((3,1,4,1), (6,7,8,9))
5210:     1621
5211:     ''')
5212: 
5213: add_newdoc('numpy.core.multiarray', 'unravel_index',
5214:     '''
5215:     unravel_index(indices, dims, order='C')
5216: 
5217:     Converts a flat index or array of flat indices into a tuple
5218:     of coordinate arrays.
5219: 
5220:     Parameters
5221:     ----------
5222:     indices : array_like
5223:         An integer array whose elements are indices into the flattened
5224:         version of an array of dimensions ``dims``. Before version 1.6.0,
5225:         this function accepted just one index value.
5226:     dims : tuple of ints
5227:         The shape of the array to use for unraveling ``indices``.
5228:     order : {'C', 'F'}, optional
5229:         Determines whether the indices should be viewed as indexing in
5230:         row-major (C-style) or column-major (Fortran-style) order.
5231: 
5232:         .. versionadded:: 1.6.0
5233: 
5234:     Returns
5235:     -------
5236:     unraveled_coords : tuple of ndarray
5237:         Each array in the tuple has the same shape as the ``indices``
5238:         array.
5239: 
5240:     See Also
5241:     --------
5242:     ravel_multi_index
5243: 
5244:     Examples
5245:     --------
5246:     >>> np.unravel_index([22, 41, 37], (7,6))
5247:     (array([3, 6, 6]), array([4, 5, 1]))
5248:     >>> np.unravel_index([31, 41, 13], (7,6), order='F')
5249:     (array([3, 6, 6]), array([4, 5, 1]))
5250: 
5251:     >>> np.unravel_index(1621, (6,7,8,9))
5252:     (3, 1, 4, 1)
5253: 
5254:     ''')
5255: 
5256: add_newdoc('numpy.core.multiarray', 'add_docstring',
5257:     '''
5258:     add_docstring(obj, docstring)
5259: 
5260:     Add a docstring to a built-in obj if possible.
5261:     If the obj already has a docstring raise a RuntimeError
5262:     If this routine does not know how to add a docstring to the object
5263:     raise a TypeError
5264:     ''')
5265: 
5266: add_newdoc('numpy.core.umath', '_add_newdoc_ufunc',
5267:     '''
5268:     add_ufunc_docstring(ufunc, new_docstring)
5269: 
5270:     Replace the docstring for a ufunc with new_docstring.
5271:     This method will only work if the current docstring for
5272:     the ufunc is NULL. (At the C level, i.e. when ufunc->doc is NULL.)
5273: 
5274:     Parameters
5275:     ----------
5276:     ufunc : numpy.ufunc
5277:         A ufunc whose current doc is NULL.
5278:     new_docstring : string
5279:         The new docstring for the ufunc.
5280: 
5281:     Notes
5282:     -----
5283:     This method allocates memory for new_docstring on
5284:     the heap. Technically this creates a mempory leak, since this
5285:     memory will not be reclaimed until the end of the program
5286:     even if the ufunc itself is removed. However this will only
5287:     be a problem if the user is repeatedly creating ufuncs with
5288:     no documentation, adding documentation via add_newdoc_ufunc,
5289:     and then throwing away the ufunc.
5290:     ''')
5291: 
5292: add_newdoc('numpy.core.multiarray', 'packbits',
5293:     '''
5294:     packbits(myarray, axis=None)
5295: 
5296:     Packs the elements of a binary-valued array into bits in a uint8 array.
5297: 
5298:     The result is padded to full bytes by inserting zero bits at the end.
5299: 
5300:     Parameters
5301:     ----------
5302:     myarray : array_like
5303:         An integer type array whose elements should be packed to bits.
5304:     axis : int, optional
5305:         The dimension over which bit-packing is done.
5306:         ``None`` implies packing the flattened array.
5307: 
5308:     Returns
5309:     -------
5310:     packed : ndarray
5311:         Array of type uint8 whose elements represent bits corresponding to the
5312:         logical (0 or nonzero) value of the input elements. The shape of
5313:         `packed` has the same number of dimensions as the input (unless `axis`
5314:         is None, in which case the output is 1-D).
5315: 
5316:     See Also
5317:     --------
5318:     unpackbits: Unpacks elements of a uint8 array into a binary-valued output
5319:                 array.
5320: 
5321:     Examples
5322:     --------
5323:     >>> a = np.array([[[1,0,1],
5324:     ...                [0,1,0]],
5325:     ...               [[1,1,0],
5326:     ...                [0,0,1]]])
5327:     >>> b = np.packbits(a, axis=-1)
5328:     >>> b
5329:     array([[[160],[64]],[[192],[32]]], dtype=uint8)
5330: 
5331:     Note that in binary 160 = 1010 0000, 64 = 0100 0000, 192 = 1100 0000,
5332:     and 32 = 0010 0000.
5333: 
5334:     ''')
5335: 
5336: add_newdoc('numpy.core.multiarray', 'unpackbits',
5337:     '''
5338:     unpackbits(myarray, axis=None)
5339: 
5340:     Unpacks elements of a uint8 array into a binary-valued output array.
5341: 
5342:     Each element of `myarray` represents a bit-field that should be unpacked
5343:     into a binary-valued output array. The shape of the output array is either
5344:     1-D (if `axis` is None) or the same shape as the input array with unpacking
5345:     done along the axis specified.
5346: 
5347:     Parameters
5348:     ----------
5349:     myarray : ndarray, uint8 type
5350:        Input array.
5351:     axis : int, optional
5352:        Unpacks along this axis.
5353: 
5354:     Returns
5355:     -------
5356:     unpacked : ndarray, uint8 type
5357:        The elements are binary-valued (0 or 1).
5358: 
5359:     See Also
5360:     --------
5361:     packbits : Packs the elements of a binary-valued array into bits in a uint8
5362:                array.
5363: 
5364:     Examples
5365:     --------
5366:     >>> a = np.array([[2], [7], [23]], dtype=np.uint8)
5367:     >>> a
5368:     array([[ 2],
5369:            [ 7],
5370:            [23]], dtype=uint8)
5371:     >>> b = np.unpackbits(a, axis=1)
5372:     >>> b
5373:     array([[0, 0, 0, 0, 0, 0, 1, 0],
5374:            [0, 0, 0, 0, 0, 1, 1, 1],
5375:            [0, 0, 0, 1, 0, 1, 1, 1]], dtype=uint8)
5376: 
5377:     ''')
5378: 
5379: 
5380: ##############################################################################
5381: #
5382: # Documentation for ufunc attributes and methods
5383: #
5384: ##############################################################################
5385: 
5386: 
5387: ##############################################################################
5388: #
5389: # ufunc object
5390: #
5391: ##############################################################################
5392: 
5393: add_newdoc('numpy.core', 'ufunc',
5394:     '''
5395:     Functions that operate element by element on whole arrays.
5396: 
5397:     To see the documentation for a specific ufunc, use np.info().  For
5398:     example, np.info(np.sin).  Because ufuncs are written in C
5399:     (for speed) and linked into Python with NumPy's ufunc facility,
5400:     Python's help() function finds this page whenever help() is called
5401:     on a ufunc.
5402: 
5403:     A detailed explanation of ufuncs can be found in the "ufuncs.rst"
5404:     file in the NumPy reference guide.
5405: 
5406:     Unary ufuncs:
5407:     =============
5408: 
5409:     op(X, out=None)
5410:     Apply op to X elementwise
5411: 
5412:     Parameters
5413:     ----------
5414:     X : array_like
5415:         Input array.
5416:     out : array_like
5417:         An array to store the output. Must be the same shape as `X`.
5418: 
5419:     Returns
5420:     -------
5421:     r : array_like
5422:         `r` will have the same shape as `X`; if out is provided, `r`
5423:         will be equal to out.
5424: 
5425:     Binary ufuncs:
5426:     ==============
5427: 
5428:     op(X, Y, out=None)
5429:     Apply `op` to `X` and `Y` elementwise. May "broadcast" to make
5430:     the shapes of `X` and `Y` congruent.
5431: 
5432:     The broadcasting rules are:
5433: 
5434:     * Dimensions of length 1 may be prepended to either array.
5435:     * Arrays may be repeated along dimensions of length 1.
5436: 
5437:     Parameters
5438:     ----------
5439:     X : array_like
5440:         First input array.
5441:     Y : array_like
5442:         Second input array.
5443:     out : array_like
5444:         An array to store the output. Must be the same shape as the
5445:         output would have.
5446: 
5447:     Returns
5448:     -------
5449:     r : array_like
5450:         The return value; if out is provided, `r` will be equal to out.
5451: 
5452:     ''')
5453: 
5454: 
5455: ##############################################################################
5456: #
5457: # ufunc attributes
5458: #
5459: ##############################################################################
5460: 
5461: add_newdoc('numpy.core', 'ufunc', ('identity',
5462:     '''
5463:     The identity value.
5464: 
5465:     Data attribute containing the identity element for the ufunc, if it has one.
5466:     If it does not, the attribute value is None.
5467: 
5468:     Examples
5469:     --------
5470:     >>> np.add.identity
5471:     0
5472:     >>> np.multiply.identity
5473:     1
5474:     >>> np.power.identity
5475:     1
5476:     >>> print(np.exp.identity)
5477:     None
5478:     '''))
5479: 
5480: add_newdoc('numpy.core', 'ufunc', ('nargs',
5481:     '''
5482:     The number of arguments.
5483: 
5484:     Data attribute containing the number of arguments the ufunc takes, including
5485:     optional ones.
5486: 
5487:     Notes
5488:     -----
5489:     Typically this value will be one more than what you might expect because all
5490:     ufuncs take  the optional "out" argument.
5491: 
5492:     Examples
5493:     --------
5494:     >>> np.add.nargs
5495:     3
5496:     >>> np.multiply.nargs
5497:     3
5498:     >>> np.power.nargs
5499:     3
5500:     >>> np.exp.nargs
5501:     2
5502:     '''))
5503: 
5504: add_newdoc('numpy.core', 'ufunc', ('nin',
5505:     '''
5506:     The number of inputs.
5507: 
5508:     Data attribute containing the number of arguments the ufunc treats as input.
5509: 
5510:     Examples
5511:     --------
5512:     >>> np.add.nin
5513:     2
5514:     >>> np.multiply.nin
5515:     2
5516:     >>> np.power.nin
5517:     2
5518:     >>> np.exp.nin
5519:     1
5520:     '''))
5521: 
5522: add_newdoc('numpy.core', 'ufunc', ('nout',
5523:     '''
5524:     The number of outputs.
5525: 
5526:     Data attribute containing the number of arguments the ufunc treats as output.
5527: 
5528:     Notes
5529:     -----
5530:     Since all ufuncs can take output arguments, this will always be (at least) 1.
5531: 
5532:     Examples
5533:     --------
5534:     >>> np.add.nout
5535:     1
5536:     >>> np.multiply.nout
5537:     1
5538:     >>> np.power.nout
5539:     1
5540:     >>> np.exp.nout
5541:     1
5542: 
5543:     '''))
5544: 
5545: add_newdoc('numpy.core', 'ufunc', ('ntypes',
5546:     '''
5547:     The number of types.
5548: 
5549:     The number of numerical NumPy types - of which there are 18 total - on which
5550:     the ufunc can operate.
5551: 
5552:     See Also
5553:     --------
5554:     numpy.ufunc.types
5555: 
5556:     Examples
5557:     --------
5558:     >>> np.add.ntypes
5559:     18
5560:     >>> np.multiply.ntypes
5561:     18
5562:     >>> np.power.ntypes
5563:     17
5564:     >>> np.exp.ntypes
5565:     7
5566:     >>> np.remainder.ntypes
5567:     14
5568: 
5569:     '''))
5570: 
5571: add_newdoc('numpy.core', 'ufunc', ('types',
5572:     '''
5573:     Returns a list with types grouped input->output.
5574: 
5575:     Data attribute listing the data-type "Domain-Range" groupings the ufunc can
5576:     deliver. The data-types are given using the character codes.
5577: 
5578:     See Also
5579:     --------
5580:     numpy.ufunc.ntypes
5581: 
5582:     Examples
5583:     --------
5584:     >>> np.add.types
5585:     ['??->?', 'bb->b', 'BB->B', 'hh->h', 'HH->H', 'ii->i', 'II->I', 'll->l',
5586:     'LL->L', 'qq->q', 'QQ->Q', 'ff->f', 'dd->d', 'gg->g', 'FF->F', 'DD->D',
5587:     'GG->G', 'OO->O']
5588: 
5589:     >>> np.multiply.types
5590:     ['??->?', 'bb->b', 'BB->B', 'hh->h', 'HH->H', 'ii->i', 'II->I', 'll->l',
5591:     'LL->L', 'qq->q', 'QQ->Q', 'ff->f', 'dd->d', 'gg->g', 'FF->F', 'DD->D',
5592:     'GG->G', 'OO->O']
5593: 
5594:     >>> np.power.types
5595:     ['bb->b', 'BB->B', 'hh->h', 'HH->H', 'ii->i', 'II->I', 'll->l', 'LL->L',
5596:     'qq->q', 'QQ->Q', 'ff->f', 'dd->d', 'gg->g', 'FF->F', 'DD->D', 'GG->G',
5597:     'OO->O']
5598: 
5599:     >>> np.exp.types
5600:     ['f->f', 'd->d', 'g->g', 'F->F', 'D->D', 'G->G', 'O->O']
5601: 
5602:     >>> np.remainder.types
5603:     ['bb->b', 'BB->B', 'hh->h', 'HH->H', 'ii->i', 'II->I', 'll->l', 'LL->L',
5604:     'qq->q', 'QQ->Q', 'ff->f', 'dd->d', 'gg->g', 'OO->O']
5605: 
5606:     '''))
5607: 
5608: 
5609: ##############################################################################
5610: #
5611: # ufunc methods
5612: #
5613: ##############################################################################
5614: 
5615: add_newdoc('numpy.core', 'ufunc', ('reduce',
5616:     '''
5617:     reduce(a, axis=0, dtype=None, out=None, keepdims=False)
5618: 
5619:     Reduces `a`'s dimension by one, by applying ufunc along one axis.
5620: 
5621:     Let :math:`a.shape = (N_0, ..., N_i, ..., N_{M-1})`.  Then
5622:     :math:`ufunc.reduce(a, axis=i)[k_0, ..,k_{i-1}, k_{i+1}, .., k_{M-1}]` =
5623:     the result of iterating `j` over :math:`range(N_i)`, cumulatively applying
5624:     ufunc to each :math:`a[k_0, ..,k_{i-1}, j, k_{i+1}, .., k_{M-1}]`.
5625:     For a one-dimensional array, reduce produces results equivalent to:
5626:     ::
5627: 
5628:      r = op.identity # op = ufunc
5629:      for i in range(len(A)):
5630:        r = op(r, A[i])
5631:      return r
5632: 
5633:     For example, add.reduce() is equivalent to sum().
5634: 
5635:     Parameters
5636:     ----------
5637:     a : array_like
5638:         The array to act on.
5639:     axis : None or int or tuple of ints, optional
5640:         Axis or axes along which a reduction is performed.
5641:         The default (`axis` = 0) is perform a reduction over the first
5642:         dimension of the input array. `axis` may be negative, in
5643:         which case it counts from the last to the first axis.
5644: 
5645:         .. versionadded:: 1.7.0
5646: 
5647:         If this is `None`, a reduction is performed over all the axes.
5648:         If this is a tuple of ints, a reduction is performed on multiple
5649:         axes, instead of a single axis or all the axes as before.
5650: 
5651:         For operations which are either not commutative or not associative,
5652:         doing a reduction over multiple axes is not well-defined. The
5653:         ufuncs do not currently raise an exception in this case, but will
5654:         likely do so in the future.
5655:     dtype : data-type code, optional
5656:         The type used to represent the intermediate results. Defaults
5657:         to the data-type of the output array if this is provided, or
5658:         the data-type of the input array if no output array is provided.
5659:     out : ndarray, optional
5660:         A location into which the result is stored. If not provided, a
5661:         freshly-allocated array is returned.
5662:     keepdims : bool, optional
5663:         If this is set to True, the axes which are reduced are left
5664:         in the result as dimensions with size one. With this option,
5665:         the result will broadcast correctly against the original `arr`.
5666: 
5667:         .. versionadded:: 1.7.0
5668: 
5669:     Returns
5670:     -------
5671:     r : ndarray
5672:         The reduced array. If `out` was supplied, `r` is a reference to it.
5673: 
5674:     Examples
5675:     --------
5676:     >>> np.multiply.reduce([2,3,5])
5677:     30
5678: 
5679:     A multi-dimensional array example:
5680: 
5681:     >>> X = np.arange(8).reshape((2,2,2))
5682:     >>> X
5683:     array([[[0, 1],
5684:             [2, 3]],
5685:            [[4, 5],
5686:             [6, 7]]])
5687:     >>> np.add.reduce(X, 0)
5688:     array([[ 4,  6],
5689:            [ 8, 10]])
5690:     >>> np.add.reduce(X) # confirm: default axis value is 0
5691:     array([[ 4,  6],
5692:            [ 8, 10]])
5693:     >>> np.add.reduce(X, 1)
5694:     array([[ 2,  4],
5695:            [10, 12]])
5696:     >>> np.add.reduce(X, 2)
5697:     array([[ 1,  5],
5698:            [ 9, 13]])
5699: 
5700:     '''))
5701: 
5702: add_newdoc('numpy.core', 'ufunc', ('accumulate',
5703:     '''
5704:     accumulate(array, axis=0, dtype=None, out=None)
5705: 
5706:     Accumulate the result of applying the operator to all elements.
5707: 
5708:     For a one-dimensional array, accumulate produces results equivalent to::
5709: 
5710:       r = np.empty(len(A))
5711:       t = op.identity        # op = the ufunc being applied to A's  elements
5712:       for i in range(len(A)):
5713:           t = op(t, A[i])
5714:           r[i] = t
5715:       return r
5716: 
5717:     For example, add.accumulate() is equivalent to np.cumsum().
5718: 
5719:     For a multi-dimensional array, accumulate is applied along only one
5720:     axis (axis zero by default; see Examples below) so repeated use is
5721:     necessary if one wants to accumulate over multiple axes.
5722: 
5723:     Parameters
5724:     ----------
5725:     array : array_like
5726:         The array to act on.
5727:     axis : int, optional
5728:         The axis along which to apply the accumulation; default is zero.
5729:     dtype : data-type code, optional
5730:         The data-type used to represent the intermediate results. Defaults
5731:         to the data-type of the output array if such is provided, or the
5732:         the data-type of the input array if no output array is provided.
5733:     out : ndarray, optional
5734:         A location into which the result is stored. If not provided a
5735:         freshly-allocated array is returned.
5736: 
5737:     Returns
5738:     -------
5739:     r : ndarray
5740:         The accumulated values. If `out` was supplied, `r` is a reference to
5741:         `out`.
5742: 
5743:     Examples
5744:     --------
5745:     1-D array examples:
5746: 
5747:     >>> np.add.accumulate([2, 3, 5])
5748:     array([ 2,  5, 10])
5749:     >>> np.multiply.accumulate([2, 3, 5])
5750:     array([ 2,  6, 30])
5751: 
5752:     2-D array examples:
5753: 
5754:     >>> I = np.eye(2)
5755:     >>> I
5756:     array([[ 1.,  0.],
5757:            [ 0.,  1.]])
5758: 
5759:     Accumulate along axis 0 (rows), down columns:
5760: 
5761:     >>> np.add.accumulate(I, 0)
5762:     array([[ 1.,  0.],
5763:            [ 1.,  1.]])
5764:     >>> np.add.accumulate(I) # no axis specified = axis zero
5765:     array([[ 1.,  0.],
5766:            [ 1.,  1.]])
5767: 
5768:     Accumulate along axis 1 (columns), through rows:
5769: 
5770:     >>> np.add.accumulate(I, 1)
5771:     array([[ 1.,  1.],
5772:            [ 0.,  1.]])
5773: 
5774:     '''))
5775: 
5776: add_newdoc('numpy.core', 'ufunc', ('reduceat',
5777:     '''
5778:     reduceat(a, indices, axis=0, dtype=None, out=None)
5779: 
5780:     Performs a (local) reduce with specified slices over a single axis.
5781: 
5782:     For i in ``range(len(indices))``, `reduceat` computes
5783:     ``ufunc.reduce(a[indices[i]:indices[i+1]])``, which becomes the i-th
5784:     generalized "row" parallel to `axis` in the final result (i.e., in a
5785:     2-D array, for example, if `axis = 0`, it becomes the i-th row, but if
5786:     `axis = 1`, it becomes the i-th column).  There are three exceptions to this:
5787: 
5788:     * when ``i = len(indices) - 1`` (so for the last index),
5789:       ``indices[i+1] = a.shape[axis]``.
5790:     * if ``indices[i] >= indices[i + 1]``, the i-th generalized "row" is
5791:       simply ``a[indices[i]]``.
5792:     * if ``indices[i] >= len(a)`` or ``indices[i] < 0``, an error is raised.
5793: 
5794:     The shape of the output depends on the size of `indices`, and may be
5795:     larger than `a` (this happens if ``len(indices) > a.shape[axis]``).
5796: 
5797:     Parameters
5798:     ----------
5799:     a : array_like
5800:         The array to act on.
5801:     indices : array_like
5802:         Paired indices, comma separated (not colon), specifying slices to
5803:         reduce.
5804:     axis : int, optional
5805:         The axis along which to apply the reduceat.
5806:     dtype : data-type code, optional
5807:         The type used to represent the intermediate results. Defaults
5808:         to the data type of the output array if this is provided, or
5809:         the data type of the input array if no output array is provided.
5810:     out : ndarray, optional
5811:         A location into which the result is stored. If not provided a
5812:         freshly-allocated array is returned.
5813: 
5814:     Returns
5815:     -------
5816:     r : ndarray
5817:         The reduced values. If `out` was supplied, `r` is a reference to
5818:         `out`.
5819: 
5820:     Notes
5821:     -----
5822:     A descriptive example:
5823: 
5824:     If `a` is 1-D, the function `ufunc.accumulate(a)` is the same as
5825:     ``ufunc.reduceat(a, indices)[::2]`` where `indices` is
5826:     ``range(len(array) - 1)`` with a zero placed
5827:     in every other element:
5828:     ``indices = zeros(2 * len(a) - 1)``, ``indices[1::2] = range(1, len(a))``.
5829: 
5830:     Don't be fooled by this attribute's name: `reduceat(a)` is not
5831:     necessarily smaller than `a`.
5832: 
5833:     Examples
5834:     --------
5835:     To take the running sum of four successive values:
5836: 
5837:     >>> np.add.reduceat(np.arange(8),[0,4, 1,5, 2,6, 3,7])[::2]
5838:     array([ 6, 10, 14, 18])
5839: 
5840:     A 2-D example:
5841: 
5842:     >>> x = np.linspace(0, 15, 16).reshape(4,4)
5843:     >>> x
5844:     array([[  0.,   1.,   2.,   3.],
5845:            [  4.,   5.,   6.,   7.],
5846:            [  8.,   9.,  10.,  11.],
5847:            [ 12.,  13.,  14.,  15.]])
5848: 
5849:     ::
5850: 
5851:      # reduce such that the result has the following five rows:
5852:      # [row1 + row2 + row3]
5853:      # [row4]
5854:      # [row2]
5855:      # [row3]
5856:      # [row1 + row2 + row3 + row4]
5857: 
5858:     >>> np.add.reduceat(x, [0, 3, 1, 2, 0])
5859:     array([[ 12.,  15.,  18.,  21.],
5860:            [ 12.,  13.,  14.,  15.],
5861:            [  4.,   5.,   6.,   7.],
5862:            [  8.,   9.,  10.,  11.],
5863:            [ 24.,  28.,  32.,  36.]])
5864: 
5865:     ::
5866: 
5867:      # reduce such that result has the following two columns:
5868:      # [col1 * col2 * col3, col4]
5869: 
5870:     >>> np.multiply.reduceat(x, [0, 3], 1)
5871:     array([[    0.,     3.],
5872:            [  120.,     7.],
5873:            [  720.,    11.],
5874:            [ 2184.,    15.]])
5875: 
5876:     '''))
5877: 
5878: add_newdoc('numpy.core', 'ufunc', ('outer',
5879:     '''
5880:     outer(A, B)
5881: 
5882:     Apply the ufunc `op` to all pairs (a, b) with a in `A` and b in `B`.
5883: 
5884:     Let ``M = A.ndim``, ``N = B.ndim``. Then the result, `C`, of
5885:     ``op.outer(A, B)`` is an array of dimension M + N such that:
5886: 
5887:     .. math:: C[i_0, ..., i_{M-1}, j_0, ..., j_{N-1}] =
5888:        op(A[i_0, ..., i_{M-1}], B[j_0, ..., j_{N-1}])
5889: 
5890:     For `A` and `B` one-dimensional, this is equivalent to::
5891: 
5892:       r = empty(len(A),len(B))
5893:       for i in range(len(A)):
5894:           for j in range(len(B)):
5895:               r[i,j] = op(A[i], B[j]) # op = ufunc in question
5896: 
5897:     Parameters
5898:     ----------
5899:     A : array_like
5900:         First array
5901:     B : array_like
5902:         Second array
5903: 
5904:     Returns
5905:     -------
5906:     r : ndarray
5907:         Output array
5908: 
5909:     See Also
5910:     --------
5911:     numpy.outer
5912: 
5913:     Examples
5914:     --------
5915:     >>> np.multiply.outer([1, 2, 3], [4, 5, 6])
5916:     array([[ 4,  5,  6],
5917:            [ 8, 10, 12],
5918:            [12, 15, 18]])
5919: 
5920:     A multi-dimensional example:
5921: 
5922:     >>> A = np.array([[1, 2, 3], [4, 5, 6]])
5923:     >>> A.shape
5924:     (2, 3)
5925:     >>> B = np.array([[1, 2, 3, 4]])
5926:     >>> B.shape
5927:     (1, 4)
5928:     >>> C = np.multiply.outer(A, B)
5929:     >>> C.shape; C
5930:     (2, 3, 1, 4)
5931:     array([[[[ 1,  2,  3,  4]],
5932:             [[ 2,  4,  6,  8]],
5933:             [[ 3,  6,  9, 12]]],
5934:            [[[ 4,  8, 12, 16]],
5935:             [[ 5, 10, 15, 20]],
5936:             [[ 6, 12, 18, 24]]]])
5937: 
5938:     '''))
5939: 
5940: add_newdoc('numpy.core', 'ufunc', ('at',
5941:     '''
5942:     at(a, indices, b=None)
5943: 
5944:     Performs unbuffered in place operation on operand 'a' for elements
5945:     specified by 'indices'. For addition ufunc, this method is equivalent to
5946:     `a[indices] += b`, except that results are accumulated for elements that
5947:     are indexed more than once. For example, `a[[0,0]] += 1` will only
5948:     increment the first element once because of buffering, whereas
5949:     `add.at(a, [0,0], 1)` will increment the first element twice.
5950: 
5951:     .. versionadded:: 1.8.0
5952: 
5953:     Parameters
5954:     ----------
5955:     a : array_like
5956:         The array to perform in place operation on.
5957:     indices : array_like or tuple
5958:         Array like index object or slice object for indexing into first
5959:         operand. If first operand has multiple dimensions, indices can be a
5960:         tuple of array like index objects or slice objects.
5961:     b : array_like
5962:         Second operand for ufuncs requiring two operands. Operand must be
5963:         broadcastable over first operand after indexing or slicing.
5964: 
5965:     Examples
5966:     --------
5967:     Set items 0 and 1 to their negative values:
5968: 
5969:     >>> a = np.array([1, 2, 3, 4])
5970:     >>> np.negative.at(a, [0, 1])
5971:     >>> print(a)
5972:     array([-1, -2, 3, 4])
5973: 
5974:     ::
5975: 
5976:     Increment items 0 and 1, and increment item 2 twice:
5977: 
5978:     >>> a = np.array([1, 2, 3, 4])
5979:     >>> np.add.at(a, [0, 1, 2, 2], 1)
5980:     >>> print(a)
5981:     array([2, 3, 5, 4])
5982: 
5983:     ::
5984: 
5985:     Add items 0 and 1 in first array to second array,
5986:     and store results in first array:
5987: 
5988:     >>> a = np.array([1, 2, 3, 4])
5989:     >>> b = np.array([1, 2])
5990:     >>> np.add.at(a, [0, 1], b)
5991:     >>> print(a)
5992:     array([2, 4, 3, 4])
5993: 
5994:     '''))
5995: 
5996: ##############################################################################
5997: #
5998: # Documentation for dtype attributes and methods
5999: #
6000: ##############################################################################
6001: 
6002: ##############################################################################
6003: #
6004: # dtype object
6005: #
6006: ##############################################################################
6007: 
6008: add_newdoc('numpy.core.multiarray', 'dtype',
6009:     '''
6010:     dtype(obj, align=False, copy=False)
6011: 
6012:     Create a data type object.
6013: 
6014:     A numpy array is homogeneous, and contains elements described by a
6015:     dtype object. A dtype object can be constructed from different
6016:     combinations of fundamental numeric types.
6017: 
6018:     Parameters
6019:     ----------
6020:     obj
6021:         Object to be converted to a data type object.
6022:     align : bool, optional
6023:         Add padding to the fields to match what a C compiler would output
6024:         for a similar C-struct. Can be ``True`` only if `obj` is a dictionary
6025:         or a comma-separated string. If a struct dtype is being created,
6026:         this also sets a sticky alignment flag ``isalignedstruct``.
6027:     copy : bool, optional
6028:         Make a new copy of the data-type object. If ``False``, the result
6029:         may just be a reference to a built-in data-type object.
6030: 
6031:     See also
6032:     --------
6033:     result_type
6034: 
6035:     Examples
6036:     --------
6037:     Using array-scalar type:
6038: 
6039:     >>> np.dtype(np.int16)
6040:     dtype('int16')
6041: 
6042:     Structured type, one field name 'f1', containing int16:
6043: 
6044:     >>> np.dtype([('f1', np.int16)])
6045:     dtype([('f1', '<i2')])
6046: 
6047:     Structured type, one field named 'f1', in itself containing a structured
6048:     type with one field:
6049: 
6050:     >>> np.dtype([('f1', [('f1', np.int16)])])
6051:     dtype([('f1', [('f1', '<i2')])])
6052: 
6053:     Structured type, two fields: the first field contains an unsigned int, the
6054:     second an int32:
6055: 
6056:     >>> np.dtype([('f1', np.uint), ('f2', np.int32)])
6057:     dtype([('f1', '<u4'), ('f2', '<i4')])
6058: 
6059:     Using array-protocol type strings:
6060: 
6061:     >>> np.dtype([('a','f8'),('b','S10')])
6062:     dtype([('a', '<f8'), ('b', '|S10')])
6063: 
6064:     Using comma-separated field formats.  The shape is (2,3):
6065: 
6066:     >>> np.dtype("i4, (2,3)f8")
6067:     dtype([('f0', '<i4'), ('f1', '<f8', (2, 3))])
6068: 
6069:     Using tuples.  ``int`` is a fixed type, 3 the field's shape.  ``void``
6070:     is a flexible type, here of size 10:
6071: 
6072:     >>> np.dtype([('hello',(np.int,3)),('world',np.void,10)])
6073:     dtype([('hello', '<i4', 3), ('world', '|V10')])
6074: 
6075:     Subdivide ``int16`` into 2 ``int8``'s, called x and y.  0 and 1 are
6076:     the offsets in bytes:
6077: 
6078:     >>> np.dtype((np.int16, {'x':(np.int8,0), 'y':(np.int8,1)}))
6079:     dtype(('<i2', [('x', '|i1'), ('y', '|i1')]))
6080: 
6081:     Using dictionaries.  Two fields named 'gender' and 'age':
6082: 
6083:     >>> np.dtype({'names':['gender','age'], 'formats':['S1',np.uint8]})
6084:     dtype([('gender', '|S1'), ('age', '|u1')])
6085: 
6086:     Offsets in bytes, here 0 and 25:
6087: 
6088:     >>> np.dtype({'surname':('S25',0),'age':(np.uint8,25)})
6089:     dtype([('surname', '|S25'), ('age', '|u1')])
6090: 
6091:     ''')
6092: 
6093: ##############################################################################
6094: #
6095: # dtype attributes
6096: #
6097: ##############################################################################
6098: 
6099: add_newdoc('numpy.core.multiarray', 'dtype', ('alignment',
6100:     '''
6101:     The required alignment (bytes) of this data-type according to the compiler.
6102: 
6103:     More information is available in the C-API section of the manual.
6104: 
6105:     '''))
6106: 
6107: add_newdoc('numpy.core.multiarray', 'dtype', ('byteorder',
6108:     '''
6109:     A character indicating the byte-order of this data-type object.
6110: 
6111:     One of:
6112: 
6113:     ===  ==============
6114:     '='  native
6115:     '<'  little-endian
6116:     '>'  big-endian
6117:     '|'  not applicable
6118:     ===  ==============
6119: 
6120:     All built-in data-type objects have byteorder either '=' or '|'.
6121: 
6122:     Examples
6123:     --------
6124: 
6125:     >>> dt = np.dtype('i2')
6126:     >>> dt.byteorder
6127:     '='
6128:     >>> # endian is not relevant for 8 bit numbers
6129:     >>> np.dtype('i1').byteorder
6130:     '|'
6131:     >>> # or ASCII strings
6132:     >>> np.dtype('S2').byteorder
6133:     '|'
6134:     >>> # Even if specific code is given, and it is native
6135:     >>> # '=' is the byteorder
6136:     >>> import sys
6137:     >>> sys_is_le = sys.byteorder == 'little'
6138:     >>> native_code = sys_is_le and '<' or '>'
6139:     >>> swapped_code = sys_is_le and '>' or '<'
6140:     >>> dt = np.dtype(native_code + 'i2')
6141:     >>> dt.byteorder
6142:     '='
6143:     >>> # Swapped code shows up as itself
6144:     >>> dt = np.dtype(swapped_code + 'i2')
6145:     >>> dt.byteorder == swapped_code
6146:     True
6147: 
6148:     '''))
6149: 
6150: add_newdoc('numpy.core.multiarray', 'dtype', ('char',
6151:     '''A unique character code for each of the 21 different built-in types.'''))
6152: 
6153: add_newdoc('numpy.core.multiarray', 'dtype', ('descr',
6154:     '''
6155:     Array-interface compliant full description of the data-type.
6156: 
6157:     The format is that required by the 'descr' key in the
6158:     `__array_interface__` attribute.
6159: 
6160:     '''))
6161: 
6162: add_newdoc('numpy.core.multiarray', 'dtype', ('fields',
6163:     '''
6164:     Dictionary of named fields defined for this data type, or ``None``.
6165: 
6166:     The dictionary is indexed by keys that are the names of the fields.
6167:     Each entry in the dictionary is a tuple fully describing the field::
6168: 
6169:       (dtype, offset[, title])
6170: 
6171:     If present, the optional title can be any object (if it is a string
6172:     or unicode then it will also be a key in the fields dictionary,
6173:     otherwise it's meta-data). Notice also that the first two elements
6174:     of the tuple can be passed directly as arguments to the ``ndarray.getfield``
6175:     and ``ndarray.setfield`` methods.
6176: 
6177:     See Also
6178:     --------
6179:     ndarray.getfield, ndarray.setfield
6180: 
6181:     Examples
6182:     --------
6183:     >>> dt = np.dtype([('name', np.str_, 16), ('grades', np.float64, (2,))])
6184:     >>> print(dt.fields)
6185:     {'grades': (dtype(('float64',(2,))), 16), 'name': (dtype('|S16'), 0)}
6186: 
6187:     '''))
6188: 
6189: add_newdoc('numpy.core.multiarray', 'dtype', ('flags',
6190:     '''
6191:     Bit-flags describing how this data type is to be interpreted.
6192: 
6193:     Bit-masks are in `numpy.core.multiarray` as the constants
6194:     `ITEM_HASOBJECT`, `LIST_PICKLE`, `ITEM_IS_POINTER`, `NEEDS_INIT`,
6195:     `NEEDS_PYAPI`, `USE_GETITEM`, `USE_SETITEM`. A full explanation
6196:     of these flags is in C-API documentation; they are largely useful
6197:     for user-defined data-types.
6198: 
6199:     '''))
6200: 
6201: add_newdoc('numpy.core.multiarray', 'dtype', ('hasobject',
6202:     '''
6203:     Boolean indicating whether this dtype contains any reference-counted
6204:     objects in any fields or sub-dtypes.
6205: 
6206:     Recall that what is actually in the ndarray memory representing
6207:     the Python object is the memory address of that object (a pointer).
6208:     Special handling may be required, and this attribute is useful for
6209:     distinguishing data types that may contain arbitrary Python objects
6210:     and data-types that won't.
6211: 
6212:     '''))
6213: 
6214: add_newdoc('numpy.core.multiarray', 'dtype', ('isbuiltin',
6215:     '''
6216:     Integer indicating how this dtype relates to the built-in dtypes.
6217: 
6218:     Read-only.
6219: 
6220:     =  ========================================================================
6221:     0  if this is a structured array type, with fields
6222:     1  if this is a dtype compiled into numpy (such as ints, floats etc)
6223:     2  if the dtype is for a user-defined numpy type
6224:        A user-defined type uses the numpy C-API machinery to extend
6225:        numpy to handle a new array type. See
6226:        :ref:`user.user-defined-data-types` in the Numpy manual.
6227:     =  ========================================================================
6228: 
6229:     Examples
6230:     --------
6231:     >>> dt = np.dtype('i2')
6232:     >>> dt.isbuiltin
6233:     1
6234:     >>> dt = np.dtype('f8')
6235:     >>> dt.isbuiltin
6236:     1
6237:     >>> dt = np.dtype([('field1', 'f8')])
6238:     >>> dt.isbuiltin
6239:     0
6240: 
6241:     '''))
6242: 
6243: add_newdoc('numpy.core.multiarray', 'dtype', ('isnative',
6244:     '''
6245:     Boolean indicating whether the byte order of this dtype is native
6246:     to the platform.
6247: 
6248:     '''))
6249: 
6250: add_newdoc('numpy.core.multiarray', 'dtype', ('isalignedstruct',
6251:     '''
6252:     Boolean indicating whether the dtype is a struct which maintains
6253:     field alignment. This flag is sticky, so when combining multiple
6254:     structs together, it is preserved and produces new dtypes which
6255:     are also aligned.
6256:     '''))
6257: 
6258: add_newdoc('numpy.core.multiarray', 'dtype', ('itemsize',
6259:     '''
6260:     The element size of this data-type object.
6261: 
6262:     For 18 of the 21 types this number is fixed by the data-type.
6263:     For the flexible data-types, this number can be anything.
6264: 
6265:     '''))
6266: 
6267: add_newdoc('numpy.core.multiarray', 'dtype', ('kind',
6268:     '''
6269:     A character code (one of 'biufcmMOSUV') identifying the general kind of data.
6270: 
6271:     =  ======================
6272:     b  boolean
6273:     i  signed integer
6274:     u  unsigned integer
6275:     f  floating-point
6276:     c  complex floating-point
6277:     m  timedelta
6278:     M  datetime
6279:     O  object
6280:     S  (byte-)string
6281:     U  Unicode
6282:     V  void
6283:     =  ======================
6284: 
6285:     '''))
6286: 
6287: add_newdoc('numpy.core.multiarray', 'dtype', ('name',
6288:     '''
6289:     A bit-width name for this data-type.
6290: 
6291:     Un-sized flexible data-type objects do not have this attribute.
6292: 
6293:     '''))
6294: 
6295: add_newdoc('numpy.core.multiarray', 'dtype', ('names',
6296:     '''
6297:     Ordered list of field names, or ``None`` if there are no fields.
6298: 
6299:     The names are ordered according to increasing byte offset. This can be
6300:     used, for example, to walk through all of the named fields in offset order.
6301: 
6302:     Examples
6303:     --------
6304:     >>> dt = np.dtype([('name', np.str_, 16), ('grades', np.float64, (2,))])
6305:     >>> dt.names
6306:     ('name', 'grades')
6307: 
6308:     '''))
6309: 
6310: add_newdoc('numpy.core.multiarray', 'dtype', ('num',
6311:     '''
6312:     A unique number for each of the 21 different built-in types.
6313: 
6314:     These are roughly ordered from least-to-most precision.
6315: 
6316:     '''))
6317: 
6318: add_newdoc('numpy.core.multiarray', 'dtype', ('shape',
6319:     '''
6320:     Shape tuple of the sub-array if this data type describes a sub-array,
6321:     and ``()`` otherwise.
6322: 
6323:     '''))
6324: 
6325: add_newdoc('numpy.core.multiarray', 'dtype', ('str',
6326:     '''The array-protocol typestring of this data-type object.'''))
6327: 
6328: add_newdoc('numpy.core.multiarray', 'dtype', ('subdtype',
6329:     '''
6330:     Tuple ``(item_dtype, shape)`` if this `dtype` describes a sub-array, and
6331:     None otherwise.
6332: 
6333:     The *shape* is the fixed shape of the sub-array described by this
6334:     data type, and *item_dtype* the data type of the array.
6335: 
6336:     If a field whose dtype object has this attribute is retrieved,
6337:     then the extra dimensions implied by *shape* are tacked on to
6338:     the end of the retrieved array.
6339: 
6340:     '''))
6341: 
6342: add_newdoc('numpy.core.multiarray', 'dtype', ('type',
6343:     '''The type object used to instantiate a scalar of this data-type.'''))
6344: 
6345: ##############################################################################
6346: #
6347: # dtype methods
6348: #
6349: ##############################################################################
6350: 
6351: add_newdoc('numpy.core.multiarray', 'dtype', ('newbyteorder',
6352:     '''
6353:     newbyteorder(new_order='S')
6354: 
6355:     Return a new dtype with a different byte order.
6356: 
6357:     Changes are also made in all fields and sub-arrays of the data type.
6358: 
6359:     Parameters
6360:     ----------
6361:     new_order : string, optional
6362:         Byte order to force; a value from the byte order specifications
6363:         below.  The default value ('S') results in swapping the current
6364:         byte order.  `new_order` codes can be any of:
6365: 
6366:         * 'S' - swap dtype from current to opposite endian
6367:         * {'<', 'L'} - little endian
6368:         * {'>', 'B'} - big endian
6369:         * {'=', 'N'} - native order
6370:         * {'|', 'I'} - ignore (no change to byte order)
6371: 
6372:         The code does a case-insensitive check on the first letter of
6373:         `new_order` for these alternatives.  For example, any of '>'
6374:         or 'B' or 'b' or 'brian' are valid to specify big-endian.
6375: 
6376:     Returns
6377:     -------
6378:     new_dtype : dtype
6379:         New dtype object with the given change to the byte order.
6380: 
6381:     Notes
6382:     -----
6383:     Changes are also made in all fields and sub-arrays of the data type.
6384: 
6385:     Examples
6386:     --------
6387:     >>> import sys
6388:     >>> sys_is_le = sys.byteorder == 'little'
6389:     >>> native_code = sys_is_le and '<' or '>'
6390:     >>> swapped_code = sys_is_le and '>' or '<'
6391:     >>> native_dt = np.dtype(native_code+'i2')
6392:     >>> swapped_dt = np.dtype(swapped_code+'i2')
6393:     >>> native_dt.newbyteorder('S') == swapped_dt
6394:     True
6395:     >>> native_dt.newbyteorder() == swapped_dt
6396:     True
6397:     >>> native_dt == swapped_dt.newbyteorder('S')
6398:     True
6399:     >>> native_dt == swapped_dt.newbyteorder('=')
6400:     True
6401:     >>> native_dt == swapped_dt.newbyteorder('N')
6402:     True
6403:     >>> native_dt == native_dt.newbyteorder('|')
6404:     True
6405:     >>> np.dtype('<i2') == native_dt.newbyteorder('<')
6406:     True
6407:     >>> np.dtype('<i2') == native_dt.newbyteorder('L')
6408:     True
6409:     >>> np.dtype('>i2') == native_dt.newbyteorder('>')
6410:     True
6411:     >>> np.dtype('>i2') == native_dt.newbyteorder('B')
6412:     True
6413: 
6414:     '''))
6415: 
6416: 
6417: ##############################################################################
6418: #
6419: # Datetime-related Methods
6420: #
6421: ##############################################################################
6422: 
6423: add_newdoc('numpy.core.multiarray', 'busdaycalendar',
6424:     '''
6425:     busdaycalendar(weekmask='1111100', holidays=None)
6426: 
6427:     A business day calendar object that efficiently stores information
6428:     defining valid days for the busday family of functions.
6429: 
6430:     The default valid days are Monday through Friday ("business days").
6431:     A busdaycalendar object can be specified with any set of weekly
6432:     valid days, plus an optional "holiday" dates that always will be invalid.
6433: 
6434:     Once a busdaycalendar object is created, the weekmask and holidays
6435:     cannot be modified.
6436: 
6437:     .. versionadded:: 1.7.0
6438: 
6439:     Parameters
6440:     ----------
6441:     weekmask : str or array_like of bool, optional
6442:         A seven-element array indicating which of Monday through Sunday are
6443:         valid days. May be specified as a length-seven list or array, like
6444:         [1,1,1,1,1,0,0]; a length-seven string, like '1111100'; or a string
6445:         like "Mon Tue Wed Thu Fri", made up of 3-character abbreviations for
6446:         weekdays, optionally separated by white space. Valid abbreviations
6447:         are: Mon Tue Wed Thu Fri Sat Sun
6448:     holidays : array_like of datetime64[D], optional
6449:         An array of dates to consider as invalid dates, no matter which
6450:         weekday they fall upon.  Holiday dates may be specified in any
6451:         order, and NaT (not-a-time) dates are ignored.  This list is
6452:         saved in a normalized form that is suited for fast calculations
6453:         of valid days.
6454: 
6455:     Returns
6456:     -------
6457:     out : busdaycalendar
6458:         A business day calendar object containing the specified
6459:         weekmask and holidays values.
6460: 
6461:     See Also
6462:     --------
6463:     is_busday : Returns a boolean array indicating valid days.
6464:     busday_offset : Applies an offset counted in valid days.
6465:     busday_count : Counts how many valid days are in a half-open date range.
6466: 
6467:     Attributes
6468:     ----------
6469:     Note: once a busdaycalendar object is created, you cannot modify the
6470:     weekmask or holidays.  The attributes return copies of internal data.
6471:     weekmask : (copy) seven-element array of bool
6472:     holidays : (copy) sorted array of datetime64[D]
6473: 
6474:     Examples
6475:     --------
6476:     >>> # Some important days in July
6477:     ... bdd = np.busdaycalendar(
6478:     ...             holidays=['2011-07-01', '2011-07-04', '2011-07-17'])
6479:     >>> # Default is Monday to Friday weekdays
6480:     ... bdd.weekmask
6481:     array([ True,  True,  True,  True,  True, False, False], dtype='bool')
6482:     >>> # Any holidays already on the weekend are removed
6483:     ... bdd.holidays
6484:     array(['2011-07-01', '2011-07-04'], dtype='datetime64[D]')
6485:     ''')
6486: 
6487: add_newdoc('numpy.core.multiarray', 'busdaycalendar', ('weekmask',
6488:     '''A copy of the seven-element boolean mask indicating valid days.'''))
6489: 
6490: add_newdoc('numpy.core.multiarray', 'busdaycalendar', ('holidays',
6491:     '''A copy of the holiday array indicating additional invalid days.'''))
6492: 
6493: add_newdoc('numpy.core.multiarray', 'is_busday',
6494:     '''
6495:     is_busday(dates, weekmask='1111100', holidays=None, busdaycal=None, out=None)
6496: 
6497:     Calculates which of the given dates are valid days, and which are not.
6498: 
6499:     .. versionadded:: 1.7.0
6500: 
6501:     Parameters
6502:     ----------
6503:     dates : array_like of datetime64[D]
6504:         The array of dates to process.
6505:     weekmask : str or array_like of bool, optional
6506:         A seven-element array indicating which of Monday through Sunday are
6507:         valid days. May be specified as a length-seven list or array, like
6508:         [1,1,1,1,1,0,0]; a length-seven string, like '1111100'; or a string
6509:         like "Mon Tue Wed Thu Fri", made up of 3-character abbreviations for
6510:         weekdays, optionally separated by white space. Valid abbreviations
6511:         are: Mon Tue Wed Thu Fri Sat Sun
6512:     holidays : array_like of datetime64[D], optional
6513:         An array of dates to consider as invalid dates.  They may be
6514:         specified in any order, and NaT (not-a-time) dates are ignored.
6515:         This list is saved in a normalized form that is suited for
6516:         fast calculations of valid days.
6517:     busdaycal : busdaycalendar, optional
6518:         A `busdaycalendar` object which specifies the valid days. If this
6519:         parameter is provided, neither weekmask nor holidays may be
6520:         provided.
6521:     out : array of bool, optional
6522:         If provided, this array is filled with the result.
6523: 
6524:     Returns
6525:     -------
6526:     out : array of bool
6527:         An array with the same shape as ``dates``, containing True for
6528:         each valid day, and False for each invalid day.
6529: 
6530:     See Also
6531:     --------
6532:     busdaycalendar: An object that specifies a custom set of valid days.
6533:     busday_offset : Applies an offset counted in valid days.
6534:     busday_count : Counts how many valid days are in a half-open date range.
6535: 
6536:     Examples
6537:     --------
6538:     >>> # The weekdays are Friday, Saturday, and Monday
6539:     ... np.is_busday(['2011-07-01', '2011-07-02', '2011-07-18'],
6540:     ...                 holidays=['2011-07-01', '2011-07-04', '2011-07-17'])
6541:     array([False, False,  True], dtype='bool')
6542:     ''')
6543: 
6544: add_newdoc('numpy.core.multiarray', 'busday_offset',
6545:     '''
6546:     busday_offset(dates, offsets, roll='raise', weekmask='1111100', holidays=None, busdaycal=None, out=None)
6547: 
6548:     First adjusts the date to fall on a valid day according to
6549:     the ``roll`` rule, then applies offsets to the given dates
6550:     counted in valid days.
6551: 
6552:     .. versionadded:: 1.7.0
6553: 
6554:     Parameters
6555:     ----------
6556:     dates : array_like of datetime64[D]
6557:         The array of dates to process.
6558:     offsets : array_like of int
6559:         The array of offsets, which is broadcast with ``dates``.
6560:     roll : {'raise', 'nat', 'forward', 'following', 'backward', 'preceding', 'modifiedfollowing', 'modifiedpreceding'}, optional
6561:         How to treat dates that do not fall on a valid day. The default
6562:         is 'raise'.
6563: 
6564:           * 'raise' means to raise an exception for an invalid day.
6565:           * 'nat' means to return a NaT (not-a-time) for an invalid day.
6566:           * 'forward' and 'following' mean to take the first valid day
6567:             later in time.
6568:           * 'backward' and 'preceding' mean to take the first valid day
6569:             earlier in time.
6570:           * 'modifiedfollowing' means to take the first valid day
6571:             later in time unless it is across a Month boundary, in which
6572:             case to take the first valid day earlier in time.
6573:           * 'modifiedpreceding' means to take the first valid day
6574:             earlier in time unless it is across a Month boundary, in which
6575:             case to take the first valid day later in time.
6576:     weekmask : str or array_like of bool, optional
6577:         A seven-element array indicating which of Monday through Sunday are
6578:         valid days. May be specified as a length-seven list or array, like
6579:         [1,1,1,1,1,0,0]; a length-seven string, like '1111100'; or a string
6580:         like "Mon Tue Wed Thu Fri", made up of 3-character abbreviations for
6581:         weekdays, optionally separated by white space. Valid abbreviations
6582:         are: Mon Tue Wed Thu Fri Sat Sun
6583:     holidays : array_like of datetime64[D], optional
6584:         An array of dates to consider as invalid dates.  They may be
6585:         specified in any order, and NaT (not-a-time) dates are ignored.
6586:         This list is saved in a normalized form that is suited for
6587:         fast calculations of valid days.
6588:     busdaycal : busdaycalendar, optional
6589:         A `busdaycalendar` object which specifies the valid days. If this
6590:         parameter is provided, neither weekmask nor holidays may be
6591:         provided.
6592:     out : array of datetime64[D], optional
6593:         If provided, this array is filled with the result.
6594: 
6595:     Returns
6596:     -------
6597:     out : array of datetime64[D]
6598:         An array with a shape from broadcasting ``dates`` and ``offsets``
6599:         together, containing the dates with offsets applied.
6600: 
6601:     See Also
6602:     --------
6603:     busdaycalendar: An object that specifies a custom set of valid days.
6604:     is_busday : Returns a boolean array indicating valid days.
6605:     busday_count : Counts how many valid days are in a half-open date range.
6606: 
6607:     Examples
6608:     --------
6609:     >>> # First business day in October 2011 (not accounting for holidays)
6610:     ... np.busday_offset('2011-10', 0, roll='forward')
6611:     numpy.datetime64('2011-10-03','D')
6612:     >>> # Last business day in February 2012 (not accounting for holidays)
6613:     ... np.busday_offset('2012-03', -1, roll='forward')
6614:     numpy.datetime64('2012-02-29','D')
6615:     >>> # Third Wednesday in January 2011
6616:     ... np.busday_offset('2011-01', 2, roll='forward', weekmask='Wed')
6617:     numpy.datetime64('2011-01-19','D')
6618:     >>> # 2012 Mother's Day in Canada and the U.S.
6619:     ... np.busday_offset('2012-05', 1, roll='forward', weekmask='Sun')
6620:     numpy.datetime64('2012-05-13','D')
6621: 
6622:     >>> # First business day on or after a date
6623:     ... np.busday_offset('2011-03-20', 0, roll='forward')
6624:     numpy.datetime64('2011-03-21','D')
6625:     >>> np.busday_offset('2011-03-22', 0, roll='forward')
6626:     numpy.datetime64('2011-03-22','D')
6627:     >>> # First business day after a date
6628:     ... np.busday_offset('2011-03-20', 1, roll='backward')
6629:     numpy.datetime64('2011-03-21','D')
6630:     >>> np.busday_offset('2011-03-22', 1, roll='backward')
6631:     numpy.datetime64('2011-03-23','D')
6632:     ''')
6633: 
6634: add_newdoc('numpy.core.multiarray', 'busday_count',
6635:     '''
6636:     busday_count(begindates, enddates, weekmask='1111100', holidays=[], busdaycal=None, out=None)
6637: 
6638:     Counts the number of valid days between `begindates` and
6639:     `enddates`, not including the day of `enddates`.
6640: 
6641:     If ``enddates`` specifies a date value that is earlier than the
6642:     corresponding ``begindates`` date value, the count will be negative.
6643: 
6644:     .. versionadded:: 1.7.0
6645: 
6646:     Parameters
6647:     ----------
6648:     begindates : array_like of datetime64[D]
6649:         The array of the first dates for counting.
6650:     enddates : array_like of datetime64[D]
6651:         The array of the end dates for counting, which are excluded
6652:         from the count themselves.
6653:     weekmask : str or array_like of bool, optional
6654:         A seven-element array indicating which of Monday through Sunday are
6655:         valid days. May be specified as a length-seven list or array, like
6656:         [1,1,1,1,1,0,0]; a length-seven string, like '1111100'; or a string
6657:         like "Mon Tue Wed Thu Fri", made up of 3-character abbreviations for
6658:         weekdays, optionally separated by white space. Valid abbreviations
6659:         are: Mon Tue Wed Thu Fri Sat Sun
6660:     holidays : array_like of datetime64[D], optional
6661:         An array of dates to consider as invalid dates.  They may be
6662:         specified in any order, and NaT (not-a-time) dates are ignored.
6663:         This list is saved in a normalized form that is suited for
6664:         fast calculations of valid days.
6665:     busdaycal : busdaycalendar, optional
6666:         A `busdaycalendar` object which specifies the valid days. If this
6667:         parameter is provided, neither weekmask nor holidays may be
6668:         provided.
6669:     out : array of int, optional
6670:         If provided, this array is filled with the result.
6671: 
6672:     Returns
6673:     -------
6674:     out : array of int
6675:         An array with a shape from broadcasting ``begindates`` and ``enddates``
6676:         together, containing the number of valid days between
6677:         the begin and end dates.
6678: 
6679:     See Also
6680:     --------
6681:     busdaycalendar: An object that specifies a custom set of valid days.
6682:     is_busday : Returns a boolean array indicating valid days.
6683:     busday_offset : Applies an offset counted in valid days.
6684: 
6685:     Examples
6686:     --------
6687:     >>> # Number of weekdays in January 2011
6688:     ... np.busday_count('2011-01', '2011-02')
6689:     21
6690:     >>> # Number of weekdays in 2011
6691:     ...  np.busday_count('2011', '2012')
6692:     260
6693:     >>> # Number of Saturdays in 2011
6694:     ... np.busday_count('2011', '2012', weekmask='Sat')
6695:     53
6696:     ''')
6697: 
6698: ##############################################################################
6699: #
6700: # nd_grid instances
6701: #
6702: ##############################################################################
6703: 
6704: add_newdoc('numpy.lib.index_tricks', 'mgrid',
6705:     '''
6706:     `nd_grid` instance which returns a dense multi-dimensional "meshgrid".
6707: 
6708:     An instance of `numpy.lib.index_tricks.nd_grid` which returns an dense
6709:     (or fleshed out) mesh-grid when indexed, so that each returned argument
6710:     has the same shape.  The dimensions and number of the output arrays are
6711:     equal to the number of indexing dimensions.  If the step length is not a
6712:     complex number, then the stop is not inclusive.
6713: 
6714:     However, if the step length is a **complex number** (e.g. 5j), then
6715:     the integer part of its magnitude is interpreted as specifying the
6716:     number of points to create between the start and stop values, where
6717:     the stop value **is inclusive**.
6718: 
6719:     Returns
6720:     ----------
6721:     mesh-grid `ndarrays` all of the same dimensions
6722: 
6723:     See Also
6724:     --------
6725:     numpy.lib.index_tricks.nd_grid : class of `ogrid` and `mgrid` objects
6726:     ogrid : like mgrid but returns open (not fleshed out) mesh grids
6727:     r_ : array concatenator
6728: 
6729:     Examples
6730:     --------
6731:     >>> np.mgrid[0:5,0:5]
6732:     array([[[0, 0, 0, 0, 0],
6733:             [1, 1, 1, 1, 1],
6734:             [2, 2, 2, 2, 2],
6735:             [3, 3, 3, 3, 3],
6736:             [4, 4, 4, 4, 4]],
6737:            [[0, 1, 2, 3, 4],
6738:             [0, 1, 2, 3, 4],
6739:             [0, 1, 2, 3, 4],
6740:             [0, 1, 2, 3, 4],
6741:             [0, 1, 2, 3, 4]]])
6742:     >>> np.mgrid[-1:1:5j]
6743:     array([-1. , -0.5,  0. ,  0.5,  1. ])
6744: 
6745:     ''')
6746: 
6747: add_newdoc('numpy.lib.index_tricks', 'ogrid',
6748:     '''
6749:     `nd_grid` instance which returns an open multi-dimensional "meshgrid".
6750: 
6751:     An instance of `numpy.lib.index_tricks.nd_grid` which returns an open
6752:     (i.e. not fleshed out) mesh-grid when indexed, so that only one dimension
6753:     of each returned array is greater than 1.  The dimension and number of the
6754:     output arrays are equal to the number of indexing dimensions.  If the step
6755:     length is not a complex number, then the stop is not inclusive.
6756: 
6757:     However, if the step length is a **complex number** (e.g. 5j), then
6758:     the integer part of its magnitude is interpreted as specifying the
6759:     number of points to create between the start and stop values, where
6760:     the stop value **is inclusive**.
6761: 
6762:     Returns
6763:     ----------
6764:     mesh-grid `ndarrays` with only one dimension :math:`\\neq 1`
6765: 
6766:     See Also
6767:     --------
6768:     np.lib.index_tricks.nd_grid : class of `ogrid` and `mgrid` objects
6769:     mgrid : like `ogrid` but returns dense (or fleshed out) mesh grids
6770:     r_ : array concatenator
6771: 
6772:     Examples
6773:     --------
6774:     >>> from numpy import ogrid
6775:     >>> ogrid[-1:1:5j]
6776:     array([-1. , -0.5,  0. ,  0.5,  1. ])
6777:     >>> ogrid[0:5,0:5]
6778:     [array([[0],
6779:             [1],
6780:             [2],
6781:             [3],
6782:             [4]]), array([[0, 1, 2, 3, 4]])]
6783: 
6784:     ''')
6785: 
6786: 
6787: ##############################################################################
6788: #
6789: # Documentation for `generic` attributes and methods
6790: #
6791: ##############################################################################
6792: 
6793: add_newdoc('numpy.core.numerictypes', 'generic',
6794:     '''
6795:     Base class for numpy scalar types.
6796: 
6797:     Class from which most (all?) numpy scalar types are derived.  For
6798:     consistency, exposes the same API as `ndarray`, despite many
6799:     consequent attributes being either "get-only," or completely irrelevant.
6800:     This is the class from which it is strongly suggested users should derive
6801:     custom scalar types.
6802: 
6803:     ''')
6804: 
6805: # Attributes
6806: 
6807: add_newdoc('numpy.core.numerictypes', 'generic', ('T',
6808:     '''
6809:     Not implemented (virtual attribute)
6810: 
6811:     Class generic exists solely to derive numpy scalars from, and possesses,
6812:     albeit unimplemented, all the attributes of the ndarray class so as to
6813:     provide a uniform API.
6814: 
6815:     See Also
6816:     --------
6817:     The corresponding attribute of the derived class of interest.
6818: 
6819:     '''))
6820: 
6821: add_newdoc('numpy.core.numerictypes', 'generic', ('base',
6822:     '''
6823:     Not implemented (virtual attribute)
6824: 
6825:     Class generic exists solely to derive numpy scalars from, and possesses,
6826:     albeit unimplemented, all the attributes of the ndarray class so as to
6827:     a uniform API.
6828: 
6829:     See Also
6830:     --------
6831:     The corresponding attribute of the derived class of interest.
6832: 
6833:     '''))
6834: 
6835: add_newdoc('numpy.core.numerictypes', 'generic', ('data',
6836:     '''Pointer to start of data.'''))
6837: 
6838: add_newdoc('numpy.core.numerictypes', 'generic', ('dtype',
6839:     '''Get array data-descriptor.'''))
6840: 
6841: add_newdoc('numpy.core.numerictypes', 'generic', ('flags',
6842:     '''The integer value of flags.'''))
6843: 
6844: add_newdoc('numpy.core.numerictypes', 'generic', ('flat',
6845:     '''A 1-D view of the scalar.'''))
6846: 
6847: add_newdoc('numpy.core.numerictypes', 'generic', ('imag',
6848:     '''The imaginary part of the scalar.'''))
6849: 
6850: add_newdoc('numpy.core.numerictypes', 'generic', ('itemsize',
6851:     '''The length of one element in bytes.'''))
6852: 
6853: add_newdoc('numpy.core.numerictypes', 'generic', ('nbytes',
6854:     '''The length of the scalar in bytes.'''))
6855: 
6856: add_newdoc('numpy.core.numerictypes', 'generic', ('ndim',
6857:     '''The number of array dimensions.'''))
6858: 
6859: add_newdoc('numpy.core.numerictypes', 'generic', ('real',
6860:     '''The real part of the scalar.'''))
6861: 
6862: add_newdoc('numpy.core.numerictypes', 'generic', ('shape',
6863:     '''Tuple of array dimensions.'''))
6864: 
6865: add_newdoc('numpy.core.numerictypes', 'generic', ('size',
6866:     '''The number of elements in the gentype.'''))
6867: 
6868: add_newdoc('numpy.core.numerictypes', 'generic', ('strides',
6869:     '''Tuple of bytes steps in each dimension.'''))
6870: 
6871: # Methods
6872: 
6873: add_newdoc('numpy.core.numerictypes', 'generic', ('all',
6874:     '''
6875:     Not implemented (virtual attribute)
6876: 
6877:     Class generic exists solely to derive numpy scalars from, and possesses,
6878:     albeit unimplemented, all the attributes of the ndarray class
6879:     so as to provide a uniform API.
6880: 
6881:     See Also
6882:     --------
6883:     The corresponding attribute of the derived class of interest.
6884: 
6885:     '''))
6886: 
6887: add_newdoc('numpy.core.numerictypes', 'generic', ('any',
6888:     '''
6889:     Not implemented (virtual attribute)
6890: 
6891:     Class generic exists solely to derive numpy scalars from, and possesses,
6892:     albeit unimplemented, all the attributes of the ndarray class
6893:     so as to provide a uniform API.
6894: 
6895:     See Also
6896:     --------
6897:     The corresponding attribute of the derived class of interest.
6898: 
6899:     '''))
6900: 
6901: add_newdoc('numpy.core.numerictypes', 'generic', ('argmax',
6902:     '''
6903:     Not implemented (virtual attribute)
6904: 
6905:     Class generic exists solely to derive numpy scalars from, and possesses,
6906:     albeit unimplemented, all the attributes of the ndarray class
6907:     so as to provide a uniform API.
6908: 
6909:     See Also
6910:     --------
6911:     The corresponding attribute of the derived class of interest.
6912: 
6913:     '''))
6914: 
6915: add_newdoc('numpy.core.numerictypes', 'generic', ('argmin',
6916:     '''
6917:     Not implemented (virtual attribute)
6918: 
6919:     Class generic exists solely to derive numpy scalars from, and possesses,
6920:     albeit unimplemented, all the attributes of the ndarray class
6921:     so as to provide a uniform API.
6922: 
6923:     See Also
6924:     --------
6925:     The corresponding attribute of the derived class of interest.
6926: 
6927:     '''))
6928: 
6929: add_newdoc('numpy.core.numerictypes', 'generic', ('argsort',
6930:     '''
6931:     Not implemented (virtual attribute)
6932: 
6933:     Class generic exists solely to derive numpy scalars from, and possesses,
6934:     albeit unimplemented, all the attributes of the ndarray class
6935:     so as to provide a uniform API.
6936: 
6937:     See Also
6938:     --------
6939:     The corresponding attribute of the derived class of interest.
6940: 
6941:     '''))
6942: 
6943: add_newdoc('numpy.core.numerictypes', 'generic', ('astype',
6944:     '''
6945:     Not implemented (virtual attribute)
6946: 
6947:     Class generic exists solely to derive numpy scalars from, and possesses,
6948:     albeit unimplemented, all the attributes of the ndarray class
6949:     so as to provide a uniform API.
6950: 
6951:     See Also
6952:     --------
6953:     The corresponding attribute of the derived class of interest.
6954: 
6955:     '''))
6956: 
6957: add_newdoc('numpy.core.numerictypes', 'generic', ('byteswap',
6958:     '''
6959:     Not implemented (virtual attribute)
6960: 
6961:     Class generic exists solely to derive numpy scalars from, and possesses,
6962:     albeit unimplemented, all the attributes of the ndarray class so as to
6963:     provide a uniform API.
6964: 
6965:     See Also
6966:     --------
6967:     The corresponding attribute of the derived class of interest.
6968: 
6969:     '''))
6970: 
6971: add_newdoc('numpy.core.numerictypes', 'generic', ('choose',
6972:     '''
6973:     Not implemented (virtual attribute)
6974: 
6975:     Class generic exists solely to derive numpy scalars from, and possesses,
6976:     albeit unimplemented, all the attributes of the ndarray class
6977:     so as to provide a uniform API.
6978: 
6979:     See Also
6980:     --------
6981:     The corresponding attribute of the derived class of interest.
6982: 
6983:     '''))
6984: 
6985: add_newdoc('numpy.core.numerictypes', 'generic', ('clip',
6986:     '''
6987:     Not implemented (virtual attribute)
6988: 
6989:     Class generic exists solely to derive numpy scalars from, and possesses,
6990:     albeit unimplemented, all the attributes of the ndarray class
6991:     so as to provide a uniform API.
6992: 
6993:     See Also
6994:     --------
6995:     The corresponding attribute of the derived class of interest.
6996: 
6997:     '''))
6998: 
6999: add_newdoc('numpy.core.numerictypes', 'generic', ('compress',
7000:     '''
7001:     Not implemented (virtual attribute)
7002: 
7003:     Class generic exists solely to derive numpy scalars from, and possesses,
7004:     albeit unimplemented, all the attributes of the ndarray class
7005:     so as to provide a uniform API.
7006: 
7007:     See Also
7008:     --------
7009:     The corresponding attribute of the derived class of interest.
7010: 
7011:     '''))
7012: 
7013: add_newdoc('numpy.core.numerictypes', 'generic', ('conjugate',
7014:     '''
7015:     Not implemented (virtual attribute)
7016: 
7017:     Class generic exists solely to derive numpy scalars from, and possesses,
7018:     albeit unimplemented, all the attributes of the ndarray class
7019:     so as to provide a uniform API.
7020: 
7021:     See Also
7022:     --------
7023:     The corresponding attribute of the derived class of interest.
7024: 
7025:     '''))
7026: 
7027: add_newdoc('numpy.core.numerictypes', 'generic', ('copy',
7028:     '''
7029:     Not implemented (virtual attribute)
7030: 
7031:     Class generic exists solely to derive numpy scalars from, and possesses,
7032:     albeit unimplemented, all the attributes of the ndarray class
7033:     so as to provide a uniform API.
7034: 
7035:     See Also
7036:     --------
7037:     The corresponding attribute of the derived class of interest.
7038: 
7039:     '''))
7040: 
7041: add_newdoc('numpy.core.numerictypes', 'generic', ('cumprod',
7042:     '''
7043:     Not implemented (virtual attribute)
7044: 
7045:     Class generic exists solely to derive numpy scalars from, and possesses,
7046:     albeit unimplemented, all the attributes of the ndarray class
7047:     so as to provide a uniform API.
7048: 
7049:     See Also
7050:     --------
7051:     The corresponding attribute of the derived class of interest.
7052: 
7053:     '''))
7054: 
7055: add_newdoc('numpy.core.numerictypes', 'generic', ('cumsum',
7056:     '''
7057:     Not implemented (virtual attribute)
7058: 
7059:     Class generic exists solely to derive numpy scalars from, and possesses,
7060:     albeit unimplemented, all the attributes of the ndarray class
7061:     so as to provide a uniform API.
7062: 
7063:     See Also
7064:     --------
7065:     The corresponding attribute of the derived class of interest.
7066: 
7067:     '''))
7068: 
7069: add_newdoc('numpy.core.numerictypes', 'generic', ('diagonal',
7070:     '''
7071:     Not implemented (virtual attribute)
7072: 
7073:     Class generic exists solely to derive numpy scalars from, and possesses,
7074:     albeit unimplemented, all the attributes of the ndarray class
7075:     so as to provide a uniform API.
7076: 
7077:     See Also
7078:     --------
7079:     The corresponding attribute of the derived class of interest.
7080: 
7081:     '''))
7082: 
7083: add_newdoc('numpy.core.numerictypes', 'generic', ('dump',
7084:     '''
7085:     Not implemented (virtual attribute)
7086: 
7087:     Class generic exists solely to derive numpy scalars from, and possesses,
7088:     albeit unimplemented, all the attributes of the ndarray class
7089:     so as to provide a uniform API.
7090: 
7091:     See Also
7092:     --------
7093:     The corresponding attribute of the derived class of interest.
7094: 
7095:     '''))
7096: 
7097: add_newdoc('numpy.core.numerictypes', 'generic', ('dumps',
7098:     '''
7099:     Not implemented (virtual attribute)
7100: 
7101:     Class generic exists solely to derive numpy scalars from, and possesses,
7102:     albeit unimplemented, all the attributes of the ndarray class
7103:     so as to provide a uniform API.
7104: 
7105:     See Also
7106:     --------
7107:     The corresponding attribute of the derived class of interest.
7108: 
7109:     '''))
7110: 
7111: add_newdoc('numpy.core.numerictypes', 'generic', ('fill',
7112:     '''
7113:     Not implemented (virtual attribute)
7114: 
7115:     Class generic exists solely to derive numpy scalars from, and possesses,
7116:     albeit unimplemented, all the attributes of the ndarray class
7117:     so as to provide a uniform API.
7118: 
7119:     See Also
7120:     --------
7121:     The corresponding attribute of the derived class of interest.
7122: 
7123:     '''))
7124: 
7125: add_newdoc('numpy.core.numerictypes', 'generic', ('flatten',
7126:     '''
7127:     Not implemented (virtual attribute)
7128: 
7129:     Class generic exists solely to derive numpy scalars from, and possesses,
7130:     albeit unimplemented, all the attributes of the ndarray class
7131:     so as to provide a uniform API.
7132: 
7133:     See Also
7134:     --------
7135:     The corresponding attribute of the derived class of interest.
7136: 
7137:     '''))
7138: 
7139: add_newdoc('numpy.core.numerictypes', 'generic', ('getfield',
7140:     '''
7141:     Not implemented (virtual attribute)
7142: 
7143:     Class generic exists solely to derive numpy scalars from, and possesses,
7144:     albeit unimplemented, all the attributes of the ndarray class
7145:     so as to provide a uniform API.
7146: 
7147:     See Also
7148:     --------
7149:     The corresponding attribute of the derived class of interest.
7150: 
7151:     '''))
7152: 
7153: add_newdoc('numpy.core.numerictypes', 'generic', ('item',
7154:     '''
7155:     Not implemented (virtual attribute)
7156: 
7157:     Class generic exists solely to derive numpy scalars from, and possesses,
7158:     albeit unimplemented, all the attributes of the ndarray class
7159:     so as to provide a uniform API.
7160: 
7161:     See Also
7162:     --------
7163:     The corresponding attribute of the derived class of interest.
7164: 
7165:     '''))
7166: 
7167: add_newdoc('numpy.core.numerictypes', 'generic', ('itemset',
7168:     '''
7169:     Not implemented (virtual attribute)
7170: 
7171:     Class generic exists solely to derive numpy scalars from, and possesses,
7172:     albeit unimplemented, all the attributes of the ndarray class
7173:     so as to provide a uniform API.
7174: 
7175:     See Also
7176:     --------
7177:     The corresponding attribute of the derived class of interest.
7178: 
7179:     '''))
7180: 
7181: add_newdoc('numpy.core.numerictypes', 'generic', ('max',
7182:     '''
7183:     Not implemented (virtual attribute)
7184: 
7185:     Class generic exists solely to derive numpy scalars from, and possesses,
7186:     albeit unimplemented, all the attributes of the ndarray class
7187:     so as to provide a uniform API.
7188: 
7189:     See Also
7190:     --------
7191:     The corresponding attribute of the derived class of interest.
7192: 
7193:     '''))
7194: 
7195: add_newdoc('numpy.core.numerictypes', 'generic', ('mean',
7196:     '''
7197:     Not implemented (virtual attribute)
7198: 
7199:     Class generic exists solely to derive numpy scalars from, and possesses,
7200:     albeit unimplemented, all the attributes of the ndarray class
7201:     so as to provide a uniform API.
7202: 
7203:     See Also
7204:     --------
7205:     The corresponding attribute of the derived class of interest.
7206: 
7207:     '''))
7208: 
7209: add_newdoc('numpy.core.numerictypes', 'generic', ('min',
7210:     '''
7211:     Not implemented (virtual attribute)
7212: 
7213:     Class generic exists solely to derive numpy scalars from, and possesses,
7214:     albeit unimplemented, all the attributes of the ndarray class
7215:     so as to provide a uniform API.
7216: 
7217:     See Also
7218:     --------
7219:     The corresponding attribute of the derived class of interest.
7220: 
7221:     '''))
7222: 
7223: add_newdoc('numpy.core.numerictypes', 'generic', ('newbyteorder',
7224:     '''
7225:     newbyteorder(new_order='S')
7226: 
7227:     Return a new `dtype` with a different byte order.
7228: 
7229:     Changes are also made in all fields and sub-arrays of the data type.
7230: 
7231:     The `new_order` code can be any from the following:
7232: 
7233:     * 'S' - swap dtype from current to opposite endian
7234:     * {'<', 'L'} - little endian
7235:     * {'>', 'B'} - big endian
7236:     * {'=', 'N'} - native order
7237:     * {'|', 'I'} - ignore (no change to byte order)
7238: 
7239:     Parameters
7240:     ----------
7241:     new_order : str, optional
7242:         Byte order to force; a value from the byte order specifications
7243:         above.  The default value ('S') results in swapping the current
7244:         byte order. The code does a case-insensitive check on the first
7245:         letter of `new_order` for the alternatives above.  For example,
7246:         any of 'B' or 'b' or 'biggish' are valid to specify big-endian.
7247: 
7248: 
7249:     Returns
7250:     -------
7251:     new_dtype : dtype
7252:         New `dtype` object with the given change to the byte order.
7253: 
7254:     '''))
7255: 
7256: add_newdoc('numpy.core.numerictypes', 'generic', ('nonzero',
7257:     '''
7258:     Not implemented (virtual attribute)
7259: 
7260:     Class generic exists solely to derive numpy scalars from, and possesses,
7261:     albeit unimplemented, all the attributes of the ndarray class
7262:     so as to provide a uniform API.
7263: 
7264:     See Also
7265:     --------
7266:     The corresponding attribute of the derived class of interest.
7267: 
7268:     '''))
7269: 
7270: add_newdoc('numpy.core.numerictypes', 'generic', ('prod',
7271:     '''
7272:     Not implemented (virtual attribute)
7273: 
7274:     Class generic exists solely to derive numpy scalars from, and possesses,
7275:     albeit unimplemented, all the attributes of the ndarray class
7276:     so as to provide a uniform API.
7277: 
7278:     See Also
7279:     --------
7280:     The corresponding attribute of the derived class of interest.
7281: 
7282:     '''))
7283: 
7284: add_newdoc('numpy.core.numerictypes', 'generic', ('ptp',
7285:     '''
7286:     Not implemented (virtual attribute)
7287: 
7288:     Class generic exists solely to derive numpy scalars from, and possesses,
7289:     albeit unimplemented, all the attributes of the ndarray class
7290:     so as to provide a uniform API.
7291: 
7292:     See Also
7293:     --------
7294:     The corresponding attribute of the derived class of interest.
7295: 
7296:     '''))
7297: 
7298: add_newdoc('numpy.core.numerictypes', 'generic', ('put',
7299:     '''
7300:     Not implemented (virtual attribute)
7301: 
7302:     Class generic exists solely to derive numpy scalars from, and possesses,
7303:     albeit unimplemented, all the attributes of the ndarray class
7304:     so as to provide a uniform API.
7305: 
7306:     See Also
7307:     --------
7308:     The corresponding attribute of the derived class of interest.
7309: 
7310:     '''))
7311: 
7312: add_newdoc('numpy.core.numerictypes', 'generic', ('ravel',
7313:     '''
7314:     Not implemented (virtual attribute)
7315: 
7316:     Class generic exists solely to derive numpy scalars from, and possesses,
7317:     albeit unimplemented, all the attributes of the ndarray class
7318:     so as to provide a uniform API.
7319: 
7320:     See Also
7321:     --------
7322:     The corresponding attribute of the derived class of interest.
7323: 
7324:     '''))
7325: 
7326: add_newdoc('numpy.core.numerictypes', 'generic', ('repeat',
7327:     '''
7328:     Not implemented (virtual attribute)
7329: 
7330:     Class generic exists solely to derive numpy scalars from, and possesses,
7331:     albeit unimplemented, all the attributes of the ndarray class
7332:     so as to provide a uniform API.
7333: 
7334:     See Also
7335:     --------
7336:     The corresponding attribute of the derived class of interest.
7337: 
7338:     '''))
7339: 
7340: add_newdoc('numpy.core.numerictypes', 'generic', ('reshape',
7341:     '''
7342:     Not implemented (virtual attribute)
7343: 
7344:     Class generic exists solely to derive numpy scalars from, and possesses,
7345:     albeit unimplemented, all the attributes of the ndarray class
7346:     so as to provide a uniform API.
7347: 
7348:     See Also
7349:     --------
7350:     The corresponding attribute of the derived class of interest.
7351: 
7352:     '''))
7353: 
7354: add_newdoc('numpy.core.numerictypes', 'generic', ('resize',
7355:     '''
7356:     Not implemented (virtual attribute)
7357: 
7358:     Class generic exists solely to derive numpy scalars from, and possesses,
7359:     albeit unimplemented, all the attributes of the ndarray class
7360:     so as to provide a uniform API.
7361: 
7362:     See Also
7363:     --------
7364:     The corresponding attribute of the derived class of interest.
7365: 
7366:     '''))
7367: 
7368: add_newdoc('numpy.core.numerictypes', 'generic', ('round',
7369:     '''
7370:     Not implemented (virtual attribute)
7371: 
7372:     Class generic exists solely to derive numpy scalars from, and possesses,
7373:     albeit unimplemented, all the attributes of the ndarray class
7374:     so as to provide a uniform API.
7375: 
7376:     See Also
7377:     --------
7378:     The corresponding attribute of the derived class of interest.
7379: 
7380:     '''))
7381: 
7382: add_newdoc('numpy.core.numerictypes', 'generic', ('searchsorted',
7383:     '''
7384:     Not implemented (virtual attribute)
7385: 
7386:     Class generic exists solely to derive numpy scalars from, and possesses,
7387:     albeit unimplemented, all the attributes of the ndarray class
7388:     so as to provide a uniform API.
7389: 
7390:     See Also
7391:     --------
7392:     The corresponding attribute of the derived class of interest.
7393: 
7394:     '''))
7395: 
7396: add_newdoc('numpy.core.numerictypes', 'generic', ('setfield',
7397:     '''
7398:     Not implemented (virtual attribute)
7399: 
7400:     Class generic exists solely to derive numpy scalars from, and possesses,
7401:     albeit unimplemented, all the attributes of the ndarray class
7402:     so as to provide a uniform API.
7403: 
7404:     See Also
7405:     --------
7406:     The corresponding attribute of the derived class of interest.
7407: 
7408:     '''))
7409: 
7410: add_newdoc('numpy.core.numerictypes', 'generic', ('setflags',
7411:     '''
7412:     Not implemented (virtual attribute)
7413: 
7414:     Class generic exists solely to derive numpy scalars from, and possesses,
7415:     albeit unimplemented, all the attributes of the ndarray class so as to
7416:     provide a uniform API.
7417: 
7418:     See Also
7419:     --------
7420:     The corresponding attribute of the derived class of interest.
7421: 
7422:     '''))
7423: 
7424: add_newdoc('numpy.core.numerictypes', 'generic', ('sort',
7425:     '''
7426:     Not implemented (virtual attribute)
7427: 
7428:     Class generic exists solely to derive numpy scalars from, and possesses,
7429:     albeit unimplemented, all the attributes of the ndarray class
7430:     so as to provide a uniform API.
7431: 
7432:     See Also
7433:     --------
7434:     The corresponding attribute of the derived class of interest.
7435: 
7436:     '''))
7437: 
7438: add_newdoc('numpy.core.numerictypes', 'generic', ('squeeze',
7439:     '''
7440:     Not implemented (virtual attribute)
7441: 
7442:     Class generic exists solely to derive numpy scalars from, and possesses,
7443:     albeit unimplemented, all the attributes of the ndarray class
7444:     so as to provide a uniform API.
7445: 
7446:     See Also
7447:     --------
7448:     The corresponding attribute of the derived class of interest.
7449: 
7450:     '''))
7451: 
7452: add_newdoc('numpy.core.numerictypes', 'generic', ('std',
7453:     '''
7454:     Not implemented (virtual attribute)
7455: 
7456:     Class generic exists solely to derive numpy scalars from, and possesses,
7457:     albeit unimplemented, all the attributes of the ndarray class
7458:     so as to provide a uniform API.
7459: 
7460:     See Also
7461:     --------
7462:     The corresponding attribute of the derived class of interest.
7463: 
7464:     '''))
7465: 
7466: add_newdoc('numpy.core.numerictypes', 'generic', ('sum',
7467:     '''
7468:     Not implemented (virtual attribute)
7469: 
7470:     Class generic exists solely to derive numpy scalars from, and possesses,
7471:     albeit unimplemented, all the attributes of the ndarray class
7472:     so as to provide a uniform API.
7473: 
7474:     See Also
7475:     --------
7476:     The corresponding attribute of the derived class of interest.
7477: 
7478:     '''))
7479: 
7480: add_newdoc('numpy.core.numerictypes', 'generic', ('swapaxes',
7481:     '''
7482:     Not implemented (virtual attribute)
7483: 
7484:     Class generic exists solely to derive numpy scalars from, and possesses,
7485:     albeit unimplemented, all the attributes of the ndarray class
7486:     so as to provide a uniform API.
7487: 
7488:     See Also
7489:     --------
7490:     The corresponding attribute of the derived class of interest.
7491: 
7492:     '''))
7493: 
7494: add_newdoc('numpy.core.numerictypes', 'generic', ('take',
7495:     '''
7496:     Not implemented (virtual attribute)
7497: 
7498:     Class generic exists solely to derive numpy scalars from, and possesses,
7499:     albeit unimplemented, all the attributes of the ndarray class
7500:     so as to provide a uniform API.
7501: 
7502:     See Also
7503:     --------
7504:     The corresponding attribute of the derived class of interest.
7505: 
7506:     '''))
7507: 
7508: add_newdoc('numpy.core.numerictypes', 'generic', ('tofile',
7509:     '''
7510:     Not implemented (virtual attribute)
7511: 
7512:     Class generic exists solely to derive numpy scalars from, and possesses,
7513:     albeit unimplemented, all the attributes of the ndarray class
7514:     so as to provide a uniform API.
7515: 
7516:     See Also
7517:     --------
7518:     The corresponding attribute of the derived class of interest.
7519: 
7520:     '''))
7521: 
7522: add_newdoc('numpy.core.numerictypes', 'generic', ('tolist',
7523:     '''
7524:     Not implemented (virtual attribute)
7525: 
7526:     Class generic exists solely to derive numpy scalars from, and possesses,
7527:     albeit unimplemented, all the attributes of the ndarray class
7528:     so as to provide a uniform API.
7529: 
7530:     See Also
7531:     --------
7532:     The corresponding attribute of the derived class of interest.
7533: 
7534:     '''))
7535: 
7536: add_newdoc('numpy.core.numerictypes', 'generic', ('tostring',
7537:     '''
7538:     Not implemented (virtual attribute)
7539: 
7540:     Class generic exists solely to derive numpy scalars from, and possesses,
7541:     albeit unimplemented, all the attributes of the ndarray class
7542:     so as to provide a uniform API.
7543: 
7544:     See Also
7545:     --------
7546:     The corresponding attribute of the derived class of interest.
7547: 
7548:     '''))
7549: 
7550: add_newdoc('numpy.core.numerictypes', 'generic', ('trace',
7551:     '''
7552:     Not implemented (virtual attribute)
7553: 
7554:     Class generic exists solely to derive numpy scalars from, and possesses,
7555:     albeit unimplemented, all the attributes of the ndarray class
7556:     so as to provide a uniform API.
7557: 
7558:     See Also
7559:     --------
7560:     The corresponding attribute of the derived class of interest.
7561: 
7562:     '''))
7563: 
7564: add_newdoc('numpy.core.numerictypes', 'generic', ('transpose',
7565:     '''
7566:     Not implemented (virtual attribute)
7567: 
7568:     Class generic exists solely to derive numpy scalars from, and possesses,
7569:     albeit unimplemented, all the attributes of the ndarray class
7570:     so as to provide a uniform API.
7571: 
7572:     See Also
7573:     --------
7574:     The corresponding attribute of the derived class of interest.
7575: 
7576:     '''))
7577: 
7578: add_newdoc('numpy.core.numerictypes', 'generic', ('var',
7579:     '''
7580:     Not implemented (virtual attribute)
7581: 
7582:     Class generic exists solely to derive numpy scalars from, and possesses,
7583:     albeit unimplemented, all the attributes of the ndarray class
7584:     so as to provide a uniform API.
7585: 
7586:     See Also
7587:     --------
7588:     The corresponding attribute of the derived class of interest.
7589: 
7590:     '''))
7591: 
7592: add_newdoc('numpy.core.numerictypes', 'generic', ('view',
7593:     '''
7594:     Not implemented (virtual attribute)
7595: 
7596:     Class generic exists solely to derive numpy scalars from, and possesses,
7597:     albeit unimplemented, all the attributes of the ndarray class
7598:     so as to provide a uniform API.
7599: 
7600:     See Also
7601:     --------
7602:     The corresponding attribute of the derived class of interest.
7603: 
7604:     '''))
7605: 
7606: 
7607: ##############################################################################
7608: #
7609: # Documentation for other scalar classes
7610: #
7611: ##############################################################################
7612: 
7613: add_newdoc('numpy.core.numerictypes', 'bool_',
7614:     '''Numpy's Boolean type.  Character code: ``?``.  Alias: bool8''')
7615: 
7616: add_newdoc('numpy.core.numerictypes', 'complex64',
7617:     '''
7618:     Complex number type composed of two 32 bit floats. Character code: 'F'.
7619: 
7620:     ''')
7621: 
7622: add_newdoc('numpy.core.numerictypes', 'complex128',
7623:     '''
7624:     Complex number type composed of two 64 bit floats. Character code: 'D'.
7625:     Python complex compatible.
7626: 
7627:     ''')
7628: 
7629: add_newdoc('numpy.core.numerictypes', 'complex256',
7630:     '''
7631:     Complex number type composed of two 128-bit floats. Character code: 'G'.
7632: 
7633:     ''')
7634: 
7635: add_newdoc('numpy.core.numerictypes', 'float32',
7636:     '''
7637:     32-bit floating-point number. Character code 'f'. C float compatible.
7638: 
7639:     ''')
7640: 
7641: add_newdoc('numpy.core.numerictypes', 'float64',
7642:     '''
7643:     64-bit floating-point number. Character code 'd'. Python float compatible.
7644: 
7645:     ''')
7646: 
7647: add_newdoc('numpy.core.numerictypes', 'float96',
7648:     '''
7649:     ''')
7650: 
7651: add_newdoc('numpy.core.numerictypes', 'float128',
7652:     '''
7653:     128-bit floating-point number. Character code: 'g'. C long float
7654:     compatible.
7655: 
7656:     ''')
7657: 
7658: add_newdoc('numpy.core.numerictypes', 'int8',
7659:     '''8-bit integer. Character code ``b``. C char compatible.''')
7660: 
7661: add_newdoc('numpy.core.numerictypes', 'int16',
7662:     '''16-bit integer. Character code ``h``. C short compatible.''')
7663: 
7664: add_newdoc('numpy.core.numerictypes', 'int32',
7665:     '''32-bit integer. Character code 'i'. C int compatible.''')
7666: 
7667: add_newdoc('numpy.core.numerictypes', 'int64',
7668:     '''64-bit integer. Character code 'l'. Python int compatible.''')
7669: 
7670: add_newdoc('numpy.core.numerictypes', 'object_',
7671:     '''Any Python object.  Character code: 'O'.''')
7672: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_21086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, (-1)), 'str', '\nThis is only meant to add docs to objects defined in C-extension modules.\nThe purpose is to allow easier editing of the docstrings without\nrequiring a re-compile.\n\nNOTE: Many of the methods of ndarray have corresponding functions.\n      If you update these docstrings, please keep also the ones in\n      core/fromnumeric.py, core/defmatrix.py up-to-date.\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from numpy.lib import add_newdoc' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/')
import_21087 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy.lib')

if (type(import_21087) is not StypyTypeError):

    if (import_21087 != 'pyd_module'):
        __import__(import_21087)
        sys_modules_21088 = sys.modules[import_21087]
        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy.lib', sys_modules_21088.module_type_store, module_type_store, ['add_newdoc'])
        nest_module(stypy.reporting.localization.Localization(__file__, 13, 0), __file__, sys_modules_21088, sys_modules_21088.module_type_store, module_type_store)
    else:
        from numpy.lib import add_newdoc

        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy.lib', None, module_type_store, ['add_newdoc'], [add_newdoc])

else:
    # Assigning a type to the variable 'numpy.lib' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy.lib', import_21087)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/')


# Call to add_newdoc(...): (line 23)
# Processing the call arguments (line 23)
str_21090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 11), 'str', 'numpy.core')
str_21091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 25), 'str', 'flatiter')
str_21092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, (-1)), 'str', "\n    Flat iterator object to iterate over arrays.\n\n    A `flatiter` iterator is returned by ``x.flat`` for any array `x`.\n    It allows iterating over the array as if it were a 1-D array,\n    either in a for-loop or by calling its `next` method.\n\n    Iteration is done in row-major, C-style order (the last\n    index varying the fastest). The iterator can also be indexed using\n    basic slicing or advanced indexing.\n\n    See Also\n    --------\n    ndarray.flat : Return a flat iterator over an array.\n    ndarray.flatten : Returns a flattened copy of an array.\n\n    Notes\n    -----\n    A `flatiter` iterator can not be constructed directly from Python code\n    by calling the `flatiter` constructor.\n\n    Examples\n    --------\n    >>> x = np.arange(6).reshape(2, 3)\n    >>> fl = x.flat\n    >>> type(fl)\n    <type 'numpy.flatiter'>\n    >>> for item in fl:\n    ...     print(item)\n    ...\n    0\n    1\n    2\n    3\n    4\n    5\n\n    >>> fl[2:4]\n    array([2, 3])\n\n    ")
# Processing the call keyword arguments (line 23)
kwargs_21093 = {}
# Getting the type of 'add_newdoc' (line 23)
add_newdoc_21089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 23)
add_newdoc_call_result_21094 = invoke(stypy.reporting.localization.Localization(__file__, 23, 0), add_newdoc_21089, *[str_21090, str_21091, str_21092], **kwargs_21093)


# Call to add_newdoc(...): (line 68)
# Processing the call arguments (line 68)
str_21096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 11), 'str', 'numpy.core')
str_21097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 25), 'str', 'flatiter')

# Obtaining an instance of the builtin type 'tuple' (line 68)
tuple_21098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 38), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 68)
# Adding element type (line 68)
str_21099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 38), 'str', 'base')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 38), tuple_21098, str_21099)
# Adding element type (line 68)
str_21100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, (-1)), 'str', '\n    A reference to the array that is iterated over.\n\n    Examples\n    --------\n    >>> x = np.arange(5)\n    >>> fl = x.flat\n    >>> fl.base is x\n    True\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 38), tuple_21098, str_21100)

# Processing the call keyword arguments (line 68)
kwargs_21101 = {}
# Getting the type of 'add_newdoc' (line 68)
add_newdoc_21095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 68)
add_newdoc_call_result_21102 = invoke(stypy.reporting.localization.Localization(__file__, 68, 0), add_newdoc_21095, *[str_21096, str_21097, tuple_21098], **kwargs_21101)


# Call to add_newdoc(...): (line 83)
# Processing the call arguments (line 83)
str_21104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 11), 'str', 'numpy.core')
str_21105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 25), 'str', 'flatiter')

# Obtaining an instance of the builtin type 'tuple' (line 83)
tuple_21106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 38), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 83)
# Adding element type (line 83)
str_21107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 38), 'str', 'coords')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 38), tuple_21106, str_21107)
# Adding element type (line 83)
str_21108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, (-1)), 'str', '\n    An N-dimensional tuple of current coordinates.\n\n    Examples\n    --------\n    >>> x = np.arange(6).reshape(2, 3)\n    >>> fl = x.flat\n    >>> fl.coords\n    (0, 0)\n    >>> fl.next()\n    0\n    >>> fl.coords\n    (0, 1)\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 38), tuple_21106, str_21108)

# Processing the call keyword arguments (line 83)
kwargs_21109 = {}
# Getting the type of 'add_newdoc' (line 83)
add_newdoc_21103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 83)
add_newdoc_call_result_21110 = invoke(stypy.reporting.localization.Localization(__file__, 83, 0), add_newdoc_21103, *[str_21104, str_21105, tuple_21106], **kwargs_21109)


# Call to add_newdoc(...): (line 102)
# Processing the call arguments (line 102)
str_21112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 11), 'str', 'numpy.core')
str_21113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 25), 'str', 'flatiter')

# Obtaining an instance of the builtin type 'tuple' (line 102)
tuple_21114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 38), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 102)
# Adding element type (line 102)
str_21115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 38), 'str', 'index')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 38), tuple_21114, str_21115)
# Adding element type (line 102)
str_21116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, (-1)), 'str', '\n    Current flat index into the array.\n\n    Examples\n    --------\n    >>> x = np.arange(6).reshape(2, 3)\n    >>> fl = x.flat\n    >>> fl.index\n    0\n    >>> fl.next()\n    0\n    >>> fl.index\n    1\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 38), tuple_21114, str_21116)

# Processing the call keyword arguments (line 102)
kwargs_21117 = {}
# Getting the type of 'add_newdoc' (line 102)
add_newdoc_21111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 102)
add_newdoc_call_result_21118 = invoke(stypy.reporting.localization.Localization(__file__, 102, 0), add_newdoc_21111, *[str_21112, str_21113, tuple_21114], **kwargs_21117)


# Call to add_newdoc(...): (line 121)
# Processing the call arguments (line 121)
str_21120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 11), 'str', 'numpy.core')
str_21121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 25), 'str', 'flatiter')

# Obtaining an instance of the builtin type 'tuple' (line 121)
tuple_21122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 38), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 121)
# Adding element type (line 121)
str_21123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 38), 'str', '__array__')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 38), tuple_21122, str_21123)
# Adding element type (line 121)
str_21124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, (-1)), 'str', '__array__(type=None) Get array from iterator\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 38), tuple_21122, str_21124)

# Processing the call keyword arguments (line 121)
kwargs_21125 = {}
# Getting the type of 'add_newdoc' (line 121)
add_newdoc_21119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 121)
add_newdoc_call_result_21126 = invoke(stypy.reporting.localization.Localization(__file__, 121, 0), add_newdoc_21119, *[str_21120, str_21121, tuple_21122], **kwargs_21125)


# Call to add_newdoc(...): (line 127)
# Processing the call arguments (line 127)
str_21128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 11), 'str', 'numpy.core')
str_21129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 25), 'str', 'flatiter')

# Obtaining an instance of the builtin type 'tuple' (line 127)
tuple_21130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 38), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 127)
# Adding element type (line 127)
str_21131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 38), 'str', 'copy')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 38), tuple_21130, str_21131)
# Adding element type (line 127)
str_21132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, (-1)), 'str', '\n    copy()\n\n    Get a copy of the iterator as a 1-D array.\n\n    Examples\n    --------\n    >>> x = np.arange(6).reshape(2, 3)\n    >>> x\n    array([[0, 1, 2],\n           [3, 4, 5]])\n    >>> fl = x.flat\n    >>> fl.copy()\n    array([0, 1, 2, 3, 4, 5])\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 38), tuple_21130, str_21132)

# Processing the call keyword arguments (line 127)
kwargs_21133 = {}
# Getting the type of 'add_newdoc' (line 127)
add_newdoc_21127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 127)
add_newdoc_call_result_21134 = invoke(stypy.reporting.localization.Localization(__file__, 127, 0), add_newdoc_21127, *[str_21128, str_21129, tuple_21130], **kwargs_21133)


# Call to add_newdoc(...): (line 152)
# Processing the call arguments (line 152)
str_21136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 11), 'str', 'numpy.core')
str_21137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 25), 'str', 'nditer')
str_21138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 377, (-1)), 'str', '\n    Efficient multi-dimensional iterator object to iterate over arrays.\n    To get started using this object, see the\n    :ref:`introductory guide to array iteration <arrays.nditer>`.\n\n    Parameters\n    ----------\n    op : ndarray or sequence of array_like\n        The array(s) to iterate over.\n    flags : sequence of str, optional\n        Flags to control the behavior of the iterator.\n\n          * "buffered" enables buffering when required.\n          * "c_index" causes a C-order index to be tracked.\n          * "f_index" causes a Fortran-order index to be tracked.\n          * "multi_index" causes a multi-index, or a tuple of indices\n            with one per iteration dimension, to be tracked.\n          * "common_dtype" causes all the operands to be converted to\n            a common data type, with copying or buffering as necessary.\n          * "delay_bufalloc" delays allocation of the buffers until\n            a reset() call is made. Allows "allocate" operands to\n            be initialized before their values are copied into the buffers.\n          * "external_loop" causes the `values` given to be\n            one-dimensional arrays with multiple values instead of\n            zero-dimensional arrays.\n          * "grow_inner" allows the `value` array sizes to be made\n            larger than the buffer size when both "buffered" and\n            "external_loop" is used.\n          * "ranged" allows the iterator to be restricted to a sub-range\n            of the iterindex values.\n          * "refs_ok" enables iteration of reference types, such as\n            object arrays.\n          * "reduce_ok" enables iteration of "readwrite" operands\n            which are broadcasted, also known as reduction operands.\n          * "zerosize_ok" allows `itersize` to be zero.\n    op_flags : list of list of str, optional\n        This is a list of flags for each operand. At minimum, one of\n        "readonly", "readwrite", or "writeonly" must be specified.\n\n          * "readonly" indicates the operand will only be read from.\n          * "readwrite" indicates the operand will be read from and written to.\n          * "writeonly" indicates the operand will only be written to.\n          * "no_broadcast" prevents the operand from being broadcasted.\n          * "contig" forces the operand data to be contiguous.\n          * "aligned" forces the operand data to be aligned.\n          * "nbo" forces the operand data to be in native byte order.\n          * "copy" allows a temporary read-only copy if required.\n          * "updateifcopy" allows a temporary read-write copy if required.\n          * "allocate" causes the array to be allocated if it is None\n            in the `op` parameter.\n          * "no_subtype" prevents an "allocate" operand from using a subtype.\n          * "arraymask" indicates that this operand is the mask to use\n            for selecting elements when writing to operands with the\n            \'writemasked\' flag set. The iterator does not enforce this,\n            but when writing from a buffer back to the array, it only\n            copies those elements indicated by this mask.\n          * \'writemasked\' indicates that only elements where the chosen\n            \'arraymask\' operand is True will be written to.\n    op_dtypes : dtype or tuple of dtype(s), optional\n        The required data type(s) of the operands. If copying or buffering\n        is enabled, the data will be converted to/from their original types.\n    order : {\'C\', \'F\', \'A\', \'K\'}, optional\n        Controls the iteration order. \'C\' means C order, \'F\' means\n        Fortran order, \'A\' means \'F\' order if all the arrays are Fortran\n        contiguous, \'C\' order otherwise, and \'K\' means as close to the\n        order the array elements appear in memory as possible. This also\n        affects the element memory order of "allocate" operands, as they\n        are allocated to be compatible with iteration order.\n        Default is \'K\'.\n    casting : {\'no\', \'equiv\', \'safe\', \'same_kind\', \'unsafe\'}, optional\n        Controls what kind of data casting may occur when making a copy\n        or buffering.  Setting this to \'unsafe\' is not recommended,\n        as it can adversely affect accumulations.\n\n          * \'no\' means the data types should not be cast at all.\n          * \'equiv\' means only byte-order changes are allowed.\n          * \'safe\' means only casts which can preserve values are allowed.\n          * \'same_kind\' means only safe casts or casts within a kind,\n            like float64 to float32, are allowed.\n          * \'unsafe\' means any data conversions may be done.\n    op_axes : list of list of ints, optional\n        If provided, is a list of ints or None for each operands.\n        The list of axes for an operand is a mapping from the dimensions\n        of the iterator to the dimensions of the operand. A value of\n        -1 can be placed for entries, causing that dimension to be\n        treated as "newaxis".\n    itershape : tuple of ints, optional\n        The desired shape of the iterator. This allows "allocate" operands\n        with a dimension mapped by op_axes not corresponding to a dimension\n        of a different operand to get a value not equal to 1 for that\n        dimension.\n    buffersize : int, optional\n        When buffering is enabled, controls the size of the temporary\n        buffers. Set to 0 for the default value.\n\n    Attributes\n    ----------\n    dtypes : tuple of dtype(s)\n        The data types of the values provided in `value`. This may be\n        different from the operand data types if buffering is enabled.\n    finished : bool\n        Whether the iteration over the operands is finished or not.\n    has_delayed_bufalloc : bool\n        If True, the iterator was created with the "delay_bufalloc" flag,\n        and no reset() function was called on it yet.\n    has_index : bool\n        If True, the iterator was created with either the "c_index" or\n        the "f_index" flag, and the property `index` can be used to\n        retrieve it.\n    has_multi_index : bool\n        If True, the iterator was created with the "multi_index" flag,\n        and the property `multi_index` can be used to retrieve it.\n    index :\n        When the "c_index" or "f_index" flag was used, this property\n        provides access to the index. Raises a ValueError if accessed\n        and `has_index` is False.\n    iterationneedsapi : bool\n        Whether iteration requires access to the Python API, for example\n        if one of the operands is an object array.\n    iterindex : int\n        An index which matches the order of iteration.\n    itersize : int\n        Size of the iterator.\n    itviews :\n        Structured view(s) of `operands` in memory, matching the reordered\n        and optimized iterator access pattern.\n    multi_index :\n        When the "multi_index" flag was used, this property\n        provides access to the index. Raises a ValueError if accessed\n        accessed and `has_multi_index` is False.\n    ndim : int\n        The iterator\'s dimension.\n    nop : int\n        The number of iterator operands.\n    operands : tuple of operand(s)\n        The array(s) to be iterated over.\n    shape : tuple of ints\n        Shape tuple, the shape of the iterator.\n    value :\n        Value of `operands` at current iteration. Normally, this is a\n        tuple of array scalars, but if the flag "external_loop" is used,\n        it is a tuple of one dimensional arrays.\n\n    Notes\n    -----\n    `nditer` supersedes `flatiter`.  The iterator implementation behind\n    `nditer` is also exposed by the Numpy C API.\n\n    The Python exposure supplies two iteration interfaces, one which follows\n    the Python iterator protocol, and another which mirrors the C-style\n    do-while pattern.  The native Python approach is better in most cases, but\n    if you need the iterator\'s coordinates or index, use the C-style pattern.\n\n    Examples\n    --------\n    Here is how we might write an ``iter_add`` function, using the\n    Python iterator protocol::\n\n        def iter_add_py(x, y, out=None):\n            addop = np.add\n            it = np.nditer([x, y, out], [],\n                        [[\'readonly\'], [\'readonly\'], [\'writeonly\',\'allocate\']])\n            for (a, b, c) in it:\n                addop(a, b, out=c)\n            return it.operands[2]\n\n    Here is the same function, but following the C-style pattern::\n\n        def iter_add(x, y, out=None):\n            addop = np.add\n\n            it = np.nditer([x, y, out], [],\n                        [[\'readonly\'], [\'readonly\'], [\'writeonly\',\'allocate\']])\n\n            while not it.finished:\n                addop(it[0], it[1], out=it[2])\n                it.iternext()\n\n            return it.operands[2]\n\n    Here is an example outer product function::\n\n        def outer_it(x, y, out=None):\n            mulop = np.multiply\n\n            it = np.nditer([x, y, out], [\'external_loop\'],\n                    [[\'readonly\'], [\'readonly\'], [\'writeonly\', \'allocate\']],\n                    op_axes=[range(x.ndim)+[-1]*y.ndim,\n                             [-1]*x.ndim+range(y.ndim),\n                             None])\n\n            for (a, b, c) in it:\n                mulop(a, b, out=c)\n\n            return it.operands[2]\n\n        >>> a = np.arange(2)+1\n        >>> b = np.arange(3)+1\n        >>> outer_it(a,b)\n        array([[1, 2, 3],\n               [2, 4, 6]])\n\n    Here is an example function which operates like a "lambda" ufunc::\n\n        def luf(lamdaexpr, *args, **kwargs):\n            "luf(lambdaexpr, op1, ..., opn, out=None, order=\'K\', casting=\'safe\', buffersize=0)"\n            nargs = len(args)\n            op = (kwargs.get(\'out\',None),) + args\n            it = np.nditer(op, [\'buffered\',\'external_loop\'],\n                    [[\'writeonly\',\'allocate\',\'no_broadcast\']] +\n                                    [[\'readonly\',\'nbo\',\'aligned\']]*nargs,\n                    order=kwargs.get(\'order\',\'K\'),\n                    casting=kwargs.get(\'casting\',\'safe\'),\n                    buffersize=kwargs.get(\'buffersize\',0))\n            while not it.finished:\n                it[0] = lamdaexpr(*it[1:])\n                it.iternext()\n            return it.operands[0]\n\n        >>> a = np.arange(5)\n        >>> b = np.ones(5)\n        >>> luf(lambda i,j:i*i + j/2, a, b)\n        array([  0.5,   1.5,   4.5,   9.5,  16.5])\n\n    ')
# Processing the call keyword arguments (line 152)
kwargs_21139 = {}
# Getting the type of 'add_newdoc' (line 152)
add_newdoc_21135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 152)
add_newdoc_call_result_21140 = invoke(stypy.reporting.localization.Localization(__file__, 152, 0), add_newdoc_21135, *[str_21136, str_21137, str_21138], **kwargs_21139)


# Call to add_newdoc(...): (line 381)
# Processing the call arguments (line 381)
str_21142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 11), 'str', 'numpy.core')
str_21143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 25), 'str', 'nditer')

# Obtaining an instance of the builtin type 'tuple' (line 381)
tuple_21144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 36), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 381)
# Adding element type (line 381)
str_21145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 36), 'str', 'copy')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 381, 36), tuple_21144, str_21145)
# Adding element type (line 381)
str_21146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, (-1)), 'str', '\n    copy()\n\n    Get a copy of the iterator in its current state.\n\n    Examples\n    --------\n    >>> x = np.arange(10)\n    >>> y = x + 1\n    >>> it = np.nditer([x, y])\n    >>> it.next()\n    (array(0), array(1))\n    >>> it2 = it.copy()\n    >>> it2.next()\n    (array(1), array(2))\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 381, 36), tuple_21144, str_21146)

# Processing the call keyword arguments (line 381)
kwargs_21147 = {}
# Getting the type of 'add_newdoc' (line 381)
add_newdoc_21141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 381)
add_newdoc_call_result_21148 = invoke(stypy.reporting.localization.Localization(__file__, 381, 0), add_newdoc_21141, *[str_21142, str_21143, tuple_21144], **kwargs_21147)


# Call to add_newdoc(...): (line 400)
# Processing the call arguments (line 400)
str_21150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 11), 'str', 'numpy.core')
str_21151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 25), 'str', 'nditer')

# Obtaining an instance of the builtin type 'tuple' (line 400)
tuple_21152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 36), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 400)
# Adding element type (line 400)
str_21153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 36), 'str', 'debug_print')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 400, 36), tuple_21152, str_21153)
# Adding element type (line 400)
str_21154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, (-1)), 'str', '\n    debug_print()\n\n    Print the current state of the `nditer` instance and debug info to stdout.\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 400, 36), tuple_21152, str_21154)

# Processing the call keyword arguments (line 400)
kwargs_21155 = {}
# Getting the type of 'add_newdoc' (line 400)
add_newdoc_21149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 400)
add_newdoc_call_result_21156 = invoke(stypy.reporting.localization.Localization(__file__, 400, 0), add_newdoc_21149, *[str_21150, str_21151, tuple_21152], **kwargs_21155)


# Call to add_newdoc(...): (line 408)
# Processing the call arguments (line 408)
str_21158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 408, 11), 'str', 'numpy.core')
str_21159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 408, 25), 'str', 'nditer')

# Obtaining an instance of the builtin type 'tuple' (line 408)
tuple_21160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 408, 36), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 408)
# Adding element type (line 408)
str_21161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 408, 36), 'str', 'enable_external_loop')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 408, 36), tuple_21160, str_21161)
# Adding element type (line 408)
str_21162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, (-1)), 'str', '\n    enable_external_loop()\n\n    When the "external_loop" was not used during construction, but\n    is desired, this modifies the iterator to behave as if the flag\n    was specified.\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 408, 36), tuple_21160, str_21162)

# Processing the call keyword arguments (line 408)
kwargs_21163 = {}
# Getting the type of 'add_newdoc' (line 408)
add_newdoc_21157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 408)
add_newdoc_call_result_21164 = invoke(stypy.reporting.localization.Localization(__file__, 408, 0), add_newdoc_21157, *[str_21158, str_21159, tuple_21160], **kwargs_21163)


# Call to add_newdoc(...): (line 418)
# Processing the call arguments (line 418)
str_21166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 11), 'str', 'numpy.core')
str_21167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 25), 'str', 'nditer')

# Obtaining an instance of the builtin type 'tuple' (line 418)
tuple_21168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 36), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 418)
# Adding element type (line 418)
str_21169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 36), 'str', 'iternext')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 418, 36), tuple_21168, str_21169)
# Adding element type (line 418)
str_21170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 431, (-1)), 'str', '\n    iternext()\n\n    Check whether iterations are left, and perform a single internal iteration\n    without returning the result.  Used in the C-style pattern do-while\n    pattern.  For an example, see `nditer`.\n\n    Returns\n    -------\n    iternext : bool\n        Whether or not there are iterations left.\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 418, 36), tuple_21168, str_21170)

# Processing the call keyword arguments (line 418)
kwargs_21171 = {}
# Getting the type of 'add_newdoc' (line 418)
add_newdoc_21165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 418)
add_newdoc_call_result_21172 = invoke(stypy.reporting.localization.Localization(__file__, 418, 0), add_newdoc_21165, *[str_21166, str_21167, tuple_21168], **kwargs_21171)


# Call to add_newdoc(...): (line 433)
# Processing the call arguments (line 433)
str_21174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 11), 'str', 'numpy.core')
str_21175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 25), 'str', 'nditer')

# Obtaining an instance of the builtin type 'tuple' (line 433)
tuple_21176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 36), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 433)
# Adding element type (line 433)
str_21177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 36), 'str', 'remove_axis')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 433, 36), tuple_21176, str_21177)
# Adding element type (line 433)
str_21178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 440, (-1)), 'str', '\n    remove_axis(i)\n\n    Removes axis `i` from the iterator. Requires that the flag "multi_index"\n    be enabled.\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 433, 36), tuple_21176, str_21178)

# Processing the call keyword arguments (line 433)
kwargs_21179 = {}
# Getting the type of 'add_newdoc' (line 433)
add_newdoc_21173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 433)
add_newdoc_call_result_21180 = invoke(stypy.reporting.localization.Localization(__file__, 433, 0), add_newdoc_21173, *[str_21174, str_21175, tuple_21176], **kwargs_21179)


# Call to add_newdoc(...): (line 442)
# Processing the call arguments (line 442)
str_21182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 442, 11), 'str', 'numpy.core')
str_21183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 442, 25), 'str', 'nditer')

# Obtaining an instance of the builtin type 'tuple' (line 442)
tuple_21184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 442, 36), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 442)
# Adding element type (line 442)
str_21185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 442, 36), 'str', 'remove_multi_index')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 442, 36), tuple_21184, str_21185)
# Adding element type (line 442)
str_21186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 449, (-1)), 'str', '\n    remove_multi_index()\n\n    When the "multi_index" flag was specified, this removes it, allowing\n    the internal iteration structure to be optimized further.\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 442, 36), tuple_21184, str_21186)

# Processing the call keyword arguments (line 442)
kwargs_21187 = {}
# Getting the type of 'add_newdoc' (line 442)
add_newdoc_21181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 442)
add_newdoc_call_result_21188 = invoke(stypy.reporting.localization.Localization(__file__, 442, 0), add_newdoc_21181, *[str_21182, str_21183, tuple_21184], **kwargs_21187)


# Call to add_newdoc(...): (line 451)
# Processing the call arguments (line 451)
str_21190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 451, 11), 'str', 'numpy.core')
str_21191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 451, 25), 'str', 'nditer')

# Obtaining an instance of the builtin type 'tuple' (line 451)
tuple_21192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 451, 36), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 451)
# Adding element type (line 451)
str_21193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 451, 36), 'str', 'reset')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 451, 36), tuple_21192, str_21193)
# Adding element type (line 451)
str_21194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 457, (-1)), 'str', '\n    reset()\n\n    Reset the iterator to its initial state.\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 451, 36), tuple_21192, str_21194)

# Processing the call keyword arguments (line 451)
kwargs_21195 = {}
# Getting the type of 'add_newdoc' (line 451)
add_newdoc_21189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 451)
add_newdoc_call_result_21196 = invoke(stypy.reporting.localization.Localization(__file__, 451, 0), add_newdoc_21189, *[str_21190, str_21191, tuple_21192], **kwargs_21195)


# Call to add_newdoc(...): (line 467)
# Processing the call arguments (line 467)
str_21198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 467, 11), 'str', 'numpy.core')
str_21199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 467, 25), 'str', 'broadcast')
str_21200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 506, (-1)), 'str', '\n    Produce an object that mimics broadcasting.\n\n    Parameters\n    ----------\n    in1, in2, ... : array_like\n        Input parameters.\n\n    Returns\n    -------\n    b : broadcast object\n        Broadcast the input parameters against one another, and\n        return an object that encapsulates the result.\n        Amongst others, it has ``shape`` and ``nd`` properties, and\n        may be used as an iterator.\n\n    Examples\n    --------\n    Manually adding two vectors, using broadcasting:\n\n    >>> x = np.array([[1], [2], [3]])\n    >>> y = np.array([4, 5, 6])\n    >>> b = np.broadcast(x, y)\n\n    >>> out = np.empty(b.shape)\n    >>> out.flat = [u+v for (u,v) in b]\n    >>> out\n    array([[ 5.,  6.,  7.],\n           [ 6.,  7.,  8.],\n           [ 7.,  8.,  9.]])\n\n    Compare against built-in broadcasting:\n\n    >>> x + y\n    array([[5, 6, 7],\n           [6, 7, 8],\n           [7, 8, 9]])\n\n    ')
# Processing the call keyword arguments (line 467)
kwargs_21201 = {}
# Getting the type of 'add_newdoc' (line 467)
add_newdoc_21197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 467)
add_newdoc_call_result_21202 = invoke(stypy.reporting.localization.Localization(__file__, 467, 0), add_newdoc_21197, *[str_21198, str_21199, str_21200], **kwargs_21201)


# Call to add_newdoc(...): (line 510)
# Processing the call arguments (line 510)
str_21204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 510, 11), 'str', 'numpy.core')
str_21205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 510, 25), 'str', 'broadcast')

# Obtaining an instance of the builtin type 'tuple' (line 510)
tuple_21206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 510, 39), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 510)
# Adding element type (line 510)
str_21207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 510, 39), 'str', 'index')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 510, 39), tuple_21206, str_21207)
# Adding element type (line 510)
str_21208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 526, (-1)), 'str', '\n    current index in broadcasted result\n\n    Examples\n    --------\n    >>> x = np.array([[1], [2], [3]])\n    >>> y = np.array([4, 5, 6])\n    >>> b = np.broadcast(x, y)\n    >>> b.index\n    0\n    >>> b.next(), b.next(), b.next()\n    ((1, 4), (1, 5), (1, 6))\n    >>> b.index\n    3\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 510, 39), tuple_21206, str_21208)

# Processing the call keyword arguments (line 510)
kwargs_21209 = {}
# Getting the type of 'add_newdoc' (line 510)
add_newdoc_21203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 510)
add_newdoc_call_result_21210 = invoke(stypy.reporting.localization.Localization(__file__, 510, 0), add_newdoc_21203, *[str_21204, str_21205, tuple_21206], **kwargs_21209)


# Call to add_newdoc(...): (line 528)
# Processing the call arguments (line 528)
str_21212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 528, 11), 'str', 'numpy.core')
str_21213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 528, 25), 'str', 'broadcast')

# Obtaining an instance of the builtin type 'tuple' (line 528)
tuple_21214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 528, 39), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 528)
# Adding element type (line 528)
str_21215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 528, 39), 'str', 'iters')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 528, 39), tuple_21214, str_21215)
# Adding element type (line 528)
str_21216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 548, (-1)), 'str', '\n    tuple of iterators along ``self``\'s "components."\n\n    Returns a tuple of `numpy.flatiter` objects, one for each "component"\n    of ``self``.\n\n    See Also\n    --------\n    numpy.flatiter\n\n    Examples\n    --------\n    >>> x = np.array([1, 2, 3])\n    >>> y = np.array([[4], [5], [6]])\n    >>> b = np.broadcast(x, y)\n    >>> row, col = b.iters\n    >>> row.next(), col.next()\n    (1, 4)\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 528, 39), tuple_21214, str_21216)

# Processing the call keyword arguments (line 528)
kwargs_21217 = {}
# Getting the type of 'add_newdoc' (line 528)
add_newdoc_21211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 528)
add_newdoc_call_result_21218 = invoke(stypy.reporting.localization.Localization(__file__, 528, 0), add_newdoc_21211, *[str_21212, str_21213, tuple_21214], **kwargs_21217)


# Call to add_newdoc(...): (line 550)
# Processing the call arguments (line 550)
str_21220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 550, 11), 'str', 'numpy.core')
str_21221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 550, 25), 'str', 'broadcast')

# Obtaining an instance of the builtin type 'tuple' (line 550)
tuple_21222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 550, 39), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 550)
# Adding element type (line 550)
str_21223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 550, 39), 'str', 'nd')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 550, 39), tuple_21222, str_21223)
# Adding element type (line 550)
str_21224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 562, (-1)), 'str', '\n    Number of dimensions of broadcasted result.\n\n    Examples\n    --------\n    >>> x = np.array([1, 2, 3])\n    >>> y = np.array([[4], [5], [6]])\n    >>> b = np.broadcast(x, y)\n    >>> b.nd\n    2\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 550, 39), tuple_21222, str_21224)

# Processing the call keyword arguments (line 550)
kwargs_21225 = {}
# Getting the type of 'add_newdoc' (line 550)
add_newdoc_21219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 550)
add_newdoc_call_result_21226 = invoke(stypy.reporting.localization.Localization(__file__, 550, 0), add_newdoc_21219, *[str_21220, str_21221, tuple_21222], **kwargs_21225)


# Call to add_newdoc(...): (line 564)
# Processing the call arguments (line 564)
str_21228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 564, 11), 'str', 'numpy.core')
str_21229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 564, 25), 'str', 'broadcast')

# Obtaining an instance of the builtin type 'tuple' (line 564)
tuple_21230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 564, 39), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 564)
# Adding element type (line 564)
str_21231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 564, 39), 'str', 'numiter')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 564, 39), tuple_21230, str_21231)
# Adding element type (line 564)
str_21232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 576, (-1)), 'str', '\n    Number of iterators possessed by the broadcasted result.\n\n    Examples\n    --------\n    >>> x = np.array([1, 2, 3])\n    >>> y = np.array([[4], [5], [6]])\n    >>> b = np.broadcast(x, y)\n    >>> b.numiter\n    2\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 564, 39), tuple_21230, str_21232)

# Processing the call keyword arguments (line 564)
kwargs_21233 = {}
# Getting the type of 'add_newdoc' (line 564)
add_newdoc_21227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 564)
add_newdoc_call_result_21234 = invoke(stypy.reporting.localization.Localization(__file__, 564, 0), add_newdoc_21227, *[str_21228, str_21229, tuple_21230], **kwargs_21233)


# Call to add_newdoc(...): (line 578)
# Processing the call arguments (line 578)
str_21236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 578, 11), 'str', 'numpy.core')
str_21237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 578, 25), 'str', 'broadcast')

# Obtaining an instance of the builtin type 'tuple' (line 578)
tuple_21238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 578, 39), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 578)
# Adding element type (line 578)
str_21239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 578, 39), 'str', 'shape')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 578, 39), tuple_21238, str_21239)
# Adding element type (line 578)
str_21240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 590, (-1)), 'str', '\n    Shape of broadcasted result.\n\n    Examples\n    --------\n    >>> x = np.array([1, 2, 3])\n    >>> y = np.array([[4], [5], [6]])\n    >>> b = np.broadcast(x, y)\n    >>> b.shape\n    (3, 3)\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 578, 39), tuple_21238, str_21240)

# Processing the call keyword arguments (line 578)
kwargs_21241 = {}
# Getting the type of 'add_newdoc' (line 578)
add_newdoc_21235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 578)
add_newdoc_call_result_21242 = invoke(stypy.reporting.localization.Localization(__file__, 578, 0), add_newdoc_21235, *[str_21236, str_21237, tuple_21238], **kwargs_21241)


# Call to add_newdoc(...): (line 592)
# Processing the call arguments (line 592)
str_21244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 592, 11), 'str', 'numpy.core')
str_21245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 592, 25), 'str', 'broadcast')

# Obtaining an instance of the builtin type 'tuple' (line 592)
tuple_21246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 592, 39), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 592)
# Adding element type (line 592)
str_21247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 592, 39), 'str', 'size')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 592, 39), tuple_21246, str_21247)
# Adding element type (line 592)
str_21248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 604, (-1)), 'str', '\n    Total size of broadcasted result.\n\n    Examples\n    --------\n    >>> x = np.array([1, 2, 3])\n    >>> y = np.array([[4], [5], [6]])\n    >>> b = np.broadcast(x, y)\n    >>> b.size\n    9\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 592, 39), tuple_21246, str_21248)

# Processing the call keyword arguments (line 592)
kwargs_21249 = {}
# Getting the type of 'add_newdoc' (line 592)
add_newdoc_21243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 592)
add_newdoc_call_result_21250 = invoke(stypy.reporting.localization.Localization(__file__, 592, 0), add_newdoc_21243, *[str_21244, str_21245, tuple_21246], **kwargs_21249)


# Call to add_newdoc(...): (line 606)
# Processing the call arguments (line 606)
str_21252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 606, 11), 'str', 'numpy.core')
str_21253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 606, 25), 'str', 'broadcast')

# Obtaining an instance of the builtin type 'tuple' (line 606)
tuple_21254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 606, 39), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 606)
# Adding element type (line 606)
str_21255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 606, 39), 'str', 'reset')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 606, 39), tuple_21254, str_21255)
# Adding element type (line 606)
str_21256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 635, (-1)), 'str', "\n    reset()\n\n    Reset the broadcasted result's iterator(s).\n\n    Parameters\n    ----------\n    None\n\n    Returns\n    -------\n    None\n\n    Examples\n    --------\n    >>> x = np.array([1, 2, 3])\n    >>> y = np.array([[4], [5], [6]]\n    >>> b = np.broadcast(x, y)\n    >>> b.index\n    0\n    >>> b.next(), b.next(), b.next()\n    ((1, 4), (2, 4), (3, 4))\n    >>> b.index\n    3\n    >>> b.reset()\n    >>> b.index\n    0\n\n    ")
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 606, 39), tuple_21254, str_21256)

# Processing the call keyword arguments (line 606)
kwargs_21257 = {}
# Getting the type of 'add_newdoc' (line 606)
add_newdoc_21251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 606, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 606)
add_newdoc_call_result_21258 = invoke(stypy.reporting.localization.Localization(__file__, 606, 0), add_newdoc_21251, *[str_21252, str_21253, tuple_21254], **kwargs_21257)


# Call to add_newdoc(...): (line 643)
# Processing the call arguments (line 643)
str_21260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 643, 11), 'str', 'numpy.core.multiarray')
str_21261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 643, 36), 'str', 'array')
str_21262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 733, (-1)), 'str', "\n    array(object, dtype=None, copy=True, order=None, subok=False, ndmin=0)\n\n    Create an array.\n\n    Parameters\n    ----------\n    object : array_like\n        An array, any object exposing the array interface, an\n        object whose __array__ method returns an array, or any\n        (nested) sequence.\n    dtype : data-type, optional\n        The desired data-type for the array.  If not given, then\n        the type will be determined as the minimum type required\n        to hold the objects in the sequence.  This argument can only\n        be used to 'upcast' the array.  For downcasting, use the\n        .astype(t) method.\n    copy : bool, optional\n        If true (default), then the object is copied.  Otherwise, a copy\n        will only be made if __array__ returns a copy, if obj is a\n        nested sequence, or if a copy is needed to satisfy any of the other\n        requirements (`dtype`, `order`, etc.).\n    order : {'C', 'F', 'A'}, optional\n        Specify the order of the array.  If order is 'C', then the array\n        will be in C-contiguous order (last-index varies the fastest).\n        If order is 'F', then the returned array will be in\n        Fortran-contiguous order (first-index varies the fastest).\n        If order is 'A' (default), then the returned array may be\n        in any order (either C-, Fortran-contiguous, or even discontiguous),\n        unless a copy is required, in which case it will be C-contiguous.\n    subok : bool, optional\n        If True, then sub-classes will be passed-through, otherwise\n        the returned array will be forced to be a base-class array (default).\n    ndmin : int, optional\n        Specifies the minimum number of dimensions that the resulting\n        array should have.  Ones will be pre-pended to the shape as\n        needed to meet this requirement.\n\n    Returns\n    -------\n    out : ndarray\n        An array object satisfying the specified requirements.\n\n    See Also\n    --------\n    empty, empty_like, zeros, zeros_like, ones, ones_like, fill\n\n    Examples\n    --------\n    >>> np.array([1, 2, 3])\n    array([1, 2, 3])\n\n    Upcasting:\n\n    >>> np.array([1, 2, 3.0])\n    array([ 1.,  2.,  3.])\n\n    More than one dimension:\n\n    >>> np.array([[1, 2], [3, 4]])\n    array([[1, 2],\n           [3, 4]])\n\n    Minimum dimensions 2:\n\n    >>> np.array([1, 2, 3], ndmin=2)\n    array([[1, 2, 3]])\n\n    Type provided:\n\n    >>> np.array([1, 2, 3], dtype=complex)\n    array([ 1.+0.j,  2.+0.j,  3.+0.j])\n\n    Data-type consisting of more than one element:\n\n    >>> x = np.array([(1,2),(3,4)],dtype=[('a','<i4'),('b','<i4')])\n    >>> x['a']\n    array([1, 3])\n\n    Creating an array from sub-classes:\n\n    >>> np.array(np.mat('1 2; 3 4'))\n    array([[1, 2],\n           [3, 4]])\n\n    >>> np.array(np.mat('1 2; 3 4'), subok=True)\n    matrix([[1, 2],\n            [3, 4]])\n\n    ")
# Processing the call keyword arguments (line 643)
kwargs_21263 = {}
# Getting the type of 'add_newdoc' (line 643)
add_newdoc_21259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 643, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 643)
add_newdoc_call_result_21264 = invoke(stypy.reporting.localization.Localization(__file__, 643, 0), add_newdoc_21259, *[str_21260, str_21261, str_21262], **kwargs_21263)


# Call to add_newdoc(...): (line 735)
# Processing the call arguments (line 735)
str_21266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 735, 11), 'str', 'numpy.core.multiarray')
str_21267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 735, 36), 'str', 'empty')
str_21268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 779, (-1)), 'str', "\n    empty(shape, dtype=float, order='C')\n\n    Return a new array of given shape and type, without initializing entries.\n\n    Parameters\n    ----------\n    shape : int or tuple of int\n        Shape of the empty array\n    dtype : data-type, optional\n        Desired output data-type.\n    order : {'C', 'F'}, optional\n        Whether to store multi-dimensional data in row-major\n        (C-style) or column-major (Fortran-style) order in\n        memory.\n\n    Returns\n    -------\n    out : ndarray\n        Array of uninitialized (arbitrary) data of the given shape, dtype, and\n        order.  Object arrays will be initialized to None.\n\n    See Also\n    --------\n    empty_like, zeros, ones\n\n    Notes\n    -----\n    `empty`, unlike `zeros`, does not set the array values to zero,\n    and may therefore be marginally faster.  On the other hand, it requires\n    the user to manually set all the values in the array, and should be\n    used with caution.\n\n    Examples\n    --------\n    >>> np.empty([2, 2])\n    array([[ -9.74499359e+001,   6.69583040e-309],\n           [  2.13182611e-314,   3.06959433e-309]])         #random\n\n    >>> np.empty([2, 2], dtype=int)\n    array([[-1073741821, -1067949133],\n           [  496041986,    19249760]])                     #random\n\n    ")
# Processing the call keyword arguments (line 735)
kwargs_21269 = {}
# Getting the type of 'add_newdoc' (line 735)
add_newdoc_21265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 735, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 735)
add_newdoc_call_result_21270 = invoke(stypy.reporting.localization.Localization(__file__, 735, 0), add_newdoc_21265, *[str_21266, str_21267, str_21268], **kwargs_21269)


# Call to add_newdoc(...): (line 781)
# Processing the call arguments (line 781)
str_21272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 781, 11), 'str', 'numpy.core.multiarray')
str_21273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 781, 36), 'str', 'empty_like')
str_21274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 839, (-1)), 'str', "\n    empty_like(a, dtype=None, order='K', subok=True)\n\n    Return a new array with the same shape and type as a given array.\n\n    Parameters\n    ----------\n    a : array_like\n        The shape and data-type of `a` define these same attributes of the\n        returned array.\n    dtype : data-type, optional\n        Overrides the data type of the result.\n\n        .. versionadded:: 1.6.0\n    order : {'C', 'F', 'A', or 'K'}, optional\n        Overrides the memory layout of the result. 'C' means C-order,\n        'F' means F-order, 'A' means 'F' if ``a`` is Fortran contiguous,\n        'C' otherwise. 'K' means match the layout of ``a`` as closely\n        as possible.\n\n        .. versionadded:: 1.6.0\n    subok : bool, optional.\n        If True, then the newly created array will use the sub-class\n        type of 'a', otherwise it will be a base-class array. Defaults\n        to True.\n\n    Returns\n    -------\n    out : ndarray\n        Array of uninitialized (arbitrary) data with the same\n        shape and type as `a`.\n\n    See Also\n    --------\n    ones_like : Return an array of ones with shape and type of input.\n    zeros_like : Return an array of zeros with shape and type of input.\n    empty : Return a new uninitialized array.\n    ones : Return a new array setting values to one.\n    zeros : Return a new array setting values to zero.\n\n    Notes\n    -----\n    This function does *not* initialize the returned array; to do that use\n    `zeros_like` or `ones_like` instead.  It may be marginally faster than\n    the functions that do set the array values.\n\n    Examples\n    --------\n    >>> a = ([1,2,3], [4,5,6])                         # a is array-like\n    >>> np.empty_like(a)\n    array([[-1073741821, -1073741821,           3],    #random\n           [          0,           0, -1073741821]])\n    >>> a = np.array([[1., 2., 3.],[4.,5.,6.]])\n    >>> np.empty_like(a)\n    array([[ -2.00000715e+000,   1.48219694e-323,  -2.00000572e+000],#random\n           [  4.38791518e-305,  -2.00000715e+000,   4.17269252e-309]])\n\n    ")
# Processing the call keyword arguments (line 781)
kwargs_21275 = {}
# Getting the type of 'add_newdoc' (line 781)
add_newdoc_21271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 781, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 781)
add_newdoc_call_result_21276 = invoke(stypy.reporting.localization.Localization(__file__, 781, 0), add_newdoc_21271, *[str_21272, str_21273, str_21274], **kwargs_21275)


# Call to add_newdoc(...): (line 842)
# Processing the call arguments (line 842)
str_21278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 842, 11), 'str', 'numpy.core.multiarray')
str_21279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 842, 36), 'str', 'scalar')
str_21280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 854, (-1)), 'str', '\n    scalar(dtype, obj)\n\n    Return a new scalar array of the given type initialized with obj.\n\n    This function is meant mainly for pickle support. `dtype` must be a\n    valid data-type descriptor. If `dtype` corresponds to an object\n    descriptor, then `obj` can be any object, otherwise `obj` must be a\n    string. If `obj` is not given, it will be interpreted as None for object\n    type and as zeros for all other types.\n\n    ')
# Processing the call keyword arguments (line 842)
kwargs_21281 = {}
# Getting the type of 'add_newdoc' (line 842)
add_newdoc_21277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 842, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 842)
add_newdoc_call_result_21282 = invoke(stypy.reporting.localization.Localization(__file__, 842, 0), add_newdoc_21277, *[str_21278, str_21279, str_21280], **kwargs_21281)


# Call to add_newdoc(...): (line 856)
# Processing the call arguments (line 856)
str_21284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 856, 11), 'str', 'numpy.core.multiarray')
str_21285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 856, 36), 'str', 'zeros')
str_21286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 907, (-1)), 'str', "\n    zeros(shape, dtype=float, order='C')\n\n    Return a new array of given shape and type, filled with zeros.\n\n    Parameters\n    ----------\n    shape : int or sequence of ints\n        Shape of the new array, e.g., ``(2, 3)`` or ``2``.\n    dtype : data-type, optional\n        The desired data-type for the array, e.g., `numpy.int8`.  Default is\n        `numpy.float64`.\n    order : {'C', 'F'}, optional\n        Whether to store multidimensional data in C- or Fortran-contiguous\n        (row- or column-wise) order in memory.\n\n    Returns\n    -------\n    out : ndarray\n        Array of zeros with the given shape, dtype, and order.\n\n    See Also\n    --------\n    zeros_like : Return an array of zeros with shape and type of input.\n    ones_like : Return an array of ones with shape and type of input.\n    empty_like : Return an empty array with shape and type of input.\n    ones : Return a new array setting values to one.\n    empty : Return a new uninitialized array.\n\n    Examples\n    --------\n    >>> np.zeros(5)\n    array([ 0.,  0.,  0.,  0.,  0.])\n\n    >>> np.zeros((5,), dtype=np.int)\n    array([0, 0, 0, 0, 0])\n\n    >>> np.zeros((2, 1))\n    array([[ 0.],\n           [ 0.]])\n\n    >>> s = (2,2)\n    >>> np.zeros(s)\n    array([[ 0.,  0.],\n           [ 0.,  0.]])\n\n    >>> np.zeros((2,), dtype=[('x', 'i4'), ('y', 'i4')]) # custom dtype\n    array([(0, 0), (0, 0)],\n          dtype=[('x', '<i4'), ('y', '<i4')])\n\n    ")
# Processing the call keyword arguments (line 856)
kwargs_21287 = {}
# Getting the type of 'add_newdoc' (line 856)
add_newdoc_21283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 856, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 856)
add_newdoc_call_result_21288 = invoke(stypy.reporting.localization.Localization(__file__, 856, 0), add_newdoc_21283, *[str_21284, str_21285, str_21286], **kwargs_21287)


# Call to add_newdoc(...): (line 909)
# Processing the call arguments (line 909)
str_21290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 909, 11), 'str', 'numpy.core.multiarray')
str_21291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 909, 36), 'str', 'count_nonzero')
str_21292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 935, (-1)), 'str', '\n    count_nonzero(a)\n\n    Counts the number of non-zero values in the array ``a``.\n\n    Parameters\n    ----------\n    a : array_like\n        The array for which to count non-zeros.\n\n    Returns\n    -------\n    count : int or array of int\n        Number of non-zero values in the array.\n\n    See Also\n    --------\n    nonzero : Return the coordinates of all the non-zero values.\n\n    Examples\n    --------\n    >>> np.count_nonzero(np.eye(4))\n    4\n    >>> np.count_nonzero([[0,1,7,0,0],[3,0,0,2,19]])\n    5\n    ')
# Processing the call keyword arguments (line 909)
kwargs_21293 = {}
# Getting the type of 'add_newdoc' (line 909)
add_newdoc_21289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 909, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 909)
add_newdoc_call_result_21294 = invoke(stypy.reporting.localization.Localization(__file__, 909, 0), add_newdoc_21289, *[str_21290, str_21291, str_21292], **kwargs_21293)


# Call to add_newdoc(...): (line 937)
# Processing the call arguments (line 937)
str_21296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 937, 11), 'str', 'numpy.core.multiarray')
str_21297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 937, 36), 'str', 'set_typeDict')
str_21298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 943, (-1)), 'str', 'set_typeDict(dict)\n\n    Set the internal dictionary that can look up an array type using a\n    registered code.\n\n    ')
# Processing the call keyword arguments (line 937)
kwargs_21299 = {}
# Getting the type of 'add_newdoc' (line 937)
add_newdoc_21295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 937, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 937)
add_newdoc_call_result_21300 = invoke(stypy.reporting.localization.Localization(__file__, 937, 0), add_newdoc_21295, *[str_21296, str_21297, str_21298], **kwargs_21299)


# Call to add_newdoc(...): (line 945)
# Processing the call arguments (line 945)
str_21302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 945, 11), 'str', 'numpy.core.multiarray')
str_21303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 945, 36), 'str', 'fromstring')
str_21304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 995, (-1)), 'str', "\n    fromstring(string, dtype=float, count=-1, sep='')\n\n    A new 1-D array initialized from raw binary or text data in a string.\n\n    Parameters\n    ----------\n    string : str\n        A string containing the data.\n    dtype : data-type, optional\n        The data type of the array; default: float.  For binary input data,\n        the data must be in exactly this format.\n    count : int, optional\n        Read this number of `dtype` elements from the data.  If this is\n        negative (the default), the count will be determined from the\n        length of the data.\n    sep : str, optional\n        If not provided or, equivalently, the empty string, the data will\n        be interpreted as binary data; otherwise, as ASCII text with\n        decimal numbers.  Also in this latter case, this argument is\n        interpreted as the string separating numbers in the data; extra\n        whitespace between elements is also ignored.\n\n    Returns\n    -------\n    arr : ndarray\n        The constructed array.\n\n    Raises\n    ------\n    ValueError\n        If the string is not the correct size to satisfy the requested\n        `dtype` and `count`.\n\n    See Also\n    --------\n    frombuffer, fromfile, fromiter\n\n    Examples\n    --------\n    >>> np.fromstring('\\x01\\x02', dtype=np.uint8)\n    array([1, 2], dtype=uint8)\n    >>> np.fromstring('1 2', dtype=int, sep=' ')\n    array([1, 2])\n    >>> np.fromstring('1, 2', dtype=int, sep=',')\n    array([1, 2])\n    >>> np.fromstring('\\x01\\x02\\x03\\x04\\x05', dtype=np.uint8, count=3)\n    array([1, 2, 3], dtype=uint8)\n\n    ")
# Processing the call keyword arguments (line 945)
kwargs_21305 = {}
# Getting the type of 'add_newdoc' (line 945)
add_newdoc_21301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 945, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 945)
add_newdoc_call_result_21306 = invoke(stypy.reporting.localization.Localization(__file__, 945, 0), add_newdoc_21301, *[str_21302, str_21303, str_21304], **kwargs_21305)


# Call to add_newdoc(...): (line 997)
# Processing the call arguments (line 997)
str_21308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 997, 11), 'str', 'numpy.core.multiarray')
str_21309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 997, 36), 'str', 'fromiter')
str_21310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1029, (-1)), 'str', '\n    fromiter(iterable, dtype, count=-1)\n\n    Create a new 1-dimensional array from an iterable object.\n\n    Parameters\n    ----------\n    iterable : iterable object\n        An iterable object providing data for the array.\n    dtype : data-type\n        The data-type of the returned array.\n    count : int, optional\n        The number of items to read from *iterable*.  The default is -1,\n        which means all data is read.\n\n    Returns\n    -------\n    out : ndarray\n        The output array.\n\n    Notes\n    -----\n    Specify `count` to improve performance.  It allows ``fromiter`` to\n    pre-allocate the output array, instead of resizing it on demand.\n\n    Examples\n    --------\n    >>> iterable = (x*x for x in range(5))\n    >>> np.fromiter(iterable, np.float)\n    array([  0.,   1.,   4.,   9.,  16.])\n\n    ')
# Processing the call keyword arguments (line 997)
kwargs_21311 = {}
# Getting the type of 'add_newdoc' (line 997)
add_newdoc_21307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 997, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 997)
add_newdoc_call_result_21312 = invoke(stypy.reporting.localization.Localization(__file__, 997, 0), add_newdoc_21307, *[str_21308, str_21309, str_21310], **kwargs_21311)


# Call to add_newdoc(...): (line 1031)
# Processing the call arguments (line 1031)
str_21314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1031, 11), 'str', 'numpy.core.multiarray')
str_21315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1031, 36), 'str', 'fromfile')
str_21316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1104, (-1)), 'str', '\n    fromfile(file, dtype=float, count=-1, sep=\'\')\n\n    Construct an array from data in a text or binary file.\n\n    A highly efficient way of reading binary data with a known data-type,\n    as well as parsing simply formatted text files.  Data written using the\n    `tofile` method can be read using this function.\n\n    Parameters\n    ----------\n    file : file or str\n        Open file object or filename.\n    dtype : data-type\n        Data type of the returned array.\n        For binary files, it is used to determine the size and byte-order\n        of the items in the file.\n    count : int\n        Number of items to read. ``-1`` means all items (i.e., the complete\n        file).\n    sep : str\n        Separator between items if file is a text file.\n        Empty ("") separator means the file should be treated as binary.\n        Spaces (" ") in the separator match zero or more whitespace characters.\n        A separator consisting only of spaces must match at least one\n        whitespace.\n\n    See also\n    --------\n    load, save\n    ndarray.tofile\n    loadtxt : More flexible way of loading data from a text file.\n\n    Notes\n    -----\n    Do not rely on the combination of `tofile` and `fromfile` for\n    data storage, as the binary files generated are are not platform\n    independent.  In particular, no byte-order or data-type information is\n    saved.  Data can be stored in the platform independent ``.npy`` format\n    using `save` and `load` instead.\n\n    Examples\n    --------\n    Construct an ndarray:\n\n    >>> dt = np.dtype([(\'time\', [(\'min\', int), (\'sec\', int)]),\n    ...                (\'temp\', float)])\n    >>> x = np.zeros((1,), dtype=dt)\n    >>> x[\'time\'][\'min\'] = 10; x[\'temp\'] = 98.25\n    >>> x\n    array([((10, 0), 98.25)],\n          dtype=[(\'time\', [(\'min\', \'<i4\'), (\'sec\', \'<i4\')]), (\'temp\', \'<f8\')])\n\n    Save the raw data to disk:\n\n    >>> import os\n    >>> fname = os.tmpnam()\n    >>> x.tofile(fname)\n\n    Read the raw data from disk:\n\n    >>> np.fromfile(fname, dtype=dt)\n    array([((10, 0), 98.25)],\n          dtype=[(\'time\', [(\'min\', \'<i4\'), (\'sec\', \'<i4\')]), (\'temp\', \'<f8\')])\n\n    The recommended way to store and load data:\n\n    >>> np.save(fname, x)\n    >>> np.load(fname + \'.npy\')\n    array([((10, 0), 98.25)],\n          dtype=[(\'time\', [(\'min\', \'<i4\'), (\'sec\', \'<i4\')]), (\'temp\', \'<f8\')])\n\n    ')
# Processing the call keyword arguments (line 1031)
kwargs_21317 = {}
# Getting the type of 'add_newdoc' (line 1031)
add_newdoc_21313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1031, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 1031)
add_newdoc_call_result_21318 = invoke(stypy.reporting.localization.Localization(__file__, 1031, 0), add_newdoc_21313, *[str_21314, str_21315, str_21316], **kwargs_21317)


# Call to add_newdoc(...): (line 1106)
# Processing the call arguments (line 1106)
str_21320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1106, 11), 'str', 'numpy.core.multiarray')
str_21321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1106, 36), 'str', 'frombuffer')
str_21322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1142, (-1)), 'str', "\n    frombuffer(buffer, dtype=float, count=-1, offset=0)\n\n    Interpret a buffer as a 1-dimensional array.\n\n    Parameters\n    ----------\n    buffer : buffer_like\n        An object that exposes the buffer interface.\n    dtype : data-type, optional\n        Data-type of the returned array; default: float.\n    count : int, optional\n        Number of items to read. ``-1`` means all data in the buffer.\n    offset : int, optional\n        Start reading the buffer from this offset; default: 0.\n\n    Notes\n    -----\n    If the buffer has data that is not in machine byte-order, this should\n    be specified as part of the data-type, e.g.::\n\n      >>> dt = np.dtype(int)\n      >>> dt = dt.newbyteorder('>')\n      >>> np.frombuffer(buf, dtype=dt)\n\n    The data of the resulting array will not be byteswapped, but will be\n    interpreted correctly.\n\n    Examples\n    --------\n    >>> s = 'hello world'\n    >>> np.frombuffer(s, dtype='S1', count=5, offset=6)\n    array(['w', 'o', 'r', 'l', 'd'],\n          dtype='|S1')\n\n    ")
# Processing the call keyword arguments (line 1106)
kwargs_21323 = {}
# Getting the type of 'add_newdoc' (line 1106)
add_newdoc_21319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1106, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 1106)
add_newdoc_call_result_21324 = invoke(stypy.reporting.localization.Localization(__file__, 1106, 0), add_newdoc_21319, *[str_21320, str_21321, str_21322], **kwargs_21323)


# Call to add_newdoc(...): (line 1144)
# Processing the call arguments (line 1144)
str_21326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1144, 11), 'str', 'numpy.core.multiarray')
str_21327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1144, 36), 'str', 'concatenate')
str_21328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1217, (-1)), 'str', '\n    concatenate((a1, a2, ...), axis=0)\n\n    Join a sequence of arrays along an existing axis.\n\n    Parameters\n    ----------\n    a1, a2, ... : sequence of array_like\n        The arrays must have the same shape, except in the dimension\n        corresponding to `axis` (the first, by default).\n    axis : int, optional\n        The axis along which the arrays will be joined.  Default is 0.\n\n    Returns\n    -------\n    res : ndarray\n        The concatenated array.\n\n    See Also\n    --------\n    ma.concatenate : Concatenate function that preserves input masks.\n    array_split : Split an array into multiple sub-arrays of equal or\n                  near-equal size.\n    split : Split array into a list of multiple sub-arrays of equal size.\n    hsplit : Split array into multiple sub-arrays horizontally (column wise)\n    vsplit : Split array into multiple sub-arrays vertically (row wise)\n    dsplit : Split array into multiple sub-arrays along the 3rd axis (depth).\n    stack : Stack a sequence of arrays along a new axis.\n    hstack : Stack arrays in sequence horizontally (column wise)\n    vstack : Stack arrays in sequence vertically (row wise)\n    dstack : Stack arrays in sequence depth wise (along third dimension)\n\n    Notes\n    -----\n    When one or more of the arrays to be concatenated is a MaskedArray,\n    this function will return a MaskedArray object instead of an ndarray,\n    but the input masks are *not* preserved. In cases where a MaskedArray\n    is expected as input, use the ma.concatenate function from the masked\n    array module instead.\n\n    Examples\n    --------\n    >>> a = np.array([[1, 2], [3, 4]])\n    >>> b = np.array([[5, 6]])\n    >>> np.concatenate((a, b), axis=0)\n    array([[1, 2],\n           [3, 4],\n           [5, 6]])\n    >>> np.concatenate((a, b.T), axis=1)\n    array([[1, 2, 5],\n           [3, 4, 6]])\n\n    This function will not preserve masking of MaskedArray inputs.\n\n    >>> a = np.ma.arange(3)\n    >>> a[1] = np.ma.masked\n    >>> b = np.arange(2, 5)\n    >>> a\n    masked_array(data = [0 -- 2],\n                 mask = [False  True False],\n           fill_value = 999999)\n    >>> b\n    array([2, 3, 4])\n    >>> np.concatenate([a, b])\n    masked_array(data = [0 1 2 2 3 4],\n                 mask = False,\n           fill_value = 999999)\n    >>> np.ma.concatenate([a, b])\n    masked_array(data = [0 -- 2 2 3 4],\n                 mask = [False  True False False False False],\n           fill_value = 999999)\n\n    ')
# Processing the call keyword arguments (line 1144)
kwargs_21329 = {}
# Getting the type of 'add_newdoc' (line 1144)
add_newdoc_21325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1144, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 1144)
add_newdoc_call_result_21330 = invoke(stypy.reporting.localization.Localization(__file__, 1144, 0), add_newdoc_21325, *[str_21326, str_21327, str_21328], **kwargs_21329)


# Call to add_newdoc(...): (line 1219)
# Processing the call arguments (line 1219)
str_21332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1219, 11), 'str', 'numpy.core')
str_21333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1219, 25), 'str', 'inner')
str_21334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1291, (-1)), 'str', '\n    inner(a, b)\n\n    Inner product of two arrays.\n\n    Ordinary inner product of vectors for 1-D arrays (without complex\n    conjugation), in higher dimensions a sum product over the last axes.\n\n    Parameters\n    ----------\n    a, b : array_like\n        If `a` and `b` are nonscalar, their last dimensions must match.\n\n    Returns\n    -------\n    out : ndarray\n        `out.shape = a.shape[:-1] + b.shape[:-1]`\n\n    Raises\n    ------\n    ValueError\n        If the last dimension of `a` and `b` has different size.\n\n    See Also\n    --------\n    tensordot : Sum products over arbitrary axes.\n    dot : Generalised matrix product, using second last dimension of `b`.\n    einsum : Einstein summation convention.\n\n    Notes\n    -----\n    For vectors (1-D arrays) it computes the ordinary inner-product::\n\n        np.inner(a, b) = sum(a[:]*b[:])\n\n    More generally, if `ndim(a) = r > 0` and `ndim(b) = s > 0`::\n\n        np.inner(a, b) = np.tensordot(a, b, axes=(-1,-1))\n\n    or explicitly::\n\n        np.inner(a, b)[i0,...,ir-1,j0,...,js-1]\n             = sum(a[i0,...,ir-1,:]*b[j0,...,js-1,:])\n\n    In addition `a` or `b` may be scalars, in which case::\n\n       np.inner(a,b) = a*b\n\n    Examples\n    --------\n    Ordinary inner product for vectors:\n\n    >>> a = np.array([1,2,3])\n    >>> b = np.array([0,1,0])\n    >>> np.inner(a, b)\n    2\n\n    A multidimensional example:\n\n    >>> a = np.arange(24).reshape((2,3,4))\n    >>> b = np.arange(4)\n    >>> np.inner(a, b)\n    array([[ 14,  38,  62],\n           [ 86, 110, 134]])\n\n    An example where `b` is a scalar:\n\n    >>> np.inner(np.eye(2), 7)\n    array([[ 7.,  0.],\n           [ 0.,  7.]])\n\n    ')
# Processing the call keyword arguments (line 1219)
kwargs_21335 = {}
# Getting the type of 'add_newdoc' (line 1219)
add_newdoc_21331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1219, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 1219)
add_newdoc_call_result_21336 = invoke(stypy.reporting.localization.Localization(__file__, 1219, 0), add_newdoc_21331, *[str_21332, str_21333, str_21334], **kwargs_21335)


# Call to add_newdoc(...): (line 1293)
# Processing the call arguments (line 1293)
str_21338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1293, 11), 'str', 'numpy.core')
str_21339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1293, 25), 'str', 'fastCopyAndTranspose')
str_21340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1294, 4), 'str', '_fastCopyAndTranspose(a)')
# Processing the call keyword arguments (line 1293)
kwargs_21341 = {}
# Getting the type of 'add_newdoc' (line 1293)
add_newdoc_21337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1293, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 1293)
add_newdoc_call_result_21342 = invoke(stypy.reporting.localization.Localization(__file__, 1293, 0), add_newdoc_21337, *[str_21338, str_21339, str_21340], **kwargs_21341)


# Call to add_newdoc(...): (line 1296)
# Processing the call arguments (line 1296)
str_21344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1296, 11), 'str', 'numpy.core.multiarray')
str_21345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1296, 36), 'str', 'correlate')
str_21346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1297, 4), 'str', 'cross_correlate(a,v, mode=0)')
# Processing the call keyword arguments (line 1296)
kwargs_21347 = {}
# Getting the type of 'add_newdoc' (line 1296)
add_newdoc_21343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1296, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 1296)
add_newdoc_call_result_21348 = invoke(stypy.reporting.localization.Localization(__file__, 1296, 0), add_newdoc_21343, *[str_21344, str_21345, str_21346], **kwargs_21347)


# Call to add_newdoc(...): (line 1299)
# Processing the call arguments (line 1299)
str_21350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1299, 11), 'str', 'numpy.core.multiarray')
str_21351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1299, 36), 'str', 'arange')
str_21352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1358, (-1)), 'str', '\n    arange([start,] stop[, step,], dtype=None)\n\n    Return evenly spaced values within a given interval.\n\n    Values are generated within the half-open interval ``[start, stop)``\n    (in other words, the interval including `start` but excluding `stop`).\n    For integer arguments the function is equivalent to the Python built-in\n    `range <http://docs.python.org/lib/built-in-funcs.html>`_ function,\n    but returns an ndarray rather than a list.\n\n    When using a non-integer step, such as 0.1, the results will often not\n    be consistent.  It is better to use ``linspace`` for these cases.\n\n    Parameters\n    ----------\n    start : number, optional\n        Start of interval.  The interval includes this value.  The default\n        start value is 0.\n    stop : number\n        End of interval.  The interval does not include this value, except\n        in some cases where `step` is not an integer and floating point\n        round-off affects the length of `out`.\n    step : number, optional\n        Spacing between values.  For any output `out`, this is the distance\n        between two adjacent values, ``out[i+1] - out[i]``.  The default\n        step size is 1.  If `step` is specified, `start` must also be given.\n    dtype : dtype\n        The type of the output array.  If `dtype` is not given, infer the data\n        type from the other input arguments.\n\n    Returns\n    -------\n    arange : ndarray\n        Array of evenly spaced values.\n\n        For floating point arguments, the length of the result is\n        ``ceil((stop - start)/step)``.  Because of floating point overflow,\n        this rule may result in the last element of `out` being greater\n        than `stop`.\n\n    See Also\n    --------\n    linspace : Evenly spaced numbers with careful handling of endpoints.\n    ogrid: Arrays of evenly spaced numbers in N-dimensions.\n    mgrid: Grid-shaped arrays of evenly spaced numbers in N-dimensions.\n\n    Examples\n    --------\n    >>> np.arange(3)\n    array([0, 1, 2])\n    >>> np.arange(3.0)\n    array([ 0.,  1.,  2.])\n    >>> np.arange(3,7)\n    array([3, 4, 5, 6])\n    >>> np.arange(3,7,2)\n    array([3, 5])\n\n    ')
# Processing the call keyword arguments (line 1299)
kwargs_21353 = {}
# Getting the type of 'add_newdoc' (line 1299)
add_newdoc_21349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1299, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 1299)
add_newdoc_call_result_21354 = invoke(stypy.reporting.localization.Localization(__file__, 1299, 0), add_newdoc_21349, *[str_21350, str_21351, str_21352], **kwargs_21353)


# Call to add_newdoc(...): (line 1360)
# Processing the call arguments (line 1360)
str_21356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1360, 11), 'str', 'numpy.core.multiarray')
str_21357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1360, 36), 'str', '_get_ndarray_c_version')
str_21358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1365, (-1)), 'str', '_get_ndarray_c_version()\n\n    Return the compile time NDARRAY_VERSION number.\n\n    ')
# Processing the call keyword arguments (line 1360)
kwargs_21359 = {}
# Getting the type of 'add_newdoc' (line 1360)
add_newdoc_21355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1360, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 1360)
add_newdoc_call_result_21360 = invoke(stypy.reporting.localization.Localization(__file__, 1360, 0), add_newdoc_21355, *[str_21356, str_21357, str_21358], **kwargs_21359)


# Call to add_newdoc(...): (line 1367)
# Processing the call arguments (line 1367)
str_21362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1367, 11), 'str', 'numpy.core.multiarray')
str_21363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1367, 36), 'str', '_reconstruct')
str_21364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1372, (-1)), 'str', '_reconstruct(subtype, shape, dtype)\n\n    Construct an empty array. Used by Pickles.\n\n    ')
# Processing the call keyword arguments (line 1367)
kwargs_21365 = {}
# Getting the type of 'add_newdoc' (line 1367)
add_newdoc_21361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1367, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 1367)
add_newdoc_call_result_21366 = invoke(stypy.reporting.localization.Localization(__file__, 1367, 0), add_newdoc_21361, *[str_21362, str_21363, str_21364], **kwargs_21365)


# Call to add_newdoc(...): (line 1375)
# Processing the call arguments (line 1375)
str_21368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1375, 11), 'str', 'numpy.core.multiarray')
str_21369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1375, 36), 'str', 'set_string_function')
str_21370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1381, (-1)), 'str', '\n    set_string_function(f, repr=1)\n\n    Internal method to set a function to be used when pretty printing arrays.\n\n    ')
# Processing the call keyword arguments (line 1375)
kwargs_21371 = {}
# Getting the type of 'add_newdoc' (line 1375)
add_newdoc_21367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1375, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 1375)
add_newdoc_call_result_21372 = invoke(stypy.reporting.localization.Localization(__file__, 1375, 0), add_newdoc_21367, *[str_21368, str_21369, str_21370], **kwargs_21371)


# Call to add_newdoc(...): (line 1383)
# Processing the call arguments (line 1383)
str_21374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1383, 11), 'str', 'numpy.core.multiarray')
str_21375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1383, 36), 'str', 'set_numeric_ops')
str_21376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1425, (-1)), 'str', '\n    set_numeric_ops(op1=func1, op2=func2, ...)\n\n    Set numerical operators for array objects.\n\n    Parameters\n    ----------\n    op1, op2, ... : callable\n        Each ``op = func`` pair describes an operator to be replaced.\n        For example, ``add = lambda x, y: np.add(x, y) % 5`` would replace\n        addition by modulus 5 addition.\n\n    Returns\n    -------\n    saved_ops : list of callables\n        A list of all operators, stored before making replacements.\n\n    Notes\n    -----\n    .. WARNING::\n       Use with care!  Incorrect usage may lead to memory errors.\n\n    A function replacing an operator cannot make use of that operator.\n    For example, when replacing add, you may not use ``+``.  Instead,\n    directly call ufuncs.\n\n    Examples\n    --------\n    >>> def add_mod5(x, y):\n    ...     return np.add(x, y) % 5\n    ...\n    >>> old_funcs = np.set_numeric_ops(add=add_mod5)\n\n    >>> x = np.arange(12).reshape((3, 4))\n    >>> x + x\n    array([[0, 2, 4, 1],\n           [3, 0, 2, 4],\n           [1, 3, 0, 2]])\n\n    >>> ignore = np.set_numeric_ops(**old_funcs) # restore operators\n\n    ')
# Processing the call keyword arguments (line 1383)
kwargs_21377 = {}
# Getting the type of 'add_newdoc' (line 1383)
add_newdoc_21373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1383, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 1383)
add_newdoc_call_result_21378 = invoke(stypy.reporting.localization.Localization(__file__, 1383, 0), add_newdoc_21373, *[str_21374, str_21375, str_21376], **kwargs_21377)


# Call to add_newdoc(...): (line 1427)
# Processing the call arguments (line 1427)
str_21380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1427, 11), 'str', 'numpy.core.multiarray')
str_21381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1427, 36), 'str', 'where')
str_21382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1496, (-1)), 'str', '\n    where(condition, [x, y])\n\n    Return elements, either from `x` or `y`, depending on `condition`.\n\n    If only `condition` is given, return ``condition.nonzero()``.\n\n    Parameters\n    ----------\n    condition : array_like, bool\n        When True, yield `x`, otherwise yield `y`.\n    x, y : array_like, optional\n        Values from which to choose. `x` and `y` need to have the same\n        shape as `condition`.\n\n    Returns\n    -------\n    out : ndarray or tuple of ndarrays\n        If both `x` and `y` are specified, the output array contains\n        elements of `x` where `condition` is True, and elements from\n        `y` elsewhere.\n\n        If only `condition` is given, return the tuple\n        ``condition.nonzero()``, the indices where `condition` is True.\n\n    See Also\n    --------\n    nonzero, choose\n\n    Notes\n    -----\n    If `x` and `y` are given and input arrays are 1-D, `where` is\n    equivalent to::\n\n        [xv if c else yv for (c,xv,yv) in zip(condition,x,y)]\n\n    Examples\n    --------\n    >>> np.where([[True, False], [True, True]],\n    ...          [[1, 2], [3, 4]],\n    ...          [[9, 8], [7, 6]])\n    array([[1, 8],\n           [3, 4]])\n\n    >>> np.where([[0, 1], [1, 0]])\n    (array([0, 1]), array([1, 0]))\n\n    >>> x = np.arange(9.).reshape(3, 3)\n    >>> np.where( x > 5 )\n    (array([2, 2, 2]), array([0, 1, 2]))\n    >>> x[np.where( x > 3.0 )]               # Note: result is 1D.\n    array([ 4.,  5.,  6.,  7.,  8.])\n    >>> np.where(x < 5, x, -1)               # Note: broadcasting.\n    array([[ 0.,  1.,  2.],\n           [ 3.,  4., -1.],\n           [-1., -1., -1.]])\n\n    Find the indices of elements of `x` that are in `goodvalues`.\n\n    >>> goodvalues = [3, 4, 7]\n    >>> ix = np.in1d(x.ravel(), goodvalues).reshape(x.shape)\n    >>> ix\n    array([[False, False, False],\n           [ True,  True, False],\n           [False,  True, False]], dtype=bool)\n    >>> np.where(ix)\n    (array([1, 1, 2]), array([0, 1, 1]))\n\n    ')
# Processing the call keyword arguments (line 1427)
kwargs_21383 = {}
# Getting the type of 'add_newdoc' (line 1427)
add_newdoc_21379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1427, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 1427)
add_newdoc_call_result_21384 = invoke(stypy.reporting.localization.Localization(__file__, 1427, 0), add_newdoc_21379, *[str_21380, str_21381, str_21382], **kwargs_21383)


# Call to add_newdoc(...): (line 1499)
# Processing the call arguments (line 1499)
str_21386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1499, 11), 'str', 'numpy.core.multiarray')
str_21387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1499, 36), 'str', 'lexsort')
str_21388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1573, (-1)), 'str', '\n    lexsort(keys, axis=-1)\n\n    Perform an indirect sort using a sequence of keys.\n\n    Given multiple sorting keys, which can be interpreted as columns in a\n    spreadsheet, lexsort returns an array of integer indices that describes\n    the sort order by multiple columns. The last key in the sequence is used\n    for the primary sort order, the second-to-last key for the secondary sort\n    order, and so on. The keys argument must be a sequence of objects that\n    can be converted to arrays of the same shape. If a 2D array is provided\n    for the keys argument, it\'s rows are interpreted as the sorting keys and\n    sorting is according to the last row, second last row etc.\n\n    Parameters\n    ----------\n    keys : (k, N) array or tuple containing k (N,)-shaped sequences\n        The `k` different "columns" to be sorted.  The last column (or row if\n        `keys` is a 2D array) is the primary sort key.\n    axis : int, optional\n        Axis to be indirectly sorted.  By default, sort over the last axis.\n\n    Returns\n    -------\n    indices : (N,) ndarray of ints\n        Array of indices that sort the keys along the specified axis.\n\n    See Also\n    --------\n    argsort : Indirect sort.\n    ndarray.sort : In-place sort.\n    sort : Return a sorted copy of an array.\n\n    Examples\n    --------\n    Sort names: first by surname, then by name.\n\n    >>> surnames =    (\'Hertz\',    \'Galilei\', \'Hertz\')\n    >>> first_names = (\'Heinrich\', \'Galileo\', \'Gustav\')\n    >>> ind = np.lexsort((first_names, surnames))\n    >>> ind\n    array([1, 2, 0])\n\n    >>> [surnames[i] + ", " + first_names[i] for i in ind]\n    [\'Galilei, Galileo\', \'Hertz, Gustav\', \'Hertz, Heinrich\']\n\n    Sort two columns of numbers:\n\n    >>> a = [1,5,1,4,3,4,4] # First column\n    >>> b = [9,4,0,4,0,2,1] # Second column\n    >>> ind = np.lexsort((b,a)) # Sort by a, then by b\n    >>> print(ind)\n    [2 0 4 6 5 3 1]\n\n    >>> [(a[i],b[i]) for i in ind]\n    [(1, 0), (1, 9), (3, 0), (4, 1), (4, 2), (4, 4), (5, 4)]\n\n    Note that sorting is first according to the elements of ``a``.\n    Secondary sorting is according to the elements of ``b``.\n\n    A normal ``argsort`` would have yielded:\n\n    >>> [(a[i],b[i]) for i in np.argsort(a)]\n    [(1, 9), (1, 0), (3, 0), (4, 4), (4, 2), (4, 1), (5, 4)]\n\n    Structured arrays are sorted lexically by ``argsort``:\n\n    >>> x = np.array([(1,9), (5,4), (1,0), (4,4), (3,0), (4,2), (4,1)],\n    ...              dtype=np.dtype([(\'x\', int), (\'y\', int)]))\n\n    >>> np.argsort(x) # or np.argsort(x, order=(\'x\', \'y\'))\n    array([2, 0, 4, 6, 5, 3, 1])\n\n    ')
# Processing the call keyword arguments (line 1499)
kwargs_21389 = {}
# Getting the type of 'add_newdoc' (line 1499)
add_newdoc_21385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1499, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 1499)
add_newdoc_call_result_21390 = invoke(stypy.reporting.localization.Localization(__file__, 1499, 0), add_newdoc_21385, *[str_21386, str_21387, str_21388], **kwargs_21389)


# Call to add_newdoc(...): (line 1575)
# Processing the call arguments (line 1575)
str_21392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1575, 11), 'str', 'numpy.core.multiarray')
str_21393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1575, 36), 'str', 'can_cast')
str_21394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1681, (-1)), 'str', "\n    can_cast(from, totype, casting = 'safe')\n\n    Returns True if cast between data types can occur according to the\n    casting rule.  If from is a scalar or array scalar, also returns\n    True if the scalar value can be cast without overflow or truncation\n    to an integer.\n\n    Parameters\n    ----------\n    from : dtype, dtype specifier, scalar, or array\n        Data type, scalar, or array to cast from.\n    totype : dtype or dtype specifier\n        Data type to cast to.\n    casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional\n        Controls what kind of data casting may occur.\n\n          * 'no' means the data types should not be cast at all.\n          * 'equiv' means only byte-order changes are allowed.\n          * 'safe' means only casts which can preserve values are allowed.\n          * 'same_kind' means only safe casts or casts within a kind,\n            like float64 to float32, are allowed.\n          * 'unsafe' means any data conversions may be done.\n\n    Returns\n    -------\n    out : bool\n        True if cast can occur according to the casting rule.\n\n    Notes\n    -----\n    Starting in NumPy 1.9, can_cast function now returns False in 'safe'\n    casting mode for integer/float dtype and string dtype if the string dtype\n    length is not long enough to store the max integer/float value converted\n    to a string. Previously can_cast in 'safe' mode returned True for\n    integer/float dtype and a string dtype of any length.\n\n    See also\n    --------\n    dtype, result_type\n\n    Examples\n    --------\n    Basic examples\n\n    >>> np.can_cast(np.int32, np.int64)\n    True\n    >>> np.can_cast(np.float64, np.complex)\n    True\n    >>> np.can_cast(np.complex, np.float)\n    False\n\n    >>> np.can_cast('i8', 'f8')\n    True\n    >>> np.can_cast('i8', 'f4')\n    False\n    >>> np.can_cast('i4', 'S4')\n    False\n\n    Casting scalars\n\n    >>> np.can_cast(100, 'i1')\n    True\n    >>> np.can_cast(150, 'i1')\n    False\n    >>> np.can_cast(150, 'u1')\n    True\n\n    >>> np.can_cast(3.5e100, np.float32)\n    False\n    >>> np.can_cast(1000.0, np.float32)\n    True\n\n    Array scalar checks the value, array does not\n\n    >>> np.can_cast(np.array(1000.0), np.float32)\n    True\n    >>> np.can_cast(np.array([1000.0]), np.float32)\n    False\n\n    Using the casting rules\n\n    >>> np.can_cast('i8', 'i8', 'no')\n    True\n    >>> np.can_cast('<i8', '>i8', 'no')\n    False\n\n    >>> np.can_cast('<i8', '>i8', 'equiv')\n    True\n    >>> np.can_cast('<i4', '>i8', 'equiv')\n    False\n\n    >>> np.can_cast('<i4', '>i8', 'safe')\n    True\n    >>> np.can_cast('<i8', '>i4', 'safe')\n    False\n\n    >>> np.can_cast('<i8', '>i4', 'same_kind')\n    True\n    >>> np.can_cast('<i8', '>u4', 'same_kind')\n    False\n\n    >>> np.can_cast('<i8', '>u4', 'unsafe')\n    True\n\n    ")
# Processing the call keyword arguments (line 1575)
kwargs_21395 = {}
# Getting the type of 'add_newdoc' (line 1575)
add_newdoc_21391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1575, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 1575)
add_newdoc_call_result_21396 = invoke(stypy.reporting.localization.Localization(__file__, 1575, 0), add_newdoc_21391, *[str_21392, str_21393, str_21394], **kwargs_21395)


# Call to add_newdoc(...): (line 1683)
# Processing the call arguments (line 1683)
str_21398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1683, 11), 'str', 'numpy.core.multiarray')
str_21399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1683, 36), 'str', 'promote_types')
str_21400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1733, (-1)), 'str', "\n    promote_types(type1, type2)\n\n    Returns the data type with the smallest size and smallest scalar\n    kind to which both ``type1`` and ``type2`` may be safely cast.\n    The returned data type is always in native byte order.\n\n    This function is symmetric and associative.\n\n    Parameters\n    ----------\n    type1 : dtype or dtype specifier\n        First data type.\n    type2 : dtype or dtype specifier\n        Second data type.\n\n    Returns\n    -------\n    out : dtype\n        The promoted data type.\n\n    Notes\n    -----\n    .. versionadded:: 1.6.0\n\n    Starting in NumPy 1.9, promote_types function now returns a valid string\n    length when given an integer or float dtype as one argument and a string\n    dtype as another argument. Previously it always returned the input string\n    dtype, even if it wasn't long enough to store the max integer/float value\n    converted to a string.\n\n    See Also\n    --------\n    result_type, dtype, can_cast\n\n    Examples\n    --------\n    >>> np.promote_types('f4', 'f8')\n    dtype('float64')\n\n    >>> np.promote_types('i8', 'f4')\n    dtype('float64')\n\n    >>> np.promote_types('>i8', '<c8')\n    dtype('complex128')\n\n    >>> np.promote_types('i4', 'S8')\n    dtype('S11')\n\n    ")
# Processing the call keyword arguments (line 1683)
kwargs_21401 = {}
# Getting the type of 'add_newdoc' (line 1683)
add_newdoc_21397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1683, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 1683)
add_newdoc_call_result_21402 = invoke(stypy.reporting.localization.Localization(__file__, 1683, 0), add_newdoc_21397, *[str_21398, str_21399, str_21400], **kwargs_21401)


# Call to add_newdoc(...): (line 1735)
# Processing the call arguments (line 1735)
str_21404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1735, 11), 'str', 'numpy.core.multiarray')
str_21405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1735, 36), 'str', 'min_scalar_type')
str_21406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1781, (-1)), 'str', "\n    min_scalar_type(a)\n\n    For scalar ``a``, returns the data type with the smallest size\n    and smallest scalar kind which can hold its value.  For non-scalar\n    array ``a``, returns the vector's dtype unmodified.\n\n    Floating point values are not demoted to integers,\n    and complex values are not demoted to floats.\n\n    Parameters\n    ----------\n    a : scalar or array_like\n        The value whose minimal data type is to be found.\n\n    Returns\n    -------\n    out : dtype\n        The minimal data type.\n\n    Notes\n    -----\n    .. versionadded:: 1.6.0\n\n    See Also\n    --------\n    result_type, promote_types, dtype, can_cast\n\n    Examples\n    --------\n    >>> np.min_scalar_type(10)\n    dtype('uint8')\n\n    >>> np.min_scalar_type(-260)\n    dtype('int16')\n\n    >>> np.min_scalar_type(3.1)\n    dtype('float16')\n\n    >>> np.min_scalar_type(1e50)\n    dtype('float64')\n\n    >>> np.min_scalar_type(np.arange(4,dtype='f8'))\n    dtype('float64')\n\n    ")
# Processing the call keyword arguments (line 1735)
kwargs_21407 = {}
# Getting the type of 'add_newdoc' (line 1735)
add_newdoc_21403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1735, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 1735)
add_newdoc_call_result_21408 = invoke(stypy.reporting.localization.Localization(__file__, 1735, 0), add_newdoc_21403, *[str_21404, str_21405, str_21406], **kwargs_21407)


# Call to add_newdoc(...): (line 1783)
# Processing the call arguments (line 1783)
str_21410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1783, 11), 'str', 'numpy.core.multiarray')
str_21411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1783, 36), 'str', 'result_type')
str_21412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1850, (-1)), 'str', "\n    result_type(*arrays_and_dtypes)\n\n    Returns the type that results from applying the NumPy\n    type promotion rules to the arguments.\n\n    Type promotion in NumPy works similarly to the rules in languages\n    like C++, with some slight differences.  When both scalars and\n    arrays are used, the array's type takes precedence and the actual value\n    of the scalar is taken into account.\n\n    For example, calculating 3*a, where a is an array of 32-bit floats,\n    intuitively should result in a 32-bit float output.  If the 3 is a\n    32-bit integer, the NumPy rules indicate it can't convert losslessly\n    into a 32-bit float, so a 64-bit float should be the result type.\n    By examining the value of the constant, '3', we see that it fits in\n    an 8-bit integer, which can be cast losslessly into the 32-bit float.\n\n    Parameters\n    ----------\n    arrays_and_dtypes : list of arrays and dtypes\n        The operands of some operation whose result type is needed.\n\n    Returns\n    -------\n    out : dtype\n        The result type.\n\n    See also\n    --------\n    dtype, promote_types, min_scalar_type, can_cast\n\n    Notes\n    -----\n    .. versionadded:: 1.6.0\n\n    The specific algorithm used is as follows.\n\n    Categories are determined by first checking which of boolean,\n    integer (int/uint), or floating point (float/complex) the maximum\n    kind of all the arrays and the scalars are.\n\n    If there are only scalars or the maximum category of the scalars\n    is higher than the maximum category of the arrays,\n    the data types are combined with :func:`promote_types`\n    to produce the return value.\n\n    Otherwise, `min_scalar_type` is called on each array, and\n    the resulting data types are all combined with :func:`promote_types`\n    to produce the return value.\n\n    The set of int values is not a subset of the uint values for types\n    with the same number of bits, something not reflected in\n    :func:`min_scalar_type`, but handled as a special case in `result_type`.\n\n    Examples\n    --------\n    >>> np.result_type(3, np.arange(7, dtype='i1'))\n    dtype('int8')\n\n    >>> np.result_type('i4', 'c8')\n    dtype('complex128')\n\n    >>> np.result_type(3.0, -2)\n    dtype('float64')\n\n    ")
# Processing the call keyword arguments (line 1783)
kwargs_21413 = {}
# Getting the type of 'add_newdoc' (line 1783)
add_newdoc_21409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1783, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 1783)
add_newdoc_call_result_21414 = invoke(stypy.reporting.localization.Localization(__file__, 1783, 0), add_newdoc_21409, *[str_21410, str_21411, str_21412], **kwargs_21413)


# Call to add_newdoc(...): (line 1852)
# Processing the call arguments (line 1852)
str_21416 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1852, 11), 'str', 'numpy.core.multiarray')
str_21417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1852, 36), 'str', 'newbuffer')
str_21418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1868, (-1)), 'str', '\n    newbuffer(size)\n\n    Return a new uninitialized buffer object.\n\n    Parameters\n    ----------\n    size : int\n        Size in bytes of returned buffer object.\n\n    Returns\n    -------\n    newbuffer : buffer object\n        Returned, uninitialized buffer object of `size` bytes.\n\n    ')
# Processing the call keyword arguments (line 1852)
kwargs_21419 = {}
# Getting the type of 'add_newdoc' (line 1852)
add_newdoc_21415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1852, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 1852)
add_newdoc_call_result_21420 = invoke(stypy.reporting.localization.Localization(__file__, 1852, 0), add_newdoc_21415, *[str_21416, str_21417, str_21418], **kwargs_21419)


# Call to add_newdoc(...): (line 1870)
# Processing the call arguments (line 1870)
str_21422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1870, 11), 'str', 'numpy.core.multiarray')
str_21423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1870, 36), 'str', 'getbuffer')
str_21424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1902, (-1)), 'str', "\n    getbuffer(obj [,offset[, size]])\n\n    Create a buffer object from the given object referencing a slice of\n    length size starting at offset.\n\n    Default is the entire buffer. A read-write buffer is attempted followed\n    by a read-only buffer.\n\n    Parameters\n    ----------\n    obj : object\n\n    offset : int, optional\n\n    size : int, optional\n\n    Returns\n    -------\n    buffer_obj : buffer\n\n    Examples\n    --------\n    >>> buf = np.getbuffer(np.ones(5), 1, 3)\n    >>> len(buf)\n    3\n    >>> buf[0]\n    '\\x00'\n    >>> buf\n    <read-write buffer for 0x8af1e70, size 3, offset 1 at 0x8ba4ec0>\n\n    ")
# Processing the call keyword arguments (line 1870)
kwargs_21425 = {}
# Getting the type of 'add_newdoc' (line 1870)
add_newdoc_21421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1870, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 1870)
add_newdoc_call_result_21426 = invoke(stypy.reporting.localization.Localization(__file__, 1870, 0), add_newdoc_21421, *[str_21422, str_21423, str_21424], **kwargs_21425)


# Call to add_newdoc(...): (line 1904)
# Processing the call arguments (line 1904)
str_21428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1904, 11), 'str', 'numpy.core')
str_21429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1904, 25), 'str', 'dot')
str_21430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1977, (-1)), 'str', "\n    dot(a, b, out=None)\n\n    Dot product of two arrays.\n\n    For 2-D arrays it is equivalent to matrix multiplication, and for 1-D\n    arrays to inner product of vectors (without complex conjugation). For\n    N dimensions it is a sum product over the last axis of `a` and\n    the second-to-last of `b`::\n\n        dot(a, b)[i,j,k,m] = sum(a[i,j,:] * b[k,:,m])\n\n    Parameters\n    ----------\n    a : array_like\n        First argument.\n    b : array_like\n        Second argument.\n    out : ndarray, optional\n        Output argument. This must have the exact kind that would be returned\n        if it was not used. In particular, it must have the right type, must be\n        C-contiguous, and its dtype must be the dtype that would be returned\n        for `dot(a,b)`. This is a performance feature. Therefore, if these\n        conditions are not met, an exception is raised, instead of attempting\n        to be flexible.\n\n    Returns\n    -------\n    output : ndarray\n        Returns the dot product of `a` and `b`.  If `a` and `b` are both\n        scalars or both 1-D arrays then a scalar is returned; otherwise\n        an array is returned.\n        If `out` is given, then it is returned.\n\n    Raises\n    ------\n    ValueError\n        If the last dimension of `a` is not the same size as\n        the second-to-last dimension of `b`.\n\n    See Also\n    --------\n    vdot : Complex-conjugating dot product.\n    tensordot : Sum products over arbitrary axes.\n    einsum : Einstein summation convention.\n    matmul : '@' operator as method with out parameter.\n\n    Examples\n    --------\n    >>> np.dot(3, 4)\n    12\n\n    Neither argument is complex-conjugated:\n\n    >>> np.dot([2j, 3j], [2j, 3j])\n    (-13+0j)\n\n    For 2-D arrays it is the matrix product:\n\n    >>> a = [[1, 0], [0, 1]]\n    >>> b = [[4, 1], [2, 2]]\n    >>> np.dot(a, b)\n    array([[4, 1],\n           [2, 2]])\n\n    >>> a = np.arange(3*4*5*6).reshape((3,4,5,6))\n    >>> b = np.arange(3*4*5*6)[::-1].reshape((5,4,6,3))\n    >>> np.dot(a, b)[2,3,2,1,2,2]\n    499128\n    >>> sum(a[2,3,2,:] * b[1,2,:,2])\n    499128\n\n    ")
# Processing the call keyword arguments (line 1904)
kwargs_21431 = {}
# Getting the type of 'add_newdoc' (line 1904)
add_newdoc_21427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1904, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 1904)
add_newdoc_call_result_21432 = invoke(stypy.reporting.localization.Localization(__file__, 1904, 0), add_newdoc_21427, *[str_21428, str_21429, str_21430], **kwargs_21431)


# Call to add_newdoc(...): (line 1979)
# Processing the call arguments (line 1979)
str_21434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1979, 11), 'str', 'numpy.core')
str_21435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1979, 25), 'str', 'matmul')
str_21436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2100, (-1)), 'str', "\n    matmul(a, b, out=None)\n\n    Matrix product of two arrays.\n\n    The behavior depends on the arguments in the following way.\n\n    - If both arguments are 2-D they are multiplied like conventional\n      matrices.\n    - If either argument is N-D, N > 2, it is treated as a stack of\n      matrices residing in the last two indexes and broadcast accordingly.\n    - If the first argument is 1-D, it is promoted to a matrix by\n      prepending a 1 to its dimensions. After matrix multiplication\n      the prepended 1 is removed.\n    - If the second argument is 1-D, it is promoted to a matrix by\n      appending a 1 to its dimensions. After matrix multiplication\n      the appended 1 is removed.\n\n    Multiplication by a scalar is not allowed, use ``*`` instead. Note that\n    multiplying a stack of matrices with a vector will result in a stack of\n    vectors, but matmul will not recognize it as such.\n\n    ``matmul`` differs from ``dot`` in two important ways.\n\n    - Multiplication by scalars is not allowed.\n    - Stacks of matrices are broadcast together as if the matrices\n      were elements.\n\n    .. warning::\n       This function is preliminary and included in Numpy 1.10 for testing\n       and documentation. Its semantics will not change, but the number and\n       order of the optional arguments will.\n\n    .. versionadded:: 1.10.0\n\n    Parameters\n    ----------\n    a : array_like\n        First argument.\n    b : array_like\n        Second argument.\n    out : ndarray, optional\n        Output argument. This must have the exact kind that would be returned\n        if it was not used. In particular, it must have the right type, must be\n        C-contiguous, and its dtype must be the dtype that would be returned\n        for `dot(a,b)`. This is a performance feature. Therefore, if these\n        conditions are not met, an exception is raised, instead of attempting\n        to be flexible.\n\n    Returns\n    -------\n    output : ndarray\n        Returns the dot product of `a` and `b`.  If `a` and `b` are both\n        1-D arrays then a scalar is returned; otherwise an array is\n        returned.  If `out` is given, then it is returned.\n\n    Raises\n    ------\n    ValueError\n        If the last dimension of `a` is not the same size as\n        the second-to-last dimension of `b`.\n\n        If scalar value is passed.\n\n    See Also\n    --------\n    vdot : Complex-conjugating dot product.\n    tensordot : Sum products over arbitrary axes.\n    einsum : Einstein summation convention.\n    dot : alternative matrix product with different broadcasting rules.\n\n    Notes\n    -----\n    The matmul function implements the semantics of the `@` operator introduced\n    in Python 3.5 following PEP465.\n\n    Examples\n    --------\n    For 2-D arrays it is the matrix product:\n\n    >>> a = [[1, 0], [0, 1]]\n    >>> b = [[4, 1], [2, 2]]\n    >>> np.matmul(a, b)\n    array([[4, 1],\n           [2, 2]])\n\n    For 2-D mixed with 1-D, the result is the usual.\n\n    >>> a = [[1, 0], [0, 1]]\n    >>> b = [1, 2]\n    >>> np.matmul(a, b)\n    array([1, 2])\n    >>> np.matmul(b, a)\n    array([1, 2])\n\n\n    Broadcasting is conventional for stacks of arrays\n\n    >>> a = np.arange(2*2*4).reshape((2,2,4))\n    >>> b = np.arange(2*2*4).reshape((2,4,2))\n    >>> np.matmul(a,b).shape\n    (2, 2, 2)\n    >>> np.matmul(a,b)[0,1,1]\n    98\n    >>> sum(a[0,1,:] * b[0,:,1])\n    98\n\n    Vector, vector returns the scalar inner product, but neither argument\n    is complex-conjugated:\n\n    >>> np.matmul([2j, 3j], [2j, 3j])\n    (-13+0j)\n\n    Scalar multiplication raises an error.\n\n    >>> np.matmul([1,2], 3)\n    Traceback (most recent call last):\n    ...\n    ValueError: Scalar operands are not allowed, use '*' instead\n\n    ")
# Processing the call keyword arguments (line 1979)
kwargs_21437 = {}
# Getting the type of 'add_newdoc' (line 1979)
add_newdoc_21433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1979, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 1979)
add_newdoc_call_result_21438 = invoke(stypy.reporting.localization.Localization(__file__, 1979, 0), add_newdoc_21433, *[str_21434, str_21435, str_21436], **kwargs_21437)


# Call to add_newdoc(...): (line 2103)
# Processing the call arguments (line 2103)
str_21440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2103, 11), 'str', 'numpy.core')
str_21441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2103, 25), 'str', 'einsum')
str_21442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2321, (-1)), 'str', "\n    einsum(subscripts, *operands, out=None, dtype=None, order='K', casting='safe')\n\n    Evaluates the Einstein summation convention on the operands.\n\n    Using the Einstein summation convention, many common multi-dimensional\n    array operations can be represented in a simple fashion.  This function\n    provides a way to compute such summations. The best way to understand this\n    function is to try the examples below, which show how many common NumPy\n    functions can be implemented as calls to `einsum`.\n\n    Parameters\n    ----------\n    subscripts : str\n        Specifies the subscripts for summation.\n    operands : list of array_like\n        These are the arrays for the operation.\n    out : ndarray, optional\n        If provided, the calculation is done into this array.\n    dtype : data-type, optional\n        If provided, forces the calculation to use the data type specified.\n        Note that you may have to also give a more liberal `casting`\n        parameter to allow the conversions.\n    order : {'C', 'F', 'A', 'K'}, optional\n        Controls the memory layout of the output. 'C' means it should\n        be C contiguous. 'F' means it should be Fortran contiguous,\n        'A' means it should be 'F' if the inputs are all 'F', 'C' otherwise.\n        'K' means it should be as close to the layout as the inputs as\n        is possible, including arbitrarily permuted axes.\n        Default is 'K'.\n    casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional\n        Controls what kind of data casting may occur.  Setting this to\n        'unsafe' is not recommended, as it can adversely affect accumulations.\n\n          * 'no' means the data types should not be cast at all.\n          * 'equiv' means only byte-order changes are allowed.\n          * 'safe' means only casts which can preserve values are allowed.\n          * 'same_kind' means only safe casts or casts within a kind,\n            like float64 to float32, are allowed.\n          * 'unsafe' means any data conversions may be done.\n\n    Returns\n    -------\n    output : ndarray\n        The calculation based on the Einstein summation convention.\n\n    See Also\n    --------\n    dot, inner, outer, tensordot\n\n    Notes\n    -----\n    .. versionadded:: 1.6.0\n\n    The subscripts string is a comma-separated list of subscript labels,\n    where each label refers to a dimension of the corresponding operand.\n    Repeated subscripts labels in one operand take the diagonal.  For example,\n    ``np.einsum('ii', a)`` is equivalent to ``np.trace(a)``.\n\n    Whenever a label is repeated, it is summed, so ``np.einsum('i,i', a, b)``\n    is equivalent to ``np.inner(a,b)``.  If a label appears only once,\n    it is not summed, so ``np.einsum('i', a)`` produces a view of ``a``\n    with no changes.\n\n    The order of labels in the output is by default alphabetical.  This\n    means that ``np.einsum('ij', a)`` doesn't affect a 2D array, while\n    ``np.einsum('ji', a)`` takes its transpose.\n\n    The output can be controlled by specifying output subscript labels\n    as well.  This specifies the label order, and allows summing to\n    be disallowed or forced when desired.  The call ``np.einsum('i->', a)``\n    is like ``np.sum(a, axis=-1)``, and ``np.einsum('ii->i', a)``\n    is like ``np.diag(a)``.  The difference is that `einsum` does not\n    allow broadcasting by default.\n\n    To enable and control broadcasting, use an ellipsis.  Default\n    NumPy-style broadcasting is done by adding an ellipsis\n    to the left of each term, like ``np.einsum('...ii->...i', a)``.\n    To take the trace along the first and last axes,\n    you can do ``np.einsum('i...i', a)``, or to do a matrix-matrix\n    product with the left-most indices instead of rightmost, you can do\n    ``np.einsum('ij...,jk...->ik...', a, b)``.\n\n    When there is only one operand, no axes are summed, and no output\n    parameter is provided, a view into the operand is returned instead\n    of a new array.  Thus, taking the diagonal as ``np.einsum('ii->i', a)``\n    produces a view.\n\n    An alternative way to provide the subscripts and operands is as\n    ``einsum(op0, sublist0, op1, sublist1, ..., [sublistout])``. The examples\n    below have corresponding `einsum` calls with the two parameter methods.\n\n    .. versionadded:: 1.10.0\n\n    Views returned from einsum are now writeable whenever the input array\n    is writeable. For example, ``np.einsum('ijk...->kji...', a)`` will now\n    have the same effect as ``np.swapaxes(a, 0, 2)`` and\n    ``np.einsum('ii->i', a)`` will return a writeable view of the diagonal\n    of a 2D array.\n\n    Examples\n    --------\n    >>> a = np.arange(25).reshape(5,5)\n    >>> b = np.arange(5)\n    >>> c = np.arange(6).reshape(2,3)\n\n    >>> np.einsum('ii', a)\n    60\n    >>> np.einsum(a, [0,0])\n    60\n    >>> np.trace(a)\n    60\n\n    >>> np.einsum('ii->i', a)\n    array([ 0,  6, 12, 18, 24])\n    >>> np.einsum(a, [0,0], [0])\n    array([ 0,  6, 12, 18, 24])\n    >>> np.diag(a)\n    array([ 0,  6, 12, 18, 24])\n\n    >>> np.einsum('ij,j', a, b)\n    array([ 30,  80, 130, 180, 230])\n    >>> np.einsum(a, [0,1], b, [1])\n    array([ 30,  80, 130, 180, 230])\n    >>> np.dot(a, b)\n    array([ 30,  80, 130, 180, 230])\n    >>> np.einsum('...j,j', a, b)\n    array([ 30,  80, 130, 180, 230])\n\n    >>> np.einsum('ji', c)\n    array([[0, 3],\n           [1, 4],\n           [2, 5]])\n    >>> np.einsum(c, [1,0])\n    array([[0, 3],\n           [1, 4],\n           [2, 5]])\n    >>> c.T\n    array([[0, 3],\n           [1, 4],\n           [2, 5]])\n\n    >>> np.einsum('..., ...', 3, c)\n    array([[ 0,  3,  6],\n           [ 9, 12, 15]])\n    >>> np.einsum(3, [Ellipsis], c, [Ellipsis])\n    array([[ 0,  3,  6],\n           [ 9, 12, 15]])\n    >>> np.multiply(3, c)\n    array([[ 0,  3,  6],\n           [ 9, 12, 15]])\n\n    >>> np.einsum('i,i', b, b)\n    30\n    >>> np.einsum(b, [0], b, [0])\n    30\n    >>> np.inner(b,b)\n    30\n\n    >>> np.einsum('i,j', np.arange(2)+1, b)\n    array([[0, 1, 2, 3, 4],\n           [0, 2, 4, 6, 8]])\n    >>> np.einsum(np.arange(2)+1, [0], b, [1])\n    array([[0, 1, 2, 3, 4],\n           [0, 2, 4, 6, 8]])\n    >>> np.outer(np.arange(2)+1, b)\n    array([[0, 1, 2, 3, 4],\n           [0, 2, 4, 6, 8]])\n\n    >>> np.einsum('i...->...', a)\n    array([50, 55, 60, 65, 70])\n    >>> np.einsum(a, [0,Ellipsis], [Ellipsis])\n    array([50, 55, 60, 65, 70])\n    >>> np.sum(a, axis=0)\n    array([50, 55, 60, 65, 70])\n\n    >>> a = np.arange(60.).reshape(3,4,5)\n    >>> b = np.arange(24.).reshape(4,3,2)\n    >>> np.einsum('ijk,jil->kl', a, b)\n    array([[ 4400.,  4730.],\n           [ 4532.,  4874.],\n           [ 4664.,  5018.],\n           [ 4796.,  5162.],\n           [ 4928.,  5306.]])\n    >>> np.einsum(a, [0,1,2], b, [1,0,3], [2,3])\n    array([[ 4400.,  4730.],\n           [ 4532.,  4874.],\n           [ 4664.,  5018.],\n           [ 4796.,  5162.],\n           [ 4928.,  5306.]])\n    >>> np.tensordot(a,b, axes=([1,0],[0,1]))\n    array([[ 4400.,  4730.],\n           [ 4532.,  4874.],\n           [ 4664.,  5018.],\n           [ 4796.,  5162.],\n           [ 4928.,  5306.]])\n\n    >>> a = np.arange(6).reshape((3,2))\n    >>> b = np.arange(12).reshape((4,3))\n    >>> np.einsum('ki,jk->ij', a, b)\n    array([[10, 28, 46, 64],\n           [13, 40, 67, 94]])\n    >>> np.einsum('ki,...k->i...', a, b)\n    array([[10, 28, 46, 64],\n           [13, 40, 67, 94]])\n    >>> np.einsum('k...,jk', a, b)\n    array([[10, 28, 46, 64],\n           [13, 40, 67, 94]])\n\n    >>> # since version 1.10.0\n    >>> a = np.zeros((3, 3))\n    >>> np.einsum('ii->i', a)[:] = 1\n    >>> a\n    array([[ 1.,  0.,  0.],\n           [ 0.,  1.,  0.],\n           [ 0.,  0.,  1.]])\n\n    ")
# Processing the call keyword arguments (line 2103)
kwargs_21443 = {}
# Getting the type of 'add_newdoc' (line 2103)
add_newdoc_21439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2103, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 2103)
add_newdoc_call_result_21444 = invoke(stypy.reporting.localization.Localization(__file__, 2103, 0), add_newdoc_21439, *[str_21440, str_21441, str_21442], **kwargs_21443)


# Call to add_newdoc(...): (line 2323)
# Processing the call arguments (line 2323)
str_21446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2323, 11), 'str', 'numpy.core')
str_21447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2323, 25), 'str', 'vdot')
str_21448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2376, (-1)), 'str', '\n    vdot(a, b)\n\n    Return the dot product of two vectors.\n\n    The vdot(`a`, `b`) function handles complex numbers differently than\n    dot(`a`, `b`).  If the first argument is complex the complex conjugate\n    of the first argument is used for the calculation of the dot product.\n\n    Note that `vdot` handles multidimensional arrays differently than `dot`:\n    it does *not* perform a matrix product, but flattens input arguments\n    to 1-D vectors first. Consequently, it should only be used for vectors.\n\n    Parameters\n    ----------\n    a : array_like\n        If `a` is complex the complex conjugate is taken before calculation\n        of the dot product.\n    b : array_like\n        Second argument to the dot product.\n\n    Returns\n    -------\n    output : ndarray\n        Dot product of `a` and `b`.  Can be an int, float, or\n        complex depending on the types of `a` and `b`.\n\n    See Also\n    --------\n    dot : Return the dot product without using the complex conjugate of the\n          first argument.\n\n    Examples\n    --------\n    >>> a = np.array([1+2j,3+4j])\n    >>> b = np.array([5+6j,7+8j])\n    >>> np.vdot(a, b)\n    (70-8j)\n    >>> np.vdot(b, a)\n    (70+8j)\n\n    Note that higher-dimensional arrays are flattened!\n\n    >>> a = np.array([[1, 4], [5, 6]])\n    >>> b = np.array([[4, 1], [2, 2]])\n    >>> np.vdot(a, b)\n    30\n    >>> np.vdot(b, a)\n    30\n    >>> 1*4 + 4*1 + 5*2 + 6*2\n    30\n\n    ')
# Processing the call keyword arguments (line 2323)
kwargs_21449 = {}
# Getting the type of 'add_newdoc' (line 2323)
add_newdoc_21445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2323, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 2323)
add_newdoc_call_result_21450 = invoke(stypy.reporting.localization.Localization(__file__, 2323, 0), add_newdoc_21445, *[str_21446, str_21447, str_21448], **kwargs_21449)


# Call to add_newdoc(...): (line 2393)
# Processing the call arguments (line 2393)
str_21452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2393, 11), 'str', 'numpy.core.multiarray')
str_21453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2393, 36), 'str', 'ndarray')
str_21454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2512, (-1)), 'str', '\n    ndarray(shape, dtype=float, buffer=None, offset=0,\n            strides=None, order=None)\n\n    An array object represents a multidimensional, homogeneous array\n    of fixed-size items.  An associated data-type object describes the\n    format of each element in the array (its byte-order, how many bytes it\n    occupies in memory, whether it is an integer, a floating point number,\n    or something else, etc.)\n\n    Arrays should be constructed using `array`, `zeros` or `empty` (refer\n    to the See Also section below).  The parameters given here refer to\n    a low-level method (`ndarray(...)`) for instantiating an array.\n\n    For more information, refer to the `numpy` module and examine the\n    the methods and attributes of an array.\n\n    Parameters\n    ----------\n    (for the __new__ method; see Notes below)\n\n    shape : tuple of ints\n        Shape of created array.\n    dtype : data-type, optional\n        Any object that can be interpreted as a numpy data type.\n    buffer : object exposing buffer interface, optional\n        Used to fill the array with data.\n    offset : int, optional\n        Offset of array data in buffer.\n    strides : tuple of ints, optional\n        Strides of data in memory.\n    order : {\'C\', \'F\'}, optional\n        Row-major (C-style) or column-major (Fortran-style) order.\n\n    Attributes\n    ----------\n    T : ndarray\n        Transpose of the array.\n    data : buffer\n        The array\'s elements, in memory.\n    dtype : dtype object\n        Describes the format of the elements in the array.\n    flags : dict\n        Dictionary containing information related to memory use, e.g.,\n        \'C_CONTIGUOUS\', \'OWNDATA\', \'WRITEABLE\', etc.\n    flat : numpy.flatiter object\n        Flattened version of the array as an iterator.  The iterator\n        allows assignments, e.g., ``x.flat = 3`` (See `ndarray.flat` for\n        assignment examples; TODO).\n    imag : ndarray\n        Imaginary part of the array.\n    real : ndarray\n        Real part of the array.\n    size : int\n        Number of elements in the array.\n    itemsize : int\n        The memory use of each array element in bytes.\n    nbytes : int\n        The total number of bytes required to store the array data,\n        i.e., ``itemsize * size``.\n    ndim : int\n        The array\'s number of dimensions.\n    shape : tuple of ints\n        Shape of the array.\n    strides : tuple of ints\n        The step-size required to move from one element to the next in\n        memory. For example, a contiguous ``(3, 4)`` array of type\n        ``int16`` in C-order has strides ``(8, 2)``.  This implies that\n        to move from element to element in memory requires jumps of 2 bytes.\n        To move from row-to-row, one needs to jump 8 bytes at a time\n        (``2 * 4``).\n    ctypes : ctypes object\n        Class containing properties of the array needed for interaction\n        with ctypes.\n    base : ndarray\n        If the array is a view into another array, that array is its `base`\n        (unless that array is also a view).  The `base` array is where the\n        array data is actually stored.\n\n    See Also\n    --------\n    array : Construct an array.\n    zeros : Create an array, each element of which is zero.\n    empty : Create an array, but leave its allocated memory unchanged (i.e.,\n            it contains "garbage").\n    dtype : Create a data-type.\n\n    Notes\n    -----\n    There are two modes of creating an array using ``__new__``:\n\n    1. If `buffer` is None, then only `shape`, `dtype`, and `order`\n       are used.\n    2. If `buffer` is an object exposing the buffer interface, then\n       all keywords are interpreted.\n\n    No ``__init__`` method is needed because the array is fully initialized\n    after the ``__new__`` method.\n\n    Examples\n    --------\n    These examples illustrate the low-level `ndarray` constructor.  Refer\n    to the `See Also` section above for easier ways of constructing an\n    ndarray.\n\n    First mode, `buffer` is None:\n\n    >>> np.ndarray(shape=(2,2), dtype=float, order=\'F\')\n    array([[ -1.13698227e+002,   4.25087011e-303],\n           [  2.88528414e-306,   3.27025015e-309]])         #random\n\n    Second mode:\n\n    >>> np.ndarray((2,), buffer=np.array([1,2,3]),\n    ...            offset=np.int_().itemsize,\n    ...            dtype=int) # offset = 1*itemsize, i.e. skip first element\n    array([2, 3])\n\n    ')
# Processing the call keyword arguments (line 2393)
kwargs_21455 = {}
# Getting the type of 'add_newdoc' (line 2393)
add_newdoc_21451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2393, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 2393)
add_newdoc_call_result_21456 = invoke(stypy.reporting.localization.Localization(__file__, 2393, 0), add_newdoc_21451, *[str_21452, str_21453, str_21454], **kwargs_21455)


# Call to add_newdoc(...): (line 2522)
# Processing the call arguments (line 2522)
str_21458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2522, 11), 'str', 'numpy.core.multiarray')
str_21459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2522, 36), 'str', 'ndarray')

# Obtaining an instance of the builtin type 'tuple' (line 2522)
tuple_21460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2522, 48), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 2522)
# Adding element type (line 2522)
str_21461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2522, 48), 'str', '__array_interface__')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2522, 48), tuple_21460, str_21461)
# Adding element type (line 2522)
str_21462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2523, 4), 'str', 'Array protocol: Python side.')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2522, 48), tuple_21460, str_21462)

# Processing the call keyword arguments (line 2522)
kwargs_21463 = {}
# Getting the type of 'add_newdoc' (line 2522)
add_newdoc_21457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2522, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 2522)
add_newdoc_call_result_21464 = invoke(stypy.reporting.localization.Localization(__file__, 2522, 0), add_newdoc_21457, *[str_21458, str_21459, tuple_21460], **kwargs_21463)


# Call to add_newdoc(...): (line 2526)
# Processing the call arguments (line 2526)
str_21466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2526, 11), 'str', 'numpy.core.multiarray')
str_21467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2526, 36), 'str', 'ndarray')

# Obtaining an instance of the builtin type 'tuple' (line 2526)
tuple_21468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2526, 48), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 2526)
# Adding element type (line 2526)
str_21469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2526, 48), 'str', '__array_finalize__')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2526, 48), tuple_21468, str_21469)
# Adding element type (line 2526)
str_21470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2527, 4), 'str', 'None.')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2526, 48), tuple_21468, str_21470)

# Processing the call keyword arguments (line 2526)
kwargs_21471 = {}
# Getting the type of 'add_newdoc' (line 2526)
add_newdoc_21465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2526, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 2526)
add_newdoc_call_result_21472 = invoke(stypy.reporting.localization.Localization(__file__, 2526, 0), add_newdoc_21465, *[str_21466, str_21467, tuple_21468], **kwargs_21471)


# Call to add_newdoc(...): (line 2530)
# Processing the call arguments (line 2530)
str_21474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2530, 11), 'str', 'numpy.core.multiarray')
str_21475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2530, 36), 'str', 'ndarray')

# Obtaining an instance of the builtin type 'tuple' (line 2530)
tuple_21476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2530, 48), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 2530)
# Adding element type (line 2530)
str_21477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2530, 48), 'str', '__array_priority__')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2530, 48), tuple_21476, str_21477)
# Adding element type (line 2530)
str_21478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2531, 4), 'str', 'Array priority.')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2530, 48), tuple_21476, str_21478)

# Processing the call keyword arguments (line 2530)
kwargs_21479 = {}
# Getting the type of 'add_newdoc' (line 2530)
add_newdoc_21473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2530, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 2530)
add_newdoc_call_result_21480 = invoke(stypy.reporting.localization.Localization(__file__, 2530, 0), add_newdoc_21473, *[str_21474, str_21475, tuple_21476], **kwargs_21479)


# Call to add_newdoc(...): (line 2534)
# Processing the call arguments (line 2534)
str_21482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2534, 11), 'str', 'numpy.core.multiarray')
str_21483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2534, 36), 'str', 'ndarray')

# Obtaining an instance of the builtin type 'tuple' (line 2534)
tuple_21484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2534, 48), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 2534)
# Adding element type (line 2534)
str_21485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2534, 48), 'str', '__array_struct__')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2534, 48), tuple_21484, str_21485)
# Adding element type (line 2534)
str_21486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2535, 4), 'str', 'Array protocol: C-struct side.')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2534, 48), tuple_21484, str_21486)

# Processing the call keyword arguments (line 2534)
kwargs_21487 = {}
# Getting the type of 'add_newdoc' (line 2534)
add_newdoc_21481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2534, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 2534)
add_newdoc_call_result_21488 = invoke(stypy.reporting.localization.Localization(__file__, 2534, 0), add_newdoc_21481, *[str_21482, str_21483, tuple_21484], **kwargs_21487)


# Call to add_newdoc(...): (line 2538)
# Processing the call arguments (line 2538)
str_21490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2538, 11), 'str', 'numpy.core.multiarray')
str_21491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2538, 36), 'str', 'ndarray')

# Obtaining an instance of the builtin type 'tuple' (line 2538)
tuple_21492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2538, 48), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 2538)
# Adding element type (line 2538)
str_21493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2538, 48), 'str', '_as_parameter_')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2538, 48), tuple_21492, str_21493)
# Adding element type (line 2538)
str_21494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2542, (-1)), 'str', 'Allow the array to be interpreted as a ctypes object by returning the\n    data-memory location as an integer\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2538, 48), tuple_21492, str_21494)

# Processing the call keyword arguments (line 2538)
kwargs_21495 = {}
# Getting the type of 'add_newdoc' (line 2538)
add_newdoc_21489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2538, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 2538)
add_newdoc_call_result_21496 = invoke(stypy.reporting.localization.Localization(__file__, 2538, 0), add_newdoc_21489, *[str_21490, str_21491, tuple_21492], **kwargs_21495)


# Call to add_newdoc(...): (line 2545)
# Processing the call arguments (line 2545)
str_21498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2545, 11), 'str', 'numpy.core.multiarray')
str_21499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2545, 36), 'str', 'ndarray')

# Obtaining an instance of the builtin type 'tuple' (line 2545)
tuple_21500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2545, 48), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 2545)
# Adding element type (line 2545)
str_21501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2545, 48), 'str', 'base')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2545, 48), tuple_21500, str_21501)
# Adding element type (line 2545)
str_21502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2563, (-1)), 'str', '\n    Base object if memory is from some other object.\n\n    Examples\n    --------\n    The base of an array that owns its memory is None:\n\n    >>> x = np.array([1,2,3,4])\n    >>> x.base is None\n    True\n\n    Slicing creates a view, whose memory is shared with x:\n\n    >>> y = x[2:]\n    >>> y.base is x\n    True\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2545, 48), tuple_21500, str_21502)

# Processing the call keyword arguments (line 2545)
kwargs_21503 = {}
# Getting the type of 'add_newdoc' (line 2545)
add_newdoc_21497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2545, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 2545)
add_newdoc_call_result_21504 = invoke(stypy.reporting.localization.Localization(__file__, 2545, 0), add_newdoc_21497, *[str_21498, str_21499, tuple_21500], **kwargs_21503)


# Call to add_newdoc(...): (line 2566)
# Processing the call arguments (line 2566)
str_21506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2566, 11), 'str', 'numpy.core.multiarray')
str_21507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2566, 36), 'str', 'ndarray')

# Obtaining an instance of the builtin type 'tuple' (line 2566)
tuple_21508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2566, 48), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 2566)
# Adding element type (line 2566)
str_21509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2566, 48), 'str', 'ctypes')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2566, 48), tuple_21508, str_21509)
# Adding element type (line 2566)
str_21510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2666, (-1)), 'str', '\n    An object to simplify the interaction of the array with the ctypes\n    module.\n\n    This attribute creates an object that makes it easier to use arrays\n    when calling shared libraries with the ctypes module. The returned\n    object has, among others, data, shape, and strides attributes (see\n    Notes below) which themselves return ctypes objects that can be used\n    as arguments to a shared library.\n\n    Parameters\n    ----------\n    None\n\n    Returns\n    -------\n    c : Python object\n        Possessing attributes data, shape, strides, etc.\n\n    See Also\n    --------\n    numpy.ctypeslib\n\n    Notes\n    -----\n    Below are the public attributes of this object which were documented\n    in "Guide to NumPy" (we have omitted undocumented public attributes,\n    as well as documented private attributes):\n\n    * data: A pointer to the memory area of the array as a Python integer.\n      This memory area may contain data that is not aligned, or not in correct\n      byte-order. The memory area may not even be writeable. The array\n      flags and data-type of this array should be respected when passing this\n      attribute to arbitrary C-code to avoid trouble that can include Python\n      crashing. User Beware! The value of this attribute is exactly the same\n      as self._array_interface_[\'data\'][0].\n\n    * shape (c_intp*self.ndim): A ctypes array of length self.ndim where\n      the basetype is the C-integer corresponding to dtype(\'p\') on this\n      platform. This base-type could be c_int, c_long, or c_longlong\n      depending on the platform. The c_intp type is defined accordingly in\n      numpy.ctypeslib. The ctypes array contains the shape of the underlying\n      array.\n\n    * strides (c_intp*self.ndim): A ctypes array of length self.ndim where\n      the basetype is the same as for the shape attribute. This ctypes array\n      contains the strides information from the underlying array. This strides\n      information is important for showing how many bytes must be jumped to\n      get to the next element in the array.\n\n    * data_as(obj): Return the data pointer cast to a particular c-types object.\n      For example, calling self._as_parameter_ is equivalent to\n      self.data_as(ctypes.c_void_p). Perhaps you want to use the data as a\n      pointer to a ctypes array of floating-point data:\n      self.data_as(ctypes.POINTER(ctypes.c_double)).\n\n    * shape_as(obj): Return the shape tuple as an array of some other c-types\n      type. For example: self.shape_as(ctypes.c_short).\n\n    * strides_as(obj): Return the strides tuple as an array of some other\n      c-types type. For example: self.strides_as(ctypes.c_longlong).\n\n    Be careful using the ctypes attribute - especially on temporary\n    arrays or arrays constructed on the fly. For example, calling\n    ``(a+b).ctypes.data_as(ctypes.c_void_p)`` returns a pointer to memory\n    that is invalid because the array created as (a+b) is deallocated\n    before the next Python statement. You can avoid this problem using\n    either ``c=a+b`` or ``ct=(a+b).ctypes``. In the latter case, ct will\n    hold a reference to the array until ct is deleted or re-assigned.\n\n    If the ctypes module is not available, then the ctypes attribute\n    of array objects still returns something useful, but ctypes objects\n    are not returned and errors may be raised instead. In particular,\n    the object will still have the as parameter attribute which will\n    return an integer equal to the data attribute.\n\n    Examples\n    --------\n    >>> import ctypes\n    >>> x\n    array([[0, 1],\n           [2, 3]])\n    >>> x.ctypes.data\n    30439712\n    >>> x.ctypes.data_as(ctypes.POINTER(ctypes.c_long))\n    <ctypes.LP_c_long object at 0x01F01300>\n    >>> x.ctypes.data_as(ctypes.POINTER(ctypes.c_long)).contents\n    c_long(0)\n    >>> x.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)).contents\n    c_longlong(4294967296L)\n    >>> x.ctypes.shape\n    <numpy.core._internal.c_long_Array_2 object at 0x01FFD580>\n    >>> x.ctypes.shape_as(ctypes.c_long)\n    <numpy.core._internal.c_long_Array_2 object at 0x01FCE620>\n    >>> x.ctypes.strides\n    <numpy.core._internal.c_long_Array_2 object at 0x01FCE620>\n    >>> x.ctypes.strides_as(ctypes.c_longlong)\n    <numpy.core._internal.c_longlong_Array_2 object at 0x01F01300>\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2566, 48), tuple_21508, str_21510)

# Processing the call keyword arguments (line 2566)
kwargs_21511 = {}
# Getting the type of 'add_newdoc' (line 2566)
add_newdoc_21505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2566, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 2566)
add_newdoc_call_result_21512 = invoke(stypy.reporting.localization.Localization(__file__, 2566, 0), add_newdoc_21505, *[str_21506, str_21507, tuple_21508], **kwargs_21511)


# Call to add_newdoc(...): (line 2669)
# Processing the call arguments (line 2669)
str_21514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2669, 11), 'str', 'numpy.core.multiarray')
str_21515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2669, 36), 'str', 'ndarray')

# Obtaining an instance of the builtin type 'tuple' (line 2669)
tuple_21516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2669, 48), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 2669)
# Adding element type (line 2669)
str_21517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2669, 48), 'str', 'data')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2669, 48), tuple_21516, str_21517)
# Adding element type (line 2669)
str_21518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2670, 4), 'str', "Python buffer object pointing to the start of the array's data.")
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2669, 48), tuple_21516, str_21518)

# Processing the call keyword arguments (line 2669)
kwargs_21519 = {}
# Getting the type of 'add_newdoc' (line 2669)
add_newdoc_21513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2669, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 2669)
add_newdoc_call_result_21520 = invoke(stypy.reporting.localization.Localization(__file__, 2669, 0), add_newdoc_21513, *[str_21514, str_21515, tuple_21516], **kwargs_21519)


# Call to add_newdoc(...): (line 2673)
# Processing the call arguments (line 2673)
str_21522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2673, 11), 'str', 'numpy.core.multiarray')
str_21523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2673, 36), 'str', 'ndarray')

# Obtaining an instance of the builtin type 'tuple' (line 2673)
tuple_21524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2673, 48), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 2673)
# Adding element type (line 2673)
str_21525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2673, 48), 'str', 'dtype')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2673, 48), tuple_21524, str_21525)
# Adding element type (line 2673)
str_21526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2699, (-1)), 'str', "\n    Data-type of the array's elements.\n\n    Parameters\n    ----------\n    None\n\n    Returns\n    -------\n    d : numpy dtype object\n\n    See Also\n    --------\n    numpy.dtype\n\n    Examples\n    --------\n    >>> x\n    array([[0, 1],\n           [2, 3]])\n    >>> x.dtype\n    dtype('int32')\n    >>> type(x.dtype)\n    <type 'numpy.dtype'>\n\n    ")
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2673, 48), tuple_21524, str_21526)

# Processing the call keyword arguments (line 2673)
kwargs_21527 = {}
# Getting the type of 'add_newdoc' (line 2673)
add_newdoc_21521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2673, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 2673)
add_newdoc_call_result_21528 = invoke(stypy.reporting.localization.Localization(__file__, 2673, 0), add_newdoc_21521, *[str_21522, str_21523, tuple_21524], **kwargs_21527)


# Call to add_newdoc(...): (line 2702)
# Processing the call arguments (line 2702)
str_21530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2702, 11), 'str', 'numpy.core.multiarray')
str_21531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2702, 36), 'str', 'ndarray')

# Obtaining an instance of the builtin type 'tuple' (line 2702)
tuple_21532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2702, 48), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 2702)
# Adding element type (line 2702)
str_21533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2702, 48), 'str', 'imag')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2702, 48), tuple_21532, str_21533)
# Adding element type (line 2702)
str_21534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2714, (-1)), 'str', "\n    The imaginary part of the array.\n\n    Examples\n    --------\n    >>> x = np.sqrt([1+0j, 0+1j])\n    >>> x.imag\n    array([ 0.        ,  0.70710678])\n    >>> x.imag.dtype\n    dtype('float64')\n\n    ")
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2702, 48), tuple_21532, str_21534)

# Processing the call keyword arguments (line 2702)
kwargs_21535 = {}
# Getting the type of 'add_newdoc' (line 2702)
add_newdoc_21529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2702, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 2702)
add_newdoc_call_result_21536 = invoke(stypy.reporting.localization.Localization(__file__, 2702, 0), add_newdoc_21529, *[str_21530, str_21531, tuple_21532], **kwargs_21535)


# Call to add_newdoc(...): (line 2717)
# Processing the call arguments (line 2717)
str_21538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2717, 11), 'str', 'numpy.core.multiarray')
str_21539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2717, 36), 'str', 'ndarray')

# Obtaining an instance of the builtin type 'tuple' (line 2717)
tuple_21540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2717, 48), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 2717)
# Adding element type (line 2717)
str_21541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2717, 48), 'str', 'itemsize')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2717, 48), tuple_21540, str_21541)
# Adding element type (line 2717)
str_21542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2730, (-1)), 'str', '\n    Length of one array element in bytes.\n\n    Examples\n    --------\n    >>> x = np.array([1,2,3], dtype=np.float64)\n    >>> x.itemsize\n    8\n    >>> x = np.array([1,2,3], dtype=np.complex128)\n    >>> x.itemsize\n    16\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2717, 48), tuple_21540, str_21542)

# Processing the call keyword arguments (line 2717)
kwargs_21543 = {}
# Getting the type of 'add_newdoc' (line 2717)
add_newdoc_21537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2717, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 2717)
add_newdoc_call_result_21544 = invoke(stypy.reporting.localization.Localization(__file__, 2717, 0), add_newdoc_21537, *[str_21538, str_21539, tuple_21540], **kwargs_21543)


# Call to add_newdoc(...): (line 2733)
# Processing the call arguments (line 2733)
str_21546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2733, 11), 'str', 'numpy.core.multiarray')
str_21547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2733, 36), 'str', 'ndarray')

# Obtaining an instance of the builtin type 'tuple' (line 2733)
tuple_21548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2733, 48), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 2733)
# Adding element type (line 2733)
str_21549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2733, 48), 'str', 'flags')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2733, 48), tuple_21548, str_21549)
# Adding element type (line 2733)
str_21550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2801, (-1)), 'str', "\n    Information about the memory layout of the array.\n\n    Attributes\n    ----------\n    C_CONTIGUOUS (C)\n        The data is in a single, C-style contiguous segment.\n    F_CONTIGUOUS (F)\n        The data is in a single, Fortran-style contiguous segment.\n    OWNDATA (O)\n        The array owns the memory it uses or borrows it from another object.\n    WRITEABLE (W)\n        The data area can be written to.  Setting this to False locks\n        the data, making it read-only.  A view (slice, etc.) inherits WRITEABLE\n        from its base array at creation time, but a view of a writeable\n        array may be subsequently locked while the base array remains writeable.\n        (The opposite is not true, in that a view of a locked array may not\n        be made writeable.  However, currently, locking a base object does not\n        lock any views that already reference it, so under that circumstance it\n        is possible to alter the contents of a locked array via a previously\n        created writeable view onto it.)  Attempting to change a non-writeable\n        array raises a RuntimeError exception.\n    ALIGNED (A)\n        The data and all elements are aligned appropriately for the hardware.\n    UPDATEIFCOPY (U)\n        This array is a copy of some other array. When this array is\n        deallocated, the base array will be updated with the contents of\n        this array.\n    FNC\n        F_CONTIGUOUS and not C_CONTIGUOUS.\n    FORC\n        F_CONTIGUOUS or C_CONTIGUOUS (one-segment test).\n    BEHAVED (B)\n        ALIGNED and WRITEABLE.\n    CARRAY (CA)\n        BEHAVED and C_CONTIGUOUS.\n    FARRAY (FA)\n        BEHAVED and F_CONTIGUOUS and not C_CONTIGUOUS.\n\n    Notes\n    -----\n    The `flags` object can be accessed dictionary-like (as in ``a.flags['WRITEABLE']``),\n    or by using lowercased attribute names (as in ``a.flags.writeable``). Short flag\n    names are only supported in dictionary access.\n\n    Only the UPDATEIFCOPY, WRITEABLE, and ALIGNED flags can be changed by\n    the user, via direct assignment to the attribute or dictionary entry,\n    or by calling `ndarray.setflags`.\n\n    The array flags cannot be set arbitrarily:\n\n    - UPDATEIFCOPY can only be set ``False``.\n    - ALIGNED can only be set ``True`` if the data is truly aligned.\n    - WRITEABLE can only be set ``True`` if the array owns its own memory\n      or the ultimate owner of the memory exposes a writeable buffer\n      interface or is a string.\n\n    Arrays can be both C-style and Fortran-style contiguous simultaneously.\n    This is clear for 1-dimensional arrays, but can also be true for higher\n    dimensional arrays.\n\n    Even for contiguous arrays a stride for a given dimension\n    ``arr.strides[dim]`` may be *arbitrary* if ``arr.shape[dim] == 1``\n    or the array has no elements.\n    It does *not* generally hold that ``self.strides[-1] == self.itemsize``\n    for C-style contiguous arrays or ``self.strides[0] == self.itemsize`` for\n    Fortran-style contiguous arrays is true.\n    ")
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2733, 48), tuple_21548, str_21550)

# Processing the call keyword arguments (line 2733)
kwargs_21551 = {}
# Getting the type of 'add_newdoc' (line 2733)
add_newdoc_21545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2733, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 2733)
add_newdoc_call_result_21552 = invoke(stypy.reporting.localization.Localization(__file__, 2733, 0), add_newdoc_21545, *[str_21546, str_21547, tuple_21548], **kwargs_21551)


# Call to add_newdoc(...): (line 2804)
# Processing the call arguments (line 2804)
str_21554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2804, 11), 'str', 'numpy.core.multiarray')
str_21555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2804, 36), 'str', 'ndarray')

# Obtaining an instance of the builtin type 'tuple' (line 2804)
tuple_21556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2804, 48), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 2804)
# Adding element type (line 2804)
str_21557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2804, 48), 'str', 'flat')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2804, 48), tuple_21556, str_21557)
# Adding element type (line 2804)
str_21558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2843, (-1)), 'str', "\n    A 1-D iterator over the array.\n\n    This is a `numpy.flatiter` instance, which acts similarly to, but is not\n    a subclass of, Python's built-in iterator object.\n\n    See Also\n    --------\n    flatten : Return a copy of the array collapsed into one dimension.\n\n    flatiter\n\n    Examples\n    --------\n    >>> x = np.arange(1, 7).reshape(2, 3)\n    >>> x\n    array([[1, 2, 3],\n           [4, 5, 6]])\n    >>> x.flat[3]\n    4\n    >>> x.T\n    array([[1, 4],\n           [2, 5],\n           [3, 6]])\n    >>> x.T.flat[3]\n    5\n    >>> type(x.flat)\n    <type 'numpy.flatiter'>\n\n    An assignment example:\n\n    >>> x.flat = 3; x\n    array([[3, 3, 3],\n           [3, 3, 3]])\n    >>> x.flat[[1,4]] = 1; x\n    array([[3, 1, 3],\n           [3, 1, 3]])\n\n    ")
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2804, 48), tuple_21556, str_21558)

# Processing the call keyword arguments (line 2804)
kwargs_21559 = {}
# Getting the type of 'add_newdoc' (line 2804)
add_newdoc_21553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2804, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 2804)
add_newdoc_call_result_21560 = invoke(stypy.reporting.localization.Localization(__file__, 2804, 0), add_newdoc_21553, *[str_21554, str_21555, tuple_21556], **kwargs_21559)


# Call to add_newdoc(...): (line 2846)
# Processing the call arguments (line 2846)
str_21562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2846, 11), 'str', 'numpy.core.multiarray')
str_21563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2846, 36), 'str', 'ndarray')

# Obtaining an instance of the builtin type 'tuple' (line 2846)
tuple_21564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2846, 48), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 2846)
# Adding element type (line 2846)
str_21565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2846, 48), 'str', 'nbytes')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2846, 48), tuple_21564, str_21565)
# Adding element type (line 2846)
str_21566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2863, (-1)), 'str', '\n    Total bytes consumed by the elements of the array.\n\n    Notes\n    -----\n    Does not include memory consumed by non-element attributes of the\n    array object.\n\n    Examples\n    --------\n    >>> x = np.zeros((3,5,2), dtype=np.complex128)\n    >>> x.nbytes\n    480\n    >>> np.prod(x.shape) * x.itemsize\n    480\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2846, 48), tuple_21564, str_21566)

# Processing the call keyword arguments (line 2846)
kwargs_21567 = {}
# Getting the type of 'add_newdoc' (line 2846)
add_newdoc_21561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2846, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 2846)
add_newdoc_call_result_21568 = invoke(stypy.reporting.localization.Localization(__file__, 2846, 0), add_newdoc_21561, *[str_21562, str_21563, tuple_21564], **kwargs_21567)


# Call to add_newdoc(...): (line 2866)
# Processing the call arguments (line 2866)
str_21570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2866, 11), 'str', 'numpy.core.multiarray')
str_21571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2866, 36), 'str', 'ndarray')

# Obtaining an instance of the builtin type 'tuple' (line 2866)
tuple_21572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2866, 48), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 2866)
# Adding element type (line 2866)
str_21573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2866, 48), 'str', 'ndim')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2866, 48), tuple_21572, str_21573)
# Adding element type (line 2866)
str_21574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2879, (-1)), 'str', '\n    Number of array dimensions.\n\n    Examples\n    --------\n    >>> x = np.array([1, 2, 3])\n    >>> x.ndim\n    1\n    >>> y = np.zeros((2, 3, 4))\n    >>> y.ndim\n    3\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2866, 48), tuple_21572, str_21574)

# Processing the call keyword arguments (line 2866)
kwargs_21575 = {}
# Getting the type of 'add_newdoc' (line 2866)
add_newdoc_21569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2866, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 2866)
add_newdoc_call_result_21576 = invoke(stypy.reporting.localization.Localization(__file__, 2866, 0), add_newdoc_21569, *[str_21570, str_21571, tuple_21572], **kwargs_21575)


# Call to add_newdoc(...): (line 2882)
# Processing the call arguments (line 2882)
str_21578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2882, 11), 'str', 'numpy.core.multiarray')
str_21579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2882, 36), 'str', 'ndarray')

# Obtaining an instance of the builtin type 'tuple' (line 2882)
tuple_21580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2882, 48), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 2882)
# Adding element type (line 2882)
str_21581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2882, 48), 'str', 'real')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2882, 48), tuple_21580, str_21581)
# Adding element type (line 2882)
str_21582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2898, (-1)), 'str', "\n    The real part of the array.\n\n    Examples\n    --------\n    >>> x = np.sqrt([1+0j, 0+1j])\n    >>> x.real\n    array([ 1.        ,  0.70710678])\n    >>> x.real.dtype\n    dtype('float64')\n\n    See Also\n    --------\n    numpy.real : equivalent function\n\n    ")
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2882, 48), tuple_21580, str_21582)

# Processing the call keyword arguments (line 2882)
kwargs_21583 = {}
# Getting the type of 'add_newdoc' (line 2882)
add_newdoc_21577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2882, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 2882)
add_newdoc_call_result_21584 = invoke(stypy.reporting.localization.Localization(__file__, 2882, 0), add_newdoc_21577, *[str_21578, str_21579, tuple_21580], **kwargs_21583)


# Call to add_newdoc(...): (line 2901)
# Processing the call arguments (line 2901)
str_21586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2901, 11), 'str', 'numpy.core.multiarray')
str_21587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2901, 36), 'str', 'ndarray')

# Obtaining an instance of the builtin type 'tuple' (line 2901)
tuple_21588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2901, 48), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 2901)
# Adding element type (line 2901)
str_21589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2901, 48), 'str', 'shape')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2901, 48), tuple_21588, str_21589)
# Adding element type (line 2901)
str_21590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2928, (-1)), 'str', '\n    Tuple of array dimensions.\n\n    Notes\n    -----\n    May be used to "reshape" the array, as long as this would not\n    require a change in the total number of elements\n\n    Examples\n    --------\n    >>> x = np.array([1, 2, 3, 4])\n    >>> x.shape\n    (4,)\n    >>> y = np.zeros((2, 3, 4))\n    >>> y.shape\n    (2, 3, 4)\n    >>> y.shape = (3, 8)\n    >>> y\n    array([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],\n           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],\n           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])\n    >>> y.shape = (3, 6)\n    Traceback (most recent call last):\n      File "<stdin>", line 1, in <module>\n    ValueError: total size of new array must be unchanged\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2901, 48), tuple_21588, str_21590)

# Processing the call keyword arguments (line 2901)
kwargs_21591 = {}
# Getting the type of 'add_newdoc' (line 2901)
add_newdoc_21585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2901, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 2901)
add_newdoc_call_result_21592 = invoke(stypy.reporting.localization.Localization(__file__, 2901, 0), add_newdoc_21585, *[str_21586, str_21587, tuple_21588], **kwargs_21591)


# Call to add_newdoc(...): (line 2931)
# Processing the call arguments (line 2931)
str_21594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2931, 11), 'str', 'numpy.core.multiarray')
str_21595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2931, 36), 'str', 'ndarray')

# Obtaining an instance of the builtin type 'tuple' (line 2931)
tuple_21596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2931, 48), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 2931)
# Adding element type (line 2931)
str_21597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2931, 48), 'str', 'size')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2931, 48), tuple_21596, str_21597)
# Adding element type (line 2931)
str_21598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2946, (-1)), 'str', "\n    Number of elements in the array.\n\n    Equivalent to ``np.prod(a.shape)``, i.e., the product of the array's\n    dimensions.\n\n    Examples\n    --------\n    >>> x = np.zeros((3, 5, 2), dtype=np.complex128)\n    >>> x.size\n    30\n    >>> np.prod(x.shape)\n    30\n\n    ")
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2931, 48), tuple_21596, str_21598)

# Processing the call keyword arguments (line 2931)
kwargs_21599 = {}
# Getting the type of 'add_newdoc' (line 2931)
add_newdoc_21593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2931, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 2931)
add_newdoc_call_result_21600 = invoke(stypy.reporting.localization.Localization(__file__, 2931, 0), add_newdoc_21593, *[str_21594, str_21595, tuple_21596], **kwargs_21599)


# Call to add_newdoc(...): (line 2949)
# Processing the call arguments (line 2949)
str_21602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2949, 11), 'str', 'numpy.core.multiarray')
str_21603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2949, 36), 'str', 'ndarray')

# Obtaining an instance of the builtin type 'tuple' (line 2949)
tuple_21604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2949, 48), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 2949)
# Adding element type (line 2949)
str_21605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2949, 48), 'str', 'strides')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2949, 48), tuple_21604, str_21605)
# Adding element type (line 2949)
str_21606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3008, (-1)), 'str', '\n    Tuple of bytes to step in each dimension when traversing an array.\n\n    The byte offset of element ``(i[0], i[1], ..., i[n])`` in an array `a`\n    is::\n\n        offset = sum(np.array(i) * a.strides)\n\n    A more detailed explanation of strides can be found in the\n    "ndarray.rst" file in the NumPy reference guide.\n\n    Notes\n    -----\n    Imagine an array of 32-bit integers (each 4 bytes)::\n\n      x = np.array([[0, 1, 2, 3, 4],\n                    [5, 6, 7, 8, 9]], dtype=np.int32)\n\n    This array is stored in memory as 40 bytes, one after the other\n    (known as a contiguous block of memory).  The strides of an array tell\n    us how many bytes we have to skip in memory to move to the next position\n    along a certain axis.  For example, we have to skip 4 bytes (1 value) to\n    move to the next column, but 20 bytes (5 values) to get to the same\n    position in the next row.  As such, the strides for the array `x` will be\n    ``(20, 4)``.\n\n    See Also\n    --------\n    numpy.lib.stride_tricks.as_strided\n\n    Examples\n    --------\n    >>> y = np.reshape(np.arange(2*3*4), (2,3,4))\n    >>> y\n    array([[[ 0,  1,  2,  3],\n            [ 4,  5,  6,  7],\n            [ 8,  9, 10, 11]],\n           [[12, 13, 14, 15],\n            [16, 17, 18, 19],\n            [20, 21, 22, 23]]])\n    >>> y.strides\n    (48, 16, 4)\n    >>> y[1,1,1]\n    17\n    >>> offset=sum(y.strides * np.array((1,1,1)))\n    >>> offset/y.itemsize\n    17\n\n    >>> x = np.reshape(np.arange(5*6*7*8), (5,6,7,8)).transpose(2,3,1,0)\n    >>> x.strides\n    (32, 4, 224, 1344)\n    >>> i = np.array([3,5,2,2])\n    >>> offset = sum(i * x.strides)\n    >>> x[3,5,2,2]\n    813\n    >>> offset / x.itemsize\n    813\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2949, 48), tuple_21604, str_21606)

# Processing the call keyword arguments (line 2949)
kwargs_21607 = {}
# Getting the type of 'add_newdoc' (line 2949)
add_newdoc_21601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2949, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 2949)
add_newdoc_call_result_21608 = invoke(stypy.reporting.localization.Localization(__file__, 2949, 0), add_newdoc_21601, *[str_21602, str_21603, tuple_21604], **kwargs_21607)


# Call to add_newdoc(...): (line 3011)
# Processing the call arguments (line 3011)
str_21610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3011, 11), 'str', 'numpy.core.multiarray')
str_21611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3011, 36), 'str', 'ndarray')

# Obtaining an instance of the builtin type 'tuple' (line 3011)
tuple_21612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3011, 48), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 3011)
# Adding element type (line 3011)
str_21613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3011, 48), 'str', 'T')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3011, 48), tuple_21612, str_21613)
# Adding element type (line 3011)
str_21614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3031, (-1)), 'str', '\n    Same as self.transpose(), except that self is returned if\n    self.ndim < 2.\n\n    Examples\n    --------\n    >>> x = np.array([[1.,2.],[3.,4.]])\n    >>> x\n    array([[ 1.,  2.],\n           [ 3.,  4.]])\n    >>> x.T\n    array([[ 1.,  3.],\n           [ 2.,  4.]])\n    >>> x = np.array([1.,2.,3.,4.])\n    >>> x\n    array([ 1.,  2.,  3.,  4.])\n    >>> x.T\n    array([ 1.,  2.,  3.,  4.])\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3011, 48), tuple_21612, str_21614)

# Processing the call keyword arguments (line 3011)
kwargs_21615 = {}
# Getting the type of 'add_newdoc' (line 3011)
add_newdoc_21609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3011, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 3011)
add_newdoc_call_result_21616 = invoke(stypy.reporting.localization.Localization(__file__, 3011, 0), add_newdoc_21609, *[str_21610, str_21611, tuple_21612], **kwargs_21615)


# Call to add_newdoc(...): (line 3041)
# Processing the call arguments (line 3041)
str_21618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3041, 11), 'str', 'numpy.core.multiarray')
str_21619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3041, 36), 'str', 'ndarray')

# Obtaining an instance of the builtin type 'tuple' (line 3041)
tuple_21620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3041, 48), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 3041)
# Adding element type (line 3041)
str_21621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3041, 48), 'str', '__array__')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3041, 48), tuple_21620, str_21621)
# Adding element type (line 3041)
str_21622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3048, (-1)), 'str', ' a.__array__(|dtype) -> reference if type unchanged, copy otherwise.\n\n    Returns either a new reference to self if dtype is not given or a new array\n    of provided data type if dtype is different from the current dtype of the\n    array.\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3041, 48), tuple_21620, str_21622)

# Processing the call keyword arguments (line 3041)
kwargs_21623 = {}
# Getting the type of 'add_newdoc' (line 3041)
add_newdoc_21617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3041, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 3041)
add_newdoc_call_result_21624 = invoke(stypy.reporting.localization.Localization(__file__, 3041, 0), add_newdoc_21617, *[str_21618, str_21619, tuple_21620], **kwargs_21623)


# Call to add_newdoc(...): (line 3051)
# Processing the call arguments (line 3051)
str_21626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3051, 11), 'str', 'numpy.core.multiarray')
str_21627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3051, 36), 'str', 'ndarray')

# Obtaining an instance of the builtin type 'tuple' (line 3051)
tuple_21628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3051, 48), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 3051)
# Adding element type (line 3051)
str_21629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3051, 48), 'str', '__array_prepare__')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3051, 48), tuple_21628, str_21629)
# Adding element type (line 3051)
str_21630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3054, (-1)), 'str', 'a.__array_prepare__(obj) -> Object of same type as ndarray object obj.\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3051, 48), tuple_21628, str_21630)

# Processing the call keyword arguments (line 3051)
kwargs_21631 = {}
# Getting the type of 'add_newdoc' (line 3051)
add_newdoc_21625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3051, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 3051)
add_newdoc_call_result_21632 = invoke(stypy.reporting.localization.Localization(__file__, 3051, 0), add_newdoc_21625, *[str_21626, str_21627, tuple_21628], **kwargs_21631)


# Call to add_newdoc(...): (line 3057)
# Processing the call arguments (line 3057)
str_21634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3057, 11), 'str', 'numpy.core.multiarray')
str_21635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3057, 36), 'str', 'ndarray')

# Obtaining an instance of the builtin type 'tuple' (line 3057)
tuple_21636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3057, 48), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 3057)
# Adding element type (line 3057)
str_21637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3057, 48), 'str', '__array_wrap__')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3057, 48), tuple_21636, str_21637)
# Adding element type (line 3057)
str_21638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3060, (-1)), 'str', 'a.__array_wrap__(obj) -> Object of same type as ndarray object a.\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3057, 48), tuple_21636, str_21638)

# Processing the call keyword arguments (line 3057)
kwargs_21639 = {}
# Getting the type of 'add_newdoc' (line 3057)
add_newdoc_21633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3057, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 3057)
add_newdoc_call_result_21640 = invoke(stypy.reporting.localization.Localization(__file__, 3057, 0), add_newdoc_21633, *[str_21634, str_21635, tuple_21636], **kwargs_21639)


# Call to add_newdoc(...): (line 3063)
# Processing the call arguments (line 3063)
str_21642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3063, 11), 'str', 'numpy.core.multiarray')
str_21643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3063, 36), 'str', 'ndarray')

# Obtaining an instance of the builtin type 'tuple' (line 3063)
tuple_21644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3063, 48), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 3063)
# Adding element type (line 3063)
str_21645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3063, 48), 'str', '__copy__')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3063, 48), tuple_21644, str_21645)
# Adding element type (line 3063)
str_21646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3076, (-1)), 'str', "a.__copy__([order])\n\n    Return a copy of the array.\n\n    Parameters\n    ----------\n    order : {'C', 'F', 'A'}, optional\n        If order is 'C' (False) then the result is contiguous (default).\n        If order is 'Fortran' (True) then the result has fortran order.\n        If order is 'Any' (None) then the result has fortran order\n        only if the array already is in fortran order.\n\n    ")
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3063, 48), tuple_21644, str_21646)

# Processing the call keyword arguments (line 3063)
kwargs_21647 = {}
# Getting the type of 'add_newdoc' (line 3063)
add_newdoc_21641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3063, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 3063)
add_newdoc_call_result_21648 = invoke(stypy.reporting.localization.Localization(__file__, 3063, 0), add_newdoc_21641, *[str_21642, str_21643, tuple_21644], **kwargs_21647)


# Call to add_newdoc(...): (line 3079)
# Processing the call arguments (line 3079)
str_21650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3079, 11), 'str', 'numpy.core.multiarray')
str_21651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3079, 36), 'str', 'ndarray')

# Obtaining an instance of the builtin type 'tuple' (line 3079)
tuple_21652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3079, 48), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 3079)
# Adding element type (line 3079)
str_21653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3079, 48), 'str', '__deepcopy__')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3079, 48), tuple_21652, str_21653)
# Adding element type (line 3079)
str_21654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3084, (-1)), 'str', 'a.__deepcopy__() -> Deep copy of array.\n\n    Used if copy.deepcopy is called on an array.\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3079, 48), tuple_21652, str_21654)

# Processing the call keyword arguments (line 3079)
kwargs_21655 = {}
# Getting the type of 'add_newdoc' (line 3079)
add_newdoc_21649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3079, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 3079)
add_newdoc_call_result_21656 = invoke(stypy.reporting.localization.Localization(__file__, 3079, 0), add_newdoc_21649, *[str_21650, str_21651, tuple_21652], **kwargs_21655)


# Call to add_newdoc(...): (line 3087)
# Processing the call arguments (line 3087)
str_21658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3087, 11), 'str', 'numpy.core.multiarray')
str_21659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3087, 36), 'str', 'ndarray')

# Obtaining an instance of the builtin type 'tuple' (line 3087)
tuple_21660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3087, 48), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 3087)
# Adding element type (line 3087)
str_21661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3087, 48), 'str', '__reduce__')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3087, 48), tuple_21660, str_21661)
# Adding element type (line 3087)
str_21662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3092, (-1)), 'str', 'a.__reduce__()\n\n    For pickling.\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3087, 48), tuple_21660, str_21662)

# Processing the call keyword arguments (line 3087)
kwargs_21663 = {}
# Getting the type of 'add_newdoc' (line 3087)
add_newdoc_21657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3087, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 3087)
add_newdoc_call_result_21664 = invoke(stypy.reporting.localization.Localization(__file__, 3087, 0), add_newdoc_21657, *[str_21658, str_21659, tuple_21660], **kwargs_21663)


# Call to add_newdoc(...): (line 3095)
# Processing the call arguments (line 3095)
str_21666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3095, 11), 'str', 'numpy.core.multiarray')
str_21667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3095, 36), 'str', 'ndarray')

# Obtaining an instance of the builtin type 'tuple' (line 3095)
tuple_21668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3095, 48), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 3095)
# Adding element type (line 3095)
str_21669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3095, 48), 'str', '__setstate__')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3095, 48), tuple_21668, str_21669)
# Adding element type (line 3095)
str_21670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3110, (-1)), 'str', "a.__setstate__(version, shape, dtype, isfortran, rawdata)\n\n    For unpickling.\n\n    Parameters\n    ----------\n    version : int\n        optional pickle version. If omitted defaults to 0.\n    shape : tuple\n    dtype : data-type\n    isFortran : bool\n    rawdata : string or list\n        a binary string with the data (or a list if 'a' is an object array)\n\n    ")
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3095, 48), tuple_21668, str_21670)

# Processing the call keyword arguments (line 3095)
kwargs_21671 = {}
# Getting the type of 'add_newdoc' (line 3095)
add_newdoc_21665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3095, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 3095)
add_newdoc_call_result_21672 = invoke(stypy.reporting.localization.Localization(__file__, 3095, 0), add_newdoc_21665, *[str_21666, str_21667, tuple_21668], **kwargs_21671)


# Call to add_newdoc(...): (line 3113)
# Processing the call arguments (line 3113)
str_21674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3113, 11), 'str', 'numpy.core.multiarray')
str_21675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3113, 36), 'str', 'ndarray')

# Obtaining an instance of the builtin type 'tuple' (line 3113)
tuple_21676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3113, 48), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 3113)
# Adding element type (line 3113)
str_21677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3113, 48), 'str', 'all')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3113, 48), tuple_21676, str_21677)
# Adding element type (line 3113)
str_21678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3125, (-1)), 'str', '\n    a.all(axis=None, out=None, keepdims=False)\n\n    Returns True if all elements evaluate to True.\n\n    Refer to `numpy.all` for full documentation.\n\n    See Also\n    --------\n    numpy.all : equivalent function\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3113, 48), tuple_21676, str_21678)

# Processing the call keyword arguments (line 3113)
kwargs_21679 = {}
# Getting the type of 'add_newdoc' (line 3113)
add_newdoc_21673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3113, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 3113)
add_newdoc_call_result_21680 = invoke(stypy.reporting.localization.Localization(__file__, 3113, 0), add_newdoc_21673, *[str_21674, str_21675, tuple_21676], **kwargs_21679)


# Call to add_newdoc(...): (line 3128)
# Processing the call arguments (line 3128)
str_21682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3128, 11), 'str', 'numpy.core.multiarray')
str_21683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3128, 36), 'str', 'ndarray')

# Obtaining an instance of the builtin type 'tuple' (line 3128)
tuple_21684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3128, 48), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 3128)
# Adding element type (line 3128)
str_21685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3128, 48), 'str', 'any')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3128, 48), tuple_21684, str_21685)
# Adding element type (line 3128)
str_21686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3140, (-1)), 'str', '\n    a.any(axis=None, out=None, keepdims=False)\n\n    Returns True if any of the elements of `a` evaluate to True.\n\n    Refer to `numpy.any` for full documentation.\n\n    See Also\n    --------\n    numpy.any : equivalent function\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3128, 48), tuple_21684, str_21686)

# Processing the call keyword arguments (line 3128)
kwargs_21687 = {}
# Getting the type of 'add_newdoc' (line 3128)
add_newdoc_21681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3128, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 3128)
add_newdoc_call_result_21688 = invoke(stypy.reporting.localization.Localization(__file__, 3128, 0), add_newdoc_21681, *[str_21682, str_21683, tuple_21684], **kwargs_21687)


# Call to add_newdoc(...): (line 3143)
# Processing the call arguments (line 3143)
str_21690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3143, 11), 'str', 'numpy.core.multiarray')
str_21691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3143, 36), 'str', 'ndarray')

# Obtaining an instance of the builtin type 'tuple' (line 3143)
tuple_21692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3143, 48), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 3143)
# Adding element type (line 3143)
str_21693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3143, 48), 'str', 'argmax')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3143, 48), tuple_21692, str_21693)
# Adding element type (line 3143)
str_21694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3155, (-1)), 'str', '\n    a.argmax(axis=None, out=None)\n\n    Return indices of the maximum values along the given axis.\n\n    Refer to `numpy.argmax` for full documentation.\n\n    See Also\n    --------\n    numpy.argmax : equivalent function\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3143, 48), tuple_21692, str_21694)

# Processing the call keyword arguments (line 3143)
kwargs_21695 = {}
# Getting the type of 'add_newdoc' (line 3143)
add_newdoc_21689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3143, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 3143)
add_newdoc_call_result_21696 = invoke(stypy.reporting.localization.Localization(__file__, 3143, 0), add_newdoc_21689, *[str_21690, str_21691, tuple_21692], **kwargs_21695)


# Call to add_newdoc(...): (line 3158)
# Processing the call arguments (line 3158)
str_21698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3158, 11), 'str', 'numpy.core.multiarray')
str_21699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3158, 36), 'str', 'ndarray')

# Obtaining an instance of the builtin type 'tuple' (line 3158)
tuple_21700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3158, 48), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 3158)
# Adding element type (line 3158)
str_21701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3158, 48), 'str', 'argmin')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3158, 48), tuple_21700, str_21701)
# Adding element type (line 3158)
str_21702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3170, (-1)), 'str', '\n    a.argmin(axis=None, out=None)\n\n    Return indices of the minimum values along the given axis of `a`.\n\n    Refer to `numpy.argmin` for detailed documentation.\n\n    See Also\n    --------\n    numpy.argmin : equivalent function\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3158, 48), tuple_21700, str_21702)

# Processing the call keyword arguments (line 3158)
kwargs_21703 = {}
# Getting the type of 'add_newdoc' (line 3158)
add_newdoc_21697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3158, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 3158)
add_newdoc_call_result_21704 = invoke(stypy.reporting.localization.Localization(__file__, 3158, 0), add_newdoc_21697, *[str_21698, str_21699, tuple_21700], **kwargs_21703)


# Call to add_newdoc(...): (line 3173)
# Processing the call arguments (line 3173)
str_21706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3173, 11), 'str', 'numpy.core.multiarray')
str_21707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3173, 36), 'str', 'ndarray')

# Obtaining an instance of the builtin type 'tuple' (line 3173)
tuple_21708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3173, 48), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 3173)
# Adding element type (line 3173)
str_21709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3173, 48), 'str', 'argsort')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3173, 48), tuple_21708, str_21709)
# Adding element type (line 3173)
str_21710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3185, (-1)), 'str', "\n    a.argsort(axis=-1, kind='quicksort', order=None)\n\n    Returns the indices that would sort this array.\n\n    Refer to `numpy.argsort` for full documentation.\n\n    See Also\n    --------\n    numpy.argsort : equivalent function\n\n    ")
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3173, 48), tuple_21708, str_21710)

# Processing the call keyword arguments (line 3173)
kwargs_21711 = {}
# Getting the type of 'add_newdoc' (line 3173)
add_newdoc_21705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3173, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 3173)
add_newdoc_call_result_21712 = invoke(stypy.reporting.localization.Localization(__file__, 3173, 0), add_newdoc_21705, *[str_21706, str_21707, tuple_21708], **kwargs_21711)


# Call to add_newdoc(...): (line 3188)
# Processing the call arguments (line 3188)
str_21714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3188, 11), 'str', 'numpy.core.multiarray')
str_21715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3188, 36), 'str', 'ndarray')

# Obtaining an instance of the builtin type 'tuple' (line 3188)
tuple_21716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3188, 48), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 3188)
# Adding element type (line 3188)
str_21717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3188, 48), 'str', 'argpartition')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3188, 48), tuple_21716, str_21717)
# Adding element type (line 3188)
str_21718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3202, (-1)), 'str', "\n    a.argpartition(kth, axis=-1, kind='introselect', order=None)\n\n    Returns the indices that would partition this array.\n\n    Refer to `numpy.argpartition` for full documentation.\n\n    .. versionadded:: 1.8.0\n\n    See Also\n    --------\n    numpy.argpartition : equivalent function\n\n    ")
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3188, 48), tuple_21716, str_21718)

# Processing the call keyword arguments (line 3188)
kwargs_21719 = {}
# Getting the type of 'add_newdoc' (line 3188)
add_newdoc_21713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3188, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 3188)
add_newdoc_call_result_21720 = invoke(stypy.reporting.localization.Localization(__file__, 3188, 0), add_newdoc_21713, *[str_21714, str_21715, tuple_21716], **kwargs_21719)


# Call to add_newdoc(...): (line 3205)
# Processing the call arguments (line 3205)
str_21722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3205, 11), 'str', 'numpy.core.multiarray')
str_21723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3205, 36), 'str', 'ndarray')

# Obtaining an instance of the builtin type 'tuple' (line 3205)
tuple_21724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3205, 48), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 3205)
# Adding element type (line 3205)
str_21725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3205, 48), 'str', 'astype')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3205, 48), tuple_21724, str_21725)
# Adding element type (line 3205)
str_21726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3271, (-1)), 'str', "\n    a.astype(dtype, order='K', casting='unsafe', subok=True, copy=True)\n\n    Copy of the array, cast to a specified type.\n\n    Parameters\n    ----------\n    dtype : str or dtype\n        Typecode or data-type to which the array is cast.\n    order : {'C', 'F', 'A', 'K'}, optional\n        Controls the memory layout order of the result.\n        'C' means C order, 'F' means Fortran order, 'A'\n        means 'F' order if all the arrays are Fortran contiguous,\n        'C' order otherwise, and 'K' means as close to the\n        order the array elements appear in memory as possible.\n        Default is 'K'.\n    casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional\n        Controls what kind of data casting may occur. Defaults to 'unsafe'\n        for backwards compatibility.\n\n          * 'no' means the data types should not be cast at all.\n          * 'equiv' means only byte-order changes are allowed.\n          * 'safe' means only casts which can preserve values are allowed.\n          * 'same_kind' means only safe casts or casts within a kind,\n            like float64 to float32, are allowed.\n          * 'unsafe' means any data conversions may be done.\n    subok : bool, optional\n        If True, then sub-classes will be passed-through (default), otherwise\n        the returned array will be forced to be a base-class array.\n    copy : bool, optional\n        By default, astype always returns a newly allocated array. If this\n        is set to false, and the `dtype`, `order`, and `subok`\n        requirements are satisfied, the input array is returned instead\n        of a copy.\n\n    Returns\n    -------\n    arr_t : ndarray\n        Unless `copy` is False and the other conditions for returning the input\n        array are satisfied (see description for `copy` input parameter), `arr_t`\n        is a new array of the same shape as the input array, with dtype, order\n        given by `dtype`, `order`.\n\n    Notes\n    -----\n    Starting in NumPy 1.9, astype method now returns an error if the string\n    dtype to cast to is not long enough in 'safe' casting mode to hold the max\n    value of integer/float array that is being casted. Previously the casting\n    was allowed even if the result was truncated.\n\n    Raises\n    ------\n    ComplexWarning\n        When casting from complex to float or int. To avoid this,\n        one should use ``a.real.astype(t)``.\n\n    Examples\n    --------\n    >>> x = np.array([1, 2, 2.5])\n    >>> x\n    array([ 1. ,  2. ,  2.5])\n\n    >>> x.astype(int)\n    array([1, 2, 2])\n\n    ")
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3205, 48), tuple_21724, str_21726)

# Processing the call keyword arguments (line 3205)
kwargs_21727 = {}
# Getting the type of 'add_newdoc' (line 3205)
add_newdoc_21721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3205, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 3205)
add_newdoc_call_result_21728 = invoke(stypy.reporting.localization.Localization(__file__, 3205, 0), add_newdoc_21721, *[str_21722, str_21723, tuple_21724], **kwargs_21727)


# Call to add_newdoc(...): (line 3274)
# Processing the call arguments (line 3274)
str_21730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3274, 11), 'str', 'numpy.core.multiarray')
str_21731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3274, 36), 'str', 'ndarray')

# Obtaining an instance of the builtin type 'tuple' (line 3274)
tuple_21732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3274, 48), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 3274)
# Adding element type (line 3274)
str_21733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3274, 48), 'str', 'byteswap')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3274, 48), tuple_21732, str_21733)
# Adding element type (line 3274)
str_21734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3311, (-1)), 'str', "\n    a.byteswap(inplace)\n\n    Swap the bytes of the array elements\n\n    Toggle between low-endian and big-endian data representation by\n    returning a byteswapped array, optionally swapped in-place.\n\n    Parameters\n    ----------\n    inplace : bool, optional\n        If ``True``, swap bytes in-place, default is ``False``.\n\n    Returns\n    -------\n    out : ndarray\n        The byteswapped array. If `inplace` is ``True``, this is\n        a view to self.\n\n    Examples\n    --------\n    >>> A = np.array([1, 256, 8755], dtype=np.int16)\n    >>> map(hex, A)\n    ['0x1', '0x100', '0x2233']\n    >>> A.byteswap(True)\n    array([  256,     1, 13090], dtype=int16)\n    >>> map(hex, A)\n    ['0x100', '0x1', '0x3322']\n\n    Arrays of strings are not swapped\n\n    >>> A = np.array(['ceg', 'fac'])\n    >>> A.byteswap()\n    array(['ceg', 'fac'],\n          dtype='|S3')\n\n    ")
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3274, 48), tuple_21732, str_21734)

# Processing the call keyword arguments (line 3274)
kwargs_21735 = {}
# Getting the type of 'add_newdoc' (line 3274)
add_newdoc_21729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3274, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 3274)
add_newdoc_call_result_21736 = invoke(stypy.reporting.localization.Localization(__file__, 3274, 0), add_newdoc_21729, *[str_21730, str_21731, tuple_21732], **kwargs_21735)


# Call to add_newdoc(...): (line 3314)
# Processing the call arguments (line 3314)
str_21738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3314, 11), 'str', 'numpy.core.multiarray')
str_21739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3314, 36), 'str', 'ndarray')

# Obtaining an instance of the builtin type 'tuple' (line 3314)
tuple_21740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3314, 48), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 3314)
# Adding element type (line 3314)
str_21741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3314, 48), 'str', 'choose')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3314, 48), tuple_21740, str_21741)
# Adding element type (line 3314)
str_21742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3326, (-1)), 'str', "\n    a.choose(choices, out=None, mode='raise')\n\n    Use an index array to construct a new array from a set of choices.\n\n    Refer to `numpy.choose` for full documentation.\n\n    See Also\n    --------\n    numpy.choose : equivalent function\n\n    ")
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3314, 48), tuple_21740, str_21742)

# Processing the call keyword arguments (line 3314)
kwargs_21743 = {}
# Getting the type of 'add_newdoc' (line 3314)
add_newdoc_21737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3314, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 3314)
add_newdoc_call_result_21744 = invoke(stypy.reporting.localization.Localization(__file__, 3314, 0), add_newdoc_21737, *[str_21738, str_21739, tuple_21740], **kwargs_21743)


# Call to add_newdoc(...): (line 3329)
# Processing the call arguments (line 3329)
str_21746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3329, 11), 'str', 'numpy.core.multiarray')
str_21747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3329, 36), 'str', 'ndarray')

# Obtaining an instance of the builtin type 'tuple' (line 3329)
tuple_21748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3329, 48), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 3329)
# Adding element type (line 3329)
str_21749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3329, 48), 'str', 'clip')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3329, 48), tuple_21748, str_21749)
# Adding element type (line 3329)
str_21750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3342, (-1)), 'str', '\n    a.clip(min=None, max=None, out=None)\n\n    Return an array whose values are limited to ``[min, max]``.\n    One of max or min must be given.\n\n    Refer to `numpy.clip` for full documentation.\n\n    See Also\n    --------\n    numpy.clip : equivalent function\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3329, 48), tuple_21748, str_21750)

# Processing the call keyword arguments (line 3329)
kwargs_21751 = {}
# Getting the type of 'add_newdoc' (line 3329)
add_newdoc_21745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3329, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 3329)
add_newdoc_call_result_21752 = invoke(stypy.reporting.localization.Localization(__file__, 3329, 0), add_newdoc_21745, *[str_21746, str_21747, tuple_21748], **kwargs_21751)


# Call to add_newdoc(...): (line 3345)
# Processing the call arguments (line 3345)
str_21754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3345, 11), 'str', 'numpy.core.multiarray')
str_21755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3345, 36), 'str', 'ndarray')

# Obtaining an instance of the builtin type 'tuple' (line 3345)
tuple_21756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3345, 48), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 3345)
# Adding element type (line 3345)
str_21757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3345, 48), 'str', 'compress')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3345, 48), tuple_21756, str_21757)
# Adding element type (line 3345)
str_21758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3357, (-1)), 'str', '\n    a.compress(condition, axis=None, out=None)\n\n    Return selected slices of this array along given axis.\n\n    Refer to `numpy.compress` for full documentation.\n\n    See Also\n    --------\n    numpy.compress : equivalent function\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3345, 48), tuple_21756, str_21758)

# Processing the call keyword arguments (line 3345)
kwargs_21759 = {}
# Getting the type of 'add_newdoc' (line 3345)
add_newdoc_21753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3345, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 3345)
add_newdoc_call_result_21760 = invoke(stypy.reporting.localization.Localization(__file__, 3345, 0), add_newdoc_21753, *[str_21754, str_21755, tuple_21756], **kwargs_21759)


# Call to add_newdoc(...): (line 3360)
# Processing the call arguments (line 3360)
str_21762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3360, 11), 'str', 'numpy.core.multiarray')
str_21763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3360, 36), 'str', 'ndarray')

# Obtaining an instance of the builtin type 'tuple' (line 3360)
tuple_21764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3360, 48), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 3360)
# Adding element type (line 3360)
str_21765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3360, 48), 'str', 'conj')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3360, 48), tuple_21764, str_21765)
# Adding element type (line 3360)
str_21766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3372, (-1)), 'str', '\n    a.conj()\n\n    Complex-conjugate all elements.\n\n    Refer to `numpy.conjugate` for full documentation.\n\n    See Also\n    --------\n    numpy.conjugate : equivalent function\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3360, 48), tuple_21764, str_21766)

# Processing the call keyword arguments (line 3360)
kwargs_21767 = {}
# Getting the type of 'add_newdoc' (line 3360)
add_newdoc_21761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3360, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 3360)
add_newdoc_call_result_21768 = invoke(stypy.reporting.localization.Localization(__file__, 3360, 0), add_newdoc_21761, *[str_21762, str_21763, tuple_21764], **kwargs_21767)


# Call to add_newdoc(...): (line 3375)
# Processing the call arguments (line 3375)
str_21770 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3375, 11), 'str', 'numpy.core.multiarray')
str_21771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3375, 36), 'str', 'ndarray')

# Obtaining an instance of the builtin type 'tuple' (line 3375)
tuple_21772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3375, 48), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 3375)
# Adding element type (line 3375)
str_21773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3375, 48), 'str', 'conjugate')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3375, 48), tuple_21772, str_21773)
# Adding element type (line 3375)
str_21774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3387, (-1)), 'str', '\n    a.conjugate()\n\n    Return the complex conjugate, element-wise.\n\n    Refer to `numpy.conjugate` for full documentation.\n\n    See Also\n    --------\n    numpy.conjugate : equivalent function\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3375, 48), tuple_21772, str_21774)

# Processing the call keyword arguments (line 3375)
kwargs_21775 = {}
# Getting the type of 'add_newdoc' (line 3375)
add_newdoc_21769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3375, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 3375)
add_newdoc_call_result_21776 = invoke(stypy.reporting.localization.Localization(__file__, 3375, 0), add_newdoc_21769, *[str_21770, str_21771, tuple_21772], **kwargs_21775)


# Call to add_newdoc(...): (line 3390)
# Processing the call arguments (line 3390)
str_21778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3390, 11), 'str', 'numpy.core.multiarray')
str_21779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3390, 36), 'str', 'ndarray')

# Obtaining an instance of the builtin type 'tuple' (line 3390)
tuple_21780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3390, 48), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 3390)
# Adding element type (line 3390)
str_21781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3390, 48), 'str', 'copy')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3390, 48), tuple_21780, str_21781)
# Adding element type (line 3390)
str_21782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3430, (-1)), 'str', "\n    a.copy(order='C')\n\n    Return a copy of the array.\n\n    Parameters\n    ----------\n    order : {'C', 'F', 'A', 'K'}, optional\n        Controls the memory layout of the copy. 'C' means C-order,\n        'F' means F-order, 'A' means 'F' if `a` is Fortran contiguous,\n        'C' otherwise. 'K' means match the layout of `a` as closely\n        as possible. (Note that this function and :func:numpy.copy are very\n        similar, but have different default values for their order=\n        arguments.)\n\n    See also\n    --------\n    numpy.copy\n    numpy.copyto\n\n    Examples\n    --------\n    >>> x = np.array([[1,2,3],[4,5,6]], order='F')\n\n    >>> y = x.copy()\n\n    >>> x.fill(0)\n\n    >>> x\n    array([[0, 0, 0],\n           [0, 0, 0]])\n\n    >>> y\n    array([[1, 2, 3],\n           [4, 5, 6]])\n\n    >>> y.flags['C_CONTIGUOUS']\n    True\n\n    ")
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3390, 48), tuple_21780, str_21782)

# Processing the call keyword arguments (line 3390)
kwargs_21783 = {}
# Getting the type of 'add_newdoc' (line 3390)
add_newdoc_21777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3390, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 3390)
add_newdoc_call_result_21784 = invoke(stypy.reporting.localization.Localization(__file__, 3390, 0), add_newdoc_21777, *[str_21778, str_21779, tuple_21780], **kwargs_21783)


# Call to add_newdoc(...): (line 3433)
# Processing the call arguments (line 3433)
str_21786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3433, 11), 'str', 'numpy.core.multiarray')
str_21787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3433, 36), 'str', 'ndarray')

# Obtaining an instance of the builtin type 'tuple' (line 3433)
tuple_21788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3433, 48), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 3433)
# Adding element type (line 3433)
str_21789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3433, 48), 'str', 'cumprod')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3433, 48), tuple_21788, str_21789)
# Adding element type (line 3433)
str_21790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3445, (-1)), 'str', '\n    a.cumprod(axis=None, dtype=None, out=None)\n\n    Return the cumulative product of the elements along the given axis.\n\n    Refer to `numpy.cumprod` for full documentation.\n\n    See Also\n    --------\n    numpy.cumprod : equivalent function\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3433, 48), tuple_21788, str_21790)

# Processing the call keyword arguments (line 3433)
kwargs_21791 = {}
# Getting the type of 'add_newdoc' (line 3433)
add_newdoc_21785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3433, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 3433)
add_newdoc_call_result_21792 = invoke(stypy.reporting.localization.Localization(__file__, 3433, 0), add_newdoc_21785, *[str_21786, str_21787, tuple_21788], **kwargs_21791)


# Call to add_newdoc(...): (line 3448)
# Processing the call arguments (line 3448)
str_21794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3448, 11), 'str', 'numpy.core.multiarray')
str_21795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3448, 36), 'str', 'ndarray')

# Obtaining an instance of the builtin type 'tuple' (line 3448)
tuple_21796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3448, 48), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 3448)
# Adding element type (line 3448)
str_21797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3448, 48), 'str', 'cumsum')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3448, 48), tuple_21796, str_21797)
# Adding element type (line 3448)
str_21798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3460, (-1)), 'str', '\n    a.cumsum(axis=None, dtype=None, out=None)\n\n    Return the cumulative sum of the elements along the given axis.\n\n    Refer to `numpy.cumsum` for full documentation.\n\n    See Also\n    --------\n    numpy.cumsum : equivalent function\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3448, 48), tuple_21796, str_21798)

# Processing the call keyword arguments (line 3448)
kwargs_21799 = {}
# Getting the type of 'add_newdoc' (line 3448)
add_newdoc_21793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3448, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 3448)
add_newdoc_call_result_21800 = invoke(stypy.reporting.localization.Localization(__file__, 3448, 0), add_newdoc_21793, *[str_21794, str_21795, tuple_21796], **kwargs_21799)


# Call to add_newdoc(...): (line 3463)
# Processing the call arguments (line 3463)
str_21802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3463, 11), 'str', 'numpy.core.multiarray')
str_21803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3463, 36), 'str', 'ndarray')

# Obtaining an instance of the builtin type 'tuple' (line 3463)
tuple_21804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3463, 48), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 3463)
# Adding element type (line 3463)
str_21805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3463, 48), 'str', 'diagonal')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3463, 48), tuple_21804, str_21805)
# Adding element type (line 3463)
str_21806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3477, (-1)), 'str', '\n    a.diagonal(offset=0, axis1=0, axis2=1)\n\n    Return specified diagonals. In NumPy 1.9 the returned array is a\n    read-only view instead of a copy as in previous NumPy versions.  In\n    a future version the read-only restriction will be removed.\n\n    Refer to :func:`numpy.diagonal` for full documentation.\n\n    See Also\n    --------\n    numpy.diagonal : equivalent function\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3463, 48), tuple_21804, str_21806)

# Processing the call keyword arguments (line 3463)
kwargs_21807 = {}
# Getting the type of 'add_newdoc' (line 3463)
add_newdoc_21801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3463, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 3463)
add_newdoc_call_result_21808 = invoke(stypy.reporting.localization.Localization(__file__, 3463, 0), add_newdoc_21801, *[str_21802, str_21803, tuple_21804], **kwargs_21807)


# Call to add_newdoc(...): (line 3480)
# Processing the call arguments (line 3480)
str_21810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3480, 11), 'str', 'numpy.core.multiarray')
str_21811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3480, 36), 'str', 'ndarray')

# Obtaining an instance of the builtin type 'tuple' (line 3480)
tuple_21812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3480, 48), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 3480)
# Adding element type (line 3480)
str_21813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3480, 48), 'str', 'dot')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3480, 48), tuple_21812, str_21813)
# Adding element type (line 3480)
str_21814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3506, (-1)), 'str', '\n    a.dot(b, out=None)\n\n    Dot product of two arrays.\n\n    Refer to `numpy.dot` for full documentation.\n\n    See Also\n    --------\n    numpy.dot : equivalent function\n\n    Examples\n    --------\n    >>> a = np.eye(2)\n    >>> b = np.ones((2, 2)) * 2\n    >>> a.dot(b)\n    array([[ 2.,  2.],\n           [ 2.,  2.]])\n\n    This array method can be conveniently chained:\n\n    >>> a.dot(b).dot(b)\n    array([[ 8.,  8.],\n           [ 8.,  8.]])\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3480, 48), tuple_21812, str_21814)

# Processing the call keyword arguments (line 3480)
kwargs_21815 = {}
# Getting the type of 'add_newdoc' (line 3480)
add_newdoc_21809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3480, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 3480)
add_newdoc_call_result_21816 = invoke(stypy.reporting.localization.Localization(__file__, 3480, 0), add_newdoc_21809, *[str_21810, str_21811, tuple_21812], **kwargs_21815)


# Call to add_newdoc(...): (line 3509)
# Processing the call arguments (line 3509)
str_21818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3509, 11), 'str', 'numpy.core.multiarray')
str_21819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3509, 36), 'str', 'ndarray')

# Obtaining an instance of the builtin type 'tuple' (line 3509)
tuple_21820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3509, 48), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 3509)
# Adding element type (line 3509)
str_21821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3509, 48), 'str', 'dump')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3509, 48), tuple_21820, str_21821)
# Adding element type (line 3509)
str_21822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3520, (-1)), 'str', 'a.dump(file)\n\n    Dump a pickle of the array to the specified file.\n    The array can be read back with pickle.load or numpy.load.\n\n    Parameters\n    ----------\n    file : str\n        A string naming the dump file.\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3509, 48), tuple_21820, str_21822)

# Processing the call keyword arguments (line 3509)
kwargs_21823 = {}
# Getting the type of 'add_newdoc' (line 3509)
add_newdoc_21817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3509, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 3509)
add_newdoc_call_result_21824 = invoke(stypy.reporting.localization.Localization(__file__, 3509, 0), add_newdoc_21817, *[str_21818, str_21819, tuple_21820], **kwargs_21823)


# Call to add_newdoc(...): (line 3523)
# Processing the call arguments (line 3523)
str_21826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3523, 11), 'str', 'numpy.core.multiarray')
str_21827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3523, 36), 'str', 'ndarray')

# Obtaining an instance of the builtin type 'tuple' (line 3523)
tuple_21828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3523, 48), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 3523)
# Adding element type (line 3523)
str_21829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3523, 48), 'str', 'dumps')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3523, 48), tuple_21828, str_21829)
# Adding element type (line 3523)
str_21830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3534, (-1)), 'str', '\n    a.dumps()\n\n    Returns the pickle of the array as a string.\n    pickle.loads or numpy.loads will convert the string back to an array.\n\n    Parameters\n    ----------\n    None\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3523, 48), tuple_21828, str_21830)

# Processing the call keyword arguments (line 3523)
kwargs_21831 = {}
# Getting the type of 'add_newdoc' (line 3523)
add_newdoc_21825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3523, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 3523)
add_newdoc_call_result_21832 = invoke(stypy.reporting.localization.Localization(__file__, 3523, 0), add_newdoc_21825, *[str_21826, str_21827, tuple_21828], **kwargs_21831)


# Call to add_newdoc(...): (line 3537)
# Processing the call arguments (line 3537)
str_21834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3537, 11), 'str', 'numpy.core.multiarray')
str_21835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3537, 36), 'str', 'ndarray')

# Obtaining an instance of the builtin type 'tuple' (line 3537)
tuple_21836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3537, 48), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 3537)
# Adding element type (line 3537)
str_21837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3537, 48), 'str', 'fill')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3537, 48), tuple_21836, str_21837)
# Adding element type (line 3537)
str_21838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3559, (-1)), 'str', '\n    a.fill(value)\n\n    Fill the array with a scalar value.\n\n    Parameters\n    ----------\n    value : scalar\n        All elements of `a` will be assigned this value.\n\n    Examples\n    --------\n    >>> a = np.array([1, 2])\n    >>> a.fill(0)\n    >>> a\n    array([0, 0])\n    >>> a = np.empty(2)\n    >>> a.fill(1)\n    >>> a\n    array([ 1.,  1.])\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3537, 48), tuple_21836, str_21838)

# Processing the call keyword arguments (line 3537)
kwargs_21839 = {}
# Getting the type of 'add_newdoc' (line 3537)
add_newdoc_21833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3537, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 3537)
add_newdoc_call_result_21840 = invoke(stypy.reporting.localization.Localization(__file__, 3537, 0), add_newdoc_21833, *[str_21834, str_21835, tuple_21836], **kwargs_21839)


# Call to add_newdoc(...): (line 3562)
# Processing the call arguments (line 3562)
str_21842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3562, 11), 'str', 'numpy.core.multiarray')
str_21843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3562, 36), 'str', 'ndarray')

# Obtaining an instance of the builtin type 'tuple' (line 3562)
tuple_21844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3562, 48), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 3562)
# Adding element type (line 3562)
str_21845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3562, 48), 'str', 'flatten')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3562, 48), tuple_21844, str_21845)
# Adding element type (line 3562)
str_21846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3597, (-1)), 'str', "\n    a.flatten(order='C')\n\n    Return a copy of the array collapsed into one dimension.\n\n    Parameters\n    ----------\n    order : {'C', 'F', 'A', 'K'}, optional\n        'C' means to flatten in row-major (C-style) order.\n        'F' means to flatten in column-major (Fortran-\n        style) order. 'A' means to flatten in column-major\n        order if `a` is Fortran *contiguous* in memory,\n        row-major order otherwise. 'K' means to flatten\n        `a` in the order the elements occur in memory.\n        The default is 'C'.\n\n    Returns\n    -------\n    y : ndarray\n        A copy of the input array, flattened to one dimension.\n\n    See Also\n    --------\n    ravel : Return a flattened array.\n    flat : A 1-D flat iterator over the array.\n\n    Examples\n    --------\n    >>> a = np.array([[1,2], [3,4]])\n    >>> a.flatten()\n    array([1, 2, 3, 4])\n    >>> a.flatten('F')\n    array([1, 3, 2, 4])\n\n    ")
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3562, 48), tuple_21844, str_21846)

# Processing the call keyword arguments (line 3562)
kwargs_21847 = {}
# Getting the type of 'add_newdoc' (line 3562)
add_newdoc_21841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3562, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 3562)
add_newdoc_call_result_21848 = invoke(stypy.reporting.localization.Localization(__file__, 3562, 0), add_newdoc_21841, *[str_21842, str_21843, tuple_21844], **kwargs_21847)


# Call to add_newdoc(...): (line 3600)
# Processing the call arguments (line 3600)
str_21850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3600, 11), 'str', 'numpy.core.multiarray')
str_21851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3600, 36), 'str', 'ndarray')

# Obtaining an instance of the builtin type 'tuple' (line 3600)
tuple_21852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3600, 48), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 3600)
# Adding element type (line 3600)
str_21853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3600, 48), 'str', 'getfield')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3600, 48), tuple_21852, str_21853)
# Adding element type (line 3600)
str_21854 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3639, (-1)), 'str', '\n    a.getfield(dtype, offset=0)\n\n    Returns a field of the given array as a certain type.\n\n    A field is a view of the array data with a given data-type. The values in\n    the view are determined by the given type and the offset into the current\n    array in bytes. The offset needs to be such that the view dtype fits in the\n    array dtype; for example an array of dtype complex128 has 16-byte elements.\n    If taking a view with a 32-bit integer (4 bytes), the offset needs to be\n    between 0 and 12 bytes.\n\n    Parameters\n    ----------\n    dtype : str or dtype\n        The data type of the view. The dtype size of the view can not be larger\n        than that of the array itself.\n    offset : int\n        Number of bytes to skip before beginning the element view.\n\n    Examples\n    --------\n    >>> x = np.diag([1.+1.j]*2)\n    >>> x[1, 1] = 2 + 4.j\n    >>> x\n    array([[ 1.+1.j,  0.+0.j],\n           [ 0.+0.j,  2.+4.j]])\n    >>> x.getfield(np.float64)\n    array([[ 1.,  0.],\n           [ 0.,  2.]])\n\n    By choosing an offset of 8 bytes we can select the complex part of the\n    array for our view:\n\n    >>> x.getfield(np.float64, offset=8)\n    array([[ 1.,  0.],\n       [ 0.,  4.]])\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3600, 48), tuple_21852, str_21854)

# Processing the call keyword arguments (line 3600)
kwargs_21855 = {}
# Getting the type of 'add_newdoc' (line 3600)
add_newdoc_21849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3600, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 3600)
add_newdoc_call_result_21856 = invoke(stypy.reporting.localization.Localization(__file__, 3600, 0), add_newdoc_21849, *[str_21850, str_21851, tuple_21852], **kwargs_21855)


# Call to add_newdoc(...): (line 3642)
# Processing the call arguments (line 3642)
str_21858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3642, 11), 'str', 'numpy.core.multiarray')
str_21859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3642, 36), 'str', 'ndarray')

# Obtaining an instance of the builtin type 'tuple' (line 3642)
tuple_21860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3642, 48), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 3642)
# Adding element type (line 3642)
str_21861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3642, 48), 'str', 'item')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3642, 48), tuple_21860, str_21861)
# Adding element type (line 3642)
str_21862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3697, (-1)), 'str', "\n    a.item(*args)\n\n    Copy an element of an array to a standard Python scalar and return it.\n\n    Parameters\n    ----------\n    \\*args : Arguments (variable number and type)\n\n        * none: in this case, the method only works for arrays\n          with one element (`a.size == 1`), which element is\n          copied into a standard Python scalar object and returned.\n\n        * int_type: this argument is interpreted as a flat index into\n          the array, specifying which element to copy and return.\n\n        * tuple of int_types: functions as does a single int_type argument,\n          except that the argument is interpreted as an nd-index into the\n          array.\n\n    Returns\n    -------\n    z : Standard Python scalar object\n        A copy of the specified element of the array as a suitable\n        Python scalar\n\n    Notes\n    -----\n    When the data type of `a` is longdouble or clongdouble, item() returns\n    a scalar array object because there is no available Python scalar that\n    would not lose information. Void arrays return a buffer object for item(),\n    unless fields are defined, in which case a tuple is returned.\n\n    `item` is very similar to a[args], except, instead of an array scalar,\n    a standard Python scalar is returned. This can be useful for speeding up\n    access to elements of the array and doing arithmetic on elements of the\n    array using Python's optimized math.\n\n    Examples\n    --------\n    >>> x = np.random.randint(9, size=(3, 3))\n    >>> x\n    array([[3, 1, 7],\n           [2, 8, 3],\n           [8, 5, 3]])\n    >>> x.item(3)\n    2\n    >>> x.item(7)\n    5\n    >>> x.item((0, 1))\n    1\n    >>> x.item((2, 2))\n    3\n\n    ")
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3642, 48), tuple_21860, str_21862)

# Processing the call keyword arguments (line 3642)
kwargs_21863 = {}
# Getting the type of 'add_newdoc' (line 3642)
add_newdoc_21857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3642, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 3642)
add_newdoc_call_result_21864 = invoke(stypy.reporting.localization.Localization(__file__, 3642, 0), add_newdoc_21857, *[str_21858, str_21859, tuple_21860], **kwargs_21863)


# Call to add_newdoc(...): (line 3700)
# Processing the call arguments (line 3700)
str_21866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3700, 11), 'str', 'numpy.core.multiarray')
str_21867 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3700, 36), 'str', 'ndarray')

# Obtaining an instance of the builtin type 'tuple' (line 3700)
tuple_21868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3700, 48), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 3700)
# Adding element type (line 3700)
str_21869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3700, 48), 'str', 'itemset')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3700, 48), tuple_21868, str_21869)
# Adding element type (line 3700)
str_21870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3743, (-1)), 'str', "\n    a.itemset(*args)\n\n    Insert scalar into an array (scalar is cast to array's dtype, if possible)\n\n    There must be at least 1 argument, and define the last argument\n    as *item*.  Then, ``a.itemset(*args)`` is equivalent to but faster\n    than ``a[args] = item``.  The item should be a scalar value and `args`\n    must select a single item in the array `a`.\n\n    Parameters\n    ----------\n    \\*args : Arguments\n        If one argument: a scalar, only used in case `a` is of size 1.\n        If two arguments: the last argument is the value to be set\n        and must be a scalar, the first argument specifies a single array\n        element location. It is either an int or a tuple.\n\n    Notes\n    -----\n    Compared to indexing syntax, `itemset` provides some speed increase\n    for placing a scalar into a particular location in an `ndarray`,\n    if you must do this.  However, generally this is discouraged:\n    among other problems, it complicates the appearance of the code.\n    Also, when using `itemset` (and `item`) inside a loop, be sure\n    to assign the methods to a local variable to avoid the attribute\n    look-up at each loop iteration.\n\n    Examples\n    --------\n    >>> x = np.random.randint(9, size=(3, 3))\n    >>> x\n    array([[3, 1, 7],\n           [2, 8, 3],\n           [8, 5, 3]])\n    >>> x.itemset(4, 0)\n    >>> x.itemset((2, 2), 9)\n    >>> x\n    array([[3, 1, 7],\n           [2, 0, 3],\n           [8, 5, 9]])\n\n    ")
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3700, 48), tuple_21868, str_21870)

# Processing the call keyword arguments (line 3700)
kwargs_21871 = {}
# Getting the type of 'add_newdoc' (line 3700)
add_newdoc_21865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3700, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 3700)
add_newdoc_call_result_21872 = invoke(stypy.reporting.localization.Localization(__file__, 3700, 0), add_newdoc_21865, *[str_21866, str_21867, tuple_21868], **kwargs_21871)


# Call to add_newdoc(...): (line 3746)
# Processing the call arguments (line 3746)
str_21874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3746, 11), 'str', 'numpy.core.multiarray')
str_21875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3746, 36), 'str', 'ndarray')

# Obtaining an instance of the builtin type 'tuple' (line 3746)
tuple_21876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3746, 48), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 3746)
# Adding element type (line 3746)
str_21877 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3746, 48), 'str', 'max')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3746, 48), tuple_21876, str_21877)
# Adding element type (line 3746)
str_21878 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3758, (-1)), 'str', '\n    a.max(axis=None, out=None)\n\n    Return the maximum along a given axis.\n\n    Refer to `numpy.amax` for full documentation.\n\n    See Also\n    --------\n    numpy.amax : equivalent function\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3746, 48), tuple_21876, str_21878)

# Processing the call keyword arguments (line 3746)
kwargs_21879 = {}
# Getting the type of 'add_newdoc' (line 3746)
add_newdoc_21873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3746, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 3746)
add_newdoc_call_result_21880 = invoke(stypy.reporting.localization.Localization(__file__, 3746, 0), add_newdoc_21873, *[str_21874, str_21875, tuple_21876], **kwargs_21879)


# Call to add_newdoc(...): (line 3761)
# Processing the call arguments (line 3761)
str_21882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3761, 11), 'str', 'numpy.core.multiarray')
str_21883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3761, 36), 'str', 'ndarray')

# Obtaining an instance of the builtin type 'tuple' (line 3761)
tuple_21884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3761, 48), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 3761)
# Adding element type (line 3761)
str_21885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3761, 48), 'str', 'mean')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3761, 48), tuple_21884, str_21885)
# Adding element type (line 3761)
str_21886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3773, (-1)), 'str', '\n    a.mean(axis=None, dtype=None, out=None, keepdims=False)\n\n    Returns the average of the array elements along given axis.\n\n    Refer to `numpy.mean` for full documentation.\n\n    See Also\n    --------\n    numpy.mean : equivalent function\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3761, 48), tuple_21884, str_21886)

# Processing the call keyword arguments (line 3761)
kwargs_21887 = {}
# Getting the type of 'add_newdoc' (line 3761)
add_newdoc_21881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3761, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 3761)
add_newdoc_call_result_21888 = invoke(stypy.reporting.localization.Localization(__file__, 3761, 0), add_newdoc_21881, *[str_21882, str_21883, tuple_21884], **kwargs_21887)


# Call to add_newdoc(...): (line 3776)
# Processing the call arguments (line 3776)
str_21890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3776, 11), 'str', 'numpy.core.multiarray')
str_21891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3776, 36), 'str', 'ndarray')

# Obtaining an instance of the builtin type 'tuple' (line 3776)
tuple_21892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3776, 48), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 3776)
# Adding element type (line 3776)
str_21893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3776, 48), 'str', 'min')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3776, 48), tuple_21892, str_21893)
# Adding element type (line 3776)
str_21894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3788, (-1)), 'str', '\n    a.min(axis=None, out=None, keepdims=False)\n\n    Return the minimum along a given axis.\n\n    Refer to `numpy.amin` for full documentation.\n\n    See Also\n    --------\n    numpy.amin : equivalent function\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3776, 48), tuple_21892, str_21894)

# Processing the call keyword arguments (line 3776)
kwargs_21895 = {}
# Getting the type of 'add_newdoc' (line 3776)
add_newdoc_21889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3776, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 3776)
add_newdoc_call_result_21896 = invoke(stypy.reporting.localization.Localization(__file__, 3776, 0), add_newdoc_21889, *[str_21890, str_21891, tuple_21892], **kwargs_21895)


# Call to add_newdoc(...): (line 3791)
# Processing the call arguments (line 3791)
str_21898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3791, 11), 'str', 'numpy.core.multiarray')
str_21899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3791, 36), 'str', 'shares_memory')
str_21900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3830, (-1)), 'str', '\n    shares_memory(a, b, max_work=None)\n\n    Determine if two arrays share memory\n\n    Parameters\n    ----------\n    a, b : ndarray\n        Input arrays\n    max_work : int, optional\n        Effort to spend on solving the overlap problem (maximum number\n        of candidate solutions to consider). The following special\n        values are recognized:\n\n        max_work=MAY_SHARE_EXACT  (default)\n            The problem is solved exactly. In this case, the function returns\n            True only if there is an element shared between the arrays.\n        max_work=MAY_SHARE_BOUNDS\n            Only the memory bounds of a and b are checked.\n\n    Raises\n    ------\n    numpy.TooHardError\n        Exceeded max_work.\n\n    Returns\n    -------\n    out : bool\n\n    See Also\n    --------\n    may_share_memory\n\n    Examples\n    --------\n    >>> np.may_share_memory(np.array([1,2]), np.array([5,8,9]))\n    False\n\n    ')
# Processing the call keyword arguments (line 3791)
kwargs_21901 = {}
# Getting the type of 'add_newdoc' (line 3791)
add_newdoc_21897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3791, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 3791)
add_newdoc_call_result_21902 = invoke(stypy.reporting.localization.Localization(__file__, 3791, 0), add_newdoc_21897, *[str_21898, str_21899, str_21900], **kwargs_21901)


# Call to add_newdoc(...): (line 3833)
# Processing the call arguments (line 3833)
str_21904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3833, 11), 'str', 'numpy.core.multiarray')
str_21905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3833, 36), 'str', 'may_share_memory')
str_21906 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3869, (-1)), 'str', '\n    may_share_memory(a, b, max_work=None)\n\n    Determine if two arrays might share memory\n\n    A return of True does not necessarily mean that the two arrays\n    share any element.  It just means that they *might*.\n\n    Only the memory bounds of a and b are checked by default.\n\n    Parameters\n    ----------\n    a, b : ndarray\n        Input arrays\n    max_work : int, optional\n        Effort to spend on solving the overlap problem.  See\n        `shares_memory` for details.  Default for ``may_share_memory``\n        is to do a bounds check.\n\n    Returns\n    -------\n    out : bool\n\n    See Also\n    --------\n    shares_memory\n\n    Examples\n    --------\n    >>> np.may_share_memory(np.array([1,2]), np.array([5,8,9]))\n    False\n    >>> x = np.zeros([3, 4])\n    >>> np.may_share_memory(x[:,0], x[:,1])\n    True\n\n    ')
# Processing the call keyword arguments (line 3833)
kwargs_21907 = {}
# Getting the type of 'add_newdoc' (line 3833)
add_newdoc_21903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3833, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 3833)
add_newdoc_call_result_21908 = invoke(stypy.reporting.localization.Localization(__file__, 3833, 0), add_newdoc_21903, *[str_21904, str_21905, str_21906], **kwargs_21907)


# Call to add_newdoc(...): (line 3872)
# Processing the call arguments (line 3872)
str_21910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3872, 11), 'str', 'numpy.core.multiarray')
str_21911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3872, 36), 'str', 'ndarray')

# Obtaining an instance of the builtin type 'tuple' (line 3872)
tuple_21912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3872, 48), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 3872)
# Adding element type (line 3872)
str_21913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3872, 48), 'str', 'newbyteorder')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3872, 48), tuple_21912, str_21913)
# Adding element type (line 3872)
str_21914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3911, (-1)), 'str', "\n    arr.newbyteorder(new_order='S')\n\n    Return the array with the same data viewed with a different byte order.\n\n    Equivalent to::\n\n        arr.view(arr.dtype.newbytorder(new_order))\n\n    Changes are also made in all fields and sub-arrays of the array data\n    type.\n\n\n\n    Parameters\n    ----------\n    new_order : string, optional\n        Byte order to force; a value from the byte order specifications\n        below. `new_order` codes can be any of:\n\n        * 'S' - swap dtype from current to opposite endian\n        * {'<', 'L'} - little endian\n        * {'>', 'B'} - big endian\n        * {'=', 'N'} - native order\n        * {'|', 'I'} - ignore (no change to byte order)\n\n        The default value ('S') results in swapping the current\n        byte order. The code does a case-insensitive check on the first\n        letter of `new_order` for the alternatives above.  For example,\n        any of 'B' or 'b' or 'biggish' are valid to specify big-endian.\n\n\n    Returns\n    -------\n    new_arr : array\n        New array object with the dtype reflecting given change to the\n        byte order.\n\n    ")
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3872, 48), tuple_21912, str_21914)

# Processing the call keyword arguments (line 3872)
kwargs_21915 = {}
# Getting the type of 'add_newdoc' (line 3872)
add_newdoc_21909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3872, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 3872)
add_newdoc_call_result_21916 = invoke(stypy.reporting.localization.Localization(__file__, 3872, 0), add_newdoc_21909, *[str_21910, str_21911, tuple_21912], **kwargs_21915)


# Call to add_newdoc(...): (line 3914)
# Processing the call arguments (line 3914)
str_21918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3914, 11), 'str', 'numpy.core.multiarray')
str_21919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3914, 36), 'str', 'ndarray')

# Obtaining an instance of the builtin type 'tuple' (line 3914)
tuple_21920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3914, 48), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 3914)
# Adding element type (line 3914)
str_21921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3914, 48), 'str', 'nonzero')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3914, 48), tuple_21920, str_21921)
# Adding element type (line 3914)
str_21922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3926, (-1)), 'str', '\n    a.nonzero()\n\n    Return the indices of the elements that are non-zero.\n\n    Refer to `numpy.nonzero` for full documentation.\n\n    See Also\n    --------\n    numpy.nonzero : equivalent function\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3914, 48), tuple_21920, str_21922)

# Processing the call keyword arguments (line 3914)
kwargs_21923 = {}
# Getting the type of 'add_newdoc' (line 3914)
add_newdoc_21917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3914, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 3914)
add_newdoc_call_result_21924 = invoke(stypy.reporting.localization.Localization(__file__, 3914, 0), add_newdoc_21917, *[str_21918, str_21919, tuple_21920], **kwargs_21923)


# Call to add_newdoc(...): (line 3929)
# Processing the call arguments (line 3929)
str_21926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3929, 11), 'str', 'numpy.core.multiarray')
str_21927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3929, 36), 'str', 'ndarray')

# Obtaining an instance of the builtin type 'tuple' (line 3929)
tuple_21928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3929, 48), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 3929)
# Adding element type (line 3929)
str_21929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3929, 48), 'str', 'prod')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3929, 48), tuple_21928, str_21929)
# Adding element type (line 3929)
str_21930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3941, (-1)), 'str', '\n    a.prod(axis=None, dtype=None, out=None, keepdims=False)\n\n    Return the product of the array elements over the given axis\n\n    Refer to `numpy.prod` for full documentation.\n\n    See Also\n    --------\n    numpy.prod : equivalent function\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3929, 48), tuple_21928, str_21930)

# Processing the call keyword arguments (line 3929)
kwargs_21931 = {}
# Getting the type of 'add_newdoc' (line 3929)
add_newdoc_21925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3929, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 3929)
add_newdoc_call_result_21932 = invoke(stypy.reporting.localization.Localization(__file__, 3929, 0), add_newdoc_21925, *[str_21926, str_21927, tuple_21928], **kwargs_21931)


# Call to add_newdoc(...): (line 3944)
# Processing the call arguments (line 3944)
str_21934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3944, 11), 'str', 'numpy.core.multiarray')
str_21935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3944, 36), 'str', 'ndarray')

# Obtaining an instance of the builtin type 'tuple' (line 3944)
tuple_21936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3944, 48), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 3944)
# Adding element type (line 3944)
str_21937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3944, 48), 'str', 'ptp')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3944, 48), tuple_21936, str_21937)
# Adding element type (line 3944)
str_21938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3956, (-1)), 'str', '\n    a.ptp(axis=None, out=None)\n\n    Peak to peak (maximum - minimum) value along a given axis.\n\n    Refer to `numpy.ptp` for full documentation.\n\n    See Also\n    --------\n    numpy.ptp : equivalent function\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3944, 48), tuple_21936, str_21938)

# Processing the call keyword arguments (line 3944)
kwargs_21939 = {}
# Getting the type of 'add_newdoc' (line 3944)
add_newdoc_21933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3944, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 3944)
add_newdoc_call_result_21940 = invoke(stypy.reporting.localization.Localization(__file__, 3944, 0), add_newdoc_21933, *[str_21934, str_21935, tuple_21936], **kwargs_21939)


# Call to add_newdoc(...): (line 3959)
# Processing the call arguments (line 3959)
str_21942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3959, 11), 'str', 'numpy.core.multiarray')
str_21943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3959, 36), 'str', 'ndarray')

# Obtaining an instance of the builtin type 'tuple' (line 3959)
tuple_21944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3959, 48), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 3959)
# Adding element type (line 3959)
str_21945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3959, 48), 'str', 'put')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3959, 48), tuple_21944, str_21945)
# Adding element type (line 3959)
str_21946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3971, (-1)), 'str', "\n    a.put(indices, values, mode='raise')\n\n    Set ``a.flat[n] = values[n]`` for all `n` in indices.\n\n    Refer to `numpy.put` for full documentation.\n\n    See Also\n    --------\n    numpy.put : equivalent function\n\n    ")
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3959, 48), tuple_21944, str_21946)

# Processing the call keyword arguments (line 3959)
kwargs_21947 = {}
# Getting the type of 'add_newdoc' (line 3959)
add_newdoc_21941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3959, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 3959)
add_newdoc_call_result_21948 = invoke(stypy.reporting.localization.Localization(__file__, 3959, 0), add_newdoc_21941, *[str_21942, str_21943, tuple_21944], **kwargs_21947)


# Call to add_newdoc(...): (line 3973)
# Processing the call arguments (line 3973)
str_21950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3973, 11), 'str', 'numpy.core.multiarray')
str_21951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3973, 36), 'str', 'copyto')
str_21952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4004, (-1)), 'str', "\n    copyto(dst, src, casting='same_kind', where=None)\n\n    Copies values from one array to another, broadcasting as necessary.\n\n    Raises a TypeError if the `casting` rule is violated, and if\n    `where` is provided, it selects which elements to copy.\n\n    .. versionadded:: 1.7.0\n\n    Parameters\n    ----------\n    dst : ndarray\n        The array into which values are copied.\n    src : array_like\n        The array from which values are copied.\n    casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional\n        Controls what kind of data casting may occur when copying.\n\n          * 'no' means the data types should not be cast at all.\n          * 'equiv' means only byte-order changes are allowed.\n          * 'safe' means only casts which can preserve values are allowed.\n          * 'same_kind' means only safe casts or casts within a kind,\n            like float64 to float32, are allowed.\n          * 'unsafe' means any data conversions may be done.\n    where : array_like of bool, optional\n        A boolean array which is broadcasted to match the dimensions\n        of `dst`, and selects elements to copy from `src` to `dst`\n        wherever it contains the value True.\n\n    ")
# Processing the call keyword arguments (line 3973)
kwargs_21953 = {}
# Getting the type of 'add_newdoc' (line 3973)
add_newdoc_21949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3973, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 3973)
add_newdoc_call_result_21954 = invoke(stypy.reporting.localization.Localization(__file__, 3973, 0), add_newdoc_21949, *[str_21950, str_21951, str_21952], **kwargs_21953)


# Call to add_newdoc(...): (line 4006)
# Processing the call arguments (line 4006)
str_21956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4006, 11), 'str', 'numpy.core.multiarray')
str_21957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4006, 36), 'str', 'putmask')
str_21958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4046, (-1)), 'str', '\n    putmask(a, mask, values)\n\n    Changes elements of an array based on conditional and input values.\n\n    Sets ``a.flat[n] = values[n]`` for each n where ``mask.flat[n]==True``.\n\n    If `values` is not the same size as `a` and `mask` then it will repeat.\n    This gives behavior different from ``a[mask] = values``.\n\n    Parameters\n    ----------\n    a : array_like\n        Target array.\n    mask : array_like\n        Boolean mask array. It has to be the same shape as `a`.\n    values : array_like\n        Values to put into `a` where `mask` is True. If `values` is smaller\n        than `a` it will be repeated.\n\n    See Also\n    --------\n    place, put, take, copyto\n\n    Examples\n    --------\n    >>> x = np.arange(6).reshape(2, 3)\n    >>> np.putmask(x, x>2, x**2)\n    >>> x\n    array([[ 0,  1,  2],\n           [ 9, 16, 25]])\n\n    If `values` is smaller than `a` it is repeated:\n\n    >>> x = np.arange(5)\n    >>> np.putmask(x, x>1, [-33, -44])\n    >>> x\n    array([  0,   1, -33, -44, -33])\n\n    ')
# Processing the call keyword arguments (line 4006)
kwargs_21959 = {}
# Getting the type of 'add_newdoc' (line 4006)
add_newdoc_21955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4006, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 4006)
add_newdoc_call_result_21960 = invoke(stypy.reporting.localization.Localization(__file__, 4006, 0), add_newdoc_21955, *[str_21956, str_21957, str_21958], **kwargs_21959)


# Call to add_newdoc(...): (line 4049)
# Processing the call arguments (line 4049)
str_21962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4049, 11), 'str', 'numpy.core.multiarray')
str_21963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4049, 36), 'str', 'ndarray')

# Obtaining an instance of the builtin type 'tuple' (line 4049)
tuple_21964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4049, 48), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 4049)
# Adding element type (line 4049)
str_21965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4049, 48), 'str', 'ravel')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 4049, 48), tuple_21964, str_21965)
# Adding element type (line 4049)
str_21966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4063, (-1)), 'str', '\n    a.ravel([order])\n\n    Return a flattened array.\n\n    Refer to `numpy.ravel` for full documentation.\n\n    See Also\n    --------\n    numpy.ravel : equivalent function\n\n    ndarray.flat : a flat iterator on the array.\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 4049, 48), tuple_21964, str_21966)

# Processing the call keyword arguments (line 4049)
kwargs_21967 = {}
# Getting the type of 'add_newdoc' (line 4049)
add_newdoc_21961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4049, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 4049)
add_newdoc_call_result_21968 = invoke(stypy.reporting.localization.Localization(__file__, 4049, 0), add_newdoc_21961, *[str_21962, str_21963, tuple_21964], **kwargs_21967)


# Call to add_newdoc(...): (line 4066)
# Processing the call arguments (line 4066)
str_21970 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4066, 11), 'str', 'numpy.core.multiarray')
str_21971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4066, 36), 'str', 'ndarray')

# Obtaining an instance of the builtin type 'tuple' (line 4066)
tuple_21972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4066, 48), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 4066)
# Adding element type (line 4066)
str_21973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4066, 48), 'str', 'repeat')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 4066, 48), tuple_21972, str_21973)
# Adding element type (line 4066)
str_21974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4078, (-1)), 'str', '\n    a.repeat(repeats, axis=None)\n\n    Repeat elements of an array.\n\n    Refer to `numpy.repeat` for full documentation.\n\n    See Also\n    --------\n    numpy.repeat : equivalent function\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 4066, 48), tuple_21972, str_21974)

# Processing the call keyword arguments (line 4066)
kwargs_21975 = {}
# Getting the type of 'add_newdoc' (line 4066)
add_newdoc_21969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4066, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 4066)
add_newdoc_call_result_21976 = invoke(stypy.reporting.localization.Localization(__file__, 4066, 0), add_newdoc_21969, *[str_21970, str_21971, tuple_21972], **kwargs_21975)


# Call to add_newdoc(...): (line 4081)
# Processing the call arguments (line 4081)
str_21978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4081, 11), 'str', 'numpy.core.multiarray')
str_21979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4081, 36), 'str', 'ndarray')

# Obtaining an instance of the builtin type 'tuple' (line 4081)
tuple_21980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4081, 48), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 4081)
# Adding element type (line 4081)
str_21981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4081, 48), 'str', 'reshape')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 4081, 48), tuple_21980, str_21981)
# Adding element type (line 4081)
str_21982 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4093, (-1)), 'str', "\n    a.reshape(shape, order='C')\n\n    Returns an array containing the same data with a new shape.\n\n    Refer to `numpy.reshape` for full documentation.\n\n    See Also\n    --------\n    numpy.reshape : equivalent function\n\n    ")
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 4081, 48), tuple_21980, str_21982)

# Processing the call keyword arguments (line 4081)
kwargs_21983 = {}
# Getting the type of 'add_newdoc' (line 4081)
add_newdoc_21977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4081, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 4081)
add_newdoc_call_result_21984 = invoke(stypy.reporting.localization.Localization(__file__, 4081, 0), add_newdoc_21977, *[str_21978, str_21979, tuple_21980], **kwargs_21983)


# Call to add_newdoc(...): (line 4096)
# Processing the call arguments (line 4096)
str_21986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4096, 11), 'str', 'numpy.core.multiarray')
str_21987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4096, 36), 'str', 'ndarray')

# Obtaining an instance of the builtin type 'tuple' (line 4096)
tuple_21988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4096, 48), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 4096)
# Adding element type (line 4096)
str_21989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4096, 48), 'str', 'resize')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 4096, 48), tuple_21988, str_21989)
# Adding element type (line 4096)
str_21990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4182, (-1)), 'str', "\n    a.resize(new_shape, refcheck=True)\n\n    Change shape and size of array in-place.\n\n    Parameters\n    ----------\n    new_shape : tuple of ints, or `n` ints\n        Shape of resized array.\n    refcheck : bool, optional\n        If False, reference count will not be checked. Default is True.\n\n    Returns\n    -------\n    None\n\n    Raises\n    ------\n    ValueError\n        If `a` does not own its own data or references or views to it exist,\n        and the data memory must be changed.\n\n    SystemError\n        If the `order` keyword argument is specified. This behaviour is a\n        bug in NumPy.\n\n    See Also\n    --------\n    resize : Return a new array with the specified shape.\n\n    Notes\n    -----\n    This reallocates space for the data area if necessary.\n\n    Only contiguous arrays (data elements consecutive in memory) can be\n    resized.\n\n    The purpose of the reference count check is to make sure you\n    do not use this array as a buffer for another Python object and then\n    reallocate the memory. However, reference counts can increase in\n    other ways so if you are sure that you have not shared the memory\n    for this array with another Python object, then you may safely set\n    `refcheck` to False.\n\n    Examples\n    --------\n    Shrinking an array: array is flattened (in the order that the data are\n    stored in memory), resized, and reshaped:\n\n    >>> a = np.array([[0, 1], [2, 3]], order='C')\n    >>> a.resize((2, 1))\n    >>> a\n    array([[0],\n           [1]])\n\n    >>> a = np.array([[0, 1], [2, 3]], order='F')\n    >>> a.resize((2, 1))\n    >>> a\n    array([[0],\n           [2]])\n\n    Enlarging an array: as above, but missing entries are filled with zeros:\n\n    >>> b = np.array([[0, 1], [2, 3]])\n    >>> b.resize(2, 3) # new_shape parameter doesn't have to be a tuple\n    >>> b\n    array([[0, 1, 2],\n           [3, 0, 0]])\n\n    Referencing an array prevents resizing...\n\n    >>> c = a\n    >>> a.resize((1, 1))\n    Traceback (most recent call last):\n    ...\n    ValueError: cannot resize an array that has been referenced ...\n\n    Unless `refcheck` is False:\n\n    >>> a.resize((1, 1), refcheck=False)\n    >>> a\n    array([[0]])\n    >>> c\n    array([[0]])\n\n    ")
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 4096, 48), tuple_21988, str_21990)

# Processing the call keyword arguments (line 4096)
kwargs_21991 = {}
# Getting the type of 'add_newdoc' (line 4096)
add_newdoc_21985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4096, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 4096)
add_newdoc_call_result_21992 = invoke(stypy.reporting.localization.Localization(__file__, 4096, 0), add_newdoc_21985, *[str_21986, str_21987, tuple_21988], **kwargs_21991)


# Call to add_newdoc(...): (line 4185)
# Processing the call arguments (line 4185)
str_21994 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4185, 11), 'str', 'numpy.core.multiarray')
str_21995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4185, 36), 'str', 'ndarray')

# Obtaining an instance of the builtin type 'tuple' (line 4185)
tuple_21996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4185, 48), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 4185)
# Adding element type (line 4185)
str_21997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4185, 48), 'str', 'round')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 4185, 48), tuple_21996, str_21997)
# Adding element type (line 4185)
str_21998 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4197, (-1)), 'str', '\n    a.round(decimals=0, out=None)\n\n    Return `a` with each element rounded to the given number of decimals.\n\n    Refer to `numpy.around` for full documentation.\n\n    See Also\n    --------\n    numpy.around : equivalent function\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 4185, 48), tuple_21996, str_21998)

# Processing the call keyword arguments (line 4185)
kwargs_21999 = {}
# Getting the type of 'add_newdoc' (line 4185)
add_newdoc_21993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4185, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 4185)
add_newdoc_call_result_22000 = invoke(stypy.reporting.localization.Localization(__file__, 4185, 0), add_newdoc_21993, *[str_21994, str_21995, tuple_21996], **kwargs_21999)


# Call to add_newdoc(...): (line 4200)
# Processing the call arguments (line 4200)
str_22002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4200, 11), 'str', 'numpy.core.multiarray')
str_22003 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4200, 36), 'str', 'ndarray')

# Obtaining an instance of the builtin type 'tuple' (line 4200)
tuple_22004 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4200, 48), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 4200)
# Adding element type (line 4200)
str_22005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4200, 48), 'str', 'searchsorted')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 4200, 48), tuple_22004, str_22005)
# Adding element type (line 4200)
str_22006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4212, (-1)), 'str', "\n    a.searchsorted(v, side='left', sorter=None)\n\n    Find indices where elements of v should be inserted in a to maintain order.\n\n    For full documentation, see `numpy.searchsorted`\n\n    See Also\n    --------\n    numpy.searchsorted : equivalent function\n\n    ")
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 4200, 48), tuple_22004, str_22006)

# Processing the call keyword arguments (line 4200)
kwargs_22007 = {}
# Getting the type of 'add_newdoc' (line 4200)
add_newdoc_22001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4200, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 4200)
add_newdoc_call_result_22008 = invoke(stypy.reporting.localization.Localization(__file__, 4200, 0), add_newdoc_22001, *[str_22002, str_22003, tuple_22004], **kwargs_22007)


# Call to add_newdoc(...): (line 4215)
# Processing the call arguments (line 4215)
str_22010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4215, 11), 'str', 'numpy.core.multiarray')
str_22011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4215, 36), 'str', 'ndarray')

# Obtaining an instance of the builtin type 'tuple' (line 4215)
tuple_22012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4215, 48), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 4215)
# Adding element type (line 4215)
str_22013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4215, 48), 'str', 'setfield')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 4215, 48), tuple_22012, str_22013)
# Adding element type (line 4215)
str_22014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4263, (-1)), 'str', "\n    a.setfield(val, dtype, offset=0)\n\n    Put a value into a specified place in a field defined by a data-type.\n\n    Place `val` into `a`'s field defined by `dtype` and beginning `offset`\n    bytes into the field.\n\n    Parameters\n    ----------\n    val : object\n        Value to be placed in field.\n    dtype : dtype object\n        Data-type of the field in which to place `val`.\n    offset : int, optional\n        The number of bytes into the field at which to place `val`.\n\n    Returns\n    -------\n    None\n\n    See Also\n    --------\n    getfield\n\n    Examples\n    --------\n    >>> x = np.eye(3)\n    >>> x.getfield(np.float64)\n    array([[ 1.,  0.,  0.],\n           [ 0.,  1.,  0.],\n           [ 0.,  0.,  1.]])\n    >>> x.setfield(3, np.int32)\n    >>> x.getfield(np.int32)\n    array([[3, 3, 3],\n           [3, 3, 3],\n           [3, 3, 3]])\n    >>> x\n    array([[  1.00000000e+000,   1.48219694e-323,   1.48219694e-323],\n           [  1.48219694e-323,   1.00000000e+000,   1.48219694e-323],\n           [  1.48219694e-323,   1.48219694e-323,   1.00000000e+000]])\n    >>> x.setfield(np.eye(3), np.int32)\n    >>> x\n    array([[ 1.,  0.,  0.],\n           [ 0.,  1.,  0.],\n           [ 0.,  0.,  1.]])\n\n    ")
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 4215, 48), tuple_22012, str_22014)

# Processing the call keyword arguments (line 4215)
kwargs_22015 = {}
# Getting the type of 'add_newdoc' (line 4215)
add_newdoc_22009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4215, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 4215)
add_newdoc_call_result_22016 = invoke(stypy.reporting.localization.Localization(__file__, 4215, 0), add_newdoc_22009, *[str_22010, str_22011, tuple_22012], **kwargs_22015)


# Call to add_newdoc(...): (line 4266)
# Processing the call arguments (line 4266)
str_22018 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4266, 11), 'str', 'numpy.core.multiarray')
str_22019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4266, 36), 'str', 'ndarray')

# Obtaining an instance of the builtin type 'tuple' (line 4266)
tuple_22020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4266, 48), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 4266)
# Adding element type (line 4266)
str_22021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4266, 48), 'str', 'setflags')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 4266, 48), tuple_22020, str_22021)
# Adding element type (line 4266)
str_22022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4335, (-1)), 'str', '\n    a.setflags(write=None, align=None, uic=None)\n\n    Set array flags WRITEABLE, ALIGNED, and UPDATEIFCOPY, respectively.\n\n    These Boolean-valued flags affect how numpy interprets the memory\n    area used by `a` (see Notes below). The ALIGNED flag can only\n    be set to True if the data is actually aligned according to the type.\n    The UPDATEIFCOPY flag can never be set to True. The flag WRITEABLE\n    can only be set to True if the array owns its own memory, or the\n    ultimate owner of the memory exposes a writeable buffer interface,\n    or is a string. (The exception for string is made so that unpickling\n    can be done without copying memory.)\n\n    Parameters\n    ----------\n    write : bool, optional\n        Describes whether or not `a` can be written to.\n    align : bool, optional\n        Describes whether or not `a` is aligned properly for its type.\n    uic : bool, optional\n        Describes whether or not `a` is a copy of another "base" array.\n\n    Notes\n    -----\n    Array flags provide information about how the memory area used\n    for the array is to be interpreted. There are 6 Boolean flags\n    in use, only three of which can be changed by the user:\n    UPDATEIFCOPY, WRITEABLE, and ALIGNED.\n\n    WRITEABLE (W) the data area can be written to;\n\n    ALIGNED (A) the data and strides are aligned appropriately for the hardware\n    (as determined by the compiler);\n\n    UPDATEIFCOPY (U) this array is a copy of some other array (referenced\n    by .base). When this array is deallocated, the base array will be\n    updated with the contents of this array.\n\n    All flags can be accessed using their first (upper case) letter as well\n    as the full name.\n\n    Examples\n    --------\n    >>> y\n    array([[3, 1, 7],\n           [2, 0, 0],\n           [8, 5, 9]])\n    >>> y.flags\n      C_CONTIGUOUS : True\n      F_CONTIGUOUS : False\n      OWNDATA : True\n      WRITEABLE : True\n      ALIGNED : True\n      UPDATEIFCOPY : False\n    >>> y.setflags(write=0, align=0)\n    >>> y.flags\n      C_CONTIGUOUS : True\n      F_CONTIGUOUS : False\n      OWNDATA : True\n      WRITEABLE : False\n      ALIGNED : False\n      UPDATEIFCOPY : False\n    >>> y.setflags(uic=1)\n    Traceback (most recent call last):\n      File "<stdin>", line 1, in <module>\n    ValueError: cannot set UPDATEIFCOPY flag to True\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 4266, 48), tuple_22020, str_22022)

# Processing the call keyword arguments (line 4266)
kwargs_22023 = {}
# Getting the type of 'add_newdoc' (line 4266)
add_newdoc_22017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4266, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 4266)
add_newdoc_call_result_22024 = invoke(stypy.reporting.localization.Localization(__file__, 4266, 0), add_newdoc_22017, *[str_22018, str_22019, tuple_22020], **kwargs_22023)


# Call to add_newdoc(...): (line 4338)
# Processing the call arguments (line 4338)
str_22026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4338, 11), 'str', 'numpy.core.multiarray')
str_22027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4338, 36), 'str', 'ndarray')

# Obtaining an instance of the builtin type 'tuple' (line 4338)
tuple_22028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4338, 48), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 4338)
# Adding element type (line 4338)
str_22029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4338, 48), 'str', 'sort')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 4338, 48), tuple_22028, str_22029)
# Adding element type (line 4338)
str_22030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4391, (-1)), 'str', "\n    a.sort(axis=-1, kind='quicksort', order=None)\n\n    Sort an array, in-place.\n\n    Parameters\n    ----------\n    axis : int, optional\n        Axis along which to sort. Default is -1, which means sort along the\n        last axis.\n    kind : {'quicksort', 'mergesort', 'heapsort'}, optional\n        Sorting algorithm. Default is 'quicksort'.\n    order : str or list of str, optional\n        When `a` is an array with fields defined, this argument specifies\n        which fields to compare first, second, etc.  A single field can\n        be specified as a string, and not all fields need be specified,\n        but unspecified fields will still be used, in the order in which\n        they come up in the dtype, to break ties.\n\n    See Also\n    --------\n    numpy.sort : Return a sorted copy of an array.\n    argsort : Indirect sort.\n    lexsort : Indirect stable sort on multiple keys.\n    searchsorted : Find elements in sorted array.\n    partition: Partial sort.\n\n    Notes\n    -----\n    See ``sort`` for notes on the different sorting algorithms.\n\n    Examples\n    --------\n    >>> a = np.array([[1,4], [3,1]])\n    >>> a.sort(axis=1)\n    >>> a\n    array([[1, 4],\n           [1, 3]])\n    >>> a.sort(axis=0)\n    >>> a\n    array([[1, 3],\n           [1, 4]])\n\n    Use the `order` keyword to specify a field to use when sorting a\n    structured array:\n\n    >>> a = np.array([('a', 2), ('c', 1)], dtype=[('x', 'S1'), ('y', int)])\n    >>> a.sort(order='y')\n    >>> a\n    array([('c', 1), ('a', 2)],\n          dtype=[('x', '|S1'), ('y', '<i4')])\n\n    ")
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 4338, 48), tuple_22028, str_22030)

# Processing the call keyword arguments (line 4338)
kwargs_22031 = {}
# Getting the type of 'add_newdoc' (line 4338)
add_newdoc_22025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4338, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 4338)
add_newdoc_call_result_22032 = invoke(stypy.reporting.localization.Localization(__file__, 4338, 0), add_newdoc_22025, *[str_22026, str_22027, tuple_22028], **kwargs_22031)


# Call to add_newdoc(...): (line 4394)
# Processing the call arguments (line 4394)
str_22034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4394, 11), 'str', 'numpy.core.multiarray')
str_22035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4394, 36), 'str', 'ndarray')

# Obtaining an instance of the builtin type 'tuple' (line 4394)
tuple_22036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4394, 48), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 4394)
# Adding element type (line 4394)
str_22037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4394, 48), 'str', 'partition')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 4394, 48), tuple_22036, str_22037)
# Adding element type (line 4394)
str_22038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4446, (-1)), 'str', "\n    a.partition(kth, axis=-1, kind='introselect', order=None)\n\n    Rearranges the elements in the array in such a way that value of the\n    element in kth position is in the position it would be in a sorted array.\n    All elements smaller than the kth element are moved before this element and\n    all equal or greater are moved behind it. The ordering of the elements in\n    the two partitions is undefined.\n\n    .. versionadded:: 1.8.0\n\n    Parameters\n    ----------\n    kth : int or sequence of ints\n        Element index to partition by. The kth element value will be in its\n        final sorted position and all smaller elements will be moved before it\n        and all equal or greater elements behind it.\n        The order all elements in the partitions is undefined.\n        If provided with a sequence of kth it will partition all elements\n        indexed by kth of them into their sorted position at once.\n    axis : int, optional\n        Axis along which to sort. Default is -1, which means sort along the\n        last axis.\n    kind : {'introselect'}, optional\n        Selection algorithm. Default is 'introselect'.\n    order : str or list of str, optional\n        When `a` is an array with fields defined, this argument specifies\n        which fields to compare first, second, etc.  A single field can\n        be specified as a string, and not all fields need be specified,\n        but unspecified fields will still be used, in the order in which\n        they come up in the dtype, to break ties.\n\n    See Also\n    --------\n    numpy.partition : Return a parititioned copy of an array.\n    argpartition : Indirect partition.\n    sort : Full sort.\n\n    Notes\n    -----\n    See ``np.partition`` for notes on the different algorithms.\n\n    Examples\n    --------\n    >>> a = np.array([3, 4, 2, 1])\n    >>> a.partition(a, 3)\n    >>> a\n    array([2, 1, 3, 4])\n\n    >>> a.partition((1, 3))\n    array([1, 2, 3, 4])\n    ")
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 4394, 48), tuple_22036, str_22038)

# Processing the call keyword arguments (line 4394)
kwargs_22039 = {}
# Getting the type of 'add_newdoc' (line 4394)
add_newdoc_22033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4394, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 4394)
add_newdoc_call_result_22040 = invoke(stypy.reporting.localization.Localization(__file__, 4394, 0), add_newdoc_22033, *[str_22034, str_22035, tuple_22036], **kwargs_22039)


# Call to add_newdoc(...): (line 4449)
# Processing the call arguments (line 4449)
str_22042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4449, 11), 'str', 'numpy.core.multiarray')
str_22043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4449, 36), 'str', 'ndarray')

# Obtaining an instance of the builtin type 'tuple' (line 4449)
tuple_22044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4449, 48), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 4449)
# Adding element type (line 4449)
str_22045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4449, 48), 'str', 'squeeze')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 4449, 48), tuple_22044, str_22045)
# Adding element type (line 4449)
str_22046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4461, (-1)), 'str', '\n    a.squeeze(axis=None)\n\n    Remove single-dimensional entries from the shape of `a`.\n\n    Refer to `numpy.squeeze` for full documentation.\n\n    See Also\n    --------\n    numpy.squeeze : equivalent function\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 4449, 48), tuple_22044, str_22046)

# Processing the call keyword arguments (line 4449)
kwargs_22047 = {}
# Getting the type of 'add_newdoc' (line 4449)
add_newdoc_22041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4449, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 4449)
add_newdoc_call_result_22048 = invoke(stypy.reporting.localization.Localization(__file__, 4449, 0), add_newdoc_22041, *[str_22042, str_22043, tuple_22044], **kwargs_22047)


# Call to add_newdoc(...): (line 4464)
# Processing the call arguments (line 4464)
str_22050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4464, 11), 'str', 'numpy.core.multiarray')
str_22051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4464, 36), 'str', 'ndarray')

# Obtaining an instance of the builtin type 'tuple' (line 4464)
tuple_22052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4464, 48), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 4464)
# Adding element type (line 4464)
str_22053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4464, 48), 'str', 'std')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 4464, 48), tuple_22052, str_22053)
# Adding element type (line 4464)
str_22054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4476, (-1)), 'str', '\n    a.std(axis=None, dtype=None, out=None, ddof=0, keepdims=False)\n\n    Returns the standard deviation of the array elements along given axis.\n\n    Refer to `numpy.std` for full documentation.\n\n    See Also\n    --------\n    numpy.std : equivalent function\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 4464, 48), tuple_22052, str_22054)

# Processing the call keyword arguments (line 4464)
kwargs_22055 = {}
# Getting the type of 'add_newdoc' (line 4464)
add_newdoc_22049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4464, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 4464)
add_newdoc_call_result_22056 = invoke(stypy.reporting.localization.Localization(__file__, 4464, 0), add_newdoc_22049, *[str_22050, str_22051, tuple_22052], **kwargs_22055)


# Call to add_newdoc(...): (line 4479)
# Processing the call arguments (line 4479)
str_22058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4479, 11), 'str', 'numpy.core.multiarray')
str_22059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4479, 36), 'str', 'ndarray')

# Obtaining an instance of the builtin type 'tuple' (line 4479)
tuple_22060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4479, 48), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 4479)
# Adding element type (line 4479)
str_22061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4479, 48), 'str', 'sum')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 4479, 48), tuple_22060, str_22061)
# Adding element type (line 4479)
str_22062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4491, (-1)), 'str', '\n    a.sum(axis=None, dtype=None, out=None, keepdims=False)\n\n    Return the sum of the array elements over the given axis.\n\n    Refer to `numpy.sum` for full documentation.\n\n    See Also\n    --------\n    numpy.sum : equivalent function\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 4479, 48), tuple_22060, str_22062)

# Processing the call keyword arguments (line 4479)
kwargs_22063 = {}
# Getting the type of 'add_newdoc' (line 4479)
add_newdoc_22057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4479, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 4479)
add_newdoc_call_result_22064 = invoke(stypy.reporting.localization.Localization(__file__, 4479, 0), add_newdoc_22057, *[str_22058, str_22059, tuple_22060], **kwargs_22063)


# Call to add_newdoc(...): (line 4494)
# Processing the call arguments (line 4494)
str_22066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4494, 11), 'str', 'numpy.core.multiarray')
str_22067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4494, 36), 'str', 'ndarray')

# Obtaining an instance of the builtin type 'tuple' (line 4494)
tuple_22068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4494, 48), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 4494)
# Adding element type (line 4494)
str_22069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4494, 48), 'str', 'swapaxes')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 4494, 48), tuple_22068, str_22069)
# Adding element type (line 4494)
str_22070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4506, (-1)), 'str', '\n    a.swapaxes(axis1, axis2)\n\n    Return a view of the array with `axis1` and `axis2` interchanged.\n\n    Refer to `numpy.swapaxes` for full documentation.\n\n    See Also\n    --------\n    numpy.swapaxes : equivalent function\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 4494, 48), tuple_22068, str_22070)

# Processing the call keyword arguments (line 4494)
kwargs_22071 = {}
# Getting the type of 'add_newdoc' (line 4494)
add_newdoc_22065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4494, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 4494)
add_newdoc_call_result_22072 = invoke(stypy.reporting.localization.Localization(__file__, 4494, 0), add_newdoc_22065, *[str_22066, str_22067, tuple_22068], **kwargs_22071)


# Call to add_newdoc(...): (line 4509)
# Processing the call arguments (line 4509)
str_22074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4509, 11), 'str', 'numpy.core.multiarray')
str_22075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4509, 36), 'str', 'ndarray')

# Obtaining an instance of the builtin type 'tuple' (line 4509)
tuple_22076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4509, 48), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 4509)
# Adding element type (line 4509)
str_22077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4509, 48), 'str', 'take')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 4509, 48), tuple_22076, str_22077)
# Adding element type (line 4509)
str_22078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4521, (-1)), 'str', "\n    a.take(indices, axis=None, out=None, mode='raise')\n\n    Return an array formed from the elements of `a` at the given indices.\n\n    Refer to `numpy.take` for full documentation.\n\n    See Also\n    --------\n    numpy.take : equivalent function\n\n    ")
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 4509, 48), tuple_22076, str_22078)

# Processing the call keyword arguments (line 4509)
kwargs_22079 = {}
# Getting the type of 'add_newdoc' (line 4509)
add_newdoc_22073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4509, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 4509)
add_newdoc_call_result_22080 = invoke(stypy.reporting.localization.Localization(__file__, 4509, 0), add_newdoc_22073, *[str_22074, str_22075, tuple_22076], **kwargs_22079)


# Call to add_newdoc(...): (line 4524)
# Processing the call arguments (line 4524)
str_22082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4524, 11), 'str', 'numpy.core.multiarray')
str_22083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4524, 36), 'str', 'ndarray')

# Obtaining an instance of the builtin type 'tuple' (line 4524)
tuple_22084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4524, 48), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 4524)
# Adding element type (line 4524)
str_22085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4524, 48), 'str', 'tofile')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 4524, 48), tuple_22084, str_22085)
# Adding element type (line 4524)
str_22086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4556, (-1)), 'str', '\n    a.tofile(fid, sep="", format="%s")\n\n    Write array to a file as text or binary (default).\n\n    Data is always written in \'C\' order, independent of the order of `a`.\n    The data produced by this method can be recovered using the function\n    fromfile().\n\n    Parameters\n    ----------\n    fid : file or str\n        An open file object, or a string containing a filename.\n    sep : str\n        Separator between array items for text output.\n        If "" (empty), a binary file is written, equivalent to\n        ``file.write(a.tobytes())``.\n    format : str\n        Format string for text file output.\n        Each entry in the array is formatted to text by first converting\n        it to the closest Python type, and then using "format" % item.\n\n    Notes\n    -----\n    This is a convenience function for quick storage of array data.\n    Information on endianness and precision is lost, so this method is not a\n    good choice for files intended to archive data or transport data between\n    machines with different endianness. Some of these problems can be overcome\n    by outputting the data as text files, at the expense of speed and file\n    size.\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 4524, 48), tuple_22084, str_22086)

# Processing the call keyword arguments (line 4524)
kwargs_22087 = {}
# Getting the type of 'add_newdoc' (line 4524)
add_newdoc_22081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4524, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 4524)
add_newdoc_call_result_22088 = invoke(stypy.reporting.localization.Localization(__file__, 4524, 0), add_newdoc_22081, *[str_22082, str_22083, tuple_22084], **kwargs_22087)


# Call to add_newdoc(...): (line 4559)
# Processing the call arguments (line 4559)
str_22090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4559, 11), 'str', 'numpy.core.multiarray')
str_22091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4559, 36), 'str', 'ndarray')

# Obtaining an instance of the builtin type 'tuple' (line 4559)
tuple_22092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4559, 48), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 4559)
# Adding element type (line 4559)
str_22093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4559, 48), 'str', 'tolist')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 4559, 48), tuple_22092, str_22093)
# Adding element type (line 4559)
str_22094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4592, (-1)), 'str', '\n    a.tolist()\n\n    Return the array as a (possibly nested) list.\n\n    Return a copy of the array data as a (nested) Python list.\n    Data items are converted to the nearest compatible Python type.\n\n    Parameters\n    ----------\n    none\n\n    Returns\n    -------\n    y : list\n        The possibly nested list of array elements.\n\n    Notes\n    -----\n    The array may be recreated, ``a = np.array(a.tolist())``.\n\n    Examples\n    --------\n    >>> a = np.array([1, 2])\n    >>> a.tolist()\n    [1, 2]\n    >>> a = np.array([[1, 2], [3, 4]])\n    >>> list(a)\n    [array([1, 2]), array([3, 4])]\n    >>> a.tolist()\n    [[1, 2], [3, 4]]\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 4559, 48), tuple_22092, str_22094)

# Processing the call keyword arguments (line 4559)
kwargs_22095 = {}
# Getting the type of 'add_newdoc' (line 4559)
add_newdoc_22089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4559, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 4559)
add_newdoc_call_result_22096 = invoke(stypy.reporting.localization.Localization(__file__, 4559, 0), add_newdoc_22089, *[str_22090, str_22091, tuple_22092], **kwargs_22095)


# Assigning a Str to a Name (line 4595):
str_22097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4629, (-1)), 'str', "\n    a.{name}(order='C')\n\n    Construct Python bytes containing the raw data bytes in the array.\n\n    Constructs Python bytes showing a copy of the raw contents of\n    data memory. The bytes object can be produced in either 'C' or 'Fortran',\n    or 'Any' order (the default is 'C'-order). 'Any' order means C-order\n    unless the F_CONTIGUOUS flag in the array is set, in which case it\n    means 'Fortran' order.\n\n    {deprecated}\n\n    Parameters\n    ----------\n    order : {{'C', 'F', None}}, optional\n        Order of the data for multidimensional arrays:\n        C, Fortran, or the same as for the original array.\n\n    Returns\n    -------\n    s : bytes\n        Python bytes exhibiting a copy of `a`'s raw data.\n\n    Examples\n    --------\n    >>> x = np.array([[0, 1], [2, 3]])\n    >>> x.tobytes()\n    b'\\x00\\x00\\x00\\x00\\x01\\x00\\x00\\x00\\x02\\x00\\x00\\x00\\x03\\x00\\x00\\x00'\n    >>> x.tobytes('C') == x.tobytes()\n    True\n    >>> x.tobytes('F')\n    b'\\x00\\x00\\x00\\x00\\x02\\x00\\x00\\x00\\x01\\x00\\x00\\x00\\x03\\x00\\x00\\x00'\n\n    ")
# Assigning a type to the variable 'tobytesdoc' (line 4595)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4595, 0), 'tobytesdoc', str_22097)

# Call to add_newdoc(...): (line 4631)
# Processing the call arguments (line 4631)
str_22099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4631, 11), 'str', 'numpy.core.multiarray')
str_22100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4631, 36), 'str', 'ndarray')

# Obtaining an instance of the builtin type 'tuple' (line 4632)
tuple_22101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4632, 12), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 4632)
# Adding element type (line 4632)
str_22102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4632, 12), 'str', 'tostring')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 4632, 12), tuple_22101, str_22102)
# Adding element type (line 4632)

# Call to format(...): (line 4632)
# Processing the call keyword arguments (line 4632)
str_22105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4632, 47), 'str', 'tostring')
keyword_22106 = str_22105
str_22107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4634, 42), 'str', 'This function is a compatibility alias for tobytes. Despite its name it returns bytes not strings.')
keyword_22108 = str_22107
kwargs_22109 = {'deprecated': keyword_22108, 'name': keyword_22106}
# Getting the type of 'tobytesdoc' (line 4632)
tobytesdoc_22103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4632, 24), 'tobytesdoc', False)
# Obtaining the member 'format' of a type (line 4632)
format_22104 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 4632, 24), tobytesdoc_22103, 'format')
# Calling format(args, kwargs) (line 4632)
format_call_result_22110 = invoke(stypy.reporting.localization.Localization(__file__, 4632, 24), format_22104, *[], **kwargs_22109)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 4632, 12), tuple_22101, format_call_result_22110)

# Processing the call keyword arguments (line 4631)
kwargs_22111 = {}
# Getting the type of 'add_newdoc' (line 4631)
add_newdoc_22098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4631, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 4631)
add_newdoc_call_result_22112 = invoke(stypy.reporting.localization.Localization(__file__, 4631, 0), add_newdoc_22098, *[str_22099, str_22100, tuple_22101], **kwargs_22111)


# Call to add_newdoc(...): (line 4638)
# Processing the call arguments (line 4638)
str_22114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4638, 11), 'str', 'numpy.core.multiarray')
str_22115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4638, 36), 'str', 'ndarray')

# Obtaining an instance of the builtin type 'tuple' (line 4639)
tuple_22116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4639, 12), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 4639)
# Adding element type (line 4639)
str_22117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4639, 12), 'str', 'tobytes')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 4639, 12), tuple_22116, str_22117)
# Adding element type (line 4639)

# Call to format(...): (line 4639)
# Processing the call keyword arguments (line 4639)
str_22120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4639, 46), 'str', 'tobytes')
keyword_22121 = str_22120
str_22122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4640, 52), 'str', '.. versionadded:: 1.9.0')
keyword_22123 = str_22122
kwargs_22124 = {'deprecated': keyword_22123, 'name': keyword_22121}
# Getting the type of 'tobytesdoc' (line 4639)
tobytesdoc_22118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4639, 23), 'tobytesdoc', False)
# Obtaining the member 'format' of a type (line 4639)
format_22119 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 4639, 23), tobytesdoc_22118, 'format')
# Calling format(args, kwargs) (line 4639)
format_call_result_22125 = invoke(stypy.reporting.localization.Localization(__file__, 4639, 23), format_22119, *[], **kwargs_22124)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 4639, 12), tuple_22116, format_call_result_22125)

# Processing the call keyword arguments (line 4638)
kwargs_22126 = {}
# Getting the type of 'add_newdoc' (line 4638)
add_newdoc_22113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4638, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 4638)
add_newdoc_call_result_22127 = invoke(stypy.reporting.localization.Localization(__file__, 4638, 0), add_newdoc_22113, *[str_22114, str_22115, tuple_22116], **kwargs_22126)


# Call to add_newdoc(...): (line 4642)
# Processing the call arguments (line 4642)
str_22129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4642, 11), 'str', 'numpy.core.multiarray')
str_22130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4642, 36), 'str', 'ndarray')

# Obtaining an instance of the builtin type 'tuple' (line 4642)
tuple_22131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4642, 48), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 4642)
# Adding element type (line 4642)
str_22132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4642, 48), 'str', 'trace')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 4642, 48), tuple_22131, str_22132)
# Adding element type (line 4642)
str_22133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4654, (-1)), 'str', '\n    a.trace(offset=0, axis1=0, axis2=1, dtype=None, out=None)\n\n    Return the sum along diagonals of the array.\n\n    Refer to `numpy.trace` for full documentation.\n\n    See Also\n    --------\n    numpy.trace : equivalent function\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 4642, 48), tuple_22131, str_22133)

# Processing the call keyword arguments (line 4642)
kwargs_22134 = {}
# Getting the type of 'add_newdoc' (line 4642)
add_newdoc_22128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4642, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 4642)
add_newdoc_call_result_22135 = invoke(stypy.reporting.localization.Localization(__file__, 4642, 0), add_newdoc_22128, *[str_22129, str_22130, tuple_22131], **kwargs_22134)


# Call to add_newdoc(...): (line 4657)
# Processing the call arguments (line 4657)
str_22137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4657, 11), 'str', 'numpy.core.multiarray')
str_22138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4657, 36), 'str', 'ndarray')

# Obtaining an instance of the builtin type 'tuple' (line 4657)
tuple_22139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4657, 48), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 4657)
# Adding element type (line 4657)
str_22140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4657, 48), 'str', 'transpose')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 4657, 48), tuple_22139, str_22140)
# Adding element type (line 4657)
str_22141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4708, (-1)), 'str', '\n    a.transpose(*axes)\n\n    Returns a view of the array with axes transposed.\n\n    For a 1-D array, this has no effect. (To change between column and\n    row vectors, first cast the 1-D array into a matrix object.)\n    For a 2-D array, this is the usual matrix transpose.\n    For an n-D array, if axes are given, their order indicates how the\n    axes are permuted (see Examples). If axes are not provided and\n    ``a.shape = (i[0], i[1], ... i[n-2], i[n-1])``, then\n    ``a.transpose().shape = (i[n-1], i[n-2], ... i[1], i[0])``.\n\n    Parameters\n    ----------\n    axes : None, tuple of ints, or `n` ints\n\n     * None or no argument: reverses the order of the axes.\n\n     * tuple of ints: `i` in the `j`-th place in the tuple means `a`\'s\n       `i`-th axis becomes `a.transpose()`\'s `j`-th axis.\n\n     * `n` ints: same as an n-tuple of the same ints (this form is\n       intended simply as a "convenience" alternative to the tuple form)\n\n    Returns\n    -------\n    out : ndarray\n        View of `a`, with axes suitably permuted.\n\n    See Also\n    --------\n    ndarray.T : Array property returning the array transposed.\n\n    Examples\n    --------\n    >>> a = np.array([[1, 2], [3, 4]])\n    >>> a\n    array([[1, 2],\n           [3, 4]])\n    >>> a.transpose()\n    array([[1, 3],\n           [2, 4]])\n    >>> a.transpose((1, 0))\n    array([[1, 3],\n           [2, 4]])\n    >>> a.transpose(1, 0)\n    array([[1, 3],\n           [2, 4]])\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 4657, 48), tuple_22139, str_22141)

# Processing the call keyword arguments (line 4657)
kwargs_22142 = {}
# Getting the type of 'add_newdoc' (line 4657)
add_newdoc_22136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4657, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 4657)
add_newdoc_call_result_22143 = invoke(stypy.reporting.localization.Localization(__file__, 4657, 0), add_newdoc_22136, *[str_22137, str_22138, tuple_22139], **kwargs_22142)


# Call to add_newdoc(...): (line 4711)
# Processing the call arguments (line 4711)
str_22145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4711, 11), 'str', 'numpy.core.multiarray')
str_22146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4711, 36), 'str', 'ndarray')

# Obtaining an instance of the builtin type 'tuple' (line 4711)
tuple_22147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4711, 48), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 4711)
# Adding element type (line 4711)
str_22148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4711, 48), 'str', 'var')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 4711, 48), tuple_22147, str_22148)
# Adding element type (line 4711)
str_22149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4723, (-1)), 'str', '\n    a.var(axis=None, dtype=None, out=None, ddof=0, keepdims=False)\n\n    Returns the variance of the array elements, along given axis.\n\n    Refer to `numpy.var` for full documentation.\n\n    See Also\n    --------\n    numpy.var : equivalent function\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 4711, 48), tuple_22147, str_22149)

# Processing the call keyword arguments (line 4711)
kwargs_22150 = {}
# Getting the type of 'add_newdoc' (line 4711)
add_newdoc_22144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4711, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 4711)
add_newdoc_call_result_22151 = invoke(stypy.reporting.localization.Localization(__file__, 4711, 0), add_newdoc_22144, *[str_22145, str_22146, tuple_22147], **kwargs_22150)


# Call to add_newdoc(...): (line 4726)
# Processing the call arguments (line 4726)
str_22153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4726, 11), 'str', 'numpy.core.multiarray')
str_22154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4726, 36), 'str', 'ndarray')

# Obtaining an instance of the builtin type 'tuple' (line 4726)
tuple_22155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4726, 48), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 4726)
# Adding element type (line 4726)
str_22156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4726, 48), 'str', 'view')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 4726, 48), tuple_22155, str_22156)
# Adding element type (line 4726)
str_22157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4823, (-1)), 'str', '\n    a.view(dtype=None, type=None)\n\n    New view of array with the same data.\n\n    Parameters\n    ----------\n    dtype : data-type or ndarray sub-class, optional\n        Data-type descriptor of the returned view, e.g., float32 or int16. The\n        default, None, results in the view having the same data-type as `a`.\n        This argument can also be specified as an ndarray sub-class, which\n        then specifies the type of the returned object (this is equivalent to\n        setting the ``type`` parameter).\n    type : Python type, optional\n        Type of the returned view, e.g., ndarray or matrix.  Again, the\n        default None results in type preservation.\n\n    Notes\n    -----\n    ``a.view()`` is used two different ways:\n\n    ``a.view(some_dtype)`` or ``a.view(dtype=some_dtype)`` constructs a view\n    of the array\'s memory with a different data-type.  This can cause a\n    reinterpretation of the bytes of memory.\n\n    ``a.view(ndarray_subclass)`` or ``a.view(type=ndarray_subclass)`` just\n    returns an instance of `ndarray_subclass` that looks at the same array\n    (same shape, dtype, etc.)  This does not cause a reinterpretation of the\n    memory.\n\n    For ``a.view(some_dtype)``, if ``some_dtype`` has a different number of\n    bytes per entry than the previous dtype (for example, converting a\n    regular array to a structured array), then the behavior of the view\n    cannot be predicted just from the superficial appearance of ``a`` (shown\n    by ``print(a)``). It also depends on exactly how ``a`` is stored in\n    memory. Therefore if ``a`` is C-ordered versus fortran-ordered, versus\n    defined as a slice or transpose, etc., the view may give different\n    results.\n\n\n    Examples\n    --------\n    >>> x = np.array([(1, 2)], dtype=[(\'a\', np.int8), (\'b\', np.int8)])\n\n    Viewing array data using a different type and dtype:\n\n    >>> y = x.view(dtype=np.int16, type=np.matrix)\n    >>> y\n    matrix([[513]], dtype=int16)\n    >>> print(type(y))\n    <class \'numpy.matrixlib.defmatrix.matrix\'>\n\n    Creating a view on a structured array so it can be used in calculations\n\n    >>> x = np.array([(1, 2),(3,4)], dtype=[(\'a\', np.int8), (\'b\', np.int8)])\n    >>> xv = x.view(dtype=np.int8).reshape(-1,2)\n    >>> xv\n    array([[1, 2],\n           [3, 4]], dtype=int8)\n    >>> xv.mean(0)\n    array([ 2.,  3.])\n\n    Making changes to the view changes the underlying array\n\n    >>> xv[0,1] = 20\n    >>> print(x)\n    [(1, 20) (3, 4)]\n\n    Using a view to convert an array to a recarray:\n\n    >>> z = x.view(np.recarray)\n    >>> z.a\n    array([1], dtype=int8)\n\n    Views share data:\n\n    >>> x[0] = (9, 10)\n    >>> z[0]\n    (9, 10)\n\n    Views that change the dtype size (bytes per entry) should normally be\n    avoided on arrays defined by slices, transposes, fortran-ordering, etc.:\n\n    >>> x = np.array([[1,2,3],[4,5,6]], dtype=np.int16)\n    >>> y = x[:, 0:2]\n    >>> y\n    array([[1, 2],\n           [4, 5]], dtype=int16)\n    >>> y.view(dtype=[(\'width\', np.int16), (\'length\', np.int16)])\n    Traceback (most recent call last):\n      File "<stdin>", line 1, in <module>\n    ValueError: new type not compatible with array.\n    >>> z = y.copy()\n    >>> z.view(dtype=[(\'width\', np.int16), (\'length\', np.int16)])\n    array([[(1, 2)],\n           [(4, 5)]], dtype=[(\'width\', \'<i2\'), (\'length\', \'<i2\')])\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 4726, 48), tuple_22155, str_22157)

# Processing the call keyword arguments (line 4726)
kwargs_22158 = {}
# Getting the type of 'add_newdoc' (line 4726)
add_newdoc_22152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4726, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 4726)
add_newdoc_call_result_22159 = invoke(stypy.reporting.localization.Localization(__file__, 4726, 0), add_newdoc_22152, *[str_22153, str_22154, tuple_22155], **kwargs_22158)


# Call to add_newdoc(...): (line 4832)
# Processing the call arguments (line 4832)
str_22161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4832, 11), 'str', 'numpy.core.umath')
str_22162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4832, 31), 'str', 'frompyfunc')
str_22163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4870, (-1)), 'str', "\n    frompyfunc(func, nin, nout)\n\n    Takes an arbitrary Python function and returns a Numpy ufunc.\n\n    Can be used, for example, to add broadcasting to a built-in Python\n    function (see Examples section).\n\n    Parameters\n    ----------\n    func : Python function object\n        An arbitrary Python function.\n    nin : int\n        The number of input arguments.\n    nout : int\n        The number of objects returned by `func`.\n\n    Returns\n    -------\n    out : ufunc\n        Returns a Numpy universal function (``ufunc``) object.\n\n    Notes\n    -----\n    The returned ufunc always returns PyObject arrays.\n\n    Examples\n    --------\n    Use frompyfunc to add broadcasting to the Python function ``oct``:\n\n    >>> oct_array = np.frompyfunc(oct, 1, 1)\n    >>> oct_array(np.array((10, 30, 100)))\n    array([012, 036, 0144], dtype=object)\n    >>> np.array((oct(10), oct(30), oct(100))) # for comparison\n    array(['012', '036', '0144'],\n          dtype='|S4')\n\n    ")
# Processing the call keyword arguments (line 4832)
kwargs_22164 = {}
# Getting the type of 'add_newdoc' (line 4832)
add_newdoc_22160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4832, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 4832)
add_newdoc_call_result_22165 = invoke(stypy.reporting.localization.Localization(__file__, 4832, 0), add_newdoc_22160, *[str_22161, str_22162, str_22163], **kwargs_22164)


# Call to add_newdoc(...): (line 4872)
# Processing the call arguments (line 4872)
str_22167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4872, 11), 'str', 'numpy.core.umath')
str_22168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4872, 31), 'str', 'geterrobj')
str_22169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4934, (-1)), 'str', '\n    geterrobj()\n\n    Return the current object that defines floating-point error handling.\n\n    The error object contains all information that defines the error handling\n    behavior in Numpy. `geterrobj` is used internally by the other\n    functions that get and set error handling behavior (`geterr`, `seterr`,\n    `geterrcall`, `seterrcall`).\n\n    Returns\n    -------\n    errobj : list\n        The error object, a list containing three elements:\n        [internal numpy buffer size, error mask, error callback function].\n\n        The error mask is a single integer that holds the treatment information\n        on all four floating point errors. The information for each error type\n        is contained in three bits of the integer. If we print it in base 8, we\n        can see what treatment is set for "invalid", "under", "over", and\n        "divide" (in that order). The printed string can be interpreted with\n\n        * 0 : \'ignore\'\n        * 1 : \'warn\'\n        * 2 : \'raise\'\n        * 3 : \'call\'\n        * 4 : \'print\'\n        * 5 : \'log\'\n\n    See Also\n    --------\n    seterrobj, seterr, geterr, seterrcall, geterrcall\n    getbufsize, setbufsize\n\n    Notes\n    -----\n    For complete documentation of the types of floating-point exceptions and\n    treatment options, see `seterr`.\n\n    Examples\n    --------\n    >>> np.geterrobj()  # first get the defaults\n    [10000, 0, None]\n\n    >>> def err_handler(type, flag):\n    ...     print("Floating point error (%s), with flag %s" % (type, flag))\n    ...\n    >>> old_bufsize = np.setbufsize(20000)\n    >>> old_err = np.seterr(divide=\'raise\')\n    >>> old_handler = np.seterrcall(err_handler)\n    >>> np.geterrobj()\n    [20000, 2, <function err_handler at 0x91dcaac>]\n\n    >>> old_err = np.seterr(all=\'ignore\')\n    >>> np.base_repr(np.geterrobj()[1], 8)\n    \'0\'\n    >>> old_err = np.seterr(divide=\'warn\', over=\'log\', under=\'call\',\n                            invalid=\'print\')\n    >>> np.base_repr(np.geterrobj()[1], 8)\n    \'4351\'\n\n    ')
# Processing the call keyword arguments (line 4872)
kwargs_22170 = {}
# Getting the type of 'add_newdoc' (line 4872)
add_newdoc_22166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4872, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 4872)
add_newdoc_call_result_22171 = invoke(stypy.reporting.localization.Localization(__file__, 4872, 0), add_newdoc_22166, *[str_22167, str_22168, str_22169], **kwargs_22170)


# Call to add_newdoc(...): (line 4936)
# Processing the call arguments (line 4936)
str_22173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4936, 11), 'str', 'numpy.core.umath')
str_22174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4936, 31), 'str', 'seterrobj')
str_22175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4993, (-1)), 'str', '\n    seterrobj(errobj)\n\n    Set the object that defines floating-point error handling.\n\n    The error object contains all information that defines the error handling\n    behavior in Numpy. `seterrobj` is used internally by the other\n    functions that set error handling behavior (`seterr`, `seterrcall`).\n\n    Parameters\n    ----------\n    errobj : list\n        The error object, a list containing three elements:\n        [internal numpy buffer size, error mask, error callback function].\n\n        The error mask is a single integer that holds the treatment information\n        on all four floating point errors. The information for each error type\n        is contained in three bits of the integer. If we print it in base 8, we\n        can see what treatment is set for "invalid", "under", "over", and\n        "divide" (in that order). The printed string can be interpreted with\n\n        * 0 : \'ignore\'\n        * 1 : \'warn\'\n        * 2 : \'raise\'\n        * 3 : \'call\'\n        * 4 : \'print\'\n        * 5 : \'log\'\n\n    See Also\n    --------\n    geterrobj, seterr, geterr, seterrcall, geterrcall\n    getbufsize, setbufsize\n\n    Notes\n    -----\n    For complete documentation of the types of floating-point exceptions and\n    treatment options, see `seterr`.\n\n    Examples\n    --------\n    >>> old_errobj = np.geterrobj()  # first get the defaults\n    >>> old_errobj\n    [10000, 0, None]\n\n    >>> def err_handler(type, flag):\n    ...     print("Floating point error (%s), with flag %s" % (type, flag))\n    ...\n    >>> new_errobj = [20000, 12, err_handler]\n    >>> np.seterrobj(new_errobj)\n    >>> np.base_repr(12, 8)  # int for divide=4 (\'print\') and over=1 (\'warn\')\n    \'14\'\n    >>> np.geterr()\n    {\'over\': \'warn\', \'divide\': \'print\', \'invalid\': \'ignore\', \'under\': \'ignore\'}\n    >>> np.geterrcall() is err_handler\n    True\n\n    ')
# Processing the call keyword arguments (line 4936)
kwargs_22176 = {}
# Getting the type of 'add_newdoc' (line 4936)
add_newdoc_22172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4936, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 4936)
add_newdoc_call_result_22177 = invoke(stypy.reporting.localization.Localization(__file__, 4936, 0), add_newdoc_22172, *[str_22173, str_22174, str_22175], **kwargs_22176)


# Call to add_newdoc(...): (line 5002)
# Processing the call arguments (line 5002)
str_22179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5002, 11), 'str', 'numpy.core.multiarray')
str_22180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5002, 36), 'str', 'digitize')
str_22181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5080, (-1)), 'str', '\n    digitize(x, bins, right=False)\n\n    Return the indices of the bins to which each value in input array belongs.\n\n    Each index ``i`` returned is such that ``bins[i-1] <= x < bins[i]`` if\n    `bins` is monotonically increasing, or ``bins[i-1] > x >= bins[i]`` if\n    `bins` is monotonically decreasing. If values in `x` are beyond the\n    bounds of `bins`, 0 or ``len(bins)`` is returned as appropriate. If right\n    is True, then the right bin is closed so that the index ``i`` is such\n    that ``bins[i-1] < x <= bins[i]`` or bins[i-1] >= x > bins[i]`` if `bins`\n    is monotonically increasing or decreasing, respectively.\n\n    Parameters\n    ----------\n    x : array_like\n        Input array to be binned. Prior to Numpy 1.10.0, this array had to\n        be 1-dimensional, but can now have any shape.\n    bins : array_like\n        Array of bins. It has to be 1-dimensional and monotonic.\n    right : bool, optional\n        Indicating whether the intervals include the right or the left bin\n        edge. Default behavior is (right==False) indicating that the interval\n        does not include the right edge. The left bin end is open in this\n        case, i.e., bins[i-1] <= x < bins[i] is the default behavior for\n        monotonically increasing bins.\n\n    Returns\n    -------\n    out : ndarray of ints\n        Output array of indices, of same shape as `x`.\n\n    Raises\n    ------\n    ValueError\n        If `bins` is not monotonic.\n    TypeError\n        If the type of the input is complex.\n\n    See Also\n    --------\n    bincount, histogram, unique\n\n    Notes\n    -----\n    If values in `x` are such that they fall outside the bin range,\n    attempting to index `bins` with the indices that `digitize` returns\n    will result in an IndexError.\n\n    .. versionadded:: 1.10.0\n\n    `np.digitize` is  implemented in terms of `np.searchsorted`. This means\n    that a binary search is used to bin the values, which scales much better\n    for larger number of bins than the previous linear search. It also removes\n    the requirement for the input array to be 1-dimensional.\n\n    Examples\n    --------\n    >>> x = np.array([0.2, 6.4, 3.0, 1.6])\n    >>> bins = np.array([0.0, 1.0, 2.5, 4.0, 10.0])\n    >>> inds = np.digitize(x, bins)\n    >>> inds\n    array([1, 4, 3, 2])\n    >>> for n in range(x.size):\n    ...   print(bins[inds[n]-1], "<=", x[n], "<", bins[inds[n]])\n    ...\n    0.0 <= 0.2 < 1.0\n    4.0 <= 6.4 < 10.0\n    2.5 <= 3.0 < 4.0\n    1.0 <= 1.6 < 2.5\n\n    >>> x = np.array([1.2, 10.0, 12.4, 15.5, 20.])\n    >>> bins = np.array([0, 5, 10, 15, 20])\n    >>> np.digitize(x,bins,right=True)\n    array([1, 2, 3, 4, 4])\n    >>> np.digitize(x,bins,right=False)\n    array([1, 3, 3, 4, 5])\n    ')
# Processing the call keyword arguments (line 5002)
kwargs_22182 = {}
# Getting the type of 'add_newdoc' (line 5002)
add_newdoc_22178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5002, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 5002)
add_newdoc_call_result_22183 = invoke(stypy.reporting.localization.Localization(__file__, 5002, 0), add_newdoc_22178, *[str_22179, str_22180, str_22181], **kwargs_22182)


# Call to add_newdoc(...): (line 5082)
# Processing the call arguments (line 5082)
str_22185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5082, 11), 'str', 'numpy.core.multiarray')
str_22186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5082, 36), 'str', 'bincount')
str_22187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5153, (-1)), 'str', '\n    bincount(x, weights=None, minlength=None)\n\n    Count number of occurrences of each value in array of non-negative ints.\n\n    The number of bins (of size 1) is one larger than the largest value in\n    `x`. If `minlength` is specified, there will be at least this number\n    of bins in the output array (though it will be longer if necessary,\n    depending on the contents of `x`).\n    Each bin gives the number of occurrences of its index value in `x`.\n    If `weights` is specified the input array is weighted by it, i.e. if a\n    value ``n`` is found at position ``i``, ``out[n] += weight[i]`` instead\n    of ``out[n] += 1``.\n\n    Parameters\n    ----------\n    x : array_like, 1 dimension, nonnegative ints\n        Input array.\n    weights : array_like, optional\n        Weights, array of the same shape as `x`.\n    minlength : int, optional\n        A minimum number of bins for the output array.\n\n        .. versionadded:: 1.6.0\n\n    Returns\n    -------\n    out : ndarray of ints\n        The result of binning the input array.\n        The length of `out` is equal to ``np.amax(x)+1``.\n\n    Raises\n    ------\n    ValueError\n        If the input is not 1-dimensional, or contains elements with negative\n        values, or if `minlength` is non-positive.\n    TypeError\n        If the type of the input is float or complex.\n\n    See Also\n    --------\n    histogram, digitize, unique\n\n    Examples\n    --------\n    >>> np.bincount(np.arange(5))\n    array([1, 1, 1, 1, 1])\n    >>> np.bincount(np.array([0, 1, 1, 3, 2, 1, 7]))\n    array([1, 3, 1, 1, 0, 0, 0, 1])\n\n    >>> x = np.array([0, 1, 1, 3, 2, 1, 7, 23])\n    >>> np.bincount(x).size == np.amax(x)+1\n    True\n\n    The input array needs to be of integer dtype, otherwise a\n    TypeError is raised:\n\n    >>> np.bincount(np.arange(5, dtype=np.float))\n    Traceback (most recent call last):\n      File "<stdin>", line 1, in <module>\n    TypeError: array cannot be safely cast to required type\n\n    A possible use of ``bincount`` is to perform sums over\n    variable-size chunks of an array, using the ``weights`` keyword.\n\n    >>> w = np.array([0.3, 0.5, 0.2, 0.7, 1., -0.6]) # weights\n    >>> x = np.array([0, 1, 1, 2, 2, 2])\n    >>> np.bincount(x,  weights=w)\n    array([ 0.3,  0.7,  1.1])\n\n    ')
# Processing the call keyword arguments (line 5082)
kwargs_22188 = {}
# Getting the type of 'add_newdoc' (line 5082)
add_newdoc_22184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5082, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 5082)
add_newdoc_call_result_22189 = invoke(stypy.reporting.localization.Localization(__file__, 5082, 0), add_newdoc_22184, *[str_22185, str_22186, str_22187], **kwargs_22188)


# Call to add_newdoc(...): (line 5155)
# Processing the call arguments (line 5155)
str_22191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5155, 11), 'str', 'numpy.core.multiarray')
str_22192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5155, 36), 'str', 'ravel_multi_index')
str_22193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5211, (-1)), 'str', "\n    ravel_multi_index(multi_index, dims, mode='raise', order='C')\n\n    Converts a tuple of index arrays into an array of flat\n    indices, applying boundary modes to the multi-index.\n\n    Parameters\n    ----------\n    multi_index : tuple of array_like\n        A tuple of integer arrays, one array for each dimension.\n    dims : tuple of ints\n        The shape of array into which the indices from ``multi_index`` apply.\n    mode : {'raise', 'wrap', 'clip'}, optional\n        Specifies how out-of-bounds indices are handled.  Can specify\n        either one mode or a tuple of modes, one mode per index.\n\n        * 'raise' -- raise an error (default)\n        * 'wrap' -- wrap around\n        * 'clip' -- clip to the range\n\n        In 'clip' mode, a negative index which would normally\n        wrap will clip to 0 instead.\n    order : {'C', 'F'}, optional\n        Determines whether the multi-index should be viewed as\n        indexing in row-major (C-style) or column-major\n        (Fortran-style) order.\n\n    Returns\n    -------\n    raveled_indices : ndarray\n        An array of indices into the flattened version of an array\n        of dimensions ``dims``.\n\n    See Also\n    --------\n    unravel_index\n\n    Notes\n    -----\n    .. versionadded:: 1.6.0\n\n    Examples\n    --------\n    >>> arr = np.array([[3,6,6],[4,5,1]])\n    >>> np.ravel_multi_index(arr, (7,6))\n    array([22, 41, 37])\n    >>> np.ravel_multi_index(arr, (7,6), order='F')\n    array([31, 41, 13])\n    >>> np.ravel_multi_index(arr, (4,6), mode='clip')\n    array([22, 23, 19])\n    >>> np.ravel_multi_index(arr, (4,4), mode=('clip','wrap'))\n    array([12, 13, 13])\n\n    >>> np.ravel_multi_index((3,1,4,1), (6,7,8,9))\n    1621\n    ")
# Processing the call keyword arguments (line 5155)
kwargs_22194 = {}
# Getting the type of 'add_newdoc' (line 5155)
add_newdoc_22190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5155, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 5155)
add_newdoc_call_result_22195 = invoke(stypy.reporting.localization.Localization(__file__, 5155, 0), add_newdoc_22190, *[str_22191, str_22192, str_22193], **kwargs_22194)


# Call to add_newdoc(...): (line 5213)
# Processing the call arguments (line 5213)
str_22197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5213, 11), 'str', 'numpy.core.multiarray')
str_22198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5213, 36), 'str', 'unravel_index')
str_22199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5254, (-1)), 'str', "\n    unravel_index(indices, dims, order='C')\n\n    Converts a flat index or array of flat indices into a tuple\n    of coordinate arrays.\n\n    Parameters\n    ----------\n    indices : array_like\n        An integer array whose elements are indices into the flattened\n        version of an array of dimensions ``dims``. Before version 1.6.0,\n        this function accepted just one index value.\n    dims : tuple of ints\n        The shape of the array to use for unraveling ``indices``.\n    order : {'C', 'F'}, optional\n        Determines whether the indices should be viewed as indexing in\n        row-major (C-style) or column-major (Fortran-style) order.\n\n        .. versionadded:: 1.6.0\n\n    Returns\n    -------\n    unraveled_coords : tuple of ndarray\n        Each array in the tuple has the same shape as the ``indices``\n        array.\n\n    See Also\n    --------\n    ravel_multi_index\n\n    Examples\n    --------\n    >>> np.unravel_index([22, 41, 37], (7,6))\n    (array([3, 6, 6]), array([4, 5, 1]))\n    >>> np.unravel_index([31, 41, 13], (7,6), order='F')\n    (array([3, 6, 6]), array([4, 5, 1]))\n\n    >>> np.unravel_index(1621, (6,7,8,9))\n    (3, 1, 4, 1)\n\n    ")
# Processing the call keyword arguments (line 5213)
kwargs_22200 = {}
# Getting the type of 'add_newdoc' (line 5213)
add_newdoc_22196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5213, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 5213)
add_newdoc_call_result_22201 = invoke(stypy.reporting.localization.Localization(__file__, 5213, 0), add_newdoc_22196, *[str_22197, str_22198, str_22199], **kwargs_22200)


# Call to add_newdoc(...): (line 5256)
# Processing the call arguments (line 5256)
str_22203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5256, 11), 'str', 'numpy.core.multiarray')
str_22204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5256, 36), 'str', 'add_docstring')
str_22205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5264, (-1)), 'str', '\n    add_docstring(obj, docstring)\n\n    Add a docstring to a built-in obj if possible.\n    If the obj already has a docstring raise a RuntimeError\n    If this routine does not know how to add a docstring to the object\n    raise a TypeError\n    ')
# Processing the call keyword arguments (line 5256)
kwargs_22206 = {}
# Getting the type of 'add_newdoc' (line 5256)
add_newdoc_22202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5256, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 5256)
add_newdoc_call_result_22207 = invoke(stypy.reporting.localization.Localization(__file__, 5256, 0), add_newdoc_22202, *[str_22203, str_22204, str_22205], **kwargs_22206)


# Call to add_newdoc(...): (line 5266)
# Processing the call arguments (line 5266)
str_22209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5266, 11), 'str', 'numpy.core.umath')
str_22210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5266, 31), 'str', '_add_newdoc_ufunc')
str_22211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5290, (-1)), 'str', '\n    add_ufunc_docstring(ufunc, new_docstring)\n\n    Replace the docstring for a ufunc with new_docstring.\n    This method will only work if the current docstring for\n    the ufunc is NULL. (At the C level, i.e. when ufunc->doc is NULL.)\n\n    Parameters\n    ----------\n    ufunc : numpy.ufunc\n        A ufunc whose current doc is NULL.\n    new_docstring : string\n        The new docstring for the ufunc.\n\n    Notes\n    -----\n    This method allocates memory for new_docstring on\n    the heap. Technically this creates a mempory leak, since this\n    memory will not be reclaimed until the end of the program\n    even if the ufunc itself is removed. However this will only\n    be a problem if the user is repeatedly creating ufuncs with\n    no documentation, adding documentation via add_newdoc_ufunc,\n    and then throwing away the ufunc.\n    ')
# Processing the call keyword arguments (line 5266)
kwargs_22212 = {}
# Getting the type of 'add_newdoc' (line 5266)
add_newdoc_22208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5266, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 5266)
add_newdoc_call_result_22213 = invoke(stypy.reporting.localization.Localization(__file__, 5266, 0), add_newdoc_22208, *[str_22209, str_22210, str_22211], **kwargs_22212)


# Call to add_newdoc(...): (line 5292)
# Processing the call arguments (line 5292)
str_22215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5292, 11), 'str', 'numpy.core.multiarray')
str_22216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5292, 36), 'str', 'packbits')
str_22217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5334, (-1)), 'str', '\n    packbits(myarray, axis=None)\n\n    Packs the elements of a binary-valued array into bits in a uint8 array.\n\n    The result is padded to full bytes by inserting zero bits at the end.\n\n    Parameters\n    ----------\n    myarray : array_like\n        An integer type array whose elements should be packed to bits.\n    axis : int, optional\n        The dimension over which bit-packing is done.\n        ``None`` implies packing the flattened array.\n\n    Returns\n    -------\n    packed : ndarray\n        Array of type uint8 whose elements represent bits corresponding to the\n        logical (0 or nonzero) value of the input elements. The shape of\n        `packed` has the same number of dimensions as the input (unless `axis`\n        is None, in which case the output is 1-D).\n\n    See Also\n    --------\n    unpackbits: Unpacks elements of a uint8 array into a binary-valued output\n                array.\n\n    Examples\n    --------\n    >>> a = np.array([[[1,0,1],\n    ...                [0,1,0]],\n    ...               [[1,1,0],\n    ...                [0,0,1]]])\n    >>> b = np.packbits(a, axis=-1)\n    >>> b\n    array([[[160],[64]],[[192],[32]]], dtype=uint8)\n\n    Note that in binary 160 = 1010 0000, 64 = 0100 0000, 192 = 1100 0000,\n    and 32 = 0010 0000.\n\n    ')
# Processing the call keyword arguments (line 5292)
kwargs_22218 = {}
# Getting the type of 'add_newdoc' (line 5292)
add_newdoc_22214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5292, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 5292)
add_newdoc_call_result_22219 = invoke(stypy.reporting.localization.Localization(__file__, 5292, 0), add_newdoc_22214, *[str_22215, str_22216, str_22217], **kwargs_22218)


# Call to add_newdoc(...): (line 5336)
# Processing the call arguments (line 5336)
str_22221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5336, 11), 'str', 'numpy.core.multiarray')
str_22222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5336, 36), 'str', 'unpackbits')
str_22223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5377, (-1)), 'str', '\n    unpackbits(myarray, axis=None)\n\n    Unpacks elements of a uint8 array into a binary-valued output array.\n\n    Each element of `myarray` represents a bit-field that should be unpacked\n    into a binary-valued output array. The shape of the output array is either\n    1-D (if `axis` is None) or the same shape as the input array with unpacking\n    done along the axis specified.\n\n    Parameters\n    ----------\n    myarray : ndarray, uint8 type\n       Input array.\n    axis : int, optional\n       Unpacks along this axis.\n\n    Returns\n    -------\n    unpacked : ndarray, uint8 type\n       The elements are binary-valued (0 or 1).\n\n    See Also\n    --------\n    packbits : Packs the elements of a binary-valued array into bits in a uint8\n               array.\n\n    Examples\n    --------\n    >>> a = np.array([[2], [7], [23]], dtype=np.uint8)\n    >>> a\n    array([[ 2],\n           [ 7],\n           [23]], dtype=uint8)\n    >>> b = np.unpackbits(a, axis=1)\n    >>> b\n    array([[0, 0, 0, 0, 0, 0, 1, 0],\n           [0, 0, 0, 0, 0, 1, 1, 1],\n           [0, 0, 0, 1, 0, 1, 1, 1]], dtype=uint8)\n\n    ')
# Processing the call keyword arguments (line 5336)
kwargs_22224 = {}
# Getting the type of 'add_newdoc' (line 5336)
add_newdoc_22220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5336, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 5336)
add_newdoc_call_result_22225 = invoke(stypy.reporting.localization.Localization(__file__, 5336, 0), add_newdoc_22220, *[str_22221, str_22222, str_22223], **kwargs_22224)


# Call to add_newdoc(...): (line 5393)
# Processing the call arguments (line 5393)
str_22227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5393, 11), 'str', 'numpy.core')
str_22228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5393, 25), 'str', 'ufunc')
str_22229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5452, (-1)), 'str', '\n    Functions that operate element by element on whole arrays.\n\n    To see the documentation for a specific ufunc, use np.info().  For\n    example, np.info(np.sin).  Because ufuncs are written in C\n    (for speed) and linked into Python with NumPy\'s ufunc facility,\n    Python\'s help() function finds this page whenever help() is called\n    on a ufunc.\n\n    A detailed explanation of ufuncs can be found in the "ufuncs.rst"\n    file in the NumPy reference guide.\n\n    Unary ufuncs:\n    =============\n\n    op(X, out=None)\n    Apply op to X elementwise\n\n    Parameters\n    ----------\n    X : array_like\n        Input array.\n    out : array_like\n        An array to store the output. Must be the same shape as `X`.\n\n    Returns\n    -------\n    r : array_like\n        `r` will have the same shape as `X`; if out is provided, `r`\n        will be equal to out.\n\n    Binary ufuncs:\n    ==============\n\n    op(X, Y, out=None)\n    Apply `op` to `X` and `Y` elementwise. May "broadcast" to make\n    the shapes of `X` and `Y` congruent.\n\n    The broadcasting rules are:\n\n    * Dimensions of length 1 may be prepended to either array.\n    * Arrays may be repeated along dimensions of length 1.\n\n    Parameters\n    ----------\n    X : array_like\n        First input array.\n    Y : array_like\n        Second input array.\n    out : array_like\n        An array to store the output. Must be the same shape as the\n        output would have.\n\n    Returns\n    -------\n    r : array_like\n        The return value; if out is provided, `r` will be equal to out.\n\n    ')
# Processing the call keyword arguments (line 5393)
kwargs_22230 = {}
# Getting the type of 'add_newdoc' (line 5393)
add_newdoc_22226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5393, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 5393)
add_newdoc_call_result_22231 = invoke(stypy.reporting.localization.Localization(__file__, 5393, 0), add_newdoc_22226, *[str_22227, str_22228, str_22229], **kwargs_22230)


# Call to add_newdoc(...): (line 5461)
# Processing the call arguments (line 5461)
str_22233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5461, 11), 'str', 'numpy.core')
str_22234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5461, 25), 'str', 'ufunc')

# Obtaining an instance of the builtin type 'tuple' (line 5461)
tuple_22235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5461, 35), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 5461)
# Adding element type (line 5461)
str_22236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5461, 35), 'str', 'identity')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5461, 35), tuple_22235, str_22236)
# Adding element type (line 5461)
str_22237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5478, (-1)), 'str', '\n    The identity value.\n\n    Data attribute containing the identity element for the ufunc, if it has one.\n    If it does not, the attribute value is None.\n\n    Examples\n    --------\n    >>> np.add.identity\n    0\n    >>> np.multiply.identity\n    1\n    >>> np.power.identity\n    1\n    >>> print(np.exp.identity)\n    None\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5461, 35), tuple_22235, str_22237)

# Processing the call keyword arguments (line 5461)
kwargs_22238 = {}
# Getting the type of 'add_newdoc' (line 5461)
add_newdoc_22232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5461, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 5461)
add_newdoc_call_result_22239 = invoke(stypy.reporting.localization.Localization(__file__, 5461, 0), add_newdoc_22232, *[str_22233, str_22234, tuple_22235], **kwargs_22238)


# Call to add_newdoc(...): (line 5480)
# Processing the call arguments (line 5480)
str_22241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5480, 11), 'str', 'numpy.core')
str_22242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5480, 25), 'str', 'ufunc')

# Obtaining an instance of the builtin type 'tuple' (line 5480)
tuple_22243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5480, 35), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 5480)
# Adding element type (line 5480)
str_22244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5480, 35), 'str', 'nargs')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5480, 35), tuple_22243, str_22244)
# Adding element type (line 5480)
str_22245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5502, (-1)), 'str', '\n    The number of arguments.\n\n    Data attribute containing the number of arguments the ufunc takes, including\n    optional ones.\n\n    Notes\n    -----\n    Typically this value will be one more than what you might expect because all\n    ufuncs take  the optional "out" argument.\n\n    Examples\n    --------\n    >>> np.add.nargs\n    3\n    >>> np.multiply.nargs\n    3\n    >>> np.power.nargs\n    3\n    >>> np.exp.nargs\n    2\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5480, 35), tuple_22243, str_22245)

# Processing the call keyword arguments (line 5480)
kwargs_22246 = {}
# Getting the type of 'add_newdoc' (line 5480)
add_newdoc_22240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5480, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 5480)
add_newdoc_call_result_22247 = invoke(stypy.reporting.localization.Localization(__file__, 5480, 0), add_newdoc_22240, *[str_22241, str_22242, tuple_22243], **kwargs_22246)


# Call to add_newdoc(...): (line 5504)
# Processing the call arguments (line 5504)
str_22249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5504, 11), 'str', 'numpy.core')
str_22250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5504, 25), 'str', 'ufunc')

# Obtaining an instance of the builtin type 'tuple' (line 5504)
tuple_22251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5504, 35), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 5504)
# Adding element type (line 5504)
str_22252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5504, 35), 'str', 'nin')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5504, 35), tuple_22251, str_22252)
# Adding element type (line 5504)
str_22253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5520, (-1)), 'str', '\n    The number of inputs.\n\n    Data attribute containing the number of arguments the ufunc treats as input.\n\n    Examples\n    --------\n    >>> np.add.nin\n    2\n    >>> np.multiply.nin\n    2\n    >>> np.power.nin\n    2\n    >>> np.exp.nin\n    1\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5504, 35), tuple_22251, str_22253)

# Processing the call keyword arguments (line 5504)
kwargs_22254 = {}
# Getting the type of 'add_newdoc' (line 5504)
add_newdoc_22248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5504, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 5504)
add_newdoc_call_result_22255 = invoke(stypy.reporting.localization.Localization(__file__, 5504, 0), add_newdoc_22248, *[str_22249, str_22250, tuple_22251], **kwargs_22254)


# Call to add_newdoc(...): (line 5522)
# Processing the call arguments (line 5522)
str_22257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5522, 11), 'str', 'numpy.core')
str_22258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5522, 25), 'str', 'ufunc')

# Obtaining an instance of the builtin type 'tuple' (line 5522)
tuple_22259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5522, 35), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 5522)
# Adding element type (line 5522)
str_22260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5522, 35), 'str', 'nout')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5522, 35), tuple_22259, str_22260)
# Adding element type (line 5522)
str_22261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5543, (-1)), 'str', '\n    The number of outputs.\n\n    Data attribute containing the number of arguments the ufunc treats as output.\n\n    Notes\n    -----\n    Since all ufuncs can take output arguments, this will always be (at least) 1.\n\n    Examples\n    --------\n    >>> np.add.nout\n    1\n    >>> np.multiply.nout\n    1\n    >>> np.power.nout\n    1\n    >>> np.exp.nout\n    1\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5522, 35), tuple_22259, str_22261)

# Processing the call keyword arguments (line 5522)
kwargs_22262 = {}
# Getting the type of 'add_newdoc' (line 5522)
add_newdoc_22256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5522, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 5522)
add_newdoc_call_result_22263 = invoke(stypy.reporting.localization.Localization(__file__, 5522, 0), add_newdoc_22256, *[str_22257, str_22258, tuple_22259], **kwargs_22262)


# Call to add_newdoc(...): (line 5545)
# Processing the call arguments (line 5545)
str_22265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5545, 11), 'str', 'numpy.core')
str_22266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5545, 25), 'str', 'ufunc')

# Obtaining an instance of the builtin type 'tuple' (line 5545)
tuple_22267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5545, 35), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 5545)
# Adding element type (line 5545)
str_22268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5545, 35), 'str', 'ntypes')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5545, 35), tuple_22267, str_22268)
# Adding element type (line 5545)
str_22269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5569, (-1)), 'str', '\n    The number of types.\n\n    The number of numerical NumPy types - of which there are 18 total - on which\n    the ufunc can operate.\n\n    See Also\n    --------\n    numpy.ufunc.types\n\n    Examples\n    --------\n    >>> np.add.ntypes\n    18\n    >>> np.multiply.ntypes\n    18\n    >>> np.power.ntypes\n    17\n    >>> np.exp.ntypes\n    7\n    >>> np.remainder.ntypes\n    14\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5545, 35), tuple_22267, str_22269)

# Processing the call keyword arguments (line 5545)
kwargs_22270 = {}
# Getting the type of 'add_newdoc' (line 5545)
add_newdoc_22264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5545, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 5545)
add_newdoc_call_result_22271 = invoke(stypy.reporting.localization.Localization(__file__, 5545, 0), add_newdoc_22264, *[str_22265, str_22266, tuple_22267], **kwargs_22270)


# Call to add_newdoc(...): (line 5571)
# Processing the call arguments (line 5571)
str_22273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5571, 11), 'str', 'numpy.core')
str_22274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5571, 25), 'str', 'ufunc')

# Obtaining an instance of the builtin type 'tuple' (line 5571)
tuple_22275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5571, 35), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 5571)
# Adding element type (line 5571)
str_22276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5571, 35), 'str', 'types')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5571, 35), tuple_22275, str_22276)
# Adding element type (line 5571)
str_22277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5606, (-1)), 'str', '\n    Returns a list with types grouped input->output.\n\n    Data attribute listing the data-type "Domain-Range" groupings the ufunc can\n    deliver. The data-types are given using the character codes.\n\n    See Also\n    --------\n    numpy.ufunc.ntypes\n\n    Examples\n    --------\n    >>> np.add.types\n    [\'??->?\', \'bb->b\', \'BB->B\', \'hh->h\', \'HH->H\', \'ii->i\', \'II->I\', \'ll->l\',\n    \'LL->L\', \'qq->q\', \'QQ->Q\', \'ff->f\', \'dd->d\', \'gg->g\', \'FF->F\', \'DD->D\',\n    \'GG->G\', \'OO->O\']\n\n    >>> np.multiply.types\n    [\'??->?\', \'bb->b\', \'BB->B\', \'hh->h\', \'HH->H\', \'ii->i\', \'II->I\', \'ll->l\',\n    \'LL->L\', \'qq->q\', \'QQ->Q\', \'ff->f\', \'dd->d\', \'gg->g\', \'FF->F\', \'DD->D\',\n    \'GG->G\', \'OO->O\']\n\n    >>> np.power.types\n    [\'bb->b\', \'BB->B\', \'hh->h\', \'HH->H\', \'ii->i\', \'II->I\', \'ll->l\', \'LL->L\',\n    \'qq->q\', \'QQ->Q\', \'ff->f\', \'dd->d\', \'gg->g\', \'FF->F\', \'DD->D\', \'GG->G\',\n    \'OO->O\']\n\n    >>> np.exp.types\n    [\'f->f\', \'d->d\', \'g->g\', \'F->F\', \'D->D\', \'G->G\', \'O->O\']\n\n    >>> np.remainder.types\n    [\'bb->b\', \'BB->B\', \'hh->h\', \'HH->H\', \'ii->i\', \'II->I\', \'ll->l\', \'LL->L\',\n    \'qq->q\', \'QQ->Q\', \'ff->f\', \'dd->d\', \'gg->g\', \'OO->O\']\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5571, 35), tuple_22275, str_22277)

# Processing the call keyword arguments (line 5571)
kwargs_22278 = {}
# Getting the type of 'add_newdoc' (line 5571)
add_newdoc_22272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5571, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 5571)
add_newdoc_call_result_22279 = invoke(stypy.reporting.localization.Localization(__file__, 5571, 0), add_newdoc_22272, *[str_22273, str_22274, tuple_22275], **kwargs_22278)


# Call to add_newdoc(...): (line 5615)
# Processing the call arguments (line 5615)
str_22281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5615, 11), 'str', 'numpy.core')
str_22282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5615, 25), 'str', 'ufunc')

# Obtaining an instance of the builtin type 'tuple' (line 5615)
tuple_22283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5615, 35), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 5615)
# Adding element type (line 5615)
str_22284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5615, 35), 'str', 'reduce')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5615, 35), tuple_22283, str_22284)
# Adding element type (line 5615)
str_22285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5700, (-1)), 'str', "\n    reduce(a, axis=0, dtype=None, out=None, keepdims=False)\n\n    Reduces `a`'s dimension by one, by applying ufunc along one axis.\n\n    Let :math:`a.shape = (N_0, ..., N_i, ..., N_{M-1})`.  Then\n    :math:`ufunc.reduce(a, axis=i)[k_0, ..,k_{i-1}, k_{i+1}, .., k_{M-1}]` =\n    the result of iterating `j` over :math:`range(N_i)`, cumulatively applying\n    ufunc to each :math:`a[k_0, ..,k_{i-1}, j, k_{i+1}, .., k_{M-1}]`.\n    For a one-dimensional array, reduce produces results equivalent to:\n    ::\n\n     r = op.identity # op = ufunc\n     for i in range(len(A)):\n       r = op(r, A[i])\n     return r\n\n    For example, add.reduce() is equivalent to sum().\n\n    Parameters\n    ----------\n    a : array_like\n        The array to act on.\n    axis : None or int or tuple of ints, optional\n        Axis or axes along which a reduction is performed.\n        The default (`axis` = 0) is perform a reduction over the first\n        dimension of the input array. `axis` may be negative, in\n        which case it counts from the last to the first axis.\n\n        .. versionadded:: 1.7.0\n\n        If this is `None`, a reduction is performed over all the axes.\n        If this is a tuple of ints, a reduction is performed on multiple\n        axes, instead of a single axis or all the axes as before.\n\n        For operations which are either not commutative or not associative,\n        doing a reduction over multiple axes is not well-defined. The\n        ufuncs do not currently raise an exception in this case, but will\n        likely do so in the future.\n    dtype : data-type code, optional\n        The type used to represent the intermediate results. Defaults\n        to the data-type of the output array if this is provided, or\n        the data-type of the input array if no output array is provided.\n    out : ndarray, optional\n        A location into which the result is stored. If not provided, a\n        freshly-allocated array is returned.\n    keepdims : bool, optional\n        If this is set to True, the axes which are reduced are left\n        in the result as dimensions with size one. With this option,\n        the result will broadcast correctly against the original `arr`.\n\n        .. versionadded:: 1.7.0\n\n    Returns\n    -------\n    r : ndarray\n        The reduced array. If `out` was supplied, `r` is a reference to it.\n\n    Examples\n    --------\n    >>> np.multiply.reduce([2,3,5])\n    30\n\n    A multi-dimensional array example:\n\n    >>> X = np.arange(8).reshape((2,2,2))\n    >>> X\n    array([[[0, 1],\n            [2, 3]],\n           [[4, 5],\n            [6, 7]]])\n    >>> np.add.reduce(X, 0)\n    array([[ 4,  6],\n           [ 8, 10]])\n    >>> np.add.reduce(X) # confirm: default axis value is 0\n    array([[ 4,  6],\n           [ 8, 10]])\n    >>> np.add.reduce(X, 1)\n    array([[ 2,  4],\n           [10, 12]])\n    >>> np.add.reduce(X, 2)\n    array([[ 1,  5],\n           [ 9, 13]])\n\n    ")
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5615, 35), tuple_22283, str_22285)

# Processing the call keyword arguments (line 5615)
kwargs_22286 = {}
# Getting the type of 'add_newdoc' (line 5615)
add_newdoc_22280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5615, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 5615)
add_newdoc_call_result_22287 = invoke(stypy.reporting.localization.Localization(__file__, 5615, 0), add_newdoc_22280, *[str_22281, str_22282, tuple_22283], **kwargs_22286)


# Call to add_newdoc(...): (line 5702)
# Processing the call arguments (line 5702)
str_22289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5702, 11), 'str', 'numpy.core')
str_22290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5702, 25), 'str', 'ufunc')

# Obtaining an instance of the builtin type 'tuple' (line 5702)
tuple_22291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5702, 35), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 5702)
# Adding element type (line 5702)
str_22292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5702, 35), 'str', 'accumulate')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5702, 35), tuple_22291, str_22292)
# Adding element type (line 5702)
str_22293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5774, (-1)), 'str', "\n    accumulate(array, axis=0, dtype=None, out=None)\n\n    Accumulate the result of applying the operator to all elements.\n\n    For a one-dimensional array, accumulate produces results equivalent to::\n\n      r = np.empty(len(A))\n      t = op.identity        # op = the ufunc being applied to A's  elements\n      for i in range(len(A)):\n          t = op(t, A[i])\n          r[i] = t\n      return r\n\n    For example, add.accumulate() is equivalent to np.cumsum().\n\n    For a multi-dimensional array, accumulate is applied along only one\n    axis (axis zero by default; see Examples below) so repeated use is\n    necessary if one wants to accumulate over multiple axes.\n\n    Parameters\n    ----------\n    array : array_like\n        The array to act on.\n    axis : int, optional\n        The axis along which to apply the accumulation; default is zero.\n    dtype : data-type code, optional\n        The data-type used to represent the intermediate results. Defaults\n        to the data-type of the output array if such is provided, or the\n        the data-type of the input array if no output array is provided.\n    out : ndarray, optional\n        A location into which the result is stored. If not provided a\n        freshly-allocated array is returned.\n\n    Returns\n    -------\n    r : ndarray\n        The accumulated values. If `out` was supplied, `r` is a reference to\n        `out`.\n\n    Examples\n    --------\n    1-D array examples:\n\n    >>> np.add.accumulate([2, 3, 5])\n    array([ 2,  5, 10])\n    >>> np.multiply.accumulate([2, 3, 5])\n    array([ 2,  6, 30])\n\n    2-D array examples:\n\n    >>> I = np.eye(2)\n    >>> I\n    array([[ 1.,  0.],\n           [ 0.,  1.]])\n\n    Accumulate along axis 0 (rows), down columns:\n\n    >>> np.add.accumulate(I, 0)\n    array([[ 1.,  0.],\n           [ 1.,  1.]])\n    >>> np.add.accumulate(I) # no axis specified = axis zero\n    array([[ 1.,  0.],\n           [ 1.,  1.]])\n\n    Accumulate along axis 1 (columns), through rows:\n\n    >>> np.add.accumulate(I, 1)\n    array([[ 1.,  1.],\n           [ 0.,  1.]])\n\n    ")
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5702, 35), tuple_22291, str_22293)

# Processing the call keyword arguments (line 5702)
kwargs_22294 = {}
# Getting the type of 'add_newdoc' (line 5702)
add_newdoc_22288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5702, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 5702)
add_newdoc_call_result_22295 = invoke(stypy.reporting.localization.Localization(__file__, 5702, 0), add_newdoc_22288, *[str_22289, str_22290, tuple_22291], **kwargs_22294)


# Call to add_newdoc(...): (line 5776)
# Processing the call arguments (line 5776)
str_22297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5776, 11), 'str', 'numpy.core')
str_22298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5776, 25), 'str', 'ufunc')

# Obtaining an instance of the builtin type 'tuple' (line 5776)
tuple_22299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5776, 35), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 5776)
# Adding element type (line 5776)
str_22300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5776, 35), 'str', 'reduceat')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5776, 35), tuple_22299, str_22300)
# Adding element type (line 5776)
str_22301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5876, (-1)), 'str', '\n    reduceat(a, indices, axis=0, dtype=None, out=None)\n\n    Performs a (local) reduce with specified slices over a single axis.\n\n    For i in ``range(len(indices))``, `reduceat` computes\n    ``ufunc.reduce(a[indices[i]:indices[i+1]])``, which becomes the i-th\n    generalized "row" parallel to `axis` in the final result (i.e., in a\n    2-D array, for example, if `axis = 0`, it becomes the i-th row, but if\n    `axis = 1`, it becomes the i-th column).  There are three exceptions to this:\n\n    * when ``i = len(indices) - 1`` (so for the last index),\n      ``indices[i+1] = a.shape[axis]``.\n    * if ``indices[i] >= indices[i + 1]``, the i-th generalized "row" is\n      simply ``a[indices[i]]``.\n    * if ``indices[i] >= len(a)`` or ``indices[i] < 0``, an error is raised.\n\n    The shape of the output depends on the size of `indices`, and may be\n    larger than `a` (this happens if ``len(indices) > a.shape[axis]``).\n\n    Parameters\n    ----------\n    a : array_like\n        The array to act on.\n    indices : array_like\n        Paired indices, comma separated (not colon), specifying slices to\n        reduce.\n    axis : int, optional\n        The axis along which to apply the reduceat.\n    dtype : data-type code, optional\n        The type used to represent the intermediate results. Defaults\n        to the data type of the output array if this is provided, or\n        the data type of the input array if no output array is provided.\n    out : ndarray, optional\n        A location into which the result is stored. If not provided a\n        freshly-allocated array is returned.\n\n    Returns\n    -------\n    r : ndarray\n        The reduced values. If `out` was supplied, `r` is a reference to\n        `out`.\n\n    Notes\n    -----\n    A descriptive example:\n\n    If `a` is 1-D, the function `ufunc.accumulate(a)` is the same as\n    ``ufunc.reduceat(a, indices)[::2]`` where `indices` is\n    ``range(len(array) - 1)`` with a zero placed\n    in every other element:\n    ``indices = zeros(2 * len(a) - 1)``, ``indices[1::2] = range(1, len(a))``.\n\n    Don\'t be fooled by this attribute\'s name: `reduceat(a)` is not\n    necessarily smaller than `a`.\n\n    Examples\n    --------\n    To take the running sum of four successive values:\n\n    >>> np.add.reduceat(np.arange(8),[0,4, 1,5, 2,6, 3,7])[::2]\n    array([ 6, 10, 14, 18])\n\n    A 2-D example:\n\n    >>> x = np.linspace(0, 15, 16).reshape(4,4)\n    >>> x\n    array([[  0.,   1.,   2.,   3.],\n           [  4.,   5.,   6.,   7.],\n           [  8.,   9.,  10.,  11.],\n           [ 12.,  13.,  14.,  15.]])\n\n    ::\n\n     # reduce such that the result has the following five rows:\n     # [row1 + row2 + row3]\n     # [row4]\n     # [row2]\n     # [row3]\n     # [row1 + row2 + row3 + row4]\n\n    >>> np.add.reduceat(x, [0, 3, 1, 2, 0])\n    array([[ 12.,  15.,  18.,  21.],\n           [ 12.,  13.,  14.,  15.],\n           [  4.,   5.,   6.,   7.],\n           [  8.,   9.,  10.,  11.],\n           [ 24.,  28.,  32.,  36.]])\n\n    ::\n\n     # reduce such that result has the following two columns:\n     # [col1 * col2 * col3, col4]\n\n    >>> np.multiply.reduceat(x, [0, 3], 1)\n    array([[    0.,     3.],\n           [  120.,     7.],\n           [  720.,    11.],\n           [ 2184.,    15.]])\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5776, 35), tuple_22299, str_22301)

# Processing the call keyword arguments (line 5776)
kwargs_22302 = {}
# Getting the type of 'add_newdoc' (line 5776)
add_newdoc_22296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5776, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 5776)
add_newdoc_call_result_22303 = invoke(stypy.reporting.localization.Localization(__file__, 5776, 0), add_newdoc_22296, *[str_22297, str_22298, tuple_22299], **kwargs_22302)


# Call to add_newdoc(...): (line 5878)
# Processing the call arguments (line 5878)
str_22305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5878, 11), 'str', 'numpy.core')
str_22306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5878, 25), 'str', 'ufunc')

# Obtaining an instance of the builtin type 'tuple' (line 5878)
tuple_22307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5878, 35), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 5878)
# Adding element type (line 5878)
str_22308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5878, 35), 'str', 'outer')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5878, 35), tuple_22307, str_22308)
# Adding element type (line 5878)
str_22309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5938, (-1)), 'str', '\n    outer(A, B)\n\n    Apply the ufunc `op` to all pairs (a, b) with a in `A` and b in `B`.\n\n    Let ``M = A.ndim``, ``N = B.ndim``. Then the result, `C`, of\n    ``op.outer(A, B)`` is an array of dimension M + N such that:\n\n    .. math:: C[i_0, ..., i_{M-1}, j_0, ..., j_{N-1}] =\n       op(A[i_0, ..., i_{M-1}], B[j_0, ..., j_{N-1}])\n\n    For `A` and `B` one-dimensional, this is equivalent to::\n\n      r = empty(len(A),len(B))\n      for i in range(len(A)):\n          for j in range(len(B)):\n              r[i,j] = op(A[i], B[j]) # op = ufunc in question\n\n    Parameters\n    ----------\n    A : array_like\n        First array\n    B : array_like\n        Second array\n\n    Returns\n    -------\n    r : ndarray\n        Output array\n\n    See Also\n    --------\n    numpy.outer\n\n    Examples\n    --------\n    >>> np.multiply.outer([1, 2, 3], [4, 5, 6])\n    array([[ 4,  5,  6],\n           [ 8, 10, 12],\n           [12, 15, 18]])\n\n    A multi-dimensional example:\n\n    >>> A = np.array([[1, 2, 3], [4, 5, 6]])\n    >>> A.shape\n    (2, 3)\n    >>> B = np.array([[1, 2, 3, 4]])\n    >>> B.shape\n    (1, 4)\n    >>> C = np.multiply.outer(A, B)\n    >>> C.shape; C\n    (2, 3, 1, 4)\n    array([[[[ 1,  2,  3,  4]],\n            [[ 2,  4,  6,  8]],\n            [[ 3,  6,  9, 12]]],\n           [[[ 4,  8, 12, 16]],\n            [[ 5, 10, 15, 20]],\n            [[ 6, 12, 18, 24]]]])\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5878, 35), tuple_22307, str_22309)

# Processing the call keyword arguments (line 5878)
kwargs_22310 = {}
# Getting the type of 'add_newdoc' (line 5878)
add_newdoc_22304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5878, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 5878)
add_newdoc_call_result_22311 = invoke(stypy.reporting.localization.Localization(__file__, 5878, 0), add_newdoc_22304, *[str_22305, str_22306, tuple_22307], **kwargs_22310)


# Call to add_newdoc(...): (line 5940)
# Processing the call arguments (line 5940)
str_22313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5940, 11), 'str', 'numpy.core')
str_22314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5940, 25), 'str', 'ufunc')

# Obtaining an instance of the builtin type 'tuple' (line 5940)
tuple_22315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5940, 35), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 5940)
# Adding element type (line 5940)
str_22316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5940, 35), 'str', 'at')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5940, 35), tuple_22315, str_22316)
# Adding element type (line 5940)
str_22317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5994, (-1)), 'str', "\n    at(a, indices, b=None)\n\n    Performs unbuffered in place operation on operand 'a' for elements\n    specified by 'indices'. For addition ufunc, this method is equivalent to\n    `a[indices] += b`, except that results are accumulated for elements that\n    are indexed more than once. For example, `a[[0,0]] += 1` will only\n    increment the first element once because of buffering, whereas\n    `add.at(a, [0,0], 1)` will increment the first element twice.\n\n    .. versionadded:: 1.8.0\n\n    Parameters\n    ----------\n    a : array_like\n        The array to perform in place operation on.\n    indices : array_like or tuple\n        Array like index object or slice object for indexing into first\n        operand. If first operand has multiple dimensions, indices can be a\n        tuple of array like index objects or slice objects.\n    b : array_like\n        Second operand for ufuncs requiring two operands. Operand must be\n        broadcastable over first operand after indexing or slicing.\n\n    Examples\n    --------\n    Set items 0 and 1 to their negative values:\n\n    >>> a = np.array([1, 2, 3, 4])\n    >>> np.negative.at(a, [0, 1])\n    >>> print(a)\n    array([-1, -2, 3, 4])\n\n    ::\n\n    Increment items 0 and 1, and increment item 2 twice:\n\n    >>> a = np.array([1, 2, 3, 4])\n    >>> np.add.at(a, [0, 1, 2, 2], 1)\n    >>> print(a)\n    array([2, 3, 5, 4])\n\n    ::\n\n    Add items 0 and 1 in first array to second array,\n    and store results in first array:\n\n    >>> a = np.array([1, 2, 3, 4])\n    >>> b = np.array([1, 2])\n    >>> np.add.at(a, [0, 1], b)\n    >>> print(a)\n    array([2, 4, 3, 4])\n\n    ")
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5940, 35), tuple_22315, str_22317)

# Processing the call keyword arguments (line 5940)
kwargs_22318 = {}
# Getting the type of 'add_newdoc' (line 5940)
add_newdoc_22312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5940, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 5940)
add_newdoc_call_result_22319 = invoke(stypy.reporting.localization.Localization(__file__, 5940, 0), add_newdoc_22312, *[str_22313, str_22314, tuple_22315], **kwargs_22318)


# Call to add_newdoc(...): (line 6008)
# Processing the call arguments (line 6008)
str_22321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6008, 11), 'str', 'numpy.core.multiarray')
str_22322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6008, 36), 'str', 'dtype')
str_22323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6091, (-1)), 'str', '\n    dtype(obj, align=False, copy=False)\n\n    Create a data type object.\n\n    A numpy array is homogeneous, and contains elements described by a\n    dtype object. A dtype object can be constructed from different\n    combinations of fundamental numeric types.\n\n    Parameters\n    ----------\n    obj\n        Object to be converted to a data type object.\n    align : bool, optional\n        Add padding to the fields to match what a C compiler would output\n        for a similar C-struct. Can be ``True`` only if `obj` is a dictionary\n        or a comma-separated string. If a struct dtype is being created,\n        this also sets a sticky alignment flag ``isalignedstruct``.\n    copy : bool, optional\n        Make a new copy of the data-type object. If ``False``, the result\n        may just be a reference to a built-in data-type object.\n\n    See also\n    --------\n    result_type\n\n    Examples\n    --------\n    Using array-scalar type:\n\n    >>> np.dtype(np.int16)\n    dtype(\'int16\')\n\n    Structured type, one field name \'f1\', containing int16:\n\n    >>> np.dtype([(\'f1\', np.int16)])\n    dtype([(\'f1\', \'<i2\')])\n\n    Structured type, one field named \'f1\', in itself containing a structured\n    type with one field:\n\n    >>> np.dtype([(\'f1\', [(\'f1\', np.int16)])])\n    dtype([(\'f1\', [(\'f1\', \'<i2\')])])\n\n    Structured type, two fields: the first field contains an unsigned int, the\n    second an int32:\n\n    >>> np.dtype([(\'f1\', np.uint), (\'f2\', np.int32)])\n    dtype([(\'f1\', \'<u4\'), (\'f2\', \'<i4\')])\n\n    Using array-protocol type strings:\n\n    >>> np.dtype([(\'a\',\'f8\'),(\'b\',\'S10\')])\n    dtype([(\'a\', \'<f8\'), (\'b\', \'|S10\')])\n\n    Using comma-separated field formats.  The shape is (2,3):\n\n    >>> np.dtype("i4, (2,3)f8")\n    dtype([(\'f0\', \'<i4\'), (\'f1\', \'<f8\', (2, 3))])\n\n    Using tuples.  ``int`` is a fixed type, 3 the field\'s shape.  ``void``\n    is a flexible type, here of size 10:\n\n    >>> np.dtype([(\'hello\',(np.int,3)),(\'world\',np.void,10)])\n    dtype([(\'hello\', \'<i4\', 3), (\'world\', \'|V10\')])\n\n    Subdivide ``int16`` into 2 ``int8``\'s, called x and y.  0 and 1 are\n    the offsets in bytes:\n\n    >>> np.dtype((np.int16, {\'x\':(np.int8,0), \'y\':(np.int8,1)}))\n    dtype((\'<i2\', [(\'x\', \'|i1\'), (\'y\', \'|i1\')]))\n\n    Using dictionaries.  Two fields named \'gender\' and \'age\':\n\n    >>> np.dtype({\'names\':[\'gender\',\'age\'], \'formats\':[\'S1\',np.uint8]})\n    dtype([(\'gender\', \'|S1\'), (\'age\', \'|u1\')])\n\n    Offsets in bytes, here 0 and 25:\n\n    >>> np.dtype({\'surname\':(\'S25\',0),\'age\':(np.uint8,25)})\n    dtype([(\'surname\', \'|S25\'), (\'age\', \'|u1\')])\n\n    ')
# Processing the call keyword arguments (line 6008)
kwargs_22324 = {}
# Getting the type of 'add_newdoc' (line 6008)
add_newdoc_22320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6008, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 6008)
add_newdoc_call_result_22325 = invoke(stypy.reporting.localization.Localization(__file__, 6008, 0), add_newdoc_22320, *[str_22321, str_22322, str_22323], **kwargs_22324)


# Call to add_newdoc(...): (line 6099)
# Processing the call arguments (line 6099)
str_22327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6099, 11), 'str', 'numpy.core.multiarray')
str_22328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6099, 36), 'str', 'dtype')

# Obtaining an instance of the builtin type 'tuple' (line 6099)
tuple_22329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6099, 46), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 6099)
# Adding element type (line 6099)
str_22330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6099, 46), 'str', 'alignment')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6099, 46), tuple_22329, str_22330)
# Adding element type (line 6099)
str_22331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6105, (-1)), 'str', '\n    The required alignment (bytes) of this data-type according to the compiler.\n\n    More information is available in the C-API section of the manual.\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6099, 46), tuple_22329, str_22331)

# Processing the call keyword arguments (line 6099)
kwargs_22332 = {}
# Getting the type of 'add_newdoc' (line 6099)
add_newdoc_22326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6099, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 6099)
add_newdoc_call_result_22333 = invoke(stypy.reporting.localization.Localization(__file__, 6099, 0), add_newdoc_22326, *[str_22327, str_22328, tuple_22329], **kwargs_22332)


# Call to add_newdoc(...): (line 6107)
# Processing the call arguments (line 6107)
str_22335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6107, 11), 'str', 'numpy.core.multiarray')
str_22336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6107, 36), 'str', 'dtype')

# Obtaining an instance of the builtin type 'tuple' (line 6107)
tuple_22337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6107, 46), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 6107)
# Adding element type (line 6107)
str_22338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6107, 46), 'str', 'byteorder')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6107, 46), tuple_22337, str_22338)
# Adding element type (line 6107)
str_22339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6148, (-1)), 'str', "\n    A character indicating the byte-order of this data-type object.\n\n    One of:\n\n    ===  ==============\n    '='  native\n    '<'  little-endian\n    '>'  big-endian\n    '|'  not applicable\n    ===  ==============\n\n    All built-in data-type objects have byteorder either '=' or '|'.\n\n    Examples\n    --------\n\n    >>> dt = np.dtype('i2')\n    >>> dt.byteorder\n    '='\n    >>> # endian is not relevant for 8 bit numbers\n    >>> np.dtype('i1').byteorder\n    '|'\n    >>> # or ASCII strings\n    >>> np.dtype('S2').byteorder\n    '|'\n    >>> # Even if specific code is given, and it is native\n    >>> # '=' is the byteorder\n    >>> import sys\n    >>> sys_is_le = sys.byteorder == 'little'\n    >>> native_code = sys_is_le and '<' or '>'\n    >>> swapped_code = sys_is_le and '>' or '<'\n    >>> dt = np.dtype(native_code + 'i2')\n    >>> dt.byteorder\n    '='\n    >>> # Swapped code shows up as itself\n    >>> dt = np.dtype(swapped_code + 'i2')\n    >>> dt.byteorder == swapped_code\n    True\n\n    ")
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6107, 46), tuple_22337, str_22339)

# Processing the call keyword arguments (line 6107)
kwargs_22340 = {}
# Getting the type of 'add_newdoc' (line 6107)
add_newdoc_22334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6107, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 6107)
add_newdoc_call_result_22341 = invoke(stypy.reporting.localization.Localization(__file__, 6107, 0), add_newdoc_22334, *[str_22335, str_22336, tuple_22337], **kwargs_22340)


# Call to add_newdoc(...): (line 6150)
# Processing the call arguments (line 6150)
str_22343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6150, 11), 'str', 'numpy.core.multiarray')
str_22344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6150, 36), 'str', 'dtype')

# Obtaining an instance of the builtin type 'tuple' (line 6150)
tuple_22345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6150, 46), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 6150)
# Adding element type (line 6150)
str_22346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6150, 46), 'str', 'char')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6150, 46), tuple_22345, str_22346)
# Adding element type (line 6150)
str_22347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6151, 4), 'str', 'A unique character code for each of the 21 different built-in types.')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6150, 46), tuple_22345, str_22347)

# Processing the call keyword arguments (line 6150)
kwargs_22348 = {}
# Getting the type of 'add_newdoc' (line 6150)
add_newdoc_22342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6150, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 6150)
add_newdoc_call_result_22349 = invoke(stypy.reporting.localization.Localization(__file__, 6150, 0), add_newdoc_22342, *[str_22343, str_22344, tuple_22345], **kwargs_22348)


# Call to add_newdoc(...): (line 6153)
# Processing the call arguments (line 6153)
str_22351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6153, 11), 'str', 'numpy.core.multiarray')
str_22352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6153, 36), 'str', 'dtype')

# Obtaining an instance of the builtin type 'tuple' (line 6153)
tuple_22353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6153, 46), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 6153)
# Adding element type (line 6153)
str_22354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6153, 46), 'str', 'descr')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6153, 46), tuple_22353, str_22354)
# Adding element type (line 6153)
str_22355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6160, (-1)), 'str', "\n    Array-interface compliant full description of the data-type.\n\n    The format is that required by the 'descr' key in the\n    `__array_interface__` attribute.\n\n    ")
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6153, 46), tuple_22353, str_22355)

# Processing the call keyword arguments (line 6153)
kwargs_22356 = {}
# Getting the type of 'add_newdoc' (line 6153)
add_newdoc_22350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6153, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 6153)
add_newdoc_call_result_22357 = invoke(stypy.reporting.localization.Localization(__file__, 6153, 0), add_newdoc_22350, *[str_22351, str_22352, tuple_22353], **kwargs_22356)


# Call to add_newdoc(...): (line 6162)
# Processing the call arguments (line 6162)
str_22359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6162, 11), 'str', 'numpy.core.multiarray')
str_22360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6162, 36), 'str', 'dtype')

# Obtaining an instance of the builtin type 'tuple' (line 6162)
tuple_22361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6162, 46), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 6162)
# Adding element type (line 6162)
str_22362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6162, 46), 'str', 'fields')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6162, 46), tuple_22361, str_22362)
# Adding element type (line 6162)
str_22363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6187, (-1)), 'str', "\n    Dictionary of named fields defined for this data type, or ``None``.\n\n    The dictionary is indexed by keys that are the names of the fields.\n    Each entry in the dictionary is a tuple fully describing the field::\n\n      (dtype, offset[, title])\n\n    If present, the optional title can be any object (if it is a string\n    or unicode then it will also be a key in the fields dictionary,\n    otherwise it's meta-data). Notice also that the first two elements\n    of the tuple can be passed directly as arguments to the ``ndarray.getfield``\n    and ``ndarray.setfield`` methods.\n\n    See Also\n    --------\n    ndarray.getfield, ndarray.setfield\n\n    Examples\n    --------\n    >>> dt = np.dtype([('name', np.str_, 16), ('grades', np.float64, (2,))])\n    >>> print(dt.fields)\n    {'grades': (dtype(('float64',(2,))), 16), 'name': (dtype('|S16'), 0)}\n\n    ")
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6162, 46), tuple_22361, str_22363)

# Processing the call keyword arguments (line 6162)
kwargs_22364 = {}
# Getting the type of 'add_newdoc' (line 6162)
add_newdoc_22358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6162, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 6162)
add_newdoc_call_result_22365 = invoke(stypy.reporting.localization.Localization(__file__, 6162, 0), add_newdoc_22358, *[str_22359, str_22360, tuple_22361], **kwargs_22364)


# Call to add_newdoc(...): (line 6189)
# Processing the call arguments (line 6189)
str_22367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6189, 11), 'str', 'numpy.core.multiarray')
str_22368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6189, 36), 'str', 'dtype')

# Obtaining an instance of the builtin type 'tuple' (line 6189)
tuple_22369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6189, 46), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 6189)
# Adding element type (line 6189)
str_22370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6189, 46), 'str', 'flags')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6189, 46), tuple_22369, str_22370)
# Adding element type (line 6189)
str_22371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6199, (-1)), 'str', '\n    Bit-flags describing how this data type is to be interpreted.\n\n    Bit-masks are in `numpy.core.multiarray` as the constants\n    `ITEM_HASOBJECT`, `LIST_PICKLE`, `ITEM_IS_POINTER`, `NEEDS_INIT`,\n    `NEEDS_PYAPI`, `USE_GETITEM`, `USE_SETITEM`. A full explanation\n    of these flags is in C-API documentation; they are largely useful\n    for user-defined data-types.\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6189, 46), tuple_22369, str_22371)

# Processing the call keyword arguments (line 6189)
kwargs_22372 = {}
# Getting the type of 'add_newdoc' (line 6189)
add_newdoc_22366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6189, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 6189)
add_newdoc_call_result_22373 = invoke(stypy.reporting.localization.Localization(__file__, 6189, 0), add_newdoc_22366, *[str_22367, str_22368, tuple_22369], **kwargs_22372)


# Call to add_newdoc(...): (line 6201)
# Processing the call arguments (line 6201)
str_22375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6201, 11), 'str', 'numpy.core.multiarray')
str_22376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6201, 36), 'str', 'dtype')

# Obtaining an instance of the builtin type 'tuple' (line 6201)
tuple_22377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6201, 46), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 6201)
# Adding element type (line 6201)
str_22378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6201, 46), 'str', 'hasobject')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6201, 46), tuple_22377, str_22378)
# Adding element type (line 6201)
str_22379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6212, (-1)), 'str', "\n    Boolean indicating whether this dtype contains any reference-counted\n    objects in any fields or sub-dtypes.\n\n    Recall that what is actually in the ndarray memory representing\n    the Python object is the memory address of that object (a pointer).\n    Special handling may be required, and this attribute is useful for\n    distinguishing data types that may contain arbitrary Python objects\n    and data-types that won't.\n\n    ")
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6201, 46), tuple_22377, str_22379)

# Processing the call keyword arguments (line 6201)
kwargs_22380 = {}
# Getting the type of 'add_newdoc' (line 6201)
add_newdoc_22374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6201, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 6201)
add_newdoc_call_result_22381 = invoke(stypy.reporting.localization.Localization(__file__, 6201, 0), add_newdoc_22374, *[str_22375, str_22376, tuple_22377], **kwargs_22380)


# Call to add_newdoc(...): (line 6214)
# Processing the call arguments (line 6214)
str_22383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6214, 11), 'str', 'numpy.core.multiarray')
str_22384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6214, 36), 'str', 'dtype')

# Obtaining an instance of the builtin type 'tuple' (line 6214)
tuple_22385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6214, 46), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 6214)
# Adding element type (line 6214)
str_22386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6214, 46), 'str', 'isbuiltin')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6214, 46), tuple_22385, str_22386)
# Adding element type (line 6214)
str_22387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6241, (-1)), 'str', "\n    Integer indicating how this dtype relates to the built-in dtypes.\n\n    Read-only.\n\n    =  ========================================================================\n    0  if this is a structured array type, with fields\n    1  if this is a dtype compiled into numpy (such as ints, floats etc)\n    2  if the dtype is for a user-defined numpy type\n       A user-defined type uses the numpy C-API machinery to extend\n       numpy to handle a new array type. See\n       :ref:`user.user-defined-data-types` in the Numpy manual.\n    =  ========================================================================\n\n    Examples\n    --------\n    >>> dt = np.dtype('i2')\n    >>> dt.isbuiltin\n    1\n    >>> dt = np.dtype('f8')\n    >>> dt.isbuiltin\n    1\n    >>> dt = np.dtype([('field1', 'f8')])\n    >>> dt.isbuiltin\n    0\n\n    ")
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6214, 46), tuple_22385, str_22387)

# Processing the call keyword arguments (line 6214)
kwargs_22388 = {}
# Getting the type of 'add_newdoc' (line 6214)
add_newdoc_22382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6214, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 6214)
add_newdoc_call_result_22389 = invoke(stypy.reporting.localization.Localization(__file__, 6214, 0), add_newdoc_22382, *[str_22383, str_22384, tuple_22385], **kwargs_22388)


# Call to add_newdoc(...): (line 6243)
# Processing the call arguments (line 6243)
str_22391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6243, 11), 'str', 'numpy.core.multiarray')
str_22392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6243, 36), 'str', 'dtype')

# Obtaining an instance of the builtin type 'tuple' (line 6243)
tuple_22393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6243, 46), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 6243)
# Adding element type (line 6243)
str_22394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6243, 46), 'str', 'isnative')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6243, 46), tuple_22393, str_22394)
# Adding element type (line 6243)
str_22395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6248, (-1)), 'str', '\n    Boolean indicating whether the byte order of this dtype is native\n    to the platform.\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6243, 46), tuple_22393, str_22395)

# Processing the call keyword arguments (line 6243)
kwargs_22396 = {}
# Getting the type of 'add_newdoc' (line 6243)
add_newdoc_22390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6243, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 6243)
add_newdoc_call_result_22397 = invoke(stypy.reporting.localization.Localization(__file__, 6243, 0), add_newdoc_22390, *[str_22391, str_22392, tuple_22393], **kwargs_22396)


# Call to add_newdoc(...): (line 6250)
# Processing the call arguments (line 6250)
str_22399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6250, 11), 'str', 'numpy.core.multiarray')
str_22400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6250, 36), 'str', 'dtype')

# Obtaining an instance of the builtin type 'tuple' (line 6250)
tuple_22401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6250, 46), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 6250)
# Adding element type (line 6250)
str_22402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6250, 46), 'str', 'isalignedstruct')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6250, 46), tuple_22401, str_22402)
# Adding element type (line 6250)
str_22403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6256, (-1)), 'str', '\n    Boolean indicating whether the dtype is a struct which maintains\n    field alignment. This flag is sticky, so when combining multiple\n    structs together, it is preserved and produces new dtypes which\n    are also aligned.\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6250, 46), tuple_22401, str_22403)

# Processing the call keyword arguments (line 6250)
kwargs_22404 = {}
# Getting the type of 'add_newdoc' (line 6250)
add_newdoc_22398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6250, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 6250)
add_newdoc_call_result_22405 = invoke(stypy.reporting.localization.Localization(__file__, 6250, 0), add_newdoc_22398, *[str_22399, str_22400, tuple_22401], **kwargs_22404)


# Call to add_newdoc(...): (line 6258)
# Processing the call arguments (line 6258)
str_22407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6258, 11), 'str', 'numpy.core.multiarray')
str_22408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6258, 36), 'str', 'dtype')

# Obtaining an instance of the builtin type 'tuple' (line 6258)
tuple_22409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6258, 46), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 6258)
# Adding element type (line 6258)
str_22410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6258, 46), 'str', 'itemsize')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6258, 46), tuple_22409, str_22410)
# Adding element type (line 6258)
str_22411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6265, (-1)), 'str', '\n    The element size of this data-type object.\n\n    For 18 of the 21 types this number is fixed by the data-type.\n    For the flexible data-types, this number can be anything.\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6258, 46), tuple_22409, str_22411)

# Processing the call keyword arguments (line 6258)
kwargs_22412 = {}
# Getting the type of 'add_newdoc' (line 6258)
add_newdoc_22406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6258, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 6258)
add_newdoc_call_result_22413 = invoke(stypy.reporting.localization.Localization(__file__, 6258, 0), add_newdoc_22406, *[str_22407, str_22408, tuple_22409], **kwargs_22412)


# Call to add_newdoc(...): (line 6267)
# Processing the call arguments (line 6267)
str_22415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6267, 11), 'str', 'numpy.core.multiarray')
str_22416 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6267, 36), 'str', 'dtype')

# Obtaining an instance of the builtin type 'tuple' (line 6267)
tuple_22417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6267, 46), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 6267)
# Adding element type (line 6267)
str_22418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6267, 46), 'str', 'kind')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6267, 46), tuple_22417, str_22418)
# Adding element type (line 6267)
str_22419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6285, (-1)), 'str', "\n    A character code (one of 'biufcmMOSUV') identifying the general kind of data.\n\n    =  ======================\n    b  boolean\n    i  signed integer\n    u  unsigned integer\n    f  floating-point\n    c  complex floating-point\n    m  timedelta\n    M  datetime\n    O  object\n    S  (byte-)string\n    U  Unicode\n    V  void\n    =  ======================\n\n    ")
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6267, 46), tuple_22417, str_22419)

# Processing the call keyword arguments (line 6267)
kwargs_22420 = {}
# Getting the type of 'add_newdoc' (line 6267)
add_newdoc_22414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6267, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 6267)
add_newdoc_call_result_22421 = invoke(stypy.reporting.localization.Localization(__file__, 6267, 0), add_newdoc_22414, *[str_22415, str_22416, tuple_22417], **kwargs_22420)


# Call to add_newdoc(...): (line 6287)
# Processing the call arguments (line 6287)
str_22423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6287, 11), 'str', 'numpy.core.multiarray')
str_22424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6287, 36), 'str', 'dtype')

# Obtaining an instance of the builtin type 'tuple' (line 6287)
tuple_22425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6287, 46), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 6287)
# Adding element type (line 6287)
str_22426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6287, 46), 'str', 'name')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6287, 46), tuple_22425, str_22426)
# Adding element type (line 6287)
str_22427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6293, (-1)), 'str', '\n    A bit-width name for this data-type.\n\n    Un-sized flexible data-type objects do not have this attribute.\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6287, 46), tuple_22425, str_22427)

# Processing the call keyword arguments (line 6287)
kwargs_22428 = {}
# Getting the type of 'add_newdoc' (line 6287)
add_newdoc_22422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6287, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 6287)
add_newdoc_call_result_22429 = invoke(stypy.reporting.localization.Localization(__file__, 6287, 0), add_newdoc_22422, *[str_22423, str_22424, tuple_22425], **kwargs_22428)


# Call to add_newdoc(...): (line 6295)
# Processing the call arguments (line 6295)
str_22431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6295, 11), 'str', 'numpy.core.multiarray')
str_22432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6295, 36), 'str', 'dtype')

# Obtaining an instance of the builtin type 'tuple' (line 6295)
tuple_22433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6295, 46), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 6295)
# Adding element type (line 6295)
str_22434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6295, 46), 'str', 'names')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6295, 46), tuple_22433, str_22434)
# Adding element type (line 6295)
str_22435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6308, (-1)), 'str', "\n    Ordered list of field names, or ``None`` if there are no fields.\n\n    The names are ordered according to increasing byte offset. This can be\n    used, for example, to walk through all of the named fields in offset order.\n\n    Examples\n    --------\n    >>> dt = np.dtype([('name', np.str_, 16), ('grades', np.float64, (2,))])\n    >>> dt.names\n    ('name', 'grades')\n\n    ")
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6295, 46), tuple_22433, str_22435)

# Processing the call keyword arguments (line 6295)
kwargs_22436 = {}
# Getting the type of 'add_newdoc' (line 6295)
add_newdoc_22430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6295, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 6295)
add_newdoc_call_result_22437 = invoke(stypy.reporting.localization.Localization(__file__, 6295, 0), add_newdoc_22430, *[str_22431, str_22432, tuple_22433], **kwargs_22436)


# Call to add_newdoc(...): (line 6310)
# Processing the call arguments (line 6310)
str_22439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6310, 11), 'str', 'numpy.core.multiarray')
str_22440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6310, 36), 'str', 'dtype')

# Obtaining an instance of the builtin type 'tuple' (line 6310)
tuple_22441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6310, 46), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 6310)
# Adding element type (line 6310)
str_22442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6310, 46), 'str', 'num')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6310, 46), tuple_22441, str_22442)
# Adding element type (line 6310)
str_22443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6316, (-1)), 'str', '\n    A unique number for each of the 21 different built-in types.\n\n    These are roughly ordered from least-to-most precision.\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6310, 46), tuple_22441, str_22443)

# Processing the call keyword arguments (line 6310)
kwargs_22444 = {}
# Getting the type of 'add_newdoc' (line 6310)
add_newdoc_22438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6310, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 6310)
add_newdoc_call_result_22445 = invoke(stypy.reporting.localization.Localization(__file__, 6310, 0), add_newdoc_22438, *[str_22439, str_22440, tuple_22441], **kwargs_22444)


# Call to add_newdoc(...): (line 6318)
# Processing the call arguments (line 6318)
str_22447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6318, 11), 'str', 'numpy.core.multiarray')
str_22448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6318, 36), 'str', 'dtype')

# Obtaining an instance of the builtin type 'tuple' (line 6318)
tuple_22449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6318, 46), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 6318)
# Adding element type (line 6318)
str_22450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6318, 46), 'str', 'shape')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6318, 46), tuple_22449, str_22450)
# Adding element type (line 6318)
str_22451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6323, (-1)), 'str', '\n    Shape tuple of the sub-array if this data type describes a sub-array,\n    and ``()`` otherwise.\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6318, 46), tuple_22449, str_22451)

# Processing the call keyword arguments (line 6318)
kwargs_22452 = {}
# Getting the type of 'add_newdoc' (line 6318)
add_newdoc_22446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6318, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 6318)
add_newdoc_call_result_22453 = invoke(stypy.reporting.localization.Localization(__file__, 6318, 0), add_newdoc_22446, *[str_22447, str_22448, tuple_22449], **kwargs_22452)


# Call to add_newdoc(...): (line 6325)
# Processing the call arguments (line 6325)
str_22455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6325, 11), 'str', 'numpy.core.multiarray')
str_22456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6325, 36), 'str', 'dtype')

# Obtaining an instance of the builtin type 'tuple' (line 6325)
tuple_22457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6325, 46), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 6325)
# Adding element type (line 6325)
str_22458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6325, 46), 'str', 'str')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6325, 46), tuple_22457, str_22458)
# Adding element type (line 6325)
str_22459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6326, 4), 'str', 'The array-protocol typestring of this data-type object.')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6325, 46), tuple_22457, str_22459)

# Processing the call keyword arguments (line 6325)
kwargs_22460 = {}
# Getting the type of 'add_newdoc' (line 6325)
add_newdoc_22454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6325, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 6325)
add_newdoc_call_result_22461 = invoke(stypy.reporting.localization.Localization(__file__, 6325, 0), add_newdoc_22454, *[str_22455, str_22456, tuple_22457], **kwargs_22460)


# Call to add_newdoc(...): (line 6328)
# Processing the call arguments (line 6328)
str_22463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6328, 11), 'str', 'numpy.core.multiarray')
str_22464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6328, 36), 'str', 'dtype')

# Obtaining an instance of the builtin type 'tuple' (line 6328)
tuple_22465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6328, 46), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 6328)
# Adding element type (line 6328)
str_22466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6328, 46), 'str', 'subdtype')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6328, 46), tuple_22465, str_22466)
# Adding element type (line 6328)
str_22467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6340, (-1)), 'str', '\n    Tuple ``(item_dtype, shape)`` if this `dtype` describes a sub-array, and\n    None otherwise.\n\n    The *shape* is the fixed shape of the sub-array described by this\n    data type, and *item_dtype* the data type of the array.\n\n    If a field whose dtype object has this attribute is retrieved,\n    then the extra dimensions implied by *shape* are tacked on to\n    the end of the retrieved array.\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6328, 46), tuple_22465, str_22467)

# Processing the call keyword arguments (line 6328)
kwargs_22468 = {}
# Getting the type of 'add_newdoc' (line 6328)
add_newdoc_22462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6328, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 6328)
add_newdoc_call_result_22469 = invoke(stypy.reporting.localization.Localization(__file__, 6328, 0), add_newdoc_22462, *[str_22463, str_22464, tuple_22465], **kwargs_22468)


# Call to add_newdoc(...): (line 6342)
# Processing the call arguments (line 6342)
str_22471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6342, 11), 'str', 'numpy.core.multiarray')
str_22472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6342, 36), 'str', 'dtype')

# Obtaining an instance of the builtin type 'tuple' (line 6342)
tuple_22473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6342, 46), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 6342)
# Adding element type (line 6342)
str_22474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6342, 46), 'str', 'type')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6342, 46), tuple_22473, str_22474)
# Adding element type (line 6342)
str_22475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6343, 4), 'str', 'The type object used to instantiate a scalar of this data-type.')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6342, 46), tuple_22473, str_22475)

# Processing the call keyword arguments (line 6342)
kwargs_22476 = {}
# Getting the type of 'add_newdoc' (line 6342)
add_newdoc_22470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6342, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 6342)
add_newdoc_call_result_22477 = invoke(stypy.reporting.localization.Localization(__file__, 6342, 0), add_newdoc_22470, *[str_22471, str_22472, tuple_22473], **kwargs_22476)


# Call to add_newdoc(...): (line 6351)
# Processing the call arguments (line 6351)
str_22479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6351, 11), 'str', 'numpy.core.multiarray')
str_22480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6351, 36), 'str', 'dtype')

# Obtaining an instance of the builtin type 'tuple' (line 6351)
tuple_22481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6351, 46), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 6351)
# Adding element type (line 6351)
str_22482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6351, 46), 'str', 'newbyteorder')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6351, 46), tuple_22481, str_22482)
# Adding element type (line 6351)
str_22483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6414, (-1)), 'str', "\n    newbyteorder(new_order='S')\n\n    Return a new dtype with a different byte order.\n\n    Changes are also made in all fields and sub-arrays of the data type.\n\n    Parameters\n    ----------\n    new_order : string, optional\n        Byte order to force; a value from the byte order specifications\n        below.  The default value ('S') results in swapping the current\n        byte order.  `new_order` codes can be any of:\n\n        * 'S' - swap dtype from current to opposite endian\n        * {'<', 'L'} - little endian\n        * {'>', 'B'} - big endian\n        * {'=', 'N'} - native order\n        * {'|', 'I'} - ignore (no change to byte order)\n\n        The code does a case-insensitive check on the first letter of\n        `new_order` for these alternatives.  For example, any of '>'\n        or 'B' or 'b' or 'brian' are valid to specify big-endian.\n\n    Returns\n    -------\n    new_dtype : dtype\n        New dtype object with the given change to the byte order.\n\n    Notes\n    -----\n    Changes are also made in all fields and sub-arrays of the data type.\n\n    Examples\n    --------\n    >>> import sys\n    >>> sys_is_le = sys.byteorder == 'little'\n    >>> native_code = sys_is_le and '<' or '>'\n    >>> swapped_code = sys_is_le and '>' or '<'\n    >>> native_dt = np.dtype(native_code+'i2')\n    >>> swapped_dt = np.dtype(swapped_code+'i2')\n    >>> native_dt.newbyteorder('S') == swapped_dt\n    True\n    >>> native_dt.newbyteorder() == swapped_dt\n    True\n    >>> native_dt == swapped_dt.newbyteorder('S')\n    True\n    >>> native_dt == swapped_dt.newbyteorder('=')\n    True\n    >>> native_dt == swapped_dt.newbyteorder('N')\n    True\n    >>> native_dt == native_dt.newbyteorder('|')\n    True\n    >>> np.dtype('<i2') == native_dt.newbyteorder('<')\n    True\n    >>> np.dtype('<i2') == native_dt.newbyteorder('L')\n    True\n    >>> np.dtype('>i2') == native_dt.newbyteorder('>')\n    True\n    >>> np.dtype('>i2') == native_dt.newbyteorder('B')\n    True\n\n    ")
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6351, 46), tuple_22481, str_22483)

# Processing the call keyword arguments (line 6351)
kwargs_22484 = {}
# Getting the type of 'add_newdoc' (line 6351)
add_newdoc_22478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6351, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 6351)
add_newdoc_call_result_22485 = invoke(stypy.reporting.localization.Localization(__file__, 6351, 0), add_newdoc_22478, *[str_22479, str_22480, tuple_22481], **kwargs_22484)


# Call to add_newdoc(...): (line 6423)
# Processing the call arguments (line 6423)
str_22487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6423, 11), 'str', 'numpy.core.multiarray')
str_22488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6423, 36), 'str', 'busdaycalendar')
str_22489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6485, (-1)), 'str', '\n    busdaycalendar(weekmask=\'1111100\', holidays=None)\n\n    A business day calendar object that efficiently stores information\n    defining valid days for the busday family of functions.\n\n    The default valid days are Monday through Friday ("business days").\n    A busdaycalendar object can be specified with any set of weekly\n    valid days, plus an optional "holiday" dates that always will be invalid.\n\n    Once a busdaycalendar object is created, the weekmask and holidays\n    cannot be modified.\n\n    .. versionadded:: 1.7.0\n\n    Parameters\n    ----------\n    weekmask : str or array_like of bool, optional\n        A seven-element array indicating which of Monday through Sunday are\n        valid days. May be specified as a length-seven list or array, like\n        [1,1,1,1,1,0,0]; a length-seven string, like \'1111100\'; or a string\n        like "Mon Tue Wed Thu Fri", made up of 3-character abbreviations for\n        weekdays, optionally separated by white space. Valid abbreviations\n        are: Mon Tue Wed Thu Fri Sat Sun\n    holidays : array_like of datetime64[D], optional\n        An array of dates to consider as invalid dates, no matter which\n        weekday they fall upon.  Holiday dates may be specified in any\n        order, and NaT (not-a-time) dates are ignored.  This list is\n        saved in a normalized form that is suited for fast calculations\n        of valid days.\n\n    Returns\n    -------\n    out : busdaycalendar\n        A business day calendar object containing the specified\n        weekmask and holidays values.\n\n    See Also\n    --------\n    is_busday : Returns a boolean array indicating valid days.\n    busday_offset : Applies an offset counted in valid days.\n    busday_count : Counts how many valid days are in a half-open date range.\n\n    Attributes\n    ----------\n    Note: once a busdaycalendar object is created, you cannot modify the\n    weekmask or holidays.  The attributes return copies of internal data.\n    weekmask : (copy) seven-element array of bool\n    holidays : (copy) sorted array of datetime64[D]\n\n    Examples\n    --------\n    >>> # Some important days in July\n    ... bdd = np.busdaycalendar(\n    ...             holidays=[\'2011-07-01\', \'2011-07-04\', \'2011-07-17\'])\n    >>> # Default is Monday to Friday weekdays\n    ... bdd.weekmask\n    array([ True,  True,  True,  True,  True, False, False], dtype=\'bool\')\n    >>> # Any holidays already on the weekend are removed\n    ... bdd.holidays\n    array([\'2011-07-01\', \'2011-07-04\'], dtype=\'datetime64[D]\')\n    ')
# Processing the call keyword arguments (line 6423)
kwargs_22490 = {}
# Getting the type of 'add_newdoc' (line 6423)
add_newdoc_22486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6423, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 6423)
add_newdoc_call_result_22491 = invoke(stypy.reporting.localization.Localization(__file__, 6423, 0), add_newdoc_22486, *[str_22487, str_22488, str_22489], **kwargs_22490)


# Call to add_newdoc(...): (line 6487)
# Processing the call arguments (line 6487)
str_22493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6487, 11), 'str', 'numpy.core.multiarray')
str_22494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6487, 36), 'str', 'busdaycalendar')

# Obtaining an instance of the builtin type 'tuple' (line 6487)
tuple_22495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6487, 55), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 6487)
# Adding element type (line 6487)
str_22496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6487, 55), 'str', 'weekmask')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6487, 55), tuple_22495, str_22496)
# Adding element type (line 6487)
str_22497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6488, 4), 'str', 'A copy of the seven-element boolean mask indicating valid days.')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6487, 55), tuple_22495, str_22497)

# Processing the call keyword arguments (line 6487)
kwargs_22498 = {}
# Getting the type of 'add_newdoc' (line 6487)
add_newdoc_22492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6487, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 6487)
add_newdoc_call_result_22499 = invoke(stypy.reporting.localization.Localization(__file__, 6487, 0), add_newdoc_22492, *[str_22493, str_22494, tuple_22495], **kwargs_22498)


# Call to add_newdoc(...): (line 6490)
# Processing the call arguments (line 6490)
str_22501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6490, 11), 'str', 'numpy.core.multiarray')
str_22502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6490, 36), 'str', 'busdaycalendar')

# Obtaining an instance of the builtin type 'tuple' (line 6490)
tuple_22503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6490, 55), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 6490)
# Adding element type (line 6490)
str_22504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6490, 55), 'str', 'holidays')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6490, 55), tuple_22503, str_22504)
# Adding element type (line 6490)
str_22505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6491, 4), 'str', 'A copy of the holiday array indicating additional invalid days.')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6490, 55), tuple_22503, str_22505)

# Processing the call keyword arguments (line 6490)
kwargs_22506 = {}
# Getting the type of 'add_newdoc' (line 6490)
add_newdoc_22500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6490, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 6490)
add_newdoc_call_result_22507 = invoke(stypy.reporting.localization.Localization(__file__, 6490, 0), add_newdoc_22500, *[str_22501, str_22502, tuple_22503], **kwargs_22506)


# Call to add_newdoc(...): (line 6493)
# Processing the call arguments (line 6493)
str_22509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6493, 11), 'str', 'numpy.core.multiarray')
str_22510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6493, 36), 'str', 'is_busday')
str_22511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6542, (-1)), 'str', '\n    is_busday(dates, weekmask=\'1111100\', holidays=None, busdaycal=None, out=None)\n\n    Calculates which of the given dates are valid days, and which are not.\n\n    .. versionadded:: 1.7.0\n\n    Parameters\n    ----------\n    dates : array_like of datetime64[D]\n        The array of dates to process.\n    weekmask : str or array_like of bool, optional\n        A seven-element array indicating which of Monday through Sunday are\n        valid days. May be specified as a length-seven list or array, like\n        [1,1,1,1,1,0,0]; a length-seven string, like \'1111100\'; or a string\n        like "Mon Tue Wed Thu Fri", made up of 3-character abbreviations for\n        weekdays, optionally separated by white space. Valid abbreviations\n        are: Mon Tue Wed Thu Fri Sat Sun\n    holidays : array_like of datetime64[D], optional\n        An array of dates to consider as invalid dates.  They may be\n        specified in any order, and NaT (not-a-time) dates are ignored.\n        This list is saved in a normalized form that is suited for\n        fast calculations of valid days.\n    busdaycal : busdaycalendar, optional\n        A `busdaycalendar` object which specifies the valid days. If this\n        parameter is provided, neither weekmask nor holidays may be\n        provided.\n    out : array of bool, optional\n        If provided, this array is filled with the result.\n\n    Returns\n    -------\n    out : array of bool\n        An array with the same shape as ``dates``, containing True for\n        each valid day, and False for each invalid day.\n\n    See Also\n    --------\n    busdaycalendar: An object that specifies a custom set of valid days.\n    busday_offset : Applies an offset counted in valid days.\n    busday_count : Counts how many valid days are in a half-open date range.\n\n    Examples\n    --------\n    >>> # The weekdays are Friday, Saturday, and Monday\n    ... np.is_busday([\'2011-07-01\', \'2011-07-02\', \'2011-07-18\'],\n    ...                 holidays=[\'2011-07-01\', \'2011-07-04\', \'2011-07-17\'])\n    array([False, False,  True], dtype=\'bool\')\n    ')
# Processing the call keyword arguments (line 6493)
kwargs_22512 = {}
# Getting the type of 'add_newdoc' (line 6493)
add_newdoc_22508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6493, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 6493)
add_newdoc_call_result_22513 = invoke(stypy.reporting.localization.Localization(__file__, 6493, 0), add_newdoc_22508, *[str_22509, str_22510, str_22511], **kwargs_22512)


# Call to add_newdoc(...): (line 6544)
# Processing the call arguments (line 6544)
str_22515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6544, 11), 'str', 'numpy.core.multiarray')
str_22516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6544, 36), 'str', 'busday_offset')
str_22517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6632, (-1)), 'str', '\n    busday_offset(dates, offsets, roll=\'raise\', weekmask=\'1111100\', holidays=None, busdaycal=None, out=None)\n\n    First adjusts the date to fall on a valid day according to\n    the ``roll`` rule, then applies offsets to the given dates\n    counted in valid days.\n\n    .. versionadded:: 1.7.0\n\n    Parameters\n    ----------\n    dates : array_like of datetime64[D]\n        The array of dates to process.\n    offsets : array_like of int\n        The array of offsets, which is broadcast with ``dates``.\n    roll : {\'raise\', \'nat\', \'forward\', \'following\', \'backward\', \'preceding\', \'modifiedfollowing\', \'modifiedpreceding\'}, optional\n        How to treat dates that do not fall on a valid day. The default\n        is \'raise\'.\n\n          * \'raise\' means to raise an exception for an invalid day.\n          * \'nat\' means to return a NaT (not-a-time) for an invalid day.\n          * \'forward\' and \'following\' mean to take the first valid day\n            later in time.\n          * \'backward\' and \'preceding\' mean to take the first valid day\n            earlier in time.\n          * \'modifiedfollowing\' means to take the first valid day\n            later in time unless it is across a Month boundary, in which\n            case to take the first valid day earlier in time.\n          * \'modifiedpreceding\' means to take the first valid day\n            earlier in time unless it is across a Month boundary, in which\n            case to take the first valid day later in time.\n    weekmask : str or array_like of bool, optional\n        A seven-element array indicating which of Monday through Sunday are\n        valid days. May be specified as a length-seven list or array, like\n        [1,1,1,1,1,0,0]; a length-seven string, like \'1111100\'; or a string\n        like "Mon Tue Wed Thu Fri", made up of 3-character abbreviations for\n        weekdays, optionally separated by white space. Valid abbreviations\n        are: Mon Tue Wed Thu Fri Sat Sun\n    holidays : array_like of datetime64[D], optional\n        An array of dates to consider as invalid dates.  They may be\n        specified in any order, and NaT (not-a-time) dates are ignored.\n        This list is saved in a normalized form that is suited for\n        fast calculations of valid days.\n    busdaycal : busdaycalendar, optional\n        A `busdaycalendar` object which specifies the valid days. If this\n        parameter is provided, neither weekmask nor holidays may be\n        provided.\n    out : array of datetime64[D], optional\n        If provided, this array is filled with the result.\n\n    Returns\n    -------\n    out : array of datetime64[D]\n        An array with a shape from broadcasting ``dates`` and ``offsets``\n        together, containing the dates with offsets applied.\n\n    See Also\n    --------\n    busdaycalendar: An object that specifies a custom set of valid days.\n    is_busday : Returns a boolean array indicating valid days.\n    busday_count : Counts how many valid days are in a half-open date range.\n\n    Examples\n    --------\n    >>> # First business day in October 2011 (not accounting for holidays)\n    ... np.busday_offset(\'2011-10\', 0, roll=\'forward\')\n    numpy.datetime64(\'2011-10-03\',\'D\')\n    >>> # Last business day in February 2012 (not accounting for holidays)\n    ... np.busday_offset(\'2012-03\', -1, roll=\'forward\')\n    numpy.datetime64(\'2012-02-29\',\'D\')\n    >>> # Third Wednesday in January 2011\n    ... np.busday_offset(\'2011-01\', 2, roll=\'forward\', weekmask=\'Wed\')\n    numpy.datetime64(\'2011-01-19\',\'D\')\n    >>> # 2012 Mother\'s Day in Canada and the U.S.\n    ... np.busday_offset(\'2012-05\', 1, roll=\'forward\', weekmask=\'Sun\')\n    numpy.datetime64(\'2012-05-13\',\'D\')\n\n    >>> # First business day on or after a date\n    ... np.busday_offset(\'2011-03-20\', 0, roll=\'forward\')\n    numpy.datetime64(\'2011-03-21\',\'D\')\n    >>> np.busday_offset(\'2011-03-22\', 0, roll=\'forward\')\n    numpy.datetime64(\'2011-03-22\',\'D\')\n    >>> # First business day after a date\n    ... np.busday_offset(\'2011-03-20\', 1, roll=\'backward\')\n    numpy.datetime64(\'2011-03-21\',\'D\')\n    >>> np.busday_offset(\'2011-03-22\', 1, roll=\'backward\')\n    numpy.datetime64(\'2011-03-23\',\'D\')\n    ')
# Processing the call keyword arguments (line 6544)
kwargs_22518 = {}
# Getting the type of 'add_newdoc' (line 6544)
add_newdoc_22514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6544, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 6544)
add_newdoc_call_result_22519 = invoke(stypy.reporting.localization.Localization(__file__, 6544, 0), add_newdoc_22514, *[str_22515, str_22516, str_22517], **kwargs_22518)


# Call to add_newdoc(...): (line 6634)
# Processing the call arguments (line 6634)
str_22521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6634, 11), 'str', 'numpy.core.multiarray')
str_22522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6634, 36), 'str', 'busday_count')
str_22523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6696, (-1)), 'str', '\n    busday_count(begindates, enddates, weekmask=\'1111100\', holidays=[], busdaycal=None, out=None)\n\n    Counts the number of valid days between `begindates` and\n    `enddates`, not including the day of `enddates`.\n\n    If ``enddates`` specifies a date value that is earlier than the\n    corresponding ``begindates`` date value, the count will be negative.\n\n    .. versionadded:: 1.7.0\n\n    Parameters\n    ----------\n    begindates : array_like of datetime64[D]\n        The array of the first dates for counting.\n    enddates : array_like of datetime64[D]\n        The array of the end dates for counting, which are excluded\n        from the count themselves.\n    weekmask : str or array_like of bool, optional\n        A seven-element array indicating which of Monday through Sunday are\n        valid days. May be specified as a length-seven list or array, like\n        [1,1,1,1,1,0,0]; a length-seven string, like \'1111100\'; or a string\n        like "Mon Tue Wed Thu Fri", made up of 3-character abbreviations for\n        weekdays, optionally separated by white space. Valid abbreviations\n        are: Mon Tue Wed Thu Fri Sat Sun\n    holidays : array_like of datetime64[D], optional\n        An array of dates to consider as invalid dates.  They may be\n        specified in any order, and NaT (not-a-time) dates are ignored.\n        This list is saved in a normalized form that is suited for\n        fast calculations of valid days.\n    busdaycal : busdaycalendar, optional\n        A `busdaycalendar` object which specifies the valid days. If this\n        parameter is provided, neither weekmask nor holidays may be\n        provided.\n    out : array of int, optional\n        If provided, this array is filled with the result.\n\n    Returns\n    -------\n    out : array of int\n        An array with a shape from broadcasting ``begindates`` and ``enddates``\n        together, containing the number of valid days between\n        the begin and end dates.\n\n    See Also\n    --------\n    busdaycalendar: An object that specifies a custom set of valid days.\n    is_busday : Returns a boolean array indicating valid days.\n    busday_offset : Applies an offset counted in valid days.\n\n    Examples\n    --------\n    >>> # Number of weekdays in January 2011\n    ... np.busday_count(\'2011-01\', \'2011-02\')\n    21\n    >>> # Number of weekdays in 2011\n    ...  np.busday_count(\'2011\', \'2012\')\n    260\n    >>> # Number of Saturdays in 2011\n    ... np.busday_count(\'2011\', \'2012\', weekmask=\'Sat\')\n    53\n    ')
# Processing the call keyword arguments (line 6634)
kwargs_22524 = {}
# Getting the type of 'add_newdoc' (line 6634)
add_newdoc_22520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6634, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 6634)
add_newdoc_call_result_22525 = invoke(stypy.reporting.localization.Localization(__file__, 6634, 0), add_newdoc_22520, *[str_22521, str_22522, str_22523], **kwargs_22524)


# Call to add_newdoc(...): (line 6704)
# Processing the call arguments (line 6704)
str_22527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6704, 11), 'str', 'numpy.lib.index_tricks')
str_22528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6704, 37), 'str', 'mgrid')
str_22529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6745, (-1)), 'str', '\n    `nd_grid` instance which returns a dense multi-dimensional "meshgrid".\n\n    An instance of `numpy.lib.index_tricks.nd_grid` which returns an dense\n    (or fleshed out) mesh-grid when indexed, so that each returned argument\n    has the same shape.  The dimensions and number of the output arrays are\n    equal to the number of indexing dimensions.  If the step length is not a\n    complex number, then the stop is not inclusive.\n\n    However, if the step length is a **complex number** (e.g. 5j), then\n    the integer part of its magnitude is interpreted as specifying the\n    number of points to create between the start and stop values, where\n    the stop value **is inclusive**.\n\n    Returns\n    ----------\n    mesh-grid `ndarrays` all of the same dimensions\n\n    See Also\n    --------\n    numpy.lib.index_tricks.nd_grid : class of `ogrid` and `mgrid` objects\n    ogrid : like mgrid but returns open (not fleshed out) mesh grids\n    r_ : array concatenator\n\n    Examples\n    --------\n    >>> np.mgrid[0:5,0:5]\n    array([[[0, 0, 0, 0, 0],\n            [1, 1, 1, 1, 1],\n            [2, 2, 2, 2, 2],\n            [3, 3, 3, 3, 3],\n            [4, 4, 4, 4, 4]],\n           [[0, 1, 2, 3, 4],\n            [0, 1, 2, 3, 4],\n            [0, 1, 2, 3, 4],\n            [0, 1, 2, 3, 4],\n            [0, 1, 2, 3, 4]]])\n    >>> np.mgrid[-1:1:5j]\n    array([-1. , -0.5,  0. ,  0.5,  1. ])\n\n    ')
# Processing the call keyword arguments (line 6704)
kwargs_22530 = {}
# Getting the type of 'add_newdoc' (line 6704)
add_newdoc_22526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6704, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 6704)
add_newdoc_call_result_22531 = invoke(stypy.reporting.localization.Localization(__file__, 6704, 0), add_newdoc_22526, *[str_22527, str_22528, str_22529], **kwargs_22530)


# Call to add_newdoc(...): (line 6747)
# Processing the call arguments (line 6747)
str_22533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6747, 11), 'str', 'numpy.lib.index_tricks')
str_22534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6747, 37), 'str', 'ogrid')
str_22535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6784, (-1)), 'str', '\n    `nd_grid` instance which returns an open multi-dimensional "meshgrid".\n\n    An instance of `numpy.lib.index_tricks.nd_grid` which returns an open\n    (i.e. not fleshed out) mesh-grid when indexed, so that only one dimension\n    of each returned array is greater than 1.  The dimension and number of the\n    output arrays are equal to the number of indexing dimensions.  If the step\n    length is not a complex number, then the stop is not inclusive.\n\n    However, if the step length is a **complex number** (e.g. 5j), then\n    the integer part of its magnitude is interpreted as specifying the\n    number of points to create between the start and stop values, where\n    the stop value **is inclusive**.\n\n    Returns\n    ----------\n    mesh-grid `ndarrays` with only one dimension :math:`\\neq 1`\n\n    See Also\n    --------\n    np.lib.index_tricks.nd_grid : class of `ogrid` and `mgrid` objects\n    mgrid : like `ogrid` but returns dense (or fleshed out) mesh grids\n    r_ : array concatenator\n\n    Examples\n    --------\n    >>> from numpy import ogrid\n    >>> ogrid[-1:1:5j]\n    array([-1. , -0.5,  0. ,  0.5,  1. ])\n    >>> ogrid[0:5,0:5]\n    [array([[0],\n            [1],\n            [2],\n            [3],\n            [4]]), array([[0, 1, 2, 3, 4]])]\n\n    ')
# Processing the call keyword arguments (line 6747)
kwargs_22536 = {}
# Getting the type of 'add_newdoc' (line 6747)
add_newdoc_22532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6747, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 6747)
add_newdoc_call_result_22537 = invoke(stypy.reporting.localization.Localization(__file__, 6747, 0), add_newdoc_22532, *[str_22533, str_22534, str_22535], **kwargs_22536)


# Call to add_newdoc(...): (line 6793)
# Processing the call arguments (line 6793)
str_22539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6793, 11), 'str', 'numpy.core.numerictypes')
str_22540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6793, 38), 'str', 'generic')
str_22541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6803, (-1)), 'str', '\n    Base class for numpy scalar types.\n\n    Class from which most (all?) numpy scalar types are derived.  For\n    consistency, exposes the same API as `ndarray`, despite many\n    consequent attributes being either "get-only," or completely irrelevant.\n    This is the class from which it is strongly suggested users should derive\n    custom scalar types.\n\n    ')
# Processing the call keyword arguments (line 6793)
kwargs_22542 = {}
# Getting the type of 'add_newdoc' (line 6793)
add_newdoc_22538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6793, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 6793)
add_newdoc_call_result_22543 = invoke(stypy.reporting.localization.Localization(__file__, 6793, 0), add_newdoc_22538, *[str_22539, str_22540, str_22541], **kwargs_22542)


# Call to add_newdoc(...): (line 6807)
# Processing the call arguments (line 6807)
str_22545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6807, 11), 'str', 'numpy.core.numerictypes')
str_22546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6807, 38), 'str', 'generic')

# Obtaining an instance of the builtin type 'tuple' (line 6807)
tuple_22547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6807, 50), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 6807)
# Adding element type (line 6807)
str_22548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6807, 50), 'str', 'T')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6807, 50), tuple_22547, str_22548)
# Adding element type (line 6807)
str_22549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6819, (-1)), 'str', '\n    Not implemented (virtual attribute)\n\n    Class generic exists solely to derive numpy scalars from, and possesses,\n    albeit unimplemented, all the attributes of the ndarray class so as to\n    provide a uniform API.\n\n    See Also\n    --------\n    The corresponding attribute of the derived class of interest.\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6807, 50), tuple_22547, str_22549)

# Processing the call keyword arguments (line 6807)
kwargs_22550 = {}
# Getting the type of 'add_newdoc' (line 6807)
add_newdoc_22544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6807, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 6807)
add_newdoc_call_result_22551 = invoke(stypy.reporting.localization.Localization(__file__, 6807, 0), add_newdoc_22544, *[str_22545, str_22546, tuple_22547], **kwargs_22550)


# Call to add_newdoc(...): (line 6821)
# Processing the call arguments (line 6821)
str_22553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6821, 11), 'str', 'numpy.core.numerictypes')
str_22554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6821, 38), 'str', 'generic')

# Obtaining an instance of the builtin type 'tuple' (line 6821)
tuple_22555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6821, 50), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 6821)
# Adding element type (line 6821)
str_22556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6821, 50), 'str', 'base')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6821, 50), tuple_22555, str_22556)
# Adding element type (line 6821)
str_22557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6833, (-1)), 'str', '\n    Not implemented (virtual attribute)\n\n    Class generic exists solely to derive numpy scalars from, and possesses,\n    albeit unimplemented, all the attributes of the ndarray class so as to\n    a uniform API.\n\n    See Also\n    --------\n    The corresponding attribute of the derived class of interest.\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6821, 50), tuple_22555, str_22557)

# Processing the call keyword arguments (line 6821)
kwargs_22558 = {}
# Getting the type of 'add_newdoc' (line 6821)
add_newdoc_22552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6821, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 6821)
add_newdoc_call_result_22559 = invoke(stypy.reporting.localization.Localization(__file__, 6821, 0), add_newdoc_22552, *[str_22553, str_22554, tuple_22555], **kwargs_22558)


# Call to add_newdoc(...): (line 6835)
# Processing the call arguments (line 6835)
str_22561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6835, 11), 'str', 'numpy.core.numerictypes')
str_22562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6835, 38), 'str', 'generic')

# Obtaining an instance of the builtin type 'tuple' (line 6835)
tuple_22563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6835, 50), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 6835)
# Adding element type (line 6835)
str_22564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6835, 50), 'str', 'data')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6835, 50), tuple_22563, str_22564)
# Adding element type (line 6835)
str_22565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6836, 4), 'str', 'Pointer to start of data.')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6835, 50), tuple_22563, str_22565)

# Processing the call keyword arguments (line 6835)
kwargs_22566 = {}
# Getting the type of 'add_newdoc' (line 6835)
add_newdoc_22560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6835, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 6835)
add_newdoc_call_result_22567 = invoke(stypy.reporting.localization.Localization(__file__, 6835, 0), add_newdoc_22560, *[str_22561, str_22562, tuple_22563], **kwargs_22566)


# Call to add_newdoc(...): (line 6838)
# Processing the call arguments (line 6838)
str_22569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6838, 11), 'str', 'numpy.core.numerictypes')
str_22570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6838, 38), 'str', 'generic')

# Obtaining an instance of the builtin type 'tuple' (line 6838)
tuple_22571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6838, 50), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 6838)
# Adding element type (line 6838)
str_22572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6838, 50), 'str', 'dtype')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6838, 50), tuple_22571, str_22572)
# Adding element type (line 6838)
str_22573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6839, 4), 'str', 'Get array data-descriptor.')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6838, 50), tuple_22571, str_22573)

# Processing the call keyword arguments (line 6838)
kwargs_22574 = {}
# Getting the type of 'add_newdoc' (line 6838)
add_newdoc_22568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6838, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 6838)
add_newdoc_call_result_22575 = invoke(stypy.reporting.localization.Localization(__file__, 6838, 0), add_newdoc_22568, *[str_22569, str_22570, tuple_22571], **kwargs_22574)


# Call to add_newdoc(...): (line 6841)
# Processing the call arguments (line 6841)
str_22577 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6841, 11), 'str', 'numpy.core.numerictypes')
str_22578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6841, 38), 'str', 'generic')

# Obtaining an instance of the builtin type 'tuple' (line 6841)
tuple_22579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6841, 50), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 6841)
# Adding element type (line 6841)
str_22580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6841, 50), 'str', 'flags')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6841, 50), tuple_22579, str_22580)
# Adding element type (line 6841)
str_22581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6842, 4), 'str', 'The integer value of flags.')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6841, 50), tuple_22579, str_22581)

# Processing the call keyword arguments (line 6841)
kwargs_22582 = {}
# Getting the type of 'add_newdoc' (line 6841)
add_newdoc_22576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6841, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 6841)
add_newdoc_call_result_22583 = invoke(stypy.reporting.localization.Localization(__file__, 6841, 0), add_newdoc_22576, *[str_22577, str_22578, tuple_22579], **kwargs_22582)


# Call to add_newdoc(...): (line 6844)
# Processing the call arguments (line 6844)
str_22585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6844, 11), 'str', 'numpy.core.numerictypes')
str_22586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6844, 38), 'str', 'generic')

# Obtaining an instance of the builtin type 'tuple' (line 6844)
tuple_22587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6844, 50), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 6844)
# Adding element type (line 6844)
str_22588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6844, 50), 'str', 'flat')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6844, 50), tuple_22587, str_22588)
# Adding element type (line 6844)
str_22589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6845, 4), 'str', 'A 1-D view of the scalar.')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6844, 50), tuple_22587, str_22589)

# Processing the call keyword arguments (line 6844)
kwargs_22590 = {}
# Getting the type of 'add_newdoc' (line 6844)
add_newdoc_22584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6844, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 6844)
add_newdoc_call_result_22591 = invoke(stypy.reporting.localization.Localization(__file__, 6844, 0), add_newdoc_22584, *[str_22585, str_22586, tuple_22587], **kwargs_22590)


# Call to add_newdoc(...): (line 6847)
# Processing the call arguments (line 6847)
str_22593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6847, 11), 'str', 'numpy.core.numerictypes')
str_22594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6847, 38), 'str', 'generic')

# Obtaining an instance of the builtin type 'tuple' (line 6847)
tuple_22595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6847, 50), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 6847)
# Adding element type (line 6847)
str_22596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6847, 50), 'str', 'imag')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6847, 50), tuple_22595, str_22596)
# Adding element type (line 6847)
str_22597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6848, 4), 'str', 'The imaginary part of the scalar.')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6847, 50), tuple_22595, str_22597)

# Processing the call keyword arguments (line 6847)
kwargs_22598 = {}
# Getting the type of 'add_newdoc' (line 6847)
add_newdoc_22592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6847, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 6847)
add_newdoc_call_result_22599 = invoke(stypy.reporting.localization.Localization(__file__, 6847, 0), add_newdoc_22592, *[str_22593, str_22594, tuple_22595], **kwargs_22598)


# Call to add_newdoc(...): (line 6850)
# Processing the call arguments (line 6850)
str_22601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6850, 11), 'str', 'numpy.core.numerictypes')
str_22602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6850, 38), 'str', 'generic')

# Obtaining an instance of the builtin type 'tuple' (line 6850)
tuple_22603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6850, 50), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 6850)
# Adding element type (line 6850)
str_22604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6850, 50), 'str', 'itemsize')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6850, 50), tuple_22603, str_22604)
# Adding element type (line 6850)
str_22605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6851, 4), 'str', 'The length of one element in bytes.')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6850, 50), tuple_22603, str_22605)

# Processing the call keyword arguments (line 6850)
kwargs_22606 = {}
# Getting the type of 'add_newdoc' (line 6850)
add_newdoc_22600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6850, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 6850)
add_newdoc_call_result_22607 = invoke(stypy.reporting.localization.Localization(__file__, 6850, 0), add_newdoc_22600, *[str_22601, str_22602, tuple_22603], **kwargs_22606)


# Call to add_newdoc(...): (line 6853)
# Processing the call arguments (line 6853)
str_22609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6853, 11), 'str', 'numpy.core.numerictypes')
str_22610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6853, 38), 'str', 'generic')

# Obtaining an instance of the builtin type 'tuple' (line 6853)
tuple_22611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6853, 50), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 6853)
# Adding element type (line 6853)
str_22612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6853, 50), 'str', 'nbytes')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6853, 50), tuple_22611, str_22612)
# Adding element type (line 6853)
str_22613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6854, 4), 'str', 'The length of the scalar in bytes.')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6853, 50), tuple_22611, str_22613)

# Processing the call keyword arguments (line 6853)
kwargs_22614 = {}
# Getting the type of 'add_newdoc' (line 6853)
add_newdoc_22608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6853, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 6853)
add_newdoc_call_result_22615 = invoke(stypy.reporting.localization.Localization(__file__, 6853, 0), add_newdoc_22608, *[str_22609, str_22610, tuple_22611], **kwargs_22614)


# Call to add_newdoc(...): (line 6856)
# Processing the call arguments (line 6856)
str_22617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6856, 11), 'str', 'numpy.core.numerictypes')
str_22618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6856, 38), 'str', 'generic')

# Obtaining an instance of the builtin type 'tuple' (line 6856)
tuple_22619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6856, 50), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 6856)
# Adding element type (line 6856)
str_22620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6856, 50), 'str', 'ndim')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6856, 50), tuple_22619, str_22620)
# Adding element type (line 6856)
str_22621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6857, 4), 'str', 'The number of array dimensions.')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6856, 50), tuple_22619, str_22621)

# Processing the call keyword arguments (line 6856)
kwargs_22622 = {}
# Getting the type of 'add_newdoc' (line 6856)
add_newdoc_22616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6856, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 6856)
add_newdoc_call_result_22623 = invoke(stypy.reporting.localization.Localization(__file__, 6856, 0), add_newdoc_22616, *[str_22617, str_22618, tuple_22619], **kwargs_22622)


# Call to add_newdoc(...): (line 6859)
# Processing the call arguments (line 6859)
str_22625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6859, 11), 'str', 'numpy.core.numerictypes')
str_22626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6859, 38), 'str', 'generic')

# Obtaining an instance of the builtin type 'tuple' (line 6859)
tuple_22627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6859, 50), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 6859)
# Adding element type (line 6859)
str_22628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6859, 50), 'str', 'real')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6859, 50), tuple_22627, str_22628)
# Adding element type (line 6859)
str_22629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6860, 4), 'str', 'The real part of the scalar.')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6859, 50), tuple_22627, str_22629)

# Processing the call keyword arguments (line 6859)
kwargs_22630 = {}
# Getting the type of 'add_newdoc' (line 6859)
add_newdoc_22624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6859, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 6859)
add_newdoc_call_result_22631 = invoke(stypy.reporting.localization.Localization(__file__, 6859, 0), add_newdoc_22624, *[str_22625, str_22626, tuple_22627], **kwargs_22630)


# Call to add_newdoc(...): (line 6862)
# Processing the call arguments (line 6862)
str_22633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6862, 11), 'str', 'numpy.core.numerictypes')
str_22634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6862, 38), 'str', 'generic')

# Obtaining an instance of the builtin type 'tuple' (line 6862)
tuple_22635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6862, 50), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 6862)
# Adding element type (line 6862)
str_22636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6862, 50), 'str', 'shape')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6862, 50), tuple_22635, str_22636)
# Adding element type (line 6862)
str_22637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6863, 4), 'str', 'Tuple of array dimensions.')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6862, 50), tuple_22635, str_22637)

# Processing the call keyword arguments (line 6862)
kwargs_22638 = {}
# Getting the type of 'add_newdoc' (line 6862)
add_newdoc_22632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6862, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 6862)
add_newdoc_call_result_22639 = invoke(stypy.reporting.localization.Localization(__file__, 6862, 0), add_newdoc_22632, *[str_22633, str_22634, tuple_22635], **kwargs_22638)


# Call to add_newdoc(...): (line 6865)
# Processing the call arguments (line 6865)
str_22641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6865, 11), 'str', 'numpy.core.numerictypes')
str_22642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6865, 38), 'str', 'generic')

# Obtaining an instance of the builtin type 'tuple' (line 6865)
tuple_22643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6865, 50), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 6865)
# Adding element type (line 6865)
str_22644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6865, 50), 'str', 'size')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6865, 50), tuple_22643, str_22644)
# Adding element type (line 6865)
str_22645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6866, 4), 'str', 'The number of elements in the gentype.')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6865, 50), tuple_22643, str_22645)

# Processing the call keyword arguments (line 6865)
kwargs_22646 = {}
# Getting the type of 'add_newdoc' (line 6865)
add_newdoc_22640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6865, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 6865)
add_newdoc_call_result_22647 = invoke(stypy.reporting.localization.Localization(__file__, 6865, 0), add_newdoc_22640, *[str_22641, str_22642, tuple_22643], **kwargs_22646)


# Call to add_newdoc(...): (line 6868)
# Processing the call arguments (line 6868)
str_22649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6868, 11), 'str', 'numpy.core.numerictypes')
str_22650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6868, 38), 'str', 'generic')

# Obtaining an instance of the builtin type 'tuple' (line 6868)
tuple_22651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6868, 50), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 6868)
# Adding element type (line 6868)
str_22652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6868, 50), 'str', 'strides')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6868, 50), tuple_22651, str_22652)
# Adding element type (line 6868)
str_22653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6869, 4), 'str', 'Tuple of bytes steps in each dimension.')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6868, 50), tuple_22651, str_22653)

# Processing the call keyword arguments (line 6868)
kwargs_22654 = {}
# Getting the type of 'add_newdoc' (line 6868)
add_newdoc_22648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6868, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 6868)
add_newdoc_call_result_22655 = invoke(stypy.reporting.localization.Localization(__file__, 6868, 0), add_newdoc_22648, *[str_22649, str_22650, tuple_22651], **kwargs_22654)


# Call to add_newdoc(...): (line 6873)
# Processing the call arguments (line 6873)
str_22657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6873, 11), 'str', 'numpy.core.numerictypes')
str_22658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6873, 38), 'str', 'generic')

# Obtaining an instance of the builtin type 'tuple' (line 6873)
tuple_22659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6873, 50), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 6873)
# Adding element type (line 6873)
str_22660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6873, 50), 'str', 'all')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6873, 50), tuple_22659, str_22660)
# Adding element type (line 6873)
str_22661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6885, (-1)), 'str', '\n    Not implemented (virtual attribute)\n\n    Class generic exists solely to derive numpy scalars from, and possesses,\n    albeit unimplemented, all the attributes of the ndarray class\n    so as to provide a uniform API.\n\n    See Also\n    --------\n    The corresponding attribute of the derived class of interest.\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6873, 50), tuple_22659, str_22661)

# Processing the call keyword arguments (line 6873)
kwargs_22662 = {}
# Getting the type of 'add_newdoc' (line 6873)
add_newdoc_22656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6873, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 6873)
add_newdoc_call_result_22663 = invoke(stypy.reporting.localization.Localization(__file__, 6873, 0), add_newdoc_22656, *[str_22657, str_22658, tuple_22659], **kwargs_22662)


# Call to add_newdoc(...): (line 6887)
# Processing the call arguments (line 6887)
str_22665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6887, 11), 'str', 'numpy.core.numerictypes')
str_22666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6887, 38), 'str', 'generic')

# Obtaining an instance of the builtin type 'tuple' (line 6887)
tuple_22667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6887, 50), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 6887)
# Adding element type (line 6887)
str_22668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6887, 50), 'str', 'any')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6887, 50), tuple_22667, str_22668)
# Adding element type (line 6887)
str_22669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6899, (-1)), 'str', '\n    Not implemented (virtual attribute)\n\n    Class generic exists solely to derive numpy scalars from, and possesses,\n    albeit unimplemented, all the attributes of the ndarray class\n    so as to provide a uniform API.\n\n    See Also\n    --------\n    The corresponding attribute of the derived class of interest.\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6887, 50), tuple_22667, str_22669)

# Processing the call keyword arguments (line 6887)
kwargs_22670 = {}
# Getting the type of 'add_newdoc' (line 6887)
add_newdoc_22664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6887, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 6887)
add_newdoc_call_result_22671 = invoke(stypy.reporting.localization.Localization(__file__, 6887, 0), add_newdoc_22664, *[str_22665, str_22666, tuple_22667], **kwargs_22670)


# Call to add_newdoc(...): (line 6901)
# Processing the call arguments (line 6901)
str_22673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6901, 11), 'str', 'numpy.core.numerictypes')
str_22674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6901, 38), 'str', 'generic')

# Obtaining an instance of the builtin type 'tuple' (line 6901)
tuple_22675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6901, 50), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 6901)
# Adding element type (line 6901)
str_22676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6901, 50), 'str', 'argmax')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6901, 50), tuple_22675, str_22676)
# Adding element type (line 6901)
str_22677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6913, (-1)), 'str', '\n    Not implemented (virtual attribute)\n\n    Class generic exists solely to derive numpy scalars from, and possesses,\n    albeit unimplemented, all the attributes of the ndarray class\n    so as to provide a uniform API.\n\n    See Also\n    --------\n    The corresponding attribute of the derived class of interest.\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6901, 50), tuple_22675, str_22677)

# Processing the call keyword arguments (line 6901)
kwargs_22678 = {}
# Getting the type of 'add_newdoc' (line 6901)
add_newdoc_22672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6901, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 6901)
add_newdoc_call_result_22679 = invoke(stypy.reporting.localization.Localization(__file__, 6901, 0), add_newdoc_22672, *[str_22673, str_22674, tuple_22675], **kwargs_22678)


# Call to add_newdoc(...): (line 6915)
# Processing the call arguments (line 6915)
str_22681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6915, 11), 'str', 'numpy.core.numerictypes')
str_22682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6915, 38), 'str', 'generic')

# Obtaining an instance of the builtin type 'tuple' (line 6915)
tuple_22683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6915, 50), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 6915)
# Adding element type (line 6915)
str_22684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6915, 50), 'str', 'argmin')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6915, 50), tuple_22683, str_22684)
# Adding element type (line 6915)
str_22685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6927, (-1)), 'str', '\n    Not implemented (virtual attribute)\n\n    Class generic exists solely to derive numpy scalars from, and possesses,\n    albeit unimplemented, all the attributes of the ndarray class\n    so as to provide a uniform API.\n\n    See Also\n    --------\n    The corresponding attribute of the derived class of interest.\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6915, 50), tuple_22683, str_22685)

# Processing the call keyword arguments (line 6915)
kwargs_22686 = {}
# Getting the type of 'add_newdoc' (line 6915)
add_newdoc_22680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6915, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 6915)
add_newdoc_call_result_22687 = invoke(stypy.reporting.localization.Localization(__file__, 6915, 0), add_newdoc_22680, *[str_22681, str_22682, tuple_22683], **kwargs_22686)


# Call to add_newdoc(...): (line 6929)
# Processing the call arguments (line 6929)
str_22689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6929, 11), 'str', 'numpy.core.numerictypes')
str_22690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6929, 38), 'str', 'generic')

# Obtaining an instance of the builtin type 'tuple' (line 6929)
tuple_22691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6929, 50), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 6929)
# Adding element type (line 6929)
str_22692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6929, 50), 'str', 'argsort')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6929, 50), tuple_22691, str_22692)
# Adding element type (line 6929)
str_22693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6941, (-1)), 'str', '\n    Not implemented (virtual attribute)\n\n    Class generic exists solely to derive numpy scalars from, and possesses,\n    albeit unimplemented, all the attributes of the ndarray class\n    so as to provide a uniform API.\n\n    See Also\n    --------\n    The corresponding attribute of the derived class of interest.\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6929, 50), tuple_22691, str_22693)

# Processing the call keyword arguments (line 6929)
kwargs_22694 = {}
# Getting the type of 'add_newdoc' (line 6929)
add_newdoc_22688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6929, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 6929)
add_newdoc_call_result_22695 = invoke(stypy.reporting.localization.Localization(__file__, 6929, 0), add_newdoc_22688, *[str_22689, str_22690, tuple_22691], **kwargs_22694)


# Call to add_newdoc(...): (line 6943)
# Processing the call arguments (line 6943)
str_22697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6943, 11), 'str', 'numpy.core.numerictypes')
str_22698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6943, 38), 'str', 'generic')

# Obtaining an instance of the builtin type 'tuple' (line 6943)
tuple_22699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6943, 50), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 6943)
# Adding element type (line 6943)
str_22700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6943, 50), 'str', 'astype')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6943, 50), tuple_22699, str_22700)
# Adding element type (line 6943)
str_22701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6955, (-1)), 'str', '\n    Not implemented (virtual attribute)\n\n    Class generic exists solely to derive numpy scalars from, and possesses,\n    albeit unimplemented, all the attributes of the ndarray class\n    so as to provide a uniform API.\n\n    See Also\n    --------\n    The corresponding attribute of the derived class of interest.\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6943, 50), tuple_22699, str_22701)

# Processing the call keyword arguments (line 6943)
kwargs_22702 = {}
# Getting the type of 'add_newdoc' (line 6943)
add_newdoc_22696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6943, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 6943)
add_newdoc_call_result_22703 = invoke(stypy.reporting.localization.Localization(__file__, 6943, 0), add_newdoc_22696, *[str_22697, str_22698, tuple_22699], **kwargs_22702)


# Call to add_newdoc(...): (line 6957)
# Processing the call arguments (line 6957)
str_22705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6957, 11), 'str', 'numpy.core.numerictypes')
str_22706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6957, 38), 'str', 'generic')

# Obtaining an instance of the builtin type 'tuple' (line 6957)
tuple_22707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6957, 50), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 6957)
# Adding element type (line 6957)
str_22708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6957, 50), 'str', 'byteswap')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6957, 50), tuple_22707, str_22708)
# Adding element type (line 6957)
str_22709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6969, (-1)), 'str', '\n    Not implemented (virtual attribute)\n\n    Class generic exists solely to derive numpy scalars from, and possesses,\n    albeit unimplemented, all the attributes of the ndarray class so as to\n    provide a uniform API.\n\n    See Also\n    --------\n    The corresponding attribute of the derived class of interest.\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6957, 50), tuple_22707, str_22709)

# Processing the call keyword arguments (line 6957)
kwargs_22710 = {}
# Getting the type of 'add_newdoc' (line 6957)
add_newdoc_22704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6957, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 6957)
add_newdoc_call_result_22711 = invoke(stypy.reporting.localization.Localization(__file__, 6957, 0), add_newdoc_22704, *[str_22705, str_22706, tuple_22707], **kwargs_22710)


# Call to add_newdoc(...): (line 6971)
# Processing the call arguments (line 6971)
str_22713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6971, 11), 'str', 'numpy.core.numerictypes')
str_22714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6971, 38), 'str', 'generic')

# Obtaining an instance of the builtin type 'tuple' (line 6971)
tuple_22715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6971, 50), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 6971)
# Adding element type (line 6971)
str_22716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6971, 50), 'str', 'choose')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6971, 50), tuple_22715, str_22716)
# Adding element type (line 6971)
str_22717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6983, (-1)), 'str', '\n    Not implemented (virtual attribute)\n\n    Class generic exists solely to derive numpy scalars from, and possesses,\n    albeit unimplemented, all the attributes of the ndarray class\n    so as to provide a uniform API.\n\n    See Also\n    --------\n    The corresponding attribute of the derived class of interest.\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6971, 50), tuple_22715, str_22717)

# Processing the call keyword arguments (line 6971)
kwargs_22718 = {}
# Getting the type of 'add_newdoc' (line 6971)
add_newdoc_22712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6971, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 6971)
add_newdoc_call_result_22719 = invoke(stypy.reporting.localization.Localization(__file__, 6971, 0), add_newdoc_22712, *[str_22713, str_22714, tuple_22715], **kwargs_22718)


# Call to add_newdoc(...): (line 6985)
# Processing the call arguments (line 6985)
str_22721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6985, 11), 'str', 'numpy.core.numerictypes')
str_22722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6985, 38), 'str', 'generic')

# Obtaining an instance of the builtin type 'tuple' (line 6985)
tuple_22723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6985, 50), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 6985)
# Adding element type (line 6985)
str_22724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6985, 50), 'str', 'clip')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6985, 50), tuple_22723, str_22724)
# Adding element type (line 6985)
str_22725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6997, (-1)), 'str', '\n    Not implemented (virtual attribute)\n\n    Class generic exists solely to derive numpy scalars from, and possesses,\n    albeit unimplemented, all the attributes of the ndarray class\n    so as to provide a uniform API.\n\n    See Also\n    --------\n    The corresponding attribute of the derived class of interest.\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6985, 50), tuple_22723, str_22725)

# Processing the call keyword arguments (line 6985)
kwargs_22726 = {}
# Getting the type of 'add_newdoc' (line 6985)
add_newdoc_22720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6985, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 6985)
add_newdoc_call_result_22727 = invoke(stypy.reporting.localization.Localization(__file__, 6985, 0), add_newdoc_22720, *[str_22721, str_22722, tuple_22723], **kwargs_22726)


# Call to add_newdoc(...): (line 6999)
# Processing the call arguments (line 6999)
str_22729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6999, 11), 'str', 'numpy.core.numerictypes')
str_22730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6999, 38), 'str', 'generic')

# Obtaining an instance of the builtin type 'tuple' (line 6999)
tuple_22731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6999, 50), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 6999)
# Adding element type (line 6999)
str_22732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6999, 50), 'str', 'compress')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6999, 50), tuple_22731, str_22732)
# Adding element type (line 6999)
str_22733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7011, (-1)), 'str', '\n    Not implemented (virtual attribute)\n\n    Class generic exists solely to derive numpy scalars from, and possesses,\n    albeit unimplemented, all the attributes of the ndarray class\n    so as to provide a uniform API.\n\n    See Also\n    --------\n    The corresponding attribute of the derived class of interest.\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6999, 50), tuple_22731, str_22733)

# Processing the call keyword arguments (line 6999)
kwargs_22734 = {}
# Getting the type of 'add_newdoc' (line 6999)
add_newdoc_22728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6999, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 6999)
add_newdoc_call_result_22735 = invoke(stypy.reporting.localization.Localization(__file__, 6999, 0), add_newdoc_22728, *[str_22729, str_22730, tuple_22731], **kwargs_22734)


# Call to add_newdoc(...): (line 7013)
# Processing the call arguments (line 7013)
str_22737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7013, 11), 'str', 'numpy.core.numerictypes')
str_22738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7013, 38), 'str', 'generic')

# Obtaining an instance of the builtin type 'tuple' (line 7013)
tuple_22739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7013, 50), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 7013)
# Adding element type (line 7013)
str_22740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7013, 50), 'str', 'conjugate')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7013, 50), tuple_22739, str_22740)
# Adding element type (line 7013)
str_22741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7025, (-1)), 'str', '\n    Not implemented (virtual attribute)\n\n    Class generic exists solely to derive numpy scalars from, and possesses,\n    albeit unimplemented, all the attributes of the ndarray class\n    so as to provide a uniform API.\n\n    See Also\n    --------\n    The corresponding attribute of the derived class of interest.\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7013, 50), tuple_22739, str_22741)

# Processing the call keyword arguments (line 7013)
kwargs_22742 = {}
# Getting the type of 'add_newdoc' (line 7013)
add_newdoc_22736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7013, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 7013)
add_newdoc_call_result_22743 = invoke(stypy.reporting.localization.Localization(__file__, 7013, 0), add_newdoc_22736, *[str_22737, str_22738, tuple_22739], **kwargs_22742)


# Call to add_newdoc(...): (line 7027)
# Processing the call arguments (line 7027)
str_22745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7027, 11), 'str', 'numpy.core.numerictypes')
str_22746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7027, 38), 'str', 'generic')

# Obtaining an instance of the builtin type 'tuple' (line 7027)
tuple_22747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7027, 50), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 7027)
# Adding element type (line 7027)
str_22748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7027, 50), 'str', 'copy')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7027, 50), tuple_22747, str_22748)
# Adding element type (line 7027)
str_22749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7039, (-1)), 'str', '\n    Not implemented (virtual attribute)\n\n    Class generic exists solely to derive numpy scalars from, and possesses,\n    albeit unimplemented, all the attributes of the ndarray class\n    so as to provide a uniform API.\n\n    See Also\n    --------\n    The corresponding attribute of the derived class of interest.\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7027, 50), tuple_22747, str_22749)

# Processing the call keyword arguments (line 7027)
kwargs_22750 = {}
# Getting the type of 'add_newdoc' (line 7027)
add_newdoc_22744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7027, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 7027)
add_newdoc_call_result_22751 = invoke(stypy.reporting.localization.Localization(__file__, 7027, 0), add_newdoc_22744, *[str_22745, str_22746, tuple_22747], **kwargs_22750)


# Call to add_newdoc(...): (line 7041)
# Processing the call arguments (line 7041)
str_22753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7041, 11), 'str', 'numpy.core.numerictypes')
str_22754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7041, 38), 'str', 'generic')

# Obtaining an instance of the builtin type 'tuple' (line 7041)
tuple_22755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7041, 50), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 7041)
# Adding element type (line 7041)
str_22756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7041, 50), 'str', 'cumprod')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7041, 50), tuple_22755, str_22756)
# Adding element type (line 7041)
str_22757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7053, (-1)), 'str', '\n    Not implemented (virtual attribute)\n\n    Class generic exists solely to derive numpy scalars from, and possesses,\n    albeit unimplemented, all the attributes of the ndarray class\n    so as to provide a uniform API.\n\n    See Also\n    --------\n    The corresponding attribute of the derived class of interest.\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7041, 50), tuple_22755, str_22757)

# Processing the call keyword arguments (line 7041)
kwargs_22758 = {}
# Getting the type of 'add_newdoc' (line 7041)
add_newdoc_22752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7041, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 7041)
add_newdoc_call_result_22759 = invoke(stypy.reporting.localization.Localization(__file__, 7041, 0), add_newdoc_22752, *[str_22753, str_22754, tuple_22755], **kwargs_22758)


# Call to add_newdoc(...): (line 7055)
# Processing the call arguments (line 7055)
str_22761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7055, 11), 'str', 'numpy.core.numerictypes')
str_22762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7055, 38), 'str', 'generic')

# Obtaining an instance of the builtin type 'tuple' (line 7055)
tuple_22763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7055, 50), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 7055)
# Adding element type (line 7055)
str_22764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7055, 50), 'str', 'cumsum')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7055, 50), tuple_22763, str_22764)
# Adding element type (line 7055)
str_22765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7067, (-1)), 'str', '\n    Not implemented (virtual attribute)\n\n    Class generic exists solely to derive numpy scalars from, and possesses,\n    albeit unimplemented, all the attributes of the ndarray class\n    so as to provide a uniform API.\n\n    See Also\n    --------\n    The corresponding attribute of the derived class of interest.\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7055, 50), tuple_22763, str_22765)

# Processing the call keyword arguments (line 7055)
kwargs_22766 = {}
# Getting the type of 'add_newdoc' (line 7055)
add_newdoc_22760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7055, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 7055)
add_newdoc_call_result_22767 = invoke(stypy.reporting.localization.Localization(__file__, 7055, 0), add_newdoc_22760, *[str_22761, str_22762, tuple_22763], **kwargs_22766)


# Call to add_newdoc(...): (line 7069)
# Processing the call arguments (line 7069)
str_22769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7069, 11), 'str', 'numpy.core.numerictypes')
str_22770 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7069, 38), 'str', 'generic')

# Obtaining an instance of the builtin type 'tuple' (line 7069)
tuple_22771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7069, 50), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 7069)
# Adding element type (line 7069)
str_22772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7069, 50), 'str', 'diagonal')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7069, 50), tuple_22771, str_22772)
# Adding element type (line 7069)
str_22773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7081, (-1)), 'str', '\n    Not implemented (virtual attribute)\n\n    Class generic exists solely to derive numpy scalars from, and possesses,\n    albeit unimplemented, all the attributes of the ndarray class\n    so as to provide a uniform API.\n\n    See Also\n    --------\n    The corresponding attribute of the derived class of interest.\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7069, 50), tuple_22771, str_22773)

# Processing the call keyword arguments (line 7069)
kwargs_22774 = {}
# Getting the type of 'add_newdoc' (line 7069)
add_newdoc_22768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7069, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 7069)
add_newdoc_call_result_22775 = invoke(stypy.reporting.localization.Localization(__file__, 7069, 0), add_newdoc_22768, *[str_22769, str_22770, tuple_22771], **kwargs_22774)


# Call to add_newdoc(...): (line 7083)
# Processing the call arguments (line 7083)
str_22777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7083, 11), 'str', 'numpy.core.numerictypes')
str_22778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7083, 38), 'str', 'generic')

# Obtaining an instance of the builtin type 'tuple' (line 7083)
tuple_22779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7083, 50), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 7083)
# Adding element type (line 7083)
str_22780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7083, 50), 'str', 'dump')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7083, 50), tuple_22779, str_22780)
# Adding element type (line 7083)
str_22781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7095, (-1)), 'str', '\n    Not implemented (virtual attribute)\n\n    Class generic exists solely to derive numpy scalars from, and possesses,\n    albeit unimplemented, all the attributes of the ndarray class\n    so as to provide a uniform API.\n\n    See Also\n    --------\n    The corresponding attribute of the derived class of interest.\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7083, 50), tuple_22779, str_22781)

# Processing the call keyword arguments (line 7083)
kwargs_22782 = {}
# Getting the type of 'add_newdoc' (line 7083)
add_newdoc_22776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7083, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 7083)
add_newdoc_call_result_22783 = invoke(stypy.reporting.localization.Localization(__file__, 7083, 0), add_newdoc_22776, *[str_22777, str_22778, tuple_22779], **kwargs_22782)


# Call to add_newdoc(...): (line 7097)
# Processing the call arguments (line 7097)
str_22785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7097, 11), 'str', 'numpy.core.numerictypes')
str_22786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7097, 38), 'str', 'generic')

# Obtaining an instance of the builtin type 'tuple' (line 7097)
tuple_22787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7097, 50), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 7097)
# Adding element type (line 7097)
str_22788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7097, 50), 'str', 'dumps')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7097, 50), tuple_22787, str_22788)
# Adding element type (line 7097)
str_22789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7109, (-1)), 'str', '\n    Not implemented (virtual attribute)\n\n    Class generic exists solely to derive numpy scalars from, and possesses,\n    albeit unimplemented, all the attributes of the ndarray class\n    so as to provide a uniform API.\n\n    See Also\n    --------\n    The corresponding attribute of the derived class of interest.\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7097, 50), tuple_22787, str_22789)

# Processing the call keyword arguments (line 7097)
kwargs_22790 = {}
# Getting the type of 'add_newdoc' (line 7097)
add_newdoc_22784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7097, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 7097)
add_newdoc_call_result_22791 = invoke(stypy.reporting.localization.Localization(__file__, 7097, 0), add_newdoc_22784, *[str_22785, str_22786, tuple_22787], **kwargs_22790)


# Call to add_newdoc(...): (line 7111)
# Processing the call arguments (line 7111)
str_22793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7111, 11), 'str', 'numpy.core.numerictypes')
str_22794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7111, 38), 'str', 'generic')

# Obtaining an instance of the builtin type 'tuple' (line 7111)
tuple_22795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7111, 50), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 7111)
# Adding element type (line 7111)
str_22796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7111, 50), 'str', 'fill')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7111, 50), tuple_22795, str_22796)
# Adding element type (line 7111)
str_22797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7123, (-1)), 'str', '\n    Not implemented (virtual attribute)\n\n    Class generic exists solely to derive numpy scalars from, and possesses,\n    albeit unimplemented, all the attributes of the ndarray class\n    so as to provide a uniform API.\n\n    See Also\n    --------\n    The corresponding attribute of the derived class of interest.\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7111, 50), tuple_22795, str_22797)

# Processing the call keyword arguments (line 7111)
kwargs_22798 = {}
# Getting the type of 'add_newdoc' (line 7111)
add_newdoc_22792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7111, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 7111)
add_newdoc_call_result_22799 = invoke(stypy.reporting.localization.Localization(__file__, 7111, 0), add_newdoc_22792, *[str_22793, str_22794, tuple_22795], **kwargs_22798)


# Call to add_newdoc(...): (line 7125)
# Processing the call arguments (line 7125)
str_22801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7125, 11), 'str', 'numpy.core.numerictypes')
str_22802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7125, 38), 'str', 'generic')

# Obtaining an instance of the builtin type 'tuple' (line 7125)
tuple_22803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7125, 50), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 7125)
# Adding element type (line 7125)
str_22804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7125, 50), 'str', 'flatten')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7125, 50), tuple_22803, str_22804)
# Adding element type (line 7125)
str_22805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7137, (-1)), 'str', '\n    Not implemented (virtual attribute)\n\n    Class generic exists solely to derive numpy scalars from, and possesses,\n    albeit unimplemented, all the attributes of the ndarray class\n    so as to provide a uniform API.\n\n    See Also\n    --------\n    The corresponding attribute of the derived class of interest.\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7125, 50), tuple_22803, str_22805)

# Processing the call keyword arguments (line 7125)
kwargs_22806 = {}
# Getting the type of 'add_newdoc' (line 7125)
add_newdoc_22800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7125, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 7125)
add_newdoc_call_result_22807 = invoke(stypy.reporting.localization.Localization(__file__, 7125, 0), add_newdoc_22800, *[str_22801, str_22802, tuple_22803], **kwargs_22806)


# Call to add_newdoc(...): (line 7139)
# Processing the call arguments (line 7139)
str_22809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7139, 11), 'str', 'numpy.core.numerictypes')
str_22810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7139, 38), 'str', 'generic')

# Obtaining an instance of the builtin type 'tuple' (line 7139)
tuple_22811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7139, 50), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 7139)
# Adding element type (line 7139)
str_22812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7139, 50), 'str', 'getfield')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7139, 50), tuple_22811, str_22812)
# Adding element type (line 7139)
str_22813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7151, (-1)), 'str', '\n    Not implemented (virtual attribute)\n\n    Class generic exists solely to derive numpy scalars from, and possesses,\n    albeit unimplemented, all the attributes of the ndarray class\n    so as to provide a uniform API.\n\n    See Also\n    --------\n    The corresponding attribute of the derived class of interest.\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7139, 50), tuple_22811, str_22813)

# Processing the call keyword arguments (line 7139)
kwargs_22814 = {}
# Getting the type of 'add_newdoc' (line 7139)
add_newdoc_22808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7139, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 7139)
add_newdoc_call_result_22815 = invoke(stypy.reporting.localization.Localization(__file__, 7139, 0), add_newdoc_22808, *[str_22809, str_22810, tuple_22811], **kwargs_22814)


# Call to add_newdoc(...): (line 7153)
# Processing the call arguments (line 7153)
str_22817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7153, 11), 'str', 'numpy.core.numerictypes')
str_22818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7153, 38), 'str', 'generic')

# Obtaining an instance of the builtin type 'tuple' (line 7153)
tuple_22819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7153, 50), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 7153)
# Adding element type (line 7153)
str_22820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7153, 50), 'str', 'item')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7153, 50), tuple_22819, str_22820)
# Adding element type (line 7153)
str_22821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7165, (-1)), 'str', '\n    Not implemented (virtual attribute)\n\n    Class generic exists solely to derive numpy scalars from, and possesses,\n    albeit unimplemented, all the attributes of the ndarray class\n    so as to provide a uniform API.\n\n    See Also\n    --------\n    The corresponding attribute of the derived class of interest.\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7153, 50), tuple_22819, str_22821)

# Processing the call keyword arguments (line 7153)
kwargs_22822 = {}
# Getting the type of 'add_newdoc' (line 7153)
add_newdoc_22816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7153, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 7153)
add_newdoc_call_result_22823 = invoke(stypy.reporting.localization.Localization(__file__, 7153, 0), add_newdoc_22816, *[str_22817, str_22818, tuple_22819], **kwargs_22822)


# Call to add_newdoc(...): (line 7167)
# Processing the call arguments (line 7167)
str_22825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7167, 11), 'str', 'numpy.core.numerictypes')
str_22826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7167, 38), 'str', 'generic')

# Obtaining an instance of the builtin type 'tuple' (line 7167)
tuple_22827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7167, 50), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 7167)
# Adding element type (line 7167)
str_22828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7167, 50), 'str', 'itemset')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7167, 50), tuple_22827, str_22828)
# Adding element type (line 7167)
str_22829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7179, (-1)), 'str', '\n    Not implemented (virtual attribute)\n\n    Class generic exists solely to derive numpy scalars from, and possesses,\n    albeit unimplemented, all the attributes of the ndarray class\n    so as to provide a uniform API.\n\n    See Also\n    --------\n    The corresponding attribute of the derived class of interest.\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7167, 50), tuple_22827, str_22829)

# Processing the call keyword arguments (line 7167)
kwargs_22830 = {}
# Getting the type of 'add_newdoc' (line 7167)
add_newdoc_22824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7167, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 7167)
add_newdoc_call_result_22831 = invoke(stypy.reporting.localization.Localization(__file__, 7167, 0), add_newdoc_22824, *[str_22825, str_22826, tuple_22827], **kwargs_22830)


# Call to add_newdoc(...): (line 7181)
# Processing the call arguments (line 7181)
str_22833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7181, 11), 'str', 'numpy.core.numerictypes')
str_22834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7181, 38), 'str', 'generic')

# Obtaining an instance of the builtin type 'tuple' (line 7181)
tuple_22835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7181, 50), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 7181)
# Adding element type (line 7181)
str_22836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7181, 50), 'str', 'max')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7181, 50), tuple_22835, str_22836)
# Adding element type (line 7181)
str_22837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7193, (-1)), 'str', '\n    Not implemented (virtual attribute)\n\n    Class generic exists solely to derive numpy scalars from, and possesses,\n    albeit unimplemented, all the attributes of the ndarray class\n    so as to provide a uniform API.\n\n    See Also\n    --------\n    The corresponding attribute of the derived class of interest.\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7181, 50), tuple_22835, str_22837)

# Processing the call keyword arguments (line 7181)
kwargs_22838 = {}
# Getting the type of 'add_newdoc' (line 7181)
add_newdoc_22832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7181, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 7181)
add_newdoc_call_result_22839 = invoke(stypy.reporting.localization.Localization(__file__, 7181, 0), add_newdoc_22832, *[str_22833, str_22834, tuple_22835], **kwargs_22838)


# Call to add_newdoc(...): (line 7195)
# Processing the call arguments (line 7195)
str_22841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7195, 11), 'str', 'numpy.core.numerictypes')
str_22842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7195, 38), 'str', 'generic')

# Obtaining an instance of the builtin type 'tuple' (line 7195)
tuple_22843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7195, 50), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 7195)
# Adding element type (line 7195)
str_22844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7195, 50), 'str', 'mean')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7195, 50), tuple_22843, str_22844)
# Adding element type (line 7195)
str_22845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7207, (-1)), 'str', '\n    Not implemented (virtual attribute)\n\n    Class generic exists solely to derive numpy scalars from, and possesses,\n    albeit unimplemented, all the attributes of the ndarray class\n    so as to provide a uniform API.\n\n    See Also\n    --------\n    The corresponding attribute of the derived class of interest.\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7195, 50), tuple_22843, str_22845)

# Processing the call keyword arguments (line 7195)
kwargs_22846 = {}
# Getting the type of 'add_newdoc' (line 7195)
add_newdoc_22840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7195, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 7195)
add_newdoc_call_result_22847 = invoke(stypy.reporting.localization.Localization(__file__, 7195, 0), add_newdoc_22840, *[str_22841, str_22842, tuple_22843], **kwargs_22846)


# Call to add_newdoc(...): (line 7209)
# Processing the call arguments (line 7209)
str_22849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7209, 11), 'str', 'numpy.core.numerictypes')
str_22850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7209, 38), 'str', 'generic')

# Obtaining an instance of the builtin type 'tuple' (line 7209)
tuple_22851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7209, 50), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 7209)
# Adding element type (line 7209)
str_22852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7209, 50), 'str', 'min')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7209, 50), tuple_22851, str_22852)
# Adding element type (line 7209)
str_22853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7221, (-1)), 'str', '\n    Not implemented (virtual attribute)\n\n    Class generic exists solely to derive numpy scalars from, and possesses,\n    albeit unimplemented, all the attributes of the ndarray class\n    so as to provide a uniform API.\n\n    See Also\n    --------\n    The corresponding attribute of the derived class of interest.\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7209, 50), tuple_22851, str_22853)

# Processing the call keyword arguments (line 7209)
kwargs_22854 = {}
# Getting the type of 'add_newdoc' (line 7209)
add_newdoc_22848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7209, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 7209)
add_newdoc_call_result_22855 = invoke(stypy.reporting.localization.Localization(__file__, 7209, 0), add_newdoc_22848, *[str_22849, str_22850, tuple_22851], **kwargs_22854)


# Call to add_newdoc(...): (line 7223)
# Processing the call arguments (line 7223)
str_22857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7223, 11), 'str', 'numpy.core.numerictypes')
str_22858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7223, 38), 'str', 'generic')

# Obtaining an instance of the builtin type 'tuple' (line 7223)
tuple_22859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7223, 50), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 7223)
# Adding element type (line 7223)
str_22860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7223, 50), 'str', 'newbyteorder')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7223, 50), tuple_22859, str_22860)
# Adding element type (line 7223)
str_22861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7254, (-1)), 'str', "\n    newbyteorder(new_order='S')\n\n    Return a new `dtype` with a different byte order.\n\n    Changes are also made in all fields and sub-arrays of the data type.\n\n    The `new_order` code can be any from the following:\n\n    * 'S' - swap dtype from current to opposite endian\n    * {'<', 'L'} - little endian\n    * {'>', 'B'} - big endian\n    * {'=', 'N'} - native order\n    * {'|', 'I'} - ignore (no change to byte order)\n\n    Parameters\n    ----------\n    new_order : str, optional\n        Byte order to force; a value from the byte order specifications\n        above.  The default value ('S') results in swapping the current\n        byte order. The code does a case-insensitive check on the first\n        letter of `new_order` for the alternatives above.  For example,\n        any of 'B' or 'b' or 'biggish' are valid to specify big-endian.\n\n\n    Returns\n    -------\n    new_dtype : dtype\n        New `dtype` object with the given change to the byte order.\n\n    ")
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7223, 50), tuple_22859, str_22861)

# Processing the call keyword arguments (line 7223)
kwargs_22862 = {}
# Getting the type of 'add_newdoc' (line 7223)
add_newdoc_22856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7223, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 7223)
add_newdoc_call_result_22863 = invoke(stypy.reporting.localization.Localization(__file__, 7223, 0), add_newdoc_22856, *[str_22857, str_22858, tuple_22859], **kwargs_22862)


# Call to add_newdoc(...): (line 7256)
# Processing the call arguments (line 7256)
str_22865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7256, 11), 'str', 'numpy.core.numerictypes')
str_22866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7256, 38), 'str', 'generic')

# Obtaining an instance of the builtin type 'tuple' (line 7256)
tuple_22867 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7256, 50), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 7256)
# Adding element type (line 7256)
str_22868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7256, 50), 'str', 'nonzero')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7256, 50), tuple_22867, str_22868)
# Adding element type (line 7256)
str_22869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7268, (-1)), 'str', '\n    Not implemented (virtual attribute)\n\n    Class generic exists solely to derive numpy scalars from, and possesses,\n    albeit unimplemented, all the attributes of the ndarray class\n    so as to provide a uniform API.\n\n    See Also\n    --------\n    The corresponding attribute of the derived class of interest.\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7256, 50), tuple_22867, str_22869)

# Processing the call keyword arguments (line 7256)
kwargs_22870 = {}
# Getting the type of 'add_newdoc' (line 7256)
add_newdoc_22864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7256, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 7256)
add_newdoc_call_result_22871 = invoke(stypy.reporting.localization.Localization(__file__, 7256, 0), add_newdoc_22864, *[str_22865, str_22866, tuple_22867], **kwargs_22870)


# Call to add_newdoc(...): (line 7270)
# Processing the call arguments (line 7270)
str_22873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7270, 11), 'str', 'numpy.core.numerictypes')
str_22874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7270, 38), 'str', 'generic')

# Obtaining an instance of the builtin type 'tuple' (line 7270)
tuple_22875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7270, 50), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 7270)
# Adding element type (line 7270)
str_22876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7270, 50), 'str', 'prod')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7270, 50), tuple_22875, str_22876)
# Adding element type (line 7270)
str_22877 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7282, (-1)), 'str', '\n    Not implemented (virtual attribute)\n\n    Class generic exists solely to derive numpy scalars from, and possesses,\n    albeit unimplemented, all the attributes of the ndarray class\n    so as to provide a uniform API.\n\n    See Also\n    --------\n    The corresponding attribute of the derived class of interest.\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7270, 50), tuple_22875, str_22877)

# Processing the call keyword arguments (line 7270)
kwargs_22878 = {}
# Getting the type of 'add_newdoc' (line 7270)
add_newdoc_22872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7270, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 7270)
add_newdoc_call_result_22879 = invoke(stypy.reporting.localization.Localization(__file__, 7270, 0), add_newdoc_22872, *[str_22873, str_22874, tuple_22875], **kwargs_22878)


# Call to add_newdoc(...): (line 7284)
# Processing the call arguments (line 7284)
str_22881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7284, 11), 'str', 'numpy.core.numerictypes')
str_22882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7284, 38), 'str', 'generic')

# Obtaining an instance of the builtin type 'tuple' (line 7284)
tuple_22883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7284, 50), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 7284)
# Adding element type (line 7284)
str_22884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7284, 50), 'str', 'ptp')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7284, 50), tuple_22883, str_22884)
# Adding element type (line 7284)
str_22885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7296, (-1)), 'str', '\n    Not implemented (virtual attribute)\n\n    Class generic exists solely to derive numpy scalars from, and possesses,\n    albeit unimplemented, all the attributes of the ndarray class\n    so as to provide a uniform API.\n\n    See Also\n    --------\n    The corresponding attribute of the derived class of interest.\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7284, 50), tuple_22883, str_22885)

# Processing the call keyword arguments (line 7284)
kwargs_22886 = {}
# Getting the type of 'add_newdoc' (line 7284)
add_newdoc_22880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7284, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 7284)
add_newdoc_call_result_22887 = invoke(stypy.reporting.localization.Localization(__file__, 7284, 0), add_newdoc_22880, *[str_22881, str_22882, tuple_22883], **kwargs_22886)


# Call to add_newdoc(...): (line 7298)
# Processing the call arguments (line 7298)
str_22889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7298, 11), 'str', 'numpy.core.numerictypes')
str_22890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7298, 38), 'str', 'generic')

# Obtaining an instance of the builtin type 'tuple' (line 7298)
tuple_22891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7298, 50), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 7298)
# Adding element type (line 7298)
str_22892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7298, 50), 'str', 'put')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7298, 50), tuple_22891, str_22892)
# Adding element type (line 7298)
str_22893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7310, (-1)), 'str', '\n    Not implemented (virtual attribute)\n\n    Class generic exists solely to derive numpy scalars from, and possesses,\n    albeit unimplemented, all the attributes of the ndarray class\n    so as to provide a uniform API.\n\n    See Also\n    --------\n    The corresponding attribute of the derived class of interest.\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7298, 50), tuple_22891, str_22893)

# Processing the call keyword arguments (line 7298)
kwargs_22894 = {}
# Getting the type of 'add_newdoc' (line 7298)
add_newdoc_22888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7298, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 7298)
add_newdoc_call_result_22895 = invoke(stypy.reporting.localization.Localization(__file__, 7298, 0), add_newdoc_22888, *[str_22889, str_22890, tuple_22891], **kwargs_22894)


# Call to add_newdoc(...): (line 7312)
# Processing the call arguments (line 7312)
str_22897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7312, 11), 'str', 'numpy.core.numerictypes')
str_22898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7312, 38), 'str', 'generic')

# Obtaining an instance of the builtin type 'tuple' (line 7312)
tuple_22899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7312, 50), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 7312)
# Adding element type (line 7312)
str_22900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7312, 50), 'str', 'ravel')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7312, 50), tuple_22899, str_22900)
# Adding element type (line 7312)
str_22901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7324, (-1)), 'str', '\n    Not implemented (virtual attribute)\n\n    Class generic exists solely to derive numpy scalars from, and possesses,\n    albeit unimplemented, all the attributes of the ndarray class\n    so as to provide a uniform API.\n\n    See Also\n    --------\n    The corresponding attribute of the derived class of interest.\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7312, 50), tuple_22899, str_22901)

# Processing the call keyword arguments (line 7312)
kwargs_22902 = {}
# Getting the type of 'add_newdoc' (line 7312)
add_newdoc_22896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7312, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 7312)
add_newdoc_call_result_22903 = invoke(stypy.reporting.localization.Localization(__file__, 7312, 0), add_newdoc_22896, *[str_22897, str_22898, tuple_22899], **kwargs_22902)


# Call to add_newdoc(...): (line 7326)
# Processing the call arguments (line 7326)
str_22905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7326, 11), 'str', 'numpy.core.numerictypes')
str_22906 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7326, 38), 'str', 'generic')

# Obtaining an instance of the builtin type 'tuple' (line 7326)
tuple_22907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7326, 50), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 7326)
# Adding element type (line 7326)
str_22908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7326, 50), 'str', 'repeat')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7326, 50), tuple_22907, str_22908)
# Adding element type (line 7326)
str_22909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7338, (-1)), 'str', '\n    Not implemented (virtual attribute)\n\n    Class generic exists solely to derive numpy scalars from, and possesses,\n    albeit unimplemented, all the attributes of the ndarray class\n    so as to provide a uniform API.\n\n    See Also\n    --------\n    The corresponding attribute of the derived class of interest.\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7326, 50), tuple_22907, str_22909)

# Processing the call keyword arguments (line 7326)
kwargs_22910 = {}
# Getting the type of 'add_newdoc' (line 7326)
add_newdoc_22904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7326, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 7326)
add_newdoc_call_result_22911 = invoke(stypy.reporting.localization.Localization(__file__, 7326, 0), add_newdoc_22904, *[str_22905, str_22906, tuple_22907], **kwargs_22910)


# Call to add_newdoc(...): (line 7340)
# Processing the call arguments (line 7340)
str_22913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7340, 11), 'str', 'numpy.core.numerictypes')
str_22914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7340, 38), 'str', 'generic')

# Obtaining an instance of the builtin type 'tuple' (line 7340)
tuple_22915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7340, 50), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 7340)
# Adding element type (line 7340)
str_22916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7340, 50), 'str', 'reshape')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7340, 50), tuple_22915, str_22916)
# Adding element type (line 7340)
str_22917 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7352, (-1)), 'str', '\n    Not implemented (virtual attribute)\n\n    Class generic exists solely to derive numpy scalars from, and possesses,\n    albeit unimplemented, all the attributes of the ndarray class\n    so as to provide a uniform API.\n\n    See Also\n    --------\n    The corresponding attribute of the derived class of interest.\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7340, 50), tuple_22915, str_22917)

# Processing the call keyword arguments (line 7340)
kwargs_22918 = {}
# Getting the type of 'add_newdoc' (line 7340)
add_newdoc_22912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7340, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 7340)
add_newdoc_call_result_22919 = invoke(stypy.reporting.localization.Localization(__file__, 7340, 0), add_newdoc_22912, *[str_22913, str_22914, tuple_22915], **kwargs_22918)


# Call to add_newdoc(...): (line 7354)
# Processing the call arguments (line 7354)
str_22921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7354, 11), 'str', 'numpy.core.numerictypes')
str_22922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7354, 38), 'str', 'generic')

# Obtaining an instance of the builtin type 'tuple' (line 7354)
tuple_22923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7354, 50), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 7354)
# Adding element type (line 7354)
str_22924 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7354, 50), 'str', 'resize')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7354, 50), tuple_22923, str_22924)
# Adding element type (line 7354)
str_22925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7366, (-1)), 'str', '\n    Not implemented (virtual attribute)\n\n    Class generic exists solely to derive numpy scalars from, and possesses,\n    albeit unimplemented, all the attributes of the ndarray class\n    so as to provide a uniform API.\n\n    See Also\n    --------\n    The corresponding attribute of the derived class of interest.\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7354, 50), tuple_22923, str_22925)

# Processing the call keyword arguments (line 7354)
kwargs_22926 = {}
# Getting the type of 'add_newdoc' (line 7354)
add_newdoc_22920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7354, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 7354)
add_newdoc_call_result_22927 = invoke(stypy.reporting.localization.Localization(__file__, 7354, 0), add_newdoc_22920, *[str_22921, str_22922, tuple_22923], **kwargs_22926)


# Call to add_newdoc(...): (line 7368)
# Processing the call arguments (line 7368)
str_22929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7368, 11), 'str', 'numpy.core.numerictypes')
str_22930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7368, 38), 'str', 'generic')

# Obtaining an instance of the builtin type 'tuple' (line 7368)
tuple_22931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7368, 50), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 7368)
# Adding element type (line 7368)
str_22932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7368, 50), 'str', 'round')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7368, 50), tuple_22931, str_22932)
# Adding element type (line 7368)
str_22933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7380, (-1)), 'str', '\n    Not implemented (virtual attribute)\n\n    Class generic exists solely to derive numpy scalars from, and possesses,\n    albeit unimplemented, all the attributes of the ndarray class\n    so as to provide a uniform API.\n\n    See Also\n    --------\n    The corresponding attribute of the derived class of interest.\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7368, 50), tuple_22931, str_22933)

# Processing the call keyword arguments (line 7368)
kwargs_22934 = {}
# Getting the type of 'add_newdoc' (line 7368)
add_newdoc_22928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7368, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 7368)
add_newdoc_call_result_22935 = invoke(stypy.reporting.localization.Localization(__file__, 7368, 0), add_newdoc_22928, *[str_22929, str_22930, tuple_22931], **kwargs_22934)


# Call to add_newdoc(...): (line 7382)
# Processing the call arguments (line 7382)
str_22937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7382, 11), 'str', 'numpy.core.numerictypes')
str_22938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7382, 38), 'str', 'generic')

# Obtaining an instance of the builtin type 'tuple' (line 7382)
tuple_22939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7382, 50), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 7382)
# Adding element type (line 7382)
str_22940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7382, 50), 'str', 'searchsorted')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7382, 50), tuple_22939, str_22940)
# Adding element type (line 7382)
str_22941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7394, (-1)), 'str', '\n    Not implemented (virtual attribute)\n\n    Class generic exists solely to derive numpy scalars from, and possesses,\n    albeit unimplemented, all the attributes of the ndarray class\n    so as to provide a uniform API.\n\n    See Also\n    --------\n    The corresponding attribute of the derived class of interest.\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7382, 50), tuple_22939, str_22941)

# Processing the call keyword arguments (line 7382)
kwargs_22942 = {}
# Getting the type of 'add_newdoc' (line 7382)
add_newdoc_22936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7382, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 7382)
add_newdoc_call_result_22943 = invoke(stypy.reporting.localization.Localization(__file__, 7382, 0), add_newdoc_22936, *[str_22937, str_22938, tuple_22939], **kwargs_22942)


# Call to add_newdoc(...): (line 7396)
# Processing the call arguments (line 7396)
str_22945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7396, 11), 'str', 'numpy.core.numerictypes')
str_22946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7396, 38), 'str', 'generic')

# Obtaining an instance of the builtin type 'tuple' (line 7396)
tuple_22947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7396, 50), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 7396)
# Adding element type (line 7396)
str_22948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7396, 50), 'str', 'setfield')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7396, 50), tuple_22947, str_22948)
# Adding element type (line 7396)
str_22949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7408, (-1)), 'str', '\n    Not implemented (virtual attribute)\n\n    Class generic exists solely to derive numpy scalars from, and possesses,\n    albeit unimplemented, all the attributes of the ndarray class\n    so as to provide a uniform API.\n\n    See Also\n    --------\n    The corresponding attribute of the derived class of interest.\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7396, 50), tuple_22947, str_22949)

# Processing the call keyword arguments (line 7396)
kwargs_22950 = {}
# Getting the type of 'add_newdoc' (line 7396)
add_newdoc_22944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7396, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 7396)
add_newdoc_call_result_22951 = invoke(stypy.reporting.localization.Localization(__file__, 7396, 0), add_newdoc_22944, *[str_22945, str_22946, tuple_22947], **kwargs_22950)


# Call to add_newdoc(...): (line 7410)
# Processing the call arguments (line 7410)
str_22953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7410, 11), 'str', 'numpy.core.numerictypes')
str_22954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7410, 38), 'str', 'generic')

# Obtaining an instance of the builtin type 'tuple' (line 7410)
tuple_22955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7410, 50), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 7410)
# Adding element type (line 7410)
str_22956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7410, 50), 'str', 'setflags')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7410, 50), tuple_22955, str_22956)
# Adding element type (line 7410)
str_22957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7422, (-1)), 'str', '\n    Not implemented (virtual attribute)\n\n    Class generic exists solely to derive numpy scalars from, and possesses,\n    albeit unimplemented, all the attributes of the ndarray class so as to\n    provide a uniform API.\n\n    See Also\n    --------\n    The corresponding attribute of the derived class of interest.\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7410, 50), tuple_22955, str_22957)

# Processing the call keyword arguments (line 7410)
kwargs_22958 = {}
# Getting the type of 'add_newdoc' (line 7410)
add_newdoc_22952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7410, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 7410)
add_newdoc_call_result_22959 = invoke(stypy.reporting.localization.Localization(__file__, 7410, 0), add_newdoc_22952, *[str_22953, str_22954, tuple_22955], **kwargs_22958)


# Call to add_newdoc(...): (line 7424)
# Processing the call arguments (line 7424)
str_22961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7424, 11), 'str', 'numpy.core.numerictypes')
str_22962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7424, 38), 'str', 'generic')

# Obtaining an instance of the builtin type 'tuple' (line 7424)
tuple_22963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7424, 50), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 7424)
# Adding element type (line 7424)
str_22964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7424, 50), 'str', 'sort')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7424, 50), tuple_22963, str_22964)
# Adding element type (line 7424)
str_22965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7436, (-1)), 'str', '\n    Not implemented (virtual attribute)\n\n    Class generic exists solely to derive numpy scalars from, and possesses,\n    albeit unimplemented, all the attributes of the ndarray class\n    so as to provide a uniform API.\n\n    See Also\n    --------\n    The corresponding attribute of the derived class of interest.\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7424, 50), tuple_22963, str_22965)

# Processing the call keyword arguments (line 7424)
kwargs_22966 = {}
# Getting the type of 'add_newdoc' (line 7424)
add_newdoc_22960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7424, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 7424)
add_newdoc_call_result_22967 = invoke(stypy.reporting.localization.Localization(__file__, 7424, 0), add_newdoc_22960, *[str_22961, str_22962, tuple_22963], **kwargs_22966)


# Call to add_newdoc(...): (line 7438)
# Processing the call arguments (line 7438)
str_22969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7438, 11), 'str', 'numpy.core.numerictypes')
str_22970 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7438, 38), 'str', 'generic')

# Obtaining an instance of the builtin type 'tuple' (line 7438)
tuple_22971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7438, 50), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 7438)
# Adding element type (line 7438)
str_22972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7438, 50), 'str', 'squeeze')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7438, 50), tuple_22971, str_22972)
# Adding element type (line 7438)
str_22973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7450, (-1)), 'str', '\n    Not implemented (virtual attribute)\n\n    Class generic exists solely to derive numpy scalars from, and possesses,\n    albeit unimplemented, all the attributes of the ndarray class\n    so as to provide a uniform API.\n\n    See Also\n    --------\n    The corresponding attribute of the derived class of interest.\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7438, 50), tuple_22971, str_22973)

# Processing the call keyword arguments (line 7438)
kwargs_22974 = {}
# Getting the type of 'add_newdoc' (line 7438)
add_newdoc_22968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7438, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 7438)
add_newdoc_call_result_22975 = invoke(stypy.reporting.localization.Localization(__file__, 7438, 0), add_newdoc_22968, *[str_22969, str_22970, tuple_22971], **kwargs_22974)


# Call to add_newdoc(...): (line 7452)
# Processing the call arguments (line 7452)
str_22977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7452, 11), 'str', 'numpy.core.numerictypes')
str_22978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7452, 38), 'str', 'generic')

# Obtaining an instance of the builtin type 'tuple' (line 7452)
tuple_22979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7452, 50), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 7452)
# Adding element type (line 7452)
str_22980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7452, 50), 'str', 'std')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7452, 50), tuple_22979, str_22980)
# Adding element type (line 7452)
str_22981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7464, (-1)), 'str', '\n    Not implemented (virtual attribute)\n\n    Class generic exists solely to derive numpy scalars from, and possesses,\n    albeit unimplemented, all the attributes of the ndarray class\n    so as to provide a uniform API.\n\n    See Also\n    --------\n    The corresponding attribute of the derived class of interest.\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7452, 50), tuple_22979, str_22981)

# Processing the call keyword arguments (line 7452)
kwargs_22982 = {}
# Getting the type of 'add_newdoc' (line 7452)
add_newdoc_22976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7452, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 7452)
add_newdoc_call_result_22983 = invoke(stypy.reporting.localization.Localization(__file__, 7452, 0), add_newdoc_22976, *[str_22977, str_22978, tuple_22979], **kwargs_22982)


# Call to add_newdoc(...): (line 7466)
# Processing the call arguments (line 7466)
str_22985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7466, 11), 'str', 'numpy.core.numerictypes')
str_22986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7466, 38), 'str', 'generic')

# Obtaining an instance of the builtin type 'tuple' (line 7466)
tuple_22987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7466, 50), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 7466)
# Adding element type (line 7466)
str_22988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7466, 50), 'str', 'sum')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7466, 50), tuple_22987, str_22988)
# Adding element type (line 7466)
str_22989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7478, (-1)), 'str', '\n    Not implemented (virtual attribute)\n\n    Class generic exists solely to derive numpy scalars from, and possesses,\n    albeit unimplemented, all the attributes of the ndarray class\n    so as to provide a uniform API.\n\n    See Also\n    --------\n    The corresponding attribute of the derived class of interest.\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7466, 50), tuple_22987, str_22989)

# Processing the call keyword arguments (line 7466)
kwargs_22990 = {}
# Getting the type of 'add_newdoc' (line 7466)
add_newdoc_22984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7466, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 7466)
add_newdoc_call_result_22991 = invoke(stypy.reporting.localization.Localization(__file__, 7466, 0), add_newdoc_22984, *[str_22985, str_22986, tuple_22987], **kwargs_22990)


# Call to add_newdoc(...): (line 7480)
# Processing the call arguments (line 7480)
str_22993 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7480, 11), 'str', 'numpy.core.numerictypes')
str_22994 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7480, 38), 'str', 'generic')

# Obtaining an instance of the builtin type 'tuple' (line 7480)
tuple_22995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7480, 50), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 7480)
# Adding element type (line 7480)
str_22996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7480, 50), 'str', 'swapaxes')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7480, 50), tuple_22995, str_22996)
# Adding element type (line 7480)
str_22997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7492, (-1)), 'str', '\n    Not implemented (virtual attribute)\n\n    Class generic exists solely to derive numpy scalars from, and possesses,\n    albeit unimplemented, all the attributes of the ndarray class\n    so as to provide a uniform API.\n\n    See Also\n    --------\n    The corresponding attribute of the derived class of interest.\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7480, 50), tuple_22995, str_22997)

# Processing the call keyword arguments (line 7480)
kwargs_22998 = {}
# Getting the type of 'add_newdoc' (line 7480)
add_newdoc_22992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7480, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 7480)
add_newdoc_call_result_22999 = invoke(stypy.reporting.localization.Localization(__file__, 7480, 0), add_newdoc_22992, *[str_22993, str_22994, tuple_22995], **kwargs_22998)


# Call to add_newdoc(...): (line 7494)
# Processing the call arguments (line 7494)
str_23001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7494, 11), 'str', 'numpy.core.numerictypes')
str_23002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7494, 38), 'str', 'generic')

# Obtaining an instance of the builtin type 'tuple' (line 7494)
tuple_23003 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7494, 50), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 7494)
# Adding element type (line 7494)
str_23004 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7494, 50), 'str', 'take')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7494, 50), tuple_23003, str_23004)
# Adding element type (line 7494)
str_23005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7506, (-1)), 'str', '\n    Not implemented (virtual attribute)\n\n    Class generic exists solely to derive numpy scalars from, and possesses,\n    albeit unimplemented, all the attributes of the ndarray class\n    so as to provide a uniform API.\n\n    See Also\n    --------\n    The corresponding attribute of the derived class of interest.\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7494, 50), tuple_23003, str_23005)

# Processing the call keyword arguments (line 7494)
kwargs_23006 = {}
# Getting the type of 'add_newdoc' (line 7494)
add_newdoc_23000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7494, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 7494)
add_newdoc_call_result_23007 = invoke(stypy.reporting.localization.Localization(__file__, 7494, 0), add_newdoc_23000, *[str_23001, str_23002, tuple_23003], **kwargs_23006)


# Call to add_newdoc(...): (line 7508)
# Processing the call arguments (line 7508)
str_23009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7508, 11), 'str', 'numpy.core.numerictypes')
str_23010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7508, 38), 'str', 'generic')

# Obtaining an instance of the builtin type 'tuple' (line 7508)
tuple_23011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7508, 50), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 7508)
# Adding element type (line 7508)
str_23012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7508, 50), 'str', 'tofile')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7508, 50), tuple_23011, str_23012)
# Adding element type (line 7508)
str_23013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7520, (-1)), 'str', '\n    Not implemented (virtual attribute)\n\n    Class generic exists solely to derive numpy scalars from, and possesses,\n    albeit unimplemented, all the attributes of the ndarray class\n    so as to provide a uniform API.\n\n    See Also\n    --------\n    The corresponding attribute of the derived class of interest.\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7508, 50), tuple_23011, str_23013)

# Processing the call keyword arguments (line 7508)
kwargs_23014 = {}
# Getting the type of 'add_newdoc' (line 7508)
add_newdoc_23008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7508, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 7508)
add_newdoc_call_result_23015 = invoke(stypy.reporting.localization.Localization(__file__, 7508, 0), add_newdoc_23008, *[str_23009, str_23010, tuple_23011], **kwargs_23014)


# Call to add_newdoc(...): (line 7522)
# Processing the call arguments (line 7522)
str_23017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7522, 11), 'str', 'numpy.core.numerictypes')
str_23018 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7522, 38), 'str', 'generic')

# Obtaining an instance of the builtin type 'tuple' (line 7522)
tuple_23019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7522, 50), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 7522)
# Adding element type (line 7522)
str_23020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7522, 50), 'str', 'tolist')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7522, 50), tuple_23019, str_23020)
# Adding element type (line 7522)
str_23021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7534, (-1)), 'str', '\n    Not implemented (virtual attribute)\n\n    Class generic exists solely to derive numpy scalars from, and possesses,\n    albeit unimplemented, all the attributes of the ndarray class\n    so as to provide a uniform API.\n\n    See Also\n    --------\n    The corresponding attribute of the derived class of interest.\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7522, 50), tuple_23019, str_23021)

# Processing the call keyword arguments (line 7522)
kwargs_23022 = {}
# Getting the type of 'add_newdoc' (line 7522)
add_newdoc_23016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7522, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 7522)
add_newdoc_call_result_23023 = invoke(stypy.reporting.localization.Localization(__file__, 7522, 0), add_newdoc_23016, *[str_23017, str_23018, tuple_23019], **kwargs_23022)


# Call to add_newdoc(...): (line 7536)
# Processing the call arguments (line 7536)
str_23025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7536, 11), 'str', 'numpy.core.numerictypes')
str_23026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7536, 38), 'str', 'generic')

# Obtaining an instance of the builtin type 'tuple' (line 7536)
tuple_23027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7536, 50), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 7536)
# Adding element type (line 7536)
str_23028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7536, 50), 'str', 'tostring')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7536, 50), tuple_23027, str_23028)
# Adding element type (line 7536)
str_23029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7548, (-1)), 'str', '\n    Not implemented (virtual attribute)\n\n    Class generic exists solely to derive numpy scalars from, and possesses,\n    albeit unimplemented, all the attributes of the ndarray class\n    so as to provide a uniform API.\n\n    See Also\n    --------\n    The corresponding attribute of the derived class of interest.\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7536, 50), tuple_23027, str_23029)

# Processing the call keyword arguments (line 7536)
kwargs_23030 = {}
# Getting the type of 'add_newdoc' (line 7536)
add_newdoc_23024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7536, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 7536)
add_newdoc_call_result_23031 = invoke(stypy.reporting.localization.Localization(__file__, 7536, 0), add_newdoc_23024, *[str_23025, str_23026, tuple_23027], **kwargs_23030)


# Call to add_newdoc(...): (line 7550)
# Processing the call arguments (line 7550)
str_23033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7550, 11), 'str', 'numpy.core.numerictypes')
str_23034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7550, 38), 'str', 'generic')

# Obtaining an instance of the builtin type 'tuple' (line 7550)
tuple_23035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7550, 50), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 7550)
# Adding element type (line 7550)
str_23036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7550, 50), 'str', 'trace')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7550, 50), tuple_23035, str_23036)
# Adding element type (line 7550)
str_23037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7562, (-1)), 'str', '\n    Not implemented (virtual attribute)\n\n    Class generic exists solely to derive numpy scalars from, and possesses,\n    albeit unimplemented, all the attributes of the ndarray class\n    so as to provide a uniform API.\n\n    See Also\n    --------\n    The corresponding attribute of the derived class of interest.\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7550, 50), tuple_23035, str_23037)

# Processing the call keyword arguments (line 7550)
kwargs_23038 = {}
# Getting the type of 'add_newdoc' (line 7550)
add_newdoc_23032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7550, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 7550)
add_newdoc_call_result_23039 = invoke(stypy.reporting.localization.Localization(__file__, 7550, 0), add_newdoc_23032, *[str_23033, str_23034, tuple_23035], **kwargs_23038)


# Call to add_newdoc(...): (line 7564)
# Processing the call arguments (line 7564)
str_23041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7564, 11), 'str', 'numpy.core.numerictypes')
str_23042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7564, 38), 'str', 'generic')

# Obtaining an instance of the builtin type 'tuple' (line 7564)
tuple_23043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7564, 50), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 7564)
# Adding element type (line 7564)
str_23044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7564, 50), 'str', 'transpose')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7564, 50), tuple_23043, str_23044)
# Adding element type (line 7564)
str_23045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7576, (-1)), 'str', '\n    Not implemented (virtual attribute)\n\n    Class generic exists solely to derive numpy scalars from, and possesses,\n    albeit unimplemented, all the attributes of the ndarray class\n    so as to provide a uniform API.\n\n    See Also\n    --------\n    The corresponding attribute of the derived class of interest.\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7564, 50), tuple_23043, str_23045)

# Processing the call keyword arguments (line 7564)
kwargs_23046 = {}
# Getting the type of 'add_newdoc' (line 7564)
add_newdoc_23040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7564, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 7564)
add_newdoc_call_result_23047 = invoke(stypy.reporting.localization.Localization(__file__, 7564, 0), add_newdoc_23040, *[str_23041, str_23042, tuple_23043], **kwargs_23046)


# Call to add_newdoc(...): (line 7578)
# Processing the call arguments (line 7578)
str_23049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7578, 11), 'str', 'numpy.core.numerictypes')
str_23050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7578, 38), 'str', 'generic')

# Obtaining an instance of the builtin type 'tuple' (line 7578)
tuple_23051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7578, 50), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 7578)
# Adding element type (line 7578)
str_23052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7578, 50), 'str', 'var')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7578, 50), tuple_23051, str_23052)
# Adding element type (line 7578)
str_23053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7590, (-1)), 'str', '\n    Not implemented (virtual attribute)\n\n    Class generic exists solely to derive numpy scalars from, and possesses,\n    albeit unimplemented, all the attributes of the ndarray class\n    so as to provide a uniform API.\n\n    See Also\n    --------\n    The corresponding attribute of the derived class of interest.\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7578, 50), tuple_23051, str_23053)

# Processing the call keyword arguments (line 7578)
kwargs_23054 = {}
# Getting the type of 'add_newdoc' (line 7578)
add_newdoc_23048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7578, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 7578)
add_newdoc_call_result_23055 = invoke(stypy.reporting.localization.Localization(__file__, 7578, 0), add_newdoc_23048, *[str_23049, str_23050, tuple_23051], **kwargs_23054)


# Call to add_newdoc(...): (line 7592)
# Processing the call arguments (line 7592)
str_23057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7592, 11), 'str', 'numpy.core.numerictypes')
str_23058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7592, 38), 'str', 'generic')

# Obtaining an instance of the builtin type 'tuple' (line 7592)
tuple_23059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7592, 50), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 7592)
# Adding element type (line 7592)
str_23060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7592, 50), 'str', 'view')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7592, 50), tuple_23059, str_23060)
# Adding element type (line 7592)
str_23061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7604, (-1)), 'str', '\n    Not implemented (virtual attribute)\n\n    Class generic exists solely to derive numpy scalars from, and possesses,\n    albeit unimplemented, all the attributes of the ndarray class\n    so as to provide a uniform API.\n\n    See Also\n    --------\n    The corresponding attribute of the derived class of interest.\n\n    ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7592, 50), tuple_23059, str_23061)

# Processing the call keyword arguments (line 7592)
kwargs_23062 = {}
# Getting the type of 'add_newdoc' (line 7592)
add_newdoc_23056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7592, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 7592)
add_newdoc_call_result_23063 = invoke(stypy.reporting.localization.Localization(__file__, 7592, 0), add_newdoc_23056, *[str_23057, str_23058, tuple_23059], **kwargs_23062)


# Call to add_newdoc(...): (line 7613)
# Processing the call arguments (line 7613)
str_23065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7613, 11), 'str', 'numpy.core.numerictypes')
str_23066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7613, 38), 'str', 'bool_')
str_23067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7614, 4), 'str', "Numpy's Boolean type.  Character code: ``?``.  Alias: bool8")
# Processing the call keyword arguments (line 7613)
kwargs_23068 = {}
# Getting the type of 'add_newdoc' (line 7613)
add_newdoc_23064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7613, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 7613)
add_newdoc_call_result_23069 = invoke(stypy.reporting.localization.Localization(__file__, 7613, 0), add_newdoc_23064, *[str_23065, str_23066, str_23067], **kwargs_23068)


# Call to add_newdoc(...): (line 7616)
# Processing the call arguments (line 7616)
str_23071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7616, 11), 'str', 'numpy.core.numerictypes')
str_23072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7616, 38), 'str', 'complex64')
str_23073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7620, (-1)), 'str', "\n    Complex number type composed of two 32 bit floats. Character code: 'F'.\n\n    ")
# Processing the call keyword arguments (line 7616)
kwargs_23074 = {}
# Getting the type of 'add_newdoc' (line 7616)
add_newdoc_23070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7616, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 7616)
add_newdoc_call_result_23075 = invoke(stypy.reporting.localization.Localization(__file__, 7616, 0), add_newdoc_23070, *[str_23071, str_23072, str_23073], **kwargs_23074)


# Call to add_newdoc(...): (line 7622)
# Processing the call arguments (line 7622)
str_23077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7622, 11), 'str', 'numpy.core.numerictypes')
str_23078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7622, 38), 'str', 'complex128')
str_23079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7627, (-1)), 'str', "\n    Complex number type composed of two 64 bit floats. Character code: 'D'.\n    Python complex compatible.\n\n    ")
# Processing the call keyword arguments (line 7622)
kwargs_23080 = {}
# Getting the type of 'add_newdoc' (line 7622)
add_newdoc_23076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7622, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 7622)
add_newdoc_call_result_23081 = invoke(stypy.reporting.localization.Localization(__file__, 7622, 0), add_newdoc_23076, *[str_23077, str_23078, str_23079], **kwargs_23080)


# Call to add_newdoc(...): (line 7629)
# Processing the call arguments (line 7629)
str_23083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7629, 11), 'str', 'numpy.core.numerictypes')
str_23084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7629, 38), 'str', 'complex256')
str_23085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7633, (-1)), 'str', "\n    Complex number type composed of two 128-bit floats. Character code: 'G'.\n\n    ")
# Processing the call keyword arguments (line 7629)
kwargs_23086 = {}
# Getting the type of 'add_newdoc' (line 7629)
add_newdoc_23082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7629, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 7629)
add_newdoc_call_result_23087 = invoke(stypy.reporting.localization.Localization(__file__, 7629, 0), add_newdoc_23082, *[str_23083, str_23084, str_23085], **kwargs_23086)


# Call to add_newdoc(...): (line 7635)
# Processing the call arguments (line 7635)
str_23089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7635, 11), 'str', 'numpy.core.numerictypes')
str_23090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7635, 38), 'str', 'float32')
str_23091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7639, (-1)), 'str', "\n    32-bit floating-point number. Character code 'f'. C float compatible.\n\n    ")
# Processing the call keyword arguments (line 7635)
kwargs_23092 = {}
# Getting the type of 'add_newdoc' (line 7635)
add_newdoc_23088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7635, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 7635)
add_newdoc_call_result_23093 = invoke(stypy.reporting.localization.Localization(__file__, 7635, 0), add_newdoc_23088, *[str_23089, str_23090, str_23091], **kwargs_23092)


# Call to add_newdoc(...): (line 7641)
# Processing the call arguments (line 7641)
str_23095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7641, 11), 'str', 'numpy.core.numerictypes')
str_23096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7641, 38), 'str', 'float64')
str_23097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7645, (-1)), 'str', "\n    64-bit floating-point number. Character code 'd'. Python float compatible.\n\n    ")
# Processing the call keyword arguments (line 7641)
kwargs_23098 = {}
# Getting the type of 'add_newdoc' (line 7641)
add_newdoc_23094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7641, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 7641)
add_newdoc_call_result_23099 = invoke(stypy.reporting.localization.Localization(__file__, 7641, 0), add_newdoc_23094, *[str_23095, str_23096, str_23097], **kwargs_23098)


# Call to add_newdoc(...): (line 7647)
# Processing the call arguments (line 7647)
str_23101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7647, 11), 'str', 'numpy.core.numerictypes')
str_23102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7647, 38), 'str', 'float96')
str_23103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7649, (-1)), 'str', '\n    ')
# Processing the call keyword arguments (line 7647)
kwargs_23104 = {}
# Getting the type of 'add_newdoc' (line 7647)
add_newdoc_23100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7647, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 7647)
add_newdoc_call_result_23105 = invoke(stypy.reporting.localization.Localization(__file__, 7647, 0), add_newdoc_23100, *[str_23101, str_23102, str_23103], **kwargs_23104)


# Call to add_newdoc(...): (line 7651)
# Processing the call arguments (line 7651)
str_23107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7651, 11), 'str', 'numpy.core.numerictypes')
str_23108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7651, 38), 'str', 'float128')
str_23109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7656, (-1)), 'str', "\n    128-bit floating-point number. Character code: 'g'. C long float\n    compatible.\n\n    ")
# Processing the call keyword arguments (line 7651)
kwargs_23110 = {}
# Getting the type of 'add_newdoc' (line 7651)
add_newdoc_23106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7651, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 7651)
add_newdoc_call_result_23111 = invoke(stypy.reporting.localization.Localization(__file__, 7651, 0), add_newdoc_23106, *[str_23107, str_23108, str_23109], **kwargs_23110)


# Call to add_newdoc(...): (line 7658)
# Processing the call arguments (line 7658)
str_23113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7658, 11), 'str', 'numpy.core.numerictypes')
str_23114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7658, 38), 'str', 'int8')
str_23115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7659, 4), 'str', '8-bit integer. Character code ``b``. C char compatible.')
# Processing the call keyword arguments (line 7658)
kwargs_23116 = {}
# Getting the type of 'add_newdoc' (line 7658)
add_newdoc_23112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7658, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 7658)
add_newdoc_call_result_23117 = invoke(stypy.reporting.localization.Localization(__file__, 7658, 0), add_newdoc_23112, *[str_23113, str_23114, str_23115], **kwargs_23116)


# Call to add_newdoc(...): (line 7661)
# Processing the call arguments (line 7661)
str_23119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7661, 11), 'str', 'numpy.core.numerictypes')
str_23120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7661, 38), 'str', 'int16')
str_23121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7662, 4), 'str', '16-bit integer. Character code ``h``. C short compatible.')
# Processing the call keyword arguments (line 7661)
kwargs_23122 = {}
# Getting the type of 'add_newdoc' (line 7661)
add_newdoc_23118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7661, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 7661)
add_newdoc_call_result_23123 = invoke(stypy.reporting.localization.Localization(__file__, 7661, 0), add_newdoc_23118, *[str_23119, str_23120, str_23121], **kwargs_23122)


# Call to add_newdoc(...): (line 7664)
# Processing the call arguments (line 7664)
str_23125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7664, 11), 'str', 'numpy.core.numerictypes')
str_23126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7664, 38), 'str', 'int32')
str_23127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7665, 4), 'str', "32-bit integer. Character code 'i'. C int compatible.")
# Processing the call keyword arguments (line 7664)
kwargs_23128 = {}
# Getting the type of 'add_newdoc' (line 7664)
add_newdoc_23124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7664, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 7664)
add_newdoc_call_result_23129 = invoke(stypy.reporting.localization.Localization(__file__, 7664, 0), add_newdoc_23124, *[str_23125, str_23126, str_23127], **kwargs_23128)


# Call to add_newdoc(...): (line 7667)
# Processing the call arguments (line 7667)
str_23131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7667, 11), 'str', 'numpy.core.numerictypes')
str_23132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7667, 38), 'str', 'int64')
str_23133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7668, 4), 'str', "64-bit integer. Character code 'l'. Python int compatible.")
# Processing the call keyword arguments (line 7667)
kwargs_23134 = {}
# Getting the type of 'add_newdoc' (line 7667)
add_newdoc_23130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7667, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 7667)
add_newdoc_call_result_23135 = invoke(stypy.reporting.localization.Localization(__file__, 7667, 0), add_newdoc_23130, *[str_23131, str_23132, str_23133], **kwargs_23134)


# Call to add_newdoc(...): (line 7670)
# Processing the call arguments (line 7670)
str_23137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7670, 11), 'str', 'numpy.core.numerictypes')
str_23138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7670, 38), 'str', 'object_')
str_23139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7671, 4), 'str', "Any Python object.  Character code: 'O'.")
# Processing the call keyword arguments (line 7670)
kwargs_23140 = {}
# Getting the type of 'add_newdoc' (line 7670)
add_newdoc_23136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7670, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 7670)
add_newdoc_call_result_23141 = invoke(stypy.reporting.localization.Localization(__file__, 7670, 0), add_newdoc_23136, *[str_23137, str_23138, str_23139], **kwargs_23140)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
