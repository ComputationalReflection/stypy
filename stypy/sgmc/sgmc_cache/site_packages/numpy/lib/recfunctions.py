
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Collection of utilities to manipulate structured arrays.
3: 
4: Most of these functions were initially implemented by John Hunter for
5: matplotlib.  They have been rewritten and extended for convenience.
6: 
7: '''
8: from __future__ import division, absolute_import, print_function
9: 
10: import sys
11: import itertools
12: import numpy as np
13: import numpy.ma as ma
14: from numpy import ndarray, recarray
15: from numpy.ma import MaskedArray
16: from numpy.ma.mrecords import MaskedRecords
17: from numpy.lib._iotools import _is_string_like
18: from numpy.compat import basestring
19: 
20: if sys.version_info[0] < 3:
21:     from future_builtins import zip
22: 
23: _check_fill_value = np.ma.core._check_fill_value
24: 
25: 
26: __all__ = [
27:     'append_fields', 'drop_fields', 'find_duplicates',
28:     'get_fieldstructure', 'join_by', 'merge_arrays',
29:     'rec_append_fields', 'rec_drop_fields', 'rec_join',
30:     'recursive_fill_fields', 'rename_fields', 'stack_arrays',
31:     ]
32: 
33: 
34: def recursive_fill_fields(input, output):
35:     '''
36:     Fills fields from output with fields from input,
37:     with support for nested structures.
38: 
39:     Parameters
40:     ----------
41:     input : ndarray
42:         Input array.
43:     output : ndarray
44:         Output array.
45: 
46:     Notes
47:     -----
48:     * `output` should be at least the same size as `input`
49: 
50:     Examples
51:     --------
52:     >>> from numpy.lib import recfunctions as rfn
53:     >>> a = np.array([(1, 10.), (2, 20.)], dtype=[('A', int), ('B', float)])
54:     >>> b = np.zeros((3,), dtype=a.dtype)
55:     >>> rfn.recursive_fill_fields(a, b)
56:     array([(1, 10.0), (2, 20.0), (0, 0.0)],
57:           dtype=[('A', '<i4'), ('B', '<f8')])
58: 
59:     '''
60:     newdtype = output.dtype
61:     for field in newdtype.names:
62:         try:
63:             current = input[field]
64:         except ValueError:
65:             continue
66:         if current.dtype.names:
67:             recursive_fill_fields(current, output[field])
68:         else:
69:             output[field][:len(current)] = current
70:     return output
71: 
72: 
73: def get_names(adtype):
74:     '''
75:     Returns the field names of the input datatype as a tuple.
76: 
77:     Parameters
78:     ----------
79:     adtype : dtype
80:         Input datatype
81: 
82:     Examples
83:     --------
84:     >>> from numpy.lib import recfunctions as rfn
85:     >>> rfn.get_names(np.empty((1,), dtype=int)) is None
86:     True
87:     >>> rfn.get_names(np.empty((1,), dtype=[('A',int), ('B', float)]))
88:     ('A', 'B')
89:     >>> adtype = np.dtype([('a', int), ('b', [('ba', int), ('bb', int)])])
90:     >>> rfn.get_names(adtype)
91:     ('a', ('b', ('ba', 'bb')))
92:     '''
93:     listnames = []
94:     names = adtype.names
95:     for name in names:
96:         current = adtype[name]
97:         if current.names:
98:             listnames.append((name, tuple(get_names(current))))
99:         else:
100:             listnames.append(name)
101:     return tuple(listnames) or None
102: 
103: 
104: def get_names_flat(adtype):
105:     '''
106:     Returns the field names of the input datatype as a tuple. Nested structure
107:     are flattend beforehand.
108: 
109:     Parameters
110:     ----------
111:     adtype : dtype
112:         Input datatype
113: 
114:     Examples
115:     --------
116:     >>> from numpy.lib import recfunctions as rfn
117:     >>> rfn.get_names_flat(np.empty((1,), dtype=int)) is None
118:     True
119:     >>> rfn.get_names_flat(np.empty((1,), dtype=[('A',int), ('B', float)]))
120:     ('A', 'B')
121:     >>> adtype = np.dtype([('a', int), ('b', [('ba', int), ('bb', int)])])
122:     >>> rfn.get_names_flat(adtype)
123:     ('a', 'b', 'ba', 'bb')
124:     '''
125:     listnames = []
126:     names = adtype.names
127:     for name in names:
128:         listnames.append(name)
129:         current = adtype[name]
130:         if current.names:
131:             listnames.extend(get_names_flat(current))
132:     return tuple(listnames) or None
133: 
134: 
135: def flatten_descr(ndtype):
136:     '''
137:     Flatten a structured data-type description.
138: 
139:     Examples
140:     --------
141:     >>> from numpy.lib import recfunctions as rfn
142:     >>> ndtype = np.dtype([('a', '<i4'), ('b', [('ba', '<f8'), ('bb', '<i4')])])
143:     >>> rfn.flatten_descr(ndtype)
144:     (('a', dtype('int32')), ('ba', dtype('float64')), ('bb', dtype('int32')))
145: 
146:     '''
147:     names = ndtype.names
148:     if names is None:
149:         return ndtype.descr
150:     else:
151:         descr = []
152:         for field in names:
153:             (typ, _) = ndtype.fields[field]
154:             if typ.names:
155:                 descr.extend(flatten_descr(typ))
156:             else:
157:                 descr.append((field, typ))
158:         return tuple(descr)
159: 
160: 
161: def zip_descr(seqarrays, flatten=False):
162:     '''
163:     Combine the dtype description of a series of arrays.
164: 
165:     Parameters
166:     ----------
167:     seqarrays : sequence of arrays
168:         Sequence of arrays
169:     flatten : {boolean}, optional
170:         Whether to collapse nested descriptions.
171:     '''
172:     newdtype = []
173:     if flatten:
174:         for a in seqarrays:
175:             newdtype.extend(flatten_descr(a.dtype))
176:     else:
177:         for a in seqarrays:
178:             current = a.dtype
179:             names = current.names or ()
180:             if len(names) > 1:
181:                 newdtype.append(('', current.descr))
182:             else:
183:                 newdtype.extend(current.descr)
184:     return np.dtype(newdtype).descr
185: 
186: 
187: def get_fieldstructure(adtype, lastname=None, parents=None,):
188:     '''
189:     Returns a dictionary with fields indexing lists of their parent fields.
190: 
191:     This function is used to simplify access to fields nested in other fields.
192: 
193:     Parameters
194:     ----------
195:     adtype : np.dtype
196:         Input datatype
197:     lastname : optional
198:         Last processed field name (used internally during recursion).
199:     parents : dictionary
200:         Dictionary of parent fields (used interbally during recursion).
201: 
202:     Examples
203:     --------
204:     >>> from numpy.lib import recfunctions as rfn
205:     >>> ndtype =  np.dtype([('A', int),
206:     ...                     ('B', [('BA', int),
207:     ...                            ('BB', [('BBA', int), ('BBB', int)])])])
208:     >>> rfn.get_fieldstructure(ndtype)
209:     ... # XXX: possible regression, order of BBA and BBB is swapped
210:     {'A': [], 'B': [], 'BA': ['B'], 'BB': ['B'], 'BBA': ['B', 'BB'], 'BBB': ['B', 'BB']}
211: 
212:     '''
213:     if parents is None:
214:         parents = {}
215:     names = adtype.names
216:     for name in names:
217:         current = adtype[name]
218:         if current.names:
219:             if lastname:
220:                 parents[name] = [lastname, ]
221:             else:
222:                 parents[name] = []
223:             parents.update(get_fieldstructure(current, name, parents))
224:         else:
225:             lastparent = [_ for _ in (parents.get(lastname, []) or [])]
226:             if lastparent:
227:                 lastparent.append(lastname)
228:             elif lastname:
229:                 lastparent = [lastname, ]
230:             parents[name] = lastparent or []
231:     return parents or None
232: 
233: 
234: def _izip_fields_flat(iterable):
235:     '''
236:     Returns an iterator of concatenated fields from a sequence of arrays,
237:     collapsing any nested structure.
238: 
239:     '''
240:     for element in iterable:
241:         if isinstance(element, np.void):
242:             for f in _izip_fields_flat(tuple(element)):
243:                 yield f
244:         else:
245:             yield element
246: 
247: 
248: def _izip_fields(iterable):
249:     '''
250:     Returns an iterator of concatenated fields from a sequence of arrays.
251: 
252:     '''
253:     for element in iterable:
254:         if (hasattr(element, '__iter__') and
255:                 not isinstance(element, basestring)):
256:             for f in _izip_fields(element):
257:                 yield f
258:         elif isinstance(element, np.void) and len(tuple(element)) == 1:
259:             for f in _izip_fields(element):
260:                 yield f
261:         else:
262:             yield element
263: 
264: 
265: def izip_records(seqarrays, fill_value=None, flatten=True):
266:     '''
267:     Returns an iterator of concatenated items from a sequence of arrays.
268: 
269:     Parameters
270:     ----------
271:     seqarrays : sequence of arrays
272:         Sequence of arrays.
273:     fill_value : {None, integer}
274:         Value used to pad shorter iterables.
275:     flatten : {True, False},
276:         Whether to
277:     '''
278:     # OK, that's a complete ripoff from Python2.6 itertools.izip_longest
279:     def sentinel(counter=([fill_value] * (len(seqarrays) - 1)).pop):
280:         "Yields the fill_value or raises IndexError"
281:         yield counter()
282:     #
283:     fillers = itertools.repeat(fill_value)
284:     iters = [itertools.chain(it, sentinel(), fillers) for it in seqarrays]
285:     # Should we flatten the items, or just use a nested approach
286:     if flatten:
287:         zipfunc = _izip_fields_flat
288:     else:
289:         zipfunc = _izip_fields
290:     #
291:     try:
292:         for tup in zip(*iters):
293:             yield tuple(zipfunc(tup))
294:     except IndexError:
295:         pass
296: 
297: 
298: def _fix_output(output, usemask=True, asrecarray=False):
299:     '''
300:     Private function: return a recarray, a ndarray, a MaskedArray
301:     or a MaskedRecords depending on the input parameters
302:     '''
303:     if not isinstance(output, MaskedArray):
304:         usemask = False
305:     if usemask:
306:         if asrecarray:
307:             output = output.view(MaskedRecords)
308:     else:
309:         output = ma.filled(output)
310:         if asrecarray:
311:             output = output.view(recarray)
312:     return output
313: 
314: 
315: def _fix_defaults(output, defaults=None):
316:     '''
317:     Update the fill_value and masked data of `output`
318:     from the default given in a dictionary defaults.
319:     '''
320:     names = output.dtype.names
321:     (data, mask, fill_value) = (output.data, output.mask, output.fill_value)
322:     for (k, v) in (defaults or {}).items():
323:         if k in names:
324:             fill_value[k] = v
325:             data[k][mask[k]] = v
326:     return output
327: 
328: 
329: def merge_arrays(seqarrays, fill_value=-1, flatten=False,
330:                  usemask=False, asrecarray=False):
331:     '''
332:     Merge arrays field by field.
333: 
334:     Parameters
335:     ----------
336:     seqarrays : sequence of ndarrays
337:         Sequence of arrays
338:     fill_value : {float}, optional
339:         Filling value used to pad missing data on the shorter arrays.
340:     flatten : {False, True}, optional
341:         Whether to collapse nested fields.
342:     usemask : {False, True}, optional
343:         Whether to return a masked array or not.
344:     asrecarray : {False, True}, optional
345:         Whether to return a recarray (MaskedRecords) or not.
346: 
347:     Examples
348:     --------
349:     >>> from numpy.lib import recfunctions as rfn
350:     >>> rfn.merge_arrays((np.array([1, 2]), np.array([10., 20., 30.])))
351:     masked_array(data = [(1, 10.0) (2, 20.0) (--, 30.0)],
352:                  mask = [(False, False) (False, False) (True, False)],
353:            fill_value = (999999, 1e+20),
354:                 dtype = [('f0', '<i4'), ('f1', '<f8')])
355: 
356:     >>> rfn.merge_arrays((np.array([1, 2]), np.array([10., 20., 30.])),
357:     ...              usemask=False)
358:     array([(1, 10.0), (2, 20.0), (-1, 30.0)],
359:           dtype=[('f0', '<i4'), ('f1', '<f8')])
360:     >>> rfn.merge_arrays((np.array([1, 2]).view([('a', int)]),
361:     ...               np.array([10., 20., 30.])),
362:     ...              usemask=False, asrecarray=True)
363:     rec.array([(1, 10.0), (2, 20.0), (-1, 30.0)],
364:               dtype=[('a', '<i4'), ('f1', '<f8')])
365: 
366:     Notes
367:     -----
368:     * Without a mask, the missing value will be filled with something,
369:     * depending on what its corresponding type:
370:             -1      for integers
371:             -1.0    for floating point numbers
372:             '-'     for characters
373:             '-1'    for strings
374:             True    for boolean values
375:     * XXX: I just obtained these values empirically
376:     '''
377:     # Only one item in the input sequence ?
378:     if (len(seqarrays) == 1):
379:         seqarrays = np.asanyarray(seqarrays[0])
380:     # Do we have a single ndarray as input ?
381:     if isinstance(seqarrays, (ndarray, np.void)):
382:         seqdtype = seqarrays.dtype
383:         if (not flatten) or \
384:            (zip_descr((seqarrays,), flatten=True) == seqdtype.descr):
385:             # Minimal processing needed: just make sure everythng's a-ok
386:             seqarrays = seqarrays.ravel()
387:             # Make sure we have named fields
388:             if not seqdtype.names:
389:                 seqdtype = [('', seqdtype)]
390:             # Find what type of array we must return
391:             if usemask:
392:                 if asrecarray:
393:                     seqtype = MaskedRecords
394:                 else:
395:                     seqtype = MaskedArray
396:             elif asrecarray:
397:                 seqtype = recarray
398:             else:
399:                 seqtype = ndarray
400:             return seqarrays.view(dtype=seqdtype, type=seqtype)
401:         else:
402:             seqarrays = (seqarrays,)
403:     else:
404:         # Make sure we have arrays in the input sequence
405:         seqarrays = [np.asanyarray(_m) for _m in seqarrays]
406:     # Find the sizes of the inputs and their maximum
407:     sizes = tuple(a.size for a in seqarrays)
408:     maxlength = max(sizes)
409:     # Get the dtype of the output (flattening if needed)
410:     newdtype = zip_descr(seqarrays, flatten=flatten)
411:     # Initialize the sequences for data and mask
412:     seqdata = []
413:     seqmask = []
414:     # If we expect some kind of MaskedArray, make a special loop.
415:     if usemask:
416:         for (a, n) in zip(seqarrays, sizes):
417:             nbmissing = (maxlength - n)
418:             # Get the data and mask
419:             data = a.ravel().__array__()
420:             mask = ma.getmaskarray(a).ravel()
421:             # Get the filling value (if needed)
422:             if nbmissing:
423:                 fval = _check_fill_value(fill_value, a.dtype)
424:                 if isinstance(fval, (ndarray, np.void)):
425:                     if len(fval.dtype) == 1:
426:                         fval = fval.item()[0]
427:                         fmsk = True
428:                     else:
429:                         fval = np.array(fval, dtype=a.dtype, ndmin=1)
430:                         fmsk = np.ones((1,), dtype=mask.dtype)
431:             else:
432:                 fval = None
433:                 fmsk = True
434:             # Store an iterator padding the input to the expected length
435:             seqdata.append(itertools.chain(data, [fval] * nbmissing))
436:             seqmask.append(itertools.chain(mask, [fmsk] * nbmissing))
437:         # Create an iterator for the data
438:         data = tuple(izip_records(seqdata, flatten=flatten))
439:         output = ma.array(np.fromiter(data, dtype=newdtype, count=maxlength),
440:                           mask=list(izip_records(seqmask, flatten=flatten)))
441:         if asrecarray:
442:             output = output.view(MaskedRecords)
443:     else:
444:         # Same as before, without the mask we don't need...
445:         for (a, n) in zip(seqarrays, sizes):
446:             nbmissing = (maxlength - n)
447:             data = a.ravel().__array__()
448:             if nbmissing:
449:                 fval = _check_fill_value(fill_value, a.dtype)
450:                 if isinstance(fval, (ndarray, np.void)):
451:                     if len(fval.dtype) == 1:
452:                         fval = fval.item()[0]
453:                     else:
454:                         fval = np.array(fval, dtype=a.dtype, ndmin=1)
455:             else:
456:                 fval = None
457:             seqdata.append(itertools.chain(data, [fval] * nbmissing))
458:         output = np.fromiter(tuple(izip_records(seqdata, flatten=flatten)),
459:                              dtype=newdtype, count=maxlength)
460:         if asrecarray:
461:             output = output.view(recarray)
462:     # And we're done...
463:     return output
464: 
465: 
466: def drop_fields(base, drop_names, usemask=True, asrecarray=False):
467:     '''
468:     Return a new array with fields in `drop_names` dropped.
469: 
470:     Nested fields are supported.
471: 
472:     Parameters
473:     ----------
474:     base : array
475:         Input array
476:     drop_names : string or sequence
477:         String or sequence of strings corresponding to the names of the
478:         fields to drop.
479:     usemask : {False, True}, optional
480:         Whether to return a masked array or not.
481:     asrecarray : string or sequence, optional
482:         Whether to return a recarray or a mrecarray (`asrecarray=True`) or
483:         a plain ndarray or masked array with flexible dtype. The default
484:         is False.
485: 
486:     Examples
487:     --------
488:     >>> from numpy.lib import recfunctions as rfn
489:     >>> a = np.array([(1, (2, 3.0)), (4, (5, 6.0))],
490:     ...   dtype=[('a', int), ('b', [('ba', float), ('bb', int)])])
491:     >>> rfn.drop_fields(a, 'a')
492:     array([((2.0, 3),), ((5.0, 6),)],
493:           dtype=[('b', [('ba', '<f8'), ('bb', '<i4')])])
494:     >>> rfn.drop_fields(a, 'ba')
495:     array([(1, (3,)), (4, (6,))],
496:           dtype=[('a', '<i4'), ('b', [('bb', '<i4')])])
497:     >>> rfn.drop_fields(a, ['ba', 'bb'])
498:     array([(1,), (4,)],
499:           dtype=[('a', '<i4')])
500:     '''
501:     if _is_string_like(drop_names):
502:         drop_names = [drop_names, ]
503:     else:
504:         drop_names = set(drop_names)
505: 
506:     def _drop_descr(ndtype, drop_names):
507:         names = ndtype.names
508:         newdtype = []
509:         for name in names:
510:             current = ndtype[name]
511:             if name in drop_names:
512:                 continue
513:             if current.names:
514:                 descr = _drop_descr(current, drop_names)
515:                 if descr:
516:                     newdtype.append((name, descr))
517:             else:
518:                 newdtype.append((name, current))
519:         return newdtype
520: 
521:     newdtype = _drop_descr(base.dtype, drop_names)
522:     if not newdtype:
523:         return None
524: 
525:     output = np.empty(base.shape, dtype=newdtype)
526:     output = recursive_fill_fields(base, output)
527:     return _fix_output(output, usemask=usemask, asrecarray=asrecarray)
528: 
529: 
530: def rec_drop_fields(base, drop_names):
531:     '''
532:     Returns a new numpy.recarray with fields in `drop_names` dropped.
533:     '''
534:     return drop_fields(base, drop_names, usemask=False, asrecarray=True)
535: 
536: 
537: def rename_fields(base, namemapper):
538:     '''
539:     Rename the fields from a flexible-datatype ndarray or recarray.
540: 
541:     Nested fields are supported.
542: 
543:     Parameters
544:     ----------
545:     base : ndarray
546:         Input array whose fields must be modified.
547:     namemapper : dictionary
548:         Dictionary mapping old field names to their new version.
549: 
550:     Examples
551:     --------
552:     >>> from numpy.lib import recfunctions as rfn
553:     >>> a = np.array([(1, (2, [3.0, 30.])), (4, (5, [6.0, 60.]))],
554:     ...   dtype=[('a', int),('b', [('ba', float), ('bb', (float, 2))])])
555:     >>> rfn.rename_fields(a, {'a':'A', 'bb':'BB'})
556:     array([(1, (2.0, [3.0, 30.0])), (4, (5.0, [6.0, 60.0]))],
557:           dtype=[('A', '<i4'), ('b', [('ba', '<f8'), ('BB', '<f8', 2)])])
558: 
559:     '''
560:     def _recursive_rename_fields(ndtype, namemapper):
561:         newdtype = []
562:         for name in ndtype.names:
563:             newname = namemapper.get(name, name)
564:             current = ndtype[name]
565:             if current.names:
566:                 newdtype.append(
567:                     (newname, _recursive_rename_fields(current, namemapper))
568:                     )
569:             else:
570:                 newdtype.append((newname, current))
571:         return newdtype
572:     newdtype = _recursive_rename_fields(base.dtype, namemapper)
573:     return base.view(newdtype)
574: 
575: 
576: def append_fields(base, names, data, dtypes=None,
577:                   fill_value=-1, usemask=True, asrecarray=False):
578:     '''
579:     Add new fields to an existing array.
580: 
581:     The names of the fields are given with the `names` arguments,
582:     the corresponding values with the `data` arguments.
583:     If a single field is appended, `names`, `data` and `dtypes` do not have
584:     to be lists but just values.
585: 
586:     Parameters
587:     ----------
588:     base : array
589:         Input array to extend.
590:     names : string, sequence
591:         String or sequence of strings corresponding to the names
592:         of the new fields.
593:     data : array or sequence of arrays
594:         Array or sequence of arrays storing the fields to add to the base.
595:     dtypes : sequence of datatypes, optional
596:         Datatype or sequence of datatypes.
597:         If None, the datatypes are estimated from the `data`.
598:     fill_value : {float}, optional
599:         Filling value used to pad missing data on the shorter arrays.
600:     usemask : {False, True}, optional
601:         Whether to return a masked array or not.
602:     asrecarray : {False, True}, optional
603:         Whether to return a recarray (MaskedRecords) or not.
604: 
605:     '''
606:     # Check the names
607:     if isinstance(names, (tuple, list)):
608:         if len(names) != len(data):
609:             msg = "The number of arrays does not match the number of names"
610:             raise ValueError(msg)
611:     elif isinstance(names, basestring):
612:         names = [names, ]
613:         data = [data, ]
614:     #
615:     if dtypes is None:
616:         data = [np.array(a, copy=False, subok=True) for a in data]
617:         data = [a.view([(name, a.dtype)]) for (name, a) in zip(names, data)]
618:     else:
619:         if not isinstance(dtypes, (tuple, list)):
620:             dtypes = [dtypes, ]
621:         if len(data) != len(dtypes):
622:             if len(dtypes) == 1:
623:                 dtypes = dtypes * len(data)
624:             else:
625:                 msg = "The dtypes argument must be None, a dtype, or a list."
626:                 raise ValueError(msg)
627:         data = [np.array(a, copy=False, subok=True, dtype=d).view([(n, d)])
628:                 for (a, n, d) in zip(data, names, dtypes)]
629:     #
630:     base = merge_arrays(base, usemask=usemask, fill_value=fill_value)
631:     if len(data) > 1:
632:         data = merge_arrays(data, flatten=True, usemask=usemask,
633:                             fill_value=fill_value)
634:     else:
635:         data = data.pop()
636:     #
637:     output = ma.masked_all(max(len(base), len(data)),
638:                            dtype=base.dtype.descr + data.dtype.descr)
639:     output = recursive_fill_fields(base, output)
640:     output = recursive_fill_fields(data, output)
641:     #
642:     return _fix_output(output, usemask=usemask, asrecarray=asrecarray)
643: 
644: 
645: def rec_append_fields(base, names, data, dtypes=None):
646:     '''
647:     Add new fields to an existing array.
648: 
649:     The names of the fields are given with the `names` arguments,
650:     the corresponding values with the `data` arguments.
651:     If a single field is appended, `names`, `data` and `dtypes` do not have
652:     to be lists but just values.
653: 
654:     Parameters
655:     ----------
656:     base : array
657:         Input array to extend.
658:     names : string, sequence
659:         String or sequence of strings corresponding to the names
660:         of the new fields.
661:     data : array or sequence of arrays
662:         Array or sequence of arrays storing the fields to add to the base.
663:     dtypes : sequence of datatypes, optional
664:         Datatype or sequence of datatypes.
665:         If None, the datatypes are estimated from the `data`.
666: 
667:     See Also
668:     --------
669:     append_fields
670: 
671:     Returns
672:     -------
673:     appended_array : np.recarray
674:     '''
675:     return append_fields(base, names, data=data, dtypes=dtypes,
676:                          asrecarray=True, usemask=False)
677: 
678: 
679: def stack_arrays(arrays, defaults=None, usemask=True, asrecarray=False,
680:                  autoconvert=False):
681:     '''
682:     Superposes arrays fields by fields
683: 
684:     Parameters
685:     ----------
686:     arrays : array or sequence
687:         Sequence of input arrays.
688:     defaults : dictionary, optional
689:         Dictionary mapping field names to the corresponding default values.
690:     usemask : {True, False}, optional
691:         Whether to return a MaskedArray (or MaskedRecords is
692:         `asrecarray==True`) or a ndarray.
693:     asrecarray : {False, True}, optional
694:         Whether to return a recarray (or MaskedRecords if `usemask==True`)
695:         or just a flexible-type ndarray.
696:     autoconvert : {False, True}, optional
697:         Whether automatically cast the type of the field to the maximum.
698: 
699:     Examples
700:     --------
701:     >>> from numpy.lib import recfunctions as rfn
702:     >>> x = np.array([1, 2,])
703:     >>> rfn.stack_arrays(x) is x
704:     True
705:     >>> z = np.array([('A', 1), ('B', 2)], dtype=[('A', '|S3'), ('B', float)])
706:     >>> zz = np.array([('a', 10., 100.), ('b', 20., 200.), ('c', 30., 300.)],
707:     ...   dtype=[('A', '|S3'), ('B', float), ('C', float)])
708:     >>> test = rfn.stack_arrays((z,zz))
709:     >>> test
710:     masked_array(data = [('A', 1.0, --) ('B', 2.0, --) ('a', 10.0, 100.0) ('b', 20.0, 200.0)
711:      ('c', 30.0, 300.0)],
712:                  mask = [(False, False, True) (False, False, True) (False, False, False)
713:      (False, False, False) (False, False, False)],
714:            fill_value = ('N/A', 1e+20, 1e+20),
715:                 dtype = [('A', '|S3'), ('B', '<f8'), ('C', '<f8')])
716: 
717:     '''
718:     if isinstance(arrays, ndarray):
719:         return arrays
720:     elif len(arrays) == 1:
721:         return arrays[0]
722:     seqarrays = [np.asanyarray(a).ravel() for a in arrays]
723:     nrecords = [len(a) for a in seqarrays]
724:     ndtype = [a.dtype for a in seqarrays]
725:     fldnames = [d.names for d in ndtype]
726:     #
727:     dtype_l = ndtype[0]
728:     newdescr = dtype_l.descr
729:     names = [_[0] for _ in newdescr]
730:     for dtype_n in ndtype[1:]:
731:         for descr in dtype_n.descr:
732:             name = descr[0] or ''
733:             if name not in names:
734:                 newdescr.append(descr)
735:                 names.append(name)
736:             else:
737:                 nameidx = names.index(name)
738:                 current_descr = newdescr[nameidx]
739:                 if autoconvert:
740:                     if np.dtype(descr[1]) > np.dtype(current_descr[-1]):
741:                         current_descr = list(current_descr)
742:                         current_descr[-1] = descr[1]
743:                         newdescr[nameidx] = tuple(current_descr)
744:                 elif descr[1] != current_descr[-1]:
745:                     raise TypeError("Incompatible type '%s' <> '%s'" %
746:                                     (dict(newdescr)[name], descr[1]))
747:     # Only one field: use concatenate
748:     if len(newdescr) == 1:
749:         output = ma.concatenate(seqarrays)
750:     else:
751:         #
752:         output = ma.masked_all((np.sum(nrecords),), newdescr)
753:         offset = np.cumsum(np.r_[0, nrecords])
754:         seen = []
755:         for (a, n, i, j) in zip(seqarrays, fldnames, offset[:-1], offset[1:]):
756:             names = a.dtype.names
757:             if names is None:
758:                 output['f%i' % len(seen)][i:j] = a
759:             else:
760:                 for name in n:
761:                     output[name][i:j] = a[name]
762:                     if name not in seen:
763:                         seen.append(name)
764:     #
765:     return _fix_output(_fix_defaults(output, defaults),
766:                        usemask=usemask, asrecarray=asrecarray)
767: 
768: 
769: def find_duplicates(a, key=None, ignoremask=True, return_index=False):
770:     '''
771:     Find the duplicates in a structured array along a given key
772: 
773:     Parameters
774:     ----------
775:     a : array-like
776:         Input array
777:     key : {string, None}, optional
778:         Name of the fields along which to check the duplicates.
779:         If None, the search is performed by records
780:     ignoremask : {True, False}, optional
781:         Whether masked data should be discarded or considered as duplicates.
782:     return_index : {False, True}, optional
783:         Whether to return the indices of the duplicated values.
784: 
785:     Examples
786:     --------
787:     >>> from numpy.lib import recfunctions as rfn
788:     >>> ndtype = [('a', int)]
789:     >>> a = np.ma.array([1, 1, 1, 2, 2, 3, 3],
790:     ...         mask=[0, 0, 1, 0, 0, 0, 1]).view(ndtype)
791:     >>> rfn.find_duplicates(a, ignoremask=True, return_index=True)
792:     ... # XXX: judging by the output, the ignoremask flag has no effect
793:     '''
794:     a = np.asanyarray(a).ravel()
795:     # Get a dictionary of fields
796:     fields = get_fieldstructure(a.dtype)
797:     # Get the sorting data (by selecting the corresponding field)
798:     base = a
799:     if key:
800:         for f in fields[key]:
801:             base = base[f]
802:         base = base[key]
803:     # Get the sorting indices and the sorted data
804:     sortidx = base.argsort()
805:     sortedbase = base[sortidx]
806:     sorteddata = sortedbase.filled()
807:     # Compare the sorting data
808:     flag = (sorteddata[:-1] == sorteddata[1:])
809:     # If masked data must be ignored, set the flag to false where needed
810:     if ignoremask:
811:         sortedmask = sortedbase.recordmask
812:         flag[sortedmask[1:]] = False
813:     flag = np.concatenate(([False], flag))
814:     # We need to take the point on the left as well (else we're missing it)
815:     flag[:-1] = flag[:-1] + flag[1:]
816:     duplicates = a[sortidx][flag]
817:     if return_index:
818:         return (duplicates, sortidx[flag])
819:     else:
820:         return duplicates
821: 
822: 
823: def join_by(key, r1, r2, jointype='inner', r1postfix='1', r2postfix='2',
824:                 defaults=None, usemask=True, asrecarray=False):
825:     '''
826:     Join arrays `r1` and `r2` on key `key`.
827: 
828:     The key should be either a string or a sequence of string corresponding
829:     to the fields used to join the array.  An exception is raised if the
830:     `key` field cannot be found in the two input arrays.  Neither `r1` nor
831:     `r2` should have any duplicates along `key`: the presence of duplicates
832:     will make the output quite unreliable. Note that duplicates are not
833:     looked for by the algorithm.
834: 
835:     Parameters
836:     ----------
837:     key : {string, sequence}
838:         A string or a sequence of strings corresponding to the fields used
839:         for comparison.
840:     r1, r2 : arrays
841:         Structured arrays.
842:     jointype : {'inner', 'outer', 'leftouter'}, optional
843:         If 'inner', returns the elements common to both r1 and r2.
844:         If 'outer', returns the common elements as well as the elements of
845:         r1 not in r2 and the elements of not in r2.
846:         If 'leftouter', returns the common elements and the elements of r1
847:         not in r2.
848:     r1postfix : string, optional
849:         String appended to the names of the fields of r1 that are present
850:         in r2 but absent of the key.
851:     r2postfix : string, optional
852:         String appended to the names of the fields of r2 that are present
853:         in r1 but absent of the key.
854:     defaults : {dictionary}, optional
855:         Dictionary mapping field names to the corresponding default values.
856:     usemask : {True, False}, optional
857:         Whether to return a MaskedArray (or MaskedRecords is
858:         `asrecarray==True`) or a ndarray.
859:     asrecarray : {False, True}, optional
860:         Whether to return a recarray (or MaskedRecords if `usemask==True`)
861:         or just a flexible-type ndarray.
862: 
863:     Notes
864:     -----
865:     * The output is sorted along the key.
866:     * A temporary array is formed by dropping the fields not in the key for
867:       the two arrays and concatenating the result. This array is then
868:       sorted, and the common entries selected. The output is constructed by
869:       filling the fields with the selected entries. Matching is not
870:       preserved if there are some duplicates...
871: 
872:     '''
873:     # Check jointype
874:     if jointype not in ('inner', 'outer', 'leftouter'):
875:         raise ValueError(
876:                 "The 'jointype' argument should be in 'inner', "
877:                 "'outer' or 'leftouter' (got '%s' instead)" % jointype
878:                 )
879:     # If we have a single key, put it in a tuple
880:     if isinstance(key, basestring):
881:         key = (key,)
882: 
883:     # Check the keys
884:     for name in key:
885:         if name not in r1.dtype.names:
886:             raise ValueError('r1 does not have key field %s' % name)
887:         if name not in r2.dtype.names:
888:             raise ValueError('r2 does not have key field %s' % name)
889: 
890:     # Make sure we work with ravelled arrays
891:     r1 = r1.ravel()
892:     r2 = r2.ravel()
893:     # Fixme: nb2 below is never used. Commenting out for pyflakes.
894:     # (nb1, nb2) = (len(r1), len(r2))
895:     nb1 = len(r1)
896:     (r1names, r2names) = (r1.dtype.names, r2.dtype.names)
897: 
898:     # Check the names for collision
899:     if (set.intersection(set(r1names), set(r2names)).difference(key) and
900:             not (r1postfix or r2postfix)):
901:         msg = "r1 and r2 contain common names, r1postfix and r2postfix "
902:         msg += "can't be empty"
903:         raise ValueError(msg)
904: 
905:     # Make temporary arrays of just the keys
906:     r1k = drop_fields(r1, [n for n in r1names if n not in key])
907:     r2k = drop_fields(r2, [n for n in r2names if n not in key])
908: 
909:     # Concatenate the two arrays for comparison
910:     aux = ma.concatenate((r1k, r2k))
911:     idx_sort = aux.argsort(order=key)
912:     aux = aux[idx_sort]
913:     #
914:     # Get the common keys
915:     flag_in = ma.concatenate(([False], aux[1:] == aux[:-1]))
916:     flag_in[:-1] = flag_in[1:] + flag_in[:-1]
917:     idx_in = idx_sort[flag_in]
918:     idx_1 = idx_in[(idx_in < nb1)]
919:     idx_2 = idx_in[(idx_in >= nb1)] - nb1
920:     (r1cmn, r2cmn) = (len(idx_1), len(idx_2))
921:     if jointype == 'inner':
922:         (r1spc, r2spc) = (0, 0)
923:     elif jointype == 'outer':
924:         idx_out = idx_sort[~flag_in]
925:         idx_1 = np.concatenate((idx_1, idx_out[(idx_out < nb1)]))
926:         idx_2 = np.concatenate((idx_2, idx_out[(idx_out >= nb1)] - nb1))
927:         (r1spc, r2spc) = (len(idx_1) - r1cmn, len(idx_2) - r2cmn)
928:     elif jointype == 'leftouter':
929:         idx_out = idx_sort[~flag_in]
930:         idx_1 = np.concatenate((idx_1, idx_out[(idx_out < nb1)]))
931:         (r1spc, r2spc) = (len(idx_1) - r1cmn, 0)
932:     # Select the entries from each input
933:     (s1, s2) = (r1[idx_1], r2[idx_2])
934:     #
935:     # Build the new description of the output array .......
936:     # Start with the key fields
937:     ndtype = [list(_) for _ in r1k.dtype.descr]
938:     # Add the other fields
939:     ndtype.extend(list(_) for _ in r1.dtype.descr if _[0] not in key)
940:     # Find the new list of names (it may be different from r1names)
941:     names = list(_[0] for _ in ndtype)
942:     for desc in r2.dtype.descr:
943:         desc = list(desc)
944:         name = desc[0]
945:         # Have we seen the current name already ?
946:         if name in names:
947:             nameidx = ndtype.index(desc)
948:             current = ndtype[nameidx]
949:             # The current field is part of the key: take the largest dtype
950:             if name in key:
951:                 current[-1] = max(desc[1], current[-1])
952:             # The current field is not part of the key: add the suffixes
953:             else:
954:                 current[0] += r1postfix
955:                 desc[0] += r2postfix
956:                 ndtype.insert(nameidx + 1, desc)
957:         #... we haven't: just add the description to the current list
958:         else:
959:             names.extend(desc[0])
960:             ndtype.append(desc)
961:     # Revert the elements to tuples
962:     ndtype = [tuple(_) for _ in ndtype]
963:     # Find the largest nb of common fields :
964:     # r1cmn and r2cmn should be equal, but...
965:     cmn = max(r1cmn, r2cmn)
966:     # Construct an empty array
967:     output = ma.masked_all((cmn + r1spc + r2spc,), dtype=ndtype)
968:     names = output.dtype.names
969:     for f in r1names:
970:         selected = s1[f]
971:         if f not in names or (f in r2names and not r2postfix and f not in key):
972:             f += r1postfix
973:         current = output[f]
974:         current[:r1cmn] = selected[:r1cmn]
975:         if jointype in ('outer', 'leftouter'):
976:             current[cmn:cmn + r1spc] = selected[r1cmn:]
977:     for f in r2names:
978:         selected = s2[f]
979:         if f not in names or (f in r1names and not r1postfix and f not in key):
980:             f += r2postfix
981:         current = output[f]
982:         current[:r2cmn] = selected[:r2cmn]
983:         if (jointype == 'outer') and r2spc:
984:             current[-r2spc:] = selected[r2cmn:]
985:     # Sort and finalize the output
986:     output.sort(order=key)
987:     kwargs = dict(usemask=usemask, asrecarray=asrecarray)
988:     return _fix_output(_fix_defaults(output, defaults), **kwargs)
989: 
990: 
991: def rec_join(key, r1, r2, jointype='inner', r1postfix='1', r2postfix='2',
992:              defaults=None):
993:     '''
994:     Join arrays `r1` and `r2` on keys.
995:     Alternative to join_by, that always returns a np.recarray.
996: 
997:     See Also
998:     --------
999:     join_by : equivalent function
1000:     '''
1001:     kwargs = dict(jointype=jointype, r1postfix=r1postfix, r2postfix=r2postfix,
1002:                   defaults=defaults, usemask=False, asrecarray=True)
1003:     return join_by(key, r1, r2, **kwargs)
1004: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_122707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, (-1)), 'str', '\nCollection of utilities to manipulate structured arrays.\n\nMost of these functions were initially implemented by John Hunter for\nmatplotlib.  They have been rewritten and extended for convenience.\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'import sys' statement (line 10)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'import itertools' statement (line 11)
import itertools

import_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'itertools', itertools, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'import numpy' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/lib/')
import_122708 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'numpy')

if (type(import_122708) is not StypyTypeError):

    if (import_122708 != 'pyd_module'):
        __import__(import_122708)
        sys_modules_122709 = sys.modules[import_122708]
        import_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'np', sys_modules_122709.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'numpy', import_122708)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/lib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'import numpy.ma' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/lib/')
import_122710 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy.ma')

if (type(import_122710) is not StypyTypeError):

    if (import_122710 != 'pyd_module'):
        __import__(import_122710)
        sys_modules_122711 = sys.modules[import_122710]
        import_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'ma', sys_modules_122711.module_type_store, module_type_store)
    else:
        import numpy.ma as ma

        import_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'ma', numpy.ma, module_type_store)

else:
    # Assigning a type to the variable 'numpy.ma' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy.ma', import_122710)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/lib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'from numpy import ndarray, recarray' statement (line 14)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/lib/')
import_122712 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'numpy')

if (type(import_122712) is not StypyTypeError):

    if (import_122712 != 'pyd_module'):
        __import__(import_122712)
        sys_modules_122713 = sys.modules[import_122712]
        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'numpy', sys_modules_122713.module_type_store, module_type_store, ['ndarray', 'recarray'])
        nest_module(stypy.reporting.localization.Localization(__file__, 14, 0), __file__, sys_modules_122713, sys_modules_122713.module_type_store, module_type_store)
    else:
        from numpy import ndarray, recarray

        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'numpy', None, module_type_store, ['ndarray', 'recarray'], [ndarray, recarray])

else:
    # Assigning a type to the variable 'numpy' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'numpy', import_122712)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/lib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 0))

# 'from numpy.ma import MaskedArray' statement (line 15)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/lib/')
import_122714 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'numpy.ma')

if (type(import_122714) is not StypyTypeError):

    if (import_122714 != 'pyd_module'):
        __import__(import_122714)
        sys_modules_122715 = sys.modules[import_122714]
        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'numpy.ma', sys_modules_122715.module_type_store, module_type_store, ['MaskedArray'])
        nest_module(stypy.reporting.localization.Localization(__file__, 15, 0), __file__, sys_modules_122715, sys_modules_122715.module_type_store, module_type_store)
    else:
        from numpy.ma import MaskedArray

        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'numpy.ma', None, module_type_store, ['MaskedArray'], [MaskedArray])

else:
    # Assigning a type to the variable 'numpy.ma' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'numpy.ma', import_122714)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/lib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 16, 0))

# 'from numpy.ma.mrecords import MaskedRecords' statement (line 16)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/lib/')
import_122716 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'numpy.ma.mrecords')

if (type(import_122716) is not StypyTypeError):

    if (import_122716 != 'pyd_module'):
        __import__(import_122716)
        sys_modules_122717 = sys.modules[import_122716]
        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'numpy.ma.mrecords', sys_modules_122717.module_type_store, module_type_store, ['MaskedRecords'])
        nest_module(stypy.reporting.localization.Localization(__file__, 16, 0), __file__, sys_modules_122717, sys_modules_122717.module_type_store, module_type_store)
    else:
        from numpy.ma.mrecords import MaskedRecords

        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'numpy.ma.mrecords', None, module_type_store, ['MaskedRecords'], [MaskedRecords])

else:
    # Assigning a type to the variable 'numpy.ma.mrecords' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'numpy.ma.mrecords', import_122716)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/lib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 17, 0))

# 'from numpy.lib._iotools import _is_string_like' statement (line 17)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/lib/')
import_122718 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'numpy.lib._iotools')

if (type(import_122718) is not StypyTypeError):

    if (import_122718 != 'pyd_module'):
        __import__(import_122718)
        sys_modules_122719 = sys.modules[import_122718]
        import_from_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'numpy.lib._iotools', sys_modules_122719.module_type_store, module_type_store, ['_is_string_like'])
        nest_module(stypy.reporting.localization.Localization(__file__, 17, 0), __file__, sys_modules_122719, sys_modules_122719.module_type_store, module_type_store)
    else:
        from numpy.lib._iotools import _is_string_like

        import_from_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'numpy.lib._iotools', None, module_type_store, ['_is_string_like'], [_is_string_like])

else:
    # Assigning a type to the variable 'numpy.lib._iotools' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'numpy.lib._iotools', import_122718)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/lib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 18, 0))

# 'from numpy.compat import basestring' statement (line 18)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/lib/')
import_122720 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'numpy.compat')

if (type(import_122720) is not StypyTypeError):

    if (import_122720 != 'pyd_module'):
        __import__(import_122720)
        sys_modules_122721 = sys.modules[import_122720]
        import_from_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'numpy.compat', sys_modules_122721.module_type_store, module_type_store, ['basestring'])
        nest_module(stypy.reporting.localization.Localization(__file__, 18, 0), __file__, sys_modules_122721, sys_modules_122721.module_type_store, module_type_store)
    else:
        from numpy.compat import basestring

        import_from_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'numpy.compat', None, module_type_store, ['basestring'], [basestring])

else:
    # Assigning a type to the variable 'numpy.compat' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'numpy.compat', import_122720)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/lib/')




# Obtaining the type of the subscript
int_122722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 20), 'int')
# Getting the type of 'sys' (line 20)
sys_122723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 3), 'sys')
# Obtaining the member 'version_info' of a type (line 20)
version_info_122724 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 3), sys_122723, 'version_info')
# Obtaining the member '__getitem__' of a type (line 20)
getitem___122725 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 3), version_info_122724, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 20)
subscript_call_result_122726 = invoke(stypy.reporting.localization.Localization(__file__, 20, 3), getitem___122725, int_122722)

int_122727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 25), 'int')
# Applying the binary operator '<' (line 20)
result_lt_122728 = python_operator(stypy.reporting.localization.Localization(__file__, 20, 3), '<', subscript_call_result_122726, int_122727)

# Testing the type of an if condition (line 20)
if_condition_122729 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 20, 0), result_lt_122728)
# Assigning a type to the variable 'if_condition_122729' (line 20)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'if_condition_122729', if_condition_122729)
# SSA begins for if statement (line 20)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 21, 4))

# 'from future_builtins import zip' statement (line 21)
from future_builtins import zip

import_from_module(stypy.reporting.localization.Localization(__file__, 21, 4), 'future_builtins', None, module_type_store, ['zip'], [zip])

# SSA join for if statement (line 20)
module_type_store = module_type_store.join_ssa_context()


# Assigning a Attribute to a Name (line 23):

# Assigning a Attribute to a Name (line 23):
# Getting the type of 'np' (line 23)
np_122730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 20), 'np')
# Obtaining the member 'ma' of a type (line 23)
ma_122731 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 20), np_122730, 'ma')
# Obtaining the member 'core' of a type (line 23)
core_122732 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 20), ma_122731, 'core')
# Obtaining the member '_check_fill_value' of a type (line 23)
_check_fill_value_122733 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 20), core_122732, '_check_fill_value')
# Assigning a type to the variable '_check_fill_value' (line 23)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), '_check_fill_value', _check_fill_value_122733)

# Assigning a List to a Name (line 26):

# Assigning a List to a Name (line 26):
__all__ = ['append_fields', 'drop_fields', 'find_duplicates', 'get_fieldstructure', 'join_by', 'merge_arrays', 'rec_append_fields', 'rec_drop_fields', 'rec_join', 'recursive_fill_fields', 'rename_fields', 'stack_arrays']
module_type_store.set_exportable_members(['append_fields', 'drop_fields', 'find_duplicates', 'get_fieldstructure', 'join_by', 'merge_arrays', 'rec_append_fields', 'rec_drop_fields', 'rec_join', 'recursive_fill_fields', 'rename_fields', 'stack_arrays'])

# Obtaining an instance of the builtin type 'list' (line 26)
list_122734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 26)
# Adding element type (line 26)
str_122735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 4), 'str', 'append_fields')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 10), list_122734, str_122735)
# Adding element type (line 26)
str_122736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 21), 'str', 'drop_fields')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 10), list_122734, str_122736)
# Adding element type (line 26)
str_122737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 36), 'str', 'find_duplicates')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 10), list_122734, str_122737)
# Adding element type (line 26)
str_122738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 4), 'str', 'get_fieldstructure')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 10), list_122734, str_122738)
# Adding element type (line 26)
str_122739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 26), 'str', 'join_by')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 10), list_122734, str_122739)
# Adding element type (line 26)
str_122740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 37), 'str', 'merge_arrays')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 10), list_122734, str_122740)
# Adding element type (line 26)
str_122741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 4), 'str', 'rec_append_fields')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 10), list_122734, str_122741)
# Adding element type (line 26)
str_122742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 25), 'str', 'rec_drop_fields')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 10), list_122734, str_122742)
# Adding element type (line 26)
str_122743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 44), 'str', 'rec_join')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 10), list_122734, str_122743)
# Adding element type (line 26)
str_122744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 4), 'str', 'recursive_fill_fields')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 10), list_122734, str_122744)
# Adding element type (line 26)
str_122745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 29), 'str', 'rename_fields')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 10), list_122734, str_122745)
# Adding element type (line 26)
str_122746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 46), 'str', 'stack_arrays')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 10), list_122734, str_122746)

# Assigning a type to the variable '__all__' (line 26)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), '__all__', list_122734)

@norecursion
def recursive_fill_fields(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'recursive_fill_fields'
    module_type_store = module_type_store.open_function_context('recursive_fill_fields', 34, 0, False)
    
    # Passed parameters checking function
    recursive_fill_fields.stypy_localization = localization
    recursive_fill_fields.stypy_type_of_self = None
    recursive_fill_fields.stypy_type_store = module_type_store
    recursive_fill_fields.stypy_function_name = 'recursive_fill_fields'
    recursive_fill_fields.stypy_param_names_list = ['input', 'output']
    recursive_fill_fields.stypy_varargs_param_name = None
    recursive_fill_fields.stypy_kwargs_param_name = None
    recursive_fill_fields.stypy_call_defaults = defaults
    recursive_fill_fields.stypy_call_varargs = varargs
    recursive_fill_fields.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'recursive_fill_fields', ['input', 'output'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'recursive_fill_fields', localization, ['input', 'output'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'recursive_fill_fields(...)' code ##################

    str_122747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, (-1)), 'str', "\n    Fills fields from output with fields from input,\n    with support for nested structures.\n\n    Parameters\n    ----------\n    input : ndarray\n        Input array.\n    output : ndarray\n        Output array.\n\n    Notes\n    -----\n    * `output` should be at least the same size as `input`\n\n    Examples\n    --------\n    >>> from numpy.lib import recfunctions as rfn\n    >>> a = np.array([(1, 10.), (2, 20.)], dtype=[('A', int), ('B', float)])\n    >>> b = np.zeros((3,), dtype=a.dtype)\n    >>> rfn.recursive_fill_fields(a, b)\n    array([(1, 10.0), (2, 20.0), (0, 0.0)],\n          dtype=[('A', '<i4'), ('B', '<f8')])\n\n    ")
    
    # Assigning a Attribute to a Name (line 60):
    
    # Assigning a Attribute to a Name (line 60):
    # Getting the type of 'output' (line 60)
    output_122748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 15), 'output')
    # Obtaining the member 'dtype' of a type (line 60)
    dtype_122749 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 15), output_122748, 'dtype')
    # Assigning a type to the variable 'newdtype' (line 60)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'newdtype', dtype_122749)
    
    # Getting the type of 'newdtype' (line 61)
    newdtype_122750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 17), 'newdtype')
    # Obtaining the member 'names' of a type (line 61)
    names_122751 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 17), newdtype_122750, 'names')
    # Testing the type of a for loop iterable (line 61)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 61, 4), names_122751)
    # Getting the type of the for loop variable (line 61)
    for_loop_var_122752 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 61, 4), names_122751)
    # Assigning a type to the variable 'field' (line 61)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), 'field', for_loop_var_122752)
    # SSA begins for a for statement (line 61)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # SSA begins for try-except statement (line 62)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Subscript to a Name (line 63):
    
    # Assigning a Subscript to a Name (line 63):
    
    # Obtaining the type of the subscript
    # Getting the type of 'field' (line 63)
    field_122753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 28), 'field')
    # Getting the type of 'input' (line 63)
    input_122754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 22), 'input')
    # Obtaining the member '__getitem__' of a type (line 63)
    getitem___122755 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 22), input_122754, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 63)
    subscript_call_result_122756 = invoke(stypy.reporting.localization.Localization(__file__, 63, 22), getitem___122755, field_122753)
    
    # Assigning a type to the variable 'current' (line 63)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 12), 'current', subscript_call_result_122756)
    # SSA branch for the except part of a try statement (line 62)
    # SSA branch for the except 'ValueError' branch of a try statement (line 62)
    module_type_store.open_ssa_branch('except')
    # SSA join for try-except statement (line 62)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'current' (line 66)
    current_122757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 11), 'current')
    # Obtaining the member 'dtype' of a type (line 66)
    dtype_122758 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 11), current_122757, 'dtype')
    # Obtaining the member 'names' of a type (line 66)
    names_122759 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 11), dtype_122758, 'names')
    # Testing the type of an if condition (line 66)
    if_condition_122760 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 66, 8), names_122759)
    # Assigning a type to the variable 'if_condition_122760' (line 66)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'if_condition_122760', if_condition_122760)
    # SSA begins for if statement (line 66)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to recursive_fill_fields(...): (line 67)
    # Processing the call arguments (line 67)
    # Getting the type of 'current' (line 67)
    current_122762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 34), 'current', False)
    
    # Obtaining the type of the subscript
    # Getting the type of 'field' (line 67)
    field_122763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 50), 'field', False)
    # Getting the type of 'output' (line 67)
    output_122764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 43), 'output', False)
    # Obtaining the member '__getitem__' of a type (line 67)
    getitem___122765 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 43), output_122764, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 67)
    subscript_call_result_122766 = invoke(stypy.reporting.localization.Localization(__file__, 67, 43), getitem___122765, field_122763)
    
    # Processing the call keyword arguments (line 67)
    kwargs_122767 = {}
    # Getting the type of 'recursive_fill_fields' (line 67)
    recursive_fill_fields_122761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 12), 'recursive_fill_fields', False)
    # Calling recursive_fill_fields(args, kwargs) (line 67)
    recursive_fill_fields_call_result_122768 = invoke(stypy.reporting.localization.Localization(__file__, 67, 12), recursive_fill_fields_122761, *[current_122762, subscript_call_result_122766], **kwargs_122767)
    
    # SSA branch for the else part of an if statement (line 66)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Subscript (line 69):
    
    # Assigning a Name to a Subscript (line 69):
    # Getting the type of 'current' (line 69)
    current_122769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 43), 'current')
    
    # Obtaining the type of the subscript
    # Getting the type of 'field' (line 69)
    field_122770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 19), 'field')
    # Getting the type of 'output' (line 69)
    output_122771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 12), 'output')
    # Obtaining the member '__getitem__' of a type (line 69)
    getitem___122772 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 12), output_122771, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 69)
    subscript_call_result_122773 = invoke(stypy.reporting.localization.Localization(__file__, 69, 12), getitem___122772, field_122770)
    
    
    # Call to len(...): (line 69)
    # Processing the call arguments (line 69)
    # Getting the type of 'current' (line 69)
    current_122775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 31), 'current', False)
    # Processing the call keyword arguments (line 69)
    kwargs_122776 = {}
    # Getting the type of 'len' (line 69)
    len_122774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 27), 'len', False)
    # Calling len(args, kwargs) (line 69)
    len_call_result_122777 = invoke(stypy.reporting.localization.Localization(__file__, 69, 27), len_122774, *[current_122775], **kwargs_122776)
    
    slice_122778 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 69, 12), None, len_call_result_122777, None)
    # Storing an element on a container (line 69)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 12), subscript_call_result_122773, (slice_122778, current_122769))
    # SSA join for if statement (line 66)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'output' (line 70)
    output_122779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 11), 'output')
    # Assigning a type to the variable 'stypy_return_type' (line 70)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'stypy_return_type', output_122779)
    
    # ################# End of 'recursive_fill_fields(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'recursive_fill_fields' in the type store
    # Getting the type of 'stypy_return_type' (line 34)
    stypy_return_type_122780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_122780)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'recursive_fill_fields'
    return stypy_return_type_122780

# Assigning a type to the variable 'recursive_fill_fields' (line 34)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), 'recursive_fill_fields', recursive_fill_fields)

@norecursion
def get_names(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'get_names'
    module_type_store = module_type_store.open_function_context('get_names', 73, 0, False)
    
    # Passed parameters checking function
    get_names.stypy_localization = localization
    get_names.stypy_type_of_self = None
    get_names.stypy_type_store = module_type_store
    get_names.stypy_function_name = 'get_names'
    get_names.stypy_param_names_list = ['adtype']
    get_names.stypy_varargs_param_name = None
    get_names.stypy_kwargs_param_name = None
    get_names.stypy_call_defaults = defaults
    get_names.stypy_call_varargs = varargs
    get_names.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'get_names', ['adtype'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'get_names', localization, ['adtype'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'get_names(...)' code ##################

    str_122781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, (-1)), 'str', "\n    Returns the field names of the input datatype as a tuple.\n\n    Parameters\n    ----------\n    adtype : dtype\n        Input datatype\n\n    Examples\n    --------\n    >>> from numpy.lib import recfunctions as rfn\n    >>> rfn.get_names(np.empty((1,), dtype=int)) is None\n    True\n    >>> rfn.get_names(np.empty((1,), dtype=[('A',int), ('B', float)]))\n    ('A', 'B')\n    >>> adtype = np.dtype([('a', int), ('b', [('ba', int), ('bb', int)])])\n    >>> rfn.get_names(adtype)\n    ('a', ('b', ('ba', 'bb')))\n    ")
    
    # Assigning a List to a Name (line 93):
    
    # Assigning a List to a Name (line 93):
    
    # Obtaining an instance of the builtin type 'list' (line 93)
    list_122782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 93)
    
    # Assigning a type to the variable 'listnames' (line 93)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'listnames', list_122782)
    
    # Assigning a Attribute to a Name (line 94):
    
    # Assigning a Attribute to a Name (line 94):
    # Getting the type of 'adtype' (line 94)
    adtype_122783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 12), 'adtype')
    # Obtaining the member 'names' of a type (line 94)
    names_122784 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 12), adtype_122783, 'names')
    # Assigning a type to the variable 'names' (line 94)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'names', names_122784)
    
    # Getting the type of 'names' (line 95)
    names_122785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 16), 'names')
    # Testing the type of a for loop iterable (line 95)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 95, 4), names_122785)
    # Getting the type of the for loop variable (line 95)
    for_loop_var_122786 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 95, 4), names_122785)
    # Assigning a type to the variable 'name' (line 95)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 4), 'name', for_loop_var_122786)
    # SSA begins for a for statement (line 95)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Subscript to a Name (line 96):
    
    # Assigning a Subscript to a Name (line 96):
    
    # Obtaining the type of the subscript
    # Getting the type of 'name' (line 96)
    name_122787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 25), 'name')
    # Getting the type of 'adtype' (line 96)
    adtype_122788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 18), 'adtype')
    # Obtaining the member '__getitem__' of a type (line 96)
    getitem___122789 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 18), adtype_122788, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 96)
    subscript_call_result_122790 = invoke(stypy.reporting.localization.Localization(__file__, 96, 18), getitem___122789, name_122787)
    
    # Assigning a type to the variable 'current' (line 96)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'current', subscript_call_result_122790)
    
    # Getting the type of 'current' (line 97)
    current_122791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 11), 'current')
    # Obtaining the member 'names' of a type (line 97)
    names_122792 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 11), current_122791, 'names')
    # Testing the type of an if condition (line 97)
    if_condition_122793 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 97, 8), names_122792)
    # Assigning a type to the variable 'if_condition_122793' (line 97)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'if_condition_122793', if_condition_122793)
    # SSA begins for if statement (line 97)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 98)
    # Processing the call arguments (line 98)
    
    # Obtaining an instance of the builtin type 'tuple' (line 98)
    tuple_122796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 30), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 98)
    # Adding element type (line 98)
    # Getting the type of 'name' (line 98)
    name_122797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 30), 'name', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 30), tuple_122796, name_122797)
    # Adding element type (line 98)
    
    # Call to tuple(...): (line 98)
    # Processing the call arguments (line 98)
    
    # Call to get_names(...): (line 98)
    # Processing the call arguments (line 98)
    # Getting the type of 'current' (line 98)
    current_122800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 52), 'current', False)
    # Processing the call keyword arguments (line 98)
    kwargs_122801 = {}
    # Getting the type of 'get_names' (line 98)
    get_names_122799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 42), 'get_names', False)
    # Calling get_names(args, kwargs) (line 98)
    get_names_call_result_122802 = invoke(stypy.reporting.localization.Localization(__file__, 98, 42), get_names_122799, *[current_122800], **kwargs_122801)
    
    # Processing the call keyword arguments (line 98)
    kwargs_122803 = {}
    # Getting the type of 'tuple' (line 98)
    tuple_122798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 36), 'tuple', False)
    # Calling tuple(args, kwargs) (line 98)
    tuple_call_result_122804 = invoke(stypy.reporting.localization.Localization(__file__, 98, 36), tuple_122798, *[get_names_call_result_122802], **kwargs_122803)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 30), tuple_122796, tuple_call_result_122804)
    
    # Processing the call keyword arguments (line 98)
    kwargs_122805 = {}
    # Getting the type of 'listnames' (line 98)
    listnames_122794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 12), 'listnames', False)
    # Obtaining the member 'append' of a type (line 98)
    append_122795 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 12), listnames_122794, 'append')
    # Calling append(args, kwargs) (line 98)
    append_call_result_122806 = invoke(stypy.reporting.localization.Localization(__file__, 98, 12), append_122795, *[tuple_122796], **kwargs_122805)
    
    # SSA branch for the else part of an if statement (line 97)
    module_type_store.open_ssa_branch('else')
    
    # Call to append(...): (line 100)
    # Processing the call arguments (line 100)
    # Getting the type of 'name' (line 100)
    name_122809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 29), 'name', False)
    # Processing the call keyword arguments (line 100)
    kwargs_122810 = {}
    # Getting the type of 'listnames' (line 100)
    listnames_122807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 12), 'listnames', False)
    # Obtaining the member 'append' of a type (line 100)
    append_122808 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 12), listnames_122807, 'append')
    # Calling append(args, kwargs) (line 100)
    append_call_result_122811 = invoke(stypy.reporting.localization.Localization(__file__, 100, 12), append_122808, *[name_122809], **kwargs_122810)
    
    # SSA join for if statement (line 97)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Evaluating a boolean operation
    
    # Call to tuple(...): (line 101)
    # Processing the call arguments (line 101)
    # Getting the type of 'listnames' (line 101)
    listnames_122813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 17), 'listnames', False)
    # Processing the call keyword arguments (line 101)
    kwargs_122814 = {}
    # Getting the type of 'tuple' (line 101)
    tuple_122812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 11), 'tuple', False)
    # Calling tuple(args, kwargs) (line 101)
    tuple_call_result_122815 = invoke(stypy.reporting.localization.Localization(__file__, 101, 11), tuple_122812, *[listnames_122813], **kwargs_122814)
    
    # Getting the type of 'None' (line 101)
    None_122816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 31), 'None')
    # Applying the binary operator 'or' (line 101)
    result_or_keyword_122817 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 11), 'or', tuple_call_result_122815, None_122816)
    
    # Assigning a type to the variable 'stypy_return_type' (line 101)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'stypy_return_type', result_or_keyword_122817)
    
    # ################# End of 'get_names(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_names' in the type store
    # Getting the type of 'stypy_return_type' (line 73)
    stypy_return_type_122818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_122818)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_names'
    return stypy_return_type_122818

# Assigning a type to the variable 'get_names' (line 73)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 0), 'get_names', get_names)

@norecursion
def get_names_flat(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'get_names_flat'
    module_type_store = module_type_store.open_function_context('get_names_flat', 104, 0, False)
    
    # Passed parameters checking function
    get_names_flat.stypy_localization = localization
    get_names_flat.stypy_type_of_self = None
    get_names_flat.stypy_type_store = module_type_store
    get_names_flat.stypy_function_name = 'get_names_flat'
    get_names_flat.stypy_param_names_list = ['adtype']
    get_names_flat.stypy_varargs_param_name = None
    get_names_flat.stypy_kwargs_param_name = None
    get_names_flat.stypy_call_defaults = defaults
    get_names_flat.stypy_call_varargs = varargs
    get_names_flat.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'get_names_flat', ['adtype'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'get_names_flat', localization, ['adtype'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'get_names_flat(...)' code ##################

    str_122819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, (-1)), 'str', "\n    Returns the field names of the input datatype as a tuple. Nested structure\n    are flattend beforehand.\n\n    Parameters\n    ----------\n    adtype : dtype\n        Input datatype\n\n    Examples\n    --------\n    >>> from numpy.lib import recfunctions as rfn\n    >>> rfn.get_names_flat(np.empty((1,), dtype=int)) is None\n    True\n    >>> rfn.get_names_flat(np.empty((1,), dtype=[('A',int), ('B', float)]))\n    ('A', 'B')\n    >>> adtype = np.dtype([('a', int), ('b', [('ba', int), ('bb', int)])])\n    >>> rfn.get_names_flat(adtype)\n    ('a', 'b', 'ba', 'bb')\n    ")
    
    # Assigning a List to a Name (line 125):
    
    # Assigning a List to a Name (line 125):
    
    # Obtaining an instance of the builtin type 'list' (line 125)
    list_122820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 125)
    
    # Assigning a type to the variable 'listnames' (line 125)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 4), 'listnames', list_122820)
    
    # Assigning a Attribute to a Name (line 126):
    
    # Assigning a Attribute to a Name (line 126):
    # Getting the type of 'adtype' (line 126)
    adtype_122821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 12), 'adtype')
    # Obtaining the member 'names' of a type (line 126)
    names_122822 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 12), adtype_122821, 'names')
    # Assigning a type to the variable 'names' (line 126)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 4), 'names', names_122822)
    
    # Getting the type of 'names' (line 127)
    names_122823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 16), 'names')
    # Testing the type of a for loop iterable (line 127)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 127, 4), names_122823)
    # Getting the type of the for loop variable (line 127)
    for_loop_var_122824 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 127, 4), names_122823)
    # Assigning a type to the variable 'name' (line 127)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 4), 'name', for_loop_var_122824)
    # SSA begins for a for statement (line 127)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to append(...): (line 128)
    # Processing the call arguments (line 128)
    # Getting the type of 'name' (line 128)
    name_122827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 25), 'name', False)
    # Processing the call keyword arguments (line 128)
    kwargs_122828 = {}
    # Getting the type of 'listnames' (line 128)
    listnames_122825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'listnames', False)
    # Obtaining the member 'append' of a type (line 128)
    append_122826 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 8), listnames_122825, 'append')
    # Calling append(args, kwargs) (line 128)
    append_call_result_122829 = invoke(stypy.reporting.localization.Localization(__file__, 128, 8), append_122826, *[name_122827], **kwargs_122828)
    
    
    # Assigning a Subscript to a Name (line 129):
    
    # Assigning a Subscript to a Name (line 129):
    
    # Obtaining the type of the subscript
    # Getting the type of 'name' (line 129)
    name_122830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 25), 'name')
    # Getting the type of 'adtype' (line 129)
    adtype_122831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 18), 'adtype')
    # Obtaining the member '__getitem__' of a type (line 129)
    getitem___122832 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 18), adtype_122831, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 129)
    subscript_call_result_122833 = invoke(stypy.reporting.localization.Localization(__file__, 129, 18), getitem___122832, name_122830)
    
    # Assigning a type to the variable 'current' (line 129)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'current', subscript_call_result_122833)
    
    # Getting the type of 'current' (line 130)
    current_122834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 11), 'current')
    # Obtaining the member 'names' of a type (line 130)
    names_122835 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 11), current_122834, 'names')
    # Testing the type of an if condition (line 130)
    if_condition_122836 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 130, 8), names_122835)
    # Assigning a type to the variable 'if_condition_122836' (line 130)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 8), 'if_condition_122836', if_condition_122836)
    # SSA begins for if statement (line 130)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to extend(...): (line 131)
    # Processing the call arguments (line 131)
    
    # Call to get_names_flat(...): (line 131)
    # Processing the call arguments (line 131)
    # Getting the type of 'current' (line 131)
    current_122840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 44), 'current', False)
    # Processing the call keyword arguments (line 131)
    kwargs_122841 = {}
    # Getting the type of 'get_names_flat' (line 131)
    get_names_flat_122839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 29), 'get_names_flat', False)
    # Calling get_names_flat(args, kwargs) (line 131)
    get_names_flat_call_result_122842 = invoke(stypy.reporting.localization.Localization(__file__, 131, 29), get_names_flat_122839, *[current_122840], **kwargs_122841)
    
    # Processing the call keyword arguments (line 131)
    kwargs_122843 = {}
    # Getting the type of 'listnames' (line 131)
    listnames_122837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 12), 'listnames', False)
    # Obtaining the member 'extend' of a type (line 131)
    extend_122838 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 12), listnames_122837, 'extend')
    # Calling extend(args, kwargs) (line 131)
    extend_call_result_122844 = invoke(stypy.reporting.localization.Localization(__file__, 131, 12), extend_122838, *[get_names_flat_call_result_122842], **kwargs_122843)
    
    # SSA join for if statement (line 130)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Evaluating a boolean operation
    
    # Call to tuple(...): (line 132)
    # Processing the call arguments (line 132)
    # Getting the type of 'listnames' (line 132)
    listnames_122846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 17), 'listnames', False)
    # Processing the call keyword arguments (line 132)
    kwargs_122847 = {}
    # Getting the type of 'tuple' (line 132)
    tuple_122845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 11), 'tuple', False)
    # Calling tuple(args, kwargs) (line 132)
    tuple_call_result_122848 = invoke(stypy.reporting.localization.Localization(__file__, 132, 11), tuple_122845, *[listnames_122846], **kwargs_122847)
    
    # Getting the type of 'None' (line 132)
    None_122849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 31), 'None')
    # Applying the binary operator 'or' (line 132)
    result_or_keyword_122850 = python_operator(stypy.reporting.localization.Localization(__file__, 132, 11), 'or', tuple_call_result_122848, None_122849)
    
    # Assigning a type to the variable 'stypy_return_type' (line 132)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 4), 'stypy_return_type', result_or_keyword_122850)
    
    # ################# End of 'get_names_flat(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_names_flat' in the type store
    # Getting the type of 'stypy_return_type' (line 104)
    stypy_return_type_122851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_122851)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_names_flat'
    return stypy_return_type_122851

# Assigning a type to the variable 'get_names_flat' (line 104)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 0), 'get_names_flat', get_names_flat)

@norecursion
def flatten_descr(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'flatten_descr'
    module_type_store = module_type_store.open_function_context('flatten_descr', 135, 0, False)
    
    # Passed parameters checking function
    flatten_descr.stypy_localization = localization
    flatten_descr.stypy_type_of_self = None
    flatten_descr.stypy_type_store = module_type_store
    flatten_descr.stypy_function_name = 'flatten_descr'
    flatten_descr.stypy_param_names_list = ['ndtype']
    flatten_descr.stypy_varargs_param_name = None
    flatten_descr.stypy_kwargs_param_name = None
    flatten_descr.stypy_call_defaults = defaults
    flatten_descr.stypy_call_varargs = varargs
    flatten_descr.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'flatten_descr', ['ndtype'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'flatten_descr', localization, ['ndtype'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'flatten_descr(...)' code ##################

    str_122852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, (-1)), 'str', "\n    Flatten a structured data-type description.\n\n    Examples\n    --------\n    >>> from numpy.lib import recfunctions as rfn\n    >>> ndtype = np.dtype([('a', '<i4'), ('b', [('ba', '<f8'), ('bb', '<i4')])])\n    >>> rfn.flatten_descr(ndtype)\n    (('a', dtype('int32')), ('ba', dtype('float64')), ('bb', dtype('int32')))\n\n    ")
    
    # Assigning a Attribute to a Name (line 147):
    
    # Assigning a Attribute to a Name (line 147):
    # Getting the type of 'ndtype' (line 147)
    ndtype_122853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 12), 'ndtype')
    # Obtaining the member 'names' of a type (line 147)
    names_122854 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 12), ndtype_122853, 'names')
    # Assigning a type to the variable 'names' (line 147)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 4), 'names', names_122854)
    
    # Type idiom detected: calculating its left and rigth part (line 148)
    # Getting the type of 'names' (line 148)
    names_122855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 7), 'names')
    # Getting the type of 'None' (line 148)
    None_122856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 16), 'None')
    
    (may_be_122857, more_types_in_union_122858) = may_be_none(names_122855, None_122856)

    if may_be_122857:

        if more_types_in_union_122858:
            # Runtime conditional SSA (line 148)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Getting the type of 'ndtype' (line 149)
        ndtype_122859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 15), 'ndtype')
        # Obtaining the member 'descr' of a type (line 149)
        descr_122860 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 15), ndtype_122859, 'descr')
        # Assigning a type to the variable 'stypy_return_type' (line 149)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'stypy_return_type', descr_122860)

        if more_types_in_union_122858:
            # Runtime conditional SSA for else branch (line 148)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_122857) or more_types_in_union_122858):
        
        # Assigning a List to a Name (line 151):
        
        # Assigning a List to a Name (line 151):
        
        # Obtaining an instance of the builtin type 'list' (line 151)
        list_122861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 151)
        
        # Assigning a type to the variable 'descr' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'descr', list_122861)
        
        # Getting the type of 'names' (line 152)
        names_122862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 21), 'names')
        # Testing the type of a for loop iterable (line 152)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 152, 8), names_122862)
        # Getting the type of the for loop variable (line 152)
        for_loop_var_122863 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 152, 8), names_122862)
        # Assigning a type to the variable 'field' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'field', for_loop_var_122863)
        # SSA begins for a for statement (line 152)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Subscript to a Tuple (line 153):
        
        # Assigning a Subscript to a Name (line 153):
        
        # Obtaining the type of the subscript
        int_122864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 12), 'int')
        
        # Obtaining the type of the subscript
        # Getting the type of 'field' (line 153)
        field_122865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 37), 'field')
        # Getting the type of 'ndtype' (line 153)
        ndtype_122866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 23), 'ndtype')
        # Obtaining the member 'fields' of a type (line 153)
        fields_122867 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 23), ndtype_122866, 'fields')
        # Obtaining the member '__getitem__' of a type (line 153)
        getitem___122868 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 23), fields_122867, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 153)
        subscript_call_result_122869 = invoke(stypy.reporting.localization.Localization(__file__, 153, 23), getitem___122868, field_122865)
        
        # Obtaining the member '__getitem__' of a type (line 153)
        getitem___122870 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 12), subscript_call_result_122869, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 153)
        subscript_call_result_122871 = invoke(stypy.reporting.localization.Localization(__file__, 153, 12), getitem___122870, int_122864)
        
        # Assigning a type to the variable 'tuple_var_assignment_122690' (line 153)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 12), 'tuple_var_assignment_122690', subscript_call_result_122871)
        
        # Assigning a Subscript to a Name (line 153):
        
        # Obtaining the type of the subscript
        int_122872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 12), 'int')
        
        # Obtaining the type of the subscript
        # Getting the type of 'field' (line 153)
        field_122873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 37), 'field')
        # Getting the type of 'ndtype' (line 153)
        ndtype_122874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 23), 'ndtype')
        # Obtaining the member 'fields' of a type (line 153)
        fields_122875 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 23), ndtype_122874, 'fields')
        # Obtaining the member '__getitem__' of a type (line 153)
        getitem___122876 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 23), fields_122875, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 153)
        subscript_call_result_122877 = invoke(stypy.reporting.localization.Localization(__file__, 153, 23), getitem___122876, field_122873)
        
        # Obtaining the member '__getitem__' of a type (line 153)
        getitem___122878 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 12), subscript_call_result_122877, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 153)
        subscript_call_result_122879 = invoke(stypy.reporting.localization.Localization(__file__, 153, 12), getitem___122878, int_122872)
        
        # Assigning a type to the variable 'tuple_var_assignment_122691' (line 153)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 12), 'tuple_var_assignment_122691', subscript_call_result_122879)
        
        # Assigning a Name to a Name (line 153):
        # Getting the type of 'tuple_var_assignment_122690' (line 153)
        tuple_var_assignment_122690_122880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 12), 'tuple_var_assignment_122690')
        # Assigning a type to the variable 'typ' (line 153)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 13), 'typ', tuple_var_assignment_122690_122880)
        
        # Assigning a Name to a Name (line 153):
        # Getting the type of 'tuple_var_assignment_122691' (line 153)
        tuple_var_assignment_122691_122881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 12), 'tuple_var_assignment_122691')
        # Assigning a type to the variable '_' (line 153)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 18), '_', tuple_var_assignment_122691_122881)
        
        # Getting the type of 'typ' (line 154)
        typ_122882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 15), 'typ')
        # Obtaining the member 'names' of a type (line 154)
        names_122883 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 15), typ_122882, 'names')
        # Testing the type of an if condition (line 154)
        if_condition_122884 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 154, 12), names_122883)
        # Assigning a type to the variable 'if_condition_122884' (line 154)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 12), 'if_condition_122884', if_condition_122884)
        # SSA begins for if statement (line 154)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to extend(...): (line 155)
        # Processing the call arguments (line 155)
        
        # Call to flatten_descr(...): (line 155)
        # Processing the call arguments (line 155)
        # Getting the type of 'typ' (line 155)
        typ_122888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 43), 'typ', False)
        # Processing the call keyword arguments (line 155)
        kwargs_122889 = {}
        # Getting the type of 'flatten_descr' (line 155)
        flatten_descr_122887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 29), 'flatten_descr', False)
        # Calling flatten_descr(args, kwargs) (line 155)
        flatten_descr_call_result_122890 = invoke(stypy.reporting.localization.Localization(__file__, 155, 29), flatten_descr_122887, *[typ_122888], **kwargs_122889)
        
        # Processing the call keyword arguments (line 155)
        kwargs_122891 = {}
        # Getting the type of 'descr' (line 155)
        descr_122885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 16), 'descr', False)
        # Obtaining the member 'extend' of a type (line 155)
        extend_122886 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 16), descr_122885, 'extend')
        # Calling extend(args, kwargs) (line 155)
        extend_call_result_122892 = invoke(stypy.reporting.localization.Localization(__file__, 155, 16), extend_122886, *[flatten_descr_call_result_122890], **kwargs_122891)
        
        # SSA branch for the else part of an if statement (line 154)
        module_type_store.open_ssa_branch('else')
        
        # Call to append(...): (line 157)
        # Processing the call arguments (line 157)
        
        # Obtaining an instance of the builtin type 'tuple' (line 157)
        tuple_122895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 30), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 157)
        # Adding element type (line 157)
        # Getting the type of 'field' (line 157)
        field_122896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 30), 'field', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 157, 30), tuple_122895, field_122896)
        # Adding element type (line 157)
        # Getting the type of 'typ' (line 157)
        typ_122897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 37), 'typ', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 157, 30), tuple_122895, typ_122897)
        
        # Processing the call keyword arguments (line 157)
        kwargs_122898 = {}
        # Getting the type of 'descr' (line 157)
        descr_122893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 16), 'descr', False)
        # Obtaining the member 'append' of a type (line 157)
        append_122894 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 16), descr_122893, 'append')
        # Calling append(args, kwargs) (line 157)
        append_call_result_122899 = invoke(stypy.reporting.localization.Localization(__file__, 157, 16), append_122894, *[tuple_122895], **kwargs_122898)
        
        # SSA join for if statement (line 154)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to tuple(...): (line 158)
        # Processing the call arguments (line 158)
        # Getting the type of 'descr' (line 158)
        descr_122901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 21), 'descr', False)
        # Processing the call keyword arguments (line 158)
        kwargs_122902 = {}
        # Getting the type of 'tuple' (line 158)
        tuple_122900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 15), 'tuple', False)
        # Calling tuple(args, kwargs) (line 158)
        tuple_call_result_122903 = invoke(stypy.reporting.localization.Localization(__file__, 158, 15), tuple_122900, *[descr_122901], **kwargs_122902)
        
        # Assigning a type to the variable 'stypy_return_type' (line 158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'stypy_return_type', tuple_call_result_122903)

        if (may_be_122857 and more_types_in_union_122858):
            # SSA join for if statement (line 148)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # ################# End of 'flatten_descr(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'flatten_descr' in the type store
    # Getting the type of 'stypy_return_type' (line 135)
    stypy_return_type_122904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_122904)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'flatten_descr'
    return stypy_return_type_122904

# Assigning a type to the variable 'flatten_descr' (line 135)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 0), 'flatten_descr', flatten_descr)

@norecursion
def zip_descr(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'False' (line 161)
    False_122905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 33), 'False')
    defaults = [False_122905]
    # Create a new context for function 'zip_descr'
    module_type_store = module_type_store.open_function_context('zip_descr', 161, 0, False)
    
    # Passed parameters checking function
    zip_descr.stypy_localization = localization
    zip_descr.stypy_type_of_self = None
    zip_descr.stypy_type_store = module_type_store
    zip_descr.stypy_function_name = 'zip_descr'
    zip_descr.stypy_param_names_list = ['seqarrays', 'flatten']
    zip_descr.stypy_varargs_param_name = None
    zip_descr.stypy_kwargs_param_name = None
    zip_descr.stypy_call_defaults = defaults
    zip_descr.stypy_call_varargs = varargs
    zip_descr.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'zip_descr', ['seqarrays', 'flatten'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'zip_descr', localization, ['seqarrays', 'flatten'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'zip_descr(...)' code ##################

    str_122906 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, (-1)), 'str', '\n    Combine the dtype description of a series of arrays.\n\n    Parameters\n    ----------\n    seqarrays : sequence of arrays\n        Sequence of arrays\n    flatten : {boolean}, optional\n        Whether to collapse nested descriptions.\n    ')
    
    # Assigning a List to a Name (line 172):
    
    # Assigning a List to a Name (line 172):
    
    # Obtaining an instance of the builtin type 'list' (line 172)
    list_122907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 172)
    
    # Assigning a type to the variable 'newdtype' (line 172)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 4), 'newdtype', list_122907)
    
    # Getting the type of 'flatten' (line 173)
    flatten_122908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 7), 'flatten')
    # Testing the type of an if condition (line 173)
    if_condition_122909 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 173, 4), flatten_122908)
    # Assigning a type to the variable 'if_condition_122909' (line 173)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 4), 'if_condition_122909', if_condition_122909)
    # SSA begins for if statement (line 173)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'seqarrays' (line 174)
    seqarrays_122910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 17), 'seqarrays')
    # Testing the type of a for loop iterable (line 174)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 174, 8), seqarrays_122910)
    # Getting the type of the for loop variable (line 174)
    for_loop_var_122911 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 174, 8), seqarrays_122910)
    # Assigning a type to the variable 'a' (line 174)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'a', for_loop_var_122911)
    # SSA begins for a for statement (line 174)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to extend(...): (line 175)
    # Processing the call arguments (line 175)
    
    # Call to flatten_descr(...): (line 175)
    # Processing the call arguments (line 175)
    # Getting the type of 'a' (line 175)
    a_122915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 42), 'a', False)
    # Obtaining the member 'dtype' of a type (line 175)
    dtype_122916 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 42), a_122915, 'dtype')
    # Processing the call keyword arguments (line 175)
    kwargs_122917 = {}
    # Getting the type of 'flatten_descr' (line 175)
    flatten_descr_122914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 28), 'flatten_descr', False)
    # Calling flatten_descr(args, kwargs) (line 175)
    flatten_descr_call_result_122918 = invoke(stypy.reporting.localization.Localization(__file__, 175, 28), flatten_descr_122914, *[dtype_122916], **kwargs_122917)
    
    # Processing the call keyword arguments (line 175)
    kwargs_122919 = {}
    # Getting the type of 'newdtype' (line 175)
    newdtype_122912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 12), 'newdtype', False)
    # Obtaining the member 'extend' of a type (line 175)
    extend_122913 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 12), newdtype_122912, 'extend')
    # Calling extend(args, kwargs) (line 175)
    extend_call_result_122920 = invoke(stypy.reporting.localization.Localization(__file__, 175, 12), extend_122913, *[flatten_descr_call_result_122918], **kwargs_122919)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 173)
    module_type_store.open_ssa_branch('else')
    
    # Getting the type of 'seqarrays' (line 177)
    seqarrays_122921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 17), 'seqarrays')
    # Testing the type of a for loop iterable (line 177)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 177, 8), seqarrays_122921)
    # Getting the type of the for loop variable (line 177)
    for_loop_var_122922 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 177, 8), seqarrays_122921)
    # Assigning a type to the variable 'a' (line 177)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'a', for_loop_var_122922)
    # SSA begins for a for statement (line 177)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Attribute to a Name (line 178):
    
    # Assigning a Attribute to a Name (line 178):
    # Getting the type of 'a' (line 178)
    a_122923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 22), 'a')
    # Obtaining the member 'dtype' of a type (line 178)
    dtype_122924 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 22), a_122923, 'dtype')
    # Assigning a type to the variable 'current' (line 178)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 12), 'current', dtype_122924)
    
    # Assigning a BoolOp to a Name (line 179):
    
    # Assigning a BoolOp to a Name (line 179):
    
    # Evaluating a boolean operation
    # Getting the type of 'current' (line 179)
    current_122925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 20), 'current')
    # Obtaining the member 'names' of a type (line 179)
    names_122926 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 20), current_122925, 'names')
    
    # Obtaining an instance of the builtin type 'tuple' (line 179)
    tuple_122927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 37), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 179)
    
    # Applying the binary operator 'or' (line 179)
    result_or_keyword_122928 = python_operator(stypy.reporting.localization.Localization(__file__, 179, 20), 'or', names_122926, tuple_122927)
    
    # Assigning a type to the variable 'names' (line 179)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 12), 'names', result_or_keyword_122928)
    
    
    
    # Call to len(...): (line 180)
    # Processing the call arguments (line 180)
    # Getting the type of 'names' (line 180)
    names_122930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 19), 'names', False)
    # Processing the call keyword arguments (line 180)
    kwargs_122931 = {}
    # Getting the type of 'len' (line 180)
    len_122929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 15), 'len', False)
    # Calling len(args, kwargs) (line 180)
    len_call_result_122932 = invoke(stypy.reporting.localization.Localization(__file__, 180, 15), len_122929, *[names_122930], **kwargs_122931)
    
    int_122933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 28), 'int')
    # Applying the binary operator '>' (line 180)
    result_gt_122934 = python_operator(stypy.reporting.localization.Localization(__file__, 180, 15), '>', len_call_result_122932, int_122933)
    
    # Testing the type of an if condition (line 180)
    if_condition_122935 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 180, 12), result_gt_122934)
    # Assigning a type to the variable 'if_condition_122935' (line 180)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 12), 'if_condition_122935', if_condition_122935)
    # SSA begins for if statement (line 180)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 181)
    # Processing the call arguments (line 181)
    
    # Obtaining an instance of the builtin type 'tuple' (line 181)
    tuple_122938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 33), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 181)
    # Adding element type (line 181)
    str_122939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 33), 'str', '')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 181, 33), tuple_122938, str_122939)
    # Adding element type (line 181)
    # Getting the type of 'current' (line 181)
    current_122940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 37), 'current', False)
    # Obtaining the member 'descr' of a type (line 181)
    descr_122941 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 37), current_122940, 'descr')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 181, 33), tuple_122938, descr_122941)
    
    # Processing the call keyword arguments (line 181)
    kwargs_122942 = {}
    # Getting the type of 'newdtype' (line 181)
    newdtype_122936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 16), 'newdtype', False)
    # Obtaining the member 'append' of a type (line 181)
    append_122937 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 16), newdtype_122936, 'append')
    # Calling append(args, kwargs) (line 181)
    append_call_result_122943 = invoke(stypy.reporting.localization.Localization(__file__, 181, 16), append_122937, *[tuple_122938], **kwargs_122942)
    
    # SSA branch for the else part of an if statement (line 180)
    module_type_store.open_ssa_branch('else')
    
    # Call to extend(...): (line 183)
    # Processing the call arguments (line 183)
    # Getting the type of 'current' (line 183)
    current_122946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 32), 'current', False)
    # Obtaining the member 'descr' of a type (line 183)
    descr_122947 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 32), current_122946, 'descr')
    # Processing the call keyword arguments (line 183)
    kwargs_122948 = {}
    # Getting the type of 'newdtype' (line 183)
    newdtype_122944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 16), 'newdtype', False)
    # Obtaining the member 'extend' of a type (line 183)
    extend_122945 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 16), newdtype_122944, 'extend')
    # Calling extend(args, kwargs) (line 183)
    extend_call_result_122949 = invoke(stypy.reporting.localization.Localization(__file__, 183, 16), extend_122945, *[descr_122947], **kwargs_122948)
    
    # SSA join for if statement (line 180)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 173)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to dtype(...): (line 184)
    # Processing the call arguments (line 184)
    # Getting the type of 'newdtype' (line 184)
    newdtype_122952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 20), 'newdtype', False)
    # Processing the call keyword arguments (line 184)
    kwargs_122953 = {}
    # Getting the type of 'np' (line 184)
    np_122950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 11), 'np', False)
    # Obtaining the member 'dtype' of a type (line 184)
    dtype_122951 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 11), np_122950, 'dtype')
    # Calling dtype(args, kwargs) (line 184)
    dtype_call_result_122954 = invoke(stypy.reporting.localization.Localization(__file__, 184, 11), dtype_122951, *[newdtype_122952], **kwargs_122953)
    
    # Obtaining the member 'descr' of a type (line 184)
    descr_122955 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 11), dtype_call_result_122954, 'descr')
    # Assigning a type to the variable 'stypy_return_type' (line 184)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 4), 'stypy_return_type', descr_122955)
    
    # ################# End of 'zip_descr(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'zip_descr' in the type store
    # Getting the type of 'stypy_return_type' (line 161)
    stypy_return_type_122956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_122956)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'zip_descr'
    return stypy_return_type_122956

# Assigning a type to the variable 'zip_descr' (line 161)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 0), 'zip_descr', zip_descr)

@norecursion
def get_fieldstructure(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 187)
    None_122957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 40), 'None')
    # Getting the type of 'None' (line 187)
    None_122958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 54), 'None')
    defaults = [None_122957, None_122958]
    # Create a new context for function 'get_fieldstructure'
    module_type_store = module_type_store.open_function_context('get_fieldstructure', 187, 0, False)
    
    # Passed parameters checking function
    get_fieldstructure.stypy_localization = localization
    get_fieldstructure.stypy_type_of_self = None
    get_fieldstructure.stypy_type_store = module_type_store
    get_fieldstructure.stypy_function_name = 'get_fieldstructure'
    get_fieldstructure.stypy_param_names_list = ['adtype', 'lastname', 'parents']
    get_fieldstructure.stypy_varargs_param_name = None
    get_fieldstructure.stypy_kwargs_param_name = None
    get_fieldstructure.stypy_call_defaults = defaults
    get_fieldstructure.stypy_call_varargs = varargs
    get_fieldstructure.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'get_fieldstructure', ['adtype', 'lastname', 'parents'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'get_fieldstructure', localization, ['adtype', 'lastname', 'parents'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'get_fieldstructure(...)' code ##################

    str_122959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, (-1)), 'str', "\n    Returns a dictionary with fields indexing lists of their parent fields.\n\n    This function is used to simplify access to fields nested in other fields.\n\n    Parameters\n    ----------\n    adtype : np.dtype\n        Input datatype\n    lastname : optional\n        Last processed field name (used internally during recursion).\n    parents : dictionary\n        Dictionary of parent fields (used interbally during recursion).\n\n    Examples\n    --------\n    >>> from numpy.lib import recfunctions as rfn\n    >>> ndtype =  np.dtype([('A', int),\n    ...                     ('B', [('BA', int),\n    ...                            ('BB', [('BBA', int), ('BBB', int)])])])\n    >>> rfn.get_fieldstructure(ndtype)\n    ... # XXX: possible regression, order of BBA and BBB is swapped\n    {'A': [], 'B': [], 'BA': ['B'], 'BB': ['B'], 'BBA': ['B', 'BB'], 'BBB': ['B', 'BB']}\n\n    ")
    
    # Type idiom detected: calculating its left and rigth part (line 213)
    # Getting the type of 'parents' (line 213)
    parents_122960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 7), 'parents')
    # Getting the type of 'None' (line 213)
    None_122961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 18), 'None')
    
    (may_be_122962, more_types_in_union_122963) = may_be_none(parents_122960, None_122961)

    if may_be_122962:

        if more_types_in_union_122963:
            # Runtime conditional SSA (line 213)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Dict to a Name (line 214):
        
        # Assigning a Dict to a Name (line 214):
        
        # Obtaining an instance of the builtin type 'dict' (line 214)
        dict_122964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 18), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 214)
        
        # Assigning a type to the variable 'parents' (line 214)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 8), 'parents', dict_122964)

        if more_types_in_union_122963:
            # SSA join for if statement (line 213)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Attribute to a Name (line 215):
    
    # Assigning a Attribute to a Name (line 215):
    # Getting the type of 'adtype' (line 215)
    adtype_122965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 12), 'adtype')
    # Obtaining the member 'names' of a type (line 215)
    names_122966 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 12), adtype_122965, 'names')
    # Assigning a type to the variable 'names' (line 215)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 4), 'names', names_122966)
    
    # Getting the type of 'names' (line 216)
    names_122967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 16), 'names')
    # Testing the type of a for loop iterable (line 216)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 216, 4), names_122967)
    # Getting the type of the for loop variable (line 216)
    for_loop_var_122968 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 216, 4), names_122967)
    # Assigning a type to the variable 'name' (line 216)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 4), 'name', for_loop_var_122968)
    # SSA begins for a for statement (line 216)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Subscript to a Name (line 217):
    
    # Assigning a Subscript to a Name (line 217):
    
    # Obtaining the type of the subscript
    # Getting the type of 'name' (line 217)
    name_122969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 25), 'name')
    # Getting the type of 'adtype' (line 217)
    adtype_122970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 18), 'adtype')
    # Obtaining the member '__getitem__' of a type (line 217)
    getitem___122971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 18), adtype_122970, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 217)
    subscript_call_result_122972 = invoke(stypy.reporting.localization.Localization(__file__, 217, 18), getitem___122971, name_122969)
    
    # Assigning a type to the variable 'current' (line 217)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 8), 'current', subscript_call_result_122972)
    
    # Getting the type of 'current' (line 218)
    current_122973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 11), 'current')
    # Obtaining the member 'names' of a type (line 218)
    names_122974 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 11), current_122973, 'names')
    # Testing the type of an if condition (line 218)
    if_condition_122975 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 218, 8), names_122974)
    # Assigning a type to the variable 'if_condition_122975' (line 218)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 8), 'if_condition_122975', if_condition_122975)
    # SSA begins for if statement (line 218)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'lastname' (line 219)
    lastname_122976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 15), 'lastname')
    # Testing the type of an if condition (line 219)
    if_condition_122977 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 219, 12), lastname_122976)
    # Assigning a type to the variable 'if_condition_122977' (line 219)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 12), 'if_condition_122977', if_condition_122977)
    # SSA begins for if statement (line 219)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a List to a Subscript (line 220):
    
    # Assigning a List to a Subscript (line 220):
    
    # Obtaining an instance of the builtin type 'list' (line 220)
    list_122978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 32), 'list')
    # Adding type elements to the builtin type 'list' instance (line 220)
    # Adding element type (line 220)
    # Getting the type of 'lastname' (line 220)
    lastname_122979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 33), 'lastname')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 220, 32), list_122978, lastname_122979)
    
    # Getting the type of 'parents' (line 220)
    parents_122980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 16), 'parents')
    # Getting the type of 'name' (line 220)
    name_122981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 24), 'name')
    # Storing an element on a container (line 220)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 220, 16), parents_122980, (name_122981, list_122978))
    # SSA branch for the else part of an if statement (line 219)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a List to a Subscript (line 222):
    
    # Assigning a List to a Subscript (line 222):
    
    # Obtaining an instance of the builtin type 'list' (line 222)
    list_122982 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 32), 'list')
    # Adding type elements to the builtin type 'list' instance (line 222)
    
    # Getting the type of 'parents' (line 222)
    parents_122983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 16), 'parents')
    # Getting the type of 'name' (line 222)
    name_122984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 24), 'name')
    # Storing an element on a container (line 222)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 222, 16), parents_122983, (name_122984, list_122982))
    # SSA join for if statement (line 219)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to update(...): (line 223)
    # Processing the call arguments (line 223)
    
    # Call to get_fieldstructure(...): (line 223)
    # Processing the call arguments (line 223)
    # Getting the type of 'current' (line 223)
    current_122988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 46), 'current', False)
    # Getting the type of 'name' (line 223)
    name_122989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 55), 'name', False)
    # Getting the type of 'parents' (line 223)
    parents_122990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 61), 'parents', False)
    # Processing the call keyword arguments (line 223)
    kwargs_122991 = {}
    # Getting the type of 'get_fieldstructure' (line 223)
    get_fieldstructure_122987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 27), 'get_fieldstructure', False)
    # Calling get_fieldstructure(args, kwargs) (line 223)
    get_fieldstructure_call_result_122992 = invoke(stypy.reporting.localization.Localization(__file__, 223, 27), get_fieldstructure_122987, *[current_122988, name_122989, parents_122990], **kwargs_122991)
    
    # Processing the call keyword arguments (line 223)
    kwargs_122993 = {}
    # Getting the type of 'parents' (line 223)
    parents_122985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 12), 'parents', False)
    # Obtaining the member 'update' of a type (line 223)
    update_122986 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 12), parents_122985, 'update')
    # Calling update(args, kwargs) (line 223)
    update_call_result_122994 = invoke(stypy.reporting.localization.Localization(__file__, 223, 12), update_122986, *[get_fieldstructure_call_result_122992], **kwargs_122993)
    
    # SSA branch for the else part of an if statement (line 218)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a ListComp to a Name (line 225):
    
    # Assigning a ListComp to a Name (line 225):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Evaluating a boolean operation
    
    # Call to get(...): (line 225)
    # Processing the call arguments (line 225)
    # Getting the type of 'lastname' (line 225)
    lastname_122998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 50), 'lastname', False)
    
    # Obtaining an instance of the builtin type 'list' (line 225)
    list_122999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 60), 'list')
    # Adding type elements to the builtin type 'list' instance (line 225)
    
    # Processing the call keyword arguments (line 225)
    kwargs_123000 = {}
    # Getting the type of 'parents' (line 225)
    parents_122996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 38), 'parents', False)
    # Obtaining the member 'get' of a type (line 225)
    get_122997 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 38), parents_122996, 'get')
    # Calling get(args, kwargs) (line 225)
    get_call_result_123001 = invoke(stypy.reporting.localization.Localization(__file__, 225, 38), get_122997, *[lastname_122998, list_122999], **kwargs_123000)
    
    
    # Obtaining an instance of the builtin type 'list' (line 225)
    list_123002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 67), 'list')
    # Adding type elements to the builtin type 'list' instance (line 225)
    
    # Applying the binary operator 'or' (line 225)
    result_or_keyword_123003 = python_operator(stypy.reporting.localization.Localization(__file__, 225, 38), 'or', get_call_result_123001, list_123002)
    
    comprehension_123004 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 225, 26), result_or_keyword_123003)
    # Assigning a type to the variable '_' (line 225)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 26), '_', comprehension_123004)
    # Getting the type of '_' (line 225)
    __122995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 26), '_')
    list_123005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 26), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 225, 26), list_123005, __122995)
    # Assigning a type to the variable 'lastparent' (line 225)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 12), 'lastparent', list_123005)
    
    # Getting the type of 'lastparent' (line 226)
    lastparent_123006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 15), 'lastparent')
    # Testing the type of an if condition (line 226)
    if_condition_123007 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 226, 12), lastparent_123006)
    # Assigning a type to the variable 'if_condition_123007' (line 226)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 12), 'if_condition_123007', if_condition_123007)
    # SSA begins for if statement (line 226)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 227)
    # Processing the call arguments (line 227)
    # Getting the type of 'lastname' (line 227)
    lastname_123010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 34), 'lastname', False)
    # Processing the call keyword arguments (line 227)
    kwargs_123011 = {}
    # Getting the type of 'lastparent' (line 227)
    lastparent_123008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 16), 'lastparent', False)
    # Obtaining the member 'append' of a type (line 227)
    append_123009 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 16), lastparent_123008, 'append')
    # Calling append(args, kwargs) (line 227)
    append_call_result_123012 = invoke(stypy.reporting.localization.Localization(__file__, 227, 16), append_123009, *[lastname_123010], **kwargs_123011)
    
    # SSA branch for the else part of an if statement (line 226)
    module_type_store.open_ssa_branch('else')
    
    # Getting the type of 'lastname' (line 228)
    lastname_123013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 17), 'lastname')
    # Testing the type of an if condition (line 228)
    if_condition_123014 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 228, 17), lastname_123013)
    # Assigning a type to the variable 'if_condition_123014' (line 228)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 17), 'if_condition_123014', if_condition_123014)
    # SSA begins for if statement (line 228)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a List to a Name (line 229):
    
    # Assigning a List to a Name (line 229):
    
    # Obtaining an instance of the builtin type 'list' (line 229)
    list_123015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 29), 'list')
    # Adding type elements to the builtin type 'list' instance (line 229)
    # Adding element type (line 229)
    # Getting the type of 'lastname' (line 229)
    lastname_123016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 30), 'lastname')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 229, 29), list_123015, lastname_123016)
    
    # Assigning a type to the variable 'lastparent' (line 229)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 16), 'lastparent', list_123015)
    # SSA join for if statement (line 228)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 226)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BoolOp to a Subscript (line 230):
    
    # Assigning a BoolOp to a Subscript (line 230):
    
    # Evaluating a boolean operation
    # Getting the type of 'lastparent' (line 230)
    lastparent_123017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 28), 'lastparent')
    
    # Obtaining an instance of the builtin type 'list' (line 230)
    list_123018 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 42), 'list')
    # Adding type elements to the builtin type 'list' instance (line 230)
    
    # Applying the binary operator 'or' (line 230)
    result_or_keyword_123019 = python_operator(stypy.reporting.localization.Localization(__file__, 230, 28), 'or', lastparent_123017, list_123018)
    
    # Getting the type of 'parents' (line 230)
    parents_123020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 12), 'parents')
    # Getting the type of 'name' (line 230)
    name_123021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 20), 'name')
    # Storing an element on a container (line 230)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 230, 12), parents_123020, (name_123021, result_or_keyword_123019))
    # SSA join for if statement (line 218)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Evaluating a boolean operation
    # Getting the type of 'parents' (line 231)
    parents_123022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 11), 'parents')
    # Getting the type of 'None' (line 231)
    None_123023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 22), 'None')
    # Applying the binary operator 'or' (line 231)
    result_or_keyword_123024 = python_operator(stypy.reporting.localization.Localization(__file__, 231, 11), 'or', parents_123022, None_123023)
    
    # Assigning a type to the variable 'stypy_return_type' (line 231)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 4), 'stypy_return_type', result_or_keyword_123024)
    
    # ################# End of 'get_fieldstructure(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_fieldstructure' in the type store
    # Getting the type of 'stypy_return_type' (line 187)
    stypy_return_type_123025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_123025)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_fieldstructure'
    return stypy_return_type_123025

# Assigning a type to the variable 'get_fieldstructure' (line 187)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 0), 'get_fieldstructure', get_fieldstructure)

@norecursion
def _izip_fields_flat(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_izip_fields_flat'
    module_type_store = module_type_store.open_function_context('_izip_fields_flat', 234, 0, False)
    
    # Passed parameters checking function
    _izip_fields_flat.stypy_localization = localization
    _izip_fields_flat.stypy_type_of_self = None
    _izip_fields_flat.stypy_type_store = module_type_store
    _izip_fields_flat.stypy_function_name = '_izip_fields_flat'
    _izip_fields_flat.stypy_param_names_list = ['iterable']
    _izip_fields_flat.stypy_varargs_param_name = None
    _izip_fields_flat.stypy_kwargs_param_name = None
    _izip_fields_flat.stypy_call_defaults = defaults
    _izip_fields_flat.stypy_call_varargs = varargs
    _izip_fields_flat.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_izip_fields_flat', ['iterable'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_izip_fields_flat', localization, ['iterable'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_izip_fields_flat(...)' code ##################

    str_123026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, (-1)), 'str', '\n    Returns an iterator of concatenated fields from a sequence of arrays,\n    collapsing any nested structure.\n\n    ')
    
    # Getting the type of 'iterable' (line 240)
    iterable_123027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 19), 'iterable')
    # Testing the type of a for loop iterable (line 240)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 240, 4), iterable_123027)
    # Getting the type of the for loop variable (line 240)
    for_loop_var_123028 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 240, 4), iterable_123027)
    # Assigning a type to the variable 'element' (line 240)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 4), 'element', for_loop_var_123028)
    # SSA begins for a for statement (line 240)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Call to isinstance(...): (line 241)
    # Processing the call arguments (line 241)
    # Getting the type of 'element' (line 241)
    element_123030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 22), 'element', False)
    # Getting the type of 'np' (line 241)
    np_123031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 31), 'np', False)
    # Obtaining the member 'void' of a type (line 241)
    void_123032 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 31), np_123031, 'void')
    # Processing the call keyword arguments (line 241)
    kwargs_123033 = {}
    # Getting the type of 'isinstance' (line 241)
    isinstance_123029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 11), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 241)
    isinstance_call_result_123034 = invoke(stypy.reporting.localization.Localization(__file__, 241, 11), isinstance_123029, *[element_123030, void_123032], **kwargs_123033)
    
    # Testing the type of an if condition (line 241)
    if_condition_123035 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 241, 8), isinstance_call_result_123034)
    # Assigning a type to the variable 'if_condition_123035' (line 241)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 8), 'if_condition_123035', if_condition_123035)
    # SSA begins for if statement (line 241)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Call to _izip_fields_flat(...): (line 242)
    # Processing the call arguments (line 242)
    
    # Call to tuple(...): (line 242)
    # Processing the call arguments (line 242)
    # Getting the type of 'element' (line 242)
    element_123038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 45), 'element', False)
    # Processing the call keyword arguments (line 242)
    kwargs_123039 = {}
    # Getting the type of 'tuple' (line 242)
    tuple_123037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 39), 'tuple', False)
    # Calling tuple(args, kwargs) (line 242)
    tuple_call_result_123040 = invoke(stypy.reporting.localization.Localization(__file__, 242, 39), tuple_123037, *[element_123038], **kwargs_123039)
    
    # Processing the call keyword arguments (line 242)
    kwargs_123041 = {}
    # Getting the type of '_izip_fields_flat' (line 242)
    _izip_fields_flat_123036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 21), '_izip_fields_flat', False)
    # Calling _izip_fields_flat(args, kwargs) (line 242)
    _izip_fields_flat_call_result_123042 = invoke(stypy.reporting.localization.Localization(__file__, 242, 21), _izip_fields_flat_123036, *[tuple_call_result_123040], **kwargs_123041)
    
    # Testing the type of a for loop iterable (line 242)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 242, 12), _izip_fields_flat_call_result_123042)
    # Getting the type of the for loop variable (line 242)
    for_loop_var_123043 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 242, 12), _izip_fields_flat_call_result_123042)
    # Assigning a type to the variable 'f' (line 242)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 12), 'f', for_loop_var_123043)
    # SSA begins for a for statement (line 242)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    # Creating a generator
    # Getting the type of 'f' (line 243)
    f_123044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 22), 'f')
    GeneratorType_123045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 16), 'GeneratorType')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 243, 16), GeneratorType_123045, f_123044)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 16), 'stypy_return_type', GeneratorType_123045)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 241)
    module_type_store.open_ssa_branch('else')
    # Creating a generator
    # Getting the type of 'element' (line 245)
    element_123046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 18), 'element')
    GeneratorType_123047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 12), 'GeneratorType')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 245, 12), GeneratorType_123047, element_123046)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 12), 'stypy_return_type', GeneratorType_123047)
    # SSA join for if statement (line 241)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '_izip_fields_flat(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_izip_fields_flat' in the type store
    # Getting the type of 'stypy_return_type' (line 234)
    stypy_return_type_123048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_123048)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_izip_fields_flat'
    return stypy_return_type_123048

# Assigning a type to the variable '_izip_fields_flat' (line 234)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 0), '_izip_fields_flat', _izip_fields_flat)

@norecursion
def _izip_fields(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_izip_fields'
    module_type_store = module_type_store.open_function_context('_izip_fields', 248, 0, False)
    
    # Passed parameters checking function
    _izip_fields.stypy_localization = localization
    _izip_fields.stypy_type_of_self = None
    _izip_fields.stypy_type_store = module_type_store
    _izip_fields.stypy_function_name = '_izip_fields'
    _izip_fields.stypy_param_names_list = ['iterable']
    _izip_fields.stypy_varargs_param_name = None
    _izip_fields.stypy_kwargs_param_name = None
    _izip_fields.stypy_call_defaults = defaults
    _izip_fields.stypy_call_varargs = varargs
    _izip_fields.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_izip_fields', ['iterable'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_izip_fields', localization, ['iterable'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_izip_fields(...)' code ##################

    str_123049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, (-1)), 'str', '\n    Returns an iterator of concatenated fields from a sequence of arrays.\n\n    ')
    
    # Getting the type of 'iterable' (line 253)
    iterable_123050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 19), 'iterable')
    # Testing the type of a for loop iterable (line 253)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 253, 4), iterable_123050)
    # Getting the type of the for loop variable (line 253)
    for_loop_var_123051 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 253, 4), iterable_123050)
    # Assigning a type to the variable 'element' (line 253)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 4), 'element', for_loop_var_123051)
    # SSA begins for a for statement (line 253)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Evaluating a boolean operation
    
    # Call to hasattr(...): (line 254)
    # Processing the call arguments (line 254)
    # Getting the type of 'element' (line 254)
    element_123053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 20), 'element', False)
    str_123054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 29), 'str', '__iter__')
    # Processing the call keyword arguments (line 254)
    kwargs_123055 = {}
    # Getting the type of 'hasattr' (line 254)
    hasattr_123052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 12), 'hasattr', False)
    # Calling hasattr(args, kwargs) (line 254)
    hasattr_call_result_123056 = invoke(stypy.reporting.localization.Localization(__file__, 254, 12), hasattr_123052, *[element_123053, str_123054], **kwargs_123055)
    
    
    
    # Call to isinstance(...): (line 255)
    # Processing the call arguments (line 255)
    # Getting the type of 'element' (line 255)
    element_123058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 31), 'element', False)
    # Getting the type of 'basestring' (line 255)
    basestring_123059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 40), 'basestring', False)
    # Processing the call keyword arguments (line 255)
    kwargs_123060 = {}
    # Getting the type of 'isinstance' (line 255)
    isinstance_123057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 20), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 255)
    isinstance_call_result_123061 = invoke(stypy.reporting.localization.Localization(__file__, 255, 20), isinstance_123057, *[element_123058, basestring_123059], **kwargs_123060)
    
    # Applying the 'not' unary operator (line 255)
    result_not__123062 = python_operator(stypy.reporting.localization.Localization(__file__, 255, 16), 'not', isinstance_call_result_123061)
    
    # Applying the binary operator 'and' (line 254)
    result_and_keyword_123063 = python_operator(stypy.reporting.localization.Localization(__file__, 254, 12), 'and', hasattr_call_result_123056, result_not__123062)
    
    # Testing the type of an if condition (line 254)
    if_condition_123064 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 254, 8), result_and_keyword_123063)
    # Assigning a type to the variable 'if_condition_123064' (line 254)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 8), 'if_condition_123064', if_condition_123064)
    # SSA begins for if statement (line 254)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Call to _izip_fields(...): (line 256)
    # Processing the call arguments (line 256)
    # Getting the type of 'element' (line 256)
    element_123066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 34), 'element', False)
    # Processing the call keyword arguments (line 256)
    kwargs_123067 = {}
    # Getting the type of '_izip_fields' (line 256)
    _izip_fields_123065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 21), '_izip_fields', False)
    # Calling _izip_fields(args, kwargs) (line 256)
    _izip_fields_call_result_123068 = invoke(stypy.reporting.localization.Localization(__file__, 256, 21), _izip_fields_123065, *[element_123066], **kwargs_123067)
    
    # Testing the type of a for loop iterable (line 256)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 256, 12), _izip_fields_call_result_123068)
    # Getting the type of the for loop variable (line 256)
    for_loop_var_123069 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 256, 12), _izip_fields_call_result_123068)
    # Assigning a type to the variable 'f' (line 256)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 12), 'f', for_loop_var_123069)
    # SSA begins for a for statement (line 256)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    # Creating a generator
    # Getting the type of 'f' (line 257)
    f_123070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 22), 'f')
    GeneratorType_123071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 16), 'GeneratorType')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 16), GeneratorType_123071, f_123070)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 16), 'stypy_return_type', GeneratorType_123071)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 254)
    module_type_store.open_ssa_branch('else')
    
    
    # Evaluating a boolean operation
    
    # Call to isinstance(...): (line 258)
    # Processing the call arguments (line 258)
    # Getting the type of 'element' (line 258)
    element_123073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 24), 'element', False)
    # Getting the type of 'np' (line 258)
    np_123074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 33), 'np', False)
    # Obtaining the member 'void' of a type (line 258)
    void_123075 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 33), np_123074, 'void')
    # Processing the call keyword arguments (line 258)
    kwargs_123076 = {}
    # Getting the type of 'isinstance' (line 258)
    isinstance_123072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 13), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 258)
    isinstance_call_result_123077 = invoke(stypy.reporting.localization.Localization(__file__, 258, 13), isinstance_123072, *[element_123073, void_123075], **kwargs_123076)
    
    
    
    # Call to len(...): (line 258)
    # Processing the call arguments (line 258)
    
    # Call to tuple(...): (line 258)
    # Processing the call arguments (line 258)
    # Getting the type of 'element' (line 258)
    element_123080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 56), 'element', False)
    # Processing the call keyword arguments (line 258)
    kwargs_123081 = {}
    # Getting the type of 'tuple' (line 258)
    tuple_123079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 50), 'tuple', False)
    # Calling tuple(args, kwargs) (line 258)
    tuple_call_result_123082 = invoke(stypy.reporting.localization.Localization(__file__, 258, 50), tuple_123079, *[element_123080], **kwargs_123081)
    
    # Processing the call keyword arguments (line 258)
    kwargs_123083 = {}
    # Getting the type of 'len' (line 258)
    len_123078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 46), 'len', False)
    # Calling len(args, kwargs) (line 258)
    len_call_result_123084 = invoke(stypy.reporting.localization.Localization(__file__, 258, 46), len_123078, *[tuple_call_result_123082], **kwargs_123083)
    
    int_123085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 69), 'int')
    # Applying the binary operator '==' (line 258)
    result_eq_123086 = python_operator(stypy.reporting.localization.Localization(__file__, 258, 46), '==', len_call_result_123084, int_123085)
    
    # Applying the binary operator 'and' (line 258)
    result_and_keyword_123087 = python_operator(stypy.reporting.localization.Localization(__file__, 258, 13), 'and', isinstance_call_result_123077, result_eq_123086)
    
    # Testing the type of an if condition (line 258)
    if_condition_123088 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 258, 13), result_and_keyword_123087)
    # Assigning a type to the variable 'if_condition_123088' (line 258)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 13), 'if_condition_123088', if_condition_123088)
    # SSA begins for if statement (line 258)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Call to _izip_fields(...): (line 259)
    # Processing the call arguments (line 259)
    # Getting the type of 'element' (line 259)
    element_123090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 34), 'element', False)
    # Processing the call keyword arguments (line 259)
    kwargs_123091 = {}
    # Getting the type of '_izip_fields' (line 259)
    _izip_fields_123089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 21), '_izip_fields', False)
    # Calling _izip_fields(args, kwargs) (line 259)
    _izip_fields_call_result_123092 = invoke(stypy.reporting.localization.Localization(__file__, 259, 21), _izip_fields_123089, *[element_123090], **kwargs_123091)
    
    # Testing the type of a for loop iterable (line 259)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 259, 12), _izip_fields_call_result_123092)
    # Getting the type of the for loop variable (line 259)
    for_loop_var_123093 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 259, 12), _izip_fields_call_result_123092)
    # Assigning a type to the variable 'f' (line 259)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 12), 'f', for_loop_var_123093)
    # SSA begins for a for statement (line 259)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    # Creating a generator
    # Getting the type of 'f' (line 260)
    f_123094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 22), 'f')
    GeneratorType_123095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 16), 'GeneratorType')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 260, 16), GeneratorType_123095, f_123094)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 16), 'stypy_return_type', GeneratorType_123095)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 258)
    module_type_store.open_ssa_branch('else')
    # Creating a generator
    # Getting the type of 'element' (line 262)
    element_123096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 18), 'element')
    GeneratorType_123097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 12), 'GeneratorType')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 262, 12), GeneratorType_123097, element_123096)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 12), 'stypy_return_type', GeneratorType_123097)
    # SSA join for if statement (line 258)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 254)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '_izip_fields(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_izip_fields' in the type store
    # Getting the type of 'stypy_return_type' (line 248)
    stypy_return_type_123098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_123098)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_izip_fields'
    return stypy_return_type_123098

# Assigning a type to the variable '_izip_fields' (line 248)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 0), '_izip_fields', _izip_fields)

@norecursion
def izip_records(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 265)
    None_123099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 39), 'None')
    # Getting the type of 'True' (line 265)
    True_123100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 53), 'True')
    defaults = [None_123099, True_123100]
    # Create a new context for function 'izip_records'
    module_type_store = module_type_store.open_function_context('izip_records', 265, 0, False)
    
    # Passed parameters checking function
    izip_records.stypy_localization = localization
    izip_records.stypy_type_of_self = None
    izip_records.stypy_type_store = module_type_store
    izip_records.stypy_function_name = 'izip_records'
    izip_records.stypy_param_names_list = ['seqarrays', 'fill_value', 'flatten']
    izip_records.stypy_varargs_param_name = None
    izip_records.stypy_kwargs_param_name = None
    izip_records.stypy_call_defaults = defaults
    izip_records.stypy_call_varargs = varargs
    izip_records.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'izip_records', ['seqarrays', 'fill_value', 'flatten'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'izip_records', localization, ['seqarrays', 'fill_value', 'flatten'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'izip_records(...)' code ##################

    str_123101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, (-1)), 'str', '\n    Returns an iterator of concatenated items from a sequence of arrays.\n\n    Parameters\n    ----------\n    seqarrays : sequence of arrays\n        Sequence of arrays.\n    fill_value : {None, integer}\n        Value used to pad shorter iterables.\n    flatten : {True, False},\n        Whether to\n    ')

    @norecursion
    def sentinel(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        
        # Obtaining an instance of the builtin type 'list' (line 279)
        list_123102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 279)
        # Adding element type (line 279)
        # Getting the type of 'fill_value' (line 279)
        fill_value_123103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 27), 'fill_value')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 279, 26), list_123102, fill_value_123103)
        
        
        # Call to len(...): (line 279)
        # Processing the call arguments (line 279)
        # Getting the type of 'seqarrays' (line 279)
        seqarrays_123105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 46), 'seqarrays', False)
        # Processing the call keyword arguments (line 279)
        kwargs_123106 = {}
        # Getting the type of 'len' (line 279)
        len_123104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 42), 'len', False)
        # Calling len(args, kwargs) (line 279)
        len_call_result_123107 = invoke(stypy.reporting.localization.Localization(__file__, 279, 42), len_123104, *[seqarrays_123105], **kwargs_123106)
        
        int_123108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 59), 'int')
        # Applying the binary operator '-' (line 279)
        result_sub_123109 = python_operator(stypy.reporting.localization.Localization(__file__, 279, 42), '-', len_call_result_123107, int_123108)
        
        # Applying the binary operator '*' (line 279)
        result_mul_123110 = python_operator(stypy.reporting.localization.Localization(__file__, 279, 26), '*', list_123102, result_sub_123109)
        
        # Obtaining the member 'pop' of a type (line 279)
        pop_123111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 26), result_mul_123110, 'pop')
        defaults = [pop_123111]
        # Create a new context for function 'sentinel'
        module_type_store = module_type_store.open_function_context('sentinel', 279, 4, False)
        
        # Passed parameters checking function
        sentinel.stypy_localization = localization
        sentinel.stypy_type_of_self = None
        sentinel.stypy_type_store = module_type_store
        sentinel.stypy_function_name = 'sentinel'
        sentinel.stypy_param_names_list = ['counter']
        sentinel.stypy_varargs_param_name = None
        sentinel.stypy_kwargs_param_name = None
        sentinel.stypy_call_defaults = defaults
        sentinel.stypy_call_varargs = varargs
        sentinel.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'sentinel', ['counter'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'sentinel', localization, ['counter'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'sentinel(...)' code ##################

        str_123112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 8), 'str', 'Yields the fill_value or raises IndexError')
        # Creating a generator
        
        # Call to counter(...): (line 281)
        # Processing the call keyword arguments (line 281)
        kwargs_123114 = {}
        # Getting the type of 'counter' (line 281)
        counter_123113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 14), 'counter', False)
        # Calling counter(args, kwargs) (line 281)
        counter_call_result_123115 = invoke(stypy.reporting.localization.Localization(__file__, 281, 14), counter_123113, *[], **kwargs_123114)
        
        GeneratorType_123116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 8), 'GeneratorType')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 281, 8), GeneratorType_123116, counter_call_result_123115)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 8), 'stypy_return_type', GeneratorType_123116)
        
        # ################# End of 'sentinel(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'sentinel' in the type store
        # Getting the type of 'stypy_return_type' (line 279)
        stypy_return_type_123117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_123117)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'sentinel'
        return stypy_return_type_123117

    # Assigning a type to the variable 'sentinel' (line 279)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 4), 'sentinel', sentinel)
    
    # Assigning a Call to a Name (line 283):
    
    # Assigning a Call to a Name (line 283):
    
    # Call to repeat(...): (line 283)
    # Processing the call arguments (line 283)
    # Getting the type of 'fill_value' (line 283)
    fill_value_123120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 31), 'fill_value', False)
    # Processing the call keyword arguments (line 283)
    kwargs_123121 = {}
    # Getting the type of 'itertools' (line 283)
    itertools_123118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 14), 'itertools', False)
    # Obtaining the member 'repeat' of a type (line 283)
    repeat_123119 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 14), itertools_123118, 'repeat')
    # Calling repeat(args, kwargs) (line 283)
    repeat_call_result_123122 = invoke(stypy.reporting.localization.Localization(__file__, 283, 14), repeat_123119, *[fill_value_123120], **kwargs_123121)
    
    # Assigning a type to the variable 'fillers' (line 283)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 4), 'fillers', repeat_call_result_123122)
    
    # Assigning a ListComp to a Name (line 284):
    
    # Assigning a ListComp to a Name (line 284):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'seqarrays' (line 284)
    seqarrays_123132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 64), 'seqarrays')
    comprehension_123133 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 284, 13), seqarrays_123132)
    # Assigning a type to the variable 'it' (line 284)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 13), 'it', comprehension_123133)
    
    # Call to chain(...): (line 284)
    # Processing the call arguments (line 284)
    # Getting the type of 'it' (line 284)
    it_123125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 29), 'it', False)
    
    # Call to sentinel(...): (line 284)
    # Processing the call keyword arguments (line 284)
    kwargs_123127 = {}
    # Getting the type of 'sentinel' (line 284)
    sentinel_123126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 33), 'sentinel', False)
    # Calling sentinel(args, kwargs) (line 284)
    sentinel_call_result_123128 = invoke(stypy.reporting.localization.Localization(__file__, 284, 33), sentinel_123126, *[], **kwargs_123127)
    
    # Getting the type of 'fillers' (line 284)
    fillers_123129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 45), 'fillers', False)
    # Processing the call keyword arguments (line 284)
    kwargs_123130 = {}
    # Getting the type of 'itertools' (line 284)
    itertools_123123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 13), 'itertools', False)
    # Obtaining the member 'chain' of a type (line 284)
    chain_123124 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 13), itertools_123123, 'chain')
    # Calling chain(args, kwargs) (line 284)
    chain_call_result_123131 = invoke(stypy.reporting.localization.Localization(__file__, 284, 13), chain_123124, *[it_123125, sentinel_call_result_123128, fillers_123129], **kwargs_123130)
    
    list_123134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 13), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 284, 13), list_123134, chain_call_result_123131)
    # Assigning a type to the variable 'iters' (line 284)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 4), 'iters', list_123134)
    
    # Getting the type of 'flatten' (line 286)
    flatten_123135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 7), 'flatten')
    # Testing the type of an if condition (line 286)
    if_condition_123136 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 286, 4), flatten_123135)
    # Assigning a type to the variable 'if_condition_123136' (line 286)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 4), 'if_condition_123136', if_condition_123136)
    # SSA begins for if statement (line 286)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 287):
    
    # Assigning a Name to a Name (line 287):
    # Getting the type of '_izip_fields_flat' (line 287)
    _izip_fields_flat_123137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 18), '_izip_fields_flat')
    # Assigning a type to the variable 'zipfunc' (line 287)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 8), 'zipfunc', _izip_fields_flat_123137)
    # SSA branch for the else part of an if statement (line 286)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Name (line 289):
    
    # Assigning a Name to a Name (line 289):
    # Getting the type of '_izip_fields' (line 289)
    _izip_fields_123138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 18), '_izip_fields')
    # Assigning a type to the variable 'zipfunc' (line 289)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 8), 'zipfunc', _izip_fields_123138)
    # SSA join for if statement (line 286)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # SSA begins for try-except statement (line 291)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    
    # Call to zip(...): (line 292)
    # Getting the type of 'iters' (line 292)
    iters_123140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 24), 'iters', False)
    # Processing the call keyword arguments (line 292)
    kwargs_123141 = {}
    # Getting the type of 'zip' (line 292)
    zip_123139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 19), 'zip', False)
    # Calling zip(args, kwargs) (line 292)
    zip_call_result_123142 = invoke(stypy.reporting.localization.Localization(__file__, 292, 19), zip_123139, *[iters_123140], **kwargs_123141)
    
    # Testing the type of a for loop iterable (line 292)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 292, 8), zip_call_result_123142)
    # Getting the type of the for loop variable (line 292)
    for_loop_var_123143 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 292, 8), zip_call_result_123142)
    # Assigning a type to the variable 'tup' (line 292)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'tup', for_loop_var_123143)
    # SSA begins for a for statement (line 292)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    # Creating a generator
    
    # Call to tuple(...): (line 293)
    # Processing the call arguments (line 293)
    
    # Call to zipfunc(...): (line 293)
    # Processing the call arguments (line 293)
    # Getting the type of 'tup' (line 293)
    tup_123146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 32), 'tup', False)
    # Processing the call keyword arguments (line 293)
    kwargs_123147 = {}
    # Getting the type of 'zipfunc' (line 293)
    zipfunc_123145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 24), 'zipfunc', False)
    # Calling zipfunc(args, kwargs) (line 293)
    zipfunc_call_result_123148 = invoke(stypy.reporting.localization.Localization(__file__, 293, 24), zipfunc_123145, *[tup_123146], **kwargs_123147)
    
    # Processing the call keyword arguments (line 293)
    kwargs_123149 = {}
    # Getting the type of 'tuple' (line 293)
    tuple_123144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 18), 'tuple', False)
    # Calling tuple(args, kwargs) (line 293)
    tuple_call_result_123150 = invoke(stypy.reporting.localization.Localization(__file__, 293, 18), tuple_123144, *[zipfunc_call_result_123148], **kwargs_123149)
    
    GeneratorType_123151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 12), 'GeneratorType')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 293, 12), GeneratorType_123151, tuple_call_result_123150)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 12), 'stypy_return_type', GeneratorType_123151)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the except part of a try statement (line 291)
    # SSA branch for the except 'IndexError' branch of a try statement (line 291)
    module_type_store.open_ssa_branch('except')
    pass
    # SSA join for try-except statement (line 291)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'izip_records(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'izip_records' in the type store
    # Getting the type of 'stypy_return_type' (line 265)
    stypy_return_type_123152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_123152)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'izip_records'
    return stypy_return_type_123152

# Assigning a type to the variable 'izip_records' (line 265)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 0), 'izip_records', izip_records)

@norecursion
def _fix_output(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'True' (line 298)
    True_123153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 32), 'True')
    # Getting the type of 'False' (line 298)
    False_123154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 49), 'False')
    defaults = [True_123153, False_123154]
    # Create a new context for function '_fix_output'
    module_type_store = module_type_store.open_function_context('_fix_output', 298, 0, False)
    
    # Passed parameters checking function
    _fix_output.stypy_localization = localization
    _fix_output.stypy_type_of_self = None
    _fix_output.stypy_type_store = module_type_store
    _fix_output.stypy_function_name = '_fix_output'
    _fix_output.stypy_param_names_list = ['output', 'usemask', 'asrecarray']
    _fix_output.stypy_varargs_param_name = None
    _fix_output.stypy_kwargs_param_name = None
    _fix_output.stypy_call_defaults = defaults
    _fix_output.stypy_call_varargs = varargs
    _fix_output.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_fix_output', ['output', 'usemask', 'asrecarray'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_fix_output', localization, ['output', 'usemask', 'asrecarray'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_fix_output(...)' code ##################

    str_123155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, (-1)), 'str', '\n    Private function: return a recarray, a ndarray, a MaskedArray\n    or a MaskedRecords depending on the input parameters\n    ')
    
    
    
    # Call to isinstance(...): (line 303)
    # Processing the call arguments (line 303)
    # Getting the type of 'output' (line 303)
    output_123157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 22), 'output', False)
    # Getting the type of 'MaskedArray' (line 303)
    MaskedArray_123158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 30), 'MaskedArray', False)
    # Processing the call keyword arguments (line 303)
    kwargs_123159 = {}
    # Getting the type of 'isinstance' (line 303)
    isinstance_123156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 11), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 303)
    isinstance_call_result_123160 = invoke(stypy.reporting.localization.Localization(__file__, 303, 11), isinstance_123156, *[output_123157, MaskedArray_123158], **kwargs_123159)
    
    # Applying the 'not' unary operator (line 303)
    result_not__123161 = python_operator(stypy.reporting.localization.Localization(__file__, 303, 7), 'not', isinstance_call_result_123160)
    
    # Testing the type of an if condition (line 303)
    if_condition_123162 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 303, 4), result_not__123161)
    # Assigning a type to the variable 'if_condition_123162' (line 303)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 4), 'if_condition_123162', if_condition_123162)
    # SSA begins for if statement (line 303)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 304):
    
    # Assigning a Name to a Name (line 304):
    # Getting the type of 'False' (line 304)
    False_123163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 18), 'False')
    # Assigning a type to the variable 'usemask' (line 304)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 8), 'usemask', False_123163)
    # SSA join for if statement (line 303)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'usemask' (line 305)
    usemask_123164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 7), 'usemask')
    # Testing the type of an if condition (line 305)
    if_condition_123165 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 305, 4), usemask_123164)
    # Assigning a type to the variable 'if_condition_123165' (line 305)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 305, 4), 'if_condition_123165', if_condition_123165)
    # SSA begins for if statement (line 305)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'asrecarray' (line 306)
    asrecarray_123166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 11), 'asrecarray')
    # Testing the type of an if condition (line 306)
    if_condition_123167 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 306, 8), asrecarray_123166)
    # Assigning a type to the variable 'if_condition_123167' (line 306)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 8), 'if_condition_123167', if_condition_123167)
    # SSA begins for if statement (line 306)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 307):
    
    # Assigning a Call to a Name (line 307):
    
    # Call to view(...): (line 307)
    # Processing the call arguments (line 307)
    # Getting the type of 'MaskedRecords' (line 307)
    MaskedRecords_123170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 33), 'MaskedRecords', False)
    # Processing the call keyword arguments (line 307)
    kwargs_123171 = {}
    # Getting the type of 'output' (line 307)
    output_123168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 21), 'output', False)
    # Obtaining the member 'view' of a type (line 307)
    view_123169 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 21), output_123168, 'view')
    # Calling view(args, kwargs) (line 307)
    view_call_result_123172 = invoke(stypy.reporting.localization.Localization(__file__, 307, 21), view_123169, *[MaskedRecords_123170], **kwargs_123171)
    
    # Assigning a type to the variable 'output' (line 307)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 12), 'output', view_call_result_123172)
    # SSA join for if statement (line 306)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 305)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 309):
    
    # Assigning a Call to a Name (line 309):
    
    # Call to filled(...): (line 309)
    # Processing the call arguments (line 309)
    # Getting the type of 'output' (line 309)
    output_123175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 27), 'output', False)
    # Processing the call keyword arguments (line 309)
    kwargs_123176 = {}
    # Getting the type of 'ma' (line 309)
    ma_123173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 17), 'ma', False)
    # Obtaining the member 'filled' of a type (line 309)
    filled_123174 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 17), ma_123173, 'filled')
    # Calling filled(args, kwargs) (line 309)
    filled_call_result_123177 = invoke(stypy.reporting.localization.Localization(__file__, 309, 17), filled_123174, *[output_123175], **kwargs_123176)
    
    # Assigning a type to the variable 'output' (line 309)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 8), 'output', filled_call_result_123177)
    
    # Getting the type of 'asrecarray' (line 310)
    asrecarray_123178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 11), 'asrecarray')
    # Testing the type of an if condition (line 310)
    if_condition_123179 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 310, 8), asrecarray_123178)
    # Assigning a type to the variable 'if_condition_123179' (line 310)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 8), 'if_condition_123179', if_condition_123179)
    # SSA begins for if statement (line 310)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 311):
    
    # Assigning a Call to a Name (line 311):
    
    # Call to view(...): (line 311)
    # Processing the call arguments (line 311)
    # Getting the type of 'recarray' (line 311)
    recarray_123182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 33), 'recarray', False)
    # Processing the call keyword arguments (line 311)
    kwargs_123183 = {}
    # Getting the type of 'output' (line 311)
    output_123180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 21), 'output', False)
    # Obtaining the member 'view' of a type (line 311)
    view_123181 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 21), output_123180, 'view')
    # Calling view(args, kwargs) (line 311)
    view_call_result_123184 = invoke(stypy.reporting.localization.Localization(__file__, 311, 21), view_123181, *[recarray_123182], **kwargs_123183)
    
    # Assigning a type to the variable 'output' (line 311)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 12), 'output', view_call_result_123184)
    # SSA join for if statement (line 310)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 305)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'output' (line 312)
    output_123185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 11), 'output')
    # Assigning a type to the variable 'stypy_return_type' (line 312)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 4), 'stypy_return_type', output_123185)
    
    # ################# End of '_fix_output(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_fix_output' in the type store
    # Getting the type of 'stypy_return_type' (line 298)
    stypy_return_type_123186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_123186)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_fix_output'
    return stypy_return_type_123186

# Assigning a type to the variable '_fix_output' (line 298)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 0), '_fix_output', _fix_output)

@norecursion
def _fix_defaults(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 315)
    None_123187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 35), 'None')
    defaults = [None_123187]
    # Create a new context for function '_fix_defaults'
    module_type_store = module_type_store.open_function_context('_fix_defaults', 315, 0, False)
    
    # Passed parameters checking function
    _fix_defaults.stypy_localization = localization
    _fix_defaults.stypy_type_of_self = None
    _fix_defaults.stypy_type_store = module_type_store
    _fix_defaults.stypy_function_name = '_fix_defaults'
    _fix_defaults.stypy_param_names_list = ['output', 'defaults']
    _fix_defaults.stypy_varargs_param_name = None
    _fix_defaults.stypy_kwargs_param_name = None
    _fix_defaults.stypy_call_defaults = defaults
    _fix_defaults.stypy_call_varargs = varargs
    _fix_defaults.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_fix_defaults', ['output', 'defaults'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_fix_defaults', localization, ['output', 'defaults'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_fix_defaults(...)' code ##################

    str_123188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, (-1)), 'str', '\n    Update the fill_value and masked data of `output`\n    from the default given in a dictionary defaults.\n    ')
    
    # Assigning a Attribute to a Name (line 320):
    
    # Assigning a Attribute to a Name (line 320):
    # Getting the type of 'output' (line 320)
    output_123189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 12), 'output')
    # Obtaining the member 'dtype' of a type (line 320)
    dtype_123190 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 12), output_123189, 'dtype')
    # Obtaining the member 'names' of a type (line 320)
    names_123191 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 12), dtype_123190, 'names')
    # Assigning a type to the variable 'names' (line 320)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 320, 4), 'names', names_123191)
    
    # Assigning a Tuple to a Tuple (line 321):
    
    # Assigning a Attribute to a Name (line 321):
    # Getting the type of 'output' (line 321)
    output_123192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 32), 'output')
    # Obtaining the member 'data' of a type (line 321)
    data_123193 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 321, 32), output_123192, 'data')
    # Assigning a type to the variable 'tuple_assignment_122692' (line 321)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 321, 4), 'tuple_assignment_122692', data_123193)
    
    # Assigning a Attribute to a Name (line 321):
    # Getting the type of 'output' (line 321)
    output_123194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 45), 'output')
    # Obtaining the member 'mask' of a type (line 321)
    mask_123195 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 321, 45), output_123194, 'mask')
    # Assigning a type to the variable 'tuple_assignment_122693' (line 321)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 321, 4), 'tuple_assignment_122693', mask_123195)
    
    # Assigning a Attribute to a Name (line 321):
    # Getting the type of 'output' (line 321)
    output_123196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 58), 'output')
    # Obtaining the member 'fill_value' of a type (line 321)
    fill_value_123197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 321, 58), output_123196, 'fill_value')
    # Assigning a type to the variable 'tuple_assignment_122694' (line 321)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 321, 4), 'tuple_assignment_122694', fill_value_123197)
    
    # Assigning a Name to a Name (line 321):
    # Getting the type of 'tuple_assignment_122692' (line 321)
    tuple_assignment_122692_123198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 4), 'tuple_assignment_122692')
    # Assigning a type to the variable 'data' (line 321)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 321, 5), 'data', tuple_assignment_122692_123198)
    
    # Assigning a Name to a Name (line 321):
    # Getting the type of 'tuple_assignment_122693' (line 321)
    tuple_assignment_122693_123199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 4), 'tuple_assignment_122693')
    # Assigning a type to the variable 'mask' (line 321)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 321, 11), 'mask', tuple_assignment_122693_123199)
    
    # Assigning a Name to a Name (line 321):
    # Getting the type of 'tuple_assignment_122694' (line 321)
    tuple_assignment_122694_123200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 4), 'tuple_assignment_122694')
    # Assigning a type to the variable 'fill_value' (line 321)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 321, 17), 'fill_value', tuple_assignment_122694_123200)
    
    
    # Call to items(...): (line 322)
    # Processing the call keyword arguments (line 322)
    kwargs_123205 = {}
    
    # Evaluating a boolean operation
    # Getting the type of 'defaults' (line 322)
    defaults_123201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 19), 'defaults', False)
    
    # Obtaining an instance of the builtin type 'dict' (line 322)
    dict_123202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 31), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 322)
    
    # Applying the binary operator 'or' (line 322)
    result_or_keyword_123203 = python_operator(stypy.reporting.localization.Localization(__file__, 322, 19), 'or', defaults_123201, dict_123202)
    
    # Obtaining the member 'items' of a type (line 322)
    items_123204 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 19), result_or_keyword_123203, 'items')
    # Calling items(args, kwargs) (line 322)
    items_call_result_123206 = invoke(stypy.reporting.localization.Localization(__file__, 322, 19), items_123204, *[], **kwargs_123205)
    
    # Testing the type of a for loop iterable (line 322)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 322, 4), items_call_result_123206)
    # Getting the type of the for loop variable (line 322)
    for_loop_var_123207 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 322, 4), items_call_result_123206)
    # Assigning a type to the variable 'k' (line 322)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 322, 4), 'k', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 322, 4), for_loop_var_123207))
    # Assigning a type to the variable 'v' (line 322)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 322, 4), 'v', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 322, 4), for_loop_var_123207))
    # SSA begins for a for statement (line 322)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Getting the type of 'k' (line 323)
    k_123208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 11), 'k')
    # Getting the type of 'names' (line 323)
    names_123209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 16), 'names')
    # Applying the binary operator 'in' (line 323)
    result_contains_123210 = python_operator(stypy.reporting.localization.Localization(__file__, 323, 11), 'in', k_123208, names_123209)
    
    # Testing the type of an if condition (line 323)
    if_condition_123211 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 323, 8), result_contains_123210)
    # Assigning a type to the variable 'if_condition_123211' (line 323)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 8), 'if_condition_123211', if_condition_123211)
    # SSA begins for if statement (line 323)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Subscript (line 324):
    
    # Assigning a Name to a Subscript (line 324):
    # Getting the type of 'v' (line 324)
    v_123212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 28), 'v')
    # Getting the type of 'fill_value' (line 324)
    fill_value_123213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 12), 'fill_value')
    # Getting the type of 'k' (line 324)
    k_123214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 23), 'k')
    # Storing an element on a container (line 324)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 324, 12), fill_value_123213, (k_123214, v_123212))
    
    # Assigning a Name to a Subscript (line 325):
    
    # Assigning a Name to a Subscript (line 325):
    # Getting the type of 'v' (line 325)
    v_123215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 31), 'v')
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 325)
    k_123216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 17), 'k')
    # Getting the type of 'data' (line 325)
    data_123217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 12), 'data')
    # Obtaining the member '__getitem__' of a type (line 325)
    getitem___123218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 12), data_123217, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 325)
    subscript_call_result_123219 = invoke(stypy.reporting.localization.Localization(__file__, 325, 12), getitem___123218, k_123216)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 325)
    k_123220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 25), 'k')
    # Getting the type of 'mask' (line 325)
    mask_123221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 20), 'mask')
    # Obtaining the member '__getitem__' of a type (line 325)
    getitem___123222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 20), mask_123221, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 325)
    subscript_call_result_123223 = invoke(stypy.reporting.localization.Localization(__file__, 325, 20), getitem___123222, k_123220)
    
    # Storing an element on a container (line 325)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 325, 12), subscript_call_result_123219, (subscript_call_result_123223, v_123215))
    # SSA join for if statement (line 323)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'output' (line 326)
    output_123224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 11), 'output')
    # Assigning a type to the variable 'stypy_return_type' (line 326)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 4), 'stypy_return_type', output_123224)
    
    # ################# End of '_fix_defaults(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_fix_defaults' in the type store
    # Getting the type of 'stypy_return_type' (line 315)
    stypy_return_type_123225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_123225)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_fix_defaults'
    return stypy_return_type_123225

# Assigning a type to the variable '_fix_defaults' (line 315)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 0), '_fix_defaults', _fix_defaults)

@norecursion
def merge_arrays(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_123226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 39), 'int')
    # Getting the type of 'False' (line 329)
    False_123227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 51), 'False')
    # Getting the type of 'False' (line 330)
    False_123228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 25), 'False')
    # Getting the type of 'False' (line 330)
    False_123229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 43), 'False')
    defaults = [int_123226, False_123227, False_123228, False_123229]
    # Create a new context for function 'merge_arrays'
    module_type_store = module_type_store.open_function_context('merge_arrays', 329, 0, False)
    
    # Passed parameters checking function
    merge_arrays.stypy_localization = localization
    merge_arrays.stypy_type_of_self = None
    merge_arrays.stypy_type_store = module_type_store
    merge_arrays.stypy_function_name = 'merge_arrays'
    merge_arrays.stypy_param_names_list = ['seqarrays', 'fill_value', 'flatten', 'usemask', 'asrecarray']
    merge_arrays.stypy_varargs_param_name = None
    merge_arrays.stypy_kwargs_param_name = None
    merge_arrays.stypy_call_defaults = defaults
    merge_arrays.stypy_call_varargs = varargs
    merge_arrays.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'merge_arrays', ['seqarrays', 'fill_value', 'flatten', 'usemask', 'asrecarray'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'merge_arrays', localization, ['seqarrays', 'fill_value', 'flatten', 'usemask', 'asrecarray'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'merge_arrays(...)' code ##################

    str_123230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 376, (-1)), 'str', "\n    Merge arrays field by field.\n\n    Parameters\n    ----------\n    seqarrays : sequence of ndarrays\n        Sequence of arrays\n    fill_value : {float}, optional\n        Filling value used to pad missing data on the shorter arrays.\n    flatten : {False, True}, optional\n        Whether to collapse nested fields.\n    usemask : {False, True}, optional\n        Whether to return a masked array or not.\n    asrecarray : {False, True}, optional\n        Whether to return a recarray (MaskedRecords) or not.\n\n    Examples\n    --------\n    >>> from numpy.lib import recfunctions as rfn\n    >>> rfn.merge_arrays((np.array([1, 2]), np.array([10., 20., 30.])))\n    masked_array(data = [(1, 10.0) (2, 20.0) (--, 30.0)],\n                 mask = [(False, False) (False, False) (True, False)],\n           fill_value = (999999, 1e+20),\n                dtype = [('f0', '<i4'), ('f1', '<f8')])\n\n    >>> rfn.merge_arrays((np.array([1, 2]), np.array([10., 20., 30.])),\n    ...              usemask=False)\n    array([(1, 10.0), (2, 20.0), (-1, 30.0)],\n          dtype=[('f0', '<i4'), ('f1', '<f8')])\n    >>> rfn.merge_arrays((np.array([1, 2]).view([('a', int)]),\n    ...               np.array([10., 20., 30.])),\n    ...              usemask=False, asrecarray=True)\n    rec.array([(1, 10.0), (2, 20.0), (-1, 30.0)],\n              dtype=[('a', '<i4'), ('f1', '<f8')])\n\n    Notes\n    -----\n    * Without a mask, the missing value will be filled with something,\n    * depending on what its corresponding type:\n            -1      for integers\n            -1.0    for floating point numbers\n            '-'     for characters\n            '-1'    for strings\n            True    for boolean values\n    * XXX: I just obtained these values empirically\n    ")
    
    
    
    # Call to len(...): (line 378)
    # Processing the call arguments (line 378)
    # Getting the type of 'seqarrays' (line 378)
    seqarrays_123232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 12), 'seqarrays', False)
    # Processing the call keyword arguments (line 378)
    kwargs_123233 = {}
    # Getting the type of 'len' (line 378)
    len_123231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 8), 'len', False)
    # Calling len(args, kwargs) (line 378)
    len_call_result_123234 = invoke(stypy.reporting.localization.Localization(__file__, 378, 8), len_123231, *[seqarrays_123232], **kwargs_123233)
    
    int_123235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 378, 26), 'int')
    # Applying the binary operator '==' (line 378)
    result_eq_123236 = python_operator(stypy.reporting.localization.Localization(__file__, 378, 8), '==', len_call_result_123234, int_123235)
    
    # Testing the type of an if condition (line 378)
    if_condition_123237 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 378, 4), result_eq_123236)
    # Assigning a type to the variable 'if_condition_123237' (line 378)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 378, 4), 'if_condition_123237', if_condition_123237)
    # SSA begins for if statement (line 378)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 379):
    
    # Assigning a Call to a Name (line 379):
    
    # Call to asanyarray(...): (line 379)
    # Processing the call arguments (line 379)
    
    # Obtaining the type of the subscript
    int_123240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 379, 44), 'int')
    # Getting the type of 'seqarrays' (line 379)
    seqarrays_123241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 34), 'seqarrays', False)
    # Obtaining the member '__getitem__' of a type (line 379)
    getitem___123242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 379, 34), seqarrays_123241, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 379)
    subscript_call_result_123243 = invoke(stypy.reporting.localization.Localization(__file__, 379, 34), getitem___123242, int_123240)
    
    # Processing the call keyword arguments (line 379)
    kwargs_123244 = {}
    # Getting the type of 'np' (line 379)
    np_123238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 20), 'np', False)
    # Obtaining the member 'asanyarray' of a type (line 379)
    asanyarray_123239 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 379, 20), np_123238, 'asanyarray')
    # Calling asanyarray(args, kwargs) (line 379)
    asanyarray_call_result_123245 = invoke(stypy.reporting.localization.Localization(__file__, 379, 20), asanyarray_123239, *[subscript_call_result_123243], **kwargs_123244)
    
    # Assigning a type to the variable 'seqarrays' (line 379)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 379, 8), 'seqarrays', asanyarray_call_result_123245)
    # SSA join for if statement (line 378)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to isinstance(...): (line 381)
    # Processing the call arguments (line 381)
    # Getting the type of 'seqarrays' (line 381)
    seqarrays_123247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 18), 'seqarrays', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 381)
    tuple_123248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 30), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 381)
    # Adding element type (line 381)
    # Getting the type of 'ndarray' (line 381)
    ndarray_123249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 30), 'ndarray', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 381, 30), tuple_123248, ndarray_123249)
    # Adding element type (line 381)
    # Getting the type of 'np' (line 381)
    np_123250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 39), 'np', False)
    # Obtaining the member 'void' of a type (line 381)
    void_123251 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 381, 39), np_123250, 'void')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 381, 30), tuple_123248, void_123251)
    
    # Processing the call keyword arguments (line 381)
    kwargs_123252 = {}
    # Getting the type of 'isinstance' (line 381)
    isinstance_123246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 7), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 381)
    isinstance_call_result_123253 = invoke(stypy.reporting.localization.Localization(__file__, 381, 7), isinstance_123246, *[seqarrays_123247, tuple_123248], **kwargs_123252)
    
    # Testing the type of an if condition (line 381)
    if_condition_123254 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 381, 4), isinstance_call_result_123253)
    # Assigning a type to the variable 'if_condition_123254' (line 381)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 381, 4), 'if_condition_123254', if_condition_123254)
    # SSA begins for if statement (line 381)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Attribute to a Name (line 382):
    
    # Assigning a Attribute to a Name (line 382):
    # Getting the type of 'seqarrays' (line 382)
    seqarrays_123255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 19), 'seqarrays')
    # Obtaining the member 'dtype' of a type (line 382)
    dtype_123256 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 382, 19), seqarrays_123255, 'dtype')
    # Assigning a type to the variable 'seqdtype' (line 382)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 382, 8), 'seqdtype', dtype_123256)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'flatten' (line 383)
    flatten_123257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 16), 'flatten')
    # Applying the 'not' unary operator (line 383)
    result_not__123258 = python_operator(stypy.reporting.localization.Localization(__file__, 383, 12), 'not', flatten_123257)
    
    
    
    # Call to zip_descr(...): (line 384)
    # Processing the call arguments (line 384)
    
    # Obtaining an instance of the builtin type 'tuple' (line 384)
    tuple_123260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 23), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 384)
    # Adding element type (line 384)
    # Getting the type of 'seqarrays' (line 384)
    seqarrays_123261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 23), 'seqarrays', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 384, 23), tuple_123260, seqarrays_123261)
    
    # Processing the call keyword arguments (line 384)
    # Getting the type of 'True' (line 384)
    True_123262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 44), 'True', False)
    keyword_123263 = True_123262
    kwargs_123264 = {'flatten': keyword_123263}
    # Getting the type of 'zip_descr' (line 384)
    zip_descr_123259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 12), 'zip_descr', False)
    # Calling zip_descr(args, kwargs) (line 384)
    zip_descr_call_result_123265 = invoke(stypy.reporting.localization.Localization(__file__, 384, 12), zip_descr_123259, *[tuple_123260], **kwargs_123264)
    
    # Getting the type of 'seqdtype' (line 384)
    seqdtype_123266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 53), 'seqdtype')
    # Obtaining the member 'descr' of a type (line 384)
    descr_123267 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 384, 53), seqdtype_123266, 'descr')
    # Applying the binary operator '==' (line 384)
    result_eq_123268 = python_operator(stypy.reporting.localization.Localization(__file__, 384, 12), '==', zip_descr_call_result_123265, descr_123267)
    
    # Applying the binary operator 'or' (line 383)
    result_or_keyword_123269 = python_operator(stypy.reporting.localization.Localization(__file__, 383, 11), 'or', result_not__123258, result_eq_123268)
    
    # Testing the type of an if condition (line 383)
    if_condition_123270 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 383, 8), result_or_keyword_123269)
    # Assigning a type to the variable 'if_condition_123270' (line 383)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 8), 'if_condition_123270', if_condition_123270)
    # SSA begins for if statement (line 383)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 386):
    
    # Assigning a Call to a Name (line 386):
    
    # Call to ravel(...): (line 386)
    # Processing the call keyword arguments (line 386)
    kwargs_123273 = {}
    # Getting the type of 'seqarrays' (line 386)
    seqarrays_123271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 24), 'seqarrays', False)
    # Obtaining the member 'ravel' of a type (line 386)
    ravel_123272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 24), seqarrays_123271, 'ravel')
    # Calling ravel(args, kwargs) (line 386)
    ravel_call_result_123274 = invoke(stypy.reporting.localization.Localization(__file__, 386, 24), ravel_123272, *[], **kwargs_123273)
    
    # Assigning a type to the variable 'seqarrays' (line 386)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 386, 12), 'seqarrays', ravel_call_result_123274)
    
    
    # Getting the type of 'seqdtype' (line 388)
    seqdtype_123275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 19), 'seqdtype')
    # Obtaining the member 'names' of a type (line 388)
    names_123276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 19), seqdtype_123275, 'names')
    # Applying the 'not' unary operator (line 388)
    result_not__123277 = python_operator(stypy.reporting.localization.Localization(__file__, 388, 15), 'not', names_123276)
    
    # Testing the type of an if condition (line 388)
    if_condition_123278 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 388, 12), result_not__123277)
    # Assigning a type to the variable 'if_condition_123278' (line 388)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 12), 'if_condition_123278', if_condition_123278)
    # SSA begins for if statement (line 388)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a List to a Name (line 389):
    
    # Assigning a List to a Name (line 389):
    
    # Obtaining an instance of the builtin type 'list' (line 389)
    list_123279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 27), 'list')
    # Adding type elements to the builtin type 'list' instance (line 389)
    # Adding element type (line 389)
    
    # Obtaining an instance of the builtin type 'tuple' (line 389)
    tuple_123280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 29), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 389)
    # Adding element type (line 389)
    str_123281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 29), 'str', '')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 389, 29), tuple_123280, str_123281)
    # Adding element type (line 389)
    # Getting the type of 'seqdtype' (line 389)
    seqdtype_123282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 33), 'seqdtype')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 389, 29), tuple_123280, seqdtype_123282)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 389, 27), list_123279, tuple_123280)
    
    # Assigning a type to the variable 'seqdtype' (line 389)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 16), 'seqdtype', list_123279)
    # SSA join for if statement (line 388)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'usemask' (line 391)
    usemask_123283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 15), 'usemask')
    # Testing the type of an if condition (line 391)
    if_condition_123284 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 391, 12), usemask_123283)
    # Assigning a type to the variable 'if_condition_123284' (line 391)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 391, 12), 'if_condition_123284', if_condition_123284)
    # SSA begins for if statement (line 391)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'asrecarray' (line 392)
    asrecarray_123285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 19), 'asrecarray')
    # Testing the type of an if condition (line 392)
    if_condition_123286 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 392, 16), asrecarray_123285)
    # Assigning a type to the variable 'if_condition_123286' (line 392)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 392, 16), 'if_condition_123286', if_condition_123286)
    # SSA begins for if statement (line 392)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 393):
    
    # Assigning a Name to a Name (line 393):
    # Getting the type of 'MaskedRecords' (line 393)
    MaskedRecords_123287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 30), 'MaskedRecords')
    # Assigning a type to the variable 'seqtype' (line 393)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 20), 'seqtype', MaskedRecords_123287)
    # SSA branch for the else part of an if statement (line 392)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Name (line 395):
    
    # Assigning a Name to a Name (line 395):
    # Getting the type of 'MaskedArray' (line 395)
    MaskedArray_123288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 30), 'MaskedArray')
    # Assigning a type to the variable 'seqtype' (line 395)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 395, 20), 'seqtype', MaskedArray_123288)
    # SSA join for if statement (line 392)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 391)
    module_type_store.open_ssa_branch('else')
    
    # Getting the type of 'asrecarray' (line 396)
    asrecarray_123289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 17), 'asrecarray')
    # Testing the type of an if condition (line 396)
    if_condition_123290 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 396, 17), asrecarray_123289)
    # Assigning a type to the variable 'if_condition_123290' (line 396)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 396, 17), 'if_condition_123290', if_condition_123290)
    # SSA begins for if statement (line 396)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 397):
    
    # Assigning a Name to a Name (line 397):
    # Getting the type of 'recarray' (line 397)
    recarray_123291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 26), 'recarray')
    # Assigning a type to the variable 'seqtype' (line 397)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 397, 16), 'seqtype', recarray_123291)
    # SSA branch for the else part of an if statement (line 396)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Name (line 399):
    
    # Assigning a Name to a Name (line 399):
    # Getting the type of 'ndarray' (line 399)
    ndarray_123292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 26), 'ndarray')
    # Assigning a type to the variable 'seqtype' (line 399)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 399, 16), 'seqtype', ndarray_123292)
    # SSA join for if statement (line 396)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 391)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to view(...): (line 400)
    # Processing the call keyword arguments (line 400)
    # Getting the type of 'seqdtype' (line 400)
    seqdtype_123295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 40), 'seqdtype', False)
    keyword_123296 = seqdtype_123295
    # Getting the type of 'seqtype' (line 400)
    seqtype_123297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 55), 'seqtype', False)
    keyword_123298 = seqtype_123297
    kwargs_123299 = {'dtype': keyword_123296, 'type': keyword_123298}
    # Getting the type of 'seqarrays' (line 400)
    seqarrays_123293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 19), 'seqarrays', False)
    # Obtaining the member 'view' of a type (line 400)
    view_123294 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 400, 19), seqarrays_123293, 'view')
    # Calling view(args, kwargs) (line 400)
    view_call_result_123300 = invoke(stypy.reporting.localization.Localization(__file__, 400, 19), view_123294, *[], **kwargs_123299)
    
    # Assigning a type to the variable 'stypy_return_type' (line 400)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 400, 12), 'stypy_return_type', view_call_result_123300)
    # SSA branch for the else part of an if statement (line 383)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Tuple to a Name (line 402):
    
    # Assigning a Tuple to a Name (line 402):
    
    # Obtaining an instance of the builtin type 'tuple' (line 402)
    tuple_123301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 402, 25), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 402)
    # Adding element type (line 402)
    # Getting the type of 'seqarrays' (line 402)
    seqarrays_123302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 25), 'seqarrays')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 402, 25), tuple_123301, seqarrays_123302)
    
    # Assigning a type to the variable 'seqarrays' (line 402)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 402, 12), 'seqarrays', tuple_123301)
    # SSA join for if statement (line 383)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 381)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a ListComp to a Name (line 405):
    
    # Assigning a ListComp to a Name (line 405):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'seqarrays' (line 405)
    seqarrays_123308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 49), 'seqarrays')
    comprehension_123309 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 405, 21), seqarrays_123308)
    # Assigning a type to the variable '_m' (line 405)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 405, 21), '_m', comprehension_123309)
    
    # Call to asanyarray(...): (line 405)
    # Processing the call arguments (line 405)
    # Getting the type of '_m' (line 405)
    _m_123305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 35), '_m', False)
    # Processing the call keyword arguments (line 405)
    kwargs_123306 = {}
    # Getting the type of 'np' (line 405)
    np_123303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 21), 'np', False)
    # Obtaining the member 'asanyarray' of a type (line 405)
    asanyarray_123304 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 405, 21), np_123303, 'asanyarray')
    # Calling asanyarray(args, kwargs) (line 405)
    asanyarray_call_result_123307 = invoke(stypy.reporting.localization.Localization(__file__, 405, 21), asanyarray_123304, *[_m_123305], **kwargs_123306)
    
    list_123310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 21), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 405, 21), list_123310, asanyarray_call_result_123307)
    # Assigning a type to the variable 'seqarrays' (line 405)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 405, 8), 'seqarrays', list_123310)
    # SSA join for if statement (line 381)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 407):
    
    # Assigning a Call to a Name (line 407):
    
    # Call to tuple(...): (line 407)
    # Processing the call arguments (line 407)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 407, 18, True)
    # Calculating comprehension expression
    # Getting the type of 'seqarrays' (line 407)
    seqarrays_123314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 34), 'seqarrays', False)
    comprehension_123315 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 407, 18), seqarrays_123314)
    # Assigning a type to the variable 'a' (line 407)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 407, 18), 'a', comprehension_123315)
    # Getting the type of 'a' (line 407)
    a_123312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 18), 'a', False)
    # Obtaining the member 'size' of a type (line 407)
    size_123313 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 407, 18), a_123312, 'size')
    list_123316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 18), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 407, 18), list_123316, size_123313)
    # Processing the call keyword arguments (line 407)
    kwargs_123317 = {}
    # Getting the type of 'tuple' (line 407)
    tuple_123311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 12), 'tuple', False)
    # Calling tuple(args, kwargs) (line 407)
    tuple_call_result_123318 = invoke(stypy.reporting.localization.Localization(__file__, 407, 12), tuple_123311, *[list_123316], **kwargs_123317)
    
    # Assigning a type to the variable 'sizes' (line 407)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 407, 4), 'sizes', tuple_call_result_123318)
    
    # Assigning a Call to a Name (line 408):
    
    # Assigning a Call to a Name (line 408):
    
    # Call to max(...): (line 408)
    # Processing the call arguments (line 408)
    # Getting the type of 'sizes' (line 408)
    sizes_123320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 20), 'sizes', False)
    # Processing the call keyword arguments (line 408)
    kwargs_123321 = {}
    # Getting the type of 'max' (line 408)
    max_123319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 16), 'max', False)
    # Calling max(args, kwargs) (line 408)
    max_call_result_123322 = invoke(stypy.reporting.localization.Localization(__file__, 408, 16), max_123319, *[sizes_123320], **kwargs_123321)
    
    # Assigning a type to the variable 'maxlength' (line 408)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 408, 4), 'maxlength', max_call_result_123322)
    
    # Assigning a Call to a Name (line 410):
    
    # Assigning a Call to a Name (line 410):
    
    # Call to zip_descr(...): (line 410)
    # Processing the call arguments (line 410)
    # Getting the type of 'seqarrays' (line 410)
    seqarrays_123324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 25), 'seqarrays', False)
    # Processing the call keyword arguments (line 410)
    # Getting the type of 'flatten' (line 410)
    flatten_123325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 44), 'flatten', False)
    keyword_123326 = flatten_123325
    kwargs_123327 = {'flatten': keyword_123326}
    # Getting the type of 'zip_descr' (line 410)
    zip_descr_123323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 15), 'zip_descr', False)
    # Calling zip_descr(args, kwargs) (line 410)
    zip_descr_call_result_123328 = invoke(stypy.reporting.localization.Localization(__file__, 410, 15), zip_descr_123323, *[seqarrays_123324], **kwargs_123327)
    
    # Assigning a type to the variable 'newdtype' (line 410)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 410, 4), 'newdtype', zip_descr_call_result_123328)
    
    # Assigning a List to a Name (line 412):
    
    # Assigning a List to a Name (line 412):
    
    # Obtaining an instance of the builtin type 'list' (line 412)
    list_123329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 412, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 412)
    
    # Assigning a type to the variable 'seqdata' (line 412)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 412, 4), 'seqdata', list_123329)
    
    # Assigning a List to a Name (line 413):
    
    # Assigning a List to a Name (line 413):
    
    # Obtaining an instance of the builtin type 'list' (line 413)
    list_123330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 413)
    
    # Assigning a type to the variable 'seqmask' (line 413)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 413, 4), 'seqmask', list_123330)
    
    # Getting the type of 'usemask' (line 415)
    usemask_123331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 7), 'usemask')
    # Testing the type of an if condition (line 415)
    if_condition_123332 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 415, 4), usemask_123331)
    # Assigning a type to the variable 'if_condition_123332' (line 415)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 415, 4), 'if_condition_123332', if_condition_123332)
    # SSA begins for if statement (line 415)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Call to zip(...): (line 416)
    # Processing the call arguments (line 416)
    # Getting the type of 'seqarrays' (line 416)
    seqarrays_123334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 26), 'seqarrays', False)
    # Getting the type of 'sizes' (line 416)
    sizes_123335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 37), 'sizes', False)
    # Processing the call keyword arguments (line 416)
    kwargs_123336 = {}
    # Getting the type of 'zip' (line 416)
    zip_123333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 22), 'zip', False)
    # Calling zip(args, kwargs) (line 416)
    zip_call_result_123337 = invoke(stypy.reporting.localization.Localization(__file__, 416, 22), zip_123333, *[seqarrays_123334, sizes_123335], **kwargs_123336)
    
    # Testing the type of a for loop iterable (line 416)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 416, 8), zip_call_result_123337)
    # Getting the type of the for loop variable (line 416)
    for_loop_var_123338 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 416, 8), zip_call_result_123337)
    # Assigning a type to the variable 'a' (line 416)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 416, 8), 'a', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 416, 8), for_loop_var_123338))
    # Assigning a type to the variable 'n' (line 416)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 416, 8), 'n', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 416, 8), for_loop_var_123338))
    # SSA begins for a for statement (line 416)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a BinOp to a Name (line 417):
    
    # Assigning a BinOp to a Name (line 417):
    # Getting the type of 'maxlength' (line 417)
    maxlength_123339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 25), 'maxlength')
    # Getting the type of 'n' (line 417)
    n_123340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 37), 'n')
    # Applying the binary operator '-' (line 417)
    result_sub_123341 = python_operator(stypy.reporting.localization.Localization(__file__, 417, 25), '-', maxlength_123339, n_123340)
    
    # Assigning a type to the variable 'nbmissing' (line 417)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 417, 12), 'nbmissing', result_sub_123341)
    
    # Assigning a Call to a Name (line 419):
    
    # Assigning a Call to a Name (line 419):
    
    # Call to __array__(...): (line 419)
    # Processing the call keyword arguments (line 419)
    kwargs_123347 = {}
    
    # Call to ravel(...): (line 419)
    # Processing the call keyword arguments (line 419)
    kwargs_123344 = {}
    # Getting the type of 'a' (line 419)
    a_123342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 19), 'a', False)
    # Obtaining the member 'ravel' of a type (line 419)
    ravel_123343 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 419, 19), a_123342, 'ravel')
    # Calling ravel(args, kwargs) (line 419)
    ravel_call_result_123345 = invoke(stypy.reporting.localization.Localization(__file__, 419, 19), ravel_123343, *[], **kwargs_123344)
    
    # Obtaining the member '__array__' of a type (line 419)
    array___123346 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 419, 19), ravel_call_result_123345, '__array__')
    # Calling __array__(args, kwargs) (line 419)
    array___call_result_123348 = invoke(stypy.reporting.localization.Localization(__file__, 419, 19), array___123346, *[], **kwargs_123347)
    
    # Assigning a type to the variable 'data' (line 419)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 419, 12), 'data', array___call_result_123348)
    
    # Assigning a Call to a Name (line 420):
    
    # Assigning a Call to a Name (line 420):
    
    # Call to ravel(...): (line 420)
    # Processing the call keyword arguments (line 420)
    kwargs_123355 = {}
    
    # Call to getmaskarray(...): (line 420)
    # Processing the call arguments (line 420)
    # Getting the type of 'a' (line 420)
    a_123351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 35), 'a', False)
    # Processing the call keyword arguments (line 420)
    kwargs_123352 = {}
    # Getting the type of 'ma' (line 420)
    ma_123349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 19), 'ma', False)
    # Obtaining the member 'getmaskarray' of a type (line 420)
    getmaskarray_123350 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 420, 19), ma_123349, 'getmaskarray')
    # Calling getmaskarray(args, kwargs) (line 420)
    getmaskarray_call_result_123353 = invoke(stypy.reporting.localization.Localization(__file__, 420, 19), getmaskarray_123350, *[a_123351], **kwargs_123352)
    
    # Obtaining the member 'ravel' of a type (line 420)
    ravel_123354 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 420, 19), getmaskarray_call_result_123353, 'ravel')
    # Calling ravel(args, kwargs) (line 420)
    ravel_call_result_123356 = invoke(stypy.reporting.localization.Localization(__file__, 420, 19), ravel_123354, *[], **kwargs_123355)
    
    # Assigning a type to the variable 'mask' (line 420)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 420, 12), 'mask', ravel_call_result_123356)
    
    # Getting the type of 'nbmissing' (line 422)
    nbmissing_123357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 15), 'nbmissing')
    # Testing the type of an if condition (line 422)
    if_condition_123358 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 422, 12), nbmissing_123357)
    # Assigning a type to the variable 'if_condition_123358' (line 422)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 422, 12), 'if_condition_123358', if_condition_123358)
    # SSA begins for if statement (line 422)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 423):
    
    # Assigning a Call to a Name (line 423):
    
    # Call to _check_fill_value(...): (line 423)
    # Processing the call arguments (line 423)
    # Getting the type of 'fill_value' (line 423)
    fill_value_123360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 41), 'fill_value', False)
    # Getting the type of 'a' (line 423)
    a_123361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 53), 'a', False)
    # Obtaining the member 'dtype' of a type (line 423)
    dtype_123362 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 53), a_123361, 'dtype')
    # Processing the call keyword arguments (line 423)
    kwargs_123363 = {}
    # Getting the type of '_check_fill_value' (line 423)
    _check_fill_value_123359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 23), '_check_fill_value', False)
    # Calling _check_fill_value(args, kwargs) (line 423)
    _check_fill_value_call_result_123364 = invoke(stypy.reporting.localization.Localization(__file__, 423, 23), _check_fill_value_123359, *[fill_value_123360, dtype_123362], **kwargs_123363)
    
    # Assigning a type to the variable 'fval' (line 423)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 16), 'fval', _check_fill_value_call_result_123364)
    
    
    # Call to isinstance(...): (line 424)
    # Processing the call arguments (line 424)
    # Getting the type of 'fval' (line 424)
    fval_123366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 30), 'fval', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 424)
    tuple_123367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 424, 37), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 424)
    # Adding element type (line 424)
    # Getting the type of 'ndarray' (line 424)
    ndarray_123368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 37), 'ndarray', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 424, 37), tuple_123367, ndarray_123368)
    # Adding element type (line 424)
    # Getting the type of 'np' (line 424)
    np_123369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 46), 'np', False)
    # Obtaining the member 'void' of a type (line 424)
    void_123370 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 424, 46), np_123369, 'void')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 424, 37), tuple_123367, void_123370)
    
    # Processing the call keyword arguments (line 424)
    kwargs_123371 = {}
    # Getting the type of 'isinstance' (line 424)
    isinstance_123365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 19), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 424)
    isinstance_call_result_123372 = invoke(stypy.reporting.localization.Localization(__file__, 424, 19), isinstance_123365, *[fval_123366, tuple_123367], **kwargs_123371)
    
    # Testing the type of an if condition (line 424)
    if_condition_123373 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 424, 16), isinstance_call_result_123372)
    # Assigning a type to the variable 'if_condition_123373' (line 424)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 424, 16), 'if_condition_123373', if_condition_123373)
    # SSA begins for if statement (line 424)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    
    # Call to len(...): (line 425)
    # Processing the call arguments (line 425)
    # Getting the type of 'fval' (line 425)
    fval_123375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 27), 'fval', False)
    # Obtaining the member 'dtype' of a type (line 425)
    dtype_123376 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 425, 27), fval_123375, 'dtype')
    # Processing the call keyword arguments (line 425)
    kwargs_123377 = {}
    # Getting the type of 'len' (line 425)
    len_123374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 23), 'len', False)
    # Calling len(args, kwargs) (line 425)
    len_call_result_123378 = invoke(stypy.reporting.localization.Localization(__file__, 425, 23), len_123374, *[dtype_123376], **kwargs_123377)
    
    int_123379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 425, 42), 'int')
    # Applying the binary operator '==' (line 425)
    result_eq_123380 = python_operator(stypy.reporting.localization.Localization(__file__, 425, 23), '==', len_call_result_123378, int_123379)
    
    # Testing the type of an if condition (line 425)
    if_condition_123381 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 425, 20), result_eq_123380)
    # Assigning a type to the variable 'if_condition_123381' (line 425)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 425, 20), 'if_condition_123381', if_condition_123381)
    # SSA begins for if statement (line 425)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 426):
    
    # Assigning a Subscript to a Name (line 426):
    
    # Obtaining the type of the subscript
    int_123382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 426, 43), 'int')
    
    # Call to item(...): (line 426)
    # Processing the call keyword arguments (line 426)
    kwargs_123385 = {}
    # Getting the type of 'fval' (line 426)
    fval_123383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 31), 'fval', False)
    # Obtaining the member 'item' of a type (line 426)
    item_123384 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 426, 31), fval_123383, 'item')
    # Calling item(args, kwargs) (line 426)
    item_call_result_123386 = invoke(stypy.reporting.localization.Localization(__file__, 426, 31), item_123384, *[], **kwargs_123385)
    
    # Obtaining the member '__getitem__' of a type (line 426)
    getitem___123387 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 426, 31), item_call_result_123386, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 426)
    subscript_call_result_123388 = invoke(stypy.reporting.localization.Localization(__file__, 426, 31), getitem___123387, int_123382)
    
    # Assigning a type to the variable 'fval' (line 426)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 426, 24), 'fval', subscript_call_result_123388)
    
    # Assigning a Name to a Name (line 427):
    
    # Assigning a Name to a Name (line 427):
    # Getting the type of 'True' (line 427)
    True_123389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 31), 'True')
    # Assigning a type to the variable 'fmsk' (line 427)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 427, 24), 'fmsk', True_123389)
    # SSA branch for the else part of an if statement (line 425)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 429):
    
    # Assigning a Call to a Name (line 429):
    
    # Call to array(...): (line 429)
    # Processing the call arguments (line 429)
    # Getting the type of 'fval' (line 429)
    fval_123392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 40), 'fval', False)
    # Processing the call keyword arguments (line 429)
    # Getting the type of 'a' (line 429)
    a_123393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 52), 'a', False)
    # Obtaining the member 'dtype' of a type (line 429)
    dtype_123394 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 429, 52), a_123393, 'dtype')
    keyword_123395 = dtype_123394
    int_123396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 429, 67), 'int')
    keyword_123397 = int_123396
    kwargs_123398 = {'dtype': keyword_123395, 'ndmin': keyword_123397}
    # Getting the type of 'np' (line 429)
    np_123390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 31), 'np', False)
    # Obtaining the member 'array' of a type (line 429)
    array_123391 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 429, 31), np_123390, 'array')
    # Calling array(args, kwargs) (line 429)
    array_call_result_123399 = invoke(stypy.reporting.localization.Localization(__file__, 429, 31), array_123391, *[fval_123392], **kwargs_123398)
    
    # Assigning a type to the variable 'fval' (line 429)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 429, 24), 'fval', array_call_result_123399)
    
    # Assigning a Call to a Name (line 430):
    
    # Assigning a Call to a Name (line 430):
    
    # Call to ones(...): (line 430)
    # Processing the call arguments (line 430)
    
    # Obtaining an instance of the builtin type 'tuple' (line 430)
    tuple_123402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 430, 40), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 430)
    # Adding element type (line 430)
    int_123403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 430, 40), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 430, 40), tuple_123402, int_123403)
    
    # Processing the call keyword arguments (line 430)
    # Getting the type of 'mask' (line 430)
    mask_123404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 51), 'mask', False)
    # Obtaining the member 'dtype' of a type (line 430)
    dtype_123405 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 51), mask_123404, 'dtype')
    keyword_123406 = dtype_123405
    kwargs_123407 = {'dtype': keyword_123406}
    # Getting the type of 'np' (line 430)
    np_123400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 31), 'np', False)
    # Obtaining the member 'ones' of a type (line 430)
    ones_123401 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 31), np_123400, 'ones')
    # Calling ones(args, kwargs) (line 430)
    ones_call_result_123408 = invoke(stypy.reporting.localization.Localization(__file__, 430, 31), ones_123401, *[tuple_123402], **kwargs_123407)
    
    # Assigning a type to the variable 'fmsk' (line 430)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 430, 24), 'fmsk', ones_call_result_123408)
    # SSA join for if statement (line 425)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 424)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 422)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Name (line 432):
    
    # Assigning a Name to a Name (line 432):
    # Getting the type of 'None' (line 432)
    None_123409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 23), 'None')
    # Assigning a type to the variable 'fval' (line 432)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 432, 16), 'fval', None_123409)
    
    # Assigning a Name to a Name (line 433):
    
    # Assigning a Name to a Name (line 433):
    # Getting the type of 'True' (line 433)
    True_123410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 23), 'True')
    # Assigning a type to the variable 'fmsk' (line 433)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 433, 16), 'fmsk', True_123410)
    # SSA join for if statement (line 422)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to append(...): (line 435)
    # Processing the call arguments (line 435)
    
    # Call to chain(...): (line 435)
    # Processing the call arguments (line 435)
    # Getting the type of 'data' (line 435)
    data_123415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 43), 'data', False)
    
    # Obtaining an instance of the builtin type 'list' (line 435)
    list_123416 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 435, 49), 'list')
    # Adding type elements to the builtin type 'list' instance (line 435)
    # Adding element type (line 435)
    # Getting the type of 'fval' (line 435)
    fval_123417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 50), 'fval', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 435, 49), list_123416, fval_123417)
    
    # Getting the type of 'nbmissing' (line 435)
    nbmissing_123418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 58), 'nbmissing', False)
    # Applying the binary operator '*' (line 435)
    result_mul_123419 = python_operator(stypy.reporting.localization.Localization(__file__, 435, 49), '*', list_123416, nbmissing_123418)
    
    # Processing the call keyword arguments (line 435)
    kwargs_123420 = {}
    # Getting the type of 'itertools' (line 435)
    itertools_123413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 27), 'itertools', False)
    # Obtaining the member 'chain' of a type (line 435)
    chain_123414 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 435, 27), itertools_123413, 'chain')
    # Calling chain(args, kwargs) (line 435)
    chain_call_result_123421 = invoke(stypy.reporting.localization.Localization(__file__, 435, 27), chain_123414, *[data_123415, result_mul_123419], **kwargs_123420)
    
    # Processing the call keyword arguments (line 435)
    kwargs_123422 = {}
    # Getting the type of 'seqdata' (line 435)
    seqdata_123411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 12), 'seqdata', False)
    # Obtaining the member 'append' of a type (line 435)
    append_123412 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 435, 12), seqdata_123411, 'append')
    # Calling append(args, kwargs) (line 435)
    append_call_result_123423 = invoke(stypy.reporting.localization.Localization(__file__, 435, 12), append_123412, *[chain_call_result_123421], **kwargs_123422)
    
    
    # Call to append(...): (line 436)
    # Processing the call arguments (line 436)
    
    # Call to chain(...): (line 436)
    # Processing the call arguments (line 436)
    # Getting the type of 'mask' (line 436)
    mask_123428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 43), 'mask', False)
    
    # Obtaining an instance of the builtin type 'list' (line 436)
    list_123429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 436, 49), 'list')
    # Adding type elements to the builtin type 'list' instance (line 436)
    # Adding element type (line 436)
    # Getting the type of 'fmsk' (line 436)
    fmsk_123430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 50), 'fmsk', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 436, 49), list_123429, fmsk_123430)
    
    # Getting the type of 'nbmissing' (line 436)
    nbmissing_123431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 58), 'nbmissing', False)
    # Applying the binary operator '*' (line 436)
    result_mul_123432 = python_operator(stypy.reporting.localization.Localization(__file__, 436, 49), '*', list_123429, nbmissing_123431)
    
    # Processing the call keyword arguments (line 436)
    kwargs_123433 = {}
    # Getting the type of 'itertools' (line 436)
    itertools_123426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 27), 'itertools', False)
    # Obtaining the member 'chain' of a type (line 436)
    chain_123427 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 436, 27), itertools_123426, 'chain')
    # Calling chain(args, kwargs) (line 436)
    chain_call_result_123434 = invoke(stypy.reporting.localization.Localization(__file__, 436, 27), chain_123427, *[mask_123428, result_mul_123432], **kwargs_123433)
    
    # Processing the call keyword arguments (line 436)
    kwargs_123435 = {}
    # Getting the type of 'seqmask' (line 436)
    seqmask_123424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 12), 'seqmask', False)
    # Obtaining the member 'append' of a type (line 436)
    append_123425 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 436, 12), seqmask_123424, 'append')
    # Calling append(args, kwargs) (line 436)
    append_call_result_123436 = invoke(stypy.reporting.localization.Localization(__file__, 436, 12), append_123425, *[chain_call_result_123434], **kwargs_123435)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 438):
    
    # Assigning a Call to a Name (line 438):
    
    # Call to tuple(...): (line 438)
    # Processing the call arguments (line 438)
    
    # Call to izip_records(...): (line 438)
    # Processing the call arguments (line 438)
    # Getting the type of 'seqdata' (line 438)
    seqdata_123439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 34), 'seqdata', False)
    # Processing the call keyword arguments (line 438)
    # Getting the type of 'flatten' (line 438)
    flatten_123440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 51), 'flatten', False)
    keyword_123441 = flatten_123440
    kwargs_123442 = {'flatten': keyword_123441}
    # Getting the type of 'izip_records' (line 438)
    izip_records_123438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 21), 'izip_records', False)
    # Calling izip_records(args, kwargs) (line 438)
    izip_records_call_result_123443 = invoke(stypy.reporting.localization.Localization(__file__, 438, 21), izip_records_123438, *[seqdata_123439], **kwargs_123442)
    
    # Processing the call keyword arguments (line 438)
    kwargs_123444 = {}
    # Getting the type of 'tuple' (line 438)
    tuple_123437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 15), 'tuple', False)
    # Calling tuple(args, kwargs) (line 438)
    tuple_call_result_123445 = invoke(stypy.reporting.localization.Localization(__file__, 438, 15), tuple_123437, *[izip_records_call_result_123443], **kwargs_123444)
    
    # Assigning a type to the variable 'data' (line 438)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 438, 8), 'data', tuple_call_result_123445)
    
    # Assigning a Call to a Name (line 439):
    
    # Assigning a Call to a Name (line 439):
    
    # Call to array(...): (line 439)
    # Processing the call arguments (line 439)
    
    # Call to fromiter(...): (line 439)
    # Processing the call arguments (line 439)
    # Getting the type of 'data' (line 439)
    data_123450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 38), 'data', False)
    # Processing the call keyword arguments (line 439)
    # Getting the type of 'newdtype' (line 439)
    newdtype_123451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 50), 'newdtype', False)
    keyword_123452 = newdtype_123451
    # Getting the type of 'maxlength' (line 439)
    maxlength_123453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 66), 'maxlength', False)
    keyword_123454 = maxlength_123453
    kwargs_123455 = {'count': keyword_123454, 'dtype': keyword_123452}
    # Getting the type of 'np' (line 439)
    np_123448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 26), 'np', False)
    # Obtaining the member 'fromiter' of a type (line 439)
    fromiter_123449 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 439, 26), np_123448, 'fromiter')
    # Calling fromiter(args, kwargs) (line 439)
    fromiter_call_result_123456 = invoke(stypy.reporting.localization.Localization(__file__, 439, 26), fromiter_123449, *[data_123450], **kwargs_123455)
    
    # Processing the call keyword arguments (line 439)
    
    # Call to list(...): (line 440)
    # Processing the call arguments (line 440)
    
    # Call to izip_records(...): (line 440)
    # Processing the call arguments (line 440)
    # Getting the type of 'seqmask' (line 440)
    seqmask_123459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 49), 'seqmask', False)
    # Processing the call keyword arguments (line 440)
    # Getting the type of 'flatten' (line 440)
    flatten_123460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 66), 'flatten', False)
    keyword_123461 = flatten_123460
    kwargs_123462 = {'flatten': keyword_123461}
    # Getting the type of 'izip_records' (line 440)
    izip_records_123458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 36), 'izip_records', False)
    # Calling izip_records(args, kwargs) (line 440)
    izip_records_call_result_123463 = invoke(stypy.reporting.localization.Localization(__file__, 440, 36), izip_records_123458, *[seqmask_123459], **kwargs_123462)
    
    # Processing the call keyword arguments (line 440)
    kwargs_123464 = {}
    # Getting the type of 'list' (line 440)
    list_123457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 31), 'list', False)
    # Calling list(args, kwargs) (line 440)
    list_call_result_123465 = invoke(stypy.reporting.localization.Localization(__file__, 440, 31), list_123457, *[izip_records_call_result_123463], **kwargs_123464)
    
    keyword_123466 = list_call_result_123465
    kwargs_123467 = {'mask': keyword_123466}
    # Getting the type of 'ma' (line 439)
    ma_123446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 17), 'ma', False)
    # Obtaining the member 'array' of a type (line 439)
    array_123447 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 439, 17), ma_123446, 'array')
    # Calling array(args, kwargs) (line 439)
    array_call_result_123468 = invoke(stypy.reporting.localization.Localization(__file__, 439, 17), array_123447, *[fromiter_call_result_123456], **kwargs_123467)
    
    # Assigning a type to the variable 'output' (line 439)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 439, 8), 'output', array_call_result_123468)
    
    # Getting the type of 'asrecarray' (line 441)
    asrecarray_123469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 11), 'asrecarray')
    # Testing the type of an if condition (line 441)
    if_condition_123470 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 441, 8), asrecarray_123469)
    # Assigning a type to the variable 'if_condition_123470' (line 441)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 441, 8), 'if_condition_123470', if_condition_123470)
    # SSA begins for if statement (line 441)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 442):
    
    # Assigning a Call to a Name (line 442):
    
    # Call to view(...): (line 442)
    # Processing the call arguments (line 442)
    # Getting the type of 'MaskedRecords' (line 442)
    MaskedRecords_123473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 33), 'MaskedRecords', False)
    # Processing the call keyword arguments (line 442)
    kwargs_123474 = {}
    # Getting the type of 'output' (line 442)
    output_123471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 21), 'output', False)
    # Obtaining the member 'view' of a type (line 442)
    view_123472 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 442, 21), output_123471, 'view')
    # Calling view(args, kwargs) (line 442)
    view_call_result_123475 = invoke(stypy.reporting.localization.Localization(__file__, 442, 21), view_123472, *[MaskedRecords_123473], **kwargs_123474)
    
    # Assigning a type to the variable 'output' (line 442)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 442, 12), 'output', view_call_result_123475)
    # SSA join for if statement (line 441)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 415)
    module_type_store.open_ssa_branch('else')
    
    
    # Call to zip(...): (line 445)
    # Processing the call arguments (line 445)
    # Getting the type of 'seqarrays' (line 445)
    seqarrays_123477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 26), 'seqarrays', False)
    # Getting the type of 'sizes' (line 445)
    sizes_123478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 37), 'sizes', False)
    # Processing the call keyword arguments (line 445)
    kwargs_123479 = {}
    # Getting the type of 'zip' (line 445)
    zip_123476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 22), 'zip', False)
    # Calling zip(args, kwargs) (line 445)
    zip_call_result_123480 = invoke(stypy.reporting.localization.Localization(__file__, 445, 22), zip_123476, *[seqarrays_123477, sizes_123478], **kwargs_123479)
    
    # Testing the type of a for loop iterable (line 445)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 445, 8), zip_call_result_123480)
    # Getting the type of the for loop variable (line 445)
    for_loop_var_123481 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 445, 8), zip_call_result_123480)
    # Assigning a type to the variable 'a' (line 445)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 445, 8), 'a', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 445, 8), for_loop_var_123481))
    # Assigning a type to the variable 'n' (line 445)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 445, 8), 'n', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 445, 8), for_loop_var_123481))
    # SSA begins for a for statement (line 445)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a BinOp to a Name (line 446):
    
    # Assigning a BinOp to a Name (line 446):
    # Getting the type of 'maxlength' (line 446)
    maxlength_123482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 25), 'maxlength')
    # Getting the type of 'n' (line 446)
    n_123483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 37), 'n')
    # Applying the binary operator '-' (line 446)
    result_sub_123484 = python_operator(stypy.reporting.localization.Localization(__file__, 446, 25), '-', maxlength_123482, n_123483)
    
    # Assigning a type to the variable 'nbmissing' (line 446)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 446, 12), 'nbmissing', result_sub_123484)
    
    # Assigning a Call to a Name (line 447):
    
    # Assigning a Call to a Name (line 447):
    
    # Call to __array__(...): (line 447)
    # Processing the call keyword arguments (line 447)
    kwargs_123490 = {}
    
    # Call to ravel(...): (line 447)
    # Processing the call keyword arguments (line 447)
    kwargs_123487 = {}
    # Getting the type of 'a' (line 447)
    a_123485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 19), 'a', False)
    # Obtaining the member 'ravel' of a type (line 447)
    ravel_123486 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 447, 19), a_123485, 'ravel')
    # Calling ravel(args, kwargs) (line 447)
    ravel_call_result_123488 = invoke(stypy.reporting.localization.Localization(__file__, 447, 19), ravel_123486, *[], **kwargs_123487)
    
    # Obtaining the member '__array__' of a type (line 447)
    array___123489 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 447, 19), ravel_call_result_123488, '__array__')
    # Calling __array__(args, kwargs) (line 447)
    array___call_result_123491 = invoke(stypy.reporting.localization.Localization(__file__, 447, 19), array___123489, *[], **kwargs_123490)
    
    # Assigning a type to the variable 'data' (line 447)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 447, 12), 'data', array___call_result_123491)
    
    # Getting the type of 'nbmissing' (line 448)
    nbmissing_123492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 15), 'nbmissing')
    # Testing the type of an if condition (line 448)
    if_condition_123493 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 448, 12), nbmissing_123492)
    # Assigning a type to the variable 'if_condition_123493' (line 448)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 448, 12), 'if_condition_123493', if_condition_123493)
    # SSA begins for if statement (line 448)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 449):
    
    # Assigning a Call to a Name (line 449):
    
    # Call to _check_fill_value(...): (line 449)
    # Processing the call arguments (line 449)
    # Getting the type of 'fill_value' (line 449)
    fill_value_123495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 41), 'fill_value', False)
    # Getting the type of 'a' (line 449)
    a_123496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 53), 'a', False)
    # Obtaining the member 'dtype' of a type (line 449)
    dtype_123497 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 449, 53), a_123496, 'dtype')
    # Processing the call keyword arguments (line 449)
    kwargs_123498 = {}
    # Getting the type of '_check_fill_value' (line 449)
    _check_fill_value_123494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 23), '_check_fill_value', False)
    # Calling _check_fill_value(args, kwargs) (line 449)
    _check_fill_value_call_result_123499 = invoke(stypy.reporting.localization.Localization(__file__, 449, 23), _check_fill_value_123494, *[fill_value_123495, dtype_123497], **kwargs_123498)
    
    # Assigning a type to the variable 'fval' (line 449)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 449, 16), 'fval', _check_fill_value_call_result_123499)
    
    
    # Call to isinstance(...): (line 450)
    # Processing the call arguments (line 450)
    # Getting the type of 'fval' (line 450)
    fval_123501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 30), 'fval', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 450)
    tuple_123502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 450, 37), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 450)
    # Adding element type (line 450)
    # Getting the type of 'ndarray' (line 450)
    ndarray_123503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 37), 'ndarray', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 450, 37), tuple_123502, ndarray_123503)
    # Adding element type (line 450)
    # Getting the type of 'np' (line 450)
    np_123504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 46), 'np', False)
    # Obtaining the member 'void' of a type (line 450)
    void_123505 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 450, 46), np_123504, 'void')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 450, 37), tuple_123502, void_123505)
    
    # Processing the call keyword arguments (line 450)
    kwargs_123506 = {}
    # Getting the type of 'isinstance' (line 450)
    isinstance_123500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 19), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 450)
    isinstance_call_result_123507 = invoke(stypy.reporting.localization.Localization(__file__, 450, 19), isinstance_123500, *[fval_123501, tuple_123502], **kwargs_123506)
    
    # Testing the type of an if condition (line 450)
    if_condition_123508 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 450, 16), isinstance_call_result_123507)
    # Assigning a type to the variable 'if_condition_123508' (line 450)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 450, 16), 'if_condition_123508', if_condition_123508)
    # SSA begins for if statement (line 450)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    
    # Call to len(...): (line 451)
    # Processing the call arguments (line 451)
    # Getting the type of 'fval' (line 451)
    fval_123510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 27), 'fval', False)
    # Obtaining the member 'dtype' of a type (line 451)
    dtype_123511 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 451, 27), fval_123510, 'dtype')
    # Processing the call keyword arguments (line 451)
    kwargs_123512 = {}
    # Getting the type of 'len' (line 451)
    len_123509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 23), 'len', False)
    # Calling len(args, kwargs) (line 451)
    len_call_result_123513 = invoke(stypy.reporting.localization.Localization(__file__, 451, 23), len_123509, *[dtype_123511], **kwargs_123512)
    
    int_123514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 451, 42), 'int')
    # Applying the binary operator '==' (line 451)
    result_eq_123515 = python_operator(stypy.reporting.localization.Localization(__file__, 451, 23), '==', len_call_result_123513, int_123514)
    
    # Testing the type of an if condition (line 451)
    if_condition_123516 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 451, 20), result_eq_123515)
    # Assigning a type to the variable 'if_condition_123516' (line 451)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 451, 20), 'if_condition_123516', if_condition_123516)
    # SSA begins for if statement (line 451)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 452):
    
    # Assigning a Subscript to a Name (line 452):
    
    # Obtaining the type of the subscript
    int_123517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 452, 43), 'int')
    
    # Call to item(...): (line 452)
    # Processing the call keyword arguments (line 452)
    kwargs_123520 = {}
    # Getting the type of 'fval' (line 452)
    fval_123518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 31), 'fval', False)
    # Obtaining the member 'item' of a type (line 452)
    item_123519 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 452, 31), fval_123518, 'item')
    # Calling item(args, kwargs) (line 452)
    item_call_result_123521 = invoke(stypy.reporting.localization.Localization(__file__, 452, 31), item_123519, *[], **kwargs_123520)
    
    # Obtaining the member '__getitem__' of a type (line 452)
    getitem___123522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 452, 31), item_call_result_123521, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 452)
    subscript_call_result_123523 = invoke(stypy.reporting.localization.Localization(__file__, 452, 31), getitem___123522, int_123517)
    
    # Assigning a type to the variable 'fval' (line 452)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 452, 24), 'fval', subscript_call_result_123523)
    # SSA branch for the else part of an if statement (line 451)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 454):
    
    # Assigning a Call to a Name (line 454):
    
    # Call to array(...): (line 454)
    # Processing the call arguments (line 454)
    # Getting the type of 'fval' (line 454)
    fval_123526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 40), 'fval', False)
    # Processing the call keyword arguments (line 454)
    # Getting the type of 'a' (line 454)
    a_123527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 52), 'a', False)
    # Obtaining the member 'dtype' of a type (line 454)
    dtype_123528 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 454, 52), a_123527, 'dtype')
    keyword_123529 = dtype_123528
    int_123530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 454, 67), 'int')
    keyword_123531 = int_123530
    kwargs_123532 = {'dtype': keyword_123529, 'ndmin': keyword_123531}
    # Getting the type of 'np' (line 454)
    np_123524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 31), 'np', False)
    # Obtaining the member 'array' of a type (line 454)
    array_123525 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 454, 31), np_123524, 'array')
    # Calling array(args, kwargs) (line 454)
    array_call_result_123533 = invoke(stypy.reporting.localization.Localization(__file__, 454, 31), array_123525, *[fval_123526], **kwargs_123532)
    
    # Assigning a type to the variable 'fval' (line 454)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 454, 24), 'fval', array_call_result_123533)
    # SSA join for if statement (line 451)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 450)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 448)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Name (line 456):
    
    # Assigning a Name to a Name (line 456):
    # Getting the type of 'None' (line 456)
    None_123534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 23), 'None')
    # Assigning a type to the variable 'fval' (line 456)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 16), 'fval', None_123534)
    # SSA join for if statement (line 448)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to append(...): (line 457)
    # Processing the call arguments (line 457)
    
    # Call to chain(...): (line 457)
    # Processing the call arguments (line 457)
    # Getting the type of 'data' (line 457)
    data_123539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 43), 'data', False)
    
    # Obtaining an instance of the builtin type 'list' (line 457)
    list_123540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 457, 49), 'list')
    # Adding type elements to the builtin type 'list' instance (line 457)
    # Adding element type (line 457)
    # Getting the type of 'fval' (line 457)
    fval_123541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 50), 'fval', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 457, 49), list_123540, fval_123541)
    
    # Getting the type of 'nbmissing' (line 457)
    nbmissing_123542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 58), 'nbmissing', False)
    # Applying the binary operator '*' (line 457)
    result_mul_123543 = python_operator(stypy.reporting.localization.Localization(__file__, 457, 49), '*', list_123540, nbmissing_123542)
    
    # Processing the call keyword arguments (line 457)
    kwargs_123544 = {}
    # Getting the type of 'itertools' (line 457)
    itertools_123537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 27), 'itertools', False)
    # Obtaining the member 'chain' of a type (line 457)
    chain_123538 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 457, 27), itertools_123537, 'chain')
    # Calling chain(args, kwargs) (line 457)
    chain_call_result_123545 = invoke(stypy.reporting.localization.Localization(__file__, 457, 27), chain_123538, *[data_123539, result_mul_123543], **kwargs_123544)
    
    # Processing the call keyword arguments (line 457)
    kwargs_123546 = {}
    # Getting the type of 'seqdata' (line 457)
    seqdata_123535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 12), 'seqdata', False)
    # Obtaining the member 'append' of a type (line 457)
    append_123536 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 457, 12), seqdata_123535, 'append')
    # Calling append(args, kwargs) (line 457)
    append_call_result_123547 = invoke(stypy.reporting.localization.Localization(__file__, 457, 12), append_123536, *[chain_call_result_123545], **kwargs_123546)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 458):
    
    # Assigning a Call to a Name (line 458):
    
    # Call to fromiter(...): (line 458)
    # Processing the call arguments (line 458)
    
    # Call to tuple(...): (line 458)
    # Processing the call arguments (line 458)
    
    # Call to izip_records(...): (line 458)
    # Processing the call arguments (line 458)
    # Getting the type of 'seqdata' (line 458)
    seqdata_123552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 48), 'seqdata', False)
    # Processing the call keyword arguments (line 458)
    # Getting the type of 'flatten' (line 458)
    flatten_123553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 65), 'flatten', False)
    keyword_123554 = flatten_123553
    kwargs_123555 = {'flatten': keyword_123554}
    # Getting the type of 'izip_records' (line 458)
    izip_records_123551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 35), 'izip_records', False)
    # Calling izip_records(args, kwargs) (line 458)
    izip_records_call_result_123556 = invoke(stypy.reporting.localization.Localization(__file__, 458, 35), izip_records_123551, *[seqdata_123552], **kwargs_123555)
    
    # Processing the call keyword arguments (line 458)
    kwargs_123557 = {}
    # Getting the type of 'tuple' (line 458)
    tuple_123550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 29), 'tuple', False)
    # Calling tuple(args, kwargs) (line 458)
    tuple_call_result_123558 = invoke(stypy.reporting.localization.Localization(__file__, 458, 29), tuple_123550, *[izip_records_call_result_123556], **kwargs_123557)
    
    # Processing the call keyword arguments (line 458)
    # Getting the type of 'newdtype' (line 459)
    newdtype_123559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 35), 'newdtype', False)
    keyword_123560 = newdtype_123559
    # Getting the type of 'maxlength' (line 459)
    maxlength_123561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 51), 'maxlength', False)
    keyword_123562 = maxlength_123561
    kwargs_123563 = {'count': keyword_123562, 'dtype': keyword_123560}
    # Getting the type of 'np' (line 458)
    np_123548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 17), 'np', False)
    # Obtaining the member 'fromiter' of a type (line 458)
    fromiter_123549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 458, 17), np_123548, 'fromiter')
    # Calling fromiter(args, kwargs) (line 458)
    fromiter_call_result_123564 = invoke(stypy.reporting.localization.Localization(__file__, 458, 17), fromiter_123549, *[tuple_call_result_123558], **kwargs_123563)
    
    # Assigning a type to the variable 'output' (line 458)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 458, 8), 'output', fromiter_call_result_123564)
    
    # Getting the type of 'asrecarray' (line 460)
    asrecarray_123565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 11), 'asrecarray')
    # Testing the type of an if condition (line 460)
    if_condition_123566 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 460, 8), asrecarray_123565)
    # Assigning a type to the variable 'if_condition_123566' (line 460)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 460, 8), 'if_condition_123566', if_condition_123566)
    # SSA begins for if statement (line 460)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 461):
    
    # Assigning a Call to a Name (line 461):
    
    # Call to view(...): (line 461)
    # Processing the call arguments (line 461)
    # Getting the type of 'recarray' (line 461)
    recarray_123569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 33), 'recarray', False)
    # Processing the call keyword arguments (line 461)
    kwargs_123570 = {}
    # Getting the type of 'output' (line 461)
    output_123567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 21), 'output', False)
    # Obtaining the member 'view' of a type (line 461)
    view_123568 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 461, 21), output_123567, 'view')
    # Calling view(args, kwargs) (line 461)
    view_call_result_123571 = invoke(stypy.reporting.localization.Localization(__file__, 461, 21), view_123568, *[recarray_123569], **kwargs_123570)
    
    # Assigning a type to the variable 'output' (line 461)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 461, 12), 'output', view_call_result_123571)
    # SSA join for if statement (line 460)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 415)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'output' (line 463)
    output_123572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 11), 'output')
    # Assigning a type to the variable 'stypy_return_type' (line 463)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 463, 4), 'stypy_return_type', output_123572)
    
    # ################# End of 'merge_arrays(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'merge_arrays' in the type store
    # Getting the type of 'stypy_return_type' (line 329)
    stypy_return_type_123573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_123573)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'merge_arrays'
    return stypy_return_type_123573

# Assigning a type to the variable 'merge_arrays' (line 329)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 0), 'merge_arrays', merge_arrays)

@norecursion
def drop_fields(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'True' (line 466)
    True_123574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 42), 'True')
    # Getting the type of 'False' (line 466)
    False_123575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 59), 'False')
    defaults = [True_123574, False_123575]
    # Create a new context for function 'drop_fields'
    module_type_store = module_type_store.open_function_context('drop_fields', 466, 0, False)
    
    # Passed parameters checking function
    drop_fields.stypy_localization = localization
    drop_fields.stypy_type_of_self = None
    drop_fields.stypy_type_store = module_type_store
    drop_fields.stypy_function_name = 'drop_fields'
    drop_fields.stypy_param_names_list = ['base', 'drop_names', 'usemask', 'asrecarray']
    drop_fields.stypy_varargs_param_name = None
    drop_fields.stypy_kwargs_param_name = None
    drop_fields.stypy_call_defaults = defaults
    drop_fields.stypy_call_varargs = varargs
    drop_fields.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'drop_fields', ['base', 'drop_names', 'usemask', 'asrecarray'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'drop_fields', localization, ['base', 'drop_names', 'usemask', 'asrecarray'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'drop_fields(...)' code ##################

    str_123576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 500, (-1)), 'str', "\n    Return a new array with fields in `drop_names` dropped.\n\n    Nested fields are supported.\n\n    Parameters\n    ----------\n    base : array\n        Input array\n    drop_names : string or sequence\n        String or sequence of strings corresponding to the names of the\n        fields to drop.\n    usemask : {False, True}, optional\n        Whether to return a masked array or not.\n    asrecarray : string or sequence, optional\n        Whether to return a recarray or a mrecarray (`asrecarray=True`) or\n        a plain ndarray or masked array with flexible dtype. The default\n        is False.\n\n    Examples\n    --------\n    >>> from numpy.lib import recfunctions as rfn\n    >>> a = np.array([(1, (2, 3.0)), (4, (5, 6.0))],\n    ...   dtype=[('a', int), ('b', [('ba', float), ('bb', int)])])\n    >>> rfn.drop_fields(a, 'a')\n    array([((2.0, 3),), ((5.0, 6),)],\n          dtype=[('b', [('ba', '<f8'), ('bb', '<i4')])])\n    >>> rfn.drop_fields(a, 'ba')\n    array([(1, (3,)), (4, (6,))],\n          dtype=[('a', '<i4'), ('b', [('bb', '<i4')])])\n    >>> rfn.drop_fields(a, ['ba', 'bb'])\n    array([(1,), (4,)],\n          dtype=[('a', '<i4')])\n    ")
    
    
    # Call to _is_string_like(...): (line 501)
    # Processing the call arguments (line 501)
    # Getting the type of 'drop_names' (line 501)
    drop_names_123578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 23), 'drop_names', False)
    # Processing the call keyword arguments (line 501)
    kwargs_123579 = {}
    # Getting the type of '_is_string_like' (line 501)
    _is_string_like_123577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 7), '_is_string_like', False)
    # Calling _is_string_like(args, kwargs) (line 501)
    _is_string_like_call_result_123580 = invoke(stypy.reporting.localization.Localization(__file__, 501, 7), _is_string_like_123577, *[drop_names_123578], **kwargs_123579)
    
    # Testing the type of an if condition (line 501)
    if_condition_123581 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 501, 4), _is_string_like_call_result_123580)
    # Assigning a type to the variable 'if_condition_123581' (line 501)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 501, 4), 'if_condition_123581', if_condition_123581)
    # SSA begins for if statement (line 501)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a List to a Name (line 502):
    
    # Assigning a List to a Name (line 502):
    
    # Obtaining an instance of the builtin type 'list' (line 502)
    list_123582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 502, 21), 'list')
    # Adding type elements to the builtin type 'list' instance (line 502)
    # Adding element type (line 502)
    # Getting the type of 'drop_names' (line 502)
    drop_names_123583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 22), 'drop_names')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 502, 21), list_123582, drop_names_123583)
    
    # Assigning a type to the variable 'drop_names' (line 502)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 502, 8), 'drop_names', list_123582)
    # SSA branch for the else part of an if statement (line 501)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 504):
    
    # Assigning a Call to a Name (line 504):
    
    # Call to set(...): (line 504)
    # Processing the call arguments (line 504)
    # Getting the type of 'drop_names' (line 504)
    drop_names_123585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 25), 'drop_names', False)
    # Processing the call keyword arguments (line 504)
    kwargs_123586 = {}
    # Getting the type of 'set' (line 504)
    set_123584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 21), 'set', False)
    # Calling set(args, kwargs) (line 504)
    set_call_result_123587 = invoke(stypy.reporting.localization.Localization(__file__, 504, 21), set_123584, *[drop_names_123585], **kwargs_123586)
    
    # Assigning a type to the variable 'drop_names' (line 504)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 504, 8), 'drop_names', set_call_result_123587)
    # SSA join for if statement (line 501)
    module_type_store = module_type_store.join_ssa_context()
    

    @norecursion
    def _drop_descr(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_drop_descr'
        module_type_store = module_type_store.open_function_context('_drop_descr', 506, 4, False)
        
        # Passed parameters checking function
        _drop_descr.stypy_localization = localization
        _drop_descr.stypy_type_of_self = None
        _drop_descr.stypy_type_store = module_type_store
        _drop_descr.stypy_function_name = '_drop_descr'
        _drop_descr.stypy_param_names_list = ['ndtype', 'drop_names']
        _drop_descr.stypy_varargs_param_name = None
        _drop_descr.stypy_kwargs_param_name = None
        _drop_descr.stypy_call_defaults = defaults
        _drop_descr.stypy_call_varargs = varargs
        _drop_descr.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_drop_descr', ['ndtype', 'drop_names'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_drop_descr', localization, ['ndtype', 'drop_names'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_drop_descr(...)' code ##################

        
        # Assigning a Attribute to a Name (line 507):
        
        # Assigning a Attribute to a Name (line 507):
        # Getting the type of 'ndtype' (line 507)
        ndtype_123588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 16), 'ndtype')
        # Obtaining the member 'names' of a type (line 507)
        names_123589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 507, 16), ndtype_123588, 'names')
        # Assigning a type to the variable 'names' (line 507)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 507, 8), 'names', names_123589)
        
        # Assigning a List to a Name (line 508):
        
        # Assigning a List to a Name (line 508):
        
        # Obtaining an instance of the builtin type 'list' (line 508)
        list_123590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 508, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 508)
        
        # Assigning a type to the variable 'newdtype' (line 508)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 508, 8), 'newdtype', list_123590)
        
        # Getting the type of 'names' (line 509)
        names_123591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 20), 'names')
        # Testing the type of a for loop iterable (line 509)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 509, 8), names_123591)
        # Getting the type of the for loop variable (line 509)
        for_loop_var_123592 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 509, 8), names_123591)
        # Assigning a type to the variable 'name' (line 509)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 509, 8), 'name', for_loop_var_123592)
        # SSA begins for a for statement (line 509)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Subscript to a Name (line 510):
        
        # Assigning a Subscript to a Name (line 510):
        
        # Obtaining the type of the subscript
        # Getting the type of 'name' (line 510)
        name_123593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 29), 'name')
        # Getting the type of 'ndtype' (line 510)
        ndtype_123594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 22), 'ndtype')
        # Obtaining the member '__getitem__' of a type (line 510)
        getitem___123595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 510, 22), ndtype_123594, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 510)
        subscript_call_result_123596 = invoke(stypy.reporting.localization.Localization(__file__, 510, 22), getitem___123595, name_123593)
        
        # Assigning a type to the variable 'current' (line 510)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 510, 12), 'current', subscript_call_result_123596)
        
        
        # Getting the type of 'name' (line 511)
        name_123597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 15), 'name')
        # Getting the type of 'drop_names' (line 511)
        drop_names_123598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 23), 'drop_names')
        # Applying the binary operator 'in' (line 511)
        result_contains_123599 = python_operator(stypy.reporting.localization.Localization(__file__, 511, 15), 'in', name_123597, drop_names_123598)
        
        # Testing the type of an if condition (line 511)
        if_condition_123600 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 511, 12), result_contains_123599)
        # Assigning a type to the variable 'if_condition_123600' (line 511)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 511, 12), 'if_condition_123600', if_condition_123600)
        # SSA begins for if statement (line 511)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA join for if statement (line 511)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'current' (line 513)
        current_123601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 15), 'current')
        # Obtaining the member 'names' of a type (line 513)
        names_123602 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 513, 15), current_123601, 'names')
        # Testing the type of an if condition (line 513)
        if_condition_123603 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 513, 12), names_123602)
        # Assigning a type to the variable 'if_condition_123603' (line 513)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 513, 12), 'if_condition_123603', if_condition_123603)
        # SSA begins for if statement (line 513)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 514):
        
        # Assigning a Call to a Name (line 514):
        
        # Call to _drop_descr(...): (line 514)
        # Processing the call arguments (line 514)
        # Getting the type of 'current' (line 514)
        current_123605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 36), 'current', False)
        # Getting the type of 'drop_names' (line 514)
        drop_names_123606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 45), 'drop_names', False)
        # Processing the call keyword arguments (line 514)
        kwargs_123607 = {}
        # Getting the type of '_drop_descr' (line 514)
        _drop_descr_123604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 24), '_drop_descr', False)
        # Calling _drop_descr(args, kwargs) (line 514)
        _drop_descr_call_result_123608 = invoke(stypy.reporting.localization.Localization(__file__, 514, 24), _drop_descr_123604, *[current_123605, drop_names_123606], **kwargs_123607)
        
        # Assigning a type to the variable 'descr' (line 514)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 514, 16), 'descr', _drop_descr_call_result_123608)
        
        # Getting the type of 'descr' (line 515)
        descr_123609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 19), 'descr')
        # Testing the type of an if condition (line 515)
        if_condition_123610 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 515, 16), descr_123609)
        # Assigning a type to the variable 'if_condition_123610' (line 515)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 515, 16), 'if_condition_123610', if_condition_123610)
        # SSA begins for if statement (line 515)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 516)
        # Processing the call arguments (line 516)
        
        # Obtaining an instance of the builtin type 'tuple' (line 516)
        tuple_123613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 516, 37), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 516)
        # Adding element type (line 516)
        # Getting the type of 'name' (line 516)
        name_123614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 37), 'name', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 516, 37), tuple_123613, name_123614)
        # Adding element type (line 516)
        # Getting the type of 'descr' (line 516)
        descr_123615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 43), 'descr', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 516, 37), tuple_123613, descr_123615)
        
        # Processing the call keyword arguments (line 516)
        kwargs_123616 = {}
        # Getting the type of 'newdtype' (line 516)
        newdtype_123611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 20), 'newdtype', False)
        # Obtaining the member 'append' of a type (line 516)
        append_123612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 516, 20), newdtype_123611, 'append')
        # Calling append(args, kwargs) (line 516)
        append_call_result_123617 = invoke(stypy.reporting.localization.Localization(__file__, 516, 20), append_123612, *[tuple_123613], **kwargs_123616)
        
        # SSA join for if statement (line 515)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 513)
        module_type_store.open_ssa_branch('else')
        
        # Call to append(...): (line 518)
        # Processing the call arguments (line 518)
        
        # Obtaining an instance of the builtin type 'tuple' (line 518)
        tuple_123620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 518, 33), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 518)
        # Adding element type (line 518)
        # Getting the type of 'name' (line 518)
        name_123621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 33), 'name', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 518, 33), tuple_123620, name_123621)
        # Adding element type (line 518)
        # Getting the type of 'current' (line 518)
        current_123622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 39), 'current', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 518, 33), tuple_123620, current_123622)
        
        # Processing the call keyword arguments (line 518)
        kwargs_123623 = {}
        # Getting the type of 'newdtype' (line 518)
        newdtype_123618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 16), 'newdtype', False)
        # Obtaining the member 'append' of a type (line 518)
        append_123619 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 518, 16), newdtype_123618, 'append')
        # Calling append(args, kwargs) (line 518)
        append_call_result_123624 = invoke(stypy.reporting.localization.Localization(__file__, 518, 16), append_123619, *[tuple_123620], **kwargs_123623)
        
        # SSA join for if statement (line 513)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'newdtype' (line 519)
        newdtype_123625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 15), 'newdtype')
        # Assigning a type to the variable 'stypy_return_type' (line 519)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 519, 8), 'stypy_return_type', newdtype_123625)
        
        # ################# End of '_drop_descr(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_drop_descr' in the type store
        # Getting the type of 'stypy_return_type' (line 506)
        stypy_return_type_123626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_123626)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_drop_descr'
        return stypy_return_type_123626

    # Assigning a type to the variable '_drop_descr' (line 506)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 506, 4), '_drop_descr', _drop_descr)
    
    # Assigning a Call to a Name (line 521):
    
    # Assigning a Call to a Name (line 521):
    
    # Call to _drop_descr(...): (line 521)
    # Processing the call arguments (line 521)
    # Getting the type of 'base' (line 521)
    base_123628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 27), 'base', False)
    # Obtaining the member 'dtype' of a type (line 521)
    dtype_123629 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 521, 27), base_123628, 'dtype')
    # Getting the type of 'drop_names' (line 521)
    drop_names_123630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 39), 'drop_names', False)
    # Processing the call keyword arguments (line 521)
    kwargs_123631 = {}
    # Getting the type of '_drop_descr' (line 521)
    _drop_descr_123627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 15), '_drop_descr', False)
    # Calling _drop_descr(args, kwargs) (line 521)
    _drop_descr_call_result_123632 = invoke(stypy.reporting.localization.Localization(__file__, 521, 15), _drop_descr_123627, *[dtype_123629, drop_names_123630], **kwargs_123631)
    
    # Assigning a type to the variable 'newdtype' (line 521)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 521, 4), 'newdtype', _drop_descr_call_result_123632)
    
    
    # Getting the type of 'newdtype' (line 522)
    newdtype_123633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 11), 'newdtype')
    # Applying the 'not' unary operator (line 522)
    result_not__123634 = python_operator(stypy.reporting.localization.Localization(__file__, 522, 7), 'not', newdtype_123633)
    
    # Testing the type of an if condition (line 522)
    if_condition_123635 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 522, 4), result_not__123634)
    # Assigning a type to the variable 'if_condition_123635' (line 522)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 522, 4), 'if_condition_123635', if_condition_123635)
    # SSA begins for if statement (line 522)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'None' (line 523)
    None_123636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 15), 'None')
    # Assigning a type to the variable 'stypy_return_type' (line 523)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 523, 8), 'stypy_return_type', None_123636)
    # SSA join for if statement (line 522)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 525):
    
    # Assigning a Call to a Name (line 525):
    
    # Call to empty(...): (line 525)
    # Processing the call arguments (line 525)
    # Getting the type of 'base' (line 525)
    base_123639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 22), 'base', False)
    # Obtaining the member 'shape' of a type (line 525)
    shape_123640 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 525, 22), base_123639, 'shape')
    # Processing the call keyword arguments (line 525)
    # Getting the type of 'newdtype' (line 525)
    newdtype_123641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 40), 'newdtype', False)
    keyword_123642 = newdtype_123641
    kwargs_123643 = {'dtype': keyword_123642}
    # Getting the type of 'np' (line 525)
    np_123637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 13), 'np', False)
    # Obtaining the member 'empty' of a type (line 525)
    empty_123638 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 525, 13), np_123637, 'empty')
    # Calling empty(args, kwargs) (line 525)
    empty_call_result_123644 = invoke(stypy.reporting.localization.Localization(__file__, 525, 13), empty_123638, *[shape_123640], **kwargs_123643)
    
    # Assigning a type to the variable 'output' (line 525)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 525, 4), 'output', empty_call_result_123644)
    
    # Assigning a Call to a Name (line 526):
    
    # Assigning a Call to a Name (line 526):
    
    # Call to recursive_fill_fields(...): (line 526)
    # Processing the call arguments (line 526)
    # Getting the type of 'base' (line 526)
    base_123646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 35), 'base', False)
    # Getting the type of 'output' (line 526)
    output_123647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 41), 'output', False)
    # Processing the call keyword arguments (line 526)
    kwargs_123648 = {}
    # Getting the type of 'recursive_fill_fields' (line 526)
    recursive_fill_fields_123645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 13), 'recursive_fill_fields', False)
    # Calling recursive_fill_fields(args, kwargs) (line 526)
    recursive_fill_fields_call_result_123649 = invoke(stypy.reporting.localization.Localization(__file__, 526, 13), recursive_fill_fields_123645, *[base_123646, output_123647], **kwargs_123648)
    
    # Assigning a type to the variable 'output' (line 526)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 526, 4), 'output', recursive_fill_fields_call_result_123649)
    
    # Call to _fix_output(...): (line 527)
    # Processing the call arguments (line 527)
    # Getting the type of 'output' (line 527)
    output_123651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 23), 'output', False)
    # Processing the call keyword arguments (line 527)
    # Getting the type of 'usemask' (line 527)
    usemask_123652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 39), 'usemask', False)
    keyword_123653 = usemask_123652
    # Getting the type of 'asrecarray' (line 527)
    asrecarray_123654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 59), 'asrecarray', False)
    keyword_123655 = asrecarray_123654
    kwargs_123656 = {'usemask': keyword_123653, 'asrecarray': keyword_123655}
    # Getting the type of '_fix_output' (line 527)
    _fix_output_123650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 11), '_fix_output', False)
    # Calling _fix_output(args, kwargs) (line 527)
    _fix_output_call_result_123657 = invoke(stypy.reporting.localization.Localization(__file__, 527, 11), _fix_output_123650, *[output_123651], **kwargs_123656)
    
    # Assigning a type to the variable 'stypy_return_type' (line 527)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 527, 4), 'stypy_return_type', _fix_output_call_result_123657)
    
    # ################# End of 'drop_fields(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'drop_fields' in the type store
    # Getting the type of 'stypy_return_type' (line 466)
    stypy_return_type_123658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_123658)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'drop_fields'
    return stypy_return_type_123658

# Assigning a type to the variable 'drop_fields' (line 466)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 466, 0), 'drop_fields', drop_fields)

@norecursion
def rec_drop_fields(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'rec_drop_fields'
    module_type_store = module_type_store.open_function_context('rec_drop_fields', 530, 0, False)
    
    # Passed parameters checking function
    rec_drop_fields.stypy_localization = localization
    rec_drop_fields.stypy_type_of_self = None
    rec_drop_fields.stypy_type_store = module_type_store
    rec_drop_fields.stypy_function_name = 'rec_drop_fields'
    rec_drop_fields.stypy_param_names_list = ['base', 'drop_names']
    rec_drop_fields.stypy_varargs_param_name = None
    rec_drop_fields.stypy_kwargs_param_name = None
    rec_drop_fields.stypy_call_defaults = defaults
    rec_drop_fields.stypy_call_varargs = varargs
    rec_drop_fields.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'rec_drop_fields', ['base', 'drop_names'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'rec_drop_fields', localization, ['base', 'drop_names'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'rec_drop_fields(...)' code ##################

    str_123659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 533, (-1)), 'str', '\n    Returns a new numpy.recarray with fields in `drop_names` dropped.\n    ')
    
    # Call to drop_fields(...): (line 534)
    # Processing the call arguments (line 534)
    # Getting the type of 'base' (line 534)
    base_123661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 23), 'base', False)
    # Getting the type of 'drop_names' (line 534)
    drop_names_123662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 29), 'drop_names', False)
    # Processing the call keyword arguments (line 534)
    # Getting the type of 'False' (line 534)
    False_123663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 49), 'False', False)
    keyword_123664 = False_123663
    # Getting the type of 'True' (line 534)
    True_123665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 67), 'True', False)
    keyword_123666 = True_123665
    kwargs_123667 = {'usemask': keyword_123664, 'asrecarray': keyword_123666}
    # Getting the type of 'drop_fields' (line 534)
    drop_fields_123660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 11), 'drop_fields', False)
    # Calling drop_fields(args, kwargs) (line 534)
    drop_fields_call_result_123668 = invoke(stypy.reporting.localization.Localization(__file__, 534, 11), drop_fields_123660, *[base_123661, drop_names_123662], **kwargs_123667)
    
    # Assigning a type to the variable 'stypy_return_type' (line 534)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 534, 4), 'stypy_return_type', drop_fields_call_result_123668)
    
    # ################# End of 'rec_drop_fields(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'rec_drop_fields' in the type store
    # Getting the type of 'stypy_return_type' (line 530)
    stypy_return_type_123669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_123669)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'rec_drop_fields'
    return stypy_return_type_123669

# Assigning a type to the variable 'rec_drop_fields' (line 530)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 530, 0), 'rec_drop_fields', rec_drop_fields)

@norecursion
def rename_fields(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'rename_fields'
    module_type_store = module_type_store.open_function_context('rename_fields', 537, 0, False)
    
    # Passed parameters checking function
    rename_fields.stypy_localization = localization
    rename_fields.stypy_type_of_self = None
    rename_fields.stypy_type_store = module_type_store
    rename_fields.stypy_function_name = 'rename_fields'
    rename_fields.stypy_param_names_list = ['base', 'namemapper']
    rename_fields.stypy_varargs_param_name = None
    rename_fields.stypy_kwargs_param_name = None
    rename_fields.stypy_call_defaults = defaults
    rename_fields.stypy_call_varargs = varargs
    rename_fields.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'rename_fields', ['base', 'namemapper'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'rename_fields', localization, ['base', 'namemapper'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'rename_fields(...)' code ##################

    str_123670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 559, (-1)), 'str', "\n    Rename the fields from a flexible-datatype ndarray or recarray.\n\n    Nested fields are supported.\n\n    Parameters\n    ----------\n    base : ndarray\n        Input array whose fields must be modified.\n    namemapper : dictionary\n        Dictionary mapping old field names to their new version.\n\n    Examples\n    --------\n    >>> from numpy.lib import recfunctions as rfn\n    >>> a = np.array([(1, (2, [3.0, 30.])), (4, (5, [6.0, 60.]))],\n    ...   dtype=[('a', int),('b', [('ba', float), ('bb', (float, 2))])])\n    >>> rfn.rename_fields(a, {'a':'A', 'bb':'BB'})\n    array([(1, (2.0, [3.0, 30.0])), (4, (5.0, [6.0, 60.0]))],\n          dtype=[('A', '<i4'), ('b', [('ba', '<f8'), ('BB', '<f8', 2)])])\n\n    ")

    @norecursion
    def _recursive_rename_fields(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_recursive_rename_fields'
        module_type_store = module_type_store.open_function_context('_recursive_rename_fields', 560, 4, False)
        
        # Passed parameters checking function
        _recursive_rename_fields.stypy_localization = localization
        _recursive_rename_fields.stypy_type_of_self = None
        _recursive_rename_fields.stypy_type_store = module_type_store
        _recursive_rename_fields.stypy_function_name = '_recursive_rename_fields'
        _recursive_rename_fields.stypy_param_names_list = ['ndtype', 'namemapper']
        _recursive_rename_fields.stypy_varargs_param_name = None
        _recursive_rename_fields.stypy_kwargs_param_name = None
        _recursive_rename_fields.stypy_call_defaults = defaults
        _recursive_rename_fields.stypy_call_varargs = varargs
        _recursive_rename_fields.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_recursive_rename_fields', ['ndtype', 'namemapper'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_recursive_rename_fields', localization, ['ndtype', 'namemapper'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_recursive_rename_fields(...)' code ##################

        
        # Assigning a List to a Name (line 561):
        
        # Assigning a List to a Name (line 561):
        
        # Obtaining an instance of the builtin type 'list' (line 561)
        list_123671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 561, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 561)
        
        # Assigning a type to the variable 'newdtype' (line 561)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 561, 8), 'newdtype', list_123671)
        
        # Getting the type of 'ndtype' (line 562)
        ndtype_123672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 20), 'ndtype')
        # Obtaining the member 'names' of a type (line 562)
        names_123673 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 562, 20), ndtype_123672, 'names')
        # Testing the type of a for loop iterable (line 562)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 562, 8), names_123673)
        # Getting the type of the for loop variable (line 562)
        for_loop_var_123674 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 562, 8), names_123673)
        # Assigning a type to the variable 'name' (line 562)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 562, 8), 'name', for_loop_var_123674)
        # SSA begins for a for statement (line 562)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 563):
        
        # Assigning a Call to a Name (line 563):
        
        # Call to get(...): (line 563)
        # Processing the call arguments (line 563)
        # Getting the type of 'name' (line 563)
        name_123677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 37), 'name', False)
        # Getting the type of 'name' (line 563)
        name_123678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 43), 'name', False)
        # Processing the call keyword arguments (line 563)
        kwargs_123679 = {}
        # Getting the type of 'namemapper' (line 563)
        namemapper_123675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 22), 'namemapper', False)
        # Obtaining the member 'get' of a type (line 563)
        get_123676 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 563, 22), namemapper_123675, 'get')
        # Calling get(args, kwargs) (line 563)
        get_call_result_123680 = invoke(stypy.reporting.localization.Localization(__file__, 563, 22), get_123676, *[name_123677, name_123678], **kwargs_123679)
        
        # Assigning a type to the variable 'newname' (line 563)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 563, 12), 'newname', get_call_result_123680)
        
        # Assigning a Subscript to a Name (line 564):
        
        # Assigning a Subscript to a Name (line 564):
        
        # Obtaining the type of the subscript
        # Getting the type of 'name' (line 564)
        name_123681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 29), 'name')
        # Getting the type of 'ndtype' (line 564)
        ndtype_123682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 22), 'ndtype')
        # Obtaining the member '__getitem__' of a type (line 564)
        getitem___123683 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 564, 22), ndtype_123682, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 564)
        subscript_call_result_123684 = invoke(stypy.reporting.localization.Localization(__file__, 564, 22), getitem___123683, name_123681)
        
        # Assigning a type to the variable 'current' (line 564)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 564, 12), 'current', subscript_call_result_123684)
        
        # Getting the type of 'current' (line 565)
        current_123685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 15), 'current')
        # Obtaining the member 'names' of a type (line 565)
        names_123686 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 565, 15), current_123685, 'names')
        # Testing the type of an if condition (line 565)
        if_condition_123687 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 565, 12), names_123686)
        # Assigning a type to the variable 'if_condition_123687' (line 565)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 565, 12), 'if_condition_123687', if_condition_123687)
        # SSA begins for if statement (line 565)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 566)
        # Processing the call arguments (line 566)
        
        # Obtaining an instance of the builtin type 'tuple' (line 567)
        tuple_123690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 567, 21), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 567)
        # Adding element type (line 567)
        # Getting the type of 'newname' (line 567)
        newname_123691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 21), 'newname', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 567, 21), tuple_123690, newname_123691)
        # Adding element type (line 567)
        
        # Call to _recursive_rename_fields(...): (line 567)
        # Processing the call arguments (line 567)
        # Getting the type of 'current' (line 567)
        current_123693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 55), 'current', False)
        # Getting the type of 'namemapper' (line 567)
        namemapper_123694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 64), 'namemapper', False)
        # Processing the call keyword arguments (line 567)
        kwargs_123695 = {}
        # Getting the type of '_recursive_rename_fields' (line 567)
        _recursive_rename_fields_123692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 30), '_recursive_rename_fields', False)
        # Calling _recursive_rename_fields(args, kwargs) (line 567)
        _recursive_rename_fields_call_result_123696 = invoke(stypy.reporting.localization.Localization(__file__, 567, 30), _recursive_rename_fields_123692, *[current_123693, namemapper_123694], **kwargs_123695)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 567, 21), tuple_123690, _recursive_rename_fields_call_result_123696)
        
        # Processing the call keyword arguments (line 566)
        kwargs_123697 = {}
        # Getting the type of 'newdtype' (line 566)
        newdtype_123688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 16), 'newdtype', False)
        # Obtaining the member 'append' of a type (line 566)
        append_123689 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 566, 16), newdtype_123688, 'append')
        # Calling append(args, kwargs) (line 566)
        append_call_result_123698 = invoke(stypy.reporting.localization.Localization(__file__, 566, 16), append_123689, *[tuple_123690], **kwargs_123697)
        
        # SSA branch for the else part of an if statement (line 565)
        module_type_store.open_ssa_branch('else')
        
        # Call to append(...): (line 570)
        # Processing the call arguments (line 570)
        
        # Obtaining an instance of the builtin type 'tuple' (line 570)
        tuple_123701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 570, 33), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 570)
        # Adding element type (line 570)
        # Getting the type of 'newname' (line 570)
        newname_123702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 33), 'newname', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 570, 33), tuple_123701, newname_123702)
        # Adding element type (line 570)
        # Getting the type of 'current' (line 570)
        current_123703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 42), 'current', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 570, 33), tuple_123701, current_123703)
        
        # Processing the call keyword arguments (line 570)
        kwargs_123704 = {}
        # Getting the type of 'newdtype' (line 570)
        newdtype_123699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 16), 'newdtype', False)
        # Obtaining the member 'append' of a type (line 570)
        append_123700 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 570, 16), newdtype_123699, 'append')
        # Calling append(args, kwargs) (line 570)
        append_call_result_123705 = invoke(stypy.reporting.localization.Localization(__file__, 570, 16), append_123700, *[tuple_123701], **kwargs_123704)
        
        # SSA join for if statement (line 565)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'newdtype' (line 571)
        newdtype_123706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 15), 'newdtype')
        # Assigning a type to the variable 'stypy_return_type' (line 571)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 571, 8), 'stypy_return_type', newdtype_123706)
        
        # ################# End of '_recursive_rename_fields(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_recursive_rename_fields' in the type store
        # Getting the type of 'stypy_return_type' (line 560)
        stypy_return_type_123707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_123707)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_recursive_rename_fields'
        return stypy_return_type_123707

    # Assigning a type to the variable '_recursive_rename_fields' (line 560)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 560, 4), '_recursive_rename_fields', _recursive_rename_fields)
    
    # Assigning a Call to a Name (line 572):
    
    # Assigning a Call to a Name (line 572):
    
    # Call to _recursive_rename_fields(...): (line 572)
    # Processing the call arguments (line 572)
    # Getting the type of 'base' (line 572)
    base_123709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 40), 'base', False)
    # Obtaining the member 'dtype' of a type (line 572)
    dtype_123710 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 572, 40), base_123709, 'dtype')
    # Getting the type of 'namemapper' (line 572)
    namemapper_123711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 52), 'namemapper', False)
    # Processing the call keyword arguments (line 572)
    kwargs_123712 = {}
    # Getting the type of '_recursive_rename_fields' (line 572)
    _recursive_rename_fields_123708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 15), '_recursive_rename_fields', False)
    # Calling _recursive_rename_fields(args, kwargs) (line 572)
    _recursive_rename_fields_call_result_123713 = invoke(stypy.reporting.localization.Localization(__file__, 572, 15), _recursive_rename_fields_123708, *[dtype_123710, namemapper_123711], **kwargs_123712)
    
    # Assigning a type to the variable 'newdtype' (line 572)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 572, 4), 'newdtype', _recursive_rename_fields_call_result_123713)
    
    # Call to view(...): (line 573)
    # Processing the call arguments (line 573)
    # Getting the type of 'newdtype' (line 573)
    newdtype_123716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 21), 'newdtype', False)
    # Processing the call keyword arguments (line 573)
    kwargs_123717 = {}
    # Getting the type of 'base' (line 573)
    base_123714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 11), 'base', False)
    # Obtaining the member 'view' of a type (line 573)
    view_123715 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 573, 11), base_123714, 'view')
    # Calling view(args, kwargs) (line 573)
    view_call_result_123718 = invoke(stypy.reporting.localization.Localization(__file__, 573, 11), view_123715, *[newdtype_123716], **kwargs_123717)
    
    # Assigning a type to the variable 'stypy_return_type' (line 573)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 573, 4), 'stypy_return_type', view_call_result_123718)
    
    # ################# End of 'rename_fields(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'rename_fields' in the type store
    # Getting the type of 'stypy_return_type' (line 537)
    stypy_return_type_123719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_123719)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'rename_fields'
    return stypy_return_type_123719

# Assigning a type to the variable 'rename_fields' (line 537)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 537, 0), 'rename_fields', rename_fields)

@norecursion
def append_fields(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 576)
    None_123720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 44), 'None')
    int_123721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 577, 29), 'int')
    # Getting the type of 'True' (line 577)
    True_123722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 41), 'True')
    # Getting the type of 'False' (line 577)
    False_123723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 58), 'False')
    defaults = [None_123720, int_123721, True_123722, False_123723]
    # Create a new context for function 'append_fields'
    module_type_store = module_type_store.open_function_context('append_fields', 576, 0, False)
    
    # Passed parameters checking function
    append_fields.stypy_localization = localization
    append_fields.stypy_type_of_self = None
    append_fields.stypy_type_store = module_type_store
    append_fields.stypy_function_name = 'append_fields'
    append_fields.stypy_param_names_list = ['base', 'names', 'data', 'dtypes', 'fill_value', 'usemask', 'asrecarray']
    append_fields.stypy_varargs_param_name = None
    append_fields.stypy_kwargs_param_name = None
    append_fields.stypy_call_defaults = defaults
    append_fields.stypy_call_varargs = varargs
    append_fields.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'append_fields', ['base', 'names', 'data', 'dtypes', 'fill_value', 'usemask', 'asrecarray'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'append_fields', localization, ['base', 'names', 'data', 'dtypes', 'fill_value', 'usemask', 'asrecarray'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'append_fields(...)' code ##################

    str_123724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 605, (-1)), 'str', '\n    Add new fields to an existing array.\n\n    The names of the fields are given with the `names` arguments,\n    the corresponding values with the `data` arguments.\n    If a single field is appended, `names`, `data` and `dtypes` do not have\n    to be lists but just values.\n\n    Parameters\n    ----------\n    base : array\n        Input array to extend.\n    names : string, sequence\n        String or sequence of strings corresponding to the names\n        of the new fields.\n    data : array or sequence of arrays\n        Array or sequence of arrays storing the fields to add to the base.\n    dtypes : sequence of datatypes, optional\n        Datatype or sequence of datatypes.\n        If None, the datatypes are estimated from the `data`.\n    fill_value : {float}, optional\n        Filling value used to pad missing data on the shorter arrays.\n    usemask : {False, True}, optional\n        Whether to return a masked array or not.\n    asrecarray : {False, True}, optional\n        Whether to return a recarray (MaskedRecords) or not.\n\n    ')
    
    
    # Call to isinstance(...): (line 607)
    # Processing the call arguments (line 607)
    # Getting the type of 'names' (line 607)
    names_123726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 607, 18), 'names', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 607)
    tuple_123727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 607, 26), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 607)
    # Adding element type (line 607)
    # Getting the type of 'tuple' (line 607)
    tuple_123728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 607, 26), 'tuple', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 607, 26), tuple_123727, tuple_123728)
    # Adding element type (line 607)
    # Getting the type of 'list' (line 607)
    list_123729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 607, 33), 'list', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 607, 26), tuple_123727, list_123729)
    
    # Processing the call keyword arguments (line 607)
    kwargs_123730 = {}
    # Getting the type of 'isinstance' (line 607)
    isinstance_123725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 607, 7), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 607)
    isinstance_call_result_123731 = invoke(stypy.reporting.localization.Localization(__file__, 607, 7), isinstance_123725, *[names_123726, tuple_123727], **kwargs_123730)
    
    # Testing the type of an if condition (line 607)
    if_condition_123732 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 607, 4), isinstance_call_result_123731)
    # Assigning a type to the variable 'if_condition_123732' (line 607)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 607, 4), 'if_condition_123732', if_condition_123732)
    # SSA begins for if statement (line 607)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    
    # Call to len(...): (line 608)
    # Processing the call arguments (line 608)
    # Getting the type of 'names' (line 608)
    names_123734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 15), 'names', False)
    # Processing the call keyword arguments (line 608)
    kwargs_123735 = {}
    # Getting the type of 'len' (line 608)
    len_123733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 11), 'len', False)
    # Calling len(args, kwargs) (line 608)
    len_call_result_123736 = invoke(stypy.reporting.localization.Localization(__file__, 608, 11), len_123733, *[names_123734], **kwargs_123735)
    
    
    # Call to len(...): (line 608)
    # Processing the call arguments (line 608)
    # Getting the type of 'data' (line 608)
    data_123738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 29), 'data', False)
    # Processing the call keyword arguments (line 608)
    kwargs_123739 = {}
    # Getting the type of 'len' (line 608)
    len_123737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 25), 'len', False)
    # Calling len(args, kwargs) (line 608)
    len_call_result_123740 = invoke(stypy.reporting.localization.Localization(__file__, 608, 25), len_123737, *[data_123738], **kwargs_123739)
    
    # Applying the binary operator '!=' (line 608)
    result_ne_123741 = python_operator(stypy.reporting.localization.Localization(__file__, 608, 11), '!=', len_call_result_123736, len_call_result_123740)
    
    # Testing the type of an if condition (line 608)
    if_condition_123742 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 608, 8), result_ne_123741)
    # Assigning a type to the variable 'if_condition_123742' (line 608)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 608, 8), 'if_condition_123742', if_condition_123742)
    # SSA begins for if statement (line 608)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 609):
    
    # Assigning a Str to a Name (line 609):
    str_123743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 609, 18), 'str', 'The number of arrays does not match the number of names')
    # Assigning a type to the variable 'msg' (line 609)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 609, 12), 'msg', str_123743)
    
    # Call to ValueError(...): (line 610)
    # Processing the call arguments (line 610)
    # Getting the type of 'msg' (line 610)
    msg_123745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 29), 'msg', False)
    # Processing the call keyword arguments (line 610)
    kwargs_123746 = {}
    # Getting the type of 'ValueError' (line 610)
    ValueError_123744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 610)
    ValueError_call_result_123747 = invoke(stypy.reporting.localization.Localization(__file__, 610, 18), ValueError_123744, *[msg_123745], **kwargs_123746)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 610, 12), ValueError_call_result_123747, 'raise parameter', BaseException)
    # SSA join for if statement (line 608)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 607)
    module_type_store.open_ssa_branch('else')
    
    # Type idiom detected: calculating its left and rigth part (line 611)
    # Getting the type of 'basestring' (line 611)
    basestring_123748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 27), 'basestring')
    # Getting the type of 'names' (line 611)
    names_123749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 20), 'names')
    
    (may_be_123750, more_types_in_union_123751) = may_be_subtype(basestring_123748, names_123749)

    if may_be_123750:

        if more_types_in_union_123751:
            # Runtime conditional SSA (line 611)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'names' (line 611)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 611, 9), 'names', remove_not_subtype_from_union(names_123749, basestring))
        
        # Assigning a List to a Name (line 612):
        
        # Assigning a List to a Name (line 612):
        
        # Obtaining an instance of the builtin type 'list' (line 612)
        list_123752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 612, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 612)
        # Adding element type (line 612)
        # Getting the type of 'names' (line 612)
        names_123753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 612, 17), 'names')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 612, 16), list_123752, names_123753)
        
        # Assigning a type to the variable 'names' (line 612)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 612, 8), 'names', list_123752)
        
        # Assigning a List to a Name (line 613):
        
        # Assigning a List to a Name (line 613):
        
        # Obtaining an instance of the builtin type 'list' (line 613)
        list_123754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 613, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 613)
        # Adding element type (line 613)
        # Getting the type of 'data' (line 613)
        data_123755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 16), 'data')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 613, 15), list_123754, data_123755)
        
        # Assigning a type to the variable 'data' (line 613)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 613, 8), 'data', list_123754)

        if more_types_in_union_123751:
            # SSA join for if statement (line 611)
            module_type_store = module_type_store.join_ssa_context()


    
    # SSA join for if statement (line 607)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 615)
    # Getting the type of 'dtypes' (line 615)
    dtypes_123756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 7), 'dtypes')
    # Getting the type of 'None' (line 615)
    None_123757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 17), 'None')
    
    (may_be_123758, more_types_in_union_123759) = may_be_none(dtypes_123756, None_123757)

    if may_be_123758:

        if more_types_in_union_123759:
            # Runtime conditional SSA (line 615)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a ListComp to a Name (line 616):
        
        # Assigning a ListComp to a Name (line 616):
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'data' (line 616)
        data_123769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 61), 'data')
        comprehension_123770 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 616, 16), data_123769)
        # Assigning a type to the variable 'a' (line 616)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 616, 16), 'a', comprehension_123770)
        
        # Call to array(...): (line 616)
        # Processing the call arguments (line 616)
        # Getting the type of 'a' (line 616)
        a_123762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 25), 'a', False)
        # Processing the call keyword arguments (line 616)
        # Getting the type of 'False' (line 616)
        False_123763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 33), 'False', False)
        keyword_123764 = False_123763
        # Getting the type of 'True' (line 616)
        True_123765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 46), 'True', False)
        keyword_123766 = True_123765
        kwargs_123767 = {'subok': keyword_123766, 'copy': keyword_123764}
        # Getting the type of 'np' (line 616)
        np_123760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 16), 'np', False)
        # Obtaining the member 'array' of a type (line 616)
        array_123761 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 616, 16), np_123760, 'array')
        # Calling array(args, kwargs) (line 616)
        array_call_result_123768 = invoke(stypy.reporting.localization.Localization(__file__, 616, 16), array_123761, *[a_123762], **kwargs_123767)
        
        list_123771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 616, 16), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 616, 16), list_123771, array_call_result_123768)
        # Assigning a type to the variable 'data' (line 616)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 616, 8), 'data', list_123771)
        
        # Assigning a ListComp to a Name (line 617):
        
        # Assigning a ListComp to a Name (line 617):
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to zip(...): (line 617)
        # Processing the call arguments (line 617)
        # Getting the type of 'names' (line 617)
        names_123782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 63), 'names', False)
        # Getting the type of 'data' (line 617)
        data_123783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 70), 'data', False)
        # Processing the call keyword arguments (line 617)
        kwargs_123784 = {}
        # Getting the type of 'zip' (line 617)
        zip_123781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 59), 'zip', False)
        # Calling zip(args, kwargs) (line 617)
        zip_call_result_123785 = invoke(stypy.reporting.localization.Localization(__file__, 617, 59), zip_123781, *[names_123782, data_123783], **kwargs_123784)
        
        comprehension_123786 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 617, 16), zip_call_result_123785)
        # Assigning a type to the variable 'name' (line 617)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 617, 16), 'name', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 617, 16), comprehension_123786))
        # Assigning a type to the variable 'a' (line 617)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 617, 16), 'a', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 617, 16), comprehension_123786))
        
        # Call to view(...): (line 617)
        # Processing the call arguments (line 617)
        
        # Obtaining an instance of the builtin type 'list' (line 617)
        list_123774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 617, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 617)
        # Adding element type (line 617)
        
        # Obtaining an instance of the builtin type 'tuple' (line 617)
        tuple_123775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 617, 25), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 617)
        # Adding element type (line 617)
        # Getting the type of 'name' (line 617)
        name_123776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 25), 'name', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 617, 25), tuple_123775, name_123776)
        # Adding element type (line 617)
        # Getting the type of 'a' (line 617)
        a_123777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 31), 'a', False)
        # Obtaining the member 'dtype' of a type (line 617)
        dtype_123778 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 617, 31), a_123777, 'dtype')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 617, 25), tuple_123775, dtype_123778)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 617, 23), list_123774, tuple_123775)
        
        # Processing the call keyword arguments (line 617)
        kwargs_123779 = {}
        # Getting the type of 'a' (line 617)
        a_123772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 16), 'a', False)
        # Obtaining the member 'view' of a type (line 617)
        view_123773 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 617, 16), a_123772, 'view')
        # Calling view(args, kwargs) (line 617)
        view_call_result_123780 = invoke(stypy.reporting.localization.Localization(__file__, 617, 16), view_123773, *[list_123774], **kwargs_123779)
        
        list_123787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 617, 16), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 617, 16), list_123787, view_call_result_123780)
        # Assigning a type to the variable 'data' (line 617)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 617, 8), 'data', list_123787)

        if more_types_in_union_123759:
            # Runtime conditional SSA for else branch (line 615)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_123758) or more_types_in_union_123759):
        
        
        
        # Call to isinstance(...): (line 619)
        # Processing the call arguments (line 619)
        # Getting the type of 'dtypes' (line 619)
        dtypes_123789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 26), 'dtypes', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 619)
        tuple_123790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 619, 35), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 619)
        # Adding element type (line 619)
        # Getting the type of 'tuple' (line 619)
        tuple_123791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 35), 'tuple', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 619, 35), tuple_123790, tuple_123791)
        # Adding element type (line 619)
        # Getting the type of 'list' (line 619)
        list_123792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 42), 'list', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 619, 35), tuple_123790, list_123792)
        
        # Processing the call keyword arguments (line 619)
        kwargs_123793 = {}
        # Getting the type of 'isinstance' (line 619)
        isinstance_123788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 15), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 619)
        isinstance_call_result_123794 = invoke(stypy.reporting.localization.Localization(__file__, 619, 15), isinstance_123788, *[dtypes_123789, tuple_123790], **kwargs_123793)
        
        # Applying the 'not' unary operator (line 619)
        result_not__123795 = python_operator(stypy.reporting.localization.Localization(__file__, 619, 11), 'not', isinstance_call_result_123794)
        
        # Testing the type of an if condition (line 619)
        if_condition_123796 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 619, 8), result_not__123795)
        # Assigning a type to the variable 'if_condition_123796' (line 619)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 619, 8), 'if_condition_123796', if_condition_123796)
        # SSA begins for if statement (line 619)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a List to a Name (line 620):
        
        # Assigning a List to a Name (line 620):
        
        # Obtaining an instance of the builtin type 'list' (line 620)
        list_123797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 620, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 620)
        # Adding element type (line 620)
        # Getting the type of 'dtypes' (line 620)
        dtypes_123798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 22), 'dtypes')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 620, 21), list_123797, dtypes_123798)
        
        # Assigning a type to the variable 'dtypes' (line 620)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 620, 12), 'dtypes', list_123797)
        # SSA join for if statement (line 619)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Call to len(...): (line 621)
        # Processing the call arguments (line 621)
        # Getting the type of 'data' (line 621)
        data_123800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 15), 'data', False)
        # Processing the call keyword arguments (line 621)
        kwargs_123801 = {}
        # Getting the type of 'len' (line 621)
        len_123799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 11), 'len', False)
        # Calling len(args, kwargs) (line 621)
        len_call_result_123802 = invoke(stypy.reporting.localization.Localization(__file__, 621, 11), len_123799, *[data_123800], **kwargs_123801)
        
        
        # Call to len(...): (line 621)
        # Processing the call arguments (line 621)
        # Getting the type of 'dtypes' (line 621)
        dtypes_123804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 28), 'dtypes', False)
        # Processing the call keyword arguments (line 621)
        kwargs_123805 = {}
        # Getting the type of 'len' (line 621)
        len_123803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 24), 'len', False)
        # Calling len(args, kwargs) (line 621)
        len_call_result_123806 = invoke(stypy.reporting.localization.Localization(__file__, 621, 24), len_123803, *[dtypes_123804], **kwargs_123805)
        
        # Applying the binary operator '!=' (line 621)
        result_ne_123807 = python_operator(stypy.reporting.localization.Localization(__file__, 621, 11), '!=', len_call_result_123802, len_call_result_123806)
        
        # Testing the type of an if condition (line 621)
        if_condition_123808 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 621, 8), result_ne_123807)
        # Assigning a type to the variable 'if_condition_123808' (line 621)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 621, 8), 'if_condition_123808', if_condition_123808)
        # SSA begins for if statement (line 621)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        
        # Call to len(...): (line 622)
        # Processing the call arguments (line 622)
        # Getting the type of 'dtypes' (line 622)
        dtypes_123810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 19), 'dtypes', False)
        # Processing the call keyword arguments (line 622)
        kwargs_123811 = {}
        # Getting the type of 'len' (line 622)
        len_123809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 15), 'len', False)
        # Calling len(args, kwargs) (line 622)
        len_call_result_123812 = invoke(stypy.reporting.localization.Localization(__file__, 622, 15), len_123809, *[dtypes_123810], **kwargs_123811)
        
        int_123813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 622, 30), 'int')
        # Applying the binary operator '==' (line 622)
        result_eq_123814 = python_operator(stypy.reporting.localization.Localization(__file__, 622, 15), '==', len_call_result_123812, int_123813)
        
        # Testing the type of an if condition (line 622)
        if_condition_123815 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 622, 12), result_eq_123814)
        # Assigning a type to the variable 'if_condition_123815' (line 622)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 622, 12), 'if_condition_123815', if_condition_123815)
        # SSA begins for if statement (line 622)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 623):
        
        # Assigning a BinOp to a Name (line 623):
        # Getting the type of 'dtypes' (line 623)
        dtypes_123816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 25), 'dtypes')
        
        # Call to len(...): (line 623)
        # Processing the call arguments (line 623)
        # Getting the type of 'data' (line 623)
        data_123818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 38), 'data', False)
        # Processing the call keyword arguments (line 623)
        kwargs_123819 = {}
        # Getting the type of 'len' (line 623)
        len_123817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 34), 'len', False)
        # Calling len(args, kwargs) (line 623)
        len_call_result_123820 = invoke(stypy.reporting.localization.Localization(__file__, 623, 34), len_123817, *[data_123818], **kwargs_123819)
        
        # Applying the binary operator '*' (line 623)
        result_mul_123821 = python_operator(stypy.reporting.localization.Localization(__file__, 623, 25), '*', dtypes_123816, len_call_result_123820)
        
        # Assigning a type to the variable 'dtypes' (line 623)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 623, 16), 'dtypes', result_mul_123821)
        # SSA branch for the else part of an if statement (line 622)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Str to a Name (line 625):
        
        # Assigning a Str to a Name (line 625):
        str_123822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 625, 22), 'str', 'The dtypes argument must be None, a dtype, or a list.')
        # Assigning a type to the variable 'msg' (line 625)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 625, 16), 'msg', str_123822)
        
        # Call to ValueError(...): (line 626)
        # Processing the call arguments (line 626)
        # Getting the type of 'msg' (line 626)
        msg_123824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 33), 'msg', False)
        # Processing the call keyword arguments (line 626)
        kwargs_123825 = {}
        # Getting the type of 'ValueError' (line 626)
        ValueError_123823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 22), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 626)
        ValueError_call_result_123826 = invoke(stypy.reporting.localization.Localization(__file__, 626, 22), ValueError_123823, *[msg_123824], **kwargs_123825)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 626, 16), ValueError_call_result_123826, 'raise parameter', BaseException)
        # SSA join for if statement (line 622)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 621)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a ListComp to a Name (line 627):
        
        # Assigning a ListComp to a Name (line 627):
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to zip(...): (line 628)
        # Processing the call arguments (line 628)
        # Getting the type of 'data' (line 628)
        data_123846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 628, 37), 'data', False)
        # Getting the type of 'names' (line 628)
        names_123847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 628, 43), 'names', False)
        # Getting the type of 'dtypes' (line 628)
        dtypes_123848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 628, 50), 'dtypes', False)
        # Processing the call keyword arguments (line 628)
        kwargs_123849 = {}
        # Getting the type of 'zip' (line 628)
        zip_123845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 628, 33), 'zip', False)
        # Calling zip(args, kwargs) (line 628)
        zip_call_result_123850 = invoke(stypy.reporting.localization.Localization(__file__, 628, 33), zip_123845, *[data_123846, names_123847, dtypes_123848], **kwargs_123849)
        
        comprehension_123851 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 627, 16), zip_call_result_123850)
        # Assigning a type to the variable 'a' (line 627)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 627, 16), 'a', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 627, 16), comprehension_123851))
        # Assigning a type to the variable 'n' (line 627)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 627, 16), 'n', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 627, 16), comprehension_123851))
        # Assigning a type to the variable 'd' (line 627)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 627, 16), 'd', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 627, 16), comprehension_123851))
        
        # Call to view(...): (line 627)
        # Processing the call arguments (line 627)
        
        # Obtaining an instance of the builtin type 'list' (line 627)
        list_123839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 627, 66), 'list')
        # Adding type elements to the builtin type 'list' instance (line 627)
        # Adding element type (line 627)
        
        # Obtaining an instance of the builtin type 'tuple' (line 627)
        tuple_123840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 627, 68), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 627)
        # Adding element type (line 627)
        # Getting the type of 'n' (line 627)
        n_123841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 68), 'n', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 627, 68), tuple_123840, n_123841)
        # Adding element type (line 627)
        # Getting the type of 'd' (line 627)
        d_123842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 71), 'd', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 627, 68), tuple_123840, d_123842)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 627, 66), list_123839, tuple_123840)
        
        # Processing the call keyword arguments (line 627)
        kwargs_123843 = {}
        
        # Call to array(...): (line 627)
        # Processing the call arguments (line 627)
        # Getting the type of 'a' (line 627)
        a_123829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 25), 'a', False)
        # Processing the call keyword arguments (line 627)
        # Getting the type of 'False' (line 627)
        False_123830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 33), 'False', False)
        keyword_123831 = False_123830
        # Getting the type of 'True' (line 627)
        True_123832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 46), 'True', False)
        keyword_123833 = True_123832
        # Getting the type of 'd' (line 627)
        d_123834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 58), 'd', False)
        keyword_123835 = d_123834
        kwargs_123836 = {'subok': keyword_123833, 'copy': keyword_123831, 'dtype': keyword_123835}
        # Getting the type of 'np' (line 627)
        np_123827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 16), 'np', False)
        # Obtaining the member 'array' of a type (line 627)
        array_123828 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 627, 16), np_123827, 'array')
        # Calling array(args, kwargs) (line 627)
        array_call_result_123837 = invoke(stypy.reporting.localization.Localization(__file__, 627, 16), array_123828, *[a_123829], **kwargs_123836)
        
        # Obtaining the member 'view' of a type (line 627)
        view_123838 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 627, 16), array_call_result_123837, 'view')
        # Calling view(args, kwargs) (line 627)
        view_call_result_123844 = invoke(stypy.reporting.localization.Localization(__file__, 627, 16), view_123838, *[list_123839], **kwargs_123843)
        
        list_123852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 627, 16), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 627, 16), list_123852, view_call_result_123844)
        # Assigning a type to the variable 'data' (line 627)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 627, 8), 'data', list_123852)

        if (may_be_123758 and more_types_in_union_123759):
            # SSA join for if statement (line 615)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 630):
    
    # Assigning a Call to a Name (line 630):
    
    # Call to merge_arrays(...): (line 630)
    # Processing the call arguments (line 630)
    # Getting the type of 'base' (line 630)
    base_123854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 630, 24), 'base', False)
    # Processing the call keyword arguments (line 630)
    # Getting the type of 'usemask' (line 630)
    usemask_123855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 630, 38), 'usemask', False)
    keyword_123856 = usemask_123855
    # Getting the type of 'fill_value' (line 630)
    fill_value_123857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 630, 58), 'fill_value', False)
    keyword_123858 = fill_value_123857
    kwargs_123859 = {'usemask': keyword_123856, 'fill_value': keyword_123858}
    # Getting the type of 'merge_arrays' (line 630)
    merge_arrays_123853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 630, 11), 'merge_arrays', False)
    # Calling merge_arrays(args, kwargs) (line 630)
    merge_arrays_call_result_123860 = invoke(stypy.reporting.localization.Localization(__file__, 630, 11), merge_arrays_123853, *[base_123854], **kwargs_123859)
    
    # Assigning a type to the variable 'base' (line 630)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 630, 4), 'base', merge_arrays_call_result_123860)
    
    
    
    # Call to len(...): (line 631)
    # Processing the call arguments (line 631)
    # Getting the type of 'data' (line 631)
    data_123862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 631, 11), 'data', False)
    # Processing the call keyword arguments (line 631)
    kwargs_123863 = {}
    # Getting the type of 'len' (line 631)
    len_123861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 631, 7), 'len', False)
    # Calling len(args, kwargs) (line 631)
    len_call_result_123864 = invoke(stypy.reporting.localization.Localization(__file__, 631, 7), len_123861, *[data_123862], **kwargs_123863)
    
    int_123865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 631, 19), 'int')
    # Applying the binary operator '>' (line 631)
    result_gt_123866 = python_operator(stypy.reporting.localization.Localization(__file__, 631, 7), '>', len_call_result_123864, int_123865)
    
    # Testing the type of an if condition (line 631)
    if_condition_123867 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 631, 4), result_gt_123866)
    # Assigning a type to the variable 'if_condition_123867' (line 631)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 631, 4), 'if_condition_123867', if_condition_123867)
    # SSA begins for if statement (line 631)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 632):
    
    # Assigning a Call to a Name (line 632):
    
    # Call to merge_arrays(...): (line 632)
    # Processing the call arguments (line 632)
    # Getting the type of 'data' (line 632)
    data_123869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 632, 28), 'data', False)
    # Processing the call keyword arguments (line 632)
    # Getting the type of 'True' (line 632)
    True_123870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 632, 42), 'True', False)
    keyword_123871 = True_123870
    # Getting the type of 'usemask' (line 632)
    usemask_123872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 632, 56), 'usemask', False)
    keyword_123873 = usemask_123872
    # Getting the type of 'fill_value' (line 633)
    fill_value_123874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 633, 39), 'fill_value', False)
    keyword_123875 = fill_value_123874
    kwargs_123876 = {'usemask': keyword_123873, 'fill_value': keyword_123875, 'flatten': keyword_123871}
    # Getting the type of 'merge_arrays' (line 632)
    merge_arrays_123868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 632, 15), 'merge_arrays', False)
    # Calling merge_arrays(args, kwargs) (line 632)
    merge_arrays_call_result_123877 = invoke(stypy.reporting.localization.Localization(__file__, 632, 15), merge_arrays_123868, *[data_123869], **kwargs_123876)
    
    # Assigning a type to the variable 'data' (line 632)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 632, 8), 'data', merge_arrays_call_result_123877)
    # SSA branch for the else part of an if statement (line 631)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 635):
    
    # Assigning a Call to a Name (line 635):
    
    # Call to pop(...): (line 635)
    # Processing the call keyword arguments (line 635)
    kwargs_123880 = {}
    # Getting the type of 'data' (line 635)
    data_123878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 15), 'data', False)
    # Obtaining the member 'pop' of a type (line 635)
    pop_123879 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 635, 15), data_123878, 'pop')
    # Calling pop(args, kwargs) (line 635)
    pop_call_result_123881 = invoke(stypy.reporting.localization.Localization(__file__, 635, 15), pop_123879, *[], **kwargs_123880)
    
    # Assigning a type to the variable 'data' (line 635)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 635, 8), 'data', pop_call_result_123881)
    # SSA join for if statement (line 631)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 637):
    
    # Assigning a Call to a Name (line 637):
    
    # Call to masked_all(...): (line 637)
    # Processing the call arguments (line 637)
    
    # Call to max(...): (line 637)
    # Processing the call arguments (line 637)
    
    # Call to len(...): (line 637)
    # Processing the call arguments (line 637)
    # Getting the type of 'base' (line 637)
    base_123886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 35), 'base', False)
    # Processing the call keyword arguments (line 637)
    kwargs_123887 = {}
    # Getting the type of 'len' (line 637)
    len_123885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 31), 'len', False)
    # Calling len(args, kwargs) (line 637)
    len_call_result_123888 = invoke(stypy.reporting.localization.Localization(__file__, 637, 31), len_123885, *[base_123886], **kwargs_123887)
    
    
    # Call to len(...): (line 637)
    # Processing the call arguments (line 637)
    # Getting the type of 'data' (line 637)
    data_123890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 46), 'data', False)
    # Processing the call keyword arguments (line 637)
    kwargs_123891 = {}
    # Getting the type of 'len' (line 637)
    len_123889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 42), 'len', False)
    # Calling len(args, kwargs) (line 637)
    len_call_result_123892 = invoke(stypy.reporting.localization.Localization(__file__, 637, 42), len_123889, *[data_123890], **kwargs_123891)
    
    # Processing the call keyword arguments (line 637)
    kwargs_123893 = {}
    # Getting the type of 'max' (line 637)
    max_123884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 27), 'max', False)
    # Calling max(args, kwargs) (line 637)
    max_call_result_123894 = invoke(stypy.reporting.localization.Localization(__file__, 637, 27), max_123884, *[len_call_result_123888, len_call_result_123892], **kwargs_123893)
    
    # Processing the call keyword arguments (line 637)
    # Getting the type of 'base' (line 638)
    base_123895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 33), 'base', False)
    # Obtaining the member 'dtype' of a type (line 638)
    dtype_123896 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 638, 33), base_123895, 'dtype')
    # Obtaining the member 'descr' of a type (line 638)
    descr_123897 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 638, 33), dtype_123896, 'descr')
    # Getting the type of 'data' (line 638)
    data_123898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 52), 'data', False)
    # Obtaining the member 'dtype' of a type (line 638)
    dtype_123899 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 638, 52), data_123898, 'dtype')
    # Obtaining the member 'descr' of a type (line 638)
    descr_123900 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 638, 52), dtype_123899, 'descr')
    # Applying the binary operator '+' (line 638)
    result_add_123901 = python_operator(stypy.reporting.localization.Localization(__file__, 638, 33), '+', descr_123897, descr_123900)
    
    keyword_123902 = result_add_123901
    kwargs_123903 = {'dtype': keyword_123902}
    # Getting the type of 'ma' (line 637)
    ma_123882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 13), 'ma', False)
    # Obtaining the member 'masked_all' of a type (line 637)
    masked_all_123883 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 637, 13), ma_123882, 'masked_all')
    # Calling masked_all(args, kwargs) (line 637)
    masked_all_call_result_123904 = invoke(stypy.reporting.localization.Localization(__file__, 637, 13), masked_all_123883, *[max_call_result_123894], **kwargs_123903)
    
    # Assigning a type to the variable 'output' (line 637)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 637, 4), 'output', masked_all_call_result_123904)
    
    # Assigning a Call to a Name (line 639):
    
    # Assigning a Call to a Name (line 639):
    
    # Call to recursive_fill_fields(...): (line 639)
    # Processing the call arguments (line 639)
    # Getting the type of 'base' (line 639)
    base_123906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 639, 35), 'base', False)
    # Getting the type of 'output' (line 639)
    output_123907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 639, 41), 'output', False)
    # Processing the call keyword arguments (line 639)
    kwargs_123908 = {}
    # Getting the type of 'recursive_fill_fields' (line 639)
    recursive_fill_fields_123905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 639, 13), 'recursive_fill_fields', False)
    # Calling recursive_fill_fields(args, kwargs) (line 639)
    recursive_fill_fields_call_result_123909 = invoke(stypy.reporting.localization.Localization(__file__, 639, 13), recursive_fill_fields_123905, *[base_123906, output_123907], **kwargs_123908)
    
    # Assigning a type to the variable 'output' (line 639)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 639, 4), 'output', recursive_fill_fields_call_result_123909)
    
    # Assigning a Call to a Name (line 640):
    
    # Assigning a Call to a Name (line 640):
    
    # Call to recursive_fill_fields(...): (line 640)
    # Processing the call arguments (line 640)
    # Getting the type of 'data' (line 640)
    data_123911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 35), 'data', False)
    # Getting the type of 'output' (line 640)
    output_123912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 41), 'output', False)
    # Processing the call keyword arguments (line 640)
    kwargs_123913 = {}
    # Getting the type of 'recursive_fill_fields' (line 640)
    recursive_fill_fields_123910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 13), 'recursive_fill_fields', False)
    # Calling recursive_fill_fields(args, kwargs) (line 640)
    recursive_fill_fields_call_result_123914 = invoke(stypy.reporting.localization.Localization(__file__, 640, 13), recursive_fill_fields_123910, *[data_123911, output_123912], **kwargs_123913)
    
    # Assigning a type to the variable 'output' (line 640)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 640, 4), 'output', recursive_fill_fields_call_result_123914)
    
    # Call to _fix_output(...): (line 642)
    # Processing the call arguments (line 642)
    # Getting the type of 'output' (line 642)
    output_123916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 642, 23), 'output', False)
    # Processing the call keyword arguments (line 642)
    # Getting the type of 'usemask' (line 642)
    usemask_123917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 642, 39), 'usemask', False)
    keyword_123918 = usemask_123917
    # Getting the type of 'asrecarray' (line 642)
    asrecarray_123919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 642, 59), 'asrecarray', False)
    keyword_123920 = asrecarray_123919
    kwargs_123921 = {'usemask': keyword_123918, 'asrecarray': keyword_123920}
    # Getting the type of '_fix_output' (line 642)
    _fix_output_123915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 642, 11), '_fix_output', False)
    # Calling _fix_output(args, kwargs) (line 642)
    _fix_output_call_result_123922 = invoke(stypy.reporting.localization.Localization(__file__, 642, 11), _fix_output_123915, *[output_123916], **kwargs_123921)
    
    # Assigning a type to the variable 'stypy_return_type' (line 642)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 642, 4), 'stypy_return_type', _fix_output_call_result_123922)
    
    # ################# End of 'append_fields(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'append_fields' in the type store
    # Getting the type of 'stypy_return_type' (line 576)
    stypy_return_type_123923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_123923)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'append_fields'
    return stypy_return_type_123923

# Assigning a type to the variable 'append_fields' (line 576)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 576, 0), 'append_fields', append_fields)

@norecursion
def rec_append_fields(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 645)
    None_123924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 645, 48), 'None')
    defaults = [None_123924]
    # Create a new context for function 'rec_append_fields'
    module_type_store = module_type_store.open_function_context('rec_append_fields', 645, 0, False)
    
    # Passed parameters checking function
    rec_append_fields.stypy_localization = localization
    rec_append_fields.stypy_type_of_self = None
    rec_append_fields.stypy_type_store = module_type_store
    rec_append_fields.stypy_function_name = 'rec_append_fields'
    rec_append_fields.stypy_param_names_list = ['base', 'names', 'data', 'dtypes']
    rec_append_fields.stypy_varargs_param_name = None
    rec_append_fields.stypy_kwargs_param_name = None
    rec_append_fields.stypy_call_defaults = defaults
    rec_append_fields.stypy_call_varargs = varargs
    rec_append_fields.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'rec_append_fields', ['base', 'names', 'data', 'dtypes'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'rec_append_fields', localization, ['base', 'names', 'data', 'dtypes'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'rec_append_fields(...)' code ##################

    str_123925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 674, (-1)), 'str', '\n    Add new fields to an existing array.\n\n    The names of the fields are given with the `names` arguments,\n    the corresponding values with the `data` arguments.\n    If a single field is appended, `names`, `data` and `dtypes` do not have\n    to be lists but just values.\n\n    Parameters\n    ----------\n    base : array\n        Input array to extend.\n    names : string, sequence\n        String or sequence of strings corresponding to the names\n        of the new fields.\n    data : array or sequence of arrays\n        Array or sequence of arrays storing the fields to add to the base.\n    dtypes : sequence of datatypes, optional\n        Datatype or sequence of datatypes.\n        If None, the datatypes are estimated from the `data`.\n\n    See Also\n    --------\n    append_fields\n\n    Returns\n    -------\n    appended_array : np.recarray\n    ')
    
    # Call to append_fields(...): (line 675)
    # Processing the call arguments (line 675)
    # Getting the type of 'base' (line 675)
    base_123927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 25), 'base', False)
    # Getting the type of 'names' (line 675)
    names_123928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 31), 'names', False)
    # Processing the call keyword arguments (line 675)
    # Getting the type of 'data' (line 675)
    data_123929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 43), 'data', False)
    keyword_123930 = data_123929
    # Getting the type of 'dtypes' (line 675)
    dtypes_123931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 56), 'dtypes', False)
    keyword_123932 = dtypes_123931
    # Getting the type of 'True' (line 676)
    True_123933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 676, 36), 'True', False)
    keyword_123934 = True_123933
    # Getting the type of 'False' (line 676)
    False_123935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 676, 50), 'False', False)
    keyword_123936 = False_123935
    kwargs_123937 = {'dtypes': keyword_123932, 'usemask': keyword_123936, 'data': keyword_123930, 'asrecarray': keyword_123934}
    # Getting the type of 'append_fields' (line 675)
    append_fields_123926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 11), 'append_fields', False)
    # Calling append_fields(args, kwargs) (line 675)
    append_fields_call_result_123938 = invoke(stypy.reporting.localization.Localization(__file__, 675, 11), append_fields_123926, *[base_123927, names_123928], **kwargs_123937)
    
    # Assigning a type to the variable 'stypy_return_type' (line 675)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 675, 4), 'stypy_return_type', append_fields_call_result_123938)
    
    # ################# End of 'rec_append_fields(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'rec_append_fields' in the type store
    # Getting the type of 'stypy_return_type' (line 645)
    stypy_return_type_123939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 645, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_123939)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'rec_append_fields'
    return stypy_return_type_123939

# Assigning a type to the variable 'rec_append_fields' (line 645)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 645, 0), 'rec_append_fields', rec_append_fields)

@norecursion
def stack_arrays(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 679)
    None_123940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 34), 'None')
    # Getting the type of 'True' (line 679)
    True_123941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 48), 'True')
    # Getting the type of 'False' (line 679)
    False_123942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 65), 'False')
    # Getting the type of 'False' (line 680)
    False_123943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 680, 29), 'False')
    defaults = [None_123940, True_123941, False_123942, False_123943]
    # Create a new context for function 'stack_arrays'
    module_type_store = module_type_store.open_function_context('stack_arrays', 679, 0, False)
    
    # Passed parameters checking function
    stack_arrays.stypy_localization = localization
    stack_arrays.stypy_type_of_self = None
    stack_arrays.stypy_type_store = module_type_store
    stack_arrays.stypy_function_name = 'stack_arrays'
    stack_arrays.stypy_param_names_list = ['arrays', 'defaults', 'usemask', 'asrecarray', 'autoconvert']
    stack_arrays.stypy_varargs_param_name = None
    stack_arrays.stypy_kwargs_param_name = None
    stack_arrays.stypy_call_defaults = defaults
    stack_arrays.stypy_call_varargs = varargs
    stack_arrays.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'stack_arrays', ['arrays', 'defaults', 'usemask', 'asrecarray', 'autoconvert'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'stack_arrays', localization, ['arrays', 'defaults', 'usemask', 'asrecarray', 'autoconvert'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'stack_arrays(...)' code ##################

    str_123944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 717, (-1)), 'str', "\n    Superposes arrays fields by fields\n\n    Parameters\n    ----------\n    arrays : array or sequence\n        Sequence of input arrays.\n    defaults : dictionary, optional\n        Dictionary mapping field names to the corresponding default values.\n    usemask : {True, False}, optional\n        Whether to return a MaskedArray (or MaskedRecords is\n        `asrecarray==True`) or a ndarray.\n    asrecarray : {False, True}, optional\n        Whether to return a recarray (or MaskedRecords if `usemask==True`)\n        or just a flexible-type ndarray.\n    autoconvert : {False, True}, optional\n        Whether automatically cast the type of the field to the maximum.\n\n    Examples\n    --------\n    >>> from numpy.lib import recfunctions as rfn\n    >>> x = np.array([1, 2,])\n    >>> rfn.stack_arrays(x) is x\n    True\n    >>> z = np.array([('A', 1), ('B', 2)], dtype=[('A', '|S3'), ('B', float)])\n    >>> zz = np.array([('a', 10., 100.), ('b', 20., 200.), ('c', 30., 300.)],\n    ...   dtype=[('A', '|S3'), ('B', float), ('C', float)])\n    >>> test = rfn.stack_arrays((z,zz))\n    >>> test\n    masked_array(data = [('A', 1.0, --) ('B', 2.0, --) ('a', 10.0, 100.0) ('b', 20.0, 200.0)\n     ('c', 30.0, 300.0)],\n                 mask = [(False, False, True) (False, False, True) (False, False, False)\n     (False, False, False) (False, False, False)],\n           fill_value = ('N/A', 1e+20, 1e+20),\n                dtype = [('A', '|S3'), ('B', '<f8'), ('C', '<f8')])\n\n    ")
    
    
    # Call to isinstance(...): (line 718)
    # Processing the call arguments (line 718)
    # Getting the type of 'arrays' (line 718)
    arrays_123946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 18), 'arrays', False)
    # Getting the type of 'ndarray' (line 718)
    ndarray_123947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 26), 'ndarray', False)
    # Processing the call keyword arguments (line 718)
    kwargs_123948 = {}
    # Getting the type of 'isinstance' (line 718)
    isinstance_123945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 7), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 718)
    isinstance_call_result_123949 = invoke(stypy.reporting.localization.Localization(__file__, 718, 7), isinstance_123945, *[arrays_123946, ndarray_123947], **kwargs_123948)
    
    # Testing the type of an if condition (line 718)
    if_condition_123950 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 718, 4), isinstance_call_result_123949)
    # Assigning a type to the variable 'if_condition_123950' (line 718)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 718, 4), 'if_condition_123950', if_condition_123950)
    # SSA begins for if statement (line 718)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'arrays' (line 719)
    arrays_123951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 719, 15), 'arrays')
    # Assigning a type to the variable 'stypy_return_type' (line 719)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 719, 8), 'stypy_return_type', arrays_123951)
    # SSA branch for the else part of an if statement (line 718)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Call to len(...): (line 720)
    # Processing the call arguments (line 720)
    # Getting the type of 'arrays' (line 720)
    arrays_123953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 720, 13), 'arrays', False)
    # Processing the call keyword arguments (line 720)
    kwargs_123954 = {}
    # Getting the type of 'len' (line 720)
    len_123952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 720, 9), 'len', False)
    # Calling len(args, kwargs) (line 720)
    len_call_result_123955 = invoke(stypy.reporting.localization.Localization(__file__, 720, 9), len_123952, *[arrays_123953], **kwargs_123954)
    
    int_123956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 720, 24), 'int')
    # Applying the binary operator '==' (line 720)
    result_eq_123957 = python_operator(stypy.reporting.localization.Localization(__file__, 720, 9), '==', len_call_result_123955, int_123956)
    
    # Testing the type of an if condition (line 720)
    if_condition_123958 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 720, 9), result_eq_123957)
    # Assigning a type to the variable 'if_condition_123958' (line 720)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 720, 9), 'if_condition_123958', if_condition_123958)
    # SSA begins for if statement (line 720)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining the type of the subscript
    int_123959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 721, 22), 'int')
    # Getting the type of 'arrays' (line 721)
    arrays_123960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 721, 15), 'arrays')
    # Obtaining the member '__getitem__' of a type (line 721)
    getitem___123961 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 721, 15), arrays_123960, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 721)
    subscript_call_result_123962 = invoke(stypy.reporting.localization.Localization(__file__, 721, 15), getitem___123961, int_123959)
    
    # Assigning a type to the variable 'stypy_return_type' (line 721)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 721, 8), 'stypy_return_type', subscript_call_result_123962)
    # SSA join for if statement (line 720)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 718)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a ListComp to a Name (line 722):
    
    # Assigning a ListComp to a Name (line 722):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'arrays' (line 722)
    arrays_123971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 722, 51), 'arrays')
    comprehension_123972 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 722, 17), arrays_123971)
    # Assigning a type to the variable 'a' (line 722)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 722, 17), 'a', comprehension_123972)
    
    # Call to ravel(...): (line 722)
    # Processing the call keyword arguments (line 722)
    kwargs_123969 = {}
    
    # Call to asanyarray(...): (line 722)
    # Processing the call arguments (line 722)
    # Getting the type of 'a' (line 722)
    a_123965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 722, 31), 'a', False)
    # Processing the call keyword arguments (line 722)
    kwargs_123966 = {}
    # Getting the type of 'np' (line 722)
    np_123963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 722, 17), 'np', False)
    # Obtaining the member 'asanyarray' of a type (line 722)
    asanyarray_123964 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 722, 17), np_123963, 'asanyarray')
    # Calling asanyarray(args, kwargs) (line 722)
    asanyarray_call_result_123967 = invoke(stypy.reporting.localization.Localization(__file__, 722, 17), asanyarray_123964, *[a_123965], **kwargs_123966)
    
    # Obtaining the member 'ravel' of a type (line 722)
    ravel_123968 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 722, 17), asanyarray_call_result_123967, 'ravel')
    # Calling ravel(args, kwargs) (line 722)
    ravel_call_result_123970 = invoke(stypy.reporting.localization.Localization(__file__, 722, 17), ravel_123968, *[], **kwargs_123969)
    
    list_123973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 722, 17), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 722, 17), list_123973, ravel_call_result_123970)
    # Assigning a type to the variable 'seqarrays' (line 722)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 722, 4), 'seqarrays', list_123973)
    
    # Assigning a ListComp to a Name (line 723):
    
    # Assigning a ListComp to a Name (line 723):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'seqarrays' (line 723)
    seqarrays_123978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 723, 32), 'seqarrays')
    comprehension_123979 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 723, 16), seqarrays_123978)
    # Assigning a type to the variable 'a' (line 723)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 723, 16), 'a', comprehension_123979)
    
    # Call to len(...): (line 723)
    # Processing the call arguments (line 723)
    # Getting the type of 'a' (line 723)
    a_123975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 723, 20), 'a', False)
    # Processing the call keyword arguments (line 723)
    kwargs_123976 = {}
    # Getting the type of 'len' (line 723)
    len_123974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 723, 16), 'len', False)
    # Calling len(args, kwargs) (line 723)
    len_call_result_123977 = invoke(stypy.reporting.localization.Localization(__file__, 723, 16), len_123974, *[a_123975], **kwargs_123976)
    
    list_123980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 723, 16), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 723, 16), list_123980, len_call_result_123977)
    # Assigning a type to the variable 'nrecords' (line 723)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 723, 4), 'nrecords', list_123980)
    
    # Assigning a ListComp to a Name (line 724):
    
    # Assigning a ListComp to a Name (line 724):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'seqarrays' (line 724)
    seqarrays_123983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 724, 31), 'seqarrays')
    comprehension_123984 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 724, 14), seqarrays_123983)
    # Assigning a type to the variable 'a' (line 724)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 724, 14), 'a', comprehension_123984)
    # Getting the type of 'a' (line 724)
    a_123981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 724, 14), 'a')
    # Obtaining the member 'dtype' of a type (line 724)
    dtype_123982 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 724, 14), a_123981, 'dtype')
    list_123985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 724, 14), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 724, 14), list_123985, dtype_123982)
    # Assigning a type to the variable 'ndtype' (line 724)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 724, 4), 'ndtype', list_123985)
    
    # Assigning a ListComp to a Name (line 725):
    
    # Assigning a ListComp to a Name (line 725):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'ndtype' (line 725)
    ndtype_123988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 725, 33), 'ndtype')
    comprehension_123989 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 725, 16), ndtype_123988)
    # Assigning a type to the variable 'd' (line 725)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 725, 16), 'd', comprehension_123989)
    # Getting the type of 'd' (line 725)
    d_123986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 725, 16), 'd')
    # Obtaining the member 'names' of a type (line 725)
    names_123987 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 725, 16), d_123986, 'names')
    list_123990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 725, 16), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 725, 16), list_123990, names_123987)
    # Assigning a type to the variable 'fldnames' (line 725)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 725, 4), 'fldnames', list_123990)
    
    # Assigning a Subscript to a Name (line 727):
    
    # Assigning a Subscript to a Name (line 727):
    
    # Obtaining the type of the subscript
    int_123991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 727, 21), 'int')
    # Getting the type of 'ndtype' (line 727)
    ndtype_123992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 727, 14), 'ndtype')
    # Obtaining the member '__getitem__' of a type (line 727)
    getitem___123993 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 727, 14), ndtype_123992, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 727)
    subscript_call_result_123994 = invoke(stypy.reporting.localization.Localization(__file__, 727, 14), getitem___123993, int_123991)
    
    # Assigning a type to the variable 'dtype_l' (line 727)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 727, 4), 'dtype_l', subscript_call_result_123994)
    
    # Assigning a Attribute to a Name (line 728):
    
    # Assigning a Attribute to a Name (line 728):
    # Getting the type of 'dtype_l' (line 728)
    dtype_l_123995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 728, 15), 'dtype_l')
    # Obtaining the member 'descr' of a type (line 728)
    descr_123996 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 728, 15), dtype_l_123995, 'descr')
    # Assigning a type to the variable 'newdescr' (line 728)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 728, 4), 'newdescr', descr_123996)
    
    # Assigning a ListComp to a Name (line 729):
    
    # Assigning a ListComp to a Name (line 729):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'newdescr' (line 729)
    newdescr_124001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 729, 27), 'newdescr')
    comprehension_124002 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 729, 13), newdescr_124001)
    # Assigning a type to the variable '_' (line 729)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 729, 13), '_', comprehension_124002)
    
    # Obtaining the type of the subscript
    int_123997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 729, 15), 'int')
    # Getting the type of '_' (line 729)
    __123998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 729, 13), '_')
    # Obtaining the member '__getitem__' of a type (line 729)
    getitem___123999 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 729, 13), __123998, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 729)
    subscript_call_result_124000 = invoke(stypy.reporting.localization.Localization(__file__, 729, 13), getitem___123999, int_123997)
    
    list_124003 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 729, 13), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 729, 13), list_124003, subscript_call_result_124000)
    # Assigning a type to the variable 'names' (line 729)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 729, 4), 'names', list_124003)
    
    
    # Obtaining the type of the subscript
    int_124004 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 730, 26), 'int')
    slice_124005 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 730, 19), int_124004, None, None)
    # Getting the type of 'ndtype' (line 730)
    ndtype_124006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 730, 19), 'ndtype')
    # Obtaining the member '__getitem__' of a type (line 730)
    getitem___124007 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 730, 19), ndtype_124006, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 730)
    subscript_call_result_124008 = invoke(stypy.reporting.localization.Localization(__file__, 730, 19), getitem___124007, slice_124005)
    
    # Testing the type of a for loop iterable (line 730)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 730, 4), subscript_call_result_124008)
    # Getting the type of the for loop variable (line 730)
    for_loop_var_124009 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 730, 4), subscript_call_result_124008)
    # Assigning a type to the variable 'dtype_n' (line 730)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 730, 4), 'dtype_n', for_loop_var_124009)
    # SSA begins for a for statement (line 730)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Getting the type of 'dtype_n' (line 731)
    dtype_n_124010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 731, 21), 'dtype_n')
    # Obtaining the member 'descr' of a type (line 731)
    descr_124011 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 731, 21), dtype_n_124010, 'descr')
    # Testing the type of a for loop iterable (line 731)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 731, 8), descr_124011)
    # Getting the type of the for loop variable (line 731)
    for_loop_var_124012 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 731, 8), descr_124011)
    # Assigning a type to the variable 'descr' (line 731)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 731, 8), 'descr', for_loop_var_124012)
    # SSA begins for a for statement (line 731)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a BoolOp to a Name (line 732):
    
    # Assigning a BoolOp to a Name (line 732):
    
    # Evaluating a boolean operation
    
    # Obtaining the type of the subscript
    int_124013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 732, 25), 'int')
    # Getting the type of 'descr' (line 732)
    descr_124014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 732, 19), 'descr')
    # Obtaining the member '__getitem__' of a type (line 732)
    getitem___124015 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 732, 19), descr_124014, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 732)
    subscript_call_result_124016 = invoke(stypy.reporting.localization.Localization(__file__, 732, 19), getitem___124015, int_124013)
    
    str_124017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 732, 31), 'str', '')
    # Applying the binary operator 'or' (line 732)
    result_or_keyword_124018 = python_operator(stypy.reporting.localization.Localization(__file__, 732, 19), 'or', subscript_call_result_124016, str_124017)
    
    # Assigning a type to the variable 'name' (line 732)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 732, 12), 'name', result_or_keyword_124018)
    
    
    # Getting the type of 'name' (line 733)
    name_124019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 733, 15), 'name')
    # Getting the type of 'names' (line 733)
    names_124020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 733, 27), 'names')
    # Applying the binary operator 'notin' (line 733)
    result_contains_124021 = python_operator(stypy.reporting.localization.Localization(__file__, 733, 15), 'notin', name_124019, names_124020)
    
    # Testing the type of an if condition (line 733)
    if_condition_124022 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 733, 12), result_contains_124021)
    # Assigning a type to the variable 'if_condition_124022' (line 733)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 733, 12), 'if_condition_124022', if_condition_124022)
    # SSA begins for if statement (line 733)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 734)
    # Processing the call arguments (line 734)
    # Getting the type of 'descr' (line 734)
    descr_124025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 734, 32), 'descr', False)
    # Processing the call keyword arguments (line 734)
    kwargs_124026 = {}
    # Getting the type of 'newdescr' (line 734)
    newdescr_124023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 734, 16), 'newdescr', False)
    # Obtaining the member 'append' of a type (line 734)
    append_124024 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 734, 16), newdescr_124023, 'append')
    # Calling append(args, kwargs) (line 734)
    append_call_result_124027 = invoke(stypy.reporting.localization.Localization(__file__, 734, 16), append_124024, *[descr_124025], **kwargs_124026)
    
    
    # Call to append(...): (line 735)
    # Processing the call arguments (line 735)
    # Getting the type of 'name' (line 735)
    name_124030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 735, 29), 'name', False)
    # Processing the call keyword arguments (line 735)
    kwargs_124031 = {}
    # Getting the type of 'names' (line 735)
    names_124028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 735, 16), 'names', False)
    # Obtaining the member 'append' of a type (line 735)
    append_124029 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 735, 16), names_124028, 'append')
    # Calling append(args, kwargs) (line 735)
    append_call_result_124032 = invoke(stypy.reporting.localization.Localization(__file__, 735, 16), append_124029, *[name_124030], **kwargs_124031)
    
    # SSA branch for the else part of an if statement (line 733)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 737):
    
    # Assigning a Call to a Name (line 737):
    
    # Call to index(...): (line 737)
    # Processing the call arguments (line 737)
    # Getting the type of 'name' (line 737)
    name_124035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 737, 38), 'name', False)
    # Processing the call keyword arguments (line 737)
    kwargs_124036 = {}
    # Getting the type of 'names' (line 737)
    names_124033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 737, 26), 'names', False)
    # Obtaining the member 'index' of a type (line 737)
    index_124034 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 737, 26), names_124033, 'index')
    # Calling index(args, kwargs) (line 737)
    index_call_result_124037 = invoke(stypy.reporting.localization.Localization(__file__, 737, 26), index_124034, *[name_124035], **kwargs_124036)
    
    # Assigning a type to the variable 'nameidx' (line 737)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 737, 16), 'nameidx', index_call_result_124037)
    
    # Assigning a Subscript to a Name (line 738):
    
    # Assigning a Subscript to a Name (line 738):
    
    # Obtaining the type of the subscript
    # Getting the type of 'nameidx' (line 738)
    nameidx_124038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 738, 41), 'nameidx')
    # Getting the type of 'newdescr' (line 738)
    newdescr_124039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 738, 32), 'newdescr')
    # Obtaining the member '__getitem__' of a type (line 738)
    getitem___124040 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 738, 32), newdescr_124039, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 738)
    subscript_call_result_124041 = invoke(stypy.reporting.localization.Localization(__file__, 738, 32), getitem___124040, nameidx_124038)
    
    # Assigning a type to the variable 'current_descr' (line 738)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 738, 16), 'current_descr', subscript_call_result_124041)
    
    # Getting the type of 'autoconvert' (line 739)
    autoconvert_124042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 739, 19), 'autoconvert')
    # Testing the type of an if condition (line 739)
    if_condition_124043 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 739, 16), autoconvert_124042)
    # Assigning a type to the variable 'if_condition_124043' (line 739)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 739, 16), 'if_condition_124043', if_condition_124043)
    # SSA begins for if statement (line 739)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    
    # Call to dtype(...): (line 740)
    # Processing the call arguments (line 740)
    
    # Obtaining the type of the subscript
    int_124046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 740, 38), 'int')
    # Getting the type of 'descr' (line 740)
    descr_124047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 740, 32), 'descr', False)
    # Obtaining the member '__getitem__' of a type (line 740)
    getitem___124048 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 740, 32), descr_124047, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 740)
    subscript_call_result_124049 = invoke(stypy.reporting.localization.Localization(__file__, 740, 32), getitem___124048, int_124046)
    
    # Processing the call keyword arguments (line 740)
    kwargs_124050 = {}
    # Getting the type of 'np' (line 740)
    np_124044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 740, 23), 'np', False)
    # Obtaining the member 'dtype' of a type (line 740)
    dtype_124045 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 740, 23), np_124044, 'dtype')
    # Calling dtype(args, kwargs) (line 740)
    dtype_call_result_124051 = invoke(stypy.reporting.localization.Localization(__file__, 740, 23), dtype_124045, *[subscript_call_result_124049], **kwargs_124050)
    
    
    # Call to dtype(...): (line 740)
    # Processing the call arguments (line 740)
    
    # Obtaining the type of the subscript
    int_124054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 740, 67), 'int')
    # Getting the type of 'current_descr' (line 740)
    current_descr_124055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 740, 53), 'current_descr', False)
    # Obtaining the member '__getitem__' of a type (line 740)
    getitem___124056 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 740, 53), current_descr_124055, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 740)
    subscript_call_result_124057 = invoke(stypy.reporting.localization.Localization(__file__, 740, 53), getitem___124056, int_124054)
    
    # Processing the call keyword arguments (line 740)
    kwargs_124058 = {}
    # Getting the type of 'np' (line 740)
    np_124052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 740, 44), 'np', False)
    # Obtaining the member 'dtype' of a type (line 740)
    dtype_124053 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 740, 44), np_124052, 'dtype')
    # Calling dtype(args, kwargs) (line 740)
    dtype_call_result_124059 = invoke(stypy.reporting.localization.Localization(__file__, 740, 44), dtype_124053, *[subscript_call_result_124057], **kwargs_124058)
    
    # Applying the binary operator '>' (line 740)
    result_gt_124060 = python_operator(stypy.reporting.localization.Localization(__file__, 740, 23), '>', dtype_call_result_124051, dtype_call_result_124059)
    
    # Testing the type of an if condition (line 740)
    if_condition_124061 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 740, 20), result_gt_124060)
    # Assigning a type to the variable 'if_condition_124061' (line 740)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 740, 20), 'if_condition_124061', if_condition_124061)
    # SSA begins for if statement (line 740)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 741):
    
    # Assigning a Call to a Name (line 741):
    
    # Call to list(...): (line 741)
    # Processing the call arguments (line 741)
    # Getting the type of 'current_descr' (line 741)
    current_descr_124063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 741, 45), 'current_descr', False)
    # Processing the call keyword arguments (line 741)
    kwargs_124064 = {}
    # Getting the type of 'list' (line 741)
    list_124062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 741, 40), 'list', False)
    # Calling list(args, kwargs) (line 741)
    list_call_result_124065 = invoke(stypy.reporting.localization.Localization(__file__, 741, 40), list_124062, *[current_descr_124063], **kwargs_124064)
    
    # Assigning a type to the variable 'current_descr' (line 741)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 741, 24), 'current_descr', list_call_result_124065)
    
    # Assigning a Subscript to a Subscript (line 742):
    
    # Assigning a Subscript to a Subscript (line 742):
    
    # Obtaining the type of the subscript
    int_124066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 742, 50), 'int')
    # Getting the type of 'descr' (line 742)
    descr_124067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 44), 'descr')
    # Obtaining the member '__getitem__' of a type (line 742)
    getitem___124068 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 742, 44), descr_124067, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 742)
    subscript_call_result_124069 = invoke(stypy.reporting.localization.Localization(__file__, 742, 44), getitem___124068, int_124066)
    
    # Getting the type of 'current_descr' (line 742)
    current_descr_124070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 24), 'current_descr')
    int_124071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 742, 38), 'int')
    # Storing an element on a container (line 742)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 742, 24), current_descr_124070, (int_124071, subscript_call_result_124069))
    
    # Assigning a Call to a Subscript (line 743):
    
    # Assigning a Call to a Subscript (line 743):
    
    # Call to tuple(...): (line 743)
    # Processing the call arguments (line 743)
    # Getting the type of 'current_descr' (line 743)
    current_descr_124073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 743, 50), 'current_descr', False)
    # Processing the call keyword arguments (line 743)
    kwargs_124074 = {}
    # Getting the type of 'tuple' (line 743)
    tuple_124072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 743, 44), 'tuple', False)
    # Calling tuple(args, kwargs) (line 743)
    tuple_call_result_124075 = invoke(stypy.reporting.localization.Localization(__file__, 743, 44), tuple_124072, *[current_descr_124073], **kwargs_124074)
    
    # Getting the type of 'newdescr' (line 743)
    newdescr_124076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 743, 24), 'newdescr')
    # Getting the type of 'nameidx' (line 743)
    nameidx_124077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 743, 33), 'nameidx')
    # Storing an element on a container (line 743)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 743, 24), newdescr_124076, (nameidx_124077, tuple_call_result_124075))
    # SSA join for if statement (line 740)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 739)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Obtaining the type of the subscript
    int_124078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 744, 27), 'int')
    # Getting the type of 'descr' (line 744)
    descr_124079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 744, 21), 'descr')
    # Obtaining the member '__getitem__' of a type (line 744)
    getitem___124080 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 744, 21), descr_124079, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 744)
    subscript_call_result_124081 = invoke(stypy.reporting.localization.Localization(__file__, 744, 21), getitem___124080, int_124078)
    
    
    # Obtaining the type of the subscript
    int_124082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 744, 47), 'int')
    # Getting the type of 'current_descr' (line 744)
    current_descr_124083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 744, 33), 'current_descr')
    # Obtaining the member '__getitem__' of a type (line 744)
    getitem___124084 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 744, 33), current_descr_124083, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 744)
    subscript_call_result_124085 = invoke(stypy.reporting.localization.Localization(__file__, 744, 33), getitem___124084, int_124082)
    
    # Applying the binary operator '!=' (line 744)
    result_ne_124086 = python_operator(stypy.reporting.localization.Localization(__file__, 744, 21), '!=', subscript_call_result_124081, subscript_call_result_124085)
    
    # Testing the type of an if condition (line 744)
    if_condition_124087 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 744, 21), result_ne_124086)
    # Assigning a type to the variable 'if_condition_124087' (line 744)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 744, 21), 'if_condition_124087', if_condition_124087)
    # SSA begins for if statement (line 744)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 745)
    # Processing the call arguments (line 745)
    str_124089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 745, 36), 'str', "Incompatible type '%s' <> '%s'")
    
    # Obtaining an instance of the builtin type 'tuple' (line 746)
    tuple_124090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 746, 37), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 746)
    # Adding element type (line 746)
    
    # Obtaining the type of the subscript
    # Getting the type of 'name' (line 746)
    name_124091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 746, 52), 'name', False)
    
    # Call to dict(...): (line 746)
    # Processing the call arguments (line 746)
    # Getting the type of 'newdescr' (line 746)
    newdescr_124093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 746, 42), 'newdescr', False)
    # Processing the call keyword arguments (line 746)
    kwargs_124094 = {}
    # Getting the type of 'dict' (line 746)
    dict_124092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 746, 37), 'dict', False)
    # Calling dict(args, kwargs) (line 746)
    dict_call_result_124095 = invoke(stypy.reporting.localization.Localization(__file__, 746, 37), dict_124092, *[newdescr_124093], **kwargs_124094)
    
    # Obtaining the member '__getitem__' of a type (line 746)
    getitem___124096 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 746, 37), dict_call_result_124095, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 746)
    subscript_call_result_124097 = invoke(stypy.reporting.localization.Localization(__file__, 746, 37), getitem___124096, name_124091)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 746, 37), tuple_124090, subscript_call_result_124097)
    # Adding element type (line 746)
    
    # Obtaining the type of the subscript
    int_124098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 746, 65), 'int')
    # Getting the type of 'descr' (line 746)
    descr_124099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 746, 59), 'descr', False)
    # Obtaining the member '__getitem__' of a type (line 746)
    getitem___124100 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 746, 59), descr_124099, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 746)
    subscript_call_result_124101 = invoke(stypy.reporting.localization.Localization(__file__, 746, 59), getitem___124100, int_124098)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 746, 37), tuple_124090, subscript_call_result_124101)
    
    # Applying the binary operator '%' (line 745)
    result_mod_124102 = python_operator(stypy.reporting.localization.Localization(__file__, 745, 36), '%', str_124089, tuple_124090)
    
    # Processing the call keyword arguments (line 745)
    kwargs_124103 = {}
    # Getting the type of 'TypeError' (line 745)
    TypeError_124088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 745, 26), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 745)
    TypeError_call_result_124104 = invoke(stypy.reporting.localization.Localization(__file__, 745, 26), TypeError_124088, *[result_mod_124102], **kwargs_124103)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 745, 20), TypeError_call_result_124104, 'raise parameter', BaseException)
    # SSA join for if statement (line 744)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 739)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 733)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to len(...): (line 748)
    # Processing the call arguments (line 748)
    # Getting the type of 'newdescr' (line 748)
    newdescr_124106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 748, 11), 'newdescr', False)
    # Processing the call keyword arguments (line 748)
    kwargs_124107 = {}
    # Getting the type of 'len' (line 748)
    len_124105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 748, 7), 'len', False)
    # Calling len(args, kwargs) (line 748)
    len_call_result_124108 = invoke(stypy.reporting.localization.Localization(__file__, 748, 7), len_124105, *[newdescr_124106], **kwargs_124107)
    
    int_124109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 748, 24), 'int')
    # Applying the binary operator '==' (line 748)
    result_eq_124110 = python_operator(stypy.reporting.localization.Localization(__file__, 748, 7), '==', len_call_result_124108, int_124109)
    
    # Testing the type of an if condition (line 748)
    if_condition_124111 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 748, 4), result_eq_124110)
    # Assigning a type to the variable 'if_condition_124111' (line 748)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 748, 4), 'if_condition_124111', if_condition_124111)
    # SSA begins for if statement (line 748)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 749):
    
    # Assigning a Call to a Name (line 749):
    
    # Call to concatenate(...): (line 749)
    # Processing the call arguments (line 749)
    # Getting the type of 'seqarrays' (line 749)
    seqarrays_124114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 749, 32), 'seqarrays', False)
    # Processing the call keyword arguments (line 749)
    kwargs_124115 = {}
    # Getting the type of 'ma' (line 749)
    ma_124112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 749, 17), 'ma', False)
    # Obtaining the member 'concatenate' of a type (line 749)
    concatenate_124113 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 749, 17), ma_124112, 'concatenate')
    # Calling concatenate(args, kwargs) (line 749)
    concatenate_call_result_124116 = invoke(stypy.reporting.localization.Localization(__file__, 749, 17), concatenate_124113, *[seqarrays_124114], **kwargs_124115)
    
    # Assigning a type to the variable 'output' (line 749)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 749, 8), 'output', concatenate_call_result_124116)
    # SSA branch for the else part of an if statement (line 748)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 752):
    
    # Assigning a Call to a Name (line 752):
    
    # Call to masked_all(...): (line 752)
    # Processing the call arguments (line 752)
    
    # Obtaining an instance of the builtin type 'tuple' (line 752)
    tuple_124119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 752, 32), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 752)
    # Adding element type (line 752)
    
    # Call to sum(...): (line 752)
    # Processing the call arguments (line 752)
    # Getting the type of 'nrecords' (line 752)
    nrecords_124122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 752, 39), 'nrecords', False)
    # Processing the call keyword arguments (line 752)
    kwargs_124123 = {}
    # Getting the type of 'np' (line 752)
    np_124120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 752, 32), 'np', False)
    # Obtaining the member 'sum' of a type (line 752)
    sum_124121 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 752, 32), np_124120, 'sum')
    # Calling sum(args, kwargs) (line 752)
    sum_call_result_124124 = invoke(stypy.reporting.localization.Localization(__file__, 752, 32), sum_124121, *[nrecords_124122], **kwargs_124123)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 752, 32), tuple_124119, sum_call_result_124124)
    
    # Getting the type of 'newdescr' (line 752)
    newdescr_124125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 752, 52), 'newdescr', False)
    # Processing the call keyword arguments (line 752)
    kwargs_124126 = {}
    # Getting the type of 'ma' (line 752)
    ma_124117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 752, 17), 'ma', False)
    # Obtaining the member 'masked_all' of a type (line 752)
    masked_all_124118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 752, 17), ma_124117, 'masked_all')
    # Calling masked_all(args, kwargs) (line 752)
    masked_all_call_result_124127 = invoke(stypy.reporting.localization.Localization(__file__, 752, 17), masked_all_124118, *[tuple_124119, newdescr_124125], **kwargs_124126)
    
    # Assigning a type to the variable 'output' (line 752)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 752, 8), 'output', masked_all_call_result_124127)
    
    # Assigning a Call to a Name (line 753):
    
    # Assigning a Call to a Name (line 753):
    
    # Call to cumsum(...): (line 753)
    # Processing the call arguments (line 753)
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 753)
    tuple_124130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 753, 33), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 753)
    # Adding element type (line 753)
    int_124131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 753, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 753, 33), tuple_124130, int_124131)
    # Adding element type (line 753)
    # Getting the type of 'nrecords' (line 753)
    nrecords_124132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 753, 36), 'nrecords', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 753, 33), tuple_124130, nrecords_124132)
    
    # Getting the type of 'np' (line 753)
    np_124133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 753, 27), 'np', False)
    # Obtaining the member 'r_' of a type (line 753)
    r__124134 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 753, 27), np_124133, 'r_')
    # Obtaining the member '__getitem__' of a type (line 753)
    getitem___124135 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 753, 27), r__124134, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 753)
    subscript_call_result_124136 = invoke(stypy.reporting.localization.Localization(__file__, 753, 27), getitem___124135, tuple_124130)
    
    # Processing the call keyword arguments (line 753)
    kwargs_124137 = {}
    # Getting the type of 'np' (line 753)
    np_124128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 753, 17), 'np', False)
    # Obtaining the member 'cumsum' of a type (line 753)
    cumsum_124129 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 753, 17), np_124128, 'cumsum')
    # Calling cumsum(args, kwargs) (line 753)
    cumsum_call_result_124138 = invoke(stypy.reporting.localization.Localization(__file__, 753, 17), cumsum_124129, *[subscript_call_result_124136], **kwargs_124137)
    
    # Assigning a type to the variable 'offset' (line 753)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 753, 8), 'offset', cumsum_call_result_124138)
    
    # Assigning a List to a Name (line 754):
    
    # Assigning a List to a Name (line 754):
    
    # Obtaining an instance of the builtin type 'list' (line 754)
    list_124139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 754, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 754)
    
    # Assigning a type to the variable 'seen' (line 754)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 754, 8), 'seen', list_124139)
    
    
    # Call to zip(...): (line 755)
    # Processing the call arguments (line 755)
    # Getting the type of 'seqarrays' (line 755)
    seqarrays_124141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 755, 32), 'seqarrays', False)
    # Getting the type of 'fldnames' (line 755)
    fldnames_124142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 755, 43), 'fldnames', False)
    
    # Obtaining the type of the subscript
    int_124143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 755, 61), 'int')
    slice_124144 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 755, 53), None, int_124143, None)
    # Getting the type of 'offset' (line 755)
    offset_124145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 755, 53), 'offset', False)
    # Obtaining the member '__getitem__' of a type (line 755)
    getitem___124146 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 755, 53), offset_124145, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 755)
    subscript_call_result_124147 = invoke(stypy.reporting.localization.Localization(__file__, 755, 53), getitem___124146, slice_124144)
    
    
    # Obtaining the type of the subscript
    int_124148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 755, 73), 'int')
    slice_124149 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 755, 66), int_124148, None, None)
    # Getting the type of 'offset' (line 755)
    offset_124150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 755, 66), 'offset', False)
    # Obtaining the member '__getitem__' of a type (line 755)
    getitem___124151 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 755, 66), offset_124150, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 755)
    subscript_call_result_124152 = invoke(stypy.reporting.localization.Localization(__file__, 755, 66), getitem___124151, slice_124149)
    
    # Processing the call keyword arguments (line 755)
    kwargs_124153 = {}
    # Getting the type of 'zip' (line 755)
    zip_124140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 755, 28), 'zip', False)
    # Calling zip(args, kwargs) (line 755)
    zip_call_result_124154 = invoke(stypy.reporting.localization.Localization(__file__, 755, 28), zip_124140, *[seqarrays_124141, fldnames_124142, subscript_call_result_124147, subscript_call_result_124152], **kwargs_124153)
    
    # Testing the type of a for loop iterable (line 755)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 755, 8), zip_call_result_124154)
    # Getting the type of the for loop variable (line 755)
    for_loop_var_124155 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 755, 8), zip_call_result_124154)
    # Assigning a type to the variable 'a' (line 755)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 755, 8), 'a', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 755, 8), for_loop_var_124155))
    # Assigning a type to the variable 'n' (line 755)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 755, 8), 'n', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 755, 8), for_loop_var_124155))
    # Assigning a type to the variable 'i' (line 755)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 755, 8), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 755, 8), for_loop_var_124155))
    # Assigning a type to the variable 'j' (line 755)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 755, 8), 'j', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 755, 8), for_loop_var_124155))
    # SSA begins for a for statement (line 755)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Attribute to a Name (line 756):
    
    # Assigning a Attribute to a Name (line 756):
    # Getting the type of 'a' (line 756)
    a_124156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 756, 20), 'a')
    # Obtaining the member 'dtype' of a type (line 756)
    dtype_124157 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 756, 20), a_124156, 'dtype')
    # Obtaining the member 'names' of a type (line 756)
    names_124158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 756, 20), dtype_124157, 'names')
    # Assigning a type to the variable 'names' (line 756)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 756, 12), 'names', names_124158)
    
    # Type idiom detected: calculating its left and rigth part (line 757)
    # Getting the type of 'names' (line 757)
    names_124159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 757, 15), 'names')
    # Getting the type of 'None' (line 757)
    None_124160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 757, 24), 'None')
    
    (may_be_124161, more_types_in_union_124162) = may_be_none(names_124159, None_124160)

    if may_be_124161:

        if more_types_in_union_124162:
            # Runtime conditional SSA (line 757)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Name to a Subscript (line 758):
        
        # Assigning a Name to a Subscript (line 758):
        # Getting the type of 'a' (line 758)
        a_124163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 758, 49), 'a')
        
        # Obtaining the type of the subscript
        str_124164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 758, 23), 'str', 'f%i')
        
        # Call to len(...): (line 758)
        # Processing the call arguments (line 758)
        # Getting the type of 'seen' (line 758)
        seen_124166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 758, 35), 'seen', False)
        # Processing the call keyword arguments (line 758)
        kwargs_124167 = {}
        # Getting the type of 'len' (line 758)
        len_124165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 758, 31), 'len', False)
        # Calling len(args, kwargs) (line 758)
        len_call_result_124168 = invoke(stypy.reporting.localization.Localization(__file__, 758, 31), len_124165, *[seen_124166], **kwargs_124167)
        
        # Applying the binary operator '%' (line 758)
        result_mod_124169 = python_operator(stypy.reporting.localization.Localization(__file__, 758, 23), '%', str_124164, len_call_result_124168)
        
        # Getting the type of 'output' (line 758)
        output_124170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 758, 16), 'output')
        # Obtaining the member '__getitem__' of a type (line 758)
        getitem___124171 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 758, 16), output_124170, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 758)
        subscript_call_result_124172 = invoke(stypy.reporting.localization.Localization(__file__, 758, 16), getitem___124171, result_mod_124169)
        
        # Getting the type of 'i' (line 758)
        i_124173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 758, 42), 'i')
        # Getting the type of 'j' (line 758)
        j_124174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 758, 44), 'j')
        slice_124175 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 758, 16), i_124173, j_124174, None)
        # Storing an element on a container (line 758)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 758, 16), subscript_call_result_124172, (slice_124175, a_124163))

        if more_types_in_union_124162:
            # Runtime conditional SSA for else branch (line 757)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_124161) or more_types_in_union_124162):
        
        # Getting the type of 'n' (line 760)
        n_124176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 760, 28), 'n')
        # Testing the type of a for loop iterable (line 760)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 760, 16), n_124176)
        # Getting the type of the for loop variable (line 760)
        for_loop_var_124177 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 760, 16), n_124176)
        # Assigning a type to the variable 'name' (line 760)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 760, 16), 'name', for_loop_var_124177)
        # SSA begins for a for statement (line 760)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Subscript to a Subscript (line 761):
        
        # Assigning a Subscript to a Subscript (line 761):
        
        # Obtaining the type of the subscript
        # Getting the type of 'name' (line 761)
        name_124178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 761, 42), 'name')
        # Getting the type of 'a' (line 761)
        a_124179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 761, 40), 'a')
        # Obtaining the member '__getitem__' of a type (line 761)
        getitem___124180 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 761, 40), a_124179, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 761)
        subscript_call_result_124181 = invoke(stypy.reporting.localization.Localization(__file__, 761, 40), getitem___124180, name_124178)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'name' (line 761)
        name_124182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 761, 27), 'name')
        # Getting the type of 'output' (line 761)
        output_124183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 761, 20), 'output')
        # Obtaining the member '__getitem__' of a type (line 761)
        getitem___124184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 761, 20), output_124183, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 761)
        subscript_call_result_124185 = invoke(stypy.reporting.localization.Localization(__file__, 761, 20), getitem___124184, name_124182)
        
        # Getting the type of 'i' (line 761)
        i_124186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 761, 33), 'i')
        # Getting the type of 'j' (line 761)
        j_124187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 761, 35), 'j')
        slice_124188 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 761, 20), i_124186, j_124187, None)
        # Storing an element on a container (line 761)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 761, 20), subscript_call_result_124185, (slice_124188, subscript_call_result_124181))
        
        
        # Getting the type of 'name' (line 762)
        name_124189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 762, 23), 'name')
        # Getting the type of 'seen' (line 762)
        seen_124190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 762, 35), 'seen')
        # Applying the binary operator 'notin' (line 762)
        result_contains_124191 = python_operator(stypy.reporting.localization.Localization(__file__, 762, 23), 'notin', name_124189, seen_124190)
        
        # Testing the type of an if condition (line 762)
        if_condition_124192 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 762, 20), result_contains_124191)
        # Assigning a type to the variable 'if_condition_124192' (line 762)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 762, 20), 'if_condition_124192', if_condition_124192)
        # SSA begins for if statement (line 762)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 763)
        # Processing the call arguments (line 763)
        # Getting the type of 'name' (line 763)
        name_124195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 763, 36), 'name', False)
        # Processing the call keyword arguments (line 763)
        kwargs_124196 = {}
        # Getting the type of 'seen' (line 763)
        seen_124193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 763, 24), 'seen', False)
        # Obtaining the member 'append' of a type (line 763)
        append_124194 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 763, 24), seen_124193, 'append')
        # Calling append(args, kwargs) (line 763)
        append_call_result_124197 = invoke(stypy.reporting.localization.Localization(__file__, 763, 24), append_124194, *[name_124195], **kwargs_124196)
        
        # SSA join for if statement (line 762)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        

        if (may_be_124161 and more_types_in_union_124162):
            # SSA join for if statement (line 757)
            module_type_store = module_type_store.join_ssa_context()


    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 748)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to _fix_output(...): (line 765)
    # Processing the call arguments (line 765)
    
    # Call to _fix_defaults(...): (line 765)
    # Processing the call arguments (line 765)
    # Getting the type of 'output' (line 765)
    output_124200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 765, 37), 'output', False)
    # Getting the type of 'defaults' (line 765)
    defaults_124201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 765, 45), 'defaults', False)
    # Processing the call keyword arguments (line 765)
    kwargs_124202 = {}
    # Getting the type of '_fix_defaults' (line 765)
    _fix_defaults_124199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 765, 23), '_fix_defaults', False)
    # Calling _fix_defaults(args, kwargs) (line 765)
    _fix_defaults_call_result_124203 = invoke(stypy.reporting.localization.Localization(__file__, 765, 23), _fix_defaults_124199, *[output_124200, defaults_124201], **kwargs_124202)
    
    # Processing the call keyword arguments (line 765)
    # Getting the type of 'usemask' (line 766)
    usemask_124204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 766, 31), 'usemask', False)
    keyword_124205 = usemask_124204
    # Getting the type of 'asrecarray' (line 766)
    asrecarray_124206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 766, 51), 'asrecarray', False)
    keyword_124207 = asrecarray_124206
    kwargs_124208 = {'usemask': keyword_124205, 'asrecarray': keyword_124207}
    # Getting the type of '_fix_output' (line 765)
    _fix_output_124198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 765, 11), '_fix_output', False)
    # Calling _fix_output(args, kwargs) (line 765)
    _fix_output_call_result_124209 = invoke(stypy.reporting.localization.Localization(__file__, 765, 11), _fix_output_124198, *[_fix_defaults_call_result_124203], **kwargs_124208)
    
    # Assigning a type to the variable 'stypy_return_type' (line 765)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 765, 4), 'stypy_return_type', _fix_output_call_result_124209)
    
    # ################# End of 'stack_arrays(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'stack_arrays' in the type store
    # Getting the type of 'stypy_return_type' (line 679)
    stypy_return_type_124210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_124210)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'stack_arrays'
    return stypy_return_type_124210

# Assigning a type to the variable 'stack_arrays' (line 679)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 679, 0), 'stack_arrays', stack_arrays)

@norecursion
def find_duplicates(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 769)
    None_124211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 769, 27), 'None')
    # Getting the type of 'True' (line 769)
    True_124212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 769, 44), 'True')
    # Getting the type of 'False' (line 769)
    False_124213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 769, 63), 'False')
    defaults = [None_124211, True_124212, False_124213]
    # Create a new context for function 'find_duplicates'
    module_type_store = module_type_store.open_function_context('find_duplicates', 769, 0, False)
    
    # Passed parameters checking function
    find_duplicates.stypy_localization = localization
    find_duplicates.stypy_type_of_self = None
    find_duplicates.stypy_type_store = module_type_store
    find_duplicates.stypy_function_name = 'find_duplicates'
    find_duplicates.stypy_param_names_list = ['a', 'key', 'ignoremask', 'return_index']
    find_duplicates.stypy_varargs_param_name = None
    find_duplicates.stypy_kwargs_param_name = None
    find_duplicates.stypy_call_defaults = defaults
    find_duplicates.stypy_call_varargs = varargs
    find_duplicates.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'find_duplicates', ['a', 'key', 'ignoremask', 'return_index'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'find_duplicates', localization, ['a', 'key', 'ignoremask', 'return_index'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'find_duplicates(...)' code ##################

    str_124214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 793, (-1)), 'str', "\n    Find the duplicates in a structured array along a given key\n\n    Parameters\n    ----------\n    a : array-like\n        Input array\n    key : {string, None}, optional\n        Name of the fields along which to check the duplicates.\n        If None, the search is performed by records\n    ignoremask : {True, False}, optional\n        Whether masked data should be discarded or considered as duplicates.\n    return_index : {False, True}, optional\n        Whether to return the indices of the duplicated values.\n\n    Examples\n    --------\n    >>> from numpy.lib import recfunctions as rfn\n    >>> ndtype = [('a', int)]\n    >>> a = np.ma.array([1, 1, 1, 2, 2, 3, 3],\n    ...         mask=[0, 0, 1, 0, 0, 0, 1]).view(ndtype)\n    >>> rfn.find_duplicates(a, ignoremask=True, return_index=True)\n    ... # XXX: judging by the output, the ignoremask flag has no effect\n    ")
    
    # Assigning a Call to a Name (line 794):
    
    # Assigning a Call to a Name (line 794):
    
    # Call to ravel(...): (line 794)
    # Processing the call keyword arguments (line 794)
    kwargs_124221 = {}
    
    # Call to asanyarray(...): (line 794)
    # Processing the call arguments (line 794)
    # Getting the type of 'a' (line 794)
    a_124217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 794, 22), 'a', False)
    # Processing the call keyword arguments (line 794)
    kwargs_124218 = {}
    # Getting the type of 'np' (line 794)
    np_124215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 794, 8), 'np', False)
    # Obtaining the member 'asanyarray' of a type (line 794)
    asanyarray_124216 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 794, 8), np_124215, 'asanyarray')
    # Calling asanyarray(args, kwargs) (line 794)
    asanyarray_call_result_124219 = invoke(stypy.reporting.localization.Localization(__file__, 794, 8), asanyarray_124216, *[a_124217], **kwargs_124218)
    
    # Obtaining the member 'ravel' of a type (line 794)
    ravel_124220 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 794, 8), asanyarray_call_result_124219, 'ravel')
    # Calling ravel(args, kwargs) (line 794)
    ravel_call_result_124222 = invoke(stypy.reporting.localization.Localization(__file__, 794, 8), ravel_124220, *[], **kwargs_124221)
    
    # Assigning a type to the variable 'a' (line 794)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 794, 4), 'a', ravel_call_result_124222)
    
    # Assigning a Call to a Name (line 796):
    
    # Assigning a Call to a Name (line 796):
    
    # Call to get_fieldstructure(...): (line 796)
    # Processing the call arguments (line 796)
    # Getting the type of 'a' (line 796)
    a_124224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 796, 32), 'a', False)
    # Obtaining the member 'dtype' of a type (line 796)
    dtype_124225 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 796, 32), a_124224, 'dtype')
    # Processing the call keyword arguments (line 796)
    kwargs_124226 = {}
    # Getting the type of 'get_fieldstructure' (line 796)
    get_fieldstructure_124223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 796, 13), 'get_fieldstructure', False)
    # Calling get_fieldstructure(args, kwargs) (line 796)
    get_fieldstructure_call_result_124227 = invoke(stypy.reporting.localization.Localization(__file__, 796, 13), get_fieldstructure_124223, *[dtype_124225], **kwargs_124226)
    
    # Assigning a type to the variable 'fields' (line 796)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 796, 4), 'fields', get_fieldstructure_call_result_124227)
    
    # Assigning a Name to a Name (line 798):
    
    # Assigning a Name to a Name (line 798):
    # Getting the type of 'a' (line 798)
    a_124228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 798, 11), 'a')
    # Assigning a type to the variable 'base' (line 798)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 798, 4), 'base', a_124228)
    
    # Getting the type of 'key' (line 799)
    key_124229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 799, 7), 'key')
    # Testing the type of an if condition (line 799)
    if_condition_124230 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 799, 4), key_124229)
    # Assigning a type to the variable 'if_condition_124230' (line 799)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 799, 4), 'if_condition_124230', if_condition_124230)
    # SSA begins for if statement (line 799)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'key' (line 800)
    key_124231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 800, 24), 'key')
    # Getting the type of 'fields' (line 800)
    fields_124232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 800, 17), 'fields')
    # Obtaining the member '__getitem__' of a type (line 800)
    getitem___124233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 800, 17), fields_124232, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 800)
    subscript_call_result_124234 = invoke(stypy.reporting.localization.Localization(__file__, 800, 17), getitem___124233, key_124231)
    
    # Testing the type of a for loop iterable (line 800)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 800, 8), subscript_call_result_124234)
    # Getting the type of the for loop variable (line 800)
    for_loop_var_124235 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 800, 8), subscript_call_result_124234)
    # Assigning a type to the variable 'f' (line 800)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 800, 8), 'f', for_loop_var_124235)
    # SSA begins for a for statement (line 800)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Subscript to a Name (line 801):
    
    # Assigning a Subscript to a Name (line 801):
    
    # Obtaining the type of the subscript
    # Getting the type of 'f' (line 801)
    f_124236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 801, 24), 'f')
    # Getting the type of 'base' (line 801)
    base_124237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 801, 19), 'base')
    # Obtaining the member '__getitem__' of a type (line 801)
    getitem___124238 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 801, 19), base_124237, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 801)
    subscript_call_result_124239 = invoke(stypy.reporting.localization.Localization(__file__, 801, 19), getitem___124238, f_124236)
    
    # Assigning a type to the variable 'base' (line 801)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 801, 12), 'base', subscript_call_result_124239)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Subscript to a Name (line 802):
    
    # Assigning a Subscript to a Name (line 802):
    
    # Obtaining the type of the subscript
    # Getting the type of 'key' (line 802)
    key_124240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 802, 20), 'key')
    # Getting the type of 'base' (line 802)
    base_124241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 802, 15), 'base')
    # Obtaining the member '__getitem__' of a type (line 802)
    getitem___124242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 802, 15), base_124241, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 802)
    subscript_call_result_124243 = invoke(stypy.reporting.localization.Localization(__file__, 802, 15), getitem___124242, key_124240)
    
    # Assigning a type to the variable 'base' (line 802)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 802, 8), 'base', subscript_call_result_124243)
    # SSA join for if statement (line 799)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 804):
    
    # Assigning a Call to a Name (line 804):
    
    # Call to argsort(...): (line 804)
    # Processing the call keyword arguments (line 804)
    kwargs_124246 = {}
    # Getting the type of 'base' (line 804)
    base_124244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 804, 14), 'base', False)
    # Obtaining the member 'argsort' of a type (line 804)
    argsort_124245 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 804, 14), base_124244, 'argsort')
    # Calling argsort(args, kwargs) (line 804)
    argsort_call_result_124247 = invoke(stypy.reporting.localization.Localization(__file__, 804, 14), argsort_124245, *[], **kwargs_124246)
    
    # Assigning a type to the variable 'sortidx' (line 804)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 804, 4), 'sortidx', argsort_call_result_124247)
    
    # Assigning a Subscript to a Name (line 805):
    
    # Assigning a Subscript to a Name (line 805):
    
    # Obtaining the type of the subscript
    # Getting the type of 'sortidx' (line 805)
    sortidx_124248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 805, 22), 'sortidx')
    # Getting the type of 'base' (line 805)
    base_124249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 805, 17), 'base')
    # Obtaining the member '__getitem__' of a type (line 805)
    getitem___124250 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 805, 17), base_124249, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 805)
    subscript_call_result_124251 = invoke(stypy.reporting.localization.Localization(__file__, 805, 17), getitem___124250, sortidx_124248)
    
    # Assigning a type to the variable 'sortedbase' (line 805)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 805, 4), 'sortedbase', subscript_call_result_124251)
    
    # Assigning a Call to a Name (line 806):
    
    # Assigning a Call to a Name (line 806):
    
    # Call to filled(...): (line 806)
    # Processing the call keyword arguments (line 806)
    kwargs_124254 = {}
    # Getting the type of 'sortedbase' (line 806)
    sortedbase_124252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 806, 17), 'sortedbase', False)
    # Obtaining the member 'filled' of a type (line 806)
    filled_124253 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 806, 17), sortedbase_124252, 'filled')
    # Calling filled(args, kwargs) (line 806)
    filled_call_result_124255 = invoke(stypy.reporting.localization.Localization(__file__, 806, 17), filled_124253, *[], **kwargs_124254)
    
    # Assigning a type to the variable 'sorteddata' (line 806)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 806, 4), 'sorteddata', filled_call_result_124255)
    
    # Assigning a Compare to a Name (line 808):
    
    # Assigning a Compare to a Name (line 808):
    
    
    # Obtaining the type of the subscript
    int_124256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 808, 24), 'int')
    slice_124257 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 808, 12), None, int_124256, None)
    # Getting the type of 'sorteddata' (line 808)
    sorteddata_124258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 808, 12), 'sorteddata')
    # Obtaining the member '__getitem__' of a type (line 808)
    getitem___124259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 808, 12), sorteddata_124258, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 808)
    subscript_call_result_124260 = invoke(stypy.reporting.localization.Localization(__file__, 808, 12), getitem___124259, slice_124257)
    
    
    # Obtaining the type of the subscript
    int_124261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 808, 42), 'int')
    slice_124262 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 808, 31), int_124261, None, None)
    # Getting the type of 'sorteddata' (line 808)
    sorteddata_124263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 808, 31), 'sorteddata')
    # Obtaining the member '__getitem__' of a type (line 808)
    getitem___124264 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 808, 31), sorteddata_124263, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 808)
    subscript_call_result_124265 = invoke(stypy.reporting.localization.Localization(__file__, 808, 31), getitem___124264, slice_124262)
    
    # Applying the binary operator '==' (line 808)
    result_eq_124266 = python_operator(stypy.reporting.localization.Localization(__file__, 808, 12), '==', subscript_call_result_124260, subscript_call_result_124265)
    
    # Assigning a type to the variable 'flag' (line 808)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 808, 4), 'flag', result_eq_124266)
    
    # Getting the type of 'ignoremask' (line 810)
    ignoremask_124267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 810, 7), 'ignoremask')
    # Testing the type of an if condition (line 810)
    if_condition_124268 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 810, 4), ignoremask_124267)
    # Assigning a type to the variable 'if_condition_124268' (line 810)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 810, 4), 'if_condition_124268', if_condition_124268)
    # SSA begins for if statement (line 810)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Attribute to a Name (line 811):
    
    # Assigning a Attribute to a Name (line 811):
    # Getting the type of 'sortedbase' (line 811)
    sortedbase_124269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 811, 21), 'sortedbase')
    # Obtaining the member 'recordmask' of a type (line 811)
    recordmask_124270 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 811, 21), sortedbase_124269, 'recordmask')
    # Assigning a type to the variable 'sortedmask' (line 811)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 811, 8), 'sortedmask', recordmask_124270)
    
    # Assigning a Name to a Subscript (line 812):
    
    # Assigning a Name to a Subscript (line 812):
    # Getting the type of 'False' (line 812)
    False_124271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 812, 31), 'False')
    # Getting the type of 'flag' (line 812)
    flag_124272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 812, 8), 'flag')
    
    # Obtaining the type of the subscript
    int_124273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 812, 24), 'int')
    slice_124274 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 812, 13), int_124273, None, None)
    # Getting the type of 'sortedmask' (line 812)
    sortedmask_124275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 812, 13), 'sortedmask')
    # Obtaining the member '__getitem__' of a type (line 812)
    getitem___124276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 812, 13), sortedmask_124275, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 812)
    subscript_call_result_124277 = invoke(stypy.reporting.localization.Localization(__file__, 812, 13), getitem___124276, slice_124274)
    
    # Storing an element on a container (line 812)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 812, 8), flag_124272, (subscript_call_result_124277, False_124271))
    # SSA join for if statement (line 810)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 813):
    
    # Assigning a Call to a Name (line 813):
    
    # Call to concatenate(...): (line 813)
    # Processing the call arguments (line 813)
    
    # Obtaining an instance of the builtin type 'tuple' (line 813)
    tuple_124280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 813, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 813)
    # Adding element type (line 813)
    
    # Obtaining an instance of the builtin type 'list' (line 813)
    list_124281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 813, 27), 'list')
    # Adding type elements to the builtin type 'list' instance (line 813)
    # Adding element type (line 813)
    # Getting the type of 'False' (line 813)
    False_124282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 813, 28), 'False', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 813, 27), list_124281, False_124282)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 813, 27), tuple_124280, list_124281)
    # Adding element type (line 813)
    # Getting the type of 'flag' (line 813)
    flag_124283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 813, 36), 'flag', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 813, 27), tuple_124280, flag_124283)
    
    # Processing the call keyword arguments (line 813)
    kwargs_124284 = {}
    # Getting the type of 'np' (line 813)
    np_124278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 813, 11), 'np', False)
    # Obtaining the member 'concatenate' of a type (line 813)
    concatenate_124279 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 813, 11), np_124278, 'concatenate')
    # Calling concatenate(args, kwargs) (line 813)
    concatenate_call_result_124285 = invoke(stypy.reporting.localization.Localization(__file__, 813, 11), concatenate_124279, *[tuple_124280], **kwargs_124284)
    
    # Assigning a type to the variable 'flag' (line 813)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 813, 4), 'flag', concatenate_call_result_124285)
    
    # Assigning a BinOp to a Subscript (line 815):
    
    # Assigning a BinOp to a Subscript (line 815):
    
    # Obtaining the type of the subscript
    int_124286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 815, 22), 'int')
    slice_124287 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 815, 16), None, int_124286, None)
    # Getting the type of 'flag' (line 815)
    flag_124288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 815, 16), 'flag')
    # Obtaining the member '__getitem__' of a type (line 815)
    getitem___124289 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 815, 16), flag_124288, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 815)
    subscript_call_result_124290 = invoke(stypy.reporting.localization.Localization(__file__, 815, 16), getitem___124289, slice_124287)
    
    
    # Obtaining the type of the subscript
    int_124291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 815, 33), 'int')
    slice_124292 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 815, 28), int_124291, None, None)
    # Getting the type of 'flag' (line 815)
    flag_124293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 815, 28), 'flag')
    # Obtaining the member '__getitem__' of a type (line 815)
    getitem___124294 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 815, 28), flag_124293, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 815)
    subscript_call_result_124295 = invoke(stypy.reporting.localization.Localization(__file__, 815, 28), getitem___124294, slice_124292)
    
    # Applying the binary operator '+' (line 815)
    result_add_124296 = python_operator(stypy.reporting.localization.Localization(__file__, 815, 16), '+', subscript_call_result_124290, subscript_call_result_124295)
    
    # Getting the type of 'flag' (line 815)
    flag_124297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 815, 4), 'flag')
    int_124298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 815, 10), 'int')
    slice_124299 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 815, 4), None, int_124298, None)
    # Storing an element on a container (line 815)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 815, 4), flag_124297, (slice_124299, result_add_124296))
    
    # Assigning a Subscript to a Name (line 816):
    
    # Assigning a Subscript to a Name (line 816):
    
    # Obtaining the type of the subscript
    # Getting the type of 'flag' (line 816)
    flag_124300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 816, 28), 'flag')
    
    # Obtaining the type of the subscript
    # Getting the type of 'sortidx' (line 816)
    sortidx_124301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 816, 19), 'sortidx')
    # Getting the type of 'a' (line 816)
    a_124302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 816, 17), 'a')
    # Obtaining the member '__getitem__' of a type (line 816)
    getitem___124303 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 816, 17), a_124302, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 816)
    subscript_call_result_124304 = invoke(stypy.reporting.localization.Localization(__file__, 816, 17), getitem___124303, sortidx_124301)
    
    # Obtaining the member '__getitem__' of a type (line 816)
    getitem___124305 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 816, 17), subscript_call_result_124304, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 816)
    subscript_call_result_124306 = invoke(stypy.reporting.localization.Localization(__file__, 816, 17), getitem___124305, flag_124300)
    
    # Assigning a type to the variable 'duplicates' (line 816)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 816, 4), 'duplicates', subscript_call_result_124306)
    
    # Getting the type of 'return_index' (line 817)
    return_index_124307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 817, 7), 'return_index')
    # Testing the type of an if condition (line 817)
    if_condition_124308 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 817, 4), return_index_124307)
    # Assigning a type to the variable 'if_condition_124308' (line 817)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 817, 4), 'if_condition_124308', if_condition_124308)
    # SSA begins for if statement (line 817)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'tuple' (line 818)
    tuple_124309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 818, 16), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 818)
    # Adding element type (line 818)
    # Getting the type of 'duplicates' (line 818)
    duplicates_124310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 818, 16), 'duplicates')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 818, 16), tuple_124309, duplicates_124310)
    # Adding element type (line 818)
    
    # Obtaining the type of the subscript
    # Getting the type of 'flag' (line 818)
    flag_124311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 818, 36), 'flag')
    # Getting the type of 'sortidx' (line 818)
    sortidx_124312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 818, 28), 'sortidx')
    # Obtaining the member '__getitem__' of a type (line 818)
    getitem___124313 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 818, 28), sortidx_124312, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 818)
    subscript_call_result_124314 = invoke(stypy.reporting.localization.Localization(__file__, 818, 28), getitem___124313, flag_124311)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 818, 16), tuple_124309, subscript_call_result_124314)
    
    # Assigning a type to the variable 'stypy_return_type' (line 818)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 818, 8), 'stypy_return_type', tuple_124309)
    # SSA branch for the else part of an if statement (line 817)
    module_type_store.open_ssa_branch('else')
    # Getting the type of 'duplicates' (line 820)
    duplicates_124315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 820, 15), 'duplicates')
    # Assigning a type to the variable 'stypy_return_type' (line 820)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 820, 8), 'stypy_return_type', duplicates_124315)
    # SSA join for if statement (line 817)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'find_duplicates(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'find_duplicates' in the type store
    # Getting the type of 'stypy_return_type' (line 769)
    stypy_return_type_124316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 769, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_124316)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'find_duplicates'
    return stypy_return_type_124316

# Assigning a type to the variable 'find_duplicates' (line 769)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 769, 0), 'find_duplicates', find_duplicates)

@norecursion
def join_by(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_124317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 823, 34), 'str', 'inner')
    str_124318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 823, 53), 'str', '1')
    str_124319 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 823, 68), 'str', '2')
    # Getting the type of 'None' (line 824)
    None_124320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 824, 25), 'None')
    # Getting the type of 'True' (line 824)
    True_124321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 824, 39), 'True')
    # Getting the type of 'False' (line 824)
    False_124322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 824, 56), 'False')
    defaults = [str_124317, str_124318, str_124319, None_124320, True_124321, False_124322]
    # Create a new context for function 'join_by'
    module_type_store = module_type_store.open_function_context('join_by', 823, 0, False)
    
    # Passed parameters checking function
    join_by.stypy_localization = localization
    join_by.stypy_type_of_self = None
    join_by.stypy_type_store = module_type_store
    join_by.stypy_function_name = 'join_by'
    join_by.stypy_param_names_list = ['key', 'r1', 'r2', 'jointype', 'r1postfix', 'r2postfix', 'defaults', 'usemask', 'asrecarray']
    join_by.stypy_varargs_param_name = None
    join_by.stypy_kwargs_param_name = None
    join_by.stypy_call_defaults = defaults
    join_by.stypy_call_varargs = varargs
    join_by.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'join_by', ['key', 'r1', 'r2', 'jointype', 'r1postfix', 'r2postfix', 'defaults', 'usemask', 'asrecarray'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'join_by', localization, ['key', 'r1', 'r2', 'jointype', 'r1postfix', 'r2postfix', 'defaults', 'usemask', 'asrecarray'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'join_by(...)' code ##################

    str_124323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 872, (-1)), 'str', "\n    Join arrays `r1` and `r2` on key `key`.\n\n    The key should be either a string or a sequence of string corresponding\n    to the fields used to join the array.  An exception is raised if the\n    `key` field cannot be found in the two input arrays.  Neither `r1` nor\n    `r2` should have any duplicates along `key`: the presence of duplicates\n    will make the output quite unreliable. Note that duplicates are not\n    looked for by the algorithm.\n\n    Parameters\n    ----------\n    key : {string, sequence}\n        A string or a sequence of strings corresponding to the fields used\n        for comparison.\n    r1, r2 : arrays\n        Structured arrays.\n    jointype : {'inner', 'outer', 'leftouter'}, optional\n        If 'inner', returns the elements common to both r1 and r2.\n        If 'outer', returns the common elements as well as the elements of\n        r1 not in r2 and the elements of not in r2.\n        If 'leftouter', returns the common elements and the elements of r1\n        not in r2.\n    r1postfix : string, optional\n        String appended to the names of the fields of r1 that are present\n        in r2 but absent of the key.\n    r2postfix : string, optional\n        String appended to the names of the fields of r2 that are present\n        in r1 but absent of the key.\n    defaults : {dictionary}, optional\n        Dictionary mapping field names to the corresponding default values.\n    usemask : {True, False}, optional\n        Whether to return a MaskedArray (or MaskedRecords is\n        `asrecarray==True`) or a ndarray.\n    asrecarray : {False, True}, optional\n        Whether to return a recarray (or MaskedRecords if `usemask==True`)\n        or just a flexible-type ndarray.\n\n    Notes\n    -----\n    * The output is sorted along the key.\n    * A temporary array is formed by dropping the fields not in the key for\n      the two arrays and concatenating the result. This array is then\n      sorted, and the common entries selected. The output is constructed by\n      filling the fields with the selected entries. Matching is not\n      preserved if there are some duplicates...\n\n    ")
    
    
    # Getting the type of 'jointype' (line 874)
    jointype_124324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 874, 7), 'jointype')
    
    # Obtaining an instance of the builtin type 'tuple' (line 874)
    tuple_124325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 874, 24), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 874)
    # Adding element type (line 874)
    str_124326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 874, 24), 'str', 'inner')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 874, 24), tuple_124325, str_124326)
    # Adding element type (line 874)
    str_124327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 874, 33), 'str', 'outer')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 874, 24), tuple_124325, str_124327)
    # Adding element type (line 874)
    str_124328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 874, 42), 'str', 'leftouter')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 874, 24), tuple_124325, str_124328)
    
    # Applying the binary operator 'notin' (line 874)
    result_contains_124329 = python_operator(stypy.reporting.localization.Localization(__file__, 874, 7), 'notin', jointype_124324, tuple_124325)
    
    # Testing the type of an if condition (line 874)
    if_condition_124330 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 874, 4), result_contains_124329)
    # Assigning a type to the variable 'if_condition_124330' (line 874)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 874, 4), 'if_condition_124330', if_condition_124330)
    # SSA begins for if statement (line 874)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 875)
    # Processing the call arguments (line 875)
    str_124332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 876, 16), 'str', "The 'jointype' argument should be in 'inner', 'outer' or 'leftouter' (got '%s' instead)")
    # Getting the type of 'jointype' (line 877)
    jointype_124333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 877, 62), 'jointype', False)
    # Applying the binary operator '%' (line 876)
    result_mod_124334 = python_operator(stypy.reporting.localization.Localization(__file__, 876, 16), '%', str_124332, jointype_124333)
    
    # Processing the call keyword arguments (line 875)
    kwargs_124335 = {}
    # Getting the type of 'ValueError' (line 875)
    ValueError_124331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 875, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 875)
    ValueError_call_result_124336 = invoke(stypy.reporting.localization.Localization(__file__, 875, 14), ValueError_124331, *[result_mod_124334], **kwargs_124335)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 875, 8), ValueError_call_result_124336, 'raise parameter', BaseException)
    # SSA join for if statement (line 874)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 880)
    # Getting the type of 'basestring' (line 880)
    basestring_124337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 880, 23), 'basestring')
    # Getting the type of 'key' (line 880)
    key_124338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 880, 18), 'key')
    
    (may_be_124339, more_types_in_union_124340) = may_be_subtype(basestring_124337, key_124338)

    if may_be_124339:

        if more_types_in_union_124340:
            # Runtime conditional SSA (line 880)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'key' (line 880)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 880, 4), 'key', remove_not_subtype_from_union(key_124338, basestring))
        
        # Assigning a Tuple to a Name (line 881):
        
        # Assigning a Tuple to a Name (line 881):
        
        # Obtaining an instance of the builtin type 'tuple' (line 881)
        tuple_124341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 881, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 881)
        # Adding element type (line 881)
        # Getting the type of 'key' (line 881)
        key_124342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 881, 15), 'key')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 881, 15), tuple_124341, key_124342)
        
        # Assigning a type to the variable 'key' (line 881)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 881, 8), 'key', tuple_124341)

        if more_types_in_union_124340:
            # SSA join for if statement (line 880)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Getting the type of 'key' (line 884)
    key_124343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 884, 16), 'key')
    # Testing the type of a for loop iterable (line 884)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 884, 4), key_124343)
    # Getting the type of the for loop variable (line 884)
    for_loop_var_124344 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 884, 4), key_124343)
    # Assigning a type to the variable 'name' (line 884)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 884, 4), 'name', for_loop_var_124344)
    # SSA begins for a for statement (line 884)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Getting the type of 'name' (line 885)
    name_124345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 885, 11), 'name')
    # Getting the type of 'r1' (line 885)
    r1_124346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 885, 23), 'r1')
    # Obtaining the member 'dtype' of a type (line 885)
    dtype_124347 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 885, 23), r1_124346, 'dtype')
    # Obtaining the member 'names' of a type (line 885)
    names_124348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 885, 23), dtype_124347, 'names')
    # Applying the binary operator 'notin' (line 885)
    result_contains_124349 = python_operator(stypy.reporting.localization.Localization(__file__, 885, 11), 'notin', name_124345, names_124348)
    
    # Testing the type of an if condition (line 885)
    if_condition_124350 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 885, 8), result_contains_124349)
    # Assigning a type to the variable 'if_condition_124350' (line 885)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 885, 8), 'if_condition_124350', if_condition_124350)
    # SSA begins for if statement (line 885)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 886)
    # Processing the call arguments (line 886)
    str_124352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 886, 29), 'str', 'r1 does not have key field %s')
    # Getting the type of 'name' (line 886)
    name_124353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 886, 63), 'name', False)
    # Applying the binary operator '%' (line 886)
    result_mod_124354 = python_operator(stypy.reporting.localization.Localization(__file__, 886, 29), '%', str_124352, name_124353)
    
    # Processing the call keyword arguments (line 886)
    kwargs_124355 = {}
    # Getting the type of 'ValueError' (line 886)
    ValueError_124351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 886, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 886)
    ValueError_call_result_124356 = invoke(stypy.reporting.localization.Localization(__file__, 886, 18), ValueError_124351, *[result_mod_124354], **kwargs_124355)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 886, 12), ValueError_call_result_124356, 'raise parameter', BaseException)
    # SSA join for if statement (line 885)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'name' (line 887)
    name_124357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 887, 11), 'name')
    # Getting the type of 'r2' (line 887)
    r2_124358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 887, 23), 'r2')
    # Obtaining the member 'dtype' of a type (line 887)
    dtype_124359 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 887, 23), r2_124358, 'dtype')
    # Obtaining the member 'names' of a type (line 887)
    names_124360 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 887, 23), dtype_124359, 'names')
    # Applying the binary operator 'notin' (line 887)
    result_contains_124361 = python_operator(stypy.reporting.localization.Localization(__file__, 887, 11), 'notin', name_124357, names_124360)
    
    # Testing the type of an if condition (line 887)
    if_condition_124362 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 887, 8), result_contains_124361)
    # Assigning a type to the variable 'if_condition_124362' (line 887)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 887, 8), 'if_condition_124362', if_condition_124362)
    # SSA begins for if statement (line 887)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 888)
    # Processing the call arguments (line 888)
    str_124364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 888, 29), 'str', 'r2 does not have key field %s')
    # Getting the type of 'name' (line 888)
    name_124365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 888, 63), 'name', False)
    # Applying the binary operator '%' (line 888)
    result_mod_124366 = python_operator(stypy.reporting.localization.Localization(__file__, 888, 29), '%', str_124364, name_124365)
    
    # Processing the call keyword arguments (line 888)
    kwargs_124367 = {}
    # Getting the type of 'ValueError' (line 888)
    ValueError_124363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 888, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 888)
    ValueError_call_result_124368 = invoke(stypy.reporting.localization.Localization(__file__, 888, 18), ValueError_124363, *[result_mod_124366], **kwargs_124367)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 888, 12), ValueError_call_result_124368, 'raise parameter', BaseException)
    # SSA join for if statement (line 887)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 891):
    
    # Assigning a Call to a Name (line 891):
    
    # Call to ravel(...): (line 891)
    # Processing the call keyword arguments (line 891)
    kwargs_124371 = {}
    # Getting the type of 'r1' (line 891)
    r1_124369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 891, 9), 'r1', False)
    # Obtaining the member 'ravel' of a type (line 891)
    ravel_124370 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 891, 9), r1_124369, 'ravel')
    # Calling ravel(args, kwargs) (line 891)
    ravel_call_result_124372 = invoke(stypy.reporting.localization.Localization(__file__, 891, 9), ravel_124370, *[], **kwargs_124371)
    
    # Assigning a type to the variable 'r1' (line 891)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 891, 4), 'r1', ravel_call_result_124372)
    
    # Assigning a Call to a Name (line 892):
    
    # Assigning a Call to a Name (line 892):
    
    # Call to ravel(...): (line 892)
    # Processing the call keyword arguments (line 892)
    kwargs_124375 = {}
    # Getting the type of 'r2' (line 892)
    r2_124373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 892, 9), 'r2', False)
    # Obtaining the member 'ravel' of a type (line 892)
    ravel_124374 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 892, 9), r2_124373, 'ravel')
    # Calling ravel(args, kwargs) (line 892)
    ravel_call_result_124376 = invoke(stypy.reporting.localization.Localization(__file__, 892, 9), ravel_124374, *[], **kwargs_124375)
    
    # Assigning a type to the variable 'r2' (line 892)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 892, 4), 'r2', ravel_call_result_124376)
    
    # Assigning a Call to a Name (line 895):
    
    # Assigning a Call to a Name (line 895):
    
    # Call to len(...): (line 895)
    # Processing the call arguments (line 895)
    # Getting the type of 'r1' (line 895)
    r1_124378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 895, 14), 'r1', False)
    # Processing the call keyword arguments (line 895)
    kwargs_124379 = {}
    # Getting the type of 'len' (line 895)
    len_124377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 895, 10), 'len', False)
    # Calling len(args, kwargs) (line 895)
    len_call_result_124380 = invoke(stypy.reporting.localization.Localization(__file__, 895, 10), len_124377, *[r1_124378], **kwargs_124379)
    
    # Assigning a type to the variable 'nb1' (line 895)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 895, 4), 'nb1', len_call_result_124380)
    
    # Assigning a Tuple to a Tuple (line 896):
    
    # Assigning a Attribute to a Name (line 896):
    # Getting the type of 'r1' (line 896)
    r1_124381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 896, 26), 'r1')
    # Obtaining the member 'dtype' of a type (line 896)
    dtype_124382 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 896, 26), r1_124381, 'dtype')
    # Obtaining the member 'names' of a type (line 896)
    names_124383 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 896, 26), dtype_124382, 'names')
    # Assigning a type to the variable 'tuple_assignment_122695' (line 896)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 896, 4), 'tuple_assignment_122695', names_124383)
    
    # Assigning a Attribute to a Name (line 896):
    # Getting the type of 'r2' (line 896)
    r2_124384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 896, 42), 'r2')
    # Obtaining the member 'dtype' of a type (line 896)
    dtype_124385 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 896, 42), r2_124384, 'dtype')
    # Obtaining the member 'names' of a type (line 896)
    names_124386 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 896, 42), dtype_124385, 'names')
    # Assigning a type to the variable 'tuple_assignment_122696' (line 896)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 896, 4), 'tuple_assignment_122696', names_124386)
    
    # Assigning a Name to a Name (line 896):
    # Getting the type of 'tuple_assignment_122695' (line 896)
    tuple_assignment_122695_124387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 896, 4), 'tuple_assignment_122695')
    # Assigning a type to the variable 'r1names' (line 896)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 896, 5), 'r1names', tuple_assignment_122695_124387)
    
    # Assigning a Name to a Name (line 896):
    # Getting the type of 'tuple_assignment_122696' (line 896)
    tuple_assignment_122696_124388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 896, 4), 'tuple_assignment_122696')
    # Assigning a type to the variable 'r2names' (line 896)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 896, 14), 'r2names', tuple_assignment_122696_124388)
    
    
    # Evaluating a boolean operation
    
    # Call to difference(...): (line 899)
    # Processing the call arguments (line 899)
    # Getting the type of 'key' (line 899)
    key_124402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 899, 64), 'key', False)
    # Processing the call keyword arguments (line 899)
    kwargs_124403 = {}
    
    # Call to intersection(...): (line 899)
    # Processing the call arguments (line 899)
    
    # Call to set(...): (line 899)
    # Processing the call arguments (line 899)
    # Getting the type of 'r1names' (line 899)
    r1names_124392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 899, 29), 'r1names', False)
    # Processing the call keyword arguments (line 899)
    kwargs_124393 = {}
    # Getting the type of 'set' (line 899)
    set_124391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 899, 25), 'set', False)
    # Calling set(args, kwargs) (line 899)
    set_call_result_124394 = invoke(stypy.reporting.localization.Localization(__file__, 899, 25), set_124391, *[r1names_124392], **kwargs_124393)
    
    
    # Call to set(...): (line 899)
    # Processing the call arguments (line 899)
    # Getting the type of 'r2names' (line 899)
    r2names_124396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 899, 43), 'r2names', False)
    # Processing the call keyword arguments (line 899)
    kwargs_124397 = {}
    # Getting the type of 'set' (line 899)
    set_124395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 899, 39), 'set', False)
    # Calling set(args, kwargs) (line 899)
    set_call_result_124398 = invoke(stypy.reporting.localization.Localization(__file__, 899, 39), set_124395, *[r2names_124396], **kwargs_124397)
    
    # Processing the call keyword arguments (line 899)
    kwargs_124399 = {}
    # Getting the type of 'set' (line 899)
    set_124389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 899, 8), 'set', False)
    # Obtaining the member 'intersection' of a type (line 899)
    intersection_124390 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 899, 8), set_124389, 'intersection')
    # Calling intersection(args, kwargs) (line 899)
    intersection_call_result_124400 = invoke(stypy.reporting.localization.Localization(__file__, 899, 8), intersection_124390, *[set_call_result_124394, set_call_result_124398], **kwargs_124399)
    
    # Obtaining the member 'difference' of a type (line 899)
    difference_124401 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 899, 8), intersection_call_result_124400, 'difference')
    # Calling difference(args, kwargs) (line 899)
    difference_call_result_124404 = invoke(stypy.reporting.localization.Localization(__file__, 899, 8), difference_124401, *[key_124402], **kwargs_124403)
    
    
    
    # Evaluating a boolean operation
    # Getting the type of 'r1postfix' (line 900)
    r1postfix_124405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 900, 17), 'r1postfix')
    # Getting the type of 'r2postfix' (line 900)
    r2postfix_124406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 900, 30), 'r2postfix')
    # Applying the binary operator 'or' (line 900)
    result_or_keyword_124407 = python_operator(stypy.reporting.localization.Localization(__file__, 900, 17), 'or', r1postfix_124405, r2postfix_124406)
    
    # Applying the 'not' unary operator (line 900)
    result_not__124408 = python_operator(stypy.reporting.localization.Localization(__file__, 900, 12), 'not', result_or_keyword_124407)
    
    # Applying the binary operator 'and' (line 899)
    result_and_keyword_124409 = python_operator(stypy.reporting.localization.Localization(__file__, 899, 8), 'and', difference_call_result_124404, result_not__124408)
    
    # Testing the type of an if condition (line 899)
    if_condition_124410 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 899, 4), result_and_keyword_124409)
    # Assigning a type to the variable 'if_condition_124410' (line 899)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 899, 4), 'if_condition_124410', if_condition_124410)
    # SSA begins for if statement (line 899)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 901):
    
    # Assigning a Str to a Name (line 901):
    str_124411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 901, 14), 'str', 'r1 and r2 contain common names, r1postfix and r2postfix ')
    # Assigning a type to the variable 'msg' (line 901)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 901, 8), 'msg', str_124411)
    
    # Getting the type of 'msg' (line 902)
    msg_124412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 902, 8), 'msg')
    str_124413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 902, 15), 'str', "can't be empty")
    # Applying the binary operator '+=' (line 902)
    result_iadd_124414 = python_operator(stypy.reporting.localization.Localization(__file__, 902, 8), '+=', msg_124412, str_124413)
    # Assigning a type to the variable 'msg' (line 902)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 902, 8), 'msg', result_iadd_124414)
    
    
    # Call to ValueError(...): (line 903)
    # Processing the call arguments (line 903)
    # Getting the type of 'msg' (line 903)
    msg_124416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 903, 25), 'msg', False)
    # Processing the call keyword arguments (line 903)
    kwargs_124417 = {}
    # Getting the type of 'ValueError' (line 903)
    ValueError_124415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 903, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 903)
    ValueError_call_result_124418 = invoke(stypy.reporting.localization.Localization(__file__, 903, 14), ValueError_124415, *[msg_124416], **kwargs_124417)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 903, 8), ValueError_call_result_124418, 'raise parameter', BaseException)
    # SSA join for if statement (line 899)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 906):
    
    # Assigning a Call to a Name (line 906):
    
    # Call to drop_fields(...): (line 906)
    # Processing the call arguments (line 906)
    # Getting the type of 'r1' (line 906)
    r1_124420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 906, 22), 'r1', False)
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'r1names' (line 906)
    r1names_124425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 906, 38), 'r1names', False)
    comprehension_124426 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 906, 27), r1names_124425)
    # Assigning a type to the variable 'n' (line 906)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 906, 27), 'n', comprehension_124426)
    
    # Getting the type of 'n' (line 906)
    n_124422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 906, 49), 'n', False)
    # Getting the type of 'key' (line 906)
    key_124423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 906, 58), 'key', False)
    # Applying the binary operator 'notin' (line 906)
    result_contains_124424 = python_operator(stypy.reporting.localization.Localization(__file__, 906, 49), 'notin', n_124422, key_124423)
    
    # Getting the type of 'n' (line 906)
    n_124421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 906, 27), 'n', False)
    list_124427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 906, 27), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 906, 27), list_124427, n_124421)
    # Processing the call keyword arguments (line 906)
    kwargs_124428 = {}
    # Getting the type of 'drop_fields' (line 906)
    drop_fields_124419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 906, 10), 'drop_fields', False)
    # Calling drop_fields(args, kwargs) (line 906)
    drop_fields_call_result_124429 = invoke(stypy.reporting.localization.Localization(__file__, 906, 10), drop_fields_124419, *[r1_124420, list_124427], **kwargs_124428)
    
    # Assigning a type to the variable 'r1k' (line 906)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 906, 4), 'r1k', drop_fields_call_result_124429)
    
    # Assigning a Call to a Name (line 907):
    
    # Assigning a Call to a Name (line 907):
    
    # Call to drop_fields(...): (line 907)
    # Processing the call arguments (line 907)
    # Getting the type of 'r2' (line 907)
    r2_124431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 907, 22), 'r2', False)
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'r2names' (line 907)
    r2names_124436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 907, 38), 'r2names', False)
    comprehension_124437 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 907, 27), r2names_124436)
    # Assigning a type to the variable 'n' (line 907)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 907, 27), 'n', comprehension_124437)
    
    # Getting the type of 'n' (line 907)
    n_124433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 907, 49), 'n', False)
    # Getting the type of 'key' (line 907)
    key_124434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 907, 58), 'key', False)
    # Applying the binary operator 'notin' (line 907)
    result_contains_124435 = python_operator(stypy.reporting.localization.Localization(__file__, 907, 49), 'notin', n_124433, key_124434)
    
    # Getting the type of 'n' (line 907)
    n_124432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 907, 27), 'n', False)
    list_124438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 907, 27), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 907, 27), list_124438, n_124432)
    # Processing the call keyword arguments (line 907)
    kwargs_124439 = {}
    # Getting the type of 'drop_fields' (line 907)
    drop_fields_124430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 907, 10), 'drop_fields', False)
    # Calling drop_fields(args, kwargs) (line 907)
    drop_fields_call_result_124440 = invoke(stypy.reporting.localization.Localization(__file__, 907, 10), drop_fields_124430, *[r2_124431, list_124438], **kwargs_124439)
    
    # Assigning a type to the variable 'r2k' (line 907)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 907, 4), 'r2k', drop_fields_call_result_124440)
    
    # Assigning a Call to a Name (line 910):
    
    # Assigning a Call to a Name (line 910):
    
    # Call to concatenate(...): (line 910)
    # Processing the call arguments (line 910)
    
    # Obtaining an instance of the builtin type 'tuple' (line 910)
    tuple_124443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 910, 26), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 910)
    # Adding element type (line 910)
    # Getting the type of 'r1k' (line 910)
    r1k_124444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 910, 26), 'r1k', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 910, 26), tuple_124443, r1k_124444)
    # Adding element type (line 910)
    # Getting the type of 'r2k' (line 910)
    r2k_124445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 910, 31), 'r2k', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 910, 26), tuple_124443, r2k_124445)
    
    # Processing the call keyword arguments (line 910)
    kwargs_124446 = {}
    # Getting the type of 'ma' (line 910)
    ma_124441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 910, 10), 'ma', False)
    # Obtaining the member 'concatenate' of a type (line 910)
    concatenate_124442 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 910, 10), ma_124441, 'concatenate')
    # Calling concatenate(args, kwargs) (line 910)
    concatenate_call_result_124447 = invoke(stypy.reporting.localization.Localization(__file__, 910, 10), concatenate_124442, *[tuple_124443], **kwargs_124446)
    
    # Assigning a type to the variable 'aux' (line 910)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 910, 4), 'aux', concatenate_call_result_124447)
    
    # Assigning a Call to a Name (line 911):
    
    # Assigning a Call to a Name (line 911):
    
    # Call to argsort(...): (line 911)
    # Processing the call keyword arguments (line 911)
    # Getting the type of 'key' (line 911)
    key_124450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 911, 33), 'key', False)
    keyword_124451 = key_124450
    kwargs_124452 = {'order': keyword_124451}
    # Getting the type of 'aux' (line 911)
    aux_124448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 911, 15), 'aux', False)
    # Obtaining the member 'argsort' of a type (line 911)
    argsort_124449 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 911, 15), aux_124448, 'argsort')
    # Calling argsort(args, kwargs) (line 911)
    argsort_call_result_124453 = invoke(stypy.reporting.localization.Localization(__file__, 911, 15), argsort_124449, *[], **kwargs_124452)
    
    # Assigning a type to the variable 'idx_sort' (line 911)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 911, 4), 'idx_sort', argsort_call_result_124453)
    
    # Assigning a Subscript to a Name (line 912):
    
    # Assigning a Subscript to a Name (line 912):
    
    # Obtaining the type of the subscript
    # Getting the type of 'idx_sort' (line 912)
    idx_sort_124454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 912, 14), 'idx_sort')
    # Getting the type of 'aux' (line 912)
    aux_124455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 912, 10), 'aux')
    # Obtaining the member '__getitem__' of a type (line 912)
    getitem___124456 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 912, 10), aux_124455, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 912)
    subscript_call_result_124457 = invoke(stypy.reporting.localization.Localization(__file__, 912, 10), getitem___124456, idx_sort_124454)
    
    # Assigning a type to the variable 'aux' (line 912)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 912, 4), 'aux', subscript_call_result_124457)
    
    # Assigning a Call to a Name (line 915):
    
    # Assigning a Call to a Name (line 915):
    
    # Call to concatenate(...): (line 915)
    # Processing the call arguments (line 915)
    
    # Obtaining an instance of the builtin type 'tuple' (line 915)
    tuple_124460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 915, 30), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 915)
    # Adding element type (line 915)
    
    # Obtaining an instance of the builtin type 'list' (line 915)
    list_124461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 915, 30), 'list')
    # Adding type elements to the builtin type 'list' instance (line 915)
    # Adding element type (line 915)
    # Getting the type of 'False' (line 915)
    False_124462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 915, 31), 'False', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 915, 30), list_124461, False_124462)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 915, 30), tuple_124460, list_124461)
    # Adding element type (line 915)
    
    
    # Obtaining the type of the subscript
    int_124463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 915, 43), 'int')
    slice_124464 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 915, 39), int_124463, None, None)
    # Getting the type of 'aux' (line 915)
    aux_124465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 915, 39), 'aux', False)
    # Obtaining the member '__getitem__' of a type (line 915)
    getitem___124466 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 915, 39), aux_124465, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 915)
    subscript_call_result_124467 = invoke(stypy.reporting.localization.Localization(__file__, 915, 39), getitem___124466, slice_124464)
    
    
    # Obtaining the type of the subscript
    int_124468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 915, 55), 'int')
    slice_124469 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 915, 50), None, int_124468, None)
    # Getting the type of 'aux' (line 915)
    aux_124470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 915, 50), 'aux', False)
    # Obtaining the member '__getitem__' of a type (line 915)
    getitem___124471 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 915, 50), aux_124470, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 915)
    subscript_call_result_124472 = invoke(stypy.reporting.localization.Localization(__file__, 915, 50), getitem___124471, slice_124469)
    
    # Applying the binary operator '==' (line 915)
    result_eq_124473 = python_operator(stypy.reporting.localization.Localization(__file__, 915, 39), '==', subscript_call_result_124467, subscript_call_result_124472)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 915, 30), tuple_124460, result_eq_124473)
    
    # Processing the call keyword arguments (line 915)
    kwargs_124474 = {}
    # Getting the type of 'ma' (line 915)
    ma_124458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 915, 14), 'ma', False)
    # Obtaining the member 'concatenate' of a type (line 915)
    concatenate_124459 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 915, 14), ma_124458, 'concatenate')
    # Calling concatenate(args, kwargs) (line 915)
    concatenate_call_result_124475 = invoke(stypy.reporting.localization.Localization(__file__, 915, 14), concatenate_124459, *[tuple_124460], **kwargs_124474)
    
    # Assigning a type to the variable 'flag_in' (line 915)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 915, 4), 'flag_in', concatenate_call_result_124475)
    
    # Assigning a BinOp to a Subscript (line 916):
    
    # Assigning a BinOp to a Subscript (line 916):
    
    # Obtaining the type of the subscript
    int_124476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 916, 27), 'int')
    slice_124477 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 916, 19), int_124476, None, None)
    # Getting the type of 'flag_in' (line 916)
    flag_in_124478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 916, 19), 'flag_in')
    # Obtaining the member '__getitem__' of a type (line 916)
    getitem___124479 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 916, 19), flag_in_124478, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 916)
    subscript_call_result_124480 = invoke(stypy.reporting.localization.Localization(__file__, 916, 19), getitem___124479, slice_124477)
    
    
    # Obtaining the type of the subscript
    int_124481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 916, 42), 'int')
    slice_124482 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 916, 33), None, int_124481, None)
    # Getting the type of 'flag_in' (line 916)
    flag_in_124483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 916, 33), 'flag_in')
    # Obtaining the member '__getitem__' of a type (line 916)
    getitem___124484 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 916, 33), flag_in_124483, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 916)
    subscript_call_result_124485 = invoke(stypy.reporting.localization.Localization(__file__, 916, 33), getitem___124484, slice_124482)
    
    # Applying the binary operator '+' (line 916)
    result_add_124486 = python_operator(stypy.reporting.localization.Localization(__file__, 916, 19), '+', subscript_call_result_124480, subscript_call_result_124485)
    
    # Getting the type of 'flag_in' (line 916)
    flag_in_124487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 916, 4), 'flag_in')
    int_124488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 916, 13), 'int')
    slice_124489 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 916, 4), None, int_124488, None)
    # Storing an element on a container (line 916)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 916, 4), flag_in_124487, (slice_124489, result_add_124486))
    
    # Assigning a Subscript to a Name (line 917):
    
    # Assigning a Subscript to a Name (line 917):
    
    # Obtaining the type of the subscript
    # Getting the type of 'flag_in' (line 917)
    flag_in_124490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 917, 22), 'flag_in')
    # Getting the type of 'idx_sort' (line 917)
    idx_sort_124491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 917, 13), 'idx_sort')
    # Obtaining the member '__getitem__' of a type (line 917)
    getitem___124492 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 917, 13), idx_sort_124491, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 917)
    subscript_call_result_124493 = invoke(stypy.reporting.localization.Localization(__file__, 917, 13), getitem___124492, flag_in_124490)
    
    # Assigning a type to the variable 'idx_in' (line 917)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 917, 4), 'idx_in', subscript_call_result_124493)
    
    # Assigning a Subscript to a Name (line 918):
    
    # Assigning a Subscript to a Name (line 918):
    
    # Obtaining the type of the subscript
    
    # Getting the type of 'idx_in' (line 918)
    idx_in_124494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 918, 20), 'idx_in')
    # Getting the type of 'nb1' (line 918)
    nb1_124495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 918, 29), 'nb1')
    # Applying the binary operator '<' (line 918)
    result_lt_124496 = python_operator(stypy.reporting.localization.Localization(__file__, 918, 20), '<', idx_in_124494, nb1_124495)
    
    # Getting the type of 'idx_in' (line 918)
    idx_in_124497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 918, 12), 'idx_in')
    # Obtaining the member '__getitem__' of a type (line 918)
    getitem___124498 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 918, 12), idx_in_124497, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 918)
    subscript_call_result_124499 = invoke(stypy.reporting.localization.Localization(__file__, 918, 12), getitem___124498, result_lt_124496)
    
    # Assigning a type to the variable 'idx_1' (line 918)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 918, 4), 'idx_1', subscript_call_result_124499)
    
    # Assigning a BinOp to a Name (line 919):
    
    # Assigning a BinOp to a Name (line 919):
    
    # Obtaining the type of the subscript
    
    # Getting the type of 'idx_in' (line 919)
    idx_in_124500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 919, 20), 'idx_in')
    # Getting the type of 'nb1' (line 919)
    nb1_124501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 919, 30), 'nb1')
    # Applying the binary operator '>=' (line 919)
    result_ge_124502 = python_operator(stypy.reporting.localization.Localization(__file__, 919, 20), '>=', idx_in_124500, nb1_124501)
    
    # Getting the type of 'idx_in' (line 919)
    idx_in_124503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 919, 12), 'idx_in')
    # Obtaining the member '__getitem__' of a type (line 919)
    getitem___124504 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 919, 12), idx_in_124503, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 919)
    subscript_call_result_124505 = invoke(stypy.reporting.localization.Localization(__file__, 919, 12), getitem___124504, result_ge_124502)
    
    # Getting the type of 'nb1' (line 919)
    nb1_124506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 919, 38), 'nb1')
    # Applying the binary operator '-' (line 919)
    result_sub_124507 = python_operator(stypy.reporting.localization.Localization(__file__, 919, 12), '-', subscript_call_result_124505, nb1_124506)
    
    # Assigning a type to the variable 'idx_2' (line 919)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 919, 4), 'idx_2', result_sub_124507)
    
    # Assigning a Tuple to a Tuple (line 920):
    
    # Assigning a Call to a Name (line 920):
    
    # Call to len(...): (line 920)
    # Processing the call arguments (line 920)
    # Getting the type of 'idx_1' (line 920)
    idx_1_124509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 920, 26), 'idx_1', False)
    # Processing the call keyword arguments (line 920)
    kwargs_124510 = {}
    # Getting the type of 'len' (line 920)
    len_124508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 920, 22), 'len', False)
    # Calling len(args, kwargs) (line 920)
    len_call_result_124511 = invoke(stypy.reporting.localization.Localization(__file__, 920, 22), len_124508, *[idx_1_124509], **kwargs_124510)
    
    # Assigning a type to the variable 'tuple_assignment_122697' (line 920)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 920, 4), 'tuple_assignment_122697', len_call_result_124511)
    
    # Assigning a Call to a Name (line 920):
    
    # Call to len(...): (line 920)
    # Processing the call arguments (line 920)
    # Getting the type of 'idx_2' (line 920)
    idx_2_124513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 920, 38), 'idx_2', False)
    # Processing the call keyword arguments (line 920)
    kwargs_124514 = {}
    # Getting the type of 'len' (line 920)
    len_124512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 920, 34), 'len', False)
    # Calling len(args, kwargs) (line 920)
    len_call_result_124515 = invoke(stypy.reporting.localization.Localization(__file__, 920, 34), len_124512, *[idx_2_124513], **kwargs_124514)
    
    # Assigning a type to the variable 'tuple_assignment_122698' (line 920)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 920, 4), 'tuple_assignment_122698', len_call_result_124515)
    
    # Assigning a Name to a Name (line 920):
    # Getting the type of 'tuple_assignment_122697' (line 920)
    tuple_assignment_122697_124516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 920, 4), 'tuple_assignment_122697')
    # Assigning a type to the variable 'r1cmn' (line 920)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 920, 5), 'r1cmn', tuple_assignment_122697_124516)
    
    # Assigning a Name to a Name (line 920):
    # Getting the type of 'tuple_assignment_122698' (line 920)
    tuple_assignment_122698_124517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 920, 4), 'tuple_assignment_122698')
    # Assigning a type to the variable 'r2cmn' (line 920)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 920, 12), 'r2cmn', tuple_assignment_122698_124517)
    
    
    # Getting the type of 'jointype' (line 921)
    jointype_124518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 921, 7), 'jointype')
    str_124519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 921, 19), 'str', 'inner')
    # Applying the binary operator '==' (line 921)
    result_eq_124520 = python_operator(stypy.reporting.localization.Localization(__file__, 921, 7), '==', jointype_124518, str_124519)
    
    # Testing the type of an if condition (line 921)
    if_condition_124521 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 921, 4), result_eq_124520)
    # Assigning a type to the variable 'if_condition_124521' (line 921)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 921, 4), 'if_condition_124521', if_condition_124521)
    # SSA begins for if statement (line 921)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Tuple to a Tuple (line 922):
    
    # Assigning a Num to a Name (line 922):
    int_124522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 922, 26), 'int')
    # Assigning a type to the variable 'tuple_assignment_122699' (line 922)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 922, 8), 'tuple_assignment_122699', int_124522)
    
    # Assigning a Num to a Name (line 922):
    int_124523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 922, 29), 'int')
    # Assigning a type to the variable 'tuple_assignment_122700' (line 922)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 922, 8), 'tuple_assignment_122700', int_124523)
    
    # Assigning a Name to a Name (line 922):
    # Getting the type of 'tuple_assignment_122699' (line 922)
    tuple_assignment_122699_124524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 922, 8), 'tuple_assignment_122699')
    # Assigning a type to the variable 'r1spc' (line 922)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 922, 9), 'r1spc', tuple_assignment_122699_124524)
    
    # Assigning a Name to a Name (line 922):
    # Getting the type of 'tuple_assignment_122700' (line 922)
    tuple_assignment_122700_124525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 922, 8), 'tuple_assignment_122700')
    # Assigning a type to the variable 'r2spc' (line 922)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 922, 16), 'r2spc', tuple_assignment_122700_124525)
    # SSA branch for the else part of an if statement (line 921)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'jointype' (line 923)
    jointype_124526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 923, 9), 'jointype')
    str_124527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 923, 21), 'str', 'outer')
    # Applying the binary operator '==' (line 923)
    result_eq_124528 = python_operator(stypy.reporting.localization.Localization(__file__, 923, 9), '==', jointype_124526, str_124527)
    
    # Testing the type of an if condition (line 923)
    if_condition_124529 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 923, 9), result_eq_124528)
    # Assigning a type to the variable 'if_condition_124529' (line 923)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 923, 9), 'if_condition_124529', if_condition_124529)
    # SSA begins for if statement (line 923)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 924):
    
    # Assigning a Subscript to a Name (line 924):
    
    # Obtaining the type of the subscript
    
    # Getting the type of 'flag_in' (line 924)
    flag_in_124530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 924, 28), 'flag_in')
    # Applying the '~' unary operator (line 924)
    result_inv_124531 = python_operator(stypy.reporting.localization.Localization(__file__, 924, 27), '~', flag_in_124530)
    
    # Getting the type of 'idx_sort' (line 924)
    idx_sort_124532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 924, 18), 'idx_sort')
    # Obtaining the member '__getitem__' of a type (line 924)
    getitem___124533 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 924, 18), idx_sort_124532, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 924)
    subscript_call_result_124534 = invoke(stypy.reporting.localization.Localization(__file__, 924, 18), getitem___124533, result_inv_124531)
    
    # Assigning a type to the variable 'idx_out' (line 924)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 924, 8), 'idx_out', subscript_call_result_124534)
    
    # Assigning a Call to a Name (line 925):
    
    # Assigning a Call to a Name (line 925):
    
    # Call to concatenate(...): (line 925)
    # Processing the call arguments (line 925)
    
    # Obtaining an instance of the builtin type 'tuple' (line 925)
    tuple_124537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 925, 32), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 925)
    # Adding element type (line 925)
    # Getting the type of 'idx_1' (line 925)
    idx_1_124538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 925, 32), 'idx_1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 925, 32), tuple_124537, idx_1_124538)
    # Adding element type (line 925)
    
    # Obtaining the type of the subscript
    
    # Getting the type of 'idx_out' (line 925)
    idx_out_124539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 925, 48), 'idx_out', False)
    # Getting the type of 'nb1' (line 925)
    nb1_124540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 925, 58), 'nb1', False)
    # Applying the binary operator '<' (line 925)
    result_lt_124541 = python_operator(stypy.reporting.localization.Localization(__file__, 925, 48), '<', idx_out_124539, nb1_124540)
    
    # Getting the type of 'idx_out' (line 925)
    idx_out_124542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 925, 39), 'idx_out', False)
    # Obtaining the member '__getitem__' of a type (line 925)
    getitem___124543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 925, 39), idx_out_124542, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 925)
    subscript_call_result_124544 = invoke(stypy.reporting.localization.Localization(__file__, 925, 39), getitem___124543, result_lt_124541)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 925, 32), tuple_124537, subscript_call_result_124544)
    
    # Processing the call keyword arguments (line 925)
    kwargs_124545 = {}
    # Getting the type of 'np' (line 925)
    np_124535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 925, 16), 'np', False)
    # Obtaining the member 'concatenate' of a type (line 925)
    concatenate_124536 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 925, 16), np_124535, 'concatenate')
    # Calling concatenate(args, kwargs) (line 925)
    concatenate_call_result_124546 = invoke(stypy.reporting.localization.Localization(__file__, 925, 16), concatenate_124536, *[tuple_124537], **kwargs_124545)
    
    # Assigning a type to the variable 'idx_1' (line 925)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 925, 8), 'idx_1', concatenate_call_result_124546)
    
    # Assigning a Call to a Name (line 926):
    
    # Assigning a Call to a Name (line 926):
    
    # Call to concatenate(...): (line 926)
    # Processing the call arguments (line 926)
    
    # Obtaining an instance of the builtin type 'tuple' (line 926)
    tuple_124549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 926, 32), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 926)
    # Adding element type (line 926)
    # Getting the type of 'idx_2' (line 926)
    idx_2_124550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 926, 32), 'idx_2', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 926, 32), tuple_124549, idx_2_124550)
    # Adding element type (line 926)
    
    # Obtaining the type of the subscript
    
    # Getting the type of 'idx_out' (line 926)
    idx_out_124551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 926, 48), 'idx_out', False)
    # Getting the type of 'nb1' (line 926)
    nb1_124552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 926, 59), 'nb1', False)
    # Applying the binary operator '>=' (line 926)
    result_ge_124553 = python_operator(stypy.reporting.localization.Localization(__file__, 926, 48), '>=', idx_out_124551, nb1_124552)
    
    # Getting the type of 'idx_out' (line 926)
    idx_out_124554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 926, 39), 'idx_out', False)
    # Obtaining the member '__getitem__' of a type (line 926)
    getitem___124555 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 926, 39), idx_out_124554, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 926)
    subscript_call_result_124556 = invoke(stypy.reporting.localization.Localization(__file__, 926, 39), getitem___124555, result_ge_124553)
    
    # Getting the type of 'nb1' (line 926)
    nb1_124557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 926, 67), 'nb1', False)
    # Applying the binary operator '-' (line 926)
    result_sub_124558 = python_operator(stypy.reporting.localization.Localization(__file__, 926, 39), '-', subscript_call_result_124556, nb1_124557)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 926, 32), tuple_124549, result_sub_124558)
    
    # Processing the call keyword arguments (line 926)
    kwargs_124559 = {}
    # Getting the type of 'np' (line 926)
    np_124547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 926, 16), 'np', False)
    # Obtaining the member 'concatenate' of a type (line 926)
    concatenate_124548 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 926, 16), np_124547, 'concatenate')
    # Calling concatenate(args, kwargs) (line 926)
    concatenate_call_result_124560 = invoke(stypy.reporting.localization.Localization(__file__, 926, 16), concatenate_124548, *[tuple_124549], **kwargs_124559)
    
    # Assigning a type to the variable 'idx_2' (line 926)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 926, 8), 'idx_2', concatenate_call_result_124560)
    
    # Assigning a Tuple to a Tuple (line 927):
    
    # Assigning a BinOp to a Name (line 927):
    
    # Call to len(...): (line 927)
    # Processing the call arguments (line 927)
    # Getting the type of 'idx_1' (line 927)
    idx_1_124562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 927, 30), 'idx_1', False)
    # Processing the call keyword arguments (line 927)
    kwargs_124563 = {}
    # Getting the type of 'len' (line 927)
    len_124561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 927, 26), 'len', False)
    # Calling len(args, kwargs) (line 927)
    len_call_result_124564 = invoke(stypy.reporting.localization.Localization(__file__, 927, 26), len_124561, *[idx_1_124562], **kwargs_124563)
    
    # Getting the type of 'r1cmn' (line 927)
    r1cmn_124565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 927, 39), 'r1cmn')
    # Applying the binary operator '-' (line 927)
    result_sub_124566 = python_operator(stypy.reporting.localization.Localization(__file__, 927, 26), '-', len_call_result_124564, r1cmn_124565)
    
    # Assigning a type to the variable 'tuple_assignment_122701' (line 927)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 927, 8), 'tuple_assignment_122701', result_sub_124566)
    
    # Assigning a BinOp to a Name (line 927):
    
    # Call to len(...): (line 927)
    # Processing the call arguments (line 927)
    # Getting the type of 'idx_2' (line 927)
    idx_2_124568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 927, 50), 'idx_2', False)
    # Processing the call keyword arguments (line 927)
    kwargs_124569 = {}
    # Getting the type of 'len' (line 927)
    len_124567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 927, 46), 'len', False)
    # Calling len(args, kwargs) (line 927)
    len_call_result_124570 = invoke(stypy.reporting.localization.Localization(__file__, 927, 46), len_124567, *[idx_2_124568], **kwargs_124569)
    
    # Getting the type of 'r2cmn' (line 927)
    r2cmn_124571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 927, 59), 'r2cmn')
    # Applying the binary operator '-' (line 927)
    result_sub_124572 = python_operator(stypy.reporting.localization.Localization(__file__, 927, 46), '-', len_call_result_124570, r2cmn_124571)
    
    # Assigning a type to the variable 'tuple_assignment_122702' (line 927)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 927, 8), 'tuple_assignment_122702', result_sub_124572)
    
    # Assigning a Name to a Name (line 927):
    # Getting the type of 'tuple_assignment_122701' (line 927)
    tuple_assignment_122701_124573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 927, 8), 'tuple_assignment_122701')
    # Assigning a type to the variable 'r1spc' (line 927)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 927, 9), 'r1spc', tuple_assignment_122701_124573)
    
    # Assigning a Name to a Name (line 927):
    # Getting the type of 'tuple_assignment_122702' (line 927)
    tuple_assignment_122702_124574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 927, 8), 'tuple_assignment_122702')
    # Assigning a type to the variable 'r2spc' (line 927)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 927, 16), 'r2spc', tuple_assignment_122702_124574)
    # SSA branch for the else part of an if statement (line 923)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'jointype' (line 928)
    jointype_124575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 928, 9), 'jointype')
    str_124576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 928, 21), 'str', 'leftouter')
    # Applying the binary operator '==' (line 928)
    result_eq_124577 = python_operator(stypy.reporting.localization.Localization(__file__, 928, 9), '==', jointype_124575, str_124576)
    
    # Testing the type of an if condition (line 928)
    if_condition_124578 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 928, 9), result_eq_124577)
    # Assigning a type to the variable 'if_condition_124578' (line 928)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 928, 9), 'if_condition_124578', if_condition_124578)
    # SSA begins for if statement (line 928)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 929):
    
    # Assigning a Subscript to a Name (line 929):
    
    # Obtaining the type of the subscript
    
    # Getting the type of 'flag_in' (line 929)
    flag_in_124579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 929, 28), 'flag_in')
    # Applying the '~' unary operator (line 929)
    result_inv_124580 = python_operator(stypy.reporting.localization.Localization(__file__, 929, 27), '~', flag_in_124579)
    
    # Getting the type of 'idx_sort' (line 929)
    idx_sort_124581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 929, 18), 'idx_sort')
    # Obtaining the member '__getitem__' of a type (line 929)
    getitem___124582 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 929, 18), idx_sort_124581, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 929)
    subscript_call_result_124583 = invoke(stypy.reporting.localization.Localization(__file__, 929, 18), getitem___124582, result_inv_124580)
    
    # Assigning a type to the variable 'idx_out' (line 929)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 929, 8), 'idx_out', subscript_call_result_124583)
    
    # Assigning a Call to a Name (line 930):
    
    # Assigning a Call to a Name (line 930):
    
    # Call to concatenate(...): (line 930)
    # Processing the call arguments (line 930)
    
    # Obtaining an instance of the builtin type 'tuple' (line 930)
    tuple_124586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 930, 32), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 930)
    # Adding element type (line 930)
    # Getting the type of 'idx_1' (line 930)
    idx_1_124587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 930, 32), 'idx_1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 930, 32), tuple_124586, idx_1_124587)
    # Adding element type (line 930)
    
    # Obtaining the type of the subscript
    
    # Getting the type of 'idx_out' (line 930)
    idx_out_124588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 930, 48), 'idx_out', False)
    # Getting the type of 'nb1' (line 930)
    nb1_124589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 930, 58), 'nb1', False)
    # Applying the binary operator '<' (line 930)
    result_lt_124590 = python_operator(stypy.reporting.localization.Localization(__file__, 930, 48), '<', idx_out_124588, nb1_124589)
    
    # Getting the type of 'idx_out' (line 930)
    idx_out_124591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 930, 39), 'idx_out', False)
    # Obtaining the member '__getitem__' of a type (line 930)
    getitem___124592 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 930, 39), idx_out_124591, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 930)
    subscript_call_result_124593 = invoke(stypy.reporting.localization.Localization(__file__, 930, 39), getitem___124592, result_lt_124590)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 930, 32), tuple_124586, subscript_call_result_124593)
    
    # Processing the call keyword arguments (line 930)
    kwargs_124594 = {}
    # Getting the type of 'np' (line 930)
    np_124584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 930, 16), 'np', False)
    # Obtaining the member 'concatenate' of a type (line 930)
    concatenate_124585 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 930, 16), np_124584, 'concatenate')
    # Calling concatenate(args, kwargs) (line 930)
    concatenate_call_result_124595 = invoke(stypy.reporting.localization.Localization(__file__, 930, 16), concatenate_124585, *[tuple_124586], **kwargs_124594)
    
    # Assigning a type to the variable 'idx_1' (line 930)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 930, 8), 'idx_1', concatenate_call_result_124595)
    
    # Assigning a Tuple to a Tuple (line 931):
    
    # Assigning a BinOp to a Name (line 931):
    
    # Call to len(...): (line 931)
    # Processing the call arguments (line 931)
    # Getting the type of 'idx_1' (line 931)
    idx_1_124597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 931, 30), 'idx_1', False)
    # Processing the call keyword arguments (line 931)
    kwargs_124598 = {}
    # Getting the type of 'len' (line 931)
    len_124596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 931, 26), 'len', False)
    # Calling len(args, kwargs) (line 931)
    len_call_result_124599 = invoke(stypy.reporting.localization.Localization(__file__, 931, 26), len_124596, *[idx_1_124597], **kwargs_124598)
    
    # Getting the type of 'r1cmn' (line 931)
    r1cmn_124600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 931, 39), 'r1cmn')
    # Applying the binary operator '-' (line 931)
    result_sub_124601 = python_operator(stypy.reporting.localization.Localization(__file__, 931, 26), '-', len_call_result_124599, r1cmn_124600)
    
    # Assigning a type to the variable 'tuple_assignment_122703' (line 931)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 931, 8), 'tuple_assignment_122703', result_sub_124601)
    
    # Assigning a Num to a Name (line 931):
    int_124602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 931, 46), 'int')
    # Assigning a type to the variable 'tuple_assignment_122704' (line 931)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 931, 8), 'tuple_assignment_122704', int_124602)
    
    # Assigning a Name to a Name (line 931):
    # Getting the type of 'tuple_assignment_122703' (line 931)
    tuple_assignment_122703_124603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 931, 8), 'tuple_assignment_122703')
    # Assigning a type to the variable 'r1spc' (line 931)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 931, 9), 'r1spc', tuple_assignment_122703_124603)
    
    # Assigning a Name to a Name (line 931):
    # Getting the type of 'tuple_assignment_122704' (line 931)
    tuple_assignment_122704_124604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 931, 8), 'tuple_assignment_122704')
    # Assigning a type to the variable 'r2spc' (line 931)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 931, 16), 'r2spc', tuple_assignment_122704_124604)
    # SSA join for if statement (line 928)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 923)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 921)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Tuple to a Tuple (line 933):
    
    # Assigning a Subscript to a Name (line 933):
    
    # Obtaining the type of the subscript
    # Getting the type of 'idx_1' (line 933)
    idx_1_124605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 933, 19), 'idx_1')
    # Getting the type of 'r1' (line 933)
    r1_124606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 933, 16), 'r1')
    # Obtaining the member '__getitem__' of a type (line 933)
    getitem___124607 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 933, 16), r1_124606, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 933)
    subscript_call_result_124608 = invoke(stypy.reporting.localization.Localization(__file__, 933, 16), getitem___124607, idx_1_124605)
    
    # Assigning a type to the variable 'tuple_assignment_122705' (line 933)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 933, 4), 'tuple_assignment_122705', subscript_call_result_124608)
    
    # Assigning a Subscript to a Name (line 933):
    
    # Obtaining the type of the subscript
    # Getting the type of 'idx_2' (line 933)
    idx_2_124609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 933, 30), 'idx_2')
    # Getting the type of 'r2' (line 933)
    r2_124610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 933, 27), 'r2')
    # Obtaining the member '__getitem__' of a type (line 933)
    getitem___124611 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 933, 27), r2_124610, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 933)
    subscript_call_result_124612 = invoke(stypy.reporting.localization.Localization(__file__, 933, 27), getitem___124611, idx_2_124609)
    
    # Assigning a type to the variable 'tuple_assignment_122706' (line 933)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 933, 4), 'tuple_assignment_122706', subscript_call_result_124612)
    
    # Assigning a Name to a Name (line 933):
    # Getting the type of 'tuple_assignment_122705' (line 933)
    tuple_assignment_122705_124613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 933, 4), 'tuple_assignment_122705')
    # Assigning a type to the variable 's1' (line 933)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 933, 5), 's1', tuple_assignment_122705_124613)
    
    # Assigning a Name to a Name (line 933):
    # Getting the type of 'tuple_assignment_122706' (line 933)
    tuple_assignment_122706_124614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 933, 4), 'tuple_assignment_122706')
    # Assigning a type to the variable 's2' (line 933)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 933, 9), 's2', tuple_assignment_122706_124614)
    
    # Assigning a ListComp to a Name (line 937):
    
    # Assigning a ListComp to a Name (line 937):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'r1k' (line 937)
    r1k_124619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 937, 31), 'r1k')
    # Obtaining the member 'dtype' of a type (line 937)
    dtype_124620 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 937, 31), r1k_124619, 'dtype')
    # Obtaining the member 'descr' of a type (line 937)
    descr_124621 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 937, 31), dtype_124620, 'descr')
    comprehension_124622 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 937, 14), descr_124621)
    # Assigning a type to the variable '_' (line 937)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 937, 14), '_', comprehension_124622)
    
    # Call to list(...): (line 937)
    # Processing the call arguments (line 937)
    # Getting the type of '_' (line 937)
    __124616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 937, 19), '_', False)
    # Processing the call keyword arguments (line 937)
    kwargs_124617 = {}
    # Getting the type of 'list' (line 937)
    list_124615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 937, 14), 'list', False)
    # Calling list(args, kwargs) (line 937)
    list_call_result_124618 = invoke(stypy.reporting.localization.Localization(__file__, 937, 14), list_124615, *[__124616], **kwargs_124617)
    
    list_124623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 937, 14), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 937, 14), list_124623, list_call_result_124618)
    # Assigning a type to the variable 'ndtype' (line 937)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 937, 4), 'ndtype', list_124623)
    
    # Call to extend(...): (line 939)
    # Processing the call arguments (line 939)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 939, 18, True)
    # Calculating comprehension expression
    # Getting the type of 'r1' (line 939)
    r1_124636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 939, 35), 'r1', False)
    # Obtaining the member 'dtype' of a type (line 939)
    dtype_124637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 939, 35), r1_124636, 'dtype')
    # Obtaining the member 'descr' of a type (line 939)
    descr_124638 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 939, 35), dtype_124637, 'descr')
    comprehension_124639 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 939, 18), descr_124638)
    # Assigning a type to the variable '_' (line 939)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 939, 18), '_', comprehension_124639)
    
    
    # Obtaining the type of the subscript
    int_124630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 939, 55), 'int')
    # Getting the type of '_' (line 939)
    __124631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 939, 53), '_', False)
    # Obtaining the member '__getitem__' of a type (line 939)
    getitem___124632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 939, 53), __124631, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 939)
    subscript_call_result_124633 = invoke(stypy.reporting.localization.Localization(__file__, 939, 53), getitem___124632, int_124630)
    
    # Getting the type of 'key' (line 939)
    key_124634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 939, 65), 'key', False)
    # Applying the binary operator 'notin' (line 939)
    result_contains_124635 = python_operator(stypy.reporting.localization.Localization(__file__, 939, 53), 'notin', subscript_call_result_124633, key_124634)
    
    
    # Call to list(...): (line 939)
    # Processing the call arguments (line 939)
    # Getting the type of '_' (line 939)
    __124627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 939, 23), '_', False)
    # Processing the call keyword arguments (line 939)
    kwargs_124628 = {}
    # Getting the type of 'list' (line 939)
    list_124626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 939, 18), 'list', False)
    # Calling list(args, kwargs) (line 939)
    list_call_result_124629 = invoke(stypy.reporting.localization.Localization(__file__, 939, 18), list_124626, *[__124627], **kwargs_124628)
    
    list_124640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 939, 18), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 939, 18), list_124640, list_call_result_124629)
    # Processing the call keyword arguments (line 939)
    kwargs_124641 = {}
    # Getting the type of 'ndtype' (line 939)
    ndtype_124624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 939, 4), 'ndtype', False)
    # Obtaining the member 'extend' of a type (line 939)
    extend_124625 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 939, 4), ndtype_124624, 'extend')
    # Calling extend(args, kwargs) (line 939)
    extend_call_result_124642 = invoke(stypy.reporting.localization.Localization(__file__, 939, 4), extend_124625, *[list_124640], **kwargs_124641)
    
    
    # Assigning a Call to a Name (line 941):
    
    # Assigning a Call to a Name (line 941):
    
    # Call to list(...): (line 941)
    # Processing the call arguments (line 941)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 941, 17, True)
    # Calculating comprehension expression
    # Getting the type of 'ndtype' (line 941)
    ndtype_124648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 941, 31), 'ndtype', False)
    comprehension_124649 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 941, 17), ndtype_124648)
    # Assigning a type to the variable '_' (line 941)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 941, 17), '_', comprehension_124649)
    
    # Obtaining the type of the subscript
    int_124644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 941, 19), 'int')
    # Getting the type of '_' (line 941)
    __124645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 941, 17), '_', False)
    # Obtaining the member '__getitem__' of a type (line 941)
    getitem___124646 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 941, 17), __124645, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 941)
    subscript_call_result_124647 = invoke(stypy.reporting.localization.Localization(__file__, 941, 17), getitem___124646, int_124644)
    
    list_124650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 941, 17), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 941, 17), list_124650, subscript_call_result_124647)
    # Processing the call keyword arguments (line 941)
    kwargs_124651 = {}
    # Getting the type of 'list' (line 941)
    list_124643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 941, 12), 'list', False)
    # Calling list(args, kwargs) (line 941)
    list_call_result_124652 = invoke(stypy.reporting.localization.Localization(__file__, 941, 12), list_124643, *[list_124650], **kwargs_124651)
    
    # Assigning a type to the variable 'names' (line 941)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 941, 4), 'names', list_call_result_124652)
    
    # Getting the type of 'r2' (line 942)
    r2_124653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 942, 16), 'r2')
    # Obtaining the member 'dtype' of a type (line 942)
    dtype_124654 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 942, 16), r2_124653, 'dtype')
    # Obtaining the member 'descr' of a type (line 942)
    descr_124655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 942, 16), dtype_124654, 'descr')
    # Testing the type of a for loop iterable (line 942)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 942, 4), descr_124655)
    # Getting the type of the for loop variable (line 942)
    for_loop_var_124656 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 942, 4), descr_124655)
    # Assigning a type to the variable 'desc' (line 942)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 942, 4), 'desc', for_loop_var_124656)
    # SSA begins for a for statement (line 942)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 943):
    
    # Assigning a Call to a Name (line 943):
    
    # Call to list(...): (line 943)
    # Processing the call arguments (line 943)
    # Getting the type of 'desc' (line 943)
    desc_124658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 943, 20), 'desc', False)
    # Processing the call keyword arguments (line 943)
    kwargs_124659 = {}
    # Getting the type of 'list' (line 943)
    list_124657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 943, 15), 'list', False)
    # Calling list(args, kwargs) (line 943)
    list_call_result_124660 = invoke(stypy.reporting.localization.Localization(__file__, 943, 15), list_124657, *[desc_124658], **kwargs_124659)
    
    # Assigning a type to the variable 'desc' (line 943)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 943, 8), 'desc', list_call_result_124660)
    
    # Assigning a Subscript to a Name (line 944):
    
    # Assigning a Subscript to a Name (line 944):
    
    # Obtaining the type of the subscript
    int_124661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 944, 20), 'int')
    # Getting the type of 'desc' (line 944)
    desc_124662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 944, 15), 'desc')
    # Obtaining the member '__getitem__' of a type (line 944)
    getitem___124663 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 944, 15), desc_124662, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 944)
    subscript_call_result_124664 = invoke(stypy.reporting.localization.Localization(__file__, 944, 15), getitem___124663, int_124661)
    
    # Assigning a type to the variable 'name' (line 944)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 944, 8), 'name', subscript_call_result_124664)
    
    
    # Getting the type of 'name' (line 946)
    name_124665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 946, 11), 'name')
    # Getting the type of 'names' (line 946)
    names_124666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 946, 19), 'names')
    # Applying the binary operator 'in' (line 946)
    result_contains_124667 = python_operator(stypy.reporting.localization.Localization(__file__, 946, 11), 'in', name_124665, names_124666)
    
    # Testing the type of an if condition (line 946)
    if_condition_124668 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 946, 8), result_contains_124667)
    # Assigning a type to the variable 'if_condition_124668' (line 946)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 946, 8), 'if_condition_124668', if_condition_124668)
    # SSA begins for if statement (line 946)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 947):
    
    # Assigning a Call to a Name (line 947):
    
    # Call to index(...): (line 947)
    # Processing the call arguments (line 947)
    # Getting the type of 'desc' (line 947)
    desc_124671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 947, 35), 'desc', False)
    # Processing the call keyword arguments (line 947)
    kwargs_124672 = {}
    # Getting the type of 'ndtype' (line 947)
    ndtype_124669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 947, 22), 'ndtype', False)
    # Obtaining the member 'index' of a type (line 947)
    index_124670 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 947, 22), ndtype_124669, 'index')
    # Calling index(args, kwargs) (line 947)
    index_call_result_124673 = invoke(stypy.reporting.localization.Localization(__file__, 947, 22), index_124670, *[desc_124671], **kwargs_124672)
    
    # Assigning a type to the variable 'nameidx' (line 947)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 947, 12), 'nameidx', index_call_result_124673)
    
    # Assigning a Subscript to a Name (line 948):
    
    # Assigning a Subscript to a Name (line 948):
    
    # Obtaining the type of the subscript
    # Getting the type of 'nameidx' (line 948)
    nameidx_124674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 948, 29), 'nameidx')
    # Getting the type of 'ndtype' (line 948)
    ndtype_124675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 948, 22), 'ndtype')
    # Obtaining the member '__getitem__' of a type (line 948)
    getitem___124676 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 948, 22), ndtype_124675, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 948)
    subscript_call_result_124677 = invoke(stypy.reporting.localization.Localization(__file__, 948, 22), getitem___124676, nameidx_124674)
    
    # Assigning a type to the variable 'current' (line 948)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 948, 12), 'current', subscript_call_result_124677)
    
    
    # Getting the type of 'name' (line 950)
    name_124678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 950, 15), 'name')
    # Getting the type of 'key' (line 950)
    key_124679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 950, 23), 'key')
    # Applying the binary operator 'in' (line 950)
    result_contains_124680 = python_operator(stypy.reporting.localization.Localization(__file__, 950, 15), 'in', name_124678, key_124679)
    
    # Testing the type of an if condition (line 950)
    if_condition_124681 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 950, 12), result_contains_124680)
    # Assigning a type to the variable 'if_condition_124681' (line 950)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 950, 12), 'if_condition_124681', if_condition_124681)
    # SSA begins for if statement (line 950)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Subscript (line 951):
    
    # Assigning a Call to a Subscript (line 951):
    
    # Call to max(...): (line 951)
    # Processing the call arguments (line 951)
    
    # Obtaining the type of the subscript
    int_124683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 951, 39), 'int')
    # Getting the type of 'desc' (line 951)
    desc_124684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 951, 34), 'desc', False)
    # Obtaining the member '__getitem__' of a type (line 951)
    getitem___124685 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 951, 34), desc_124684, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 951)
    subscript_call_result_124686 = invoke(stypy.reporting.localization.Localization(__file__, 951, 34), getitem___124685, int_124683)
    
    
    # Obtaining the type of the subscript
    int_124687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 951, 51), 'int')
    # Getting the type of 'current' (line 951)
    current_124688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 951, 43), 'current', False)
    # Obtaining the member '__getitem__' of a type (line 951)
    getitem___124689 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 951, 43), current_124688, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 951)
    subscript_call_result_124690 = invoke(stypy.reporting.localization.Localization(__file__, 951, 43), getitem___124689, int_124687)
    
    # Processing the call keyword arguments (line 951)
    kwargs_124691 = {}
    # Getting the type of 'max' (line 951)
    max_124682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 951, 30), 'max', False)
    # Calling max(args, kwargs) (line 951)
    max_call_result_124692 = invoke(stypy.reporting.localization.Localization(__file__, 951, 30), max_124682, *[subscript_call_result_124686, subscript_call_result_124690], **kwargs_124691)
    
    # Getting the type of 'current' (line 951)
    current_124693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 951, 16), 'current')
    int_124694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 951, 24), 'int')
    # Storing an element on a container (line 951)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 951, 16), current_124693, (int_124694, max_call_result_124692))
    # SSA branch for the else part of an if statement (line 950)
    module_type_store.open_ssa_branch('else')
    
    # Getting the type of 'current' (line 954)
    current_124695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 954, 16), 'current')
    
    # Obtaining the type of the subscript
    int_124696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 954, 24), 'int')
    # Getting the type of 'current' (line 954)
    current_124697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 954, 16), 'current')
    # Obtaining the member '__getitem__' of a type (line 954)
    getitem___124698 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 954, 16), current_124697, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 954)
    subscript_call_result_124699 = invoke(stypy.reporting.localization.Localization(__file__, 954, 16), getitem___124698, int_124696)
    
    # Getting the type of 'r1postfix' (line 954)
    r1postfix_124700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 954, 30), 'r1postfix')
    # Applying the binary operator '+=' (line 954)
    result_iadd_124701 = python_operator(stypy.reporting.localization.Localization(__file__, 954, 16), '+=', subscript_call_result_124699, r1postfix_124700)
    # Getting the type of 'current' (line 954)
    current_124702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 954, 16), 'current')
    int_124703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 954, 24), 'int')
    # Storing an element on a container (line 954)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 954, 16), current_124702, (int_124703, result_iadd_124701))
    
    
    # Getting the type of 'desc' (line 955)
    desc_124704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 955, 16), 'desc')
    
    # Obtaining the type of the subscript
    int_124705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 955, 21), 'int')
    # Getting the type of 'desc' (line 955)
    desc_124706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 955, 16), 'desc')
    # Obtaining the member '__getitem__' of a type (line 955)
    getitem___124707 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 955, 16), desc_124706, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 955)
    subscript_call_result_124708 = invoke(stypy.reporting.localization.Localization(__file__, 955, 16), getitem___124707, int_124705)
    
    # Getting the type of 'r2postfix' (line 955)
    r2postfix_124709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 955, 27), 'r2postfix')
    # Applying the binary operator '+=' (line 955)
    result_iadd_124710 = python_operator(stypy.reporting.localization.Localization(__file__, 955, 16), '+=', subscript_call_result_124708, r2postfix_124709)
    # Getting the type of 'desc' (line 955)
    desc_124711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 955, 16), 'desc')
    int_124712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 955, 21), 'int')
    # Storing an element on a container (line 955)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 955, 16), desc_124711, (int_124712, result_iadd_124710))
    
    
    # Call to insert(...): (line 956)
    # Processing the call arguments (line 956)
    # Getting the type of 'nameidx' (line 956)
    nameidx_124715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 956, 30), 'nameidx', False)
    int_124716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 956, 40), 'int')
    # Applying the binary operator '+' (line 956)
    result_add_124717 = python_operator(stypy.reporting.localization.Localization(__file__, 956, 30), '+', nameidx_124715, int_124716)
    
    # Getting the type of 'desc' (line 956)
    desc_124718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 956, 43), 'desc', False)
    # Processing the call keyword arguments (line 956)
    kwargs_124719 = {}
    # Getting the type of 'ndtype' (line 956)
    ndtype_124713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 956, 16), 'ndtype', False)
    # Obtaining the member 'insert' of a type (line 956)
    insert_124714 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 956, 16), ndtype_124713, 'insert')
    # Calling insert(args, kwargs) (line 956)
    insert_call_result_124720 = invoke(stypy.reporting.localization.Localization(__file__, 956, 16), insert_124714, *[result_add_124717, desc_124718], **kwargs_124719)
    
    # SSA join for if statement (line 950)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 946)
    module_type_store.open_ssa_branch('else')
    
    # Call to extend(...): (line 959)
    # Processing the call arguments (line 959)
    
    # Obtaining the type of the subscript
    int_124723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 959, 30), 'int')
    # Getting the type of 'desc' (line 959)
    desc_124724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 959, 25), 'desc', False)
    # Obtaining the member '__getitem__' of a type (line 959)
    getitem___124725 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 959, 25), desc_124724, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 959)
    subscript_call_result_124726 = invoke(stypy.reporting.localization.Localization(__file__, 959, 25), getitem___124725, int_124723)
    
    # Processing the call keyword arguments (line 959)
    kwargs_124727 = {}
    # Getting the type of 'names' (line 959)
    names_124721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 959, 12), 'names', False)
    # Obtaining the member 'extend' of a type (line 959)
    extend_124722 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 959, 12), names_124721, 'extend')
    # Calling extend(args, kwargs) (line 959)
    extend_call_result_124728 = invoke(stypy.reporting.localization.Localization(__file__, 959, 12), extend_124722, *[subscript_call_result_124726], **kwargs_124727)
    
    
    # Call to append(...): (line 960)
    # Processing the call arguments (line 960)
    # Getting the type of 'desc' (line 960)
    desc_124731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 960, 26), 'desc', False)
    # Processing the call keyword arguments (line 960)
    kwargs_124732 = {}
    # Getting the type of 'ndtype' (line 960)
    ndtype_124729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 960, 12), 'ndtype', False)
    # Obtaining the member 'append' of a type (line 960)
    append_124730 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 960, 12), ndtype_124729, 'append')
    # Calling append(args, kwargs) (line 960)
    append_call_result_124733 = invoke(stypy.reporting.localization.Localization(__file__, 960, 12), append_124730, *[desc_124731], **kwargs_124732)
    
    # SSA join for if statement (line 946)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a ListComp to a Name (line 962):
    
    # Assigning a ListComp to a Name (line 962):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'ndtype' (line 962)
    ndtype_124738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 962, 32), 'ndtype')
    comprehension_124739 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 962, 14), ndtype_124738)
    # Assigning a type to the variable '_' (line 962)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 962, 14), '_', comprehension_124739)
    
    # Call to tuple(...): (line 962)
    # Processing the call arguments (line 962)
    # Getting the type of '_' (line 962)
    __124735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 962, 20), '_', False)
    # Processing the call keyword arguments (line 962)
    kwargs_124736 = {}
    # Getting the type of 'tuple' (line 962)
    tuple_124734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 962, 14), 'tuple', False)
    # Calling tuple(args, kwargs) (line 962)
    tuple_call_result_124737 = invoke(stypy.reporting.localization.Localization(__file__, 962, 14), tuple_124734, *[__124735], **kwargs_124736)
    
    list_124740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 962, 14), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 962, 14), list_124740, tuple_call_result_124737)
    # Assigning a type to the variable 'ndtype' (line 962)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 962, 4), 'ndtype', list_124740)
    
    # Assigning a Call to a Name (line 965):
    
    # Assigning a Call to a Name (line 965):
    
    # Call to max(...): (line 965)
    # Processing the call arguments (line 965)
    # Getting the type of 'r1cmn' (line 965)
    r1cmn_124742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 965, 14), 'r1cmn', False)
    # Getting the type of 'r2cmn' (line 965)
    r2cmn_124743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 965, 21), 'r2cmn', False)
    # Processing the call keyword arguments (line 965)
    kwargs_124744 = {}
    # Getting the type of 'max' (line 965)
    max_124741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 965, 10), 'max', False)
    # Calling max(args, kwargs) (line 965)
    max_call_result_124745 = invoke(stypy.reporting.localization.Localization(__file__, 965, 10), max_124741, *[r1cmn_124742, r2cmn_124743], **kwargs_124744)
    
    # Assigning a type to the variable 'cmn' (line 965)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 965, 4), 'cmn', max_call_result_124745)
    
    # Assigning a Call to a Name (line 967):
    
    # Assigning a Call to a Name (line 967):
    
    # Call to masked_all(...): (line 967)
    # Processing the call arguments (line 967)
    
    # Obtaining an instance of the builtin type 'tuple' (line 967)
    tuple_124748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 967, 28), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 967)
    # Adding element type (line 967)
    # Getting the type of 'cmn' (line 967)
    cmn_124749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 967, 28), 'cmn', False)
    # Getting the type of 'r1spc' (line 967)
    r1spc_124750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 967, 34), 'r1spc', False)
    # Applying the binary operator '+' (line 967)
    result_add_124751 = python_operator(stypy.reporting.localization.Localization(__file__, 967, 28), '+', cmn_124749, r1spc_124750)
    
    # Getting the type of 'r2spc' (line 967)
    r2spc_124752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 967, 42), 'r2spc', False)
    # Applying the binary operator '+' (line 967)
    result_add_124753 = python_operator(stypy.reporting.localization.Localization(__file__, 967, 40), '+', result_add_124751, r2spc_124752)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 967, 28), tuple_124748, result_add_124753)
    
    # Processing the call keyword arguments (line 967)
    # Getting the type of 'ndtype' (line 967)
    ndtype_124754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 967, 57), 'ndtype', False)
    keyword_124755 = ndtype_124754
    kwargs_124756 = {'dtype': keyword_124755}
    # Getting the type of 'ma' (line 967)
    ma_124746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 967, 13), 'ma', False)
    # Obtaining the member 'masked_all' of a type (line 967)
    masked_all_124747 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 967, 13), ma_124746, 'masked_all')
    # Calling masked_all(args, kwargs) (line 967)
    masked_all_call_result_124757 = invoke(stypy.reporting.localization.Localization(__file__, 967, 13), masked_all_124747, *[tuple_124748], **kwargs_124756)
    
    # Assigning a type to the variable 'output' (line 967)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 967, 4), 'output', masked_all_call_result_124757)
    
    # Assigning a Attribute to a Name (line 968):
    
    # Assigning a Attribute to a Name (line 968):
    # Getting the type of 'output' (line 968)
    output_124758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 968, 12), 'output')
    # Obtaining the member 'dtype' of a type (line 968)
    dtype_124759 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 968, 12), output_124758, 'dtype')
    # Obtaining the member 'names' of a type (line 968)
    names_124760 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 968, 12), dtype_124759, 'names')
    # Assigning a type to the variable 'names' (line 968)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 968, 4), 'names', names_124760)
    
    # Getting the type of 'r1names' (line 969)
    r1names_124761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 969, 13), 'r1names')
    # Testing the type of a for loop iterable (line 969)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 969, 4), r1names_124761)
    # Getting the type of the for loop variable (line 969)
    for_loop_var_124762 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 969, 4), r1names_124761)
    # Assigning a type to the variable 'f' (line 969)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 969, 4), 'f', for_loop_var_124762)
    # SSA begins for a for statement (line 969)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Subscript to a Name (line 970):
    
    # Assigning a Subscript to a Name (line 970):
    
    # Obtaining the type of the subscript
    # Getting the type of 'f' (line 970)
    f_124763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 970, 22), 'f')
    # Getting the type of 's1' (line 970)
    s1_124764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 970, 19), 's1')
    # Obtaining the member '__getitem__' of a type (line 970)
    getitem___124765 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 970, 19), s1_124764, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 970)
    subscript_call_result_124766 = invoke(stypy.reporting.localization.Localization(__file__, 970, 19), getitem___124765, f_124763)
    
    # Assigning a type to the variable 'selected' (line 970)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 970, 8), 'selected', subscript_call_result_124766)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'f' (line 971)
    f_124767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 971, 11), 'f')
    # Getting the type of 'names' (line 971)
    names_124768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 971, 20), 'names')
    # Applying the binary operator 'notin' (line 971)
    result_contains_124769 = python_operator(stypy.reporting.localization.Localization(__file__, 971, 11), 'notin', f_124767, names_124768)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'f' (line 971)
    f_124770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 971, 30), 'f')
    # Getting the type of 'r2names' (line 971)
    r2names_124771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 971, 35), 'r2names')
    # Applying the binary operator 'in' (line 971)
    result_contains_124772 = python_operator(stypy.reporting.localization.Localization(__file__, 971, 30), 'in', f_124770, r2names_124771)
    
    
    # Getting the type of 'r2postfix' (line 971)
    r2postfix_124773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 971, 51), 'r2postfix')
    # Applying the 'not' unary operator (line 971)
    result_not__124774 = python_operator(stypy.reporting.localization.Localization(__file__, 971, 47), 'not', r2postfix_124773)
    
    # Applying the binary operator 'and' (line 971)
    result_and_keyword_124775 = python_operator(stypy.reporting.localization.Localization(__file__, 971, 30), 'and', result_contains_124772, result_not__124774)
    
    # Getting the type of 'f' (line 971)
    f_124776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 971, 65), 'f')
    # Getting the type of 'key' (line 971)
    key_124777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 971, 74), 'key')
    # Applying the binary operator 'notin' (line 971)
    result_contains_124778 = python_operator(stypy.reporting.localization.Localization(__file__, 971, 65), 'notin', f_124776, key_124777)
    
    # Applying the binary operator 'and' (line 971)
    result_and_keyword_124779 = python_operator(stypy.reporting.localization.Localization(__file__, 971, 30), 'and', result_and_keyword_124775, result_contains_124778)
    
    # Applying the binary operator 'or' (line 971)
    result_or_keyword_124780 = python_operator(stypy.reporting.localization.Localization(__file__, 971, 11), 'or', result_contains_124769, result_and_keyword_124779)
    
    # Testing the type of an if condition (line 971)
    if_condition_124781 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 971, 8), result_or_keyword_124780)
    # Assigning a type to the variable 'if_condition_124781' (line 971)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 971, 8), 'if_condition_124781', if_condition_124781)
    # SSA begins for if statement (line 971)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'f' (line 972)
    f_124782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 972, 12), 'f')
    # Getting the type of 'r1postfix' (line 972)
    r1postfix_124783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 972, 17), 'r1postfix')
    # Applying the binary operator '+=' (line 972)
    result_iadd_124784 = python_operator(stypy.reporting.localization.Localization(__file__, 972, 12), '+=', f_124782, r1postfix_124783)
    # Assigning a type to the variable 'f' (line 972)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 972, 12), 'f', result_iadd_124784)
    
    # SSA join for if statement (line 971)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Subscript to a Name (line 973):
    
    # Assigning a Subscript to a Name (line 973):
    
    # Obtaining the type of the subscript
    # Getting the type of 'f' (line 973)
    f_124785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 973, 25), 'f')
    # Getting the type of 'output' (line 973)
    output_124786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 973, 18), 'output')
    # Obtaining the member '__getitem__' of a type (line 973)
    getitem___124787 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 973, 18), output_124786, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 973)
    subscript_call_result_124788 = invoke(stypy.reporting.localization.Localization(__file__, 973, 18), getitem___124787, f_124785)
    
    # Assigning a type to the variable 'current' (line 973)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 973, 8), 'current', subscript_call_result_124788)
    
    # Assigning a Subscript to a Subscript (line 974):
    
    # Assigning a Subscript to a Subscript (line 974):
    
    # Obtaining the type of the subscript
    # Getting the type of 'r1cmn' (line 974)
    r1cmn_124789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 974, 36), 'r1cmn')
    slice_124790 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 974, 26), None, r1cmn_124789, None)
    # Getting the type of 'selected' (line 974)
    selected_124791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 974, 26), 'selected')
    # Obtaining the member '__getitem__' of a type (line 974)
    getitem___124792 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 974, 26), selected_124791, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 974)
    subscript_call_result_124793 = invoke(stypy.reporting.localization.Localization(__file__, 974, 26), getitem___124792, slice_124790)
    
    # Getting the type of 'current' (line 974)
    current_124794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 974, 8), 'current')
    # Getting the type of 'r1cmn' (line 974)
    r1cmn_124795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 974, 17), 'r1cmn')
    slice_124796 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 974, 8), None, r1cmn_124795, None)
    # Storing an element on a container (line 974)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 974, 8), current_124794, (slice_124796, subscript_call_result_124793))
    
    
    # Getting the type of 'jointype' (line 975)
    jointype_124797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 975, 11), 'jointype')
    
    # Obtaining an instance of the builtin type 'tuple' (line 975)
    tuple_124798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 975, 24), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 975)
    # Adding element type (line 975)
    str_124799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 975, 24), 'str', 'outer')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 975, 24), tuple_124798, str_124799)
    # Adding element type (line 975)
    str_124800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 975, 33), 'str', 'leftouter')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 975, 24), tuple_124798, str_124800)
    
    # Applying the binary operator 'in' (line 975)
    result_contains_124801 = python_operator(stypy.reporting.localization.Localization(__file__, 975, 11), 'in', jointype_124797, tuple_124798)
    
    # Testing the type of an if condition (line 975)
    if_condition_124802 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 975, 8), result_contains_124801)
    # Assigning a type to the variable 'if_condition_124802' (line 975)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 975, 8), 'if_condition_124802', if_condition_124802)
    # SSA begins for if statement (line 975)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Subscript (line 976):
    
    # Assigning a Subscript to a Subscript (line 976):
    
    # Obtaining the type of the subscript
    # Getting the type of 'r1cmn' (line 976)
    r1cmn_124803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 976, 48), 'r1cmn')
    slice_124804 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 976, 39), r1cmn_124803, None, None)
    # Getting the type of 'selected' (line 976)
    selected_124805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 976, 39), 'selected')
    # Obtaining the member '__getitem__' of a type (line 976)
    getitem___124806 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 976, 39), selected_124805, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 976)
    subscript_call_result_124807 = invoke(stypy.reporting.localization.Localization(__file__, 976, 39), getitem___124806, slice_124804)
    
    # Getting the type of 'current' (line 976)
    current_124808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 976, 12), 'current')
    # Getting the type of 'cmn' (line 976)
    cmn_124809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 976, 20), 'cmn')
    # Getting the type of 'cmn' (line 976)
    cmn_124810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 976, 24), 'cmn')
    # Getting the type of 'r1spc' (line 976)
    r1spc_124811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 976, 30), 'r1spc')
    # Applying the binary operator '+' (line 976)
    result_add_124812 = python_operator(stypy.reporting.localization.Localization(__file__, 976, 24), '+', cmn_124810, r1spc_124811)
    
    slice_124813 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 976, 12), cmn_124809, result_add_124812, None)
    # Storing an element on a container (line 976)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 976, 12), current_124808, (slice_124813, subscript_call_result_124807))
    # SSA join for if statement (line 975)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'r2names' (line 977)
    r2names_124814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 977, 13), 'r2names')
    # Testing the type of a for loop iterable (line 977)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 977, 4), r2names_124814)
    # Getting the type of the for loop variable (line 977)
    for_loop_var_124815 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 977, 4), r2names_124814)
    # Assigning a type to the variable 'f' (line 977)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 977, 4), 'f', for_loop_var_124815)
    # SSA begins for a for statement (line 977)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Subscript to a Name (line 978):
    
    # Assigning a Subscript to a Name (line 978):
    
    # Obtaining the type of the subscript
    # Getting the type of 'f' (line 978)
    f_124816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 978, 22), 'f')
    # Getting the type of 's2' (line 978)
    s2_124817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 978, 19), 's2')
    # Obtaining the member '__getitem__' of a type (line 978)
    getitem___124818 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 978, 19), s2_124817, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 978)
    subscript_call_result_124819 = invoke(stypy.reporting.localization.Localization(__file__, 978, 19), getitem___124818, f_124816)
    
    # Assigning a type to the variable 'selected' (line 978)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 978, 8), 'selected', subscript_call_result_124819)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'f' (line 979)
    f_124820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 979, 11), 'f')
    # Getting the type of 'names' (line 979)
    names_124821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 979, 20), 'names')
    # Applying the binary operator 'notin' (line 979)
    result_contains_124822 = python_operator(stypy.reporting.localization.Localization(__file__, 979, 11), 'notin', f_124820, names_124821)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'f' (line 979)
    f_124823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 979, 30), 'f')
    # Getting the type of 'r1names' (line 979)
    r1names_124824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 979, 35), 'r1names')
    # Applying the binary operator 'in' (line 979)
    result_contains_124825 = python_operator(stypy.reporting.localization.Localization(__file__, 979, 30), 'in', f_124823, r1names_124824)
    
    
    # Getting the type of 'r1postfix' (line 979)
    r1postfix_124826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 979, 51), 'r1postfix')
    # Applying the 'not' unary operator (line 979)
    result_not__124827 = python_operator(stypy.reporting.localization.Localization(__file__, 979, 47), 'not', r1postfix_124826)
    
    # Applying the binary operator 'and' (line 979)
    result_and_keyword_124828 = python_operator(stypy.reporting.localization.Localization(__file__, 979, 30), 'and', result_contains_124825, result_not__124827)
    
    # Getting the type of 'f' (line 979)
    f_124829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 979, 65), 'f')
    # Getting the type of 'key' (line 979)
    key_124830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 979, 74), 'key')
    # Applying the binary operator 'notin' (line 979)
    result_contains_124831 = python_operator(stypy.reporting.localization.Localization(__file__, 979, 65), 'notin', f_124829, key_124830)
    
    # Applying the binary operator 'and' (line 979)
    result_and_keyword_124832 = python_operator(stypy.reporting.localization.Localization(__file__, 979, 30), 'and', result_and_keyword_124828, result_contains_124831)
    
    # Applying the binary operator 'or' (line 979)
    result_or_keyword_124833 = python_operator(stypy.reporting.localization.Localization(__file__, 979, 11), 'or', result_contains_124822, result_and_keyword_124832)
    
    # Testing the type of an if condition (line 979)
    if_condition_124834 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 979, 8), result_or_keyword_124833)
    # Assigning a type to the variable 'if_condition_124834' (line 979)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 979, 8), 'if_condition_124834', if_condition_124834)
    # SSA begins for if statement (line 979)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'f' (line 980)
    f_124835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 980, 12), 'f')
    # Getting the type of 'r2postfix' (line 980)
    r2postfix_124836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 980, 17), 'r2postfix')
    # Applying the binary operator '+=' (line 980)
    result_iadd_124837 = python_operator(stypy.reporting.localization.Localization(__file__, 980, 12), '+=', f_124835, r2postfix_124836)
    # Assigning a type to the variable 'f' (line 980)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 980, 12), 'f', result_iadd_124837)
    
    # SSA join for if statement (line 979)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Subscript to a Name (line 981):
    
    # Assigning a Subscript to a Name (line 981):
    
    # Obtaining the type of the subscript
    # Getting the type of 'f' (line 981)
    f_124838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 981, 25), 'f')
    # Getting the type of 'output' (line 981)
    output_124839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 981, 18), 'output')
    # Obtaining the member '__getitem__' of a type (line 981)
    getitem___124840 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 981, 18), output_124839, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 981)
    subscript_call_result_124841 = invoke(stypy.reporting.localization.Localization(__file__, 981, 18), getitem___124840, f_124838)
    
    # Assigning a type to the variable 'current' (line 981)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 981, 8), 'current', subscript_call_result_124841)
    
    # Assigning a Subscript to a Subscript (line 982):
    
    # Assigning a Subscript to a Subscript (line 982):
    
    # Obtaining the type of the subscript
    # Getting the type of 'r2cmn' (line 982)
    r2cmn_124842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 982, 36), 'r2cmn')
    slice_124843 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 982, 26), None, r2cmn_124842, None)
    # Getting the type of 'selected' (line 982)
    selected_124844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 982, 26), 'selected')
    # Obtaining the member '__getitem__' of a type (line 982)
    getitem___124845 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 982, 26), selected_124844, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 982)
    subscript_call_result_124846 = invoke(stypy.reporting.localization.Localization(__file__, 982, 26), getitem___124845, slice_124843)
    
    # Getting the type of 'current' (line 982)
    current_124847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 982, 8), 'current')
    # Getting the type of 'r2cmn' (line 982)
    r2cmn_124848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 982, 17), 'r2cmn')
    slice_124849 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 982, 8), None, r2cmn_124848, None)
    # Storing an element on a container (line 982)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 982, 8), current_124847, (slice_124849, subscript_call_result_124846))
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'jointype' (line 983)
    jointype_124850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 983, 12), 'jointype')
    str_124851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 983, 24), 'str', 'outer')
    # Applying the binary operator '==' (line 983)
    result_eq_124852 = python_operator(stypy.reporting.localization.Localization(__file__, 983, 12), '==', jointype_124850, str_124851)
    
    # Getting the type of 'r2spc' (line 983)
    r2spc_124853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 983, 37), 'r2spc')
    # Applying the binary operator 'and' (line 983)
    result_and_keyword_124854 = python_operator(stypy.reporting.localization.Localization(__file__, 983, 11), 'and', result_eq_124852, r2spc_124853)
    
    # Testing the type of an if condition (line 983)
    if_condition_124855 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 983, 8), result_and_keyword_124854)
    # Assigning a type to the variable 'if_condition_124855' (line 983)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 983, 8), 'if_condition_124855', if_condition_124855)
    # SSA begins for if statement (line 983)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Subscript (line 984):
    
    # Assigning a Subscript to a Subscript (line 984):
    
    # Obtaining the type of the subscript
    # Getting the type of 'r2cmn' (line 984)
    r2cmn_124856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 984, 40), 'r2cmn')
    slice_124857 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 984, 31), r2cmn_124856, None, None)
    # Getting the type of 'selected' (line 984)
    selected_124858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 984, 31), 'selected')
    # Obtaining the member '__getitem__' of a type (line 984)
    getitem___124859 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 984, 31), selected_124858, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 984)
    subscript_call_result_124860 = invoke(stypy.reporting.localization.Localization(__file__, 984, 31), getitem___124859, slice_124857)
    
    # Getting the type of 'current' (line 984)
    current_124861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 984, 12), 'current')
    
    # Getting the type of 'r2spc' (line 984)
    r2spc_124862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 984, 21), 'r2spc')
    # Applying the 'usub' unary operator (line 984)
    result___neg___124863 = python_operator(stypy.reporting.localization.Localization(__file__, 984, 20), 'usub', r2spc_124862)
    
    slice_124864 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 984, 12), result___neg___124863, None, None)
    # Storing an element on a container (line 984)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 984, 12), current_124861, (slice_124864, subscript_call_result_124860))
    # SSA join for if statement (line 983)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to sort(...): (line 986)
    # Processing the call keyword arguments (line 986)
    # Getting the type of 'key' (line 986)
    key_124867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 986, 22), 'key', False)
    keyword_124868 = key_124867
    kwargs_124869 = {'order': keyword_124868}
    # Getting the type of 'output' (line 986)
    output_124865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 986, 4), 'output', False)
    # Obtaining the member 'sort' of a type (line 986)
    sort_124866 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 986, 4), output_124865, 'sort')
    # Calling sort(args, kwargs) (line 986)
    sort_call_result_124870 = invoke(stypy.reporting.localization.Localization(__file__, 986, 4), sort_124866, *[], **kwargs_124869)
    
    
    # Assigning a Call to a Name (line 987):
    
    # Assigning a Call to a Name (line 987):
    
    # Call to dict(...): (line 987)
    # Processing the call keyword arguments (line 987)
    # Getting the type of 'usemask' (line 987)
    usemask_124872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 987, 26), 'usemask', False)
    keyword_124873 = usemask_124872
    # Getting the type of 'asrecarray' (line 987)
    asrecarray_124874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 987, 46), 'asrecarray', False)
    keyword_124875 = asrecarray_124874
    kwargs_124876 = {'usemask': keyword_124873, 'asrecarray': keyword_124875}
    # Getting the type of 'dict' (line 987)
    dict_124871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 987, 13), 'dict', False)
    # Calling dict(args, kwargs) (line 987)
    dict_call_result_124877 = invoke(stypy.reporting.localization.Localization(__file__, 987, 13), dict_124871, *[], **kwargs_124876)
    
    # Assigning a type to the variable 'kwargs' (line 987)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 987, 4), 'kwargs', dict_call_result_124877)
    
    # Call to _fix_output(...): (line 988)
    # Processing the call arguments (line 988)
    
    # Call to _fix_defaults(...): (line 988)
    # Processing the call arguments (line 988)
    # Getting the type of 'output' (line 988)
    output_124880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 988, 37), 'output', False)
    # Getting the type of 'defaults' (line 988)
    defaults_124881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 988, 45), 'defaults', False)
    # Processing the call keyword arguments (line 988)
    kwargs_124882 = {}
    # Getting the type of '_fix_defaults' (line 988)
    _fix_defaults_124879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 988, 23), '_fix_defaults', False)
    # Calling _fix_defaults(args, kwargs) (line 988)
    _fix_defaults_call_result_124883 = invoke(stypy.reporting.localization.Localization(__file__, 988, 23), _fix_defaults_124879, *[output_124880, defaults_124881], **kwargs_124882)
    
    # Processing the call keyword arguments (line 988)
    # Getting the type of 'kwargs' (line 988)
    kwargs_124884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 988, 58), 'kwargs', False)
    kwargs_124885 = {'kwargs_124884': kwargs_124884}
    # Getting the type of '_fix_output' (line 988)
    _fix_output_124878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 988, 11), '_fix_output', False)
    # Calling _fix_output(args, kwargs) (line 988)
    _fix_output_call_result_124886 = invoke(stypy.reporting.localization.Localization(__file__, 988, 11), _fix_output_124878, *[_fix_defaults_call_result_124883], **kwargs_124885)
    
    # Assigning a type to the variable 'stypy_return_type' (line 988)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 988, 4), 'stypy_return_type', _fix_output_call_result_124886)
    
    # ################# End of 'join_by(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'join_by' in the type store
    # Getting the type of 'stypy_return_type' (line 823)
    stypy_return_type_124887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 823, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_124887)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'join_by'
    return stypy_return_type_124887

# Assigning a type to the variable 'join_by' (line 823)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 823, 0), 'join_by', join_by)

@norecursion
def rec_join(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_124888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 991, 35), 'str', 'inner')
    str_124889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 991, 54), 'str', '1')
    str_124890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 991, 69), 'str', '2')
    # Getting the type of 'None' (line 992)
    None_124891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 992, 22), 'None')
    defaults = [str_124888, str_124889, str_124890, None_124891]
    # Create a new context for function 'rec_join'
    module_type_store = module_type_store.open_function_context('rec_join', 991, 0, False)
    
    # Passed parameters checking function
    rec_join.stypy_localization = localization
    rec_join.stypy_type_of_self = None
    rec_join.stypy_type_store = module_type_store
    rec_join.stypy_function_name = 'rec_join'
    rec_join.stypy_param_names_list = ['key', 'r1', 'r2', 'jointype', 'r1postfix', 'r2postfix', 'defaults']
    rec_join.stypy_varargs_param_name = None
    rec_join.stypy_kwargs_param_name = None
    rec_join.stypy_call_defaults = defaults
    rec_join.stypy_call_varargs = varargs
    rec_join.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'rec_join', ['key', 'r1', 'r2', 'jointype', 'r1postfix', 'r2postfix', 'defaults'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'rec_join', localization, ['key', 'r1', 'r2', 'jointype', 'r1postfix', 'r2postfix', 'defaults'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'rec_join(...)' code ##################

    str_124892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1000, (-1)), 'str', '\n    Join arrays `r1` and `r2` on keys.\n    Alternative to join_by, that always returns a np.recarray.\n\n    See Also\n    --------\n    join_by : equivalent function\n    ')
    
    # Assigning a Call to a Name (line 1001):
    
    # Assigning a Call to a Name (line 1001):
    
    # Call to dict(...): (line 1001)
    # Processing the call keyword arguments (line 1001)
    # Getting the type of 'jointype' (line 1001)
    jointype_124894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1001, 27), 'jointype', False)
    keyword_124895 = jointype_124894
    # Getting the type of 'r1postfix' (line 1001)
    r1postfix_124896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1001, 47), 'r1postfix', False)
    keyword_124897 = r1postfix_124896
    # Getting the type of 'r2postfix' (line 1001)
    r2postfix_124898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1001, 68), 'r2postfix', False)
    keyword_124899 = r2postfix_124898
    # Getting the type of 'defaults' (line 1002)
    defaults_124900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1002, 27), 'defaults', False)
    keyword_124901 = defaults_124900
    # Getting the type of 'False' (line 1002)
    False_124902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1002, 45), 'False', False)
    keyword_124903 = False_124902
    # Getting the type of 'True' (line 1002)
    True_124904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1002, 63), 'True', False)
    keyword_124905 = True_124904
    kwargs_124906 = {'asrecarray': keyword_124905, 'r1postfix': keyword_124897, 'jointype': keyword_124895, 'defaults': keyword_124901, 'r2postfix': keyword_124899, 'usemask': keyword_124903}
    # Getting the type of 'dict' (line 1001)
    dict_124893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1001, 13), 'dict', False)
    # Calling dict(args, kwargs) (line 1001)
    dict_call_result_124907 = invoke(stypy.reporting.localization.Localization(__file__, 1001, 13), dict_124893, *[], **kwargs_124906)
    
    # Assigning a type to the variable 'kwargs' (line 1001)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1001, 4), 'kwargs', dict_call_result_124907)
    
    # Call to join_by(...): (line 1003)
    # Processing the call arguments (line 1003)
    # Getting the type of 'key' (line 1003)
    key_124909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1003, 19), 'key', False)
    # Getting the type of 'r1' (line 1003)
    r1_124910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1003, 24), 'r1', False)
    # Getting the type of 'r2' (line 1003)
    r2_124911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1003, 28), 'r2', False)
    # Processing the call keyword arguments (line 1003)
    # Getting the type of 'kwargs' (line 1003)
    kwargs_124912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1003, 34), 'kwargs', False)
    kwargs_124913 = {'kwargs_124912': kwargs_124912}
    # Getting the type of 'join_by' (line 1003)
    join_by_124908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1003, 11), 'join_by', False)
    # Calling join_by(args, kwargs) (line 1003)
    join_by_call_result_124914 = invoke(stypy.reporting.localization.Localization(__file__, 1003, 11), join_by_124908, *[key_124909, r1_124910, r2_124911], **kwargs_124913)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1003)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1003, 4), 'stypy_return_type', join_by_call_result_124914)
    
    # ################# End of 'rec_join(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'rec_join' in the type store
    # Getting the type of 'stypy_return_type' (line 991)
    stypy_return_type_124915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 991, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_124915)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'rec_join'
    return stypy_return_type_124915

# Assigning a type to the variable 'rec_join' (line 991)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 991, 0), 'rec_join', rec_join)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
