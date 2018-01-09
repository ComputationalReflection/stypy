
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Record Arrays
3: =============
4: Record arrays expose the fields of structured arrays as properties.
5: 
6: Most commonly, ndarrays contain elements of a single type, e.g. floats,
7: integers, bools etc.  However, it is possible for elements to be combinations
8: of these using structured types, such as::
9: 
10:   >>> a = np.array([(1, 2.0), (1, 2.0)], dtype=[('x', int), ('y', float)])
11:   >>> a
12:   array([(1, 2.0), (1, 2.0)],
13:         dtype=[('x', '<i4'), ('y', '<f8')])
14: 
15: Here, each element consists of two fields: x (and int), and y (a float).
16: This is known as a structured array.  The different fields are analogous
17: to columns in a spread-sheet.  The different fields can be accessed as
18: one would a dictionary::
19: 
20:   >>> a['x']
21:   array([1, 1])
22: 
23:   >>> a['y']
24:   array([ 2.,  2.])
25: 
26: Record arrays allow us to access fields as properties::
27: 
28:   >>> ar = np.rec.array(a)
29: 
30:   >>> ar.x
31:   array([1, 1])
32: 
33:   >>> ar.y
34:   array([ 2.,  2.])
35: 
36: '''
37: from __future__ import division, absolute_import, print_function
38: 
39: import sys
40: import os
41: 
42: from . import numeric as sb
43: from . import numerictypes as nt
44: from numpy.compat import isfileobj, bytes, long
45: 
46: # All of the functions allow formats to be a dtype
47: __all__ = ['record', 'recarray', 'format_parser']
48: 
49: 
50: ndarray = sb.ndarray
51: 
52: _byteorderconv = {'b':'>',
53:                   'l':'<',
54:                   'n':'=',
55:                   'B':'>',
56:                   'L':'<',
57:                   'N':'=',
58:                   'S':'s',
59:                   's':'s',
60:                   '>':'>',
61:                   '<':'<',
62:                   '=':'=',
63:                   '|':'|',
64:                   'I':'|',
65:                   'i':'|'}
66: 
67: # formats regular expression
68: # allows multidimension spec with a tuple syntax in front
69: # of the letter code '(2,3)f4' and ' (  2 ,  3  )  f4  '
70: # are equally allowed
71: 
72: numfmt = nt.typeDict
73: 
74: def find_duplicate(list):
75:     '''Find duplication in a list, return a list of duplicated elements'''
76:     dup = []
77:     for i in range(len(list)):
78:         if (list[i] in list[i + 1:]):
79:             if (list[i] not in dup):
80:                 dup.append(list[i])
81:     return dup
82: 
83: class format_parser:
84:     '''
85:     Class to convert formats, names, titles description to a dtype.
86: 
87:     After constructing the format_parser object, the dtype attribute is
88:     the converted data-type:
89:     ``dtype = format_parser(formats, names, titles).dtype``
90: 
91:     Attributes
92:     ----------
93:     dtype : dtype
94:         The converted data-type.
95: 
96:     Parameters
97:     ----------
98:     formats : str or list of str
99:         The format description, either specified as a string with
100:         comma-separated format descriptions in the form ``'f8, i4, a5'``, or
101:         a list of format description strings  in the form
102:         ``['f8', 'i4', 'a5']``.
103:     names : str or list/tuple of str
104:         The field names, either specified as a comma-separated string in the
105:         form ``'col1, col2, col3'``, or as a list or tuple of strings in the
106:         form ``['col1', 'col2', 'col3']``.
107:         An empty list can be used, in that case default field names
108:         ('f0', 'f1', ...) are used.
109:     titles : sequence
110:         Sequence of title strings. An empty list can be used to leave titles
111:         out.
112:     aligned : bool, optional
113:         If True, align the fields by padding as the C-compiler would.
114:         Default is False.
115:     byteorder : str, optional
116:         If specified, all the fields will be changed to the
117:         provided byte-order.  Otherwise, the default byte-order is
118:         used. For all available string specifiers, see `dtype.newbyteorder`.
119: 
120:     See Also
121:     --------
122:     dtype, typename, sctype2char
123: 
124:     Examples
125:     --------
126:     >>> np.format_parser(['f8', 'i4', 'a5'], ['col1', 'col2', 'col3'],
127:     ...                  ['T1', 'T2', 'T3']).dtype
128:     dtype([(('T1', 'col1'), '<f8'), (('T2', 'col2'), '<i4'),
129:            (('T3', 'col3'), '|S5')])
130: 
131:     `names` and/or `titles` can be empty lists. If `titles` is an empty list,
132:     titles will simply not appear. If `names` is empty, default field names
133:     will be used.
134: 
135:     >>> np.format_parser(['f8', 'i4', 'a5'], ['col1', 'col2', 'col3'],
136:     ...                  []).dtype
137:     dtype([('col1', '<f8'), ('col2', '<i4'), ('col3', '|S5')])
138:     >>> np.format_parser(['f8', 'i4', 'a5'], [], []).dtype
139:     dtype([('f0', '<f8'), ('f1', '<i4'), ('f2', '|S5')])
140: 
141:     '''
142: 
143:     def __init__(self, formats, names, titles, aligned=False, byteorder=None):
144:         self._parseFormats(formats, aligned)
145:         self._setfieldnames(names, titles)
146:         self._createdescr(byteorder)
147:         self.dtype = self._descr
148: 
149:     def _parseFormats(self, formats, aligned=0):
150:         ''' Parse the field formats '''
151: 
152:         if formats is None:
153:             raise ValueError("Need formats argument")
154:         if isinstance(formats, list):
155:             if len(formats) < 2:
156:                 formats.append('')
157:             formats = ','.join(formats)
158:         dtype = sb.dtype(formats, aligned)
159:         fields = dtype.fields
160:         if fields is None:
161:             dtype = sb.dtype([('f1', dtype)], aligned)
162:             fields = dtype.fields
163:         keys = dtype.names
164:         self._f_formats = [fields[key][0] for key in keys]
165:         self._offsets = [fields[key][1] for key in keys]
166:         self._nfields = len(keys)
167: 
168:     def _setfieldnames(self, names, titles):
169:         '''convert input field names into a list and assign to the _names
170:         attribute '''
171: 
172:         if (names):
173:             if (type(names) in [list, tuple]):
174:                 pass
175:             elif isinstance(names, str):
176:                 names = names.split(',')
177:             else:
178:                 raise NameError("illegal input names %s" % repr(names))
179: 
180:             self._names = [n.strip() for n in names[:self._nfields]]
181:         else:
182:             self._names = []
183: 
184:         # if the names are not specified, they will be assigned as
185:         #  "f0, f1, f2,..."
186:         # if not enough names are specified, they will be assigned as "f[n],
187:         # f[n+1],..." etc. where n is the number of specified names..."
188:         self._names += ['f%d' % i for i in range(len(self._names),
189:                                                  self._nfields)]
190:         # check for redundant names
191:         _dup = find_duplicate(self._names)
192:         if _dup:
193:             raise ValueError("Duplicate field names: %s" % _dup)
194: 
195:         if (titles):
196:             self._titles = [n.strip() for n in titles[:self._nfields]]
197:         else:
198:             self._titles = []
199:             titles = []
200: 
201:         if (self._nfields > len(titles)):
202:             self._titles += [None] * (self._nfields - len(titles))
203: 
204:     def _createdescr(self, byteorder):
205:         descr = sb.dtype({'names':self._names,
206:                           'formats':self._f_formats,
207:                           'offsets':self._offsets,
208:                           'titles':self._titles})
209:         if (byteorder is not None):
210:             byteorder = _byteorderconv[byteorder[0]]
211:             descr = descr.newbyteorder(byteorder)
212: 
213:         self._descr = descr
214: 
215: class record(nt.void):
216:     '''A data-type scalar that allows field access as attribute lookup.
217:     '''
218: 
219:     # manually set name and module so that this class's type shows up
220:     # as numpy.record when printed
221:     __name__ = 'record'
222:     __module__ = 'numpy'
223: 
224:     def __repr__(self):
225:         return self.__str__()
226: 
227:     def __str__(self):
228:         return str(self.item())
229: 
230:     def __getattribute__(self, attr):
231:         if attr in ['setfield', 'getfield', 'dtype']:
232:             return nt.void.__getattribute__(self, attr)
233:         try:
234:             return nt.void.__getattribute__(self, attr)
235:         except AttributeError:
236:             pass
237:         fielddict = nt.void.__getattribute__(self, 'dtype').fields
238:         res = fielddict.get(attr, None)
239:         if res:
240:             obj = self.getfield(*res[:2])
241:             # if it has fields return a record,
242:             # otherwise return the object
243:             try:
244:                 dt = obj.dtype
245:             except AttributeError:
246:                 #happens if field is Object type
247:                 return obj
248:             if dt.fields:
249:                 return obj.view((self.__class__, obj.dtype.fields))
250:             return obj
251:         else:
252:             raise AttributeError("'record' object has no "
253:                     "attribute '%s'" % attr)
254: 
255:     def __setattr__(self, attr, val):
256:         if attr in ['setfield', 'getfield', 'dtype']:
257:             raise AttributeError("Cannot set '%s' attribute" % attr)
258:         fielddict = nt.void.__getattribute__(self, 'dtype').fields
259:         res = fielddict.get(attr, None)
260:         if res:
261:             return self.setfield(val, *res[:2])
262:         else:
263:             if getattr(self, attr, None):
264:                 return nt.void.__setattr__(self, attr, val)
265:             else:
266:                 raise AttributeError("'record' object has no "
267:                         "attribute '%s'" % attr)
268: 
269:     def __getitem__(self, indx):
270:         obj = nt.void.__getitem__(self, indx)
271: 
272:         # copy behavior of record.__getattribute__,
273:         if isinstance(obj, nt.void) and obj.dtype.fields:
274:             return obj.view((self.__class__, obj.dtype.fields))
275:         else:
276:             # return a single element
277:             return obj
278: 
279:     def pprint(self):
280:         '''Pretty-print all fields.'''
281:         # pretty-print all fields
282:         names = self.dtype.names
283:         maxlen = max(len(name) for name in names)
284:         rows = []
285:         fmt = '%% %ds: %%s' % maxlen
286:         for name in names:
287:             rows.append(fmt % (name, getattr(self, name)))
288:         return "\n".join(rows)
289: 
290: # The recarray is almost identical to a standard array (which supports
291: #   named fields already)  The biggest difference is that it can use
292: #   attribute-lookup to find the fields and it is constructed using
293: #   a record.
294: 
295: # If byteorder is given it forces a particular byteorder on all
296: #  the fields (and any subfields)
297: 
298: class recarray(ndarray):
299:     '''Construct an ndarray that allows field access using attributes.
300: 
301:     Arrays may have a data-types containing fields, analogous
302:     to columns in a spread sheet.  An example is ``[(x, int), (y, float)]``,
303:     where each entry in the array is a pair of ``(int, float)``.  Normally,
304:     these attributes are accessed using dictionary lookups such as ``arr['x']``
305:     and ``arr['y']``.  Record arrays allow the fields to be accessed as members
306:     of the array, using ``arr.x`` and ``arr.y``.
307: 
308:     Parameters
309:     ----------
310:     shape : tuple
311:         Shape of output array.
312:     dtype : data-type, optional
313:         The desired data-type.  By default, the data-type is determined
314:         from `formats`, `names`, `titles`, `aligned` and `byteorder`.
315:     formats : list of data-types, optional
316:         A list containing the data-types for the different columns, e.g.
317:         ``['i4', 'f8', 'i4']``.  `formats` does *not* support the new
318:         convention of using types directly, i.e. ``(int, float, int)``.
319:         Note that `formats` must be a list, not a tuple.
320:         Given that `formats` is somewhat limited, we recommend specifying
321:         `dtype` instead.
322:     names : tuple of str, optional
323:         The name of each column, e.g. ``('x', 'y', 'z')``.
324:     buf : buffer, optional
325:         By default, a new array is created of the given shape and data-type.
326:         If `buf` is specified and is an object exposing the buffer interface,
327:         the array will use the memory from the existing buffer.  In this case,
328:         the `offset` and `strides` keywords are available.
329: 
330:     Other Parameters
331:     ----------------
332:     titles : tuple of str, optional
333:         Aliases for column names.  For example, if `names` were
334:         ``('x', 'y', 'z')`` and `titles` is
335:         ``('x_coordinate', 'y_coordinate', 'z_coordinate')``, then
336:         ``arr['x']`` is equivalent to both ``arr.x`` and ``arr.x_coordinate``.
337:     byteorder : {'<', '>', '='}, optional
338:         Byte-order for all fields.
339:     aligned : bool, optional
340:         Align the fields in memory as the C-compiler would.
341:     strides : tuple of ints, optional
342:         Buffer (`buf`) is interpreted according to these strides (strides
343:         define how many bytes each array element, row, column, etc.
344:         occupy in memory).
345:     offset : int, optional
346:         Start reading buffer (`buf`) from this offset onwards.
347:     order : {'C', 'F'}, optional
348:         Row-major (C-style) or column-major (Fortran-style) order.
349: 
350:     Returns
351:     -------
352:     rec : recarray
353:         Empty array of the given shape and type.
354: 
355:     See Also
356:     --------
357:     rec.fromrecords : Construct a record array from data.
358:     record : fundamental data-type for `recarray`.
359:     format_parser : determine a data-type from formats, names, titles.
360: 
361:     Notes
362:     -----
363:     This constructor can be compared to ``empty``: it creates a new record
364:     array but does not fill it with data.  To create a record array from data,
365:     use one of the following methods:
366: 
367:     1. Create a standard ndarray and convert it to a record array,
368:        using ``arr.view(np.recarray)``
369:     2. Use the `buf` keyword.
370:     3. Use `np.rec.fromrecords`.
371: 
372:     Examples
373:     --------
374:     Create an array with two fields, ``x`` and ``y``:
375: 
376:     >>> x = np.array([(1.0, 2), (3.0, 4)], dtype=[('x', float), ('y', int)])
377:     >>> x
378:     array([(1.0, 2), (3.0, 4)],
379:           dtype=[('x', '<f8'), ('y', '<i4')])
380: 
381:     >>> x['x']
382:     array([ 1.,  3.])
383: 
384:     View the array as a record array:
385: 
386:     >>> x = x.view(np.recarray)
387: 
388:     >>> x.x
389:     array([ 1.,  3.])
390: 
391:     >>> x.y
392:     array([2, 4])
393: 
394:     Create a new, empty record array:
395: 
396:     >>> np.recarray((2,),
397:     ... dtype=[('x', int), ('y', float), ('z', int)]) #doctest: +SKIP
398:     rec.array([(-1073741821, 1.2249118382103472e-301, 24547520),
399:            (3471280, 1.2134086255804012e-316, 0)],
400:           dtype=[('x', '<i4'), ('y', '<f8'), ('z', '<i4')])
401: 
402:     '''
403: 
404:     # manually set name and module so that this class's type shows
405:     # up as "numpy.recarray" when printed
406:     __name__ = 'recarray'
407:     __module__ = 'numpy'
408: 
409:     def __new__(subtype, shape, dtype=None, buf=None, offset=0, strides=None,
410:                 formats=None, names=None, titles=None,
411:                 byteorder=None, aligned=False, order='C'):
412: 
413:         if dtype is not None:
414:             descr = sb.dtype(dtype)
415:         else:
416:             descr = format_parser(formats, names, titles, aligned, byteorder)._descr
417: 
418:         if buf is None:
419:             self = ndarray.__new__(subtype, shape, (record, descr), order=order)
420:         else:
421:             self = ndarray.__new__(subtype, shape, (record, descr),
422:                                       buffer=buf, offset=offset,
423:                                       strides=strides, order=order)
424:         return self
425: 
426:     def __array_finalize__(self, obj):
427:         if self.dtype.type is not record:
428:             # if self.dtype is not np.record, invoke __setattr__ which will
429:             # convert it to a record if it is a void dtype.
430:             self.dtype = self.dtype
431: 
432:     def __getattribute__(self, attr):
433:         # See if ndarray has this attr, and return it if so. (note that this
434:         # means a field with the same name as an ndarray attr cannot be
435:         # accessed by attribute).
436:         try:
437:             return object.__getattribute__(self, attr)
438:         except AttributeError:  # attr must be a fieldname
439:             pass
440: 
441:         # look for a field with this name
442:         fielddict = ndarray.__getattribute__(self, 'dtype').fields
443:         try:
444:             res = fielddict[attr][:2]
445:         except (TypeError, KeyError):
446:             raise AttributeError("recarray has no attribute %s" % attr)
447:         obj = self.getfield(*res)
448: 
449:         # At this point obj will always be a recarray, since (see
450:         # PyArray_GetField) the type of obj is inherited. Next, if obj.dtype is
451:         # non-structured, convert it to an ndarray. Then if obj is structured
452:         # with void type convert it to the same dtype.type (eg to preserve
453:         # numpy.record type if present), since nested structured fields do not
454:         # inherit type. Don't do this for non-void structures though.
455:         if obj.dtype.fields:
456:             if issubclass(obj.dtype.type, nt.void):
457:                 return obj.view(dtype=(self.dtype.type, obj.dtype))
458:             return obj
459:         else:
460:             return obj.view(ndarray)
461: 
462:     # Save the dictionary.
463:     # If the attr is a field name and not in the saved dictionary
464:     # Undo any "setting" of the attribute and do a setfield
465:     # Thus, you can't create attributes on-the-fly that are field names.
466:     def __setattr__(self, attr, val):
467: 
468:         # Automatically convert (void) structured types to records
469:         # (but not non-void structures, subarrays, or non-structured voids)
470:         if attr == 'dtype' and issubclass(val.type, nt.void) and val.fields:
471:             val = sb.dtype((record, val))
472: 
473:         newattr = attr not in self.__dict__
474:         try:
475:             ret = object.__setattr__(self, attr, val)
476:         except:
477:             fielddict = ndarray.__getattribute__(self, 'dtype').fields or {}
478:             if attr not in fielddict:
479:                 exctype, value = sys.exc_info()[:2]
480:                 raise exctype(value)
481:         else:
482:             fielddict = ndarray.__getattribute__(self, 'dtype').fields or {}
483:             if attr not in fielddict:
484:                 return ret
485:             if newattr:
486:                 # We just added this one or this setattr worked on an
487:                 # internal attribute.
488:                 try:
489:                     object.__delattr__(self, attr)
490:                 except:
491:                     return ret
492:         try:
493:             res = fielddict[attr][:2]
494:         except (TypeError, KeyError):
495:             raise AttributeError("record array has no attribute %s" % attr)
496:         return self.setfield(val, *res)
497: 
498:     def __getitem__(self, indx):
499:         obj = super(recarray, self).__getitem__(indx)
500: 
501:         # copy behavior of getattr, except that here
502:         # we might also be returning a single element
503:         if isinstance(obj, ndarray):
504:             if obj.dtype.fields:
505:                 obj = obj.view(type(self))
506:                 if issubclass(obj.dtype.type, nt.void):
507:                     return obj.view(dtype=(self.dtype.type, obj.dtype))
508:                 return obj
509:             else:
510:                 return obj.view(type=ndarray)
511:         else:
512:             # return a single element
513:             return obj
514: 
515:     def __repr__(self):
516:         # get data/shape string. logic taken from numeric.array_repr
517:         if self.size > 0 or self.shape == (0,):
518:             lst = sb.array2string(self, separator=', ')
519:         else:
520:             # show zero-length shape unless it is (0,)
521:             lst = "[], shape=%s" % (repr(self.shape),)
522: 
523:         if (self.dtype.type is record
524:                 or (not issubclass(self.dtype.type, nt.void))):
525:             # If this is a full record array (has numpy.record dtype),
526:             # or if it has a scalar (non-void) dtype with no records,
527:             # represent it using the rec.array function. Since rec.array
528:             # converts dtype to a numpy.record for us, convert back
529:             # to non-record before printing
530:             plain_dtype = self.dtype
531:             if plain_dtype.type is record:
532:                 plain_dtype = sb.dtype((nt.void, plain_dtype))
533:             lf = '\n'+' '*len("rec.array(")
534:             return ('rec.array(%s, %sdtype=%s)' %
535:                           (lst, lf, plain_dtype))
536:         else:
537:             # otherwise represent it using np.array plus a view
538:             # This should only happen if the user is playing
539:             # strange games with dtypes.
540:             lf = '\n'+' '*len("array(")
541:             return ('array(%s, %sdtype=%s).view(numpy.recarray)' %
542:                           (lst, lf, str(self.dtype)))
543: 
544:     def field(self, attr, val=None):
545:         if isinstance(attr, int):
546:             names = ndarray.__getattribute__(self, 'dtype').names
547:             attr = names[attr]
548: 
549:         fielddict = ndarray.__getattribute__(self, 'dtype').fields
550: 
551:         res = fielddict[attr][:2]
552: 
553:         if val is None:
554:             obj = self.getfield(*res)
555:             if obj.dtype.fields:
556:                 return obj
557:             return obj.view(ndarray)
558:         else:
559:             return self.setfield(val, *res)
560: 
561: 
562: def fromarrays(arrayList, dtype=None, shape=None, formats=None,
563:                names=None, titles=None, aligned=False, byteorder=None):
564:     ''' create a record array from a (flat) list of arrays
565: 
566:     >>> x1=np.array([1,2,3,4])
567:     >>> x2=np.array(['a','dd','xyz','12'])
568:     >>> x3=np.array([1.1,2,3,4])
569:     >>> r = np.core.records.fromarrays([x1,x2,x3],names='a,b,c')
570:     >>> print(r[1])
571:     (2, 'dd', 2.0)
572:     >>> x1[1]=34
573:     >>> r.a
574:     array([1, 2, 3, 4])
575:     '''
576: 
577:     arrayList = [sb.asarray(x) for x in arrayList]
578: 
579:     if shape is None or shape == 0:
580:         shape = arrayList[0].shape
581: 
582:     if isinstance(shape, int):
583:         shape = (shape,)
584: 
585:     if formats is None and dtype is None:
586:         # go through each object in the list to see if it is an ndarray
587:         # and determine the formats.
588:         formats = []
589:         for obj in arrayList:
590:             if not isinstance(obj, ndarray):
591:                 raise ValueError("item in the array list must be an ndarray.")
592:             formats.append(obj.dtype.str)
593:         formats = ','.join(formats)
594: 
595:     if dtype is not None:
596:         descr = sb.dtype(dtype)
597:         _names = descr.names
598:     else:
599:         parsed = format_parser(formats, names, titles, aligned, byteorder)
600:         _names = parsed._names
601:         descr = parsed._descr
602: 
603:     # Determine shape from data-type.
604:     if len(descr) != len(arrayList):
605:         raise ValueError("mismatch between the number of fields "
606:                 "and the number of arrays")
607: 
608:     d0 = descr[0].shape
609:     nn = len(d0)
610:     if nn > 0:
611:         shape = shape[:-nn]
612: 
613:     for k, obj in enumerate(arrayList):
614:         nn = len(descr[k].shape)
615:         testshape = obj.shape[:len(obj.shape) - nn]
616:         if testshape != shape:
617:             raise ValueError("array-shape mismatch in array %d" % k)
618: 
619:     _array = recarray(shape, descr)
620: 
621:     # populate the record array (makes a copy)
622:     for i in range(len(arrayList)):
623:         _array[_names[i]] = arrayList[i]
624: 
625:     return _array
626: 
627: # shape must be 1-d if you use list of lists...
628: def fromrecords(recList, dtype=None, shape=None, formats=None, names=None,
629:                 titles=None, aligned=False, byteorder=None):
630:     ''' create a recarray from a list of records in text form
631: 
632:         The data in the same field can be heterogeneous, they will be promoted
633:         to the highest data type.  This method is intended for creating
634:         smaller record arrays.  If used to create large array without formats
635:         defined
636: 
637:         r=fromrecords([(2,3.,'abc')]*100000)
638: 
639:         it can be slow.
640: 
641:         If formats is None, then this will auto-detect formats. Use list of
642:         tuples rather than list of lists for faster processing.
643: 
644:     >>> r=np.core.records.fromrecords([(456,'dbe',1.2),(2,'de',1.3)],
645:     ... names='col1,col2,col3')
646:     >>> print(r[0])
647:     (456, 'dbe', 1.2)
648:     >>> r.col1
649:     array([456,   2])
650:     >>> r.col2
651:     array(['dbe', 'de'],
652:           dtype='|S3')
653:     >>> import pickle
654:     >>> print(pickle.loads(pickle.dumps(r)))
655:     [(456, 'dbe', 1.2) (2, 'de', 1.3)]
656:     '''
657: 
658:     nfields = len(recList[0])
659:     if formats is None and dtype is None:  # slower
660:         obj = sb.array(recList, dtype=object)
661:         arrlist = [sb.array(obj[..., i].tolist()) for i in range(nfields)]
662:         return fromarrays(arrlist, formats=formats, shape=shape, names=names,
663:                           titles=titles, aligned=aligned, byteorder=byteorder)
664: 
665:     if dtype is not None:
666:         descr = sb.dtype((record, dtype))
667:     else:
668:         descr = format_parser(formats, names, titles, aligned, byteorder)._descr
669: 
670:     try:
671:         retval = sb.array(recList, dtype=descr)
672:     except TypeError:  # list of lists instead of list of tuples
673:         if (shape is None or shape == 0):
674:             shape = len(recList)
675:         if isinstance(shape, (int, long)):
676:             shape = (shape,)
677:         if len(shape) > 1:
678:             raise ValueError("Can only deal with 1-d array.")
679:         _array = recarray(shape, descr)
680:         for k in range(_array.size):
681:             _array[k] = tuple(recList[k])
682:         return _array
683:     else:
684:         if shape is not None and retval.shape != shape:
685:             retval.shape = shape
686: 
687:     res = retval.view(recarray)
688: 
689:     return res
690: 
691: 
692: def fromstring(datastring, dtype=None, shape=None, offset=0, formats=None,
693:                names=None, titles=None, aligned=False, byteorder=None):
694:     ''' create a (read-only) record array from binary data contained in
695:     a string'''
696: 
697:     if dtype is None and formats is None:
698:         raise ValueError("Must have dtype= or formats=")
699: 
700:     if dtype is not None:
701:         descr = sb.dtype(dtype)
702:     else:
703:         descr = format_parser(formats, names, titles, aligned, byteorder)._descr
704: 
705:     itemsize = descr.itemsize
706:     if (shape is None or shape == 0 or shape == -1):
707:         shape = (len(datastring) - offset) / itemsize
708: 
709:     _array = recarray(shape, descr, buf=datastring, offset=offset)
710:     return _array
711: 
712: def get_remaining_size(fd):
713:     try:
714:         fn = fd.fileno()
715:     except AttributeError:
716:         return os.path.getsize(fd.name) - fd.tell()
717:     st = os.fstat(fn)
718:     size = st.st_size - fd.tell()
719:     return size
720: 
721: def fromfile(fd, dtype=None, shape=None, offset=0, formats=None,
722:              names=None, titles=None, aligned=False, byteorder=None):
723:     '''Create an array from binary file data
724: 
725:     If file is a string then that file is opened, else it is assumed
726:     to be a file object.
727: 
728:     >>> from tempfile import TemporaryFile
729:     >>> a = np.empty(10,dtype='f8,i4,a5')
730:     >>> a[5] = (0.5,10,'abcde')
731:     >>>
732:     >>> fd=TemporaryFile()
733:     >>> a = a.newbyteorder('<')
734:     >>> a.tofile(fd)
735:     >>>
736:     >>> fd.seek(0)
737:     >>> r=np.core.records.fromfile(fd, formats='f8,i4,a5', shape=10,
738:     ... byteorder='<')
739:     >>> print(r[5])
740:     (0.5, 10, 'abcde')
741:     >>> r.shape
742:     (10,)
743:     '''
744: 
745:     if (shape is None or shape == 0):
746:         shape = (-1,)
747:     elif isinstance(shape, (int, long)):
748:         shape = (shape,)
749: 
750:     name = 0
751:     if isinstance(fd, str):
752:         name = 1
753:         fd = open(fd, 'rb')
754:     if (offset > 0):
755:         fd.seek(offset, 1)
756:     size = get_remaining_size(fd)
757: 
758:     if dtype is not None:
759:         descr = sb.dtype(dtype)
760:     else:
761:         descr = format_parser(formats, names, titles, aligned, byteorder)._descr
762: 
763:     itemsize = descr.itemsize
764: 
765:     shapeprod = sb.array(shape).prod()
766:     shapesize = shapeprod * itemsize
767:     if shapesize < 0:
768:         shape = list(shape)
769:         shape[shape.index(-1)] = size / -shapesize
770:         shape = tuple(shape)
771:         shapeprod = sb.array(shape).prod()
772: 
773:     nbytes = shapeprod * itemsize
774: 
775:     if nbytes > size:
776:         raise ValueError(
777:                 "Not enough bytes left in file for specified shape and type")
778: 
779:     # create the array
780:     _array = recarray(shape, descr)
781:     nbytesread = fd.readinto(_array.data)
782:     if nbytesread != nbytes:
783:         raise IOError("Didn't read as many bytes as expected")
784:     if name:
785:         fd.close()
786: 
787:     return _array
788: 
789: def array(obj, dtype=None, shape=None, offset=0, strides=None, formats=None,
790:           names=None, titles=None, aligned=False, byteorder=None, copy=True):
791:     '''Construct a record array from a wide-variety of objects.
792:     '''
793: 
794:     if ((isinstance(obj, (type(None), str)) or isfileobj(obj)) and
795:            (formats is None) and (dtype is None)):
796:         raise ValueError("Must define formats (or dtype) if object is "
797:                          "None, string, or an open file")
798: 
799:     kwds = {}
800:     if dtype is not None:
801:         dtype = sb.dtype(dtype)
802:     elif formats is not None:
803:         dtype = format_parser(formats, names, titles,
804:                               aligned, byteorder)._descr
805:     else:
806:         kwds = {'formats': formats,
807:                 'names': names,
808:                 'titles': titles,
809:                 'aligned': aligned,
810:                 'byteorder': byteorder
811:                 }
812: 
813:     if obj is None:
814:         if shape is None:
815:             raise ValueError("Must define a shape if obj is None")
816:         return recarray(shape, dtype, buf=obj, offset=offset, strides=strides)
817: 
818:     elif isinstance(obj, bytes):
819:         return fromstring(obj, dtype, shape=shape, offset=offset, **kwds)
820: 
821:     elif isinstance(obj, (list, tuple)):
822:         if isinstance(obj[0], (tuple, list)):
823:             return fromrecords(obj, dtype=dtype, shape=shape, **kwds)
824:         else:
825:             return fromarrays(obj, dtype=dtype, shape=shape, **kwds)
826: 
827:     elif isinstance(obj, recarray):
828:         if dtype is not None and (obj.dtype != dtype):
829:             new = obj.view(dtype)
830:         else:
831:             new = obj
832:         if copy:
833:             new = new.copy()
834:         return new
835: 
836:     elif isfileobj(obj):
837:         return fromfile(obj, dtype=dtype, shape=shape, offset=offset)
838: 
839:     elif isinstance(obj, ndarray):
840:         if dtype is not None and (obj.dtype != dtype):
841:             new = obj.view(dtype)
842:         else:
843:             new = obj
844:         if copy:
845:             new = new.copy()
846:         return new.view(recarray)
847: 
848:     else:
849:         interface = getattr(obj, "__array_interface__", None)
850:         if interface is None or not isinstance(interface, dict):
851:             raise ValueError("Unknown input type")
852:         obj = sb.array(obj)
853:         if dtype is not None and (obj.dtype != dtype):
854:             obj = obj.view(dtype)
855:         return obj.view(recarray)
856: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_12232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, (-1)), 'str', "\nRecord Arrays\n=============\nRecord arrays expose the fields of structured arrays as properties.\n\nMost commonly, ndarrays contain elements of a single type, e.g. floats,\nintegers, bools etc.  However, it is possible for elements to be combinations\nof these using structured types, such as::\n\n  >>> a = np.array([(1, 2.0), (1, 2.0)], dtype=[('x', int), ('y', float)])\n  >>> a\n  array([(1, 2.0), (1, 2.0)],\n        dtype=[('x', '<i4'), ('y', '<f8')])\n\nHere, each element consists of two fields: x (and int), and y (a float).\nThis is known as a structured array.  The different fields are analogous\nto columns in a spread-sheet.  The different fields can be accessed as\none would a dictionary::\n\n  >>> a['x']\n  array([1, 1])\n\n  >>> a['y']\n  array([ 2.,  2.])\n\nRecord arrays allow us to access fields as properties::\n\n  >>> ar = np.rec.array(a)\n\n  >>> ar.x\n  array([1, 1])\n\n  >>> ar.y\n  array([ 2.,  2.])\n\n")
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 39, 0))

# 'import sys' statement (line 39)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 39, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 40, 0))

# 'import os' statement (line 40)
import os

import_module(stypy.reporting.localization.Localization(__file__, 40, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 42, 0))

# 'from numpy.core import sb' statement (line 42)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/core/')
import_12233 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 42, 0), 'numpy.core')

if (type(import_12233) is not StypyTypeError):

    if (import_12233 != 'pyd_module'):
        __import__(import_12233)
        sys_modules_12234 = sys.modules[import_12233]
        import_from_module(stypy.reporting.localization.Localization(__file__, 42, 0), 'numpy.core', sys_modules_12234.module_type_store, module_type_store, ['numeric'])
        nest_module(stypy.reporting.localization.Localization(__file__, 42, 0), __file__, sys_modules_12234, sys_modules_12234.module_type_store, module_type_store)
    else:
        from numpy.core import numeric as sb

        import_from_module(stypy.reporting.localization.Localization(__file__, 42, 0), 'numpy.core', None, module_type_store, ['numeric'], [sb])

else:
    # Assigning a type to the variable 'numpy.core' (line 42)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 0), 'numpy.core', import_12233)

# Adding an alias
module_type_store.add_alias('sb', 'numeric')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/core/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 43, 0))

# 'from numpy.core import nt' statement (line 43)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/core/')
import_12235 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 43, 0), 'numpy.core')

if (type(import_12235) is not StypyTypeError):

    if (import_12235 != 'pyd_module'):
        __import__(import_12235)
        sys_modules_12236 = sys.modules[import_12235]
        import_from_module(stypy.reporting.localization.Localization(__file__, 43, 0), 'numpy.core', sys_modules_12236.module_type_store, module_type_store, ['numerictypes'])
        nest_module(stypy.reporting.localization.Localization(__file__, 43, 0), __file__, sys_modules_12236, sys_modules_12236.module_type_store, module_type_store)
    else:
        from numpy.core import numerictypes as nt

        import_from_module(stypy.reporting.localization.Localization(__file__, 43, 0), 'numpy.core', None, module_type_store, ['numerictypes'], [nt])

else:
    # Assigning a type to the variable 'numpy.core' (line 43)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 0), 'numpy.core', import_12235)

# Adding an alias
module_type_store.add_alias('nt', 'numerictypes')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/core/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 44, 0))

# 'from numpy.compat import isfileobj, bytes, long' statement (line 44)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/core/')
import_12237 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 44, 0), 'numpy.compat')

if (type(import_12237) is not StypyTypeError):

    if (import_12237 != 'pyd_module'):
        __import__(import_12237)
        sys_modules_12238 = sys.modules[import_12237]
        import_from_module(stypy.reporting.localization.Localization(__file__, 44, 0), 'numpy.compat', sys_modules_12238.module_type_store, module_type_store, ['isfileobj', 'bytes', 'long'])
        nest_module(stypy.reporting.localization.Localization(__file__, 44, 0), __file__, sys_modules_12238, sys_modules_12238.module_type_store, module_type_store)
    else:
        from numpy.compat import isfileobj, bytes, long

        import_from_module(stypy.reporting.localization.Localization(__file__, 44, 0), 'numpy.compat', None, module_type_store, ['isfileobj', 'bytes', 'long'], [isfileobj, bytes, long])

else:
    # Assigning a type to the variable 'numpy.compat' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 0), 'numpy.compat', import_12237)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/core/')


# Assigning a List to a Name (line 47):

# Assigning a List to a Name (line 47):
__all__ = ['record', 'recarray', 'format_parser']
module_type_store.set_exportable_members(['record', 'recarray', 'format_parser'])

# Obtaining an instance of the builtin type 'list' (line 47)
list_12239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 47)
# Adding element type (line 47)
str_12240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 11), 'str', 'record')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 10), list_12239, str_12240)
# Adding element type (line 47)
str_12241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 21), 'str', 'recarray')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 10), list_12239, str_12241)
# Adding element type (line 47)
str_12242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 33), 'str', 'format_parser')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 10), list_12239, str_12242)

# Assigning a type to the variable '__all__' (line 47)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 0), '__all__', list_12239)

# Assigning a Attribute to a Name (line 50):

# Assigning a Attribute to a Name (line 50):
# Getting the type of 'sb' (line 50)
sb_12243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 10), 'sb')
# Obtaining the member 'ndarray' of a type (line 50)
ndarray_12244 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 10), sb_12243, 'ndarray')
# Assigning a type to the variable 'ndarray' (line 50)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 0), 'ndarray', ndarray_12244)

# Assigning a Dict to a Name (line 52):

# Assigning a Dict to a Name (line 52):

# Obtaining an instance of the builtin type 'dict' (line 52)
dict_12245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 17), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 52)
# Adding element type (key, value) (line 52)
str_12246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 18), 'str', 'b')
str_12247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 22), 'str', '>')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 17), dict_12245, (str_12246, str_12247))
# Adding element type (key, value) (line 52)
str_12248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 18), 'str', 'l')
str_12249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 22), 'str', '<')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 17), dict_12245, (str_12248, str_12249))
# Adding element type (key, value) (line 52)
str_12250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 18), 'str', 'n')
str_12251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 22), 'str', '=')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 17), dict_12245, (str_12250, str_12251))
# Adding element type (key, value) (line 52)
str_12252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 18), 'str', 'B')
str_12253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 22), 'str', '>')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 17), dict_12245, (str_12252, str_12253))
# Adding element type (key, value) (line 52)
str_12254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 18), 'str', 'L')
str_12255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 22), 'str', '<')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 17), dict_12245, (str_12254, str_12255))
# Adding element type (key, value) (line 52)
str_12256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 18), 'str', 'N')
str_12257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 22), 'str', '=')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 17), dict_12245, (str_12256, str_12257))
# Adding element type (key, value) (line 52)
str_12258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 18), 'str', 'S')
str_12259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 22), 'str', 's')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 17), dict_12245, (str_12258, str_12259))
# Adding element type (key, value) (line 52)
str_12260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 18), 'str', 's')
str_12261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 22), 'str', 's')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 17), dict_12245, (str_12260, str_12261))
# Adding element type (key, value) (line 52)
str_12262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 18), 'str', '>')
str_12263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 22), 'str', '>')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 17), dict_12245, (str_12262, str_12263))
# Adding element type (key, value) (line 52)
str_12264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 18), 'str', '<')
str_12265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 22), 'str', '<')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 17), dict_12245, (str_12264, str_12265))
# Adding element type (key, value) (line 52)
str_12266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 18), 'str', '=')
str_12267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 22), 'str', '=')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 17), dict_12245, (str_12266, str_12267))
# Adding element type (key, value) (line 52)
str_12268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 18), 'str', '|')
str_12269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 22), 'str', '|')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 17), dict_12245, (str_12268, str_12269))
# Adding element type (key, value) (line 52)
str_12270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 18), 'str', 'I')
str_12271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 22), 'str', '|')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 17), dict_12245, (str_12270, str_12271))
# Adding element type (key, value) (line 52)
str_12272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 18), 'str', 'i')
str_12273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 22), 'str', '|')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 17), dict_12245, (str_12272, str_12273))

# Assigning a type to the variable '_byteorderconv' (line 52)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 0), '_byteorderconv', dict_12245)

# Assigning a Attribute to a Name (line 72):

# Assigning a Attribute to a Name (line 72):
# Getting the type of 'nt' (line 72)
nt_12274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 9), 'nt')
# Obtaining the member 'typeDict' of a type (line 72)
typeDict_12275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 9), nt_12274, 'typeDict')
# Assigning a type to the variable 'numfmt' (line 72)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 0), 'numfmt', typeDict_12275)

@norecursion
def find_duplicate(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'find_duplicate'
    module_type_store = module_type_store.open_function_context('find_duplicate', 74, 0, False)
    
    # Passed parameters checking function
    find_duplicate.stypy_localization = localization
    find_duplicate.stypy_type_of_self = None
    find_duplicate.stypy_type_store = module_type_store
    find_duplicate.stypy_function_name = 'find_duplicate'
    find_duplicate.stypy_param_names_list = ['list']
    find_duplicate.stypy_varargs_param_name = None
    find_duplicate.stypy_kwargs_param_name = None
    find_duplicate.stypy_call_defaults = defaults
    find_duplicate.stypy_call_varargs = varargs
    find_duplicate.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'find_duplicate', ['list'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'find_duplicate', localization, ['list'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'find_duplicate(...)' code ##################

    str_12276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 4), 'str', 'Find duplication in a list, return a list of duplicated elements')
    
    # Assigning a List to a Name (line 76):
    
    # Assigning a List to a Name (line 76):
    
    # Obtaining an instance of the builtin type 'list' (line 76)
    list_12277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 10), 'list')
    # Adding type elements to the builtin type 'list' instance (line 76)
    
    # Assigning a type to the variable 'dup' (line 76)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'dup', list_12277)
    
    
    # Call to range(...): (line 77)
    # Processing the call arguments (line 77)
    
    # Call to len(...): (line 77)
    # Processing the call arguments (line 77)
    # Getting the type of 'list' (line 77)
    list_12280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 23), 'list', False)
    # Processing the call keyword arguments (line 77)
    kwargs_12281 = {}
    # Getting the type of 'len' (line 77)
    len_12279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 19), 'len', False)
    # Calling len(args, kwargs) (line 77)
    len_call_result_12282 = invoke(stypy.reporting.localization.Localization(__file__, 77, 19), len_12279, *[list_12280], **kwargs_12281)
    
    # Processing the call keyword arguments (line 77)
    kwargs_12283 = {}
    # Getting the type of 'range' (line 77)
    range_12278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 13), 'range', False)
    # Calling range(args, kwargs) (line 77)
    range_call_result_12284 = invoke(stypy.reporting.localization.Localization(__file__, 77, 13), range_12278, *[len_call_result_12282], **kwargs_12283)
    
    # Testing the type of a for loop iterable (line 77)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 77, 4), range_call_result_12284)
    # Getting the type of the for loop variable (line 77)
    for_loop_var_12285 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 77, 4), range_call_result_12284)
    # Assigning a type to the variable 'i' (line 77)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 4), 'i', for_loop_var_12285)
    # SSA begins for a for statement (line 77)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 78)
    i_12286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 17), 'i')
    # Getting the type of 'list' (line 78)
    list_12287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 12), 'list')
    # Obtaining the member '__getitem__' of a type (line 78)
    getitem___12288 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 12), list_12287, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 78)
    subscript_call_result_12289 = invoke(stypy.reporting.localization.Localization(__file__, 78, 12), getitem___12288, i_12286)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 78)
    i_12290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 28), 'i')
    int_12291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 32), 'int')
    # Applying the binary operator '+' (line 78)
    result_add_12292 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 28), '+', i_12290, int_12291)
    
    slice_12293 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 78, 23), result_add_12292, None, None)
    # Getting the type of 'list' (line 78)
    list_12294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 23), 'list')
    # Obtaining the member '__getitem__' of a type (line 78)
    getitem___12295 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 23), list_12294, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 78)
    subscript_call_result_12296 = invoke(stypy.reporting.localization.Localization(__file__, 78, 23), getitem___12295, slice_12293)
    
    # Applying the binary operator 'in' (line 78)
    result_contains_12297 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 12), 'in', subscript_call_result_12289, subscript_call_result_12296)
    
    # Testing the type of an if condition (line 78)
    if_condition_12298 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 78, 8), result_contains_12297)
    # Assigning a type to the variable 'if_condition_12298' (line 78)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'if_condition_12298', if_condition_12298)
    # SSA begins for if statement (line 78)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 79)
    i_12299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 21), 'i')
    # Getting the type of 'list' (line 79)
    list_12300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 16), 'list')
    # Obtaining the member '__getitem__' of a type (line 79)
    getitem___12301 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 16), list_12300, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 79)
    subscript_call_result_12302 = invoke(stypy.reporting.localization.Localization(__file__, 79, 16), getitem___12301, i_12299)
    
    # Getting the type of 'dup' (line 79)
    dup_12303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 31), 'dup')
    # Applying the binary operator 'notin' (line 79)
    result_contains_12304 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 16), 'notin', subscript_call_result_12302, dup_12303)
    
    # Testing the type of an if condition (line 79)
    if_condition_12305 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 79, 12), result_contains_12304)
    # Assigning a type to the variable 'if_condition_12305' (line 79)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 12), 'if_condition_12305', if_condition_12305)
    # SSA begins for if statement (line 79)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 80)
    # Processing the call arguments (line 80)
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 80)
    i_12308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 32), 'i', False)
    # Getting the type of 'list' (line 80)
    list_12309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 27), 'list', False)
    # Obtaining the member '__getitem__' of a type (line 80)
    getitem___12310 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 27), list_12309, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 80)
    subscript_call_result_12311 = invoke(stypy.reporting.localization.Localization(__file__, 80, 27), getitem___12310, i_12308)
    
    # Processing the call keyword arguments (line 80)
    kwargs_12312 = {}
    # Getting the type of 'dup' (line 80)
    dup_12306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 16), 'dup', False)
    # Obtaining the member 'append' of a type (line 80)
    append_12307 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 16), dup_12306, 'append')
    # Calling append(args, kwargs) (line 80)
    append_call_result_12313 = invoke(stypy.reporting.localization.Localization(__file__, 80, 16), append_12307, *[subscript_call_result_12311], **kwargs_12312)
    
    # SSA join for if statement (line 79)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 78)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'dup' (line 81)
    dup_12314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 11), 'dup')
    # Assigning a type to the variable 'stypy_return_type' (line 81)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'stypy_return_type', dup_12314)
    
    # ################# End of 'find_duplicate(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'find_duplicate' in the type store
    # Getting the type of 'stypy_return_type' (line 74)
    stypy_return_type_12315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_12315)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'find_duplicate'
    return stypy_return_type_12315

# Assigning a type to the variable 'find_duplicate' (line 74)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 0), 'find_duplicate', find_duplicate)
# Declaration of the 'format_parser' class

class format_parser:
    str_12316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, (-1)), 'str', "\n    Class to convert formats, names, titles description to a dtype.\n\n    After constructing the format_parser object, the dtype attribute is\n    the converted data-type:\n    ``dtype = format_parser(formats, names, titles).dtype``\n\n    Attributes\n    ----------\n    dtype : dtype\n        The converted data-type.\n\n    Parameters\n    ----------\n    formats : str or list of str\n        The format description, either specified as a string with\n        comma-separated format descriptions in the form ``'f8, i4, a5'``, or\n        a list of format description strings  in the form\n        ``['f8', 'i4', 'a5']``.\n    names : str or list/tuple of str\n        The field names, either specified as a comma-separated string in the\n        form ``'col1, col2, col3'``, or as a list or tuple of strings in the\n        form ``['col1', 'col2', 'col3']``.\n        An empty list can be used, in that case default field names\n        ('f0', 'f1', ...) are used.\n    titles : sequence\n        Sequence of title strings. An empty list can be used to leave titles\n        out.\n    aligned : bool, optional\n        If True, align the fields by padding as the C-compiler would.\n        Default is False.\n    byteorder : str, optional\n        If specified, all the fields will be changed to the\n        provided byte-order.  Otherwise, the default byte-order is\n        used. For all available string specifiers, see `dtype.newbyteorder`.\n\n    See Also\n    --------\n    dtype, typename, sctype2char\n\n    Examples\n    --------\n    >>> np.format_parser(['f8', 'i4', 'a5'], ['col1', 'col2', 'col3'],\n    ...                  ['T1', 'T2', 'T3']).dtype\n    dtype([(('T1', 'col1'), '<f8'), (('T2', 'col2'), '<i4'),\n           (('T3', 'col3'), '|S5')])\n\n    `names` and/or `titles` can be empty lists. If `titles` is an empty list,\n    titles will simply not appear. If `names` is empty, default field names\n    will be used.\n\n    >>> np.format_parser(['f8', 'i4', 'a5'], ['col1', 'col2', 'col3'],\n    ...                  []).dtype\n    dtype([('col1', '<f8'), ('col2', '<i4'), ('col3', '|S5')])\n    >>> np.format_parser(['f8', 'i4', 'a5'], [], []).dtype\n    dtype([('f0', '<f8'), ('f1', '<i4'), ('f2', '|S5')])\n\n    ")

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'False' (line 143)
        False_12317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 55), 'False')
        # Getting the type of 'None' (line 143)
        None_12318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 72), 'None')
        defaults = [False_12317, None_12318]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 143, 4, False)
        # Assigning a type to the variable 'self' (line 144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'format_parser.__init__', ['formats', 'names', 'titles', 'aligned', 'byteorder'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['formats', 'names', 'titles', 'aligned', 'byteorder'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Call to _parseFormats(...): (line 144)
        # Processing the call arguments (line 144)
        # Getting the type of 'formats' (line 144)
        formats_12321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 27), 'formats', False)
        # Getting the type of 'aligned' (line 144)
        aligned_12322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 36), 'aligned', False)
        # Processing the call keyword arguments (line 144)
        kwargs_12323 = {}
        # Getting the type of 'self' (line 144)
        self_12319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'self', False)
        # Obtaining the member '_parseFormats' of a type (line 144)
        _parseFormats_12320 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 8), self_12319, '_parseFormats')
        # Calling _parseFormats(args, kwargs) (line 144)
        _parseFormats_call_result_12324 = invoke(stypy.reporting.localization.Localization(__file__, 144, 8), _parseFormats_12320, *[formats_12321, aligned_12322], **kwargs_12323)
        
        
        # Call to _setfieldnames(...): (line 145)
        # Processing the call arguments (line 145)
        # Getting the type of 'names' (line 145)
        names_12327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 28), 'names', False)
        # Getting the type of 'titles' (line 145)
        titles_12328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 35), 'titles', False)
        # Processing the call keyword arguments (line 145)
        kwargs_12329 = {}
        # Getting the type of 'self' (line 145)
        self_12325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 8), 'self', False)
        # Obtaining the member '_setfieldnames' of a type (line 145)
        _setfieldnames_12326 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 8), self_12325, '_setfieldnames')
        # Calling _setfieldnames(args, kwargs) (line 145)
        _setfieldnames_call_result_12330 = invoke(stypy.reporting.localization.Localization(__file__, 145, 8), _setfieldnames_12326, *[names_12327, titles_12328], **kwargs_12329)
        
        
        # Call to _createdescr(...): (line 146)
        # Processing the call arguments (line 146)
        # Getting the type of 'byteorder' (line 146)
        byteorder_12333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 26), 'byteorder', False)
        # Processing the call keyword arguments (line 146)
        kwargs_12334 = {}
        # Getting the type of 'self' (line 146)
        self_12331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'self', False)
        # Obtaining the member '_createdescr' of a type (line 146)
        _createdescr_12332 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 8), self_12331, '_createdescr')
        # Calling _createdescr(args, kwargs) (line 146)
        _createdescr_call_result_12335 = invoke(stypy.reporting.localization.Localization(__file__, 146, 8), _createdescr_12332, *[byteorder_12333], **kwargs_12334)
        
        
        # Assigning a Attribute to a Attribute (line 147):
        
        # Assigning a Attribute to a Attribute (line 147):
        # Getting the type of 'self' (line 147)
        self_12336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 21), 'self')
        # Obtaining the member '_descr' of a type (line 147)
        _descr_12337 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 21), self_12336, '_descr')
        # Getting the type of 'self' (line 147)
        self_12338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'self')
        # Setting the type of the member 'dtype' of a type (line 147)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 8), self_12338, 'dtype', _descr_12337)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def _parseFormats(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_12339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 45), 'int')
        defaults = [int_12339]
        # Create a new context for function '_parseFormats'
        module_type_store = module_type_store.open_function_context('_parseFormats', 149, 4, False)
        # Assigning a type to the variable 'self' (line 150)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        format_parser._parseFormats.__dict__.__setitem__('stypy_localization', localization)
        format_parser._parseFormats.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        format_parser._parseFormats.__dict__.__setitem__('stypy_type_store', module_type_store)
        format_parser._parseFormats.__dict__.__setitem__('stypy_function_name', 'format_parser._parseFormats')
        format_parser._parseFormats.__dict__.__setitem__('stypy_param_names_list', ['formats', 'aligned'])
        format_parser._parseFormats.__dict__.__setitem__('stypy_varargs_param_name', None)
        format_parser._parseFormats.__dict__.__setitem__('stypy_kwargs_param_name', None)
        format_parser._parseFormats.__dict__.__setitem__('stypy_call_defaults', defaults)
        format_parser._parseFormats.__dict__.__setitem__('stypy_call_varargs', varargs)
        format_parser._parseFormats.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        format_parser._parseFormats.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'format_parser._parseFormats', ['formats', 'aligned'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_parseFormats', localization, ['formats', 'aligned'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_parseFormats(...)' code ##################

        str_12340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 8), 'str', ' Parse the field formats ')
        
        # Type idiom detected: calculating its left and rigth part (line 152)
        # Getting the type of 'formats' (line 152)
        formats_12341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 11), 'formats')
        # Getting the type of 'None' (line 152)
        None_12342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 22), 'None')
        
        (may_be_12343, more_types_in_union_12344) = may_be_none(formats_12341, None_12342)

        if may_be_12343:

            if more_types_in_union_12344:
                # Runtime conditional SSA (line 152)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to ValueError(...): (line 153)
            # Processing the call arguments (line 153)
            str_12346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 29), 'str', 'Need formats argument')
            # Processing the call keyword arguments (line 153)
            kwargs_12347 = {}
            # Getting the type of 'ValueError' (line 153)
            ValueError_12345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 18), 'ValueError', False)
            # Calling ValueError(args, kwargs) (line 153)
            ValueError_call_result_12348 = invoke(stypy.reporting.localization.Localization(__file__, 153, 18), ValueError_12345, *[str_12346], **kwargs_12347)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 153, 12), ValueError_call_result_12348, 'raise parameter', BaseException)

            if more_types_in_union_12344:
                # SSA join for if statement (line 152)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 154)
        # Getting the type of 'list' (line 154)
        list_12349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 31), 'list')
        # Getting the type of 'formats' (line 154)
        formats_12350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 22), 'formats')
        
        (may_be_12351, more_types_in_union_12352) = may_be_subtype(list_12349, formats_12350)

        if may_be_12351:

            if more_types_in_union_12352:
                # Runtime conditional SSA (line 154)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'formats' (line 154)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'formats', remove_not_subtype_from_union(formats_12350, list))
            
            
            
            # Call to len(...): (line 155)
            # Processing the call arguments (line 155)
            # Getting the type of 'formats' (line 155)
            formats_12354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 19), 'formats', False)
            # Processing the call keyword arguments (line 155)
            kwargs_12355 = {}
            # Getting the type of 'len' (line 155)
            len_12353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 15), 'len', False)
            # Calling len(args, kwargs) (line 155)
            len_call_result_12356 = invoke(stypy.reporting.localization.Localization(__file__, 155, 15), len_12353, *[formats_12354], **kwargs_12355)
            
            int_12357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 30), 'int')
            # Applying the binary operator '<' (line 155)
            result_lt_12358 = python_operator(stypy.reporting.localization.Localization(__file__, 155, 15), '<', len_call_result_12356, int_12357)
            
            # Testing the type of an if condition (line 155)
            if_condition_12359 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 155, 12), result_lt_12358)
            # Assigning a type to the variable 'if_condition_12359' (line 155)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 12), 'if_condition_12359', if_condition_12359)
            # SSA begins for if statement (line 155)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to append(...): (line 156)
            # Processing the call arguments (line 156)
            str_12362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 31), 'str', '')
            # Processing the call keyword arguments (line 156)
            kwargs_12363 = {}
            # Getting the type of 'formats' (line 156)
            formats_12360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 16), 'formats', False)
            # Obtaining the member 'append' of a type (line 156)
            append_12361 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 16), formats_12360, 'append')
            # Calling append(args, kwargs) (line 156)
            append_call_result_12364 = invoke(stypy.reporting.localization.Localization(__file__, 156, 16), append_12361, *[str_12362], **kwargs_12363)
            
            # SSA join for if statement (line 155)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Assigning a Call to a Name (line 157):
            
            # Assigning a Call to a Name (line 157):
            
            # Call to join(...): (line 157)
            # Processing the call arguments (line 157)
            # Getting the type of 'formats' (line 157)
            formats_12367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 31), 'formats', False)
            # Processing the call keyword arguments (line 157)
            kwargs_12368 = {}
            str_12365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 22), 'str', ',')
            # Obtaining the member 'join' of a type (line 157)
            join_12366 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 22), str_12365, 'join')
            # Calling join(args, kwargs) (line 157)
            join_call_result_12369 = invoke(stypy.reporting.localization.Localization(__file__, 157, 22), join_12366, *[formats_12367], **kwargs_12368)
            
            # Assigning a type to the variable 'formats' (line 157)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 12), 'formats', join_call_result_12369)

            if more_types_in_union_12352:
                # SSA join for if statement (line 154)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Name (line 158):
        
        # Assigning a Call to a Name (line 158):
        
        # Call to dtype(...): (line 158)
        # Processing the call arguments (line 158)
        # Getting the type of 'formats' (line 158)
        formats_12372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 25), 'formats', False)
        # Getting the type of 'aligned' (line 158)
        aligned_12373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 34), 'aligned', False)
        # Processing the call keyword arguments (line 158)
        kwargs_12374 = {}
        # Getting the type of 'sb' (line 158)
        sb_12370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 16), 'sb', False)
        # Obtaining the member 'dtype' of a type (line 158)
        dtype_12371 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 16), sb_12370, 'dtype')
        # Calling dtype(args, kwargs) (line 158)
        dtype_call_result_12375 = invoke(stypy.reporting.localization.Localization(__file__, 158, 16), dtype_12371, *[formats_12372, aligned_12373], **kwargs_12374)
        
        # Assigning a type to the variable 'dtype' (line 158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'dtype', dtype_call_result_12375)
        
        # Assigning a Attribute to a Name (line 159):
        
        # Assigning a Attribute to a Name (line 159):
        # Getting the type of 'dtype' (line 159)
        dtype_12376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 17), 'dtype')
        # Obtaining the member 'fields' of a type (line 159)
        fields_12377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 17), dtype_12376, 'fields')
        # Assigning a type to the variable 'fields' (line 159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'fields', fields_12377)
        
        # Type idiom detected: calculating its left and rigth part (line 160)
        # Getting the type of 'fields' (line 160)
        fields_12378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 11), 'fields')
        # Getting the type of 'None' (line 160)
        None_12379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 21), 'None')
        
        (may_be_12380, more_types_in_union_12381) = may_be_none(fields_12378, None_12379)

        if may_be_12380:

            if more_types_in_union_12381:
                # Runtime conditional SSA (line 160)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 161):
            
            # Assigning a Call to a Name (line 161):
            
            # Call to dtype(...): (line 161)
            # Processing the call arguments (line 161)
            
            # Obtaining an instance of the builtin type 'list' (line 161)
            list_12384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 29), 'list')
            # Adding type elements to the builtin type 'list' instance (line 161)
            # Adding element type (line 161)
            
            # Obtaining an instance of the builtin type 'tuple' (line 161)
            tuple_12385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 31), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 161)
            # Adding element type (line 161)
            str_12386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 31), 'str', 'f1')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 161, 31), tuple_12385, str_12386)
            # Adding element type (line 161)
            # Getting the type of 'dtype' (line 161)
            dtype_12387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 37), 'dtype', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 161, 31), tuple_12385, dtype_12387)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 161, 29), list_12384, tuple_12385)
            
            # Getting the type of 'aligned' (line 161)
            aligned_12388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 46), 'aligned', False)
            # Processing the call keyword arguments (line 161)
            kwargs_12389 = {}
            # Getting the type of 'sb' (line 161)
            sb_12382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 20), 'sb', False)
            # Obtaining the member 'dtype' of a type (line 161)
            dtype_12383 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 20), sb_12382, 'dtype')
            # Calling dtype(args, kwargs) (line 161)
            dtype_call_result_12390 = invoke(stypy.reporting.localization.Localization(__file__, 161, 20), dtype_12383, *[list_12384, aligned_12388], **kwargs_12389)
            
            # Assigning a type to the variable 'dtype' (line 161)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 12), 'dtype', dtype_call_result_12390)
            
            # Assigning a Attribute to a Name (line 162):
            
            # Assigning a Attribute to a Name (line 162):
            # Getting the type of 'dtype' (line 162)
            dtype_12391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 21), 'dtype')
            # Obtaining the member 'fields' of a type (line 162)
            fields_12392 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 21), dtype_12391, 'fields')
            # Assigning a type to the variable 'fields' (line 162)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 12), 'fields', fields_12392)

            if more_types_in_union_12381:
                # SSA join for if statement (line 160)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Attribute to a Name (line 163):
        
        # Assigning a Attribute to a Name (line 163):
        # Getting the type of 'dtype' (line 163)
        dtype_12393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 15), 'dtype')
        # Obtaining the member 'names' of a type (line 163)
        names_12394 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 15), dtype_12393, 'names')
        # Assigning a type to the variable 'keys' (line 163)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 8), 'keys', names_12394)
        
        # Assigning a ListComp to a Attribute (line 164):
        
        # Assigning a ListComp to a Attribute (line 164):
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'keys' (line 164)
        keys_12402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 53), 'keys')
        comprehension_12403 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 27), keys_12402)
        # Assigning a type to the variable 'key' (line 164)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 27), 'key', comprehension_12403)
        
        # Obtaining the type of the subscript
        int_12395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 39), 'int')
        
        # Obtaining the type of the subscript
        # Getting the type of 'key' (line 164)
        key_12396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 34), 'key')
        # Getting the type of 'fields' (line 164)
        fields_12397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 27), 'fields')
        # Obtaining the member '__getitem__' of a type (line 164)
        getitem___12398 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 27), fields_12397, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 164)
        subscript_call_result_12399 = invoke(stypy.reporting.localization.Localization(__file__, 164, 27), getitem___12398, key_12396)
        
        # Obtaining the member '__getitem__' of a type (line 164)
        getitem___12400 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 27), subscript_call_result_12399, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 164)
        subscript_call_result_12401 = invoke(stypy.reporting.localization.Localization(__file__, 164, 27), getitem___12400, int_12395)
        
        list_12404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 27), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 27), list_12404, subscript_call_result_12401)
        # Getting the type of 'self' (line 164)
        self_12405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 8), 'self')
        # Setting the type of the member '_f_formats' of a type (line 164)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 8), self_12405, '_f_formats', list_12404)
        
        # Assigning a ListComp to a Attribute (line 165):
        
        # Assigning a ListComp to a Attribute (line 165):
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'keys' (line 165)
        keys_12413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 51), 'keys')
        comprehension_12414 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 25), keys_12413)
        # Assigning a type to the variable 'key' (line 165)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 25), 'key', comprehension_12414)
        
        # Obtaining the type of the subscript
        int_12406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 37), 'int')
        
        # Obtaining the type of the subscript
        # Getting the type of 'key' (line 165)
        key_12407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 32), 'key')
        # Getting the type of 'fields' (line 165)
        fields_12408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 25), 'fields')
        # Obtaining the member '__getitem__' of a type (line 165)
        getitem___12409 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 25), fields_12408, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 165)
        subscript_call_result_12410 = invoke(stypy.reporting.localization.Localization(__file__, 165, 25), getitem___12409, key_12407)
        
        # Obtaining the member '__getitem__' of a type (line 165)
        getitem___12411 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 25), subscript_call_result_12410, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 165)
        subscript_call_result_12412 = invoke(stypy.reporting.localization.Localization(__file__, 165, 25), getitem___12411, int_12406)
        
        list_12415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 25), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 25), list_12415, subscript_call_result_12412)
        # Getting the type of 'self' (line 165)
        self_12416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 8), 'self')
        # Setting the type of the member '_offsets' of a type (line 165)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 8), self_12416, '_offsets', list_12415)
        
        # Assigning a Call to a Attribute (line 166):
        
        # Assigning a Call to a Attribute (line 166):
        
        # Call to len(...): (line 166)
        # Processing the call arguments (line 166)
        # Getting the type of 'keys' (line 166)
        keys_12418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 28), 'keys', False)
        # Processing the call keyword arguments (line 166)
        kwargs_12419 = {}
        # Getting the type of 'len' (line 166)
        len_12417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 24), 'len', False)
        # Calling len(args, kwargs) (line 166)
        len_call_result_12420 = invoke(stypy.reporting.localization.Localization(__file__, 166, 24), len_12417, *[keys_12418], **kwargs_12419)
        
        # Getting the type of 'self' (line 166)
        self_12421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'self')
        # Setting the type of the member '_nfields' of a type (line 166)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 8), self_12421, '_nfields', len_call_result_12420)
        
        # ################# End of '_parseFormats(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_parseFormats' in the type store
        # Getting the type of 'stypy_return_type' (line 149)
        stypy_return_type_12422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_12422)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_parseFormats'
        return stypy_return_type_12422


    @norecursion
    def _setfieldnames(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_setfieldnames'
        module_type_store = module_type_store.open_function_context('_setfieldnames', 168, 4, False)
        # Assigning a type to the variable 'self' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        format_parser._setfieldnames.__dict__.__setitem__('stypy_localization', localization)
        format_parser._setfieldnames.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        format_parser._setfieldnames.__dict__.__setitem__('stypy_type_store', module_type_store)
        format_parser._setfieldnames.__dict__.__setitem__('stypy_function_name', 'format_parser._setfieldnames')
        format_parser._setfieldnames.__dict__.__setitem__('stypy_param_names_list', ['names', 'titles'])
        format_parser._setfieldnames.__dict__.__setitem__('stypy_varargs_param_name', None)
        format_parser._setfieldnames.__dict__.__setitem__('stypy_kwargs_param_name', None)
        format_parser._setfieldnames.__dict__.__setitem__('stypy_call_defaults', defaults)
        format_parser._setfieldnames.__dict__.__setitem__('stypy_call_varargs', varargs)
        format_parser._setfieldnames.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        format_parser._setfieldnames.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'format_parser._setfieldnames', ['names', 'titles'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_setfieldnames', localization, ['names', 'titles'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_setfieldnames(...)' code ##################

        str_12423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, (-1)), 'str', 'convert input field names into a list and assign to the _names\n        attribute ')
        
        # Getting the type of 'names' (line 172)
        names_12424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 12), 'names')
        # Testing the type of an if condition (line 172)
        if_condition_12425 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 172, 8), names_12424)
        # Assigning a type to the variable 'if_condition_12425' (line 172)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 8), 'if_condition_12425', if_condition_12425)
        # SSA begins for if statement (line 172)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        
        # Call to type(...): (line 173)
        # Processing the call arguments (line 173)
        # Getting the type of 'names' (line 173)
        names_12427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 21), 'names', False)
        # Processing the call keyword arguments (line 173)
        kwargs_12428 = {}
        # Getting the type of 'type' (line 173)
        type_12426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 16), 'type', False)
        # Calling type(args, kwargs) (line 173)
        type_call_result_12429 = invoke(stypy.reporting.localization.Localization(__file__, 173, 16), type_12426, *[names_12427], **kwargs_12428)
        
        
        # Obtaining an instance of the builtin type 'list' (line 173)
        list_12430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 173)
        # Adding element type (line 173)
        # Getting the type of 'list' (line 173)
        list_12431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 32), 'list')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 173, 31), list_12430, list_12431)
        # Adding element type (line 173)
        # Getting the type of 'tuple' (line 173)
        tuple_12432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 38), 'tuple')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 173, 31), list_12430, tuple_12432)
        
        # Applying the binary operator 'in' (line 173)
        result_contains_12433 = python_operator(stypy.reporting.localization.Localization(__file__, 173, 16), 'in', type_call_result_12429, list_12430)
        
        # Testing the type of an if condition (line 173)
        if_condition_12434 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 173, 12), result_contains_12433)
        # Assigning a type to the variable 'if_condition_12434' (line 173)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 12), 'if_condition_12434', if_condition_12434)
        # SSA begins for if statement (line 173)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        pass
        # SSA branch for the else part of an if statement (line 173)
        module_type_store.open_ssa_branch('else')
        
        # Type idiom detected: calculating its left and rigth part (line 175)
        # Getting the type of 'str' (line 175)
        str_12435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 35), 'str')
        # Getting the type of 'names' (line 175)
        names_12436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 28), 'names')
        
        (may_be_12437, more_types_in_union_12438) = may_be_subtype(str_12435, names_12436)

        if may_be_12437:

            if more_types_in_union_12438:
                # Runtime conditional SSA (line 175)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'names' (line 175)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 17), 'names', remove_not_subtype_from_union(names_12436, str))
            
            # Assigning a Call to a Name (line 176):
            
            # Assigning a Call to a Name (line 176):
            
            # Call to split(...): (line 176)
            # Processing the call arguments (line 176)
            str_12441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 36), 'str', ',')
            # Processing the call keyword arguments (line 176)
            kwargs_12442 = {}
            # Getting the type of 'names' (line 176)
            names_12439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 24), 'names', False)
            # Obtaining the member 'split' of a type (line 176)
            split_12440 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 24), names_12439, 'split')
            # Calling split(args, kwargs) (line 176)
            split_call_result_12443 = invoke(stypy.reporting.localization.Localization(__file__, 176, 24), split_12440, *[str_12441], **kwargs_12442)
            
            # Assigning a type to the variable 'names' (line 176)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 16), 'names', split_call_result_12443)

            if more_types_in_union_12438:
                # Runtime conditional SSA for else branch (line 175)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_12437) or more_types_in_union_12438):
            # Assigning a type to the variable 'names' (line 175)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 17), 'names', remove_subtype_from_union(names_12436, str))
            
            # Call to NameError(...): (line 178)
            # Processing the call arguments (line 178)
            str_12445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 32), 'str', 'illegal input names %s')
            
            # Call to repr(...): (line 178)
            # Processing the call arguments (line 178)
            # Getting the type of 'names' (line 178)
            names_12447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 64), 'names', False)
            # Processing the call keyword arguments (line 178)
            kwargs_12448 = {}
            # Getting the type of 'repr' (line 178)
            repr_12446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 59), 'repr', False)
            # Calling repr(args, kwargs) (line 178)
            repr_call_result_12449 = invoke(stypy.reporting.localization.Localization(__file__, 178, 59), repr_12446, *[names_12447], **kwargs_12448)
            
            # Applying the binary operator '%' (line 178)
            result_mod_12450 = python_operator(stypy.reporting.localization.Localization(__file__, 178, 32), '%', str_12445, repr_call_result_12449)
            
            # Processing the call keyword arguments (line 178)
            kwargs_12451 = {}
            # Getting the type of 'NameError' (line 178)
            NameError_12444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 22), 'NameError', False)
            # Calling NameError(args, kwargs) (line 178)
            NameError_call_result_12452 = invoke(stypy.reporting.localization.Localization(__file__, 178, 22), NameError_12444, *[result_mod_12450], **kwargs_12451)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 178, 16), NameError_call_result_12452, 'raise parameter', BaseException)

            if (may_be_12437 and more_types_in_union_12438):
                # SSA join for if statement (line 175)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for if statement (line 173)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a ListComp to a Attribute (line 180):
        
        # Assigning a ListComp to a Attribute (line 180):
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Obtaining the type of the subscript
        # Getting the type of 'self' (line 180)
        self_12457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 53), 'self')
        # Obtaining the member '_nfields' of a type (line 180)
        _nfields_12458 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 53), self_12457, '_nfields')
        slice_12459 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 180, 46), None, _nfields_12458, None)
        # Getting the type of 'names' (line 180)
        names_12460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 46), 'names')
        # Obtaining the member '__getitem__' of a type (line 180)
        getitem___12461 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 46), names_12460, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 180)
        subscript_call_result_12462 = invoke(stypy.reporting.localization.Localization(__file__, 180, 46), getitem___12461, slice_12459)
        
        comprehension_12463 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 180, 27), subscript_call_result_12462)
        # Assigning a type to the variable 'n' (line 180)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 27), 'n', comprehension_12463)
        
        # Call to strip(...): (line 180)
        # Processing the call keyword arguments (line 180)
        kwargs_12455 = {}
        # Getting the type of 'n' (line 180)
        n_12453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 27), 'n', False)
        # Obtaining the member 'strip' of a type (line 180)
        strip_12454 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 27), n_12453, 'strip')
        # Calling strip(args, kwargs) (line 180)
        strip_call_result_12456 = invoke(stypy.reporting.localization.Localization(__file__, 180, 27), strip_12454, *[], **kwargs_12455)
        
        list_12464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 27), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 180, 27), list_12464, strip_call_result_12456)
        # Getting the type of 'self' (line 180)
        self_12465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 12), 'self')
        # Setting the type of the member '_names' of a type (line 180)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 12), self_12465, '_names', list_12464)
        # SSA branch for the else part of an if statement (line 172)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a List to a Attribute (line 182):
        
        # Assigning a List to a Attribute (line 182):
        
        # Obtaining an instance of the builtin type 'list' (line 182)
        list_12466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 182)
        
        # Getting the type of 'self' (line 182)
        self_12467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 12), 'self')
        # Setting the type of the member '_names' of a type (line 182)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 12), self_12467, '_names', list_12466)
        # SSA join for if statement (line 172)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'self' (line 188)
        self_12468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 8), 'self')
        # Obtaining the member '_names' of a type (line 188)
        _names_12469 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 8), self_12468, '_names')
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to range(...): (line 188)
        # Processing the call arguments (line 188)
        
        # Call to len(...): (line 188)
        # Processing the call arguments (line 188)
        # Getting the type of 'self' (line 188)
        self_12475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 53), 'self', False)
        # Obtaining the member '_names' of a type (line 188)
        _names_12476 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 53), self_12475, '_names')
        # Processing the call keyword arguments (line 188)
        kwargs_12477 = {}
        # Getting the type of 'len' (line 188)
        len_12474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 49), 'len', False)
        # Calling len(args, kwargs) (line 188)
        len_call_result_12478 = invoke(stypy.reporting.localization.Localization(__file__, 188, 49), len_12474, *[_names_12476], **kwargs_12477)
        
        # Getting the type of 'self' (line 189)
        self_12479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 49), 'self', False)
        # Obtaining the member '_nfields' of a type (line 189)
        _nfields_12480 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 49), self_12479, '_nfields')
        # Processing the call keyword arguments (line 188)
        kwargs_12481 = {}
        # Getting the type of 'range' (line 188)
        range_12473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 43), 'range', False)
        # Calling range(args, kwargs) (line 188)
        range_call_result_12482 = invoke(stypy.reporting.localization.Localization(__file__, 188, 43), range_12473, *[len_call_result_12478, _nfields_12480], **kwargs_12481)
        
        comprehension_12483 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 188, 24), range_call_result_12482)
        # Assigning a type to the variable 'i' (line 188)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 24), 'i', comprehension_12483)
        str_12470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 24), 'str', 'f%d')
        # Getting the type of 'i' (line 188)
        i_12471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 32), 'i')
        # Applying the binary operator '%' (line 188)
        result_mod_12472 = python_operator(stypy.reporting.localization.Localization(__file__, 188, 24), '%', str_12470, i_12471)
        
        list_12484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 24), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 188, 24), list_12484, result_mod_12472)
        # Applying the binary operator '+=' (line 188)
        result_iadd_12485 = python_operator(stypy.reporting.localization.Localization(__file__, 188, 8), '+=', _names_12469, list_12484)
        # Getting the type of 'self' (line 188)
        self_12486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 8), 'self')
        # Setting the type of the member '_names' of a type (line 188)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 8), self_12486, '_names', result_iadd_12485)
        
        
        # Assigning a Call to a Name (line 191):
        
        # Assigning a Call to a Name (line 191):
        
        # Call to find_duplicate(...): (line 191)
        # Processing the call arguments (line 191)
        # Getting the type of 'self' (line 191)
        self_12488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 30), 'self', False)
        # Obtaining the member '_names' of a type (line 191)
        _names_12489 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 30), self_12488, '_names')
        # Processing the call keyword arguments (line 191)
        kwargs_12490 = {}
        # Getting the type of 'find_duplicate' (line 191)
        find_duplicate_12487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 15), 'find_duplicate', False)
        # Calling find_duplicate(args, kwargs) (line 191)
        find_duplicate_call_result_12491 = invoke(stypy.reporting.localization.Localization(__file__, 191, 15), find_duplicate_12487, *[_names_12489], **kwargs_12490)
        
        # Assigning a type to the variable '_dup' (line 191)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 8), '_dup', find_duplicate_call_result_12491)
        
        # Getting the type of '_dup' (line 192)
        _dup_12492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 11), '_dup')
        # Testing the type of an if condition (line 192)
        if_condition_12493 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 192, 8), _dup_12492)
        # Assigning a type to the variable 'if_condition_12493' (line 192)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 8), 'if_condition_12493', if_condition_12493)
        # SSA begins for if statement (line 192)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 193)
        # Processing the call arguments (line 193)
        str_12495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 29), 'str', 'Duplicate field names: %s')
        # Getting the type of '_dup' (line 193)
        _dup_12496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 59), '_dup', False)
        # Applying the binary operator '%' (line 193)
        result_mod_12497 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 29), '%', str_12495, _dup_12496)
        
        # Processing the call keyword arguments (line 193)
        kwargs_12498 = {}
        # Getting the type of 'ValueError' (line 193)
        ValueError_12494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 193)
        ValueError_call_result_12499 = invoke(stypy.reporting.localization.Localization(__file__, 193, 18), ValueError_12494, *[result_mod_12497], **kwargs_12498)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 193, 12), ValueError_call_result_12499, 'raise parameter', BaseException)
        # SSA join for if statement (line 192)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'titles' (line 195)
        titles_12500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 12), 'titles')
        # Testing the type of an if condition (line 195)
        if_condition_12501 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 195, 8), titles_12500)
        # Assigning a type to the variable 'if_condition_12501' (line 195)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 8), 'if_condition_12501', if_condition_12501)
        # SSA begins for if statement (line 195)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a ListComp to a Attribute (line 196):
        
        # Assigning a ListComp to a Attribute (line 196):
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Obtaining the type of the subscript
        # Getting the type of 'self' (line 196)
        self_12506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 55), 'self')
        # Obtaining the member '_nfields' of a type (line 196)
        _nfields_12507 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 55), self_12506, '_nfields')
        slice_12508 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 196, 47), None, _nfields_12507, None)
        # Getting the type of 'titles' (line 196)
        titles_12509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 47), 'titles')
        # Obtaining the member '__getitem__' of a type (line 196)
        getitem___12510 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 47), titles_12509, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 196)
        subscript_call_result_12511 = invoke(stypy.reporting.localization.Localization(__file__, 196, 47), getitem___12510, slice_12508)
        
        comprehension_12512 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 28), subscript_call_result_12511)
        # Assigning a type to the variable 'n' (line 196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 28), 'n', comprehension_12512)
        
        # Call to strip(...): (line 196)
        # Processing the call keyword arguments (line 196)
        kwargs_12504 = {}
        # Getting the type of 'n' (line 196)
        n_12502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 28), 'n', False)
        # Obtaining the member 'strip' of a type (line 196)
        strip_12503 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 28), n_12502, 'strip')
        # Calling strip(args, kwargs) (line 196)
        strip_call_result_12505 = invoke(stypy.reporting.localization.Localization(__file__, 196, 28), strip_12503, *[], **kwargs_12504)
        
        list_12513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 28), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 28), list_12513, strip_call_result_12505)
        # Getting the type of 'self' (line 196)
        self_12514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 12), 'self')
        # Setting the type of the member '_titles' of a type (line 196)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 12), self_12514, '_titles', list_12513)
        # SSA branch for the else part of an if statement (line 195)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a List to a Attribute (line 198):
        
        # Assigning a List to a Attribute (line 198):
        
        # Obtaining an instance of the builtin type 'list' (line 198)
        list_12515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 198)
        
        # Getting the type of 'self' (line 198)
        self_12516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 12), 'self')
        # Setting the type of the member '_titles' of a type (line 198)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 12), self_12516, '_titles', list_12515)
        
        # Assigning a List to a Name (line 199):
        
        # Assigning a List to a Name (line 199):
        
        # Obtaining an instance of the builtin type 'list' (line 199)
        list_12517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 199)
        
        # Assigning a type to the variable 'titles' (line 199)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 12), 'titles', list_12517)
        # SSA join for if statement (line 195)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'self' (line 201)
        self_12518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 12), 'self')
        # Obtaining the member '_nfields' of a type (line 201)
        _nfields_12519 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 12), self_12518, '_nfields')
        
        # Call to len(...): (line 201)
        # Processing the call arguments (line 201)
        # Getting the type of 'titles' (line 201)
        titles_12521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 32), 'titles', False)
        # Processing the call keyword arguments (line 201)
        kwargs_12522 = {}
        # Getting the type of 'len' (line 201)
        len_12520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 28), 'len', False)
        # Calling len(args, kwargs) (line 201)
        len_call_result_12523 = invoke(stypy.reporting.localization.Localization(__file__, 201, 28), len_12520, *[titles_12521], **kwargs_12522)
        
        # Applying the binary operator '>' (line 201)
        result_gt_12524 = python_operator(stypy.reporting.localization.Localization(__file__, 201, 12), '>', _nfields_12519, len_call_result_12523)
        
        # Testing the type of an if condition (line 201)
        if_condition_12525 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 201, 8), result_gt_12524)
        # Assigning a type to the variable 'if_condition_12525' (line 201)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 8), 'if_condition_12525', if_condition_12525)
        # SSA begins for if statement (line 201)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'self' (line 202)
        self_12526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 12), 'self')
        # Obtaining the member '_titles' of a type (line 202)
        _titles_12527 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 12), self_12526, '_titles')
        
        # Obtaining an instance of the builtin type 'list' (line 202)
        list_12528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 202)
        # Adding element type (line 202)
        # Getting the type of 'None' (line 202)
        None_12529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 29), 'None')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 28), list_12528, None_12529)
        
        # Getting the type of 'self' (line 202)
        self_12530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 38), 'self')
        # Obtaining the member '_nfields' of a type (line 202)
        _nfields_12531 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 38), self_12530, '_nfields')
        
        # Call to len(...): (line 202)
        # Processing the call arguments (line 202)
        # Getting the type of 'titles' (line 202)
        titles_12533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 58), 'titles', False)
        # Processing the call keyword arguments (line 202)
        kwargs_12534 = {}
        # Getting the type of 'len' (line 202)
        len_12532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 54), 'len', False)
        # Calling len(args, kwargs) (line 202)
        len_call_result_12535 = invoke(stypy.reporting.localization.Localization(__file__, 202, 54), len_12532, *[titles_12533], **kwargs_12534)
        
        # Applying the binary operator '-' (line 202)
        result_sub_12536 = python_operator(stypy.reporting.localization.Localization(__file__, 202, 38), '-', _nfields_12531, len_call_result_12535)
        
        # Applying the binary operator '*' (line 202)
        result_mul_12537 = python_operator(stypy.reporting.localization.Localization(__file__, 202, 28), '*', list_12528, result_sub_12536)
        
        # Applying the binary operator '+=' (line 202)
        result_iadd_12538 = python_operator(stypy.reporting.localization.Localization(__file__, 202, 12), '+=', _titles_12527, result_mul_12537)
        # Getting the type of 'self' (line 202)
        self_12539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 12), 'self')
        # Setting the type of the member '_titles' of a type (line 202)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 12), self_12539, '_titles', result_iadd_12538)
        
        # SSA join for if statement (line 201)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_setfieldnames(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_setfieldnames' in the type store
        # Getting the type of 'stypy_return_type' (line 168)
        stypy_return_type_12540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_12540)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_setfieldnames'
        return stypy_return_type_12540


    @norecursion
    def _createdescr(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_createdescr'
        module_type_store = module_type_store.open_function_context('_createdescr', 204, 4, False)
        # Assigning a type to the variable 'self' (line 205)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        format_parser._createdescr.__dict__.__setitem__('stypy_localization', localization)
        format_parser._createdescr.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        format_parser._createdescr.__dict__.__setitem__('stypy_type_store', module_type_store)
        format_parser._createdescr.__dict__.__setitem__('stypy_function_name', 'format_parser._createdescr')
        format_parser._createdescr.__dict__.__setitem__('stypy_param_names_list', ['byteorder'])
        format_parser._createdescr.__dict__.__setitem__('stypy_varargs_param_name', None)
        format_parser._createdescr.__dict__.__setitem__('stypy_kwargs_param_name', None)
        format_parser._createdescr.__dict__.__setitem__('stypy_call_defaults', defaults)
        format_parser._createdescr.__dict__.__setitem__('stypy_call_varargs', varargs)
        format_parser._createdescr.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        format_parser._createdescr.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'format_parser._createdescr', ['byteorder'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_createdescr', localization, ['byteorder'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_createdescr(...)' code ##################

        
        # Assigning a Call to a Name (line 205):
        
        # Assigning a Call to a Name (line 205):
        
        # Call to dtype(...): (line 205)
        # Processing the call arguments (line 205)
        
        # Obtaining an instance of the builtin type 'dict' (line 205)
        dict_12543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 25), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 205)
        # Adding element type (key, value) (line 205)
        str_12544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 26), 'str', 'names')
        # Getting the type of 'self' (line 205)
        self_12545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 34), 'self', False)
        # Obtaining the member '_names' of a type (line 205)
        _names_12546 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 34), self_12545, '_names')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 205, 25), dict_12543, (str_12544, _names_12546))
        # Adding element type (key, value) (line 205)
        str_12547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 26), 'str', 'formats')
        # Getting the type of 'self' (line 206)
        self_12548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 36), 'self', False)
        # Obtaining the member '_f_formats' of a type (line 206)
        _f_formats_12549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 36), self_12548, '_f_formats')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 205, 25), dict_12543, (str_12547, _f_formats_12549))
        # Adding element type (key, value) (line 205)
        str_12550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 26), 'str', 'offsets')
        # Getting the type of 'self' (line 207)
        self_12551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 36), 'self', False)
        # Obtaining the member '_offsets' of a type (line 207)
        _offsets_12552 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 36), self_12551, '_offsets')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 205, 25), dict_12543, (str_12550, _offsets_12552))
        # Adding element type (key, value) (line 205)
        str_12553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 26), 'str', 'titles')
        # Getting the type of 'self' (line 208)
        self_12554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 35), 'self', False)
        # Obtaining the member '_titles' of a type (line 208)
        _titles_12555 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 35), self_12554, '_titles')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 205, 25), dict_12543, (str_12553, _titles_12555))
        
        # Processing the call keyword arguments (line 205)
        kwargs_12556 = {}
        # Getting the type of 'sb' (line 205)
        sb_12541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 16), 'sb', False)
        # Obtaining the member 'dtype' of a type (line 205)
        dtype_12542 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 16), sb_12541, 'dtype')
        # Calling dtype(args, kwargs) (line 205)
        dtype_call_result_12557 = invoke(stypy.reporting.localization.Localization(__file__, 205, 16), dtype_12542, *[dict_12543], **kwargs_12556)
        
        # Assigning a type to the variable 'descr' (line 205)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 8), 'descr', dtype_call_result_12557)
        
        # Type idiom detected: calculating its left and rigth part (line 209)
        # Getting the type of 'byteorder' (line 209)
        byteorder_12558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 8), 'byteorder')
        # Getting the type of 'None' (line 209)
        None_12559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 29), 'None')
        
        (may_be_12560, more_types_in_union_12561) = may_not_be_none(byteorder_12558, None_12559)

        if may_be_12560:

            if more_types_in_union_12561:
                # Runtime conditional SSA (line 209)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Subscript to a Name (line 210):
            
            # Assigning a Subscript to a Name (line 210):
            
            # Obtaining the type of the subscript
            
            # Obtaining the type of the subscript
            int_12562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 49), 'int')
            # Getting the type of 'byteorder' (line 210)
            byteorder_12563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 39), 'byteorder')
            # Obtaining the member '__getitem__' of a type (line 210)
            getitem___12564 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 39), byteorder_12563, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 210)
            subscript_call_result_12565 = invoke(stypy.reporting.localization.Localization(__file__, 210, 39), getitem___12564, int_12562)
            
            # Getting the type of '_byteorderconv' (line 210)
            _byteorderconv_12566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 24), '_byteorderconv')
            # Obtaining the member '__getitem__' of a type (line 210)
            getitem___12567 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 24), _byteorderconv_12566, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 210)
            subscript_call_result_12568 = invoke(stypy.reporting.localization.Localization(__file__, 210, 24), getitem___12567, subscript_call_result_12565)
            
            # Assigning a type to the variable 'byteorder' (line 210)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 12), 'byteorder', subscript_call_result_12568)
            
            # Assigning a Call to a Name (line 211):
            
            # Assigning a Call to a Name (line 211):
            
            # Call to newbyteorder(...): (line 211)
            # Processing the call arguments (line 211)
            # Getting the type of 'byteorder' (line 211)
            byteorder_12571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 39), 'byteorder', False)
            # Processing the call keyword arguments (line 211)
            kwargs_12572 = {}
            # Getting the type of 'descr' (line 211)
            descr_12569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 20), 'descr', False)
            # Obtaining the member 'newbyteorder' of a type (line 211)
            newbyteorder_12570 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 20), descr_12569, 'newbyteorder')
            # Calling newbyteorder(args, kwargs) (line 211)
            newbyteorder_call_result_12573 = invoke(stypy.reporting.localization.Localization(__file__, 211, 20), newbyteorder_12570, *[byteorder_12571], **kwargs_12572)
            
            # Assigning a type to the variable 'descr' (line 211)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 12), 'descr', newbyteorder_call_result_12573)

            if more_types_in_union_12561:
                # SSA join for if statement (line 209)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Name to a Attribute (line 213):
        
        # Assigning a Name to a Attribute (line 213):
        # Getting the type of 'descr' (line 213)
        descr_12574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 22), 'descr')
        # Getting the type of 'self' (line 213)
        self_12575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 8), 'self')
        # Setting the type of the member '_descr' of a type (line 213)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 8), self_12575, '_descr', descr_12574)
        
        # ################# End of '_createdescr(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_createdescr' in the type store
        # Getting the type of 'stypy_return_type' (line 204)
        stypy_return_type_12576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_12576)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_createdescr'
        return stypy_return_type_12576


# Assigning a type to the variable 'format_parser' (line 83)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 0), 'format_parser', format_parser)
# Declaration of the 'record' class
# Getting the type of 'nt' (line 215)
nt_12577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 13), 'nt')
# Obtaining the member 'void' of a type (line 215)
void_12578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 13), nt_12577, 'void')

class record(void_12578, ):
    str_12579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, (-1)), 'str', 'A data-type scalar that allows field access as attribute lookup.\n    ')
    
    # Assigning a Str to a Name (line 221):
    
    # Assigning a Str to a Name (line 222):

    @norecursion
    def stypy__repr__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__repr__'
        module_type_store = module_type_store.open_function_context('__repr__', 224, 4, False)
        # Assigning a type to the variable 'self' (line 225)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        record.stypy__repr__.__dict__.__setitem__('stypy_localization', localization)
        record.stypy__repr__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        record.stypy__repr__.__dict__.__setitem__('stypy_type_store', module_type_store)
        record.stypy__repr__.__dict__.__setitem__('stypy_function_name', 'record.__repr__')
        record.stypy__repr__.__dict__.__setitem__('stypy_param_names_list', [])
        record.stypy__repr__.__dict__.__setitem__('stypy_varargs_param_name', None)
        record.stypy__repr__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        record.stypy__repr__.__dict__.__setitem__('stypy_call_defaults', defaults)
        record.stypy__repr__.__dict__.__setitem__('stypy_call_varargs', varargs)
        record.stypy__repr__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        record.stypy__repr__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'record.__repr__', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to __str__(...): (line 225)
        # Processing the call keyword arguments (line 225)
        kwargs_12582 = {}
        # Getting the type of 'self' (line 225)
        self_12580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 15), 'self', False)
        # Obtaining the member '__str__' of a type (line 225)
        str___12581 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 15), self_12580, '__str__')
        # Calling __str__(args, kwargs) (line 225)
        str___call_result_12583 = invoke(stypy.reporting.localization.Localization(__file__, 225, 15), str___12581, *[], **kwargs_12582)
        
        # Assigning a type to the variable 'stypy_return_type' (line 225)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 8), 'stypy_return_type', str___call_result_12583)
        
        # ################# End of '__repr__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__repr__' in the type store
        # Getting the type of 'stypy_return_type' (line 224)
        stypy_return_type_12584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_12584)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__repr__'
        return stypy_return_type_12584


    @norecursion
    def stypy__str__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__str__'
        module_type_store = module_type_store.open_function_context('__str__', 227, 4, False)
        # Assigning a type to the variable 'self' (line 228)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        record.stypy__str__.__dict__.__setitem__('stypy_localization', localization)
        record.stypy__str__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        record.stypy__str__.__dict__.__setitem__('stypy_type_store', module_type_store)
        record.stypy__str__.__dict__.__setitem__('stypy_function_name', 'record.__str__')
        record.stypy__str__.__dict__.__setitem__('stypy_param_names_list', [])
        record.stypy__str__.__dict__.__setitem__('stypy_varargs_param_name', None)
        record.stypy__str__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        record.stypy__str__.__dict__.__setitem__('stypy_call_defaults', defaults)
        record.stypy__str__.__dict__.__setitem__('stypy_call_varargs', varargs)
        record.stypy__str__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        record.stypy__str__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'record.__str__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__str__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__str__(...)' code ##################

        
        # Call to str(...): (line 228)
        # Processing the call arguments (line 228)
        
        # Call to item(...): (line 228)
        # Processing the call keyword arguments (line 228)
        kwargs_12588 = {}
        # Getting the type of 'self' (line 228)
        self_12586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 19), 'self', False)
        # Obtaining the member 'item' of a type (line 228)
        item_12587 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 19), self_12586, 'item')
        # Calling item(args, kwargs) (line 228)
        item_call_result_12589 = invoke(stypy.reporting.localization.Localization(__file__, 228, 19), item_12587, *[], **kwargs_12588)
        
        # Processing the call keyword arguments (line 228)
        kwargs_12590 = {}
        # Getting the type of 'str' (line 228)
        str_12585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 15), 'str', False)
        # Calling str(args, kwargs) (line 228)
        str_call_result_12591 = invoke(stypy.reporting.localization.Localization(__file__, 228, 15), str_12585, *[item_call_result_12589], **kwargs_12590)
        
        # Assigning a type to the variable 'stypy_return_type' (line 228)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 8), 'stypy_return_type', str_call_result_12591)
        
        # ################# End of '__str__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__str__' in the type store
        # Getting the type of 'stypy_return_type' (line 227)
        stypy_return_type_12592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_12592)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__str__'
        return stypy_return_type_12592


    @norecursion
    def __getattribute__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__getattribute__'
        module_type_store = module_type_store.open_function_context('__getattribute__', 230, 4, False)
        # Assigning a type to the variable 'self' (line 231)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        record.__getattribute__.__dict__.__setitem__('stypy_localization', localization)
        record.__getattribute__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        record.__getattribute__.__dict__.__setitem__('stypy_type_store', module_type_store)
        record.__getattribute__.__dict__.__setitem__('stypy_function_name', 'record.__getattribute__')
        record.__getattribute__.__dict__.__setitem__('stypy_param_names_list', ['attr'])
        record.__getattribute__.__dict__.__setitem__('stypy_varargs_param_name', None)
        record.__getattribute__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        record.__getattribute__.__dict__.__setitem__('stypy_call_defaults', defaults)
        record.__getattribute__.__dict__.__setitem__('stypy_call_varargs', varargs)
        record.__getattribute__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        record.__getattribute__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'record.__getattribute__', ['attr'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__getattribute__', localization, ['attr'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__getattribute__(...)' code ##################

        
        
        # Getting the type of 'attr' (line 231)
        attr_12593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 11), 'attr')
        
        # Obtaining an instance of the builtin type 'list' (line 231)
        list_12594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 231)
        # Adding element type (line 231)
        str_12595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 20), 'str', 'setfield')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 231, 19), list_12594, str_12595)
        # Adding element type (line 231)
        str_12596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 32), 'str', 'getfield')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 231, 19), list_12594, str_12596)
        # Adding element type (line 231)
        str_12597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 44), 'str', 'dtype')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 231, 19), list_12594, str_12597)
        
        # Applying the binary operator 'in' (line 231)
        result_contains_12598 = python_operator(stypy.reporting.localization.Localization(__file__, 231, 11), 'in', attr_12593, list_12594)
        
        # Testing the type of an if condition (line 231)
        if_condition_12599 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 231, 8), result_contains_12598)
        # Assigning a type to the variable 'if_condition_12599' (line 231)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 8), 'if_condition_12599', if_condition_12599)
        # SSA begins for if statement (line 231)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to __getattribute__(...): (line 232)
        # Processing the call arguments (line 232)
        # Getting the type of 'self' (line 232)
        self_12603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 44), 'self', False)
        # Getting the type of 'attr' (line 232)
        attr_12604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 50), 'attr', False)
        # Processing the call keyword arguments (line 232)
        kwargs_12605 = {}
        # Getting the type of 'nt' (line 232)
        nt_12600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 19), 'nt', False)
        # Obtaining the member 'void' of a type (line 232)
        void_12601 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 19), nt_12600, 'void')
        # Obtaining the member '__getattribute__' of a type (line 232)
        getattribute___12602 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 19), void_12601, '__getattribute__')
        # Calling __getattribute__(args, kwargs) (line 232)
        getattribute___call_result_12606 = invoke(stypy.reporting.localization.Localization(__file__, 232, 19), getattribute___12602, *[self_12603, attr_12604], **kwargs_12605)
        
        # Assigning a type to the variable 'stypy_return_type' (line 232)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 12), 'stypy_return_type', getattribute___call_result_12606)
        # SSA join for if statement (line 231)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # SSA begins for try-except statement (line 233)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to __getattribute__(...): (line 234)
        # Processing the call arguments (line 234)
        # Getting the type of 'self' (line 234)
        self_12610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 44), 'self', False)
        # Getting the type of 'attr' (line 234)
        attr_12611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 50), 'attr', False)
        # Processing the call keyword arguments (line 234)
        kwargs_12612 = {}
        # Getting the type of 'nt' (line 234)
        nt_12607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 19), 'nt', False)
        # Obtaining the member 'void' of a type (line 234)
        void_12608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 19), nt_12607, 'void')
        # Obtaining the member '__getattribute__' of a type (line 234)
        getattribute___12609 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 19), void_12608, '__getattribute__')
        # Calling __getattribute__(args, kwargs) (line 234)
        getattribute___call_result_12613 = invoke(stypy.reporting.localization.Localization(__file__, 234, 19), getattribute___12609, *[self_12610, attr_12611], **kwargs_12612)
        
        # Assigning a type to the variable 'stypy_return_type' (line 234)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 12), 'stypy_return_type', getattribute___call_result_12613)
        # SSA branch for the except part of a try statement (line 233)
        # SSA branch for the except 'AttributeError' branch of a try statement (line 233)
        module_type_store.open_ssa_branch('except')
        pass
        # SSA join for try-except statement (line 233)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Attribute to a Name (line 237):
        
        # Assigning a Attribute to a Name (line 237):
        
        # Call to __getattribute__(...): (line 237)
        # Processing the call arguments (line 237)
        # Getting the type of 'self' (line 237)
        self_12617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 45), 'self', False)
        str_12618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 51), 'str', 'dtype')
        # Processing the call keyword arguments (line 237)
        kwargs_12619 = {}
        # Getting the type of 'nt' (line 237)
        nt_12614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 20), 'nt', False)
        # Obtaining the member 'void' of a type (line 237)
        void_12615 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 20), nt_12614, 'void')
        # Obtaining the member '__getattribute__' of a type (line 237)
        getattribute___12616 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 20), void_12615, '__getattribute__')
        # Calling __getattribute__(args, kwargs) (line 237)
        getattribute___call_result_12620 = invoke(stypy.reporting.localization.Localization(__file__, 237, 20), getattribute___12616, *[self_12617, str_12618], **kwargs_12619)
        
        # Obtaining the member 'fields' of a type (line 237)
        fields_12621 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 20), getattribute___call_result_12620, 'fields')
        # Assigning a type to the variable 'fielddict' (line 237)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 8), 'fielddict', fields_12621)
        
        # Assigning a Call to a Name (line 238):
        
        # Assigning a Call to a Name (line 238):
        
        # Call to get(...): (line 238)
        # Processing the call arguments (line 238)
        # Getting the type of 'attr' (line 238)
        attr_12624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 28), 'attr', False)
        # Getting the type of 'None' (line 238)
        None_12625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 34), 'None', False)
        # Processing the call keyword arguments (line 238)
        kwargs_12626 = {}
        # Getting the type of 'fielddict' (line 238)
        fielddict_12622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 14), 'fielddict', False)
        # Obtaining the member 'get' of a type (line 238)
        get_12623 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 14), fielddict_12622, 'get')
        # Calling get(args, kwargs) (line 238)
        get_call_result_12627 = invoke(stypy.reporting.localization.Localization(__file__, 238, 14), get_12623, *[attr_12624, None_12625], **kwargs_12626)
        
        # Assigning a type to the variable 'res' (line 238)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 8), 'res', get_call_result_12627)
        
        # Getting the type of 'res' (line 239)
        res_12628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 11), 'res')
        # Testing the type of an if condition (line 239)
        if_condition_12629 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 239, 8), res_12628)
        # Assigning a type to the variable 'if_condition_12629' (line 239)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 8), 'if_condition_12629', if_condition_12629)
        # SSA begins for if statement (line 239)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 240):
        
        # Assigning a Call to a Name (line 240):
        
        # Call to getfield(...): (line 240)
        
        # Obtaining the type of the subscript
        int_12632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 38), 'int')
        slice_12633 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 240, 33), None, int_12632, None)
        # Getting the type of 'res' (line 240)
        res_12634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 33), 'res', False)
        # Obtaining the member '__getitem__' of a type (line 240)
        getitem___12635 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 33), res_12634, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 240)
        subscript_call_result_12636 = invoke(stypy.reporting.localization.Localization(__file__, 240, 33), getitem___12635, slice_12633)
        
        # Processing the call keyword arguments (line 240)
        kwargs_12637 = {}
        # Getting the type of 'self' (line 240)
        self_12630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 18), 'self', False)
        # Obtaining the member 'getfield' of a type (line 240)
        getfield_12631 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 18), self_12630, 'getfield')
        # Calling getfield(args, kwargs) (line 240)
        getfield_call_result_12638 = invoke(stypy.reporting.localization.Localization(__file__, 240, 18), getfield_12631, *[subscript_call_result_12636], **kwargs_12637)
        
        # Assigning a type to the variable 'obj' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 12), 'obj', getfield_call_result_12638)
        
        
        # SSA begins for try-except statement (line 243)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Attribute to a Name (line 244):
        
        # Assigning a Attribute to a Name (line 244):
        # Getting the type of 'obj' (line 244)
        obj_12639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 21), 'obj')
        # Obtaining the member 'dtype' of a type (line 244)
        dtype_12640 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 21), obj_12639, 'dtype')
        # Assigning a type to the variable 'dt' (line 244)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 16), 'dt', dtype_12640)
        # SSA branch for the except part of a try statement (line 243)
        # SSA branch for the except 'AttributeError' branch of a try statement (line 243)
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'obj' (line 247)
        obj_12641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 23), 'obj')
        # Assigning a type to the variable 'stypy_return_type' (line 247)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 16), 'stypy_return_type', obj_12641)
        # SSA join for try-except statement (line 243)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'dt' (line 248)
        dt_12642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 15), 'dt')
        # Obtaining the member 'fields' of a type (line 248)
        fields_12643 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 15), dt_12642, 'fields')
        # Testing the type of an if condition (line 248)
        if_condition_12644 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 248, 12), fields_12643)
        # Assigning a type to the variable 'if_condition_12644' (line 248)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 12), 'if_condition_12644', if_condition_12644)
        # SSA begins for if statement (line 248)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to view(...): (line 249)
        # Processing the call arguments (line 249)
        
        # Obtaining an instance of the builtin type 'tuple' (line 249)
        tuple_12647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 33), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 249)
        # Adding element type (line 249)
        # Getting the type of 'self' (line 249)
        self_12648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 33), 'self', False)
        # Obtaining the member '__class__' of a type (line 249)
        class___12649 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 33), self_12648, '__class__')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 249, 33), tuple_12647, class___12649)
        # Adding element type (line 249)
        # Getting the type of 'obj' (line 249)
        obj_12650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 49), 'obj', False)
        # Obtaining the member 'dtype' of a type (line 249)
        dtype_12651 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 49), obj_12650, 'dtype')
        # Obtaining the member 'fields' of a type (line 249)
        fields_12652 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 49), dtype_12651, 'fields')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 249, 33), tuple_12647, fields_12652)
        
        # Processing the call keyword arguments (line 249)
        kwargs_12653 = {}
        # Getting the type of 'obj' (line 249)
        obj_12645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 23), 'obj', False)
        # Obtaining the member 'view' of a type (line 249)
        view_12646 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 23), obj_12645, 'view')
        # Calling view(args, kwargs) (line 249)
        view_call_result_12654 = invoke(stypy.reporting.localization.Localization(__file__, 249, 23), view_12646, *[tuple_12647], **kwargs_12653)
        
        # Assigning a type to the variable 'stypy_return_type' (line 249)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 16), 'stypy_return_type', view_call_result_12654)
        # SSA join for if statement (line 248)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'obj' (line 250)
        obj_12655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 19), 'obj')
        # Assigning a type to the variable 'stypy_return_type' (line 250)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 12), 'stypy_return_type', obj_12655)
        # SSA branch for the else part of an if statement (line 239)
        module_type_store.open_ssa_branch('else')
        
        # Call to AttributeError(...): (line 252)
        # Processing the call arguments (line 252)
        str_12657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 33), 'str', "'record' object has no attribute '%s'")
        # Getting the type of 'attr' (line 253)
        attr_12658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 39), 'attr', False)
        # Applying the binary operator '%' (line 252)
        result_mod_12659 = python_operator(stypy.reporting.localization.Localization(__file__, 252, 33), '%', str_12657, attr_12658)
        
        # Processing the call keyword arguments (line 252)
        kwargs_12660 = {}
        # Getting the type of 'AttributeError' (line 252)
        AttributeError_12656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 18), 'AttributeError', False)
        # Calling AttributeError(args, kwargs) (line 252)
        AttributeError_call_result_12661 = invoke(stypy.reporting.localization.Localization(__file__, 252, 18), AttributeError_12656, *[result_mod_12659], **kwargs_12660)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 252, 12), AttributeError_call_result_12661, 'raise parameter', BaseException)
        # SSA join for if statement (line 239)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__getattribute__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__getattribute__' in the type store
        # Getting the type of 'stypy_return_type' (line 230)
        stypy_return_type_12662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_12662)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__getattribute__'
        return stypy_return_type_12662


    @norecursion
    def __setattr__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__setattr__'
        module_type_store = module_type_store.open_function_context('__setattr__', 255, 4, False)
        # Assigning a type to the variable 'self' (line 256)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        record.__setattr__.__dict__.__setitem__('stypy_localization', localization)
        record.__setattr__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        record.__setattr__.__dict__.__setitem__('stypy_type_store', module_type_store)
        record.__setattr__.__dict__.__setitem__('stypy_function_name', 'record.__setattr__')
        record.__setattr__.__dict__.__setitem__('stypy_param_names_list', ['attr', 'val'])
        record.__setattr__.__dict__.__setitem__('stypy_varargs_param_name', None)
        record.__setattr__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        record.__setattr__.__dict__.__setitem__('stypy_call_defaults', defaults)
        record.__setattr__.__dict__.__setitem__('stypy_call_varargs', varargs)
        record.__setattr__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        record.__setattr__.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'record.__setattr__', ['attr', 'val'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__setattr__', localization, ['attr', 'val'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__setattr__(...)' code ##################

        
        
        # Getting the type of 'attr' (line 256)
        attr_12663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 11), 'attr')
        
        # Obtaining an instance of the builtin type 'list' (line 256)
        list_12664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 256)
        # Adding element type (line 256)
        str_12665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 20), 'str', 'setfield')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 256, 19), list_12664, str_12665)
        # Adding element type (line 256)
        str_12666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 32), 'str', 'getfield')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 256, 19), list_12664, str_12666)
        # Adding element type (line 256)
        str_12667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 44), 'str', 'dtype')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 256, 19), list_12664, str_12667)
        
        # Applying the binary operator 'in' (line 256)
        result_contains_12668 = python_operator(stypy.reporting.localization.Localization(__file__, 256, 11), 'in', attr_12663, list_12664)
        
        # Testing the type of an if condition (line 256)
        if_condition_12669 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 256, 8), result_contains_12668)
        # Assigning a type to the variable 'if_condition_12669' (line 256)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 8), 'if_condition_12669', if_condition_12669)
        # SSA begins for if statement (line 256)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to AttributeError(...): (line 257)
        # Processing the call arguments (line 257)
        str_12671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 33), 'str', "Cannot set '%s' attribute")
        # Getting the type of 'attr' (line 257)
        attr_12672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 63), 'attr', False)
        # Applying the binary operator '%' (line 257)
        result_mod_12673 = python_operator(stypy.reporting.localization.Localization(__file__, 257, 33), '%', str_12671, attr_12672)
        
        # Processing the call keyword arguments (line 257)
        kwargs_12674 = {}
        # Getting the type of 'AttributeError' (line 257)
        AttributeError_12670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 18), 'AttributeError', False)
        # Calling AttributeError(args, kwargs) (line 257)
        AttributeError_call_result_12675 = invoke(stypy.reporting.localization.Localization(__file__, 257, 18), AttributeError_12670, *[result_mod_12673], **kwargs_12674)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 257, 12), AttributeError_call_result_12675, 'raise parameter', BaseException)
        # SSA join for if statement (line 256)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Attribute to a Name (line 258):
        
        # Assigning a Attribute to a Name (line 258):
        
        # Call to __getattribute__(...): (line 258)
        # Processing the call arguments (line 258)
        # Getting the type of 'self' (line 258)
        self_12679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 45), 'self', False)
        str_12680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 51), 'str', 'dtype')
        # Processing the call keyword arguments (line 258)
        kwargs_12681 = {}
        # Getting the type of 'nt' (line 258)
        nt_12676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 20), 'nt', False)
        # Obtaining the member 'void' of a type (line 258)
        void_12677 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 20), nt_12676, 'void')
        # Obtaining the member '__getattribute__' of a type (line 258)
        getattribute___12678 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 20), void_12677, '__getattribute__')
        # Calling __getattribute__(args, kwargs) (line 258)
        getattribute___call_result_12682 = invoke(stypy.reporting.localization.Localization(__file__, 258, 20), getattribute___12678, *[self_12679, str_12680], **kwargs_12681)
        
        # Obtaining the member 'fields' of a type (line 258)
        fields_12683 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 20), getattribute___call_result_12682, 'fields')
        # Assigning a type to the variable 'fielddict' (line 258)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 8), 'fielddict', fields_12683)
        
        # Assigning a Call to a Name (line 259):
        
        # Assigning a Call to a Name (line 259):
        
        # Call to get(...): (line 259)
        # Processing the call arguments (line 259)
        # Getting the type of 'attr' (line 259)
        attr_12686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 28), 'attr', False)
        # Getting the type of 'None' (line 259)
        None_12687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 34), 'None', False)
        # Processing the call keyword arguments (line 259)
        kwargs_12688 = {}
        # Getting the type of 'fielddict' (line 259)
        fielddict_12684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 14), 'fielddict', False)
        # Obtaining the member 'get' of a type (line 259)
        get_12685 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 14), fielddict_12684, 'get')
        # Calling get(args, kwargs) (line 259)
        get_call_result_12689 = invoke(stypy.reporting.localization.Localization(__file__, 259, 14), get_12685, *[attr_12686, None_12687], **kwargs_12688)
        
        # Assigning a type to the variable 'res' (line 259)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 8), 'res', get_call_result_12689)
        
        # Getting the type of 'res' (line 260)
        res_12690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 11), 'res')
        # Testing the type of an if condition (line 260)
        if_condition_12691 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 260, 8), res_12690)
        # Assigning a type to the variable 'if_condition_12691' (line 260)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 8), 'if_condition_12691', if_condition_12691)
        # SSA begins for if statement (line 260)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to setfield(...): (line 261)
        # Processing the call arguments (line 261)
        # Getting the type of 'val' (line 261)
        val_12694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 33), 'val', False)
        
        # Obtaining the type of the subscript
        int_12695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 44), 'int')
        slice_12696 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 261, 39), None, int_12695, None)
        # Getting the type of 'res' (line 261)
        res_12697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 39), 'res', False)
        # Obtaining the member '__getitem__' of a type (line 261)
        getitem___12698 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 39), res_12697, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 261)
        subscript_call_result_12699 = invoke(stypy.reporting.localization.Localization(__file__, 261, 39), getitem___12698, slice_12696)
        
        # Processing the call keyword arguments (line 261)
        kwargs_12700 = {}
        # Getting the type of 'self' (line 261)
        self_12692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 19), 'self', False)
        # Obtaining the member 'setfield' of a type (line 261)
        setfield_12693 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 19), self_12692, 'setfield')
        # Calling setfield(args, kwargs) (line 261)
        setfield_call_result_12701 = invoke(stypy.reporting.localization.Localization(__file__, 261, 19), setfield_12693, *[val_12694, subscript_call_result_12699], **kwargs_12700)
        
        # Assigning a type to the variable 'stypy_return_type' (line 261)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 12), 'stypy_return_type', setfield_call_result_12701)
        # SSA branch for the else part of an if statement (line 260)
        module_type_store.open_ssa_branch('else')
        
        
        # Call to getattr(...): (line 263)
        # Processing the call arguments (line 263)
        # Getting the type of 'self' (line 263)
        self_12703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 23), 'self', False)
        # Getting the type of 'attr' (line 263)
        attr_12704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 29), 'attr', False)
        # Getting the type of 'None' (line 263)
        None_12705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 35), 'None', False)
        # Processing the call keyword arguments (line 263)
        kwargs_12706 = {}
        # Getting the type of 'getattr' (line 263)
        getattr_12702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 15), 'getattr', False)
        # Calling getattr(args, kwargs) (line 263)
        getattr_call_result_12707 = invoke(stypy.reporting.localization.Localization(__file__, 263, 15), getattr_12702, *[self_12703, attr_12704, None_12705], **kwargs_12706)
        
        # Testing the type of an if condition (line 263)
        if_condition_12708 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 263, 12), getattr_call_result_12707)
        # Assigning a type to the variable 'if_condition_12708' (line 263)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 12), 'if_condition_12708', if_condition_12708)
        # SSA begins for if statement (line 263)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to __setattr__(...): (line 264)
        # Processing the call arguments (line 264)
        # Getting the type of 'self' (line 264)
        self_12712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 43), 'self', False)
        # Getting the type of 'attr' (line 264)
        attr_12713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 49), 'attr', False)
        # Getting the type of 'val' (line 264)
        val_12714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 55), 'val', False)
        # Processing the call keyword arguments (line 264)
        kwargs_12715 = {}
        # Getting the type of 'nt' (line 264)
        nt_12709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 23), 'nt', False)
        # Obtaining the member 'void' of a type (line 264)
        void_12710 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 23), nt_12709, 'void')
        # Obtaining the member '__setattr__' of a type (line 264)
        setattr___12711 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 23), void_12710, '__setattr__')
        # Calling __setattr__(args, kwargs) (line 264)
        setattr___call_result_12716 = invoke(stypy.reporting.localization.Localization(__file__, 264, 23), setattr___12711, *[self_12712, attr_12713, val_12714], **kwargs_12715)
        
        # Assigning a type to the variable 'stypy_return_type' (line 264)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 16), 'stypy_return_type', setattr___call_result_12716)
        # SSA branch for the else part of an if statement (line 263)
        module_type_store.open_ssa_branch('else')
        
        # Call to AttributeError(...): (line 266)
        # Processing the call arguments (line 266)
        str_12718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 37), 'str', "'record' object has no attribute '%s'")
        # Getting the type of 'attr' (line 267)
        attr_12719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 43), 'attr', False)
        # Applying the binary operator '%' (line 266)
        result_mod_12720 = python_operator(stypy.reporting.localization.Localization(__file__, 266, 37), '%', str_12718, attr_12719)
        
        # Processing the call keyword arguments (line 266)
        kwargs_12721 = {}
        # Getting the type of 'AttributeError' (line 266)
        AttributeError_12717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 22), 'AttributeError', False)
        # Calling AttributeError(args, kwargs) (line 266)
        AttributeError_call_result_12722 = invoke(stypy.reporting.localization.Localization(__file__, 266, 22), AttributeError_12717, *[result_mod_12720], **kwargs_12721)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 266, 16), AttributeError_call_result_12722, 'raise parameter', BaseException)
        # SSA join for if statement (line 263)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 260)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__setattr__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__setattr__' in the type store
        # Getting the type of 'stypy_return_type' (line 255)
        stypy_return_type_12723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_12723)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__setattr__'
        return stypy_return_type_12723


    @norecursion
    def __getitem__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__getitem__'
        module_type_store = module_type_store.open_function_context('__getitem__', 269, 4, False)
        # Assigning a type to the variable 'self' (line 270)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        record.__getitem__.__dict__.__setitem__('stypy_localization', localization)
        record.__getitem__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        record.__getitem__.__dict__.__setitem__('stypy_type_store', module_type_store)
        record.__getitem__.__dict__.__setitem__('stypy_function_name', 'record.__getitem__')
        record.__getitem__.__dict__.__setitem__('stypy_param_names_list', ['indx'])
        record.__getitem__.__dict__.__setitem__('stypy_varargs_param_name', None)
        record.__getitem__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        record.__getitem__.__dict__.__setitem__('stypy_call_defaults', defaults)
        record.__getitem__.__dict__.__setitem__('stypy_call_varargs', varargs)
        record.__getitem__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        record.__getitem__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'record.__getitem__', ['indx'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__getitem__', localization, ['indx'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__getitem__(...)' code ##################

        
        # Assigning a Call to a Name (line 270):
        
        # Assigning a Call to a Name (line 270):
        
        # Call to __getitem__(...): (line 270)
        # Processing the call arguments (line 270)
        # Getting the type of 'self' (line 270)
        self_12727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 34), 'self', False)
        # Getting the type of 'indx' (line 270)
        indx_12728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 40), 'indx', False)
        # Processing the call keyword arguments (line 270)
        kwargs_12729 = {}
        # Getting the type of 'nt' (line 270)
        nt_12724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 14), 'nt', False)
        # Obtaining the member 'void' of a type (line 270)
        void_12725 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 14), nt_12724, 'void')
        # Obtaining the member '__getitem__' of a type (line 270)
        getitem___12726 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 14), void_12725, '__getitem__')
        # Calling __getitem__(args, kwargs) (line 270)
        getitem___call_result_12730 = invoke(stypy.reporting.localization.Localization(__file__, 270, 14), getitem___12726, *[self_12727, indx_12728], **kwargs_12729)
        
        # Assigning a type to the variable 'obj' (line 270)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 8), 'obj', getitem___call_result_12730)
        
        
        # Evaluating a boolean operation
        
        # Call to isinstance(...): (line 273)
        # Processing the call arguments (line 273)
        # Getting the type of 'obj' (line 273)
        obj_12732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 22), 'obj', False)
        # Getting the type of 'nt' (line 273)
        nt_12733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 27), 'nt', False)
        # Obtaining the member 'void' of a type (line 273)
        void_12734 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 27), nt_12733, 'void')
        # Processing the call keyword arguments (line 273)
        kwargs_12735 = {}
        # Getting the type of 'isinstance' (line 273)
        isinstance_12731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 273)
        isinstance_call_result_12736 = invoke(stypy.reporting.localization.Localization(__file__, 273, 11), isinstance_12731, *[obj_12732, void_12734], **kwargs_12735)
        
        # Getting the type of 'obj' (line 273)
        obj_12737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 40), 'obj')
        # Obtaining the member 'dtype' of a type (line 273)
        dtype_12738 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 40), obj_12737, 'dtype')
        # Obtaining the member 'fields' of a type (line 273)
        fields_12739 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 40), dtype_12738, 'fields')
        # Applying the binary operator 'and' (line 273)
        result_and_keyword_12740 = python_operator(stypy.reporting.localization.Localization(__file__, 273, 11), 'and', isinstance_call_result_12736, fields_12739)
        
        # Testing the type of an if condition (line 273)
        if_condition_12741 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 273, 8), result_and_keyword_12740)
        # Assigning a type to the variable 'if_condition_12741' (line 273)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 8), 'if_condition_12741', if_condition_12741)
        # SSA begins for if statement (line 273)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to view(...): (line 274)
        # Processing the call arguments (line 274)
        
        # Obtaining an instance of the builtin type 'tuple' (line 274)
        tuple_12744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 29), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 274)
        # Adding element type (line 274)
        # Getting the type of 'self' (line 274)
        self_12745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 29), 'self', False)
        # Obtaining the member '__class__' of a type (line 274)
        class___12746 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 29), self_12745, '__class__')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 274, 29), tuple_12744, class___12746)
        # Adding element type (line 274)
        # Getting the type of 'obj' (line 274)
        obj_12747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 45), 'obj', False)
        # Obtaining the member 'dtype' of a type (line 274)
        dtype_12748 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 45), obj_12747, 'dtype')
        # Obtaining the member 'fields' of a type (line 274)
        fields_12749 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 45), dtype_12748, 'fields')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 274, 29), tuple_12744, fields_12749)
        
        # Processing the call keyword arguments (line 274)
        kwargs_12750 = {}
        # Getting the type of 'obj' (line 274)
        obj_12742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 19), 'obj', False)
        # Obtaining the member 'view' of a type (line 274)
        view_12743 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 19), obj_12742, 'view')
        # Calling view(args, kwargs) (line 274)
        view_call_result_12751 = invoke(stypy.reporting.localization.Localization(__file__, 274, 19), view_12743, *[tuple_12744], **kwargs_12750)
        
        # Assigning a type to the variable 'stypy_return_type' (line 274)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 12), 'stypy_return_type', view_call_result_12751)
        # SSA branch for the else part of an if statement (line 273)
        module_type_store.open_ssa_branch('else')
        # Getting the type of 'obj' (line 277)
        obj_12752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 19), 'obj')
        # Assigning a type to the variable 'stypy_return_type' (line 277)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 12), 'stypy_return_type', obj_12752)
        # SSA join for if statement (line 273)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__getitem__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__getitem__' in the type store
        # Getting the type of 'stypy_return_type' (line 269)
        stypy_return_type_12753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_12753)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__getitem__'
        return stypy_return_type_12753


    @norecursion
    def pprint(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'pprint'
        module_type_store = module_type_store.open_function_context('pprint', 279, 4, False)
        # Assigning a type to the variable 'self' (line 280)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        record.pprint.__dict__.__setitem__('stypy_localization', localization)
        record.pprint.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        record.pprint.__dict__.__setitem__('stypy_type_store', module_type_store)
        record.pprint.__dict__.__setitem__('stypy_function_name', 'record.pprint')
        record.pprint.__dict__.__setitem__('stypy_param_names_list', [])
        record.pprint.__dict__.__setitem__('stypy_varargs_param_name', None)
        record.pprint.__dict__.__setitem__('stypy_kwargs_param_name', None)
        record.pprint.__dict__.__setitem__('stypy_call_defaults', defaults)
        record.pprint.__dict__.__setitem__('stypy_call_varargs', varargs)
        record.pprint.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        record.pprint.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'record.pprint', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'pprint', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'pprint(...)' code ##################

        str_12754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 8), 'str', 'Pretty-print all fields.')
        
        # Assigning a Attribute to a Name (line 282):
        
        # Assigning a Attribute to a Name (line 282):
        # Getting the type of 'self' (line 282)
        self_12755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 16), 'self')
        # Obtaining the member 'dtype' of a type (line 282)
        dtype_12756 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 16), self_12755, 'dtype')
        # Obtaining the member 'names' of a type (line 282)
        names_12757 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 16), dtype_12756, 'names')
        # Assigning a type to the variable 'names' (line 282)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 8), 'names', names_12757)
        
        # Assigning a Call to a Name (line 283):
        
        # Assigning a Call to a Name (line 283):
        
        # Call to max(...): (line 283)
        # Processing the call arguments (line 283)
        # Calculating generator expression
        module_type_store = module_type_store.open_function_context('list comprehension expression', 283, 21, True)
        # Calculating comprehension expression
        # Getting the type of 'names' (line 283)
        names_12763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 43), 'names', False)
        comprehension_12764 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 283, 21), names_12763)
        # Assigning a type to the variable 'name' (line 283)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 21), 'name', comprehension_12764)
        
        # Call to len(...): (line 283)
        # Processing the call arguments (line 283)
        # Getting the type of 'name' (line 283)
        name_12760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 25), 'name', False)
        # Processing the call keyword arguments (line 283)
        kwargs_12761 = {}
        # Getting the type of 'len' (line 283)
        len_12759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 21), 'len', False)
        # Calling len(args, kwargs) (line 283)
        len_call_result_12762 = invoke(stypy.reporting.localization.Localization(__file__, 283, 21), len_12759, *[name_12760], **kwargs_12761)
        
        list_12765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 21), 'list')
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 283, 21), list_12765, len_call_result_12762)
        # Processing the call keyword arguments (line 283)
        kwargs_12766 = {}
        # Getting the type of 'max' (line 283)
        max_12758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 17), 'max', False)
        # Calling max(args, kwargs) (line 283)
        max_call_result_12767 = invoke(stypy.reporting.localization.Localization(__file__, 283, 17), max_12758, *[list_12765], **kwargs_12766)
        
        # Assigning a type to the variable 'maxlen' (line 283)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 8), 'maxlen', max_call_result_12767)
        
        # Assigning a List to a Name (line 284):
        
        # Assigning a List to a Name (line 284):
        
        # Obtaining an instance of the builtin type 'list' (line 284)
        list_12768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 284)
        
        # Assigning a type to the variable 'rows' (line 284)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 8), 'rows', list_12768)
        
        # Assigning a BinOp to a Name (line 285):
        
        # Assigning a BinOp to a Name (line 285):
        str_12769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 14), 'str', '%% %ds: %%s')
        # Getting the type of 'maxlen' (line 285)
        maxlen_12770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 30), 'maxlen')
        # Applying the binary operator '%' (line 285)
        result_mod_12771 = python_operator(stypy.reporting.localization.Localization(__file__, 285, 14), '%', str_12769, maxlen_12770)
        
        # Assigning a type to the variable 'fmt' (line 285)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 8), 'fmt', result_mod_12771)
        
        # Getting the type of 'names' (line 286)
        names_12772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 20), 'names')
        # Testing the type of a for loop iterable (line 286)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 286, 8), names_12772)
        # Getting the type of the for loop variable (line 286)
        for_loop_var_12773 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 286, 8), names_12772)
        # Assigning a type to the variable 'name' (line 286)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 8), 'name', for_loop_var_12773)
        # SSA begins for a for statement (line 286)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to append(...): (line 287)
        # Processing the call arguments (line 287)
        # Getting the type of 'fmt' (line 287)
        fmt_12776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 24), 'fmt', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 287)
        tuple_12777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 31), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 287)
        # Adding element type (line 287)
        # Getting the type of 'name' (line 287)
        name_12778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 31), 'name', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 287, 31), tuple_12777, name_12778)
        # Adding element type (line 287)
        
        # Call to getattr(...): (line 287)
        # Processing the call arguments (line 287)
        # Getting the type of 'self' (line 287)
        self_12780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 45), 'self', False)
        # Getting the type of 'name' (line 287)
        name_12781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 51), 'name', False)
        # Processing the call keyword arguments (line 287)
        kwargs_12782 = {}
        # Getting the type of 'getattr' (line 287)
        getattr_12779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 37), 'getattr', False)
        # Calling getattr(args, kwargs) (line 287)
        getattr_call_result_12783 = invoke(stypy.reporting.localization.Localization(__file__, 287, 37), getattr_12779, *[self_12780, name_12781], **kwargs_12782)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 287, 31), tuple_12777, getattr_call_result_12783)
        
        # Applying the binary operator '%' (line 287)
        result_mod_12784 = python_operator(stypy.reporting.localization.Localization(__file__, 287, 24), '%', fmt_12776, tuple_12777)
        
        # Processing the call keyword arguments (line 287)
        kwargs_12785 = {}
        # Getting the type of 'rows' (line 287)
        rows_12774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 12), 'rows', False)
        # Obtaining the member 'append' of a type (line 287)
        append_12775 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 12), rows_12774, 'append')
        # Calling append(args, kwargs) (line 287)
        append_call_result_12786 = invoke(stypy.reporting.localization.Localization(__file__, 287, 12), append_12775, *[result_mod_12784], **kwargs_12785)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to join(...): (line 288)
        # Processing the call arguments (line 288)
        # Getting the type of 'rows' (line 288)
        rows_12789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 25), 'rows', False)
        # Processing the call keyword arguments (line 288)
        kwargs_12790 = {}
        str_12787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 15), 'str', '\n')
        # Obtaining the member 'join' of a type (line 288)
        join_12788 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 15), str_12787, 'join')
        # Calling join(args, kwargs) (line 288)
        join_call_result_12791 = invoke(stypy.reporting.localization.Localization(__file__, 288, 15), join_12788, *[rows_12789], **kwargs_12790)
        
        # Assigning a type to the variable 'stypy_return_type' (line 288)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 8), 'stypy_return_type', join_call_result_12791)
        
        # ################# End of 'pprint(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'pprint' in the type store
        # Getting the type of 'stypy_return_type' (line 279)
        stypy_return_type_12792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_12792)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'pprint'
        return stypy_return_type_12792


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 215, 0, False)
        # Assigning a type to the variable 'self' (line 216)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'record.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'record' (line 215)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 0), 'record', record)

# Assigning a Str to a Name (line 221):
str_12793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 15), 'str', 'record')
# Getting the type of 'record'
record_12794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'record')
# Setting the type of the member '__name__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), record_12794, '__name__', str_12793)

# Assigning a Str to a Name (line 222):
str_12795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 17), 'str', 'numpy')
# Getting the type of 'record'
record_12796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'record')
# Setting the type of the member '__module__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), record_12796, '__module__', str_12795)
# Declaration of the 'recarray' class
# Getting the type of 'ndarray' (line 298)
ndarray_12797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 15), 'ndarray')

class recarray(ndarray_12797, ):
    str_12798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 402, (-1)), 'str', "Construct an ndarray that allows field access using attributes.\n\n    Arrays may have a data-types containing fields, analogous\n    to columns in a spread sheet.  An example is ``[(x, int), (y, float)]``,\n    where each entry in the array is a pair of ``(int, float)``.  Normally,\n    these attributes are accessed using dictionary lookups such as ``arr['x']``\n    and ``arr['y']``.  Record arrays allow the fields to be accessed as members\n    of the array, using ``arr.x`` and ``arr.y``.\n\n    Parameters\n    ----------\n    shape : tuple\n        Shape of output array.\n    dtype : data-type, optional\n        The desired data-type.  By default, the data-type is determined\n        from `formats`, `names`, `titles`, `aligned` and `byteorder`.\n    formats : list of data-types, optional\n        A list containing the data-types for the different columns, e.g.\n        ``['i4', 'f8', 'i4']``.  `formats` does *not* support the new\n        convention of using types directly, i.e. ``(int, float, int)``.\n        Note that `formats` must be a list, not a tuple.\n        Given that `formats` is somewhat limited, we recommend specifying\n        `dtype` instead.\n    names : tuple of str, optional\n        The name of each column, e.g. ``('x', 'y', 'z')``.\n    buf : buffer, optional\n        By default, a new array is created of the given shape and data-type.\n        If `buf` is specified and is an object exposing the buffer interface,\n        the array will use the memory from the existing buffer.  In this case,\n        the `offset` and `strides` keywords are available.\n\n    Other Parameters\n    ----------------\n    titles : tuple of str, optional\n        Aliases for column names.  For example, if `names` were\n        ``('x', 'y', 'z')`` and `titles` is\n        ``('x_coordinate', 'y_coordinate', 'z_coordinate')``, then\n        ``arr['x']`` is equivalent to both ``arr.x`` and ``arr.x_coordinate``.\n    byteorder : {'<', '>', '='}, optional\n        Byte-order for all fields.\n    aligned : bool, optional\n        Align the fields in memory as the C-compiler would.\n    strides : tuple of ints, optional\n        Buffer (`buf`) is interpreted according to these strides (strides\n        define how many bytes each array element, row, column, etc.\n        occupy in memory).\n    offset : int, optional\n        Start reading buffer (`buf`) from this offset onwards.\n    order : {'C', 'F'}, optional\n        Row-major (C-style) or column-major (Fortran-style) order.\n\n    Returns\n    -------\n    rec : recarray\n        Empty array of the given shape and type.\n\n    See Also\n    --------\n    rec.fromrecords : Construct a record array from data.\n    record : fundamental data-type for `recarray`.\n    format_parser : determine a data-type from formats, names, titles.\n\n    Notes\n    -----\n    This constructor can be compared to ``empty``: it creates a new record\n    array but does not fill it with data.  To create a record array from data,\n    use one of the following methods:\n\n    1. Create a standard ndarray and convert it to a record array,\n       using ``arr.view(np.recarray)``\n    2. Use the `buf` keyword.\n    3. Use `np.rec.fromrecords`.\n\n    Examples\n    --------\n    Create an array with two fields, ``x`` and ``y``:\n\n    >>> x = np.array([(1.0, 2), (3.0, 4)], dtype=[('x', float), ('y', int)])\n    >>> x\n    array([(1.0, 2), (3.0, 4)],\n          dtype=[('x', '<f8'), ('y', '<i4')])\n\n    >>> x['x']\n    array([ 1.,  3.])\n\n    View the array as a record array:\n\n    >>> x = x.view(np.recarray)\n\n    >>> x.x\n    array([ 1.,  3.])\n\n    >>> x.y\n    array([2, 4])\n\n    Create a new, empty record array:\n\n    >>> np.recarray((2,),\n    ... dtype=[('x', int), ('y', float), ('z', int)]) #doctest: +SKIP\n    rec.array([(-1073741821, 1.2249118382103472e-301, 24547520),\n           (3471280, 1.2134086255804012e-316, 0)],\n          dtype=[('x', '<i4'), ('y', '<f8'), ('z', '<i4')])\n\n    ")
    
    # Assigning a Str to a Name (line 406):
    
    # Assigning a Str to a Name (line 407):

    @norecursion
    def __new__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 409)
        None_12799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 38), 'None')
        # Getting the type of 'None' (line 409)
        None_12800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 48), 'None')
        int_12801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 61), 'int')
        # Getting the type of 'None' (line 409)
        None_12802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 72), 'None')
        # Getting the type of 'None' (line 410)
        None_12803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 24), 'None')
        # Getting the type of 'None' (line 410)
        None_12804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 36), 'None')
        # Getting the type of 'None' (line 410)
        None_12805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 49), 'None')
        # Getting the type of 'None' (line 411)
        None_12806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 26), 'None')
        # Getting the type of 'False' (line 411)
        False_12807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 40), 'False')
        str_12808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 53), 'str', 'C')
        defaults = [None_12799, None_12800, int_12801, None_12802, None_12803, None_12804, None_12805, None_12806, False_12807, str_12808]
        # Create a new context for function '__new__'
        module_type_store = module_type_store.open_function_context('__new__', 409, 4, False)
        # Assigning a type to the variable 'self' (line 410)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 410, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        recarray.__new__.__dict__.__setitem__('stypy_localization', localization)
        recarray.__new__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        recarray.__new__.__dict__.__setitem__('stypy_type_store', module_type_store)
        recarray.__new__.__dict__.__setitem__('stypy_function_name', 'recarray.__new__')
        recarray.__new__.__dict__.__setitem__('stypy_param_names_list', ['shape', 'dtype', 'buf', 'offset', 'strides', 'formats', 'names', 'titles', 'byteorder', 'aligned', 'order'])
        recarray.__new__.__dict__.__setitem__('stypy_varargs_param_name', None)
        recarray.__new__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        recarray.__new__.__dict__.__setitem__('stypy_call_defaults', defaults)
        recarray.__new__.__dict__.__setitem__('stypy_call_varargs', varargs)
        recarray.__new__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        recarray.__new__.__dict__.__setitem__('stypy_declared_arg_number', 12)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'recarray.__new__', ['shape', 'dtype', 'buf', 'offset', 'strides', 'formats', 'names', 'titles', 'byteorder', 'aligned', 'order'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__new__', localization, ['shape', 'dtype', 'buf', 'offset', 'strides', 'formats', 'names', 'titles', 'byteorder', 'aligned', 'order'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__new__(...)' code ##################

        
        # Type idiom detected: calculating its left and rigth part (line 413)
        # Getting the type of 'dtype' (line 413)
        dtype_12809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 8), 'dtype')
        # Getting the type of 'None' (line 413)
        None_12810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 24), 'None')
        
        (may_be_12811, more_types_in_union_12812) = may_not_be_none(dtype_12809, None_12810)

        if may_be_12811:

            if more_types_in_union_12812:
                # Runtime conditional SSA (line 413)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 414):
            
            # Assigning a Call to a Name (line 414):
            
            # Call to dtype(...): (line 414)
            # Processing the call arguments (line 414)
            # Getting the type of 'dtype' (line 414)
            dtype_12815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 29), 'dtype', False)
            # Processing the call keyword arguments (line 414)
            kwargs_12816 = {}
            # Getting the type of 'sb' (line 414)
            sb_12813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 20), 'sb', False)
            # Obtaining the member 'dtype' of a type (line 414)
            dtype_12814 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 414, 20), sb_12813, 'dtype')
            # Calling dtype(args, kwargs) (line 414)
            dtype_call_result_12817 = invoke(stypy.reporting.localization.Localization(__file__, 414, 20), dtype_12814, *[dtype_12815], **kwargs_12816)
            
            # Assigning a type to the variable 'descr' (line 414)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 414, 12), 'descr', dtype_call_result_12817)

            if more_types_in_union_12812:
                # Runtime conditional SSA for else branch (line 413)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_12811) or more_types_in_union_12812):
            
            # Assigning a Attribute to a Name (line 416):
            
            # Assigning a Attribute to a Name (line 416):
            
            # Call to format_parser(...): (line 416)
            # Processing the call arguments (line 416)
            # Getting the type of 'formats' (line 416)
            formats_12819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 34), 'formats', False)
            # Getting the type of 'names' (line 416)
            names_12820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 43), 'names', False)
            # Getting the type of 'titles' (line 416)
            titles_12821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 50), 'titles', False)
            # Getting the type of 'aligned' (line 416)
            aligned_12822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 58), 'aligned', False)
            # Getting the type of 'byteorder' (line 416)
            byteorder_12823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 67), 'byteorder', False)
            # Processing the call keyword arguments (line 416)
            kwargs_12824 = {}
            # Getting the type of 'format_parser' (line 416)
            format_parser_12818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 20), 'format_parser', False)
            # Calling format_parser(args, kwargs) (line 416)
            format_parser_call_result_12825 = invoke(stypy.reporting.localization.Localization(__file__, 416, 20), format_parser_12818, *[formats_12819, names_12820, titles_12821, aligned_12822, byteorder_12823], **kwargs_12824)
            
            # Obtaining the member '_descr' of a type (line 416)
            _descr_12826 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 416, 20), format_parser_call_result_12825, '_descr')
            # Assigning a type to the variable 'descr' (line 416)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 416, 12), 'descr', _descr_12826)

            if (may_be_12811 and more_types_in_union_12812):
                # SSA join for if statement (line 413)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 418)
        # Getting the type of 'buf' (line 418)
        buf_12827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 11), 'buf')
        # Getting the type of 'None' (line 418)
        None_12828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 18), 'None')
        
        (may_be_12829, more_types_in_union_12830) = may_be_none(buf_12827, None_12828)

        if may_be_12829:

            if more_types_in_union_12830:
                # Runtime conditional SSA (line 418)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 419):
            
            # Assigning a Call to a Name (line 419):
            
            # Call to __new__(...): (line 419)
            # Processing the call arguments (line 419)
            # Getting the type of 'subtype' (line 419)
            subtype_12833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 35), 'subtype', False)
            # Getting the type of 'shape' (line 419)
            shape_12834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 44), 'shape', False)
            
            # Obtaining an instance of the builtin type 'tuple' (line 419)
            tuple_12835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 419, 52), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 419)
            # Adding element type (line 419)
            # Getting the type of 'record' (line 419)
            record_12836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 52), 'record', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 419, 52), tuple_12835, record_12836)
            # Adding element type (line 419)
            # Getting the type of 'descr' (line 419)
            descr_12837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 60), 'descr', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 419, 52), tuple_12835, descr_12837)
            
            # Processing the call keyword arguments (line 419)
            # Getting the type of 'order' (line 419)
            order_12838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 74), 'order', False)
            keyword_12839 = order_12838
            kwargs_12840 = {'order': keyword_12839}
            # Getting the type of 'ndarray' (line 419)
            ndarray_12831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 19), 'ndarray', False)
            # Obtaining the member '__new__' of a type (line 419)
            new___12832 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 419, 19), ndarray_12831, '__new__')
            # Calling __new__(args, kwargs) (line 419)
            new___call_result_12841 = invoke(stypy.reporting.localization.Localization(__file__, 419, 19), new___12832, *[subtype_12833, shape_12834, tuple_12835], **kwargs_12840)
            
            # Assigning a type to the variable 'self' (line 419)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 419, 12), 'self', new___call_result_12841)

            if more_types_in_union_12830:
                # Runtime conditional SSA for else branch (line 418)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_12829) or more_types_in_union_12830):
            
            # Assigning a Call to a Name (line 421):
            
            # Assigning a Call to a Name (line 421):
            
            # Call to __new__(...): (line 421)
            # Processing the call arguments (line 421)
            # Getting the type of 'subtype' (line 421)
            subtype_12844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 35), 'subtype', False)
            # Getting the type of 'shape' (line 421)
            shape_12845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 44), 'shape', False)
            
            # Obtaining an instance of the builtin type 'tuple' (line 421)
            tuple_12846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 421, 52), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 421)
            # Adding element type (line 421)
            # Getting the type of 'record' (line 421)
            record_12847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 52), 'record', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 421, 52), tuple_12846, record_12847)
            # Adding element type (line 421)
            # Getting the type of 'descr' (line 421)
            descr_12848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 60), 'descr', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 421, 52), tuple_12846, descr_12848)
            
            # Processing the call keyword arguments (line 421)
            # Getting the type of 'buf' (line 422)
            buf_12849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 45), 'buf', False)
            keyword_12850 = buf_12849
            # Getting the type of 'offset' (line 422)
            offset_12851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 57), 'offset', False)
            keyword_12852 = offset_12851
            # Getting the type of 'strides' (line 423)
            strides_12853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 46), 'strides', False)
            keyword_12854 = strides_12853
            # Getting the type of 'order' (line 423)
            order_12855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 61), 'order', False)
            keyword_12856 = order_12855
            kwargs_12857 = {'buffer': keyword_12850, 'strides': keyword_12854, 'order': keyword_12856, 'offset': keyword_12852}
            # Getting the type of 'ndarray' (line 421)
            ndarray_12842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 19), 'ndarray', False)
            # Obtaining the member '__new__' of a type (line 421)
            new___12843 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 421, 19), ndarray_12842, '__new__')
            # Calling __new__(args, kwargs) (line 421)
            new___call_result_12858 = invoke(stypy.reporting.localization.Localization(__file__, 421, 19), new___12843, *[subtype_12844, shape_12845, tuple_12846], **kwargs_12857)
            
            # Assigning a type to the variable 'self' (line 421)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 421, 12), 'self', new___call_result_12858)

            if (may_be_12829 and more_types_in_union_12830):
                # SSA join for if statement (line 418)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 'self' (line 424)
        self_12859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 15), 'self')
        # Assigning a type to the variable 'stypy_return_type' (line 424)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 424, 8), 'stypy_return_type', self_12859)
        
        # ################# End of '__new__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__new__' in the type store
        # Getting the type of 'stypy_return_type' (line 409)
        stypy_return_type_12860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_12860)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__new__'
        return stypy_return_type_12860


    @norecursion
    def __array_finalize__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__array_finalize__'
        module_type_store = module_type_store.open_function_context('__array_finalize__', 426, 4, False)
        # Assigning a type to the variable 'self' (line 427)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 427, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        recarray.__array_finalize__.__dict__.__setitem__('stypy_localization', localization)
        recarray.__array_finalize__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        recarray.__array_finalize__.__dict__.__setitem__('stypy_type_store', module_type_store)
        recarray.__array_finalize__.__dict__.__setitem__('stypy_function_name', 'recarray.__array_finalize__')
        recarray.__array_finalize__.__dict__.__setitem__('stypy_param_names_list', ['obj'])
        recarray.__array_finalize__.__dict__.__setitem__('stypy_varargs_param_name', None)
        recarray.__array_finalize__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        recarray.__array_finalize__.__dict__.__setitem__('stypy_call_defaults', defaults)
        recarray.__array_finalize__.__dict__.__setitem__('stypy_call_varargs', varargs)
        recarray.__array_finalize__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        recarray.__array_finalize__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'recarray.__array_finalize__', ['obj'], None, None, defaults, varargs, kwargs)

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

        
        
        # Getting the type of 'self' (line 427)
        self_12861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 11), 'self')
        # Obtaining the member 'dtype' of a type (line 427)
        dtype_12862 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 427, 11), self_12861, 'dtype')
        # Obtaining the member 'type' of a type (line 427)
        type_12863 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 427, 11), dtype_12862, 'type')
        # Getting the type of 'record' (line 427)
        record_12864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 34), 'record')
        # Applying the binary operator 'isnot' (line 427)
        result_is_not_12865 = python_operator(stypy.reporting.localization.Localization(__file__, 427, 11), 'isnot', type_12863, record_12864)
        
        # Testing the type of an if condition (line 427)
        if_condition_12866 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 427, 8), result_is_not_12865)
        # Assigning a type to the variable 'if_condition_12866' (line 427)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 427, 8), 'if_condition_12866', if_condition_12866)
        # SSA begins for if statement (line 427)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Attribute (line 430):
        
        # Assigning a Attribute to a Attribute (line 430):
        # Getting the type of 'self' (line 430)
        self_12867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 25), 'self')
        # Obtaining the member 'dtype' of a type (line 430)
        dtype_12868 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 25), self_12867, 'dtype')
        # Getting the type of 'self' (line 430)
        self_12869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 12), 'self')
        # Setting the type of the member 'dtype' of a type (line 430)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 12), self_12869, 'dtype', dtype_12868)
        # SSA join for if statement (line 427)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__array_finalize__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__array_finalize__' in the type store
        # Getting the type of 'stypy_return_type' (line 426)
        stypy_return_type_12870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_12870)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__array_finalize__'
        return stypy_return_type_12870


    @norecursion
    def __getattribute__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__getattribute__'
        module_type_store = module_type_store.open_function_context('__getattribute__', 432, 4, False)
        # Assigning a type to the variable 'self' (line 433)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 433, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        recarray.__getattribute__.__dict__.__setitem__('stypy_localization', localization)
        recarray.__getattribute__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        recarray.__getattribute__.__dict__.__setitem__('stypy_type_store', module_type_store)
        recarray.__getattribute__.__dict__.__setitem__('stypy_function_name', 'recarray.__getattribute__')
        recarray.__getattribute__.__dict__.__setitem__('stypy_param_names_list', ['attr'])
        recarray.__getattribute__.__dict__.__setitem__('stypy_varargs_param_name', None)
        recarray.__getattribute__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        recarray.__getattribute__.__dict__.__setitem__('stypy_call_defaults', defaults)
        recarray.__getattribute__.__dict__.__setitem__('stypy_call_varargs', varargs)
        recarray.__getattribute__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        recarray.__getattribute__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'recarray.__getattribute__', ['attr'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__getattribute__', localization, ['attr'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__getattribute__(...)' code ##################

        
        
        # SSA begins for try-except statement (line 436)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to __getattribute__(...): (line 437)
        # Processing the call arguments (line 437)
        # Getting the type of 'self' (line 437)
        self_12873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 43), 'self', False)
        # Getting the type of 'attr' (line 437)
        attr_12874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 49), 'attr', False)
        # Processing the call keyword arguments (line 437)
        kwargs_12875 = {}
        # Getting the type of 'object' (line 437)
        object_12871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 19), 'object', False)
        # Obtaining the member '__getattribute__' of a type (line 437)
        getattribute___12872 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 19), object_12871, '__getattribute__')
        # Calling __getattribute__(args, kwargs) (line 437)
        getattribute___call_result_12876 = invoke(stypy.reporting.localization.Localization(__file__, 437, 19), getattribute___12872, *[self_12873, attr_12874], **kwargs_12875)
        
        # Assigning a type to the variable 'stypy_return_type' (line 437)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 437, 12), 'stypy_return_type', getattribute___call_result_12876)
        # SSA branch for the except part of a try statement (line 436)
        # SSA branch for the except 'AttributeError' branch of a try statement (line 436)
        module_type_store.open_ssa_branch('except')
        pass
        # SSA join for try-except statement (line 436)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Attribute to a Name (line 442):
        
        # Assigning a Attribute to a Name (line 442):
        
        # Call to __getattribute__(...): (line 442)
        # Processing the call arguments (line 442)
        # Getting the type of 'self' (line 442)
        self_12879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 45), 'self', False)
        str_12880 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 442, 51), 'str', 'dtype')
        # Processing the call keyword arguments (line 442)
        kwargs_12881 = {}
        # Getting the type of 'ndarray' (line 442)
        ndarray_12877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 20), 'ndarray', False)
        # Obtaining the member '__getattribute__' of a type (line 442)
        getattribute___12878 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 442, 20), ndarray_12877, '__getattribute__')
        # Calling __getattribute__(args, kwargs) (line 442)
        getattribute___call_result_12882 = invoke(stypy.reporting.localization.Localization(__file__, 442, 20), getattribute___12878, *[self_12879, str_12880], **kwargs_12881)
        
        # Obtaining the member 'fields' of a type (line 442)
        fields_12883 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 442, 20), getattribute___call_result_12882, 'fields')
        # Assigning a type to the variable 'fielddict' (line 442)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 442, 8), 'fielddict', fields_12883)
        
        
        # SSA begins for try-except statement (line 443)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Subscript to a Name (line 444):
        
        # Assigning a Subscript to a Name (line 444):
        
        # Obtaining the type of the subscript
        int_12884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 444, 35), 'int')
        slice_12885 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 444, 18), None, int_12884, None)
        
        # Obtaining the type of the subscript
        # Getting the type of 'attr' (line 444)
        attr_12886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 28), 'attr')
        # Getting the type of 'fielddict' (line 444)
        fielddict_12887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 18), 'fielddict')
        # Obtaining the member '__getitem__' of a type (line 444)
        getitem___12888 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 444, 18), fielddict_12887, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 444)
        subscript_call_result_12889 = invoke(stypy.reporting.localization.Localization(__file__, 444, 18), getitem___12888, attr_12886)
        
        # Obtaining the member '__getitem__' of a type (line 444)
        getitem___12890 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 444, 18), subscript_call_result_12889, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 444)
        subscript_call_result_12891 = invoke(stypy.reporting.localization.Localization(__file__, 444, 18), getitem___12890, slice_12885)
        
        # Assigning a type to the variable 'res' (line 444)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 12), 'res', subscript_call_result_12891)
        # SSA branch for the except part of a try statement (line 443)
        # SSA branch for the except 'Tuple' branch of a try statement (line 443)
        module_type_store.open_ssa_branch('except')
        
        # Call to AttributeError(...): (line 446)
        # Processing the call arguments (line 446)
        str_12893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 446, 33), 'str', 'recarray has no attribute %s')
        # Getting the type of 'attr' (line 446)
        attr_12894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 66), 'attr', False)
        # Applying the binary operator '%' (line 446)
        result_mod_12895 = python_operator(stypy.reporting.localization.Localization(__file__, 446, 33), '%', str_12893, attr_12894)
        
        # Processing the call keyword arguments (line 446)
        kwargs_12896 = {}
        # Getting the type of 'AttributeError' (line 446)
        AttributeError_12892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 18), 'AttributeError', False)
        # Calling AttributeError(args, kwargs) (line 446)
        AttributeError_call_result_12897 = invoke(stypy.reporting.localization.Localization(__file__, 446, 18), AttributeError_12892, *[result_mod_12895], **kwargs_12896)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 446, 12), AttributeError_call_result_12897, 'raise parameter', BaseException)
        # SSA join for try-except statement (line 443)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 447):
        
        # Assigning a Call to a Name (line 447):
        
        # Call to getfield(...): (line 447)
        # Getting the type of 'res' (line 447)
        res_12900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 29), 'res', False)
        # Processing the call keyword arguments (line 447)
        kwargs_12901 = {}
        # Getting the type of 'self' (line 447)
        self_12898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 14), 'self', False)
        # Obtaining the member 'getfield' of a type (line 447)
        getfield_12899 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 447, 14), self_12898, 'getfield')
        # Calling getfield(args, kwargs) (line 447)
        getfield_call_result_12902 = invoke(stypy.reporting.localization.Localization(__file__, 447, 14), getfield_12899, *[res_12900], **kwargs_12901)
        
        # Assigning a type to the variable 'obj' (line 447)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 447, 8), 'obj', getfield_call_result_12902)
        
        # Getting the type of 'obj' (line 455)
        obj_12903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 11), 'obj')
        # Obtaining the member 'dtype' of a type (line 455)
        dtype_12904 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 455, 11), obj_12903, 'dtype')
        # Obtaining the member 'fields' of a type (line 455)
        fields_12905 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 455, 11), dtype_12904, 'fields')
        # Testing the type of an if condition (line 455)
        if_condition_12906 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 455, 8), fields_12905)
        # Assigning a type to the variable 'if_condition_12906' (line 455)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 455, 8), 'if_condition_12906', if_condition_12906)
        # SSA begins for if statement (line 455)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Call to issubclass(...): (line 456)
        # Processing the call arguments (line 456)
        # Getting the type of 'obj' (line 456)
        obj_12908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 26), 'obj', False)
        # Obtaining the member 'dtype' of a type (line 456)
        dtype_12909 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 456, 26), obj_12908, 'dtype')
        # Obtaining the member 'type' of a type (line 456)
        type_12910 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 456, 26), dtype_12909, 'type')
        # Getting the type of 'nt' (line 456)
        nt_12911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 42), 'nt', False)
        # Obtaining the member 'void' of a type (line 456)
        void_12912 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 456, 42), nt_12911, 'void')
        # Processing the call keyword arguments (line 456)
        kwargs_12913 = {}
        # Getting the type of 'issubclass' (line 456)
        issubclass_12907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 15), 'issubclass', False)
        # Calling issubclass(args, kwargs) (line 456)
        issubclass_call_result_12914 = invoke(stypy.reporting.localization.Localization(__file__, 456, 15), issubclass_12907, *[type_12910, void_12912], **kwargs_12913)
        
        # Testing the type of an if condition (line 456)
        if_condition_12915 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 456, 12), issubclass_call_result_12914)
        # Assigning a type to the variable 'if_condition_12915' (line 456)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 12), 'if_condition_12915', if_condition_12915)
        # SSA begins for if statement (line 456)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to view(...): (line 457)
        # Processing the call keyword arguments (line 457)
        
        # Obtaining an instance of the builtin type 'tuple' (line 457)
        tuple_12918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 457, 39), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 457)
        # Adding element type (line 457)
        # Getting the type of 'self' (line 457)
        self_12919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 39), 'self', False)
        # Obtaining the member 'dtype' of a type (line 457)
        dtype_12920 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 457, 39), self_12919, 'dtype')
        # Obtaining the member 'type' of a type (line 457)
        type_12921 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 457, 39), dtype_12920, 'type')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 457, 39), tuple_12918, type_12921)
        # Adding element type (line 457)
        # Getting the type of 'obj' (line 457)
        obj_12922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 56), 'obj', False)
        # Obtaining the member 'dtype' of a type (line 457)
        dtype_12923 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 457, 56), obj_12922, 'dtype')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 457, 39), tuple_12918, dtype_12923)
        
        keyword_12924 = tuple_12918
        kwargs_12925 = {'dtype': keyword_12924}
        # Getting the type of 'obj' (line 457)
        obj_12916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 23), 'obj', False)
        # Obtaining the member 'view' of a type (line 457)
        view_12917 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 457, 23), obj_12916, 'view')
        # Calling view(args, kwargs) (line 457)
        view_call_result_12926 = invoke(stypy.reporting.localization.Localization(__file__, 457, 23), view_12917, *[], **kwargs_12925)
        
        # Assigning a type to the variable 'stypy_return_type' (line 457)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 457, 16), 'stypy_return_type', view_call_result_12926)
        # SSA join for if statement (line 456)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'obj' (line 458)
        obj_12927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 19), 'obj')
        # Assigning a type to the variable 'stypy_return_type' (line 458)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 458, 12), 'stypy_return_type', obj_12927)
        # SSA branch for the else part of an if statement (line 455)
        module_type_store.open_ssa_branch('else')
        
        # Call to view(...): (line 460)
        # Processing the call arguments (line 460)
        # Getting the type of 'ndarray' (line 460)
        ndarray_12930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 28), 'ndarray', False)
        # Processing the call keyword arguments (line 460)
        kwargs_12931 = {}
        # Getting the type of 'obj' (line 460)
        obj_12928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 19), 'obj', False)
        # Obtaining the member 'view' of a type (line 460)
        view_12929 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 460, 19), obj_12928, 'view')
        # Calling view(args, kwargs) (line 460)
        view_call_result_12932 = invoke(stypy.reporting.localization.Localization(__file__, 460, 19), view_12929, *[ndarray_12930], **kwargs_12931)
        
        # Assigning a type to the variable 'stypy_return_type' (line 460)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 460, 12), 'stypy_return_type', view_call_result_12932)
        # SSA join for if statement (line 455)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__getattribute__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__getattribute__' in the type store
        # Getting the type of 'stypy_return_type' (line 432)
        stypy_return_type_12933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_12933)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__getattribute__'
        return stypy_return_type_12933


    @norecursion
    def __setattr__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__setattr__'
        module_type_store = module_type_store.open_function_context('__setattr__', 466, 4, False)
        # Assigning a type to the variable 'self' (line 467)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 467, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        recarray.__setattr__.__dict__.__setitem__('stypy_localization', localization)
        recarray.__setattr__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        recarray.__setattr__.__dict__.__setitem__('stypy_type_store', module_type_store)
        recarray.__setattr__.__dict__.__setitem__('stypy_function_name', 'recarray.__setattr__')
        recarray.__setattr__.__dict__.__setitem__('stypy_param_names_list', ['attr', 'val'])
        recarray.__setattr__.__dict__.__setitem__('stypy_varargs_param_name', None)
        recarray.__setattr__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        recarray.__setattr__.__dict__.__setitem__('stypy_call_defaults', defaults)
        recarray.__setattr__.__dict__.__setitem__('stypy_call_varargs', varargs)
        recarray.__setattr__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        recarray.__setattr__.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'recarray.__setattr__', ['attr', 'val'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__setattr__', localization, ['attr', 'val'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__setattr__(...)' code ##################

        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'attr' (line 470)
        attr_12934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 11), 'attr')
        str_12935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 470, 19), 'str', 'dtype')
        # Applying the binary operator '==' (line 470)
        result_eq_12936 = python_operator(stypy.reporting.localization.Localization(__file__, 470, 11), '==', attr_12934, str_12935)
        
        
        # Call to issubclass(...): (line 470)
        # Processing the call arguments (line 470)
        # Getting the type of 'val' (line 470)
        val_12938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 42), 'val', False)
        # Obtaining the member 'type' of a type (line 470)
        type_12939 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 470, 42), val_12938, 'type')
        # Getting the type of 'nt' (line 470)
        nt_12940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 52), 'nt', False)
        # Obtaining the member 'void' of a type (line 470)
        void_12941 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 470, 52), nt_12940, 'void')
        # Processing the call keyword arguments (line 470)
        kwargs_12942 = {}
        # Getting the type of 'issubclass' (line 470)
        issubclass_12937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 31), 'issubclass', False)
        # Calling issubclass(args, kwargs) (line 470)
        issubclass_call_result_12943 = invoke(stypy.reporting.localization.Localization(__file__, 470, 31), issubclass_12937, *[type_12939, void_12941], **kwargs_12942)
        
        # Applying the binary operator 'and' (line 470)
        result_and_keyword_12944 = python_operator(stypy.reporting.localization.Localization(__file__, 470, 11), 'and', result_eq_12936, issubclass_call_result_12943)
        # Getting the type of 'val' (line 470)
        val_12945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 65), 'val')
        # Obtaining the member 'fields' of a type (line 470)
        fields_12946 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 470, 65), val_12945, 'fields')
        # Applying the binary operator 'and' (line 470)
        result_and_keyword_12947 = python_operator(stypy.reporting.localization.Localization(__file__, 470, 11), 'and', result_and_keyword_12944, fields_12946)
        
        # Testing the type of an if condition (line 470)
        if_condition_12948 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 470, 8), result_and_keyword_12947)
        # Assigning a type to the variable 'if_condition_12948' (line 470)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 470, 8), 'if_condition_12948', if_condition_12948)
        # SSA begins for if statement (line 470)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 471):
        
        # Assigning a Call to a Name (line 471):
        
        # Call to dtype(...): (line 471)
        # Processing the call arguments (line 471)
        
        # Obtaining an instance of the builtin type 'tuple' (line 471)
        tuple_12951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 471, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 471)
        # Adding element type (line 471)
        # Getting the type of 'record' (line 471)
        record_12952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 28), 'record', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 471, 28), tuple_12951, record_12952)
        # Adding element type (line 471)
        # Getting the type of 'val' (line 471)
        val_12953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 36), 'val', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 471, 28), tuple_12951, val_12953)
        
        # Processing the call keyword arguments (line 471)
        kwargs_12954 = {}
        # Getting the type of 'sb' (line 471)
        sb_12949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 18), 'sb', False)
        # Obtaining the member 'dtype' of a type (line 471)
        dtype_12950 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 471, 18), sb_12949, 'dtype')
        # Calling dtype(args, kwargs) (line 471)
        dtype_call_result_12955 = invoke(stypy.reporting.localization.Localization(__file__, 471, 18), dtype_12950, *[tuple_12951], **kwargs_12954)
        
        # Assigning a type to the variable 'val' (line 471)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 471, 12), 'val', dtype_call_result_12955)
        # SSA join for if statement (line 470)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Compare to a Name (line 473):
        
        # Assigning a Compare to a Name (line 473):
        
        # Getting the type of 'attr' (line 473)
        attr_12956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 18), 'attr')
        # Getting the type of 'self' (line 473)
        self_12957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 30), 'self')
        # Obtaining the member '__dict__' of a type (line 473)
        dict___12958 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 473, 30), self_12957, '__dict__')
        # Applying the binary operator 'notin' (line 473)
        result_contains_12959 = python_operator(stypy.reporting.localization.Localization(__file__, 473, 18), 'notin', attr_12956, dict___12958)
        
        # Assigning a type to the variable 'newattr' (line 473)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 473, 8), 'newattr', result_contains_12959)
        
        
        # SSA begins for try-except statement (line 474)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Name (line 475):
        
        # Assigning a Call to a Name (line 475):
        
        # Call to __setattr__(...): (line 475)
        # Processing the call arguments (line 475)
        # Getting the type of 'self' (line 475)
        self_12962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 37), 'self', False)
        # Getting the type of 'attr' (line 475)
        attr_12963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 43), 'attr', False)
        # Getting the type of 'val' (line 475)
        val_12964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 49), 'val', False)
        # Processing the call keyword arguments (line 475)
        kwargs_12965 = {}
        # Getting the type of 'object' (line 475)
        object_12960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 18), 'object', False)
        # Obtaining the member '__setattr__' of a type (line 475)
        setattr___12961 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 475, 18), object_12960, '__setattr__')
        # Calling __setattr__(args, kwargs) (line 475)
        setattr___call_result_12966 = invoke(stypy.reporting.localization.Localization(__file__, 475, 18), setattr___12961, *[self_12962, attr_12963, val_12964], **kwargs_12965)
        
        # Assigning a type to the variable 'ret' (line 475)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 475, 12), 'ret', setattr___call_result_12966)
        # SSA branch for the except part of a try statement (line 474)
        # SSA branch for the except '<any exception>' branch of a try statement (line 474)
        module_type_store.open_ssa_branch('except')
        
        # Assigning a BoolOp to a Name (line 477):
        
        # Assigning a BoolOp to a Name (line 477):
        
        # Evaluating a boolean operation
        
        # Call to __getattribute__(...): (line 477)
        # Processing the call arguments (line 477)
        # Getting the type of 'self' (line 477)
        self_12969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 49), 'self', False)
        str_12970 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 477, 55), 'str', 'dtype')
        # Processing the call keyword arguments (line 477)
        kwargs_12971 = {}
        # Getting the type of 'ndarray' (line 477)
        ndarray_12967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 24), 'ndarray', False)
        # Obtaining the member '__getattribute__' of a type (line 477)
        getattribute___12968 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 477, 24), ndarray_12967, '__getattribute__')
        # Calling __getattribute__(args, kwargs) (line 477)
        getattribute___call_result_12972 = invoke(stypy.reporting.localization.Localization(__file__, 477, 24), getattribute___12968, *[self_12969, str_12970], **kwargs_12971)
        
        # Obtaining the member 'fields' of a type (line 477)
        fields_12973 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 477, 24), getattribute___call_result_12972, 'fields')
        
        # Obtaining an instance of the builtin type 'dict' (line 477)
        dict_12974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 477, 74), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 477)
        
        # Applying the binary operator 'or' (line 477)
        result_or_keyword_12975 = python_operator(stypy.reporting.localization.Localization(__file__, 477, 24), 'or', fields_12973, dict_12974)
        
        # Assigning a type to the variable 'fielddict' (line 477)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 477, 12), 'fielddict', result_or_keyword_12975)
        
        
        # Getting the type of 'attr' (line 478)
        attr_12976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 15), 'attr')
        # Getting the type of 'fielddict' (line 478)
        fielddict_12977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 27), 'fielddict')
        # Applying the binary operator 'notin' (line 478)
        result_contains_12978 = python_operator(stypy.reporting.localization.Localization(__file__, 478, 15), 'notin', attr_12976, fielddict_12977)
        
        # Testing the type of an if condition (line 478)
        if_condition_12979 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 478, 12), result_contains_12978)
        # Assigning a type to the variable 'if_condition_12979' (line 478)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 478, 12), 'if_condition_12979', if_condition_12979)
        # SSA begins for if statement (line 478)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Tuple (line 479):
        
        # Assigning a Subscript to a Name (line 479):
        
        # Obtaining the type of the subscript
        int_12980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 479, 16), 'int')
        
        # Obtaining the type of the subscript
        int_12981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 479, 49), 'int')
        slice_12982 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 479, 33), None, int_12981, None)
        
        # Call to exc_info(...): (line 479)
        # Processing the call keyword arguments (line 479)
        kwargs_12985 = {}
        # Getting the type of 'sys' (line 479)
        sys_12983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 33), 'sys', False)
        # Obtaining the member 'exc_info' of a type (line 479)
        exc_info_12984 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 479, 33), sys_12983, 'exc_info')
        # Calling exc_info(args, kwargs) (line 479)
        exc_info_call_result_12986 = invoke(stypy.reporting.localization.Localization(__file__, 479, 33), exc_info_12984, *[], **kwargs_12985)
        
        # Obtaining the member '__getitem__' of a type (line 479)
        getitem___12987 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 479, 33), exc_info_call_result_12986, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 479)
        subscript_call_result_12988 = invoke(stypy.reporting.localization.Localization(__file__, 479, 33), getitem___12987, slice_12982)
        
        # Obtaining the member '__getitem__' of a type (line 479)
        getitem___12989 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 479, 16), subscript_call_result_12988, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 479)
        subscript_call_result_12990 = invoke(stypy.reporting.localization.Localization(__file__, 479, 16), getitem___12989, int_12980)
        
        # Assigning a type to the variable 'tuple_var_assignment_12230' (line 479)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 479, 16), 'tuple_var_assignment_12230', subscript_call_result_12990)
        
        # Assigning a Subscript to a Name (line 479):
        
        # Obtaining the type of the subscript
        int_12991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 479, 16), 'int')
        
        # Obtaining the type of the subscript
        int_12992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 479, 49), 'int')
        slice_12993 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 479, 33), None, int_12992, None)
        
        # Call to exc_info(...): (line 479)
        # Processing the call keyword arguments (line 479)
        kwargs_12996 = {}
        # Getting the type of 'sys' (line 479)
        sys_12994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 33), 'sys', False)
        # Obtaining the member 'exc_info' of a type (line 479)
        exc_info_12995 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 479, 33), sys_12994, 'exc_info')
        # Calling exc_info(args, kwargs) (line 479)
        exc_info_call_result_12997 = invoke(stypy.reporting.localization.Localization(__file__, 479, 33), exc_info_12995, *[], **kwargs_12996)
        
        # Obtaining the member '__getitem__' of a type (line 479)
        getitem___12998 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 479, 33), exc_info_call_result_12997, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 479)
        subscript_call_result_12999 = invoke(stypy.reporting.localization.Localization(__file__, 479, 33), getitem___12998, slice_12993)
        
        # Obtaining the member '__getitem__' of a type (line 479)
        getitem___13000 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 479, 16), subscript_call_result_12999, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 479)
        subscript_call_result_13001 = invoke(stypy.reporting.localization.Localization(__file__, 479, 16), getitem___13000, int_12991)
        
        # Assigning a type to the variable 'tuple_var_assignment_12231' (line 479)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 479, 16), 'tuple_var_assignment_12231', subscript_call_result_13001)
        
        # Assigning a Name to a Name (line 479):
        # Getting the type of 'tuple_var_assignment_12230' (line 479)
        tuple_var_assignment_12230_13002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 16), 'tuple_var_assignment_12230')
        # Assigning a type to the variable 'exctype' (line 479)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 479, 16), 'exctype', tuple_var_assignment_12230_13002)
        
        # Assigning a Name to a Name (line 479):
        # Getting the type of 'tuple_var_assignment_12231' (line 479)
        tuple_var_assignment_12231_13003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 16), 'tuple_var_assignment_12231')
        # Assigning a type to the variable 'value' (line 479)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 479, 25), 'value', tuple_var_assignment_12231_13003)
        
        # Call to exctype(...): (line 480)
        # Processing the call arguments (line 480)
        # Getting the type of 'value' (line 480)
        value_13005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 30), 'value', False)
        # Processing the call keyword arguments (line 480)
        kwargs_13006 = {}
        # Getting the type of 'exctype' (line 480)
        exctype_13004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 22), 'exctype', False)
        # Calling exctype(args, kwargs) (line 480)
        exctype_call_result_13007 = invoke(stypy.reporting.localization.Localization(__file__, 480, 22), exctype_13004, *[value_13005], **kwargs_13006)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 480, 16), exctype_call_result_13007, 'raise parameter', BaseException)
        # SSA join for if statement (line 478)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else branch of a try statement (line 474)
        module_type_store.open_ssa_branch('except else')
        
        # Assigning a BoolOp to a Name (line 482):
        
        # Assigning a BoolOp to a Name (line 482):
        
        # Evaluating a boolean operation
        
        # Call to __getattribute__(...): (line 482)
        # Processing the call arguments (line 482)
        # Getting the type of 'self' (line 482)
        self_13010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 49), 'self', False)
        str_13011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 482, 55), 'str', 'dtype')
        # Processing the call keyword arguments (line 482)
        kwargs_13012 = {}
        # Getting the type of 'ndarray' (line 482)
        ndarray_13008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 24), 'ndarray', False)
        # Obtaining the member '__getattribute__' of a type (line 482)
        getattribute___13009 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 482, 24), ndarray_13008, '__getattribute__')
        # Calling __getattribute__(args, kwargs) (line 482)
        getattribute___call_result_13013 = invoke(stypy.reporting.localization.Localization(__file__, 482, 24), getattribute___13009, *[self_13010, str_13011], **kwargs_13012)
        
        # Obtaining the member 'fields' of a type (line 482)
        fields_13014 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 482, 24), getattribute___call_result_13013, 'fields')
        
        # Obtaining an instance of the builtin type 'dict' (line 482)
        dict_13015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 482, 74), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 482)
        
        # Applying the binary operator 'or' (line 482)
        result_or_keyword_13016 = python_operator(stypy.reporting.localization.Localization(__file__, 482, 24), 'or', fields_13014, dict_13015)
        
        # Assigning a type to the variable 'fielddict' (line 482)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 482, 12), 'fielddict', result_or_keyword_13016)
        
        
        # Getting the type of 'attr' (line 483)
        attr_13017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 15), 'attr')
        # Getting the type of 'fielddict' (line 483)
        fielddict_13018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 27), 'fielddict')
        # Applying the binary operator 'notin' (line 483)
        result_contains_13019 = python_operator(stypy.reporting.localization.Localization(__file__, 483, 15), 'notin', attr_13017, fielddict_13018)
        
        # Testing the type of an if condition (line 483)
        if_condition_13020 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 483, 12), result_contains_13019)
        # Assigning a type to the variable 'if_condition_13020' (line 483)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 483, 12), 'if_condition_13020', if_condition_13020)
        # SSA begins for if statement (line 483)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'ret' (line 484)
        ret_13021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 23), 'ret')
        # Assigning a type to the variable 'stypy_return_type' (line 484)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 484, 16), 'stypy_return_type', ret_13021)
        # SSA join for if statement (line 483)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'newattr' (line 485)
        newattr_13022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 15), 'newattr')
        # Testing the type of an if condition (line 485)
        if_condition_13023 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 485, 12), newattr_13022)
        # Assigning a type to the variable 'if_condition_13023' (line 485)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 485, 12), 'if_condition_13023', if_condition_13023)
        # SSA begins for if statement (line 485)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # SSA begins for try-except statement (line 488)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to __delattr__(...): (line 489)
        # Processing the call arguments (line 489)
        # Getting the type of 'self' (line 489)
        self_13026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 39), 'self', False)
        # Getting the type of 'attr' (line 489)
        attr_13027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 45), 'attr', False)
        # Processing the call keyword arguments (line 489)
        kwargs_13028 = {}
        # Getting the type of 'object' (line 489)
        object_13024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 20), 'object', False)
        # Obtaining the member '__delattr__' of a type (line 489)
        delattr___13025 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 489, 20), object_13024, '__delattr__')
        # Calling __delattr__(args, kwargs) (line 489)
        delattr___call_result_13029 = invoke(stypy.reporting.localization.Localization(__file__, 489, 20), delattr___13025, *[self_13026, attr_13027], **kwargs_13028)
        
        # SSA branch for the except part of a try statement (line 488)
        # SSA branch for the except '<any exception>' branch of a try statement (line 488)
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'ret' (line 491)
        ret_13030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 27), 'ret')
        # Assigning a type to the variable 'stypy_return_type' (line 491)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 491, 20), 'stypy_return_type', ret_13030)
        # SSA join for try-except statement (line 488)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 485)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for try-except statement (line 474)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # SSA begins for try-except statement (line 492)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Subscript to a Name (line 493):
        
        # Assigning a Subscript to a Name (line 493):
        
        # Obtaining the type of the subscript
        int_13031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 493, 35), 'int')
        slice_13032 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 493, 18), None, int_13031, None)
        
        # Obtaining the type of the subscript
        # Getting the type of 'attr' (line 493)
        attr_13033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 28), 'attr')
        # Getting the type of 'fielddict' (line 493)
        fielddict_13034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 18), 'fielddict')
        # Obtaining the member '__getitem__' of a type (line 493)
        getitem___13035 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 493, 18), fielddict_13034, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 493)
        subscript_call_result_13036 = invoke(stypy.reporting.localization.Localization(__file__, 493, 18), getitem___13035, attr_13033)
        
        # Obtaining the member '__getitem__' of a type (line 493)
        getitem___13037 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 493, 18), subscript_call_result_13036, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 493)
        subscript_call_result_13038 = invoke(stypy.reporting.localization.Localization(__file__, 493, 18), getitem___13037, slice_13032)
        
        # Assigning a type to the variable 'res' (line 493)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 493, 12), 'res', subscript_call_result_13038)
        # SSA branch for the except part of a try statement (line 492)
        # SSA branch for the except 'Tuple' branch of a try statement (line 492)
        module_type_store.open_ssa_branch('except')
        
        # Call to AttributeError(...): (line 495)
        # Processing the call arguments (line 495)
        str_13040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 495, 33), 'str', 'record array has no attribute %s')
        # Getting the type of 'attr' (line 495)
        attr_13041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 70), 'attr', False)
        # Applying the binary operator '%' (line 495)
        result_mod_13042 = python_operator(stypy.reporting.localization.Localization(__file__, 495, 33), '%', str_13040, attr_13041)
        
        # Processing the call keyword arguments (line 495)
        kwargs_13043 = {}
        # Getting the type of 'AttributeError' (line 495)
        AttributeError_13039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 18), 'AttributeError', False)
        # Calling AttributeError(args, kwargs) (line 495)
        AttributeError_call_result_13044 = invoke(stypy.reporting.localization.Localization(__file__, 495, 18), AttributeError_13039, *[result_mod_13042], **kwargs_13043)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 495, 12), AttributeError_call_result_13044, 'raise parameter', BaseException)
        # SSA join for try-except statement (line 492)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to setfield(...): (line 496)
        # Processing the call arguments (line 496)
        # Getting the type of 'val' (line 496)
        val_13047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 29), 'val', False)
        # Getting the type of 'res' (line 496)
        res_13048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 35), 'res', False)
        # Processing the call keyword arguments (line 496)
        kwargs_13049 = {}
        # Getting the type of 'self' (line 496)
        self_13045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 15), 'self', False)
        # Obtaining the member 'setfield' of a type (line 496)
        setfield_13046 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 496, 15), self_13045, 'setfield')
        # Calling setfield(args, kwargs) (line 496)
        setfield_call_result_13050 = invoke(stypy.reporting.localization.Localization(__file__, 496, 15), setfield_13046, *[val_13047, res_13048], **kwargs_13049)
        
        # Assigning a type to the variable 'stypy_return_type' (line 496)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 496, 8), 'stypy_return_type', setfield_call_result_13050)
        
        # ################# End of '__setattr__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__setattr__' in the type store
        # Getting the type of 'stypy_return_type' (line 466)
        stypy_return_type_13051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_13051)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__setattr__'
        return stypy_return_type_13051


    @norecursion
    def __getitem__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__getitem__'
        module_type_store = module_type_store.open_function_context('__getitem__', 498, 4, False)
        # Assigning a type to the variable 'self' (line 499)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 499, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        recarray.__getitem__.__dict__.__setitem__('stypy_localization', localization)
        recarray.__getitem__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        recarray.__getitem__.__dict__.__setitem__('stypy_type_store', module_type_store)
        recarray.__getitem__.__dict__.__setitem__('stypy_function_name', 'recarray.__getitem__')
        recarray.__getitem__.__dict__.__setitem__('stypy_param_names_list', ['indx'])
        recarray.__getitem__.__dict__.__setitem__('stypy_varargs_param_name', None)
        recarray.__getitem__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        recarray.__getitem__.__dict__.__setitem__('stypy_call_defaults', defaults)
        recarray.__getitem__.__dict__.__setitem__('stypy_call_varargs', varargs)
        recarray.__getitem__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        recarray.__getitem__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'recarray.__getitem__', ['indx'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__getitem__', localization, ['indx'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__getitem__(...)' code ##################

        
        # Assigning a Call to a Name (line 499):
        
        # Assigning a Call to a Name (line 499):
        
        # Call to __getitem__(...): (line 499)
        # Processing the call arguments (line 499)
        # Getting the type of 'indx' (line 499)
        indx_13058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 48), 'indx', False)
        # Processing the call keyword arguments (line 499)
        kwargs_13059 = {}
        
        # Call to super(...): (line 499)
        # Processing the call arguments (line 499)
        # Getting the type of 'recarray' (line 499)
        recarray_13053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 20), 'recarray', False)
        # Getting the type of 'self' (line 499)
        self_13054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 30), 'self', False)
        # Processing the call keyword arguments (line 499)
        kwargs_13055 = {}
        # Getting the type of 'super' (line 499)
        super_13052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 14), 'super', False)
        # Calling super(args, kwargs) (line 499)
        super_call_result_13056 = invoke(stypy.reporting.localization.Localization(__file__, 499, 14), super_13052, *[recarray_13053, self_13054], **kwargs_13055)
        
        # Obtaining the member '__getitem__' of a type (line 499)
        getitem___13057 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 499, 14), super_call_result_13056, '__getitem__')
        # Calling __getitem__(args, kwargs) (line 499)
        getitem___call_result_13060 = invoke(stypy.reporting.localization.Localization(__file__, 499, 14), getitem___13057, *[indx_13058], **kwargs_13059)
        
        # Assigning a type to the variable 'obj' (line 499)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 499, 8), 'obj', getitem___call_result_13060)
        
        
        # Call to isinstance(...): (line 503)
        # Processing the call arguments (line 503)
        # Getting the type of 'obj' (line 503)
        obj_13062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 22), 'obj', False)
        # Getting the type of 'ndarray' (line 503)
        ndarray_13063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 27), 'ndarray', False)
        # Processing the call keyword arguments (line 503)
        kwargs_13064 = {}
        # Getting the type of 'isinstance' (line 503)
        isinstance_13061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 503)
        isinstance_call_result_13065 = invoke(stypy.reporting.localization.Localization(__file__, 503, 11), isinstance_13061, *[obj_13062, ndarray_13063], **kwargs_13064)
        
        # Testing the type of an if condition (line 503)
        if_condition_13066 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 503, 8), isinstance_call_result_13065)
        # Assigning a type to the variable 'if_condition_13066' (line 503)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 503, 8), 'if_condition_13066', if_condition_13066)
        # SSA begins for if statement (line 503)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'obj' (line 504)
        obj_13067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 15), 'obj')
        # Obtaining the member 'dtype' of a type (line 504)
        dtype_13068 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 504, 15), obj_13067, 'dtype')
        # Obtaining the member 'fields' of a type (line 504)
        fields_13069 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 504, 15), dtype_13068, 'fields')
        # Testing the type of an if condition (line 504)
        if_condition_13070 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 504, 12), fields_13069)
        # Assigning a type to the variable 'if_condition_13070' (line 504)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 504, 12), 'if_condition_13070', if_condition_13070)
        # SSA begins for if statement (line 504)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 505):
        
        # Assigning a Call to a Name (line 505):
        
        # Call to view(...): (line 505)
        # Processing the call arguments (line 505)
        
        # Call to type(...): (line 505)
        # Processing the call arguments (line 505)
        # Getting the type of 'self' (line 505)
        self_13074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 36), 'self', False)
        # Processing the call keyword arguments (line 505)
        kwargs_13075 = {}
        # Getting the type of 'type' (line 505)
        type_13073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 31), 'type', False)
        # Calling type(args, kwargs) (line 505)
        type_call_result_13076 = invoke(stypy.reporting.localization.Localization(__file__, 505, 31), type_13073, *[self_13074], **kwargs_13075)
        
        # Processing the call keyword arguments (line 505)
        kwargs_13077 = {}
        # Getting the type of 'obj' (line 505)
        obj_13071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 22), 'obj', False)
        # Obtaining the member 'view' of a type (line 505)
        view_13072 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 505, 22), obj_13071, 'view')
        # Calling view(args, kwargs) (line 505)
        view_call_result_13078 = invoke(stypy.reporting.localization.Localization(__file__, 505, 22), view_13072, *[type_call_result_13076], **kwargs_13077)
        
        # Assigning a type to the variable 'obj' (line 505)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 505, 16), 'obj', view_call_result_13078)
        
        
        # Call to issubclass(...): (line 506)
        # Processing the call arguments (line 506)
        # Getting the type of 'obj' (line 506)
        obj_13080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 30), 'obj', False)
        # Obtaining the member 'dtype' of a type (line 506)
        dtype_13081 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 506, 30), obj_13080, 'dtype')
        # Obtaining the member 'type' of a type (line 506)
        type_13082 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 506, 30), dtype_13081, 'type')
        # Getting the type of 'nt' (line 506)
        nt_13083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 46), 'nt', False)
        # Obtaining the member 'void' of a type (line 506)
        void_13084 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 506, 46), nt_13083, 'void')
        # Processing the call keyword arguments (line 506)
        kwargs_13085 = {}
        # Getting the type of 'issubclass' (line 506)
        issubclass_13079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 19), 'issubclass', False)
        # Calling issubclass(args, kwargs) (line 506)
        issubclass_call_result_13086 = invoke(stypy.reporting.localization.Localization(__file__, 506, 19), issubclass_13079, *[type_13082, void_13084], **kwargs_13085)
        
        # Testing the type of an if condition (line 506)
        if_condition_13087 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 506, 16), issubclass_call_result_13086)
        # Assigning a type to the variable 'if_condition_13087' (line 506)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 506, 16), 'if_condition_13087', if_condition_13087)
        # SSA begins for if statement (line 506)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to view(...): (line 507)
        # Processing the call keyword arguments (line 507)
        
        # Obtaining an instance of the builtin type 'tuple' (line 507)
        tuple_13090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 507, 43), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 507)
        # Adding element type (line 507)
        # Getting the type of 'self' (line 507)
        self_13091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 43), 'self', False)
        # Obtaining the member 'dtype' of a type (line 507)
        dtype_13092 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 507, 43), self_13091, 'dtype')
        # Obtaining the member 'type' of a type (line 507)
        type_13093 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 507, 43), dtype_13092, 'type')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 507, 43), tuple_13090, type_13093)
        # Adding element type (line 507)
        # Getting the type of 'obj' (line 507)
        obj_13094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 60), 'obj', False)
        # Obtaining the member 'dtype' of a type (line 507)
        dtype_13095 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 507, 60), obj_13094, 'dtype')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 507, 43), tuple_13090, dtype_13095)
        
        keyword_13096 = tuple_13090
        kwargs_13097 = {'dtype': keyword_13096}
        # Getting the type of 'obj' (line 507)
        obj_13088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 27), 'obj', False)
        # Obtaining the member 'view' of a type (line 507)
        view_13089 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 507, 27), obj_13088, 'view')
        # Calling view(args, kwargs) (line 507)
        view_call_result_13098 = invoke(stypy.reporting.localization.Localization(__file__, 507, 27), view_13089, *[], **kwargs_13097)
        
        # Assigning a type to the variable 'stypy_return_type' (line 507)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 507, 20), 'stypy_return_type', view_call_result_13098)
        # SSA join for if statement (line 506)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'obj' (line 508)
        obj_13099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 23), 'obj')
        # Assigning a type to the variable 'stypy_return_type' (line 508)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 508, 16), 'stypy_return_type', obj_13099)
        # SSA branch for the else part of an if statement (line 504)
        module_type_store.open_ssa_branch('else')
        
        # Call to view(...): (line 510)
        # Processing the call keyword arguments (line 510)
        # Getting the type of 'ndarray' (line 510)
        ndarray_13102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 37), 'ndarray', False)
        keyword_13103 = ndarray_13102
        kwargs_13104 = {'type': keyword_13103}
        # Getting the type of 'obj' (line 510)
        obj_13100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 23), 'obj', False)
        # Obtaining the member 'view' of a type (line 510)
        view_13101 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 510, 23), obj_13100, 'view')
        # Calling view(args, kwargs) (line 510)
        view_call_result_13105 = invoke(stypy.reporting.localization.Localization(__file__, 510, 23), view_13101, *[], **kwargs_13104)
        
        # Assigning a type to the variable 'stypy_return_type' (line 510)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 510, 16), 'stypy_return_type', view_call_result_13105)
        # SSA join for if statement (line 504)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 503)
        module_type_store.open_ssa_branch('else')
        # Getting the type of 'obj' (line 513)
        obj_13106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 19), 'obj')
        # Assigning a type to the variable 'stypy_return_type' (line 513)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 513, 12), 'stypy_return_type', obj_13106)
        # SSA join for if statement (line 503)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__getitem__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__getitem__' in the type store
        # Getting the type of 'stypy_return_type' (line 498)
        stypy_return_type_13107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_13107)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__getitem__'
        return stypy_return_type_13107


    @norecursion
    def stypy__repr__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__repr__'
        module_type_store = module_type_store.open_function_context('__repr__', 515, 4, False)
        # Assigning a type to the variable 'self' (line 516)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 516, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        recarray.stypy__repr__.__dict__.__setitem__('stypy_localization', localization)
        recarray.stypy__repr__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        recarray.stypy__repr__.__dict__.__setitem__('stypy_type_store', module_type_store)
        recarray.stypy__repr__.__dict__.__setitem__('stypy_function_name', 'recarray.__repr__')
        recarray.stypy__repr__.__dict__.__setitem__('stypy_param_names_list', [])
        recarray.stypy__repr__.__dict__.__setitem__('stypy_varargs_param_name', None)
        recarray.stypy__repr__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        recarray.stypy__repr__.__dict__.__setitem__('stypy_call_defaults', defaults)
        recarray.stypy__repr__.__dict__.__setitem__('stypy_call_varargs', varargs)
        recarray.stypy__repr__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        recarray.stypy__repr__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'recarray.__repr__', [], None, None, defaults, varargs, kwargs)

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

        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'self' (line 517)
        self_13108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 11), 'self')
        # Obtaining the member 'size' of a type (line 517)
        size_13109 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 517, 11), self_13108, 'size')
        int_13110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 517, 23), 'int')
        # Applying the binary operator '>' (line 517)
        result_gt_13111 = python_operator(stypy.reporting.localization.Localization(__file__, 517, 11), '>', size_13109, int_13110)
        
        
        # Getting the type of 'self' (line 517)
        self_13112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 28), 'self')
        # Obtaining the member 'shape' of a type (line 517)
        shape_13113 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 517, 28), self_13112, 'shape')
        
        # Obtaining an instance of the builtin type 'tuple' (line 517)
        tuple_13114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 517, 43), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 517)
        # Adding element type (line 517)
        int_13115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 517, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 517, 43), tuple_13114, int_13115)
        
        # Applying the binary operator '==' (line 517)
        result_eq_13116 = python_operator(stypy.reporting.localization.Localization(__file__, 517, 28), '==', shape_13113, tuple_13114)
        
        # Applying the binary operator 'or' (line 517)
        result_or_keyword_13117 = python_operator(stypy.reporting.localization.Localization(__file__, 517, 11), 'or', result_gt_13111, result_eq_13116)
        
        # Testing the type of an if condition (line 517)
        if_condition_13118 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 517, 8), result_or_keyword_13117)
        # Assigning a type to the variable 'if_condition_13118' (line 517)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 517, 8), 'if_condition_13118', if_condition_13118)
        # SSA begins for if statement (line 517)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 518):
        
        # Assigning a Call to a Name (line 518):
        
        # Call to array2string(...): (line 518)
        # Processing the call arguments (line 518)
        # Getting the type of 'self' (line 518)
        self_13121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 34), 'self', False)
        # Processing the call keyword arguments (line 518)
        str_13122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 518, 50), 'str', ', ')
        keyword_13123 = str_13122
        kwargs_13124 = {'separator': keyword_13123}
        # Getting the type of 'sb' (line 518)
        sb_13119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 18), 'sb', False)
        # Obtaining the member 'array2string' of a type (line 518)
        array2string_13120 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 518, 18), sb_13119, 'array2string')
        # Calling array2string(args, kwargs) (line 518)
        array2string_call_result_13125 = invoke(stypy.reporting.localization.Localization(__file__, 518, 18), array2string_13120, *[self_13121], **kwargs_13124)
        
        # Assigning a type to the variable 'lst' (line 518)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 518, 12), 'lst', array2string_call_result_13125)
        # SSA branch for the else part of an if statement (line 517)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a BinOp to a Name (line 521):
        
        # Assigning a BinOp to a Name (line 521):
        str_13126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 521, 18), 'str', '[], shape=%s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 521)
        tuple_13127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 521, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 521)
        # Adding element type (line 521)
        
        # Call to repr(...): (line 521)
        # Processing the call arguments (line 521)
        # Getting the type of 'self' (line 521)
        self_13129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 41), 'self', False)
        # Obtaining the member 'shape' of a type (line 521)
        shape_13130 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 521, 41), self_13129, 'shape')
        # Processing the call keyword arguments (line 521)
        kwargs_13131 = {}
        # Getting the type of 'repr' (line 521)
        repr_13128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 36), 'repr', False)
        # Calling repr(args, kwargs) (line 521)
        repr_call_result_13132 = invoke(stypy.reporting.localization.Localization(__file__, 521, 36), repr_13128, *[shape_13130], **kwargs_13131)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 521, 36), tuple_13127, repr_call_result_13132)
        
        # Applying the binary operator '%' (line 521)
        result_mod_13133 = python_operator(stypy.reporting.localization.Localization(__file__, 521, 18), '%', str_13126, tuple_13127)
        
        # Assigning a type to the variable 'lst' (line 521)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 521, 12), 'lst', result_mod_13133)
        # SSA join for if statement (line 517)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'self' (line 523)
        self_13134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 12), 'self')
        # Obtaining the member 'dtype' of a type (line 523)
        dtype_13135 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 523, 12), self_13134, 'dtype')
        # Obtaining the member 'type' of a type (line 523)
        type_13136 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 523, 12), dtype_13135, 'type')
        # Getting the type of 'record' (line 523)
        record_13137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 31), 'record')
        # Applying the binary operator 'is' (line 523)
        result_is__13138 = python_operator(stypy.reporting.localization.Localization(__file__, 523, 12), 'is', type_13136, record_13137)
        
        
        
        # Call to issubclass(...): (line 524)
        # Processing the call arguments (line 524)
        # Getting the type of 'self' (line 524)
        self_13140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 35), 'self', False)
        # Obtaining the member 'dtype' of a type (line 524)
        dtype_13141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 524, 35), self_13140, 'dtype')
        # Obtaining the member 'type' of a type (line 524)
        type_13142 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 524, 35), dtype_13141, 'type')
        # Getting the type of 'nt' (line 524)
        nt_13143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 52), 'nt', False)
        # Obtaining the member 'void' of a type (line 524)
        void_13144 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 524, 52), nt_13143, 'void')
        # Processing the call keyword arguments (line 524)
        kwargs_13145 = {}
        # Getting the type of 'issubclass' (line 524)
        issubclass_13139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 24), 'issubclass', False)
        # Calling issubclass(args, kwargs) (line 524)
        issubclass_call_result_13146 = invoke(stypy.reporting.localization.Localization(__file__, 524, 24), issubclass_13139, *[type_13142, void_13144], **kwargs_13145)
        
        # Applying the 'not' unary operator (line 524)
        result_not__13147 = python_operator(stypy.reporting.localization.Localization(__file__, 524, 20), 'not', issubclass_call_result_13146)
        
        # Applying the binary operator 'or' (line 523)
        result_or_keyword_13148 = python_operator(stypy.reporting.localization.Localization(__file__, 523, 12), 'or', result_is__13138, result_not__13147)
        
        # Testing the type of an if condition (line 523)
        if_condition_13149 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 523, 8), result_or_keyword_13148)
        # Assigning a type to the variable 'if_condition_13149' (line 523)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 523, 8), 'if_condition_13149', if_condition_13149)
        # SSA begins for if statement (line 523)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Name (line 530):
        
        # Assigning a Attribute to a Name (line 530):
        # Getting the type of 'self' (line 530)
        self_13150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 26), 'self')
        # Obtaining the member 'dtype' of a type (line 530)
        dtype_13151 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 530, 26), self_13150, 'dtype')
        # Assigning a type to the variable 'plain_dtype' (line 530)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 530, 12), 'plain_dtype', dtype_13151)
        
        
        # Getting the type of 'plain_dtype' (line 531)
        plain_dtype_13152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 15), 'plain_dtype')
        # Obtaining the member 'type' of a type (line 531)
        type_13153 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 531, 15), plain_dtype_13152, 'type')
        # Getting the type of 'record' (line 531)
        record_13154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 35), 'record')
        # Applying the binary operator 'is' (line 531)
        result_is__13155 = python_operator(stypy.reporting.localization.Localization(__file__, 531, 15), 'is', type_13153, record_13154)
        
        # Testing the type of an if condition (line 531)
        if_condition_13156 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 531, 12), result_is__13155)
        # Assigning a type to the variable 'if_condition_13156' (line 531)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 531, 12), 'if_condition_13156', if_condition_13156)
        # SSA begins for if statement (line 531)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 532):
        
        # Assigning a Call to a Name (line 532):
        
        # Call to dtype(...): (line 532)
        # Processing the call arguments (line 532)
        
        # Obtaining an instance of the builtin type 'tuple' (line 532)
        tuple_13159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 532, 40), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 532)
        # Adding element type (line 532)
        # Getting the type of 'nt' (line 532)
        nt_13160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 40), 'nt', False)
        # Obtaining the member 'void' of a type (line 532)
        void_13161 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 532, 40), nt_13160, 'void')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 532, 40), tuple_13159, void_13161)
        # Adding element type (line 532)
        # Getting the type of 'plain_dtype' (line 532)
        plain_dtype_13162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 49), 'plain_dtype', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 532, 40), tuple_13159, plain_dtype_13162)
        
        # Processing the call keyword arguments (line 532)
        kwargs_13163 = {}
        # Getting the type of 'sb' (line 532)
        sb_13157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 30), 'sb', False)
        # Obtaining the member 'dtype' of a type (line 532)
        dtype_13158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 532, 30), sb_13157, 'dtype')
        # Calling dtype(args, kwargs) (line 532)
        dtype_call_result_13164 = invoke(stypy.reporting.localization.Localization(__file__, 532, 30), dtype_13158, *[tuple_13159], **kwargs_13163)
        
        # Assigning a type to the variable 'plain_dtype' (line 532)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 532, 16), 'plain_dtype', dtype_call_result_13164)
        # SSA join for if statement (line 531)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 533):
        
        # Assigning a BinOp to a Name (line 533):
        str_13165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 533, 17), 'str', '\n')
        str_13166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 533, 22), 'str', ' ')
        
        # Call to len(...): (line 533)
        # Processing the call arguments (line 533)
        str_13168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 533, 30), 'str', 'rec.array(')
        # Processing the call keyword arguments (line 533)
        kwargs_13169 = {}
        # Getting the type of 'len' (line 533)
        len_13167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 26), 'len', False)
        # Calling len(args, kwargs) (line 533)
        len_call_result_13170 = invoke(stypy.reporting.localization.Localization(__file__, 533, 26), len_13167, *[str_13168], **kwargs_13169)
        
        # Applying the binary operator '*' (line 533)
        result_mul_13171 = python_operator(stypy.reporting.localization.Localization(__file__, 533, 22), '*', str_13166, len_call_result_13170)
        
        # Applying the binary operator '+' (line 533)
        result_add_13172 = python_operator(stypy.reporting.localization.Localization(__file__, 533, 17), '+', str_13165, result_mul_13171)
        
        # Assigning a type to the variable 'lf' (line 533)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 533, 12), 'lf', result_add_13172)
        str_13173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 534, 20), 'str', 'rec.array(%s, %sdtype=%s)')
        
        # Obtaining an instance of the builtin type 'tuple' (line 535)
        tuple_13174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 535, 27), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 535)
        # Adding element type (line 535)
        # Getting the type of 'lst' (line 535)
        lst_13175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 27), 'lst')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 535, 27), tuple_13174, lst_13175)
        # Adding element type (line 535)
        # Getting the type of 'lf' (line 535)
        lf_13176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 32), 'lf')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 535, 27), tuple_13174, lf_13176)
        # Adding element type (line 535)
        # Getting the type of 'plain_dtype' (line 535)
        plain_dtype_13177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 36), 'plain_dtype')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 535, 27), tuple_13174, plain_dtype_13177)
        
        # Applying the binary operator '%' (line 534)
        result_mod_13178 = python_operator(stypy.reporting.localization.Localization(__file__, 534, 20), '%', str_13173, tuple_13174)
        
        # Assigning a type to the variable 'stypy_return_type' (line 534)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 534, 12), 'stypy_return_type', result_mod_13178)
        # SSA branch for the else part of an if statement (line 523)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a BinOp to a Name (line 540):
        
        # Assigning a BinOp to a Name (line 540):
        str_13179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 540, 17), 'str', '\n')
        str_13180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 540, 22), 'str', ' ')
        
        # Call to len(...): (line 540)
        # Processing the call arguments (line 540)
        str_13182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 540, 30), 'str', 'array(')
        # Processing the call keyword arguments (line 540)
        kwargs_13183 = {}
        # Getting the type of 'len' (line 540)
        len_13181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 26), 'len', False)
        # Calling len(args, kwargs) (line 540)
        len_call_result_13184 = invoke(stypy.reporting.localization.Localization(__file__, 540, 26), len_13181, *[str_13182], **kwargs_13183)
        
        # Applying the binary operator '*' (line 540)
        result_mul_13185 = python_operator(stypy.reporting.localization.Localization(__file__, 540, 22), '*', str_13180, len_call_result_13184)
        
        # Applying the binary operator '+' (line 540)
        result_add_13186 = python_operator(stypy.reporting.localization.Localization(__file__, 540, 17), '+', str_13179, result_mul_13185)
        
        # Assigning a type to the variable 'lf' (line 540)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 540, 12), 'lf', result_add_13186)
        str_13187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 541, 20), 'str', 'array(%s, %sdtype=%s).view(numpy.recarray)')
        
        # Obtaining an instance of the builtin type 'tuple' (line 542)
        tuple_13188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 542, 27), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 542)
        # Adding element type (line 542)
        # Getting the type of 'lst' (line 542)
        lst_13189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 27), 'lst')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 542, 27), tuple_13188, lst_13189)
        # Adding element type (line 542)
        # Getting the type of 'lf' (line 542)
        lf_13190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 32), 'lf')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 542, 27), tuple_13188, lf_13190)
        # Adding element type (line 542)
        
        # Call to str(...): (line 542)
        # Processing the call arguments (line 542)
        # Getting the type of 'self' (line 542)
        self_13192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 40), 'self', False)
        # Obtaining the member 'dtype' of a type (line 542)
        dtype_13193 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 542, 40), self_13192, 'dtype')
        # Processing the call keyword arguments (line 542)
        kwargs_13194 = {}
        # Getting the type of 'str' (line 542)
        str_13191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 36), 'str', False)
        # Calling str(args, kwargs) (line 542)
        str_call_result_13195 = invoke(stypy.reporting.localization.Localization(__file__, 542, 36), str_13191, *[dtype_13193], **kwargs_13194)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 542, 27), tuple_13188, str_call_result_13195)
        
        # Applying the binary operator '%' (line 541)
        result_mod_13196 = python_operator(stypy.reporting.localization.Localization(__file__, 541, 20), '%', str_13187, tuple_13188)
        
        # Assigning a type to the variable 'stypy_return_type' (line 541)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 541, 12), 'stypy_return_type', result_mod_13196)
        # SSA join for if statement (line 523)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__repr__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__repr__' in the type store
        # Getting the type of 'stypy_return_type' (line 515)
        stypy_return_type_13197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_13197)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__repr__'
        return stypy_return_type_13197


    @norecursion
    def field(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 544)
        None_13198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 30), 'None')
        defaults = [None_13198]
        # Create a new context for function 'field'
        module_type_store = module_type_store.open_function_context('field', 544, 4, False)
        # Assigning a type to the variable 'self' (line 545)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 545, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        recarray.field.__dict__.__setitem__('stypy_localization', localization)
        recarray.field.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        recarray.field.__dict__.__setitem__('stypy_type_store', module_type_store)
        recarray.field.__dict__.__setitem__('stypy_function_name', 'recarray.field')
        recarray.field.__dict__.__setitem__('stypy_param_names_list', ['attr', 'val'])
        recarray.field.__dict__.__setitem__('stypy_varargs_param_name', None)
        recarray.field.__dict__.__setitem__('stypy_kwargs_param_name', None)
        recarray.field.__dict__.__setitem__('stypy_call_defaults', defaults)
        recarray.field.__dict__.__setitem__('stypy_call_varargs', varargs)
        recarray.field.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        recarray.field.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'recarray.field', ['attr', 'val'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'field', localization, ['attr', 'val'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'field(...)' code ##################

        
        # Type idiom detected: calculating its left and rigth part (line 545)
        # Getting the type of 'int' (line 545)
        int_13199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 28), 'int')
        # Getting the type of 'attr' (line 545)
        attr_13200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 22), 'attr')
        
        (may_be_13201, more_types_in_union_13202) = may_be_subtype(int_13199, attr_13200)

        if may_be_13201:

            if more_types_in_union_13202:
                # Runtime conditional SSA (line 545)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'attr' (line 545)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 545, 8), 'attr', remove_not_subtype_from_union(attr_13200, int))
            
            # Assigning a Attribute to a Name (line 546):
            
            # Assigning a Attribute to a Name (line 546):
            
            # Call to __getattribute__(...): (line 546)
            # Processing the call arguments (line 546)
            # Getting the type of 'self' (line 546)
            self_13205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 45), 'self', False)
            str_13206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 546, 51), 'str', 'dtype')
            # Processing the call keyword arguments (line 546)
            kwargs_13207 = {}
            # Getting the type of 'ndarray' (line 546)
            ndarray_13203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 20), 'ndarray', False)
            # Obtaining the member '__getattribute__' of a type (line 546)
            getattribute___13204 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 546, 20), ndarray_13203, '__getattribute__')
            # Calling __getattribute__(args, kwargs) (line 546)
            getattribute___call_result_13208 = invoke(stypy.reporting.localization.Localization(__file__, 546, 20), getattribute___13204, *[self_13205, str_13206], **kwargs_13207)
            
            # Obtaining the member 'names' of a type (line 546)
            names_13209 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 546, 20), getattribute___call_result_13208, 'names')
            # Assigning a type to the variable 'names' (line 546)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 546, 12), 'names', names_13209)
            
            # Assigning a Subscript to a Name (line 547):
            
            # Assigning a Subscript to a Name (line 547):
            
            # Obtaining the type of the subscript
            # Getting the type of 'attr' (line 547)
            attr_13210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 25), 'attr')
            # Getting the type of 'names' (line 547)
            names_13211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 19), 'names')
            # Obtaining the member '__getitem__' of a type (line 547)
            getitem___13212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 547, 19), names_13211, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 547)
            subscript_call_result_13213 = invoke(stypy.reporting.localization.Localization(__file__, 547, 19), getitem___13212, attr_13210)
            
            # Assigning a type to the variable 'attr' (line 547)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 547, 12), 'attr', subscript_call_result_13213)

            if more_types_in_union_13202:
                # SSA join for if statement (line 545)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Attribute to a Name (line 549):
        
        # Assigning a Attribute to a Name (line 549):
        
        # Call to __getattribute__(...): (line 549)
        # Processing the call arguments (line 549)
        # Getting the type of 'self' (line 549)
        self_13216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 45), 'self', False)
        str_13217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 549, 51), 'str', 'dtype')
        # Processing the call keyword arguments (line 549)
        kwargs_13218 = {}
        # Getting the type of 'ndarray' (line 549)
        ndarray_13214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 20), 'ndarray', False)
        # Obtaining the member '__getattribute__' of a type (line 549)
        getattribute___13215 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 549, 20), ndarray_13214, '__getattribute__')
        # Calling __getattribute__(args, kwargs) (line 549)
        getattribute___call_result_13219 = invoke(stypy.reporting.localization.Localization(__file__, 549, 20), getattribute___13215, *[self_13216, str_13217], **kwargs_13218)
        
        # Obtaining the member 'fields' of a type (line 549)
        fields_13220 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 549, 20), getattribute___call_result_13219, 'fields')
        # Assigning a type to the variable 'fielddict' (line 549)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 549, 8), 'fielddict', fields_13220)
        
        # Assigning a Subscript to a Name (line 551):
        
        # Assigning a Subscript to a Name (line 551):
        
        # Obtaining the type of the subscript
        int_13221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 551, 31), 'int')
        slice_13222 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 551, 14), None, int_13221, None)
        
        # Obtaining the type of the subscript
        # Getting the type of 'attr' (line 551)
        attr_13223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 24), 'attr')
        # Getting the type of 'fielddict' (line 551)
        fielddict_13224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 14), 'fielddict')
        # Obtaining the member '__getitem__' of a type (line 551)
        getitem___13225 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 551, 14), fielddict_13224, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 551)
        subscript_call_result_13226 = invoke(stypy.reporting.localization.Localization(__file__, 551, 14), getitem___13225, attr_13223)
        
        # Obtaining the member '__getitem__' of a type (line 551)
        getitem___13227 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 551, 14), subscript_call_result_13226, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 551)
        subscript_call_result_13228 = invoke(stypy.reporting.localization.Localization(__file__, 551, 14), getitem___13227, slice_13222)
        
        # Assigning a type to the variable 'res' (line 551)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 551, 8), 'res', subscript_call_result_13228)
        
        # Type idiom detected: calculating its left and rigth part (line 553)
        # Getting the type of 'val' (line 553)
        val_13229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 11), 'val')
        # Getting the type of 'None' (line 553)
        None_13230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 18), 'None')
        
        (may_be_13231, more_types_in_union_13232) = may_be_none(val_13229, None_13230)

        if may_be_13231:

            if more_types_in_union_13232:
                # Runtime conditional SSA (line 553)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 554):
            
            # Assigning a Call to a Name (line 554):
            
            # Call to getfield(...): (line 554)
            # Getting the type of 'res' (line 554)
            res_13235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 33), 'res', False)
            # Processing the call keyword arguments (line 554)
            kwargs_13236 = {}
            # Getting the type of 'self' (line 554)
            self_13233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 18), 'self', False)
            # Obtaining the member 'getfield' of a type (line 554)
            getfield_13234 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 554, 18), self_13233, 'getfield')
            # Calling getfield(args, kwargs) (line 554)
            getfield_call_result_13237 = invoke(stypy.reporting.localization.Localization(__file__, 554, 18), getfield_13234, *[res_13235], **kwargs_13236)
            
            # Assigning a type to the variable 'obj' (line 554)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 554, 12), 'obj', getfield_call_result_13237)
            
            # Getting the type of 'obj' (line 555)
            obj_13238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 15), 'obj')
            # Obtaining the member 'dtype' of a type (line 555)
            dtype_13239 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 555, 15), obj_13238, 'dtype')
            # Obtaining the member 'fields' of a type (line 555)
            fields_13240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 555, 15), dtype_13239, 'fields')
            # Testing the type of an if condition (line 555)
            if_condition_13241 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 555, 12), fields_13240)
            # Assigning a type to the variable 'if_condition_13241' (line 555)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 555, 12), 'if_condition_13241', if_condition_13241)
            # SSA begins for if statement (line 555)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'obj' (line 556)
            obj_13242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 23), 'obj')
            # Assigning a type to the variable 'stypy_return_type' (line 556)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 556, 16), 'stypy_return_type', obj_13242)
            # SSA join for if statement (line 555)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Call to view(...): (line 557)
            # Processing the call arguments (line 557)
            # Getting the type of 'ndarray' (line 557)
            ndarray_13245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 28), 'ndarray', False)
            # Processing the call keyword arguments (line 557)
            kwargs_13246 = {}
            # Getting the type of 'obj' (line 557)
            obj_13243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 19), 'obj', False)
            # Obtaining the member 'view' of a type (line 557)
            view_13244 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 557, 19), obj_13243, 'view')
            # Calling view(args, kwargs) (line 557)
            view_call_result_13247 = invoke(stypy.reporting.localization.Localization(__file__, 557, 19), view_13244, *[ndarray_13245], **kwargs_13246)
            
            # Assigning a type to the variable 'stypy_return_type' (line 557)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 557, 12), 'stypy_return_type', view_call_result_13247)

            if more_types_in_union_13232:
                # Runtime conditional SSA for else branch (line 553)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_13231) or more_types_in_union_13232):
            
            # Call to setfield(...): (line 559)
            # Processing the call arguments (line 559)
            # Getting the type of 'val' (line 559)
            val_13250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 33), 'val', False)
            # Getting the type of 'res' (line 559)
            res_13251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 39), 'res', False)
            # Processing the call keyword arguments (line 559)
            kwargs_13252 = {}
            # Getting the type of 'self' (line 559)
            self_13248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 19), 'self', False)
            # Obtaining the member 'setfield' of a type (line 559)
            setfield_13249 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 559, 19), self_13248, 'setfield')
            # Calling setfield(args, kwargs) (line 559)
            setfield_call_result_13253 = invoke(stypy.reporting.localization.Localization(__file__, 559, 19), setfield_13249, *[val_13250, res_13251], **kwargs_13252)
            
            # Assigning a type to the variable 'stypy_return_type' (line 559)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 559, 12), 'stypy_return_type', setfield_call_result_13253)

            if (may_be_13231 and more_types_in_union_13232):
                # SSA join for if statement (line 553)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # ################# End of 'field(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'field' in the type store
        # Getting the type of 'stypy_return_type' (line 544)
        stypy_return_type_13254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_13254)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'field'
        return stypy_return_type_13254


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 298, 0, False)
        # Assigning a type to the variable 'self' (line 299)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'recarray.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'recarray' (line 298)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 0), 'recarray', recarray)

# Assigning a Str to a Name (line 406):
str_13255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 15), 'str', 'recarray')
# Getting the type of 'recarray'
recarray_13256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'recarray')
# Setting the type of the member '__name__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), recarray_13256, '__name__', str_13255)

# Assigning a Str to a Name (line 407):
str_13257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 17), 'str', 'numpy')
# Getting the type of 'recarray'
recarray_13258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'recarray')
# Setting the type of the member '__module__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), recarray_13258, '__module__', str_13257)

@norecursion
def fromarrays(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 562)
    None_13259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 32), 'None')
    # Getting the type of 'None' (line 562)
    None_13260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 44), 'None')
    # Getting the type of 'None' (line 562)
    None_13261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 58), 'None')
    # Getting the type of 'None' (line 563)
    None_13262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 21), 'None')
    # Getting the type of 'None' (line 563)
    None_13263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 34), 'None')
    # Getting the type of 'False' (line 563)
    False_13264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 48), 'False')
    # Getting the type of 'None' (line 563)
    None_13265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 65), 'None')
    defaults = [None_13259, None_13260, None_13261, None_13262, None_13263, False_13264, None_13265]
    # Create a new context for function 'fromarrays'
    module_type_store = module_type_store.open_function_context('fromarrays', 562, 0, False)
    
    # Passed parameters checking function
    fromarrays.stypy_localization = localization
    fromarrays.stypy_type_of_self = None
    fromarrays.stypy_type_store = module_type_store
    fromarrays.stypy_function_name = 'fromarrays'
    fromarrays.stypy_param_names_list = ['arrayList', 'dtype', 'shape', 'formats', 'names', 'titles', 'aligned', 'byteorder']
    fromarrays.stypy_varargs_param_name = None
    fromarrays.stypy_kwargs_param_name = None
    fromarrays.stypy_call_defaults = defaults
    fromarrays.stypy_call_varargs = varargs
    fromarrays.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'fromarrays', ['arrayList', 'dtype', 'shape', 'formats', 'names', 'titles', 'aligned', 'byteorder'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'fromarrays', localization, ['arrayList', 'dtype', 'shape', 'formats', 'names', 'titles', 'aligned', 'byteorder'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'fromarrays(...)' code ##################

    str_13266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 575, (-1)), 'str', " create a record array from a (flat) list of arrays\n\n    >>> x1=np.array([1,2,3,4])\n    >>> x2=np.array(['a','dd','xyz','12'])\n    >>> x3=np.array([1.1,2,3,4])\n    >>> r = np.core.records.fromarrays([x1,x2,x3],names='a,b,c')\n    >>> print(r[1])\n    (2, 'dd', 2.0)\n    >>> x1[1]=34\n    >>> r.a\n    array([1, 2, 3, 4])\n    ")
    
    # Assigning a ListComp to a Name (line 577):
    
    # Assigning a ListComp to a Name (line 577):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'arrayList' (line 577)
    arrayList_13272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 40), 'arrayList')
    comprehension_13273 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 577, 17), arrayList_13272)
    # Assigning a type to the variable 'x' (line 577)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 577, 17), 'x', comprehension_13273)
    
    # Call to asarray(...): (line 577)
    # Processing the call arguments (line 577)
    # Getting the type of 'x' (line 577)
    x_13269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 28), 'x', False)
    # Processing the call keyword arguments (line 577)
    kwargs_13270 = {}
    # Getting the type of 'sb' (line 577)
    sb_13267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 17), 'sb', False)
    # Obtaining the member 'asarray' of a type (line 577)
    asarray_13268 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 577, 17), sb_13267, 'asarray')
    # Calling asarray(args, kwargs) (line 577)
    asarray_call_result_13271 = invoke(stypy.reporting.localization.Localization(__file__, 577, 17), asarray_13268, *[x_13269], **kwargs_13270)
    
    list_13274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 577, 17), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 577, 17), list_13274, asarray_call_result_13271)
    # Assigning a type to the variable 'arrayList' (line 577)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 577, 4), 'arrayList', list_13274)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'shape' (line 579)
    shape_13275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 7), 'shape')
    # Getting the type of 'None' (line 579)
    None_13276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 16), 'None')
    # Applying the binary operator 'is' (line 579)
    result_is__13277 = python_operator(stypy.reporting.localization.Localization(__file__, 579, 7), 'is', shape_13275, None_13276)
    
    
    # Getting the type of 'shape' (line 579)
    shape_13278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 24), 'shape')
    int_13279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 579, 33), 'int')
    # Applying the binary operator '==' (line 579)
    result_eq_13280 = python_operator(stypy.reporting.localization.Localization(__file__, 579, 24), '==', shape_13278, int_13279)
    
    # Applying the binary operator 'or' (line 579)
    result_or_keyword_13281 = python_operator(stypy.reporting.localization.Localization(__file__, 579, 7), 'or', result_is__13277, result_eq_13280)
    
    # Testing the type of an if condition (line 579)
    if_condition_13282 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 579, 4), result_or_keyword_13281)
    # Assigning a type to the variable 'if_condition_13282' (line 579)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 579, 4), 'if_condition_13282', if_condition_13282)
    # SSA begins for if statement (line 579)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Attribute to a Name (line 580):
    
    # Assigning a Attribute to a Name (line 580):
    
    # Obtaining the type of the subscript
    int_13283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 580, 26), 'int')
    # Getting the type of 'arrayList' (line 580)
    arrayList_13284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 16), 'arrayList')
    # Obtaining the member '__getitem__' of a type (line 580)
    getitem___13285 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 580, 16), arrayList_13284, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 580)
    subscript_call_result_13286 = invoke(stypy.reporting.localization.Localization(__file__, 580, 16), getitem___13285, int_13283)
    
    # Obtaining the member 'shape' of a type (line 580)
    shape_13287 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 580, 16), subscript_call_result_13286, 'shape')
    # Assigning a type to the variable 'shape' (line 580)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 580, 8), 'shape', shape_13287)
    # SSA join for if statement (line 579)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 582)
    # Getting the type of 'int' (line 582)
    int_13288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 25), 'int')
    # Getting the type of 'shape' (line 582)
    shape_13289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 18), 'shape')
    
    (may_be_13290, more_types_in_union_13291) = may_be_subtype(int_13288, shape_13289)

    if may_be_13290:

        if more_types_in_union_13291:
            # Runtime conditional SSA (line 582)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'shape' (line 582)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 582, 4), 'shape', remove_not_subtype_from_union(shape_13289, int))
        
        # Assigning a Tuple to a Name (line 583):
        
        # Assigning a Tuple to a Name (line 583):
        
        # Obtaining an instance of the builtin type 'tuple' (line 583)
        tuple_13292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 583, 17), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 583)
        # Adding element type (line 583)
        # Getting the type of 'shape' (line 583)
        shape_13293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 17), 'shape')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 583, 17), tuple_13292, shape_13293)
        
        # Assigning a type to the variable 'shape' (line 583)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 583, 8), 'shape', tuple_13292)

        if more_types_in_union_13291:
            # SSA join for if statement (line 582)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'formats' (line 585)
    formats_13294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 7), 'formats')
    # Getting the type of 'None' (line 585)
    None_13295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 18), 'None')
    # Applying the binary operator 'is' (line 585)
    result_is__13296 = python_operator(stypy.reporting.localization.Localization(__file__, 585, 7), 'is', formats_13294, None_13295)
    
    
    # Getting the type of 'dtype' (line 585)
    dtype_13297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 27), 'dtype')
    # Getting the type of 'None' (line 585)
    None_13298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 36), 'None')
    # Applying the binary operator 'is' (line 585)
    result_is__13299 = python_operator(stypy.reporting.localization.Localization(__file__, 585, 27), 'is', dtype_13297, None_13298)
    
    # Applying the binary operator 'and' (line 585)
    result_and_keyword_13300 = python_operator(stypy.reporting.localization.Localization(__file__, 585, 7), 'and', result_is__13296, result_is__13299)
    
    # Testing the type of an if condition (line 585)
    if_condition_13301 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 585, 4), result_and_keyword_13300)
    # Assigning a type to the variable 'if_condition_13301' (line 585)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 585, 4), 'if_condition_13301', if_condition_13301)
    # SSA begins for if statement (line 585)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a List to a Name (line 588):
    
    # Assigning a List to a Name (line 588):
    
    # Obtaining an instance of the builtin type 'list' (line 588)
    list_13302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 588, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 588)
    
    # Assigning a type to the variable 'formats' (line 588)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 588, 8), 'formats', list_13302)
    
    # Getting the type of 'arrayList' (line 589)
    arrayList_13303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 19), 'arrayList')
    # Testing the type of a for loop iterable (line 589)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 589, 8), arrayList_13303)
    # Getting the type of the for loop variable (line 589)
    for_loop_var_13304 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 589, 8), arrayList_13303)
    # Assigning a type to the variable 'obj' (line 589)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 589, 8), 'obj', for_loop_var_13304)
    # SSA begins for a for statement (line 589)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    
    # Call to isinstance(...): (line 590)
    # Processing the call arguments (line 590)
    # Getting the type of 'obj' (line 590)
    obj_13306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 30), 'obj', False)
    # Getting the type of 'ndarray' (line 590)
    ndarray_13307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 35), 'ndarray', False)
    # Processing the call keyword arguments (line 590)
    kwargs_13308 = {}
    # Getting the type of 'isinstance' (line 590)
    isinstance_13305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 19), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 590)
    isinstance_call_result_13309 = invoke(stypy.reporting.localization.Localization(__file__, 590, 19), isinstance_13305, *[obj_13306, ndarray_13307], **kwargs_13308)
    
    # Applying the 'not' unary operator (line 590)
    result_not__13310 = python_operator(stypy.reporting.localization.Localization(__file__, 590, 15), 'not', isinstance_call_result_13309)
    
    # Testing the type of an if condition (line 590)
    if_condition_13311 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 590, 12), result_not__13310)
    # Assigning a type to the variable 'if_condition_13311' (line 590)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 590, 12), 'if_condition_13311', if_condition_13311)
    # SSA begins for if statement (line 590)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 591)
    # Processing the call arguments (line 591)
    str_13313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 591, 33), 'str', 'item in the array list must be an ndarray.')
    # Processing the call keyword arguments (line 591)
    kwargs_13314 = {}
    # Getting the type of 'ValueError' (line 591)
    ValueError_13312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 22), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 591)
    ValueError_call_result_13315 = invoke(stypy.reporting.localization.Localization(__file__, 591, 22), ValueError_13312, *[str_13313], **kwargs_13314)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 591, 16), ValueError_call_result_13315, 'raise parameter', BaseException)
    # SSA join for if statement (line 590)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to append(...): (line 592)
    # Processing the call arguments (line 592)
    # Getting the type of 'obj' (line 592)
    obj_13318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 27), 'obj', False)
    # Obtaining the member 'dtype' of a type (line 592)
    dtype_13319 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 592, 27), obj_13318, 'dtype')
    # Obtaining the member 'str' of a type (line 592)
    str_13320 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 592, 27), dtype_13319, 'str')
    # Processing the call keyword arguments (line 592)
    kwargs_13321 = {}
    # Getting the type of 'formats' (line 592)
    formats_13316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 12), 'formats', False)
    # Obtaining the member 'append' of a type (line 592)
    append_13317 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 592, 12), formats_13316, 'append')
    # Calling append(args, kwargs) (line 592)
    append_call_result_13322 = invoke(stypy.reporting.localization.Localization(__file__, 592, 12), append_13317, *[str_13320], **kwargs_13321)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 593):
    
    # Assigning a Call to a Name (line 593):
    
    # Call to join(...): (line 593)
    # Processing the call arguments (line 593)
    # Getting the type of 'formats' (line 593)
    formats_13325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 27), 'formats', False)
    # Processing the call keyword arguments (line 593)
    kwargs_13326 = {}
    str_13323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 593, 18), 'str', ',')
    # Obtaining the member 'join' of a type (line 593)
    join_13324 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 593, 18), str_13323, 'join')
    # Calling join(args, kwargs) (line 593)
    join_call_result_13327 = invoke(stypy.reporting.localization.Localization(__file__, 593, 18), join_13324, *[formats_13325], **kwargs_13326)
    
    # Assigning a type to the variable 'formats' (line 593)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 593, 8), 'formats', join_call_result_13327)
    # SSA join for if statement (line 585)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 595)
    # Getting the type of 'dtype' (line 595)
    dtype_13328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 4), 'dtype')
    # Getting the type of 'None' (line 595)
    None_13329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 20), 'None')
    
    (may_be_13330, more_types_in_union_13331) = may_not_be_none(dtype_13328, None_13329)

    if may_be_13330:

        if more_types_in_union_13331:
            # Runtime conditional SSA (line 595)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 596):
        
        # Assigning a Call to a Name (line 596):
        
        # Call to dtype(...): (line 596)
        # Processing the call arguments (line 596)
        # Getting the type of 'dtype' (line 596)
        dtype_13334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 25), 'dtype', False)
        # Processing the call keyword arguments (line 596)
        kwargs_13335 = {}
        # Getting the type of 'sb' (line 596)
        sb_13332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 16), 'sb', False)
        # Obtaining the member 'dtype' of a type (line 596)
        dtype_13333 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 596, 16), sb_13332, 'dtype')
        # Calling dtype(args, kwargs) (line 596)
        dtype_call_result_13336 = invoke(stypy.reporting.localization.Localization(__file__, 596, 16), dtype_13333, *[dtype_13334], **kwargs_13335)
        
        # Assigning a type to the variable 'descr' (line 596)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 596, 8), 'descr', dtype_call_result_13336)
        
        # Assigning a Attribute to a Name (line 597):
        
        # Assigning a Attribute to a Name (line 597):
        # Getting the type of 'descr' (line 597)
        descr_13337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 17), 'descr')
        # Obtaining the member 'names' of a type (line 597)
        names_13338 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 597, 17), descr_13337, 'names')
        # Assigning a type to the variable '_names' (line 597)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 597, 8), '_names', names_13338)

        if more_types_in_union_13331:
            # Runtime conditional SSA for else branch (line 595)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_13330) or more_types_in_union_13331):
        
        # Assigning a Call to a Name (line 599):
        
        # Assigning a Call to a Name (line 599):
        
        # Call to format_parser(...): (line 599)
        # Processing the call arguments (line 599)
        # Getting the type of 'formats' (line 599)
        formats_13340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 31), 'formats', False)
        # Getting the type of 'names' (line 599)
        names_13341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 40), 'names', False)
        # Getting the type of 'titles' (line 599)
        titles_13342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 47), 'titles', False)
        # Getting the type of 'aligned' (line 599)
        aligned_13343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 55), 'aligned', False)
        # Getting the type of 'byteorder' (line 599)
        byteorder_13344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 64), 'byteorder', False)
        # Processing the call keyword arguments (line 599)
        kwargs_13345 = {}
        # Getting the type of 'format_parser' (line 599)
        format_parser_13339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 17), 'format_parser', False)
        # Calling format_parser(args, kwargs) (line 599)
        format_parser_call_result_13346 = invoke(stypy.reporting.localization.Localization(__file__, 599, 17), format_parser_13339, *[formats_13340, names_13341, titles_13342, aligned_13343, byteorder_13344], **kwargs_13345)
        
        # Assigning a type to the variable 'parsed' (line 599)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 599, 8), 'parsed', format_parser_call_result_13346)
        
        # Assigning a Attribute to a Name (line 600):
        
        # Assigning a Attribute to a Name (line 600):
        # Getting the type of 'parsed' (line 600)
        parsed_13347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 17), 'parsed')
        # Obtaining the member '_names' of a type (line 600)
        _names_13348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 600, 17), parsed_13347, '_names')
        # Assigning a type to the variable '_names' (line 600)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 600, 8), '_names', _names_13348)
        
        # Assigning a Attribute to a Name (line 601):
        
        # Assigning a Attribute to a Name (line 601):
        # Getting the type of 'parsed' (line 601)
        parsed_13349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 16), 'parsed')
        # Obtaining the member '_descr' of a type (line 601)
        _descr_13350 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 601, 16), parsed_13349, '_descr')
        # Assigning a type to the variable 'descr' (line 601)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 601, 8), 'descr', _descr_13350)

        if (may_be_13330 and more_types_in_union_13331):
            # SSA join for if statement (line 595)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    
    # Call to len(...): (line 604)
    # Processing the call arguments (line 604)
    # Getting the type of 'descr' (line 604)
    descr_13352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 11), 'descr', False)
    # Processing the call keyword arguments (line 604)
    kwargs_13353 = {}
    # Getting the type of 'len' (line 604)
    len_13351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 7), 'len', False)
    # Calling len(args, kwargs) (line 604)
    len_call_result_13354 = invoke(stypy.reporting.localization.Localization(__file__, 604, 7), len_13351, *[descr_13352], **kwargs_13353)
    
    
    # Call to len(...): (line 604)
    # Processing the call arguments (line 604)
    # Getting the type of 'arrayList' (line 604)
    arrayList_13356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 25), 'arrayList', False)
    # Processing the call keyword arguments (line 604)
    kwargs_13357 = {}
    # Getting the type of 'len' (line 604)
    len_13355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 21), 'len', False)
    # Calling len(args, kwargs) (line 604)
    len_call_result_13358 = invoke(stypy.reporting.localization.Localization(__file__, 604, 21), len_13355, *[arrayList_13356], **kwargs_13357)
    
    # Applying the binary operator '!=' (line 604)
    result_ne_13359 = python_operator(stypy.reporting.localization.Localization(__file__, 604, 7), '!=', len_call_result_13354, len_call_result_13358)
    
    # Testing the type of an if condition (line 604)
    if_condition_13360 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 604, 4), result_ne_13359)
    # Assigning a type to the variable 'if_condition_13360' (line 604)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 604, 4), 'if_condition_13360', if_condition_13360)
    # SSA begins for if statement (line 604)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 605)
    # Processing the call arguments (line 605)
    str_13362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 605, 25), 'str', 'mismatch between the number of fields and the number of arrays')
    # Processing the call keyword arguments (line 605)
    kwargs_13363 = {}
    # Getting the type of 'ValueError' (line 605)
    ValueError_13361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 605)
    ValueError_call_result_13364 = invoke(stypy.reporting.localization.Localization(__file__, 605, 14), ValueError_13361, *[str_13362], **kwargs_13363)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 605, 8), ValueError_call_result_13364, 'raise parameter', BaseException)
    # SSA join for if statement (line 604)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Attribute to a Name (line 608):
    
    # Assigning a Attribute to a Name (line 608):
    
    # Obtaining the type of the subscript
    int_13365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 608, 15), 'int')
    # Getting the type of 'descr' (line 608)
    descr_13366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 9), 'descr')
    # Obtaining the member '__getitem__' of a type (line 608)
    getitem___13367 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 608, 9), descr_13366, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 608)
    subscript_call_result_13368 = invoke(stypy.reporting.localization.Localization(__file__, 608, 9), getitem___13367, int_13365)
    
    # Obtaining the member 'shape' of a type (line 608)
    shape_13369 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 608, 9), subscript_call_result_13368, 'shape')
    # Assigning a type to the variable 'd0' (line 608)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 608, 4), 'd0', shape_13369)
    
    # Assigning a Call to a Name (line 609):
    
    # Assigning a Call to a Name (line 609):
    
    # Call to len(...): (line 609)
    # Processing the call arguments (line 609)
    # Getting the type of 'd0' (line 609)
    d0_13371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 609, 13), 'd0', False)
    # Processing the call keyword arguments (line 609)
    kwargs_13372 = {}
    # Getting the type of 'len' (line 609)
    len_13370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 609, 9), 'len', False)
    # Calling len(args, kwargs) (line 609)
    len_call_result_13373 = invoke(stypy.reporting.localization.Localization(__file__, 609, 9), len_13370, *[d0_13371], **kwargs_13372)
    
    # Assigning a type to the variable 'nn' (line 609)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 609, 4), 'nn', len_call_result_13373)
    
    
    # Getting the type of 'nn' (line 610)
    nn_13374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 7), 'nn')
    int_13375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 610, 12), 'int')
    # Applying the binary operator '>' (line 610)
    result_gt_13376 = python_operator(stypy.reporting.localization.Localization(__file__, 610, 7), '>', nn_13374, int_13375)
    
    # Testing the type of an if condition (line 610)
    if_condition_13377 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 610, 4), result_gt_13376)
    # Assigning a type to the variable 'if_condition_13377' (line 610)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 610, 4), 'if_condition_13377', if_condition_13377)
    # SSA begins for if statement (line 610)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 611):
    
    # Assigning a Subscript to a Name (line 611):
    
    # Obtaining the type of the subscript
    
    # Getting the type of 'nn' (line 611)
    nn_13378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 24), 'nn')
    # Applying the 'usub' unary operator (line 611)
    result___neg___13379 = python_operator(stypy.reporting.localization.Localization(__file__, 611, 23), 'usub', nn_13378)
    
    slice_13380 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 611, 16), None, result___neg___13379, None)
    # Getting the type of 'shape' (line 611)
    shape_13381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 16), 'shape')
    # Obtaining the member '__getitem__' of a type (line 611)
    getitem___13382 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 611, 16), shape_13381, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 611)
    subscript_call_result_13383 = invoke(stypy.reporting.localization.Localization(__file__, 611, 16), getitem___13382, slice_13380)
    
    # Assigning a type to the variable 'shape' (line 611)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 611, 8), 'shape', subscript_call_result_13383)
    # SSA join for if statement (line 610)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to enumerate(...): (line 613)
    # Processing the call arguments (line 613)
    # Getting the type of 'arrayList' (line 613)
    arrayList_13385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 28), 'arrayList', False)
    # Processing the call keyword arguments (line 613)
    kwargs_13386 = {}
    # Getting the type of 'enumerate' (line 613)
    enumerate_13384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 18), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 613)
    enumerate_call_result_13387 = invoke(stypy.reporting.localization.Localization(__file__, 613, 18), enumerate_13384, *[arrayList_13385], **kwargs_13386)
    
    # Testing the type of a for loop iterable (line 613)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 613, 4), enumerate_call_result_13387)
    # Getting the type of the for loop variable (line 613)
    for_loop_var_13388 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 613, 4), enumerate_call_result_13387)
    # Assigning a type to the variable 'k' (line 613)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 613, 4), 'k', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 613, 4), for_loop_var_13388))
    # Assigning a type to the variable 'obj' (line 613)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 613, 4), 'obj', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 613, 4), for_loop_var_13388))
    # SSA begins for a for statement (line 613)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 614):
    
    # Assigning a Call to a Name (line 614):
    
    # Call to len(...): (line 614)
    # Processing the call arguments (line 614)
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 614)
    k_13390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 614, 23), 'k', False)
    # Getting the type of 'descr' (line 614)
    descr_13391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 614, 17), 'descr', False)
    # Obtaining the member '__getitem__' of a type (line 614)
    getitem___13392 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 614, 17), descr_13391, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 614)
    subscript_call_result_13393 = invoke(stypy.reporting.localization.Localization(__file__, 614, 17), getitem___13392, k_13390)
    
    # Obtaining the member 'shape' of a type (line 614)
    shape_13394 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 614, 17), subscript_call_result_13393, 'shape')
    # Processing the call keyword arguments (line 614)
    kwargs_13395 = {}
    # Getting the type of 'len' (line 614)
    len_13389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 614, 13), 'len', False)
    # Calling len(args, kwargs) (line 614)
    len_call_result_13396 = invoke(stypy.reporting.localization.Localization(__file__, 614, 13), len_13389, *[shape_13394], **kwargs_13395)
    
    # Assigning a type to the variable 'nn' (line 614)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 614, 8), 'nn', len_call_result_13396)
    
    # Assigning a Subscript to a Name (line 615):
    
    # Assigning a Subscript to a Name (line 615):
    
    # Obtaining the type of the subscript
    
    # Call to len(...): (line 615)
    # Processing the call arguments (line 615)
    # Getting the type of 'obj' (line 615)
    obj_13398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 35), 'obj', False)
    # Obtaining the member 'shape' of a type (line 615)
    shape_13399 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 615, 35), obj_13398, 'shape')
    # Processing the call keyword arguments (line 615)
    kwargs_13400 = {}
    # Getting the type of 'len' (line 615)
    len_13397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 31), 'len', False)
    # Calling len(args, kwargs) (line 615)
    len_call_result_13401 = invoke(stypy.reporting.localization.Localization(__file__, 615, 31), len_13397, *[shape_13399], **kwargs_13400)
    
    # Getting the type of 'nn' (line 615)
    nn_13402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 48), 'nn')
    # Applying the binary operator '-' (line 615)
    result_sub_13403 = python_operator(stypy.reporting.localization.Localization(__file__, 615, 31), '-', len_call_result_13401, nn_13402)
    
    slice_13404 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 615, 20), None, result_sub_13403, None)
    # Getting the type of 'obj' (line 615)
    obj_13405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 20), 'obj')
    # Obtaining the member 'shape' of a type (line 615)
    shape_13406 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 615, 20), obj_13405, 'shape')
    # Obtaining the member '__getitem__' of a type (line 615)
    getitem___13407 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 615, 20), shape_13406, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 615)
    subscript_call_result_13408 = invoke(stypy.reporting.localization.Localization(__file__, 615, 20), getitem___13407, slice_13404)
    
    # Assigning a type to the variable 'testshape' (line 615)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 615, 8), 'testshape', subscript_call_result_13408)
    
    
    # Getting the type of 'testshape' (line 616)
    testshape_13409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 11), 'testshape')
    # Getting the type of 'shape' (line 616)
    shape_13410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 24), 'shape')
    # Applying the binary operator '!=' (line 616)
    result_ne_13411 = python_operator(stypy.reporting.localization.Localization(__file__, 616, 11), '!=', testshape_13409, shape_13410)
    
    # Testing the type of an if condition (line 616)
    if_condition_13412 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 616, 8), result_ne_13411)
    # Assigning a type to the variable 'if_condition_13412' (line 616)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 616, 8), 'if_condition_13412', if_condition_13412)
    # SSA begins for if statement (line 616)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 617)
    # Processing the call arguments (line 617)
    str_13414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 617, 29), 'str', 'array-shape mismatch in array %d')
    # Getting the type of 'k' (line 617)
    k_13415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 66), 'k', False)
    # Applying the binary operator '%' (line 617)
    result_mod_13416 = python_operator(stypy.reporting.localization.Localization(__file__, 617, 29), '%', str_13414, k_13415)
    
    # Processing the call keyword arguments (line 617)
    kwargs_13417 = {}
    # Getting the type of 'ValueError' (line 617)
    ValueError_13413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 617)
    ValueError_call_result_13418 = invoke(stypy.reporting.localization.Localization(__file__, 617, 18), ValueError_13413, *[result_mod_13416], **kwargs_13417)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 617, 12), ValueError_call_result_13418, 'raise parameter', BaseException)
    # SSA join for if statement (line 616)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 619):
    
    # Assigning a Call to a Name (line 619):
    
    # Call to recarray(...): (line 619)
    # Processing the call arguments (line 619)
    # Getting the type of 'shape' (line 619)
    shape_13420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 22), 'shape', False)
    # Getting the type of 'descr' (line 619)
    descr_13421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 29), 'descr', False)
    # Processing the call keyword arguments (line 619)
    kwargs_13422 = {}
    # Getting the type of 'recarray' (line 619)
    recarray_13419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 13), 'recarray', False)
    # Calling recarray(args, kwargs) (line 619)
    recarray_call_result_13423 = invoke(stypy.reporting.localization.Localization(__file__, 619, 13), recarray_13419, *[shape_13420, descr_13421], **kwargs_13422)
    
    # Assigning a type to the variable '_array' (line 619)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 619, 4), '_array', recarray_call_result_13423)
    
    
    # Call to range(...): (line 622)
    # Processing the call arguments (line 622)
    
    # Call to len(...): (line 622)
    # Processing the call arguments (line 622)
    # Getting the type of 'arrayList' (line 622)
    arrayList_13426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 23), 'arrayList', False)
    # Processing the call keyword arguments (line 622)
    kwargs_13427 = {}
    # Getting the type of 'len' (line 622)
    len_13425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 19), 'len', False)
    # Calling len(args, kwargs) (line 622)
    len_call_result_13428 = invoke(stypy.reporting.localization.Localization(__file__, 622, 19), len_13425, *[arrayList_13426], **kwargs_13427)
    
    # Processing the call keyword arguments (line 622)
    kwargs_13429 = {}
    # Getting the type of 'range' (line 622)
    range_13424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 13), 'range', False)
    # Calling range(args, kwargs) (line 622)
    range_call_result_13430 = invoke(stypy.reporting.localization.Localization(__file__, 622, 13), range_13424, *[len_call_result_13428], **kwargs_13429)
    
    # Testing the type of a for loop iterable (line 622)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 622, 4), range_call_result_13430)
    # Getting the type of the for loop variable (line 622)
    for_loop_var_13431 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 622, 4), range_call_result_13430)
    # Assigning a type to the variable 'i' (line 622)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 622, 4), 'i', for_loop_var_13431)
    # SSA begins for a for statement (line 622)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Subscript to a Subscript (line 623):
    
    # Assigning a Subscript to a Subscript (line 623):
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 623)
    i_13432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 38), 'i')
    # Getting the type of 'arrayList' (line 623)
    arrayList_13433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 28), 'arrayList')
    # Obtaining the member '__getitem__' of a type (line 623)
    getitem___13434 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 623, 28), arrayList_13433, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 623)
    subscript_call_result_13435 = invoke(stypy.reporting.localization.Localization(__file__, 623, 28), getitem___13434, i_13432)
    
    # Getting the type of '_array' (line 623)
    _array_13436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 8), '_array')
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 623)
    i_13437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 22), 'i')
    # Getting the type of '_names' (line 623)
    _names_13438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 15), '_names')
    # Obtaining the member '__getitem__' of a type (line 623)
    getitem___13439 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 623, 15), _names_13438, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 623)
    subscript_call_result_13440 = invoke(stypy.reporting.localization.Localization(__file__, 623, 15), getitem___13439, i_13437)
    
    # Storing an element on a container (line 623)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 623, 8), _array_13436, (subscript_call_result_13440, subscript_call_result_13435))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of '_array' (line 625)
    _array_13441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 11), '_array')
    # Assigning a type to the variable 'stypy_return_type' (line 625)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 625, 4), 'stypy_return_type', _array_13441)
    
    # ################# End of 'fromarrays(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'fromarrays' in the type store
    # Getting the type of 'stypy_return_type' (line 562)
    stypy_return_type_13442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_13442)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'fromarrays'
    return stypy_return_type_13442

# Assigning a type to the variable 'fromarrays' (line 562)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 562, 0), 'fromarrays', fromarrays)

@norecursion
def fromrecords(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 628)
    None_13443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 628, 31), 'None')
    # Getting the type of 'None' (line 628)
    None_13444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 628, 43), 'None')
    # Getting the type of 'None' (line 628)
    None_13445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 628, 57), 'None')
    # Getting the type of 'None' (line 628)
    None_13446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 628, 69), 'None')
    # Getting the type of 'None' (line 629)
    None_13447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 629, 23), 'None')
    # Getting the type of 'False' (line 629)
    False_13448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 629, 37), 'False')
    # Getting the type of 'None' (line 629)
    None_13449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 629, 54), 'None')
    defaults = [None_13443, None_13444, None_13445, None_13446, None_13447, False_13448, None_13449]
    # Create a new context for function 'fromrecords'
    module_type_store = module_type_store.open_function_context('fromrecords', 628, 0, False)
    
    # Passed parameters checking function
    fromrecords.stypy_localization = localization
    fromrecords.stypy_type_of_self = None
    fromrecords.stypy_type_store = module_type_store
    fromrecords.stypy_function_name = 'fromrecords'
    fromrecords.stypy_param_names_list = ['recList', 'dtype', 'shape', 'formats', 'names', 'titles', 'aligned', 'byteorder']
    fromrecords.stypy_varargs_param_name = None
    fromrecords.stypy_kwargs_param_name = None
    fromrecords.stypy_call_defaults = defaults
    fromrecords.stypy_call_varargs = varargs
    fromrecords.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'fromrecords', ['recList', 'dtype', 'shape', 'formats', 'names', 'titles', 'aligned', 'byteorder'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'fromrecords', localization, ['recList', 'dtype', 'shape', 'formats', 'names', 'titles', 'aligned', 'byteorder'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'fromrecords(...)' code ##################

    str_13450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 656, (-1)), 'str', " create a recarray from a list of records in text form\n\n        The data in the same field can be heterogeneous, they will be promoted\n        to the highest data type.  This method is intended for creating\n        smaller record arrays.  If used to create large array without formats\n        defined\n\n        r=fromrecords([(2,3.,'abc')]*100000)\n\n        it can be slow.\n\n        If formats is None, then this will auto-detect formats. Use list of\n        tuples rather than list of lists for faster processing.\n\n    >>> r=np.core.records.fromrecords([(456,'dbe',1.2),(2,'de',1.3)],\n    ... names='col1,col2,col3')\n    >>> print(r[0])\n    (456, 'dbe', 1.2)\n    >>> r.col1\n    array([456,   2])\n    >>> r.col2\n    array(['dbe', 'de'],\n          dtype='|S3')\n    >>> import pickle\n    >>> print(pickle.loads(pickle.dumps(r)))\n    [(456, 'dbe', 1.2) (2, 'de', 1.3)]\n    ")
    
    # Assigning a Call to a Name (line 658):
    
    # Assigning a Call to a Name (line 658):
    
    # Call to len(...): (line 658)
    # Processing the call arguments (line 658)
    
    # Obtaining the type of the subscript
    int_13452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 658, 26), 'int')
    # Getting the type of 'recList' (line 658)
    recList_13453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 658, 18), 'recList', False)
    # Obtaining the member '__getitem__' of a type (line 658)
    getitem___13454 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 658, 18), recList_13453, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 658)
    subscript_call_result_13455 = invoke(stypy.reporting.localization.Localization(__file__, 658, 18), getitem___13454, int_13452)
    
    # Processing the call keyword arguments (line 658)
    kwargs_13456 = {}
    # Getting the type of 'len' (line 658)
    len_13451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 658, 14), 'len', False)
    # Calling len(args, kwargs) (line 658)
    len_call_result_13457 = invoke(stypy.reporting.localization.Localization(__file__, 658, 14), len_13451, *[subscript_call_result_13455], **kwargs_13456)
    
    # Assigning a type to the variable 'nfields' (line 658)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 658, 4), 'nfields', len_call_result_13457)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'formats' (line 659)
    formats_13458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 7), 'formats')
    # Getting the type of 'None' (line 659)
    None_13459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 18), 'None')
    # Applying the binary operator 'is' (line 659)
    result_is__13460 = python_operator(stypy.reporting.localization.Localization(__file__, 659, 7), 'is', formats_13458, None_13459)
    
    
    # Getting the type of 'dtype' (line 659)
    dtype_13461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 27), 'dtype')
    # Getting the type of 'None' (line 659)
    None_13462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 36), 'None')
    # Applying the binary operator 'is' (line 659)
    result_is__13463 = python_operator(stypy.reporting.localization.Localization(__file__, 659, 27), 'is', dtype_13461, None_13462)
    
    # Applying the binary operator 'and' (line 659)
    result_and_keyword_13464 = python_operator(stypy.reporting.localization.Localization(__file__, 659, 7), 'and', result_is__13460, result_is__13463)
    
    # Testing the type of an if condition (line 659)
    if_condition_13465 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 659, 4), result_and_keyword_13464)
    # Assigning a type to the variable 'if_condition_13465' (line 659)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 659, 4), 'if_condition_13465', if_condition_13465)
    # SSA begins for if statement (line 659)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 660):
    
    # Assigning a Call to a Name (line 660):
    
    # Call to array(...): (line 660)
    # Processing the call arguments (line 660)
    # Getting the type of 'recList' (line 660)
    recList_13468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 23), 'recList', False)
    # Processing the call keyword arguments (line 660)
    # Getting the type of 'object' (line 660)
    object_13469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 38), 'object', False)
    keyword_13470 = object_13469
    kwargs_13471 = {'dtype': keyword_13470}
    # Getting the type of 'sb' (line 660)
    sb_13466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 14), 'sb', False)
    # Obtaining the member 'array' of a type (line 660)
    array_13467 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 660, 14), sb_13466, 'array')
    # Calling array(args, kwargs) (line 660)
    array_call_result_13472 = invoke(stypy.reporting.localization.Localization(__file__, 660, 14), array_13467, *[recList_13468], **kwargs_13471)
    
    # Assigning a type to the variable 'obj' (line 660)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 660, 8), 'obj', array_call_result_13472)
    
    # Assigning a ListComp to a Name (line 661):
    
    # Assigning a ListComp to a Name (line 661):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to range(...): (line 661)
    # Processing the call arguments (line 661)
    # Getting the type of 'nfields' (line 661)
    nfields_13486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 65), 'nfields', False)
    # Processing the call keyword arguments (line 661)
    kwargs_13487 = {}
    # Getting the type of 'range' (line 661)
    range_13485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 59), 'range', False)
    # Calling range(args, kwargs) (line 661)
    range_call_result_13488 = invoke(stypy.reporting.localization.Localization(__file__, 661, 59), range_13485, *[nfields_13486], **kwargs_13487)
    
    comprehension_13489 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 661, 19), range_call_result_13488)
    # Assigning a type to the variable 'i' (line 661)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 661, 19), 'i', comprehension_13489)
    
    # Call to array(...): (line 661)
    # Processing the call arguments (line 661)
    
    # Call to tolist(...): (line 661)
    # Processing the call keyword arguments (line 661)
    kwargs_13481 = {}
    
    # Obtaining the type of the subscript
    Ellipsis_13475 = Ellipsis
    # Getting the type of 'i' (line 661)
    i_13476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 37), 'i', False)
    # Getting the type of 'obj' (line 661)
    obj_13477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 28), 'obj', False)
    # Obtaining the member '__getitem__' of a type (line 661)
    getitem___13478 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 661, 28), obj_13477, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 661)
    subscript_call_result_13479 = invoke(stypy.reporting.localization.Localization(__file__, 661, 28), getitem___13478, (Ellipsis_13475, i_13476))
    
    # Obtaining the member 'tolist' of a type (line 661)
    tolist_13480 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 661, 28), subscript_call_result_13479, 'tolist')
    # Calling tolist(args, kwargs) (line 661)
    tolist_call_result_13482 = invoke(stypy.reporting.localization.Localization(__file__, 661, 28), tolist_13480, *[], **kwargs_13481)
    
    # Processing the call keyword arguments (line 661)
    kwargs_13483 = {}
    # Getting the type of 'sb' (line 661)
    sb_13473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 19), 'sb', False)
    # Obtaining the member 'array' of a type (line 661)
    array_13474 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 661, 19), sb_13473, 'array')
    # Calling array(args, kwargs) (line 661)
    array_call_result_13484 = invoke(stypy.reporting.localization.Localization(__file__, 661, 19), array_13474, *[tolist_call_result_13482], **kwargs_13483)
    
    list_13490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 661, 19), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 661, 19), list_13490, array_call_result_13484)
    # Assigning a type to the variable 'arrlist' (line 661)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 661, 8), 'arrlist', list_13490)
    
    # Call to fromarrays(...): (line 662)
    # Processing the call arguments (line 662)
    # Getting the type of 'arrlist' (line 662)
    arrlist_13492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 26), 'arrlist', False)
    # Processing the call keyword arguments (line 662)
    # Getting the type of 'formats' (line 662)
    formats_13493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 43), 'formats', False)
    keyword_13494 = formats_13493
    # Getting the type of 'shape' (line 662)
    shape_13495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 58), 'shape', False)
    keyword_13496 = shape_13495
    # Getting the type of 'names' (line 662)
    names_13497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 71), 'names', False)
    keyword_13498 = names_13497
    # Getting the type of 'titles' (line 663)
    titles_13499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 33), 'titles', False)
    keyword_13500 = titles_13499
    # Getting the type of 'aligned' (line 663)
    aligned_13501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 49), 'aligned', False)
    keyword_13502 = aligned_13501
    # Getting the type of 'byteorder' (line 663)
    byteorder_13503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 68), 'byteorder', False)
    keyword_13504 = byteorder_13503
    kwargs_13505 = {'shape': keyword_13496, 'titles': keyword_13500, 'names': keyword_13498, 'formats': keyword_13494, 'aligned': keyword_13502, 'byteorder': keyword_13504}
    # Getting the type of 'fromarrays' (line 662)
    fromarrays_13491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 15), 'fromarrays', False)
    # Calling fromarrays(args, kwargs) (line 662)
    fromarrays_call_result_13506 = invoke(stypy.reporting.localization.Localization(__file__, 662, 15), fromarrays_13491, *[arrlist_13492], **kwargs_13505)
    
    # Assigning a type to the variable 'stypy_return_type' (line 662)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 662, 8), 'stypy_return_type', fromarrays_call_result_13506)
    # SSA join for if statement (line 659)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 665)
    # Getting the type of 'dtype' (line 665)
    dtype_13507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 4), 'dtype')
    # Getting the type of 'None' (line 665)
    None_13508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 20), 'None')
    
    (may_be_13509, more_types_in_union_13510) = may_not_be_none(dtype_13507, None_13508)

    if may_be_13509:

        if more_types_in_union_13510:
            # Runtime conditional SSA (line 665)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 666):
        
        # Assigning a Call to a Name (line 666):
        
        # Call to dtype(...): (line 666)
        # Processing the call arguments (line 666)
        
        # Obtaining an instance of the builtin type 'tuple' (line 666)
        tuple_13513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 666, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 666)
        # Adding element type (line 666)
        # Getting the type of 'record' (line 666)
        record_13514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 26), 'record', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 666, 26), tuple_13513, record_13514)
        # Adding element type (line 666)
        # Getting the type of 'dtype' (line 666)
        dtype_13515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 34), 'dtype', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 666, 26), tuple_13513, dtype_13515)
        
        # Processing the call keyword arguments (line 666)
        kwargs_13516 = {}
        # Getting the type of 'sb' (line 666)
        sb_13511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 16), 'sb', False)
        # Obtaining the member 'dtype' of a type (line 666)
        dtype_13512 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 666, 16), sb_13511, 'dtype')
        # Calling dtype(args, kwargs) (line 666)
        dtype_call_result_13517 = invoke(stypy.reporting.localization.Localization(__file__, 666, 16), dtype_13512, *[tuple_13513], **kwargs_13516)
        
        # Assigning a type to the variable 'descr' (line 666)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 666, 8), 'descr', dtype_call_result_13517)

        if more_types_in_union_13510:
            # Runtime conditional SSA for else branch (line 665)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_13509) or more_types_in_union_13510):
        
        # Assigning a Attribute to a Name (line 668):
        
        # Assigning a Attribute to a Name (line 668):
        
        # Call to format_parser(...): (line 668)
        # Processing the call arguments (line 668)
        # Getting the type of 'formats' (line 668)
        formats_13519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 30), 'formats', False)
        # Getting the type of 'names' (line 668)
        names_13520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 39), 'names', False)
        # Getting the type of 'titles' (line 668)
        titles_13521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 46), 'titles', False)
        # Getting the type of 'aligned' (line 668)
        aligned_13522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 54), 'aligned', False)
        # Getting the type of 'byteorder' (line 668)
        byteorder_13523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 63), 'byteorder', False)
        # Processing the call keyword arguments (line 668)
        kwargs_13524 = {}
        # Getting the type of 'format_parser' (line 668)
        format_parser_13518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 16), 'format_parser', False)
        # Calling format_parser(args, kwargs) (line 668)
        format_parser_call_result_13525 = invoke(stypy.reporting.localization.Localization(__file__, 668, 16), format_parser_13518, *[formats_13519, names_13520, titles_13521, aligned_13522, byteorder_13523], **kwargs_13524)
        
        # Obtaining the member '_descr' of a type (line 668)
        _descr_13526 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 668, 16), format_parser_call_result_13525, '_descr')
        # Assigning a type to the variable 'descr' (line 668)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 668, 8), 'descr', _descr_13526)

        if (may_be_13509 and more_types_in_union_13510):
            # SSA join for if statement (line 665)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # SSA begins for try-except statement (line 670)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 671):
    
    # Assigning a Call to a Name (line 671):
    
    # Call to array(...): (line 671)
    # Processing the call arguments (line 671)
    # Getting the type of 'recList' (line 671)
    recList_13529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 671, 26), 'recList', False)
    # Processing the call keyword arguments (line 671)
    # Getting the type of 'descr' (line 671)
    descr_13530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 671, 41), 'descr', False)
    keyword_13531 = descr_13530
    kwargs_13532 = {'dtype': keyword_13531}
    # Getting the type of 'sb' (line 671)
    sb_13527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 671, 17), 'sb', False)
    # Obtaining the member 'array' of a type (line 671)
    array_13528 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 671, 17), sb_13527, 'array')
    # Calling array(args, kwargs) (line 671)
    array_call_result_13533 = invoke(stypy.reporting.localization.Localization(__file__, 671, 17), array_13528, *[recList_13529], **kwargs_13532)
    
    # Assigning a type to the variable 'retval' (line 671)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 671, 8), 'retval', array_call_result_13533)
    # SSA branch for the except part of a try statement (line 670)
    # SSA branch for the except 'TypeError' branch of a try statement (line 670)
    module_type_store.open_ssa_branch('except')
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'shape' (line 673)
    shape_13534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 673, 12), 'shape')
    # Getting the type of 'None' (line 673)
    None_13535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 673, 21), 'None')
    # Applying the binary operator 'is' (line 673)
    result_is__13536 = python_operator(stypy.reporting.localization.Localization(__file__, 673, 12), 'is', shape_13534, None_13535)
    
    
    # Getting the type of 'shape' (line 673)
    shape_13537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 673, 29), 'shape')
    int_13538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 673, 38), 'int')
    # Applying the binary operator '==' (line 673)
    result_eq_13539 = python_operator(stypy.reporting.localization.Localization(__file__, 673, 29), '==', shape_13537, int_13538)
    
    # Applying the binary operator 'or' (line 673)
    result_or_keyword_13540 = python_operator(stypy.reporting.localization.Localization(__file__, 673, 12), 'or', result_is__13536, result_eq_13539)
    
    # Testing the type of an if condition (line 673)
    if_condition_13541 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 673, 8), result_or_keyword_13540)
    # Assigning a type to the variable 'if_condition_13541' (line 673)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 673, 8), 'if_condition_13541', if_condition_13541)
    # SSA begins for if statement (line 673)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 674):
    
    # Assigning a Call to a Name (line 674):
    
    # Call to len(...): (line 674)
    # Processing the call arguments (line 674)
    # Getting the type of 'recList' (line 674)
    recList_13543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 674, 24), 'recList', False)
    # Processing the call keyword arguments (line 674)
    kwargs_13544 = {}
    # Getting the type of 'len' (line 674)
    len_13542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 674, 20), 'len', False)
    # Calling len(args, kwargs) (line 674)
    len_call_result_13545 = invoke(stypy.reporting.localization.Localization(__file__, 674, 20), len_13542, *[recList_13543], **kwargs_13544)
    
    # Assigning a type to the variable 'shape' (line 674)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 674, 12), 'shape', len_call_result_13545)
    # SSA join for if statement (line 673)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to isinstance(...): (line 675)
    # Processing the call arguments (line 675)
    # Getting the type of 'shape' (line 675)
    shape_13547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 22), 'shape', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 675)
    tuple_13548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 675, 30), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 675)
    # Adding element type (line 675)
    # Getting the type of 'int' (line 675)
    int_13549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 30), 'int', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 675, 30), tuple_13548, int_13549)
    # Adding element type (line 675)
    # Getting the type of 'long' (line 675)
    long_13550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 35), 'long', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 675, 30), tuple_13548, long_13550)
    
    # Processing the call keyword arguments (line 675)
    kwargs_13551 = {}
    # Getting the type of 'isinstance' (line 675)
    isinstance_13546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 11), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 675)
    isinstance_call_result_13552 = invoke(stypy.reporting.localization.Localization(__file__, 675, 11), isinstance_13546, *[shape_13547, tuple_13548], **kwargs_13551)
    
    # Testing the type of an if condition (line 675)
    if_condition_13553 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 675, 8), isinstance_call_result_13552)
    # Assigning a type to the variable 'if_condition_13553' (line 675)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 675, 8), 'if_condition_13553', if_condition_13553)
    # SSA begins for if statement (line 675)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Tuple to a Name (line 676):
    
    # Assigning a Tuple to a Name (line 676):
    
    # Obtaining an instance of the builtin type 'tuple' (line 676)
    tuple_13554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 676, 21), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 676)
    # Adding element type (line 676)
    # Getting the type of 'shape' (line 676)
    shape_13555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 676, 21), 'shape')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 676, 21), tuple_13554, shape_13555)
    
    # Assigning a type to the variable 'shape' (line 676)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 676, 12), 'shape', tuple_13554)
    # SSA join for if statement (line 675)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to len(...): (line 677)
    # Processing the call arguments (line 677)
    # Getting the type of 'shape' (line 677)
    shape_13557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 15), 'shape', False)
    # Processing the call keyword arguments (line 677)
    kwargs_13558 = {}
    # Getting the type of 'len' (line 677)
    len_13556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 11), 'len', False)
    # Calling len(args, kwargs) (line 677)
    len_call_result_13559 = invoke(stypy.reporting.localization.Localization(__file__, 677, 11), len_13556, *[shape_13557], **kwargs_13558)
    
    int_13560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 677, 24), 'int')
    # Applying the binary operator '>' (line 677)
    result_gt_13561 = python_operator(stypy.reporting.localization.Localization(__file__, 677, 11), '>', len_call_result_13559, int_13560)
    
    # Testing the type of an if condition (line 677)
    if_condition_13562 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 677, 8), result_gt_13561)
    # Assigning a type to the variable 'if_condition_13562' (line 677)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 677, 8), 'if_condition_13562', if_condition_13562)
    # SSA begins for if statement (line 677)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 678)
    # Processing the call arguments (line 678)
    str_13564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 678, 29), 'str', 'Can only deal with 1-d array.')
    # Processing the call keyword arguments (line 678)
    kwargs_13565 = {}
    # Getting the type of 'ValueError' (line 678)
    ValueError_13563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 678, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 678)
    ValueError_call_result_13566 = invoke(stypy.reporting.localization.Localization(__file__, 678, 18), ValueError_13563, *[str_13564], **kwargs_13565)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 678, 12), ValueError_call_result_13566, 'raise parameter', BaseException)
    # SSA join for if statement (line 677)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 679):
    
    # Assigning a Call to a Name (line 679):
    
    # Call to recarray(...): (line 679)
    # Processing the call arguments (line 679)
    # Getting the type of 'shape' (line 679)
    shape_13568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 26), 'shape', False)
    # Getting the type of 'descr' (line 679)
    descr_13569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 33), 'descr', False)
    # Processing the call keyword arguments (line 679)
    kwargs_13570 = {}
    # Getting the type of 'recarray' (line 679)
    recarray_13567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 17), 'recarray', False)
    # Calling recarray(args, kwargs) (line 679)
    recarray_call_result_13571 = invoke(stypy.reporting.localization.Localization(__file__, 679, 17), recarray_13567, *[shape_13568, descr_13569], **kwargs_13570)
    
    # Assigning a type to the variable '_array' (line 679)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 679, 8), '_array', recarray_call_result_13571)
    
    
    # Call to range(...): (line 680)
    # Processing the call arguments (line 680)
    # Getting the type of '_array' (line 680)
    _array_13573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 680, 23), '_array', False)
    # Obtaining the member 'size' of a type (line 680)
    size_13574 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 680, 23), _array_13573, 'size')
    # Processing the call keyword arguments (line 680)
    kwargs_13575 = {}
    # Getting the type of 'range' (line 680)
    range_13572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 680, 17), 'range', False)
    # Calling range(args, kwargs) (line 680)
    range_call_result_13576 = invoke(stypy.reporting.localization.Localization(__file__, 680, 17), range_13572, *[size_13574], **kwargs_13575)
    
    # Testing the type of a for loop iterable (line 680)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 680, 8), range_call_result_13576)
    # Getting the type of the for loop variable (line 680)
    for_loop_var_13577 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 680, 8), range_call_result_13576)
    # Assigning a type to the variable 'k' (line 680)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 680, 8), 'k', for_loop_var_13577)
    # SSA begins for a for statement (line 680)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Subscript (line 681):
    
    # Assigning a Call to a Subscript (line 681):
    
    # Call to tuple(...): (line 681)
    # Processing the call arguments (line 681)
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 681)
    k_13579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 38), 'k', False)
    # Getting the type of 'recList' (line 681)
    recList_13580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 30), 'recList', False)
    # Obtaining the member '__getitem__' of a type (line 681)
    getitem___13581 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 681, 30), recList_13580, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 681)
    subscript_call_result_13582 = invoke(stypy.reporting.localization.Localization(__file__, 681, 30), getitem___13581, k_13579)
    
    # Processing the call keyword arguments (line 681)
    kwargs_13583 = {}
    # Getting the type of 'tuple' (line 681)
    tuple_13578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 24), 'tuple', False)
    # Calling tuple(args, kwargs) (line 681)
    tuple_call_result_13584 = invoke(stypy.reporting.localization.Localization(__file__, 681, 24), tuple_13578, *[subscript_call_result_13582], **kwargs_13583)
    
    # Getting the type of '_array' (line 681)
    _array_13585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 12), '_array')
    # Getting the type of 'k' (line 681)
    k_13586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 19), 'k')
    # Storing an element on a container (line 681)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 681, 12), _array_13585, (k_13586, tuple_call_result_13584))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of '_array' (line 682)
    _array_13587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 682, 15), '_array')
    # Assigning a type to the variable 'stypy_return_type' (line 682)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 682, 8), 'stypy_return_type', _array_13587)
    # SSA branch for the else branch of a try statement (line 670)
    module_type_store.open_ssa_branch('except else')
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'shape' (line 684)
    shape_13588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 684, 11), 'shape')
    # Getting the type of 'None' (line 684)
    None_13589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 684, 24), 'None')
    # Applying the binary operator 'isnot' (line 684)
    result_is_not_13590 = python_operator(stypy.reporting.localization.Localization(__file__, 684, 11), 'isnot', shape_13588, None_13589)
    
    
    # Getting the type of 'retval' (line 684)
    retval_13591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 684, 33), 'retval')
    # Obtaining the member 'shape' of a type (line 684)
    shape_13592 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 684, 33), retval_13591, 'shape')
    # Getting the type of 'shape' (line 684)
    shape_13593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 684, 49), 'shape')
    # Applying the binary operator '!=' (line 684)
    result_ne_13594 = python_operator(stypy.reporting.localization.Localization(__file__, 684, 33), '!=', shape_13592, shape_13593)
    
    # Applying the binary operator 'and' (line 684)
    result_and_keyword_13595 = python_operator(stypy.reporting.localization.Localization(__file__, 684, 11), 'and', result_is_not_13590, result_ne_13594)
    
    # Testing the type of an if condition (line 684)
    if_condition_13596 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 684, 8), result_and_keyword_13595)
    # Assigning a type to the variable 'if_condition_13596' (line 684)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 684, 8), 'if_condition_13596', if_condition_13596)
    # SSA begins for if statement (line 684)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Attribute (line 685):
    
    # Assigning a Name to a Attribute (line 685):
    # Getting the type of 'shape' (line 685)
    shape_13597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 685, 27), 'shape')
    # Getting the type of 'retval' (line 685)
    retval_13598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 685, 12), 'retval')
    # Setting the type of the member 'shape' of a type (line 685)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 685, 12), retval_13598, 'shape', shape_13597)
    # SSA join for if statement (line 684)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for try-except statement (line 670)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 687):
    
    # Assigning a Call to a Name (line 687):
    
    # Call to view(...): (line 687)
    # Processing the call arguments (line 687)
    # Getting the type of 'recarray' (line 687)
    recarray_13601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 687, 22), 'recarray', False)
    # Processing the call keyword arguments (line 687)
    kwargs_13602 = {}
    # Getting the type of 'retval' (line 687)
    retval_13599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 687, 10), 'retval', False)
    # Obtaining the member 'view' of a type (line 687)
    view_13600 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 687, 10), retval_13599, 'view')
    # Calling view(args, kwargs) (line 687)
    view_call_result_13603 = invoke(stypy.reporting.localization.Localization(__file__, 687, 10), view_13600, *[recarray_13601], **kwargs_13602)
    
    # Assigning a type to the variable 'res' (line 687)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 687, 4), 'res', view_call_result_13603)
    # Getting the type of 'res' (line 689)
    res_13604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 11), 'res')
    # Assigning a type to the variable 'stypy_return_type' (line 689)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 689, 4), 'stypy_return_type', res_13604)
    
    # ################# End of 'fromrecords(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'fromrecords' in the type store
    # Getting the type of 'stypy_return_type' (line 628)
    stypy_return_type_13605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 628, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_13605)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'fromrecords'
    return stypy_return_type_13605

# Assigning a type to the variable 'fromrecords' (line 628)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 628, 0), 'fromrecords', fromrecords)

@norecursion
def fromstring(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 692)
    None_13606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 33), 'None')
    # Getting the type of 'None' (line 692)
    None_13607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 45), 'None')
    int_13608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 692, 58), 'int')
    # Getting the type of 'None' (line 692)
    None_13609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 69), 'None')
    # Getting the type of 'None' (line 693)
    None_13610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 693, 21), 'None')
    # Getting the type of 'None' (line 693)
    None_13611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 693, 34), 'None')
    # Getting the type of 'False' (line 693)
    False_13612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 693, 48), 'False')
    # Getting the type of 'None' (line 693)
    None_13613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 693, 65), 'None')
    defaults = [None_13606, None_13607, int_13608, None_13609, None_13610, None_13611, False_13612, None_13613]
    # Create a new context for function 'fromstring'
    module_type_store = module_type_store.open_function_context('fromstring', 692, 0, False)
    
    # Passed parameters checking function
    fromstring.stypy_localization = localization
    fromstring.stypy_type_of_self = None
    fromstring.stypy_type_store = module_type_store
    fromstring.stypy_function_name = 'fromstring'
    fromstring.stypy_param_names_list = ['datastring', 'dtype', 'shape', 'offset', 'formats', 'names', 'titles', 'aligned', 'byteorder']
    fromstring.stypy_varargs_param_name = None
    fromstring.stypy_kwargs_param_name = None
    fromstring.stypy_call_defaults = defaults
    fromstring.stypy_call_varargs = varargs
    fromstring.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'fromstring', ['datastring', 'dtype', 'shape', 'offset', 'formats', 'names', 'titles', 'aligned', 'byteorder'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'fromstring', localization, ['datastring', 'dtype', 'shape', 'offset', 'formats', 'names', 'titles', 'aligned', 'byteorder'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'fromstring(...)' code ##################

    str_13614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 695, (-1)), 'str', ' create a (read-only) record array from binary data contained in\n    a string')
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'dtype' (line 697)
    dtype_13615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 7), 'dtype')
    # Getting the type of 'None' (line 697)
    None_13616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 16), 'None')
    # Applying the binary operator 'is' (line 697)
    result_is__13617 = python_operator(stypy.reporting.localization.Localization(__file__, 697, 7), 'is', dtype_13615, None_13616)
    
    
    # Getting the type of 'formats' (line 697)
    formats_13618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 25), 'formats')
    # Getting the type of 'None' (line 697)
    None_13619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 36), 'None')
    # Applying the binary operator 'is' (line 697)
    result_is__13620 = python_operator(stypy.reporting.localization.Localization(__file__, 697, 25), 'is', formats_13618, None_13619)
    
    # Applying the binary operator 'and' (line 697)
    result_and_keyword_13621 = python_operator(stypy.reporting.localization.Localization(__file__, 697, 7), 'and', result_is__13617, result_is__13620)
    
    # Testing the type of an if condition (line 697)
    if_condition_13622 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 697, 4), result_and_keyword_13621)
    # Assigning a type to the variable 'if_condition_13622' (line 697)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 697, 4), 'if_condition_13622', if_condition_13622)
    # SSA begins for if statement (line 697)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 698)
    # Processing the call arguments (line 698)
    str_13624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 698, 25), 'str', 'Must have dtype= or formats=')
    # Processing the call keyword arguments (line 698)
    kwargs_13625 = {}
    # Getting the type of 'ValueError' (line 698)
    ValueError_13623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 698)
    ValueError_call_result_13626 = invoke(stypy.reporting.localization.Localization(__file__, 698, 14), ValueError_13623, *[str_13624], **kwargs_13625)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 698, 8), ValueError_call_result_13626, 'raise parameter', BaseException)
    # SSA join for if statement (line 697)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 700)
    # Getting the type of 'dtype' (line 700)
    dtype_13627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 700, 4), 'dtype')
    # Getting the type of 'None' (line 700)
    None_13628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 700, 20), 'None')
    
    (may_be_13629, more_types_in_union_13630) = may_not_be_none(dtype_13627, None_13628)

    if may_be_13629:

        if more_types_in_union_13630:
            # Runtime conditional SSA (line 700)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 701):
        
        # Assigning a Call to a Name (line 701):
        
        # Call to dtype(...): (line 701)
        # Processing the call arguments (line 701)
        # Getting the type of 'dtype' (line 701)
        dtype_13633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 701, 25), 'dtype', False)
        # Processing the call keyword arguments (line 701)
        kwargs_13634 = {}
        # Getting the type of 'sb' (line 701)
        sb_13631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 701, 16), 'sb', False)
        # Obtaining the member 'dtype' of a type (line 701)
        dtype_13632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 701, 16), sb_13631, 'dtype')
        # Calling dtype(args, kwargs) (line 701)
        dtype_call_result_13635 = invoke(stypy.reporting.localization.Localization(__file__, 701, 16), dtype_13632, *[dtype_13633], **kwargs_13634)
        
        # Assigning a type to the variable 'descr' (line 701)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 701, 8), 'descr', dtype_call_result_13635)

        if more_types_in_union_13630:
            # Runtime conditional SSA for else branch (line 700)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_13629) or more_types_in_union_13630):
        
        # Assigning a Attribute to a Name (line 703):
        
        # Assigning a Attribute to a Name (line 703):
        
        # Call to format_parser(...): (line 703)
        # Processing the call arguments (line 703)
        # Getting the type of 'formats' (line 703)
        formats_13637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 703, 30), 'formats', False)
        # Getting the type of 'names' (line 703)
        names_13638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 703, 39), 'names', False)
        # Getting the type of 'titles' (line 703)
        titles_13639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 703, 46), 'titles', False)
        # Getting the type of 'aligned' (line 703)
        aligned_13640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 703, 54), 'aligned', False)
        # Getting the type of 'byteorder' (line 703)
        byteorder_13641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 703, 63), 'byteorder', False)
        # Processing the call keyword arguments (line 703)
        kwargs_13642 = {}
        # Getting the type of 'format_parser' (line 703)
        format_parser_13636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 703, 16), 'format_parser', False)
        # Calling format_parser(args, kwargs) (line 703)
        format_parser_call_result_13643 = invoke(stypy.reporting.localization.Localization(__file__, 703, 16), format_parser_13636, *[formats_13637, names_13638, titles_13639, aligned_13640, byteorder_13641], **kwargs_13642)
        
        # Obtaining the member '_descr' of a type (line 703)
        _descr_13644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 703, 16), format_parser_call_result_13643, '_descr')
        # Assigning a type to the variable 'descr' (line 703)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 703, 8), 'descr', _descr_13644)

        if (may_be_13629 and more_types_in_union_13630):
            # SSA join for if statement (line 700)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Attribute to a Name (line 705):
    
    # Assigning a Attribute to a Name (line 705):
    # Getting the type of 'descr' (line 705)
    descr_13645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 15), 'descr')
    # Obtaining the member 'itemsize' of a type (line 705)
    itemsize_13646 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 705, 15), descr_13645, 'itemsize')
    # Assigning a type to the variable 'itemsize' (line 705)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 705, 4), 'itemsize', itemsize_13646)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'shape' (line 706)
    shape_13647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 706, 8), 'shape')
    # Getting the type of 'None' (line 706)
    None_13648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 706, 17), 'None')
    # Applying the binary operator 'is' (line 706)
    result_is__13649 = python_operator(stypy.reporting.localization.Localization(__file__, 706, 8), 'is', shape_13647, None_13648)
    
    
    # Getting the type of 'shape' (line 706)
    shape_13650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 706, 25), 'shape')
    int_13651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 706, 34), 'int')
    # Applying the binary operator '==' (line 706)
    result_eq_13652 = python_operator(stypy.reporting.localization.Localization(__file__, 706, 25), '==', shape_13650, int_13651)
    
    # Applying the binary operator 'or' (line 706)
    result_or_keyword_13653 = python_operator(stypy.reporting.localization.Localization(__file__, 706, 8), 'or', result_is__13649, result_eq_13652)
    
    # Getting the type of 'shape' (line 706)
    shape_13654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 706, 39), 'shape')
    int_13655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 706, 48), 'int')
    # Applying the binary operator '==' (line 706)
    result_eq_13656 = python_operator(stypy.reporting.localization.Localization(__file__, 706, 39), '==', shape_13654, int_13655)
    
    # Applying the binary operator 'or' (line 706)
    result_or_keyword_13657 = python_operator(stypy.reporting.localization.Localization(__file__, 706, 8), 'or', result_or_keyword_13653, result_eq_13656)
    
    # Testing the type of an if condition (line 706)
    if_condition_13658 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 706, 4), result_or_keyword_13657)
    # Assigning a type to the variable 'if_condition_13658' (line 706)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 706, 4), 'if_condition_13658', if_condition_13658)
    # SSA begins for if statement (line 706)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 707):
    
    # Assigning a BinOp to a Name (line 707):
    
    # Call to len(...): (line 707)
    # Processing the call arguments (line 707)
    # Getting the type of 'datastring' (line 707)
    datastring_13660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 707, 21), 'datastring', False)
    # Processing the call keyword arguments (line 707)
    kwargs_13661 = {}
    # Getting the type of 'len' (line 707)
    len_13659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 707, 17), 'len', False)
    # Calling len(args, kwargs) (line 707)
    len_call_result_13662 = invoke(stypy.reporting.localization.Localization(__file__, 707, 17), len_13659, *[datastring_13660], **kwargs_13661)
    
    # Getting the type of 'offset' (line 707)
    offset_13663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 707, 35), 'offset')
    # Applying the binary operator '-' (line 707)
    result_sub_13664 = python_operator(stypy.reporting.localization.Localization(__file__, 707, 17), '-', len_call_result_13662, offset_13663)
    
    # Getting the type of 'itemsize' (line 707)
    itemsize_13665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 707, 45), 'itemsize')
    # Applying the binary operator 'div' (line 707)
    result_div_13666 = python_operator(stypy.reporting.localization.Localization(__file__, 707, 16), 'div', result_sub_13664, itemsize_13665)
    
    # Assigning a type to the variable 'shape' (line 707)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 707, 8), 'shape', result_div_13666)
    # SSA join for if statement (line 706)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 709):
    
    # Assigning a Call to a Name (line 709):
    
    # Call to recarray(...): (line 709)
    # Processing the call arguments (line 709)
    # Getting the type of 'shape' (line 709)
    shape_13668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 709, 22), 'shape', False)
    # Getting the type of 'descr' (line 709)
    descr_13669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 709, 29), 'descr', False)
    # Processing the call keyword arguments (line 709)
    # Getting the type of 'datastring' (line 709)
    datastring_13670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 709, 40), 'datastring', False)
    keyword_13671 = datastring_13670
    # Getting the type of 'offset' (line 709)
    offset_13672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 709, 59), 'offset', False)
    keyword_13673 = offset_13672
    kwargs_13674 = {'buf': keyword_13671, 'offset': keyword_13673}
    # Getting the type of 'recarray' (line 709)
    recarray_13667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 709, 13), 'recarray', False)
    # Calling recarray(args, kwargs) (line 709)
    recarray_call_result_13675 = invoke(stypy.reporting.localization.Localization(__file__, 709, 13), recarray_13667, *[shape_13668, descr_13669], **kwargs_13674)
    
    # Assigning a type to the variable '_array' (line 709)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 709, 4), '_array', recarray_call_result_13675)
    # Getting the type of '_array' (line 710)
    _array_13676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 710, 11), '_array')
    # Assigning a type to the variable 'stypy_return_type' (line 710)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 710, 4), 'stypy_return_type', _array_13676)
    
    # ################# End of 'fromstring(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'fromstring' in the type store
    # Getting the type of 'stypy_return_type' (line 692)
    stypy_return_type_13677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_13677)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'fromstring'
    return stypy_return_type_13677

# Assigning a type to the variable 'fromstring' (line 692)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 692, 0), 'fromstring', fromstring)

@norecursion
def get_remaining_size(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'get_remaining_size'
    module_type_store = module_type_store.open_function_context('get_remaining_size', 712, 0, False)
    
    # Passed parameters checking function
    get_remaining_size.stypy_localization = localization
    get_remaining_size.stypy_type_of_self = None
    get_remaining_size.stypy_type_store = module_type_store
    get_remaining_size.stypy_function_name = 'get_remaining_size'
    get_remaining_size.stypy_param_names_list = ['fd']
    get_remaining_size.stypy_varargs_param_name = None
    get_remaining_size.stypy_kwargs_param_name = None
    get_remaining_size.stypy_call_defaults = defaults
    get_remaining_size.stypy_call_varargs = varargs
    get_remaining_size.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'get_remaining_size', ['fd'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'get_remaining_size', localization, ['fd'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'get_remaining_size(...)' code ##################

    
    
    # SSA begins for try-except statement (line 713)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 714):
    
    # Assigning a Call to a Name (line 714):
    
    # Call to fileno(...): (line 714)
    # Processing the call keyword arguments (line 714)
    kwargs_13680 = {}
    # Getting the type of 'fd' (line 714)
    fd_13678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 714, 13), 'fd', False)
    # Obtaining the member 'fileno' of a type (line 714)
    fileno_13679 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 714, 13), fd_13678, 'fileno')
    # Calling fileno(args, kwargs) (line 714)
    fileno_call_result_13681 = invoke(stypy.reporting.localization.Localization(__file__, 714, 13), fileno_13679, *[], **kwargs_13680)
    
    # Assigning a type to the variable 'fn' (line 714)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 714, 8), 'fn', fileno_call_result_13681)
    # SSA branch for the except part of a try statement (line 713)
    # SSA branch for the except 'AttributeError' branch of a try statement (line 713)
    module_type_store.open_ssa_branch('except')
    
    # Call to getsize(...): (line 716)
    # Processing the call arguments (line 716)
    # Getting the type of 'fd' (line 716)
    fd_13685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 716, 31), 'fd', False)
    # Obtaining the member 'name' of a type (line 716)
    name_13686 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 716, 31), fd_13685, 'name')
    # Processing the call keyword arguments (line 716)
    kwargs_13687 = {}
    # Getting the type of 'os' (line 716)
    os_13682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 716, 15), 'os', False)
    # Obtaining the member 'path' of a type (line 716)
    path_13683 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 716, 15), os_13682, 'path')
    # Obtaining the member 'getsize' of a type (line 716)
    getsize_13684 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 716, 15), path_13683, 'getsize')
    # Calling getsize(args, kwargs) (line 716)
    getsize_call_result_13688 = invoke(stypy.reporting.localization.Localization(__file__, 716, 15), getsize_13684, *[name_13686], **kwargs_13687)
    
    
    # Call to tell(...): (line 716)
    # Processing the call keyword arguments (line 716)
    kwargs_13691 = {}
    # Getting the type of 'fd' (line 716)
    fd_13689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 716, 42), 'fd', False)
    # Obtaining the member 'tell' of a type (line 716)
    tell_13690 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 716, 42), fd_13689, 'tell')
    # Calling tell(args, kwargs) (line 716)
    tell_call_result_13692 = invoke(stypy.reporting.localization.Localization(__file__, 716, 42), tell_13690, *[], **kwargs_13691)
    
    # Applying the binary operator '-' (line 716)
    result_sub_13693 = python_operator(stypy.reporting.localization.Localization(__file__, 716, 15), '-', getsize_call_result_13688, tell_call_result_13692)
    
    # Assigning a type to the variable 'stypy_return_type' (line 716)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 716, 8), 'stypy_return_type', result_sub_13693)
    # SSA join for try-except statement (line 713)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 717):
    
    # Assigning a Call to a Name (line 717):
    
    # Call to fstat(...): (line 717)
    # Processing the call arguments (line 717)
    # Getting the type of 'fn' (line 717)
    fn_13696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 717, 18), 'fn', False)
    # Processing the call keyword arguments (line 717)
    kwargs_13697 = {}
    # Getting the type of 'os' (line 717)
    os_13694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 717, 9), 'os', False)
    # Obtaining the member 'fstat' of a type (line 717)
    fstat_13695 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 717, 9), os_13694, 'fstat')
    # Calling fstat(args, kwargs) (line 717)
    fstat_call_result_13698 = invoke(stypy.reporting.localization.Localization(__file__, 717, 9), fstat_13695, *[fn_13696], **kwargs_13697)
    
    # Assigning a type to the variable 'st' (line 717)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 717, 4), 'st', fstat_call_result_13698)
    
    # Assigning a BinOp to a Name (line 718):
    
    # Assigning a BinOp to a Name (line 718):
    # Getting the type of 'st' (line 718)
    st_13699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 11), 'st')
    # Obtaining the member 'st_size' of a type (line 718)
    st_size_13700 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 718, 11), st_13699, 'st_size')
    
    # Call to tell(...): (line 718)
    # Processing the call keyword arguments (line 718)
    kwargs_13703 = {}
    # Getting the type of 'fd' (line 718)
    fd_13701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 24), 'fd', False)
    # Obtaining the member 'tell' of a type (line 718)
    tell_13702 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 718, 24), fd_13701, 'tell')
    # Calling tell(args, kwargs) (line 718)
    tell_call_result_13704 = invoke(stypy.reporting.localization.Localization(__file__, 718, 24), tell_13702, *[], **kwargs_13703)
    
    # Applying the binary operator '-' (line 718)
    result_sub_13705 = python_operator(stypy.reporting.localization.Localization(__file__, 718, 11), '-', st_size_13700, tell_call_result_13704)
    
    # Assigning a type to the variable 'size' (line 718)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 718, 4), 'size', result_sub_13705)
    # Getting the type of 'size' (line 719)
    size_13706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 719, 11), 'size')
    # Assigning a type to the variable 'stypy_return_type' (line 719)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 719, 4), 'stypy_return_type', size_13706)
    
    # ################# End of 'get_remaining_size(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_remaining_size' in the type store
    # Getting the type of 'stypy_return_type' (line 712)
    stypy_return_type_13707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 712, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_13707)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_remaining_size'
    return stypy_return_type_13707

# Assigning a type to the variable 'get_remaining_size' (line 712)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 712, 0), 'get_remaining_size', get_remaining_size)

@norecursion
def fromfile(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 721)
    None_13708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 721, 23), 'None')
    # Getting the type of 'None' (line 721)
    None_13709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 721, 35), 'None')
    int_13710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 721, 48), 'int')
    # Getting the type of 'None' (line 721)
    None_13711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 721, 59), 'None')
    # Getting the type of 'None' (line 722)
    None_13712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 722, 19), 'None')
    # Getting the type of 'None' (line 722)
    None_13713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 722, 32), 'None')
    # Getting the type of 'False' (line 722)
    False_13714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 722, 46), 'False')
    # Getting the type of 'None' (line 722)
    None_13715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 722, 63), 'None')
    defaults = [None_13708, None_13709, int_13710, None_13711, None_13712, None_13713, False_13714, None_13715]
    # Create a new context for function 'fromfile'
    module_type_store = module_type_store.open_function_context('fromfile', 721, 0, False)
    
    # Passed parameters checking function
    fromfile.stypy_localization = localization
    fromfile.stypy_type_of_self = None
    fromfile.stypy_type_store = module_type_store
    fromfile.stypy_function_name = 'fromfile'
    fromfile.stypy_param_names_list = ['fd', 'dtype', 'shape', 'offset', 'formats', 'names', 'titles', 'aligned', 'byteorder']
    fromfile.stypy_varargs_param_name = None
    fromfile.stypy_kwargs_param_name = None
    fromfile.stypy_call_defaults = defaults
    fromfile.stypy_call_varargs = varargs
    fromfile.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'fromfile', ['fd', 'dtype', 'shape', 'offset', 'formats', 'names', 'titles', 'aligned', 'byteorder'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'fromfile', localization, ['fd', 'dtype', 'shape', 'offset', 'formats', 'names', 'titles', 'aligned', 'byteorder'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'fromfile(...)' code ##################

    str_13716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 743, (-1)), 'str', "Create an array from binary file data\n\n    If file is a string then that file is opened, else it is assumed\n    to be a file object.\n\n    >>> from tempfile import TemporaryFile\n    >>> a = np.empty(10,dtype='f8,i4,a5')\n    >>> a[5] = (0.5,10,'abcde')\n    >>>\n    >>> fd=TemporaryFile()\n    >>> a = a.newbyteorder('<')\n    >>> a.tofile(fd)\n    >>>\n    >>> fd.seek(0)\n    >>> r=np.core.records.fromfile(fd, formats='f8,i4,a5', shape=10,\n    ... byteorder='<')\n    >>> print(r[5])\n    (0.5, 10, 'abcde')\n    >>> r.shape\n    (10,)\n    ")
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'shape' (line 745)
    shape_13717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 745, 8), 'shape')
    # Getting the type of 'None' (line 745)
    None_13718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 745, 17), 'None')
    # Applying the binary operator 'is' (line 745)
    result_is__13719 = python_operator(stypy.reporting.localization.Localization(__file__, 745, 8), 'is', shape_13717, None_13718)
    
    
    # Getting the type of 'shape' (line 745)
    shape_13720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 745, 25), 'shape')
    int_13721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 745, 34), 'int')
    # Applying the binary operator '==' (line 745)
    result_eq_13722 = python_operator(stypy.reporting.localization.Localization(__file__, 745, 25), '==', shape_13720, int_13721)
    
    # Applying the binary operator 'or' (line 745)
    result_or_keyword_13723 = python_operator(stypy.reporting.localization.Localization(__file__, 745, 8), 'or', result_is__13719, result_eq_13722)
    
    # Testing the type of an if condition (line 745)
    if_condition_13724 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 745, 4), result_or_keyword_13723)
    # Assigning a type to the variable 'if_condition_13724' (line 745)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 745, 4), 'if_condition_13724', if_condition_13724)
    # SSA begins for if statement (line 745)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Tuple to a Name (line 746):
    
    # Assigning a Tuple to a Name (line 746):
    
    # Obtaining an instance of the builtin type 'tuple' (line 746)
    tuple_13725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 746, 17), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 746)
    # Adding element type (line 746)
    int_13726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 746, 17), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 746, 17), tuple_13725, int_13726)
    
    # Assigning a type to the variable 'shape' (line 746)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 746, 8), 'shape', tuple_13725)
    # SSA branch for the else part of an if statement (line 745)
    module_type_store.open_ssa_branch('else')
    
    
    # Call to isinstance(...): (line 747)
    # Processing the call arguments (line 747)
    # Getting the type of 'shape' (line 747)
    shape_13728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 747, 20), 'shape', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 747)
    tuple_13729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 747, 28), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 747)
    # Adding element type (line 747)
    # Getting the type of 'int' (line 747)
    int_13730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 747, 28), 'int', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 747, 28), tuple_13729, int_13730)
    # Adding element type (line 747)
    # Getting the type of 'long' (line 747)
    long_13731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 747, 33), 'long', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 747, 28), tuple_13729, long_13731)
    
    # Processing the call keyword arguments (line 747)
    kwargs_13732 = {}
    # Getting the type of 'isinstance' (line 747)
    isinstance_13727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 747, 9), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 747)
    isinstance_call_result_13733 = invoke(stypy.reporting.localization.Localization(__file__, 747, 9), isinstance_13727, *[shape_13728, tuple_13729], **kwargs_13732)
    
    # Testing the type of an if condition (line 747)
    if_condition_13734 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 747, 9), isinstance_call_result_13733)
    # Assigning a type to the variable 'if_condition_13734' (line 747)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 747, 9), 'if_condition_13734', if_condition_13734)
    # SSA begins for if statement (line 747)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Tuple to a Name (line 748):
    
    # Assigning a Tuple to a Name (line 748):
    
    # Obtaining an instance of the builtin type 'tuple' (line 748)
    tuple_13735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 748, 17), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 748)
    # Adding element type (line 748)
    # Getting the type of 'shape' (line 748)
    shape_13736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 748, 17), 'shape')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 748, 17), tuple_13735, shape_13736)
    
    # Assigning a type to the variable 'shape' (line 748)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 748, 8), 'shape', tuple_13735)
    # SSA join for if statement (line 747)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 745)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Num to a Name (line 750):
    
    # Assigning a Num to a Name (line 750):
    int_13737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 750, 11), 'int')
    # Assigning a type to the variable 'name' (line 750)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 750, 4), 'name', int_13737)
    
    # Type idiom detected: calculating its left and rigth part (line 751)
    # Getting the type of 'str' (line 751)
    str_13738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 751, 22), 'str')
    # Getting the type of 'fd' (line 751)
    fd_13739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 751, 18), 'fd')
    
    (may_be_13740, more_types_in_union_13741) = may_be_subtype(str_13738, fd_13739)

    if may_be_13740:

        if more_types_in_union_13741:
            # Runtime conditional SSA (line 751)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'fd' (line 751)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 751, 4), 'fd', remove_not_subtype_from_union(fd_13739, str))
        
        # Assigning a Num to a Name (line 752):
        
        # Assigning a Num to a Name (line 752):
        int_13742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 752, 15), 'int')
        # Assigning a type to the variable 'name' (line 752)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 752, 8), 'name', int_13742)
        
        # Assigning a Call to a Name (line 753):
        
        # Assigning a Call to a Name (line 753):
        
        # Call to open(...): (line 753)
        # Processing the call arguments (line 753)
        # Getting the type of 'fd' (line 753)
        fd_13744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 753, 18), 'fd', False)
        str_13745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 753, 22), 'str', 'rb')
        # Processing the call keyword arguments (line 753)
        kwargs_13746 = {}
        # Getting the type of 'open' (line 753)
        open_13743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 753, 13), 'open', False)
        # Calling open(args, kwargs) (line 753)
        open_call_result_13747 = invoke(stypy.reporting.localization.Localization(__file__, 753, 13), open_13743, *[fd_13744, str_13745], **kwargs_13746)
        
        # Assigning a type to the variable 'fd' (line 753)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 753, 8), 'fd', open_call_result_13747)

        if more_types_in_union_13741:
            # SSA join for if statement (line 751)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Getting the type of 'offset' (line 754)
    offset_13748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 754, 8), 'offset')
    int_13749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 754, 17), 'int')
    # Applying the binary operator '>' (line 754)
    result_gt_13750 = python_operator(stypy.reporting.localization.Localization(__file__, 754, 8), '>', offset_13748, int_13749)
    
    # Testing the type of an if condition (line 754)
    if_condition_13751 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 754, 4), result_gt_13750)
    # Assigning a type to the variable 'if_condition_13751' (line 754)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 754, 4), 'if_condition_13751', if_condition_13751)
    # SSA begins for if statement (line 754)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to seek(...): (line 755)
    # Processing the call arguments (line 755)
    # Getting the type of 'offset' (line 755)
    offset_13754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 755, 16), 'offset', False)
    int_13755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 755, 24), 'int')
    # Processing the call keyword arguments (line 755)
    kwargs_13756 = {}
    # Getting the type of 'fd' (line 755)
    fd_13752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 755, 8), 'fd', False)
    # Obtaining the member 'seek' of a type (line 755)
    seek_13753 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 755, 8), fd_13752, 'seek')
    # Calling seek(args, kwargs) (line 755)
    seek_call_result_13757 = invoke(stypy.reporting.localization.Localization(__file__, 755, 8), seek_13753, *[offset_13754, int_13755], **kwargs_13756)
    
    # SSA join for if statement (line 754)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 756):
    
    # Assigning a Call to a Name (line 756):
    
    # Call to get_remaining_size(...): (line 756)
    # Processing the call arguments (line 756)
    # Getting the type of 'fd' (line 756)
    fd_13759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 756, 30), 'fd', False)
    # Processing the call keyword arguments (line 756)
    kwargs_13760 = {}
    # Getting the type of 'get_remaining_size' (line 756)
    get_remaining_size_13758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 756, 11), 'get_remaining_size', False)
    # Calling get_remaining_size(args, kwargs) (line 756)
    get_remaining_size_call_result_13761 = invoke(stypy.reporting.localization.Localization(__file__, 756, 11), get_remaining_size_13758, *[fd_13759], **kwargs_13760)
    
    # Assigning a type to the variable 'size' (line 756)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 756, 4), 'size', get_remaining_size_call_result_13761)
    
    # Type idiom detected: calculating its left and rigth part (line 758)
    # Getting the type of 'dtype' (line 758)
    dtype_13762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 758, 4), 'dtype')
    # Getting the type of 'None' (line 758)
    None_13763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 758, 20), 'None')
    
    (may_be_13764, more_types_in_union_13765) = may_not_be_none(dtype_13762, None_13763)

    if may_be_13764:

        if more_types_in_union_13765:
            # Runtime conditional SSA (line 758)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 759):
        
        # Assigning a Call to a Name (line 759):
        
        # Call to dtype(...): (line 759)
        # Processing the call arguments (line 759)
        # Getting the type of 'dtype' (line 759)
        dtype_13768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 759, 25), 'dtype', False)
        # Processing the call keyword arguments (line 759)
        kwargs_13769 = {}
        # Getting the type of 'sb' (line 759)
        sb_13766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 759, 16), 'sb', False)
        # Obtaining the member 'dtype' of a type (line 759)
        dtype_13767 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 759, 16), sb_13766, 'dtype')
        # Calling dtype(args, kwargs) (line 759)
        dtype_call_result_13770 = invoke(stypy.reporting.localization.Localization(__file__, 759, 16), dtype_13767, *[dtype_13768], **kwargs_13769)
        
        # Assigning a type to the variable 'descr' (line 759)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 759, 8), 'descr', dtype_call_result_13770)

        if more_types_in_union_13765:
            # Runtime conditional SSA for else branch (line 758)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_13764) or more_types_in_union_13765):
        
        # Assigning a Attribute to a Name (line 761):
        
        # Assigning a Attribute to a Name (line 761):
        
        # Call to format_parser(...): (line 761)
        # Processing the call arguments (line 761)
        # Getting the type of 'formats' (line 761)
        formats_13772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 761, 30), 'formats', False)
        # Getting the type of 'names' (line 761)
        names_13773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 761, 39), 'names', False)
        # Getting the type of 'titles' (line 761)
        titles_13774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 761, 46), 'titles', False)
        # Getting the type of 'aligned' (line 761)
        aligned_13775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 761, 54), 'aligned', False)
        # Getting the type of 'byteorder' (line 761)
        byteorder_13776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 761, 63), 'byteorder', False)
        # Processing the call keyword arguments (line 761)
        kwargs_13777 = {}
        # Getting the type of 'format_parser' (line 761)
        format_parser_13771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 761, 16), 'format_parser', False)
        # Calling format_parser(args, kwargs) (line 761)
        format_parser_call_result_13778 = invoke(stypy.reporting.localization.Localization(__file__, 761, 16), format_parser_13771, *[formats_13772, names_13773, titles_13774, aligned_13775, byteorder_13776], **kwargs_13777)
        
        # Obtaining the member '_descr' of a type (line 761)
        _descr_13779 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 761, 16), format_parser_call_result_13778, '_descr')
        # Assigning a type to the variable 'descr' (line 761)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 761, 8), 'descr', _descr_13779)

        if (may_be_13764 and more_types_in_union_13765):
            # SSA join for if statement (line 758)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Attribute to a Name (line 763):
    
    # Assigning a Attribute to a Name (line 763):
    # Getting the type of 'descr' (line 763)
    descr_13780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 763, 15), 'descr')
    # Obtaining the member 'itemsize' of a type (line 763)
    itemsize_13781 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 763, 15), descr_13780, 'itemsize')
    # Assigning a type to the variable 'itemsize' (line 763)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 763, 4), 'itemsize', itemsize_13781)
    
    # Assigning a Call to a Name (line 765):
    
    # Assigning a Call to a Name (line 765):
    
    # Call to prod(...): (line 765)
    # Processing the call keyword arguments (line 765)
    kwargs_13788 = {}
    
    # Call to array(...): (line 765)
    # Processing the call arguments (line 765)
    # Getting the type of 'shape' (line 765)
    shape_13784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 765, 25), 'shape', False)
    # Processing the call keyword arguments (line 765)
    kwargs_13785 = {}
    # Getting the type of 'sb' (line 765)
    sb_13782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 765, 16), 'sb', False)
    # Obtaining the member 'array' of a type (line 765)
    array_13783 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 765, 16), sb_13782, 'array')
    # Calling array(args, kwargs) (line 765)
    array_call_result_13786 = invoke(stypy.reporting.localization.Localization(__file__, 765, 16), array_13783, *[shape_13784], **kwargs_13785)
    
    # Obtaining the member 'prod' of a type (line 765)
    prod_13787 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 765, 16), array_call_result_13786, 'prod')
    # Calling prod(args, kwargs) (line 765)
    prod_call_result_13789 = invoke(stypy.reporting.localization.Localization(__file__, 765, 16), prod_13787, *[], **kwargs_13788)
    
    # Assigning a type to the variable 'shapeprod' (line 765)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 765, 4), 'shapeprod', prod_call_result_13789)
    
    # Assigning a BinOp to a Name (line 766):
    
    # Assigning a BinOp to a Name (line 766):
    # Getting the type of 'shapeprod' (line 766)
    shapeprod_13790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 766, 16), 'shapeprod')
    # Getting the type of 'itemsize' (line 766)
    itemsize_13791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 766, 28), 'itemsize')
    # Applying the binary operator '*' (line 766)
    result_mul_13792 = python_operator(stypy.reporting.localization.Localization(__file__, 766, 16), '*', shapeprod_13790, itemsize_13791)
    
    # Assigning a type to the variable 'shapesize' (line 766)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 766, 4), 'shapesize', result_mul_13792)
    
    
    # Getting the type of 'shapesize' (line 767)
    shapesize_13793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 767, 7), 'shapesize')
    int_13794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 767, 19), 'int')
    # Applying the binary operator '<' (line 767)
    result_lt_13795 = python_operator(stypy.reporting.localization.Localization(__file__, 767, 7), '<', shapesize_13793, int_13794)
    
    # Testing the type of an if condition (line 767)
    if_condition_13796 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 767, 4), result_lt_13795)
    # Assigning a type to the variable 'if_condition_13796' (line 767)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 767, 4), 'if_condition_13796', if_condition_13796)
    # SSA begins for if statement (line 767)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 768):
    
    # Assigning a Call to a Name (line 768):
    
    # Call to list(...): (line 768)
    # Processing the call arguments (line 768)
    # Getting the type of 'shape' (line 768)
    shape_13798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 768, 21), 'shape', False)
    # Processing the call keyword arguments (line 768)
    kwargs_13799 = {}
    # Getting the type of 'list' (line 768)
    list_13797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 768, 16), 'list', False)
    # Calling list(args, kwargs) (line 768)
    list_call_result_13800 = invoke(stypy.reporting.localization.Localization(__file__, 768, 16), list_13797, *[shape_13798], **kwargs_13799)
    
    # Assigning a type to the variable 'shape' (line 768)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 768, 8), 'shape', list_call_result_13800)
    
    # Assigning a BinOp to a Subscript (line 769):
    
    # Assigning a BinOp to a Subscript (line 769):
    # Getting the type of 'size' (line 769)
    size_13801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 769, 33), 'size')
    
    # Getting the type of 'shapesize' (line 769)
    shapesize_13802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 769, 41), 'shapesize')
    # Applying the 'usub' unary operator (line 769)
    result___neg___13803 = python_operator(stypy.reporting.localization.Localization(__file__, 769, 40), 'usub', shapesize_13802)
    
    # Applying the binary operator 'div' (line 769)
    result_div_13804 = python_operator(stypy.reporting.localization.Localization(__file__, 769, 33), 'div', size_13801, result___neg___13803)
    
    # Getting the type of 'shape' (line 769)
    shape_13805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 769, 8), 'shape')
    
    # Call to index(...): (line 769)
    # Processing the call arguments (line 769)
    int_13808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 769, 26), 'int')
    # Processing the call keyword arguments (line 769)
    kwargs_13809 = {}
    # Getting the type of 'shape' (line 769)
    shape_13806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 769, 14), 'shape', False)
    # Obtaining the member 'index' of a type (line 769)
    index_13807 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 769, 14), shape_13806, 'index')
    # Calling index(args, kwargs) (line 769)
    index_call_result_13810 = invoke(stypy.reporting.localization.Localization(__file__, 769, 14), index_13807, *[int_13808], **kwargs_13809)
    
    # Storing an element on a container (line 769)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 769, 8), shape_13805, (index_call_result_13810, result_div_13804))
    
    # Assigning a Call to a Name (line 770):
    
    # Assigning a Call to a Name (line 770):
    
    # Call to tuple(...): (line 770)
    # Processing the call arguments (line 770)
    # Getting the type of 'shape' (line 770)
    shape_13812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 770, 22), 'shape', False)
    # Processing the call keyword arguments (line 770)
    kwargs_13813 = {}
    # Getting the type of 'tuple' (line 770)
    tuple_13811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 770, 16), 'tuple', False)
    # Calling tuple(args, kwargs) (line 770)
    tuple_call_result_13814 = invoke(stypy.reporting.localization.Localization(__file__, 770, 16), tuple_13811, *[shape_13812], **kwargs_13813)
    
    # Assigning a type to the variable 'shape' (line 770)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 770, 8), 'shape', tuple_call_result_13814)
    
    # Assigning a Call to a Name (line 771):
    
    # Assigning a Call to a Name (line 771):
    
    # Call to prod(...): (line 771)
    # Processing the call keyword arguments (line 771)
    kwargs_13821 = {}
    
    # Call to array(...): (line 771)
    # Processing the call arguments (line 771)
    # Getting the type of 'shape' (line 771)
    shape_13817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 771, 29), 'shape', False)
    # Processing the call keyword arguments (line 771)
    kwargs_13818 = {}
    # Getting the type of 'sb' (line 771)
    sb_13815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 771, 20), 'sb', False)
    # Obtaining the member 'array' of a type (line 771)
    array_13816 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 771, 20), sb_13815, 'array')
    # Calling array(args, kwargs) (line 771)
    array_call_result_13819 = invoke(stypy.reporting.localization.Localization(__file__, 771, 20), array_13816, *[shape_13817], **kwargs_13818)
    
    # Obtaining the member 'prod' of a type (line 771)
    prod_13820 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 771, 20), array_call_result_13819, 'prod')
    # Calling prod(args, kwargs) (line 771)
    prod_call_result_13822 = invoke(stypy.reporting.localization.Localization(__file__, 771, 20), prod_13820, *[], **kwargs_13821)
    
    # Assigning a type to the variable 'shapeprod' (line 771)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 771, 8), 'shapeprod', prod_call_result_13822)
    # SSA join for if statement (line 767)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 773):
    
    # Assigning a BinOp to a Name (line 773):
    # Getting the type of 'shapeprod' (line 773)
    shapeprod_13823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 773, 13), 'shapeprod')
    # Getting the type of 'itemsize' (line 773)
    itemsize_13824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 773, 25), 'itemsize')
    # Applying the binary operator '*' (line 773)
    result_mul_13825 = python_operator(stypy.reporting.localization.Localization(__file__, 773, 13), '*', shapeprod_13823, itemsize_13824)
    
    # Assigning a type to the variable 'nbytes' (line 773)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 773, 4), 'nbytes', result_mul_13825)
    
    
    # Getting the type of 'nbytes' (line 775)
    nbytes_13826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 775, 7), 'nbytes')
    # Getting the type of 'size' (line 775)
    size_13827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 775, 16), 'size')
    # Applying the binary operator '>' (line 775)
    result_gt_13828 = python_operator(stypy.reporting.localization.Localization(__file__, 775, 7), '>', nbytes_13826, size_13827)
    
    # Testing the type of an if condition (line 775)
    if_condition_13829 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 775, 4), result_gt_13828)
    # Assigning a type to the variable 'if_condition_13829' (line 775)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 775, 4), 'if_condition_13829', if_condition_13829)
    # SSA begins for if statement (line 775)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 776)
    # Processing the call arguments (line 776)
    str_13831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 777, 16), 'str', 'Not enough bytes left in file for specified shape and type')
    # Processing the call keyword arguments (line 776)
    kwargs_13832 = {}
    # Getting the type of 'ValueError' (line 776)
    ValueError_13830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 776, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 776)
    ValueError_call_result_13833 = invoke(stypy.reporting.localization.Localization(__file__, 776, 14), ValueError_13830, *[str_13831], **kwargs_13832)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 776, 8), ValueError_call_result_13833, 'raise parameter', BaseException)
    # SSA join for if statement (line 775)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 780):
    
    # Assigning a Call to a Name (line 780):
    
    # Call to recarray(...): (line 780)
    # Processing the call arguments (line 780)
    # Getting the type of 'shape' (line 780)
    shape_13835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 780, 22), 'shape', False)
    # Getting the type of 'descr' (line 780)
    descr_13836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 780, 29), 'descr', False)
    # Processing the call keyword arguments (line 780)
    kwargs_13837 = {}
    # Getting the type of 'recarray' (line 780)
    recarray_13834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 780, 13), 'recarray', False)
    # Calling recarray(args, kwargs) (line 780)
    recarray_call_result_13838 = invoke(stypy.reporting.localization.Localization(__file__, 780, 13), recarray_13834, *[shape_13835, descr_13836], **kwargs_13837)
    
    # Assigning a type to the variable '_array' (line 780)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 780, 4), '_array', recarray_call_result_13838)
    
    # Assigning a Call to a Name (line 781):
    
    # Assigning a Call to a Name (line 781):
    
    # Call to readinto(...): (line 781)
    # Processing the call arguments (line 781)
    # Getting the type of '_array' (line 781)
    _array_13841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 781, 29), '_array', False)
    # Obtaining the member 'data' of a type (line 781)
    data_13842 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 781, 29), _array_13841, 'data')
    # Processing the call keyword arguments (line 781)
    kwargs_13843 = {}
    # Getting the type of 'fd' (line 781)
    fd_13839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 781, 17), 'fd', False)
    # Obtaining the member 'readinto' of a type (line 781)
    readinto_13840 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 781, 17), fd_13839, 'readinto')
    # Calling readinto(args, kwargs) (line 781)
    readinto_call_result_13844 = invoke(stypy.reporting.localization.Localization(__file__, 781, 17), readinto_13840, *[data_13842], **kwargs_13843)
    
    # Assigning a type to the variable 'nbytesread' (line 781)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 781, 4), 'nbytesread', readinto_call_result_13844)
    
    
    # Getting the type of 'nbytesread' (line 782)
    nbytesread_13845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 782, 7), 'nbytesread')
    # Getting the type of 'nbytes' (line 782)
    nbytes_13846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 782, 21), 'nbytes')
    # Applying the binary operator '!=' (line 782)
    result_ne_13847 = python_operator(stypy.reporting.localization.Localization(__file__, 782, 7), '!=', nbytesread_13845, nbytes_13846)
    
    # Testing the type of an if condition (line 782)
    if_condition_13848 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 782, 4), result_ne_13847)
    # Assigning a type to the variable 'if_condition_13848' (line 782)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 782, 4), 'if_condition_13848', if_condition_13848)
    # SSA begins for if statement (line 782)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to IOError(...): (line 783)
    # Processing the call arguments (line 783)
    str_13850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 783, 22), 'str', "Didn't read as many bytes as expected")
    # Processing the call keyword arguments (line 783)
    kwargs_13851 = {}
    # Getting the type of 'IOError' (line 783)
    IOError_13849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 783, 14), 'IOError', False)
    # Calling IOError(args, kwargs) (line 783)
    IOError_call_result_13852 = invoke(stypy.reporting.localization.Localization(__file__, 783, 14), IOError_13849, *[str_13850], **kwargs_13851)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 783, 8), IOError_call_result_13852, 'raise parameter', BaseException)
    # SSA join for if statement (line 782)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'name' (line 784)
    name_13853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 784, 7), 'name')
    # Testing the type of an if condition (line 784)
    if_condition_13854 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 784, 4), name_13853)
    # Assigning a type to the variable 'if_condition_13854' (line 784)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 784, 4), 'if_condition_13854', if_condition_13854)
    # SSA begins for if statement (line 784)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to close(...): (line 785)
    # Processing the call keyword arguments (line 785)
    kwargs_13857 = {}
    # Getting the type of 'fd' (line 785)
    fd_13855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 785, 8), 'fd', False)
    # Obtaining the member 'close' of a type (line 785)
    close_13856 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 785, 8), fd_13855, 'close')
    # Calling close(args, kwargs) (line 785)
    close_call_result_13858 = invoke(stypy.reporting.localization.Localization(__file__, 785, 8), close_13856, *[], **kwargs_13857)
    
    # SSA join for if statement (line 784)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of '_array' (line 787)
    _array_13859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 787, 11), '_array')
    # Assigning a type to the variable 'stypy_return_type' (line 787)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 787, 4), 'stypy_return_type', _array_13859)
    
    # ################# End of 'fromfile(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'fromfile' in the type store
    # Getting the type of 'stypy_return_type' (line 721)
    stypy_return_type_13860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 721, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_13860)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'fromfile'
    return stypy_return_type_13860

# Assigning a type to the variable 'fromfile' (line 721)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 721, 0), 'fromfile', fromfile)

@norecursion
def array(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 789)
    None_13861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 789, 21), 'None')
    # Getting the type of 'None' (line 789)
    None_13862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 789, 33), 'None')
    int_13863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 789, 46), 'int')
    # Getting the type of 'None' (line 789)
    None_13864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 789, 57), 'None')
    # Getting the type of 'None' (line 789)
    None_13865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 789, 71), 'None')
    # Getting the type of 'None' (line 790)
    None_13866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 790, 16), 'None')
    # Getting the type of 'None' (line 790)
    None_13867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 790, 29), 'None')
    # Getting the type of 'False' (line 790)
    False_13868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 790, 43), 'False')
    # Getting the type of 'None' (line 790)
    None_13869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 790, 60), 'None')
    # Getting the type of 'True' (line 790)
    True_13870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 790, 71), 'True')
    defaults = [None_13861, None_13862, int_13863, None_13864, None_13865, None_13866, None_13867, False_13868, None_13869, True_13870]
    # Create a new context for function 'array'
    module_type_store = module_type_store.open_function_context('array', 789, 0, False)
    
    # Passed parameters checking function
    array.stypy_localization = localization
    array.stypy_type_of_self = None
    array.stypy_type_store = module_type_store
    array.stypy_function_name = 'array'
    array.stypy_param_names_list = ['obj', 'dtype', 'shape', 'offset', 'strides', 'formats', 'names', 'titles', 'aligned', 'byteorder', 'copy']
    array.stypy_varargs_param_name = None
    array.stypy_kwargs_param_name = None
    array.stypy_call_defaults = defaults
    array.stypy_call_varargs = varargs
    array.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'array', ['obj', 'dtype', 'shape', 'offset', 'strides', 'formats', 'names', 'titles', 'aligned', 'byteorder', 'copy'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'array', localization, ['obj', 'dtype', 'shape', 'offset', 'strides', 'formats', 'names', 'titles', 'aligned', 'byteorder', 'copy'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'array(...)' code ##################

    str_13871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 792, (-1)), 'str', 'Construct a record array from a wide-variety of objects.\n    ')
    
    
    # Evaluating a boolean operation
    
    # Evaluating a boolean operation
    
    # Call to isinstance(...): (line 794)
    # Processing the call arguments (line 794)
    # Getting the type of 'obj' (line 794)
    obj_13873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 794, 20), 'obj', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 794)
    tuple_13874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 794, 26), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 794)
    # Adding element type (line 794)
    
    # Call to type(...): (line 794)
    # Processing the call arguments (line 794)
    # Getting the type of 'None' (line 794)
    None_13876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 794, 31), 'None', False)
    # Processing the call keyword arguments (line 794)
    kwargs_13877 = {}
    # Getting the type of 'type' (line 794)
    type_13875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 794, 26), 'type', False)
    # Calling type(args, kwargs) (line 794)
    type_call_result_13878 = invoke(stypy.reporting.localization.Localization(__file__, 794, 26), type_13875, *[None_13876], **kwargs_13877)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 794, 26), tuple_13874, type_call_result_13878)
    # Adding element type (line 794)
    # Getting the type of 'str' (line 794)
    str_13879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 794, 38), 'str', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 794, 26), tuple_13874, str_13879)
    
    # Processing the call keyword arguments (line 794)
    kwargs_13880 = {}
    # Getting the type of 'isinstance' (line 794)
    isinstance_13872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 794, 9), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 794)
    isinstance_call_result_13881 = invoke(stypy.reporting.localization.Localization(__file__, 794, 9), isinstance_13872, *[obj_13873, tuple_13874], **kwargs_13880)
    
    
    # Call to isfileobj(...): (line 794)
    # Processing the call arguments (line 794)
    # Getting the type of 'obj' (line 794)
    obj_13883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 794, 57), 'obj', False)
    # Processing the call keyword arguments (line 794)
    kwargs_13884 = {}
    # Getting the type of 'isfileobj' (line 794)
    isfileobj_13882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 794, 47), 'isfileobj', False)
    # Calling isfileobj(args, kwargs) (line 794)
    isfileobj_call_result_13885 = invoke(stypy.reporting.localization.Localization(__file__, 794, 47), isfileobj_13882, *[obj_13883], **kwargs_13884)
    
    # Applying the binary operator 'or' (line 794)
    result_or_keyword_13886 = python_operator(stypy.reporting.localization.Localization(__file__, 794, 9), 'or', isinstance_call_result_13881, isfileobj_call_result_13885)
    
    
    # Getting the type of 'formats' (line 795)
    formats_13887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 795, 12), 'formats')
    # Getting the type of 'None' (line 795)
    None_13888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 795, 23), 'None')
    # Applying the binary operator 'is' (line 795)
    result_is__13889 = python_operator(stypy.reporting.localization.Localization(__file__, 795, 12), 'is', formats_13887, None_13888)
    
    # Applying the binary operator 'and' (line 794)
    result_and_keyword_13890 = python_operator(stypy.reporting.localization.Localization(__file__, 794, 8), 'and', result_or_keyword_13886, result_is__13889)
    
    # Getting the type of 'dtype' (line 795)
    dtype_13891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 795, 34), 'dtype')
    # Getting the type of 'None' (line 795)
    None_13892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 795, 43), 'None')
    # Applying the binary operator 'is' (line 795)
    result_is__13893 = python_operator(stypy.reporting.localization.Localization(__file__, 795, 34), 'is', dtype_13891, None_13892)
    
    # Applying the binary operator 'and' (line 794)
    result_and_keyword_13894 = python_operator(stypy.reporting.localization.Localization(__file__, 794, 8), 'and', result_and_keyword_13890, result_is__13893)
    
    # Testing the type of an if condition (line 794)
    if_condition_13895 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 794, 4), result_and_keyword_13894)
    # Assigning a type to the variable 'if_condition_13895' (line 794)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 794, 4), 'if_condition_13895', if_condition_13895)
    # SSA begins for if statement (line 794)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 796)
    # Processing the call arguments (line 796)
    str_13897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 796, 25), 'str', 'Must define formats (or dtype) if object is None, string, or an open file')
    # Processing the call keyword arguments (line 796)
    kwargs_13898 = {}
    # Getting the type of 'ValueError' (line 796)
    ValueError_13896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 796, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 796)
    ValueError_call_result_13899 = invoke(stypy.reporting.localization.Localization(__file__, 796, 14), ValueError_13896, *[str_13897], **kwargs_13898)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 796, 8), ValueError_call_result_13899, 'raise parameter', BaseException)
    # SSA join for if statement (line 794)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Dict to a Name (line 799):
    
    # Assigning a Dict to a Name (line 799):
    
    # Obtaining an instance of the builtin type 'dict' (line 799)
    dict_13900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 799, 11), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 799)
    
    # Assigning a type to the variable 'kwds' (line 799)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 799, 4), 'kwds', dict_13900)
    
    # Type idiom detected: calculating its left and rigth part (line 800)
    # Getting the type of 'dtype' (line 800)
    dtype_13901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 800, 4), 'dtype')
    # Getting the type of 'None' (line 800)
    None_13902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 800, 20), 'None')
    
    (may_be_13903, more_types_in_union_13904) = may_not_be_none(dtype_13901, None_13902)

    if may_be_13903:

        if more_types_in_union_13904:
            # Runtime conditional SSA (line 800)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 801):
        
        # Assigning a Call to a Name (line 801):
        
        # Call to dtype(...): (line 801)
        # Processing the call arguments (line 801)
        # Getting the type of 'dtype' (line 801)
        dtype_13907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 801, 25), 'dtype', False)
        # Processing the call keyword arguments (line 801)
        kwargs_13908 = {}
        # Getting the type of 'sb' (line 801)
        sb_13905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 801, 16), 'sb', False)
        # Obtaining the member 'dtype' of a type (line 801)
        dtype_13906 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 801, 16), sb_13905, 'dtype')
        # Calling dtype(args, kwargs) (line 801)
        dtype_call_result_13909 = invoke(stypy.reporting.localization.Localization(__file__, 801, 16), dtype_13906, *[dtype_13907], **kwargs_13908)
        
        # Assigning a type to the variable 'dtype' (line 801)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 801, 8), 'dtype', dtype_call_result_13909)

        if more_types_in_union_13904:
            # Runtime conditional SSA for else branch (line 800)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_13903) or more_types_in_union_13904):
        
        # Type idiom detected: calculating its left and rigth part (line 802)
        # Getting the type of 'formats' (line 802)
        formats_13910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 802, 9), 'formats')
        # Getting the type of 'None' (line 802)
        None_13911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 802, 24), 'None')
        
        (may_be_13912, more_types_in_union_13913) = may_not_be_none(formats_13910, None_13911)

        if may_be_13912:

            if more_types_in_union_13913:
                # Runtime conditional SSA (line 802)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Attribute to a Name (line 803):
            
            # Assigning a Attribute to a Name (line 803):
            
            # Call to format_parser(...): (line 803)
            # Processing the call arguments (line 803)
            # Getting the type of 'formats' (line 803)
            formats_13915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 803, 30), 'formats', False)
            # Getting the type of 'names' (line 803)
            names_13916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 803, 39), 'names', False)
            # Getting the type of 'titles' (line 803)
            titles_13917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 803, 46), 'titles', False)
            # Getting the type of 'aligned' (line 804)
            aligned_13918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 804, 30), 'aligned', False)
            # Getting the type of 'byteorder' (line 804)
            byteorder_13919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 804, 39), 'byteorder', False)
            # Processing the call keyword arguments (line 803)
            kwargs_13920 = {}
            # Getting the type of 'format_parser' (line 803)
            format_parser_13914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 803, 16), 'format_parser', False)
            # Calling format_parser(args, kwargs) (line 803)
            format_parser_call_result_13921 = invoke(stypy.reporting.localization.Localization(__file__, 803, 16), format_parser_13914, *[formats_13915, names_13916, titles_13917, aligned_13918, byteorder_13919], **kwargs_13920)
            
            # Obtaining the member '_descr' of a type (line 803)
            _descr_13922 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 803, 16), format_parser_call_result_13921, '_descr')
            # Assigning a type to the variable 'dtype' (line 803)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 803, 8), 'dtype', _descr_13922)

            if more_types_in_union_13913:
                # Runtime conditional SSA for else branch (line 802)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_13912) or more_types_in_union_13913):
            
            # Assigning a Dict to a Name (line 806):
            
            # Assigning a Dict to a Name (line 806):
            
            # Obtaining an instance of the builtin type 'dict' (line 806)
            dict_13923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 806, 15), 'dict')
            # Adding type elements to the builtin type 'dict' instance (line 806)
            # Adding element type (key, value) (line 806)
            str_13924 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 806, 16), 'str', 'formats')
            # Getting the type of 'formats' (line 806)
            formats_13925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 806, 27), 'formats')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 806, 15), dict_13923, (str_13924, formats_13925))
            # Adding element type (key, value) (line 806)
            str_13926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 807, 16), 'str', 'names')
            # Getting the type of 'names' (line 807)
            names_13927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 807, 25), 'names')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 806, 15), dict_13923, (str_13926, names_13927))
            # Adding element type (key, value) (line 806)
            str_13928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 808, 16), 'str', 'titles')
            # Getting the type of 'titles' (line 808)
            titles_13929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 808, 26), 'titles')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 806, 15), dict_13923, (str_13928, titles_13929))
            # Adding element type (key, value) (line 806)
            str_13930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 809, 16), 'str', 'aligned')
            # Getting the type of 'aligned' (line 809)
            aligned_13931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 809, 27), 'aligned')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 806, 15), dict_13923, (str_13930, aligned_13931))
            # Adding element type (key, value) (line 806)
            str_13932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 810, 16), 'str', 'byteorder')
            # Getting the type of 'byteorder' (line 810)
            byteorder_13933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 810, 29), 'byteorder')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 806, 15), dict_13923, (str_13932, byteorder_13933))
            
            # Assigning a type to the variable 'kwds' (line 806)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 806, 8), 'kwds', dict_13923)

            if (may_be_13912 and more_types_in_union_13913):
                # SSA join for if statement (line 802)
                module_type_store = module_type_store.join_ssa_context()


        

        if (may_be_13903 and more_types_in_union_13904):
            # SSA join for if statement (line 800)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 813)
    # Getting the type of 'obj' (line 813)
    obj_13934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 813, 7), 'obj')
    # Getting the type of 'None' (line 813)
    None_13935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 813, 14), 'None')
    
    (may_be_13936, more_types_in_union_13937) = may_be_none(obj_13934, None_13935)

    if may_be_13936:

        if more_types_in_union_13937:
            # Runtime conditional SSA (line 813)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Type idiom detected: calculating its left and rigth part (line 814)
        # Getting the type of 'shape' (line 814)
        shape_13938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 814, 11), 'shape')
        # Getting the type of 'None' (line 814)
        None_13939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 814, 20), 'None')
        
        (may_be_13940, more_types_in_union_13941) = may_be_none(shape_13938, None_13939)

        if may_be_13940:

            if more_types_in_union_13941:
                # Runtime conditional SSA (line 814)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to ValueError(...): (line 815)
            # Processing the call arguments (line 815)
            str_13943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 815, 29), 'str', 'Must define a shape if obj is None')
            # Processing the call keyword arguments (line 815)
            kwargs_13944 = {}
            # Getting the type of 'ValueError' (line 815)
            ValueError_13942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 815, 18), 'ValueError', False)
            # Calling ValueError(args, kwargs) (line 815)
            ValueError_call_result_13945 = invoke(stypy.reporting.localization.Localization(__file__, 815, 18), ValueError_13942, *[str_13943], **kwargs_13944)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 815, 12), ValueError_call_result_13945, 'raise parameter', BaseException)

            if more_types_in_union_13941:
                # SSA join for if statement (line 814)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Call to recarray(...): (line 816)
        # Processing the call arguments (line 816)
        # Getting the type of 'shape' (line 816)
        shape_13947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 816, 24), 'shape', False)
        # Getting the type of 'dtype' (line 816)
        dtype_13948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 816, 31), 'dtype', False)
        # Processing the call keyword arguments (line 816)
        # Getting the type of 'obj' (line 816)
        obj_13949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 816, 42), 'obj', False)
        keyword_13950 = obj_13949
        # Getting the type of 'offset' (line 816)
        offset_13951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 816, 54), 'offset', False)
        keyword_13952 = offset_13951
        # Getting the type of 'strides' (line 816)
        strides_13953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 816, 70), 'strides', False)
        keyword_13954 = strides_13953
        kwargs_13955 = {'strides': keyword_13954, 'buf': keyword_13950, 'offset': keyword_13952}
        # Getting the type of 'recarray' (line 816)
        recarray_13946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 816, 15), 'recarray', False)
        # Calling recarray(args, kwargs) (line 816)
        recarray_call_result_13956 = invoke(stypy.reporting.localization.Localization(__file__, 816, 15), recarray_13946, *[shape_13947, dtype_13948], **kwargs_13955)
        
        # Assigning a type to the variable 'stypy_return_type' (line 816)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 816, 8), 'stypy_return_type', recarray_call_result_13956)

        if more_types_in_union_13937:
            # Runtime conditional SSA for else branch (line 813)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_13936) or more_types_in_union_13937):
        
        # Type idiom detected: calculating its left and rigth part (line 818)
        # Getting the type of 'bytes' (line 818)
        bytes_13957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 818, 25), 'bytes')
        # Getting the type of 'obj' (line 818)
        obj_13958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 818, 20), 'obj')
        
        (may_be_13959, more_types_in_union_13960) = may_be_subtype(bytes_13957, obj_13958)

        if may_be_13959:

            if more_types_in_union_13960:
                # Runtime conditional SSA (line 818)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'obj' (line 818)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 818, 9), 'obj', remove_not_subtype_from_union(obj_13958, bytes))
            
            # Call to fromstring(...): (line 819)
            # Processing the call arguments (line 819)
            # Getting the type of 'obj' (line 819)
            obj_13962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 819, 26), 'obj', False)
            # Getting the type of 'dtype' (line 819)
            dtype_13963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 819, 31), 'dtype', False)
            # Processing the call keyword arguments (line 819)
            # Getting the type of 'shape' (line 819)
            shape_13964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 819, 44), 'shape', False)
            keyword_13965 = shape_13964
            # Getting the type of 'offset' (line 819)
            offset_13966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 819, 58), 'offset', False)
            keyword_13967 = offset_13966
            # Getting the type of 'kwds' (line 819)
            kwds_13968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 819, 68), 'kwds', False)
            kwargs_13969 = {'kwds_13968': kwds_13968, 'shape': keyword_13965, 'offset': keyword_13967}
            # Getting the type of 'fromstring' (line 819)
            fromstring_13961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 819, 15), 'fromstring', False)
            # Calling fromstring(args, kwargs) (line 819)
            fromstring_call_result_13970 = invoke(stypy.reporting.localization.Localization(__file__, 819, 15), fromstring_13961, *[obj_13962, dtype_13963], **kwargs_13969)
            
            # Assigning a type to the variable 'stypy_return_type' (line 819)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 819, 8), 'stypy_return_type', fromstring_call_result_13970)

            if more_types_in_union_13960:
                # Runtime conditional SSA for else branch (line 818)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_13959) or more_types_in_union_13960):
            # Assigning a type to the variable 'obj' (line 818)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 818, 9), 'obj', remove_subtype_from_union(obj_13958, bytes))
            
            
            # Call to isinstance(...): (line 821)
            # Processing the call arguments (line 821)
            # Getting the type of 'obj' (line 821)
            obj_13972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 821, 20), 'obj', False)
            
            # Obtaining an instance of the builtin type 'tuple' (line 821)
            tuple_13973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 821, 26), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 821)
            # Adding element type (line 821)
            # Getting the type of 'list' (line 821)
            list_13974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 821, 26), 'list', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 821, 26), tuple_13973, list_13974)
            # Adding element type (line 821)
            # Getting the type of 'tuple' (line 821)
            tuple_13975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 821, 32), 'tuple', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 821, 26), tuple_13973, tuple_13975)
            
            # Processing the call keyword arguments (line 821)
            kwargs_13976 = {}
            # Getting the type of 'isinstance' (line 821)
            isinstance_13971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 821, 9), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 821)
            isinstance_call_result_13977 = invoke(stypy.reporting.localization.Localization(__file__, 821, 9), isinstance_13971, *[obj_13972, tuple_13973], **kwargs_13976)
            
            # Testing the type of an if condition (line 821)
            if_condition_13978 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 821, 9), isinstance_call_result_13977)
            # Assigning a type to the variable 'if_condition_13978' (line 821)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 821, 9), 'if_condition_13978', if_condition_13978)
            # SSA begins for if statement (line 821)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            
            # Call to isinstance(...): (line 822)
            # Processing the call arguments (line 822)
            
            # Obtaining the type of the subscript
            int_13980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 822, 26), 'int')
            # Getting the type of 'obj' (line 822)
            obj_13981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 822, 22), 'obj', False)
            # Obtaining the member '__getitem__' of a type (line 822)
            getitem___13982 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 822, 22), obj_13981, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 822)
            subscript_call_result_13983 = invoke(stypy.reporting.localization.Localization(__file__, 822, 22), getitem___13982, int_13980)
            
            
            # Obtaining an instance of the builtin type 'tuple' (line 822)
            tuple_13984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 822, 31), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 822)
            # Adding element type (line 822)
            # Getting the type of 'tuple' (line 822)
            tuple_13985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 822, 31), 'tuple', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 822, 31), tuple_13984, tuple_13985)
            # Adding element type (line 822)
            # Getting the type of 'list' (line 822)
            list_13986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 822, 38), 'list', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 822, 31), tuple_13984, list_13986)
            
            # Processing the call keyword arguments (line 822)
            kwargs_13987 = {}
            # Getting the type of 'isinstance' (line 822)
            isinstance_13979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 822, 11), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 822)
            isinstance_call_result_13988 = invoke(stypy.reporting.localization.Localization(__file__, 822, 11), isinstance_13979, *[subscript_call_result_13983, tuple_13984], **kwargs_13987)
            
            # Testing the type of an if condition (line 822)
            if_condition_13989 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 822, 8), isinstance_call_result_13988)
            # Assigning a type to the variable 'if_condition_13989' (line 822)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 822, 8), 'if_condition_13989', if_condition_13989)
            # SSA begins for if statement (line 822)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to fromrecords(...): (line 823)
            # Processing the call arguments (line 823)
            # Getting the type of 'obj' (line 823)
            obj_13991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 823, 31), 'obj', False)
            # Processing the call keyword arguments (line 823)
            # Getting the type of 'dtype' (line 823)
            dtype_13992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 823, 42), 'dtype', False)
            keyword_13993 = dtype_13992
            # Getting the type of 'shape' (line 823)
            shape_13994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 823, 55), 'shape', False)
            keyword_13995 = shape_13994
            # Getting the type of 'kwds' (line 823)
            kwds_13996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 823, 64), 'kwds', False)
            kwargs_13997 = {'dtype': keyword_13993, 'shape': keyword_13995, 'kwds_13996': kwds_13996}
            # Getting the type of 'fromrecords' (line 823)
            fromrecords_13990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 823, 19), 'fromrecords', False)
            # Calling fromrecords(args, kwargs) (line 823)
            fromrecords_call_result_13998 = invoke(stypy.reporting.localization.Localization(__file__, 823, 19), fromrecords_13990, *[obj_13991], **kwargs_13997)
            
            # Assigning a type to the variable 'stypy_return_type' (line 823)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 823, 12), 'stypy_return_type', fromrecords_call_result_13998)
            # SSA branch for the else part of an if statement (line 822)
            module_type_store.open_ssa_branch('else')
            
            # Call to fromarrays(...): (line 825)
            # Processing the call arguments (line 825)
            # Getting the type of 'obj' (line 825)
            obj_14000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 825, 30), 'obj', False)
            # Processing the call keyword arguments (line 825)
            # Getting the type of 'dtype' (line 825)
            dtype_14001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 825, 41), 'dtype', False)
            keyword_14002 = dtype_14001
            # Getting the type of 'shape' (line 825)
            shape_14003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 825, 54), 'shape', False)
            keyword_14004 = shape_14003
            # Getting the type of 'kwds' (line 825)
            kwds_14005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 825, 63), 'kwds', False)
            kwargs_14006 = {'dtype': keyword_14002, 'shape': keyword_14004, 'kwds_14005': kwds_14005}
            # Getting the type of 'fromarrays' (line 825)
            fromarrays_13999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 825, 19), 'fromarrays', False)
            # Calling fromarrays(args, kwargs) (line 825)
            fromarrays_call_result_14007 = invoke(stypy.reporting.localization.Localization(__file__, 825, 19), fromarrays_13999, *[obj_14000], **kwargs_14006)
            
            # Assigning a type to the variable 'stypy_return_type' (line 825)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 825, 12), 'stypy_return_type', fromarrays_call_result_14007)
            # SSA join for if statement (line 822)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA branch for the else part of an if statement (line 821)
            module_type_store.open_ssa_branch('else')
            
            
            # Call to isinstance(...): (line 827)
            # Processing the call arguments (line 827)
            # Getting the type of 'obj' (line 827)
            obj_14009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 827, 20), 'obj', False)
            # Getting the type of 'recarray' (line 827)
            recarray_14010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 827, 25), 'recarray', False)
            # Processing the call keyword arguments (line 827)
            kwargs_14011 = {}
            # Getting the type of 'isinstance' (line 827)
            isinstance_14008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 827, 9), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 827)
            isinstance_call_result_14012 = invoke(stypy.reporting.localization.Localization(__file__, 827, 9), isinstance_14008, *[obj_14009, recarray_14010], **kwargs_14011)
            
            # Testing the type of an if condition (line 827)
            if_condition_14013 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 827, 9), isinstance_call_result_14012)
            # Assigning a type to the variable 'if_condition_14013' (line 827)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 827, 9), 'if_condition_14013', if_condition_14013)
            # SSA begins for if statement (line 827)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            
            # Evaluating a boolean operation
            
            # Getting the type of 'dtype' (line 828)
            dtype_14014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 828, 11), 'dtype')
            # Getting the type of 'None' (line 828)
            None_14015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 828, 24), 'None')
            # Applying the binary operator 'isnot' (line 828)
            result_is_not_14016 = python_operator(stypy.reporting.localization.Localization(__file__, 828, 11), 'isnot', dtype_14014, None_14015)
            
            
            # Getting the type of 'obj' (line 828)
            obj_14017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 828, 34), 'obj')
            # Obtaining the member 'dtype' of a type (line 828)
            dtype_14018 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 828, 34), obj_14017, 'dtype')
            # Getting the type of 'dtype' (line 828)
            dtype_14019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 828, 47), 'dtype')
            # Applying the binary operator '!=' (line 828)
            result_ne_14020 = python_operator(stypy.reporting.localization.Localization(__file__, 828, 34), '!=', dtype_14018, dtype_14019)
            
            # Applying the binary operator 'and' (line 828)
            result_and_keyword_14021 = python_operator(stypy.reporting.localization.Localization(__file__, 828, 11), 'and', result_is_not_14016, result_ne_14020)
            
            # Testing the type of an if condition (line 828)
            if_condition_14022 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 828, 8), result_and_keyword_14021)
            # Assigning a type to the variable 'if_condition_14022' (line 828)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 828, 8), 'if_condition_14022', if_condition_14022)
            # SSA begins for if statement (line 828)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 829):
            
            # Assigning a Call to a Name (line 829):
            
            # Call to view(...): (line 829)
            # Processing the call arguments (line 829)
            # Getting the type of 'dtype' (line 829)
            dtype_14025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 829, 27), 'dtype', False)
            # Processing the call keyword arguments (line 829)
            kwargs_14026 = {}
            # Getting the type of 'obj' (line 829)
            obj_14023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 829, 18), 'obj', False)
            # Obtaining the member 'view' of a type (line 829)
            view_14024 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 829, 18), obj_14023, 'view')
            # Calling view(args, kwargs) (line 829)
            view_call_result_14027 = invoke(stypy.reporting.localization.Localization(__file__, 829, 18), view_14024, *[dtype_14025], **kwargs_14026)
            
            # Assigning a type to the variable 'new' (line 829)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 829, 12), 'new', view_call_result_14027)
            # SSA branch for the else part of an if statement (line 828)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Name to a Name (line 831):
            
            # Assigning a Name to a Name (line 831):
            # Getting the type of 'obj' (line 831)
            obj_14028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 831, 18), 'obj')
            # Assigning a type to the variable 'new' (line 831)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 831, 12), 'new', obj_14028)
            # SSA join for if statement (line 828)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Getting the type of 'copy' (line 832)
            copy_14029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 832, 11), 'copy')
            # Testing the type of an if condition (line 832)
            if_condition_14030 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 832, 8), copy_14029)
            # Assigning a type to the variable 'if_condition_14030' (line 832)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 832, 8), 'if_condition_14030', if_condition_14030)
            # SSA begins for if statement (line 832)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 833):
            
            # Assigning a Call to a Name (line 833):
            
            # Call to copy(...): (line 833)
            # Processing the call keyword arguments (line 833)
            kwargs_14033 = {}
            # Getting the type of 'new' (line 833)
            new_14031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 833, 18), 'new', False)
            # Obtaining the member 'copy' of a type (line 833)
            copy_14032 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 833, 18), new_14031, 'copy')
            # Calling copy(args, kwargs) (line 833)
            copy_call_result_14034 = invoke(stypy.reporting.localization.Localization(__file__, 833, 18), copy_14032, *[], **kwargs_14033)
            
            # Assigning a type to the variable 'new' (line 833)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 833, 12), 'new', copy_call_result_14034)
            # SSA join for if statement (line 832)
            module_type_store = module_type_store.join_ssa_context()
            
            # Getting the type of 'new' (line 834)
            new_14035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 834, 15), 'new')
            # Assigning a type to the variable 'stypy_return_type' (line 834)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 834, 8), 'stypy_return_type', new_14035)
            # SSA branch for the else part of an if statement (line 827)
            module_type_store.open_ssa_branch('else')
            
            
            # Call to isfileobj(...): (line 836)
            # Processing the call arguments (line 836)
            # Getting the type of 'obj' (line 836)
            obj_14037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 836, 19), 'obj', False)
            # Processing the call keyword arguments (line 836)
            kwargs_14038 = {}
            # Getting the type of 'isfileobj' (line 836)
            isfileobj_14036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 836, 9), 'isfileobj', False)
            # Calling isfileobj(args, kwargs) (line 836)
            isfileobj_call_result_14039 = invoke(stypy.reporting.localization.Localization(__file__, 836, 9), isfileobj_14036, *[obj_14037], **kwargs_14038)
            
            # Testing the type of an if condition (line 836)
            if_condition_14040 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 836, 9), isfileobj_call_result_14039)
            # Assigning a type to the variable 'if_condition_14040' (line 836)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 836, 9), 'if_condition_14040', if_condition_14040)
            # SSA begins for if statement (line 836)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to fromfile(...): (line 837)
            # Processing the call arguments (line 837)
            # Getting the type of 'obj' (line 837)
            obj_14042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 837, 24), 'obj', False)
            # Processing the call keyword arguments (line 837)
            # Getting the type of 'dtype' (line 837)
            dtype_14043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 837, 35), 'dtype', False)
            keyword_14044 = dtype_14043
            # Getting the type of 'shape' (line 837)
            shape_14045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 837, 48), 'shape', False)
            keyword_14046 = shape_14045
            # Getting the type of 'offset' (line 837)
            offset_14047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 837, 62), 'offset', False)
            keyword_14048 = offset_14047
            kwargs_14049 = {'dtype': keyword_14044, 'shape': keyword_14046, 'offset': keyword_14048}
            # Getting the type of 'fromfile' (line 837)
            fromfile_14041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 837, 15), 'fromfile', False)
            # Calling fromfile(args, kwargs) (line 837)
            fromfile_call_result_14050 = invoke(stypy.reporting.localization.Localization(__file__, 837, 15), fromfile_14041, *[obj_14042], **kwargs_14049)
            
            # Assigning a type to the variable 'stypy_return_type' (line 837)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 837, 8), 'stypy_return_type', fromfile_call_result_14050)
            # SSA branch for the else part of an if statement (line 836)
            module_type_store.open_ssa_branch('else')
            
            
            # Call to isinstance(...): (line 839)
            # Processing the call arguments (line 839)
            # Getting the type of 'obj' (line 839)
            obj_14052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 839, 20), 'obj', False)
            # Getting the type of 'ndarray' (line 839)
            ndarray_14053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 839, 25), 'ndarray', False)
            # Processing the call keyword arguments (line 839)
            kwargs_14054 = {}
            # Getting the type of 'isinstance' (line 839)
            isinstance_14051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 839, 9), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 839)
            isinstance_call_result_14055 = invoke(stypy.reporting.localization.Localization(__file__, 839, 9), isinstance_14051, *[obj_14052, ndarray_14053], **kwargs_14054)
            
            # Testing the type of an if condition (line 839)
            if_condition_14056 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 839, 9), isinstance_call_result_14055)
            # Assigning a type to the variable 'if_condition_14056' (line 839)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 839, 9), 'if_condition_14056', if_condition_14056)
            # SSA begins for if statement (line 839)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            
            # Evaluating a boolean operation
            
            # Getting the type of 'dtype' (line 840)
            dtype_14057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 840, 11), 'dtype')
            # Getting the type of 'None' (line 840)
            None_14058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 840, 24), 'None')
            # Applying the binary operator 'isnot' (line 840)
            result_is_not_14059 = python_operator(stypy.reporting.localization.Localization(__file__, 840, 11), 'isnot', dtype_14057, None_14058)
            
            
            # Getting the type of 'obj' (line 840)
            obj_14060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 840, 34), 'obj')
            # Obtaining the member 'dtype' of a type (line 840)
            dtype_14061 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 840, 34), obj_14060, 'dtype')
            # Getting the type of 'dtype' (line 840)
            dtype_14062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 840, 47), 'dtype')
            # Applying the binary operator '!=' (line 840)
            result_ne_14063 = python_operator(stypy.reporting.localization.Localization(__file__, 840, 34), '!=', dtype_14061, dtype_14062)
            
            # Applying the binary operator 'and' (line 840)
            result_and_keyword_14064 = python_operator(stypy.reporting.localization.Localization(__file__, 840, 11), 'and', result_is_not_14059, result_ne_14063)
            
            # Testing the type of an if condition (line 840)
            if_condition_14065 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 840, 8), result_and_keyword_14064)
            # Assigning a type to the variable 'if_condition_14065' (line 840)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 840, 8), 'if_condition_14065', if_condition_14065)
            # SSA begins for if statement (line 840)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 841):
            
            # Assigning a Call to a Name (line 841):
            
            # Call to view(...): (line 841)
            # Processing the call arguments (line 841)
            # Getting the type of 'dtype' (line 841)
            dtype_14068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 841, 27), 'dtype', False)
            # Processing the call keyword arguments (line 841)
            kwargs_14069 = {}
            # Getting the type of 'obj' (line 841)
            obj_14066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 841, 18), 'obj', False)
            # Obtaining the member 'view' of a type (line 841)
            view_14067 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 841, 18), obj_14066, 'view')
            # Calling view(args, kwargs) (line 841)
            view_call_result_14070 = invoke(stypy.reporting.localization.Localization(__file__, 841, 18), view_14067, *[dtype_14068], **kwargs_14069)
            
            # Assigning a type to the variable 'new' (line 841)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 841, 12), 'new', view_call_result_14070)
            # SSA branch for the else part of an if statement (line 840)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Name to a Name (line 843):
            
            # Assigning a Name to a Name (line 843):
            # Getting the type of 'obj' (line 843)
            obj_14071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 843, 18), 'obj')
            # Assigning a type to the variable 'new' (line 843)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 843, 12), 'new', obj_14071)
            # SSA join for if statement (line 840)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Getting the type of 'copy' (line 844)
            copy_14072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 844, 11), 'copy')
            # Testing the type of an if condition (line 844)
            if_condition_14073 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 844, 8), copy_14072)
            # Assigning a type to the variable 'if_condition_14073' (line 844)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 844, 8), 'if_condition_14073', if_condition_14073)
            # SSA begins for if statement (line 844)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 845):
            
            # Assigning a Call to a Name (line 845):
            
            # Call to copy(...): (line 845)
            # Processing the call keyword arguments (line 845)
            kwargs_14076 = {}
            # Getting the type of 'new' (line 845)
            new_14074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 845, 18), 'new', False)
            # Obtaining the member 'copy' of a type (line 845)
            copy_14075 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 845, 18), new_14074, 'copy')
            # Calling copy(args, kwargs) (line 845)
            copy_call_result_14077 = invoke(stypy.reporting.localization.Localization(__file__, 845, 18), copy_14075, *[], **kwargs_14076)
            
            # Assigning a type to the variable 'new' (line 845)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 845, 12), 'new', copy_call_result_14077)
            # SSA join for if statement (line 844)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Call to view(...): (line 846)
            # Processing the call arguments (line 846)
            # Getting the type of 'recarray' (line 846)
            recarray_14080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 846, 24), 'recarray', False)
            # Processing the call keyword arguments (line 846)
            kwargs_14081 = {}
            # Getting the type of 'new' (line 846)
            new_14078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 846, 15), 'new', False)
            # Obtaining the member 'view' of a type (line 846)
            view_14079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 846, 15), new_14078, 'view')
            # Calling view(args, kwargs) (line 846)
            view_call_result_14082 = invoke(stypy.reporting.localization.Localization(__file__, 846, 15), view_14079, *[recarray_14080], **kwargs_14081)
            
            # Assigning a type to the variable 'stypy_return_type' (line 846)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 846, 8), 'stypy_return_type', view_call_result_14082)
            # SSA branch for the else part of an if statement (line 839)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Call to a Name (line 849):
            
            # Assigning a Call to a Name (line 849):
            
            # Call to getattr(...): (line 849)
            # Processing the call arguments (line 849)
            # Getting the type of 'obj' (line 849)
            obj_14084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 849, 28), 'obj', False)
            str_14085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 849, 33), 'str', '__array_interface__')
            # Getting the type of 'None' (line 849)
            None_14086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 849, 56), 'None', False)
            # Processing the call keyword arguments (line 849)
            kwargs_14087 = {}
            # Getting the type of 'getattr' (line 849)
            getattr_14083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 849, 20), 'getattr', False)
            # Calling getattr(args, kwargs) (line 849)
            getattr_call_result_14088 = invoke(stypy.reporting.localization.Localization(__file__, 849, 20), getattr_14083, *[obj_14084, str_14085, None_14086], **kwargs_14087)
            
            # Assigning a type to the variable 'interface' (line 849)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 849, 8), 'interface', getattr_call_result_14088)
            
            
            # Evaluating a boolean operation
            
            # Getting the type of 'interface' (line 850)
            interface_14089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 850, 11), 'interface')
            # Getting the type of 'None' (line 850)
            None_14090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 850, 24), 'None')
            # Applying the binary operator 'is' (line 850)
            result_is__14091 = python_operator(stypy.reporting.localization.Localization(__file__, 850, 11), 'is', interface_14089, None_14090)
            
            
            
            # Call to isinstance(...): (line 850)
            # Processing the call arguments (line 850)
            # Getting the type of 'interface' (line 850)
            interface_14093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 850, 47), 'interface', False)
            # Getting the type of 'dict' (line 850)
            dict_14094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 850, 58), 'dict', False)
            # Processing the call keyword arguments (line 850)
            kwargs_14095 = {}
            # Getting the type of 'isinstance' (line 850)
            isinstance_14092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 850, 36), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 850)
            isinstance_call_result_14096 = invoke(stypy.reporting.localization.Localization(__file__, 850, 36), isinstance_14092, *[interface_14093, dict_14094], **kwargs_14095)
            
            # Applying the 'not' unary operator (line 850)
            result_not__14097 = python_operator(stypy.reporting.localization.Localization(__file__, 850, 32), 'not', isinstance_call_result_14096)
            
            # Applying the binary operator 'or' (line 850)
            result_or_keyword_14098 = python_operator(stypy.reporting.localization.Localization(__file__, 850, 11), 'or', result_is__14091, result_not__14097)
            
            # Testing the type of an if condition (line 850)
            if_condition_14099 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 850, 8), result_or_keyword_14098)
            # Assigning a type to the variable 'if_condition_14099' (line 850)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 850, 8), 'if_condition_14099', if_condition_14099)
            # SSA begins for if statement (line 850)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to ValueError(...): (line 851)
            # Processing the call arguments (line 851)
            str_14101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 851, 29), 'str', 'Unknown input type')
            # Processing the call keyword arguments (line 851)
            kwargs_14102 = {}
            # Getting the type of 'ValueError' (line 851)
            ValueError_14100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 851, 18), 'ValueError', False)
            # Calling ValueError(args, kwargs) (line 851)
            ValueError_call_result_14103 = invoke(stypy.reporting.localization.Localization(__file__, 851, 18), ValueError_14100, *[str_14101], **kwargs_14102)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 851, 12), ValueError_call_result_14103, 'raise parameter', BaseException)
            # SSA join for if statement (line 850)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Assigning a Call to a Name (line 852):
            
            # Assigning a Call to a Name (line 852):
            
            # Call to array(...): (line 852)
            # Processing the call arguments (line 852)
            # Getting the type of 'obj' (line 852)
            obj_14106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 852, 23), 'obj', False)
            # Processing the call keyword arguments (line 852)
            kwargs_14107 = {}
            # Getting the type of 'sb' (line 852)
            sb_14104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 852, 14), 'sb', False)
            # Obtaining the member 'array' of a type (line 852)
            array_14105 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 852, 14), sb_14104, 'array')
            # Calling array(args, kwargs) (line 852)
            array_call_result_14108 = invoke(stypy.reporting.localization.Localization(__file__, 852, 14), array_14105, *[obj_14106], **kwargs_14107)
            
            # Assigning a type to the variable 'obj' (line 852)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 852, 8), 'obj', array_call_result_14108)
            
            
            # Evaluating a boolean operation
            
            # Getting the type of 'dtype' (line 853)
            dtype_14109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 853, 11), 'dtype')
            # Getting the type of 'None' (line 853)
            None_14110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 853, 24), 'None')
            # Applying the binary operator 'isnot' (line 853)
            result_is_not_14111 = python_operator(stypy.reporting.localization.Localization(__file__, 853, 11), 'isnot', dtype_14109, None_14110)
            
            
            # Getting the type of 'obj' (line 853)
            obj_14112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 853, 34), 'obj')
            # Obtaining the member 'dtype' of a type (line 853)
            dtype_14113 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 853, 34), obj_14112, 'dtype')
            # Getting the type of 'dtype' (line 853)
            dtype_14114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 853, 47), 'dtype')
            # Applying the binary operator '!=' (line 853)
            result_ne_14115 = python_operator(stypy.reporting.localization.Localization(__file__, 853, 34), '!=', dtype_14113, dtype_14114)
            
            # Applying the binary operator 'and' (line 853)
            result_and_keyword_14116 = python_operator(stypy.reporting.localization.Localization(__file__, 853, 11), 'and', result_is_not_14111, result_ne_14115)
            
            # Testing the type of an if condition (line 853)
            if_condition_14117 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 853, 8), result_and_keyword_14116)
            # Assigning a type to the variable 'if_condition_14117' (line 853)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 853, 8), 'if_condition_14117', if_condition_14117)
            # SSA begins for if statement (line 853)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 854):
            
            # Assigning a Call to a Name (line 854):
            
            # Call to view(...): (line 854)
            # Processing the call arguments (line 854)
            # Getting the type of 'dtype' (line 854)
            dtype_14120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 854, 27), 'dtype', False)
            # Processing the call keyword arguments (line 854)
            kwargs_14121 = {}
            # Getting the type of 'obj' (line 854)
            obj_14118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 854, 18), 'obj', False)
            # Obtaining the member 'view' of a type (line 854)
            view_14119 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 854, 18), obj_14118, 'view')
            # Calling view(args, kwargs) (line 854)
            view_call_result_14122 = invoke(stypy.reporting.localization.Localization(__file__, 854, 18), view_14119, *[dtype_14120], **kwargs_14121)
            
            # Assigning a type to the variable 'obj' (line 854)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 854, 12), 'obj', view_call_result_14122)
            # SSA join for if statement (line 853)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Call to view(...): (line 855)
            # Processing the call arguments (line 855)
            # Getting the type of 'recarray' (line 855)
            recarray_14125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 855, 24), 'recarray', False)
            # Processing the call keyword arguments (line 855)
            kwargs_14126 = {}
            # Getting the type of 'obj' (line 855)
            obj_14123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 855, 15), 'obj', False)
            # Obtaining the member 'view' of a type (line 855)
            view_14124 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 855, 15), obj_14123, 'view')
            # Calling view(args, kwargs) (line 855)
            view_call_result_14127 = invoke(stypy.reporting.localization.Localization(__file__, 855, 15), view_14124, *[recarray_14125], **kwargs_14126)
            
            # Assigning a type to the variable 'stypy_return_type' (line 855)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 855, 8), 'stypy_return_type', view_call_result_14127)
            # SSA join for if statement (line 839)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 836)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 827)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 821)
            module_type_store = module_type_store.join_ssa_context()
            

            if (may_be_13959 and more_types_in_union_13960):
                # SSA join for if statement (line 818)
                module_type_store = module_type_store.join_ssa_context()


        

        if (may_be_13936 and more_types_in_union_13937):
            # SSA join for if statement (line 813)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # ################# End of 'array(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'array' in the type store
    # Getting the type of 'stypy_return_type' (line 789)
    stypy_return_type_14128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 789, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_14128)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'array'
    return stypy_return_type_14128

# Assigning a type to the variable 'array' (line 789)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 789, 0), 'array', array)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
